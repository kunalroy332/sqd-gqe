# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ffsim sampler for the GQE + SQD pipeline.

For each generated circuit (a sequence of operator-pool indices) it:

  1. prepares the Hartree-Fock state in the fixed-particle-number FCI space,
  2. applies the selected ffsim unitaries in order,
  3. samples computational-basis bitstrings, and
  4. returns a ``counts`` dict keyed by ``(alpha_occ, beta_occ)`` integer
     occupation bitmasks -- the backend-neutral form consumed by
     ``DeterminantSubspace.from_bitstring_counts``.

ffsim's sampled integers use the concatenation ``s_b s_a`` (beta in the high
``norb`` bits, alpha in the low ``norb`` bits; orbital 0 = LSB per sector), so
decoding is a mask/shift with no reordering. This is the *same* convention
qiskit-addon-sqd expects, so downstream SQD needs no re-ordering either.

The returned counts feed ``ConfigurationSet.from_sample_counts``.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

from sqd_gqe.gqe.operator_pool import FfsimBasedPool


def select_top_k(xp, w, k: int):
    """Indices of the ``k`` largest weights, breaking cutoff ties by lower index."""
    size = int(w.size)
    k = min(int(k), size)
    if k == size:
        return xp.arange(size)
    sel = xp.argpartition(w, -k)[-k:]
    threshold = w[sel].min()
    strict = xp.flatnonzero(w > threshold)          # at most k-1 of them
    need = k - int(strict.size)
    if need <= 0:
        return strict[:k]
    tied = xp.flatnonzero(w == threshold)           # ascending, so lowest index wins
    return xp.concatenate([strict, tied[:need]])


# Drop zero/numerical-noise weights instead of padding the SQD subspace.
MIN_TOP_K_WEIGHT = 1e-12


def pseudo_counts(prob, shots: int):
    """Weights rescaled to ``shots`` integer counts, at least 1 each -- what sampling
    would converge to with infinite statistics. Shared by the top-k samplers."""
    return np.maximum(1, np.rint(prob / prob.sum() * shots)).astype(np.int64)


def resolve_min_prob(min_prob, shots: int) -> float | None:
    """The top-k floor as a probability: ``"auto"`` is ``1 / shots`` (an expected count of
    one), a number is taken as is, and None turns the floor off."""
    if min_prob is None:
        return None
    if min_prob == "auto":
        return 1.0 / float(shots)
    value = float(min_prob)
    if not value >= 0.0:
        raise ValueError(f"min_prob must be 'auto', None or a probability >= 0, not {min_prob!r}")
    return value


def floor_and_count(prob, shots: int, min_prob: float | None):
    """Apply the top-k floor to probabilities of a normalised state.

    Returns ``(keep, counts)``: a mask over ``prob`` and the counts of the kept entries.
    A top-k selection would otherwise hand a determinant a count even when a run of
    ``shots`` would almost never draw it, and a trained policy learns to spread its state
    that thin: fine for the reward at 32q, a collapse at 46q (docs/mps.md). With a floor the
    counts are what ``shots`` would give, ``prob * shots``, not renormalised. If nothing
    clears the floor only the largest is kept: a state no run could sample earns a
    one-determinant subspace, which is the right verdict.
    """
    prob = np.asarray(prob)
    if min_prob is None:
        return np.ones(prob.shape, dtype=bool), pseudo_counts(prob, shots)
    keep = prob >= min_prob
    if not keep.any():
        keep = prob == prob.max()
    return keep, np.maximum(1, np.rint(prob[keep] * shots)).astype(np.int64)


class FfsimSampler:
    def __init__(
        self,
        operator_pool: FfsimBasedPool,
        shots_count: int,
        seed: int | None = 0,
        rayon_threads: int | None = None,
    ):
        self.pool = operator_pool
        self.shots_count = int(shots_count)
        self.norb = operator_pool.norb
        self.nelec = tuple(operator_pool.nelec)
        self._rng = np.random.default_rng(seed)
        if rayon_threads is not None:
            import os
            os.environ.setdefault("RAYON_NUM_THREADS", str(int(rayon_threads)))
        # Side-channel matching IbmHardwareSampler.last_depths -- ffsim circuits are
        # simulated directly on the statevector, never transpiled to a real device's
        # native gate set, so "depth" here is the pool-sequence length (len(seq)), the
        # only depth-like quantity that exists on this path. Populated by run().
        #
        # NOTE: every circuit in a batch has the same len(seq) (== cfg.ngates + 1 for
        # the BOS token) -- ffsim has no equivalent of a real device's variable
        # transpiled depth, so pipeline.py's max_depth_for_diag skip will never
        # actually fire here (all-or-nothing per batch, not per-circuit). It only
        # does useful work against IbmHardwareSampler's real, per-circuit-variable
        # transpiled depths.
        self.last_depths: list[int] = []

    # -- state preparation --------------------------------------------------- #
    def prepare_state(self, seq: Sequence[int]) -> np.ndarray:
        """Apply the operator sequence to the Hartree-Fock reference state."""
        import ffsim

        vec = ffsim.hartree_fock_state(self.norb, self.nelec)
        for idx in seq:
            vec = self.pool[int(idx)].apply(vec, self.norb, self.nelec)
        return vec

    # -- sampling ------------------------------------------------------------ #
    def _draw_seed(self) -> int:
        """One int64 seed per circuit, so runs are reproducible."""
        return int(self._rng.integers(0, 2**63 - 1))

    def sample_state(self, seq: Sequence[int], seed: int | None = None) -> dict[tuple[int, int], int]:
        import ffsim

        vec = self.prepare_state(seq)
        shot_seed = self._draw_seed() if seed is None else int(seed)
        samples = ffsim.sample_state_vector(
            vec,
            norb=self.norb,
            nelec=self.nelec,
            shots=self.shots_count,
            bitstring_type=ffsim.BitstringType.INT,
            seed=shot_seed,
        )
        return self._ints_to_counts(samples)

    def _ints_to_counts(self, samples) -> dict[tuple[int, int], int]:
        """Decode ffsim integer samples into ``{(alpha_occ, beta_occ): count}``."""
        mask = (1 << self.norb) - 1
        counts: Counter = Counter()
        for s in np.asarray(samples).tolist():
            s = int(s)
            alpha = s & mask          # low norb bits
            beta = s >> self.norb     # high norb bits
            counts[(alpha, beta)] += 1
        return dict(counts)

    @torch.no_grad()
    def run(self, state: dict) -> list[dict[tuple[int, int], int]]:
        """Sample every circuit in the batch.

        Args:
            state: dict with key ``"idx"`` -> tensor of shape (batch, seq_len)
                   of operator-pool indices (first column is the BOS/idx-0 token).

        Returns:
            list of counts dicts, one per circuit, in the same order.
        """
        idx_output = state["idx"]
        if torch.is_tensor(idx_output):
            idx_output = idx_output.detach().cpu().tolist()
        # len(row) counts the BOS/idx-0 token too, same convention run.py's own
        # n_gates_selected reporting uses elsewhere -- not real circuit depth (ffsim
        # never transpiles), just this sampler's only depth-like quantity. Every row
        # has the same length, so this never actually varies within a batch (see the
        # NOTE on self.last_depths above) -- no real 2q-gate count exists on this
        # path either (ffsim simulates the statevector directly, no gate-level
        # circuit object), so last_two_qubit_gates stays empty rather than faked.
        self.last_depths = [len(row) for row in idx_output]
        return [self.sample_state(row) for row in idx_output]


class ExactTopKSampler(FfsimSampler):
    """Select the largest nonzero exact-state amplitudes without drawing shots.

    ``min_prob`` drops determinants a run of ``shots_count`` would not draw: ``"auto"``
    (the default) is ``1 / shots_count``, a number sets the floor, None turns it off
    (see ``floor_and_count``). See docs/mps.md for usage and the shot-noise comparison.
    """

    def __init__(self, operator_pool, shots_count: int, top_k: int, seed: int | None = 0,
                 rayon_threads: int | None = None, min_prob: float | str | None = "auto"):
        super().__init__(operator_pool, shots_count, seed=seed, rayon_threads=rayon_threads)
        self.top_k = int(top_k)
        self.min_prob = resolve_min_prob(min_prob, self.shots_count)
        self._strings = None

    def _string_tables(self):
        if self._strings is None:
            import ffsim
            dim_a, dim_b = ffsim.dims(self.norb, self.nelec)
            self._strings = (
                np.asarray(ffsim.addresses_to_strings(np.arange(dim_a), self.norb, self.nelec[0]),
                           dtype=np.uint64),
                np.asarray(ffsim.addresses_to_strings(np.arange(dim_b), self.norb, self.nelec[1]),
                           dtype=np.uint64),
                dim_b,
            )
        return self._strings

    def sample_state(self, seq: Sequence[int], seed: int | None = None) -> dict[tuple[int, int], int]:
        vec = self.prepare_state(seq)
        a_str, b_str, dim_b = self._string_tables()
        w = np.abs(np.asarray(vec).reshape(-1)) ** 2
        idx = select_top_k(np, w, self.top_k)
        idx = idx[w[idx] > MIN_TOP_K_WEIGHT * w.sum()]
        # Largest weight first. Rounding can give two determinants the same count, and
        # the stable sort downstream then keeps this order, so it has to be by weight.
        idx = idx[np.argsort(-w[idx], kind="stable")]
        keep, pseudo = floor_and_count(w[idx] / w.sum(), self.shots_count, self.min_prob)
        idx = idx[keep]
        ia, ib = np.divmod(idx, dim_b)
        return {(int(a_str[i]), int(b_str[j])): int(c) for i, j, c in zip(ia, ib, pseudo)}
