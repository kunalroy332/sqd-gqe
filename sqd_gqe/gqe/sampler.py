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
    def sample_state(self, seq: Sequence[int]) -> dict[tuple[int, int], int]:
        import ffsim

        vec = self.prepare_state(seq)
        # Draw an int64-per-sample seed from our RNG so runs are reproducible.
        shot_seed = int(self._rng.integers(0, 2**63 - 1))
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
