# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Select particle-number-valid top-K determinants from a truncated MPS without shots.

In right-canonical form the right environment is the identity, so a contracted
prefix vector's squared norm is its total probability mass. This bounds each
completion from above. Electron-count feasibility prunes invalid prefixes exactly.
The beam also prunes by mass; ``last_topk_exact`` certifies the result only when
every discarded bound is at most the smallest retained probability.

This certificate concerns the truncated MPS, not the exact circuit state.
Usage and measurements: docs/mps.md.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from sqd_gqe.gqe.mps_sampler import MpsSampler, _free, _valid_fraction
from sqd_gqe.gqe.sampler import MIN_TOP_K_WEIGHT, floor_and_count, resolve_min_prob


class MpsTopKSampler(MpsSampler):
    """Reuse MpsSampler's circuit preparation and decoding, extracting amplitudes.

    Args:
        top_k: maximum number of weighted determinants to return.
        beam_width: retained prefixes per site, default ``4 * top_k``.
        min_prob: floor on |c|^2, as in ``ExactTopKSampler``: ``"auto"`` (the default) is
            ``1 / shots_count``, a number sets it, None turns it off.
        **kwargs: MpsSampler options, including ``max_bond``.

    Requires ``sqd.enlarge_method=None`` and ``sqd.recover_iterations=0`` to keep
    completion and recovery disabled (the solver may still form a product subspace).
    """

    def __init__(self, *args, top_k: int = 1000, beam_width: int | None = None,
                 min_prob: float | str | None = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = int(top_k)
        self.min_prob = resolve_min_prob(min_prob, self.shots_count)
        self.beam_width = int(beam_width) if beam_width else 4 * self.top_k

        #: Whether the returned set is certified as this MPS's true top-K.
        self.last_topk_exact = None
        #: Fraction of the whole MPS's probability captured, including invalid sectors.
        self.last_captured_weight = None

    # -- MPS extraction ------------------------------------------------------ #
    def _mps_tensors(self, state) -> list[np.ndarray]:
        """Site tensors as ``(left, 2, right)`` arrays, promoted for stable QR/products."""
        xp = _array_module()
        out = list(state.compute_output_state())
        shapes = [tuple(np.shape(t)) for t in out]
        tensors = [_as_site_tensor(xp.asarray(t), k, len(out), shapes).astype(xp.complex128)
                   for k, t in enumerate(out)]
        if tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1:
            raise _layout_error(
                f"boundary bonds are {tensors[0].shape[0]} and {tensors[-1].shape[2]}, "
                f"expected 1 and 1", shapes)
        return tensors

    @staticmethod
    def _right_canonicalize(tensors: list[np.ndarray]) -> list[np.ndarray]:
        """Normalise and make sites right-isometric so prefix norms bound probability."""
        xp = _array_module()
        # Rebind each tensor rather than copying the whole MPS.
        out = list(tensors)
        for k in range(len(out) - 1, 0, -1):
            dl, d, dr = out[k].shape
            # LQ via the QR of the transpose: m = L @ Q with Q having orthonormal rows.
            q, r = xp.linalg.qr(out[k].reshape(dl, d * dr).conj().T)
            out[k] = q.conj().T.reshape(q.shape[1], d, dr)
            out[k - 1] = xp.einsum("abc,cd->abd", out[k - 1], r.conj().T)
        # With sites 1.. right-isometric, site 0 carries the whole norm.
        norm = float(xp.sqrt(xp.einsum("abc,abc->", out[0], out[0].conj()).real))
        if norm > 0:
            out[0] = out[0] / norm
        return out

    def _site_spin(self, n_sites: int, layout) -> np.ndarray:
        """Map sites to alpha/beta through the same bit order and layout as decoding."""
        logical_at = None if layout is None else {p: i for i, p in enumerate(layout)}
        is_alpha = np.empty(n_sites, dtype=bool)
        for k in range(n_sites):
            pos = k if self._bit_order == "qubit0_left" else n_sites - 1 - k
            logical = pos if logical_at is None else logical_at[pos]
            is_alpha[k] = logical < self.norb
        return is_alpha

    # -- the search ---------------------------------------------------------- #
    def _search(self, tensors: list[np.ndarray], is_alpha: np.ndarray):
        """Return ``(bitstrings, probabilities, exact)`` in decreasing probability."""
        xp = _array_module()
        na, nb = self.nelec
        # Alpha / beta sites strictly after site k, for the "can still reach nelec" test.
        alpha_left = xp.asarray(np.cumsum(is_alpha[::-1])[::-1] - is_alpha)
        beta_left = xp.asarray(np.cumsum(~is_alpha[::-1])[::-1] - (~is_alpha))

        vec = xp.ones((1, 1), dtype=xp.complex128)   # prefix vectors, (beam, bond)
        bits = xp.zeros((1, 0), dtype=xp.uint8)
        count_a = xp.zeros(1, dtype=xp.int32)
        count_b = xp.zeros(1, dtype=xp.int32)
        worst_pruned = 0.0

        for k, a in enumerate(tensors):
            width = len(bits)
            vec = xp.concatenate([vec @ a[:, 0, :], vec @ a[:, 1, :]], axis=0)
            new_bit = xp.concatenate([xp.zeros(width, dtype=xp.uint8),
                                      xp.ones(width, dtype=xp.uint8)])
            bits = xp.concatenate([xp.concatenate([bits, bits]), new_bit[:, None]], axis=1)
            count_a = xp.concatenate([count_a, count_a + (1 if is_alpha[k] else 0)])
            count_b = xp.concatenate([count_b, count_b + (0 if is_alpha[k] else 1)])

            feasible = ((count_a <= na) & (count_b <= nb)
                        & (count_a + alpha_left[k] >= na)
                        & (count_b + beta_left[k] >= nb))
            vec, bits = vec[feasible], bits[feasible]
            count_a, count_b = count_a[feasible], count_b[feasible]
            if len(vec) == 0:
                # Keep the certificate honest if the constraints become unreachable.
                return [], np.zeros(0), worst_pruned == 0.0

            if len(vec) > self.beam_width:
                mass = xp.einsum("ij,ij->i", vec, vec.conj()).real
                part = xp.argpartition(mass, -self.beam_width)
                keep = part[-self.beam_width:]
                worst_pruned = max(worst_pruned,
                                   float(mass[part[:-self.beam_width]].max()))
                vec, bits = vec[keep], bits[keep]
                count_a, count_b = count_a[keep], count_b[keep]

        prob = xp.einsum("ij,ij->i", vec, vec.conj()).real
        best = xp.argsort(prob)[::-1][:min(self.top_k, len(prob))]
        prob = _to_host(prob[best])
        strings = ["".join("1" if x else "0" for x in row) for row in _to_host(bits[best])]

        # Short of top_k, a pruned branch could have supplied the missing entries, so the
        # usual bound does not settle it on its own.
        exact = worst_pruned == 0.0 if len(prob) < self.top_k else worst_pruned <= prob[-1]
        return strings, prob, bool(exact)

    # -- the sampler interface ----------------------------------------------- #
    def sample_state(self, seq: Sequence[int], seed: int | None = None) -> dict[tuple[int, int], int]:
        """Return amplitude-weighted pseudo-counts; ``seed`` is unused."""
        tqc, layout, t_transpile = self._prepare(seq)

        if self._exceeds_max_gates(tqc):
            self.last_topk_exact = False
            self.last_captured_weight = 0.0
            self.last_timings = {"transpile": t_transpile, "build": 0.0,
                                 "canonicalize": 0.0, "search": 0.0}
            return {}

        # compute_output_state performs the same contraction used by shot sampling.
        state = self._network_state(tqc)
        try:
            t = time.perf_counter()
            tensors = self._mps_tensors(state)
            t_build = time.perf_counter() - t
        finally:
            _free(state)

        t = time.perf_counter()
        tensors = self._right_canonicalize(tensors)
        t_canon = time.perf_counter() - t

        t = time.perf_counter()
        strings, prob, exact = self._search(tensors, self._site_spin(len(tensors), layout))
        t_search = time.perf_counter() - t
        # The canonicalised MPS has norm 1, so the cutoff is relative to the full state.
        keep = prob > MIN_TOP_K_WEIGHT
        strings, prob = [s for s, k in zip(strings, keep) if k], prob[keep]

        self.last_topk_exact = exact
        self.last_captured_weight = float(prob.sum()) if len(prob) else 0.0
        self.last_timings = {"transpile": t_transpile, "build": t_build,
                             "canonicalize": t_canon, "search": t_search}
        # The search can return valid bitstrings that all carry zero weight, e.g. when
        # the state has no amplitude in the right particle-number sector.
        total = self.last_captured_weight
        if not (np.isfinite(total) and total > 0.0):
            self.last_valid_fraction = 0.0
            self._log("no particle-number-valid bitstring with nonzero weight survived "
                      "the search")
            return {}

        keep, pseudo = floor_and_count(prob, self.shots_count, self.min_prob)
        strings = [s for s, k in zip(strings, keep) if k]
        counts = self._decode({s: int(c) for s, c in zip(strings, pseudo)}, layout)
        # Always 1.0 unless the site-to-spin mapping is wrong, which is why it is checked.
        self.last_valid_fraction = _valid_fraction(counts, self.nelec)
        self._log(f"{tqc.size()} gates -> top-{len(counts)} determinants, "
                  f"captured weight {self.last_captured_weight:.4f}, "
                  f"{'provably exact' if exact else 'beam-limited'} "
                  f"(transpile {t_transpile:.1f}s, build {t_build:.1f}s, "
                  f"canonicalize {t_canon:.1f}s, search {t_search:.1f}s, "
                  f"chi<={self.max_bond})")
        return counts


def _layout_error(what: str, shapes) -> RuntimeError:
    return RuntimeError(f"cuQuantum MPS output state: {what}. "
                        f"Site shapes as returned: {shapes}")


def _as_site_tensor(a, k: int, n: int, shapes):
    """Restore boundary dummy bonds to obtain ``(left, physical, right)``.

    Use site position: a boundary tensor with shape (2, 2) is otherwise ambiguous.
    """
    if a.ndim == 2:
        if k == 0:
            a = a.reshape((1,) + a.shape)
        elif k == n - 1:
            a = a.reshape(a.shape + (1,))
        else:
            raise _layout_error(
                f"interior site {k} is rank-2 with shape {a.shape}; only the two boundary "
                f"sites may drop a bond", shapes)
    if a.ndim != 3:
        raise _layout_error(f"site {k} has shape {a.shape}, expected rank 3", shapes)
    if a.shape[1] != 2:
        raise _layout_error(
            f"site {k} has shape {a.shape}, which puts no physical index in the middle",
            shapes)
    return a


def _to_host(a):
    return a.get() if hasattr(a, "get") else a


def _array_module():
    """Search on the GPU when available; use NumPy for CPU tests."""
    try:
        import cupy
        return cupy
    except ImportError:
        return np
