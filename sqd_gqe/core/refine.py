# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Subspace refinement via a non-orthogonal generalized eigenproblem.

Implements the subspace-refinement map: given several CI
wavefunctions defined on overlapping determinant sets, embed them on the union
basis, solve ``H c = E S c`` in the non-orthogonal basis they span, form the
merged amplitudes ``A_x = sum_j c_j a_x^(j)`` for the lowest root, and keep the
top-``max_size`` determinants by ``|A_x|^2``. This is the primary accuracy
driver of the pipeline and is backend-independent.

The projected Hamiltonian ``H_ij = <psi_i|H|psi_j>`` has two implementations,
both PySCF-only (no external selected-CI library):

- ``_projected_hamiltonian_sparse`` (default): builds ``H`` directly on the
  ``N`` determinants in the merged union, via ``libfci.FCIpspace_h0tril``
  (off-diagonal, the same C routine ``direct_spin1.pspace`` uses for its
  Davidson preconditioner subspace) plus a restricted Slater-Condon diagonal
  formula (matching ``pyscf.fci.fci_slow.make_hdiag``'s reference convention,
  verified against it to machine precision). Cost scales with ``N`` alone --
  confirmed flat (~5ms) from full FCI dim 1e9 through 1.6e15 in a synthetic
  benchmark, since neither this method nor its diagonal ever touches a
  ``dim_a * dim_b``-sized array.
- ``_projected_hamiltonian_full`` (opt-in via ``use_sparse_hamiltonian=False``):
  the original method -- scatters each merged CI vector into a FULL FCI
  vector, calls ``direct_spin1.contract_2e`` (whole-space Hamiltonian-vector
  product), gathers back down to the ``N`` entries needed. Correct, but its
  cost scales with the full active-space FCI dimension
  (``C(norb,n_alpha)*C(norb,n_beta)``), not ``N`` -- this dominated per-
  iteration training cost at 28-30 qubits (measured on real cluster runs:
  refine cost overtook diagonalization cost there) and would be prohibitive
  at 40+ qubits. Kept only as a slower, independently-implemented
  cross-check -- verified numerically identical to the sparse path to
  1e-13..1e-32 precision on h2/n2/co2/fe_cn6/co_nh3_6/ferrocene before the
  sparse path became the default.
"""

from __future__ import annotations

import ctypes
from typing import Sequence

import numpy as np
from pyscf import ao2mo
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.direct_spin1 import libfci
from scipy.linalg import eigh as scipy_eigh

import warnings

from sqd_gqe.core.configuration import Configuration, ConfigurationSet, CIVector, fold_rhf_half_strings
from sqd_gqe.core.system import FCI_DIM_LIMIT


def union_strings(ci_vectors: Sequence[CIVector], return_inverse: bool = False):
    """Set union of the (alpha, beta) determinant strings of several CI vectors.

    Returns an ``(N, 2)`` uint64 array (and, optionally, the inverse index of
    the concatenated input into it, in input order).
    """
    str_lists = [np.asarray(v.strings, dtype=np.uint64).reshape(-1, 2) for v in ci_vectors]
    all_pairs = np.concatenate(str_lists, axis=0) if str_lists else np.zeros((0, 2), dtype=np.uint64)
    view = np.ascontiguousarray(all_pairs).view([("a", np.uint64), ("b", np.uint64)]).ravel()
    uniq, inv = np.unique(view, return_inverse=True)
    strings = uniq.view(np.uint64).reshape(-1, 2)
    return (strings, inv.ravel()) if return_inverse else strings


def truncate_by_weight(strings: np.ndarray, weights: np.ndarray, max_size: int) -> ConfigurationSet:
    """Keep the ``max_size`` determinants of largest weight, ordered by decreasing weight."""
    k = min(int(max_size), int(weights.size))
    if k <= 0:
        return ConfigurationSet([])
    idx = np.argpartition(weights, -k)[-k:]
    idx = idx[np.argsort(weights[idx])[::-1]]
    configs = [Configuration([int(strings[i, 0]), int(strings[i, 1])]) for i in idx]
    return ConfigurationSet(configs)


def input_weighted_half_strings(ci_vectors):
    """Fold every input's (alpha, beta) support into one half-string list, ranked by the
    summed input ``|c|^2`` marginals. Placeholder (uniform) inputs contribute a majority
    vote. Returns ``(uniq int64, weights)``; used to bound TP-UFD before its solve and as
    the compression fallback when SBD returns no carryover list.
    """
    parts, ws = [], []
    for v in ci_vectors:
        if np.asarray(v).size == 0:
            continue
        parts.append(np.asarray(v.strings, dtype=np.uint64).reshape(-1, 2))
        ws.append(np.abs(np.asarray(v, dtype=np.complex128)) ** 2)
    if not parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return fold_rhf_half_strings(np.concatenate(parts), np.concatenate(ws))


class NonOrthogonalSolver:
    """Solve ``H c = E S c`` on the span of given CI wavefunctions (PySCF FCI)."""

    def __init__(self, hamiltonian, nelec: tuple[int, int], norb: int,
                 use_sparse_hamiltonian: bool = True):
        self.hamiltonian = hamiltonian
        self.nelec = tuple(nelec)
        self.norb = int(norb)
        self.h1 = np.asarray(hamiltonian.one_body, dtype=np.float64, order="C")
        # chemist-order ERI, restored to the packed form both H-build methods expect
        self.eri = ao2mo.restore(1, np.asarray(hamiltonian.two_body, dtype=np.float64), self.norb)
        self.core_energy = float(hamiltonian.core_energy)
        self._na = self.nelec[0]
        self._nb = self.nelec[1]
        self._dim_a = cistring.num_strings(self.norb, self._na)
        self._dim_b = cistring.num_strings(self.norb, self._nb)
        # See module docstring for why sparse is the default and full is an
        # explicit, verified-equivalent fallback -- not two independently
        # "correct enough" options, one is a cross-check on the other.
        self.use_sparse_hamiltonian = bool(use_sparse_hamiltonian)
        self._diagj = np.einsum("iijj->ij", self.eri)
        self._diagk = np.einsum("ijji->ij", self.eri)

    def solve(self, ci_vectors: Sequence[CIVector], s_rcond: float = 1e-10):
        strings, C = self._embed_on_union(ci_vectors)
        S = C.conj().T @ C
        S = 0.5 * (S + S.T.conj())
        H = self._projected_hamiltonian(strings, C, S)
        H = 0.5 * (H + H.T.conj())
        energies, coeffs = self._solve_stable(H, S, s_rcond=s_rcond)
        merged = C @ coeffs[:, 0]
        return energies, strings, merged

    def _projected_hamiltonian(self, strings: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
        if self.use_sparse_hamiltonian:
            return self._projected_hamiltonian_sparse(strings, C, S)
        return self._projected_hamiltonian_full(strings, C, S)

    # -- embedding ----------------------------------------------------------- #
    def _embed_on_union(self, ci_vectors: Sequence[CIVector]):
        str_lists = [np.asarray(v.strings, dtype=np.uint64) for v in ci_vectors]
        lens = np.array([x.shape[0] for x in str_lists], dtype=np.int64)
        strings, inv = union_strings(ci_vectors, return_inverse=True)

        N, K = strings.shape[0], len(ci_vectors)
        C = np.zeros((N, K), dtype=np.complex128)
        offset = 0
        for j, (v, L) in enumerate(zip(ci_vectors, lens)):
            idx = inv[offset : offset + L]
            C[idx, j] = np.asarray(v, dtype=np.complex128)
            offset += L
        return strings, C

    # -- projected Hamiltonian: sparse, restricted to the N determinants ----- #
    def _restricted_hdiag(self, strings: np.ndarray) -> np.ndarray:
        """Diagonal Slater-Condon energy for each of the N determinants directly
        (no full-space array) -- matches pyscf.fci.fci_slow.make_hdiag's formula
        (verified against it to 1e-14 on real molecules; see module docstring),
        which direct_spin1.make_hdiag's compiled routine also implements but
        only via a full ``dim_a*dim_b``-sized array, exactly the cost this
        method exists to avoid."""
        N = strings.shape[0]
        out = np.empty(N, dtype=np.float64)
        for i in range(N):
            aocc = np.array([p for p in range(self.norb) if (int(strings[i, 0]) >> p) & 1], dtype=int)
            bocc = np.array([p for p in range(self.norb) if (int(strings[i, 1]) >> p) & 1], dtype=int)
            e1 = (self.h1[aocc, aocc].sum() if aocc.size else 0.0) + \
                 (self.h1[bocc, bocc].sum() if bocc.size else 0.0)
            e2 = 0.0
            if aocc.size:
                e2 += self._diagj[np.ix_(aocc, aocc)].sum() - self._diagk[np.ix_(aocc, aocc)].sum()
            if bocc.size:
                e2 += self._diagj[np.ix_(bocc, bocc)].sum() - self._diagk[np.ix_(bocc, bocc)].sum()
            if aocc.size and bocc.size:
                e2 += self._diagj[np.ix_(aocc, bocc)].sum() + self._diagj[np.ix_(bocc, aocc)].sum()
            out[i] = e1 + 0.5 * e2
        return out

    def _projected_hamiltonian_sparse(self, strings: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
        N = strings.shape[0]
        stra = strings[:, 0].astype(np.uint64)
        strb = strings[:, 1].astype(np.uint64)
        h1e_c = np.ascontiguousarray(self.h1)
        eri_c = np.ascontiguousarray(self.eri)
        # FCIpspace_h0tril: the same C routine direct_spin1.pspace() uses to build
        # its Davidson-preconditioner subspace Hamiltonian -- it already accepts
        # an explicit, arbitrary (alpha,beta) PAIR list (not a dense alpha x beta
        # grid, which is what ruled out pyscf.fci.selected_ci for this use), so it
        # applies directly to the union-of-CI-vectors determinant list here. Fills
        # only the lower triangle and leaves the diagonal at 0 (same as pspace()'s
        # own usage) -- the diagonal is filled separately below.
        h0 = np.zeros((N, N), dtype=np.float64)
        libfci.FCIpspace_h0tril(
            h0.ctypes.data_as(ctypes.c_void_p),
            h1e_c.ctypes.data_as(ctypes.c_void_p),
            eri_c.ctypes.data_as(ctypes.c_void_p),
            stra.ctypes.data_as(ctypes.c_void_p),
            strb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.norb), ctypes.c_int(N),
        )
        H_det = h0 + h0.T
        idx = np.arange(N)
        H_det[idx, idx] = self._restricted_hdiag(strings)
        H = C.conj().T @ H_det.astype(np.complex128) @ C
        H += self.core_energy * S
        return H

    # -- projected Hamiltonian via full-FCI contraction (opt-in fallback) ---- #
    def _full_fci_addresses(self, strings: np.ndarray) -> np.ndarray:
        """Map each (alpha, beta) determinant to its address in the full FCI vector."""
        addr_a = cistring.strs2addr(self.norb, self._na, strings[:, 0].astype(np.uint64))
        addr_b = cistring.strs2addr(self.norb, self._nb, strings[:, 1].astype(np.uint64))
        return addr_a * self._dim_b + addr_b

    def _projected_hamiltonian_full(self, strings: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
        addresses = self._full_fci_addresses(strings)
        N, K = C.shape
        # Scatter the K subspace vectors into full FCI vectors, apply H, gather back.
        HC = np.zeros((N, K), dtype=np.complex128)
        h2e = direct_spin1.absorb_h1e(self.h1, self.eri, self.norb, self.nelec, 0.5)
        full = np.zeros(self._dim_a * self._dim_b, dtype=np.float64)
        for j in range(K):
            cj = C[:, j]
            for part, dtype_cast in ((cj.real, 1.0), (cj.imag, 1.0j)):
                if not np.any(part):
                    continue
                full[:] = 0.0
                full[addresses] = part
                fvec = full.reshape(self._dim_a, self._dim_b)
                hc = direct_spin1.contract_2e(h2e, fvec, self.norb, self.nelec).reshape(-1)
                HC[:, j] += dtype_cast * hc[addresses]
        H = C.conj().T @ HC
        H += self.core_energy * S
        return H

    def diagonalize_subspace(self, strings: np.ndarray):
        """Exact diagonalization of H in an orthonormal determinant subspace.

        ``strings`` is an ``(N, 2)`` uint64 array of (alpha, beta) determinants.
        Returns ``(energy, coeffs)`` for the lowest root -- a PySCF-only parity
        check against the SQD diagonalizer.
        """
        strings = np.asarray(strings, dtype=np.uint64).reshape(-1, 2)
        N = strings.shape[0]
        C = np.eye(N, dtype=np.complex128)
        H = self._projected_hamiltonian(strings, C, np.eye(N))
        H = 0.5 * (H + H.T.conj())
        w, v = np.linalg.eigh(H)
        return float(w[0].real), v[:, 0]

    def _solve_stable(self, H, S, s_rcond: float):
        se, U = np.linalg.eigh(S)
        se = np.real(se)
        tol = float(s_rcond) * float(np.max(se))
        mask = se > tol
        if not np.any(mask):
            raise ValueError("Overlap matrix S is numerically singular.")
        Ured = U[:, mask]
        Sred = se[mask]
        H_red = Ured.conj().T @ H @ Ured
        w, v = scipy_eigh(H_red, np.diag(Sred), subset_by_index=(0, 0), check_finite=False)
        return w, Ured @ v


class SubspaceMerger:
    """Merge CI wavefunctions into a compact determinant set (refinement map)."""

    needs_full_fci = True   # _projected_hamiltonian contracts over the full FCI space

    def __init__(self, hamiltonian, nelec, norb, use_sparse_hamiltonian: bool = True):
        self.solver = NonOrthogonalSolver(hamiltonian, nelec, int(norb),
                                          use_sparse_hamiltonian=use_sparse_hamiltonian)
        self.last_info: dict = {}

    def merge(self, ci_vectors, max_size: int, s_rcond: float = 1e-10) -> ConfigurationSet:
        energies, strings, merged = self.solver.solve(ci_vectors, s_rcond=s_rcond)
        self.last_info = {"pre_dim": int(strings.shape[0]), "pre_energy": float(energies[0].real)}
        return truncate_by_weight(strings, np.abs(merged) ** 2, max_size)


# --------------------------------------------------------------------------- #
# Union Full Diagonalization (UFD) refinement -- proof-of-principle           #
# --------------------------------------------------------------------------- #
class DetUfdMerger:
    """Determinant UFD: diagonalize H exactly on the *union* of the input
    determinant sets (no GEVP), then keep the top-``max_size`` by ``|c_x|^2``.

    Uses ``NonOrthogonalSolver.diagonalize_subspace`` (dense eigh + full-FCI
    contraction), so this is a small-system reference implementation guarded by
    ``dense_limit``; a sparse determinant-space solver is out of scope here.
    """

    needs_full_fci = True   # diagonalize_subspace contracts over the full FCI space

    def __init__(self, hamiltonian, nelec, norb, dense_limit: int = 20_000):
        self.solver = NonOrthogonalSolver(hamiltonian, nelec, int(norb))
        self.dense_limit = int(dense_limit)
        self.last_info: dict = {}

    def merge(self, ci_vectors, max_size: int, **_) -> ConfigurationSet:
        vecs = [v for v in ci_vectors if np.asarray(v).size]
        if not vecs:
            return ConfigurationSet([])
        strings, C = self.solver._embed_on_union(vecs)
        pre_truncated = False
        if strings.shape[0] > self.dense_limit:
            # Dense solver: bound the union *before* solving by the summed input |c|^2
            # (the same quantity the post-solve truncation uses, approximated from the inputs).
            w_in = (np.abs(C) ** 2).sum(axis=1)
            keep = np.argsort(w_in)[::-1][: self.dense_limit]
            warnings.warn(f"det-UFD union {strings.shape[0]} > dense_limit={self.dense_limit}; "
                          "pre-truncating by input weights before the dense solve")
            strings = strings[np.sort(keep)]
            pre_truncated = True
        energy, coeffs = self.solver.diagonalize_subspace(strings)
        self.last_info = {"pre_dim": int(strings.shape[0]), "pre_energy": float(energy),
                          "pre_truncated": pre_truncated}
        return truncate_by_weight(strings, np.abs(coeffs) ** 2, max_size)


class TpUfdMerger:
    """Tensor-product UFD (RHF only): union the alpha and beta half-strings of
    all inputs into one list ``A``, solve H on the tensor closure ``A x A``, and
    compress to ``n_alpha`` half-strings by marginal weight (CPU, exact via
    ``solve_sci`` amplitudes) or by SBD's carryover list (GPU, energy-only).

    ``merge`` returns the full ``keep x keep`` product so the pipeline's
    ``diagonalize()`` (which re-forms ``unique(alpha) x unique(beta)``) solves
    exactly the compressed tensor basis. Budget: ``n_alpha = isqrt(max_size)``
    unless given explicitly.
    """

    needs_full_fci = False  # cost is |A|^2 (bounded by closure_limit), never the FCI dimension

    def __init__(self, hamiltonian, nelec, norb, spin_sq: float | None = 0.0,
                 sbd=None, n_alpha: int | None = None, closure_limit: int = FCI_DIM_LIMIT):
        self.nelec = tuple(nelec)
        self.closure_limit = int(closure_limit)
        if self.nelec[0] != self.nelec[1]:
            raise ValueError(f"tp_ufd refinement is RHF-only (nelec[0] == nelec[1]); got nelec={self.nelec}")
        self.norb = int(norb)
        self.spin_sq = spin_sq
        self.sbd = sbd
        self.n_alpha = n_alpha
        self._hcore = np.asarray(hamiltonian.one_body, dtype=np.float64, order="C")
        self._eri = np.asarray(hamiltonian.two_body, dtype=np.float64, order="C")
        self._core_energy = float(hamiltonian.core_energy)
        self.last_info: dict = {}

    def merge(self, ci_vectors, max_size: int, **_) -> ConfigurationSet:
        vecs = [v for v in ci_vectors if np.asarray(v).size]
        strings = union_strings(vecs)
        if strings.shape[0] == 0:
            return ConfigurationSet([])
        # Half-string union, ranked by the inputs' own |c|^2 marginals.
        A, w_in = input_weighted_half_strings(vecs)
        pre_truncated = False
        if A.size ** 2 > self.closure_limit:
            # Same role as FCI_DIM_LIMIT for gevp: never launch a solve we cannot finish.
            # Bound |A| before solving using the input marginals (cumulative global refinement
            # is where the union can grow without bound).
            n_keep = max(int(np.sqrt(self.closure_limit)), 1)
            warnings.warn(f"tp_ufd closure {A.size}^2 > closure_limit={self.closure_limit}; "
                          f"pre-truncating the half-string union to {n_keep} by input weights")
            order = np.argsort(w_in)[::-1][:n_keep]
            A, w_in = A[order], w_in[order]
            resort = np.argsort(A)
            A, w_in = A[resort], w_in[resort]
            pre_truncated = True
        n_a = self.n_alpha or max(int(np.sqrt(int(max_size))), 1)
        n_a = min(int(n_a), int(A.size))
        compression = "marginal"

        if self.sbd is None:
            from qiskit_addon_sqd.fermion import solve_sci

            result = solve_sci((A, A), self._hcore, self._eri, self.norb, self.nelec, spin_sq=self.spin_sq)
            e_pre = float(result.energy) + self._core_energy
            amps = np.asarray(result.sci_state.amplitudes).reshape(A.size, A.size)
            w = np.abs(amps) ** 2
            p = w.sum(axis=1) + w.sum(axis=0)          # alpha + beta marginals (RHF)
            keep = A[np.argpartition(p, -n_a)[-n_a:]]
        else:
            e_pre, _occ_a, carry = self.sbd._solve_once(A, solver_overrides={"carryover_ratio": n_a / A.size})
            if carry.size:
                keep, compression = carry.astype(np.int64), "carryover"
            else:
                # No carryover list from the binary: keep the n_a half-strings the inputs
                # weight most, rather than silently retaining the whole union.
                warnings.warn("tp_ufd: SBD returned no carryover list; falling back to input-weight compression")
                keep, compression = A[np.argpartition(w_in, -n_a)[-n_a:]], "input_weights_fallback"

        hartree_fock = (1 << self.nelec[0]) - 1
        keep = np.unique(np.concatenate([[hartree_fock], keep]))   # HF forced, as in recover_and_diagonalize
        # rho_tp compares the tensor closure with the union of the inputs' own determinant
        # supports. With SBD placeholder inputs the support is only the diagonal (a, a) pairs,
        # so the ratio would just be |A| and means nothing -- report None in that case.
        diagonal_only = bool(np.all(strings[:, 0] == strings[:, 1]))
        self.last_info = {
            "pre_dim": int(A.size) ** 2, "pre_energy": e_pre, "n_alpha_union": int(A.size),
            "n_alpha": int(keep.size), "compression": compression, "pre_truncated": pre_truncated,
            "rho_tp": None if diagonal_only else int(A.size) ** 2 / int(strings.shape[0]),
        }
        # Return the retained tensor basis as its |keep| diagonal pairs (a, a), not the |keep|^2
        # product: both diagonalizers re-form unique(alpha) x unique(beta) from the set, so the
        # closure is identical, while materializing the product cost 38 s of Python object
        # construction for a 5696-string union (3.2e7 pairs) against a 6.5 s GPU solve.
        # Consequence: ConfigurationSet.size is the half-string count for tp_ufd; the closure
        # dimension is last_info["retained_closure"].
        self.last_info["retained_closure"] = int(keep.size) ** 2
        return ConfigurationSet([Configuration([int(a), int(a)]) for a in keep])


def build_merger(method: str, hamiltonian, nelec, norb, *, spin_sq=0.0, sbd=None,
                 diag_backend: str = "sqd", n_alpha: int | None = None, dense_limit: int = 20_000,
                 closure_limit: int = FCI_DIM_LIMIT, use_sparse_hamiltonian: bool = True):
    """Refinement-map factory: ``"gevp"`` (default) | ``"ufd"`` | ``"tp_ufd"``."""
    if method == "gevp":
        if diag_backend == "sbd":
            raise ValueError('refine_method="gevp" needs CI amplitudes; diag_backend="sbd" is energy-only -- use "tp_ufd"')
        return SubspaceMerger(hamiltonian, nelec, norb, use_sparse_hamiltonian=use_sparse_hamiltonian)
    if method == "ufd":
        if diag_backend == "sbd":
            raise ValueError('refine_method="ufd" needs CI amplitudes; diag_backend="sbd" is energy-only -- use "tp_ufd"')
        return DetUfdMerger(hamiltonian, nelec, norb, dense_limit=dense_limit)
    if method == "tp_ufd":
        if diag_backend == "fci":
            # TpUfdMerger returns the retained basis as diagonal (a, a) pairs and relies on the
            # diagonalizer re-forming unique(alpha) x unique(beta); the fci parity path solves
            # exactly the rows it is given, so it would silently solve only the diagonal.
            raise ValueError('refine_method="tp_ufd" is not supported with diag_backend="fci" (use "sqd" or "sbd")')
        return TpUfdMerger(hamiltonian, nelec, norb, spin_sq=spin_sq,
                           sbd=sbd if diag_backend == "sbd" else None, n_alpha=n_alpha,
                           closure_limit=closure_limit)
    raise ValueError(f"Unknown refine_method: {method!r} (expected gevp | ufd | tp_ufd)")
