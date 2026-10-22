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

"""GPU diagonalizer backend: shells out to SBD's native ``diag-gpu`` binary (RHF only).

SBD ("Sparse/Selected Block Diagonalization") is a proven GPU+MPI selected-CI diagonalizer that
lives in a sibling repo (``qcsc-prefect/algorithms/sbd``). This wrapper depends on NEITHER
Prefect nor any qcsc-prefect Python package -- it reuses only the native binary itself plus the
plain, dependency-free file-format helpers in ``sbd_io.py`` (ported from that repo's
``sbd/solver_job.py``, which has zero Prefect coupling in its own file-I/O functions).

Scope of this first cut (see docs/2026-08-13's SBD integration notes for the full picture):
  * RHF only. No UHF wiring (the native ``diag-gpu_uhf-mpi-*`` binaries exist but are out of
    scope here -- a UHF ``ActiveSpaceHamiltonian`` variant doesn't exist anywhere in this repo).
  * Single GPU only (``adet_comm_size``/``bdet_comm_size`` fixed at 1 -- no multi-node/multi-GPU
    MPI grid).
  * ENERGY-ONLY. The native binary's output (davidson_energy.txt, occ_a.txt, carryover.bin) does
    not include per-determinant CI amplitudes -- there is no way to reconstruct
    qiskit-addon-sqd's real CIVector from it. diagonalize() returns a PLACEHOLDER CIVector (a
    single unit-amplitude entry) so the return type satisfies pipeline.py's contract, but this is
    NOT physical. Safe for direct energy comparison against "sqd"/"fci" on one diagonalize()
    call; NOT safe for SubspaceMerger's multi-call amplitude merging (confirmed by reading
    SubspaceMerger's code: it needs each circuit's per-determinant amplitude vector for a
    non-orthogonal GEVP solve -- RDMs, the other physical quantity ``diag-gpu --rdm=1`` can
    emit, are a marginal/aggregate quantity and cannot substitute; there is no way to adapt the
    merge step to this binary's output without a different refinement algorithm entirely) -- do
    not select diag_backend="sbd" for run_molecule() calls that rely on the local/global refine
    steps. ``recover_and_diagonalize`` (below) does NOT go through SubspaceMerger, so it is not
    affected by this limitation. refine_method="tp_ufd" (TpUfdMerger in core/refine.py) IS
    supported: it only needs each circuit's alpha half-string list (the placeholder's support)
    and compresses via this binary's carryover list instead of amplitudes.
  * The launcher MUST be mpirun, not srun (see sbd_launcher default in configs_sqd/default.yaml)
    -- confirmed empirically (2026-08-15): srun hangs this MPI-linked binary indefinitely (0% GPU
    utilization sustained for 20+ real minutes on a trivial 322-determinant subspace) because
    ROQUO's srun cannot bootstrap the HPC-X OpenMPI 5 diag-gpu links against (same failure this
    sibling repo's own README warns about for the multi-GPU binary). mpirun -np 1 completes the
    identical call in 2.3s. Requires `module load hpcx/2.50` for mpirun to be on PATH -- it is
    NOT bundled with the nvhpc module despite that README's phrasing.

``recover_and_diagonalize`` (added 2026-08-15) replicates the real multi-iteration recovery loop
found in the sibling repo's ``sbd/sqd.py`` (``walker_sqd``, RHF/close-shell branch), stripped of
all Prefect coupling: each step calls ``qiskit_addon_sqd.configuration_recovery.recover_configurations``
(CPU, cheap -- picks which determinants to try next from the average orbital occupancies) then
diagonalizes the resulting subspace with ``diag-gpu`` (GPU, the actual expensive step) instead of
qiskit-addon-sqd's own CPU FCI solver. The original CPU-only recovery loop
(``sqd_gqe/sqd/backend_sqd.py``'s ``SqdDiagonalizer.recover_and_diagonalize``) took over an hour
per iteration at 46 qubits and never completed 15 iterations in a 3-hour job -- this GPU path is
what the sibling repo's own recovery runs actually use (389s for a comparable-scale single step,
per a real recorded run).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pyscf.tools import fcidump

from sqd_gqe.core.configuration import ConfigurationSet, CIVector, make_ci_vector, fold_rhf_half_strings
from sqd_gqe.sqd import sbd_io

DEFAULT_SOLVER_KWARGS = {
    "task_comm_size": 1,
    "adet_comm_size": 1,
    "bdet_comm_size": 1,
    "block": 20,
    "iteration": 5,
    "tolerance": 0.01,
    "carryover_ratio": 0.5,
    "carryover_type": 1,
    "rdm": 0,
}


def _popcount_int64(x: np.ndarray) -> np.ndarray:
    """Per-element population count for a non-negative int64 array (numpy has no
    builtin popcount before 2.0, and this module supports older numpy)."""
    v = np.asarray(x, dtype=np.int64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(v).astype(np.int64)
    out = np.zeros(v.shape, dtype=np.int64)
    t = v.copy()
    while np.any(t):
        out += (t & 1).astype(np.int64)
        t >>= 1
    return out


class SbdSolverError(RuntimeError):
    """Raised when the native diag-gpu binary exits nonzero."""


class SbdGpuDiagonalizer:
    def __init__(self, hamiltonian, nelec: tuple[int, int], norb: int,
                 binary_path: str, launcher: list[str] | None = None,
                 solver_kwargs: dict | None = None, work_dir: str | None = None,
                 timeout_s: float | None = 600.0, max_dim: int | None = None):
        self.nelec = tuple(nelec)
        self.norb = int(norb)
        self.binary_path = binary_path
        self.launcher = list(launcher) if launcher else []
        self.solver_kwargs = {**DEFAULT_SOLVER_KWARGS, **(solver_kwargs or {})}
        self.work_dir = work_dir
        self.timeout_s = timeout_s
        # Net-CI-dimension ceiling for the one-shot diagonalize() path (below). expand()'s
        # max_size only caps the number of (alpha,beta) PAIRS kept -- folding those pairs into
        # unique alpha half-strings (RHF) can land anywhere from that count up to its square,
        # depending on real measured overlap, so max_size alone does not bound net dim. This
        # caused a real CUDA OOM (job 160318: 20,245 unique alphas from a max_size=20,000 pair
        # cap -> ~410M net dim). max_dim, when set, is enforced as sqrt(max_dim) on the folded
        # alpha count itself -- a hard ceiling independent of how the fold turns out.
        self.max_dim = int(max_dim) if max_dim is not None else None
        self._hcore = np.asarray(hamiltonian.one_body, dtype=np.float64, order="C")
        self._eri = np.asarray(hamiltonian.two_body, dtype=np.float64, order="C")
        self._core_energy = float(hamiltonian.core_energy)

    def diagonalize(self, config_set: ConfigurationSet) -> tuple[float, CIVector]:
        if config_set.size == 0:
            return float("inf"), make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64))

        configs = np.asarray(config_set.configs, dtype=np.uint64).reshape(-1, 2)
        # RHF: alpha and beta half-strings are interchangeable -- fold both columns into the
        # alpha list (same as recover_and_diagonalize) so a spin-completion truncated by
        # max_size cannot drop half-strings that only appear on the beta side.
        strs_a, _ = fold_rhf_half_strings(configs)

        # expand()'s max_size only caps the (alpha,beta) PAIR count going in -- folding those
        # pairs into unique alphas can land anywhere from that count up to its square, so
        # strs_a.size is NOT bounded by max_size on its own (job 160318: 20,245 unique alphas
        # from a 20,000 pair cap -> ~410M net dim, CUDA OOM). Enforce a hard net-dim ceiling
        # here: if the fold overshot sqrt(max_dim) alphas, truncate to the most-measured ones
        # -- configs is already ordered by descending sample count (ConfigurationSet.
        # from_sample_counts / expand() both preserve that order), so walking configs in order
        # and keeping the first sqrt(max_dim) distinct alpha values encountered prioritizes
        # the same determinants recover_and_diagonalize's weighted subsample would favor,
        # rather than np.unique's arbitrary value-sorted truncation.
        if self.max_dim is not None:
            alpha_cap = max(1, int(round(self.max_dim ** 0.5)))
            if strs_a.size > alpha_cap:
                seen: set[int] = set()
                ordered_unique: list[int] = []
                for a, b in configs:
                    for half in (int(a), int(b)):
                        if half not in seen:
                            seen.add(half)
                            ordered_unique.append(half)
                            if len(ordered_unique) >= alpha_cap:
                                break
                    if len(ordered_unique) >= alpha_cap:
                        break
                print(f"[sbd-diagonalize] one-shot: fold produced {strs_a.size} unique alpha "
                      f"determinants (net dim {strs_a.size ** 2}), exceeding the max-dim={self.max_dim} "
                      f"ceiling (alpha cap {alpha_cap}) -- truncating to the {len(ordered_unique)} "
                      "most-measured alphas", file=sys.stderr, flush=True)
                strs_a = np.asarray(sorted(ordered_unique), dtype=np.int64)

        print(f"[sbd-diagonalize] one-shot: {strs_a.size} alpha determinants "
              f"({strs_a.size ** 2} net dim) on GPU...", file=sys.stderr, flush=True)
        energy, _occ_a, _carryover_a = self._solve_once(strs_a)

        # PLACEHOLDER CIVector -- see module docstring. NOT a real amplitude vector: uniform
        # weights over the diagonal (a, a) pairs, so only the *support* (the alpha list this
        # solve used) is meaningful. That is exactly what TpUfdMerger (refine_method="tp_ufd")
        # consumes; gevp/ufd refinement would merge these fake amplitudes and stay unsupported.
        n = strs_a.size
        pairs = np.stack([strs_a, strs_a], axis=1).astype(np.uint64)
        placeholder = make_ci_vector(np.full(n, 1.0 / np.sqrt(n)), pairs)
        return energy, placeholder

    # -- low-level single solve, exposing occupancies + carryover ------------- #
    def _solve_once(self, strs_a: np.ndarray,
                    solver_overrides: dict | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        """Run one diag-gpu call on an explicit RHF alpha-determinant list.

        ``solver_overrides`` are merged over ``self.solver_kwargs`` for this call only
        (e.g. ``{"carryover_ratio": r}`` so TpUfdMerger can set the half-string budget).

        Returns ``(energy, orbital_occupancies, carryover_determinants)`` --
        ``orbital_occupancies`` is a length-``norb`` float array (``occ_a.txt``,
        RHF so beta occupancy is identical by symmetry) and
        ``carryover_determinants`` is an int64 array of alpha determinant
        integers read back from ``carryover.bin`` (empty if the file is absent
        or ``carryover_ratio=0``).
        """
        persistent = self.work_dir is not None
        work_dir = Path(self.work_dir or tempfile.mkdtemp(prefix="sbd_gpu_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._solve_in(work_dir, strs_a, solver_overrides, keep_stdout=persistent)
        finally:
            if not persistent:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _solve_in(self, work_dir: Path, strs_a: np.ndarray, solver_overrides: dict | None,
                  keep_stdout: bool) -> tuple[float, np.ndarray, np.ndarray]:

        # diag-gpu reads a fixed filename from its cwd -- no CLI flag for the FCIDUMP path.
        fcidump.from_integrals(
            str(work_dir / "fcidump.txt"), self._hcore, self._eri, self.norb, self.nelec,
        )
        sbd_io.write_dets_bin(work_dir / "AlphaDets.bin", strs_a, self.norb)

        solver_kwargs = {**self.solver_kwargs, **(solver_overrides or {})}
        args = sbd_io.build_solver_args(solver_kwargs)
        cmd = self.launcher + [self.binary_path] + args
        # Retry on a subprocess timeout: confirmed on real 38q multi-GPU runs that
        # this is an intermittent single-call hang (178/179 calls at the identical
        # 1000-det/1e6-net-dim config completed in ~12s; one call hung the full
        # 1800s timeout with no error, matching this session's earlier NFS-stall/
        # UCX-flakiness findings on this same shared cluster) -- not a systematic
        # scale issue, so retrying the same call is the right response rather than
        # raising the timeout further.
        max_attempts = 3
        result = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = subprocess.run(
                    cmd, cwd=str(work_dir), timeout=self.timeout_s, capture_output=True, text=True,
                )
                break
            except subprocess.TimeoutExpired:
                if attempt == max_attempts:
                    raise
                print(f"[sbd-solve] attempt {attempt}/{max_attempts} timed out after "
                      f"{self.timeout_s}s; retrying...", file=sys.stderr, flush=True)
        if result.returncode != 0:
            tail = (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]
            raise SbdSolverError(
                f"diag-gpu exited {result.returncode} (cmd={cmd}):\n{tail}"
            )

        if keep_stdout:  # only when the caller asked for a persistent work_dir
            (work_dir / "diag_stdout.txt").write_text(result.stdout or "")

        # davidson_energy.txt is the pure electronic energy (FCIDUMP's ECORE defaults to 0) --
        # add the active space's core energy back, same as backend_sqd.py's SqdDiagonalizer.
        energy = sbd_io.read_scalar_txt(work_dir / "davidson_energy.txt") + self._core_energy

        occ_path = work_dir / "occ_a.txt"
        occ_a = np.loadtxt(str(occ_path)) if occ_path.exists() else np.full(self.norb, np.nan)
        occ_a = np.atleast_1d(occ_a).astype(np.float64)

        carryover_path = work_dir / "carryover.bin"
        if carryover_path.exists() and solver_kwargs.get("carryover_ratio", 0) > 0:
            carryover_bits = sbd_io.read_carryover_bin(carryover_path, self.norb)
            weights = (1 << np.arange(self.norb - 1, -1, -1)).astype(np.int64)
            carryover_a = (carryover_bits.astype(np.int64) * weights).sum(axis=1)
        else:
            carryover_a = np.empty(0, dtype=np.int64)

        return energy, occ_a, carryover_a

    def recover_and_diagonalize(
        self,
        counts: dict[tuple[int, int], int],
        samples_per_batch: int = 512,
        num_batches: int = 1,
        max_iterations: int = 10,
        seed: int = 0,
        subspace_dim: int | None = None,
        diagnostics_out: list | None = None,
        resume_state: dict | None = None,
        final_state_out: dict | None = None,
    ) -> tuple[float, CIVector, list[float]]:
        """Self-consistent configuration recovery (CPU, cheap) + diagonalization (GPU, fast).

        Replicates the real multi-step recovery loop from the sibling repo's
        ``sbd/sqd.py`` (``walker_sqd``, RHF/close-shell branch), stripped of
        Prefect: each step (1) recovers configurations from the current
        average orbital occupancies via
        ``qiskit_addon_sqd.configuration_recovery.recover_configurations``,
        (2) post-selects by Hamming weight, (3) subsamples down to
        ``subspace_dim`` alpha determinants (forcing HF + the previous step's
        carryover determinants, then randomly drawing the rest from the
        recovered pool), (4) diagonalizes with ``diag-gpu``, (5) feeds that
        step's ``occ_a.txt``/carryover forward into the next step. Keeps the
        best (lowest-energy) step throughout, matching the "keep the best
        pass" behavior of the reference implementation.

        ``num_batches`` (K-batch SQD, arXiv:2405.05068) draws several
        independent subspaces per step from the same recovered distribution
        and keeps the lowest-energy one -- more GPU calls per step, steadier
        occupancy signal into the next step. ``num_batches=1`` is the simplest
        case (this project's samples are already scarce at 46q real-hardware
        scale, so K-batch mainly matters once more shots are available).

        Returns ``(best_energy, placeholder_CIVector, history)`` where
        ``history`` is the per-step energy list, matching
        ``SqdDiagonalizer.recover_and_diagonalize``'s return shape.

        ``diagnostics_out`` (optional): if a list is passed, this call appends one
        dict per step (plus an initial "step 0" pre-recovery entry) with
        ``{"step": int, "n_valid": int, "energy": float, "excitation_hist": dict}``
        -- excitation_hist is HF/S/D/T/Q/higher counts via
        ``circuit_metrics.excitation_rank_histogram`` on that step's (alpha, alpha)
        RHF determinant pairs. Side-channel, not part of the return tuple, so this
        stays backward compatible with existing 3-value unpacking at every call site.

        ``resume_state``/``final_state_out`` (optional): let a caller genuinely
        CONTINUE a prior call's recovery trajectory -- e.g. run 10 iterations, save
        the state, later extend to 25 MORE on the exact same raw ``counts`` (same
        real-hardware sample, no resampling) picking up exactly where step 10 left
        off, rather than restarting step 1 with a bigger ``max_iterations`` (which
        would just redo the same first 10 steps identically and waste no real
        hardware, but also isn't a real "continue"). ``final_state_out``, if an
        empty dict is passed, is filled in-place after the loop with everything
        needed to resume: ``avg_occ`` (the running per-orbital occupancy estimate),
        ``carryover_a``, ``best_energy``, ``best_strs_a``, ``last_accepted_energy``,
        ``history`` (so far), and ``rng_state`` (the numpy Generator's bit-generator
        state, so the exact same pseudorandom sequence continues rather than
        restarting from ``seed``). Pass that same dict back in as ``resume_state``
        on the next call (with a fresh/larger ``max_iterations`` -- steps are
        numbered from 0 again in that call's own logging, but the recovery
        trajectory itself is a real continuation, not a restart). When
        ``resume_state`` is None (default), starts fresh exactly as before.
        """
        import sys
        from qiskit_addon_sqd.configuration_recovery import (
            post_select_by_hamming_weight,
            recover_configurations,
        )

        norb = self.norb
        num_elec_a, num_elec_b = self.nelec
        if num_elec_a != num_elec_b:
            raise ValueError(
                f"recover_and_diagonalize is RHF-only (requires num_elec_a == num_elec_b); "
                f"got nelec={self.nelec}"
            )
        _subspace_dim_was_explicit = subspace_dim is not None
        subspace_dim = subspace_dim or (samples_per_batch * num_batches)
        print(f"[sbd-recover] subspace_dim={subspace_dim} (net CI dim target "
              f"~{subspace_dim ** 2}) -- "
              f"{'explicit' if _subspace_dim_was_explicit else f'DEFAULTED from samples_per_batch({samples_per_batch})*num_batches({num_batches}), NOT derived from max_dim'}",
              file=sys.stderr, flush=True)

        if not counts:
            return float("inf"), make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64)), []

        # counts keys are already (alpha_int, beta_int) occupation bitmasks (see
        # ConfigurationSet.from_sample_counts). alpha_bits/beta_bits keep column i =
        # orbital i, the order avg_occ is in. recover_configurations wants each half
        # in qiskit order instead, column 0 = orbital norb-1: it pairs column j with
        # np.flip(avg_occupancies)[j]. Passing orbital order mirrored the occupancies,
        # so recovery took electrons out of the fullest orbitals and put them in the
        # emptiest -- backend_sqd.py's MSB-first strings were right all along.
        pairs = list(counts.items())
        alphas = np.array([a for (a, b), _ in pairs], dtype=np.int64)
        betas = np.array([b for (a, b), _ in pairs], dtype=np.int64)
        weights = np.array([c for _, c in pairs], dtype=np.float64)
        probs = weights / weights.sum()

        bit_idx = np.arange(norb)
        alpha_bits = ((alphas[:, None] >> bit_idx[None, :]) & 1).astype(bool)
        beta_bits = ((betas[:, None] >> bit_idx[None, :]) & 1).astype(bool)
        bitstring_matrix = np.concatenate([beta_bits[:, ::-1], alpha_bits[:, ::-1]], axis=1)

        hartreefock = (1 << num_elec_a) - 1

        if resume_state is not None:
            # Genuine continuation: pick up exactly where a prior call's final_state_out
            # left off, on the SAME raw counts/bitstring_matrix (the caller is responsible
            # for passing the same `counts` used to produce resume_state -- this function
            # has no way to verify that itself). rng resumes its exact pseudorandom
            # sequence via the saved bit-generator state, rather than restarting from `seed`.
            rng = np.random.default_rng(seed)
            rng.bit_generator.state = resume_state["rng_state"]
            avg_occ = resume_state["avg_occ"]
            carryover_a = resume_state["carryover_a"]
            best_energy = resume_state["best_energy"]
            best_strs_a = resume_state["best_strs_a"]
            last_accepted_energy = resume_state["last_accepted_energy"]
            history: list[float] = list(resume_state["history"])
        else:
            rng = np.random.default_rng(seed)
            # avg_occ seeds from the raw pool's own mean occupancy (no external CCSD
            # reference available at this call site) -- recover_configurations only needs
            # a plausible starting bias, and it's overwritten by the GPU solver's real
            # occ_a.txt after step 1.
            avg_occ_a = alpha_bits.astype(np.float64).T @ probs
            avg_occ_b = beta_bits.astype(np.float64).T @ probs
            avg_occ = (avg_occ_a, avg_occ_b)
            carryover_a = np.empty(0, dtype=np.int64)
            best_energy = float("inf")
            best_strs_a = np.array([hartreefock], dtype=np.int64)
            # Tracks the last ACCEPTED step's energy (not best_energy, which a bad step
            # would otherwise corrupt) -- the discontinuity guard below compares against
            # this, not against best_energy, so one bad step can't make a second bad
            # step at a similar wrong energy look "consistent" and slip through.
            last_accepted_energy: float | None = None
            history: list[float] = []

        if diagnostics_out is not None and resume_state is None:
            # step-0 (pre-recovery) diagnostic: raw sampled bitstrings, before any
            # recovery -- "valid" = conserves the target electron count per spin,
            # same criterion post_select_by_hamming_weight uses inside the loop.
            # Skipped on resume: the raw sample itself hasn't changed, so this would
            # just duplicate the first call's own step-0 entry.
            n_valid_0 = int(((alpha_bits.sum(axis=1) == num_elec_a) &
                             (beta_bits.sum(axis=1) == num_elec_b)).sum())
            try:
                # Real, independent (alpha, beta) pairs from the raw sample -- use the
                # genuine excitation_rank_histogram, not the RHF-spatial-only helper (that
                # one is for a single shared spatial string per determinant, e.g. strs_a
                # below; passing (alphas, alphas) here previously discarded the real betas
                # entirely and double-counted every alpha-only change as a "double" --
                # confirmed as a real bug, see rhf_spatial_excitation_rank_histogram's
                # docstring in circuit_metrics.py).
                from sqd_gqe.gqe.circuit_metrics import excitation_rank_histogram
                exc_hist_0 = excitation_rank_histogram(
                    np.stack([alphas, betas], axis=1), self.nelec, max_named_rank=4)
            except Exception:
                exc_hist_0 = {}
            diagnostics_out.append({
                "step": 0, "n_valid": n_valid_0, "n_sampled": len(pairs),
                "energy": None, "excitation_hist": exc_hist_0,
            })

        for step in range(max_iterations):
            try:
                recovered_bits, recovered_probs = recover_configurations(
                    bitstring_matrix=bitstring_matrix,
                    probabilities=probs,
                    avg_occupancies=avg_occ,
                    num_elec_a=num_elec_a,
                    num_elec_b=num_elec_b,
                    rand_seed=rng,
                )
            except ValueError as e:
                # avg_occ can still saturate to the exact {0,1} boundary despite the clip
                # above (e.g. every occupied bit in a partition agreeing exactly) -- skip
                # this step rather than losing the whole multi-step loop's progress so far.
                print(f"[sbd-recover] step {step + 1}/{max_iterations}: "
                      f"recover_configurations failed ({e}); skipping this step",
                      file=sys.stderr, flush=True)
                history.append(float("inf"))
                continue
            post_mask = post_select_by_hamming_weight(
                recovered_bits, hamming_right=num_elec_a, hamming_left=num_elec_b,
            )
            post_bits = recovered_bits[post_mask]
            post_probs = np.abs(recovered_probs[post_mask])
            print(f"[sbd-recover] step {step + 1}/{max_iterations}: "
                  f"{post_bits.shape[0]}/{recovered_bits.shape[0]} configs survived "
                  "post-selection", file=sys.stderr, flush=True)
            if post_bits.shape[0] == 0:
                print(f"[sbd-recover] step {step + 1}/{max_iterations}: "
                      "0 configs survived; skipping this step", file=sys.stderr, flush=True)
                history.append(float("inf"))
                continue
            post_probs = post_probs / post_probs.sum()

            # RHF: fold beta+alpha halves together (same as sbd/sqd.py's subsample_close_shell)
            # since alpha==beta determinants for a closed-shell reference.
            # Each half is in qiskit order, so column j weighs 2**(norb-1-j).
            msb_weights = 1 << bit_idx[::-1]
            post_alpha_int = (post_bits[:, norb:].astype(np.int64) * msb_weights).sum(axis=1)
            post_beta_int = (post_bits[:, :norb].astype(np.int64) * msb_weights).sum(axis=1)
            uniq, uniq_probs = fold_rhf_half_strings(np.stack([post_alpha_int, post_beta_int], axis=1), post_probs)

            forced = np.concatenate(([hartreefock], carryover_a))
            forced = np.unique(forced)
            budget = max(int(subspace_dim) - len(forced), 0)
            candidates_mask = ~np.isin(uniq, forced)
            candidates = uniq[candidates_mask]
            cand_probs = uniq_probs[candidates_mask]
            if candidates.size and cand_probs.sum() > 0 and budget > 0:
                n_draw = min(budget, candidates.size)
                drawn = rng.choice(
                    candidates, size=n_draw, replace=False, p=cand_probs / cand_probs.sum(),
                )
            else:
                drawn = np.empty(0, dtype=np.int64)
            strs_a = np.unique(np.concatenate([forced, drawn])).astype(np.int64)

            _dim_flag = " [!! MISMATCH vs subspace_dim target]" if strs_a.size > subspace_dim else ""
            print(f"[sbd-recover] step {step + 1}/{max_iterations}: diagonalizing "
                  f"{strs_a.size} alpha determinants ({strs_a.size ** 2} net dim) on GPU "
                  f"(subspace_dim target={subspace_dim}){_dim_flag}...",
                  file=sys.stderr, flush=True)
            step_energy, step_occ_a, step_carryover_a = self._solve_once(strs_a)
            print(f"[sbd-recover] step {step + 1}/{max_iterations}: energy={step_energy:.6f}",
                  file=sys.stderr, flush=True)

            # Discontinuity guard: diag-gpu's Davidson solver (--iteration/--tolerance,
            # an iterative eigensolver, not exact diagonalization) can converge to a
            # spurious root at some subspace fill fractions -- confirmed directly:
            # beh2 (14q, RHF, 7 orbitals) returned energies 300-9900 mHa off CASCI at
            # several RANDOM VALID subspace sizes between 10 and 34 of its 35 alpha
            # strings, with no monotonic relationship to subspace size, and a
            # real recover_and_diagonalize run on the same molecule produced a step
            # energy 2.7 Ha *below* its CASCI reference -- a variational-bound
            # violation impossible for a genuine diagonalization of a real subspace.
            # No exact reference is available to check against in general (CASCI is
            # intractable past FCI_DIM_LIMIT, e.g. the 46q real-hardware target this
            # is heading toward, where CCSD(T) is the only reference and is itself
            # approximate) -- so this checks internal consistency instead: reject a
            # step whose energy drops by more than DISCONTINUITY_HA from the last
            # ACCEPTED step's energy. A real physical trajectory does not jump by
            # whole Hartree between consecutive recovery steps (the subspace changes
            # incrementally step to step); a solver artifact can. Compares against
            # last_accepted_energy, not best_energy, so one bad step can't set a new
            # "normal" that a second bad step then looks consistent with.
            DISCONTINUITY_HA = 0.5
            step_rejected = (
                last_accepted_energy is not None
                and np.isfinite(step_energy)
                and (last_accepted_energy - step_energy) > DISCONTINUITY_HA
            )
            if step_rejected:
                print(f"[sbd-recover] step {step + 1}/{max_iterations}: REJECTED -- "
                      f"energy {step_energy:.6f} drops {last_accepted_energy - step_energy:.3f} Ha "
                      f"below the last accepted step ({last_accepted_energy:.6f}), "
                      f"exceeding the {DISCONTINUITY_HA} Ha discontinuity guard; "
                      "treating as a solver artifact, not feeding forward",
                      file=sys.stderr, flush=True)
                history.append(float("inf"))
                # Do NOT update best_energy, avg_occ, or carryover_a from a rejected
                # step -- feed the same occupancies/carryover forward again next step,
                # same as the existing "0 configs survived" skip path above.
                continue
            history.append(step_energy)

            if diagnostics_out is not None:
                try:
                    # strs_a is the alpha-string list SBD internally forms into the full
                    # strs_a x strs_a Cartesian product (RHF: one list used for both spin
                    # sectors) -- histogram the FULL product, not just the diagonal
                    # (strs_a[i], strs_a[i]) subset. The diagonal-only version (via
                    # rhf_spatial_excitation_rank_histogram, or the older buggy
                    # np.stack([strs_a, strs_a]) convention) can never show odd ranks
                    # (singles, triples): k(a_i)+k(a_i)=2*k(a_i) is always even. Confirmed
                    # by a collaborator (2026-09-24) comparing against a real raw-sample
                    # histogram that does show nonzero singles/triples -- see
                    # rhf_cartesian_excitation_rank_histogram's own docstring.
                    from sqd_gqe.gqe.circuit_metrics import rhf_cartesian_excitation_rank_histogram
                    exc_hist_step = rhf_cartesian_excitation_rank_histogram(
                        strs_a, self.nelec, max_named_rank=4)
                except Exception:
                    exc_hist_step = {}
                diagnostics_out.append({
                    "step": step + 1, "n_valid": int(strs_a.size), "n_sampled": len(pairs),
                    "energy": step_energy, "excitation_hist": exc_hist_step,
                })

            last_accepted_energy = step_energy
            if step_energy < best_energy:
                best_energy = step_energy
                best_strs_a = strs_a

            # feed forward: RHF occupancies are symmetric across spin; carryover from this
            # step forces those determinants into the next step's forced set.
            #
            # Clip away from the exact {0,1} boundary: a small/homogeneous subsample (the
            # candidate pool shrinks each step as post-selection narrows it, same as the
            # reference sbd/sqd.py) can saturate an orbital's occupancy to exactly 0.0 or
            # 1.0. qiskit_addon_sqd's _p_flip_0_to_1/_p_flip_1_to_0 both return exactly 0.0
            # at that boundary, and recover_configurations then divides by
            # sum(probs_left[occupied bits]) with no zero-guard -- a NaN crash confirmed on
            # real hardware data at step 5->6 (occupancies saturated after the subsample
            # shrank to 193 alpha determinants). qiskit-addon-sqd's own
            # diagonalize_fermionic_hamiltonian doesn't hit this because it keeps
            # subspace_dim fixed every step; recover_and_diagonalize's shrinking subsample
            # (matching the real reference algorithm) needs this guard instead.
            if np.all(np.isfinite(step_occ_a)):
                clipped = np.clip(step_occ_a, 1e-3, 1 - 1e-3)
                avg_occ = (clipped, clipped)
            if step_carryover_a.size:
                # Filter the solver's carryover list to determinants that actually
                # conserve the alpha electron count. diag-gpu's carryover.bin gives no
                # such guarantee, and a wrong-population determinant that reaches the
                # next step's `forced` set poisons the whole subspace:
                #
                # Confirmed on CCC 2026-09-20 -- ch2o (norb=10, nelec=(6,6)) reached
                # strs_a = [0, 63, 960], i.e. popcount 0 / 6 / 4 where only 6 is valid.
                # Inside diag-gpu, helper.h's GenerateExcitation then classified those
                # pairs as double excitations via difference()==4 (a symmetric-difference
                # popcount, which a population-changing pair also satisfies) while
                # OrbitalDifference() returned fewer than 2 creation/annihilation
                # orbitals, so cr[1]/an[1] read uninitialized reserve()d memory. The
                # garbage indices reached the GPU and made TwoExcite's parity() walk
                # index dets[] out of bounds -- compute-sanitizer reported "Invalid
                # __global__ read ... is out of bounds", surfacing as
                # cudaErrorIllegalAddress / SIGABRT (exit 134) at a rate that varied
                # 18-100% by compute host, because what the garbage index lands on
                # depends on the surrounding allocation layout.
                _keep = _popcount_int64(step_carryover_a) == num_elec_a
                _n_bad = int((~_keep).sum())
                if _n_bad:
                    print(f"[sbd-recover] step {step + 1}/{max_iterations}: dropped "
                          f"{_n_bad}/{step_carryover_a.size} carryover determinants whose "
                          f"alpha population != {num_elec_a}", file=sys.stderr, flush=True)
                _filtered = step_carryover_a[_keep]
                if _filtered.size:
                    carryover_a = _filtered

        if final_state_out is not None:
            final_state_out.update({
                "avg_occ": avg_occ,
                "carryover_a": carryover_a,
                "best_energy": best_energy,
                "best_strs_a": best_strs_a,
                "last_accepted_energy": last_accepted_energy,
                "history": list(history),
                "rng_state": rng.bit_generator.state,
            })

        # PLACEHOLDER (no amplitudes, see module docstring): uniform weights over the (a, a)
        # pairs of the best step's alpha list, so refine_method="tp_ufd" with
        # refine_input="recovered" can union the *recovered* half-strings across circuits.
        n = best_strs_a.size
        pairs = np.stack([best_strs_a, best_strs_a], axis=1).astype(np.uint64)
        placeholder = make_ci_vector(np.full(n, 1.0 / np.sqrt(n)), pairs)
        return best_energy, placeholder, history
