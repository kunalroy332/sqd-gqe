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

"""SQD pipeline (qiskit-addon-sqd + ffsim backend).

For each generated circuit: sample configurations from the ffsim state, select
those that conserve electron number, expand by spin-symmetry completion,
diagonalize the sampled subspace with qiskit-addon-sqd, then run the shared
local/global subspace merger (non-orthogonal GEVP refinement).
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch

from sqd_gqe.core.system import MolecularSystem, FCI_DIM_LIMIT
from sqd_gqe.core.configuration import ConfigurationSet, make_ci_vector, CIVector
from sqd_gqe.core.refine import NonOrthogonalSolver, build_merger
from sqd_gqe.core.report import SqdSample, SqdReport
from sqd_gqe.gqe.sampler import FfsimSampler
from sqd_gqe.gqe.operator_pool import FfsimBasedPool
from sqd_gqe.sqd.backend_sqd import SqdDiagonalizer


class SqdPipeline:
    def __init__(
        self,
        system: MolecularSystem,
        operator_pool: FfsimBasedPool,
        sampler: FfsimSampler,
        max_size: int = 100_000,
        max_cycle: int = 200,
        expand_method: str = "symmetry_completion",
        diag_backend: str = "sqd",   # "sqd" (qiskit-addon-sqd) | "fci" (PySCF parity) | "sbd" (GPU)
        spin_sq: float | None = 0.0,
        sbd_binary_path: str | None = None,
        sbd_launcher: list[str] | None = None,
        sbd_solver_kwargs: dict | None = None,
        sbd_timeout_s: float | None = 600.0,
        recover_iterations: int = 0,
        recover_subspace_dim: int | None = None,
        use_sparse_refine_hamiltonian: bool = True,
        refine_method: str = "gevp",       # "gevp" | "ufd" (det union, fci dense) | "tp_ufd" (half-string union)
        global_refine: str = "bounded",    # "bounded" | "cumulative"
        cumulative_max_dim: int = 5_000_000,
        tp_ufd_n_alpha: int | None = None,
        ufd_dense_limit: int = 20_000,
        refine_input: str = "oneshot",     # "oneshot" | "recovered": which per-circuit state feeds refinement
        max_depth_for_diag: int | None = None,
    ):
        self.system = system
        self.hamiltonian = system.active_hamiltonian
        self.norb = system.norb
        self.nelec = system.nelec
        self.operator_pool = operator_pool
        self.sampler = sampler
        self.max_size = max_size
        self.max_cycle = max_cycle
        self.expand_method = expand_method
        self.diag_backend = diag_backend
        # >0 switches each sampled circuit's reward energy from a one-shot
        # diagonalize() to the self-consistent recover_and_diagonalize() loop
        # (recover_configurations on CPU + diagonalize on GPU/CPU each step,
        # see backend_sbd_gpu.py/backend_sqd.py). The refine (merge) steps use
        # the one-shot CIVector unless refine_input="recovered", in which case the
        # recovery loop's final state (sqd: real amplitudes; sbd: placeholder whose
        # support is the best step's alpha list, usable by tp_ufd) is merged instead.
        self.recover_iterations = int(recover_iterations)
        # alpha-determinant count for recover_and_diagonalize's GPU (sbd) step each
        # recovery iteration -- RHF, so net CI dim = this^2. Without this, sbd's own
        # recover_and_diagonalize defaults to samples_per_batch*num_batches=512 dets
        # (262144 net dim), silently ignoring max_size/--max-dim -- the same gap
        # run.py's --recover path was patched for after job 63162 (see run.py's
        # recover_subspace_dim comment), which never touched this training path.
        self.recover_subspace_dim = recover_subspace_dim
        self.refine_method = refine_method
        if refine_input not in ("oneshot", "recovered"):
            raise ValueError(f"refine_input must be 'oneshot' or 'recovered'; got {refine_input!r}")
        self.refine_input = refine_input
        # Real hardware jobs occasionally transpile one circuit in a batch to a depth
        # 10-1000x the rest (a bad gate-pool pick composing badly into the device's
        # native gate set) -- diagonalizing it costs the same expensive GPU call as
        # every other circuit for a result that's virtually always discarded anyway
        # (it never wins "keep the best of the batch"). None (default) diagonalizes
        # every circuit regardless of depth, matching all behavior before this flag
        # existed. Set to skip diagonalize()/recover() for any circuit whose sampler-
        # reported depth (IbmHardwareSampler.last_depths: real transpiled depth;
        # FfsimSampler.last_depths: pool-sequence length, ffsim never transpiles) exceeds
        # this -- the circuit still gets a SqdSample (energy=inf, diagonalize_skipped_deep=True,
        # depth recorded) so its hardware counts and rejection are visible in the report/
        # history, it just never reaches the GPU/CPU solver.
        self.max_depth_for_diag = max_depth_for_diag
        # Purely cosmetic: lets the sampler's own post-transpile log line report how many
        # circuits will be rejected, right there, before diagonalize() runs -- the sampler
        # never changes its own sample/decode behavior based on this.
        if hasattr(sampler, "max_depth_for_diag_for_log"):
            sampler.max_depth_for_diag_for_log = max_depth_for_diag
        self.global_refine = global_refine
        self.cumulative_max_dim = int(cumulative_max_dim)
        self.global_refined_state: CIVector | None = None
        self.timings: dict[str, float] = {}

        # SubspaceMerger's NonOrthogonalSolver projects H onto the merged
        # subspace. The ORIGINAL implementation (still available via
        # use_sparse_refine_hamiltonian=False, see refine.py's module
        # docstring) did this via exact full-active-space FCI contraction
        # (PySCF direct_spin1.contract_2e) -- its cost was set by
        # system.fci_dimension(), NOT by the sampled/merged subspace size, so
        # it was completely independent of max_size/subspace_dim. At 38q
        # (SnOH3-Me tier M, 19o/24e) this was ~2.5e9-element/~20GB-per-vector
        # -- confirmed to run past a 12hr Slurm wall-clock limit with zero
        # iterations completed (job 62055). The DEFAULT sparse implementation
        # (refine.py's _projected_hamiltonian_sparse) removes this scaling
        # entirely -- verified >8000x faster and numerically identical at
        # co2/d-block scale (see refine.py's module docstring) -- but does NOT
        # change the FCI_DIM_LIMIT gate below, which exists for a different
        # reason (system.casci_energy()'s own tractability, not refine's old
        # cost problem) and was not re-evaluated as part of this fix. The gate
        # itself is computed below, once self.merger exists, since "ufd"/
        # "tp_ufd" mergers don't all share gevp's full-FCI-contraction cost
        # profile (see needs_full_fci on each merger class).

        self._sqd = SqdDiagonalizer(self.hamiltonian, self.nelec, self.norb, spin_sq=spin_sq)
        # PySCF-FCI parity path: a single-vector solve on the subspace basis.
        self._fci = NonOrthogonalSolver(self.hamiltonian, self.nelec, self.norb,
                                        use_sparse_hamiltonian=use_sparse_refine_hamiltonian)
        self._sbd = None
        if diag_backend == "sbd":
            if not sbd_binary_path:
                raise ValueError('diag_backend="sbd" requires cfg.sqd.sbd_binary_path to be set')
            from sqd_gqe.sqd.backend_sbd_gpu import SbdGpuDiagonalizer
            self._sbd = SbdGpuDiagonalizer(
                self.hamiltonian, self.nelec, self.norb, binary_path=sbd_binary_path,
                launcher=sbd_launcher, solver_kwargs=sbd_solver_kwargs, timeout_s=sbd_timeout_s,
                max_dim=max_size,
            )
        self.merger = build_merger(
            refine_method, self.hamiltonian, self.nelec, self.norb, spin_sq=spin_sq, sbd=self._sbd,
            diag_backend=diag_backend, n_alpha=tp_ufd_n_alpha, dense_limit=ufd_dense_limit,
            use_sparse_hamiltonian=use_sparse_refine_hamiltonian,
        )
        # Mergers that contract over the full FCI space (gevp, ufd) declare needs_full_fci;
        # tp_ufd bounds itself by closure_limit instead.
        self.refine_intractable = bool(getattr(self.merger, "needs_full_fci", True)) and \
            system.fci_dimension() > FCI_DIM_LIMIT

    def close(self) -> None:
        """Release resources owned by the configured sampler."""
        close = getattr(self.sampler, "close", None)
        if close is not None:
            close()

    @contextmanager
    def _timed(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + (time.perf_counter() - start)

    # -- diagonalization ----------------------------------------------------- #
    def diagonalize(self, config_set: ConfigurationSet) -> tuple[float, CIVector]:
        if self.diag_backend == "fci":
            return self._diagonalize_fci(config_set)
        if self.diag_backend == "sbd":
            return self._sbd.diagonalize(config_set)
        return self._sqd.diagonalize(config_set)

    def energy_of_counts(self, counts: dict[tuple[int, int], int],
                         diagnostics_out: list | None = None) -> float:
        """Reward-path energy for one circuit's raw sampled counts.

        One-shot diagonalize() when recover_iterations==0 (default, cheap).
        Otherwise runs recover_iterations steps of self-consistent
        configuration recovery on whichever backend is active (sbd/GPU or
        sqd/CPU) and returns the best energy seen across steps -- a much
        more accurate reward signal, at recover_iterations x the per-circuit
        diagonalization cost.

        ``diagnostics_out``: optional list passed straight through to the GPU
        (sbd) backend's recover_and_diagonalize -- see its docstring. No effect
        on the CPU (sqd) backend, which doesn't support this side-channel.
        """
        if self.recover_iterations <= 0:
            config_set = ConfigurationSet.from_sample_counts(counts)
            valid = config_set.select_by_nelec(self.nelec)
            expanded = valid.expand(max_size=self.max_size, method=self.expand_method)
            energy, _ci = self.diagonalize(expanded)
            return energy
        energy, _ci = self.recover(counts, diagnostics_out=diagnostics_out)
        return energy

    def recover(self, counts: dict[tuple[int, int], int],
                diagnostics_out: list | None = None) -> tuple[float, CIVector]:
        """Configuration-recovery loop on the active backend; returns (best energy, state).

        On sqd the state is the real CI vector of the final recovery step. On sbd it is
        the placeholder whose *support* is the best step's alpha list (see
        backend_sbd_gpu.py) -- enough for refine_method="tp_ufd" (refine_input="recovered").
        """
        recoverer = self._sbd if (self.diag_backend == "sbd" and self._sbd is not None) else self._sqd
        kwargs = {"max_iterations": self.recover_iterations}
        if recoverer is self._sbd:
            if diagnostics_out is not None:
                kwargs["diagnostics_out"] = diagnostics_out
            if self.recover_subspace_dim is not None:
                kwargs["subspace_dim"] = self.recover_subspace_dim
                import sys
                print(f"[pipeline] recover() subspace_dim={self.recover_subspace_dim} "
                      f"(net CI dim target ~{self.recover_subspace_dim ** 2})",
                      file=sys.stderr, flush=True)
            else:
                import sys
                print("[pipeline] recover() subspace_dim NOT SET (self.recover_subspace_dim "
                      "is None) -- recover_and_diagonalize will default to "
                      "samples_per_batch*num_batches, NOT derived from max_dim",
                      file=sys.stderr, flush=True)
        energy, ci, _history = recoverer.recover_and_diagonalize(counts, **kwargs)
        return energy, ci

    def _diagonalize_fci(self, config_set: ConfigurationSet) -> tuple[float, CIVector]:
        if config_set.size == 0:
            return float("inf"), make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64))
        strings = np.asarray(config_set.configs, dtype=np.uint64).reshape(-1, 2)
        energy, coeffs = self._fci.diagonalize_subspace(strings)
        return energy, make_ci_vector(coeffs, strings)

    def _global_merge(self, ci_vectors, local_ci, refine_dim: int):
        """Global (cross-iteration) refinement input set.

        ``bounded``: previous retained state U this batch's local state, capped at
        ``refine_dim`` (the original behaviour). ``cumulative``: previous state U
        this batch's local state U every per-circuit state, capped only at ``cumulative_max_dim``
        (the retained set lives in ``global_refined_state.strings``, so nothing new
        needs checkpointing).
        """
        if self.global_refine == "cumulative":
            return self.merger.merge([self.global_refined_state, local_ci, *ci_vectors],
                                     max_size=self.cumulative_max_dim)
        return self.merger.merge([self.global_refined_state, local_ci], max_size=refine_dim)

    # -- main loop ----------------------------------------------------------- #
    def run(self, states: dict[str, torch.Tensor], collect_recovery_diagnostics: bool = False) -> SqdReport:
        self.timings = {}
        with self._timed("sample"):
            sampled_counts = self.sampler.run(states)
        # Populated by both FfsimSampler and IbmHardwareSampler's run() as a side-channel
        # (see their last_depths docstrings) -- same length/order as sampled_counts, or
        # empty if the sampler predates this attribute (defensive, not expected here).
        depths = list(getattr(self.sampler, "last_depths", []) or [])

        samples = []
        ci_vectors = []
        for i, (seq, counts) in enumerate(zip(states["idx"].tolist(), sampled_counts)):
            gate_counts = self.operator_pool.get_gate_count(seq)
            depth = depths[i] if i < len(depths) else None
            if self.max_depth_for_diag is not None and depth is not None \
                    and depth > self.max_depth_for_diag:
                # Skip the expensive diagonalization entirely -- energy=inf so this
                # circuit can never win "keep the best of the batch" (matching what
                # would happen anyway almost every time), but its hardware counts and
                # the fact that it was rejected are still recorded for later inspection.
                samples.append(
                    SqdSample(
                        seq=tuple(seq), energy=float("inf"), n_sampled=len(counts),
                        n_symmetry_valid=None, subspace_size=0,
                        cx_count=gate_counts.get("cx", 0), total_gates=gate_counts.get("total", 0),
                        counts=counts, depth=depth, diagonalize_skipped_deep=True,
                    )
                )
                # A skipped circuit contributes no CIVector -- ci_vectors must stay
                # aligned 1:1 with samples for the refine step below, so append a
                # zero-size placeholder rather than leaving a gap.
                ci_vectors.append(make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64)))
                continue
            config_set = ConfigurationSet.from_sample_counts(counts)
            valid = config_set.select_by_nelec(self.nelec)
            expanded = valid.expand(max_size=self.max_size, method=self.expand_method)
            recovery_diag: list | None = [] if collect_recovery_diagnostics else None
            with self._timed("diagonalize"):
                one_shot_energy, ci = self.diagonalize(expanded)
                if self.recover_iterations > 0:
                    energy, recovered_ci = self.recover(counts, diagnostics_out=recovery_diag)
                    if self.refine_input == "recovered":
                        ci = recovered_ci
                else:
                    energy = one_shot_energy
            ci_vectors.append(ci)
            samples.append(
                SqdSample(
                    seq=tuple(seq),
                    energy=energy,
                    n_sampled=len(counts),
                    n_symmetry_valid=valid.size,
                    subspace_size=expanded.size,
                    cx_count=gate_counts.get("cx", 0),
                    total_gates=gate_counts.get("total", 0),
                    recovery_diagnostics=recovery_diag,
                    counts=counts,
                    depth=depth,
                )
            )

        local_refined = None
        global_refined = None
        # max_depth_for_diag can skip every circuit in an unlucky/small batch (all
        # sampled deeper than the threshold) -- ci_vectors is then all zero-size
        # placeholders, and GevpMerger's overlap matrix is degenerate on an all-empty
        # union ("Overlap matrix S is numerically singular", confirmed directly).
        # Nothing to refine in that case; leave local/global refined as None, same
        # as the ordinary "no circuits sampled" shape elsewhere in this class.
        any_diagonalized = any(np.asarray(v).size for v in ci_vectors)
        if self.refine_intractable:
            self.global_refined_state = ci_vectors[-1] if ci_vectors else None
        elif not any_diagonalized:
            pass
        else:
            with self._timed("refine_local"):
                local_set = self.merger.merge(ci_vectors, max_size=self.max_size)
                local_energy, local_ci = self.diagonalize(local_set)
            local_refined = SqdSample(
                seq=None, energy=local_energy, n_sampled=None, n_symmetry_valid=None,
                subspace_size=local_set.size, cx_count=None, total_gates=None,
            )

            if self.global_refined_state is None:
                self.global_refined_state = local_ci
            else:
                with self._timed("refine_global"):
                    global_set = self._global_merge(ci_vectors, local_ci, self.max_size)
                    global_energy, global_ci = self.diagonalize(global_set)
                self.global_refined_state = global_ci
                global_refined = SqdSample(
                    seq=None, energy=global_energy, n_sampled=None, n_symmetry_valid=None,
                    subspace_size=global_set.size, cx_count=None, total_gates=None,
                )

        return SqdReport(samples=tuple(samples), local_refined=local_refined, global_refined=global_refined)

    # -- dynamic subspace size ------------------------------------------------ #
    def solve_adaptive(
        self,
        valid: ConfigurationSet,
        start_dim: int,
        growth: float = 2.0,
        max_dim_cap: int = 1_000_000,
        energy_tol_ha: float = 1.6e-4,
        time_budget_s: float | None = None,
    ) -> tuple[float, CIVector, int, list[dict]]:
        """Grow the subspace size from a predicted starting guess until the
        energy stops moving, instead of diagonalizing at one fixed max_dim.

        This is the runtime robustness net around ResourceHead's prediction:
        the predictor gives a starting point, this loop corrects it when wrong
        (e.g. extrapolating to a molecule outside the predictor's training
        distribution) by doubling ``dim`` while the energy is still changing by
        more than ``energy_tol_ha`` (tighter than chemical accuracy, so growth
        stops just past the point that matters), and stops at ``max_dim_cap``
        or ``time_budget_s`` if convergence is never reached. Generalizes the
        manual CO2 sweep (2000 -> 20000 -> 50000 -> 1e6) in
        benchmark/co2_dim_fix.py into an automatic per-molecule procedure.

        Returns (energy, ci_vector, dim_used, history) where history is the
        list of {"dim", "energy"} tried, for diagnostics/plotting.
        """
        dim = max(int(start_dim), 1)
        prev_energy = None
        history: list[dict] = []
        energy, ci = float("inf"), make_ci_vector(np.zeros(0), np.zeros((0, 2), dtype=np.uint64))
        t0 = time.perf_counter()
        while True:
            expanded = valid.expand(max_size=dim, method=self.expand_method)
            energy, ci = self.diagonalize(expanded)
            history.append({"dim": dim, "energy": energy})
            converged = prev_energy is not None and abs(energy - prev_energy) < energy_tol_ha
            cap_hit = dim >= max_dim_cap
            time_hit = time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s
            if converged or cap_hit or time_hit:
                break
            prev_energy = energy
            dim = min(int(dim * growth), max_dim_cap)
        return energy, ci, dim, history

    def run_adaptive(
        self,
        states: dict[str, torch.Tensor],
        start_dim: int,
        growth: float = 2.0,
        max_dim_cap: int = 1_000_000,
        energy_tol_ha: float = 1.6e-4,
        time_budget_s: float | None = None,
    ) -> SqdReport:
        """Same shape as run(), but each circuit's diagonalization uses
        solve_adaptive() instead of a fixed self.max_size. The refine (merge)
        steps use the largest dim any circuit in this batch actually needed,
        so they stay consistent with what the batch required rather than an
        arbitrary fixed config value."""
        self.timings = {}
        with self._timed("sample"):
            sampled_counts = self.sampler.run(states)

        samples = []
        ci_vectors = []
        dims_used = []
        for seq, counts in zip(states["idx"].tolist(), sampled_counts):
            gate_counts = self.operator_pool.get_gate_count(seq)
            config_set = ConfigurationSet.from_sample_counts(counts)
            valid = config_set.select_by_nelec(self.nelec)
            with self._timed("diagonalize"):
                energy, ci, dim_used, _hist = self.solve_adaptive(
                    valid, start_dim, growth, max_dim_cap, energy_tol_ha, time_budget_s)
                if self.recover_iterations > 0:
                    energy, recovered_ci = self.recover(counts)
                    if self.refine_input == "recovered":
                        ci = recovered_ci
            dims_used.append(dim_used)
            ci_vectors.append(ci)
            samples.append(
                SqdSample(
                    seq=tuple(seq),
                    energy=energy,
                    n_sampled=len(counts),
                    n_symmetry_valid=valid.size,
                    subspace_size=dim_used,
                    cx_count=gate_counts.get("cx", 0),
                    total_gates=gate_counts.get("total", 0),
                    counts=counts,
                )
            )

        local_refined = None
        global_refined = None
        if self.refine_intractable:
            self.global_refined_state = ci_vectors[-1] if ci_vectors else None
        else:
            refine_dim = max(dims_used) if dims_used else self.max_size
            with self._timed("refine_local"):
                local_set = self.merger.merge(ci_vectors, max_size=refine_dim)
                local_energy, local_ci = self.diagonalize(local_set)
            local_refined = SqdSample(
                seq=None, energy=local_energy, n_sampled=None, n_symmetry_valid=None,
                subspace_size=local_set.size, cx_count=None, total_gates=None,
            )

            if self.global_refined_state is None:
                self.global_refined_state = local_ci
            else:
                with self._timed("refine_global"):
                    global_set = self._global_merge(ci_vectors, local_ci, refine_dim)
                    global_energy, global_ci = self.diagonalize(global_set)
                self.global_refined_state = global_ci
                global_refined = SqdSample(
                    seq=None, energy=global_energy, n_sampled=None, n_symmetry_valid=None,
                    subspace_size=global_set.size, cx_count=None, total_gates=None,
                )

        return SqdReport(samples=tuple(samples), local_refined=local_refined, global_refined=global_refined)
