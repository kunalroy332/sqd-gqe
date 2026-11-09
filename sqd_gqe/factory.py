# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Builder for the SQD (ffsim + qiskit-addon-sqd) arm.

Constructs the transformer policy, group-relative objective, beta schedule,
molecular system, metric reporter, ffsim operator pool + sampler, and the SQD
pipeline. Fully self-contained: imports only from ``sqd_gqe``.
"""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate

from sqd_gqe.core.objective import GRPOObjective, GSPOObjective
from sqd_gqe.core.reporter import MetricReporter
from sqd_gqe.gqe.operator_pool import FfsimLUCJPool
from sqd_gqe.gqe.sampler import FfsimSampler
from sqd_gqe.sqd.pipeline import SqdPipeline


class SqdFactory:
    def __init__(self):
        self.system = None
        self.operator_pool = None

    # -- classical parts ----------------------------------------------------- #
    def build_policy(self, cfg):
        vocab_size = self.build_operator_pool(cfg).get_vocab_size()
        print(f"Vocab size: {vocab_size}")
        cfg.vocab_size = vocab_size
        return instantiate(cfg.model, vocab_size=vocab_size, ngates=cfg.ngates)

    def build_beta_schedule(self, cfg):
        return instantiate(cfg.trainer.beta_schedule)

    def build_system(self, cfg):
        if self.system is None:
            self.system = instantiate(cfg.molecule)
        return self.system

    def build_reporter(self, cfg):
        system = self.build_system(cfg)
        refs = {}
        for key in cfg.reference_keys:
            if key == "hf_energy":
                refs[key] = system.hf.e_tot
            elif key == "R-CASCI":
                refs[key] = system.casci_energy()
                print(f"CASCI Energy: {refs[key]}")
            elif key == "R-CCSD":
                refs[key] = system.ccsd_energy()
                print(f"CCSD Energy: {refs[key]}")
        return MetricReporter(reference_energies=refs)

    def build_objective(self, cfg):
        name = cfg.trainer.loss.type
        match name:
            case "grpo":
                assert cfg.trainer.batch_size == cfg.trainer.num_samples, \
                    "batch_size must equal num_samples for GRPO"
                return GRPOObjective(cfg.trainer.loss.clip_grpo_low, cfg.trainer.loss.clip_grpo_high)
            case "gspo":
                assert cfg.trainer.batch_size == cfg.trainer.num_samples, \
                    "batch_size must equal num_samples for GSPO"
                return GSPOObjective(cfg.trainer.loss.clip_gspo_low, cfg.trainer.loss.clip_gspo_high)
            case _:
                raise ValueError(f"Unknown objective: {name}")

    # -- quantum parts ------------------------------------------------------- #
    def build_operator_pool(self, cfg):
        if self.operator_pool is not None:
            return self.operator_pool
        system = self.build_system(cfg)
        pool_cfg = cfg.operator_pool
        match pool_cfg.spec:
            case "ffsim_lucj":
                pool = FfsimLUCJPool(
                    system,
                    params=pool_cfg.get("params", None),
                    angle_source=pool_cfg.get("angle_source", "ffsim_native"),
                    n_reps=pool_cfg.get("n_reps", None),
                    include_orbital_rotations=pool_cfg.get("include_orbital_rotations", True),
                    include_givens=pool_cfg.get("include_givens", True),
                    include_diag_coulomb=pool_cfg.get("include_diag_coulomb", True),
                    include_num_num=pool_cfg.get("include_num_num", True),
                    n_orbital_rotations=pool_cfg.get("n_orbital_rotations", 4),
                    seed=pool_cfg.get("seed", 0),
                    max_vocab=pool_cfg.get("max_vocab", None),
                )
            case _:
                raise ValueError(f"Unknown ffsim operator pool spec: {pool_cfg.spec}")
        self.operator_pool = pool
        return pool

    def build_sampler(self, cfg, backend=None, shots: int | None = None,
                      max_dim: int | None = None):
        """The sampler for this config. ``shots`` and ``max_dim`` override cfg.sampler.shots
        and cfg.sqd.max_dim (build_pipeline may predict them). ``max_dim`` is the top-k
        kinds' K unless sampler.top_k is set."""
        pool = self.build_operator_pool(cfg)
        shots = int(cfg.sampler.shots if shots is None else shots)
        sampler_kind = cfg.sampler.get("kind", "ffsim")
        if sampler_kind in ("topk", "mps_topk"):
            if max_dim is None:
                max_dim = cfg.sqd.max_dim
            top_k = cfg.sampler.get("top_k", None) or max_dim
        # Device noise applies to every shot-drawing kind (ffsim, mps) by one rule: an
        # explicit p_readout, or a *_noisy_sim backend for the gate and T1 terms.
        p_readout = float(cfg.sampler.get("p_readout", 0.0) or 0.0)
        is_noisy_sim = backend is not None and getattr(backend, "sim_kind", None) == "noisy_sim"
        noisy = (p_readout > 0 or float(cfg.sampler.get("p_readout_beta", None) or 0) > 0
                 or is_noisy_sim)
        # amplitude_damping needs backend.t1_us/gate_time_2q_ns, which only a *_noisy_sim
        # backend carries, so the flag is ignored without one.
        noise_kw = dict(
            p_readout=p_readout,
            backend=backend if is_noisy_sim else None,
            amplitude_damping=bool(cfg.sampler.get("amplitude_damping", False)) and is_noisy_sim,
            noise_optimization_level=int(cfg.sampler.get("noise_optimization_level", 3)),
            p_readout_beta=(None if cfg.sampler.get("p_readout_beta", None) is None
                            else float(cfg.sampler.get("p_readout_beta"))),
            depth_noise=(None if cfg.sampler.get("depth_noise", None) is None
                         else tuple(float(x) for x in cfg.sampler.get("depth_noise"))),
        )
        if sampler_kind in ("topk", "mps_topk") and noisy:
            # top-k kinds read amplitudes, so there are no shots for the noise to act on
            raise ValueError(f"sampler.kind={sampler_kind!r} reads amplitudes and cannot carry "
                             "device noise; use a shot-drawing kind (ffsim, mps)")
        if sampler_kind == "ibm_hardware":
            from sqd_gqe.gqe.ibm_hardware_sampler import IbmHardwareSampler
            import os
            backend_name = cfg.sampler.get("ibm_backend_name", None)
            if not backend_name:
                raise ValueError('cfg.sampler.kind="ibm_hardware" requires cfg.sampler.ibm_backend_name')
            sampler = IbmHardwareSampler(
                pool, shots_count=shots, backend_name=backend_name,
                seed=cfg.trainer.seed,
                channel=cfg.sampler.get("ibm_channel", None) or "ibm_quantum_platform",
                instance=cfg.sampler.get("ibm_instance", None) or os.environ.get("IBM_CRN"),
                token=os.environ.get("IBM_API_KEY"),
                fake_backend=bool(cfg.sampler.get("ibm_fake_backend", False)),
                save_dir=cfg.sampler.get("ibm_save_dir", None),
                dynamical_decoupling=bool(cfg.sampler.get("ibm_dynamical_decoupling", False)),
                dd_sequence_type=cfg.sampler.get("ibm_dd_sequence_type", "XpXm"),
                init_qubits=bool(cfg.sampler.get("ibm_init_qubits", True)),
                session_mode=bool(cfg.sampler.get("ibm_session_mode", False)),
                session_max_time=cfg.sampler.get("ibm_session_max_time", None),
            )
        elif sampler_kind in ("mps", "mps_topk"):
            mps_kwargs = dict(
                shots_count=shots,
                max_bond=int(cfg.sampler.get("mps_max_bond", 256)),
                abs_cutoff=float(cfg.sampler.get("mps_abs_cutoff", 1e-5)),
                rel_cutoff=float(cfg.sampler.get("mps_rel_cutoff", 1e-5)),
                dtype=cfg.sampler.get("mps_dtype", "complex64"),
                route=bool(cfg.sampler.get("mps_route", False)),
                max_gates=cfg.sampler.get("mps_max_gates", None),
                gauge_option=cfg.sampler.get("mps_gauge_option", "simple"),
                canonical_center=cfg.sampler.get("mps_canonical_center", None),
                seed=cfg.trainer.seed,
            )
            if sampler_kind == "mps_topk":
                from sqd_gqe.gqe.mps_topk_sampler import MpsTopKSampler
                sampler = MpsTopKSampler(pool, top_k=int(top_k),
                                         beam_width=cfg.sampler.get("mps_beam_width", None),
                                         min_prob=cfg.sampler.get("min_prob", "auto"),
                                         **mps_kwargs)
            elif noisy:
                from sqd_gqe.gqe.mps_sampler import NoisyMpsSampler
                sampler = NoisyMpsSampler(pool, **noise_kw, **mps_kwargs)
            else:
                from sqd_gqe.gqe.mps_sampler import MpsSampler
                sampler = MpsSampler(pool, **mps_kwargs)
        elif sampler_kind == "topk":
            from sqd_gqe.gqe.sampler import ExactTopKSampler
            sampler = ExactTopKSampler(
                pool, shots_count=shots, top_k=int(top_k), seed=cfg.trainer.seed,
                rayon_threads=cfg.sampler.get("rayon_threads", None),
                min_prob=cfg.sampler.get("min_prob", "auto"),
            )
        elif sampler_kind != "ffsim":
            raise ValueError(f"Unknown sampler.kind={sampler_kind!r}")
        elif noisy:
            from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler
            sampler = NoisyFfsimSampler(
                pool, shots_count=shots, seed=cfg.trainer.seed,
                rayon_threads=cfg.sampler.get("rayon_threads", None), **noise_kw,
            )
        else:
            sampler = FfsimSampler(
                pool, shots_count=shots, seed=cfg.trainer.seed,
                rayon_threads=cfg.sampler.get("rayon_threads", None),
            )
        return sampler

    def build_pipeline(self, cfg, backend=None):
        """``backend`` (a HardwareBackend) is only required when
        cfg.dynamic_resources is true -- it's part of the ResourceHead's
        conditioning descriptor. Ignored otherwise."""
        if cfg.sampler.get("kind", "ffsim") == "mps_topk":
            incompatible = []
            if cfg.sqd.get("enlarge_method", "symmetry_completion") != "None":
                incompatible.append("sqd.enlarge_method=None")
            if int(cfg.sqd.get("recover_iterations", 5)) != 0:
                incompatible.append("sqd.recover_iterations=0")
            if incompatible:
                raise ValueError(
                    "sampler.kind='mps_topk' requires " + " and ".join(incompatible)
                    + " to preserve the selected top-K subspace."
                )

        system = self.build_system(cfg)
        pool = self.build_operator_pool(cfg)

        max_dim, shots = cfg.sqd.max_dim, cfg.sampler.shots
        if bool(cfg.get("dynamic_resources", False)):
            max_dim, shots = self._predict_resources(cfg, system, backend)

        sampler = self.build_sampler(cfg, backend, shots, max_dim)
        sbd_launcher = cfg.sqd.get("sbd_launcher", None)
        sbd_solver_kwargs = cfg.sqd.get("sbd_solver_kwargs", None)
        pipeline = SqdPipeline(
            system,
            pool,
            sampler,
            max_size=max_dim,
            expand_method=cfg.sqd.enlarge_method,
            max_cycle=cfg.sqd.max_cycle,
            diag_backend=cfg.sqd.get("diag_backend", "sqd"),
            spin_sq=cfg.sqd.get("spin_sq", 0.0),
            sbd_binary_path=cfg.sqd.get("sbd_binary_path", None),
            sbd_launcher=list(sbd_launcher) if sbd_launcher is not None else None,
            sbd_solver_kwargs=dict(sbd_solver_kwargs) if sbd_solver_kwargs is not None else None,
            sbd_timeout_s=cfg.sqd.get("sbd_timeout_s", 600),
            recover_iterations=cfg.sqd.get("recover_iterations", 5),
            recover_subspace_dim=self._recover_subspace_dim(cfg, max_dim),
            use_sparse_refine_hamiltonian=cfg.sqd.get("use_sparse_refine_hamiltonian", True),
            refine_method=cfg.sqd.get("refine_method", "gevp"),
            global_refine=cfg.sqd.get("global_refine", "bounded"),
            cumulative_max_dim=int(cfg.sqd.get("cumulative_max_dim", 5_000_000)),
            tp_ufd_n_alpha=cfg.sqd.get("tp_ufd_n_alpha", None),
            ufd_dense_limit=int(cfg.sqd.get("ufd_dense_limit", 20_000)),
            refine_input=cfg.sqd.get("refine_input", "oneshot"),
            max_depth_for_diag=cfg.sqd.get("max_depth_for_diag", None),
        )
        pipeline.dynamic_resources = bool(cfg.get("dynamic_resources", False))
        pipeline.start_dim = max_dim
        return pipeline

    #: shots is fixed, not predicted: a fixed shot count already lets the
    #: adaptive-doubling solve range up to ~1e8-scale subspaces (see
    #: SqdPipeline.solve_adaptive), so there's little to gain from also
    #: learning it -- one fewer thing to get wrong, dimension is the lever
    #: that actually matters here.
    FIXED_SHOTS = 10_000

    def _recover_subspace_dim(self, cfg, max_dim):
        """Alpha-determinant count for recover_and_diagonalize's GPU (sbd) step
        each recovery iteration -- RHF, so net CI dim = this^2. Explicit
        cfg.sqd.recover_subspace_dim wins; otherwise derive from max_dim
        (--max-dim's net-dim target) so recover_iterations>0 training actually
        uses the requested subspace size instead of sbd's own 512-det default
        (see SqdPipeline.recover_subspace_dim's docstring/comment)."""
        explicit = cfg.sqd.get("recover_subspace_dim", None)
        if explicit is not None:
            print(f"[factory] recover_subspace_dim={int(explicit)} (explicit, "
                  f"net CI dim target ~{int(explicit) ** 2}) -- max_dim={max_dim} unused here")
            return int(explicit)
        if max_dim is None:
            print("[factory] recover_subspace_dim=None (max_dim also None) -- "
                  "recover_and_diagonalize will default to samples_per_batch*num_batches")
            return None
        derived = max(1, int(round(max_dim ** 0.5)))
        print(f"[factory] recover_subspace_dim={derived} (derived from max_dim={max_dim}, "
              f"net CI dim target ~{derived ** 2})")
        return derived

    def _predict_resources(self, cfg, system, backend):
        from sqd_gqe.core.conditioning import ConditioningContext, ResourceHead

        ckpt = cfg.get("resource_head_checkpoint", None)
        if not ckpt or not Path(ckpt).exists():
            print(f"[dynamic_resources] no ResourceHead checkpoint at {ckpt!r}; "
                  f"falling back to sqd.max_dim from config, shots fixed at {self.FIXED_SHOTS}")
            return cfg.sqd.max_dim, self.FIXED_SHOTS
        head = ResourceHead.load(ckpt)
        cond = ConditioningContext.build(system, backend)
        max_dim, _predicted_shots = head.predict(cond.tensor())
        print(f"[dynamic_resources] predicted max_dim={max_dim}, shots fixed at {self.FIXED_SHOTS} "
              f"(ResourceHead's shots prediction {_predicted_shots} is unused -- see FIXED_SHOTS)")
        return max_dim, self.FIXED_SHOTS
