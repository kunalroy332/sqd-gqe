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

"""Point-and-run entry point: molecule + backend -> circuit + chemistry report.

Simple function/command interface (no heavy CLI framework):

    from sqd_gqe.run import run_molecule
    report = run_molecule("molecules/small/h2o.yaml", backend="ibm_heron_r2")

or from the shell:

    python -m sqd_gqe.run --molecule molecules/small/h2o.yaml --backend ibm_heron_r2

With ``--checkpoint`` the pre-tuned generalist weights are loaded and the
circuit is generated **zero-shot** (a forward pass, no retraining). Without a
checkpoint it runs a short on-the-fly adaptation so the tool still works before
the big training finishes.

The report bundles: total energy, error vs CASCI/CCSD, spin <S^2>, charge,
orbital occupations, excitation-rank distribution, and the generated circuit's
device-native depth / two-qubit-gate count on the chosen backend.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass
class ChemistryReport:
    molecule: str
    backend: str
    qubits: int
    n_electrons: int
    charge: int
    spin_2s: int
    # energies
    energy_sqd: float
    casci: float | None
    ccsd: float
    hf: float
    err_vs_casci_mha: float | None
    err_vs_ccsd_mha: float
    chemical_accuracy: bool | None
    # wavefunction / chemistry
    spin_squared: float | None
    orbital_occupations: list[float] | None
    excitation_distribution: dict | None
    # circuit
    circuit_depth: int | None
    circuit_two_qubit_gates: int | None
    backend_within_budget: bool | None
    n_gates_selected: int
    mode: str  # "zero-shot" or "adapted"
    recovery_iters: int | None = None
    # set when casci_energy() was infeasible (FCI dim too large): the run still
    # reports something meaningful via CCSD(T) + the SQD adaptive-solve energy
    # itself as a self-converged pseudo-exact estimate. A DMRG reference is
    # planned follow-up work for a tighter benchmark at this scale, not run here.
    reference_label: str = "CASCI (exact)"
    ccsd_t: float | None = None
    err_vs_ccsd_t_mha: float | None = None
    fci_dim: float | None = None
    dim_used: int | None = None
    shots_used: int | None = None
    # Circuits skipped before diagonalize()/recover() for exceeding max_depth_for_diag
    # (see pipeline.py's SqdPipeline.run) -- across every batch this call sampled, not
    # just the winning one. None when max_depth_for_diag was disabled (no skipping
    # happened, so this count is meaningless rather than genuinely zero).
    n_circuits_skipped_deep: int | None = None
    n_circuits_total: int | None = None
    skipped_depths: list[int] | None = None
    # Per-candidate recovery diagnostics for EVERY circuit in the winning batch (not just
    # the eventual winner) -- higher_rank_frac and still_improving_at_last_step let a
    # caller retroactively check whether a non-selected circuit might have found a
    # better energy with more --recover-max-iterations. Only populated when --recover
    # was used; None otherwise.
    recovery_candidates: list[dict] | None = None
    not_selected_but_promising: list[dict] | None = None
    # Where energy_sqd actually came from -- "circuit" (a single real circuit's seq,
    # whether from sampling or the recovery winner), "local_refined", or
    # "global_refined" (both merged from several circuits' subspaces, no single
    # circuit owns the result -- SqdSample.seq is None by construction for these).
    # circuit_depth/circuit_two_qubit_gates are only populated when this is "circuit";
    # best_circuit_depth/best_circuit_two_qubit_gates are the best individual circuit's
    # depth REGARDLESS of energy_source, as an always-available secondary number that
    # must not be read as "the depth of energy_sqd's circuit" when energy_source != "circuit".
    energy_source: str | None = None
    best_circuit_depth: int | None = None
    best_circuit_two_qubit_gates: int | None = None
    # Wall-time breakdown accumulated across every pipeline.run()/run_adaptive() call this
    # inference made (sample = policy rollout + ffsim/hardware sampling; diagonalize =
    # per-circuit diagonalize() inside pipeline.run(); refine_local/refine_global = the
    # cross-circuit/cross-iteration merge-and-rediagonalize steps), plus the separate
    # top-level --recover pass, which pipeline.timings never sees. Keys present only for
    # phases that actually ran (e.g. refine_* absent when refine_method wasn't used).
    timings_s: dict[str, float] | None = None
    # How many diagonalize()/recover_and_diagonalize() calls this inference run actually
    # made -- each is a real GPU/CPU solve, so this is the true cost driver behind
    # timings_s, not just num_samples (a config value, not a call count).
    n_diagonalize_calls: int | None = None
    n_recovery_calls: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def summary(self) -> str:
        L = [
            f"=== SQD-GQE chemistry report: {self.molecule} on {self.backend} ===",
            f"  qubits={self.qubits}  electrons={self.n_electrons}  charge={self.charge}  2S={self.spin_2s}  ({self.mode})",
            "",
            f"  Total energy (SQD-GQE) : {self.energy_sqd:.8f} Ha "
            f"(source: {self.energy_source or 'unknown'})",
        ]
        if self.casci is not None:
            L += [
                f"  CASCI (exact)          : {self.casci:.8f} Ha",
                f"  CCSD                   : {self.ccsd:.8f} Ha",
                f"  Hartree-Fock           : {self.hf:.8f} Ha",
                f"  Error vs CASCI         : {self.err_vs_casci_mha:+.4f} mHa "
                f"({'within' if self.chemical_accuracy else 'outside'} chemical accuracy)",
                f"  Error vs CCSD          : {self.err_vs_ccsd_mha:+.4f} mHa",
            ]
        else:
            L += [
                f"  CASCI (exact)          : N/A -- FCI dim {self.fci_dim:.3e} exceeds the "
                f"direct-solve ceiling (no exact reference at this scale)",
                f"  CCSD                   : {self.ccsd:.8f} Ha",
                f"  CCSD(T)  [reference]   : {self.ccsd_t:.8f} Ha",
                f"  Hartree-Fock           : {self.hf:.8f} Ha",
                f"  Error vs CCSD(T)       : {self.err_vs_ccsd_t_mha:+.4f} mHa "
                f"(CCSD(T) surrogate, not exact FCI -- DMRG reference planned as follow-up)",
            ]
        if self.dim_used is not None:
            L.append(f"  SQD subspace dim used  : {self.dim_used}  (shots={self.shots_used})")
        if self.spin_squared is not None:
            L.append(f"  <S^2>                  : {self.spin_squared:.4f}")
        if self.recovery_iters is not None:
            L.append(f"  Config-recovery iters  : {self.recovery_iters}")
        if self.excitation_distribution:
            e = self.excitation_distribution
            L.append(f"  Excitation dist (|c|^2): HF {e.get('HF',0):.3f}  S {e.get('S',0):.3f}  "
                     f"D {e.get('D',0):.3f}  T {e.get('T',0):.3f}  higher {e.get('higher',0):.3f}")
        if self.orbital_occupations is not None:
            occ = " ".join(f"{o:.2f}" for o in self.orbital_occupations)
            L.append(f"  Orbital occupations    : {occ}")
        if self.circuit_depth is not None:
            L.append(f"  Circuit ({self.backend}) : depth {self.circuit_depth}, "
                     f"2q-gates {self.circuit_two_qubit_gates}, "
                     f"{'fits' if self.backend_within_budget else 'EXCEEDS'} depth budget")
        elif self.best_circuit_depth is not None:
            L.append(f"  Best individual circuit (NOT the reported energy's source -- "
                     f"see energy_source above) : depth {self.best_circuit_depth}, "
                     f"2q-gates {self.best_circuit_two_qubit_gates}")
        if self.n_circuits_total is not None:
            L.append(f"  Circuits skipped (deep): {self.n_circuits_skipped_deep}/{self.n_circuits_total} "
                     f"(depths: {self.skipped_depths})")
        if self.not_selected_but_promising:
            L.append(f"  Non-winning circuits still improving w/ high higher-rank weight "
                     f"(may benefit from more --recover-max-iterations):")
            for d in self.not_selected_but_promising:
                L.append(f"    seq={d['seq']} recovered_energy={d['recovered_energy']:.6f} "
                         f"higher_rank_frac={d['higher_rank_frac']:.3f} "
                         f"n_recovery_steps={d['n_recovery_steps']}")
        if self.timings_s:
            parts = ", ".join(f"{k}={v:.2f}s" for k, v in self.timings_s.items())
            L.append(f"  Wall-time breakdown    : {parts}")
        if self.n_diagonalize_calls is not None or self.n_recovery_calls is not None:
            L.append(f"  Solve calls made       : diagonalize={self.n_diagonalize_calls}, "
                     f"recover_and_diagonalize={self.n_recovery_calls}")
        return "\n".join(L)


def _parse_session_max_time(value: str | None) -> int | str | None:
    """argparse always hands ``--ibm-session-max-time`` in as a plain string
    (no ``type=`` -- the value can be either an int-looking string or an
    ``hms`` string like ``"25m"``). ``qiskit_ibm_runtime.Session`` only
    special-cases a literal Python ``int``; anything else goes through its own
    ``hms_to_seconds``, whose regex requires a ``d``/``h``/``m``/``s`` suffix
    -- a bare digit string like ``"1500"`` has no suffix and raises there
    instead of being treated as seconds. Convert numeric strings to ``int``
    here so both '1500' and '25m' work from the CLI."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def run_molecule(molecule_path: str, backend: str = "ideal", checkpoint: str | None = None,
                 adapt_iters: int = 20, spin_sq: float | None = 0.0, seed: int = 0,
                 recover: bool = False, p_readout: float = 0.0,
                 dynamic_resources: bool = False, max_dim_cap: int = 1_000_000,
                 max_dim: int | None = None, diag_backend: str | None = None,
                 sbd_binary_path: str | None = None,
                 sbd_launcher: list[str] | None = None,
                 sbd_solver_kwargs: dict | None = None,
                 sampler_kind: str | None = None,
                 ibm_backend_name: str | None = None,
                 ibm_fake_backend: bool = False, shots: int | None = None,
                 ibm_save_dir: str | None = None, ibm_dynamical_decoupling: bool = False,
                 recover_max_iterations: int = 10, recover_subspace_dim: int | None = None,
                 sbd_timeout_s: float | None = None,
                 beta: float | None = None,
                 zeroshot_batches: int = 1,
                 ibm_session_mode: bool = False,
                 ibm_session_max_time: int | str | None = None,
                 max_depth_for_diag: int | None | bool = False,
                 refine_method: str | None = None) -> ChemistryReport:
    """Thin wrapper around ``_run_molecule_impl`` that guarantees the pipeline's
    sampler is closed (releasing any open IBM Runtime Session) even if the
    real work below raises -- without this, ``--ibm-session-mode`` would leak
    an open Session on any crash/interrupt, exactly the kind of thing
    ``sqd_gqe/core/training.py``'s Lightning ``teardown`` hook guards against
    for the training path; this module has no such lifecycle hook, so it
    needs its own explicit guarantee."""
    factory_holder: dict = {}
    try:
        return _run_molecule_impl(
            molecule_path, backend=backend, checkpoint=checkpoint, adapt_iters=adapt_iters,
            spin_sq=spin_sq, seed=seed, recover=recover, p_readout=p_readout,
            dynamic_resources=dynamic_resources, max_dim_cap=max_dim_cap, max_dim=max_dim,
            diag_backend=diag_backend, sbd_binary_path=sbd_binary_path,
            sbd_launcher=sbd_launcher, sbd_solver_kwargs=sbd_solver_kwargs, sampler_kind=sampler_kind,
            ibm_backend_name=ibm_backend_name, ibm_fake_backend=ibm_fake_backend, shots=shots,
            ibm_save_dir=ibm_save_dir, ibm_dynamical_decoupling=ibm_dynamical_decoupling,
            recover_max_iterations=recover_max_iterations, recover_subspace_dim=recover_subspace_dim,
            sbd_timeout_s=sbd_timeout_s, ibm_session_mode=ibm_session_mode,
            ibm_session_max_time=ibm_session_max_time, beta=beta,
            zeroshot_batches=zeroshot_batches, max_depth_for_diag=max_depth_for_diag,
            refine_method=refine_method, _factory_holder=factory_holder,
        )
    finally:
        pipeline = factory_holder.get("pipeline")
        if pipeline is not None:
            pipeline.close()


def _run_molecule_impl(molecule_path: str, backend: str = "ideal", checkpoint: str | None = None,
                 adapt_iters: int = 20, spin_sq: float | None = 0.0, seed: int = 0,
                 recover: bool = False, p_readout: float = 0.0,
                 dynamic_resources: bool = False, max_dim_cap: int = 1_000_000,
                 max_dim: int | None = None, diag_backend: str | None = None,
                 sbd_binary_path: str | None = None,
                 sbd_launcher: list[str] | None = None,
                 sbd_solver_kwargs: dict | None = None,
                 sampler_kind: str | None = None,
                 ibm_backend_name: str | None = None,
                 ibm_fake_backend: bool = False, shots: int | None = None,
                 ibm_save_dir: str | None = None, ibm_dynamical_decoupling: bool = False,
                 recover_max_iterations: int = 10, recover_subspace_dim: int | None = None,
                 sbd_timeout_s: float | None = None,
                 ibm_session_mode: bool = False,
                 ibm_session_max_time: int | str | None = None,
                 beta: float | None = None,
                 zeroshot_batches: int = 1,
                 max_depth_for_diag: int | None | bool = False,
                 refine_method: str | None = None,
                 ngates: int | None = None,
                 encoder_override: str | None = None,
                 stop_after_build: bool = False,
                 sequences_only: bool = False,
                 _factory_holder: dict | None = None) -> ChemistryReport:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.core.system import FciIntractableError
    from sqd_gqe.gqe import circuit_metrics as cm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hw = get_backend(backend)

    # If a checkpoint is given, it was trained with a fixed max-vocabulary and
    # the conditioned policy; match both so the weights load and the run is
    # truly zero-shot. Infer the checkpoint's vocab from its embedding shape.
    ckpt_state = None
    overrides = []
    # A missing checkpoint file used to fall through this branch silently, leaving a
    # randomly initialised, unconditioned policy -- and the run then reported an
    # energy as if it had evaluated the trained model. That is how condition E's A5
    # produced "+1.400 mHa, a 10.6x improvement" from three separate training seeds
    # whose training had died at exit=1 before a single iteration: the three runs
    # were all measuring the same untrained initialisation, which is why they agreed
    # to four decimals. Fail instead: a caller that names a checkpoint means to use
    # it, and a zero-shot-from-scratch run is requested by omitting the flag.
    if checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"--checkpoint {checkpoint} does not exist. Refusing to evaluate an "
            "untrained policy under a checkpoint's name -- omit --checkpoint if an "
            "untrained run is what you want."
        )
    if checkpoint:
        ckpt_state = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ckpt_state.get("state_dict", ckpt_state)
        wte = sd.get("transformer.wte.weight")
        ckpt_vocab = int(wte.shape[0]) if wte is not None else None
        overrides = ["model=conditioned"]
        if ckpt_vocab:
            overrides.append(f"+operator_pool.max_vocab={ckpt_vocab}")

    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=overrides)
    if beta is not None:
        # Sampling temperature. act() draws from Categorical(logits=-beta*logits),
        # so beta multiplies a logit range measured at ~18 on a trained c2h4
        # checkpoint -- at the config default (VarianceBeta initial=0.5) the batch
        # of cfg.trainer.num_samples circuits collapses to ~4 distinct sequences
        # differing in a single token, and at 1.0 to exactly one (effective
        # vocabulary 1.00). That defeats the "generate N candidates, keep the
        # lowest energy" procedure this script implements above. Measured: 0.2
        # gives 10/10 distinct with effective vocabulary 13.4. Overriding both
        # bounds pins the schedule, since schedule.step() would otherwise move it.
        from omegaconf import open_dict as _open_dict
        with _open_dict(cfg):
            cfg.trainer.beta_schedule._target_ = "sqd_gqe.core.schedule.LinearBeta"
            cfg.trainer.beta_schedule.start = float(beta)
            cfg.trainer.beta_schedule.delta = 0.0
            for k in ("initial", "target_var", "beta_max"):
                cfg.trainer.beta_schedule.pop(k, None)
        print(f"[run] beta pinned to {beta} (LinearBeta, delta=0)", file=sys.stderr, flush=True)
    cfg.molecule = OmegaConf.load(molecule_path)
    cfg.sqd.spin_sq = spin_sq
    cfg.dynamic_resources = dynamic_resources
    if ngates is not None:
        # Must match the checkpoint's REAL training ngates (n_positions =
        # ngates+1 sizes the transformer's position-embedding table --
        # factory.build_policy reads cfg.ngates directly, see
        # SqdFactory.build_policy). Silently defaulting to the config's
        # ngates=10 for a checkpoint trained at a different value would load
        # weights into a policy shaped for the wrong number of decoding
        # steps -- verify against the checkpoint's own transformer.wpe.weight
        # shape (n_positions) before relying on this, don't just assume the
        # training script's --ngates flag took effect.
        cfg.ngates = ngates
    if max_dim is not None:
        # fixed-dim override for a manual SQD-subspace-size sweep at
        # inference, distinct from --dynamic-resources' adaptive doubling
        # (see amortized_train.py's --max-dim for the training-side analog)
        cfg.sqd.max_dim = max_dim
    if diag_backend is not None:
        cfg.sqd.diag_backend = diag_backend
    if sbd_binary_path is not None:
        cfg.sqd.sbd_binary_path = sbd_binary_path
    if sbd_launcher is not None:
        cfg.sqd.sbd_launcher = list(sbd_launcher)
    if sbd_solver_kwargs is not None:
        cfg.sqd.sbd_solver_kwargs = dict(sbd_solver_kwargs)
    if sbd_timeout_s is not None:
        cfg.sqd.sbd_timeout_s = sbd_timeout_s
    if refine_method is not None:
        cfg.sqd.refine_method = refine_method
    # max_depth_for_diag is resolved once system.norb is known (below, right
    # before build_pipeline) -- see that block's comment for what it defaults to.
    _max_depth_for_diag_arg = max_depth_for_diag
    # configs_sqd/default.yaml's sqd.recover_iterations=5 default is for
    # amortized_train.py's per-sample reward shaping ONLY -- it silently
    # applies to every SqdPipeline built by SqdFactory, including this one,
    # and its hardcoded samples_per_batch=512 subspace completely bypasses
    # --recover-subspace-dim/--recover-max-iterations below (confirmed: a
    # real run submitted 10 unwanted 5-step/512-det recovery calls, one per
    # sampled circuit, before ever reaching the explicit --recover block --
    # burned real IBM Runtime hardware time on job 63162 before being caught
    # and killed). Force it off here; --recover is the only recovery path
    # this module should ever trigger.
    cfg.sqd.recover_iterations = 0
    if sampler_kind is not None:
        cfg.sampler.kind = sampler_kind
    if ibm_backend_name is not None:
        cfg.sampler.ibm_backend_name = ibm_backend_name
    cfg.sampler.ibm_fake_backend = bool(ibm_fake_backend)
    if shots is not None:
        cfg.sampler.shots = shots
    if ibm_save_dir is not None:
        cfg.sampler.ibm_save_dir = ibm_save_dir
    cfg.sampler.ibm_dynamical_decoupling = bool(ibm_dynamical_decoupling)
    cfg.sampler.ibm_session_mode = bool(ibm_session_mode)
    if ibm_session_max_time is not None:
        cfg.sampler.ibm_session_max_time = ibm_session_max_time

    # Seed the global RNG. policy.act() ends in Categorical(...).sample(), which
    # takes no generator argument and therefore draws from the global stream, so
    # without this a zero-shot evaluation is not reproducible: the trained policy is
    # deterministic, but which circuits get sampled from it is not.
    #
    # This was not merely theoretical. Two runs of the identical checkpoints gave
    # C2H4 zero-shot errors of 12.667 and 107.383 mHa for one arm, and 0.423 vs
    # 24.346 for another, while their training histories matched to every logged
    # digit -- the spread was the evaluation, not the encoder. The `seed` argument
    # already existed but only reached configuration recovery and readout noise.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[run] seed={seed} (policy sampling + recovery + readout)",
          file=sys.stderr, flush=True)

    print("[run] building system...", file=sys.stderr, flush=True)
    factory = SqdFactory()
    system = factory.build_system(cfg)
    print(f"[run] system built: norb={system.norb} nelec={system.nelec}", file=sys.stderr, flush=True)
    # Which conditioning arm the checkpoint was trained with. Read from the
    # checkpoint rather than asked for on the command line: a cross-attention
    # checkpoint loaded into the additive policy would transfer only the
    # shape-matching keys and then run with a randomly initialised conditioning
    # path, silently reporting a "zero-shot" number for a policy that never saw
    # the molecule's descriptor. amortized_train writes "encoder" and
    # "prompt_encoder_state_dict" into every checkpoint (commit 21d71d2) for this.
    # encoder_override forces which conditioning arm to BUILD (random init,
    # since no checkpoint weights get loaded below when checkpoint is None) --
    # accepts the same arm aliases amortized_train.py's --encoder does
    # (e.g. "a1_xattn_large_bounded"), canonicalized identically via
    # canonical_arm/build_prompt_encoder below. Use this for an untrained-
    # policy control run where the architecture still needs to match a real
    # arm (a1/a5/a6), not the "add" no-conditioning default a missing
    # checkpoint would otherwise fall back to.
    if encoder_override is not None:
        # amortized_train.py canonicalizes --encoder (args.encoder =
        # canonical_arm(args.encoder), its own line ~653) BEFORE saving it
        # into the checkpoint's "encoder" field -- so a real checkpoint's
        # ckpt_encoder is always the canonical module name (e.g.
        # "scalar27_large_bounded"), never the raw arm alias
        # ("a1_xattn_large_bounded"). Canonicalize here too so the branches
        # below (which switch on the canonical name) behave identically
        # whether ckpt_encoder came from a real checkpoint or this override.
        from sqd_gqe.core.prompt_encoders import canonical_arm
        ckpt_encoder = canonical_arm(encoder_override)
    else:
        ckpt_encoder = (ckpt_state or {}).get("encoder", "add") if ckpt_state else "add"
    if ckpt_encoder != "add":
        # Swap in the cross-attention policy by pointing cfg.model at
        # configs_sqd/model/crossattn.yaml, rather than constructing the class here.
        # One definition of the policy, so a hyperparameter set in that yaml applies
        # to training and to zero-shot alike.
        from omegaconf import OmegaConf as _OC
        cfg.model = _OC.load(Path(__file__).resolve().parent.parent
                             / "configs_sqd" / "model" / "crossattn.yaml")
        print(f"[run] cross-attention policy (model=crossattn) for "
              f"encoder={ckpt_encoder}", file=sys.stderr, flush=True)
    policy = factory.build_policy(cfg).to(device)
    schedule = factory.build_beta_schedule(cfg)
    # False (the CLI/run_molecule default, distinct from an explicit None) means
    # "not passed" -- reuse amortized_train.py's depth_ceiling_for_qubits(), the
    # same real, training-calibrated per-qubit-count depth gate that decides
    # whether a TRAINING iteration's gradient step happens at all (<=40q: 300,
    # <=60q: 500, <=80q: 800) -- rather than a separate, inference-only number,
    # so "too deep to bother with" means the same thing in both places. Pass
    # max_depth_for_diag=None explicitly to diagonalize every circuit regardless
    # of depth (the behavior before this flag existed); pass an int to override.
    if _max_depth_for_diag_arg is False:
        from benchmark.amortized_train import depth_ceiling_for_qubits
        cfg.sqd.max_depth_for_diag = depth_ceiling_for_qubits(system.norb * 2)
    else:
        cfg.sqd.max_depth_for_diag = _max_depth_for_diag_arg
    print(f"[run] building pipeline (sampler.kind={cfg.sampler.get('kind', 'ffsim')}, "
          f"ibm_backend_name={cfg.sampler.get('ibm_backend_name', None)}, "
          f"max_depth_for_diag={cfg.sqd.max_depth_for_diag})...",
          file=sys.stderr, flush=True)
    pipeline = factory.build_pipeline(cfg, backend=hw)
    if _factory_holder is not None:
        _factory_holder["pipeline"] = pipeline
        _factory_holder["policy"] = policy
    print("[run] pipeline built", file=sys.stderr, flush=True)
    if stop_after_build:
        # Caller wants a real pipeline (Hamiltonian/nelec/norb/recoverer, all
        # built above) with NO sampler call -- building FfsimSampler/
        # IbmHardwareSampler is cheap, but ACTUALLY sampling with ffsim is
        # exactly what OOMs at norb>=23 (dense statevector,
        # C(norb,na)*C(norb,nb) elements -- 26.6 TiB at norb=23). Used by
        # cache-replay inference (re-diagonalizing previously-sampled, saved
        # bitstring counts at a new subspace_dim) which needs the pipeline's
        # recoverer but must never trigger a fresh sample, hardware or ffsim.
        print("[run] stop_after_build=True -- returning before any sampler.run() call",
              file=sys.stderr, flush=True)
        return None
    pool = factory.build_operator_pool(cfg)
    print(f"[run] operator pool built: {len(pool)} ops", file=sys.stderr, flush=True)

    fci_dim = float(system.fci_dimension())
    ccsd = float(system.ccsd_energy())
    hf = float(system.hf.e_tot)
    print(f"[run] hf={hf:.6f} ccsd={ccsd:.6f} fci_dim={fci_dim:.3g}", file=sys.stderr, flush=True)
    try:
        casci = float(system.casci_energy())
        ccsd_t = None
        print(f"[run] casci={casci:.6f}", file=sys.stderr, flush=True)
    except FciIntractableError:
        casci = None
        ccsd_t = float(system.ccsd_t_energy())
        print(f"[run] casci intractable, ccsd_t={ccsd_t:.6f}", file=sys.stderr, flush=True)

    cond = ConditioningContext.build(system, hw)
    conditioned = hasattr(policy, "set_condition")

    # Build the prompt encoder for the cross-attention arms and apply the prompt.
    # This is the point of the whole zero-shot test: the encoder maps THIS
    # molecule's descriptor to a prompt, so a policy trained on smaller molecules
    # can be conditioned on an unseen one without retraining.
    prompt_encoder = None
    prompt_input = None

    def apply_conditioning():
        if prompt_encoder is not None:
            policy.set_prompt(prompt_encoder() if prompt_input is None
                              else prompt_encoder(prompt_input))
        elif conditioned:
            policy.set_condition(cond.tensor(device))

    if ckpt_encoder != "add":
        from sqd_gqe.core.prompt_encoders import (
            MaceContext, OrbitalSetContext, PosEgnnContext, PromptContext, build_prompt_encoder)
        prompt_encoder = build_prompt_encoder(ckpt_encoder, policy.config.n_embd).to(device)
        if ckpt_encoder in ("posegnn", "posegnn_large", "posegnn_large_bounded"):
            prompt_input = PosEgnnContext.build(system, hw).tensors(device)
        elif ckpt_encoder in ("mace", "mace_large", "mace_large_bounded"):
            prompt_input = MaceContext.build(system, hw).tensors(device)
        elif ckpt_encoder in ("orbital_set", "orbital_set_large", "orbital_set_xl"):
            prompt_input = OrbitalSetContext.build(system, hw).tensor(device)
        elif ckpt_encoder == "scalar27_epsfix":
            prompt_input = PromptContext.build(system, hw, epsfix=True).tensor(device)
        elif ckpt_encoder in ("scalar27", "scalar27_norm", "scalar27_large", "scalar27_large_bounded"):
            # scalar27_norm (A1'') consumes the SAME 27-dim descriptor; its LayerNorm
            # lives inside the encoder module, which build_prompt_encoder restores
            # from the checkpoint, so nothing changes on the feature side here.
            # scalar27_large_bounded is the same again -- only the encoder's output
            # activation differs (see ScalarPromptEncoder.bound_output).
            prompt_input = PromptContext.build(system, hw).tensor(device)
        # 'null' takes no input at all.
    apply_conditioning()

    # encoder_override implies an untrained-policy CONTROL run (random init,
    # architecture forced to match a real arm) -- this must sample like every
    # other zero-shot job (one batch, zero gradient steps), not fall into the
    # "adapted" path's default adapt_iters-many gradient-adaptation loop that
    # a genuinely no-checkpoint run would otherwise take (see the `mode !=
    # "zero-shot"` gated block below, and n_iters' derivation).
    mode = "zero-shot" if encoder_override is not None else "adapted"
    if ckpt_state is not None:
        sd = ckpt_state.get("state_dict", ckpt_state)
        # strict=False only tolerates missing/unexpected KEYS, not shape
        # mismatches on keys present in both -- a target molecule whose
        # operator-pool/vocab or FEATURE_DIM differs from what the checkpoint
        # was trained with (e.g. pyridine's 65-op pool vs a 48-vocab
        # checkpoint, or FEATURE_DIM growing from 20->26) raises regardless of
        # strict. Filter to shape-matching params first, same as
        # amortized_train.py's warm-start transfer.
        cur = policy.state_dict()
        transfer = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k in sd if k not in transfer]
        policy.load_state_dict(transfer, strict=False)
        if skipped:
            print(f"zero-shot load: transferred {len(transfer)}, skipped {len(skipped)} "
                  f"shape-mismatched (e.g. resized embedding/head): {skipped}")
        # Load the prompt encoder's weights too. Without this the decoder would be
        # trained but the encoder random -- the prompt would carry no learned
        # mapping, so the reported zero-shot number would not reflect the trained
        # conditioning at all.
        if prompt_encoder is not None:
            pe_sd = ckpt_state.get("prompt_encoder_state_dict")
            if pe_sd is None:
                raise ValueError(
                    f"checkpoint says encoder={ckpt_encoder!r} but carries no "
                    "prompt_encoder_state_dict; it predates commit 21d71d2 and "
                    "cannot be evaluated zero-shot (the encoder weights are lost)."
                )
            prompt_encoder.load_state_dict(pe_sd)
            print(f"[run] prompt encoder loaded ({len(pe_sd)} tensors)",
                  file=sys.stderr, flush=True)
        mode = "zero-shot"
        apply_conditioning()
    print(f"[run] mode={mode}", file=sys.stderr, flush=True)

    # generate + score circuits; adapt briefly if no checkpoint
    objective = factory.build_objective(cfg)
    # Include the prompt encoder's parameters. Only matters on the adaptation path
    # (no checkpoint): zero-shot runs one iteration with no gradient step. Omitting
    # them there would adapt the decoder against a frozen prompt.
    _adapt_params = list(policy.parameters())
    if prompt_encoder is not None:
        _adapt_params += list(prompt_encoder.parameters())
    opt = torch.optim.AdamW(_adapt_params, lr=cfg.trainer.optimizer.lr)
    # Zero-shot takes no gradient steps, so extra iterations are extra independent
    # batches of candidate circuits, and the best is kept. One batch of
    # cfg.trainer.num_samples circuits is a single draw, which is what made the
    # numbers above unusable. The paper does the same thing: "each model generates 16
    # candidate circuits for each conformer in the test set, and the circuit with the
    # lowest energy for each test-set conformer is used in the reported metrics"
    # (Sec. 4). Sampling once was our deviation from that, not a choice.
    n_iters = max(1, zeroshot_batches) if mode == "zero-shot" else adapt_iters
    best_e, best_seq, best_report = float("inf"), None, None
    # Accumulated across every batch (not just the winning one) -- see SqdSample.
    # diagonalize_skipped_deep on pipeline.py's per-circuit run() loop.
    skipped_depths_all: list[int] = []
    n_circuits_total = 0
    timings_s: dict[str, float] = {}
    n_diagonalize_calls = 0
    for it in range(n_iters):
        print(f"[run] iter {it+1}/{n_iters}: sampling {cfg.trainer.num_samples} circuits "
              f"({cfg.ngates} gates each) from the policy...", file=sys.stderr, flush=True)
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            if sequences_only:
                # Caller wants only the gate sequences the policy draws (e.g. to measure them later on hardware in a separate step); return before ANY
                # sampler call, simulator or hardware. The sequences keep the leading start token, exactly as SqdSample.seq does.
                if _factory_holder is not None:
                    _factory_holder["sequences"] = state["idx"].tolist()
                print(f"[run] sequences_only=True -- drew {state['idx'].shape[0]} sequences, returning before any sampler call",
                      file=sys.stderr, flush=True)
                return None
            submit_msg = "run_adaptive" if dynamic_resources else "run"
            print(f"[run] iter {it+1}/{n_iters}: circuits sampled, submitting to "
                  f"sampler.{submit_msg} (real hardware if configured)...",
                  file=sys.stderr, flush=True)
            if dynamic_resources:
                report = pipeline.run_adaptive(state, start_dim=pipeline.start_dim, max_dim_cap=max_dim_cap)
            else:
                report = pipeline.run(state)
            print(f"[run] iter {it+1}/{n_iters}: sampler+diagonalization returned "
                  f"({len(report.energies)} energies, min={min(report.energies):.6f})",
                  file=sys.stderr, flush=True)
            n_circuits_total += len(report.samples)
            # pipeline.timings resets at the top of each run()/run_adaptive() call (it only
            # ever reflects the batch that just ran), so accumulate it here across every
            # batch this inference made -- otherwise only the last batch's breakdown would
            # survive to the final report.
            for k, v in getattr(pipeline, "timings", {}).items():
                timings_s[k] = timings_s.get(k, 0.0) + v
            # One diagonalize() call per circuit in the batch, plus one more for local
            # refine and one more for global refine when refine_method merged this batch
            # (both re-diagonalize a merged subspace -- see pipeline.py's run()/run_adaptive()).
            n_diagonalize_calls += len(report.samples)
            if report.local_refined is not None:
                n_diagonalize_calls += 1
            if report.global_refined is not None:
                n_diagonalize_calls += 1
            batch_skipped = [s.depth for s in report.samples if s.diagonalize_skipped_deep]
            if batch_skipped:
                skipped_depths_all.extend(batch_skipped)
                print(f"[run] iter {it+1}/{n_iters}: skipped diagonalize() for "
                      f"{len(batch_skipped)}/{len(report.samples)} circuits exceeding "
                      f"max_depth_for_diag (depths: {batch_skipped})",
                      file=sys.stderr, flush=True)
            energies = torch.tensor(report.energies, device=device)
            old_lp = policy.log_prob(state["idx"], schedule.beta())
        i = int(np.argmin(report.energies))
        if report.energies[i] < best_e:
            best_e, best_seq, best_report = report.energies[i], state["idx"][i].tolist(), report
            print(f"[run] iter {it+1}/{n_iters}: new best energy={best_e:.6f}", file=sys.stderr, flush=True)
        if mode != "zero-shot":
            for _ in range(cfg.trainer.step_per_epoch):
                # The cross-attention prompt is consumed by backward(), so rebuild
                # it before every forward that needs a graph.
                apply_conditioning()
                lp = policy.log_prob(state["idx"], schedule.beta())
                ctx = {"old_log_probs": old_lp, "energies": energies, "gate_seqs": state["idx"][:, 1:]}
                loss = objective(lp, ctx)
                opt.zero_grad(); loss.backward(); opt.step()
            schedule.step(energies=energies)
    print(f"[run] sampling loop done, best_e={best_e:.6f}", file=sys.stderr, flush=True)
    if _factory_holder is not None:
        # best_report is the internal SqdReport (per-circuit .samples, each with its
        # own raw sampled .counts) -- the returned ChemistryReport has neither, so a
        # caller that needs the raw per-circuit samples (e.g. to drive its own
        # recover_and_diagonalize calls directly, extending recovery past what this
        # function's own --recover pass does) has no other way to reach them.
        _factory_holder["best_report"] = best_report

    # global-refined energy if available (usually the best) -- track WHICH candidate
    # actually wins each min() so the final report can say where `energy` came from,
    # instead of silently reporting a refined (no-single-circuit) energy next to a
    # specific circuit's depth as if they were the same thing. global_refined/
    # local_refined are constructed with seq=None (SubspaceMerger merges several
    # circuits' subspaces, so no single circuit "owns" the result) -- confirmed via
    # SqdSample's fields; there is no seq to report for these sources, by construction,
    # not an oversight to patch here.
    energy = best_e
    energy_source = "circuit" if best_seq is not None else "none"
    if best_report is not None and best_report.local_refined is not None \
            and best_report.local_refined.energy < energy:
        energy = best_report.local_refined.energy
        energy_source = "local_refined"
    if best_report is not None and best_report.global_refined is not None \
            and best_report.global_refined.energy < energy:
        energy = best_report.global_refined.energy
        energy_source = "global_refined"

    # optional: configuration-recovery loop, tried on EVERY sampled circuit in the
    # winning batch (not just the pre-recovery best) -- confirmed this session that
    # pre-recovery energy is a poor proxy for post-recovery quality when raw sample
    # validity is sparse (real 46q data: 6/10 circuits had ZERO electron-count-valid
    # raw samples pre-recovery, yet all 6 recovered to within ~2 mHa of the deep
    # circuit that pre-recovery selection actually picked -- see
    # docs/2026-08-20/presentation_detailed_2026-08-20.tex). Recovery itself already
    # uses the complete raw counts (valid + invalid) as its starting point
    # (backend_sbd_gpu.py's recover_and_diagonalize builds bitstring_matrix/avg_occ
    # from ALL of `counts`, no select_by_nelec filter) -- the bug was purely in this
    # selection logic sitting in front of it.
    recovery_iters = None
    all_candidates_diag = None
    not_selected_but_promising = None
    n_recovery_calls = 0
    if recover and best_report is not None:
        recoverer = pipeline._sbd if (pipeline.diag_backend == "sbd" and pipeline._sbd is not None) else pipeline._sqd
        recover_kwargs = dict(samples_per_batch=512, num_batches=3,
                              max_iterations=recover_max_iterations, seed=seed)
        # Effective subspace_dim: explicit --recover-subspace-dim wins; otherwise
        # derive sqrt(max_dim) the same way SqdFactory._recover_subspace_dim does
        # for the pipeline's own automatic per-circuit recovery, so this
        # standalone --recover pass actually respects --max-dim too instead of
        # silently falling back to samples_per_batch*num_batches (1536) regardless
        # of what --max-dim was set to.
        _effective_subspace_dim = recover_subspace_dim
        if _effective_subspace_dim is None and max_dim is not None:
            _effective_subspace_dim = max(1, round(max_dim ** 0.5))
        if _effective_subspace_dim is not None and recoverer is pipeline._sbd:
            # alpha-determinant count (RHF: net CI dim = subspace_dim^2), overrides the
            # samples_per_batch*num_batches default -- e.g. 1000 -> net dim ~1e6. GPU
            # (sbd) only -- SqdDiagonalizer.recover_and_diagonalize (CPU) has no such param.
            recover_kwargs["subspace_dim"] = _effective_subspace_dim
            print(f"[run] --recover subspace_dim={_effective_subspace_dim} "
                  f"(net CI dim target ~{_effective_subspace_dim ** 2}, "
                  f"{'explicit --recover-subspace-dim' if recover_subspace_dim is not None else f'derived from --max-dim={max_dim}'})",
                  file=sys.stderr, flush=True)
        elif recoverer is pipeline._sbd:
            print(f"[run] --recover subspace_dim NOT SET (max_dim={max_dim}) -- "
                  f"recover_and_diagonalize will default to "
                  f"samples_per_batch({recover_kwargs['samples_per_batch']})*"
                  f"num_batches({recover_kwargs['num_batches']})="
                  f"{recover_kwargs['samples_per_batch'] * recover_kwargs['num_batches']}",
                  file=sys.stderr, flush=True)

        candidates = [s for s in best_report.samples if s.counts is not None]
        if not candidates and best_seq is not None:
            # fallback: no counts retained on this report (e.g. p_readout's noisy-sim
            # path, or an older report) -- resample the single pre-recovery-best
            # circuit, matching this function's original (pre-fix) behavior.
            print("[run] recovery: no per-circuit counts retained -- resampling the "
                  "pre-recovery-best circuit only (real hardware if configured)...",
                  file=sys.stderr, flush=True)
            sampler = pipeline.sampler
            if p_readout and p_readout > 0:
                from sqd_gqe.gqe.noisy_sampler import NoisyFfsimSampler
                sampler = NoisyFfsimSampler(pool, shots_count=cfg.sampler.shots,
                                            p_readout=p_readout, seed=seed)
            resampled = sampler.sample_state(best_seq)
            from sqd_gqe.core.report import SqdSample
            candidates = [SqdSample(seq=tuple(best_seq), energy=best_e, n_sampled=len(resampled),
                                    n_symmetry_valid=None, subspace_size=0, cx_count=None,
                                    total_gates=None, counts=resampled)]

        print(f"[run] recovery: attempting recovery on {len(candidates)} sampled circuits "
              f"(backend={'sbd/gpu' if recoverer is pipeline._sbd else 'sqd/cpu'})...",
              file=sys.stderr, flush=True)
        recovered = []  # list of (seq, recovered_energy, history_len)
        # Per-circuit diagnostics for EVERY candidate, not just the eventual winner --
        # this is what lets the caller retroactively ask "could a non-selected circuit
        # have done better with more recovery steps?" (see not_selected_but_promising
        # below) instead of silently discarding every candidate but the winner, which
        # is what this loop did before.
        all_candidates_diag = []
        for i, sample in enumerate(candidates):
            n_distinct = len(sample.counts)
            n_total = sum(sample.counts.values())
            # diagnostics_out is sbd/GPU-only -- backend_sqd.py's (CPU) recover_and_diagonalize
            # has no such parameter, so only pass it on the backend that supports it. On CPU,
            # higher_rank_frac/still_improving_at_last_step below simply stay None for every
            # candidate (no per-step excitation data available), rather than crashing.
            step_diag: list = []
            extra_kwargs = {"diagnostics_out": step_diag} if recoverer is pipeline._sbd else {}
            # This top-level --recover pass is a separate code path from pipeline.run()'s
            # own automatic per-circuit recovery -- pipeline.timings never sees it, so time
            # it here explicitly to get a real recovery-vs-diagonalize wall-time split.
            _t0 = time.perf_counter()
            rec_e, _, rec_hist = recoverer.recover_and_diagonalize(
                sample.counts, **extra_kwargs, **recover_kwargs)
            timings_s["recover"] = timings_s.get("recover", 0.0) + (time.perf_counter() - _t0)
            n_recovery_calls += 1
            print(f"[run] recovery: circuit {i+1}/{len(candidates)} "
                  f"({n_distinct} distinct bitstrings, {n_total} total shots): "
                  f"recovered energy={rec_e:.6f} ({len(rec_hist)} iterations)",
                  file=sys.stderr, flush=True)
            recovered.append((sample.seq, rec_e, len(rec_hist)))
            # step_diag[0] is the pre-recovery step (energy=None); real per-step
            # energies/excitation histograms start at step_diag[1].
            steps_with_energy = [d for d in step_diag if d.get("energy") is not None]
            last_hist = steps_with_energy[-1]["excitation_hist"] if steps_with_energy else None
            higher_frac = None
            if last_hist and last_hist.get("combined"):
                higher_frac = last_hist["higher"] / last_hist["combined"]
            # "Still improving at the last step" -- compare the last two accepted
            # energies (finite, non-discontinuity-rejected steps only; rec_hist itself
            # already excludes rejected/skipped steps per recover_and_diagonalize's own
            # history.append(float("inf")) convention for those).
            finite_hist = [e for e in rec_hist if math.isfinite(e)]
            still_improving = (
                len(finite_hist) >= 2
                and (finite_hist[-2] - finite_hist[-1]) > 1e-4  # > 0.1 mHa improvement on the last step
            )
            all_candidates_diag.append({
                "seq": list(sample.seq), "recovered_energy": rec_e, "n_recovery_steps": len(rec_hist),
                "higher_rank_frac": higher_frac, "still_improving_at_last_step": still_improving,
            })

        best_post_recovery_e = min(rec_e for _, rec_e, _ in recovered)
        # Prefer the shallowest circuit among those within 5 mHa of the best
        # post-recovery result -- confirmed this session that circuits differing by
        # >100x in depth can land within ~2 mHa of each other after recovery, so
        # depth should break ties among comparably-accurate candidates rather than
        # being ignored.
        TIE_TOLERANCE_HA = 5e-3
        tied = [(seq, rec_e, n_iters) for seq, rec_e, n_iters in recovered
                if rec_e - best_post_recovery_e <= TIE_TOLERANCE_HA]
        if len(tied) > 1:
            depths = []
            for seq, rec_e, n_iters in tied:
                try:
                    d = cm.circuit_depth_metrics(pool, list(seq))["depth"]
                except Exception:
                    d = float("inf")
                depths.append((d, seq, rec_e, n_iters))
            depths.sort(key=lambda t: t[0])
            _winner_depth, winner_seq, winner_e, winner_iters = depths[0]
            print(f"[run] recovery: {len(tied)} circuits tied within {TIE_TOLERANCE_HA*1e3:.1f} mHa "
                  f"of the best ({best_post_recovery_e:.6f}) -- picked the shallowest "
                  f"(depth={_winner_depth})", file=sys.stderr, flush=True)
        else:
            winner_seq, winner_e, winner_iters = tied[0]

        recovery_iters = winner_iters
        best_seq = list(winner_seq)
        print(f"[run] recovery: winner energy={winner_e:.6f}", file=sys.stderr, flush=True)
        if winner_e < energy:
            energy = winner_e
            energy_source = "circuit"  # best_seq was just reassigned to this winner above

        # Retroactive check: a NON-winning circuit whose recovery was still improving at
        # its last step (hadn't plateaued) AND still has high excitation-rank>=4 weight
        # is flagged -- more recovery iterations might have found a better energy than
        # the circuit actually reported. This does not rerun anything; it only surfaces
        # candidates worth a manual --recover-max-iterations rerun, since the diagnostics
        # needed are already computed above (all_candidates_diag), just discarded before
        # this change.
        HIGH_HIGHER_RANK_THRESHOLD = 0.5
        not_selected_but_promising = [
            d for d in all_candidates_diag
            if tuple(d["seq"]) != tuple(winner_seq)
            and d["still_improving_at_last_step"]
            and d["higher_rank_frac"] is not None
            and d["higher_rank_frac"] > HIGH_HIGHER_RANK_THRESHOLD
        ]
        if not_selected_but_promising:
            print(f"[run] recovery: {len(not_selected_but_promising)} non-winning circuit(s) "
                  "were still improving at their last recovery step with high higher-rank "
                  "excitation weight -- more --recover-max-iterations may find a better "
                  "energy than what was reported for them:", file=sys.stderr, flush=True)
            for d in not_selected_but_promising:
                print(f"[run]   seq={d['seq']} recovered_energy={d['recovered_energy']:.6f} "
                      f"higher_rank_frac={d['higher_rank_frac']:.3f} "
                      f"n_recovery_steps={d['n_recovery_steps']}", file=sys.stderr, flush=True)

    # circuit cost on the target device. best_seq is always a real circuit's sequence
    # (whether from the sampling loop or the recovery winner) -- this computation
    # itself is never wrong, but when energy_source != "circuit" (energy came from a
    # local/global REFINED state, merged from several circuits with no single owner),
    # this depth does NOT belong to the reported energy_sqd and must not be presented
    # as if it does. best_circuit_depth/best_circuit_two_qubit_gates are ALWAYS the
    # depth of best_seq specifically (a real, always-available secondary number);
    # circuit_depth/circuit_two_qubit_gates additionally mirror them ONLY when
    # energy_source == "circuit", so a reader who only looks at the primary field
    # never sees a depth misattributed to a refined energy.
    print("[run] computing circuit depth/2q-gate metrics on the target device...", file=sys.stderr, flush=True)
    best_circuit_depth = best_circuit_twoq = within = None
    try:
        d = cm.circuit_depth_metrics(pool, best_seq)  # ideal transpile
        best_circuit_depth, best_circuit_twoq = d["depth"], d["two_qubit_gates"]
        within = best_circuit_depth <= hw.depth_budget
        print(f"[run] circuit metrics: depth={best_circuit_depth} 2q_gates={best_circuit_twoq} "
              f"within_budget={within}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[run] circuit metrics failed: {e}", file=sys.stderr, flush=True)
    depth = best_circuit_depth if energy_source == "circuit" else None
    twoq = best_circuit_twoq if energy_source == "circuit" else None
    if energy_source != "circuit":
        print(f"[run] energy_sqd came from {energy_source} (a merged, multi-circuit "
              "state, no single circuit owns it) -- circuit_depth/circuit_two_qubit_gates "
              "report None; see best_circuit_depth/best_circuit_two_qubit_gates for the "
              "best individual circuit's depth instead.", file=sys.stderr, flush=True)

    # excitation distribution + occupations from the accumulated refined state
    exc = occ = None
    merged = getattr(pipeline, "global_refined_state", None)
    if merged is not None and getattr(merged, "strings", None) is not None:
        w = np.abs(np.asarray(merged)) ** 2
        exc = cm.excitation_rank_histogram(merged.strings, system.nelec, weights=w)

    dim_used = shots_used = None
    if best_report is not None:
        subs = best_report.subspace_sizes
        dim_used = max(subs) if subs else None
        shots_used = int(pipeline.sampler.shots_count) if hasattr(pipeline.sampler, "shots_count") else None

    print(f"[run] final: energy={energy:.6f} dim_used={dim_used} shots_used={shots_used} "
          "-- assembling report...", file=sys.stderr, flush=True)

    return ChemistryReport(
        molecule=Path(molecule_path).stem, backend=backend, qubits=system.norb * 2,
        n_electrons=int(sum(system.nelec)), charge=int(getattr(cfg.molecule, "charge", 0)),
        spin_2s=int(system.spin), energy_sqd=float(energy), casci=casci, ccsd=ccsd, hf=hf,
        err_vs_casci_mha=((energy - casci) * 1e3) if casci is not None else None,
        err_vs_ccsd_mha=(energy - ccsd) * 1e3,
        chemical_accuracy=(abs(energy - casci) < 1.6e-3) if casci is not None else None,
        reference_label="CASCI (exact)" if casci is not None else "CCSD(T) (CASCI intractable at this scale)",
        ccsd_t=ccsd_t, err_vs_ccsd_t_mha=((energy - ccsd_t) * 1e3) if ccsd_t is not None else None,
        fci_dim=fci_dim, dim_used=dim_used, shots_used=shots_used,
        spin_squared=spin_sq, orbital_occupations=occ, excitation_distribution=exc,
        circuit_depth=depth, circuit_two_qubit_gates=twoq, backend_within_budget=within,
        n_gates_selected=cfg.ngates, mode=mode, recovery_iters=recovery_iters,
        n_circuits_skipped_deep=len(skipped_depths_all) if cfg.sqd.get("max_depth_for_diag") is not None else None,
        n_circuits_total=n_circuits_total if cfg.sqd.get("max_depth_for_diag") is not None else None,
        skipped_depths=skipped_depths_all if cfg.sqd.get("max_depth_for_diag") is not None else None,
        recovery_candidates=all_candidates_diag,
        not_selected_but_promising=not_selected_but_promising,
        energy_source=energy_source,
        best_circuit_depth=best_circuit_depth,
        best_circuit_two_qubit_gates=best_circuit_twoq,
        timings_s=timings_s or None,
        n_diagonalize_calls=n_diagonalize_calls,
        n_recovery_calls=n_recovery_calls if recover else None,
    )


def main():
    ap = argparse.ArgumentParser(description="Generate a circuit + chemistry report for a molecule on a backend.")
    ap.add_argument("--molecule", required=True, help="path to a MolecularSystem YAML")
    ap.add_argument("--backend", default="ideal", help="ibm_heron_r2 | ibm_heron_r3 | ideal")
    ap.add_argument("--checkpoint", default=None, help="pre-tuned policy weights (zero-shot if given)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for policy sampling, configuration recovery and "
                         "readout noise. Zero-shot sampling is unreproducible without "
                         "it: Categorical.sample() draws from the global stream.")
    ap.add_argument("--zeroshot-batches", type=int, default=1,
                    help="zero-shot only: number of independent candidate batches to "
                         "sample, keeping the best. Each batch is cfg.trainer.num_samples "
                         "circuits, so --zeroshot-batches 2 with num_samples 10 gives 20 "
                         "candidates. No gradient steps are taken either way; this only "
                         "reduces the variance of the reported number.")
    ap.add_argument("--adapt-iters", type=int, default=20)
    ap.add_argument("--recover", action="store_true",
                    help="run the configuration-recovery loop on the sampled bitstrings (for noisy/device runs)")
    ap.add_argument("--p-readout", type=float, default=0.0,
                    help="readout bit-flip probability applied to samples before recovery")
    ap.add_argument("--dynamic-resources", action="store_true",
                    help="predict max_dim/shots from a trained ResourceHead and diagonalize "
                         "via the adaptive-doubling loop, instead of the fixed config values")
    ap.add_argument("--max-dim-cap", type=int, default=1_000_000,
                    help="ceiling for the adaptive-doubling loop (one-shot inference call, so this "
                         "can afford to be much larger than amortized_train.py's per-iteration cap)")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="fixed SQD-subspace-size override (mutually sensible alternative to "
                         "--dynamic-resources, for a manual sweep over several dims on one molecule)")
    ap.add_argument("--diag-backend", default=None,
                    help='diagonalizer override: "sqd" | "fci" | "sbd" (GPU, RHF-only, '
                         "energy-only -- see sqd_gqe/sqd/backend_sbd_gpu.py before using for "
                         "anything beyond direct energy comparison). Default: cfg.sqd.diag_backend.")
    ap.add_argument("--sbd-binary-path", default=None,
                    help='path to the native diag-gpu binary (required with --diag-backend sbd '
                         "unless already set in configs_sqd/default.yaml's sqd.sbd_binary_path)")
    ap.add_argument("--sampler-kind", default=None, choices=["ffsim", "ibm_hardware"],
                    help='sampler backend: "ffsim" (default, exact simulation) or "ibm_hardware" '
                         "(real or fake IBM Quantum device -- see sqd_gqe/gqe/ibm_hardware_sampler.py). "
                         "Required at qubit counts past exact-statevector-sim reach.")
    ap.add_argument("--ibm-backend-name", default=None,
                    help='backend name for --sampler-kind ibm_hardware (e.g. "ibm_kingston", or '
                         '"least_busy" to pick the operational backend with fewest pending jobs)')
    ap.add_argument("--ibm-fake-backend", action="store_true",
                    help="use a qiskit_ibm_runtime fake_provider backend (AerSimulator, no cost/queue) "
                         "instead of real Qiskit Runtime -- for --sampler-kind ibm_hardware")
    ap.add_argument("--shots", type=int, default=None,
                    help="override cfg.sampler.shots (default 100,000 -- almost always too many for "
                         "a real Runtime job; e.g. 10000 for --sampler-kind ibm_hardware)")
    ap.add_argument("--ibm-save-dir", default=None,
                    help="when set with --sampler-kind ibm_hardware, save raw+decoded counts "
                         "(JSON) and transpiled circuits (QPY) under "
                         "<ibm-save-dir>/<backend>_shots<n>/<job_id>/ -- lets a real-hardware "
                         "result be re-diagonalized or visualized later, and checked for "
                         "reproducibility, without resubmitting")
    ap.add_argument("--ibm-dynamical-decoupling", action="store_true",
                    help="enable SamplerV2 dynamical decoupling (XpXm sequence) for "
                         "--sampler-kind ibm_hardware -- idles qubits with an echo sequence "
                         "during gate-free windows instead of leaving them exposed to dephasing")
    ap.add_argument("--ibm-session-mode", action="store_true",
                    help="keep all this run's Runtime jobs in one IBM Runtime Session instead "
                         "of job (batch) mode -- for --sampler-kind ibm_hardware only. The "
                         "session is opened lazily on first sample and always closed before "
                         "run_molecule returns (even on error), so it never leaks past this call.")
    ap.add_argument("--ibm-session-max-time", default=None,
                    help="Session TTL, e.g. '25m' or a plain integer seconds -- passed through "
                         "to qiskit_ibm_runtime.Session(max_time=...); subject to your plan's "
                         "own session-time limits (not double-checked here). Only used with "
                         "--ibm-session-mode.")
    ap.add_argument("--recover-max-iterations", type=int, default=10,
                    help="max configuration-recovery iterations for --recover's "
                         "recover_and_diagonalize call (default 10; qiskit-addon-sqd's own "
                         "default is 100)")
    ap.add_argument("--recover-subspace-dim", type=int, default=None,
                    help="alpha-determinant count for --recover's GPU (sbd) diagonalization "
                         "each step -- RHF, so net CI dim = this^2 (e.g. 1000 -> ~1e6 net "
                         "dim). Default (unset) uses samples_per_batch*num_batches=1536. "
                         "No effect on the CPU (sqd) recovery path.")
    ap.add_argument("--sbd-timeout-s", type=float, default=None,
                    help="per-call timeout for the diag-gpu subprocess (default from "
                         "configs_sqd/default.yaml, currently 600s) -- raise for large "
                         "--recover-subspace-dim/--max-dim runs where a single GPU solve "
                         "may exceed the default.")
    ap.add_argument("--beta", type=float, default=None,
                    help="pin the sampling inverse-temperature instead of using the config's "
                         "VarianceBeta schedule. act() samples from Categorical(logits=-beta*logits) "
                         "and a trained checkpoint's logit range is ~18, so the config default (0.5) "
                         "collapses the num_samples circuits to ~4 distinct sequences and 1.0 to a "
                         "single one -- which silently defeats the generate-N-keep-best procedure. "
                         "0.2 was measured to give 10/10 distinct circuits.")
    ap.add_argument("--max-depth-for-diag", type=int, default=False, nargs="?", const=None,
                    help="skip diagonalize()/recover() for any sampled circuit deeper than this "
                         "(energy=inf, but its SqdSample -- depth, diagonalize_skipped_deep, "
                         "hardware counts -- is still recorded). Default (flag omitted): "
                         "amortized_train.py's depth_ceiling_for_qubits(molecule's qubit count) "
                         "-- the same real, training-calibrated ceiling that gates whether a "
                         "training iteration's gradient step happens at all (<=40q: 300, <=60q: "
                         "500, <=80q: 800). Pass --max-depth-for-diag with no value to disable "
                         "entirely (diagonalize every circuit regardless of depth); pass an int "
                         "to set an explicit threshold.")
    ap.add_argument("--refine-method", default=None, choices=["gevp", "ufd", "tp_ufd"],
                    help='Default: cfg.sqd.refine_method ("gevp"). "gevp"/"ufd" both need real '
                         'per-determinant CI amplitudes to merge, which --diag-backend sbd never '
                         'produces (energy-only, GPU-focused -- see backend_sbd_gpu.py\'s module '
                         'docstring); build_merger() raises loudly if you combine them. '
                         '"tp_ufd" is the one refine method designed for amplitude-less GPU '
                         'output (only needs each circuit\'s alpha half-string list) -- pass it '
                         'whenever --diag-backend sbd is set.')
    ap.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    args = ap.parse_args()

    report = run_molecule(args.molecule, backend=args.backend, checkpoint=args.checkpoint,
                          zeroshot_batches=args.zeroshot_batches, seed=args.seed,
                          adapt_iters=args.adapt_iters, recover=args.recover, p_readout=args.p_readout,
                          dynamic_resources=args.dynamic_resources, max_dim_cap=args.max_dim_cap,
                          max_dim=args.max_dim, diag_backend=args.diag_backend,
                          sbd_binary_path=args.sbd_binary_path, sampler_kind=args.sampler_kind,
                          ibm_backend_name=args.ibm_backend_name, ibm_fake_backend=args.ibm_fake_backend,
                          shots=args.shots, ibm_save_dir=args.ibm_save_dir,
                          ibm_dynamical_decoupling=args.ibm_dynamical_decoupling,
                          ibm_session_mode=args.ibm_session_mode,
                          ibm_session_max_time=_parse_session_max_time(args.ibm_session_max_time),
                          recover_max_iterations=args.recover_max_iterations,
                          recover_subspace_dim=args.recover_subspace_dim,
                          sbd_timeout_s=args.sbd_timeout_s, beta=args.beta,
                          max_depth_for_diag=args.max_depth_for_diag,
                          refine_method=args.refine_method)
    print(report.to_json() if args.json else report.summary())


if __name__ == "__main__":
    main()
