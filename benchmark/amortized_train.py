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

"""Amortized (multi-molecule) trainer for the conditioned SQD-GQE policy.

Trains ONE conditioned policy across several molecules so that, at inference, a
new molecule needs no retraining --- just a forward pass with its descriptor.
Each iteration samples a molecule from the training set, conditions the policy
on (molecule, backend), rolls out circuits, scores them with SQD, and does a
GRPO update. Logs, per iteration:

  * GRPO loss and the sampled/refined energies (training curves),
  * per-molecule best error vs CASCI (accuracy curves),
  * neuron-learning diagnostics (grad norm, weight-change norm, dead-unit
    fraction, conditioning-encoder activation stats) --- "how much is the
    network actually learning".

Everything is written to experiments/amortized/ (git-ignored). Checkpoints the
tuned weights for zero-shot use by sqd_gqe.run.

Future-proofing: the policy is built through the factory/config, so the
transformer core can be swapped (a different model config) without touching
this trainer --- the conditioning + reward + logging are model-agnostic.
"""

from __future__ import annotations

import argparse

from sqd_gqe.core.prompt_encoders import canonical_arm
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"
CHEM_ACC_HA = 1.6e-3


def _write_resilient(write_fn, what: str, retries: int = 2, delay_s: float = 1.0) -> bool:
    """Run write_fn(), retrying on transient HPC-filesystem errors (NFS/Lustre
    "stale file handle", a brief remount making the parent directory itself
    look missing, etc.) instead of letting a filesystem hiccup kill an
    hours-long run. torch.save() can wrap the underlying OSError as a plain
    RuntimeError, so this catches broadly rather than just OSError."""
    for attempt in range(retries + 1):
        try:
            write_fn()
            return True
        except Exception as e:
            if attempt < retries:
                print(f"[WARN] {what}: {e} (retry {attempt + 1}/{retries})", flush=True)
                time.sleep(delay_s)
            else:
                print(f"[WARN] {what}: {e} (giving up after {retries} retries)", flush=True)
    return False


def _parse_session_max_time(value):
    """argparse always hands ``--ibm-session-max-time`` in as a plain string
    (no ``type=`` -- it can be either an int-looking string or an ``hms``
    string like ``"25m"``). ``qiskit_ibm_runtime.Session`` only special-cases
    a literal Python ``int``; anything else goes through its own
    ``hms_to_seconds``, whose regex requires a ``d``/``h``/``m``/``s`` suffix
    -- a bare digit string like ``"1500"`` has no suffix and raises there
    instead of being treated as seconds. Convert numeric strings to ``int``
    here so both '1500' and '25m' work from the CLI (same helper as
    sqd_gqe/run.py's --ibm-session-max-time)."""
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def load_system_policy(molecule_yaml, backend_name, device, shared_policy=None, max_vocab=32,
                       dynamic_resources=False, max_dim=None, n_reps=None, shots=None, ngates=None,
                       amplitude_damping=False, sampler_kind=None, ibm_backend_name=None,
                       ibm_fake_backend=False, extra_overrides=None, ibm_save_dir=None,
                       num_samples=None,
                       ibm_dynamical_decoupling=False, diag_backend=None, sbd_binary_path=None,
                       recover_iterations=None, sbd_timeout_s=None, encoder="add",
                       ibm_session_mode=False, ibm_session_max_time=None, refine_method=None,
                       recover_subspace_dim=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.core.system import FciIntractableError

    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    overrides = ["model=conditioned", f"+operator_pool.max_vocab={max_vocab}"]
    if extra_overrides:
        overrides.extend(extra_overrides)
    if n_reps is not None:
        # n_reps=null (default) means "auto from CCSD double-factorization" --
        # for ethanol's sparse 10e/16orb active space that picked n_reps=110,
        # whose UCJ seed op alone transpiles to depth ~51600 / ~128000 2q-gates
        # on Heron r2 (heavy-hex routing of a near-arbitrary two-body unitary
        # blows up badly; not a bug, see the CH4 dense-op precedent in the
        # deck). Capping it keeps every molecule's seed op in a comparable,
        # sane depth range so the accuracy-gated cost reward has something
        # tractable to shape away from, instead of one single token dominating
        # depth by 2-3 orders of magnitude before training even starts.
        overrides.append(f"operator_pool.n_reps={n_reps}")
    if ngates is not None:
        # ngates=10 (default) is far shorter than what a closely-matched
        # published GQE pipeline (arXiv:2607.23988, same 6-layer/43M-param
        # GPT-2 architecture) needed to reach chemical accuracy at comparable
        # qubit counts (L=80 for their 24-32q rungs) -- their own finding is
        # that circuit LENGTH, not a richer operator vocabulary, is the real
        # scaling lever: a longer circuit samples a richer, more diverse set
        # of determinants (the GEVP subspace grows near-linearly with L), so
        # a short circuit's sampled configuration set can be exhausted well
        # before max_dim is, which look identical to a "dimension-limited"
        # plateau but isn't. This directly resizes the transformer's position
        # embeddings (max_positions = ngates + 1), so it cannot be
        # warm-started from a checkpoint trained at a different ngates.
        overrides.append(f"ngates={ngates}")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        # fixed max vocabulary so ONE conditioned policy head serves every molecule
        cfg = compose(config_name="default", overrides=overrides)
    cfg.molecule = OmegaConf.load(molecule_yaml)
    cfg.dynamic_resources = dynamic_resources
    if max_dim is not None:
        cfg.sqd.max_dim = max_dim
    if recover_subspace_dim is not None:
        # Explicit alpha-determinant target for the GPU recovery loop
        # (recover_and_diagonalize), overriding SqdFactory._recover_subspace_dim's
        # sqrt(max_dim) derivation -- net CI dim = this^2. Independent of max_dim,
        # which still caps the cheaper one-shot per-iteration reward diagonalization
        # directly (unsquared, see ConfigurationSet.expand's max_size in pipeline.py).
        cfg.sqd.recover_subspace_dim = recover_subspace_dim
    if diag_backend is not None:
        cfg.sqd.diag_backend = diag_backend
    if sbd_binary_path is not None:
        cfg.sqd.sbd_binary_path = sbd_binary_path
    if recover_iterations is not None:
        cfg.sqd.recover_iterations = recover_iterations
    if sbd_timeout_s is not None:
        cfg.sqd.sbd_timeout_s = sbd_timeout_s
    # cfg.sqd.refine_method's own default ("gevp") needs real per-determinant CI
    # amplitudes to merge -- diag_backend="sbd" is GPU/energy-only (see
    # backend_sbd_gpu.py's module docstring) and never produces them, so
    # build_merger() raises loudly rather than silently merging placeholder
    # amplitudes as if they were physical. "tp_ufd" is the one refine method
    # designed for sbd's output (only needs each circuit's alpha half-string
    # list via sbd's carryover). Default to it automatically whenever sbd is
    # the diagonalizer, since no other choice is ever structurally valid here --
    # not left for the caller to remember, unlike run.py's zero-shot CLI (where
    # the caller explicitly names --diag-backend sbd too, but a wrong default
    # there only affects one inference call, not an hours-long training run).
    effective_diag_backend = diag_backend if diag_backend is not None else cfg.sqd.diag_backend
    if refine_method is not None:
        cfg.sqd.refine_method = refine_method
    elif effective_diag_backend == "sbd":
        cfg.sqd.refine_method = "tp_ufd"
    if num_samples is not None:
        # Circuits generated per iteration -- the pool the "keep the best" step picks
        # from, and the batch GRPO computes its advantage over. batch_size and
        # buffer_size track it because the run is on-policy (the yaml comments say so
        # explicitly), so leaving them at 10 while raising num_samples would silently
        # drop the extra circuits before the update.
        #
        # The default of 10 is small for this role. GQCO (arXiv:2501.16986) samples
        # 1024 circuits per step at 3 qubits and 96 at 10, and its DPO objective is
        # likewise "best against the rest". Measured cost here makes a larger batch
        # affordable: one iteration is 0.74 s wall (sample 0.11 s, diagonalize 0.18 s,
        # backward 0.01 s), so the diagonalization that scales with the batch is a
        # fifth of it -- 400 iterations took 4.8 minutes total.
        cfg.trainer.num_samples = int(num_samples)
        cfg.trainer.batch_size = int(num_samples)
        cfg.trainer.buffer_size = int(num_samples)
    if shots is not None:
        cfg.sampler.shots = shots
    cfg.sampler.amplitude_damping = bool(amplitude_damping)
    if sampler_kind is not None:
        cfg.sampler.kind = sampler_kind
    if ibm_backend_name is not None:
        cfg.sampler.ibm_backend_name = ibm_backend_name
    cfg.sampler.ibm_fake_backend = bool(ibm_fake_backend)
    if ibm_save_dir is not None:
        cfg.sampler.ibm_save_dir = ibm_save_dir
    cfg.sampler.ibm_dynamical_decoupling = bool(ibm_dynamical_decoupling)
    cfg.sampler.ibm_session_mode = bool(ibm_session_mode)
    if ibm_session_max_time is not None:
        cfg.sampler.ibm_session_max_time = ibm_session_max_time

    factory = SqdFactory()
    system = factory.build_system(cfg)
    hw = get_backend(backend_name)
    pipeline = factory.build_pipeline(cfg, backend=hw)
    schedule = factory.build_beta_schedule(cfg)
    objective = factory.build_objective(cfg)
    cond = ConditioningContext.build(system, hw)
    pool = factory.build_operator_pool(cfg)
    if encoder == "add":
        # existing additive path: the encoder lives inside the policy, and the
        # conditioning is a 1-D descriptor re-encoded inside every forward.
        policy = shared_policy or factory.build_policy(cfg).to(device)
        prompt_feats = None
    else:
        # cross-attention path. Built here rather than in factory.py so the
        # comparison cannot perturb the additive arm: nothing in configs_sqd or
        # factory.build_policy changes, and `--encoder add` takes the identical
        # code path it did before this flag existed.
        from sqd_gqe.core.prompt_encoders import (
            MaceContext, OrbitalSetContext, PosEgnnContext, PromptContext)
        # Route through configs_sqd/model/crossattn.yaml rather than constructing
        # CrossAttnPolicy here, so the policy has exactly one definition. Building
        # it in two places invites the two drifting apart -- a hyperparameter
        # changed in the yaml would silently not apply to these runs.
        if shared_policy is not None:
            policy = shared_policy
        else:
            from hydra import compose as _compose, initialize_config_dir as _icd
            with _icd(version_base="1.3", config_dir=abs_cfg):
                xcfg = _compose(config_name="default",
                                overrides=[o for o in overrides
                                           if not o.startswith("model=")]
                                          + ["model=crossattn"])
            xcfg.molecule = cfg.molecule
            policy = factory.build_policy(xcfg).to(device)
        # Features are a property of the molecule, so compute them once here
        # rather than per iteration. orbital_set is the only one whose length
        # varies with the active space (norb tokens); the rest are single-token.
        if encoder in ("posegnn", "posegnn_large", "posegnn_large_bounded"):
            # A5 returns a (embedding, features) pair rather than one tensor: the
            # per-atom POS-EGNN embedding and the descriptor stay separate prompt
            # tokens, so they cannot be packed into a single array.
            prompt_feats = PosEgnnContext.build(system, hw).tensors(device)
        elif encoder in ("mace", "mace_large", "mace_large_bounded"):
            # A6: same shape as A5, MACE-MP-0 embedding instead of POS-EGNN.
            prompt_feats = MaceContext.build(system, hw).tensors(device)
        elif encoder in ("orbital_set", "orbital_set_large", "orbital_set_xl"):
            # A2-large takes the SAME per-orbital features as A2; only the encoder
            # module differs (self-attention across orbitals instead of a shared
            # per-orbital MLP), so the two arms isolate encoder capacity.
            prompt_feats = OrbitalSetContext.build(system, hw).tensor(device)
        elif encoder == "scalar27_epsfix":
            prompt_feats = PromptContext.build(system, hw, epsfix=True).tensor(device)
        elif encoder in ("scalar27", "scalar27_norm", "scalar27_large", "scalar27_large_bounded"):
            # A1'' takes the SAME features as A1-xattn -- the only difference is the
            # LayerNorm inside NormScalarPromptEncoder, so the two arms differ by
            # normalisation alone. scalar27_large_bounded is A1-xattn-large's exact
            # input too -- the bounded variant differs only in the encoder's output
            # activation (see ScalarPromptEncoder.bound_output), never in what feeds it.
            prompt_feats = PromptContext.build(system, hw).tensor(device)
        else:                                     # null -- takes no input at all
            prompt_feats = None
    try:
        casci = float(system.casci_energy())
        ref_label = "casci"
    except FciIntractableError:
        casci = float(system.ccsd_t_energy())
        ref_label = "ccsd_t"
    return dict(cfg=cfg, system=system, pipeline=pipeline, schedule=schedule,
                objective=objective, cond=cond, policy=policy, pool=pool, hw=hw,
                casci=casci, ref_label=ref_label, prompt_feats=prompt_feats,
                hf_energy=float(system.hf.e_tot),
                sampler_kind=sampler_kind if sampler_kind is not None else "ffsim")


def neuron_diagnostics(policy, prev_flat, prompt_encoder=None, prev_flat_encoder=None):
    """Grad norm and weight-change norm, over the policy AND the prompt encoder.

    The prompt encoder has to be included or the diagnostics miss the very thing
    this experiment varies. On the cross-attention arms it is a separate module, so
    a policy-only grad_norm reports nothing about whether the encoder is learning --
    measured on n2/orbital_set, the encoder contributes 0.46 against the policy's
    38.36, so it is invisible in a policy-only figure (1.2%) yet is exactly the
    quantity that separates one arm from another.

    grad_norm_encoder / weight_delta_encoder are reported separately as well, since
    an encoder that has stopped moving while the decoder keeps training is the
    failure mode worth catching, and it cannot be seen in a combined norm. This
    needs its own previous-snapshot flat (prev_flat_encoder), distinct from the
    combined policy+encoder prev_flat, since the encoder's own delta cannot be
    recovered from the combined one when the two modules' parameter counts differ
    run to run (e.g. --encoder-lr-mult trials against a plain add run).
    """
    def _gn(params):
        g2 = 0.0
        for p in params:
            if p.grad is not None:
                g2 += float(p.grad.detach().pow(2).sum())
        return g2 ** 0.5

    pol_params = list(policy.parameters())
    enc_params = list(prompt_encoder.parameters()) if prompt_encoder is not None else []
    flat = torch.cat([p.detach().reshape(-1) for p in pol_params + enc_params])
    dW = float((flat - prev_flat).norm()) if prev_flat is not None and prev_flat.numel() == flat.numel() else 0.0
    out = {"grad_norm": _gn(pol_params + enc_params),
           "grad_norm_policy": _gn(pol_params),
           "weight_delta": dW, "n_params": int(flat.numel())}
    enc_flat = None
    if prompt_encoder is not None:
        out["grad_norm_encoder"] = _gn(enc_params)
        enc_flat = torch.cat([p.detach().reshape(-1) for p in enc_params])
        out["encoder_weight_norm"] = float(enc_flat.norm())
        dW_enc = (float((enc_flat - prev_flat_encoder).norm())
                  if prev_flat_encoder is not None and prev_flat_encoder.numel() == enc_flat.numel()
                  else 0.0)
        out["weight_delta_encoder"] = dW_enc
    return out, flat, enc_flat


def depth_ceiling_for_qubits(n_qubits: int) -> int:
    """Max acceptable real-transpiled depth for the best circuit, by active-space size.

    Calibrated to this project's real training data (not arbitrary): at 38q/46q
    real-hardware scale, sampled circuits swing across three orders of magnitude
    (19 to 11000+ depth) within the same 10-circuit batch, and the reward's
    accuracy-gated depth penalty (cost_reward.py) never rejects an update for
    this -- it only nudges reward magnitude once a circuit is already accurate.
    This ceiling instead gates whether a gradient update happens at all.
    """
    if n_qubits <= 40:
        return 300
    if n_qubits <= 60:
        return 500
    if n_qubits <= 80:
        return 800
    raise ValueError(f"no depth ceiling configured for {n_qubits} qubits (>80)")


def main():
    """Thin wrapper around ``_main_impl`` that guarantees every env's pipeline
    is closed (releasing any open IBM Runtime Session, --ibm-session-mode)
    even if training crashes or is interrupted -- there is no Lightning-style
    teardown hook on this plain training loop (see sqd_gqe/core/training.py's
    ``teardown`` for the equivalent on the train_sqd.py/LightningModule path,
    which this script doesn't use)."""
    envs_holder: list = []
    try:
        _main_impl(envs_holder)
    finally:
        for env in envs_holder:
            pipeline = env.get("pipeline") if isinstance(env, dict) else None
            if pipeline is not None:
                pipeline.close()


def _main_impl(_envs_holder: list):
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", required=True,
                    help="training molecule YAMLs. Each entry may optionally end in "
                         "':ffsim' or ':ibm_hardware' to pin that ONE molecule to a "
                         "different sampler than --sampler-kind (e.g. "
                         "'molecules/small/SnOH3-me_S_....yaml:ibm_hardware' alongside "
                         "other entries with no suffix, which use --sampler-kind as "
                         "before) -- for a mixed pool where one molecule needs real "
                         "hardware to avoid ffsim's cost at that scale but the rest "
                         "don't.")
    ap.add_argument("--backend", default="ibm_heron_r2")
    ap.add_argument("--amplitude-damping", action="store_true",
                    help="train with T1-decay-derived directional (1->0) noise on top of the "
                         "existing symmetric readout/gate-noise proxy -- models the dominant "
                         "real-hardware failure mode (collapse toward |0...0>) that neither "
                         "existing noise term teaches resilience against. Requires a *_noisy_sim "
                         "--backend (same gating as the existing gate-noise proxy) or this is a no-op.")
    ap.add_argument("--sampler-kind", default=None, choices=["ffsim", "ibm_hardware"],
                    help="'ibm_hardware' samples every training-iteration batch as ONE real "
                         "Qiskit Runtime job (see IbmHardwareSampler._sample_batch) instead of "
                         "ffsim simulation -- real QPU time/cost per iteration. Requires "
                         "--ibm-backend-name and IBM_API_KEY/IBM_CRN env vars (sourced from "
                         ".env.local before this script starts, never passed on the CLI).")
    ap.add_argument("--ibm-backend-name", default=None,
                    help='real (or fake, with --ibm-fake-backend) IBM backend name, e.g. '
                         '"ibm_kingston" -- required when --sampler-kind ibm_hardware.')
    ap.add_argument("--ibm-fake-backend", action="store_true",
                    help="use a qiskit_ibm_runtime.fake_provider backend + AerSimulator instead "
                         "of real hardware for --sampler-kind ibm_hardware -- no credentials/"
                         "queue/cost, for smoke-testing the training-loop wiring itself.")
    ap.add_argument("--ibm-save-dir", default=None,
                    help="when set with --sampler-kind ibm_hardware, save every training "
                         "iteration's raw+decoded counts (JSON) and transpiled circuits (QPY) "
                         "under <ibm-save-dir>/<backend>_shots<n>/<job_id>/ -- for later "
                         "re-diagonalization, circuit visualization, or reproducibility checks.")
    ap.add_argument("--ibm-dynamical-decoupling", action="store_true",
                    help="enable SamplerV2 dynamical decoupling (XpXm sequence) for "
                         "--sampler-kind ibm_hardware -- idles qubits with an echo sequence "
                         "during gate-free windows instead of leaving them exposed to dephasing")
    ap.add_argument("--ibm-session-mode", action="store_true",
                    help="keep each molecule's Runtime jobs in one IBM Runtime Session across "
                         "every training iteration, instead of job (batch) mode -- for "
                         "--sampler-kind ibm_hardware only. Every session opened is closed before "
                         "main() returns, even on a crash/interrupt.")
    ap.add_argument("--ibm-session-max-time", default=None,
                    help="Session TTL, e.g. '25m' or a plain integer seconds -- passed through "
                         "to qiskit_ibm_runtime.Session(max_time=...); subject to your plan's "
                         "own session-time limits (not double-checked here). Only used with "
                         "--ibm-session-mode.")
    ap.add_argument("--hydra-override", action="append", default=[],
                    help="arbitrary additional hydra override (e.g. "
                         "'trainer.beta_schedule.target_var=1e-3'), repeatable. Escape hatch for "
                         "one-off tuning (e.g. VarianceBeta's target_var/delta) without adding a "
                         "dedicated flag for every config knob.")
    ap.add_argument("--iters", type=int, default=60, help="total GQE iterations (across molecules)")
    ap.add_argument("--cost-lambda", type=float, default=0.0,
                    help="accuracy-gated 2q-gate penalty weight (Ha/CX); >0 makes circuits shallow")
    ap.add_argument("--cost-lambda-dim", type=float, default=0.0,
                    help="accuracy-gated SQD-subspace-size penalty weight (Ha per log10(dim)); "
                         ">0 makes circuits easy to diagonalize, not just shallow (needs --dynamic-resources)")
    ap.add_argument("--cost-lambda-real-depth", type=float, default=0.0,
                    help="accuracy-gated REAL backend-transpiled-depth penalty weight (Ha per "
                         "log10(depth)) -- fixes --cost-lambda's cheap per-op gate-count proxy, which "
                         "under-prices heavy-hex SWAP overhead and let ethanol's policy converge to a "
                         "confidently accurate but ~15000-deep circuit family. >0 computes a real "
                         "transpile for every sampled circuit each iteration (not just the logged best "
                         "one), so this is more expensive per iteration than --cost-lambda alone.")
    ap.add_argument("--cost-lambda-excitation", type=float, default=0.0,
                    help="accuracy-gated penalty weight (Ha per unit fraction, 0..1) on the sampled "
                         "circuit's excitation-rank->3 ('higher') weight fraction vs Hartree-Fock -- "
                         "rewards a physically sensible S/D/T-dominated wavefunction instead of just a "
                         "low energy number. Motivated by real SnOH3-L zero-shot inference where the "
                         "accepted circuit's recovered state sat at ~97.5% 'higher'-rank weight, 0% "
                         "singles/doubles -- accuracy-gated-accepted but not CCSD-like. Computed from "
                         "each sampled circuit's raw counts (cheap, no extra diagonalization). Gated by "
                         "--gate-loose-mha, not the strict chemical-accuracy threshold -- see that "
                         "flag's help for why.")
    ap.add_argument("--cost-lambda-hf", type=float, default=0.0,
                    help="UNGATED penalty weight (Ha per unit fraction, 0..1) on the share of right-spin-sector shots that are exactly the "
                         "Hartree-Fock determinant. Counters the noisy-device collapse to a circuit that does nothing (Training 2, Stage 2).")
    ap.add_argument("--cost-lambda-spin", type=float, default=0.0,
                    help="UNGATED penalty weight (Ha per unit fraction) on the alpha/beta imbalance |alpha-only - beta-only| of the right-spin-sector "
                         "shots. Counters circuits that excite one spin channel only (seen on ibm_kobe in Training 2, Stage 2).")
    ap.add_argument("--entropy-coef", type=float, default=0.0,
                    help="entropy bonus: loss -= coef * mean token entropy of the policy on the sampled sequences. Counters collapse of the policy "
                         "onto a single sequence.")
    ap.add_argument("--cost-lambda-coverage", type=float, default=0.0,
                    help="accuracy-gated penalty weight (Ha per unit idle-qubit fraction, 0..1) "
                         "pushing sampled circuits to use more of the molecule's qubits, not just "
                         "reach a low energy on whichever qubits they happen to touch -- motivated "
                         "by the real finding that --ngates=8 circuits structurally cap active-qubit "
                         "count at ngates*4 regardless of the molecule's true qubit count (see "
                         "docs/reports/session_status_20260923.html Sec 25-26). idle_fracs[i] = "
                         "1.0 - active_qubits/n_qubits, from the same real backend-transpiled circuit "
                         "already computed for --cost-lambda-real-depth (no new transpile when both "
                         "are set). Gated by --gate-loose-mha, same as --cost-lambda-real-depth/"
                         "--cost-lambda-excitation.")
    ap.add_argument("--cost-lambda-overshoot", type=float, default=0.0,
                    help="UNGATED penalty weight (Ha per log10-unit overshoot ratio) once a "
                         "circuit's real transpiled depth exceeds the active depth ceiling "
                         "(--depth-ceiling-max or depth_ceiling_for_qubits()) -- applies REGARDLESS "
                         "of accuracy, unlike every other cost term here. Real fix for "
                         "--cost-lambda-coverage's own failure mode found in production (report Sec "
                         "26.9-26.10): on a hard molecule that never reaches --gate-loose-mha, none "
                         "of the accuracy-gated terms ever fire, so nothing limited depth and the "
                         "hard depth-ceiling gate on backward() (gate_passed) just skipped the "
                         "gradient update outright with no corrective signal, letting depth climb "
                         "pass-over-pass with no way back (observed: dW=0.0000 for 3 straight "
                         "iterations with depth climbing 542->7873 on the same repeated molecule). "
                         "Zero at or under the ceiling, grows continuously past it, on every sampled "
                         "circuit whether or not it's accurate yet.")
    ap.add_argument("--gate-loose-mha", type=float, default=None,
                    help="wider accuracy gate (mHa) that --cost-lambda-real-depth and "
                         "--cost-lambda-excitation use instead of the strict chemical-accuracy "
                         "threshold (1.6 mHa) that --cost-lambda/--cost-lambda-dim keep requiring. "
                         "On this project's harder curriculum molecules (fe_cn6, co2, c2h4), "
                         "raw_err_mha routinely sits at 20-290+ mHa in production -- with only the "
                         "strict gate, the excitation/depth penalties essentially never switch on for "
                         "exactly the molecules where the excitation-rank problem is worst, so they can "
                         "only ever shape behavior on molecules already solved almost exactly. Does NOT "
                         "loosen the guarantee that --cost-lambda's gate-count penalty (or --cost-lambda-dim) "
                         "trades away real accuracy -- those two keep the strict gate. Default (unset): "
                         "same as the strict threshold, i.e. no change from before this flag existed.")
    ap.add_argument("--dynamic-resources", action="store_true",
                    help="predict max_dim/shots per molecule from ResourceHead (or fall back to config "
                         "values without a checkpoint) and use that as the FIXED size for this molecule's "
                         "training -- NOT the adaptive-doubling loop, which is one-shot-only (reserved for "
                         "sqd_gqe.run's inference path and build_resource_dataset.py): growing every "
                         "sampled circuit through several doubling steps EVERY iteration measured >20min "
                         "for a single 32q iteration, prohibitive for round-robin training.")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="override cfg.sqd.max_dim uniformly for every training molecule (no ResourceHead "
                         "checkpoint exists yet the first time this trainer runs, so --dynamic-resources "
                         "alone falls back to the config default of 2000 -- too small for the 32q+ tier)")
    ap.add_argument("--recover-subspace-dim", type=int, default=None,
                    help="explicit alpha-determinant target for the GPU recovery loop "
                         "(recover_and_diagonalize), overriding the sqrt(max_dim) derivation in "
                         "SqdFactory._recover_subspace_dim -- net CI dim = this^2. Independent of "
                         "--max-dim, which still caps the cheaper one-shot per-iteration reward "
                         "diagonalization directly (not squared).")
    ap.add_argument("--depth-ceiling-max", type=int, default=None,
                    help="flat override for depth_ceiling_for_qubits()'s per-qubit-tier lookup "
                         "(300/500/800 for <=40q/<=60q/<=80q) -- the hard accept/reject gate on "
                         "whether a gradient update happens at all, distinct from the reward's own "
                         "soft accuracy-gated depth penalty. Real depth-vs-outcome data (2026-09-27, "
                         "--ngates=20/40 experiment) showed no failures clustering just above the "
                         "default tiers -- passing circuits land at depth 40-284, failing ones are "
                         "typically 3-40x over -- so this exists to test a genuinely larger ceiling "
                         "(e.g. 600), not to nudge a marginal one.")
    ap.add_argument("--diag-backend", default=None,
                    help='diagonalizer override: "sqd" (CPU, default) | "fci" | "sbd" (GPU) -- the '
                         "CPU diagonalizer can take over an hour per iteration at 32q+ real-hardware "
                         "scale (found via Stage F's stalls); use sbd for any real training run past "
                         "~28 qubits. See sqd_gqe/sqd/backend_sbd_gpu.py before using for anything "
                         "beyond direct energy comparison (no real CIVector, single-call only).")
    ap.add_argument("--sbd-binary-path", default=None,
                    help='path to the native diag-gpu binary (required with --diag-backend sbd '
                         "unless already set in configs_sqd/default.yaml's sqd.sbd_binary_path)")
    ap.add_argument("--refine-method", default=None, choices=["gevp", "ufd", "tp_ufd"],
                    help='Default: "tp_ufd" whenever --diag-backend sbd is set (gevp/ufd both need '
                         "real per-determinant CI amplitudes, which sbd's GPU/energy-only output "
                         "never produces -- build_merger() raises loudly rather than silently "
                         'merging placeholder amplitudes); "gevp" (cfg.sqd.refine_method\'s own '
                         "default) otherwise. Override only if you know what you're asking for.")
    ap.add_argument("--recover-iterations", type=int, default=None,
                    help="self-consistent configuration-recovery steps feeding each sampled "
                         "circuit's reward energy (default from configs_sqd/default.yaml is 5; "
                         "0 disables recovery and reverts to a one-shot diagonalize() reward). "
                         "Each step reruns diagonalization, so this multiplies per-circuit "
                         "diagonalization cost by recover-iterations.")
    ap.add_argument("--sbd-timeout-s", type=float, default=None,
                    help="per-call timeout for the diag-gpu/diag-gpu-mpi subprocess (default from "
                         "configs_sqd/default.yaml, currently 600s) -- raise for large --max-dim "
                         "runs where a single GPU/multi-GPU Davidson solve may exceed the default "
                         "(confirmed: a 4-GPU 1e6-net-dim call timed out at 600s on job 65092).")
    ap.add_argument("--shots", type=int, default=10_000,
                    help="fixed shots for every molecule (not tuned per-molecule/dynamically): 10k already "
                         "lets the adaptive-doubling solve range up to ~1e8-scale subspaces at inference, "
                         "so there's little to gain from also varying shots during training)")
    ap.add_argument("--log-circuit-detail", action="store_true",
                    help="log per-circuit depth/2q-gates/excitation-decomposition "
                         "(HF/S/D/T/Q/higher) and valid-bitstring counts + energy at the "
                         "start and end of each circuit's recovery loop, for all "
                         "num_samples circuits every iteration -- written to history.json's "
                         "row['circuits']. Off by default (adds real_depths-equivalent "
                         "transpile cost + recovery diagnostics collection overhead).")
    ap.add_argument("--n-reps", type=int, default=None,
                    help="override operator_pool.n_reps (UCJ repetitions) uniformly; default (null) picks "
                         "reps from CCSD double-factorization, which can pick unexpectedly large values "
                         "for sparser active spaces (e.g. 110 for ethanol's 10e/16orb) whose transpiled "
                         "depth then dwarfs every other pool op by 2-3 orders of magnitude")
    ap.add_argument("--ngates", type=int, default=None,
                    help="override cfg.ngates (circuit length / gate-tokens per sequence) uniformly; "
                         "default (null) keeps the config default of 10. Resizes the transformer's "
                         "position embeddings, so a run with a non-default ngates cannot be "
                         "--warmstart-ed from a checkpoint trained at a different ngates.")
    ap.add_argument("--max-vocab", type=int, default=32, help="padded common vocabulary size")
    ap.add_argument("--warmstart", default=None, help="checkpoint to warm-start from (transformer body transfers)")
    ap.add_argument("--track-circuits", default=None, metavar="FILE.jsonl",
                    help="write, for every iteration, the generated token sequences and their chemistry (JSON lines; see "
                         "sqd_gqe/gqe/training_tracking.py): operator families and Givens rotations relative to the Fermi level, "
                         "the sampled (alpha, beta) pairs split into the right and the wrong spin sector (alpha<->beta transfer, "
                         "electron-number errors), spin-resolved excitations and the HOMO->LUMO weights, the policy's token "
                         "entropy, and a summary of these under the 'track' key of each history row. The first line holds the run "
                         "metadata (arguments, library versions, file hashes). Off by default.")
    ap.add_argument("--warmstart-encoder", action="store_true",
                    help="also restore the prompt encoder from the --warmstart checkpoint's prompt_encoder_state_dict. "
                         "Off by default: until this flag existed the training warm start restored only the policy, so the "
                         "encoder restarted from its seed initialisation at every warm start and every resume.")
    ap.add_argument("--out", default=str(EXP / "amortized_run.json"))
    ap.add_argument("--ckpt", default=str(EXP / "amortized_policy.pt"))
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="save the policy checkpoint every N iterations, not just at the end -- "
                         "so a crash (e.g. a transient filesystem error) doesn't lose the whole run")
    ap.add_argument("--seed", type=int, default=32,
                    help="RNG seed for policy weight init and rollout sampling. Default matches "
                         "configs_sqd/trainer/default.yaml's trainer.seed, which factory.py already "
                         "threads into the sampler and operator pool -- this flag closes the two "
                         "paths that config never reached (see the set_seed call below)")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="circuits generated per training iteration (default 10 from "
                         "configs_sqd/trainer/default.yaml). This is the pool the best-of-N "
                         "step selects from and the batch GRPO takes its advantage over; "
                         "batch_size and buffer_size follow it, since the run is on-policy. "
                         "Measured at 0.74 s per iteration with 10, of which the "
                         "batch-scaling part (diagonalization) is 0.18 s, so raising it is "
                         "cheap. For reference GQCO (arXiv:2501.16986) uses 96-1024.")
    ap.add_argument("--encoder", default="add",
                    choices=[
                        # arm names, matching the handover table's Name column
                        "baseline", "a0",
                        "a1_xattn", "a1_xattn_large", "a1_xattn_norm", "a1_xattn_epsfix",
                        "a2", "a2_large", "a2_xl",
                        "a5", "a5_large",
                        "a6_mace", "a6_mace_large",
                        # module names, kept so existing scripts, checkpoints and logs
                        # keep resolving
                        "add", "null",
                        "scalar27", "scalar27_epsfix", "scalar27_norm", "scalar27_large",
                        "orbital_set", "orbital_set_large", "orbital_set_xl",
                        "posegnn", "posegnn_large", "mace", "mace_large",
                        # bounded variants (tanh-clamped encoder output, see
                        # prompt_encoders.py) of the focus arms
                        "a1_xattn_large_bounded", "a5_large_bounded", "a6_mace_large_bounded",
                        "scalar27_large_bounded", "posegnn_large_bounded", "mace_large_bounded"],
                    help="which conditioning encoder to compare. 'add' keeps the existing additive "
                         "path (ConditionedTransformerPolicy, encoder built into the policy); every "
                         "other value swaps in CrossAttnPolicy plus the named prompt encoder, whose "
                         "prompt length is free (orbital_set gives one token per active orbital). "
                         "Only the encoder and the injection route change -- pipeline, objective, "
                         "gate and seed are identical, so a measured gap is attributable")
    ap.add_argument("--encoder-lr-mult", type=float, default=1.0,
                    help="prompt encoder LR = base LR * this multiplier, as a separate AdamW "
                         "param group from the policy. Default 1.0 keeps every arm's original "
                         "single-LR behaviour; only changes anything when passed explicitly. "
                         "No-op when --encoder add (there is no separate encoder module).")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="clip_grad_norm_ max norm over policy+encoder params, applied after "
                         "backward() and after the grad_norm/grad_norm_encoder diagnostics are "
                         "snapshotted (so those metrics report the raw, pre-clip gradient) and "
                         "before opt.step(). Default None (off), matching every existing arm's "
                         "measured behaviour.")
    ap.add_argument("--size-bias-correction", action="store_true",
                    help="bias the round-robin molecule schedule away from molecules with a high "
                         "measured std(energies)==0 fallback rate (see reward_degenerate in the "
                         "training loop) -- the cross-iteration analogue of inverse-size curriculum "
                         "sampling that applies to this repo's real one-molecule-per-batch training "
                         "loop. Default off, so existing reproduction commands are unaffected.")
    ap.add_argument("--size-curriculum", action="store_true",
                    help="explicit 3-phase progressive size curriculum (--warmup-frac/"
                         "--focus-frac split --iters into warmup/focus/harden phases): "
                         "phase 1 -- plain round-robin over every molecule (unchanged); "
                         "phase 2 -- round-robin restricted to the top --curriculum-top-frac "
                         "largest molecules by qubit count; phase 3 -- same restricted set, "
                         "plus Gaussian noise injected into that iteration's reward "
                         "(cost_adjusted_energies output) via --curriculum-noise-std. "
                         "Default off; combine with --size-bias-correction only if you "
                         "want both the fallback-rate reroll AND the phase restriction "
                         "(the phase restriction is applied first, then the reroll acts "
                         "within whatever set phase 2/3 already restricted to).")
    ap.add_argument("--warmup-frac", type=float, default=0.25,
                    help="--size-curriculum only: fraction of --iters spent in phase 1 "
                         "(plain round-robin over all molecules). Default 0.25, "
                         "matching the proposal's 100/400 epochs.")
    ap.add_argument("--focus-frac", type=float, default=0.5,
                    help="--size-curriculum only: fraction of --iters spent in phase 2 "
                         "(focus on the largest molecules, no noise). Default 0.5, "
                         "matching the proposal's 200/400 epochs (101-300). Phase 3 gets "
                         "whatever iterations remain after warmup_frac+focus_frac.")
    ap.add_argument("--curriculum-top-frac", type=float, default=0.2,
                    help="--size-curriculum only: fraction of training molecules (by "
                         "qubit count) treated as the 'largest' set in phases 2-3. "
                         "Default 0.2 per the proposal's 'top 20%%'; rounded up to at "
                         "least 1 molecule so a small training set still has a focus set.")
    ap.add_argument("--curriculum-noise-std", type=float, default=0.01,
                    help="--size-curriculum only: std (Ha) of Gaussian noise added to "
                         "the reward tensor during phase 3 ('noise-injected GRPO "
                         "rewards to harden policy generalization'). Default 0.01 Ha, "
                         "matching this branch's other quantum-hardware-readiness noise "
                         "convention (see prompt_encoders.py's docstrings on latent "
                         "bounding for the same order-of-magnitude reasoning).")
    ap.add_argument("--reward-normalize", action="store_true",
                    help="use a reference-relative, clipped reward "
                         "(R = (E_HF-E_ref)/(E_circuit-E_ref) - 1, clipped at "
                         "--reward-normalize-max) instead of raw energy -- mapped from "
                         "arXiv:2607.22468 Eq.1 (ADAPT-GQE). Default off, so existing "
                         "arms' measured numbers stay reproducible unless requested.")
    ap.add_argument("--reward-normalize-max", type=float, default=30.0,
                    help="--reward-normalize only: clip value for the normalized "
                         "reward (paper's own value, not yet validated for this "
                         "codebase's molecule set/energy scale).")
    ap.add_argument("--retry-on-fail", type=int, default=0,
                    help="on a gate_passed=False iteration, stay on the SAME molecule "
                         "for up to this many additional consecutive iterations before "
                         "round-robin/curriculum/size-bias-correction advance to the next "
                         "one, instead of always advancing regardless of outcome. "
                         "Consumes iteration budget (a retry is one fewer turn for some "
                         "other molecule), mirroring a collaborator's own c2h4 "
                         "resampling experiment. Default 0 (off) -- j always advances "
                         "exactly as before.")
    ap.add_argument("--early-stop-window", type=int, default=None,
                    help="stop training early if the overall gate_passed rate, "
                         "averaged over the trailing --early-stop-window iterations, "
                         "has not improved by at least --early-stop-min-improve versus "
                         "the window ending --early-stop-window iterations earlier "
                         "(i.e. comparing two adjacent, non-overlapping windows of this "
                         "size). Needs at least 2x this many iterations to have run "
                         "before the check can fire. Default None (off) -- long "
                         "--iters budgets (e.g. 1000) run to completion unless this is "
                         "set, matching every existing run's behavior.")
    ap.add_argument("--early-stop-min-improve", type=float, default=0.02,
                    help="--early-stop-window only: minimum absolute increase in "
                         "windowed gate_passed rate required to count as still "
                         "improving. Default 0.02 (2 percentage points) -- small "
                         "enough to not trigger on ordinary seed noise (this "
                         "session's between-seed spreads are typically 5-15 points), "
                         "large enough to not require a literal all-time-high every "
                         "check.")
    ap.add_argument("--gate-energy-mha", type=float, default=100.0,
                    help="energy_ok threshold (mHa) on this iteration's raw best error "
                         "-- gate_passed requires abs(raw_err_mha) <= this value before "
                         "a backward pass runs at all. Default 100.0, the originally "
                         "hardcoded value (chosen because chemical accuracy, 1.6mHa, "
                         "rejected every iteration observed this session -- not "
                         "curriculum- or qubit-count-aware). Exposed as a flag to test "
                         "whether a looser gate (e.g. 200) lets training keep updating "
                         "on molecules whose best circuit plateaus above 100mHa, "
                         "instead of freezing (see docs/20260920/production_v1_report/"
                         "diagnostics_200iter.html for the stall this was measured on).")

    args = ap.parse_args()
    # Arm names from the handover table resolve to the module names the rest of this
    # file compares against ("add") and writes into checkpoints.
    args.encoder = canonical_arm(args.encoder)

    # Per-molecule --sampler-kind override: "path.yaml:ibm_hardware" pins that one
    # molecule to a different sampler than the global --sampler-kind, so a mixed
    # pool (e.g. 14 light molecules on ffsim + one 32q molecule that needs real
    # hardware to avoid ffsim's confirmed 20+min/iteration statevector-sampling
    # wall at that scale) can round-robin together in ONE process instead of
    # needing separate runs + --warmstart chaining. No colon in any existing
    # molecule path this session, so this is fully backward compatible -- every
    # entry without a colon uses args.sampler_kind exactly as before.
    molecule_paths = []
    molecule_sampler_overrides = []
    for m in args.molecules:
        if ":" in m and m.rsplit(":", 1)[1] in ("ffsim", "ibm_hardware"):
            path, kind = m.rsplit(":", 1)
            molecule_paths.append(path)
            molecule_sampler_overrides.append(kind)
        else:
            molecule_paths.append(m)
            molecule_sampler_overrides.append(args.sampler_kind)
    args.molecules = molecule_paths

    from sqd_gqe.gqe.cost_reward import cost_adjusted_energies
    from sqd_gqe.gqe import circuit_metrics as cm

    EXP.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session_max_time = _parse_session_max_time(args.ibm_session_max_time)

    # Seed BEFORE build_policy: weight init consumes the global torch RNG, and
    # policy.act()'s Categorical.sample() takes no generator argument, so both
    # the initial weights and every rollout draw come from that same global
    # stream. factory.py already passes cfg.trainer.seed to the sampler and the
    # operator pool, but nothing seeded these two paths -- without this, two runs
    # of the same config diverge from iteration 0, which makes an encoder-vs-encoder
    # comparison unattributable (is the gap the encoder, or the init?).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"seed={args.seed} (policy init + rollout sampling; sampler/pool via cfg.trainer.seed)",
          flush=True)

    # build a shared policy from the first molecule, then reuse it for all
    first = load_system_policy(args.molecules[0], args.backend, device, max_vocab=args.max_vocab,
                               dynamic_resources=args.dynamic_resources, max_dim=args.max_dim, n_reps=args.n_reps,
                               shots=args.shots, ngates=args.ngates, amplitude_damping=args.amplitude_damping,
                               sampler_kind=molecule_sampler_overrides[0], ibm_backend_name=args.ibm_backend_name,
                               ibm_fake_backend=args.ibm_fake_backend, extra_overrides=args.hydra_override,
                               num_samples=args.num_samples,
                               ibm_save_dir=args.ibm_save_dir,
                               ibm_dynamical_decoupling=args.ibm_dynamical_decoupling,
                               ibm_session_mode=args.ibm_session_mode, ibm_session_max_time=session_max_time,
                               diag_backend=args.diag_backend, sbd_binary_path=args.sbd_binary_path,
                               recover_iterations=args.recover_iterations, sbd_timeout_s=args.sbd_timeout_s,
                               encoder=args.encoder, refine_method=args.refine_method,
                               recover_subspace_dim=args.recover_subspace_dim)
    _envs_holder.append(first)
    policy = first["policy"]
    conditioned = hasattr(policy, "set_condition")

    # The prompt encoder is a separate module on the cross-attention arms (the
    # additive arm's encoder is inside the policy), so it needs building here and
    # -- critically -- its parameters must reach the optimizer below, or the arm
    # would train the decoder against a frozen random prompt.
    prompt_encoder = None
    if args.encoder != "add":
        from sqd_gqe.core.prompt_encoders import build_prompt_encoder
        prompt_encoder = build_prompt_encoder(args.encoder, policy.config.n_embd).to(device)
    # warm-start: transfer whatever matches (transformer body + encoder); a
    # resized embedding/head (larger max_vocab) simply starts fresh.
    if args.warmstart and Path(args.warmstart).exists():
        ws = torch.load(args.warmstart, map_location=device, weights_only=False)
        sd = ws.get("state_dict", ws)
        cur = policy.state_dict()
        # keep only params whose shape matches (drops the resized embedding/head)
        transfer = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        skipped = [k for k in sd if k not in transfer]
        policy.load_state_dict(transfer, strict=False)
        print(f"warm-started from {args.warmstart}: transferred {len(transfer)}, "
              f"reset {len(skipped)} (e.g. resized embedding/head): {skipped}", flush=True)
        if args.warmstart_encoder:
            pe_sd = ws.get("prompt_encoder_state_dict")
            if prompt_encoder is None or pe_sd is None:
                print("--warmstart-encoder: the checkpoint has no prompt_encoder_state_dict (or this arm has no separate "
                      "encoder); encoder left at its initialisation", flush=True)
            else:
                prompt_encoder.load_state_dict(pe_sd)
                print(f"prompt encoder restored from {args.warmstart} ({len(pe_sd)} tensors)", flush=True)
        else:
            print("prompt encoder NOT restored (no --warmstart-encoder): it starts from its seed initialisation", flush=True)
    # pre-load every molecule's pipeline/system (built once)
    envs = [first] + [load_system_policy(m, args.backend, device, shared_policy=policy, max_vocab=args.max_vocab,
                                         dynamic_resources=args.dynamic_resources, max_dim=args.max_dim,
                                         n_reps=args.n_reps, shots=args.shots, ngates=args.ngates,
                                         amplitude_damping=args.amplitude_damping,
                                         sampler_kind=sk, ibm_backend_name=args.ibm_backend_name,
                                         ibm_fake_backend=args.ibm_fake_backend, extra_overrides=args.hydra_override,
                               num_samples=args.num_samples,
                                         ibm_save_dir=args.ibm_save_dir,
                                         ibm_dynamical_decoupling=args.ibm_dynamical_decoupling,
                                         ibm_session_mode=args.ibm_session_mode, ibm_session_max_time=session_max_time,
                                         diag_backend=args.diag_backend, sbd_binary_path=args.sbd_binary_path,
                                         recover_iterations=args.recover_iterations, sbd_timeout_s=args.sbd_timeout_s,
                                         encoder=args.encoder, refine_method=args.refine_method,
                                         recover_subspace_dim=args.recover_subspace_dim)
                      for m, sk in zip(args.molecules[1:], molecule_sampler_overrides[1:])]
    names = [Path(m).stem for m in args.molecules]
    _envs_holder.extend(envs[1:])  # first already appended above, before this list comprehension

    # Progressive size curriculum (--size-curriculum): ranks molecules by qubit
    # count (norb*2, the same size axis n_qubits_now/depth_ceiling_for_qubits
    # already use below) once, up front. large_idx is the top
    # --curriculum-top-frac (default 20%) largest molecules' indices into envs,
    # rounded up so a small training set (e.g. 7 molecules here) still gets at
    # least one molecule in the "large" set rather than rounding to zero.
    n_qubits_per_env = [e["system"].norb * 2 for e in envs]
    order_by_size = sorted(range(len(envs)), key=lambda i: n_qubits_per_env[i])
    n_large = max(1, math.ceil(len(envs) * args.curriculum_top_frac))
    large_idx = set(order_by_size[-n_large:])
    if args.size_curriculum:
        print("size curriculum: large set = %s (top %.0f%% by qubit count, n_qubits=%s)" % (
            [names[i] for i in sorted(large_idx)], args.curriculum_top_frac * 100,
            [n_qubits_per_env[i] for i in sorted(large_idx)]), flush=True)

    base_lr = first["cfg"].trainer.optimizer.lr
    weight_decay = first["cfg"].trainer.optimizer.weight_decay
    # Separate param groups so --encoder-lr-mult can give the prompt encoder its
    # own LR relative to the policy's. Default 1.0 keeps every existing arm's
    # single-LR behaviour identical -- this only changes anything when requested.
    # The two modules differ by 15-100x in parameter count (0.1-3.4M encoder vs.
    # ~57M policy) and the encoder is a freshly-initialized small MLP/GNN head
    # feeding a from-scratch decoder, not a pretrained model being fine-tuned, so
    # there is no LoRA/frozen-backbone split to make here -- only the LR itself.
    param_groups = [{"params": list(policy.parameters()), "lr": base_lr}]
    if prompt_encoder is not None:
        encoder_lr = base_lr * args.encoder_lr_mult
        param_groups.append({"params": list(prompt_encoder.parameters()), "lr": encoder_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print("encoder=%s policy=%.2fM prompt_encoder=%s lr=%.2e encoder_lr_mult=%.2f" % (
        args.encoder, sum(p.numel() for p in policy.parameters()) / 1e6,
        "none (built into policy)" if prompt_encoder is None
        else "%.3fM" % (sum(p.numel() for p in prompt_encoder.parameters()) / 1e6),
        base_lr, args.encoder_lr_mult), flush=True)

    history = []
    track_f = None
    if args.track_circuits:
        import hashlib
        import os
        import platform
        import subprocess
        import sys as _sys

        def _ver(mod):
            try:
                return __import__(mod).__version__
            except Exception:
                return None

        def _sha(path):
            try:
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except Exception:
                return None
        here = Path(__file__).resolve()
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=str(here.parent.parent)).stdout.strip() or None
        except Exception:
            commit = None
        meta = {"kind": "run_meta", "argv": _sys.argv,
                "args": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v)) for k, v in vars(args).items()},
                "git_commit": commit, "python": platform.python_version(), "host": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "versions": {m: _ver(m) for m in ("torch", "numpy", "transformers", "ffsim", "qiskit", "qiskit_ibm_runtime", "pyscf")},
                "sha256": {"amortized_train.py": _sha(here),
                           "training_tracking.py": _sha(here.parent.parent / "sqd_gqe" / "gqe" / "training_tracking.py")}}
        Path(args.track_circuits).parent.mkdir(parents=True, exist_ok=True)
        track_f = open(args.track_circuits, "a")
        track_f.write(json.dumps(meta, default=str) + "\n")
        track_f.flush()
    per_mol_best = {n: float("inf") for n in names}
    prev_flat = None
    prev_flat_encoder = None
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    # Running estimate of each molecule's std(energies)==0 fallback rate (see
    # reward_degenerate above), used only by --size-bias-correction to bias the
    # round-robin schedule away from saturated molecules. EMA, not a plain
    # running mean, so it tracks the current regime rather than the whole run's
    # history -- irrelevant for the many arms that leave the flag off (default),
    # where names[j] is used unmodified and this dict is purely diagnostic.
    fallback_rate = {n: 1.0 for n in names}  # optimistic prior: assume saturated
    FALLBACK_EMA = 0.05

    # Tracks the best-so-far checkpoint SEPARATELY from the periodic one
    # (args.ckpt, overwritten every --checkpoint-every iterations): a
    # training run can find a great shallow-and-accurate circuit, then drift
    # into a worse basin later (observed directly: c2h4/ibm_kingston found
    # depth=166 at chem accuracy around iteration 30, then jumped to
    # depth~2029 -- 2.3x worse than the LUCJ1 baseline -- by iteration 37 and
    # never recovered through iteration 47). Without this, the periodic
    # checkpoint at iteration 39 silently overwrote the good result with the
    # worse one, and it was unrecoverable. "Best" = lowest real depth among
    # chem-accurate iterations; falls back to lowest |err| if none have hit
    # chem accuracy yet.
    best_ckpt_score = (False, float("inf"))  # (is_chem_accurate, depth_or_abs_err)

    # --retry-on-fail state: sticky_j is the molecule index a failed gate is
    # retrying, retries_left counts remaining attempts on it. Applied AFTER
    # round-robin/curriculum/size-bias-correction compute their own j each
    # iteration, overriding it while a retry is active; once retries_left
    # hits 0 (or the gate passes), control returns to whatever the normal
    # schedule picks, untouched. Stays None/0 when --retry-on-fail is 0, so
    # every existing arm's j sequence is unaffected.
    sticky_j = None
    retries_left = 0

    for it in range(args.iters):
        iter_t0 = time.perf_counter()
        j = it % len(envs)            # round-robin over molecules
        curriculum_phase = 1
        if args.size_curriculum and len(envs) > 1:
            # Phase boundaries as fractions of --iters, matching the proposal's
            # 100/300/400-of-400 structure but scaled to whatever --iters this
            # run actually uses (this session's runs use --iters 165, not 400).
            warmup_end = args.warmup_frac * args.iters
            focus_end = warmup_end + args.focus_frac * args.iters
            if it < warmup_end:
                curriculum_phase = 1                     # all molecules, unrestricted
            else:
                curriculum_phase = 2 if it < focus_end else 3
                # Restrict round-robin to the top --curriculum-top-frac largest
                # molecules (large_idx, computed once above from qubit count).
                # Cycling j through large_idx directly (rather than rerolling,
                # as --size-bias-correction does) keeps phase 2/3 exactly
                # round-robin WITHIN the restricted set, matching the
                # proposal's "filter training batches" wording rather than a
                # probabilistic bias.
                large_order = sorted(large_idx)
                j = large_order[it % len(large_order)]
        if args.size_bias_correction and len(envs) > 1:
            # ENCODER.md's size bias is a cross-iteration training-signal
            # asymmetry (saturated small molecules give near-zero real gradient
            # far more often), not a within-batch one -- batches here are always
            # single-molecule (see j above), so there is nothing to stratify
            # within a batch. The fix that actually applies is re-weighting which
            # molecule round-robin lands on: reroll this turn's pick, with
            # probability equal to its measured fallback rate, to the
            # round-robin's NEXT molecule instead -- a minimal, auditable
            # translation of "inverse-size curriculum sampling" onto a schedule
            # that has no notion of molecule size, only fallback history. Default
            # off (args.size_bias_correction=False) leaves j untouched, so every
            # existing arm's reproducibility is unaffected unless requested.
            if rng.random() < fallback_rate[names[j]]:
                j = (j + 1) % len(envs)
        if args.retry_on_fail > 0 and retries_left > 0:
            j = sticky_j            # override whatever schedule picked above
        env = envs[j]; name = names[j]
        cfg, system, pipeline, schedule, objective, cond = (
            env["cfg"], env["system"], env["pipeline"], env["schedule"], env["objective"], env["cond"])
        # Re-apply conditioning for this molecule. On the cross-attention arms the
        # prompt is part of the autograd graph and is consumed by backward(), so it
        # has to be rebuilt before every forward that needs a graph -- not just once
        # per iteration. set_prompt_now() below is called again inside the gradient
        # loop for exactly that reason.
        def set_prompt_now():
            if prompt_encoder is not None:
                feats = env["prompt_feats"]
                policy.set_prompt(prompt_encoder() if feats is None
                                  else prompt_encoder(feats))
            elif conditioned:
                policy.set_condition(cond.tensor(device))

        set_prompt_now()

        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        beta_sampled = float(schedule.beta())
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            # NOTE: --dynamic-resources only affects the *fixed* max_dim/shots
            # factory.py predicts once per molecule (env["pipeline"].start_dim) --
            # training uses the ordinary single-diagonalize run(), not
            # run_adaptive(). Growing every one of num_samples circuits through
            # several doubling steps EVERY iteration measured >20min for a
            # single ethanol (32q) iteration (killed by the Slurm time limit) --
            # the adaptive loop is reserved for one-shot uses that pay its cost
            # once: sqd_gqe.run's --dynamic-resources inference path and
            # build_resource_dataset.py's per-(molecule,shots) sweep.
            report = pipeline.run(state, collect_recovery_diagnostics=args.log_circuit_detail)
            seqs = state["idx"].tolist()
            # real per-sample transpiled depth (opt_level=1, cheap -- same
            # level as the diagnostic logging below) for EVERY sampled
            # circuit, not just the best one: this is what the reward's
            # real_depths term needs to actually price hardware cost, instead
            # of operator_pool.get_gate_count's cheap per-op estimate. That
            # estimate is what let ethanol's policy converge to a confidently
            # accurate but ~15000-deep circuit family -- the cheap proxy just
            # never priced the real SWAP overhead. See cost_reward.py.
            # Also computed unconditionally under --log-circuit-detail, since the
            # per-circuit report needs every circuit's real depth/2q-gates, not
            # just the reward's gated subset.
            real_depths = None
            real_two_q = None
            idle_fracs = None
            t_transpile0 = time.perf_counter()
            if (args.cost_lambda_real_depth > 0 or args.cost_lambda_coverage > 0
                    or args.cost_lambda_overshoot > 0 or args.log_circuit_detail):
                real_depths = []
                real_two_q = []
                idle_fracs = []
                for seq in seqs:
                    try:
                        hwc_i = cm.circuit_depth_on_backend(
                            env["pool"], seq, env["hw"], optimization_level=1)
                        real_depths.append(hwc_i["depth"])
                        real_two_q.append(hwc_i["two_qubit_gates"])
                        idle_fracs.append(1.0 - hwc_i["active_qubits"] / hwc_i["n_qubits"])
                    except Exception:
                        real_depths.append(0)
                        real_two_q.append(0)
                        idle_fracs.append(1.0)
            t_transpile_s = time.perf_counter() - t_transpile0
            # Fraction of each circuit's sampled-configuration weight sitting at
            # excitation rank >3 relative to HF (computed from the RAW sampled
            # counts, SqdSample.counts -- refine-independent, unlike
            # ChemistryReport.excitation_distribution which needs a merged/
            # refined state that never materializes for FCI-intractable
            # molecules). See cost_reward.py's higher_rank_fracs docstring for
            # why this term exists: a circuit can be accuracy-gated-accepted
            # while its wavefunction is ~entirely "higher"-rank, physically
            # nonsensical for a genuine low-energy state (confirmed on real
            # SnOH3-L zero-shot inference: ~97.5% higher, 0% S/D).
            # Always computed (not gated behind --cost-lambda-excitation/--log-circuit-detail):
            # pure Python over already-in-memory counts dicts, no extra diagonalization, so
            # there's no real cost reason to skip it -- and the whole point of logging
            # higher_rank_frac to history.json every iteration (below) is to let the trend be
            # MONITORED independently of whether the reward penalty is actually turned on.
            higher_rank_fracs = []
            for s in report.samples:
                if not s.counts:
                    higher_rank_fracs.append(0.0)
                    continue
                hist = cm.excitation_rank_histogram(
                    list(s.counts.keys()), env["system"].nelec,
                    weights=list(s.counts.values()))
                higher_rank_fracs.append(
                    hist["higher"] / hist["combined"] if hist["combined"] else 0.0)
            # (b) accuracy-gated cost penalty: once a circuit is within chemical
            # accuracy, penalize its 2-qubit-gate count so the policy learns
            # SHALLOW-and-accurate circuits (transfers to zero-shot). Optionally
            # (b') penalize the REAL transpiled depth (see above) -- the fix for
            # (b)'s proxy under-pricing SWAP overhead. Optionally (c) also
            # penalize the SQD subspace size actually needed (report.subspace_sizes,
            # from run_adaptive) so the policy learns circuits that are easy to
            # diagonalize, not just shallow. Optionally (d) reward LOW higher-rank
            # excitation weight, so an accuracy-gated circuit is pushed toward a
            # physically sensible S/D/T-dominated wavefunction, not just a low
            # energy number. Optionally (e) reward LOW idle-qubit fraction, so an
            # accuracy-gated circuit is pushed to actually use the molecule's
            # qubits rather than converging on a shallow local-gate-only circuit
            # that ignores most of them (see cost_reward.py's idle_fracs docstring).
            # Optionally (f) an UNGATED depth-overshoot penalty, applying even to
            # inaccurate circuits -- the real fix for lam_coverage's own failure
            # mode found in production (report Sec 26.9-26.10): every other
            # depth-related term is accuracy-gated, so on a hard molecule that
            # never reaches chem_acc_loose, nothing limited depth and the hard
            # depth_ceiling gate (below) just skipped backward() outright with
            # no corrective gradient, letting depth climb pass-over-pass with no
            # way back. n_qubits_now/depth_ceiling_now need computing here (not
            # only later, where the ORIGINAL code first needed them) since the
            # reward call happens before the best-circuit selection below.
            n_qubits_now_for_reward = env["system"].norb * 2
            depth_ceiling_for_reward = (args.depth_ceiling_max if args.depth_ceiling_max is not None
                                         else depth_ceiling_for_qubits(n_qubits_now_for_reward))
            from sqd_gqe.gqe.training_tracking import hf_fraction as _hf_fraction, spin_imbalance as _spin_imbalance
            spin_imbs = [_spin_imbalance(s_.counts, env["system"].nelec) for s_ in report.samples]
            hf_fracs = [_hf_fraction(s_.counts, env["system"].nelec) for s_ in report.samples]
            adj = cost_adjusted_energies(report.energies, seqs, env["pool"],
                                         env["casci"], args.cost_lambda,
                                         dims_used=report.subspace_sizes,
                                         lam_dim=args.cost_lambda_dim,
                                         real_depths=real_depths,
                                         lam_real_depth=args.cost_lambda_real_depth,
                                         higher_rank_fracs=higher_rank_fracs,
                                         lam_excitation=args.cost_lambda_excitation,
                                         idle_fracs=idle_fracs,
                                         lam_coverage=args.cost_lambda_coverage,
                                         depth_ceiling=depth_ceiling_for_reward,
                                         lam_overshoot=args.cost_lambda_overshoot,
                                         hf_fracs=hf_fracs, lam_hf=args.cost_lambda_hf,
                                         spin_imbalances=spin_imbs, lam_spin=args.cost_lambda_spin,
                                         chem_acc_loose=(args.gate_loose_mha * 1e-3
                                                          if args.gate_loose_mha is not None else None))
            energies = torch.tensor(adj, device=device)
            if args.reward_normalize:
                # arXiv:2607.22468 Eq.1 (ADAPT-GQE / Quantinuum+NVIDIA): a
                # reference-relative, clipped reward, R = min((E_HF - E_ref) /
                # (E_circuit - E_ref) - 1, R_max), rather than raw energy. Ha-scale
                # errors that are equally "bad" in absolute terms carry different
                # weight depending on how hard the molecule already is (how far HF
                # sits from the reference) -- this is the mapped-intent analogue of
                # this session's size-bias investigation: small molecules whose
                # HF-to-reference gap is tiny get proportionally punished less for
                # the same absolute mHa error than large molecules with a big gap,
                # which is the opposite direction of "bigger molecules get less
                # signal" and may directly counteract it. Converted to a MINIMIZED
                # quantity (this codebase's advantage() computes mean-energies, not
                # mean-reward-energies) by negating: lower "energies" is better
                # here, so store -R rather than R directly. R_max=30 matches the
                # paper's own clip value (chosen there to stop near-exact circuits
                # producing unbounded reward); its Ha scale doesn't automatically
                # transfer, so this is a starting point, not a validated constant
                # for this codebase's molecule set.
                hf_e = env["hf_energy"]
                ref_e = env["casci"]
                denom = energies - ref_e
                # denom -> 0 for a near-exact circuit is the same singularity the
                # paper's R_max clip exists to guard against; guard the division
                # itself too, since torch does not raise on divide-by-zero.
                safe_denom = torch.where(denom.abs() < 1e-8,
                                         torch.full_like(denom, 1e-8) * denom.sign().clamp(min=1e-8),
                                         denom)
                reward = torch.clamp((hf_e - ref_e) / safe_denom - 1.0, max=args.reward_normalize_max)
                energies = -reward
            if args.size_curriculum and curriculum_phase == 3:
                # Phase 3 ("harden policy generalization"): perturb only the
                # reward the OBJECTIVE sees, not report.energies/raw_best_e/
                # the gate below -- those still judge the real circuit, so
                # gate_passed/err_mha/checkpointing stay meaningful. This is
                # also where a real quantum device's own shot noise would
                # enter (a future --sampler-kind ibm_hardware run gets this
                # "for free" from real hardware; injecting it here lets the
                # ffsim/exact-simulation training runs rehearse the same
                # robustness without needing hardware access).
                energies = energies + torch.randn_like(energies) * args.curriculum_noise_std
            set_prompt_now()
            old_lp = policy.log_prob(state["idx"], schedule.beta())

        best_i = int(np.argmin(report.energies))
        best_seq = seqs[best_i]
        # raw_best_e is this iteration's best SAMPLED circuit, with no refinement
        # mixed in. best_e adds the local/global refined energies, and
        # global_refined_state persists across iterations (pipeline.py:206), so
        # best_e reports the best subspace accumulated over the whole run rather
        # than what this iteration's circuits achieved.
        #
        # That distinction decides the gate. Measured on Sn S(32q): an all-identity
        # circuit scores raw +83.8 mHa, yet best_e reports +26.0 mHa because an
        # earlier LUCJ circuit had already grown the accumulated subspace -- the
        # same +26.0 a real LUCJ circuit gets. With identical energies and only
        # depth differing, the shallower circuit wins, and the empty one is
        # shallowest. Every arm collapsed to it: 73 PASSes across five arms, all
        # with 2q=0, none with a real circuit.
        #
        # Gating on raw closes that loophole -- an empty circuit is judged on the
        # +83.8 mHa it actually produced. Note the GRPO reward was always raw
        # (cost_adjusted_energies takes report.energies), so the collapse came from
        # the gate deciding WHICH iterations get to train at all: deep circuits with
        # good raw energies were skipped, so their reward never reached backward().
        raw_best_e = min(report.energies)
        best_e = raw_best_e
        if report.global_refined is not None:
            best_e = min(best_e, report.global_refined.energy)
        if report.local_refined is not None:
            best_e = min(best_e, report.local_refined.energy)
        per_mol_best[name] = min(per_mol_best[name], best_e)

        # depth/2q-gate count of the best circuit on the target hardware, so we
        # can plot energy AND depth vs iteration -- the reward optimizes both,
        # not just energy (see docs/presentation_amortized.tex).
        t_bestdepth0 = time.perf_counter()
        try:
            # optimization_level=1, not the default 2: this SABRE-routes a
            # potentially gate-dense circuit (large auto-n_reps UCJ ops can
            # still slip through) on every single iteration -- a trend plot
            # doesn't need level-2 precision, and level-2 here is what turned
            # one training iteration into a >15min transpile (see git log).
            hwc = cm.circuit_depth_on_backend(env["pool"], best_seq, env["hw"], optimization_level=1)
            depth, two_q = hwc["depth"], hwc["two_qubit_gates"]
        except Exception:
            depth, two_q = None, None
        t_bestdepth_s = time.perf_counter() - t_bestdepth0
        dim_used = max(report.subspace_sizes) if report.subspace_sizes else None
        shots_used = int(pipeline.sampler.shots_count) if hasattr(pipeline.sampler, "shots_count") else None

        # Hard accept/reject gate on whether a gradient update happens at all --
        # distinct from cost_adjusted_energies' soft, accuracy-gated depth penalty
        # (which only nudges reward magnitude once a circuit is already accurate,
        # and never blocks an update). Depth ceiling scales with active-space size
        # since 38q/46q real-hardware batches showed a much wider depth spread than
        # 32q. Calibrated against real 38q data, not chemical accuracy (1.6mHa
        # would have rejected every iteration observed this session).
        n_qubits_now = env["system"].norb * 2
        err_mha_now = (best_e - env["casci"]) * 1e3
        raw_err_mha_now = (raw_best_e - env["casci"]) * 1e3
        depth_ceiling_now = (args.depth_ceiling_max if args.depth_ceiling_max is not None
                              else depth_ceiling_for_qubits(n_qubits_now))
        depth_ok = depth is not None and depth <= depth_ceiling_now
        # Gate on the raw error, not best_e -- see raw_best_e above. err_mha (the
        # accumulated figure) is still logged so the runs stay comparable with
        # earlier results and so the gap between the two is visible.
        energy_ok = abs(raw_err_mha_now) <= args.gate_energy_mha
        # An all-identity circuit has no two-qubit gates, and it clears the depth
        # ceiling trivially -- so depth_ok alone rewards the policy for emitting
        # nothing. Measured across every arm and seed of the 4-16q, 4-16q+CO2, and
        # 4-22q runs, 71-98% of gated-through iterations were 2q=0, i.e. the policy
        # converged on the empty circuit regardless of which encoder conditioned it.
        # Requiring at least one two-qubit gate makes "do nothing" inadmissible
        # without touching the ceiling itself: real circuits that exceed the ceiling
        # still SKIP, exactly as before.
        twoq_ok = two_q is not None and two_q > 0
        gate_passed = depth_ok and energy_ok and twoq_ok

        # --retry-on-fail bookkeeping: a fresh failure on molecule j starts a
        # retry streak (retries_left = --retry-on-fail); each subsequent
        # failure on that same molecule spends one retry; a pass or running
        # out of retries clears the streak, returning control to the normal
        # schedule (round-robin/curriculum/size-bias-correction) next
        # iteration.
        if args.retry_on_fail > 0:
            if gate_passed:
                sticky_j, retries_left = None, 0
            elif sticky_j != j:
                sticky_j, retries_left = j, args.retry_on_fail
            else:
                retries_left -= 1
                if retries_left <= 0:
                    sticky_j, retries_left = None, 0

        # Real, measurable analogue of "GRPO group reward variance" for this repo:
        # GRPOObjective (objective.py) falls back to pure best-circuit imitation
        # whenever this iteration's sampled batch has std(energies)==0, which
        # ENCODER.md documents as the mechanism behind the size bias -- small,
        # subspace-saturated molecules hit this far more often, starving them of
        # real policy-gradient signal across the training run (a cross-iteration
        # asymmetry; batches are always single-molecule here, so there is no
        # within-batch reward-domination to bin by size). Logged per iteration so
        # --size-bias-correction (below) has a real per-molecule rate to react to.
        reward_degenerate = bool(torch.std(energies) == 0)
        fallback_rate[name] += FALLBACK_EMA * (float(reward_degenerate) - fallback_rate[name])
        # Batch-level reward spread stats -- the direct analogue of "GRPO group
        # reward variance" from the size-bias proposal, computed on the SAME
        # tensor the objective actually sees (post cost_adjusted_energies, not
        # the raw report.energies), so this is exactly what feeds advantage().
        energies_mean = float(energies.mean())
        energies_std = float(energies.std())
        energies_min = float(energies.min())
        energies_max = float(energies.max())
        n_distinct_energies = int(torch.unique(energies).numel())

        last_loss = None
        diag = {}
        t_backward0 = time.perf_counter()
        if gate_passed:
            for _ in range(cfg.trainer.step_per_epoch):
                set_prompt_now()          # fresh graph through the prompt encoder
                if args.entropy_coef > 0:
                    lp, _ent_tok = policy.log_prob(state["idx"], schedule.beta(), return_entropy=True)
                else:
                    lp = policy.log_prob(state["idx"], schedule.beta())
                ctx = {"old_log_probs": old_lp, "energies": energies, "gate_seqs": state["idx"][:, 1:]}
                loss = objective(lp, ctx)
                if args.entropy_coef > 0:
                    loss = loss - args.entropy_coef * _ent_tok.mean()
                opt.zero_grad(); loss.backward()
                diag, prev_flat, prev_flat_encoder = neuron_diagnostics(
                    policy, prev_flat, prompt_encoder, prev_flat_encoder)
                diag["clip_fired"] = False
                if args.grad_clip is not None:
                    # Clips AFTER the diagnostic snapshot above, so grad_norm/
                    # grad_norm_encoder keep reporting the raw pre-clip gradient
                    # (what the run actually produced) while opt.step() applies
                    # the clipped version -- clip_params matches neuron_diagnostics'
                    # own combined-params list so the clip norm is directly
                    # comparable to grad_norm. clip_fired flags whether this
                    # iteration's raw norm actually exceeded the threshold (the
                    # GRPO-safeguard analogue of a PPO clip-fraction stat).
                    clip_params = list(policy.parameters())
                    if prompt_encoder is not None:
                        clip_params += list(prompt_encoder.parameters())
                    torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
                    diag["clip_fired"] = diag["grad_norm"] > args.grad_clip
                # grad_norm_ratio: encoder's share of the combined gradient norm --
                # the direct, per-iteration version of the 0.46-vs-38.36 (1.2%)
                # comparison neuron_diagnostics' own docstring cites, so a stalled
                # encoder (ratio -> 0 while the policy keeps moving) is visible in
                # the trend plot instead of needing a one-off manual check.
                if prompt_encoder is not None and diag["grad_norm"] > 0:
                    diag["grad_norm_ratio_encoder"] = diag["grad_norm_encoder"] / diag["grad_norm"]
                opt.step()
                last_loss = float(loss.detach())
        else:
            # no backward ran -- grad_norm/weight_delta correctly report ~0,
            # signaling a skipped (diagnostic-only) iteration in history.json.
            diag, prev_flat, prev_flat_encoder = neuron_diagnostics(
                policy, prev_flat, prompt_encoder, prev_flat_encoder)
        t_backward_s = time.perf_counter() - t_backward0
        schedule.step(energies=energies)

        # per-stage wall-clock breakdown: sample = policy rollout + real-hardware
        # sampling (pipeline.timings["sample"]); diagonalize = recover_and_diagonalize
        # (10 GPU solves/circuit when --recover-iterations>0) or one-shot diagonalize
        # (pipeline.timings["diagonalize"]); refine = local/global subspace merge
        # (pipeline.timings["refine_local"/"refine_global"]); real_depth_transpile =
        # the per-sample cost-reward transpile loop above; best_depth_transpile = the
        # single best-circuit depth check just above; backward = policy gradient step.
        t_sample_s = pipeline.timings.get("sample", 0.0)
        t_diag_s = pipeline.timings.get("diagonalize", 0.0)
        t_refine_s = pipeline.timings.get("refine_local", 0.0) + pipeline.timings.get("refine_global", 0.0)
        iter_wall_s = time.perf_counter() - iter_t0

        track_summary, track_per = None, None
        if track_f is not None:
            from sqd_gqe.gqe import training_tracking as _tt
            with torch.no_grad():
                set_prompt_now()
                _lp, _ent = policy.log_prob(state["idx"], beta_sampled, return_entropy=True)
            track_summary, track_per = _tt.iteration_tracking(
                seqs, [s.counts for s in report.samples], env["pool"], system.nelec, system.norb, report.energies,
                real_depths, real_two_q, idle_fracs, float(_ent.mean()), best_i, beta_sampled)

        circuits_detail = None
        if args.log_circuit_detail:
            circuits_detail = []
            for i, sample in enumerate(report.samples):
                diag_list = sample.recovery_diagnostics or []
                before = diag_list[0] if diag_list else None
                after = diag_list[-1] if diag_list else None
                circuits_detail.append({
                    "depth": real_depths[i] if real_depths else None,
                    "two_qubit_gates": real_two_q[i] if real_two_q else None,
                    "n_valid_before": before["n_valid"] if before else sample.n_symmetry_valid,
                    "n_valid_after": after["n_valid"] if after else None,
                    "energy_before": before["energy"] if before else None,
                    "energy_after": after["energy"] if after else sample.energy,
                    "excitation_before": before["excitation_hist"] if before else None,
                    "excitation_after": after["excitation_hist"] if after else None,
                })

        row = {"iter": it, "molecule": name, "loss": last_loss,
               "n_qubits": n_qubits_now,
               "best_energy": float(best_e), "casci": env["casci"], "ref_label": env["ref_label"],
               "err_mha": err_mha_now,
               # raw_* is what the gate now judges; keeping both makes the
               # accumulation's contribution readable as err_mha - raw_err_mha.
               "raw_best_energy": float(raw_best_e), "raw_err_mha": raw_err_mha_now,
               "best_so_far_err_mha": (per_mol_best[name] - env["casci"]) * 1e3,
               "depth": depth, "two_qubit_gates": two_q,
               "higher_rank_frac": higher_rank_fracs[best_i] if higher_rank_fracs else None,
               "idle_frac": idle_fracs[best_i] if idle_fracs else None,
               "dim_used": dim_used, "shots_used": shots_used,
               "beta": float(schedule.beta()),
               "wall_s": iter_wall_s,
               "sample_s": t_sample_s, "diagonalize_s": t_diag_s, "refine_s": t_refine_s,
               "real_depth_transpile_s": t_transpile_s, "best_depth_transpile_s": t_bestdepth_s,
               "backward_s": t_backward_s,
               "gate_passed": gate_passed,
               "gate_depth_ok": depth_ok, "gate_energy_ok": energy_ok, "gate_twoq_ok": twoq_ok,
               "reward_degenerate": reward_degenerate, "fallback_rate": fallback_rate[name],
               # batch-level reward-spread stats -- see the comment above where
               # these are computed, right after reward_degenerate.
               "energies_mean": energies_mean, "energies_std": energies_std,
               "energies_min": energies_min, "energies_max": energies_max,
               "n_distinct_energies": n_distinct_energies,
               # constant per run, logged per row anyway so every plotting/slide
               # script can read them straight off history.json with no need to
               # cross-reference the run's CLI invocation or stdout banner.
               "base_lr": base_lr, "encoder_lr_mult": args.encoder_lr_mult,
               "grad_clip": args.grad_clip, "size_bias_correction": args.size_bias_correction,
               "size_curriculum": args.size_curriculum, "curriculum_phase": curriculum_phase,
               "retry_on_fail": args.retry_on_fail, "retries_left": retries_left,
               "sampler_kind": env.get("sampler_kind", "ffsim"),
               "circuits": circuits_detail,
               "track": track_summary,
               **diag}
        history.append(row)
        if track_f is not None:
            track_f.write(json.dumps({"kind": "iteration", "iter": it, "molecule": name, "n_qubits": n_qubits_now,
                                      "sampler_kind": env.get("sampler_kind", "ffsim"), "gate_passed": gate_passed,
                                      "raw_err_mha": raw_err_mha_now, "summary": track_summary, "circuits": track_per},
                                     default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            track_f.flush()
        loss_str = f"{last_loss:.3f}" if last_loss is not None else "N/A"
        gate_reason = ""
        if not gate_passed:
            why = []
            if not depth_ok:
                why.append("deep")
            if not twoq_ok:
                why.append("empty")
            if not energy_ok:
                why.append("energy")
            gate_reason = "(" + ",".join(why) + ")"
        print(f"[AMORTIZED] it={it:3d} mol={name:6s} loss={loss_str} "
              f"err={row['err_mha']:+.3f}mHa raw={row['raw_err_mha']:+.3f}mHa "
              f"depth={depth} 2q={two_q} dim={dim_used} "
              f"grad={diag['grad_norm']:.3f} dW={diag['weight_delta']:.4f} beta={row['beta']:.3f} "
              f"gate={'PASS' if gate_passed else 'SKIP' + gate_reason} "
              f"wall={iter_wall_s:.1f}s [sample={t_sample_s:.1f}s diag={t_diag_s:.1f}s "
              f"refine={t_refine_s:.1f}s transpile={t_transpile_s + t_bestdepth_s:.1f}s "
              f"backward={t_backward_s:.1f}s]",
              flush=True)

        # best-so-far checkpoint, saved to a SEPARATE path from the periodic
        # one -- see best_ckpt_score's definition above for why this exists.
        is_acc = abs(row["err_mha"]) * 1e-3 <= CHEM_ACC_HA
        # depth can be None if circuit_depth_on_backend raised (caught above)
        # -- treat that as "no depth signal", same as not-yet-accurate, so it
        # never wins a comparison against a real depth number.
        cur_score = (is_acc and depth is not None, depth if depth is not None else abs(row["err_mha"]))
        is_better = (cur_score[0] and not best_ckpt_score[0]) or (
            cur_score[0] == best_ckpt_score[0] and cur_score[1] < best_ckpt_score[1]
        )
        if is_better:
            best_ckpt_score = cur_score
            best_ckpt_path = str(Path(args.ckpt).with_suffix("")) + "_best" + Path(args.ckpt).suffix
            if _write_resilient(lambda: torch.save(
                    {"state_dict": policy.state_dict(), "encoder": args.encoder,
                     "prompt_encoder_state_dict": (None if prompt_encoder is None
                                                  else prompt_encoder.state_dict()), "molecules": names, "iter": it,
                     "err_mha": row["err_mha"], "depth": depth, "two_qubit_gates": two_q},
                    best_ckpt_path),
                    f"best checkpoint at it={it}"):
                print(f"  -> new best checkpoint (chem_acc={is_acc}, score={cur_score[1]:.4g}) "
                      f"saved to {best_ckpt_path}", flush=True)
        # Both writes go through _write_resilient: shared HPC filesystems can
        # throw transient errors (e.g. NFS/Lustre "stale file handle", or a
        # brief remount making the parent directory itself look missing) on
        # frequent small writes to the same path -- a run that's already
        # hours in shouldn't die on a filesystem hiccup. A short retry
        # absorbs a one-off blip; checkpointing periodically (not just at the
        # end) means even a longer outage still leaves usable, recent
        # progress. (This combination is exactly what a real run needed: a
        # sustained stale-file-handle spell over several consecutive
        # iterations, then a "parent directory does not exist" RuntimeError
        # from torch.save -- not an OSError subclass, so the original
        # except OSError here didn't even catch it.)
        _write_resilient(lambda: Path(args.out).write_text(json.dumps(
            {"molecules": names, "backend": args.backend, "history": history,
             "per_mol_best_err_mha": {n: (per_mol_best[n] - e["casci"]) * 1e3
                                      for n, e in zip(names, envs)},
             "fallback_rate_final": dict(fallback_rate),
             "run_config": {
                 "encoder": args.encoder, "seed": args.seed, "max_dim": args.max_dim,
                 "iters": args.iters, "ngates": args.ngates, "num_samples": args.num_samples,
                 "recover_iterations": args.recover_iterations, "diag_backend": args.diag_backend,
                 "refine_method": args.refine_method or ("tp_ufd" if args.diag_backend == "sbd" else "gevp"),
                 "base_lr": base_lr, "encoder_lr_mult": args.encoder_lr_mult,
                 "weight_decay": weight_decay,
                 "grad_clip": args.grad_clip, "size_bias_correction": args.size_bias_correction,
                 "size_curriculum": args.size_curriculum, "warmup_frac": args.warmup_frac,
                 "focus_frac": args.focus_frac, "curriculum_top_frac": args.curriculum_top_frac,
                 "curriculum_noise_std": args.curriculum_noise_std,
                 "curriculum_large_molecules": ([names[i] for i in sorted(large_idx)]
                                                if args.size_curriculum else None),
                 "reward_normalize": args.reward_normalize,
                 "reward_normalize_max": args.reward_normalize_max,
                 "early_stop_window": args.early_stop_window,
                 "early_stop_min_improve": args.early_stop_min_improve,
                 "stopped_early": len(history) < args.iters,
                 "sampler_kind_by_molecule": {n: e.get("sampler_kind", "ffsim")
                                              for n, e in zip(names, envs)},
                 "policy_params_m": sum(p.numel() for p in policy.parameters()) / 1e6,
                 "encoder_params_m": (None if prompt_encoder is None
                                      else sum(p.numel() for p in prompt_encoder.parameters()) / 1e6),
             },
             "wall_s": time.perf_counter() - t0}, indent=2)), f"{args.out} at it={it}")
        if (it + 1) % args.checkpoint_every == 0:
            _write_resilient(lambda: torch.save(
                {"state_dict": policy.state_dict(), "encoder": args.encoder,
                     "prompt_encoder_state_dict": (None if prompt_encoder is None
                                                  else prompt_encoder.state_dict()), "molecules": names, "iter": it}, args.ckpt),
                f"checkpoint at it={it}")

        # --early-stop-window: compare two adjacent, non-overlapping trailing
        # windows of gate_passed rate (all molecules pooled) -- e.g. window=50
        # compares iterations [it-99, it-50) against [it-49, it]. Needs at
        # least 2*window iterations of history before the first check can
        # fire, so a short --iters run below that threshold always runs to
        # completion regardless of this flag.
        if args.early_stop_window is not None and it + 1 >= 2 * args.early_stop_window:
            w = args.early_stop_window
            recent = history[-w:]
            prior = history[-2 * w:-w]
            recent_rate = sum(1 for r in recent if r["gate_passed"]) / len(recent)
            prior_rate = sum(1 for r in prior if r["gate_passed"]) / len(prior)
            if recent_rate - prior_rate < args.early_stop_min_improve:
                print(f"[AMORTIZED] early stop at it={it}: trailing gate_pass rate "
                      f"{prior_rate:.3f} -> {recent_rate:.3f} "
                      f"(window={w}, min_improve={args.early_stop_min_improve})",
                      flush=True)
                break

    if _write_resilient(lambda: torch.save(
            {"state_dict": policy.state_dict(), "encoder": args.encoder,
                     "prompt_encoder_state_dict": (None if prompt_encoder is None
                                                  else prompt_encoder.state_dict()), "molecules": names, "iter": args.iters - 1}, args.ckpt),
            "final checkpoint"):
        print(f"saved checkpoint -> {args.ckpt}")
    else:
        print(f"[WARN] final checkpoint write failed -- last periodic checkpoint at "
              f"it={((args.iters - 1) // args.checkpoint_every) * args.checkpoint_every - 1} should still be on disk")
    _plot(history, names)
    _plot_depth_energy(history, names)


def _plot(history, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [r["iter"] for r in history]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # loss
    ax[0, 0].plot(it, [r["loss"] for r in history], color="#0F62FE")
    ax[0, 0].set_title("GRPO loss"); ax[0, 0].set_xlabel("iteration"); ax[0, 0].grid(alpha=.3)
    # per-molecule best-so-far accuracy
    for n in names:
        xs = [r["iter"] for r in history if r["molecule"] == n]
        ys = [abs(r["best_so_far_err_mha"]) for r in history if r["molecule"] == n]
        ax[0, 1].plot(xs, [max(y, 1e-4) for y in ys], marker=".", label=n)
    ax[0, 1].axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray")
    ax[0, 1].set_yscale("log"); ax[0, 1].set_title("best |err| vs CASCI per molecule (mHa)")
    ax[0, 1].set_xlabel("iteration"); ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=.3)
    # gradient norm
    ax[1, 0].plot(it, [r["grad_norm"] for r in history], color="#F58518")
    ax[1, 0].set_title("gradient norm (learning signal)"); ax[1, 0].set_xlabel("iteration"); ax[1, 0].grid(alpha=.3)
    # weight change
    ax[1, 1].plot(it, [r["weight_delta"] for r in history], color="#E45756")
    ax[1, 1].set_title("weight-change norm per step (how much the net moves)")
    ax[1, 1].set_xlabel("iteration"); ax[1, 1].grid(alpha=.3)
    fig.suptitle("Amortized SQD-GQE training diagnostics")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(EXP / "amortized_diagnostics.png", dpi=130); plt.close(fig)
    print(f"wrote {EXP/'amortized_diagnostics.png'}")


def _plot_depth_energy(history, names):
    """Energy error AND circuit depth vs iteration, per molecule -- the reward
    is jointly minimizing both (accuracy-gated 2q-gate/dim penalty), so a plot
    of energy alone would miss half the story. Depth here is the backend-native
    depth (Heron's cz/rz/sx/x, via circuit_depth_on_backend), i.e. what the
    cost penalty is actually computed from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, n in enumerate(names):
        rows = [r for r in history if r["molecule"] == n]
        xs = [r["iter"] for r in rows]
        err = [max(abs(r["err_mha"]), 1e-4) for r in rows]
        depth = [r["depth"] for r in rows if r["depth"] is not None]
        depth_xs = [r["iter"] for r in rows if r["depth"] is not None]
        ax[0].plot(xs, err, marker=".", ms=3, color=colors[i % 10], label=n)
        ax[1].plot(depth_xs, depth, marker=".", ms=3, color=colors[i % 10], label=n)
    ax[0].axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", lw=1)
    ax[0].set_yscale("log"); ax[0].set_ylabel("|err| vs ref (mHa)")
    ax[0].set_title("Best circuit's energy error, per iteration")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
    ax[1].set_ylabel("depth (Heron-native: cz/rz/sx/x)")
    ax[1].set_xlabel("iteration")
    ax[1].set_title("Best circuit's device-transpiled depth, per iteration")
    ax[1].grid(alpha=.3)
    fig.suptitle("Energy and depth are jointly optimized (accuracy-gated cost reward)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(EXP / "depth_energy_vs_iter.png", dpi=130); plt.close(fig)
    print(f"wrote {EXP/'depth_energy_vs_iter.png'}")


if __name__ == "__main__":
    main()
