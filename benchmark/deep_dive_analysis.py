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

"""Deep-dive diagnostic sweep on ONE trained checkpoint, for a given molecule.

NOTE on comparing these numbers to training/amortized_train.py's logged
err_mha: this script diagonalizes ONE representative circuit's samples in
isolation, on purpose (to isolate that one circuit's bitstring quality).
Training's reported best_e is the min over ALL num_samples circuits in the
batch, AND benefits from the local/global GEVP-refine merge across the whole
batch (SqdPipeline.run's refine_local/refine_global) -- so training's number
is typically noticeably better than any single circuit's raw diagonalize
here. That gap is expected, not a discrepancy.

Sub-experiments:
  A. Energy vs SQD max_dim (GPT-generated circuit, fixed shots) -- same
     circuit/samples reused across dims (no re-sampling), like
     build_resource_dataset.py. This is a property of the *diagonalization*,
     not the circuit's transpilation -- independent of optimization_level.
  B. Energy vs config-recovery depth (0/5/10 max_iterations), fixed dim --
     also diagonalization-only, independent of optimization_level.
  C. Energy: GPT-generated circuit vs standalone LUCJ circuits at several
     n_reps (shallow/medium/deep), all at the SAME fixed dim -- quantifies
     the accuracy "lag" (if any) between the policy's chosen shallow circuit
     and a deeper, more expressive hand-built ansatz alone.
  D. Depth/2q-gate count at optimization_level 2 vs 3, for the GPT circuit
     and each LUCJ-reps circuit -- a TRANSPILATION-only comparison. Listed
     separately from A-C because optimization_level cannot change A-C's
     energies (they never touch a transpiled circuit): the level only
     affects how well qiskit's SABRE router packs a circuit onto physical
     hardware, not the underlying statevector the energy is computed from.

Output: experiments/amortized/deep_dive_<tag>.json + several PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"
FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
CHEM_ACC_HA = 1.6e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--backend", default="ibm_heron_r2")
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--dims", nargs="+", type=int, default=[200, 2000, 10_000, 50_000, 100_000])
    ap.add_argument("--recovery-iters", nargs="+", type=int, default=[0, 5, 10])
    ap.add_argument("--recovery-dim", type=int, default=100_000)
    # reps=1 (the shallowest baseline) and reps=2 (the pool's actual cap) --
    # NOT 6/16: those don't exist in the current pool (LUCJ_N_REPS=2 is the
    # ceiling everywhere now), so comparing against them tells us nothing
    # about what the trained policy could realistically have picked.
    ap.add_argument("--lucj-reps", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--lucj-dim", type=int, default=100_000)
    ap.add_argument("--opt-levels", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import torch
    import numpy as np
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.core.system import FciIntractableError
    from sqd_gqe.core.configuration import ConfigurationSet
    from sqd_gqe.gqe import circuit_metrics as cm

    EXP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.molecule).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck)
    vocab = int(sd["transformer.wte.weight"].shape[0])

    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=["model=conditioned", f"+operator_pool.max_vocab={vocab}"])
    cfg.molecule = OmegaConf.load(args.molecule)
    cfg.sampler.shots = args.shots

    f = SqdFactory()
    system = f.build_system(cfg)
    hw = get_backend(args.backend)
    policy = f.build_policy(cfg).to(device)
    pipeline = f.build_pipeline(cfg, backend=hw)
    schedule = f.build_beta_schedule(cfg)
    pool = f.build_operator_pool(cfg)
    try:
        ref, ref_label = float(system.casci_energy()), "casci"
    except FciIntractableError:
        ref, ref_label = float(system.ccsd_t_energy()), "ccsd_t"
    print(f"[DEEPDIVE] {tag}: ref={ref:.6f} ({ref_label}), qubits={2*system.norb}")

    cur = policy.state_dict()
    policy.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)
    if hasattr(policy, "set_condition"):
        policy.set_condition(ConditioningContext.build(system, hw).tensor(device))

    # -- one representative GPT circuit, sampled once ------------------------
    state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
    with torch.no_grad():
        for _ in range(cfg.ngates):
            nxt = policy.act(state, schedule.beta())
            state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
        sampled_counts = pipeline.sampler.run(state)
    best_valid, best_e0, best_seq, gpt_counts = None, float("inf"), None, None
    for seq, counts in zip(state["idx"].tolist(), sampled_counts):
        valid = ConfigurationSet.from_sample_counts(counts).select_by_nelec(pipeline.nelec)
        if valid.size == 0:
            continue
        e0, _ = pipeline.diagonalize(valid.expand(max_size=args.lucj_dim, method=pipeline.expand_method))
        if e0 < best_e0:
            best_e0, best_valid, best_seq, gpt_counts = e0, valid, seq[1:], counts
    print(f"[DEEPDIVE] {tag}: GPT seq len={len(best_seq)}", flush=True)

    results = {"tag": tag, "ref": ref, "ref_label": ref_label, "qubits": 2 * system.norb, "shots": args.shots}

    # -- A: energy vs max_dim (GPT circuit) ----------------------------------
    dim_rows = []
    for d in sorted(args.dims):
        e, _ = pipeline.diagonalize(best_valid.expand(max_size=d, method=pipeline.expand_method))
        err = (e - ref) * 1e3
        dim_rows.append({"max_dim": d, "energy": float(e), "err_mha": float(err)})
        print(f"[DEEPDIVE] {tag} dim={d:>9d} E={e:.6f} err={err:+.3f}mHa", flush=True)
    results["dim_sweep"] = dim_rows

    # -- B: energy vs recovery depth (GPT circuit, fixed dim) ---------------
    rec_rows = []
    for n_iter in args.recovery_iters:
        if n_iter == 0:
            e, _ = pipeline.diagonalize(best_valid.expand(max_size=args.recovery_dim, method=pipeline.expand_method))
        else:
            e, _, _hist = pipeline._sqd.recover_and_diagonalize(
                gpt_counts, samples_per_batch=min(512, args.shots), num_batches=3, max_iterations=n_iter, seed=0)
        err = (e - ref) * 1e3
        rec_rows.append({"recovery_iters": n_iter, "energy": float(e), "err_mha": float(err)})
        print(f"[DEEPDIVE] {tag} recovery_iters={n_iter:2d} E={e:.6f} err={err:+.3f}mHa", flush=True)
    results["recovery_sweep"] = rec_rows

    # -- C: GPT circuit vs standalone LUCJ at several reps, same fixed dim ---
    import ffsim
    from sqd_gqe.gqe.sampler import FfsimSampler

    norb = system.norb
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    t2 = system.ccsd_amplitudes["t2"]
    t1 = system.ccsd_amplitudes["t1"]

    lucj_rows = []
    lucj_gate_fns = {}
    for reps in args.lucj_reps:
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=reps, interaction_pairs=(pairs_aa, pairs_ab))
        vec = ffsim.hartree_fock_state(norb, pipeline.nelec)
        vec = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=pipeline.nelec, copy=True)
        rng_seed = 1000 + reps
        samples = ffsim.sample_state_vector(vec, norb=norb, nelec=pipeline.nelec, shots=args.shots,
                                            bitstring_type=ffsim.BitstringType.INT, seed=rng_seed)
        mask = (1 << norb) - 1
        from collections import Counter
        counts = Counter()
        for s in np.asarray(samples).tolist():
            counts[(int(s) & mask, int(s) >> norb)] += 1
        valid = ConfigurationSet.from_sample_counts(dict(counts)).select_by_nelec(pipeline.nelec)
        e, _ = pipeline.diagonalize(valid.expand(max_size=args.lucj_dim, method=pipeline.expand_method)) \
            if valid.size > 0 else (float("inf"), None)
        err = (e - ref) * 1e3
        lucj_rows.append({"n_reps": int(ucj_op.n_reps), "energy": float(e), "err_mha": float(err)})
        from ffsim.qiskit import UCJOpSpinBalancedJW
        lucj_gate_fns[reps] = lambda nb, op=ucj_op: UCJOpSpinBalancedJW(op)
        print(f"[DEEPDIVE] {tag} LUCJ(reps={ucj_op.n_reps}) E={e:.6f} err={err:+.3f}mHa", flush=True)
    gpt_e_at_lucj_dim, _ = pipeline.diagonalize(best_valid.expand(max_size=args.lucj_dim, method=pipeline.expand_method))
    gpt_err_at_lucj_dim = (gpt_e_at_lucj_dim - ref) * 1e3
    print(f"[DEEPDIVE] {tag} GPT-generated @dim={args.lucj_dim} E={gpt_e_at_lucj_dim:.6f} "
          f"err={gpt_err_at_lucj_dim:+.3f}mHa", flush=True)
    results["lucj_comparison"] = {"fixed_dim": args.lucj_dim, "lucj": lucj_rows,
                                  "gpt": {"energy": float(gpt_e_at_lucj_dim), "err_mha": float(gpt_err_at_lucj_dim)}}

    # -- D: depth/2q-count at each optimization_level (transpilation only) ---
    from qiskit import QuantumCircuit
    n_qubits = 2 * norb
    depth_rows = []
    for opt in args.opt_levels:
        qc_gpt = QuantumCircuit(n_qubits)
        for idx in best_seq:
            op = pool[int(idx)]
            if getattr(op, "qiskit_gate_fn", None) is not None:
                qc_gpt.append(op.qiskit_gate_fn(norb), range(n_qubits))
        c = hw.cost(qc_gpt, optimization_level=opt)
        depth_rows.append({"circuit": "GPT-generated", "opt_level": opt, **c})
        for reps, gate_fn in lucj_gate_fns.items():
            qc = QuantumCircuit(n_qubits)
            qc.append(gate_fn(norb), range(n_qubits))
            c = hw.cost(qc, optimization_level=opt)
            depth_rows.append({"circuit": f"LUCJ(reps={reps})", "opt_level": opt, **c})
        print(f"[DEEPDIVE] {tag} opt_level={opt} depths: " +
              ", ".join(f"{r['circuit']}={r['depth']}" for r in depth_rows if r["opt_level"] == opt), flush=True)
    results["depth_by_opt_level"] = depth_rows

    Path(EXP / f"deep_dive_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {EXP / f'deep_dive_{tag}.json'}")

    # -- plots ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([r["max_dim"] for r in dim_rows], [max(abs(r["err_mha"]), 1e-4) for r in dim_rows], "o-", color="#0F62FE")
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SQD max_dim"); ax.set_ylabel(f"|error| vs {ref_label} (mHa)")
    ax.set_title(f"{tag} ({2*system.norb}q): energy vs SQD dimension (GPT circuit, fixed shots={args.shots})")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / f"deepdive_{tag}_dim_sweep.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([r["recovery_iters"] for r in rec_rows], [max(abs(r["err_mha"]), 1e-4) for r in rec_rows], "o-", color="#E45756")
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("config-recovery max_iterations (0 = raw diagonalize)")
    ax.set_ylabel(f"|error| vs {ref_label} (mHa)")
    ax.set_title(f"{tag} ({2*system.norb}q): energy vs recovery depth (dim={args.recovery_dim}, shots={args.shots})")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / f"deepdive_{tag}_recovery_sweep.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [f"LUCJ\n(reps={r['n_reps']})" for r in lucj_rows] + ["GPT\n-generated"]
    errs = [max(abs(r["err_mha"]), 1e-4) for r in lucj_rows] + [max(abs(gpt_err_at_lucj_dim), 1e-4)]
    colors = ["#F58518"] * len(lucj_rows) + ["#0F62FE"]
    ax.bar(labels, errs, color=colors)
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy")
    ax.set_yscale("log"); ax.set_ylabel(f"|error| vs {ref_label} (mHa)")
    ax.set_title(f"{tag} ({2*system.norb}q): deep LUCJ alone vs GPT-generated (both @ dim={args.lucj_dim})")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / f"deepdive_{tag}_lucj_vs_gpt.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    circuits = sorted({r["circuit"] for r in depth_rows})
    width = 0.35
    xs = np.arange(len(circuits))
    for i, opt in enumerate(args.opt_levels):
        vals = [next(r["depth"] for r in depth_rows if r["circuit"] == c and r["opt_level"] == opt) for c in circuits]
        ax.bar(xs + (i - 0.5) * width, vals, width, label=f"opt_level={opt}")
    ax.set_xticks(xs); ax.set_xticklabels(circuits, rotation=20, ha="right")
    ax.set_ylabel("transpiled depth"); ax.axhline(hw.depth_budget, ls="--", color="gray", label=f"{args.backend} budget")
    ax.set_title(f"{tag} ({2*system.norb}q): transpiled depth by optimization_level")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / f"deepdive_{tag}_depth_by_opt.png", dpi=130); plt.close(fig)

    print(f"wrote deepdive_{tag}_{{dim_sweep,recovery_sweep,lucj_vs_gpt,depth_by_opt}}.png")


if __name__ == "__main__":
    main()
