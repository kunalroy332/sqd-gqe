# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Depth-vs-accuracy trade-off: train GQE with a gated gate-cost penalty.

For each cost weight lambda, train the policy with reward = -energy above
chemical accuracy and -(energy + lambda*CX) once within it, then report the
best chemically-accurate circuit's depth/CX and its energy error. Plots
achieved error vs circuit depth, with the LUCJ and SqDRIFT/Trotter depths as
reference lines --- the concrete "GQE reaches accuracy at lower depth" claim.

Output: docs/report/depth_sweep_{tag}.png + depth_sweep_{tag}.md + .json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"
CHEM_ACC_HA = 1.6e-3


def build(cfg, device):
    from sqd_gqe.factory import SqdFactory
    f = SqdFactory()
    system = f.build_system(cfg)
    policy = f.build_policy(cfg).to(device)
    schedule = f.build_beta_schedule(cfg)
    pipeline = f.build_pipeline(cfg)
    objective = f.build_objective(cfg)
    pool = f.build_operator_pool(cfg)
    return f, system, policy, schedule, pipeline, objective, pool


def train_one(cfg, lam, device):
    """Train with a gated cost penalty; return the best chem-accurate circuit."""
    from sqd_gqe.gqe.cost_reward import cost_adjusted_energies
    from sqd_gqe.gqe import circuit_metrics as cm

    _, system, policy, schedule, pipeline, objective, pool = build(cfg, device)
    ref = float(system.casci_energy())
    opt_cfg = cfg.trainer.optimizer
    optimizer = getattr(torch.optim, opt_cfg.cls)(policy.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)

    best = None  # (depth, cx, energy) of the shallowest chem-accurate circuit
    for _ in range(cfg.trainer.max_iters):
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
            seqs = state["idx"].tolist()
            # effective reward-energy = energy + gated depth penalty
            adj = cost_adjusted_energies(report.energies, seqs, pool, ref, lam)
            energies = torch.tensor(adj, device=device)
            old_lp = policy.log_prob(state["idx"], schedule.beta())

        # track the shallowest circuit that is within chemical accuracy
        for e, seq in zip(report.energies, seqs):
            if abs(e - ref) <= CHEM_ACC_HA:
                d = cm.circuit_depth_metrics(pool, seq)
                cand = (d["depth"], d["two_qubit_gates"], float(e))
                if best is None or cand[0] < best[0]:
                    best = cand

        for _ in range(cfg.trainer.step_per_epoch):
            lp = policy.log_prob(state["idx"], schedule.beta())
            ctx = {"old_log_probs": old_lp, "energies": energies, "gate_seqs": state["idx"][:, 1:]}
            loss = objective(lp, ctx)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        schedule.step(energies=energies)

    return {"lambda": lam, "ref": ref,
            "best_depth": best[0] if best else None,
            "best_cx": best[1] if best else None,
            "best_energy": best[2] if best else None,
            "best_err_mha": (best[2] - ref) * 1e3 if best else None}


def baseline_depths(cfg, device):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("dc", str(Path(__file__).with_name("depth_compare.py")))
    dc = ilu.module_from_spec(spec); spec.loader.exec_module(dc)
    _, system, _, _, _, _, _ = build(cfg, device)
    lucj = dc.lucj_depth(system)          # (depth, cx, total)
    sqd = dc.sqdrift_depth(system, n_steps=1)
    return {"lucj_depth": lucj[0], "lucj_cx": lucj[1], "sqdrift_depth": sqd[0], "sqdrift_cx": sqd[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default=None)
    ap.add_argument("--tag", default="16q")
    ap.add_argument("--bond-length", type=float, default=1.1)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cfg_for():
        ov = [f"molecule.geometry.bond_length={args.bond_length}", f"trainer.max_iters={args.iters}"]
        if args.molecule:
            ov.insert(0, f"molecule={args.molecule}")
        with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
            return compose(config_name="default", overrides=ov)

    base = baseline_depths(cfg_for(), device)
    rows = []
    for lam in args.lambdas:
        r = train_one(cfg_for(), lam, device)
        rows.append(r)
        print(f"[DEPTH_SWEEP {args.tag}] lambda={lam:.0e}  best_depth={r['best_depth']}  "
              f"cx={r['best_cx']}  err={r['best_err_mha']} mHa", flush=True)
        (REPORT_DIR / f"depth_sweep_{args.tag}.json").write_text(
            json.dumps({"baselines": base, "sweep": rows}, indent=2))

    _plot(rows, base, args.tag)
    print(f"wrote depth_sweep_{args.tag}.png/.md/.json")


def _plot(rows, base, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["best_depth"] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if pts:
        ax.scatter([r["best_depth"] for r in pts], [abs(r["best_err_mha"]) for r in pts],
                   c="#0F62FE", zorder=3, label="SQD-GQE (gated cost penalty)")
        for r in pts:
            ax.annotate(f"λ={r['lambda']:.0e}", (r["best_depth"], abs(r["best_err_mha"])),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(base["lucj_depth"], color="#E45756", ls="--", label=f"LUCJ depth ({base['lucj_depth']})")
    ax.axvline(base["sqdrift_depth"], color="#B279A2", ls=":", label=f"SqDRIFT depth ({base['sqdrift_depth']})")
    ax.axhline(CHEM_ACC_HA * 1e3, color="gray", ls="--", label="chemical accuracy")
    ax.set_xlabel("circuit depth (transpiled)"); ax.set_ylabel("|error| vs CASCI (mHa)")
    ax.set_title(f"Depth vs accuracy — SQD-GQE {tag}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT_DIR / f"depth_sweep_{tag}.png", dpi=130); plt.close(fig)

    md = [f"# Depth vs accuracy — cost-aware SQD-GQE ({tag})", "",
          "Gated reward: -energy above chemical accuracy, then -(energy + λ·CX). "
          "Best chemically-accurate circuit per λ.", "",
          f"Baseline depths: LUCJ {base['lucj_depth']} (CX {base['lucj_cx']}), "
          f"SqDRIFT/Trotter {base['sqdrift_depth']} (CX {base['sqdrift_cx']}).", "",
          "| λ | best depth | best CX | err vs CASCI (mHa) |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['lambda']:.0e} | {r['best_depth']} | {r['best_cx']} | "
                  f"{r['best_err_mha'] if r['best_err_mha'] is None else round(r['best_err_mha'],4)} |")
    md += ["", f"![depth sweep](depth_sweep_{tag}.png)"]
    (REPORT_DIR / f"depth_sweep_{tag}.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
