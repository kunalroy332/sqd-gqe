# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Noisy vs ideal N2 PES comparison, plus the generated-circuit comparison.

For each bond length, trains the GQE policy with a NOISY sampler (readout
bit-flip + finite shots), records the best SQD energy with and without
configuration recovery, and stores the best generated circuit (op sequence +
transpiled depth). An ideal (noiseless) PES point is computed for reference.

Outputs (docs/report/):
  * pes_noise.json                  all per-point data + best circuits
  * pes_noise_curve.png             ideal vs noisy vs noisy+recovery vs CASCI/CCSD
  * pes_noise_error.png             |error| vs CASCI (log) for each
  * circuit_compare.md              ideal-best vs noisy-best circuit (ops, depth, CX)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"
CHEM_ACC_HA = 1.6e-3


def _train(cfg, device):
    from sqd_gqe.factory import SqdFactory
    f = SqdFactory()
    system = f.build_system(cfg)
    policy = f.build_policy(cfg).to(device)
    schedule = f.build_beta_schedule(cfg)
    pipeline = f.build_pipeline(cfg)
    objective = f.build_objective(cfg)
    opt_cfg = cfg.trainer.optimizer
    optimizer = getattr(torch.optim, opt_cfg.cls)(policy.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)

    best_e, best_seq = float("inf"), None
    for _ in range(cfg.trainer.max_iters):
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
            energies = torch.tensor(report.energies, device=device)
            old_lp = policy.log_prob(state["idx"], schedule.beta())
        i = int(np.argmin(report.energies))
        if report.energies[i] < best_e:
            best_e, best_seq = report.energies[i], state["idx"][i].tolist()
        for _ in range(cfg.trainer.step_per_epoch):
            lp = policy.log_prob(state["idx"], schedule.beta())
            ctx = {"old_log_probs": old_lp, "energies": energies, "gate_seqs": state["idx"][:, 1:]}
            loss = objective(lp, ctx)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        schedule.step(energies=energies)
    return f, system, pipeline, best_e, best_seq


def run_point(bond_length, p_readout, iters):
    from hydra import compose, initialize_config_dir
    from sqd_gqe.gqe import circuit_metrics as cm

    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cfg_with(overrides):
        with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
            return compose(config_name="default", overrides=overrides)

    base = [f"molecule.geometry.bond_length={bond_length}", f"trainer.max_iters={iters}"]

    # noisy training
    cfg_n = cfg_with(base + [f"sampler.p_readout={p_readout}"])
    fn, system, pipe_n, noisy_best, noisy_seq = _train(cfg_n, device)
    casci = float(system.casci_energy()); ccsd = float(system.ccsd_energy())

    # noisy samples + configuration recovery on the best noisy circuit
    counts_n = pipe_n.sampler.sample_state(noisy_seq)
    rec_e, _, rec_hist = pipe_n._sqd.recover_and_diagonalize(
        counts_n, samples_per_batch=512, num_batches=3, max_iterations=10, seed=0
    )

    # ideal training (p_readout=0) for reference + ideal best circuit
    cfg_i = cfg_with(base + ["sampler.p_readout=0.0"])
    fi, _, pipe_i, ideal_best, ideal_seq = _train(cfg_i, device)

    # circuit metrics for the two best circuits
    pool_n, pool_i = fn.build_operator_pool(cfg_n), fi.build_operator_pool(cfg_i)
    depth_noisy = cm.circuit_depth_metrics(pool_n, noisy_seq)
    depth_ideal = cm.circuit_depth_metrics(pool_i, ideal_seq)

    return {
        "bond_length": bond_length, "casci": casci, "ccsd": ccsd,
        "ideal": float(ideal_best), "noisy": float(noisy_best), "noisy_recovered": float(rec_e),
        "recovery_iters": len(rec_hist), "recovery_traj": rec_hist,
        "ideal_seq": [int(x) for x in ideal_seq], "noisy_seq": [int(x) for x in noisy_seq],
        "ideal_ops": [pool_i[int(x)].name for x in ideal_seq],
        "noisy_ops": [pool_n[int(x)].name for x in noisy_seq],
        "depth_ideal": depth_ideal, "depth_noisy": depth_noisy,
    }


def plot(points, p_readout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = sorted(points, key=lambda p: p["bond_length"]); x = [p["bond_length"] for p in r]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, [p["casci"] for p in r], "k-", label="CASCI (exact)")
    ax.plot(x, [p["ideal"] for p in r], "o-", color="#4C78A8", label="SQD-GQE (ideal ffsim)")
    ax.plot(x, [p["noisy"] for p in r], "x--", color="#E45756", label=f"SQD-GQE (noisy, p={p_readout})")
    ax.plot(x, [p["noisy_recovered"] for p in r], "^-", color="#F58518", label="noisy + config recovery")
    ax.plot(x, [p["ccsd"] for p in r], "s:", color="green", label="CCSD")
    ax.set_xlabel("N–N bond length (Å)"); ax.set_ylabel("energy (Ha)")
    ax.set_title(f"N₂ PES — ideal vs noisy (readout p={p_readout})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "pes_noise_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fl = 1e-4
    def e(key): return [max(abs(p[key] - p["casci"]) * 1e3, fl) for p in r]
    ax.plot(x, e("ideal"), "o-", color="#4C78A8", label="ideal")
    ax.plot(x, e("noisy"), "x--", color="#E45756", label="noisy")
    ax.plot(x, e("noisy_recovered"), "^-", color="#F58518", label="noisy + recovery")
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy")
    ax.set_yscale("log"); ax.set_xlabel("N–N bond length (Å)"); ax.set_ylabel("|error| vs CASCI (mHa)")
    ax.set_title("Noise impact on accuracy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "pes_noise_error.png", dpi=130); plt.close(fig)


def circuit_md(points, p_readout):
    r = sorted(points, key=lambda p: p["bond_length"])
    lines = [
        "# Generated circuit — noisy vs ideal",
        "",
        f"Readout noise p={p_readout}. Best circuit found by the policy at each bond length.",
        "",
        "| bond (Å) | ideal depth | ideal CX | noisy depth | noisy CX | same op sequence? |",
        "|---|---|---|---|---|---|",
    ]
    for p in r:
        di, dn = p["depth_ideal"], p["depth_noisy"]
        same = "yes" if p["ideal_seq"] == p["noisy_seq"] else "no"
        lines.append(f"| {p['bond_length']:.2f} | {di['depth']} | {di['two_qubit_gates']} | "
                     f"{dn['depth']} | {dn['two_qubit_gates']} | {same} |")
    lines += ["", "## Example op sequences (equilibrium, R≈1.1 Å)"]
    eq = min(r, key=lambda p: abs(p["bond_length"] - 1.1))
    lines += [f"- **ideal:** {' → '.join(eq['ideal_ops'])}",
              f"- **noisy:** {' → '.join(eq['noisy_ops'])}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-readout", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--bond-lengths", nargs="+", type=float,
                    default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    points = []
    for bl in args.bond_lengths:
        p = run_point(bl, args.p_readout, args.iters)
        points.append(p)
        print(f"[NOISE] R={bl:.2f}  ideal={p['ideal']:.6f}  noisy={p['noisy']:.6f}  "
              f"recovered={p['noisy_recovered']:.6f}  CASCI={p['casci']:.6f}  rec_iters={p['recovery_iters']}", flush=True)
        (REPORT_DIR / "pes_noise.json").write_text(json.dumps(points, indent=2))

    plot(points, args.p_readout)
    (REPORT_DIR / "circuit_compare.md").write_text(circuit_md(points, args.p_readout))
    print(circuit_md(points, args.p_readout))
    print("wrote pes_noise_curve.png, pes_noise_error.png, circuit_compare.md")


if __name__ == "__main__":
    main()
