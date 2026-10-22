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

"""N2 PES with per-point recovery trajectories (config recovery + GEVP refinement).

For each bond length:
  * run a short GQE loop, recording the GEVP-refinement energy per GQE iteration
    (local/global merge) -- how the subspace 'recovers' the correlation energy;
  * take the best circuit's samples and run qiskit-addon-sqd configuration
    recovery, recording the energy per recovery iteration.

Produces two multi-panel grid figures (one subplot per bond length):
  docs/report/pes_recovery_gqe.png     (GEVP refinement vs GQE iteration)
  docs/report/pes_recovery_config.png  (config recovery vs recovery iteration)
plus docs/report/pes_recovery.json with all trajectories and counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"


def run_point(bond_length, iters, base_config="default"):
    from hydra import compose, initialize_config_dir
    from sqd_gqe.factory import SqdFactory

    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
        cfg = compose(config_name=base_config,
                      overrides=[f"molecule.geometry.bond_length={bond_length}"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = SqdFactory()
    system = f.build_system(cfg)
    policy = f.build_policy(cfg).to(device)
    schedule = f.build_beta_schedule(cfg)
    pipeline = f.build_pipeline(cfg)
    objective = f.build_objective(cfg)
    opt_cfg = cfg.trainer.optimizer
    optimizer = getattr(torch.optim, opt_cfg.cls)(policy.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    casci = float(system.casci_energy())

    # GQE loop, capturing the global-refined (GEVP) energy per iteration
    gqe_traj = []
    best_energy, best_state = float("inf"), None
    for _ in range(iters):
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
            energies = torch.tensor(report.energies, device=device)
            old_lp = policy.log_prob(state["idx"], schedule.beta())
        refined = report.global_refined or report.local_refined
        gqe_traj.append(float(refined.energy))
        if min(report.energies) < best_energy:
            best_energy = min(report.energies)
            best_state = state["idx"][int(np.argmin(report.energies))].tolist()
        for _ in range(cfg.trainer.step_per_epoch):
            lp = policy.log_prob(state["idx"], schedule.beta())
            ctx = {"old_log_probs": old_lp, "energies": energies, "gate_seqs": state["idx"][:, 1:]}
            loss = objective(lp, ctx)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        schedule.step(energies=energies)

    # config recovery on the best circuit's samples
    counts = pipeline.sampler.sample_state(best_state)
    _, _, recov_traj = pipeline._sqd.recover_and_diagonalize(
        counts, samples_per_batch=512, num_batches=3, max_iterations=8, seed=0
    )

    return {
        "bond_length": bond_length,
        "casci": casci,
        "gqe_refine_traj": gqe_traj,               # energy per GQE iteration (GEVP)
        "config_recovery_traj": recov_traj,        # energy per config-recovery iteration
        "best_sample_energy": best_energy,
    }


def _grid(points, key, title, fname, ylabel="energy − CASCI (mHa)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = sorted(points, key=lambda p: p["bond_length"])
    n = len(r)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for i, p in enumerate(r):
        ax = axes[i // ncol][i % ncol]
        traj = p[key]
        casci = p["casci"]
        if traj:
            y = [max(abs(e - casci) * 1e3, 1e-5) for e in traj]
            ax.plot(range(len(y)), y, marker="o", ms=3)
            ax.set_yscale("log")
        ax.axhline(1.6, ls="--", color="gray", lw=0.8)
        ax.set_title(f"R = {p['bond_length']:.2f} Å  (n={len(traj)})", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.supxlabel("iteration"); fig.supylabel("|energy − CASCI| (mHa, log)")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    fig.savefig(REPORT_DIR / fname, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--bond-lengths", nargs="+", type=float,
                    default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    points = []
    for bl in args.bond_lengths:
        p = run_point(bl, args.iters)
        points.append(p)
        print(f"[REC] R={bl:.2f}  gqe_iters={len(p['gqe_refine_traj'])}  "
              f"recovery_iters={len(p['config_recovery_traj'])}  "
              f"recovery_final={p['config_recovery_traj'][-1] if p['config_recovery_traj'] else 'NA'}", flush=True)
        (REPORT_DIR / "pes_recovery.json").write_text(json.dumps(points, indent=2))

    _grid(points, "gqe_refine_traj", "GEVP refinement per GQE iteration (per bond length)", "pes_recovery_gqe.png")
    _grid(points, "config_recovery_traj", "Configuration recovery per iteration (per bond length)", "pes_recovery_config.png")
    print(f"wrote {REPORT_DIR/'pes_recovery_gqe.png'} and pes_recovery_config.png")


if __name__ == "__main__":
    main()
