# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Error vs SQD subspace dimension at equilibrium N2.

Samples once from a GQE-style circuit at a fixed geometry, then diagonalizes the
same determinant pool truncated to a range of subspace dimensions (max_dim),
tracking the energy error vs CASCI. Shows how accuracy grows with the SQD
subspace size --- the core accuracy/cost trade-off of SQD.

Output: docs/report/dim_sweep.png + docs/report/dim_sweep.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"
CHEM_ACC_HA = 1.6e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bond-length", type=float, default=1.1)   # near equilibrium
    ap.add_argument("--dims", nargs="+", type=int,
                    default=[10, 25, 50, 100, 200, 400, 800, 1600, 3136])
    ap.add_argument("--warm-iters", type=int, default=10)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.core.configuration import ConfigurationSet

    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
        cfg = compose(config_name="default",
                      overrides=[f"molecule.geometry.bond_length={args.bond_length}"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = SqdFactory()
    system = f.build_system(cfg)
    policy = f.build_policy(cfg).to(device)
    schedule = f.build_beta_schedule(cfg)
    pipeline = f.build_pipeline(cfg)
    casci = float(system.casci_energy())

    # briefly warm the policy, then grab the best circuit's raw samples
    best_e, best_seq = float("inf"), None
    for _ in range(args.warm_iters):
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
        i = int(np.argmin(report.energies))
        if report.energies[i] < best_e:
            best_e, best_seq = report.energies[i], state["idx"][i].tolist()
        schedule.step(energies=torch.tensor(report.energies, device=device))

    counts = pipeline.sampler.sample_state(best_seq)
    full = ConfigurationSet.from_sample_counts(counts).select_by_nelec(system.nelec)
    full_expanded = full.expand(max_size=10_000_000)  # symmetry-completed, untruncated
    print(f"sampled determinants (symmetry-completed): {full_expanded.size}")

    rows = []
    for d in args.dims:
        sub = ConfigurationSet(full_expanded.configs[:d])
        e, _ = pipeline._sqd.diagonalize(sub)
        err = (e - casci) * 1e3
        rows.append({"dim": min(d, full_expanded.size), "energy": float(e), "err_mha": float(err)})
        print(f"  dim={d:5d}  E={e:.6f}  err={err:+.4f} mHa", flush=True)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([r["dim"] for r in rows], [max(abs(r["err_mha"]), 1e-4) for r in rows], "o-", color="#4C78A8")
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy (1.6 mHa)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SQD subspace dimension (max_dim)")
    ax.set_ylabel("|energy − CASCI| (mHa)")
    ax.set_title(f"N₂ @ {args.bond_length} Å — accuracy vs SQD dimension"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "dim_sweep.png", dpi=130); plt.close(fig)

    md = ["# Accuracy vs SQD subspace dimension",
          "",
          f"N₂ at R={args.bond_length} Å, 16 qubits. CASCI = {casci:.6f} Ha. "
          f"Symmetry-completed sampled space = {full_expanded.size} determinants.",
          "",
          "| max_dim | energy (Ha) | err vs CASCI (mHa) |", "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dim']} | {r['energy']:.6f} | {r['err_mha']:+.4f} |")
    md += ["", "![dim sweep](dim_sweep.png)"]
    (REPORT_DIR / "dim_sweep.md").write_text("\n".join(md) + "\n")
    (REPORT_DIR / "dim_sweep.json").write_text(json.dumps(rows, indent=2))
    print("wrote dim_sweep.png + dim_sweep.md")


if __name__ == "__main__":
    main()
