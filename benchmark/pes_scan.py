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

"""N2 potential energy surface scan with the SQD-GQE pipeline.

For each bond length, trains the GQE policy for a few iterations, takes the best
SQD energy, and compares to CASCI (exact in this active space) and CCSD. Plots
the PES (SQD vs CASCI vs CCSD) and the SQD error vs CASCI across the curve.
Everything is written to docs/report/.

N2 is the molecule the pipeline is trained on; scanning the bond length probes
accuracy across geometries, including the strongly-correlated stretched-bond
regime where single-reference methods (CCSD) break down.

Usage:
    python benchmark/pes_scan.py --iters 40 \
        --bond-lengths 0.9 1.0 1.1 1.2 1.3 1.4 1.6 1.8 2.0 2.2 2.4 2.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"
CHEM_ACC_HA = 1.6e-3


def train_point(bond_length, iters, base_config="default", molecule=None, iters_override=None,
                sqdrift_samples=500):
    from hydra import compose, initialize_config_dir
    from sqd_gqe.factory import SqdFactory

    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    overrides = [f"molecule.geometry.bond_length={bond_length}"]
    if molecule:
        overrides.insert(0, f"molecule={molecule}")
    if iters_override is not None:
        overrides.append(f"trainer.max_iters={iters_override}")
    with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
        cfg = compose(config_name=base_config, overrides=overrides)

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
    ccsd = float(system.ccsd_energy())
    hf = float(system.hf.e_tot)

    best = float("inf")
    for _ in range(iters):
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
            energies = torch.tensor(report.energies, device=device)
            old_lp = policy.log_prob(state["idx"], schedule.beta())
        cands = [min(report.energies)]
        if report.local_refined:
            cands.append(report.local_refined.energy)
        if report.global_refined:
            cands.append(report.global_refined.energy)
        best = min(best, min(cands))
        for _ in range(cfg.trainer.step_per_epoch):
            lp = policy.log_prob(state["idx"], schedule.beta())
            ctx = {"old_log_probs": old_lp, "energies": energies, "gate_seqs": state["idx"][:, 1:]}
            loss = objective(lp, ctx)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        schedule.step(energies=energies)

    # fixed-ansatz reference baselines through the SAME SQD pipeline
    from sqd_gqe.gqe import baselines
    from sqd_gqe.core.configuration import ConfigurationSet

    def _sqd_energy_from_counts(counts):
        cs = ConfigurationSet.from_sample_counts(counts).select_by_nelec(system.nelec).expand(cfg.sqd.max_dim)
        e, _ = pipeline._sqd.diagonalize(cs)
        return float(e)

    lucj = _sqd_energy_from_counts(baselines.lucj_counts(system, shots=cfg.sampler.shots, seed=cfg.trainer.seed))
    try:
        sqdrift = _sqd_energy_from_counts(
            baselines.sqdrift_counts(system, shots=cfg.sampler.shots, time=1.0, n_samples=sqdrift_samples, seed=cfg.trainer.seed)
        )
    except Exception as e:
        print(f"  [warn] SqDRIFT failed at R={bond_length}: {type(e).__name__}: {e}", flush=True)
        sqdrift = None

    return {"bond_length": bond_length, "hf": hf, "ccsd": ccsd, "casci": casci,
            "sqd": best, "lucj": lucj, "sqdrift": sqdrift,
            "err_vs_casci_ha": best - casci, "err_vs_casci_mha": (best - casci) * 1e3}


def plot(points, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sfx = f"_{tag}" if tag else ""
    r = sorted(points, key=lambda p: p["bond_length"])
    x = [p["bond_length"] for p in r]

    def col(key):
        return [p.get(key) for p in r]

    def err_mha(key):
        return [abs(p[key] - p["casci"]) * 1e3 if p.get(key) is not None else None for p in r]

    # PES
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, col("casci"), "k-", label="CASCI (exact)")
    ax.plot(x, col("ccsd"), "g--", marker="s", ms=4, label="CCSD")
    ax.plot(x, col("sqd"), "o-", color="#4C78A8", label="SQD-GQE")
    ax.plot(x, col("lucj"), "^-", color="#E45756", ms=5, label="LUCJ (baseline)")
    if any(v is not None for v in col("sqdrift")):
        ax.plot(x, col("sqdrift"), "v-", color="#B279A2", ms=5, label="SqDRIFT (baseline)")
    ax.plot(x, col("hf"), ":", color="gray", label="HF")
    ax.set_xlabel("N–N bond length (Å)"); ax.set_ylabel("energy (Ha)")
    ax.set_title("N₂ potential energy surface"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT_DIR / f"pes_curve{sfx}.png", dpi=130); plt.close(fig)

    # error vs CASCI (log)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    floor = 1e-4
    def clip(vals):
        return [max(v, floor) if v is not None else None for v in vals]
    ax.plot(x, clip(err_mha("sqd")), "o-", color="#4C78A8", label="|SQD-GQE − CASCI|")
    ax.plot(x, clip(err_mha("lucj")), "^-", color="#E45756", label="|LUCJ − CASCI|")
    if any(v is not None for v in col("sqdrift")):
        ax.plot(x, clip(err_mha("sqdrift")), "v-", color="#B279A2", label="|SqDRIFT − CASCI|")
    ax.plot(x, clip([abs((p["ccsd"] - p["casci"]) * 1e3) for p in r]), "s--", color="green", label="|CCSD − CASCI|")
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy (1.6 mHa)")
    ax.set_yscale("log")
    ax.set_xlabel("N–N bond length (Å)"); ax.set_ylabel("|error| vs CASCI (mHa)")
    ax.set_title("N₂ PES accuracy vs CASCI"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT_DIR / f"pes_error{sfx}.png", dpi=130); plt.close(fig)


def markdown(points, tag='', qubits=16, label='STO-3G, (10e, 8o) = 16 qubits'):
    sfx = f'_{tag}' if tag else ''
    r = sorted(points, key=lambda p: p["bond_length"])
    lines = [
        f"# N₂ potential energy surface — SQD-GQE ({qubits} qubits)",
        "",
        f"{label}. CASCI is exact in this active space.",
        "",
        "| bond (Å) | CASCI | SQD-GQE | LUCJ | SqDRIFT | CCSD | SQD err (mHa) | LUCJ err (mHa) | SqDRIFT err (mHa) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(v): return f"{v:.6f}" if v is not None else "—"
    def emh(v, c): return f"{(v-c)*1e3:+.3f}" if v is not None else "—"
    for p in r:
        c = p["casci"]
        lines.append(
            f"| {p['bond_length']:.2f} | {c:.6f} | {fmt(p['sqd'])} | {fmt(p.get('lucj'))} | "
            f"{fmt(p.get('sqdrift'))} | {fmt(p['ccsd'])} | {emh(p['sqd'], c)} | "
            f"{emh(p.get('lucj'), c)} | {emh(p.get('sqdrift'), c)} |"
        )
    worst = max(abs(p["err_vs_casci_mha"]) for p in r)
    lines += [
        "",
        f"Max |SQD − CASCI| across the curve: {worst:.4f} mHa "
        f"({'within' if worst < CHEM_ACC_HA*1e3 else 'exceeds'} chemical accuracy).",
        "",
        f"![PES](pes_curve{sfx}.png)",
        "",
        f"![error](pes_error{sfx}.png)",
    ]
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_cr", str(Path(__file__).with_name("compare_report.py")))
    _cr = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_cr)
    return "\n".join(lines) + "\n" + _cr.SYSTEM_INFO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--molecule", default=None, help="hydra molecule config name (e.g. n2_28q)")
    ap.add_argument("--tag", default="", help="filename suffix for figures/report (e.g. 28q)")
    ap.add_argument("--qubits", type=int, default=16)
    ap.add_argument("--label", default="STO-3G, (10e, 8o) = 16 qubits")
    ap.add_argument("--sqdrift-samples", type=int, default=500)
    ap.add_argument("--bond-lengths", nargs="+", type=float,
                    default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    points = []
    for bl in args.bond_lengths:
        p = train_point(bl, args.iters, molecule=args.molecule, sqdrift_samples=args.sqdrift_samples)
        points.append(p)
        print(f"[PES{sfx}] R={bl:.2f}A  SQD={p['sqd']:.6f}  CASCI={p['casci']:.6f}  "
              f"err={p['err_vs_casci_mha']:+.4f} mHa", flush=True)
        (REPORT_DIR / f"pes_points{sfx}.json").write_text(json.dumps(points, indent=2))

    plot(points, tag=args.tag)
    (REPORT_DIR / f"pes{sfx}.md").write_text(markdown(points, tag=args.tag, qubits=args.qubits, label=args.label))
    print(f"wrote pes{sfx}.md + pes_curve{sfx}.png + pes_error{sfx}.png")


if __name__ == "__main__":
    main()
