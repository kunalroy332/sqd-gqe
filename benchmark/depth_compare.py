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

"""Circuit-depth comparison: GPT-generated (SQD-GQE) vs LUCJ vs SqDRIFT/Trotter.

Builds the Jordan-Wigner qiskit circuit for each state preparation, transpiles
to a common basis {rz,ry,rx,cx,h}, and reports depth + two-qubit-gate count.
This shows the hardware-cost advantage of the learned GQE circuit over the
fixed-ansatz (LUCJ) and time-evolution (SqDRIFT) baselines.

Output: docs/report/depth_compare_{tag}.png + depth_compare_{tag}.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"
BASIS = ["rz", "ry", "rx", "cx", "h"]


def _transpile_counts(qc):
    from qiskit import transpile
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=1)
    ops = tqc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "cp"))
    return int(tqc.depth()), int(two_q), int(sum(ops.values()))


def gpt_depth(system, pool, ngates, seed=0):
    from sqd_gqe.gqe import circuit_metrics as cm
    rng = np.random.default_rng(seed)
    seq = list(rng.integers(0, len(pool), size=ngates))
    d = cm.circuit_depth_metrics(pool, seq)
    return d["depth"], d["two_qubit_gates"], d["total_gates"]


def lucj_depth(system):
    import ffsim
    from ffsim.qiskit import UCJOpSpinBalancedJW
    from qiskit import QuantumCircuit
    norb, nelec = system.norb, tuple(system.nelec)
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        system.ccsd_amplitudes["t2"], t1=system.ccsd_amplitudes["t1"],
        interaction_pairs=(pairs_aa, pairs_ab))
    qc = QuantumCircuit(2 * norb)
    qc.append(UCJOpSpinBalancedJW(op), range(2 * norb))
    return _transpile_counts(qc)


def sqdrift_depth(system, n_steps=1, time=1.0):
    """Double-factorized Trotter circuit as the depth proxy for the time-evolution
    (SqDRIFT) baseline (qDRIFT is a randomized product of the same evolution terms)."""
    import ffsim
    from ffsim.qiskit import SimulateTrotterDoubleFactorizedJW
    from qiskit import QuantumCircuit
    norb, nelec = system.norb, tuple(system.nelec)
    mol_ham = ffsim.MolecularHamiltonian(
        np.asarray(system.active_hamiltonian.one_body, float),
        np.asarray(system.active_hamiltonian.two_body, float),
        constant=float(system.active_hamiltonian.core_energy))
    dfh = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(mol_ham)
    qc = QuantumCircuit(2 * norb)
    qc.append(SimulateTrotterDoubleFactorizedJW(dfh, time, n_steps=n_steps), range(2 * norb))
    return _transpile_counts(qc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default=None)
    ap.add_argument("--tag", default="16q")
    ap.add_argument("--bond-length", type=float, default=1.1)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from sqd_gqe.factory import SqdFactory

    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    ov = [f"molecule.geometry.bond_length={args.bond_length}"]
    if args.molecule:
        ov.insert(0, f"molecule={args.molecule}")
    with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
        cfg = compose(config_name="default", overrides=ov)

    f = SqdFactory()
    system = f.build_system(cfg)
    pool = f.build_operator_pool(cfg)

    rows = []
    rows.append(("SQD-GQE (learned)", *gpt_depth(system, pool, cfg.ngates)))
    rows.append(("LUCJ (fixed ansatz)", *lucj_depth(system)))
    rows.append(("SqDRIFT / Trotter (1 step)", *sqdrift_depth(system, n_steps=1)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r[0] for r in rows]
    depths = [r[1] for r in rows]
    cxs = [r[2] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = ["#0F62FE", "#E45756", "#B279A2"]
    a1.bar(labels, depths, color=colors); a1.set_ylabel("circuit depth"); a1.set_title("Depth")
    a2.bar(labels, cxs, color=colors); a2.set_ylabel("two-qubit (CX) gates"); a2.set_title("Two-qubit gates")
    for ax in (a1, a2):
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        for i, v in enumerate([depths, cxs][0 if ax is a1 else 1]):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    n = 2 * system.norb
    fig.suptitle(f"Generated-circuit cost — {args.tag} ({n} qubits), transpiled to rz/ry/rx/cx/h")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(REPORT_DIR / f"depth_compare_{args.tag}.png", dpi=130); plt.close(fig)

    md = [f"# Generated-circuit depth: SQD-GQE vs baselines ({args.tag})",
          "",
          f"{n} qubits, transpiled to {{rz,ry,rx,cx,h}}. Lower = more hardware-friendly.",
          "",
          "| preparation | depth | two-qubit (CX) | total gates |",
          "|---|---|---|---|"]
    for name, d, cx, tot in rows:
        md.append(f"| {name} | {d} | {cx} | {tot} |")
    md += ["", f"![depth]({'depth_compare_'+args.tag}.png)"]
    (REPORT_DIR / f"depth_compare_{args.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
