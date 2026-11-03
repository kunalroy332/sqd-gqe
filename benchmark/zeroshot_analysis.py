# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Zero-shot analysis for the amortized policy: accuracy + depth vs baselines.

For each molecule, loads the pre-tuned checkpoint, generates a circuit zero-shot
(run_molecule), and compares the generated-circuit depth/2q count against LUCJ
and SqDRIFT/Trotter transpiled to the SAME hardware backend (Heron r2). Produces
per-molecule bars + an accuracy summary, written to experiments/amortized/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP = REPO_ROOT / "experiments" / "amortized"

def baseline_depths_on_backend(system, hw):
    """LUCJ and SqDRIFT/Trotter transpiled to the given hardware backend."""
    import ffsim
    from ffsim.qiskit import UCJOpSpinBalancedJW, SimulateTrotterDoubleFactorizedJW
    from qiskit import QuantumCircuit
    import numpy as np
    norb, nelec = system.norb, tuple(system.nelec)

    def cost(qc):
        c = hw.cost(qc)
        return c["depth"], c["two_qubit_gates"]

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    ucj = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        system.ccsd_amplitudes["t2"], t1=system.ccsd_amplitudes["t1"],
        interaction_pairs=(pairs_aa, pairs_ab))
    q1 = QuantumCircuit(2 * norb); q1.append(UCJOpSpinBalancedJW(ucj), range(2 * norb))
    lucj = cost(q1)

    mol_ham = ffsim.MolecularHamiltonian(
        np.asarray(system.active_hamiltonian.one_body, float),
        np.asarray(system.active_hamiltonian.two_body, float),
        constant=float(system.active_hamiltonian.core_energy))
    dfh = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(mol_ham)
    q2 = QuantumCircuit(2 * norb); q2.append(SimulateTrotterDoubleFactorizedJW(dfh, 1.0, n_steps=1), range(2 * norb))
    sqd = cost(q2)
    return {"lucj": lucj, "sqdrift": sqd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="ibm_heron_r2")
    ap.add_argument("--molecules", nargs="+", required=True)
    ap.add_argument("--sampler-kind", default=None, choices=["ffsim", "ibm_hardware"],
                    help="sampling implementation (default: ffsim)")
    ap.add_argument("--ibm-backend-name", default=None,
                    help="real IBM backend used when --sampler-kind=ibm_hardware")
    ap.add_argument("--shots", type=int, default=None,
                    help="shots per generated circuit")
    ap.add_argument("--max-dim", type=int, default=None,
                    help="maximum SQD subspace dimension")
    ap.add_argument("--ibm-save-dir", default=None,
                    help="directory in which to retain raw counts and transpiled circuits")
    ap.add_argument("--ibm-dynamical-decoupling", action="store_true",
                    help="enable the IBM Runtime XpXm dynamical-decoupling sequence")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="skip the second molecular-system build used for LUCJ/SqDRIFT baselines")
    ap.add_argument("--output-json", default=str(EXP / "zeroshot_analysis.json"),
                    help="path for the accumulated per-molecule JSON results")
    ap.add_argument("--output-plot", default=str(EXP / "zeroshot_depth.png"),
                    help="path for the summary plot")
    args = ap.parse_args()

    output_json = Path(args.output_json)
    output_plot = Path(args.output_plot)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    from sqd_gqe.run import run_molecule
    from sqd_gqe.hw.backends import get_backend

    hw = get_backend(args.backend)
    rows = []
    for m in args.molecules:
        name = Path(m).stem
        rep = run_molecule(
            m,
            backend=args.backend,
            checkpoint=args.checkpoint,
            sampler_kind=args.sampler_kind,
            ibm_backend_name=args.ibm_backend_name,
            shots=args.shots,
            max_dim=args.max_dim,
            ibm_save_dir=args.ibm_save_dir,
            ibm_dynamical_decoupling=args.ibm_dynamical_decoupling,
        )
        if args.skip_baselines:
            base = {"lucj": (None, None), "sqdrift": (None, None)}
        else:
            # Baselines need a fresh MolecularSystem because run_molecule does
            # not expose its internal system. Keep this optional: native
            # chemistry-library failures here must not force hardware reruns.
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
            from sqd_gqe.factory import SqdFactory

            abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
            with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
                cfg = compose(config_name="default")
            cfg.molecule = OmegaConf.load(m)
            system = SqdFactory().build_system(cfg)
            base = baseline_depths_on_backend(system, hw)
        row = {"molecule": name, "qubits": rep.qubits, "mode": rep.mode,
               "err_vs_casci_mha": rep.err_vs_casci_mha, "chem_acc": rep.chemical_accuracy,
               "gqe_depth": rep.circuit_depth, "gqe_cx": rep.circuit_two_qubit_gates,
               "lucj_depth": base["lucj"][0], "lucj_cx": base["lucj"][1],
               "sqdrift_depth": base["sqdrift"][0], "sqdrift_cx": base["sqdrift"][1]}
        rows.append(row)
        baseline_summary = (
            "baselines=skipped" if args.skip_baselines else
            f"LUCJ d={base['lucj'][0]}/cx={base['lucj'][1]} "
            f"SqDRIFT d={base['sqdrift'][0]}/cx={base['sqdrift'][1]}"
        )
        print(f"[ZS] {name:8s} q={rep.qubits} err={rep.err_vs_casci_mha:+.3f}mHa "
              f"GQE d={rep.circuit_depth}/cx={rep.circuit_two_qubit_gates} "
              f"{baseline_summary}", flush=True)
        output_json.write_text(json.dumps(rows, indent=2))

    _plot(rows, args.backend, output_plot)


def _plot(rows, backend, output_plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = [r["molecule"] for r in rows]
    x = np.arange(len(names)); w = 0.25

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    have_baselines = all(r["lucj_cx"] is not None and r["sqdrift_cx"] is not None for r in rows)
    if have_baselines:
        a1.bar(x - w, [r["gqe_cx"] for r in rows], w, label="SQD-GQE", color="#0F62FE")
        a1.bar(x, [r["lucj_cx"] for r in rows], w, label="LUCJ", color="#E45756")
        a1.bar(x + w, [r["sqdrift_cx"] for r in rows], w, label="SqDRIFT", color="#B279A2")
    else:
        a1.bar(x, [r["gqe_cx"] for r in rows], 0.5, label="SQD-GQE", color="#0F62FE")
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(names, rotation=30, fontsize=8)
    a1.set_ylabel(f"two-qubit gates on {backend} (log)")
    a1.set_title("Circuit cost vs baselines" if have_baselines else "SQD-GQE circuit cost")
    a1.legend(fontsize=8)

    errs = [max(abs(r["err_vs_casci_mha"]), 1e-4) for r in rows]
    a2.bar(x, errs, color=["#0F62FE" if r["chem_acc"] else "#E45756" for r in rows])
    a2.axhline(1.6, ls="--", color="gray", label="chemical accuracy")
    a2.set_yscale("log"); a2.set_xticks(x); a2.set_xticklabels(names, rotation=30, fontsize=8)
    a2.set_ylabel("|err| vs CASCI (mHa, log)"); a2.set_title("Zero-shot accuracy"); a2.legend(fontsize=8)
    comparison = "SQD-GQE vs LUCJ/SqDRIFT" if have_baselines else "SQD-GQE"
    fig.suptitle(f"Zero-shot amortized {comparison} on {backend}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_plot, dpi=130); plt.close(fig)
    print(f"wrote {output_plot}")


if __name__ == "__main__":
    main()
