# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Combined accuracy + SQD-dimension + wall-time vs qubit count (3-panel).

Ties the whole scaling story together: for each molecule (by qubit count),
plot (top) rollout wall-time, (mid) SQD dimension -- FCI space vs the max_dim
actually needed, (bottom) best achieved error vs CASCI/reference. Every
number here is read from a stored artifact (no hand-curated dicts -- an
earlier version of this figure had BEST_ERR/MAXDIM_USED as literal Python
dicts, which silently went stale the moment new runs landed):

  * time_vs_size.json         -- real measured wall-time + FCI dim (always
                                  present; run benchmark/time_vs_size.py first)
  * resource_dataset.json     -- minimal max_dim/shots hitting chemical
                                  accuracy, small/medium tier (build via
                                  benchmark/build_resource_dataset.py)
  * co2_dimfix.json           -- CO2's zero-shot max_dim sweep (the manual
                                  precursor to resource_dataset.json)
  * full{4,3,2}_run.json      -- amortized training run's per-molecule best
                                  error (whichever is newest/present)
  * pyridine_report.json      -- pyridine's zero-shot --dynamic-resources
                                  inference report (40q, held out)

Molecules missing a best-error source are plotted for time/FCI-dim only
(explicitly logged, not silently dropped) -- see MISSING_ACCURACY below.

Output: docs/figures/combined_vs_qubits.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "amortized"
FIG = ROOT / "docs" / "figures"
IBM, RED, ORANGE = "#0F62FE", "#E45756", "#F58518"
CHEM = 1.6  # mHa


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _best_err_from_run(run_json: dict) -> dict[str, float]:
    return {n: abs(e) for n, e in run_json.get("per_mol_best_err_mha", {}).items()}


def _best_err_from_co2_dimfix(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return abs(min(rows, key=lambda r: abs(r["err_mha"]))["err_mha"])


def main():
    tv = _load_json(EXP / "time_vs_size.json")
    if not tv:
        raise SystemExit(f"missing {EXP/'time_vs_size.json'} -- run benchmark/time_vs_size.py first")
    rows = sorted(tv, key=lambda x: x["qubits"])
    q = [r["qubits"] for r in rows]
    names = [r["molecule"] for r in rows]

    # -- best error vs reference, per molecule, from whichever real source has it --
    best_err: dict[str, float] = {}
    max_dim_used: dict[str, int] = {}

    for run_name in ("full4_run.json", "full3_run.json", "full2_run.json", "amortized_run.json"):
        run = _load_json(EXP / run_name)
        if run:
            best_err.update({n: e for n, e in _best_err_from_run(run).items() if n not in best_err})

    rd = _load_json(EXP / "resource_dataset.json")
    if rd:
        for r in rd:
            if r["reachable"]:
                max_dim_used.setdefault(r["molecule"], r["label_dim"])

    co2_sweep = _load_json(EXP / "co2_dimfix.json")
    if co2_sweep:
        e = _best_err_from_co2_dimfix(co2_sweep)
        if e is not None:
            best_err.setdefault("co2", e)
        max_dim_used.setdefault("co2", max(r["max_dim"] for r in co2_sweep))

    pyridine_report = _load_json(EXP / "pyridine_report.json")
    if pyridine_report:
        e = pyridine_report.get("err_vs_ccsd_t_mha")
        if e is not None:
            best_err.setdefault("pyridine", abs(e))
        if pyridine_report.get("dim_used"):
            max_dim_used.setdefault("pyridine", pyridine_report["dim_used"])

    missing_accuracy = [n for n in names if n not in best_err]
    if missing_accuracy:
        print(f"[combined_vs_qubits] no best-error source for: {missing_accuracy} "
              f"-- plotted for time/FCI-dim only, no accuracy point")

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)

    a1.plot(q, [r["t_rollout"] for r in rows], "o-", color=IBM)
    a1.set_yscale("log"); a1.set_ylabel("rollout wall-clock (s)")
    a1.set_title("Accuracy, SQD dimension, and wall-time vs qubit count")
    a1.grid(alpha=0.3, which="both")
    for r in rows:
        a1.annotate(r["molecule"], (r["qubits"], r["t_rollout"]), fontsize=6, xytext=(3, 2), textcoords="offset points")

    a2.plot(q, [r["fci_dim"] for r in rows], "s-", color=ORANGE, label="full FCI space")
    dims = [max_dim_used.get(n) for n in names]
    a2.plot([x for x, d in zip(q, dims) if d], [d for d in dims if d], "^--", color=IBM,
            label="min max_dim to hit chem. acc. (measured)")
    a2.set_yscale("log"); a2.set_ylabel("SQD dimension (# determinants)")
    a2.legend(fontsize=8); a2.grid(alpha=0.3, which="both")

    errs = [best_err.get(n) for n in names]
    colors = [IBM if (e is not None and e < CHEM) else RED for e in errs]
    xs_e = [x for x, e in zip(q, errs) if e is not None]
    ys_e = [e for e in errs if e is not None]
    a3.scatter(xs_e, ys_e, c=[c for c, e in zip(colors, errs) if e is not None], zorder=3)
    a3.plot(xs_e, ys_e, "-", color="#999", zorder=1)
    a3.axhline(CHEM, ls="--", color="gray", label="chemical accuracy (1.6 mHa)")
    a3.set_yscale("log"); a3.set_ylabel("best |error| vs ref (mHa)")
    a3.set_xlabel("qubits (2 x active orbitals)")
    a3.legend(fontsize=8); a3.grid(alpha=0.3, which="both")
    for n, x, e in zip(names, q, errs):
        if e is not None:
            a3.annotate(f"{max(e, 1e-4):.2g}", (x, max(e, 1e-4)), fontsize=6, xytext=(3, 2), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(FIG / "combined_vs_qubits.png", dpi=130)
    plt.close(fig)
    print("wrote combined_vs_qubits.png")


if __name__ == "__main__":
    main()
