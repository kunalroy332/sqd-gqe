#!/usr/bin/env python3
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

"""Plot the four training diagnostics from amortized run 2763751 on miyabi

The script reads the saved run JSON; loading the policy checkpoint or parsing
the console log is unnecessary because all requested series are in ``history``.

Usage:
    python benchmark/plot_amortized_diagnostics.py
    python benchmark/plot_amortized_diagnostics.py --output path/to/plot.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = (
    REPO_ROOT
    / "outputs"
    / "amortized-11mol-current-2763751_opbs"
    / "11mrr_run.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments" / "amortized" / "amortized_diagnostics_2763751.png"
)
CHEMICAL_ACCURACY_MHA = 1.6
LOG_FLOOR_MHA = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss, CASCI error, gradient norm, and weight change."
    )
    parser.add_argument(
        "--run-json",
        type=Path,
        default=DEFAULT_RUN,
        help=f"amortized run JSON (default: {DEFAULT_RUN})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output PNG (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dpi", type=int, default=160, help="output resolution")
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, list[dict], list[str]]:
    with path.open() as handle:
        run = json.load(handle)

    history = run.get("history")
    molecules = run.get("molecules")
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path} has no non-empty 'history' list")
    if not isinstance(molecules, list) or not molecules:
        raise ValueError(f"{path} has no non-empty 'molecules' list")

    required = {
        "iter",
        "molecule",
        "loss",
        "best_so_far_err_mha",
        "grad_norm",
        "weight_delta",
    }
    for index, row in enumerate(history):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"history row {index} is missing {sorted(missing)}")

    non_casci = sorted(
        {row.get("ref_label") for row in history if row.get("ref_label") != "casci"}
    )
    if non_casci:
        raise ValueError(
            "CASCI was requested, but the run contains other reference labels: "
            f"{non_casci}"
        )
    return run, history, molecules


def finite_xy(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Return rows having a real finite value (loss is null on gated skips)."""
    pairs = [
        (row["iter"], float(row[key]))
        for row in history
        if row[key] is not None and math.isfinite(float(row[key]))
    ]
    return [x for x, _ in pairs], [y for _, y in pairs]


def plot_diagnostics(history: list[dict], molecules: list[str], output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    loss_ax, error_ax, grad_ax, weight_ax = axes.flat

    # Loss exists only when the hard training gate allowed a gradient update.
    loss_x, loss_y = finite_xy(history, "loss")
    loss_ax.plot(loss_x, loss_y, ".-", color="#0F62FE", linewidth=1.1, markersize=3)
    loss_ax.set_title("GRPO loss")
    loss_ax.set_ylabel("loss")

    colors = plt.get_cmap("tab20").colors
    for index, molecule in enumerate(molecules):
        rows = [row for row in history if row["molecule"] == molecule]
        x = [row["iter"] for row in rows]
        y = [max(abs(float(row["best_so_far_err_mha"])), LOG_FLOOR_MHA) for row in rows]
        error_ax.plot(
            x,
            y,
            ".-",
            linewidth=1.0,
            markersize=4,
            color=colors[index % len(colors)],
            label=molecule,
        )
    error_ax.axhline(
        CHEMICAL_ACCURACY_MHA,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="chemical accuracy (1.6 mHa)",
    )
    error_ax.set_yscale("log")
    error_ax.set_title("Best |error| vs CASCI per molecule")
    error_ax.set_ylabel("best |error| (mHa)")
    error_ax.legend(fontsize=7, ncol=2)

    grad_x, grad_y = finite_xy(history, "grad_norm")
    grad_ax.plot(grad_x, grad_y, color="#F58518", linewidth=1.1)
    grad_ax.set_title("Gradient norm")
    grad_ax.set_ylabel(r"$\Vert\nabla_\theta L\Vert_2$")

    weight_x, weight_y = finite_xy(history, "weight_delta")
    weight_ax.plot(weight_x, weight_y, color="#E45756", linewidth=1.1)
    weight_ax.set_title("Weight-change norm per step")
    weight_ax.set_ylabel(r"$\Vert\Delta\theta\Vert_2$")

    for axis in axes.flat:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.3)

    fig.suptitle("Amortized SQD-GQE training diagnostics — Miyabi job 2763751")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run, history, molecules = load_run(args.run_json)
    plot_diagnostics(history, molecules, args.output, args.dpi)

    n_losses = sum(row["loss"] is not None for row in history)
    print(f"Read {len(history)} iterations for {len(molecules)} molecules from {args.run_json}")
    print(f"Plotted {n_losses} GRPO updates; {len(history) - n_losses} gated skips had no loss")
    print(f"Backend: {run.get('backend', 'unknown')}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
