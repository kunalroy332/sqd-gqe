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

"""Plot a Lightning run's best-so-far energy error versus epoch.

Input may be a local ``run-*.wandb`` file/directory or JSON exported by
``read_wandb_offline.py``.  Errors are shown as absolute mHa on a logarithmic
axis, with the 1.6 mHa chemical-accuracy threshold marked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from benchmark.read_wandb_offline import read_history
except ModuleNotFoundError:  # Direct execution from the benchmark/ directory.
    from read_wandb_offline import read_history


SERIES = {
    "GQE-optimized(best_so_far)/energy - R-CASCI": "best sampled",
    "Local-refined(best_so_far)/energy - R-CASCI": "best local-refined",
    "Global-refined(best_so_far)/energy - R-CASCI": "best global-refined",
}
CHEMICAL_ACCURACY_MHA = 1.6
LOG_FLOOR_MHA = 1e-6


def load_history(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_file() and source.suffix.lower() == ".json":
        payload = json.loads(source.read_text())
        rows = payload.get("history") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Exported JSON must be a list or contain a 'history' list")
        return rows
    return read_history(source)[1]


def points(rows: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        epoch = row.get("epoch", row.get("_step"))
        value = row.get(metric)
        if not isinstance(epoch, (int, float)) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(epoch)) or not math.isfinite(float(value)):
            continue
        xs.append(float(epoch))
        ys.append(max(abs(float(value)) * 1000.0, LOG_FLOOR_MHA))
    return xs, ys


def default_output(source: str | Path) -> Path:
    path = Path(source)
    return (path if path.is_dir() else path.parent) / "casci_error_vs_epoch.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="offline W&B path or exported history JSON")
    parser.add_argument("--out", help="output PNG (default: beside the input)")
    parser.add_argument("--title", default="N2 16-qubit training convergence")
    args = parser.parse_args()

    try:
        rows = load_history(args.run)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    available = {metric: points(rows, metric) for metric in SERIES}
    available = {metric: xy for metric, xy in available.items() if xy[0]}
    if not available:
        parser.error(
            "No best-so-far R-CASCI metrics were found. Use "
            "read_wandb_offline.py --list-metrics to inspect the run."
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for metric, (xs, ys) in available.items():
        ax.plot(xs, ys, marker=".", ms=4, lw=1.3, label=SERIES[metric])
    ax.axhline(
        CHEMICAL_ACCURACY_MHA,
        color="gray",
        ls="--",
        lw=1,
        label="chemical accuracy (1.6 mHa)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|energy - R-CASCI| (mHa)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()

    output = Path(args.out) if args.out else default_output(args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
