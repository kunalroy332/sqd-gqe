# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Plot 2-qubit gate (CNOT/CZ) count per training iteration, per molecule.

depth_energy_vs_iter.png (amortized_train.py's own plot) already covers
energy error + depth per iteration; this isolates the two_qubit_gates count
specifically (recorded per-iteration in the run JSON's history but not
plotted anywhere yet), since depth and 2q-gate count can move independently
(e.g. more parallelism vs strictly fewer gates).

Reads an amortized_train.py run JSON (e.g. experiments/amortized/full8_run.json)
-- no GPU/torch required.

Output: docs/2026-07-30/figures/cnot_vs_iter.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs" / "2026-07-30" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--out", default=str(FIG / "cnot_vs_iter.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.run_json).read_text())
    history, names = data["history"], data["molecules"]

    # tab10 only has 10 distinct colors -- with >10 molecules (e.g. the
    # 12-molecule round-robin) it wraps and two different molecules get the
    # same color. tab20 covers up to 20 without collision.
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, n in enumerate(names):
        rows = [r for r in history if r["molecule"] == n and r.get("two_qubit_gates") is not None]
        xs = [r["iter"] for r in rows]
        ys = [r["two_qubit_gates"] for r in rows]
        ax.plot(xs, ys, marker=".", ms=3, color=colors[i % 20], label=n)
    ax.set_xlabel("iteration")
    ax.set_ylabel("2-qubit gate count (device-native, Heron cz)")
    ax.set_title(f"Best circuit's 2-qubit gate count per iteration ({data.get('backend', '')})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
