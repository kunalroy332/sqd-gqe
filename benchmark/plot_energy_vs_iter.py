# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Plot energy error per training iteration, per molecule.

Companion to plot_cnot_vs_iter.py -- same run-JSON input, isolates the
accuracy trace specifically (log-scale |err| vs chemical accuracy), so
progress can be read without waiting for amortized_train.py's own
depth+energy combined plot (which only writes at the END of a run).

Reads an amortized_train.py run JSON (e.g. experiments/amortized/full9_run.json),
works on a partial/in-progress run just as well as a finished one -- no
GPU/torch required.

Output: docs/2026-07-30/figures/energy_vs_iter.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs" / "2026-07-30" / "figures"
CHEM_ACC_HA = 1.6e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--out", default=str(FIG / "energy_vs_iter.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.run_json).read_text())
    history, names = data["history"], data["molecules"]
    n_done = len(history)

    # tab10 only has 10 distinct colors -- with >10 molecules (e.g. the
    # 12-molecule round-robin) it wraps and two different molecules get the
    # same color. tab20 covers up to 20 without collision.
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, n in enumerate(names):
        rows = [r for r in history if r["molecule"] == n]
        xs = [r["iter"] for r in rows]
        ys = [max(abs(r["err_mha"]), 1e-4) for r in rows]
        ax.plot(xs, ys, marker=".", ms=3, color=colors[i % 20], label=n)
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", lw=1, label="chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("|err| vs ref (mHa)")
    ax.set_title(f"Best circuit's energy error per iteration ({data.get('backend', '')}, "
                 f"{n_done} iterations so far)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
