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

"""Plot the SQD dimension ResourceHead's labeled dataset actually needed,
grouped by molecule and backend noise regime (ideal vs *_noisy_sim variants).

Reads experiments/amortized/resource_dataset.json (built by
build_resource_dataset.py's multi-backend sweep, see benchmark/merge_resource_shards.py)
-- no GPU/torch required, pure plotting over the already-collected labels.

Output: docs/2026-07-30/figures/resource_dim_vs_noise.png
"""

from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"
FIG = Path(__file__).resolve().parent.parent / "docs" / "2026-07-30" / "figures"

BACKEND_ORDER = ["ideal", "ibm_heron_r2_noisy_sim", "ibm_heron_r3_noisy_sim", "ibm_nighthawk_r1_noisy_sim"]
BACKEND_LABEL = {
    "ideal": "ideal",
    "ibm_heron_r2_noisy_sim": "Heron r2\n(noisy sim)",
    "ibm_heron_r3_noisy_sim": "Heron r3\n(noisy sim)",
    "ibm_nighthawk_r1_noisy_sim": "Nighthawk r1\n(noisy sim)",
}


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((EXP / "resource_dataset.json").read_text())
    molecules = sorted({r["molecule"] for r in rows}, key=lambda m: max(
        r["label_dim"] for r in rows if r["molecule"] == m))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(molecules))
    width = 0.2
    colors = ["#4C78A8", "#F58518", "#E45756", "#B279A2"]
    for i, backend in enumerate(BACKEND_ORDER):
        dims, hatches = [], []
        for m in molecules:
            match = next((r for r in rows if r["molecule"] == m and r["backend"] == backend), None)
            dims.append(match["label_dim"] if match else 0)
            hatches.append("" if (match and match["reachable"]) else "//")
        bars = ax.bar(x + (i - 1.5) * width, dims, width, label=BACKEND_LABEL[backend], color=colors[i])
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(molecules, rotation=30, ha="right")
    ax.set_ylabel("SQD subspace dim needed (log scale)")
    ax.set_title("ResourceHead training data: dim needed vs backend noise regime\n"
                  "(hatched bars = chemical accuracy NOT reached by dim growth alone at 10,000 shots)")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "resource_dim_vs_noise.png", dpi=130)
    print(f"wrote {FIG / 'resource_dim_vs_noise.png'}")


if __name__ == "__main__":
    main()
