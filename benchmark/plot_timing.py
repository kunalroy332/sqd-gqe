# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Per-step timing breakdown: 16 vs 28 qubits.

16q numbers are the measured means from the 100-iteration run log
(ffsim_16q.json); 28q numbers are from the smoke-test measurement (one
pipeline batch + CASCI at 28q). Produces a grouped bar chart and a table.

Output: docs/report/timing_breakdown.png + timing_breakdown.md
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent / "docs" / "report"

# measured 28q per-batch step times (s), from the smoke test (M-circuit batch)
T28 = {"sample": 1.42, "diagonalize": 87.62, "refine": 30.37}
CASCI28 = 89.2


def mean16():
    d = json.loads((REPORT_DIR / "ffsim_16q.json").read_text())
    h = d["history"]
    def m(key):
        v = [r["timings"].get(key, 0) for r in h if key in r["timings"]]
        return sum(v) / len(v) if v else 0.0
    return {
        "sample": m("sample"),
        "diagonalize": m("diagonalize"),
        "refine": m("refine_local") + m("refine_global"),
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t16 = mean16()
    steps = ["sample", "diagonalize", "refine"]
    labels = ["ffsim sampling", "SQD diagonalize", "GEVP refinement"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(steps))
    w = 0.38
    b16 = [t16[s] for s in steps]
    b28 = [T28[s] for s in steps]
    ax.bar([i - w / 2 for i in x], b16, w, label="16 qubits", color="#4C78A8")
    ax.bar([i + w / 2 for i in x], b28, w, label="28 qubits", color="#0F62FE")
    for i, v in enumerate(b16):
        ax.text(i - w / 2, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(b28):
        ax.text(i + w / 2, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("time per step (s, log)")
    ax.set_title("Per-step timing: 16 vs 28 qubits (log scale)")
    ax.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "timing_breakdown.png", dpi=130); plt.close(fig)

    md = [
        "# Per-step timing: 16 vs 28 qubits",
        "",
        "16q = mean over the 100-iteration run; 28q = smoke-test measurement (per M-circuit batch).",
        "",
        "| step | 16q (s) | 28q (s) | slowdown |",
        "|---|---|---|---|",
    ]
    for s, lab in zip(steps, labels):
        md.append(f"| {lab} | {t16[s]:.3f} | {T28[s]:.1f} | {T28[s]/max(t16[s],1e-9):.0f}× |")
    md += [
        f"| CASCI reference (per geometry) | ~0.3 | {CASCI28:.0f} | — |",
        "",
        "SQD diagonalization dominates and scales steeply with the active space "
        f"(~{T28['diagonalize']/max(t16['diagonalize'],1e-9):.0f}× from 16q to 28q), because the sampled "
        "determinant subspaces are far larger. Sampling stays cheap (ffsim). This is why the 28q PES used a reduced "
        "iteration budget.",
        "",
        "![timing](timing_breakdown.png)",
    ]
    (REPORT_DIR / "timing_breakdown.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
