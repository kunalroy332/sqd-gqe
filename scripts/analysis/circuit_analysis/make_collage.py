# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Collated excitation-grid figures: one panel per compound (smallest to largest active space) for a fixed policy.

  python3 make_collage.py            -> circuit_analysis/figures/collage_a1-46q_trained.png and collage_a1-70q_trained.png
Panels use the as-submitted (live) depth, the raw-shot excitation fractions (HF, S, D, T, Q, Higher+; % of 100,000 shots) and mark the
two lowest-ranking-energy circuits ("best", "2nd"). A compound without a finished run for the policy is shown as an empty panel saying
"queued" (run exists in the plan) or "not run".
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_circuit_grid import BIN_COLORS, BIN_LABELS, CMAP, NORM, ROWS
from tables_data import LABEL, ROOT, coverage_matrix, load

OUT = ROOT / "figures"


def panel(ax, run, rank, title):
    circs = run["circuits"]
    n = len(circs)
    frac = np.zeros((len(ROWS), n))
    for j, c in enumerate(circs):
        h = c["raw_hist"]
        t = h["combined"] or 1.0
        for i, (k, _) in enumerate(ROWS):
            frac[i, j] = 100.0 * h[k] / t
    ax.imshow(np.minimum(frac, 99.999), cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest")
    for i in range(len(ROWS)):
        for j in range(n):
            v = frac[i, j]
            dark = v < 30 or 40 <= v < 50 or v >= 90
            ax.text(j, i, f"{v:.0f}" if v >= 1 or v == 0 else "<1", ha="center", va="center", fontsize=5.2,
                    color="white" if dark else "black", fontweight="bold")
    for j, c in enumerate(circs):
        ax.text(j, -0.62, str(c["depth_live"]), ha="center", va="bottom", fontsize=5.2, fontweight="bold", clip_on=False)
    ax.text(-0.62, -0.62, "Depth", ha="right", va="bottom", fontsize=5.2, fontweight="bold", clip_on=False)
    ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS], fontsize=5.5)
    ax.set_xticks(range(n), [f"{j + 1}\n" + {1: "best", 2: "2nd"}.get(rank.get(j), "") for j in range(n)], fontsize=5.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    ax.set_title(title, fontsize=7, pad=14, fontweight="bold")


def empty(ax, title, why):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999"); sp.set_linestyle("--")
    ax.text(0.5, 0.5, why, ha="center", va="center", fontsize=7, style="italic", transform=ax.transAxes)
    ax.set_title(title, fontsize=7, pad=14, fontweight="bold")


def collage(lineage, policy="trained"):
    d = load()
    cm = coverage_matrix(d)
    mols = sorted(LABEL, key=lambda k: (LABEL[k][1], LABEL[k][2]))
    items = []
    for mk in mols:
        st = cm[mk][(policy, lineage)]
        if st == "not run":
            continue
        items.append((mk, st))
    ncol = 2
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.0 * nrow + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (mk, st) in enumerate(items):
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        title = f"{LABEL[mk][0]}"
        if st == "done":
            tag = next(t for t, r in d.items() if r["mol"] == mk and r["lineage"] == lineage and r["policy"] == policy)
            run = json.loads((ROOT / "data" / f"{tag}.json").read_text())
            rr = json.loads((ROOT / "rerank" / f"{tag}.json").read_text())
            fin = sorted((c for c in rr["circuits"] if c["energy_40000"] is not None), key=lambda c: c["energy_40000"])
            rank = {c["index"]: i + 1 for i, c in enumerate(fin[:2])}
            panel(ax, run, rank, title)
        else:
            empty(ax, title, "queued: run not yet performed\n(cluster maintenance)")
    handles = [Patch(facecolor=c, edgecolor="k", label=l) for c, l in zip(BIN_COLORS, BIN_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=5.5, frameon=False, handlelength=1.1, columnspacing=0.8)
    fig.suptitle(f"{lineage.upper()} {policy}: raw-shot excitation composition of the 10 sampled circuits per compound "
                 "(% of 100,000 shots; depth = as submitted to ibm_kobe)", fontsize=7.5, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.975), h_pad=1.6, w_pad=0.8)
    out = OUT / f"collage_{lineage}_{policy}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, f"({len(items)} panels)")


if __name__ == "__main__":
    for lin in ("a1-46q", "a1-70q"):
        collage(lin)
