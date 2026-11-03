# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Grid plot of the raw-shot excitation distribution of each sampled circuit.

One figure per (molecule, lineage, policy) JSON written by resample_raw_counts.py.
X axis: the circuits of the batch. Y axis: excitation rank (HF, S, D, T, Q, Higher+).
Cell colour and text: percentage of that circuit's 100,000 raw shots at that rank,
binned in 10-point steps (red = 90-100 ... deep blue = 0-10). The right-hand axis
shows each circuit's transpiled depth as a marker only; it does not enter the colours.

Usage: python plot_circuit_grid.py --in-dir <json dir> --out-dir <figure dir> [--no-hf]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# 0-10 ... 90-100 %, in the order requested.
BIN_COLORS = [
    "#0b1fa8",  # 0-10   deep blue
    "#87cefa",  # 10-20  light sky blue
    "#00e5ee",  # 20-30  cyan
    "#e000e0",  # 30-40  magenta
    "#1fa01f",  # 40-50  green
    "#90ee90",  # 50-60  light green
    "#c8e03c",  # 60-70  greenish yellow
    "#ff9a1a",  # 70-80  orange
    "#ff8fc0",  # 80-90  pink
    "#e01010",  # 90-100 red
]
BIN_LABELS = [f"{10 * i}-{10 * (i + 1)}%" for i in range(10)]
CMAP = ListedColormap(BIN_COLORS)
NORM = BoundaryNorm(np.arange(0, 110, 10), CMAP.N)
ROWS = [("HF", "HF"), ("S", "S"), ("D", "D"), ("T", "T"), ("Q", "Q"), ("higher", "Higher+")]
FONT = 22
# Per-circuit energies come from the one-shot ranking-stage diagonalization of the original
# sample-only runs. Their logs (274 lines across 41 files, all identical) read
# "exceeding the max-dim=2000 ceiling (alpha cap 45) -- truncating to the 45 most-measured
# alphas", so the subspace is the 45 most-measured alpha strings: net CI dim 45^2 = 2025.
SUBSPACE_TEXT_2000 = "up to 45 most-measured alpha strings (max-dim 2000, net CI dim up to 2025)"
SUBSPACE_TEXT_40000 = "up to 200 most-measured alpha strings (max-dim 40000, net CI dim up to 40000)"
SUBSPACE_TEXT = SUBSPACE_TEXT_2000


def make_figure(run: dict, out_path: Path, show_hf: bool = True):
    rows = [r for r in ROWS if show_hf or r[0] != "HF"]
    circs = run["circuits"]
    n = len(circs)
    # fraction (in %) of each circuit's raw shots at every rank
    frac = np.zeros((len(rows), n))
    for j, c in enumerate(circs):
        h = c["raw_hist"]
        total = h["combined"] if h["combined"] else 1.0
        for i, (key, _) in enumerate(rows):
            frac[i, j] = 100.0 * h[key] / total

    fig_h = 7.5
    fig, ax = plt.subplots(figsize=(3 * fig_h, fig_h))  # 3x wider than tall
    # rows drawn top-to-bottom as listed; clip 100% into the top bin
    ax.imshow(np.minimum(frac, 99.999), cmap=CMAP, norm=NORM, aspect="auto",
              interpolation="nearest")
    for i in range(len(rows)):
        for j in range(n):
            v = frac[i, j]
            dark = v < 30 or 40 <= v < 50 or v >= 90
            ax.text(j, i, f"{v:.0f}" if v >= 1 or v == 0 else "<1", ha="center", va="center",
                    fontsize=FONT - 4, color="white" if dark else "black", fontweight="bold")

    labels = []
    for c in circs:
        if "new_rank" in c:  # fresh 40000-dim ranking: best/2nd by energy; E+D 1/2 = energy + distinct-bitstring rank
            tag = {1: "best", 2: "2nd"}.get(c["new_rank"], "")
            old = ("E+D 1" if c["is_best"] else "") + ("E+D 2" if c["is_second"] else "")
            labels.append(f"{c['index'] + 1}\n{tag}\n{old}")
            continue
        tag = ("best" if c["is_best"] else "") + ("2nd" if c["is_second"] else "")
        labels.append(f"{c['index'] + 1}\n{tag}" if tag else f"{c['index'] + 1}\n")
    ax.set_xticks(range(n), labels, fontsize=FONT - 2)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=FONT)
    ax.set_xlabel("Sampled circuit  (best / 2nd = two lowest energies; E+D 1 / 2 = top two by energy rank + distinct-bitstring rank)"
                  if any("new_rank" in c for c in circs) else
                  "Sampled circuit  (best / 2nd = best and second-best after recovery)", fontsize=FONT)
    ax.set_ylabel("Excitation rank", fontsize=FONT)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # transpiled depth written above each column, in a row just under the title
    from matplotlib.transforms import blended_transform_factory
    row_tf = blended_transform_factory(ax.transData, ax.transAxes)  # x: column, y: axes fraction
    for j, c in enumerate(circs):
        dep = c.get("depth_live", c["cost"].get("depth"))
        ax.text(j, 1.03, f"{dep}" if dep is not None else "n/a", transform=row_tf,
                ha="center", va="bottom", fontsize=FONT, fontweight="bold", clip_on=False)
    ax.text(-0.55, 1.03, "Depth (live)", transform=row_tf, ha="right", va="bottom",
            fontsize=FONT, fontweight="bold", clip_on=False)

    # per-circuit energy from the ranking-stage diagonalization, one row above the depth row
    for j, c in enumerate(circs):
        e = c.get("ranking_energy")
        ok = e is not None and np.isfinite(e)
        ax.text(j, 1.135, f"{e:.4f}" if ok else "n/a", transform=row_tf, ha="center",
                va="bottom", fontsize=FONT - 2, fontweight="bold", clip_on=False)
    ax.text(-0.55, 1.135, "Energy (E$_h$)", transform=row_tf, ha="right", va="bottom",
            fontsize=FONT - 2, fontweight="bold", clip_on=False)

    ax.set_title(f"{run['molecule_label']}  |  {run['lineage']} {run['policy_mode']}  |  "
                 f"{run['shots']:,} raw shots per circuit\n"
                 f"Energy from diagonalization in a subspace of {SUBSPACE_TEXT}",
                 fontsize=FONT + 2, pad=100)
    if any(not (c.get("ranking_energy") is not None and np.isfinite(c["ranking_energy"])) for c in circs):
        fig.text(0.5, 0.005, f"n/a: none of the {run['shots']:,} shots has the correct electron numbers (N$_\\alpha$, N$_\\beta$), so the circuit "
                 "yields no determinant to diagonalize and its energy is undefined (not infinite).",
                 ha="center", va="bottom", fontsize=FONT - 7, style="italic")
    handles = [Patch(facecolor=c, edgecolor="k", label=l) for c, l in zip(BIN_COLORS, BIN_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=FONT - 6,
               frameon=False, bbox_to_anchor=(0.5, 0.035), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rerank-dir", help="use the 40000-dim ranking energies (energy_40000) from this directory")
    ap.add_argument("--no-hf", action="store_true", help="drop the HF row")
    args = ap.parse_args()
    files = sorted(Path(args.in_dir).glob("*.json"))
    for f in files:
        run = json.loads(f.read_text())
        if args.rerank_dir:
            global SUBSPACE_TEXT
            SUBSPACE_TEXT = SUBSPACE_TEXT_40000
            rr = json.loads((Path(args.rerank_dir) / f.name).read_text())
            for c, r in zip(run["circuits"], rr["circuits"]):
                assert c["seq"] == r["seq"], f.name
                c["ranking_energy"] = r["energy_40000"]
            fin = sorted((c for c in run["circuits"] if c["ranking_energy"] is not None), key=lambda c: c["ranking_energy"])
            for k, c in enumerate(fin):
                c["new_rank"] = k + 1
            for c in run["circuits"]:
                c.setdefault("new_rank", 99)
        out = Path(args.out_dir) / f"{f.stem}.png"
        make_figure(run, out, show_hf=not args.no_hf)
        print("wrote", out)
    print(f"{len(files)} figures")


if __name__ == "__main__":
    main()
