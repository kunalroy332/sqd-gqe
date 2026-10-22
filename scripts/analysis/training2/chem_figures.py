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

"""Figures 'chemistry learnt during training' for the appendix: HOMO-LUMO content and alpha/beta balance, Stage 1 and Stage 2 of Training 2.

Usage: python3 chem_figures.py <dir with out/*.json of experiments/training2> <output figures dir>
Writes chem_stage1.png, chem_stage2_a1_46q.png (all three regimes), chem_stage2_a5_46q.png and chem_stage2_70q.png (HF penalty and tuned reward; the no-penalty runs of these lineages were stopped after 17-18 iterations). All quantities come from the per-iteration 'track' summaries
(sqd_gqe/gqe/training_tracking.py); the lines are rolling means over 10 iterations, the dots are single iterations. Training 1 has no tracking.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T2, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LIN = [("a1", "46q", "A1-46q"), ("a5", "46q", "A5-46q"), ("a1", "70q", "A1-70q"), ("a5", "70q", "A5-70q")]
ARMS = [("t2_s2", "no penalty", "#D9480F"), ("t2p_s2", "HF penalty 0.05", "#2B9348"), ("t2t_s2", "tuned reward", "#7B2CBF")]
ALPHA, BETA = "#0F62FE", "#D9480F"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.2})


def hist(n):
    try:
        return json.load(open(T2 / "out" / f"{n}.json"))["history"]
    except Exception:
        return []


def series(h, *path):
    xs, ys = [], []
    for r in h:
        t = r.get("track")
        for p in path:
            t = t.get(p) if isinstance(t, dict) else None
        if t is not None:
            xs.append(r["iter"]); ys.append(t)
    return np.array(xs), np.array(ys, float)


def roll(y, w=10):
    if len(y) < 2:
        return y
    k = min(w, len(y)); c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(len(y) - len(c), np.nan), c])


def line(ax, h, path, color, label=None, ls="-", dots=True):
    x, y = series(h, *path)
    if len(x) == 0:
        return
    if dots:
        ax.plot(x, y, ".", ms=2.5, color=color, alpha=0.25)
    ax.plot(x, roll(y), color=color, lw=2, ls=ls, label=label)


def frame(nrows, title, rowlabels):
    fig, ax = plt.subplots(nrows, 4, figsize=(15, 3.1 * nrows + 0.8), sharex="col", squeeze=False)
    for j, (_, _, lab) in enumerate(LIN):
        ax[0, j].set_title(lab, fontsize=11, fontweight="bold"); ax[-1, j].set_xlabel("iteration")
    for i, rl in enumerate(rowlabels):
        ax[i, 0].set_ylabel(rl, fontsize=9)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig, ax


# ---- Stage 1 -------------------------------------------------------------------------------------------------------------------------
fig, ax = frame(3, "Training 2, Stage 1 (simulator): HOMO-LUMO content and alpha/beta balance of what the policy generates",
                ["Givens rotations between the HOMO\nand the LUMO, per circuit", "share of the sampled determinants\nthat are HOMO$\\to$LUMO excitations",
                 "excited weight by spin channel\n(share of the sampled determinants)"])
for j, (enc, size, _) in enumerate(LIN):
    h = hist(f"t2_s1_{enc}_{size}")
    line(ax[0, j], h, ("seq_mean", "n_givens_homo_lumo"), "#0F62FE")
    line(ax[1, j], h, ("exc_mean", "valid_homo_lumo_any"), "#0F62FE")
    line(ax[2, j], h, ("exc_mean", "valid_alpha_only"), ALPHA, "alpha only", dots=False)
    line(ax[2, j], h, ("exc_mean", "valid_beta_only"), BETA, "beta only", ls="--", dots=False)
ax[2, 0].legend(frameon=False, fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / "chem_stage1.png", dpi=170); plt.close(fig)

# ---- Stage 2: A1-46q has all three regimes, the other lineages only HF penalty and tuned (the no-penalty runs were stopped after 17-18 iterations) ----
def stage2_figure(lins, arms, fname, title, ncols):
    """Rows: HOMO-LUMO rotations, HOMO->LUMO share, Hartree-Fock share, alpha-only/beta-only weight. One column per lineage; every panel has data."""
    rowl = ["Givens rotations between the HOMO\nand the LUMO, per circuit", "share of the valid shots that are\nHOMO$\\to$LUMO excitations",
            "share of the valid shots that are\nthe Hartree-Fock determinant", "weight of alpha-only (solid) and\nbeta-only (dashed) excitations\n(share of the valid shots)"]
    fig, ax = plt.subplots(4, ncols, figsize=(max(4.4 * ncols + 1.2, 7.4), 11.5), sharex="col", squeeze=False)
    for j, (enc, size, lab) in enumerate(lins):
        ax[0, j].set_title(lab, fontsize=11, fontweight="bold"); ax[-1, j].set_xlabel("iteration")
        for arm, name, col in arms:
            h = hist(f"{arm}_{enc}_{size}")
            if not h:
                raise SystemExit(f"missing run {arm}_{enc}_{size}")
            line(ax[0, j], h, ("seq_mean", "n_givens_homo_lumo"), col, name, dots=False)
            line(ax[1, j], h, ("exc_mean", "valid_homo_lumo_any"), col, dots=False)
            line(ax[2, j], h, ("exc_mean", "valid_hf"), col, dots=False)
            line(ax[3, j], h, ("exc_mean", "valid_alpha_only"), col, ls="-", dots=False)
            line(ax[3, j], h, ("exc_mean", "valid_beta_only"), col, ls="--", dots=False)
    for i, rl in enumerate(rowl):
        ax[i, 0].set_ylabel(rl, fontsize=9)
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[3, 0].plot([], [], color="gray", ls="-", label="alpha only"); ax[3, 0].plot([], [], color="gray", ls="--", label="beta only")
    ax[3, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / fname, dpi=170); plt.close(fig)


ALL3 = ARMS
TWO = [a for a in ARMS if a[0] != "t2_s2"]
stage2_figure([LIN[0]], ALL3, "chem_stage2_a1_46q.png",
              "Stage 2 (ibm_kobe), A1-46q: three reward regimes, 150 iterations each", 1)
stage2_figure([LIN[1], LIN[2], LIN[3]][:1], TWO, "chem_stage2_a5_46q.png",
              "Stage 2 (ibm_kobe), A5-46q: HF penalty and tuned reward", 1)
stage2_figure([LIN[2], LIN[3]], TWO, "chem_stage2_70q.png",
              "Stage 2 (ibm_kobe), 70-qubit lineages: HF penalty and tuned reward", 2)
print("wrote stage-1 and three stage-2 figures to", OUT)
