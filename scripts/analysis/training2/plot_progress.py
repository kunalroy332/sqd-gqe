# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Progress plots of Training 2 (Stage 1 while it runs; Stage 2 after) against Training 1.

Usage: python3 plot_progress.py <dir with out/*.json copied from experiments/training2> <training-1 dir with the production histories> <outdir>
Training 2 histories carry the per-iteration 'track' summary (sqd_gqe/gqe/training_tracking.py). Training 1 has no tracking, only loss, raw error,
gate pass and depth. The curriculum cycles through the molecules in a fixed order, so raw curves are jagged; the thick line is a rolling mean over one cycle.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T2, T1, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
LIN = [("a1", "46q", "A1-46q"), ("a5", "46q", "A5-46q"), ("a1", "70q", "A1-70q"), ("a5", "70q", "A5-70q")]
C1, C2, GRAY = "#0F62FE", "#D9480F", "#8A8A8A"
CYCLE = 14
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.2})


def hist(path):
    return json.load(open(path))["history"] if Path(path).exists() else []


def roll(x, w=CYCLE):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x
    k = min(w, len(x))
    c = np.convolve(np.nan_to_num(x), np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(len(x) - len(c), np.nan), c])


def series(h, key, adm=False):
    its = [r["iter"] for r in h if (r.get("gate_passed") or not adm) and r.get(key) is not None]
    return its, [r[key] for r in h if (r.get("gate_passed") or not adm) and r.get(key) is not None]


def trk(h, *path):
    xs, ys = [], []
    for r in h:
        t = r.get("track")
        if not t:
            continue
        for p in path:
            t = t.get(p) if isinstance(t, dict) else None
        if t is not None:
            xs.append(r["iter"]); ys.append(t)
    return xs, ys


# ---- figure 1: training metrics, Training 2 against Training 1 --------------------------------------------------------------------
fig, ax = plt.subplots(4, 4, figsize=(15, 11), sharex="col")
for j, (enc, size, lab) in enumerate(LIN):
    h2 = hist(T2 / "out" / f"t2_s1_{enc}_{size}.json")
    h1 = hist(T1 / f"production_coverage{size[:2]}_{enc}_v3_s1234.json")
    n = len(h2)
    h1 = h1[:max(n, 1)] if n else h1
    ax[0, j].set_title(f"{lab}  (Training 2: {n} iterations so far)", fontsize=10, fontweight="bold")
    for h, c, nm in ((h1, GRAY, "Training 1"), (h2, C2, "Training 2")):
        if not h:
            continue
        x, y = series(h, "loss", adm=True); ax[0, j].plot(x, y, ".", ms=3, color=c, alpha=0.5); ax[0, j].plot(x, roll(y, 10), color=c, lw=2, label=nm)
        x, y = series(h, "raw_err_mha", adm=True); ax[1, j].plot(x, y, ".", ms=3, color=c, alpha=0.5); ax[1, j].plot(x, roll(y, 10), color=c, lw=2)
        gp = [1.0 if r["gate_passed"] else 0.0 for r in h]
        ax[2, j].plot([r["iter"] for r in h], [100 * v for v in roll(gp, 10)], color=c, lw=2)
        x, y = series(h, "depth"); ax[3, j].plot(x, y, ".", ms=3, color=c, alpha=0.6)
    ax[1, j].set_yscale("symlog", linthresh=10); ax[1, j].set_yticks([-100, -10, 0, 10, 30, 100, 300]); ax[1, j].set_yticklabels(["-100", "-10", "0", "10", "30", "100", "300"], fontsize=7); ax[1, j].tick_params(labelleft=True)
    ax[3, j].set_yscale("log")
    ax[3, j].set_xlabel("iteration")
ax[0, 0].set_ylabel("loss (admitted)"); ax[1, 0].set_ylabel("raw error (mHa, admitted)"); ax[2, 0].set_ylabel("gate pass (%, rolling 10)")
ax[3, 0].set_ylabel("depth of the best circuit")
ax[0, 0].legend(frameon=False)
fig.suptitle("Stage 1 (ffsim): Training 1 (gray, same number of iterations) against Training 2 (orange)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "t2_stage1_progress_training.png", dpi=150)
plt.close(fig)

# ---- figure 2: what the policy generates and how the sampled states look (Training 2 only) ----------------------------------------
rows = [
    ("policy effective\nvocabulary (of 110)", [("policy_effective_vocab",)], None),
    ("fraction of\nidentity tokens", [("seq_mean", "frac_identity")], None),
    ("Givens rotations\nper circuit, by role", [("seq_mean", "n_givens_homo_lumo"), ("seq_mean", "n_givens_occ_occ"), ("seq_mean", "n_givens_virt_virt")],
     ["HOMO-LUMO", "occupied-occupied", "virtual-virtual"]),
    ("gates that can change\nthe HF state (per circuit)", [("seq_mean", "n_hf_changing_gates")], None),
    ("sampled states\n(right-spin-sector shots)", [("exc_mean", "valid_alpha_only"), ("exc_mean", "valid_beta_only"), ("exc_mean", "valid_alpha_and_beta")],
     ["alpha -> alpha", "beta -> beta", "alpha and beta"]),
    ("weight on HOMO->LUMO\nexcitations", [("exc_mean", "valid_homo_lumo_alpha"), ("exc_mean", "valid_homo_lumo_beta"), ("exc_mean", "valid_homo_lumo_alpha_beta")],
     ["alpha", "beta", "alpha and beta"]),
]
fig, ax = plt.subplots(len(rows), 4, figsize=(15, 15), sharex="col")
cols = [C1, C2, "#2B9348", "#7B2CBF"]
for j, (enc, size, lab) in enumerate(LIN):
    h2 = hist(T2 / "out" / f"t2_s1_{enc}_{size}.json")
    ax[0, j].set_title(f"{lab}  ({len(h2)} iterations)", fontsize=10, fontweight="bold")
    for i, (title, paths, names) in enumerate(rows):
        for k, p in enumerate(paths):
            x, y = trk(h2, *p)
            if not x:
                continue
            ax[i, j].plot(x, y, ".", ms=3, color=cols[k], alpha=0.35)
            ax[i, j].plot(x, roll(y), color=cols[k], lw=2, label=names[k] if names else None)
        if names and j == 0:
            ax[i, j].legend(frameon=False, fontsize=7)
        if j == 0:
            ax[i, j].set_ylabel(title, fontsize=8)
    ax[-1, j].set_xlabel("iteration")
fig.suptitle("Training 2, Stage 1 (ffsim): what the policy generates and how the sampled states look\n"
             "points: iterations; line: rolling mean over one 14-molecule cycle; all shots are in the right spin sector by construction",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "t2_stage1_progress_tracking.png", dpi=150)
print("wrote", OUT / "t2_stage1_progress_training.png", "and", OUT / "t2_stage1_progress_tracking.png")


# ---- Stage 2 (ibm_kobe) ------------------------------------------------------------------------------------------------------------
S2_T1 = {"A1-46q": ["hw_continue_46q_a1_s1234_v4"], "A1-70q": ["kawasaki_hw_70q_a1_s1234_v5"],
         "A5-46q": ["hw_continue_46q_a5_s1234_v1", "hw_continue_46q_a5_s1234_v2", "hw_continue_46q_a5_s1234_v3"],
         "A5-70q": ["kawasaki_hw_70q_a5_s1234_v4"]}


def t1_stage2(lab):
    rows, off = [], 0
    for f in S2_T1[lab]:
        h = hist(T1 / f"{f}.json")
        rows += [dict(r, iter=off + r["iter"]) for r in h]
        off += len(h)
    return rows


H2 = {lab: hist(T2 / "out" / f"t2_s2_{enc}_{size}.json") for enc, size, lab in LIN}
HP = {lab: hist(T2 / "out" / f"t2p_s2_{enc}_{size}.json") for enc, size, lab in LIN}
C3 = "#2B9348"
fig, ax = plt.subplots(4, 4, figsize=(15, 11), sharex="col")
for j, (enc, size, lab) in enumerate(LIN):
    h2 = H2[lab]
    n = len(h2)
    h1 = t1_stage2(lab)
    h1 = h1[:max(n, 1)] if n else h1
    hp = HP[lab]
    h1 = h1[:max(n, len(hp), 1)]
    ax[0, j].set_title(f"{lab}  (no penalty {n}, penalty {len(hp)} it.)", fontsize=10, fontweight="bold")
    for h, c in ((h1, GRAY), (h2, C2), (hp, C3)):
        if not h:
            continue
        x, y = series(h, "loss", adm=True); ax[0, j].plot(x, y, ".", ms=3, color=c, alpha=0.5)
        if len(y): ax[0, j].plot(x, roll(y, 5), color=c, lw=2, label={GRAY: "Training 1", C2: "Training 2, no penalty", C3: "Training 2, HF penalty 0.05"}[c])
        x, y = series(h, "raw_err_mha", adm=True); ax[1, j].plot(x, y, ".", ms=3, color=c, alpha=0.5)
        if len(y): ax[1, j].plot(x, roll(y, 5), color=c, lw=2)
        gp = [1.0 if r["gate_passed"] else 0.0 for r in h]
        ax[2, j].plot([r["iter"] for r in h], [100 * v for v in roll(gp, 9)], color=c, lw=2)
        x, y = series(h, "depth"); ax[3, j].plot(x, y, ".", ms=3, color=c, alpha=0.6)
    ax[1, j].set_yscale("symlog", linthresh=10); ax[1, j].set_yticks([-100, -10, 0, 10, 30, 100, 300]); ax[1, j].set_yticklabels(["-100", "-10", "0", "10", "30", "100", "300"], fontsize=7); ax[1, j].tick_params(labelleft=True)
    ax[3, j].set_yscale("log")
    ax[3, j].set_xlabel("iteration")
ax[0, 0].set_ylabel("loss (admitted)"); ax[1, 0].set_ylabel("raw error (mHa, admitted)"); ax[2, 0].set_ylabel("gate pass (%, rolling 9)")
ax[3, 0].set_ylabel("depth of the best circuit")
ax[0, 0].legend(frameon=False)
fig.suptitle("Stage 2 (ibm_kobe): Training 1 (gray), Training 2 without penalty (orange; A5 and A1-70q cancelled early), Training 2 with HF penalty (green)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "t2_stage2_progress_training.png", dpi=150)
plt.close(fig)

rows2 = [
    ("policy effective\nvocabulary (of 110)", [("policy_effective_vocab",)], None),
    ("fraction of\nidentity tokens", [("seq_mean", "frac_identity")], None),
    ("Givens rotations\nper circuit, by role", [("seq_mean", "n_givens_homo_lumo"), ("seq_mean", "n_givens_occ_occ"), ("seq_mean", "n_givens_virt_virt")],
     ["HOMO-LUMO", "occupied-occupied", "virtual-virtual"]),
    ("hardware shots\n(fraction of all shots)", [("exc_mean", "frac_valid"), ("exc_mean", "frac_spin_transfer"), ("exc_mean", "frac_number_error")],
     ["right spin sector", "alpha<->beta transfer", "wrong electron number"]),
    ("sampled states\n(right-spin-sector shots)", [("exc_mean", "valid_alpha_only"), ("exc_mean", "valid_beta_only"), ("exc_mean", "valid_alpha_and_beta")],
     ["alpha -> alpha", "beta -> beta", "alpha and beta"]),
    ("weight on HOMO->LUMO\nexcitations", [("exc_mean", "valid_homo_lumo_alpha"), ("exc_mean", "valid_homo_lumo_beta"), ("exc_mean", "valid_homo_lumo_alpha_beta")],
     ["alpha", "beta", "alpha and beta"]),
]
fig, ax = plt.subplots(len(rows2), 4, figsize=(15, 15), sharex="col")
for j, (enc, size, lab) in enumerate(LIN):
    h2 = H2[lab]
    hp_ = HP[lab]
    ax[0, j].set_title(f"{lab}  (no penalty {len(h2)} it., HF penalty {len(hp_)} it.)", fontsize=9, fontweight="bold")
    for i, (title, paths, names) in enumerate(rows2):
        for k, p in enumerate(paths):
            x, y = trk(h2, *p)
            if x:
                ax[i, j].plot(x, y, ".", ms=3, color=cols[k], alpha=0.35)
                ax[i, j].plot(x, roll(y, 6), color=cols[k], lw=2, label=names[k] if names else None)
            xp, yp = trk(hp_, *p)
            if xp:
                ax[i, j].plot(xp, roll(yp, 6), color=cols[k], lw=2, ls="--", label=(names[k] if names and not x else None))
        if names and j == 0:
            ax[i, j].legend(frameon=False, fontsize=7)
        if j == 0:
            ax[i, j].set_ylabel(title, fontsize=8)
    ax[-1, j].set_xlabel("iteration")
fig.suptitle("Training 2, Stage 2 (ibm_kobe): what the policy generates and how the hardware-sampled states look\n"
             "solid: no penalty (only A1-46q ran 150 iterations; the others were cancelled early); dashed: HF penalty 0.05 (all four lineages, 150 iterations); rolling mean over two cycles", fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "t2_stage2_progress_tracking.png", dpi=150)
print("wrote stage-2 figures")


# ---- tuned-reward case (t2t_s2_*): compared with the no-penalty and HF-penalty runs --------------------------------------------------------
HT = {lab: hist(T2 / "out" / f"t2t_s2_{enc}_{size}.json") for enc, size, lab in LIN}
C4 = "#7B2CBF"
fig, ax = plt.subplots(4, 4, figsize=(15, 11), sharex="col")
for j, (enc, size, lab) in enumerate(LIN):
    ht, hp, h2 = HT[lab], HP[lab], H2[lab]
    ax[0, j].set_title(f"{lab}: tuned {len(ht)}, HF pen. {len(hp)}, none {len(h2)} it.", fontsize=9, fontweight="bold")
    for h, c, nm in ((h2, C2, "no penalty"), (hp, C3, "HF penalty 0.05"), (ht, C4, "tuned reward")):
        if not h:
            continue
        x, y = series(h, "loss", adm=True); ax[0, j].plot(x, y, ".", ms=3, color=c, alpha=0.4)
        if len(y): ax[0, j].plot(x, roll(y, 5), color=c, lw=2, label=nm)
        x, y = series(h, "raw_err_mha", adm=True); ax[1, j].plot(x, y, ".", ms=3, color=c, alpha=0.4)
        if len(y): ax[1, j].plot(x, roll(y, 5), color=c, lw=2)
        gp = [1.0 if r["gate_passed"] else 0.0 for r in h]
        ax[2, j].plot([r["iter"] for r in h], [100 * v for v in roll(gp, 9)], color=c, lw=2)
        x, y = series(h, "depth"); ax[3, j].plot(x, y, ".", ms=3, color=c, alpha=0.6)
    ax[1, j].set_yscale("symlog", linthresh=10); ax[1, j].set_yticks([-100, -10, 0, 10, 30, 100, 300]); ax[1, j].set_yticklabels(["-100", "-10", "0", "10", "30", "100", "300"], fontsize=7); ax[1, j].tick_params(labelleft=True); ax[3, j].set_yscale("log"); ax[3, j].set_xlabel("iteration")
ax[0, 0].set_ylabel("loss (admitted)"); ax[1, 0].set_ylabel("raw error (mHa, admitted)"); ax[2, 0].set_ylabel("gate pass (%, rolling 9)")
ax[3, 0].set_ylabel("depth of the best circuit"); ax[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Stage 2 (ibm_kobe), tuned reward (purple) against the HF penalty 0.05 (green) and no penalty (orange; only A1-46q ran the 150 iterations)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "t2_stage2_tuned_training.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(len(rows2), 4, figsize=(15, 15), sharex="col")
for j, (enc, size, lab) in enumerate(LIN):
    ht, hp, h2 = HT[lab], HP[lab], H2[lab]
    ax[0, j].set_title(f"{lab}  (tuned {len(ht)} it.)", fontsize=10, fontweight="bold")
    for i, (title, paths, names) in enumerate(rows2):
        for k, p in enumerate(paths):
            for h, ls, lw, al in ((h2, ":", 1.5, 1), (hp, "--", 1.8, 1), (ht, "-", 2.4, 1)):
                x, y = trk(h, *p)
                if x:
                    if ls == "-":
                        ax[i, j].plot(x, y, ".", ms=3, color=cols[k], alpha=0.3)
                    ax[i, j].plot(x, roll(y, 6), color=cols[k], lw=lw, ls=ls, alpha=al,
                                  label=names[k] if (names and ls == "-") else None)
        if names and j == 0:
            ax[i, j].legend(frameon=False, fontsize=7)
        if j == 0:
            ax[i, j].set_ylabel(title, fontsize=8)
    ax[-1, j].set_xlabel("iteration")
fig.suptitle("Stage 2 (ibm_kobe), tuned reward: what the policy generates and how the hardware-sampled states look\n"
             "solid: tuned reward (points = iterations); dashed: HF penalty 0.05; dotted: no penalty; rolling mean over two cycles of the three molecules",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "t2_stage2_tuned_tracking.png", dpi=150)
print("wrote tuned-reward figures")
