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

"""Training 1 versus Training 2: loss, raw error, gate pass, depth of the best circuit and the tracked quantities, by stage and lineage.

Usage: python3 compare_trainings.py <dir with Training-2 out/*.json> <dir with Training-1 histories (production_final/out)> [paper.tex]
Prints the tables; with paper.tex it rewrites the block between '% BEGIN T2COMPARE' and '% END T2COMPARE' (created before '% BEGIN TUNETABLE' if absent).
All numbers are single-seed. 'Last 30' means the last 30 iterations of the run. Raw error is the best circuit's raw SQD energy error on admitted iterations only.
Runs that were cancelled or are still running are reported with the number of iterations they have; nothing is extrapolated.
"""
import json
import sys
from pathlib import Path

import numpy as np

T2, T1 = Path(sys.argv[1]), Path(sys.argv[2])
TEX = Path(sys.argv[3]) if len(sys.argv) > 3 else None
LIN = [("a1", "46q", "A1-46q"), ("a5", "46q", "A5-46q"), ("a1", "70q", "A1-70q"), ("a5", "70q", "A5-70q")]
T1S1 = {"a1_46q": "production_coverage46_a1_v3_s1234", "a5_46q": "production_coverage46_a5_v3_s1234",
        "a1_70q": "production_coverage70_a1_v3_s1234", "a5_70q": "production_coverage70_a5_v3_s1234"}
T1S2 = {"a1_46q": ["hw_continue_46q_a1_s1234_v4"], "a1_70q": ["kawasaki_hw_70q_a1_s1234_v5"],
        "a5_46q": ["hw_continue_46q_a5_s1234_v1", "hw_continue_46q_a5_s1234_v2", "hw_continue_46q_a5_s1234_v3"], "a5_70q": ["kawasaki_hw_70q_a5_s1234_v4"]}


def load(p):
    try:
        return json.load(open(p))["history"]
    except Exception:
        return []


def t1s2(k):
    h = []
    for f in T1S2[k]:
        h += load(T1 / f"{f}.json")
    return h


def tr(h, sec, key, a=-30):
    v = []
    for r in h[a:]:
        t = (r.get("track") or {}).get(sec)
        if t and t.get(key) is not None:
            v.append(t[key])
    return float(np.mean(v)) if v else float("nan")


def summ(h):
    if not h:
        return None
    adm = [r["raw_err_mha"] for r in h if r.get("gate_passed") and r.get("raw_err_mha") is not None]
    last = [r["raw_err_mha"] for r in h[-30:] if r.get("gate_passed") and r.get("raw_err_mha") is not None]
    d = np.array([r["depth"] for r in h])
    pv = [(r.get("track") or {}).get("policy_effective_vocab") for r in h[-30:]]
    pv = [x for x in pv if x]
    return dict(n=len(h), gate=100 * np.mean([r["gate_passed"] for r in h]), err_last=float(np.mean(last)) if last else float("nan"),
                err_all=float(np.mean(adm)) if adm else float("nan"), best=min(adm) if adm else float("nan"), depth=float(np.median(d)),
                deep=100 * float(np.mean(d > 1000)), loss0=float(np.mean([r["loss"] for r in h[:10] if r.get("loss") is not None] or [np.nan])),
                voc=float(np.mean(pv)) if pv else float("nan"), valid=tr(h, "exc_mean", "frac_valid"), hf=tr(h, "exc_mean", "valid_hf"),
                hl=tr(h, "exc_mean", "valid_homo_lumo_any"), hlg=tr(h, "seq_mean", "n_givens_homo_lumo"),
                imb=abs(tr(h, "exc_mean", "valid_alpha_only") - tr(h, "exc_mean", "valid_beta_only")))


def f(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


rows = []
for enc, size, lab in LIN:
    k = f"{enc}_{size}"
    runs = [("T1 Stage 1", load(T1 / f"{T1S1[k]}.json")), ("T2 Stage 1", load(T2 / "out" / f"t2_s1_{k}.json")),
            ("T1 Stage 2", t1s2(k)), ("T2 Stage 2, no penalty", load(T2 / "out" / f"t2_s2_{k}.json")),
            ("T2 Stage 2, HF penalty", load(T2 / "out" / f"t2p_s2_{k}.json")), ("T2 Stage 2, tuned", load(T2 / "out" / f"t2t_s2_{k}.json"))]
    for name, h in runs:
        s = summ(h)
        if s:
            rows.append((lab, name, s))

print(f"{'lineage':7s} {'run':24s} {'n':>4s} {'gate%':>5s} {'err last30':>10s} {'err all':>8s} {'best':>7s} {'med depth':>9s} {'%>1000':>6s} {'eff.voc':>7s} {'valid':>6s} {'HF':>5s} {'HL':>5s} {'|a-b|':>6s}")
for lab, name, s in rows:
    print(f"{lab:7s} {name:24s} {s['n']:4d} {s['gate']:5.0f} {f(s['err_last']):>10s} {f(s['err_all']):>8s} {f(s['best']):>7s} {s['depth']:9.0f} {s['deep']:6.0f} "
          f"{f(s['voc']):>7s} {f(s['valid'],2):>6s} {f(s['hf'],2):>5s} {f(s['hl'],2):>5s} {f(s['imb'],3):>6s}")

if TEX:
    L = ["% BEGIN T2COMPARE", "\\begin{table*}[ht]", "\\centering",
         "\\caption{Training~1 against Training~2 (single seed per run), by stage and lineage. $n$ is the number of iterations of the run; ``gate'' the share of iterations that passed the admission gate; "
         "``error'' the mean raw SQD error (mHa) of the best circuit over the admitted iterations of the last 30 iterations (``last'') and of the whole run (``all''); ``depth'' the median transpiled depth of the best circuit "
         "and ``$>$1000'' the share of iterations whose best circuit is deeper than 1{,}000; ``eff.\\ voc.'' the policy's effective vocabulary, ``valid'' the share of shots in the right spin sector, ``HF'' the share of the valid shots "
         "that is the Hartree--Fock determinant, ``HL'' the share that is a HOMO$\\to$LUMO excitation and ``$|\\alpha-\\beta|$'' the difference of the alpha-only and beta-only weights among the valid shots, all three tracked "
         "quantities over the last 30 iterations (Training~1 did not record them). Runs that were cancelled (T2 Stage~2 without penalty, A5-46q, A5-70q) or are still running (tuned) are shown with the iterations they have.}",
         "\\label{tab:t2-compare}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llrrrrrrrrrrrr@{}}", "\\toprule",
         "Lineage & Run & $n$ & gate (\\%) & error last & error all & depth & $>$1000 (\\%) & eff.\\ voc. & valid & HF & HL & $|\\alpha-\\beta|$ \\\\", "\\midrule"]
    prev = None
    for lab, name, s in rows:
        if prev and lab != prev:
            L.append("\\midrule")
        prev = lab
        L.append(f"{lab} & {name} & {s['n']} & {s['gate']:.0f} & {f(s['err_last'])} & {f(s['err_all'])} & {s['depth']:.0f} & {s['deep']:.0f} & {f(s['voc'])} & {f(s['valid'],2)} & {f(s['hf'],2)} & {f(s['hl'],2)} & {f(s['imb'],3)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", "% END T2COMPARE"]
    b = "\n".join(L)
    s_ = TEX.read_text()
    if "% BEGIN T2COMPARE" in s_:
        a, z = s_.index("% BEGIN T2COMPARE"), s_.index("% END T2COMPARE") + len("% END T2COMPARE")
        s_ = s_[:a] + b + s_[z:]
    else:
        k = s_.index("% BEGIN TUNETABLE")
        s_ = s_[:k] + b + "\n\n" + s_[k:]
    TEX.write_text(s_)
    print("wrote table", "tab:t2-compare")
