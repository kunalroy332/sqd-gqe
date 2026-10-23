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

"""Three-way comparison tables (Training 1, HF penalty, tuned reward) from the stored inference results.
Usage: python3 three_way_load.py   (builds $INF_DIR/R3.pkl from the pulled trees)  then  python3 three_way_tables.py PAPER.tex
Compares the (molecule, lineage) cells in which all three sets have the best AND the second circuit recovered. Values depend on a campaign still
running when the table was made, so they are set in red. Rewrites the blocks '% BEGIN THREEWAY'..'% END THREEWAY' (main text, medians by lineage)
and '% BEGIN THREEWAYAPP'..'% END THREEWAYAPP' (appendix, every cell)."""
import os, sys, pickle, numpy as np
INF = os.environ.get("INF_DIR", "/tmp/inf")
R = pickle.load(open(INF + "/R3.pkl", "rb"))
NAME = {"sno32": "S 32q", "sno38": "M 38q", "L46": "L 46q", "llno28": "LL 56q", "h2o_s19": "h2o S 38q", "h2o_m_no23": "h2o M 46q", "h2o_l28": "h2o L 56q",
        "h2o_ll33": "h2o LL 66q", "2h2o_s22": "2h2o S 44q", "2h2o_m_no29": "2h2o M 58q", "2h2o_l35": "2h2o L 70q"}
ORDER = ["sno32", "h2o_s19", "sno38", "2h2o_s22", "L46", "h2o_m_no23", "llno28", "h2o_l28", "2h2o_m_no29", "h2o_ll33", "2h2o_l35"]
SETS = [("T1", "Training 1"), ("HF", "HF penalty"), ("tuned", "tuned")]
cells = sorted([k for k, v in R.items() if {"T1", "HF", "tuned"} <= set(v)], key=lambda k: (ORDER.index(k[0]), k[1]))
RED = lambda s: s   # all three sets are finished for the compared cells; nothing is pending
f = lambda x, p=1: "--" if x is None else f"{x:.{p}f}"
pc = lambda x: "--" if x is None else f"{100 * x:.1f}"
med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else None


def main_block():
    L = ["% BEGIN THREEWAY", "\\begin{table*}[t]", "\\centering",
         "\\caption{The three model sets on the cells in which all three have the best and second circuit recovered. Medians over cells, by lineage (last block: all cells): error of the best circuit, of the second circuit and of the better of the two, against DMRG (mHa); as-submitted depth of the best circuit; and the shot composition of the best circuit in percent: share of all shots in the right spin sector (valid), Hartree--Fock determinant share and HOMO$\\to$LUMO share of the valid shots, and the alpha-only, beta-only and both-spin shares of the valid shots. }",
         "\\label{tab:three-way}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}llrrrrrrrrrrr@{}}", "\\toprule",
         "Lineage & Model set & cells & best err & 2nd err & min(best,2nd) & depth & valid & HF & HL & $\\alpha$ only & $\\beta$ only & both \\\\", "\\midrule"]
    for lin in ("a1-46q", "a5-46q", "a1-70q", "a5-70q", None):
        ks = [k for k in cells if lin is None or k[1] == lin]
        if not ks:
            continue
        for s, lab in SETS:
            b = [R[k][s]["best"] for k in ks]; c = [R[k][s]["second"] for k in ks]
            mn = [min(x["err"], y["err"]) for x, y in zip(b, c)]
            vals = [f"{lin.upper() if lin else 'all'}" if s == "T1" else "", lab, str(len(ks)), f(med([x["err"] for x in b]), 2), f(med([x["err"] for x in c]), 2), f(med(mn), 2),
                    f"{med([x['depth'] for x in b]):.0f}"] + [pc(med([x[q] for x in b])) for q in ("valid", "hf", "hl", "a", "b", "both")]
            L.append(" & ".join(RED(v) if i >= 2 and v not in ("", "--") else v for i, v in enumerate(vals)) + " \\\\")
        L.append("\\midrule" if lin else "")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", "% END THREEWAY"]
    return "\n".join(x for x in L if x != "")


def app_block():
    HDR = "Target & Lineage & Set & best E$_h$ & err & 2nd E$_h$ & err & depth & valid (\\%) & HF (\\%) & HL (\\%) & $\\alpha$/$\\beta$/both (\\%) \\\\"
    L = ["% BEGIN THREEWAYAPP", "{\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{longtable}{@{}lllrrrrrrrrrr@{}}", "\\toprule",
         HDR, "\\midrule", "\\endfirsthead", HDR, "\\midrule", "\\endhead"]
    for k in cells:
        for i, (s, lab) in enumerate(SETS):
            b, c = R[k][s]["best"], R[k][s]["second"]
            row = [NAME[k[0]] if i == 0 else "", k[1].upper() if i == 0 else "", lab, f(b["E"], 6), f(b["err"], 2), f(c["E"], 6), f(c["err"], 2), str(b["depth"]),
                   pc(b["valid"]), pc(b["hf"]), pc(b["hl"]), "/".join(pc(b[q]) for q in ("a", "b", "both"))]
            L.append(" & ".join(row[:3]) + " & " + " & ".join(RED(v) for v in row[3:]) + " \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{longtable}}", "% END THREEWAYAPP"]
    return "\n".join(L)


if __name__ == "__main__":
    tex = sys.argv[1]
    s = open(tex).read()
    for name, blk in (("THREEWAY", main_block()), ("THREEWAYAPP", app_block())):
        a, z = f"% BEGIN {name}\n", f"% END {name}"
        if a.strip() not in s:
            print("marker missing:", name); continue
        i = s.index(a.strip()); j = s.index(z) + len(z)
        s = s[:i] + blk + s[j:]
    open(tex, "w").write(s)
    print(len(cells), "cells compared")
