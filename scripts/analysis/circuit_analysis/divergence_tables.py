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

"""Per-circuit divergence table (appendix) and summary table (main text) for the manuscript, from <tree>/divergence/*.json.
Usage: python3 divergence_tables.py PAPER.tex    (rewrites the blocks '% BEGIN DIVAPP'..'% END DIVAPP' and '% BEGIN DIVSUM'..'% END DIVSUM')"""
import glob, json, os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TREES = [("Training 1", "circuit_analysis"), ("HF penalty", "circuit_analysis_hf"), ("tuned", "circuit_analysis_tuned")]

NAME = {"sno32": "SnOH3-me S (32q)", "h2o_s19": "SnOH3-me-h2o S (38q)", "2h2o_s22": "SnOH3-me-2h2o S (44q)"}
ORDER = ["sno32", "h2o_s19", "2h2o_s22"]
LIN = ["a1-46q", "a1-70q", "a5-46q", "a5-70q"]
f3 = lambda x: f"{x:.3f}"


def load(tree):
    sel = {}
    for f in glob.glob(f"{REPO}/{tree}/selection_batch*.json"):
        sel.update(json.load(open(f)))
    out = {}
    for f in sorted(glob.glob(f"{REPO}/{tree}/divergence/*_trained.json")):
        dv = json.load(open(f)); tag = dv["tag"]
        data = {c["index"]: c for c in json.load(open(f"{REPO}/{tree}/data/{tag}.json"))["circuits"]}
        s = sel.get(tag, {})
        out[tag] = (dv["molecule"], [(c, data[c["index"]], "best" if c["index"] == s.get("best") else "2nd" if c["index"] == s.get("second") else "") for c in dv["circuits"]])
    return out


def summary_block():
    L = ["% BEGIN DIVSUM", "\\begin{table}[!tp]", "\\centering",
         "\\caption{Ideal (noiseless) distance of the generated circuits from the idealised LUCJ circuit (Section~\\ref{sec:divergence}), medians over the circuits of the three small tiers (SnOH3-me S, SnOH3-me-h2o S, SnOH3-me-2h2o S; ten circuits per run; 8 runs for Training~1, 12 for the HF penalty and 11 for the tuned reward, whose SnOH3-me-2h2o S run with A5-70q has no best and second circuit). $F$: state fidelity; $H$: Hellinger distance between the determinant distributions; $d_{\\mathrm{HF}}$: distance of the two states up to a global phase (a lower bound of the operator distance); ``near'': circuits with $F>0.5$. Selected = best and second circuit of the ranking. The per-circuit values are in Appendix~\\ref{sec:appendix-divergence}.}",
         "\\label{tab:divergence-summary}", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.15}", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         "Set & Circuits & $n$ & $F$ & $H$ & $d_{\\mathrm{HF}}$ & near & depth \\\\", "\\midrule"]
    for lab, tree in TREES:
        runs = load(tree)
        grp = {"selected": [], "not selected": []}
        for tag, (mk, cs) in runs.items():
            if not any(r for _, _, r in cs):
                continue   # no ranking selection for this run yet
            for c, d, role in cs:
                grp["selected" if role else "not selected"].append((c["F"], c["H"], c["d_HF"], d["depth_live"]))
        for k, v in grp.items():
            if not v: continue
            a = np.array(v)
            L.append(f"{lab if k == 'selected' else ''} & {k} & {len(v)} & {f3(np.median(a[:,0]))} & {f3(np.median(a[:,1]))} & {f3(np.median(a[:,2]))} & {int((a[:,0] > 0.5).sum())}/{len(v)} & {np.median(a[:,3]):.0f} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table}", "% END DIVSUM"]
    return "\n".join(L)


def app_block():
    L = ["% BEGIN DIVAPP"]
    for lab, tree in TREES:
        runs = load(tree)
        L += ["\\begin{longtable}{@{}lllrrrrrrr@{}}",
              f"\\caption{{Per-circuit ideal divergence from the LUCJ circuit, {lab} set (Section~\\ref{{sec:divergence}}). $F$, $H$, JS (bits) and $d_{{\\mathrm{{HF}}}}$ as in Table~\\ref{{tab:divergence-summary}}; op.\\ norm: lower bound of $\\|U_g-U_{{\\mathrm{{LUCJ}}}}\\|$ (maximum 2), computed for the best and second circuit of SnOH3-me S only; depth: as-submitted depth. ``sel.'': best or second circuit.}}\\label{{tab:divergence-{tree}}}\\\\",
              "\\toprule", "Run & \\# & sel. & depth & $F$ & $H$ & JS & $d_{\\mathrm{HF}}$ & op.\\ norm & \\\\", "\\midrule", "\\endfirsthead",
              "\\toprule", "Run & \\# & sel. & depth & $F$ & $H$ & JS & $d_{\\mathrm{HF}}$ & op.\\ norm & \\\\", "\\midrule", "\\endhead"]
        for mk in ORDER:
            for lin in LIN:
                tag = f"{mk}_{lin}_trained"
                if tag not in runs: continue
                for i, (c, d, role) in enumerate(runs[tag][1]):
                    run = f"{NAME[mk]}, {lin.upper()}" if i == 0 else ""
                    op = f"{c['op_norm']:.2f}" if c.get("op_norm") is not None else "--"
                    L.append(f"{run} & {c['index'] + 1} & {role} & {d['depth_live']} & {f3(c['F'])} & {f3(c['H'])} & {f3(c['JS'])} & {f3(c['d_HF'])} & {op} & \\\\")
                L.append("\\addlinespace")
        L += ["\\bottomrule", "\\end{longtable}", ""]
    L.append("% END DIVAPP")
    return "\n".join(L)


if __name__ == "__main__":
    tex = sys.argv[1]; s = open(tex).read()
    for name, blk in (("DIVSUM", summary_block()), ("DIVAPP", app_block())):
        a, z = f"% BEGIN {name}", f"% END {name}"
        if a not in s:
            print("marker missing:", name); continue
        i = s.index(a); j = s.index(z) + len(z)
        s = s[:i] + blk + s[j:]
    open(tex, "w").write(s); print("ok")
