# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Per-iteration training time (Stage 1 and the Stage-2 variants), written as a LaTeX block for the manuscript.

Usage: python3 timing_summary.py <dir with out/*.json from experiments/training2> <paper.tex>
Reads wall_s, sample_s, diagonalize_s, refine_s, real_depth_transpile_s + best_depth_transpile_s and backward_s of every history row.
Iterations stalled by the operator-side mistake (a Runtime job left queued on ibm_kobe by a cancelled Slurm job blocked the device queue)
are excluded from the means: an iteration is counted as stalled when its sampling time exceeds STALL_S = 3600 s; the number excluded is reported.
Rewrites the block between '% BEGIN T2TIMING' and '% END T2TIMING' in paper.tex (the block is created before '\\section{Additional training curves}' if absent).
QPU seconds billed by IBM are not in the histories; they come from the IBM job records and are added when the run has finished.
"""
import json
import sys
from pathlib import Path

import numpy as np

STALL_S = 3600.0
T1DIR = Path("/tmp/stage_all")
T1_S1 = {"a1_46q": ["production_coverage46_a1_v3_s1234"], "a5_46q": ["production_coverage46_a5_v3_s1234"],
         "a1_70q": ["production_coverage70_a1_v3_s1234"], "a5_70q": ["production_coverage70_a5_v3_s1234"]}
T1_S2 = {"a1_46q": ["hw_continue_46q_a1_s1234_v4"], "a1_70q": ["kawasaki_hw_70q_a1_s1234_v5"],
         "a5_46q": ["hw_continue_46q_a5_s1234_v1", "hw_continue_46q_a5_s1234_v2", "hw_continue_46q_a5_s1234_v3"], "a5_70q": ["kawasaki_hw_70q_a5_s1234_v4"]}
RUNS = [("Stage 1", "t2_s1_{lin}_{tag}"), ("Stage 2, no penalty", "t2_s2_{lin}_{tag}"),
        ("Stage 2, HF penalty", "t2p_s2_{lin}_{tag}"), ("Stage 2, tuned", "t2t_s2_{lin}_{tag}")]
LIN = [("a1", "46q", "A1-46q"), ("a5", "46q", "A5-46q"), ("a1", "70q", "A1-70q"), ("a5", "70q", "A5-70q")]


def load(p):
    try:
        return json.load(open(p))["history"]
    except Exception:
        return []


def row(h, hardware):
    # only hardware iterations can stall on the IBM queue; a long simulator iteration (ferrocene in Stage 1) is real work and is kept
    keep = [r for r in h if (not hardware) or (r.get("sample_s") or 0) <= STALL_S]
    if not keep:
        return None
    m = lambda f: float(np.mean([f(r) for r in keep]))
    return dict(n=len(h), stalled=len(h) - len(keep), wall=m(lambda r: r["wall_s"]), sample=m(lambda r: r["sample_s"]),
                diag=m(lambda r: r.get("diagonalize_s") or 0), refine=m(lambda r: r.get("refine_s") or 0),
                transp=m(lambda r: (r.get("real_depth_transpile_s") or 0) + (r.get("best_depth_transpile_s") or 0)),
                back=m(lambda r: r.get("backward_s") or 0),
                total_h=sum(r["wall_s"] for r in keep) / 3600)


def block(root):
    L = ["% BEGIN T2TIMING", "\\begin{table}[t]", "\\centering",
         "\\caption{Time per iteration (seconds, mean over the iterations of each run) by pipeline step, Stage~1 on the simulator and Stage~2 on \\code{ibm\\_kobe}; the last column is the total wall time of the run with the stalled Stage-2 iterations removed. "
         "In the Stage-2 rows, iterations whose circuit sampling took longer than one hour (device-queue stalls caused by Runtime jobs left queued by cancelled Slurm jobs, our own error) are "
         "excluded from the means and from the total, and counted in the column ``stalled''; Stage-1 iterations are all kept, including the ferrocene iterations, whose simulation can take up to an hour. "
         "The sampling column of Stage~2 includes the IBM device queue. $n$ is the number of iterations the run has.}",
         "\\label{tab:t2-timing}", "\\scriptsize", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{@{}llrrrrrrrrr@{}}", "\\toprule",
         "Run & Lineage & $n$ & wall & sampling & diagonalize & refine & transpile & backward & stalled & total (h) \\\\", "\\midrule"]
    for si, (label, pat) in enumerate(RUNS):
        for lin, tag, name in LIN:
            if pat in ("T1S1", "T1S2"):
                h = []
                for f in (T1_S1 if pat == "T1S1" else T1_S2)[f"{lin}_{tag}"]:
                    h += load(T1DIR / f"{f}.json")
            else:
                h = load(root / "out" / (pat.format(lin=lin, tag=tag) + ".json"))
            r = row(h, hardware="Stage 2" in label) if h else None
            if r is None:
                continue
            L.append(f"{label} & {name} & {r['n']} & {r['wall']:.0f} & {r['sample']:.0f} & {r['diag']:.0f} & {r['refine']:.0f} & {r['transp']:.0f} & {r['back']:.1f} & {r['stalled']} & {r['total_h']:.1f} \\\\")
        L.append("\\midrule" if si < len(RUNS) - 1 else "")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", "% END T2TIMING"]
    return "\n".join(x for x in L if x != "")


if __name__ == "__main__":
    root, tex = Path(sys.argv[1]), Path(sys.argv[2])
    b = block(root)
    s = tex.read_text()
    if "% BEGIN T2TIMING" in s:
        a, z = s.index("% BEGIN T2TIMING"), s.index("% END T2TIMING") + len("% END T2TIMING")
        s = s[:a] + b + s[z:]
    else:
        k = s.index("\\section{Additional training curves}")
        s = s[:k] + b + "\n\n" + s[k:]
    tex.write_text(s)
    print(b)
