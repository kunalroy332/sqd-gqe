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

"""Check that figures quoted in the manuscript's running text equal what the stored data give.
Each claim computes its value from the result files and requires the formatted string to occur in paper.tex.
A failure means either the prose is stale or the claim definition here is wrong; both need a human look."""
import json
import re
import statistics as st
import sys
from pathlib import Path

from analysis_numbers import numbers as analysis
from tables_data import LABEL, ROOT, load

TEX = (Path(__file__).resolve().parents[3] / "docs/report/manuscript/paper.tex").read_text()
T = re.sub(r"\s+", " ", TEX)
d = load()
LUCJ = {"L46": (356, 3040), "h2o_m_no23": (356, 3040), "llno28": (428, 4248), "2h2o_m_no29": (446, 4912), "h2o_ll33": (506, 5880), "2h2o_l35": (536, 6952)}
fails, n = [], 0


def f(x):
    return f"{x:.2f}" if abs(x) < 1 else f"{x:.1f}"


def claim(name, text):
    global n
    n += 1
    if re.sub(r"\s+", " ", text) not in T:
        fails.append(f"{name}: expected «{text}»")


def errs(r, keys=("best", "second", "merged", "oldbest", "oldsecond")):
    return [r["t"][k]["err"] for k in keys if r["t"][k]["done"]]


# lowest error per target (DMRG discussion): done results only
for mk, lab in (("h2o_m_no23", "SnOH3-me-h2o M (46q)"), ("h2o_ll33", "SnOH3-me-h2o LL (66q)"), ("L46", "SnOH3-me L (46q)"),
                ("llno28", "SnOH3-me LL (56q)"), ("2h2o_l35", "SnOH3-me-2h2o L (70q)")):
    lo = min(e for r in d.values() if r["mol"] == mk for e in errs(r))
    claim(f"lowest {mk}", f"{f(lo)}~mHa ({lab})")
# h2o-M
h = [r for r in d.values() if r["mol"] == "h2o_m_no23"]
tr = [r for r in h if r["policy"] == "trained"]
un = [r for r in h if r["policy"] == "untrained"]
bests = [r["t"]["best"]["err"] for r in tr if r["t"]["best"]["done"]]
claim("h2o-M trained best range", f"{f(min(bests))}--{f(max(bests))}~mHa")
mg = [r["t"]["merged"]["err"] for r in tr if r["t"]["merged"]["done"]]
claim("h2o-M trained merged range", f"{f(min(mg))}--{f(max(mg))}~mHa")
ed = [e for r in tr for e in errs(r, ("oldbest", "oldsecond"))]
claim("h2o-M E+D range", f"{f(min(ed))}--{f(max(ed))}~mHa")
claim("h2o-M untrained range", f"{f(min(e for r in un for e in errs(r)))}--{f(max(e for r in un for e in errs(r)))}~mHa")
vs = [r["t"]["best"]["valid_shots"] for r in tr]
claim("new best valid shots", f"${min(vs) / 1000:.1f}\\times10^{{3}}$--${max(vs) / 10000:.1f}\\times10^{{4}}$")
vo = [r["t"][k]["valid_shots"] for r in tr for k in ("oldbest", "oldsecond") if r["t"][k]["done"]]
claim("E+D valid shots", f"({min(vo)}--{max(vo)})")
dep = [r["t"]["best"]["depth"] for r in tr]
q2 = [r["t"]["best"]["two_q"] for r in tr]
claim("h2o-M depth/2q", f"as-submitted depth {min(dep)}--{max(dep)}, {min(q2)}--{max(q2)} two-qubit gates")
# L tier
L = [r for r in d.values() if r["mol"] == "L46"]
ut = [e for r in L if r["policy"] == "untrained" for e in errs(r)]
claim("L untrained range", f"between {round(min(ut))} and {round(max(ut))}~mHa")
Ltr = {r["lineage"]: r for r in L if r["policy"] == "trained"}
claim("L A1-46q", f"A1-46q: {f(Ltr['a1-46q']['t']['best']['err'])} best / {f(Ltr['a1-46q']['t']['second']['err'])} second")
claim("L A5-46q", f"A5-46q: {f(Ltr['a5-46q']['t']['best']['err'])} best")
claim("L A1-70q", f"A1-70q: {f(Ltr['a1-70q']['t']['best']['err'])} best")
claim("L A5-70q", f"A5-70q: {f(Ltr['a5-70q']['t']['best']['err'])} best")
claim("L trained live depth", f"(as-submitted depth {min(r['t']['best']['depth'] for r in Ltr.values())}--{max(r['t']['best']['depth'] for r in Ltr.values())})")
claim("L trained 2q", f"have {min(r['t']['best']['two_q'] for r in Ltr.values())}--{max(r['t']['best']['two_q'] for r in Ltr.values())} two-qubit gates")
ub = [(r["t"][k]["depth"], r["t"][k]["two_q"]) for r in L if r["policy"] == "untrained" for k in ("best", "second")]
claim("L untrained cost", f"depth {min(x[0] for x in ub):,}--{max(x[0] for x in ub):,} and {min(x[1] for x in ub):,}--{max(x[1] for x in ub):,} two-qubit gates".replace(",", "{,}"))
# LL
ll_t = d["llno28_a5-70q_trained"]["t"]; ll_u = d["llno28_a1-70q_untrained"]["t"]
claim("LL trained", f"{f(ll_t['best']['err'])}~mHa at as-submitted depth {ll_t['best']['depth']} with {ll_t['best']['two_q']} two-qubit gates")
claim("LL untrained", f"{f(ll_u['best']['err'])}~mHa (as-submitted depth {ll_u['best']['depth']}, {ll_u['best']['valid_shots']:,} valid shots)".replace("12,522", "12{,}522"))
llx = d["llno28_a1-70q_trained"]["t"]
claim("LL A1 exception", f"selected a circuit of depth {llx['best']['depth']:,} with {llx['best']['two_q']:,} two-qubit gates".replace(",", "{,}"))
lle = [llx[k]["err"] for k in ("best","second","merged","oldbest","oldsecond")]
claim("LL A1 errors", f"finished at {min(lle):.1f}--{max(lle):.1f}~mHa above DMRG")
claim("LL A1 valid", f"{sum(1 for c in json.loads((ROOT / 'rerank' / 'llno28_a1-70q_trained.json').read_text())['circuits'] if c['valid_shots'] > 0)} of its ten circuits")
# vs LUCJ at 56-70q
rr = [r for r in d.values() if r["policy"] == "trained" and r["nq"] >= 56 and r["tag"] != "llno28_a1-70q_trained"]  # the selected circuit and its cost exist even if its recovery is unfinished
dr = [LUCJ[r["mol"]][0] / r["t"]["best"]["depth"] for r in rr]
qr = [LUCJ[r["mol"]][1] / r["t"]["best"]["two_q"] for r in rr]
claim("56-70q depth", f"have depth {min(r['t']['best']['depth'] for r in rr)}--{max(r['t']['best']['depth'] for r in rr)} and {min(r['t']['best']['two_q'] for r in rr)}--{max(r['t']['best']['two_q'] for r in rr)} two-qubit gates")
claim("56-70q ratios", f"{round(min(dr))}--{round(max(dr))}$\\times$ shallower and {round(min(qr))}--{round(max(qr))}$\\times$ fewer")
meds_d, meds_q, below = [], [], []
for r in rr:
    C = json.loads((ROOT / "data" / f"{r['tag']}.json").read_text())["circuits"]
    meds_d.append(st.median(c["depth_live"] for c in C)); meds_q.append(st.median(c["two_qubit_live"] for c in C))
    below.append(sum(1 for c in C if c["depth_live"] < LUCJ[r["mol"]][0]))
claim("typical depth", f"median as-submitted depth is {round(min(meds_d)):,}--{round(max(meds_d)):,}".replace(",", "{,}"))
claim("typical 2q", f"median two-qubit-gate count {round(min(meds_q)):,}--{round(max(meds_q)):,}".replace(",", "{,}"))
claim("shallower count", f"only {min(below)}--{max(below)} of the ten circuits")
# coverage
bad = tot = runs_bad = 0
mxidle = 0
two = 0
for tag in d:
    c = json.loads((ROOT / "connectivity" / f"{tag}.json").read_text())
    nq = c["n_logical_qubits"]
    b = 0
    for x in c["circuits"]:
        tot += 1
        idle = nq - x["logical"]["qubits_touched_any_gate"]
        if idle > 0:
            bad += 1; b += 1; mxidle = max(mxidle, idle)
        if x["logical"]["qubits_in_2q_gates"] < nq:
            two += 1
    runs_bad += b > 0
claim("coverage", f"{tot - bad} circuits carry at least one gate on every one")
claim("coverage idle", f"{bad} circuits in {runs_bad} runs (all trained policies) leave between 2 and {mxidle} qubits")
claim("coverage 2q", f"{two} circuits have at least one qubit that is in no two-qubit gate")
# 2h2o-M
m = d["2h2o_m_no29_a1-70q_trained"]["t"]
claim("2h2o-M second", f"{f(m['second']['err'])}~mHa for the second and E+D circuits")
claim("2h2o-M merged", f"{f(m['merged']['err'])}~mHa for merged")
if m["best"].get("partial"):
    claim("2h2o-M partial", f"is {m['best']['p_err']:.1f}~mHa above DMRG")
# small spaces
Sb = [r["t"]["best"]["err"] for r in d.values() if r["mol"] == "sno32"]
claim("S best", f"within {f(min(Sb))}--{f(max(Sb))}~mHa")
# analysis block numbers are generated; verify they are current
a = analysis()
claim("bands", f"median {a['valid_bands'][0]['median']:.1f}~mHa for circuits with fewer than 100 valid shots")
claim("best vs second", f"in {a['best_vs_second']['best_not_worse']}, i.e.")
# discussion: shot cost against LUCJ
import json as _j
from cost_numbers import numbers as _cn
_sh = _cn()["inference_shots"]
_gq, _lq = _sh["fresh"]["qpu_s"] / _sh["fresh"]["jobs"], _sh["lucj"]["qpu_s"] / _sh["lucj"]["jobs"]
claim("QPU per fresh run", f"{_gq:.0f}~s of billed QPU time on average")
claim("QPU per LUCJ job", f"a mean of {_lq:.0f}~s of QPU time per LUCJ job, about {_gq / _lq:.0f} times less")
print(f"{n} prose claims checked, {len(fails)} do not match")
for x in fails:
    print("STALE?", x)
sys.exit(1 if fails else 0)
