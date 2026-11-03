# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Self-contained HTML report + raw data for one model set: Training 1, HF penalty (T2) or tuned reward (T2).

Usage:  python3 make_html_sets.py t1|hf|tuned       (reads circuit_analysis[_hf|_tuned]/, /tmp/t2/{out,track}, /tmp/stage_all, /tmp/inf/R3.pkl)
Output: docs/reports/report_<set>.html  and  docs/reports/report_<set>_data/ (per-run JSON, summary CSVs).
Anything that depends on a run, recovery or analysis that is not finished is set in red (class pend) with the note that it is updated when the jobs finish.
"""
import base64
import csv
import html
import io
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SET = sys.argv[1]
REPO = Path(__file__).resolve().parents[3]
TREE = {"t1": "circuit_analysis", "hf": "circuit_analysis_hf", "tuned": "circuit_analysis_tuned"}[SET]
os.environ["CA_ROOT"] = str(REPO / TREE)
sys.path.insert(0, str(Path(__file__).parent))
from tables_data import DMRG, LABEL, ROOT, load  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T2 = Path("/tmp/t2/out")
OUTDIR = REPO / "docs/reports"
NAMES = {"t1": "training1", "hf": "hf_penalty", "tuned": "tuned_reward"}
E = html.escape
LINS = ["a1-46q", "a5-46q", "a1-70q", "a5-70q"]
ALLMOL = ["sno32", "h2o_s19", "sno38", "2h2o_s22", "L46", "h2o_m_no23", "llno28", "h2o_l28", "2h2o_m_no29", "h2o_ll33", "2h2o_l35"]
FIRST = {"a1-46q": {"2h2o_s22", "L46", "h2o_m_no23", "h2o_s19", "sno32"}, "a5-46q": {"2h2o_s22", "L46", "h2o_m_no23", "h2o_s19", "sno32"},
         "a1-70q": {"2h2o_l35", "2h2o_m_no29", "L46", "h2o_ll33", "h2o_m_no23", "llno28", "sno32"}, "a5-70q": {"2h2o_l35", "2h2o_m_no29", "L46", "h2o_ll33", "h2o_m_no23", "llno28", "sno32"}}
NEWT1 = {"sno38", "h2o_l28"}
if SET == "t1":
    PLANNED = {l: sorted(FIRST[l] | NEWT1, key=ALLMOL.index) for l in LINS}
else:
    PLANNED = {l: list(ALLMOL) for l in LINS}
NPLAN = sum(len(v) for v in PLANNED.values())
TITLE = {"t1": "Training 1", "hf": "Training 2, Hartree-Fock penalty (main set)", "tuned": "Training 2, jointly tuned reward"}[SET]
T2PRE = {"hf": "t2p_s2", "tuned": "t2t_s2"}.get(SET)
LAMBDA = {"t1": "no HF term (lambda_exc 2.0, lambda_cov 10.0)", "hf": "lambda_HF 0.05 (lambda_exc 2.0, lambda_cov 10.0)",
          "tuned": "lambda_HF 0.07, lambda_spin 0.77, c_H 0.26, lambda_exc 3.9, lambda_cov 7.2"}[SET]
QPU_T2 = {"hf": {"a1-46q": (149.4, 59.8), "a5-46q": (148.5, 59.4), "a1-70q": (174.8, 69.9), "a5-70q": (175.1, 70.0)},
          "tuned": {"a1-46q": (211.3, 84.5), "a5-46q": (206.3, 82.5), "a1-70q": (365.7, 160.1), "a5-70q": (391.5, 156.6)}}
PEND = '<span class="pend">{}</span>'
NOTE = "updated when the jobs finish"


def pend(t="pending"):
    return PEND.format(t)


def fnum(x, p=2):
    return "&ndash;" if x is None else f"{x:.{p}f}"


def pct(x):
    return "&ndash;" if x is None else f"{100 * x:.1f}"


def table(headers, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img alt="figure" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


# ------------------------------------------------------------------ data
d = load()
runs = sorted(d.values(), key=lambda r: (r["order"], r["lineage"], r["policy"]))
trained = [r for r in runs if r["policy"] == "trained"]
untrained = [r for r in runs if r["policy"] == "untrained"]
data = {r["tag"]: json.loads((ROOT / "data" / f"{r['tag']}.json").read_text()) for r in runs}
rerank = {r["tag"]: json.loads((ROOT / "rerank" / f"{r['tag']}.json").read_text()) for r in runs}
chem_dir = ROOT / "chem"
chem = {}
for r in runs:
    c = chem_dir / f"{r['tag']}.json"
    chem[r["tag"]] = {x["index"]: x for x in json.loads(c.read_text())["circuits"]} if c.exists() else {}
rec = defaultdict(dict)
for f in (ROOT / "rerecovery").glob("*__*.json"):
    tag, tg = f.stem.split("__")
    rec[tag][tg] = json.loads(f.read_text())


def circ_chem(tag, idx):
    """Chemistry fields of one circuit: from the run file (HF/tuned, stored at inference) or the Training-1 backfill."""
    c = next((x for x in data[tag]["circuits"] if x["index"] == idx), {})
    if c.get("exc_stats"):
        return c.get("exc_stats"), c.get("seq_stats") or {}, c.get("hf_fraction"), c.get("spin_imbalance")
    b = chem[tag].get(idx)
    if b:
        return b.get("exc_stats"), b.get("seq_stats") or {}, b.get("hf_fraction"), b.get("spin_imbalance")
    return None, {}, None, None


def best_chem(r):
    e, s, hf, imb = circ_chem(r["tag"], r["t"]["best"]["idx"])
    return e, s


have = {(r["mol"], r["lineage"]) for r in trained}
NOSEL = [(mk, lin) for lin in LINS for mk in PLANNED[lin] if (mk, lin) not in {(r["mol"], r["lineage"]) for r in trained}] if SET != "t1" else []
PENDING_CELLS = sum(1 for r in trained for e in r["t"].values() if not e["done"])
nfin = sum(1 for r in trained if r["t"]["best"]["done"])
nmeas = len(trained)
_nsel = len(NOSEL)
status = (f"all {NPLAN} planned trained runs were sampled on ibm_kobe; {nfin} have a recovered best circuit"
          + (f", and {_nsel} run{'s' if _nsel != 1 else ''} had a single circuit with valid shots, so the ranking found no best and second circuit" if _nsel else "")
          + (f"; {PENDING_CELLS} recovery result{'s' if PENDING_CELLS != 1 else ''} still running" if PENDING_CELLS else "")) if SET != "t1" else \
         (f"all {nmeas} Training 1 runs are available: the 24 of the first campaign and the 8 on the two tiers added later (SnOH3-me M 38q and SnOH3-me-h2o L 56q, four lineages); the other 12 (molecule x lineage) cells were not run with this set")
_red = "Entries in <span class=\"pend\">red</span> are not available." if (PENDING_CELLS or SET == "t1") else "Nothing in this report is pending."
BANNER = (f'<div class="note warn"><b>Status.</b> {status}. {_red} '
          f'No number in this report is taken from another model set. Reference date of the data: 2026-10-23.</div>')

# ------------------------------------------------------------------ training data
def hist(name):
    try:
        return json.load(open(T2 / f"{name}.json"))["history"]
    except Exception:
        return []


def trk(h, *path):
    xs = []
    for r in h:
        t = r.get("track")
        for p in path:
            t = t.get(p) if isinstance(t, dict) else None
        if t is not None:
            xs.append(t)
    return xs


def tr_stats(h, last=20):
    if not h:
        return None
    hh = h[-last:]
    g = lambda *p: med(trk(hh, *p))
    gm = lambda *p: float(np.mean(trk(hh, *p))) if trk(hh, *p) else None
    return {"n": len(h), "gate": float(np.mean([r["gate_passed"] for r in h])), "err_last10": float(np.mean([r["raw_err_mha"] for r in h[-10:]])),
            "depth_last10": med([r["depth"] for r in h[-10:]]), "wall_h": sum(r["wall_s"] for r in h) / 3600,
            "eff_vocab": gm("policy_effective_vocab"),
            "best": {k: gm("exc_best", k) for k in ("frac_valid", "valid_hf", "valid_homo_lumo_any", "valid_alpha_only", "valid_beta_only", "valid_alpha_and_beta")},
            "mean": {k: gm("exc_mean", k) for k in ("frac_valid", "valid_hf", "valid_homo_lumo_any", "valid_alpha_only", "valid_beta_only", "valid_alpha_and_beta")},
            "seq": {k: gm("seq_best", k) for k in ("frac_identity", "n_hf_changing_gates", "frac_givens_homo_lumo", "n_givens", "n_lucj", "n_orb_rot", "n_diag_coulomb", "n_num_num", "orbital_coverage")}}


# ------------------------------------------------------------------ html
out = []
A = out.append
A(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SQD-GQE zero-shot inference report: {E(TITLE)}</title><style>
:root{{--bg:#fff;--fg:#1b1f24;--mut:#59636e;--line:#d0d7de;--acc:#0b5cad;--card:#f6f8fa;--warn:#9a6700;--red:#cf222e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--acc:#58a6ff;--card:#161b22;--warn:#d29922;--red:#ff7b72}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:1.7rem;margin:.2em 0}}h2{{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}}h3{{margin-top:1.6em}}
a{{color:var(--acc)}}code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}}
.tw{{overflow-x:auto;margin:.8em 0}}table{{border-collapse:collapse;font-size:.85rem;white-space:nowrap}}
th,td{{border:1px solid var(--line);padding:3px 8px;text-align:right}}th{{background:var(--card);position:sticky;top:0}}
td:first-child,th:first-child{{text-align:left}}
.note{{background:var(--card);border-left:4px solid var(--acc);padding:8px 14px;margin:1em 0}}
.warn{{border-left-color:var(--red)}}.pend{{color:var(--red);font-weight:600}}
details{{border:1px solid var(--line);border-radius:6px;margin:10px 0;padding:6px 12px}}summary{{cursor:pointer;font-weight:600}}
img{{max-width:100%;height:auto}}nav a{{margin-right:12px}}.mut{{color:var(--mut)}}
</style></head><body><main>
<h1>Zero-shot inference of the {E(TITLE)} models</h1>
<p class="mut">Part of the three-report release of the amortized SQD-GQE study (Training 1, HF penalty, tuned reward). Reward: {E(LAMBDA)}.</p>
{BANNER}
<nav><a href="#overview">Overview</a><a href="#training">Training</a><a href="#summary">Master table</a><a href="#chemistry">Chemistry at inference</a>
<a href="#tvi">Training vs inference</a><a href="#threeway">Three-way comparison</a><a href="#timing">Timing</a><a href="#runs">Per-run detail</a><a href="#raw">Raw data</a></nav>
<h2 id="overview">1. What this set is</h2>""")

OV = {
    "t1": "Training 1 is the first recipe: Stage 1 with a 32-operator vocabulary (simulator, early stop after 90-132 iterations), Stage 2 with a 74- or 110-operator vocabulary on <code>ibm_kobe</code> (88-150 iterations, chains of resumed runs), "
          "output head, token embedding and prompt encoder re-initialised at the handoff, best-score checkpoints. The reward has the excitation, coverage and overshoot terms only. It carries no per-iteration chemistry tracking.",
    "hf": "The main set of the paper. Training 2: vocabulary 110 in both stages, every tensor carried from Stage 1 to Stage 2, fixed 150 + 150 iterations without early stop, per-iteration chemistry tracking; Stage 2 adds a Hartree-Fock penalty "
          "(lambda_HF = 0.05) to the reward. Final (not best-score) Stage-2 policies are used. Stage 1 is shared with the tuned set and the no-penalty run.",
    "tuned": "Training 2 with the jointly tuned reward (configuration 3 of the 42-configuration search: lambda_HF 0.07, lambda_spin 0.77, entropy coefficient c_H 0.26, lambda_exc 3.9, lambda_cov 7.2). Same Stage 1 as the HF-penalty set. "
             "The A1-70q policy is the one saved at iteration 129 of 137 (the run ended by a segmentation fault in the transpiler); no resume was done."}[SET]
A(f"<p>{OV}</p>")
A("""<p>Inference protocol (identical for all three sets): the policy draws <b>10</b> gate sequences (seed 1234) for an unseen SnOH<sub>3</sub>-Me-family active space; each circuit is transpiled (<code>optimization_level=1</code>) and run for
<b>100,000 shots</b> on <code>ibm_kobe</code> (one Runtime job, no error mitigation). Each circuit is ranked by a one-shot diagonalization of its valid shots (net CI dimension 40,000); <i>best</i> and <i>second</i> are the two lowest ranking energies,
<i>merged</i> pools the shots of every circuit with a finite ranking energy, and <i>E+D 1/2</i> are the two circuits of the earlier energy-plus-distinct-bitstring rule. Each is recovered with the configuration-recovery SQD loop (GPU, subspace 20,000, 25 iterations).
Error = recovered energy minus the DMRG energy of the molecule (bond dimension 4000), in mHa.</p>
<p><b>Chemistry quantities</b> (from the measured bitstrings): <i>valid</i> = share of the 100,000 shots with the target alpha and beta electron numbers; <i>HF</i> = share of the valid shots that is exactly the Hartree-Fock determinant; <i>HL</i> = share of the valid shots with a HOMO&rarr;LUMO excitation;
<i>alpha only / beta only / both</i> = share of the valid shots that excite only the alpha channel, only the beta channel, or both. Sequence statistics: fraction of identity (padding) tokens, number of Givens rotations and of HF-changing gates, share of Givens rotations on the HOMO-LUMO pair, operator-family counts.</p>""")

# ------------------------------------------------------------------ training
A('<h2 id="training">2. Training of this set</h2>')
if SET == "t1":
    S1 = {"a1-46q": ["production_coverage46_a1_v3_s1234"], "a5-46q": ["production_coverage46_a5_v3_s1234"], "a1-70q": ["production_coverage70_a1_v3_s1234"], "a5-70q": ["production_coverage70_a5_v3_s1234"]}
    S2 = {"a1-46q": ["hw_continue_46q_a1_s1234_v4"], "a1-70q": ["kawasaki_hw_70q_a1_s1234_v5"], "a5-46q": ["hw_continue_46q_a5_s1234_v1", "hw_continue_46q_a5_s1234_v2", "hw_continue_46q_a5_s1234_v3"], "a5-70q": ["kawasaki_hw_70q_a5_s1234_v4"]}
    rows = []
    for lin in LINS:
        row = [lin.upper()]
        for st in (S1, S2):
            h = []
            for f in st[lin]:
                p = Path("/tmp/stage_all") / f"{f}.json"
                if p.exists():
                    h += json.load(open(p))["history"]
            if h:
                row += [len(h), f"{np.mean([r['gate_passed'] for r in h]) * 100:.0f}", f"{np.mean([r['raw_err_mha'] for r in h[-10:]]):.1f}", f"{med([r['depth'] for r in h[-10:]]):.0f}",
                        f"{sum(r['wall_s'] for r in h if (r.get('sample_s') or 0) <= 3600) / 3600:.1f}"]
            else:
                row += [pend("n/a")] * 5
        rows.append(row)
    A(table(["Lineage", "S1 iters", "S1 gate %", "S1 raw err last10 (mHa)", "S1 depth last10", "S1 wall (h)", "S2 iters", "S2 gate %", "S2 raw err last10 (mHa)", "S2 depth last10", "S2 wall (h, stalls removed)"], rows))
    A("<p>Training 1 has no per-iteration chemistry tracking, so the training-time chemistry (valid share, HF share, effective vocabulary) does not exist for this set; the comparison with training in section 5 is therefore only possible for the two Training 2 sets. "
      "The stage-2 QPU time of Training 1 was 819 minutes for the 589 jobs IBM returns for the account (11,780 circuits, 117,800,000 shots), 83 s per job; 212 further Kawasaki-era jobs are not returned and their QPU time is unknown.</p>")
    A("""<h3>The earlier hyperparameter and reward-tuning narrative (summary)</h3><ul>
<li><b>Depth ceiling dominates.</b> A 36-job sweep of the depth ceiling and the coverage and overshoot weights showed the depth ceiling as the main lever against zero-gradient stalls; lambda_cov = 10, lambda_over = 2 and lambda_exc = 2 were fixed there (lambda_exc from the finding that an accepted circuit's recovered state sat at 97.5% high-rank weight).</li>
<li><b>Energy gate loosened at 70 qubits.</b> The 46-qubit energy tolerance admitted zero circuits for many iterations at 70 qubits (a deadlock, since no admitted sample means no gradient); loosening it by a factor of five restored a pass rate near 100%.</li>
<li><b>Hartree-Fock penalty test.</b> 24-qubit CPU test (C<sub>2</sub>H<sub>4</sub>, 3 seeds, 60 iterations, 5% readout flips), lambda_HF in {0, 0.05, 0.2, 1.0}: at 0.05 the HF share of valid shots fell from 57% to 37%, HOMO-LUMO rotations doubled and the raw error stayed within 1 mHa; at 0.2 and 1.0 the policy escaped HF onto a single excited determinant (raw error 44 and 75 mHa). Hence lambda_HF = 0.05 for the hardware runs of the HF-penalty set.</li>
<li><b>Joint search.</b> After the HF penalty the 46-qubit policies settled on beta-only circuits and collapsed onto one sequence. A grid of lambda_spin and c_H, then 40 Latin-hypercube configurations plus 2 reference settings (2 seeds, 80 iterations, noise model fitted to ibm_kobe counts) led to configuration 3 (the tuned set). It was chosen on the 26 configurations finished at that time, not on the final ranking.</li></ul>""")
else:
    pre = T2PRE
    rows, tst = [], {}
    for lin in LINS:
        k = lin.replace("-", "_")
        h1, h2 = hist(f"t2_s1_{k}"), hist(f"{pre}_{k}")
        s1, s2 = tr_stats(h1), tr_stats(h2)
        tst[lin] = (s1, s2)
        q = QPU_T2[SET][lin]
        rows.append([lin.upper(), s1["n"], f"{s1['gate'] * 100:.0f}", f"{s1['err_last10']:.1f}", f"{s1['depth_last10']:.0f}", f"{s1['wall_h']:.1f}",
                     s2["n"], f"{s2['gate'] * 100:.0f}", f"{s2['err_last10']:.1f}", f"{s2['depth_last10']:.0f}", f"{s2['wall_h']:.1f}", f"{q[0]:.0f}", f"{q[1]:.1f}"])
    A(table(["Lineage", "S1 iters", "S1 gate %", "S1 raw err last10 (mHa)", "S1 depth last10", "S1 wall (h)", "S2 iters", "S2 gate %", "S2 raw err last10 (mHa)", "S2 depth last10", "S2 wall (h)", "QPU min (S2)", "QPU s/job"], rows))
    A("<p>Raw error is against the CASCI/CCSD(T) surrogate of the transition-metal training molecules, not DMRG; read it as relative. Stage 1 is the shared simulator stage. QPU time is IBM-billed time of the Stage-2 Runtime jobs (20 circuits x 10,000 shots each).</p>")
    # figure: effective vocab and chemistry over training
    fig, ax = plt.subplots(2, 4, figsize=(15, 6), sharex="col")
    for j, lin in enumerate(LINS):
        k = lin.replace("-", "_")
        for hh, off, lab in ((hist(f"t2_s1_{k}"), 0, "Stage 1"), (hist(f"{pre}_{k}"), 150, "Stage 2")):
            it = [r["iter"] + off if r["iter"] <= 150 else r["iter"] for r in hh]
            ev = [(r.get("track") or {}).get("policy_effective_vocab") for r in hh]
            ax[0][j].plot(it, ev, lw=1.2, label=lab)
            for key, c in (("valid_hf", "#2B9348"), ("valid_alpha_only", "#0F62FE"), ("valid_beta_only", "#D9480F")):
                if lab == "Stage 2":
                    y = [((r.get("track") or {}).get("exc_best") or {}).get(key) for r in hh]
                    ax[1][j].plot(it, y, color=c, lw=1.0, label=key.replace("valid_", ""))
        ax[0][j].set_title(lin.upper()); ax[0][j].set_yscale("log"); ax[0][j].grid(alpha=.2)
        ax[1][j].grid(alpha=.2); ax[1][j].set_xlabel("iteration (Stage 2 shifted by 150)")
    ax[0][0].set_ylabel("effective vocabulary"); ax[1][0].set_ylabel("share of valid shots, best circuit")
    ax[0][0].legend(fontsize=7); ax[1][0].legend(fontsize=7)
    A("<h3>Policy collapse and chemistry during training</h3>" + b64fig(fig))
    A("<p>Top: effective vocabulary of the policy (exponential of the mean token entropy). Bottom: Stage-2 composition of the valid shots of the best circuit of each iteration (HF determinant share, alpha-only, beta-only). "
      + ("The HF-penalty policy has about one effective token from iteration ~50-55 of Stage 2 and its 46-qubit circuits are strongly beta-only." if SET == "hf" else
         "The tuned policy keeps an effective vocabulary of about 20-27 tokens and balanced spin channels at the price of deeper 70-qubit circuits.") + "</p>")
    if SET == "tuned":
        A('<div class="note">A1-70q: the run ended by a segmentation fault at iteration 137; the policy used at inference was saved at iteration 129.</div>')

# ------------------------------------------------------------------ master table
A('<h2 id="summary">3. Master table: every planned run</h2>')
A(f"<p>Energy (E<sub>h</sub>) / error against DMRG (mHa) of the <i>best</i> circuit, with depth and two-qubit gate count as submitted to <code>ibm_kobe</code>. Cells not yet available are red.</p>")
rows = []
byk = {(r["mol"], r["lineage"]): r for r in trained}
for mk in sorted(ALLMOL, key=lambda m: (LABEL[m][1], LABEL[m][2])):
    for lin in LINS:
        r = byk.get((mk, lin))
        if r is None:
            if mk in PLANNED[lin]:
                rows.append([E(LABEL[mk][0]), lin.upper(), "only one circuit with valid shots: no best/second pair"] + ["&ndash;"] * 6)
            continue
        t = r["t"]
        b, s, m_ = t["best"], t["second"], t["merged"]
        ce = lambda e: pend("recovery running") if not e["done"] else f"{e['err']:.2f}"
        rows.append([E(LABEL[mk][0]), lin.upper(), f"{b['energy']:.6f}" if b["done"] else pend("running"), ce(b), ce(s), ce(m_), b["depth"], b["two_q"], b["valid_shots"]])
A(table(["Molecule", "Lineage", "Best E (Eh)", "Best err (mHa)", "Second err (mHa)", "Merged err (mHa)", "Depth", "2q gates", "Valid shots (best)"], rows))
if untrained:
    A("<h3>Untrained controls (randomly initialised policies, independent of the training recipe)</h3>")
    A(table(["Molecule", "Lineage", "Best err (mHa)", "Second err (mHa)", "Depth", "2q gates"],
            [[E(r["label"]), r["lineage"].upper(), fnum(r["t"]["best"].get("err"), 1), fnum(r["t"]["second"].get("err"), 1), r["t"]["best"]["depth"], r["t"]["best"]["two_q"]] for r in untrained]))
fin = [r["t"]["best"]["err"] for r in trained if r["t"]["best"]["done"]]
if fin:
    e = np.array(fin)
    A(f"<p><b>Summary of the {len(fin)} finished trained runs:</b> median error of the best circuit {np.median(e):.2f} mHa; {int((e < 1).sum())} within 1 mHa, {int((e < 5).sum())} within 5 mHa, {int((e < 15).sum())} within 15 mHa. "
      + (f"The other {NPLAN - len(fin)} planned run(s) have no best circuit (single circuit with valid shots)." if NPLAN - len(fin) else "") + "</p>")

# ------------------------------------------------------------------ chemistry at inference
A('<h2 id="chemistry">4. Chemistry of the generated circuits on unseen molecules</h2>')
nochem = [r["tag"] for r in trained if best_chem(r)[0] is None]
if nochem:
    A(f'<div class="note warn"><span class="pend">{len(nochem)} trained runs have no stored chemistry quantities</span> (Training 1 stored them only for 20 runs, recomputed from the stored shots); they are left out of the aggregates and shown as &ndash;.</div>')
keys = [("frac_valid", "valid %"), ("valid_hf", "HF %"), ("valid_homo_lumo_any", "HL %"), ("valid_alpha_only", "&alpha; only %"), ("valid_beta_only", "&beta; only %"), ("valid_alpha_and_beta", "both %")]
agg = defaultdict(lambda: defaultdict(list))
rows = []
for r in trained:
    e, s = best_chem(r)
    if e is None:
        continue
    for k, _ in keys:
        agg[r["lineage"]][k].append(e.get(k))
    for k in ("frac_identity", "n_hf_changing_gates", "frac_givens_homo_lumo", "n_givens"):
        agg[r["lineage"]][k].append(s.get(k))
    rows.append([E(r["label"]), r["lineage"].upper()] + [pct(e.get(k)) for k, _ in keys] + [fnum(s.get("frac_identity"), 2), s.get("n_givens", "&ndash;"), s.get("n_lucj", "&ndash;") if s else "&ndash;",
                 s.get("n_orb_rot", "&ndash;"), s.get("n_diag_coulomb", "&ndash;"), s.get("n_num_num", "&ndash;"), s.get("n_hf_changing_gates", "&ndash;"), fnum(s.get("frac_givens_homo_lumo"), 2)])
A("<h3>Best circuit of every finished run</h3>")
A(table(["Molecule", "Lineage"] + [k[1] for k in keys] + ["identity frac", "Givens", "LUCJ", "orb_rot", "diag_coulomb", "num_num", "HF-changing gates", "Givens on HOMO-LUMO"], rows))
if SET != "t1":
    rowsm = []
    for r in trained:
        d_ = data[r["tag"]]["circuits"]
        ex = [c.get("exc_stats") or {} for c in d_]
        rowsm.append([E(r["label"]), r["lineage"].upper(), pct(float(np.mean([x.get("frac_valid", 0) for x in ex]))), pct(med([x.get("valid_hf") for x in ex if x.get("frac_valid", 0) > 0])),
                      pct(med([x.get("valid_beta_only") for x in ex if x.get("frac_valid", 0) > 0])), pct(med([x.get("valid_alpha_only") for x in ex if x.get("frac_valid", 0) > 0])),
                      int(sum(1 for x in ex if x.get("frac_valid", 0) > 0)), f"{med([c.get('depth_live') for c in d_]):.0f}"])
    A("<h3>All ten circuits of every finished run (mean/median over the circuits that have valid shots)</h3>")
    A(table(["Molecule", "Lineage", "valid % (mean of 10)", "HF % (median)", "&beta; only % (median)", "&alpha; only % (median)", "circuits with valid shots", "median depth (all 10)"], rowsm))
# aggregate figure
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
w = 0.12
labs = [k[1].replace("&alpha;", "a").replace("&beta;", "b") for k in keys]
for i, lin in enumerate(LINS):
    vals = [100 * (med(agg[lin][k]) or 0) for k, _ in keys] if agg[lin][keys[0][0]] else [0] * len(keys)
    ax[0].bar(np.arange(len(keys)) + (i - 1.5) * 0.2, vals, 0.2, label=lin.upper())
ax[0].set_xticks(range(len(keys))); ax[0].set_xticklabels(labs, fontsize=8); ax[0].legend(fontsize=7); ax[0].set_title("median over finished runs, best circuit (%)"); ax[0].grid(alpha=.2)
for i, lin in enumerate(LINS):
    ax[1].bar(i, 100 * (med(agg[lin]["frac_identity"]) or 0), 0.5)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([l.upper() for l in LINS]); ax[1].set_title("identity (padding) tokens in the best circuit's sequence (%)"); ax[1].grid(alpha=.2)
A("<h3>Median over the finished runs, by lineage</h3>" + b64fig(fig))
A('<p>Aggregates are medians over the runs that have a result.</p>')

# ------------------------------------------------------------------ training vs inference
A('<h2 id="tvi">5. Training chemistry against inference chemistry</h2>')
if SET == "t1":
    A('<div class="note warn"><b>Not available for Training 1.</b> Training 1 logged no per-iteration chemistry (no tracking), so there is no training-time curve to correlate with the inference chemistry of section 4. '
      'The same comparison for the two Training 2 sets is in <code>report_hf_penalty.html</code> and <code>report_tuned_reward.html</code>.</div>')
else:
    A("<p>Left block: the last 20 iterations of Stage 2 of the training run, best circuit of each iteration (<code>track.exc_best</code>, <code>track.seq_best</code>) and the mean over the 20 circuits sampled per iteration. "
      "Right block: the same quantities for the zero-shot circuits on the unseen molecules (median over the finished runs of the lineage: best circuit and mean over the ten circuits of a run). "
      "Training molecules are the transition-metal complexes at 23/35 orbitals (a different molecule class); inference molecules are the SnOH<sub>3</sub>-Me family.</p>")
    rows = []
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    ck = ["frac_valid", "valid_hf", "valid_homo_lumo_any", "valid_alpha_only", "valid_beta_only", "valid_alpha_and_beta"]
    cl = ["valid", "HF", "HL", "a-only", "b-only", "both"]
    for j, lin in enumerate(LINS):
        s1, s2 = tst[lin]
        rl = [r for r in trained if r["lineage"] == lin and best_chem(r)[0] is not None]
        inf_best = {k: med([best_chem(r)[0].get(k) for r in rl]) for k in ck}
        inf_mean = {}
        for k in ck:
            xs = []
            for r in rl:
                ex = [c.get("exc_stats") or {} for c in data[r["tag"]]["circuits"]]
                xs.append(float(np.mean([x.get(k, 0) or 0 for x in ex])))
            inf_mean[k] = med(xs)
        sq = {k: med([best_chem(r)[1].get(k) for r in rl]) for k in ("frac_identity", "n_hf_changing_gates", "frac_givens_homo_lumo")}
        for lab, tv, iv in (("best circuit", s2["best"], inf_best), ("mean over circuits", s2["mean"], inf_mean)):
            rows.append([lin.upper(), lab] + [pct(tv[k]) for k in ck] + ["|"] + [pct(iv[k]) if iv[k] is not None else pend("&ndash;") for k in ck] + [len(rl)])
        rows.append([lin.upper(), "sequence (best)", f"eff. vocab {s2['eff_vocab']:.2f}", f"identity {s2['seq']['frac_identity']:.2f}", f"HF-chg gates {s2['seq']['n_hf_changing_gates']:.1f}",
                     f"Givens HL {s2['seq']['frac_givens_homo_lumo']:.2f}", "", "", "|", f"identity {fnum(sq['frac_identity'], 2)}", f"HF-chg gates {fnum(sq['n_hf_changing_gates'], 1)}",
                     f"Givens HL {fnum(sq['frac_givens_homo_lumo'], 2)}", "", "", "", len(rl)])
        x = np.arange(len(ck))
        ax[j].bar(x - 0.2, [100 * (s2["best"][k] or 0) for k in ck], 0.4, label="training (Stage 2, last 20 it.)")
        ax[j].bar(x + 0.2, [100 * (inf_best[k] or 0) for k in ck], 0.4, label="inference (median over runs)")
        ax[j].set_xticks(x); ax[j].set_xticklabels(cl, fontsize=8); ax[j].set_title(lin.upper()); ax[j].grid(alpha=.2)
    ax[0].set_ylabel("% (valid: of 10,000/100,000 shots; others: of valid shots)"); ax[0].legend(fontsize=7)
    A(table(["Lineage", "quantity"] + [c + " % (train)" for c in cl] + [""] + [c + " % (infer)" for c in cl] + ["runs"], rows))
    A("<h3>Best circuit: training against inference</h3>" + b64fig(fig))
    A("<p><b>Reading.</b> " + (
        "In training the HF-penalty policy has collapsed onto about one effective token, a small right-spin-sector share and almost no HF weight, with the 46-qubit circuits strongly beta-only. The zero-shot circuits on the unseen molecules are drawn from the same policy, but the shots of the selected best circuit "
        "keep a much larger HF share and a larger valid share than in training; the beta-only dominance of the 46-qubit policies reappears or weakens depending on the molecule (see the per-run table). The HF penalty therefore reshapes the spin balance of the circuits more than it removes HF weight on the unseen molecules."
        if SET == "hf" else
        "The tuned policy keeps an effective vocabulary of 20-27 tokens and balanced spin channels in training (small HF share, near-zero alpha/beta asymmetry). At inference, compare its valid and HF shares with the HF-penalty report: the tuned circuits have the lowest HF share of the three sets but do not recover closer to DMRG.") +
        " Differences are descriptive; no ablation separates molecule class from recipe.</p>")

# ------------------------------------------------------------------ three way
A('<h2 id="threeway">6. Three-way comparison (cells in which all three sets are finished)</h2>')
try:
    R = pickle.load(open("/tmp/inf/R3.pkl", "rb"))
    cells = sorted([k for k, v in R.items() if {"T1", "HF", "tuned"} <= set(v)], key=lambda k: (ALLMOL.index(k[0]), k[1]))
    SETN = [("T1", "Training 1"), ("HF", "HF penalty"), ("tuned", "tuned reward")]
    rows = []
    for s_, lab in SETN:
        b = [R[k][s_]["best"] for k in cells]
        c2 = [R[k][s_]["second"] for k in cells]
        mn = [min(x["err"], y["err"]) for x, y in zip(b, c2)]
        rows.append([("<b>" + lab + "</b>") if {"t1": "T1", "hf": "HF", "tuned": "tuned"}[SET] == s_ else lab, len(cells), fnum(med([x["err"] for x in b]), 2), fnum(float(np.mean([x["err"] for x in b])), 1),
                     fnum(med([x["err"] for x in c2]), 2), fnum(med(mn), 2), f"{med([x['depth'] for x in b]):.0f}"] + [pct(med([x[q] for x in b])) for q in ("valid", "hf", "hl", "a", "b", "both")])
    A(table(["Set", "cells", "best err median (mHa)", "best err mean (mHa)", "2nd err median", "min(best,2nd) median", "depth (best)", "valid %", "HF %", "HL %", "&alpha; only %", "&beta; only %", "both %"], rows))
    A(f'<p>{len(cells)} cells (molecule x lineage) were run in all three sets: the 24 of the first campaign and the 8 on SnOH3-me M and SnOH3-me-h2o L. Differences of 0.1-1 mHa in the median are below the spread between runs and are not a ranking (one training seed per set).</p>')
    rows = []
    for k in cells:
        row = [E(LABEL[k[0]][0]), k[1].upper()]
        for s_, _ in SETN:
            b, c2 = R[k][s_]["best"], R[k][s_]["second"]
            row += [f"{b['err']:.2f}", f"{c2['err']:.2f}", b["depth"], pct(b["valid"]), pct(b["hf"]), f"{pct(b['a'])}/{pct(b['b'])}/{pct(b['both'])}"]
        rows.append(row)
    hdr = ["Molecule", "Lineage"]
    for _, lab in SETN:
        hdr += [f"{lab}: best err", "2nd err", "depth", "valid %", "HF %", "a/b/both %"]
    A("<details><summary>Per-cell table</summary>" + table(hdr, rows) + "</details>")
except Exception as ex:  # noqa: BLE001
    A(f'<p class="pend">Three-way data not available ({E(str(ex))}).</p>')

# ------------------------------------------------------------------ timing
A('<h2 id="timing">7. Timing and cost of the inference campaign</h2>')
tm = {}
p = ROOT / "timing_sacct.psv"
if p.exists():
    for line in p.read_text().splitlines():
        q = line.split("|")
        if len(q) >= 8:
            tm[q[0]] = {"name": q[1], "state": q[2], "el": int(q[4])}
grp = defaultdict(list)
for v in tm.values():
    kind = v["name"].split("_")[0]
    grp[kind].append(v)
names = {"ca": "Hardware sampling + transpilation (one Runtime job per run)", "rr": "Ranking diagonalization", "rc": "Recovery (2 GPUs)", "xnorb": "Sequence generation / cache", "fin": "Selection"}
rows = []
for k, lab in names.items():
    v = grp.get(k, [])
    ok = [x["el"] for x in v if x["state"] == "COMPLETED"]
    bad = [x for x in v if x["state"] != "COMPLETED"]
    if v:
        rows.append([lab, len(ok), f"{sum(ok) / 3600:.1f}", f"{np.mean(ok) / 60:.0f}" if ok else "&ndash;", len(bad)])
A(table(["Stage (Slurm jobs of this set)", "completed jobs", "wall (h)", "mean per job (min)", "failed/cancelled/timed out"], rows))
if SET == "t1":
    A("<p>Training-1 inference: 32 runs, 320 circuits, 32,000,000 shots, 285 QPU minutes (534 s per job) for the fresh-shot campaign; the QPU time of the first campaign and LUCJ jobs is in the manuscript.</p>")
else:
    qf = REPO / TREE / "qpu_fresh.json"
    if qf.exists():
        qj = json.loads(qf.read_text())
        qs = [v["quantum_seconds"] for k, v in qj["per_run"].items() if not k.endswith("untrained") and v["quantum_seconds"] is not None]
        A(f"<p><b>Billed QPU time of the inference sampling jobs (IBM job records):</b> {len(qs)} trained-run jobs, {sum(qs):,.0f} s ({sum(qs) / 60:.1f} min), "
          f"{np.mean(qs):.0f} s per job (range {min(qs)}&ndash;{max(qs)} s); each run is 10 circuits x 100,000 shots = 1,000,000 shots, one Runtime job. "
          f"Per-run values are in the data folder (<code>qpu_fresh.json</code>). The untrained-control jobs in the same file belong to the Training-1 campaign.</p>")
        if len(qs) < NPLAN:
            A(f'<p><span class="pend">{NPLAN - len(qs)} of the {NPLAN} planned runs are not measured yet; their QPU time is added when they finish.</span></p>')
    else:
        A(f'<p><span class="pend">QPU seconds of this set\'s inference jobs have not been queried yet</span> ({nmeas} runs measured of {NPLAN}; each run is 10 circuits x 100,000 shots = 1,000,000 shots, one Runtime job); {NOTE}. </p>')
    A("<p>Slurm times above are only for the jobs that already finished and are listed in the job table; the ranking jobs (rr_) do not carry a Slurm id in the stored files and are not listed.</p>")

# ---- three-set inference timing (same numbers as the manuscript table, from inference_timing.py)
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("inference_timing", str(Path(__file__).with_name("inference_timing.py"))); _it = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_it)
_rows = []
for _lab, _tree, _fn, _plan in _it.SETS:
    _g, _bad, _run = _it.stats(_tree, _fn)
    _q = _it.qpu(_tree, _plan)
    _rec = _g["rec_bs"] + _g["rec_ed"] + _g["rec_m"]
    _done = len(_g["sample"]) >= _plan and len(_run) == 0
    _c = (lambda t: t) if _done else (lambda t: pend(t))
    _rows.append([_lab, len(_g["sample"]), f"{sum(_g['sample']) / 3600:.1f}", f"{sum(_g['rank']) / 3600:.1f}", _c(f"{len(_rec)}"), _c(f"{sum(_rec) / 3600:.1f}"), _c(f"{2 * sum(_rec) / 3600:.1f}"),
                  f"{_q[0]:,.0f} s ({_q[0] / 60:.0f} min), {_q[0] / max(_q[1], 1):.0f} s/job" if _q and _q[0] is not None else pend("not retrieved")])
A("<h3>The three model sets side by side</h3>" + table(["set", "sampling jobs", "sampling wall (h)", "ranking wall (h)", "recoveries", "recovery wall (h)", "recovery GPU-h", "IBM-billed QPU time (sampling)"], _rows))
A("<p>Trained runs only (Training 1: its 24 original runs). Red: the set is not finished (recoveries still running or queued); regenerate with <code>refresh_all.sh</code>.</p>")
A(f"<p>Training wall time and QPU time of this set are in section 2. Calibration of <code>ibm_kobe</code> during the experiments is copied unchanged in the Training-1 report.</p>")


# ------------------------------------------------------------------ ideal divergence from the LUCJ circuit
DIVDIR = REPO / TREE / "divergence"
MOLNQ = {r['tag']: r['nq'] for r in runs}
def div_load(tag):
    f = DIVDIR / f"{tag}.json"
    if not f.exists():
        return None
    return {c["index"]: c for c in json.loads(f.read_text())["circuits"]}
def dfmt(x, p=3):
    return "&ndash;" if x is None else f"{x:.{p}f}"
DIV_NA = '<span class="mut">n/a</span>'

A('<h2 id="div">7b. Divergence of the generated circuits from the idealised LUCJ circuit</h2>')
A("<p>For every generated circuit the ideal (noiseless) state U<sub>g</sub>|HF&gt; is compared with the LUCJ state U<sub>LUCJ</sub>|HF&gt; "
  "(<code>ffsim.UCJOpSpinBalanced.from_t_amplitudes</code>, n_reps = 1, the circuit of the hardware LUCJ baseline), with exact statevectors in the full N-electron space. "
  "<b>F</b> = |&lt;&psi;<sub>g</sub>|&phi;&gt;|&sup2; is the state fidelity (Jozsa 1994; Nielsen &amp; Chuang 2010); <b>H</b> = &radic;(1 &minus; BC) is the Hellinger distance (Hellinger 1909) with BC = &Sigma;&radic;(pq) the Bhattacharyya coefficient (Bhattacharyya 1943) "
  "of the determinant distributions p = |&psi;<sub>g</sub>|&sup2;, q = |&phi;|&sup2;; <b>JS</b> is the Jensen&ndash;Shannon divergence in bits (Lin 1991), which is bounded and finite for disjoint supports, which is why the Kullback&ndash;Leibler divergence is not used (it is infinite when the supports differ); "
  "<b>d<sub>HF</sub></b> = &radic;(2 &minus; 2|&lt;&psi;<sub>g</sub>|&phi;&gt;|) is the distance of the two states up to a global phase, a lower bound of the operator distance between the circuits; "
  "<b>op-norm LB</b> is a lower bound of the operator norm &#8741;U<sub>g</sub> &minus; U<sub>LUCJ</sub>&#8741; (maximum 2) from a Lanczos iteration with exact inverses of every pool operator (Lanczos 1950), computed only for the best and second circuit of the 16-orbital tier. "
  "Only tiers whose full N-electron vector fits in memory (SnOH3-me S, SnOH3-me-h2o S and SnOH3-me-2h2o S: 16, 19 and 22 orbitals, up to 44 qubits) are computed; larger tiers and SnOH3-me M (38 qubits, 12+12 electrons in 19 orbitals: 2.5&times;10<sup>9</sup> determinants, 41 GB per state vector) are n/a. The ideal state is not what the hardware samples: noise spreads the measured shots over excited determinants (see the manuscript).</p>")
drows = []
for r in runs:
    dv = div_load(r["tag"])
    if not dv:
        continue
    sel = r["sel"]
    for role, idx in (("best", sel["best"]), ("2nd", sel["second"])):
        c = dv.get(idx)
        if c:
            drows.append([E(r["label"]), r["lineage"].upper(), role, dfmt(c["F"]), dfmt(c["H"]), dfmt(c["JS"]), dfmt(c["d_HF"]), dfmt(c.get("op_norm"), 2) if c.get("op_norm") is not None else "&ndash;"])
if drows:
    A(table(["target", "lineage", "circuit", "F", "H", "JS", "d_HF", "op-norm LB"], drows))
else:
    A('<p>' + pend("divergence not computed yet for this set") + '</p>')

# ------------------------------------------------------------------ per-run detail

A('<h2 id="runs">8. Per-run detail</h2>')
for r in runs:
    tag = r["tag"]
    rk = {c["index"]: c for c in rerank[tag]["circuits"]}
    sel = r["sel"]
    A(f'<details id="{E(tag)}"><summary>{E(r["label"])} &mdash; {r["lineage"].upper()} {r["policy"]}</summary>')
    A(f"<p class=\"mut\">DMRG reference {DMRG[r['mol']]:.9f} Eh. Hardware job ids: {E(', '.join(data[tag].get('ibm_runtime_job_ids') or []))}. Selected: best = circuit {sel['best'] + 1}, second = {sel['second'] + 1}, "
      f"merged = {', '.join(str(i + 1) for i in sel['merged'])}.</p>")
    crow = []
    for c in data[tag]["circuits"]:
        rr_ = rk.get(c["index"], {})
        e, s, hf, imb = circ_chem(tag, c["index"])
        e = e or {}
        mark = ("best " if c["index"] == sel["best"] else "") + ("2nd" if c["index"] == sel["second"] else "")
        dv = (div_load(tag) or {}).get(c["index"])
        dcols = ([dfmt(dv["F"]), dfmt(dv["H"]), dfmt(dv["JS"]), dfmt(dv["d_HF"]), dfmt(dv.get("op_norm"), 2) if dv.get("op_norm") is not None else "&ndash;"]
                 if dv else [DIV_NA if (MOLNQ[tag] > 44 or tag.startswith("sno38_")) else pend("not computed")] * 5)
        crow.append([c["index"] + 1, mark, c.get("depth_live"), c.get("two_qubit_live"), rr_.get("valid_shots"), fnum(rr_.get("energy_40000"), 5) if rr_.get("energy_40000") is not None else "&ndash;",
                     pct(e.get("frac_valid")) if e else "&ndash;", pct(e.get("valid_hf")) if e else "&ndash;", pct(e.get("valid_homo_lumo_any")) if e else "&ndash;",
                     "/".join(pct(e.get(k)) for k in ("valid_alpha_only", "valid_beta_only", "valid_alpha_and_beta")) if e else "&ndash;",
                     fnum(s.get("frac_identity"), 2) if s else "&ndash;", s.get("n_givens", "&ndash;") if s else "&ndash;", s.get("n_hf_changing_gates", "&ndash;") if s else "&ndash;"] + dcols)
    A("<h4>The ten circuits</h4>" + table(["#", "", "depth", "2q", "valid shots", "ranking E (Eh)", "valid %", "HF %", "HL %", "a/b/both %", "identity frac", "Givens", "HF-chg gates", "fidelity F", "Hellinger H", "JS (bits)", "d_HF", "op-norm LB"], crow))
    rrows = []
    for tg in ("best", "second", "merged", "oldbest", "oldsecond"):
        en = r["t"][tg]
        if not en["done"]:
            rrows.append([tg, pend("recovery running"), "", ""])
            continue
        rrows.append([tg if not en["alias"] else f"{tg} (= {en['alias']})", f"{en['energy']:.6f}", f"{en['err']:.2f}", fnum(en.get("si"), 2)])
    A("<h4>Recoveries</h4>" + table(["result", "energy (Eh)", "error (mHa)", "error after singles injection (mHa)"], rrows))
    A("</details>")

# ------------------------------------------------------------------ raw
A('<h2 id="raw">9. Raw data</h2>')
dd = OUTDIR / f"report_{NAMES[SET]}_data"
dd.mkdir(parents=True, exist_ok=True)
for r in runs:
    tag = r["tag"]
    rawj = {"tag": tag, "molecule": r["label"], "dmrg": DMRG[r["mol"]], "selection": r["sel"], "circuits": data[tag]["circuits"], "rerank": rerank[tag]["circuits"],
            "recoveries": {tg: {k: v for k, v in x.items() if k != "recovered_alpha_strings"} for tg, x in rec.get(tag, {}).items()},
            "ibm_runtime_job_ids": data[tag].get("ibm_runtime_job_ids"), "stage2_chemistry_fields": "exc_stats, seq_stats, hf_fraction, spin_imbalance per circuit"}
    (dd / f"{tag}.json").write_text(json.dumps(rawj))
with open(dd / "summary_best_circuit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "molecule", "lineage", "policy", "best_energy", "best_err_mha", "second_err_mha", "merged_err_mha", "depth", "two_q", "valid_shots", "frac_valid", "valid_hf", "valid_homo_lumo_any", "alpha_only", "beta_only", "both"])
    for r in runs:
        b, s, m_ = r["t"]["best"], r["t"]["second"], r["t"]["merged"]
        e, _ = best_chem(r)
        e = e or {}
        w.writerow([r["tag"], r["label"], r["lineage"], r["policy"], b.get("energy", ""), b.get("err", ""), s.get("err", ""), m_.get("err", ""), b["depth"], b["two_q"], b["valid_shots"],
                    e.get("frac_valid", ""), e.get("valid_hf", ""), e.get("valid_homo_lumo_any", ""), e.get("valid_alpha_only", ""), e.get("valid_beta_only", ""), e.get("valid_alpha_and_beta", "")])
A(f'<p>Per-run JSON (circuits with sequences, shot-level chemistry, ranking and recovery results; the recovered alpha strings are omitted) and a CSV in <code>report_{NAMES[SET]}_data/</code> next to this file: '
  f'<a href="report_{NAMES[SET]}_data/summary_best_circuit.csv">summary_best_circuit.csv</a>; '
  + ", ".join(f'<a href="report_{NAMES[SET]}_data/{E(r["tag"])}.json">{E(r["tag"])}</a>' for r in runs) + '. Complete counts and the transpiled circuits stay in the Zenodo archive of the first campaign; those of this set are on the cluster '
  f'(<code>experiments/inference_{"t1" if SET == "t1" else SET}/circuit_analysis_ibm_jobs</code>) (the circuit archive and raw counts are in the deposit).</p>')
A("</main></body></html>")
OUTDIR.mkdir(exist_ok=True)
path = OUTDIR / f"report_{NAMES[SET]}.html"
path.write_text("\n".join(out))
print("wrote", path, f"{path.stat().st_size / 1e6:.2f} MB")
