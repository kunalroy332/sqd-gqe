# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Build the self-contained data-description page (for the Zenodo deposit) from the stored result files.

Inputs (repo-root relative): circuit_analysis/{data,rerank,rerecovery,connectivity}, timing_sacct.psv,
raw_archive/circuit_analysis_archive/<tag>/manifest.json, figures/<tag>.png.
Output: circuit_analysis/zenodo/sqd-gqe_zenodo.html (figures embedded as base64, one file).
"""
import base64
import html
import json
from pathlib import Path

from tables_data import DMRG, LABEL, LIN_ORDER, ROOT, TARGETS, coverage_matrix, load, missing_runs, run_key

LIN = ["a1-46q", "a1-70q", "a5-46q", "a5-70q"]
E = html.escape
TRAIN_NORB = {"46q": 23, "70q": 35}


def fmt_err(e):
    if not e["done"] and e.get("partial"):
        x = e["p_err"]
        return f'&le;{x:.2f}&dagger;' if abs(x) < 1 else f'&le;{x:.1f}&dagger;'
    if not e["done"]:
        return '<span class="pend">not finished</span>'
    x = e["err"]
    return f"{x:.2f}" if abs(x) < 1 else f"{x:.1f}"


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def load_timing():
    t = {}
    for line in (ROOT / "timing_sacct.psv").read_text().splitlines():
        p = line.split("|")
        if len(p) >= 8:
            t[p[0]] = {"name": p[1], "state": p[2], "elapsed": int(p[4]), "start": p[5], "end": p[6], "node": p[7]}
    return t


def table(headers, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join(r if isinstance(r, str) else "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def main():
    d = load()
    tm = load_timing()
    runs = sorted(d.values(), key=run_key)
    data = {r["tag"]: json.loads((ROOT / "data" / f"{r['tag']}.json").read_text()) for r in runs}
    rerank = {r["tag"]: json.loads((ROOT / "rerank" / f"{r['tag']}.json").read_text()) for r in runs}
    conn = {r["tag"]: json.loads((ROOT / "connectivity" / f"{r['tag']}.json").read_text()) for r in runs}
    man = {r["tag"]: json.loads((ROOT / "raw_archive/circuit_analysis_archive" / r["tag"] / "manifest.json").read_text()) for r in runs}
    rec = {}
    for f in (ROOT / "rerecovery").glob("*__*.json"):
        tag, tg = f.stem.split("__")
        rec.setdefault(tag, {})[tg] = json.loads(f.read_text())
    npend = sum(1 for r in runs for e in r["t"].values() if not e["done"])
    ncell = sum(len(r["t"]) for r in runs)

    out = []
    A = out.append
    A(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SQD-GQE sampled-circuit dataset: description</title><style>
:root{{--bg:#fff;--fg:#1b1f24;--mut:#59636e;--line:#d0d7de;--acc:#0b5cad;--card:#f6f8fa;--warn:#9a6700}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--acc:#58a6ff;--card:#161b22;--warn:#d29922}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 80px}}
h1{{font-size:1.7rem;margin:.2em 0}}h2{{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.2em}}h3{{margin-top:1.6em}}
a{{color:var(--acc)}}code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}}
.tw{{overflow-x:auto;margin:.8em 0}}table{{border-collapse:collapse;font-size:.85rem;white-space:nowrap}}
th,td{{border:1px solid var(--line);padding:3px 8px;text-align:right}}th{{background:var(--card);position:sticky;top:0}}
td:first-child,th:first-child{{text-align:left}}td.l,th.l{{text-align:left}}
.note{{background:var(--card);border-left:4px solid var(--acc);padding:8px 14px;margin:1em 0}}
.warn{{border-left-color:var(--warn)}}.pend{{color:var(--warn);font-style:italic}}
details{{border:1px solid var(--line);border-radius:6px;margin:10px 0;padding:6px 12px}}summary{{cursor:pointer;font-weight:600}}
img{{max-width:100%;height:auto}}nav a{{margin-right:12px}}.mut{{color:var(--mut)}}
</style></head><body><main>
<h1>Sampled-circuit dataset for the amortized SQD-GQE manuscript</h1>
<p class="mut">Description of the hardware-sampled circuits, the raw and recovered excitation structure, circuit depth and
two-qubit connectivity, ranking and recovery energies, and timing.</p>
<nav><a href="#overview">Overview</a><a href="#defs">Definitions</a><a href="#molecules">Molecules</a><a href="#policies">Policies</a>
<a href="#summary">Summary table</a><a href="#coverage">Coverage &amp; connectivity</a><a href="#timing">Timing</a><a href="#runs">Per-run detail</a>
<a href="#collage">Collages</a><a href="#calibration">Calibration</a><a href="#cost">Cost</a></nav>
<h2 id="overview">1. What is in this deposit</h2>
<p>{len(runs)} runs. A run is one trained (or untrained-control) policy sampled zero-shot on one unseen SnOH<sub>3</sub>-Me-family
molecule. Each run has <b>10 circuits</b> sampled from the policy; each circuit was executed on IBM <code>ibm_kobe</code> with
<b>100,000 shots</b> (transpile <code>optimization_level=1</code>, <code>seed_transpiler=0</code>, no dynamical decoupling, no error mitigation).
The raw counts and the transpiled circuits of every run are stored, so every number below can be recomputed without further hardware time.</p>
<div class="note">Each run was <em>re-sampled</em> on hardware to obtain per-circuit counts; the first campaign stored counts only for the two
circuits it had selected. All values here come from the re-sampled data. {ncell - npend} of {ncell} (run, result-type) recoveries are finished.{'' if not npend else ' The others are still running on the cluster.'} Nothing was estimated.</div>
<h3>Pipeline</h3>
<ol><li><b>Sample</b>: the policy generates 10 gate sequences; each is built as a circuit, transpiled and run for 100,000 shots.</li>
<li><b>Rank</b>: each circuit's electron-number-conserving shots are expanded to alpha strings and diagonalized once in a subspace of up to
<b>200</b> most-measured alpha strings (net CI dimension up to 40,000). This is the <em>ranking energy</em>. A circuit with no valid shot (no shot with the correct alpha and beta electron numbers) yields no determinant to diagonalize, so its ranking energy is undefined; it is shown as "n/a", not as an infinite energy, and the circuit is excluded from selection.</li>
<li><b>Select</b>: <i>best</i> and <i>second</i> are the two circuits with the lowest ranking energy. <i>merged</i> pools the shots of every circuit with a finite ranking energy.
<i>E+D&nbsp;1</i> and <i>E+D&nbsp;2</i> are the two circuits with the lowest sum of the energy rank and the distinct-bitstring rank (the same 40,000-determinant ranking; the eight untrained controls use the energy rank of an earlier 2,000-determinant ranking).</li>
<li><b>Recover</b>: each selected set of shots goes through the iterative configuration-recovery SQD loop (GPU, subspace dimension 20,000, 25 iterations, seed 1234).</li></ol>
<h2 id="defs">2. Definitions used in every table</h2>
<ul>
<li><b>&Delta;E (mHa)</b> = E<sub>SQD</sub> &minus; E<sub>DMRG</sub> of that molecule, times 1000 (signed). Positive means the SQD energy is above DMRG.</li>
<li><b>Valid shots</b>: shots (out of 100,000) with the correct alpha and beta electron numbers. Only these enter the ranking and the recovery.</li>
<li><b>Raw excitation</b> (HF, S, D, T, Q, Higher): classification of the <i>measured bitstrings themselves</i> by excitation rank relative to the Hartree-Fock determinant, counted over all 100,000 shots weighted by multiplicity.</li>
<li><b>Subspace excitation</b>: the same classification applied to the <i>determinant pairs</i> (alpha, beta) of the n&times;n product of recovered alpha strings that the solver diagonalizes. These counts total n<sup>2</sup>, not 100,000, and "Higher" is dominated by the pairing itself, so they are not a statement about individual shots.</li>
<li><b>Depth / 2q (live, used everywhere)</b>: depth and two-qubit-gate count of the circuit as transpiled and submitted to <code>ibm_kobe</code> (<code>optimization_level=1</code>, routing onto the real device included), stored in <code>circuits.qpy</code>. All summary tables, the paper and this page quote these.</li>
<li><b>Depth / 2q (offline estimate)</b>: the same circuit transpiled against the <code>ibm_heron_r2</code> basis and connectivity without a live backend. It is reported only in the per-circuit tables for reference; it is typically smaller for deep circuits because it does not include the routing choices made for the real device.</li>
<li><b>Logical connectivity</b>: the generated circuit translated to {{rz, sx, cx}} <em>without</em> a coupling map. Two qubits are "connected" if at least one two-qubit gate acts on them. Reported: qubits touched by any gate, qubits in a two-qubit gate, distinct connected pairs, pair density (pairs / C(n,2)), connected components of the pair graph, and the size of the largest component.</li>
<li><b>Physical connectivity</b>: the same on the stored transpiled circuit, i.e. on the heavy-hex couplers actually used (limited by device topology, so far fewer distinct pairs).</li>
</ul>
""")

    # molecules
    mrows = []
    seen = {}
    for r in runs:
        m = man[r["tag"]]
        seen.setdefault(r["mol"], m)
    for mk, m in sorted(seen.items(), key=lambda kv: (LABEL[kv[0]][1], LABEL[kv[0]][2])):
        mrows.append([E(LABEL[mk][0]), m["norb"], E(str(m["nelec"])), m["n_qubits"], f"{DMRG[mk]:.9f}", E(m["molecule_path"])])
    A('<h2 id="molecules">3. Molecules</h2><p>Active spaces of the Sn(OH)<sub>3</sub>-Me family (bare, mono-hydrate "h2o", di-hydrate "2h2o"), MP2 natural orbitals, def2-SVP. '
      'Qubits = 2 &times; orbitals. DMRG reference energies are from bond dimension 4000, 17 sweeps.</p>')
    A(table(["Molecule (qubits)", "Orbitals", "Electrons (&alpha;,&beta;)", "Qubits", "DMRG (E<sub>h</sub>)", "Definition file"], mrows))

    # policies
    prow = []
    seenp = {}
    for r in runs:
        m = man[r["tag"]]
        key = (r["lineage"], r["policy"])
        seenp.setdefault(key, m)
    for (lin, pol), m in sorted(seenp.items(), key=lambda kv: (not kv[0][0].startswith("a1"), LIN.index(kv[0][0]), kv[0][1] != "trained")):
        q = lin.split("-")[1]
        enc = "scalar-descriptor cross-attention (A1)" if lin.startswith("a1") else "PosEGNN structure embedding (A5)"
        if pol == "trained":
            ck = f"<code>{E(Path(m['checkpoint']).name)}</code><br>sha256 <code>{(m['checkpoint_sha256'] or '')[:16]}&hellip;</code>"
        else:
            ck = "none (randomly initialised policy, same architecture)"
        prow.append([lin.upper(), pol, E(enc), f"trained at {TRAIN_NORB[q]} orbitals ({q})" if pol == "trained" else "n/a", ck])
    A('<h2 id="policies">4. Policies</h2><p>Four trained lineages (A1/A5 &times; 46q/70q) and, as controls, untrained policies of the same architecture. '
      'A target whose orbital count differs from the lineage\'s training orbital count is a <b>cross-norb</b> case: the operator vocabulary is rebuilt per molecule, '
      'so token indices need not mean the same gate as in training.</p>')
    A(table(["Lineage", "Policy", "Conditioning", "Training active space", "Checkpoint"], prow))

    cm = coverage_matrix(d)
    mx = [[E(LABEL[mk][0])] + [{"done": "done", "queued": '<span class="pend">queued</span>', "not run": "&ndash;"}[cm[mk][(p, l)]] for p in ("trained", "untrained") for l in LIN_ORDER]
          for mk in sorted(LABEL, key=lambda k: (LABEL[k][1], LABEL[k][2]))]
    A('<h2 id="matrix">4b. Which runs exist</h2><p><b>done</b>: sampled, ranked and recovered. '
      '&ndash;: not part of the design (untrained controls exist for a subset of targets). Columns are the lineage (A1/A5, trained at 46 or 70 qubits).</p>')
    A(table(["Target"] + [f"Trained {l.upper()}" for l in LIN_ORDER] + [f"Untrained {l.upper()}" for l in LIN_ORDER], mx))

    # summary
    def srow(r):
        if r.get("missing"):
            return (f'<tr><td>{E(r["label"])}</td><td>{r["lineage"].upper()} {r["policy"]}</td><td class="l" colspan="8"><i>queued: to be run when the cluster is available '
                    '(job chain prepared in the data release)</i></td></tr>')
        t = r["t"]
        return [E(r["label"]), f'<a href="#{r["tag"]}">{r["lineage"].upper()} {r["policy"]}</a>',
                fmt_err(t["best"]), fmt_err(t["second"]), fmt_err(t["merged"]), fmt_err(t["oldbest"]), fmt_err(t["oldsecond"]),
                f"{t['best']['depth']}/{t['best']['two_q']}", f"{t['best']['valid_shots']}/{t['second']['valid_shots']}",
                "yes" if t["best"].get("has_alpha") else "no" if t["best"]["done"] else "&ndash;"]
    allr = sorted(list(runs) + missing_runs(), key=run_key)
    sec = lambda txt: f'<tr><th class="l" colspan="10">{txt}</th></tr>'
    srows = [sec("Trained policies")] + [srow(r) for r in allr if r["policy"] == "trained"] + \
            [sec("Untrained controls")] + [srow(r) for r in allr if r["policy"] != "trained"]
    A('<h2 id="summary">5. Summary: recovered error for every run</h2>'
      '<p>Rows run from the smallest active space (32 qubits) to the largest (70 qubits); within a molecule the A1 policies come first, then the A5 policies. Trained policies and untrained controls are listed in separate blocks.</p>'
      '<div class="note"><b>Two ways of choosing circuits, and why there are five result columns.</b> '
      '<b>best</b> and <b>second</b> are the two circuits (of the 10 sampled) with the lowest ranking energy, where the ranking energy is one diagonalization of that circuit&rsquo;s own valid shots in a subspace of up to 200 alpha strings (net dimension 40,000). '
      '<b>E+D 1</b> and <b>E+D 2</b> are the two circuits with the lowest sum of two ranks: the rank of the ranking energy (the same 40,000-determinant ranking; only the eight untrained controls use the energy rank of an earlier 2,000-determinant ranking) '
      'and the rank by number of distinct measured bitstrings (more is better); the circuit with the best combined rank is E+D 1 and the next is E+D 2. They are different circuits whenever energy alone and energy-plus-distinctness disagree, '
      'and they are the same circuit as best/second when the rules agree (then the result is simply reused). <b>merged</b> pools the shots of all circuits with a finite ranking energy. '
      'All five are recovered with the same 20,000-dimensional, 25-iteration procedure.</div>'
      '<p>Error in mHa above DMRG. Depth/2q are the as-submitted (live) cost of the best circuit. "Valid shots" are for the best and second circuit; a pick that rests on a few dozen valid shots is a weak pick whatever its ranking energy.</p>')
    A(table(["Target", "Lineage / policy", "best", "second", "merged", "E+D 1", "E+D 2", "Depth/2q (best)", "Valid shots (best/2nd)", "Recovered alpha strings saved"], srows))

    # coverage & connectivity
    crows = []
    for r in runs:
        cs = conn[r["tag"]]["circuits"]
        n = conn[r["tag"]]["n_logical_qubits"]
        full = sum(1 for c in cs if c["logical"]["qubits_touched_any_gate"] == n)
        one = sum(1 for c in cs if c["logical"]["components"] == 1 and c["logical"]["qubits_in_2q_gates"] == n)
        alltoall = sum(1 for c in cs if c["logical"]["distinct_2q_pairs"] == n * (n - 1) // 2)
        meanp = sum(c["logical"]["distinct_2q_pairs"] for c in cs) / len(cs)
        pp = sum(c["physical"]["distinct_2q_pairs"] for c in cs) / len(cs)
        crows.append([E(r["label"]), f'{r["lineage"].upper()} {r["policy"]}', n, f"{full}/10", f"{one}/10", f"{alltoall}/10", f"{meanp:.0f}",
                      f"{pp:.0f}", f"{min(c['logical']['largest_component'] for c in cs)}&ndash;{max(c['logical']['largest_component'] for c in cs)}"])
    A("""<h2 id="coverage">6. Circuit coverage and two-qubit connectivity</h2><p>Rows are ordered from the smallest to the largest active space, A1 policies before A5.</p>
<p><b>Coverage</b>: how many of the target's qubits a circuit touches. <b>Connectivity</b>: which qubits are linked by two-qubit gates, and whether they form one connected
block. Both are computed for every one of the 10 circuits of every run (see the per-circuit tables in Section 8 for the individual values).</p>
<div class="note"><b>Why every count is "out of 10".</b> Each run samples exactly 10 circuits from its policy, so each of the first three count columns says in how many of those 10 circuits a property holds
(0/10 = never, 10/10 = every circuit). The shots (100,000 per circuit) are not involved here; connectivity is a property of the circuit, not of its measurement outcomes.<br>
<b>Circuits touching all qubits</b>: circuits in which every qubit of the register has at least one gate acting on it. Fewer than 10 means some circuits leave qubits idle.<br>
<b>Circuits with all qubits in one 2q-connected block</b>: draw a graph with one node per qubit and an edge between two qubits whenever at least one two-qubit gate acts on that pair. A circuit counts here if this graph is a single connected piece containing all qubits, so entanglement can in principle spread between any two qubits through chains of two-qubit gates. Otherwise the register splits into separate blocks, for example an alpha-spin half and a beta-spin half that never interact.<br>
<b>Circuits with all-to-all logical pairs</b>: circuits in which every possible pair of qubits (n(n&minus;1)/2 pairs) has a direct two-qubit gate, ignoring the device layout (&ldquo;logical&rdquo;). This happens for circuits that contain full-register operators such as orbital rotations or LUCJ. It is a stricter property than the previous column.<br>
<b>Mean distinct logical 2q pairs</b>: the number of different qubit pairs that share at least one two-qubit gate in the logical circuit, averaged over the 10 circuits (maximum n(n&minus;1)/2).<br>
<b>Mean distinct physical 2q pairs (couplers)</b>: the same count on the circuit that actually ran on <code>ibm_kobe</code>, i.e. the number of distinct physical couplers used, averaged over the 10 circuits. It is much smaller than the logical count because the device is a heavy-hex lattice where each qubit has at most three neighbours, so long-range pairs are routed through swaps.<br>
<b>Largest connected block (min&ndash;max over 10)</b>: in each circuit, the size of the biggest group of qubits that are all linked through two-qubit gates (the largest connected piece of the graph above). The cell shows the smallest and the largest such size among the 10 circuits; a value equal to the number of qubits means that circuit is fully connected, and half the number of qubits typically means two separate spin blocks.</div>
<div class="note warn">Coverage is measured at gate level: a qubit is "touched" if at least one gate of the translated circuit acts on it. This is stricter than the older
<code>active_qubits</code> field (which named any qubit appearing in a composite instruction) and it shows that some circuits leave qubits idle.</div>
<div class="note">Reading the table: a circuit whose gates only act inside the alpha-spin qubits and inside the beta-spin qubits (never between the two halves) has
<b>two connected components of n/2 qubits each</b>; a circuit containing a full-register operator (orbital rotation, diagonal Coulomb, LUCJ) links <b>all n qubits in one component</b> and,
before routing, has an all-to-all pair graph. After routing onto the heavy-hex device, the number of distinct physical pairs is limited by the coupler count.</div>""")
    A(table(["Target", "Lineage / policy", "Qubits", "Circuits touching all qubits", "Circuits with all qubits in one 2q-connected block",
             "Circuits with all-to-all logical pairs", "Mean distinct logical 2q pairs", "Mean distinct physical 2q pairs (couplers)", "Largest connected block (min&ndash;max over 10)"], crows))

    # timing
    trows = []
    for r in runs:
        cj = data[r["tag"]].get("slurm_job_id")
        ca = tm.get(str(cj))
        rr = next((v for v in tm.values() if v["name"] == f"rr_{r['tag']}" and v["state"] == "COMPLETED"), None)
        cells = []
        for tg in ("best", "second", "merged", "oldbest", "oldsecond"):
            x = rec.get(r["tag"], {}).get(tg if not r["t"][tg]["alias"] else r["t"][tg]["alias"])
            sj = str(x["slurm_job_id"]) if x else None
            cells.append(hms(tm[sj]["elapsed"]) if sj in tm else ('<span class="pend">&ndash;</span>'))
        trows.append([E(r["label"]), f'{r["lineage"].upper()} {r["policy"]}', hms(ca["elapsed"]) if ca else "&ndash;", hms(rr["elapsed"]) if rr else "&ndash;"] + cells)
    import statistics as st
    rows_t = []
    for r in runs:
        for tg, x in rec.get(r["tag"], {}).items():
            sj = str(x["slurm_job_id"])
            if sj in tm:
                rows_t.append((x["excitation"]["n_dets"], tm[sj]["elapsed"], tg))
    def band(lo, hi):
        v = [t for nd, t, _ in rows_t if lo <= nd < hi]
        return [f"{lo:.0e}&ndash;{hi:.0e}" if hi < 1e12 else "&ge; 4e+08 (cap)", len(v), f"{min(v) / 60:.0f}", f"{st.median(v) / 60:.0f}", f"{max(v) / 60:.0f}"] if v else None
    bands = [b for b in (band(0, 1e7), band(1e7, 1e8), band(1e8, 3.99e8), band(3.99e8, 1e13)) if b]
    kind = []
    for lab, keys in (("best / second", ("best", "second")), ("E+D 1 / E+D 2", ("oldbest", "oldsecond")), ("merged", ("merged",))):
        v = [t for nd, t, tg in rows_t if tg in keys]
        kind.append([lab, len(v), f"{min(v) / 60:.0f}", f"{st.median(v) / 60:.0f}", f"{max(v) / 60:.0f}"])
    A("""<h2 id="timing">7. Timing</h2>
<p>Wall-clock times are Slurm elapsed times (<code>sacct</code>) on the RIKEN roquo cluster; the file <code>timing_sacct.psv</code> lists every job. Rows are ordered from the smallest to the largest active space, A1 before A5.</p>
<div class="note"><b>What the columns are.</b> <b>Sample + transpile</b>: one hardware round trip for the 10-circuit batch plus circuit construction and transpilation, including time in IBM&rsquo;s device queue.
<b>Rank</b>: the single 40,000-dimensional ranking diagonalization of all 10 circuits.
<b>Recovery (diagonalization) &ndash; best, second, merged, E+D 1, E+D 2</b>: the wall time of the 25-iteration SQD recovery for that result type. Each recovery is <b>25 successive GPU diagonalizations</b> of the Hamiltonian projected on the current set of determinants
(two GPUs, subspace dimension 20,000 alpha strings); no hardware is involved. best/second are the two lowest-ranking-energy circuits, E+D 1/E+D 2 the two circuits chosen by the energy-rank-plus-distinct-bitstring rule (Section 5);
a result that reuses another circuit&rsquo;s recovery because the picks coincide shows that job&rsquo;s time. IBM-side queue and execution times of the fresh hardware jobs were not queried; the IBM job ids are in the per-run tables.</div>
<h3>Why recovery times range from minutes to many hours</h3>
<p>The cost of one diagonalization grows with the number of determinants in the projected subspace, which is the square of the number of recovered alpha strings (alpha &times; beta product). It is therefore set by two things: how large the active space is, and how many distinct alpha strings the shots supply.</p>
<ul>
<li><b>Active-space size caps the subspace.</b> For the 32-qubit target there are only a few thousand alpha strings in total, so the net dimension stays near 10<sup>6</sup>&ndash;3&times;10<sup>6</sup> and a single-circuit recovery takes minutes. At 46 qubits and above the recovered set usually reaches the 20,000-alpha-string cap, i.e. 4&times;10<sup>8</sup> determinants (smaller sets occur when a circuit has few valid shots), and each of the 25 diagonalizations is orders of magnitude larger; the medians in the tables below show the effect.</li>
<li><b>Merged pools about ten times the shots.</b> It combines the shots of all finite-energy circuits (up to 1,000,000 shots), so it supplies many more distinct alpha strings and reaches the cap immediately, whereas a single circuit with few valid shots supplies few strings and a smaller subspace. The measured medians below show merged taking about twice as long as a single circuit.</li>
<li><b>Few valid shots means a small subspace.</b> A circuit with tens of valid shots yields only tens of alpha strings in the first iteration, so its early diagonalizations are cheap; later iterations add configurations and the set grows.</li>
<li><b>Part of the spread is not explained by dimension.</b> Jobs at the same capped dimension range from under an hour to about ten hours (for example two recoveries on the SnOH3-me-2h2o L (70q) target, both at the cap, took 1.4 h and 8.5 h). The number of solver iterations per diagonalization depends on how well the starting vectors resemble the ground state and was not recorded, and up to about 40 of this project&rsquo;s own recovery jobs ran concurrently on shared nodes, whose effect on speed was not measured. We therefore do not attribute the remaining spread to a single cause.</li>
<li>Note that rejected iterations (energy jumps rejected by the 0.5 E<sub>h</sub> guard) still cost a full diagonalization, so a run with many rejections is not cheaper.</li>
</ul>
<p>Measured, over all finished recoveries:</p>""")
    A(table(["Net determinants at the last iteration", "jobs", "min (min)", "median (min)", "max (min)"], bands))
    A(table(["Result type", "jobs", "min (min)", "median (min)", "max (min)"], kind))
    A(table(["Target", "Lineage / policy", "Sample + transpile", "Rank", "Recovery (25 diagonalizations): best", "second", "merged", "E+D 1", "E+D 2"], trows))

    # per-run
    A('<h2 id="runs">8. Per-run detail</h2><p>Runs are ordered from the smallest to the largest active space, A1 before A5. In the circuit tables the role column uses the same names as Section 5 (best/second = two lowest ranking energies; E+D 1/E+D 2 = energy-rank-plus-distinct-bitstring picks). Each section: figure, per-circuit table, recoveries, operator sequences. Click to expand.</p>')
    for r in runs:
        tag = r["tag"]
        run = data[tag]
        rr = {c["index"]: c for c in rerank[tag]["circuits"]}
        cn = {c["index"]: c for c in conn[tag]["circuits"]}
        m = man[tag]
        new_rank = {c["index"]: k + 1 for k, c in enumerate(sorted((c for c in rr.values() if c["energy_40000"] is not None), key=lambda c: c["energy_40000"]))}
        sel = r["sel"]
        img = base64.b64encode((ROOT / "figures" / f"{tag}.png").read_bytes()).decode()
        crow = []
        for c in run["circuits"]:
            i = c["index"]
            tags = []
            if new_rank.get(i) == 1: tags.append("best")
            if new_rank.get(i) == 2: tags.append("second")
            if i == sel["old_best"]: tags.append("E+D 1")
            if i == sel["old_second"]: tags.append("E+D 2")
            h = c["raw_hist"]
            ct = cn[i]
            lg, ph = ct["logical"], ct["physical"]
            en = rr[i]["energy_40000"]
            crow.append([i + 1, E(", ".join(tags)) or "&ndash;", c["depth_live"], c["two_qubit_live"], rr[i]["depth"], rr[i]["two_qubit_gates"],
                         rr[i]["valid_shots"], rr[i]["valid_distinct"], f"{en:.4f}" if en is not None else "n/a (0 valid shots)",
                         int(h["HF"]), int(h["S"]), int(h["D"]), int(h["T"]), int(h["Q"]), int(h["higher"]),
                         f"{lg['qubits_touched_any_gate']}/{lg['qubits_total']}", lg["distinct_2q_pairs"], lg["components"], lg["largest_component"],
                         ph["qubits_touched_any_gate"], ph["distinct_2q_pairs"], ph["components"], ph["two_qubit_gate_count"]])
        A(f'<details id="{tag}"><summary>{E(r["label"])} &mdash; {r["lineage"].upper()} {r["policy"]}</summary>')
        A(f'<p class="mut">{r["nq"]} qubits &middot; DMRG {r["ref"]:.9f} E<sub>h</sub> &middot; {run["shots"]:,} shots per circuit &middot; IBM Runtime job(s): '
          f'<code>{E(", ".join(run["ibm_runtime_job_ids"]))}</code> &middot; Slurm sampling job {E(str(run.get("slurm_job_id")))} &middot; generation limit {m["ngates"]} gates</p>')
        A(f'<img alt="excitation grid {tag}" src="data:image/png;base64,{img}">')
        A("<h4>Circuits</h4><p class=\"mut\">Raw excitation columns are shot counts out of 100,000. Live = the circuit as submitted (used in all summaries); offline = estimate without a live backend. Logical/physical connectivity as defined in Section 2.</p>")
        A(table(["#", "role", "depth (live)", "2q (live)", "depth (offline)", "2q (offline)", "valid shots", "valid distinct", "ranking energy (E<sub>h</sub>)",
                 "HF", "S", "D", "T", "Q", "Higher", "logical qubits touched", "logical 2q pairs", "logical components", "largest block",
                 "physical qubits touched", "physical 2q pairs", "physical components", "physical 2q gates"], crow))
        rrows = []
        for tg in TARGETS:
            e = r["t"][tg]
            src = e["alias"] or tg
            x = rec.get(tag, {}).get(src)
            circ = ", ".join(str(i + 1) for i in e["idx"]) if tg == "merged" else str(e["idx"] + 1)
            if not e["done"] and e.get("partial"):
                rrows.append([tg, circ, f"&le;{e['p_energy']:.6f}&dagger;", fmt_err(e), "&ndash;"] + ["&ndash;"] * 8 +
                             [f"partial: {e['p_iters']} of 25 iterations seen"])
                continue
            if not e["done"] or not x:
                rrows.append([tg if not e["alias"] else f"{tg} (= {e['alias']})", circ] + ['<span class="pend">not finished</span>'] + [""] * 11)
                continue
            ex, rw = x["excitation"], x["raw_shot_excitation"]
            hist = x["step_history"]
            fin = [h for h in hist if h == h and abs(h) != float("inf")]
            acc = hist.index(min(fin)) + 1 if fin else None
            sj = str(x["slurm_job_id"])
            rrows.append([tg if not e["alias"] else f"{tg} (= {e['alias']})", circ, f"{x['final_energy']:.6f}", fmt_err(e),
                          f"{e['si']:.2f}" if e.get("si") is not None else "&ndash;", int(ex["S"]), int(ex["D"]), int(ex["T"]), int(ex["Q"]), int(ex["higher"]),
                          f"{int(rw['combined']):,}", f"{int(rw['HF'])}/{int(rw['S'])}/{int(rw['D'])}/{int(rw['T'])}/{int(rw['Q'])}/{int(rw['higher'])}",
                          f"{acc}/{len(hist)}", hms(tm[sj]["elapsed"]) if sj in tm else "&ndash;"])
        A("<h4>Recoveries</h4><p class=\"mut\">Subspace excitation = pair-counted composition of the recovered n&times;n determinant set. Raw excitation of the pooled shots is HF/S/D/T/Q/Higher. "
          "'Best iteration' is the iteration whose energy was kept, out of 25.</p>")
        A(table(["result", "circuit(s)", "energy (E<sub>h</sub>)", "error (mHa)", "error after singles injection (mHa)", "subspace S", "subspace D", "subspace T", "subspace Q", "subspace Higher",
                 "shots pooled", "raw excitation HF/S/D/T/Q/Higher", "best iteration", "elapsed"], rrows))
        ops = []
        for i, cc in enumerate(m["circuits"]):
            ops.append([i + 1, E(" ".join(str(t) for t in cc["sequence"])), E(" ".join(cc["operator_names"]))])
        A("<h4>Gate sequences</h4>" + table(["#", "token sequence", "operators"], ops).replace("<td>", '<td class="l">'))
        A("</details>")



    import base64 as _b
    _cf = [(ROOT / "figures" / f"collage_{l}_trained.png") for l in ("a1-46q", "a1-70q")]
    if all(p.exists() for p in _cf):
        A('<h2 id="collage">Collated excitation grids (A1 policy)</h2><p>One panel per compound, smallest to largest active space; numbers above the columns are the circuit depth as submitted to <code>ibm_kobe</code>; '
          'best/2nd mark the two lowest-ranking-energy circuits. A compound without a finished run is shown as queued.</p>')
        for p, lab in zip(_cf, ("A1 policy trained at 46 qubits", "A1 policy trained at 70 qubits")):
            A(f'<h3>{lab}</h3><img alt="{lab}" src="data:image/png;base64,{_b.b64encode(p.read_bytes()).decode()}">')
    # ---- cost + analysis (same numbers as the manuscript)

    cal_path = ROOT / "kobe_calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        def crow(r, first):
            if "error" in r:
                return [first, "not available from IBM"] + [""] * 6
            g = lambda k, key="median": (r[k] or {}).get(key)
            pc = lambda k, p, key="median": f"{100 * g(k, key):.{p}f}" if g(k, key) is not None else "&ndash;"
            return [first, E(r["last_update_date"][:16]), f"{g('t1_us'):.0f}", f"{g('t2_us'):.0f}", pc("readout_error", 2), pc("sx_error", 3), pc("two_qubit_error", 2), pc("two_qubit_error", 1, "worst")]
        chead = ["Snapshot requested (UTC)", "IBM calibration time (JST)", "T1 median (&micro;s)", "T2 median (&micro;s)", "Readout error (%)", "1q <code>sx</code> error (%)", "2q <code>cz</code> error (%)", "worst 2q error (%)"]
        A('<h2 id="calibration">ibm_kobe calibration during the experiments</h2><p>Medians over the device&rsquo;s qubits (couplers) of the calibration IBM reported as in force at the stated time, fetched '
          f'{E(cal["fetched_utc"][:16].replace("T", " "))} UTC from the IBM Quantum backend properties. First table: one snapshot every 6 hours over the last 4 days. Second table: the calibration in force when each fresh-shot sampling job was created '
          '(a job&rsquo;s actual calibration is the one in force at its execution time, which can differ by up to one calibration interval). IBM recalibrates several times a day. A worst 2q error of 100% means IBM lists some couplers with error 1 (unusable; 10 of the 352 listed two-qubit gate entries, both directions, in the first snapshot); the median is unaffected.</p>')
        A(table(chead, [crow(r, E(r["requested_utc"][:16].replace("T", " "))) for r in reversed(cal["daily"])]))
        A(table(["Run, job created (UTC)"] + chead[1:], [crow(r, E(r["run"].replace("_", " ") + ", " + r.get("created_utc", "")[:16].replace("T", " "))) for r in sorted(cal["jobs"], key=lambda r: r.get("created_utc", ""))]))
    from cost_numbers import STAGE1, STAGE2, numbers as cost_numbers
    from analysis_numbers import numbers as analysis_numbers
    cn, an = cost_numbers(), analysis_numbers()
    nf = lambda x: f"{x:,}"
    qp = lambda g: (f"{cn['inference_shots'][g]['qpu_s']:,.0f}" if g in cn["inference_shots"] and cn["inference_shots"][g].get("qpu_s") else ("not retrieved" if g != "training" else "see table below"))
    A('<h2 id="cost">Cost: shots, QPU time and wall time</h2><p>Stage 1 of training is simulator-only. Stage 2 and all inference use <code>ibm_kobe</code>. Shot counts follow from the configurations; '
      'QPU time is IBM-billed seconds per job (filled in when retrieved with <code>fetch_qpu_usage.py</code>; otherwise "not retrieved"). Wall times are Slurm elapsed times.</p>')
    trn = cn["training"]
    A("<h3>Training</h3>")
    A(table(["Lineage", "Stage-1 iterations", "Stage-1 wall (h)", "Stage-2 iterations", "Stage-2 circuits", "Stage-2 shots", "Runtime jobs", "Stage-2 wall (h)"],
            [[k, v["s1_iters"], f"{v['s1_hours']:.2f}", v["s2_iters"], nf(v["circuits"]), nf(v["shots"]), v["jobs"], f"{v['s2_hours']:.2f}"] for k, v in trn.items()] +
            [["Total", sum(v["s1_iters"] for v in trn.values()), f"{cn['training_totals']['s1_hours']:.2f}", cn["training_totals"]["s2_iters"], nf(cn["training_totals"]["circuits"]),
              nf(cn["training_totals"]["shots"]), cn["training_totals"]["jobs"], f"{cn['training_totals']['s2_hours']:.2f}"]]))
    A(f"<p>Stage-2 shots = iterations &times; 20 circuits &times; 10,000 shots, one Runtime job per iteration. QPU time of the training jobs: {qp('training')}. Iterations lost to crashed Runtime jobs are not counted, so shot totals are lower bounds.</p>")
    A(table(["Lineage", "Stage", "Sample (s)", "Diagonalize (s)", "Refine (s)", "Transpile (s)", "Backward (s)", "Total (s)", "Sample share"],
            [[k, st_, *(f"{x:.1f}" for x in tab[k][2][:5]), f"{tab[k][2][5]:.1f}", f"{100 * tab[k][2][0] / tab[k][2][5]:.0f}%"] for st_, tab in (("1", STAGE1), ("2", STAGE2)) for k in tab]))
    msr = cn.get("measured")
    if msr:
        tt = msr["training"]
        A(f"<h3>Measured QPU time of hardware training (IBM job records, ibm_kobe, {msr['window'][0]} to {msr['window'][1]})</h3>")
        A(table(["Model size", "Iterations (jobs)", "Circuits", "Shots", "QPU time (min)", "mean s/iter.", "median s/iter.", "range s/iter.", "s/circuit"],
                [[lab, s["jobs"], nf(s["circuits"]), nf(s["shots"]), f"{s['qpu_s'] / 60:.0f}", f"{s['per_iter_mean_s']:.1f}", f"{s['per_iter_median_s']:.0f}",
                  f"{s['per_iter_min_s']:.0f}&ndash;{s['per_iter_max_s']:.0f}", f"{s['per_circuit_mean_s']:.2f}"] for lab, s in (("46-qubit models", tt["46q"]), ("70-qubit models", tt["70q"]), ("All", tt["total"]))]))
        A(f"<p>One job = one training iteration (20 circuits &times; 10,000 shots). The size is the number of qubits carrying a gate in the transpiled circuits (38&ndash;47 vs 56&ndash;74). The minimum billed time of a job is about 55 s; {tt['errored_jobs']} further jobs ended in error ({tt['errored_qpu_s']:.0f} s). "
          f"The {nf(tt['total']['jobs'])} completed jobs exceed the 512 iterations of the paper's four lineages; IBM's records do not say which model a job belonged to, so the difference is not attributed. "
          f"{msr['training_log_ids']['not_retrievable']} training job ids in the logs of the earlier Kawasaki-era 46-qubit runs are not returned by IBM for this account; their QPU time is unknown and not included.</p>")
    sh = cn["inference_shots"]
    A("<h3>Inference</h3>")
    A(table(["Campaign", "Runs", "Circuits", "Shots", "Runtime jobs", "QPU time (s)"],
            [["Fresh-shot (this release)", sh["fresh"]["runs"], nf(sh["fresh"]["circuits"]), nf(sh["fresh"]["shots"]), sh["fresh"]["jobs"], qp("fresh")],
             ["First campaign (same design)", sh["first"]["runs"], nf(sh["first"]["circuits"]), nf(sh["first"]["shots"]), sh["first"]["jobs"], qp("first")],
             ["LUCJ baseline", sh["lucj"]["runs"], sh["lucj"]["circuits"], nf(sh["lucj"]["shots"]), sh["lucj"]["jobs"], qp("lucj")],
             ["Total", sh["total"]["runs"], nf(sh["total"]["circuits"]), nf(sh["total"]["shots"]), sh["total"]["jobs"], "&ndash;"]]))
    w = cn["wall"]
    recall = [x for v in w["recovery"].values() for x in v]
    def wr(name, secs, g):
        return [name, len(secs), f"{sum(secs) / 3600:.1f}", f"{sum(secs) / len(secs) / 60:.0f}", g, f"{sum(secs) * g / 3600:.1f}"]
    A(table(["Stage (finished jobs)", "Jobs", "Wall (h)", "Mean per job (min)", "GPUs", "GPU-hours"],
            [wr("Sampling and transpilation", w["sample"], 1), wr("Ranking diagonalization", w["rank"], 2)] + [wr("Recovery: " + g, v, 2) for g, v in w["recovery"].items()] + [wr("Recovery: all", recall, 2)]))
    A("<h3>What distinguishes a good recovery (observational)</h3>")
    A(f"<p>{an['n_recoveries']} distinct individual-circuit recoveries. Valid shots, depth and connectivity are correlated, and runs are few; these are not controlled experiments.</p>")
    A(table(["Valid shots of the recovered circuit", "Recoveries (trained, &ge;44 qubits)", "Median error (mHa)", "Share below 25 mHa"],
            [[f"{b['lo']}&ndash;{b['hi'] if b['hi'] < 10**8 else '&infin;'}", b["n"], f"{b['median']:.1f}" if b["median"] is not None else "&ndash;", f"{100 * b['share_below_25']:.0f}%" if b["share_below_25"] is not None else "&ndash;"] for b in an["valid_bands"]]))
    A(f"<p>Of {an['best_vs_second']['runs']} trained runs with both best and second recovered, the best (lowest ranking energy) recovered no higher than the second in {an['best_vs_second']['best_not_worse']}. "
      f"Circuits with every qubit gated and all qubits in one 2q-connected block: median {an['block']['one_block']['median']:.1f} mHa ({an['block']['one_block']['n']} recoveries); others: median {an['block']['other']['median']:.1f} mHa ({an['block']['other']['n']}). "
      "The connected circuits are also the deep ones, so this may reflect depth or valid shots rather than connectivity.</p>")

    A("</main></body></html>")
    dst = ROOT / "zenodo"
    dst.mkdir(exist_ok=True)
    (dst / "sqd-gqe_zenodo.html").write_text("\n".join(out))
    print("wrote", dst / "sqd-gqe_zenodo.html", f"{(dst / 'sqd-gqe_zenodo.html').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
