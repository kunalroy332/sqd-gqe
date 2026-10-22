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

"""Write the analysis subsection, the computational-cost section and the appendix key-number tables into paper.tex
(blocks between '% BEGIN ...' / '% END ...' markers; inserted before \\section{Discussion} on the first run). Numbers come from
cost_numbers.py, analysis_numbers.py and the stored result files; nothing is typed by hand.
"""
import re
import sys
from pathlib import Path

from analysis_numbers import numbers as analysis
from cost_numbers import STAGE1, STAGE2, numbers as cost, timing
from tables_data import LABEL, LIN_ORDER, ROOT, coverage_matrix

TEX = Path(__file__).resolve().parents[3] / "docs/report/manuscript/paper.tex"
H = lambda s: f"{s / 3600:.1f}"
N = lambda x: f"{x:,}".replace(",", "{,}")


def qpu_cell(c, grp):
    q = c["qpu"].get(grp)
    if not q or q.get("quantum_seconds") is None:
        return "--"
    return f"{q['quantum_seconds']:,.0f}".replace(",", "{,}") + ("" if q["found"] == q["jobs"] else f"$^{{a}}$")


def build():
    c, a = cost(), analysis()
    ms = c.get('measured')
    tm = timing()
    bad = {"FAILED", "TIMEOUT", "CANCELLED", "CANCELLED+", "NODE_FAIL", "OUT_OF_MEMORY"}
    waste = [v["elapsed"] for v in tm.values() if v["name"].startswith(("rc_", "rr_", "ca_")) and v["state"] in bad]
    s1, s2 = c["training"], c["training_totals"]
    blk = a["block"]
    vb = a["valid_bands"]
    bs = a["best_vs_second"]

    analysis_tex = f"""% BEGIN ANALYSIS
\\subsection{{What distinguishes a good recovery: valid shots, selection and connectivity}}
\\label{{sec:analysis-selection}}
The stored per-circuit data (released with the paper, see Appendix~\\ref{{sec:appendix-data}}) allow a few observational checks. They are not controlled
experiments: valid-shot count, circuit depth and connectivity are correlated with one another, and the runs are few.
We use the {a['n_recoveries']} distinct individual-circuit recoveries (best, second and the two E+D picks, duplicates removed).

\\paragraph{{Valid shots.}} For trained policies on the targets of 44 qubits and above, the recovered error falls steeply with the number of valid shots
of the recovered circuit: median {vb[0]['median']:.1f}~mHa for circuits with fewer than 100 valid shots ({vb[0]['n']} recoveries), {vb[1]['median']:.1f}~mHa for 100--1000
({vb[1]['n']}), {vb[2]['median']:.1f}~mHa for 1000--10{{,}}000 ({vb[2]['n']}; all below 25~mHa) and {vb[3]['median']:.1f}~mHa above 10{{,}}000 ({vb[3]['n']}). Within a single molecule the
Spearman correlation between $\\log_{{10}}$ valid shots and error is mostly negative (median over {len(a['spearman'])} molecules {a['spearman_median']:.2f};
range {min(s['rho'] for s in a['spearman']):.2f} to {max(s['rho'] for s in a['spearman']):.2f}); the positive values occur at the smallest active spaces, where every
recovery is within a fraction of a mHa and the ordering is noise.

\\paragraph{{Ranking energy does not order recoveries.}} Of the {bs['runs']} trained runs in which both the best and the second circuit were recovered, the best circuit
(lowest ranking energy) recovered to an energy no higher than the second in {bs['best_not_worse']}, i.e.\\ about half. The ranking energy is a useful screen (it discards
circuits with too few valid shots) but its order among the leading circuits is not informative about the recovered order.

\\paragraph{{Connectivity.}} Splitting the same recoveries by the logical two-qubit connectivity of the circuit (Appendix~\\ref{{sec:appendix-data}}), circuits in which every
qubit carries a gate and all qubits lie in one two-qubit-connected block reach a median {blk['one_block']['median']:.1f}~mHa ({blk['one_block']['n']} recoveries,
{blk['one_block']['below_25']} below 25~mHa), whereas the others (circuits with idle qubits or with the register split into separate blocks, typically the local alpha/beta
operators) reach a median {blk['other']['median']:.1f}~mHa ({blk['other']['n']} recoveries, {blk['other']['below_25']} below 25~mHa). The fully connected circuits are also the deep ones, so this
difference may reflect depth or valid-shot count rather than connectivity as such; we did not run the ablation that would separate them.
% END ANALYSIS
"""

    rows1 = "\n".join(
        f"{k} & {s1[k]['s1_iters']} & {s1[k]['s1_hours']:.2f} & {s1[k]['s2_iters']} & {N(s1[k]['circuits'])} & {N(s1[k]['shots'])} & {s1[k]['jobs']} & {s1[k]['s2_hours']:.2f} \\\\" for k in s1)
    rows1 += (f"\n\\midrule\nTotal & {sum(v['s1_iters'] for v in s1.values())} & {s2['s1_hours']:.2f} & {s2['s2_iters']} & {N(s2['circuits'])} & {N(s2['shots'])} & {s2['jobs']} & {s2['s2_hours']:.2f} \\\\")
    qt = qpu_cell(c, "training")
    qf = qpu_cell(c, "fresh")
    qfi = qpu_cell(c, "first")
    ql = qpu_cell(c, "lucj")

    def br(stage, tab):
        return "\n".join(f"{k} & {stage} & " + " & ".join(f"{x:.1f}" for x in tab[k][2][:5]) + f" & {tab[k][2][5]:.1f} & {100 * tab[k][2][0] / tab[k][2][5]:.0f}\\% \\\\" for k in tab)

    w = c["wall"]
    sh = c["inference_shots"]

    def wr(name, secs, gpus):
        return f"{name} & {len(secs)} & {H(sum(secs))} & {sum(secs) / len(secs) / 60:.0f} & {gpus} & {H(sum(secs) * gpus)} \\\\"

    rec_all = [x for v in w["recovery"].values() for x in v]
    infer_rows = "\n".join([wr("Sampling and transpilation (IBM round trip, 1 GPU job)", w["sample"], 1), wr("Ranking diagonalization (10 circuits)", w["rank"], 2)] +
                           [wr(f"Recovery: {g}", v, 2) for g, v in w["recovery"].items()] +
                           [wr("Recovery: all finished", rec_all, 2)])
    tot_h = sum(w["sample"]) + sum(w["rank"]) + sum(rec_all)
    gpu_h = sum(w["sample"]) + 2 * sum(w["rank"]) + 2 * sum(rec_all)
    qf_ = c['qpu'].get('fresh')
    if qf_ and qf_.get('found'):
        vs = [v['quantum_seconds'] for v in qf_['per_job'].values()]
        qpu_fresh_sentence = (f" IBM billed {N(round(qf_['quantum_seconds']))}~s ({qf_['quantum_seconds'] / 3600:.1f}~h) of QPU time for the {qf_['found']} fresh-shot jobs, "
                              f"{N(round(min(vs)))}--{N(round(max(vs)))}~s per job (10 circuits, {N(1000000)} shots each).")
    else:
        qpu_fresh_sentence = ""
    cal_sentence = (" The \\code{ibm\\_kobe} calibration in force during the experiments (T$_1$, T$_2$, readout, single- and two-qubit gate errors) is given in Section~\\ref{sec:kobe-calibration}, Tables~\\ref{tab:kobe-calibration-daily} and~\\ref{tab:kobe-calibration-jobs}."
                    if (ROOT / "kobe_calibration.json").exists() else "")
    def qc(g):
        v = sh[g].get('qpu_s')
        return f"{N(round(v))}" if v else "--"
    sh_ = c['inference_shots']
    qtot = f"{N(round(sh_['total']['qpu_s']))}" if sh_['total'].get('qpu_s') else "--"
    inf_qpu_total = f" and {N(round(sh_['total']['qpu_s']))}~s ({sh_['total']['qpu_s'] / 60:.0f}~min) of billed QPU time" if sh_['total'].get('qpu_s') else ""
    train_measured_text, qpu_table, grand_total_sentence = "", "", ""
    if ms:
        t = ms['training']; tl = ms['training_log_ids']
        j46, j70, jt = t['46q'], t['70q'], t['total']
        train_measured_text = (f" IBM's own job records for the account show {jt['jobs']} completed hardware-training jobs on \\code{{ibm\\_kobe}} on {ms['window'][0]} and {ms['window'][1]} "
            f"({N(jt['circuits'])} circuits, {N(jt['shots'])} shots), with {jt['qpu_s'] / 60:.0f}~QPU minutes ({jt['qpu_h']:.1f}~h) billed in total (Table~\\ref{{tab:cost-summary}}): "
            f"{j46['jobs']} iterations of the 46-qubit models took {j46['qpu_s'] / 60:.0f}~min ({j46['per_iter_mean_s']:.0f}~s per iteration on average) and {j70['jobs']} iterations of the 70-qubit models took "
            f"{j70['qpu_s'] / 60:.0f}~min ({j70['per_iter_mean_s']:.0f}~s per iteration). The 70-qubit iterations cost {j70['per_iter_mean_s'] / j46['per_iter_mean_s']:.1f} times as much QPU time as the 46-qubit ones. "
            f"The number of completed jobs exceeds the {s2['s2_iters']} iterations of Table~\\ref{{tab:stage2-summary}} by {jt['jobs'] - s2['s2_iters']} and the measured shots ({N(jt['shots'])}) exceed the computed ones ({N(s2['shots'])}); "
            f"IBM's records do not say which model or run a job belonged to, so we cannot attribute the difference (iterations repeated after crashed and resumed runs, and runs of other model arms, are possible causes). "
            f"The logs of the earlier 46-qubit runs on the retired \\code{{ibm\\_kawasaki}} allocation list {tl['not_retrievable']} further training jobs that IBM does not return for this account; "
            f"their QPU time ({N(20 * tl['not_retrievable'])} circuits, {N(200000 * tl['not_retrievable'])} shots) is unknown and is not included.")
        def row(lab, s):
            return (f"{lab} & {s['jobs']} & {N(s['circuits'])} & {N(s['shots'])} & {s['qpu_s'] / 60:.0f} & {s['per_iter_mean_s']:.1f} & {s['per_iter_median_s']:.0f} & "
                    f"{s['per_iter_min_s']:.0f}--{s['per_iter_max_s']:.0f} & {s['per_circuit_mean_s']:.2f} \\\\")
        qpu_table = (
"\\begin{table}[t]\n\\centering\n\\caption{IBM-billed QPU time of the hardware training on \\code{ibm\\_kobe} (completed jobs, one job = one training iteration of 20 circuits at 10{,}000 shots), "
"by model size; the size is the number of qubits that carry a gate in the transpiled circuits (38--47 for the 46-qubit models, 56--74 for the 70-qubit models). "
"The minimum billed time of a job is about 55~s. "
f"{ms['training']['errored_jobs']} further jobs ended in error ({ms['training']['errored_qpu_s']:.0f}~s).}}\n\\label{{tab:cost-summary}}\n\\scriptsize\n\\resizebox{{\\linewidth}}{{!}}{{%\n"
"\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\nModel size & Iterations & Circuits & Shots & QPU time (min) & mean s/iter. & median s/iter. & range s/iter. & s/circuit \\\\\n\\midrule\n"
+ row("46-qubit models", j46) + "\n" + row("70-qubit models", j70) + "\n\\midrule\n" + row("All", jt) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
        all_s = jt['qpu_s'] + (sh_['total'].get('qpu_s') or 0)
        grand_total_sentence = (f" Together with the {N(round(jt['qpu_s']))}~s ({jt['qpu_s'] / 60:.0f}~min) of the hardware training, the project used about {all_s / 60:.0f}~QPU minutes ({all_s / 3600:.1f}~h) on \\code{{ibm\\_kobe}}, "
                                f"not counting the Kawasaki-era training jobs that IBM no longer returns.")
    # ---- main-text table: every quantum resource in one place (training from IBM's job records, inference campaigns)
    def qm(sec, partial=False):
        return f"{sec / 60:.0f}" + ("$^{a}$" if partial else "")
    if ms:
        fresh_found = (c['qpu'].get('fresh') or {}).get('found')
        fresh_partial = bool(fresh_found) and fresh_found < sh['fresh']['jobs']
        trows = []
        for lab, g in (("Training Stage 2, 46-qubit models", j46), ("Training Stage 2, 70-qubit models", j70)):
            trows.append(f"{lab} & {N(g['circuits'])} & {N(g['shots'])} & {g['jobs']} & {qm(g['qpu_s'])} & {g['qpu_s'] / g['jobs']:.0f} \\\\")
        trows.append(f"\\midrule\nTraining total & {N(jt['circuits'])} & {N(jt['shots'])} & {jt['jobs']} & {qm(jt['qpu_s'])} & {jt['qpu_s'] / jt['jobs']:.0f} \\\\\n\\midrule")
        for lab, g in (("Inference, fresh-shot runs (this paper)", 'fresh'), ("Inference, first campaign", 'first'), ("Inference, LUCJ baseline", 'lucj')):
            q = sh[g].get('qpu_s')
            pa = (g == 'fresh' and fresh_partial)
            trows.append(f"{lab} & {N(sh[g]['circuits'])} & {N(sh[g]['shots'])} & {sh[g]['jobs']} & {qm(q, pa) if q else '--'} & {q / (fresh_found if pa else sh[g]['jobs']):.0f} \\\\")
        trows.append(f"\\midrule\nInference total & {N(sh['total']['circuits'])} & {N(sh['total']['shots'])} & {sh['total']['jobs']} & {qm(sh['total']['qpu_s'], fresh_partial) if sh['total'].get('qpu_s') else '--'} & \\\\\n\\midrule")
        gs = jt['shots'] + sh['total']['shots']; gj = jt['jobs'] + sh['total']['jobs']; gq = jt['qpu_s'] + (sh['total'].get('qpu_s') or 0)
        trows.append(f"Training and inference & {N(jt['circuits'] + sh['total']['circuits'])} & {N(gs)} & {gj} & {qm(gq, fresh_partial)} & \\\\")
        fn = (f" $^{{a}}$Billed time was retrieved for {fresh_found} of the {sh['fresh']['jobs']} fresh-shot jobs; the others are not yet in the job records used here." if fresh_partial else "")
        cost_summary_table = (
"\\begin{table}[t]\n\\centering\n\\caption{Quantum resources of the project on \\code{ibm\\_kobe}: circuits, shots, Qiskit Runtime jobs and IBM-billed QPU time (minutes) "
"from the account's own job records. A training job is one training iteration (20 circuits at 10{,}000 shots; the size of a model is the number of qubits carrying a gate, 38--47 for the 46-qubit and 56--74 for the 70-qubit models; "
f"{ms['training']['errored_jobs']} further jobs ended in error, {ms['training']['errored_qpu_s']:.0f}~s); an inference job is one run (10 circuits at 100{{,}}000 shots). Stage~1 of the training is a simulation and uses no QPU time. "
"The Kawasaki-era training jobs that IBM no longer returns are not included." + fn + "}\n\\label{tab:cost-summary}\n\\scriptsize\n\\resizebox{\\linewidth}{!}{%\n"
"\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nPhase & Circuits & Shots & Jobs & QPU time (min) & mean QPU s/job \\\\\n\\midrule\n" + "\n".join(trows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    else:
        cost_summary_table = ""
    cost_appendix_table = (
"\\begin{table}[ht]\n\\centering\n\\caption{Complementary cost figures. Top: computed Stage-2 training budget per lineage (iterations and wall time per lineage are in Tables~\\ref{tab:stage1-summary} and~\\ref{tab:stage2-summary}; the measured QPU time is in Table~\\ref{tab:cost-summary}); "
"shots = iterations $\\times$ 20 circuits $\\times$ 10{,}000 shots, one Runtime job per iteration. Bottom: classical wall time of the fresh-shot inference campaign by stage (finished Slurm jobs; failed, timed-out or cancelled jobs are not counted). "
f"Total {H(tot_h)}~h wall, {H(gpu_h)}~GPU-hours.}}\n\\label{{tab:cost-appendix}}\n\\small\n\\resizebox{{\\linewidth}}{{!}}{{%\n\\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}\n\\toprule\n"
"Lineage & Stage-2 iterations & Circuits & Shots & Runtime jobs & \\\\\n\\midrule\n"
+ "\n".join(f"{k} & {s1[k]['s2_iters']} & {N(s1[k]['circuits'])} & {N(s1[k]['shots'])} & {s1[k]['jobs']} & \\\\" for k in s1)
+ f"\n{'Total'} & {s2['s2_iters']} & {N(s2['circuits'])} & {N(s2['shots'])} & {s2['jobs']} & \\\\\n\\midrule\n"
"Inference stage & Jobs & Wall (h) & Mean per job (min) & GPUs & GPU-hours \\\\\n\\midrule\n" + infer_rows + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    cost_tex = f"""% BEGIN COST
\\section{{Computational cost: training and inference}}
\\label{{sec:cost}}
This section accounts for the quantum and classical resources behind the results. Wall-clock figures are Slurm elapsed times on the roquo cluster; shot counts
follow from the run configurations (circuits per iteration or per run times shots per circuit); QPU time is IBM's billed quantum time per job. The same numbers, with
the per-run breakdown, are in Appendix~\\ref{{sec:appendix-data}} and in the released HTML.{cal_sentence}

\\subsection{{Training}}
Stage 1 runs on an exact-statevector simulator and uses no QPU time. Stage 2 samples on \\code{{ibm\\_kobe}} at {N(c['training']['A1-46q']['shots'] // c['training']['A1-46q']['circuits'])} shots per circuit,
20 circuits per iteration batched in one Qiskit Runtime job, so each iteration is {N(20 * 10000)} shots and one job. Over the four lineages Stage 2 comprises {s2['s2_iters']} iterations,
{N(s2['circuits'])} circuits and {N(s2['shots'])} shots; Stage 1 took {s2['s1_hours']:.1f}~h and Stage 2 {s2['s2_hours']:.1f}~h of wall time in total.
The Stage-2 iteration counts are those of Table~\\ref{{tab:stage2-summary}} (A5-46q is the paused run); iterations lost to crashed Runtime jobs are not counted, so the shot totals are lower bounds on the shots submitted.
Table~\\ref{{tab:cost-summary}} gives the quantum resources and Table~\\ref{{tab:cost-appendix}} (Appendix~\\ref{{sec:appendix-cost}}) the computed budget per lineage.{train_measured_text}

{cost_summary_table}
The mean iteration time by pipeline step is in Figure~\\ref{{fig:stage-timing}} (Tables~\\ref{{tab:stage1-timing-table}} and~\\ref{{tab:stage2-timing-table}}, Appendix~\\ref{{sec:appendix-training-detail}}). In Stage 1 the circuit sampling (exact simulation) accounts for the largest share
({100 * STAGE1['A1-46q'][2][0] / STAGE1['A1-46q'][2][5]:.0f}--{100 * STAGE1['A5-70q'][2][0] / STAGE1['A5-70q'][2][5]:.0f}\\% of an iteration); in Stage 2 the sampling share is
{100 * min(v[2][0] / v[2][5] for v in STAGE2.values()):.0f}--{100 * max(v[2][0] / v[2][5] for v in STAGE2.values()):.0f}\\% and includes the IBM device queue, with the SQD diagonalization/recovery taking most of the remainder.
The refinement, transpilation and backward steps together are under 5\\% of an iteration in every lineage.

\\subsection{{Inference}}
Inference runs once on hardware per run (a batch of 10 circuits at 100{{,}}000 shots each, one Runtime job) and then entirely on classical hardware. The fresh-shot campaign
of this paper has {sh['fresh']['runs']} runs, {N(sh['fresh']['circuits'])} circuits and {N(sh['fresh']['shots'])} shots; the first campaign that preceded it used {sh['first']['jobs']} Runtime jobs ({N(sh['first']['shots'])} shots; more jobs than the {sh['first']['runs']} runs because some runs were sampled more than once), and the LUCJ baseline {sh['lucj']['jobs']} jobs
({N(sh['lucj']['shots'])} shots; one per molecule at each of the two transpilation levels). The grand total is {N(sh['total']['shots'])} shots in {sh['total']['jobs']} Runtime jobs{inf_qpu_total}.
Table~\\ref{{tab:cost-summary}} summarizes these counts, which for the first campaign and LUCJ are IBM's own job records.{qpu_fresh_sentence}{grand_total_sentence}

Table~\\ref{{tab:cost-appendix}} (bottom) gives the classical wall time of the fresh-shot campaign by stage. The {len(rec_all)} finished recoveries took {H(sum(rec_all))}~h of wall time on two GPUs each
({H(2 * sum(rec_all))}~GPU-hours), far more than the hardware round trips ({H(sum(w['sample']))}~h in total, mostly IBM queue and local transpilation) or the rankings
({H(sum(w['rank']))}~h). Merged recoveries are the most expensive per job because they pool up to ten times the shots; see the Zenodo release for the per-run times. Jobs that failed, timed out or were cancelled (bad nodes, the maintenance window) consumed a further
{H(sum(waste))}~h of wall time without producing a result and are not counted in the table.

% END COST
"""
    costapp_tex = ("% BEGIN COSTAPP\n\\section{Computational cost: complementary figures}\n\\label{sec:appendix-cost}\n"
                   "The main-text cost section (Section~\\ref{sec:cost}, Table~\\ref{tab:cost-summary}) gives the quantum resources. The computed training budget per lineage and the classical wall time of the "
                   "inference campaign are collected here.\n\n" + cost_appendix_table + "% END COSTAPP\n")
    cm = coverage_matrix()
    mrow = "\n".join(LABEL[mk][0] + " & " + " & ".join(cm[mk][(p, l)] for p in ("trained", "untrained") for l in LIN_ORDER) + " \\\\"
                      for mk in sorted(LABEL, key=lambda k: (LABEL[k][1], LABEL[k][2])))
    runmatrix_tex = f"""% BEGIN RUNMATRIX
\\begin{{table}}[ht]
\\centering
\\caption{{Which (target, policy) runs have fresh-shot results. ``done'': sampled and ranked, and recovered; ``not run'': not part of the design (untrained controls were run for a subset of targets). Columns are the lineage (A1/A5, trained at 46 or 70 qubits).}}
\\label{{tab:run-matrix}}
\\scriptsize
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}l*{{8}}{{c}}@{{}}}}
\\toprule
 & \\multicolumn{{4}}{{c}}{{Trained}} & \\multicolumn{{4}}{{c}}{{Untrained control}} \\\\
Target & A1-46q & A1-70q & A5-46q & A5-70q & A1-46q & A1-70q & A5-46q & A5-70q \\\\
\\midrule
{mrow}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
% END RUNMATRIX
"""
    key_tex = f"""% BEGIN KEYNUM
\\section{{Dataset-level numbers}}
\\label{{sec:appendix-data}}
The deposit accompanying this paper (Zenodo; a self-contained HTML description plus the raw counts, transpiled circuits, result files and code) contains, for each of the
{c['n_runs']} runs, the 10 sampled circuits with their counts (100{{,}}000 shots each), depth and two-qubit-gate counts (offline and as submitted), raw and recovered excitation
compositions, the five recovery energies, logical and physical two-qubit connectivity (qubits touched, distinct connected pairs, connected blocks), Slurm timing and the \\code{{ibm\\_kobe}}
calibration. The tables of this appendix are the numbers most often needed to reuse the data: Tables~\\ref{{tab:zeroshot-detail}}--\\ref{{tab:raw-shot-excitation}} (energies,
depth, valid shots, excitation), Table~\\ref{{tab:coverage-idle}} (coverage), the cost tables of Section~\\ref{{sec:cost}} and, once fetched, the \\code{{ibm\\_kobe}} calibration tables.
Circuits were generated and transpiled with Qiskit \\cite{{javadiabhari2024qiskit}}.

% END KEYNUM
"""
    return analysis_tex, cost_tex, key_tex, costapp_tex, runmatrix_tex


def main():
    an, co, ke, ca, rm = build()
    t = TEX.read_text()
    for name, block in (("ANALYSIS", an), ("COST", co), ("COSTAPP", ca), ("RUNMATRIX", rm), ("KEYNUM", ke)):
        pat = re.compile(r"% BEGIN " + name + r".*?% END " + name + r"\n", re.S)
        if pat.search(t):
            t = pat.sub(lambda m: block, t)
        elif name == "RUNMATRIX":
            raise SystemExit("RUNMATRIX markers missing in paper.tex")
        elif name == "COSTAPP":
            k = "% BEGIN KEYNUM"
            assert k in t
            t = t.replace(k, block + "\n" + k, 1)
        elif name in ("ANALYSIS", "COST"):
            k = "\\section{Discussion}"
            assert k in t
            t = t.replace(k, block + "\n" + k, 1)
        else:
            k = "\\section{Data and circuit availability}"
            assert k in t
            t = t.replace(k, block + "\n" + k, 1)
    TEX.write_text(t)
    print("cost/analysis blocks written")


if __name__ == "__main__":
    main()
