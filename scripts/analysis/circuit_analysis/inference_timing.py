# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Inference timing and cost table of the two training configurations (Hartree-Fock penalty, tuned reward), rebuilt from the cluster's Slurm records and IBM's QPU records.

    python3 inference_timing.py --refresh [--tex PATH]     # re-pull sacct from roquo, rebuild the per-set Slurm exports, rewrite the table block in the paper
    python3 inference_timing.py [--tex PATH]               # rebuild the table from the files already on disk

The IBM QPU files (<tree>/qpu_fresh.json) need your key and are made by fetch_qpu_inference_sets.py (hf | tuned); Missing or partial inputs are set in red in the table; nothing is estimated. Rewrites '% BEGIN INFTIMING'..'% END INFTIMING'.
"""
import argparse, collections, glob, json, os, re, subprocess, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SETS = [("HF penalty", "data/hf", "timing_sacct_hf.psv", 44), ("tuned reward", "data/tuned", "timing_sacct.psv", 44)]
PAT = {"data/hf": r"^(ca|rr|fin)_hf_", "data/tuned": r"^(ca|rr|fin)_tuned_"}
RED = lambda s: "{\\color{red}" + s + "}"


def refresh():
    out = subprocess.run(["ssh", "roquo", "sacct -u <USER> -S 2026-10-05 -X -n -P -o JobID,JobName%60,State,Elapsed,ElapsedRaw,Start,End,NodeList"], capture_output=True, text=True, timeout=300).stdout
    rows = [l.split("|") for l in out.splitlines() if l.count("|") >= 7]
    for _, tree, fname, _ in SETS:
        ids = set()
        for f in glob.glob(f"{REPO}/{tree}/rerecovery/*__*.json") + glob.glob(f"{REPO}/{tree}/data/*.json"):
            try:
                ids.add(str(json.load(open(f)).get("slurm_job_id")))
            except Exception:
                pass
        keep = [r for r in rows if (r[0] in ids) or re.match(PAT[tree], r[1])]
        if False:
            old = {l.split("|")[0] for l in open(f"{REPO}/{tree}/{fname}")}
            keep = [r for r in keep if r[0] not in old]
            if keep:
                open(f"{REPO}/{tree}/{fname}", "a").write("\n".join("|".join(r) for r in keep) + "\n")
        else:
            open(f"{REPO}/{tree}/{fname}", "w").write("\n".join("|".join(r) for r in keep) + "\n")
        print(f"refreshed {tree}: {len(keep)} records")


def classify(name):
    if name.startswith("ca_"): return "sample"
    if name.startswith("rr_"): return "rank"
    if name.startswith("fin_"): return "select"
    if name.startswith("rc_"):
        return {"b": "rec_bs", "s": "rec_bs", "ob": "rec_ed", "os": "rec_ed", "m": "rec_m"}.get(name.split("_")[-1])
    return None


def stats(tree, fname):
    ids = set()
    for f in glob.glob(f"{REPO}/{tree}/rerecovery/*__*.json") + glob.glob(f"{REPO}/{tree}/data/*.json"):
        try: ids.add(str(json.load(open(f)).get("slurm_job_id")))
        except Exception: pass
    g = collections.defaultdict(list); bad = []; running = []
    for l in open(f"{REPO}/{tree}/{fname}"):
        r = l.rstrip("\n").split("|")
        if len(r) < 8: continue
        jid, name, state, el = r[0], r[1], r[2], int(r[4])
        if tree == "circuit_analysis" and not (jid in ids or re.match(PAT[tree], name)): continue
        k = classify(name)
        if not k or "untrained" in name or re.search(r"_u_(b|s|m|ob|os)$", name): continue   # trained runs only   # trained runs only: the untrained controls were run once, for all sets
        if state == "COMPLETED": g[k].append(el)
        elif state in ("RUNNING", "PENDING", "REQUEUED"): running.append(name)
        else: bad.append((name, state, el))
    return g, bad, running


def qpu(tree, n_runs):
    p = f"{REPO}/{tree}/qpu_fresh.json"
    if tree == "circuit_analysis" and not os.path.exists(p):   # no complete record yet: the first campaign's IBM records cover the 24 original jobs only
        j = json.load(open(f"{REPO}/{tree}/qpu_usage.json"))["fresh"]["per_job"]
        ids = [i for f in glob.glob(f"{REPO}/{tree}/data/*_trained.json") for i in json.load(open(f)).get("ibm_runtime_job_ids", [])]
        q = [j[i]["quantum_seconds"] for i in ids if i in j and j[i]["quantum_seconds"] is not None]
        return sum(q), len(q), len(ids)
    if not os.path.exists(p): return None
    j = json.load(open(p)); q = [v["quantum_seconds"] for k, v in j["per_run"].items() if not k.endswith("untrained") and v["quantum_seconds"] is not None]
    return sum(q), len(q), n_runs


def block():
    L = ["% BEGIN INFTIMING", "\\begin{table*}[!t]", "\\centering",
         "\\caption{Inference timing and cost of the two training configurations (one zero-shot run = ten circuits sampled once on \\code{ibm\\_kobe} at 100{,}000 shots, ranked, then up to five recoveries). Wall times are Slurm elapsed times of the finished jobs on the roquo cluster; ``jobs'' counts finished jobs, failed, cancelled and timed-out jobs are listed separately and not counted. Recovery jobs use two GPUs each. QPU time is the time IBM billed for the sampling jobs (job records); the untrained controls are not included. Recoveries still running when the table was made are listed in the row ``Still running'' and are not counted.}",
         "\\label{tab:inference-timing}", "\\small", "\\renewcommand{\\arraystretch}{1.15}", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule", "Set & Stage & jobs & wall (h) & mean per job (min) & GPU-hours \\\\", "\\midrule"]
    nm = {"sample": ("Sampling and transpilation (IBM round trip)", 1), "rank": ("Ranking diagonalisation", 2), "rec_bs": ("Recovery: best / second", 2), "rec_ed": ("Recovery: E+D 1 / 2", 2), "rec_m": ("Recovery: merged", 2)}
    for lab, tree, fname, nplan in SETS:
        g, bad, running = stats(tree, fname)
        q = qpu(tree, nplan)
        nsamp = len(g["sample"])
        # a set is complete when every planned run was sampled, nothing is still queued or running, and the recoveries equal the number of
        # distinct recovery results the run files expect (best, second, merged and one result per distinct E+D pick); Training 1 is closed, so it counts as complete
        nrec = len(g["rec_bs"]) + len(g["rec_ed"]) + len(g["rec_m"])
        done = nsamp >= nplan and len(running) == 0
        first = True
        for k, (sl, gpus) in nm.items():
            v = g[k]
            if not v: continue
            cells = [lab if first else "", sl, str(len(v)), f"{sum(v) / 3600:.1f}", f"{np.mean(v) / 60:.0f}", f"{gpus * sum(v) / 3600:.1f}"]
            if not done and k.startswith("rec"): cells = [cells[0], cells[1]] + [RED(c) for c in cells[2:]]
            L.append(" & ".join(cells) + " \\\\"); first = False
        rec = g["rec_bs"] + g["rec_ed"] + g["rec_m"]
        if rec:
            c = ["", "All recoveries", str(len(rec)), f"{sum(rec) / 3600:.1f}", f"{np.mean(rec) / 60:.0f}", f"{2 * sum(rec) / 3600:.1f}"]
            L.append(" & ".join(c if done else [c[0], c[1]] + [RED(x) for x in c[2:]]) + " \\\\")
        if q and q[0] is not None:
            qc = (f"{q[1]} of {q[2]} jobs retrieved: " if q[1] != q[2] else f"{q[1]}/{q[2]} jobs: ") + f"{q[0]:,.0f}".replace(",", "{,}") + f" s ({q[0] / 60:.0f} min), {q[0] / max(q[1], 1):.0f} s per job"
            L.append(f" & QPU time billed by IBM (sampling) & \\multicolumn{{4}}{{l}}{{{qc}}} \\\\")
        else:
            L.append(f" & QPU time billed by IBM (sampling) & \\multicolumn{{4}}{{l}}{{{RED('not retrieved')}}} \\\\")
        if running: L.append(f" & Still running (not counted) & {len(running)} & & & \\\\")
        nb = len(bad)
        if nb: L.append(f" & Not counted: failed, cancelled or timed out & {nb} & {sum(b[2] for b in bad) / 3600:.2f} & & \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table*}", "% END INFTIMING"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--refresh", action="store_true"); ap.add_argument("--tex", default=f"{REPO}/docs/report/manuscript/paper.tex")
    a = ap.parse_args()
    if a.refresh: refresh()
    b = block(); s = open(a.tex).read()
    if "% BEGIN INFTIMING" in s:
        i = s.index("% BEGIN INFTIMING"); j = s.index("% END INFTIMING") + len("% END INFTIMING"); s = s[:i] + b + s[j:]; open(a.tex, "w").write(s); print("table rewritten")
    else:
        print("marker missing; table block follows\n" + b)
