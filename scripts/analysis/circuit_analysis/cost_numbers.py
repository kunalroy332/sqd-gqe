# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shots, QPU time and wall time of training and inference, computed from stored files (single source for paper and HTML).

Training figures come from the stage logs as tabulated in the manuscript (iterations, per-iteration stage times); the shot counts follow from the
Stage-2 configuration (20 circuits per iteration, 10,000 shots per circuit, one Qiskit Runtime job per iteration). QPU seconds are read from
circuit_analysis/qpu_usage.json when it exists (written by fetch_qpu_usage.py from IBM's own job metrics); otherwise they are reported as unavailable.
"""
import json
import statistics as st
from pathlib import Path

from tables_data import ROOT, load

# (iterations, total wall hours, per-iteration seconds: sample, diagonalize, refine, transpile, backward, total) -- manuscript Tables stage1/stage2
STAGE1 = {"A1-46q": (90, 12.05, (278.2, 188.2, 8.8, 4.6, 1.15, 481.9)), "A1-70q": (90, 14.18, (358.1, 193.0, 9.1, 4.5, 1.15, 567.2)),
          "A5-46q": (118, 13.62, (282.1, 121.3, 5.8, 4.7, 1.12, 415.7)), "A5-70q": (132, 24.20, (451.4, 191.3, 9.2, 6.1, 1.18, 660.1))}
STAGE2 = {"A1-46q": (124, 14.09, (242.1, 154.9, 8.3, 1.5, 0.61, 408.9)), "A1-70q": (150, 21.59, (213.0, 280.8, 15.4, 6.4, 1.25, 530.5)),
          "A5-46q": (88, 9.62, (141.5, 233.8, 12.4, 2.7, 0.55, 393.6)), "A5-70q": (150, 21.93, (211.7, 285.7, 15.3, 6.0, 1.25, 539.2))}
CIRC_PER_ITER, SHOTS_PER_CIRC_TRAIN = 20, 10_000
CIRC_PER_RUN, SHOTS_PER_CIRC_INFER = 10, 100_000
LUCJ_MOLECULES = 9


def qpu():
    p = ROOT / "qpu_usage.json"
    return json.loads(p.read_text()) if p.exists() else {}


def timing():
    t = {}
    for line in (ROOT / "timing_sacct.psv").read_text().splitlines():
        p = line.split("|")
        if len(p) >= 8:
            t[p[0]] = {"name": p[1], "state": p[2], "elapsed": int(p[4])}
    return t


def numbers():
    d = load()
    tm = timing()
    runs = list(d.values())
    out = {"n_runs": len(runs)}
    # training
    tr = {}
    for k in STAGE2:
        it, hrs, _ = STAGE2[k]
        tr[k] = {"s1_iters": STAGE1[k][0], "s1_hours": STAGE1[k][1], "s2_iters": it, "s2_hours": hrs,
                 "circuits": it * CIRC_PER_ITER, "shots": it * CIRC_PER_ITER * SHOTS_PER_CIRC_TRAIN, "jobs": it}
    out["training"] = tr
    out["training_totals"] = {"s1_hours": sum(v["s1_hours"] for v in tr.values()), "s2_hours": sum(v["s2_hours"] for v in tr.values()),
                              "s2_iters": sum(v["s2_iters"] for v in tr.values()), "circuits": sum(v["circuits"] for v in tr.values()),
                              "shots": sum(v["shots"] for v in tr.values()), "jobs": sum(v["jobs"] for v in tr.values())}
    # inference shots
    fresh = len(runs) * CIRC_PER_RUN * SHOTS_PER_CIRC_INFER
    out["inference_shots"] = {"fresh": {"runs": len(runs), "circuits": len(runs) * CIRC_PER_RUN, "shots": fresh, "jobs": len(runs)},
                              "first": {"runs": len(runs), "circuits": len(runs) * CIRC_PER_RUN, "shots": fresh, "jobs": len(runs)},
                              "lucj": {"runs": LUCJ_MOLECULES, "circuits": LUCJ_MOLECULES, "shots": LUCJ_MOLECULES * SHOTS_PER_CIRC_INFER, "jobs": LUCJ_MOLECULES}}
    qsum_p = ROOT / "qpu_training_summary.json"
    out["measured"] = json.loads(qsum_p.read_text()) if qsum_p.exists() else None
    if out["measured"]:   # IBM's own job records replace the counts inferred from the run design
        m = out["measured"]
        out["inference_shots"]["first"].update(circuits=m["first_campaign"]["circuits"], shots=m["first_campaign"]["shots"], jobs=m["first_campaign"]["jobs"], qpu_s=m["first_campaign"]["qpu_s"])
        out["inference_shots"]["lucj"].update(circuits=m["lucj"]["circuits"], shots=m["lucj"]["shots"], jobs=m["lucj"]["jobs"], qpu_s=m["lucj"]["qpu_s"])
        out["inference_shots"]["fresh"]["qpu_s"] = m["fresh"]["qpu_s"]
        qu = qpu().get("fresh")   # the per-job fetch of fresh-shot ids covers jobs the account inventory did not
        if qu and qu.get("quantum_seconds") and qu.get("found", 0) >= m["fresh"]["jobs"]:
            out["inference_shots"]["fresh"]["qpu_s"] = qu["quantum_seconds"]
    out["inference_shots"]["total"] = {k: sum(out["inference_shots"][g][k] for g in ("fresh", "first", "lucj")) for k in ("runs", "circuits", "shots", "jobs")}
    if out["measured"]:
        out["inference_shots"]["total"]["qpu_s"] = sum(out["inference_shots"][g]["qpu_s"] for g in ("fresh", "first", "lucj"))
    # inference wall (finished Slurm jobs of the fresh-shot campaign)
    ca, rr = [], []
    for r in runs:
        sj = json.loads((ROOT / "data" / f"{r['tag']}.json").read_text()).get("slurm_job_id")
        if str(sj) in tm and tm[str(sj)]["state"] == "COMPLETED":
            ca.append(tm[str(sj)]["elapsed"])
        j = next((v for v in tm.values() if v["name"] == f"rr_{r['tag']}" and v["state"] == "COMPLETED"), None)
        if j:
            rr.append(j["elapsed"])
    rc = {}
    seen = set()
    for f in (ROOT / "rerecovery").glob("*__*.json"):
        x = json.loads(f.read_text())
        sj = str(x["slurm_job_id"])
        if sj in tm and sj not in seen:
            seen.add(sj)
            tg = f.stem.split("__")[1]
            grp = {"best": "best / second", "second": "best / second", "merged": "merged", "oldbest": "E+D 1 / E+D 2", "oldsecond": "E+D 1 / E+D 2"}[tg]
            rc.setdefault(grp, []).append(tm[sj]["elapsed"])
    out["wall"] = {"sample": ca, "rank": rr, "recovery": rc}
    out["qpu"] = qpu()
    return out


if __name__ == "__main__":
    n = numbers()
    print(json.dumps({k: v for k, v in n.items() if k not in ("wall",)}, indent=1)[:1500])
    for k, v in n["wall"].items():
        if isinstance(v, dict):
            for g, l in v.items(): print(k, g, len(l), round(sum(l) / 3600, 1), "h")
        else:
            print(k, len(v), round(sum(v) / 3600, 1), "h")
