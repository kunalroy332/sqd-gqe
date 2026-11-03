# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Download ibm_kobe calibration snapshots for the days the experiments ran.

Run it yourself (it needs your IBM Quantum credentials: either the account saved with
QiskitRuntimeService.save_account, or the IBM_API_KEY / IBM_CRN environment variables). It prints no secret.

  python3 fetch_kobe_calibration.py [--days 4] [--step-hours 6] [--out circuit_analysis/kobe_calibration.json]
                                    [--extra-jobs ids.txt]

Two kinds of snapshots are saved:
  daily   : calibration in force every --step-hours over the last --days days (UTC);
  per-job : calibration in force at the creation time of every IBM Runtime job id found in
            circuit_analysis/data/*.json (the fresh-shot inference samples) and in --extra-jobs
            (a text file with one job id per line, e.g. training jobs).
For each snapshot: median/mean/worst of T1, T2 (us), readout error, 1q gate error (sx) and 2q gate error (cz),
number of calibrated qubits and couplers, plus IBM's own last_update_date. Per-qubit arrays are stored too.
Historical calibration is only available for a limited time on the IBM side; snapshots it cannot return are
recorded with their error text, not skipped silently.
"""
import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "circuit_analysis"


def summarize(props):
    q = range(len(props.qubits))
    t1 = [props.t1(i) * 1e6 for i in q if props.t1(i) is not None]
    t2 = [props.t2(i) * 1e6 for i in q if props.t2(i) is not None]
    ro = [props.readout_error(i) for i in q if props.readout_error(i) is not None]
    g1, g2 = [], []
    for g in props.gates:
        try:
            e = props.gate_error(g.gate, g.qubits)
        except Exception:  # noqa: BLE001
            continue
        if g.gate == "sx" and len(g.qubits) == 1:
            g1.append(e)
        elif g.gate in ("cz", "ecr", "cx") and len(g.qubits) == 2:
            g2.append(e)

    def s(v, worst=max):
        return {"median": st.median(v), "mean": st.fmean(v), "worst": worst(v), "n": len(v)} if v else None

    return {"last_update_date": str(props.last_update_date), "n_qubits": len(props.qubits),
            "t1_us": s(t1, min), "t2_us": s(t2, min), "readout_error": s(ro), "sx_error": s(g1), "two_qubit_error": s(g2),
            "arrays": {"t1_us": t1, "t2_us": t2, "readout_error": ro, "sx_error": g1, "two_qubit_error": g2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--step-hours", type=int, default=6)
    ap.add_argument("--diagnose", action="store_true", help="print the instances and recent jobs this account can see, then exit")
    ap.add_argument("--out", default=str(ROOT / "kobe_calibration.json"))
    ap.add_argument("--extra-jobs")
    a = ap.parse_args()

    from ibm_service import diagnose, get_service
    service = get_service()
    if a.diagnose:
        diagnose(service)
        return
    be = service.backend("ibm_kobe")
    now = dt.datetime.now(dt.timezone.utc)

    out = {"backend": "ibm_kobe", "fetched_utc": now.isoformat(), "daily": [], "jobs": []}
    for k in range(0, a.days * 24 // a.step_hours + 1):
        t = now - dt.timedelta(hours=k * a.step_hours)
        rec = {"requested_utc": t.isoformat()}
        try:
            rec.update(summarize(be.properties(datetime=t)))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        out["daily"].append(rec)
        print("daily", t.isoformat(timespec="minutes"), "error" in rec and "ERR" or "ok", file=sys.stderr, flush=True)

    ids = {}
    for f in sorted((ROOT / "data").glob("*.json")):
        for jid in json.loads(f.read_text()).get("ibm_runtime_job_ids", []):
            ids[jid] = f.stem
    if a.extra_jobs:
        for line in Path(a.extra_jobs).read_text().split():
            ids.setdefault(line, "extra")
    for jid, tag in ids.items():
        rec = {"job_id": jid, "run": tag}
        try:
            job = service.job(jid)
            created = job.creation_date
            rec["created_utc"] = created.astimezone(dt.timezone.utc).isoformat()
            rec.update(summarize(be.properties(datetime=created)))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        out["jobs"].append(rec)
        print("job", jid, tag, "error" in rec and "ERR" or "ok", file=sys.stderr, flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1))
    ok = sum(1 for r in out["daily"] + out["jobs"] if "error" not in r)
    print(f"wrote {a.out}: {ok} of {len(out['daily']) + len(out['jobs'])} snapshots returned data")


if __name__ == "__main__":
    main()
