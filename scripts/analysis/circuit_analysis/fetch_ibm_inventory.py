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

"""Account-level inventory of every IBM Runtime job (all backends) with billed QPU time, classified by what it ran.

Runs where the IBM credentials live (the cluster job fetch_inventory_on_cluster.sbatch sources .env.local; nothing secret is printed or written).
For each job: id, backend, creation time, status, billed quantum seconds, number of circuits (pubs) and shots per circuit.
Classification (by the real job contents, not by log files, so overwritten logs do not matter):
  training   : 20 circuits x 10,000 shots        (Stage-2 hardware training, one job per iteration)
  inference  : 10 circuits x 100,000 shots       (zero-shot sampling of a run, first campaign and fresh-shot campaign)
  lucj       : 1 circuit  x 100,000 shots        (LUCJ baseline)
  other      : anything else (tests, probes, other projects of the account)
Output: circuit_analysis/ibm_inventory.json  (+ summary printed).
Usage: fetch_ibm_inventory.py [--since 2026-07-01] [--out PATH] [--no-inputs]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "circuit_analysis"


def pubs_and_shots(job):
    inp = job.inputs
    pubs = inp.get("pubs") if isinstance(inp, dict) else None
    if not pubs:
        return None, None, None
    shots = None
    for p in pubs:
        if isinstance(p, (tuple, list)) and len(p) >= 3 and isinstance(p[-1], int):
            shots = p[-1]
    if shots is None and isinstance(inp, dict):
        shots = (inp.get("run_options") or {}).get("shots")
    return len(pubs), shots, active_qubits(pubs)


def active_qubits(pubs):
    """Median over the job's circuits of the number of qubits that carry a gate (measurements ignored): ~46-50 for the 46-qubit models, ~70-76 for the 70-qubit models."""
    import statistics
    counts = []
    for p in pubs:
        c = p[0] if isinstance(p, (tuple, list)) else p
        try:
            counts.append(len({c.find_bit(q).index for i in c.data if i.operation.name not in ("measure", "barrier") for q in i.qubits}))
        except Exception:  # noqa: BLE001
            pass
    return int(statistics.median(counts)) if counts else None


def classify(n, shots):
    if (n, shots) == (20, 10000):
        return "training"
    if (n, shots) == (10, 100000):
        return "inference"
    if (n, shots) == (1, 100000):
        return "lucj"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-07-01")
    ap.add_argument("--until", default=None, help="only jobs created before this date (UTC)")
    ap.add_argument("--out", default=str(ROOT / "ibm_inventory.json"))
    ap.add_argument("--no-inputs", action="store_true")
    a = ap.parse_args()
    from ibm_service import get_service
    service = get_service()
    since = dt.datetime.fromisoformat(a.since).replace(tzinfo=dt.timezone.utc)

    jobs, skip = [], 0
    while True:
        kw = {"created_before": dt.datetime.fromisoformat(a.until).replace(tzinfo=dt.timezone.utc)} if a.until else {}
        page = service.jobs(limit=200, skip=skip, created_after=since, descending=False, **kw)
        if not page:
            break
        jobs += page
        skip += len(page)
        print("listed", len(jobs), file=sys.stderr, flush=True)
        if len(page) < 200:
            break

    recs = []
    for k, job in enumerate(jobs):
        r = {"job_id": job.job_id()}
        try:
            r["backend"] = job.backend().name if job.backend() else None
            r["created_utc"] = job.creation_date.astimezone(dt.timezone.utc).isoformat()
            r["status"] = str(job.status())
            r["quantum_seconds"] = job.usage()
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        if not a.no_inputs and r.get("quantum_seconds"):
            try:
                n, shots, act = pubs_and_shots(job)
                r.update(n_circuits=n, shots_per_circuit=shots, active_qubits=act, kind=classify(n, shots))
            except Exception as e:  # noqa: BLE001
                r["inputs_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        recs.append(r)
        if k % 50 == 0:
            print(f"processed {k + 1}/{len(jobs)}", file=sys.stderr, flush=True)
            Path(a.out).write_text(json.dumps({"since": a.since, "jobs": recs}, indent=0))
    Path(a.out).write_text(json.dumps({"since": a.since, "n_listed": len(jobs), "jobs": recs}, indent=0))

    tot = {}
    for r in recs:
        key = (r.get("kind") or "unclassified", r.get("backend"))
        t = tot.setdefault(key, {"jobs": 0, "circuits": 0, "shots": 0, "qpu_s": 0.0})
        t["jobs"] += 1
        t["circuits"] += r.get("n_circuits") or 0
        t["shots"] += (r.get("n_circuits") or 0) * (r.get("shots_per_circuit") or 0)
        t["qpu_s"] += r.get("quantum_seconds") or 0.0
    print(f"{len(recs)} jobs since {a.since}")
    for (kind, be), t in sorted(tot.items(), key=lambda kv: str(kv[0])):
        print(f"{kind:12s} {str(be):14s} jobs {t['jobs']:5d} circuits {t['circuits']:7d} shots {t['shots']:13,d} QPU {t['qpu_s']:10.0f} s")


if __name__ == "__main__":
    main()
