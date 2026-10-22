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

"""Measure the depth and two-qubit-gate count of the LUCJ circuits that actually ran on ibm_kobe, directly from the IBM jobs.

Run it yourself (needs the cluster account's IBM_API_KEY / IBM_CRN exported in your shell; nothing secret is printed):

  /tmp/ibmvenv/bin/python scripts/analysis/circuit_analysis/lucj_from_ibm.py [--since 2026-09-01] [--job-ids ids.txt]

With --job-ids (one id per line, e.g. taken from the cluster logs when roquo is back) it measures exactly those jobs. Without it, it lists the account's ibm_kobe
jobs since --since that contain a single circuit at 100,000 shots (the LUCJ benchmark design) and measures each; the matching of a job to a molecule is then by the
qubit count and creation order and must be confirmed by hand (two targets share 56 qubits).

Metrics are defined exactly as for the GQE circuits (verified on 10 of 10 stored GQE circuits): depth = circuit.depth(), two-qubit gates = circuit.num_nonlocal_gates(),
both on the transpiled circuit stored in the job, i.e. as submitted. Writes circuit_analysis/lucj_from_ibm.json.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "circuit_analysis"


def circuits_of(job):
    inp = job.inputs
    pubs = inp.get("pubs") if isinstance(inp, dict) else None
    out = []
    for p in pubs or []:
        out.append(p[0] if isinstance(p, (tuple, list)) else p)
    shots = None
    for p in pubs or []:
        if isinstance(p, (tuple, list)) and len(p) >= 3 and isinstance(p[-1], int):
            shots = p[-1]
    shots = shots or (inp.get("run_options") or {}).get("shots") if isinstance(inp, dict) else shots
    return out, shots


def measure(job):
    cs, shots = circuits_of(job)
    rec = {"job_id": job.job_id(), "created": str(job.creation_date), "n_circuits": len(cs), "shots": shots}
    if cs:
        c = cs[0]
        active = {c.find_bit(q).index for i in c.data if i.operation.name not in ("barrier",) for q in i.qubits}
        rec.update(depth=c.depth(), two_qubit_gates=c.num_nonlocal_gates(), active_qubits=len(active), ops=dict(c.count_ops()))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-01")
    ap.add_argument("--job-ids")
    ap.add_argument("--out", default=str(ROOT / "lucj_from_ibm.json"))
    a = ap.parse_args()
    from ibm_service import get_service
    service = get_service()
    recs = []
    if a.job_ids:
        for jid in Path(a.job_ids).read_text().split():
            recs.append(measure(service.job(jid)))
    else:
        since = dt.datetime.fromisoformat(a.since).replace(tzinfo=dt.timezone.utc)
        for job in service.jobs(limit=500, backend_name="ibm_kobe", created_after=since, descending=False):
            try:
                r = measure(job)
            except Exception as e:  # noqa: BLE001
                r = {"job_id": job.job_id(), "error": f"{type(e).__name__}: {str(e)[:100]}"}
            if r.get("n_circuits") == 1 and r.get("shots") == 100000:
                recs.append(r)
    Path(a.out).write_text(json.dumps(recs, indent=1))
    print(f"{len(recs)} candidate LUCJ-style jobs (1 circuit, 100,000 shots)")
    for r in recs:
        print(r.get("created", "")[:16], r["job_id"], "active", r.get("active_qubits"), "depth", r.get("depth"), "2q", r.get("two_qubit_gates"))


if __name__ == "__main__":
    main()
