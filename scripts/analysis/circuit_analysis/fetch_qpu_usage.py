# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Collect IBM's own QPU-time accounting for the hardware jobs of this project.

Run it yourself (needs your IBM Quantum credentials, saved account or IBM_API_KEY / IBM_CRN; nothing secret is printed):

  python3 fetch_qpu_usage.py [--ids-file ids.json] [--out circuit_analysis/qpu_usage.json]

Groups of job ids:
  fresh : the fresh-shot inference sampling jobs, read from circuit_analysis/data/*.json (automatic);
  others: --ids-file, a JSON object {"first": [...], "lucj": [...], "training": [...]} with the Runtime job ids of the
          first inference campaign, the LUCJ baseline and the Stage-2 hardware training. These ids are not stored in the repository
          (they are in the cluster logs / output files of those runs), so they must be supplied.
For each job: quantum_seconds (billed QPU time) and wall seconds from the job metrics, plus the job's shots*circuits when available.
Jobs IBM cannot return are listed under "missing", not guessed.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "circuit_analysis"


def usage(service, jid):
    job = service.job(jid)
    m = job.metrics() or {}
    u = (m.get("usage") or {})
    q = u.get("quantum_seconds")
    if q is None:
        try:
            q = job.usage()
        except Exception:  # noqa: BLE001
            q = None
    return {"quantum_seconds": q, "seconds": u.get("seconds"), "created": str(m.get("timestamps", {}).get("created")),
            "finished": str(m.get("timestamps", {}).get("finished"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file")
    ap.add_argument("--diagnose", action="store_true", help="print the instances and recent jobs this account can see, then exit")
    ap.add_argument("--out", default=str(ROOT / "qpu_usage.json"))
    a = ap.parse_args()
    from ibm_service import diagnose, get_service
    service = get_service()
    if a.diagnose:
        diagnose(service)
        return
    groups = {"fresh": sorted({j for f in (ROOT / "data").glob("*.json") for j in json.loads(f.read_text()).get("ibm_runtime_job_ids", [])})}
    if a.ids_file:
        groups.update(json.loads(Path(a.ids_file).read_text()))
    out = {}
    for g, ids in groups.items():
        per, missing = {}, []
        for jid in ids:
            try:
                per[jid] = usage(service, jid)
            except Exception as e:  # noqa: BLE001
                missing.append({"job_id": jid, "error": f"{type(e).__name__}: {e}"})
        qs = [v["quantum_seconds"] for v in per.values() if v["quantum_seconds"] is not None]
        out[g] = {"jobs": len(ids), "found": len(qs), "quantum_seconds": sum(qs) if qs else None, "per_job": per, "missing": missing}
        print(g, f"{len(qs)}/{len(ids)} jobs, QPU seconds: {sum(qs) if qs else 'n/a'}")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
