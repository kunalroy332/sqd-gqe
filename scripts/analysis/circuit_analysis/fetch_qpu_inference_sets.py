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

"""Billed QPU time of the inference sampling jobs of one model set (read-only: queries IBM job records, runs nothing on hardware).

Usage (needs IBM_API_KEY and IBM_CRN in your shell; nothing secret is printed or written):
    python3 fetch_qpu_inference_sets.py hf       # circuit_analysis_hf/data/*.json      -> circuit_analysis_hf/qpu_fresh.json
    python3 fetch_qpu_inference_sets.py tuned    # circuit_analysis_tuned/data/*.json   -> circuit_analysis_tuned/qpu_fresh.json
    python3 fetch_qpu_inference_sets.py t1       # circuit_analysis/data/*.json         -> circuit_analysis/qpu_fresh.json
    python3 fetch_qpu_inference_sets.py --diagnose   # list the instances / recent jobs the account can see
The Runtime job ids are read from the per-run files; the billed time is job.usage() (quantum_seconds), falling back to the job metrics
(usage.quantum_seconds / qpu_charge_time_seconds). Jobs IBM cannot return are listed under "missing", never guessed.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DIRS = {"hf": REPO / "circuit_analysis_hf", "tuned": REPO / "circuit_analysis_tuned", "t1": REPO / "circuit_analysis"}


def billed(service, jid):
    job = service.job(jid)
    q = None
    try:
        q = job.usage()
    except Exception:  # noqa: BLE001
        pass
    m = {}
    try:
        m = job.metrics() or {}
    except Exception:  # noqa: BLE001
        pass
    u = m.get("usage") or {}
    if q is None:
        q = u.get("quantum_seconds", u.get("qpu_charge_time_seconds"))
    return {"quantum_seconds": q, "seconds": u.get("seconds"), "created": str(m.get("timestamps", {}).get("created")),
            "finished": str(m.get("timestamps", {}).get("finished")), "backend": str(job.backend())}


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from ibm_service import diagnose, get_service
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    service = get_service()
    if arg == "--diagnose":
        diagnose(service)
        return
    root = DIRS[arg]
    per, missing = {}, []
    for f in sorted((root / "data").glob("*.json")):
        run = json.loads(f.read_text())
        for jid in run.get("ibm_runtime_job_ids", []):
            try:
                per[f.stem] = {"job_id": jid, **billed(service, jid)}
            except Exception as e:  # noqa: BLE001
                missing.append({"run": f.stem, "job_id": jid, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    qs = [v["quantum_seconds"] for v in per.values() if v["quantum_seconds"] is not None]
    out = {"set": arg, "jobs": len(per) + len(missing), "found": len(qs), "quantum_seconds": sum(qs) if qs else None, "per_run": per, "missing": missing}
    (root / "qpu_fresh.json").write_text(json.dumps(out, indent=1))
    print(f"{arg}: {len(qs)}/{out['jobs']} jobs, QPU seconds {sum(qs) if qs else 'n/a'}; wrote {root / 'qpu_fresh.json'}")


if __name__ == "__main__":
    main()
