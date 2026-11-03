# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""IBM-billed QPU time of hardware training (by model size), the first inference campaign and the LUCJ baseline.

Inputs (circuit_analysis/): inv_sep.json, inv_oct.json, ibm_inventory.json (account-level inventory from fetch_ibm_inventory.py: every job with
billed quantum seconds, circuits, shots, active qubits), qpu_usage_logs.json and ids_logs.json (jobs identified in the project's cluster logs),
qpu_usage.json (the 29 fresh-shot inference jobs). Output: circuit_analysis/qpu_training_summary.json.

Hardware-training jobs = 20 circuits x 10,000 shots, one Runtime job per iteration. Model size is taken from the number of qubits that carry a gate in the
transpiled circuits (46-qubit models: 38-47; 70-qubit models: 56-74; the gap at 48-55 separates them cleanly), so it does not depend on log files.
The 212 training job ids that appear in the logs of the earlier ibm_kawasaki runs are not returned by IBM for this account; their QPU time is unknown.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "circuit_analysis"
SPLIT = 50


def stats(js):
    q = [j["quantum_seconds"] for j in js]
    if not q:
        return None
    n = len(js)
    return {"jobs": n, "circuits": 20 * n, "shots": 20 * 10_000 * n, "qpu_s": sum(q), "qpu_h": sum(q) / 3600,
            "per_iter_mean_s": st.mean(q), "per_iter_median_s": st.median(q), "per_iter_min_s": min(q), "per_iter_max_s": max(q),
            "per_circuit_mean_s": sum(q) / (20 * n)}


def main():
    inv = {}
    for f in ("ibm_inventory", "inv_sep", "inv_oct"):
        p = ROOT / f"{f}.json"
        if p.exists():
            for j in json.loads(p.read_text())["jobs"]:
                inv[j["job_id"]] = j
    tr = [j for j in inv.values() if j.get("kind") == "training"]
    done = [j for j in tr if j.get("status") == "DONE"]
    err = [j for j in tr if j.get("status") != "DONE"]
    small = [j for j in done if (j.get("active_qubits") or 0) < SPLIT]
    big = [j for j in done if (j.get("active_qubits") or 0) >= SPLIT]
    dates = sorted(j["created_utc"][:10] for j in done)
    logs = json.loads((ROOT / "qpu_usage_logs.json").read_text())
    ids = json.loads((ROOT / "ids_logs.json").read_text())
    out = {
        "backend": "ibm_kobe", "window": [dates[0], dates[-1]], "split_active_qubits": SPLIT,
        "training": {"46q": stats(small), "70q": stats(big), "total": stats(done),
                     "errored_jobs": len(err), "errored_qpu_s": sum(j.get("quantum_seconds") or 0 for j in err)},
        "training_log_ids": {"in_logs": len(ids["training"]), "retrievable": logs["training"]["found"],
                              "retrievable_qpu_s": logs["training"]["quantum_seconds"], "not_retrievable": len(ids["training"]) - logs["training"]["found"]},
        "first_campaign": {"jobs": logs["first"]["found"], "circuits": 10 * logs["first"]["found"], "shots": 10 * 100_000 * logs["first"]["found"], "qpu_s": logs["first"]["quantum_seconds"]},
        "lucj": {"jobs": logs["lucj"]["found"], "circuits": logs["lucj"]["found"], "shots": 100_000 * logs["lucj"]["found"], "qpu_s": logs["lucj"]["quantum_seconds"]},
    }
    fresh = json.loads((ROOT / "qpu_usage.json").read_text())["fresh"]
    out["fresh"] = {"jobs": fresh["found"], "qpu_s": fresh["quantum_seconds"]}
    (ROOT / "qpu_training_summary.json").write_text(json.dumps(out, indent=1))
    t = out["training"]
    print("training (ibm_kobe, DONE jobs):")
    for k in ("46q", "70q", "total"):
        s = t[k]
        print(f"  {k:5s} jobs {s['jobs']:4d}  circuits {s['circuits']:6,d}  shots {s['shots']:12,d}  QPU {s['qpu_s']:8.0f} s = {s['qpu_h']:5.2f} h  per iteration mean {s['per_iter_mean_s']:.1f} s (median {s['per_iter_median_s']:.0f}, {s['per_iter_min_s']:.0f}-{s['per_iter_max_s']:.0f})  per circuit {s['per_circuit_mean_s']:.2f} s")
    print("  errored jobs:", t["errored_jobs"], "QPU", t["errored_qpu_s"], "s; window", out["window"])
    print("log ids:", out["training_log_ids"])
    print("first campaign:", out["first_campaign"]); print("lucj:", out["lucj"]); print("fresh:", out["fresh"])


if __name__ == "__main__":
    main()
