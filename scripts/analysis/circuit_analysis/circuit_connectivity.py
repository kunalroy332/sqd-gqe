# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit connectivity of every sampled circuit (for the data description).

For each run tag, loads the stored circuits and reports, per circuit:
  logical   : the generated circuit translated to {rz, sx, cx} WITHOUT a coupling map (optimization_level=0),
              i.e. which logical qubit pairs the policy's gates actually entangle;
  physical  : the transpiled circuit that ran on ibm_kobe (circuits.qpy): which physical couplers carry 2q gates.
Metrics: qubits touched by any gate, qubits touched by a 2q gate, distinct 2q pairs, density of the pair graph
(pairs / C(n,2)), connected components of the 2q-pair graph, size of the largest component, max/mean degree.

Usage: circuit_connectivity.py --tags t1 t2 ...   (writes circuit_analysis_connectivity/<tag>.json)
"""
import argparse
import glob
import json
import os
import sys

from qiskit import qpy, transpile

BASE = os.environ.get("INFER_ROOT", "/path/to/sqd-gqe-work/experiments/crossnorb70q_infer")   # each model set has its own tree


def graph_metrics(n_total, pairs, touched_any):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = {}
    for a, b in pairs:
        parent[find(a)] = find(b)
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    comps = {}
    for q in list(deg):
        comps.setdefault(find(q), set()).add(q)
    sizes = sorted((len(c) for c in comps.values()), reverse=True)
    q2 = len(deg)
    return {
        "qubits_total": n_total, "qubits_touched_any_gate": len(touched_any), "qubits_in_2q_gates": q2,
        "distinct_2q_pairs": len(pairs),
        "pair_density": (len(pairs) / (n_total * (n_total - 1) / 2)) if n_total > 1 else 0.0,
        "components": len(sizes), "largest_component": sizes[0] if sizes else 0,
        "largest_component_fraction": (sizes[0] / n_total) if sizes else 0.0,
        "max_degree": max(deg.values()) if deg else 0, "mean_degree": (sum(deg.values()) / q2) if q2 else 0.0,
    }


def analyse(qc, n_total):
    pairs, touched = set(), set()
    nq = 0
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        touched.update(qs)
        if len(qs) == 2:
            pairs.add(tuple(sorted(qs)))
            nq += 1
    m = graph_metrics(n_total, pairs, touched)
    m["two_qubit_gate_count"] = nq
    return m


def one(tag):
    lg = qpy.load(open(f"{BASE}/circuit_analysis_archive/{tag}/logical_circuits.qpy", "rb"))
    tp = qpy.load(open(glob.glob(f"{BASE}/circuit_analysis_ibm_jobs/{tag}/ibm_kobe_shots100000/*/circuits.qpy")[0], "rb"))
    n = lg[0].num_qubits
    out = []
    for i, (l, t) in enumerate(zip(lg, tp)):
        lt = transpile(l, basis_gates=["rz", "sx", "cx"], optimization_level=0, seed_transpiler=0)
        out.append({"index": i, "logical": analyse(lt, n), "physical": analyse(t, t.num_qubits),
                    "physical_total_qubits": t.num_qubits})
    os.makedirs(f"{BASE}/circuit_analysis_connectivity", exist_ok=True)
    json.dump({"tag": tag, "n_logical_qubits": n, "circuits": out}, open(f"{BASE}/circuit_analysis_connectivity/{tag}.json", "w"), indent=1)
    print("done", tag, file=sys.stderr, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    for t in ap.parse_args().tags:
        try:
            one(t)
        except Exception as e:  # noqa: BLE001
            print("FAILED", t, type(e).__name__, e, file=sys.stderr, flush=True)
