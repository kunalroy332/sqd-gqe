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

"""For each re-sampled run: how many of each circuit's raw shots conserve the electron
numbers (popcount(alpha)==n_alpha and popcount(beta)==n_beta)? The ranking-stage
energy keeps only such bitstrings (ConfigurationSet.select_by_nelec), so a circuit
with ZERO valid shots gets energy = inf regardless of the diagonalization subspace.

Standard library only, so it runs on the cluster login node.
Usage: python3 valid_shot_check.py <experiments/crossnorb70q_infer dir>
"""
import glob
import json
import os
import sys

base = sys.argv[1]
for rj in sorted(glob.glob(f"{base}/circuit_analysis/*.json")):
    tag = os.path.splitext(os.path.basename(rj))[0]
    run = json.load(open(rj))
    cache = json.load(open(f"{base}/cache/crossnorb70q_{tag}_sample.json"))
    na, nb = cache["nelec"]
    cj = glob.glob(f"{base}/circuit_analysis_ibm_jobs/{tag}/ibm_kobe_shots100000/*/counts.json")
    if not cj:
        print(f"{tag}: no counts.json"); continue
    counts = json.load(open(cj[0]))["circuits"]
    print(f"\n{tag}  (nelec={na},{nb})")
    print("  idx  depth  rank_E          valid_shots  valid_distinct  n_distinct")
    for c, cc in zip(run["circuits"], counts):
        valid = vdist = 0
        for k, n in cc["decoded_counts"].items():
            a, b = (int(x) for x in k.split(","))
            if bin(a).count("1") == na and bin(b).count("1") == nb:
                valid += n; vdist += 1
        e = c["ranking_energy"]
        es = f"{e:.4f}" if e is not None and e == e and abs(e) != float("inf") else "inf"
        print(f"  {c['index'] + 1:>3} {c['cost']['depth']:>6}  {es:>12}  {valid:>11}  {vdist:>14}  {len(cc['decoded_counts']):>10}")
