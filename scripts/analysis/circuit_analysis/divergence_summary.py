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

"""Summary statistics of the ideal divergence from the LUCJ circuit, per set, for the manuscript text.
Usage: python3 divergence_summary.py   (reads <tree>/divergence/*.json and <tree>/selection_batch1.json)"""
import glob, json, os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TREES = {"t1": "circuit_analysis", "hf": "circuit_analysis_hf", "tuned": "circuit_analysis_tuned"}
out = {}
for k, t in TREES.items():
    sel = {}
    for f in glob.glob(f"{REPO}/{t}/selection_batch*.json"):
        sel.update(json.load(open(f)))
    R = {"best": [], "second": [], "other": []}
    ops = []
    for f in sorted(glob.glob(f"{REPO}/{t}/divergence/*_trained.json")):
        dv = json.load(open(f)); tag = dv["tag"]; s = sel.get(tag)
        if not s:
            continue
        data = {c["index"]: c for c in json.load(open(f"{REPO}/{t}/data/{tag}.json"))["circuits"]}
        for c in dv["circuits"]:
            role = "best" if c["index"] == s["best"] else "second" if c["index"] == s["second"] else "other"
            R[role].append((c["F"], c["H"], c["JS"], c["d_HF"], data[c["index"]]["depth_live"], dv["molecule"]))
            if c.get("op_norm") is not None:
                ops.append((tag, c["index"], role, c["op_norm"]))
    out[k] = {"roles": {r: len(v) for r, v in R.items()}, "ops": ops}
    print(f"== {k}")
    for role, v in R.items():
        if not v: continue
        a = np.array([x[:5] for x in v])
        print(f"  {role:6s} n={len(v):3d}  F med {np.median(a[:,0]):.3f}  H med {np.median(a[:,1]):.3f}  JS med {np.median(a[:,2]):.3f}  d_HF med {np.median(a[:,3]):.3f}  depth med {np.median(a[:,4]):.0f}  F>0.5: {int((a[:,0]>0.5).sum())}/{len(v)}")
    if ops:
        o = [x[3] for x in ops]
        print(f"  op-norm LB n={len(o)} min {min(o):.2f} med {np.median(o):.2f} max {max(o):.2f}")
json.dump(out, open("/tmp/div_summary.json", "w"), default=str)
