# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Add the Training-2-style chemistry quantities to inference runs that were made before they were recorded.

For every stored run in circuit_analysis/data/<tag>.json this recomputes, from the SAME stored per-circuit shots
(circuit_analysis/raw_archive/circuit_analysis_ibm_jobs/<tag>/.../counts.json -> decoded_counts) and the stored token sequences
(cache['batch_summary'][i]['seq']), the per-circuit exc_stats (spin-resolved excitation analysis), hf_fraction and spin_imbalance, and,
when a pool is available, seq_stats. Writes circuit_analysis/chem/<tag>.json. No hardware call, no diagonalization.

The operator pool is needed for seq_stats only (operator families, Givens roles); pass --with-pool on a cluster node that can rebuild it.
Usage: python3 backfill_chem_stats.py <circuit_analysis dir> [--with-pool]
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sqd_gqe.gqe.training_tracking import hf_fraction, spin_excitation_stats, spin_imbalance  # noqa: E402

root = Path(sys.argv[1])
out = root / "chem"
out.mkdir(exist_ok=True)
n = 0
for f in sorted(glob.glob(str(root / "data" / "*.json"))):
    tag = Path(f).stem
    d = json.load(open(f))
    key = f"{d['molecule_key']}_{d['lineage']}_{d['policy_mode']}"
    cj = glob.glob(str(root / "raw_archive" / "circuit_analysis_ibm_jobs" / key / "ibm_kobe_shots100000" / "*" / "counts.json"))
    cache = glob.glob(str(root / "raw_archive" / "cache" / f"*{key}*_sample.json"))
    if len(cj) != 1 or len(cache) != 1:
        print("skip", tag, "counts", len(cj), "cache", len(cache)); continue
    c = json.load(open(cache[0])); nelec, norb = tuple(c["nelec"]), int(c["norb"])
    saved = json.load(open(cj[0]))["circuits"]
    rows = []
    for i, x in enumerate(saved):
        dc = {tuple(int(t) for t in k.split(",")): v for k, v in x["decoded_counts"].items()}
        rows.append({"index": i, "seq": c["batch_summary"][i]["seq"], "depth": x["depth"], "exc_stats": spin_excitation_stats(dc, nelec, norb),
                     "hf_fraction": hf_fraction(dc, nelec), "spin_imbalance": spin_imbalance(dc, nelec)})
    json.dump({"tag": key, "nelec": list(nelec), "norb": norb, "circuits": rows}, open(out / f"{key}.json", "w"))
    n += 1
print("wrote", n, "files to", out)
