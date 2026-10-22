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

"""Operator-family and Givens-role statistics of the Training-1 inference sequences (stored in the first campaign's caches, which have no seq_stats).
Rebuilds each run's operator pool exactly as resample_raw_counts.py does (no sampler call, no hardware) and applies training_tracking.sequence_stats to the FULL stored sequence including the start token, exactly as resample_raw_counts.py does for the new model sets, so the three sets are counted the same way (one identity token per circuit more than in training, where the start token is dropped).
Usage (cluster, in a job): python3 t1_seq_stats.py --lineage a1-46q   -> experiments/inference_t1/seq_stats/<tag>.json"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts/analysis/circuit_analysis")
from ca_common import normalize, remap_paths
from sqd_gqe.run import _run_molecule_impl
from sqd_gqe.gqe.training_tracking import sequence_stats
GPU_BIN = "/path/to/qcsc-prefect/algorithms/sbd/native/diag-gpu-mpi"
EXP = "/path/to/sqd-gqe-work/experiments"
ap = argparse.ArgumentParser(); ap.add_argument("--lineage", required=True); a = ap.parse_args()
os.makedirs(f"{EXP}/inference_t1/seq_stats", exist_ok=True)
for f in sorted(glob.glob(f"{EXP}/crossnorb70q_infer/cache/crossnorb70q_*_{a.lineage}_trained_sample.json")):
    c = normalize(remap_paths(json.load(open(f)))); tag = f"{c['molecule_key']}_{c['lineage']}_{c['policy_mode']}"
    out = f"{EXP}/inference_t1/seq_stats/{tag}.json"
    if os.path.exists(out): continue
    holder = {}
    _run_molecule_impl(c["molecule_path"], backend="ideal", checkpoint=c["checkpoint"], adapt_iters=0, seed=1234, zeroshot_batches=1, diag_backend="sbd",
                       sbd_binary_path=GPU_BIN, sbd_launcher=["mpirun", "-np", "2"], sbd_solver_kwargs={"task_comm_size": 1, "adet_comm_size": 1, "bdet_comm_size": 2},
                       recover=False, sampler_kind="ffsim", shots=c["shots"], max_depth_for_diag=500, ngates=c["ngates"], refine_method="tp_ufd",
                       encoder_override=None, stop_after_build=True, _factory_holder=holder)
    pool, nelec = holder["pipeline"].operator_pool, tuple(c["nelec"])
    rows = [sequence_stats(list(x["seq"]), pool, nelec, int(c["norb"])) for x in c["batch_summary"]]
    json.dump({"tag": tag, "circuits": rows}, open(out, "w")); print("wrote", out, flush=True)
    holder["pipeline"].close()
