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

"""Reproducibility archive for every re-sampled run.

For each run tag writes, under experiments/crossnorb70q_infer/circuit_analysis_archive/<tag>/:
  logical_circuits.qpy   the 10 pre-transpile (logical) circuits, rebuilt from the stored
                         gate sequences with the same operator pool used for sampling
  manifest.json          everything needed to rerun or audit the run: molecule, checkpoint
                         (+ sha256), lineage/policy, shots, seeds, transpile settings, IBM
                         Runtime job id(s), the 10 gate sequences with the NAME of every
                         operator in them, paths of the saved counts / transpiled circuits,
                         library versions and sha256 of every script used.
The raw per-circuit counts and the transpiled circuits (what actually ran) are already saved by
resample_raw_counts.py as circuit_analysis_ibm_jobs/<tag>/ibm_kobe_shots100000/<job>/{counts.json,circuits.qpy}.

Usage (Slurm job): python save_manifest.py --tags tag1 tag2 ...
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import warnings

warnings.filterwarnings("ignore")

from sqd_gqe.run import _run_molecule_impl
from ca_common import load_cache

GPU_BIN = "/path/to/qcsc-prefect/algorithms/sbd/native/diag-gpu-mpi"
BASE = os.environ.get("INFER_ROOT", "/path/to/sqd-gqe-work/experiments/crossnorb70q_infer")  # Training 2 arms set INFER_ROOT to their own tree
REPO = "/path/to/sqd-gqe-work"
SCRIPTS = [
    "scripts/analysis/circuit_analysis/resample_raw_counts.py",
    "scripts/analysis/circuit_analysis/rerank_circuits.py",
    "scripts/analysis/circuit_analysis/rerecover_target.py",
    "scripts/analysis/circuit_analysis/select_new.py",
    "scripts/analysis/circuit_analysis/ca_common.py",
    "scripts/analysis/coverage_training/infer_crossnorb_70q_lineages.py",
    "sqd_gqe/gqe/ibm_hardware_sampler.py",
    "sqd_gqe/sqd/pipeline.py",
    "sqd_gqe/sqd/backend_sbd_gpu.py",
]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("qiskit", "qiskit_ibm_runtime", "ffsim", "numpy", "torch"):
        try:
            out[m] = __import__(m).__version__
        except Exception as e:  # noqa: BLE001
            out[m] = f"unavailable ({type(e).__name__})"
    return out


def one(tag):
    from qiskit import qpy
    from sqd_gqe.gqe.ibm_hardware_sampler import IbmHardwareSampler

    cache = load_cache(tag)
    run = json.load(open(f"{BASE}/circuit_analysis/{tag}.json"))
    seqs = [list(c["seq"]) for c in run["circuits"]]
    trained = cache["policy_mode"] == "trained"
    enc = {"a1": "a1_xattn_large_bounded", "a5": "a5_large_bounded"}[cache["lineage"].split("-")[0]]
    holder: dict = {}
    _run_molecule_impl(
        cache["molecule_path"], backend="ideal",
        checkpoint=cache["checkpoint"] if trained else None, adapt_iters=0, seed=1234,
        zeroshot_batches=1, diag_backend="sbd", sbd_binary_path=GPU_BIN,
        sbd_launcher=["mpirun", "-np", "2"],
        sbd_solver_kwargs={"task_comm_size": 1, "adet_comm_size": 1, "bdet_comm_size": 2},
        recover=False, sampler_kind="ffsim", shots=cache["shots"], max_depth_for_diag=500,
        ngates=cache["ngates"], refine_method="tp_ufd",
        encoder_override=None if trained else enc, stop_after_build=True, _factory_holder=holder,
    )
    pipeline = holder["pipeline"]
    pool = pipeline.operator_pool
    sampler = IbmHardwareSampler(pool, shots_count=cache["shots"], backend_name="ibm_kobe", seed=1234)
    logical = [sampler.prepare_circuit(s) for s in seqs]

    od = f"{BASE}/circuit_analysis_archive/{tag}"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/logical_circuits.qpy", "wb") as f:
        qpy.dump(logical, f)

    ibm = glob.glob(f"{BASE}/circuit_analysis_ibm_jobs/{tag}/ibm_kobe_shots100000/*/")
    ckpt = cache["checkpoint"]
    manifest = {
        "tag": tag, "molecule_label": cache["molecule_label"], "molecule_path": cache["molecule_path"],
        "lineage": cache["lineage"], "policy_mode": cache["policy_mode"],
        "checkpoint": ckpt, "checkpoint_sha256": sha256(ckpt) if ckpt and os.path.exists(ckpt) else None,
        "ngates": cache["ngates"], "n_qubits": cache["n_qubits"], "nelec": cache["nelec"], "norb": cache["norb"],
        "shots_per_circuit": cache["shots"], "backend": "ibm_kobe",
        "transpile": {"optimization_level": 1, "seed_transpiler": 0,
                      "note": "IbmHardwareSampler defaults; no dynamical decoupling, no error mitigation"},
        "original_cache": cache.get("cache_source") or "see ca_common.cache_path(tag)",
        "ibm_runtime_job_ids": run["ibm_runtime_job_ids"], "resample_slurm_job_id": run.get("slurm_job_id"),
        "saved_data": {
            "counts_json": [p + "counts.json" for p in ibm],
            "transpiled_circuits_qpy": [p + "circuits.qpy" for p in ibm],
            "logical_circuits_qpy": f"{od}/logical_circuits.qpy",
            "per_circuit_summary": f"{BASE}/circuit_analysis/{tag}.json",
        },
        "circuits": [
            {"index": i, "sequence": s, "operator_names": [pool[int(t)].name for t in s],
             "is_best_old_rule": run["circuits"][i]["is_best"], "is_second_old_rule": run["circuits"][i]["is_second"],
             "depth": run["circuits"][i]["cost"]["depth"], "two_qubit_gates": run["circuits"][i]["cost"]["two_qubit_gates"]}
            for i, s in enumerate(seqs)
        ],
        "versions": versions(),
        "script_sha256": {p: sha256(f"{REPO}/{p}") for p in SCRIPTS if os.path.exists(f"{REPO}/{p}")},
    }
    json.dump(manifest, open(f"{od}/manifest.json", "w"), indent=1)
    print(f"[manifest] {tag}: wrote {od} ({len(logical)} logical circuits)", file=sys.stderr, flush=True)
    pipeline.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    for t in args.tags:
        try:
            one(t)
        except Exception as e:  # noqa: BLE001
            print(f"[manifest] {t}: FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
