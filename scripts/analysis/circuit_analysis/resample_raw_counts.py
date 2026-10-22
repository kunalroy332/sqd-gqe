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

"""Re-sample the 10 gate sequences of one completed cross-norb inference run on
real IBM hardware and keep the raw per-circuit excitation histogram.

Why this exists: infer_crossnorb_70q_lineages.py's cache only stores raw counts
for best / second-best / pooled, never for all 10 circuits of the batch, so a
per-circuit raw-shot analysis is impossible from existing data. The 10 gate
sequences ARE stored (cache["batch_summary"][i]["seq"]); this script re-runs
exactly those sequences (same pool, same transpile seed, optimization_level=1)
with fresh shots, saves every circuit's raw counts via ibm_save_dir, and writes
a small JSON with one raw excitation histogram per circuit. No diagonalization.

Usage (on roquo, inside a Slurm job):
    python resample_raw_counts.py --cache <..._sample.json>
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from sqd_gqe.run import _run_molecule_impl
from ca_common import normalize, remap_paths
from sqd_gqe.gqe.circuit_metrics import raw_shot_excitation_histogram
from sqd_gqe.gqe.training_tracking import sequence_stats, spin_excitation_stats, hf_fraction, spin_imbalance

GPU_BIN = "/path/to/qcsc-prefect/algorithms/sbd/native/diag-gpu-mpi"
BASE = os.environ.get("INFER_ROOT", "/path/to/sqd-gqe-work/experiments/crossnorb70q_infer")  # Training 2 arms set INFER_ROOT to their own tree
OUT_DIR = f"{BASE}/circuit_analysis"
SAVE_DIR = f"{BASE}/circuit_analysis_ibm_jobs"
SHOTS = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    with open(args.cache) as f:
        cache = normalize(remap_paths(json.load(f)))
    tag = f"{cache['molecule_key']}_{cache['lineage']}_{cache['policy_mode']}"
    seqs = [list(c["seq"]) for c in cache["batch_summary"]]
    best_seq, second_seq = list(cache["best"]["seq"]), list(cache["second_best"]["seq"])
    print(f"[circuit-analysis] {tag}: {len(seqs)} stored sequences", file=sys.stderr, flush=True)

    # Rebuild the SAME operator pool the original sample-only run used (checkpoint vocab
    # padding included), exactly as do_from_cache does, but do not sample or diagonalize.
    trained = cache["policy_mode"] == "trained"
    holder: dict = {}
    _run_molecule_impl(
        cache["molecule_path"], backend="ideal",
        checkpoint=cache["checkpoint"] if trained else None, adapt_iters=0, seed=1234,
        zeroshot_batches=1, diag_backend="sbd", sbd_binary_path=GPU_BIN,
        sbd_launcher=["mpirun", "-np", "2"],
        sbd_solver_kwargs={"task_comm_size": 1, "adet_comm_size": 1, "bdet_comm_size": 2},
        recover=False, sampler_kind="ffsim", shots=cache["shots"], max_depth_for_diag=500,
        ngates=cache["ngates"], refine_method="tp_ufd",
        encoder_override=None if trained else _encoder_for(cache["lineage"]),
        stop_after_build=True, _factory_holder=holder,
    )
    pipeline = holder["pipeline"]
    pool, nelec = pipeline.operator_pool, tuple(cache["nelec"])
    bad = [i for s in seqs for i in s if int(i) >= len(pool)]
    if bad:
        raise RuntimeError(f"{tag}: {len(bad)} token ids exceed the rebuilt pool ({len(pool)}); "
                           "pool does not match the original run -- refusing to sample")

    from sqd_gqe.gqe.ibm_hardware_sampler import IbmHardwareSampler
    save_dir = f"{SAVE_DIR}/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    sampler = IbmHardwareSampler(
        pool, shots_count=SHOTS, backend_name="ibm_kobe", seed=1234,
        channel="ibm_quantum_platform",
        instance=os.environ.get("IBM_CRN"), token=os.environ.get("IBM_API_KEY"),
        save_dir=save_dir, dynamical_decoupling=False,
    )
    decoded = sampler._sample_batch(seqs)  # ONE Runtime job, all circuits as pubs
    job_ids = sorted(d for b in os.listdir(save_dir)
                     for d in os.listdir(os.path.join(save_dir, b)))

    circuits = []
    for i, (seq, counts, meta) in enumerate(zip(seqs, decoded, cache["batch_summary"])):
        h = raw_shot_excitation_histogram(counts, nelec, max_named_rank=4)
        circuits.append({
            # Training-2 arm caches (generate_sequences.py) carry no earlier ranking: best/second are chosen later from the 40000-dimensional ranking, so no circuit is
            # flagged here, and the cost is the live transpile depth measured by this very hardware job.
            "index": i, "seq": seq, "is_best": (False if cache.get("arm") else seq == best_seq), "is_second": (False if cache.get("arm") else seq == second_seq),
            "ranking_energy": meta["energy"], "depth_skipped": meta["depth_skipped"],
            "cost": (meta["cost"] if meta.get("cost") else {"depth": int(sampler.last_depths[i]), "two_qubit_gates": int(sampler.last_two_qubit_gates[i])}),
            "depth_live": int(sampler.last_depths[i]),
            "two_qubit_live": int(sampler.last_two_qubit_gates[i]),
            "raw_hist": {k: float(h[k]) for k in ("HF", "S", "D", "T", "Q", "higher", "combined", "n_dets")},
            # the same per-circuit chemistry quantities that Training 2 stored every iteration (sqd_gqe/gqe/training_tracking.py): operator
            # families and Givens roles of the generated sequence, and the spin-resolved excitation analysis of the measured shots
            "seq_stats": sequence_stats(seq, pool, nelec, int(cache["norb"])),
            "exc_stats": spin_excitation_stats(counts, nelec, int(cache["norb"])),
            "hf_fraction": hf_fraction(counts, nelec), "spin_imbalance": spin_imbalance(counts, nelec),
            "n_distinct_measured": len(counts),
        })
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/{tag}.json"
    with open(out, "w") as f:
        json.dump({
            "molecule_key": cache["molecule_key"], "molecule_label": cache["molecule_label"],
            "lineage": cache["lineage"], "policy_mode": cache["policy_mode"],
            "n_qubits": cache["n_qubits"], "shots": SHOTS,
            "ibm_runtime_job_ids": job_ids, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "circuits": circuits,
        }, f, indent=1)
    print(f"[circuit-analysis] wrote {out} (ibm jobs {job_ids})", file=sys.stderr, flush=True)
    pipeline.close()


def _encoder_for(lineage: str) -> str:
    return {"a1": "a1_xattn_large_bounded", "a5": "a5_large_bounded"}[lineage.split("-")[0]]


if __name__ == "__main__":
    main()
