# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Re-run the final recovery for ONE target (best, second or merged) of one run, using the
fresh per-circuit raw counts saved by resample_raw_counts.py.

Recovery path is identical to infer_crossnorb_70q_lineages.py's do_from_cache.run_final:
recover_and_diagonalize(subspace_dim=20000, 25 iterations, seed 1234), excitation of the
recovered alpha set, singles injection. Only the input counts differ (fresh shots, and the
circuit(s) chosen by the new ranking).

Usage: rerecover_target.py --tag T --target best|second|merged --idx 3        (best/second)
       rerecover_target.py --tag T --target merged --idx 0,1,4,5              (pooled circuits)
"""
import argparse
import glob
import json
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, "/path/to/sqd-gqe-work/scripts/analysis/coverage_training")

from infer_crossnorb_70q_lineages import (  # noqa: E402
    FINAL_RECOVER_ITERS, GPU_BIN, RANKING_PAIR_CAP, excitation_of, singles_injected_energy,
)
from sqd_gqe.run import _run_molecule_impl  # noqa: E402
from ca_common import load_cache  # noqa: E402
from sqd_gqe.gqe.circuit_metrics import raw_shot_excitation_histogram  # noqa: E402

BASE = os.environ.get("INFER_ROOT", "/path/to/sqd-gqe-work/experiments/crossnorb70q_infer")  # Training 2 arms set INFER_ROOT to their own tree
SUBSPACE_DIM = 20000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", required=True, choices=["best", "second", "merged", "oldbest", "oldsecond"])
    ap.add_argument("--idx", required=True, help="circuit index (0-based); comma list for merged")
    args = ap.parse_args()
    tag, target = args.tag, args.target
    idx = [int(x) for x in args.idx.split(",")]
    if target != "merged" and len(idx) != 1:
        raise SystemExit("best/second take exactly one circuit index")

    cache = load_cache(tag)
    run = json.load(open(f"{BASE}/circuit_analysis/{tag}.json"))
    cj = glob.glob(f"{BASE}/circuit_analysis_ibm_jobs/{tag}/ibm_kobe_shots100000/*/counts.json")
    assert len(cj) == 1, cj
    saved = json.load(open(cj[0]))["circuits"]

    pooled: dict = defaultdict(int)
    for i in idx:
        for k, n in saved[i]["decoded_counts"].items():
            pooled[tuple(int(x) for x in k.split(","))] += n
    counts = dict(pooled)
    print(f"[rerecover] {tag} {target} circuits {idx}: {len(counts)} distinct, "
          f"{sum(counts.values())} shots", file=sys.stderr, flush=True)

    trained = cache["policy_mode"] == "trained"
    enc = {"a1": "a1_xattn_large_bounded", "a5": "a5_large_bounded"}[cache["lineage"].split("-")[0]]
    holder: dict = {}
    _run_molecule_impl(
        cache["molecule_path"], backend="ideal",
        checkpoint=cache["checkpoint"] if trained else None, adapt_iters=0, seed=1234,
        zeroshot_batches=1, diag_backend="sbd", sbd_binary_path=GPU_BIN,
        sbd_launcher=["mpirun", "-np", "2"],
        sbd_solver_kwargs={"task_comm_size": 1, "adet_comm_size": 1, "bdet_comm_size": 2},
        sbd_timeout_s=36000.0, max_dim=RANKING_PAIR_CAP, refine_method="tp_ufd",
        recover=False, sampler_kind="ffsim", shots=cache["shots"], max_depth_for_diag=500,
        ngates=cache["ngates"], encoder_override=None if trained else enc,
        stop_after_build=True, _factory_holder=holder,
    )
    pipeline = holder["pipeline"]
    recoverer = pipeline._sbd
    nelec, norb = tuple(cache["nelec"]), cache["norb"]

    diag: list = []
    state: dict = {}
    energy, _ci, hist = recoverer.recover_and_diagonalize(
        counts, samples_per_batch=512, num_batches=1, subspace_dim=SUBSPACE_DIM,
        max_iterations=FINAL_RECOVER_ITERS, seed=1234, diagnostics_out=diag, final_state_out=state)
    strs_a = state["best_strs_a"]
    exc = excitation_of(strs_a, nelec)
    s_energy, n_new, n_aug = singles_injected_energy(recoverer, strs_a, nelec, norb, label=target.upper())

    circ = [run["circuits"][i] for i in idx]
    out = {
        "tag": tag, "target": target, "circuit_indices": idx,
        "molecule_label": cache["molecule_label"], "lineage": cache["lineage"],
        "policy_mode": cache["policy_mode"], "n_qubits": cache["n_qubits"],
        "subspace_dim": SUBSPACE_DIM, "iterations": FINAL_RECOVER_ITERS,
        "n_distinct": len(counts), "n_shots": sum(counts.values()),
        "final_energy": energy, "step_history": hist, "excitation": exc,
        "singles_injected_energy": s_energy, "singles_injected_n_new": n_new,
        "singles_injected_n_dets_total": n_aug,
        "recovered_alpha_strings": [int(x) for x in strs_a],
        "input_counts_source": "circuit_analysis_ibm_jobs/<tag>/ibm_kobe_shots100000/<job>/counts.json "
                               "(decoded_counts of the listed circuit_indices, summed for merged)",
        "raw_shot_excitation": raw_shot_excitation_histogram(counts, nelec, max_named_rank=4),
        "cost": circ[0]["cost"] if target != "merged" else None,
        "seq": circ[0]["seq"] if target != "merged" else None,
        "ibm_runtime_job_ids": run["ibm_runtime_job_ids"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    od = f"{BASE}/circuit_analysis_rerecovery"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/{tag}__{target}.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"[rerecover] wrote {od}/{tag}__{target}.json energy={energy}", file=sys.stderr, flush=True)
    pipeline.close()


if __name__ == "__main__":
    main()
