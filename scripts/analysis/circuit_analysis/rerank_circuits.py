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

"""Re-compute every circuit's ranking-stage energy with a larger subspace.

Uses the per-circuit raw counts saved by resample_raw_counts.py and the SAME energy
path as the original ranking (SqdPipeline.energy_of_counts: keep electron-number-
conserving bitstrings, expand, fold to unique alphas, one-shot GPU diagonalization),
but with max_dim=40000 instead of 2000, i.e. an alpha cap of 200 and net CI dim 40000.

Circuits with zero valid (electron-number-conserving) shots still give inf: there is
nothing to diagonalize, whatever the subspace size.

Usage (on roquo, in a Slurm job):  python rerank_circuits.py --tag <molecule>_<lineage>_<policy>
"""
import argparse
import glob
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from sqd_gqe.run import _run_molecule_impl
from ca_common import load_cache

GPU_BIN = "/path/to/qcsc-prefect/algorithms/sbd/native/diag-gpu-mpi"
BASE = os.environ.get("INFER_ROOT", "/path/to/sqd-gqe-work/experiments/crossnorb70q_infer")  # Training 2 arms set INFER_ROOT to their own tree
RANK_MAX_DIM = 40000  # alpha cap = round(sqrt(40000)) = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag

    cache = load_cache(tag)
    run = json.load(open(f"{BASE}/circuit_analysis/{tag}.json"))
    cj = glob.glob(f"{BASE}/circuit_analysis_ibm_jobs/{tag}/ibm_kobe_shots100000/*/counts.json")
    assert len(cj) == 1, f"{tag}: expected one counts.json, found {cj}"
    saved = json.load(open(cj[0]))["circuits"]
    assert len(saved) == len(run["circuits"])

    trained = cache["policy_mode"] == "trained"
    enc = {"a1": "a1_xattn_large_bounded", "a5": "a5_large_bounded"}[cache["lineage"].split("-")[0]]
    holder: dict = {}
    _run_molecule_impl(
        cache["molecule_path"], backend="ideal",
        checkpoint=cache["checkpoint"] if trained else None, adapt_iters=0, seed=1234,
        zeroshot_batches=1, diag_backend="sbd", sbd_binary_path=GPU_BIN,
        sbd_launcher=["mpirun", "-np", "2"],
        sbd_solver_kwargs={"task_comm_size": 1, "adet_comm_size": 1, "bdet_comm_size": 2},
        sbd_timeout_s=36000.0, max_dim=RANK_MAX_DIM, refine_method="tp_ufd",
        recover=False, sampler_kind="ffsim", shots=cache["shots"], max_depth_for_diag=500,
        ngates=cache["ngates"], encoder_override=None if trained else enc,
        stop_after_build=True, _factory_holder=holder,
    )
    pipeline = holder["pipeline"]
    assert pipeline._sbd.max_dim == RANK_MAX_DIM, f"sbd max_dim is {pipeline._sbd.max_dim}"
    na, nb = cache["nelec"]

    out_circuits = []
    for c, cc in zip(run["circuits"], saved):
        counts = {tuple(int(x) for x in k.split(",")): n for k, n in cc["decoded_counts"].items()}
        valid = sum(n for (a, b), n in counts.items()
                    if bin(a).count("1") == na and bin(b).count("1") == nb)
        vdist = sum(1 for (a, b) in counts if bin(a).count("1") == na and bin(b).count("1") == nb)
        e = float(pipeline.energy_of_counts(counts)) if valid else float("inf")
        print(f"[rerank] {tag} circuit {c['index'] + 1}: valid_shots={valid} valid_distinct={vdist} "
              f"energy(40000)={e}", file=sys.stderr, flush=True)
        out_circuits.append({
            "index": c["index"], "seq": c["seq"], "valid_shots": valid, "valid_distinct": vdist,
            "energy_40000": e if math.isfinite(e) else None,
            "energy_2000_orig": c["ranking_energy"] if (c["ranking_energy"] is not None
                                                        and math.isfinite(c["ranking_energy"])) else None,
            "depth": c["cost"]["depth"], "two_qubit_gates": c["cost"]["two_qubit_gates"],
            "old_is_best": c["is_best"], "old_is_second": c["is_second"],
        })

    # Training-2 arms have no earlier (2000-dimensional) ranking, so the "E+D" picks of the first campaign (energy rank + number-of-distinct-bitstrings rank,
    # the old selection rule) are computed here from the 40000-dimensional energy and the measured number of distinct bitstrings. For the first campaign's own
    # runs nothing changes (their flags come from the stored ranking).
    if cache.get("arm"):
        fin = [c for c in out_circuits if c["energy_40000"] is not None]
        by_e = sorted(fin, key=lambda c: c["energy_40000"])
        ndist = {c["index"]: run["circuits"][c["index"]].get("n_distinct_measured", 0) for c in fin}
        by_d = sorted(fin, key=lambda c: -ndist[c["index"]])
        er = {c["index"]: i + 1 for i, c in enumerate(by_e)}
        dr = {c["index"]: i + 1 for i, c in enumerate(by_d)}
        order = sorted(fin, key=lambda c: (er[c["index"]] + dr[c["index"]], c["energy_40000"]))
        for c in out_circuits:
            c["old_is_best"] = bool(order) and c["index"] == order[0]["index"]
            c["old_is_second"] = len(order) > 1 and c["index"] == order[1]["index"]
            c["ed_rank_sum"] = (er[c["index"]] + dr[c["index"]]) if c["index"] in er else None

    od = f"{BASE}/circuit_analysis_rerank"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/{tag}.json", "w") as f:
        json.dump({"tag": tag, "rank_max_dim": RANK_MAX_DIM, "alpha_cap": 200,
                   "ibm_runtime_job_ids": run["ibm_runtime_job_ids"], "circuits": out_circuits}, f, indent=1)
    print(f"[rerank] wrote {od}/{tag}.json", file=sys.stderr, flush=True)
    pipeline.close()


if __name__ == "__main__":
    main()
