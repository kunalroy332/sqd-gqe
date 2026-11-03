# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Training-2 inference, step 1: generate the 10 gate sequences of one (arm, lineage, molecule) run. NO hardware call.

The earlier campaign sampled the sequences and measured them on ibm_kobe in one step (infer_crossnorb_70q_lineages.py --sample-only), and every later step
(resample_raw_counts.py, rerank_circuits.py, select_new.py, rerecover_target.py) reads the cache that step wrote. This script reproduces the part that matters
for a new checkpoint -- build the policy of the checkpoint on the target molecule, draw 10 sequences with the same sampler settings (seed 1234, ngates from the
checkpoint) -- and writes a cache of the same layout, with the best and second circuits left empty (they are chosen after the shots exist, by the ranking
step). resample_raw_counts.py then measures the stored sequences on hardware, rerank_circuits.py ranks them, select_new.py picks best, second, merged and
the two E+D circuits, and rerecover_target.py recovers each.

Arms (checkpoints are the FINAL Stage-2 policies of Training 2, <exp>/ckpt/<run>.pt, not the *_best.pt files):
    hf     t2p_s2_<lineage>   HF penalty 0.05
    tuned  t2t_s2_<lineage>   jointly tuned reward (A1-70q: the iteration-129 policy; that run ended by a segmentation fault at iteration 137)

Usage (cluster, inside a Slurm job):
    INFER_ROOT=<exp>/inference_hf python3 generate_sequences.py --arm hf --lineage a1-46q --molecule-key h2o_m_no23
    python3 generate_sequences.py --list          # the 48 (arm, lineage, molecule) runs; --dry prints what would be written
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

EXP = "/path/to/sqd-gqe-work/experiments"
T2 = f"{EXP}/training2"
GPU_BIN = "/path/to/qcsc-prefect/algorithms/sbd/native/diag-gpu-mpi"
SHOTS = 100_000
SEED = 1234
ARMS = {"hf": ("t2p_s2", "inference_hf"), "tuned": ("t2t_s2", "inference_tuned"), "t1": (None, "inference_t1")}
ENC = {"a1": "a1_xattn_large_bounded", "a5": "a5_large_bounded"}
NGATES = {"46q": 28, "70q": 44}

# The 24 trained runs of the first campaign: lineage -> molecule keys (key, path, qubits, label). Same molecules, same files, so the three regimes are comparable.
M = {
    "L46": ("molecules/large/SnOH3-me_L_23o32e_mp2no_def2svp.yaml", 46, "SnOH3-me L (norb=23, 46q)"),
    "sno32": ("molecules/small/SnOH3-me_S_16o24e_mp2no_def2svp.yaml", 32, "SnOH3-me S (norb=16, 32q)"),
    "h2o_s19": ("molecules/small/SnOH3-me-h2o_S_19o30e_mp2no_def2svp.yaml", 38, "SnOH3-me-h2o S (norb=19, 38q)"),
    "2h2o_s22": ("molecules/small/SnOH3-me-2h2o_S_22o36e_mp2no_def2svp.yaml", 44, "SnOH3-me-2h2o S (norb=22, 44q)"),
    "h2o_m_no23": ("molecules/medium/SnOH3-me-h2o_M_23o30e_mp2no_def2svp.yaml", 46, "SnOH3-me-h2o M (norb=23, 46q)"),
    "sno38": ("molecules/medium/SnOH3-me_M_19o24e_mp2no_def2svp.yaml", 38, "SnOH3-me M (norb=19, 38q)"),       # not in the first campaign
    "h2o_l28": ("molecules/large/SnOH3-me-h2o_L_28o40e_mp2no_def2svp.yaml", 56, "SnOH3-me-h2o L (norb=28, 56q)"),  # not in the first campaign's results
    "llno28": ("molecules/large/SnOH3-me_LL_28o42e_mp2no_def2svp.yaml", 56, "SnOH3-me LL (norb=28, 56q)"),
    "2h2o_m_no29": ("molecules/medium/SnOH3-me-2h2o_M_29o36e_mp2no_def2svp.yaml", 58, "SnOH3-me-2h2o M (norb=29, 58q)"),
    "h2o_ll33": ("molecules/large/SnOH3-me-h2o_LL_33o50e_mp2no_def2svp.yaml", 66, "SnOH3-me-h2o LL (norb=33, 66q)"),
    "2h2o_l35": ("molecules/large/SnOH3-me-2h2o_L_35o48e_mp2no_def2svp.yaml", 70, "SnOH3-me-2h2o L (norb=35, 70q)"),
}
ALL_MOLECULES = ["sno32", "sno38", "L46", "llno28", "h2o_s19", "h2o_m_no23", "h2o_l28", "h2o_ll33", "2h2o_s22", "2h2o_m_no29", "2h2o_l35"]   # the 11 active-space tiers
TARGETS = {lin: list(ALL_MOLECULES) for lin in ("a1-46q", "a5-46q", "a1-70q", "a5-70q")}
# The first campaign ran the 46q lineages on the 5 tiers of at most 46 qubits and the 70q lineages on 7; the others are new for every model set.
FIRST_CAMPAIGN = {
    "a1-46q": {"2h2o_s22", "L46", "h2o_m_no23", "h2o_s19", "sno32"}, "a5-46q": {"2h2o_s22", "L46", "h2o_m_no23", "h2o_s19", "sno32"},
    "a1-70q": {"2h2o_l35", "2h2o_m_no29", "L46", "h2o_ll33", "h2o_m_no23", "llno28", "sno32"}, "a5-70q": {"2h2o_l35", "2h2o_m_no29", "L46", "h2o_ll33", "h2o_m_no23", "llno28", "sno32"},
}


def all_runs(arms=("hf", "tuned")):
    return [(arm, lin, mk) for arm in arms for lin, mks in TARGETS.items() for mk in mks]


T1_CKPT = {   # the checkpoints the first campaign used (Training 1, *_best.pt)
    "a1-46q": f"{EXP}/production_final/ckpt/hw_continue_46q_a1_s1234_v4_best.pt", "a5-46q": f"{EXP}/production_final/ckpt/hw_continue_46q_a5_s1234_v3_best.pt",
    "a1-70q": f"{EXP}/production_final/ckpt/kawasaki_hw_70q_a1_s1234_v6_best.pt", "a5-70q": f"{EXP}/production_final/ckpt/kawasaki_hw_70q_a5_s1234_v5_best.pt",
}


def checkpoint(arm, lineage):
    if arm == "t1":
        return T1_CKPT[lineage]
    run = f"{ARMS[arm][0]}_{lineage.replace('-', '_')}"
    return f"{T2}/ckpt/{run}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--lineage", choices=list(TARGETS))
    ap.add_argument("--molecule-key", choices=list(M))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the plan and check the checkpoint, write nothing")
    a = ap.parse_args()
    if a.list:
        for arm, lin, mk in all_runs():
            print(f"{arm:6s} {lin:7s} {mk:12s} {M[mk][1]}q  ckpt={os.path.basename(checkpoint(arm, lin))}")
        print(len(all_runs()), "runs")
        return
    assert a.arm and a.lineage and a.molecule_key, "need --arm --lineage --molecule-key"
    if a.molecule_key not in TARGETS[a.lineage]:
        sys.exit(f"{a.molecule_key} is not a target of {a.lineage}")
    root = os.environ.get("INFER_ROOT") or f"{EXP}/{ARMS[a.arm][1]}"
    path, nq, label = M[a.molecule_key]
    ckpt = checkpoint(a.arm, a.lineage)
    tag = f"{a.molecule_key}_{a.lineage}_trained"
    out = f"{root}/cache/crossnorb70q_{tag}_sample.json"
    if a.dry:
        print(f"[dry] {a.arm} {tag}: checkpoint {ckpt} (exists={os.path.exists(ckpt)}), molecule {path}, would write {out}")
        return

    import torch
    from sqd_gqe.run import _run_molecule_impl
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)
    real_ngates = int(state["transformer.wpe.weight"].shape[0]) - 1
    want = NGATES["46q" if "46q" in a.lineage else "70q"]
    if real_ngates != want:
        print(f"[generate] WARNING: checkpoint ngates {real_ngates} != expected {want}; using the checkpoint's", file=sys.stderr, flush=True)
    vocab = int(state["transformer.wte.weight"].shape[0])
    print(f"[generate] {a.arm} {tag}: ckpt={os.path.basename(ckpt)} vocab={vocab} ngates={real_ngates} encoder_state={sd.get('prompt_encoder_state_dict') is not None}",
          file=sys.stderr, flush=True)

    holder = {}
    # sequences_only=True returns right after the policy has drawn its 10 sequences, before any sampler call (no ffsim, no hardware, no diagonalization).
    # The policy, conditioning, seed and settings are those of the hardware run, so the sequences are the draw that run would have measured.
    _run_molecule_impl(
        path, backend="ideal", checkpoint=ckpt, adapt_iters=0, seed=SEED, zeroshot_batches=1, diag_backend="sbd", sbd_binary_path=GPU_BIN,
        sbd_launcher=["mpirun", "-np", "2"], sbd_solver_kwargs={"task_comm_size": 1, "adet_comm_size": 1, "bdet_comm_size": 2},
        recover=False, sampler_kind="ffsim", shots=1000, max_depth_for_diag=False, ngates=real_ngates, refine_method="tp_ufd",
        encoder_override=None, sequences_only=True, _factory_holder=holder)
    pipeline = holder["pipeline"]
    seqs = [list(s) for s in holder["sequences"]]
    nelec = list(pipeline.nelec)
    norb = int(pipeline.norb if hasattr(pipeline, "norb") else nq // 2)
    assert len(seqs) == 10, f"expected 10 sequences, got {len(seqs)}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cache = {
        "molecule_key": a.molecule_key, "molecule_path": path, "molecule_label": label, "lineage": a.lineage, "policy_mode": "trained",
        "checkpoint": ckpt, "n_qubits": nq, "ngates": real_ngates, "shots": SHOTS, "seed": SEED, "max_depth_for_diag": 50000,
        "nelec": nelec, "norb": norb, "arm": a.arm,
        "batch_summary": [{"seq": s, "energy": None, "n_distinct": None, "depth_skipped": False, "cost": {}} for s in seqs],
        "best": {"seq": seqs[0], "raw_energy": None, "cost": {}, "counts": {}},
        "second_best": {"seq": seqs[1], "raw_energy": None, "cost": {}, "counts": {}},
        "merged": {"counts": {}},
    }
    json.dump(cache, open(out, "w"))
    print(f"[generate] wrote {out}", file=sys.stderr, flush=True)
    pipeline.close()


if __name__ == "__main__":
    main()
