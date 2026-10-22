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

"""Compare a trained checkpoint's zero-shot circuit against a LUCJ(reps=1)
baseline, both transpiled to REAL IBM Quantum hardware.

For each molecule: run zero-shot inference to get the policy's best-energy
circuit (same generation loop as sqd_gqe/run.py's run_molecule, duplicated
here since best_seq isn't exposed through ChemistryReport), build the
LUCJ(reps=1) baseline the same way benchmark/verify_shallow_vs_lucj1.py does
(operator_pool.n_reps=1 override, not read out of the trained pool whose seed
op defaults to reps=2), transpile both to the given real backend, and report
total gates + 2-qubit-gate count for each.

optimization_level defaults to 1: level 3's SABRE routing search hangs on
LUCJ-derived circuits at 38+ qubits (confirmed empirically -- see
docs/2026-08-13 notes / job 30138), while level 1 transpiles the same
circuits in ~1s.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def zero_shot_all_seqs(molecule_yaml: str, checkpoint: str, num_samples: int = 10, seed: int = 0):
    """Duplicates run_molecule's generation loop (sqd_gqe/run.py) far enough
    to recover every sampled sequence -- ChemistryReport doesn't expose these.

    Returns ALL num_samples sequences, not just one: transpiled gate/2q-count
    for a fixed op-count sequence varies noticeably by WHICH (p,q,theta)
    values got selected (different qubit pairs route differently on a real
    heavy-hex-ish device), and torch sampling isn't bitwise-reproducible
    across separate process launches even with the same seed (confirmed
    empirically -- picking a single "representative" sample, even by
    matching real-op count, is not a stable point estimate). Average across
    the whole batch instead."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_state = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt_state.get("state_dict", ckpt_state)
    wte = sd.get("transformer.wte.weight")
    ckpt_vocab = int(wte.shape[0]) if wte is not None else None
    overrides = ["model=conditioned"]
    if ckpt_vocab:
        overrides.append(f"+operator_pool.max_vocab={ckpt_vocab}")

    abs_cfg = str(REPO_ROOT / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=overrides)
    cfg.molecule = OmegaConf.load(molecule_yaml)
    cfg.trainer.num_samples = num_samples
    cfg.trainer.seed = seed

    factory = SqdFactory()
    system = factory.build_system(cfg)
    policy = factory.build_policy(cfg).to(device)
    schedule = factory.build_beta_schedule(cfg)
    pool = factory.build_operator_pool(cfg)

    hw = get_backend("ideal")
    cond = ConditioningContext.build(system, hw)
    if hasattr(policy, "set_condition"):
        policy.set_condition(cond.tensor(device))

    policy.load_state_dict(
        {k: v for k, v in sd.items() if k in policy.state_dict()
         and policy.state_dict()[k].shape == v.shape},
        strict=False,
    )

    torch.manual_seed(seed)
    state = {"idx": torch.zeros((num_samples, 1), dtype=torch.long, device=device)}
    with torch.no_grad():
        for _ in range(cfg.ngates):
            nxt = policy.act(state, schedule.beta())
            state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)

    idx_batch = state["idx"][:, 1:].cpu().tolist()
    return pool, idx_batch


def lucj1_pool_and_seq(molecule_yaml: str):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory

    abs_cfg = str(REPO_ROOT / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=["operator_pool.n_reps=1"])
    cfg.molecule = OmegaConf.load(molecule_yaml)
    factory = SqdFactory()
    factory.build_system(cfg)
    pool = factory.build_operator_pool(cfg)
    lucj_idx = next(i for i, op in enumerate(pool) if op.name.startswith("LUCJ"))
    return pool, [lucj_idx]


def build_circuit(pool, seq):
    from qiskit import QuantumCircuit

    norb = pool.norb
    n_qubits = 2 * norb
    qc = QuantumCircuit(n_qubits)
    skipped = []
    for idx in seq:
        op = pool[int(idx)]
        if op.name == "I":
            continue
        if op.qiskit_gate_fn is None:
            skipped.append(op.name)
            continue
        gate = op.qiskit_gate_fn(norb)
        qc.append(gate, range(n_qubits))
    return qc, skipped


def transpile_and_count(qc, backend, optimization_level: int):
    from qiskit import transpile

    tqc = transpile(qc, backend=backend, optimization_level=optimization_level, seed_transpiler=0)
    ops = tqc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cz", "cx", "ecr", "rzz"))
    return {
        "depth": int(tqc.depth()),
        "total_gates": int(sum(ops.values())),
        "two_qubit_gates": int(two_q),
        "op_breakdown": dict(ops),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--molecules", nargs="+", required=True)
    ap.add_argument("--backend-name", default="ibm_miami")
    ap.add_argument("--fake-backend", action="store_true",
                     help="use qiskit_ibm_runtime.fake_provider backend instead of real hardware")
    ap.add_argument("--optimization-level", type=int, default=1)
    ap.add_argument("--num-samples", type=int, default=10)
    args = ap.parse_args()

    if args.fake_backend:
        import qiskit_ibm_runtime.fake_provider as fake_provider
        backend = getattr(fake_provider, args.backend_name)()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["IBM_API_KEY"],
            instance=os.environ["IBM_CRN"],
        )
        backend = service.backend(args.backend_name)

    for molecule_yaml in args.molecules:
        name = Path(molecule_yaml).stem
        print(f"\n=== {name} ===")

        trained_pool, trained_seqs = zero_shot_all_seqs(molecule_yaml, args.checkpoint, args.num_samples)
        all_skipped = []
        per_sample_metrics = []
        for seq in trained_seqs:
            qc, skipped = build_circuit(trained_pool, seq)
            all_skipped.extend(skipped)
            per_sample_metrics.append(transpile_and_count(qc, backend, args.optimization_level))
        if all_skipped:
            print(f"  WARNING: {len(all_skipped)} ops skipped (no qiskit_gate_fn): {sorted(set(all_skipped))}")

        n = len(per_sample_metrics)
        avg_depth = sum(m["depth"] for m in per_sample_metrics) / n
        avg_total = sum(m["total_gates"] for m in per_sample_metrics) / n
        avg_2q = sum(m["two_qubit_gates"] for m in per_sample_metrics) / n
        min_2q = min(m["two_qubit_gates"] for m in per_sample_metrics)
        max_2q = max(m["two_qubit_gates"] for m in per_sample_metrics)

        lucj1_pool, lucj1_seq = lucj1_pool_and_seq(molecule_yaml)
        lucj1_qc, _ = build_circuit(lucj1_pool, lucj1_seq)
        lucj1_metrics = transpile_and_count(lucj1_qc, backend, args.optimization_level)

        print(f"  trained (avg over {n} samples, 10 ops each): depth={avg_depth:.0f}, "
              f"total_gates={avg_total:.0f}, 2q_gates={avg_2q:.0f} (range {min_2q}-{max_2q})")
        print(f"  LUCJ(reps=1):        depth={lucj1_metrics['depth']}, "
              f"total_gates={lucj1_metrics['total_gates']}, "
              f"2q_gates={lucj1_metrics['two_qubit_gates']}")
        depth_ratio = avg_depth / max(lucj1_metrics["depth"], 1)
        cx_ratio = avg_2q / max(lucj1_metrics["two_qubit_gates"], 1)
        print(f"  ratio (trained_avg/lucj1): depth={depth_ratio:.2f}x, 2q_gates={cx_ratio:.2f}x")


if __name__ == "__main__":
    main()
