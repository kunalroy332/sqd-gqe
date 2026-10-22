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

"""Audit which operator-pool families a trained checkpoint actually selects at
zero-shot inference time, for the given molecules.

Only LUCJ-seed / orbital-rotation / diag-Coulomb ops have a ``qiskit_gate_fn``
today (see sqd_gqe/gqe/operator_pool.py); Givens and number-number ops do not,
so any hardware sampler cannot build a real circuit for a sequence that uses
them. This script generates circuits the same way zero-shot inference does
(policy.act loop, no diagonalization needed) and reports the op-family usage
histogram, so we know whether closing that qiskit_gate_fn gap is required
before real/fake hardware sampling can run the actual trained checkpoint.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch


def op_family(name: str) -> str:
    # op names look like "LUCJ", "orb_rot[3]", "givens(2,3;0.500)", "diag_coulomb(t=0.300)", "num_num(1,2)"
    return re.split(r"[\[\(]", name, maxsplit=1)[0]


def audit_molecule(molecule_path: str, checkpoint: str, seed: int = 0, num_samples: int = 10):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_state = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt_state.get("state_dict", ckpt_state)
    wte = sd.get("transformer.wte.weight")
    ckpt_vocab = int(wte.shape[0]) if wte is not None else None
    overrides = ["model=conditioned"]
    if ckpt_vocab:
        overrides.append(f"+operator_pool.max_vocab={ckpt_vocab}")

    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=overrides)
    cfg.molecule = OmegaConf.load(molecule_path)
    cfg.trainer.num_samples = num_samples
    cfg.trainer.seed = seed

    factory = SqdFactory()
    system = factory.build_system(cfg)
    policy = factory.build_policy(cfg).to(device)
    schedule = factory.build_beta_schedule(cfg)
    pool = factory.build_operator_pool(cfg)

    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext

    hw = get_backend("ideal")
    cond = ConditioningContext.build(system, hw)
    if hasattr(policy, "set_condition"):
        policy.set_condition(cond.tensor(device))

    policy.load_state_dict(
        {k: v for k, v in sd.items() if k in policy.state_dict()
         and policy.state_dict()[k].shape == v.shape},
        strict=False,
    )

    state = {"idx": torch.zeros((num_samples, 1), dtype=torch.long, device=device)}
    with torch.no_grad():
        for _ in range(cfg.ngates):
            nxt = policy.act(state, schedule.beta())
            state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)

    idx_batch = state["idx"][:, 1:].cpu().tolist()  # drop BOS/idx-0 token
    family_counts: Counter = Counter()
    has_qiskit_gate_fn: dict[str, bool] = {}
    for row in idx_batch:
        for idx in row:
            op = pool[int(idx)]
            fam = op_family(op.name)
            family_counts[fam] += 1
            has_qiskit_gate_fn[fam] = op.qiskit_gate_fn is not None
    return family_counts, has_qiskit_gate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--molecules", nargs="+", required=True)
    ap.add_argument("--num-samples", type=int, default=10)
    args = ap.parse_args()

    overall_families: Counter = Counter()
    overall_coverage: dict[str, bool] = {}
    for m in args.molecules:
        name = Path(m).stem
        counts, coverage = audit_molecule(m, args.checkpoint, num_samples=args.num_samples)
        total = sum(counts.values())
        print(f"\n[{name}] {total} op selections across {args.num_samples} sampled circuits:")
        for fam, n in counts.most_common():
            covered = "OK" if coverage[fam] else "**NO qiskit_gate_fn**"
            print(f"  {fam:15s} {n:5d} ({100*n/total:5.1f}%)  {covered}")
        overall_families.update(counts)
        overall_coverage.update(coverage)

    uncovered = [fam for fam, ok in overall_coverage.items() if not ok]
    print("\n=== SUMMARY ===")
    if uncovered:
        n_uncovered = sum(overall_families[f] for f in uncovered)
        n_total = sum(overall_families.values())
        print(f"UNCOVERED op families selected: {uncovered} "
              f"({n_uncovered}/{n_total} = {100*n_uncovered/n_total:.1f}% of all selections)")
        print("qiskit_gate_fn gap MUST be closed before hardware sampling can run this checkpoint.")
    else:
        print("All selected op families have qiskit_gate_fn. No gap-close needed for these molecules.")


if __name__ == "__main__":
    main()
