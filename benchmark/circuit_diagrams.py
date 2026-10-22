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

"""Draw example circuits: the standalone LUCJ pool op, and the GPT-generated
best circuit, both ideal and transpiled to a real backend's native gates +
connectivity. For a 32q/40q circuit the FULL transpiled depth (thousands of
gates) isn't something a static image can show usefully -- each diagram is
explicitly a truncated slice (first --max-gates-shown gates), labeled with
the true total depth/2q-count so the truncation is never mistaken for the
whole picture.

Output: docs/figures/circuit_<molecule>_<lucj|gpt>_<ideal|transpiled>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"


def _draw(qc, title, path, max_gates_shown=40):
    """Draw a (possibly truncated) slice of qc, labeled with the true depth."""
    import matplotlib
    matplotlib.use("Agg")
    from qiskit import QuantumCircuit

    total_depth = qc.depth()
    total_ops = sum(qc.count_ops().values())
    shown = qc
    truncated = False
    if total_ops > max_gates_shown:
        shown = QuantumCircuit(qc.num_qubits)
        for instr in qc.data[:max_gates_shown]:
            shown.append(instr.operation, instr.qubits, instr.clbits)
        truncated = True

    suffix = f" -- first {max_gates_shown} of {total_ops} gates (depth {total_depth})" if truncated \
        else f" -- {total_ops} gates, depth {total_depth}"
    fig = shown.draw(output="mpl", fold=-1, idle_wires=False, style={"fontsize": 7})
    fig.suptitle(title + suffix, fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"wrote {path} ({'truncated, ' if truncated else ''}true depth={total_depth}, "
          f"true gates={total_ops})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="ibm_heron_r2")
    ap.add_argument("--max-gates-shown", type=int, default=40)
    ap.add_argument("--select-dim", type=int, default=100_000,
                    help="SQD dim used to pick the best-energy sample from the batch -- must be "
                         "large enough to actually resolve differences between samples, not a tiny "
                         "value that can't distinguish a good circuit from a padded one")
    ap.add_argument("--tag", default=None, help="filename tag, defaults to molecule stem")
    args = ap.parse_args()

    import torch
    from qiskit import QuantumCircuit
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.core.configuration import ConfigurationSet

    FIG.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.molecule).stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck)
    vocab = int(sd["transformer.wte.weight"].shape[0])

    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=["model=conditioned", f"+operator_pool.max_vocab={vocab}"])
    cfg.molecule = OmegaConf.load(args.molecule)

    f = SqdFactory()
    system = f.build_system(cfg)
    hw = get_backend(args.backend)
    policy = f.build_policy(cfg).to(device)
    schedule = f.build_beta_schedule(cfg)
    pool = f.build_operator_pool(cfg)
    pipeline = f.build_pipeline(cfg, backend=hw)
    cur = policy.state_dict()
    policy.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)
    if hasattr(policy, "set_condition"):
        policy.set_condition(ConditioningContext.build(system, hw).tensor(device))

    norb = pool.norb
    n_qubits = 2 * norb

    # 1) the standalone LUCJ pool op (find it by name -- see operator_pool.py)
    lucj_idx = next((i for i, op in enumerate(pool.pool) if op.name.startswith("LUCJ(")), None)
    if lucj_idx is not None:
        op = pool.pool[lucj_idx]
        qc_ideal = QuantumCircuit(n_qubits)
        qc_ideal.append(op.qiskit_gate_fn(norb), range(n_qubits))
        _draw(qc_ideal, f"{tag} ({n_qubits}q) -- {op.name}, ideal gates",
              FIG / f"circuit_{tag}_lucj_ideal.png", args.max_gates_shown)
        qc_t = hw.transpile(qc_ideal)
        _draw(qc_t, f"{tag} ({n_qubits}q) -- {op.name}, transpiled to {args.backend}",
              FIG / f"circuit_{tag}_lucj_transpiled.png", args.max_gates_shown)
    else:
        print(f"[WARN] no LUCJ(...) op found in {tag}'s pool -- skipping")

    # 2) the GPT-generated best circuit -- select the LOWEST-small-dim-energy
    # circuit across the sampled batch (same criterion as
    # deep_dive_analysis.py), not just sample index 0: GRPO sampling has
    # per-rollout variance, and an arbitrary sample can be a degenerate
    # local-primitives-only rollout with no chemistry-seed block at all --
    # not representative of what the policy typically produces.
    state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
    with torch.no_grad():
        for _ in range(cfg.ngates):
            nxt = policy.act(state, schedule.beta())
            state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
        sampled_counts = pipeline.sampler.run(state)
    best_seq, best_e0 = None, float("inf")
    for seq_i, counts in zip(state["idx"][:, 1:].tolist(), sampled_counts):
        valid = ConfigurationSet.from_sample_counts(counts).select_by_nelec(pipeline.nelec)
        if valid.size == 0:
            continue
        e0, _ = pipeline.diagonalize(valid.expand(max_size=args.select_dim, method=pipeline.expand_method))
        if e0 < best_e0:
            best_e0, best_seq = e0, seq_i
    seq = best_seq if best_seq is not None else state["idx"][0, 1:].tolist()
    qc_gpt = QuantumCircuit(n_qubits)
    est_ops = []
    for idx in seq:
        op = pool[int(idx)]
        if getattr(op, "qiskit_gate_fn", None) is not None:
            qc_gpt.append(op.qiskit_gate_fn(norb), range(n_qubits))
        elif op.name != "I":
            est_ops.append(op.name)
    if est_ops:
        print(f"[NOTE] {len(est_ops)} selected op(s) have no qiskit_gate_fn (not drawn): {est_ops}")
    _draw(qc_gpt, f"{tag} ({n_qubits}q) -- GPT-generated circuit, ideal gates",
          FIG / f"circuit_{tag}_gpt_ideal.png", args.max_gates_shown)
    qc_gpt_t = hw.transpile(qc_gpt)
    _draw(qc_gpt_t, f"{tag} ({n_qubits}q) -- GPT-generated circuit, transpiled to {args.backend}",
          FIG / f"circuit_{tag}_gpt_transpiled.png", args.max_gates_shown)


if __name__ == "__main__":
    main()
