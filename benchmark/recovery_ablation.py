# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Does iterative configuration-recovery actually help, or is the trained
policy's raw sample quality already enough? An ablation at FIXED shots and
FIXED SQD dimension across recovery depth: 0 (no recovery -- just the plain
expand+diagonalize path), 2, and 5 max_iterations of
qiskit-addon-sqd's diagonalize_fermionic_hamiltonian (recover -> subsample ->
diagonalize, repeated).

This isolates the bitstring-quality question from the dimension question:
same circuit, same samples, same max_dim throughout -- only recovery depth
varies. If 0-recovery is already close to 5-recovery, the policy's sampled
configurations are already well-formed (little to repair). If there's a
large gap, recovery is doing real work fixing symmetry-broken/off-target
samples.

Output: experiments/amortized/recovery_ablation.json + .png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"
CHEM_ACC_HA = 1.6e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--molecules", nargs="+", required=True)
    ap.add_argument("--backend", default="ibm_heron_r2")
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--max-dim", type=int, default=8_000)
    ap.add_argument("--recovery-iters", nargs="+", type=int, default=[0, 2, 5])
    ap.add_argument("--out", default=str(EXP / "recovery_ablation.json"))
    args = ap.parse_args()

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.core.system import FciIntractableError

    EXP.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck)
    vocab = int(sd["transformer.wte.weight"].shape[0])

    rows = []
    for mol in args.molecules:
        with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
            cfg = compose(config_name="default", overrides=["model=conditioned", f"+operator_pool.max_vocab={vocab}"])
        cfg.molecule = OmegaConf.load(mol)
        cfg.sqd.max_dim = args.max_dim
        cfg.sampler.shots = args.shots

        f = SqdFactory()
        system = f.build_system(cfg)
        hw = get_backend(args.backend)
        policy = f.build_policy(cfg).to(device)
        pipeline = f.build_pipeline(cfg, backend=hw)
        schedule = f.build_beta_schedule(cfg)
        pool = f.build_operator_pool(cfg)
        try:
            ref, ref_label = float(system.casci_energy()), "casci"
        except FciIntractableError:
            ref, ref_label = float(system.ccsd_t_energy()), "ccsd_t"

        cur = policy.state_dict()
        policy.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)
        if hasattr(policy, "set_condition"):
            policy.set_condition(ConditioningContext.build(system, hw).tensor(device))

        # one zero-shot rollout: pick the best circuit's samples, fixed shots/dim
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
        best_i = int(__import__("numpy").argmin(report.energies))
        best_seq = state["idx"][best_i].tolist()
        counts = pipeline.sampler.sample_state(best_seq)

        mol_name = Path(mol).stem
        for n_iter in args.recovery_iters:
            if n_iter == 0:
                from sqd_gqe.core.configuration import ConfigurationSet
                valid = ConfigurationSet.from_sample_counts(counts).select_by_nelec(pipeline.nelec)
                energy, _ = pipeline.diagonalize(valid.expand(max_size=args.max_dim, method=pipeline.expand_method))
            else:
                energy, _, _hist = pipeline._sqd.recover_and_diagonalize(
                    counts, samples_per_batch=min(512, args.shots), num_batches=3,
                    max_iterations=n_iter, seed=0)
            err = (energy - ref) * 1e3
            rows.append({"molecule": mol_name, "recovery_iters": n_iter, "energy": float(energy),
                        "ref": ref, "ref_label": ref_label, "err_mha": float(err),
                        "max_dim": args.max_dim, "shots": args.shots})
            print(f"[RECOVERY-ABLATION] {mol_name} recovery_iters={n_iter:2d} "
                  f"E={energy:.6f} err={err:+.3f}mHa (ref={ref_label})", flush=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({r["molecule"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = plt.cm.tab10.colors
    for i, n in enumerate(names):
        rs = sorted([r for r in rows if r["molecule"] == n], key=lambda r: r["recovery_iters"])
        ax.plot([r["recovery_iters"] for r in rs], [max(abs(r["err_mha"]), 1e-4) for r in rs],
                "o-", color=colors[i % 10], label=n)
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy")
    ax.set_yscale("log")
    ax.set_xlabel("config-recovery max_iterations (0 = raw diagonalize, no recovery)")
    ax.set_ylabel("|error| vs reference (mHa)")
    ax.set_title(f"Recovery-depth ablation (fixed shots={rows[0]['shots'] if rows else '?'}, "
                 f"max_dim={rows[0]['max_dim'] if rows else '?'})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(EXP / "recovery_ablation.png", dpi=130)
    plt.close(fig)
    print(f"wrote {EXP/'recovery_ablation.png'}")


if __name__ == "__main__":
    main()
