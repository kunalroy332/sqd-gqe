# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Does bumping max_dim + shots recover CO2 (28q) accuracy zero-shot?

The generalist (trained <=24q) failed zero-shot on CO2 at 28q (+398 mHa) because
max_dim=2000 captured a vanishing fraction of the ~4M-dim FCI space. Here we
sweep max_dim (and shots) at 28q with the SAME checkpoint (still zero-shot) and
track error vs CASCI, to see whether a larger subspace crosses chemical accuracy.

Output: experiments/amortized/co2_dimfix.json + co2_dimfix.png
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
    ap.add_argument("--molecule", default="molecules/medium/co2.yaml")
    ap.add_argument("--backend", default="ibm_heron_r2")
    ap.add_argument("--max-dims", nargs="+", type=int, default=[2000, 8000, 20000, 50000, 100000])
    ap.add_argument("--shots", type=int, default=300000)
    args = ap.parse_args()

    import torch, numpy as np
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")

    # infer checkpoint vocab, build conditioned policy once
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck)
    vocab = int(sd["transformer.wte.weight"].shape[0])

    rows = []
    for md in args.max_dims:
        with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
            cfg = compose(config_name="default",
                          overrides=["model=conditioned", f"+operator_pool.max_vocab={vocab}"])
        cfg.molecule = OmegaConf.load(args.molecule)
        cfg.sqd.max_dim = md
        cfg.sampler.shots = args.shots

        f = SqdFactory()
        system = f.build_system(cfg)
        policy = f.build_policy(cfg).to(device)
        pipeline = f.build_pipeline(cfg)
        schedule = f.build_beta_schedule(cfg)
        pool = f.build_operator_pool(cfg)
        casci = float(system.casci_energy())
        cur = policy.state_dict()
        policy.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)
        if hasattr(policy, "set_condition"):
            policy.set_condition(ConditioningContext.build(system, get_backend(args.backend)).tensor(device))

        # single zero-shot rollout (no training)
        state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
        with torch.no_grad():
            for _ in range(cfg.ngates):
                nxt = policy.act(state, schedule.beta())
                state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
            report = pipeline.run(state)
        best = min(report.energies)
        if report.global_refined is not None:
            best = min(best, report.global_refined.energy)
        if report.local_refined is not None:
            best = min(best, report.local_refined.energy)
        err = (best - casci) * 1e3
        rows.append({"max_dim": md, "shots": args.shots, "energy": float(best),
                     "casci": casci, "err_mha": float(err)})
        print(f"[CO2FIX] max_dim={md:6d} shots={args.shots} E={best:.5f} CASCI={casci:.5f} err={err:+.3f} mHa", flush=True)
        (EXP / "co2_dimfix.json").write_text(json.dumps(rows, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([r["max_dim"] for r in rows], [max(abs(r["err_mha"]), 1e-4) for r in rows],
            "o-", color="#0F62FE")
    ax.axhline(CHEM_ACC_HA * 1e3, ls="--", color="gray", label="chemical accuracy")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SQD max_dim (subspace size)"); ax.set_ylabel("|error| vs CASCI (mHa)")
    ax.set_title(f"CO2 28q zero-shot: does bumping max_dim recover accuracy? (shots={args.shots})")
    ax.legend()
    fig.tight_layout(); fig.savefig(EXP / "co2_dimfix.png", dpi=130); plt.close(fig)
    print(f"wrote {EXP/'co2_dimfix.png'}")


if __name__ == "__main__":
    main()
