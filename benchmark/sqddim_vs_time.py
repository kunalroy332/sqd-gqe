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

"""SQD-dimension vs diagonalization time, for different molecule sizes.

For each molecule (spanning 16q..28q), sample once, then time the SQD
diagonalization (solve_sci) on the sampled subspace truncated to a range of
max_dim (SQD dimension) values. Reveals how solve time scales with the SQD
subspace size, and how that scaling steepens with qubit count.

Output: experiments/amortized/sqddim_vs_time.json + docs/figures/sqddim_vs_time.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"


def measure_molecule(molecule_yaml, max_dims, shots, checkpoint=None):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.core.configuration import ConfigurationSet
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.hw.backends import get_backend

    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    # use the trained (conditioned) policy so the sampled subspace is realistic;
    # a fresh policy at 24q+ collapses to ~1 determinant (nothing to diagonalize)
    ov, sd, vocab = [], None, None
    if checkpoint and Path(checkpoint).exists():
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        vocab = int(sd["transformer.wte.weight"].shape[0])
        ov = ["model=conditioned", f"+operator_pool.max_vocab={vocab}"]
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=ov)
    cfg.molecule = OmegaConf.load(molecule_yaml)
    cfg.sampler.shots = shots

    f = SqdFactory()
    system = f.build_system(cfg)
    policy = f.build_policy(cfg)
    schedule = f.build_beta_schedule(cfg)
    pipeline = f.build_pipeline(cfg)
    if sd is not None:
        cur = policy.state_dict()
        policy.load_state_dict({k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}, strict=False)
        if hasattr(policy, "set_condition"):
            policy.set_condition(ConditioningContext.build(system, get_backend("ibm_heron_r2")).tensor("cpu"))

    # one rollout to get a realistic sampled subspace
    state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long)}
    for _ in range(cfg.ngates):
        nxt = policy.act(state, schedule.beta())
        state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
    counts = pipeline.sampler.sample_state(state["idx"][0].tolist())
    full = ConfigurationSet.from_sample_counts(counts).select_by_nelec(system.nelec).expand(max(max_dims))
    avail = full.size
    name = Path(molecule_yaml).stem
    q = system.norb * 2

    pts = []
    for md in max_dims:
        if md > avail:
            continue
        sub = ConfigurationSet(full.configs[:md])
        t = time.perf_counter()
        _ = pipeline._sqd.diagonalize(sub)
        dt = time.perf_counter() - t
        pts.append({"max_dim": md, "diag_s": dt})
        print(f"[SQDDIM] {name:6s} q={q} max_dim={md:>7d} diag={dt:8.2f}s", flush=True)
    return {"molecule": name, "qubits": q, "avail_dets": avail, "points": pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--max-dims", nargs="+", type=int,
                    default=[500, 2000, 8000, 20000, 50000, 100000])
    ap.add_argument("--molecules", nargs="+", required=True)
    ap.add_argument("--checkpoint", default=None, help="trained policy (for a realistic sampled subspace)")
    args = ap.parse_args()
    EXP.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    results = []
    for m in args.molecules:
        try:
            results.append(measure_molecule(m, args.max_dims, args.shots, checkpoint=args.checkpoint))
        except Exception as e:
            print(f"[SQDDIM] {Path(m).stem}: FAIL {type(e).__name__}: {e}", flush=True)
        (EXP / "sqddim_vs_time.json").write_text(json.dumps(results, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C78A8", "#0F62FE", "#F58518", "#E45756", "#B279A2"]
    for i, r in enumerate(sorted(results, key=lambda x: x["qubits"])):
        pts = r["points"]
        if not pts:
            continue
        ax.plot([p["max_dim"] for p in pts], [p["diag_s"] for p in pts], "o-",
                color=colors[i % len(colors)], label=f"{r['molecule']} ({r['qubits']}q)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SQD dimension (max_dim = # determinants)")
    ax.set_ylabel("SQD diagonalization time (s, log)")
    ax.set_title("SQD-dimension vs time, per molecule size")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "sqddim_vs_time.png", dpi=130); plt.close(fig)
    print(f"wrote {FIG/'sqddim_vs_time.png'}")


if __name__ == "__main__":
    main()
