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

"""Build the labeled dataset ResourceHead is trained on: for each
(molecule, backend) pair, the smallest max_dim that reaches chemical accuracy.

For each molecule x backend x shots option, generates one zero-shot circuit
from a trained checkpoint, then grows the SQD subspace with SqdPipeline's
adaptive doubling loop (SAME samples reused across dims -- no re-sampling per
dim, that would be the co2_dim_fix.py-style inefficiency this avoids). The
loop's own history already gives us (dim, energy) at every size tried, so we
just scan that history for the first dim crossing chemical accuracy against
CASCI (or CCSD(T) where CASCI is intractable) -- one pipeline.solve_adaptive()
call per (molecule, backend, shots), not one call per dim.

Looping over --backends (not just one) is what actually teaches ResourceHead
noise-sensitivity: molecule_features(system, backend) already encodes
err_1q/err_2q/err_readout + a sim_kind one-hot, but a dataset built against a
single backend never shows the model an example where more noise needed a
bigger subspace. Only backends whose sim_kind routes through
NoisyFfsimSampler's gate+readout proxy (the "*_noisy_sim" variants, plus
"ideal" as the zero-noise anchor) actually vary the sampled bitstring quality
in this codebase -- plain "real" backends inject no simulated noise here, so
they'd just add label noise with no signal; pick --backends accordingly.

Across the shots options for one (molecule, backend), we keep the smallest
shots for which SOME dim still reaches accuracy (more shots never hurts
accuracy, only cost) and its associated minimal dim. That (log10 dim,
log10 shots) pair is the label. See core/conditioning.py's ResourceHead and
train_resource_head.py.

Output: experiments/amortized/resource_dataset.json (or --out, for sharding
across multiple concurrent jobs -- see benchmark/merge_resource_shards.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"
CHEM_ACC_HA = 1.6e-3


def _load(molecule_yaml, checkpoint_sd, vocab, backend_name, shots, start_dim, device):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.core.conditioning import ConditioningContext
    from sqd_gqe.core.system import FciIntractableError

    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=["model=conditioned", f"+operator_pool.max_vocab={vocab}"])
    cfg.molecule = OmegaConf.load(molecule_yaml)
    cfg.sampler.shots = shots
    cfg.sqd.max_dim = start_dim  # unused once we call solve_adaptive directly

    f = SqdFactory()
    system = f.build_system(cfg)
    hw = get_backend(backend_name)
    policy = f.build_policy(cfg).to(device)
    pipeline = f.build_pipeline(cfg, backend=hw)
    schedule = f.build_beta_schedule(cfg)
    cur = policy.state_dict()
    policy.load_state_dict({k: v for k, v in checkpoint_sd.items() if k in cur and cur[k].shape == v.shape},
                          strict=False)
    if hasattr(policy, "set_condition"):
        policy.set_condition(ConditioningContext.build(system, hw).tensor(device))
    try:
        ref, ref_label = float(system.casci_energy()), "casci"
    except FciIntractableError:
        ref, ref_label = float(system.ccsd_t_energy()), "ccsd_t"
    return system, hw, policy, pipeline, schedule, cfg, ref, ref_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--molecules", nargs="+", required=True)
    ap.add_argument("--backends", nargs="+", default=["ideal"],
                    help="e.g. ideal ibm_heron_r2_noisy_sim ibm_heron_r3_noisy_sim "
                         "ibm_nighthawk_r1_noisy_sim -- only *_noisy_sim/ideal backends "
                         "actually vary sampled bitstring quality in this codebase")
    ap.add_argument("--shots-options", nargs="+", type=int, default=[10_000])
    ap.add_argument("--start-dim", type=int, default=2_000)
    ap.add_argument("--growth", type=float, default=2.0)
    ap.add_argument("--max-dim-cap", type=int, default=100_000)
    ap.add_argument("--out", default=str(EXP / "resource_dataset.json"))
    args = ap.parse_args()

    import torch
    from sqd_gqe.core.configuration import ConfigurationSet
    from sqd_gqe.core.conditioning import molecule_features

    EXP.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck)
    vocab = int(sd["transformer.wte.weight"].shape[0])

    rows = []
    for mol in args.molecules:
        for backend_name in args.backends:
            per_shots = []
            for shots in args.shots_options:
                system, hw, policy, pipeline, schedule, cfg, ref, ref_label = _load(
                    mol, sd, vocab, backend_name, shots, args.start_dim, device)

                state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long, device=device)}
                with torch.no_grad():
                    for _ in range(cfg.ngates):
                        nxt = policy.act(state, schedule.beta())
                        state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
                    sampled_counts = pipeline.sampler.run(state)

                # use the circuit whose small-dim energy is lowest as the "best" circuit
                best_valid, best_energy0 = None, float("inf")
                for counts in sampled_counts:
                    valid = ConfigurationSet.from_sample_counts(counts).select_by_nelec(pipeline.nelec)
                    if valid.size == 0:
                        continue
                    e0, _ = pipeline.diagonalize(valid.expand(max_size=args.start_dim, method=pipeline.expand_method))
                    if e0 < best_energy0:
                        best_energy0, best_valid = e0, valid
                if best_valid is None:
                    print(f"[RESOURCE-DATA] {mol} backend={backend_name} shots={shots}: "
                          f"no valid samples, skipping")
                    continue

                _, _, dim_reached, history = pipeline.solve_adaptive(
                    best_valid, args.start_dim, args.growth, args.max_dim_cap, energy_tol_ha=CHEM_ACC_HA / 10)
                min_dim_hit = next((h["dim"] for h in history if abs(h["energy"] - ref) <= CHEM_ACC_HA), None)
                per_shots.append({"shots": shots, "min_dim_hit": min_dim_hit, "dim_reached": dim_reached,
                                  "history": history, "ref": ref, "ref_label": ref_label})
                print(f"[RESOURCE-DATA] {mol} backend={backend_name} shots={shots} "
                      f"min_dim_hit={min_dim_hit} (reached {dim_reached}, ref={ref_label})", flush=True)

            feasible = [p for p in per_shots if p["min_dim_hit"] is not None]
            chosen = min(feasible, key=lambda p: p["shots"]) if feasible else (
                max(per_shots, key=lambda p: p["shots"]) if per_shots else None)
            if chosen is None:
                continue
            feats = molecule_features(system, hw).tolist()
            rows.append({
                "molecule": Path(mol).stem,
                "backend": backend_name,
                "sim_kind": getattr(hw, "sim_kind", None),
                "features": feats,
                "label_dim": chosen["min_dim_hit"] or chosen["dim_reached"],
                "label_shots": chosen["shots"],
                "reachable": chosen["min_dim_hit"] is not None,
                "ref_label": chosen["ref_label"],
                "sweep": per_shots,
            })
            Path(args.out).write_text(json.dumps(rows, indent=2))

    print(f"wrote {args.out} ({len(rows)} rows over {len(args.molecules)} molecules "
          f"x {len(args.backends)} backends)")


if __name__ == "__main__":
    main()
