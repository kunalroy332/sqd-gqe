# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Training/inference time vs molecule size (qubit count / active-space dim).

For a range of molecules spanning 4-28 qubits, measures the wall-clock of the
core operations at their natural size:
  * CASCI reference (one-off setup cost),
  * one rollout = sample + SQD diagonalize + GEVP refine (this IS the per-
    iteration training cost AND the per-call inference cost),
so we can plot time vs qubits and vs FCI-space dimension.

Output: experiments/amortized/time_vs_size.json + docs/figures/time_vs_size.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
EXP = Path(__file__).resolve().parent.parent / "experiments" / "amortized"


def measure(molecule_yaml):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory

    abs_cfg = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default")
    cfg.molecule = OmegaConf.load(molecule_yaml)

    f = SqdFactory()
    system = f.build_system(cfg)
    policy = f.build_policy(cfg)
    schedule = f.build_beta_schedule(cfg)
    pipeline = f.build_pipeline(cfg)

    norb = system.norb
    na, nb = system.nelec
    fci_dim = math.comb(norb, na) * math.comb(norb, nb)

    from sqd_gqe.core.system import FciIntractableError
    t = time.perf_counter()
    try:
        _ = system.casci_energy()
        ref_used = "casci"
    except FciIntractableError:
        _ = system.ccsd_t_energy()   # still tractable; CASCI is not (see FciIntractableError)
        ref_used = "ccsd_t"
    t_casci = time.perf_counter() - t

    # one rollout (= one training iteration's forward cost = one inference call)
    state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long)}
    for _ in range(cfg.ngates):
        nxt = policy.act(state, schedule.beta())
        state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
    t = time.perf_counter(); _ = pipeline.run(state); t_rollout = time.perf_counter() - t
    tm = dict(pipeline.timings)

    return {"molecule": Path(molecule_yaml).stem, "qubits": norb * 2, "fci_dim": int(fci_dim),
            "t_casci": t_casci, "ref_used": ref_used, "t_rollout": t_rollout,
            "t_sample": tm.get("sample", 0), "t_diag": tm.get("diagonalize", 0),
            "t_refine": tm.get("refine_local", 0) + tm.get("refine_global", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", required=True)
    args = ap.parse_args()
    EXP.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in args.molecules:
        try:
            r = measure(m); rows.append(r)
            print(f"[TIME] {r['molecule']:8s} q={r['qubits']:2d} fci={r['fci_dim']:>9d} "
                  f"CASCI={r['t_casci']:7.2f}s rollout={r['t_rollout']:7.2f}s "
                  f"(sample {r['t_sample']:.2f}, diag {r['t_diag']:.2f}, refine {r['t_refine']:.2f})", flush=True)
        except Exception as e:
            print(f"[TIME] {Path(m).stem}: FAIL {type(e).__name__}: {e}", flush=True)
        (EXP / "time_vs_size.json").write_text(json.dumps(rows, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = sorted(rows, key=lambda x: x["qubits"])
    q = [x["qubits"] for x in r]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot(q, [x["t_rollout"] for x in r], "o-", color="#0F62FE", label="rollout (train iter = inference call)")
    a1.plot(q, [x["t_diag"] for x in r], "s--", color="#E45756", label="SQD diagonalize")
    a1.plot(q, [x["t_casci"] for x in r], "^:", color="#8A8A8A", label="CASCI reference")
    a1.set_yscale("log"); a1.set_xlabel("qubits (2 x active orbitals)"); a1.set_ylabel("wall-clock (s, log)")
    a1.set_title("Time vs qubit count"); a1.legend(fontsize=8); a1.grid(alpha=.3)
    for x in r:
        a1.annotate(x["molecule"], (x["qubits"], x["t_rollout"]), fontsize=6, xytext=(2, 2), textcoords="offset points")
    a2.plot([x["fci_dim"] for x in r], [x["t_rollout"] for x in r], "o-", color="#0F62FE")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("FCI subspace dimension"); a2.set_ylabel("rollout wall-clock (s, log)")
    a2.set_title("Time vs FCI-space dimension"); a2.grid(alpha=.3)
    fig.suptitle("Training / inference time vs molecule size (ffsim + SQD, GB200)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "time_vs_size.png", dpi=130); plt.close(fig)
    print(f"wrote {FIG/'time_vs_size.png'}")


if __name__ == "__main__":
    main()
