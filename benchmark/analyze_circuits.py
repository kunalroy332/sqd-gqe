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

"""Compute circuit-analysis metrics for a config and merge into a training log.

Runs the ffsim circuit-analysis metrics (excitation-rank distribution,
expressibility KL vs Haar, chemistry suitability) once for a given config and
writes them into the run's JSON under an "analysis" key, so compare_report.py
can render them. Kept separate from training so it can be re-run cheaply.

Usage:
    python benchmark/analyze_circuits.py --json docs/report/ffsim_16q.json [--overrides ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="training log to enrich in place")
    ap.add_argument("--config", default="default")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--n-express", type=int, default=120)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    abs_dir = str(Path(__file__).resolve().parent.parent / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_dir):
        cfg = compose(config_name=args.config, overrides=list(args.overrides))

    import ffsim
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.gqe import circuit_metrics as cm

    f = SqdFactory()
    system = f.build_system(cfg)
    pool = f.build_operator_pool(cfg)
    policy = f.build_policy(cfg)
    pipeline = f.build_pipeline(cfg)
    ngates = cfg.ngates
    dim = ffsim.dim(system.norb, system.nelec)

    analysis = {"fci_dim": int(dim)}

    # expressibility: uniform-random circuits vs Haar
    state_fn = cm.make_state_fn_from_pool(pool, ngates)
    analysis["expressibility"] = cm.expressibility_kl(state_fn, dim, n_states=args.n_express, n_bins=60, seed=0)

    # chemistry suitability of a representative sampled circuit vs CCSD reference
    rng = np.random.default_rng(0)
    seq = list(rng.integers(0, len(pool), size=ngates))

    # transpiled circuit metrics (depth / 2-qubit gates / total) for that circuit
    try:
        analysis["circuit_depth"] = cm.circuit_depth_metrics(pool, seq)
    except Exception as e:
        analysis["circuit_depth"] = {"error": f"{type(e).__name__}: {e}"}
    ref = cm.ccsd_reference_state(system, pool)
    ham = ffsim.MolecularHamiltonian(
        system.active_hamiltonian.one_body,
        system.active_hamiltonian.two_body,
        constant=float(system.active_hamiltonian.core_energy),
    )
    analysis["chemistry_suitability"] = cm.chemistry_suitability(pool, seq, reference_state=ref, hamiltonian=ham)

    # excitation-rank distribution of the accumulated global-refined state
    state = {"idx": torch.zeros((cfg.trainer.num_samples, 1), dtype=torch.long)}
    for _ in range(ngates):
        nxt = policy.act(state, f.build_beta_schedule(cfg).beta())
        state["idx"] = torch.cat((state["idx"], nxt.unsqueeze(1)), dim=1)
    _ = pipeline.run(state)
    _ = pipeline.run(state)  # second call so a global-refined state exists
    merged = pipeline.global_refined_state
    if merged is not None and getattr(merged, "strings", None) is not None:
        w = np.abs(np.asarray(merged)) ** 2
        analysis["excitation_distribution"] = cm.excitation_rank_histogram(merged.strings, system.nelec, weights=w)

    data = json.loads(Path(args.json).read_text())
    data["analysis"] = analysis
    Path(args.json).write_text(json.dumps(data, indent=2))
    print(f"enriched {args.json} with analysis:")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
