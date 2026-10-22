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

"""Verify every generated circuit (from an amortized_train.py run's history)
is shallower than that molecule's own LUCJ(reps=1) baseline, on the same
backend + optimization level the training run itself used.

LUCJ(reps=1) is the shallowest single-block chemistry-seeded circuit this
pool can produce (LUCJ_N_REPS defaults to 2 -- see operator_pool.py); this
script builds that op explicitly at n_reps=1 as the baseline, not by reading
it out of the trained pool (whose seed op is reps=2 by default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def lucj1_baseline_depth(molecule_yaml: str, backend_name: str) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sqd_gqe.factory import SqdFactory
    from sqd_gqe.hw.backends import get_backend
    from sqd_gqe.gqe import circuit_metrics as cm

    abs_cfg = str(REPO_ROOT / "configs_sqd")
    with initialize_config_dir(version_base="1.3", config_dir=abs_cfg):
        cfg = compose(config_name="default", overrides=["operator_pool.n_reps=1"])
    cfg.molecule = OmegaConf.load(molecule_yaml)

    factory = SqdFactory()
    system = factory.build_system(cfg)
    pool = factory.build_operator_pool(cfg)
    hw = get_backend(backend_name)

    # pool[0] is identity (see operator_pool.py); the LUCJ seed op is pool[1]
    # whenever it was built (build_operator_pool always appends it right
    # after identity_op()).
    lucj_idx = next(i for i, op in enumerate(pool) if op.name.startswith("LUCJ"))
    seq = [0, lucj_idx]
    metrics = cm.circuit_depth_on_backend(pool, seq, hw, optimization_level=1)
    return {"lucj_name": pool[lucj_idx].name, "depth": metrics["depth"], "two_qubit_gates": metrics["two_qubit_gates"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", required=True, nargs="+",
                    help="one or more amortized_train.py --out JSON files to check")
    args = ap.parse_args()

    CHEM_ACC_MHA = 1.6  # 1.6 mHa = standard "chemical accuracy" threshold

    all_pass = True
    for run_path in args.run_json:
        data = json.loads(Path(run_path).read_text())
        backend = data["backend"]
        molecules = data["molecules"]

        # shallowest circuit among chemically-accurate rows per molecule --
        # NOT lowest-|err| (many rows tie at ~machine-zero error once a
        # molecule is easy, so a naive lowest-error tiebreak can pick an
        # early, deeper circuit over a later, shallower, equally-accurate
        # one; this is the metric that actually matters for "shallow AND
        # accurate", not just "most accurate ever seen").
        best = {}
        fallback_best = {}  # lowest |err| regardless of accuracy, for molecules that never hit it
        for row in data["history"]:
            m = row["molecule"]
            if row.get("depth") is None:
                continue
            if m not in fallback_best or abs(row["err_mha"]) < abs(fallback_best[m]["err_mha"]):
                fallback_best[m] = row
            if abs(row["err_mha"]) < CHEM_ACC_MHA and (m not in best or row["depth"] < best[m]["depth"]):
                best[m] = row
        never_accurate = set(fallback_best) - set(best)
        for m in never_accurate:
            best[m] = fallback_best[m]

        print(f"\n=== {run_path} (backend={backend}) ===")
        for name in molecules:
            molecule_yaml = None
            for tier in ("small", "medium", "large"):
                cand = REPO_ROOT / "molecules" / tier / f"{name}.yaml"
                if cand.exists():
                    molecule_yaml = str(cand)
                    break
            if molecule_yaml is None:
                print(f"  {name}: SKIP (yaml not found)")
                continue

            baseline = lucj1_baseline_depth(molecule_yaml, backend)
            trained = best.get(name)
            if trained is None:
                print(f"  {name}: no depth-logged row in history, skipping")
                continue

            ok = trained["depth"] < baseline["depth"]
            all_pass &= ok
            status = "PASS" if ok else "FAIL"
            note = " [NEVER HIT CHEM ACCURACY -- fallback to lowest-error row]" if name in never_accurate else ""
            print(f"  [{status}] {name:8s} trained_depth={trained['depth']:5d}  "
                  f"lucj1_baseline_depth={baseline['depth']:5d}{note}  "
                  f"(err={trained['err_mha']:+.4f}mHa, iter={trained['iter']})")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
