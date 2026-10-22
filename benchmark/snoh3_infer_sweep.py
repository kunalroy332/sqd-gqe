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

"""Zero-shot inference sweep for Sn(OH)3-Me: S/M/L/LL active spaces x several
fixed SQD subspace dimensions, using a checkpoint trained by
slurm/train_snoh3_tiers.sh.

Not a training script -- this calls sqd_gqe.run.run_molecule() directly
(--checkpoint given => zero-shot, no on-the-fly adaptation), once per
(molecule, max_dim) pair, and collects the reports into one JSON file for
later analysis (accuracy vs dim, per active-space size).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MOLECULES = {
    "S": "molecules/small/SnOH3-me_S_16o24e_mp2no_def2svp.yaml",
    "M": "molecules/medium/SnOH3-me_M_19o24e_mp2no_def2svp.yaml",
    "L": "molecules/large/SnOH3-me_L_23o32e_mp2no_def2svp.yaml",
    "LL": "molecules/large/SnOH3-me_LL_28o42e_mp2no_def2svp.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="policy weights from amortized_train.py")
    ap.add_argument("--spaces", nargs="+", default=list(MOLECULES.keys()),
                    help="which of S/M/L/LL to run (default: all four)")
    ap.add_argument("--dims", nargs="+", type=int, default=[500, 2_000, 10_000, 50_000],
                    help="fixed SQD subspace sizes to sweep per active space")
    ap.add_argument("--backend", default="ideal")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "amortized" / "snoh3_infer_sweep.json"))
    args = ap.parse_args()

    from sqd_gqe.run import run_molecule

    results = []
    for space in args.spaces:
        molecule_path = str(REPO_ROOT / MOLECULES[space])
        for dim in args.dims:
            print(f"=== {space} (dim={dim}) ===", flush=True)
            report = run_molecule(
                molecule_path, backend=args.backend, checkpoint=args.checkpoint, max_dim=dim,
            )
            print(report.summary(), flush=True)
            results.append({"space": space, "max_dim": dim, **asdict(report)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} reports to {out_path}")


if __name__ == "__main__":
    main()
