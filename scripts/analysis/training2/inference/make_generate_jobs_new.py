# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Slurm jobs that generate the sequences of the runs added when the campaign was extended from 24 to 44 runs per model set (all 11 active-space tiers).
New runs: HF penalty and tuned, every (lineage, tier) the first campaign did not run (20 per set); Training 1 ("t1"), the two tiers the first campaign never ran
(SnOH3-me M = sno38, SnOH3-me-h2o L = h2o_l28) on the four lineages (8 runs). No hardware. Usage: python3 make_generate_jobs_new.py <outdir>"""
import sys
from pathlib import Path

import generate_sequences as g
import make_generate_jobs as base

R, V = base.R, base.V


def new_runs():
    runs = []
    for arm in ("hf", "tuned"):
        for lin, mks in g.TARGETS.items():
            runs += [(arm, lin, m) for m in mks if m not in g.FIRST_CAMPAIGN[lin]]
    for lin in g.TARGETS:
        runs += [("t1", lin, m) for m in ("sno38", "h2o_l28")]
    return runs


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sub = ["#!/usr/bin/env bash", f"mkdir -p {R}/experiments/inference_logs"]
    by = {}
    for arm, lin, mk in new_runs():
        by.setdefault((arm, lin), []).append(mk)
    for (arm, lin), mks in by.items():
        name = f"gen2_{arm}_{lin}"
        body = base.HEAD.format(name=name, R=R, V=V, root=g.ARMS[arm][1])
        for mk in mks:
            body += f"PYTHONPATH=. {V}/bin/python scripts/analysis/training2/inference/generate_sequences.py --arm {arm} --lineage {lin} --molecule-key {mk} || echo FAILED {mk}\n"
        (out / f"{name}.sbatch").write_text(body)
        sub.append(f"sbatch --parsable {R}/experiments/inference_logs/gen2_jobs/{name}.sbatch")
    (out / "submit_gen2.sh").write_text("\n".join(sub) + "\n")
    print(len(new_runs()), "new runs in", len(by), "jobs")


if __name__ == "__main__":
    main(sys.argv[1])
