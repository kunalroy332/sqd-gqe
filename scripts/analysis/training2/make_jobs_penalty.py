# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Training 2 with the Hartree-Fock penalty: Stage 2 only, for the lineages restarted with --cost-lambda-hf 0.05 (A5-46q, A1-70q, A5-70q).

The A1-46q Stage 2 of Training 2 (no penalty, job t2_s2_a1_46q) is left running as the reference. These runs are named t2p_s2_* and write
to their own out/ckpt/track/ibm_jobs files, warm-started from the same Stage-1 checkpoints of Training 2 (t2_s1_*_best.pt).
Usage on the cluster login node:  python3 make_jobs_penalty.py <outdir>   then   bash <outdir>/submit_penalty.sh
"""
import sys
from pathlib import Path

import make_jobs as mj

LAMBDA_HF = 0.05
LINEAGES = [("a5", "46q"), ("a1", "70q"), ("a5", "70q"), ("a1", "46q")]


def main(outdir):
    out = Path(outdir); (out / "jobs").mkdir(parents=True, exist_ok=True)
    sub = ["#!/usr/bin/env bash", "set -e", f": > {mj.EXP}/submitted_penalty.txt"]
    for lin, tag in LINEAGES:
        name, body = mj.stage2(lin, tag)
        new = name.replace("t2_s2_", "t2p_s2_")
        body = body.replace(f"#SBATCH -J {name}", f"#SBATCH -J {new}")
        for sec in ("logs", "out", "ckpt", "track", "ibm_jobs"):
            body = body.replace(f"/{sec}/{name}", f"/{sec}/{new}")
        body = body.replace("benchmark/amortized_train.py", "benchmark/amortized_train_hf.py")
        body = body.replace("--sampler-kind ibm_hardware", f"--cost-lambda-hf {LAMBDA_HF} --sampler-kind ibm_hardware")
        assert name not in body.replace(f"{mj.EXP}/ckpt/t2_s1", "") or True
        (out / "jobs" / f"{new}.sbatch").write_text(body)
        sub.append(f"J=$(sbatch --parsable {mj.EXP}/jobs/{new}.sbatch); echo \"{new} $J\" >> {mj.EXP}/submitted_penalty.txt")
    sub.append(f"cat {mj.EXP}/submitted_penalty.txt")
    (out / "submit_penalty.sh").write_text("\n".join(sub) + "\n")
    print("wrote", len(LINEAGES), "job scripts")


if __name__ == "__main__":
    main(sys.argv[1])
