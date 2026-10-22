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

"""Training 2, Stage 2 with the jointly tuned reward (config 3 of the C2H4 noisy-simulator search, scripts/analysis/hf_penalty/make_jobs_ccc_tune.py):
--cost-lambda-hf 0.07 --cost-lambda-spin 0.77 --entropy-coef 0.26 --cost-lambda-excitation 3.9 --cost-lambda-coverage 7.2.
Selection (last 20 of 80 iterations, 2 seeds): raw error 15.0 mHa (reference ~11), HF share of valid shots 0.18, effective vocabulary 8.1 (no collapse),
2.8 HF-changing gates per circuit, depth 76. All four lineages, warm-started from the Training-2 Stage-1 checkpoints; named t2t_s2_*.
Code copy: experiments/hf_tune/tune_repo2 (has the spin and entropy terms). Usage on the roquo login node: python3 make_jobs_tuned.py <outdir>"""
import sys
from pathlib import Path

import make_jobs as mj

TUNED = "--cost-lambda-hf 0.07 --cost-lambda-spin 0.77 --entropy-coef 0.26"
LINEAGES = [("a1", "46q"), ("a5", "46q"), ("a1", "70q"), ("a5", "70q")]
CODE = f"{mj.REPO}/experiments/hf_tune/tune_repo2"


def main(outdir):
    out = Path(outdir); (out / "jobs").mkdir(parents=True, exist_ok=True)
    sub = ["#!/usr/bin/env bash", "set -e", f": > {mj.EXP}/submitted_tuned.txt"]
    for lin, tag in LINEAGES:
        name, body = mj.stage2(lin, tag)
        new = name.replace("t2_s2_", "t2t_s2_")
        body = body.replace(f"#SBATCH -J {name}", f"#SBATCH -J {new}")
        for sec in ("logs", "out", "ckpt", "track", "ibm_jobs"):
            body = body.replace(f"/{sec}/{name}", f"/{sec}/{new}")
        assert "--cost-lambda-excitation 2.0 --cost-lambda-coverage 10.0" in body
        body = body.replace("--cost-lambda-excitation 2.0 --cost-lambda-coverage 10.0", "--cost-lambda-excitation 3.9 --cost-lambda-coverage 7.2")
        body = body.replace("--sampler-kind ibm_hardware", f"{TUNED} --sampler-kind ibm_hardware")
        body = body.replace(f"cd {mj.REPO}\n", f"cd {CODE}\n")
        (out / "jobs" / f"{new}.sbatch").write_text(body)
        sub.append(f"J=$(sbatch --parsable {mj.EXP}/jobs/{new}.sbatch); echo \"{new} $J\" >> {mj.EXP}/submitted_tuned.txt")
    sub.append(f"cat {mj.EXP}/submitted_tuned.txt")
    (out / "submit_tuned.sh").write_text("\n".join(sub) + "\n")
    print("wrote", len(LINEAGES), "job scripts")


if __name__ == "__main__":
    main(sys.argv[1])
