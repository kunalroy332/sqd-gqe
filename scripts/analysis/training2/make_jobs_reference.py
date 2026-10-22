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

"""Training 2, Stage 2, no penalty: re-run of the three reference lineages whose first runs were cancelled (A5-46q, A1-70q, A5-70q).
Named t2r_s2_* with their own out/ckpt/track/ibm_jobs files; the cancelled partial runs (t2_s2_*) are left untouched.
Uses the unmodified benchmark/amortized_train.py, as the A1-46q reference t2_s2_a1_46q did. Usage on the cluster login node: python3 make_jobs_reference.py <outdir>"""
import sys
from pathlib import Path

import make_jobs as mj

LINEAGES = [("a5", "46q"), ("a1", "70q"), ("a5", "70q")]


def main(outdir):
    out = Path(outdir); (out / "jobs").mkdir(parents=True, exist_ok=True)
    sub = ["#!/usr/bin/env bash", "set -e", f": > {mj.EXP}/submitted_reference.txt"]
    for lin, tag in LINEAGES:
        name, body = mj.stage2(lin, tag)
        new = name.replace("t2_s2_", "t2r_s2_")
        body = body.replace(f"#SBATCH -J {name}", f"#SBATCH -J {new}")
        for sec in ("logs", "out", "ckpt", "track", "ibm_jobs"):
            body = body.replace(f"/{sec}/{name}", f"/{sec}/{new}")
        (out / "jobs" / f"{new}.sbatch").write_text(body)
        sub.append(f"J=$(sbatch --parsable {mj.EXP}/jobs/{new}.sbatch); echo \"{new} $J\" >> {mj.EXP}/submitted_reference.txt")
    sub.append(f"cat {mj.EXP}/submitted_reference.txt")
    (out / "submit_reference.sh").write_text("\n".join(sub) + "\n")
    print("wrote", len(LINEAGES), "job scripts")


if __name__ == "__main__":
    main(sys.argv[1])
