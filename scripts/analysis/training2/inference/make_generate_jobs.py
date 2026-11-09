# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Slurm jobs for the sequence generation of the 48 Training-2 inference runs (no hardware): one job per (arm, lineage) that loops over that lineage's molecules.
Writes <out>/gen_<arm>_<lineage>.sbatch and <out>/submit_gen.sh. Usage on a machine with numpy-free python: python3 make_generate_jobs.py <outdir>"""
import sys
from pathlib import Path

R = "/path/to/sqd-gqe-work"
V = "/path/to/sqd-gqe/.venv"
ARMS = {"hf": "inference_hf", "tuned": "inference_tuned"}
TARGETS = {
    "a1-46q": ["2h2o_s22", "L46", "h2o_m_no23", "h2o_s19", "sno32"], "a5-46q": ["2h2o_s22", "L46", "h2o_m_no23", "h2o_s19", "sno32"],
    "a1-70q": ["2h2o_l35", "2h2o_m_no29", "L46", "h2o_ll33", "h2o_m_no23", "llno28", "sno32"],
    "a5-70q": ["2h2o_l35", "2h2o_m_no29", "L46", "h2o_ll33", "h2o_m_no23", "llno28", "sno32"],
}
HEAD = """#!/usr/bin/env bash
#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J {name}
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH -o {R}/experiments/inference_logs/{name}.out
#SBATCH -e {R}/experiments/inference_logs/{name}.err
module load cuda/13.2 nvhpc/26.5 hpcx/2.50
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
SP=$({V}/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:$LD_LIBRARY_PATH"
cd {R}
export INFER_ROOT={R}/experiments/{root}
"""


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sub = ["#!/usr/bin/env bash", f"mkdir -p {R}/experiments/inference_logs"]
    for arm, root in ARMS.items():
        for lin, mks in TARGETS.items():
            name = f"gen_{arm}_{lin}"
            body = HEAD.format(name=name, R=R, V=V, root=root)
            for mk in mks:
                body += f"PYTHONPATH=. {V}/bin/python scripts/analysis/training2/inference/generate_sequences.py --arm {arm} --lineage {lin} --molecule-key {mk} || echo FAILED {mk}\n"
            (out / f"{name}.sbatch").write_text(body)
            sub.append(f"sbatch --parsable {R}/experiments/inference_logs/{name}.sbatch")
    (out / "submit_gen.sh").write_text("\n".join(sub) + "\n")
    print("wrote", len(ARMS) * len(TARGETS), "job scripts")


if __name__ == "__main__":
    main(sys.argv[1])
