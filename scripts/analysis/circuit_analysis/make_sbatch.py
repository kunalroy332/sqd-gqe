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

"""Write one sbatch file per completed cross-norb run for resample_raw_counts.py.

Reads a text file of result filenames (crossnorb70q_<tag>_sub20000_job<N>.json),
one per line, and writes ca_<tag>.sbatch into the given output directory."""
import re
import sys
from pathlib import Path

from ca_common import cache_path

TEMPLATE = """#!/usr/bin/env bash
#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J ca_{tag}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o /path/to/sqd-gqe-work/experiments/production_final/logs/ca_{tag}.out
#SBATCH -e /path/to/sqd-gqe-work/experiments/production_final/logs/ca_{tag}.err

module load cuda/13.2 nvhpc/26.5 hpcx/2.50
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
set -a
source /path/to/sqd-gqe/.env.local
set +a
VENV=/path/to/sqd-gqe/.venv
SP=$($VENV/bin/python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:$LD_LIBRARY_PATH"
cd /path/to/sqd-gqe-work
PYTHONPATH=. $VENV/bin/python scripts/analysis/circuit_analysis/resample_raw_counts.py --cache {cache}
"""


def main(list_file: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = 0
    for line in Path(list_file).read_text().split():
        m = re.match(r"crossnorb70q_(.+)_sub20000_job\d+\.json$", line)
        tag = m.group(1) if m else (line if line.startswith("L46_") else None)
        if tag:
            (Path(out_dir) / f"ca_{tag}.sbatch").write_text(TEMPLATE.format(tag=tag, cache=cache_path(tag)))
            n += 1
    print(f"wrote {n} sbatch files to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
