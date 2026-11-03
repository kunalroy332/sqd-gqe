# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Write one rerank sbatch per run tag. Usage: make_rerank_sbatch.py <out_dir> <tag> [<tag> ...]"""
import sys
from pathlib import Path

TEMPLATE = """#!/usr/bin/env bash
#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J rr_{tag}
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --time=3:00:00
#SBATCH --time-min=0:40:00
#SBATCH --exclude=qr01n08,qr02n11,qr04n16
#SBATCH -o /path/to/sqd-gqe-work/experiments/production_final/logs/rr_{tag}.out
#SBATCH -e /path/to/sqd-gqe-work/experiments/production_final/logs/rr_{tag}.err

module load cuda/13.2 nvhpc/26.5 hpcx/2.50
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
set -a
source /path/to/sqd-gqe/.env.local
set +a
VENV=/path/to/sqd-gqe/.venv
SP=$($VENV/bin/python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:$LD_LIBRARY_PATH"
cd /path/to/sqd-gqe-work
PYTHONPATH=. $VENV/bin/python scripts/analysis/circuit_analysis/rerank_circuits.py --tag {tag}
"""

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for tag in sys.argv[2:]:
    (out / f"rr_{tag}.sbatch").write_text(TEMPLATE.format(tag=tag))
print(f"wrote {len(sys.argv) - 2} files")
