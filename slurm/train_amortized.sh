#!/usr/bin/env bash
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#SBATCH --job-name=sqd-gqe-amortized
#SBATCH --partition=roquo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
set -euo pipefail

module purge
module load cuda/13.3 gcc/14 hpcx/2.50
VENV="${VENV:-.venv_sqd}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:${LD_LIBRARY_PATH:-}"

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1

# Round-robin amortized trainer over the small+medium(+large) molecule tiers.
python benchmark/amortized_train.py "$@"
