#!/usr/bin/env bash
#SBATCH --job-name=sqd-gqe-ffsim
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
VENV="${VENV:-.venv_arm}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# The bundled scipy/numpy OpenBLAS wheels need their lib dirs on the loader path.
SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:${LD_LIBRARY_PATH:-}"

# ffsim's Rust core is CPU-multithreaded; the GPU is used for the transformer.
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1

python train_sqd.py "$@"
