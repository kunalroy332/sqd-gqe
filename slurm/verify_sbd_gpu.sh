#!/usr/bin/env bash
#SBATCH --job-name=verify-sbd-gpu
#SBATCH --partition=roquo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
set -euo pipefail

module purge
module load cuda/13.3 gcc/14 hpcx/2.50
VENV="${VENV:-.venv_arm}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:${LD_LIBRARY_PATH:-}"

# The in-allocation `srun` launched from inside benchmark/verify_sbd_gpu.py must NOT inherit this
# wrapper job's own #SBATCH --cpus-per-task -- srun refuses to start when SLURM_CPUS_PER_TASK and
# SLURM_TRES_PER_TASK disagree ("cpus-per-task set by two different environment variables"),
# confirmed empirically (job 21916). Same fix qcsc-prefect's own in-allocation sweep scripts use.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

# Stage 0 (2026-08-09) confirmed diag-gpu links and runs identically under this module set and
# SBD's own (cuda/13.2 nvhpc/26.5) -- no library mismatch either way, so we keep our existing set.
SBD_BIN="${SBD_BIN:-/home/<USER>/qcsc-prefect/algorithms/sbd/native/diag-gpu}"
test -x "$SBD_BIN" || { echo "SBD binary not found/executable at $SBD_BIN" >&2; exit 1; }

python benchmark/verify_sbd_gpu.py --binary-path "$SBD_BIN" --launcher-cmd "srun --gpu-bind=closest"
