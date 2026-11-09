#!/usr/bin/env bash
#SBATCH --job-name=sqd-gqe-mps
#SBATCH --partition=roquo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Submit with --account=<project>; pass Hydra overrides after the script name.
# For mps_topk, also pass sqd.enlarge_method=None sqd.recover_iterations=0.
# Setup, commands and measurements: docs/mps.md.
set -euo pipefail

# sbatch copies this script into its spool; resolve helpers from the submit directory.
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/mps_runtime_env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python train_sqd.py sampler.kind=mps "$@"
