#!/usr/bin/env bash
#SBATCH --job-name=mps-bond-sweep
#SBATCH --partition=roquo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
# Submit with --account=<project>; arguments go to benchmark/mps_bond_sweep.py.
# Use --skip-energy for sampling only, or --sampler-kind mps_topk --with-exact
# to compare determinant selection with the exact simulator. See docs/mps.md.
set -euo pipefail

# sbatch copies this script into its spool; resolve helpers from the submit directory.
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/mps_runtime_env.sh"
python benchmark/mps_bond_sweep.py "$@"
