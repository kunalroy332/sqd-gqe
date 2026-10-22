#!/usr/bin/env bash
#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J ibm_inventory
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time=3:00:00
#SBATCH -o /path/to/sqd-gqe-work/experiments/production_final/logs/ibm_inventory.out
#SBATCH -e /path/to/sqd-gqe-work/experiments/production_final/logs/ibm_inventory.err
# Lists every IBM Runtime job of the cluster account and its billed QPU time (read-only). No credential is printed.
set -euo pipefail
module load cuda/13.2 nvhpc/26.5 hpcx/2.50
set -a; source /path/to/sqd-gqe/.env.local; set +a
VENV=/path/to/sqd-gqe/.venv
cd /path/to/sqd-gqe-work
mkdir -p circuit_analysis
PYTHONPATH=scripts/analysis/circuit_analysis $VENV/bin/python scripts/analysis/circuit_analysis/fetch_ibm_inventory.py --since 2026-07-01
