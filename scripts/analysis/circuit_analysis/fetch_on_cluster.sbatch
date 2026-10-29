#!/usr/bin/env bash
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

#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J fetch_ibm
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH -o /path/to/sqd-gqe-work/experiments/production_final/logs/fetch_ibm.out
#SBATCH -e /path/to/sqd-gqe-work/experiments/production_final/logs/fetch_ibm.err
# Run on roquo with the cluster's own IBM account (.env.local), after the maintenance window:
#   sbatch scripts/analysis/circuit_analysis/fetch_on_cluster.sbatch
# Writes circuit_analysis/qpu_usage.json and circuit_analysis/kobe_calibration.json under the repo; copy them to the laptop, then run
# make_cost_section.py and make_calibration_table.py there. No credentials are printed.
set -euo pipefail
module load cuda/13.2 nvhpc/26.5 hpcx/2.50
set -a; source /path/to/sqd-gqe/.env.local; set +a
VENV=/path/to/sqd-gqe/.venv
REPO=/path/to/sqd-gqe-work
LOGS=$REPO/experiments/production_final/logs
cd "$REPO"
mkdir -p circuit_analysis
# fresh-shot inference job ids: the data/*.json of the laptop are mirrored from experiments/crossnorb70q_infer/circuit_analysis on the cluster
mkdir -p circuit_analysis/data && cp experiments/crossnorb70q_infer/circuit_analysis/*.json circuit_analysis/data/
$VENV/bin/python scripts/analysis/circuit_analysis/collect_job_ids.py "$LOGS" --out circuit_analysis/ids.json
$VENV/bin/python scripts/analysis/circuit_analysis/fetch_qpu_usage.py --ids-file circuit_analysis/ids.json --out circuit_analysis/qpu_usage.json
$VENV/bin/python scripts/analysis/circuit_analysis/fetch_kobe_calibration.py --out circuit_analysis/kobe_calibration.json
