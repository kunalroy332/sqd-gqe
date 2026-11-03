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

#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time=0:30:00
#SBATCH -o /path/to/sqd-gqe-work/experiments/production_final/logs/fin_%x.out
#SBATCH -e /path/to/sqd-gqe-work/experiments/production_final/logs/fin_%x.err
# Selection + submission of the five recovery jobs for ONE run ($TAG), after its ranking finished.
set -euo pipefail
REPO=/path/to/sqd-gqe-work
SCR=$REPO/scripts/analysis/circuit_analysis
OUT=$REPO/experiments/crossnorb70q_infer
VENV=/path/to/sqd-gqe/.venv
: "${TAG:?TAG must be set (submit_missing_runs.sh does this)}"
cd "$REPO"
W=$OUT/missing_runs_work/$TAG
mkdir -p "$W/rerank_in"
cp "$OUT/circuit_analysis_rerank/$TAG.json" "$W/rerank_in/"
# select_new.py: best/second = two lowest 40000-dim ranking energies, merged = all finite, plus E+D 1/2 = the earlier rule's picks
PYTHONPATH=. $VENV/bin/python "$SCR/select_new.py" --rerank-dir "$W/rerank_in" --out "$W/selection.json" --write-sbatch "$W/rc"
# long recoveries, no maintenance deadline: fixed 36 h, 2 GPUs each (these ran 4 min to 12 h in the other runs)
for f in $(cat "$W/rc/SUBMIT_ORDER.txt"); do
  sed -i 's/^#SBATCH --time=.*/#SBATCH --time=36:00:00/; s/^#SBATCH --time-min=.*/#SBATCH --time-min=36:00:00/' "$W/rc/$f"
  sbatch "$W/rc/$f"
done
echo "submitted recoveries for $TAG; results appear in $OUT/circuit_analysis_rerecovery/${TAG}__*.json"
