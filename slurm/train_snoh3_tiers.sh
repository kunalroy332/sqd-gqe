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

#SBATCH --job-name=snoh3-train
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

SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:${LD_LIBRARY_PATH:-}"

export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1

# TIER selects which molecule set trains in this job (submit one job per
# tier so each gets its own wall-clock budget and checkpoint). CO2/ethanol
# AND Sn(OH)3-Me's "S" active space (32q) are excluded from round-robin
# training: a smoke test (2026-08-08, job 20652) measured a SINGLE S
# iteration at num_samples=10 taking 20+ minutes (0% GPU, ~1 CPU core busy
# the whole time -- ffsim/pyscf sampling+diagonalization cost scaling with
# S's ~3.3e6 FCI dim, not a bug) even at reduced shots=2000/max_dim=2000.
# Same order of cost as CO2 (28q)/ethanol (32q), already excluded for the
# identical reason. S/M/L/LL are zero-shot eval only (slurm/infer_snoh3.sh),
# never trained on directly -- consistent with benzene/Fe2S2's existing
# held-out pattern in this branch.
TIER="${TIER:-small}"
MAX_DIM="${MAX_DIM:-2000}"

case "$TIER" in
  small)
    MOLECULES="molecules/small/h2.yaml molecules/small/lih.yaml molecules/small/h2o.yaml \
               molecules/small/hf.yaml molecules/small/beh2.yaml molecules/small/nh3.yaml \
               molecules/small/n2.yaml molecules/small/ch4.yaml"
    ;;
  medium)
    MOLECULES="molecules/medium/ch2o.yaml molecules/medium/ch3oh.yaml molecules/medium/hcn.yaml \
               molecules/medium/c2h4.yaml"
    ;;
  *)
    echo "Unknown TIER=$TIER (expected small|medium)" >&2
    exit 1
    ;;
esac

python benchmark/amortized_train.py \
  --molecules $MOLECULES \
  --iters 300 \
  --max-vocab 64 \
  --max-dim "$MAX_DIM" \
  --shots 10000 \
  --out "experiments/amortized/${TIER}_300iter_run.json" \
  --ckpt "experiments/amortized/${TIER}_300iter_policy.pt" \
  --checkpoint-every 10
