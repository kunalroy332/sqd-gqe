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

# Set up the Python environment for SQD-GQE on the roquo cluster.
# Run once on a login node from the repo root:  bash slurm/bootstrap_roquo.sh
set -euo pipefail

module purge
module load cuda/13.3 gcc/14 hpcx/2.50

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip "setuptools<82" wheel

# GPU torch wheels (adjust the index to the cluster's CUDA if needed).
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [ -n "$TORCH_INDEX_URL" ]; then
    pip install --extra-index-url "$TORCH_INDEX_URL" -e .
else
    pip install -e .
fi

# Sanity: the bundled scipy/numpy OpenBLAS wheels need their lib dirs on the
# loader path (and hot kernels only run on compute nodes, not the login node).
SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:${LD_LIBRARY_PATH:-}"

echo
echo "Environment ready in $VENV."
echo "NOTE: run compute via Slurm (srun/sbatch) -- the login node is a QEMU CPU"
echo "      that SIGILLs on ffsim/OpenBLAS numerical kernels."
echo "To install the development extras as well:"
echo "    pip install -e '.[dev]'"
