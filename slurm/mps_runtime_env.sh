# Shared setup for train_mps.sh and mps_bond_sweep.sh, sourced from the submit directory.
# shellcheck disable=SC1091
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/sbd_runtime_env.sh"

if [ -z "${VENV:-}" ]; then
    for candidate in .venv_sqd .venv_arm .venv; do
        [ -f "$candidate/bin/activate" ] && VENV="$candidate" && break
    done
fi
if [ -z "${VENV:-}" ] || [ ! -f "$VENV/bin/activate" ]; then
    echo 'no usable virtualenv (looked for $VENV, .venv_sqd, .venv_arm, .venv)' >&2
    exit 1
fi
echo "using venv: $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# A shared venv may have an editable install pointing at a different checkout.
_mps_checkout="$(python -c 'import os, sqd_gqe; print(os.path.dirname(os.path.dirname(sqd_gqe.__file__)))')"
if [ "$_mps_checkout" != "$PWD" ]; then
    echo "sqd_gqe resolves to $_mps_checkout but this job runs in $PWD" >&2
    echo "Set PYTHONPATH=$PWD or submit from the intended checkout." >&2
    exit 1
fi
echo "sqd_gqe: $_mps_checkout"

_mps_site="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$_mps_site/scipy.libs:$_mps_site/numpy.libs:${LD_LIBRARY_PATH:-}"
export RAYON_NUM_THREADS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-16}}"
export OMP_NUM_THREADS=1
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2097152}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# GPU-only dependencies stay out of pyproject.toml. Lock a shared venv during installation.
CUQUANTUM_PKG="${CUQUANTUM_PKG:-cuquantum-python-cu13}"
(
    flock 9
    if ! python -c 'import cuquantum, cupy' 2>/dev/null; then
        echo "installing $CUQUANTUM_PKG into $VENV (one-time)" >&2
        case "$CUQUANTUM_PKG" in
            *-cu12*) pip install --quiet "$CUQUANTUM_PKG" cupy-cuda12x ;;
            *) pip install --quiet "$CUQUANTUM_PKG" cupy-cuda13x ;;
        esac
    fi
) 9>"$VENV/.cuquantum-install.lock"
python -c "import cuquantum, cupy; print('cuquantum', cuquantum.__version__, '| cupy', cupy.__version__)"
unset _mps_checkout _mps_site candidate
