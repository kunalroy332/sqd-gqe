# Native SBD runtime for jobs without environment modules.
# backend_sbd_gpu.py merges SBD_LD_LIBRARY_PATH into the native child's loader path,
# keeping SDK CUDA libraries out of the Python/cuQuantum process.

_sbd_add_libdir() {
    [ -d "$1" ] || return 0
    case ":${SBD_LD_LIBRARY_PATH:-}:" in *":$1:"*) return 0 ;; esac
    SBD_LD_LIBRARY_PATH="${SBD_LD_LIBRARY_PATH:+${SBD_LD_LIBRARY_PATH}:}$1"
}

if ! command -v mpirun >/dev/null 2>&1; then
    for _d in /opt/hpcx-*/ompi5/bin; do
        if [ -x "$_d/mpirun" ]; then export PATH="$_d:$PATH"; break; fi
    done
fi
if command -v mpirun >/dev/null 2>&1; then
    echo "mpirun: $(command -v mpirun)"
    # Override OMPI's build-time prefix, which can point at a removed build directory.
    if [ -z "${OPAL_PREFIX:-}" ]; then
        OPAL_PREFIX="$(cd "$(dirname "$(command -v mpirun)")/.." && pwd)"
        export OPAL_PREFIX
        export PRTE_PREFIX="${PRTE_PREFIX:-$OPAL_PREFIX}"
    fi
    echo "OPAL_PREFIX: ${OPAL_PREFIX}"
else
    echo "note: no mpirun on PATH; sqd.diag_backend=sbd will refuse to start" >&2
fi

for _d in /opt/hpcx-*/ompi5/lib /opt/hpcx-*/ucx/lib /opt/hpcx-*/ucc/lib; do
    _sbd_add_libdir "$_d"
done

# Choose one NVHPC release: mixing releases can load the wrong libnccl.so.2.
# Set SBD_NVHPC_ROOT to match the build; otherwise use the highest-sorting release.
if [ -z "${SBD_NVHPC_ROOT:-}" ]; then
    for _d in $(ls -d /opt/nvidia/hpc_sdk/Linux_*/*/ 2>/dev/null | sort -r); do
        [ -d "${_d}compilers/lib" ] && SBD_NVHPC_ROOT="${_d%/}" && break
    done
fi
if [ -n "${SBD_NVHPC_ROOT:-}" ]; then
    echo "nvhpc root: ${SBD_NVHPC_ROOT}"
    for _d in "${SBD_NVHPC_ROOT}"/math_libs/*/targets/*/lib \
              "${SBD_NVHPC_ROOT}"/comm_libs/*/nccl/lib \
              "${SBD_NVHPC_ROOT}"/cuda/*/targets/*/lib \
              "${SBD_NVHPC_ROOT}"/compilers/lib; do
        _sbd_add_libdir "$_d"
    done
else
    echo "note: no NVHPC install found under /opt/nvidia/hpc_sdk" >&2
fi
unset _d
export SBD_LD_LIBRARY_PATH="${SBD_LD_LIBRARY_PATH:-}"
echo "SBD_LD_LIBRARY_PATH: ${SBD_LD_LIBRARY_PATH:-<empty>}"
