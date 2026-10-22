# Environment setup: single-node, single-GPU, and multi-GPU

This covers building a working Python environment for `sqd_gqe` and, separately, the native
GPU diagonalizer binary (`diag-gpu`) that `--diag-backend sbd` shells out to. The Python
environment is the same regardless of node/GPU count; the native binary has a different build
for single-GPU vs. multi-GPU.

## 1. Python environment (any node)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip "setuptools<82" wheel
pip install -e .                 # ffsim, qiskit-addon-sqd, qiskit, PyTorch, ...
pip install -e '.[dev]'          # plus pytest, ruff and pre-commit for development
```

`pyproject.toml`'s pins are the versions actually validated against real IBM Quantum hardware
runs on RIKEN roquo (`ibm_quantum_platform` channel) — notably `qiskit==2.5.1` and
`qiskit-ibm-runtime==0.49.0`. If you're pinned to an older `qiskit-ibm-runtime` (e.g. `0.37`)
elsewhere, that version predates `ibm_quantum_platform` support and will not work with the
real-hardware sampler in this repo.

### Two environment fixes that are load-bearing, not optional

Every real training/inference run on roquo needed both of these — without them, imports
segfault or crash at import time on that cluster's aarch64 nodes (may or may not apply on your
platform, but cheap to always set):

```bash
# scipy/numpy ship bundled shared libs under site-packages/{scipy,numpy}.libs -- make sure
# they're on the loader path before anything imports scipy/numpy transitively.
SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:${LD_LIBRARY_PATH:-}"

# Works around a glibc static-TLS exhaustion crash seen when many extension modules
# (torch, qiskit-aer, ffsim, ...) are imported together.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
```

### IBM Quantum credentials (real hardware only)

Copy `.env.example` (repo root) to `.env.local`, fill in `IBM_API_KEY`/`IBM_CRN`, then before
running anything that uses `--sampler-kind ibm_hardware`:

```bash
set -a; source .env.local; set +a
```

## 2. Single-GPU: `--diag-backend sbd` with the plain `diag-gpu` binary

`--diag-backend sqd` (the default) runs entirely in Python (`qiskit-addon-sqd`, CPU) and needs
no extra build step — start there if you don't need GPU-accelerated diagonalization.

`--diag-backend sbd` shells out to a **native binary** built from a separate repository
(`https://github.com/r-ccs-cms/sbd`, not part of `sqd-gqe`):

```bash
module load cuda gcc          # or your platform's equivalent CUDA + C++/MPI compiler modules
git clone https://github.com/r-ccs-cms/sbd.git
cd sbd/native                 # build script lives alongside main.cc in the sbd repo
CCCOM=mpic++ ./build_sbd_gpu.sh
# produces ./diag-gpu (RHF) -- pass UHF=1 for the unrestricted variant, ./diag-gpu_uhf
```

Point `sqd_gqe` at it:

```bash
--diag-backend sbd --sbd-binary-path /path/to/diag-gpu
```

This is what every real single-GPU result in this project used (e.g. the 32-qubit training
run and its +0.0046 mHa chemical-accuracy inference result).

## 3. Multi-GPU (single node): validated recipe

**What's actually validated**: 4 GPUs on **one physical node** (Slurm `-N 1 --gres=gpu:4`),
using an MPI+NCCL build of the same binary, `diag-gpu_uhf-mpi-hpcx250`. This is what the
38-qubit real-hardware training run in this project used, and it is the recipe below — not a
theoretical extrapolation.

### Build

Requires `hpcx/2.50` specifically as a **standalone module** — not whatever HPC-X ships bundled
inside the CUDA toolkit's `comm_libs`, which is a different, incompatible build. Confirm with
`module avail hpcx` on your cluster; if only a bundled variant exists, this build may need
adjusting.

```bash
module load cuda gcc hpcx/2.50
git clone --branch non-cuda-aware-mpi https://github.com/rwakizaka/sbd.git sbd_mpi
cd sbd_mpi
# apply the two patches documented in this fork's own README before building:
#  - sbd/chemistry/tpb/sbdiag.h: shrink the NCCL warm-up buffer allocation
#    (thrust::device_vector<double> A(W.size(), 0.0) -> A(1, 0.0))
#  - sbd/chemistry/tpb/helper.h: hoist GenerateExcitation's cr/an scratch buffers
#    outside the OMP parallel-for loop
./build_sbd_mpi_uhf_hpcx250.sh
# produces diag-gpu_uhf-mpi-hpcx250
```

### Run (4 GPUs, 1 node)

Two environment variables were found necessary for NCCL to bootstrap correctly on this
cluster's interconnect, even at single-node scale:

```bash
export NCCL_SOCKET_IFNAME=ib0        # match your cluster's InfiniBand/interconnect interface
unset OMPI_MCA_mca_base_env_list
```

Wire the launcher and the 4-way determinant-partition grid (`a=2 x b=2`) through Hydra
overrides:

```bash
--diag-backend sbd \
--sbd-binary-path /path/to/diag-gpu_uhf-mpi-hpcx250 \
--hydra-override "sqd.sbd_launcher=['srun','--mpi=pmix','--gpu-bind=closest','-n','4']" \
--hydra-override sqd.sbd_solver_kwargs.adet_comm_size=2 \
--hydra-override sqd.sbd_solver_kwargs.bdet_comm_size=2 \
--hydra-override sqd.sbd_solver_kwargs.task_comm_size=1
```

The `-n 4` in the launcher list is required explicitly — when this binary is invoked from
inside a nested `srun` (Python's own `subprocess.run` call, itself already running inside a
Slurm job), `srun` does **not** automatically inherit the parent job's task count and will
otherwise launch only 1 rank, which crashes with a shape-mismatch error from the binary itself.

**Grid constraint**: `task_comm_size x adet_comm_size x bdet_comm_size x h_comm_size` must
equal the total MPI world size (4, here) — see `sqd_gqe/sqd/backend_sbd_gpu.py`'s
`solver_kwargs` handling.

### Sizing recovery to the GPU count

`--recover-subspace-dim` sets the alpha-determinant count per recovery step (net CI dimension
= this value squared, RHF). At 4 GPUs / 1M net dimension, real recovery calls completed in
roughly 12 seconds each; occasional intermittent stalls were observed under sustained load —
`backend_sbd_gpu.py`'s `_solve_once` retries a stalled call up to 3 times before raising, and
`--sbd-timeout-s` should be generous (900s+) rather than the 600s default at this scale.

## 4. True multi-node (>1 physical node): not yet validated

Everything above is single-node. **A genuinely multi-node run (Slurm `-N 2` or more) was never
actually tested in this project** — do not assume the recipe above extends unchanged.

What would need to change, based on how the binary and launcher work:

- Slurm request: `-N <nodes> --gres=gpu:<gpus_per_node>`, with `-n` in the `sbd_launcher`
  override raised to match the **total** rank count across all nodes (not per-node).
- The determinant-partition grid (`adet_comm_size x bdet_comm_size x ... = total ranks`) would
  need rebalancing for the new total rank count — the 2x2 grid above is sized for exactly 4
  ranks.
- The two NCCL bootstrap environment variables above (`NCCL_SOCKET_IFNAME`,
  `OMPI_MCA_mca_base_env_list`) are, per the upstream `sbd` repo's own README, specifically
  documented as needed for **multi-node** NCCL bootstrap — they were applied at single-node
  scale in this project "defensively" (harmless there) but are the two things most likely to
  actually matter once multiple physical nodes are involved.
- Network topology (which interconnect/interface name to put in `NCCL_SOCKET_IFNAME`) is
  cluster-specific and would need to be confirmed on the target system, not assumed.

Treat this section as a starting point for the next person to try multi-node, not a proven
recipe.
