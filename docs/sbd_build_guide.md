# Building your own copy of the SBD diagonalizer (independent of this repo's account)

This guide is for building your own `diag`/`diag-gpu`/`diag-gpu-mpi`-equivalent binary from
scratch, on your own account, with no dependency on anyone else's pre-built binary or filesystem
access. Every fact below was verified directly against real, public source (repo READMEs, build
scripts, and source diffs), not assumed. Where something is genuinely account-specific tribal
knowledge rather than upstream documentation, that is called out explicitly.

## 0. What you're actually building, and where each piece really comes from

There are **two independent things** commonly called "the SBD binary," built from **different
source trees**, and it matters which one you want:

| Binary | Source | GPU count | Status |
|---|---|---|---|
| `diag` (CPU) / `diag-gpu` (single GPU) | Public [`r-ccs-cms/sbd`](https://github.com/r-ccs-cms/sbd) (Apache-2.0) | 0 or 1 | Fully reproducible from the public repo alone |
| `diag-gpu-mpi` / `diag-gpu_uhf-mpi*` (multi-GPU) | A **fork**, `rwakizaka/sbd`, branch `non-cuda-aware-mpi` — **plus two hand-applied source patches, not upstream anywhere** | 4+ (multi-node validated) | Reproducible, but requires the fork + manual patches below |

Both binaries are compiled from a **local, vendored copy of `main.cc`**, not the cloned repo's own
`apps/chemistry_tpb_selected_basis_diagonalization/main.cc` — the clone is used only for its
`include/` headers (`-I$SBD_DIR/include`). If you skip this and try to build the clone's own
`apps/.../main.cc` directly, you will get a working `diag` binary, but **not** the
`SBD_PREFECT`-gated `davidson_energy.txt`/`occ_a.txt`/`occ_b.txt`/`carryover.bin` output format
unless you also enable that flag — see step 2.

## 1. License and citation

`r-ccs-cms/sbd` is Apache License 2.0 (`LICENSE.txt`, copyright RIKEN 2025 + AMD 2026) — free to
clone and build independently. The root `README.md` asks that you cite:
- arXiv:2511.00224 (the algorithm)
- arXiv:2601.16637 (the GPU/Thrust implementation)

if you use it in research output.

## 2. Building the CPU / single-GPU binary (`diag` / `diag-gpu`)

This path is fully public and reproducible with no forks or patches.

```bash
git clone https://github.com/r-ccs-cms/sbd.git sbd
```

The binary is produced by compiling a `main.cc` **you provide** against that clone's headers —
either the clone's own `apps/chemistry_tpb_selected_basis_diagonalization/main.cc` (plain public
behavior), or your own copy with `-DSBD_PREFECT` added if you want the
`davidson_energy.txt`/`occ_a.txt`/`occ_b.txt`/`carryover.bin` output format that Python tooling
built around this project consumes. `-DSBD_PREFECT` is a real, documented flag in that source —
it gates a block that writes exactly those four files (confirmed present at the repo's earliest
tagged release with this app, `v1.3.0`, and unchanged in spirit through current `main`).

**CPU build** (adapt the compiler/flags to your own site — these are Intel-oneAPI-flavored
example flags, not universal requirements):

```bash
mpiicpc -std=c++17 -qopenmp -O3 -DSBD_PREFECT \
  -c main.cc -o main.o -I sbd/include
mpiicpc -std=c++17 -qopenmp -O3 -o diag main.o -qmkl=parallel
```

**Single-GPU build** (NVHPC/Thrust flavor):

```bash
mpic++ -std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST -DSBD_PREFECT \
  -c main.cc -o main.o -I sbd/include
mpic++ -std=c++17 -mp -cuda -fast -gpu=mem:unified -DSBD_THRUST -DSBD_PREFECT \
  -o diag-gpu main.o -lblas -llapack
```

Add `-D_UHF` to both compile and link steps for the unrestricted (open-shell) variant
(`diag_uhf`/`diag-gpu_uhf`); omit it for the restricted (closed-shell, RHF) default.

You will need a working MPI C++ wrapper (`mpic++`/`mpiicpc`/`mpiicpx`/your site's equivalent) and,
for the GPU build, an NVHPC (or equivalent CUDA-aware) toolchain on `PATH`. The exact module-load
commands are **site-specific** — the public repo does not document any particular HPC site's
module system, and neither does this guide try to invent one for a cluster you don't have.

## 3. Building the multi-GPU binary (`diag-gpu-mpi`) — requires a fork + two manual patches

**This is not reproducible from `r-ccs-cms/sbd`'s public `main` alone.** The multi-rank,
NCCL/cuBLAS-accelerated path lives in a separate fork:

```bash
git clone --branch non-cuda-aware-mpi https://github.com/rwakizaka/sbd.git sbd_mpi
```

Two source patches must be applied to that fork's `apps/.../main.cc` (or your own vendored copy
built against its headers) **before** building — neither is upstream anywhere:

### Patch A — NCCL warm-up buffer size fix

In `include/sbd/chemistry/tpb/sbdiag.h`, the warm-up all-reduce allocates a full-width device
vector; on ranks holding a differently-sized slice, this deadlocks whenever the determinant count
isn't evenly divisible by `bdet_comm_size`. Fix: shrink the warm-up buffer to length 1.

```cpp
// before:
thrust::device_vector<double> A(W.size(), 0.0)
// after:
thrust::device_vector<double> A(1, 0.0)
```

### Patch B — undefined `cobits` in the `SBD_PREFECT` carryover block

The fork's `SBD_PREFECT`-gated carryover-writing block references an identifier, `cobits`, that is
declared nowhere in the file. The real per-spin carryover data lives in two vectors that already
exist elsewhere in the same function, `co_adet` and `co_bdet`. Rewrite that block to write both
spins to `carryover.bin` (alpha) and `carryover_b.bin` (beta) instead of the broken single-file
version. The exact rewritten C++ block (safe to paste in as-is, matching the byte layout your
Python reader expects — packed bitstrings, big-endian bit order, `(norb+7)//8` bytes per
configuration) is available on request; it is mechanical (variable substitution + a loop over two
spins), not a design decision, so there is nothing else to choose here.

### Build (CMake, NVHPC/Thrust preset)

```bash
module load <your site's gcc>          # e.g. gcc/14
module load <your site's nvhpc>        # e.g. nvhpc/26.5
module load <your site's hpcx>         # AFTER nvhpc, so its mpic++ takes PATH precedence
                                        # over nvhpc's own bundled hpcx copy

cmake -S . -B build/thrust --toolchain cmake/toolchains/nvhpc.cmake \
  -DSBD_GPU_BACKEND=thrust -DSBD_GPU_ARCH=<your GPU's compute capability, e.g. cc100> \
  -DSBD_USE_NCCL=ON -DSBD_USE_CUBLAS=ON \
  -DSBD_USE_RANK_DISTRIBUTION=ON -DSBD_USE_BLOCK_RANK_DISTRIBUTION=ON \
  -DCMAKE_CXX_FLAGS="-DSBD_PREFECT -DSBD_NON_CUDA_AWARE_MPI -I<path to nccl.h's directory>" \
  -DCMAKE_EXE_LINKER_FLAGS="-L<nccl lib dir> -lnccl -L<cublas lib dir> -lcublas -cudalib=cublas"

cmake --build build/thrust --target tpb_diag
cp build/thrust/apps/chemistry_tpb_selected_basis_diagonalization/diag ./diag-gpu-mpi
```

Add `-D_UHF` to `CMAKE_CXX_FLAGS` for the unrestricted variant (`diag-gpu_uhf-mpi`), which also
needs `--bdetfile` support at runtime (see step 4).

**Verify the link** before trusting the binary:

```bash
ldd ./diag-gpu-mpi | grep -iE 'nccl|mpi|cublas'
```

You should see `libnccl.so.2` and `libcublas` resolved, and `libmpi.so.40` pointing at your site's
*standalone* MPI module (not a bundled copy inside the CUDA toolkit) — mixing the two MPI stacks is
a documented real failure mode (NCCL bootstrap ring deadlocks at 64+ ranks), not a hypothetical
concern.

### Optional: a real correctness/livelock fix worth carrying forward

Under `-gpu=mem:unified`, `GenerateExcitation` in `include/sbd/chemistry/tpb/helper.h` reallocates
two small scratch vectors on the heap inside an OpenMP-parallel loop; at scale (tens of thousands
of iterations × many threads), NVHPC's managed-memory pool allocator's lock contention can livelock
(a real, filed issue: [r-ccs-cms/sbd#80](https://github.com/r-ccs-cms/sbd/issues/80)). Fix: hoist
the two vectors outside the parallel loop, one allocation per thread, reused via the existing
`.clear()` call at the top of each use. This is correctness-preserving, not a workaround —
`OrbitalDifference` already clears both vectors before writing, so reuse is safe.

## 4. Running it

- **Use `mpirun` or `srun --mpi=pmix`, matching whatever MPI stack you actually linked against.**
  A mismatched launcher (e.g. a Slurm `srun` that can't bootstrap a standalone HPC-X OpenMPI build)
  can hang with 0% GPU utilization and no error message — check this first if a run seems stuck
  rather than assuming the binary itself is broken.
- The rank product `task_comm_size × adet_comm_size × bdet_comm_size` must equal your total MPI
  world size (`nodes × ranks-per-node`). Pick `adet_comm_size`/`bdet_comm_size` as close to a
  square grid as your node count allows.
- GPU binding is typically automatic inside the binary itself
  (`cudaSetDevice(mpi_rank % cudaGetDeviceCount())`) — don't also force device selection via your
  launcher's own GPU-binding flags unless you've confirmed the binary doesn't already do it; doing
  both can conflict.
- Real invocation shape:
  ```bash
  mpirun -np <N> ./diag-gpu-mpi \
    --task_comm_size 1 --adet_comm_size <a> --bdet_comm_size <b> \
    --block 20 --iteration 5 --tolerance 0.01 \
    --adetfile AlphaDets.bin --carryoverfile carryover.txt
  ```

## 5. What is and isn't yours to reproduce independently

- The CPU/single-GPU path (§2) is 100% public, upstream, and Apache-licensed — clone it and build
  it under your own account with no dependency on anyone else's setup.
- The multi-GPU path (§3) requires a third party's fork (`rwakizaka/sbd`) plus two source patches
  that exist **only as hand-applied edits**, not committed anywhere public at the time of writing.
  You are reproducing documented, mechanical fixes (not privileged information), but you do need
  to apply them yourself — there is no single "clone and build" command that gets you there.
- Nothing in either path requires access to anyone else's account, home directory, or pre-built
  binary. If someone is asking for read access to an existing compiled binary instead of building
  their own, redirect them to this guide — it's the cleaner, more portable path, and it gives them
  a binary tied to their own account and toolchain rather than a copy of someone else's.
