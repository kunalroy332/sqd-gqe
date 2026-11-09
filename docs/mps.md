# cuQuantum MPS samplers: usage and recorded measurements

`tutorial/04_mps_backends.ipynb` runs one circuit through cuQuantum (via sqd-gqe and directly),
quimb and Qiskit Aer, and compares them with an exact `ffsim` reference.

## Sampler selection

| `sampler.kind` | State representation | Determinant selection |
|---|---|---|
| `ffsim` | Exact CI vector on CPU | Shots |
| `topk` | Exact CI vector on CPU | Largest probabilities |
| `mps` | Truncated MPS on GPU | Shots; SQD postselection or configuration recovery |
| `mps_topk` | Truncated MPS on GPU | Beam search restricted to valid particle numbers |

With `sampler.p_readout > 0` or a `*_noisy_sim` backend, `mps` uses `NoisyMpsSampler`,
sharing `NoisyFfsimSampler`'s per-bit noise transform. Gate noise needs a noisy backend;
T1 noise also needs `sampler.amplitude_damping=true`. `last_valid_fraction` is measured
after noise. Top-K samplers reject noise settings. The transform supports up to
64 spatial orbitals (128 qubits), subject to the chosen solver's limits.

MPS avoids allocating the full CI vector. Accuracy depends on `sampler.mps_max_bond`
(chi), the singular-value cutoffs and the circuit. All pool operators must provide
`qiskit_gate_fn`. The default dtype is `complex64`.

### Default settings

The defaults are `mps_gauge_option=simple`, `mps_dtype=complex64` and cutoffs of `1e-5`.
With cuQuantum's own `free` gauge the singular values are not Schmidt values, so a
fixed chi truncates badly and the cutoffs drop the wrong weight. Measured on ROQUO GB200
(jobs 179261, 179343 and 179588), with the same transpiled circuits throughout:

| | `free`, complex64, 1e-5 (old default) | `simple`, complex64, 1e-5 (default) | `simple`, complex128, rel 1e-8 |
|---|---|---|---|
| 28q N2 lucj2, fidelity with the exact state, chi=64 | 0.851 | 0.9995 | 0.9995 |
| 28q N2, `mps_topk` recall vs exact top-K, chi=128 (6 circuits) | 0.52 | 0.997 | 0.998 |
| 28q N2, `mps_topk` energy vs exact top-K, chi=128 (5 of 6 circuits) | -7 to -85 mHa | 0.0 mHa | 0.0 mHa |
| 38q Sn(OH)3-Me M mixed10, fidelity with the reference, chi=512 | 0.71 | 1.0000 | reference |
| 38q Sn(OH)3-Me M mixed10, MPS build time, chi=512 | 236 s | 24 s | 121 s |

`complex128` with the `free` gauge does not fix it (0.9145 at 28q lucj2, chi=64), and
`canonical_center` changes no fidelity. At 38q the reference is the `simple`, complex128 state at chi=512, which chi=256 already matches to 1.0000. The sixth 28q circuit (`rand1`, a state spread
thinly over many determinants) gives +279 mHa at chi>=128 under both `simple` settings
alike, so it is not a settings effect.

Both top-K samplers use `sampler.top_k`, defaulting to the effective `max_dim`
(predicted when `dynamic_resources=true`). They draw no shots. They discard probabilities
at or below `1e-12` of the full state norm, and then apply `sampler.min_prob`. For
comparisons, use `sqd.enlarge_method=None` and `sqd.recover_iterations=0`. The factory
enforces these settings for `mps_topk`.

`sampler.min_prob` (default `auto`, which is `1 / sampler.shots`) drops determinants a run
of `sampler.shots` would not be expected to draw even once. Kept determinants get the count
those shots would give, `|c|^2 * shots`. If nothing clears the floor, only the largest is
kept. A number sets the floor directly, for example `3e-5` for three expected counts at
100,000 shots, and `null` keeps every weighted determinant with counts rescaled to `shots`.
Without the floor a trained policy learns to spread its state thin: the top K still score
well, but no real run could sample them. On `dev` such a 32q policy looked fine at 32q
and collapsed at 46q (four of six circuits 0.5-2.7 Ha above RHF), while one trained with
the floor transferred and beat random circuits by about 20 mHa. Set `sampler.shots` to
the shot count a hardware run would take, since it defines the floor. Only the factor 1
has been measured.

`sampler.mps_beam_width` defaults to `4 * top_k`. The `provably exact` log and
`last_topk_exact` certify selection from the **truncated MPS**, not the exact circuit
state. Increase beam width when the result is `beam-limited`. GPU contraction and floating
point arithmetic can still change selected determinants between runs.

`last_captured_weight` is the returned probability mass relative to the whole MPS,
including incorrect particle-number sectors. A low value can reflect leakage, a diffuse
distribution or an incomplete beam search. `mps.last_valid_fraction` measures the valid
shot fraction; `mps_topk.last_valid_fraction=1` only means its returned strings are valid.

`sqd.max_dim` caps selected pairs before diagonalization. The CPU `sqd` solver uses the
product of unique alpha and beta strings. The current RHF-only SBD backend folds both
columns into one list, caps it at `round(sqrt(max_dim))`, and solves its square product
space (`max_dim=1000` allows at most 32 half-strings, or 1024 product states). Thus top-K
pair count, recall and the solver's actual dimension measure different things.

## Running on ROQUO

Submit from a checkout installed in your virtualenv (`pip install -e .`).
Both Slurm scripts source `slurm/mps_runtime_env.sh`, which
selects `$VENV` or the first available `.venv_sqd`, `.venv_arm`, `.venv`, checks the import
path, and installs cuQuantum/CuPy under a lock if they cannot be imported. The default
package is `cuquantum-python-cu13` with `cupy-cuda13x`; set `CUQUANTUM_PKG` to a compatible
version (a `*-cu12` package selects `cupy-cuda12x`). Existing imports bypass installation.
Keep the cuQuantum library and `cuquantum-python` at the same CUDA major version. W&B defaults to offline.

For `sqd.diag_backend=sbd`, supply an externally built binary compatible with the node's
MPI, NVHPC and system libraries; see the [SBD build guide](sbd_build_guide.md).
Its `SBD_PREFECT` output must include `davidson_energy.txt`, `occ_a.txt` and `carryover.bin`.
`slurm/sbd_runtime_env.sh` discovers MPI/NVHPC under `/opt`; set `SBD_NVHPC_ROOT` to the
release used to build the binary, or supply extra directories in `SBD_LD_LIBRARY_PATH`.
The latter is prepended to `LD_LIBRARY_PATH` only for the native SBD subprocess.

The `snoh3_s`, `snoh3_m` and `snoh3_l` Hydra configs link to the existing S/M/L molecule YAMLs (32q, 38q, 46q).
Replace the account and binary path below. The training script defaults to `mps`:

```bash
sbatch --account=<project> slurm/train_mps.sh molecule=snoh3_m 'reference_keys=[]' \
    sqd.diag_backend=sbd sqd.sbd_binary_path=/absolute/path/to/diag-gpu \
    sqd.recover_iterations=0 sqd.max_dim=1000 sampler.mps_max_bond=256 trainer.max_iters=5
```

To train with amplitude selection, also pass `sampler.kind=mps_topk` and
`sqd.enlarge_method=None`. `None` is the literal string used by this configuration.
There is no `sqd.refine=false` option: default GEVP refinement is skipped automatically
when the system exceeds the full-CI dimension limit. SBD does not return CI amplitudes
for GEVP/UFD refinement; use its supported `tp_ufd` path if refinement is needed.

Compare bond dimensions on the three circuits used in the tables:

```bash
sbatch --account=<project> --time=12:00:00 slurm/mps_bond_sweep.sh \
    --molecule snoh3_m --sampler-kind mps_topk --with-exact \
    --sequences mixed10 lucj1 lucj2 --bonds 64 128 256 512 1024 \
    --beam-width 32000 --max-dim 1000 --sbd-binary-path /absolute/path/to/diag-gpu
```

For 46q, change to `--molecule snoh3_l` and omit `--with-exact`: its CI vector has
`C(23,16)^2 = 60,101,954,649` entries (about 0.96 TB at `complex128`, before workspace).
For a smaller GPU smoke run, use `--molecule n2 --diag-backend sqd --bonds 64`.
For shot-noise measurements, use `--sampler-kind mps --seeds 4`; the sweep disables
recovery and disables completion for `mps_topk`. Reports are saved after each measurement
to `docs/report/mps_bond_sweep/<molecule>.json`; use `--out` to keep runs separately.

Summary rows show seed means, success counts (`ok/all`) and differences between chi means.
Non-finite energies are recorded as failures; incomplete seed groups are excluded from
ranking. Seed ranges are reported separately for each chi. The last chi step reports mean
absolute energy changes and checks circuit-pair gap changes against their reference gaps,
so opposite shifts cannot cancel and common energy offsets do not imply a ranking change.
Reference ordering reversals are reported per circuit pair using matching seed IDs,
so a distant third circuit cannot hide uncertainty in a close pair.

## Recorded measurements

The tables below retain the original branch reports; they have **not been rerun after
the sampler and SBD fixes**, and they used the old `free` gauge. Those reports did not record a source revision. Solver
subspace construction and top-K filtering/ordering affect energies and ranking, so the
tables are historical observations rather than validation of the current implementation.
The commands above use the current interface.

### 38q training cost and gate cap

ROQUO GB200, 2026-09-16; Sn(OH)3-Me M tier (`snoh3_m`), 19 orbitals, (12,12) electrons,
10 circuits/rollout, chi=256, 100,000 shots, `max_dim=1000`, recovery off, no reference
circuits. CCSD(T): -479.3019447232744 Ha.

| Configuration | Sample / circuit | Diagonalize / circuit | Rollout (10 circuits) |
|---|---|---|---|
| ffsim + CPU `solve_sci` | 613 s | 3463 s | ~11 h (extrapolated) |
| cuQuantum MPS + CPU `solve_sci` | ~32 s | 3463 s | ~10 h |
| cuQuantum MPS + `sbd` GPU | 32 s | 2.8 s | **521 s** (mean of 6) |

Both sampling and diagonalization were bottlenecks. Per-circuit MPS cost varied strongly
with gate count; one 265 s circuit accounted for 265 of 317 s sampling in a rollout.

| Transpiled gates | Contract + sample | Particle-number-valid |
|---|---|---|
| 118-136 (local ops only) | 1.6-1.7 s | 1.000 |
| 2752-2766 | 4.0 s | 1.000 |
| 6161 | 26.5 s | 1.000 |
| 8778 | 28.8 s | 1.000 |
| 18080 / 18106 | 131 / 142 s | 0.527 / 0.539 |
| 28168 | 265.0 s | 0.668 |

`sampler.mps_max_gates=10000` skipped 9/60 circuits in the recorded comparison:

| | Best energy | vs CCSD(T) | Best circuit's subspace | Rollout |
|---|---|---|---|---|
| no cap | -479.264252 | **+37.7 mHa** | 1000 | 521 s |
| cap at 10000 | -479.113617 | **+188.3 mHa** | **4** | 91 s |

The observed speedup was 5.7x with a 150 mHa loss; it does not establish a safe cap.
The cap remains off by default. Skips return an empty subspace (+inf energy). The current
objective replaces non-finite energies with the worst finite energy plus 10 Ha before
normalization; an entirely failed group contributes zero loss. Rerun capped training
with this handling before attributing the historical collapse solely to the gate cap.

### 38q shot noise

Exact ffsim sampling, 100,000 shots, four seeds, `max_dim=1000`, SBD:

| circuit | seed mean | spread across seeds |
|---|---|---|
| lucj1 | -479.262400 | 0.82 mHa |
| mixed10 | -479.261672 | 4.01 mHa |
| lucj2 | -479.260126 | 9.65 mHa |

The means span 2.27 mHa; seed variation changed their ordering. Chi=1024 MPS sampling
gave spreads of 2.30, 6.54 and 4.19 mHa against a 2.72 mHa difference between circuits.
These predate solver corrections, which need not cancel across seeds; remeasure the
noise floor for current runs. GRPO/GSPO normalize energy differences within a group and
also supervise its winner, so incorrect ordering affects training. Top-K removes shot
noise, while truncation, beam limits and numerical variation still need assessment.

### 38q amplitude selection

K=1000, beam width 32000, SBD, completion/recovery off. Recall compares returned pairs
with exact top-K selection; delta energy is in mHa relative to that selection's SQD
energy. Single-determinant `local10` was excluded because zero-weight padding makes
recall uninformative. It is now filtered out by both samplers.

| chi | recall mixed10 | lucj1 | lucj2 | ΔE mixed10 | lucj1 | lucj2 |
|---|---|---|---|---|---|---|
| 64 | 0.000 | 0.000 | 0.000 | +2628 | +2525 | +3707 |
| 128 | 0.191 | 0.213 | 0.039 | +62.7 | +31.9 | +1094 |
| 256 | 0.278 | 0.241 | 0.260 | **-22.6** | **-26.6** | **-13.2** |
| 512 | 0.331 | 0.325 | 0.276 | **-25.9** | **-34.5** | **-41.0** |
| 1024 | 0.372 | 0.381 | 0.358 | **-15.2** | **-32.1** | **-29.3** |

The recorded MPS selections had lower energies despite low recall. Pair overlap alone
does not determine the product subspace used by the solver; this observation does not
establish that truncation improves selection under the current SBD dimension cap.

| chi | tau | ordering (best first) | spread |
|---|---|---|---|
| exact | +1.00 | mixed10 < lucj2 < lucj1 | 18.3 mHa |
| 64 | -0.33 | lucj1 < mixed10 < lucj2 | 1182 mHa |
| 128 | -0.33 | lucj1 < mixed10 < lucj2 | 1062 mHa |
| 256 | +0.33 | mixed10 < lucj1 < lucj2 | 27.4 mHa |
| 512 | **+1.00** | mixed10 < lucj2 < lucj1 | 9.7 mHa |
| 1024 | +0.33 | mixed10 < lucj1 < lucj2 | 3.9 mHa |

Chi=512 matched the recorded exact ordering; chi=1024 inverted `lucj1`/`lucj2`, separated
by 0.26 mHa in the reference. The last doubling shifted energies by 8.2 mHa on average.
Completion reduced the exact-selection spread from 18.3 to 1.4 mHa in that experiment.
Repeats reportedly differed by about 0.1 mHa at chi>=256 and by hundreds of mHa at
chi<=128, so absence of shot noise did not imply reproducibility.

### 46q convergence and cost

Sn(OH)3-Me L tier (`snoh3_l`), 23 orbitals, (16,16) electrons, RHF -479.121325 Ha.
K=1000, beam width 32000, SBD, completion/recovery off; no exact-state reference.
`mixed10` has nine local operators plus one LUCJ block, `lucj1` one LUCJ block and
`lucj2` two; the recorded circuits transpiled to about 18,500 gates.

| chi | mixed10 | lucj1 | lucj2 | spread | s/circuit |
|---|---|---|---|---|---|
| 128 | -477.193782 | -477.294974 | -476.074285 | 1220.7 mHa | 77-163 |
| 256 | -479.221184 | -479.202663 | -479.149885 | 71.3 mHa | 159-365 |
| 512 | -479.261421 | -479.258661 | -479.204361 | 57.1 mHa | 361-853 |
| 1024 | -479.270968 | -479.269800 | -479.236131 | 34.8 mHa | 886-2149 |

Successive determinant-set overlaps:

| circuit | 128->256 | 256->512 | 512->1024 |
|---|---|---|---|
| mixed10 | 0.000 | 0.399 | 0.505 |
| lucj1 | 0.000 | 0.398 | 0.508 |
| lucj2 | 0.000 | 0.262 | 0.541 |

The last doubling moved energies by -9.5, -11.1 and -31.8 mHa (mean magnitude 17.5)
and replaced about half the pairs: chi=1024 was not converged in this sweep. The
`mixed10`/`lucj1` gap was only 1.17 mHa, despite ordering remaining stable from chi=256.
Completion reduced the spread from 34.8 to 19.5 mHa. At chi=128, valid-only top-K still
gave a poor energy, whereas shot sampling had a valid fraction of 0.20.
`lucj2` took 2149 s at chi=1024, so this accuracy/cost balance needs reassessment before
using it for training; larger chi is not a demonstrated solution for these circuits.
