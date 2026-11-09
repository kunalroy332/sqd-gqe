# Reproducing the study

This file lists, for each result of the manuscript, what regenerates it, what it needs, and the command. Three levels:

| Level | Needs | Reproduces |
|---|---|---|
| **A. Analysis** | A laptop, `pip install -e ".[dev]"` | Every table, figure number and HTML report built from the stored results in `data/` |
| **B. Recovery** | A GPU node, the SBD binary (`docs/sbd_build_guide.md`) | The five recovery energies of a run from its stored circuits and counts |
| **C. Training and sampling** | An IBM Quantum account with access to `ibm_kobe`, a GPU cluster | The trained policies and the hardware measurements |

Level C costs real QPU time (about 650 QPU minutes for the four main training runs, 1,175 for the tuned reward, and 12.8 QPU hours
for the two inference sets, including repeated draws) and cannot be rerun exactly: the device and its noise change. The measured counts of the study are therefore
not recomputed here; only the hardware-free steps are.

## A. Analysis (laptop, minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q                                      # 212 tests
cd scripts/analysis/circuit_analysis
python three_way_load.py                                  # pairs the two configurations (writes ../../../data/R3.pkl)
python headline_numbers.py                                # the numbers of the abstract, results and conclusion
python make_html_sets.py hf && python make_html_sets.py tuned      # the two HTML reports (docs/reports/)
python divergence_tables.py <paper.tex>                   # only if you have the manuscript source
```

`headline_numbers.py` prints, among others (the values the manuscript quotes):

| Quantity | Value |
|---|---|
| Hartree-Fock-penalty runs with a recovered best circuit | 43 of 44 |
| Median deviation from DMRG of the best circuit | 2.85 mHa (13 runs within 1 mHa, 28 within 5, 42 within 15) |
| Untrained controls: number, median deviation, runs within 15 mHa | 8, 105.5 mHa, 1 |
| Selected circuits at 56 to 70 qubits: depth, two-qubit gates | 44 to 134, 64 to 160 |
| Their ratio to the LUCJ circuit: depth, two-qubit gates | 4.0 to 12.2, 34.3 to 72.4 |
| Paired comparison (43 test cases): HF penalty, tuned reward, tuned lower in | 2.85, 2.40 mHa, 26 cases |
| Tuned-reward best circuits above 30 mHa | 41.1, 41.1 and 198.3 mHa |

The deviation is the signed `E_SQD - E_DMRG`; the DMRG energies are in `molecules/large/DMRG/dmrg_results.csv`.

## B. Recovery from the stored circuits (one GPU)

The sampled gate-token sequences of every run are in `data/{hf,tuned}/data/`. Rebuild a circuit, sample it (hardware or simulator) and
recover the energy with:

```bash
export INFER_ROOT=/path/to/work/inference_hf
python scripts/analysis/circuit_analysis/resample_raw_counts.py --cache <cache>.json   # IBM hardware: IBM_API_KEY, IBM_CRN
python scripts/analysis/circuit_analysis/rerank_circuits.py     --tag <tag>
python scripts/analysis/circuit_analysis/select_new.py          --tag <tag>
python scripts/analysis/circuit_analysis/rerecover_target.py    --tag <tag> --target best --idx 0
```

The recovery uses the SBD diagonalizer (`--diag-backend sbd`). The recovery size is `D = 20,000` strings per spin sector (the closed-shell
product has up to `D^2` determinants) and 25 iterations; the ranking uses `D = 200`.

## C. Training and inference (cluster and IBM Quantum)

```bash
cp .env.example .env.local        # IBM_API_KEY and IBM_CRN; never commit it
python scripts/analysis/training2/make_jobs.py        <outdir>     # Stage 1 (ffsim) and Stage 2 (no penalty); edit the paths at the top
python scripts/analysis/training2/make_jobs_penalty.py <outdir>    # Stage 2 with lambda_HF = 0.05 (the main configuration)
python scripts/analysis/training2/make_jobs_tuned.py   <outdir>    # Stage 2 with the tuned reward
bash <outdir>/submit_all.sh
```

The recipe (seed 1234; `ngates` 28 at 46 qubits and 44 at 70; vocabulary 110; 150 + 150 iterations; 20 circuits per iteration;
10,000 shots per circuit; recovery size 20,000 in Stage 1 and 40,000 in Stage 2) is written into the generated job scripts and, for the study,
is stored in the first line of every tracking file as `argv` and `args`. The reward weights of the two configurations are listed in the manuscript (Methods, "The reward and its weights") and are set by
`make_jobs_penalty.py` (lambda_HF = 0.05) and `make_jobs_tuned.py` (lambda_HF = 0.07, lambda_spin = 0.77, c_H = 0.26, lambda_exc = 3.9, lambda_cov = 7.2).

The configuration of the inference campaign is in `scripts/analysis/training2/inference/` (`generate_sequences.py` draws the ten
sequences of a checkpoint; `make_chain.py` writes the chain of Slurm jobs that samples, ranks and recovers them).

## What is not reproducible from this repository

* The hardware counts: the device, its calibration and its queue differ from run to run (`data/ibm/kobe_calibration.json` records the snapshots).
* The trained checkpoints are not included (about 700 MB). Retraining from the seed gives a different policy because the hardware stage is stochastic.
* DMRG: the reference energies are included (`molecules/large/DMRG/`); the calculation itself needs the cluster and an external DMRG code.
