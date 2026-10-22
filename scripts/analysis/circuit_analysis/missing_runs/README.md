# Four runs that are missing from the fresh-shot tables

The paper and the Zenodo HTML list four (molecule, policy) runs as **queued**. Their first-campaign samples exist
(`experiments/crossnorb70q_infer/cache/crossnorb70q_<tag>_sample.json`, local copy in `circuit_analysis/raw_archive/cache/`), but the fresh-shot
pipeline (re-sample, rank, recover) was not run for them. The cluster (roquo) is in maintenance until 2026-10-16 12:00 JST.

| Run (tag) | Molecule | Qubits | Policy | What the same molecule already has |
|---|---|---|---|---|
| `sno32_a1-46q_trained` | SnOH3-me S | 32 | A1, trained at 46q | A1-70q, A5-46q, A5-70q trained |
| `h2o_s19_a5-46q_trained` | SnOH3-me-h2o S | 38 | A5, trained at 46q | A1-46q trained and untrained |
| `llno28_a1-70q_trained` | SnOH3-me LL | 56 | A1, trained at 70q | A5-70q trained, A1-70q untrained |
| `2h2o_m_no29_a5-70q_trained` | SnOH3-me-2h2o M | 58 | A5, trained at 70q | A1-70q trained |

(For `2h2o_m_no29` the request said "A5 trained"; A5-70q is used because the A1 run on this molecule is the 70q lineage. Add `2h2o_m_no29_a5-46q_trained`
to `TAGS` in the submit script only if the 46q lineage is wanted too; it needs a first-campaign cache that this repository does not contain.)

## What the scripts do

`submit_missing_runs.sh` submits **four independent chains in parallel** (one per tag):

1. `ca_<tag>`: re-sample the 10 stored gate sequences on `ibm_kobe`, 100,000 shots each (`resample_raw_counts.py`); saves `counts.json` and `circuits.qpy`.
   Needs the IBM credentials that the cluster account already uses (`/path/to/sqd-gqe/.env.local`, variables `IBM_API_KEY`, `IBM_CRN`). 12 h limit is raised to 24 h.
2. `rr_<tag>` (after `ca_` succeeds): 40,000-dimensional ranking energy of every circuit (`rerank_circuits.py`).
3. `fin_<tag>` (after `rr_` succeeds): `select_new.py` picks best, second, merged and the two E+D circuits, writes the five recovery job files and submits them
   (2 GPUs, 36 h each, no deadline). `rerecover_target.py` runs the 25-iteration recovery for each.

Everything else (bad-node exclusion `--exclude=qr01n08,qr02n11,qr04n16`) comes from the same generators as the other 29 runs. The three nodes had broken GPU stacks before the maintenance;
re-check with a short test job before trusting them, and edit the `--exclude` line in `make_sbatch.py`, `make_rerank_sbatch.py` and `select_new.py` if they were repaired.

## Run it

```bash
ssh roquo
cd /path/to/sqd-gqe-work
git pull                      # branch feat/snoh3-variant-launcher has these scripts
bash scripts/analysis/circuit_analysis/missing_runs/submit_missing_runs.sh
squeue -u $USER -o '%.10i %.30j %.8T %R'
```

Expected cost per run (from the other 29 runs): sampling 20 min to 3 h (IBM queue), ranking 1-2 min, five recoveries 4 min to about 12 h each (the 58q run is the slowest).
Total IBM usage: 4 runs x 10 circuits x 100,000 shots = 4,000,000 shots, 4 Runtime jobs.

## Bring the results into the paper and the HTML (on the laptop)

```bash
R=/path/to/sqd-gqe-work/experiments/crossnorb70q_infer
for d in circuit_analysis circuit_analysis_rerank circuit_analysis_rerecovery circuit_analysis_ibm_jobs circuit_analysis_archive; do
  rsync -a roquo:$R/$d/ circuit_analysis/raw_archive/$d/ ; done
# per-run files used by the generators:
rsync -a roquo:$R/circuit_analysis/ circuit_analysis/data/
rsync -a roquo:$R/circuit_analysis_rerank/ circuit_analysis/rerank/
rsync -a roquo:$R/circuit_analysis_rerecovery/ circuit_analysis/rerecovery/
```
Then (1) run `select_new.py` once on the four new rerank files and save its output as `circuit_analysis/selection_batch11.json` (`tables_data.load` reads every `selection_batch*.json`);
(2) run `save_manifest.py --tags ...` and `circuit_connectivity.py --tags ...` as Slurm jobs and copy `circuit_analysis_archive/` and `circuit_analysis_connectivity/` back (`circuit_analysis/connectivity/`);
(3) update `timing_sacct.psv` (`sacct -X -P -n --format=JobID,JobName%60,State,Elapsed,ElapsedRaw,Start,End,NodeList`);
(4) remove the four entries from `MISSING_RUNS` in `tables_data.py`; (5) `python3 make_tables.py && python3 make_cost_section.py && python3 make_html.py && python3 check_consistency.py`, then recompile.
The cost section counts runs from the data files, so shots, jobs and wall time update by themselves (29 -> 33 runs).
