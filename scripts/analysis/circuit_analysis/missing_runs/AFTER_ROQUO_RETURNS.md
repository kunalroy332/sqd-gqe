# Checklist for when roquo is back (maintenance ends 2026-10-16 12:00 JST)

Everything below is deferred on purpose. The paper, the HTML and the checkers already state each limitation; this file says what to do and what to delete afterwards.

## 1. Missing QPU time (training, first campaign, LUCJ)
Only the 29 fresh-shot inference jobs have IBM-billed QPU time (15,798 s). The Stage-2 training, first-campaign and LUCJ job ids are not stored in the repository.
- On roquo: `sbatch scripts/analysis/circuit_analysis/fetch_on_cluster.sbatch` (scans the logs with `collect_job_ids.py`, writes `circuit_analysis/ids.json`, `qpu_usage.json`, `kobe_calibration.json`).
  Check the per-file id counts it prints; the id patterns in the old logs were not verified.
- Copy the JSONs to the laptop, then `python3 scripts/analysis/circuit_analysis/make_cost_section.py && python3 scripts/analysis/circuit_analysis/make_html.py`, recompile.
- The "Total" QPU cell in the inference-shots table stays "--" until all three groups are present; fill it in `make_cost_section.py` once they are.
- IBM may no longer return old jobs; the script lists those as `missing`. Report the totals as "retrieved for N of M jobs".

## 2. LUCJ depth and two-qubit gates "as submitted"
Table `tab:lucj-opt-compare` is our own transpilation estimate, not read from the hardware job. Two ways to get the real numbers:
- Now, from a terminal with the cluster account's `IBM_API_KEY`/`IBM_CRN` exported:
  `/tmp/ibmvenv/bin/python scripts/analysis/circuit_analysis/lucj_from_ibm.py --since 2026-09-01` lists the single-circuit, 100,000-shot Kobe jobs and measures them
  (depth = `circuit.depth()`, two-qubit gates = `num_nonlocal_gates()`, the same definitions as the GQE numbers, verified on 10 of 10 stored circuits). Assign jobs to molecules by qubit count and creation order (two targets have 56 qubits; confirm by hand).
- Or on roquo: take the job ids from the LUCJ logs (`infer_all_lucj_baseline*`) or from `IBM_SAVE_DIR/<molecule>_opt1/ibm_kobe_shots100000/<job>/circuits.qpy` and run `lucj_from_ibm.py --job-ids ids.txt`.
Then update, in this order: `tab:lucj-opt-compare` (add as-submitted columns), the ratios in the LUCJ paragraph (6-10x shallower, 36-45x fewer two-qubit gates; Yuri's ~20x depth figure
should be re-derived from these), the abstract and conclusion, `LUCJ` in `check_prose.py`, then rerun the three checkers.

## 3. Prose figures
`check_prose.py` recomputes the numbers quoted in the running text (36 claims) and requires them to occur in `paper.tex`. Run it after every data change; a "STALE?" line means the prose or the claim definition needs a look.
Last run: 36 of 36 match. Full set of checks: `check_numbers.py` (2,245 table/HTML checks), `check_consistency.py` (paper vs HTML summary), `check_prose.py`.

## 4. Unfinished results: when they come back, delete the placeholders
Unfinished: 3 recoveries (`2h2o_m_no29_a1-70q_trained__best`, `2h2o_s22_a1-46q_trained__merged`, `h2o_m_no23_a5-70q_trained__oldsecond`; jobs 181726-181728 are queued)
and 4 runs (`sno32_a1-46q_trained`, `h2o_s19_a5-46q_trained`, `llno28_a1-70q_trained`, `2h2o_m_no29_a5-70q_trained`; chain in this folder).
When their results are in `circuit_analysis/`:
1. Copy results per the README in this folder; `python3 make_tables.py` etc. regenerate every table.
2. Delete `circuit_analysis/partial_recoveries.json` (the partial/upper-bound values). The dagger and "pend." text then disappears from tables by itself because `done` becomes true.
3. Remove the four entries from `MISSING_RUNS` in `tables_data.py` (the "queued" rows and matrix cells then become "done").
4. Edit the prose that describes the placeholders:
   - the "SnOH3-me-2h2o M (58q): the best circuit's recovery was interrupted ... iteration 16 of 25" sentence in the DMRG discussion;
   - the dagger/"pend." sentence in `PENDING` (`captions.py`) and the "queued" sentences (`detail_X` caption, run-matrix caption, HTML note, HTML limitations bullet);
   - the conclusion sentence "four runs and three recoveries are still to be completed" and the abstract if it mentions them;
   - the 2h2o-M numbers (3.7 mHa partial, 41.1 and 21.3 mHa) and the list of "lowest errors at the time of writing".
5. Rerun `check_prose.py`; it will flag every quoted number that moved.
6. Update the cost section automatically (runs 29 -> 33, shots +4,000,000, wall time) and rerun `fetch_on_cluster.sbatch` so the new jobs get QPU time.

## 5. Also pending
- Rotate the IBM API key that was pasted into the chat.
- Verify the training figures (iterations, hours, per-stage times in `cost_numbers.py`) against the Stage-1/2 logs on roquo; check "75x shallower after warm-up" with live (not offline) depth.
- Confirm the first-campaign shot count (assumed 29 runs x 10 circuits x 100,000) from its logs.
