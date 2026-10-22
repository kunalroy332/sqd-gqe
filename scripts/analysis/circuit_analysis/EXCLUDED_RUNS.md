# Excluded runs

Runs that were analysed but are deliberately left out of the manuscript, the tables, the figures, the cost counts and the Zenodo page.
Their files are kept in `circuit_analysis/excluded/<run>/` (same sub-folders as `circuit_analysis/`), so nothing was deleted.

## `h2o_s19_a1-46q_untrained`: SnOH3-me-h2o S (38q), A1-46q lineage, untrained control

Excluded at the author's request on 2026-10-15, after the question of whether an untrained control reaching the DMRG answer at 38 qubits
weakens the case for training. It is not a data fault: the run completed normally (sampling, ranking and all five recoveries), and
every other run at 32-44 qubits, trained or not, also recovers to within 0.5 mHa, which the manuscript attributes to the 20,000-dimensional
recovery subspace capturing the exact solution whichever circuit seeds it. It is the only untrained control at a 32-44 qubit tier.

DMRG reference -555.108832786 E_h. Recovery: subspace dimension 20,000, 25 iterations.

| Result | Circuit (1-indexed) | Depth (live) | 2q (live) | Valid shots | Energy (E_h) | Error vs DMRG (mHa) |
|---|---|---|---|---|---|---|
| best | 2 | 265 | 780 | 3549 | -555.108826 | 0.01 |
| second | 7 | 278 | 768 | 48 | -555.108826 | 0.01 |
| merged | pooled | | | | -555.108826 | 0.01 |
| oldbest (E+D 1) (= second) | 7 | 278 | 768 | 48 | -555.108826 | 0.01 |
| oldsecond (E+D 2) | 10 | 18107 | 29544 | 4 | -555.108826 | 0.01 |

Sampling job: IBM Runtime `db3ngncvf2bc73ctfv0g` (555 s billed QPU time, created 2026-10-08). Slurm jobs removed from `timing_sacct.psv`:
`ca_`, three `rr_` attempts (two failed on bad nodes, one completed) and the four `rc_..._u_*` recoveries; the removed rows are in
`excluded/h2o_s19_a1-46q_untrained/timing_sacct_rows.psv`.

Effect of the exclusion on the counts: 33 -> 32 fresh-shot runs, 330 -> 320 circuits, 33,000,000 -> 32,000,000 shots, 33 -> 32 fresh-shot
jobs (QPU time 17,659 s -> 17,104 s), 25 -> 24 cross-norb runs.

To restore: move the files back from `excluded/h2o_s19_a1-46q_untrained/` to the matching folders, put the entry in `excluded/h2o_s19_a1-46q_untrained/selection_entry.json`
back into `selection_batch3.json`, the rows in `timing_sacct_rows.psv` back into `timing_sacct.psv`, the job in `qpu_usage_job.json` back
into `qpu_usage.json` (group `fresh`), then rerun `make_tables.py`, `make_cost_section.py`, `make_calibration_table.py`, `make_collage.py`,
`make_html.py` and the three checkers, and update the run counts in `paper.tex`.
