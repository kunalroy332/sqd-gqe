# Training 2: Stage 1 + Stage 2 with the warm-start fixes and per-iteration tracking

Training 1 (the trainings behind the first manuscript version) is left untouched: its checkpoints, histories and logs stay in
`experiments/production_final/`. Training 2 writes only to `experiments/training2/`.

## What differs from Training 1

| | Training 1 | Training 2 |
|---|---|---|
| Operator vocabulary | 32 in Stage 1, 74 / 110 in Stage 2 | 110 at both stages (`--max-vocab 110`) |
| Token embedding and output head at the Stage-1 to Stage-2 handoff | re-initialised (shape mismatch, `reset 2`) | carried over (`transferred 125, reset 0`) |
| Prompt encoder at the handoff | re-initialised | restored (`--warmstart-encoder`) |
| Iterations | self-stop at 90-132 (Stage 1); 88-150 (Stage 2, several resumed runs) | fixed 150 + 150, no early stop |
| Tracking | loss, error, depth of the best circuit | plus per iteration and per circuit: tokens, operator families, Givens roles, spin-resolved excitations, policy entropy |
| Hardware job data | not kept | `--ibm-save-dir`: raw counts and transpiled circuits of every job |

Everything else is the Training-1 recipe: seed 1234, `ngates` 28 (46q) / 44 (70q), 20 circuits per iteration, SBD GPU recovery (3 iterations),
Stage 1 on the same 14 molecules (`max-dim` 20,000), Stage 2 on Fe(CN)6, Co(NH3)6 and ferrocene at 23 or 35 orbitals (`max-dim` 40,000,
10,000 shots, `ibm_kobe`), depth ceiling 10,000 (46q) / 20,000 then 50,000 (70q Stage 1 / Stage 2), energy gate 100 mHa (500 mHa for the 70q Stage 2).

## Run

```
# on the cluster login node, from the repository root (code and tracking module as in this commit)
python3 scripts/analysis/training2/make_jobs.py experiments/training2
bash experiments/training2/submit_all.sh          # 4 Stage-1 jobs; each Stage-2 job depends on its Stage-1 job (afterok)
```
Stage 2 needs `IBM_API_KEY` and `IBM_CRN` in `/path/to/sqd-gqe/.env.local` (sourced by the job script, never printed). Stage 2 starts with a
pre-flight check that the Stage-1 `_best.pt` has vocabulary 110 and a stored prompt encoder, and stops otherwise.

Outputs in `experiments/training2/`: `out/t2_s{1,2}_<enc>_<size>.json` (history), `ckpt/` (checkpoints), `track/*.jsonl` (see below),
`ibm_jobs/` (hardware jobs), `logs/`. The first line of every `track/*.jsonl` is the run metadata: command line, all arguments, git commit
(if the cluster copy is a checkout), Python and library versions, SHA-256 of `amortized_train.py` and `training_tracking.py`, Slurm job id.

## What is tracked per iteration (`track/*.jsonl`, one line per iteration; `track` key in each history row)

For every one of the 20 sampled circuits (and as means and for the best circuit):

* the token sequence, and its operator families (identity, LUCJ, orbital rotations, Givens, diagonal-Coulomb, number-number);
* the Givens rotations by role relative to the Fermi level: HOMO-LUMO (the only neighbouring pair that changes the Hartree-Fock determinant),
  occupied-occupied, virtual-virtual; the number of gates that can change the Hartree-Fock determinant at all; orbital coverage; token entropy;
  repeats; fraction of identity tokens;
* real transpiled depth, two-qubit gates, idle-qubit fraction, energy;
* the sampled (alpha, beta) pairs: the fraction in the **right spin sector** (n_alpha, n_beta preserved) against the **wrong** one (an electron
  moved between the alpha and the beta channel, or an electron-number error); among the right-sector shots the excitation resolved by spin:
  alpha-only (alpha -> alpha), beta-only (beta -> beta), both, HOMO -> LUMO of alpha, of beta and of both, single and double excitations by spin pair;
  the alpha/beta asymmetry of the excited weight;
* the policy's mean token entropy and effective vocabulary.

Every operator of the pool conserves n_alpha and n_beta separately, so on the noiseless ffsim simulator (Stage 1) the wrong-spin-sector fraction is
exactly zero; on hardware (Stage 2) it measures noise. Stage 1 and Stage 2 use different molecules, so the two stages are compared with the same
definitions of every quantity, not on the same circuits.

## Comparison with Training 1

`compare_trainings.py` (written once the runs have finished) puts the Training-1 histories (`production_final`) next to these for loss, error,
gate-pass rate, depth and two-qubit gates of the best circuit, and shows the tracked quantities, which Training 1 does not have.
