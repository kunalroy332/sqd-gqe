#!/usr/bin/env bash
# Run the four missing fresh-shot runs in parallel on roquo (after the maintenance window).
#   cd /path/to/sqd-gqe-work && bash scripts/analysis/circuit_analysis/missing_runs/submit_missing_runs.sh
# Each run is an independent chain:  ca_<tag> (IBM sampling, 10 circuits x 100,000 shots)
#                                    -> rr_<tag> (40000-dim ranking)  -> fin_<tag> (selection + submits the 5 recovery jobs).
# Needs the IBM credentials of the cluster account (sourced from /path/to/sqd-gqe/.env.local by the ca_ job).
# Re-running is safe: it asks before resubmitting a tag whose outputs already exist.
set -euo pipefail
REPO=/path/to/sqd-gqe-work
SCR=$REPO/scripts/analysis/circuit_analysis
OUT=$REPO/experiments/crossnorb70q_infer
TAGS=(sno32_a1-46q_trained h2o_s19_a5-46q_trained llno28_a1-70q_trained 2h2o_m_no29_a5-70q_trained)
GEN=$REPO/experiments/production_final/logs/missing_runs_sbatch
mkdir -p "$GEN"
cd "$REPO"

for tag in "${TAGS[@]}"; do
  if [ -f "$OUT/circuit_analysis/$tag.json" ] || [ -f "$OUT/circuit_analysis_rerank/$tag.json" ]; then
    echo "SKIP $tag: outputs already exist (delete them to redo)"; continue
  fi
  [ -f "$OUT/cache/crossnorb70q_${tag}_sample.json" ] || { echo "MISSING first-campaign cache for $tag"; exit 1; }
  # generate the ca_ and rr_ job files with the same generators used for the other 29 runs
  echo "crossnorb70q_${tag}_sub20000_job0.json" > "$GEN/$tag.list"
  python3 "$SCR/make_sbatch.py" "$GEN/$tag.list" "$GEN/ca" >/dev/null
  python3 "$SCR/make_rerank_sbatch.py" "$GEN/rr" "$tag" >/dev/null
  # no maintenance deadline any more: give the jobs full time, no --time-min squeeze
  sed -i 's/^#SBATCH --time=.*/#SBATCH --time=24:00:00/; /^#SBATCH --time-min/d' "$GEN/rr/rr_$tag.sbatch"
  sed -i 's/^#SBATCH --time=.*/#SBATCH --time=24:00:00/' "$GEN/ca/ca_$tag.sbatch"
  J1=$(sbatch --parsable "$GEN/ca/ca_$tag.sbatch")
  J2=$(sbatch --parsable --dependency=afterok:$J1 "$GEN/rr/rr_$tag.sbatch")
  J3=$(sbatch --parsable --dependency=afterok:$J2 -J "fin_$tag" --export=ALL,TAG="$tag" "$SCR/missing_runs/finish_run.sbatch")
  echo "$tag: ca=$J1 rr=$J2 fin=$J3"
done
echo "squeue -u \$USER -o '%.10i %.30j %.8T %R'   # follow progress"
