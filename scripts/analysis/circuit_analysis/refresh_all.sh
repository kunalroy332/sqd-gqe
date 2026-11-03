#!/usr/bin/env bash
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# One command to refresh every inference-derived number, table and report from the cluster. Run from anywhere.
#   scripts/analysis/circuit_analysis/refresh_all.sh
# It does NOT need your IBM key. The QPU files (qpu_fresh.json) are made separately with fetch_qpu_inference_sets.py <hf|tuned>; rerun that
# first (with your key inline) when new sampling jobs have finished, then run this script.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
B=roquo:/path/to/sqd-gqe-work/experiments
cd "$REPO"
for s in hf tuned; do
  T=circuit_analysis_$s
  rsync -a $B/inference_$s/circuit_analysis/ $T/data/
  rsync -a $B/inference_$s/circuit_analysis_rerank/ $T/rerank/
  rsync -a $B/inference_$s/circuit_analysis_rerecovery/ $T/rerecovery/
  W=$(mktemp -d); rsync -a --include='*/' --include='selection.json' --exclude='*' $B/inference_$s/work/ "$W"/
  python3 - "$T" "$W" <<'PY'
import glob, json, os, sys
tree, wd = sys.argv[1:3]
p = f"{tree}/selection_batch1.json"; sel = json.load(open(p)); n0 = len(sel)
for f in glob.glob(f"{wd}/*/selection.json"):
    t = os.path.basename(os.path.dirname(f)); d = json.load(open(f)); d = d.get(t, d)
    if "best" in d: sel[t] = d          # a run with an empty selection (ranking not finished) is skipped
json.dump(sel, open(p, "w"), indent=1); print(tree, "selections", n0, "->", len(sel))
PY
done
cd scripts/analysis/circuit_analysis
python3 inference_timing.py --refresh
CA_ROOT="$REPO/circuit_analysis_hf" python3 make_tables.py
(cd /tmp/inf && for s in hf tuned t1; do for d in circuit_analysis circuit_analysis_rerecovery; do rsync -a $B/inference_$s/$d/ inference_$s/$d/ 2>/dev/null || true; done; done)
INF_DIR=/tmp/inf python3 three_way_load.py > /dev/null
INF_DIR=/tmp/inf python3 three_way_tables.py "$REPO/docs/report/manuscript/paper.tex"
python3 divergence_tables.py "$REPO/docs/report/manuscript/paper.tex"
for s in hf tuned t1; do python3 make_html_sets.py $s | tail -1; done
(cd "$REPO/docs/report/manuscript" && pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1; pdflatex -interaction=nonstopmode paper.tex 2>&1 | grep -E "Output written|^!")
echo "Prose numbers (medians, counts, cost sentences) are not regenerated; see the checklist in docs/report/manuscript/README_refresh.txt"
