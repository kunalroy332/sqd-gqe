# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Per-circuit table + checks for the raw-excitation hypothesis.

For every run JSON in --data-dir (from resample_raw_counts.py), join the
matching original result JSON (for n_distinct per circuit and the earlier
best/second raw histograms) and report:
  1. sanity: do the fresh best/second raw histograms have the same dominant rank
     as the earlier ones (shots are fresh, so only the overall shape should match)?
  2. per circuit: depth, ranking-stage energy, n_distinct, low-excitation fraction
     (HF+S+D+T of the raw shots), and whether it was picked best/second.
  3. is the low-excitation fraction related to the ranking-stage energy, to
     n_distinct, and to being selected?
"""
import argparse
import glob
import json
import math
from pathlib import Path


def low_frac(h):
    return (h["HF"] + h["S"] + h["D"] + h["T"]) / h["combined"]


def dominant(h):
    return max(("HF", "S", "D", "T", "Q", "higher"), key=lambda k: h[k])


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _replay_selection(circuits, n_distinct):
    """Re-apply infer_crossnorb_70q_lineages.py's rank_combined: rank by energy
    (ascending) plus rank by n_distinct (descending), ties keep batch order."""
    idx = [i for i, c in enumerate(circuits)
           if c["ranking_energy"] is not None and math.isfinite(c["ranking_energy"]) and n_distinct[i]]
    by_e = sorted(idx, key=lambda i: circuits[i]["ranking_energy"])
    e_rank = {i: r + 1 for r, i in enumerate(by_e)}
    by_d = sorted(idx, key=lambda i: -n_distinct[i])
    d_rank = {i: r + 1 for r, i in enumerate(by_d)}
    scored = sorted(idx, key=lambda i: (e_rank[i] + d_rank[i], circuits[i]["ranking_energy"]))
    return scored, e_rank, d_rank


def _within_run(run, nd):
    cs = run["circuits"]
    fin = [i for i, c in enumerate(cs) if c["ranking_energy"] is not None and math.isfinite(c["ranking_energy"])]
    lowest = min(fin, key=lambda i: cs[i]["ranking_energy"])
    lf = low_frac(cs[lowest]["raw_hist"])
    scored, e_rank, d_rank = _replay_selection(cs, nd)
    got = [i for i, c in enumerate(cs) if c["is_best"] or c["is_second"]]
    replay_ok = set(scored[:2]) == set(got)
    print(f"  lowest ranking-stage energy: circuit {lowest + 1} (low-excitation raw fraction {100 * lf:.1f}%, "
          f"selected={'yes' if lowest in got else 'NO'}; energy rank {e_rank[lowest]}, distinct rank {d_rank[lowest]}, "
          f"combined score {e_rank[lowest] + d_rank[lowest]})")
    print(f"  selection replay (energy rank + n_distinct rank) reproduces the stored best/2nd: {replay_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", default="/tmp/stage_data")
    args = ap.parse_args()

    allc = []  # (low_frac, finite_energy, n_distinct, selected, tag)
    for f in sorted(glob.glob(f"{args.data_dir}/*.json")):
        run = json.load(open(f))
        tag = Path(f).stem
        cand = glob.glob(f"{args.results_dir}/crossnorb70q_{tag}_sub20000_job*.json")
        old = json.load(open(cand[0])) if cand else None
        print(f"\n=== {tag}  ({run['molecule_label']}) ===")
        if old:
            for key, flag in (("best", "is_best"), ("second_best", "is_second")):
                new_c = [c for c in run["circuits"] if c[flag]]
                if not new_c:
                    print(f"  sanity {key}: circuit not found in the re-run batch")
                    continue
                hn, ho = new_c[0]["raw_hist"], old[key]["raw_shot_excitation"]
                same = dominant(hn) == dominant(ho)
                print(f"  sanity {key}: dominant new={dominant(hn)} old={dominant(ho)} "
                      f"{'OK' if same else '*** DIFFERS ***'}  low-frac new={low_frac(hn):.3f} old={low_frac(ho):.3f}")
        nd = [c["n_distinct"] for c in old["batch_summary"]] if old else [None] * len(run["circuits"])
        print("  idx depth   rank_E        n_distinct  low%   sel")
        for c, n in zip(run["circuits"], nd):
            lf = low_frac(c["raw_hist"])
            e = c["ranking_energy"]
            fin = e is not None and math.isfinite(e)
            sel = "best" if c["is_best"] else ("2nd" if c["is_second"] else "")
            print(f"  {c['index'] + 1:>3} {c['cost']['depth']!s:>6} {e if fin else 'inf':>12}  {n!s:>10} {100 * lf:6.1f}  {sel}")
            allc.append((lf, e if fin else None, n, bool(sel), tag))
        if old:
            _within_run(run, nd)

    print("\n=== pooled over all runs given ===")
    fin = [a for a in allc if a[1] is not None]
    print(f"circuits: {len(allc)}, with finite ranking energy: {len(fin)}")
    hi = [a for a in allc if a[0] >= 0.5]
    print(f"circuits with >=50% of raw shots at HF/S/D/T: {len(hi)}; of those, selected best/2nd: {sum(a[3] for a in hi)}")
    lo = [a for a in allc if a[0] < 0.5]
    print(f"circuits with <50%: {len(lo)}; of those, selected best/2nd: {sum(a[3] for a in lo)}")
    nd_ok = [a for a in allc if a[2] is not None]
    if len(nd_ok) > 3:
        print(f"Spearman(low-excitation fraction, n_distinct) = {spearman([a[0] for a in nd_ok], [a[2] for a in nd_ok]):+.2f}")
    # ranking energy is NOT pooled across molecules (different absolute energy scales); see per-run lines.


if __name__ == "__main__":
    main()
