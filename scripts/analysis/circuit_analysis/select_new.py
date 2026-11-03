# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""New selection from the 40000-dim re-ranking, and the re-recovery job files.

Criterion (replaces rank_combined = energy rank + n_distinct rank): rank circuits by their
ranking-stage energy at net CI dim 40000 (alpha cap 200), ascending, among circuits with a
finite energy. Best / second = the two lowest. Merged = pool of every finite-energy circuit
(same pooling rule as before, now over the new finite set). Circuits with zero valid shots
have energy inf and are excluded, as before.

Usage: select_new.py --rerank-dir DIR --out selection.json [--write-sbatch DIR]
"""
import argparse
import json
from pathlib import Path

SBATCH = """#!/usr/bin/env bash
#SBATCH -A <SLURM_ACCOUNT>
#SBATCH -p roquo
#SBATCH -J rc_{short}
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --time=13:00:00
#SBATCH --time-min=3:00:00
#SBATCH --exclude=qr01n08,qr02n11,qr04n16
#SBATCH -o /path/to/sqd-gqe-work/experiments/production_final/logs/rc_{tag}__{target}.out
#SBATCH -e /path/to/sqd-gqe-work/experiments/production_final/logs/rc_{tag}__{target}.err

module load cuda/13.2 nvhpc/26.5 hpcx/2.50
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
set -a
source /path/to/sqd-gqe/.env.local
set +a
VENV=/path/to/sqd-gqe/.venv
SP=$($VENV/bin/python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/scipy.libs:$SP/numpy.libs:$LD_LIBRARY_PATH"
cd /path/to/sqd-gqe-work
PYTHONPATH=. $VENV/bin/python scripts/analysis/circuit_analysis/rerecover_target.py --tag {tag} --target {target} --idx {idx}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-sbatch")
    args = ap.parse_args()

    sel = {}
    for f in sorted(Path(args.rerank_dir).glob("*.json")):
        r = json.loads(f.read_text())
        cs = r["circuits"]
        fin = sorted((c for c in cs if c["energy_40000"] is not None), key=lambda c: c["energy_40000"])
        if len(fin) < 2:
            print(f"{r['tag']}: only {len(fin)} finite circuits -- needs manual handling")
            continue
        old_b = next((c["index"] for c in cs if c["old_is_best"]), None)
        old_s = next((c["index"] for c in cs if c["old_is_second"]), None)
        sel[r["tag"]] = {
            "best": fin[0]["index"], "second": fin[1]["index"],
            "merged": sorted(c["index"] for c in fin),
            "old_best": old_b, "old_second": old_s,
            "best_changed": fin[0]["index"] != old_b, "second_changed": fin[1]["index"] != old_s,
            "oldbest_alias": ("best" if old_b == fin[0]["index"] else "second" if old_b == fin[1]["index"] else None),
            "oldsecond_alias": ("best" if old_s == fin[0]["index"] else "second" if old_s == fin[1]["index"] else None),
            "n_finite_new": len(fin), "n_finite_old": sum(c["energy_2000_orig"] is not None for c in cs),
            "best_energy_40000": fin[0]["energy_40000"], "second_energy_40000": fin[1]["energy_40000"],
            "best_depth": fin[0]["depth"], "second_depth": fin[1]["depth"],
            "best_valid_shots": fin[0]["valid_shots"], "second_valid_shots": fin[1]["valid_shots"],
        }
    Path(args.out).write_text(json.dumps(sel, indent=1))

    print(f"{'run':34s} best old->new  2nd old->new  finite old->new  new best E(40000)   valid shots (best/2nd)")
    for t, s in sel.items():
        print(f"{t:34s} {s['old_best'] + 1:>3}->{s['best'] + 1:<3}{'*' if s['best_changed'] else ' '}      "
              f"{s['old_second'] + 1:>3}->{s['second'] + 1:<3}{'*' if s['second_changed'] else ' '}     "
              f"{s['n_finite_old']:>2}->{s['n_finite_new']:<2}         {s['best_energy_40000']:>14.4f}      "
              f"{s['best_valid_shots']}/{s['second_valid_shots']}")
    print(f"\n{len(sel)} runs; best changed in {sum(s['best_changed'] for s in sel.values())}, "
          f"second changed in {sum(s['second_changed'] for s in sel.values())}  (* = changed)")

    if args.write_sbatch:
        out = Path(args.write_sbatch)
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        # Priority = submission order (Slurm is roughly FIFO within one user): the new-rule
        # best + merged for every run first, then second, then the two old-rule extras.
        order = [("best", "best"), ("merged", "merged"), ("second", "second"),
                 ("oldbest", "old_best"), ("oldsecond", "old_second")]
        names = []
        for target, key in order:
          for t, s in sel.items():
            if target in ("oldbest", "oldsecond") and s[f"{target}_alias"]:
                continue  # same circuit as a new pick: reuse that result, do not recompute
            idx = ",".join(map(str, s["merged"])) if target == "merged" else str(s[key] if target.startswith("old") else s[target])
            if True:
                short = f"{t.replace('_trained', '_t').replace('_untrained', '_u')}_{ {'best':'b','second':'s','merged':'m','oldbest':'ob','oldsecond':'os'}[target] }"[:40]
                (out / f"rc_{t}__{target}.sbatch").write_text(
                    SBATCH.format(tag=t, target=target, idx=idx, short=short))
                names.append(f"rc_{t}__{target}.sbatch")
                n += 1
        (out / "SUBMIT_ORDER.txt").write_text("\n".join(names) + "\n")
        print(f"wrote {n} re-recovery sbatch files to {out}")


if __name__ == "__main__":
    main()
