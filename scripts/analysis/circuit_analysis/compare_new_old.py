# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compare re-recovery results (fresh shots, new ranking) with the original recovered results.

Prints, per run and target, the final recovered energy and error vs DMRG (mHa), and the
singles-injected error, next to the original run's best / second / merged. Anything where the
NEW best is worse than the OLD best is flagged.

Usage: compare_new_old.py --new-dir circuit_analysis/rerecovery [--old-dir /tmp/stage_data]
"""
import argparse
import glob
import json
from pathlib import Path

DMRG = {
    "sno32": -479.193381728830, "h2o_s19": -555.108832785690, "2h2o_s22": -631.083312007953,
    "h2o_m_no23": -555.125158274414, "llno28": -479.315056933618, "h2o_l28": -555.135057773478,
    "2h2o_m_no29": -631.118615946406, "h2o_ll33": -555.137396193158, "2h2o_l35": -631.132237762872,
    "L46": -479.314216460012,
}
TARGETS = ("best", "second", "merged", "oldbest", "oldsecond")


def mol_key(tag):
    return "L46" if tag.startswith("L46_") else next(k for k in sorted(DMRG, key=len, reverse=True) if tag.startswith(k + "_"))


def old_file(old_dir, tag):
    if tag.startswith("L46_"):
        g = f"{old_dir}/unseen46q_{tag[4:]}_sub20000_job*.json"
    else:
        g = f"{old_dir}/crossnorb70q_{tag}_sub20000_job*.json"
    f = sorted(glob.glob(g))
    return json.loads(Path(f[0]).read_text()) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--old-dir", default="/tmp/stage_data")
    args = ap.parse_args()

    new = {}
    for f in Path(args.new_dir).glob("*__*.json"):
        tag, target = f.stem.split("__")
        new.setdefault(tag, {})[target] = json.loads(f.read_text())

    worse = []
    for tag in sorted(new):
        ref = DMRG[mol_key(tag)]
        old = old_file(args.old_dir, tag)
        err = lambda e: (ref - e) * -1000 if e is not None else float("nan")  # positive = above DMRG
        print(f"\n{tag}")
        print(f"  {'target':10s} {'circ':>10s} {'err mHa':>9s} {'SI err':>8s}")
        for t in TARGETS:
            r = new[tag].get(t)
            if not r:
                continue
            print(f"  NEW {t:8s} {','.join(str(i + 1) for i in r['circuit_indices'])[:10]:>10s} "
                  f"{err(r['final_energy']):9.2f} {err(r['singles_injected_energy']):8.2f}")
        if old:
            for k in ("best", "second_best", "merged"):
                o = old[k]
                print(f"  OLD {k:8s} {'':>10s} {err(o['final_energy']):9.2f} {err(o.get('singles_injected_energy')):8.2f}")
            if "best" in new[tag]:
                nb, ob = err(new[tag]["best"]["final_energy"]), err(old["best"]["final_energy"])
                if nb > ob + 0.05:
                    worse.append((tag, nb, ob))
    print("\nNEW best WORSE than OLD best (err mHa, new vs old):", worse if worse else "none")


if __name__ == "__main__":
    main()
