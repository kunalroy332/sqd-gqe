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

"""Selection/recovery statistics quoted in the manuscript's analysis subsection (computed from the stored result files)."""
import json
import math
import statistics as st

from tables_data import ROOT, load


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.fmean(rx), st.fmean(ry)
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den if den else float("nan")


def numbers():
    d = load()
    rec = []
    for tag, r in d.items():
        seen = set()
        for k in ("best", "second", "oldbest", "oldsecond"):
            e = r["t"][k]
            if not e["done"] or e["idx"] in seen:
                continue
            seen.add(e["idx"])
            cn = json.loads((ROOT / "connectivity" / f"{tag}.json").read_text())["circuits"][e["idx"]]["logical"]
            rec.append({"tag": tag, "mol": r["mol"], "label": r["label"], "pol": r["policy"], "err": e["err"], "valid": e["valid_shots"],
                        "block": cn["components"] == 1 and cn["qubits_touched_any_gate"] == cn["qubits_total"], "nq": r["nq"]})
    out = {"n_recoveries": len(rec)}
    big = [x for x in rec if x["pol"] == "trained" and x["nq"] >= 44]
    bands = []
    for lo, hi in [(0, 100), (100, 1000), (1000, 10_000), (10_000, 10 ** 9)]:
        xs = [x["err"] for x in big if lo <= x["valid"] < hi]
        bands.append({"lo": lo, "hi": hi, "n": len(xs), "median": st.median(xs) if xs else None,
                      "share_below_25": (sum(1 for e in xs if e < 25) / len(xs)) if xs else None})
    out["valid_bands"] = bands
    sp = []
    for mol in sorted({x["mol"] for x in rec}):
        xs = [x for x in rec if x["mol"] == mol and x["pol"] == "trained" and x["valid"] > 0]
        if len(xs) >= 5:
            sp.append({"label": xs[0]["label"], "n": len(xs), "rho": spearman([math.log10(x["valid"]) for x in xs], [x["err"] for x in xs])})
    out["spearman"] = sp
    out["spearman_median"] = st.median(s["rho"] for s in sp)
    n = agree = 0
    for r in d.values():
        b, s = r["t"]["best"], r["t"]["second"]
        if b["done"] and s["done"] and r["policy"] == "trained":
            n += 1
            agree += b["err"] <= s["err"]
    out["best_vs_second"] = {"runs": n, "best_not_worse": agree}
    blk = [x["err"] for x in big if x["block"]]
    oth = [x["err"] for x in big if not x["block"]]
    out["block"] = {"one_block": {"n": len(blk), "median": st.median(blk), "below_25": sum(1 for e in blk if e < 25)},
                    "other": {"n": len(oth), "median": st.median(oth), "below_25": sum(1 for e in oth if e < 25)}}
    return out


if __name__ == "__main__":
    print(json.dumps(numbers(), indent=1))
