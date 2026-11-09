# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Single source of the numbers quoted in the abstract, results, discussion and conclusion of the manuscript.
Usage: python3 headline_numbers.py [--json OUT]   (reads data/hf/ and data/tuned/ and data/R3.pkl, which three_way_load.py builds from them)"""
import collections, glob, json, os, pickle, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.dirname(__file__))
LU = {"llno28": (428, 4248), "h2o_l28": (430, 4360), "2h2o_m_no29": (446, 4912), "h2o_ll33": (506, 5880), "2h2o_l35": (536, 6952)}


def load(tree):
    os.environ["CA_ROOT"] = os.path.join(REPO, tree)
    import importlib, tables_data as T
    importlib.reload(T)
    return T.load(), T


def main():
    out = {}
    d, T = load("data/hf")
    tr = [v for v in d.values() if v["policy"] == "trained"]
    un = [v for v in d.values() if v["policy"] == "untrained"]
    fin = [v for v in tr if v["t"]["best"].get("err") is not None]
    e = np.array([v["t"]["best"]["err"] for v in fin])
    out["hf_rows"] = len(tr); out["hf_best_recovered"] = len(fin)
    out["hf_median_best_err"] = float(np.median(e)); out["hf_within_1"] = int((e < 1).sum()); out["hf_within_5"] = int((e < 5).sum()); out["hf_within_15"] = int((e < 15).sum())
    eu = [v["t"]["best"]["err"] for v in un if v["t"]["best"].get("err") is not None]
    out["untrained_n"] = len(eu); out["untrained_median"] = float(np.median(eu)); out["untrained_within_15"] = int(sum(x < 15 for x in eu))
    dd = [v["t"]["best"]["depth"] for v in fin]; qq = [v["t"]["best"]["two_q"] for v in fin]
    out["hf_depth"] = [int(min(dd)), float(np.median(dd)), int(max(dd))]; out["hf_2q"] = [int(min(qq)), float(np.median(qq)), int(max(qq))]
    big = [v for v in fin if v["mol"] in LU]
    out["hf_5670_n"] = len(big)
    out["hf_5670_depth"] = [min(v["t"]["best"]["depth"] for v in big), max(v["t"]["best"]["depth"] for v in big)]
    out["hf_5670_2q"] = [min(v["t"]["best"]["two_q"] for v in big), max(v["t"]["best"]["two_q"] for v in big)]
    rd = [LU[v["mol"]][0] / v["t"]["best"]["depth"] for v in big]; rq = [LU[v["mol"]][1] / v["t"]["best"]["two_q"] for v in big]
    out["hf_5670_depth_ratio"] = [min(rd), max(rd)]; out["hf_5670_2q_ratio"] = [min(rq), max(rq)]
    out["hf_5670_err"] = [min(v["t"]["best"]["err"] for v in big), max(v["t"]["best"]["err"] for v in big)]
    sm = [v for v in fin if v["nq"] <= 44 and v["mol"] != "sno38"]
    out["hf_le44_excl_sno38_err"] = [min(v["t"]["best"]["err"] for v in sm), max(v["t"]["best"]["err"] for v in sm)]
    out["hf_sno38_err"] = sorted(v["t"]["best"]["err"] for v in fin if v["mol"] == "sno38")
    out["hf_hydrateM_err"] = sorted(v["t"]["best"]["err"] for v in fin if v["mol"] == "h2o_m_no23")
    out["hf_bs_n"] = sum(1 for v in fin if v["t"]["second"].get("err") is not None)
    out["hf_best_le_second"] = sum(1 for v in fin if v["t"]["second"].get("err") is not None and v["t"]["best"]["err"] <= v["t"]["second"]["err"])
    best_per = collections.defaultdict(list)
    for v in fin:
        best_per[v["mol"]].append(min(v["t"][k]["err"] for k in ("best", "second", "merged", "oldbest", "oldsecond") if v["t"][k].get("err") is not None))
    out["hf_lowest_per_tier"] = {T.LABEL[m][0]: round(min(x), 2) for m, x in best_per.items()}
    # distinct recoveries >= 44 q by valid-shot bin
    rows = []
    for v in tr:
        if v["nq"] < 44: continue
        seen = set()
        for tg in ("best", "second", "oldbest", "oldsecond"):
            c = v["t"][tg]
            if c.get("err") is None or c["idx"] in seen: continue
            seen.add(c["idx"]); rows.append((c["valid_shots"], c["err"]))
    out["valid_bins"] = [[lo, sum(1 for r in rows if lo <= r[0] < hi), float(np.median([r[1] for r in rows if lo <= r[0] < hi])), sum(1 for r in rows if lo <= r[0] < hi and r[1] < 25)] for lo, hi in ((0, 100), (100, 1000), (1000, 10000), (10000, 1e12))]
    out["valid_bins_total"] = len(rows)
    out["pending_cells_hf"] = sum(1 for v in d.values() for x in v["t"].values() if not x["done"])
    dt, _ = load("data/tuned")
    out["pending_cells_tuned"] = sum(1 for v in dt.values() for x in v["t"].values() if not x["done"])
    out["tuned_rows"] = sum(1 for v in dt.values() if v["policy"] == "trained")
    R = pickle.load(open(os.path.join(REPO, "data", "R3.pkl"), "rb"))
    cells = [k for k, v in R.items() if {"HF", "tuned"} <= set(v)]
    out["pair_cells"] = len(cells)
    for s in ("HF", "tuned"):
        b = [R[k][s]["best"]["err"] for k in cells]; c = [R[k][s]["second"]["err"] for k in cells]
        out[f"pair_{s}"] = {"best_med": float(np.median(b)), "best_mean": float(np.mean(b)), "second_med": float(np.median(c)), "min2_med": float(np.median([min(x, y) for x, y in zip(b, c)])),
                            "depth_med": float(np.median([R[k][s]["best"]["depth"] for k in cells])),
                            "within_1": int(sum(x < 1 for x in b)), "within_5": int(sum(x < 5 for x in b)), "within_15": int(sum(x < 15 for x in b)), "above_30": sorted(round(x, 1) for x in b if x > 30)}
    out["pair_tuned_lower_best"] = int(sum(R[k]["tuned"]["best"]["err"] < R[k]["HF"]["best"]["err"] for k in cells))
    print(json.dumps(out, indent=1, default=float))
    if "--json" in sys.argv: json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
