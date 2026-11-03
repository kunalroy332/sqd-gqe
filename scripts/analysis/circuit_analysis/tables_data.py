# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Consolidate the fresh-shot re-ranking / re-recovery results per run for the manuscript tables.

Reads circuit_analysis/{rerank,rerecovery}/ and selection_batch*.json (all relative to the repo root)
and returns, per run tag, the five targets (best, second, merged, oldbest, oldsecond) with
alias resolution (an old-rule pick that coincides with a new pick reuses that result).
"""
import glob
import json
from pathlib import Path

import os
# CA_ROOT selects the data tree: the default is the first campaign (Training 1); the HF-penalty model set lives in circuit_analysis_hf, the tuned set in circuit_analysis_tuned.
ROOT = Path(os.environ["CA_ROOT"]) if os.environ.get("CA_ROOT") else Path(__file__).resolve().parents[3] / "circuit_analysis"
DMRG = {
    "sno32": -479.193381728830, "h2o_s19": -555.108832785690, "2h2o_s22": -631.083312007953,
    "h2o_m_no23": -555.125158274414, "llno28": -479.315056933618, "h2o_l28": -555.135057773478,
    "2h2o_m_no29": -631.118615946406, "h2o_ll33": -555.137396193158, "2h2o_l35": -631.132237762872,
    "L46": -479.314216460012,
    "sno38": -479.302278277561, "h2o_l28": -555.135057773478,   # the two tiers added for the HF-penalty / tuned / Training-2 campaign
}
LABEL = {  # molecule key -> (full display name, qubits, tie-break order)
    "sno32": ("SnOH3-me S (32q)", 32, 1), "h2o_s19": ("SnOH3-me-h2o S (38q)", 38, 3), "2h2o_s22": ("SnOH3-me-2h2o S (44q)", 44, 6),
    "L46": ("SnOH3-me L (46q)", 46, 0), "h2o_m_no23": ("SnOH3-me-h2o M (46q)", 46, 4), "llno28": ("SnOH3-me LL (56q)", 56, 2),
    "2h2o_m_no29": ("SnOH3-me-2h2o M (58q)", 58, 7), "h2o_ll33": ("SnOH3-me-h2o LL (66q)", 66, 5), "2h2o_l35": ("SnOH3-me-2h2o L (70q)", 70, 8),
    "sno38": ("SnOH3-me M (38q)", 38, 9), "h2o_l28": ("SnOH3-me-h2o L (56q)", 56, 10),
}
TIER = {"sno32": "S", "h2o_s19": "S", "2h2o_s22": "S", "L46": "L", "h2o_m_no23": "M", "llno28": "LL",
        "2h2o_m_no29": "M", "h2o_ll33": "LL", "2h2o_l35": "L", "sno38": "M", "h2o_l28": "L"}
MOL_NAME = {"sno32": "SnOH3-me", "L46": "SnOH3-me", "llno28": "SnOH3-me", "h2o_s19": "SnOH3-me-h2o", "h2o_m_no23": "SnOH3-me-h2o",
            "h2o_ll33": "SnOH3-me-h2o", "2h2o_s22": "SnOH3-me-2h2o", "2h2o_m_no29": "SnOH3-me-2h2o", "2h2o_l35": "SnOH3-me-2h2o",
            "sno38": "SnOH3-me", "h2o_l28": "SnOH3-me-h2o"}
LIN_ORDER = ["a1-46q", "a1-70q", "a5-46q", "a5-70q"]


def run_key(r):
    """Small -> large active space; within a molecule A1 before A5, 46q- before 70q-trained, trained before untrained."""
    return (r["nq"], r["order"], LIN_ORDER.index(r["lineage"]), r["policy"] != "trained")


TARGETS = ("best", "second", "merged", "oldbest", "oldsecond")


def mol_key(tag):
    return "L46" if tag.startswith("L46_") else next(k for k in sorted(DMRG, key=len, reverse=True) if tag.startswith(k + "_"))


def lineage_policy(tag):
    rest = tag[4:] if tag.startswith("L46_") else tag[len(mol_key(tag)) + 1:]
    lin, pol = rest.rsplit("_", 1)
    return lin, pol


def load():
    sel = {}
    for f in sorted(glob.glob(str(ROOT / "selection_batch*.json"))):
        sel.update(json.loads(Path(f).read_text()))
    rr = {Path(f).stem: json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "rerank" / "*.json"))}
    live = {Path(f).stem: {c["index"]: c for c in json.loads(Path(f).read_text())["circuits"]} for f in glob.glob(str(ROOT / "data" / "*.json"))}
    rc = {}
    for f in glob.glob(str(ROOT / "rerecovery" / "*__*.json")):
        tag, tg = Path(f).stem.split("__")
        rc.setdefault(tag, {})[tg] = json.loads(Path(f).read_text())
    partial = json.loads((ROOT / "partial_recoveries.json").read_text()) if (ROOT / "partial_recoveries.json").exists() else {}
    out = {}
    for tag, s in sel.items():
        mk = mol_key(tag)
        ref = DMRG[mk]
        lin, pol = lineage_policy(tag)
        circ = {c["index"]: c for c in rr[tag]["circuits"]}
        rec = {}
        for t in TARGETS:
            src = t
            if t == "oldbest" and s["oldbest_alias"]:
                src = s["oldbest_alias"]
            if t == "oldsecond" and s["oldsecond_alias"]:
                src = s["oldsecond_alias"]
            r = rc.get(tag, {}).get(src)
            idx = (s["merged"] if t == "merged" else s["old_best"] if t == "oldbest" else s["old_second"] if t == "oldsecond" else s[t])
            entry = {"idx": idx, "alias": src if src != t else None, "done": r is not None}
            if t != "merged":
                c = circ[idx]
                lv = live[tag][idx]
                entry.update(depth_offline=c["depth"], two_q_offline=c["two_qubit_gates"], depth=lv["depth_live"], two_q=lv["two_qubit_live"],
                             valid_shots=c["valid_shots"], e40000=c["energy_40000"])
            if r:
                entry.update(energy=r["final_energy"], err=(r["final_energy"] - ref) * 1000,
                             si=(r["singles_injected_energy"] - ref) * 1000 if r.get("singles_injected_energy") is not None else None,
                             exc=r.get("excitation"), raw=r.get("raw_shot_excitation"),
                             has_alpha=("recovered_alpha_strings" in r))
            p = partial.get(f"{tag}__{src}")
            if p and not entry["done"]:
                entry.update(partial=True, p_energy=p["energy"], p_err=(p["energy"] - ref) * 1000, p_iters=p["iterations_seen"])
            rec[t] = entry
        out[tag] = {"tag": tag, "mol": mk, "label": LABEL[mk][0], "nq": LABEL[mk][1], "order": LABEL[mk][2],
                    "lineage": lin, "policy": pol, "ref": ref, "t": rec, "sel": s,
                    "n_finite": s["n_finite_new"]}
    return out


if __name__ == "__main__":
    d = load()
    f = lambda e: "  pend" if not e["done"] else f"{e['err']:6.2f}"
    print(f"{'tag':34s} best   2nd   merged  oldb   olds   | sel(b,s) old(b,s)  validshots(b/s)")
    for tag in sorted(d, key=lambda t: (d[t]["order"], t)):
        r = d[tag]; t = r["t"]
        print(f"{tag:34s} {f(t['best'])} {f(t['second'])} {f(t['merged'])} {f(t['oldbest'])} {f(t['oldsecond'])}"
              f" | {t['best']['idx']+1},{t['second']['idx']+1}  {t['oldbest']['idx']+1},{t['oldsecond']['idx']+1}  "
              f"{t['best']['valid_shots']}/{t['second']['valid_shots']}")
    print(sum(1 for r in d.values() for e in r["t"].values() if not e["done"]), "pending cells of", sum(len(r["t"]) for r in d.values()))


MISSING_RUNS = []


def missing_runs():
    """Runs whose first-campaign sample exists but whose fresh-shot pipeline could not be run (cluster down); queued for after maintenance."""
    out = []
    for mk, lin, pol in MISSING_RUNS:
        out.append({"tag": f"{mk}_{lin}_{pol}", "mol": mk, "label": LABEL[mk][0], "nq": LABEL[mk][1], "order": LABEL[mk][2],
                    "lineage": lin, "policy": pol, "missing": True})
    return out


def coverage_matrix(d=None):
    """{molecule key: {(policy, lineage): 'done' | 'queued' | 'not run'}} for all nine targets."""
    d = d or load()
    miss = {(m, l, p) for m, l, p in MISSING_RUNS}
    have = {(r["mol"], r["lineage"], r["policy"]) for r in d.values()}
    out = {}
    for mk in LABEL:
        out[mk] = {(p, l): ("done" if (mk, l, p) in have else "queued" if (mk, l, p) in miss else "not run")
                   for p in ("trained", "untrained") for l in LIN_ORDER}
    return out
