# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Independent check: numbers printed in the manuscript tables and in the HTML versus the raw result files
(rerecovery/*.json, data/*.json, connectivity/*.json, rerank/*.json). Does not use the table generators' formatting code."""
import html as H
import json
import re
import sys
from pathlib import Path

from tables_data import LABEL, ROOT

TEX = (Path(__file__).resolve().parents[3] / "docs/report/manuscript/paper.tex").read_text()
HTML = (ROOT / "zenodo/sqd-gqe_zenodo.html").read_text()
runs = {}
for f in (ROOT / "data").glob("*.json"):
    runs[f.stem] = json.loads(f.read_text())
rerank = {f.stem: json.loads(f.read_text()) for f in (ROOT / "rerank").glob("*.json")}
conn = {f.stem: json.loads(f.read_text()) for f in (ROOT / "connectivity").glob("*.json")}
rec = {}
for f in (ROOT / "rerecovery").glob("*__*.json"):
    t, g = f.stem.split("__")
    rec.setdefault(t, {})[g] = json.loads(f.read_text())
sel = {}
for f in sorted(ROOT.glob("selection_batch*.json")):
    sel.update(json.loads(f.read_text()))
name2tag = {}
for tag in runs:
    mk = "L46" if tag.startswith("L46_") else next(k for k in sorted(LABEL, key=len, reverse=True) if tag.startswith(k + "_"))
    lin, pol = (tag[4:] if tag.startswith("L46_") else tag[len(mk) + 1:]).rsplit("_", 1)
    name2tag[(LABEL[mk][0], lin.replace("a1", "A1").replace("a5", "A5"), pol)] = tag
bad = []
n = 0


def chk(cond, msg):
    global n
    n += 1
    if not cond:
        bad.append(msg)


def env(label, kind="table"):
    m = re.search(r"\\label\{" + re.escape(label) + r"\}.*?\\end\{(?:tabular|longtable)\}", TEX, re.S)
    return m.group(0)


def rows(label):
    out = []
    for line in env(label).splitlines():
        if "&" in line and line.rstrip().endswith("\\\\"):
            out.append([x.strip() for x in line.rstrip()[:-2].split("&")])
    return out


def num(x):
    return float(re.sub(r"[^0-9eE+\-.]", "", x.replace("$-$", "-").replace("\\le", "")))


# 1. excitation tables: energy, S, D, T, Q, higher (paper)
for label in ("tab:zeroshot-excitation", "tab:crossnorb-excitation"):
    for r in rows(label):
        if len(r) < 9 or r[1] not in ("best", "second", "merged") or "pend" in r[2] or r[2].startswith("--"):
            continue
        parts = r[0].rsplit(" ", 2)
        name = r[0]
        key = None
        for (lab, lin, pol), tag in name2tag.items():
            if r[0] == f"{lab} {lin} {pol}" or (label == "tab:zeroshot-excitation" and r[0] == f"{lin} {pol}" and lab == "SnOH3-me L (46q)"):
                key = tag
        if not key:
            bad.append(f"{label}: cannot map row {r[0]}"); continue
        tg = r[1]
        s = sel[key]
        x = rec[key].get(tg if tg == "merged" else tg)
        if x is None:
            continue
        e = x["excitation"]
        chk(abs(num(r[2]) - x["final_energy"]) < 6e-7, f"{label} {key} {tg} energy {r[2]} vs {x['final_energy']}")
        for col, k in zip(r[4:9], ("S", "D", "T", "Q", "higher")):
            chk(int(num(col)) == int(e[k]), f"{label} {key} {tg} {k} {col} vs {e[k]}")
# 2. raw-shot table (paper)
for r in rows("tab:raw-shot-excitation"):
    if len(r) < 9 or r[1] not in ("best", "second", "merged") or r[2] == "--":
        continue
    key = next((t for (lab, lin, pol), t in name2tag.items() if r[0] == f"{lab} {lin} {pol}"), None)
    if not key or r[1] not in rec[key]:
        bad.append(f"raw: cannot map {r[0]} {r[1]}"); continue
    w = rec[key][r[1]]["raw_shot_excitation"]
    for col, k in zip(r[2:9], ("HF", "S", "D", "T", "Q", "higher", "combined")):
        chk(int(num(col)) == int(w[k]), f"raw {key} {r[1]} {k} {col} vs {w[k]}")
# 3. detail tables: live depth / 2q of best and second, valid shots
for label in ("tab:zeroshot-detail", "tab:crossnorb-detail"):
    for r in rows(label):
        if "multicolumn" in " ".join(r) or "queued" in " ".join(r):
            continue
        l46 = label == "tab:zeroshot-detail"
        lab = "SnOH3-me L (46q)" if l46 else r[0]
        lp = r[0] if l46 else r[1]
        if len(lp.split()) != 2:
            continue
        lin, pol = lp.split()
        tag = name2tag.get((lab, lin, pol))
        if not tag:
            continue
        off = 1 if l46 else 2
        d_b, d_s = r[off].split("/")
        q_b, q_s = r[off + 1].split("/")
        v_b, v_s = r[off + 2].split("/")
        # best/second indices from the recoveries' circuit_indices (independent of the selection json)
        bi = rec.get(tag, {}).get("best", {}).get("circuit_indices", [sel[tag]["best"]])[0]
        si = rec.get(tag, {}).get("second", {}).get("circuit_indices", [sel[tag]["second"]])[0]
        C = runs[tag]["circuits"]
        chk(int(d_b) == C[bi]["depth_live"] and int(d_s) == C[si]["depth_live"], f"{label} {tag} depth {d_b}/{d_s}")
        chk(int(q_b) == C[bi]["two_qubit_live"] and int(q_s) == C[si]["two_qubit_live"], f"{label} {tag} 2q {q_b}/{q_s}")
        rr = {c["index"]: c for c in rerank[tag]["circuits"]}
        chk(int(v_b) == rr[bi]["valid_shots"] and int(v_s) == rr[si]["valid_shots"], f"{label} {tag} valid {v_b}/{v_s}")
# 4. coverage table (paper) vs connectivity
for r in rows("tab:coverage-idle"):
    if len(r) < 7:
        continue
    lab, lp = r[0], r[1]
    if len(lp.split()) != 2:
        continue
    lin, pol = lp.split()
    tag = name2tag.get((lab, lin, pol))
    if not tag:
        continue
    cs = conn[tag]["circuits"]
    t = [c["logical"]["qubits_touched_any_gate"] for c in cs]
    chk(r[3].replace(" ", "") == f"{min(t)}/{max(t)}", f"coverage {tag} {r[3]} vs {min(t)}/{max(t)}")
    chk(int(r[6].split("/")[0]) == sum(1 for x in t if x < conn[tag]["n_logical_qubits"]), f"coverage {tag} idle count {r[6]}")
# 5. HTML per-run recoveries: energy and error vs json
for tag in runs:
    m = re.search(r'<details id="' + re.escape(tag) + r'">(.*?)</details>', HTML, re.S)
    if not m:
        bad.append(f"html: no section for {tag}"); continue
    blk = m.group(1)
    for g, x in rec.get(tag, {}).items():
        cell = f"{x['final_energy']:.6f}"
        chk(cell in blk, f"html {tag} {g} energy {cell} missing")
    # depth live of every circuit appears in the circuits table
    for c in runs[tag]["circuits"]:
        chk(f"<td>{c['depth_live']}</td>" in blk, f"html {tag} live depth {c['depth_live']} missing")
# 6. raw histogram sanity (all circuits)
for tag, r in runs.items():
    for c in r["circuits"]:
        h = c["raw_hist"]
        chk(abs(sum(h[k] for k in ("HF", "S", "D", "T", "Q", "higher")) - h["combined"]) < 1, f"hist sum {tag}")
        chk(h["combined"] == r["shots"] or h["combined"] <= r["shots"], f"hist total {tag}")
print(f"{n} checks, {len(bad)} failures")
for b in bad[:40]:
    print("FAIL", b)
sys.exit(1 if bad else 0)
