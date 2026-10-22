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

"""Check that the numbers in the manuscript tables and in the Zenodo HTML are identical (same source, same formatting)."""
import html
import re
import sys
from pathlib import Path

from tables_data import ROOT, load, run_key

TEX = Path(__file__).resolve().parents[3] / "docs/report/manuscript/paper.tex"
HTML = ROOT / "zenodo/sqd-gqe_zenodo.html"


def strip(x):
    x = re.sub(r"<span[^>]*>.*?</span>", "pend.", x)
    x = html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
    x = x.replace("≤", r"$\le$").replace("†", r"$^{\dagger}$")
    return x


def html_summary():
    h = HTML.read_text()
    sec = h[h.index('id="summary"'):h.index('id="coverage"')]
    rows = re.findall(r"<tr>(.*?)</tr>", sec, re.S)
    out = {}
    for r in rows[1:]:
        c = [strip(x) for x in re.findall(r"<td>(.*?)</td>", r, re.S)]
        if len(c) >= 9:
            out[(c[0], c[1])] = c[2:9]
    return out


def tex_rows(label):
    t = TEX.read_text()
    m = re.search(r"\\label\{" + label + r"\}.*?\\end\{tabular\}", t, re.S)
    rows = {}
    for line in m.group(0).splitlines():
        if "&" in line and line.rstrip().endswith("\\\\") and "Lineage" not in line:
            c = [x.strip().rstrip("\\").strip() for x in line.rstrip().rstrip("\\\\").split("&")]
            rows[tuple(c[:2])] = c
    return rows


def main():
    d = load()
    hs = html_summary()
    tx = {}
    tx.update(tex_rows("tab:crossnorb-detail"))
    bad = ok = 0
    for r in sorted(d.values(), key=run_key):
        name = f"{r['lineage'].upper()} {r['policy']}"
        if r["mol"] == "L46":
            continue
        key_h = (r["label"], name.replace("A1-46Q", "A1-46Q"))
        hrow = hs.get((r["label"], name))
        trow = tx.get((r["label"], f"{r['lineage'].replace('a1', 'A1').replace('a5', 'A5')} {r['policy']}"))
        if not hrow or not trow:
            print("MISSING", r["tag"], bool(hrow), bool(trow)); bad += 1; continue
        # html: best, second, merged, E+D1, E+D2 | tex: err best/2nd/merged, err E+D 1/2
        h_err = hrow[:5]
        t_err = trow[5].split("/") + trow[6].split("/")
        if [strip(x) for x in h_err] != [strip(x) for x in t_err]:
            print("MISMATCH", r["tag"], h_err, t_err); bad += 1
        else:
            ok += 1
        # valid shots
        if hrow[6] != trow[4]:
            print("MISMATCH valid shots", r["tag"], hrow[6], trow[4]); bad += 1
    print(f"cross-norb table vs HTML summary: {ok} runs identical, {bad} problems")
    # L-tier
    tl = tex_rows("tab:zeroshot-detail")
    okl = badl = 0
    for r in sorted(d.values(), key=run_key):
        if r["mol"] != "L46":
            continue
        name = f"{r['lineage'].replace('a1', 'A1').replace('a5', 'A5')} {r['policy']}"
        trow = tl.get((name, ))
        trow = next((v for k, v in tl.items() if k[0] == name), None)
        hrow = hs.get((r["label"], f"{r['lineage'].upper()} {r['policy']}"))
        if not trow or not hrow:
            print("MISSING L", r["tag"]); badl += 1; continue
        t_err = trow[4].split("/") + trow[5].split("/")
        if [strip(x) for x in hrow[:5]] != [strip(x) for x in t_err]:
            print("MISMATCH L", r["tag"], hrow[:5], t_err); badl += 1
        else:
            okl += 1
    print(f"L-tier table vs HTML summary: {okl} runs identical, {badl} problems")
    sys.exit(1 if bad + badl else 0)


if __name__ == "__main__":
    main()
