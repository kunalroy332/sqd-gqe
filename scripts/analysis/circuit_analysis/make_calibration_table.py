# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Insert the ibm_kobe calibration tables into the manuscript from circuit_analysis/kobe_calibration.json.

Idempotent: the block between '% BEGIN KOBE CAL' and '% END KOBE CAL' is replaced (inserted before
\\section{Data and circuit availability} the first time). Nothing is written if the json is missing.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
J = ROOT / "circuit_analysis/kobe_calibration.json"
TEX = ROOT / "docs/report/manuscript/paper.tex"


def f(x, p):
    return "--" if x is None else f"{x:.{p}f}"


def row(r, first):
    if "error" in r:
        return f"{first} & \\multicolumn{{6}}{{l}}{{not available from IBM}} \\\\"
    g = lambda k, key="median": (r[k] or {}).get(key)
    return (f"{first} & {f(g('t1_us'), 0)} & {f(g('t2_us'), 0)} & {f(100 * g('readout_error'), 2) if g('readout_error') is not None else '--'} & "
            f"{f(100 * g('sx_error'), 3) if g('sx_error') is not None else '--'} & {f(100 * g('two_qubit_error'), 2) if g('two_qubit_error') is not None else '--'} & "
            f"{f(100 * g('two_qubit_error', 'worst'), 1) if g('two_qubit_error') is not None else '--'} \\\\")


def main():
    if not J.exists():
        print("kobe_calibration.json not found; nothing inserted"); sys.exit(1)
    d = json.loads(J.read_text())
    cols_short = "T$_1$ ($\\mu$s) & T$_2$ ($\\mu$s) & RO (\\%) & 1q (\\%) & 2q (\\%) & worst 2q (\\%)"
    cols = ("T$_1$ ($\\mu$s) & T$_2$ ($\\mu$s) & readout error (\\%) & 1q (\\code{sx}) error (\\%) & 2q (\\code{cz}) error (\\%) & worst 2q error (\\%)")
    hdr = "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\nSnapshot (UTC) & " + cols + " \\\\\n\\midrule\n"
    daily = [r for r in d["daily"] if "error" not in r or True]
    days = "\n".join(row(r, r["requested_utc"][:16].replace("T", " ")) for r in reversed(daily))
    jobs = "\n".join(row(r, f"{r['run'].replace('_', ' ')}, {r['created_utc'][:16].replace('T', ' ')}" if "created_utc" in r else r["job_id"])
                     for r in sorted(d["jobs"], key=lambda r: r.get("created_utc", "")))
    block = f"""% BEGIN KOBE CAL
\\subsection{{\\code{{ibm\\_kobe}} calibration during the experiments}}
\\label{{sec:kobe-calibration}}
Median over the device's qubits (or couplers) of the calibration IBM reported as in force at the stated time (UTC), from the IBM Quantum
backend properties (fetched {d['fetched_utc'][:16].replace('T', ' ')} UTC). T$_1$/T$_2$ are medians; the last column is the worst two-qubit error on the device.
The first table samples the device every few hours over the last days of training and inference; the second gives the calibration in force when each
fresh-shot hardware sampling job was created. IBM recalibrates several times a day, so a job's actual calibration is the snapshot in force at its execution time. A worst two-qubit error of 100\\% means that IBM lists some couplers with error 1 (unusable; 10 of the 352 listed two-qubit gate entries, both directions, in the first snapshot); the median is unaffected.

\\begin{{table}}[ht]
\\centering
\\caption{{\\code{{ibm\\_kobe}} calibration snapshots over the last days of training and inference.}}
\\label{{tab:kobe-calibration-daily}}
\\scriptsize
{hdr}{days}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

{{\\scriptsize
\\begin{{longtable}}{{@{{}}p{{2.3in}}rrrrrr@{{}}}}
\\caption{{\\code{{ibm\\_kobe}} calibration in force at the creation of each fresh-shot sampling job (run, job creation time UTC). RO: readout error; 1q: single-qubit \\code{{sx}} error; 2q: median two-qubit \\code{{cz}} error; worst 2q: largest two-qubit error on the device.}}\\label{{tab:kobe-calibration-jobs}}\\\\
\\toprule
Run, job created (UTC) & {cols_short} \\\\
\\midrule
\\endfirsthead
\\toprule
Run, job created (UTC) & {cols_short} \\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot
{jobs}
\\end{{longtable}}}}
% END KOBE CAL
"""
    t = TEX.read_text()
    if "% BEGIN KOBE CAL" in t:
        t = re.sub(r"% BEGIN KOBE CAL.*?% END KOBE CAL\n", lambda m: block, t, flags=re.S)
    else:
        key = "\\section{Data and circuit availability}"
        assert key in t
        t = t.replace(key, block + "\n" + key, 1)
    TEX.write_text(t)
    n_ok = sum(1 for r in d["daily"] + d["jobs"] if "error" not in r)
    print(f"inserted calibration tables ({n_ok} of {len(d['daily']) + len(d['jobs'])} snapshots had data)")


if __name__ == "__main__":
    main()
