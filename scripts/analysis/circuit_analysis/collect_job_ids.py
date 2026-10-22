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

"""Scan cluster logs for IBM Runtime job ids and write ids.json for fetch_qpu_usage.py.

Groups (by log file name): training = hw_continue_*, kawasaki_*; first = infer_crossnorb_sample_* and infer_all_sample_* (first inference campaign);
lucj = infer_all_lucj_baseline*. An id is a 20-character token starting with 'd' found on a line that mentions 'job'. The per-file counts are printed
so the result can be sanity-checked: the id patterns in older logs were not verified, so a group that comes out empty or implausible must be checked by hand.
Usage: collect_job_ids.py LOG_DIR [--out ids.json]
"""
import argparse
import json
import re
from pathlib import Path

ID = re.compile(r"\b(d[a-z0-9]{19})\b")
GROUPS = {"training": ("hw_continue_", "kawasaki_"), "first": ("infer_crossnorb_sample_", "infer_all_sample_"), "lucj": ("infer_all_lucj_baseline", "infer_lucj_sample_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", default="ids.json")
    a = ap.parse_args()
    out = {g: [] for g in GROUPS}
    for f in sorted(Path(a.log_dir).glob("*")):
        grp = next((g for g, pre in GROUPS.items() if f.name.startswith(pre)), None)
        if not grp or not f.is_file():
            continue
        ids = []
        for line in f.read_text(errors="ignore").splitlines():
            if "job" in line.lower():
                ids += ID.findall(line)
        new = [i for i in dict.fromkeys(ids) if i not in out[grp]]
        out[grp] += new
        print(f"{grp:9s} {f.name:60s} {len(new)} new ids")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print({g: len(v) for g, v in out.items()})


if __name__ == "__main__":
    main()
