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

"""Merge per-node resource-dataset shards (from parallelized
build_resource_dataset.py runs, one shard per idle-node job) into one
combined experiments/amortized/resource_dataset.json for train_resource_head.py.

Dedupes by (molecule, backend) -- last shard wins on a collision, so a shard
can be safely re-run/overwritten without hand-editing the merge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged: dict[tuple[str, str], dict] = {}
    for shard_path in args.shards:
        p = Path(shard_path)
        if not p.exists():
            print(f"[MERGE] skipping missing shard {p}")
            continue
        rows = json.loads(p.read_text())
        for row in rows:
            key = (row["molecule"], row.get("backend", "unknown"))
            merged[key] = row
        print(f"[MERGE] {p}: {len(rows)} rows")

    out_rows = list(merged.values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_rows, indent=2))
    print(f"wrote {args.out} ({len(out_rows)} unique (molecule, backend) rows "
          f"from {len(args.shards)} shards)")


if __name__ == "__main__":
    main()
