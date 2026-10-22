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

"""Inspect or export history from a local W&B ``run-*.wandb`` file.

This reads the binary record directly; it does not contact W&B or require the
run to be synced.  ``wandb`` must be installed, so on Miyabi this script should
normally be run inside the same Apptainer image used for training.

Examples::

    python benchmark/read_wandb_offline.py outputs/sqd-gqe/my-run/wandb/offline-run-*
    python benchmark/read_wandb_offline.py RUN_DIR --json history.json --csv history.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def find_run_file(path: str | Path) -> Path:
    """Resolve a .wandb file, an offline-run directory, or its parent."""
    source = Path(path)
    if source.is_file():
        if source.suffix != ".wandb":
            raise ValueError(f"Expected a .wandb file, got: {source}")
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Offline W&B path does not exist: {source}")

    matches = sorted(source.glob("run-*.wandb"))
    if not matches:
        matches = sorted(source.glob("offline-run-*/run-*.wandb"))
    if not matches:
        matches = sorted(source.glob("wandb/offline-run-*/run-*.wandb"))
    if len(matches) != 1:
        detail = "none" if not matches else ", ".join(str(p) for p in matches)
        raise ValueError(f"Expected exactly one run-*.wandb below {source}; found {detail}")
    return matches[0]


def _item_key(item: Any) -> str:
    if getattr(item, "key", ""):
        return item.key
    nested = list(getattr(item, "nested_key", ()))
    return "/".join(nested)


def _item_value(item: Any) -> Any:
    raw = item.value_json
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def read_history(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Return the resolved run file and all W&B history records."""
    run_file = find_run_file(path)
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:
        raise RuntimeError(
            "The wandb package is required to read its binary run file. "
            "Run this script inside the training Apptainer image."
        ) from exc

    store = DataStore()
    store.open_for_scan(str(run_file))
    rows: list[dict[str, Any]] = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row = {_item_key(item): _item_value(item) for item in record.history.item}
        row.pop("", None)
        if row:
            rows.append(row)
    return run_file, rows


def metric_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row})


def write_json(path: str | Path, run_file: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"source": str(run_file), "history": rows}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = metric_names(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value if value is None or isinstance(value, (str, int, float, bool))
                    else json.dumps(value)
                    for key, value in row.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help=".wandb file, offline-run directory, or W&B parent directory")
    parser.add_argument("--json", metavar="PATH", help="export all history records as JSON")
    parser.add_argument("--csv", metavar="PATH", help="export a flat history table as CSV")
    parser.add_argument("--list-metrics", action="store_true", help="print every recorded metric name")
    args = parser.parse_args()

    try:
        run_file, rows = read_history(args.run)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    epochs = [row["epoch"] for row in rows if isinstance(row.get("epoch"), (int, float))]
    epoch_summary = f", epochs {min(epochs):g}..{max(epochs):g}" if epochs else ""
    print(f"Read {len(rows)} history records{epoch_summary} from {run_file}")
    if args.list_metrics:
        print("\n".join(metric_names(rows)))
    if args.json:
        write_json(args.json, run_file, rows)
        print(f"Wrote {args.json}")
    if args.csv:
        write_csv(args.csv, rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
