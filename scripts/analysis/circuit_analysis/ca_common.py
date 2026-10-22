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

"""Shared cache loading for the circuit-analysis scripts.

Tags are either a cross-norb run (<molecule_key>_<lineage>_<policy>, cache in
crossnorb70q_infer/cache) or an L-tier matched-norb run (L46_<lineage>_<policy>, cache in
unseen46q_infer/cache, whose cache names the molecule differently). All outputs of the
circuit-analysis pipeline go under crossnorb70q_infer for both.
"""
import json

import os
ROOT = "/path/to/sqd-gqe-work/experiments"
INFER_ROOT = os.environ.get("INFER_ROOT")   # set for the Training 2 inference arms: their caches live under their own tree
L46_LABEL = "SnOH3-me L (norb=23, 46q)"


def cache_path(tag: str) -> str:
    if INFER_ROOT:
        return f"{INFER_ROOT}/cache/crossnorb70q_{tag}_sample.json"
    if tag.startswith("L46_"):
        return f"{ROOT}/unseen46q_infer/cache/unseen46q_{tag[len('L46_'):]}_sample.json"
    return f"{ROOT}/crossnorb70q_infer/cache/crossnorb70q_{tag}_sample.json"


def normalize(cache: dict) -> dict:
    """Fill the molecule_* keys the L-tier cache lacks; cross-norb caches pass through."""
    if "molecule_path" not in cache and "molecule" in cache:
        cache["molecule_path"] = cache["molecule"]
        cache.setdefault("molecule_key", "L46")
        cache.setdefault("molecule_label", L46_LABEL)
    return cache


OLD_HOME = "/path/to/old-home"   # home before the 2026-10 file-system move
NEW_HOME = "/path/to/home"


def remap_paths(obj):
    """The stored caches contain absolute paths (checkpoints, molecule files) from before the file-system move; translate them on load."""
    if isinstance(obj, str):
        return obj.replace(OLD_HOME, NEW_HOME)
    if isinstance(obj, list):
        return [remap_paths(x) for x in obj]
    if isinstance(obj, dict):
        return {k: remap_paths(v) for k, v in obj.items()}
    return obj


def load_cache(tag: str) -> dict:
    with open(cache_path(tag)) as f:
        return normalize(remap_paths(json.load(f)))
