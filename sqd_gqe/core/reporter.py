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

"""Metric extraction and Weights & Biases reporting (self-contained)."""

from __future__ import annotations

from functools import singledispatch
from typing import Any, Mapping

import numpy as np

from sqd_gqe.core.report import SqdReport, SqdSample


class MetricReporter:
    def __init__(self, reference_energies: Mapping[str, float] | None = None):
        self.reference_energies = reference_energies

    def report(self, pl_module, entries: list[dict[str, Any]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for e in entries:
            metrics.update(collect_metrics(e["result"], prefix=e["prefix"], ref_energy=self.reference_energies))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return metrics


def _key(key: str, prefix: str) -> str:
    return key if prefix == "" else f"{prefix}/{key}"


@singledispatch
def collect_metrics(result: Any, *, prefix: str = "", ref_energy: dict[str, float] | None = None) -> dict[str, float]:
    return {}


@collect_metrics.register
def _(result: SqdSample, *, prefix: str = "", ref_energy: dict[str, float] | None = None) -> dict[str, float]:
    d: dict[str, float] = {}
    if ref_energy:
        for ref_key, ref_val in ref_energy.items():
            d[_key(f"energy - {ref_key}", prefix)] = result.energy - ref_val
    else:
        d[_key("energy", prefix)] = result.energy
    if result.subspace_size is not None:
        d[_key("subspace_dim", prefix)] = result.subspace_size
    if result.n_sampled is not None:
        d[_key("num_sampled_basis", prefix)] = result.n_sampled
    if result.n_symmetry_valid is not None:
        d[_key("num_symmetry_preserving_basis", prefix)] = result.n_symmetry_valid
    if result.cx_count is not None:
        d[_key("cx_count", prefix)] = result.cx_count
    if result.total_gates is not None:
        d[_key("total_gates", prefix)] = result.total_gates
    return d


@collect_metrics.register
def _(result: SqdReport, *, prefix: str = "", ref_energy: dict[str, float] | None = None) -> dict[str, float]:
    if len(result.samples) == 0:
        return {}
    d: dict[str, float] = {}
    d[_key("energy/min", prefix)] = min(result.energies)
    d[_key("energy/mean", prefix)] = float(np.mean(result.energies))
    d[_key("energy/std", prefix)] = float(np.std(result.energies))
    d[_key("subspace_dim/max", prefix)] = min(result.subspace_sizes)
    d[_key("cx_count/max", prefix)] = max(result.cx_counts)
    d[_key("total_gates/max", prefix)] = max(result.total_gates)
    return d
