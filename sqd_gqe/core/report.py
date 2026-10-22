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

"""Result records produced by one SQD iteration (self-contained)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SqdSample:
    """Outcome of diagonalizing one circuit's sampled subspace."""

    seq: tuple[int, ...] | None
    energy: float
    n_sampled: int | None
    n_symmetry_valid: int | None
    subspace_size: int
    cx_count: int | None
    total_gates: int | None
    counts: dict | None = None
    recovery_diagnostics: list | None = None
    depth: int | None = None
    diagonalize_skipped_deep: bool = False


@dataclass(frozen=True, slots=True)
class SqdReport:
    """All samples of one iteration plus the refined results."""

    samples: tuple[SqdSample, ...]
    local_refined: SqdSample | None = None
    global_refined: SqdSample | None = None

    @property
    def energies(self) -> list[float]:
        return [s.energy for s in self.samples]

    @property
    def subspace_sizes(self) -> list[int]:
        return [s.subspace_size for s in self.samples]

    @property
    def cx_counts(self) -> list[int]:
        return [s.cx_count for s in self.samples]

    @property
    def total_gates(self) -> list[int]:
        return [s.total_gates for s in self.samples]

    @property
    def best(self) -> SqdSample | None:
        return min(self.samples, key=lambda s: s.energy) if self.samples else None
