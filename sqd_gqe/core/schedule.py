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

"""Inverse-temperature (beta) schedules for GQE sampling (self-contained).

Beta controls the exploration/exploitation trade-off when the policy samples
operator sequences: low beta explores, high beta exploits.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class BetaSchedule(ABC):
    @abstractmethod
    def beta(self) -> float:
        ...

    @abstractmethod
    def step(self, **kwargs) -> None:
        ...


class LinearBeta(BetaSchedule):
    """Increase beta by a fixed delta each iteration."""

    def __init__(self, start: float, delta: float):
        self.current = start
        self.delta = delta

    def beta(self) -> float:
        return self.current

    def step(self, **kwargs):
        self.current += self.delta


class CosineBeta(BetaSchedule):
    """Oscillate beta between a min and max on a cosine cycle."""

    def __init__(self, minimum: float, maximum: float, period: int):
        self.minimum = minimum
        self.maximum = maximum
        self.period = period
        self.iteration = 0
        self.current = (maximum + minimum) / 2

    def beta(self) -> float:
        return self.current

    def step(self, **kwargs):
        self.iteration += 1
        self.current = (self.maximum + self.minimum) / 2 - (self.maximum - self.minimum) / 2 * math.cos(
            2 * math.pi * self.iteration / self.period
        )


class VarianceBeta(BetaSchedule):
    """Adapt beta to the batch energy variance (raise beta when converged).

    ``beta_max`` caps how far this can climb -- found empirically (Stage B's
    hardware-aware c2h4 retrain, 2026-08) that with no ceiling, beta kept
    rising past 1.0 to 1.16+ once the batch's energy variance stayed low,
    permanently killing exploration: gradient/weight-updates flatlined to
    ~0.001-0.01 and accuracy drifted WORSE than an earlier, less-exploited
    iteration (best err was at iteration 3, not anywhere in the later
    "converged" phase) -- the policy got stuck sampling a fixed, no-longer-
    improving circuit rather than continuing to search for a better one.
    Loosening target_var alone only delays this; it doesn't stop beta from
    eventually saturating once variance drops. Default matches the initial
    value used across this project's cost-aware retrains (0.5) plus enough
    headroom (2x) to still exploit meaningfully without going unbounded.
    """

    def __init__(self, initial: float, delta: float, target_var: float, beta_max: float = 1.0):
        self.current = initial
        self.delta = delta
        self.target_var = target_var
        self.beta_max = beta_max

    def beta(self) -> float:
        return self.current

    def step(self, **kwargs):
        energies = kwargs["energies"]
        if energies.var().item() > self.target_var:
            self.current += self.delta
        else:
            self.current -= self.delta
        self.current = max(min(self.current, self.beta_max), 0.01)
