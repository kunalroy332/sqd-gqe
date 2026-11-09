# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Policy-gradient objectives for GQE training (self-contained).

Group-relative objectives that turn subspace energies into advantages and apply
a clipped importance-sampling update, following GRPO and the sequence-level GSPO
variant. Lower energy -> higher reward -> higher advantage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class PolicyObjective(ABC):
    @abstractmethod
    def __call__(self, log_probs: torch.Tensor, context: dict) -> torch.Tensor:
        ...

    def _check(self, context: dict) -> None:
        missing = self.required_keys - set(context.keys())
        if missing:
            raise ValueError(
                f"{type(self).__name__} needs context keys {sorted(missing)}; got {sorted(context.keys())}"
            )


class GroupRelativeObjective(PolicyObjective):
    def __init__(self, clip_low: float = 0.2, clip_high: float = 0.28):
        self.required_keys = {"gate_seqs", "energies", "old_log_probs"}
        self.clip_low = clip_low
        self.clip_high = clip_high

    @staticmethod
    def _finite_energies(energies: torch.Tensor) -> torch.Tensor | None:
        """Rank failed solves last; return None when the whole group failed."""
        finite = torch.isfinite(energies)
        if not bool(finite.any()):
            return None
        if not bool(finite.all()):
            # Worst finite + 10 Ha. cost_reward.cost_adjusted_energies uses best finite + 10 Ha
            # on its own (benchmark-only) path. Not a shared constant: both rank a failure
            # last while the finite spread stays under 10 Ha.
            energies = torch.where(finite, energies, energies[finite].max() + 10.0)
        return energies

    @staticmethod
    def _normalise(energies):
        return (energies.mean() - energies) / (energies.std() + 1e-8)

    def __call__(self, log_probs, context):
        self._check(context)
        energies = self._finite_energies(context["energies"])
        if energies is None:
            return log_probs.sum() * 0.0
        win = torch.argmin(energies)
        loss = -torch.mean(log_probs[win])
        if torch.std(energies) == 0:
            return loss
        adv = self._normalise(energies)
        ratio = self._ratio(log_probs, context)
        if ratio.ndim == 2:
            adv = adv.unsqueeze(1)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
        return loss - min((clipped_ratio * adv).mean(), (ratio * adv).mean())

    @abstractmethod
    def _ratio(self, log_probs, context):
        """Importance ratio per token (GRPO) or per sequence (GSPO)."""


class GRPOObjective(GroupRelativeObjective):
    """Token-level clipped group-relative policy optimization."""

    def _ratio(self, log_probs, context):
        return torch.exp(log_probs - context["old_log_probs"])


class GSPOObjective(GroupRelativeObjective):
    """Sequence-level group-relative policy optimization."""

    def _ratio(self, log_probs, context):
        seq_len = context["gate_seqs"].shape[1]
        return torch.exp((log_probs.sum(dim=1) - context["old_log_probs"].sum(dim=1)) / seq_len)
