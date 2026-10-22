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

    def advantage(self, energies: torch.Tensor) -> torch.Tensor:
        return (energies.mean() - energies) / (energies.std() + 1e-8)


class GRPOObjective(GroupRelativeObjective):
    """Token-level clipped group-relative policy optimization."""

    def __call__(self, log_probs, context):
        self._check(context)
        win = torch.argmin(context["energies"])
        loss = -torch.mean(log_probs[win])
        if torch.std(context["energies"]) == 0:
            return loss
        adv = self.advantage(context["energies"])
        ratio = torch.exp(log_probs - context["old_log_probs"])
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
        clipped = (clipped_ratio * adv.unsqueeze(1)).mean()
        unclipped = (ratio * adv.unsqueeze(1)).mean()
        loss -= min(clipped, unclipped)
        return loss


class GSPOObjective(GroupRelativeObjective):
    """Sequence-level group-relative policy optimization."""

    def __call__(self, log_probs, context):
        self._check(context)
        seq_log_probs = log_probs.sum(dim=1)
        win = torch.argmin(context["energies"])
        loss = -torch.mean(log_probs[win])
        if torch.std(context["energies"]) == 0:
            return loss
        adv = self.advantage(context["energies"])
        seq_len = context["gate_seqs"].shape[1]
        old_seq_log_probs = context["old_log_probs"].sum(dim=1)
        ratio = torch.exp((seq_log_probs - old_seq_log_probs) / seq_len)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
        clipped = (clipped_ratio * adv).mean()
        unclipped = (ratio * adv).mean()
        loss -= min(clipped, unclipped)
        return loss
