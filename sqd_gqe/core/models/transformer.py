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

"""Transformer (decoder-only) policy for GQE circuit generation (self-contained).

A GPT-2 decoder produces logits over the operator-pool vocabulary. The policy
samples operator tokens autoregressively (with a KV cache for fast rollout and a
repetition penalty), and exposes per-step log-probabilities for the policy
update. Sampling temperature enters as an inverse-temperature (beta) factor on
the logits, matching the GQE energy-as-reward formulation.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from sqd_gqe.core.models.policy import BasePolicy


class CompactConfig(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(n_layer=6, n_head=6, **kwargs)


class TransformerPolicy(GPT2LMHeadModel, BasePolicy):
    def __init__(self, small, repetition_penalty, vocab_size, ngates):
        max_positions = int(ngates) + 1
        cfg = (CompactConfig if small else GPT2Config)(vocab_size=vocab_size, n_positions=max_positions)
        self.repetition_penalty = repetition_penalty
        super().__init__(cfg)
        self._tril_cache: dict = {}

    # -- log-probabilities over full sequences ------------------------------ #
    def log_prob(self, indices, beta, return_entropy=False):
        logits = self(indices).logits[:, :-1, :]   # (B, L-1, V); position t predicts t+1
        labels = indices[:, 1:]
        prefix = indices[:, :-1]

        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._penalize_sequence(logits, prefix, float(self.repetition_penalty))

        log_probs = F.log_softmax(-beta * logits, dim=-1)
        token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
        if return_entropy:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            return token_logp, entropy
        return token_logp

    # -- incremental sampling ----------------------------------------------- #
    def act(self, state, beta):
        idx = state["idx"]
        past = state.get("past_key_values", None)
        out = self(idx, use_cache=True) if past is None else self(idx[:, -1:], past_key_values=past, use_cache=True)
        state["past_key_values"] = out.past_key_values
        logits = out.logits[:, -1, :]
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._penalize_last(logits, idx, float(self.repetition_penalty))
        return Categorical(logits=-beta * logits).sample()

    # -- repetition penalty helpers ----------------------------------------- #
    @staticmethod
    def _penalize_last(logits, input_ids, penalty: float):
        gathered = torch.gather(logits, 1, input_ids)
        updated = torch.where(gathered < 0, gathered / penalty, gathered * penalty)
        return logits.scatter(1, input_ids, updated)

    def _tril(self, S: int, device: torch.device):
        key = (S, device.type, device.index if device.type == "cuda" else -1)
        cached = self._tril_cache.get(key)
        if cached is not None:
            return cached
        ti, sj = torch.tril_indices(S, S, offset=0, device=device)
        self._tril_cache[key] = (ti, sj)
        return ti, sj

    def _penalize_sequence(self, logits, prefix_ids, penalty: float):
        B, S, V = logits.shape
        device = logits.device
        ti, sj = self._tril(S, device)
        batch_offsets = (torch.arange(B, device=device) * S).unsqueeze(1)
        rows = (batch_offsets + ti.unsqueeze(0)).reshape(-1)
        cols = prefix_ids[:, sj].reshape(-1)
        flat = logits.reshape(B * S, V)
        vals = flat[rows, cols]
        updated = torch.where(vals < 0, vals / penalty, vals * penalty)
        out = flat.clone()
        out.index_put_((rows, cols), updated, accumulate=False)
        return out.view(B, S, V)
