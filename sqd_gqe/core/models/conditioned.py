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

"""Molecule/backend-conditioned transformer policy.

Wraps the GPT-2 ``TransformerPolicy`` and injects a (molecule, backend)
conditioning vector as an additive bias on the token embeddings, so a single
set of weights generates circuits appropriate to the current system without
retraining. The conditioning vector is produced by ``ConditioningEncoder`` from
a size-invariant descriptor, enabling zero-shot transfer to unseen molecules.

The conditioning is set once per rollout via ``set_condition(features)``; the
underlying autoregressive ``act`` / ``log_prob`` are otherwise unchanged, so the
existing training loop works with no modification.
"""

from __future__ import annotations

import torch

from sqd_gqe.core.models.transformer import TransformerPolicy
from sqd_gqe.core.conditioning import ConditioningEncoder, FEATURE_DIM


class ConditionedTransformerPolicy(TransformerPolicy):
    def __init__(self, small, repetition_penalty, vocab_size, ngates, feature_dim: int = FEATURE_DIM):
        super().__init__(small, repetition_penalty, vocab_size, ngates)
        self.encoder = ConditioningEncoder(embed_dim=self.config.n_embd, feature_dim=feature_dim)
        self._cond_feats = None  # stored raw descriptor; encoder runs per forward

    def set_condition(self, features: torch.Tensor):
        """Store the (molecule, backend) descriptor for subsequent act/log_prob.

        We keep the raw features (not the encoded vector) and run the encoder
        fresh inside each forward, so every backward pass builds its own graph
        through the encoder --- required because the training loop calls
        backward() many times per rollout on the same conditioning.
        """
        self._cond_feats = features.detach().to(next(self.parameters()).device).float()

    def _inject(self, input_ids):
        """Token embeddings + conditioning bias -> inputs_embeds for the GPT-2 core."""
        emb = self.transformer.wte(input_ids)          # (B, L, E)
        if self._cond_feats is not None:
            cond = self.encoder(self._cond_feats)      # fresh graph each forward
            emb = emb + cond.view(1, 1, -1)
        return emb

    # -- override act/log_prob to route through inputs_embeds --------------- #
    def log_prob(self, indices, beta, return_entropy=False):
        from torch.nn import functional as F
        inputs_embeds = self._inject(indices)
        logits = self(inputs_embeds=inputs_embeds).logits[:, :-1, :]
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

    def act(self, state, beta):
        from torch.distributions import Categorical
        idx = state["idx"]
        # conditioned path uses a full forward on inputs_embeds each step (no KV
        # cache) -- simpler and correct; ngates is small so cost is modest.
        inputs_embeds = self._inject(idx)
        logits = self(inputs_embeds=inputs_embeds).logits[:, -1, :]
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._penalize_last(logits, idx, float(self.repetition_penalty))
        return Categorical(logits=-beta * logits).sample()
