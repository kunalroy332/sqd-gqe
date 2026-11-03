# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cross-attention conditioned transformer policy.

An alternative to ``ConditionedTransformerPolicy``'s additive bias: instead of
collapsing the conditioning to one vector and adding it at every token position,
the encoder emits a *prompt* of arbitrary length and the GPT-2 core attends to it
through cross-attention layers.

Why this matters for encoder comparison: the additive path forces every encoder
to produce exactly one vector, so an encoder whose natural output is per-orbital
(norb vectors) or per-atom (n_atoms vectors) has to be pooled first -- and then a
measured difference conflates "which encoder" with "how it was pooled". With
cross-attention the prompt length is free, so encoders can be swapped without
changing anything downstream. Measured on transformers 4.57.3: prompt lengths 7
and 13 both run through the same decoder, and changing the prompt changes the
logits (the cross-attention is actually consulted).

``add_cross_attention=True`` is mandatory. Without it, passing
``encoder_hidden_states`` raises ValueError -- it is not silently ignored, so
there is no risk of a run that looks fine while the conditioning does nothing.

Unlike the additive path, this one keeps the KV cache in ``act()``: the prompt is
identical at every generation step, so ``past_key_values`` stays valid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from sqd_gqe.core.models.policy import BasePolicy


class CompactCrossAttnConfig(GPT2Config):
    """6-layer/6-head GPT-2 with cross-attention, matching CompactConfig's size."""

    def __init__(self, **kwargs):
        kwargs.setdefault("n_layer", 6)
        kwargs.setdefault("n_head", 6)
        super().__init__(add_cross_attention=True, **kwargs)


class CrossAttnPolicy(GPT2LMHeadModel, BasePolicy):
    """GPT-2 policy conditioned via cross-attention on a variable-length prompt.

    The prompt is supplied by whatever encoder is under test through
    ``set_prompt(prompt)``; this class only requires it to be
    ``(1 or B, L, n_embd)`` with L free. Set it once per rollout, exactly like
    ``ConditionedTransformerPolicy.set_condition``, so the existing training loop
    needs no change beyond the ``hasattr`` dispatch it already does.
    """

    def __init__(self, small, repetition_penalty, vocab_size, ngates):
        max_positions = int(ngates) + 1
        cfg_cls = CompactCrossAttnConfig if small else GPT2Config
        cfg = cfg_cls(vocab_size=vocab_size, n_positions=max_positions,
                      **({} if small else {"add_cross_attention": True}))
        self.repetition_penalty = repetition_penalty
        super().__init__(cfg)
        self._prompt = None
        self._tril_cache: dict = {}

    # -- conditioning ------------------------------------------------------- #
    def set_prompt(self, prompt: torch.Tensor):
        """Store the encoder's prompt for subsequent act/log_prob.

        Kept attached to its graph (no ``detach()``): the encoder that produced
        it is trained jointly, so gradients must flow back through the prompt.
        This differs from ``set_condition``, which stores raw features and re-runs
        its encoder inside every forward -- here the caller re-runs the encoder
        per rollout instead, which is enough because the prompt does not change
        within a rollout.
        """
        self._prompt = prompt

    def _prompt_for(self, batch_size: int):
        """Broadcast the stored prompt to the current batch size.

        Uses ``expand().contiguous()`` rather than plain ``expand()``: GPT2's
        cross-attention runs ``c_attn(encoder_hidden_states)``, which does
        ``x.view(-1, x.size(-1))`` internally, and a bare expanded view fails that
        with "view size is not compatible with input tensor's size and stride"
        (stride 0 along the batch dim spans two contiguous subspaces). Hit this
        directly on the first run. ``repeat`` would work too but allocates the
        same memory; this keeps the intent (broadcast) visible.
        """
        if self._prompt is None:
            return None
        p = self._prompt
        if p.dim() == 2:                      # (L, E) -> (1, L, E)
            p = p.unsqueeze(0)
        if p.shape[0] == 1 and batch_size > 1:
            p = p.expand(batch_size, -1, -1).contiguous()
        return p

    # -- log-probabilities over full sequences ------------------------------ #
    def log_prob(self, indices, beta, return_entropy=False):
        enc = self._prompt_for(indices.shape[0])
        out = self(input_ids=indices, encoder_hidden_states=enc)
        logits = out.logits[:, :-1, :]        # position t predicts t+1
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

    # -- incremental sampling (KV cache is valid here) ---------------------- #
    def act(self, state, beta):
        idx = state["idx"]
        enc = self._prompt_for(idx.shape[0])
        past = state.get("past_key_values", None)
        # The prompt is constant across steps, so the cache stays correct -- the
        # additive path cannot do this because it rebuilds inputs_embeds each step.
        if past is None:
            out = self(input_ids=idx, encoder_hidden_states=enc, use_cache=True)
        else:
            out = self(input_ids=idx[:, -1:], encoder_hidden_states=enc,
                       past_key_values=past, use_cache=True)
        state["past_key_values"] = out.past_key_values
        logits = out.logits[:, -1, :]
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            logits = self._penalize_last(logits, idx, float(self.repetition_penalty))
        return Categorical(logits=-beta * logits).sample()

    # -- repetition penalty helpers (same math as TransformerPolicy) -------- #
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
