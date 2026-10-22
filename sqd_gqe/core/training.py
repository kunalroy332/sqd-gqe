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

"""PyTorch-Lightning training loop for SQD-GQE (self-contained).

Runs the on-policy GQE loop: roll out a batch of circuits from the transformer
policy, score them by SQD subspace energy, push to the experience buffer, and
update the policy with a group-relative objective. Backend-agnostic -- it only
touches the factory's builders and the pipeline's ``run(state)`` contract.
"""

from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from sqd_gqe.core.buffer import ExperienceBuffer, ExperienceDataset
from sqd_gqe.core.configuration import make_ci_vector
from sqd_gqe.core.report import SqdSample


class SqdTrainer(pl.LightningModule):
    def __init__(self, factory, config):
        super().__init__()
        self.config = config
        self.factory = factory
        self.objective = factory.build_objective(config)
        self.pipeline = factory.build_pipeline(config)
        self.model = factory.build_policy(config)
        self.schedule = factory.build_beta_schedule(config)
        self.reporter = factory.build_reporter(config)
        self.warmup_size = config.trainer.warmup_size
        self.ngates = config.ngates
        self.num_samples = config.trainer.num_samples
        self.best: SqdSample | None = None
        self.best_local: SqdSample | None = None
        self.best_global: SqdSample | None = None
        self.buffer = ExperienceBuffer(size=config.trainer.buffer_size)

    # -- lifecycle ----------------------------------------------------------- #
    def on_fit_start(self):
        run = self.logger.experiment
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        while len(self.buffer) < self.warmup_size:
            self.rollout(log=False)
        super().on_fit_start()

    def on_train_epoch_start(self):
        report = self.rollout(log=True)
        entries = [
            {"result": report, "prefix": "GQE-optimized"},
            {"result": self.best, "prefix": "GQE-optimized(best_so_far)"},
        ]
        if self.best_local is not None:
            entries.append({"result": self.best_local, "prefix": "Local-refined(best_so_far)"})
        if self.best_global is not None:
            entries.append({"result": self.best_global, "prefix": "Global-refined(best_so_far)"})
        self.reporter.report(self, entries)
        super().on_train_epoch_start()

    def teardown(self, stage):
        # In session mode this tells IBM Runtime not to accept more jobs.  The
        # hook is also called when Lightning tears down after an exception.
        try:
            self.pipeline.close()
        finally:
            super().teardown(stage)

    # -- rollout ------------------------------------------------------------- #
    def rollout(self, log=False):
        state = {
            "idx": torch.zeros((self.num_samples, 1), dtype=torch.long, device=self.device)
        }
        with torch.no_grad():
            for _ in range(self.ngates):
                next_tokens = self.model.act(state, self.schedule.beta())
                state = self._append(state, next_tokens)

            report = self.pipeline.run(state)
            energies = torch.tensor(report.energies, device=self.device)
            behavior_log_probs = self.model.log_prob(state["idx"], self.schedule.beta())
            for seq, energy, blp in zip(state["idx"], energies, behavior_log_probs):
                self.buffer.add(seq.detach().cpu(), energy.detach().cpu(), blp.detach().cpu())

            if self.best is None or energies.min() < self.best.energy:
                self.best = report.best
            if self.best_local is None or report.local_refined.energy < self.best_local.energy:
                self.best_local = report.local_refined
            if report.global_refined is not None and (
                self.best_global is None or report.global_refined.energy < self.best_global.energy
            ):
                self.best_global = report.global_refined

        self.schedule.step(energies=energies)
        return report

    # -- optimization -------------------------------------------------------- #
    def training_step(self, batch, _):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        log_probs = self.model.log_prob(batch["idx"], self.schedule.beta())
        context = {
            "old_log_probs": batch["old_log_probs"],
            "energies": batch["energy"],
            "gate_seqs": batch["idx"][:, 1:],
        }
        loss = self.objective(log_probs, context)
        self.log("trainer/loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("trainer/beta", self.schedule.beta(), on_step=False, on_epoch=True, logger=True)
        return loss

    def _append(self, state, next_token):
        state["idx"] = torch.cat((state["idx"], next_token.unsqueeze(1)), dim=1)
        return state

    def configure_optimizers(self):
        opt_cfg = self.config.trainer.optimizer
        optimizer_cls = getattr(torch.optim, opt_cfg.cls)
        return {"optimizer": optimizer_cls(self.model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)}

    def train_dataloader(self):
        return DataLoader(
            ExperienceDataset(self.buffer, self.config.trainer.step_per_epoch),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
        )

    # -- checkpoint ---------------------------------------------------------- #
    def on_save_checkpoint(self, checkpoint):
        merged = self.pipeline.global_refined_state
        checkpoint["extra_info"] = {
            "beta": self.schedule.beta(),
            "best": self.best,
            "global_refined_state": {
                "coeffs": np.asarray(merged) if merged is not None else None,
                "strs": getattr(merged, "strings", None),
            },
        }
        self.buffer.save(f"{self.config.output}/buffer.pkl")

    def on_load_checkpoint(self, checkpoint):
        info = checkpoint.get("extra_info", {})
        if "beta" in info:
            self.schedule.current = info["beta"]
        if "best" in info:
            self.best = info["best"]
        state = info.get("global_refined_state")
        if state and state.get("coeffs") is not None:
            self.pipeline.global_refined_state = make_ci_vector(state["coeffs"], state["strs"])
        self.buffer.load(f"{self.config.output}/buffer.pkl")

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
