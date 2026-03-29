"""Main training loop for the P4/P60 PINN.

Implements the pseudocode from Section 14 of the spec, with stage-wise
training from Section 13.6.

Stage C: control warm-start (embeddings frozen at zero, ecology off)
Stage D: perturbation warm-start (embeddings unfrozen, ecology off)
Stage E: ecology on growth (enabled)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config.schema import RunConfig
from ..data.core import EndpointProblem, FiniteMeasure
from ..losses.uot import UOTLoss
from ..losses.weak_form import WeakFormLoss
from ..losses.counts import CountLikelihood
from ..losses.regularizers import RolloutRegularizer
from ..models.full_model import FullDynamicsModel
from ..models.weighted_sde import WeightedParticleSimulator
from ..models.simulator import initialise_particles


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    loss_total: List[float] = field(default_factory=list)
    loss_end: List[float] = field(default_factory=list)
    loss_weak: List[float] = field(default_factory=list)
    loss_count: List[float] = field(default_factory=list)
    loss_reg: List[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "epoch": self.epochs,
            "loss_total": self.loss_total,
            "loss_end": self.loss_end,
            "loss_weak": self.loss_weak,
            "loss_count": self.loss_count,
            "loss_reg": self.loss_reg,
        })


def _build_target_dicts(
    endpoint: EndpointProblem,
    perturbation_ids: List[str],
    device: str,
    dtype: torch.dtype,
) -> Tuple[Dict, Dict]:
    """Build target support and log-weight dicts for UOT loss."""
    target_support, target_logw = {}, {}
    for pid in perturbation_ids:
        if pid not in endpoint.terminal:
            continue
        mu: FiniteMeasure = endpoint.terminal[pid]
        sup, w = mu.to_torch(device=device, dtype=dtype)
        target_support[pid] = sup
        target_logw[pid] = torch.log(w + 1e-30)
    return target_support, target_logw


class Trainer:
    """Main trainer for the full dynamics model.

    Parameters
    ----------
    model: FullDynamicsModel
    config: RunConfig
    endpoint: pooled EndpointProblem
    supported_pids: perturbation ids with sufficient support
    count_data: optional (growth_steps, logw_steps, tau_steps, exposures, counts, n_totals)
    output_dir: where to save checkpoints and logs
    """

    def __init__(
        self,
        model: FullDynamicsModel,
        config: RunConfig,
        endpoint: EndpointProblem,
        supported_pids: List[str],
        count_data: Optional[dict] = None,
        output_dir: str = "outputs",
    ) -> None:
        self.model = model
        self.config = config
        self.endpoint = endpoint
        self.supported_pids = supported_pids
        self.count_data = count_data
        self.device = config.resolve_device()
        self.dtype = torch.float32
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tc = config.training
        sc = config.simulation
        needs_history = (tc.lambda_weak > 0) or (tc.lambda_count > 0)

        self.simulator = WeightedParticleSimulator(
            n_steps=sc.n_steps,
            store_history=needs_history,
        )
        self.uot_loss = UOTLoss(
            eps=tc.sinkhorn_epsilon,
            tau=tc.sinkhorn_tau,
            max_iter=tc.sinkhorn_max_iter,
        )
        self.weak_loss = WeakFormLoss(
            n_test_functions=tc.n_test_functions,
            bandwidth=tc.test_function_bandwidth,
            latent_dim=config.latent.dim,
        )
        self.count_lik = CountLikelihood()
        self.regularizer = RolloutRegularizer(
            lambda_embed=tc.lambda_reg_embed,
            lambda_diffusion=tc.lambda_reg_diffusion,
            lambda_drift=tc.lambda_reg_net,
            lambda_growth=tc.lambda_reg_net,
        )

        self._target_support, self._target_logw = _build_target_dicts(
            endpoint, supported_pids, self.device, self.dtype)

        self.model.to(self.device)
        self.weak_loss.to(self.device)
        self.count_lik.to(self.device)

        self.history = TrainingHistory()
        self._best_loss = math.inf
        self._patience_counter = 0

    def _build_optimizer(self, stage: str) -> torch.optim.Optimizer:
        tc = self.config.training
        # Separate learning rates for embedding vs network params
        embed_params = []
        net_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "embedding" in name:
                embed_params.append(p)
            else:
                net_params.append(p)
        # Also include count_lik params
        net_params += list(self.count_lik.parameters())

        param_groups = [
            {"params": net_params, "lr": tc.lr_net},
            {"params": embed_params, "lr": tc.lr_embed},
        ]
        optimizer_cls = torch.optim.AdamW if tc.optimizer == "adamw" else torch.optim.Adam
        return optimizer_cls(param_groups, weight_decay=tc.weight_decay)

    def _one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        tc = self.config.training
        sc = self.config.simulation
        self.model.train()

        torch.manual_seed(self.config.training.seed + seed_offset + epoch)

        G = len(self.supported_pids)

        # Initialise particles from P4 endpoint
        z0, logw0, log_m0 = initialise_particles(
            self.endpoint,
            self.supported_pids,
            n_particles=sc.n_particles,
            device=self.device,
            dtype=self.dtype,
            seed=self.config.training.seed + epoch,
        )

        # Rollout
        rollout = self.simulator.rollout(
            z0=z0,
            logw0=logw0,
            model=self.model,
            log_m0=log_m0,
        )

        # --- Endpoint UOT loss (absolute log-weights) ---
        # logw from rollout is relative (starts at log(1/N)).
        # Absolute weight of particle i in group g = M0_g * exp(logw_i).
        # -> absolute log-weight = log_m0[g] + logw[g, i]
        pred_logw_abs = rollout.terminal_logw + log_m0.unsqueeze(-1)  # [G, N]
        loss_end, _ = self.uot_loss(
            pred_z=rollout.terminal_z,
            pred_logw_abs=pred_logw_abs,
            target_support=self._target_support,
            target_logw=self._target_logw,
            perturbation_ids=self.supported_pids,
        )

        # --- Weak-form residual loss ---
        loss_weak = torch.tensor(0.0, device=self.device)
        if tc.lambda_weak > 0 and rollout.drift_steps is not None:
            loss_weak = self.weak_loss(
                z_steps=rollout.z_steps,
                logw_steps=rollout.logw_steps,
                drift_steps=rollout.drift_steps,
                sigma_steps=rollout.sigma_steps,
                growth_steps=rollout.growth_steps,
                tau_steps=rollout.tau_steps,
                refresh_centers=(epoch % 10 == 0),
            )

        # --- Count likelihood ---
        loss_count = torch.tensor(0.0, device=self.device)
        if tc.lambda_count > 0 and self.count_data is not None and rollout.growth_steps is not None:
            cd = self.count_data
            loss_count = self.count_lik(
                growth_steps=rollout.growth_steps,
                logw_steps=rollout.logw_steps,
                tau_steps=rollout.tau_steps,
                exposures=torch.tensor(cd["exposures"], dtype=self.dtype, device=self.device),
                count_matrix=torch.tensor(cd["counts"], dtype=self.dtype, device=self.device),
                n_totals=torch.tensor(cd["n_totals"], dtype=self.dtype, device=self.device),
            )

        # --- Regularization ---
        embeddings = self.model.embedding(self.supported_pids)
        loss_reg = self.regularizer(
            embeddings=embeddings,
            drift_steps=rollout.drift_steps if rollout.drift_steps is not None
                        else torch.zeros(1, G, sc.n_particles, self.model.latent_dim, device=self.device),
            sigma_steps=rollout.sigma_steps if rollout.sigma_steps is not None
                        else torch.zeros(1, G, sc.n_particles, self.model.latent_dim, device=self.device),
            growth_steps=rollout.growth_steps if rollout.growth_steps is not None
                         else torch.zeros(1, G, sc.n_particles, device=self.device),
        )
        loss_reg = loss_reg + self.model.regularization()

        # --- Total ---
        loss = (
            tc.lambda_end * loss_end
            + tc.lambda_weak * loss_weak
            + tc.lambda_count * loss_count
            + loss_reg
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        optimizer.step()

        return {
            "loss_total": float(loss.item()),
            "loss_end": float(loss_end.item()),
            "loss_weak": float(loss_weak.item()),
            "loss_count": float(loss_count.item()),
            "loss_reg": float(loss_reg.item()),
        }

    def _save_checkpoint(self, epoch: int, tag: str = "best") -> None:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "count_lik_state": self.count_lik.state_dict(),
            "config": self.config.model_dump(),
            "perturbation_ids": self.supported_pids,
        }, path)

    def train(self, stage: str = "all", n_epochs: Optional[int] = None) -> TrainingHistory:
        tc = self.config.training
        epochs = n_epochs or tc.epochs

        # Stage-based parameter freezing
        if stage == "C":
            self.model.freeze_embeddings()
            self.model.freeze_ecology()
        elif stage == "D":
            self.model.unfreeze_embeddings()
            self.model.freeze_ecology()
        elif stage in ("E", "F"):
            self.model.unfreeze_embeddings()
            self.model.unfreeze_ecology()

        optimizer = self._build_optimizer(stage)
        start = time.time()

        for epoch in range(epochs):
            metrics = self._one_epoch(optimizer, epoch)

            self.history.epochs.append(epoch)
            self.history.loss_total.append(metrics["loss_total"])
            self.history.loss_end.append(metrics["loss_end"])
            self.history.loss_weak.append(metrics["loss_weak"])
            self.history.loss_count.append(metrics["loss_count"])
            self.history.loss_reg.append(metrics["loss_reg"])

            # Best checkpoint
            if metrics["loss_total"] < self._best_loss:
                self._best_loss = metrics["loss_total"]
                self._patience_counter = 0
                self._save_checkpoint(epoch, "best")
            else:
                self._patience_counter += 1

            if epoch % tc.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"[{stage}] Epoch {epoch:4d}/{epochs} | "
                    f"total={metrics['loss_total']:.4f} "
                    f"end={metrics['loss_end']:.4f} "
                    f"weak={metrics['loss_weak']:.4f} "
                    f"count={metrics['loss_count']:.4f} "
                    f"reg={metrics['loss_reg']:.4f} | "
                    f"t={elapsed:.1f}s"
                )

            if epoch % tc.checkpoint_every == 0:
                self._save_checkpoint(epoch, f"epoch{epoch:04d}")

            # Early stopping
            if self._patience_counter >= tc.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save history
        df = self.history.to_dataframe()
        df.to_csv(self.output_dir / "training_history.csv", index=False)
        return self.history
