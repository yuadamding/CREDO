"""Main training loop for the P4/P60 PINN.

Implements the pseudocode from Section 14 of the spec, with stage-wise
training from Section 13.6.

Stage C: control warm-start (embeddings frozen at zero, ecology off)
Stage D: perturbation warm-start (embeddings unfrozen, ecology off)
Stage E: ecology on growth (enabled)

Enhancements over baseline:
  - Cosine-annealing LR with linear warm-up
  - Exponential moving average (EMA) of model weights for evaluation
  - Trajectory regularization always active when lambda_reg > 0
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import nullcontext
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


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # after each optimizer step:
        ema.update()
        # for evaluation:
        ema.apply_shadow()
        evaluate(model)
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        self.backup = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up for `warmup_epochs`, then cosine decay to `eta_min`."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up from eta_min to base_lr
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                base_lr * (self.eta_min_ratio + (1.0 - self.eta_min_ratio) * alpha)
                for base_lr in self.base_lrs
            ]
        # Cosine decay after warm-up
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            base_lr * (self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cosine_factor)
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

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
    ema_decay: EMA decay rate (0 to disable)
    warmup_epochs: number of linear warm-up epochs
    """

    def __init__(
        self,
        model: FullDynamicsModel,
        config: RunConfig,
        endpoint: EndpointProblem,
        supported_pids: List[str],
        count_data: Optional[dict] = None,
        output_dir: str = "outputs",
        ema_decay: float = 0.995,
        warmup_epochs: int = 50,
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
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs

        tc = config.training
        sc = config.simulation
        # Always store trajectory history when any regularization is active
        has_trajectory_reg = (
            tc.lambda_reg_net > 0 or tc.lambda_reg_diffusion > 0
        )
        needs_history = (
            (tc.lambda_weak > 0) or (tc.lambda_count > 0) or has_trajectory_reg
        )
        self.precision = tc.precision
        self.autocast_enabled = self.device.startswith("cuda") and self.precision in {"fp16", "bf16"}
        self.compute_dtype = (
            torch.float16 if self.precision == "fp16"
            else torch.bfloat16 if self.precision == "bf16"
            else torch.float32
        )
        scaler_enabled = self.device.startswith("cuda") and self.precision == "fp16"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

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
        rollout_dtype = self.compute_dtype if self.autocast_enabled else self.dtype

        # Initialise particles from P4 endpoint
        z0, logw0, log_m0 = initialise_particles(
            self.endpoint,
            self.supported_pids,
            n_particles=sc.n_particles,
            device=self.device,
            dtype=rollout_dtype,
            seed=self.config.training.seed + epoch,
        )

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.compute_dtype)
            if self.autocast_enabled
            else nullcontext()
        )

        with autocast_ctx:
            rollout = self.simulator.rollout(
                z0=z0,
                logw0=logw0,
                model=self.model,
                log_m0=log_m0,
            )

        # --- Endpoint UOT loss (absolute log-weights) ---
        # Keep OT and mass terms in fp32 for numerical stability even when rollout used AMP.
        pred_logw_abs = rollout.terminal_logw.float() + log_m0.float().unsqueeze(-1)  # [G, N]
        loss_end, _ = self.uot_loss(
            pred_z=rollout.terminal_z.float(),
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
        embeddings = self.model.embedding(self.supported_pids).float()
        loss_reg = self.regularizer(
            embeddings=embeddings,
            drift_steps=rollout.drift_steps.float() if rollout.drift_steps is not None
                        else torch.zeros(1, G, sc.n_particles, self.model.latent_dim, device=self.device),
            sigma_steps=rollout.sigma_steps.float() if rollout.sigma_steps is not None
                        else torch.zeros(1, G, sc.n_particles, self.model.latent_dim, device=self.device),
            growth_steps=rollout.growth_steps.float() if rollout.growth_steps is not None
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
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
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

    def _save_checkpoint(self, epoch: int, tag: str = "best",
                         ema: Optional[EMA] = None) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "count_lik_state": self.count_lik.state_dict(),
            "config": self.config.model_dump(),
            "perturbation_ids": self.supported_pids,
        }
        if ema is not None:
            state["ema_state"] = ema.state_dict()
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(state, path)

    def _save_ema_checkpoint(self, epoch: int, ema: EMA) -> None:
        """Save a checkpoint with EMA weights applied."""
        ema.apply_shadow()
        path = self.output_dir / "checkpoint_best_ema.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "count_lik_state": self.count_lik.state_dict(),
            "config": self.config.model_dump(),
            "perturbation_ids": self.supported_pids,
            "is_ema": True,
        }, path)
        ema.restore()

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

        control_ref_warmup = 0
        if (
            stage == "all"
            and
            getattr(self.model, "control_mode", None) == "soft_ref"
            and getattr(self.model.embedding, "reference_embedding", None) is not None
            and tc.control_ref_warmup_epochs > 0
        ):
            control_ref_warmup = min(tc.control_ref_warmup_epochs, epochs)
            self.model.freeze_control_reference()
            if control_ref_warmup > 0:
                print(
                    f"[{stage}] Freezing control reference for the first "
                    f"{control_ref_warmup} epochs"
                )

        optimizer = self._build_optimizer(stage)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=epochs,
            eta_min_ratio=0.1,
        )

        # Initialize EMA
        ema = EMA(self.model, decay=self.ema_decay) if self.ema_decay > 0 else None

        start = time.time()

        for epoch in range(epochs):
            if control_ref_warmup > 0 and epoch == control_ref_warmup:
                self.model.unfreeze_control_reference()
                print(f"[{stage}] Released control reference at epoch {epoch}")

            metrics = self._one_epoch(optimizer, epoch)
            scheduler.step()

            # Update EMA after each optimizer step
            if ema is not None:
                ema.update()

            self.history.epochs.append(epoch)
            self.history.loss_total.append(metrics["loss_total"])
            self.history.loss_end.append(metrics["loss_end"])
            self.history.loss_weak.append(metrics["loss_weak"])
            self.history.loss_count.append(metrics["loss_count"])
            self.history.loss_reg.append(metrics["loss_reg"])

            # Best checkpoint (training weights)
            if metrics["loss_total"] < self._best_loss:
                self._best_loss = metrics["loss_total"]
                self._patience_counter = 0
                self._save_checkpoint(epoch, "best", ema=ema)
                # Also save EMA-specific checkpoint
                if ema is not None:
                    self._save_ema_checkpoint(epoch, ema)
            else:
                self._patience_counter += 1

            if epoch % tc.log_every == 0:
                elapsed = time.time() - start
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[{stage}] Epoch {epoch:4d}/{epochs} | "
                    f"total={metrics['loss_total']:.4f} "
                    f"end={metrics['loss_end']:.4f} "
                    f"weak={metrics['loss_weak']:.4f} "
                    f"count={metrics['loss_count']:.4f} "
                    f"reg={metrics['loss_reg']:.4f} | "
                    f"lr={cur_lr:.2e} t={elapsed:.1f}s"
                )

            if epoch % tc.checkpoint_every == 0:
                self._save_checkpoint(epoch, f"epoch{epoch:04d}", ema=ema)

            # Early stopping
            if self._patience_counter >= tc.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Final EMA checkpoint
        if ema is not None:
            self._save_ema_checkpoint(epochs - 1, ema)

        # Save history
        df = self.history.to_dataframe()
        df.to_csv(self.output_dir / "training_history.csv", index=False)
        return self.history
