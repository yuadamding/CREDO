"""Main training loop for the P4/P60 PINN.

The trainer supports either a single joint stage (`stage="all"`) or explicit
stage-wise warm starts:

Stage C: controls only, embeddings frozen, ecology off
Stage D: all perturbations, embeddings unfrozen, ecology off
Stage E: all perturbations, ecology on growth

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
from ..models.context import GroupStatistics
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
    stages: List[str] = field(default_factory=list)
    n_active_perturbations: List[int] = field(default_factory=list)
    perturbation_batch_size: List[int] = field(default_factory=list)
    loss_total: List[float] = field(default_factory=list)
    loss_end: List[float] = field(default_factory=list)
    loss_weak: List[float] = field(default_factory=list)
    loss_count: List[float] = field(default_factory=list)
    loss_reg: List[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "epoch": self.epochs,
            "stage": self.stages,
            "n_active_perturbations": self.n_active_perturbations,
            "perturbation_batch_size": self.perturbation_batch_size,
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
        self.training_devices = config.resolve_training_devices()
        self.device = self.training_devices[0]
        self.dtype = torch.float32
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs
        self._model_replicas: Dict[str, FullDynamicsModel] = {}
        self._weak_loss_replicas: Dict[str, WeakFormLoss] = {}

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

    def _can_use_multi_gpu(self) -> bool:
        return (
            len(self.training_devices) > 1
            and all(device.startswith("cuda") for device in self.training_devices)
            and torch.cuda.is_available()
        )

    def _split_perturbation_ids(self, perturbation_ids: List[str], devices: List[str]) -> List[Tuple[str, List[str], slice]]:
        n_items = len(perturbation_ids)
        n_shards = min(len(devices), n_items)
        if n_shards <= 1:
            return [(devices[0], perturbation_ids, slice(0, n_items))]
        base = n_items // n_shards
        extra = n_items % n_shards
        shards: List[Tuple[str, List[str], slice]] = []
        start = 0
        for shard_idx, device in enumerate(devices[:n_shards]):
            size = base + (1 if shard_idx < extra else 0)
            stop = start + size
            local_pids = perturbation_ids[start:stop]
            if local_pids:
                shards.append((device, local_pids, slice(start, stop)))
            start = stop
        return shards

    def _perturbation_batch_size(self, perturbation_ids: List[str]) -> int:
        limit = int(getattr(self.config.training, "max_active_perturbations", 0) or 0)
        if limit <= 0:
            return len(perturbation_ids)
        return max(1, min(limit, len(perturbation_ids)))

    def _chunk_perturbation_ids(self, perturbation_ids: List[str]) -> List[List[str]]:
        batch_size = self._perturbation_batch_size(perturbation_ids)
        if batch_size >= len(perturbation_ids):
            return [perturbation_ids]
        return [
            perturbation_ids[start:start + batch_size]
            for start in range(0, len(perturbation_ids), batch_size)
        ]

    def _sync_replica_from_primary(self, replica: FullDynamicsModel) -> None:
        replica.load_state_dict(self.model.state_dict())
        primary_params = dict(self.model.named_parameters())
        for name, param in replica.named_parameters():
            if name in primary_params:
                param.requires_grad_(primary_params[name].requires_grad)
        primary_buffers = dict(self.model.named_buffers())
        for name, buffer in replica.named_buffers():
            if name in primary_buffers:
                buffer.copy_(primary_buffers[name].to(device=buffer.device, dtype=buffer.dtype))
        replica.train(self.model.training)

    def _get_model_for_device(self, device: str) -> FullDynamicsModel:
        if device == self.device:
            return self.model
        replica = self._model_replicas.get(device)
        if replica is None:
            replica = copy.deepcopy(self.model).to(device)
            self._model_replicas[device] = replica
        self._sync_replica_from_primary(replica)
        return replica

    def _sync_weak_loss_state(self, weak_loss: WeakFormLoss) -> None:
        weak_loss.load_state_dict(self.weak_loss.state_dict())
        weak_loss._centers_initialized = self.weak_loss._centers_initialized
        if hasattr(self.weak_loss, "_adaptive_bandwidth"):
            weak_loss._adaptive_bandwidth = self.weak_loss._adaptive_bandwidth
        elif hasattr(weak_loss, "_adaptive_bandwidth"):
            delattr(weak_loss, "_adaptive_bandwidth")

    def _get_weak_loss_for_device(self, device: str) -> WeakFormLoss:
        if device == self.device:
            return self.weak_loss
        replica = self._weak_loss_replicas.get(device)
        if replica is None:
            replica = copy.deepcopy(self.weak_loss).to(device)
            self._weak_loss_replicas[device] = replica
        self._sync_weak_loss_state(replica)
        return replica

    def _prepare_multi_gpu_weak_losses(self, z_ref: torch.Tensor, refresh_centers: bool) -> None:
        if not refresh_centers and self.weak_loss._centers_initialized:
            for device in self.training_devices[1:]:
                self._sync_weak_loss_state(self._get_weak_loss_for_device(device))
            return
        if refresh_centers or not self.weak_loss._centers_initialized:
            self.weak_loss.refresh_test_functions(z_ref.float())
        for device in self.training_devices[1:]:
            self._sync_weak_loss_state(self._get_weak_loss_for_device(device))

    def _accumulate_replica_gradients(
        self,
        replicas: Dict[str, FullDynamicsModel],
    ) -> None:
        primary_params = dict(self.model.named_parameters())
        for device, replica in replicas.items():
            if device == self.device:
                continue
            for name, replica_param in replica.named_parameters():
                if replica_param.grad is None or name not in primary_params:
                    continue
                primary_param = primary_params[name]
                if not primary_param.requires_grad:
                    continue
                grad = replica_param.grad.detach().to(self.device)
                if primary_param.grad is None:
                    primary_param.grad = grad.clone()
                else:
                    primary_param.grad.add_(grad)

    @staticmethod
    def _weighted_shard_loss(loss_value: torch.Tensor, local_groups: int, total_groups: int) -> torch.Tensor:
        if total_groups <= 0:
            return loss_value
        return loss_value * (float(local_groups) / float(total_groups))

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
        stage: str,
        perturbation_ids: List[str],
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        if not self._can_use_multi_gpu():
            batch_size = self._perturbation_batch_size(perturbation_ids)
            if batch_size < len(perturbation_ids):
                return self._one_epoch_chunked(
                    optimizer=optimizer,
                    epoch=epoch,
                    stage=stage,
                    perturbation_ids=perturbation_ids,
                    seed_offset=seed_offset,
                )
        if self._can_use_multi_gpu() and len(perturbation_ids) > 1:
            return self._one_epoch_multi_gpu(
                optimizer=optimizer,
                epoch=epoch,
                stage=stage,
                perturbation_ids=perturbation_ids,
                seed_offset=seed_offset,
            )

        tc = self.config.training
        sc = self.config.simulation
        self.model.train()

        torch.manual_seed(self.config.training.seed + seed_offset + epoch)

        G = len(perturbation_ids)
        rollout_dtype = self.compute_dtype if self.autocast_enabled else self.dtype

        # Initialise particles from P4 endpoint
        z0, logw0, log_m0 = initialise_particles(
            self.endpoint,
            perturbation_ids,
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
                perturbation_ids=perturbation_ids,
            )

        # --- Endpoint UOT loss (absolute log-weights) ---
        # Keep OT and mass terms in fp32 for numerical stability even when rollout used AMP.
        pred_logw_abs = rollout.terminal_logw.float() + log_m0.float().unsqueeze(-1)  # [G, N]
        loss_end, _ = self.uot_loss(
            pred_z=rollout.terminal_z.float(),
            pred_logw_abs=pred_logw_abs,
            target_support=self._target_support,
            target_logw=self._target_logw,
            perturbation_ids=perturbation_ids,
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
        if (
            tc.lambda_count > 0
            and self.count_data is not None
            and rollout.growth_steps is not None
            and perturbation_ids == self.supported_pids
        ):
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
        embeddings = self.model.embedding(perturbation_ids).float()
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
            "n_active_perturbations": len(perturbation_ids),
            "perturbation_batch_size": len(perturbation_ids),
            "loss_total": float(loss.item()),
            "loss_end": float(loss_end.item()),
            "loss_weak": float(loss_weak.item()),
            "loss_count": float(loss_count.item()),
            "loss_reg": float(loss_reg.item()),
        }

    def _one_epoch_chunked(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        stage: str,
        perturbation_ids: List[str],
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        tc = self.config.training
        sc = self.config.simulation
        if tc.lambda_count > 0:
            raise NotImplementedError(
                "Perturbation chunking is not implemented for count loss."
            )

        self.model.train()
        torch.manual_seed(self.config.training.seed + seed_offset + epoch)

        rollout_dtype = self.compute_dtype if self.autocast_enabled else self.dtype
        perturbation_chunks = self._chunk_perturbation_ids(perturbation_ids)
        total_groups = len(perturbation_ids)
        batch_size = max(len(chunk) for chunk in perturbation_chunks)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.compute_dtype)
            if self.autocast_enabled
            else nullcontext()
        )

        if tc.lambda_weak > 0 and perturbation_chunks:
            refresh_centers = (epoch % 10 == 0) or (not self.weak_loss._centers_initialized)
            if refresh_centers:
                z_ref, _, _ = initialise_particles(
                    self.endpoint,
                    perturbation_ids,
                    n_particles=sc.n_particles,
                    device=self.device,
                    dtype=rollout_dtype,
                    seed=self.config.training.seed + seed_offset + epoch,
                )
                self.weak_loss.refresh_test_functions(z_ref.float())
                del z_ref

        metrics = {
            "n_active_perturbations": len(perturbation_ids),
            "perturbation_batch_size": batch_size,
            "loss_total": 0.0,
            "loss_end": 0.0,
            "loss_weak": 0.0,
            "loss_count": 0.0,
            "loss_reg": 0.0,
        }

        optimizer.zero_grad()

        for chunk_idx, chunk_pids in enumerate(perturbation_chunks):
            local_groups = len(chunk_pids)
            chunk_seed = self.config.training.seed + seed_offset + epoch * 1000 + chunk_idx
            torch.manual_seed(chunk_seed)

            z0, logw0, log_m0 = initialise_particles(
                self.endpoint,
                chunk_pids,
                n_particles=sc.n_particles,
                device=self.device,
                dtype=rollout_dtype,
                seed=chunk_seed,
            )

            with autocast_ctx:
                rollout = self.simulator.rollout(
                    z0=z0,
                    logw0=logw0,
                    model=self.model,
                    log_m0=log_m0,
                    perturbation_ids=chunk_pids,
                )

            pred_logw_abs = rollout.terminal_logw.float() + log_m0.float().unsqueeze(-1)
            loss_end, _ = self.uot_loss(
                pred_z=rollout.terminal_z.float(),
                pred_logw_abs=pred_logw_abs,
                target_support=self._target_support,
                target_logw=self._target_logw,
                perturbation_ids=chunk_pids,
            )

            loss_weak = torch.tensor(0.0, device=self.device)
            if tc.lambda_weak > 0 and rollout.drift_steps is not None:
                local_loss_weak = self.weak_loss(
                    z_steps=rollout.z_steps,
                    logw_steps=rollout.logw_steps,
                    drift_steps=rollout.drift_steps,
                    sigma_steps=rollout.sigma_steps,
                    growth_steps=rollout.growth_steps,
                    tau_steps=rollout.tau_steps,
                    refresh_centers=False,
                )
                loss_weak = self._weighted_shard_loss(
                    local_loss_weak,
                    local_groups,
                    total_groups,
                )

            loss_reg = self.regularizer(
                embeddings=self.model.embedding(chunk_pids).float(),
                drift_steps=rollout.drift_steps.float()
                if rollout.drift_steps is not None
                else torch.zeros(1, local_groups, sc.n_particles, self.model.latent_dim, device=self.device),
                sigma_steps=rollout.sigma_steps.float()
                if rollout.sigma_steps is not None
                else torch.zeros(1, local_groups, sc.n_particles, self.model.latent_dim, device=self.device),
                growth_steps=rollout.growth_steps.float()
                if rollout.growth_steps is not None
                else torch.zeros(1, local_groups, sc.n_particles, device=self.device),
            )
            loss_reg = self._weighted_shard_loss(loss_reg, local_groups, total_groups)

            loss = (
                tc.lambda_end * loss_end
                + tc.lambda_weak * loss_weak
                + loss_reg
            )

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            metrics["loss_total"] += float(loss.item())
            metrics["loss_end"] += float(loss_end.item())
            metrics["loss_weak"] += float(loss_weak.item())
            metrics["loss_reg"] += float(loss_reg.item())

            del rollout, pred_logw_abs, z0, logw0, log_m0

        model_reg = self.model.regularization()
        if self.scaler.is_enabled():
            self.scaler.scale(model_reg).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            model_reg.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            optimizer.step()

        metrics["loss_reg"] += float(model_reg.item())
        metrics["loss_total"] += float(model_reg.item())
        return metrics

    def _one_epoch_multi_gpu(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        stage: str,
        perturbation_ids: List[str],
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        tc = self.config.training
        sc = self.config.simulation
        if tc.lambda_count > 0:
            raise NotImplementedError("Multi-GPU single-model training does not yet support count loss.")
        if self.precision == "fp16":
            raise NotImplementedError("Multi-GPU single-model training currently supports fp32/bf16, not fp16.")

        self.model.train()
        torch.manual_seed(self.config.training.seed + seed_offset + epoch)

        rollout_dtype = self.compute_dtype if self.autocast_enabled else self.dtype
        shards = self._split_perturbation_ids(perturbation_ids, self.training_devices)
        if len(shards) <= 1:
            raise RuntimeError("Multi-GPU epoch requested without a real perturbation shard split.")

        models: Dict[str, FullDynamicsModel] = {}
        for device, _, _ in shards:
            models[device] = self._get_model_for_device(device)
            models[device].train()

        z0_full, logw0_full, log_m0_full = initialise_particles(
            self.endpoint,
            perturbation_ids,
            n_particles=sc.n_particles,
            device=self.device,
            dtype=rollout_dtype,
            seed=self.config.training.seed + epoch,
        )

        if tc.lambda_weak > 0:
            self._prepare_multi_gpu_weak_losses(
                z_ref=z0_full,
                refresh_centers=(epoch % 10 == 0),
            )

        def autocast_ctx_for(device: str):
            if self.autocast_enabled and device.startswith("cuda"):
                return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
            return nullcontext()

        shard_state: Dict[str, Dict[str, object]] = {}
        for device, local_pids, local_slice in shards:
            shard_state[device] = {
                "pids": local_pids,
                "slice": local_slice,
                "z": z0_full[local_slice].to(device),
                "logw": logw0_full[local_slice].to(device),
                "log_m0": log_m0_full[local_slice].to(device),
                "z_list": [z0_full[local_slice].to(device)],
                "logw_list": [logw0_full[local_slice].to(device)],
                "drift_list": [],
                "sigma_list": [],
                "growth_list": [],
            }

        tau_steps = torch.linspace(
            0.0,
            1.0,
            sc.n_steps + 1,
            device=self.device,
            dtype=rollout_dtype,
        )
        dtau = 1.0 / sc.n_steps
        total_groups = len(perturbation_ids)

        optimizer.zero_grad()
        for device, replica in models.items():
            if device != self.device:
                replica.zero_grad(set_to_none=True)

        for step_idx in range(sc.n_steps):
            tau_primary = tau_steps[step_idx]
            summary_parts: list[GroupStatistics] = []
            local_cache: Dict[str, Dict[str, torch.Tensor]] = {}

            for device, local_pids, _ in shards:
                state = shard_state[device]
                model = models[device]
                z_local = state["z"]
                logw_local = state["logw"]
                log_m0_local = state["log_m0"]
                with autocast_ctx_for(device):
                    a_local = model.embedding(local_pids)
                    eta_local, phi_local = model.context_agg.encode_particles(z_local)
                    stats_local, _, _ = model.context_agg.summarize_groups(
                        z_local,
                        logw_local,
                        log_m0_local,
                        eta=eta_local,
                        phi=phi_local,
                    )
                summary_parts.append(
                    GroupStatistics(
                        log_n_g=stats_local.log_n_g.to(self.device),
                        eta_g=stats_local.eta_g.to(self.device),
                        phi_g=stats_local.phi_g.to(self.device),
                    )
                )
                local_cache[device] = {
                    "a": a_local,
                    "eta": eta_local,
                }

            global_stats = GroupStatistics(
                log_n_g=torch.cat([stats.log_n_g for stats in summary_parts], dim=0),
                eta_g=torch.cat([stats.eta_g for stats in summary_parts], dim=0),
                phi_g=torch.cat([stats.phi_g for stats in summary_parts], dim=0),
            )
            global_context = self.model.context_agg.context_from_group_statistics(global_stats)

            noise_full = torch.randn_like(z0_full)

            for device, local_pids, local_slice in shards:
                state = shard_state[device]
                model = models[device]
                z_local = state["z"]
                logw_local = state["logw"]
                a_local = local_cache[device]["a"]
                eta_local = local_cache[device]["eta"]
                tau_local = tau_primary.to(device)
                ctx_local = global_context.context.to(device=device, dtype=z_local.dtype)
                q_local = global_context.q.to(device=device, dtype=z_local.dtype)
                s_local = global_context.s.to(device=device, dtype=z_local.dtype)
                with autocast_ctx_for(device):
                    coeffs = model.coeff_nets(
                        z=z_local,
                        tau=tau_local,
                        context=ctx_local,
                        a=a_local,
                        eta_z=eta_local,
                        q=q_local,
                        s=s_local,
                    )

                state["drift_list"].append(coeffs.drift)
                state["sigma_list"].append(coeffs.sigma_diag)
                state["growth_list"].append(coeffs.growth)

                noise_local = noise_full[local_slice].to(device)
                z_next = z_local + coeffs.drift * dtau + coeffs.sigma_diag * (dtau ** 0.5) * noise_local
                logw_next = logw_local + coeffs.growth * dtau
                state["z"] = z_next
                state["logw"] = logw_next
                state["z_list"].append(z_next)
                state["logw_list"].append(logw_next)

        loss_end = torch.tensor(0.0, device=self.device)
        loss_weak = torch.tensor(0.0, device=self.device)
        loss_count = torch.tensor(0.0, device=self.device)
        loss_reg = torch.tensor(0.0, device=self.device)

        for device, local_pids, _ in shards:
            state = shard_state[device]
            local_groups = len(local_pids)
            z_steps = torch.stack(state["z_list"], dim=0)
            logw_steps = torch.stack(state["logw_list"], dim=0)
            drift_steps = torch.stack(state["drift_list"], dim=0)
            sigma_steps = torch.stack(state["sigma_list"], dim=0)
            growth_steps = torch.stack(state["growth_list"], dim=0)
            pred_logw_abs = logw_steps[-1].float() + state["log_m0"].float().unsqueeze(-1)
            local_target_support, local_target_logw = _build_target_dicts(
                self.endpoint,
                local_pids,
                device,
                self.dtype,
            )
            local_loss_end, _ = self.uot_loss(
                pred_z=z_steps[-1].float(),
                pred_logw_abs=pred_logw_abs,
                target_support=local_target_support,
                target_logw=local_target_logw,
                perturbation_ids=local_pids,
            )
            loss_end = loss_end + local_loss_end.to(self.device)

            if tc.lambda_weak > 0:
                weak_loss_module = self._get_weak_loss_for_device(device)
                local_loss_weak = weak_loss_module(
                    z_steps=z_steps,
                    logw_steps=logw_steps,
                    drift_steps=drift_steps,
                    sigma_steps=sigma_steps,
                    growth_steps=growth_steps,
                    tau_steps=tau_steps.to(device),
                    refresh_centers=False,
                )
                loss_weak = loss_weak + self._weighted_shard_loss(
                    local_loss_weak,
                    local_groups,
                    total_groups,
                ).to(self.device)

            local_embeddings = models[device].embedding(local_pids).float()
            local_loss_reg = self.regularizer(
                embeddings=local_embeddings,
                drift_steps=drift_steps.float(),
                sigma_steps=sigma_steps.float(),
                growth_steps=growth_steps.float(),
            )
            loss_reg = loss_reg + self._weighted_shard_loss(
                local_loss_reg,
                local_groups,
                total_groups,
            ).to(self.device)

        loss_reg = loss_reg + self.model.regularization()
        loss = (
            tc.lambda_end * loss_end
            + tc.lambda_weak * loss_weak
            + tc.lambda_count * loss_count
            + loss_reg
        )

        loss.backward()
        self._accumulate_replica_gradients(models)
        nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        optimizer.step()

        return {
            "n_active_perturbations": len(perturbation_ids),
            "perturbation_batch_size": len(perturbation_ids),
            "loss_total": float(loss.item()),
            "loss_end": float(loss_end.item()),
            "loss_weak": float(loss_weak.item()),
            "loss_count": float(loss_count.item()),
            "loss_reg": float(loss_reg.item()),
        }

    def _active_perturbation_ids(self, stage: str) -> List[str]:
        if stage != "C":
            return self.supported_pids
        control_ids = [pid for pid in self.supported_pids if pid in self.model.control_ids]
        if not control_ids:
            raise ValueError("Stage C requested but no control perturbations are available.")
        return control_ids

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
        start_epoch = (self.history.epochs[-1] + 1) if self.history.epochs else 0
        self._best_loss = math.inf
        self._patience_counter = 0
        active_pids = self._active_perturbation_ids(stage)
        perturbation_batch_size = self._perturbation_batch_size(active_pids)

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

            absolute_epoch = start_epoch + epoch
            metrics = self._one_epoch(
                optimizer,
                epoch,
                stage=stage,
                perturbation_ids=active_pids,
            )
            scheduler.step()

            # Update EMA after each optimizer step
            if ema is not None:
                ema.update()

            self.history.epochs.append(absolute_epoch)
            self.history.stages.append(stage)
            self.history.n_active_perturbations.append(int(metrics.get("n_active_perturbations", len(active_pids))))
            self.history.perturbation_batch_size.append(int(metrics.get("perturbation_batch_size", perturbation_batch_size)))
            self.history.loss_total.append(metrics["loss_total"])
            self.history.loss_end.append(metrics["loss_end"])
            self.history.loss_weak.append(metrics["loss_weak"])
            self.history.loss_count.append(metrics["loss_count"])
            self.history.loss_reg.append(metrics["loss_reg"])

            # Best checkpoint (training weights)
            if metrics["loss_total"] < self._best_loss:
                self._best_loss = metrics["loss_total"]
                self._patience_counter = 0
                self._save_checkpoint(absolute_epoch, "best", ema=ema)
                # Also save EMA-specific checkpoint
                if ema is not None:
                    self._save_ema_checkpoint(absolute_epoch, ema)
            else:
                self._patience_counter += 1

            if epoch % tc.log_every == 0:
                elapsed = time.time() - start
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[{stage}] Epoch {absolute_epoch:4d} "
                    f"(stage {epoch + 1:4d}/{epochs}, pids={len(active_pids)}, "
                    f"batch_pids={int(metrics.get('perturbation_batch_size', perturbation_batch_size))}) | "
                    f"total={metrics['loss_total']:.4f} "
                    f"end={metrics['loss_end']:.4f} "
                    f"weak={metrics['loss_weak']:.4f} "
                    f"count={metrics['loss_count']:.4f} "
                    f"reg={metrics['loss_reg']:.4f} | "
                    f"lr={cur_lr:.2e} t={elapsed:.1f}s"
                )

            if epoch % tc.checkpoint_every == 0:
                self._save_checkpoint(absolute_epoch, f"epoch{absolute_epoch:04d}", ema=ema)

            # Early stopping
            if self._patience_counter >= tc.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Final EMA checkpoint
        if ema is not None:
            self._save_ema_checkpoint(start_epoch + epochs - 1, ema)

        # Save history
        df = self.history.to_dataframe()
        df.to_csv(self.output_dir / "training_history.csv", index=False)
        return self.history
