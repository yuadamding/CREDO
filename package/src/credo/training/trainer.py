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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config.schema import RunConfig
from ..data.core import EndpointProblem, FiniteMeasure
from ..losses.endpoint import EndpointGeometryMassLoss
from ..losses.weak_form import WeakFormLoss
from ..losses.counts import CountLikelihood
from ..losses.causal_attention import (
    context_smoothness_loss,
    control_edge_null_loss,
    edge_sparsity_loss,
    guide_concordance_loss,
    mediator_orthogonality_loss,
)
from ..losses.regularizers import RolloutRegularizer
from ..models.context import GroupStatistics
from ..models.full_model import FullDynamicsModel
from ..models.weighted_sde import WeightedParticleSimulator
from ..models.simulator import _stable_seed_offset, initialise_particles
from .manifest import append_run_manifest_record, build_run_manifest, write_run_manifest


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
    loss_causal: List[float] = field(default_factory=list)
    loss_extra: List[float] = field(default_factory=list)
    context_norm: List[float] = field(default_factory=list)
    q_entropy: List[float] = field(default_factory=list)
    freq_entropy: List[float] = field(default_factory=list)
    within_attention_entropy: List[float] = field(default_factory=list)
    group_attention_entropy: List[float] = field(default_factory=list)
    within_effective_keys: List[float] = field(default_factory=list)
    group_effective_keys: List[float] = field(default_factory=list)
    mass_log_range: List[float] = field(default_factory=list)
    state_to_mediator_effective_keys: List[float] = field(default_factory=list)
    local_to_global_mediator_effective_keys: List[float] = field(default_factory=list)
    mediator_to_group_effective_keys: List[float] = field(default_factory=list)
    edge_sparsity: List[float] = field(default_factory=list)
    effective_edge_mean: List[float] = field(default_factory=list)
    baseline_edge_mean: List[float] = field(default_factory=list)
    residual_edge_sparsity_loss: List[float] = field(default_factory=list)
    edge_entropy: List[float] = field(default_factory=list)
    control_edge_norm: List[float] = field(default_factory=list)
    mediator_orthogonality: List[float] = field(default_factory=list)
    residual_edge_abs_mean: List[float] = field(default_factory=list)
    residual_edge_signed_mean: List[float] = field(default_factory=list)
    mediator_usage_entropy: List[float] = field(default_factory=list)
    mediator_usage_min: List[float] = field(default_factory=list)
    mediator_usage_max: List[float] = field(default_factory=list)
    terminal_ess_frac_mean: List[float] = field(default_factory=list)
    terminal_ess_frac_min: List[float] = field(default_factory=list)
    min_ess_frac_mean: List[float] = field(default_factory=list)
    max_weight_frac_mean: List[float] = field(default_factory=list)
    logw_range_max: List[float] = field(default_factory=list)
    ess_gate_status: List[str] = field(default_factory=list)

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
            "loss_causal": self.loss_causal,
            "loss_extra": self.loss_extra,
            "context_norm": self.context_norm,
            "q_entropy": self.q_entropy,
            "freq_entropy": self.freq_entropy,
            "within_attention_entropy": self.within_attention_entropy,
            "group_attention_entropy": self.group_attention_entropy,
            "within_effective_keys": self.within_effective_keys,
            "group_effective_keys": self.group_effective_keys,
            "mass_log_range": self.mass_log_range,
            "state_to_mediator_effective_keys": self.state_to_mediator_effective_keys,
            "local_to_global_mediator_effective_keys": self.local_to_global_mediator_effective_keys,
            "mediator_to_group_effective_keys": self.mediator_to_group_effective_keys,
            "edge_sparsity": self.edge_sparsity,
            "effective_edge_mean": self.effective_edge_mean,
            "baseline_edge_mean": self.baseline_edge_mean,
            "residual_edge_sparsity_loss": self.residual_edge_sparsity_loss,
            "edge_entropy": self.edge_entropy,
            "control_edge_norm": self.control_edge_norm,
            "mediator_orthogonality": self.mediator_orthogonality,
            "residual_edge_abs_mean": self.residual_edge_abs_mean,
            "residual_edge_signed_mean": self.residual_edge_signed_mean,
            "mediator_usage_entropy": self.mediator_usage_entropy,
            "mediator_usage_min": self.mediator_usage_min,
            "mediator_usage_max": self.mediator_usage_max,
            "terminal_ess_frac_mean": self.terminal_ess_frac_mean,
            "terminal_ess_frac_min": self.terminal_ess_frac_min,
            "min_ess_frac_mean": self.min_ess_frac_mean,
            "max_weight_frac_mean": self.max_weight_frac_mean,
            "logw_range_max": self.logw_range_max,
            "ess_gate_status": self.ess_gate_status,
        })


_DIAGNOSTIC_KEYS = (
    "context_norm",
    "q_entropy",
    "freq_entropy",
    "within_attention_entropy",
    "group_attention_entropy",
    "within_effective_keys",
    "group_effective_keys",
    "mass_log_range",
    "state_to_mediator_effective_keys",
    "local_to_global_mediator_effective_keys",
    "mediator_to_group_effective_keys",
    "edge_sparsity",
    "effective_edge_mean",
    "baseline_edge_mean",
    "residual_edge_sparsity_loss",
    "edge_entropy",
    "control_edge_norm",
    "mediator_orthogonality",
    "residual_edge_abs_mean",
    "residual_edge_signed_mean",
    "mediator_usage_entropy",
    "mediator_usage_min",
    "mediator_usage_max",
)


_GLOBAL_CONTEXT_BACKENDS = {"transformer", "causal_attention"}


def _uses_global_context_backend(model: FullDynamicsModel) -> bool:
    return getattr(model, "context_kind", "mlp") in _GLOBAL_CONTEXT_BACKENDS


def _uses_global_ecological_context(model: FullDynamicsModel) -> bool:
    """Return true when sharding perturbations would alter ecology semantics."""
    if _uses_global_context_backend(model):
        return True
    coeff_nets = getattr(model, "coeff_nets", None)
    return bool(getattr(coeff_nets, "ecological_growth", False))


def _nan_diagnostics() -> Dict[str, float]:
    return {key: math.nan for key in _DIAGNOSTIC_KEYS}


def _diagnostics_from_rollout(rollout) -> Dict[str, float]:
    metrics = _nan_diagnostics()
    diagnostics = getattr(rollout, "context_diagnostics", None)
    if not diagnostics:
        metrics.update(_weight_diagnostics_from_rollout(rollout))
        return metrics
    for key in _DIAGNOSTIC_KEYS:
        value = diagnostics.get(key)
        if value is not None and value.numel() > 0:
            metrics[key] = float(value.float().mean().item())
    metrics.update(_weight_diagnostics_from_rollout(rollout))
    return metrics


def _weight_diagnostics_from_rollout(rollout) -> Dict[str, float]:
    ess_frac = getattr(rollout, "ess_frac_steps", None)
    max_weight_frac = getattr(rollout, "max_weight_frac_steps", None)
    logw_range = getattr(rollout, "logw_range_steps", None)
    return _weight_diagnostics_from_tensors(
        ess_frac_steps=ess_frac,
        max_weight_frac_steps=max_weight_frac,
        logw_range_steps=logw_range,
    )


def _weight_diagnostics_from_tensors(
    *,
    ess_frac_steps: Optional[torch.Tensor],
    max_weight_frac_steps: Optional[torch.Tensor],
    logw_range_steps: Optional[torch.Tensor],
) -> Dict[str, float]:
    metrics = {
        "terminal_ess_frac_mean": math.nan,
        "terminal_ess_frac_min": math.nan,
        "min_ess_frac_mean": math.nan,
        "max_weight_frac_mean": math.nan,
        "logw_range_max": math.nan,
    }
    if ess_frac_steps is not None and ess_frac_steps.numel() > 0:
        ess = ess_frac_steps.float()
        terminal = ess[-1]
        metrics["terminal_ess_frac_mean"] = float(terminal.mean().item())
        metrics["terminal_ess_frac_min"] = float(terminal.min().item())
        metrics["min_ess_frac_mean"] = float(ess.min(dim=0).values.mean().item())
    if max_weight_frac_steps is not None and max_weight_frac_steps.numel() > 0:
        metrics["max_weight_frac_mean"] = float(max_weight_frac_steps.float().max(dim=0).values.mean().item())
    if logw_range_steps is not None and logw_range_steps.numel() > 0:
        metrics["logw_range_max"] = float(logw_range_steps.float().max().item())
    return metrics


def _ess_gate_status(metrics: Dict[str, float], training_config) -> str:
    terminal_min = metrics.get("terminal_ess_frac_min", math.nan)
    max_weight = metrics.get("max_weight_frac_mean", math.nan)
    if not math.isfinite(terminal_min) or not math.isfinite(max_weight):
        return "not_available"
    if terminal_min < training_config.ess_fail_frac or max_weight > training_config.ess_max_weight_frac_fail:
        return "fail"
    if terminal_min < training_config.ess_claim_grade_min_frac:
        return "claim_grade_blocked"
    if terminal_min < training_config.ess_warn_frac:
        return "warn"
    return "pass"


def _diagnostics_from_lists(values_by_key: Dict[str, list[float]]) -> Dict[str, float]:
    metrics = _nan_diagnostics()
    for key in _DIAGNOSTIC_KEYS:
        values = values_by_key.get(key, [])
        if values:
            metrics[key] = float(np.mean(values))
    return metrics


def _append_context_diagnostics(values_by_key: Dict[str, list[float]], diagnostics) -> None:
    if diagnostics is None:
        return
    for key in _DIAGNOSTIC_KEYS:
        value = getattr(diagnostics, key, None)
        if value is not None:
            values_by_key.setdefault(key, []).append(float(value.float().mean().item()))


def _build_target_dicts(
    endpoint: EndpointProblem,
    perturbation_ids: List[str],
    device: str,
    dtype: torch.dtype,
) -> Tuple[Dict, Dict]:
    """Build target support and log-weight dicts for endpoint finite-measure loss."""
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
        particle_sampling: str = "uniform",
        context_override_provider: Optional[Callable[..., Any]] = None,
        extra_loss_callback: Optional[Callable[..., Tuple[torch.Tensor, Dict[str, float]]]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.endpoint = endpoint
        self.supported_pids = supported_pids
        self.count_data = count_data
        self.particle_sampling = particle_sampling
        self.context_override_provider = context_override_provider
        self.extra_loss_callback = extra_loss_callback
        self.training_devices = config.resolve_training_devices()
        self.device = self.training_devices[0]
        self.dtype = torch.float32
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs
        self._model_replicas: Dict[str, FullDynamicsModel] = {}
        self._weak_loss_replicas: Dict[str, WeakFormLoss] = {}
        self._target_cache: Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = {}
        self._count_tensor_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        embedding_map = endpoint.metadata.get("measure_to_embedding")
        if not isinstance(embedding_map, dict):
            embedding_map = endpoint.metadata.get("embedding_ids")
        self._embedding_ids_by_pid: Dict[str, str] = (
            {str(pid): str(embed_id) for pid, embed_id in embedding_map.items()}
            if isinstance(embedding_map, dict)
            else {}
        )

        tc = config.training
        sc = config.simulation
        # Always store trajectory history when any regularization is active
        has_trajectory_reg = (
            tc.lambda_reg_net > 0 or tc.lambda_reg_diffusion > 0
        )
        self._needs_rollout_history = (
            (tc.lambda_weak > 0) or (tc.lambda_count > 0) or has_trajectory_reg
            or _uses_global_context_backend(model)
            or (extra_loss_callback is not None)
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
            store_history=self._needs_rollout_history,
        )
        self.uot_loss = EndpointGeometryMassLoss(
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
            lambda_diffusion=tc.lambda_reg_diffusion,
            lambda_drift=tc.lambda_reg_net,
            lambda_growth=tc.lambda_reg_net,
        )

        self._target_support, self._target_logw = self._get_target_dicts_for_device(self.device)

        self.model.to(self.device)
        self.weak_loss.to(self.device)
        self.count_lik.to(self.device)

        self.history = TrainingHistory()
        self._best_loss = math.inf
        self._best_checkpoint_path: Optional[Path] = None
        self._patience_counter = 0
        self._divergence_counter = 0

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

    def _embedding_ids_for_pids(self, perturbation_ids: List[str]) -> List[str]:
        """Return model embedding IDs for endpoint measure keys."""
        if not self._embedding_ids_by_pid:
            return list(perturbation_ids)
        return [self._embedding_ids_by_pid.get(str(pid), str(pid)) for pid in perturbation_ids]

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

    def _get_target_dicts_for_device(
        self,
        device: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        cached = self._target_cache.get(device)
        if cached is None:
            cached = _build_target_dicts(
                self.endpoint,
                self.supported_pids,
                device,
                self.dtype,
            )
            self._target_cache[device] = cached
        return cached

    def _subset_target_dicts(
        self,
        device: str,
        perturbation_ids: List[str],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        target_support, target_logw = self._get_target_dicts_for_device(device)
        local_support = {pid: target_support[pid] for pid in perturbation_ids if pid in target_support}
        local_logw = {pid: target_logw[pid] for pid in perturbation_ids if pid in target_logw}
        return local_support, local_logw

    def _get_count_tensors_for_device(self, device: str) -> Optional[Dict[str, torch.Tensor]]:
        if self.count_data is None:
            return None
        cached = self._count_tensor_cache.get(device)
        if cached is None:
            cached = {
                "exposures": torch.as_tensor(self.count_data["exposures"], dtype=self.dtype, device=device),
                "counts": torch.as_tensor(self.count_data["counts"], dtype=self.dtype, device=device),
                "n_totals": torch.as_tensor(self.count_data["n_totals"], dtype=self.dtype, device=device),
            }
            self._count_tensor_cache[device] = cached
        return cached

    def _make_noise_generator(self, device: str, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
        generator.manual_seed(int(seed))
        return generator

    @staticmethod
    def _pid_seed(base_seed: int, pid: str, *, salt: int = 0) -> int:
        return int(base_seed) + int(salt) + _stable_seed_offset(pid)

    def _initialise_particles_stable_by_pid(
        self,
        perturbation_ids: List[str],
        *,
        n_particles: int,
        dtype: torch.dtype,
        base_seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_parts: list[torch.Tensor] = []
        logw_parts: list[torch.Tensor] = []
        log_m0_parts: list[torch.Tensor] = []
        for pid in perturbation_ids:
            z_pid, logw_pid, log_m0_pid = initialise_particles(
                self.endpoint,
                [pid],
                n_particles=n_particles,
                device=self.device,
                dtype=dtype,
                seed=self._pid_seed(base_seed, pid),
                sampling=self.particle_sampling,
            )
            z_parts.append(z_pid)
            logw_parts.append(logw_pid)
            log_m0_parts.append(log_m0_pid)
        return (
            torch.cat(z_parts, dim=0),
            torch.cat(logw_parts, dim=0),
            torch.cat(log_m0_parts, dim=0),
        )

    def _sample_chunk_noise_stable_by_pid(
        self,
        perturbation_ids: List[str],
        like: torch.Tensor,
        *,
        base_seed: int,
        step_idx: int,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        shape_tail = tuple(like.shape[1:])
        for pid in perturbation_ids:
            generator = self._make_noise_generator(
                self.device,
                self._pid_seed(base_seed, pid, salt=10_000 + 1_000_003 * int(step_idx)),
            )
            parts.append(
                torch.randn(
                    (1,) + shape_tail,
                    device=like.device,
                    dtype=like.dtype,
                    generator=generator,
                )
            )
        return torch.cat(parts, dim=0)

    def _uses_causal_attention_loss(self) -> bool:
        if getattr(self.model, "context_kind", "mlp") != "causal_attention":
            return False
        tc = self.config.training
        return any(
            weight > 0
            for weight in (
                tc.lambda_causal_ctrl_edge,
                tc.lambda_causal_guide,
                tc.lambda_causal_sparse,
                tc.lambda_causal_orth,
                tc.lambda_causal_ctx_smooth,
            )
        )

    def _control_mask_for_pids(self, perturbation_ids: List[str], device: torch.device | str) -> torch.Tensor:
        control_ids = set(getattr(self.model, "control_ids", set()))
        control_measure_keys = set(self.endpoint.metadata.get("control_measure_keys", []))
        embedding_ids = self._embedding_ids_for_pids(perturbation_ids)
        return torch.tensor(
            [
                pid in control_measure_keys or embed_pid in control_ids
                for pid, embed_pid in zip(perturbation_ids, embedding_ids)
            ],
            device=device,
            dtype=torch.bool,
        )

    def _target_ids_for_guides(self, perturbation_ids: List[str]) -> Tuple[List[str], bool]:
        if self.count_data is not None:
            mapping = self.count_data.get("target_ids_by_pid")
            if isinstance(mapping, dict):
                missing = [pid for pid in perturbation_ids if pid not in mapping]
                if missing:
                    raise ValueError(
                        "lambda_causal_guide > 0 requires target_ids_by_pid entries "
                        f"for all perturbations; missing {missing[:5]}."
                    )
                return [str(mapping[pid]) for pid in perturbation_ids], True
            target_ids = self.count_data.get("target_ids")
            if target_ids is not None and len(target_ids) == len(self.supported_pids):
                target_by_pid = {
                    pid: str(target)
                    for pid, target in zip(self.supported_pids, target_ids)
                }
                return [target_by_pid.get(pid, pid) for pid in perturbation_ids], True
        endpoint_targets = self.endpoint.metadata.get("target_ids")
        if isinstance(endpoint_targets, dict):
            return [str(endpoint_targets.get(pid, pid)) for pid in perturbation_ids], True
        return list(perturbation_ids), False

    def _causal_loss_scale(self, epoch: Optional[int]) -> float:
        if epoch is None:
            return 1.0
        tc = self.config.training
        if epoch < tc.causal_loss_start_epoch:
            return 0.0
        return min(
            1.0,
            float(epoch - tc.causal_loss_start_epoch + 1) / float(tc.causal_loss_ramp_epochs),
        )

    def _validate_count_order(self, perturbation_ids: List[str]) -> None:
        if self.count_data is None or "perturbation_ids" not in self.count_data:
            return
        count_pids = [str(pid) for pid in self.count_data["perturbation_ids"]]
        if list(perturbation_ids) != count_pids:
            raise ValueError(
                "Count loss perturbation order mismatch: "
                f"rollout={list(perturbation_ids)[:5]}..., count_data={count_pids[:5]}..."
            )

    def _causal_attention_loss_from_tensors(
        self,
        *,
        edge_scores_steps: Optional[torch.Tensor],
        residual_edge_scores_steps: Optional[torch.Tensor],
        residual_edge_magnitude_steps: Optional[torch.Tensor],
        mediator_tokens_steps: Optional[torch.Tensor],
        growth_context_steps: Optional[torch.Tensor],
        tau_steps: Optional[torch.Tensor],
        perturbation_ids: List[str],
        epoch: Optional[int] = None,
        causal_delta_steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if not self._uses_causal_attention_loss():
            return loss
        tc = self.config.training
        target_ids: Optional[List[str]] = None
        if tc.lambda_causal_guide > 0:
            target_ids, explicit_targets = self._target_ids_for_guides(perturbation_ids)
            if not explicit_targets:
                raise ValueError(
                    "lambda_causal_guide > 0 requires target_ids_by_pid or target_ids in count_data. "
                    "Guide concordance would otherwise be a no-op."
                )
            if all(str(target) == str(pid) for target, pid in zip(target_ids, perturbation_ids)):
                raise ValueError(
                    "lambda_causal_guide > 0 requires a non-identity target map. "
                    "Guide concordance would otherwise be a no-op."
                )
        scale = self._causal_loss_scale(epoch)
        if scale <= 0:
            return loss

        residual_edges = (
            None
            if residual_edge_scores_steps is None
            else residual_edge_scores_steps.float().mean(dim=0)
        )
        residual_edges_for_control = (
            None
            if residual_edge_scores_steps is None
            else residual_edge_scores_steps.float().square().mean(dim=0).sqrt()
        )
        residual_edge_magnitude = (
            None if residual_edges is None else residual_edges.abs()
            if residual_edge_magnitude_steps is None
            else residual_edge_magnitude_steps.float().mean(dim=0)
        )

        if residual_edges_for_control is not None and tc.lambda_causal_ctrl_edge > 0:
            control_mask = self._control_mask_for_pids(perturbation_ids, residual_edges_for_control.device)
            loss = loss + tc.lambda_causal_ctrl_edge * control_edge_null_loss(
                residual_edges_for_control,
                control_mask,
            )
        if residual_edges is not None and tc.lambda_causal_guide > 0:
            loss = loss + tc.lambda_causal_guide * guide_concordance_loss(
                residual_edges,
                target_ids or list(perturbation_ids),
            )
        if residual_edge_magnitude is not None and tc.lambda_causal_sparse > 0:
            loss = loss + tc.lambda_causal_sparse * edge_sparsity_loss(residual_edge_magnitude)
        if mediator_tokens_steps is not None and tc.lambda_causal_orth > 0:
            loss = loss + tc.lambda_causal_orth * mediator_orthogonality_loss(
                mediator_tokens_steps[-1].float()
            )
        if (
            (causal_delta_steps is not None or growth_context_steps is not None)
            and tau_steps is not None
            and tc.lambda_causal_ctx_smooth > 0
        ):
            if (
                getattr(self.model, "context_kind", "mlp") == "causal_attention"
                and causal_delta_steps is None
            ):
                raise ValueError(
                    "CEA context smoothness requires causal_delta_steps. "
                    "Do not fall back to full growth_context_steps for CEA."
                )
            smooth_context_steps = causal_delta_steps if causal_delta_steps is not None else growth_context_steps
            loss = loss + tc.lambda_causal_ctx_smooth * context_smoothness_loss(
                smooth_context_steps.float(),
                tau_steps.float(),
            )
        return loss * scale

    def _causal_attention_loss_from_rollout(
        self,
        rollout,
        perturbation_ids: List[str],
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        return self._causal_attention_loss_from_tensors(
            edge_scores_steps=getattr(rollout, "causal_edge_scores_steps", None),
            residual_edge_scores_steps=getattr(rollout, "causal_residual_edge_scores_steps", None),
            residual_edge_magnitude_steps=getattr(rollout, "causal_residual_edge_magnitude_steps", None),
            mediator_tokens_steps=getattr(rollout, "causal_mediator_tokens_steps", None),
            growth_context_steps=getattr(rollout, "causal_growth_context_steps", None),
            causal_delta_steps=getattr(rollout, "causal_delta_steps", None),
            tau_steps=rollout.tau_steps,
            perturbation_ids=perturbation_ids,
            epoch=epoch,
        )

    @staticmethod
    def _weighted_shard_loss(loss_value: torch.Tensor, local_groups: int, total_groups: int) -> torch.Tensor:
        if total_groups <= 0:
            return loss_value
        return loss_value * (float(local_groups) / float(total_groups))

    def _build_optimizer(self, stage: str) -> torch.optim.Optimizer:
        tc = self.config.training
        def _no_decay(name: str) -> bool:
            return name.endswith(".bias") or "norm" in name.lower() or "layernorm" in name.lower()

        grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
            ("net", False): [],
            ("net", True): [],
            ("embed", False): [],
            ("embed", True): [],
            ("transformer", False): [],
            ("transformer", True): [],
            ("causal", False): [],
            ("causal", True): [],
        }

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if getattr(self.model, "context_kind", "mlp") == "causal_attention" and name.startswith("context_agg."):
                group = "causal"
            elif getattr(self.model, "context_kind", "mlp") == "transformer" and name.startswith("context_agg."):
                group = "transformer"
            elif "embedding" in name:
                group = "embed"
            else:
                group = "net"
            grouped[(group, _no_decay(name))].append(p)

        count_params = [p for p in self.count_lik.parameters() if p.requires_grad]
        grouped[("net", False)].extend(count_params)

        param_groups = []
        specs = {
            "net": (tc.lr_net, tc.weight_decay),
            "embed": (tc.lr_embed, tc.weight_decay),
            "transformer": (tc.lr_transformer, tc.transformer_weight_decay),
            "causal": (tc.lr_causal_attention, tc.causal_attention_weight_decay),
        }
        for group, (lr, decay) in specs.items():
            decay_params = grouped[(group, False)]
            no_decay_params = grouped[(group, True)]
            if decay_params:
                param_groups.append({"params": decay_params, "lr": lr, "weight_decay": decay})
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
        optimizer_cls = torch.optim.AdamW if tc.optimizer == "adamw" else torch.optim.Adam
        return optimizer_cls(param_groups, weight_decay=0.0)

    def _one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        stage: str,
        perturbation_ids: List[str],
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        if self._can_use_multi_gpu() and _uses_global_ecological_context(self.model):
            raise ValueError(
                "Global ecological context is not supported with multi-GPU sharding yet. "
                "Use a single training device so context is computed from the full perturbation set."
            )
        if self._can_use_multi_gpu() and (
            self.context_override_provider is not None or self.extra_loss_callback is not None
        ):
            raise ValueError(
                "Custom context overrides or extra rollout losses are not supported with "
                "the current multi-GPU single-model path."
            )
        if not self._can_use_multi_gpu():
            batch_size = self._perturbation_batch_size(perturbation_ids)
            if batch_size < len(perturbation_ids):
                if self.context_override_provider is not None or self.extra_loss_callback is not None:
                    raise ValueError(
                        "Custom context overrides or extra rollout losses require full perturbation "
                        "rollout in the current trainer. Set max_active_perturbations=0."
                    )
                if _uses_global_ecological_context(self.model):
                    batching_mode = getattr(
                        self.config.training,
                        "global_context_batching",
                        "full_context_cache",
                    )
                    if batching_mode == "error":
                        raise ValueError(
                            "Global ecological context cannot be computed from "
                            "chunk-local perturbation batches. Set "
                            "global_context_batching='full_context_cache' to use "
                            "the exact full-context chunked trainer."
                        )
                    if batching_mode != "full_context_cache":
                        raise ValueError(
                            "global_context_batching must be 'full_context_cache' "
                            "or 'error' for the current trainer."
                        )
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
            seed=self.config.training.seed + seed_offset + epoch,
            sampling=self.particle_sampling,
        )

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.compute_dtype)
            if self.autocast_enabled
            else nullcontext()
        )

        with autocast_ctx:
            context_override = None
            if self.context_override_provider is not None:
                context_override = self.context_override_provider(
                    trainer=self,
                    model=self.model,
                    endpoint=self.endpoint,
                    perturbation_ids=perturbation_ids,
                    z0=z0,
                    logw0=logw0,
                    log_m0=log_m0,
                    epoch=epoch,
                    stage=stage,
                )
            rollout = self.simulator.rollout(
                z0=z0,
                logw0=logw0,
                model=self.model,
                log_m0=log_m0,
                perturbation_ids=perturbation_ids,
                embedding_ids=self._embedding_ids_for_pids(perturbation_ids),
                context_override=context_override,
            )

        # --- Endpoint geometry-plus-log-mass loss (absolute log-weights) ---
        # Keep geometry and mass terms in fp32 for numerical stability even when rollout used AMP.
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
            self._validate_count_order(perturbation_ids)
            cd = self._get_count_tensors_for_device(self.device)
            loss_count = self.count_lik(
                growth_steps=rollout.growth_steps,
                logw_steps=rollout.logw_steps,
                tau_steps=rollout.tau_steps,
                exposures=cd["exposures"],
                count_matrix=cd["counts"],
                n_totals=cd["n_totals"],
            )

        # --- Regularization ---
        loss_reg = self.regularizer(
            drift_steps=rollout.drift_steps.float() if rollout.drift_steps is not None else None,
            sigma_steps=rollout.sigma_steps.float() if rollout.sigma_steps is not None else None,
            growth_steps=rollout.growth_steps.float() if rollout.growth_steps is not None else None,
        )
        loss_reg = loss_reg + self.model.regularization(lambda_embed=tc.lambda_reg_embed)
        loss_reg = loss_reg + self.model.growth_bias_regularization(
            lambda_growth_bias=tc.lambda_reg_growth_bias
        )
        loss_causal = self._causal_attention_loss_from_rollout(rollout, perturbation_ids, epoch=epoch)
        loss_extra = torch.tensor(0.0, device=self.device)
        extra_metrics: Dict[str, float] = {}
        if self.extra_loss_callback is not None:
            loss_extra, extra_metrics = self.extra_loss_callback(
                trainer=self,
                rollout=rollout,
                perturbation_ids=perturbation_ids,
                epoch=epoch,
                stage=stage,
            )

        # --- Total ---
        loss = (
            tc.lambda_end * loss_end
            + tc.lambda_weak * loss_weak
            + tc.lambda_count * loss_count
            + loss_reg
            + loss_causal
            + loss_extra
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

        metrics = {
            "n_active_perturbations": len(perturbation_ids),
            "perturbation_batch_size": len(perturbation_ids),
            "loss_total": float(loss.item()),
            "loss_end": float(loss_end.item()),
            "loss_weak": float(loss_weak.item()),
            "loss_count": float(loss_count.item()),
            "loss_reg": float(loss_reg.item()),
            "loss_causal": float(loss_causal.item()),
            "loss_extra": float(loss_extra.item()),
            **_diagnostics_from_rollout(rollout),
        }
        metrics.update(extra_metrics)
        metrics["ess_gate_status"] = _ess_gate_status(metrics, tc)
        return metrics

    def _one_epoch_chunked(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        stage: str,
        perturbation_ids: List[str],
        seed_offset: int = 0,
    ) -> Dict[str, float]:
        if self.context_override_provider is not None or self.extra_loss_callback is not None:
            raise ValueError(
                "Custom context overrides or extra rollout losses are not implemented in "
                "the chunked trainer path. Set max_active_perturbations=0 or implement "
                "chunk-aware single-time hooks."
            )
        tc = self.config.training
        sc = self.config.simulation

        self.model.train()
        torch.manual_seed(self.config.training.seed + seed_offset + epoch)

        rollout_dtype = self.compute_dtype if self.autocast_enabled else self.dtype
        perturbation_chunks = self._chunk_perturbation_ids(perturbation_ids)
        total_groups = len(perturbation_ids)
        batch_size = max(len(chunk) for chunk in perturbation_chunks)
        store_history = self._needs_rollout_history

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

        chunk_states: list[dict] = []
        base_seed = self.config.training.seed + seed_offset + epoch * 1000
        for chunk_pids in perturbation_chunks:
            z0, logw0, log_m0 = self._initialise_particles_stable_by_pid(
                chunk_pids,
                n_particles=sc.n_particles,
                dtype=rollout_dtype,
                base_seed=base_seed,
            )
            state = {
                "pids": chunk_pids,
                "z": z0,
                "logw": logw0,
                "log_m0": log_m0,
                "ess_frac_list": [WeightedParticleSimulator.ess_fraction(logw0).detach()],
                "max_weight_frac_list": [WeightedParticleSimulator.max_weight_fraction(logw0).detach()],
                "logw_range_list": [WeightedParticleSimulator.log_weight_range(logw0).detach()],
            }
            if store_history:
                state.update(
                    {
                        "z_list": [z0],
                        "logw_list": [logw0],
                        "drift_list": [],
                        "sigma_list": [],
                        "growth_list": [],
                    }
                )
            chunk_states.append(state)

        tau_steps = torch.linspace(
            0.0,
            1.0,
            sc.n_steps + 1,
            device=self.device,
            dtype=rollout_dtype,
        )
        dtau = 1.0 / sc.n_steps
        diagnostic_values: dict[str, list[float]] = {}
        causal_edge_scores_list: list[torch.Tensor] = []
        causal_residual_edge_scores_list: list[torch.Tensor] = []
        causal_residual_edge_magnitude_list: list[torch.Tensor] = []
        causal_mediator_tokens_list: list[torch.Tensor] = []
        causal_growth_context_list: list[torch.Tensor] = []
        causal_delta_list: list[torch.Tensor] = []

        for step_idx in range(sc.n_steps):
            tau_k = tau_steps[step_idx]

            if _uses_global_context_backend(self.model):
                pids_all: list[str] = []
                group_slices: list[slice] = []
                start = 0
                for state in chunk_states:
                    pids_all.extend(state["pids"])
                    stop = start + len(state["pids"])
                    group_slices.append(slice(start, stop))
                    start = stop
                embedding_ids_all = self._embedding_ids_for_pids(pids_all)

                with autocast_ctx:
                    z_all = torch.cat([state["z"] for state in chunk_states], dim=0)
                    logw_all = torch.cat([state["logw"] for state in chunk_states], dim=0)
                    log_m0_all = torch.cat([state["log_m0"] for state in chunk_states], dim=0)
                    a_all = self.model.embedding(embedding_ids_all)
                    residual_all = self.model.embedding.residuals(embedding_ids_all)
                    b_all = self.model.embedding.growth_intercepts(embedding_ids_all)
                    context_kind = getattr(self.model, "context_kind", "mlp")
                    if context_kind == "causal_attention":
                        ctx_state = self.model.context_agg(
                            z_all,
                            logw_all,
                            a_all,
                            log_m0_all,
                            tau=tau_k,
                            residual=residual_all,
                        )
                    else:
                        ctx_state = self.model.context_agg(
                            z_all,
                            logw_all,
                            a_all,
                            log_m0_all,
                            tau=tau_k,
                        )
                    _append_context_diagnostics(diagnostic_values, ctx_state.diagnostics)
                    if context_kind == "causal_attention":
                        edge_scores = getattr(ctx_state, "edge_scores_gm", None)
                        if edge_scores is not None:
                            causal_edge_scores_list.append(edge_scores)
                        residual_edge_scores = getattr(ctx_state, "residual_edge_scores_gm", None)
                        if residual_edge_scores is not None:
                            causal_residual_edge_scores_list.append(residual_edge_scores)
                        residual_edge_magnitude = getattr(ctx_state, "residual_edge_magnitude_gm", None)
                        if residual_edge_magnitude is not None:
                            causal_residual_edge_magnitude_list.append(residual_edge_magnitude)
                        mediator_tokens = getattr(ctx_state, "mediator_tokens", None)
                        if mediator_tokens is not None:
                            causal_mediator_tokens_list.append(mediator_tokens)
                        ctx_growth_context = getattr(ctx_state, "growth_context", None)
                        if ctx_growth_context is not None:
                            causal_growth_context_list.append(ctx_growth_context)
                            if ctx_growth_context.ndim == 2 and ctx_state.context.ndim == 1:
                                causal_delta_list.append(ctx_growth_context - ctx_state.context[None, :])
                            else:
                                causal_delta_list.append(ctx_growth_context - ctx_state.context)
                    eta_all, _ = self.model.context_agg.encode_particles(z_all)
                    base_context = ctx_state.context
                    growth_context = getattr(ctx_state, "growth_context", None)
                    if (
                        context_kind == "causal_attention"
                        and not getattr(self.model, "causal_growth_only", True)
                        and growth_context is not None
                    ):
                        base_context = growth_context
                    if (
                        getattr(self.model, "transformer_growth_only", False)
                        and getattr(self.model, "meanfield_context_agg", None) is not None
                    ):
                        base_state = self.model.meanfield_context_agg(
                            z_all,
                            logw_all,
                            a_all,
                            log_m0_all,
                            tau=tau_k,
                        )
                        base_context = base_state.context
                        if growth_context is None:
                            growth_context = ctx_state.context

                for state, group_slice in zip(chunk_states, group_slices):
                    base_context_local = base_context
                    if (
                        base_context.ndim == 2
                        and base_context.shape[0] == len(pids_all)
                    ):
                        base_context_local = base_context[group_slice]
                    growth_context_local = growth_context
                    if (
                        growth_context is not None
                        and growth_context.ndim == 2
                        and growth_context.shape[0] == len(pids_all)
                    ):
                        growth_context_local = growth_context[group_slice]
                    with autocast_ctx:
                        coeffs = self.model.coeff_nets(
                            z=state["z"],
                            tau=tau_k,
                            context=base_context_local,
                            a=a_all[group_slice],
                            growth_intercept=b_all[group_slice],
                            eta_z=eta_all[group_slice],
                            q=ctx_state.q,
                            s=ctx_state.s,
                            growth_context=growth_context_local,
                        )
                    noise = self._sample_chunk_noise_stable_by_pid(
                        state["pids"],
                        state["z"],
                        base_seed=base_seed,
                        step_idx=step_idx,
                    )
                    z_next = state["z"] + coeffs.drift * dtau + coeffs.sigma_diag * (dtau ** 0.5) * noise
                    logw_next = state["logw"] + coeffs.growth * dtau
                    if store_history:
                        state["drift_list"].append(coeffs.drift)
                        state["sigma_list"].append(coeffs.sigma_diag)
                        state["growth_list"].append(coeffs.growth)
                        state["z_list"].append(z_next)
                        state["logw_list"].append(logw_next)
                    state["z"] = z_next
                    state["logw"] = logw_next
                    state["ess_frac_list"].append(WeightedParticleSimulator.ess_fraction(logw_next).detach())
                    state["max_weight_frac_list"].append(
                        WeightedParticleSimulator.max_weight_fraction(logw_next).detach()
                    )
                    state["logw_range_list"].append(WeightedParticleSimulator.log_weight_range(logw_next).detach())
                continue

            summary_parts: list[GroupStatistics] = []
            local_cache: list[dict[str, torch.Tensor]] = []

            for state in chunk_states:
                local_embedding_ids = self._embedding_ids_for_pids(state["pids"])
                with autocast_ctx:
                    a_local = self.model.embedding(local_embedding_ids)
                    b_local = self.model.embedding.growth_intercepts(local_embedding_ids)
                    eta_local, phi_local = self.model.context_agg.encode_particles(state["z"])
                    stats_local, _, _ = self.model.context_agg.summarize_groups(
                        state["z"],
                        state["logw"],
                        state["log_m0"],
                        eta=eta_local,
                        phi=phi_local,
                    )
                summary_parts.append(stats_local)
                local_cache.append({"a": a_local, "b": b_local, "eta": eta_local})

            global_stats = GroupStatistics(
                log_n_g=torch.cat([stats.log_n_g for stats in summary_parts], dim=0),
                eta_g=torch.cat([stats.eta_g for stats in summary_parts], dim=0),
                phi_g=torch.cat([stats.phi_g for stats in summary_parts], dim=0),
            )
            global_context = self.model.context_agg.context_from_group_statistics(global_stats)

            for state, cache_entry in zip(chunk_states, local_cache):
                with autocast_ctx:
                    coeffs = self.model.coeff_nets(
                        z=state["z"],
                        tau=tau_k,
                        context=global_context.context,
                        a=cache_entry["a"],
                        growth_intercept=cache_entry["b"],
                        eta_z=cache_entry["eta"],
                        q=global_context.q,
                        s=global_context.s,
                    )
                noise = self._sample_chunk_noise_stable_by_pid(
                    state["pids"],
                    state["z"],
                    base_seed=base_seed,
                    step_idx=step_idx,
                )
                z_next = state["z"] + coeffs.drift * dtau + coeffs.sigma_diag * (dtau ** 0.5) * noise
                logw_next = state["logw"] + coeffs.growth * dtau
                if store_history:
                    state["drift_list"].append(coeffs.drift)
                    state["sigma_list"].append(coeffs.sigma_diag)
                    state["growth_list"].append(coeffs.growth)
                    state["z_list"].append(z_next)
                    state["logw_list"].append(logw_next)
                state["z"] = z_next
                state["logw"] = logw_next
                state["ess_frac_list"].append(WeightedParticleSimulator.ess_fraction(logw_next).detach())
                state["max_weight_frac_list"].append(
                    WeightedParticleSimulator.max_weight_fraction(logw_next).detach()
                )
                state["logw_range_list"].append(WeightedParticleSimulator.log_weight_range(logw_next).detach())

        metrics = {
            "n_active_perturbations": len(perturbation_ids),
            "perturbation_batch_size": batch_size,
            "loss_total": 0.0,
            "loss_end": 0.0,
            "loss_weak": 0.0,
            "loss_count": 0.0,
            "loss_reg": 0.0,
            "loss_causal": 0.0,
            **_diagnostics_from_lists(diagnostic_values),
            **_weight_diagnostics_from_tensors(
                ess_frac_steps=torch.cat(
                    [torch.stack(state["ess_frac_list"], dim=0) for state in chunk_states],
                    dim=1,
                ),
                max_weight_frac_steps=torch.cat(
                    [torch.stack(state["max_weight_frac_list"], dim=0) for state in chunk_states],
                    dim=1,
                ),
                logw_range_steps=torch.cat(
                    [torch.stack(state["logw_range_list"], dim=0) for state in chunk_states],
                    dim=1,
                ),
            ),
        }

        optimizer.zero_grad()

        loss_end = torch.tensor(0.0, device=self.device)
        loss_weak = torch.tensor(0.0, device=self.device)
        loss_count = torch.tensor(0.0, device=self.device)
        loss_reg = torch.tensor(0.0, device=self.device)
        loss_causal = torch.tensor(0.0, device=self.device)

        for state in chunk_states:
            local_groups = len(state["pids"])
            if store_history:
                z_steps = torch.stack(state["z_list"], dim=0)
                logw_steps = torch.stack(state["logw_list"], dim=0)
                drift_steps = torch.stack(state["drift_list"], dim=0)
                sigma_steps = torch.stack(state["sigma_list"], dim=0)
                growth_steps = torch.stack(state["growth_list"], dim=0)
                terminal_z = z_steps[-1]
                terminal_logw = logw_steps[-1]
            else:
                z_steps = None
                logw_steps = None
                drift_steps = None
                sigma_steps = None
                growth_steps = None
                terminal_z = state["z"]
                terminal_logw = state["logw"]

            pred_logw_abs = terminal_logw.float() + state["log_m0"].float().unsqueeze(-1)
            local_target_support, local_target_logw = self._subset_target_dicts(self.device, state["pids"])
            local_loss_end, _ = self.uot_loss(
                pred_z=terminal_z.float(),
                pred_logw_abs=pred_logw_abs,
                target_support=local_target_support,
                target_logw=local_target_logw,
                perturbation_ids=state["pids"],
            )
            loss_end = loss_end + local_loss_end

            if tc.lambda_weak > 0 and drift_steps is not None:
                local_loss_weak = self.weak_loss(
                    z_steps=z_steps,
                    logw_steps=logw_steps,
                    drift_steps=drift_steps,
                    sigma_steps=sigma_steps,
                    growth_steps=growth_steps,
                    tau_steps=tau_steps,
                    refresh_centers=False,
                )
                loss_weak = loss_weak + self._weighted_shard_loss(
                    local_loss_weak,
                    local_groups,
                    total_groups,
                )

            local_loss_reg = self.regularizer(
                drift_steps=drift_steps.float() if drift_steps is not None else None,
                sigma_steps=sigma_steps.float() if sigma_steps is not None else None,
                growth_steps=growth_steps.float() if growth_steps is not None else None,
            )
            loss_reg = loss_reg + self._weighted_shard_loss(
                local_loss_reg,
                local_groups,
                total_groups,
            )

        if tc.lambda_count > 0 and self.count_data is not None and perturbation_ids == self.supported_pids:
            self._validate_count_order(perturbation_ids)
            growth_all = torch.cat(
                [torch.stack(state["growth_list"], dim=0) for state in chunk_states],
                dim=1,
            )
            logw_all = torch.cat(
                [torch.stack(state["logw_list"], dim=0) for state in chunk_states],
                dim=1,
            )
            cd = self._get_count_tensors_for_device(self.device)
            loss_count = self.count_lik(
                growth_steps=growth_all.float(),
                logw_steps=logw_all.float(),
                tau_steps=tau_steps.float(),
                exposures=cd["exposures"],
                count_matrix=cd["counts"],
                n_totals=cd["n_totals"],
            )

        if causal_edge_scores_list or causal_mediator_tokens_list or causal_growth_context_list:
            loss_causal = self._causal_attention_loss_from_tensors(
                edge_scores_steps=(
                    torch.stack(causal_edge_scores_list, dim=0)
                    if causal_edge_scores_list
                    else None
                ),
                residual_edge_scores_steps=(
                    torch.stack(causal_residual_edge_scores_list, dim=0)
                    if causal_residual_edge_scores_list
                    else None
                ),
                residual_edge_magnitude_steps=(
                    torch.stack(causal_residual_edge_magnitude_list, dim=0)
                    if causal_residual_edge_magnitude_list
                    else None
                ),
                mediator_tokens_steps=(
                    torch.stack(causal_mediator_tokens_list, dim=0)
                    if causal_mediator_tokens_list
                    else None
                ),
                growth_context_steps=(
                    torch.stack(causal_growth_context_list, dim=0)
                    if causal_growth_context_list
                    else None
                ),
                causal_delta_steps=(
                    torch.stack(causal_delta_list, dim=0)
                    if causal_delta_list
                    else None
                ),
                tau_steps=tau_steps,
                perturbation_ids=perturbation_ids,
                epoch=epoch,
            )
        loss_extra = torch.tensor(0.0, device=self.device)

        loss = (
            tc.lambda_end * loss_end
            + tc.lambda_weak * loss_weak
            + tc.lambda_count * loss_count
            + loss_reg
            + loss_causal
            + loss_extra
        )

        model_reg = self.model.regularization(lambda_embed=tc.lambda_reg_embed)
        model_reg = model_reg + self.model.growth_bias_regularization(
            lambda_growth_bias=tc.lambda_reg_growth_bias
        )
        if self.scaler.is_enabled():
            self.scaler.scale(loss + model_reg).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            (loss + model_reg).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
            optimizer.step()

        metrics["loss_end"] = float(loss_end.item())
        metrics["loss_weak"] = float(loss_weak.item())
        metrics["loss_count"] = float(loss_count.item())
        metrics["loss_reg"] = float((loss_reg + model_reg).item())
        metrics["loss_causal"] = float(loss_causal.item())
        metrics["loss_extra"] = float(loss_extra.item())
        metrics["loss_total"] = float((loss + model_reg).item())
        metrics["ess_gate_status"] = _ess_gate_status(metrics, tc)
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
        if _uses_global_ecological_context(self.model):
            raise ValueError(
                "Global ecological context is not supported with multi-GPU sharding yet. "
                "Use a single training device so context is computed from the full perturbation set."
            )
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
        store_history = self._needs_rollout_history

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
            seed=self.config.training.seed + seed_offset + epoch,
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
        base_seed = self.config.training.seed + seed_offset + epoch * 1000
        noise_generators: Dict[str, torch.Generator] = {}
        for device, local_pids, local_slice in shards:
            noise_generators[device] = self._make_noise_generator(device, base_seed + 1009 * (len(noise_generators) + 1))
            shard_state[device] = {
                "pids": local_pids,
                "slice": local_slice,
                "z": z0_full[local_slice].to(device),
                "logw": logw0_full[local_slice].to(device),
                "log_m0": log_m0_full[local_slice].to(device),
            }
            if store_history:
                shard_state[device].update(
                    {
                        "z_list": [shard_state[device]["z"]],
                        "logw_list": [shard_state[device]["logw"]],
                        "drift_list": [],
                        "sigma_list": [],
                        "growth_list": [],
                    }
                )

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
                local_embedding_ids = self._embedding_ids_for_pids(local_pids)
                with autocast_ctx_for(device):
                    a_local = model.embedding(local_embedding_ids)
                    b_local = model.embedding.growth_intercepts(local_embedding_ids)
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
                    "b": b_local,
                    "eta": eta_local,
                }

            global_stats = GroupStatistics(
                log_n_g=torch.cat([stats.log_n_g for stats in summary_parts], dim=0),
                eta_g=torch.cat([stats.eta_g for stats in summary_parts], dim=0),
                phi_g=torch.cat([stats.phi_g for stats in summary_parts], dim=0),
            )
            global_context = self.model.context_agg.context_from_group_statistics(global_stats)

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
                        growth_intercept=local_cache[device]["b"],
                        eta_z=eta_local,
                        q=q_local,
                        s=s_local,
                    )

                if store_history:
                    state["drift_list"].append(coeffs.drift)
                    state["sigma_list"].append(coeffs.sigma_diag)
                    state["growth_list"].append(coeffs.growth)

                noise_local = torch.randn(
                    z_local.shape,
                    device=device,
                    dtype=z_local.dtype,
                    generator=noise_generators[device],
                )
                z_next = z_local + coeffs.drift * dtau + coeffs.sigma_diag * (dtau ** 0.5) * noise_local
                logw_next = logw_local + coeffs.growth * dtau
                state["z"] = z_next
                state["logw"] = logw_next
                if store_history:
                    state["z_list"].append(z_next)
                    state["logw_list"].append(logw_next)

        loss_end = torch.tensor(0.0, device=self.device)
        loss_weak = torch.tensor(0.0, device=self.device)
        loss_count = torch.tensor(0.0, device=self.device)
        loss_reg = torch.tensor(0.0, device=self.device)

        for device, local_pids, _ in shards:
            state = shard_state[device]
            local_groups = len(local_pids)
            if store_history:
                z_steps = torch.stack(state["z_list"], dim=0)
                logw_steps = torch.stack(state["logw_list"], dim=0)
                drift_steps = torch.stack(state["drift_list"], dim=0)
                sigma_steps = torch.stack(state["sigma_list"], dim=0)
                growth_steps = torch.stack(state["growth_list"], dim=0)
                terminal_z = z_steps[-1]
                terminal_logw = logw_steps[-1]
            else:
                z_steps = None
                logw_steps = None
                drift_steps = None
                sigma_steps = None
                growth_steps = None
                terminal_z = state["z"]
                terminal_logw = state["logw"]
            pred_logw_abs = terminal_logw.float() + state["log_m0"].float().unsqueeze(-1)
            local_target_support, local_target_logw = self._subset_target_dicts(device, local_pids)
            local_loss_end, _ = self.uot_loss(
                pred_z=terminal_z.float(),
                pred_logw_abs=pred_logw_abs,
                target_support=local_target_support,
                target_logw=local_target_logw,
                perturbation_ids=local_pids,
            )
            loss_end = loss_end + local_loss_end.to(self.device)

            if tc.lambda_weak > 0 and drift_steps is not None:
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

            local_loss_reg = self.regularizer(
                drift_steps=drift_steps.float() if drift_steps is not None else None,
                sigma_steps=sigma_steps.float() if sigma_steps is not None else None,
                growth_steps=growth_steps.float() if growth_steps is not None else None,
            )
            loss_reg = loss_reg + self._weighted_shard_loss(
                local_loss_reg,
                local_groups,
                total_groups,
            ).to(self.device)

        loss_reg = loss_reg + self.model.regularization(lambda_embed=tc.lambda_reg_embed)
        loss_reg = loss_reg + self.model.growth_bias_regularization(
            lambda_growth_bias=tc.lambda_reg_growth_bias
        )
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

        metrics = {
            "n_active_perturbations": len(perturbation_ids),
            "perturbation_batch_size": len(perturbation_ids),
            "loss_total": float(loss.item()),
            "loss_end": float(loss_end.item()),
            "loss_weak": float(loss_weak.item()),
            "loss_count": float(loss_count.item()),
            "loss_reg": float(loss_reg.item()),
            "loss_causal": 0.0,
        }
        metrics["ess_gate_status"] = _ess_gate_status(metrics, tc)
        return metrics

    def _active_perturbation_ids(self, stage: str) -> List[str]:
        if stage != "C":
            return self.supported_pids
        control_ids = [pid for pid in self.supported_pids if pid in self.model.control_ids]
        if not control_ids:
            raise ValueError("Stage C requested but no control perturbations are available.")
        return control_ids

    def _save_checkpoint(self, epoch: int, tag: str = "best",
                         ema: Optional[EMA] = None) -> Path:
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
        return path

    def _restore_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "count_lik_state" in ckpt:
            self.count_lik.load_state_dict(ckpt["count_lik_state"])

    def _save_ema_checkpoint(self, epoch: int, ema: EMA, tag: str = "best_ema") -> None:
        """Save a checkpoint with EMA weights applied."""
        ema.apply_shadow()
        path = self.output_dir / f"checkpoint_{tag}.pt"
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
        self._best_checkpoint_path = None
        self._patience_counter = 0
        self._divergence_counter = 0
        active_pids = self._active_perturbation_ids(stage)
        perturbation_batch_size = self._perturbation_batch_size(active_pids)
        run_manifest = build_run_manifest(
            config=self.config.model_dump(),
            supported_pids=self.supported_pids,
            active_pids=active_pids,
            stage=stage,
            n_epochs=epochs,
            output_dir=self.output_dir,
        )
        write_run_manifest(
            self.output_dir / "run_manifest.json",
            run_manifest,
        )
        append_run_manifest_record(self.output_dir / "run_manifest_stages.jsonl", run_manifest)
        print(
            f"[{stage}] Active perturbations={len(active_pids)} "
            f"batch_pids={perturbation_batch_size}"
        )

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
        last_epoch = start_epoch - 1

        for epoch in range(epochs):
            if control_ref_warmup > 0 and epoch == control_ref_warmup:
                self.model.unfreeze_control_reference()
                print(f"[{stage}] Released control reference at epoch {epoch}")

            absolute_epoch = start_epoch + epoch
            last_epoch = absolute_epoch
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
            self.history.loss_causal.append(float(metrics.get("loss_causal", 0.0)))
            self.history.loss_extra.append(float(metrics.get("loss_extra", 0.0)))
            self.history.context_norm.append(float(metrics.get("context_norm", math.nan)))
            self.history.q_entropy.append(float(metrics.get("q_entropy", math.nan)))
            self.history.freq_entropy.append(float(metrics.get("freq_entropy", math.nan)))
            self.history.within_attention_entropy.append(float(metrics.get("within_attention_entropy", math.nan)))
            self.history.group_attention_entropy.append(float(metrics.get("group_attention_entropy", math.nan)))
            self.history.within_effective_keys.append(float(metrics.get("within_effective_keys", math.nan)))
            self.history.group_effective_keys.append(float(metrics.get("group_effective_keys", math.nan)))
            self.history.mass_log_range.append(float(metrics.get("mass_log_range", math.nan)))
            self.history.state_to_mediator_effective_keys.append(
                float(metrics.get("state_to_mediator_effective_keys", math.nan))
            )
            self.history.local_to_global_mediator_effective_keys.append(
                float(metrics.get("local_to_global_mediator_effective_keys", math.nan))
            )
            self.history.mediator_to_group_effective_keys.append(
                float(metrics.get("mediator_to_group_effective_keys", math.nan))
            )
            self.history.edge_sparsity.append(float(metrics.get("edge_sparsity", math.nan)))
            self.history.effective_edge_mean.append(float(metrics.get("effective_edge_mean", math.nan)))
            self.history.baseline_edge_mean.append(float(metrics.get("baseline_edge_mean", math.nan)))
            self.history.residual_edge_sparsity_loss.append(
                float(metrics.get("residual_edge_sparsity_loss", math.nan))
            )
            self.history.edge_entropy.append(float(metrics.get("edge_entropy", math.nan)))
            self.history.control_edge_norm.append(float(metrics.get("control_edge_norm", math.nan)))
            self.history.mediator_orthogonality.append(float(metrics.get("mediator_orthogonality", math.nan)))
            self.history.residual_edge_abs_mean.append(float(metrics.get("residual_edge_abs_mean", math.nan)))
            self.history.residual_edge_signed_mean.append(float(metrics.get("residual_edge_signed_mean", math.nan)))
            self.history.mediator_usage_entropy.append(float(metrics.get("mediator_usage_entropy", math.nan)))
            self.history.mediator_usage_min.append(float(metrics.get("mediator_usage_min", math.nan)))
            self.history.mediator_usage_max.append(float(metrics.get("mediator_usage_max", math.nan)))
            self.history.terminal_ess_frac_mean.append(float(metrics.get("terminal_ess_frac_mean", math.nan)))
            self.history.terminal_ess_frac_min.append(float(metrics.get("terminal_ess_frac_min", math.nan)))
            self.history.min_ess_frac_mean.append(float(metrics.get("min_ess_frac_mean", math.nan)))
            self.history.max_weight_frac_mean.append(float(metrics.get("max_weight_frac_mean", math.nan)))
            self.history.logw_range_max.append(float(metrics.get("logw_range_max", math.nan)))
            self.history.ess_gate_status.append(str(metrics.get("ess_gate_status", "not_available")))

            # Best checkpoint (training weights)
            if metrics["loss_total"] < self._best_loss:
                self._best_loss = metrics["loss_total"]
                self._patience_counter = 0
                self._divergence_counter = 0
                self._best_checkpoint_path = self._save_checkpoint(absolute_epoch, "best", ema=ema)
                # Also save EMA-specific checkpoint
                if ema is not None:
                    self._save_ema_checkpoint(absolute_epoch, ema, tag="best_ema")
            else:
                self._patience_counter += 1

            should_stop_for_divergence = False
            loss_total = metrics["loss_total"]
            if not math.isfinite(loss_total):
                self._divergence_counter += tc.divergence_patience
                print(
                    f"[{stage}] Divergence warning at epoch {absolute_epoch}: "
                    f"loss_total is non-finite ({loss_total})"
                )
                should_stop_for_divergence = True
            elif (
                epoch >= tc.divergence_min_epochs
                and math.isfinite(self._best_loss)
                and self._best_loss < math.inf
                and loss_total > self._best_loss * tc.divergence_factor
            ):
                self._divergence_counter += 1
                print(
                    f"[{stage}] Divergence warning at epoch {absolute_epoch}: "
                    f"loss_total={loss_total:.4g} exceeds "
                    f"{tc.divergence_factor:.1f}x best_loss={self._best_loss:.4g} "
                    f"({self._divergence_counter}/{tc.divergence_patience})"
                )
                should_stop_for_divergence = self._divergence_counter >= tc.divergence_patience
            else:
                self._divergence_counter = 0

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
                    f"reg={metrics['loss_reg']:.4f} "
                    f"causal={float(metrics.get('loss_causal', 0.0)):.4f} | "
                    f"lr={cur_lr:.2e} t={elapsed:.1f}s"
                )

            if epoch % tc.checkpoint_every == 0:
                self._save_checkpoint(absolute_epoch, f"epoch{absolute_epoch:04d}", ema=ema)

            # Early stopping
            if self._patience_counter >= tc.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if should_stop_for_divergence:
                print(f"[{stage}] Stopping after repeated divergence at epoch {absolute_epoch}")
                break

        # Final EMA checkpoint. Keep this separate from checkpoint_best_ema.pt
        # so a late unstable trajectory cannot overwrite the best EMA weights.
        if ema is not None and last_epoch >= start_epoch:
            self._save_ema_checkpoint(last_epoch, ema, tag="final_ema")

        if self._best_checkpoint_path is not None and self._best_checkpoint_path.exists():
            self._restore_checkpoint(self._best_checkpoint_path)

        # Save history
        df = self.history.to_dataframe()
        df.to_csv(self.output_dir / "training_history.csv", index=False)
        return self.history
