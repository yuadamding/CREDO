"""Multi-time endpoint utilities for trajectory CREDO experiments."""
from __future__ import annotations

from typing import Dict, Iterable, Literal, Sequence, Tuple

import torch
import torch.nn as nn

from ..data.core import MeasureKey, SparseTrajectoryProblem, TrajectoryProblem
from ..models.weighted_sde import ParticleRollout
from .endpoint import EndpointGeometryMassLoss


def make_observed_tau_grid(
    observed_taus: Sequence[float],
    steps_per_interval: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a rollout grid that contains every observed tau exactly."""
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be >= 1")
    if len(observed_taus) < 2:
        raise ValueError("Need at least two observed tau values")
    taus = [float(tau) for tau in observed_taus]
    if any(b <= a for a, b in zip(taus[:-1], taus[1:])):
        raise ValueError(f"observed_taus must be strictly increasing, got {taus}")

    pieces = []
    for start, stop in zip(taus[:-1], taus[1:]):
        segment = torch.linspace(start, stop, steps_per_interval + 1, device=device, dtype=dtype)
        if pieces:
            segment = segment[1:]
        pieces.append(segment)
    return torch.cat(pieces)


def checkpoint_indices_for_taus(
    tau_steps: torch.Tensor,
    time_labels: Sequence[str],
    observed_taus: Sequence[float],
    *,
    atol: float = 1e-6,
) -> Dict[str, int]:
    """Map observed time labels to exact/near-exact indices in ``tau_steps``."""
    if tau_steps.ndim != 1:
        raise ValueError("tau_steps must be a 1D tensor")
    if len(time_labels) != len(observed_taus):
        raise ValueError("time_labels and observed_taus must have the same length")

    out: Dict[str, int] = {}
    for label, tau in zip(time_labels, observed_taus):
        distances = (tau_steps.detach().cpu() - float(tau)).abs()
        idx = int(torch.argmin(distances).item())
        if float(distances[idx]) > atol:
            raise ValueError(f"No rollout checkpoint for {label!r} at tau={tau}")
        out[str(label)] = idx
    return out


def checkpoint_indices_for_trajectory(
    tau_steps: torch.Tensor,
    trajectory: TrajectoryProblem,
    *,
    time_labels: Sequence[str] | None = None,
    atol: float = 1e-6,
) -> Dict[str, int]:
    labels = list(trajectory.time_labels if time_labels is None else time_labels)
    return checkpoint_indices_for_taus(
        tau_steps,
        labels,
        [trajectory.tau(label) for label in labels],
        atol=atol,
    )


def build_target_tensors_by_time(
    trajectory: TrajectoryProblem | SparseTrajectoryProblem,
    *,
    time_labels: Iterable[str] | None = None,
    perturbation_ids: Sequence[str] | None = None,
    measure_keys: Sequence[MeasureKey] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, Dict[MeasureKey, torch.Tensor]], Dict[str, Dict[MeasureKey, torch.Tensor]]]:
    """Convert trajectory measures into endpoint-loss target dictionaries.

    Pooled trajectories use perturbation-id keys.  Sample-aware trajectories use
    ``(sample_id, perturbation_id)`` keys and should pass the same key ordering
    to ``MultiTimeEndpointLoss``.
    """
    labels = list(trajectory.time_labels if time_labels is None else time_labels)
    if not labels:
        raise ValueError("time_labels must contain at least one label")
    missing = [label for label in labels if label not in trajectory.measures]
    if missing:
        raise KeyError(f"Unknown trajectory time labels: {missing}")
    if measure_keys is not None and perturbation_ids is not None:
        raise ValueError("Pass either measure_keys or perturbation_ids, not both.")
    if measure_keys is not None:
        keys = list(measure_keys)
    elif perturbation_ids is not None:
        requested = {str(pid) for pid in perturbation_ids}
        if all(isinstance(key, str) for key in trajectory.keys):
            keys = [str(pid) for pid in perturbation_ids]
        else:
            keys = [key for key in trajectory.keys if isinstance(key, tuple) and key[1] in requested]
    else:
        keys = list(trajectory.keys)

    target_support: Dict[str, Dict[MeasureKey, torch.Tensor]] = {}
    target_logw: Dict[str, Dict[MeasureKey, torch.Tensor]] = {}
    for label in labels:
        target_support[label] = {}
        target_logw[label] = {}
        for key in keys:
            if key not in trajectory.measures[label]:
                continue
            mu = trajectory.measures[label][key]
            sup, weights = mu.to_torch(device=str(device), dtype=dtype)
            target_support[label][key] = sup
            target_logw[label][key] = torch.log(weights + 1e-30)
    return target_support, target_logw


class MultiTimeEndpointLoss(nn.Module):
    """Apply endpoint geometry-plus-log-mass loss at rollout checkpoints."""

    def __init__(
        self,
        uot_loss: EndpointGeometryMassLoss,
        time_weights: Dict[str, float] | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        fail_on_empty: bool = True,
        normalize_time_weights: bool = False,
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.uot_loss = uot_loss
        self.time_weights = time_weights or {}
        self.reduction = reduction
        self.fail_on_empty = fail_on_empty
        self.normalize_time_weights = normalize_time_weights

    def forward(
        self,
        rollout: ParticleRollout,
        checkpoint_indices: Dict[str, int],
        target_support_by_time: Dict[str, Dict[MeasureKey, torch.Tensor]],
        target_logw_by_time: Dict[str, Dict[MeasureKey, torch.Tensor]],
        perturbation_ids: list[MeasureKey] | None = None,
        *,
        prediction_keys: list[MeasureKey] | None = None,
        embedding_ids: list[str] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rollout.log_m0 is None:
            raise ValueError("MultiTimeEndpointLoss requires rollout.log_m0 for absolute masses")
        keys = list(prediction_keys if prediction_keys is not None else perturbation_ids or [])
        if not keys:
            raise ValueError("MultiTimeEndpointLoss requires prediction_keys or perturbation_ids")
        if embedding_ids is not None and len(embedding_ids) != len(keys):
            raise ValueError("embedding_ids length must match prediction key count")
        if rollout.z_steps.shape[1] != len(keys):
            raise ValueError(
                "rollout group dimension must match prediction key count: "
                f"{rollout.z_steps.shape[1]} != {len(keys)}"
            )

        total = torch.tensor(0.0, device=rollout.z_steps.device, dtype=rollout.z_steps.dtype)
        logs: Dict[str, torch.Tensor] = {}
        active_weight_sum = 0.0

        for time_label, idx in checkpoint_indices.items():
            if time_label not in target_support_by_time:
                raise KeyError(f"Missing target support for checkpoint {time_label!r}")
            if time_label not in target_logw_by_time:
                raise KeyError(f"Missing target log-weights for checkpoint {time_label!r}")

            target_support = target_support_by_time[time_label]
            target_logw = target_logw_by_time[time_label]
            active_ids = [pid for pid in keys if pid in target_support]
            n_active = len(active_ids)
            n_missing = len(keys) - n_active
            logs[f"endpoint/{time_label}/n_active_keys"] = torch.tensor(
                n_active, device=rollout.z_steps.device
            )
            logs[f"endpoint/{time_label}/n_missing_keys"] = torch.tensor(
                n_missing, device=rollout.z_steps.device
            )
            logs[f"endpoint/{time_label}/time_weight"] = torch.tensor(
                float(self.time_weights.get(time_label, 1.0)),
                device=rollout.z_steps.device,
                dtype=rollout.z_steps.dtype,
            )
            if n_active == 0:
                if self.fail_on_empty:
                    raise ValueError(f"No active target keys for checkpoint {time_label!r}")
                continue

            pred_z = rollout.z_steps[idx]
            pred_logw_abs = rollout.log_m0[:, None] + rollout.logw_steps[idx]
            kwargs = {
                "pred_z": pred_z,
                "pred_logw_abs": pred_logw_abs,
                "target_support": target_support,
                "target_logw": target_logw,
                "perturbation_ids": keys,
            }
            if isinstance(self.uot_loss, EndpointGeometryMassLoss):
                kwargs["fail_on_missing_target"] = False
            loss_t, components = self.uot_loss.component_dict(**kwargs)
            loss_scaled = loss_t / float(n_active) if self.reduction == "mean" else loss_t
            weight = float(self.time_weights.get(time_label, 1.0))
            total = total + weight * loss_scaled
            active_weight_sum += weight
            logs[f"endpoint/{time_label}"] = loss_scaled.detach()
            logs[f"endpoint/{time_label}/loss_sum"] = loss_t.detach()
            logs[f"endpoint/{time_label}/loss_mean_per_key"] = (loss_t / float(n_active)).detach()
            geom_values = [components[pid]["geom"] for pid in active_ids if pid in components]
            mass_values = [components[pid]["mass"] for pid in active_ids if pid in components]
            if geom_values:
                logs[f"endpoint/{time_label}/geom_mean"] = torch.stack(geom_values).mean().detach()
                logs[f"endpoint/{time_label}/mass_mean"] = torch.stack(mass_values).mean().detach()

        if self.normalize_time_weights and active_weight_sum > 0.0:
            total = total / active_weight_sum
            logs["endpoint/time_weight_normalizer"] = torch.tensor(
                active_weight_sum, device=rollout.z_steps.device, dtype=rollout.z_steps.dtype
            )

        return total, logs
