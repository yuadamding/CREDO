"""Multi-time endpoint utilities for trajectory CREDO experiments."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn

from ..data.core import TrajectoryProblem
from ..models.weighted_sde import ParticleRollout
from .uot import UOTLoss


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
    trajectory: TrajectoryProblem,
    *,
    time_labels: Iterable[str] | None = None,
    perturbation_ids: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    """Convert pooled trajectory measures into UOT target dictionaries."""
    labels = list(trajectory.time_labels if time_labels is None else time_labels)
    if not labels:
        raise ValueError("time_labels must contain at least one label")
    missing = [label for label in labels if label not in trajectory.measures]
    if missing:
        raise KeyError(f"Unknown trajectory time labels: {missing}")
    if not all(isinstance(key, str) for label in labels for key in trajectory.measures[label]):
        raise ValueError("build_target_tensors_by_time expects pooled perturbation-id trajectory keys")
    pids = list(perturbation_ids or trajectory.perturbation_ids)

    target_support: Dict[str, Dict[str, torch.Tensor]] = {}
    target_logw: Dict[str, Dict[str, torch.Tensor]] = {}
    for label in labels:
        target_support[label] = {}
        target_logw[label] = {}
        for pid in pids:
            if pid not in trajectory.measures[label]:
                continue
            mu = trajectory.measures[label][pid]
            sup, weights = mu.to_torch(device=str(device), dtype=dtype)
            target_support[label][pid] = sup
            target_logw[label][pid] = torch.log(weights + 1e-30)
    return target_support, target_logw


class MultiTimeEndpointLoss(nn.Module):
    """Apply the existing endpoint UOT proxy at multiple rollout checkpoints."""

    def __init__(
        self,
        uot_loss: UOTLoss,
        time_weights: Dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.uot_loss = uot_loss
        self.time_weights = time_weights or {}

    def forward(
        self,
        rollout: ParticleRollout,
        checkpoint_indices: Dict[str, int],
        target_support_by_time: Dict[str, Dict[str, torch.Tensor]],
        target_logw_by_time: Dict[str, Dict[str, torch.Tensor]],
        perturbation_ids: list[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rollout.log_m0 is None:
            raise ValueError("MultiTimeEndpointLoss requires rollout.log_m0 for absolute masses")

        total = torch.tensor(0.0, device=rollout.z_steps.device, dtype=rollout.z_steps.dtype)
        logs: Dict[str, torch.Tensor] = {}

        for time_label, idx in checkpoint_indices.items():
            if time_label not in target_support_by_time:
                continue

            pred_z = rollout.z_steps[idx]
            pred_logw_abs = rollout.log_m0[:, None] + rollout.logw_steps[idx]
            loss_t, _ = self.uot_loss(
                pred_z=pred_z,
                pred_logw_abs=pred_logw_abs,
                target_support=target_support_by_time[time_label],
                target_logw=target_logw_by_time[time_label],
                perturbation_ids=perturbation_ids,
            )
            weight = float(self.time_weights.get(time_label, 1.0))
            total = total + weight * loss_t
            logs[f"endpoint/{time_label}"] = loss_t.detach()

        return total, logs
