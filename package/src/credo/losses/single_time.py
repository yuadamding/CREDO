"""Single-time effect-path regularizers and claim helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch


def control_null_effect_loss(effect_scores: torch.Tensor, is_control: torch.Tensor) -> torch.Tensor:
    """Penalize nonzero effects for held-out or leave-one-control references."""
    if effect_scores.ndim == 0:
        effect_scores = effect_scores.reshape(1)
    mask = is_control.to(device=effect_scores.device, dtype=torch.bool)
    if mask.shape[0] != effect_scores.shape[0]:
        raise ValueError("is_control must have one entry per effect score.")
    if not bool(mask.any()):
        return effect_scores.new_tensor(0.0)
    return effect_scores[mask].square().mean()


def minimal_effect_action_loss(
    *,
    drift_steps: torch.Tensor | None = None,
    sigma_steps: torch.Tensor | None = None,
    growth_steps: torch.Tensor | None = None,
    drift_weight: float = 1.0,
    diffusion_weight: float = 1.0,
    growth_weight: float = 1.0,
) -> torch.Tensor:
    """Small-action regularizer for underidentified single-time effect paths."""
    tensors = [tensor for tensor in (drift_steps, sigma_steps, growth_steps) if tensor is not None]
    if not tensors:
        return torch.tensor(0.0)
    loss = tensors[0].new_tensor(0.0)
    if drift_steps is not None and drift_weight > 0:
        loss = loss + float(drift_weight) * drift_steps.square().mean()
    if sigma_steps is not None and diffusion_weight > 0:
        loss = loss + float(diffusion_weight) * sigma_steps.square().mean()
    if growth_steps is not None and growth_weight > 0:
        loss = loss + float(growth_weight) * growth_steps.square().mean()
    return loss


def guide_concordance_effect_loss(effect_scores: torch.Tensor, target_ids: Sequence[str]) -> torch.Tensor:
    """Penalize guide-level effects that disagree within target genes."""
    if effect_scores.ndim != 1:
        raise ValueError("effect_scores must be a 1D tensor.")
    if len(target_ids) != int(effect_scores.shape[0]):
        raise ValueError("target_ids must have one entry per effect score.")
    by_target: dict[str, list[int]] = defaultdict(list)
    for idx, target in enumerate(target_ids):
        by_target[str(target)].append(idx)
    losses = []
    for indices in by_target.values():
        if len(indices) < 2:
            continue
        values = effect_scores[torch.as_tensor(indices, device=effect_scores.device)]
        losses.append((values - values.mean()).square().mean())
    if not losses:
        return effect_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


__all__ = [
    "control_null_effect_loss",
    "guide_concordance_effect_loss",
    "minimal_effect_action_loss",
]
