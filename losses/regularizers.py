"""Regularization losses for the full model."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def embedding_shrinkage(embeddings: torch.Tensor) -> torch.Tensor:
    """L2 shrinkage on non-control perturbation embeddings."""
    return (embeddings ** 2).mean()


def diffusion_magnitude_penalty(sigma_diag: torch.Tensor) -> torch.Tensor:
    """Penalise large diagonal diffusion coefficients. sigma_diag: [G, N, d]."""
    return (sigma_diag ** 2).mean()


def drift_action_penalty(drift: torch.Tensor) -> torch.Tensor:
    """Penalise large drift magnitudes. drift: [G, N, d]."""
    return (drift ** 2).mean()


def growth_action_penalty(growth: torch.Tensor) -> torch.Tensor:
    """Penalise large absolute growth rates. growth: [G, N]."""
    return (growth ** 2).mean()


class RolloutRegularizer(nn.Module):
    """Computes all rollout-level regularization terms.

    Parameters
    ----------
    lambda_embed, lambda_diffusion, lambda_drift, lambda_growth:
        Weights for each penalty.
    """

    def __init__(
        self,
        lambda_embed: float = 1e-4,
        lambda_diffusion: float = 1e-4,
        lambda_drift: float = 1e-4,
        lambda_growth: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_embed = lambda_embed
        self.lambda_diffusion = lambda_diffusion
        self.lambda_drift = lambda_drift
        self.lambda_growth = lambda_growth

    def forward(
        self,
        embeddings: torch.Tensor,     # [G, r]
        drift_steps: Optional[torch.Tensor],    # [K, G, N, d]
        sigma_steps: Optional[torch.Tensor],    # [K, G, N, d]
        growth_steps: Optional[torch.Tensor],   # [K, G, N]
    ) -> torch.Tensor:
        reg = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
        if self.lambda_embed > 0:
            reg = reg + self.lambda_embed * embedding_shrinkage(embeddings)
        if self.lambda_diffusion > 0 and sigma_steps is not None:
            reg = reg + self.lambda_diffusion * diffusion_magnitude_penalty(sigma_steps)
        if self.lambda_drift > 0 and drift_steps is not None:
            reg = reg + self.lambda_drift * drift_action_penalty(drift_steps)
        if self.lambda_growth > 0 and growth_steps is not None:
            reg = reg + self.lambda_growth * growth_action_penalty(growth_steps)
        return reg
