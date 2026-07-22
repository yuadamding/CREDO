"""Frozen expression VAE used by transformer-SDE v2 artifacts."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExpressionVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        encoder: list[nn.Module] = []
        width = self.input_dim
        for _ in range(max(self.depth, 1)):
            encoder.extend((nn.Linear(width, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout)))
            width = self.hidden_dim
        self.encoder = nn.Sequential(*encoder)
        self.mu_head = nn.Linear(width, self.latent_dim)
        self.logvar_head = nn.Linear(width, self.latent_dim)
        decoder: list[nn.Module] = []
        width = self.latent_dim
        for _ in range(max(self.depth, 1)):
            decoder.extend((nn.Linear(width, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout)))
            width = self.hidden_dim
        decoder.append(nn.Linear(width, self.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(x)
        return self.decode(self.reparameterize(mean, log_variance)), mean, log_variance


__all__ = ["ExpressionVAE"]
