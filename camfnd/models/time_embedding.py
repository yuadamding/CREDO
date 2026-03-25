from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TimeEmbedding(nn.Module):
    """Small deterministic time-feature map for the P4 -> P60 interval.

    The Stage-I benchmark only uses t in [0, 1], so a low-order Fourier basis is
    sufficient. The embedding is intentionally simple because Step 3 does not yet
    include mean-field context.
    """

    def __init__(self, n_frequencies: int = 2) -> None:
        super().__init__()
        if int(n_frequencies) < 0:
            raise ValueError("n_frequencies must be nonnegative.")
        self.n_frequencies = int(n_frequencies)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.n_frequencies

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t[:, None]
        elif t.ndim != 2 or t.shape[1] != 1:
            raise ValueError("t must have shape [], [N], or [N, 1].")

        feats = [t]
        for k in range(1, self.n_frequencies + 1):
            omega_t = 2.0 * math.pi * k * t
            feats.append(torch.sin(omega_t))
            feats.append(torch.cos(omega_t))
        return torch.cat(feats, dim=1)
