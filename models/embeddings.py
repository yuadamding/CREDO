"""Perturbation embeddings with exact zero control anchor.

The control perturbation always maps to the zero vector.  Non-control
perturbations learn free embeddings initialised to small random values
(or one-hot if embedding_dim >= n_non_controls).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class PerturbationEmbedding(nn.Module):
    """Learnable perturbation embeddings, controls fixed at zero.

    Parameters
    ----------
    perturbation_ids:
        All perturbation ids in order.  This list defines the index mapping.
    control_ids:
        Subset that are controls.  Their embeddings are always zero.
    embedding_dim:
        Dimension r of each embedding vector.
    """

    def __init__(
        self,
        perturbation_ids: List[str],
        control_ids: List[str],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.perturbation_ids = perturbation_ids
        self.control_ids = set(control_ids)
        self.embedding_dim = embedding_dim

        self._id_to_idx: Dict[str, int] = {p: i for i, p in enumerate(perturbation_ids)}

        non_ctrl = [p for p in perturbation_ids if p not in self.control_ids]
        self.non_control_ids = non_ctrl
        self._nc_to_local: Dict[str, int] = {p: i for i, p in enumerate(non_ctrl)}

        n_nc = len(non_ctrl)
        # initialise embeddings
        weight = torch.zeros(n_nc, embedding_dim)
        if n_nc > 0:
            if embedding_dim >= n_nc:
                # one-hot initialisation for identifiability
                for i in range(n_nc):
                    if i < embedding_dim:
                        weight[i, i] = 1.0
            else:
                nn.init.xavier_uniform_(weight)

        if n_nc > 0:
            self.embeddings = nn.Parameter(weight)
        else:
            self.register_parameter("embeddings", None)

        # Device sentinel so .to(device) propagates even when embeddings is None
        self.register_buffer("_device_sentinel", torch.zeros(1))

    def forward(self, perturbation_ids: List[str]) -> torch.Tensor:
        """Return embedding matrix of shape [len(perturbation_ids), r]."""
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        out = torch.zeros(len(perturbation_ids), self.embedding_dim, device=device, dtype=dtype)
        for i, pid in enumerate(perturbation_ids):
            if pid not in self.control_ids and self.embeddings is not None:
                local_idx = self._nc_to_local[pid]
                out[i] = self.embeddings[local_idx]
        return out  # controls remain exactly zero

    def control_anchor_is_exact(self) -> bool:
        """Verify that control embeddings are exactly zero (no gradient flow)."""
        if self.embeddings is None:
            return True
        for pid in self.control_ids:
            if pid in self._nc_to_local:
                local_idx = self._nc_to_local[pid]
                if not torch.all(self.embeddings[local_idx] == 0).item():
                    return False
        return True

    def regularization(self) -> torch.Tensor:
        """L2 norm of non-control embeddings for shrinkage regularization."""
        if self.embeddings is None:
            return torch.tensor(0.0)
        return (self.embeddings ** 2).mean()

    def snapshot(self) -> Dict[str, List[float]]:
        """Return embedding values as a plain dict for logging."""
        out = {}
        for pid in self.perturbation_ids:
            if pid in self.control_ids:
                out[pid] = [0.0] * self.embedding_dim
            elif self.embeddings is not None:
                local_idx = self._nc_to_local[pid]
                out[pid] = self.embeddings[local_idx].detach().cpu().tolist()
        return out


class TimeEmbedding(nn.Module):
    """Deterministic Fourier basis for normalized time tau in [0, 1].

    Output: [tau, sin(pi*tau), cos(pi*tau), sin(2*pi*tau), cos(2*pi*tau), ...]
    output_dim = 1 + 2 * n_frequencies
    """

    def __init__(self, n_frequencies: int = 4) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.n_frequencies

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: scalar or [...] -> [..., output_dim]."""
        parts = [tau.unsqueeze(-1)]
        for k in range(1, self.n_frequencies + 1):
            parts.append(torch.sin(k * torch.pi * tau).unsqueeze(-1))
            parts.append(torch.cos(k * torch.pi * tau).unsqueeze(-1))
        return torch.cat(parts, dim=-1)
