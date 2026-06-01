"""Perturbation embeddings with anchored, free, or soft-reference control modes."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class PerturbationEmbedding(nn.Module):
    """Learnable perturbation embeddings with flexible control handling.

    Parameters
    ----------
    perturbation_ids:
        All perturbation ids in order.  This list defines the index mapping.
    control_ids:
        Subset that are biological controls.
    embedding_dim:
        Dimension r of each embedding vector.
    control_mode:
        - ``anchored``: controls are fixed at exactly zero
        - ``free``: controls are treated like ordinary learnable perturbations
        - ``soft_ref``: a shared reference embedding is learned, and controls have
          exactly zero residual around that reference
    control_ref_penalty:
        Additional L2 penalty weight applied to the shared reference embedding when
        ``control_mode='soft_ref'``.
    """

    def __init__(
        self,
        perturbation_ids: List[str],
        control_ids: List[str],
        embedding_dim: int,
        control_mode: str = "soft_ref",
        control_ref_penalty: float = 5e-4,
        use_growth_intercept: bool = True,
        shared_guide_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.perturbation_ids = perturbation_ids
        self.all_control_ids = set(control_ids)
        self.control_mode = control_mode
        self.anchor_controls = control_mode == "anchored"
        self.control_ids = set(control_ids) if control_mode in {"anchored", "soft_ref"} else set()
        self.embedding_dim = embedding_dim
        self.control_ref_penalty = float(control_ref_penalty)
        self.shared_guide_embedding = bool(shared_guide_embedding)

        self._id_to_idx: Dict[str, int] = {p: i for i, p in enumerate(perturbation_ids)}

        non_ctrl = [p for p in perturbation_ids if p not in self.control_ids]
        self.non_control_ids = non_ctrl
        self._nc_to_local: Dict[str, int] = {p: i for i, p in enumerate(non_ctrl)}

        n_nc = len(non_ctrl)
        # initialise residual / free embeddings
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
            if use_growth_intercept:
                self.growth_bias = nn.Parameter(torch.zeros(n_nc))
            else:
                self.register_parameter("growth_bias", None)
        else:
            self.register_parameter("embeddings", None)
            self.register_parameter("growth_bias", None)

        if self.control_mode == "soft_ref":
            self.reference_embedding = nn.Parameter(torch.zeros(embedding_dim))
        else:
            self.register_parameter("reference_embedding", None)

        if self.shared_guide_embedding:
            self.shared_embedding = nn.Parameter(torch.zeros(embedding_dim))
            if use_growth_intercept:
                self.shared_growth_bias = nn.Parameter(torch.zeros(()))
            else:
                self.register_parameter("shared_growth_bias", None)
        else:
            self.register_parameter("shared_embedding", None)
            self.register_parameter("shared_growth_bias", None)

        # Device sentinel so .to(device) propagates even when embeddings is None
        self.register_buffer("_device_sentinel", torch.zeros(1))

    def forward(self, perturbation_ids: List[str]) -> torch.Tensor:
        """Return embedding matrix of shape [len(perturbation_ids), r]."""
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        if self.shared_guide_embedding:
            return self.shared_embedding.to(device=device, dtype=dtype).unsqueeze(0).expand(
                len(perturbation_ids),
                -1,
            )
        out = torch.zeros(len(perturbation_ids), self.embedding_dim, device=device, dtype=dtype)
        for i, pid in enumerate(perturbation_ids):
            if pid not in self.control_ids and self.embeddings is not None:
                local_idx = self._nc_to_local[pid]
                out[i] = self.embeddings[local_idx]
        if self.reference_embedding is not None:
            out = out + self.reference_embedding.to(device=device, dtype=dtype).unsqueeze(0)
        return out

    def residuals(self, perturbation_ids: List[str]) -> torch.Tensor:
        """Return perturbation residuals without the shared soft reference.

        In ``soft_ref`` mode, controls have exactly zero residual and
        non-controls return their learned residual around the reference.  In
        ``anchored`` mode this matches the effective embedding.  In ``free``
        mode all learned perturbation embeddings are treated as residuals.
        """
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        out = torch.zeros(len(perturbation_ids), self.embedding_dim, device=device, dtype=dtype)
        if self.shared_guide_embedding:
            out = self.shared_embedding.to(device=device, dtype=dtype).unsqueeze(0).expand(
                len(perturbation_ids),
                -1,
            ).clone()
            for i, pid in enumerate(perturbation_ids):
                if pid in self.all_control_ids:
                    out[i].zero_()
            return out
        if self.embeddings is None:
            return out
        for i, pid in enumerate(perturbation_ids):
            if pid in self._nc_to_local:
                out[i] = self.embeddings[self._nc_to_local[pid]]
        return out

    def control_anchor_is_exact(self) -> bool:
        """Verify that anchored-mode control residuals are exactly zero."""
        if self.shared_guide_embedding:
            return False
        if not self.anchor_controls:
            return False
        if self.embeddings is None:
            return True
        for pid in self.control_ids:
            if pid in self._nc_to_local:
                local_idx = self._nc_to_local[pid]
                if not torch.all(self.embeddings[local_idx] == 0).item():
                    return False
        return True

    def growth_intercepts(self, perturbation_ids: List[str]) -> torch.Tensor:
        """Return explicit perturbation growth intercepts ``b_g`` of shape [G]."""
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        out = torch.zeros(len(perturbation_ids), device=device, dtype=dtype)
        if self.shared_guide_embedding:
            if self.shared_growth_bias is None:
                return out
            return self.shared_growth_bias.to(device=device, dtype=dtype).expand(len(perturbation_ids))
        if self.growth_bias is None:
            return out
        for i, pid in enumerate(perturbation_ids):
            if pid not in self.control_ids:
                out[i] = self.growth_bias[self._nc_to_local[pid]]
        return out

    def regularization(self, lambda_embed: float = 0.0) -> torch.Tensor:
        """Regularization over residual embeddings and the shared control reference."""
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        reg = torch.tensor(0.0, device=device, dtype=dtype)
        if self.shared_guide_embedding:
            if self.shared_embedding is not None and lambda_embed > 0:
                reg = reg + float(lambda_embed) * (self.shared_embedding ** 2).mean()
            return reg
        if self.embeddings is not None and lambda_embed > 0:
            reg = reg + float(lambda_embed) * (self.embeddings ** 2).mean()
        if self.reference_embedding is not None and self.control_ref_penalty > 0:
            reg = reg + self.control_ref_penalty * (self.reference_embedding ** 2).mean()
        return reg

    def growth_bias_regularization(self, lambda_growth_bias: float = 0.0) -> torch.Tensor:
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        reg = torch.tensor(0.0, device=device, dtype=dtype)
        if self.shared_guide_embedding:
            if self.shared_growth_bias is not None and lambda_growth_bias > 0:
                reg = reg + float(lambda_growth_bias) * (self.shared_growth_bias ** 2)
            return reg
        if self.growth_bias is not None and lambda_growth_bias > 0:
            reg = reg + float(lambda_growth_bias) * (self.growth_bias ** 2).mean()
        return reg

    def snapshot(self) -> Dict[str, List[float]]:
        """Return embedding values as a plain dict for logging."""
        out = {}
        if self.shared_guide_embedding:
            value = self.shared_embedding.detach().cpu()
            for pid in self.perturbation_ids:
                out[pid] = value.tolist()
            out["__shared_guide__"] = value.tolist()
            return out
        ref = (
            self.reference_embedding.detach().cpu()
            if self.reference_embedding is not None
            else None
        )
        for pid in self.perturbation_ids:
            if pid in self.control_ids and ref is None:
                out[pid] = [0.0] * self.embedding_dim
            elif self.embeddings is not None and pid in self._nc_to_local:
                local_idx = self._nc_to_local[pid]
                value = self.embeddings[local_idx].detach().cpu()
                if ref is not None:
                    value = value + ref
                out[pid] = value.tolist()
            elif ref is not None:
                out[pid] = ref.tolist()
        if ref is not None:
            out["__control_reference__"] = ref.tolist()
        return out

    def freeze_reference(self) -> None:
        if self.reference_embedding is not None:
            self.reference_embedding.requires_grad_(False)

    def unfreeze_reference(self) -> None:
        if self.reference_embedding is not None:
            self.reference_embedding.requires_grad_(True)


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
