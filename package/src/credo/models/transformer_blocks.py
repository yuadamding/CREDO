"""Permutation-equivariant transformer blocks for finite-measure particles."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MassBiasedCrossAttention(nn.Module):
    """Multi-head cross-attention with an optional additive key log-weight bias."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.mass_attention_temperature = float(mass_attention_temperature)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_entropy: torch.Tensor | None = None
        self.last_effective_keys: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention.

        Parameters
        ----------
        query:
            Tensor with shape ``[B, Tq, D]``.
        key, value:
            Tensors with shape ``[B, Tk, D]``.
        key_log_weights:
            Optional absolute or relative log weights with shape ``[B, Tk]``.
            A per-row constant shift has no effect on the softmax, so callers
            may pass stabilized log weights as long as mass offsets are restored
            for explicit finite-measure reductions elsewhere.
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        q = self.q_proj(query).view(B, Tq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if key_log_weights is not None:
            if key_log_weights.shape != (B, Tk):
                raise ValueError(
                    f"key_log_weights must have shape {(B, Tk)}, got {tuple(key_log_weights.shape)}"
                )
            key_log_weights32 = key_log_weights.float()
            stable = key_log_weights32 - torch.logsumexp(key_log_weights32, dim=-1, keepdim=True)
            logits = logits + self.mass_attention_temperature * stable.to(logits.dtype)[:, None, None, :]

        weights = torch.softmax(logits, dim=-1)
        entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(dim=-1)
        self.last_attention_entropy = entropy.mean().detach()
        self.last_effective_keys = entropy.exp().mean().detach()
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.dim)
        return self.out_proj(out)


class FeedForwardBlock(nn.Module):
    """Transformer feed-forward block with pre-normalization."""

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden_dim or 4 * dim)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class MassBiasedSelfAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention with additive key mass bias."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MassBiasedCrossAttention(
            dim=dim,
            heads=heads,
            dropout=dropout,
            mass_attention_temperature=mass_attention_temperature,
        )
        self.ff = FeedForwardBlock(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, key_log_weights: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(
            self.norm(x),
            x,
            x,
            key_log_weights=key_log_weights,
        )
        return self.ff(x)


class InducedSetAttentionBlock(nn.Module):
    """Set Transformer-style particle block using learned inducing tokens.

    The block is permutation equivariant over input particles because it uses no
    positional encodings and broadcasts the same learned inducing tokens to each
    set.  Mass bias is applied only when inducing tokens read from particles;
    final finite-measure summaries should still use explicit log-space
    reductions.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        n_inducing: int = 16,
        layers: int = 2,
        dropout: float = 0.0,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if n_inducing < 1:
            raise ValueError("n_inducing must be >= 1")
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.inducing = nn.Parameter(torch.randn(n_inducing, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ind_norm": nn.LayerNorm(dim),
                        "x_norm": nn.LayerNorm(dim),
                        "ind_attn": MassBiasedCrossAttention(
                            dim,
                            heads=heads,
                            dropout=dropout,
                            mass_attention_temperature=mass_attention_temperature,
                        ),
                        "x_attn": MassBiasedCrossAttention(
                            dim,
                            heads=heads,
                            dropout=dropout,
                            mass_attention_temperature=mass_attention_temperature,
                        ),
                        "ind_ff": FeedForwardBlock(dim, dropout=dropout),
                        "x_ff": FeedForwardBlock(dim, dropout=dropout),
                    }
                )
            )

    def forward(self, x: torch.Tensor, *, key_log_weights: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        inducing = self.inducing.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            inducing = inducing + layer["ind_attn"](
                layer["ind_norm"](inducing),
                x,
                x,
                key_log_weights=key_log_weights,
            )
            inducing = layer["ind_ff"](inducing)
            x = x + layer["x_attn"](layer["x_norm"](x), inducing, inducing)
            x = layer["x_ff"](x)
        return x


__all__ = [
    "FeedForwardBlock",
    "InducedSetAttentionBlock",
    "MassBiasedCrossAttention",
    "MassBiasedSelfAttentionBlock",
]
