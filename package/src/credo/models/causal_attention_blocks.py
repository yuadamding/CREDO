"""Graph-masked, mass-biased attention blocks for CREDO-CEA."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MassGraphMaskedCrossAttention(nn.Module):
    """Multi-head cross-attention with mass bias and intervention masks."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.05,
        mass_attention_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.mass_attention_temperature = float(mass_attention_temperature)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
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
        graph_mask: torch.Tensor | None = None,
        do_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply masked attention.

        Parameters
        ----------
        query, key, value:
            Tensors with shapes ``[B, Q, H]`` and ``[B, K, H]``.
        key_log_weights:
            Optional log finite-measure weights with shape ``[B, K]``.  These
            bias attention only; authoritative mass summaries must still use
            explicit log-space reductions in the context module.
        graph_mask:
            Optional boolean mask with shape ``[B, Q, K]`` where True means an
            edge is allowed by the causal graph.
        do_mask:
            Optional boolean mask with shape ``[B, Q, K]`` where True means an
            edge is removed by an explicit intervention.
        """
        B, Q, H = query.shape
        _, K, _ = key.shape
        q = self.q_proj(query).view(B, Q, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, K, self.heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_log_weights is not None:
            if key_log_weights.shape != (B, K):
                raise ValueError(
                    f"key_log_weights must have shape {(B, K)}, got {tuple(key_log_weights.shape)}"
                )
            lw = key_log_weights.float()
            stable = lw - torch.logsumexp(lw, dim=-1, keepdim=True)
            logits = logits + self.mass_attention_temperature * stable.to(logits.dtype)[:, None, None, :]

        if graph_mask is not None:
            if graph_mask.shape != (B, Q, K):
                raise ValueError(f"graph_mask must have shape {(B, Q, K)}, got {tuple(graph_mask.shape)}")
            logits = logits.masked_fill(~graph_mask[:, None, :, :], torch.finfo(logits.dtype).min)

        if do_mask is not None:
            if do_mask.shape != (B, Q, K):
                raise ValueError(f"do_mask must have shape {(B, Q, K)}, got {tuple(do_mask.shape)}")
            logits = logits.masked_fill(do_mask[:, None, :, :], torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        entropy = -(attn.clamp_min(1e-30) * attn.clamp_min(1e-30).log()).sum(dim=-1)
        self.last_attention_entropy = entropy.mean().detach()
        self.last_effective_keys = entropy.exp().mean().detach()

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, H)
        out = self.out_proj(out)
        if return_attention:
            return out, attn
        return out


__all__ = ["MassGraphMaskedCrossAttention"]
