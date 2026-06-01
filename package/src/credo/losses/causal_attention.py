"""Regularizers for causal ecological attention diagnostics."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def control_edge_null_loss(edge_scores_gm: torch.Tensor, is_control_g: torch.Tensor) -> torch.Tensor:
    """Penalize residual-to-mediator edges for controls."""
    mask = is_control_g.to(device=edge_scores_gm.device, dtype=torch.bool)
    if mask.numel() != edge_scores_gm.shape[0]:
        raise ValueError("is_control_g length must match edge_scores_gm group dimension")
    if not mask.any():
        return edge_scores_gm.new_tensor(0.0)
    return edge_scores_gm[mask].square().mean()


def guide_concordance_loss(edge_scores_gm: torch.Tensor, target_ids: list[str]) -> torch.Tensor:
    """Encourage guides for the same target to use similar mediator profiles."""
    if len(target_ids) != edge_scores_gm.shape[0]:
        raise ValueError("target_ids length must match edge_scores_gm group dimension")
    loss = edge_scores_gm.new_tensor(0.0)
    n_terms = 0
    for target in sorted(set(target_ids)):
        idx = [i for i, value in enumerate(target_ids) if value == target]
        if len(idx) < 2:
            continue
        values = edge_scores_gm[idx]
        loss = loss + (values - values.mean(dim=0, keepdim=True)).square().mean()
        n_terms += 1
    if n_terms == 0:
        return loss
    return loss / float(n_terms)


def edge_sparsity_loss(edge_scores_gm: torch.Tensor) -> torch.Tensor:
    """Mild sparsity pressure on mediator edge scores."""
    return edge_scores_gm.mean()


def edge_entropy_loss(edge_scores_gm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of per-group mediator usage; minimize for sparse profiles."""
    p = edge_scores_gm / edge_scores_gm.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(p * p.clamp_min(eps).log()).sum(dim=1).mean()


def mediator_orthogonality_loss(mediator_tokens: torch.Tensor) -> torch.Tensor:
    """Prevent learned mediator slots from collapsing to the same vector."""
    if mediator_tokens.ndim != 2:
        raise ValueError("mediator_tokens must have shape [M, H]")
    med = F.normalize(mediator_tokens, dim=-1)
    gram = med @ med.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - eye).square().mean()


def context_smoothness_loss(context_steps: torch.Tensor, tau_steps: torch.Tensor) -> torch.Tensor:
    """Squared context variation normalized by rollout time step."""
    if context_steps.shape[0] < 2:
        return context_steps.new_tensor(0.0)
    dt = (tau_steps[1:context_steps.shape[0]] - tau_steps[:context_steps.shape[0] - 1]).abs()
    while dt.ndim < context_steps.ndim:
        dt = dt.unsqueeze(-1)
    diffs = context_steps[1:] - context_steps[:-1]
    return (diffs.square() / dt.clamp_min(1e-8)).mean()


__all__ = [
    "context_smoothness_loss",
    "control_edge_null_loss",
    "edge_entropy_loss",
    "edge_sparsity_loss",
    "guide_concordance_loss",
    "mediator_orthogonality_loss",
]
