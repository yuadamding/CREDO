"""Intervention objects for causal ecological attention."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class CausalAttentionIntervention:
    """Declarative edge/mediator interventions for CEA rollouts.

    The intervention acts on group-to-mediator edge logits before sigmoid
    conversion.  Residual-edge protocols remove only perturbation residual
    modulation, while effective-edge protocols remove the full mediator gate.
    CREDO counterfactual semantics are still implemented by same-start/same-noise
    rollouts; this object specifies which causal-attention edges are modified.
    """

    protocol: Literal[
        "none",
        "ablate_mediators",
        "ablate_edges",
        "ablate_residual_edges",
        "ablate_effective_edges",
        "ablate_baseline_edges",
        "clamp_edges",
    ] = "none"
    ablate_mediator_ids: list[int] = field(default_factory=list)
    ablate_group_mediator_edges: list[tuple[int, int]] = field(default_factory=list)
    clamp_edge_scores_gm: torch.Tensor | None = None

    def _apply_floor(self, logits: torch.Tensor, *, floor_value: float | None = None) -> torch.Tensor:
        out = logits.clone()
        floor = torch.finfo(out.dtype).min if floor_value is None else float(floor_value)
        if self.ablate_mediator_ids:
            mediator_ids = [int(idx) for idx in self.ablate_mediator_ids]
            out[:, mediator_ids] = floor
        for group_idx, mediator_idx in self.ablate_group_mediator_edges:
            out[int(group_idx), int(mediator_idx)] = floor
        return out

    def _apply_zero(self, logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        if self.ablate_mediator_ids:
            mediator_ids = [int(idx) for idx in self.ablate_mediator_ids]
            out[:, mediator_ids] = 0.0
        for group_idx, mediator_idx in self.ablate_group_mediator_edges:
            out[int(group_idx), int(mediator_idx)] = 0.0
        return out

    def apply_residual_logits(self, residual_logits: torch.Tensor) -> torch.Tensor:
        """Zero selected perturbation-residual edge logits."""
        if self.protocol != "ablate_residual_edges":
            return residual_logits
        return self._apply_zero(residual_logits)

    def apply_baseline_logits(self, baseline_logits: torch.Tensor) -> torch.Tensor:
        """Remove selected baseline ecological edge logits."""
        if self.protocol != "ablate_baseline_edges":
            return baseline_logits
        return self._apply_floor(baseline_logits)

    def apply_effective_logits(self, edge_logits: torch.Tensor) -> torch.Tensor:
        """Return intervention-modified effective logits with shape ``[G, M]``."""
        if self.protocol not in {"ablate_mediators", "ablate_edges", "ablate_effective_edges", "clamp_edges"}:
            return edge_logits

        out = self._apply_floor(edge_logits)

        if self.clamp_edge_scores_gm is not None:
            eps = 1e-6
            scores = self.clamp_edge_scores_gm.to(device=out.device, dtype=out.dtype)
            if scores.shape != out.shape:
                raise ValueError(
                    "clamp_edge_scores_gm must have shape "
                    f"{tuple(out.shape)}, got {tuple(scores.shape)}"
                )
            out = torch.logit(scores.clamp(eps, 1.0 - eps))

        return out

    def apply_edge_logits(self, edge_logits: torch.Tensor) -> torch.Tensor:
        """Backward-compatible alias for effective-edge interventions."""
        return self.apply_effective_logits(edge_logits)

    def mediator_to_group_graph_mask(self, n_groups: int, n_mediators: int, device: torch.device) -> torch.Tensor | None:
        """Optional graph mask for mediator-to-group attention.

        The mask is only returned when every group keeps at least one mediator.
        All-edge ablations are handled exactly by value gates instead of asking
        attention to softmax over an empty key set.
        """
        if self.protocol not in {"ablate_mediators", "ablate_edges", "ablate_effective_edges"}:
            return None
        mask = torch.ones(n_groups, 1, n_mediators, dtype=torch.bool, device=device)
        if self.ablate_mediator_ids:
            mediator_ids = [int(idx) for idx in self.ablate_mediator_ids]
            mask[:, :, mediator_ids] = False
        for group_idx, mediator_idx in self.ablate_group_mediator_edges:
            mask[int(group_idx), :, int(mediator_idx)] = False
        if not mask.any(dim=-1).all():
            return None
        return mask


__all__ = ["CausalAttentionIntervention"]
