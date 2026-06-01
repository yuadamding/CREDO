"""Intervention objects for causal ecological attention."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class CausalAttentionIntervention:
    """Declarative edge/mediator interventions for CEA rollouts.

    The intervention acts on group-to-mediator edge logits before sigmoid
    conversion.  It is intentionally narrow: CREDO counterfactual semantics are
    still implemented by same-start/same-noise rollouts, while this object
    specifies which causal-attention edges are removed or clamped.
    """

    protocol: Literal[
        "none",
        "ablate_mediators",
        "ablate_edges",
        "clamp_edges",
    ] = "none"
    ablate_mediator_ids: list[int] = field(default_factory=list)
    ablate_group_mediator_edges: list[tuple[int, int]] = field(default_factory=list)
    clamp_edge_scores_gm: torch.Tensor | None = None

    def apply_edge_logits(self, edge_logits: torch.Tensor) -> torch.Tensor:
        """Return intervention-modified logits with shape ``[G, M]``."""
        if self.protocol == "none":
            return edge_logits

        out = edge_logits.clone()
        floor = torch.finfo(out.dtype).min

        if self.ablate_mediator_ids:
            mediator_ids = [int(idx) for idx in self.ablate_mediator_ids]
            out[:, mediator_ids] = floor

        for group_idx, mediator_idx in self.ablate_group_mediator_edges:
            out[int(group_idx), int(mediator_idx)] = floor

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


__all__ = ["CausalAttentionIntervention"]
