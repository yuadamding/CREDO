"""Unbalanced optimal transport divergence loss.

Weight convention
-----------------
All tensors passed to UOTLoss must be *absolute* log-weights:
    log_a[i] = log(actual_weight_i),  sum_i exp(log_a) = total_mass.

The trainer adds log_m0 to the relative rollout log-weights before calling.

Implementation
--------------
Decomposes UOT into two terms:
  (1) geometry  : debiased Sinkhorn divergence on normalised probability measures
  (2) mass      : tau * (log|mu| - log|nu|)^2

Uses geomloss.SamplesLoss for (1) when available; falls back to a correct
log-domain balanced Sinkhorn otherwise.
"""
from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pairwise_sq_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[n, d] x [m, d] -> [n, m]  squared Euclidean distances."""
    xx = (x ** 2).sum(-1, keepdim=True)     # [n, 1]
    yy = (y ** 2).sum(-1, keepdim=True).T   # [1, m]
    xy = x @ y.T                             # [n, m]
    return (xx + yy - 2 * xy).clamp(min=0)


def _balanced_sinkhorn_cost(
    log_a: torch.Tensor,  # [n]  log probability weights
    log_b: torch.Tensor,  # [m]  log probability weights
    C: torch.Tensor,      # [n, m]  cost
    eps: float,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Log-domain balanced Sinkhorn, returns scalar transport cost.

    Sinkhorn updates (Schmitzer 2019 stabilised form):
        log_v[j] = log_b[j] - logsumexp_i(-C[i,j]/eps + log_u[i])
        log_u[i] = log_a[i] - logsumexp_j(-C[i,j]/eps + log_v[j])
    Transport plan: log_pi[i,j] = log_u[i] + log_v[j] - C[i,j]/eps
    """
    M = -C / eps  # [n, m]
    log_u = log_a.clone()
    log_v = log_b.clone()

    for _ in range(max_iter):
        log_u_prev = log_u
        log_v = log_b - torch.logsumexp(M + log_u.unsqueeze(1), dim=0)   # [m]
        log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)   # [n]
        if (log_u - log_u_prev).abs().max().item() < tol:
            break

    log_pi = log_u.unsqueeze(1) + log_v.unsqueeze(0) + M  # [n, m]
    # Clamp to avoid -inf * 0 in cost
    pi = log_pi.exp()
    return (pi * C).sum()


def sinkhorn_divergence_normalized(
    x: torch.Tensor, a: torch.Tensor,   # [n, d], [n] probability weights
    y: torch.Tensor, b: torch.Tensor,   # [m, d], [m] probability weights
    eps: float = 0.1,
    max_iter: int = 200,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence on probability measures.  Always >= 0."""
    log_a = torch.log(a.clamp(min=1e-30))
    log_b = torch.log(b.clamp(min=1e-30))
    C_xy = _pairwise_sq_euclidean(x, y)
    C_xx = _pairwise_sq_euclidean(x, x)
    C_yy = _pairwise_sq_euclidean(y, y)
    ot_xy = _balanced_sinkhorn_cost(log_a, log_b, C_xy, eps, max_iter)
    ot_xx = _balanced_sinkhorn_cost(log_a, log_a, C_xx, eps, max_iter)
    ot_yy = _balanced_sinkhorn_cost(log_b, log_b, C_yy, eps, max_iter)
    return (ot_xy - 0.5 * ot_xx - 0.5 * ot_yy).clamp(min=0)


def sinkhorn_divergence(
    x: torch.Tensor, log_a: torch.Tensor,   # [n, d], [n] absolute log-weights
    y: torch.Tensor, log_b: torch.Tensor,   # [m, d], [m] absolute log-weights
    eps: float = 0.1,
    tau: float = 1.0,
    max_iter: int = 200,
) -> torch.Tensor:
    """UOT proxy: debiased geometry on normalised measures + squared log-mass penalty."""
    a_norm = torch.softmax(log_a, dim=0)  # [n]  probability weights
    b_norm = torch.softmax(log_b, dim=0)  # [m]
    geom = sinkhorn_divergence_normalized(x, a_norm, y, b_norm, eps=eps, max_iter=max_iter)
    log_mass_pred = torch.logsumexp(log_a, dim=0)
    log_mass_tgt = torch.logsumexp(log_b, dim=0)
    mass_pen = tau * (log_mass_pred - log_mass_tgt) ** 2
    return geom + mass_pen


def sinkhorn_divergence_components(
    x: torch.Tensor, log_a: torch.Tensor,
    y: torch.Tensor, log_b: torch.Tensor,
    eps: float = 0.1,
    tau: float = 1.0,
    max_iter: int = 200,
) -> Dict[str, torch.Tensor]:
    """Return geometry, log-mass penalty, and total UOT-proxy components."""
    a_norm = torch.softmax(log_a, dim=0)
    b_norm = torch.softmax(log_b, dim=0)
    geom = sinkhorn_divergence_normalized(x, a_norm, y, b_norm, eps=eps, max_iter=max_iter)
    log_mass_pred = torch.logsumexp(log_a, dim=0)
    log_mass_tgt = torch.logsumexp(log_b, dim=0)
    mass = tau * (log_mass_pred - log_mass_tgt) ** 2
    return {"geom": geom, "mass": mass, "total": geom + mass}


# ---------------------------------------------------------------------------
# UOTLoss module
# ---------------------------------------------------------------------------

class UOTLoss(nn.Module):
    """Endpoint UOT loss over absolute finite measures.

    Decomposes as geometry (Sinkhorn divergence on normalised measures) +
    mass discrepancy penalty.  Uses geomloss when available.
    """

    def __init__(
        self,
        eps: float = 0.1,
        tau: float = 1.0,
        max_iter: int = 200,
        use_geomloss: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self._geomloss_fn = None

        if use_geomloss:
            try:
                from geomloss import SamplesLoss
                self._geomloss_fn = SamplesLoss(
                    loss="sinkhorn",
                    p=2,
                    blur=eps ** 0.5,
                    debias=True,
                    backend="tensorized",
                )
            except ImportError:
                pass

    def _geometry(
        self,
        x: torch.Tensor, a_norm: torch.Tensor,
        y: torch.Tensor, b_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Debiased Sinkhorn geometry on probability measures."""
        if self._geomloss_fn is not None:
            return self._geomloss_fn(
                a_norm.unsqueeze(0), x.unsqueeze(0),
                b_norm.unsqueeze(0), y.unsqueeze(0),
            ).squeeze()
        return sinkhorn_divergence_normalized(
            x, a_norm, y, b_norm, eps=self.eps, max_iter=self.max_iter)

    def forward(
        self,
        pred_z: torch.Tensor,              # [G, N, d]
        pred_logw_abs: torch.Tensor,       # [G, N]  absolute log-weights
        target_support: Dict[Hashable, torch.Tensor],  # key -> [m, d]
        target_logw: Dict[Hashable, torch.Tensor],     # key -> [m] absolute log-weights
        perturbation_ids: list,
        weights: Optional[Dict[Hashable, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[Hashable, torch.Tensor]]:
        total, components = self.component_dict(
            pred_z=pred_z,
            pred_logw_abs=pred_logw_abs,
            target_support=target_support,
            target_logw=target_logw,
            perturbation_ids=perturbation_ids,
            weights=weights,
        )
        per_pid = {pid: values["total"] for pid, values in components.items()}
        return total, per_pid

    def component_dict(
        self,
        pred_z: torch.Tensor,
        pred_logw_abs: torch.Tensor,
        target_support: Dict[Hashable, torch.Tensor],
        target_logw: Dict[Hashable, torch.Tensor],
        perturbation_ids: list,
        weights: Optional[Dict[Hashable, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[Hashable, Dict[str, torch.Tensor]]]:
        """Return total loss and per-key geometry/mass/total components."""
        total = torch.tensor(0.0, device=pred_z.device, dtype=pred_z.dtype)
        per_pid: Dict[Hashable, Dict[str, torch.Tensor]] = {}

        for g, pid in enumerate(perturbation_ids):
            if pid not in target_support:
                continue
            x = pred_z[g]             # [N, d]
            la = pred_logw_abs[g]     # [N]  absolute
            y = target_support[pid]   # [m, d]
            lb = target_logw[pid]     # [m]  absolute

            a_norm = torch.softmax(la, dim=0)   # [N]
            b_norm = torch.softmax(lb, dim=0)   # [m]

            geom = self._geometry(x, a_norm, y, b_norm)

            log_mass_pred = torch.logsumexp(la, dim=0)
            log_mass_tgt = torch.logsumexp(lb, dim=0)
            mass_pen = self.tau * (log_mass_pred - log_mass_tgt) ** 2

            div = geom + mass_pen
            w = weights[pid] if weights else 1.0
            per_pid[pid] = {"geom": geom, "mass": mass_pen, "total": div}
            total = total + w * div

        return total, per_pid
