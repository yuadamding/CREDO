"""Low-rank ecological payoff term for the growth channel.

Phi(z, a_g, q, s) = eta(z)^T (P_0 + sum_m a_{g,m} P_m) q

where P_m in R^{K x K} are low-rank payoff matrices.
The mediator-conditioned extension P_m = P_m(s) is optional.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EcologicalPayoff(nn.Module):
    """Low-rank ecological growth interaction term.

    Parameters
    ----------
    n_programs: K  (program dimension)
    embedding_dim: r  (perturbation embedding dimension)
    n_ranks: number of payoff matrices beyond the baseline P_0
    mediator_dim: L  (if > 0, condition P_m on mediator summary s)
    """

    def __init__(
        self,
        n_programs: int,
        embedding_dim: int,
        n_ranks: int = 4,
        mediator_dim: int = 0,
    ) -> None:
        super().__init__()
        self.n_programs = n_programs
        self.embedding_dim = embedding_dim
        self.n_ranks = n_ranks
        self.mediator_dim = mediator_dim

        # Baseline payoff P_0: [K, K]
        self.P0 = nn.Parameter(torch.zeros(n_programs, n_programs))
        nn.init.xavier_uniform_(self.P0)

        # Perturbation payoffs P_1, ..., P_r: [r, K, K]
        # (we use n_ranks = embedding_dim by default, but can differ)
        actual_ranks = min(n_ranks, embedding_dim)
        self.actual_ranks = actual_ranks
        if actual_ranks > 0:
            self.P_pert = nn.Parameter(torch.zeros(actual_ranks, n_programs, n_programs))
            nn.init.normal_(self.P_pert, std=0.01)
        else:
            self.register_parameter("P_pert", None)

        # Optional mediator conditioning: map s -> P_m (small MLP)
        if mediator_dim > 0 and actual_ranks > 0:
            self.mediator_net = nn.Sequential(
                nn.Linear(mediator_dim, n_programs * n_programs * actual_ranks),
                nn.Tanh(),
            )
        else:
            self.mediator_net = None

    def forward(
        self,
        eta_z: torch.Tensor,   # [G, N, K]  program scores at particle locations
        a_g: torch.Tensor,     # [G, r]     per-perturbation embeddings (controls = 0)
        q: torch.Tensor,       # [K]        population program composition
        s: Optional[torch.Tensor] = None,   # [L] mediator summary (optional)
    ) -> torch.Tensor:
        """Return Phi(z, a_g, q) of shape [G, N].

        Phi[g, n] = eta_z[g, n, :] . P_eff[g] @ q
        P_eff[g] = P0 + sum_m a_g[g, m] * P_m
        """
        K = self.n_programs

        if q.ndim == 1:
            Pq_base = self.P0 @ q
            phi = torch.einsum("gnk,k->gn", eta_z, Pq_base)
        elif q.ndim == 2 and q.shape[0] == eta_z.shape[0]:
            Pq_base = torch.einsum("kj,gj->gk", self.P0, q)
            phi = torch.einsum("gnk,gk->gn", eta_z, Pq_base)
        else:
            raise ValueError("q must have shape [K] or [G, K].")

        if self.P_pert is not None and self.actual_ranks > 0:
            r = self.actual_ranks
            a_trunc = a_g[:, :r]                          # [G, r]

            # Per-perturbation Pq from P_pert: [G, K]
            # P_m[m] @ q  -> [r, K];  sum over m with a_g weights -> [G, K]
            if q.ndim == 1:
                P_m = self.P_pert
                if self.mediator_net is not None and s is not None:
                    if s.ndim != 1:
                        raise ValueError("Global q requires global s with shape [L].")
                    P_m = P_m + self.mediator_net(s).reshape(r, K, K)
                Pm_q = torch.einsum("rkj,j->rk", P_m, q)
                Pq_pert = torch.einsum("gr,rk->gk", a_trunc, Pm_q)
            else:
                P_m = self.P_pert.unsqueeze(0).expand(q.shape[0], -1, -1, -1)
                if self.mediator_net is not None and s is not None:
                    if s.ndim != 2 or s.shape[0] != q.shape[0]:
                        raise ValueError("Grouped q requires grouped s with shape [G, L].")
                    P_m = P_m + self.mediator_net(s).reshape(q.shape[0], r, K, K)
                Pm_q = torch.einsum("grkj,gj->grk", P_m, q)
                Pq_pert = torch.einsum("gr,grk->gk", a_trunc, Pm_q)

            # Add perturbation contribution: [G, N]
            phi = phi + torch.einsum("gnk, gk -> gn", eta_z, Pq_pert)

        return phi  # [G, N]

    def regularization(self) -> torch.Tensor:
        """Frobenius penalty on payoff matrices."""
        loss = (self.P0 ** 2).mean()
        if self.P_pert is not None:
            loss = loss + (self.P_pert ** 2).mean()
        return loss
