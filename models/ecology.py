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

        # Baseline Pq contribution from P0
        Pq_base = self.P0 @ q                             # [K]
        phi = torch.einsum("gnk, k -> gn", eta_z, Pq_base)  # [G, N]

        if self.P_pert is not None and self.actual_ranks > 0:
            r = self.actual_ranks
            a_trunc = a_g[:, :r]                          # [G, r]

            # P_m to use (optionally mediator-conditioned)
            P_m = self.P_pert                              # [r, K, K]
            if self.mediator_net is not None and s is not None:
                delta = self.mediator_net(s).reshape(r, K, K)
                P_m = P_m + delta

            # Per-perturbation Pq from P_pert: [G, K]
            # P_m[m] @ q  -> [r, K];  sum over m with a_g weights -> [G, K]
            Pm_q = torch.einsum("rkj, j -> rk", P_m, q)      # [r, K]
            Pq_pert = torch.einsum("gr, rk -> gk", a_trunc, Pm_q)  # [G, K]

            # Add perturbation contribution: [G, N]
            phi = phi + torch.einsum("gnk, gk -> gn", eta_z, Pq_pert)

        return phi  # [G, N]

    def regularization(self) -> torch.Tensor:
        """Frobenius penalty on payoff matrices."""
        loss = (self.P0 ** 2).mean()
        if self.P_pert is not None:
            loss = loss + (self.P_pert ** 2).mean()
        return loss
