"""Coefficient networks: drift, diffusion (diagonal), and growth.

Zero-embedding anchoring is structural:
    v_g(z, tau, c) = beta_v(u) + B_v(u) @ a_g
    sigma_g(z, tau, c) = softplus(beta_sigma(u) + B_sigma(u) @ a_g) + sigma_min
    r_g(z, tau, c) = r_max * tanh(beta_r(u) + B_r(u) @ a_g + b_g + Phi(...))

Whenever a perturbation has `a_g = 0`, the perturbation modulation terms vanish
exactly. Under control anchoring, controls use this zero embedding; under
control-free ablations they may instead learn nonzero embeddings.

Common input:  u = [z, gamma(tau), c_tau]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TimeEmbedding
from .ecology import EcologicalPayoff


@dataclass
class Coefficients:
    """Output of CoefficientNetworks.forward()."""
    drift: torch.Tensor      # [G, N, d]
    sigma_diag: torch.Tensor  # [G, N, d]  diagonal diffusion std
    growth: torch.Tensor     # [G, N]


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d_in = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d_in, hidden_dim), nn.Tanh()]
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, out_dim))
    return nn.Sequential(*layers)


class ControlAnchoredFieldHead(nn.Module):
    """Single coefficient field with baseline + perturbation modulation.

    baseline_net(u)  -> [out_dim]
    modulation_net(u) -> [out_dim, embedding_dim]   (matrix B)
    output = baseline + B @ a_g
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.baseline_net = _mlp(input_dim, out_dim, hidden_dim, depth)
        self.modulation_net = _mlp(input_dim, out_dim * embedding_dim, hidden_dim, depth)
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

    def forward(self, u: torch.Tensor, a_g: torch.Tensor) -> torch.Tensor:
        """
        u: [G, N, input_dim]
        a_g: [G, r]  embeddings; anchored controls have a_g = 0 exactly

        Returns [G, N, out_dim].
        """
        G, N, _ = u.shape
        baseline = self.baseline_net(u)                        # [G, N, out_dim]
        B_flat = self.modulation_net(u)                         # [G, N, out_dim * r]
        B = B_flat.view(G, N, self.out_dim, self.embedding_dim)  # [G, N, out_dim, r]
        # a_g: [G, r]  -> contract over r
        modulation = torch.einsum("gnor, gr -> gno", B, a_g)   # [G, N, out_dim]
        return baseline + modulation  # [G, N, out_dim]


class CoefficientNetworks(nn.Module):
    """Full coefficient network for drift, diffusion, and growth.

    Parameters
    ----------
    latent_dim: d
    embedding_dim: r
    context_dim: C
    hidden_dim, depth: MLP architecture
    n_time_freqs: number of Fourier frequencies for time embedding
    sigma_min: minimum diagonal diffusion std
    r_max: maximum absolute growth rate
    n_programs: K (for ecological term)
    n_payoff_ranks: low-rank ecological payoff
    ecological_growth: whether to include the ecological payoff in growth
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        n_time_freqs: int = 4,
        sigma_min: float = 1e-3,
        r_max: float = 3.0,
        n_programs: int = 8,
        n_payoff_ranks: int = 4,
        ecological_growth: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.sigma_min = sigma_min
        self.r_max = r_max
        self.ecological_growth = ecological_growth
        self.n_programs = n_programs

        self.time_embed = TimeEmbedding(n_frequencies=n_time_freqs)
        time_dim = self.time_embed.output_dim
        input_dim = latent_dim + time_dim + context_dim

        self.drift_head = ControlAnchoredFieldHead(
            input_dim, latent_dim, embedding_dim, hidden_dim, depth)
        self.sigma_head = ControlAnchoredFieldHead(
            input_dim, latent_dim, embedding_dim, hidden_dim, depth)
        # Growth head outputs scalar per particle
        self.growth_head = ControlAnchoredFieldHead(
            input_dim, 1, embedding_dim, hidden_dim, depth)
        # Per-perturbation growth offset b_g (scalar, learned)
        self.growth_offset = nn.Embedding(1, 1)  # placeholder; replaced by pert-specific
        # Actually implement as a parameter indexed by non-control perturbation
        # We will handle b_g as part of the embedding store or a separate parameter dict.
        # For simplicity, make it part of the growth_head modulation (already handled by B_r).

        if ecological_growth:
            self.ecology = EcologicalPayoff(n_programs, embedding_dim, n_payoff_ranks)
        else:
            self.ecology = None

    def _common_input(
        self,
        z: torch.Tensor,      # [G, N, d]
        tau: torch.Tensor,    # scalar or [1]
        context: torch.Tensor,  # [C]
    ) -> torch.Tensor:
        """Assemble u = [z, gamma(tau), c_tau] with shape [G, N, input_dim]."""
        G, N, d = z.shape
        tau_scalar = tau.reshape(1)
        gamma = self.time_embed(tau_scalar).squeeze(0)   # [time_dim]
        ctx_expand = context.unsqueeze(0).unsqueeze(0).expand(G, N, -1)   # [G, N, C]
        gamma_expand = gamma.unsqueeze(0).unsqueeze(0).expand(G, N, -1)   # [G, N, time_dim]
        return torch.cat([z, gamma_expand, ctx_expand], dim=-1)  # [G, N, input_dim]

    def forward(
        self,
        z: torch.Tensor,       # [G, N, d]
        tau: torch.Tensor,     # scalar
        context: torch.Tensor, # [C]
        a: torch.Tensor,       # [G, r]
        eta_z: Optional[torch.Tensor] = None,  # [G, N, K] for ecology
        q: Optional[torch.Tensor] = None,      # [K] for ecology
        s: Optional[torch.Tensor] = None,      # [L] for ecology (optional)
    ) -> Coefficients:
        u = self._common_input(z, tau, context)   # [G, N, input_dim]

        drift = self.drift_head(u, a)             # [G, N, d]
        sigma_raw = self.sigma_head(u, a)         # [G, N, d]
        sigma_diag = F.softplus(sigma_raw) + self.sigma_min  # [G, N, d]

        growth_raw = self.growth_head(u, a).squeeze(-1)  # [G, N]

        if self.ecological_growth and self.ecology is not None and eta_z is not None and q is not None:
            # a: [G, r], eta_z: [G, N, K] -- ecology.forward handles explicit dims
            phi = self.ecology(eta_z, a, q, s)  # [G, N]
            growth_raw = growth_raw + phi

        growth = self.r_max * torch.tanh(growth_raw)  # [G, N]

        return Coefficients(drift=drift, sigma_diag=sigma_diag, growth=growth)

    def regularization(self) -> torch.Tensor:
        """Action penalty: L2 on net weights (weight decay handles network params)."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.ecological_growth and self.ecology is not None:
            reg = reg + self.ecology.regularization()
        return reg
