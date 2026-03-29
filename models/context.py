"""Context aggregation: observation-driven mean-field summaries from particles.

The ContextAggregator takes the current particle cloud and returns:
  - q: population-level latent-factor composition [K]
  - s: mediator summary [L]
  - context: context vector [C] fed into coefficient networks
  - mass_g: per-perturbation total mass [G]
  - freq_g: per-perturbation relative frequency [G]

All mass computations assume absolute log-weights and use log-space reductions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ContextState:
    """Output of ContextAggregator.forward()."""
    q: torch.Tensor         # [K]  latent-factor composition
    s: torch.Tensor         # [L]  mediator summary
    context: torch.Tensor  # [C]  context vector for coefficient networks
    mass_g: torch.Tensor    # [G]  per-perturbation absolute mass
    freq_g: torch.Tensor    # [G]  per-perturbation relative frequency


class ProgramEncoder(nn.Module):
    """Maps latent coordinates to latent-factor composition eta(z) in Delta^{K-1}.

    Optionally also computes mediator features phi(z) in R^L.
    """

    def __init__(
        self,
        latent_dim: int,
        n_programs: int,
        mediator_dim: int,
        hidden_dim: int = 64,
        fixed_centroids: Optional[torch.Tensor] = None,
        assignment_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if fixed_centroids is not None:
            centroids = torch.as_tensor(fixed_centroids, dtype=torch.float32)
            if centroids.ndim != 2 or centroids.shape[1] != latent_dim:
                raise ValueError(
                    "fixed_centroids must have shape [n_programs, latent_dim], "
                    f"got {tuple(centroids.shape)} with latent_dim={latent_dim}."
                )
            self.register_buffer("fixed_centroids", centroids)
            self.n_programs = int(centroids.shape[0])
        else:
            self.register_buffer("fixed_centroids", torch.empty(0, latent_dim))
            self.n_programs = n_programs
        self.mediator_dim = mediator_dim
        self.assignment_scale = float(assignment_scale)
        self.use_fixed_centroids = fixed_centroids is not None

        # Soft latent-factor map: z -> Delta^{K-1}
        if self.use_fixed_centroids:
            self.eta_net = None
        else:
            self.eta_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.n_programs),
            )

        # Mediator feature map: z -> R^L
        self.phi_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, mediator_dim),
            nn.Tanh(),
        )

    def eta(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., d] -> [..., K], softmax-normalised."""
        if self.use_fixed_centroids:
            centers = self.fixed_centroids.to(device=z.device, dtype=z.dtype)
            diff = z.unsqueeze(-2) - centers
            sq_dist = (diff ** 2).sum(dim=-1)
            return torch.softmax(-self.assignment_scale * sq_dist, dim=-1)
        return torch.softmax(self.eta_net(z), dim=-1)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., d] -> [..., L]."""
        return self.phi_net(z)


class ContextAggregator(nn.Module):
    """Compute population-level context from the current particle cloud.

    Parameters
    ----------
    latent_dim: d
    n_programs: K
    mediator_dim: L
    context_dim: C  (output dimension of the context vector)
    hidden_dim: hidden size for MLPs
    use_identity_context: if True, context = cat(q, s) directly (no learned Psi)
    """

    def __init__(
        self,
        latent_dim: int,
        n_programs: int,
        mediator_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        use_identity_context: bool = True,
        fixed_program_centroids: Optional[torch.Tensor] = None,
        program_assignment_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = ProgramEncoder(
            latent_dim,
            n_programs,
            mediator_dim,
            hidden_dim,
            fixed_centroids=fixed_program_centroids,
            assignment_scale=program_assignment_scale,
        )
        self.n_programs = self.encoder.n_programs
        self.mediator_dim = mediator_dim
        self.context_dim = context_dim
        self.use_identity_context = use_identity_context

        input_dim = self.n_programs + mediator_dim
        if not use_identity_context:
            self.psi = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, context_dim),
            )
        else:
            # identity map; context_dim must equal n_programs + mediator_dim
            assert context_dim == input_dim, (
                f"With use_identity_context=True, context_dim ({context_dim}) "
                f"must equal n_programs + mediator_dim ({input_dim})"
            )

    def forward(
        self,
        z: torch.Tensor,     # [G, N, d]
        logw: torch.Tensor,  # [G, N]  absolute log-weights
        a: torch.Tensor,     # [G, r]  perturbation embeddings (unused here, for API compat.)
        log_m0: torch.Tensor,  # [G]  log initial mass per perturbation
    ) -> ContextState:
        G, N, d = z.shape

        # --- Absolute finite-measure mass computation (log-space) ---
        # logw has shape [G, N] and represents absolute particle log-weights relative
        # to the perturbation-specific initial mass M0_g. The total mass for group g is
        # M0_g * sum_i exp(logw[g, i]).
        log_n_g = log_m0 + torch.logsumexp(logw, dim=-1)
        # freq_g
        log_n_total = torch.logsumexp(log_n_g, dim=0)
        log_freq_g = log_n_g - log_n_total
        freq_g = log_freq_g.exp()      # [G]
        mass_g = log_n_g.exp()         # [G]

        # --- Normalised within-perturbation weights ---
        log_norm_w = logw - torch.logsumexp(logw, dim=-1, keepdim=True)  # [G, N]
        norm_w = log_norm_w.exp()  # [G, N]

        # --- Per-perturbation program averages ---
        eta = self.encoder.eta(z)   # [G, N, K]
        phi = self.encoder.phi(z)   # [G, N, L]

        # mass-weighted average across perturbations
        # eta_g: [G, K]  = E_{p_g}[eta(z)]
        eta_g = (norm_w.unsqueeze(-1) * eta).sum(dim=1)   # [G, K]
        phi_g = (norm_w.unsqueeze(-1) * phi).sum(dim=1)   # [G, L]

        # population-level:  q = sum_g f_g * eta_g
        q = (freq_g.unsqueeze(-1) * eta_g).sum(dim=0)   # [K]
        s = (freq_g.unsqueeze(-1) * phi_g).sum(dim=0)   # [L]

        # --- Context vector ---
        qs = torch.cat([q, s], dim=-1)  # [K+L]
        if self.use_identity_context:
            ctx = qs
        else:
            ctx = self.psi(qs)

        return ContextState(q=q, s=s, context=ctx, mass_g=mass_g, freq_g=freq_g)
