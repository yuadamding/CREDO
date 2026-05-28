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
from torch.utils.checkpoint import checkpoint


@dataclass
class ContextState:
    """Output of ContextAggregator.forward()."""
    q: torch.Tensor         # [K]  latent-factor composition
    s: torch.Tensor         # [L]  mediator summary
    context: torch.Tensor  # [C]  context vector for coefficient networks
    mass_g: torch.Tensor    # [G]  per-perturbation absolute mass
    freq_g: torch.Tensor    # [G]  per-perturbation relative frequency
    log_mass_g: Optional[torch.Tensor] = None  # [G] log-domain absolute mass
    log_total_mass: Optional[torch.Tensor] = None  # [] log total finite-measure mass
    diagnostics: Optional["ContextDiagnostics"] = None


@dataclass
class ContextDiagnostics:
    """Optional context diagnostics for monitoring transformer ecology."""
    within_attention_entropy: Optional[torch.Tensor] = None
    group_attention_entropy: Optional[torch.Tensor] = None
    within_effective_keys: Optional[torch.Tensor] = None
    group_effective_keys: Optional[torch.Tensor] = None
    mass_attention_temperature: Optional[torch.Tensor] = None
    context_norm: Optional[torch.Tensor] = None
    q_entropy: Optional[torch.Tensor] = None
    freq_entropy: Optional[torch.Tensor] = None
    mass_log_range: Optional[torch.Tensor] = None


@dataclass
class GroupStatistics:
    """Per-perturbation summaries that can be merged across device shards."""
    log_n_g: torch.Tensor    # [G]   absolute log-mass per perturbation
    eta_g: torch.Tensor      # [G,K] within-perturbation program averages
    phi_g: torch.Tensor      # [G,L] within-perturbation mediator averages


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
        activation_checkpointing: bool = False,
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
        self.activation_checkpointing = activation_checkpointing

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
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            logits = checkpoint(self.eta_net, z, use_reentrant=False)
        else:
            logits = self.eta_net(z)
        return torch.softmax(logits, dim=-1)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., d] -> [..., L]."""
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.phi_net, z, use_reentrant=False)
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
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ProgramEncoder(
            latent_dim,
            n_programs,
            mediator_dim,
            hidden_dim,
            fixed_centroids=fixed_program_centroids,
            assignment_scale=program_assignment_scale,
            activation_checkpointing=activation_checkpointing,
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
        tau: torch.Tensor | float | None = None,  # accepted for transformer-compatible API
    ) -> ContextState:
        stats, _, _ = self.summarize_groups(z, logw, log_m0)
        return self.context_from_group_statistics(stats)

    def encode_particles(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode particles into latent programs and mediator features."""
        eta = self.encoder.eta(z)   # [G, N, K]
        phi = self.encoder.phi(z)   # [G, N, L]
        return eta, phi

    def summarize_groups(
        self,
        z: torch.Tensor,
        logw: torch.Tensor,
        log_m0: torch.Tensor,
        eta: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
    ) -> tuple[GroupStatistics, torch.Tensor, torch.Tensor]:
        """Compute per-group summaries that can be merged exactly across shards."""
        if eta is None or phi is None:
            eta, phi = self.encode_particles(z)

        log_n_g = log_m0 + torch.logsumexp(logw, dim=-1)  # [G]
        log_norm_w = logw - torch.logsumexp(logw, dim=-1, keepdim=True)  # [G, N]
        norm_w = log_norm_w.exp()  # [G, N]

        eta_g = (norm_w.unsqueeze(-1) * eta).sum(dim=1)   # [G, K]
        phi_g = (norm_w.unsqueeze(-1) * phi).sum(dim=1)   # [G, L]

        return GroupStatistics(log_n_g=log_n_g, eta_g=eta_g, phi_g=phi_g), eta, phi

    def context_from_group_statistics(self, stats: GroupStatistics) -> ContextState:
        """Build the global context from per-group summaries."""
        log_n_total = torch.logsumexp(stats.log_n_g, dim=0)
        log_freq_g = stats.log_n_g - log_n_total
        freq_g = log_freq_g.exp()      # [G]
        mass_g = torch.exp(torch.clamp(stats.log_n_g, min=-30.0, max=30.0))   # [G]

        q = (freq_g.unsqueeze(-1) * stats.eta_g).sum(dim=0)   # [K]
        s = (freq_g.unsqueeze(-1) * stats.phi_g).sum(dim=0)   # [L]

        qs = torch.cat([q, s], dim=-1)  # [K+L]
        if self.use_identity_context:
            ctx = qs
        else:
            ctx = self.psi(qs)

        return ContextState(
            q=q,
            s=s,
            context=ctx,
            mass_g=mass_g,
            freq_g=freq_g,
            log_mass_g=stats.log_n_g,
            log_total_mass=log_n_total,
        )
