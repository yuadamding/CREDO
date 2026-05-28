"""Mass-aware transformer context aggregation for CREDO dynamics."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .context import ContextState, ProgramEncoder
from .transformer_blocks import InducedSetAttentionBlock


class MassAwareTransformerContextAggregator(nn.Module):
    """Hierarchical set transformer that preserves CREDO context semantics.

    The transformer enriches mediator features, but finite-measure quantities
    are still computed by explicit absolute-weight reductions:

    - ``mass_g = exp(log_m0 + logsumexp(logw_g))``
    - ``freq_g = softmax(log_m0 + logsumexp(logw_g))``
    - ``q`` and ``s`` are weighted with within-group normalized absolute mass
      and between-group finite-measure frequency.
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        n_programs: int,
        mediator_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        token_dim: int = 128,
        n_heads: int = 4,
        n_within_layers: int = 2,
        n_cross_layers: int = 2,
        n_inducing: int = 16,
        dropout: float = 0.05,
        fixed_program_centroids: Optional[torch.Tensor] = None,
        program_assignment_scale: float = 1.0,
        activation_checkpointing: bool = False,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if context_dim != n_programs + mediator_dim:
            raise ValueError(
                "MassAwareTransformerContextAggregator currently returns identity "
                f"contexts, so context_dim must equal n_programs + mediator_dim "
                f"({context_dim} != {n_programs + mediator_dim})."
            )
        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.mediator_dim = int(mediator_dim)
        self.context_dim = int(context_dim)
        self.activation_checkpointing = bool(activation_checkpointing)

        self.program_encoder = ProgramEncoder(
            latent_dim=latent_dim,
            n_programs=n_programs,
            mediator_dim=mediator_dim,
            hidden_dim=hidden_dim,
            fixed_centroids=fixed_program_centroids,
            assignment_scale=program_assignment_scale,
            activation_checkpointing=activation_checkpointing,
        )
        self.n_programs = self.program_encoder.n_programs
        token_in_dim = latent_dim + embedding_dim + 4
        self.token_in = nn.Sequential(
            nn.Linear(token_in_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        self.within = InducedSetAttentionBlock(
            dim=token_dim,
            heads=n_heads,
            n_inducing=n_inducing,
            layers=n_within_layers,
            dropout=dropout,
            mass_attention_temperature=mass_attention_temperature,
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross = nn.TransformerEncoder(cross_layer, num_layers=n_cross_layers)
        self.group_to_particle = nn.Linear(token_dim, token_dim)
        self.phi_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, mediator_dim),
            nn.Tanh(),
        )

    def encode_particles(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode particles into state programs and mediator features."""
        eta = self.program_encoder.eta(z)
        phi = self.program_encoder.phi(z)
        return eta, phi

    def forward(
        self,
        z: torch.Tensor,
        logw: torch.Tensor,
        a: torch.Tensor,
        log_m0: torch.Tensor,
        tau: torch.Tensor | float | None = None,
    ) -> ContextState:
        if z.ndim != 3:
            raise ValueError(f"z must have shape [G, N, d], got {tuple(z.shape)}")
        if logw.shape != z.shape[:2]:
            raise ValueError("logw must have shape [G, N]")
        if a.shape[0] != z.shape[0] or a.shape[1] != self.embedding_dim:
            raise ValueError(
                f"a must have shape [G, {self.embedding_dim}], got {tuple(a.shape)}"
            )
        if log_m0.shape[0] != z.shape[0]:
            raise ValueError("log_m0 must have shape [G]")

        G, N, _ = z.shape
        logw_abs = log_m0[:, None] + logw
        log_m_g = torch.logsumexp(logw_abs, dim=1)
        freq_g = torch.softmax(log_m_g, dim=0)
        mass_g = torch.exp(log_m_g)
        alpha_gi = torch.softmax(logw_abs, dim=1)

        eta = self.program_encoder.eta(z)

        a_tok = a[:, None, :].expand(G, N, -1)
        logw_centered = (logw_abs - log_m_g[:, None])[:, :, None]
        logm_centered = (log_m_g - log_m_g.mean()).view(G, 1, 1).expand(G, N, 1)
        freq_tok = freq_g.view(G, 1, 1).expand(G, N, 1)
        if tau is None:
            tau_tensor = torch.zeros((), dtype=z.dtype, device=z.device)
        else:
            tau_tensor = torch.as_tensor(tau, dtype=z.dtype, device=z.device).reshape(())
        tau_tok = tau_tensor.expand(G, N).unsqueeze(-1)

        tokens = torch.cat([z, a_tok, logw_centered, logm_centered, freq_tok, tau_tok], dim=-1)
        h_particles = self.token_in(tokens)
        h_particles = self.within(h_particles, key_log_weights=logw_abs)

        h_g = (alpha_gi[..., None] * h_particles).sum(dim=1)
        h_g_cross = self.cross(h_g.unsqueeze(0)).squeeze(0)
        h_particles = h_particles + self.group_to_particle(h_g_cross)[:, None, :]

        phi = self.phi_head(h_particles)
        q_g = (alpha_gi[..., None] * eta).sum(dim=1)
        s_g = (alpha_gi[..., None] * phi).sum(dim=1)
        q = (freq_g[:, None] * q_g).sum(dim=0)
        s = (freq_g[:, None] * s_g).sum(dim=0)
        context = torch.cat([q, s], dim=-1)

        return ContextState(q=q, s=s, context=context, mass_g=mass_g, freq_g=freq_g)


__all__ = ["MassAwareTransformerContextAggregator"]
