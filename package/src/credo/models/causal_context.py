"""Causal ecological attention context backend for CREDO dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .causal_attention_blocks import MassGraphMaskedCrossAttention
from .context import ContextDiagnostics, ContextState, ProgramEncoder
from .interventions import CausalAttentionIntervention


@dataclass
class CausalAttentionDiagnostics(ContextDiagnostics):
    """Diagnostics for causal ecological attention."""

    state_to_mediator_entropy: Optional[torch.Tensor] = None
    local_to_global_mediator_entropy: Optional[torch.Tensor] = None
    mediator_to_group_entropy: Optional[torch.Tensor] = None
    state_to_mediator_effective_keys: Optional[torch.Tensor] = None
    local_to_global_mediator_effective_keys: Optional[torch.Tensor] = None
    mediator_to_group_effective_keys: Optional[torch.Tensor] = None
    mediator_usage: Optional[torch.Tensor] = None
    edge_sparsity: Optional[torch.Tensor] = None
    edge_entropy: Optional[torch.Tensor] = None
    control_edge_norm: Optional[torch.Tensor] = None
    mediator_orthogonality: Optional[torch.Tensor] = None
    edge_scores_gm: Optional[torch.Tensor] = None
    baseline_to_mediator_gm: Optional[torch.Tensor] = None
    residual_to_mediator_gm: Optional[torch.Tensor] = None
    residual_to_mediator_abs_gm: Optional[torch.Tensor] = None
    residual_edge_abs_mean: Optional[torch.Tensor] = None
    residual_edge_signed_mean: Optional[torch.Tensor] = None
    mediator_usage_entropy: Optional[torch.Tensor] = None
    mediator_usage_min: Optional[torch.Tensor] = None
    mediator_usage_max: Optional[torch.Tensor] = None
    attn_mediator_to_group_gm: Optional[torch.Tensor] = None
    effective_mediator_to_growth_gm: Optional[torch.Tensor] = None
    residual_mediator_to_growth_gm: Optional[torch.Tensor] = None
    mediator_to_growth_gm: Optional[torch.Tensor] = None


@dataclass
class CausalContextState(ContextState):
    """ContextState with CEA-specific group context and edge diagnostics."""

    causal_context_g: Optional[torch.Tensor] = None
    mediator_tokens: Optional[torch.Tensor] = None
    edge_scores_gm: Optional[torch.Tensor] = None
    baseline_edge_scores_gm: Optional[torch.Tensor] = None
    residual_edge_scores_gm: Optional[torch.Tensor] = None
    residual_edge_magnitude_gm: Optional[torch.Tensor] = None


class CausalEcologicalAttentionContext(nn.Module):
    """Mass-aware, intervention-addressable mediator attention.

    CEA keeps CREDO finite-measure semantics intact: ``q``, ``s``, masses, and
    frequencies come from explicit log-space reductions.  Directed attention
    learns group-specific mediator features for the growth channel.
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        n_programs: int,
        mediator_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        token_dim: int = 64,
        n_heads: int = 4,
        n_mediators: int = 12,
        dropout: float = 0.05,
        mass_attention_temperature: float = 0.5,
        fixed_program_centroids: Optional[torch.Tensor] = None,
        program_assignment_scale: float = 1.0,
        activation_checkpointing: bool = False,
        use_sparse_edges: bool = True,
    ) -> None:
        super().__init__()
        if context_dim != n_programs + mediator_dim:
            raise ValueError(
                "CausalEcologicalAttentionContext returns identity contexts, "
                f"so context_dim must equal n_programs + mediator_dim "
                f"({context_dim} != {n_programs + mediator_dim})."
            )
        if token_dim % n_heads != 0:
            raise ValueError("token_dim must be divisible by n_heads")
        if n_mediators < 1:
            raise ValueError("n_mediators must be >= 1")

        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.n_programs = int(n_programs)
        self.mediator_dim = int(mediator_dim)
        self.context_dim = int(context_dim)
        self.token_dim = int(token_dim)
        self.n_mediators = int(n_mediators)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.use_sparse_edges = bool(use_sparse_edges)
        self.mass_attention_temperature = float(mass_attention_temperature)

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

        particle_in_dim = latent_dim + 1 + embedding_dim + embedding_dim + 3
        self.particle_tokenizer = nn.Sequential(
            nn.Linear(particle_in_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )

        group_in_dim = embedding_dim + embedding_dim + self.n_programs + latent_dim + 2
        self.group_tokenizer = nn.Sequential(
            nn.Linear(group_in_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )

        self.mediator_tokens = nn.Parameter(torch.randn(n_mediators, token_dim) * 0.02)
        self.state_to_local_med = MassGraphMaskedCrossAttention(
            dim=token_dim,
            heads=n_heads,
            dropout=dropout,
            mass_attention_temperature=mass_attention_temperature,
        )
        self.local_med_to_global_med = MassGraphMaskedCrossAttention(
            dim=token_dim,
            heads=n_heads,
            dropout=dropout,
            mass_attention_temperature=mass_attention_temperature,
        )
        self.global_med_to_group = MassGraphMaskedCrossAttention(
            dim=token_dim,
            heads=n_heads,
            dropout=dropout,
            mass_attention_temperature=0.0,
        )

        baseline_edge_in_dim = token_dim + self.n_programs + 2
        self.baseline_edge_score = nn.Sequential(
            nn.Linear(baseline_edge_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_mediators),
        )
        self.residual_edge_score = nn.Linear(embedding_dim, n_mediators, bias=False)
        nn.init.constant_(self.baseline_edge_score[-1].bias, -2.0)
        nn.init.zeros_(self.residual_edge_score.weight)
        self.phi_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, mediator_dim),
            nn.Tanh(),
        )
        self.group_context_project = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, context_dim, bias=False),
            nn.Tanh(),
        )
        self.phi_state_gate = nn.Parameter(torch.tensor(0.1))
        self.group_context_scale = nn.Parameter(torch.tensor(0.1))

    def _maybe_checkpoint(self, function, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def encode_particles(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode particles into state programs and mediator features."""
        return self.program_encoder.eta(z), self.program_encoder.phi(z)

    @staticmethod
    def _entropy(attn: torch.Tensor | None) -> torch.Tensor | None:
        if attn is None:
            return None
        prob = attn.float().clamp_min(1e-30)
        return -(prob * prob.log()).sum(dim=-1).mean().detach()

    @staticmethod
    def _effective_keys(attn: torch.Tensor | None) -> torch.Tensor | None:
        entropy = CausalEcologicalAttentionContext._entropy(attn)
        return None if entropy is None else entropy.exp().detach()

    def _tokenize_particles(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        residual: torch.Tensor,
        tau: torch.Tensor,
        logw_centered: torch.Tensor,
        log_mass_z: torch.Tensor,
        freq_g: torch.Tensor,
    ) -> torch.Tensor:
        G, N, _ = z.shape
        tau_feat = tau.reshape(1, 1, 1).expand(G, N, 1)
        features = torch.cat(
            [
                z,
                tau_feat,
                a[:, None, :].expand(G, N, -1),
                residual[:, None, :].expand(G, N, -1),
                logw_centered[..., None],
                log_mass_z[:, None, None].expand(G, N, 1),
                freq_g[:, None, None].expand(G, N, 1),
            ],
            dim=-1,
        )
        return self._maybe_checkpoint(self.particle_tokenizer, features)

    def _tokenize_groups(
        self,
        a: torch.Tensor,
        residual: torch.Tensor,
        q_g: torch.Tensor,
        z_bar_g: torch.Tensor,
        log_mass_z: torch.Tensor,
        freq_g: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [a, residual, q_g, z_bar_g, log_mass_z[:, None], freq_g[:, None]],
            dim=-1,
        )
        return self._maybe_checkpoint(self.group_tokenizer, features)

    def forward(
        self,
        z: torch.Tensor,
        logw: torch.Tensor,
        a: torch.Tensor,
        log_m0: torch.Tensor,
        tau: torch.Tensor | float | None = None,
        residual: torch.Tensor | None = None,
        intervention: CausalAttentionIntervention | None = None,
    ) -> CausalContextState:
        if z.ndim != 3:
            raise ValueError(f"z must have shape [G, N, d], got {tuple(z.shape)}")
        if logw.shape != z.shape[:2]:
            raise ValueError("logw must have shape [G, N]")
        if a.shape != (z.shape[0], self.embedding_dim):
            raise ValueError(f"a must have shape [G, {self.embedding_dim}], got {tuple(a.shape)}")
        if log_m0.shape != (z.shape[0],):
            raise ValueError("log_m0 must have shape [G]")
        if residual is None:
            residual = torch.zeros_like(a)
        if residual.shape != a.shape:
            raise ValueError("residual must have the same shape as a")

        G, N, _ = z.shape
        dtype = z.dtype
        device = z.device

        logw_abs32 = log_m0.float()[:, None] + logw.float()
        log_mass_g = torch.logsumexp(logw_abs32, dim=1)
        log_total_mass = torch.logsumexp(log_mass_g, dim=0)
        freq_g32 = torch.exp(log_mass_g - log_total_mass)
        alpha_gi32 = torch.softmax(logw_abs32, dim=1)
        freq_g = freq_g32.to(dtype=dtype)
        alpha_gi = alpha_gi32.to(dtype=dtype)
        mass_g = torch.exp(torch.clamp(log_mass_g, min=-30.0, max=30.0)).to(dtype=dtype)

        eta = self.program_encoder.eta(z)
        phi_state = self.program_encoder.phi(z)
        q_g = (alpha_gi[..., None].to(dtype=eta.dtype) * eta).sum(dim=1)
        q = (freq_g[:, None] * q_g).sum(dim=0)
        z_bar_g = (alpha_gi[..., None] * z).sum(dim=1)

        log_mass_mean = (freq_g32.detach() * log_mass_g.detach()).sum()
        log_mass_var = (freq_g32.detach() * (log_mass_g.detach() - log_mass_mean).square()).sum()
        log_mass_std = torch.sqrt(log_mass_var).clamp_min(1e-4)
        log_mass_z = ((log_mass_g - log_mass_mean) / log_mass_std).to(dtype=dtype)
        logw_centered = (logw_abs32 - log_mass_g[:, None]).to(dtype=dtype)
        tau_tensor = torch.zeros((), device=device, dtype=dtype) if tau is None else torch.as_tensor(
            tau,
            device=device,
            dtype=dtype,
        ).reshape(())

        h_particle = self._tokenize_particles(
            z,
            a,
            residual,
            tau_tensor,
            logw_centered,
            log_mass_z,
            freq_g,
        )
        h_group = self._tokenize_groups(a, residual, q_g, z_bar_g, log_mass_z, freq_g)

        med0 = self.mediator_tokens[None, :, :].expand(G, -1, -1)
        local_med, attn_state = self.state_to_local_med(
            med0,
            h_particle,
            h_particle,
            key_log_weights=logw_abs32,
            return_attention=True,
        )

        local_flat = local_med.reshape(1, G * self.n_mediators, self.token_dim)
        local_log_mass = log_mass_g[:, None].expand(G, self.n_mediators).reshape(1, G * self.n_mediators)
        global_query = self.mediator_tokens[None, :, :]
        global_med, attn_group = self.local_med_to_global_med(
            global_query,
            local_flat,
            local_flat,
            key_log_weights=local_log_mass,
            return_attention=True,
        )
        global_med = global_med.squeeze(0)

        baseline_edge_input = torch.cat(
            [h_group, q_g, log_mass_z[:, None], freq_g[:, None]],
            dim=-1,
        )
        baseline_logits = self.baseline_edge_score(baseline_edge_input)
        residual_logits = self.residual_edge_score(residual)
        if intervention is not None:
            baseline_logits = intervention.apply_baseline_logits(baseline_logits)
            residual_logits = intervention.apply_residual_logits(residual_logits)
        baseline_edge_scores = torch.sigmoid(baseline_logits)
        edge_logits = baseline_logits + residual_logits
        if intervention is not None:
            edge_logits = intervention.apply_effective_logits(edge_logits)
        edge_scores = torch.sigmoid(edge_logits)
        residual_edge_scores = edge_scores - baseline_edge_scores
        residual_edge_magnitude = residual_edge_scores.abs()
        med_keys = global_med[None, :, :].expand(G, -1, -1)
        if self.use_sparse_edges:
            med_values = med_keys * edge_scores[:, :, None].to(dtype=dtype)
            edge_gate = edge_scores.max(dim=-1).values[:, None].to(dtype=dtype)
        else:
            med_values = med_keys
            edge_gate = torch.ones(G, 1, device=device, dtype=dtype)
        group_context, attn_med = self.global_med_to_group(
            h_group[:, None, :],
            med_keys,
            med_values,
            graph_mask=(
                None
                if intervention is None
                else intervention.mediator_to_group_graph_mask(G, self.n_mediators, device)
            ),
            return_attention=True,
        )
        group_context = edge_gate * group_context.squeeze(1)

        h_particle_causal = h_particle + group_context[:, None, :]
        phi_causal = self.phi_head(h_particle_causal)
        phi = phi_causal + self.phi_state_gate * phi_state
        s_g = (alpha_gi[..., None].to(dtype=phi.dtype) * phi).sum(dim=1)
        s = (freq_g[:, None] * s_g).sum(dim=0)
        context = torch.cat([q, s], dim=-1)

        causal_delta_g = self.group_context_scale * self.group_context_project(group_context)
        growth_context_g = context[None, :] + causal_delta_g

        mediator_tokens = global_med
        med_norm = torch.nn.functional.normalize(mediator_tokens, dim=-1)
        gram = med_norm @ med_norm.T
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        edge_prob = edge_scores.float().clamp_min(1e-30)
        edge_dist = edge_prob / edge_prob.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        zero_residual_mask = residual.abs().sum(dim=-1).eq(0)
        control_edge_norm = (
            residual_edge_scores[zero_residual_mask].square().mean().sqrt().detach()
            if zero_residual_mask.any()
            else torch.zeros((), device=device, dtype=dtype)
        )
        mediator_usage = edge_scores.float().mean(dim=0)
        mediator_usage_dist = mediator_usage / mediator_usage.sum().clamp_min(1e-30)
        attn_med_gm = attn_med.mean(dim=(1, 2))
        effective_mediator_to_growth_gm = attn_med_gm * edge_scores
        residual_mediator_to_growth_gm = attn_med_gm * residual_edge_magnitude
        diagnostics = CausalAttentionDiagnostics(
            within_attention_entropy=self._entropy(attn_state),
            group_attention_entropy=self._entropy(attn_group),
            within_effective_keys=self._effective_keys(attn_state),
            group_effective_keys=self._effective_keys(attn_group),
            mass_attention_temperature=torch.tensor(
                self.mass_attention_temperature,
                dtype=dtype,
                device=device,
            ),
            context_norm=context.norm().detach(),
            q_entropy=-(q.clamp_min(1e-30) * q.clamp_min(1e-30).log()).sum().detach(),
            freq_entropy=-(freq_g32.clamp_min(1e-30) * freq_g32.clamp_min(1e-30).log()).sum().detach(),
            mass_log_range=(log_mass_g.max() - log_mass_g.min()).detach(),
            state_to_mediator_entropy=self._entropy(attn_state),
            local_to_global_mediator_entropy=self._entropy(attn_group),
            mediator_to_group_entropy=self._entropy(attn_med),
            state_to_mediator_effective_keys=self._effective_keys(attn_state),
            local_to_global_mediator_effective_keys=self._effective_keys(attn_group),
            mediator_to_group_effective_keys=self._effective_keys(attn_med),
            mediator_usage=edge_scores.mean(dim=0).detach(),
            edge_sparsity=edge_scores.mean().detach(),
            edge_entropy=-(edge_dist * edge_dist.log()).sum(dim=-1).mean().detach(),
            control_edge_norm=control_edge_norm,
            mediator_orthogonality=(gram - eye).square().mean().detach(),
            edge_scores_gm=edge_scores.detach(),
            baseline_to_mediator_gm=baseline_edge_scores.detach(),
            residual_to_mediator_gm=residual_edge_scores.detach(),
            residual_to_mediator_abs_gm=residual_edge_magnitude.detach(),
            residual_edge_abs_mean=residual_edge_magnitude.mean().detach(),
            residual_edge_signed_mean=residual_edge_scores.mean().detach(),
            mediator_usage_entropy=-(mediator_usage_dist * mediator_usage_dist.clamp_min(1e-30).log()).sum().detach(),
            mediator_usage_min=mediator_usage.min().detach(),
            mediator_usage_max=mediator_usage.max().detach(),
            attn_mediator_to_group_gm=attn_med_gm.detach(),
            effective_mediator_to_growth_gm=effective_mediator_to_growth_gm.detach(),
            residual_mediator_to_growth_gm=residual_mediator_to_growth_gm.detach(),
            mediator_to_growth_gm=effective_mediator_to_growth_gm.detach(),
        )

        return CausalContextState(
            q=q,
            s=s,
            context=context,
            mass_g=mass_g,
            freq_g=freq_g,
            log_mass_g=log_mass_g,
            log_total_mass=log_total_mass,
            diagnostics=diagnostics,
            base_context=context,
            growth_context=growth_context_g,
            causal_context_g=growth_context_g,
            mediator_tokens=mediator_tokens,
            edge_scores_gm=edge_scores,
            baseline_edge_scores_gm=baseline_edge_scores,
            residual_edge_scores_gm=residual_edge_scores,
            residual_edge_magnitude_gm=residual_edge_magnitude,
        )


__all__ = [
    "CausalAttentionDiagnostics",
    "CausalContextState",
    "CausalEcologicalAttentionContext",
]
