"""Frozen transformer-SDE v2 architecture with historical parameter names."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ContextDiagnostics:
    within_attention_entropy: torch.Tensor | None = None
    group_attention_entropy: torch.Tensor | None = None
    within_effective_keys: torch.Tensor | None = None
    group_effective_keys: torch.Tensor | None = None
    mass_attention_temperature: torch.Tensor | None = None
    context_norm: torch.Tensor | None = None
    q_entropy: torch.Tensor | None = None
    freq_entropy: torch.Tensor | None = None
    mass_log_range: torch.Tensor | None = None


@dataclass
class ContextState:
    q: torch.Tensor
    s: torch.Tensor
    context: torch.Tensor
    mass_g: torch.Tensor
    freq_g: torch.Tensor
    log_mass_g: torch.Tensor | None = None
    log_total_mass: torch.Tensor | None = None
    diagnostics: ContextDiagnostics | None = None
    base_context: torch.Tensor | None = None
    growth_context: torch.Tensor | None = None


@dataclass
class GroupStatistics:
    log_n_g: torch.Tensor
    eta_g: torch.Tensor
    phi_g: torch.Tensor


class ProgramEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_programs: int,
        mediator_dim: int,
        hidden_dim: int = 64,
        fixed_centroids: torch.Tensor | None = None,
        assignment_scale: float = 1.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if fixed_centroids is not None:
            centroids = torch.as_tensor(fixed_centroids, dtype=torch.float32)
            if centroids.ndim != 2 or centroids.shape[1] != latent_dim:
                raise ValueError("fixed_centroids must have shape [n_programs, latent_dim].")
            self.register_buffer("fixed_centroids", centroids)
            self.n_programs = int(centroids.shape[0])
        else:
            self.register_buffer("fixed_centroids", torch.empty(0, latent_dim))
            self.n_programs = n_programs
        self.mediator_dim = mediator_dim
        self.assignment_scale = float(assignment_scale)
        self.use_fixed_centroids = fixed_centroids is not None
        self.activation_checkpointing = activation_checkpointing
        if self.use_fixed_centroids:
            self.eta_net = None
        else:
            self.eta_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.n_programs),
            )
        self.phi_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, mediator_dim),
            nn.Tanh(),
        )

    def eta(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_fixed_centroids:
            centers = self.fixed_centroids.to(device=z.device, dtype=z.dtype)
            squared_distance = ((z.unsqueeze(-2) - centers) ** 2).sum(dim=-1)
            return torch.softmax(-self.assignment_scale * squared_distance, dim=-1)
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            logits = checkpoint(self.eta_net, z, use_reentrant=False)
        else:
            logits = self.eta_net(z)
        return torch.softmax(logits, dim=-1)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self.phi_net, z, use_reentrant=False)
        return self.phi_net(z)


class ContextAggregator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_programs: int,
        mediator_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
        use_identity_context: bool = True,
        fixed_program_centroids: torch.Tensor | None = None,
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
        elif context_dim != input_dim:
            raise ValueError("Identity context width must equal programs plus mediators.")

    def encode_particles(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.eta(z), self.encoder.phi(z)

    def summarize_groups(
        self,
        z: torch.Tensor,
        logw: torch.Tensor,
        log_m0: torch.Tensor,
        eta: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> tuple[GroupStatistics, torch.Tensor, torch.Tensor]:
        if eta is None or phi is None:
            eta, phi = self.encode_particles(z)
        log_n_g = log_m0 + torch.logsumexp(logw, dim=-1)
        log_normalized = logw - torch.logsumexp(logw, dim=-1, keepdim=True)
        normalized = log_normalized.exp()
        eta_g = (normalized.unsqueeze(-1) * eta).sum(dim=1)
        phi_g = (normalized.unsqueeze(-1) * phi).sum(dim=1)
        return GroupStatistics(log_n_g, eta_g, phi_g), eta, phi

    def context_from_group_statistics(self, stats: GroupStatistics) -> ContextState:
        log_total = torch.logsumexp(stats.log_n_g, dim=0)
        frequency = torch.exp(stats.log_n_g - log_total)
        mass = torch.exp(torch.clamp(stats.log_n_g, min=-30.0, max=30.0))
        q = (frequency.unsqueeze(-1) * stats.eta_g).sum(dim=0)
        s = (frequency.unsqueeze(-1) * stats.phi_g).sum(dim=0)
        joined = torch.cat((q, s), dim=-1)
        context = joined if self.use_identity_context else self.psi(joined)
        return ContextState(q, s, context, mass, frequency, stats.log_n_g, log_total)

    def forward(
        self,
        z: torch.Tensor,
        logw: torch.Tensor,
        a: torch.Tensor,
        log_m0: torch.Tensor,
        tau: torch.Tensor | float | None = None,
    ) -> ContextState:
        del a, tau
        stats, _, _ = self.summarize_groups(z, logw, log_m0)
        return self.context_from_group_statistics(stats)


class MassBiasedCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Attention width must be divisible by the head count.")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.mass_attention_temperature = float(mass_attention_temperature)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_entropy: torch.Tensor | None = None
        self.last_effective_keys: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_log_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_count, _ = query.shape
        key_count = key.shape[1]
        q = self.q_proj(query).view(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, key_count, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, key_count, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if key_log_weights is not None:
            if key_log_weights.shape != (batch, key_count):
                raise ValueError("key_log_weights shape does not match attention keys.")
            stable = key_log_weights.float() - torch.logsumexp(
                key_log_weights.float(), dim=-1, keepdim=True
            )
            logits = (
                logits + self.mass_attention_temperature * stable.to(logits.dtype)[:, None, None, :]
            )
        weights = torch.softmax(logits, dim=-1)
        entropy = -(weights.clamp_min(1e-30) * weights.clamp_min(1e-30).log()).sum(dim=-1)
        self.last_attention_entropy = entropy.mean().detach()
        self.last_effective_keys = entropy.exp().mean().detach()
        output = torch.matmul(self.dropout(weights), v)
        output = output.transpose(1, 2).contiguous().view(batch, query_count, self.dim)
        return self.out_proj(output)


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden_dim or 4 * dim)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class MassBiasedSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.0,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MassBiasedCrossAttention(dim, heads, dropout, mass_attention_temperature)
        self.ff = FeedForwardBlock(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, key_log_weights: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm(x), x, x, key_log_weights=key_log_weights)
        return self.ff(x)


class InducedSetAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        n_inducing: int = 16,
        layers: int = 2,
        dropout: float = 0.0,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if n_inducing < 1 or layers < 1:
            raise ValueError("Induced attention requires positive inducing tokens and layers.")
        self.inducing = nn.Parameter(torch.randn(n_inducing, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ind_norm": nn.LayerNorm(dim),
                        "x_norm": nn.LayerNorm(dim),
                        "ind_attn": MassBiasedCrossAttention(
                            dim, heads, dropout, mass_attention_temperature
                        ),
                        "x_attn": MassBiasedCrossAttention(
                            dim, heads, dropout, mass_attention_temperature
                        ),
                        "ind_ff": FeedForwardBlock(dim, dropout=dropout),
                        "x_ff": FeedForwardBlock(dim, dropout=dropout),
                    }
                )
            )

    def forward(
        self, x: torch.Tensor, *, key_log_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        inducing = self.inducing.unsqueeze(0).expand(x.shape[0], -1, -1)
        for layer in self.layers:
            inducing = inducing + layer["ind_attn"](
                layer["ind_norm"](inducing), x, x, key_log_weights=key_log_weights
            )
            inducing = layer["ind_ff"](inducing)
            x = x + layer["x_attn"](layer["x_norm"](x), inducing, inducing)
            x = layer["x_ff"](x)
        return x


class MassAwareTransformerContextAggregator(nn.Module):
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
        fixed_program_centroids: torch.Tensor | None = None,
        program_assignment_scale: float = 1.0,
        activation_checkpointing: bool = False,
        mass_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if context_dim != n_programs + mediator_dim:
            raise ValueError("Transformer context width must equal programs plus mediators.")
        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.mediator_dim = int(mediator_dim)
        self.context_dim = int(context_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.mass_attention_temperature = float(mass_attention_temperature)
        self.program_encoder = ProgramEncoder(
            latent_dim,
            n_programs,
            mediator_dim,
            hidden_dim,
            fixed_centroids=fixed_program_centroids,
            assignment_scale=program_assignment_scale,
            activation_checkpointing=activation_checkpointing,
        )
        self.n_programs = self.program_encoder.n_programs
        self.token_in = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim + 4, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
        )
        self.within = InducedSetAttentionBlock(
            token_dim,
            n_heads,
            n_inducing,
            n_within_layers,
            dropout,
            mass_attention_temperature,
        )
        self.cross_blocks = nn.ModuleList(
            [
                MassBiasedSelfAttentionBlock(
                    token_dim, n_heads, dropout, mass_attention_temperature
                )
                for _ in range(n_cross_layers)
            ]
        )
        self.group_to_particle = nn.Linear(token_dim, token_dim)
        self.phi_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, mediator_dim), nn.Tanh()
        )
        self.phi_state_gate = nn.Parameter(torch.tensor(0.1))

    def _maybe_checkpoint(self, function, *args):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    @staticmethod
    def _mean_attention_attr(
        modules: list[MassBiasedCrossAttention], attr: str
    ) -> torch.Tensor | None:
        values = [value for module in modules if (value := getattr(module, attr, None)) is not None]
        return None if not values else torch.stack(values).mean()

    def _diagnostics(
        self,
        context: torch.Tensor,
        q: torch.Tensor,
        frequency: torch.Tensor,
        log_mass: torch.Tensor,
    ) -> ContextDiagnostics:
        within_modules = [
            module
            for layer in self.within.layers
            for module in (layer["ind_attn"], layer["x_attn"])
        ]
        group_modules = [block.attn for block in self.cross_blocks]
        q_probability = q.clamp_min(1e-30)
        frequency_probability = frequency.clamp_min(1e-30)
        return ContextDiagnostics(
            within_attention_entropy=self._mean_attention_attr(
                within_modules, "last_attention_entropy"
            ),
            group_attention_entropy=self._mean_attention_attr(
                group_modules, "last_attention_entropy"
            ),
            within_effective_keys=self._mean_attention_attr(within_modules, "last_effective_keys"),
            group_effective_keys=self._mean_attention_attr(group_modules, "last_effective_keys"),
            mass_attention_temperature=torch.tensor(
                self.mass_attention_temperature, dtype=context.dtype, device=context.device
            ),
            context_norm=context.norm().detach(),
            q_entropy=-(q_probability * q_probability.log()).sum().detach(),
            freq_entropy=-(frequency_probability * frequency_probability.log()).sum().detach(),
            mass_log_range=(log_mass.max() - log_mass.min()).detach(),
        )

    def encode_particles(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.program_encoder.eta(z), self.program_encoder.phi(z)

    def forward(
        self,
        z: torch.Tensor,
        logw: torch.Tensor,
        a: torch.Tensor,
        log_m0: torch.Tensor,
        tau: torch.Tensor | float | None = None,
    ) -> ContextState:
        if z.ndim != 3 or logw.shape != z.shape[:2]:
            raise ValueError("Transformer context requires z [G,N,d] and logw [G,N].")
        group_count, particle_count, _ = z.shape
        absolute32 = log_m0.float()[:, None] + logw.float()
        log_mass = torch.logsumexp(absolute32, dim=1)
        log_total_mass = torch.logsumexp(log_mass, dim=0)
        frequency32 = torch.exp(log_mass - log_total_mass)
        mass = torch.exp(torch.clamp(log_mass, min=-30.0, max=30.0))
        within = torch.softmax(absolute32, dim=1).to(dtype=z.dtype)
        frequency = frequency32.to(dtype=z.dtype)
        eta = self.program_encoder.eta(z)
        phi_state = self.program_encoder.phi(z)
        a_token = a[:, None, :].expand(group_count, particle_count, -1)
        centered = (absolute32 - log_mass[:, None]).to(z.dtype)[:, :, None]
        detached_frequency = frequency32.detach()
        detached_log_mass = log_mass.detach()
        log_mass_mean = (detached_frequency * detached_log_mass).sum()
        log_mass_variance = (
            detached_frequency * (detached_log_mass - log_mass_mean).square()
        ).sum()
        log_mass_std = torch.sqrt(log_mass_variance).clamp_min(1e-4)
        log_mass_z = ((log_mass - log_mass_mean) / log_mass_std).to(z.dtype)
        log_mass_z = log_mass_z.view(group_count, 1, 1).expand(group_count, particle_count, 1)
        frequency_token = frequency.view(group_count, 1, 1).expand(group_count, particle_count, 1)
        tau_tensor = torch.as_tensor(
            0.0 if tau is None else tau, dtype=z.dtype, device=z.device
        ).reshape(())
        tau_token = tau_tensor.expand(group_count, particle_count).unsqueeze(-1)
        tokens = torch.cat((z, a_token, centered, log_mass_z, frequency_token, tau_token), dim=-1)
        particles = self._maybe_checkpoint(self.token_in, tokens)
        particles = self._maybe_checkpoint(
            lambda h, weights: self.within(h, key_log_weights=weights),
            particles,
            absolute32,
        )
        groups = (within[..., None] * particles).sum(dim=1).unsqueeze(0)
        group_log_weights = log_mass.unsqueeze(0)
        for block in self.cross_blocks:
            groups = self._maybe_checkpoint(
                lambda h, weights, layer=block: layer(h, key_log_weights=weights),
                groups,
                group_log_weights,
            )
        groups = groups.squeeze(0)
        particles = particles + self.group_to_particle(groups)[:, None, :]
        phi = self.phi_head(particles) + self.phi_state_gate * phi_state
        q_group = (within[..., None].to(eta.dtype) * eta).sum(dim=1)
        s_group = (within[..., None].to(phi.dtype) * phi).sum(dim=1)
        q = (frequency[:, None] * q_group).sum(dim=0)
        s = (frequency[:, None] * s_group).sum(dim=0)
        context = torch.cat((q, s), dim=-1)
        return ContextState(
            q,
            s,
            context,
            mass,
            frequency,
            log_mass,
            log_total_mass,
            self._diagnostics(context, q, frequency, log_mass),
        )


class PerturbationEmbedding(nn.Module):
    def __init__(
        self,
        perturbation_ids: list[str],
        control_ids: list[str],
        embedding_dim: int,
        control_mode: str = "soft_ref",
        control_ref_penalty: float = 5e-4,
        use_growth_intercept: bool = True,
        shared_guide_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.perturbation_ids = perturbation_ids
        self.all_control_ids = set(control_ids)
        self.control_mode = control_mode
        self.anchor_controls = control_mode == "anchored"
        self.control_ids = set(control_ids) if control_mode in {"anchored", "soft_ref"} else set()
        self.embedding_dim = embedding_dim
        self.control_ref_penalty = float(control_ref_penalty)
        self.shared_guide_embedding = bool(shared_guide_embedding)
        self._id_to_idx = {value: index for index, value in enumerate(perturbation_ids)}
        noncontrols = [value for value in perturbation_ids if value not in self.control_ids]
        self.non_control_ids = noncontrols
        self._nc_to_local = {value: index for index, value in enumerate(noncontrols)}
        weight = torch.zeros(len(noncontrols), embedding_dim)
        if len(noncontrols) > 0:
            if embedding_dim >= len(noncontrols):
                for index in range(len(noncontrols)):
                    if index < embedding_dim:
                        weight[index, index] = 1.0
            else:
                nn.init.xavier_uniform_(weight)
            self.embeddings = nn.Parameter(weight)
            if use_growth_intercept:
                self.growth_bias = nn.Parameter(torch.zeros(len(noncontrols)))
            else:
                self.register_parameter("growth_bias", None)
        else:
            self.register_parameter("embeddings", None)
            self.register_parameter("growth_bias", None)
        if control_mode == "soft_ref":
            self.reference_embedding = nn.Parameter(torch.zeros(embedding_dim))
        else:
            self.register_parameter("reference_embedding", None)
        if shared_guide_embedding:
            self.shared_embedding = nn.Parameter(torch.zeros(embedding_dim))
            if use_growth_intercept:
                self.shared_growth_bias = nn.Parameter(torch.zeros(()))
            else:
                self.register_parameter("shared_growth_bias", None)
        else:
            self.register_parameter("shared_embedding", None)
            self.register_parameter("shared_growth_bias", None)
        self.register_buffer("_device_sentinel", torch.zeros(1))

    def forward(self, perturbation_ids: list[str]) -> torch.Tensor:
        device = self._device_sentinel.device
        dtype = self._device_sentinel.dtype
        if self.shared_guide_embedding:
            return (
                self.shared_embedding.to(device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(len(perturbation_ids), -1)
            )
        output = torch.zeros(len(perturbation_ids), self.embedding_dim, device=device, dtype=dtype)
        for index, perturbation_id in enumerate(perturbation_ids):
            if perturbation_id not in self.control_ids and self.embeddings is not None:
                output[index] = self.embeddings[self._nc_to_local[perturbation_id]]
        if self.reference_embedding is not None:
            output = output + self.reference_embedding.to(device=device, dtype=dtype).unsqueeze(0)
        return output

    def residuals(self, perturbation_ids: list[str]) -> torch.Tensor:
        output = self._device_sentinel.new_zeros(len(perturbation_ids), self.embedding_dim)
        if self.embeddings is None:
            return output
        for index, perturbation_id in enumerate(perturbation_ids):
            local = self._nc_to_local.get(perturbation_id)
            if local is not None:
                output[index] = self.embeddings[local]
        return output

    def growth_intercepts(self, perturbation_ids: list[str]) -> torch.Tensor:
        output = self._device_sentinel.new_zeros(len(perturbation_ids))
        if self.growth_bias is None:
            return output
        for index, perturbation_id in enumerate(perturbation_ids):
            if perturbation_id not in self.control_ids:
                output[index] = self.growth_bias[self._nc_to_local[perturbation_id]]
        return output


class TimeEmbedding(nn.Module):
    def __init__(self, n_frequencies: int = 4) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.n_frequencies

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        parts = [tau.unsqueeze(-1)]
        for frequency in range(1, self.n_frequencies + 1):
            parts.append(torch.sin(frequency * torch.pi * tau).unsqueeze(-1))
            parts.append(torch.cos(frequency * torch.pi * tau).unsqueeze(-1))
        return torch.cat(parts, dim=-1)


class EcologicalPayoff(nn.Module):
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
        self.P0 = nn.Parameter(torch.zeros(n_programs, n_programs))
        nn.init.xavier_uniform_(self.P0)
        self.actual_ranks = min(n_ranks, embedding_dim)
        if self.actual_ranks > 0:
            self.P_pert = nn.Parameter(torch.zeros(self.actual_ranks, n_programs, n_programs))
            nn.init.normal_(self.P_pert, std=0.01)
        else:
            self.register_parameter("P_pert", None)
        if mediator_dim > 0 and self.actual_ranks > 0:
            self.mediator_net = nn.Sequential(
                nn.Linear(mediator_dim, n_programs * n_programs * self.actual_ranks),
                nn.Tanh(),
            )
        else:
            self.mediator_net = None

    def forward(
        self,
        eta_z: torch.Tensor,
        a_g: torch.Tensor,
        q: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        phi = torch.einsum("gnk,k->gn", eta_z, self.P0 @ q)
        if self.P_pert is not None and self.actual_ranks > 0:
            payoff = self.P_pert
            if self.mediator_net is not None and s is not None:
                payoff = payoff + self.mediator_net(s).reshape(
                    self.actual_ranks, self.n_programs, self.n_programs
                )
            payoff_q = torch.einsum("rkj,j->rk", payoff, q)
            perturbation_q = torch.einsum("gr,rk->gk", a_g[:, : self.actual_ranks], payoff_q)
            phi = phi + torch.einsum("gnk,gk->gn", eta_z, perturbation_q)
        return phi


@dataclass
class Coefficients:
    drift: torch.Tensor
    sigma_diag: torch.Tensor
    growth: torch.Tensor


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden_dim), nn.Tanh()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class ControlAnchoredFieldHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        depth: int,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.baseline_net = _mlp(input_dim, out_dim, hidden_dim, depth)
        self.modulation_net = _mlp(input_dim, out_dim * embedding_dim, hidden_dim, depth)
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.activation_checkpointing = activation_checkpointing

    def forward(self, u: torch.Tensor, a_g: torch.Tensor) -> torch.Tensor:
        group_count, particle_count, _ = u.shape
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            baseline = checkpoint(self.baseline_net, u, use_reentrant=False)
            flattened = checkpoint(self.modulation_net, u, use_reentrant=False)
        else:
            baseline = self.baseline_net(u)
            flattened = self.modulation_net(u)
        matrix = flattened.view(group_count, particle_count, self.out_dim, self.embedding_dim)
        return baseline + torch.einsum("gnor,gr->gno", matrix, a_g)


class CoefficientNetworks(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        activation_checkpointing: bool = False,
        n_time_freqs: int = 4,
        sigma_min: float = 1e-3,
        r_max: float = 3.0,
        n_programs: int = 8,
        n_payoff_ranks: int = 4,
        ecological_growth: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.sigma_min = sigma_min
        self.r_max = r_max
        self.ecological_growth = ecological_growth
        self.n_programs = n_programs
        self.activation_checkpointing = activation_checkpointing
        self.time_embed = TimeEmbedding(n_time_freqs)
        input_dim = latent_dim + self.time_embed.output_dim + context_dim
        self.drift_head = ControlAnchoredFieldHead(
            input_dim,
            latent_dim,
            embedding_dim,
            hidden_dim,
            depth,
            activation_checkpointing,
        )
        self.sigma_head = ControlAnchoredFieldHead(
            input_dim,
            latent_dim,
            embedding_dim,
            hidden_dim,
            depth,
            activation_checkpointing,
        )
        self.growth_head = ControlAnchoredFieldHead(
            input_dim, 1, embedding_dim, hidden_dim, depth, activation_checkpointing
        )
        self.ecology = (
            EcologicalPayoff(n_programs, embedding_dim, n_payoff_ranks)
            if ecological_growth
            else None
        )

    def _common_input(
        self, z: torch.Tensor, tau: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        group_count, particle_count, _ = z.shape
        gamma = self.time_embed(tau.reshape(1)).squeeze(0)
        context_expanded = context.unsqueeze(0).unsqueeze(0).expand(group_count, particle_count, -1)
        gamma_expanded = gamma.unsqueeze(0).unsqueeze(0).expand(group_count, particle_count, -1)
        return torch.cat((z, gamma_expanded, context_expanded), dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        context: torch.Tensor,
        a: torch.Tensor,
        growth_intercept: torch.Tensor | None = None,
        eta_z: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        growth_context: torch.Tensor | None = None,
    ) -> Coefficients:
        common = self._common_input(z, tau, context)
        growth_input = (
            common if growth_context is None else self._common_input(z, tau, growth_context)
        )
        drift = self.drift_head(common, a)
        sigma = F.softplus(self.sigma_head(common, a)) + self.sigma_min
        growth_raw = self.growth_head(growth_input, a).squeeze(-1)
        if growth_intercept is not None:
            growth_raw = growth_raw + growth_intercept.unsqueeze(-1)
        if self.ecology is not None and eta_z is not None and q is not None:
            growth_raw = growth_raw + self.ecology(eta_z, a, q, s)
        return Coefficients(drift, sigma, self.r_max * torch.tanh(growth_raw))


class FullDynamicsModel(nn.Module):
    """Historical v2 model. Constructor defaults are preserved for strict loading."""

    def __init__(
        self,
        perturbation_ids: list[str],
        control_ids: list[str],
        latent_dim: int = 16,
        embedding_dim: int = 8,
        n_programs: int = 8,
        mediator_dim: int = 8,
        hidden_dim: int = 128,
        depth: int = 3,
        activation_checkpointing: bool = False,
        n_time_freqs: int = 4,
        sigma_min: float = 1e-3,
        r_max: float = 3.0,
        n_payoff_ranks: int = 4,
        ecological_growth: bool = True,
        use_growth_intercept: bool = True,
        shared_guide_embedding: bool = False,
        program_centroids: torch.Tensor | None = None,
        program_assignment_scale: float = 1.0,
        control_mode: str = "soft_ref",
        control_ref_penalty: float = 5e-4,
        context_kind: Literal["mlp", "transformer"] = "mlp",
        transformer_token_dim: int = 128,
        transformer_heads: int = 4,
        transformer_within_layers: int = 2,
        transformer_cross_layers: int = 2,
        transformer_inducing: int = 16,
        transformer_dropout: float = 0.05,
        mass_attention_temperature: float = 1.0,
        transformer_growth_only: bool = True,
    ) -> None:
        super().__init__()
        self.perturbation_ids = perturbation_ids
        self.control_ids = set(control_ids)
        self.control_mode = control_mode
        self.context_kind = context_kind
        self.transformer_growth_only = bool(transformer_growth_only)
        self.anchor_controls = control_mode == "anchored"
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        if program_centroids is not None:
            program_centroids = torch.as_tensor(program_centroids, dtype=torch.float32)
            self.n_programs = int(program_centroids.shape[0])
        else:
            self.n_programs = n_programs
        self.mediator_dim = mediator_dim
        context_dim = self.n_programs + mediator_dim
        self.embedding = PerturbationEmbedding(
            perturbation_ids,
            control_ids,
            embedding_dim,
            control_mode,
            control_ref_penalty,
            use_growth_intercept,
            shared_guide_embedding,
        )
        if context_kind == "mlp":
            self.context_agg = ContextAggregator(
                latent_dim,
                self.n_programs,
                mediator_dim,
                context_dim,
                hidden_dim,
                True,
                program_centroids,
                program_assignment_scale,
                activation_checkpointing,
            )
            self.meanfield_context_agg = None
        elif context_kind == "transformer":
            self.context_agg = MassAwareTransformerContextAggregator(
                latent_dim,
                embedding_dim,
                self.n_programs,
                mediator_dim,
                context_dim,
                hidden_dim,
                transformer_token_dim,
                transformer_heads,
                transformer_within_layers,
                transformer_cross_layers,
                transformer_inducing,
                transformer_dropout,
                program_centroids,
                program_assignment_scale,
                activation_checkpointing,
                mass_attention_temperature,
            )
            if transformer_growth_only:
                self.meanfield_context_agg = ContextAggregator(
                    latent_dim,
                    self.n_programs,
                    mediator_dim,
                    context_dim,
                    hidden_dim,
                    True,
                    program_centroids,
                    program_assignment_scale,
                    activation_checkpointing,
                )
                self.meanfield_context_agg.encoder = self.context_agg.program_encoder
            else:
                self.meanfield_context_agg = None
        else:
            raise ValueError(f"Unknown context_kind {context_kind!r}.")
        self.coeff_nets = CoefficientNetworks(
            latent_dim,
            embedding_dim,
            context_dim,
            hidden_dim,
            depth,
            activation_checkpointing,
            n_time_freqs,
            sigma_min,
            r_max,
            self.n_programs,
            n_payoff_ranks,
            ecological_growth,
        )

    def get_embeddings(self, perturbation_ids: list[str] | None = None) -> torch.Tensor:
        return self.embedding(perturbation_ids or self.perturbation_ids)

    def step(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        logw: torch.Tensor,
        log_m0: torch.Tensor,
        perturbation_ids: list[str] | None = None,
    ) -> tuple[Coefficients, ContextState]:
        identifiers = perturbation_ids or self.perturbation_ids
        embedding = self.embedding(identifiers)
        growth_intercept = self.embedding.growth_intercepts(identifiers)
        context_state = self.context_agg(z, logw, embedding, log_m0, tau=tau)
        base_context = context_state.context
        growth_context = None
        if self.transformer_growth_only and self.meanfield_context_agg is not None:
            base_state = self.meanfield_context_agg(z, logw, embedding, log_m0, tau=tau)
            base_context = base_state.context
            growth_context = context_state.context
        context_state.base_context = base_context
        context_state.growth_context = (
            growth_context if growth_context is not None else base_context
        )
        eta_z, _ = self.context_agg.encode_particles(z)
        coefficients = self.coeff_nets(
            z,
            tau,
            base_context,
            embedding,
            growth_intercept,
            eta_z,
            context_state.q,
            context_state.s,
            growth_context,
        )
        return coefficients, context_state

    def assert_soft_reference(self) -> None:
        controls = list(self.control_ids)
        if not controls:
            raise AssertionError("soft-reference v2 recipe requires a control embedding.")
        residual = self.embedding.residuals(controls)
        if not torch.equal(residual, torch.zeros_like(residual)):
            raise AssertionError("Control residuals must be exactly zero.")
        effective = self.embedding(controls)
        reference = self.embedding.reference_embedding.unsqueeze(0).expand_as(effective)
        if not torch.equal(effective, reference):
            raise AssertionError("All controls must share one reference embedding.")


__all__ = ["Coefficients", "ContextState", "FullDynamicsModel"]
