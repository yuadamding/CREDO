from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from camfnd.data.contract import PerturbationCatalog
from camfnd.models.embeddings import ControlAnchoredEmbeddingStore
from camfnd.models.full_context_map import MeanFieldContextConfig, MeanFieldContextMap
from camfnd.models.time_embedding import TimeEmbedding


def _make_mlp(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")

    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(depth)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class FullCoefficientConfig:
    latent_dim: int
    embedding_dim: int = 8
    hidden_dim: int = 64
    depth: int = 2
    time_frequencies: int = 4
    context_dim: int = 8
    sigma_min: float = 0.02
    r_max: float = 2.0
    use_context: bool = True
    context_config: MeanFieldContextConfig | None = None

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.hidden_dim <= 0 or self.depth <= 0:
            raise ValueError("hidden_dim and depth must be positive.")
        if self.time_frequencies < 0:
            raise ValueError("time_frequencies must be nonnegative.")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive.")
        if self.r_max <= 0:
            raise ValueError("r_max must be positive.")
        if self.context_config is not None:
            self.context_config.validate()
            if self.context_config.latent_dim != self.latent_dim:
                raise ValueError("context_config.latent_dim must match latent_dim.")
            if self.context_config.context_dim != self.context_dim:
                raise ValueError("context_config.context_dim must match context_dim.")

    def resolved_context_config(self) -> MeanFieldContextConfig:
        if self.context_config is not None:
            return self.context_config
        return MeanFieldContextConfig(
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            use_context=self.use_context,
        )


class ControlAnchoredFieldHead(nn.Module):
    """Baseline plus perturbation modulation field head.

    For output dimension `m`, the head returns:

    - baseline: shape [N, m]
    - modulation: shape [N, m, embedding_dim]
    - value: baseline + modulation @ a_g
    """

    def __init__(self, input_dim: int, output_dim: int, embedding_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.embedding_dim = int(embedding_dim)
        self.baseline_net = _make_mlp(input_dim, hidden_dim, depth, output_dim)
        self.modulation_net = _make_mlp(input_dim, hidden_dim, depth, output_dim * embedding_dim)
        self.double()

    def forward(self, u: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if u.ndim != 2:
            raise ValueError("u must have shape [N, input_dim].")
        if embedding.ndim != 1 or embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding must have shape [{self.embedding_dim}], got {tuple(embedding.shape)}"
            )
        baseline = self.baseline_net(u)
        modulation = self.modulation_net(u).reshape(u.shape[0], self.output_dim, self.embedding_dim)
        value = baseline + torch.einsum("nke,e->nk", modulation, embedding)
        return value, baseline, modulation


class ControlAnchoredFullModel(nn.Module):
    """Full-model path with state, time, and context dependent coefficient fields."""

    def __init__(self, catalog: PerturbationCatalog, config: FullCoefficientConfig) -> None:
        super().__init__()
        catalog.validate()
        config.validate()
        self.catalog = catalog
        self.config = config
        self.embedding_store = ControlAnchoredEmbeddingStore(catalog, embedding_dim=config.embedding_dim)
        self.time_embedding = TimeEmbedding(n_frequencies=config.time_frequencies)
        self.context_map = MeanFieldContextMap(config.resolved_context_config())
        input_dim = config.latent_dim + self.time_embedding.output_dim + (
            config.context_dim if config.use_context else 0
        )
        self.drift_head = ControlAnchoredFieldHead(
            input_dim=input_dim,
            output_dim=config.latent_dim,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        self.diffusion_head = ControlAnchoredFieldHead(
            input_dim=input_dim,
            output_dim=config.latent_dim,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        self.growth_head = ControlAnchoredFieldHead(
            input_dim=input_dim,
            output_dim=1,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )
        self.double()

    def _expand_time(self, t: Tensor, batch_size: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        if t.ndim == 0:
            return t.reshape(1, 1).repeat(batch_size, 1).to(device=device, dtype=dtype)
        if t.ndim == 1:
            if t.shape[0] == 1:
                return t.reshape(1, 1).repeat(batch_size, 1).to(device=device, dtype=dtype)
            if t.shape[0] != batch_size:
                raise ValueError("Vector t must have length 1 or batch_size.")
            return t.reshape(batch_size, 1).to(device=device, dtype=dtype)
        if t.ndim == 2 and t.shape[1] == 1:
            if t.shape[0] == 1:
                return t.repeat(batch_size, 1).to(device=device, dtype=dtype)
            if t.shape[0] != batch_size:
                raise ValueError("Matrix t must have shape [1, 1] or [batch_size, 1].")
            return t.to(device=device, dtype=dtype)
        raise ValueError("t must have shape [], [N], or [N, 1].")

    def _expand_context(self, context: Tensor | None, batch_size: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        if not self.config.use_context:
            return torch.zeros(batch_size, 0, dtype=dtype, device=device)
        if context is None:
            context = torch.zeros(self.config.context_dim, dtype=dtype, device=device)
        if context.ndim == 1:
            if context.shape[0] != self.config.context_dim:
                raise ValueError(
                    f"context must have shape [{self.config.context_dim}], got {tuple(context.shape)}"
                )
            return context.reshape(1, -1).repeat(batch_size, 1).to(device=device, dtype=dtype)
        if context.ndim == 2 and context.shape == (batch_size, self.config.context_dim):
            return context.to(device=device, dtype=dtype)
        raise ValueError("context must have shape [context_dim] or [batch_size, context_dim].")

    def common_input(self, z: Tensor, t: Tensor, context: Tensor | None = None) -> Tensor:
        if z.ndim != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"z must have shape [N, {self.config.latent_dim}], got {tuple(z.shape)}."
            )
        t_input = self._expand_time(t, z.shape[0], dtype=z.dtype, device=z.device)
        time_features = self.time_embedding(t_input)
        context_features = self._expand_context(context, z.shape[0], dtype=z.dtype, device=z.device)
        return torch.cat([z, time_features, context_features], dim=1)

    def context_values(self, particles) -> Dict[str, Tensor]:
        if not self.config.use_context:
            sample_ids = sorted({state.sample_id for state in particles.values()})
            device = next(iter(particles.values())).z.device
            dtype = next(iter(particles.values())).z.dtype
            zero = torch.zeros(self.config.context_dim, dtype=dtype, device=device)
            return {sample_id: zero.clone() for sample_id in sample_ids}
        return self.context_map(particles)

    def coefficients(
        self,
        z: Tensor,
        t: Tensor,
        perturbation_id: str,
        context: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        embedding = self.embedding_store.forward_one(perturbation_id).to(device=z.device, dtype=z.dtype)
        u = self.common_input(z, t, context)
        drift, drift_baseline, drift_mod = self.drift_head(u, embedding)
        diffusion_raw, diffusion_baseline, diffusion_mod = self.diffusion_head(u, embedding)
        growth_raw, growth_baseline, growth_mod = self.growth_head(u, embedding)
        diffusion = torch.nn.functional.softplus(diffusion_raw) + self.config.sigma_min
        growth = self.config.r_max * torch.tanh(growth_raw)
        return {
            "drift": drift,
            "diffusion": diffusion,
            "growth": growth,
            "common_input": u,
            "drift_baseline": drift_baseline,
            "drift_modulation": drift_mod,
            "diffusion_baseline": diffusion_baseline,
            "diffusion_modulation": diffusion_mod,
            "growth_baseline": growth_baseline,
            "growth_modulation": growth_mod,
            "diffusion_raw": diffusion_raw,
            "growth_raw": growth_raw,
        }

    def regularization_terms(self) -> Dict[str, Tensor]:
        reg_emb = self.embedding_store.regularization()
        reg_mod = torch.stack([
            self.drift_head.modulation_net[-1].weight.pow(2).mean(),
            self.diffusion_head.modulation_net[-1].weight.pow(2).mean(),
            self.growth_head.modulation_net[-1].weight.pow(2).mean(),
        ]).mean()
        reg_nn = torch.stack([
            parameter.pow(2).mean() for parameter in self.parameters()
        ]).mean()
        reg_context = self.context_map.regularization()
        return {
            "emb": reg_emb,
            "mod": reg_mod,
            "disp": diffusion_penalty(self),
            "nn": reg_nn,
            "context": reg_context,
        }

    def control_anchor_is_exact(self) -> bool:
        return self.embedding_store.control_anchor_is_exact(atol=0.0)


def diffusion_penalty(model: ControlAnchoredFullModel) -> Tensor:
    penalties = []
    for parameter in model.diffusion_head.parameters():
        penalties.append(parameter.pow(2).mean())
    return torch.stack(penalties).mean()
