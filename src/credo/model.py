"""The single CREDO soft-reference dynamics model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DynamicsOutput:
    drift: torch.Tensor
    sigma_diag: torch.Tensor
    growth: torch.Tensor
    programs: torch.Tensor


def _mlp(input_dim: int, hidden_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(width, hidden_dim), nn.Tanh()))
        width = hidden_dim
    return nn.Sequential(*layers)


class _TimeEmbedding(nn.Module):
    def __init__(self, frequencies: int = 4) -> None:
        super().__init__()
        self.frequencies = int(frequencies)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.frequencies

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(())
        values = [tau]
        for frequency in range(1, self.frequencies + 1):
            values.extend(
                (
                    torch.sin(frequency * torch.pi * tau),
                    torch.cos(frequency * torch.pi * tau),
                )
            )
        return torch.stack(values)


class CREDOModel(nn.Module):
    """Soft-reference drift, diagonal diffusion, growth, and ecological payoff.

    Controls are absent from the residual parameter table. Every control uses
    one learned reference embedding exactly; noncontrols use reference plus a
    learned residual. Ecological composition can affect growth only.
    """

    def __init__(
        self,
        *,
        embedding_ids: Sequence[str],
        control_embedding_ids: Sequence[str],
        latent_dim: int,
        embedding_dim: int = 8,
        n_programs: int = 8,
        hidden_dim: int = 128,
        context_mode: Literal["none", "catalog_bank"] = "catalog_bank",
        sigma_min: float = 1e-3,
        growth_max: float = 3.0,
        payoff_rank: int = 4,
    ) -> None:
        super().__init__()
        ids = tuple(str(value) for value in embedding_ids)
        controls = frozenset(str(value) for value in control_embedding_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("embedding_ids must be nonempty and unique.")
        if not controls or not controls <= set(ids):
            raise ValueError("control_embedding_ids must be a nonempty subset of embedding_ids.")
        if context_mode not in {"none", "catalog_bank"}:
            raise ValueError("context_mode must be 'none' or 'catalog_bank'.")
        if min(latent_dim, embedding_dim, n_programs, hidden_dim, payoff_rank) < 1:
            raise ValueError("Model dimensions and payoff_rank must be positive.")
        if sigma_min <= 0 or growth_max <= 0:
            raise ValueError("sigma_min and growth_max must be positive.")
        self.embedding_ids = ids
        self.control_embedding_ids = controls
        self.noncontrol_embedding_ids = tuple(value for value in ids if value not in controls)
        self._residual_index = {
            value: index for index, value in enumerate(self.noncontrol_embedding_ids)
        }
        self.latent_dim = int(latent_dim)
        self.embedding_dim = int(embedding_dim)
        self.n_programs = int(n_programs)
        self.hidden_dim = int(hidden_dim)
        self.context_mode = context_mode
        self.sigma_min = float(sigma_min)
        self.growth_max = float(growth_max)
        self.payoff_rank = min(int(payoff_rank), self.n_programs)

        self.reference_embedding = nn.Parameter(torch.zeros(self.embedding_dim))
        residual = torch.empty(len(self.noncontrol_embedding_ids), self.embedding_dim)
        if residual.numel():
            nn.init.xavier_uniform_(residual)
        self.residual_embedding = nn.Parameter(residual)
        self.growth_bias = nn.Parameter(torch.zeros(len(self.noncontrol_embedding_ids)))

        self.program_encoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.n_programs),
        )
        self.time_embedding = _TimeEmbedding()
        trunk_input = self.latent_dim + self.time_embedding.output_dim
        self.trunk = _mlp(trunk_input, self.hidden_dim)

        self.drift_reference = nn.Linear(self.hidden_dim, self.latent_dim)
        self.drift_residual = nn.Linear(self.hidden_dim, self.latent_dim * self.embedding_dim)
        self.sigma_reference = nn.Linear(self.hidden_dim, self.latent_dim)
        self.sigma_residual = nn.Linear(self.hidden_dim, self.latent_dim * self.embedding_dim)
        self.growth_reference = nn.Linear(self.hidden_dim, 1)
        self.growth_residual = nn.Linear(self.hidden_dim, self.embedding_dim)

        self.payoff_reference_left = nn.Parameter(torch.empty(self.n_programs, self.payoff_rank))
        self.payoff_reference_right = nn.Parameter(torch.empty(self.payoff_rank, self.n_programs))
        self.payoff_residual_left = nn.Parameter(
            torch.zeros(self.embedding_dim, self.n_programs, self.payoff_rank)
        )
        self.payoff_residual_right = nn.Parameter(
            torch.zeros(self.embedding_dim, self.payoff_rank, self.n_programs)
        )
        nn.init.xavier_uniform_(self.payoff_reference_left)
        nn.init.xavier_uniform_(self.payoff_reference_right)
        nn.init.normal_(self.payoff_residual_left, std=0.01)
        nn.init.normal_(self.payoff_residual_right, std=0.01)

        self.growth_enabled = True
        self.context_enabled = context_mode == "catalog_bank"
        self.assert_soft_reference()

    def residuals(self, embedding_ids: Sequence[str]) -> torch.Tensor:
        """Return exact-zero control residuals and learned noncontrol residuals."""
        device = self.reference_embedding.device
        dtype = self.reference_embedding.dtype
        output = torch.zeros(len(embedding_ids), self.embedding_dim, device=device, dtype=dtype)
        for row, embedding_id in enumerate(embedding_ids):
            value = str(embedding_id)
            if value not in self.embedding_ids:
                raise KeyError(f"Unknown embedding_id {value!r}.")
            index = self._residual_index.get(value)
            if index is not None:
                output[row] = self.residual_embedding[index]
        return output

    def effective_embeddings(
        self,
        embedding_ids: Sequence[str],
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.residuals(embedding_ids)
        if residual_scale is not None:
            scale = residual_scale.to(device=residual.device, dtype=residual.dtype).reshape(-1, 1)
            if scale.shape[0] != residual.shape[0]:
                raise ValueError("residual_scale must have one value per measure.")
            residual = residual * scale
        return self.reference_embedding.unsqueeze(0) + residual

    def growth_intercepts(
        self,
        embedding_ids: Sequence[str],
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.reference_embedding.new_zeros(len(embedding_ids))
        for row, embedding_id in enumerate(embedding_ids):
            index = self._residual_index.get(str(embedding_id))
            if index is not None:
                output[row] = self.growth_bias[index]
        if residual_scale is not None:
            output = output * residual_scale.to(device=output.device, dtype=output.dtype)
        return output

    def programs(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.program_encoder(z), dim=-1)

    def summarize_context(
        self,
        z: torch.Tensor,
        absolute_log_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-measure log mass and within-measure program composition."""
        if absolute_log_weight.shape != z.shape[:2]:
            raise ValueError("absolute_log_weight must match the first two z dimensions.")
        programs = self.programs(z)
        absolute32 = absolute_log_weight.float()
        log_mass = torch.logsumexp(absolute32, dim=-1)
        normalized = torch.softmax(absolute32, dim=-1).to(dtype=z.dtype)
        mean_program = (normalized.unsqueeze(-1) * programs).sum(dim=1)
        return log_mass, mean_program

    def compose_context(
        self,
        log_mass: torch.Tensor,
        mean_program: torch.Tensor,
        context_group_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compose observation-derived ecological context within each group."""
        group = context_group_index.to(device=log_mass.device, dtype=torch.long)
        if group.ndim != 1 or len(group) != len(log_mass):
            raise ValueError("context_group_index must have one value per measure.")
        _, inverse = torch.unique(group, sorted=True, return_inverse=True)
        n_groups = int(inverse.max().item()) + 1
        context_by_group = mean_program.new_zeros(n_groups, self.n_programs)
        for index in range(n_groups):
            mask = inverse.eq(index)
            frequency = torch.softmax(log_mass[mask].float(), dim=0).to(mean_program.dtype)
            context_by_group[index] = (frequency.unsqueeze(-1) * mean_program[mask]).sum(0)
        return context_by_group[inverse]

    def _modulated_head(
        self,
        hidden: torch.Tensor,
        effective_embedding: torch.Tensor,
        reference_head: nn.Linear,
        residual_head: nn.Linear,
        output_dim: int,
    ) -> torch.Tensor:
        group_count, particle_count, _ = hidden.shape
        baseline = reference_head(hidden)
        matrix = residual_head(hidden).reshape(
            group_count, particle_count, output_dim, self.embedding_dim
        )
        modulation = torch.einsum("gnor,gr->gno", matrix, effective_embedding)
        return baseline + modulation

    def _ecological_growth(
        self,
        programs: torch.Tensor,
        effective_embedding: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        reference_payoff = self.payoff_reference_left @ self.payoff_reference_right
        residual_payoff = torch.einsum(
            "rkp,rpj->rkj", self.payoff_residual_left, self.payoff_residual_right
        )
        payoff = reference_payoff.unsqueeze(0) + torch.einsum(
            "gr,rkj->gkj", effective_embedding, residual_payoff
        )
        payoff_context = torch.einsum("gkj,gj->gk", payoff, context)
        return torch.einsum("gnk,gk->gn", programs, payoff_context)

    def forward(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        embedding_ids: Sequence[str],
        context: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
    ) -> DynamicsOutput:
        group_count, particle_count, _ = z.shape
        if len(embedding_ids) != group_count:
            raise ValueError("embedding_ids must have one value per measure.")
        if context is None:
            context = z.new_zeros(group_count, self.n_programs)
        elif context.ndim == 1:
            context = context.unsqueeze(0).expand(group_count, -1)
        elif context.shape != (group_count, self.n_programs):
            raise ValueError("context must have shape [n_measures, n_programs].")
        if not self.context_enabled:
            context = torch.zeros_like(context)

        time = self.time_embedding(tau.to(device=z.device, dtype=z.dtype))
        common = torch.cat(
            (z, time.reshape(1, 1, -1).expand(group_count, particle_count, -1)),
            dim=-1,
        )
        hidden = self.trunk(common)
        effective = self.effective_embeddings(embedding_ids, residual_scale).to(z)
        drift = self._modulated_head(
            hidden,
            effective,
            self.drift_reference,
            self.drift_residual,
            self.latent_dim,
        )
        sigma_raw = self._modulated_head(
            hidden,
            effective,
            self.sigma_reference,
            self.sigma_residual,
            self.latent_dim,
        )
        sigma_diag = F.softplus(sigma_raw) + self.sigma_min
        programs = self.programs(z)
        if self.growth_enabled:
            growth_raw = self.growth_reference(hidden).squeeze(-1)
            growth_raw = growth_raw + torch.einsum(
                "gnr,gr->gn", self.growth_residual(hidden), effective
            )
            growth_raw = growth_raw + self.growth_intercepts(
                embedding_ids, residual_scale
            ).unsqueeze(-1)
            if self.context_enabled:
                growth_raw = growth_raw + self._ecological_growth(programs, effective, context)
            growth = self.growth_max * torch.tanh(growth_raw)
        else:
            growth = z.new_zeros(group_count, particle_count)
        return DynamicsOutput(drift=drift, sigma_diag=sigma_diag, growth=growth, programs=programs)

    def set_phase(self, phase: Literal["state", "mass", "context"]) -> None:
        """Apply the fixed state, mass, or ecological continuation phase."""
        if phase not in {"state", "mass", "context"}:
            raise ValueError(f"Unknown training phase {phase!r}.")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.growth_enabled = phase != "state"
        self.context_enabled = phase == "context" and self.context_mode == "catalog_bank"
        if phase in {"state", "mass"}:
            for parameter in (
                self.reference_embedding,
                self.residual_embedding,
            ):
                parameter.requires_grad_(True)
            for module in (
                self.trunk,
                self.drift_reference,
                self.drift_residual,
                self.sigma_reference,
                self.sigma_residual,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if phase == "mass":
            self.growth_bias.requires_grad_(True)
            for module in (self.growth_reference, self.growth_residual):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if phase == "context":
            for parameter in self.program_encoder.parameters():
                parameter.requires_grad_(True)
            for parameter in (
                self.payoff_reference_left,
                self.payoff_reference_right,
                self.payoff_residual_left,
                self.payoff_residual_right,
            ):
                parameter.requires_grad_(True)

    def regularization(self) -> torch.Tensor:
        residual_penalty = (
            self.residual_embedding.square().mean()
            if self.residual_embedding.numel()
            else self.reference_embedding.new_zeros(())
        )
        return 1e-4 * (
            self.reference_embedding.square().mean()
            + residual_penalty
            + self.payoff_reference_left.square().mean()
            + self.payoff_reference_right.square().mean()
        )

    def assert_reference_branch(
        self,
        embedding_ids: Sequence[str],
        reference_mask: torch.Tensor,
    ) -> None:
        """Assert that a reference branch removes only selected residuals."""
        mask = reference_mask.to(device=self.reference_embedding.device, dtype=torch.bool)
        if mask.shape != (len(embedding_ids),):
            raise ValueError("reference_mask must have one value per embedding_id.")
        factual = self.effective_embeddings(embedding_ids)
        scale = (~mask).to(dtype=self.reference_embedding.dtype)
        reference = self.effective_embeddings(embedding_ids, scale)
        if not torch.equal(factual[~mask], reference[~mask]):
            raise AssertionError("Reference branch changed an unselected embedding.")
        expected = self.reference_embedding.unsqueeze(0).expand(int(mask.sum()), -1)
        if not torch.equal(reference[mask], expected):
            raise AssertionError("Reference branch did not remove selected residuals exactly.")

    def assert_soft_reference(self) -> None:
        controls = tuple(self.control_embedding_ids)
        residual = self.residuals(controls)
        if not torch.equal(residual, torch.zeros_like(residual)):
            raise AssertionError("Control residuals must be exactly zero.")
        effective = self.effective_embeddings(controls)
        expected = self.reference_embedding.unsqueeze(0).expand_as(effective)
        if not torch.equal(effective, expected):
            raise AssertionError("All controls must share one reference embedding.")

    def architecture(self) -> dict[str, object]:
        return {
            "embedding_ids": list(self.embedding_ids),
            "control_embedding_ids": sorted(self.control_embedding_ids),
            "latent_dim": self.latent_dim,
            "embedding_dim": self.embedding_dim,
            "n_programs": self.n_programs,
            "hidden_dim": self.hidden_dim,
            "context_mode": self.context_mode,
            "sigma_min": self.sigma_min,
            "growth_max": self.growth_max,
            "payoff_rank": self.payoff_rank,
        }
