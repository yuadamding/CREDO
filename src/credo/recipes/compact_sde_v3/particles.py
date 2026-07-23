"""Particle initialization, the one rollout, typed context, and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from ...contracts import Axis, TrajectoryData
from .model import CREDOModel


@dataclass(frozen=True)
class ParticleState:
    z: torch.Tensor
    logw: torch.Tensor
    log_m0: torch.Tensor
    measure_ids: tuple[str, ...]
    embedding_ids: tuple[str, ...]
    context_group_ids: tuple[str, ...]
    measure_indices: torch.Tensor
    residual_scale: torch.Tensor

    def __post_init__(self) -> None:
        group_count = self.z.shape[0]
        if self.z.ndim != 3 or self.logw.shape != self.z.shape[:2]:
            raise ValueError("ParticleState requires z [G,N,d] and logw [G,N].")
        if self.log_m0.shape != (group_count,):
            raise ValueError("ParticleState.log_m0 must have shape [G].")
        if any(
            len(values) != group_count
            for values in (self.measure_ids, self.embedding_ids, self.context_group_ids)
        ):
            raise ValueError("ParticleState identifiers must have one value per measure.")
        if self.measure_indices.shape != (group_count,):
            raise ValueError("ParticleState.measure_indices must have shape [G].")
        if self.residual_scale.shape != (group_count,):
            raise ValueError("ParticleState.residual_scale must have shape [G].")

    @property
    def absolute_log_weight(self) -> torch.Tensor:
        return self.log_m0[:, None] + self.logw

    def with_residual_scale(self, residual_scale: torch.Tensor) -> ParticleState:
        return ParticleState(
            z=self.z,
            logw=self.logw,
            log_m0=self.log_m0,
            measure_ids=self.measure_ids,
            embedding_ids=self.embedding_ids,
            context_group_ids=self.context_group_ids,
            measure_indices=self.measure_indices,
            residual_scale=residual_scale.to(device=self.z.device, dtype=self.z.dtype),
        )


@dataclass(frozen=True)
class ParticleRollout:
    z_steps: torch.Tensor
    logw_steps: torch.Tensor
    log_m0: torch.Tensor
    axis_grid: torch.Tensor
    measure_ids: tuple[str, ...]
    embedding_ids: tuple[str, ...]
    context_group_ids: tuple[str, ...]
    measure_indices: torch.Tensor
    residual_scale: torch.Tensor
    drift_steps: torch.Tensor
    sigma_steps: torch.Tensor
    growth_steps: torch.Tensor
    context_steps: torch.Tensor
    noise_steps: torch.Tensor

    @property
    def absolute_log_weight_steps(self) -> torch.Tensor:
        return self.log_m0[None, :, None] + self.logw_steps

    @property
    def terminal_z(self) -> torch.Tensor:
        return self.z_steps[-1]

    @property
    def terminal_absolute_log_weight(self) -> torch.Tensor:
        return self.absolute_log_weight_steps[-1]

    @property
    def diffusion_steps(self) -> torch.Tensor:
        return self.sigma_steps

    def slice_measure(self, index: int) -> ParticleRollout:
        item = slice(index, index + 1)
        return ParticleRollout(
            z_steps=self.z_steps[:, item],
            logw_steps=self.logw_steps[:, item],
            log_m0=self.log_m0[item],
            axis_grid=self.axis_grid,
            measure_ids=(self.measure_ids[index],),
            embedding_ids=(self.embedding_ids[index],),
            context_group_ids=(self.context_group_ids[index],),
            measure_indices=self.measure_indices[item],
            residual_scale=self.residual_scale[item],
            drift_steps=self.drift_steps[:, item],
            sigma_steps=self.sigma_steps[:, item],
            growth_steps=self.growth_steps[:, item],
            context_steps=(
                self.context_steps[:, item] if self.context_steps.ndim >= 3 else self.context_steps
            ),
            noise_steps=self.noise_steps[:, item],
        )


RolloutResult = ParticleRollout


class ContextProvider(Protocol):
    requires_complete_catalog: bool
    requires_full_group_rollout: bool

    def context(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        absolute_log_weight: torch.Tensor,
        state: ParticleState,
        model: CREDOModel,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class DynamicsStep:
    """Recipe-neutral coefficients and context for one integration step.

    Context is either a global vector ``[C]`` or a measure-indexed tensor whose
    leading dimension is ``G``. The rollout preserves that distinction.
    """

    drift: torch.Tensor
    sigma_diag: torch.Tensor
    growth: torch.Tensor
    context: torch.Tensor


class DynamicsKernel(Protocol):
    """Adapter boundary between a recipe model and the common EM driver."""

    def step(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        logw: torch.Tensor,
        time: torch.Tensor,
        state: ParticleState,
    ) -> DynamicsStep: ...


def _group_index(values: Sequence[str], device: torch.device) -> torch.Tensor:
    mapping: dict[str, int] = {}
    indices = []
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        indices.append(mapping[value])
    return torch.tensor(indices, device=device, dtype=torch.long)


class NoContextProvider:
    requires_complete_catalog = False
    requires_full_group_rollout = False

    def context(self, **kwargs: Any) -> torch.Tensor:
        z = kwargs["z"]
        model = kwargs["model"]
        return z.new_zeros(z.shape[0], model.n_programs)


class SelfConsistentContextProvider:
    """Compute context from every currently rolled-out measure in each group."""

    requires_complete_catalog = False
    requires_full_group_rollout = True

    def context(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        absolute_log_weight: torch.Tensor,
        state: ParticleState,
        model: CREDOModel,
    ) -> torch.Tensor:
        del step_index
        log_mass, programs = model.summarize_context(z, absolute_log_weight)
        groups = _group_index(state.context_group_ids, z.device)
        return model.compose_context(log_mass, programs, groups)


class CatalogBankProtocol(Protocol):
    def context_for_active(
        self,
        *,
        step_index: int,
        active_indices: torch.Tensor,
        active_log_mass: torch.Tensor,
        active_programs: torch.Tensor,
        model: CREDOModel,
    ) -> torch.Tensor: ...


class CatalogContextProvider:
    requires_complete_catalog = True
    requires_full_group_rollout = False

    def __init__(self, bank: CatalogBankProtocol) -> None:
        self.bank = bank

    def context(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        absolute_log_weight: torch.Tensor,
        state: ParticleState,
        model: CREDOModel,
    ) -> torch.Tensor:
        log_mass, programs = model.summarize_context(z, absolute_log_weight)
        return self.bank.context_for_active(
            step_index=step_index,
            active_indices=state.measure_indices,
            active_log_mass=log_mass,
            active_programs=programs,
            model=model,
        )


class ClampedContextProvider:
    requires_complete_catalog = False
    requires_full_group_rollout = False

    def __init__(self, context_steps: torch.Tensor) -> None:
        if context_steps.ndim != 3:
            raise ValueError("Clamped context must have shape [steps, measures, programs].")
        self.context_steps = context_steps

    def context(self, *, step_index: int, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        context = self.context_steps[step_index].to(device=z.device, dtype=z.dtype)
        if context.shape[0] != z.shape[0]:
            raise ValueError("Clamped context measure dimension does not match the rollout.")
        return context


def sample_initial_particles(
    data: TrajectoryData,
    measure_ids: Sequence[str] | None,
    n_particles: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> ParticleState:
    """Sample source particles according to finite-measure atom weights."""
    ids = tuple(data.measure_ids if measure_ids is None else (str(value) for value in measure_ids))
    if n_particles < 2 or not ids:
        raise ValueError("Sampling requires at least one measure and two particles.")
    source = data.measures[data.axis.source]
    unknown = set(ids) - set(source)
    if unknown:
        raise KeyError(f"Unknown source measure_ids: {sorted(unknown)[:5]}")
    metadata = data.measure_meta.set_index("measure_id")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    latent_dim = data.latent_dim
    z = torch.empty(len(ids), n_particles, latent_dim, device=device, dtype=dtype)
    logw = torch.full(
        (len(ids), n_particles),
        -float(np.log(n_particles)),
        device=device,
        dtype=dtype,
    )
    log_m0 = torch.empty(len(ids), device=device, dtype=dtype)
    for row, measure_id in enumerate(ids):
        measure = source[measure_id]
        support = torch.as_tensor(measure.support, device=device, dtype=dtype)
        probability = torch.as_tensor(measure.normalized_weights, device=device, dtype=dtype)
        selected = torch.multinomial(
            probability, n_particles, replacement=True, generator=generator
        )
        z[row] = support[selected]
        log_m0[row] = float(np.log(measure.total_mass))
    global_index = {measure_id: index for index, measure_id in enumerate(data.measure_ids)}
    return ParticleState(
        z=z,
        logw=logw,
        log_m0=log_m0,
        measure_ids=ids,
        embedding_ids=tuple(metadata.loc[value, "embedding_id"] for value in ids),
        context_group_ids=tuple(metadata.loc[value, "context_group_id"] for value in ids),
        measure_indices=torch.tensor(
            [global_index[value] for value in ids], device=device, dtype=torch.long
        ),
        residual_scale=torch.ones(len(ids), device=device, dtype=dtype),
    )


def axis_grid(
    axis: Axis,
    steps_per_interval: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps_per_interval < 1:
        raise ValueError("steps_per_interval must be positive.")
    checkpoints = axis.normalized_values
    values = [checkpoints[0]]
    for left, right in zip(checkpoints[:-1], checkpoints[1:], strict=False):
        interval = np.linspace(left, right, steps_per_interval + 1)[1:]
        values.extend(float(value) for value in interval)
    return torch.tensor(values, device=device, dtype=dtype)


def checkpoint_indices(axis: Axis, grid: torch.Tensor) -> dict[str, int]:
    output: dict[str, int] = {}
    for label, value in zip(axis.labels, axis.normalized_values, strict=False):
        difference = torch.abs(grid - float(value))
        index = int(torch.argmin(difference).item())
        if not torch.isclose(grid[index], grid.new_tensor(value), atol=1e-6, rtol=0):
            raise ValueError(f"Axis grid does not contain checkpoint {label!r}.")
        output[label] = index
    return output


def sample_noise(
    state: ParticleState,
    grid: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=state.z.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        (len(grid) - 1,) + tuple(state.z.shape),
        device=state.z.device,
        dtype=state.z.dtype,
        generator=generator,
    )


@dataclass(frozen=True)
class CompactV3Kernel:
    model: CREDOModel
    context_provider: ContextProvider

    def step(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        logw: torch.Tensor,
        time: torch.Tensor,
        state: ParticleState,
    ) -> DynamicsStep:
        absolute_log_weight = state.log_m0[:, None] + logw
        context = self.context_provider.context(
            step_index=step_index,
            z=z,
            absolute_log_weight=absolute_log_weight,
            state=state,
            model=self.model,
        )
        output = self.model(
            z,
            time,
            state.embedding_ids,
            context,
            state.residual_scale,
        )
        return DynamicsStep(
            drift=output.drift,
            sigma_diag=output.sigma_diag,
            growth=output.growth,
            context=context,
        )


def euler_maruyama_rollout(
    kernel: DynamicsKernel,
    initial_state: ParticleState,
    axis_grid: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
) -> ParticleRollout:
    """Execute the common state, weight, noise, and capture loop."""
    grid = axis_grid.to(device=initial_state.z.device, dtype=initial_state.z.dtype)
    if grid.ndim != 1 or len(grid) < 2 or not torch.all(grid[1:] > grid[:-1]):
        raise ValueError("axis_grid must be a strictly increasing one-dimensional tensor.")
    if noise is None:
        noise = sample_noise(initial_state, grid, seed=0)
    noise = noise.to(device=initial_state.z.device, dtype=initial_state.z.dtype)
    expected = (len(grid) - 1,) + tuple(initial_state.z.shape)
    if tuple(noise.shape) != expected:
        raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}.")

    z = initial_state.z.clone()
    logw = initial_state.logw.clone()
    z_steps = [z]
    logw_steps = [logw]
    drift_steps = []
    sigma_steps = []
    growth_steps = []
    context_steps = []
    for step_index in range(len(grid) - 1):
        output = kernel.step(
            step_index=step_index,
            z=z,
            logw=logw,
            time=grid[step_index],
            state=initial_state,
        )
        if tuple(output.drift.shape) != tuple(z.shape):
            raise ValueError("DynamicsKernel drift must match the particle-state shape.")
        if tuple(output.sigma_diag.shape) != tuple(z.shape):
            raise ValueError("DynamicsKernel diffusion must match the particle-state shape.")
        if tuple(output.growth.shape) != tuple(logw.shape):
            raise ValueError("DynamicsKernel growth must match the particle-weight shape.")
        if output.context.ndim == 0 or (
            output.context.ndim >= 2 and output.context.shape[0] != z.shape[0]
        ):
            raise ValueError(
                "DynamicsKernel context must be a global vector or have one leading "
                "row per measure."
            )
        dt = grid[step_index + 1] - grid[step_index]
        z = z + output.drift * dt + output.sigma_diag * torch.sqrt(dt) * noise[step_index]
        logw = logw + output.growth * dt
        z_steps.append(z)
        logw_steps.append(logw)
        drift_steps.append(output.drift)
        sigma_steps.append(output.sigma_diag)
        growth_steps.append(output.growth)
        context_steps.append(output.context)
    return ParticleRollout(
        z_steps=torch.stack(z_steps),
        logw_steps=torch.stack(logw_steps),
        log_m0=initial_state.log_m0,
        axis_grid=axis_grid.to(device=initial_state.z.device, dtype=torch.float32),
        measure_ids=initial_state.measure_ids,
        embedding_ids=initial_state.embedding_ids,
        context_group_ids=initial_state.context_group_ids,
        measure_indices=initial_state.measure_indices,
        residual_scale=initial_state.residual_scale,
        drift_steps=torch.stack(drift_steps),
        sigma_steps=torch.stack(sigma_steps),
        growth_steps=torch.stack(growth_steps),
        context_steps=torch.stack(context_steps),
        noise_steps=noise,
    )


def rollout(
    model: CREDOModel,
    initial_state: ParticleState,
    axis_grid: torch.Tensor,
    *,
    context_provider: ContextProvider | None = None,
    noise: torch.Tensor | None = None,
) -> ParticleRollout:
    """Run compact-v3 through the common Euler-Maruyama driver."""
    provider: ContextProvider = context_provider or (
        NoContextProvider() if model.context_mode == "none" else SelfConsistentContextProvider()
    )
    return euler_maruyama_rollout(
        CompactV3Kernel(model, provider),
        initial_state,
        axis_grid,
        noise=noise,
    )


def weight_diagnostics(logw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-measure conditional particle-weight diagnostics."""
    normalized = torch.softmax(logw.float(), dim=-1)
    ess = 1.0 / normalized.square().sum(dim=-1)
    return {
        "ess": ess,
        "ess_fraction": ess / float(logw.shape[-1]),
        "max_weight_fraction": normalized.max(dim=-1).values,
        "log_weight_range": logw.float().max(dim=-1).values - logw.float().min(dim=-1).values,
    }
