"""The single checkpoint, mass, count-block, and rollout objective."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .contracts import TrajectoryData
from .particles import ParticleRollout, checkpoint_indices, weight_diagnostics


def _owned_tensor(value: torch.Tensor | np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(np.asarray(value).copy(), dtype=dtype)


@dataclass(frozen=True)
class CountBlock:
    """One complete compositional count denominator for a group and time."""

    context_group_id: str
    time_label: str
    measure_indices: torch.Tensor | np.ndarray
    exposure: torch.Tensor | np.ndarray
    counts: torch.Tensor | np.ndarray

    def __post_init__(self) -> None:
        indices = _owned_tensor(self.measure_indices, torch.long).reshape(-1)
        exposure = _owned_tensor(self.exposure, torch.float32).reshape(-1)
        counts = _owned_tensor(self.counts, torch.float32).reshape(-1)
        object.__setattr__(self, "context_group_id", str(self.context_group_id))
        object.__setattr__(self, "time_label", str(self.time_label))
        object.__setattr__(self, "measure_indices", indices)
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "counts", counts)
        if len(indices) < 1 or not (len(indices) == len(exposure) == len(counts)):
            raise ValueError("CountBlock arrays must be nonempty and equally sized.")
        if torch.any(indices < 0) or len(torch.unique(indices)) != len(indices):
            raise ValueError("CountBlock measure_indices must be unique and nonnegative.")
        if not torch.isfinite(exposure).all() or torch.any(exposure <= 0):
            raise ValueError("CountBlock exposures must be positive and finite.")
        if not torch.isfinite(counts).all() or torch.any(counts < 0):
            raise ValueError("CountBlock counts must be nonnegative and finite.")
        if not torch.equal(counts, counts.round()):
            raise ValueError("CountBlock counts must be integer-like.")
        if counts.sum() <= 0:
            raise ValueError("CountBlock must contain at least one observed count.")

    @property
    def n_total(self) -> torch.Tensor:
        return self.counts.sum()


@dataclass
class CheckpointObjective:
    total: torch.Tensor
    geometry: torch.Tensor
    log_mass_error: torch.Tensor
    observation_count: int
    rows: list[dict[str, Any]]


@dataclass
class ObjectiveResult:
    total: torch.Tensor
    checkpoint: CheckpointObjective
    count: torch.Tensor
    regularization: torch.Tensor


def _pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        x.square().sum(-1, keepdim=True) + y.square().sum(-1).unsqueeze(0) - 2 * x @ y.T
    ).clamp_min(0)


def _sinkhorn_cost(
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    kernel = -cost / epsilon
    log_u = log_a
    log_v = log_b
    for _ in range(iterations):
        log_v = log_b - torch.logsumexp(kernel + log_u[:, None], dim=0)
        log_u = log_a - torch.logsumexp(kernel + log_v[None, :], dim=1)
    transport = torch.exp(log_u[:, None] + log_v[None, :] + kernel)
    return (transport * cost).sum()


def checkpoint_geometry(
    predicted_support: torch.Tensor,
    predicted_log_weight: torch.Tensor,
    observed_support: torch.Tensor,
    observed_log_weight: torch.Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 80,
) -> torch.Tensor:
    """Debiased normalized Sinkhorn geometry at one checkpoint."""
    predicted_probability = torch.softmax(predicted_log_weight, dim=0)
    observed_probability = torch.softmax(observed_log_weight, dim=0)
    try:
        from geomloss import SamplesLoss

        loss = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=epsilon**0.5,
            debias=True,
            backend="tensorized",
        )
        return loss(
            predicted_probability.unsqueeze(0),
            predicted_support.unsqueeze(0),
            observed_probability.unsqueeze(0),
            observed_support.unsqueeze(0),
        ).squeeze()
    except ImportError:
        log_a = torch.log(predicted_probability.clamp_min(1e-30))
        log_b = torch.log(observed_probability.clamp_min(1e-30))
        cross = 0.5 * _pairwise_squared_distance(predicted_support, observed_support)
        self_a = 0.5 * _pairwise_squared_distance(predicted_support, predicted_support)
        self_b = 0.5 * _pairwise_squared_distance(observed_support, observed_support)
        value = _sinkhorn_cost(log_a, log_b, cross, epsilon, iterations)
        value = value - 0.5 * _sinkhorn_cost(log_a, log_a, self_a, epsilon, iterations)
        value = value - 0.5 * _sinkhorn_cost(log_b, log_b, self_b, epsilon, iterations)
        return value.clamp_min(0)


def checkpoint_log_mass_error(
    predicted_log_weight: torch.Tensor,
    observed_log_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted = torch.logsumexp(predicted_log_weight.float(), dim=0)
    observed = torch.logsumexp(observed_log_weight.float(), dim=0)
    return (predicted - observed).square(), predicted, observed


def checkpoint_geometry_mass_loss(
    rollout: ParticleRollout,
    data: TrajectoryData,
    *,
    mass_weight: float,
    include_mass: bool,
    validation_source: str,
) -> CheckpointObjective:
    """Apply one checkpoint objective to any number of observed times."""
    indices = checkpoint_indices(data.axis, rollout.axis_grid)
    device = rollout.z_steps.device
    dtype = rollout.z_steps.dtype
    geometry_sum = torch.zeros((), device=device, dtype=dtype)
    mass_sum = torch.zeros((), device=device, dtype=dtype)
    rows: list[dict[str, Any]] = []
    observation_count = 0
    terminal_diagnostics = weight_diagnostics(rollout.logw_steps)
    for label in data.axis.labels[1:]:
        step = indices[label]
        for local_index, measure_id in enumerate(rollout.measure_ids):
            target = data.measures[label].get(measure_id)
            if target is None:
                continue
            target_support = torch.as_tensor(target.support, device=device, dtype=dtype)
            target_log_weight = torch.log(
                torch.as_tensor(target.weights, device=device, dtype=dtype).clamp_min(1e-30)
            )
            predicted_support = rollout.z_steps[step, local_index]
            predicted_log_weight = rollout.absolute_log_weight_steps[step, local_index]
            geometry = checkpoint_geometry(
                predicted_support,
                predicted_log_weight,
                target_support,
                target_log_weight,
            )
            mass_error, predicted_mass, observed_mass = checkpoint_log_mass_error(
                predicted_log_weight, target_log_weight
            )
            geometry_sum = geometry_sum + geometry
            mass_sum = mass_sum + mass_error
            observation_count += 1
            rows.append(
                {
                    "measure_id": measure_id,
                    "time_label": label,
                    "endpoint_role": "observed_checkpoint",
                    "validation_source": validation_source,
                    "geometry": float(geometry.detach().cpu()),
                    "log_mass_error": float(mass_error.detach().cpu()),
                    "predicted_log_mass": float(predicted_mass.detach().cpu()),
                    "observed_log_mass": float(observed_mass.detach().cpu()),
                    "ess_fraction": float(
                        terminal_diagnostics["ess_fraction"][step, local_index].detach().cpu()
                    ),
                    "max_weight_fraction": float(
                        terminal_diagnostics["max_weight_fraction"][step, local_index]
                        .detach()
                        .cpu()
                    ),
                }
            )
    if observation_count == 0:
        zero = rollout.z_steps.new_zeros(())
        return CheckpointObjective(
            total=zero,
            geometry=zero,
            log_mass_error=zero,
            observation_count=0,
            rows=[],
        )
    geometry_mean = geometry_sum / observation_count
    mass_mean = mass_sum / observation_count
    total = geometry_mean + (float(mass_weight) * mass_mean if include_mass else 0.0)
    return CheckpointObjective(
        total=total,
        geometry=geometry_mean,
        log_mass_error=mass_mean,
        observation_count=observation_count,
        rows=rows,
    )


def integrated_fitness_curve(rollout: ParticleRollout) -> torch.Tensor:
    """Cumulative conditional mean growth at every rollout grid point."""
    normalized = torch.softmax(rollout.logw_steps[:-1].float(), dim=-1).to(
        rollout.growth_steps.dtype
    )
    mean_growth = (normalized * rollout.growth_steps).sum(dim=-1)
    dt = rollout.axis_grid[1:] - rollout.axis_grid[:-1]
    increments = mean_growth * dt[:, None]
    zero = increments.new_zeros(1, increments.shape[1])
    return torch.cat((zero, torch.cumsum(increments, dim=0)), dim=0)


class FitnessBankProtocol:
    def fitness_for_active(
        self,
        *,
        time_label: str,
        active_indices: torch.Tensor,
        active_fitness: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def full_fitness(self, *, time_label: str) -> torch.Tensor:
        raise NotImplementedError


def _dirichlet_multinomial_loss(
    counts: torch.Tensor,
    probability: torch.Tensor,
    concentration: torch.Tensor,
) -> torch.Tensor:
    total = counts.sum()
    phi = concentration.exp().clamp_min(1e-4)
    alpha = phi * probability.clamp_min(1e-8)
    constant = torch.lgamma(total + 1) - torch.lgamma(counts + 1).sum()
    likelihood = constant + torch.lgamma(phi) - torch.lgamma(phi + total)
    likelihood = likelihood + (torch.lgamma(alpha + counts) - torch.lgamma(alpha)).sum()
    return -likelihood


def _count_block_nll(
    block: CountBlock,
    fitness: torch.Tensor,
    log_concentration: torch.Tensor,
) -> torch.Tensor:
    exposure = block.exposure.to(device=fitness.device, dtype=fitness.dtype)
    counts = block.counts.to(device=fitness.device, dtype=fitness.dtype)
    probability = torch.softmax(torch.log(exposure) + fitness, dim=0)
    return _dirichlet_multinomial_loss(counts, probability, log_concentration)


def count_block_loss(
    rollout: ParticleRollout,
    data: TrajectoryData,
    *,
    log_concentration: torch.Tensor,
    fitness_bank: FitnessBankProtocol | None = None,
) -> torch.Tensor:
    """Evaluate the only count API over complete grouped blocks."""
    blocks: Iterable[CountBlock] = data.count_blocks
    if not blocks:
        return rollout.z_steps.new_zeros(())
    data.axis.require_physical("Count likelihood")
    active_groups = set(rollout.context_group_ids)
    blocks = tuple(block for block in blocks if block.context_group_id in active_groups)
    if not blocks:
        return rollout.z_steps.new_zeros(())
    curve = integrated_fitness_curve(rollout)
    indices = checkpoint_indices(data.axis, rollout.axis_grid)
    active_lookup = {
        int(global_index): local_index
        for local_index, global_index in enumerate(rollout.measure_indices.tolist())
    }
    losses = []
    for block in blocks:
        if block.time_label not in indices:
            raise ValueError(f"CountBlock time {block.time_label!r} is not an axis checkpoint.")
        block_indices = block.measure_indices.to(device=curve.device, dtype=torch.long)
        active_fitness = curve[indices[block.time_label]]
        if fitness_bank is None:
            missing = [value for value in block_indices.tolist() if value not in active_lookup]
            if missing:
                raise ValueError("A partial count denominator requires a complete CatalogBank.")
            local = torch.tensor(
                [active_lookup[value] for value in block_indices.tolist()],
                device=curve.device,
                dtype=torch.long,
            )
            fitness = active_fitness.index_select(0, local)
        else:
            full = fitness_bank.fitness_for_active(
                time_label=block.time_label,
                active_indices=rollout.measure_indices,
                active_fitness=active_fitness,
            )
            fitness = full.index_select(0, block_indices)
        losses.append(_count_block_nll(block, fitness, log_concentration))
    return torch.stack(losses).mean()


def catalog_count_block_loss(
    data: TrajectoryData,
    *,
    log_concentration: torch.Tensor,
    fitness_bank: FitnessBankProtocol,
    context_group_ids: Iterable[str] | None = None,
) -> tuple[torch.Tensor, int]:
    """Score complete count blocks from a refreshed detached catalog bank."""
    data.axis.require_physical("Count likelihood")
    selected_groups = None if context_group_ids is None else set(context_group_ids)
    blocks = tuple(
        block
        for block in data.count_blocks
        if selected_groups is None or block.context_group_id in selected_groups
    )
    if not blocks:
        return log_concentration.new_zeros(()), 0
    losses = []
    for block in blocks:
        full_fitness = fitness_bank.full_fitness(time_label=block.time_label)
        indices = block.measure_indices.to(device=full_fitness.device, dtype=torch.long)
        losses.append(
            _count_block_nll(
                block,
                full_fitness.index_select(0, indices),
                log_concentration,
            )
        )
    return torch.stack(losses).mean(), len(blocks)


def validate_count_blocks(data: TrajectoryData) -> None:
    """Require every count denominator to contain all source-supported group measures."""
    if not data.count_blocks:
        return
    groups = data.measure_meta.groupby("context_group_id", observed=True).indices
    expected = {
        str(group): set(int(index) for index in indices) for group, indices in groups.items()
    }
    seen: set[tuple[str, str]] = set()
    for block in data.count_blocks:
        data.axis.index(block.time_label)
        block_key = (block.context_group_id, block.time_label)
        if block_key in seen:
            raise ValueError(f"Duplicate CountBlock for context/time {block_key!r}.")
        seen.add(block_key)
        if block.context_group_id not in expected:
            raise ValueError(f"Unknown CountBlock context group {block.context_group_id!r}.")
        observed = set(int(value) for value in block.measure_indices.tolist())
        if observed != expected[block.context_group_id]:
            raise ValueError(
                f"CountBlock {block.context_group_id!r}/{block.time_label!r} "
                "does not contain the complete source-supported denominator."
            )


def rollout_regularization(rollout: ParticleRollout) -> torch.Tensor:
    """Small auditable action penalty over generated coefficients."""
    return (
        1e-4 * rollout.drift_steps.square().mean()
        + 1e-5 * rollout.sigma_steps.square().mean()
        + 1e-4 * rollout.growth_steps.square().mean()
    )


def total_objective(
    rollout: ParticleRollout,
    data: TrajectoryData,
    *,
    mass_weight: float,
    count_weight: float,
    include_mass: bool,
    log_concentration: torch.Tensor,
    fitness_bank: FitnessBankProtocol | None = None,
    validation_source: str = "train_self_eval",
) -> ObjectiveResult:
    checkpoint = checkpoint_geometry_mass_loss(
        rollout,
        data,
        mass_weight=mass_weight,
        include_mass=include_mass,
        validation_source=validation_source,
    )
    counts = (
        count_block_loss(
            rollout,
            data,
            log_concentration=log_concentration,
            fitness_bank=fitness_bank,
        )
        if include_mass and count_weight > 0
        else rollout.z_steps.new_zeros(())
    )
    regularization = rollout_regularization(rollout)
    return ObjectiveResult(
        total=checkpoint.total + float(count_weight) * counts + regularization,
        checkpoint=checkpoint,
        count=counts,
        regularization=regularization,
    )
