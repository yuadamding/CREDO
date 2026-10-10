"""The single checkpoint, mass, count-block, and rollout objective."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ...contracts import TrajectoryData
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
    background_series_ids: tuple[str, ...] = ()
    background_fitness: torch.Tensor | np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    background_exposure: torch.Tensor | np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    background_counts: torch.Tensor | np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    source_denominator_id: str | None = None
    modeled_denominator_id: str | None = None
    conditioning_policy: str = "require_complete"

    def __post_init__(self) -> None:
        indices = _owned_tensor(self.measure_indices, torch.long).reshape(-1)
        exposure = _owned_tensor(self.exposure, torch.float32).reshape(-1)
        counts = _owned_tensor(self.counts, torch.float32).reshape(-1)
        background_ids = tuple(str(value) for value in self.background_series_ids)
        background_fitness = _owned_tensor(self.background_fitness, torch.float32).reshape(-1)
        background_exposure = _owned_tensor(self.background_exposure, torch.float32).reshape(-1)
        background_counts = _owned_tensor(self.background_counts, torch.float32).reshape(-1)
        object.__setattr__(self, "context_group_id", str(self.context_group_id))
        object.__setattr__(self, "time_label", str(self.time_label))
        object.__setattr__(self, "measure_indices", indices)
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "background_series_ids", background_ids)
        object.__setattr__(self, "background_fitness", background_fitness)
        object.__setattr__(self, "background_exposure", background_exposure)
        object.__setattr__(self, "background_counts", background_counts)
        source_denominator = (
            None if self.source_denominator_id is None else str(self.source_denominator_id)
        )
        modeled_denominator = (
            source_denominator
            if self.modeled_denominator_id is None
            else str(self.modeled_denominator_id)
        )
        object.__setattr__(self, "source_denominator_id", source_denominator)
        object.__setattr__(self, "modeled_denominator_id", modeled_denominator)
        object.__setattr__(self, "conditioning_policy", str(self.conditioning_policy))
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
            if background_counts.sum() <= 0:
                raise ValueError("CountBlock must contain at least one observed count.")
        if not (
            len(background_ids)
            == len(background_fitness)
            == len(background_exposure)
            == len(background_counts)
        ):
            raise ValueError("CountBlock background arrays and IDs must have equal length.")
        if len(background_ids) != len(set(background_ids)):
            raise ValueError("CountBlock background_series_ids must be unique.")
        if not torch.isfinite(background_fitness).all():
            raise ValueError("CountBlock background fitness must be finite.")
        if len(background_ids) and (
            not torch.isfinite(background_exposure).all() or torch.any(background_exposure <= 0)
        ):
            raise ValueError("CountBlock background exposures must be positive and finite.")
        if (
            not torch.isfinite(background_counts).all()
            or torch.any(background_counts < 0)
            or not torch.equal(background_counts, background_counts.round())
        ):
            raise ValueError("CountBlock background counts must be nonnegative integers.")
        if self.conditioning_policy not in {
            "require_complete",
            "preserve_background",
            "condition_on_selection",
        }:
            raise ValueError("CountBlock has an unknown conditioning policy.")
        if self.conditioning_policy == "condition_on_selection" and (
            source_denominator is None or modeled_denominator == source_denominator
        ):
            raise ValueError("Conditioned CountBlock requires a distinct modeled denominator.")

    @property
    def n_total(self) -> torch.Tensor:
        return self.counts.sum() + self.background_counts.sum()


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


def _center_supports(
    predicted_support: torch.Tensor,
    observed_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove a shared origin before fp32 distance calculations."""
    if predicted_support.shape[0] == 0 or observed_support.shape[0] == 0:
        raise ValueError("Checkpoint supports must contain at least one atom.")
    origin = predicted_support[0].detach()
    return predicted_support - origin, observed_support - origin


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
    stable_distances: bool = False,
    backend_observer=None,
) -> torch.Tensor:
    """Debiased normalized Sinkhorn geometry at one checkpoint."""
    predicted_probability = torch.softmax(predicted_log_weight, dim=0)
    observed_probability = torch.softmax(observed_log_weight, dim=0)
    geometry_predicted, geometry_observed = (
        _center_supports(predicted_support, observed_support)
        if stable_distances
        else (predicted_support, observed_support)
    )
    try:
        from geomloss import SamplesLoss

        loss = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=epsilon**0.5,
            debias=True,
            backend="tensorized",
        )
        value = loss(
            predicted_probability.unsqueeze(0),
            geometry_predicted.unsqueeze(0),
            observed_probability.unsqueeze(0),
            geometry_observed.unsqueeze(0),
        ).squeeze()
        if backend_observer is not None:
            backend_observer("geomloss_tensorized")
        return value
    except ImportError:
        log_a = torch.log(predicted_probability.clamp_min(1e-30))
        log_b = torch.log(observed_probability.clamp_min(1e-30))
        cross = 0.5 * _pairwise_squared_distance(
            geometry_predicted,
            geometry_observed,
        )
        self_a = 0.5 * _pairwise_squared_distance(
            geometry_predicted,
            geometry_predicted,
        )
        self_b = 0.5 * _pairwise_squared_distance(
            geometry_observed,
            geometry_observed,
        )
        value = _sinkhorn_cost(log_a, log_b, cross, epsilon, iterations)
        value = value - 0.5 * _sinkhorn_cost(log_a, log_a, self_a, epsilon, iterations)
        value = value - 0.5 * _sinkhorn_cost(log_b, log_b, self_b, epsilon, iterations)
        if backend_observer is not None:
            backend_observer(f"frozen_{iterations}_iteration_fallback")
        return value.clamp_min(0)


def checkpoint_unbalanced_sinkhorn(
    predicted_support: torch.Tensor,
    predicted_log_weight: torch.Tensor,
    observed_support: torch.Tensor,
    observed_log_weight: torch.Tensor,
    *,
    epsilon: float = 0.1,
    reach: float = 1.0,
) -> torch.Tensor:
    """KL-relaxed Sinkhorn divergence in the measures' declared mass units."""
    if epsilon <= 0:
        raise ValueError("Unbalanced Sinkhorn epsilon must be positive.")
    if reach <= 0:
        raise ValueError("Unbalanced Sinkhorn reach must be positive.")
    from geomloss import SamplesLoss

    predicted_weight = torch.exp(predicted_log_weight.float()).to(predicted_support.dtype)
    observed_weight = torch.exp(observed_log_weight.float()).to(observed_support.dtype)
    centered_predicted, centered_observed = _center_supports(
        predicted_support,
        observed_support,
    )
    loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=epsilon**0.5,
        reach=reach,
        debias=True,
        backend="tensorized",
    )
    value = loss(
        predicted_weight.unsqueeze(0),
        centered_predicted.unsqueeze(0),
        observed_weight.unsqueeze(0),
        centered_observed.unsqueeze(0),
    ).squeeze()
    return value.clamp_min(0)


def checkpoint_energy_distance(
    predicted_support: torch.Tensor,
    predicted_log_weight: torch.Tensor,
    observed_support: torch.Tensor,
    observed_log_weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted energy distance between normalized latent distributions."""
    predicted_probability = torch.softmax(predicted_log_weight.float(), dim=0).to(
        predicted_support.dtype
    )
    observed_probability = torch.softmax(observed_log_weight.float(), dim=0).to(
        observed_support.dtype
    )
    centered_predicted, centered_observed = _center_supports(
        predicted_support,
        observed_support,
    )
    distance_options = {
        "p": 2.0,
        "compute_mode": "donot_use_mm_for_euclid_dist",
    }
    cross = torch.cdist(centered_predicted, centered_observed, **distance_options)
    self_predicted = torch.cdist(
        centered_predicted,
        centered_predicted,
        **distance_options,
    )
    self_observed = torch.cdist(
        centered_observed,
        centered_observed,
        **distance_options,
    )
    cross_mean = predicted_probability @ cross @ observed_probability
    predicted_mean = predicted_probability @ self_predicted @ predicted_probability
    observed_mean = observed_probability @ self_observed @ observed_probability
    return (2 * cross_mean - predicted_mean - observed_mean).clamp_min(0)


def checkpoint_centroid_distance(
    predicted_support: torch.Tensor,
    predicted_log_weight: torch.Tensor,
    observed_support: torch.Tensor,
    observed_log_weight: torch.Tensor,
) -> torch.Tensor:
    """Euclidean error between normalized weighted latent centroids."""
    predicted_probability = torch.softmax(predicted_log_weight.float(), dim=0).to(
        predicted_support.dtype
    )
    observed_probability = torch.softmax(observed_log_weight.float(), dim=0).to(
        observed_support.dtype
    )
    centered_predicted, centered_observed = _center_supports(
        predicted_support,
        observed_support,
    )
    predicted_centroid = (predicted_probability[:, None] * centered_predicted).sum(dim=0)
    observed_centroid = (observed_probability[:, None] * centered_observed).sum(dim=0)
    return torch.linalg.vector_norm(predicted_centroid - observed_centroid)


def checkpoint_covariance_distance(
    predicted_support: torch.Tensor,
    predicted_log_weight: torch.Tensor,
    observed_support: torch.Tensor,
    observed_log_weight: torch.Tensor,
) -> torch.Tensor:
    """Frobenius error between normalized weighted latent covariance matrices."""

    def covariance(
        support: torch.Tensor,
        log_weight: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.softmax(log_weight.float(), dim=0).to(support.dtype)
        centroid = (probability[:, None] * support).sum(dim=0)
        centered = support - centroid
        return torch.einsum("n,ni,nj->ij", probability, centered, centered)

    predicted_covariance = covariance(predicted_support, predicted_log_weight)
    observed_covariance = covariance(observed_support, observed_log_weight)
    return torch.linalg.matrix_norm(
        predicted_covariance - observed_covariance,
        ord="fro",
    )


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
    sinkhorn_epsilon: float = 0.1,
    uot_reach: float = 1.0,
    time_labels: Iterable[str] | None = None,
    collect_rows: bool = True,
    collect_benchmark_metrics: bool = False,
    compute_geometry: bool = True,
) -> CheckpointObjective:
    """Apply one checkpoint objective to any number of observed times.

    ``compute_geometry=False`` skips every support-distance calculation while
    preserving mass, particle diagnostics, and NaN geometry result columns.
    """
    indices = checkpoint_indices(data.axis, rollout.axis_grid)
    selected_times = (
        set(data.axis.labels[1:]) if time_labels is None else {str(value) for value in time_labels}
    )
    unknown_times = selected_times - set(data.axis.labels[1:])
    if unknown_times:
        raise ValueError(f"Unknown objective checkpoint labels: {sorted(unknown_times)}")
    device = rollout.z_steps.device
    dtype = rollout.z_steps.dtype
    geometry_sum = torch.zeros((), device=device, dtype=dtype)
    mass_sum = torch.zeros((), device=device, dtype=dtype)
    rows: list[dict[str, Any]] = []
    row_keys: list[tuple[str, str]] = []
    row_values: list[torch.Tensor] = []
    row_centroids: list[tuple[torch.Tensor, torch.Tensor]] = []
    observation_count = 0
    terminal_diagnostics = weight_diagnostics(rollout.logw_steps) if collect_rows else None
    for label in data.axis.labels[1:]:
        if label not in selected_times:
            continue
        step = indices[label]
        for local_index, measure_id in enumerate(rollout.measure_ids):
            target = data.measures[label].get(measure_id)
            if target is None:
                continue
            target_support = torch.as_tensor(
                np.array(target.support, copy=True),
                device=device,
                dtype=dtype,
            )
            target_log_weight = torch.log(
                torch.as_tensor(
                    np.array(target.weights, copy=True),
                    device=device,
                    dtype=dtype,
                ).clamp_min(1e-30)
            )
            predicted_support = rollout.z_steps[step, local_index]
            # Materialize only this guide/checkpoint vector. Accessing the
            # full-trajectory property here allocates [time, guide, particle]
            # on every iteration; autograd retains those bases via the slices.
            predicted_log_weight = (
                rollout.log_m0[local_index] + rollout.logw_steps[step, local_index]
            )
            missing_geometry = rollout.z_steps.new_full((), float("nan"))
            geometry = (
                checkpoint_geometry(
                    predicted_support,
                    predicted_log_weight,
                    target_support,
                    target_log_weight,
                    epsilon=sinkhorn_epsilon,
                    stable_distances=collect_benchmark_metrics,
                )
                if compute_geometry
                else missing_geometry
            )
            mass_error, predicted_mass, observed_mass = checkpoint_log_mass_error(
                predicted_log_weight, target_log_weight
            )
            benchmark_values: tuple[torch.Tensor, ...] = ()
            if collect_benchmark_metrics:
                benchmark_values = (
                    (
                        checkpoint_unbalanced_sinkhorn(
                            predicted_support,
                            predicted_log_weight,
                            target_support,
                            target_log_weight,
                            epsilon=sinkhorn_epsilon,
                            reach=uot_reach,
                        ),
                        checkpoint_energy_distance(
                            predicted_support,
                            predicted_log_weight,
                            target_support,
                            target_log_weight,
                        ),
                        checkpoint_centroid_distance(
                            predicted_support,
                            predicted_log_weight,
                            target_support,
                            target_log_weight,
                        ),
                        checkpoint_covariance_distance(
                            predicted_support,
                            predicted_log_weight,
                            target_support,
                            target_log_weight,
                        ),
                    )
                    if compute_geometry
                    else (
                        missing_geometry,
                        missing_geometry,
                        missing_geometry,
                        missing_geometry,
                    )
                )
            if compute_geometry:
                geometry_sum = geometry_sum + geometry
            mass_sum = mass_sum + mass_error
            observation_count += 1
            if collect_rows:
                assert terminal_diagnostics is not None
                row_keys.append((measure_id, label))
                if collect_benchmark_metrics:
                    predicted_probability = torch.softmax(
                        predicted_log_weight.float(),
                        dim=0,
                    ).to(predicted_support.dtype)
                    observed_probability = torch.softmax(
                        target_log_weight.float(),
                        dim=0,
                    ).to(target_support.dtype)
                    row_centroids.append(
                        (
                            (predicted_probability[:, None] * predicted_support).sum(dim=0),
                            (observed_probability[:, None] * target_support).sum(dim=0),
                        )
                    )
                row_values.append(
                    torch.stack(
                        (
                            geometry,
                            mass_error,
                            predicted_mass,
                            observed_mass,
                            terminal_diagnostics["ess_fraction"][step, local_index],
                            terminal_diagnostics["max_weight_fraction"][step, local_index],
                            *benchmark_values,
                        )
                    )
                )
    if row_values:
        host_values = torch.stack(row_values).detach().cpu().tolist()
        host_centroids = [
            (
                predicted.detach().cpu().tolist(),
                observed.detach().cpu().tolist(),
            )
            for predicted, observed in row_centroids
        ]
        rows = []
        for row_index, ((measure_id, label), values) in enumerate(
            zip(row_keys, host_values, strict=True)
        ):
            row = {
                "measure_id": measure_id,
                "time_label": label,
                "endpoint_role": "observed_checkpoint",
                "validation_source": validation_source,
                "geometry": float(values[0]),
                "log_mass_error": float(values[1]),
                "predicted_log_mass": float(values[2]),
                "observed_log_mass": float(values[3]),
                "ess_fraction": float(values[4]),
                "max_weight_fraction": float(values[5]),
            }
            if collect_benchmark_metrics:
                predicted_centroid, observed_centroid = host_centroids[row_index]
                row.update(
                    {
                        "unbalanced_sinkhorn": float(values[6]),
                        "energy_distance": float(values[7]),
                        "centroid_distance": float(values[8]),
                        "covariance_distance": float(values[9]),
                        "_predicted_centroid": predicted_centroid,
                        "_observed_centroid": observed_centroid,
                    }
                )
            rows.append(row)
    if observation_count == 0:
        zero = rollout.z_steps.new_zeros(())
        return CheckpointObjective(
            total=zero,
            geometry=zero,
            log_mass_error=zero,
            observation_count=0,
            rows=[],
        )
    geometry_mean = (
        geometry_sum / observation_count
        if compute_geometry
        else rollout.z_steps.new_full((), float("nan"))
    )
    mass_mean = mass_sum / observation_count
    geometry_term = geometry_mean if compute_geometry else geometry_sum
    total = geometry_term + (float(mass_weight) * mass_mean if include_mass else 0.0)
    return CheckpointObjective(
        total=total,
        geometry=geometry_mean,
        log_mass_error=mass_mean,
        observation_count=observation_count,
        rows=rows,
    )


def integrated_fitness_curve(
    rollout: ParticleRollout,
    *,
    mass_increment: str = "integrated_mean_growth",
) -> torch.Tensor:
    """Return the explicitly selected cumulative log-mass increment.

    The default retains the historical quadrature of conditional mean growth.
    ``particle_log_mass`` instead uses the actual discrete particle weights,
    making count probabilities agree with the mass of the predicted measure.
    It intentionally preserves the log-weight dtype and its gradient graph.
    """
    if mass_increment == "particle_log_mass":
        # Remove a shared, per-measure origin before either logsumexp.  This
        # avoids subtracting two large log masses, without changing the value
        # or gradient of their difference.  Initial particle normalization is
        # not assumed; any positive finite initial measure is supported.
        logw = rollout.logw_steps
        origin = logw[0].amax(dim=-1, keepdim=True).detach()
        centered = logw - origin.unsqueeze(0)
        log_mass = torch.logsumexp(centered, dim=-1)
        return log_mass - log_mass[:1]
    if mass_increment != "integrated_mean_growth":
        raise ValueError(f"Unknown mass_increment {mass_increment!r}.")
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
    *,
    count_likelihood: str = "dirichlet_multinomial",
) -> torch.Tensor:
    exposure = block.exposure.to(device=fitness.device, dtype=fitness.dtype)
    counts = block.counts.to(device=fitness.device, dtype=fitness.dtype)
    if block.background_series_ids:
        fitness = torch.cat(
            (
                fitness,
                block.background_fitness.to(device=fitness.device, dtype=fitness.dtype),
            )
        )
        exposure = torch.cat(
            (
                exposure,
                block.background_exposure.to(device=fitness.device, dtype=fitness.dtype),
            )
        )
        counts = torch.cat(
            (
                counts,
                block.background_counts.to(device=fitness.device, dtype=fitness.dtype),
            )
        )
    logits = torch.log(exposure) + fitness
    if count_likelihood == "categorical_ce":
        # Counts are observations, not pseudocount-adjusted targets. The source
        # estimator belongs in exposure. Include every background category in
        # this one complete-catalog normalizer; zero-count terms contribute 0.
        # Normalize by the observed total so capture depth does not rescale the
        # objective. This is a prediction score, not replicate-level evidence.
        # CountBlock validates positive complete observed totals on construction.
        total = counts.sum()
        return -(counts / total * torch.log_softmax(logits, dim=0)).sum()
    if count_likelihood != "dirichlet_multinomial":
        raise ValueError(f"Unknown count likelihood {count_likelihood!r}.")
    probability = torch.softmax(logits, dim=0)
    return _dirichlet_multinomial_loss(counts, probability, log_concentration)


def count_block_loss(
    rollout: ParticleRollout,
    data: TrajectoryData,
    *,
    log_concentration: torch.Tensor,
    fitness_bank: FitnessBankProtocol | None = None,
    time_labels: Iterable[str] | None = None,
    count_likelihood: str = "dirichlet_multinomial",
    mass_increment: str = "integrated_mean_growth",
) -> torch.Tensor:
    """Evaluate the only count API over complete grouped blocks."""
    if (
        fitness_bank is not None
        and getattr(fitness_bank, "mass_increment", mass_increment) != mass_increment
    ):
        raise ValueError("Fitness bank and count loss use different mass increments.")
    blocks: Iterable[CountBlock] = data.count_blocks
    if not blocks:
        return rollout.z_steps.new_zeros(())
    data.axis.require_physical("Count likelihood")
    active_groups = set(rollout.context_group_ids)
    selected_times = None if time_labels is None else {str(value) for value in time_labels}
    blocks = tuple(
        block
        for block in blocks
        if block.context_group_id in active_groups
        and (selected_times is None or block.time_label in selected_times)
    )
    if not blocks:
        return rollout.z_steps.new_zeros(())
    curve = integrated_fitness_curve(rollout, mass_increment=mass_increment)
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
        losses.append(
            _count_block_nll(
                block, fitness, log_concentration, count_likelihood=count_likelihood
            )
        )
    return torch.stack(losses).mean()


def catalog_count_block_loss(
    data: TrajectoryData,
    *,
    log_concentration: torch.Tensor,
    fitness_bank: FitnessBankProtocol,
    context_group_ids: Iterable[str] | None = None,
    time_labels: Iterable[str] | None = None,
    count_likelihood: str = "dirichlet_multinomial",
    mass_increment: str = "integrated_mean_growth",
) -> tuple[torch.Tensor, int]:
    """Score complete count blocks from a refreshed detached catalog bank."""
    if getattr(fitness_bank, "mass_increment", mass_increment) != mass_increment:
        raise ValueError("Fitness bank and count loss use different mass increments.")
    data.axis.require_physical("Count likelihood")
    selected_groups = None if context_group_ids is None else set(context_group_ids)
    selected_times = None if time_labels is None else {str(value) for value in time_labels}
    blocks = tuple(
        block
        for block in data.count_blocks
        if selected_groups is None or block.context_group_id in selected_groups
        if selected_times is None or block.time_label in selected_times
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
                count_likelihood=count_likelihood,
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


def rollout_regularization(
    rollout: ParticleRollout,
    *,
    drift_weight: float = 1e-4,
    diffusion_weight: float = 1e-5,
    growth_weight: float = 1e-4,
) -> torch.Tensor:
    """Small auditable action penalty over generated coefficients."""
    return (
        float(drift_weight) * rollout.drift_steps.square().mean()
        + float(diffusion_weight) * rollout.sigma_steps.square().mean()
        + float(growth_weight) * rollout.growth_steps.square().mean()
    )


def total_objective(
    rollout: ParticleRollout,
    data: TrajectoryData,
    *,
    geometry_weight: float = 1.0,
    mass_weight: float,
    count_weight: float,
    include_mass: bool,
    log_concentration: torch.Tensor,
    fitness_bank: FitnessBankProtocol | None = None,
    validation_source: str = "train_self_eval",
    sinkhorn_epsilon: float = 0.1,
    time_labels: Iterable[str] | None = None,
    action_weights: tuple[float, float, float] = (1e-4, 1e-5, 1e-4),
    collect_rows: bool = True,
    count_likelihood: str = "dirichlet_multinomial",
    mass_increment: str = "integrated_mean_growth",
) -> ObjectiveResult:
    checkpoint = checkpoint_geometry_mass_loss(
        rollout,
        data,
        mass_weight=mass_weight,
        include_mass=include_mass,
        validation_source=validation_source,
        sinkhorn_epsilon=sinkhorn_epsilon,
        time_labels=time_labels,
        collect_rows=collect_rows,
    )
    counts = (
        count_block_loss(
            rollout,
            data,
            log_concentration=log_concentration,
            fitness_bank=fitness_bank,
            time_labels=time_labels,
            count_likelihood=count_likelihood,
            mass_increment=mass_increment,
        )
        if include_mass and count_weight > 0
        else rollout.z_steps.new_zeros(())
    )
    regularization = rollout_regularization(
        rollout,
        drift_weight=action_weights[0],
        diffusion_weight=action_weights[1],
        growth_weight=action_weights[2],
    )
    return ObjectiveResult(
        total=(
            float(geometry_weight) * checkpoint.geometry
            + (float(mass_weight) * checkpoint.log_mass_error if include_mass else 0.0)
            + float(count_weight) * counts
            + regularization
        ),
        checkpoint=checkpoint,
        count=counts,
        regularization=regularization,
    )
