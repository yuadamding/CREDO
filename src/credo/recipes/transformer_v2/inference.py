"""Deterministic inference for imported transformer-v2 runs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import pandas as pd
import torch

from ...contracts import Axis, CREDOStudy
from ..compact_sde_v3.objective import checkpoint_geometry_mass_loss
from ..compact_sde_v3.particles import (
    DynamicsStep,
    ParticleRollout,
    ParticleState,
    euler_maruyama_rollout,
    sample_initial_particles,
    sample_noise,
    weight_diagnostics,
)
from .importer import ImportedTransformerV2Run
from .model import FullDynamicsModel


def historical_axis_grid(
    axis: Axis,
    steps_per_interval: int,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Reproduce v2's piecewise torch.linspace checkpoint grid."""
    pieces = []
    for start, stop in zip(axis.normalized_values[:-1], axis.normalized_values[1:], strict=False):
        segment = torch.linspace(
            start,
            stop,
            steps_per_interval + 1,
            device=device,
            dtype=torch.float32,
        )
        if pieces:
            segment = segment[1:]
        pieces.append(segment)
    return torch.cat(pieces)


@dataclass(frozen=True)
class TransformerV2Kernel:
    model: FullDynamicsModel

    def step(
        self,
        *,
        step_index: int,
        z: torch.Tensor,
        logw: torch.Tensor,
        time: torch.Tensor,
        state: ParticleState,
    ) -> DynamicsStep:
        del step_index
        coefficients, context = self.model.step(
            z,
            time,
            logw,
            state.log_m0,
            list(state.embedding_ids),
            state.residual_scale,
        )
        return DynamicsStep(
            drift=coefficients.drift,
            sigma_diag=coefficients.sigma_diag,
            growth=coefficients.growth,
            context=context.context.detach(),
        )


@torch.no_grad()
def rollout_transformer_v2(
    model: FullDynamicsModel,
    initial_state: ParticleState,
    grid: torch.Tensor,
    *,
    noise: torch.Tensor,
) -> ParticleRollout:
    """Run the historical model through the common integration driver."""
    return euler_maruyama_rollout(
        TransformerV2Kernel(model),
        initial_state,
        grid,
        noise=noise,
    )


def _float_rollout(rollout: ParticleRollout) -> ParticleRollout:
    return ParticleRollout(
        z_steps=rollout.z_steps.float(),
        logw_steps=rollout.logw_steps.float(),
        log_m0=rollout.log_m0.float(),
        axis_grid=rollout.axis_grid.float(),
        measure_ids=rollout.measure_ids,
        embedding_ids=rollout.embedding_ids,
        context_group_ids=rollout.context_group_ids,
        measure_indices=rollout.measure_indices,
        residual_scale=rollout.residual_scale.float(),
        drift_steps=rollout.drift_steps.float(),
        sigma_steps=rollout.sigma_steps.float(),
        growth_steps=rollout.growth_steps.float(),
        context_steps=rollout.context_steps.float(),
        noise_steps=rollout.noise_steps.float(),
    )


@torch.no_grad()
def evaluate_replay(
    run: ImportedTransformerV2Run,
    study: CREDOStudy,
    *,
    particles: int = 640,
    steps_per_interval: int = 24,
    seed: int = 0,
    noise_seed: int | None = None,
    device: str | torch.device | None = None,
    compute_geometry: bool = True,
) -> tuple[pd.DataFrame, ParticleRollout]:
    """Run one deterministic held-out replay through the common metric contract."""
    run.require("evaluate")
    if particles < 2 or steps_per_interval < 1 or seed < 0:
        raise ValueError("Replay particles, integration steps, and seed are invalid.")
    resolved_noise_seed = seed + 1_000_003 if noise_seed is None else int(noise_seed)
    if resolved_noise_seed < 0:
        raise ValueError("Replay noise seed must be nonnegative.")
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    run.model.to(selected_device, dtype=torch.float32).eval()
    grid = historical_axis_grid(study.axis, steps_per_interval, device=selected_device)
    source = sample_initial_particles(
        study,
        study.measure_ids,
        particles,
        device=selected_device,
        dtype=dtype,
        seed=seed,
    )
    noise = sample_noise(source, grid, seed=resolved_noise_seed)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if selected_device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        particle_rollout = rollout_transformer_v2(run.model, source, grid, noise=noise)
    float_rollout = _float_rollout(particle_rollout)
    checkpoint = checkpoint_geometry_mass_loss(
        float_rollout,
        study,
        mass_weight=1.0,
        include_mass=True,
        validation_source="held_out",
        sinkhorn_epsilon=0.1,
    )
    rows = checkpoint.rows
    if not compute_geometry:
        for row in rows:
            row["geometry"] = float("nan")
    order = {value: index for index, value in enumerate(study.measure_ids)}
    time_order = {value: index for index, value in enumerate(study.axis.labels)}
    frame = pd.DataFrame(rows)
    frame["_measure_order"] = frame["measure_id"].map(order)
    frame["_time_order"] = frame["time_label"].map(time_order)
    frame = frame.sort_values(["_measure_order", "_time_order"]).drop(
        columns=["_measure_order", "_time_order"]
    )
    frame.insert(0, "recipe_id", run.recipe_id)
    frame.insert(1, "recipe_version", run.recipe_version)
    frame.insert(2, "representation_id", run.representation.representation_id)
    frame.insert(3, "split_id", run.split.split_id)
    frame["evaluation_particles"] = int(particles)
    frame["integration_steps"] = int(len(grid) - 1)
    frame["evaluation_seed"] = int(seed)
    frame["noise_seed"] = int(resolved_noise_seed)
    return frame.reset_index(drop=True), particle_rollout


@torch.no_grad()
def counterfactual_replay(
    run: ImportedTransformerV2Run,
    study: CREDOStudy,
    measure_id: str,
    *,
    context_policy: str = "self_consistent",
    same_noise: bool = True,
    n_particles: int | None = None,
    seed: int | None = None,
    steps_per_interval: int = 24,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Exact full-group, same-start, same-noise v2 reference contrast."""
    from ... import __version__
    from ...counterfactual import _energy_distance, _weighted_mean
    from ..compact_sde_v3.training import _git_state

    run.require("counterfactual")
    if context_policy != "self_consistent":
        raise ValueError(
            "transformer-v2 currently exposes exact self_consistent full-group context only."
        )
    if not same_noise:
        raise ValueError("CREDO reference counterfactuals require same_noise=True.")
    if measure_id not in study.measure_ids:
        raise KeyError(f"Unknown measure_id {measure_id!r}.")
    particles = 640 if n_particles is None else int(n_particles)
    if particles < 2:
        raise ValueError("n_particles must be at least 2.")
    if seed is None:
        import hashlib

        seed = int(hashlib.sha256(measure_id.encode()).hexdigest()[:8], 16) % 1_000_000
    if seed < 0 or steps_per_interval < 1:
        raise ValueError("Counterfactual integration steps and seed must be valid.")
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    run.model.to(selected_device, dtype=torch.float32).eval()
    grid = historical_axis_grid(study.axis, steps_per_interval, device=selected_device)
    source = sample_initial_particles(
        study,
        study.measure_ids,
        particles,
        device=selected_device,
        dtype=dtype,
        seed=seed,
    )
    noise = sample_noise(source, grid, seed=seed + 1_000_003)
    if not torch.equal(source.z, source.z.clone()) or not torch.equal(noise, noise.clone()):
        raise AssertionError("Counterfactual source or noise cloning changed values.")
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if selected_device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        factual = rollout_transformer_v2(run.model, source, grid, noise=noise)

    local_index = study.measure_ids.index(measure_id)
    reference_scale = source.residual_scale.clone()
    reference_scale[local_index] = 0
    reference_source = source.with_residual_scale(reference_scale)
    factual_embedding = run.model.embedding(list(source.embedding_ids), source.residual_scale)
    reference_embedding = run.model.embedding(
        list(source.embedding_ids), reference_source.residual_scale
    )
    unselected = torch.ones(len(source.measure_ids), dtype=torch.bool, device=selected_device)
    unselected[local_index] = False
    if not torch.equal(factual_embedding[unselected], reference_embedding[unselected]):
        raise AssertionError("Counterfactual changed an unselected perturbation residual.")
    with autocast:
        reference = rollout_transformer_v2(run.model, reference_source, grid, noise=noise)
    if (
        not torch.equal(factual.z_steps[0], reference.z_steps[0])
        or not torch.equal(factual.logw_steps[0], reference.logw_steps[0])
        or not torch.equal(factual.log_m0, reference.log_m0)
    ):
        raise AssertionError("Counterfactual branches did not use identical source particles.")
    if not torch.equal(factual.noise_steps, reference.noise_steps):
        raise AssertionError("Counterfactual branches did not use identical Brownian noise.")

    from ..compact_sde_v3.particles import checkpoint_indices

    checkpoints = checkpoint_indices(study.axis, factual.axis_grid)
    factual_diagnostics = weight_diagnostics(factual.logw_steps)
    reference_diagnostics = weight_diagnostics(reference.logw_steps)
    git_sha, _ = _git_state()
    rows = []
    for label in study.axis.labels[1:]:
        step = checkpoints[label]
        factual_support = factual.z_steps[step, local_index].float()
        reference_support = reference.z_steps[step, local_index].float()
        factual_weight = factual.absolute_log_weight_steps[step, local_index].float()
        reference_weight = reference.absolute_log_weight_steps[step, local_index].float()
        rows.append(
            {
                "recipe_id": run.recipe_id,
                "recipe_version": run.recipe_version,
                "implementation_hash": run.envelope.recipe["implementation_hash"],
                "representation_id": run.representation.representation_id,
                "split_id": run.split.split_id,
                "measure_id": measure_id,
                "time_label": label,
                "context_policy": context_policy,
                "delta_log_mass": float(
                    (
                        torch.logsumexp(factual_weight, dim=0)
                        - torch.logsumexp(reference_weight, dim=0)
                    ).cpu()
                ),
                "mean_shift_l2": float(
                    torch.linalg.vector_norm(
                        _weighted_mean(factual_support, factual_weight)
                        - _weighted_mean(reference_support, reference_weight)
                    ).cpu()
                ),
                "energy_distance": float(
                    _energy_distance(
                        factual_support,
                        factual_weight,
                        reference_support,
                        reference_weight,
                    ).cpu()
                ),
                "context_dependence_shift": float(
                    torch.linalg.vector_norm(
                        factual.context_steps[:step].float()
                        - reference.context_steps[:step].float(),
                        dim=-1,
                    )
                    .mean()
                    .cpu()
                ),
                "factual_ess": float(factual_diagnostics["ess"][step, local_index].cpu()),
                "reference_ess": float(reference_diagnostics["ess"][step, local_index].cpu()),
                "evaluation_particles": particles,
                "integration_steps": len(grid) - 1,
                "evaluation_seed": seed,
                "noise_seed": seed + 1_000_003,
                "checkpoint_sha256": run.envelope.import_provenance["source_checkpoint_sha256"],
                "package_version": __version__,
                "git_sha": git_sha,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "counterfactual_replay",
    "evaluate_replay",
    "historical_axis_grid",
    "rollout_transformer_v2",
]
