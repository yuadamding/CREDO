"""Time-indexed counterfactuals for trajectory CREDO runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pandas as pd
import torch

from ..data.core import MeasureKey, SparseTrajectoryProblem, TrajectoryProblem
from ..data.trajectory_view import embedding_id_for_measure_key
from ..training.trajectory_batch import initialise_particles_from_trajectory
from .full_model import FullDynamicsModel
from .simulator import _control_embedding_context, rollout_with_clamped_context
from .weighted_sde import ParticleRollout, WeightedParticleSimulator


TrajectoryLike = TrajectoryProblem | SparseTrajectoryProblem


def _stable_seed_offset(text: str, modulus: int = 1_000_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def _weighted_mean(z: torch.Tensor, logw: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logw, dim=0)
    return (weights[:, None] * z).sum(dim=0)


@dataclass
class TrajectoryCounterfactualResult:
    """Counterfactual trajectory outputs for one measure key."""

    measure_key: MeasureKey
    embedding_id: str
    source_label: str
    target_labels: list[str]
    tau_grid: torch.Tensor
    checkpoint_indices: dict[str, int]
    factual: ParticleRollout
    reference: ParticleRollout
    factual_clamped: ParticleRollout | None = None
    reference_clamped: ParticleRollout | None = None
    metrics_by_time: pd.DataFrame | None = None


class TrajectoryCounterfactualEngine:
    """Same-start, same-noise counterfactuals at trajectory checkpoints."""

    def __init__(
        self,
        model: FullDynamicsModel,
        simulator: WeightedParticleSimulator,
        *,
        n_particles: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.simulator = simulator
        self.n_particles = int(n_particles)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def run(
        self,
        trajectory: TrajectoryLike,
        *,
        source_label: str,
        target_labels: list[str],
        measure_key: MeasureKey,
        embedding_id: str | None = None,
        tau_grid: torch.Tensor,
        checkpoint_indices: dict[str, int] | None = None,
        common_noise: bool = True,
        clamp_context: bool = False,
        seed: int = 0,
        control_rollout_mode: str = "reference_consistent",
    ) -> TrajectoryCounterfactualResult:
        if clamp_context and not self.simulator.store_history:
            raise ValueError("clamp_context=True requires simulator.store_history=True.")
        if control_rollout_mode not in {"reference_consistent", "zero_centered"}:
            raise ValueError("control_rollout_mode must be 'reference_consistent' or 'zero_centered'.")

        embedding_id = embedding_id or embedding_id_for_measure_key(measure_key)
        if embedding_id not in self.model.perturbation_ids:
            raise KeyError(f"Model is missing embedding id {embedding_id!r}.")
        if source_label not in trajectory.measures or measure_key not in trajectory.measures[source_label]:
            raise KeyError(f"Missing source measure for {measure_key!r} at {source_label!r}.")

        labels = [source_label] + list(target_labels)
        taus = [trajectory.tau(label) for label in labels]
        if checkpoint_indices is None:
            from ..losses.multitime import checkpoint_indices_for_taus

            checkpoint_indices = checkpoint_indices_for_taus(tau_grid, labels, taus)

        z0, logw0, log_m0 = initialise_particles_from_trajectory(
            trajectory,
            source_label,
            [measure_key],
            n_particles=self.n_particles,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )
        branch_seed = int(seed) + 10_000 + _stable_seed_offset(str(measure_key))
        noise_steps = None
        if common_noise:
            noise_steps = self.simulator.sample_noise_for_tau_grid(
                z0,
                tau_grid.to(device=z0.device, dtype=z0.dtype),
                seed=branch_seed,
            )

        tau_start = float(tau_grid[0].detach().cpu())
        tau_end = float(tau_grid[-1].detach().cpu())
        factual = self.simulator.rollout(
            z0=z0,
            logw0=logw0,
            model=self.model,
            log_m0=log_m0,
            tau_start=tau_start,
            tau_end=tau_end,
            tau_grid=tau_grid,
            perturbation_ids=[embedding_id],
            noise_steps=noise_steps,
        )
        with _control_embedding_context(self.model, embedding_id, mode=control_rollout_mode):
            reference = self.simulator.rollout(
                z0=z0.clone(),
                logw0=logw0.clone(),
                model=self.model,
                log_m0=log_m0.clone(),
                tau_start=tau_start,
                tau_end=tau_end,
                tau_grid=tau_grid,
                perturbation_ids=[embedding_id],
                noise_steps=noise_steps,
            )

        factual_clamped = None
        reference_clamped = None
        if clamp_context:
            if reference.context_steps is None:
                raise ValueError("Reference rollout did not store context_steps.")
            factual_clamped = rollout_with_clamped_context(
                model=self.model,
                z0=z0,
                logw0=logw0,
                log_m0=log_m0,
                perturbation_ids=[embedding_id],
                context_steps=reference.context_steps,
                tau_start=tau_start,
                tau_end=tau_end,
                tau_grid=reference.tau_steps.detach(),
                noise_steps=noise_steps,
            )
            with _control_embedding_context(self.model, embedding_id, mode=control_rollout_mode):
                reference_clamped = rollout_with_clamped_context(
                    model=self.model,
                    z0=z0.clone(),
                    logw0=logw0.clone(),
                    log_m0=log_m0.clone(),
                    perturbation_ids=[embedding_id],
                    context_steps=reference.context_steps,
                    tau_start=tau_start,
                    tau_end=tau_end,
                    tau_grid=reference.tau_steps.detach(),
                    noise_steps=noise_steps,
                )

        metrics = self._metrics_by_time(
            measure_key=measure_key,
            embedding_id=embedding_id,
            source_label=source_label,
            target_labels=target_labels,
            checkpoint_indices=checkpoint_indices,
            factual=factual,
            reference=reference,
            factual_clamped=factual_clamped,
        )
        return TrajectoryCounterfactualResult(
            measure_key=measure_key,
            embedding_id=embedding_id,
            source_label=source_label,
            target_labels=list(target_labels),
            tau_grid=tau_grid.detach().clone(),
            checkpoint_indices=dict(checkpoint_indices),
            factual=factual,
            reference=reference,
            factual_clamped=factual_clamped,
            reference_clamped=reference_clamped,
            metrics_by_time=metrics,
        )

    def _metrics_by_time(
        self,
        *,
        measure_key: MeasureKey,
        embedding_id: str,
        source_label: str,
        target_labels: list[str],
        checkpoint_indices: dict[str, int],
        factual: ParticleRollout,
        reference: ParticleRollout,
        factual_clamped: ParticleRollout | None,
    ) -> pd.DataFrame:
        rows = []
        for label in target_labels:
            idx = checkpoint_indices[label]
            log_mass_f = factual.log_m0[0] + torch.logsumexp(factual.logw_steps[idx, 0], dim=0)
            log_mass_r = reference.log_m0[0] + torch.logsumexp(reference.logw_steps[idx, 0], dim=0)
            mean_f = _weighted_mean(factual.z_steps[idx, 0], factual.logw_steps[idx, 0])
            mean_r = _weighted_mean(reference.z_steps[idx, 0], reference.logw_steps[idx, 0])
            context_gap = float("nan")
            if factual_clamped is not None:
                mean_fc = _weighted_mean(
                    factual_clamped.z_steps[idx, 0],
                    factual_clamped.logw_steps[idx, 0],
                )
                context_gap = float(torch.linalg.norm(mean_f - mean_fc).detach().cpu())
            rows.append(
                {
                    "measure_key": str(measure_key),
                    "embedding_id": embedding_id,
                    "source_label": source_label,
                    "target_label": label,
                    "tau": float(factual.tau_steps[idx].detach().cpu()),
                    "log_mass_factual": float(log_mass_f.detach().cpu()),
                    "log_mass_reference": float(log_mass_r.detach().cpu()),
                    "delta_log_mass_fact_vs_ref": float((log_mass_f - log_mass_r).detach().cpu()),
                    "geom_shift_fact_vs_ref": float(torch.linalg.norm(mean_f - mean_r).detach().cpu()),
                    "context_dependence_geom": context_gap,
                }
            )
        return pd.DataFrame(rows)


__all__ = ["TrajectoryCounterfactualEngine", "TrajectoryCounterfactualResult"]
