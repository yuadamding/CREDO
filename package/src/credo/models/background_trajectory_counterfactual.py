"""Background-aware ecological counterfactuals for batched trajectory models."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from ..data.core import MeasureKey
from ..data.trajectory_view import TrajectoryLike
from .context import GroupStatistics
from ..training.trajectory_batch import TrajectoryContextBank, initialise_particles_from_trajectory
from .simulator import _control_embedding_context, _stable_seed_offset, rollout_with_clamped_context
from .trajectory_counterfactual import (
    TrajectoryCounterfactualEngine,
    TrajectoryCounterfactualResult,
)
from .weighted_sde import WeightedParticleSimulator
from .full_model import FullDynamicsModel


class BackgroundTrajectoryCounterfactualEngine:
    """Change one focal residual while retaining its donor's genome-wide background."""

    def __init__(
        self,
        model: FullDynamicsModel,
        simulator: WeightedParticleSimulator,
        *,
        n_particles: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not simulator.store_history:
            raise ValueError("Background counterfactuals require simulator.store_history=True.")
        self.model = model
        self.simulator = simulator
        self.n_particles = int(n_particles)
        self.device = device
        self.dtype = dtype

    def _context_bank(
        self,
        statistics: Sequence[GroupStatistics],
        context_group_index: torch.Tensor,
    ) -> TrajectoryContextBank:
        if not statistics:
            raise ValueError("background_group_statistics_steps must not be empty.")
        bank = TrajectoryContextBank(
            context_aggregator=self.model.context_agg,
            initial_statistics=statistics[0],
            context_group_index=context_group_index.to(self.device),
            n_steps=len(statistics),
            momentum=0.0,
        )
        bank.log_n_steps = torch.stack([item.log_n_g for item in statistics]).to(self.device)
        bank.eta_steps = torch.stack([item.eta_g for item in statistics]).to(self.device)
        bank.phi_steps = torch.stack([item.phi_g for item in statistics]).to(self.device)
        return bank

    @torch.no_grad()
    def run(
        self,
        trajectory: TrajectoryLike,
        *,
        source_label: str,
        target_labels: list[str],
        focal_measure_key: MeasureKey,
        focal_embedding_id: str,
        tau_grid: torch.Tensor,
        checkpoint_indices: dict[str, int],
        background_group_statistics_steps: Sequence[GroupStatistics] | None = None,
        background_context_steps: torch.Tensor | None = None,
        background_context_group_index: torch.Tensor | None = None,
        focal_global_index: int | None = None,
        focal_initial_mass: float | None = None,
        seed: int = 0,
        control_rollout_mode: str = "reference_consistent",
    ) -> TrajectoryCounterfactualResult:
        if focal_embedding_id not in self.model.perturbation_ids:
            raise KeyError(f"Model is missing focal embedding {focal_embedding_id!r}.")
        has_statistics = background_group_statistics_steps is not None
        has_context = background_context_steps is not None
        if has_statistics == has_context:
            raise ValueError(
                "Provide exactly one of background_group_statistics_steps or "
                "background_context_steps."
            )
        expected_steps = int(tau_grid.numel()) - 1
        if has_statistics and len(background_group_statistics_steps or ()) != expected_steps:
            raise ValueError("Background group-statistics length must match tau_grid transitions.")
        if has_context and background_context_steps.shape[0] != expected_steps:
            raise ValueError("Background context length must match tau_grid transitions.")
        z0, logw0, log_m0 = initialise_particles_from_trajectory(
            trajectory,
            source_label,
            [focal_measure_key],
            n_particles=self.n_particles,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )
        if focal_initial_mass is not None:
            if focal_initial_mass <= 0:
                raise ValueError("focal_initial_mass must be positive.")
            log_m0[0] = torch.log(log_m0.new_tensor(float(focal_initial_mass)))

        context_bank = None
        active_index = None
        if background_group_statistics_steps is not None:
            if background_context_group_index is None or focal_global_index is None:
                raise ValueError(
                    "Background group statistics require background_context_group_index "
                    "and focal_global_index."
                )
            context_bank = self._context_bank(
                background_group_statistics_steps,
                background_context_group_index,
            )
            active_index = torch.tensor([int(focal_global_index)], device=self.device)
            factual_context = context_bank.override(active_index, replace_active=True)
            reference_context = context_bank.override(active_index, replace_active=True)
        elif background_context_steps is not None:
            selected_context = background_context_steps
            if selected_context.ndim == 3:
                if focal_global_index is None:
                    raise ValueError(
                        "Full [K, G, C] background_context_steps require focal_global_index."
                    )
                selected_context = selected_context[:, int(focal_global_index):int(focal_global_index) + 1]
            factual_context = selected_context
            reference_context = selected_context
        else:  # Guarded above; retained for type narrowing.
            raise AssertionError("Background context contract was not resolved.")

        noise = self.simulator.sample_noise_for_tau_grid(
            z0,
            tau_grid.to(device=z0.device, dtype=z0.dtype),
            seed=seed + 10_000 + _stable_seed_offset(str(focal_measure_key)),
        )
        rollout_kwargs = {
            "z0": z0,
            "logw0": logw0,
            "model": self.model,
            "log_m0": log_m0,
            "tau_start": float(tau_grid[0]),
            "tau_end": float(tau_grid[-1]),
            "tau_grid": tau_grid,
            "perturbation_ids": [str(focal_measure_key)],
            "embedding_ids": [focal_embedding_id],
            "noise_steps": noise,
        }
        factual = self.simulator.rollout(
            **rollout_kwargs,
            context_override=factual_context,
        )
        with _control_embedding_context(
            self.model,
            focal_embedding_id,
            mode=control_rollout_mode,
        ):
            reference = self.simulator.rollout(
                **{
                    **rollout_kwargs,
                    "z0": z0.clone(),
                    "logw0": logw0.clone(),
                    "log_m0": log_m0.clone(),
                },
                context_override=reference_context,
            )

        if reference.context_steps is None:
            raise ValueError("Reference background rollout did not retain context steps.")
        factual_clamped = rollout_with_clamped_context(
            model=self.model,
            z0=z0,
            logw0=logw0,
            log_m0=log_m0,
            perturbation_ids=[str(focal_measure_key)],
            embedding_ids=[focal_embedding_id],
            context_steps=reference.context_steps,
            tau_grid=reference.tau_steps,
            tau_start=float(tau_grid[0]),
            tau_end=float(tau_grid[-1]),
            noise_steps=noise,
        )
        with _control_embedding_context(
            self.model,
            focal_embedding_id,
            mode=control_rollout_mode,
        ):
            reference_clamped = rollout_with_clamped_context(
                model=self.model,
                z0=z0.clone(),
                logw0=logw0.clone(),
                log_m0=log_m0.clone(),
                perturbation_ids=[str(focal_measure_key)],
                embedding_ids=[focal_embedding_id],
                context_steps=reference.context_steps,
                tau_grid=reference.tau_steps,
                tau_start=float(tau_grid[0]),
                tau_end=float(tau_grid[-1]),
                noise_steps=noise,
            )
        metrics = TrajectoryCounterfactualEngine._metrics_by_time(
            measure_key=focal_measure_key,
            embedding_id=focal_embedding_id,
            source_label=source_label,
            target_labels=target_labels,
            checkpoint_indices=checkpoint_indices,
            factual=factual,
            reference=reference,
            factual_clamped=factual_clamped,
        )
        return TrajectoryCounterfactualResult(
            measure_key=focal_measure_key,
            embedding_id=focal_embedding_id,
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


__all__ = ["BackgroundTrajectoryCounterfactualEngine"]
