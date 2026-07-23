"""One catalog bank, fixed continuation schedule, evaluation, and persistence."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from ...contracts import (
    MEASURE_META_COLUMNS,
    OptimizerSpec,
    Stage,
    TrainingPlan,
    TrajectoryData,
)
from ...data.splits import SplitPlan, plan_compact_trajectory_split
from ...io import RunConfig, resolved_config, validate_run_data
from ...problems import FiniteMeasureDynamicsProblem
from ...runtime import ObjectiveDescriptor
from .model import CREDOModel
from .objective import (
    CountBlock,
    catalog_count_block_loss,
    checkpoint_geometry_mass_loss,
    integrated_fitness_curve,
    total_objective,
    validate_count_blocks,
)
from .particles import (
    CatalogContextProvider,
    NoContextProvider,
    SelfConsistentContextProvider,
    axis_grid,
    checkpoint_indices,
    rollout,
    sample_initial_particles,
    sample_noise,
)


@dataclass
class CatalogBank:
    """Detached full-catalog tensors with differentiable active replacement."""

    tensors: dict[str, torch.Tensor]
    seen: dict[str, torch.Tensor]
    age: dict[str, torch.Tensor]
    context_group_index: torch.Tensor
    time_to_index: dict[str, int]
    momentum: float = 0.9
    last_full_refresh_epoch: int = -1

    @classmethod
    def empty(
        cls,
        data: TrajectoryData,
        model: CREDOModel,
        n_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CatalogBank:
        measure_count = len(data.measure_ids)
        time_count = len(data.axis.labels)
        group_values = data.measure_meta.set_index("measure_id").loc[
            list(data.measure_ids), "context_group_id"
        ]
        group_mapping = {
            value: index for index, value in enumerate(dict.fromkeys(group_values.tolist()))
        }
        group_index = torch.tensor(
            [group_mapping[value] for value in group_values], device=device, dtype=torch.long
        )
        tensors = {
            "context_log_mass": torch.zeros(n_steps, measure_count, device=device),
            "context_programs": torch.zeros(
                n_steps, measure_count, model.n_programs, device=device, dtype=dtype
            ),
            "fitness": torch.zeros(time_count, measure_count, device=device, dtype=dtype),
        }
        seen = {
            "context": torch.zeros(n_steps, measure_count, device=device, dtype=torch.bool),
            "fitness": torch.zeros(time_count, measure_count, device=device, dtype=torch.bool),
        }
        age = {name: torch.zeros_like(value, dtype=torch.long) for name, value in seen.items()}
        return cls(
            tensors=tensors,
            seen=seen,
            age=age,
            context_group_index=group_index,
            time_to_index={label: index for index, label in enumerate(data.axis.labels)},
        )

    def reset_coverage(self) -> None:
        for value in self.seen.values():
            value.zero_()
        for value in self.age.values():
            value.zero_()

    @torch.no_grad()
    def tick(self) -> None:
        for name in self.age:
            self.age[name][self.seen[name]] += 1

    @torch.no_grad()
    def _replace(
        self,
        name: str,
        destination: torch.Tensor,
        seen: torch.Tensor,
        age: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        full_refresh: bool,
    ) -> None:
        indices = indices.to(device=destination.device, dtype=torch.long)
        values = values.detach().to(device=destination.device, dtype=destination.dtype)
        previous = destination.index_select(0, indices)
        was_seen = seen.index_select(0, indices)
        blended = torch.where(
            was_seen.reshape((-1,) + (1,) * (values.ndim - 1)),
            self.momentum * previous + (1 - self.momentum) * values,
            values,
        )
        if full_refresh:
            blended = values
        destination.index_copy_(0, indices, blended)
        seen.index_fill_(0, indices, True)
        age.index_fill_(0, indices, 0)

    @torch.no_grad()
    def update_from_rollout(
        self,
        particle_rollout,
        model: CREDOModel,
        data: TrajectoryData,
        *,
        full_refresh: bool,
    ) -> None:
        active = particle_rollout.measure_indices.to(self.context_group_index.device)
        for step in range(len(particle_rollout.axis_grid) - 1):
            log_mass, programs = model.summarize_context(
                particle_rollout.z_steps[step],
                particle_rollout.absolute_log_weight_steps[step],
            )
            self._replace(
                "context_log_mass",
                self.tensors["context_log_mass"][step],
                self.seen["context"][step],
                self.age["context"][step],
                active,
                log_mass,
                full_refresh=full_refresh,
            )
            self._replace(
                "context_programs",
                self.tensors["context_programs"][step],
                self.seen["context"][step],
                self.age["context"][step],
                active,
                programs,
                full_refresh=full_refresh,
            )
        fitness = integrated_fitness_curve(particle_rollout)
        checkpoints = checkpoint_indices(data.axis, particle_rollout.axis_grid)
        for label, time_index in self.time_to_index.items():
            self._replace(
                "fitness",
                self.tensors["fitness"][time_index],
                self.seen["fitness"][time_index],
                self.age["fitness"][time_index],
                active,
                fitness[checkpoints[label]],
                full_refresh=full_refresh,
            )

    def assert_complete(self) -> None:
        incomplete = {
            name: int((~value).sum().item())
            for name, value in self.seen.items()
            if not bool(value.all())
        }
        if incomplete:
            raise RuntimeError(f"CatalogBank is incomplete: {incomplete}")

    def context_for_active(
        self,
        *,
        step_index: int,
        active_indices: torch.Tensor,
        active_log_mass: torch.Tensor,
        active_programs: torch.Tensor,
        model: CREDOModel,
    ) -> torch.Tensor:
        self.assert_complete()
        active = active_indices.to(device=self.context_group_index.device, dtype=torch.long)
        full_log_mass = self.tensors["context_log_mass"][step_index].detach().clone()
        full_programs = self.tensors["context_programs"][step_index].detach().clone()
        full_log_mass = full_log_mass.index_copy(0, active, active_log_mass.to(full_log_mass))
        full_programs = full_programs.index_copy(0, active, active_programs.to(full_programs))
        full_context = model.compose_context(full_log_mass, full_programs, self.context_group_index)
        return full_context.index_select(0, active)

    def fitness_for_active(
        self,
        *,
        time_label: str,
        active_indices: torch.Tensor,
        active_fitness: torch.Tensor,
    ) -> torch.Tensor:
        self.assert_complete()
        time_index = self.time_to_index[str(time_label)]
        active = active_indices.to(device=self.context_group_index.device, dtype=torch.long)
        full = self.tensors["fitness"][time_index].detach().clone()
        return full.index_copy(0, active, active_fitness.to(full))

    def full_fitness(self, *, time_label: str) -> torch.Tensor:
        self.assert_complete()
        time_index = self.time_to_index[str(time_label)]
        return self.tensors["fitness"][time_index].detach()

    def diagnostics(self) -> dict[str, float | int]:
        seen_values = torch.cat([value.reshape(-1) for value in self.seen.values()])
        age_values = torch.cat([value.reshape(-1) for value in self.age.values()])
        return {
            "bank_seen_fraction": float(seen_values.float().mean().item()),
            "bank_max_age": int(age_values.max().item()),
            "bank_mean_age": float(age_values.float().mean().item()),
            "last_full_refresh_epoch": int(self.last_full_refresh_epoch),
        }


@dataclass
class Trainer:
    """Compact-v3 runtime produced only from an immutable recipe plan."""

    data: TrajectoryData
    validation_data: TrajectoryData
    problem: FiniteMeasureDynamicsProblem | None
    model: CREDOModel
    config: RunConfig
    training_plan: TrainingPlan
    objective_descriptors: tuple[ObjectiveDescriptor, ...]
    device: torch.device
    dtype: torch.dtype
    log_count_concentration: torch.nn.Parameter
    bank: CatalogBank
    validation_bank: CatalogBank
    train_measure_ids: tuple[str, ...]
    validation_measure_ids: tuple[str, ...]
    train_time_labels: tuple[str, ...]
    validation_time_labels: tuple[str, ...]
    validation_source: Literal["held_out", "train_self_eval"]
    validation_strategy: Literal[
        "context_group_holdout",
        "checkpoint_holdout",
        "within_embedding_holdout",
        "train_self_eval",
    ]
    representation_scope: Literal["shared", "nested"]
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    counterfactual_rows: list[dict[str, Any]] = field(default_factory=list)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    completed_epochs: int = 0
    checkpoint_sha256: str | None = None

    @classmethod
    def from_plan(
        cls,
        data: TrajectoryData | FiniteMeasureDynamicsProblem,
        model: CREDOModel,
        config: RunConfig,
        plan: TrainingPlan,
        objectives: tuple[ObjectiveDescriptor, ...],
        *,
        device: str | torch.device | None = None,
    ) -> Trainer:
        """Execute the exact plan assembled by ``TrainingEngine``."""
        selected_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float32
        torch.manual_seed(plan.seed)
        np.random.seed(plan.seed)
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(plan.seed)
        problem = data if isinstance(data, FiniteMeasureDynamicsProblem) else None
        training_data = data.training if problem is not None else data
        validation_data = data.validation if problem is not None else data
        validate_run_data(config, training_data)
        validate_run_data(config, validation_data)
        from ...runtime import validate_training_contract
        from .recipe import recipe as compact_recipe

        validate_training_contract(compact_recipe, objectives, plan)
        if not _model_matches(model, training_data, config):
            raise ValueError("Provided model architecture disagrees with the run data or config.")
        model = model.to(device=selected_device, dtype=dtype)
        model.assert_soft_reference()
        validate_count_blocks(training_data)
        validate_count_blocks(validation_data)
        raw_split = training_data.metadata.get("split_plan")
        split = (
            SplitPlan.from_dict(raw_split)
            if isinstance(raw_split, Mapping)
            else _validation_split(training_data, config, seed=plan.seed)
        )
        representation_scope = _representation_scope(training_data, split, config)
        grid = axis_grid(
            training_data.axis,
            plan.steps_per_interval,
            device=selected_device,
            dtype=dtype,
        )
        bank = CatalogBank.empty(
            training_data,
            model,
            len(grid) - 1,
            device=selected_device,
            dtype=dtype,
        )
        validation_bank = CatalogBank.empty(
            validation_data,
            model,
            len(grid) - 1,
            device=selected_device,
            dtype=dtype,
        )
        trainer = cls(
            data=training_data,
            validation_data=validation_data,
            problem=problem,
            model=model,
            config=config,
            training_plan=plan,
            objective_descriptors=objectives,
            device=selected_device,
            dtype=dtype,
            log_count_concentration=torch.nn.Parameter(
                torch.tensor(np.log(100.0), device=selected_device, dtype=dtype)
            ),
            bank=bank,
            validation_bank=validation_bank,
            train_measure_ids=split.train_measure_ids,
            validation_measure_ids=split.validation_measure_ids,
            train_time_labels=split.train_time_labels,
            validation_time_labels=split.validation_time_labels,
            validation_source=split.source,
            validation_strategy=split.strategy,
            representation_scope=representation_scope,
        )
        trainer._fit()
        return trainer

    @property
    def settings(self) -> Any:
        return self.config.recipe_config

    @property
    def grid(self) -> torch.Tensor:
        return axis_grid(
            self.data.axis,
            self.training_plan.steps_per_interval,
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def objective_map(self) -> dict[str, ObjectiveDescriptor]:
        return {objective.name: objective for objective in self.objective_descriptors}

    def _optimizer(
        self,
        spec: OptimizerSpec,
        parameters: list[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        if spec.parameter_learning_rates or spec.parameter_weight_decays:
            raise ValueError("compact-v3 does not declare per-tag optimizer settings.")
        if spec.kind == "adam":
            optimizer_type = torch.optim.Adam
        elif spec.kind == "adamw":
            optimizer_type = torch.optim.AdamW
        else:  # pragma: no cover - OptimizerSpec validates this before execution.
            raise ValueError(f"Unsupported optimizer {spec.kind!r}.")
        return optimizer_type(
            parameters,
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )

    def _objective(self, stage: Stage, name: str) -> ObjectiveDescriptor | None:
        if name not in stage.active_objectives:
            return None
        return self.objective_map[name]

    def _objective_weight(self, stage: Stage, name: str) -> float:
        objective = self._objective(stage, name)
        return 0.0 if objective is None else float(objective.weight)

    def _fit(self) -> None:
        torch.manual_seed(self.training_plan.seed)
        np.random.seed(self.training_plan.seed)
        for stage in self.training_plan.stages:
            phase = stage.name
            if phase not in {"state", "mass", "context"}:
                raise ValueError(f"compact-v3 cannot execute stage {phase!r}.")
            if stage.epochs == 0:
                continue
            if stage.precision != "fp32":
                raise ValueError("compact-v3 currently executes released stages in fp32.")
            growth_enabled = phase != "state"
            trainable_names = self.model.set_trainable_tags(
                stage.trainable_tags,
                growth_enabled=growth_enabled,
                context_enabled=stage.context_policy == "catalog_bank",
            )
            if phase in {"mass", "context"}:
                self._refresh_bank(epoch=self.completed_epochs)
            parameters = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            count_weight = self._objective_weight(stage, "grouped_count_likelihood")
            if growth_enabled and self.data.count_blocks and count_weight > 0:
                parameters.append(self.log_count_concentration)
            optimizer = self._optimizer(stage.optimizer, parameters)
            trace = {
                "stage": phase,
                "epochs_requested": stage.epochs,
                "precision": stage.precision,
                "optimizer": type(optimizer).__name__,
                "optimizer_kind": stage.optimizer.kind,
                "learning_rate": stage.optimizer.learning_rate,
                "weight_decay": stage.optimizer.weight_decay,
                "trainable_tags": list(stage.trainable_tags),
                "trainable_parameters": list(trainable_names),
                "objective_weights": {
                    name: self.objective_map[name].weight for name in stage.active_objectives
                },
                "objective_configs": {
                    name: dict(self.objective_map[name].config) for name in stage.active_objectives
                },
                "checkpoint_metric": stage.checkpoint_metric,
                "context_policy": stage.context_policy,
            }
            self.execution_trace.append(trace)
            completed_before_stage = self.completed_epochs
            best_score = float("inf")
            best_model: dict[str, torch.Tensor] | None = None
            best_concentration: torch.Tensor | None = None
            stale_epochs = 0
            for phase_epoch in range(stage.epochs):
                train_summary = self._train_epoch(stage, optimizer, phase_epoch)
                bank_values = self.bank.diagnostics()
                if phase in {"mass", "context"}:
                    self._refresh_bank(epoch=self.completed_epochs + 1)
                evaluation = self._evaluate_ids(
                    self.validation_measure_ids,
                    include_mass=phase != "state",
                    validation_source=self.validation_source,
                )
                validation_count, validation_count_blocks = self._validation_count_loss(stage)
                score = self._validation_score(
                    stage,
                    evaluation,
                    validation_count=validation_count,
                )
                self.history_rows.append(
                    {
                        "epoch": self.completed_epochs,
                        "phase": phase,
                        "phase_epoch": phase_epoch,
                        **train_summary,
                        "validation_objective": score,
                        "validation_observations": int(len(evaluation)),
                        "validation_source": self.validation_source,
                        "validation_strategy": self.validation_strategy,
                        "validation_count_loss": validation_count,
                        "validation_count_blocks": validation_count_blocks,
                        **bank_values,
                    }
                )
                self.completed_epochs += 1
                if score < best_score - 1e-8:
                    best_score = score
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_concentration = self.log_count_concentration.detach().clone()
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= self.training_plan.early_stopping_patience:
                    break
            trace["epochs_completed"] = self.completed_epochs - completed_before_stage
            trace["selected_checkpoint_score"] = best_score
            if best_model is not None:
                self.model.load_state_dict(best_model)
                assert best_concentration is not None
                self.log_count_concentration.data.copy_(best_concentration)
            if phase in {"mass", "context"}:
                self._refresh_bank(epoch=self.completed_epochs)
        self.metrics = self.evaluate()

    def _batches(
        self,
        measure_ids: Sequence[str],
        *,
        seed: int,
        batch_size: int,
        order: Literal["random", "target_round_robin", "target_blocked"] = "random",
        data: TrajectoryData | None = None,
    ) -> Iterable[tuple[str, ...]]:
        generator = np.random.default_rng(seed)
        if order in {"target_round_robin", "target_blocked"}:
            selected_data = self.data if data is None else data
            metadata = selected_data.measure_meta.set_index("measure_id")
            grouped: dict[str, list[str]] = {}
            for measure_id in measure_ids:
                row = metadata.loc[measure_id]
                key = str(row["embedding_id"])
                if bool(row["is_control"]):
                    key = f"__control__::{row['guide_id']}"
                grouped.setdefault(key, []).append(measure_id)
            for values in grouped.values():
                generator.shuffle(values)
            if order == "target_blocked":
                keys = list(grouped)
                generator.shuffle(keys)
                batch: list[str] = []
                for key in keys:
                    target = grouped[key]
                    if batch and len(batch) + len(target) > batch_size:
                        yield tuple(batch)
                        batch = []
                    batch.extend(target)
                if batch:
                    yield tuple(batch)
                return
            values: list[str] = []
            active = list(grouped)
            while active:
                generator.shuffle(active)
                next_active = []
                for key in active:
                    values.append(grouped[key].pop())
                    if grouped[key]:
                        next_active.append(key)
                active = next_active
        elif order == "random":
            values = list(measure_ids)
            generator.shuffle(values)
        else:  # pragma: no cover - BatchingSpec validates the order.
            raise ValueError(f"Unknown batching order {order!r}.")
        for start in range(0, len(values), batch_size):
            yield tuple(values[start : start + batch_size])

    def _provider(self, stage: Stage):
        if stage.context_policy == "catalog_bank" and self.model.context_mode == "catalog_bank":
            self.bank.assert_complete()
            return CatalogContextProvider(self.bank)
        return NoContextProvider()

    @torch.no_grad()
    def _validation_count_loss(self, stage: Stage) -> tuple[float, int]:
        if (
            self._objective_weight(stage, "grouped_count_likelihood") == 0
            or not self.validation_data.count_blocks
        ):
            return 0.0, 0
        metadata = self.validation_data.measure_meta.set_index("measure_id")
        groups = {
            str(metadata.loc[measure_id, "context_group_id"])
            for measure_id in self.validation_measure_ids
        }
        value, block_count = catalog_count_block_loss(
            self.validation_data,
            log_concentration=self.log_count_concentration,
            fitness_bank=self.validation_bank,
            context_group_ids=groups,
            time_labels=self.validation_time_labels,
        )
        return float(value.cpu()), block_count

    def _validation_score(
        self,
        stage: Stage,
        evaluation: pd.DataFrame,
        *,
        validation_count: float,
    ) -> float:
        geometry = self._objective_weight(stage, "checkpoint_geometry") * float(
            evaluation["geometry"].mean()
        )
        if stage.checkpoint_metric == "validation_geometry":
            return geometry
        if stage.checkpoint_metric != "validation_total":
            raise ValueError(
                f"Unsupported compact-v3 checkpoint metric {stage.checkpoint_metric!r}."
            )
        mass = self._objective_weight(stage, "checkpoint_mass") * float(
            evaluation["log_mass_error"].mean()
        )
        count = self._objective_weight(stage, "grouped_count_likelihood") * validation_count
        return geometry + mass + count

    def _rollout_ids(
        self,
        measure_ids: Sequence[str],
        *,
        particles: int,
        seed: int,
        provider,
        data: TrajectoryData | None = None,
    ):
        selected_data = self.data if data is None else data
        state = sample_initial_particles(
            selected_data,
            measure_ids,
            particles,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
        )
        noise = sample_noise(state, self.grid, seed=seed + 1_000_003)
        return rollout(
            self.model,
            state,
            self.grid,
            context_provider=provider,
            noise=noise,
        )

    def _training_ids_for_stage(self, stage: Stage) -> tuple[str, ...]:
        phase = stage.name
        downstream = {
            measure_id
            for label in self.train_time_labels
            for measure_id in self.data.measures[label]
        }
        count_groups = {str(block.context_group_id) for block in self.data.count_blocks}
        metadata = self.data.measure_meta.set_index("measure_id")
        use_counts = self._objective_weight(stage, "grouped_count_likelihood") > 0
        selected = tuple(
            measure_id
            for measure_id in self.train_measure_ids
            if measure_id in downstream
            or (use_counts and str(metadata.loc[measure_id, "context_group_id"]) in count_groups)
        )
        if not selected:
            raise RuntimeError(f"Phase {phase!r} has no supervised training measures.")
        return selected

    def _train_epoch(
        self,
        stage: Stage,
        optimizer: torch.optim.Optimizer,
        phase_epoch: int,
    ) -> dict[str, float | int]:
        phase = stage.name
        self.model.train()
        weighted_geometry = 0.0
        weighted_mass = 0.0
        count_total = 0.0
        regularization_total = 0.0
        observations = 0
        batch_count = 0
        seed = self.training_plan.seed + self.completed_epochs * 10_000
        training_ids = self._training_ids_for_stage(stage)
        if stage.batching.mode != "measure_batches":
            raise ValueError("compact-v3 stages require measure_batches batching.")
        assert stage.batching.measures_per_batch is not None
        for batch_index, batch_ids in enumerate(
            self._batches(
                training_ids,
                seed=seed,
                batch_size=stage.batching.measures_per_batch,
                order=stage.batching.order,
            )
        ):
            self.bank.tick()
            particle_rollout = self._rollout_ids(
                batch_ids,
                particles=self.training_plan.particles,
                seed=seed + batch_index,
                provider=self._provider(stage),
            )
            mass_weight = self._objective_weight(stage, "checkpoint_mass")
            count_weight = self._objective_weight(stage, "grouped_count_likelihood")
            geometry = self._objective(stage, "checkpoint_geometry")
            geometry_config = {} if geometry is None else dict(geometry.config)
            action = self._objective(stage, "rollout_action")
            action_config = {} if action is None else dict(action.config)
            objective = total_objective(
                particle_rollout,
                self.data,
                geometry_weight=self._objective_weight(stage, "checkpoint_geometry"),
                mass_weight=mass_weight,
                count_weight=count_weight,
                include_mass=bool(
                    {"checkpoint_mass", "grouped_count_likelihood"} & set(stage.active_objectives)
                ),
                log_concentration=self.log_count_concentration,
                fitness_bank=self.bank if phase != "state" else None,
                sinkhorn_epsilon=float(geometry_config.get("sinkhorn_epsilon", 0.1)),
                time_labels=self.train_time_labels,
                action_weights=(
                    float(action_config.get("drift_weight", 0.0))
                    * (0.0 if action is None else action.weight),
                    float(action_config.get("diffusion_weight", 0.0))
                    * (0.0 if action is None else action.weight),
                    float(action_config.get("growth_weight", 0.0))
                    * (0.0 if action is None else action.weight),
                ),
            )
            model_objective = self._objective(stage, "model_regularization")
            model_config = {} if model_objective is None else dict(model_objective.config)
            model_regularization = self.model.regularization(
                coefficient=float(model_config.get("coefficient", 0.0))
                * (0.0 if model_objective is None else model_objective.weight)
            )
            loss = objective.total + model_regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.training_plan.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ],
                    max_norm=self.training_plan.gradient_clip_norm,
                )
            optimizer.step()
            if phase in {"mass", "context"}:
                self.bank.update_from_rollout(
                    particle_rollout, self.model, self.data, full_refresh=False
                )
            count = objective.checkpoint.observation_count
            weighted_geometry += float(objective.checkpoint.geometry.detach().cpu()) * count
            weighted_mass += float(objective.checkpoint.log_mass_error.detach().cpu()) * count
            count_total += float(objective.count.detach().cpu())
            regularization_total += float(
                (objective.regularization + model_regularization).detach().cpu()
            )
            observations += count
            batch_count += 1
        if observations == 0 and not (
            self._objective_weight(stage, "grouped_count_likelihood") > 0 and self.data.count_blocks
        ):
            raise RuntimeError("Training produced no active checkpoint observations.")
        geometry_mean = weighted_geometry / max(observations, 1)
        mass_mean = weighted_mass / max(observations, 1)
        count_mean = count_total / max(batch_count, 1)
        regularization_mean = regularization_total / max(batch_count, 1)
        total_mean = (
            self._objective_weight(stage, "checkpoint_geometry") * geometry_mean
            + regularization_mean
        )
        if phase != "state":
            total_mean += self._objective_weight(stage, "checkpoint_mass") * mass_mean
            total_mean += self._objective_weight(stage, "grouped_count_likelihood") * count_mean
        return {
            "train_objective": total_mean,
            "train_geometry": geometry_mean,
            "train_log_mass_error": mass_mean,
            "train_count_loss": count_mean,
            "train_observations": observations,
            "train_batches": batch_count,
        }

    @torch.no_grad()
    def _refresh_bank(self, *, epoch: int) -> None:
        """Initialize every entry using complete context groups before optimization."""
        self._refresh_bank_for(self.data, self.bank, epoch=epoch)

    @torch.no_grad()
    def _refresh_bank_for(
        self,
        data: TrajectoryData,
        bank: CatalogBank,
        *,
        epoch: int,
    ) -> None:
        self.model.eval()
        bank.reset_coverage()
        metadata = data.measure_meta.set_index("measure_id")
        grouped: dict[str, list[str]] = {}
        for measure_id in data.measure_ids:
            grouped.setdefault(metadata.loc[measure_id, "context_group_id"], []).append(measure_id)
        particles = max(2, min(16, self.training_plan.particles))
        for group_index, group_ids in enumerate(grouped.values()):
            state = sample_initial_particles(
                data,
                group_ids,
                particles,
                device=self.device,
                dtype=self.dtype,
                seed=self.training_plan.seed + epoch * 1009 + group_index,
            )
            noise = sample_noise(
                state,
                self.grid,
                seed=self.training_plan.seed + epoch * 1009 + group_index + 2_000_003,
            )
            provider = (
                SelfConsistentContextProvider()
                if self.model.context_enabled
                else NoContextProvider()
            )
            full_group_rollout = rollout(
                self.model,
                state,
                self.grid,
                context_provider=provider,
                noise=noise,
            )
            bank.update_from_rollout(full_group_rollout, self.model, data, full_refresh=True)
        bank.last_full_refresh_epoch = int(epoch)
        bank.assert_complete()

    @torch.no_grad()
    def _evaluate_ids(
        self,
        measure_ids: Sequence[str],
        *,
        include_mass: bool,
        validation_source: str,
        particles: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        evaluation_particles = (
            self.settings.evaluation.particles if particles is None else int(particles)
        )
        if evaluation_particles < 2:
            raise ValueError("Evaluation requires at least two particles.")
        evaluation_seed = self.training_plan.seed + 9_100_001 if seed is None else int(seed)
        if evaluation_seed < 0:
            raise ValueError("Evaluation seed must be nonnegative.")
        if self.model.growth_enabled or self.model.context_enabled:
            self._refresh_bank_for(
                self.validation_data,
                self.validation_bank,
                epoch=self.completed_epochs,
            )
        if self.model.context_enabled:
            provider = CatalogContextProvider(self.validation_bank)
        else:
            provider = NoContextProvider()
        for batch_index, batch_ids in enumerate(
            self._batches(
                measure_ids,
                seed=self.training_plan.seed + 9_000_001,
                batch_size=self.settings.evaluation.measures_per_batch,
                data=self.validation_data,
            )
        ):
            particle_rollout = self._rollout_ids(
                batch_ids,
                particles=evaluation_particles,
                seed=evaluation_seed + batch_index,
                provider=provider,
                data=self.validation_data,
            )
            checkpoint = checkpoint_geometry_mass_loss(
                particle_rollout,
                self.validation_data,
                mass_weight=self.settings.loss.mass,
                include_mass=include_mass,
                validation_source=validation_source,
                sinkhorn_epsilon=self.settings.loss.sinkhorn_epsilon,
                time_labels=self.validation_time_labels,
            )
            rows.extend(checkpoint.rows)
        if not rows:
            raise RuntimeError("Evaluation produced no observed checkpoint rows.")
        return pd.DataFrame(rows)

    def evaluate(
        self,
        data: TrajectoryData | FiniteMeasureDynamicsProblem | None = None,
        *,
        particles: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Evaluate held-out measures, or training measures when no holdout exists."""
        if data is not None and all(
            data is not candidate for candidate in (self.data, self.validation_data, self.problem)
        ):
            raise ValueError("External evaluation data must be loaded as a separate Trainer run.")
        return self._evaluate_ids(
            self.validation_measure_ids,
            include_mass=self.model.growth_enabled,
            validation_source=self.validation_source,
            particles=particles,
            seed=seed,
        )

    def evaluate_runtime(
        self,
        *,
        study: TrajectoryData | FiniteMeasureDynamicsProblem | None = None,
        particles: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Adapter from compact-v3 to the stable evaluation facade."""
        if kwargs:
            raise TypeError(f"Unsupported compact-v3 evaluation options: {sorted(kwargs)}")
        from ...evaluation import standardize_compact_metrics

        frame = self.evaluate(study, particles=particles, seed=seed)
        return standardize_compact_metrics(
            self,
            frame,
            particles=self.settings.evaluation.particles if particles is None else particles,
            seed=self.training_plan.seed + 9_100_001 if seed is None else seed,
        )

    def _manifest(self) -> dict[str, Any]:
        from ... import __version__

        git_sha, git_dirty = _git_state()
        distributions = {
            "anndata": "anndata",
            "geomloss": "geomloss",
            "numpy": "numpy",
            "pandas": "pandas",
            "pyarrow": "pyarrow",
            "pydantic": "pydantic",
            "pyyaml": "PyYAML",
            "torch": "torch",
        }
        dependencies = {}
        for name, distribution in distributions.items():
            try:
                dependencies[name] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                dependencies[name] = None
        accelerator_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        )
        return {
            "schema_version": 2,
            "recipe": _compact_recipe_contract(),
            "capabilities": asdict(_compact_capabilities()),
            "resolved_config": resolved_config(self.config),
            "package_version": __version__,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "command": sys.argv,
            "dependencies": dependencies,
            "runtime": {
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
                "accelerator": {
                    "type": self.device.type,
                    "name": accelerator_name,
                    "cuda_runtime": torch.version.cuda if self.device.type == "cuda" else None,
                },
            },
            "input_hashes": self.data.metadata.get("input_hashes", {}),
            "dataset": self.data.metadata.get("dataset", {}),
            "axis": {
                "kind": self.data.axis.kind,
                "source": self.data.axis.source,
                "labels": list(self.data.axis.labels),
                "values": list(self.data.axis.values),
            },
            "mass_semantics": self.data.mass_semantics.value,
            "representation_contract": self.data.representation.to_dict(),
            "training_plan": self.training_plan.to_dict(),
            "objective_descriptors": [
                objective.to_dict() for objective in self.objective_descriptors
            ],
            "execution_trace": copy.deepcopy(self.execution_trace),
            "mass_denominators": list(self.data.metadata.get("mass_denominators", [])),
            "claim_policy": self.data.claim_policy,
            "measure_meta_hash": _measure_meta_hash(self.data),
            "validation_split": {
                "source": self.validation_source,
                "strategy": self.validation_strategy,
                "train_measure_ids": list(self.train_measure_ids),
                "validation_measure_ids": list(self.validation_measure_ids),
                "train_time_labels": list(self.train_time_labels),
                "validation_time_labels": list(self.validation_time_labels),
                **(
                    {
                        "split_id": self.data.metadata["split_plan"]["split_id"],
                        "representation_evaluation": self.data.metadata["split_plan"][
                            "representation_evaluation"
                        ],
                    }
                    if isinstance(self.data.metadata.get("split_plan"), Mapping)
                    else {}
                ),
            },
            "compiled_problem": (
                None
                if self.problem is None
                else {
                    "kind": self.problem.problem_kind,
                    "problem_hash": self.problem.problem_hash,
                    "study_content_hash": self.problem.study_content_hash,
                    "selection_hash": self.problem.selection_hash,
                }
            ),
            "split_contract": _split_contract(self),
            "checkpoint_mode": "inference_only",
            "checkpoint_sha256": self.checkpoint_sha256,
            "bank_initialization": self.bank.diagnostics(),
            "ess_thresholds": {"warning_fraction": 0.2, "failure_fraction": 0.05},
            "counterfactual_status": ("evaluated" if self.counterfactual_rows else "not_requested"),
        }

    def save(self) -> Path:
        """Write one generic run manifest, state directory, and typed result tables."""
        output_dir = Path(self.config.output)
        artifact_names = {
            "run.json",
            "state/checkpoint.pt",
            "tables/history.parquet",
            "tables/predictions.parquet",
            "tables/metrics.parquet",
            "tables/diagnostics.parquet",
            "tables/counterfactuals.parquet",
        }
        if output_dir.exists():
            unknown = sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file() and str(path.relative_to(output_dir)) not in artifact_names
            )
            if unknown:
                raise FileExistsError(
                    f"Run directory contains files outside its bundle contract: {unknown}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "state").mkdir(exist_ok=True)
        (output_dir / "tables").mkdir(exist_ok=True)
        checkpoint_path = output_dir / "state/checkpoint.pt"
        torch.save(
            {
                "schema_version": 2,
                "envelope": _compact_checkpoint_envelope(self),
                "run_contract": _checkpoint_contract(self.data, self.config),
                "validation_run_contract": _checkpoint_contract(self.validation_data, self.config),
                "compiled_problem_hash": (
                    None if self.problem is None else self.problem.problem_hash
                ),
                "architecture": self.model.architecture(),
                "model_state": self.model.state_dict(),
                "log_count_concentration": self.log_count_concentration.detach().cpu(),
                "completed_epochs": self.completed_epochs,
                "train_measure_ids": self.train_measure_ids,
                "validation_measure_ids": self.validation_measure_ids,
                "train_time_labels": self.train_time_labels,
                "validation_time_labels": self.validation_time_labels,
                "validation_source": self.validation_source,
                "validation_strategy": self.validation_strategy,
                "representation_scope": self.representation_scope,
                "split_plan": self.data.metadata.get("split_plan"),
                "training_plan": self.training_plan.to_dict(),
                "objective_descriptors": [
                    objective.to_dict() for objective in self.objective_descriptors
                ],
                "execution_trace": self.execution_trace,
            },
            checkpoint_path,
        )
        self.checkpoint_sha256 = _file_sha256(checkpoint_path)
        pd.DataFrame(self.history_rows).to_parquet(
            output_dir / "tables/history.parquet", index=False
        )
        from ...evaluation import evaluation_tables, standardize_compact_metrics

        wide_metrics = standardize_compact_metrics(
            self,
            self.metrics,
            particles=self.settings.evaluation.particles,
            seed=self.training_plan.seed + 9_100_001,
        )
        run_id = f"sha256:{self.checkpoint_sha256}"
        tables = evaluation_tables(self, wide_metrics, run_id=run_id)
        tables.predictions.to_parquet(output_dir / "tables/predictions.parquet", index=False)
        tables.metrics.to_parquet(output_dir / "tables/metrics.parquet", index=False)
        tables.diagnostics.to_parquet(output_dir / "tables/diagnostics.parquet", index=False)
        from ...counterfactual import COUNTERFACTUAL_COLUMNS

        pd.DataFrame(self.counterfactual_rows, columns=COUNTERFACTUAL_COLUMNS).to_parquet(
            output_dir / "tables/counterfactuals.parquet", index=False
        )
        from ...artifacts import write_compact_run_json

        write_compact_run_json(self)
        return output_dir

    def close(self) -> None:
        owner = getattr(self, "_semantic_owner", None)
        if owner is not None:
            owner.close()
            self._semantic_owner = None

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        data: TrajectoryData | FiniteMeasureDynamicsProblem,
        config: RunConfig,
        *,
        device: str | torch.device = "cpu",
        evaluation_overrides: dict[str, Any] | None = None,
    ) -> Trainer:
        """Reload a checkpoint into the same deterministic execution contract."""
        selected_device = torch.device(device)
        settings = config.recipe_config
        if evaluation_overrides:
            evaluation = type(settings.evaluation).model_validate(
                {**settings.evaluation.model_dump(), **evaluation_overrides}
            )
            settings = settings.model_copy(update={"evaluation": evaluation})
            config = config.model_copy(update={"recipe_config": settings})
        problem = data if isinstance(data, FiniteMeasureDynamicsProblem) else None
        training_data = data.training if problem is not None else data
        validation_data = data.validation if problem is not None else data
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
        if payload.get("schema_version") != 2:
            raise ValueError("Unsupported CREDO checkpoint schema.")
        if "envelope" not in payload:
            raise ValueError("Schema-v2 checkpoint is missing its envelope.")
        compatibility_partitioned = False
        if (
            problem is None
            and payload.get("compiled_problem_hash") is not None
            and payload.get("split_plan") is not None
        ):
            compatibility_split = SplitPlan.from_dict(payload["split_plan"])
            training_data = _partition_trajectory_data(
                data,
                compatibility_split,
                validation=False,
            )
            validation_data = _partition_trajectory_data(
                data,
                compatibility_split,
                validation=True,
            )
            compatibility_partitioned = True
        validate_run_data(config, training_data)
        validate_run_data(config, validation_data)
        from ...artifacts import CheckpointEnvelope, tensor_state_sha256
        from .recipe import recipe as compact_recipe

        envelope = CheckpointEnvelope.from_dict(payload["envelope"])
        if envelope.recipe != _compact_recipe_contract():
            raise ValueError("Checkpoint recipe disagrees with compact-v3.")
        if envelope.study_contract != payload.get("run_contract"):
            raise ValueError("Checkpoint envelope disagrees with its run contract.")
        if envelope.representation_contract != training_data.representation.to_dict():
            raise ValueError("Checkpoint representation contract disagrees with the data.")
        if envelope.capabilities != asdict(_compact_capabilities()):
            raise ValueError("Checkpoint capabilities disagree with compact-v3.")
        model_contract = envelope.state["model"]
        model_state = payload["model_state"]
        if model_contract.get("tensor_count") != len(model_state) or model_contract.get(
            "semantic_hash"
        ) != tensor_state_sha256(model_state):
            raise ValueError("Checkpoint model state disagrees with its semantic hash.")
        objective_state = {"log_count_concentration": payload["log_count_concentration"]}
        objective_contract = envelope.state["objective"]
        if objective_contract.get("semantic_hash") != tensor_state_sha256(objective_state):
            raise ValueError("Checkpoint objective state disagrees with its semantic hash.")
        payload_split = {
            "strategy": payload["validation_strategy"],
            "train_measure_ids": list(payload["train_measure_ids"]),
            "validation_measure_ids": list(payload["validation_measure_ids"]),
            "train_time_labels": list(payload["train_time_labels"]),
            "validation_time_labels": list(payload["validation_time_labels"]),
            "representation_scope": payload.get("representation_scope", "shared"),
            "representation_fit_scope": training_data.representation.fit_scope,
        }
        if payload.get("split_plan") is not None:
            raw_split_plan = payload["split_plan"]
            split_plan = SplitPlan.from_dict(raw_split_plan)
            payload_split.update(
                {
                    "split_id": split_plan.split_id,
                    "representation_evaluation": split_plan.representation_evaluation,
                    "held_out_series": list(split_plan.held_out_series),
                    "held_out_checkpoints": list(split_plan.held_out_checkpoints),
                    "held_out_observations": list(split_plan.held_out_observations),
                }
            )
            if "task_kind" in raw_split_plan:
                payload_split.update(
                    {
                        "representation_protocol": split_plan.representation_protocol,
                        "task_kind": split_plan.task_kind,
                        "held_out_subject_ids": list(split_plan.held_out_subject_ids),
                        "held_out_experimental_unit_ids": list(
                            split_plan.held_out_experimental_unit_ids
                        ),
                        "held_out_perturbation_ids": list(split_plan.held_out_perturbation_ids),
                        "held_out_construct_ids": list(split_plan.held_out_construct_ids),
                        "held_out_target_ids": list(split_plan.held_out_target_ids),
                        "held_out_context_ids": list(split_plan.held_out_context_ids),
                    }
                )
        if envelope.split_contract != payload_split:
            raise ValueError("Checkpoint split state disagrees with its envelope.")
        for name in ("training_plan", "objective_descriptors", "execution_trace"):
            if envelope.training.get(name) != payload.get(name):
                raise ValueError(
                    f"Checkpoint {name.replace('_', ' ')} disagrees with its envelope."
                )
        architecture = dict(payload["architecture"])
        model = CREDOModel(**architecture).to(selected_device)
        if not _model_matches(model, training_data, config):
            raise ValueError("Checkpoint architecture disagrees with the run data or config.")
        if not _checkpoint_contract_matches(
            payload.get("run_contract"), _checkpoint_contract(training_data, config)
        ):
            raise ValueError("Checkpoint run contract disagrees with the data or config.")
        if payload.get("validation_run_contract") is not None and not _checkpoint_contract_matches(
            payload["validation_run_contract"],
            _checkpoint_contract(validation_data, config),
        ):
            raise ValueError("Checkpoint validation contract disagrees with the data or config.")
        expected_problem_hash = (
            payload.get("compiled_problem_hash")
            if compatibility_partitioned
            else (None if problem is None else problem.problem_hash)
        )
        if payload.get("compiled_problem_hash") != expected_problem_hash:
            raise ValueError("Checkpoint compiled problem hash disagrees with the data.")
        plan = compact_recipe.training_plan(training_data, settings)
        objectives = compact_recipe.build_objectives(training_data, settings)
        if payload.get("training_plan") != plan.to_dict():
            raise ValueError("Checkpoint training plan disagrees with the run config.")
        if payload.get("objective_descriptors") != [value.to_dict() for value in objectives]:
            raise ValueError("Checkpoint objectives disagree with the run config.")
        model.load_state_dict(payload["model_state"])
        grid = axis_grid(
            training_data.axis,
            plan.steps_per_interval,
            device=selected_device,
            dtype=torch.float32,
        )
        bank = CatalogBank.empty(
            training_data,
            model,
            len(grid) - 1,
            device=selected_device,
            dtype=torch.float32,
        )
        validation_bank = CatalogBank.empty(
            validation_data,
            model,
            len(grid) - 1,
            device=selected_device,
            dtype=torch.float32,
        )
        trainer = cls(
            data=training_data,
            validation_data=validation_data,
            problem=problem,
            model=model,
            config=config,
            training_plan=plan,
            objective_descriptors=objectives,
            device=selected_device,
            dtype=torch.float32,
            log_count_concentration=torch.nn.Parameter(
                payload["log_count_concentration"].to(selected_device)
            ),
            bank=bank,
            validation_bank=validation_bank,
            train_measure_ids=tuple(payload["train_measure_ids"]),
            validation_measure_ids=tuple(payload["validation_measure_ids"]),
            train_time_labels=tuple(payload["train_time_labels"]),
            validation_time_labels=tuple(payload["validation_time_labels"]),
            validation_source=payload["validation_source"],
            validation_strategy=payload["validation_strategy"],
            representation_scope=payload.get("representation_scope", "shared"),
            execution_trace=list(payload.get("execution_trace", ())),
            completed_epochs=int(payload["completed_epochs"]),
            checkpoint_sha256=_file_sha256(checkpoint_path),
        )
        final_phase = next(
            stage.name for stage in reversed(trainer.training_plan.stages) if stage.epochs > 0
        )
        trainer.model.set_phase(final_phase)  # type: ignore[arg-type]
        if final_phase in {"mass", "context"}:
            trainer._refresh_bank(epoch=trainer.completed_epochs)
        trainer.metrics = trainer.evaluate()
        return trainer


def _validation_split(
    data: TrajectoryData,
    config: RunConfig,
    *,
    seed: int | None = None,
) -> SplitPlan:
    """Compatibility entry point backed by the shared pre-compilation planner."""
    return plan_compact_trajectory_split(data, config, seed=seed)


class _TrajectorySubset(Mapping[str, Mapping[str, Any]]):
    """Lazy checkpoint/measure subset used by schema-v2 checkpoint compatibility."""

    is_lazy = True

    def __init__(
        self,
        data: TrajectoryData,
        measure_ids: tuple[str, ...],
        target_labels: tuple[str, ...],
    ) -> None:
        self._data = data
        selected = set(measure_ids)
        targets = set(target_labels)
        self.latent_dim = data.latent_dim
        self._ids = {
            label: tuple(
                measure_id
                for measure_id in measure_ids
                if measure_id in data.measures[label]
                and (label == data.axis.source or label in targets)
            )
            for label in data.axis.labels
        }
        self._selected = selected

    def __getitem__(self, label: str) -> Mapping[str, Any]:
        label = str(label)
        return MappingProxyType(
            {measure_id: self._data.measures[label][measure_id] for measure_id in self._ids[label]}
        )

    def __iter__(self):
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


def _partition_trajectory_data(
    data: TrajectoryData,
    split: SplitPlan,
    *,
    validation: bool,
) -> TrajectoryData:
    measure_ids = split.validation_measure_ids if validation else split.train_measure_ids
    target_labels = split.validation_time_labels if validation else split.train_time_labels
    metadata = data.measure_meta.loc[data.measure_meta["measure_id"].isin(measure_ids)].copy()
    order = {measure_id: index for index, measure_id in enumerate(metadata["measure_id"])}
    full_metadata = data.measure_meta.reset_index(drop=True)
    selected = set(measure_ids)
    blocks: list[CountBlock] = []
    for block in data.count_blocks:
        if block.time_label not in target_labels:
            continue
        original_indices = block.measure_indices.detach().cpu().numpy().astype(int)
        original_ids = full_metadata.iloc[original_indices]["measure_id"].astype(str).tolist()
        mask = np.asarray([measure_id in selected for measure_id in original_ids], dtype=bool)
        if not mask.any():
            continue
        if not mask.all() and block.conditioning_policy == "require_complete":
            raise ValueError(
                "Compatibility checkpoint split cuts a complete composition denominator."
            )
        selected_ids = [value for value, keep in zip(original_ids, mask, strict=True) if keep]
        exposure = block.exposure.detach().cpu().numpy()[mask]
        counts = block.counts.detach().cpu().numpy()[mask]
        modeled_denominator = block.modeled_denominator_id
        policy = block.conditioning_policy
        if not mask.all():
            policy = "condition_on_selection"
            modeled_denominator = (
                f"{block.source_denominator_id}|conditioned:{split.split_id[-16:]}"
            )
        blocks.append(
            CountBlock(
                context_group_id=block.context_group_id,
                time_label=block.time_label,
                measure_indices=np.asarray([order[value] for value in selected_ids]),
                exposure=exposure,
                counts=counts,
                background_series_ids=block.background_series_ids,
                background_fitness=block.background_fitness,
                background_exposure=block.background_exposure,
                background_counts=block.background_counts,
                source_denominator_id=block.source_denominator_id,
                modeled_denominator_id=modeled_denominator,
                conditioning_policy=policy,
            )
        )
    runtime_metadata = dict(data.metadata)
    runtime_metadata["split_plan"] = split.to_dict()
    return TrajectoryData(
        axis=data.axis,
        measures=_TrajectorySubset(data, tuple(measure_ids), tuple(target_labels)),
        measure_meta=metadata,
        mass_semantics=data.mass_semantics,
        count_blocks=tuple(blocks),
        metadata=runtime_metadata,
        representation=data.representation,
    )


def _representation_scope(
    data: TrajectoryData,
    split: SplitPlan,
    config: RunConfig,
) -> Literal["shared", "nested"]:
    representation = data.representation
    inferred: Literal["shared", "nested"] = (
        "nested"
        if representation.fit_scope in {"training_fold_source", "training_split"}
        else "shared"
    )
    if inferred == "nested" and split.source == "held_out":
        if split.strategy == "checkpoint_holdout":
            if not representation.included_time_labels:
                raise ValueError(
                    "Nested checkpoint validation requires recorded representation times."
                )
            leaked = set(split.validation_time_labels) & set(representation.included_time_labels)
            if leaked:
                raise ValueError(
                    "Nested checkpoint validation includes held-out representation times: "
                    f"{sorted(leaked)}"
                )
        elif split.strategy == "context_group_holdout":
            if not representation.included_samples:
                raise ValueError(
                    "Nested sample validation requires recorded representation samples."
                )
            metadata = data.measure_meta.set_index("measure_id")
            validation_samples = set(
                metadata.loc[list(split.validation_measure_ids), "sample_id"].astype(str)
            )
            leaked = validation_samples & set(representation.included_samples)
            if leaked:
                raise ValueError(
                    "Nested sample validation includes held-out representation samples: "
                    f"{sorted(leaked)}"
                )
        else:
            raise ValueError(
                "Nested within-sample validation cannot be verified without measure-level "
                "representation provenance."
            )
    requested = config.recipe_config.validation.representation_scope
    if requested != inferred:
        raise ValueError(
            f"validation.representation_scope={requested!r} disagrees with the "
            f"representation artifact ({inferred!r})."
        )
    return inferred


def _validate_holdout_embeddings(
    metadata: pd.DataFrame,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> None:
    if not train_ids or not validation_ids:
        raise ValueError(
            "Explicit context-group validation requires nonempty train and holdout sets."
        )
    train_embeddings = set(metadata.loc[list(train_ids), "embedding_id"])
    validation_embeddings = set(metadata.loc[list(validation_ids), "embedding_id"])
    missing = validation_embeddings - train_embeddings
    if missing:
        raise ValueError(
            "Validation embeddings must be represented in training; "
            f"missing={sorted(map(str, missing))[:5]}."
        )


def _model_matches(model: CREDOModel, data: TrajectoryData, config: RunConfig) -> bool:
    settings = config.recipe_config
    expected = {
        "embedding_ids": list(data.embedding_ids),
        "control_embedding_ids": sorted(data.control_embedding_ids),
        "latent_dim": data.latent_dim,
        "embedding_dim": settings.model.embedding_dim,
        "n_programs": settings.model.n_programs,
        "hidden_dim": settings.model.hidden_dim,
        "context_mode": settings.model.context,
        "sigma_min": 1e-3,
        "growth_max": settings.model.growth_max,
        "payoff_rank": min(4, settings.model.n_programs),
    }
    return model.architecture() == expected


def _measure_meta_hash(data: TrajectoryData) -> str:
    canonical = data.measure_meta.loc[
        :, [column for column in MEASURE_META_COLUMNS if column in data.measure_meta]
    ]
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_contract(data: TrajectoryData, config: RunConfig) -> dict[str, Any]:
    settings = config.recipe_config
    input_hashes = data.metadata.get("input_hashes", {})
    contract = {
        "axis": {
            "kind": data.axis.kind,
            "source": data.axis.source,
            "labels": list(data.axis.labels),
            "values": list(data.axis.values),
        },
        "mass_semantics": data.mass_semantics.value,
        "measure_meta_hash": _measure_meta_hash(data),
        "input_hashes": {
            str(name): str(value) for name, value in sorted(dict(input_hashes).items())
        },
        "model": settings.model.model_dump(mode="json"),
        "training": settings.training.model_dump(mode="json"),
        "validation": settings.validation.model_dump(mode="json"),
        "loss": settings.loss.model_dump(mode="json"),
    }
    semantic_problem = {
        name: data.metadata.get(name)
        for name in ("study_content_hash", "selection_hash", "compiled_problem_hash")
        if data.metadata.get(name) is not None
    }
    if semantic_problem:
        contract["semantic_problem"] = semantic_problem
    return contract


def _checkpoint_contract_matches(saved: Any, current: Mapping[str, Any]) -> bool:
    if saved == current:
        return True
    if not isinstance(saved, Mapping):
        return False
    saved_base = copy.deepcopy(dict(saved))
    current_base = copy.deepcopy(dict(current))
    saved_semantic = saved_base.pop("semantic_problem", None)
    current_semantic = current_base.pop("semantic_problem", None)
    if (saved_semantic is None) == (current_semantic is None):
        return False
    # Schema-v2 five-file callers remain reloadable through their verified file
    # hashes in either migration direction. Native-only studies lack that bridge.
    return bool(current_base.get("input_hashes")) and saved_base == current_base


def _compact_capabilities():
    from .recipe import recipe

    return recipe.capabilities


def _compact_recipe_contract() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    files = (
        root / "artifacts.py",
        root / "contracts.py",
        root / "counterfactual.py",
        root / "evaluation.py",
        root / "runtime.py",
        root / "data/splits.py",
        root / "recipes/trajectory_compiler.py",
        root / "recipes/compact_sde_v3/model.py",
        root / "recipes/compact_sde_v3/objective.py",
        root / "recipes/compact_sde_v3/particles.py",
        root / "recipes/compact_sde_v3/recipe.py",
        root / "recipes/compact_sde_v3/training.py",
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return {
        "id": "credo.compact_sde_v3",
        "version": "3.0",
        "implementation_hash": digest.hexdigest(),
    }


def _split_contract(trainer: Trainer) -> dict[str, Any]:
    contract = {
        "strategy": trainer.validation_strategy,
        "train_measure_ids": list(trainer.train_measure_ids),
        "validation_measure_ids": list(trainer.validation_measure_ids),
        "train_time_labels": list(trainer.train_time_labels),
        "validation_time_labels": list(trainer.validation_time_labels),
        "representation_scope": trainer.representation_scope,
        "representation_fit_scope": trainer.data.representation.fit_scope,
    }
    raw_split = trainer.data.metadata.get("split_plan")
    if isinstance(raw_split, Mapping):
        split = SplitPlan.from_dict(raw_split)
        contract.update(
            {
                "split_id": split.split_id,
                "representation_evaluation": split.representation_evaluation,
                "held_out_series": list(split.held_out_series),
                "held_out_checkpoints": list(split.held_out_checkpoints),
                "held_out_observations": list(split.held_out_observations),
            }
        )
        if "task_kind" in raw_split:
            contract.update(
                {
                    "representation_protocol": split.representation_protocol,
                    "task_kind": split.task_kind,
                    "held_out_subject_ids": list(split.held_out_subject_ids),
                    "held_out_experimental_unit_ids": list(split.held_out_experimental_unit_ids),
                    "held_out_perturbation_ids": list(split.held_out_perturbation_ids),
                    "held_out_construct_ids": list(split.held_out_construct_ids),
                    "held_out_target_ids": list(split.held_out_target_ids),
                    "held_out_context_ids": list(split.held_out_context_ids),
                }
            )
    return contract


def _compact_checkpoint_envelope(trainer: Trainer) -> dict[str, Any]:
    from ...artifacts import CheckpointEnvelope, CheckpointMode, tensor_state_sha256

    state = trainer.model.state_dict()
    objective_state = {"log_count_concentration": trainer.log_count_concentration.detach().cpu()}
    return CheckpointEnvelope(
        recipe=_compact_recipe_contract(),
        study_contract=_checkpoint_contract(trainer.data, trainer.config),
        representation_contract=trainer.data.representation.to_dict(),
        split_contract=_split_contract(trainer),
        state={
            "model": {
                "source_key": "model_state",
                "tensor_count": len(state),
                "semantic_hash": tensor_state_sha256(state),
            },
            "ema": None,
            "representation": {"embedded": False},
            "objective": {
                "source_key": "log_count_concentration",
                "tensor_count": len(objective_state),
                "semantic_hash": tensor_state_sha256(objective_state),
            },
            "optimizer": None,
            "scheduler": None,
            "rng": None,
        },
        training={
            "completed_epochs": trainer.completed_epochs,
            "training_recipe_available": True,
            "resume_supported": False,
            "deterministic_cpu_fresh_fit_tested": True,
            "training_plan": trainer.training_plan.to_dict(),
            "objective_descriptors": [
                objective.to_dict() for objective in trainer.objective_descriptors
            ],
            "execution_trace": copy.deepcopy(trainer.execution_trace),
        },
        capabilities=asdict(_compact_capabilities()),
        mode=CheckpointMode.INFERENCE_ONLY,
    ).to_dict()


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[4]
    if not (repository / ".git").exists():
        return None, None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                cwd=repository,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None
