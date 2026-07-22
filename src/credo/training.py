"""One catalog bank, fixed continuation schedule, evaluation, and persistence."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from .contracts import MEASURE_META_COLUMNS, TrajectoryData
from .io import RunConfig, resolved_config, validate_run_data
from .model import CREDOModel
from .objective import (
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


@dataclass(frozen=True)
class ValidationSplit:
    train_measure_ids: tuple[str, ...]
    validation_measure_ids: tuple[str, ...]
    train_time_labels: tuple[str, ...]
    validation_time_labels: tuple[str, ...]
    source: Literal["held_out", "train_self_eval"]
    strategy: Literal[
        "context_group_holdout",
        "checkpoint_holdout",
        "within_embedding_holdout",
        "train_self_eval",
    ]


@dataclass
class Trainer:
    """The only CREDO trainer, with a fixed state to mass to context schedule."""

    data: TrajectoryData
    model: CREDOModel
    config: RunConfig
    device: torch.device
    dtype: torch.dtype
    log_count_concentration: torch.nn.Parameter
    bank: CatalogBank
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
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    counterfactual_rows: list[dict[str, Any]] = field(default_factory=list)
    completed_epochs: int = 0
    checkpoint_sha256: str | None = None

    @classmethod
    def fit(
        cls,
        data: TrajectoryData,
        model: CREDOModel | None,
        config: RunConfig,
        *,
        device: str | torch.device | None = None,
    ) -> Trainer:
        """Construct and fit one run."""
        selected_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float32
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(config.training.seed)
        validate_run_data(config, data)
        if model is None:
            model = CREDOModel(
                embedding_ids=data.embedding_ids,
                control_embedding_ids=data.control_embedding_ids,
                latent_dim=data.latent_dim,
                embedding_dim=config.model.embedding_dim,
                n_programs=config.model.n_programs,
                hidden_dim=config.model.hidden_dim,
                context_mode=config.model.context,
                growth_max=config.model.growth_max,
            )
        elif not _model_matches(model, data, config):
            raise ValueError("Provided model architecture disagrees with the run data or config.")
        model = model.to(device=selected_device, dtype=dtype)
        model.assert_soft_reference()
        validate_count_blocks(data)
        split = _validation_split(data, config)
        grid = axis_grid(
            data.axis,
            config.training.steps_per_interval,
            device=selected_device,
            dtype=dtype,
        )
        bank = CatalogBank.empty(
            data,
            model,
            len(grid) - 1,
            device=selected_device,
            dtype=dtype,
        )
        trainer = cls(
            data=data,
            model=model,
            config=config,
            device=selected_device,
            dtype=dtype,
            log_count_concentration=torch.nn.Parameter(
                torch.tensor(np.log(100.0), device=selected_device, dtype=dtype)
            ),
            bank=bank,
            train_measure_ids=split.train_measure_ids,
            validation_measure_ids=split.validation_measure_ids,
            train_time_labels=split.train_time_labels,
            validation_time_labels=split.validation_time_labels,
            validation_source=split.source,
            validation_strategy=split.strategy,
        )
        trainer._fit()
        return trainer

    @property
    def grid(self) -> torch.Tensor:
        return axis_grid(
            self.data.axis,
            self.config.training.steps_per_interval,
            device=self.device,
            dtype=self.dtype,
        )

    def _phase_epochs(self) -> tuple[tuple[str, int], ...]:
        epochs = self.config.training.epochs
        return (("state", epochs.state), ("mass", epochs.mass), ("context", epochs.context))

    def _fit(self) -> None:
        torch.manual_seed(self.config.training.seed)
        np.random.seed(self.config.training.seed)
        for phase, epoch_count in self._phase_epochs():
            if epoch_count == 0:
                continue
            self.model.set_phase(phase)  # type: ignore[arg-type]
            if phase in {"mass", "context"}:
                self._refresh_bank(epoch=self.completed_epochs)
            parameters = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            if phase != "state" and self.data.count_blocks and self.config.loss.count > 0:
                parameters.append(self.log_count_concentration)
            optimizer = torch.optim.Adam(parameters, lr=self.config.training.learning_rate)
            best_score = float("inf")
            best_model: dict[str, torch.Tensor] | None = None
            best_concentration: torch.Tensor | None = None
            stale_epochs = 0
            for phase_epoch in range(epoch_count):
                train_summary = self._train_epoch(phase, optimizer, phase_epoch)
                bank_values = self.bank.diagnostics()
                if phase in {"mass", "context"}:
                    self._refresh_bank(epoch=self.completed_epochs + 1)
                evaluation = self._evaluate_ids(
                    self.validation_measure_ids,
                    include_mass=phase != "state",
                    validation_source=self.validation_source,
                )
                validation_count, validation_count_blocks = self._validation_count_loss(phase)
                score = float(
                    evaluation["geometry"].mean()
                    + (
                        self.config.loss.mass * evaluation["log_mass_error"].mean()
                        if phase != "state"
                        else 0.0
                    )
                    + self.config.loss.count * validation_count
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
                if stale_epochs >= self.config.training.patience:
                    break
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
        target_balanced: bool = False,
    ) -> Iterable[tuple[str, ...]]:
        generator = np.random.default_rng(seed)
        if target_balanced:
            metadata = self.data.measure_meta.set_index("measure_id")
            grouped: dict[str, list[str]] = {}
            for measure_id in measure_ids:
                row = metadata.loc[measure_id]
                key = str(row["embedding_id"])
                if bool(row["is_control"]):
                    key = f"__control__::{row['guide_id']}"
                grouped.setdefault(key, []).append(measure_id)
            for values in grouped.values():
                generator.shuffle(values)
            values = []
            active = list(grouped)
            while active:
                generator.shuffle(active)
                next_active = []
                for key in active:
                    values.append(grouped[key].pop())
                    if grouped[key]:
                        next_active.append(key)
                active = next_active
        else:
            values = list(measure_ids)
            generator.shuffle(values)
        for start in range(0, len(values), batch_size):
            yield tuple(values[start : start + batch_size])

    def _provider(self, phase: str):
        if phase == "context" and self.model.context_mode == "catalog_bank":
            self.bank.assert_complete()
            return CatalogContextProvider(self.bank)
        return NoContextProvider()

    @torch.no_grad()
    def _validation_count_loss(self, phase: str) -> tuple[float, int]:
        if phase == "state" or self.config.loss.count == 0 or not self.data.count_blocks:
            return 0.0, 0
        metadata = self.data.measure_meta.set_index("measure_id")
        groups = {
            str(metadata.loc[measure_id, "context_group_id"])
            for measure_id in self.validation_measure_ids
        }
        value, block_count = catalog_count_block_loss(
            self.data,
            log_concentration=self.log_count_concentration,
            fitness_bank=self.bank,
            context_group_ids=groups,
            time_labels=self.validation_time_labels,
        )
        return float(value.cpu()), block_count

    def _rollout_ids(
        self,
        measure_ids: Sequence[str],
        *,
        particles: int,
        seed: int,
        provider,
    ):
        state = sample_initial_particles(
            self.data,
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

    def _training_ids_for_phase(self, phase: str) -> tuple[str, ...]:
        downstream = {
            measure_id
            for label in self.train_time_labels
            for measure_id in self.data.measures[label]
        }
        count_groups = {str(block.context_group_id) for block in self.data.count_blocks}
        metadata = self.data.measure_meta.set_index("measure_id")
        use_counts = phase != "state" and self.config.loss.count > 0
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
        phase: str,
        optimizer: torch.optim.Optimizer,
        phase_epoch: int,
    ) -> dict[str, float | int]:
        self.model.train()
        weighted_geometry = 0.0
        weighted_mass = 0.0
        count_total = 0.0
        regularization_total = 0.0
        observations = 0
        batch_count = 0
        seed = self.config.training.seed + self.completed_epochs * 10_000
        training_ids = self._training_ids_for_phase(phase)
        for batch_index, batch_ids in enumerate(
            self._batches(
                training_ids,
                seed=seed,
                batch_size=self.config.training.measures_per_batch,
                target_balanced=self.config.training.batching == "target_balanced",
            )
        ):
            self.bank.tick()
            particle_rollout = self._rollout_ids(
                batch_ids,
                particles=self.config.training.particles,
                seed=seed + batch_index,
                provider=self._provider(phase),
            )
            objective = total_objective(
                particle_rollout,
                self.data,
                mass_weight=self.config.loss.mass,
                count_weight=self.config.loss.count,
                include_mass=phase != "state",
                log_concentration=self.log_count_concentration,
                fitness_bank=self.bank if phase != "state" else None,
                sinkhorn_epsilon=self.config.loss.sinkhorn_epsilon,
                time_labels=self.train_time_labels,
            )
            model_regularization = self.model.regularization()
            loss = objective.total + model_regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                max_norm=10.0,
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
            phase != "state" and self.config.loss.count > 0 and self.data.count_blocks
        ):
            raise RuntimeError("Training produced no active checkpoint observations.")
        geometry_mean = weighted_geometry / max(observations, 1)
        mass_mean = weighted_mass / max(observations, 1)
        count_mean = count_total / max(batch_count, 1)
        regularization_mean = regularization_total / max(batch_count, 1)
        total_mean = geometry_mean + regularization_mean
        if phase != "state":
            total_mean += self.config.loss.mass * mass_mean
            total_mean += self.config.loss.count * count_mean
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
        self.model.eval()
        self.bank.reset_coverage()
        metadata = self.data.measure_meta.set_index("measure_id")
        grouped: dict[str, list[str]] = {}
        for measure_id in self.data.measure_ids:
            grouped.setdefault(metadata.loc[measure_id, "context_group_id"], []).append(measure_id)
        particles = max(2, min(16, self.config.training.particles))
        for group_index, group_ids in enumerate(grouped.values()):
            state = sample_initial_particles(
                self.data,
                group_ids,
                particles,
                device=self.device,
                dtype=self.dtype,
                seed=self.config.training.seed + epoch * 1009 + group_index,
            )
            noise = sample_noise(
                state,
                self.grid,
                seed=self.config.training.seed + epoch * 1009 + group_index + 2_000_003,
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
            self.bank.update_from_rollout(
                full_group_rollout, self.model, self.data, full_refresh=True
            )
        self.bank.last_full_refresh_epoch = int(epoch)
        self.bank.assert_complete()

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
            self.config.evaluation.particles if particles is None else int(particles)
        )
        if evaluation_particles < 2:
            raise ValueError("Evaluation requires at least two particles.")
        evaluation_seed = self.config.training.seed + 9_100_001 if seed is None else int(seed)
        if evaluation_seed < 0:
            raise ValueError("Evaluation seed must be nonnegative.")
        provider = (
            CatalogContextProvider(self.bank) if self.model.context_enabled else NoContextProvider()
        )
        for batch_index, batch_ids in enumerate(
            self._batches(
                measure_ids,
                seed=self.config.training.seed + 9_000_001,
                batch_size=self.config.evaluation.measures_per_batch,
            )
        ):
            particle_rollout = self._rollout_ids(
                batch_ids,
                particles=evaluation_particles,
                seed=evaluation_seed + batch_index,
                provider=provider,
            )
            checkpoint = checkpoint_geometry_mass_loss(
                particle_rollout,
                self.data,
                mass_weight=self.config.loss.mass,
                include_mass=include_mass,
                validation_source=validation_source,
                sinkhorn_epsilon=self.config.loss.sinkhorn_epsilon,
                time_labels=self.validation_time_labels,
            )
            rows.extend(checkpoint.rows)
        if not rows:
            raise RuntimeError("Evaluation produced no observed checkpoint rows.")
        return pd.DataFrame(rows)

    def evaluate(
        self,
        data: TrajectoryData | None = None,
        *,
        particles: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Evaluate held-out measures, or training measures when no holdout exists."""
        if data is not None and data is not self.data:
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
        study: TrajectoryData | None = None,
        particles: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Adapter from compact-v3 to the stable evaluation facade."""
        if kwargs:
            raise TypeError(f"Unsupported compact-v3 evaluation options: {sorted(kwargs)}")
        from .evaluation import standardize_compact_metrics

        frame = self.evaluate(study, particles=particles, seed=seed)
        return standardize_compact_metrics(
            self,
            frame,
            particles=self.config.evaluation.particles if particles is None else particles,
            seed=self.config.training.seed + 9_100_001 if seed is None else seed,
        )

    def _manifest(self) -> dict[str, Any]:
        from . import __version__

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
        return {
            "schema_version": 1,
            "recipe": _compact_recipe_contract(),
            "capabilities": asdict(_compact_capabilities()),
            "resolved_config": resolved_config(self.config),
            "package_version": __version__,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "command": sys.argv,
            "dependencies": dependencies,
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
            },
            "split_contract": _split_contract(self),
            "checkpoint_mode": "inference_only",
            "checkpoint_sha256": self.checkpoint_sha256,
            "bank_initialization": self.bank.diagnostics(),
            "ess_thresholds": {"warning_fraction": 0.2, "failure_fraction": 0.05},
            "counterfactual_status": ("evaluated" if self.counterfactual_rows else "not_requested"),
        }

    def save(self) -> Path:
        """Write exactly five durable run artifacts."""
        output_dir = Path(self.config.output)
        artifact_names = {
            "manifest.json",
            "checkpoint.pt",
            "history.parquet",
            "metrics.parquet",
            "counterfactuals.parquet",
        }
        if output_dir.exists():
            unknown = sorted(
                path.name for path in output_dir.iterdir() if path.name not in artifact_names
            )
            if unknown:
                raise FileExistsError(
                    f"Run directory contains files outside the five-artifact contract: {unknown}"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "checkpoint.pt"
        torch.save(
            {
                "schema_version": 2,
                "envelope": _compact_checkpoint_envelope(self),
                "run_contract": _checkpoint_contract(self.data, self.config),
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
            },
            checkpoint_path,
        )
        self.checkpoint_sha256 = _file_sha256(checkpoint_path)
        pd.DataFrame(self.history_rows).to_parquet(output_dir / "history.parquet", index=False)
        self.metrics.to_parquet(output_dir / "metrics.parquet", index=False)
        from .counterfactual import COUNTERFACTUAL_COLUMNS

        pd.DataFrame(self.counterfactual_rows, columns=COUNTERFACTUAL_COLUMNS).to_parquet(
            output_dir / "counterfactuals.parquet", index=False
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(self._manifest(), indent=2) + "\n", encoding="utf-8"
        )
        return output_dir

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        data: TrajectoryData,
        config: RunConfig,
        *,
        device: str | torch.device = "cpu",
        evaluation_overrides: dict[str, Any] | None = None,
    ) -> Trainer:
        """Reload a checkpoint into the same deterministic execution contract."""
        selected_device = torch.device(device)
        if evaluation_overrides:
            evaluation = type(config.evaluation).model_validate(
                {**config.evaluation.model_dump(), **evaluation_overrides}
            )
            config = config.model_copy(update={"evaluation": evaluation})
        validate_run_data(config, data)
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
        if payload.get("schema_version") != 2:
            raise ValueError("Unsupported CREDO checkpoint schema.")
        if "envelope" in payload:
            from .artifacts import CheckpointEnvelope

            envelope = CheckpointEnvelope.from_dict(payload["envelope"])
            if envelope.recipe != _compact_recipe_contract():
                raise ValueError("Checkpoint recipe disagrees with compact-v3.")
            if envelope.representation_contract != data.representation.to_dict():
                raise ValueError("Checkpoint representation contract disagrees with the data.")
        architecture = dict(payload["architecture"])
        model = CREDOModel(**architecture).to(selected_device)
        if not _model_matches(model, data, config):
            raise ValueError("Checkpoint architecture disagrees with the run data or config.")
        if payload.get("run_contract") != _checkpoint_contract(data, config):
            raise ValueError("Checkpoint run contract disagrees with the data or config.")
        model.load_state_dict(payload["model_state"])
        grid = axis_grid(
            data.axis,
            config.training.steps_per_interval,
            device=selected_device,
            dtype=torch.float32,
        )
        bank = CatalogBank.empty(
            data,
            model,
            len(grid) - 1,
            device=selected_device,
            dtype=torch.float32,
        )
        trainer = cls(
            data=data,
            model=model,
            config=config,
            device=selected_device,
            dtype=torch.float32,
            log_count_concentration=torch.nn.Parameter(
                payload["log_count_concentration"].to(selected_device)
            ),
            bank=bank,
            train_measure_ids=tuple(payload["train_measure_ids"]),
            validation_measure_ids=tuple(payload["validation_measure_ids"]),
            train_time_labels=tuple(payload["train_time_labels"]),
            validation_time_labels=tuple(payload["validation_time_labels"]),
            validation_source=payload["validation_source"],
            validation_strategy=payload["validation_strategy"],
            completed_epochs=int(payload["completed_epochs"]),
            checkpoint_sha256=_file_sha256(checkpoint_path),
        )
        final_phase = "context" if config.training.epochs.context else "mass"
        if not config.training.epochs.context and not config.training.epochs.mass:
            final_phase = "state"
        trainer.model.set_phase(final_phase)  # type: ignore[arg-type]
        if final_phase in {"mass", "context"}:
            trainer._refresh_bank(epoch=trainer.completed_epochs)
        trainer.metrics = trainer.evaluate()
        return trainer


def _validation_split(
    data: TrajectoryData,
    config: RunConfig,
) -> ValidationSplit:
    downstream_labels = tuple(data.axis.labels[1:])
    eligible = [
        measure_id
        for measure_id in data.measure_ids
        if any(measure_id in data.measures[label] for label in downstream_labels)
    ]
    if not eligible:
        raise ValueError("No source measure has a downstream observation.")
    metadata = data.measure_meta.set_index("measure_id")
    validation = config.validation
    if validation.strategy == "checkpoint":
        validation_times = tuple(
            label for label in downstream_labels if label in set(validation.values)
        )
        train_times = tuple(label for label in downstream_labels if label not in validation_times)
        validation_ids = tuple(
            measure_id
            for measure_id in data.measure_ids
            if any(measure_id in data.measures[label] for label in validation_times)
        )
        if not validation_ids:
            raise ValueError("Explicit checkpoint validation has no observed measures.")
        return ValidationSplit(
            train_measure_ids=data.measure_ids,
            validation_measure_ids=validation_ids,
            train_time_labels=train_times,
            validation_time_labels=validation_times,
            source="held_out",
            strategy="checkpoint_holdout",
        )

    if validation.strategy == "context_group":
        available = set(metadata["context_group_id"].astype(str))
        requested = set(validation.values)
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown validation context groups: {sorted(unknown)}")
        validation_ids = tuple(
            measure_id
            for measure_id in eligible
            if str(metadata.loc[measure_id, "context_group_id"]) in requested
        )
        train_ids = tuple(
            measure_id
            for measure_id in data.measure_ids
            if str(metadata.loc[measure_id, "context_group_id"]) not in requested
        )
        _validate_holdout_embeddings(metadata, train_ids, validation_ids)
        return ValidationSplit(
            train_measure_ids=train_ids,
            validation_measure_ids=validation_ids,
            train_time_labels=downstream_labels,
            validation_time_labels=downstream_labels,
            source="held_out",
            strategy="context_group_holdout",
        )

    fraction = validation.fraction
    seed = config.training.seed
    if validation.strategy == "train_self_eval" or fraction <= 0 or len(eligible) < 2:
        return ValidationSplit(
            train_measure_ids=data.measure_ids,
            validation_measure_ids=tuple(eligible),
            train_time_labels=downstream_labels,
            validation_time_labels=downstream_labels,
            source="train_self_eval",
            strategy="train_self_eval",
        )

    context_groups = tuple(
        dict.fromkeys(metadata.loc[list(data.measure_ids), "context_group_id"].tolist())
    )
    if len(context_groups) > 1:
        holdout_count = min(
            max(1, int(round(len(context_groups) * fraction))),
            len(context_groups) - 1,
        )
        ordered_groups = sorted(
            context_groups,
            key=lambda value: hashlib.sha256(f"{seed}:group:{value}".encode()).hexdigest(),
        )
        for offset in range(len(ordered_groups)):
            held_out: set[str] = set()
            rotated = ordered_groups[offset:] + ordered_groups[:offset]
            for candidate in rotated:
                trial = held_out | {candidate}
                validation = tuple(
                    measure_id
                    for measure_id in eligible
                    if metadata.loc[measure_id, "context_group_id"] in trial
                )
                train = tuple(
                    measure_id
                    for measure_id in data.measure_ids
                    if metadata.loc[measure_id, "context_group_id"] not in trial
                )
                train_embeddings = {
                    metadata.loc[measure_id, "embedding_id"] for measure_id in train
                }
                validation_embeddings = {
                    metadata.loc[measure_id, "embedding_id"] for measure_id in validation
                }
                if train and validation and validation_embeddings <= train_embeddings:
                    held_out = trial
                if len(held_out) == holdout_count:
                    return ValidationSplit(
                        train_measure_ids=train,
                        validation_measure_ids=validation,
                        train_time_labels=downstream_labels,
                        validation_time_labels=downstream_labels,
                        source="held_out",
                        strategy="context_group_holdout",
                    )

    # Count outcomes are compositional within a whole context group. If a clean
    # group holdout is impossible, measure-level holdout would leak its counts.
    if data.count_blocks:
        return ValidationSplit(
            train_measure_ids=data.measure_ids,
            validation_measure_ids=tuple(eligible),
            train_time_labels=downstream_labels,
            validation_time_labels=downstream_labels,
            source="train_self_eval",
            strategy="train_self_eval",
        )

    validation_values: list[str] = []
    for embedding_id, rows in metadata.loc[eligible].groupby("embedding_id", observed=True):
        ids = rows.index.tolist()
        guides: dict[str, list[str]] = {}
        for measure_id in ids:
            guides.setdefault(str(metadata.loc[measure_id, "guide_id"]), []).append(measure_id)
        if len(guides) > 1:
            holdout_count = min(
                max(1, int(round(len(guides) * fraction))),
                len(guides) - 1,
            )
            ordered_guides = sorted(
                guides,
                key=lambda value: hashlib.sha256(
                    f"{seed}:guide:{embedding_id}:{value}".encode()
                ).hexdigest(),
            )
            for guide_id in ordered_guides[:holdout_count]:
                validation_values.extend(guides[guide_id])
        elif len(ids) > 1:
            holdout_count = min(
                max(1, int(round(len(ids) * fraction))),
                len(ids) - 1,
            )
            ordered_ids = sorted(
                ids,
                key=lambda value: hashlib.sha256(
                    f"{seed}:measure:{embedding_id}:{value}".encode()
                ).hexdigest(),
            )
            validation_values.extend(ordered_ids[:holdout_count])
    validation_set = set(validation_values)
    validation = tuple(value for value in eligible if value in validation_set)
    if not validation:
        return ValidationSplit(
            train_measure_ids=data.measure_ids,
            validation_measure_ids=tuple(eligible),
            train_time_labels=downstream_labels,
            validation_time_labels=downstream_labels,
            source="train_self_eval",
            strategy="train_self_eval",
        )
    train = tuple(value for value in data.measure_ids if value not in validation_set)
    return ValidationSplit(
        train_measure_ids=train,
        validation_measure_ids=validation,
        train_time_labels=downstream_labels,
        validation_time_labels=downstream_labels,
        source="held_out",
        strategy="within_embedding_holdout",
    )


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
    expected = {
        "embedding_ids": list(data.embedding_ids),
        "control_embedding_ids": sorted(data.control_embedding_ids),
        "latent_dim": data.latent_dim,
        "embedding_dim": config.model.embedding_dim,
        "n_programs": config.model.n_programs,
        "hidden_dim": config.model.hidden_dim,
        "context_mode": config.model.context,
        "sigma_min": 1e-3,
        "growth_max": config.model.growth_max,
        "payoff_rank": min(4, config.model.n_programs),
    }
    return model.architecture() == expected


def _measure_meta_hash(data: TrajectoryData) -> str:
    canonical = data.measure_meta.loc[:, MEASURE_META_COLUMNS]
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_contract(data: TrajectoryData, config: RunConfig) -> dict[str, Any]:
    input_hashes = data.metadata.get("input_hashes", {})
    return {
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
        "model": config.model.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "validation": config.validation.model_dump(mode="json"),
        "loss": config.loss.model_dump(mode="json"),
    }


def _compact_capabilities():
    from .recipes.compact_v3 import recipe

    return recipe.capabilities


def _compact_recipe_contract() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("model.py", "particles.py", "objective.py", "training.py"):
        digest.update((root / name).read_bytes())
    digest.update((root / "recipes/compact_v3.py").read_bytes())
    return {
        "id": "credo.compact_sde_v3",
        "version": "3.0",
        "implementation_hash": digest.hexdigest(),
    }


def _split_contract(trainer: Trainer) -> dict[str, Any]:
    return {
        "strategy": trainer.validation_strategy,
        "train_measure_ids": list(trainer.train_measure_ids),
        "validation_measure_ids": list(trainer.validation_measure_ids),
        "train_time_labels": list(trainer.train_time_labels),
        "validation_time_labels": list(trainer.validation_time_labels),
        "representation_scope": "shared",
    }


def _tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _compact_checkpoint_envelope(trainer: Trainer) -> dict[str, Any]:
    from .artifacts import CheckpointEnvelope, CheckpointMode

    state = trainer.model.state_dict()
    return CheckpointEnvelope(
        recipe=_compact_recipe_contract(),
        study_contract=_checkpoint_contract(trainer.data, trainer.config),
        representation_contract=trainer.data.representation.to_dict(),
        split_contract=_split_contract(trainer),
        state={
            "model": {
                "source_key": "model_state",
                "tensor_count": len(state),
                "semantic_hash": _tensor_state_hash(state),
            },
            "ema": None,
            "representation": {"embedded": False},
            "objective": {"source_key": "log_count_concentration"},
            "optimizer": None,
            "scheduler": None,
            "rng": None,
        },
        training={
            "completed_epochs": trainer.completed_epochs,
            "training_recipe_available": True,
            "resume_supported": False,
            "exact_retraining": True,
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
    repository = Path(__file__).resolve().parents[2]
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
