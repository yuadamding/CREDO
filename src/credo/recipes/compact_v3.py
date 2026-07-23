"""The released compact weighted-SDE recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifacts import NativeCheckpointCodec
from ..contracts import (
    BatchingSpec,
    CapabilitySet,
    CREDOStudy,
    OptimizerSpec,
    RepresentationArtifact,
    SplitSpec,
    Stage,
    TrainingPlan,
)
from ..data.splits import SplitPlan, plan_compact_split
from ..model import CREDOModel
from ..runtime import ObjectiveDescriptor, RecipeRequirements
from .trajectory_compiler import compile_trajectory_view


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompactModelConfig(_StrictConfig):
    embedding_dim: int = Field(default=8, ge=1)
    n_programs: int = Field(default=8, ge=1)
    hidden_dim: int = Field(default=128, ge=8)
    context: Literal["none", "catalog_bank"] = "catalog_bank"
    growth_max: float = Field(default=3.0, gt=0)


class CompactEpochConfig(_StrictConfig):
    state: int = Field(default=40, ge=0)
    mass: int = Field(default=20, ge=0)
    context: int = Field(default=20, ge=0)

    @model_validator(mode="after")
    def _require_training(self) -> CompactEpochConfig:
        if self.state + self.mass + self.context < 1:
            raise ValueError("At least one training stage must have a positive epoch count.")
        return self


class CompactTrainingConfig(_StrictConfig):
    epochs: CompactEpochConfig = Field(default_factory=CompactEpochConfig)
    particles: int = Field(default=64, ge=2)
    steps_per_interval: int = Field(default=4, ge=1)
    measures_per_batch: int = Field(default=256, ge=1)
    batching: Literal["random", "target_round_robin", "target_blocked"] = "random"
    learning_rate: float = Field(default=1e-3, gt=0)
    patience: int = Field(default=10, ge=1)
    seed: int = Field(default=0, ge=0)


class CompactEvaluationConfig(_StrictConfig):
    particles: int = Field(default=256, ge=2)
    measures_per_batch: int = Field(default=256, ge=1)


class CompactValidationConfig(_StrictConfig):
    strategy: Literal["auto", "context_group", "checkpoint", "train_self_eval"] = "auto"
    values: tuple[str, ...] = ()
    fraction: float = Field(default=0.2, ge=0, lt=1)
    representation_scope: Literal["shared", "nested"] = "shared"

    @model_validator(mode="after")
    def _validate_specification(self) -> CompactValidationConfig:
        explicit = self.strategy in {"context_group", "checkpoint"}
        if explicit and not self.values:
            raise ValueError(f"validation.strategy={self.strategy!r} requires values.")
        if not explicit and self.values:
            raise ValueError(f"validation.strategy={self.strategy!r} does not accept values.")
        if self.strategy == "train_self_eval" and self.fraction != 0:
            raise ValueError("train_self_eval requires validation.fraction=0.")
        if explicit and self.fraction != 0:
            raise ValueError(f"Explicit {self.strategy} validation requires fraction=0.")
        if len(set(self.values)) != len(self.values):
            raise ValueError("validation.values must be unique.")
        return self


class CompactLossConfig(_StrictConfig):
    mass: float = Field(default=1.0, ge=0)
    count: float = Field(default=0.1, ge=0)
    sinkhorn_epsilon: float = Field(default=0.1, gt=0)
    action_drift: float = Field(default=1e-4, ge=0)
    action_diffusion: float = Field(default=1e-5, ge=0)
    action_growth: float = Field(default=1e-4, ge=0)
    parameter_regularization: float = Field(default=1e-4, ge=0)


class CompactV3Config(_StrictConfig):
    model: CompactModelConfig = Field(default_factory=CompactModelConfig)
    training: CompactTrainingConfig = Field(default_factory=CompactTrainingConfig)
    evaluation: CompactEvaluationConfig = Field(default_factory=CompactEvaluationConfig)
    validation: CompactValidationConfig = Field(default_factory=CompactValidationConfig)
    loss: CompactLossConfig = Field(default_factory=CompactLossConfig)


def _plain(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="python")
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    raise TypeError("Recipe config must be a mapping, Pydantic model, or dataclass.")


class CompactSDEV3Recipe:
    recipe_id = "credo.compact_sde_v3"
    recipe_version = "3.0"
    capabilities = CapabilitySet(
        physical_axis=True,
        effect_axis=True,
        endpoint=True,
        multitime=True,
        mass=True,
        counts=True,
        weak_form=False,
        context="catalog_bank",
        context_affects=("growth",),
        fresh_training_supported=True,
        checkpoint_inference_supported=True,
        checkpoint_resume_supported=False,
        same_study_holdout_evaluation=True,
        compatible_study_evaluation=False,
        cross_dataset_evaluation=False,
        deterministic_cpu_fresh_fit=True,
        bitwise_retraining_demonstrated=False,
        counterfactual_scope="catalog_background",
    )

    def config_schema(self) -> type[CompactV3Config]:
        return CompactV3Config

    def requirements(self, config: Any) -> RecipeRequirements:
        del config
        return RecipeRequirements(
            supported_axis_kinds=frozenset({"physical_time", "effect"}),
            supported_topologies=frozenset({"chain"}),
            supported_representation_kinds=frozenset({"latent"}),
            permitted_abundance_semantics=frozenset(
                {"absolute", "relative", "capture_count", "unit"}
            ),
            requires_effect_binding=True,
            requires_reference_binding=True,
            requires_source_geometry=True,
            permits_missing_target_geometry=True,
            supports_compositions=True,
            supports_replicates=True,
            abundance_requirement="optional",
            implicit_no_channel_semantics="unit",
            reference_mode="single_global_soft_reference",
            maximum_reference_pools=1,
            context_scope="series_static",
            sample_scope="series_static",
            composition_policies=frozenset(
                {"require_complete", "preserve_background", "condition_on_selection", "drop"}
            ),
            replicate_modes=frozenset({"reject", "select", "pool"}),
        )

    def plan_split(
        self,
        view: Any,
        config: Any,
        requested: SplitSpec | None = None,
    ) -> SplitPlan:
        return plan_compact_split(view, config, requested)

    def compile_study(
        self,
        view: Any,
        split: SplitPlan | SplitSpec,
        config: Any,
    ) -> CREDOStudy:
        del config
        return compile_trajectory_view(
            view,
            split_plan=split if isinstance(split, SplitPlan) else None,
        )

    def validate_run_config(self, run_config: Any) -> None:
        config = run_config.recipe_config
        reaction_epochs = config.training.epochs.mass + config.training.epochs.context
        if run_config.axis is not None and run_config.axis.kind == "effect":
            if (run_config.data is not None and run_config.data.counts is not None) or (
                config.loss.count > 0
            ):
                raise ValueError("Count likelihood cannot be configured for an effect axis.")
            if reaction_epochs > 0 or config.loss.mass > 0:
                raise ValueError("Growth and mass fitting cannot be configured for an effect axis.")
            if config.model.context != "none":
                raise ValueError("Effect-axis runs require model.context='none'.")
        if config.model.context == "none" and config.training.epochs.context > 0:
            raise ValueError("Context epochs must be zero when model.context is 'none'.")
        if config.training.epochs.context > 0 and config.training.epochs.mass == 0:
            raise ValueError("Context training requires a positive mass stage first.")
        if run_config.data is not None and config.loss.count > 0 and run_config.data.counts is None:
            raise ValueError("Positive count loss requires data.counts.")
        if reaction_epochs == 0 and (config.loss.mass > 0 or config.loss.count > 0):
            raise ValueError("Mass and count losses require a mass or context training stage.")
        if config.training.epochs.context > 0 and config.loss.mass == 0 and config.loss.count == 0:
            raise ValueError("Context training requires mass or count supervision.")
        if config.validation.strategy == "checkpoint" and run_config.axis is not None:
            unknown = set(config.validation.values) - set(run_config.axis.labels[1:])
            if unknown:
                raise ValueError(
                    f"Checkpoint validation contains unknown downstream labels: {sorted(unknown)}"
                )
            if set(config.validation.values) == set(run_config.axis.labels[1:]):
                raise ValueError(
                    "Checkpoint validation must leave a downstream checkpoint to train."
                )

    def build_representation(
        self,
        study_source: Any,
        split: SplitSpec,
        config: Mapping[str, Any],
    ) -> RepresentationArtifact:
        del split, config
        if isinstance(study_source, RepresentationArtifact):
            return study_source
        representation = getattr(study_source, "representation", None)
        if not isinstance(representation, RepresentationArtifact):
            raise TypeError("compact-v3 requires a frozen RepresentationArtifact.")
        return representation

    def build_model(self, study: CREDOStudy, config: Mapping[str, Any]) -> CREDOModel:
        raw = _plain(config)
        model = _plain(raw.get("model", raw))
        return CREDOModel(
            embedding_ids=study.embedding_ids,
            control_embedding_ids=study.control_embedding_ids,
            latent_dim=study.latent_dim,
            embedding_dim=int(model.get("embedding_dim", 8)),
            n_programs=int(model.get("n_programs", 8)),
            hidden_dim=int(model.get("hidden_dim", 128)),
            context_mode=str(model.get("context", "catalog_bank")),
            growth_max=float(model.get("growth_max", 3.0)),
        )

    def build_objectives(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> tuple[ObjectiveDescriptor, ...]:
        raw = _plain(config)
        loss = _plain(raw.get("loss", {}))
        objectives = [
            ObjectiveDescriptor(
                "checkpoint_geometry",
                1.0,
                frozenset({"geometry"}),
                {"sinkhorn_epsilon": float(loss.get("sinkhorn_epsilon", 0.1))},
            ),
            ObjectiveDescriptor(
                "checkpoint_mass",
                float(loss.get("mass", 1.0)),
                frozenset({"mass"}),
            ),
            ObjectiveDescriptor(
                "grouped_count_likelihood",
                float(loss.get("count", 0.1)),
                frozenset({"counts"}),
            ),
            ObjectiveDescriptor(
                "rollout_action",
                1.0,
                frozenset({"drift", "diffusion", "growth"}),
                {
                    "drift_weight": float(loss.get("action_drift", 1e-4)),
                    "diffusion_weight": float(loss.get("action_diffusion", 1e-5)),
                    "growth_weight": float(loss.get("action_growth", 1e-4)),
                },
            ),
            ObjectiveDescriptor(
                "model_regularization",
                float(loss.get("parameter_regularization", 1e-4)),
                frozenset({"parameters"}),
                {"coefficient": 1.0},
            ),
        ]
        if study.axis.kind == "effect":
            objectives = [
                term
                for term in objectives
                if term.name not in {"checkpoint_mass", "grouped_count_likelihood"}
            ]
        return tuple(objectives)

    def training_plan(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> TrainingPlan:
        del study
        raw = _plain(config)
        training = _plain(raw.get("training", {}))
        epochs = _plain(training.get("epochs", {}))
        batch_size = int(training.get("measures_per_batch", 256))
        optimizer = OptimizerSpec(
            kind="adam",
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=0.0,
        )
        batching = BatchingSpec(
            "measure_batches",
            measures_per_batch=batch_size,
            order=str(training.get("batching", "random")),
        )
        return TrainingPlan(
            seed=int(training.get("seed", 0)),
            particles=int(training.get("particles", 64)),
            steps_per_interval=int(training.get("steps_per_interval", 4)),
            early_stopping_patience=int(training.get("patience", 10)),
            gradient_clip_norm=10.0,
            stages=(
                Stage(
                    "state",
                    int(epochs.get("state", 40)),
                    (
                        "reference_embedding",
                        "residual_embeddings",
                        "shared_dynamics",
                        "drift",
                        "diffusion",
                    ),
                    "fp32",
                    optimizer,
                    ("checkpoint_geometry", "rollout_action", "model_regularization"),
                    batching,
                    "none",
                    "validation_geometry",
                ),
                Stage(
                    "mass",
                    int(epochs.get("mass", 20)),
                    (
                        "reference_embedding",
                        "residual_embeddings",
                        "shared_dynamics",
                        "drift",
                        "diffusion",
                        "growth",
                    ),
                    "fp32",
                    optimizer,
                    (
                        "checkpoint_geometry",
                        "checkpoint_mass",
                        "grouped_count_likelihood",
                        "rollout_action",
                        "model_regularization",
                    ),
                    batching,
                    "none",
                    "validation_total",
                ),
                Stage(
                    "context",
                    int(epochs.get("context", 20)),
                    ("program_encoder", "ecological_payoff"),
                    "fp32",
                    optimizer,
                    (
                        "checkpoint_geometry",
                        "checkpoint_mass",
                        "grouped_count_likelihood",
                        "rollout_action",
                        "model_regularization",
                    ),
                    batching,
                    "catalog_bank",
                    "validation_total",
                ),
            ),
        )

    def checkpoint_codec(self) -> NativeCheckpointCodec:
        return NativeCheckpointCodec()

    def load_checkpoint(
        self,
        checkpoint: Any,
        study: CREDOStudy,
        config: Any,
        **kwargs: Any,
    ):
        from ..training import Trainer

        return Trainer.load(checkpoint, study, config, **kwargs)

    def execute_training(
        self,
        study: CREDOStudy,
        *,
        model: CREDOModel,
        plan: TrainingPlan,
        objectives: tuple[ObjectiveDescriptor, ...],
        run_config: Any,
        **kwargs: Any,
    ):
        from ..training import Trainer

        return Trainer.from_plan(
            study,
            model,
            run_config,
            plan,
            objectives,
            **kwargs,
        )


recipe = CompactSDEV3Recipe()

__all__ = ["CompactSDEV3Recipe", "CompactV3Config", "recipe"]
