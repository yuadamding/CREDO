"""Immutable recipe declaration for archived transformer-SDE v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...artifacts import NativeCheckpointCodec
from ...contracts import (
    BatchingSpec,
    CapabilitySet,
    CREDOStudy,
    OptimizerSpec,
    RepresentationArtifact,
    SplitSpec,
    Stage,
    TrainingPlan,
)
from ...runtime import ObjectiveDescriptor, RecipeRequirements
from ..trajectory_compiler import compile_trajectory_view
from .model import FullDynamicsModel


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TransformerV2ModelConfig(_StrictConfig):
    embedding_dim: int = Field(default=48, ge=1)
    n_programs: int = Field(default=16, ge=1)
    mediator_dim: int = Field(default=48, ge=1)
    hidden_dim: int = Field(default=384, ge=1)
    depth: int = Field(default=4, ge=1)
    activation_checkpointing: bool = False
    time_frequencies: int = Field(default=4, ge=1)
    sigma_min: float = Field(default=1e-3, gt=0)
    r_max: float = Field(default=3.0, gt=0)
    n_payoff_ranks: int = Field(default=4, ge=1)
    ecological_growth: bool = True
    use_growth_intercept: bool = True
    shared_guide_embedding: bool = False
    control_mode: Literal["anchored", "soft_ref", "free"] = "soft_ref"
    control_ref_penalty: float = Field(default=5e-4, ge=0)
    context_kind: Literal["mlp", "transformer"] = "transformer"
    transformer_token_dim: int = Field(default=128, ge=1)
    transformer_heads: int = Field(default=4, ge=1)
    transformer_within_layers: int = Field(default=1, ge=1)
    transformer_cross_layers: int = Field(default=1, ge=1)
    transformer_inducing: int = Field(default=32, ge=1)
    transformer_dropout: float = Field(default=0.05, ge=0, lt=1)
    mass_attention_temperature: float = Field(default=0.75, gt=0)
    transformer_growth_only: bool = True

    @model_validator(mode="after")
    def _validate_attention_width(self) -> TransformerV2ModelConfig:
        if self.context_kind == "transformer" and (
            self.transformer_token_dim % self.transformer_heads
        ):
            raise ValueError("transformer_token_dim must be divisible by transformer_heads.")
        return self


class TransformerV2TrainingConfig(_StrictConfig):
    optimizer: Literal["adam", "adamw"] = "adamw"
    lambda_end: float = Field(default=1.0, ge=0)
    lambda_weak: float = Field(default=0.12, ge=0)
    lambda_reg_embed: float = Field(default=1e-4, ge=0)
    lambda_reg_growth_bias: float = Field(default=1e-4, ge=0)
    lambda_reg_net: float = Field(default=1e-4, ge=0)
    lambda_reg_diffusion: float = Field(default=2e-4, ge=0)
    lr_net: float = Field(default=3e-4, gt=0)
    lr_embed: float = Field(default=1e-3, gt=0)
    lr_transformer: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=1e-6, ge=0)
    transformer_weight_decay: float = Field(default=1e-4, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    epochs: int = Field(default=500, ge=1)
    early_stop_patience: int = Field(default=50, ge=1)
    seed: int = Field(default=0, ge=0)
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    sinkhorn_epsilon: float = Field(default=0.1, gt=0)
    sinkhorn_tau: float = Field(default=1.0, gt=0)
    sinkhorn_max_iter: int = Field(default=100, ge=1)
    n_test_functions: int = Field(default=12, ge=1)
    test_function_bandwidth: float = Field(default=1.0, gt=0)


class TransformerV2SimulationConfig(_StrictConfig):
    n_particles: int = Field(default=128, ge=2)
    n_steps: int = Field(default=24, ge=1)


class TransformerV2TrajectoryConfig(_StrictConfig):
    steps_per_interval: int = Field(default=24, ge=1)
    endpoint_time_weights: dict[str, float] = Field(default_factory=dict)
    normalize_time_weights: bool = True


class TransformerV2RecipeConfig(_StrictConfig):
    model: TransformerV2ModelConfig = Field(default_factory=TransformerV2ModelConfig)
    training: TransformerV2TrainingConfig = Field(default_factory=TransformerV2TrainingConfig)
    simulation: TransformerV2SimulationConfig = Field(default_factory=TransformerV2SimulationConfig)
    trajectory_training: TransformerV2TrajectoryConfig = Field(
        default_factory=TransformerV2TrajectoryConfig
    )
    perturbation_ids: tuple[str, ...] = ()
    control_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent_integration_grid(self) -> TransformerV2RecipeConfig:
        if self.simulation.n_steps != self.trajectory_training.steps_per_interval:
            raise ValueError(
                "simulation.n_steps must equal trajectory_training.steps_per_interval."
            )
        if len(self.perturbation_ids) != len(set(self.perturbation_ids)):
            raise ValueError("perturbation_ids must be unique.")
        if len(self.control_ids) != len(set(self.control_ids)):
            raise ValueError("control_ids must be unique.")
        if self.control_ids and not set(self.control_ids) <= set(self.perturbation_ids):
            raise ValueError("control_ids must be a subset of perturbation_ids.")
        return self


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("transformer-v2 config sections must be mappings.")


class TransformerSDEV2Recipe:
    recipe_id = "credo.transformer_sde_v2"
    recipe_version = "2.0"
    state_tensor_count = 146
    state_element_count = 5_634_421
    vae_tensor_count = 14
    vae_element_count = 3_165_736
    capabilities = CapabilitySet(
        physical_axis=True,
        effect_axis=False,
        endpoint=True,
        multitime=True,
        mass=True,
        counts=False,
        weak_form=True,
        context="full_population",
        context_affects=("drift", "diffusion", "growth"),
        fresh_training_supported=False,
        checkpoint_inference_supported=True,
        checkpoint_resume_supported=False,
        same_study_holdout_evaluation=True,
        compatible_study_evaluation=False,
        cross_dataset_evaluation=False,
        deterministic_cpu_fresh_fit=False,
        bitwise_retraining_demonstrated=False,
        counterfactual_scope="full_population",
    )

    def config_schema(self) -> type[TransformerV2RecipeConfig]:
        return TransformerV2RecipeConfig

    def requirements(self, config: Any) -> RecipeRequirements:
        del config
        return RecipeRequirements(
            supported_axis_kinds=frozenset({"physical_time"}),
            supported_topologies=frozenset({"chain"}),
            supported_representation_kinds=frozenset({"latent"}),
            permitted_abundance_semantics=frozenset(
                {"absolute", "relative", "capture_count", "unit"}
            ),
            requires_reference_binding=True,
            requires_source_geometry=True,
            permits_missing_target_geometry=True,
            supports_compositions=False,
            supports_replicates=False,
        )

    def compile_study(self, view: Any, split: SplitSpec, config: Any) -> CREDOStudy:
        del split, config
        return compile_trajectory_view(view)

    def build_representation(
        self,
        study_source: Any,
        split: SplitSpec,
        config: Mapping[str, Any],
    ) -> RepresentationArtifact:
        del split, config
        if isinstance(study_source, RepresentationArtifact):
            artifact = study_source
        else:
            artifact = getattr(study_source, "representation", None)
        if not isinstance(artifact, RepresentationArtifact):
            raise TypeError("transformer-v2 requires an ExpressionVAE RepresentationArtifact.")
        if artifact.backend != "expression_vae_v2":
            raise ValueError("transformer-v2 requires backend='expression_vae_v2'.")
        return artifact

    def build_model(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> FullDynamicsModel:
        raw = _mapping(config)
        model = _mapping(raw.get("model", raw))
        perturbation_ids = [
            str(value) for value in (raw.get("perturbation_ids") or study.embedding_ids)
        ]
        control_ids = [
            str(value) for value in (raw.get("control_ids") or study.control_embedding_ids)
        ]
        return self._construct_model(
            perturbation_ids,
            control_ids,
            study.latent_dim,
            model,
        )

    @staticmethod
    def _construct_model(
        perturbation_ids: Sequence[str],
        control_ids: Sequence[str],
        latent_dim: int,
        model: Mapping[str, Any],
    ) -> FullDynamicsModel:
        return FullDynamicsModel(
            perturbation_ids=list(perturbation_ids),
            control_ids=list(control_ids),
            latent_dim=int(latent_dim),
            embedding_dim=int(model.get("embedding_dim", 48)),
            n_programs=int(model.get("n_programs", 16)),
            mediator_dim=int(model.get("mediator_dim", 48)),
            hidden_dim=int(model.get("hidden_dim", 384)),
            depth=int(model.get("depth", 4)),
            activation_checkpointing=bool(model.get("activation_checkpointing", False)),
            n_time_freqs=int(model.get("time_frequencies", 4)),
            sigma_min=float(model.get("sigma_min", 1e-3)),
            r_max=float(model.get("r_max", 3.0)),
            n_payoff_ranks=int(model.get("n_payoff_ranks", 4)),
            ecological_growth=bool(model.get("ecological_growth", True)),
            use_growth_intercept=bool(model.get("use_growth_intercept", True)),
            shared_guide_embedding=bool(model.get("shared_guide_embedding", False)),
            control_mode=str(model.get("control_mode", "soft_ref")),
            control_ref_penalty=float(model.get("control_ref_penalty", 5e-4)),
            context_kind=str(model.get("context_kind", "transformer")),
            transformer_token_dim=int(model.get("transformer_token_dim", 128)),
            transformer_heads=int(model.get("transformer_heads", 4)),
            transformer_within_layers=int(model.get("transformer_within_layers", 1)),
            transformer_cross_layers=int(model.get("transformer_cross_layers", 1)),
            transformer_inducing=int(model.get("transformer_inducing", 32)),
            transformer_dropout=float(model.get("transformer_dropout", 0.05)),
            mass_attention_temperature=float(model.get("mass_attention_temperature", 0.75)),
            transformer_growth_only=bool(model.get("transformer_growth_only", True)),
        )

    def build_objectives(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> tuple[ObjectiveDescriptor, ...]:
        study.axis.require_physical("transformer-v2 training")
        raw = _mapping(config)
        model = _mapping(raw.get("model", {}))
        training = _mapping(raw.get("training", {}))
        trajectory = _mapping(raw.get("trajectory_training", {}))
        return (
            ObjectiveDescriptor(
                "checkpoint_geometry_mass",
                float(training.get("lambda_end", 1.0)),
                frozenset({"geometry", "mass"}),
                {
                    "time_weights": trajectory.get("endpoint_time_weights", {}),
                    "normalize_time_weights": bool(trajectory.get("normalize_time_weights", True)),
                    "sinkhorn_epsilon": float(training.get("sinkhorn_epsilon", 0.1)),
                    "sinkhorn_tau": float(training.get("sinkhorn_tau", 1.0)),
                    "sinkhorn_max_iter": int(training.get("sinkhorn_max_iter", 100)),
                },
            ),
            ObjectiveDescriptor(
                "weak_form_residual",
                float(training.get("lambda_weak", 0.12)),
                frozenset({"drift", "diffusion", "growth"}),
                {
                    "n_test_functions": int(training.get("n_test_functions", 12)),
                    "bandwidth": float(training.get("test_function_bandwidth", 1.0)),
                },
            ),
            ObjectiveDescriptor(
                "rollout_regularization",
                1.0,
                frozenset({"drift", "diffusion", "growth"}),
                {
                    "drift_weight": float(training.get("lambda_reg_net", 1e-4)),
                    "diffusion_weight": float(training.get("lambda_reg_diffusion", 2e-4)),
                    "growth_weight": float(training.get("lambda_reg_net", 1e-4)),
                },
            ),
            ObjectiveDescriptor(
                "embedding_regularization",
                1.0,
                frozenset({"perturbation_embeddings"}),
                {
                    "residual_weight": float(training.get("lambda_reg_embed", 1e-4)),
                    "control_reference_weight": float(model.get("control_ref_penalty", 5e-4)),
                },
            ),
            ObjectiveDescriptor(
                "growth_intercept_regularization",
                float(training.get("lambda_reg_growth_bias", 1e-4)),
                frozenset({"growth_intercepts"}),
            ),
            ObjectiveDescriptor(
                "ecological_payoff_regularization",
                1.0 if bool(model.get("ecological_growth", True)) else 0.0,
                frozenset({"ecological_payoff"}),
            ),
        )

    def training_plan(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> TrainingPlan:
        study.axis.require_physical("transformer-v2 training")
        raw = _mapping(config)
        training = _mapping(raw.get("training", {}))
        simulation = _mapping(raw.get("simulation", {}))
        trajectory = _mapping(raw.get("trajectory_training", {}))
        parameter_rates = {
            "dynamics": float(training.get("lr_net", 3e-4)),
            "embeddings": float(training.get("lr_embed", 1e-3)),
            "transformer": float(training.get("lr_transformer", 5e-5)),
        }
        optimizer = OptimizerSpec(
            kind=str(training.get("optimizer", "adamw")),
            learning_rate=parameter_rates["dynamics"],
            weight_decay=float(training.get("weight_decay", 1e-6)),
            parameter_learning_rates=parameter_rates,
            parameter_weight_decays={
                "dynamics": float(training.get("weight_decay", 1e-6)),
                "embeddings": float(training.get("weight_decay", 1e-6)),
                "transformer": float(training.get("transformer_weight_decay", 1e-4)),
            },
        )
        return TrainingPlan(
            seed=int(training.get("seed", 0)),
            particles=int(simulation.get("n_particles", 128)),
            steps_per_interval=int(
                trajectory.get("steps_per_interval", simulation.get("n_steps", 24))
            ),
            early_stopping_patience=int(training.get("early_stop_patience", 50)),
            gradient_clip_norm=float(training.get("grad_clip", 1.0)),
            stages=(
                Stage(
                    "legacy_joint",
                    int(training.get("epochs", 500)),
                    ("dynamics", "perturbation_embeddings", "transformer_context"),
                    str(training.get("precision", "bf16")),
                    optimizer,
                    (
                        "checkpoint_geometry_mass",
                        "weak_form_residual",
                        "rollout_regularization",
                        "embedding_regularization",
                        "growth_intercept_regularization",
                        "ecological_payoff_regularization",
                    ),
                    BatchingSpec("all_keys"),
                    "full_population",
                    "validation_endpoint_loss",
                ),
            ),
        )

    def checkpoint_codec(self) -> NativeCheckpointCodec:
        return NativeCheckpointCodec()

    def load_model_state(
        self,
        model: FullDynamicsModel,
        raw_state: Mapping[str, torch.Tensor],
        ema_state: Mapping[str, torch.Tensor] | None = None,
        *,
        state: Literal["raw", "ema"] = "raw",
    ) -> None:
        selected = {name: value for name, value in raw_state.items()}
        if state == "ema":
            if not ema_state:
                raise ValueError("The legacy checkpoint contains no embedded EMA state.")
            unknown = set(ema_state) - set(selected)
            if unknown:
                raise ValueError(f"EMA state contains unknown tensors: {sorted(unknown)[:5]}")
            selected.update(ema_state)
        elif state != "raw":
            raise ValueError("state must be 'raw' or 'ema'.")
        model.load_state_dict(selected, strict=True)
        model.assert_soft_reference()


recipe = TransformerSDEV2Recipe()

__all__ = ["TransformerSDEV2Recipe", "TransformerV2RecipeConfig", "recipe"]
