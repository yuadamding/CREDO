"""Immutable recipe declaration for archived transformer-SDE v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

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
from ...runtime import ObjectiveDescriptor
from .model import FullDynamicsModel


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
        context_affects=("growth",),
        external_evaluation=True,
        resume_training=False,
        focal_counterfactual=True,
        full_group_counterfactual=True,
        exact_retraining=False,
    )

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
            str(value) for value in raw.get("perturbation_ids", study.embedding_ids)
        ]
        control_ids = [str(value) for value in raw.get("control_ids", study.control_embedding_ids)]
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
                },
            ),
            ObjectiveDescriptor(
                "weak_form_residual",
                float(training.get("lambda_weak", 0.12)),
                frozenset({"drift", "diffusion", "growth"}),
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
        parameter_rates = {
            "dynamics": float(training.get("lr_net", 3e-4)),
            "embeddings": float(training.get("lr_embed", 1e-3)),
            "transformer": float(training.get("lr_transformer", 5e-5)),
        }
        optimizer = OptimizerSpec(
            "adamw",
            parameter_rates["dynamics"],
            float(training.get("weight_decay", 1e-6)),
            parameter_rates,
        )
        return TrainingPlan(
            stages=(
                Stage(
                    "legacy_joint",
                    int(training.get("epochs", 500)),
                    ("dynamics", "perturbation_embeddings", "transformer_context"),
                    str(training.get("precision", "bf16")),
                    optimizer,
                    ("checkpoint_geometry_mass", "weak_form_residual"),
                    BatchingSpec("all_keys"),
                    "full_population",
                    "validation_endpoint_loss",
                ),
            )
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

__all__ = ["TransformerSDEV2Recipe", "recipe"]
