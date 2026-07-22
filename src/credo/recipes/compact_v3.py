"""The released compact weighted-SDE recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

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
from ..model import CREDOModel
from ..runtime import ObjectiveDescriptor


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
        external_evaluation=True,
        resume_training=False,
        focal_counterfactual=True,
        full_group_counterfactual=True,
        exact_retraining=True,
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
            ObjectiveDescriptor("checkpoint_geometry", 1.0, frozenset({"geometry"})),
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
            ObjectiveDescriptor("action", 1e-4, frozenset({"drift", "diffusion", "growth"})),
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
        batch_size = int(training.get("measures_per_batch", 32))
        optimizer = OptimizerSpec(
            kind="adamw",
            learning_rate=float(training.get("learning_rate", 3e-4)),
            weight_decay=float(training.get("weight_decay", 1e-6)),
        )
        batching = BatchingSpec("measure_batches", measures_per_batch=batch_size)
        return TrainingPlan(
            stages=(
                Stage(
                    "state",
                    int(epochs.get("state", 40)),
                    ("reference", "residuals", "drift", "diffusion"),
                    "fp32",
                    optimizer,
                    ("checkpoint_geometry", "action"),
                    batching,
                    "none",
                    "validation_geometry",
                ),
                Stage(
                    "mass",
                    int(epochs.get("mass", 20)),
                    ("growth", "growth_intercepts"),
                    "fp32",
                    optimizer,
                    (
                        "checkpoint_geometry",
                        "checkpoint_mass",
                        "grouped_count_likelihood",
                        "action",
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
                        "action",
                    ),
                    batching,
                    "catalog_bank",
                    "validation_total",
                ),
            )
        )

    def checkpoint_codec(self) -> NativeCheckpointCodec:
        return NativeCheckpointCodec()

    def fit(self, study: CREDOStudy, config: Any, **kwargs: Any):
        from ..training import Trainer

        return Trainer.fit(study, None, config, **kwargs)


recipe = CompactSDEV3Recipe()

__all__ = ["CompactSDEV3Recipe", "recipe"]
