"""Stable recipe-neutral execution protocols for CREDO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .contracts import (
    CapabilitySet,
    CREDOStudy,
    RepresentationArtifact,
    SplitSpec,
    TrainingPlan,
)


@dataclass(frozen=True)
class LossReport:
    name: str
    value: Any
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeState:
    stage: str
    epoch: int = 0
    values: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObjectiveTerm(Protocol):
    name: str
    requires: frozenset[str]

    def compute(
        self,
        rollout: Any,
        study: CREDOStudy,
        runtime_state: RuntimeState,
    ) -> LossReport: ...


@dataclass(frozen=True)
class ObjectiveDescriptor:
    """Serializable objective declaration used by immutable recipes."""

    name: str
    weight: float
    requires: frozenset[str] = frozenset()
    config: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class CheckpointCodec(Protocol):
    def encode(self, **parts: Any) -> Mapping[str, Any]: ...

    def decode(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class CREDORecipe(Protocol):
    recipe_id: str
    recipe_version: str
    capabilities: CapabilitySet

    def build_representation(
        self,
        study_source: Any,
        split: SplitSpec,
        config: Mapping[str, Any],
    ) -> RepresentationArtifact: ...

    def build_model(self, study: CREDOStudy, config: Mapping[str, Any]) -> Any: ...

    def build_objectives(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> tuple[ObjectiveDescriptor, ...]: ...

    def training_plan(
        self,
        study: CREDOStudy,
        config: Mapping[str, Any],
    ) -> TrainingPlan: ...

    def checkpoint_codec(self) -> CheckpointCodec: ...


ModelRecipe = CREDORecipe


@dataclass
class CREDORun:
    """Recipe-neutral handle consumed by evaluators and counterfactual engines."""

    recipe: CREDORecipe
    study: CREDOStudy
    model: Any
    split: SplitSpec
    representation: RepresentationArtifact
    checkpoint: Mapping[str, Any]
    runtime: Any = None

    @property
    def recipe_id(self) -> str:
        return self.recipe.recipe_id

    @property
    def recipe_version(self) -> str:
        return self.recipe.recipe_version

    @property
    def capabilities(self) -> CapabilitySet:
        return self.recipe.capabilities

    def require(self, operation: str) -> None:
        self.capabilities.require(operation)


def validate_recipe_study(recipe: CREDORecipe, study: CREDOStudy) -> None:
    """Reject a study whose scientific semantics exceed recipe capabilities."""
    axis_capability = (
        recipe.capabilities.physical_axis
        if study.axis.kind == "physical"
        else recipe.capabilities.effect_axis
    )
    if not axis_capability:
        raise ValueError(
            f"Recipe {recipe.recipe_id}@{recipe.recipe_version} does not support "
            f"a {study.axis.kind!r} axis."
        )
    if len(study.axis.labels) == 2 and not recipe.capabilities.endpoint:
        raise ValueError(
            f"Recipe {recipe.recipe_id}@{recipe.recipe_version} does not support endpoints."
        )
    if len(study.axis.labels) > 2 and not recipe.capabilities.multitime:
        raise ValueError(
            f"Recipe {recipe.recipe_id}@{recipe.recipe_version} does not support multitime data."
        )
    if study.count_blocks and not recipe.capabilities.counts:
        raise ValueError(
            f"Recipe {recipe.recipe_id}@{recipe.recipe_version} does not support counts."
        )


def validate_training_contract(
    recipe: CREDORecipe,
    objectives: tuple[ObjectiveDescriptor, ...],
    plan: TrainingPlan,
) -> None:
    """Ensure a released plan uses only its recipe's declared combination."""
    objective_names = tuple(term.name for term in objectives)
    if len(objective_names) != len(set(objective_names)):
        raise ValueError("Recipe objective names must be unique.")
    available = set(objective_names)
    for stage in plan.stages:
        if stage.epochs == 0:
            continue
        unknown = set(stage.active_objectives) - available
        if unknown:
            raise ValueError(
                f"Training stage {stage.name!r} references unknown objectives: {sorted(unknown)}"
            )
        if stage.context_policy != "none" and stage.context_policy != recipe.capabilities.context:
            raise ValueError(
                f"Training stage {stage.name!r} requests context {stage.context_policy!r}, "
                f"but the recipe declares {recipe.capabilities.context!r}."
            )
        if stage.context_policy == "full_population" and stage.batching.mode != "all_keys":
            raise ValueError("Full-population context requires all_keys batching.")


class TrainingEngine:
    """Recipe-neutral entry point for released training executors."""

    def fit(
        self,
        recipe: CREDORecipe,
        study: CREDOStudy,
        config: Any,
        **kwargs: Any,
    ) -> Any:
        validate_recipe_study(recipe, study)
        plan = recipe.training_plan(study, config)
        objectives = recipe.build_objectives(study, config)
        validate_training_contract(recipe, objectives, plan)
        executor = getattr(recipe, "fit", None)
        if not callable(executor):
            raise RuntimeError(
                f"Recipe {recipe.recipe_id}@{recipe.recipe_version} publishes a training "
                "plan but has no released training executor."
            )
        return executor(study, config, **kwargs)


__all__ = [
    "CREDORecipe",
    "CREDORun",
    "CheckpointCodec",
    "LossReport",
    "ModelRecipe",
    "ObjectiveDescriptor",
    "ObjectiveTerm",
    "RuntimeState",
    "TrainingEngine",
    "validate_recipe_study",
    "validate_training_contract",
]
