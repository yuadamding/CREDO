"""Stable recipe-neutral execution protocols for CREDO."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .contracts import (
    CapabilitySet,
    CREDOStudy,
    RepresentationArtifact,
    SplitSpec,
    TrainingPlan,
)
from .data.study import Study, StudyView
from .data.validation import ValidationIssue, ValidationReport


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "requires", frozenset(str(value) for value in self.requires))
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        if not self.name or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Objective names must be nonempty and weights nonnegative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "requires": sorted(self.requires),
            "config": dict(self.config),
        }


@runtime_checkable
class CheckpointCodec(Protocol):
    def encode(self, **parts: Any) -> Mapping[str, Any]: ...

    def decode(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RecipeRequirements:
    """Semantic study capabilities required before recipe compilation."""

    supported_axis_kinds: frozenset[str]
    supported_topologies: frozenset[str]
    supported_representation_kinds: frozenset[str]
    permitted_abundance_semantics: frozenset[str]
    requires_reference_binding: bool
    requires_source_geometry: bool
    permits_missing_target_geometry: bool
    supports_compositions: bool
    supports_replicates: bool

    def __post_init__(self) -> None:
        for name in (
            "supported_axis_kinds",
            "supported_topologies",
            "supported_representation_kinds",
            "permitted_abundance_semantics",
        ):
            values = frozenset(str(value) for value in getattr(self, name))
            if not values:
                raise ValueError(f"RecipeRequirements.{name} must be nonempty.")
            object.__setattr__(self, name, values)


@runtime_checkable
class CREDORecipe(Protocol):
    recipe_id: str
    recipe_version: str
    capabilities: CapabilitySet

    def config_schema(self) -> type[Any]: ...

    def requirements(self, config: Any) -> RecipeRequirements: ...

    def compile_study(
        self,
        view: StudyView,
        split: SplitSpec,
        config: Any,
    ) -> Any: ...

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


def validate_view_for_recipe(
    view: StudyView,
    split: SplitSpec,
    requirements: RecipeRequirements,
) -> ValidationReport:
    """Validate a semantic view before any recipe-owned tensorization."""
    del split
    issues: list[ValidationIssue] = []
    design = view.study.design
    axis_kinds = {axis.kind for axis in design.axes}
    unsupported_axes = axis_kinds - requirements.supported_axis_kinds
    if unsupported_axes:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.axis_kind",
                f"Recipe does not support axis kinds {sorted(unsupported_axes)}.",
                ("design", "axes"),
            )
        )
    if design.topology not in requirements.supported_topologies:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.topology",
                f"Recipe does not support topology {design.topology!r}.",
                ("design", "topology"),
            )
        )
    if view.representation.space_kind not in requirements.supported_representation_kinds:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.representation_kind",
                f"Recipe does not support representation kind {view.representation.space_kind!r}.",
                ("representations", view.representation_id, "space_kind"),
            )
        )
    if view.abundance_channel is None or view.study.abundance is None:
        abundance_semantics = "none"
    else:
        abundance_semantics = view.study.abundance.channels[view.abundance_channel].semantics.value
    if abundance_semantics not in requirements.permitted_abundance_semantics:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.abundance_semantics",
                f"Recipe does not support abundance semantics {abundance_semantics!r}.",
                ("abundance", view.abundance_channel or "none"),
            )
        )

    observations = view.observations()
    support_index = view.study.support_index._unsafe_view()
    support_index = support_index.loc[
        support_index["representation_id"].eq(view.representation_id)
        & support_index["observation_id"].isin(observations["observation_id"])
    ]
    available = set(support_index.loc[support_index["available"], "observation_id"])
    source = observations.loc[observations["checkpoint_id"].eq(design.source_checkpoint_id)]
    if requirements.requires_source_geometry:
        missing_source_series = set(view.series_ids) - set(source["series_id"])
        missing_source = set(source["observation_id"]) - available
        if missing_source_series or missing_source:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.source_geometry",
                    "Recipe requires one source geometry per selected series; "
                    f"missing_series={sorted(missing_source_series)[:5]}, "
                    f"missing_support={sorted(missing_source)[:5]}.",
                    ("support_index", view.representation_id),
                )
            )
    if not requirements.permits_missing_target_geometry:
        targets = observations.loc[~observations["checkpoint_id"].eq(design.source_checkpoint_id)]
        missing_targets = set(targets["observation_id"]) - available
        if missing_targets:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.target_geometry",
                    f"Recipe requires target geometry; missing={sorted(missing_targets)[:5]}.",
                    ("support_index", view.representation_id),
                )
            )
    if (
        not requirements.supports_replicates
        and observations.duplicated(["series_id", "checkpoint_id"]).any()
    ):
        duplicate = observations.loc[
            observations.duplicated(["series_id", "checkpoint_id"]),
            ["series_id", "checkpoint_id"],
        ].iloc[0]
        issues.append(
            ValidationIssue(
                "error",
                "recipe.replicates",
                "Recipe requires replicates to be selected or pooled before compilation; "
                f"duplicate={tuple(duplicate)!r}.",
                ("observations",),
            )
        )
    if requirements.requires_reference_binding:
        selected_series = view.study.series._unsafe_view()
        selected_series = selected_series.loc[selected_series["series_id"].isin(view.series_ids)]
        selected_conditions = view.study.conditions._unsafe_view()
        selected_conditions = selected_conditions.loc[
            selected_conditions["condition_id"].isin(selected_series["condition_id"])
        ]
        reference_groups = set(
            selected_conditions.loc[
                selected_conditions["is_reference"], "reference_group_id"
            ].astype(str)
        )
        unresolved = (
            set(
                selected_conditions.loc[
                    ~selected_conditions["is_reference"], "reference_group_id"
                ].astype(str)
            )
            - reference_groups
        )
        if unresolved:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.reference_binding",
                    f"Selected interventions lack resolvable reference pools: "
                    f"{sorted(unresolved)[:5]}.",
                    ("conditions", "reference_group_id"),
                )
            )
        if not reference_groups:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.reference_binding",
                    "Recipe requires at least one selected reference condition.",
                    ("conditions", "is_reference"),
                )
            )
    try:
        compositions = view.compositions()
    except ValueError as exc:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.composition_closure",
                str(exc),
                ("selection", "composition_policy"),
            )
        )
    else:
        if len(compositions) and not requirements.supports_compositions:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.compositions",
                    "Recipe does not support compositional observations.",
                    ("compositions",),
                )
            )
    return ValidationReport(tuple(issues))


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
    for objective in objectives:
        if not objective.name or not math.isfinite(objective.weight) or objective.weight < 0:
            raise ValueError("Recipe objective names must be nonempty and weights nonnegative.")
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
        study: Study | StudyView | CREDOStudy,
        config: Any,
        split: SplitSpec | None = None,
        **kwargs: Any,
    ) -> Any:
        recipe_config = (
            config.recipe_configuration()
            if callable(getattr(config, "recipe_configuration", None))
            else config
        )
        if isinstance(study, Study):
            study = study.view()
        if isinstance(study, StudyView):
            if split is None:
                representation_scope = getattr(
                    getattr(recipe_config, "validation", None),
                    "representation_scope",
                    "shared",
                )
                split = SplitSpec(
                    strategy="none",
                    representation_scope=representation_scope,
                    split_id=f"study-view:{study.semantic_hash()[:12]}",
                )
            requirements = recipe.requirements(recipe_config)
            validate_view_for_recipe(study, split, requirements).raise_for_errors()
            compiled_study = recipe.compile_study(study, split, recipe_config)
        else:
            compiled_study = study
        validate_recipe_study(recipe, compiled_study)
        recipe.capabilities.require("train")
        plan = recipe.training_plan(compiled_study, recipe_config)
        objectives = recipe.build_objectives(compiled_study, recipe_config)
        validate_training_contract(recipe, objectives, plan)
        executor = getattr(recipe, "execute_training", None)
        if not callable(executor):
            raise RuntimeError(
                f"Recipe {recipe.recipe_id}@{recipe.recipe_version} publishes a training "
                "plan but has no released training executor."
            )
        # Model initialization and recipe-owned sampling are part of the plan.
        random.seed(plan.seed)
        np.random.seed(plan.seed)
        torch.manual_seed(plan.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(plan.seed)
        model = recipe.build_model(compiled_study, recipe_config)
        result = executor(
            compiled_study,
            model=model,
            plan=plan,
            objectives=objectives,
            run_config=config,
            **kwargs,
        )
        if getattr(result, "training_plan", None) != plan:
            raise RuntimeError("Training executor did not retain the recipe's immutable plan.")
        if getattr(result, "objective_descriptors", None) != objectives:
            raise RuntimeError(
                "Training executor did not retain the recipe's objective declarations."
            )
        return result


__all__ = [
    "CREDORecipe",
    "CREDORun",
    "CheckpointCodec",
    "LossReport",
    "ModelRecipe",
    "ObjectiveDescriptor",
    "ObjectiveTerm",
    "RecipeRequirements",
    "RuntimeState",
    "TrainingEngine",
    "validate_recipe_study",
    "validate_training_contract",
    "validate_view_for_recipe",
]
