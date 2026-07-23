"""Stable recipe-neutral execution protocols for CREDO."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch

from .contracts import (
    CapabilitySet,
    CREDOStudy,
    RepresentationArtifact,
    SplitSpec,
    TrainingPlan,
)
from .data.splits import SplitPlan, validate_representation_scope, validate_split_plan
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
    requires_effect_binding: bool
    requires_reference_binding: bool
    requires_source_geometry: bool
    permits_missing_target_geometry: bool
    supports_compositions: bool
    supports_replicates: bool
    abundance_requirement: str = "required"
    implicit_no_channel_semantics: str = "none"
    reference_mode: str = "unspecified"
    maximum_reference_pools: int | None = None
    context_scope: str = "observation_varying"
    sample_scope: str = "observation_varying"
    composition_policies: frozenset[str] = frozenset(
        {"require_complete", "preserve_background", "condition_on_selection", "drop"}
    )
    replicate_modes: frozenset[str] = frozenset({"reject"})

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
        if self.abundance_requirement not in {"required", "optional", "forbidden"}:
            raise ValueError("Unknown abundance requirement.")
        if self.implicit_no_channel_semantics not in {"unit", "none"}:
            raise ValueError("Unknown implicit no-channel abundance semantics.")
        if self.maximum_reference_pools is not None and self.maximum_reference_pools < 1:
            raise ValueError("maximum_reference_pools must be positive when provided.")
        if self.context_scope not in {"none", "series_static", "observation_varying"}:
            raise ValueError("Unknown context scope.")
        if self.sample_scope not in {"series_static", "observation_varying"}:
            raise ValueError("Unknown sample scope.")
        composition_policies = frozenset(str(value) for value in self.composition_policies)
        replicate_modes = frozenset(str(value) for value in self.replicate_modes)
        unknown_compositions = composition_policies - {
            "require_complete",
            "preserve_background",
            "condition_on_selection",
            "drop",
        }
        unknown_replicates = replicate_modes - {
            "reject",
            "keep_separate",
            "pool",
            "select",
            "hierarchical",
        }
        if unknown_compositions or unknown_replicates:
            raise ValueError("RecipeRequirements contains unknown selection policies.")
        object.__setattr__(self, "composition_policies", composition_policies)
        object.__setattr__(self, "replicate_modes", replicate_modes)


@runtime_checkable
class CREDORecipe(Protocol):
    recipe_id: str
    recipe_version: str
    capabilities: CapabilitySet

    def config_schema(self) -> type[Any]: ...

    def requirements(self, config: Any) -> RecipeRequirements: ...

    def plan_split(
        self,
        view: StudyView,
        config: Any,
        requested: SplitSpec | None = None,
    ) -> SplitPlan: ...

    def compile_study(
        self,
        view: StudyView,
        split: SplitSpec | SplitPlan,
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

    def load_checkpoint(
        self,
        checkpoint: Any,
        study: Any,
        config: Any,
        **kwargs: Any,
    ) -> Any: ...


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
    split: SplitSpec | SplitPlan,
    requirements: RecipeRequirements,
) -> ValidationReport:
    """Validate a semantic view before any recipe-owned tensorization."""
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
        abundance_semantics = requirements.implicit_no_channel_semantics
        if requirements.abundance_requirement == "required":
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.abundance_required",
                    "Recipe requires an explicitly selected abundance channel.",
                    ("abundance",),
                )
            )
    else:
        abundance_semantics = view.study.abundance.channels[view.abundance_channel].semantics.value
        if requirements.abundance_requirement == "forbidden":
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.abundance_forbidden",
                    "Recipe forbids an abundance channel for this run.",
                    ("abundance", view.abundance_channel),
                )
            )
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
    replicate_policy = view.selection.replicate_policy.mode
    if replicate_policy not in requirements.replicate_modes:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.replicate_policy",
                f"Recipe does not support replicate policy {replicate_policy!r}.",
                ("selection", "replicate_policy"),
            )
        )
    if observations.duplicated(["series_id", "checkpoint_id"]).any() and (
        replicate_policy == "reject" or not requirements.supports_replicates
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
    if requirements.context_scope == "series_static" and "context_id" in observations:
        varying = observations.groupby("series_id", observed=True)["context_id"].nunique(
            dropna=False
        )
        if varying.gt(1).any():
            series_id = str(varying[varying.gt(1)].index[0])
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.context_scope",
                    f"Recipe requires series-static context; series {series_id!r} changes context.",
                    ("observations", series_id, "context_id"),
                )
            )
    if requirements.sample_scope == "series_static":
        varying = observations.groupby("series_id", observed=True)["sample_id"].nunique(
            dropna=False
        )
        if varying.gt(1).any():
            series_id = str(varying[varying.gt(1)].index[0])
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.sample_scope",
                    "Recipe requires series-static sample identity; "
                    f"series {series_id!r} changes sample.",
                    ("observations", series_id, "sample_id"),
                )
            )
    if requirements.requires_effect_binding and view.effect_binding().empty:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.effect_binding",
                "Recipe requires a selected effect binding catalog.",
                ("effect_bindings", view.selection.effect_binding_id or "none"),
            )
        )
    if requirements.requires_reference_binding:
        selected_series = view.study.series._unsafe_view()
        selected_series = selected_series.loc[selected_series["series_id"].isin(view.series_ids)]
        selected_conditions = view.study.conditions._unsafe_view()
        selected_conditions = selected_conditions.loc[
            selected_conditions["condition_id"].isin(selected_series["condition_id"])
        ]
        selected_binding = view.reference_binding()
        if selected_binding.empty:
            pool_by_condition = pd.Series(dtype=str)
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.reference_binding",
                    "Recipe requires a selected reference binding catalog.",
                    ("reference_bindings", view.selection.reference_binding_id or "none"),
                )
            )
        else:
            pool_by_condition = selected_binding.set_index("condition_id")[
                "reference_pool_id"
            ].astype(str)
            missing_binding = set(selected_conditions["condition_id"]) - set(
                pool_by_condition.index
            )
            if missing_binding:
                issues.append(
                    ValidationIssue(
                        "error",
                        "recipe.reference_binding",
                        f"Selected reference binding omits conditions: "
                        f"{sorted(missing_binding)[:5]}.",
                        ("reference_bindings", view.selection.reference_binding_id or "none"),
                    )
                )
        reference_condition_ids = set(
            selected_conditions.loc[selected_conditions["is_reference"], "condition_id"]
        )
        intervention_condition_ids = set(
            selected_conditions.loc[~selected_conditions["is_reference"], "condition_id"]
        )
        reference_groups = set(
            pool_by_condition.loc[list(reference_condition_ids & set(pool_by_condition.index))]
        )
        intervention_groups = set(
            pool_by_condition.loc[list(intervention_condition_ids & set(pool_by_condition.index))]
        )
        unresolved = intervention_groups - reference_groups
        if unresolved:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.reference_binding",
                    f"Selected interventions lack resolvable reference pools: "
                    f"{sorted(unresolved)[:5]}.",
                    ("reference_bindings", view.selection.reference_binding_id or "none"),
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
        if (
            requirements.maximum_reference_pools is not None
            and len(reference_groups) > requirements.maximum_reference_pools
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.reference_multiplicity",
                    f"Recipe reference mode {requirements.reference_mode!r} permits at most "
                    f"{requirements.maximum_reference_pools} pool(s); "
                    f"selected={sorted(reference_groups)}.",
                    ("reference_bindings", view.selection.reference_binding_id or "none"),
                )
            )
    if view.selection.composition_policy not in requirements.composition_policies:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.composition_policy",
                f"Recipe does not support composition policy "
                f"{view.selection.composition_policy!r}.",
                ("selection", "composition_policy"),
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
        recipe.capabilities.require("train")
        if isinstance(study, Study):
            study = config.view(study) if callable(getattr(config, "view", None)) else study.view()
        if isinstance(study, StudyView):
            split = recipe.plan_split(study, recipe_config, split)
            validate_split_plan(study, split)
            validate_representation_scope(study, split)
            requirements = recipe.requirements(recipe_config)
            validate_view_for_recipe(study, split, requirements).raise_for_errors()
            compiled_study = recipe.compile_study(study, split, recipe_config)
        else:
            compiled_study = study
        validate_recipe_study(recipe, compiled_study)
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


def train(
    study: Study | StudyView | CREDOStudy,
    config: Any,
    *,
    recipe: CREDORecipe | None = None,
    split: SplitSpec | None = None,
    **kwargs: Any,
) -> Any:
    """Fit one configured recipe through the semantic training boundary."""
    if recipe is None:
        from .registry import get_recipe

        recipe = get_recipe(config.recipe)
    selected = config.view(study) if isinstance(study, Study) else study
    return TrainingEngine().fit(recipe, selected, config, split=split, **kwargs)


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
    "train",
    "validate_recipe_study",
    "validate_training_contract",
    "validate_view_for_recipe",
]
