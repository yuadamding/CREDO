"""Stable recipe-neutral execution protocols for CREDO."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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
from .lps import PerturbSeqStudy, PerturbSeqView
from .problems import CompiledLPSProblem


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


def _stable_ids(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique nonempty IDs.")
    return normalized


@dataclass(frozen=True)
class PredictionQuery:
    """Stable-ID query for fitted longitudinal Perturb-seq predictions."""

    series_ids: tuple[str, ...] = ()
    checkpoint_ids: tuple[str, ...] = ()
    particles: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_ids", _stable_ids(self.series_ids, "series_ids"))
        object.__setattr__(
            self,
            "checkpoint_ids",
            _stable_ids(self.checkpoint_ids, "checkpoint_ids"),
        )
        if self.particles is not None and self.particles < 2:
            raise ValueError("PredictionQuery.particles must be at least two.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PredictionQuery.seed must be nonnegative.")


@dataclass(frozen=True)
class PredictionResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class CounterfactualQuery:
    """Same-start reference query for one perturbation-indexed population series."""

    series_id: str
    context_policy: str = "self_consistent"
    same_noise: bool = True
    particles: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", str(self.series_id))
        if not self.series_id:
            raise ValueError("CounterfactualQuery.series_id must be nonempty.")
        if self.context_policy not in {"self_consistent", "clamped"}:
            raise ValueError("Unknown counterfactual context policy.")
        if not self.same_noise:
            raise ValueError("CREDO same-start counterfactuals require same_noise=True.")
        if self.particles is not None and self.particles < 2:
            raise ValueError("CounterfactualQuery.particles must be at least two.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("CounterfactualQuery.seed must be nonnegative.")


@dataclass(frozen=True)
class CounterfactualResult:
    counterfactuals: pd.DataFrame


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
class LongitudinalPerturbSeqRequirements:
    """Biological capabilities required before an LPS recipe is compiled."""

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
    checkpoint_mode: str = "any"
    supported_intervention_timings: frozenset[str] = frozenset(
        {"before_source", "at_source", "between_checkpoints", "unknown"}
    )
    supported_perturbation_kinds: frozenset[str] = frozenset(
        {
            "crispr_ko",
            "crispr_i",
            "crispr_a",
            "base_edit",
            "prime_edit",
            "chemical",
            "cytokine",
            "receptor_blockade",
            "combination",
            "other",
        }
    )
    supports_combinations: bool = True
    supports_unseen_targets: bool = False
    representation_scope_modes: frozenset[str] = frozenset(
        {
            "external_frozen",
            "shared_all_observations",
            "shared_source_only",
            "nested_by_subject",
            "nested_by_perturbation",
            "nested_by_checkpoint",
            "fully_nested",
        }
    )
    requires_controls: bool = True
    supported_reference_scopes: frozenset[str] = frozenset(
        {"global", "subject", "experimental_unit", "context", "checkpoint", "processing_batch"}
    )
    supported_continuity_kinds: frozenset[str] = frozenset(
        {
            "same_experimental_unit",
            "matched_subject_parallel",
            "cross_sectional_population",
            "independent_replicate",
            "lineage_linked",
            "exact_lineage_traced",
            "unknown",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "supported_axis_kinds",
            "supported_topologies",
            "supported_representation_kinds",
            "permitted_abundance_semantics",
        ):
            values = frozenset(str(value) for value in getattr(self, name))
            if not values:
                raise ValueError(f"LongitudinalPerturbSeqRequirements.{name} must be nonempty.")
            object.__setattr__(self, name, values)

        if self.abundance_requirement not in {"required", "optional", "forbidden"}:
            raise ValueError("Unknown abundance requirement.")
        if self.implicit_no_channel_semantics not in {"unit", "none"}:
            raise ValueError("Unknown implicit no-channel abundance semantics.")
        if self.maximum_reference_pools is not None and self.maximum_reference_pools < 1:
            raise ValueError("maximum_reference_pools must be positive when provided.")
        if self.context_scope not in {
            "none",
            "series_static",
            "observation_varying",
            "population_ecology",
        }:
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
            raise ValueError(
                "LongitudinalPerturbSeqRequirements contains unknown selection policies."
            )
        object.__setattr__(self, "composition_policies", composition_policies)
        object.__setattr__(self, "replicate_modes", replicate_modes)
        if self.checkpoint_mode not in {"endpoint", "multitime", "any"}:
            raise ValueError("Unknown checkpoint mode.")
        for name in (
            "supported_intervention_timings",
            "supported_perturbation_kinds",
            "representation_scope_modes",
            "supported_reference_scopes",
            "supported_continuity_kinds",
        ):
            values = frozenset(str(value) for value in getattr(self, name))
            if not values:
                raise ValueError(f"LongitudinalPerturbSeqRequirements.{name} must be nonempty.")
            object.__setattr__(self, name, values)

    @property
    def context_mode(self) -> str:
        return self.context_scope

    @property
    def supports_missing_target_geometry(self) -> bool:
        return self.permits_missing_target_geometry

    @property
    def supported_replicate_modes(self) -> frozenset[str]:
        return self.replicate_modes


RecipeRequirements = LongitudinalPerturbSeqRequirements


@runtime_checkable
class LongitudinalPerturbSeqRecipe(Protocol):
    """Public recipe contract over one compiled Perturb-seq problem."""

    recipe_id: str
    recipe_version: str

    def config_schema(self) -> type[Any]: ...

    def requirements(self, config: Any) -> LongitudinalPerturbSeqRequirements: ...

    def plan_split(
        self,
        view: PerturbSeqView,
        config: Any,
        requested: SplitSpec | None = None,
    ) -> SplitPlan: ...

    def compile(
        self,
        view: PerturbSeqView,
        split: SplitPlan,
        config: Any,
    ) -> CompiledLPSProblem: ...

    def validate_compiled(
        self,
        problem: CompiledLPSProblem,
        config: Any,
    ) -> ValidationReport: ...

    def fit(self, problem: CompiledLPSProblem, config: Any, **runtime_options: Any) -> Any: ...

    def load(
        self,
        state: Path,
        problem: CompiledLPSProblem,
        config: Any,
        **runtime_options: Any,
    ) -> Any: ...

    def predict(self, run: Any, query: PredictionQuery) -> PredictionResult: ...

    def counterfactual(
        self,
        run: Any,
        query: CounterfactualQuery,
    ) -> CounterfactualResult: ...


@runtime_checkable
class CREDORecipe(LongitudinalPerturbSeqRecipe, Protocol):
    """Released executor extension retained for alpha-cycle compatibility."""

    capabilities: CapabilitySet

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


ModelRecipe = LongitudinalPerturbSeqRecipe


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
    view: StudyView | PerturbSeqView,
    split: SplitSpec | SplitPlan,
    requirements: RecipeRequirements,
) -> ValidationReport:
    """Validate a semantic view before any recipe-owned tensorization."""
    issues: list[ValidationIssue] = []
    design = view.study.design
    is_lps = isinstance(view, PerturbSeqView)
    native_lps_semantics = is_lps and "schema_v3_conversion" not in view.study.provenance
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
    checkpoint_count = len(design.checkpoint_ids)
    if requirements.checkpoint_mode == "endpoint" and checkpoint_count != 2:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.checkpoint_mode",
                "Recipe requires an endpoint design; selected design has "
                f"{checkpoint_count} checkpoints.",
                ("design", "checkpoints"),
            )
        )
    if requirements.checkpoint_mode == "multitime" and checkpoint_count < 3:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.checkpoint_mode",
                "Recipe requires at least three ordered checkpoints.",
                ("design", "checkpoints"),
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
    if is_lps and view.representation.scope_mode not in requirements.representation_scope_modes:
        issues.append(
            ValidationIssue(
                "error",
                "recipe.representation_scope",
                f"Recipe does not support representation scope {view.representation.scope_mode!r}.",
                ("representations", view.representation_id, "scope_mode"),
            )
        )
    if is_lps:
        perturbations = view.perturbations()
        unsupported_perturbations = set(perturbations["perturbation_kind"]) - set(
            requirements.supported_perturbation_kinds
        )
        if unsupported_perturbations:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.perturbation_kind",
                    "Recipe does not support perturbation kinds "
                    f"{sorted(unsupported_perturbations)}.",
                    ("perturbations", "perturbation_kind"),
                )
            )
        if requirements.requires_controls and not perturbations["is_control"].any():
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.controls",
                    "Recipe requires at least one selected experimental control.",
                    ("perturbations", "is_control"),
                )
            )
        if (
            not requirements.supports_combinations
            and perturbations["perturbation_kind"].eq("combination").any()
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.combinations",
                    "Recipe does not support combination perturbations.",
                    ("perturbations", "perturbation_kind"),
                )
            )
        selected_series = view.series()
        unsupported_continuity = set(selected_series["continuity_kind"]) - set(
            requirements.supported_continuity_kinds
        )
        if unsupported_continuity:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.continuity_kind",
                    f"Recipe does not support continuity kinds {sorted(unsupported_continuity)}.",
                    ("series", "continuity_kind"),
                )
            )
        events = view.study.intervention_events._unsafe_view()
        events = events.loc[events["series_id"].isin(view.series_ids)]
        unsupported_timings = set(events["start_relation"]) - set(
            requirements.supported_intervention_timings
        )
        if unsupported_timings:
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.intervention_timing",
                    f"Recipe does not support intervention timings {sorted(unsupported_timings)}.",
                    ("intervention_events", "start_relation"),
                )
            )
        if (
            isinstance(split, SplitPlan)
            and split.task_kind == "target_generalization"
            and not requirements.supports_unseen_targets
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.unseen_targets",
                    "Recipe does not support unseen-target generalization.",
                    ("split", "task_kind"),
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
    if requirements.context_scope == "population_ecology":
        if native_lps_semantics and (
            "population_pool_id" not in observations
            or observations["population_pool_id"].isna().any()
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "recipe.population_ecology",
                    "Population-ecology context requires a pool for every selected observation.",
                    ("observations", "population_pool_id"),
                )
            )
        elif native_lps_semantics:
            ecological = view.ecological_pools()
            selected_pool_ids = set(observations["population_pool_id"].astype(str))
            ecological_ids = set(ecological.get("population_pool_id", pd.Series(dtype=str)))
            invalid = selected_pool_ids - ecological_ids
            if invalid:
                issues.append(
                    ValidationIssue(
                        "error",
                        "recipe.population_ecology",
                        "Sequencing, capture, or computational groups cannot supply ecological "
                        f"context; invalid pools={sorted(invalid)[:5]}.",
                        ("population_pools", "pool_kind"),
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
        perturbation_key = "perturbation_id" if is_lps else "condition_id"
        control_column = "is_control" if is_lps else "is_reference"
        selected_conditions = (
            view.study.perturbations._unsafe_view()
            if is_lps
            else view.study.conditions._unsafe_view()
        )
        selected_conditions = selected_conditions.loc[
            selected_conditions[perturbation_key].isin(selected_series[perturbation_key])
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
            pool_by_condition = selected_binding.set_index(perturbation_key)[
                "reference_pool_id"
            ].astype(str)
            missing_binding = set(selected_conditions[perturbation_key]) - set(
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
            selected_conditions.loc[selected_conditions[control_column], perturbation_key]
        )
        intervention_condition_ids = set(
            selected_conditions.loc[~selected_conditions[control_column], perturbation_key]
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
                    ("perturbations" if is_lps else "conditions", control_column),
                )
            )
        if is_lps and not selected_binding.empty:
            unsupported_scopes = set(selected_binding["scope_kind"]) - set(
                requirements.supported_reference_scopes
            )
            if unsupported_scopes:
                issues.append(
                    ValidationIssue(
                        "error",
                        "recipe.reference_scope",
                        f"Recipe does not support reference scopes {sorted(unsupported_scopes)}.",
                        ("reference_bindings", "scope_kind"),
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
        study: PerturbSeqStudy | PerturbSeqView | Study | StudyView | CREDOStudy,
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
        if isinstance(study, (PerturbSeqStudy, Study)):
            study = config.view(study) if callable(getattr(config, "view", None)) else study.view()
        if isinstance(study, (PerturbSeqView, StudyView)):
            split = recipe.plan_split(study, recipe_config, split)
            validate_split_plan(study, split)
            validate_representation_scope(study, split)
            requirements = recipe.requirements(recipe_config)
            validate_view_for_recipe(study, split, requirements).raise_for_errors()
            compiler = getattr(recipe, "compile", None)
            compiled_study = (
                compiler(study, split, recipe_config)
                if callable(compiler)
                else recipe.compile_study(study, split, recipe_config)
            )
        else:
            compiled_study = study
        recipe.validate_compiled(compiled_study, recipe_config).raise_for_errors()
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
    study: PerturbSeqStudy | PerturbSeqView | Study | StudyView | CREDOStudy,
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
    selected = config.view(study) if isinstance(study, (PerturbSeqStudy, Study)) else study
    return TrainingEngine().fit(recipe, selected, config, split=split, **kwargs)


__all__ = [
    "CREDORecipe",
    "CREDORun",
    "CheckpointCodec",
    "CounterfactualQuery",
    "CounterfactualResult",
    "LossReport",
    "LongitudinalPerturbSeqRecipe",
    "LongitudinalPerturbSeqRequirements",
    "ModelRecipe",
    "ObjectiveDescriptor",
    "ObjectiveTerm",
    "PredictionQuery",
    "PredictionResult",
    "RecipeRequirements",
    "RuntimeState",
    "TrainingEngine",
    "train",
    "validate_recipe_study",
    "validate_training_contract",
    "validate_view_for_recipe",
]
