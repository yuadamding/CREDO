"""Canonical biological contract for longitudinal Perturb-seq studies."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..data.design import LongitudinalDesign
from ..data.representations import RepresentationCatalog, RepresentationSpec
from ..data.study import (
    SelectionSpec,
    StudyManifest,
    VerificationLevel,
    _apply_filter,
    _canonical_value,
    _frame_digest,
    _representation_identity,
    _semantic_digest,
)
from ..data.study import (
    Study as SchemaV3Study,
)
from ..data.support import (
    AbundanceValue,
    MeasureSnapshot,
    SupportRef,
    SupportStore,
    SupportStoreRegistry,
)
from ..data.tables import (
    AbundanceTable,
    ContextTable,
    InterventionEventTable,
    LPSCompositionTable,
    PerturbationComponentTable,
    PerturbationEffectBindingTable,
    PerturbationReferenceBindingTable,
    PerturbationTable,
    PopulationPoolTable,
    PopulationSeriesTable,
    SnapshotObservationTable,
    SupportIndexTable,
)
from ..data.validation import ValidationIssue, ValidationReport

_DEFAULT_ABUNDANCE = object()


@dataclass(frozen=True)
class PerturbSeqManifest(StudyManifest):
    """Identity and authoritative defaults for a native Perturb-seq study."""


PerturbSeqSelection = SelectionSpec


def _optional_text(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    value = str(row[column])
    return value or None


@dataclass(frozen=True)
class PerturbSeqStudy:
    """Longitudinal perturbation-indexed populations observed destructively."""

    manifest: PerturbSeqManifest
    design: LongitudinalDesign
    perturbations: PerturbationTable
    perturbation_components: PerturbationComponentTable | None
    intervention_events: InterventionEventTable
    contexts: ContextTable | None
    series: PopulationSeriesTable
    observations: SnapshotObservationTable
    representations: RepresentationCatalog
    support_index: SupportIndexTable
    supports: SupportStoreRegistry | SupportStore
    abundance: AbundanceTable | None = None
    compositions: LPSCompositionTable | None = None
    population_pools: PopulationPoolTable | None = None
    effect_bindings: PerturbationEffectBindingTable | None = None
    reference_bindings: PerturbationReferenceBindingTable | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)
    _content_hash_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.manifest.schema_version) != 4:
            raise ValueError("PerturbSeqStudy requires semantic schema_version=4.")
        supports = self.supports
        if isinstance(supports, SupportStore):
            supports = SupportStoreRegistry((supports,))
        if not isinstance(supports, SupportStoreRegistry):
            raise TypeError(
                "PerturbSeqStudy.supports must be a SupportStoreRegistry or SupportStore."
            )
        object.__setattr__(self, "supports", supports)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def validate(self, level: VerificationLevel = "semantic") -> ValidationReport:
        """Validate biological identities, observation semantics, and support storage."""
        if level not in {"none", "schema", "manifest", "semantic", "full"}:
            raise ValueError(f"Unknown validation level {level!r}.")
        if level in {"none", "schema"}:
            return ValidationReport()

        issues: list[ValidationIssue] = []
        representation_ids = set(self.representations)
        if self.manifest.primary_representation not in representation_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "manifest.primary_representation",
                    "Primary representation is absent from the representation catalog.",
                    ("manifest", "primary_representation"),
                )
            )
        if self.manifest.primary_abundance_channel is not None and (
            self.abundance is None
            or self.manifest.primary_abundance_channel not in self.abundance.channels
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "manifest.primary_abundance",
                    "Primary abundance channel is absent from the abundance catalog.",
                    ("manifest", "primary_abundance_channel"),
                )
            )
        for field_name, table in (
            ("primary_effect_binding", self.effect_bindings),
            ("primary_reference_binding", self.reference_bindings),
        ):
            binding_id = getattr(self.manifest, field_name)
            if binding_id is not None and (table is None or binding_id not in table.binding_ids):
                issues.append(
                    ValidationIssue(
                        "error",
                        f"manifest.{field_name}",
                        f"{field_name.replace('_', ' ').title()} is absent from its catalog.",
                        ("manifest", field_name),
                    )
                )
        for representation in self.representations.values():
            store_id = representation.support_store_id
            if store_id not in self.supports:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.store_missing",
                        f"Representation {representation.representation_id!r} references "
                        f"unknown store {store_id!r}.",
                        ("representations", representation.representation_id),
                    )
                )
                continue
            store = self.supports[store_id]
            if representation.representation_id not in store.representation_ids():
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.store_missing",
                        f"Store {store_id!r} does not expose representation "
                        f"{representation.representation_id!r}.",
                        ("representations", representation.representation_id),
                    )
                )
                continue
            dimension = store.dimension(representation.representation_id)
            if dimension != representation.dimension:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.dimension",
                        f"Representation {representation.representation_id!r} declares "
                        f"dimension {representation.dimension}, store reports {dimension}.",
                        ("representations", representation.representation_id, "dimension"),
                    )
                )
        if level == "manifest":
            return ValidationReport(tuple(issues))

        perturbations = self.perturbations._unsafe_view()
        series = self.series._unsafe_view()
        observations = self.observations._unsafe_view()
        support_index = self.support_index._unsafe_view()
        perturbation_ids = set(perturbations["perturbation_id"])
        series_ids = set(series["series_id"])
        observation_ids = set(observations["observation_id"])
        checkpoint_ids = set(self.design.checkpoint_ids)
        subject_ids = set(series["subject_id"].astype(str))
        experimental_unit_ids = set(series["experimental_unit_id"].astype(str))

        for representation in self.representations.values():
            unknown_series = set(representation.included_series) - series_ids
            unknown_checkpoints = set(representation.included_checkpoints) - checkpoint_ids
            unknown_subjects = set(representation.fit_subject_ids) - subject_ids
            unknown_perturbations = set(representation.fit_perturbation_ids) - perturbation_ids
            unknown_fit_checkpoints = set(representation.fit_checkpoint_ids) - checkpoint_ids
            if (
                unknown_series
                or unknown_checkpoints
                or unknown_subjects
                or unknown_perturbations
                or unknown_fit_checkpoints
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.fit_scope_fk",
                        f"Representation {representation.representation_id!r} fit scope "
                        "references unknown identities; "
                        f"series={sorted(unknown_series)[:5]}, "
                        f"subjects={sorted(unknown_subjects)[:5]}, "
                        f"perturbations={sorted(unknown_perturbations)[:5]}, "
                        "checkpoints="
                        f"{sorted(unknown_checkpoints | unknown_fit_checkpoints)[:5]}.",
                        ("representations", representation.representation_id),
                    )
                )
            fit_checkpoints = set(
                representation.fit_checkpoint_ids or representation.included_checkpoints
            )
            if representation.scope_mode == "shared_source_only" and fit_checkpoints != {
                self.design.source_checkpoint_id
            }:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.source_scope",
                        "shared_source_only must record exactly the source checkpoint in its "
                        f"fit scope; representation={representation.representation_id!r}.",
                        ("representations", representation.representation_id),
                    )
                )

        unknown_series_perturbations = set(series["perturbation_id"]) - perturbation_ids
        if unknown_series_perturbations:
            issues.append(
                ValidationIssue(
                    "error",
                    "series.perturbation_fk",
                    "Population series reference unknown perturbations: "
                    f"{sorted(unknown_series_perturbations)[:5]}.",
                    ("series", "perturbation_id"),
                )
            )
        if self.perturbation_components is not None:
            components = self.perturbation_components._unsafe_view()
            unknown = set(components["perturbation_id"]) - perturbation_ids
            if unknown:
                issues.append(
                    ValidationIssue(
                        "error",
                        "perturbation_components.perturbation_fk",
                        f"Components reference unknown perturbations: {sorted(unknown)[:5]}.",
                        ("perturbation_components", "perturbation_id"),
                    )
                )
            component_counts = components.groupby("perturbation_id", observed=True).size()
            combinations = set(
                perturbations.loc[
                    perturbations["perturbation_kind"].eq("combination"),
                    "perturbation_id",
                ]
            )
            incomplete_combinations = {
                value for value in combinations if int(component_counts.get(value, 0)) < 2
            }
            if incomplete_combinations:
                issues.append(
                    ValidationIssue(
                        "error",
                        "perturbation_components.combination_coverage",
                        "Combination perturbations require at least two normalized components; "
                        f"invalid={sorted(incomplete_combinations)[:5]}.",
                        ("perturbation_components",),
                    )
                )
        elif perturbations["perturbation_kind"].eq("combination").any():
            issues.append(
                ValidationIssue(
                    "error",
                    "perturbation_components.combination_coverage",
                    "Combination perturbations require a component table.",
                    ("perturbation_components",),
                )
            )
        events = self.intervention_events._unsafe_view()
        unknown_event_series = set(events["series_id"]) - series_ids
        if unknown_event_series:
            issues.append(
                ValidationIssue(
                    "error",
                    "intervention_events.series_fk",
                    "Intervention events reference unknown series: "
                    f"{sorted(unknown_event_series)[:5]}.",
                    ("intervention_events", "series_id"),
                )
            )
        primary_events = events.loc[events["modeled_role"].eq("primary_perturbation")]
        event_counts = primary_events.groupby("series_id", observed=True).size()
        missing_events = series_ids - set(event_counts.index)
        duplicate_events = event_counts[event_counts.ne(1)]
        if missing_events or len(duplicate_events):
            issues.append(
                ValidationIssue(
                    "error",
                    "intervention_events.primary_coverage",
                    "Every population series requires exactly one primary perturbation event; "
                    f"missing={sorted(missing_events)[:5]}, "
                    f"nonunique={sorted(map(str, duplicate_events.index))[:5]}.",
                    ("intervention_events", "modeled_role"),
                )
            )
        series_perturbation = series.set_index("series_id")["perturbation_id"]
        for row in primary_events.itertuples(index=False):
            if row.series_id in series_perturbation.index and str(row.agent_id) != str(
                series_perturbation.loc[row.series_id]
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "intervention_events.primary_alignment",
                        f"Primary event {row.event_id!r} agent does not match its "
                        "series perturbation.",
                        ("intervention_events", str(row.event_id), "agent_id"),
                    )
                )
        axis_id = self.design.axis.axis_id
        coordinates = {
            checkpoint.checkpoint_id: float(checkpoint.coordinates[axis_id])
            for checkpoint in self.design.checkpoints
        }
        source_coordinate = coordinates[self.design.source_checkpoint_id]
        last_coordinate = max(coordinates.values())
        for row in events.itertuples(index=False):
            start = None if pd.isna(row.start_coordinate) else float(row.start_coordinate)
            relation = str(row.start_relation)
            invalid = False
            if relation == "before_source" and start is not None:
                invalid = start >= source_coordinate
            elif relation == "at_source" and start is not None:
                invalid = not np.isclose(start, source_coordinate)
            elif relation == "between_checkpoints":
                invalid = start is None or not (source_coordinate < start < last_coordinate)
            elif relation == "after_last_observation" and start is not None:
                invalid = start <= last_coordinate
            if invalid:
                issues.append(
                    ValidationIssue(
                        "error",
                        "intervention_events.start_relation",
                        f"Event {row.event_id!r} coordinate is inconsistent with "
                        f"start_relation={relation!r}.",
                        ("intervention_events", str(row.event_id), "start_coordinate"),
                    )
                )

        unknown_observation_series = set(observations["series_id"]) - series_ids
        unknown_checkpoints = set(observations["checkpoint_id"]) - checkpoint_ids
        if unknown_observation_series or unknown_checkpoints:
            issues.append(
                ValidationIssue(
                    "error",
                    "observations.foreign_key",
                    "Snapshot observations have unknown identities: "
                    f"series={sorted(unknown_observation_series)[:5]}, "
                    f"checkpoints={sorted(unknown_checkpoints)[:5]}.",
                    ("observations",),
                )
            )
        observed_contexts = (
            set(observations["context_id"].dropna().astype(str))
            if "context_id" in observations
            else set()
        )
        declared_contexts = (
            set() if self.contexts is None else set(self.contexts._unsafe_view()["context_id"])
        )
        unknown_contexts = observed_contexts - declared_contexts
        if unknown_contexts:
            issues.append(
                ValidationIssue(
                    "error",
                    "observations.context_fk",
                    f"Observations reference unknown contexts: {sorted(unknown_contexts)[:5]}.",
                    ("observations", "context_id"),
                )
            )

        observed_pools = (
            set(observations["population_pool_id"].dropna().astype(str))
            if "population_pool_id" in observations
            else set()
        )
        declared_pools = (
            set()
            if self.population_pools is None
            else set(self.population_pools._unsafe_view()["population_pool_id"])
        )
        unknown_pools = observed_pools - declared_pools
        if unknown_pools:
            issues.append(
                ValidationIssue(
                    "error",
                    "observations.population_pool_fk",
                    "Observations reference unknown population pools: "
                    f"{sorted(unknown_pools)[:5]}.",
                    ("observations", "population_pool_id"),
                )
            )
        if self.population_pools is not None:
            pools = self.population_pools._unsafe_view()
            unknown_pool_checkpoints = set(pools["checkpoint_id"]) - checkpoint_ids
            unknown_pool_units = set(pools["experimental_unit_id"]) - experimental_unit_ids
            if unknown_pool_checkpoints or unknown_pool_units:
                issues.append(
                    ValidationIssue(
                        "error",
                        "population_pools.foreign_key",
                        "Population pools reference unknown study identities; "
                        f"checkpoints={sorted(unknown_pool_checkpoints)[:5]}, "
                        f"experimental_units={sorted(unknown_pool_units)[:5]}.",
                        ("population_pools",),
                    )
                )
            if "population_pool_id" in observations:
                observation_pool = observations.loc[
                    observations["population_pool_id"].notna(),
                    ["observation_id", "population_pool_id", "checkpoint_id"],
                ].merge(
                    pools[["population_pool_id", "checkpoint_id"]],
                    on="population_pool_id",
                    how="inner",
                    suffixes=("_observation", "_pool"),
                )
                mismatched = observation_pool.loc[
                    ~observation_pool["checkpoint_id_observation"].eq(
                        observation_pool["checkpoint_id_pool"]
                    )
                ]
                if len(mismatched):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "population_pools.checkpoint_alignment",
                            "Population pool checkpoint disagrees with a linked observation; "
                            f"observation={mismatched.iloc[0]['observation_id']!r}.",
                            ("population_pools",),
                        )
                    )
        duplicate_snapshots = observations.duplicated(["series_id", "checkpoint_id"], keep=False)
        if duplicate_snapshots.any():
            duplicated = observations.loc[duplicate_snapshots]
            invalid_replicates = (
                "technical_replicate_id" not in duplicated
                or duplicated["technical_replicate_id"].isna().any()
            )
            if not invalid_replicates:
                invalid_replicates = duplicated.duplicated(
                    ["series_id", "checkpoint_id", "technical_replicate_id"]
                ).any()
            if invalid_replicates:
                issues.append(
                    ValidationIssue(
                        "error",
                        "observations.replicate_identity",
                        "Replicate snapshots at one series/checkpoint require distinct "
                        "technical_replicate_id values.",
                        ("observations", "technical_replicate_id"),
                    )
                )

        expected_pairs = pd.MultiIndex.from_product(
            [observations["observation_id"], tuple(self.representations)],
            names=["observation_id", "representation_id"],
        )
        actual_pairs = pd.MultiIndex.from_frame(
            support_index[["observation_id", "representation_id"]]
        )
        invalid_support_observations = set(support_index["observation_id"]) - observation_ids
        invalid_support_representations = (
            set(support_index["representation_id"]) - representation_ids
        )
        missing_pairs = expected_pairs.difference(actual_pairs)
        extra_pairs = actual_pairs.difference(expected_pairs)
        if (
            invalid_support_observations
            or invalid_support_representations
            or len(missing_pairs)
            or len(extra_pairs)
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "support_index.coverage",
                    "Support index must declare each observation/representation pair; "
                    f"missing={missing_pairs.tolist()[:5]}, extra={extra_pairs.tolist()[:5]}.",
                    ("support_index",),
                )
            )
        geometry = observations.set_index("observation_id")["geometry_observed"]
        for row in support_index.itertuples(index=False):
            if row.observation_id not in geometry.index:
                continue
            if bool(row.available) and not bool(geometry.loc[row.observation_id]):
                issues.append(
                    ValidationIssue(
                        "error",
                        "support_index.geometry_alignment",
                        f"Support is available for geometry-missing observation "
                        f"{row.observation_id!r}.",
                        ("support_index", str(row.observation_id), str(row.representation_id)),
                    )
                )
            if not bool(row.available) or row.representation_id not in self.representations:
                continue
            representation = self.representations[row.representation_id]
            if row.store_id != representation.support_store_id:
                issues.append(
                    ValidationIssue(
                        "error",
                        "support_index.store_alignment",
                        f"Support {row.observation_id!r}/{row.representation_id!r} uses "
                        f"store {row.store_id!r}, expected {representation.support_store_id!r}.",
                        ("support_index", str(row.observation_id)),
                    )
                )
                continue
            ref = SupportRef(str(row.store_id), str(row.representation_id), str(row.support_key))
            if not self.supports.contains(ref):
                issues.append(
                    ValidationIssue(
                        "error",
                        "support.reference_fk",
                        f"Support reference is missing for {row.observation_id!r}.",
                        ("support_index", str(row.observation_id)),
                    )
                )
        supported_observations = set(
            support_index.loc[support_index["available"], "observation_id"].astype(str)
        )
        missing_geometry = set(geometry.loc[geometry].index.astype(str)) - supported_observations
        if missing_geometry:
            issues.append(
                ValidationIssue(
                    "error",
                    "support_index.geometry_coverage",
                    "Geometry-observed snapshots require at least one available representation; "
                    f"missing={sorted(missing_geometry)[:5]}.",
                    ("support_index",),
                )
            )

        abundance_observed = observations.set_index("observation_id")["abundance_observed"]
        observed_abundance_ids: set[str] = set()
        if self.abundance is not None:
            abundance = self.abundance._unsafe_view()
            unknown = set(abundance["observation_id"]) - observation_ids
            if unknown:
                issues.append(
                    ValidationIssue(
                        "error",
                        "abundance.observation_fk",
                        f"Abundance references unknown observations: {sorted(unknown)[:5]}.",
                        ("abundance", "observation_id"),
                    )
                )
            observed_abundance_ids = set(
                abundance.loc[abundance["observed"], "observation_id"].astype(str)
            )
        declared_abundance_ids = set(abundance_observed.loc[abundance_observed].index.astype(str))
        if observed_abundance_ids != declared_abundance_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "abundance.observation_alignment",
                    "abundance_observed must agree with observed abundance rows; "
                    f"missing={sorted(declared_abundance_ids - observed_abundance_ids)[:5]}, "
                    f"undeclared={sorted(observed_abundance_ids - declared_abundance_ids)[:5]}.",
                    ("observations", "abundance_observed"),
                )
            )
        if self.abundance is not None:
            abundance = self.abundance._unsafe_view()
            observed_values = abundance.loc[abundance["observed"]].merge(
                observations,
                on="observation_id",
                how="left",
                validate="many_to_one",
            )
            for channel_id, spec in self.abundance.channels.items():
                rows = observed_values.loc[observed_values["channel_id"].eq(channel_id)]
                if spec.input_channel_id is not None:
                    input_ids = set(
                        observed_values.loc[
                            observed_values["channel_id"].eq(spec.input_channel_id),
                            "observation_id",
                        ]
                    )
                    missing_inputs = set(rows["observation_id"]) - input_ids
                    if missing_inputs:
                        issues.append(
                            ValidationIssue(
                                "error",
                                "abundance.transform_coverage",
                                f"Transformed channel {channel_id!r} lacks observed input rows; "
                                f"observations={sorted(missing_inputs)[:5]}.",
                                ("abundance", channel_id),
                            )
                        )
                group_columns = {
                    "sample_checkpoint": ("sample_id", "checkpoint_id"),
                    "context_checkpoint": ("context_id", "checkpoint_id"),
                }.get(spec.denominator_scope)
                if group_columns is None or rows.empty:
                    continue
                if any(column not in rows for column in group_columns):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "abundance.denominator_scope",
                            f"Channel {channel_id!r} cannot resolve denominator scope "
                            f"{spec.denominator_scope!r} from observation metadata.",
                            ("abundance", channel_id),
                        )
                    )
                    continue
                inconsistent = rows.groupby("denominator_id", observed=True).agg(
                    **{f"{column}_count": (column, "nunique") for column in group_columns}
                )
                if inconsistent.gt(1).any(axis=None):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "abundance.denominator_scope",
                            f"Channel {channel_id!r} denominator spans multiple "
                            f"{spec.denominator_scope} groups.",
                            ("abundance", channel_id, "denominator_id"),
                        )
                    )
        if self.compositions is not None:
            compositions = self.compositions._unsafe_view()
            unknown = set(compositions["observation_id"]) - observation_ids
            if unknown:
                issues.append(
                    ValidationIssue(
                        "error",
                        "compositions.observation_fk",
                        f"Compositions reference unknown observations: {sorted(unknown)[:5]}.",
                        ("compositions", "observation_id"),
                    )
                )
            aligned = compositions.merge(
                observations[["observation_id", "series_id", "checkpoint_id"]],
                on="observation_id",
                how="inner",
                suffixes=("_composition", "_observation"),
                validate="many_to_one",
            )
            mismatched = aligned.loc[
                ~aligned["series_id_composition"].eq(aligned["series_id_observation"])
                | ~aligned["checkpoint_id_composition"].eq(aligned["checkpoint_id_observation"])
            ]
            if len(mismatched):
                issues.append(
                    ValidationIssue(
                        "error",
                        "compositions.observation_alignment",
                        "Composition series/checkpoint disagrees with its observation; "
                        f"observation={mismatched.iloc[0]['observation_id']!r}.",
                        ("compositions",),
                    )
                )
        for name, table in (
            ("effect", self.effect_bindings),
            ("reference", self.reference_bindings),
        ):
            if table is None:
                continue
            bindings = table._unsafe_view()
            unknown = set(bindings["perturbation_id"]) - perturbation_ids
            if unknown:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{name}_binding.perturbation_fk",
                        f"{name.title()} bindings reference unknown perturbations: "
                        f"{sorted(unknown)[:5]}.",
                        (f"{name}_bindings", "perturbation_id"),
                    )
                )
            for binding_id, rows in bindings.groupby("binding_id", observed=True):
                missing = perturbation_ids - set(rows["perturbation_id"])
                if missing:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"{name}_binding.coverage",
                            f"Binding {binding_id!r} omits perturbations: {sorted(missing)[:5]}.",
                            (f"{name}_bindings", str(binding_id)),
                        )
                    )
        all_effect_ids: set[str] = set()
        if self.effect_bindings is not None:
            effects = self.effect_bindings._unsafe_view()
            all_effect_ids = set(effects["effect_id"].astype(str))
            if "parent_effect_id" in effects:
                for binding_id, rows in effects.groupby("binding_id", observed=True):
                    declared = set(rows["effect_id"].astype(str))
                    parents = set(rows["parent_effect_id"].dropna().astype(str))
                    unknown = parents - declared
                    if unknown:
                        issues.append(
                            ValidationIssue(
                                "error",
                                "effect_binding.parent_fk",
                                f"Effect binding {binding_id!r} references unknown parent "
                                f"effects: {sorted(unknown)[:5]}.",
                                ("effect_bindings", str(binding_id), "parent_effect_id"),
                            )
                        )
        if self.reference_bindings is not None:
            references = self.reference_bindings._unsafe_view()
            control_ids = set(
                perturbations.loc[perturbations["is_control"], "perturbation_id"].astype(str)
            )
            for (binding_id, pool_id), rows in references.groupby(
                ["binding_id", "reference_pool_id"], observed=True
            ):
                if not (set(rows["perturbation_id"].astype(str)) & control_ids):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "reference_binding.control_pool",
                            f"Reference pool {pool_id!r} in binding {binding_id!r} has no "
                            "observed control perturbation.",
                            ("reference_bindings", str(binding_id), str(pool_id)),
                        )
                    )
            unknown_effects = (
                set(references["counterfactual_effect_id"].astype(str)) - all_effect_ids
            )
            if self.effect_bindings is not None and unknown_effects:
                issues.append(
                    ValidationIssue(
                        "error",
                        "reference_binding.counterfactual_effect_fk",
                        "Reference bindings name unknown counterfactual effects: "
                        f"{sorted(unknown_effects)[:5]}.",
                        ("reference_bindings", "counterfactual_effect_id"),
                    )
                )
        return ValidationReport(tuple(issues)).merged(
            self.supports.validate(full_scan=level == "full")
        )

    def content_hash(self) -> str:
        """Hash all biological semantics, representation contracts, and support content."""
        if self._content_hash_cache is not None:
            return self._content_hash_cache
        table_values = {
            "perturbations": self.perturbations,
            "perturbation_components": self.perturbation_components,
            "intervention_events": self.intervention_events,
            "contexts": self.contexts,
            "series": self.series,
            "observations": self.observations,
            "support_index": self.support_index,
            "abundance": self.abundance,
            "compositions": self.compositions,
            "population_pools": self.population_pools,
            "effect_bindings": self.effect_bindings,
            "reference_bindings": self.reference_bindings,
        }
        table_hashes = {
            name: (
                None
                if table is None
                else _frame_digest(table._unsafe_view(), key_columns=table.key_columns)
            )
            for name, table in table_values.items()
        }
        representation_payload = {
            representation_id: _representation_identity(self.representations[representation_id])
            for representation_id in sorted(self.representations)
        }
        support_payload: dict[str, Any] = {}
        support_index = self.support_index._unsafe_view()
        for representation_id in sorted(self.representations):
            representation = self.representations[representation_id]
            if representation.support_artifact is not None:
                support_payload[representation_id] = {
                    "artifact_sha256": representation.support_artifact.sha256,
                    "semantic_hash": representation.support_artifact.semantic_hash,
                }
                continue
            store = self.supports[representation.support_store_id]
            semantic_identity = getattr(store, "semantic_identity", None)
            if callable(semantic_identity):
                support_payload[representation_id] = _canonical_value(
                    semantic_identity(representation_id)
                )
                continue
            digest = __import__("hashlib").sha256()
            rows = support_index.loc[
                support_index["representation_id"].eq(representation_id)
                & support_index["available"]
            ].sort_values(["store_id", "support_key"])
            for row in rows.itertuples(index=False):
                ref = SupportRef(str(row.store_id), representation_id, str(row.support_key))
                law = self.supports.read(ref)
                coordinates = np.asarray(law.coordinates, dtype="<f4", order="C")
                probabilities = np.asarray(law.probabilities, dtype="<f8", order="C")
                digest.update(str(row.store_id).encode())
                digest.update(b"\0")
                digest.update(str(row.support_key).encode())
                digest.update(b"\0")
                digest.update(np.asarray(coordinates.shape, dtype="<i8").tobytes())
                digest.update(coordinates.tobytes(order="C"))
                digest.update(probabilities.tobytes(order="C"))
            support_payload[representation_id] = {"materialized_sha256": digest.hexdigest()}
        channels = (
            {}
            if self.abundance is None
            else {
                channel_id: _canonical_value(spec)
                for channel_id, spec in sorted(self.abundance.channels.items())
            }
        )
        digest = _semantic_digest(
            {
                "manifest": _canonical_value(self.manifest),
                "design": _canonical_value(self.design),
                "tables": table_hashes,
                "abundance_channels": channels,
                "representations": representation_payload,
                "supports": support_payload,
                "source_artifact_hashes": _canonical_value(self.provenance.get("input_hashes", {})),
            }
        )
        object.__setattr__(self, "_content_hash_cache", digest)
        return digest

    def snapshot(
        self,
        observation_id: str,
        *,
        representation_id: str | None = None,
        abundance_channel: str | None | object = _DEFAULT_ABUNDANCE,
    ) -> MeasureSnapshot:
        """Read one empirical support law and one optional abundance value."""
        if self._closed:
            raise RuntimeError("PerturbSeqStudy is closed.")
        observation_id = str(observation_id)
        observations = self.observations._unsafe_view()
        if not observations["observation_id"].eq(observation_id).any():
            raise KeyError(f"Unknown observation_id {observation_id!r}.")
        selected_representation = representation_id or self.manifest.primary_representation
        self.representations[selected_representation]
        support_rows = self.support_index._unsafe_view()
        support_rows = support_rows.loc[
            support_rows["observation_id"].eq(observation_id)
            & support_rows["representation_id"].eq(selected_representation)
        ]
        if len(support_rows) != 1:
            raise KeyError(
                f"Support coverage is undeclared for {observation_id!r}/"
                f"{selected_representation!r}."
            )
        row = support_rows.iloc[0]
        law = None
        if bool(row["available"]):
            law = self.supports.read(
                SupportRef(str(row["store_id"]), selected_representation, str(row["support_key"]))
            )
        selected_channel = (
            self.manifest.primary_abundance_channel
            if abundance_channel is _DEFAULT_ABUNDANCE
            else abundance_channel
        )
        value = None
        if selected_channel is not None:
            if self.abundance is None or selected_channel not in self.abundance.channels:
                raise KeyError(f"Unknown abundance channel {selected_channel!r}.")
            matches = self.abundance._unsafe_view()
            matches = matches.loc[
                matches["observation_id"].eq(observation_id)
                & matches["channel_id"].eq(selected_channel)
            ]
            if len(matches) == 1:
                abundance_row = matches.iloc[0]
                raw = abundance_row["value"]
                value = AbundanceValue(
                    observation_id=observation_id,
                    channel_id=str(selected_channel),
                    value=None if pd.isna(raw) else float(raw),
                    observed=bool(abundance_row["observed"]),
                    denominator_id=_optional_text(abundance_row, "denominator_id"),
                    transform_id=_optional_text(abundance_row, "transform_id"),
                    source_artifact_id=_optional_text(abundance_row, "source_artifact_id"),
                )
        return MeasureSnapshot(observation_id, law, value)

    def view(
        self,
        selection: PerturbSeqSelection | None = None,
        *,
        representation_id: str | None = None,
        abundance_channel: str | None | object = _DEFAULT_ABUNDANCE,
        effect_binding_id: str | None = None,
        reference_binding_id: str | None = None,
    ) -> PerturbSeqView:
        base = selection or PerturbSeqSelection()
        return PerturbSeqView(
            study=self,
            selection=base.with_bindings(
                effect_binding_id=(
                    effect_binding_id
                    or base.effect_binding_id
                    or self.manifest.primary_effect_binding
                ),
                reference_binding_id=(
                    reference_binding_id
                    or base.reference_binding_id
                    or self.manifest.primary_reference_binding
                ),
            ),
            representation_id=(
                representation_id or base.representation_id or self.manifest.primary_representation
            ),
            abundance_channel=(
                base.abundance_channel_id or self.manifest.primary_abundance_channel
                if abundance_channel is _DEFAULT_ABUNDANCE
                else abundance_channel
            ),
        )

    def close(self) -> None:
        if not self._closed:
            self.supports.close()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> PerturbSeqStudy:
        if self._closed:
            raise RuntimeError("PerturbSeqStudy is closed.")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(frozen=True)
class PerturbSeqView:
    """Typed biological selection over one immutable PerturbSeqStudy."""

    study: PerturbSeqStudy
    selection: PerturbSeqSelection
    representation_id: str
    abundance_channel: str | None

    def __post_init__(self) -> None:
        if self.representation_id not in self.study.representations:
            raise KeyError(f"Unknown representation_id {self.representation_id!r}.")
        if self.abundance_channel is not None and (
            self.study.abundance is None
            or self.abundance_channel not in self.study.abundance.channels
        ):
            raise KeyError(f"Unknown abundance channel {self.abundance_channel!r}.")
        if self.selection.effect_binding_id is not None and (
            self.study.effect_bindings is None
            or self.selection.effect_binding_id not in self.study.effect_bindings.binding_ids
        ):
            raise KeyError(f"Unknown effect binding {self.selection.effect_binding_id!r}.")
        if self.selection.reference_binding_id is not None and (
            self.study.reference_bindings is None
            or self.selection.reference_binding_id not in self.study.reference_bindings.binding_ids
        ):
            raise KeyError(f"Unknown reference binding {self.selection.reference_binding_id!r}.")

        def require_known(
            selected: tuple[str, ...] | None,
            known: set[str],
            label: str,
        ) -> None:
            if selected is None:
                return
            unknown = set(selected) - known
            if unknown:
                raise KeyError(f"Selection references unknown {label}: {sorted(unknown)[:5]}.")

        perturbations = self.study.perturbations._unsafe_view()
        series = self.study.series._unsafe_view()
        observations = self.study.observations._unsafe_view()
        require_known(
            self.selection.perturbation_ids,
            set(perturbations["perturbation_id"].astype(str)),
            "perturbations",
        )
        require_known(
            self.selection.series_ids,
            set(series["series_id"].astype(str)),
            "series",
        )
        require_known(
            self.selection.observation_ids,
            set(observations["observation_id"].astype(str)),
            "observations",
        )
        require_known(
            self.selection.checkpoint_ids,
            set(self.study.design.checkpoint_ids),
            "checkpoints",
        )
        require_known(
            self.selection.subject_ids,
            set(series["subject_id"].astype(str)),
            "subjects",
        )
        require_known(
            self.selection.experimental_unit_ids,
            set(series["experimental_unit_id"].astype(str)),
            "experimental units",
        )
        require_known(
            self.selection.control_kinds,
            set(perturbations["control_kind"].astype(str)),
            "control kinds",
        )
        if self.selection.construct_ids is not None or self.selection.target_ids is not None:
            components = self.study.perturbation_components
            if components is None:
                raise KeyError("Construct/target selection requires perturbation components.")
            component_frame = components._unsafe_view()
            require_known(
                self.selection.construct_ids,
                set(component_frame["construct_id"].astype(str)),
                "constructs",
            )
            require_known(
                self.selection.target_ids,
                set(component_frame["target_id"].astype(str)),
                "targets",
            )
        for selected, column, label in (
            (self.selection.context_ids, "context_id", "contexts"),
            (self.selection.qc_tiers, "assignment_qc", "QC tiers"),
        ):
            if selected is not None and column not in observations:
                raise KeyError(f"Selection requires missing observation column {column!r}.")
            if selected is not None:
                require_known(
                    selected,
                    set(observations[column].dropna().astype(str)),
                    label,
                )
        if self._selected_perturbation_frame().empty:
            raise ValueError("PerturbSeqSelection selects no perturbations.")
        if self._selected_series_frame().empty:
            raise ValueError("PerturbSeqSelection selects no population series.")
        if self._selected_observation_frame().empty:
            raise ValueError("PerturbSeqSelection selects no snapshot observations.")

    @property
    def representation(self) -> RepresentationSpec:
        return self.study.representations[self.representation_id]

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_series_frame()["series_id"].astype(str))

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        selected = self.selection.checkpoint_ids
        return selected if selected is not None else self.study.design.checkpoint_ids

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_observation_frame()["observation_id"].astype(str))

    @property
    def perturbation_ids(self) -> tuple[str, ...]:
        selected = set(self._selected_series_frame()["perturbation_id"].astype(str))
        order = self.study.perturbations.perturbation_ids
        return tuple(value for value in order if value in selected)

    def perturbations(self) -> pd.DataFrame:
        frame = self._selected_perturbation_frame()
        return frame.loc[frame["perturbation_id"].isin(self.perturbation_ids)].copy()

    def series(self) -> pd.DataFrame:
        return self._selected_series_frame().copy()

    def observations(self) -> pd.DataFrame:
        return self._selected_observation_frame().copy()

    def effect_binding(self) -> pd.DataFrame:
        binding_id = self.selection.effect_binding_id
        if binding_id is None or self.study.effect_bindings is None:
            return pd.DataFrame()
        frame = self.study.effect_bindings._unsafe_view()
        return frame.loc[
            frame["binding_id"].eq(binding_id)
            & frame["perturbation_id"].isin(self.perturbation_ids)
        ].copy()

    def reference_binding(self) -> pd.DataFrame:
        binding_id = self.selection.reference_binding_id
        if binding_id is None or self.study.reference_bindings is None:
            return pd.DataFrame()
        frame = self.study.reference_bindings._unsafe_view()
        return frame.loc[
            frame["binding_id"].eq(binding_id)
            & frame["perturbation_id"].isin(self.perturbation_ids)
        ].copy()

    def abundance(self) -> pd.DataFrame:
        if self.study.abundance is None or self.abundance_channel is None:
            return pd.DataFrame(columns=("observation_id", "channel_id", "value", "observed"))
        frame = self.study.abundance._unsafe_view()
        return frame.loc[
            frame["observation_id"].isin(self.observation_ids)
            & frame["channel_id"].eq(self.abundance_channel)
        ].copy()

    def compositions(self) -> pd.DataFrame:
        if self.study.compositions is None or self.selection.composition_policy == "drop":
            columns = () if self.study.compositions is None else self.study.compositions.columns
            return pd.DataFrame(columns=columns)
        selected_ids = set(self.observation_ids)
        frame = self.study.compositions._unsafe_view()
        selected = frame.loc[frame["observation_id"].isin(selected_ids)]
        touched = set(selected["composition_block_id"])
        full = frame.loc[frame["composition_block_id"].isin(touched)]
        policy = self.selection.composition_policy
        if policy == "preserve_background":
            return full.copy()
        if policy == "condition_on_selection":
            return selected.copy()
        if set(full["observation_id"]) != set(selected["observation_id"]):
            missing = sorted(set(full["observation_id"]) - set(selected["observation_id"]))[:5]
            raise ValueError(
                "Selection cuts through a composition block under require_complete; "
                f"unselected observations={missing}."
            )
        return selected.copy()

    def ecological_pools(self) -> pd.DataFrame:
        """Return only pools that constitute evidence of physical population interaction."""
        if self.study.population_pools is None:
            return pd.DataFrame()
        frame = self.study.population_pools._unsafe_view()
        return frame.loc[
            frame["population_pool_id"].isin(
                self.observations().get("population_pool_id", pd.Series(dtype=str)).dropna()
            )
            & frame["pool_kind"].isin(
                {"shared_living_culture", "shared_tissue", "competition_pool"}
            )
        ].copy()

    def snapshot(self, observation_id: str) -> MeasureSnapshot:
        if str(observation_id) not in set(self.observation_ids):
            raise KeyError(f"Observation {observation_id!r} is outside this PerturbSeqView.")
        return self.study.snapshot(
            observation_id,
            representation_id=self.representation_id,
            abundance_channel=self.abundance_channel,
        )

    def iter_snapshots(self) -> Iterator[MeasureSnapshot]:
        for observation_id in self.observation_ids:
            yield self.snapshot(observation_id)

    def validate_for(self, requirements: Any, split: Any) -> ValidationReport:
        from ..runtime import validate_view_for_recipe

        return validate_view_for_recipe(self, split, requirements)

    def semantic_hash(self) -> str:
        return _semantic_digest(
            {
                "study_content_hash": self.study.content_hash(),
                "perturbation_ids": self.perturbation_ids,
                "series_ids": self.series_ids,
                "checkpoint_ids": self.checkpoint_ids,
                "observation_ids": self.observation_ids,
                "representation_id": self.representation_id,
                "abundance_channel": self.abundance_channel,
                "selection": _canonical_value(self.selection),
            }
        )

    def _selected_perturbation_frame(self) -> pd.DataFrame:
        frame = _apply_filter(
            self.study.perturbations._unsafe_view(), self.selection.perturbation_filter
        )
        if self.selection.condition_filter:
            frame = _apply_filter(frame, self.selection.condition_filter)
        if self.selection.perturbation_ids is not None:
            frame = frame.loc[frame["perturbation_id"].isin(self.selection.perturbation_ids)]
        if self.selection.control_kinds is not None:
            frame = frame.loc[frame["control_kind"].isin(self.selection.control_kinds)]
        components = self.study.perturbation_components
        if self.selection.construct_ids is not None or self.selection.target_ids is not None:
            if components is None:
                raise KeyError("Construct/target selection requires perturbation components.")
            component_frame = components._unsafe_view()
            if self.selection.construct_ids is not None:
                component_frame = component_frame.loc[
                    component_frame["construct_id"].isin(self.selection.construct_ids)
                ]
            if self.selection.target_ids is not None:
                component_frame = component_frame.loc[
                    component_frame["target_id"].isin(self.selection.target_ids)
                ]
            frame = frame.loc[frame["perturbation_id"].isin(component_frame["perturbation_id"])]
        return frame

    def _selected_series_frame(self) -> pd.DataFrame:
        perturbation_ids = set(self._selected_perturbation_frame()["perturbation_id"])
        frame = self.study.series._unsafe_view()
        frame = frame.loc[frame["perturbation_id"].isin(perturbation_ids)]
        filters = {
            "series_id": self.selection.series_ids,
            "subject_id": self.selection.subject_ids,
            "experimental_unit_id": self.selection.experimental_unit_ids,
        }
        for column, values in filters.items():
            if values is not None:
                frame = frame.loc[frame[column].isin(values)]
        observation_series = set(
            self._filter_observation_frame(self.study.observations._unsafe_view())["series_id"]
        )
        frame = frame.loc[frame["series_id"].isin(observation_series)]
        return frame

    def _selected_observation_frame(self) -> pd.DataFrame:
        frame = self.study.observations._unsafe_view()
        frame = frame.loc[frame["series_id"].isin(self.series_ids)]
        return self._filter_observation_frame(frame)

    def _filter_observation_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        filters = {
            "observation_id": self.selection.observation_ids,
            "checkpoint_id": self.selection.checkpoint_ids,
            "context_id": self.selection.context_ids,
            "assignment_qc": self.selection.qc_tiers,
        }
        for column, values in filters.items():
            if values is not None:
                if column not in frame:
                    raise KeyError(f"Selection requires missing observation column {column!r}.")
                frame = frame.loc[frame[column].isin(values)]
        frame = _apply_filter(frame, self.selection.observation_filter)
        policy = self.selection.replicate_policy
        if policy.mode == "select":
            if "technical_replicate_id" not in frame:
                raise ValueError(
                    "Replicate selection requires observations.technical_replicate_id."
                )
            frame = frame.loc[frame["technical_replicate_id"].astype(str).eq(policy.selection_key)]
        return frame


def _condition_kind(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
    aliases = {
        "crisprko": "crispr_ko",
        "knockout": "crispr_ko",
        "crispri": "crispr_i",
        "crispr_crispri": "crispr_i",
        "crispra": "crispr_a",
        "drug": "chemical",
        "control": "other",
        "intervention": "other",
    }
    normalized = aliases.get(normalized, normalized)
    return (
        normalized
        if normalized
        in {
            "crispr_ko",
            "crispr_i",
            "crispr_a",
            "base_edit",
            "prime_edit",
            "chemical",
            "cytokine",
            "receptor_blockade",
            "combination",
        }
        else "other"
    )


def from_schema_v3(study: SchemaV3Study) -> PerturbSeqStudy:
    """Convert schema-v3 vocabulary conservatively, recording every unknown semantic."""
    if not isinstance(study, SchemaV3Study):
        raise TypeError("from_schema_v3 requires a schema-v3 Study.")
    conditions = study.conditions.to_pandas()
    perturbation_rows = []
    component_rows = []
    for row in conditions.itertuples(index=False):
        values = row._asdict()
        is_control = bool(values["is_reference"])
        control_kind = values.get("control_kind")
        if pd.isna(control_kind) if control_kind is not None else True:
            control_kind = "other" if is_control else "none"
        perturbation_rows.append(
            {
                **{
                    key: value
                    for key, value in values.items()
                    if key
                    not in {
                        "condition_id",
                        "condition_kind",
                        "is_reference",
                        "guide_id",
                        "target_gene",
                    }
                },
                "perturbation_id": str(values["condition_id"]),
                "perturbation_kind": _condition_kind(values["condition_kind"]),
                "is_control": is_control,
                "control_kind": str(control_kind),
            }
        )
        guide = values.get("guide_id")
        target = values.get("target_gene")
        if (guide is not None and pd.notna(guide)) or (target is not None and pd.notna(target)):
            perturbation_id = str(values["condition_id"])
            construct_id = str(guide) if guide is not None and pd.notna(guide) else perturbation_id
            target_id = str(target) if target is not None and pd.notna(target) else perturbation_id
            component_rows.append(
                {
                    "perturbation_id": perturbation_id,
                    "component_id": construct_id,
                    "construct_id": construct_id,
                    "target_id": target_id,
                    "component_kind": (
                        "guide" if guide is not None and pd.notna(guide) else "reagent"
                    ),
                    "dose": np.nan,
                    "dose_unit": None,
                    "component_order": 0,
                }
            )
    perturbations = PerturbationTable(pd.DataFrame(perturbation_rows))
    components = (
        PerturbationComponentTable(pd.DataFrame(component_rows)) if component_rows else None
    )

    old_series = study.series.to_pandas()
    series_rows = []
    for row in old_series.itertuples(index=False):
        values = row._asdict()
        subject_id = str(values["subject_id"])
        series_rows.append(
            {
                **{
                    key: value
                    for key, value in values.items()
                    if key
                    not in {
                        "condition_id",
                        "embedding_id",
                        "reference_role",
                        "experimental_unit_id",
                        "context_trajectory_id",
                        "biological_replicate_id",
                        "continuity_kind",
                    }
                },
                "series_id": str(values["series_id"]),
                "subject_id": subject_id,
                "experimental_unit_id": str(values.get("experimental_unit_id") or subject_id),
                "perturbation_id": str(values["condition_id"]),
                "context_trajectory_id": str(values.get("context_trajectory_id") or "unknown"),
                "biological_replicate_id": str(
                    values.get("biological_replicate_id") or values["series_id"]
                ),
                "continuity_kind": str(values.get("continuity_kind") or "unknown"),
            }
        )
    population_series = PopulationSeriesTable(pd.DataFrame(series_rows))
    events = InterventionEventTable(
        pd.DataFrame(
            [
                {
                    "event_id": f"primary::{row.series_id}",
                    "series_id": str(row.series_id),
                    "agent_id": str(row.perturbation_id),
                    "event_kind": "experimentally_assigned_perturbation",
                    "modeled_role": "primary_perturbation",
                    "start_coordinate": np.nan,
                    "end_coordinate": np.nan,
                    "start_relation": "unknown",
                    "persistent": True,
                    "dose": np.nan,
                    "dose_unit": None,
                }
                for row in population_series._unsafe_view().itertuples(index=False)
            ]
        )
    )

    old_observations = study.observations.to_pandas()
    abundance_ids: set[str] = set()
    if study.abundance is not None:
        abundance_frame = study.abundance._unsafe_view()
        abundance_ids = set(
            abundance_frame.loc[abundance_frame["observed"], "observation_id"].astype(str)
        )
    observation_rows = []
    for row in old_observations.itertuples(index=False):
        values = row._asdict()
        observation_id = str(values["observation_id"])
        observation_rows.append(
            {
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"replicate_id", "geometry_observed"}
                },
                "observation_id": observation_id,
                "geometry_observed": bool(values["geometry_observed"]),
                "abundance_observed": observation_id in abundance_ids,
                "technical_replicate_id": values.get("replicate_id"),
            }
        )
    observations = SnapshotObservationTable(pd.DataFrame(observation_rows))
    context_frame = None
    if "context_id" in observations.columns:
        context_ids = observations._unsafe_view()["context_id"].dropna().astype(str).unique()
        if len(context_ids):
            context_frame = ContextTable(
                pd.DataFrame({"context_id": context_ids, "context_kind": "legacy_unspecified"})
            )

    compositions = None
    if study.compositions is not None:
        frame = study.compositions.to_pandas()
        if "block_kind" not in frame:
            frame["block_kind"] = "sampling_stratum"
        compositions = LPSCompositionTable(frame)
    effects = None
    if study.effect_bindings is not None:
        frame = study.effect_bindings.to_pandas().rename(
            columns={"condition_id": "perturbation_id"}
        )
        effects = PerturbationEffectBindingTable(frame)
    references = None
    if study.reference_bindings is not None:
        frame = study.reference_bindings.to_pandas().rename(
            columns={"condition_id": "perturbation_id"}
        )
        scope_aliases = {
            "global_condition_pool": "global",
            "subject_condition_pool": "subject",
            "context_condition_pool": "context",
        }
        frame["scope_kind"] = frame["scope_kind"].replace(scope_aliases)
        scope_match_keys = {
            "global": (),
            "subject": ("subject_id",),
            "experimental_unit": ("experimental_unit_id",),
            "context": ("context_id",),
            "checkpoint": ("checkpoint_id",),
            "processing_batch": ("processing_batch_id",),
        }
        frame["match_keys"] = frame["scope_kind"].map(
            lambda value: json.dumps(scope_match_keys[str(value)], separators=(",", ":"))
        )
        if "counterfactual_effect_id" not in frame:
            effect_lookup = (
                {}
                if effects is None
                else effects._unsafe_view()
                .drop_duplicates("perturbation_id")
                .set_index("perturbation_id")["effect_id"]
                .astype(str)
                .to_dict()
            )
            control_ids = set(
                perturbations._unsafe_view().loc[
                    perturbations._unsafe_view()["is_control"], "perturbation_id"
                ]
            )
            control_rows = frame.loc[frame["perturbation_id"].isin(control_ids)]
            reference_effect_by_pool = {
                str(pool_id): effect_lookup.get(str(rows.iloc[0]["perturbation_id"]), "reference")
                for pool_id, rows in control_rows.groupby("reference_pool_id", observed=True)
            }
            frame["counterfactual_effect_id"] = (
                frame["reference_pool_id"].map(reference_effect_by_pool).fillna("reference")
            )
        references = PerturbationReferenceBindingTable(frame)
    provenance = dict(study.provenance)
    provenance["schema_v3_conversion"] = {
        "source_study_hash": study.content_hash(),
        "continuity_kind": "unknown unless explicitly present",
        "intervention_start_relation": "unknown unless explicitly present",
        "composition_block_kind_default": "sampling_stratum",
    }
    manifest = PerturbSeqManifest(
        schema_version=4,
        study_id=study.manifest.study_id,
        source_schema=study.manifest.source_schema,
        primary_representation=study.manifest.primary_representation,
        primary_abundance_channel=study.manifest.primary_abundance_channel,
        description=study.manifest.description,
        primary_effect_binding=study.manifest.primary_effect_binding,
        primary_reference_binding=study.manifest.primary_reference_binding,
    )
    return PerturbSeqStudy(
        manifest=manifest,
        design=LongitudinalDesign.from_study_design(study.design),
        perturbations=perturbations,
        perturbation_components=components,
        intervention_events=events,
        contexts=context_frame,
        series=population_series,
        observations=observations,
        representations=study.representations,
        support_index=study.support_index,
        supports=study.supports,
        abundance=study.abundance,
        compositions=compositions,
        population_pools=None,
        effect_bindings=effects,
        reference_bindings=references,
        provenance=provenance,
    )


__all__ = [
    "PerturbSeqManifest",
    "PerturbSeqSelection",
    "PerturbSeqStudy",
    "PerturbSeqView",
    "from_schema_v3",
]
