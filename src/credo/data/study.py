"""Storage-independent Study and StudyView objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from .design import StudyDesign
from .representations import RepresentationCatalog, RepresentationSpec
from .support import (
    AbundanceValue,
    MeasureSnapshot,
    SupportRef,
    SupportStore,
    SupportStoreRegistry,
)
from .tables import (
    AbundanceTable,
    CompositionTable,
    ConditionTable,
    ObservationTable,
    SeriesTable,
    SupportIndexTable,
)
from .validation import ValidationIssue, ValidationReport

VerificationLevel = Literal["none", "schema", "manifest", "semantic", "full"]
CompositionPolicy = Literal[
    "require_complete",
    "preserve_background",
    "condition_on_selection",
    "drop",
]


@dataclass(frozen=True)
class StudyManifest:
    """Identity and authoritative defaults for one semantic study."""

    schema_version: int
    study_id: str
    source_schema: str
    primary_representation: str
    primary_abundance_channel: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if int(self.schema_version) < 1:
            raise ValueError("StudyManifest.schema_version must be positive.")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        for name in ("study_id", "source_schema", "primary_representation"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"StudyManifest.{name} must be nonempty.")
            object.__setattr__(self, name, value)
        if self.primary_abundance_channel is not None:
            value = str(self.primary_abundance_channel)
            if not value:
                raise ValueError(
                    "StudyManifest.primary_abundance_channel must be nonempty when provided."
                )
            object.__setattr__(self, "primary_abundance_channel", value)
        object.__setattr__(self, "description", str(self.description))


@dataclass(frozen=True)
class SelectionSpec:
    """Stable-ID and metadata filters for a zero-copy study view."""

    series_ids: tuple[str, ...] | None = None
    checkpoint_ids: tuple[str, ...] | None = None
    condition_filter: Mapping[str, Any] | None = None
    observation_filter: Mapping[str, Any] | None = None
    composition_policy: CompositionPolicy = "require_complete"

    def __post_init__(self) -> None:
        for name in ("series_ids", "checkpoint_ids"):
            values = getattr(self, name)
            if values is None:
                continue
            normalized = tuple(str(value) for value in values)
            if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
                raise ValueError(f"SelectionSpec.{name} must contain unique nonempty IDs.")
            object.__setattr__(self, name, normalized)
        for name in ("condition_filter", "observation_filter"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, MappingProxyType(dict(values)))
        if self.composition_policy not in {
            "require_complete",
            "preserve_background",
            "condition_on_selection",
            "drop",
        }:
            raise ValueError(f"Unknown composition policy {self.composition_policy!r}.")


def _apply_filter(frame: pd.DataFrame, filters: Mapping[str, Any] | None) -> pd.DataFrame:
    if not filters:
        return frame
    keep = np.ones(len(frame), dtype=bool)
    for column, expected in filters.items():
        if column not in frame:
            raise KeyError(f"Unknown filter column {column!r}.")
        if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            keep &= frame[column].isin(list(expected)).to_numpy()
        else:
            keep &= frame[column].eq(expected).to_numpy()
    return frame.loc[keep]


def _optional_value(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    value = str(row[column])
    return value or None


@dataclass(frozen=True)
class Study:
    """Biological study semantics independent of table and array storage."""

    manifest: StudyManifest
    design: StudyDesign
    conditions: ConditionTable
    series: SeriesTable
    observations: ObservationTable
    support_index: SupportIndexTable
    abundance: AbundanceTable | None
    compositions: CompositionTable | None
    representations: RepresentationCatalog
    supports: SupportStoreRegistry | SupportStore
    provenance: Mapping[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        supports = self.supports
        if isinstance(supports, SupportStore):
            supports = SupportStoreRegistry((supports,))
        if not isinstance(supports, SupportStoreRegistry):
            raise TypeError("Study.supports must be a SupportStoreRegistry or SupportStore.")
        object.__setattr__(self, "supports", supports)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def validate(self, level: VerificationLevel = "semantic") -> ValidationReport:
        """Validate the requested contract level without mutating the study."""
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
        for representation in self.representations.values():
            store_id = representation.support_store_id
            if store_id not in self.supports:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.store_missing",
                        f"Representation {representation.representation_id!r} references "
                        f"unknown store {store_id!r}.",
                        ("representations", representation.representation_id, "support_store_id"),
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
            try:
                dimension = store.dimension(representation.representation_id)
            except KeyError:
                dimension = None
            if dimension is not None and dimension != representation.dimension:
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

        conditions = self.conditions._unsafe_view()
        series = self.series._unsafe_view()
        observations = self.observations._unsafe_view()
        support_index = self.support_index._unsafe_view()
        condition_ids = set(conditions["condition_id"])
        series_ids = set(series["series_id"])
        observation_ids = set(observations["observation_id"])
        checkpoint_ids = set(self.design.checkpoint_ids)

        unknown_conditions = set(series["condition_id"]) - condition_ids
        if unknown_conditions:
            issues.append(
                ValidationIssue(
                    "error",
                    "series.condition_fk",
                    f"Series reference unknown conditions: {sorted(unknown_conditions)[:5]}.",
                    ("series", "condition_id"),
                )
            )
        condition_lookup = conditions.set_index("condition_id")
        for row in series.itertuples(index=False):
            if row.condition_id not in condition_lookup.index:
                continue
            condition = condition_lookup.loc[row.condition_id]
            if str(row.embedding_id) != str(condition["embedding_id"]):
                issues.append(
                    ValidationIssue(
                        "error",
                        "series.embedding",
                        f"Series {row.series_id!r} embedding disagrees with its condition.",
                        ("series", row.series_id, "embedding_id"),
                    )
                )
            expected_role = "reference" if bool(condition["is_reference"]) else "intervention"
            if str(row.reference_role) != expected_role:
                issues.append(
                    ValidationIssue(
                        "error",
                        "series.reference_role",
                        f"Series {row.series_id!r} has reference_role {row.reference_role!r}; "
                        f"expected {expected_role!r}.",
                        ("series", row.series_id, "reference_role"),
                    )
                )
        reference_groups = set(
            conditions.loc[conditions["is_reference"], "reference_group_id"].astype(str)
        )
        unresolved_groups = (
            set(conditions.loc[~conditions["is_reference"], "reference_group_id"].astype(str))
            - reference_groups
        )
        if unresolved_groups:
            issues.append(
                ValidationIssue(
                    "warning",
                    "condition.reference_unresolved",
                    "Intervention conditions have no in-study reference member for groups "
                    f"{sorted(unresolved_groups)[:5]}; recipes may reject this view.",
                    ("conditions", "reference_group_id"),
                )
            )

        unknown_series = set(observations["series_id"]) - series_ids
        unknown_checkpoints = set(observations["checkpoint_id"]) - checkpoint_ids
        if unknown_series:
            issues.append(
                ValidationIssue(
                    "error",
                    "observation.series_fk",
                    f"Observations reference unknown series: {sorted(unknown_series)[:5]}.",
                    ("observations", "series_id"),
                )
            )
        if unknown_checkpoints:
            issues.append(
                ValidationIssue(
                    "error",
                    "observation.checkpoint_fk",
                    "Observations reference unknown checkpoints: "
                    f"{sorted(unknown_checkpoints)[:5]}.",
                    ("observations", "checkpoint_id"),
                )
            )

        invalid_index_observations = set(support_index["observation_id"]) - observation_ids
        invalid_index_representations = set(support_index["representation_id"]) - representation_ids
        if invalid_index_observations or invalid_index_representations:
            issues.append(
                ValidationIssue(
                    "error",
                    "support_index.foreign_key",
                    "Support index has unknown identities: "
                    f"observations={sorted(invalid_index_observations)[:5]}, "
                    f"representations={sorted(invalid_index_representations)[:5]}.",
                    ("support_index",),
                )
            )
        expected_pairs = pd.MultiIndex.from_product(
            [observations["observation_id"], tuple(self.representations)],
            names=["observation_id", "representation_id"],
        )
        actual_pairs = pd.MultiIndex.from_frame(
            support_index[["observation_id", "representation_id"]]
        )
        missing_pairs = expected_pairs.difference(actual_pairs)
        extra_pairs = actual_pairs.difference(expected_pairs)
        if len(missing_pairs) or len(extra_pairs):
            issues.append(
                ValidationIssue(
                    "error",
                    "support_index.coverage",
                    "Support index must declare every observation/representation pair; "
                    f"missing={missing_pairs.tolist()[:5]}, extra={extra_pairs.tolist()[:5]}.",
                    ("support_index",),
                )
            )
        geometry_lookup = observations.set_index("observation_id")["geometry_observed"]
        for row in support_index.itertuples(index=False):
            if row.observation_id not in geometry_lookup.index:
                continue
            if bool(row.available) and not bool(geometry_lookup.loc[row.observation_id]):
                issues.append(
                    ValidationIssue(
                        "error",
                        "support_index.biological_geometry",
                        f"Observation {row.observation_id!r} is biologically geometry-missing "
                        "but declares representation support.",
                        ("support_index", row.observation_id, row.representation_id),
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
                        f"Support for representation {row.representation_id!r} uses store "
                        f"{row.store_id!r}; expected {representation.support_store_id!r}.",
                        ("support_index", row.observation_id, row.representation_id, "store_id"),
                    )
                )
                continue
            ref = SupportRef(row.store_id, row.representation_id, row.support_key)
            if not self.supports.contains(ref):
                issues.append(
                    ValidationIssue(
                        "error",
                        "support.reference_fk",
                        f"Observation {row.observation_id!r} references missing support for "
                        f"representation {row.representation_id!r}.",
                        ("support_index", row.observation_id, row.representation_id),
                    )
                )

        for representation in self.representations.values():
            invalid_series = set(representation.included_series) - series_ids
            invalid_checkpoints = set(representation.included_checkpoints) - checkpoint_ids
            if invalid_series or invalid_checkpoints:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.scope_fk",
                        f"Representation {representation.representation_id!r} has unknown scope "
                        f"series={sorted(invalid_series)[:5]}, "
                        f"checkpoints={sorted(invalid_checkpoints)[:5]}.",
                        ("representations", representation.representation_id),
                    )
                )

        if self.abundance is not None:
            abundance = self.abundance._unsafe_view()
            unknown_abundance = set(abundance["observation_id"]) - observation_ids
            if unknown_abundance:
                issues.append(
                    ValidationIssue(
                        "error",
                        "abundance.observation_fk",
                        "Abundance references unknown observations: "
                        f"{sorted(unknown_abundance)[:5]}.",
                        ("abundance", "observation_id"),
                    )
                )
            issues.extend(self._validate_abundance_denominators(abundance, observations))

        if self.compositions is not None:
            issues.extend(self._validate_compositions(observations, series_ids, checkpoint_ids))

        return ValidationReport(tuple(issues)).merged(
            self.supports.validate(full_scan=level == "full")
        )

    def _validate_abundance_denominators(
        self,
        abundance: pd.DataFrame,
        observations: pd.DataFrame,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        observation_columns = ["observation_id", "sample_id", "checkpoint_id"]
        if "context_id" in observations:
            observation_columns.append("context_id")
        scoped = abundance.merge(
            observations[observation_columns],
            on="observation_id",
            how="left",
            validate="many_to_one",
        )
        for channel_id, spec in self.abundance.channels.items():  # type: ignore[union-attr]
            if spec.denominator_scope in {"none", "custom"}:
                continue
            rows = scoped.loc[scoped["channel_id"].eq(channel_id) & scoped["observed"]]
            scope_columns = (
                ["context_id", "checkpoint_id"]
                if spec.denominator_scope == "context_checkpoint"
                else ["sample_id", "checkpoint_id"]
            )
            if any(column not in rows for column in scope_columns):
                issues.append(
                    ValidationIssue(
                        "error",
                        "abundance.denominator_scope",
                        f"Channel {channel_id!r} lacks metadata for denominator scope "
                        f"{spec.denominator_scope!r}.",
                        ("abundance", channel_id, "denominator_id"),
                    )
                )
                continue
            denominator_scope = rows.groupby("denominator_id", observed=True)[
                scope_columns
            ].nunique()
            invalid_denominators = denominator_scope.ne(1).any(axis=1)
            scope_denominators = rows.groupby(scope_columns, observed=True)[
                "denominator_id"
            ].nunique()
            if invalid_denominators.any() or scope_denominators.ne(1).any():
                issues.append(
                    ValidationIssue(
                        "error",
                        "abundance.denominator_scope",
                        f"Channel {channel_id!r} denominator IDs are not one-to-one with "
                        f"{spec.denominator_scope!r} scopes.",
                        ("abundance", channel_id, "denominator_id"),
                    )
                )
        return issues

    def _validate_compositions(
        self,
        observations: pd.DataFrame,
        series_ids: set[str],
        checkpoint_ids: set[str],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        compositions = self.compositions._unsafe_view()  # type: ignore[union-attr]
        observation_ids = set(observations["observation_id"])
        unknown_series = set(compositions["series_id"]) - series_ids
        unknown_checkpoints = set(compositions["checkpoint_id"]) - checkpoint_ids
        unknown_observations = set(compositions["observation_id"]) - observation_ids
        if unknown_series or unknown_checkpoints or unknown_observations:
            issues.append(
                ValidationIssue(
                    "error",
                    "composition.foreign_key",
                    "Composition rows have unknown identities: "
                    f"series={sorted(unknown_series)[:5]}, "
                    f"checkpoints={sorted(unknown_checkpoints)[:5]}, "
                    f"observations={sorted(unknown_observations)[:5]}.",
                    ("compositions",),
                )
            )
        observation_lookup = observations.set_index("observation_id")
        for row in compositions.itertuples(index=False):
            if row.observation_id not in observation_lookup.index:
                continue
            observation = observation_lookup.loc[row.observation_id]
            if (row.series_id, row.checkpoint_id) != (
                observation["series_id"],
                observation["checkpoint_id"],
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "composition.observation_alignment",
                        f"Composition observation {row.observation_id!r} is misaligned.",
                        ("compositions", row.composition_block_id, row.observation_id),
                    )
                )
            declared_block = _optional_value(observation, "composition_block_id")
            if declared_block is not None and row.composition_block_id != declared_block:
                issues.append(
                    ValidationIssue(
                        "error",
                        "composition.block_alignment",
                        f"Composition observation {row.observation_id!r} declares block "
                        f"{row.composition_block_id!r}; expected {declared_block!r}.",
                        ("compositions", row.composition_block_id, row.observation_id),
                    )
                )
            context_id = _optional_value(observation, "context_id")
            if context_id is not None and row.context_id != context_id:
                issues.append(
                    ValidationIssue(
                        "error",
                        "composition.context_alignment",
                        f"Composition observation {row.observation_id!r} declares context "
                        f"{row.context_id!r}; expected {context_id!r}.",
                        ("compositions", row.composition_block_id, row.observation_id),
                    )
                )
        if "composition_block_id" in observations:
            declared = observations.loc[observations["composition_block_id"].notna()]
            expected_members = {
                str(block_id): set(rows["observation_id"])
                for block_id, rows in declared.groupby(
                    "composition_block_id", observed=True, sort=False
                )
            }
            actual_members = {
                str(block_id): set(rows["observation_id"])
                for block_id, rows in compositions.groupby(
                    "composition_block_id", observed=True, sort=False
                )
            }
            for block_id in sorted(set(expected_members) | set(actual_members)):
                expected = expected_members.get(block_id, set())
                actual = actual_members.get(block_id, set())
                if expected != actual:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "composition.membership",
                            f"Composition block {block_id!r} membership differs; "
                            f"missing={sorted(expected - actual)[:5]}, "
                            f"extra={sorted(actual - expected)[:5]}.",
                            ("compositions", block_id),
                        )
                    )
        return issues

    def snapshot(
        self,
        observation_id: str,
        *,
        representation_id: str | None = None,
        abundance_channel: str | None = None,
    ) -> MeasureSnapshot:
        """Read one empirical law and one selected abundance value."""
        if self._closed:
            raise RuntimeError("Study is closed.")
        observation_id = str(observation_id)
        observations = self.observations._unsafe_view()
        rows = observations.loc[observations["observation_id"].eq(observation_id)]
        if len(rows) != 1:
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
        support_row = support_rows.iloc[0]
        law = None
        if bool(support_row["available"]):
            law = self.supports.read(
                SupportRef(
                    str(support_row["store_id"]),
                    selected_representation,
                    str(support_row["support_key"]),
                )
            )
        selected_channel = abundance_channel or self.manifest.primary_abundance_channel
        abundance_value = None
        if selected_channel is not None:
            if self.abundance is None or selected_channel not in self.abundance.channels:
                raise KeyError(f"Unknown abundance channel {selected_channel!r}.")
            values = self.abundance._unsafe_view()
            matches = values.loc[
                values["observation_id"].eq(observation_id)
                & values["channel_id"].eq(selected_channel)
            ]
            if len(matches) == 1:
                value = matches.iloc[0]
                raw_value = value["value"]
                abundance_value = AbundanceValue(
                    observation_id=observation_id,
                    channel_id=selected_channel,
                    value=None if pd.isna(raw_value) else float(raw_value),
                    observed=bool(value["observed"]),
                    denominator_id=_optional_value(value, "denominator_id"),
                    transform_id=_optional_value(value, "transform_id"),
                    source_artifact_id=_optional_value(value, "source_artifact_id"),
                )
        return MeasureSnapshot(observation_id, law, abundance_value)

    def view(
        self,
        selection: SelectionSpec | None = None,
        *,
        representation_id: str | None = None,
        abundance_channel: str | None = None,
    ) -> StudyView:
        return StudyView(
            study=self,
            selection=selection or SelectionSpec(),
            representation_id=representation_id or self.manifest.primary_representation,
            abundance_channel=(
                abundance_channel
                if abundance_channel is not None
                else self.manifest.primary_abundance_channel
            ),
        )

    def close(self) -> None:
        if not self._closed:
            self.supports.close()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> Study:
        if self._closed:
            raise RuntimeError("Study is closed.")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(frozen=True)
class StudyView:
    """Selection and representation binding over a shared Study."""

    study: Study
    selection: SelectionSpec
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
        known_series = set(self.study.series.series_ids)
        known_checkpoints = set(self.study.design.checkpoint_ids)
        if self.selection.series_ids is not None:
            unknown = set(self.selection.series_ids) - known_series
            if unknown:
                raise KeyError(f"Selection references unknown series: {sorted(unknown)[:5]}.")
        if self.selection.checkpoint_ids is not None:
            unknown = set(self.selection.checkpoint_ids) - known_checkpoints
            if unknown:
                raise KeyError(f"Selection references unknown checkpoints: {sorted(unknown)[:5]}.")
        self._selected_series_frame()
        self._selected_observation_frame()

    @property
    def representation(self) -> RepresentationSpec:
        return self.study.representations[self.representation_id]

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_series_frame()["series_id"].tolist())

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        selected = self.selection.checkpoint_ids
        return selected if selected is not None else self.study.design.checkpoint_ids

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(self._selected_observation_frame()["observation_id"].tolist())

    def observations(self) -> pd.DataFrame:
        return self._selected_observation_frame().copy()

    def abundance(self) -> pd.DataFrame:
        if self.study.abundance is None or self.abundance_channel is None:
            return pd.DataFrame(columns=("observation_id", "channel_id", "value", "observed"))
        selected = set(self.observation_ids)
        frame = self.study.abundance._unsafe_view()
        return frame.loc[
            frame["observation_id"].isin(selected) & frame["channel_id"].eq(self.abundance_channel)
        ].copy()

    def compositions(self) -> pd.DataFrame:
        if self.study.compositions is None or self.selection.composition_policy == "drop":
            columns = () if self.study.compositions is None else self.study.compositions.columns
            return pd.DataFrame(columns=columns)
        selected_observations = set(self.observation_ids)
        frame = self.study.compositions._unsafe_view()
        selected_rows = frame.loc[frame["observation_id"].isin(selected_observations)]
        touched_blocks = set(selected_rows["composition_block_id"])
        if not touched_blocks:
            return selected_rows.copy()
        full_rows = frame.loc[frame["composition_block_id"].isin(touched_blocks)]
        policy = self.selection.composition_policy
        if policy == "preserve_background":
            return full_rows.copy()
        if policy == "condition_on_selection":
            return selected_rows.copy()
        full_members = set(full_rows["observation_id"])
        if full_members != set(selected_rows["observation_id"]):
            missing = sorted(full_members - set(selected_rows["observation_id"]))[:5]
            raise ValueError(
                "Selection cuts through a composition block under require_complete; "
                f"unselected observations={missing}."
            )
        return selected_rows.copy()

    def snapshot(self, observation_id: str) -> MeasureSnapshot:
        if str(observation_id) not in set(self.observation_ids):
            raise KeyError(f"Observation {observation_id!r} is outside this StudyView.")
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
        payload = {
            "study_id": self.study.manifest.study_id,
            "series_ids": self.series_ids,
            "observation_ids": self.observation_ids,
            "representation_id": self.representation_id,
            "abundance_channel": self.abundance_channel,
            "composition_policy": self.selection.composition_policy,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _selected_series_frame(self) -> pd.DataFrame:
        conditions = _apply_filter(
            self.study.conditions._unsafe_view(), self.selection.condition_filter
        )
        allowed_conditions = set(conditions["condition_id"])
        frame = self.study.series._unsafe_view()
        frame = frame.loc[frame["condition_id"].isin(allowed_conditions)]
        if self.selection.series_ids is not None:
            frame = frame.loc[frame["series_id"].isin(self.selection.series_ids)]
        return frame

    def _selected_observation_frame(self) -> pd.DataFrame:
        series_ids = set(self._selected_series_frame()["series_id"])
        frame = self.study.observations._unsafe_view()
        frame = frame.loc[frame["series_id"].isin(series_ids)]
        if self.selection.checkpoint_ids is not None:
            frame = frame.loc[frame["checkpoint_id"].isin(self.selection.checkpoint_ids)]
        return _apply_filter(frame, self.selection.observation_filter)


__all__ = [
    "CompositionPolicy",
    "SelectionSpec",
    "Study",
    "StudyManifest",
    "StudyView",
    "VerificationLevel",
]
