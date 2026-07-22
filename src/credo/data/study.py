"""Storage-independent Study and StudyView objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from .design import StudyDesign
from .representations import RepresentationCatalog, RepresentationSpec
from .support import AbundanceValue, MeasureSnapshot, SupportRef, SupportStore
from .tables import (
    AbundanceTable,
    CompositionTable,
    ConditionTable,
    ObservationTable,
    SeriesTable,
)
from .validation import ValidationIssue, ValidationReport


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


@dataclass
class Study:
    """Biological study semantics independent of table and array storage."""

    manifest: StudyManifest
    design: StudyDesign
    conditions: ConditionTable
    series: SeriesTable
    observations: ObservationTable
    abundance: AbundanceTable | None
    compositions: CompositionTable | None
    representations: RepresentationCatalog
    supports: SupportStore
    provenance: Mapping[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.supports, SupportStore):
            raise TypeError("Study.supports must implement the SupportStore protocol.")
        self.provenance = MappingProxyType(dict(self.provenance))
        self.validate(level="semantic").raise_for_errors()

    def validate(
        self,
        level: Literal["schema", "manifest", "semantic", "full"] = "semantic",
    ) -> ValidationReport:
        """Validate foreign keys and scientific invariants without changing storage."""
        if level not in {"schema", "manifest", "semantic", "full"}:
            raise ValueError(f"Unknown validation level {level!r}.")
        issues: list[ValidationIssue] = []
        representation_ids = set(self.representations)
        if self.manifest.primary_representation not in representation_ids:
            issues.append(
                ValidationIssue(
                    "error",
                    "manifest.primary_representation",
                    "Primary representation is absent from the representation catalog.",
                )
            )
        if self.manifest.primary_abundance_channel is not None:
            if self.abundance is None or (
                self.manifest.primary_abundance_channel not in self.abundance.channels
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "manifest.primary_abundance",
                        "Primary abundance channel is absent from the abundance catalog.",
                    )
                )
        store_representations = set(self.supports.representation_ids())
        missing_store_representations = representation_ids - store_representations
        if missing_store_representations:
            issues.append(
                ValidationIssue(
                    "error",
                    "representation.store_missing",
                    "Support store is missing representations: "
                    f"{sorted(missing_store_representations)}.",
                )
            )
        for representation in self.representations.values():
            if representation.support_store_id != self.supports.store_id:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.store_id",
                        f"Representation {representation.representation_id!r} declares store "
                        f"{representation.support_store_id!r}, expected "
                        f"{self.supports.store_id!r}.",
                    )
                )
            if representation.representation_id in store_representations:
                try:
                    dimension = self.supports.dimension(representation.representation_id)
                except KeyError:
                    dimension = None
                if dimension is not None and dimension != representation.dimension:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "representation.dimension",
                            f"Representation {representation.representation_id!r} declares "
                            f"dimension {representation.dimension}, store reports {dimension}.",
                        )
                    )
        if level in {"schema", "manifest"}:
            return ValidationReport(tuple(issues))

        conditions = self.conditions.to_pandas(copy=False)
        series = self.series.to_pandas(copy=False)
        observations = self.observations.to_pandas(copy=False)
        condition_ids = set(conditions["condition_id"])
        series_ids = set(series["series_id"])
        checkpoint_ids = set(self.design.checkpoint_ids)
        unknown_conditions = set(series["condition_id"]) - condition_ids
        if unknown_conditions:
            issues.append(
                ValidationIssue(
                    "error",
                    "series.condition_fk",
                    f"Series reference unknown conditions: {sorted(unknown_conditions)[:5]}.",
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
                )
            )
        if unknown_checkpoints:
            issues.append(
                ValidationIssue(
                    "error",
                    "observation.checkpoint_fk",
                    "Observations reference unknown checkpoints: "
                    f"{sorted(unknown_checkpoints)[:5]}.",
                )
            )
        observation_ids = set(observations["observation_id"])
        for representation in self.representations.values():
            included_series = set(representation.included_series)
            included_checkpoints = set(representation.included_checkpoints)
            invalid_series = included_series - series_ids
            invalid_checkpoints = included_checkpoints - checkpoint_ids
            if invalid_series or invalid_checkpoints:
                issues.append(
                    ValidationIssue(
                        "error",
                        "representation.scope_fk",
                        f"Representation {representation.representation_id!r} has unknown scope "
                        f"series={sorted(invalid_series)[:5]}, "
                        f"checkpoints={sorted(invalid_checkpoints)[:5]}.",
                    )
                )
            for row in observations.loc[observations["geometry_observed"]].itertuples(index=False):
                ref = SupportRef(representation.representation_id, row.support_key)
                if not self.supports.contains(ref):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "support.reference_fk",
                            f"Observation {row.observation_id!r} references missing support "
                            f"for representation {representation.representation_id!r}.",
                        )
                    )
        if self.abundance is not None:
            abundance = self.abundance.to_pandas(copy=False)
            unknown_observations = set(abundance["observation_id"]) - observation_ids
            if unknown_observations:
                issues.append(
                    ValidationIssue(
                        "error",
                        "abundance.observation_fk",
                        "Abundance references unknown observations: "
                        f"{sorted(unknown_observations)[:5]}.",
                    )
                )
        if self.compositions is not None:
            compositions = self.compositions.to_pandas(copy=False)
            unknown_composition_series = set(compositions["series_id"]) - series_ids
            unknown_composition_checkpoints = set(compositions["checkpoint_id"]) - checkpoint_ids
            if unknown_composition_series or unknown_composition_checkpoints:
                issues.append(
                    ValidationIssue(
                        "error",
                        "composition.foreign_key",
                        "Composition rows have unknown series/checkpoints: "
                        f"series={sorted(unknown_composition_series)[:5]}, "
                        f"checkpoints={sorted(unknown_composition_checkpoints)[:5]}.",
                    )
                )
            observation_by_pair = observations.set_index(["series_id", "checkpoint_id"])
            composition_pairs = pd.MultiIndex.from_frame(
                compositions[["series_id", "checkpoint_id"]]
            )
            missing_observation_pairs = composition_pairs.difference(observation_by_pair.index)
            if len(missing_observation_pairs):
                issues.append(
                    ValidationIssue(
                        "error",
                        "composition.observation_missing",
                        "Composition rows lack explicit observations: "
                        f"{missing_observation_pairs.tolist()[:5]}.",
                    )
                )
            if "observation_id" in compositions:
                unknown_composition_observations = (
                    set(compositions["observation_id"]) - observation_ids
                )
                if unknown_composition_observations:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "composition.observation_fk",
                            "Composition rows reference unknown observations: "
                            f"{sorted(unknown_composition_observations)[:5]}.",
                        )
                    )
                observation_pairs = observations.set_index("observation_id")[
                    ["series_id", "checkpoint_id"]
                ]
                for row in compositions.itertuples(index=False):
                    if row.observation_id not in observation_pairs.index:
                        continue
                    expected = observation_pairs.loc[row.observation_id]
                    if (row.series_id, row.checkpoint_id) != (
                        expected["series_id"],
                        expected["checkpoint_id"],
                    ):
                        issues.append(
                            ValidationIssue(
                                "error",
                                "composition.observation_alignment",
                                f"Composition observation {row.observation_id!r} is misaligned.",
                            )
                        )
            for row in compositions.itertuples(index=False):
                pair = (row.series_id, row.checkpoint_id)
                if pair not in observation_by_pair.index:
                    continue
                observation = observation_by_pair.loc[pair]
                declared_block = _optional_value(observation, "composition_block_id")
                if declared_block is not None and row.composition_block_id != declared_block:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "composition.block_alignment",
                            f"Composition row for {pair!r} declares block "
                            f"{row.composition_block_id!r}; expected {declared_block!r}.",
                        )
                    )
                context_id = _optional_value(observation, "context_id")
                if context_id is not None and row.context_id != context_id:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "composition.context_alignment",
                            f"Composition row for {pair!r} declares context {row.context_id!r}; "
                            f"expected {context_id!r}.",
                        )
                    )
            if "composition_block_id" in observations:
                declared = observations.loc[observations["composition_block_id"].notna()]
                declared_members = {
                    str(block_id): set(zip(rows["series_id"], rows["checkpoint_id"], strict=False))
                    for block_id, rows in declared.groupby(
                        "composition_block_id", observed=True, sort=False
                    )
                }
                observed_members = {
                    str(block_id): set(zip(rows["series_id"], rows["checkpoint_id"], strict=False))
                    for block_id, rows in compositions.groupby(
                        "composition_block_id", observed=True, sort=False
                    )
                }
                for block_id in sorted(set(declared_members) | set(observed_members)):
                    expected = declared_members.get(block_id, set())
                    actual = observed_members.get(block_id, set())
                    if expected != actual:
                        issues.append(
                            ValidationIssue(
                                "error",
                                "composition.membership",
                                f"Composition block {block_id!r} membership differs; "
                                f"missing={sorted(expected - actual)[:5]}, "
                                f"extra={sorted(actual - expected)[:5]}.",
                            )
                        )
        store_report = self.supports.validate(full_scan=level == "full")
        return ValidationReport(tuple(issues)).merged(store_report)

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
        observations = self.observations.to_pandas(copy=False)
        rows = observations.loc[observations["observation_id"].eq(observation_id)]
        if len(rows) != 1:
            raise KeyError(f"Unknown observation_id {observation_id!r}.")
        row = rows.iloc[0]
        selected_representation = representation_id or self.manifest.primary_representation
        self.representations[selected_representation]
        law = None
        if bool(row["geometry_observed"]):
            law = self.supports.read(SupportRef(selected_representation, row["support_key"]))
        selected_channel = abundance_channel or self.manifest.primary_abundance_channel
        abundance_value = None
        if selected_channel is not None:
            if self.abundance is None or selected_channel not in self.abundance.channels:
                raise KeyError(f"Unknown abundance channel {selected_channel!r}.")
            values = self.abundance.to_pandas(copy=False)
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
            self._closed = True

    def __enter__(self) -> Study:
        if self._closed:
            raise RuntimeError("Study is closed.")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _optional_value(row: pd.Series, column: str) -> str | None:
    if column not in row.index or pd.isna(row[column]):
        return None
    value = str(row[column])
    return value or None


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
        if self.abundance_channel is not None:
            if (
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
        # Evaluate metadata filters eagerly so invalid columns fail at construction.
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

    def _selected_series_frame(self) -> pd.DataFrame:
        conditions = _apply_filter(
            self.study.conditions.to_pandas(copy=False), self.selection.condition_filter
        )
        allowed_conditions = set(conditions["condition_id"])
        frame = self.study.series.to_pandas(copy=False)
        frame = frame.loc[frame["condition_id"].isin(allowed_conditions)]
        if self.selection.series_ids is not None:
            frame = frame.loc[frame["series_id"].isin(self.selection.series_ids)]
        return frame

    def _selected_observation_frame(self) -> pd.DataFrame:
        series_ids = set(self._selected_series_frame()["series_id"])
        frame = self.study.observations.to_pandas(copy=False)
        frame = frame.loc[frame["series_id"].isin(series_ids)]
        if self.selection.checkpoint_ids is not None:
            frame = frame.loc[frame["checkpoint_id"].isin(self.selection.checkpoint_ids)]
        return _apply_filter(frame, self.selection.observation_filter)


__all__ = ["SelectionSpec", "Study", "StudyManifest", "StudyView"]
