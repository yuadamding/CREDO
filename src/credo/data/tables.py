"""DataFrame-backed semantic catalogs for a CREDO study."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Literal

import numpy as np
import pandas as pd


def _normalize_strings(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    nullable: frozenset[str] = frozenset(),
) -> None:
    for column in columns:
        missing = frame[column].isna()
        if missing.any() and column not in nullable:
            raise ValueError(f"{column} contains missing values.")
        present = ~missing
        frame.loc[present, column] = frame.loc[present, column].astype(str)
        if frame.loc[present, column].astype(str).str.len().eq(0).any():
            raise ValueError(f"{column} contains empty values.")


def _normalize_boolean(values: pd.Series, column: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{column} contains missing values.")
    if values.dtype == bool:
        return values.astype(bool)
    normalized = values.astype(str).str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError(f"{column} must be boolean.")
    return normalized.isin({"true", "1"})


class FrameTable:
    """Read-only public wrapper around one normalized tabular catalog."""

    required_columns: ClassVar[tuple[str, ...]] = ()
    key_columns: ClassVar[tuple[str, ...]] = ()
    string_columns: ClassVar[tuple[str, ...]] = ()

    __slots__ = ("_frame",)

    def __init__(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} requires a pandas DataFrame.")
        missing = set(self.required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{type(self).__name__} is missing columns: {sorted(missing)}")
        normalized = frame.copy().reset_index(drop=True)
        _normalize_strings(normalized, self.string_columns)
        if self.key_columns and normalized.duplicated(list(self.key_columns)).any():
            duplicate = normalized.loc[
                normalized.duplicated(list(self.key_columns)), list(self.key_columns)
            ].iloc[0]
            raise ValueError(
                f"{type(self).__name__} contains a duplicate key {tuple(duplicate)!r}."
            )
        self._validate(normalized)
        self._frame = normalized

    def _validate(self, frame: pd.DataFrame) -> None:
        del frame

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self._frame.columns)

    def to_pandas(self) -> pd.DataFrame:
        """Return an owned copy of the normalized table."""
        return self._frame.copy()

    def _unsafe_view(self) -> pd.DataFrame:
        """Return the internal frame for trusted package code."""
        return self._frame


class ConditionTable(FrameTable):
    """Experimental interventions and reference identities."""

    required_columns = (
        "condition_id",
        "condition_kind",
        "is_reference",
    )
    key_columns = ("condition_id",)
    string_columns = ("condition_id", "condition_kind")

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["is_reference"] = _normalize_boolean(frame["is_reference"], "is_reference")
        optional = tuple(
            column for column in ("embedding_id", "reference_group_id") if column in frame
        )
        _normalize_strings(frame, optional, nullable=frozenset(optional))

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["condition_id"].tolist())


class SeriesTable(FrameTable):
    """Longitudinal units advanced through the study design."""

    required_columns = (
        "series_id",
        "condition_id",
        "subject_id",
    )
    key_columns = ("series_id",)
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        optional = tuple(column for column in ("embedding_id", "reference_role") if column in frame)
        _normalize_strings(frame, optional, nullable=frozenset(optional))
        if "reference_role" not in frame:
            return
        allowed = {"reference", "intervention"}
        unknown = set(frame["reference_role"].dropna()) - allowed
        if unknown:
            raise ValueError(f"SeriesTable contains unknown reference roles: {sorted(unknown)}")

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["series_id"].tolist())


class EffectBindingTable(FrameTable):
    """Run-selectable mapping from biological conditions to model effect identities."""

    required_columns = (
        "binding_id",
        "condition_id",
        "effect_id",
        "parameterization_kind",
    )
    key_columns = ("binding_id", "condition_id")
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        optional = tuple(
            column for column in ("parent_effect_id", "shrinkage_group_id") if column in frame
        )
        _normalize_strings(frame, optional, nullable=frozenset(optional))

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._frame["binding_id"].tolist()))


class ReferenceBindingTable(FrameTable):
    """Run-selectable mapping from conditions to biological reference pools."""

    required_columns = (
        "binding_id",
        "condition_id",
        "reference_pool_id",
        "scope_kind",
    )
    key_columns = ("binding_id", "condition_id")
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        optional = ("scope_key",) if "scope_key" in frame else ()
        _normalize_strings(frame, optional, nullable=frozenset(optional))

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._frame["binding_id"].tolist()))


class ObservationTable(FrameTable):
    """One assay observation of a longitudinal series at one checkpoint."""

    required_columns = (
        "observation_id",
        "series_id",
        "checkpoint_id",
        "sample_id",
        "geometry_observed",
    )
    key_columns = ("observation_id",)
    string_columns = ("observation_id", "series_id", "checkpoint_id", "sample_id")

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["geometry_observed"] = _normalize_boolean(
            frame["geometry_observed"], "geometry_observed"
        )
        optional_strings = tuple(
            column
            for column in (
                "replicate_id",
                "assay_id",
                "context_id",
                "composition_block_id",
                "processing_batch_id",
            )
            if column in frame
        )
        _normalize_strings(frame, optional_strings, nullable=frozenset(optional_strings))

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["observation_id"].tolist())


class PerturbationTable(FrameTable):
    """Experimentally assigned perturbations, distinct from targets and effects."""

    required_columns = (
        "perturbation_id",
        "perturbation_kind",
        "is_control",
        "control_kind",
    )
    key_columns = ("perturbation_id",)
    string_columns = ("perturbation_id", "perturbation_kind", "control_kind")

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["is_control"] = _normalize_boolean(frame["is_control"], "is_control")
        perturbation_kinds = {
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
        control_kinds = {
            "non_targeting",
            "safe_targeting",
            "untreated",
            "vehicle",
            "mock",
            "positive_control",
            "other",
            "none",
        }
        unknown_perturbations = set(frame["perturbation_kind"]) - perturbation_kinds
        unknown_controls = set(frame["control_kind"]) - control_kinds
        if unknown_perturbations:
            raise ValueError(
                "PerturbationTable contains unknown perturbation kinds: "
                f"{sorted(unknown_perturbations)}"
            )
        if unknown_controls:
            raise ValueError(
                f"PerturbationTable contains unknown control kinds: {sorted(unknown_controls)}"
            )
        invalid_controls = frame["is_control"] & frame["control_kind"].eq("none")
        invalid_interventions = ~frame["is_control"] & ~frame["control_kind"].eq("none")
        if invalid_controls.any() or invalid_interventions.any():
            raise ValueError("control_kind must be non-'none' exactly when is_control is true.")
        optional = tuple(
            column for column in ("library_id", "modality", "description") if column in frame
        )
        _normalize_strings(frame, optional, nullable=frozenset(optional))

    @property
    def perturbation_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["perturbation_id"].tolist())


class PerturbationComponentTable(FrameTable):
    """Normalized guide, reagent, target, and combination components."""

    required_columns = (
        "perturbation_id",
        "component_id",
        "construct_id",
        "target_id",
        "component_kind",
        "dose",
        "dose_unit",
        "component_order",
    )
    key_columns = ("perturbation_id", "component_id")
    string_columns = (
        "perturbation_id",
        "component_id",
        "construct_id",
        "target_id",
        "component_kind",
    )

    def _validate(self, frame: pd.DataFrame) -> None:
        _normalize_strings(frame, ("dose_unit",), nullable=frozenset({"dose_unit"}))
        frame["dose"] = pd.to_numeric(frame["dose"], errors="coerce")
        present_dose = frame["dose"].notna()
        if present_dose.any():
            values = frame.loc[present_dose, "dose"].to_numpy(float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Perturbation component doses must be finite and nonnegative.")
        if (present_dose != frame["dose_unit"].notna()).any():
            raise ValueError("Perturbation component dose and dose_unit must be declared together.")
        order = pd.to_numeric(frame["component_order"], errors="raise").to_numpy(float)
        if (
            not np.isfinite(order).all()
            or np.any(order < 0)
            or not np.allclose(order, np.round(order))
        ):
            raise ValueError("component_order must contain nonnegative integers.")
        frame["component_order"] = np.round(order).astype(np.int64)
        if frame.duplicated(["perturbation_id", "component_order"]).any():
            raise ValueError("component_order must be unique within each perturbation.")


class InterventionEventTable(FrameTable):
    """Timed interventions and environmental events along population series."""

    required_columns = (
        "event_id",
        "series_id",
        "agent_id",
        "event_kind",
        "modeled_role",
        "start_coordinate",
        "end_coordinate",
        "start_relation",
        "persistent",
        "dose",
        "dose_unit",
    )
    key_columns = ("event_id",)
    string_columns = (
        "event_id",
        "series_id",
        "agent_id",
        "event_kind",
        "modeled_role",
        "start_relation",
    )

    def _validate(self, frame: pd.DataFrame) -> None:
        roles = {
            "primary_perturbation",
            "background_stimulus",
            "culture_change",
            "drug_washout",
            "selection_event",
            "other_context",
        }
        relations = {
            "before_source",
            "at_source",
            "between_checkpoints",
            "after_last_observation",
            "unknown",
        }
        unknown_roles = set(frame["modeled_role"]) - roles
        unknown_relations = set(frame["start_relation"]) - relations
        if unknown_roles:
            raise ValueError(f"Unknown intervention modeled roles: {sorted(unknown_roles)}")
        if unknown_relations:
            raise ValueError(f"Unknown intervention start relations: {sorted(unknown_relations)}")
        frame["persistent"] = _normalize_boolean(frame["persistent"], "persistent")
        for column in ("start_coordinate", "end_coordinate", "dose"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            values = frame.loc[frame[column].notna(), column].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{column} must be finite when observed.")
        if (frame["dose"].dropna() < 0).any():
            raise ValueError("Intervention doses must be nonnegative.")
        _normalize_strings(frame, ("dose_unit",), nullable=frozenset({"dose_unit"}))
        if (frame["dose"].notna() != frame["dose_unit"].notna()).any():
            raise ValueError("Intervention dose and dose_unit must be declared together.")
        observed = frame["start_coordinate"].notna() & frame["end_coordinate"].notna()
        if (frame.loc[observed, "end_coordinate"] < frame.loc[observed, "start_coordinate"]).any():
            raise ValueError("Intervention end_coordinate cannot precede start_coordinate.")


class ContextTable(FrameTable):
    """Biological and technical covariate contexts, independent of perturbations."""

    required_columns = ("context_id", "context_kind")
    key_columns = ("context_id",)
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        optional = tuple(
            column
            for column in (
                "subject_id",
                "tissue",
                "cell_system",
                "stimulation",
                "disease_state",
                "covariates_json",
            )
            if column in frame
        )
        _normalize_strings(frame, optional, nullable=frozenset(optional))
        if "covariates_json" in frame:
            for index, value in frame["covariates_json"].dropna().items():
                try:
                    parsed = json.loads(str(value))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Context covariates_json at row {index} is invalid JSON."
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValueError("Context covariates_json must encode a JSON object.")


class PopulationSeriesTable(FrameTable):
    """Matched population-level series; never an implicit single-cell lineage."""

    required_columns = (
        "series_id",
        "subject_id",
        "experimental_unit_id",
        "perturbation_id",
        "context_trajectory_id",
        "biological_replicate_id",
        "continuity_kind",
    )
    key_columns = ("series_id",)
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        allowed = {
            "same_experimental_unit",
            "matched_subject_parallel",
            "cross_sectional_population",
            "independent_replicate",
            "lineage_linked",
            "exact_lineage_traced",
            "unknown",
        }
        unknown = set(frame["continuity_kind"]) - allowed
        if unknown:
            raise ValueError(f"Unknown population continuity kinds: {sorted(unknown)}")

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["series_id"].tolist())


class SnapshotObservationTable(FrameTable):
    """One destructive population observation at one experimental checkpoint."""

    required_columns = (
        "observation_id",
        "series_id",
        "checkpoint_id",
        "sample_id",
        "geometry_observed",
        "abundance_observed",
    )
    key_columns = ("observation_id",)
    string_columns = ("observation_id", "series_id", "checkpoint_id", "sample_id")

    def _validate(self, frame: pd.DataFrame) -> None:
        for column in ("geometry_observed", "abundance_observed"):
            frame[column] = _normalize_boolean(frame[column], column)
        optional = tuple(
            column
            for column in (
                "assay_id",
                "technical_replicate_id",
                "context_id",
                "population_pool_id",
                "assignment_qc",
                "processing_batch_id",
                "composition_block_id",
            )
            if column in frame
        )
        _normalize_strings(frame, optional, nullable=frozenset(optional))

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["observation_id"].tolist())


class PopulationPoolTable(FrameTable):
    """Physical or computational population groupings with explicit evidence."""

    required_columns = (
        "population_pool_id",
        "pool_kind",
        "checkpoint_id",
        "experimental_unit_id",
        "evidence_level",
        "description",
    )
    key_columns = ("population_pool_id",)
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        allowed = {
            "shared_living_culture",
            "shared_tissue",
            "shared_animal",
            "competition_pool",
            "sequencing_library",
            "capture_batch",
            "computational_group",
        }
        unknown = set(frame["pool_kind"]) - allowed
        if unknown:
            raise ValueError(f"Unknown population pool kinds: {sorted(unknown)}")


class PerturbationEffectBindingTable(FrameTable):
    """Run-selectable perturbation-to-model-effect parameterization."""

    required_columns = (
        "binding_id",
        "perturbation_id",
        "effect_id",
        "parameterization_kind",
    )
    key_columns = ("binding_id", "perturbation_id")
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        optional = tuple(
            column for column in ("parent_effect_id", "shrinkage_group_id") if column in frame
        )
        _normalize_strings(frame, optional, nullable=frozenset(optional))

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._frame["binding_id"].tolist()))


class PerturbationReferenceBindingTable(FrameTable):
    """Observed control matching and model counterfactual reference bindings."""

    required_columns = (
        "binding_id",
        "perturbation_id",
        "reference_pool_id",
        "scope_kind",
        "match_keys",
        "counterfactual_effect_id",
    )
    key_columns = ("binding_id", "perturbation_id")
    string_columns = (
        "binding_id",
        "perturbation_id",
        "reference_pool_id",
        "scope_kind",
        "counterfactual_effect_id",
    )

    def _validate(self, frame: pd.DataFrame) -> None:
        scopes = {
            "global",
            "subject",
            "experimental_unit",
            "context",
            "checkpoint",
            "processing_batch",
        }
        unknown = set(frame["scope_kind"]) - scopes
        if unknown:
            raise ValueError(f"Unknown reference binding scopes: {sorted(unknown)}")
        _normalize_strings(frame, ("match_keys",), nullable=frozenset())
        required_key = {
            "global": None,
            "subject": "subject_id",
            "experimental_unit": "experimental_unit_id",
            "context": "context_id",
            "checkpoint": "checkpoint_id",
            "processing_batch": "processing_batch_id",
        }
        for row in frame.itertuples(index=False):
            try:
                keys = json.loads(str(row.match_keys))
            except json.JSONDecodeError as exc:
                raise ValueError("Reference match_keys must be a JSON string array.") from exc
            if (
                not isinstance(keys, list)
                or any(not isinstance(value, str) or not value for value in keys)
                or len(keys) != len(set(keys))
            ):
                raise ValueError("Reference match_keys must encode unique, nonempty string keys.")
            expected = required_key[str(row.scope_kind)]
            if expected is not None and expected not in keys:
                raise ValueError(
                    f"Reference scope {row.scope_kind!r} requires match key {expected!r}."
                )

    @property
    def binding_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._frame["binding_id"].tolist()))


class SupportIndexTable(FrameTable):
    """Representation-specific support availability for each observation."""

    required_columns = (
        "observation_id",
        "representation_id",
        "store_id",
        "support_key",
        "available",
    )
    key_columns = ("observation_id", "representation_id")
    string_columns = ("observation_id", "representation_id")

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["available"] = _normalize_boolean(frame["available"], "available")
        _normalize_strings(
            frame,
            ("store_id", "support_key"),
            nullable=frozenset({"store_id", "support_key"}),
        )
        incomplete = frame["available"] & (frame["store_id"].isna() | frame["support_key"].isna())
        if incomplete.any():
            row = frame.loc[incomplete].iloc[0]
            raise ValueError(
                "Available support requires store_id and support_key for "
                f"{row['observation_id']!r}/{row['representation_id']!r}."
            )
        hidden = ~frame["available"] & (frame["store_id"].notna() | frame["support_key"].notna())
        if hidden.any():
            row = frame.loc[hidden].iloc[0]
            raise ValueError(
                "Unavailable support cannot declare store_id or support_key for "
                f"{row['observation_id']!r}/{row['representation_id']!r}."
            )
        available = frame.loc[frame["available"]]
        if available.duplicated(["store_id", "representation_id", "support_key"]).any():
            row = available.loc[
                available.duplicated(["store_id", "representation_id", "support_key"])
            ].iloc[0]
            raise ValueError(
                "SupportIndexTable contains a duplicate qualified support key "
                f"{(row['store_id'], row['representation_id'], row['support_key'])!r}."
            )


class AbundanceSemantics(StrEnum):
    """Scientific interpretation of one abundance channel."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    CAPTURE_COUNT = "capture_count"
    UNIT = "unit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AbundanceChannelSpec:
    """Interpretation, denominator scope, and zero handling for one channel."""

    channel_id: str
    semantics: AbundanceSemantics
    unit: str | None = None
    denominator_scope: Literal["none", "context_checkpoint", "sample_checkpoint", "custom"] = "none"
    zero_policy: Literal["allowed", "forbidden", "censored"] = "allowed"
    transform_id: str | None = None
    input_channel_id: str | None = None
    transform_parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", str(self.channel_id))
        object.__setattr__(self, "semantics", AbundanceSemantics(self.semantics))
        if not self.channel_id:
            raise ValueError("AbundanceChannelSpec.channel_id must be nonempty.")
        if self.unit is not None and not str(self.unit):
            raise ValueError("AbundanceChannelSpec.unit must be nonempty when provided.")
        if self.denominator_scope not in {
            "none",
            "context_checkpoint",
            "sample_checkpoint",
            "custom",
        }:
            raise ValueError(f"Unknown denominator scope {self.denominator_scope!r}.")
        if self.zero_policy not in {"allowed", "forbidden", "censored"}:
            raise ValueError(f"Unknown abundance zero policy {self.zero_policy!r}.")
        if self.transform_id is not None:
            transform_id = str(self.transform_id)
            if not transform_id:
                raise ValueError("AbundanceChannelSpec.transform_id must be nonempty when set.")
            object.__setattr__(self, "transform_id", transform_id)
        if self.input_channel_id is not None:
            input_channel_id = str(self.input_channel_id)
            if not input_channel_id:
                raise ValueError("AbundanceChannelSpec.input_channel_id must be nonempty when set.")
            object.__setattr__(self, "input_channel_id", input_channel_id)
        parameters = MappingProxyType(dict(self.transform_parameters))
        object.__setattr__(self, "transform_parameters", parameters)
        if (self.transform_id is None) != (self.input_channel_id is None):
            raise ValueError("Abundance transforms require both transform_id and input_channel_id.")
        if self.transform_id is None and parameters:
            raise ValueError("Abundance transform parameters require transform_id.")
        if self.input_channel_id == self.channel_id:
            raise ValueError("An abundance transform cannot use its output as its input.")
        requires_denominator = self.semantics in {
            AbundanceSemantics.RELATIVE,
            AbundanceSemantics.CAPTURE_COUNT,
        }
        if requires_denominator == (self.denominator_scope == "none"):
            raise ValueError(
                f"Abundance semantics {self.semantics.value!r} require a denominator scope."
            )

    @property
    def denominator_required(self) -> bool:
        return self.denominator_scope != "none"

    @property
    def permits_absolute_claim(self) -> bool:
        return self.semantics is AbundanceSemantics.ABSOLUTE

    @property
    def permits_relative_claim(self) -> bool:
        return self.semantics in {
            AbundanceSemantics.ABSOLUTE,
            AbundanceSemantics.RELATIVE,
        }


class AbundanceTable(FrameTable):
    """Observed values for one or more named abundance channels."""

    required_columns = ("observation_id", "channel_id", "value", "observed")
    key_columns = ("observation_id", "channel_id")
    string_columns = ("observation_id", "channel_id")

    __slots__ = ("_channels",)

    def __init__(
        self,
        frame: pd.DataFrame,
        channels: Mapping[str, AbundanceChannelSpec] | Sequence[AbundanceChannelSpec],
    ) -> None:
        values = tuple(channels.values()) if isinstance(channels, Mapping) else tuple(channels)
        catalog = {spec.channel_id: spec for spec in values}
        if not catalog:
            raise ValueError("AbundanceTable requires at least one channel specification.")
        if len(catalog) != len(values):
            raise ValueError("Abundance channel identifiers must be unique.")
        missing_inputs = {
            spec.input_channel_id
            for spec in values
            if spec.input_channel_id is not None and spec.input_channel_id not in catalog
        }
        if missing_inputs:
            raise ValueError(
                f"Abundance transforms reference unknown input channels: {sorted(missing_inputs)}."
            )
        self._channels = MappingProxyType(catalog)
        super().__init__(frame)

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["observed"] = _normalize_boolean(frame["observed"], "observed")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        optional_strings = tuple(
            column
            for column in ("denominator_id", "transform_id", "source_artifact_id")
            if column in frame
        )
        _normalize_strings(frame, optional_strings, nullable=frozenset(optional_strings))
        unknown = set(frame["channel_id"]) - set(self._channels)
        if unknown:
            raise ValueError(f"AbundanceTable references unknown channels: {sorted(unknown)}")
        observed = frame["observed"]
        if frame.loc[~observed, "value"].notna().any():
            raise ValueError("Unobserved abundance rows must have a missing value.")
        values = frame.loc[observed, "value"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Observed abundance values must be finite and nonnegative.")
        for channel_id, rows in frame.loc[observed].groupby("channel_id", observed=True):
            spec = self._channels[str(channel_id)]
            if spec.zero_policy == "forbidden" and rows["value"].eq(0).any():
                raise ValueError(f"Abundance channel {channel_id!r} forbids zero values.")
            if spec.denominator_required:
                if "denominator_id" not in frame:
                    raise ValueError(f"Abundance channel {channel_id!r} requires denominator_id.")
                denominator = rows["denominator_id"]
                if denominator.isna().any() or denominator.astype(str).str.len().eq(0).any():
                    raise ValueError(
                        f"Observed channel {channel_id!r} requires nonempty denominator_id."
                    )
            if spec.transform_id is not None:
                if (
                    "transform_id" not in rows
                    or not rows["transform_id"].eq(spec.transform_id).all()
                ):
                    raise ValueError(
                        f"Abundance channel {channel_id!r} requires transform_id "
                        f"{spec.transform_id!r}."
                    )
            if spec.zero_policy == "censored":
                required = {"lower_bound", "upper_bound"}
                if not required <= set(rows):
                    raise ValueError(f"Censored abundance channel {channel_id!r} requires bounds.")
                lower = pd.to_numeric(rows["lower_bound"], errors="coerce").to_numpy(float)
                upper = pd.to_numeric(rows["upper_bound"], errors="coerce").to_numpy(float)
                if (
                    not np.isfinite(lower).all()
                    or not np.isfinite(upper).all()
                    or np.any(lower < 0)
                    or np.any(upper < lower)
                    or np.any(rows["value"].to_numpy(float) < lower)
                    or np.any(rows["value"].to_numpy(float) > upper)
                ):
                    raise ValueError(
                        f"Censored abundance channel {channel_id!r} has invalid bounds."
                    )

    @property
    def channels(self) -> Mapping[str, AbundanceChannelSpec]:
        return self._channels

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(self._channels)


class CompositionTable(FrameTable):
    """Stable-ID compositional observations before runtime compilation."""

    required_columns = (
        "composition_block_id",
        "checkpoint_id",
        "context_id",
        "series_id",
        "observation_id",
        "exposure",
        "count",
        "denominator_id",
    )
    key_columns = ("composition_block_id", "observation_id")
    string_columns = (
        "composition_block_id",
        "checkpoint_id",
        "context_id",
        "series_id",
        "observation_id",
        "denominator_id",
    )

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["exposure"] = pd.to_numeric(frame["exposure"], errors="raise")
        frame["count"] = pd.to_numeric(frame["count"], errors="raise")
        exposure = frame["exposure"].to_numpy(dtype=np.float64)
        counts = frame["count"].to_numpy(dtype=np.float64)
        if not np.isfinite(exposure).all() or np.any(exposure <= 0):
            raise ValueError("Composition exposures must be positive and finite.")
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Composition counts must be nonnegative and finite.")
        if not np.allclose(counts, np.round(counts)):
            raise ValueError("Composition counts must be integer-like.")
        frame["count"] = np.round(counts).astype(np.int64)
        consistency = frame.groupby("composition_block_id", observed=True).agg(
            checkpoint_count=("checkpoint_id", "nunique"),
            context_count=("context_id", "nunique"),
            denominator_count=("denominator_id", "nunique"),
        )
        invalid = consistency.ne(1).any(axis=1)
        if invalid.any():
            block_id = str(consistency.index[invalid][0])
            raise ValueError(
                f"Composition block {block_id!r} must have one checkpoint, context, "
                "and denominator."
            )

    @property
    def block_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._frame["composition_block_id"].tolist()))


class LPSCompositionTable(CompositionTable):
    """Composition observations whose denominator kind is scientifically explicit."""

    required_columns = (*CompositionTable.required_columns, "block_kind")
    string_columns = (*CompositionTable.string_columns, "block_kind")

    def _validate(self, frame: pd.DataFrame) -> None:
        super()._validate(frame)
        allowed = {
            "sequencing_library",
            "competition_pool",
            "culture_pool",
            "capture_stratum",
            "sampling_stratum",
        }
        unknown = set(frame["block_kind"]) - allowed
        if unknown:
            raise ValueError(f"Unknown composition block kinds: {sorted(unknown)}")
        kinds = frame.groupby("composition_block_id", observed=True)["block_kind"].nunique()
        if kinds.ne(1).any():
            block_id = str(kinds[kinds.ne(1)].index[0])
            raise ValueError(f"Composition block {block_id!r} has multiple block kinds.")


__all__ = [
    "AbundanceChannelSpec",
    "AbundanceSemantics",
    "AbundanceTable",
    "CompositionTable",
    "ConditionTable",
    "ContextTable",
    "EffectBindingTable",
    "FrameTable",
    "InterventionEventTable",
    "LPSCompositionTable",
    "ObservationTable",
    "PerturbationComponentTable",
    "PerturbationEffectBindingTable",
    "PerturbationReferenceBindingTable",
    "PerturbationTable",
    "PopulationPoolTable",
    "PopulationSeriesTable",
    "ReferenceBindingTable",
    "SeriesTable",
    "SnapshotObservationTable",
    "SupportIndexTable",
]
