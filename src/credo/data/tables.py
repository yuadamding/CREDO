"""DataFrame-backed semantic catalogs for a CREDO study."""

from __future__ import annotations

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


__all__ = [
    "AbundanceChannelSpec",
    "AbundanceSemantics",
    "AbundanceTable",
    "CompositionTable",
    "ConditionTable",
    "EffectBindingTable",
    "FrameTable",
    "ObservationTable",
    "ReferenceBindingTable",
    "SeriesTable",
    "SupportIndexTable",
]
