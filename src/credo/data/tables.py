"""DataFrame-backed semantic catalogs for a CREDO study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

    def to_pandas(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the normalized table, copying by default."""
        return self._frame.copy() if copy else self._frame


class ConditionTable(FrameTable):
    """Experimental interventions and reference identities."""

    required_columns = (
        "condition_id",
        "condition_kind",
        "embedding_id",
        "reference_group_id",
        "is_reference",
    )
    key_columns = ("condition_id",)
    string_columns = (
        "condition_id",
        "condition_kind",
        "embedding_id",
        "reference_group_id",
    )

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["is_reference"] = _normalize_boolean(frame["is_reference"], "is_reference")
        mixed = frame.groupby("embedding_id", observed=True)["is_reference"].nunique()
        if (mixed > 1).any():
            embedding_id = str(mixed[mixed > 1].index[0])
            raise ValueError(
                f"embedding_id {embedding_id!r} mixes reference and intervention conditions."
            )

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["condition_id"].tolist())


class SeriesTable(FrameTable):
    """Longitudinal units advanced through the study design."""

    required_columns = (
        "series_id",
        "condition_id",
        "subject_id",
        "embedding_id",
        "reference_role",
    )
    key_columns = ("series_id",)
    string_columns = required_columns

    def _validate(self, frame: pd.DataFrame) -> None:
        allowed = {"reference", "intervention"}
        unknown = set(frame["reference_role"]) - allowed
        if unknown:
            raise ValueError(f"SeriesTable contains unknown reference roles: {sorted(unknown)}")

    @property
    def series_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["series_id"].tolist())


class ObservationTable(FrameTable):
    """One longitudinal series at one checkpoint."""

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
        if frame.duplicated(["series_id", "checkpoint_id"]).any():
            raise ValueError(
                "ObservationTable must have at most one observation per series/checkpoint."
            )
        if "support_key" not in frame:
            frame["support_key"] = pd.Series([None] * len(frame), dtype=object)
        _normalize_strings(frame, ("support_key",), nullable=frozenset({"support_key"}))
        optional_strings = tuple(
            column for column in ("context_id", "composition_block_id") if column in frame
        )
        _normalize_strings(frame, optional_strings, nullable=frozenset(optional_strings))
        missing_support = frame["geometry_observed"] & frame["support_key"].isna()
        if missing_support.any():
            observation_id = frame.loc[missing_support, "observation_id"].iloc[0]
            raise ValueError(f"Geometry-observed row {observation_id!r} requires a support_key.")
        hidden_support = ~frame["geometry_observed"] & frame["support_key"].notna()
        if hidden_support.any():
            observation_id = frame.loc[hidden_support, "observation_id"].iloc[0]
            raise ValueError(
                f"Geometry-missing row {observation_id!r} cannot declare a support_key."
            )
        observed_support = frame.loc[frame["geometry_observed"], "support_key"]
        if observed_support.duplicated().any():
            support_key = observed_support.loc[observed_support.duplicated()].iloc[0]
            raise ValueError(f"ObservationTable contains duplicate support_key {support_key!r}.")

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(self._frame["observation_id"].tolist())


class AbundanceSemantics(StrEnum):
    """Scientific interpretation of one abundance channel."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    CAPTURE_COUNT = "capture_count"
    UNIT = "unit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AbundanceChannelSpec:
    """Interpretation, claims, and zero handling for one abundance channel."""

    channel_id: str
    semantics: AbundanceSemantics
    unit: str | None
    denominator_required: bool
    permits_absolute_claim: bool
    permits_relative_claim: bool
    zero_policy: Literal["allowed", "forbidden", "censored"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", str(self.channel_id))
        object.__setattr__(self, "semantics", AbundanceSemantics(self.semantics))
        if not self.channel_id:
            raise ValueError("AbundanceChannelSpec.channel_id must be nonempty.")
        if self.unit is not None and not str(self.unit):
            raise ValueError("AbundanceChannelSpec.unit must be nonempty when provided.")
        if self.zero_policy not in {"allowed", "forbidden", "censored"}:
            raise ValueError(f"Unknown abundance zero policy {self.zero_policy!r}.")
        if self.permits_absolute_claim and self.semantics is not AbundanceSemantics.ABSOLUTE:
            raise ValueError("Only absolute abundance semantics can permit absolute claims.")


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
        self._channels = MappingProxyType(catalog)
        super().__init__(frame)

    def _validate(self, frame: pd.DataFrame) -> None:
        frame["observed"] = _normalize_boolean(frame["observed"], "observed")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        unknown = set(frame["channel_id"]) - set(self._channels)
        if unknown:
            raise ValueError(f"AbundanceTable references unknown channels: {sorted(unknown)}")
        observed = frame["observed"]
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
        "exposure",
        "count",
        "denominator_id",
    )
    key_columns = ("composition_block_id", "series_id")
    string_columns = (
        "composition_block_id",
        "checkpoint_id",
        "context_id",
        "series_id",
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
    "FrameTable",
    "ObservationTable",
    "SeriesTable",
]
