"""Strict configuration and canonical on-disk dataset I/O."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import (
    Axis,
    FiniteMeasure,
    MassSemantics,
    RepresentationArtifact,
    TrajectoryData,
    validate_measure_meta,
)

try:
    pd.set_option("future.infer_string", False)
except (KeyError, ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"CREDO config contains duplicate key {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class DataConfig(_StrictModel):
    support: Path
    latent_key: str = "X_credo"
    measure_meta: Path
    masses: Path
    counts: Path | None = None
    dataset: Path | None = None
    lazy_support: bool = True
    support_cache_size: int = Field(default=256, ge=0)


class AxisConfig(_StrictModel):
    kind: Literal["physical", "effect"]
    source: str
    labels: tuple[str, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_axis(self) -> AxisConfig:
        Axis(kind=self.kind, source=self.source, labels=self.labels, values=self.values)
        return self

    def build(self) -> Axis:
        return Axis(kind=self.kind, source=self.source, labels=self.labels, values=self.values)


class ReplicateSelectionConfig(_StrictModel):
    mode: Literal["reject", "keep_separate", "pool", "select", "hierarchical"] = "reject"
    selection_key: str | None = None
    geometry_pooling: Literal["concatenate", "weighted_prototypes"] | None = None
    abundance_pooling: Literal["sum", "mean", "exposure_weighted"] | None = None


class StudySelectionConfig(_StrictModel):
    representation_id: str | None = None
    abundance_channel: str | None = "__primary__"
    series_ids: tuple[str, ...] | None = None
    observation_ids: tuple[str, ...] | None = None
    checkpoint_ids: tuple[str, ...] | None = None
    subject_ids: tuple[str, ...] | None = None
    experimental_unit_ids: tuple[str, ...] | None = None
    perturbation_ids: tuple[str, ...] | None = None
    construct_ids: tuple[str, ...] | None = None
    target_ids: tuple[str, ...] | None = None
    context_ids: tuple[str, ...] | None = None
    control_kinds: tuple[str, ...] | None = None
    qc_tiers: tuple[str, ...] | None = None
    perturbation_filter: dict[str, Any] | None = None
    condition_filter: dict[str, Any] | None = None
    observation_filter: dict[str, Any] | None = None
    effect_binding_id: str | None = None
    reference_binding_id: str | None = None
    composition_policy: Literal[
        "require_complete",
        "preserve_background",
        "condition_on_selection",
        "drop",
    ] = "require_complete"
    replicate_policy: ReplicateSelectionConfig = Field(default_factory=ReplicateSelectionConfig)

    def build(self):
        from .data.study import ReplicatePolicy, SelectionSpec

        return SelectionSpec(
            series_ids=self.series_ids,
            observation_ids=self.observation_ids,
            checkpoint_ids=self.checkpoint_ids,
            subject_ids=self.subject_ids,
            experimental_unit_ids=self.experimental_unit_ids,
            perturbation_ids=self.perturbation_ids,
            construct_ids=self.construct_ids,
            target_ids=self.target_ids,
            context_ids=self.context_ids,
            control_kinds=self.control_kinds,
            qc_tiers=self.qc_tiers,
            perturbation_filter=self.perturbation_filter,
            condition_filter=self.condition_filter,
            observation_filter=self.observation_filter,
            representation_id=self.representation_id,
            abundance_channel_id=(
                None if self.abundance_channel == "__primary__" else self.abundance_channel
            ),
            effect_binding_id=self.effect_binding_id,
            reference_binding_id=self.reference_binding_id,
            composition_policy=self.composition_policy,
            replicate_policy=ReplicatePolicy(**self.replicate_policy.model_dump()),
        )


class RunConfig(_StrictModel):
    recipe: str = "credo.compact_sde_v3@3.0"
    study: Path | None = None
    selection: StudySelectionConfig = Field(default_factory=StudySelectionConfig)
    data: DataConfig | None = None
    axis: AxisConfig | None = None
    recipe_config: Any
    output: Path

    @model_validator(mode="after")
    def _validate_recipe(self) -> RunConfig:
        native = self.study is not None
        legacy = self.data is not None or self.axis is not None
        if native == legacy:
            raise ValueError(
                "Run config requires exactly one input contract: study, or data plus axis."
            )
        if legacy and (self.data is None or self.axis is None):
            raise ValueError("Legacy run config requires both data and axis.")
        from .registry import get_recipe

        recipe = get_recipe(self.recipe)
        canonical_identifier = f"{recipe.recipe_id}@{recipe.recipe_version}"
        object.__setattr__(self, "recipe", canonical_identifier)
        schema = recipe.config_schema()
        object.__setattr__(self, "recipe_config", schema.model_validate(self.recipe_config))
        validator = getattr(recipe, "validate_run_config", None)
        if callable(validator):
            validator(self)
        return self

    def recipe_configuration(self) -> Any:
        return self.recipe_config

    def view(self, study: Any):
        kwargs: dict[str, Any] = {
            "selection": self.selection.build(),
            "representation_id": self.selection.representation_id,
        }
        if self.selection.abundance_channel != "__primary__":
            kwargs["abundance_channel"] = self.selection.abundance_channel
        return study.view(**kwargs)


def _resolve_path(base: Path, value: Any) -> Any:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> RunConfig:
    """Load one authoritative YAML run definition; unknown keys are errors."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError("CREDO config must be a YAML mapping.")
    raw = dict(raw)
    if "data" in raw:
        data = dict(raw["data"])
        for key in ("support", "measure_meta", "masses", "counts", "dataset"):
            if key in data:
                data[key] = _resolve_path(config_path.parent, data[key])
        raw["data"] = data
    if "study" in raw:
        raw["study"] = _resolve_path(config_path.parent, raw["study"])
    if "output" in raw:
        raw["output"] = _resolve_path(config_path.parent, raw["output"])
    return RunConfig.model_validate(raw)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_manifest_path(config: DataConfig) -> Path:
    path = config.dataset if config.dataset is not None else config.support.parent / "dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical dataset manifest not found: {path}")
    return path


def _load_dataset_manifest(config: DataConfig, axis: Axis) -> tuple[dict[str, Any], Path]:
    path = _dataset_manifest_path(config)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("dataset.json must contain a JSON object.")
    allowed = {
        "schema_version",
        "axis",
        "latent_key",
        "mass_semantics",
        "description",
        "source",
        "representation",
    }
    required = {"schema_version", "axis", "latent_key", "mass_semantics", "source"}
    unknown = set(manifest) - allowed
    if unknown:
        raise ValueError(f"dataset.json contains unknown keys: {sorted(unknown)}")
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"dataset.json is missing required keys: {sorted(missing)}")
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in {1, 2}:
        raise ValueError("dataset.json schema_version must be 1 or 2.")
    if schema_version == 1 and "representation" in manifest:
        raise ValueError("dataset.json schema 1 cannot contain a representation contract.")
    if schema_version == 2 and not isinstance(manifest.get("representation"), dict):
        raise ValueError("dataset.json schema 2 requires a representation object.")
    if not isinstance(manifest["axis"], dict):
        raise ValueError("dataset.json axis must be a JSON object.")
    declared = Axis(**manifest["axis"])
    if declared != axis:
        raise ValueError("Config axis disagrees with dataset.json.")
    if not isinstance(manifest["latent_key"], str) or not manifest["latent_key"]:
        raise ValueError("dataset.json latent_key must be a nonempty string.")
    if manifest["latent_key"] != config.latent_key:
        raise ValueError("Config latent_key disagrees with dataset.json.")
    if not isinstance(manifest["source"], dict):
        raise ValueError("dataset.json source provenance must be a JSON object.")
    if "description" in manifest and not isinstance(manifest["description"], str):
        raise ValueError("dataset.json description must be a string.")
    return manifest, path


@dataclass(frozen=True)
class _H5ADObservationIndex:
    """CSR-style inverse index over categorical H5AD observation identifiers."""

    pairs: tuple[tuple[str, str], ...]
    positions: np.ndarray
    indptr: np.ndarray
    row_count: int
    has_atom_weight: bool


def _h5_string_categories(node: h5py.Group, column: str) -> tuple[str, ...] | None:
    if not isinstance(node, h5py.Group) or not {"categories", "codes"} <= set(node):
        return None
    categories_node = node["categories"]
    codes_node = node["codes"]
    if not isinstance(categories_node, h5py.Dataset) or not isinstance(codes_node, h5py.Dataset):
        return None
    categories = tuple(str(value) for value in categories_node.asstr()[:])
    if (
        not categories
        or any(not value for value in categories)
        or len(categories) != len(set(categories))
    ):
        raise ValueError(f"support.h5ad {column} categories must be unique and nonempty.")
    return categories


def _read_h5ad_observation_index(
    handle: h5py.File,
    *,
    row_count: int,
) -> _H5ADObservationIndex | None:
    obs = handle.get("obs")
    if not isinstance(obs, h5py.Group):
        raise ValueError("support.h5ad is missing its obs table.")
    missing = {"measure_id", "time_label"} - set(obs)
    if missing:
        raise ValueError(f"support.h5ad obs is missing columns: {sorted(missing)}")
    measure_node = obs["measure_id"]
    time_node = obs["time_label"]
    measure_categories = _h5_string_categories(measure_node, "measure_id")
    time_categories = _h5_string_categories(time_node, "time_label")
    if measure_categories is None or time_categories is None:
        return None
    measure_codes = measure_node["codes"]
    time_codes = time_node["codes"]
    if measure_codes.shape != (row_count,) or time_codes.shape != (row_count,):
        raise ValueError("support.h5ad categorical observation columns are misaligned.")
    atom_weight = obs.get("atom_weight")
    if atom_weight is not None and (
        not isinstance(atom_weight, h5py.Dataset) or atom_weight.shape != (row_count,)
    ):
        return None

    pair_codes = np.empty(row_count, dtype=np.int32)
    rows_per_block = 1_048_576
    measure_count = len(measure_categories)
    for start in range(0, row_count, rows_per_block):
        stop = min(start + rows_per_block, row_count)
        measure_block = np.asarray(measure_codes[start:stop], dtype=np.int64)
        time_block = np.asarray(time_codes[start:stop], dtype=np.int64)
        if (
            np.any(measure_block < 0)
            or np.any(measure_block >= measure_count)
            or np.any(time_block < 0)
            or np.any(time_block >= len(time_categories))
        ):
            raise ValueError("support.h5ad categorical observation codes are invalid.")
        if atom_weight is not None:
            weight_block = np.asarray(atom_weight[start:stop], dtype=np.float64)
            if not np.isfinite(weight_block).all() or np.any(weight_block <= 0):
                raise ValueError("support.h5ad atom_weight values must be positive and finite.")
        pair_codes[start:stop] = time_block * measure_count + measure_block

    positions = np.argsort(pair_codes, kind="stable")
    ordered_codes = pair_codes[positions]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_codes[1:] != ordered_codes[:-1]) + 1,
        )
    )
    indptr = np.concatenate((starts, np.asarray([row_count], dtype=np.int64)))
    pairs = tuple(
        (
            time_categories[int(pair_code) // measure_count],
            measure_categories[int(pair_code) % measure_count],
        )
        for pair_code in ordered_codes[starts]
    )
    positions.setflags(write=False)
    indptr.setflags(write=False)
    return _H5ADObservationIndex(
        pairs=pairs,
        positions=positions,
        indptr=indptr,
        row_count=row_count,
        has_atom_weight=atom_weight is not None,
    )


def _scan_h5_dataset(node: h5py.Dataset) -> None:
    block_bytes = max(1, node.shape[1] * node.dtype.itemsize)
    rows_per_block = max(1, (8 * 1024 * 1024) // block_bytes)
    for start in range(0, node.shape[0], rows_per_block):
        block = np.asarray(node[start : start + rows_per_block])
        try:
            finite = np.isfinite(block).all()
        except TypeError as exc:
            raise ValueError("The configured latent representation must be numeric.") from exc
        if not finite:
            raise ValueError("The configured latent representation contains non-finite values.")


def _read_support(
    path: Path,
    latent_key: str,
    *,
    lazy: bool,
    scan_values: bool = True,
) -> tuple[pd.DataFrame | _H5ADObservationIndex, np.ndarray | None, int]:
    if lazy:
        with h5py.File(path, "r") as handle:
            node = handle.get(f"obsm/{latent_key}")
            if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
                raise ValueError("Lazy support requires a dense two-dimensional HDF5 obsm dataset.")
            shape = tuple(int(value) for value in node.shape)
            if shape[0] < 1 or shape[1] < 1:
                raise ValueError("The configured latent representation is empty.")
            compact_index = _read_h5ad_observation_index(handle, row_count=shape[0])
            if compact_index is not None:
                if scan_values:
                    _scan_h5_dataset(node)
                return compact_index, None, shape[1]

    adata = ad.read_h5ad(path, backed="r" if lazy else None)
    try:
        required = {"measure_id", "time_label"}
        missing = required - set(adata.obs.columns)
        if missing:
            raise ValueError(f"support.h5ad obs is missing columns: {sorted(missing)}")
        obs = adata.obs.copy().reset_index(drop=True)
        for column in required:
            if obs[column].isna().any():
                raise ValueError(f"support.h5ad {column} contains missing values.")
            obs[column] = obs[column].astype(str)
            if obs[column].str.len().eq(0).any():
                raise ValueError(f"support.h5ad {column} contains empty values.")
        if not lazy:
            if latent_key not in adata.obsm:
                raise ValueError(f"support.h5ad is missing obsm[{latent_key!r}].")
            latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
            if latent.ndim != 2 or len(latent) != len(obs) or not np.isfinite(latent).all():
                raise ValueError(
                    "The configured latent representation must be finite and two-dimensional."
                )
            return obs, latent, int(latent.shape[1])
    finally:
        if lazy:
            adata.file.close()

    with h5py.File(path, "r") as handle:
        node = handle.get(f"obsm/{latent_key}")
        if not isinstance(node, h5py.Dataset) or len(node.shape) != 2:
            raise ValueError("Lazy support requires a dense two-dimensional HDF5 obsm dataset.")
        shape = tuple(int(value) for value in node.shape)
        if shape[0] != len(obs) or shape[1] < 1:
            raise ValueError("The configured latent representation is not aligned to obs.")
        if scan_values:
            _scan_h5_dataset(node)
    return obs, None, shape[1]


class _H5ADCheckpointMeasures(Mapping[str, FiniteMeasure]):
    def __init__(self, store: H5ADFiniteMeasureStore, label: str) -> None:
        self._store = store
        self._label = label

    def __getitem__(self, measure_id: str) -> FiniteMeasure:
        return self._store.measure(self._label, str(measure_id))

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.measure_ids_by_label[self._label])

    def __len__(self) -> int:
        return len(self._store.measure_ids_by_label[self._label])


def _position_ranges(raw_positions: Any) -> tuple[tuple[int, int], ...]:
    positions = np.sort(np.asarray(raw_positions, dtype=np.int64))
    if not len(positions):
        raise ValueError("One H5AD support law has no atoms.")
    boundaries = np.flatnonzero(np.diff(positions) != 1) + 1
    starts = np.concatenate((positions[:1], positions[boundaries]))
    stops = np.concatenate((positions[boundaries - 1] + 1, positions[-1:] + 1))
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))


class H5ADFiniteMeasureStore(Mapping[str, Mapping[str, FiniteMeasure]]):
    """Bounded, lazy finite-measure view over a dense H5AD latent cache."""

    is_lazy = True

    def __init__(
        self,
        path: Path,
        latent_key: str,
        obs: pd.DataFrame | _H5ADObservationIndex,
        masses: pd.DataFrame,
        axis: Axis,
        *,
        latent_dim: int,
        cache_size: int,
    ) -> None:
        self.path = path
        self.latent_key = latent_key
        self.latent_dim = int(latent_dim)
        self.cache_size = int(cache_size)
        self._lock = threading.RLock()
        self._handle: h5py.File | None = None
        self._cache: OrderedDict[tuple[str, str], FiniteMeasure] = OrderedDict()
        self._mass = masses.set_index(["measure_id", "time_label"])["mass"].to_dict()
        ids_by_label: dict[str, list[str]] = {label: [] for label in axis.labels}
        if isinstance(obs, _H5ADObservationIndex):
            self._pair_index = {pair: index for index, pair in enumerate(obs.pairs)}
            self._packed_positions = obs.positions
            self._packed_indptr = obs.indptr
            self._positions: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}
            self._atom_weight: np.ndarray | None = None
            self._atom_weight_on_disk = obs.has_atom_weight
            pairs = obs.pairs
        else:
            self._pair_index: dict[tuple[str, str], int] = {}
            self._packed_positions = np.empty(0, dtype=np.int64)
            self._packed_indptr = np.asarray([0], dtype=np.int64)
            self._atom_weight = (
                pd.to_numeric(obs["atom_weight"], errors="raise").to_numpy(dtype=np.float64)
                if "atom_weight" in obs
                else None
            )
            if self._atom_weight is not None and (
                not np.isfinite(self._atom_weight).all() or np.any(self._atom_weight <= 0)
            ):
                raise ValueError("support.h5ad atom_weight values must be positive and finite.")
            self._atom_weight_on_disk = False
            grouped = obs.groupby(["time_label", "measure_id"], observed=True, sort=False).indices
            self._positions = {
                (str(label), str(measure_id)): _position_ranges(raw_positions)
                for (label, measure_id), raw_positions in grouped.items()
            }
            pairs = tuple(self._positions)
        for label, measure_id in pairs:
            ids_by_label[label].append(measure_id)
        self.measure_ids_by_label = {label: tuple(ids_by_label[label]) for label in axis.labels}
        self._views = {label: _H5ADCheckpointMeasures(self, label) for label in axis.labels}

    def __getitem__(self, label: str) -> Mapping[str, FiniteMeasure]:
        return self._views[str(label)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._views)

    def __len__(self) -> int:
        return len(self._views)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def _dataset(self) -> h5py.Dataset:
        node = self._file()[f"obsm/{self.latent_key}"]
        if not isinstance(node, h5py.Dataset):  # pragma: no cover - checked at construction.
            raise TypeError("Configured latent cache is not a dense HDF5 dataset.")
        return node

    def _read_ranges(
        self,
        dataset: h5py.Dataset,
        ranges: tuple[tuple[int, int], ...],
        *,
        dtype: Any,
    ) -> np.ndarray:
        blocks = [np.asarray(dataset[start:stop], dtype=dtype) for start, stop in ranges]
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    def _row_positions(self, key: tuple[str, str]) -> np.ndarray:
        if key in self._pair_index:
            law_index = self._pair_index[key]
            start = int(self._packed_indptr[law_index])
            stop = int(self._packed_indptr[law_index + 1])
            return self._packed_positions[start:stop]
        ranges = self._positions[key]
        return np.concatenate([np.arange(start, stop, dtype=np.int64) for start, stop in ranges])

    def _read_law(
        self,
        dataset: h5py.Dataset,
        key: tuple[str, str],
        *,
        dtype: Any,
    ) -> np.ndarray:
        if key in self._pair_index:
            return np.asarray(dataset[self._row_positions(key)], dtype=dtype)
        return self._read_ranges(dataset, self._positions[key], dtype=dtype)

    def measure(self, label: str, measure_id: str) -> FiniteMeasure:
        key = (str(label), str(measure_id))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            if key not in self._pair_index and key not in self._positions:
                raise KeyError(measure_id)
            support = self._read_law(self._dataset(), key, dtype=np.float32)
            if self._atom_weight is not None:
                local = self._atom_weight[self._row_positions(key)]
            elif self._atom_weight_on_disk:
                node = self._file()["obs/atom_weight"]
                if not isinstance(node, h5py.Dataset):  # pragma: no cover - indexed at load.
                    raise TypeError("Configured atom weights are not a dense HDF5 dataset.")
                local = self._read_law(node, key, dtype=np.float64)
            else:
                local = np.ones(len(support), dtype=np.float64)
            total_mass = float(self._mass[(measure_id, label)])
            measure = FiniteMeasure(support, total_mass * local / local.sum(), total_mass)
            if self.cache_size > 0:
                self._cache[key] = measure
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return measure

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _validate_mass_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, MassSemantics]:
    required = {"measure_id", "time_label", "mass", "mass_semantics", "denominator"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"masses.parquet is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ("measure_id", "time_label", "mass_semantics", "denominator"):
        if frame[column].isna().any():
            raise ValueError(f"masses.parquet {column} contains missing values.")
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise ValueError(f"masses.parquet {column} contains empty values.")
    frame["mass"] = pd.to_numeric(frame["mass"], errors="raise")
    if frame.duplicated(["measure_id", "time_label"]).any():
        raise ValueError("masses.parquet must have one row per measure_id/time_label.")
    if not np.isfinite(frame["mass"]).all() or (frame["mass"] <= 0).any():
        raise ValueError("All masses must be positive and finite.")
    semantics_values = frame["mass_semantics"].astype(str).unique().tolist()
    if len(semantics_values) != 1:
        raise ValueError("masses.parquet must declare one mass_semantics value.")
    semantics = MassSemantics(semantics_values[0])
    if semantics is MassSemantics.UNIT and not np.allclose(frame["mass"], 1.0):
        raise ValueError("unit mass semantics requires every mass to equal 1.")
    return frame, semantics


def _read_masses(path: Path) -> tuple[pd.DataFrame, MassSemantics]:
    return _validate_mass_table(pd.read_parquet(path))


def _validate_denominators(
    masses: pd.DataFrame,
    measure_meta: pd.DataFrame,
    semantics: MassSemantics,
) -> None:
    if semantics not in {
        MassSemantics.RELATIVE_WITHIN_GROUP,
        MassSemantics.CAPTURED_COUNT,
    }:
        return
    scoped = masses.merge(
        measure_meta[["measure_id", "context_group_id"]],
        on="measure_id",
        how="left",
        validate="many_to_one",
    )
    by_scope = scoped.groupby(["context_group_id", "time_label"], observed=True)[
        "denominator"
    ].nunique()
    if (by_scope != 1).any():
        scope = by_scope[by_scope != 1].index[0]
        raise ValueError(
            "Each context-group/time mass scope must declare exactly one denominator; "
            f"invalid scope={scope!r}."
        )
    denominator_scope = scoped.groupby("denominator", observed=True).agg(
        context_groups=("context_group_id", "nunique"),
        time_labels=("time_label", "nunique"),
    )
    invalid = (denominator_scope["context_groups"] != 1) | (denominator_scope["time_labels"] != 1)
    if invalid.any():
        denominator = denominator_scope.index[invalid][0]
        raise ValueError(
            "Relative or captured-count denominator identifiers must be unique to one "
            f"context group and time; invalid denominator={denominator!r}."
        )


def _build_measures(
    obs: pd.DataFrame | _H5ADObservationIndex,
    latent: np.ndarray | None,
    masses: pd.DataFrame,
    axis: Axis,
    *,
    support_path: Path,
    latent_key: str,
    latent_dim: int,
    cache_size: int,
) -> Mapping[str, Mapping[str, FiniteMeasure]]:
    if isinstance(obs, _H5ADObservationIndex):
        observed_pairs = {(measure_id, time_label) for time_label, measure_id in obs.pairs}
        observed_labels = {time_label for time_label, _ in obs.pairs}
    else:
        observed_pairs = set(zip(obs["measure_id"], obs["time_label"], strict=False))
        observed_labels = set(obs["time_label"])
    mass_pairs = set(zip(masses["measure_id"], masses["time_label"], strict=False))
    if observed_pairs != mass_pairs:
        missing = sorted(observed_pairs - mass_pairs)[:5]
        extra = sorted(mass_pairs - observed_pairs)[:5]
        raise ValueError(f"Support and mass rows disagree; missing={missing}, extra={extra}.")
    unknown_labels = observed_labels - set(axis.labels)
    if unknown_labels:
        raise ValueError(
            f"Support contains labels outside the configured axis: {sorted(unknown_labels)}"
        )
    if latent is None:
        return H5ADFiniteMeasureStore(
            support_path,
            latent_key,
            obs,
            masses,
            axis,
            latent_dim=latent_dim,
            cache_size=cache_size,
        )
    if not isinstance(obs, pd.DataFrame):  # pragma: no cover - compact index is lazy-only.
        raise TypeError("Eager support construction requires materialized observations.")
    mass_lookup = masses.set_index(["measure_id", "time_label"])["mass"].to_dict()
    atom_weight = (
        pd.to_numeric(obs["atom_weight"], errors="raise").to_numpy(dtype=np.float64)
        if "atom_weight" in obs
        else np.ones(len(obs), dtype=np.float64)
    )
    if not np.isfinite(atom_weight).all() or np.any(atom_weight <= 0):
        raise ValueError("support.h5ad atom_weight values must be positive and finite.")
    measures: dict[str, dict[str, FiniteMeasure]] = {label: {} for label in axis.labels}
    grouped = obs.groupby(["time_label", "measure_id"], observed=True, sort=False).indices
    for (label, measure_id), positions in grouped.items():
        positions = np.asarray(positions, dtype=np.int64)
        total_mass = float(mass_lookup[(measure_id, label)])
        local = atom_weight[positions]
        weights = total_mass * local / local.sum()
        measures[label][measure_id] = FiniteMeasure(latent[positions], weights, total_mass)
    return measures


def _build_count_blocks(
    frame: pd.DataFrame | None,
    measure_meta: pd.DataFrame,
    measure_ids: tuple[str, ...],
) -> tuple[Any, ...]:
    if frame is None:
        return ()
    from .recipes.compact_sde_v3.objective import CountBlock

    index = {measure_id: idx for idx, measure_id in enumerate(measure_ids)}
    unknown = set(frame["measure_id"]) - set(index)
    if unknown:
        raise ValueError(f"counts.parquet references unknown measures: {sorted(unknown)[:5]}")
    meta_group = measure_meta.set_index("measure_id")["context_group_id"].to_dict()
    blocks = []
    for (group_id, time_label), rows in frame.groupby(
        ["context_group_id", "time_label"], observed=True, sort=False
    ):
        expected = {
            measure_id
            for measure_id, declared_group in meta_group.items()
            if declared_group == group_id
        }
        observed = set(rows["measure_id"])
        if observed != expected:
            missing_ids = sorted(expected - observed)[:5]
            extra_ids = sorted(observed - expected)[:5]
            raise ValueError(
                f"Count block {group_id!r}/{time_label!r} has an incomplete denominator; "
                f"missing={missing_ids}, extra={extra_ids}."
            )
        ordered = rows.sort_values("measure_id")
        blocks.append(
            CountBlock(
                context_group_id=group_id,
                time_label=time_label,
                measure_indices=np.asarray([index[value] for value in ordered["measure_id"]]),
                exposure=ordered["exposure"].to_numpy(),
                counts=ordered["count"].to_numpy(),
            )
        )
    return tuple(blocks)


def _normalize_count_table(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return the exact validated count representation written to disk."""
    if frame is None:
        return None
    required = {"context_group_id", "time_label", "measure_id", "exposure", "count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"counts.parquet is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ("context_group_id", "time_label", "measure_id"):
        if frame[column].isna().any():
            raise ValueError(f"counts.parquet {column} contains missing values.")
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise ValueError(f"counts.parquet {column} contains empty values.")
    frame["exposure"] = pd.to_numeric(frame["exposure"], errors="raise")
    frame["count"] = pd.to_numeric(frame["count"], errors="raise")
    if frame.duplicated(["context_group_id", "time_label", "measure_id"]).any():
        raise ValueError("counts.parquet contains duplicate block entries.")
    return frame


def _read_count_blocks(
    path: Path | None,
    measure_meta: pd.DataFrame,
    measure_ids: tuple[str, ...],
) -> tuple[Any, ...]:
    frame = None if path is None else pd.read_parquet(path)
    return _build_count_blocks(_normalize_count_table(frame), measure_meta, measure_ids)


def _load_canonical_data(data_config: DataConfig, axis: Axis) -> TrajectoryData:
    """Load the current five-file compatibility schema into ``TrajectoryData``."""
    dataset_manifest, dataset_path = _load_dataset_manifest(data_config, axis)
    measure_meta = validate_measure_meta(pd.read_parquet(data_config.measure_meta))
    obs, latent, latent_dim = _read_support(
        data_config.support,
        data_config.latent_key,
        lazy=data_config.lazy_support,
    )
    masses, mass_semantics = _read_masses(data_config.masses)
    _validate_denominators(masses, measure_meta, mass_semantics)
    declared_semantics = MassSemantics(dataset_manifest["mass_semantics"])
    if declared_semantics is not mass_semantics:
        raise ValueError("masses.parquet mass semantics disagree with dataset.json.")
    measures = _build_measures(
        obs,
        latent,
        masses,
        axis,
        support_path=data_config.support,
        latent_key=data_config.latent_key,
        latent_dim=latent_dim,
        cache_size=data_config.support_cache_size,
    )
    measure_ids = tuple(measure_meta["measure_id"].tolist())
    count_blocks = _read_count_blocks(data_config.counts, measure_meta, measure_ids)
    input_paths = {
        "support": data_config.support,
        "measure_meta": data_config.measure_meta,
        "masses": data_config.masses,
    }
    if data_config.counts is not None:
        input_paths["counts"] = data_config.counts
    input_paths["dataset"] = dataset_path
    metadata = {
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        "dataset": dataset_manifest,
        "mass_denominators": sorted(masses["denominator"].unique().tolist()),
    }
    representation = (
        None
        if int(dataset_manifest["schema_version"]) == 1
        else RepresentationArtifact.from_dict(dataset_manifest["representation"])
    )
    if (
        representation is not None
        and representation.producer.get("latent_cache_hash_kind") == "support_h5ad_file_sha256"
        and representation.latent_cache_hash != metadata["input_hashes"]["support"]
    ):
        raise ValueError("Representation latent-cache hash disagrees with support.h5ad.")
    return TrajectoryData(
        axis=axis,
        measures=measures,
        measure_meta=measure_meta,
        mass_semantics=mass_semantics,
        count_blocks=count_blocks,
        metadata=metadata,
        representation=representation,
    )


def load_data(config: RunConfig | str | Path) -> TrajectoryData:
    """Load canonical support, metadata, mass, and optional count blocks."""
    run_config = load_config(config) if isinstance(config, (str, Path)) else config
    if run_config.data is None or run_config.axis is None:
        raise ValueError("load_data() is the legacy five-file adapter; use open_study().")
    return _load_canonical_data(run_config.data, run_config.axis.build())


def write_canonical_dataset(
    output_dir: str | Path,
    *,
    support: ad.AnnData,
    measure_meta: pd.DataFrame,
    masses: pd.DataFrame,
    axis: Axis,
    mass_semantics: MassSemantics,
    latent_key: str = "X_credo",
    counts: pd.DataFrame | None = None,
    description: str = "",
    source: dict[str, Any] | None = None,
    representation: RepresentationArtifact | None = None,
) -> dict[str, Path]:
    """Write the one canonical adapter output contract."""
    output = Path(output_dir)
    if not isinstance(latent_key, str) or not latent_key:
        raise ValueError("Canonical latent_key must be a nonempty string.")
    if not isinstance(description, str):
        raise ValueError("Canonical description must be a string.")
    measure_meta = validate_measure_meta(measure_meta)
    support = support.copy()
    support.obs.index = pd.Index(support.obs.index.astype(str).to_numpy(dtype=object), dtype=object)
    if latent_key not in support.obsm:
        raise ValueError(f"Support AnnData is missing obsm[{latent_key!r}].")
    required_support = {"measure_id", "time_label"}
    if missing := required_support - set(support.obs.columns):
        raise ValueError(f"Support AnnData obs is missing columns: {sorted(missing)}")
    for column in required_support:
        if support.obs[column].isna().any():
            raise ValueError(f"Support AnnData {column} contains missing values.")
        values = support.obs[column].astype(str)
        if values.str.len().eq(0).any():
            raise ValueError(f"Support AnnData {column} contains empty values.")
        support.obs[column] = values.to_numpy(dtype=object)
    for column in support.obs.columns:
        if pd.api.types.is_string_dtype(support.obs[column].dtype):
            support.obs[column] = support.obs[column].astype(str).to_numpy(dtype=object)
    masses = masses.copy()
    masses["mass_semantics"] = MassSemantics(mass_semantics).value
    masses, written_semantics = _validate_mass_table(masses)
    if written_semantics is not MassSemantics(mass_semantics):
        raise ValueError("Mass table semantics disagree with the requested mass_semantics.")
    _validate_denominators(masses, measure_meta, written_semantics)
    obs = support.obs.copy().reset_index(drop=True)
    obs["measure_id"] = obs["measure_id"].astype(str)
    obs["time_label"] = obs["time_label"].astype(str)
    latent = np.asarray(support.obsm[latent_key], dtype=np.float32)
    if latent.ndim != 2 or len(latent) != len(obs) or not np.isfinite(latent).all():
        raise ValueError("The canonical latent representation must be finite and two-dimensional.")
    support.obsm[latent_key] = latent
    observed_pairs = set(zip(obs["measure_id"], obs["time_label"], strict=False))
    mass_pairs = set(zip(masses["measure_id"], masses["time_label"], strict=False))
    if observed_pairs != mass_pairs:
        raise ValueError("Support and mass rows must contain exactly the same measure/time pairs.")
    if unknown_labels := set(obs["time_label"]) - set(axis.labels):
        raise ValueError(f"Support contains labels outside the axis: {sorted(unknown_labels)}")
    measure_ids = tuple(measure_meta["measure_id"].tolist())
    source_ids = set(obs.loc[obs["time_label"].eq(axis.source), "measure_id"])
    if source_ids != set(measure_ids):
        raise ValueError(
            "Every metadata measure must have source support, with no unknown source IDs."
        )
    if not set(obs["measure_id"]) <= source_ids:
        raise ValueError("Downstream support contains a measure without source support.")
    if "atom_weight" in obs:
        atom_weight = pd.to_numeric(obs["atom_weight"], errors="raise").to_numpy()
        if not np.isfinite(atom_weight).all() or np.any(atom_weight <= 0):
            raise ValueError("Support atom weights must be positive and finite.")
        support.obs["atom_weight"] = atom_weight
    counts = _normalize_count_table(counts)
    count_blocks = _build_count_blocks(counts, measure_meta, measure_ids)
    if axis.kind == "effect" and count_blocks:
        raise ValueError("Count likelihood is unavailable on a nonphysical effect axis.")
    for block in count_blocks:
        axis.index(block.time_label)
    if source is not None and not isinstance(source, dict):
        raise ValueError("Canonical source provenance must be a JSON object.")
    if representation is None:
        digest = hashlib.sha256()
        digest.update(np.asarray(latent.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(latent, dtype="<f4", order="C").tobytes(order="C"))
        for measure_id, time_label in zip(obs["measure_id"], obs["time_label"], strict=True):
            digest.update(str(measure_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(time_label).encode("utf-8"))
            digest.update(b"\0")
        latent_hash = digest.hexdigest()
        representation = RepresentationArtifact(
            representation_id=f"frozen-latent:{latent_hash[:12]}",
            backend="frozen_latent",
            latent_dim=latent.shape[1],
            latent_cache_hash=latent_hash,
            fit_scope="external",
            producer={
                "source": "canonical_dataset_writer",
                "fitting_cohort_known": False,
            },
        )
    if representation.latent_dim != latent.shape[1]:
        raise ValueError("Representation latent_dim disagrees with canonical support.")
    manifest = {
        "schema_version": 2,
        "axis": {
            "kind": axis.kind,
            "source": axis.source,
            "labels": list(axis.labels),
            "values": list(axis.values),
        },
        "latent_key": latent_key,
        "mass_semantics": MassSemantics(mass_semantics).value,
        "representation": representation.to_dict(),
        "description": description,
        "source": source or {},
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    paths = {
        "support": output / "support.h5ad",
        "measure_meta": output / "measure_meta.parquet",
        "masses": output / "masses.parquet",
        "dataset": output / "dataset.json",
    }
    if counts is not None:
        paths["counts"] = output / "counts.parquet"
    if output.exists():
        expected = {path.name for path in paths.values()}
        unknown = sorted(path.name for path in output.iterdir() if path.name not in expected)
        if unknown:
            raise FileExistsError(
                f"Canonical dataset directory contains files outside its contract: {unknown}"
            )
    output.mkdir(parents=True, exist_ok=True)
    try:
        option_context = pd.option_context("future.infer_string", False)
    except (KeyError, ValueError):
        option_context = nullcontext()
    with option_context:
        support.write_h5ad(paths["support"])
    measure_meta.to_parquet(paths["measure_meta"], index=False)
    masses.to_parquet(paths["masses"], index=False)
    if counts is not None:
        counts.to_parquet(paths["counts"], index=False)
    paths["dataset"].write_text(manifest_text, encoding="utf-8")
    return paths


def resolved_config(config: RunConfig) -> dict[str, Any]:
    """Return a JSON-safe resolved configuration."""
    return config.model_dump(mode="json")


def validate_run_data(config: RunConfig, data: TrajectoryData) -> None:
    """Reject scientifically inconsistent run and loaded-data combinations."""
    if config.axis is not None and data.axis != config.axis.build():
        raise ValueError("Loaded data axis disagrees with the run configuration.")
    from .registry import get_recipe
    from .runtime import validate_recipe_study

    recipe = get_recipe(config.recipe)
    validate_recipe_study(recipe, data)
    if recipe.recipe_id != "credo.compact_sde_v3":
        return
    settings = config.recipe_config
    reaction_epochs = settings.training.epochs.mass + settings.training.epochs.context
    if data.axis.kind == "effect":
        if data.count_blocks or settings.loss.count > 0:
            raise ValueError("Count likelihood cannot be configured for an effect axis.")
        if reaction_epochs > 0 or settings.loss.mass > 0:
            raise ValueError("Growth and mass fitting cannot be configured for an effect axis.")
        if settings.model.context != "none":
            raise ValueError("Effect-axis runs require model.context='none'.")
    if settings.loss.count > 0 and not data.count_blocks:
        raise ValueError("Positive count loss requires at least one complete CountBlock.")
    if data.count_blocks and data.axis.kind != "physical":
        raise ValueError("Count likelihood requires a physical axis.")
    if data.mass_semantics is MassSemantics.UNIT:
        if reaction_epochs > 0 or settings.loss.mass > 0 or settings.loss.count > 0:
            raise ValueError("unit mass semantics permits state geometry training only.")
        if data.count_blocks:
            raise ValueError("unit mass semantics cannot be combined with count blocks.")


def validate_inputs(config: RunConfig | str | Path) -> dict[str, Any]:
    """Load and summarize a run contract without training."""
    run_config = load_config(config) if isinstance(config, (str, Path)) else config
    from .data import open_study
    from .data.splits import validate_representation_scope, validate_split_plan
    from .registry import get_recipe
    from .runtime import validate_view_for_recipe

    source = run_config if run_config.study is None else run_config.study
    owner = open_study(source, verify="semantic")
    try:
        view = run_config.view(owner)
        recipe = get_recipe(run_config.recipe)
        split = recipe.plan_split(view, run_config.recipe_config)
        validate_split_plan(view, split)
        validate_representation_scope(view, split)
        validate_view_for_recipe(
            view, split, recipe.requirements(run_config.recipe_config)
        ).raise_for_errors()
        problem = recipe.compile_study(view, split, run_config.recipe_config)
        selected_measure_count = len(view.series_ids)
    finally:
        owner.close()
    data = getattr(problem, "training", problem)
    validation_data = getattr(problem, "validation", data)
    validate_run_data(run_config, data)
    validate_run_data(run_config, validation_data)
    source = data.metadata.get("dataset", {}).get("source", {})
    return {
        "recipe": run_config.recipe,
        "measure_count": selected_measure_count,
        "training_measure_count": len(data.measure_ids),
        "validation_measure_count": len(validation_data.measure_ids),
        "embedding_count": len(data.embedding_ids),
        "control_measure_count": int(data.measure_meta["is_control"].sum()),
        "latent_dim": data.latent_dim,
        "axis_kind": data.axis.kind,
        "axis_labels": list(data.axis.labels),
        "mass_semantics": data.mass_semantics.value,
        "mass_denominator_count": len(data.metadata.get("mass_denominators", [])),
        "claim_policy": data.claim_policy,
        "count_block_count": len(data.count_blocks),
        "mass_denominator_scope": source.get("mass_denominator_scope"),
        "count_block_scope": source.get("count_block_scope"),
    }
