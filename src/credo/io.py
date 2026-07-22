"""Strict configuration and canonical on-disk dataset I/O."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import Axis, FiniteMeasure, MassSemantics, TrajectoryData, validate_measure_meta

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


class ModelConfig(_StrictModel):
    embedding_dim: int = Field(default=8, ge=1)
    n_programs: int = Field(default=8, ge=1)
    hidden_dim: int = Field(default=128, ge=8)
    context: Literal["none", "catalog_bank"] = "catalog_bank"
    growth_max: float = Field(default=3.0, gt=0)


class EpochConfig(_StrictModel):
    state: int = Field(default=40, ge=0)
    mass: int = Field(default=20, ge=0)
    context: int = Field(default=20, ge=0)

    @model_validator(mode="after")
    def _require_training(self) -> EpochConfig:
        if self.state + self.mass + self.context < 1:
            raise ValueError("At least one training phase must have a positive epoch count.")
        return self


class TrainingConfig(_StrictModel):
    epochs: EpochConfig = EpochConfig()
    particles: int = Field(default=64, ge=2)
    steps_per_interval: int = Field(default=4, ge=1)
    measures_per_batch: int = Field(default=256, ge=1)
    batching: Literal["random", "target_balanced"] = "random"
    learning_rate: float = Field(default=1e-3, gt=0)
    patience: int = Field(default=10, ge=1)
    seed: int = Field(default=0, ge=0)


class EvaluationConfig(_StrictModel):
    particles: int = Field(default=256, ge=2)
    measures_per_batch: int = Field(default=256, ge=1)


class ValidationConfig(_StrictModel):
    strategy: Literal["auto", "context_group", "checkpoint", "train_self_eval"] = "auto"
    values: tuple[str, ...] = ()
    fraction: float = Field(default=0.2, ge=0, lt=1)

    @model_validator(mode="after")
    def _validate_specification(self) -> ValidationConfig:
        explicit = self.strategy in {"context_group", "checkpoint"}
        if explicit and not self.values:
            raise ValueError(f"validation.strategy={self.strategy!r} requires values.")
        if not explicit and self.values:
            raise ValueError(f"validation.strategy={self.strategy!r} does not accept values.")
        if self.strategy == "train_self_eval" and self.fraction != 0:
            raise ValueError("train_self_eval requires validation.fraction=0.")
        if len(set(self.values)) != len(self.values):
            raise ValueError("validation.values must be unique.")
        return self


class LossConfig(_StrictModel):
    mass: float = Field(default=1.0, ge=0)
    count: float = Field(default=0.1, ge=0)
    sinkhorn_epsilon: float = Field(default=0.1, gt=0)


class RunConfig(_StrictModel):
    recipe: Literal["credo.compact_sde_v3@3.0"] = "credo.compact_sde_v3@3.0"
    data: DataConfig
    axis: AxisConfig
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    validation: ValidationConfig = ValidationConfig()
    loss: LossConfig = LossConfig()
    output: Path

    @model_validator(mode="after")
    def _validate_modes(self) -> RunConfig:
        reaction_epochs = self.training.epochs.mass + self.training.epochs.context
        if self.axis.kind == "effect":
            if self.data.counts is not None or self.loss.count > 0:
                raise ValueError("Count likelihood cannot be configured for an effect axis.")
            if reaction_epochs > 0 or self.loss.mass > 0:
                raise ValueError("Growth and mass fitting cannot be configured for an effect axis.")
            if self.model.context != "none":
                raise ValueError("Effect-axis runs require model.context='none'.")
        if self.model.context == "none" and self.training.epochs.context > 0:
            raise ValueError("Context epochs must be zero when model.context is 'none'.")
        if self.training.epochs.context > 0 and self.training.epochs.mass == 0:
            raise ValueError("Context training requires a positive mass phase first.")
        if self.loss.count > 0 and self.data.counts is None:
            raise ValueError("Positive count loss requires data.counts.")
        if reaction_epochs == 0 and (self.loss.mass > 0 or self.loss.count > 0):
            raise ValueError("Mass and count losses require a mass or context training phase.")
        if self.training.epochs.context > 0 and self.loss.mass == 0 and self.loss.count == 0:
            raise ValueError("Context training requires mass or count supervision.")
        if self.validation.strategy == "checkpoint":
            unknown = set(self.validation.values) - set(self.axis.labels[1:])
            if unknown:
                raise ValueError(
                    f"Checkpoint validation contains unknown downstream labels: {sorted(unknown)}"
                )
            if set(self.validation.values) == set(self.axis.labels[1:]):
                raise ValueError(
                    "Checkpoint validation must leave a downstream checkpoint to train."
                )
        if self.validation.strategy == "context_group" and self.validation.fraction != 0:
            raise ValueError("Explicit context-group validation requires validation.fraction=0.")
        if self.validation.strategy == "checkpoint" and self.validation.fraction != 0:
            raise ValueError("Explicit checkpoint validation requires validation.fraction=0.")
        return self


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
    data = dict(raw.get("data", {}))
    for key in ("support", "measure_meta", "masses", "counts", "dataset"):
        if key in data:
            data[key] = _resolve_path(config_path.parent, data[key])
    raw["data"] = data
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
    allowed = {"schema_version", "axis", "latent_key", "mass_semantics", "description", "source"}
    required = {"schema_version", "axis", "latent_key", "mass_semantics", "source"}
    unknown = set(manifest) - allowed
    if unknown:
        raise ValueError(f"dataset.json contains unknown keys: {sorted(unknown)}")
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"dataset.json is missing required keys: {sorted(missing)}")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("dataset.json schema_version must be 1.")
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


def _read_support(path: Path, latent_key: str) -> tuple[pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(path)
    required = {"measure_id", "time_label"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"support.h5ad obs is missing columns: {sorted(missing)}")
    if latent_key not in adata.obsm:
        raise ValueError(f"support.h5ad is missing obsm[{latent_key!r}].")
    obs = adata.obs.copy().reset_index(drop=True)
    for column in required:
        if obs[column].isna().any():
            raise ValueError(f"support.h5ad {column} contains missing values.")
        obs[column] = obs[column].astype(str)
        if obs[column].str.len().eq(0).any():
            raise ValueError(f"support.h5ad {column} contains empty values.")
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if latent.ndim != 2 or len(latent) != len(obs) or not np.isfinite(latent).all():
        raise ValueError("The configured latent representation must be finite and two-dimensional.")
    return obs, latent


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
    obs: pd.DataFrame,
    latent: np.ndarray,
    masses: pd.DataFrame,
    axis: Axis,
) -> dict[str, dict[str, FiniteMeasure]]:
    observed_pairs = set(zip(obs["measure_id"], obs["time_label"], strict=False))
    mass_pairs = set(zip(masses["measure_id"], masses["time_label"], strict=False))
    if observed_pairs != mass_pairs:
        missing = sorted(observed_pairs - mass_pairs)[:5]
        extra = sorted(mass_pairs - observed_pairs)[:5]
        raise ValueError(f"Support and mass rows disagree; missing={missing}, extra={extra}.")
    unknown_labels = set(obs["time_label"]) - set(axis.labels)
    if unknown_labels:
        raise ValueError(
            f"Support contains labels outside the configured axis: {sorted(unknown_labels)}"
        )
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
    from .objective import CountBlock

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


def load_data(config: RunConfig | str | Path) -> TrajectoryData:
    """Load canonical support, metadata, mass, and optional count blocks."""
    run_config = load_config(config) if isinstance(config, (str, Path)) else config
    axis = run_config.axis.build()
    dataset_manifest, dataset_path = _load_dataset_manifest(run_config.data, axis)
    measure_meta = validate_measure_meta(pd.read_parquet(run_config.data.measure_meta))
    obs, latent = _read_support(run_config.data.support, run_config.data.latent_key)
    masses, mass_semantics = _read_masses(run_config.data.masses)
    _validate_denominators(masses, measure_meta, mass_semantics)
    declared_semantics = MassSemantics(dataset_manifest["mass_semantics"])
    if declared_semantics is not mass_semantics:
        raise ValueError("masses.parquet mass semantics disagree with dataset.json.")
    measures = _build_measures(obs, latent, masses, axis)
    measure_ids = tuple(measure_meta["measure_id"].tolist())
    count_blocks = _read_count_blocks(run_config.data.counts, measure_meta, measure_ids)
    input_paths = {
        "support": run_config.data.support,
        "measure_meta": run_config.data.measure_meta,
        "masses": run_config.data.masses,
    }
    if run_config.data.counts is not None:
        input_paths["counts"] = run_config.data.counts
    input_paths["dataset"] = dataset_path
    metadata = {
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        "dataset": dataset_manifest,
        "mass_denominators": sorted(masses["denominator"].unique().tolist()),
    }
    return TrajectoryData(
        axis=axis,
        measures=measures,
        measure_meta=measure_meta,
        mass_semantics=mass_semantics,
        count_blocks=count_blocks,
        metadata=metadata,
    )


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
    manifest = {
        "schema_version": 1,
        "axis": {
            "kind": axis.kind,
            "source": axis.source,
            "labels": list(axis.labels),
            "values": list(axis.values),
        },
        "latent_key": latent_key,
        "mass_semantics": MassSemantics(mass_semantics).value,
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
    if data.axis != config.axis.build():
        raise ValueError("Loaded data axis disagrees with the run configuration.")
    reaction_epochs = config.training.epochs.mass + config.training.epochs.context
    if config.loss.count > 0 and not data.count_blocks:
        raise ValueError("Positive count loss requires at least one complete CountBlock.")
    if data.count_blocks and data.axis.kind != "physical":
        raise ValueError("Count likelihood requires a physical axis.")
    if data.mass_semantics is MassSemantics.UNIT:
        if reaction_epochs > 0 or config.loss.mass > 0 or config.loss.count > 0:
            raise ValueError("unit mass semantics permits state geometry training only.")
        if data.count_blocks:
            raise ValueError("unit mass semantics cannot be combined with count blocks.")


def validate_inputs(config: RunConfig | str | Path) -> dict[str, Any]:
    """Load and summarize a run contract without training."""
    run_config = load_config(config) if isinstance(config, (str, Path)) else config
    data = load_data(run_config)
    validate_run_data(run_config, data)
    source = data.metadata.get("dataset", {}).get("source", {})
    return {
        "recipe": run_config.recipe,
        "measure_count": len(data.measure_ids),
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
