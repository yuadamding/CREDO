"""Canonical scientific contracts used by every CREDO execution path."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd


class MassSemantics(StrEnum):
    """Provenance of finite-measure totals."""

    ABSOLUTE = "absolute"
    RELATIVE_WITHIN_GROUP = "relative_within_group"
    CAPTURED_COUNT = "captured_count"
    UNIT = "unit"

    @property
    def permits_absolute_growth_claim(self) -> bool:
        return self is MassSemantics.ABSOLUTE

    @property
    def permits_relative_abundance_claim(self) -> bool:
        return self in {
            MassSemantics.ABSOLUTE,
            MassSemantics.RELATIVE_WITHIN_GROUP,
        }

    @property
    def abundance_claim(self) -> str:
        return {
            MassSemantics.ABSOLUTE: "potentially_absolute",
            MassSemantics.RELATIVE_WITHIN_GROUP: "relative_only",
            MassSemantics.CAPTURED_COUNT: "diagnostic_only",
            MassSemantics.UNIT: "none",
        }[self]


@dataclass(frozen=True)
class Axis:
    """Ordered physical-time or nonphysical effect checkpoints."""

    kind: Literal["physical", "effect"]
    labels: Sequence[str]
    values: Sequence[float]
    source: str

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.labels)
        values = tuple(float(value) for value in self.values)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source", str(self.source))
        if self.kind not in {"physical", "effect"}:
            raise ValueError("Axis.kind must be 'physical' or 'effect'.")
        if len(labels) < 2 or len(labels) != len(values):
            raise ValueError("Axis requires at least two equally sized labels and values.")
        if len(set(labels)) != len(labels):
            raise ValueError("Axis labels must be unique.")
        if self.source != labels[0]:
            raise ValueError("Axis.source must be the first checkpoint label.")
        if not np.isfinite(np.asarray(values)).all():
            raise ValueError("Axis values must be finite.")
        if any(right <= left for left, right in zip(values[:-1], values[1:], strict=False)):
            raise ValueError("Axis values must be strictly increasing.")

    def index(self, label: str) -> int:
        try:
            return self.labels.index(str(label))
        except ValueError as exc:
            raise KeyError(f"Unknown axis label {label!r}.") from exc

    def normalized(self, label: str) -> float:
        value = self.values[self.index(label)]
        return (value - self.values[0]) / (self.values[-1] - self.values[0])

    @property
    def normalized_values(self) -> tuple[float, ...]:
        return tuple(self.normalized(label) for label in self.labels)

    def require_physical(self, claim: str) -> None:
        if self.kind != "physical":
            raise ValueError(f"{claim} requires a physical axis; this dataset uses an effect axis.")


RepresentationFitScope = Literal[
    "external",
    "all_source_samples",
    "training_fold_source",
    "all_checkpoints",
]


@dataclass(frozen=True)
class RepresentationArtifact:
    """Immutable provenance for the coordinates used by a dynamics recipe."""

    representation_id: str
    backend: str
    latent_dim: int
    latent_cache_hash: str
    fit_scope: RepresentationFitScope
    gene_names_hash: str | None = None
    gene_mask_hash: str | None = None
    encoder_state_hash: str | None = None
    decoder_state_hash: str | None = None
    normalization_hash: str | None = None
    included_samples: tuple[str, ...] = ()
    included_time_labels: tuple[str, ...] = ()
    producer: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "representation_id", str(self.representation_id))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "latent_dim", int(self.latent_dim))
        for name in (
            "latent_cache_hash",
            "gene_names_hash",
            "gene_mask_hash",
            "encoder_state_hash",
            "decoder_state_hash",
            "normalization_hash",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else str(value).lower())
        object.__setattr__(
            self, "included_samples", tuple(str(value) for value in self.included_samples)
        )
        object.__setattr__(
            self,
            "included_time_labels",
            tuple(str(value) for value in self.included_time_labels),
        )
        object.__setattr__(self, "producer", MappingProxyType(dict(self.producer)))
        if not self.representation_id or not self.backend:
            raise ValueError("Representation identifiers and backend must be nonempty.")
        if self.latent_dim < 1:
            raise ValueError("RepresentationArtifact.latent_dim must be positive.")
        if self.fit_scope not in {
            "external",
            "all_source_samples",
            "training_fold_source",
            "all_checkpoints",
        }:
            raise ValueError(f"Unknown representation fit_scope {self.fit_scope!r}.")
        for name in (
            "latent_cache_hash",
            "gene_names_hash",
            "gene_mask_hash",
            "encoder_state_hash",
            "decoder_state_hash",
            "normalization_hash",
        ):
            value = getattr(self, name)
            invalid_hash = value is not None and (
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            )
            if invalid_hash:
                raise ValueError(f"RepresentationArtifact.{name} must be a SHA-256 hex digest.")
        if len(set(self.included_samples)) != len(self.included_samples):
            raise ValueError("RepresentationArtifact.included_samples must be unique.")
        if len(set(self.included_time_labels)) != len(self.included_time_labels):
            raise ValueError("RepresentationArtifact.included_time_labels must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "backend": self.backend,
            "latent_dim": self.latent_dim,
            "gene_names_hash": self.gene_names_hash,
            "gene_mask_hash": self.gene_mask_hash,
            "encoder_state_hash": self.encoder_state_hash,
            "decoder_state_hash": self.decoder_state_hash,
            "latent_cache_hash": self.latent_cache_hash,
            "normalization_hash": self.normalization_hash,
            "fit_scope": self.fit_scope,
            "included_samples": list(self.included_samples),
            "included_time_labels": list(self.included_time_labels),
            "producer": dict(self.producer),
        }


@dataclass(frozen=True)
class SplitSpec:
    """Explicit dynamics and representation validation scope."""

    strategy: Literal[
        "none",
        "sample",
        "context_group",
        "guide",
        "measure",
        "embedding",
        "checkpoint",
        "external",
    ]
    train_values: tuple[str, ...] | None = None
    validation_values: tuple[str, ...] | None = None
    fold: int | None = None
    folds: int | None = None
    representation_scope: Literal["shared", "nested"] = "shared"
    split_id: str | None = None

    def __post_init__(self) -> None:
        train = None if self.train_values is None else tuple(str(v) for v in self.train_values)
        validation = None
        if self.validation_values is not None:
            validation = tuple(str(value) for value in self.validation_values)
        object.__setattr__(self, "train_values", train)
        object.__setattr__(self, "validation_values", validation)
        object.__setattr__(self, "split_id", None if self.split_id is None else str(self.split_id))
        if self.strategy not in {
            "none",
            "sample",
            "context_group",
            "guide",
            "measure",
            "embedding",
            "checkpoint",
            "external",
        }:
            raise ValueError(f"Unknown split strategy {self.strategy!r}.")
        if (self.fold is None) != (self.folds is None):
            raise ValueError("SplitSpec.fold and folds must be supplied together.")
        if self.fold is not None and not (0 <= self.fold < self.folds):  # type: ignore[operator]
            raise ValueError("SplitSpec.fold must be in [0, folds).")
        if self.representation_scope not in {"shared", "nested"}:
            raise ValueError("SplitSpec.representation_scope must be 'shared' or 'nested'.")
        if train is not None and len(train) != len(set(train)):
            raise ValueError("SplitSpec.train_values must be unique.")
        if validation is not None and len(validation) != len(set(validation)):
            raise ValueError("SplitSpec.validation_values must be unique.")
        if train is not None and validation is not None and set(train) & set(validation):
            raise ValueError("SplitSpec train and validation values must be disjoint.")


@dataclass(frozen=True)
class CapabilitySet:
    """Machine-readable scientific and execution capabilities of one recipe."""

    physical_axis: bool
    effect_axis: bool
    endpoint: bool
    multitime: bool
    mass: bool
    counts: bool
    weak_form: bool
    context: Literal["none", "full_population", "catalog_bank"]
    context_affects: tuple[str, ...]
    external_evaluation: bool
    resume_training: bool
    focal_counterfactual: bool
    full_group_counterfactual: bool
    exact_retraining: bool = False

    def __post_init__(self) -> None:
        affects = tuple(str(value) for value in self.context_affects)
        object.__setattr__(self, "context_affects", affects)
        if self.context not in {"none", "full_population", "catalog_bank"}:
            raise ValueError(f"Unknown context capability {self.context!r}.")
        unknown = set(affects) - {"drift", "diffusion", "growth"}
        if unknown:
            raise ValueError(f"Unknown context-affected coefficients: {sorted(unknown)}")
        if self.context == "none" and affects:
            raise ValueError("A no-context recipe cannot declare context-affected coefficients.")

    def require(self, operation: str) -> None:
        aliases = {
            "evaluate": "external_evaluation",
            "resume": "resume_training",
            "counterfactual": "focal_counterfactual",
            "full_group_counterfactual": "full_group_counterfactual",
            "weak_form": "weak_form",
            "counts": "counts",
            "mass": "mass",
        }
        field_name = aliases.get(operation, operation)
        if not hasattr(self, field_name):
            raise ValueError(f"Unknown recipe operation {operation!r}.")
        if not bool(getattr(self, field_name)):
            raise RuntimeError(f"Recipe does not support {operation!r}.")


@dataclass(frozen=True)
class OptimizerSpec:
    kind: Literal["adam", "adamw"]
    learning_rate: float
    weight_decay: float = 0.0
    parameter_learning_rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter_learning_rates",
            MappingProxyType(
                {str(name): float(value) for name, value in self.parameter_learning_rates.items()}
            ),
        )
        if self.kind not in {"adam", "adamw"}:
            raise ValueError(f"Unknown optimizer kind {self.kind!r}.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "Optimizer learning rate must be positive and weight decay nonnegative."
            )
        if any(value <= 0 for value in self.parameter_learning_rates.values()):
            raise ValueError("Per-parameter learning rates must be positive.")


@dataclass(frozen=True)
class BatchingSpec:
    mode: Literal["all_keys", "measure_batches"]
    measures_per_batch: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"all_keys", "measure_batches"}:
            raise ValueError(f"Unknown batching mode {self.mode!r}.")
        if self.mode == "all_keys" and self.measures_per_batch is not None:
            raise ValueError("all_keys batching cannot set measures_per_batch.")
        if self.mode == "measure_batches" and (
            self.measures_per_batch is None or self.measures_per_batch < 1
        ):
            raise ValueError("measure_batches requires a positive measures_per_batch.")


@dataclass(frozen=True)
class Stage:
    name: str
    epochs: int
    trainable_tags: tuple[str, ...]
    precision: Literal["fp32", "bf16", "fp16"]
    optimizer: OptimizerSpec
    active_objectives: tuple[str, ...]
    batching: BatchingSpec
    context_policy: Literal["none", "full_population", "catalog_bank"]
    checkpoint_metric: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trainable_tags", tuple(str(v) for v in self.trainable_tags))
        object.__setattr__(self, "active_objectives", tuple(str(v) for v in self.active_objectives))
        if not self.name or self.epochs < 0 or not self.checkpoint_metric:
            raise ValueError("Stage requires a name, nonnegative epochs, and checkpoint metric.")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unknown stage precision {self.precision!r}.")
        if self.context_policy not in {"none", "full_population", "catalog_bank"}:
            raise ValueError(f"Unknown stage context policy {self.context_policy!r}.")
        if not self.trainable_tags or len(self.trainable_tags) != len(set(self.trainable_tags)):
            raise ValueError("Stage trainable_tags must be nonempty and unique.")
        if not self.active_objectives or len(self.active_objectives) != len(
            set(self.active_objectives)
        ):
            raise ValueError("Stage active_objectives must be nonempty and unique.")


@dataclass(frozen=True)
class TrainingPlan:
    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        names = [stage.name for stage in self.stages]
        if not names or len(names) != len(set(names)):
            raise ValueError("TrainingPlan requires at least one uniquely named stage.")


@dataclass(frozen=True)
class FiniteMeasure:
    """A discrete finite measure whose atom weights sum to ``total_mass``."""

    support: np.ndarray
    weights: np.ndarray
    total_mass: float

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        total_mass = float(self.total_mass)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "total_mass", total_mass)
        if support.ndim != 2 or support.shape[0] == 0:
            raise ValueError("FiniteMeasure.support must have shape [n_atoms, latent_dim].")
        if len(weights) != len(support):
            raise ValueError("FiniteMeasure support and weights must have equal length.")
        if not np.isfinite(support).all():
            raise ValueError("FiniteMeasure support contains non-finite values.")
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("FiniteMeasure weights must be finite, nonnegative, and nonzero.")
        if not np.isfinite(total_mass) or total_mass <= 0:
            raise ValueError("FiniteMeasure total_mass must be positive and finite.")
        if not np.isclose(weights.sum(), total_mass, rtol=1e-5, atol=1e-10):
            raise ValueError("FiniteMeasure weights must sum to total_mass.")

    @property
    def latent_dim(self) -> int:
        return int(self.support.shape[1])

    @property
    def normalized_weights(self) -> np.ndarray:
        return self.weights / self.total_mass


MEASURE_META_COLUMNS = (
    "measure_id",
    "sample_id",
    "perturbation_id",
    "guide_id",
    "embedding_id",
    "target_gene",
    "context_group_id",
    "is_control",
)

SupportStore = Mapping[str, Mapping[str, np.ndarray]]
MassTable = pd.DataFrame


def _support_sha256(measures: Mapping[str, Mapping[str, FiniteMeasure]]) -> str:
    """Hash supplied latent coordinates when no adapter hash is available."""
    digest = hashlib.sha256()
    for label in sorted(measures):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        for measure_id in sorted(measures[label]):
            support = np.asarray(measures[label][measure_id].support, dtype="<f4", order="C")
            digest.update(measure_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(np.asarray(support.shape, dtype="<i8").tobytes())
            digest.update(support.tobytes(order="C"))
    return digest.hexdigest()


def validate_measure_meta(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the one-row-per-measure metadata contract."""
    missing = set(MEASURE_META_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"measure_meta is missing columns: {sorted(missing)}")
    out = frame.copy()
    for column in MEASURE_META_COLUMNS[:-1]:
        if out[column].isna().any():
            raise ValueError(f"measure_meta.{column} contains missing values.")
        out[column] = out[column].astype(str)
        if out[column].str.len().eq(0).any():
            raise ValueError(f"measure_meta.{column} contains empty values.")
    if out["measure_id"].duplicated().any():
        duplicate = out.loc[out["measure_id"].duplicated(), "measure_id"].iloc[0]
        raise ValueError(f"measure_meta contains duplicate measure_id {duplicate!r}.")
    if out["is_control"].dtype != bool:
        values = out["is_control"].astype(str).str.lower()
        if not values.isin({"true", "false", "1", "0"}).all():
            raise ValueError("measure_meta.is_control must be boolean.")
        out["is_control"] = values.isin({"true", "1"})
    if not out["is_control"].any():
        raise ValueError("measure_meta must contain at least one control measure.")
    mixed = out.groupby("embedding_id", observed=True)["is_control"].nunique()
    if (mixed > 1).any():
        embedding_id = str(mixed[mixed > 1].index[0])
        raise ValueError(f"embedding_id {embedding_id!r} mixes control and non-control measures.")
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class TrajectoryData:
    """One canonical longitudinal object for endpoint, multitime, and effect data."""

    axis: Axis
    measures: Mapping[str, Mapping[str, FiniteMeasure]]
    measure_meta: pd.DataFrame
    mass_semantics: MassSemantics
    count_blocks: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    representation: RepresentationArtifact | None = None

    def __post_init__(self) -> None:
        measure_meta = validate_measure_meta(self.measure_meta)
        semantics = MassSemantics(self.mass_semantics)
        measures = {
            str(label): MappingProxyType({str(key): value for key, value in by_id.items()})
            for label, by_id in self.measures.items()
        }
        object.__setattr__(self, "measure_meta", measure_meta)
        object.__setattr__(self, "mass_semantics", semantics)
        object.__setattr__(self, "measures", MappingProxyType(measures))
        object.__setattr__(self, "count_blocks", tuple(self.count_blocks))
        metadata = dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        representation = self.representation
        if representation is None:
            latent_hash = str(metadata.get("input_hashes", {}).get("support", ""))
            if len(latent_hash) != 64:
                latent_hash = _support_sha256(measures)
            representation = RepresentationArtifact(
                representation_id=f"frozen-latent:{latent_hash[:12]}",
                backend="frozen_latent",
                latent_dim=next(
                    measure.latent_dim
                    for by_measure in measures.values()
                    for measure in by_measure.values()
                ),
                latent_cache_hash=latent_hash,
                fit_scope="external",
                included_samples=tuple(dict.fromkeys(measure_meta["sample_id"].tolist())),
                included_time_labels=tuple(self.axis.labels),
                producer={"source": "canonical_dataset"},
            )
        object.__setattr__(self, "representation", representation)
        self.validate()

    def validate(self) -> None:
        if set(self.measures) != set(self.axis.labels):
            raise ValueError("TrajectoryData.measures must contain exactly the axis labels.")
        source_ids = set(self.measures[self.axis.source])
        metadata_ids = set(self.measure_meta["measure_id"])
        if not source_ids:
            raise ValueError("TrajectoryData requires source support.")
        if source_ids != metadata_ids:
            missing = sorted(metadata_ids - source_ids)[:5]
            extra = sorted(source_ids - metadata_ids)[:5]
            raise ValueError(
                "Every metadata measure must have source support and no source key may be unknown; "
                f"missing={missing}, extra={extra}."
            )
        latent_dims: set[int] = set()
        for label in self.axis.labels:
            downstream_ids = set(self.measures[label])
            if not downstream_ids <= source_ids:
                unknown = sorted(downstream_ids - source_ids)[:5]
                raise ValueError(
                    f"Checkpoint {label!r} has measures without source support: {unknown}"
                )
            latent_dims.update(measure.latent_dim for measure in self.measures[label].values())
        if len(latent_dims) != 1:
            raise ValueError("All finite measures must share one latent dimension.")
        if self.representation.latent_dim != next(iter(latent_dims)):
            raise ValueError(
                "RepresentationArtifact.latent_dim disagrees with finite-measure support."
            )
        if self.axis.kind == "effect" and self.count_blocks:
            raise ValueError("Count likelihood is unavailable on a nonphysical effect axis.")
        expected_by_group = {
            str(group): set(int(index) for index in indices)
            for group, indices in self.measure_meta.groupby(
                "context_group_id", observed=True
            ).indices.items()
        }
        seen_blocks: set[tuple[str, str]] = set()
        for block in self.count_blocks:
            group_id = str(block.context_group_id)
            time_label = str(block.time_label)
            self.axis.index(time_label)
            block_key = (group_id, time_label)
            if block_key in seen_blocks:
                raise ValueError(f"Duplicate CountBlock for context/time {block_key!r}.")
            seen_blocks.add(block_key)
            observed = set(int(value) for value in block.measure_indices.tolist())
            if group_id not in expected_by_group or observed != expected_by_group[group_id]:
                raise ValueError(
                    "CountBlock denominator must contain every source-supported measure "
                    f"in context group {group_id!r}."
                )
        if self.mass_semantics is MassSemantics.UNIT:
            nonunit = [
                (label, measure_id)
                for label, by_measure in self.measures.items()
                for measure_id, measure in by_measure.items()
                if not np.isclose(measure.total_mass, 1.0)
            ]
            if nonunit:
                raise ValueError(
                    "unit mass semantics requires every finite measure mass to equal 1."
                )

    @property
    def measure_ids(self) -> tuple[str, ...]:
        return tuple(self.measure_meta["measure_id"].tolist())

    @property
    def latent_dim(self) -> int:
        return next(iter(self.measures[self.axis.source].values())).latent_dim

    @property
    def embedding_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.measure_meta["embedding_id"].tolist()))

    @property
    def support(self) -> SupportStore:
        """Recipe-neutral support view keyed by checkpoint and measure ID."""
        return MappingProxyType(
            {
                label: MappingProxyType(
                    {measure_id: measure.support for measure_id, measure in by_measure.items()}
                )
                for label, by_measure in self.measures.items()
            }
        )

    @property
    def masses(self) -> MassTable:
        """Canonical one-row-per-measure/checkpoint mass table."""
        return pd.DataFrame(
            [
                {
                    "measure_id": measure_id,
                    "time_label": label,
                    "mass": measure.total_mass,
                    "mass_semantics": self.mass_semantics.value,
                }
                for label, by_measure in self.measures.items()
                for measure_id, measure in by_measure.items()
            ]
        )

    @property
    def counts(self) -> tuple[Any, ...]:
        return self.count_blocks

    @property
    def control_embedding_ids(self) -> tuple[str, ...]:
        values = self.measure_meta.loc[self.measure_meta["is_control"], "embedding_id"]
        return tuple(dict.fromkeys(values.tolist()))

    def metadata_for(self, measure_id: str) -> pd.Series:
        rows = self.measure_meta[self.measure_meta["measure_id"].eq(str(measure_id))]
        if len(rows) != 1:
            raise KeyError(f"Unknown measure_id {measure_id!r}.")
        return rows.iloc[0]

    def available(self, label: str) -> tuple[str, ...]:
        if label not in self.measures:
            raise KeyError(f"Unknown checkpoint label {label!r}.")
        return tuple(self.measures[label])

    def available_measure_ids(self, label: str) -> tuple[str, ...]:
        """Return available IDs in canonical metadata order."""
        available = set(self.available(label))
        return tuple(measure_id for measure_id in self.measure_ids if measure_id in available)

    @property
    def claim_policy(self) -> dict[str, str | bool]:
        return {
            "axis_kind": self.axis.kind,
            "claim_level": (
                "physical_trajectory" if self.axis.kind == "physical" else "single_time_effect_path"
            ),
            "physical_interpolation": self.axis.kind == "physical",
            "absolute_growth": (
                self.axis.kind == "physical" and self.mass_semantics.permits_absolute_growth_claim
            ),
            "relative_abundance": (
                self.axis.kind == "physical"
                and self.mass_semantics.permits_relative_abundance_claim
            ),
            "abundance_claim": (
                self.mass_semantics.abundance_claim if self.axis.kind == "physical" else "none"
            ),
            "count_likelihood": self.axis.kind == "physical" and bool(self.count_blocks),
        }

    def require_mass_claim(self, claim: str, *, absolute: bool = False) -> None:
        self.axis.require_physical(claim)
        if absolute and not self.mass_semantics.permits_absolute_growth_claim:
            raise ValueError(
                f"{claim} requires absolute mass; observed semantics are "
                f"{self.mass_semantics.value!r}."
            )
        if not absolute and not self.mass_semantics.permits_relative_abundance_claim:
            raise ValueError(
                f"{claim} requires informative mass; observed semantics are "
                f"{self.mass_semantics.value!r}."
            )


# ``TrajectoryData`` remains a compatibility name; all recipes consume this
# canonical study object through its stable CREDO name.
CREDOStudy = TrajectoryData
