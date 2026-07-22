"""Compatibility codec from the current five-file schema to :class:`Study`."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd

from ..contracts import (
    Axis,
    MassSemantics,
    RepresentationArtifact,
    TrajectoryData,
    validate_measure_meta,
)
from ..io import (
    DataConfig,
    H5ADFiniteMeasureStore,
    RunConfig,
    _build_count_blocks,
    _build_measures,
    _load_dataset_manifest,
    _normalize_count_table,
    _read_support,
    _sha256,
    _validate_denominators,
    _validate_mass_table,
    load_config,
)
from .design import AxisSpec, Checkpoint, StudyDesign, Transition
from .representations import ArtifactRef, RepresentationCatalog, RepresentationSpec
from .study import Study, StudyManifest
from .support import LegacyFiniteMeasureSupportStore
from .tables import (
    AbundanceChannelSpec,
    AbundanceSemantics,
    AbundanceTable,
    CompositionTable,
    ConditionTable,
    ObservationTable,
    SeriesTable,
    SupportIndexTable,
)

VerifyLevel = Literal["none", "schema", "manifest", "semantic", "full"]


def observation_id(series_id: str, checkpoint_id: str) -> str:
    """Construct a reversible, collision-safe observation identifier."""
    safe = ":._-"
    return f"{quote(str(series_id), safe=safe)}@{quote(str(checkpoint_id), safe=safe)}"


def _composition_block_id(context_id: str, checkpoint_id: str) -> str:
    safe = ":._-"
    return f"{quote(str(context_id), safe=safe)}@{quote(str(checkpoint_id), safe=safe)}"


def _design(axis: Axis) -> StudyDesign:
    axis_spec = AxisSpec(
        axis_id="primary",
        kind="physical_time" if axis.kind == "physical" else "effect",
        unit=None,
    )
    checkpoints = tuple(
        Checkpoint(
            checkpoint_id=label,
            coordinates={"primary": value},
            role=(
                "source"
                if index == 0
                else "target"
                if index == len(axis.labels) - 1
                else "intermediate"
            ),
        )
        for index, (label, value) in enumerate(zip(axis.labels, axis.values, strict=True))
    )
    transitions = tuple(
        Transition(
            transition_id=f"{left}_to_{right}",
            source_checkpoint_id=left,
            target_checkpoint_id=right,
        )
        for left, right in zip(axis.labels[:-1], axis.labels[1:], strict=True)
    )
    return StudyDesign(
        axes=(axis_spec,),
        checkpoints=checkpoints,
        transitions=transitions,
        topology="chain",
    )


def _condition_table(metadata: pd.DataFrame) -> ConditionTable:
    grouped = metadata.groupby("perturbation_id", observed=True, sort=False)
    required_consistent = ("embedding_id", "is_control")
    for column in required_consistent:
        inconsistent = grouped[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            condition_id = str(inconsistent[inconsistent].index[0])
            raise ValueError(
                f"Legacy condition {condition_id!r} has inconsistent {column!r} values."
            )
    first = grouped.nth(0).reset_index()
    reference_embeddings = tuple(
        dict.fromkeys(metadata.loc[metadata["is_control"], "embedding_id"].astype(str))
    )
    shared_reference = reference_embeddings[0] if len(reference_embeddings) == 1 else None
    frame = pd.DataFrame(
        {
            "condition_id": first["perturbation_id"].astype(str),
            "condition_kind": "legacy_condition",
            "embedding_id": first["embedding_id"].astype(str),
            "reference_group_id": [
                shared_reference
                or (str(embedding_id) if bool(is_control) else "__unspecified_reference__")
                for embedding_id, is_control in zip(
                    first["embedding_id"], first["is_control"], strict=True
                )
            ],
            "is_reference": first["is_control"].astype(bool),
        }
    )
    excluded = {
        "measure_id",
        "sample_id",
        "perturbation_id",
        "embedding_id",
        "context_group_id",
        "is_control",
    }
    for column in metadata.columns:
        if column in excluded or column in frame:
            continue
        if grouped[column].nunique(dropna=False).le(1).all():
            values = grouped[column].agg(lambda series: series.iloc[0]).reset_index(drop=True)
            frame[column] = values
    return ConditionTable(frame)


def _series_table(metadata: pd.DataFrame) -> SeriesTable:
    frame = pd.DataFrame(
        {
            "series_id": metadata["measure_id"].astype(str),
            "condition_id": metadata["perturbation_id"].astype(str),
            "subject_id": metadata["sample_id"].astype(str),
            "embedding_id": metadata["embedding_id"].astype(str),
            "reference_role": np.where(metadata["is_control"], "reference", "intervention"),
        }
    )
    excluded = {
        "measure_id",
        "sample_id",
        "perturbation_id",
        "embedding_id",
        "context_group_id",
        "is_control",
    }
    for column in metadata.columns:
        if column not in excluded and column not in frame:
            frame[column] = metadata[column].to_numpy()
    return SeriesTable(frame)


def _raw_table(data: TrajectoryData, name: str) -> pd.DataFrame | None:
    path = data.metadata.get("input_paths", {}).get(name)
    if path is None or not Path(path).is_file():
        return None
    return pd.read_parquet(path)


def _mass_table(data: TrajectoryData, masses: pd.DataFrame | None) -> pd.DataFrame:
    if masses is not None:
        return masses.copy()
    disk = _raw_table(data, "masses")
    if disk is not None:
        return disk
    frame = data.masses.rename(columns={"mass": "value"})
    context = data.measure_meta.set_index("measure_id")["context_group_id"].to_dict()
    frame["denominator"] = [
        f"legacy::{context[measure_id]}::{checkpoint_id}"
        for measure_id, checkpoint_id in zip(frame["measure_id"], frame["time_label"], strict=True)
    ]
    return frame.rename(columns={"value": "mass"})


def _count_table(data: TrajectoryData, counts: pd.DataFrame | None) -> pd.DataFrame | None:
    if counts is not None:
        return counts.copy()
    disk = _raw_table(data, "counts")
    if disk is not None:
        return disk
    if not data.count_blocks:
        return None
    rows: list[dict[str, Any]] = []
    for block in data.count_blocks:
        for index, exposure, count in zip(
            block.measure_indices, block.exposure, block.counts, strict=True
        ):
            rows.append(
                {
                    "context_group_id": str(block.context_group_id),
                    "time_label": str(block.time_label),
                    "measure_id": data.measure_ids[int(index)],
                    "exposure": float(exposure),
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def _abundance_spec(semantics: MassSemantics) -> AbundanceChannelSpec:
    mapped = {
        MassSemantics.ABSOLUTE: AbundanceSemantics.ABSOLUTE,
        MassSemantics.RELATIVE_WITHIN_GROUP: AbundanceSemantics.RELATIVE,
        MassSemantics.CAPTURED_COUNT: AbundanceSemantics.CAPTURE_COUNT,
        MassSemantics.UNIT: AbundanceSemantics.UNIT,
    }[semantics]
    return AbundanceChannelSpec(
        channel_id="legacy_mass",
        semantics=mapped,
        unit="count" if semantics is MassSemantics.CAPTURED_COUNT else None,
        denominator_scope=(
            "context_checkpoint"
            if semantics in {MassSemantics.RELATIVE_WITHIN_GROUP, MassSemantics.CAPTURED_COUNT}
            else "none"
        ),
        zero_policy="forbidden",
    )


def _artifact_ref(
    artifact_id: str,
    digest: str | None,
    *,
    path: str | None = None,
    media_type: str = "application/octet-stream",
    semantic_hash: str | None = None,
) -> ArtifactRef | None:
    if digest is None:
        if semantic_hash is None:
            return None
        digest, semantic_hash = semantic_hash, None
    resolved = Path(path).expanduser().resolve() if path is not None else None
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=str(resolved) if resolved is not None else f"urn:sha256:{digest}",
        sha256=digest,
        size_bytes=resolved.stat().st_size if resolved is not None and resolved.is_file() else None,
        media_type=media_type,
        semantic_hash=semantic_hash,
    )


@dataclass(frozen=True)
class _ResolvedFiveFile:
    data: DataConfig
    axis: Axis
    manifest: Mapping[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class _LegacyComponents:
    axis: Axis
    measures: Mapping[str, Mapping[str, Any]]
    measure_meta: pd.DataFrame
    mass_semantics: MassSemantics
    count_blocks: tuple[Any, ...]
    metadata: Mapping[str, Any]
    representation: RepresentationArtifact

    @property
    def measure_ids(self) -> tuple[str, ...]:
        return tuple(self.measure_meta["measure_id"].tolist())


def _resolved_source(
    source: RunConfig | str | Path,
    *,
    lazy_support: bool | None,
    support_cache_size: int | None,
) -> _ResolvedFiveFile:
    if isinstance(source, RunConfig):
        config = source
        updates = {}
        if lazy_support is not None:
            updates["lazy_support"] = bool(lazy_support)
        if support_cache_size is not None:
            updates["support_cache_size"] = int(support_cache_size)
        data = config.data.model_copy(update=updates)
        axis = config.axis.build()
    else:
        path = Path(source).expanduser().resolve()
        if path.suffix.lower() in {".yaml", ".yml"}:
            config = load_config(path)
            updates = {}
            if lazy_support is not None:
                updates["lazy_support"] = bool(lazy_support)
            if support_cache_size is not None:
                updates["support_cache_size"] = int(support_cache_size)
            data = config.data.model_copy(update=updates)
            axis = config.axis.build()
        else:
            manifest_path = path / "dataset.json" if path.is_dir() else path
            if manifest_path.name != "dataset.json" or not manifest_path.is_file():
                raise ValueError(
                    "Current five-file studies must be opened from a run YAML, dataset.json, "
                    "or its containing directory."
                )
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping) or not isinstance(raw.get("axis"), Mapping):
                raise ValueError("dataset.json must declare an axis object.")
            axis = Axis(**raw["axis"])
            directory = manifest_path.parent
            counts_path = directory / "counts.parquet"
            data = DataConfig(
                support=directory / "support.h5ad",
                latent_key=str(raw.get("latent_key", "")),
                measure_meta=directory / "measure_meta.parquet",
                masses=directory / "masses.parquet",
                counts=counts_path if counts_path.is_file() else None,
                dataset=manifest_path,
                lazy_support=True if lazy_support is None else lazy_support,
                support_cache_size=(256 if support_cache_size is None else support_cache_size),
            )
    manifest, manifest_path = _load_dataset_manifest(data, axis)
    return _ResolvedFiveFile(data, axis, manifest, manifest_path)


def _representation_artifact(
    resolved: _ResolvedFiveFile,
    *,
    latent_dim: int,
    verify_hashes: bool,
) -> RepresentationArtifact:
    payload = resolved.manifest.get("representation")
    if isinstance(payload, Mapping):
        artifact = RepresentationArtifact.from_dict(payload)
    else:
        digest = (
            _sha256(resolved.data.support)
            if verify_hashes
            else hashlib.sha256(str(resolved.data.support).encode("utf-8")).hexdigest()
        )
        artifact = RepresentationArtifact(
            representation_id=f"frozen-latent:{digest[:12]}",
            backend="frozen_latent",
            latent_dim=latent_dim,
            latent_cache_hash=digest,
            fit_scope="external",
            producer={"source": "five_file_v1"},
        )
    if artifact.latent_dim != latent_dim:
        raise ValueError("Representation latent dimension disagrees with support.h5ad.")
    if (
        verify_hashes
        and artifact.producer.get("latent_cache_hash_kind") == "support_h5ad_file_sha256"
        and _sha256(resolved.data.support) != artifact.latent_cache_hash
    ):
        raise ValueError("Representation latent-cache hash disagrees with support.h5ad.")
    return artifact


def _read_five_file_study(
    codec: CurrentFiveFileStudyCodec,
    source: RunConfig | str | Path,
    *,
    verify: VerifyLevel,
    lazy_support: bool | None,
    support_cache_size: int | None,
) -> Study:
    resolved = _resolved_source(
        source,
        lazy_support=lazy_support,
        support_cache_size=support_cache_size,
    )
    rank = {"none": 0, "schema": 1, "manifest": 2, "semantic": 3, "full": 4}[verify]
    cache_size = resolved.data.support_cache_size
    eager_support = rank >= 3 and not resolved.data.lazy_support
    measure_meta = validate_measure_meta(pd.read_parquet(resolved.data.measure_meta))
    observations, latent, latent_dim = _read_support(
        resolved.data.support,
        resolved.data.latent_key,
        lazy=not eager_support,
        scan_values=verify == "full",
    )
    masses, mass_semantics = _validate_mass_table(pd.read_parquet(resolved.data.masses))
    declared_semantics = MassSemantics(resolved.manifest["mass_semantics"])
    if declared_semantics is not mass_semantics:
        raise ValueError("masses.parquet mass semantics disagree with dataset.json.")
    counts = (
        None
        if resolved.data.counts is None
        else _normalize_count_table(pd.read_parquet(resolved.data.counts))
    )
    input_paths = {
        "support": resolved.data.support,
        "measure_meta": resolved.data.measure_meta,
        "masses": resolved.data.masses,
        "dataset": resolved.manifest_path,
    }
    if resolved.data.counts is not None:
        input_paths["counts"] = resolved.data.counts
    input_hashes = {name: _sha256(path) for name, path in input_paths.items()} if rank >= 2 else {}
    representation = _representation_artifact(
        resolved,
        latent_dim=latent_dim,
        verify_hashes=rank >= 2,
    )
    metadata = {
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_hashes": input_hashes,
        "dataset": dict(resolved.manifest),
        "mass_denominators": sorted(masses["denominator"].unique().tolist()),
    }
    if rank >= 3:
        _validate_denominators(masses, measure_meta, mass_semantics)
        measures = _build_measures(
            observations,
            latent,
            masses,
            resolved.axis,
            support_path=resolved.data.support,
            latent_key=resolved.data.latent_key,
            latent_dim=latent_dim,
            cache_size=cache_size,
        )
        count_blocks = _build_count_blocks(
            counts,
            measure_meta,
            tuple(measure_meta["measure_id"].tolist()),
        )
        components: TrajectoryData | _LegacyComponents = TrajectoryData(
            axis=resolved.axis,
            measures=measures,
            measure_meta=measure_meta,
            mass_semantics=mass_semantics,
            count_blocks=count_blocks,
            metadata=metadata,
            representation=representation,
        )
    else:
        measures = H5ADFiniteMeasureStore(
            resolved.data.support,
            resolved.data.latent_key,
            observations,
            masses,
            resolved.axis,
            latent_dim=latent_dim,
            cache_size=cache_size,
        )
        components = _LegacyComponents(
            axis=resolved.axis,
            measures=measures,
            measure_meta=measure_meta,
            mass_semantics=mass_semantics,
            count_blocks=(),
            metadata=metadata,
            representation=representation,
        )
    study = codec.from_trajectory(components, masses=masses, counts=counts)
    study.validate(level=verify).raise_for_errors()
    return study


class CurrentFiveFileStudyCodec:
    """Read schema-v1/v2 adapter outputs into the semantic Study model."""

    codec_id = "credo.current_five_file"
    readable_schema_versions = frozenset({1, 2})
    writable_schema_versions: frozenset[int] = frozenset()

    def probe(self, source: Any) -> bool:
        if isinstance(source, (RunConfig, TrajectoryData)):
            return True
        try:
            path = Path(source).expanduser()
        except TypeError:
            return False
        if path.suffix.lower() in {".yaml", ".yml"}:
            return path.is_file()
        manifest = path / "dataset.json" if path.is_dir() else path
        return manifest.name == "dataset.json" and manifest.is_file()

    def write(self, study: Study, destination: str | Path) -> None:
        del study, destination
        raise NotImplementedError("The five-file compatibility codec is read-only.")

    def read(
        self,
        source: RunConfig | TrajectoryData | str | Path,
        *,
        verify: VerifyLevel = "semantic",
        lazy_support: bool | None = None,
        support_cache_size: int | None = None,
    ) -> Study:
        if verify not in {"none", "schema", "manifest", "semantic", "full"}:
            raise ValueError(f"Unknown verification level {verify!r}.")
        if isinstance(source, TrajectoryData):
            study = self.from_trajectory(source)
            study.validate(level=verify).raise_for_errors()
            return study
        return _read_five_file_study(
            self,
            source,
            verify=verify,
            lazy_support=lazy_support,
            support_cache_size=support_cache_size,
        )

    def from_trajectory(
        self,
        data: TrajectoryData | _LegacyComponents,
        *,
        masses: pd.DataFrame | None = None,
        counts: pd.DataFrame | None = None,
        study_id: str | None = None,
    ) -> Study:
        """Normalize one current runtime object without changing its finite measures."""
        metadata = data.measure_meta.copy()
        design = _design(data.axis)
        conditions = _condition_table(metadata)
        series = _series_table(metadata)
        raw_counts = _count_table(data, counts)
        count_blocks = set()
        if raw_counts is not None:
            count_blocks = set(
                zip(
                    raw_counts["context_group_id"].astype(str),
                    raw_counts["time_label"].astype(str),
                    strict=False,
                )
            )
        geometry_pairs = {
            (str(series_id), str(checkpoint_id))
            for checkpoint_id in data.axis.labels
            for series_id in data.measures[checkpoint_id]
        }
        observation_rows: list[dict[str, Any]] = []
        support_pairs: dict[str, tuple[str, str]] = {}
        metadata_lookup = metadata.set_index("measure_id")
        for series_id in data.measure_ids:
            row = metadata_lookup.loc[series_id]
            for checkpoint_id in data.axis.labels:
                identifier = observation_id(series_id, checkpoint_id)
                geometry_observed = (series_id, checkpoint_id) in geometry_pairs
                support_key = identifier if geometry_observed else None
                if support_key is not None:
                    support_pairs[support_key] = (checkpoint_id, series_id)
                context_id = str(row["context_group_id"])
                block_id = (
                    _composition_block_id(context_id, checkpoint_id)
                    if (context_id, checkpoint_id) in count_blocks
                    else None
                )
                observation_rows.append(
                    {
                        "observation_id": identifier,
                        "series_id": series_id,
                        "checkpoint_id": checkpoint_id,
                        "sample_id": str(row["sample_id"]),
                        "geometry_observed": geometry_observed,
                        "context_id": context_id,
                        "composition_block_id": block_id,
                        "legacy_measure_id": series_id,
                    }
                )
        observations = ObservationTable(pd.DataFrame(observation_rows))
        raw_masses = _mass_table(data, masses)
        abundance_rows = pd.DataFrame(
            {
                "observation_id": [
                    observation_id(series_id, checkpoint_id)
                    for series_id, checkpoint_id in zip(
                        raw_masses["measure_id"], raw_masses["time_label"], strict=True
                    )
                ],
                "channel_id": "legacy_mass",
                "value": pd.to_numeric(raw_masses["mass"], errors="raise"),
                "observed": True,
                "denominator_id": raw_masses.get("denominator"),
                "source_artifact_id": "legacy_masses",
            }
        )
        abundance_spec = _abundance_spec(data.mass_semantics)
        abundance = AbundanceTable(abundance_rows, (abundance_spec,))
        compositions = None
        if raw_counts is not None:
            composition_rows = pd.DataFrame(
                {
                    "composition_block_id": [
                        _composition_block_id(context_id, checkpoint_id)
                        for context_id, checkpoint_id in zip(
                            raw_counts["context_group_id"],
                            raw_counts["time_label"],
                            strict=True,
                        )
                    ],
                    "checkpoint_id": raw_counts["time_label"].astype(str),
                    "context_id": raw_counts["context_group_id"].astype(str),
                    "series_id": raw_counts["measure_id"].astype(str),
                    "observation_id": [
                        observation_id(series_id, checkpoint_id)
                        for series_id, checkpoint_id in zip(
                            raw_counts["measure_id"], raw_counts["time_label"], strict=True
                        )
                    ],
                    "exposure": raw_counts["exposure"],
                    "count": raw_counts["count"],
                    "denominator_id": [
                        _composition_block_id(context_id, checkpoint_id)
                        for context_id, checkpoint_id in zip(
                            raw_counts["context_group_id"],
                            raw_counts["time_label"],
                            strict=True,
                        )
                    ],
                }
            )
            compositions = CompositionTable(composition_rows)
        representation = data.representation
        input_paths = data.metadata.get("input_paths", {})
        input_hashes = data.metadata.get("input_hashes", {})
        support_hash = input_hashes.get("support", representation.latent_cache_hash)
        store_id = f"five-file:{str(support_hash)[:12]}"
        support_artifact = _artifact_ref(
            "legacy_support",
            support_hash,
            path=input_paths.get("support"),
            media_type="application/x-hdf5",
        )
        included_samples = set(representation.included_samples)
        included_series = tuple(
            metadata.loc[metadata["sample_id"].astype(str).isin(included_samples), "measure_id"]
            .astype(str)
            .tolist()
        )
        representation_spec = RepresentationSpec(
            representation_id=representation.representation_id,
            backend=representation.backend,
            space_kind="latent",
            dimension=representation.latent_dim,
            support_store_id=store_id,
            support_artifact=support_artifact,
            feature_artifact=_artifact_ref(
                "legacy_features",
                representation.gene_names_hash,
                semantic_hash=representation.gene_mask_hash,
            ),
            encoder_artifact=_artifact_ref("legacy_encoder", representation.encoder_state_hash),
            decoder_artifact=_artifact_ref("legacy_decoder", representation.decoder_state_hash),
            normalization_artifact=_artifact_ref(
                "legacy_normalization", representation.normalization_hash
            ),
            included_series=included_series,
            included_checkpoints=tuple(representation.included_time_labels),
        )
        catalog = RepresentationCatalog((representation_spec,))
        supports = LegacyFiniteMeasureSupportStore(
            store_id=store_id,
            representation_id=representation.representation_id,
            latent_dim=representation.latent_dim,
            measures=data.measures,
            support_pairs=support_pairs,
        )
        support_index = SupportIndexTable(
            pd.DataFrame(
                [
                    {
                        "observation_id": row["observation_id"],
                        "representation_id": representation.representation_id,
                        "store_id": store_id if row["geometry_observed"] else None,
                        "support_key": (
                            row["observation_id"] if row["geometry_observed"] else None
                        ),
                        "available": row["geometry_observed"],
                    }
                    for row in observation_rows
                ]
            )
        )
        dataset = data.metadata.get("dataset", {})
        inferred_study_id = study_id or _infer_study_id(input_paths, dataset)
        manifest = StudyManifest(
            schema_version=3,
            study_id=inferred_study_id,
            source_schema=f"five_file_v{dataset.get('schema_version', 'unknown')}",
            primary_representation=representation.representation_id,
            primary_abundance_channel="legacy_mass",
            description=str(dataset.get("description", "")),
        )
        return Study(
            manifest=manifest,
            design=design,
            conditions=conditions,
            series=series,
            observations=observations,
            support_index=support_index,
            abundance=abundance,
            compositions=compositions,
            representations=catalog,
            supports=supports,
            provenance={
                "codec": self.codec_id,
                "legacy_dataset": dataset,
                "legacy_representation": representation.to_dict(),
                "legacy_included_samples": list(representation.included_samples),
                "input_paths": dict(input_paths),
                "input_hashes": dict(input_hashes),
                "mass_denominators": list(data.metadata.get("mass_denominators", ())),
            },
        )


def _infer_study_id(input_paths: Mapping[str, str], dataset: Mapping[str, Any]) -> str:
    dataset_path = input_paths.get("dataset")
    if dataset_path:
        name = Path(dataset_path).resolve().parent.name
        if name:
            return name
    source = dataset.get("source", {})
    source_input = source.get("input") if isinstance(source, Mapping) else None
    if source_input:
        return Path(str(source_input)).stem
    payload = json.dumps(dataset, sort_keys=True, default=str).encode("utf-8")
    return f"legacy-{hashlib.sha256(payload).hexdigest()[:12]}"


FiveFileV2Codec = CurrentFiveFileStudyCodec


def open_study(
    source: RunConfig | TrajectoryData | str | Path,
    *,
    verify: VerifyLevel = "semantic",
    lazy_support: bool | None = None,
    support_cache_size: int | None = None,
) -> Study:
    """Compatibility import for the registry-backed public loader."""
    from .codecs import open_study as open_registered_study

    return open_registered_study(
        source,
        verify=verify,
        lazy_support=lazy_support,
        support_cache_size=support_cache_size,
    )


__all__ = [
    "CurrentFiveFileStudyCodec",
    "FiveFileV2Codec",
    "observation_id",
    "open_study",
]
