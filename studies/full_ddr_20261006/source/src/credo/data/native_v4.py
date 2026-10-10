"""Transactional native schema-v4 persistence for longitudinal Perturb-seq."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlparse

import h5py
import numpy as np
import pandas as pd

from ..lps.study import PerturbSeqManifest, PerturbSeqStudy
from .design import Checkpoint, LongitudinalDesign, ProgressionAxis, Transition
from .representations import ArtifactRef, RepresentationCatalog, RepresentationSpec
from .study import VerificationLevel, _canonical_value
from .support import EmpiricalLaw, SupportRef, SupportStoreRegistry
from .tables import (
    AbundanceChannelSpec,
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
from .validation import ValidationIssue, ValidationReport

_FORMAT = "credo.native_perturb_seq_study"
_SCHEMA_VERSION = 4
_PACKED_WRITE_ATOMS = 262_144
_PACKED_WRITE_LAWS = 4_096
_PACKED_VERIFY_LAWS = 4_096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _flush_packed_batch(
    coordinates: h5py.Dataset,
    probabilities: h5py.Dataset,
    indptr: h5py.Dataset,
    dimension: int,
    atom_count: int,
    coordinate_batch: list[np.ndarray],
    probability_batch: list[np.ndarray],
    length_batch: list[int],
) -> int:
    if not length_batch:
        return atom_count
    coordinate_block = np.concatenate(coordinate_batch, axis=0)
    probability_block = np.concatenate(probability_batch)
    stop = atom_count + len(coordinate_block)
    coordinates.resize((stop, dimension))
    probabilities.resize((stop,))
    coordinates[atom_count:stop] = coordinate_block
    probabilities[atom_count:stop] = probability_block
    law_offset = len(indptr) - 1
    cumulative = atom_count + np.cumsum(length_batch, dtype=np.int64)
    indptr.resize((law_offset + len(length_batch) + 1,))
    indptr[law_offset + 1 : law_offset + len(length_batch) + 1] = cumulative
    coordinate_batch.clear()
    probability_batch.clear()
    length_batch.clear()
    return stop


def _artifact_payload(artifact: ArtifactRef | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "artifact_id": artifact.artifact_id,
        "uri": artifact.uri,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "media_type": artifact.media_type,
        "semantic_hash": artifact.semantic_hash,
    }


_REPRESENTATION_ARTIFACTS = (
    "support_artifact",
    "feature_artifact",
    "feature_selection_artifact",
    "encoder_artifact",
    "decoder_artifact",
    "normalization_artifact",
)


def _representation_payload(spec: RepresentationSpec) -> dict[str, Any]:
    return {
        "representation_id": spec.representation_id,
        "backend": spec.backend,
        "space_kind": spec.space_kind,
        "dimension": spec.dimension,
        "support_store_id": spec.support_store_id,
        **{name: _artifact_payload(getattr(spec, name)) for name in _REPRESENTATION_ARTIFACTS},
        "scope_mode": spec.scope_mode,
        "fit_split_id": spec.fit_split_id,
        "fit_selection_hash": spec.fit_selection_hash,
        "fit_subject_ids": list(spec.fit_subject_ids),
        "fit_perturbation_ids": list(spec.fit_perturbation_ids),
        "fit_checkpoint_ids": list(spec.fit_checkpoint_ids),
        "fit_observation_scope": spec.fit_observation_scope,
        "included_series": list(spec.included_series),
        "included_checkpoints": list(spec.included_checkpoints),
    }


def _representation_from_payload(payload: Mapping[str, Any]) -> RepresentationSpec:
    raw = dict(payload)
    for name in _REPRESENTATION_ARTIFACTS:
        if raw.get(name) is not None:
            raw[name] = ArtifactRef(**raw[name])
    for name in (
        "fit_subject_ids",
        "fit_perturbation_ids",
        "fit_checkpoint_ids",
        "included_series",
        "included_checkpoints",
    ):
        raw[name] = tuple(raw.get(name, ()))
    return RepresentationSpec(**raw)


def _persist_representation_artifacts(
    study: PerturbSeqStudy,
    temporary: Path,
) -> list[dict[str, Any]]:
    directory = temporary / "representations"
    copied: dict[str, str] = {}

    def persist(artifact: ArtifactRef | None, *, support: bool = False) -> dict[str, Any] | None:
        value = _artifact_payload(artifact)
        if value is None:
            return None
        assert artifact is not None
        if support:
            value["uri"] = f"urn:sha256:{artifact.sha256}"
            return value
        parsed = urlparse(artifact.uri)
        if parsed.scheme and parsed.scheme != "file":
            return value
        source = Path(
            unquote(parsed.path) if parsed.scheme == "file" else artifact.uri
        ).expanduser()
        if not source.is_file():
            raise FileNotFoundError(
                f"Local representation artifact {artifact.artifact_id!r} is missing: {source}"
            )
        source = source.resolve()
        if _sha256(source) != artifact.sha256:
            raise ValueError(f"Representation artifact {artifact.artifact_id!r} hash mismatch.")
        if artifact.size_bytes is not None and source.stat().st_size != artifact.size_bytes:
            raise ValueError(f"Representation artifact {artifact.artifact_id!r} size mismatch.")
        relative = copied.get(artifact.sha256)
        if relative is None:
            directory.mkdir(exist_ok=True)
            suffix = source.suffix or ".bin"
            relative = f"representations/{artifact.sha256}{suffix}"
            shutil.copy2(source, temporary / relative)
            copied[artifact.sha256] = relative
        value["uri"] = relative
        return value

    payloads = []
    for representation_id in sorted(study.representations):
        spec = study.representations[representation_id]
        value = _representation_payload(spec)
        for name in _REPRESENTATION_ARTIFACTS:
            value[name] = persist(getattr(spec, name), support=name == "support_artifact")
        payloads.append(value)
    return payloads


def _design_payload(design: LongitudinalDesign) -> dict[str, Any]:
    return {
        "axis": {
            "axis_id": design.axis.axis_id,
            "kind": design.axis.kind,
            "unit": design.axis.unit,
        },
        "checkpoints": [
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "coordinates": dict(checkpoint.coordinates),
                "role": checkpoint.role,
            }
            for checkpoint in design.checkpoints
        ],
        "transitions": [
            {
                "transition_id": transition.transition_id,
                "source_checkpoint_id": transition.source_checkpoint_id,
                "target_checkpoint_id": transition.target_checkpoint_id,
            }
            for transition in design.transitions
        ],
        "topology": design.topology,
    }


def _design_from_payload(payload: Mapping[str, Any]) -> LongitudinalDesign:
    return LongitudinalDesign(
        axis=ProgressionAxis(**payload["axis"]),
        checkpoints=tuple(Checkpoint(**value) for value in payload["checkpoints"]),
        transitions=tuple(Transition(**value) for value in payload["transitions"]),
        topology=payload["topology"],
    )


class NativePackedH5SupportStore:
    """Lazy packed support backend with one resizable array per representation."""

    is_lazy = True

    def __init__(
        self,
        store_id: str,
        path: Path,
        index: pd.DataFrame,
        dimensions: Mapping[str, int],
        groups: Mapping[str, str],
        semantic_identities: Mapping[str, Mapping[str, str]],
    ) -> None:
        required = {"representation_id", "support_key", "law_index"}
        missing = required - set(index)
        if missing:
            raise ValueError(f"Packed support index is missing columns: {sorted(missing)}")
        if index.duplicated(["representation_id", "support_key"]).any():
            raise ValueError("Packed support index contains duplicate qualified support keys.")
        self._store_id = str(store_id)
        self.path = Path(path)
        self._index = MappingProxyType(
            {
                (str(row.representation_id), str(row.support_key)): int(row.law_index)
                for row in index.itertuples(index=False)
            }
        )
        self._dimensions = MappingProxyType(
            {str(key): int(value) for key, value in dimensions.items()}
        )
        self._groups = MappingProxyType({str(key): str(value) for key, value in groups.items()})
        if set(self._dimensions) != set(self._groups):
            raise ValueError("Packed support group and dimension catalogs disagree.")
        self._semantic_identities = MappingProxyType(
            {str(key): MappingProxyType(dict(value)) for key, value in semantic_identities.items()}
        )
        self._lock = threading.RLock()
        self._handle: h5py.File | None = None
        self._closed = False

    @property
    def store_id(self) -> str:
        return self._store_id

    def representation_ids(self) -> tuple[str, ...]:
        return tuple(self._dimensions)

    def dimension(self, representation_id: str) -> int:
        return self._dimensions[str(representation_id)]

    def semantic_identity(self, representation_id: str) -> Mapping[str, str]:
        return self._semantic_identities[str(representation_id)]

    def contains(self, ref: SupportRef) -> bool:
        return (
            not self._closed
            and ref.store_id == self._store_id
            and (ref.representation_id, ref.support_key) in self._index
        )

    def _file(self) -> h5py.File:
        if self._closed:
            raise RuntimeError("Support store is closed.")
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def read(self, ref: SupportRef) -> EmpiricalLaw:
        if not self.contains(ref):
            raise KeyError(f"Unknown support reference {ref!r}.")
        law_index = self._index[(ref.representation_id, ref.support_key)]
        with self._lock:
            group = self._file()[f"representations/{self._groups[ref.representation_id]}"]
            start = int(group["indptr"][law_index])
            stop = int(group["indptr"][law_index + 1])
            coordinates = np.asarray(group["coordinates"][start:stop], dtype=np.float32)
            probabilities = np.asarray(group["probabilities"][start:stop], dtype=np.float64)
        return EmpiricalLaw(coordinates, probabilities)

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]:
        return MappingProxyType({ref: self.read(ref) for ref in refs})

    def validate(self, *, full_scan: bool = False) -> ValidationReport:
        if self._closed:
            return ValidationReport(
                (ValidationIssue("error", "support.closed", "Support store is closed."),)
            )
        issues: list[ValidationIssue] = []
        try:
            handle = self._file()
            unknown_representations = {
                representation_id for representation_id, _ in self._index
            } - set(self._dimensions)
            if unknown_representations:
                issues.append(
                    ValidationIssue(
                        "error",
                        "support.index",
                        "Packed support index references unknown representations: "
                        f"{sorted(unknown_representations)}.",
                    )
                )
            for representation_id, group_id in self._groups.items():
                path = f"representations/{group_id}"
                if path not in handle:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "support.missing",
                            f"Packed representation {representation_id!r} is absent.",
                        )
                    )
                    continue
                group = handle[path]
                coordinates = group["coordinates"]
                probabilities = group["probabilities"]
                indptr = np.asarray(group["indptr"][:], dtype=np.int64)
                dimension = self._dimensions[representation_id]
                entries = sorted(
                    (
                        law_index,
                        support_key,
                    )
                    for (indexed_representation, support_key), law_index in self._index.items()
                    if indexed_representation == representation_id
                )
                actual_indices = np.asarray([law_index for law_index, _ in entries], dtype=np.int64)
                expected_indices = np.arange(len(entries), dtype=np.int64)
                valid_index = np.array_equal(actual_indices, expected_indices)
                valid_indptr = (
                    len(indptr) == len(entries) + 1
                    and len(indptr) > 0
                    and indptr[0] == 0
                    and np.all(np.diff(indptr) > 0)
                    and indptr[-1] == len(coordinates)
                    and indptr[-1] == len(probabilities)
                )
                if coordinates.ndim != 2 or coordinates.shape[1] != dimension:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "support.dimension",
                            f"Packed representation {representation_id!r} has wrong dimension.",
                        )
                    )
                if not valid_index or not valid_indptr:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "support.index",
                            f"Packed representation {representation_id!r} has an invalid "
                            "law index or indptr.",
                        )
                    )
                    continue
                if not full_scan:
                    continue
                digest = hashlib.sha256()
                for offset in range(0, len(entries), _PACKED_VERIFY_LAWS):
                    batch = entries[offset : offset + _PACKED_VERIFY_LAWS]
                    atom_start = int(indptr[offset])
                    atom_stop = int(indptr[offset + len(batch)])
                    coordinate_block = np.asarray(
                        coordinates[atom_start:atom_stop],
                        dtype="<f4",
                        order="C",
                    )
                    probability_block = np.asarray(
                        probabilities[atom_start:atom_stop],
                        dtype="<f8",
                        order="C",
                    )
                    local_indptr = indptr[offset : offset + len(batch) + 1] - atom_start
                    if (
                        not np.isfinite(coordinate_block).all()
                        or not np.isfinite(probability_block).all()
                        or np.any(probability_block < 0)
                    ):
                        issues.append(
                            ValidationIssue(
                                "error",
                                "support.values",
                                f"Packed representation {representation_id!r} contains "
                                "invalid coordinates or probabilities.",
                            )
                        )
                        break
                    law_sums = np.add.reduceat(probability_block, local_indptr[:-1])
                    if not np.allclose(law_sums, 1.0, rtol=1e-7, atol=1e-10):
                        issues.append(
                            ValidationIssue(
                                "error",
                                "support.probabilities",
                                f"Packed representation {representation_id!r} contains "
                                "probabilities that do not sum to one.",
                            )
                        )
                        break
                    for index, (_, support_key) in enumerate(batch):
                        start = int(local_indptr[index])
                        stop = int(local_indptr[index + 1])
                        law_coordinates = coordinate_block[start:stop]
                        law_probabilities = probability_block[start:stop]
                        digest.update(self._store_id.encode())
                        digest.update(b"\0")
                        digest.update(support_key.encode())
                        digest.update(b"\0")
                        digest.update(np.asarray(law_coordinates.shape, dtype="<i8").tobytes())
                        digest.update(law_coordinates.tobytes(order="C"))
                        digest.update(law_probabilities.tobytes(order="C"))
                else:
                    expected = self._semantic_identities.get(representation_id, {}).get(
                        "materialized_sha256"
                    )
                    if expected != digest.hexdigest():
                        issues.append(
                            ValidationIssue(
                                "error",
                                "support.semantic_hash",
                                f"Packed support hash disagrees for {representation_id!r}.",
                            )
                        )
        except (IndexError, KeyError, OSError, ValueError) as exc:
            issues.append(ValidationIssue("error", "support.open", str(exc)))
        return ValidationReport(tuple(issues))

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._closed = True


@dataclass
class PerturbSeqStudyBuilder:
    """Typed assembly helper for native schema-v4 studies and external adapters."""

    manifest: PerturbSeqManifest
    design: LongitudinalDesign
    perturbations: PerturbationTable
    intervention_events: InterventionEventTable
    series: PopulationSeriesTable
    observations: SnapshotObservationTable
    support_index: SupportIndexTable
    representations: RepresentationCatalog
    supports: SupportStoreRegistry | Any
    perturbation_components: PerturbationComponentTable | None = None
    contexts: ContextTable | None = None
    abundance: AbundanceTable | None = None
    compositions: LPSCompositionTable | None = None
    population_pools: PopulationPoolTable | None = None
    effect_bindings: PerturbationEffectBindingTable | None = None
    reference_bindings: PerturbationReferenceBindingTable | None = None
    provenance: Mapping[str, Any] | None = None

    def build(self, *, verify: VerificationLevel = "semantic") -> PerturbSeqStudy:
        study = PerturbSeqStudy(
            manifest=self.manifest,
            design=self.design,
            perturbations=self.perturbations,
            perturbation_components=self.perturbation_components,
            intervention_events=self.intervention_events,
            contexts=self.contexts,
            series=self.series,
            observations=self.observations,
            representations=self.representations,
            support_index=self.support_index,
            supports=self.supports,
            abundance=self.abundance,
            compositions=self.compositions,
            population_pools=self.population_pools,
            effect_bindings=self.effect_bindings,
            reference_bindings=self.reference_bindings,
            provenance={} if self.provenance is None else self.provenance,
        )
        try:
            study.validate(level=verify).raise_for_errors()
        except Exception:
            study.close()
            raise
        return study


def _write_packed_store(
    study: PerturbSeqStudy,
    temporary: Path,
    store_id: str,
    store_number: int,
) -> dict[str, Any] | None:
    representations = tuple(
        value for value in study.representations.values() if value.support_store_id == store_id
    )
    if not representations:
        return None
    file_relative = f"stores/store-{store_number:04d}.h5"
    index_relative = f"stores/store-{store_number:04d}.parquet"
    dimensions = {value.representation_id: value.dimension for value in representations}
    groups = {
        value.representation_id: f"representation-{index:04d}"
        for index, value in enumerate(
            sorted(representations, key=lambda item: item.representation_id)
        )
    }
    semantic_digests = {value: hashlib.sha256() for value in dimensions}
    support_index = study.support_index._unsafe_view()
    index_rows: list[dict[str, Any]] = []
    with h5py.File(temporary / file_relative, "w") as handle:
        root = handle.create_group("representations")
        for representation_id in sorted(dimensions):
            dimension = dimensions[representation_id]
            group = root.create_group(groups[representation_id])
            coordinates = group.create_dataset(
                "coordinates",
                shape=(0, dimension),
                maxshape=(None, dimension),
                dtype=np.float32,
                chunks=(max(1, min(4096, 1_048_576 // max(4 * dimension, 1))), dimension),
                compression="gzip",
                shuffle=True,
            )
            probabilities = group.create_dataset(
                "probabilities",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(4096,),
                compression="gzip",
                shuffle=True,
            )
            indptr = group.create_dataset(
                "indptr", shape=(1,), maxshape=(None,), dtype=np.int64, chunks=True
            )
            indptr[0] = 0
            rows = support_index.loc[
                support_index["available"]
                & support_index["store_id"].eq(store_id)
                & support_index["representation_id"].eq(representation_id)
            ].sort_values("support_key")
            atom_count = 0
            coordinate_batch: list[np.ndarray] = []
            probability_batch: list[np.ndarray] = []
            length_batch: list[int] = []
            pending_atoms = 0

            for law_index, row in enumerate(rows.itertuples(index=False)):
                ref = SupportRef(store_id, representation_id, str(row.support_key))
                law = study.supports.read(ref)
                if law.dimension != dimension:
                    raise ValueError(
                        f"Representation {representation_id!r} changes support dimension."
                    )
                digest = semantic_digests[representation_id]
                canonical_coordinates = np.asarray(law.coordinates, dtype="<f4", order="C")
                canonical_probabilities = np.asarray(law.probabilities, dtype="<f8", order="C")
                coordinate_batch.append(canonical_coordinates)
                probability_batch.append(canonical_probabilities)
                length_batch.append(len(canonical_coordinates))
                pending_atoms += len(canonical_coordinates)
                digest.update(store_id.encode())
                digest.update(b"\0")
                digest.update(ref.support_key.encode())
                digest.update(b"\0")
                digest.update(np.asarray(canonical_coordinates.shape, dtype="<i8").tobytes())
                digest.update(canonical_coordinates.tobytes(order="C"))
                digest.update(canonical_probabilities.tobytes(order="C"))
                index_rows.append(
                    {
                        "representation_id": representation_id,
                        "support_key": ref.support_key,
                        "law_index": law_index,
                    }
                )
                if pending_atoms >= _PACKED_WRITE_ATOMS or len(length_batch) >= _PACKED_WRITE_LAWS:
                    atom_count = _flush_packed_batch(
                        coordinates,
                        probabilities,
                        indptr,
                        dimension,
                        atom_count,
                        coordinate_batch,
                        probability_batch,
                        length_batch,
                    )
                    pending_atoms = 0
            _flush_packed_batch(
                coordinates,
                probabilities,
                indptr,
                dimension,
                atom_count,
                coordinate_batch,
                probability_batch,
                length_batch,
            )
    pd.DataFrame(
        index_rows,
        columns=("representation_id", "support_key", "law_index"),
    ).to_parquet(temporary / index_relative, index=False)
    return {
        "store_id": store_id,
        "backend": "packed_hdf5_empirical_laws_v1",
        "path": file_relative,
        "index": index_relative,
        "dimensions": dimensions,
        "groups": groups,
        "semantic_identities": {
            representation_id: {"materialized_sha256": digest.hexdigest()}
            for representation_id, digest in semantic_digests.items()
        },
    }


class NativePerturbSeqStudyV4Codec:
    codec_id = _FORMAT
    readable_schema_versions = frozenset({_SCHEMA_VERSION})
    writable_schema_versions = frozenset({_SCHEMA_VERSION})

    def probe(self, source: Any) -> bool:
        try:
            path = Path(source).expanduser()
        except TypeError:
            return False
        manifest = path / "study.json" if path.is_dir() else path
        if manifest.name != "study.json" or not manifest.is_file():
            return False
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(payload, Mapping)
            and payload.get("format") == _FORMAT
            and payload.get("schema_version") == _SCHEMA_VERSION
        )

    def write(self, study: PerturbSeqStudy, destination: str | Path) -> Path:
        if not isinstance(study, PerturbSeqStudy):
            raise TypeError("Native schema-v4 writing requires a PerturbSeqStudy.")
        study.validate(level="semantic").raise_for_errors()
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"Native study destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            tables: dict[str, str] = {}

            def write_table(name: str, frame: pd.DataFrame) -> None:
                relative = f"{name}.parquet"
                frame.to_parquet(temporary / relative, index=False)
                tables[name] = relative

            required_tables = {
                "perturbations": study.perturbations,
                "intervention_events": study.intervention_events,
                "series": study.series,
                "observations": study.observations,
                "support_index": study.support_index,
            }
            optional_tables = {
                "perturbation_components": study.perturbation_components,
                "contexts": study.contexts,
                "abundance": study.abundance,
                "compositions": study.compositions,
                "population_pools": study.population_pools,
                "effect_bindings": study.effect_bindings,
                "reference_bindings": study.reference_bindings,
            }
            for name, table in required_tables.items():
                write_table(name, table.to_pandas())
            for name, table in optional_tables.items():
                if table is not None:
                    write_table(name, table.to_pandas())

            (temporary / "stores").mkdir()
            stores = []
            for store_number, store_id in enumerate(study.supports.store_ids):
                payload = _write_packed_store(study, temporary, store_id, store_number)
                if payload is not None:
                    stores.append(payload)
            (temporary / "provenance.json").write_text(
                json.dumps(_canonical_value(study.provenance), indent=2) + "\n",
                encoding="utf-8",
            )
            representations = _persist_representation_artifacts(study, temporary)
            files = sorted(
                path
                for path in temporary.rglob("*")
                if path.is_file() and path.name != "study.json"
            )
            artifacts = {
                str(path.relative_to(temporary)): {
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            }
            manifest = {
                "format": _FORMAT,
                "schema_version": _SCHEMA_VERSION,
                "study_content_hash": study.content_hash(),
                "manifest": _canonical_value(study.manifest),
                "design": _design_payload(study.design),
                "tables": tables,
                "abundance_channels": (
                    []
                    if study.abundance is None
                    else [
                        _canonical_value(spec)
                        for _, spec in sorted(study.abundance.channels.items())
                    ]
                ),
                "representations": representations,
                "stores": stores,
                "provenance": "provenance.json",
                "artifacts": artifacts,
            }
            (temporary / "study.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target / "study.json"

    def read(
        self,
        source: Any,
        *,
        verify: VerificationLevel = "semantic",
        **kwargs: Any,
    ) -> PerturbSeqStudy:
        if kwargs:
            raise TypeError(f"Unsupported native Perturb-seq options: {sorted(kwargs)}")
        if verify not in {"none", "schema", "manifest", "semantic", "full"}:
            raise ValueError(f"Unknown verification level {verify!r}.")
        path = Path(source).expanduser().resolve()
        manifest_path = path / "study.json" if path.is_dir() else path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != _FORMAT or payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported native PerturbSeqStudy manifest.")
        root = manifest_path.parent

        def artifact_path(relative: str) -> Path:
            candidate = (root / relative).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Native study path escapes its root: {relative!r}.")
            return candidate

        rank = {"none": 0, "schema": 1, "manifest": 2, "semantic": 3, "full": 4}[verify]
        if rank >= 2:
            for relative, artifact in payload["artifacts"].items():
                candidate = artifact_path(relative)
                if not candidate.is_file():
                    raise ValueError(f"Native study artifact is missing: {relative}")
                if candidate.stat().st_size != int(artifact["size_bytes"]):
                    raise ValueError(f"Native study artifact size mismatch: {relative}")
                if _sha256(candidate) != artifact["sha256"]:
                    raise ValueError(f"Native study artifact hash mismatch: {relative}")
        tables = payload["tables"]

        def read_table(name: str) -> pd.DataFrame:
            return pd.read_parquet(artifact_path(tables[name]))

        abundance = None
        if "abundance" in tables:
            abundance = AbundanceTable(
                read_table("abundance"),
                tuple(AbundanceChannelSpec(**value) for value in payload["abundance_channels"]),
            )
        stores = tuple(
            NativePackedH5SupportStore(
                value["store_id"],
                artifact_path(value["path"]),
                pd.read_parquet(artifact_path(value["index"])),
                value["dimensions"],
                value["groups"],
                value["semantic_identities"],
            )
            for value in payload["stores"]
        )
        representations = []
        for raw_representation in payload["representations"]:
            representation = dict(raw_representation)
            for name in _REPRESENTATION_ARTIFACTS:
                artifact = representation.get(name)
                if artifact is not None and artifact.get("uri") in payload["artifacts"]:
                    artifact = dict(artifact)
                    artifact["uri"] = str(artifact_path(artifact["uri"]))
                    representation[name] = artifact
            representations.append(_representation_from_payload(representation))
        builder = PerturbSeqStudyBuilder(
            manifest=PerturbSeqManifest(**payload["manifest"]),
            design=_design_from_payload(payload["design"]),
            perturbations=PerturbationTable(read_table("perturbations")),
            perturbation_components=(
                PerturbationComponentTable(read_table("perturbation_components"))
                if "perturbation_components" in tables
                else None
            ),
            intervention_events=InterventionEventTable(read_table("intervention_events")),
            contexts=ContextTable(read_table("contexts")) if "contexts" in tables else None,
            series=PopulationSeriesTable(read_table("series")),
            observations=SnapshotObservationTable(read_table("observations")),
            support_index=SupportIndexTable(read_table("support_index")),
            abundance=abundance,
            compositions=(
                LPSCompositionTable(read_table("compositions"))
                if "compositions" in tables
                else None
            ),
            population_pools=(
                PopulationPoolTable(read_table("population_pools"))
                if "population_pools" in tables
                else None
            ),
            effect_bindings=(
                PerturbationEffectBindingTable(read_table("effect_bindings"))
                if "effect_bindings" in tables
                else None
            ),
            reference_bindings=(
                PerturbationReferenceBindingTable(read_table("reference_bindings"))
                if "reference_bindings" in tables
                else None
            ),
            representations=RepresentationCatalog(tuple(representations)),
            supports=SupportStoreRegistry(stores),
            provenance=json.loads(artifact_path(payload["provenance"]).read_text(encoding="utf-8")),
        )
        study = builder.build(verify="none" if rank < 3 else "semantic")
        try:
            if rank >= 4:
                study.validate(level="full").raise_for_errors()
            if rank >= 2 and study.content_hash() != payload["study_content_hash"]:
                raise ValueError("Native study semantic content hash mismatch.")
        except Exception:
            study.close()
            raise
        return study


codec = NativePerturbSeqStudyV4Codec()


def write_perturb_seq_study(study: PerturbSeqStudy, destination: str | Path) -> Path:
    """Write one canonical native schema-v4 PerturbSeqStudy transactionally."""
    return codec.write(study, destination)


__all__ = [
    "NativePackedH5SupportStore",
    "NativePerturbSeqStudyV4Codec",
    "PerturbSeqStudyBuilder",
    "codec",
    "write_perturb_seq_study",
]
