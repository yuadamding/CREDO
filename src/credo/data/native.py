"""Transactional native schema-v3 persistence for semantic studies."""

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

from .design import AxisSpec, Checkpoint, StudyDesign, Transition
from .representations import ArtifactRef, RepresentationCatalog, RepresentationSpec
from .study import Study, StudyManifest, VerificationLevel, _canonical_value
from .support import EmpiricalLaw, SupportRef, SupportStoreRegistry
from .tables import (
    AbundanceChannelSpec,
    AbundanceTable,
    CompositionTable,
    ConditionTable,
    EffectBindingTable,
    ObservationTable,
    ReferenceBindingTable,
    SeriesTable,
    SupportIndexTable,
)
from .validation import ValidationIssue, ValidationReport

_FORMAT = "credo.native_study"
_SCHEMA_VERSION = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _representation_payload(spec: RepresentationSpec) -> dict[str, Any]:
    return {
        "representation_id": spec.representation_id,
        "backend": spec.backend,
        "space_kind": spec.space_kind,
        "dimension": spec.dimension,
        "support_store_id": spec.support_store_id,
        "support_artifact": _artifact_payload(spec.support_artifact),
        "feature_artifact": _artifact_payload(spec.feature_artifact),
        "encoder_artifact": _artifact_payload(spec.encoder_artifact),
        "decoder_artifact": _artifact_payload(spec.decoder_artifact),
        "normalization_artifact": _artifact_payload(spec.normalization_artifact),
        "fit_split_id": spec.fit_split_id,
        "included_series": list(spec.included_series),
        "included_checkpoints": list(spec.included_checkpoints),
    }


def _persist_representation_artifacts(
    study: Study,
    temporary: Path,
) -> list[dict[str, Any]]:
    directory = temporary / "representations"
    copied: dict[str, str] = {}

    def payload(
        artifact: ArtifactRef | None,
        *,
        support: bool = False,
    ) -> dict[str, Any] | None:
        value = _artifact_payload(artifact)
        if value is None:
            return None
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
            suffix = source.suffix if source.suffix else ".bin"
            relative = f"representations/{artifact.sha256}{suffix}"
            shutil.copy2(source, temporary / relative)
            copied[artifact.sha256] = relative
        value["uri"] = relative
        return value

    representations = []
    for representation_id in sorted(study.representations):
        spec = study.representations[representation_id]
        value = _representation_payload(spec)
        value["support_artifact"] = payload(spec.support_artifact, support=True)
        for name in (
            "feature_artifact",
            "encoder_artifact",
            "decoder_artifact",
            "normalization_artifact",
        ):
            value[name] = payload(getattr(spec, name))
        representations.append(value)
    return representations


def _representation_from_payload(payload: Mapping[str, Any]) -> RepresentationSpec:
    raw = dict(payload)
    for name in (
        "support_artifact",
        "feature_artifact",
        "encoder_artifact",
        "decoder_artifact",
        "normalization_artifact",
    ):
        if raw.get(name) is not None:
            raw[name] = ArtifactRef(**raw[name])
    raw["included_series"] = tuple(raw.get("included_series", ()))
    raw["included_checkpoints"] = tuple(raw.get("included_checkpoints", ()))
    return RepresentationSpec(**raw)


def _design_payload(design: StudyDesign) -> dict[str, Any]:
    return {
        "axes": [
            {
                "axis_id": axis.axis_id,
                "kind": axis.kind,
                "unit": axis.unit,
                "ordered": axis.ordered,
            }
            for axis in design.axes
        ],
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


def _design_from_payload(payload: Mapping[str, Any]) -> StudyDesign:
    return StudyDesign(
        axes=tuple(AxisSpec(**value) for value in payload["axes"]),
        checkpoints=tuple(Checkpoint(**value) for value in payload["checkpoints"]),
        transitions=tuple(Transition(**value) for value in payload["transitions"]),
        topology=payload["topology"],
    )


class NativeH5SupportStore:
    """Lazy native support backend indexed by stable qualified support keys."""

    is_lazy = True

    def __init__(
        self,
        store_id: str,
        path: Path,
        index: pd.DataFrame,
        dimensions: Mapping[str, int],
        semantic_identities: Mapping[str, Mapping[str, str]],
    ) -> None:
        self._store_id = str(store_id)
        self.path = Path(path)
        required = {"representation_id", "support_key", "law_id"}
        missing = required - set(index)
        if missing:
            raise ValueError(f"Native support index is missing columns: {sorted(missing)}")
        if index.duplicated(["representation_id", "support_key"]).any():
            raise ValueError("Native support index contains duplicate qualified support keys.")
        unknown_representations = set(index["representation_id"].astype(str)) - {
            str(value) for value in dimensions
        }
        if unknown_representations:
            raise ValueError(
                "Native support index contains unknown representations: "
                f"{sorted(unknown_representations)}."
            )
        self._index = MappingProxyType(
            {
                (str(row.representation_id), str(row.support_key)): str(row.law_id)
                for row in index.itertuples(index=False)
            }
        )
        self._dimensions = MappingProxyType(
            {str(key): int(value) for key, value in dimensions.items()}
        )
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
        law_id = self._index[(ref.representation_id, ref.support_key)]
        with self._lock:
            group = self._file()[f"laws/{law_id}"]
            coordinates = np.asarray(group["coordinates"], dtype=np.float32)
            probabilities = np.asarray(group["probabilities"], dtype=np.float64)
        return EmpiricalLaw(coordinates, probabilities)

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]:
        return MappingProxyType({ref: self.read(ref) for ref in refs})

    def validate(self, *, full_scan: bool = False) -> ValidationReport:
        if self._closed:
            return ValidationReport(
                (ValidationIssue("error", "support.closed", "Support store is closed."),)
            )
        issues: list[ValidationIssue] = []
        digests = {representation_id: hashlib.sha256() for representation_id in self._dimensions}
        try:
            handle = self._file()
            for (representation_id, support_key), law_id in sorted(self._index.items()):
                path = f"laws/{law_id}"
                if path not in handle:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "support.missing",
                            f"Native support {representation_id!r}/{support_key!r} is absent.",
                        )
                    )
                    continue
                if full_scan:
                    try:
                        law = self.read(SupportRef(self._store_id, representation_id, support_key))
                    except (KeyError, TypeError, ValueError) as exc:
                        issues.append(
                            ValidationIssue(
                                "error",
                                "support.invalid",
                                f"{representation_id!r}/{support_key!r}: {exc}",
                            )
                        )
                    else:
                        if law.dimension != self._dimensions[representation_id]:
                            issues.append(
                                ValidationIssue(
                                    "error",
                                    "support.dimension",
                                    f"Support {support_key!r} has dimension {law.dimension}.",
                                )
                            )
                        digest = digests[representation_id]
                        coordinates = np.asarray(law.coordinates, dtype="<f4", order="C")
                        probabilities = np.asarray(law.probabilities, dtype="<f8", order="C")
                        digest.update(self._store_id.encode("utf-8"))
                        digest.update(b"\0")
                        digest.update(support_key.encode("utf-8"))
                        digest.update(b"\0")
                        digest.update(np.asarray(coordinates.shape, dtype="<i8").tobytes())
                        digest.update(coordinates.tobytes(order="C"))
                        digest.update(probabilities.tobytes(order="C"))
        except OSError as exc:
            issues.append(ValidationIssue("error", "support.open", str(exc)))
        if full_scan:
            for representation_id, digest in digests.items():
                expected = self._semantic_identities.get(representation_id, {}).get(
                    "materialized_sha256"
                )
                if expected is None or digest.hexdigest() != expected:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "support.semantic_hash",
                            "Native support semantic hash disagrees for representation "
                            f"{representation_id!r}.",
                        )
                    )
        return ValidationReport(tuple(issues))

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._closed = True


@dataclass
class StudyBuilder:
    """Small typed assembly helper for native studies and adapters."""

    manifest: StudyManifest
    design: StudyDesign
    conditions: ConditionTable
    series: SeriesTable
    observations: ObservationTable
    support_index: SupportIndexTable
    representations: RepresentationCatalog
    supports: SupportStoreRegistry | Any
    abundance: AbundanceTable | None = None
    compositions: CompositionTable | None = None
    effect_bindings: EffectBindingTable | None = None
    reference_bindings: ReferenceBindingTable | None = None
    provenance: Mapping[str, Any] | None = None

    def build(self, *, verify: VerificationLevel = "semantic") -> Study:
        study = Study(
            manifest=self.manifest,
            design=self.design,
            conditions=self.conditions,
            series=self.series,
            observations=self.observations,
            support_index=self.support_index,
            abundance=self.abundance,
            compositions=self.compositions,
            representations=self.representations,
            supports=self.supports,
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


class NativeStudyV3Codec:
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

    def write(self, study: Study, destination: str | Path) -> Path:
        study.validate(level="semantic").raise_for_errors()
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"Native study destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
        try:
            tables: dict[str, str] = {}

            def write_table(name: str, frame: pd.DataFrame) -> None:
                relative = f"{name}.parquet"
                frame.to_parquet(temporary / relative, index=False)
                tables[name] = relative

            write_table("conditions", study.conditions.to_pandas())
            write_table("series", study.series.to_pandas())
            write_table("observations", study.observations.to_pandas())
            write_table("support_index", study.support_index.to_pandas())
            if study.abundance is not None:
                write_table("abundance", study.abundance.to_pandas())
            if study.compositions is not None:
                write_table("compositions", study.compositions.to_pandas())
            if study.effect_bindings is not None:
                write_table("effect_bindings", study.effect_bindings.to_pandas())
            if study.reference_bindings is not None:
                write_table("reference_bindings", study.reference_bindings.to_pandas())

            stores_dir = temporary / "stores"
            stores_dir.mkdir()
            support_index = study.support_index._unsafe_view()
            store_payloads = []
            for store_number, store_id in enumerate(study.supports.store_ids):
                store_representations = tuple(
                    representation
                    for representation in study.representations.values()
                    if representation.support_store_id == store_id
                )
                if not store_representations:
                    continue
                rows = support_index.loc[
                    support_index["available"] & support_index["store_id"].eq(store_id)
                ].sort_values(["representation_id", "support_key"])
                file_relative = f"stores/store-{store_number:04d}.h5"
                index_relative = f"stores/store-{store_number:04d}.parquet"
                file_path = temporary / file_relative
                index_rows = []
                dimensions = {
                    representation.representation_id: representation.dimension
                    for representation in store_representations
                }
                semantic_digests = {
                    representation.representation_id: hashlib.sha256()
                    for representation in store_representations
                }
                with h5py.File(file_path, "w") as handle:
                    laws = handle.create_group("laws")
                    for law_number, row in enumerate(rows.itertuples(index=False)):
                        law_id = f"{law_number:012d}"
                        ref = SupportRef(
                            str(row.store_id),
                            str(row.representation_id),
                            str(row.support_key),
                        )
                        law = study.supports.read(ref)
                        existing = dimensions[ref.representation_id]
                        if existing != law.dimension:
                            raise ValueError(
                                f"Representation {ref.representation_id!r} changes dimension."
                            )
                        digest = semantic_digests[ref.representation_id]
                        coordinates = np.asarray(law.coordinates, dtype="<f4", order="C")
                        probabilities = np.asarray(law.probabilities, dtype="<f8", order="C")
                        digest.update(ref.store_id.encode("utf-8"))
                        digest.update(b"\0")
                        digest.update(ref.support_key.encode("utf-8"))
                        digest.update(b"\0")
                        digest.update(np.asarray(coordinates.shape, dtype="<i8").tobytes())
                        digest.update(coordinates.tobytes(order="C"))
                        digest.update(probabilities.tobytes(order="C"))
                        group = laws.create_group(law_id)
                        group.create_dataset(
                            "coordinates",
                            data=law.coordinates,
                            compression="gzip",
                            shuffle=True,
                        )
                        group.create_dataset(
                            "probabilities",
                            data=law.probabilities,
                            compression="gzip",
                            shuffle=True,
                        )
                        index_rows.append(
                            {
                                "representation_id": ref.representation_id,
                                "support_key": ref.support_key,
                                "law_id": law_id,
                            }
                        )
                pd.DataFrame(
                    index_rows,
                    columns=("representation_id", "support_key", "law_id"),
                ).to_parquet(temporary / index_relative, index=False)
                store_payloads.append(
                    {
                        "store_id": store_id,
                        "backend": "hdf5_empirical_laws_v1",
                        "path": file_relative,
                        "index": index_relative,
                        "dimensions": dimensions,
                        "semantic_identities": {
                            representation_id: {"materialized_sha256": digest.hexdigest()}
                            for representation_id, digest in semantic_digests.items()
                        },
                    }
                )

            provenance_path = temporary / "provenance.json"
            provenance_path.write_text(
                json.dumps(_canonical_value(study.provenance), indent=2) + "\n",
                encoding="utf-8",
            )
            representation_payloads = _persist_representation_artifacts(study, temporary)
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
            abundance_channels = (
                []
                if study.abundance is None
                else [
                    _canonical_value(spec) for _, spec in sorted(study.abundance.channels.items())
                ]
            )
            manifest = {
                "format": _FORMAT,
                "schema_version": _SCHEMA_VERSION,
                "study_content_hash": study.content_hash(),
                "manifest": _canonical_value(study.manifest),
                "design": _design_payload(study.design),
                "tables": tables,
                "abundance_channels": abundance_channels,
                "representations": representation_payloads,
                "stores": store_payloads,
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
    ) -> Study:
        if kwargs:
            raise TypeError(f"Unsupported native study options: {sorted(kwargs)}")
        if verify not in {"none", "schema", "manifest", "semantic", "full"}:
            raise ValueError(f"Unknown verification level {verify!r}.")
        path = Path(source).expanduser().resolve()
        manifest_path = path / "study.json" if path.is_dir() else path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != _FORMAT or payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported native Study manifest.")
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
            channels = tuple(
                AbundanceChannelSpec(**value) for value in payload["abundance_channels"]
            )
            abundance = AbundanceTable(read_table("abundance"), channels)
        stores = []
        for store in payload["stores"]:
            stores.append(
                NativeH5SupportStore(
                    store["store_id"],
                    artifact_path(store["path"]),
                    pd.read_parquet(artifact_path(store["index"])),
                    store["dimensions"],
                    store["semantic_identities"],
                )
            )
        provenance = json.loads(artifact_path(payload["provenance"]).read_text(encoding="utf-8"))
        representation_payloads = []
        for representation in payload["representations"]:
            representation = dict(representation)
            for name in (
                "support_artifact",
                "feature_artifact",
                "encoder_artifact",
                "decoder_artifact",
                "normalization_artifact",
            ):
                artifact = representation.get(name)
                if artifact is not None and artifact.get("uri") in payload["artifacts"]:
                    artifact = dict(artifact)
                    artifact["uri"] = str(artifact_path(artifact["uri"]))
                    representation[name] = artifact
            representation_payloads.append(representation)
        manifest_raw = dict(payload["manifest"])
        study = StudyBuilder(
            manifest=StudyManifest(**manifest_raw),
            design=_design_from_payload(payload["design"]),
            conditions=ConditionTable(read_table("conditions")),
            series=SeriesTable(read_table("series")),
            observations=ObservationTable(read_table("observations")),
            support_index=SupportIndexTable(read_table("support_index")),
            abundance=abundance,
            compositions=(
                CompositionTable(read_table("compositions")) if "compositions" in tables else None
            ),
            effect_bindings=(
                EffectBindingTable(read_table("effect_bindings"))
                if "effect_bindings" in tables
                else None
            ),
            reference_bindings=(
                ReferenceBindingTable(read_table("reference_bindings"))
                if "reference_bindings" in tables
                else None
            ),
            representations=RepresentationCatalog(
                tuple(_representation_from_payload(value) for value in representation_payloads)
            ),
            supports=SupportStoreRegistry(tuple(stores)),
            provenance=provenance,
        ).build(verify="none" if rank < 3 else "semantic")
        try:
            if rank >= 4:
                study.validate(level="full").raise_for_errors()
            if rank >= 2:
                actual_hash = study.content_hash()
                if actual_hash != payload["study_content_hash"]:
                    raise ValueError(
                        "Native study semantic content hash mismatch: "
                        f"declared={payload['study_content_hash']}, actual={actual_hash}."
                    )
        except Exception:
            study.close()
            raise
        return study


codec = NativeStudyV3Codec()


def write_study(study: Study, destination: str | Path) -> Path:
    """Write one canonical native schema-v3 study transactionally."""
    return codec.write(study, destination)


__all__ = [
    "NativeH5SupportStore",
    "NativeStudyV3Codec",
    "StudyBuilder",
    "codec",
    "write_study",
]
