"""Read-only, hash-bound v4 individual-cell laws for main CREDO SupportStore.

This adapter does not encode, rescale, subsample, or fabricate particles. Opening
a real store is numerical access and must be separately admitted. The current
qualification uses synthetic fixtures only. Publication/completion and phase
release authority remain obligations of the caller's execution contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pyarrow.parquet as pq

from credo.data.support import EmpiricalLaw, SupportRef
from credo.data.validation import ValidationIssue, ValidationReport


class LatentSupportError(ValueError):
    """A bound latent artifact, source phase, or individual law is invalid."""


def _sha(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise LatentSupportError("Explicit lowercase SHA-256 identity required.")
    return value


def _int(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise LatentSupportError("Nonnegative integer required, not a bool or rounded count.")
    return value


def _contract_id(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes) -> dict:
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise LatentSupportError("Duplicate JSON key.")
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(LatentSupportError("Nonfinite JSON.")))
    if not isinstance(value, dict):
        raise LatentSupportError("Object manifest required.")
    return value


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        _sha(self.sha256)
        _int(self.size_bytes, minimum=1)


@dataclass(frozen=True)
class RepresentationIdentity:
    representation_id: str
    artifact_sha256: str
    model_sha256: str
    scaler_sha256: str
    scaler_location: str = "model.safetensors::center,scale"

    def __post_init__(self):
        if not isinstance(self.representation_id, str) or not self.representation_id:
            raise LatentSupportError("Representation ID required.")
        for value in (self.artifact_sha256, self.model_sha256, self.scaler_sha256):
            _sha(value)
        # The v4 frozen encoder's center/scale are state_dict buffers. Their
        # identity is the containing checkpoint, NOT an invented buffer digest.
        if self.scaler_location != "model.safetensors::center,scale" or self.scaler_sha256 != self.model_sha256:
            raise LatentSupportError("Scaler must be the bound embedded checkpoint buffers; no external transform.")


@dataclass(frozen=True)
class PhaseAccess:
    phase_role: str
    architecture_selection_sha256: str | None = None
    terminal_checkpoints_sha256: str | None = None
    forecast_seal_sha256: str | None = None
    assessment_sha256: str | None = None
    verified_by_lifecycle: bool = False

    def admit(self, source_ids: Sequence[str]) -> None:
        """Check whole-bundle access before opening even its numerical hashes.

        Release identities must themselves be receipt-verified by the enclosing
        execution contract; a string here is not independent proof of a seal.
        """
        if self.phase_role not in {"fitting", "selection", "forecast", "assessment", "final_refit"}:
            raise LatentSupportError("Unknown numerical phase role.")
        required = ()
        if self.phase_role in {"forecast", "assessment", "final_refit"}:
            required += (self.architecture_selection_sha256, self.terminal_checkpoints_sha256)
        if self.phase_role in {"assessment", "final_refit"}:
            required += (self.forecast_seal_sha256,)
        if self.phase_role == "final_refit":
            required += (self.assessment_sha256,)
        for identity in required:
            _sha(identity)
        if required and self.verified_by_lifecycle is not True:
            raise LatentSupportError("Release identities need receipt/custody verification by lifecycle.")
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise LatentSupportError("Nonempty unique ordered sources required.")
        for source in source_ids:
            if source not in {f"{d}_{c}" for d in ("D1", "D2", "D3", "D4")
                              for c in ("Rest", "Stim8hr", "Stim48hr")}:
                raise LatentSupportError("Only named GSE314342 sources are permitted.")
            donor, condition = source.split("_", 1)
            allowed = {"fitting": {"D3", "D4"}, "selection": {"D2"},
                       "forecast": {"D1"}, "assessment": {"D1"},
                       "final_refit": {"D1", "D2", "D3", "D4"}}[self.phase_role]
            if donor not in allowed or (self.phase_role == "forecast" and condition == "Stim48hr"):
                raise LatentSupportError("Bundle source is not permitted in this numerical phase.")


@dataclass(frozen=True)
class FrozenLatentBinding:
    store_id: str
    latent: FileIdentity
    cells: FileIdentity
    support: FileIdentity
    completion: FileIdentity
    representation_record: FileIdentity
    representation: RepresentationIdentity
    source_ids: tuple[str, ...]
    guide_count: int
    dimension: int
    retained_cells: int
    geometry_cells: int
    maximum_index_bytes: int = 512 * 1024**2
    maximum_law_bytes: int = 256 * 1024**2
    maximum_metadata_bytes: int = 64 * 1024**2
    batch_rows: int = 8192

    def __post_init__(self):
        if not isinstance(self.store_id, str) or not self.store_id:
            raise LatentSupportError("Store ID required.")
        if type(self.source_ids) is not tuple:
            raise LatentSupportError("Immutable ordered source tuple required.")
        for name in ("guide_count", "dimension", "retained_cells", "maximum_index_bytes",
                     "maximum_law_bytes", "maximum_metadata_bytes", "batch_rows"):
            _int(getattr(self, name), minimum=1)
        _int(self.geometry_cells)
        if self.geometry_cells > self.retained_cells or self.batch_rows > 8192:
            raise LatentSupportError("Invalid geometry or bounded-read size.")
        files = (self.latent, self.cells, self.support, self.completion, self.representation_record)
        if len({str(f.path.absolute()) for f in files}) != len(files):
            raise LatentSupportError("Input file aliases are forbidden.")
        for item, name in ((self.latent, "latent.h5"), (self.cells, "cells.parquet"),
                           (self.support, "support.parquet"), (self.completion, "COMPLETE.json")):
            if item.path.name != name or item.path.parent != self.latent.path.parent:
                raise LatentSupportError("Exact native latent bundle filenames required.")


def _signature(path: Path) -> tuple[int, ...]:
    for part in (path.absolute(), *path.absolute().parents):
        if part.is_symlink():
            raise LatentSupportError("Symlink input is forbidden.")
    item = path.stat()
    if not stat.S_ISREG(item.st_mode):
        raise LatentSupportError("Regular input file required.")
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def _verify_file(item: FileIdentity) -> tuple[int, ...]:
    before = _signature(item.path)
    if before[2] != item.size_bytes:
        raise LatentSupportError("Input byte size differs from binding.")
    checksum = hashlib.sha256()
    with item.path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            checksum.update(chunk)
    if before != _signature(item.path) or checksum.hexdigest() != item.sha256:
        raise LatentSupportError("Input identity mismatch or changed while hashing.")
    return before


class FrozenLatentSupportStore:
    """Native main-CREDO support backend over frozen v4 cell-level artifacts."""

    def __init__(self, binding: FrozenLatentBinding, phase: PhaseAccess, *,
                 expected_representation: RepresentationIdentity):
        phase.admit(binding.source_ids)  # before any file I/O
        if binding.representation != expected_representation:
            raise LatentSupportError("Representation/model/scaler identity mismatch.")
        self.binding, self.phase = binding, phase
        self._snapshot_cache = None
        self._closed = True
        self._files = (binding.completion, binding.representation_record,
                       binding.support, binding.cells, binding.latent)
        for item in (binding.completion, binding.representation_record, binding.support):
            if item.size_bytes > binding.maximum_metadata_bytes:
                raise LatentSupportError("Metadata read exceeds admitted bytes.")
        populations = len(binding.source_ids) * binding.guide_count
        # Owner array plus offsets, support counts and bounded scratch. No full
        # population_rows/state copy. This is an allocation admission estimate.
        estimated = binding.retained_cells * 4 + (populations + 1) * 8 * 6 + binding.batch_rows * 64
        if populations >= 2**31 or estimated > binding.maximum_index_bytes:
            raise LatentSupportError("Population index exceeds admitted memory budget.")
        # Metadata identities and declared source order precede payload hashing.
        self._signatures = {str(f.path): _verify_file(f) for f in self._files[:2]}
        self._verify_manifests()
        for item in self._files[2:]:
            self._signatures[str(item.path)] = _verify_file(item)
        if len({sig[:2] for sig in self._signatures.values()}) != len(self._files):
            raise LatentSupportError("Hardlinked input aliases are forbidden.")
        self._support = self._load_support()
        self._verify_indices()
        self._closed = False
        self._check_unchanged()

    def _verify_manifests(self):
        b = self.binding
        record = _json(b.completion.path.read_bytes())
        fitted = _json(b.representation_record.path.read_bytes())
        if (fitted.get("stage") != "fitted_representation"
                or fitted.get("schema_id") != "credo.fold_count_representation_bundle"
                or fitted.get("schema_version") != 2):
            raise LatentSupportError("Incorrect fitted representation stage.")
        if _contract_id(fitted) != b.representation.artifact_sha256:
            raise LatentSupportError("Fitted representation contract differs.")
        facts = fitted.get("facts", {})
        if facts.get("model_numerical_sha256") != b.representation.model_sha256:
            raise LatentSupportError("Frozen model/scaler checkpoint differs.")
        if facts.get("latent_dim") != b.dimension or facts.get("frozen_fitting_geometry") is not True:
            raise LatentSupportError("Frozen coordinate geometry not established.")
        model = [a for a in fitted.get("artifacts", []) if a.get("relative_uri") == "model.safetensors"]
        if len(model) != 1 or model[0].get("sha256") != b.representation.model_sha256:
            raise LatentSupportError("Fitted checkpoint inventory differs.")
        self._ordered_cell_address_sha256 = None
        if record.get("schema") == "credo.all4_ddr.latent_source.v1":
            self._verify_new_source_record(record)
            return
        if (record.get("stage") != "latent_cells" or "schema" in record
                or record.get("schema_id") != "credo.fold_count_representation_bundle"
                or record.get("schema_version") != 2):
            raise LatentSupportError("Unsupported latent publication schema/stage.")
        if record.get("parents", {}).get("representation") != b.representation.artifact_sha256:
            raise LatentSupportError("Latent parent representation differs.")
        facts = record.get("facts", {})
        expected = {"source_ids": list(b.source_ids), "complete_catalog_categories": b.guide_count,
                    "retained_individual_cells": b.retained_cells, "geometry_cells": b.geometry_cells,
                    "latent_dim": b.dimension, "latent_payload_sha256": b.latent.sha256,
                    "encoded_weight_updates": 0,
                    "law_scope": "empirical_positive_RNA_cells; captured_zero_RNA_support_retained_separately",
                    "zero_RNA_coordinate_policy": "zero_placeholder_geometry_available_false"}
        if any(facts.get(k) != v for k, v in expected.items()):
            raise LatentSupportError("Latent source/order/support/coordinate facts differ.")
        artifacts = record.get("artifacts", [])
        if len({a["relative_uri"] for a in artifacts}) != len(artifacts):
            raise LatentSupportError("Duplicate native artifact entry.")
        indexed = {a["relative_uri"]: a for a in artifacts}
        for item in (b.latent, b.cells, b.support):
            row = indexed.get(item.path.name, {})
            if row.get("sha256") != item.sha256 or row.get("size_bytes") != item.size_bytes:
                raise LatentSupportError("Native artifact inventory differs from explicit binding.")

    def _verify_new_source_record(self, record):
        b = self.binding
        expected = {"status": "encoded_frozen_representation", "source_ids": list(b.source_ids),
                    "expected_rows": b.retained_cells, "geometry_cells": b.geometry_cells,
                    "complete_catalog_categories": b.guide_count, "latent_dim": b.dimension,
                    "optimizer_updates": 0, "raw_counts_modified": False,
                    "raw_counts_copied": False, "training_complete": False,
                    "zero_RNA_coordinate_policy": "zero_placeholder_geometry_available_false",
                    "count_vector_scope": "all_captured_rows_same_encoding_pass"}
        if len(b.source_ids) != 1 or any(record.get(k) != v for k, v in expected.items()):
            raise LatentSupportError("New source export facts differ from explicit binding.")
        lineage = record.get("lineage", {})
        expected_lineage = {"representation_sha256": b.representation.artifact_sha256,
                            "model_sha256": b.representation.model_sha256,
                            "scaler_sha256": b.representation.scaler_sha256}
        if any(lineage.get(k) != v for k, v in expected_lineage.items()):
            raise LatentSupportError("New source representation/model/scaler lineage differs.")
        for key in ("input_access_sha256", "gene_axis_sha256", "implementation_sha256"):
            _sha(lineage.get(key))
        allowed_export_phases = {
            "fitting": {"development"}, "selection": {"development"},
            "forecast": {"forecast_source"}, "assessment": {"forecast_source", "assessment"},
            "final_refit": {"development", "forecast_source", "assessment", "final_refit"}}
        if record.get("phase") not in allowed_export_phases[self.phase.phase_role]:
            raise LatentSupportError("Export phase cannot be relabeled into the requested stage.")
        files = record.get("files", {})
        if set(files) != {"latent.h5", "cells.parquet", "support.parquet"}:
            raise LatentSupportError("New source export requires its exact three payloads.")
        for item in (b.latent, b.cells, b.support):
            row = files[item.path.name]
            if row.get("sha256") != item.sha256 or row.get("bytes") != item.size_bytes:
                raise LatentSupportError("New source payload inventory differs.")
        self._ordered_cell_address_sha256 = _sha(record.get("ordered_cell_address_sha256"))

    def _load_support(self):
        b = self.binding
        rows = pq.read_table(io.BytesIO(b.support.path.read_bytes()), use_threads=False).to_pylist()
        expected = {(s, g) for s in b.source_ids for g in range(b.guide_count)}
        result = {}
        for row in rows:
            guide = _int(row["guide_index"])
            key = (row["source_id"], guide)
            captured, geometry, absent = (_int(row[k]) for k in ("captured_cells", "geometry_cells", "geometry_absent_cells"))
            if key not in expected or key in result or geometry + absent != captured:
                raise LatentSupportError("Support catalog is duplicated, incomplete, or inconsistent.")
            result[key] = {"source_id": key[0], "guide_index": guide,
                           "captured_cells": captured, "geometry_cells": geometry,
                           "geometry_absent_cells": absent, "available": geometry > 0,
                           "support_key": f"{key[0]}::{guide}",
                           "law_scope": "conditional_positive_RNA_cells_not_all_captured_states"}
        if set(result) != expected or sum(r["captured_cells"] for r in result.values()) != b.retained_cells:
            raise LatentSupportError("Support table does not cover the complete declared catalog.")
        if sum(r["geometry_cells"] for r in result.values()) != b.geometry_cells:
            raise LatentSupportError("Geometry total differs from binding.")
        return result

    def _verify_indices(self):
        b = self.binding
        owner = np.full(b.retained_cells, -1, dtype=np.int32)
        geometry = np.array([self._support[(s, g)]["geometry_cells"]
                             for s in b.source_ids for g in range(b.guide_count)], dtype=np.int64)
        with h5py.File(b.latent.path, "r") as handle:
            for name in ("state", "population_offsets", "population_rows"):
                if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                    raise LatentSupportError("External or soft HDF5 dataset links forbidden.")
                item = handle[name]
                if not isinstance(item, h5py.Dataset) or item.is_virtual or item.external:
                    raise LatentSupportError("Externally backed HDF5 datasets forbidden.")
            state, offsets, rows = (handle[n] for n in ("state", "population_offsets", "population_rows"))
            if state.shape != (b.retained_cells, b.dimension) or state.dtype != np.dtype("float32"):
                raise LatentSupportError("Individual state shape/dtype differs.")
            if offsets.shape != (len(geometry) + 1,) or offsets.dtype != np.dtype("int64"):
                raise LatentSupportError("Population offsets shape/dtype differs.")
            if rows.shape != (b.geometry_cells,) or rows.dtype != np.dtype("int64"):
                raise LatentSupportError("Population rows shape/dtype differs.")
            values = offsets[:]
            if values[0] != 0 or values[-1] != b.geometry_cells or not np.array_equal(np.diff(values), geometry):
                raise LatentSupportError("Offsets disagree with positive-RNA support.")
            self._offsets = values
            for key, (left, right) in enumerate(zip(values[:-1], values[1:])):
                previous = -1
                for start in range(int(left), int(right), b.batch_rows):
                    selected = rows[start:min(start + b.batch_rows, right)]
                    if (np.any(selected < 0) or np.any(selected >= b.retained_cells)
                            or selected[0] <= previous or np.any(np.diff(selected) <= 0)
                            or np.any(owner[selected] != -1)):
                        raise LatentSupportError("Population rows are out of bounds, duplicated, or unordered.")
                    owner[selected] = key
                    previous = int(selected[-1])
        captured = np.zeros(len(geometry), dtype=np.int64)
        source_index = {s: i for i, s in enumerate(b.source_ids)}
        position = 0
        columns = ["source_id", "guide_index", "RNA_UMIs", "latent_row", "geometry_available"]
        address_digest = hashlib.sha256()
        if self._ordered_cell_address_sha256 is not None:
            columns += ["shard", "row_in_shard", "cell_id"]
        for batch in pq.ParquetFile(b.cells.path).iter_batches(batch_size=b.batch_rows, columns=columns, use_threads=False):
            records = batch.to_pylist()
            for row in records:
                if _int(row["latent_row"]) != position or position >= b.retained_cells:
                    raise LatentSupportError("Cell-to-latent rows are incomplete or reordered.")
                source, guide = row["source_id"], _int(row["guide_index"])
                if source not in source_index or guide >= b.guide_count:
                    raise LatentSupportError("Cell source/guide index differs from catalog.")
                umi = _int(row["RNA_UMIs"])
                available = row["geometry_available"]
                if type(available) is not bool or available != (umi > 0):
                    raise LatentSupportError("Geometry flag contradicts positive RNA depth.")
                key = source_index[source] * b.guide_count + guide
                if int(owner[position]) != (key if available else -1):
                    raise LatentSupportError("Population index includes wrong guide/source or zero-RNA cell.")
                if self._ordered_cell_address_sha256 is not None:
                    shard, native_row = _int(row["shard"]), _int(row["row_in_shard"])
                    if not isinstance(row["cell_id"], str) or not row["cell_id"]:
                        raise LatentSupportError("Missing canonical cell identity.")
                    content = json.dumps([source, shard, native_row, row["cell_id"], guide],
                                         sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                         allow_nan=False) + "\n"
                    address_digest.update(content.encode())
                captured[key] += 1
                position += 1
        expected = np.array([self._support[(s, g)]["captured_cells"]
                             for s in b.source_ids for g in range(b.guide_count)])
        if position != b.retained_cells or not np.array_equal(captured, expected):
            raise LatentSupportError("Captured cell metadata disagrees with complete support.")
        if self._ordered_cell_address_sha256 is not None and address_digest.hexdigest() != self._ordered_cell_address_sha256:
            raise LatentSupportError("Ordered cell/source/guide addresses differ from new export receipt.")

    @property
    def store_id(self) -> str:
        return self.binding.store_id

    def representation_ids(self) -> tuple[str, ...]:
        return (self.binding.representation.representation_id,)

    def dimension(self, representation_id: str) -> int:
        if representation_id not in self.representation_ids():
            raise KeyError(representation_id)
        return self.binding.dimension

    def _check_unchanged(self):
        if self._closed:
            raise LatentSupportError("Support store is closed.")
        # During an explicit snapshot lease, reads use only private, immutable
        # cached bytes, never the mutable paths. Check paths at BOTH lease
        # boundaries instead of performing millions of GPFS metadata operations.
        if self._snapshot_cache is not None:
            return
        self._verify_snapshot_identity()

    def _verify_snapshot_identity(self):
        if any(_signature(f.path) != self._signatures[str(f.path)] for f in self._files):
            raise LatentSupportError("Verified support inputs changed.")

    @property
    def snapshot_bytes(self) -> int:
        return self.binding.retained_cells * self.binding.dimension * 4 + self.binding.geometry_cells * 8

    @contextmanager
    def cached_snapshot(self, *, maximum_bytes: int):
        """One sequential, bounded CPU read; identical individual-cell laws.

        This opt-in lease is not a new data sample or representation. All
        payload hashes and population memberships have already been verified.
        A changed path fails the lease at exit, so its caller must not publish
        success until the context has exited. No cached array is returned.
        """
        _int(maximum_bytes, minimum=1)
        self._check_unchanged()
        if self._snapshot_cache is not None:
            raise LatentSupportError("Nested snapshot leases are forbidden.")
        if self.snapshot_bytes > maximum_bytes:
            raise LatentSupportError("Snapshot exceeds admitted CPU memory budget.")
        b = self.binding
        state = np.empty((b.retained_cells, b.dimension), dtype=np.float32)
        rows = np.empty(b.geometry_cells, dtype=np.int64)
        with h5py.File(b.latent.path, "r") as handle:
            for start in range(0, b.retained_cells, b.batch_rows):
                stop = min(start + b.batch_rows, b.retained_cells)
                handle["state"].read_direct(state, np.s_[start:stop], np.s_[start:stop])
            for start in range(0, b.geometry_cells, b.batch_rows):
                stop = min(start + b.batch_rows, b.geometry_cells)
                handle["population_rows"].read_direct(rows, np.s_[start:stop], np.s_[start:stop])
        self._verify_snapshot_identity()
        state.setflags(write=False)
        rows.setflags(write=False)
        self._snapshot_cache = (state, rows)
        try:
            yield self
        finally:
            self._snapshot_cache = None
            self._verify_snapshot_identity()

    def support_record(self, source_id: str, guide_index: int) -> dict:
        self._check_unchanged()
        _int(guide_index)
        return dict(self._support[(source_id, guide_index)])

    def ref(self, source_id: str, guide_index: int) -> SupportRef:
        record = self.support_record(source_id, guide_index)
        return SupportRef(self.store_id, self.representation_ids()[0], record["support_key"])

    def _key(self, ref: SupportRef) -> tuple[str, int]:
        if ref.store_id != self.store_id or ref.representation_id not in self.representation_ids():
            raise KeyError(ref)
        try:
            source, guide = ref.support_key.split("::")
            index = int(guide)
            if index < 0 or str(index) != guide or (source, index) not in self._support:
                raise ValueError
            return source, index
        except (ValueError, TypeError) as error:
            raise KeyError(ref) from error

    def contains(self, ref: SupportRef) -> bool:
        if self._closed:
            return False
        self._check_unchanged()
        try:
            return self._support[self._key(ref)]["available"]
        except KeyError:
            return False

    def read(self, ref: SupportRef) -> EmpiricalLaw:
        self._check_unchanged()
        source, guide = self._key(ref)
        count = self._support[(source, guide)]["geometry_cells"]
        if count == 0:
            raise LatentSupportError("State unavailable: no positive-RNA particles; no substitute is allowed.")
        b = self.binding
        if count * (b.dimension * 4 + 8) * 3 > b.maximum_law_bytes:
            raise LatentSupportError("Individual law exceeds admitted materialization budget.")
        key = b.source_ids.index(source) * b.guide_count + guide
        left, right = map(int, self._offsets[key:key + 2])
        if self._snapshot_cache is not None:
            state, rows = self._snapshot_cache
            coordinates = state[rows[left:right]]
        else:
            coordinates = np.empty((count, b.dimension), dtype=np.float32)
            with h5py.File(b.latent.path, "r") as handle:
                for start in range(left, right, b.batch_rows):
                    stop = min(start + b.batch_rows, right)
                    positions = handle["population_rows"][start:stop]
                    coordinates[start-left:stop-left] = handle["state"][positions]
        if not np.isfinite(coordinates).all():
            raise LatentSupportError("Nonfinite positive-RNA latent state.")
        self._check_unchanged()
        return EmpiricalLaw(coordinates, np.full(count, 1.0 / count, dtype=np.float64))

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]:
        # Explicit admission covers the collection, not merely each member.
        unique = tuple(dict.fromkeys(refs))
        self._check_unchanged()
        atoms = sum(self._support[self._key(ref)]["geometry_cells"] for ref in unique)
        if atoms * (self.binding.dimension * 4 + 8) * 3 > self.binding.maximum_law_bytes:
            raise LatentSupportError("Combined laws exceed admitted materialization budget.")
        return MappingProxyType({ref: self.read(ref) for ref in unique})

    def validate(self, *, full_scan: bool = False) -> ValidationReport:
        try:
            self._check_unchanged()
            if full_scan:
                for (source, guide), row in self._support.items():
                    if row["available"]:
                        self.read(self.ref(source, guide))
            return ValidationReport()
        except (ValueError, OSError, KeyError) as error:
            return ValidationReport((ValidationIssue("error", "frozen_latent.invalid", str(error)),))

    def close(self) -> None:
        self._snapshot_cache = None
        self._closed = True


class DevelopmentLatentSupportStore:
    """Unified native routing over phase-separated D3/D4 fitting and D2 selection.

    This union is for study compilation only. The study builder still must keep
    D2 out of the optimizer fitting split; routing does not change its role.
    Closing this wrapper closes its supplied underlying stores.
    """

    def __init__(self, store_id: str, stores: Sequence[FrozenLatentSupportStore], *,
                 maximum_law_bytes: int = 256 * 1024**2):
        if not isinstance(store_id, str) or not store_id or not stores:
            raise LatentSupportError("Nonempty composite identity and stores required.")
        if any(not isinstance(store, FrozenLatentSupportStore) for store in stores):
            raise LatentSupportError("Only verified frozen latent stores may be composed.")
        _int(maximum_law_bytes, minimum=1)
        self._store_id, self.maximum_law_bytes = store_id, maximum_law_bytes
        self._closed = False
        self._stores = tuple(stores)
        self._sources = {}
        self.representation = stores[0].binding.representation
        self.guide_count = stores[0].binding.guide_count
        for store in stores:
            if not isinstance(store, FrozenLatentSupportStore):
                raise LatentSupportError("Only verified frozen latent stores may be composed.")
            store._check_unchanged()
            if store.phase.phase_role not in {"fitting", "selection"}:
                raise LatentSupportError("D1 forecast/assessment stores cannot enter development routing.")
            if (store.binding.representation != self.representation
                    or store.binding.guide_count != self.guide_count
                    or store.binding.dimension != stores[0].binding.dimension):
                raise LatentSupportError("Composite representation/catalog dimensions disagree.")
            for source in store.binding.source_ids:
                if source in self._sources:
                    raise LatentSupportError("Overlapping source populations cannot be merged twice.")
                self._sources[source] = store
        self.source_ids = tuple(self._sources)

    @property
    def store_id(self):
        return self._store_id

    def representation_ids(self):
        return (self.representation.representation_id,)

    @contextmanager
    def cached_snapshot(self, *, maximum_bytes: int):
        """One admitted CPU budget across all phase-separated input stores."""
        _int(maximum_bytes, minimum=1)
        if self._closed or sum(store.snapshot_bytes for store in self._stores) > maximum_bytes:
            raise LatentSupportError("Composite snapshot is closed or exceeds memory admission.")
        with ExitStack() as stack:
            for store in self._stores:
                stack.enter_context(store.cached_snapshot(maximum_bytes=store.snapshot_bytes))
            yield self

    def dimension(self, representation_id):
        return self._stores[0].dimension(representation_id)

    def support_record(self, source_id, guide_index):
        if self._closed:
            raise LatentSupportError("Composite support store is closed.")
        return self._sources[source_id].support_record(source_id, guide_index)

    def ref(self, source_id, guide_index):
        record = self.support_record(source_id, guide_index)
        return SupportRef(self.store_id, self.representation_ids()[0], record["support_key"])

    def _route(self, ref):
        if self._closed:
            raise LatentSupportError("Composite support store is closed.")
        if ref.store_id != self.store_id or ref.representation_id not in self.representation_ids():
            raise KeyError(ref)
        store = self._sources[ref.support_key.split("::", 1)[0]]
        local = SupportRef(store.store_id, ref.representation_id, ref.support_key)
        store._key(local)  # validate canonical identity even for contains
        return store, local

    def contains(self, ref):
        if self._closed:
            return False
        try:
            store, local = self._route(ref)
            return store.contains(local)
        except KeyError:
            return False

    def read(self, ref):
        store, local = self._route(ref)
        return store.read(local)

    def read_many(self, refs):
        unique = tuple(dict.fromkeys(refs))
        atoms = 0
        for ref in unique:
            store, local = self._route(ref)
            atoms += store._support[store._key(local)]["geometry_cells"]
        if atoms * (self.dimension(self.representation_ids()[0]) * 4 + 8) * 3 > self.maximum_law_bytes:
            raise LatentSupportError("Composite laws exceed admitted materialization budget.")
        return MappingProxyType({ref: self.read(ref) for ref in unique})

    def validate(self, *, full_scan=False):
        if self._closed:
            return ValidationReport((ValidationIssue("error", "frozen_latent.closed", "Composite store closed."),))
        return ValidationReport().merged(*(store.validate(full_scan=full_scan) for store in self._stores))

    def close(self):
        self._closed = True
        for store in self._stores:
            store.close()
