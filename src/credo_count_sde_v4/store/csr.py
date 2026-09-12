"""Immutable HDF5-backed CSR storage with stable row and feature identities."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
from scipy import sparse

from ..canonical import canonical_json_bytes, contract_id, sha256_bytes, sha256_file
from ..contracts import CountStoreManifest, FeatureKey
from ..errors import ContractError, IntegrityError


@dataclass(frozen=True)
class SparseCountBatch:
    matrix: sparse.csr_matrix
    row_ids: np.ndarray[Any, Any]
    feature_index_hash: str

    @property
    def library_sizes(self) -> np.ndarray[Any, Any]:
        return cast(np.ndarray[Any, Any], np.asarray(self.matrix.sum(axis=1)).reshape(-1))

    def to_dense(self, *, byte_limit: int) -> np.ndarray[Any, Any]:
        needed = self.matrix.shape[0] * self.matrix.shape[1] * self.matrix.dtype.itemsize
        if needed > byte_limit:
            raise MemoryError(f"Dense conversion needs {needed} bytes; limit is {byte_limit}.")
        return self.matrix.toarray()


def _row_hash(row_ids: np.ndarray[Any, Any]) -> str:
    values = np.asarray(row_ids, dtype="<i8")
    return sha256_bytes(values.tobytes(order="C"))


def _feature_hash(features: tuple[FeatureKey, ...]) -> str:
    return sha256_bytes(
        canonical_json_bytes([feature.model_dump(mode="json") for feature in features])
    )


def build_count_store(
    path: Path,
    matrix: sparse.spmatrix,
    *,
    row_ids: np.ndarray[Any, Any],
    features: tuple[FeatureKey, ...],
    projected_peak_bytes: int | None = None,
) -> CountStoreManifest:
    """Build one immutable, catalog-independent CSR file."""

    if path.exists():
        raise FileExistsError(path)
    csr = sparse.csr_matrix(matrix)
    csr.sort_indices()
    if csr.shape != (len(row_ids), len(features)):
        raise ContractError("Matrix, row ID, and feature dimensions disagree.")
    ids = np.asarray(row_ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ContractError("Global row IDs must be unique.")
    feature_keys = {
        (feature.namespace, feature.namespace_version, feature.feature_id) for feature in features
    }
    if len(feature_keys) != len(features):
        raise ContractError("Composite feature keys must be unique.")
    if not np.issubdtype(csr.dtype, np.integer) or np.any(csr.data < 0):
        raise ContractError("Raw count store requires nonnegative integer values.")
    if csr.nnz and int(csr.data.max()) > np.iinfo(np.int32).max:
        raise ContractError("Raw counts exceed the int32 storage contract.")
    path.parent.mkdir(parents=True, exist_ok=True)
    logical_bytes = (
        csr.nnz * (np.dtype(np.int32).itemsize * 2)
        + (csr.shape[0] + 1) * np.dtype(np.int64).itemsize
        + len(ids) * np.dtype(np.int64).itemsize * 3
        + sum(len(feature.model_dump_json().encode("utf-8")) for feature in features)
    )
    projected_final = max(logical_bytes, 4096)
    projected_peak = projected_peak_bytes or 2 * projected_final
    capacity = shutil.disk_usage(path.parent)
    if capacity.free < int(1.20 * projected_peak) or (
        capacity.free - projected_final < int(0.20 * capacity.total)
    ):
        raise OSError("Count-store capacity gate failed: need 20% peak and post-build headroom.")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with h5py.File(temporary, "x", libver="latest") as handle:
        group = handle.create_group("X")
        group.create_dataset(
            "data", data=csr.data.astype(np.int32), compression="gzip", shuffle=True
        )
        group.create_dataset("indices", data=csr.indices.astype(np.int32), compression="gzip")
        group.create_dataset("indptr", data=csr.indptr.astype(np.int64), compression="gzip")
        handle.create_dataset("row_ids", data=ids.astype(np.int64), compression="gzip")
        sorted_order = np.argsort(ids, kind="stable")
        handle.create_dataset(
            "row_ids_sorted", data=ids[sorted_order].astype(np.int64), compression="gzip"
        )
        handle.create_dataset(
            "row_positions_sorted",
            data=sorted_order.astype(np.int64),
            compression="gzip",
        )
        encoded = [
            json.dumps(feature.model_dump(mode="json"), sort_keys=True) for feature in features
        ]
        handle.create_dataset(
            "features", data=np.asarray(encoded, dtype=h5py.string_dtype("utf-8"))
        )
        handle.attrs["schema_id"] = "credo.count_store"
        handle.attrs["schema_version"] = 1
        handle.attrs["shape"] = csr.shape
        handle.attrs["feature_index_hash"] = _feature_hash(features)
        handle.flush()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    content = sha256_file(temporary)
    payload = {
        "schema_version": 1,
        "store_id": "pending",
        "backend": "csr_hdf5",
        "rows": csr.shape[0],
        "features": csr.shape[1],
        "nnz": csr.nnz,
        "value_dtype": "int32",
        "row_ids_hash": _row_hash(ids),
        "feature_index_hash": _feature_hash(features),
        "content_sha256": content,
        "relative_uri": path.name,
    }
    payload["store_id"] = contract_id(payload, id_field="store_id")
    manifest = CountStoreManifest.model_validate(payload)
    CountStore(temporary, manifest).verify(full=True)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


class CountStore:
    """Read-only process-local handle over one CSR file."""

    def __init__(self, path: Path, manifest: CountStoreManifest | None = None) -> None:
        self.path = path
        self._manifest = manifest
        self._handle: h5py.File | None = None
        self._owner_pid: int | None = None
        self._row_index_cache: (
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None
        ) = None
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(f"Count store is not a regular file: {path}.")

    @property
    def persistent_open(self) -> bool:
        """Whether this process currently owns one persistent HDF5 handle."""

        return self._handle is not None

    @property
    def row_index_cache_bytes(self) -> int:
        """Exact NumPy bytes retained for row lookup and CSR offsets."""

        if self._row_index_cache is None:
            return 0
        return sum(array.nbytes for array in self._row_index_cache)

    @property
    def hdf5_chunk_cache_bytes(self) -> int:
        """Configured raw-data chunk-cache bytes for the persistent handle."""

        if self._handle is None:
            return 0
        self._assert_process_owner()
        return int(self._handle.id.get_access_plist().get_cache()[2])

    def open(self) -> CountStore:
        """Open one process-local reader handle and reuse its immutable row index."""

        if self._handle is not None:
            self._assert_process_owner()
            return self
        self._handle = h5py.File(self.path, "r")
        self._owner_pid = os.getpid()
        return self

    def close(self) -> None:
        """Close the process-local handle without discarding immutable index arrays."""

        if self._handle is not None:
            self._assert_process_owner()
            self._handle.close()
            self._handle = None
            self._owner_pid = None

    def _assert_process_owner(self) -> None:
        if self._handle is not None and self._owner_pid != os.getpid():
            raise IntegrityError("A persistent CountStore handle cannot cross a process boundary.")

    def __enter__(self) -> CountStore:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def manifest(self) -> CountStoreManifest:
        if self._manifest is None:
            with h5py.File(self.path, "r") as handle:
                rows, features = map(int, handle.attrs["shape"])
                row_ids = handle["row_ids"][:]
                nnz = len(handle["X/data"])
                raw_features = tuple(
                    FeatureKey.model_validate(json.loads(value))
                    for value in handle["features"].asstr()[:]
                )
            payload = {
                "schema_version": 1,
                "store_id": "pending",
                "backend": "csr_hdf5",
                "rows": rows,
                "features": features,
                "nnz": nnz,
                "value_dtype": "int32",
                "row_ids_hash": _row_hash(row_ids),
                "feature_index_hash": _feature_hash(raw_features),
                "content_sha256": sha256_file(self.path),
                "relative_uri": self.path.name,
            }
            payload["store_id"] = contract_id(payload, id_field="store_id")
            self._manifest = CountStoreManifest.model_validate(payload)
        return self._manifest

    def verify(self, *, full: bool = True) -> CountStoreManifest:
        manifest = self.manifest
        if full and sha256_file(self.path) != manifest.content_sha256:
            raise IntegrityError("Count-store content hash mismatch.")
        with h5py.File(self.path, "r") as handle:
            shape = tuple(map(int, handle.attrs["shape"]))
            indptr = handle["X/indptr"][:]
            row_ids = handle["row_ids"][:]
            sorted_ids = handle["row_ids_sorted"][:]
            sorted_positions = handle["row_positions_sorted"][:]
            raw_features = tuple(
                FeatureKey.model_validate(json.loads(value))
                for value in handle["features"].asstr()[:]
            )
            if shape != (manifest.rows, manifest.features):
                raise IntegrityError("Count-store shape mismatch.")
            if len(indptr) != shape[0] + 1 or indptr[0] != 0 or indptr[-1] != manifest.nnz:
                raise IntegrityError("Invalid CSR indptr.")
            if np.any(np.diff(indptr) < 0):
                raise IntegrityError("Invalid CSR indptr.")
            indices = handle["X/indices"]
            data = handle["X/data"]
            if len(indices) != manifest.nnz or len(data) != manifest.nnz:
                raise IntegrityError("CSR data/index lengths disagree with the manifest.")
            if (
                data.dtype != np.dtype(np.int32)
                or indices.dtype != np.dtype(np.int32)
                or indptr.dtype != np.dtype(np.int64)
            ):
                raise IntegrityError("Count-store CSR dtypes differ from the int32/int64 contract.")
            chunk_nonzeros = 16_777_216
            for start in range(0, manifest.nnz, chunk_nonzeros):
                end = min(start + chunk_nonzeros, manifest.nnz)
                index_chunk = np.asarray(indices[start:end])
                data_chunk = np.asarray(data[start:end])
                if np.any(index_chunk < 0) or np.any(index_chunk >= shape[1]):
                    raise IntegrityError("Invalid CSR indices.")
                if not np.issubdtype(data_chunk.dtype, np.integer) or np.any(data_chunk < 0):
                    raise IntegrityError("Count-store values are not nonnegative integers.")
            if _row_hash(row_ids) != manifest.row_ids_hash:
                raise IntegrityError("Row identity hash mismatch.")
            if _feature_hash(raw_features) != manifest.feature_index_hash:
                raise IntegrityError("Feature identity hash mismatch.")
            if not np.array_equal(sorted_ids, np.sort(row_ids, kind="stable")):
                raise IntegrityError("Sorted row index is inconsistent.")
            if not np.array_equal(row_ids[sorted_positions], sorted_ids):
                raise IntegrityError("Sorted row-position index is inconsistent.")
        return manifest

    def _load(self) -> tuple[sparse.csr_matrix, np.ndarray[Any, Any]]:
        with h5py.File(self.path, "r") as handle:
            shape = tuple(map(int, handle.attrs["shape"]))
            matrix = sparse.csr_matrix(
                (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
                shape=shape,
            )
            row_ids = handle["row_ids"][:]
        return matrix, row_ids

    def rows(self, row_ids: np.ndarray[Any, Any]) -> SparseCountBatch:
        requested = np.asarray(row_ids, dtype=np.int64)
        self._assert_process_owner()
        manager = (
            nullcontext(self._handle) if self._handle is not None else h5py.File(self.path, "r")
        )
        with manager as handle:
            if self._row_index_cache is None:
                self._row_index_cache = (
                    np.asarray(handle["row_ids_sorted"][:], dtype=np.int64),
                    np.asarray(handle["row_positions_sorted"][:], dtype=np.int64),
                    np.asarray(handle["X/indptr"][:], dtype=np.int64),
                )
            sorted_ids, sorted_positions, offsets = self._row_index_cache
            found = np.searchsorted(sorted_ids, requested)
            valid = found < len(sorted_ids)
            candidates = np.where(valid)[0]
            valid[candidates] = sorted_ids[found[candidates]] == requested[candidates]
            if not np.all(valid):
                missing = requested[~valid].astype(int).tolist()
                raise KeyError(f"Unknown count-store row IDs: {missing[:10]}.")
            positions = sorted_positions[found]
            # Preparation encodes the immutable row universe in store order.
            # Read a contiguous CSR span once instead of issuing one compressed
            # HDF5 read per cell. Duplicated, reordered, and disjoint requests
            # retain the general exact-order path below.
            if len(positions) and (len(positions) == 1 or np.all(np.diff(positions) == 1)):
                first = int(positions[0])
                last = int(positions[-1]) + 1
                span_offsets = offsets[first : last + 1]
                data_start, data_end = int(span_offsets[0]), int(span_offsets[-1])
                matrix = sparse.csr_matrix(
                    (
                        handle["X/data"][data_start:data_end],
                        handle["X/indices"][data_start:data_end],
                        span_offsets.astype(np.int64) - data_start,
                    ),
                    shape=(len(positions), self.manifest.features),
                )
                return SparseCountBatch(
                    matrix=matrix,
                    row_ids=requested.copy(),
                    feature_index_hash=self.manifest.feature_index_hash,
                )
            # Target-balanced samplers commonly request a small number of
            # contiguous guide runs. Read those exact CSR spans rather than
            # falling back to one HDF5 operation per cell or scanning every
            # touched 16k-row window.
            unique_positions, inverse = np.unique(positions, return_inverse=True)
            run_breaks = np.where(np.diff(unique_positions) != 1)[0] + 1
            runs = np.split(unique_positions, run_breaks)
            if len(runs) <= max(64, len(unique_positions) // 8):
                run_blocks: list[sparse.csr_matrix] = []
                for run in runs:
                    first = int(run[0])
                    last = int(run[-1]) + 1
                    span_offsets = offsets[first : last + 1]
                    data_start, data_end = int(span_offsets[0]), int(span_offsets[-1])
                    run_blocks.append(
                        sparse.csr_matrix(
                            (
                                handle["X/data"][data_start:data_end],
                                handle["X/indices"][data_start:data_end],
                                span_offsets.astype(np.int64) - data_start,
                            ),
                            shape=(last - first, self.manifest.features),
                        )
                    )
                unique_matrix = sparse.vstack(run_blocks, format="csr")
                matrix = unique_matrix[inverse].tocsr()
                return SparseCountBatch(
                    matrix=matrix,
                    row_ids=requested.copy(),
                    feature_index_hash=self.manifest.feature_index_hash,
                )
            # Large interleaved selections are read in bounded source-row
            # windows. Never materialize the entire CSR payload: the intended
            # stores can be terabyte-scale.
            if len(positions) >= 4_096:
                window_rows = 16_384
                blocks: list[sparse.csr_matrix] = []
                block_positions: list[np.ndarray[Any, Any]] = []
                for window_start in range(0, self.manifest.rows, window_rows):
                    window_end = min(window_start + window_rows, self.manifest.rows)
                    left = np.searchsorted(unique_positions, window_start, side="left")
                    right = np.searchsorted(unique_positions, window_end, side="left")
                    if left == right:
                        continue
                    wanted = unique_positions[left:right]
                    span_offsets = offsets[window_start : window_end + 1]
                    data_start, data_end = int(span_offsets[0]), int(span_offsets[-1])
                    source_block = sparse.csr_matrix(
                        (
                            handle["X/data"][data_start:data_end],
                            handle["X/indices"][data_start:data_end],
                            span_offsets.astype(np.int64) - data_start,
                        ),
                        shape=(window_end - window_start, self.manifest.features),
                    )
                    blocks.append(source_block[wanted - window_start].tocsr())
                    block_positions.append(wanted)
                if not blocks:
                    matrix = sparse.csr_matrix((0, self.manifest.features), dtype=np.float32)
                else:
                    unique_matrix = sparse.vstack(blocks, format="csr")
                    concatenated = np.concatenate(block_positions)
                    reorder = np.argsort(concatenated)
                    unique_matrix = unique_matrix[reorder]
                    matrix = unique_matrix[inverse].tocsr()
                return SparseCountBatch(
                    matrix=matrix,
                    row_ids=requested.copy(),
                    feature_index_hash=self.manifest.feature_index_hash,
                )
            unique_positions = np.unique(positions)
            row_payload: dict[int, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}
            for position in unique_positions:
                start, end = int(offsets[position]), int(offsets[position + 1])
                row_payload[int(position)] = (
                    handle["X/indices"][start:end],
                    handle["X/data"][start:end],
                )
        indices: list[np.ndarray[Any, Any]] = []
        values: list[np.ndarray[Any, Any]] = []
        indptr = [0]
        for position in positions:
            row_indices, row_values = row_payload[int(position)]
            indices.append(row_indices)
            values.append(row_values)
            indptr.append(indptr[-1] + len(row_values))
        concatenated_indices = (
            np.concatenate(indices).astype(np.int32, copy=False)
            if indices
            else np.empty(0, dtype=np.int32)
        )
        concatenated_values = (
            np.concatenate(values).astype(np.int32, copy=False)
            if values
            else np.empty(0, dtype=np.int32)
        )
        matrix = sparse.csr_matrix(
            (concatenated_values, concatenated_indices, np.asarray(indptr, dtype=np.int64)),
            shape=(len(requested), self.manifest.features),
        )
        return SparseCountBatch(
            matrix=matrix,
            row_ids=requested.copy(),
            feature_index_hash=self.manifest.feature_index_hash,
        )

    def row_ids(self) -> np.ndarray[Any, Any]:
        self._assert_process_owner()
        manager = (
            nullcontext(self._handle) if self._handle is not None else h5py.File(self.path, "r")
        )
        with manager as handle:
            return cast(np.ndarray[Any, Any], handle["row_ids"][:])

    def iter_batches(
        self,
        ordered_row_ids: np.ndarray[Any, Any],
        *,
        batch_size: int,
        cursor: int = 0,
    ) -> Iterator[tuple[int, SparseCountBatch]]:
        if batch_size <= 0 or cursor < 0:
            raise ValueError("batch_size must be positive and cursor nonnegative.")
        ids = np.asarray(ordered_row_ids, dtype=np.int64)
        while cursor < len(ids):
            end = min(cursor + batch_size, len(ids))
            yield end, self.rows(ids[cursor:end])
            cursor = end
