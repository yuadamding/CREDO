"""Direct, source-backed compact materialization verification for Dev36."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV2,
    G00CD1ExecutionAuthorityFreezeV3,
    G00CMaterializationReceiptV3,
)
from ..errors import IntegrityError
from .g00c_v2 import _expected_physical_order
from .g00c_v3 import _path
from .virtual import VirtualCanonicalCountStore

H5_SCHEMA = {"row_ids", "feature_ids", "indptr", "indices", "data"}
PHYSICAL_COLUMNS = (
    "block_index",
    "start_row_offset",
    "stop_row_offset",
    "donor_id",
    "checkpoint",
    "target_id",
    "guide_id",
)


def _hash_int64(values: np.ndarray[Any, Any], *, ordered: bool) -> str:
    array = np.asarray(values, dtype="<i8")
    if not ordered:
        array = np.sort(array, kind="stable")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV2, role: str) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev36 authority has no unique {role} implementation.")
    return matches[0]


def _role_rows(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV2 | G00CD1ExecutionAuthorityFreezeV3,
) -> dict[str, np.ndarray[Any, Any]]:
    table = pd.read_parquet(_path(root, authority.row_role_freeze))
    if tuple(table.columns) != ("row_id", "role") or table["row_id"].duplicated().any():
        raise IntegrityError("Dev36 materializer found malformed row-role authority.")
    return {
        role: frame["row_id"].to_numpy(dtype=np.int64)
        for role, frame in table.groupby(table["role"].astype(str), sort=False)
    }


def _physical_runs(
    store: VirtualCanonicalCountStore, ordered_rows: np.ndarray[Any, Any]
) -> pd.DataFrame:
    locator_ids, source_indices, source_rows = store._locator()
    positions = np.searchsorted(locator_ids, ordered_rows)
    if np.any(positions >= len(locator_ids)) or not np.array_equal(
        locator_ids[positions], ordered_rows
    ):
        raise IntegrityError("Dev36 compact rows are absent from the source locator.")
    locator_path = store.path / store.manifest.row_locator.relative_uri
    position_order = np.argsort(positions, kind="stable")
    sorted_positions = positions[position_order]
    inverse = np.argsort(position_order, kind="stable")
    with h5py.File(locator_path, "r") as handle:
        guide_codes = np.asarray(handle["guide_codes_sorted"][sorted_positions], dtype=np.int64)[
            inverse
        ]
        target_codes = np.asarray(handle["target_codes_sorted"][sorted_positions], dtype=np.int64)[
            inverse
        ]
        guide_ids = handle["guide_ids"].asstr()[:]
        target_ids = handle["target_ids"].asstr()[:]
    selected_sources = source_indices[positions]
    donors = np.asarray([store.manifest.sources[int(index)].donor_id for index in selected_sources])
    checkpoints = np.asarray(
        [store.manifest.sources[int(index)].checkpoint for index in selected_sources]
    )
    targets = target_ids[target_codes]
    guides = guide_ids[guide_codes]
    physical_rows = source_rows[positions]
    expected_sort = np.lexsort((ordered_rows, physical_rows, guides, targets, checkpoints, donors))
    if not np.array_equal(expected_sort, np.arange(len(ordered_rows))):
        raise IntegrityError("Dev36 compact rows are not in frozen physical order.")
    keys = np.column_stack((donors, checkpoints, targets, guides))
    starts = np.ones(len(ordered_rows), dtype=bool)
    if len(ordered_rows) > 1:
        starts[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    start_offsets = np.flatnonzero(starts)
    stop_offsets = np.r_[start_offsets[1:], len(ordered_rows)]
    rows: list[dict[str, object]] = []
    for block, (start, stop) in enumerate(zip(start_offsets, stop_offsets, strict=True)):
        rows.append(
            {
                "block_index": block,
                "start_row_offset": int(start),
                "stop_row_offset": int(stop),
                "donor_id": str(donors[start]),
                "checkpoint": str(checkpoints[start]),
                "target_id": str(targets[start]),
                "guide_id": str(guides[start]),
            }
        )
    return pd.DataFrame(rows, columns=PHYSICAL_COLUMNS)


def _verify_h5(
    path: Path,
    *,
    store: VirtualCanonicalCountStore,
    expected_rows: np.ndarray[Any, Any],
    feature_ids: list[str],
    feature_indices: np.ndarray[Any, Any],
    data_dtype: np.dtype[Any],
    index_dtype: np.dtype[Any],
    maximum_count: int,
    block_rows: int,
) -> None:
    with h5py.File(path, "r") as handle:
        if set(handle) != H5_SCHEMA:
            raise IntegrityError("Dev36 compact HDF5 has an invalid dataset schema.")
        row_ids = np.asarray(handle["row_ids"][:], dtype=np.int64)
        observed_features = handle["feature_ids"].asstr()[:].tolist()
        indptr = handle["indptr"]
        indices = handle["indices"]
        data = handle["data"]
        if (
            not np.array_equal(row_ids, expected_rows)
            or observed_features != feature_ids
            or data.dtype != data_dtype
            or indices.dtype != index_dtype
            or indptr.dtype != np.dtype("int64")
            or len(indptr) != len(row_ids) + 1
            or int(indptr[0]) != 0
            or int(indptr[-1]) != len(indices)
            or len(indices) != len(data)
        ):
            raise IntegrityError("Dev36 compact HDF5 identity, dtype, or CSR offsets differ.")
        for start in range(0, len(row_ids), block_rows):
            stop = min(start + block_rows, len(row_ids))
            offsets = np.asarray(indptr[start : stop + 1], dtype=np.int64)
            first, last = int(offsets[0]), int(offsets[-1])
            block_indices = np.asarray(indices[first:last])
            block_data = np.asarray(data[first:last])
            if (
                np.any(np.diff(offsets) < 0)
                or np.any(block_indices < 0)
                or np.any(block_indices >= len(feature_ids))
                or np.any(block_data < 0)
                or (len(block_data) and int(block_data.max()) > maximum_count)
            ):
                raise IntegrityError("Dev36 compact HDF5 contains invalid sparse counts.")
            observed = sparse.csr_matrix(
                (block_data, block_indices.astype(np.int64), offsets - first),
                shape=(stop - start, len(feature_ids)),
            )
            expected = store.rows(row_ids[start:stop]).matrix[:, feature_indices]
            if observed.shape != expected.shape or (observed != expected).nnz:
                raise IntegrityError("Dev36 compact bytes differ from the virtual source plane.")


def verify_g00c_materialization_v3(
    root: Path,
    store: VirtualCanonicalCountStore,
    authority: G00CD1ExecutionAuthorityFreezeV2,
    receipt: G00CMaterializationReceiptV3,
    *,
    expected_selected_feature_ids: tuple[str, ...],
    expected_selected_rows: np.ndarray[Any, Any],
) -> None:
    """Open and compare every compact/sidecar value; no callback identity is trusted."""

    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.verifier_implementation_sha256
        != _implementation_hash(authority, "materialization_verifier")
    ):
        raise IntegrityError("Dev36 materialization receipt is cross-wired.")
    features = pd.read_parquet(_path(root, receipt.selected_features))
    selected = pd.read_parquet(_path(root, receipt.selected_rows))
    if (
        tuple(features.columns) != ("rank", "canonical_index", "feature_id")
        or len(features) != receipt.selected_feature_count
        or tuple(features["feature_id"].astype(str)) != expected_selected_feature_ids
        or not np.array_equal(
            features["rank"].to_numpy(dtype=np.int64),
            np.arange(1, len(features) + 1),
        )
        or tuple(selected.columns) != ("row_id",)
        or not np.array_equal(selected["row_id"].to_numpy(dtype=np.int64), expected_selected_rows)
        or _hash_int64(expected_selected_rows, ordered=False) != receipt.selected_row_set_sha256
    ):
        raise IntegrityError("Dev36 materialization selection differs from verified results.")
    roles = _role_rows(root, authority)
    protected = roles["protected_heldout_stimulated"]
    compact_rows = np.concatenate(
        (expected_selected_rows, roles["training_validation"], roles["heldout_source_query"])
    )
    if (
        len(np.unique(compact_rows)) != len(compact_rows)
        or np.intersect1d(compact_rows, protected, assume_unique=False).size
    ):
        raise IntegrityError("Dev36 compact row set overlaps protected expression rows.")
    expected_order, _ = _expected_physical_order(store, compact_rows)
    if (
        len(expected_order) != receipt.compact_row_count
        or _hash_int64(expected_order, ordered=True) != receipt.compact_ordered_row_ids_sha256
        or _hash_int64(protected, ordered=False) != receipt.protected_row_ids_sha256
    ):
        raise IntegrityError("Dev36 compact/protected row hashes differ.")
    physical = pd.read_parquet(_path(root, receipt.physical_runs))
    if tuple(physical.columns) != PHYSICAL_COLUMNS or not physical.equals(
        _physical_runs(store, expected_order)
    ):
        raise IntegrityError("Dev36 physical run evidence differs from source metadata.")
    primary_indices = features["canonical_index"].to_numpy(dtype=np.int64)
    if (
        len(np.unique(primary_indices)) != len(primary_indices)
        or np.any(primary_indices < 0)
        or np.any(primary_indices >= store.manifest.features)
        or receipt.puro_r_canonical_index in set(primary_indices.tolist())
        or receipt.puro_r_canonical_index >= store.manifest.features
    ):
        raise IntegrityError("Dev36 primary/PuroR feature separation is invalid.")
    compact_path = _path(root, receipt.compact_payload)
    sidecar_path = _path(root, receipt.puro_r_sidecar)
    data_dtype = np.dtype(receipt.compact_data_dtype)
    index_dtype = np.dtype(receipt.compact_index_dtype)
    _verify_h5(
        compact_path,
        store=store,
        expected_rows=expected_order,
        feature_ids=list(expected_selected_feature_ids),
        feature_indices=primary_indices,
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        maximum_count=receipt.maximum_observed_count,
        block_rows=receipt.compact_verification_block_rows,
    )
    _verify_h5(
        sidecar_path,
        store=store,
        expected_rows=expected_order,
        feature_ids=[receipt.puro_r_feature_id],
        feature_indices=np.asarray([receipt.puro_r_canonical_index], dtype=np.int64),
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        maximum_count=receipt.maximum_observed_count,
        block_rows=receipt.compact_verification_block_rows,
    )
    if not (
        receipt.uninterrupted_compact_payload == receipt.compact_payload
        and receipt.resumed_compact_payload == receipt.compact_payload
        and receipt.uninterrupted_puro_r_sidecar == receipt.puro_r_sidecar
        and receipt.resumed_puro_r_sidecar == receipt.puro_r_sidecar
    ):
        raise IntegrityError("Dev36 interrupted and uninterrupted writers are not byte-identical.")
    for artifact in (
        receipt.uninterrupted_compact_payload,
        receipt.resumed_compact_payload,
        receipt.uninterrupted_puro_r_sidecar,
        receipt.resumed_puro_r_sidecar,
    ):
        _path(root, artifact)
    access = json.loads(_path(root, receipt.protected_access_receipt).read_text())
    if access != {
        "compact_rows_read": receipt.compact_row_count,
        "protected_expression_reads": 0,
        "protected_row_ids_sha256": receipt.protected_row_ids_sha256,
        "schema_version": 1,
        "status": "pass",
    }:
        raise IntegrityError("Dev36 protected-access receipt does not prove zero reads.")
    reload_receipt = json.loads(_path(root, receipt.reload_receipt).read_text())
    if reload_receipt != {
        "compact_payload_sha256": receipt.compact_payload.sha256,
        "primary_features": receipt.selected_feature_count,
        "puro_r_sidecar_sha256": receipt.puro_r_sidecar.sha256,
        "rows": receipt.compact_row_count,
        "schema_version": 1,
        "sidecar_features": 1,
        "status": "pass",
    }:
        raise IntegrityError("Dev36 full-reload receipt differs from verified bytes.")
