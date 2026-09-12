"""Source-backed compact verification with durable writer restart for Dev37."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDurableRestartReceiptV4,
    G00CMaterializationReceiptV3,
    G00CMaterializationReceiptV4,
)
from ..errors import IntegrityError
from .g00c_evidence_v4 import verify_g00c_restart_v4
from .g00c_materialization_v3 import (
    PHYSICAL_COLUMNS,
    _hash_int64,
    _physical_runs,
    _verify_h5,
)
from .g00c_v2 import _expected_physical_order
from .g00c_v3 import _path, _read_model
from .virtual import VirtualCanonicalCountStore


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV3, role: str) -> str:
    matches = [
        item.artifact.sha256
        for item in authority.implementation.implementations
        if item.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev37 authority has no unique {role} implementation.")
    return matches[0]


def _role_rows_v4(
    root: Path, authority: G00CD1ExecutionAuthorityFreezeV3
) -> dict[str, np.ndarray[Any, Any]]:
    table = pd.read_parquet(_path(root, authority.row_role_freeze))
    if tuple(table.columns) != ("row_id", "role") or table["row_id"].duplicated().any():
        raise IntegrityError("Dev37 materializer found malformed row-role authority.")
    return {
        role: frame["row_id"].to_numpy(dtype=np.int64)
        for role, frame in table.groupby(table["role"].astype(str), sort=False)
    }


def verify_g00c_materialization_v4(
    root: Path,
    store: VirtualCanonicalCountStore,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    receipt: G00CMaterializationReceiptV4,
    *,
    expected_selected_feature_ids: tuple[str, ...],
    expected_selected_rows: np.ndarray[Any, Any],
) -> G00CMaterializationReceiptV3:
    """Verify source equality while replacing Dev36's same-reference restart shortcut."""

    base = _read_model(root, receipt.base_materialization_receipt, G00CMaterializationReceiptV3)
    restart = _read_model(root, receipt.writer_restart_receipt, G00CDurableRestartReceiptV4)
    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.source_binding_id != authority.source_plane.binding_id
        or receipt.base_materialization_receipt_id != base.receipt_id
        or receipt.writer_restart_receipt_id != restart.receipt_id
        or restart.component != "writer"
        or base.execution_authority_id != authority.authority_id
        or base.verifier_implementation_sha256
        != _implementation_hash(authority, "materialization_verifier")
    ):
        raise IntegrityError("Dev37 materialization evidence is cross-wired.")
    features = pd.read_parquet(_path(root, base.selected_features))
    selected = pd.read_parquet(_path(root, base.selected_rows))
    if (
        tuple(features.columns) != ("rank", "canonical_index", "feature_id")
        or len(features) != base.selected_feature_count
        or tuple(features["feature_id"].astype(str)) != expected_selected_feature_ids
        or not np.array_equal(
            features["rank"].to_numpy(dtype=np.int64), np.arange(1, len(features) + 1)
        )
        or tuple(selected.columns) != ("row_id",)
        or not np.array_equal(selected["row_id"].to_numpy(dtype=np.int64), expected_selected_rows)
        or _hash_int64(expected_selected_rows, ordered=False) != base.selected_row_set_sha256
    ):
        raise IntegrityError("Dev37 materialization selection differs from verified results.")
    roles = _role_rows_v4(root, authority)
    protected = roles["protected_heldout_stimulated"]
    compact_rows = np.concatenate(
        (expected_selected_rows, roles["training_validation"], roles["heldout_source_query"])
    )
    if (
        len(np.unique(compact_rows)) != len(compact_rows)
        or np.intersect1d(compact_rows, protected, assume_unique=False).size
    ):
        raise IntegrityError("Dev37 compact rows overlap protected stimulated expression.")
    expected_order, _ = _expected_physical_order(store, compact_rows)
    if (
        len(expected_order) != base.compact_row_count
        or _hash_int64(expected_order, ordered=True) != base.compact_ordered_row_ids_sha256
        or _hash_int64(protected, ordered=False) != base.protected_row_ids_sha256
    ):
        raise IntegrityError("Dev37 compact/protected row hashes differ.")
    physical = pd.read_parquet(_path(root, base.physical_runs))
    if tuple(physical.columns) != PHYSICAL_COLUMNS or not physical.equals(
        _physical_runs(store, expected_order)
    ):
        raise IntegrityError("Dev37 physical-run evidence differs from G00B.")
    primary_indices = features["canonical_index"].to_numpy(dtype=np.int64)
    if (
        len(np.unique(primary_indices)) != len(primary_indices)
        or np.any(primary_indices < 0)
        or np.any(primary_indices >= store.manifest.features)
        or base.puro_r_canonical_index in set(primary_indices.tolist())
        or base.puro_r_canonical_index != authority.source_plane.puro_r_canonical_index
    ):
        raise IntegrityError("Dev37 primary/PuroR feature separation is invalid.")
    data_dtype = np.dtype(base.compact_data_dtype)
    index_dtype = np.dtype(base.compact_index_dtype)
    _verify_h5(
        _path(root, base.compact_payload),
        store=store,
        expected_rows=expected_order,
        feature_ids=list(expected_selected_feature_ids),
        feature_indices=primary_indices,
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        maximum_count=base.maximum_observed_count,
        block_rows=base.compact_verification_block_rows,
    )
    _verify_h5(
        _path(root, base.puro_r_sidecar),
        store=store,
        expected_rows=expected_order,
        feature_ids=[base.puro_r_feature_id],
        feature_indices=np.asarray([base.puro_r_canonical_index], dtype=np.int64),
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        maximum_count=base.maximum_observed_count,
        block_rows=base.compact_verification_block_rows,
    )
    if (
        base.compact_payload != base.uninterrupted_compact_payload
        or base.puro_r_sidecar != base.uninterrupted_puro_r_sidecar
        or restart.uninterrupted_outputs
        != (base.uninterrupted_compact_payload, base.uninterrupted_puro_r_sidecar)
        or restart.resumed_outputs != (base.resumed_compact_payload, base.resumed_puro_r_sidecar)
    ):
        raise IntegrityError("Dev37 writer restart receipt does not bind the compact outputs.")
    verify_g00c_restart_v4(root, restart)
    reload_receipt = json.loads(_path(root, base.reload_receipt).read_text())
    if reload_receipt != {
        "compact_payload_sha256": base.compact_payload.sha256,
        "primary_features": base.selected_feature_count,
        "puro_r_sidecar_sha256": base.puro_r_sidecar.sha256,
        "rows": base.compact_row_count,
        "schema_version": 1,
        "sidecar_features": 1,
        "status": "pass",
    }:
        raise IntegrityError("Dev37 compact reload receipt differs from source-verified bytes.")
    return base
