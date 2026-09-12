"""Independent monitor, source-access, and durable-restart verification for Dev37."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDurableRestartReceiptV4,
    G00CMonitorFreezeV1,
    G00CProcessTreeMonitorReceiptV4,
    G00CSourceAccessLedgerReceiptV4,
)
from ..errors import IntegrityError
from .g00c_v3 import _path
from .virtual import VirtualCanonicalCountStore

MONITOR_COLUMNS = (
    "monotonic_ns",
    "root_pid",
    "process_tree_rss_bytes",
    "descendants",
    "readable",
)
ACCESS_COLUMNS = (
    "sequence",
    "monotonic_ns",
    "role",
    "row_id",
    "source_index",
    "checkpoint",
)


def _ordered_int64_hash(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def verify_g00c_restart_v4(root: Path, receipt: G00CDurableRestartReceiptV4) -> None:
    """Verify fresh-process evidence, durable checkpoint bytes, and distinct equal outputs."""

    _path(root, receipt.durable_checkpoint)
    interrupted = json.loads(
        _path(root, receipt.interrupted_no_final_publication_receipt).read_text()
    )
    uninterrupted_process = json.loads(
        _path(root, receipt.uninterrupted_process_receipt).read_text()
    )
    resumed_process = json.loads(_path(root, receipt.resumed_process_receipt).read_text())
    if interrupted != {
        "attempt_id": receipt.resumed_attempt_id,
        "final_payload_published": False,
        "status": "interrupted_checkpoint_committed",
    }:
        raise IntegrityError("Dev37 interrupted attempt publication evidence differs.")
    expected_process_keys = {"attempt_id", "exit_code", "pid", "status"}
    if (
        set(uninterrupted_process) != expected_process_keys
        or set(resumed_process) != expected_process_keys
        or uninterrupted_process["attempt_id"] != receipt.uninterrupted_attempt_id
        or resumed_process["attempt_id"] != receipt.resumed_attempt_id
        or uninterrupted_process["exit_code"] != 0
        or resumed_process["exit_code"] != 0
        or uninterrupted_process["status"] != "complete"
        or resumed_process["status"] != "complete"
        or int(uninterrupted_process["pid"]) == int(resumed_process["pid"])
    ):
        raise IntegrityError("Dev37 restart process evidence is not independent and successful.")
    for uninterrupted, resumed in zip(
        receipt.uninterrupted_outputs, receipt.resumed_outputs, strict=True
    ):
        uninterrupted_path = _path(root, uninterrupted)
        resumed_path = _path(root, resumed)
        if (
            uninterrupted_path.samefile(resumed_path)
            or uninterrupted_path.read_bytes() != resumed_path.read_bytes()
        ):
            raise IntegrityError("Dev37 restart outputs are not distinct byte-identical files.")


def verify_g00c_monitor_v4(
    root: Path,
    freeze: G00CMonitorFreezeV1,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    receipt: G00CProcessTreeMonitorReceiptV4,
) -> pd.DataFrame:
    """Derive resource and temporal gates from an out-of-process trace."""

    if receipt.execution_authority_id != authority.authority_id:
        raise IntegrityError("Dev37 monitor receipt binds another authority.")
    trace = pd.read_parquet(_path(root, receipt.trace))
    if tuple(trace.columns) != MONITOR_COLUMNS or len(trace) < 2:
        raise IntegrityError("Dev37 monitor trace has another schema.")
    monotonic = trace["monotonic_ns"].to_numpy(dtype=np.int64)
    if np.any(np.diff(monotonic) <= 0):
        raise IntegrityError("Dev37 monitor timestamps are not strictly increasing.")
    readable = trace["readable"].astype(bool).to_numpy()
    unreadable = int((~readable).sum())
    consecutive = 0
    maximum_consecutive = 0
    for value in readable:
        consecutive = 0 if value else consecutive + 1
        maximum_consecutive = max(maximum_consecutive, consecutive)
    gaps_ms = np.diff(monotonic) / 1_000_000.0
    maximum_gap = float(gaps_ms.max())
    maximum_rss = int(trace.loc[readable, "process_tree_rss_bytes"].max())
    descendants = int(trace.loc[readable, "descendants"].max())
    if (
        not np.all(trace["root_pid"].to_numpy(dtype=np.int64) == receipt.monitored_root_pid)
        or receipt.samples != len(trace)
        or receipt.maximum_process_tree_rss_bytes != maximum_rss
        or receipt.unreadable_samples != unreadable
        or receipt.maximum_consecutive_unreadable_samples != maximum_consecutive
        or receipt.maximum_temporal_gap_milliseconds != maximum_gap
        or receipt.descendants_observed != descendants
        or monotonic[0] != receipt.monitor_start_monotonic_ns
        or monotonic[-1] != receipt.monitor_end_monotonic_ns
    ):
        raise IntegrityError("Dev37 monitor receipt differs from its trace.")
    if (
        unreadable > freeze.maximum_unreadable_samples
        or unreadable / len(trace) > freeze.maximum_unreadable_fraction
        or maximum_consecutive > freeze.maximum_consecutive_unreadable_samples
        or maximum_gap > freeze.maximum_temporal_gap_milliseconds
        or maximum_rss > authority.monitored_process_tree_ceiling_bytes
        or maximum_rss > freeze.maximum_process_tree_rss_bytes
    ):
        raise IntegrityError("Dev37 monitor evidence violates the frozen execution ceiling.")
    return trace


def verify_g00c_source_access_v4(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    store: VirtualCanonicalCountStore,
    receipt: G00CSourceAccessLedgerReceiptV4,
) -> pd.DataFrame:
    """Verify every role-labelled source read and derive protected access as exactly zero."""

    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.source_binding_id != authority.source_plane.binding_id
    ):
        raise IntegrityError("Dev37 source-access ledger binds another authority/source.")
    ledger = pd.read_parquet(_path(root, receipt.ledger))
    if (
        tuple(ledger.columns) != ACCESS_COLUMNS
        or len(ledger) != receipt.access_rows
        or not np.array_equal(
            ledger["sequence"].to_numpy(dtype=np.int64), np.arange(len(ledger), dtype=np.int64)
        )
    ):
        raise IntegrityError("Dev37 source-access ledger has another row surface.")
    timestamps = ledger["monotonic_ns"].to_numpy(dtype=np.int64)
    if len(timestamps) and np.any(np.diff(timestamps) < 0):
        raise IntegrityError("Dev37 source-access ledger timestamps are not monotonic.")
    roles = pd.read_parquet(_path(root, authority.row_role_freeze))
    role_sets = {
        str(role): set(frame["row_id"].to_numpy(dtype=np.int64).tolist())
        for role, frame in roles.groupby(roles["role"].astype(str), sort=False)
    }
    protected = next(
        item for item in authority.row_roles if item.role == "protected_heldout_stimulated"
    )
    if receipt.protected_row_ids_sha256 != protected.row_ids_hash:
        raise IntegrityError("Dev37 source-access receipt binds another protected row set.")
    observed_hashes: dict[str, str] = {}
    for role, frame in ledger.groupby(ledger["role"].astype(str), sort=True):
        row_ids = frame["row_id"].to_numpy(dtype=np.int64)
        observed_hashes[str(role)] = _ordered_int64_hash(row_ids)
        allowed_role = {
            "feature_ranking_training_fit": "training_fit",
            "refit_training_fit": "training_fit",
            "refit_training_validation": "training_validation",
            "materialization_training_fit": "training_fit",
            "materialization_training_validation": "training_validation",
            "materialization_heldout_source_query": "heldout_source_query",
        }.get(str(role))
        if allowed_role is None or not set(row_ids.tolist()) <= role_sets[allowed_role]:
            raise IntegrityError("Dev37 source ledger contains a row outside its declared role.")
    if observed_hashes != receipt.role_row_hashes:
        raise IntegrityError("Dev37 source-access role hashes differ from the full ledger.")
    if role_sets["protected_heldout_stimulated"] & set(ledger["row_id"].astype(int)):
        raise IntegrityError("Dev37 source ledger contains protected stimulated expression reads.")
    sorted_ids, source_indices, _ = store._locator()
    row_ids = ledger["row_id"].to_numpy(dtype=np.int64)
    positions = np.searchsorted(sorted_ids, row_ids)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], row_ids):
        raise IntegrityError("Dev37 source ledger contains unknown G00B rows.")
    observed_sources = ledger["source_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_sources, source_indices[positions].astype(np.int64)):
        raise IntegrityError("Dev37 source ledger source indices differ from G00B.")
    checkpoints = np.asarray(
        [source.checkpoint for source in store.manifest.sources], dtype=object
    )[observed_sources]
    if not np.array_equal(checkpoints.astype(str), ledger["checkpoint"].astype(str)):
        raise IntegrityError("Dev37 source ledger checkpoints differ from G00B.")
    return ledger
