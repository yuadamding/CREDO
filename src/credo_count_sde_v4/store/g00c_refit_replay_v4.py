"""Checkpoint-indexed, source-derived refit statistics and replay for Dev37."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV3,
    G00CRefitReplayReceiptV4,
    G00CRefitSeedScheduleV1,
    G00CSamplerPlanV4,
)
from ..errors import IntegrityError
from .g00c_selection import checkpoint_multinomial_refit_common_support
from .g00c_v3 import REFIT_V3_COLUMNS, _path, _read_model
from .virtual import VirtualCanonicalCountStore

REPLAY_COLUMNS = (
    "candidate_kind",
    "draw_id",
    "candidate_value",
    "validation_total_count",
    "validation_nll_sum",
    "validation_nll_per_count",
    "final_state_hash",
)
AccessCallback = Callable[[str, np.ndarray[Any, Any]], None]


def _ordered_hash(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _array_hash(values: np.ndarray[Any, Any]) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _checkpoint_codes(
    store: VirtualCanonicalCountStore, row_ids: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    sorted_ids, source_indices, _ = store._locator()
    positions = np.searchsorted(sorted_ids, row_ids)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], row_ids):
        raise IntegrityError("Dev37 refit row is absent from G00B.")
    checkpoint_map = {"Rest": 0, "Stim8hr": 1, "Stim48hr": 2}
    source_codes = np.asarray(
        [checkpoint_map[item.checkpoint] for item in store.manifest.sources], dtype=np.int8
    )
    return source_codes[source_indices[positions].astype(np.int64)]


def _thin_and_accumulate(
    matrix: sparse.csr_matrix,
    checkpoint_codes: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    *,
    seed: int,
) -> tuple[np.ndarray[Any, Any], str]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    thinned_data = rng.binomial(matrix.data.astype(np.int64), 0.5).astype(np.float64)
    digest = hashlib.sha256(np.asarray(thinned_data, dtype="<f8").tobytes(order="C")).hexdigest()
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    answer = np.zeros((3, matrix.shape[1]), dtype=np.float64)
    for checkpoint in range(3):
        selected = checkpoint_codes[rows] == checkpoint
        answer[checkpoint] = np.bincount(
            matrix.indices[selected],
            weights=thinned_data[selected] * weights[rows[selected]],
            minlength=matrix.shape[1],
        )
    return answer, digest


def _read_selected_rows(
    store: VirtualCanonicalCountStore,
    row_ids: np.ndarray[Any, Any],
    feature_indices: np.ndarray[Any, Any],
    *,
    batch_size: int,
) -> sparse.csr_matrix:
    blocks = []
    for start in range(0, len(row_ids), batch_size):
        blocks.append(store.rows(row_ids[start : start + batch_size]).matrix[:, feature_indices])
    return sparse.vstack(blocks, format="csr")


def derive_checkpoint_statistics_v4(
    store: VirtualCanonicalCountStore,
    plan: G00CSamplerPlanV4,
    trace: pd.DataFrame,
    ordered_training_rows: np.ndarray[Any, Any],
    validation_rows: np.ndarray[Any, Any],
    feature_indices: np.ndarray[Any, Any],
    schedule: G00CRefitSeedScheduleV1,
    *,
    candidate_kind: str,
    batch_size: int = 4096,
    access_callback: AccessCallback | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], tuple[str, ...], str, str]:
    """Derive [draw,candidate,checkpoint,gene] counts from verified source rows."""

    candidates = tuple(
        sorted(
            {
                int(item.candidate_value)
                for item in plan.entries
                if item.candidate_kind == candidate_kind
            }
        )
    )
    if not candidates or len(feature_indices) != 4096:
        raise IntegrityError("Dev37 refit statistics lack candidates or the 4,096-gene support.")
    candidate_index = {value: index for index, value in enumerate(candidates)}
    training = np.zeros((59, len(candidates), 3, 4096), dtype=np.float64)
    validation = np.zeros((59, 3, 4096), dtype=np.float64)
    fit_hashes: list[str] = []
    for candidate in candidates:
        prefix = (
            ordered_training_rows[:1_000_000]
            if candidate_kind == "feature_count"
            else ordered_training_rows[:candidate]
        )
        fit_hashes.append(_ordered_hash(prefix))
    thinning_digest = hashlib.sha256()
    entries = [
        (index, item)
        for index, item in enumerate(plan.entries)
        if item.candidate_kind == candidate_kind
    ]
    for entry_index, entry in entries:
        frame = trace.loc[trace["entry_index"].astype(int) == entry_index].sort_values(
            "draw_index", kind="stable"
        )
        row_ids = frame["row_id"].to_numpy(dtype=np.int64)
        weights = frame["inverse_probability_weight"].to_numpy(dtype=np.float64)
        if len(row_ids) != entry.expected_trace_rows:
            raise IntegrityError("Dev37 refit trace differs from its frozen entry size.")
        if access_callback is not None:
            access_callback("refit_training_fit", row_ids)
        matrix = _read_selected_rows(store, row_ids, feature_indices, batch_size=batch_size).astype(
            np.int64
        )
        checkpoints = _checkpoint_codes(store, row_ids)
        counts, thinning_hash = _thin_and_accumulate(
            matrix,
            checkpoints,
            weights,
            seed=int(schedule.records[entry.refit_draw_id].thinning),
        )
        training[
            entry.refit_draw_id,
            candidate_index[int(entry.candidate_value)],
        ] = counts
        thinning_digest.update(bytes.fromhex(thinning_hash))
    validation_checkpoints = _checkpoint_codes(store, validation_rows)
    for draw in range(59):
        if access_callback is not None:
            access_callback("refit_training_validation", validation_rows)
        matrix = _read_selected_rows(
            store, validation_rows, feature_indices, batch_size=batch_size
        ).astype(np.int64)
        counts, thinning_hash = _thin_and_accumulate(
            matrix,
            validation_checkpoints,
            np.ones(len(validation_rows), dtype=np.float64),
            seed=int(schedule.records[draw].validation_evaluation),
        )
        validation[draw] = counts
        thinning_digest.update(bytes.fromhex(thinning_hash))
    if candidate_kind == "feature_count" and not all(
        np.array_equal(training[:, 0], training[:, index]) for index in range(1, len(candidates))
    ):
        raise IntegrityError("Dev37 feature statistics differ before prefix collapse.")
    return (
        training,
        validation,
        tuple(fit_hashes),
        _ordered_hash(validation_rows),
        thinning_digest.hexdigest(),
    )


def _targets(records: pd.DataFrame, receipt: G00CRefitReplayReceiptV4) -> pd.DataFrame:
    audits = set(receipt.preregistered_audit_draw_ids)
    return records.loc[
        (records["candidate_value"].astype(int) == receipt.selected_candidate)
        | (records["candidate_value"].astype(int) == receipt.reference_candidate)
        | records["draw_id"].astype(int).isin(audits)
    ].sort_values(["draw_id", "candidate_value"], kind="stable")


def recompute_refit_rows_v4(
    records: pd.DataFrame,
    receipt: G00CRefitReplayReceiptV4,
    training: np.ndarray[Any, Any],
    validation: np.ndarray[Any, Any],
) -> pd.DataFrame:
    """Replay checkpoint-specific common-support intercepts from derived arrays."""

    candidate_index = {value: index for index, value in enumerate(receipt.candidate_values)}
    rows: list[dict[str, object]] = []
    for record in _targets(records, receipt).itertuples(index=False):
        draw = int(record.draw_id)
        candidate = int(record.candidate_value)
        if candidate not in candidate_index:
            raise IntegrityError("Dev37 refit record has no source-derived candidate statistics.")
        modeled = (
            candidate
            if receipt.candidate_kind == "feature_count"
            else receipt.modeled_feature_count
        )
        fit = checkpoint_multinomial_refit_common_support(
            training[draw, candidate_index[candidate]],
            modeled_features=modeled,
            per_feature_pseudocount=0.5,
        )
        probabilities = fit.expanded_probabilities
        counts = validation[draw]
        total = int(round(float(counts.sum())))
        if total <= 0:
            raise IntegrityError("Dev37 replay validation denominator must be positive.")
        nll = float(-np.sum(counts * np.log(probabilities)))
        rows.append(
            {
                "candidate_kind": receipt.candidate_kind,
                "draw_id": draw,
                "candidate_value": candidate,
                "validation_total_count": total,
                "validation_nll_sum": nll,
                "validation_nll_per_count": nll / total,
                "final_state_hash": hashlib.sha256(
                    np.asarray(probabilities, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows, columns=REPLAY_COLUMNS)


def verify_g00c_refit_replay_v4(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    receipt: G00CRefitReplayReceiptV4,
    store: VirtualCanonicalCountStore,
    plan: G00CSamplerPlanV4,
    trace: pd.DataFrame,
    ordered_training_rows: np.ndarray[Any, Any],
    validation_rows: np.ndarray[Any, Any],
    feature_indices: np.ndarray[Any, Any],
    *,
    access_callback: AccessCallback | None = None,
    additional_plan_trace: tuple[G00CSamplerPlanV4, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Re-derive sufficient statistics from G00B before replaying every frozen row."""

    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.selection_freeze_id != authority.selection_freeze_id
        or receipt.seed_schedule_id != authority.seed_schedule_id
        or receipt.source_binding_id != authority.source_plane.binding_id
    ):
        raise IntegrityError("Dev37 refit replay receipt is cross-wired.")
    schedule = _read_model(root, authority.seed_schedule, G00CRefitSeedScheduleV1)
    derived = derive_checkpoint_statistics_v4(
        store,
        plan,
        trace,
        ordered_training_rows,
        validation_rows,
        feature_indices,
        schedule,
        candidate_kind=receipt.candidate_kind,
        access_callback=access_callback,
    )
    training, validation, fit_hashes, validation_hash, thinning_hash = derived
    if additional_plan_trace is not None:
        added_plan, added_trace = additional_plan_trace
        added = derive_checkpoint_statistics_v4(
            store,
            added_plan,
            added_trace,
            ordered_training_rows,
            validation_rows,
            feature_indices,
            schedule,
            candidate_kind=receipt.candidate_kind,
            access_callback=access_callback,
        )
        (
            added_training,
            added_validation,
            added_hashes,
            added_validation_hash,
            added_thinning,
        ) = added
        if (
            not np.array_equal(validation, added_validation)
            or validation_hash != added_validation_hash
        ):
            raise IntegrityError("Dev37 extension uses another validation count vector.")
        training = np.concatenate((training, added_training), axis=1)
        fit_hashes = (*fit_hashes, *added_hashes)
        thinning_hash = hashlib.sha256(
            bytes.fromhex(thinning_hash) + bytes.fromhex(added_thinning)
        ).hexdigest()
    try:
        with np.load(_path(root, receipt.source_derived_statistics), allow_pickle=False) as payload:
            if set(payload.files) != {"candidate_values", "training_counts", "validation_counts"}:
                raise IntegrityError("Dev37 source-derived statistics have another schema.")
            candidates = tuple(np.asarray(payload["candidate_values"], dtype=np.int64).tolist())
            observed_training = np.asarray(payload["training_counts"], dtype=np.float64)
            observed_validation = np.asarray(payload["validation_counts"], dtype=np.float64)
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError("Dev37 source-derived statistics cannot be read.") from exc
    if (
        candidates != receipt.candidate_values
        or observed_training.shape != receipt.training_shape
        or observed_validation.shape != receipt.validation_shape
        or not np.array_equal(observed_training, training)
        or not np.array_equal(observed_validation, validation)
        or receipt.fit_row_hashes != fit_hashes
        or receipt.validation_row_hash != validation_hash
        or receipt.training_count_hash != _array_hash(training)
        or receipt.validation_count_vector_hash != _array_hash(validation)
        or receipt.thinning_trace_hash != thinning_hash
    ):
        raise IntegrityError("Dev37 sufficient statistics differ from source-derived values.")
    records = pd.read_parquet(_path(root, receipt.refit_records))
    if tuple(records.columns) != REFIT_V3_COLUMNS or set(records["draw_id"].astype(int)) != set(
        range(59)
    ):
        raise IntegrityError("Dev37 refit records have another row surface.")
    recomputed = recompute_refit_rows_v4(records, receipt, training, validation)
    observed = pd.read_parquet(_path(root, receipt.replayed_rows))
    if tuple(observed.columns) != REPLAY_COLUMNS or not observed.equals(recomputed):
        raise IntegrityError("Dev37 replay rows differ from checkpoint-conditioned refits.")
    indexed = records.set_index(["draw_id", "candidate_value"])
    for row in recomputed.itertuples(index=False):
        source = indexed.loc[(row.draw_id, row.candidate_value)]
        if (
            int(source["validation_total_count"]) != row.validation_total_count
            or float(source["validation_nll_sum"]) != row.validation_nll_sum
            or float(source["validation_nll_per_count"]) != row.validation_nll_per_count
            or str(source["final_state_hash"]) != row.final_state_hash
        ):
            raise IntegrityError("Dev37 refit table differs from source-derived replay.")
    return recomputed
