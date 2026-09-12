"""Dev33 fail-closed G00C verification.

This module is additive.  The Dev31/Dev32 verifier in :mod:`g00c` retains its
accepted semantics; new G00C executions must use the versioned contracts here.
"""

from __future__ import annotations

import hashlib
import json
import resource
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ..canonical import canonical_json_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    FoldNativeCompactViewContractV3,
    G00CCompactVerificationReceiptV2,
    G00CDecisionReceiptV2,
    G00CExecutionBundleV2,
    RefitReplayPolicyV1,
)
from ..errors import IntegrityError
from .g00c import _verify_sampler
from .virtual import VirtualCanonicalCountStore

REFIT_COLUMNS = (
    "candidate_kind",
    "draw_id",
    "seed",
    "candidate_value",
    "fit_row_hash",
    "validation_row_hash",
    "count_thinning_hash",
    "model_config_hash",
    "initial_state_hash",
    "final_state_hash",
    "validation_total_count",
    "validation_nll_sum",
    "validation_nll_per_count",
    "fit_status",
)


@dataclass(frozen=True)
class RefitReplayObservation:
    """Values independently recomputed from one bound refit state."""

    final_state_hash: str
    validation_total_count: int
    validation_nll_sum: float
    validation_nll_per_count: float
    fit_status: str = "pass"


class RefitReplayExecutor(Protocol):
    """Hash-identifiable implementation that reruns one frozen refit."""

    implementation_sha256: str

    def __call__(self, kind: str, record: Mapping[str, Any]) -> RefitReplayObservation: ...


def _path(root: Path, artifact: ArtifactRef) -> Path:
    path = root / artifact.relative_uri
    if not path.is_file() or sha256_file(path) != artifact.sha256:
        raise IntegrityError(f"G00C artifact failed verification: {artifact.relative_uri}.")
    return path


def _row_set_hash(values: np.ndarray[Any, Any]) -> str:
    ordered = np.sort(np.asarray(values, dtype="<i8"), kind="stable")
    return hashlib.sha256(ordered.tobytes(order="C")).hexdigest()


def _ordered_row_hash(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  CREDO's supported execution platform is Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _verify_row_roles(
    root: Path,
    store: VirtualCanonicalCountStore,
    contract: FoldNativeCompactViewContractV3,
    bundle: G00CExecutionBundleV2,
) -> dict[str, np.ndarray[Any, Any]]:
    table = pd.read_parquet(_path(root, bundle.row_roles))
    expected_columns = (
        "row_id",
        "role",
        "donor_id",
        "checkpoint",
        "in_compact_payload",
    )
    if tuple(table.columns) != expected_columns or table["row_id"].duplicated().any():
        raise IntegrityError("G00C row-role artifact has an invalid schema or duplicate rows.")
    locator_ids, source_indices, _ = store._locator()
    observed_ids = table["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(observed_ids), locator_ids):
        raise IntegrityError("G00C row roles do not cover the exact G00B eligible universe.")
    positions = np.searchsorted(locator_ids, observed_ids)
    source_ids = source_indices[positions]
    expected_donors = np.asarray(
        [store.manifest.sources[int(index)].donor_id for index in source_ids]
    )
    expected_checkpoints = np.asarray(
        [store.manifest.sources[int(index)].checkpoint for index in source_ids]
    )
    expected_times = np.asarray(
        [store.manifest.sources[int(index)].physical_time_hours for index in source_ids]
    )
    if not np.array_equal(table["donor_id"].astype(str).to_numpy(), expected_donors):
        raise IntegrityError("G00C row-role donor identities differ from G00B.")
    if not np.array_equal(table["checkpoint"].astype(str).to_numpy(), expected_checkpoints):
        raise IntegrityError("G00C row-role checkpoints differ from G00B.")
    roles = table["role"].astype(str).to_numpy()
    donors = table["donor_id"].astype(str).to_numpy()
    training = np.isin(donors, contract.training_donor_ids)
    heldout = donors == contract.heldout_donor_id
    valid = (
        (np.isin(roles, ("training_fit", "training_validation")) & training)
        | ((roles == "heldout_source_query") & heldout & (expected_times == 0))
        | ((roles == "protected_heldout_stimulated") & heldout & (expected_times > 0))
    )
    if not np.all(valid):
        raise IntegrityError("G00C row roles violate donor/checkpoint permissions.")
    compact = table["in_compact_payload"].to_numpy(dtype=bool)
    if np.any(compact[roles == "protected_heldout_stimulated"]):
        raise IntegrityError("Protected held-out stimulated rows entered the compact payload.")
    if _row_set_hash(observed_ids[compact]) != bundle.compact_row_set_sha256:
        raise IntegrityError("G00C compact row set differs from its row-role audit.")
    for record in contract.row_roles:
        selected = observed_ids[roles == record.role]
        if len(selected) != record.rows or _row_set_hash(selected) != record.row_ids_hash:
            raise IntegrityError(f"G00C row-role receipt differs for {record.role}.")
    role_rows = {role: observed_ids[roles == role] for role in set(roles)}
    role_rows["__compact__"] = observed_ids[compact]
    if contract.feature_selection.fit_rows_hash != _row_set_hash(role_rows["training_fit"]):
        raise IntegrityError("G00C feature-selection fit rows differ from training_fit.")
    if contract.feature_selection.validation_rows_hash != _row_set_hash(
        role_rows["training_validation"]
    ):
        raise IntegrityError(
            "G00C feature-selection validation rows differ from training_validation."
        )
    return role_rows


def _load_refits(
    root: Path,
    artifact: ArtifactRef,
    *,
    kind: str,
    candidates: tuple[int, ...],
    validation_row_hash: str,
) -> pd.DataFrame:
    records = pd.read_parquet(_path(root, artifact))
    if tuple(records.columns) != REFIT_COLUMNS:
        raise IntegrityError("G00C refit provenance has an invalid schema.")
    if records.duplicated(["draw_id", "candidate_value"]).any():
        raise IntegrityError("G00C refit provenance contains duplicate draw/candidate pairs.")
    pivot = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_nll_per_count"
    )
    if (
        set(records["candidate_kind"].astype(str)) != {kind}
        or tuple(pivot.columns.astype(int)) != candidates
        or tuple(pivot.index.astype(int)) != tuple(range(59))
        or pivot.isna().any().any()
    ):
        raise IntegrityError("G00C refit provenance is incomplete or uses a different grid.")
    hashes = (
        "fit_row_hash",
        "validation_row_hash",
        "count_thinning_hash",
        "model_config_hash",
        "initial_state_hash",
        "final_state_hash",
    )
    if any(not records[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all() for field in hashes):
        raise IntegrityError("G00C refit provenance contains an invalid content hash.")
    if set(records["validation_row_hash"].astype(str)) != {validation_row_hash}:
        raise IntegrityError("G00C refits bind a different validation-row set.")
    if (
        (records["fit_status"].astype(str) != "pass").any()
        or (records["validation_total_count"].astype(np.int64) <= 0).any()
        or not np.isfinite(records["validation_nll_sum"].astype(float)).all()
        or not np.isfinite(records["validation_nll_per_count"].astype(float)).all()
    ):
        raise IntegrityError("G00C refit provenance contains an invalid fit or NLL value.")
    reconstructed = records["validation_nll_sum"].astype(float) / records[
        "validation_total_count"
    ].astype(float)
    if not np.allclose(
        reconstructed,
        records["validation_nll_per_count"].astype(float),
        atol=1e-15,
        rtol=0,
    ):
        raise IntegrityError("G00C validation NLL/count is not derived from its total and sum.")
    per_draw_seeds = records.groupby("draw_id", sort=True)["seed"].nunique()
    if not (per_draw_seeds == 1).all():
        raise IntegrityError("G00C paired refits do not share one seed per draw.")
    return records


def _replay_required_refits(
    records: pd.DataFrame,
    *,
    kind: str,
    selected: int,
    reference: int,
    policy: RefitReplayPolicyV1,
    replay_refit: RefitReplayExecutor | None,
) -> None:
    if replay_refit is None:
        raise IntegrityError("G00C refit replay executor is required for Dev33 evidence.")
    if replay_refit.implementation_sha256 != policy.implementation_sha256:
        raise IntegrityError("G00C refit replay implementation differs from its contract.")
    required = (
        (records["candidate_value"].astype(int) == selected)
        | (records["candidate_value"].astype(int) == reference)
        | records["draw_id"].astype(int).isin(policy.preregistered_audit_draw_ids)
    )
    for row in (
        records.loc[required]
        .sort_values(["draw_id", "candidate_value"], kind="stable")
        .to_dict(orient="records")
    ):
        observed = replay_refit(kind, row)
        if (
            observed.fit_status != "pass"
            or observed.final_state_hash != str(row["final_state_hash"])
            or observed.validation_total_count != int(row["validation_total_count"])
            or observed.validation_nll_sum != float(row["validation_nll_sum"])
            or observed.validation_nll_per_count != float(row["validation_nll_per_count"])
        ):
            raise IntegrityError("G00C executable refit replay differs from recorded evidence.")


def sample_size_decision_v2(
    *,
    grid_stage: str,
    candidates: tuple[int, ...],
    p95: np.ndarray[Any, Any],
    epsilon: float,
) -> tuple[str, int | None]:
    """Return the frozen Dev33 base/extension decision without a self-reference pass."""

    eligible = [
        count
        for count, value in zip(candidates, np.asarray(p95, dtype=float), strict=True)
        if value <= epsilon
    ]
    if grid_stage == "base":
        submaximum = [count for count in eligible if count < 1_000_000]
        return ("selected", min(submaximum)) if submaximum else ("extension_required", None)
    if grid_stage == "extension":
        if not eligible:
            raise IntegrityError("G00C extension grid has no qualifying sample size.")
        return "selected", min(eligible)
    raise IntegrityError("G00C sample-size grid stage is invalid.")


def _verify_feature_selection(
    root: Path,
    contract: FoldNativeCompactViewContractV3,
    bundle: G00CExecutionBundleV2,
    *,
    store_feature_width: int,
    replay_refit: RefitReplayExecutor | None,
) -> tuple[list[str], np.ndarray[Any, Any]]:
    result = bundle.feature_selection
    protocol = contract.feature_selection
    if (
        result.fit_rows_hash != protocol.fit_rows_hash
        or result.validation_rows_hash != protocol.validation_rows_hash
    ):
        raise IntegrityError("G00C feature evidence binds different row roles.")
    records = _load_refits(
        root,
        result.refit_records,
        kind="feature_count",
        candidates=protocol.candidate_feature_counts,
        validation_row_hash=protocol.validation_rows_hash,
    )
    if set(records["fit_row_hash"].astype(str)) != {protocol.fit_rows_hash}:
        raise IntegrityError("G00C feature refits bind a different fit-row set.")
    pivot = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_nll_per_count"
    )
    reference = protocol.maximum_feature_count
    p95 = np.quantile(np.abs(pivot.to_numpy() - pivot[reference].to_numpy()[:, None]), 0.95, axis=0)
    means = pivot.mean(axis=0).to_numpy()
    curve = pd.read_parquet(_path(root, result.curve))
    if (
        tuple(curve.columns)
        != (
            "feature_count",
            "mean_validation_nll",
            "p95_absolute_difference_to_4096",
        )
        or tuple(curve["feature_count"].astype(int)) != protocol.candidate_feature_counts
    ):
        raise IntegrityError("G00C feature curve has an invalid schema or grid.")
    if not np.allclose(curve["mean_validation_nll"], means, atol=1e-12, rtol=0) or not np.allclose(
        curve["p95_absolute_difference_to_4096"], p95, atol=1e-12, rtol=0
    ):
        raise IntegrityError("G00C feature curve is not derived from full refit records.")
    eligible = [
        count
        for count, value in zip(protocol.candidate_feature_counts, p95, strict=True)
        if value <= protocol.minimum_improvement_margin
    ]
    if not eligible or result.selected_feature_count != min(eligible):
        raise IntegrityError("G00C selected feature prefix violates the frozen rule.")
    _replay_required_refits(
        records,
        kind="feature_count",
        selected=result.selected_feature_count,
        reference=reference,
        policy=protocol.refit_replay,
        replay_refit=replay_refit,
    )
    ordered = pd.read_parquet(_path(root, result.ordered_features))
    expected_columns = (
        "rank",
        "canonical_index",
        "feature_id",
        "in_primary_metric",
        "is_sidecar",
    )
    if tuple(ordered.columns) != expected_columns or ordered["feature_id"].duplicated().any():
        raise IntegrityError("G00C ordered-feature table has an invalid schema.")
    primary = ordered.loc[~ordered["is_sidecar"].astype(bool)]
    sidecar = ordered.loc[ordered["is_sidecar"].astype(bool)]
    if (
        len(primary) != 4096
        or not np.array_equal(primary["rank"].to_numpy(dtype=int), np.arange(1, 4097))
        or primary["canonical_index"].duplicated().any()
        or (primary["canonical_index"] < 0).any()
        or (primary["canonical_index"] >= store_feature_width).any()
        or not primary["in_primary_metric"].astype(bool).all()
        or len(sidecar) != 1
        or str(sidecar.iloc[0]["feature_id"]) != "CUSTOM001_PuroR"
        or bool(sidecar.iloc[0]["in_primary_metric"])
    ):
        raise IntegrityError("G00C ordered features violate the primary/sidecar rule.")
    selected_features = primary.iloc[: result.selected_feature_count]
    feature_order_hash = hashlib.sha256(
        canonical_json_bytes(selected_features["feature_id"].astype(str).tolist())
    ).hexdigest()
    if feature_order_hash != bundle.compact_payload_feature_order_hash:
        raise IntegrityError("G00C compact feature order differs from selection evidence.")
    if bundle.compact_payload_features != result.selected_feature_count:
        raise IntegrityError("G00C compact width differs from the selected feature prefix.")
    return (
        selected_features["feature_id"].astype(str).tolist(),
        selected_features["canonical_index"].to_numpy(dtype=np.int64),
    )


def verify_sample_size_selection_v2(
    root: Path,
    contract: FoldNativeCompactViewContractV3,
    bundle: G00CExecutionBundleV2,
    role_rows: dict[str, np.ndarray[Any, Any]],
    *,
    replay_refit: RefitReplayExecutor | None,
) -> np.ndarray[Any, Any] | None:
    """Verify the base/extension decision; ``None`` is the required base-grid stop."""

    result = bundle.sample_size_selection
    protocol = contract.sample_size_selection
    records = _load_refits(
        root,
        result.refit_records,
        kind="training_cells",
        candidates=protocol.candidate_cells,
        validation_row_hash=contract.feature_selection.validation_rows_hash,
    )
    order_table = pd.read_parquet(_path(root, result.training_scale_row_order))
    if tuple(order_table.columns) != ("rank", "row_id"):
        raise IntegrityError("G00C training-scale row order has an invalid schema.")
    ordered_rows = order_table["row_id"].to_numpy(dtype=np.int64)
    ranks = order_table["rank"].to_numpy(dtype=np.int64)
    if (
        len(ordered_rows) != protocol.candidate_cells[-1]
        or not np.array_equal(ranks, np.arange(1, len(ordered_rows) + 1))
        or len(np.unique(ordered_rows)) != len(ordered_rows)
        or not np.isin(ordered_rows, role_rows["training_fit"]).all()
    ):
        raise IntegrityError("G00C training scales are not nested training_fit prefixes.")
    prefix_hashes = {
        count: _ordered_row_hash(ordered_rows[:count]) for count in protocol.candidate_cells
    }
    observed_fit_hashes = records.groupby("candidate_value", sort=True)["fit_row_hash"].agg(set)
    if any(
        observed_fit_hashes.loc[count] != {prefix_hashes[count]}
        for count in protocol.candidate_cells
    ):
        raise IntegrityError("G00C sample-size refits bind different nested fit prefixes.")
    pivot = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_nll_per_count"
    )
    reference = protocol.candidate_cells[-1]
    p95 = np.quantile(np.abs(pivot.to_numpy() - pivot[reference].to_numpy()[:, None]), 0.95, axis=0)
    means = pivot.mean(axis=0).to_numpy()
    curve = pd.read_parquet(_path(root, result.curve))
    if (
        tuple(curve.columns)
        != (
            "training_cells",
            "mean_validation_nll",
            "p95_absolute_difference_to_nmax",
        )
        or tuple(curve["training_cells"].astype(int)) != protocol.candidate_cells
    ):
        raise IntegrityError("G00C sample-size curve has an invalid schema or grid.")
    if not np.allclose(curve["mean_validation_nll"], means, atol=1e-12, rtol=0) or not np.allclose(
        curve["p95_absolute_difference_to_nmax"], p95, atol=1e-12, rtol=0
    ):
        raise IntegrityError("G00C sample-size curve is not derived from full refit records.")
    if protocol.grid_stage == "base":
        expected_status, expected_selection = sample_size_decision_v2(
            grid_stage="base",
            candidates=protocol.candidate_cells,
            p95=p95,
            epsilon=protocol.equivalence_epsilon,
        )
    else:
        parent = json.loads(
            _path(root, protocol.base_grid_extension_required_receipt).read_text()  # type: ignore[arg-type]
        )
        if parent.get("selection_status") != "extension_required" or tuple(
            parent.get("candidate_cells", ())
        ) != (50_000, 100_000, 250_000, 500_000, 1_000_000):
            raise IntegrityError("G00C extension contract lacks its exact base-grid stop receipt.")
        expected_status, expected_selection = sample_size_decision_v2(
            grid_stage="extension",
            candidates=protocol.candidate_cells,
            p95=p95,
            epsilon=protocol.equivalence_epsilon,
        )
    if (
        result.selection_status != expected_status
        or result.selected_training_cells != expected_selection
    ):
        raise IntegrityError("G00C sample-size result violates base/extension semantics.")
    replay_selection = expected_selection if expected_selection is not None else reference
    _replay_required_refits(
        records,
        kind="training_cells",
        selected=replay_selection,
        reference=reference,
        policy=protocol.refit_replay,
        replay_refit=replay_refit,
    )
    if expected_selection is None:
        return None
    assert result.selected_training_rows is not None
    selected_table = pd.read_parquet(_path(root, result.selected_training_rows))
    if tuple(selected_table.columns) != ("row_id",):
        raise IntegrityError("G00C selected-training-row artifact has an invalid schema.")
    selected_rows = selected_table["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(selected_rows, ordered_rows[:expected_selection]):
        raise IntegrityError("G00C selected rows differ from the frozen nested prefix.")
    return selected_rows


def _expected_physical_order(
    store: VirtualCanonicalCountStore, row_ids: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], int]:
    locator_ids, source_indices, source_rows = store._locator()
    positions = np.searchsorted(locator_ids, row_ids)
    if np.any(positions >= len(locator_ids)) or not np.array_equal(locator_ids[positions], row_ids):
        raise IntegrityError("G00C physical-order rows are absent from G00B.")
    locator_path = store.path / store.manifest.row_locator.relative_uri
    position_order = np.argsort(positions, kind="stable")
    sorted_positions = positions[position_order]
    inverse_position_order = np.argsort(position_order, kind="stable")
    with h5py.File(locator_path, "r") as handle:
        guide_codes = np.asarray(handle["guide_codes_sorted"][sorted_positions], dtype=np.int64)[
            inverse_position_order
        ]
        target_codes = np.asarray(handle["target_codes_sorted"][sorted_positions], dtype=np.int64)[
            inverse_position_order
        ]
        guide_ids = handle["guide_ids"].asstr()[:]
        target_ids = handle["target_ids"].asstr()[:]
    selected_sources = source_indices[positions]
    donors = np.asarray([store.manifest.sources[int(index)].donor_id for index in selected_sources])
    times = np.asarray(
        [store.manifest.sources[int(index)].physical_time_hours for index in selected_sources]
    )
    checkpoints = np.asarray(
        [store.manifest.sources[int(index)].checkpoint for index in selected_sources]
    )
    targets = target_ids[target_codes]
    guides = guide_ids[guide_codes]
    rows = source_rows[positions]
    order = np.lexsort((row_ids, rows, guides, targets, checkpoints, times, donors))
    ordered_ids = row_ids[order]
    group_columns = np.column_stack(
        (donors[order], checkpoints[order], targets[order], guides[order])
    )
    transitions = np.ones(len(ordered_ids), dtype=bool)
    if len(ordered_ids) > 1:
        transitions[1:] = np.any(group_columns[1:] != group_columns[:-1], axis=1)
    block_count = int(transitions.sum())
    unique_groups = len({tuple(values) for values in group_columns.tolist()})
    if block_count != unique_groups:
        raise IntegrityError("G00C physical donor/checkpoint/target/guide runs are not contiguous.")
    return ordered_ids, block_count


def _verify_compact_payload_streaming(
    root: Path,
    store: VirtualCanonicalCountStore,
    contract: FoldNativeCompactViewContractV3,
    bundle: G00CExecutionBundleV2,
    role_rows: dict[str, np.ndarray[Any, Any]],
    selected_training_rows: np.ndarray[Any, Any],
    selected_feature_ids: list[str],
    selected_feature_indices: np.ndarray[Any, Any],
) -> None:
    payload_path = _path(root, bundle.compact_payload)
    expected_set = np.concatenate(
        (
            selected_training_rows,
            role_rows["training_validation"],
            role_rows["heldout_source_query"],
        )
    )
    if len(np.unique(expected_set)) != len(expected_set) or set(expected_set) != set(
        role_rows["__compact__"]
    ):
        raise IntegrityError("G00C compact row roles differ from selected permitted rows.")
    expected_order, contiguous_blocks = _expected_physical_order(store, expected_set)
    expected_data_dtype = np.dtype("uint16" if contract.count_dtype == "uint16" else "int32")
    expected_index_dtype = np.dtype("uint16" if contract.index_dtype == "uint16" else "int32")
    block_rows = contract.compact_verification_block_rows
    total_nnz = 0
    total_count_sum = 0
    maximum_loaded_nnz = 0
    with h5py.File(payload_path, "r") as handle:
        if set(handle) != {"row_ids", "feature_ids", "indptr", "indices", "data"}:
            raise IntegrityError("G00C compact payload has an invalid dataset schema.")
        row_ids = np.asarray(handle["row_ids"][:], dtype=np.int64)
        feature_ids = handle["feature_ids"].asstr()[:].tolist()
        indptr = handle["indptr"]
        indices = handle["indices"]
        data = handle["data"]
        if (
            data.dtype != expected_data_dtype
            or indices.dtype != expected_index_dtype
            or indptr.dtype != np.dtype("int64")
        ):
            raise IntegrityError("G00C compact payload dtypes differ from the frozen schema.")
        if (
            len(row_ids) != bundle.compact_payload_rows
            or len(row_ids) != len(expected_order)
            or not np.array_equal(row_ids, expected_order)
            or _row_set_hash(row_ids) != bundle.compact_row_set_sha256
            or _ordered_row_hash(row_ids) != bundle.compact_ordered_row_ids_sha256
        ):
            raise IntegrityError("G00C compact payload violates its exact physical row order.")
        if (
            feature_ids != selected_feature_ids
            or len(feature_ids) != bundle.compact_payload_features
        ):
            raise IntegrityError("G00C compact payload features differ from its selected prefix.")
        if (
            len(indptr) != len(row_ids) + 1
            or int(indptr[0]) != 0
            or int(indptr[-1]) != len(indices)
        ):
            raise IntegrityError("G00C compact payload has invalid global CSR offsets.")
        if len(indices) != len(data):
            raise IntegrityError("G00C compact payload index/value lengths differ.")
        for start in range(0, len(row_ids), block_rows):
            stop = min(start + block_rows, len(row_ids))
            offsets = np.asarray(indptr[start : stop + 1], dtype=np.int64)
            if np.any(np.diff(offsets) < 0):
                raise IntegrityError("G00C compact payload has decreasing CSR offsets.")
            first, last = int(offsets[0]), int(offsets[-1])
            block_indices = np.asarray(indices[first:last])
            block_data = np.asarray(data[first:last])
            if (
                np.any(block_indices < 0)
                or np.any(block_indices >= len(feature_ids))
                or (len(block_data) and not np.isfinite(block_data).all())
                or (len(block_data) and np.any(block_data < 0))
                or (len(block_data) and int(block_data.max()) > contract.maximum_observed_count)
            ):
                raise IntegrityError("G00C compact payload fails block CSR/count integrity.")
            local = sparse.csr_matrix(
                (block_data, block_indices.astype(np.int64), offsets - first),
                shape=(stop - start, len(feature_ids)),
            )
            source = store.rows(row_ids[start:stop]).matrix[:, selected_feature_indices]
            if source.shape != local.shape or (source != local).nnz:
                raise IntegrityError("G00C compact counts differ from the virtual source plane.")
            loaded_nnz = last - first
            maximum_loaded_nnz = max(maximum_loaded_nnz, loaded_nnz)
            total_nnz += loaded_nnz
            total_count_sum += int(np.sum(block_data.astype(np.uint64), dtype=np.uint64))
            del source, local, block_data, block_indices, offsets
    if bundle.compact_payload_counts_sha256 != bundle.compact_payload.sha256:
        raise IntegrityError("G00C compact count hash differs from its payload artifact.")
    receipt = G00CCompactVerificationReceiptV2.model_validate_json(
        _path(root, bundle.compact_verification_receipt).read_text()
    )
    current_rss = _rss_bytes()
    if (
        receipt.fold_view_id != contract.fold_view_id
        or receipt.compact_payload_sha256 != bundle.compact_payload.sha256
        or receipt.verifier_implementation_sha256 != contract.compact_verifier_implementation_sha256
        or receipt.verifier_block_rows != block_rows
        or receipt.maximum_loaded_nonzeros != maximum_loaded_nnz
        or receipt.total_rows_checked != len(expected_order)
        or receipt.total_nonzeros_checked != total_nnz
        or receipt.total_count_sum != total_count_sum
        or receipt.compact_row_set_sha256 != bundle.compact_row_set_sha256
        or receipt.compact_ordered_row_ids_sha256 != bundle.compact_ordered_row_ids_sha256
        or receipt.contiguous_physical_blocks != contiguous_blocks
        or receipt.maximum_open_source_handles > contract.maximum_source_handles
        or receipt.peak_process_rss_bytes > contract.maximum_verifier_rss_bytes
        or current_rss > contract.maximum_verifier_rss_bytes
    ):
        raise IntegrityError("G00C bounded streaming verification receipt differs from execution.")


def validate_g00c_execution_v2(
    root: Path,
    store: VirtualCanonicalCountStore,
    contract: FoldNativeCompactViewContractV3,
    bundle: G00CExecutionBundleV2,
    receipt: G00CDecisionReceiptV2,
    *,
    replay_refit: RefitReplayExecutor | None,
) -> None:
    """Recompute every Dev33 selection, replay, order, streaming, and publication gate."""

    if (
        bundle.fold_view_id != contract.fold_view_id
        or receipt.fold_view_id != contract.fold_view_id
        or receipt.execution_bundle_id != bundle.bundle_id
        or bundle.row_roles != contract.row_role_audit
        or bundle.maximum_observed_count != contract.maximum_observed_count
        or bundle.count_dtype != contract.count_dtype
        or bundle.index_dtype != contract.index_dtype
    ):
        raise IntegrityError("G00C Dev33 contract, bundle, and decision are cross-wired.")
    role_rows = _verify_row_roles(root, store, contract, bundle)
    feature_ids, feature_indices = _verify_feature_selection(
        root,
        contract,
        bundle,
        store_feature_width=store.manifest.features,
        replay_refit=replay_refit,
    )
    selected_rows = verify_sample_size_selection_v2(
        root,
        contract,
        bundle,
        role_rows,
        replay_refit=replay_refit,
    )
    if selected_rows is None:
        raise IntegrityError("G00C base grid requires a separately frozen two-million extension.")
    _verify_sampler(root, contract, bundle)  # type: ignore[arg-type]
    _verify_compact_payload_streaming(
        root,
        store,
        contract,
        bundle,
        role_rows,
        selected_rows,
        feature_ids,
        feature_indices,
    )
    publication = json.loads(_path(root, bundle.publication_manifest).read_text())
    reload_receipt = json.loads(_path(root, bundle.reload_receipt).read_text())
    if (
        publication.get("status") != "pass"
        or publication.get("fold_view_id") != contract.fold_view_id
    ):
        raise IntegrityError("G00C immutable-publication evidence failed.")
    if (
        reload_receipt.get("status") != "pass"
        or reload_receipt.get("compact_payload_sha256") != bundle.compact_payload.sha256
    ):
        raise IntegrityError("G00C full-reload evidence failed.")
    if not all(
        (
            receipt.row_roles_verified,
            receipt.feature_selection_verified,
            receipt.sample_size_selection_verified,
            receipt.refit_replay_verified,
            receipt.sampler_sequence_verified,
            receipt.physical_order_verified,
            receipt.streaming_verification_verified,
            receipt.compact_counts_verified,
            receipt.protected_rows_absent,
            receipt.immutable_publication_verified,
            receipt.full_reload_verified,
        )
    ):
        raise IntegrityError("G00C Dev33 decision flags disagree with verified evidence.")
