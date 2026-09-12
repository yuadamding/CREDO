"""Source-owned G00B verification, hierarchy derivation, and feature ranking for Dev37."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ..canonical import canonical_json_bytes, sha256_file
from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV3,
    G00CFeatureRankingReceiptV4,
    G00CHierarchyDerivationReceiptV4,
    G00CSourcePlaneBindingV4,
    VirtualCanonicalCountStoreManifestV2,
)
from ..errors import IntegrityError
from .g00c_v3 import _path
from .virtual import VirtualCanonicalCountStore

RANKING_COLUMNS = (
    "rank",
    "canonical_index",
    "feature_id",
    "poisson_deviance",
    "total_umi",
    "detection_count",
)
HIERARCHY_COLUMNS = ("row_id", "source_index", "target_code", "guide_code", "is_control")
AccessCallback = Callable[[str, np.ndarray[Any, Any]], None]


def _hash_array(values: np.ndarray[Any, Any], dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _hash_ordered_strings(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _canonical_feature_records(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text())
        records = payload["features"]
        if (
            int(payload["feature_count"]) != len(records)
            or payload["ordered_hash"] != hashlib.sha256(canonical_json_bytes(records)).hexdigest()
        ):
            raise IntegrityError("Dev37 canonical feature index hash differs from its records.")
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError("Dev37 canonical feature index is malformed.") from exc
    expected_keys = {"feature_id", "namespace", "namespace_version"}
    if any(set(record) != expected_keys for record in records):
        raise IntegrityError("Dev37 canonical feature records have another schema.")
    feature_ids = [str(record["feature_id"]) for record in records]
    if len(feature_ids) != len(set(feature_ids)):
        raise IntegrityError("Dev37 canonical feature IDs are not unique.")
    return [{key: str(record[key]) for key in expected_keys} for record in records]


def _source_feature_ids(path: Path) -> np.ndarray[Any, Any]:
    try:
        with h5py.File(path, "r") as handle:
            if "var" not in handle or "_index" not in handle["var"]:
                raise IntegrityError("Dev37 source lacks the H5AD var/_index feature authority.")
            return np.asarray(handle["var/_index"].asstr()[:], dtype=object)
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"Dev37 source feature index cannot be read: {path}.") from exc


def open_verified_source_plane_v4(
    authority_root: Path,
    binding: G00CSourcePlaneBindingV4,
    *,
    source_plane_root: Path,
    source_files_root: Path,
) -> tuple[VirtualCanonicalCountStore, tuple[str, ...]]:
    """Construct the accepted store internally and verify every bound source byte."""

    store_root = source_plane_root / binding.virtual_store_relative_uri
    manifest_path = store_root / "manifest.json"
    if (
        store_root.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != binding.accepted_g00b_manifest_sha256
        or manifest_path.read_bytes()
        != _path(authority_root, binding.accepted_g00b_parent).read_bytes()
    ):
        raise IntegrityError("Dev37 external G00B manifest differs from its accepted parent.")
    store = VirtualCanonicalCountStore(store_root, source_root=source_files_root)
    manifest = store.verify(full=True)
    if not isinstance(manifest, VirtualCanonicalCountStoreManifestV2):
        raise IntegrityError("Dev37 requires the accepted V2 virtual store.")
    source_bindings = tuple(
        (item.source_id, item.checkpoint, item.source_file_sha256) for item in binding.source_files
    )
    observed_sources = tuple(
        (item.source_id, item.checkpoint, item.source_file_sha256) for item in manifest.sources
    )
    if (
        manifest.virtual_store_id != binding.virtual_store_id
        or manifest.source_authority_id != binding.source_authority_id
        or manifest.canonical_feature_index_hash != binding.canonical_feature_index_hash
        or manifest.row_locator.sha256 != binding.row_locator_sha256
        or manifest.feature_permutations.sha256 != binding.feature_permutations_sha256
        or manifest.guide_target_crosswalk.sha256 != binding.guide_target_crosswalk_sha256
        or manifest.features != binding.feature_count
        or source_bindings != observed_sources
    ):
        raise IntegrityError("Dev37 G00B identity differs from the frozen source descriptor.")
    feature_path = _path(authority_root, binding.canonical_feature_index)
    feature_records = _canonical_feature_records(feature_path)
    if (
        len(feature_records) != binding.feature_count
        or hashlib.sha256(canonical_json_bytes(feature_records)).hexdigest()
        != binding.canonical_feature_index_hash
    ):
        raise IntegrityError("Dev37 feature-index bytes differ from the G00B identity.")
    canonical_ids = tuple(item["feature_id"] for item in feature_records)
    permutations = store._feature_permutations()
    for source, permutation in zip(manifest.sources, permutations, strict=True):
        source_ids = _source_feature_ids(source_files_root / source.relative_uri)
        if len(source_ids) != source.features or tuple(source_ids[permutation]) != canonical_ids:
            raise IntegrityError(
                f"Dev37 canonical feature mapping differs for source {source.source_id}."
            )
    puro_matches = [
        index for index, value in enumerate(canonical_ids) if value == "CUSTOM001_PuroR"
    ]
    if puro_matches != [binding.puro_r_canonical_index]:
        raise IntegrityError("Dev37 PuroR mapping differs from the frozen canonical index.")
    return store, canonical_ids


def open_verified_g00b_v4(
    authority_root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    *,
    source_plane_root: Path,
    source_files_root: Path,
) -> tuple[VirtualCanonicalCountStore, tuple[str, ...]]:
    """Construct the G00B store internally from one frozen Dev37 authority."""

    return open_verified_source_plane_v4(
        authority_root,
        authority.source_plane,
        source_plane_root=source_plane_root,
        source_files_root=source_files_root,
    )


def derive_hierarchy_from_g00b_v4(
    store: VirtualCanonicalCountStore,
    training_row_ids: np.ndarray[Any, Any],
) -> pd.DataFrame:
    """Reconstruct source, target, guide, and control fields from locator/crosswalk bytes."""

    if not isinstance(store.manifest, VirtualCanonicalCountStoreManifestV2):
        raise IntegrityError("Dev37 hierarchy requires the accepted V2 virtual store.")
    requested = np.asarray(training_row_ids, dtype=np.int64)
    sorted_ids, locator_sources, _ = store._locator()
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], requested):
        raise IntegrityError("Dev37 hierarchy rows are absent from the accepted G00B locator.")
    locator_path = store.path / store.manifest.row_locator.relative_uri
    with h5py.File(locator_path, "r") as handle:
        guide_codes = np.asarray(handle["guide_codes_sorted"][positions], dtype=np.int32)
        target_codes = np.asarray(handle["target_codes_sorted"][positions], dtype=np.int32)
        guide_ids = handle["guide_ids"].asstr()[:]
    crosswalk = pd.read_parquet(store.path / store.manifest.guide_target_crosswalk.relative_uri)
    controls = crosswalk.set_index("guide_id")["is_control"].to_dict()
    if set(controls) != set(guide_ids.tolist()):
        raise IntegrityError("Dev37 crosswalk and locator guide catalogs differ.")
    is_control = np.asarray([bool(controls[str(value)]) for value in guide_ids], dtype=bool)[
        guide_codes
    ]
    return pd.DataFrame(
        {
            "row_id": requested,
            "source_index": locator_sources[positions].astype(np.int16),
            "target_code": target_codes,
            "guide_code": guide_codes,
            "is_control": is_control,
        },
        columns=HIERARCHY_COLUMNS,
    )


def verify_g00c_hierarchy_v4(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    store: VirtualCanonicalCountStore,
) -> pd.DataFrame:
    """Compare the published hierarchy with an independent G00B reconstruction."""

    roles = pd.read_parquet(_path(root, authority.row_role_freeze))
    if tuple(roles.columns) != ("row_id", "role"):
        raise IntegrityError("Dev37 row-role artifact has another schema.")
    training = roles.loc[roles["role"].astype(str) == "training_fit", "row_id"].to_numpy(
        dtype=np.int64
    )
    derived = derive_hierarchy_from_g00b_v4(store, training)
    observed = pd.read_parquet(_path(root, authority.sampler_row_hierarchy))
    receipt = G00CHierarchyDerivationReceiptV4.model_validate_json(
        _path(root, authority.hierarchy_derivation_receipt).read_text()
    )
    if tuple(observed.columns) != HIERARCHY_COLUMNS or not observed.equals(derived):
        raise IntegrityError("Dev37 hierarchy differs from the accepted G00B metadata.")
    hashes = (
        _hash_array(derived["row_id"].to_numpy(), "<i8"),
        _hash_array(derived["source_index"].to_numpy(), "<i2"),
        _hash_array(derived["target_code"].to_numpy(), "<i4"),
        _hash_array(derived["guide_code"].to_numpy(), "<i4"),
        _hash_array(derived["is_control"].to_numpy(), "|b1"),
    )
    if (
        receipt.source_binding_id != authority.source_plane.binding_id
        or receipt.hierarchy != authority.sampler_row_hierarchy
        or receipt.row_count != len(derived)
        or hashes
        != (
            receipt.ordered_row_ids_sha256,
            receipt.source_index_sha256,
            receipt.target_code_sha256,
            receipt.guide_code_sha256,
            receipt.is_control_sha256,
        )
    ):
        raise IntegrityError("Dev37 hierarchy derivation receipt differs from recomputation.")
    return derived


def _locate_source_indices(
    store: VirtualCanonicalCountStore, row_ids: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    sorted_ids, source_indices, _ = store._locator()
    positions = np.searchsorted(sorted_ids, row_ids)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], row_ids):
        raise IntegrityError("Dev37 ranking row is absent from G00B.")
    return source_indices[positions].astype(np.int64)


def recompute_feature_ranking_v4(
    store: VirtualCanonicalCountStore,
    canonical_feature_ids: tuple[str, ...],
    fit_row_ids: np.ndarray[Any, Any],
    *,
    puro_r_canonical_index: int,
    batch_size: int = 8192,
    access_callback: AccessCallback | None = None,
) -> pd.DataFrame:
    """Stream checkpoint-conditioned Poisson deviance without dense expected counts."""

    rows = np.asarray(fit_row_ids, dtype=np.int64)
    if not len(rows) or batch_size <= 0:
        raise ValueError("Dev37 feature ranking requires nonempty rows and positive batches.")
    if len(canonical_feature_ids) != store.manifest.features:
        raise IntegrityError("Dev37 feature IDs do not cover the canonical matrix width.")
    checkpoints = ("Rest", "Stim8hr", "Stim48hr")
    checkpoint_code = {value: index for index, value in enumerate(checkpoints)}
    source_codes = np.asarray(
        [checkpoint_code[source.checkpoint] for source in store.manifest.sources], dtype=np.int8
    )
    width = store.manifest.features
    totals = np.zeros((3, width), dtype=np.float64)
    xlogx = np.zeros((3, width), dtype=np.float64)
    xlog_library = np.zeros((3, width), dtype=np.float64)
    total_library = np.zeros(3, dtype=np.float64)
    detected = np.zeros(width, dtype=np.int64)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        if access_callback is not None:
            access_callback("feature_ranking_training_fit", batch_rows)
        matrix = store.rows(batch_rows).matrix.astype(np.float64).tocsr()
        library = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
        library -= np.asarray(matrix[:, puro_r_canonical_index].todense(), dtype=np.float64).ravel()
        if np.any(library <= 0):
            raise IntegrityError("Dev37 ranking encountered a zero primary-UMI library.")
        sources = _locate_source_indices(store, batch_rows)
        row_checkpoints = source_codes[sources]
        row_positions = np.repeat(np.arange(len(batch_rows)), np.diff(matrix.indptr))
        values = matrix.data
        columns = matrix.indices
        detected += np.bincount(columns, minlength=width)
        for code in range(3):
            selected_rows = row_checkpoints == code
            if not selected_rows.any():
                continue
            total_library[code] += library[selected_rows].sum()
            selected_values = selected_rows[row_positions]
            values_t = values[selected_values]
            columns_t = columns[selected_values]
            rows_t = row_positions[selected_values]
            totals[code] += np.bincount(columns_t, weights=values_t, minlength=width)
            xlogx[code] += np.bincount(
                columns_t, weights=values_t * np.log(values_t), minlength=width
            )
            xlog_library[code] += np.bincount(
                columns_t,
                weights=values_t * np.log(library[rows_t]),
                minlength=width,
            )
    if np.any(total_library <= 0):
        raise IntegrityError("Dev37 feature reference lacks one frozen checkpoint.")
    scores: np.ndarray[Any, Any] = np.zeros(width, dtype=np.float64)
    for code in range(3):
        positive = totals[code] > 0
        log_probability = np.zeros(width, dtype=np.float64)
        log_probability[positive] = np.log(totals[code, positive] / total_library[code])
        scores[positive] += 2.0 * (
            xlogx[code, positive]
            - xlog_library[code, positive]
            - totals[code, positive] * log_probability[positive]
        )
    scores = np.maximum(scores, 0.0)
    total = totals.sum(axis=0)
    candidates = np.flatnonzero(np.arange(width) != puro_r_canonical_index)
    ids = np.asarray(canonical_feature_ids, dtype=object)
    lexical = np.argsort(
        np.asarray([str(value).encode("utf-8") for value in ids[candidates]], dtype=object),
        kind="stable",
    )
    lexical_rank = np.empty(len(candidates), dtype=np.int64)
    lexical_rank[lexical] = np.arange(len(candidates), dtype=np.int64)
    ordering = np.lexsort(
        (
            lexical_rank,
            -detected[candidates],
            -total[candidates],
            -scores[candidates],
        )
    )
    chosen = candidates[ordering]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(chosen) + 1, dtype=np.int64),
            "canonical_index": chosen.astype(np.int64),
            "feature_id": ids[chosen].astype(str),
            "poisson_deviance": scores[chosen],
            "total_umi": total[chosen],
            "detection_count": detected[chosen],
        },
        columns=RANKING_COLUMNS,
    )


def verify_g00c_feature_ranking_v4(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    store: VirtualCanonicalCountStore,
    canonical_feature_ids: tuple[str, ...],
    receipt: G00CFeatureRankingReceiptV4,
    *,
    access_callback: AccessCallback | None = None,
) -> pd.DataFrame:
    """Recompute the complete ranking and exact ID/index mapping from source bytes."""

    reference = pd.read_parquet(_path(root, authority.feature_reference_rows))
    if tuple(reference.columns) != ("rank", "row_id"):
        raise IntegrityError("Dev37 feature-reference rows have another schema.")
    fit_rows = reference["row_id"].to_numpy(dtype=np.int64)
    ranking = recompute_feature_ranking_v4(
        store,
        canonical_feature_ids,
        fit_rows,
        puro_r_canonical_index=authority.source_plane.puro_r_canonical_index,
        access_callback=access_callback,
    )
    observed = pd.read_parquet(_path(root, receipt.complete_ranking))
    ordered_ids = tuple(ranking["feature_id"].astype(str))
    if (
        len(ranking) < receipt.selected_prefix_count
        or tuple(observed.columns) != RANKING_COLUMNS
        or not observed.equals(ranking)
    ):
        raise IntegrityError("Dev37 feature ranking differs from source-derived scores.")
    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.source_binding_id != authority.source_plane.binding_id
        or receipt.fit_rows_hash != authority.feature_reference_rows_hash
        or receipt.fit_row_count != len(fit_rows)
        or receipt.canonical_feature_index_hash
        != authority.source_plane.canonical_feature_index_hash
        or receipt.puro_r_canonical_index != authority.source_plane.puro_r_canonical_index
        or receipt.ordered_feature_hash != _hash_ordered_strings(ordered_ids)
        or receipt.selected_prefix_hash != _hash_ordered_strings(ordered_ids[:4096])
    ):
        raise IntegrityError("Dev37 feature-ranking receipt differs from recomputation.")
    return ranking
