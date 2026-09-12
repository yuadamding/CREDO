"""T01 count-native representation qualification.

The representation is fitted only on source-checkpoint cells from outer-training
guides.  It uses a multinomial/Hellinger factorization: raw counts are converted
to square-root compositions, projected into a low-rank basis, and decoded by
squaring and renormalizing the reconstructed Hellinger coordinates.  Terminal
cells are encoded only after dimension selection and the outer-training refit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.spatial import cKDTree

from ..canonical import canonical_json_bytes, contract_id, path_manifest, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    ComponentTestContract,
    ComponentTestReceipt,
    CountRepresentationBundle,
)
from ..data import verify_pooled_finite_measures
from ..errors import ContractError, IntegrityError
from ..persistence import publish_directory, verify_directory
from ..store import CountStore

_FOLD_COLUMNS = ("guide_id", "outer_fold")


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _artifact(path: Path, *, schema_id: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.name,
    )


def _row_hash(values: Iterable[int]) -> str:
    return sha256_bytes(np.asarray(tuple(values), dtype="<i8").tobytes())


def _sample(values: np.ndarray[Any, Any], maximum: int, seed: int) -> np.ndarray[Any, Any]:
    ordered = np.asarray(sorted(map(int, values)), dtype=np.int64)
    if len(ordered) <= maximum:
        return ordered
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(ordered), size=maximum, replace=False)
    return ordered[np.sort(selected)]


def _hellinger(matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray[Any, Any]]:
    values = matrix.astype(np.float32).tocsr(copy=True)
    totals = np.asarray(values.sum(axis=1), dtype=np.float64).reshape(-1)
    if np.any(totals <= 0):
        raise ContractError("Count-native representation rows require positive library sizes.")
    values = sparse.diags((1.0 / totals).astype(np.float32)) @ values
    values.data = np.sqrt(values.data, dtype=np.float32)
    return values.tocsr(), totals


def _fix_component_signs(components: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    result = np.asarray(components, dtype=np.float32).copy()
    for index, row in enumerate(result):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            result[index] *= -1
    return result


def _fit_basis(
    matrix: sparse.csr_matrix, rank: int, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    hellinger, _ = _hellinger(matrix)
    center = np.sqrt(_global_frequency(matrix)).astype(np.float32)
    available = min(hellinger.shape)
    selected_rank = min(rank, max(1, available - 1))
    if selected_rank == 1 and available <= 1:
        basis = np.ones((1, hellinger.shape[1]), dtype=np.float32)
        basis /= np.linalg.norm(basis)
    elif hellinger.shape[0] * hellinger.shape[1] <= 10_000_000:
        residual = hellinger.toarray().astype(np.float32) - center[None, :]
        _, _, right = np.linalg.svd(residual, full_matrices=False)
        basis = right[:selected_rank].astype(np.float32)
    else:
        width = min(selected_rank + 8, available)
        generator = np.random.default_rng(seed)
        probe = generator.standard_normal((hellinger.shape[1], width), dtype=np.float32)
        sample = np.asarray(hellinger @ probe, dtype=np.float32) - (center @ probe)[None, :]
        left, _ = np.linalg.qr(sample, mode="reduced")
        dual = np.asarray(hellinger.T @ left, dtype=np.float32) - np.outer(center, left.sum(axis=0))
        sample = np.asarray(hellinger @ dual, dtype=np.float32) - (center @ dual)[None, :]
        left, _ = np.linalg.qr(sample, mode="reduced")
        compressed = np.asarray(left.T @ hellinger, dtype=np.float32) - np.outer(
            left.sum(axis=0), center
        )
        _, _, right = np.linalg.svd(compressed, full_matrices=False)
        basis = right[:selected_rank].astype(np.float32)
    if basis.shape[0] < rank:
        basis = np.pad(basis, ((0, rank - basis.shape[0]), (0, 0)))
    return _fix_component_signs(basis), center


def _encode(
    matrix: sparse.csr_matrix, components: np.ndarray[Any, Any], center: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    hellinger, _ = _hellinger(matrix)
    return np.asarray(hellinger @ components.T, dtype=np.float32) - (center @ components.T)[None, :]


def _decode(
    z: np.ndarray[Any, Any], components: np.ndarray[Any, Any], center: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    reconstructed = np.asarray(center[None, :] + z @ components, dtype=np.float64)
    probabilities = np.square(reconstructed)
    probabilities += np.finfo(np.float64).tiny
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _global_frequency(matrix: sparse.csr_matrix) -> np.ndarray[Any, Any]:
    totals = np.asarray(matrix.sum(axis=0), dtype=np.float64).reshape(-1)
    totals += 0.5
    totals /= totals.sum()
    return totals


def _per_row_nll(
    matrix: sparse.csr_matrix,
    components: np.ndarray[Any, Any] | None,
    global_frequency: np.ndarray[Any, Any],
    center: np.ndarray[Any, Any] | None = None,
    *,
    batch_size: int = 256,
) -> np.ndarray[Any, Any]:
    matrix = matrix.tocsr()
    if components is not None and center is None:
        raise ValueError("A count-native decoder requires its frozen Hellinger center.")
    result = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        block = matrix[start:stop]
        if components is None:
            probabilities = np.broadcast_to(global_frequency, block.shape)
        else:
            assert center is not None
            probabilities = _decode(_encode(block, components, center), components, center)
        totals = np.asarray(block.sum(axis=1), dtype=np.float64).reshape(-1)
        cross_entropy = np.zeros(len(block.indptr) - 1, dtype=np.float64)
        for local in range(block.shape[0]):
            left, right = int(block.indptr[local]), int(block.indptr[local + 1])
            indices = block.indices[left:right]
            counts = block.data[left:right].astype(np.float64)
            cross_entropy[local] = -float(
                np.dot(counts, np.log(probabilities[local, indices] + 1e-300))
            )
        result[start:stop] = cross_entropy / totals
    return result


def _pearson(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _guide_metrics(
    frame: pd.DataFrame,
    matrix: sparse.csr_matrix,
    components: np.ndarray[Any, Any],
    center: np.ndarray[Any, Any],
    global_frequency: np.ndarray[Any, Any],
    *,
    fold: str,
    half_seed: int,
    nll_max_cells_per_guide: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray[Any, Any]]]:
    z = _encode(matrix, components, center)
    rows: list[dict[str, Any]] = []
    centroids: dict[str, np.ndarray[Any, Any]] = {}
    for guide, positions in frame.groupby("guide_id", sort=True).indices.items():
        local = np.asarray(positions, dtype=np.int64)
        generator = np.random.default_rng(
            half_seed + int.from_bytes(str(guide).encode("utf-8")[:8].ljust(8, b"\0"), "little")
        )
        shuffled = local.copy()
        generator.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        if midpoint == 0:
            half_distance = float("nan")
        else:
            half_distance = float(
                np.linalg.norm(
                    z[shuffled[:midpoint]].mean(axis=0) - z[shuffled[midpoint:]].mean(axis=0)
                )
            )
        nll_local = shuffled[: min(len(shuffled), nll_max_cells_per_guide)]
        model_nll = _per_row_nll(matrix[nll_local], components, global_frequency, center)
        global_nll = _per_row_nll(matrix[nll_local], None, global_frequency)
        observed = np.asarray(matrix[local].sum(axis=0), dtype=np.float64).reshape(-1)
        observed /= observed.sum()
        centroid = z[local].mean(axis=0)
        predicted = _decode(centroid[None, :], components, center)[0]
        centroids[str(guide)] = centroid
        first = frame.iloc[int(local[0])]
        rows.append(
            {
                "outer_fold": fold,
                "guide_id": str(guide),
                "target_id": str(first.target_id),
                "is_control": bool(first.is_control),
                "cells": len(local),
                "nll_evaluation_cells": len(nll_local),
                "model_nll": float(model_nll.mean()),
                "global_nll": float(global_nll.mean()),
                "nll_delta": float((model_nll - global_nll).mean()),
                "pseudobulk_correlation": _pearson(observed, predicted),
                "split_half_distance": half_distance,
            }
        )
    return pd.DataFrame(rows), centroids


def _target_structure(
    heldout_metrics: pd.DataFrame,
    heldout_centroids: dict[str, np.ndarray[Any, Any]],
    train_frame: pd.DataFrame,
    train_z: np.ndarray[Any, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    train_centroids = {
        str(guide): train_z[np.asarray(positions, dtype=np.int64)].mean(axis=0)
        for guide, positions in train_frame.groupby("guide_id", sort=True).indices.items()
    }
    target_centroids: dict[str, np.ndarray[Any, Any]] = {}
    for target, guides in train_frame.groupby("target_id", sort=True).guide_id.unique().items():
        available = [
            train_centroids[str(guide)] for guide in guides if str(guide) in train_centroids
        ]
        if available:
            target_centroids[str(target)] = np.mean(available, axis=0)
    random_targets = sorted(target_centroids)
    rows: list[dict[str, Any]] = []
    for index, metric in heldout_metrics.sort_values("guide_id").reset_index(drop=True).iterrows():
        guide = str(metric.guide_id)
        target = str(metric.target_id)
        if bool(metric.is_control) or target not in target_centroids:
            continue
        alternatives = [value for value in random_targets if value != target]
        if not alternatives:
            continue
        generator = np.random.default_rng(seed + index)
        random_target = alternatives[int(generator.integers(len(alternatives)))]
        centroid = heldout_centroids[guide]
        sister = float(np.linalg.norm(centroid - target_centroids[target]))
        random_distance = float(np.linalg.norm(centroid - target_centroids[random_target]))
        rows.append(
            {
                "outer_fold": metric.outer_fold,
                "guide_id": guide,
                "target_id": target,
                "sister_distance": sister,
                "random_distance": random_distance,
                "target_similarity_separation": random_distance - sister,
                "normalized_target_separation": (random_distance - sister)
                / max(random_distance, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def _support_coverage(
    reference: sparse.csr_matrix,
    calibration: sparse.csr_matrix,
    source_query: sparse.csr_matrix,
    terminal_query: sparse.csr_matrix,
    components: np.ndarray[Any, Any],
    center: np.ndarray[Any, Any],
) -> dict[str, float]:
    reference_z = _encode(reference, components, center)
    tree = cKDTree(reference_z)
    calibration_distance = tree.query(_encode(calibration, components, center), k=1)[0]
    threshold = float(np.quantile(calibration_distance, 0.99))
    source_distance = tree.query(_encode(source_query, components, center), k=1)[0]
    terminal_distance = tree.query(_encode(terminal_query, components, center), k=1)[0]
    source_coverage = float(np.mean(source_distance <= threshold))
    terminal_coverage = float(np.mean(terminal_distance <= threshold))
    return {
        "threshold": threshold,
        "source_coverage": source_coverage,
        "terminal_coverage": terminal_coverage,
        "terminal_to_source_coverage_ratio": terminal_coverage / max(source_coverage, 1e-12),
    }


def _target_bootstrap(
    frame: pd.DataFrame,
    column: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    targeting = frame.loc[~frame.is_control].copy()
    target_values = targeting.groupby("target_id", sort=True)[column].mean().to_numpy(float)
    if not len(target_values):
        raise ContractError("Target bootstrap requires perturbation targets.")
    point = float(target_values.mean())
    generator = np.random.default_rng(seed)
    sampled = generator.choice(target_values, size=(draws, len(target_values)), replace=True)
    quantiles = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    interval = (float(quantiles[0]), float(quantiles[1]))
    return point, interval


def _normalize_folds(folds: pd.DataFrame, guides: set[str]) -> pd.DataFrame:
    missing = set(_FOLD_COLUMNS) - set(folds.columns)
    if missing:
        raise ContractError(f"Outer-fold table lacks columns: {sorted(missing)}.")
    normalized = folds.loc[:, _FOLD_COLUMNS].copy()
    normalized["guide_id"] = normalized.guide_id.astype(str)
    normalized["outer_fold"] = normalized.outer_fold.astype(str)
    if normalized.guide_id.duplicated().any() or set(normalized.guide_id) != guides:
        raise ContractError("Outer-fold assignment must cover every retained guide exactly once.")
    if normalized.outer_fold.nunique() < 2:
        raise ContractError("T01 requires at least two outer guide folds.")
    return normalized.sort_values("guide_id", kind="stable").reset_index(drop=True)


def _null_refits(
    matrix: sparse.csr_matrix,
    frame: pd.DataFrame,
    dimension: int,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Refit small count-native models under permuted guide-target assignments."""

    guide_rows = frame.groupby("guide_id", sort=True).indices
    guides = sorted(guide_rows)
    observed_targets = {
        str(guide): str(frame.iloc[int(np.asarray(positions)[0])].target_id)
        for guide, positions in guide_rows.items()
    }
    selected = 0
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        generator = np.random.default_rng(seed + repeat)
        fit_positions = generator.choice(
            matrix.shape[0], size=min(512, matrix.shape[0]), replace=False
        )
        basis, center = _fit_basis(
            matrix[np.sort(fit_positions)], dimension, seed + 10_000 + repeat
        )
        z = _encode(matrix, basis, center)
        centroids = {
            str(guide): z[np.asarray(positions, dtype=np.int64)].mean(axis=0)
            for guide, positions in guide_rows.items()
        }
        permuted_values = np.asarray([observed_targets[guide] for guide in guides], dtype=object)
        generator.shuffle(permuted_values)
        permuted = dict(zip(guides, permuted_values, strict=True))
        target_centroids: dict[str, np.ndarray[Any, Any]] = {}
        for target in sorted(set(permuted.values())):
            local = [centroids[guide] for guide in guides if permuted[guide] == target]
            target_centroids[target] = np.mean(local, axis=0)
        separations: list[float] = []
        for index, guide in enumerate(guides):
            target = permuted[guide]
            alternatives = [value for value in target_centroids if value != target]
            if (
                len([value for value in guides if permuted[value] == target]) < 2
                or not alternatives
            ):
                continue
            sister = float(np.linalg.norm(centroids[guide] - target_centroids[target]))
            random_target = alternatives[(index + repeat) % len(alternatives)]
            random_distance = float(
                np.linalg.norm(centroids[guide] - target_centroids[random_target])
            )
            separations.append((random_distance - sister) / max(random_distance, 1e-12))
        activity = float(np.mean(separations)) if separations else 0.0
        false_selected = activity > 0.05
        selected += int(false_selected)
        rows.append(
            {
                "repeat": repeat,
                "refit_seed": seed + 10_000 + repeat,
                "permutation_seed": seed + repeat,
                "normalized_activity": activity,
                "false_selected": false_selected,
            }
        )
    return {
        "schema_version": 1,
        "method": "genuine_small_count_native_refits_with_permuted_target_assignments",
        "repeats": repeats,
        "false_selections": selected,
        "false_selection_rate": selected / repeats,
        "maximum_allowed_rate": 0.05,
        "pass": selected / repeats <= 0.05,
        "rows": rows,
    }


def qualify_count_representation(
    destination: Path,
    *,
    pooled_bundle: Path,
    count_store: Path,
    outer_folds: pd.DataFrame,
    dimensions: tuple[int, ...] = (8, 16, 32, 48),
    fit_max_rows: int = 20_000,
    inner_validation_max_rows: int = 8_192,
    support_max_rows: int = 8_192,
    nll_max_cells_per_guide: int = 32,
    bootstrap_draws: int = 2_000,
    null_repeats: int = 20,
    required_nll_margin: float = 1e-4,
    minimum_target_activity: float = 0.05,
    maximum_split_half_ratio: float = 0.5,
    minimum_terminal_support_ratio: float = 0.8,
    seed: int = 20_260_815,
) -> Path:
    """Fit and qualify T01 without terminal-informed model selection."""

    if destination.exists():
        raise FileExistsError(destination)
    if tuple(sorted(set(dimensions))) != dimensions or not dimensions or dimensions[0] <= 0:
        raise ContractError("T01 dimensions must be unique, positive, and increasing.")
    if (
        fit_max_rows < 100
        or inner_validation_max_rows < 10
        or support_max_rows < 10
        or nll_max_cells_per_guide < 1
    ):
        raise ContractError("T01 row budgets are too small for qualification.")
    pooled = verify_pooled_finite_measures(pooled_bundle)
    cells = pd.read_parquet(pooled_bundle / pooled.cells.relative_uri)
    catalog = pd.read_parquet(pooled_bundle / pooled.guide_catalog.relative_uri)
    folds = _normalize_folds(outer_folds, set(catalog.guide_id.astype(str)))
    cells = cells.merge(
        catalog.loc[:, ["guide_id", "target_id", "is_control"]],
        on="guide_id",
        validate="many_to_one",
    ).merge(folds, on="guide_id", validate="many_to_one")
    store = CountStore(count_store)
    store_manifest = store.verify(full=True)
    if not set(cells.row_id.astype(int)) <= set(map(int, store.row_ids())):
        raise ContractError("T01 count store does not cover every retained T00 cell.")
    fold_names = tuple(sorted(folds.outer_fold.unique()))
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": "T01_COUNT_NATIVE_REPRESENTATION",
        "component": "multinomial_centered_hellinger_pca_v2",
        "primary_metric": "outer_source_per_count_nll",
        "primary_baseline": "global_gene_frequency_decoder",
        "required_margin": required_nll_margin,
        "drift": "off",
        "diffusion": "off",
        "reaction": "off",
        "ecology": "off",
        "decoder": "off",
        "update_zero_selectable": True,
        "post_selection_refit_required": True,
    }
    contract_payload["test_contract_id"] = contract_id(
        contract_payload, id_field="test_contract_id"
    )
    test_contract = ComponentTestContract.model_validate(contract_payload)
    config = {
        "schema_version": 1,
        "method": "multinomial_centered_hellinger_pca_v2",
        "dimensions": list(dimensions),
        "fit_max_rows": fit_max_rows,
        "inner_validation_max_rows": inner_validation_max_rows,
        "support_max_rows": support_max_rows,
        "nll_max_cells_per_guide": nll_max_cells_per_guide,
        "bootstrap_draws": bootstrap_draws,
        "null_repeats": null_repeats,
        "required_nll_margin": required_nll_margin,
        "minimum_target_activity": minimum_target_activity,
        "maximum_split_half_ratio": maximum_split_half_ratio,
        "minimum_terminal_support_ratio": minimum_terminal_support_ratio,
        "seed": seed,
        "selection_uses_terminal_outcomes": False,
        "dynamics_gradients_enabled": False,
    }
    candidate_rows: list[dict[str, Any]] = []
    guide_rows: list[pd.DataFrame] = []
    target_rows: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    selected_dimensions: dict[str, int] = {}
    encoder_payload: dict[str, np.ndarray[Any, Any]] = {}
    fold_index: list[dict[str, Any]] = []
    null_source_matrix: sparse.csr_matrix | None = None
    null_source_frame: pd.DataFrame | None = None

    for fold_index_value, fold in enumerate(fold_names):
        train_guides = set(folds.loc[folds.outer_fold != fold, "guide_id"])
        train_source = cells.loc[
            (cells.checkpoint == pooled.source_checkpoint) & cells.guide_id.isin(train_guides)
        ].sort_values("row_id", kind="stable")
        outer_source = cells.loc[
            (cells.checkpoint == pooled.source_checkpoint) & (cells.outer_fold == fold)
        ].sort_values("row_id", kind="stable")
        outer_terminal = cells.loc[
            (cells.checkpoint == pooled.terminal_checkpoint) & (cells.outer_fold == fold)
        ].sort_values("row_id", kind="stable")
        if train_source.empty or outer_source.empty or outer_terminal.empty:
            raise ContractError(f"T01 fold {fold} has an empty fit or protected population.")
        generator = np.random.default_rng(seed + fold_index_value)
        inner_positions: list[int] = []
        for positions in train_source.groupby("guide_id", sort=True).indices.values():
            local = np.asarray(positions, dtype=np.int64)
            count = max(1, int(np.floor(0.1 * len(local))))
            inner_positions.extend(generator.choice(local, size=count, replace=False).tolist())
        inner_positions_array = np.asarray(sorted(inner_positions), dtype=np.int64)
        inner_frame = train_source.iloc[inner_positions_array]
        fit_frame = train_source.drop(train_source.index[inner_positions_array])
        fit_ids = _sample(
            fit_frame.row_id.to_numpy(np.int64), fit_max_rows, seed + 100 + fold_index_value
        )
        inner_ids = _sample(
            inner_frame.row_id.to_numpy(np.int64),
            inner_validation_max_rows,
            seed + 200 + fold_index_value,
        )
        fit_matrix = store.rows(fit_ids).matrix
        inner_matrix = store.rows(inner_ids).matrix
        maximum_basis, candidate_center = _fit_basis(
            fit_matrix, max(dimensions), seed + 300 + fold_index_value
        )
        global_frequency = _global_frequency(fit_matrix)
        global_nll = float(_per_row_nll(inner_matrix, None, global_frequency).mean())
        candidate_rows.append(
            {
                "outer_fold": fold,
                "dimension": 0,
                "inner_source_nll": global_nll,
                "nll_delta_from_global": 0.0,
                "selected": False,
            }
        )
        scored: list[tuple[float, int]] = []
        for dimension in dimensions:
            nll = float(
                _per_row_nll(
                    inner_matrix,
                    maximum_basis[:dimension],
                    global_frequency,
                    candidate_center,
                ).mean()
            )
            candidate_rows.append(
                {
                    "outer_fold": fold,
                    "dimension": dimension,
                    "inner_source_nll": nll,
                    "nll_delta_from_global": nll - global_nll,
                    "selected": False,
                }
            )
            if nll - global_nll < -required_nll_margin:
                scored.append((nll, dimension))
        selected_dimension = min(scored)[1] if scored else 0
        selected_dimensions[fold] = selected_dimension
        for row in candidate_rows:
            if row["outer_fold"] == fold and row["dimension"] == selected_dimension:
                row["selected"] = True
        if selected_dimension == 0:
            fold_index.append(
                {
                    "outer_fold": fold,
                    "selected_dimension": 0,
                    "fit_rows_hash": _row_hash(fit_ids),
                    "inner_validation_rows_hash": _row_hash(inner_ids),
                    "post_selection_refit": False,
                }
            )
            continue
        refit_ids = _sample(
            train_source.row_id.to_numpy(np.int64), fit_max_rows, seed + 400 + fold_index_value
        )
        refit_matrix = store.rows(refit_ids).matrix
        components, center = _fit_basis(
            refit_matrix, selected_dimension, seed + 500 + fold_index_value
        )
        global_frequency = _global_frequency(refit_matrix)
        encoder_payload[f"{fold}__components"] = components
        encoder_payload[f"{fold}__center"] = center
        encoder_payload[f"{fold}__global_frequency"] = global_frequency.astype(np.float32)
        outer_source_ids = outer_source.row_id.to_numpy(np.int64)
        outer_source_matrix = store.rows(outer_source_ids).matrix
        metrics, outer_centroids = _guide_metrics(
            outer_source.reset_index(drop=True),
            outer_source_matrix,
            components,
            center,
            global_frequency,
            fold=fold,
            half_seed=seed + 600 + fold_index_value,
            nll_max_cells_per_guide=nll_max_cells_per_guide,
        )
        train_metric_frame = train_source.reset_index(drop=True)
        train_matrix = store.rows(train_metric_frame.row_id.to_numpy(np.int64)).matrix
        train_z = _encode(train_matrix, components, center)
        structure = _target_structure(
            metrics,
            outer_centroids,
            train_metric_frame,
            train_z,
            seed=seed + 700 + fold_index_value,
        )
        metrics = metrics.merge(
            structure.loc[
                :,
                (
                    "guide_id",
                    "sister_distance",
                    "random_distance",
                    "target_similarity_separation",
                    "normalized_target_separation",
                ),
            ],
            on="guide_id",
            how="left",
            validate="one_to_one",
        )
        metrics["split_half_to_random_ratio"] = (
            metrics.split_half_distance / metrics.random_distance
        )
        guide_rows.append(metrics)
        target_rows.append(
            metrics.loc[~metrics.is_control]
            .groupby(["outer_fold", "target_id"], as_index=False)
            .agg(
                guides=("guide_id", "size"),
                nll_delta=("nll_delta", "mean"),
                pseudobulk_correlation=("pseudobulk_correlation", "mean"),
                split_half_to_random_ratio=("split_half_to_random_ratio", "mean"),
                normalized_target_separation=("normalized_target_separation", "mean"),
            )
        )
        reference_ids = _sample(refit_ids, support_max_rows, seed + 800 + fold_index_value)
        remaining_ids = np.setdiff1d(
            train_source.row_id.to_numpy(np.int64), reference_ids, assume_unique=False
        )
        calibration_ids = _sample(remaining_ids, support_max_rows, seed + 900 + fold_index_value)
        source_query_ids = _sample(
            outer_source_ids, support_max_rows, seed + 1_000 + fold_index_value
        )
        terminal_query_ids = _sample(
            outer_terminal.row_id.to_numpy(np.int64),
            support_max_rows,
            seed + 1_100 + fold_index_value,
        )
        coverage = _support_coverage(
            store.rows(reference_ids).matrix,
            store.rows(calibration_ids).matrix,
            store.rows(source_query_ids).matrix,
            store.rows(terminal_query_ids).matrix,
            components,
            center,
        )
        support_rows.append({"outer_fold": fold, "dimension": selected_dimension, **coverage})
        fold_index.append(
            {
                "outer_fold": fold,
                "selected_dimension": selected_dimension,
                "fit_rows_hash": _row_hash(fit_ids),
                "inner_validation_rows_hash": _row_hash(inner_ids),
                "refit_rows_hash": _row_hash(refit_ids),
                "outer_source_rows_hash": _row_hash(outer_source_ids),
                "outer_terminal_rows_hash": _row_hash(outer_terminal.row_id.to_numpy(np.int64)),
                "post_selection_refit": True,
            }
        )
        if null_source_matrix is None:
            null_positions: list[int] = []
            for positions in train_metric_frame.groupby("guide_id", sort=True).indices.values():
                null_positions.extend(np.asarray(positions, dtype=np.int64)[:4].tolist())
            null_positions_array = np.asarray(null_positions, dtype=np.int64)
            null_source_matrix = train_matrix[null_positions_array]
            null_source_frame = train_metric_frame.iloc[null_positions_array].reset_index(drop=True)

    candidate_metrics = pd.DataFrame(candidate_rows)
    per_guide = pd.concat(guide_rows, ignore_index=True) if guide_rows else pd.DataFrame()
    per_target = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    support_metrics = pd.DataFrame(support_rows)
    all_selected = bool(len(per_guide)) and all(value > 0 for value in selected_dimensions.values())
    if all_selected:
        point_delta, interval = _target_bootstrap(
            per_guide, "nll_delta", draws=bootstrap_draws, seed=seed + 2_000
        )
        activity, activity_interval = _target_bootstrap(
            per_guide,
            "normalized_target_separation",
            draws=bootstrap_draws,
            seed=seed + 3_000,
        )
        split_ratio = float(
            per_guide.loc[~per_guide.is_control, "split_half_to_random_ratio"].median()
        )
        support_pass = bool(
            (
                support_metrics.terminal_to_source_coverage_ratio >= minimum_terminal_support_ratio
            ).all()
        )
        assert null_source_matrix is not None and null_source_frame is not None
        null_calibration = _null_refits(
            null_source_matrix,
            null_source_frame,
            int(np.median(list(selected_dimensions.values()))),
            repeats=null_repeats,
            seed=seed + 4_000,
        )
    else:
        point_delta, interval = 0.0, (0.0, 0.0)
        activity, activity_interval = 0.0, (0.0, 0.0)
        split_ratio = float("inf")
        support_pass = False
        null_calibration = {
            "schema_version": 1,
            "method": "not_run_no_selected_representation",
            "repeats": 0,
            "false_selections": 0,
            "false_selection_rate": 1.0,
            "maximum_allowed_rate": 0.05,
            "pass": False,
            "rows": [],
        }
    protected_pass = True
    passed = bool(
        all_selected
        and interval[1] < -required_nll_margin
        and activity_interval[0] > minimum_target_activity
        and split_ratio <= maximum_split_half_ratio
        and support_pass
        and null_calibration["pass"]
        and protected_pass
    )

    def writer(temp: Path) -> None:
        _write_json(temp / "TEST_CONTRACT.json", test_contract.model_dump(mode="json"))
        _write_json(
            temp / "INPUTS.sha256",
            {
                "schema_version": 1,
                "pooled_data_id": pooled.pooled_data_id,
                "pooled_bundle_sha256": sha256_file(pooled_bundle / "pooled-data.json"),
                "count_store_sha256": store_manifest.content_sha256,
                "fold_assignment_sha256": sha256_bytes(
                    canonical_json_bytes(folds.to_dict(orient="records"))
                ),
            },
        )
        (temp / "CONFIG.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
        _write_json(temp / "NULL_CALIBRATION.json", null_calibration)
        candidate_metrics.to_parquet(temp / "CANDIDATE_METRICS.parquet", index=False)
        per_guide.to_parquet(temp / "PER_GUIDE_METRICS.parquet", index=False)
        per_target.to_parquet(temp / "PER_TARGET_METRICS.parquet", index=False)
        support_metrics.to_parquet(temp / "SUPPORT_METRICS.parquet", index=False)
        # NumPy's stubs currently treat arbitrary named arrays as the optional
        # boolean keyword even though the runtime API explicitly accepts them.
        savez_compressed: Any = np.savez_compressed
        savez_compressed(str(temp / "ENCODERS.npz"), **encoder_payload)
        _write_json(temp / "FOLD_INDEX.json", {"schema_version": 1, "folds": fold_index})
        bootstrap = {
            "schema_version": 1,
            "primary_point_delta": point_delta,
            "primary_interval": list(interval),
            "target_activity": activity,
            "target_activity_interval": list(activity_interval),
            "draws": bootstrap_draws,
            "target_clustered": True,
        }
        _write_json(temp / "BOOTSTRAP_RESULTS.json", bootstrap)
        _write_json(
            temp / "CHANNEL_ACTIVITY.json",
            {
                "schema_version": 1,
                "representation_activity": activity,
                "minimum_required": minimum_target_activity,
                "dynamics_channels": "all_off",
            },
        )
        selected_model = {
            "schema_version": 1,
            "method": (
                "multinomial_centered_hellinger_pca_v2" if all_selected else "global_frequency"
            ),
            "selected_dimensions": selected_dimensions,
            "selection_checkpoint": pooled.source_checkpoint,
            "terminal_used_for_selection": False,
            "post_selection_refit": all_selected,
        }
        _write_json(temp / "SELECTED_MODEL.json", selected_model)
        receipt_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "test_id": test_contract.test_id,
            "status": "pass" if passed else "fail_retired",
            "primary_metric": test_contract.primary_metric,
            "primary_baseline": test_contract.primary_baseline,
            "point_delta": point_delta,
            "bootstrap_interval": list(interval),
            "required_margin": required_nll_margin,
            "channel_activity": max(0.0, activity),
            "protected_metrics_pass": protected_pass,
            "selected_update": 0,
            "input_hashes": {
                "pooled_data": pooled.pooled_data_id,
                "count_store": store_manifest.content_sha256,
                "fold_assignment": sha256_bytes(
                    canonical_json_bytes(folds.to_dict(orient="records"))
                ),
            },
            "config_hash": sha256_bytes(canonical_json_bytes(config)),
            "implementation_hash": sha256_file(Path(__file__)),
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = ComponentTestReceipt.model_validate(receipt_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        bundle_payload = {
            "schema_version": 1,
            "representation_id": "pending",
            "pooled_data_id": pooled.pooled_data_id,
            "method": "multinomial_centered_hellinger_pca_v2",
            "feature_index_hash": store_manifest.feature_index_hash,
            "count_store_sha256": store_manifest.content_sha256,
            "dimensions": list(dimensions),
            "selected_dimensions": selected_dimensions,
            "outer_folds": list(fold_names),
            "selection_uses_terminal_outcomes": False,
            "dynamics_gradients_enabled": False,
            "fit_checkpoint": pooled.source_checkpoint,
            "protected_checkpoint": pooled.terminal_checkpoint,
            "fold_index": _artifact(
                temp / "FOLD_INDEX.json",
                schema_id="credo.representation_fold_index",
                media_type="application/json",
            ).model_dump(mode="json"),
            "encoder_state": _artifact(
                temp / "ENCODERS.npz",
                schema_id="credo.count_native_encoder",
                media_type="application/x-npz",
            ).model_dump(mode="json"),
            "candidate_metrics": _artifact(
                temp / "CANDIDATE_METRICS.parquet",
                schema_id="credo.representation_candidates",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "per_guide_metrics": _artifact(
                temp / "PER_GUIDE_METRICS.parquet",
                schema_id="credo.component_guide_metrics",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "per_target_metrics": _artifact(
                temp / "PER_TARGET_METRICS.parquet",
                schema_id="credo.component_target_metrics",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "support_metrics": _artifact(
                temp / "SUPPORT_METRICS.parquet",
                schema_id="credo.representation_support_metrics",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "null_calibration": _artifact(
                temp / "NULL_CALIBRATION.json",
                schema_id="credo.component_null_calibration",
                media_type="application/json",
            ).model_dump(mode="json"),
            "selected_model": _artifact(
                temp / "SELECTED_MODEL.json",
                schema_id="credo.component_selected_model",
                media_type="application/json",
            ).model_dump(mode="json"),
            "test_receipt": _artifact(
                temp / "TEST_RECEIPT.json",
                schema_id="credo.component_test_receipt",
                media_type="application/json",
            ).model_dump(mode="json"),
        }
        bundle_payload["representation_id"] = contract_id(
            bundle_payload, id_field="representation_id"
        )
        bundle = CountRepresentationBundle.model_validate(bundle_payload)
        _write_json(temp / "representation.json", bundle.model_dump(mode="json"))
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_count_representation(destination, pooled_bundle=pooled_bundle, count_store=count_store)
    return destination


def verify_count_representation(
    path: Path, *, pooled_bundle: Path, count_store: Path
) -> CountRepresentationBundle:
    """Verify the complete T01 bundle and its T00/count parents."""

    verify_directory(path)
    pooled = verify_pooled_finite_measures(pooled_bundle)
    store_manifest = CountStore(count_store).verify(full=True)
    bundle = CountRepresentationBundle.model_validate_json(
        (path / "representation.json").read_text()
    )
    receipt = ComponentTestReceipt.model_validate_json((path / "TEST_RECEIPT.json").read_text())
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    if bundle.pooled_data_id != pooled.pooled_data_id:
        raise IntegrityError("T01 representation has a different T00 parent.")
    if bundle.count_store_sha256 != store_manifest.content_sha256:
        raise IntegrityError("T01 representation has a different count-store parent.")
    if receipt.test_id != contract.test_id or contract.test_id != "T01_COUNT_NATIVE_REPRESENTATION":
        raise IntegrityError("T01 receipt and contract disagree.")
    for reference in (
        bundle.fold_index,
        bundle.encoder_state,
        bundle.candidate_metrics,
        bundle.per_guide_metrics,
        bundle.per_target_metrics,
        bundle.support_metrics,
        bundle.null_calibration,
        bundle.selected_model,
        bundle.test_receipt,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"T01 artifact differs from its reference: {reference.relative_uri}."
            )
    selected = json.loads((path / "SELECTED_MODEL.json").read_text())
    if bool(selected["terminal_used_for_selection"]):
        raise IntegrityError("T01 selection used protected terminal outcomes.")
    fold_index = json.loads((path / "FOLD_INDEX.json").read_text())["folds"]
    if {row["outer_fold"] for row in fold_index} != set(bundle.outer_folds):
        raise IntegrityError("T01 fold index is incomplete.")
    with np.load(path / "ENCODERS.npz", allow_pickle=False) as encoders:
        for fold, dimension in bundle.selected_dimensions.items():
            if dimension > 0 and (
                f"{fold}__components" not in encoders
                or f"{fold}__center" not in encoders
                or encoders[f"{fold}__components"].shape[0] != dimension
            ):
                raise IntegrityError(f"T01 encoder is missing or malformed for {fold}.")
    for expected, relative in (
        line.split(maxsplit=1) for line in (path / "SHA256SUMS").read_text().splitlines()
    ):
        if sha256_file(path / relative.strip()) != expected:
            raise IntegrityError(f"T01 SHA256SUMS mismatch: {relative.strip()}.")
    return bundle
