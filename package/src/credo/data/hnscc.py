"""Reusable HNSCC data helpers for CREDO experiments."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import (
    CellStateTable,
    MassTable,
    POOLED_SAMPLE_ID,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from .filters import filter_state_supported_perturbations

P4 = "P4"
P60 = "P60"
TIME_MAP = {4.0: P4, 60.0: P60}
DEFAULT_STATE_KEY = "Cell type annotation"
DEFAULT_WTA_COLUMN = "Library"
DEFAULT_LATENT_KEY = "X_pca"
DEFAULT_TRAIN_WTAS = (
    "wta4",
    "wta5",
    "wta6",
    "wta7",
    "wta9",
    "wta13",
    "wta14",
    "wta15",
    "wta16",
    "wta17",
    "wta18",
)
DEFAULT_TEST_WTAS = ("wta8", "wta10", "wta11", "wta12")
DEFAULT_RANDOM_STRATIFY_COLS = ("Time point", "perturbation_id")


@dataclass(frozen=True)
class HNSCCSplitResult:
    split: pd.Series
    manifest: pd.DataFrame
    metadata: dict


def load_hnscc(path: str, *, latent_key: str = DEFAULT_LATENT_KEY) -> tuple[pd.DataFrame, np.ndarray]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    if latent_key not in adata.obsm:
        available = sorted(str(key) for key in adata.obsm.keys())
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        raise KeyError(
            f"Requested latent key {latent_key!r} not found in adata.obsm. "
            f"Available keys: {available}"
        )
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs, latent


def load_hnscc_obs(path: str) -> pd.DataFrame:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs


def _coerce_gene_mask(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(bool).to_numpy()
    text = values.astype(str).str.lower()
    return text.isin({"1", "true", "t", "yes", "y"}).to_numpy()


def _validate_count_matrix(
    matrix: sp.spmatrix | np.ndarray,
    name: str = "expression",
    *,
    strict: bool = True,
) -> None:
    """Validate that a matrix looks like raw counts (nonneg, integer-like)."""
    if sp.issparse(matrix):
        data = matrix.data
    else:
        data = np.asarray(matrix).ravel()
    if len(data) == 0:
        return
    if np.any(data < 0):
        raise ValueError(
            f"{name} matrix contains negative values — expected raw counts. "
            "Check that you are loading the correct layer."
        )
    sample = data[:min(100_000, len(data))]
    frac_integer = np.mean(np.abs(sample - np.round(sample)) < 1e-4)
    if frac_integer < 0.9:
        import warnings
        message = (
            f"{name} matrix has only {frac_integer:.0%} near-integer values in a sample "
            f"of {len(sample)} entries. Expected raw counts — double-check the source layer."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)


def _materialize_chunk(chunk) -> sp.csr_matrix | np.ndarray:
    if hasattr(chunk, "to_memory"):
        chunk = chunk.to_memory()
    if sp.issparse(chunk):
        return chunk.tocsr().astype(np.float32)
    return np.asarray(chunk, dtype=np.float32)


def _chunked_row_sums(
    source,
    *,
    n_rows: int,
    chunk_size: int = 4096,
) -> np.ndarray:
    if sp.issparse(source):
        return np.asarray(source.sum(axis=1)).ravel().astype(np.float32)
    if isinstance(source, np.ndarray):
        return source.sum(axis=1, dtype=np.float32)

    totals = np.zeros(n_rows, dtype=np.float32)
    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        batch = _materialize_chunk(source[start:stop])
        if sp.issparse(batch):
            totals[start:stop] = np.asarray(batch.sum(axis=1)).ravel().astype(np.float32)
        else:
            totals[start:stop] = batch.sum(axis=1, dtype=np.float32)
    return totals


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na"}:
        return None
    return text


def _row_index_array(row_indices: Sequence[int] | np.ndarray | None, *, n_rows: int) -> np.ndarray | None:
    if row_indices is None:
        return None
    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional.")
    if len(rows) and (rows.min() < 0 or rows.max() >= n_rows):
        raise IndexError("row_indices contains values outside the expression matrix.")
    return rows


def _row_batches(
    *,
    n_rows: int,
    row_indices: np.ndarray | None = None,
    chunk_size: int = 1024,
) -> list[np.ndarray]:
    chunk_size = max(1, int(chunk_size))
    if row_indices is None:
        return [
            np.arange(start, min(start + chunk_size, n_rows), dtype=np.int64)
            for start in range(0, n_rows, chunk_size)
        ]
    return [
        row_indices[start:min(start + chunk_size, len(row_indices))]
        for start in range(0, len(row_indices), chunk_size)
    ]


def _chunk_shards(row_batches: list[np.ndarray], n_workers: int) -> list[list[np.ndarray]]:
    workers = max(1, min(int(n_workers), len(row_batches)))
    shard_size = int(np.ceil(len(row_batches) / workers))
    return [
        row_batches[start:start + shard_size]
        for start in range(0, len(row_batches), shard_size)
    ]


def _materialize_expression_and_totals_serial(
    source,
    selected_idx: np.ndarray,
    *,
    n_rows: int,
    row_indices: np.ndarray | None = None,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, sp.csr_matrix]:
    rows = _row_batches(n_rows=n_rows, row_indices=row_indices, chunk_size=chunk_size)
    totals_parts: list[np.ndarray] = []
    selected_parts: list[sp.csr_matrix] = []
    for row_batch in rows:
        batch = _materialize_chunk(source[row_batch, :])
        if sp.issparse(batch):
            totals_parts.append(np.asarray(batch.sum(axis=1)).ravel().astype(np.float32))
            selected_parts.append(batch[:, selected_idx].tocsr().astype(np.float32))
        else:
            arr = np.asarray(batch, dtype=np.float32)
            totals_parts.append(arr.sum(axis=1, dtype=np.float32))
            selected_parts.append(sp.csr_matrix(arr[:, selected_idx]))
    if totals_parts:
        totals = np.concatenate(totals_parts).astype(np.float32, copy=False)
        expr = sp.vstack(selected_parts, format="csr").astype(np.float32)
    else:
        totals = np.zeros(0, dtype=np.float32)
        expr = sp.csr_matrix((0, len(selected_idx)), dtype=np.float32)
    return totals, expr


def _expression_source_from_adata(adata: ad.AnnData, *, layer: str | None, use_raw: bool):
    if layer is not None:
        return adata.layers[layer]
    if use_raw:
        return adata.raw.X
    return adata.X


def _read_expression_shard_worker(payload: tuple) -> tuple[np.ndarray, sp.csr_matrix]:
    path, layer, use_raw, selected_idx_list, row_batch_lists = payload
    selected_idx = np.asarray(selected_idx_list, dtype=np.int64)
    adata = ad.read_h5ad(path, backed="r")
    try:
        source = _expression_source_from_adata(adata, layer=layer, use_raw=use_raw)
        totals_parts: list[np.ndarray] = []
        selected_parts: list[sp.csr_matrix] = []
        for row_batch_list in row_batch_lists:
            row_batch = np.asarray(row_batch_list, dtype=np.int64)
            batch = _materialize_chunk(source[row_batch, :])
            if sp.issparse(batch):
                totals_parts.append(np.asarray(batch.sum(axis=1)).ravel().astype(np.float32))
                selected_parts.append(batch[:, selected_idx].tocsr().astype(np.float32))
            else:
                arr = np.asarray(batch, dtype=np.float32)
                totals_parts.append(arr.sum(axis=1, dtype=np.float32))
                selected_parts.append(sp.csr_matrix(arr[:, selected_idx]))
        return (
            np.concatenate(totals_parts).astype(np.float32, copy=False),
            sp.vstack(selected_parts, format="csr").astype(np.float32),
        )
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()


def _materialize_expression_and_totals_parallel(
    path: str,
    *,
    layer: str | None,
    use_raw: bool,
    selected_idx: np.ndarray,
    n_rows: int,
    row_indices: np.ndarray | None = None,
    chunk_size: int = 1024,
    n_workers: int = 0,
) -> tuple[np.ndarray, sp.csr_matrix]:
    rows = _row_batches(n_rows=n_rows, row_indices=row_indices, chunk_size=chunk_size)
    if not rows:
        return (
            np.zeros(0, dtype=np.float32),
            sp.csr_matrix((0, len(selected_idx)), dtype=np.float32),
        )
    shards = _chunk_shards(rows, n_workers)
    payloads = [
        (
            str(path),
            layer,
            bool(use_raw),
            selected_idx.astype(np.int64).tolist(),
            [row_batch.tolist() for row_batch in shard],
        )
        for shard in shards
    ]
    totals_parts: list[np.ndarray] = []
    selected_parts: list[sp.csr_matrix] = []
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        for totals, selected in pool.map(_read_expression_shard_worker, payloads):
            totals_parts.append(totals)
            selected_parts.append(selected)
    return (
        np.concatenate(totals_parts).astype(np.float32, copy=False),
        sp.vstack(selected_parts, format="csr").astype(np.float32),
    )


def _dense_cache_limit_bytes(max_gb: float) -> int:
    if max_gb <= 0:
        return 0
    return int(max_gb * (1024 ** 3))


def _maybe_dense_cache(
    matrix: sp.csr_matrix | np.ndarray,
    *,
    max_gb: float,
) -> tuple[sp.csr_matrix | np.ndarray, bool]:
    max_bytes = _dense_cache_limit_bytes(max_gb)
    if max_bytes <= 0:
        return matrix, False
    n_rows, n_cols = matrix.shape
    needed = int(n_rows) * int(n_cols) * np.dtype(np.float32).itemsize
    if needed > max_bytes:
        return matrix, False
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32), True
    return np.asarray(matrix, dtype=np.float32), True


def load_hnscc_expression(
    path: str,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    gene_mask_col: str | None = None,
    top_genes: int = 0,
    validate_counts: bool = True,
    strict_layer: bool = True,
    strict_counts: bool = True,
    allow_full_gene_scan: bool = False,
    allow_precomputed_hv_score: bool = False,
    allow_empty_gene_mask_fallback: bool = False,
    row_indices: Sequence[int] | np.ndarray | None = None,
    n_workers: int = 0,
    chunk_size: int = 1024,
) -> tuple[pd.DataFrame, sp.csr_matrix, list[str], dict]:
    """Load expression matrix from an h5ad file.

    Parameters
    ----------
    path : str
        Path to h5ad file.
    layer : str | None
        AnnData source to read. ``None`` reads ``adata.X``. If a layer name is
        given, it must exist unless ``strict_layer=False``.
    use_raw : bool
        If *layer* is ``None`` and *use_raw* is True, read ``adata.raw.X``.
        If ``adata.raw`` is absent, this now fails loudly by default instead of
        silently falling back to ``adata.X``.
    gene_mask_col : str
        Optional column in ``adata.var`` used as a candidate-gene mask.
    top_genes : int
        Maximum number of genes to select. ``0`` keeps the full candidate set.
        Direct top-gene selection from a precomputed full-dataset ``hv_score`` is
        evaluation-unsafe by default and therefore disabled unless
        ``allow_precomputed_hv_score=True``.
    validate_counts : bool
        If True, check that the loaded matrix is nonneg and integer-like.
    strict_layer : bool
        If True, a missing requested layer raises an error instead of silently
        falling back to another expression source.
    strict_counts : bool
        If True, a non-count-like matrix raises an error instead of a warning.
    """
    layer = _normalize_optional_text(layer)
    gene_mask_col = _normalize_optional_text(gene_mask_col)

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    if layer is not None and layer in adata.layers:
        source = adata.layers[layer]
        resolved_layer = layer
        var = adata.var.copy()
    elif layer is not None and strict_layer:
        available = sorted(str(key) for key in adata.layers.keys())
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        raise KeyError(
            f"Requested expression layer {layer!r} not found. Available layers: {available}. "
            "Pass --vae-layer explicitly or disable strict_layer if fallback behavior is desired."
        )
    elif layer is None and use_raw:
        if adata.raw is None:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            raise KeyError(
                "use_raw=True requested but adata.raw is not available. "
                "Disable use_raw or provide an explicit count layer."
            )
        source = adata.raw.X
        resolved_layer = "raw"
        var = adata.raw.var.copy()
    else:
        source = adata.X
        resolved_layer = "X"
        var = adata.var.copy()

    n_rows, n_cols = int(source.shape[0]), int(source.shape[1])
    resolved_rows = _row_index_array(row_indices, n_rows=n_rows)
    if resolved_rows is not None:
        obs = obs.iloc[resolved_rows].copy()
    if validate_counts:
        sample_rows = min(256, n_rows)
        sample = _materialize_chunk(source[:sample_rows])
        _validate_count_matrix(
            sample,
            name=f"expression (layer={resolved_layer!r})",
            strict=strict_counts,
        )

    if gene_mask_col:
        if gene_mask_col not in var.columns:
            available = sorted(str(col) for col in var.columns)
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            raise KeyError(
                f"Requested gene mask column {gene_mask_col!r} not found in adata.var. "
                f"Available columns: {available}"
            )
        candidate_mask = _coerce_gene_mask(var[gene_mask_col])
    else:
        candidate_mask = np.ones(len(var), dtype=bool)

    candidate_idx = np.flatnonzero(candidate_mask)
    if len(candidate_idx) == 0:
        if gene_mask_col and not allow_empty_gene_mask_fallback:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            raise ValueError(
                f"Gene mask column {gene_mask_col!r} selected zero genes. "
                "Fix the mask or explicitly enable allow_empty_gene_mask_fallback."
            )
        candidate_idx = np.arange(len(var), dtype=np.int64)

    if (
        gene_mask_col is None
        and top_genes <= 0
        and len(candidate_idx) > 5000
        and not allow_full_gene_scan
    ):
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        raise ValueError(
            "Refusing to materialize the full transcriptome candidate matrix. "
            "Pass a gene mask column such as 'hv_gene', request a smaller top_genes "
            "panel, or set allow_full_gene_scan=True explicitly."
        )

    if top_genes > 0 and len(candidate_idx) > top_genes:
        if allow_precomputed_hv_score and "hv_score" in var.columns:
            scores = pd.to_numeric(var["hv_score"], errors="coerce").fillna(-np.inf).to_numpy()
            ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
            selected_idx = ranked[:top_genes]
        else:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            raise ValueError(
                "Direct top_genes selection inside load_hnscc_expression() is disabled by default "
                "because it can use precomputed full-dataset variability statistics. "
                "Use build_vae_latent() for split-safe train-only HVG selection, or pass "
                "allow_precomputed_hv_score=True explicitly for exploratory use."
            )
    else:
        selected_idx = candidate_idx

    if int(n_workers) > 1:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        full_library_totals, expr = _materialize_expression_and_totals_parallel(
            path,
            layer=layer,
            use_raw=use_raw,
            selected_idx=selected_idx,
            n_rows=n_rows,
            row_indices=resolved_rows,
            chunk_size=chunk_size,
            n_workers=n_workers,
        )
    else:
        full_library_totals, expr = _materialize_expression_and_totals_serial(
            source,
            selected_idx,
            n_rows=n_rows,
            row_indices=resolved_rows,
            chunk_size=chunk_size,
        )

    gene_names = [str(name) for name in var.index[selected_idx].tolist()]
    meta = {
        "gene_mask_col": gene_mask_col,
        "top_genes": int(top_genes),
        "n_selected_genes": int(len(gene_names)),
        "requested_layer": layer,
        "layer": resolved_layer,
        "gene_names": gene_names,
        "selected_gene_indices": selected_idx.astype(np.int64).tolist(),
        "full_library_totals": full_library_totals,
        "row_indices": resolved_rows.astype(np.int64).tolist() if resolved_rows is not None else None,
        "expression_workers": int(n_workers),
        "expression_chunk_size": int(chunk_size),
        "allow_full_gene_scan": bool(allow_full_gene_scan),
        "allow_empty_gene_mask_fallback": bool(allow_empty_gene_mask_fallback),
    }
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs, expr, gene_names, meta


def _gene_mean_variance(matrix: sp.csr_matrix | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
        second = np.asarray(matrix.power(2).mean(axis=0)).ravel().astype(np.float64)
    else:
        arr = np.asarray(matrix, dtype=np.float64)
        mean = arr.mean(axis=0)
        second = np.mean(arr ** 2, axis=0)
    variance = np.maximum(second - mean ** 2, 0.0)
    return mean, variance


def _rank_train_hv_genes(
    matrix: sp.csr_matrix | np.ndarray,
    *,
    n_genes: int,
) -> np.ndarray:
    n_available = int(matrix.shape[1])
    if n_genes <= 0 or n_genes >= n_available:
        return np.arange(n_available, dtype=np.int64)
    mean, variance = _gene_mean_variance(matrix)
    dispersion = np.full(n_available, -np.inf, dtype=np.float64)
    valid = mean > 0
    dispersion[valid] = np.log1p(variance[valid] / np.maximum(mean[valid], 1e-8))
    ranked = np.argsort(dispersion)[::-1]
    return ranked[:n_genes].astype(np.int64)


def _rank_to_unit_scores(ranked: np.ndarray, n_items: int) -> np.ndarray:
    scores = np.zeros(n_items, dtype=np.float64)
    if n_items == 0:
        return scores
    values = np.linspace(1.0, 0.0, num=n_items, endpoint=False, dtype=np.float64)
    scores[ranked] = values
    return scores


def _rank_train_hv_genes_batch_aware(
    matrix: sp.csr_matrix | np.ndarray,
    obs: pd.DataFrame,
    *,
    n_genes: int,
    batch_col: str = DEFAULT_WTA_COLUMN,
    time_col: str = "Time point",
    min_cells_per_batch: int = 256,
) -> np.ndarray:
    n_available = int(matrix.shape[1])
    if n_genes <= 0 or n_genes >= n_available:
        return np.arange(n_available, dtype=np.int64)
    if batch_col not in obs.columns or time_col not in obs.columns:
        return _rank_train_hv_genes(matrix, n_genes=n_genes)

    time_labels = obs[time_col].astype(str).fillna("NA").reset_index(drop=True)
    batch_labels = obs[batch_col].astype(str).fillna("NA").reset_index(drop=True)
    unique_times = sorted(time_labels.unique().tolist())
    if not unique_times:
        return _rank_train_hv_genes(matrix, n_genes=n_genes)

    selected: list[int] = []
    selected_set: set[int] = set()
    quota = max(1, n_genes // max(len(unique_times), 1))
    per_time_scores: list[np.ndarray] = []

    for time_value in unique_times:
        time_mask = time_labels.eq(time_value).to_numpy()
        n_time_cells = int(time_mask.sum())
        if n_time_cells == 0:
            continue
        matrix_time = matrix[time_mask]
        ranked_time = _rank_train_hv_genes(matrix_time, n_genes=n_available)
        score_time = _rank_to_unit_scores(ranked_time, n_available)

        time_batches = batch_labels.loc[time_mask]
        batch_score_parts: list[np.ndarray] = []
        for batch_value in sorted(time_batches.unique().tolist()):
            batch_mask = time_batches.eq(batch_value).to_numpy()
            if int(batch_mask.sum()) < int(min_cells_per_batch):
                continue
            ranked_batch = _rank_train_hv_genes(matrix_time[batch_mask], n_genes=n_available)
            batch_score_parts.append(_rank_to_unit_scores(ranked_batch, n_available))

        if batch_score_parts:
            score_time = 0.5 * score_time + 0.5 * np.mean(batch_score_parts, axis=0)

        per_time_scores.append(score_time)
        for gene_idx in np.argsort(score_time)[::-1]:
            gene_idx = int(gene_idx)
            if gene_idx in selected_set:
                continue
            selected.append(gene_idx)
            selected_set.add(gene_idx)
            if len(selected) >= quota * len(per_time_scores):
                break

    global_rank = _rank_train_hv_genes(matrix, n_genes=n_available)
    global_score = _rank_to_unit_scores(global_rank, n_available)
    if per_time_scores:
        global_score = 0.5 * global_score + 0.5 * np.mean(per_time_scores, axis=0)

    for gene_idx in np.argsort(global_score)[::-1]:
        gene_idx = int(gene_idx)
        if gene_idx in selected_set:
            continue
        selected.append(gene_idx)
        selected_set.add(gene_idx)
        if len(selected) >= n_genes:
            break

    return np.asarray(selected[:n_genes], dtype=np.int64)


def _vae_cache_paths(save_dir: str | Path) -> tuple[Path, Path, Path]:
    save_path = Path(save_dir)
    return save_path, save_path / "vae_metadata.json", save_path / "latent_all_std.npy"


def _vae_cache_matches(save_dir: str | Path, *, expected: dict) -> bool:
    _, meta_path, _ = _vae_cache_paths(save_dir)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    checks = {
        "requested_layer": meta.get("requested_layer"),
        "source_layer": meta.get("source_layer"),
        "target_sum": meta.get("target_sum"),
        "selected_gene_indices": meta.get("selected_gene_indices"),
        "train_cell_indices": meta.get("train_cell_indices"),
        "kept_positions": meta.get("kept_positions"),
        "split_manifest_hash": meta.get("split_manifest_hash"),
        "vae_hyperparams": meta.get("vae_hyperparams"),
    }
    return checks == expected


def _split_manifest_hash(obs: pd.DataFrame, split: pd.Series) -> str:
    frame = pd.DataFrame(
        {
            "cell_id": obs["cell_id"].astype(str).to_numpy(),
            "split": split.astype(str).to_numpy(),
        }
    )
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def clean_perturbation_ids(obs: pd.DataFrame) -> pd.Series:
    if "perturbation_gene" in obs.columns:
        pid = obs["perturbation_gene"].astype(str).copy()
    else:
        pid = obs["target_gene"].astype(str).copy()
    pid = pid.replace({"": "ctrl", "nan": "ctrl", "None": "ctrl"})
    pid.loc[obs["is_control"].astype(bool).to_numpy()] = "ctrl"
    return pid


def time_labels(obs: pd.DataFrame) -> pd.Series:
    vals = pd.to_numeric(obs["Time point"], errors="coerce")
    labels = vals.map(TIME_MAP)
    if labels.isna().any():
        missing = sorted(vals[labels.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected time points in dataset: {missing}")
    return labels


def prepare_hnscc_obs(
    obs: pd.DataFrame,
    *,
    guide_confident_only: bool = True,
    state_key: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    keep = np.ones(len(obs), dtype=bool)
    if guide_confident_only and "guide_confident" in obs.columns:
        keep &= obs["guide_confident"].fillna(False).to_numpy(dtype=bool)
    prepared = obs.loc[keep].copy()
    kept_positions = np.flatnonzero(keep)
    prepared["perturbation_id"] = clean_perturbation_ids(prepared)
    prepared["time_label"] = time_labels(prepared)
    prepared["sample_id"] = (
        prepared["Library"]
        .astype(str)
        .replace({"": POOLED_SAMPLE_ID, "nan": POOLED_SAMPLE_ID, "None": POOLED_SAMPLE_ID})
    )
    prepared["cell_id"] = prepared["cell_id"].astype(str)
    if state_key:
        if state_key not in prepared.columns:
            raise KeyError(f"Requested state_key {state_key!r} not present in obs.")
        prepared[state_key] = prepared[state_key].astype(str)
    return prepared, kept_positions


def parse_list_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_wta_split(
    obs: pd.DataFrame,
    *,
    wta_column: str,
    train_wtas: Sequence[str],
    test_wtas: Sequence[str],
) -> dict:
    available = sorted(obs[wta_column].astype(str).unique().tolist())
    train_set = set(train_wtas)
    test_set = set(test_wtas)
    overlap = sorted(train_set & test_set)
    if overlap:
        raise ValueError(f"Train/test WTA overlap detected: {overlap}")
    missing = sorted((train_set | test_set) - set(available))
    if missing:
        raise ValueError(f"Unknown WTAs in requested split: {missing}")
    uncovered = sorted(set(available) - train_set - test_set)
    if uncovered:
        raise ValueError(f"Some WTAs are not assigned to train or test: {uncovered}")
    return {
        "available_wtas": available,
        "train_wtas": sorted(train_set),
        "test_wtas": sorted(test_set),
    }


def _build_stratify_key(obs: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    parts = [obs[col].astype(str).fillna("NA") for col in cols]
    key = parts[0].copy()
    for part in parts[1:]:
        key = key + "__" + part
    return key


def make_random_split(
    obs: pd.DataFrame,
    *,
    train_frac: float,
    seed: int,
    stratify_cols: Sequence[str] = DEFAULT_RANDOM_STRATIFY_COLS,
) -> HNSCCSplitResult:
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")
    for col in stratify_cols:
        if col not in obs.columns:
            raise KeyError(f"Stratify column {col!r} not present in obs.")

    rng = np.random.default_rng(seed)
    stratify_key = _build_stratify_key(obs, stratify_cols)
    split = pd.Series("train", index=obs.index, dtype="object", name="split")
    manifest_rows: list[dict] = []

    grouped = obs.assign(_stratify_key=stratify_key).groupby("_stratify_key", sort=True, observed=True)
    for key, frame in grouped:
        idx = frame.index.to_numpy()
        n = len(idx)
        if n <= 1:
            n_train = n
        else:
            n_train = int(round(train_frac * n))
            n_train = max(1, min(n - 1, n_train))
        perm = rng.permutation(n)
        test_idx = idx[perm[n_train:]]
        split.loc[test_idx] = "test"
        row = {"split_group": key, "n_cells": int(n), "n_train": int(n_train), "n_test": int(n - n_train)}
        for col in stratify_cols:
            row[col] = frame[col].iloc[0]
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows).sort_values(list(stratify_cols)).reset_index(drop=True)
    metadata = {
        "split_strategy": "random",
        "train_frac": float(train_frac),
        "seed": int(seed),
        "stratify_cols": list(stratify_cols),
        "n_train_cells": int((split == "train").sum()),
        "n_test_cells": int((split == "test").sum()),
        "n_split_groups": int(len(manifest)),
    }
    return HNSCCSplitResult(split=split, manifest=manifest, metadata=metadata)


def make_random_kfold_split(
    obs: pd.DataFrame,
    *,
    n_folds: int,
    fold_index: int,
    seed: int,
    stratify_cols: Sequence[str] = DEFAULT_RANDOM_STRATIFY_COLS,
) -> HNSCCSplitResult:
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if not 0 <= fold_index < n_folds:
        raise ValueError(f"fold_index must be in [0, {n_folds}), got {fold_index}.")
    for col in stratify_cols:
        if col not in obs.columns:
            raise KeyError(f"Stratify column {col!r} not present in obs.")

    rng = np.random.default_rng(seed)
    stratify_key = _build_stratify_key(obs, stratify_cols)
    fold_assign = pd.Series(-1, index=obs.index, dtype=np.int64, name="fold")
    manifest_rows: list[dict] = []

    grouped = obs.assign(_stratify_key=stratify_key).groupby("_stratify_key", sort=True, observed=True)
    for key, frame in grouped:
        idx = frame.index.to_numpy()
        n = len(idx)
        perm = rng.permutation(n)
        assigned_folds = np.arange(n, dtype=np.int64) % n_folds
        shuffled_idx = idx[perm]
        fold_assign.loc[shuffled_idx] = assigned_folds
        counts = np.bincount(assigned_folds, minlength=n_folds)
        row = {"split_group": key, "n_cells": int(n)}
        for col in stratify_cols:
            row[col] = frame[col].iloc[0]
        for fold in range(n_folds):
            row[f"fold_{fold}_cells"] = int(counts[fold])
        manifest_rows.append(row)

    split = pd.Series("train", index=obs.index, dtype="object", name="split")
    split.loc[fold_assign.eq(fold_index)] = "test"

    manifest = pd.DataFrame(manifest_rows).sort_values(list(stratify_cols)).reset_index(drop=True)
    metadata = {
        "split_strategy": "random_kfold",
        "seed": int(seed),
        "stratify_cols": list(stratify_cols),
        "n_folds": int(n_folds),
        "fold_index": int(fold_index),
        "n_train_cells": int((split == "train").sum()),
        "n_test_cells": int((split == "test").sum()),
        "n_split_groups": int(len(manifest)),
    }
    return HNSCCSplitResult(split=split, manifest=manifest, metadata=metadata)


def make_wta_split(
    obs: pd.DataFrame,
    *,
    wta_column: str,
    train_wtas: Sequence[str],
    test_wtas: Sequence[str],
) -> HNSCCSplitResult:
    meta = validate_wta_split(obs, wta_column=wta_column, train_wtas=train_wtas, test_wtas=test_wtas)
    split = pd.Series("train", index=obs.index, dtype="object", name="split")
    split.loc[obs[wta_column].astype(str).isin(set(test_wtas))] = "test"

    manifest = (
        obs.assign(split=split)
        .groupby([wta_column, "Time point", "split"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values([wta_column, "Time point", "split"])
        .reset_index(drop=True)
    )
    metadata = {
        "split_strategy": "wta",
        "wta_column": wta_column,
        "train_wtas": list(meta["train_wtas"]),
        "test_wtas": list(meta["test_wtas"]),
        "available_wtas": list(meta["available_wtas"]),
        "n_train_cells": int((split == "train").sum()),
        "n_test_cells": int((split == "test").sum()),
    }
    return HNSCCSplitResult(split=split, manifest=manifest, metadata=metadata)


def build_study_from_split(
    obs: pd.DataFrame,
    latent: np.ndarray,
    *,
    split: pd.Series,
    split_name: str,
    mass_value_col: str | None = None,
    mass_scope: str = "subset_only",
    mass_mode: str = "auto",
) -> PerturbSeqDynamicsData:
    """Build a split-specific study object.

    Support always comes from the cells assigned to ``split_name``. Mass can be
    sourced either from the split subset only (the default, suitable for held-out
    evaluation where both support and mass are split-local) or from the full
    observation table when a global abundance target is explicitly desired.
    """
    mask = split.eq(split_name).to_numpy()
    sub_obs = obs.loc[mask].copy()
    sub_latent = latent[mask]
    if len(sub_obs) == 0:
        raise ValueError(f"No cells left for split={split_name!r}")
    if mass_scope not in {"full_obs", "subset_only"}:
        raise ValueError("mass_scope must be 'full_obs' or 'subset_only'.")

    cell_df = sub_obs[["cell_id", "perturbation_id", "time_label", "sample_id"]].copy()
    mass_obs = obs if mass_scope == "full_obs" else sub_obs
    mass_source_df = mass_obs[["perturbation_id", "time_label", "sample_id"]].copy()
    mass_group_cols = ["perturbation_id", "time_label", "sample_id"]
    if mass_mode not in {"auto", "count", "group_total", "per_cell_contribution"}:
        raise ValueError("mass_mode must be 'auto', 'count', 'group_total', or 'per_cell_contribution'.")
    if mass_mode == "count":
        mass_df = (
            mass_source_df.groupby(mass_group_cols, observed=True)
            .size()
            .astype(float)
            .rename("mass")
            .reset_index()
        )
        resolved_mass_mode = f"{mass_scope}:count"
    elif mass_value_col is None:
        if mass_mode == "auto":
            mass_df = (
                mass_source_df.groupby(mass_group_cols, observed=True)
                .size()
                .astype(float)
                .rename("mass")
                .reset_index()
            )
            resolved_mass_mode = f"{mass_scope}:count:auto_no_mass_value_col"
        else:
            raise ValueError(f"mass_mode={mass_mode!r} requires mass_value_col.")
    elif mass_value_col not in mass_obs.columns:
        if mass_mode == "auto":
            mass_df = (
                mass_source_df.groupby(mass_group_cols, observed=True)
                .size()
                .astype(float)
                .rename("mass")
                .reset_index()
            )
            resolved_mass_mode = f"{mass_scope}:count:auto_missing_mass_value_col:{mass_value_col}"
        else:
            raise KeyError(f"Requested mass_value_col {mass_value_col!r} not present in obs.")
    else:
        values = pd.to_numeric(mass_obs[mass_value_col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all() or np.any(values.to_numpy() <= 0):
            raise ValueError("mass_value_col must contain positive finite mass values.")
        mass_source_df[mass_value_col] = values.to_numpy()
        group = mass_source_df.groupby(mass_group_cols, observed=True)[mass_value_col]
        constant_groups = group.nunique().le(1)
        multicell_groups = group.size().gt(1)
        ambiguous_constant = bool((constant_groups & multicell_groups).any())
        if mass_mode == "auto":
            if ambiguous_constant:
                raise ValueError(
                    "mass_value_col is constant within at least one multi-cell group. "
                    "Use mass_mode='group_total' for group-level totals or "
                    "mass_mode='per_cell_contribution' for values that should be summed."
                )
            resolved = "per_cell_contribution"
            resolved_mass_mode = f"{mass_scope}:{mass_value_col}:auto_per_cell_contribution"
        else:
            resolved = mass_mode
            resolved_mass_mode = f"{mass_scope}:{mass_value_col}:{resolved}"

        if resolved == "group_total":
            if (~constant_groups).any():
                raise ValueError("mass_mode='group_total' requires exactly one unique mass value per group.")
            mass_df = group.first().rename("mass").reset_index()
        elif resolved == "per_cell_contribution":
            mass_df = group.sum().rename("mass").reset_index()
        else:
            raise ValueError(f"Unsupported mass mode with mass_value_col: {resolved!r}")
    mass_df = mass_df.loc[pd.to_numeric(mass_df["mass"], errors="coerce").fillna(0.0) > 0].reset_index(drop=True)
    mass_df.attrs["mass_mode"] = resolved_mass_mode
    mass_df.attrs["requested_mass_mode"] = mass_mode
    if mass_mode == "auto":
        mass_df.attrs["mass_mode_resolution_reason"] = resolved_mass_mode.split(":auto_", 1)[-1] if ":auto_" in resolved_mass_mode else "auto"
    else:
        mass_df.attrs["mass_mode_resolution_reason"] = f"explicit_{mass_mode}"

    perturbation_ids = sorted(cell_df["perturbation_id"].unique().tolist())
    control_ids = sorted(sub_obs.loc[sub_obs["is_control"].astype(bool), "perturbation_id"].unique().tolist())
    if not control_ids:
        raise ValueError("No control perturbations found after filtering.")

    return PerturbSeqDynamicsData(
        time_axis=TimeAxis.p4_p60(),
        catalog=PerturbationCatalog(perturbation_ids=perturbation_ids, control_ids=control_ids),
        cell_state=CellStateTable(df=cell_df.reset_index(drop=True), latent=sub_latent),
        mass_table=MassTable(df=mass_df),
    )


def supported_intersection(
    train_data: PerturbSeqDynamicsData,
    test_data: PerturbSeqDynamicsData,
    *,
    min_cells_p4: int,
    min_cells_p60: int,
) -> list[str]:
    train_supported = set(
        filter_state_supported_perturbations(
            train_data,
            min_cells_p4=min_cells_p4,
            min_cells_p60=min_cells_p60,
        )
    )
    test_supported = set(
        filter_state_supported_perturbations(
            test_data,
            min_cells_p4=min_cells_p4,
            min_cells_p60=min_cells_p60,
        )
    )
    control_ids = set(train_data.catalog.control_ids) & set(test_data.catalog.control_ids)
    supported = sorted((train_supported & test_supported) | control_ids)
    if not supported:
        raise ValueError("No perturbations have sufficient support in both train and test.")
    return supported


def compute_state_centroids(
    obs: pd.DataFrame,
    latent: np.ndarray,
    *,
    state_key: str = DEFAULT_STATE_KEY,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    states = sorted(obs[state_key].astype(str).unique().tolist())
    rows = []
    centroids = []
    for state in states:
        mask = obs[state_key].astype(str).eq(state).to_numpy()
        state_latent = latent[mask]
        centroids.append(state_latent.mean(axis=0))
        rows.append({"state": state, "n_cells": int(mask.sum())})
    return states, np.vstack(centroids).astype(np.float32), pd.DataFrame(rows)


def build_split_summary(
    obs: pd.DataFrame,
    *,
    split: pd.Series,
    state_key: str | None = None,
) -> pd.DataFrame:
    group_cols = ["split", "Time point", "perturbation_id"]
    sort_cols = list(group_cols)
    if state_key and state_key in obs.columns:
        group_cols.append(state_key)
        sort_cols.append(state_key)
    summary = (
        obs.assign(split=split)
        .groupby(group_cols, observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )
    return summary


# ---------------------------------------------------------------------------
# End-to-end VAE latent pipeline  (Comment 2, 4, 8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAELatentResult:
    """Output of ``build_vae_latent``: latent arrays, artifact bundle, and centroids."""
    latent: np.ndarray                   # [n_cells, latent_dim] — full dataset, standardized
    train_mask: np.ndarray               # [n_cells] bool
    test_mask: np.ndarray                # [n_cells] bool
    program_centroids: np.ndarray | None # [K, latent_dim] — centroids in VAE space, or None
    bundle: object                       # VAEArtifactBundle


def build_vae_latent(
    h5ad_path: str,
    *,
    split: pd.Series,
    obs: pd.DataFrame,
    kept_positions: np.ndarray | None = None,
    latent_dim: int = 16,
    layer: str | None = None,
    use_raw: bool = False,
    gene_mask_col: str | None = None,
    n_genes: int = 2000,
    batch_aware_hvg: bool = True,
    hvg_batch_col: str = DEFAULT_WTA_COLUMN,
    hvg_time_col: str = "Time point",
    hvg_min_cells_per_batch: int = 256,
    allow_full_gene_scan: bool = False,
    target_sum: float = 1e4,
    vae_hidden_dim: int = 512,
    vae_depth: int = 2,
    vae_dropout: float = 0.1,
    vae_epochs: int = 100,
    vae_batch_size: int = 1024,
    vae_lr: float = 1e-3,
    vae_weight_decay: float = 1e-6,
    vae_kl_weight: float = 1e-3,
    vae_kl_warmup_epochs: int = 20,
    vae_val_frac: float = 0.1,
    vae_early_stop_patience: int = 15,
    vae_grad_clip: float = 1.0,
    vae_seed: int = 0,
    encode_batch_size: int = 4096,
    expression_workers: int = 0,
    expression_chunk_size: int = 1024,
    preload_dense_max_gb: float = 4.0,
    reuse_artifact: bool = True,
    vae_use_amp: bool = True,
    vae_amp_dtype: str = "bf16",
    device: str = "cpu",
    state_key: str | None = None,
    compute_centroids: bool = False,
    save_dir: str | None = None,
    commit_sha: str | None = None,
    strict_layer: bool = True,
    strict_counts: bool = True,
    allow_empty_gene_mask_fallback: bool = False,
    max_dense_cache_gb: float | None = None,
    reuse_saved_artifact: bool | None = None,
) -> VAELatentResult:
    """End-to-end VAE latent pipeline: split-safe fitting, encoding, standardization.

    This function ensures the VAE is fit on **training cells only**, encodes
    both train and test with the frozen encoder, z-scores the latent using
    training-set statistics, and optionally saves the full artifact bundle.

    Pipeline steps:

    1. Load raw counts from an explicit count source.
    2. Define the train/test split.
    3. Select genes using **training cells only**.
    4. Library-normalize + log1p using full-library totals.
    5. Fit the VAE on training cells only.
    6. Encode all cells (train + test) with the frozen encoder.
    7. Z-score the latent using training-set mean/std.
    8. Recompute program centroids in VAE latent space only if explicitly requested.
    7. Save the full artifact bundle if *save_dir* is given.

    Parameters
    ----------
    h5ad_path : str
        Path to the h5ad file with raw expression data.
    split : pd.Series
        Cell-level split assignment (index aligned with *obs*), values
        ``"train"`` / ``"test"``.
    obs : pd.DataFrame
        Prepared obs (output of ``prepare_hnscc_obs``).
    kept_positions : np.ndarray | None
        If ``prepare_hnscc_obs`` filtered cells, pass the ``kept_positions``
        array so that expression rows can be aligned with *obs*.
    latent_dim : int
        VAE latent dimensionality.
    layer, use_raw, gene_mask_col, n_genes, target_sum
        Forwarded to ``load_hnscc_expression`` and
        ``log1p_normalize_expression_matrix``.
    save_dir : str | None
        If given, save the VAE artifact bundle to this directory.

    Returns
    -------
    VAELatentResult
        Contains the standardized latent for all cells, masks, optional
        centroids, and the artifact bundle.
    """
    if max_dense_cache_gb is None:
        max_dense_cache_gb = preload_dense_max_gb
    elif float(preload_dense_max_gb) != float(max_dense_cache_gb):
        raise ValueError("Received conflicting preload_dense_max_gb and max_dense_cache_gb values.")

    if reuse_saved_artifact is None:
        reuse_saved_artifact = reuse_artifact
    elif bool(reuse_artifact) != bool(reuse_saved_artifact):
        raise ValueError("Received conflicting reuse_artifact and reuse_saved_artifact values.")

    from ..models.expression_vae import (
        VAEArtifactBundle,
        encode_expression_vae,
        fit_expression_vae,
        log1p_normalize_expression_matrix,
        maybe_materialize_dense_matrix,
        standardize_latent,
    )

    # 1. Load candidate expression matrix from an explicit count source.
    _, expr_candidate, candidate_gene_names, expr_meta = load_hnscc_expression(
        h5ad_path,
        layer=layer,
        use_raw=use_raw,
        gene_mask_col=gene_mask_col,
        top_genes=0,
        validate_counts=True,
        strict_layer=strict_layer,
        strict_counts=strict_counts,
        allow_full_gene_scan=allow_full_gene_scan,
        allow_empty_gene_mask_fallback=allow_empty_gene_mask_fallback,
        row_indices=kept_positions,
        n_workers=expression_workers,
        chunk_size=expression_chunk_size,
    )
    candidate_gene_indices = np.asarray(expr_meta["selected_gene_indices"], dtype=np.int64)
    library_totals = np.asarray(expr_meta["full_library_totals"], dtype=np.float32)

    n_cells = expr_candidate.shape[0]
    if n_cells != len(obs):
        raise ValueError(
            f"Expression matrix has {n_cells} rows but obs has {len(obs)} rows. "
            "Pass kept_positions from prepare_hnscc_obs if cells were filtered."
        )

    # 2. Define the split before any train-sensitive processing.
    train_mask = split.eq("train").to_numpy()
    test_mask = split.eq("test").to_numpy()
    train_indices = np.flatnonzero(train_mask).tolist()
    if not train_mask.any():
        raise ValueError("VAE latent construction requires at least one training cell.")

    # 3. Select genes using training cells only.
    train_candidate = expr_candidate[train_mask]
    if batch_aware_hvg:
        selected_local_idx = _rank_train_hv_genes_batch_aware(
            train_candidate,
            obs.loc[train_mask],
            n_genes=n_genes,
            batch_col=hvg_batch_col,
            time_col=hvg_time_col,
            min_cells_per_batch=hvg_min_cells_per_batch,
        )
    else:
        selected_local_idx = _rank_train_hv_genes(train_candidate, n_genes=n_genes)
    selected_gene_indices = candidate_gene_indices[selected_local_idx]
    gene_names = [candidate_gene_names[int(i)] for i in selected_local_idx.tolist()]
    expr_selected = expr_candidate[:, selected_local_idx]

    # 4. Normalize using full-library totals rather than panel-only totals.
    expr_norm = log1p_normalize_expression_matrix(
        expr_selected,
        target_sum=target_sum,
        library_totals=library_totals,
    )
    expr_norm, dense_cached = maybe_materialize_dense_matrix(
        expr_norm,
        max_gb=max_dense_cache_gb,
    )

    if sp.issparse(expr_norm):
        expr_train = expr_norm[train_mask]
    else:
        expr_train = expr_norm[train_mask]

    vae_hp = {
        "input_dim": int(expr_norm.shape[1]),
        "latent_dim": latent_dim,
        "hidden_dim": vae_hidden_dim,
        "depth": vae_depth,
        "dropout": vae_dropout,
        "batch_aware_hvg": bool(batch_aware_hvg),
        "hvg_batch_col": hvg_batch_col,
        "hvg_time_col": hvg_time_col,
        "hvg_min_cells_per_batch": int(hvg_min_cells_per_batch),
        "dense_cached": bool(dense_cached),
        "max_dense_cache_gb": float(max_dense_cache_gb),
        "allow_empty_gene_mask_fallback": bool(allow_empty_gene_mask_fallback),
        "vae_use_amp": bool(vae_use_amp),
        "vae_amp_dtype": str(vae_amp_dtype),
    }
    split_hash = _split_manifest_hash(obs, split)
    expected_cache = {
        "requested_layer": expr_meta.get("requested_layer"),
        "source_layer": expr_meta.get("layer"),
        "target_sum": target_sum,
        "selected_gene_indices": selected_gene_indices.astype(np.int64).tolist(),
        "train_cell_indices": train_indices,
        "kept_positions": kept_positions.tolist() if kept_positions is not None else None,
        "split_manifest_hash": split_hash,
        "vae_hyperparams": vae_hp,
    }

    if save_dir is not None and reuse_saved_artifact and _vae_cache_matches(save_dir, expected=expected_cache):
        bundle, model = VAEArtifactBundle.load(save_dir, device=device)
        _, _, latent_cache_path = _vae_cache_paths(save_dir)
        if latent_cache_path.exists():
            z_all_std = np.load(latent_cache_path)
        else:
            z_all = encode_expression_vae(
                model,
                expr_norm,
                batch_size=encode_batch_size,
                device=device,
                use_amp=vae_use_amp,
                amp_dtype=vae_amp_dtype,
            )
            z_all_std = bundle.latent_standardization.transform(z_all)
        program_centroids = None
        if compute_centroids:
            if not state_key:
                raise ValueError("compute_centroids=True requires a non-empty state_key.")
            train_obs = obs.loc[train_mask]
            z_train_std = z_all_std[train_mask]
            _, program_centroids, _ = compute_state_centroids(
                train_obs, z_train_std, state_key=state_key,
            )
        return VAELatentResult(
            latent=z_all_std,
            train_mask=train_mask,
            test_mask=test_mask,
            program_centroids=program_centroids,
            bundle=bundle,
        )

    model, history, summary = fit_expression_vae(
        expr_train,
        latent_dim=latent_dim,
        hidden_dim=vae_hidden_dim,
        depth=vae_depth,
        dropout=vae_dropout,
        epochs=vae_epochs,
        batch_size=vae_batch_size,
        learning_rate=vae_lr,
        weight_decay=vae_weight_decay,
        kl_weight=vae_kl_weight,
        kl_warmup_epochs=vae_kl_warmup_epochs,
        val_frac=vae_val_frac,
        early_stop_patience=vae_early_stop_patience,
        grad_clip=vae_grad_clip,
        seed=vae_seed,
        device=device,
        use_amp=vae_use_amp,
        amp_dtype=vae_amp_dtype,
    )

    # 6. Encode ALL cells with frozen encoder
    z_all = encode_expression_vae(
        model,
        expr_norm,
        batch_size=encode_batch_size,
        device=device,
        use_amp=vae_use_amp,
        amp_dtype=vae_amp_dtype,
    )

    # 7. Standardize using training-set statistics only
    z_train = z_all[train_mask]
    _, lat_stats = standardize_latent(z_train)
    z_all_std, _ = standardize_latent(z_all, stats=lat_stats)

    # 8. Recompute centroids in VAE latent space only if explicitly requested.
    program_centroids = None
    if compute_centroids:
        if not state_key:
            raise ValueError("compute_centroids=True requires a non-empty state_key.")
        train_obs = obs.loc[train_mask]
        z_train_std = z_all_std[train_mask]
        _, program_centroids, _ = compute_state_centroids(
            train_obs, z_train_std, state_key=state_key,
        )

    # 7. Build artifact bundle
    bundle = VAEArtifactBundle(
        gene_names=gene_names,
        source_layer=expr_meta.get("layer"),
        requested_layer=expr_meta.get("requested_layer"),
        selected_gene_indices=selected_gene_indices.astype(np.int64).tolist(),
        target_sum=target_sum,
        vae_hyperparams=vae_hp,
        train_cell_indices=train_indices,
        kept_positions=kept_positions.tolist() if kept_positions is not None else None,
        split_manifest_hash=split_hash,
        latent_standardization=lat_stats,
        training_summary=summary,
        commit_sha=commit_sha,
    )

    if save_dir is not None:
        bundle.save(save_dir, model, latent_all_std=z_all_std)
        history.to_csv(str(save_dir) + "/vae_training_history.csv", index=False)

    return VAELatentResult(
        latent=z_all_std,
        train_mask=train_mask,
        test_mask=test_mask,
        program_centroids=program_centroids,
        bundle=bundle,
    )


def build_study_from_vae_latent(
    vae_result: VAELatentResult,
    obs: pd.DataFrame,
    split: pd.Series,
    *,
    mass_value_col: str | None = None,
    mass_scope: str = "subset_only",
    mass_mode: str = "auto",
) -> tuple[PerturbSeqDynamicsData, PerturbSeqDynamicsData]:
    """Build train and test ``PerturbSeqDynamicsData`` from a VAE latent result.

    This is the convenience last-mile function that turns a ``VAELatentResult``
    (from ``build_vae_latent``) into the study objects consumed by the trainer
    and evaluator, so callers never need to manually slice the latent.
    """
    train_data = build_study_from_split(
        obs,
        vae_result.latent,
        split=split,
        split_name="train",
        mass_value_col=mass_value_col,
        mass_scope=mass_scope,
        mass_mode=mass_mode,
    )
    test_data = build_study_from_split(
        obs,
        vae_result.latent,
        split=split,
        split_name="test",
        mass_value_col=mass_value_col,
        mass_scope=mass_scope,
        mass_mode=mass_mode,
    )
    return train_data, test_data
