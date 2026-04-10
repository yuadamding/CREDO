"""Reusable HNSCC data helpers for CAPE experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import CellStateTable, MassTable, PerturbSeqDynamicsData, PerturbationCatalog, TimeAxis
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
DEFAULT_RANDOM_STRATIFY_COLS = ("Time point", "perturbation_id", DEFAULT_STATE_KEY)


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


def _coerce_gene_mask(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(bool).to_numpy()
    text = values.astype(str).str.lower()
    return text.isin({"1", "true", "t", "yes", "y"}).to_numpy()


def load_hnscc_expression(
    path: str,
    *,
    gene_mask_col: str = "hv_gene",
    top_genes: int = 2000,
) -> tuple[pd.DataFrame, sp.csr_matrix, list[str], dict]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    var = adata.var.copy()

    if gene_mask_col and gene_mask_col in var.columns:
        candidate_mask = _coerce_gene_mask(var[gene_mask_col])
    elif "use_genes" in var.columns:
        candidate_mask = _coerce_gene_mask(var["use_genes"])
    elif "in_original_2500_feature_set" in var.columns:
        candidate_mask = _coerce_gene_mask(var["in_original_2500_feature_set"])
    else:
        candidate_mask = np.ones(len(var), dtype=bool)

    candidate_idx = np.flatnonzero(candidate_mask)
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(len(var), dtype=np.int64)

    if top_genes > 0 and len(candidate_idx) > top_genes:
        if "hv_score" in var.columns:
            scores = pd.to_numeric(var["hv_score"], errors="coerce").fillna(-np.inf).to_numpy()
            ranked = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
            selected_idx = ranked[:top_genes]
        else:
            selected_idx = candidate_idx[:top_genes]
    else:
        selected_idx = candidate_idx

    expr = adata[:, selected_idx].X
    if hasattr(expr, "to_memory"):
        expr = expr.to_memory()
    if sp.issparse(expr):
        expr = expr.tocsr().astype(np.float32)
    else:
        expr = sp.csr_matrix(np.asarray(expr, dtype=np.float32))

    gene_names = [str(name) for name in var.index[selected_idx].tolist()]
    meta = {
        "gene_mask_col": gene_mask_col,
        "top_genes": int(top_genes),
        "n_selected_genes": int(len(gene_names)),
    }
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs, expr, gene_names, meta


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
    state_key: str = DEFAULT_STATE_KEY,
) -> tuple[pd.DataFrame, np.ndarray]:
    keep = np.ones(len(obs), dtype=bool)
    if guide_confident_only and "guide_confident" in obs.columns:
        keep &= obs["guide_confident"].fillna(False).to_numpy(dtype=bool)
    prepared = obs.loc[keep].copy()
    kept_positions = np.flatnonzero(keep)
    prepared["perturbation_id"] = clean_perturbation_ids(prepared)
    prepared["time_label"] = time_labels(prepared)
    prepared["sample_id"] = (
        prepared["Library"].astype(str).replace({"": "pooled", "nan": "pooled", "None": "pooled"})
    )
    prepared["cell_id"] = prepared["cell_id"].astype(str)
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
) -> PerturbSeqDynamicsData:
    mask = split.eq(split_name).to_numpy()
    sub_obs = obs.loc[mask].copy()
    sub_latent = latent[mask]
    if len(sub_obs) == 0:
        raise ValueError(f"No cells left for split={split_name!r}")

    cell_df = sub_obs[["cell_id", "perturbation_id", "time_label", "sample_id"]].copy()
    mass_df = (
        cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .rename("mass")
        .reset_index()
    )

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
    state_key: str = DEFAULT_STATE_KEY,
) -> pd.DataFrame:
    summary = (
        obs.assign(split=split)
        .groupby(["split", "Time point", "perturbation_id", state_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["split", "Time point", "perturbation_id", state_key])
        .reset_index(drop=True)
    )
    return summary
