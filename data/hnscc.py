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


def _validate_count_matrix(matrix: sp.spmatrix | np.ndarray, name: str = "expression") -> None:
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
        warnings.warn(
            f"{name} matrix has only {frac_integer:.0%} near-integer values in a sample "
            f"of {len(sample)} entries. Expected raw counts — double-check the source layer.",
            stacklevel=3,
        )


def load_hnscc_expression(
    path: str,
    *,
    layer: str | None = "counts",
    use_raw: bool = True,
    gene_mask_col: str = "hv_gene",
    top_genes: int = 2000,
    validate_counts: bool = True,
) -> tuple[pd.DataFrame, sp.csr_matrix, list[str], dict]:
    """Load expression matrix from an h5ad file.

    Parameters
    ----------
    path : str
        Path to h5ad file.
    layer : str | None
        AnnData layer to read. ``"counts"`` reads ``adata.layers["counts"]``;
        ``None`` reads ``adata.X``.  If the requested layer does not exist,
        falls back to ``adata.raw.X`` (when *use_raw* is True) then ``adata.X``.
    use_raw : bool
        If *layer* is missing and *use_raw* is True, try ``adata.raw.X``.
    gene_mask_col : str
        Column in ``adata.var`` used as a boolean gene mask.
    top_genes : int
        Maximum number of genes to select.
    validate_counts : bool
        If True, check that the loaded matrix is nonneg and integer-like.
    """
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

    # --- Resolve expression source ---
    resolved_layer = None
    if layer is not None and layer in adata.layers:
        expr = adata.layers[layer][:, selected_idx]
        resolved_layer = layer
    elif use_raw and adata.raw is not None:
        expr = adata.raw[:, var.index[selected_idx]].X
        resolved_layer = "raw"
    else:
        expr = adata[:, selected_idx].X
        resolved_layer = "X"

    if hasattr(expr, "to_memory"):
        expr = expr.to_memory()
    if sp.issparse(expr):
        expr = expr.tocsr().astype(np.float32)
    else:
        expr = sp.csr_matrix(np.asarray(expr, dtype=np.float32))

    if validate_counts:
        _validate_count_matrix(expr, name=f"expression (layer={resolved_layer!r})")

    gene_names = [str(name) for name in var.index[selected_idx].tolist()]
    meta = {
        "gene_mask_col": gene_mask_col,
        "top_genes": int(top_genes),
        "n_selected_genes": int(len(gene_names)),
        "layer": resolved_layer,
        "gene_names": gene_names,
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
    layer: str | None = "counts",
    use_raw: bool = True,
    gene_mask_col: str = "hv_gene",
    n_genes: int = 2000,
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
    device: str = "cpu",
    state_key: str = DEFAULT_STATE_KEY,
    compute_centroids: bool = True,
    save_dir: str | None = None,
    commit_sha: str | None = None,
) -> VAELatentResult:
    """End-to-end VAE latent pipeline: split-safe fitting, encoding, standardization.

    This function ensures the VAE is fit on **training cells only**, encodes
    both train and test with the frozen encoder, z-scores the latent using
    training-set statistics, and optionally saves the full artifact bundle.

    Pipeline steps:

    1. Load raw counts with gene selection (``load_hnscc_expression``).
    2. Library-normalize + log1p the raw counts.
    3. Subset to training cells; fit VAE on training set only.
    4. Encode all cells (train + test) with the frozen encoder.
    5. Z-score the latent using training-set mean/std.
    6. Recompute program centroids in VAE latent space (training cells).
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
    from ..models.expression_vae import (
        VAEArtifactBundle,
        LatentStandardization,
        encode_expression_vae,
        fit_expression_vae,
        log1p_normalize_expression_matrix,
        standardize_latent,
    )

    # 1. Load expression with explicit layer and count validation
    _, expr_raw, gene_names, expr_meta = load_hnscc_expression(
        h5ad_path,
        layer=layer,
        use_raw=use_raw,
        gene_mask_col=gene_mask_col,
        top_genes=n_genes,
        validate_counts=True,
    )

    # Align rows if prepare_hnscc_obs filtered some cells
    if kept_positions is not None:
        expr_raw = expr_raw[kept_positions]

    n_cells = expr_raw.shape[0]
    if n_cells != len(obs):
        raise ValueError(
            f"Expression matrix has {n_cells} rows but obs has {len(obs)} rows. "
            "Pass kept_positions from prepare_hnscc_obs if cells were filtered."
        )

    # 2. Normalize: library-size + log1p
    expr_norm = log1p_normalize_expression_matrix(expr_raw, target_sum=target_sum)

    # 3. Split-safe fitting: VAE sees ONLY training cells
    train_mask = split.eq("train").to_numpy()
    test_mask = split.eq("test").to_numpy()
    train_indices = np.flatnonzero(train_mask).tolist()

    if sp.issparse(expr_norm):
        expr_train = expr_norm[train_mask]
    else:
        expr_train = expr_norm[train_mask]

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
    )

    # 4. Encode ALL cells with frozen encoder
    z_all = encode_expression_vae(model, expr_norm, device=device)

    # 5. Standardize using training-set statistics only
    z_train = z_all[train_mask]
    _, lat_stats = standardize_latent(z_train)
    z_all_std, _ = standardize_latent(z_all, stats=lat_stats)

    # 6. Recompute centroids in VAE latent space (training cells only)
    program_centroids = None
    if compute_centroids:
        train_obs = obs.loc[train_mask]
        z_train_std = z_all_std[train_mask]
        _, program_centroids, _ = compute_state_centroids(
            train_obs, z_train_std, state_key=state_key,
        )

    # 7. Build artifact bundle
    vae_hp = {
        "input_dim": int(expr_norm.shape[1]),
        "latent_dim": latent_dim,
        "hidden_dim": vae_hidden_dim,
        "depth": vae_depth,
        "dropout": vae_dropout,
    }
    bundle = VAEArtifactBundle(
        gene_names=gene_names,
        source_layer=expr_meta.get("layer"),
        target_sum=target_sum,
        vae_hyperparams=vae_hp,
        train_cell_indices=train_indices,
        latent_standardization=lat_stats,
        training_summary=summary,
        commit_sha=commit_sha,
    )

    if save_dir is not None:
        bundle.save(save_dir, model)
        history.to_csv(str(save_dir) + "/vae_training_history.csv", index=False)

    return VAELatentResult(
        latent=z_all_std,
        train_mask=train_mask,
        test_mask=test_mask,
        program_centroids=program_centroids,
        bundle=bundle,
    )
