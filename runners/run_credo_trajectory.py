"""Generic multi-time trajectory runner for CREDO.

This runner consumes an AnnData file with trajectory-ready metadata and trains
the first production trajectory stack: full-start rollout from one source time
to all downstream observed checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parent.parent / "package"
sys.path.insert(0, str(ROOT / "src"))

from credo.config.schema import RunConfig
from credo.data.core import (
    CellStateTable,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
    POOLED_SAMPLE_ID,
)
from credo.models.full_model import FullDynamicsModel
from credo.models.expression_vae import (
    VAEArtifactBundle,
    encode_expression_vae,
    fit_expression_vae,
    log1p_normalize_expression_matrix,
    standardize_latent,
)
from credo.training.trajectory_trainer import TrajectoryTrainer


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_label_float_map(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not text:
        return out
    for item in _parse_csv(text):
        if ":" not in item:
            raise ValueError(f"Expected label:value entry, got {item!r}")
        label, value = item.split(":", 1)
        out[label.strip()] = float(value)
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float) != 0.0
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "t", "yes", "y", "control", "ctrl"})


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except Exception:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CREDO multi-time trajectory training.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-col", default="time_label")
    parser.add_argument("--physical-time-col", default="physical_time")
    parser.add_argument("--perturbation-col", default="perturbation_id")
    parser.add_argument("--sample-col", default="sample_id")
    parser.add_argument("--control-col", default="is_control")
    parser.add_argument("--mass-col", default="mass_value")
    parser.add_argument(
        "--mass-mode",
        choices=["auto", "count", "per_cell_contribution", "group_total"],
        default="auto",
        help=(
            "How to construct finite-measure masses. 'count' ignores --mass-col "
            "and uses captured cell counts; 'per_cell_contribution' sums --mass-col "
            "within each perturbation/time/sample group; 'group_total' requires "
            "--mass-col to be constant within each group and uses that value once. "
            "'auto' refuses ambiguous constant group masses."
        ),
    )
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--source-label", default="90m")
    parser.add_argument("--target-labels", default="6h,10h")
    parser.add_argument("--physical-times", default="")
    parser.add_argument("--key-mode", choices=["pooled", "sample_aware"], default="sample_aware")
    parser.add_argument("--sparse-missing", choices=["mask", "error"], default="mask")
    parser.add_argument("--latent-source", choices=["pca", "obsm", "vae"], default="pca")
    parser.add_argument("--latent-key", default="X_pca")
    parser.add_argument("--vae-layer", default="counts")
    parser.add_argument("--vae-latent-dim", type=int, default=32)
    parser.add_argument("--vae-hidden-dim", type=int, default=512)
    parser.add_argument("--vae-depth", type=int, default=2)
    parser.add_argument("--vae-dropout", type=float, default=0.1)
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--vae-batch-size", type=int, default=1024)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-weight-decay", type=float, default=1e-6)
    parser.add_argument("--vae-kl-weight", type=float, default=1e-3)
    parser.add_argument("--vae-kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--vae-val-frac", type=float, default=0.1)
    parser.add_argument("--vae-early-stop-patience", type=int, default=15)
    parser.add_argument("--vae-grad-clip", type=float, default=1.0)
    parser.add_argument("--vae-use-amp", action="store_true", default=True)
    parser.add_argument("--no-vae-use-amp", dest="vae_use_amp", action="store_false")
    parser.add_argument("--vae-amp-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--vae-fit-source-only", dest="vae_fit_source_only", action="store_true")
    parser.add_argument("--vae-fit-all-cells", dest="vae_fit_source_only", action="store_false")
    parser.set_defaults(vae_fit_source_only=True)
    parser.add_argument("--expression-gene-mask-col", default="hv_gene")
    parser.add_argument("--expression-gene-rank-col", default="")
    parser.add_argument("--expression-gene-score-col", default="")
    parser.add_argument("--expression-top-genes", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--n-particles", type=int, default=128)
    parser.add_argument("--eval-particles", type=int, default=384)
    parser.add_argument("--steps-per-interval", type=int, default=12)
    parser.add_argument("--endpoint-time-weights", default="")
    parser.add_argument("--normalize-time-weights", action="store_true", default=True)
    parser.add_argument("--no-normalize-time-weights", dest="normalize_time_weights", action="store_false")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--mediator-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--control-mode", choices=["anchored", "free", "soft_ref"], default="soft_ref")
    parser.add_argument("--lambda-weak", type=float, default=0.1)
    parser.add_argument("--lambda-count", type=float, default=0.0)
    parser.add_argument("--lambda-reg-net", type=float, default=1e-4)
    parser.add_argument("--lambda-reg-diffusion", type=float, default=1e-4)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-max-iter", type=int, default=100)
    parser.add_argument("--n-test-functions", type=int, default=12)
    parser.add_argument("--ecology-on", dest="ecological_growth", action="store_true")
    parser.add_argument("--ecology-off", dest="ecological_growth", action="store_false")
    parser.set_defaults(ecological_growth=True)
    parser.add_argument("--growth-intercept-on", dest="use_growth_intercept", action="store_true")
    parser.add_argument("--growth-intercept-off", dest="use_growth_intercept", action="store_false")
    parser.set_defaults(use_growth_intercept=True)
    return parser.parse_args(argv)


def _load_obsm_latent(adata: ad.AnnData, latent_key: str) -> np.ndarray:
    if latent_key not in adata.obsm:
        raise KeyError(f"AnnData missing obsm[{latent_key!r}].")
    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError(f"obsm[{latent_key!r}] must be a 2D latent matrix.")
    return latent


def _matrix_column_scores(matrix: object, selection_mask: np.ndarray | None = None) -> np.ndarray:
    if selection_mask is not None:
        matrix = matrix[selection_mask, :]
    if sp.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    else:
        arr = np.asarray(matrix)
        mean = arr.mean(axis=0)
        second = (arr ** 2).mean(axis=0)
    var = np.maximum(second - mean ** 2, 0.0)
    return np.log1p(var / np.maximum(mean, 1e-8))


def _sha256_bool_array(mask: np.ndarray) -> str:
    arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _column_mask_for_vae(
    adata: ad.AnnData,
    args: argparse.Namespace,
    *,
    selection_mask: np.ndarray | None = None,
) -> np.ndarray:
    n_top = int(args.expression_top_genes)
    if selection_mask is not None:
        selection_mask = np.asarray(selection_mask, dtype=bool)
        if selection_mask.shape != (adata.n_obs,):
            raise ValueError("selection_mask must have shape [adata.n_obs].")
        if not selection_mask.any():
            raise ValueError("selection_mask for VAE gene selection is empty.")
    if args.expression_gene_mask_col and args.expression_gene_mask_col in adata.var:
        mask = np.asarray(adata.var[args.expression_gene_mask_col]).astype(bool)
        if mask.any():
            if n_top > 0 and int(mask.sum()) > n_top:
                masked_idx = np.flatnonzero(mask)
                if args.expression_gene_rank_col and args.expression_gene_rank_col in adata.var:
                    rank = pd.to_numeric(adata.var[args.expression_gene_rank_col], errors="coerce").to_numpy()
                    order = np.argsort(np.where(np.isfinite(rank[masked_idx]), rank[masked_idx], np.inf))
                    keep = masked_idx[order[:n_top]]
                elif args.expression_gene_score_col and args.expression_gene_score_col in adata.var:
                    score = pd.to_numeric(adata.var[args.expression_gene_score_col], errors="coerce").to_numpy()
                    order = np.argsort(np.where(np.isfinite(score[masked_idx]), score[masked_idx], -np.inf))[::-1]
                    keep = masked_idx[order[:n_top]]
                else:
                    matrix = adata.layers[args.vae_layer] if args.vae_layer in adata.layers else adata.X
                    score = _matrix_column_scores(matrix, selection_mask=selection_mask)
                    order = np.argsort(score[masked_idx])[::-1]
                    keep = masked_idx[order[:n_top]]
                capped = np.zeros(adata.n_vars, dtype=bool)
                capped[keep] = True
                return capped
            return mask
    if n_top <= 0 or n_top >= adata.n_vars:
        return np.ones(adata.n_vars, dtype=bool)
    matrix = adata.layers[args.vae_layer] if args.vae_layer in adata.layers else adata.X
    score = _matrix_column_scores(matrix, selection_mask=selection_mask)
    keep = np.argsort(score)[::-1][:n_top]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[keep] = True
    return mask


def _load_latent(
    adata: ad.AnnData,
    row_mask: np.ndarray,
    args: argparse.Namespace,
    *,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    if args.latent_source in {"pca", "obsm"}:
        return _load_obsm_latent(adata, args.latent_key)[row_mask]
    if args.latent_source != "vae":
        raise ValueError(f"Unknown latent source {args.latent_source!r}")
    matrix = adata.layers[args.vae_layer] if args.vae_layer in adata.layers else adata.X
    if fit_mask is None:
        fit_mask = row_mask
    if fit_mask.shape != row_mask.shape:
        raise ValueError("fit_mask must have the same length as row_mask.")
    if not fit_mask.any():
        raise ValueError("VAE fit mask is empty.")
    selection_mask = fit_mask if args.vae_fit_source_only else row_mask
    gene_mask = _column_mask_for_vae(adata, args, selection_mask=selection_mask)
    matrix_all = matrix[row_mask, :][:, gene_mask]
    matrix_all = log1p_normalize_expression_matrix(matrix_all)
    matrix_fit = matrix[fit_mask, :][:, gene_mask]
    matrix_fit = log1p_normalize_expression_matrix(matrix_fit)
    device = "cuda" if args.device == "auto" else args.device
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    model, history, _summary = fit_expression_vae(
        matrix_fit,
        latent_dim=args.vae_latent_dim,
        hidden_dim=args.vae_hidden_dim,
        depth=args.vae_depth,
        dropout=args.vae_dropout,
        epochs=args.vae_epochs,
        batch_size=args.vae_batch_size,
        learning_rate=args.vae_lr,
        weight_decay=args.vae_weight_decay,
        kl_weight=args.vae_kl_weight,
        kl_warmup_epochs=args.vae_kl_warmup_epochs,
        val_frac=args.vae_val_frac,
        early_stop_patience=args.vae_early_stop_patience,
        grad_clip=args.vae_grad_clip,
        seed=args.seed,
        device=device,
        use_amp=args.vae_use_amp,
        amp_dtype=args.vae_amp_dtype,
    )
    latent = encode_expression_vae(
        model,
        matrix_all,
        batch_size=args.vae_batch_size,
        device=device,
        use_amp=args.vae_use_amp,
        amp_dtype=args.vae_amp_dtype,
    )
    fit_latent = encode_expression_vae(
        model,
        matrix_fit,
        batch_size=args.vae_batch_size,
        device=device,
        use_amp=args.vae_use_amp,
        amp_dtype=args.vae_amp_dtype,
    )
    _, stats = standardize_latent(fit_latent)
    latent_std = stats.transform(latent).astype(np.float32, copy=False)

    artifact_dir = Path(args.output_dir) / "vae_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selected_gene_indices = np.flatnonzero(gene_mask).astype(int).tolist()
    gene_names = [str(adata.var_names[i]) for i in selected_gene_indices]
    bundle = VAEArtifactBundle(
        gene_names=gene_names,
        source_layer=args.vae_layer if args.vae_layer in adata.layers else None,
        requested_layer=args.vae_layer,
        target_sum=1e4,
        vae_hyperparams={
            "input_dim": int(matrix_fit.shape[1]),
            "latent_dim": int(args.vae_latent_dim),
            "hidden_dim": int(args.vae_hidden_dim),
            "depth": int(args.vae_depth),
            "dropout": float(args.vae_dropout),
            "epochs": int(args.vae_epochs),
            "batch_size": int(args.vae_batch_size),
            "learning_rate": float(args.vae_lr),
            "weight_decay": float(args.vae_weight_decay),
            "kl_weight": float(args.vae_kl_weight),
            "kl_warmup_epochs": int(args.vae_kl_warmup_epochs),
            "gene_selection_method": (
                "rank_col"
                if args.expression_gene_rank_col and args.expression_gene_rank_col in adata.var
                else "score_col"
                if args.expression_gene_score_col and args.expression_gene_score_col in adata.var
                else "fallback_dispersion"
            ),
            "gene_selection_scope": "source_only" if args.vae_fit_source_only else "requested_cells",
            "gene_rank_col": str(args.expression_gene_rank_col or ""),
            "gene_score_col": str(args.expression_gene_score_col or ""),
            "requested_row_mask_sha256": _sha256_bool_array(row_mask),
            "vae_fit_mask_sha256": _sha256_bool_array(fit_mask),
            "gene_selection_mask_sha256": _sha256_bool_array(selection_mask),
        },
        train_cell_indices=np.flatnonzero(fit_mask).astype(int).tolist(),
        latent_standardization=stats,
        training_summary=_summary,
        selected_gene_indices=selected_gene_indices,
        commit_sha=_git_sha(),
    )
    bundle.save(artifact_dir, model, latent_all_std=latent_std)
    history.to_csv(artifact_dir / "vae_history.csv", index=False)
    np.save(artifact_dir / "vae_gene_mask.npy", gene_mask.astype(bool))
    (artifact_dir / "vae_gene_names.txt").write_text("\n".join(gene_names) + "\n", encoding="utf-8")
    return latent_std


def _physical_times(obs: pd.DataFrame, labels: list[str], args: argparse.Namespace) -> list[float]:
    mapping = _parse_label_float_map(args.physical_times)
    if not mapping:
        if args.physical_time_col not in obs.columns:
            raise KeyError(
                "Provide --physical-times label:value,... or an obs physical time column."
            )
        for label in labels:
            values = obs.loc[obs[args.time_col].astype(str).eq(label), args.physical_time_col].astype(float).unique()
            if len(values) != 1:
                raise ValueError(f"Expected exactly one physical time for {label!r}, got {values!r}")
            mapping[label] = float(values[0])
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise KeyError(f"Missing physical times for labels: {missing}")
    return [float(mapping[label]) for label in labels]


def build_study_from_anndata(args: argparse.Namespace) -> PerturbSeqDynamicsData:
    adata = ad.read_h5ad(args.data_path)
    args.adata_n_obs = int(adata.n_obs)
    args.adata_n_vars = int(adata.n_vars)
    args.adata_obs_columns = [str(col) for col in adata.obs.columns]
    args.adata_var_names_sha256 = _sha256_text_lines([str(name) for name in adata.var_names])
    obs_all = adata.obs.copy()
    target_labels = _parse_csv(args.target_labels)
    labels = [args.source_label] + target_labels

    for col in [args.time_col, args.perturbation_col, args.control_col]:
        if col not in obs_all.columns:
            raise KeyError(f"AnnData obs is missing required column {col!r}.")
    if args.key_mode == "sample_aware" and args.sample_col not in obs_all.columns:
        raise KeyError("--key-mode sample_aware requires a sample column.")

    row_mask = obs_all[args.time_col].astype(str).isin(labels).to_numpy()
    if not row_mask.any():
        raise ValueError("No cells match the requested source/target labels.")
    fit_mask = row_mask
    if args.latent_source == "vae" and args.vae_fit_source_only:
        fit_mask = row_mask & obs_all[args.time_col].astype(str).eq(str(args.source_label)).to_numpy()
    obs = obs_all.loc[row_mask].copy()
    latent = _load_latent(adata, row_mask, args, fit_mask=fit_mask)

    time_values = obs[args.time_col].astype(str)
    obs["time_label"] = time_values
    obs["perturbation_id"] = obs[args.perturbation_col].astype(str)
    if args.key_mode == "sample_aware":
        obs["sample_id"] = obs[args.sample_col].astype(str)
    else:
        obs["sample_id"] = POOLED_SAMPLE_ID
    if args.cell_id_col in obs.columns:
        obs["cell_id"] = obs[args.cell_id_col].astype(str)
    else:
        obs["cell_id"] = obs.index.astype(str)

    cell_df = obs[["cell_id", "perturbation_id", "time_label", "sample_id"]].reset_index(drop=True)
    mass_group_cols = ["perturbation_id", "time_label", "sample_id"]
    mass_col_present = bool(args.mass_col and args.mass_col in obs.columns)
    if args.mass_mode == "count":
        resolved_mass_mode = "count"
        mass_mode_reason = "explicit_count_mode"
        mass_df = (
            obs.groupby(mass_group_cols, observed=True)
            .size()
            .astype(float)
            .rename("mass")
            .reset_index()
        )
    elif not args.mass_col:
        if args.mass_mode == "auto":
            resolved_mass_mode = "count"
            mass_mode_reason = "auto_no_mass_column_requested"
            mass_df = (
                obs.groupby(mass_group_cols, observed=True)
                .size()
                .astype(float)
                .rename("mass")
                .reset_index()
            )
        else:
            raise ValueError(f"--mass-mode {args.mass_mode} requires --mass-col.")
    elif not mass_col_present:
        if args.mass_mode == "auto":
            resolved_mass_mode = "count"
            mass_mode_reason = f"auto_missing_mass_column:{args.mass_col}"
            mass_df = (
                obs.groupby(mass_group_cols, observed=True)
                .size()
                .astype(float)
                .rename("mass")
                .reset_index()
            )
        else:
            raise KeyError(f"--mass-col {args.mass_col!r} not found in AnnData obs.")
    else:
        mass_values = obs[args.mass_col].astype(float)
        if not np.isfinite(mass_values.to_numpy()).all() or np.any(mass_values.to_numpy() <= 0):
            raise ValueError("--mass-col must contain positive finite mass values.")
        group = obs.assign(_mass=mass_values).groupby(mass_group_cols, observed=True)["_mass"]
        constant_groups = group.nunique().le(1)
        multicell_groups = group.size().gt(1)
        ambiguous_constant = bool((constant_groups & multicell_groups).any())
        if args.mass_mode == "auto":
            if ambiguous_constant:
                raise ValueError(
                    "--mass-col is constant within at least one multi-cell group. Specify "
                    "--mass-mode group_total if values are group-level totals, or "
                    "--mass-mode per_cell_contribution if they should be summed."
                )
            resolved_mass_mode = "per_cell_contribution"
            mass_mode_reason = "auto_nonconstant_mass_values"
        else:
            resolved_mass_mode = args.mass_mode
            mass_mode_reason = f"explicit_{resolved_mass_mode}"

        if resolved_mass_mode == "group_total":
            bad = constant_groups[~constant_groups]
            if len(bad) > 0:
                raise ValueError("--mass-mode group_total requires exactly one unique mass value per group.")
            mass_df = group.first().rename("mass").reset_index()
        elif resolved_mass_mode == "per_cell_contribution":
            mass_df = group.sum().rename("mass").reset_index()
        else:
            raise ValueError(f"Unsupported mass mode for --mass-col: {resolved_mass_mode!r}")
    args.resolved_mass_mode = resolved_mass_mode
    args.mass_mode_resolution_reason = mass_mode_reason

    pids = sorted(obs["perturbation_id"].unique().tolist())
    control_mask = _as_bool(obs[args.control_col])
    controls = sorted(obs.loc[control_mask, "perturbation_id"].unique().tolist())
    if not controls:
        raise ValueError("At least one control perturbation is required.")

    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=_physical_times(obs, labels, args)),
        catalog=PerturbationCatalog(pids, controls),
        cell_state=CellStateTable(cell_df, latent),
        mass_table=MassTable(mass_df),
    )


def build_config(args: argparse.Namespace, latent_dim: int) -> RunConfig:
    cfg = RunConfig(output_dir=args.output_dir, device=args.device)
    cfg.git_sha = _git_sha()
    cfg.data.mass_value_col = args.mass_col
    cfg.data.mass_mode = getattr(args, "resolved_mass_mode", args.mass_mode)
    cfg.latent.source = "vae" if args.latent_source == "vae" else "pca"
    cfg.latent.key = "X_vae" if args.latent_source == "vae" else args.latent_key
    cfg.latent.dim = latent_dim
    cfg.latent.vae.layer = args.vae_layer
    cfg.latent.vae.n_genes = args.expression_top_genes
    cfg.latent.vae.hidden_dim = args.vae_hidden_dim
    cfg.latent.vae.depth = args.vae_depth
    cfg.latent.vae.dropout = args.vae_dropout
    cfg.latent.vae.epochs = args.vae_epochs
    cfg.latent.vae.batch_size = args.vae_batch_size
    cfg.latent.vae.reuse_artifact = False
    cfg.model.embedding_dim = args.embedding_dim
    cfg.model.n_programs = args.n_programs
    cfg.model.mediator_dim = args.mediator_dim
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.depth = args.depth
    cfg.model.control_mode = args.control_mode
    cfg.model.ecological_growth = args.ecological_growth
    cfg.model.use_growth_intercept = args.use_growth_intercept
    cfg.simulation.n_particles = args.n_particles
    cfg.eval.n_eval_particles = args.eval_particles
    cfg.training.epochs = args.epochs
    cfg.training.seed = args.seed
    cfg.training.precision = args.precision
    cfg.training.lambda_weak = args.lambda_weak
    cfg.training.lambda_count = args.lambda_count
    cfg.training.lambda_reg_net = args.lambda_reg_net
    cfg.training.lambda_reg_diffusion = args.lambda_reg_diffusion
    cfg.training.sinkhorn_epsilon = args.sinkhorn_epsilon
    cfg.training.sinkhorn_max_iter = args.sinkhorn_max_iter
    cfg.training.n_test_functions = args.n_test_functions
    cfg.trajectory_training.source_label = args.source_label
    cfg.trajectory_training.target_labels = _parse_csv(args.target_labels)
    cfg.trajectory_training.steps_per_interval = args.steps_per_interval
    cfg.trajectory_training.endpoint_time_weights = _parse_label_float_map(args.endpoint_time_weights)
    cfg.trajectory_training.normalize_time_weights = bool(args.normalize_time_weights)
    cfg.trajectory_training.key_mode = args.key_mode
    cfg.trajectory_training.sparse_missing = args.sparse_missing
    return cfg


def _sha256_text_lines(lines: list[str]) -> str:
    import hashlib

    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    names = ["anndata", "numpy", "pandas", "scipy", "torch"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_run_manifest(args: argparse.Namespace, output_dir: str | Path) -> None:
    from credo import __version__ as credo_version

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package_version": credo_version,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dependency_versions": _dependency_versions(),
        "resolved_mass_mode": getattr(args, "resolved_mass_mode", None),
        "mass_mode_resolution_reason": getattr(args, "mass_mode_resolution_reason", None),
        "adata_n_obs": getattr(args, "adata_n_obs", None),
        "adata_n_vars": getattr(args, "adata_n_vars", None),
        "adata_obs_columns": getattr(args, "adata_obs_columns", None),
        "adata_var_names_sha256": getattr(args, "adata_var_names_sha256", None),
        "command": " ".join(sys.argv),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_input_manifests(study: PerturbSeqDynamicsData, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    study.mass_table.df.to_csv(out / "mass_table.csv", index=False)
    cell_counts = (
        study.cell_state.df
        .groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cell_counts.to_csv(out / "cell_count_table.csv", index=False)
    mass_summary = (
        study.mass_table.df
        .groupby(["time_label", "sample_id"], observed=True)["mass"]
        .sum()
        .rename("total_mass")
        .reset_index()
    )
    mass_summary.to_csv(out / "mass_summary_by_time_sample.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    study = build_study_from_anndata(args)
    write_run_manifest(args, args.output_dir)
    write_input_manifests(study, args.output_dir)
    labels = [args.source_label] + _parse_csv(args.target_labels)
    by_sample = args.key_mode == "sample_aware"
    trajectory = study.to_sparse_trajectory_problem(by_sample=by_sample, time_labels=labels)
    cfg = build_config(args, latent_dim=study.latent_dim)

    model = FullDynamicsModel(
        perturbation_ids=trajectory.perturbation_ids,
        control_ids=[pid for pid in study.catalog.control_ids if pid in trajectory.perturbation_ids],
        latent_dim=study.latent_dim,
        embedding_dim=cfg.model.embedding_dim,
        n_programs=cfg.model.n_programs,
        mediator_dim=cfg.model.mediator_dim,
        hidden_dim=cfg.model.hidden_dim,
        depth=cfg.model.depth,
        ecological_growth=cfg.model.ecological_growth,
        use_growth_intercept=cfg.model.use_growth_intercept,
        control_mode=cfg.model.control_mode,
    )
    trainer = TrajectoryTrainer(
        model=model,
        config=cfg,
        trajectory=trajectory,
        source_label=args.source_label,
        target_labels=_parse_csv(args.target_labels),
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
