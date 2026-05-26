#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "src"))

from credo.data.hnscc import prepare_hnscc_obs
from hnscc_biology_common import candidate_gene_keys, load_signature_sets, normalize_gene_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score TNF/TSK/pEMT/CIS-like signatures in the HNSCC Perturb-seq AnnData."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="results/biology/signatures")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--use-raw", action="store_true")
    parser.add_argument("--custom-signatures", default=None)
    parser.add_argument("--group-cols", default="perturbation_id,time_label")
    parser.add_argument("--state-key", default="Cell type annotation")
    parser.add_argument("--include-state-groups", action="store_true")
    parser.add_argument("--min-genes", type=int, default=2)
    parser.add_argument("--log1p", action="store_true")
    parser.add_argument("--write-cell-scores", action="store_true")
    parser.add_argument("--guide-confident-only", dest="guide_confident_only", action="store_true")
    parser.add_argument("--include-nonconfident", dest="guide_confident_only", action="store_false")
    parser.set_defaults(guide_confident_only=True)
    return parser.parse_args()


def _normalize_gene(name: object) -> str:
    return normalize_gene_name(name)


def _as_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return np.asarray(matrix)


def _source_var_names(adata: ad.AnnData, *, use_raw: bool) -> pd.Index:
    if use_raw:
        if adata.raw is None:
            raise ValueError("--use-raw was requested but adata.raw is missing.")
        return pd.Index(adata.raw.var_names)
    return pd.Index(adata.var_names)


def _read_expression(
    adata: ad.AnnData,
    *,
    row_indices: np.ndarray,
    indices: list[int],
    use_raw: bool,
    layer: str | None,
) -> np.ndarray:
    if use_raw:
        matrix = adata.raw.X[row_indices, :][:, indices]
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not present in AnnData.")
        matrix = adata.layers[layer][row_indices, :][:, indices]
    else:
        matrix = adata.X[row_indices, :][:, indices]
    expr = _as_dense(matrix).astype(np.float32, copy=False)
    if expr.ndim != 2:
        raise ValueError(f"Expression slice must be 2D, got {expr.shape}")
    return expr


def _zscore_columns(expr: np.ndarray) -> np.ndarray:
    mean = np.nanmean(expr, axis=0, keepdims=True)
    std = np.nanstd(expr, axis=0, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (expr - mean) / std


def _resolve_signatures(var_names: pd.Index, signatures: dict[str, list[str]], min_genes: int) -> tuple[dict[str, list[int]], pd.DataFrame]:
    lookup: dict[str, int] = {}
    for idx, name in enumerate(var_names):
        lookup.setdefault(_normalize_gene(name), idx)

    resolved: dict[str, list[int]] = {}
    coverage_rows: list[dict] = []
    for sig_name, genes in signatures.items():
        matched = []
        missing = []
        matched_set = set()
        for gene in genes:
            idx = None
            for key in candidate_gene_keys(gene):
                idx = lookup.get(key)
                if idx is not None:
                    break
            if idx is None:
                missing.append(str(gene))
            elif idx not in matched_set:
                matched_set.add(idx)
                matched.append(idx)
            else:
                continue
        if len(matched) >= min_genes:
            resolved[sig_name] = matched
        coverage_rows.append(
            {
                "signature": sig_name,
                "n_requested": len(genes),
                "n_matched": len(matched),
                "matched_genes": ",".join(str(var_names[idx]) for idx in matched),
                "missing_genes": ",".join(missing),
                "used": len(matched) >= min_genes,
            }
        )
    return resolved, pd.DataFrame(coverage_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signatures = load_signature_sets(args.custom_signatures)
    adata = ad.read_h5ad(args.data_path, backed="r")
    try:
        var_names = _source_var_names(adata, use_raw=args.use_raw)
        resolved, coverage = _resolve_signatures(var_names, signatures, args.min_genes)
        coverage.to_csv(output_dir / "signature_gene_coverage.csv", index=False)
        if not resolved:
            raise ValueError("No signatures met the minimum matched-gene threshold.")

        obs, kept_positions = prepare_hnscc_obs(
            adata.obs.copy(),
            guide_confident_only=args.guide_confident_only,
            state_key=args.state_key if args.state_key else None,
        )
        unique_indices = sorted({idx for indices in resolved.values() for idx in indices})
        pos = {idx: j for j, idx in enumerate(unique_indices)}
        expr = _read_expression(
            adata,
            row_indices=kept_positions,
            indices=unique_indices,
            use_raw=args.use_raw,
            layer=args.layer,
        )
        if args.log1p:
            expr = np.log1p(np.maximum(expr, 0.0))
        expr_z = _zscore_columns(expr)

        group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]
        if args.include_state_groups and args.state_key and args.state_key in obs.columns and args.state_key not in group_cols:
            group_cols.append(args.state_key)
        missing = [col for col in group_cols if col not in obs.columns]
        if missing:
            raise KeyError(f"AnnData obs missing grouping columns: {missing}")

        score_df = obs[group_cols].copy()
        for sig_name, indices in resolved.items():
            cols = [pos[idx] for idx in indices]
            score_df[sig_name] = expr_z[:, cols].mean(axis=1)

        value_cols = list(resolved)
        grouped_obj = score_df.groupby(group_cols, observed=True)
        grouped = grouped_obj[value_cols].mean().reset_index()
        med = grouped_obj[value_cols].median().reset_index()
        std = grouped_obj[value_cols].std().reset_index()
        counts = grouped_obj.size().rename("n_cells").reset_index()
        for sig_name in value_cols:
            grouped[f"{sig_name}_median"] = med[sig_name]
            grouped[f"{sig_name}_std"] = std[sig_name]
        grouped = grouped.merge(counts, on=group_cols, how="left")

        if "Time point" in grouped.columns:
            grouped = grouped.rename(columns={"Time point": "time_label"})
        grouped.to_csv(output_dir / "signature_group_scores.csv", index=False)

        if args.write_cell_scores:
            score_df.to_csv(output_dir / "signature_cell_scores.csv", index=False)
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    print(output_dir / "signature_group_scores.csv")


if __name__ == "__main__":
    main()
