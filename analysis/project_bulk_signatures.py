#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hnscc_biology_common import candidate_gene_keys, load_signature_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project CREDO/Renz-style signatures onto a bulk RNA matrix such as GSE227919."
    )
    parser.add_argument("--expression", required=True, help="CSV/TSV expression matrix.")
    parser.add_argument("--metadata", required=True, help="CSV/TSV sample metadata.")
    parser.add_argument("--output-dir", default="results/biology/human_projection")
    parser.add_argument("--custom-signatures", default=None)
    parser.add_argument("--gene-column", default=None, help="Gene column if genes are not the matrix index.")
    parser.add_argument("--sample-column", default="sample_id")
    parser.add_argument("--stage-column", default="stage")
    parser.add_argument("--stage-order", default="Control,HkNR,Dysplasia,OSCC")
    parser.add_argument("--min-genes", type=int, default=2)
    parser.add_argument("--sep", default=None, help="Override delimiter; defaults to auto-detect.")
    return parser.parse_args()


def _read_table(path: str | Path, sep: str | None) -> pd.DataFrame:
    if sep is None:
        suffix = Path(path).suffix.lower()
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _load_expression(path: str | Path, *, gene_column: str | None, sep: str | None) -> pd.DataFrame:
    df = _read_table(path, sep)
    if gene_column is not None:
        if gene_column not in df.columns:
            raise KeyError(f"Gene column {gene_column!r} not present.")
        df = df.set_index(gene_column)
    else:
        first_col = df.columns[0]
        if not np.issubdtype(df[first_col].dtype, np.number):
            df = df.set_index(first_col)
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def _score_signature(expr_z: pd.DataFrame, genes: list[str], min_genes: int) -> tuple[pd.Series | None, list[str], list[str]]:
    lookup = {gene.upper(): gene for gene in expr_z.index.astype(str)}
    matched = []
    missing = []
    seen = set()
    for gene in genes:
        hit = None
        for key in candidate_gene_keys(gene):
            if key in lookup:
                hit = lookup[key]
                break
        if hit is None:
            missing.append(gene)
        elif hit not in seen:
            seen.add(hit)
            matched.append(hit)
    if len(matched) < min_genes:
        return None, matched, missing
    return expr_z.loc[matched].mean(axis=0), matched, missing


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expr = _load_expression(args.expression, gene_column=args.gene_column, sep=args.sep)
    meta = _read_table(args.metadata, args.sep)
    if args.sample_column not in meta.columns:
        raise KeyError(f"Metadata missing sample column {args.sample_column!r}.")
    if args.stage_column not in meta.columns:
        raise KeyError(f"Metadata missing stage column {args.stage_column!r}.")

    sample_cols = [col for col in expr.columns if col in set(meta[args.sample_column].astype(str))]
    if not sample_cols:
        raise ValueError("No expression columns matched metadata sample ids.")
    expr = expr[sample_cols]
    mean = expr.mean(axis=1)
    std = expr.std(axis=1, ddof=0).replace(0.0, 1.0)
    expr_z = expr.sub(mean, axis=0).div(std, axis=0)

    signatures = load_signature_sets(args.custom_signatures)
    scores = pd.DataFrame({args.sample_column: sample_cols})
    coverage_rows = []
    for name, genes in signatures.items():
        score, matched, missing = _score_signature(expr_z, genes, args.min_genes)
        coverage_rows.append(
            {
                "signature": name,
                "n_requested": len(genes),
                "n_matched": len(matched),
                "matched_genes": ",".join(matched),
                "missing_genes": ",".join(missing),
                "used": score is not None,
            }
        )
        if score is not None:
            scores[name] = scores[args.sample_column].map(score.to_dict())

    scores = scores.merge(meta, on=args.sample_column, how="left")
    scores.to_csv(output_dir / "bulk_signature_sample_scores.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir / "bulk_signature_gene_coverage.csv", index=False)

    stage_order = [item.strip() for item in args.stage_order.split(",") if item.strip()]
    stage_map = {stage: idx for idx, stage in enumerate(stage_order)}
    scores["_stage_ordinal"] = scores[args.stage_column].map(stage_map)
    trend_rows = []
    for name in signatures:
        if name not in scores.columns:
            continue
        sub = scores[[name, "_stage_ordinal"]].dropna()
        if len(sub) < 3 or sub["_stage_ordinal"].nunique() < 2:
            rho, pval = np.nan, np.nan
            beta = np.nan
        else:
            rho, pval = stats.spearmanr(sub["_stage_ordinal"], sub[name])
            beta = float(np.polyfit(sub["_stage_ordinal"], sub[name], deg=1)[0])
        trend_rows.append(
            {
                "signature": name,
                "n_samples": int(len(sub)),
                "spearman_r": float(rho) if not pd.isna(rho) else np.nan,
                "spearman_p": float(pval) if not pd.isna(pval) else np.nan,
                "stage_beta": beta,
            }
        )
    trends = pd.DataFrame(trend_rows).sort_values("spearman_r", ascending=False, na_position="last")
    trends.to_csv(output_dir / "bulk_signature_stage_trends.csv", index=False)
    print(output_dir / "bulk_signature_stage_trends.csv")


if __name__ == "__main__":
    main()
