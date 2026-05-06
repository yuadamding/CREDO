#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hnscc_biology_common import (
    classify_priority,
    infer_target_gene,
    write_markdown_table,
    zscore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-perturbation biological ranking tables from CREDO CV outputs."
    )
    parser.add_argument("--cv-root", required=True, help="With-guide CREDO CV root.")
    parser.add_argument("--shared-cv-root", default=None, help="Optional shared-guide/null CV root.")
    parser.add_argument("--signature-scores", default=None, help="Optional signature_group_scores.csv.")
    parser.add_argument("--human-trends", default=None, help="Optional bulk_signature_stage_trends.csv.")
    parser.add_argument("--output-dir", default="results/biology")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _collect_single_root(cv_root: Path, *, split: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    endpoint_name = f"{split}_endpoint_metrics.csv"
    state_name = f"{split}_state_metrics.csv"
    dist_name = f"{split}_state_distributions.csv"

    for endpoint_path in sorted(cv_root.rglob(endpoint_name)):
        run_dir = endpoint_path.parent
        config = _read_json(run_dir / "config.json")
        results = _read_json(run_dir / "results_summary.json")
        split_meta = config.get("split", {})
        rel_parts = run_dir.relative_to(cv_root).parts
        setting_name = rel_parts[0] if len(rel_parts) >= 2 else run_dir.parent.name

        endpoint = pd.read_csv(endpoint_path)
        state_path = run_dir / state_name
        if state_path.exists():
            state = pd.read_csv(state_path)
            df = endpoint.merge(state, on="perturbation_id", how="left")
        else:
            df = endpoint.copy()

        dist_path = run_dir / dist_name
        if dist_path.exists():
            dist = pd.read_csv(dist_path)
            if {"perturbation_id", "state", "pred_fraction", "true_fraction"} <= set(dist.columns):
                pred_top = (
                    dist.sort_values(["perturbation_id", "pred_fraction"], ascending=[True, False])
                    .drop_duplicates("perturbation_id")
                    [["perturbation_id", "state", "pred_fraction"]]
                    .rename(columns={"state": "top_pred_state", "pred_fraction": "top_pred_state_fraction"})
                )
                true_top = (
                    dist.sort_values(["perturbation_id", "true_fraction"], ascending=[True, False])
                    .drop_duplicates("perturbation_id")
                    [["perturbation_id", "state", "true_fraction"]]
                    .rename(columns={"state": "top_true_state", "true_fraction": "top_true_state_fraction"})
                )
                df = df.merge(pred_top, on="perturbation_id", how="left")
                df = df.merge(true_top, on="perturbation_id", how="left")

        df["run_dir"] = str(run_dir)
        df["setting_name"] = setting_name
        df["fold_index"] = split_meta.get("fold_index")
        df["split_strategy"] = split_meta.get("split_strategy")
        df["shared_guide_embedding"] = _as_bool(
            results.get("shared_guide_embedding", config.get("shared_guide_embedding"))
        )
        df["program_basis"] = "state_centroids" if _as_bool(config.get("use_state_centroids")) else "learned"
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No {endpoint_name} files found under {cv_root}")
    return pd.concat(rows, ignore_index=True)


def _mode_or_first(series: pd.Series):
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    mode = clean.mode()
    return mode.iloc[0] if not mode.empty else clean.iloc[0]


def _aggregate(df: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    work = df.copy()
    if "pred_expansion_ratio" in work.columns:
        work["pred_log_expansion"] = np.log(pd.to_numeric(work["pred_expansion_ratio"], errors="coerce"))
    if "true_expansion_ratio" in work.columns:
        work["true_log_expansion"] = np.log(pd.to_numeric(work["true_expansion_ratio"], errors="coerce"))
    if "expansion_ratio_gap" in work.columns:
        work["abs_expansion_ratio_gap"] = pd.to_numeric(work["expansion_ratio_gap"], errors="coerce").abs()

    agg_spec = {
        "n_folds": ("run_dir", "nunique"),
        "is_control": ("is_control", "max"),
        "n_p4_mean": ("n_init_atoms", "mean"),
        "n_p60_mean": ("n_term_atoms", "mean"),
        "mass_true_mean": ("mass_true", "mean"),
        "mass_pred_mean": ("mass_pred", "mean"),
        "mass_rel_error_mean": ("mass_rel_error", "mean"),
        "mass_rel_error_std": ("mass_rel_error", "std"),
    }
    optional = {
        "state_tv_mean": ("state_tv", "mean"),
        "state_tv_std": ("state_tv", "std"),
        "dominant_state_match_rate": ("dominant_state_match", "mean"),
        "dominant_state_true": ("dominant_state_true", _mode_or_first),
        "dominant_state_pred": ("dominant_state_pred", _mode_or_first),
        "top_pred_state": ("top_pred_state", _mode_or_first),
        "top_true_state": ("top_true_state", _mode_or_first),
        "pred_expansion_ratio_mean": ("pred_expansion_ratio", "mean"),
        "true_expansion_ratio_mean": ("true_expansion_ratio", "mean"),
        "pred_log_expansion_mean": ("pred_log_expansion", "mean"),
        "true_log_expansion_mean": ("true_log_expansion", "mean"),
        "expansion_ratio_gap_mean": ("expansion_ratio_gap", "mean"),
        "abs_expansion_ratio_gap_mean": ("abs_expansion_ratio_gap", "mean"),
    }
    for out_col, spec in optional.items():
        if spec[0] in work.columns:
            agg_spec[out_col] = spec

    out = work.groupby("perturbation_id", dropna=False).agg(**agg_spec).reset_index()
    out["target_gene"] = out["perturbation_id"].map(infer_target_gene)
    if prefix:
        rename = {
            col: f"{prefix}_{col}"
            for col in out.columns
            if col not in {"perturbation_id", "target_gene"}
        }
        out = out.rename(columns=rename)
    return out


def _load_signature_deltas(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time point" in df.columns and "time_label" not in df.columns:
        df = df.rename(columns={"Time point": "time_label"})
    if "signature" in df.columns:
        value_col = "mean_score" if "mean_score" in df.columns else "score_mean"
        wide = df.pivot_table(
            index=["perturbation_id", "time_label"],
            columns="signature",
            values=value_col,
            aggfunc="mean",
        ).reset_index()
    else:
        wide = df.copy()
    if "perturbation_id" not in wide.columns or "time_label" not in wide.columns:
        raise KeyError("Signature scores need perturbation_id and time_label columns.")

    value_cols = [col for col in wide.columns if col not in {"perturbation_id", "time_label", "n_cells"}]
    p4 = wide.loc[wide["time_label"].astype(str).str.upper().isin({"P4", "4", "4.0"})]
    p60 = wide.loc[wide["time_label"].astype(str).str.upper().isin({"P60", "60", "60.0"})]
    merged = p60[["perturbation_id", *value_cols]].merge(
        p4[["perturbation_id", *value_cols]],
        on="perturbation_id",
        how="left",
        suffixes=("_p60", "_p4"),
    )
    out = pd.DataFrame({"perturbation_id": merged["perturbation_id"]})
    for col in value_cols:
        out[f"delta_{col}_score"] = (
            pd.to_numeric(merged[f"{col}_p60"], errors="coerce")
            - pd.to_numeric(merged[f"{col}_p4"], errors="coerce")
        )
    return out


def _load_human_trends(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "signature" not in df.columns:
        raise KeyError("Human trend file must contain a signature column.")
    value_col = "spearman_r" if "spearman_r" in df.columns else "stage_beta"
    out = pd.DataFrame()
    for _, row in df.iterrows():
        out.loc[0, f"human_{row['signature']}_trend"] = row.get(value_col)
        out.loc[0, f"human_{row['signature']}_p"] = row.get("p_value", row.get("spearman_p"))
    return out


def _add_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_log_mass"] = out.get("pred_log_expansion_mean", out.get("true_log_expansion_mean"))
    for col in [
        "delta_log_mass",
        "delta_tnf_expansion_score",
        "delta_autocrine_tnf_tsk_score",
        "delta_pemt_score",
        "diffusion_action",
        "context_dependence",
        "human_autocrine_tnf_tsk_trend",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["z_delta_log_mass"] = zscore(out["delta_log_mass"]).fillna(0.0)
    out["z_delta_tnf_expansion_score"] = zscore(out["delta_tnf_expansion_score"]).fillna(0.0)
    out["z_delta_autocrine_tnf_tsk_score"] = zscore(out["delta_autocrine_tnf_tsk_score"]).fillna(0.0)
    out["z_delta_pemt_score"] = zscore(out["delta_pemt_score"]).fillna(0.0)
    out["z_diffusion_action"] = zscore(out["diffusion_action"]).fillna(0.0)
    out["z_context_dependence"] = zscore(out["context_dependence"]).fillna(0.0)
    out["z_human_stage_trend"] = zscore(out["human_autocrine_tnf_tsk_trend"]).fillna(0.0)

    out["priority_score"] = (
        0.25 * out["z_delta_log_mass"]
        + 0.20 * out["z_delta_autocrine_tnf_tsk_score"]
        + 0.15 * out["z_delta_tnf_expansion_score"]
        + 0.15 * out["z_diffusion_action"]
        + 0.15 * out["z_human_stage_trend"]
    )
    if "same_gene_sgrna_concordance" in out.columns:
        out["priority_score"] += 0.10 * zscore(out["same_gene_sgrna_concordance"]).fillna(0.0)
    if "shared_dominant_state_match_rate" in out.columns and "dominant_state_match_rate" in out.columns:
        out["shared_guide_null_gap"] = (
            pd.to_numeric(out["dominant_state_match_rate"], errors="coerce")
            - pd.to_numeric(out["shared_dominant_state_match_rate"], errors="coerce")
        )
    out["priority_class"] = out.apply(classify_priority, axis=1)
    return out.sort_values("priority_score", ascending=False, na_position="last").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _collect_single_root(Path(args.cv_root), split=args.split)
    raw.to_csv(output_dir / f"{args.split}_per_fold_perturbation_metrics.csv", index=False)
    effects = _aggregate(raw)

    if args.shared_cv_root:
        shared_raw = _collect_single_root(Path(args.shared_cv_root), split=args.split)
        shared = _aggregate(shared_raw, prefix="shared")
        effects = effects.merge(shared.drop(columns=["target_gene"], errors="ignore"), on="perturbation_id", how="left")

    if args.signature_scores:
        effects = effects.merge(_load_signature_deltas(args.signature_scores), on="perturbation_id", how="left")

    if args.human_trends:
        human = _load_human_trends(args.human_trends)
        for col, value in human.iloc[0].items():
            effects[col] = value

    ranked = _add_priority(effects)
    ranked.to_csv(output_dir / "biological_effects_per_perturbation.csv", index=False)

    preview_cols = [
        col
        for col in [
            "perturbation_id",
            "target_gene",
            "priority_class",
            "priority_score",
            "delta_log_mass",
            "delta_tnf_expansion_score",
            "delta_autocrine_tnf_tsk_score",
            "dominant_state_match_rate",
            "state_tv_mean",
            "mass_rel_error_mean",
            "shared_guide_null_gap",
        ]
        if col in ranked.columns
    ]
    write_markdown_table(
        ranked[preview_cols],
        output_dir / "biological_effects_top_hits.md",
        title="CREDO Biological Effect Candidates",
        max_rows=args.top_n,
    )
    print(output_dir / "biological_effects_per_perturbation.csv")


if __name__ == "__main__":
    main()
