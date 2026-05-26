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

EXPLICIT_MASS_MODES = {"count", "group_total", "per_cell_contribution"}
PRACTICAL_NULL_FLOORS = {
    "mass": 1e-3,
    "mean_shift": 1e-3,
    "distribution_shift": 1e-3,
    "context_dependence": 1e-3,
    "diffusion_action": 1e-3,
    "tsk_pemt_program": 1e-3,
    "tnf_expansion_program": 1e-3,
    "cis_like_program": 1e-3,
    "program_occupancy_tv": 1e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-perturbation biological ranking tables from CREDO CV outputs."
    )
    parser.add_argument("--cv-root", required=True, help="With-guide CREDO CV root.")
    parser.add_argument("--shared-cv-root", default=None, help="Optional shared-guide/null CV root.")
    parser.add_argument("--signature-scores", default=None, help="Optional signature_group_scores.csv.")
    parser.add_argument("--human-trends", default=None, help="Optional bulk_signature_stage_trends.csv.")
    parser.add_argument("--counterfactual-effects", default=None, help="Optional counterfactual_biology_effects.csv.")
    parser.add_argument(
        "--practical-null-floors-json",
        default=None,
        help=(
            "Optional JSON object or JSON file overriding practical null floors. "
            f"Allowed keys: {', '.join(sorted(PRACTICAL_NULL_FLOORS))}."
        ),
    )
    parser.add_argument("--output-dir", default="results/biology")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_practical_null_floor_overrides(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    path = Path(spec)
    raw = json.loads(path.read_text()) if path.exists() else json.loads(spec)
    if not isinstance(raw, dict):
        raise ValueError("--practical-null-floors-json must be a JSON object.")
    allowed = set(PRACTICAL_NULL_FLOORS)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown practical null floor keys: {unknown}. Allowed keys: {sorted(allowed)}")
    out: dict[str, float] = {}
    for key, value in raw.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"Practical null floor for {key!r} must be finite and nonnegative.")
        out[key] = numeric
    return out


def _apply_practical_null_floor_overrides(overrides: dict[str, float]) -> None:
    PRACTICAL_NULL_FLOORS.update(overrides)


def _first_non_null(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return False
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", ""}:
            return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.map(_parse_bool_value).astype(bool)


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
        if "is_control" in df.columns:
            df["is_control"] = _parse_bool_series(df["is_control"])

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
        data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
        requested_mass_mode = _first_non_null(
            results.get("requested_mass_mode"),
            results.get(f"{split}_requested_mass_mode"),
            config.get("requested_mass_mode"),
            data_cfg.get("requested_mass_mode"),
        )
        split_resolved_key = f"{split}_mass_mode"
        resolved_mass_mode = (
            results.get(split_resolved_key)
            or config.get(split_resolved_key)
            or results.get("resolved_mass_mode")
            or config.get("resolved_mass_mode")
            or (results.get("train_mass_mode") if split == "train" else results.get("test_mass_mode"))
            or (config.get("train_mass_mode") if split == "train" else config.get("test_mass_mode"))
            or results.get("train_mass_mode")
            or config.get("train_mass_mode")
            or requested_mass_mode
        )
        split_reason_key = f"{split}_mass_mode_resolution_reason"
        mass_mode_reason = (
            results.get(split_reason_key)
            or config.get(split_reason_key)
            or results.get("mass_mode_resolution_reason")
            or config.get("mass_mode_resolution_reason")
            or data_cfg.get("mass_mode_resolution_reason")
        )
        if requested_mass_mode is not None:
            df["requested_mass_mode"] = requested_mass_mode
        if resolved_mass_mode is not None:
            df["resolved_mass_mode"] = resolved_mass_mode
        if mass_mode_reason is not None:
            df["mass_mode_resolution_reason"] = mass_mode_reason
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


def _q75(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(0.75)) if not values.empty else float("nan")


def _q95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(0.95)) if not values.empty else float("nan")


def _sign_consistency(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values.loc[values.ne(0.0)]
    if values.empty:
        return float("nan")
    mean = float(values.mean())
    if mean == 0.0:
        return float("nan")
    return float((np.sign(values) == np.sign(mean)).mean())


def _abs_cv(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    denom = abs(float(values.mean()))
    if denom <= 1e-12:
        return float("nan")
    return float(values.std(ddof=1) / denom)


def _replicate_key(df: pd.DataFrame) -> str | None:
    if "fold_id" in df.columns:
        return "fold_id"
    if "run_dir" in df.columns:
        return "run_dir"
    return None


def _fold_support_frame(
    raw: pd.DataFrame,
    *,
    metric_col: str,
    support_col: str,
    floor_key: str,
    positive_only: bool = False,
) -> pd.DataFrame | None:
    if metric_col not in raw.columns or "is_control" not in raw.columns:
        return None
    values = pd.to_numeric(raw[metric_col], errors="coerce")
    values = values.clip(lower=0.0) if positive_only else values.abs()
    control_values = values.loc[raw["is_control"]].dropna()
    if control_values.empty:
        threshold = float("nan")
    else:
        threshold = max(_q95(control_values), PRACTICAL_NULL_FLOORS.get(floor_key, 0.0))

    work = pd.DataFrame({"perturbation_id": raw["perturbation_id"], "_value": values})
    rep_key = _replicate_key(raw)
    if rep_key is not None:
        work[rep_key] = raw[rep_key]
        work = (
            work.groupby(["perturbation_id", rep_key], dropna=False)["_value"]
            .mean()
            .reset_index()
        )

    def support(series: pd.Series) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty or pd.isna(threshold):
            return float("nan")
        return float(clean.gt(threshold).mean())

    return (
        work.groupby("perturbation_id", dropna=False)["_value"]
        .agg(**{support_col: support})
        .reset_index()
    )


def _is_explicit_mass_mode(requested: object = None, resolved: object = None, reason: object = None) -> object:
    """Return whether mass semantics came from an explicit user choice."""
    requested_s = "" if pd.isna(requested) else str(requested).strip().lower()
    resolved_s = "" if pd.isna(resolved) else str(resolved).strip().lower()
    reason_s = "" if pd.isna(reason) else str(reason).strip().lower()

    if requested_s in EXPLICIT_MASS_MODES:
        return True
    if requested_s == "auto":
        return False
    if not requested_s:
        return False
    if "auto" in resolved_s or "auto" in reason_s:
        return False
    # Unknown or legacy requested strings should be conservative: do not infer
    # claim-grade explicitness from resolved computational metadata.
    return False


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
        "resolved_mass_mode": ("resolved_mass_mode", _mode_or_first),
        "requested_mass_mode": ("requested_mass_mode", _mode_or_first),
        "mass_mode_resolution_reason": ("mass_mode_resolution_reason", _mode_or_first),
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
    extra_parts = [out]
    if "pred_log_expansion" in work.columns:
        fold_stability = (
            work.groupby("perturbation_id", dropna=False)["pred_log_expansion"]
            .agg(
                fold_log_expansion_sign_consistency=_sign_consistency,
                fold_log_expansion_abs_cv=_abs_cv,
            )
            .reset_index()
        )
        extra_parts.append(fold_stability)
    if "state_tv" in work.columns:
        state_stability = (
            work.groupby("perturbation_id", dropna=False)["state_tv"]
            .agg(fold_state_tv_abs_cv=_abs_cv)
            .reset_index()
        )
        extra_parts.append(state_stability)
    if len(extra_parts) > 1:
        out = extra_parts[0]
        for part in extra_parts[1:]:
            out = out.merge(part, on="perturbation_id", how="left")
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


def _load_counterfactual_effects(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "perturbation_id" not in df.columns:
        raise KeyError("Counterfactual effects file must contain perturbation_id.")
    raw = df.copy()
    if "is_control" in raw.columns:
        raw["is_control"] = _parse_bool_series(raw["is_control"])
    if "fold_id" in raw.columns:
        dup = raw.duplicated(["perturbation_id", "fold_id"], keep=False)
        if dup.any():
            preview = raw.loc[dup, ["perturbation_id", "fold_id"]].head(5).to_dict("records")
            raise ValueError(f"Duplicate counterfactual rows for perturbation/fold: {preview}")
    if raw["perturbation_id"].duplicated().any():
        agg_spec = {}
        if "delta_tnf_expansion_score" in raw.columns:
            raw["tnf_expansion_program_effect_pos"] = pd.to_numeric(
                raw["delta_tnf_expansion_score"],
                errors="coerce",
            ).clip(lower=0.0)
        if "delta_cis_like_score" in raw.columns:
            raw["cis_like_program_effect_pos"] = pd.to_numeric(
                raw["delta_cis_like_score"],
                errors="coerce",
            ).clip(lower=0.0)
        if {"delta_autocrine_tnf_tsk_score", "delta_pemt_score"} & set(raw.columns):
            raw["tsk_pemt_program_effect_pos"] = pd.concat(
                [
                    pd.to_numeric(
                        raw.get("delta_autocrine_tnf_tsk_score", pd.Series(np.nan, index=raw.index)),
                        errors="coerce",
                    ).clip(lower=0.0),
                    pd.to_numeric(
                        raw.get("delta_pemt_score", pd.Series(np.nan, index=raw.index)),
                        errors="coerce",
                    ).clip(lower=0.0),
                ],
                axis=1,
            ).max(axis=1)
        stability_cols = {
            "delta_log_mass_fact_vs_ref",
            "geom_shift_fact_vs_ref",
            "weighted_mean_shift_l2_fact_vs_ref",
            "energy_distance_fact_vs_ref",
            "program_occupancy_tv_fact_vs_ref",
            "tnf_expansion_program_effect_pos",
            "cis_like_program_effect_pos",
            "tsk_pemt_program_effect_pos",
            "growth_action_fact",
            "drift_action_fact",
            "diffusion_action_fact",
            "context_dependence_geom",
            "context_dependence_mass",
        }
        for col in raw.columns:
            if col == "perturbation_id":
                continue
            if pd.api.types.is_numeric_dtype(raw[col]):
                agg_spec[col] = "mean"
                if col in stability_cols:
                    raw[f"{col}_std"] = pd.to_numeric(raw[col], errors="coerce")
                    agg_spec[f"{col}_std"] = "std"
            else:
                agg_spec[col] = _mode_or_first
        out = raw.groupby("perturbation_id", dropna=False).agg(agg_spec).reset_index()
        rep_key = _replicate_key(raw)
        if rep_key is not None:
            n_reps = raw.groupby("perturbation_id", dropna=False)[rep_key].nunique()
        else:
            n_reps = raw.groupby("perturbation_id", dropna=False)["perturbation_id"].size()
        out["counterfactual_n_folds"] = out["perturbation_id"].map(n_reps).astype(int)
        for col in stability_cols:
            if col in raw.columns:
                stability = (
                    raw.groupby("perturbation_id", dropna=False)[col]
                    .agg(**{f"{col}_sign_consistency": _sign_consistency, f"{col}_abs_cv": _abs_cv})
                    .reset_index()
                )
                out = out.merge(stability, on="perturbation_id", how="left")
        fold_support_specs = [
            (
                "energy_distance_fact_vs_ref",
                "energy_distance_fact_vs_ref_fold_support",
                "distribution_shift",
                False,
            ),
            (
                "program_occupancy_tv_fact_vs_ref",
                "program_occupancy_tv_fact_vs_ref_fold_support",
                "program_occupancy_tv",
                False,
            ),
            (
                "tnf_expansion_program_effect_pos",
                "tnf_expansion_program_effect_pos_fold_support",
                "tnf_expansion_program",
                True,
            ),
            (
                "cis_like_program_effect_pos",
                "cis_like_program_effect_pos_fold_support",
                "cis_like_program",
                True,
            ),
            (
                "tsk_pemt_program_effect_pos",
                "tsk_pemt_program_effect_pos_fold_support",
                "tsk_pemt_program",
                True,
            ),
        ]
        for metric_col, support_col, floor_key, positive_only in fold_support_specs:
            support = _fold_support_frame(
                raw,
                metric_col=metric_col,
                support_col=support_col,
                floor_key=floor_key,
                positive_only=positive_only,
            )
            if support is not None:
                out = out.merge(support, on="perturbation_id", how="left")
    else:
        out = raw
        out["counterfactual_n_folds"] = 1
    if "geometry_shift_l2" in out.columns and "geom_shift_fact_vs_ref" not in out.columns:
        out["geom_shift_fact_vs_ref"] = out["geometry_shift_l2"]
    if "geom_shift_fact_vs_ref" in out.columns and "legacy_geom_shift_fact_vs_ref" not in out.columns:
        out["legacy_geom_shift_fact_vs_ref"] = out["geom_shift_fact_vs_ref"]
    if "context_dependence" in out.columns and "context_dependence_geom" not in out.columns:
        out["context_dependence_geom"] = out["context_dependence"]
    if "delta_log_mass_self_vs_clamped" in out.columns and "context_dependence_mass" not in out.columns:
        out["context_dependence_mass"] = out["delta_log_mass_self_vs_clamped"].abs()
    if "growth_action_fact" in out.columns and "growth_action" not in out.columns:
        out["growth_action"] = out["growth_action_fact"]
    if "drift_action_fact" in out.columns and "drift_action" not in out.columns:
        out["drift_action"] = out["drift_action_fact"]
    if "diffusion_action_fact" in out.columns and "diffusion_action" not in out.columns:
        out["diffusion_action"] = out["diffusion_action_fact"]
    if "terminal_entropy_factual" in out.columns and "terminal_state_entropy_fact" not in out.columns:
        out["terminal_state_entropy_fact"] = out["terminal_entropy_factual"]
    if "terminal_entropy_reference" in out.columns and "terminal_state_entropy_ref" not in out.columns:
        out["terminal_state_entropy_ref"] = out["terminal_entropy_reference"]
    return out.drop(columns=["target_gene"], errors="ignore")


def _add_guide_concordance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sgRNA_id" not in out.columns:
        out["sgRNA_id"] = out["perturbation_id"]
    metric = None
    for candidate in ["delta_log_mass_fact_vs_ref", "pred_log_expansion_mean", "true_log_expansion_mean"]:
        if candidate in out.columns:
            metric = candidate
            break
    if metric is None:
        out["same_gene_n_guides"] = np.nan
        out["same_gene_sgrna_concordance"] = np.nan
        out["same_gene_effect_abs_cv"] = np.nan
        return out

    work = out.loc[~out["target_gene"].astype(str).str.lower().eq("control")].copy()
    guide_col = "sgRNA_id" if "sgRNA_id" in work.columns else "perturbation_id"
    guide_stats = (
        work.groupby("target_gene", dropna=False)
        .agg(
            same_gene_n_guides=(guide_col, "nunique"),
            same_gene_sgrna_concordance=(metric, _sign_consistency),
            same_gene_effect_abs_cv=(metric, _abs_cv),
        )
        .reset_index()
    )
    return out.merge(guide_stats, on="target_gene", how="left")


def _add_biological_gates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "delta_log_mass" not in out.columns:
        out["delta_log_mass"] = np.nan
    is_control = _parse_bool_series(out.get("is_control", pd.Series(False, index=out.index)))
    out["is_control"] = is_control
    controls = out.loc[is_control]

    def add_metric_null(metric_col: str, prefix: str) -> None:
        metric = pd.to_numeric(out.get(metric_col, pd.Series(np.nan, index=out.index)), errors="coerce")
        current_controls = out.loc[is_control]
        control_values = pd.to_numeric(
            current_controls.get(metric_col, pd.Series(dtype=float)),
            errors="coerce",
        ).abs()
        null_q95 = _q95(control_values) if not current_controls.empty else float("nan")
        floor = PRACTICAL_NULL_FLOORS.get(prefix, 0.0)
        null_threshold = max(null_q95, floor) if not pd.isna(null_q95) else float("nan")
        out[f"{prefix}_null_abs_q95"] = null_q95
        out[f"{prefix}_null_practical_floor"] = floor
        out[f"{prefix}_null_threshold"] = null_threshold
        if pd.isna(null_q95):
            out[f"{prefix}_null_gap_pass"] = pd.NA
        else:
            out[f"{prefix}_null_gap_pass"] = metric.abs() > null_threshold

    if "weighted_mean_shift_l2_fact_vs_ref" not in out.columns and "geom_shift_fact_vs_ref" in out.columns:
        out["weighted_mean_shift_l2_fact_vs_ref"] = out["geom_shift_fact_vs_ref"]
    if "energy_distance_fact_vs_ref" not in out.columns:
        out["energy_distance_fact_vs_ref"] = np.nan
    tsk_score = pd.to_numeric(
        out.get("delta_autocrine_tnf_tsk_score", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    pemt_score = pd.to_numeric(
        out.get("delta_pemt_score", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    tnf_score_raw = pd.to_numeric(
        out.get("delta_tnf_expansion_score", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    cis_score_raw = pd.to_numeric(
        out.get("delta_cis_like_score", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["tsk_pemt_program_effect_abs"] = pd.concat([tsk_score.abs(), pemt_score.abs()], axis=1).max(axis=1)
    out["tsk_pemt_program_effect_pos"] = pd.concat(
        [tsk_score.clip(lower=0.0), pemt_score.clip(lower=0.0)],
        axis=1,
    ).max(axis=1)
    out["tnf_expansion_program_effect_pos"] = tnf_score_raw.clip(lower=0.0)
    out["cis_like_program_effect_pos"] = cis_score_raw.clip(lower=0.0)
    add_metric_null("delta_log_mass", "mass")
    add_metric_null("weighted_mean_shift_l2_fact_vs_ref", "mean_shift")
    add_metric_null("energy_distance_fact_vs_ref", "distribution_shift")
    add_metric_null("context_dependence_geom", "context_dependence")
    add_metric_null("diffusion_action", "diffusion_action")
    add_metric_null("tsk_pemt_program_effect_pos", "tsk_pemt_program")
    add_metric_null("tnf_expansion_program_effect_pos", "tnf_expansion_program")
    add_metric_null("cis_like_program_effect_pos", "cis_like_program")
    add_metric_null("program_occupancy_tv_fact_vs_ref", "program_occupancy_tv")
    out["control_null_abs_delta_log_mass_q95"] = out["mass_null_abs_q95"]
    out["negative_control_gap_pass"] = out["mass_null_gap_pass"]

    n_guides = pd.to_numeric(
        out.get("same_gene_n_guides", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    guide_conc = pd.to_numeric(
        out.get("same_gene_sgrna_concordance", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["guide_concordance_status"] = np.select(
        [n_guides.ge(2) & guide_conc.ge(0.75), n_guides.ge(2)],
        ["pass", "fail"],
        default="not_assessable",
    )
    out["guide_concordance_pass"] = out["guide_concordance_status"].eq("pass")

    cf_folds = pd.to_numeric(
        out.get("counterfactual_n_folds", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["counterfactual_replicate_pass"] = cf_folds.fillna(0) >= 2
    cf_mass_sign = pd.to_numeric(
        out.get("delta_log_mass_fact_vs_ref_sign_consistency", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["fold_stability_pass"] = out["counterfactual_replicate_pass"] & cf_mass_sign.ge(0.75)

    diffusion_sign = pd.to_numeric(
        out.get("diffusion_action_fact_sign_consistency", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["plasticity_stability_pass"] = (
        out["counterfactual_replicate_pass"]
        & diffusion_sign.ge(0.75)
    )
    distribution_fold_support = pd.to_numeric(
        out.get("energy_distance_fact_vs_ref_fold_support", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["distribution_shift_stability_pass"] = (
        out["counterfactual_replicate_pass"]
        & distribution_fold_support.ge(0.75)
    )
    program_occupancy_fold_support = pd.to_numeric(
        out.get("program_occupancy_tv_fact_vs_ref_fold_support", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["program_occupancy_stability_pass"] = (
        out["counterfactual_replicate_pass"]
        & program_occupancy_fold_support.ge(0.75)
    )
    tnf_program_fold_support = pd.to_numeric(
        out.get("tnf_expansion_program_effect_pos_fold_support", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["tnf_expansion_program_stability_pass"] = (
        out["counterfactual_replicate_pass"]
        & tnf_program_fold_support.ge(0.75)
    )
    cis_program_fold_support = pd.to_numeric(
        out.get("cis_like_program_effect_pos_fold_support", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["cis_like_program_stability_pass"] = (
        out["counterfactual_replicate_pass"]
        & cis_program_fold_support.ge(0.75)
    )
    tsk_pemt_program_fold_support = pd.to_numeric(
        out.get("tsk_pemt_program_effect_pos_fold_support", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["tsk_pemt_program_stability_pass"] = (
        out["counterfactual_replicate_pass"]
        & tsk_pemt_program_fold_support.ge(0.75)
    )

    context_available = out.get(
        "context_dependence_geom",
        pd.Series(np.nan, index=out.index),
    ).notna()
    context_sign = pd.to_numeric(
        out.get("context_dependence_geom_sign_consistency", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["ecology_ablation_pass"] = (
        out["counterfactual_replicate_pass"]
        & context_available
        & context_sign.ge(0.75)
    )
    requested_mode = out.get("requested_mass_mode")
    resolved_mode = out.get("resolved_mass_mode")
    resolution_reason = out.get("mass_mode_resolution_reason")
    if requested_mode is None and resolved_mode is None and resolution_reason is None:
        out["explicit_mass_mode_pass"] = False
    else:
        if requested_mode is None:
            requested_mode = pd.Series(pd.NA, index=out.index)
        if resolved_mode is None:
            resolved_mode = pd.Series(pd.NA, index=out.index)
        if resolution_reason is None:
            resolution_reason = pd.Series(pd.NA, index=out.index)
        out["explicit_mass_mode_pass"] = [
            _is_explicit_mass_mode(req, res, reason)
            for req, res, reason in zip(requested_mode, resolved_mode, resolution_reason)
        ]

    def null_gate(row: pd.Series, pass_col: str, name: str) -> str | None:
        value = row.get(pass_col, pd.NA)
        if pd.isna(value):
            return f"missing-{name}-null"
        if not bool(value):
            return f"below-{name}-null-gap"
        return None

    def gate(row: pd.Series) -> str:
        priority = str(row.get("priority_class", "watch"))
        if priority == "artifact-watch":
            return "artifact-watch"
        if pd.isna(row.get("delta_log_mass_fact_vs_ref", np.nan)):
            return "counterfactual-pending"
        if not bool(row.get("counterfactual_replicate_pass", False)):
            return "needs-counterfactual-replicates"
        if not bool(row.get("guide_concordance_pass", True)):
            return "needs-guide-concordance"
        explicit_mass = row.get("explicit_mass_mode_pass", pd.NA)
        if pd.notna(explicit_mass) and not bool(explicit_mass):
            return "needs-explicit-mass-mode"
        if priority == "ecology-dependent":
            null_reason = null_gate(row, "context_dependence_null_gap_pass", "context")
            if null_reason is not None:
                return null_reason
            if not bool(row.get("ecology_ablation_pass", False)):
                return "needs-context-ablation"
        elif priority == "plasticity/state-shift":
            diffusion_reason = null_gate(row, "diffusion_action_null_gap_pass", "diffusion")
            if diffusion_reason is not None:
                return diffusion_reason
            distribution_reason = null_gate(row, "distribution_shift_null_gap_pass", "distribution-shift")
            occupancy_reason = null_gate(row, "program_occupancy_tv_null_gap_pass", "program-occupancy")
            distribution_axis_ok = (
                distribution_reason is None
                and bool(row.get("distribution_shift_stability_pass", False))
            )
            occupancy_axis_ok = (
                occupancy_reason is None
                and bool(row.get("program_occupancy_stability_pass", False))
            )
            if distribution_reason is not None and occupancy_reason is not None:
                return distribution_reason
            if not (distribution_axis_ok or occupancy_axis_ok):
                return "needs-distribution-shift-stability"
            if not bool(row.get("plasticity_stability_pass", False)):
                return "needs-diffusion-stability"
        else:
            if not bool(row.get("fold_stability_pass", False)):
                return "needs-fold-stability"
            null_reason = null_gate(row, "mass_null_gap_pass", "mass")
            if null_reason is not None:
                return null_reason
        return "claim-ready"

    out["biological_interpretation_gate"] = out.apply(gate, axis=1)
    out["claim_ready"] = out["biological_interpretation_gate"].eq("claim-ready")

    def pass_series(column: str) -> pd.Series:
        values = out.get(column, pd.Series(False, index=out.index))
        return values.map(lambda value: False if pd.isna(value) else bool(value))

    common_stable = (
        out["counterfactual_replicate_pass"]
        & out["guide_concordance_pass"]
    )
    explicit_mass_ready = out["explicit_mass_mode_pass"].fillna(False).astype(bool)
    common_stable = common_stable & explicit_mass_ready
    mass_stable = common_stable & out["fold_stability_pass"]
    mass_delta = pd.to_numeric(
        out.get("delta_log_mass_fact_vs_ref", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    mass_threshold = pd.to_numeric(
        out.get("mass_null_threshold", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["expansion_claim_ready"] = (
        mass_stable
        & pass_series("mass_null_gap_pass")
        & mass_delta.gt(mass_threshold)
    )
    out["depletion_claim_ready"] = (
        mass_stable
        & pass_series("mass_null_gap_pass")
        & mass_delta.lt(-mass_threshold)
    )
    distribution_ready = (
        out["distribution_shift_stability_pass"]
        & pass_series("distribution_shift_null_gap_pass")
    )
    program_occupancy_ready = (
        out["program_occupancy_stability_pass"]
        & pass_series("program_occupancy_tv_null_gap_pass")
    )
    out["plasticity_claim_ready"] = (
        common_stable
        & out["plasticity_stability_pass"]
        & (distribution_ready | program_occupancy_ready)
        & pass_series("diffusion_action_null_gap_pass")
    )
    out["ecology_claim_ready"] = (
        common_stable
        & out["ecology_ablation_pass"]
        & pass_series("context_dependence_null_gap_pass")
    )
    tsk = pd.to_numeric(out.get("z_delta_autocrine_tnf_tsk_score", pd.Series(0.0, index=out.index)), errors="coerce")
    pemt = pd.to_numeric(out.get("z_delta_pemt_score", pd.Series(0.0, index=out.index)), errors="coerce")
    tsk_pemt_pos = pd.to_numeric(
        out.get("tsk_pemt_program_effect_pos", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    tsk_pemt_threshold = pd.to_numeric(
        out.get("tsk_pemt_program_null_threshold", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    tnf_program_pos = pd.to_numeric(
        out.get("tnf_expansion_program_effect_pos", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    tnf_program_threshold = pd.to_numeric(
        out.get("tnf_expansion_program_null_threshold", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    cis_program_pos = pd.to_numeric(
        out.get("cis_like_program_effect_pos", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    cis_program_threshold = pd.to_numeric(
        out.get("cis_like_program_null_threshold", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["tnf_expansion_claim_ready"] = (
        common_stable
        & out["tnf_expansion_program_stability_pass"]
        & tnf_program_pos.gt(tnf_program_threshold)
        & pass_series("tnf_expansion_program_null_gap_pass")
    )
    out["cis_like_claim_ready"] = (
        common_stable
        & out["cis_like_program_stability_pass"]
        & cis_program_pos.gt(cis_program_threshold)
        & pass_series("cis_like_program_null_gap_pass")
    )
    out["tsk_pemt_claim_ready"] = (
        common_stable
        & out["tsk_pemt_program_stability_pass"]
        & ((tsk >= 0.5) | (pemt >= 0.5))
        & tsk_pemt_pos.gt(tsk_pemt_threshold)
        & pass_series("tsk_pemt_program_null_gap_pass")
    )
    out["transformation_claim_ready"] = (
        out["tsk_pemt_claim_ready"]
        & (out["plasticity_claim_ready"] | out["ecology_claim_ready"])
    )
    out["claim_ready_strict"] = out["claim_ready"]

    def screening_ready(row: pd.Series) -> bool:
        priority = str(row.get("priority_class", "watch"))
        if priority == "artifact-watch":
            return False
        if pd.isna(row.get("delta_log_mass_fact_vs_ref", np.nan)):
            return False
        if not bool(row.get("counterfactual_replicate_pass", False)):
            return False
        if str(row.get("guide_concordance_status", "not_assessable")) == "fail":
            return False
        explicit_mass = row.get("explicit_mass_mode_pass", pd.NA)
        if pd.notna(explicit_mass) and not bool(explicit_mass):
            return False
        if priority == "ecology-dependent":
            return bool(row.get("ecology_ablation_pass", False)) and null_gate(
                row,
                "context_dependence_null_gap_pass",
                "context",
            ) is None
        if priority == "plasticity/state-shift":
            return (
                bool(row.get("plasticity_stability_pass", False))
                and (
                    bool(row.get("distribution_shift_stability_pass", False))
                    or bool(row.get("program_occupancy_stability_pass", False))
                )
                and all(
                    null_gate(row, pass_col, name) is None
                    for pass_col, name in [
                        ("diffusion_action_null_gap_pass", "diffusion"),
                    ]
                )
                and (
                    null_gate(row, "distribution_shift_null_gap_pass", "distribution-shift") is None
                    or null_gate(row, "program_occupancy_tv_null_gap_pass", "program-occupancy") is None
                )
            )
        if not bool(row.get("fold_stability_pass", False)):
            return False
        return null_gate(row, "mass_null_gap_pass", "mass") is None

    out["claim_ready_screening"] = out.apply(screening_ready, axis=1)
    out["claim_blocking_reasons"] = out["biological_interpretation_gate"].where(
        ~out["claim_ready"],
        "",
    )
    return out


def classify_mechanistic_v2(row: pd.Series) -> str:
    cf_mass = row.get("delta_log_mass_fact_vs_ref", np.nan)
    if pd.isna(cf_mass):
        pre = str(row.get("priority_class_pre_counterfactual", "watch"))
        if pre == "Class I":
            return "growth-high / relatively TSK-high (counterfactual pending)"
        if pre == "Class II":
            return "growth-high / TNF-expansion-only (counterfactual pending)"
        if pre in {"Class III", "Class IV"}:
            return f"{pre} (counterfactual pending)"
        return "watch (counterfactual pending)"
    growth = float(row.get("z_delta_log_mass", 0.0) or 0.0)
    tnf = float(row.get("z_delta_tnf_expansion_score", 0.0) or 0.0)
    tsk = float(row.get("z_delta_autocrine_tnf_tsk_score", 0.0) or 0.0)
    pemt = float(row.get("z_delta_pemt_score", 0.0) or 0.0)
    diffusion = float(row.get("z_diffusion_action", 0.0) or 0.0)
    context = float(row.get("z_context_dependence", 0.0) or 0.0)
    shared_gap = row.get("shared_guide_null_gap", np.nan)
    mass_err = row.get("mass_rel_error_mean", np.nan)
    mass_err_cut = row.get("_mass_error_q75", np.nan)
    if pd.notna(shared_gap) and float(shared_gap) < -0.25:
        return "artifact-watch"
    if pd.notna(mass_err) and pd.notna(mass_err_cut) and float(mass_err) > float(mass_err_cut) and growth < 0.5:
        return "artifact-watch"
    if context >= 0.75:
        return "ecology-dependent"
    if diffusion >= 0.75 and (tsk >= 0.0 or pemt >= 0.0):
        return "plasticity/state-shift"
    if growth >= 0.5 and (tsk >= 0.5 or pemt >= 0.5):
        return "transformation-prone"
    if growth >= 0.5 and tnf >= 0.5 and tsk < 0.5 and pemt < 0.5:
        return "expansion-only"
    if growth >= 0.5:
        return "growth-high"
    return "watch"


def _add_priority(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    endpoint_delta = out.get("pred_log_expansion_mean", out.get("true_log_expansion_mean"))
    out["delta_log_mass_observed_endpoint"] = endpoint_delta
    if "delta_log_mass_fact_vs_ref" in out.columns:
        out["delta_log_mass"] = pd.to_numeric(out["delta_log_mass_fact_vs_ref"], errors="coerce").fillna(endpoint_delta)
    else:
        out["delta_log_mass"] = endpoint_delta
    for col in [
        "delta_log_mass",
        "delta_tnf_expansion_score",
        "delta_autocrine_tnf_tsk_score",
        "delta_pemt_score",
        "diffusion_action",
        "context_dependence",
        "context_dependence_geom",
        "human_autocrine_tnf_tsk_trend",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["z_delta_log_mass"] = zscore(out["delta_log_mass"]).fillna(0.0)
    out["z_delta_tnf_expansion_score"] = zscore(out["delta_tnf_expansion_score"]).fillna(0.0)
    out["z_delta_autocrine_tnf_tsk_score"] = zscore(out["delta_autocrine_tnf_tsk_score"]).fillna(0.0)
    out["z_delta_pemt_score"] = zscore(out["delta_pemt_score"]).fillna(0.0)
    out["z_diffusion_action"] = zscore(out["diffusion_action"]).fillna(0.0)
    context_for_priority = out["context_dependence"]
    if context_for_priority.isna().all() and "context_dependence_geom" in out.columns:
        context_for_priority = out["context_dependence_geom"]
    out["z_context_dependence"] = zscore(context_for_priority).fillna(0.0)
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
    out["priority_class_pre_counterfactual"] = out.apply(classify_priority, axis=1)
    out["_mass_error_q75"] = _q75(out["mass_rel_error_mean"]) if "mass_rel_error_mean" in out.columns else np.nan
    out["priority_class_v2"] = out.apply(classify_mechanistic_v2, axis=1)
    out["priority_class"] = out["priority_class_v2"]
    out = out.drop(columns=["_mass_error_q75"], errors="ignore")
    out = _add_biological_gates(out)
    return out.sort_values("priority_score", ascending=False, na_position="last").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    _apply_practical_null_floor_overrides(
        _load_practical_null_floor_overrides(args.practical_null_floors_json)
    )
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

    if args.counterfactual_effects:
        effects = effects.merge(
            _load_counterfactual_effects(args.counterfactual_effects),
            on="perturbation_id",
            how="left",
        )

    effects = _add_guide_concordance(effects)
    ranked = _add_priority(effects)
    ranked.to_csv(output_dir / "biological_effects_per_perturbation.csv", index=False)
    ranked.to_csv(output_dir / "biological_effects_per_perturbation_v2.csv", index=False)

    preview_cols = [
        col
        for col in [
            "perturbation_id",
            "target_gene",
            "sgRNA_id",
            "priority_class",
            "priority_class_pre_counterfactual",
            "priority_score",
            "biological_interpretation_gate",
            "claim_ready",
            "claim_ready_strict",
            "claim_ready_screening",
            "claim_blocking_reasons",
            "fold_stability_pass",
            "guide_concordance_pass",
            "guide_concordance_status",
            "requested_mass_mode",
            "resolved_mass_mode",
            "mass_mode_resolution_reason",
            "explicit_mass_mode_pass",
            "negative_control_gap_pass",
            "mass_null_threshold",
            "distribution_shift_null_gap_pass",
            "distribution_shift_stability_pass",
            "program_occupancy_tv_null_gap_pass",
            "program_occupancy_stability_pass",
            "tnf_expansion_program_null_gap_pass",
            "tnf_expansion_program_stability_pass",
            "cis_like_program_null_gap_pass",
            "cis_like_program_stability_pass",
            "tsk_pemt_program_null_gap_pass",
            "tsk_pemt_program_stability_pass",
            "expansion_claim_ready",
            "depletion_claim_ready",
            "tnf_expansion_claim_ready",
            "cis_like_claim_ready",
            "plasticity_claim_ready",
            "ecology_claim_ready",
            "tsk_pemt_claim_ready",
            "transformation_claim_ready",
            "delta_log_mass",
            "delta_log_mass_fact_vs_ref",
            "geom_shift_fact_vs_ref",
            "energy_distance_fact_vs_ref",
            "program_occupancy_tv_fact_vs_ref",
            "delta_tnf_expansion_score",
            "delta_autocrine_tnf_tsk_score",
            "delta_pemt_score",
            "delta_lrp1_proxy_module_score",
            "delta_lrp1_epithelial_tsk_pemt_core_score",
            "delta_lrp1_caf_ecm_core_score",
            "delta_lrp1_inflammatory_myeloid_core_score",
            "diffusion_action",
            "context_dependence_geom",
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
