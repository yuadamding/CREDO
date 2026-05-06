"""Summarize HNSCC CV runs across folds and control modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _nested_get(mapping: dict, *keys: str):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _gpu_stat_gb(stats: dict | None, key: str) -> float | None:
    if not isinstance(stats, dict):
        return None
    value = stats.get(key)
    if value is None:
        return None
    return float(value) / 1024.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HNSCC cross-validation runs.")
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--group-by",
        choices=["control_mode", "setting"],
        default="control_mode",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=["balanced", "dominant_state", "test_acc"],
        default="balanced",
    )
    return parser.parse_args()


def collect_run_rows(cv_root: Path) -> list[dict]:
    rows: list[dict] = []
    for results_path in sorted(cv_root.rglob("results_summary.json")):
        run_dir = results_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        results = json.loads(results_path.read_text())
        config = json.loads(config_path.read_text())
        run_cfg = config.get("config", {})
        split_meta = _as_dict(config.get("split"))
        test_summary = _as_dict(results.get("test_summary"))
        test_state_summary = _as_dict(results.get("test_state_summary"))
        train_summary = _as_dict(results.get("train_summary"))
        train_state_summary = _as_dict(results.get("train_state_summary"))
        rel_parts = run_dir.relative_to(cv_root).parts
        setting_name = rel_parts[0] if len(rel_parts) >= 2 else run_dir.parent.name

        rows.append(
            {
                "run_dir": str(run_dir),
                "setting_name": setting_name,
                "control_mode": results.get("control_mode", config.get("control_mode", "unknown")),
                "split_strategy": split_meta.get("split_strategy"),
                "fold_index": split_meta.get("fold_index"),
                "split_label": (
                    ",".join(split_meta.get("test_wtas", []))
                    if split_meta.get("split_strategy") == "wta"
                    else (
                        f"fold_{int(split_meta.get('fold_index'))}"
                        if split_meta.get("fold_index") is not None
                        else run_dir.name
                    )
                ),
                "n_folds": split_meta.get("n_folds"),
                "seed": split_meta.get("seed"),
                "use_state_centroids": bool(results.get("use_state_centroids", config.get("use_state_centroids", False))),
                "shared_guide_embedding": bool(
                    results.get("shared_guide_embedding", config.get("shared_guide_embedding", False))
                ),
                "program_basis": (
                    "state_centroids"
                    if bool(results.get("use_state_centroids", config.get("use_state_centroids", False)))
                    else "learned"
                ),
                "ecological_growth": results.get("ecological_growth", config.get("ecological_growth")),
                "use_growth_intercept": results.get(
                    "use_growth_intercept",
                    config.get("use_growth_intercept", _nested_get(run_cfg, "model", "use_growth_intercept")),
                ),
                "training_schedule": results.get("training_schedule", config.get("training_schedule")),
                "stage_c_epochs": results.get(
                    "stage_c_epochs",
                    config.get("stage_c_epochs", _nested_get(run_cfg, "training", "stage_c_epochs")),
                ),
                "stage_d_epochs": results.get(
                    "stage_d_epochs",
                    config.get("stage_d_epochs", _nested_get(run_cfg, "training", "stage_d_epochs")),
                ),
                "epochs": results.get(
                    "epochs",
                    config.get("epochs", _nested_get(run_cfg, "training", "epochs")),
                ),
                "resolved_n_programs": results.get(
                    "resolved_n_programs",
                    config.get("resolved_n_programs", _nested_get(run_cfg, "model", "n_programs")),
                ),
                "effective_n_particles": results.get(
                    "effective_n_particles",
                    config.get(
                        "effective_n_particles",
                        _nested_get(run_cfg, "training", "n_particles"),
                    ),
                ),
                "n_steps": results.get(
                    "n_steps",
                    config.get("n_steps", _nested_get(run_cfg, "training", "n_steps")),
                ),
                "eval_particles": results.get(
                    "eval_particles",
                    config.get("eval_particles", _nested_get(run_cfg, "evaluation", "eval_particles")),
                ),
                "eval_steps": results.get(
                    "eval_steps",
                    config.get("eval_steps", _nested_get(run_cfg, "evaluation", "eval_steps")),
                ),
                "eval_target_particles": results.get(
                    "eval_target_particles",
                    config.get(
                        "eval_target_particles",
                        _nested_get(run_cfg, "evaluation", "eval_target_particles"),
                    ),
                ),
                "lambda_control_ref": results.get("lambda_control_ref", config.get("lambda_control_ref")),
                "control_ref_warmup_epochs": results.get("control_ref_warmup_epochs", config.get("control_ref_warmup_epochs")),
                "lambda_weak": results.get(
                    "lambda_weak",
                    config.get("lambda_weak", _nested_get(run_cfg, "training", "lambda_weak")),
                ),
                "lambda_reg_growth_bias": results.get(
                    "lambda_reg_growth_bias",
                    config.get(
                        "lambda_reg_growth_bias",
                        _nested_get(run_cfg, "training", "lambda_reg_growth_bias"),
                    ),
                ),
                "max_active_perturbations": results.get(
                    "max_active_perturbations",
                    config.get(
                        "max_active_perturbations",
                        _nested_get(run_cfg, "training", "max_active_perturbations"),
                    ),
                ),
                "embedding_dim": results.get(
                    "embedding_dim",
                    config.get("embedding_dim", _nested_get(run_cfg, "model", "embedding_dim")),
                ),
                "mediator_dim": results.get(
                    "mediator_dim",
                    config.get("mediator_dim", _nested_get(run_cfg, "model", "mediator_dim")),
                ),
                "hidden_dim": results.get(
                    "hidden_dim",
                    config.get("hidden_dim", _nested_get(run_cfg, "model", "hidden_dim")),
                ),
                "depth": results.get(
                    "depth",
                    config.get("depth", _nested_get(run_cfg, "model", "depth")),
                ),
                "n_supported_perturbations": results.get("n_supported_perturbations"),
                "train_time_s": results.get("train_time_s"),
                "train_peak_gpu_allocated_gb": _gpu_stat_gb(results.get("train_peak_gpu_mb"), "allocated_mb"),
                "train_peak_gpu_reserved_gb": _gpu_stat_gb(results.get("train_peak_gpu_mb"), "reserved_mb"),
                "eval_peak_gpu_allocated_gb": _gpu_stat_gb(results.get("eval_peak_gpu_mb"), "allocated_mb"),
                "eval_peak_gpu_reserved_gb": _gpu_stat_gb(results.get("eval_peak_gpu_mb"), "reserved_mb"),
                "train_mean_uot": train_summary.get("mean_uot"),
                "train_mass_rel_error": train_summary.get("mean_mass_rel_error"),
                "train_state_tv": train_state_summary.get("mean_state_tv"),
                "train_dom_acc": train_state_summary.get("dominant_state_accuracy"),
                "train_acc": train_state_summary.get("dominant_state_accuracy"),
                "train_expansion_gap": train_state_summary.get("mean_abs_expansion_ratio_gap"),
                "test_mean_uot": test_summary.get("mean_uot"),
                "test_mass_rel_error": test_summary.get("mean_mass_rel_error"),
                "test_state_tv": test_state_summary.get("mean_state_tv"),
                "test_dom_acc": test_state_summary.get("dominant_state_accuracy"),
                "test_acc": test_state_summary.get("dominant_state_accuracy"),
                "test_expansion_gap": test_state_summary.get("mean_abs_expansion_ratio_gap"),
            }
        )
    return rows


def normalize_ranking_mode(ranking_mode: str) -> str:
    if ranking_mode == "dominant_state":
        return "test_acc"
    return ranking_mode


def build_group_summary(df: pd.DataFrame, *, group_by: str, ranking_mode: str) -> pd.DataFrame:
    ranking_mode = normalize_ranking_mode(ranking_mode)
    group_key = "setting_name" if group_by == "setting" else "control_mode"
    summary = (
        df.groupby(group_key, dropna=False)
        .agg(
            control_mode=("control_mode", "first"),
            training_schedule=("training_schedule", "first"),
            ecological_growth=("ecological_growth", "first"),
            stage_c_epochs=("stage_c_epochs", "first"),
            stage_d_epochs=("stage_d_epochs", "first"),
            lambda_control_ref=("lambda_control_ref", "first"),
            lambda_weak=("lambda_weak", "first"),
            lambda_reg_growth_bias=("lambda_reg_growth_bias", "first"),
            max_active_perturbations=("max_active_perturbations", "first"),
            program_basis=("program_basis", "first"),
            shared_guide_embedding=("shared_guide_embedding", "first"),
            use_growth_intercept=("use_growth_intercept", "first"),
            embedding_dim=("embedding_dim", "first"),
            mediator_dim=("mediator_dim", "first"),
            hidden_dim=("hidden_dim", "first"),
            depth=("depth", "first"),
            resolved_n_programs=("resolved_n_programs", "first"),
            epochs=("epochs", "first"),
            effective_n_particles=("effective_n_particles", "first"),
            n_steps=("n_steps", "first"),
            eval_particles=("eval_particles", "first"),
            eval_steps=("eval_steps", "first"),
            eval_target_particles=("eval_target_particles", "first"),
            n_folds_completed=("run_dir", "count"),
            mean_test_uot=("test_mean_uot", "mean"),
            std_test_uot=("test_mean_uot", "std"),
            mean_test_mass_rel_error=("test_mass_rel_error", "mean"),
            std_test_mass_rel_error=("test_mass_rel_error", "std"),
            mean_test_state_tv=("test_state_tv", "mean"),
            std_test_state_tv=("test_state_tv", "std"),
            mean_test_dom_acc=("test_dom_acc", "mean"),
            std_test_dom_acc=("test_dom_acc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_expansion_gap=("test_expansion_gap", "mean"),
            std_test_expansion_gap=("test_expansion_gap", "std"),
            mean_train_time_s=("train_time_s", "mean"),
            total_train_time_s=("train_time_s", "sum"),
            mean_train_peak_gpu_allocated_gb=("train_peak_gpu_allocated_gb", "mean"),
            mean_train_peak_gpu_reserved_gb=("train_peak_gpu_reserved_gb", "mean"),
            mean_eval_peak_gpu_allocated_gb=("eval_peak_gpu_allocated_gb", "mean"),
            mean_eval_peak_gpu_reserved_gb=("eval_peak_gpu_reserved_gb", "mean"),
            mean_supported_perturbations=("n_supported_perturbations", "mean"),
        )
        .reset_index()
    )
    if ranking_mode == "test_acc":
        summary = summary.sort_values(
            [
                "mean_test_acc",
                "mean_test_state_tv",
                "mean_test_uot",
                "mean_test_mass_rel_error",
                "mean_test_expansion_gap",
            ],
            ascending=[False, True, True, True, True],
            na_position="last",
        )
    else:
        summary = summary.sort_values(
            ["mean_test_uot", "mean_test_state_tv", "mean_test_mass_rel_error"],
            ascending=[True, True, True],
            na_position="last",
        )
    summary = summary.reset_index(drop=True)
    return summary


def _fmt_float(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_int(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(int(value))


def write_markdown(
    summary_df: pd.DataFrame,
    per_fold_df: pd.DataFrame,
    out_path: Path,
    *,
    group_by: str,
    ranking_mode: str,
) -> None:
    ranking_mode = normalize_ranking_mode(ranking_mode)
    group_label = "Setting" if group_by == "setting" else "Control Mode"
    group_col = "setting_name" if group_by == "setting" else "control_mode"
    lines = [
        "# HNSCC CV Summary",
        "",
        f"Grouping: `{group_by}`",
        "",
        f"Ranking mode: `{ranking_mode}`",
        "",
        f"## By {group_label}",
        "",
        "| group | control_mode | basis | shared guide | schedule | ecology | growth intercept | stage C | stage D | epochs | lam_ctrl | lam_weak | growth_reg | hidden | depth | n_programs | particles | steps | eval particles | eval steps | active | folds | mean test UOT | std test UOT | mean test mass err | mean test state TV | mean test acc | std test acc | mean expansion gap | train peak GB | eval peak GB | mean train time (s) |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            "| "
            f"{getattr(row, group_col)} | {row.control_mode} | {row.program_basis} | {row.shared_guide_embedding} | "
            f"{row.training_schedule} | {row.ecological_growth} | {row.use_growth_intercept} | "
            f"{_fmt_int(row.stage_c_epochs)} | "
            f"{_fmt_int(row.stage_d_epochs)} | "
            f"{_fmt_int(row.epochs)} | "
            f"{_fmt_float(row.lambda_control_ref, 4)} | {_fmt_float(row.lambda_weak, 4)} | "
            f"{_fmt_float(row.lambda_reg_growth_bias, 4)} | "
            f"{_fmt_int(row.hidden_dim)} | "
            f"{_fmt_int(row.depth)} | "
            f"{_fmt_int(row.resolved_n_programs)} | "
            f"{_fmt_int(row.effective_n_particles)} | "
            f"{_fmt_int(row.n_steps)} | "
            f"{_fmt_int(row.eval_particles)} | "
            f"{_fmt_int(row.eval_steps)} | "
            f"{_fmt_int(row.max_active_perturbations)} | "
            f"{row.n_folds_completed} | "
            f"{_fmt_float(row.mean_test_uot)} | {_fmt_float(row.std_test_uot)} | "
            f"{_fmt_float(row.mean_test_mass_rel_error)} | {_fmt_float(row.mean_test_state_tv)} | "
            f"{_fmt_float(row.mean_test_acc)} | {_fmt_float(row.std_test_acc)} | "
            f"{_fmt_float(row.mean_test_expansion_gap)} | "
            f"{_fmt_float(row.mean_train_peak_gpu_allocated_gb, 1)} | "
            f"{_fmt_float(row.mean_eval_peak_gpu_allocated_gb, 1)} | "
            f"{_fmt_float(row.mean_train_time_s, 1)} |"
        )

    lines.extend(
        [
            "",
            "## Per Fold",
            "",
            "| group | control_mode | basis | shared guide | split | epochs | steps | test UOT | test mass err | test state TV | test acc | test expansion gap | train peak GB | run_dir |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    per_fold_sorted = per_fold_df.sort_values([group_col, "split_label", "run_dir"]).reset_index(drop=True)
    for row in per_fold_sorted.itertuples(index=False):
        split_value = row.split_label if pd.notna(row.split_label) else "n/a"
        lines.append(
            "| "
            f"{getattr(row, group_col)} | {row.control_mode} | {row.program_basis} | {row.shared_guide_embedding} | {split_value} | "
            f"{_fmt_int(row.epochs)} | {_fmt_int(row.n_steps)} | {_fmt_float(row.test_mean_uot)} | "
            f"{_fmt_float(row.test_mass_rel_error)} | {_fmt_float(row.test_state_tv)} | "
            f"{_fmt_float(row.test_acc)} | {_fmt_float(row.test_expansion_gap)} | "
            f"{_fmt_float(row.train_peak_gpu_allocated_gb, 1)} | `{row.run_dir}` |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    cv_root = Path(args.cv_root)
    output_dir = Path(args.output_dir) if args.output_dir else cv_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_run_rows(cv_root)
    if not rows:
        raise FileNotFoundError(f"No results_summary.json files found under {cv_root}")

    per_fold_df = pd.DataFrame(rows)
    summary_df = build_group_summary(
        per_fold_df,
        group_by=args.group_by,
        ranking_mode=args.ranking_mode,
    )

    per_fold_df.to_csv(output_dir / "cv_results.csv", index=False)
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)
    write_markdown(
        summary_df,
        per_fold_df,
        output_dir / "cv_summary.md",
        group_by=args.group_by,
        ranking_mode=args.ranking_mode,
    )

    print(output_dir / "cv_summary.md")


if __name__ == "__main__":
    main()
