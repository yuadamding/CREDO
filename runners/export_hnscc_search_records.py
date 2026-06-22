#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "src"))

from credo.search import (  # noqa: E402
    CREDOTrainOutput,
    CREDOTrialSpec,
    ProblemBuilderMetadata,
    claim_grade_thresholds,
    load_trial_records,
    metrics_from_history,
    reduce_trial_dirs,
    run_credo_trial,
    select_final_candidates,
    thresholds_for_profile,
    write_trial_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export completed HNSCC CREDO folds into credo.search trial records."
    )
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--profile",
        choices=["light_screen", "pareto_refit", "claim_grade", "ablation_only"],
        default="pareto_refit",
    )
    parser.add_argument("--required-folds", default="")
    parser.add_argument("--required-seeds", default="")
    parser.add_argument("--min-folds", type=int, default=0)
    parser.add_argument("--min-seeds", type=int, default=0)
    parser.add_argument("--objectives", default="")
    parser.add_argument("--sort-by", default="")
    parser.add_argument("--claim-control-null-max", type=float, default=None)
    parser.add_argument("--claim-log-mass-error-max", type=float, default=None)
    parser.add_argument("--claim-guide-concordance-max", type=float, default=None)
    parser.add_argument("--claim-require-guide-concordance", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _sha256_text(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_csv_values(raw: str) -> list[str]:
    out: list[str] = []
    for item in raw.replace(",", " ").split():
        text = item.strip()
        if text:
            out.append(text)
    return out


def _parse_seed_values(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_values(raw)]


def _fold_id(split_meta: dict[str, Any], run_dir: Path) -> str:
    fold_index = split_meta.get("fold_index")
    if fold_index is not None:
        return f"fold{int(fold_index):02d}"
    return run_dir.name


def _history_dict(path: Path) -> dict[str, list[Any]]:
    if not path.exists():
        return {}
    return pd.read_csv(path).to_dict(orient="list")


def _endpoint_eval_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    endpoint_col = "endpoint_geom_mass" if "endpoint_geom_mass" in df.columns else "uot"
    summary: dict[str, Any] = {
        "mean_endpoint_geom_mass": float(df[endpoint_col].mean()),
        "mean_mass_rel_error": float(df["mass_rel_error"].mean()) if "mass_rel_error" in df else math.nan,
        "validation_source": "held_out",
    }
    if {"mass_pred", "mass_true"}.issubset(df.columns):
        pred = pd.to_numeric(df["mass_pred"], errors="coerce").to_numpy(dtype=float)
        true = pd.to_numeric(df["mass_true"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(true) & (pred > 0.0) & (true > 0.0)
        if mask.any():
            log_residual = np.log(pred[mask]) - np.log(true[mask])
            summary["mean_log_mass_residual"] = float(np.mean(log_residual))
            summary["mean_abs_log_mass_residual"] = float(np.mean(np.abs(log_residual)))
            summary["mass_error_kind"] = "abs_log_residual"
            summary["mass_error_value"] = summary["mean_abs_log_mass_residual"]
    for key in (
        "terminal_ess_frac_min",
        "min_ess_frac_mean",
        "max_weight_frac_mean",
        "logw_range_max",
    ):
        if key in df:
            summary[key] = float(pd.to_numeric(df[key], errors="coerce").min() if "min" in key else pd.to_numeric(df[key], errors="coerce").max())
    return summary


def _counterfactual_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "delta_log_mass_fact_vs_ref" not in df:
        return {}
    out: dict[str, Any] = {}
    if "is_control" in df:
        control_mask = df["is_control"].map(_parse_bool)
        controls = df[control_mask]
    else:
        controls = df.iloc[0:0]
    if not controls.empty:
        values = pd.to_numeric(controls["delta_log_mass_fact_vs_ref"], errors="coerce").abs()
        finite = values[np.isfinite(values)]
        if not finite.empty:
            out["control_null_gap"] = float(finite.max())
            out["control_null_gap_kind"] = "absolute"
    return out


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _gpu_seconds(results: dict[str, Any]) -> float:
    value = results.get("train_time_s")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _spec_from_run(run_dir: Path, config: dict[str, Any], results: dict[str, Any]) -> CREDOTrialSpec:
    split_meta = dict(config.get("split") or {})
    cfg = dict(config.get("config") or {})
    model_cfg = dict(cfg.get("model") or {})
    training_cfg = dict(cfg.get("training") or {})
    simulation_cfg = dict(cfg.get("simulation") or {})
    eval_cfg = dict(cfg.get("eval") or {})
    latent_cfg = dict(cfg.get("latent") or {})
    seed = int(split_meta.get("seed", config.get("seed", results.get("seed", 0))) or 0)
    fold_id = _fold_id(split_meta, run_dir)
    return CREDOTrialSpec(
        dataset_kind="endpoint",
        claim_type="endpoint_reconstruction",
        split_type="random_cell",
        data_id="hnscc",
        seed=seed,
        fold_id=fold_id,
        latent_dim=int(latent_cfg.get("dim", config.get("latent_dim", 0)) or 0) or None,
        latent_source=config.get("latent_source") or latent_cfg.get("source"),
        latent_key=config.get("latent_key") or latent_cfg.get("key"),
        embedding_dim=int(config.get("embedding_dim", model_cfg.get("embedding_dim", 8))),
        n_programs=int(config.get("resolved_n_programs", model_cfg.get("n_programs", 8))),
        mediator_dim=int(config.get("mediator_dim", model_cfg.get("mediator_dim", 8))),
        hidden_dim=int(config.get("hidden_dim", model_cfg.get("hidden_dim", 128))),
        depth=int(config.get("depth", model_cfg.get("depth", 3))),
        context_kind=config.get("context_kind", model_cfg.get("context_kind", "mlp")),
        causal_token_dim=int(config.get("causal_token_dim", model_cfg.get("causal_token_dim", 64))),
        causal_heads=int(config.get("causal_heads", model_cfg.get("causal_heads", 4))),
        causal_n_mediators=int(config.get("causal_n_mediators", model_cfg.get("causal_n_mediators", 12))),
        causal_dropout=float(config.get("causal_dropout", model_cfg.get("causal_dropout", 0.05))),
        causal_mass_attention_temperature=float(
            config.get(
                "causal_mass_attention_temperature",
                model_cfg.get("causal_mass_attention_temperature", 0.5),
            )
        ),
        causal_growth_only=bool(config.get("causal_growth_only", model_cfg.get("causal_growth_only", True))),
        causal_sparse_edges=bool(config.get("causal_sparse_edges", model_cfg.get("causal_sparse_edges", True))),
        causal_residual_policy=config.get(
            "causal_residual_policy",
            model_cfg.get("causal_residual_policy", "edges_only"),
        ),
        ecological_growth=bool(config.get("ecological_growth", model_cfg.get("ecological_growth", True))),
        training_schedule=config.get("training_schedule", training_cfg.get("training_schedule", "staged")),
        epochs=int(config.get("epochs", training_cfg.get("epochs", results.get("epochs", 1)))),
        lr_net=float(config.get("lr_net", training_cfg.get("lr_net", 3e-4))),
        lr_embed=float(config.get("lr_embed", training_cfg.get("lr_embed", 1e-3))),
        lr_transformer=float(config.get("lr_transformer", training_cfg.get("lr_transformer", 5e-5))),
        lr_causal_attention=float(config.get("lr_causal_attention", training_cfg.get("lr_causal_attention", 5e-5))),
        weight_decay=float(config.get("weight_decay", training_cfg.get("weight_decay", 1e-6))),
        lambda_weak=float(config.get("lambda_weak", training_cfg.get("lambda_weak", 0.1))),
        lambda_count=float(training_cfg.get("lambda_count", 0.0)),
        lambda_reg_growth_bias=float(
            config.get("lambda_reg_growth_bias", training_cfg.get("lambda_reg_growth_bias", 1e-4))
        ),
        sinkhorn_epsilon=float(config.get("sinkhorn_epsilon", training_cfg.get("sinkhorn_epsilon", 0.1))),
        sinkhorn_tau=float(config.get("sinkhorn_tau", training_cfg.get("sinkhorn_tau", 1.0))),
        n_particles=int(config.get("effective_n_particles", simulation_cfg.get("n_particles", 128))),
        n_steps=int(config.get("n_steps", simulation_cfg.get("n_steps", 24))),
        eval_particles=int(config.get("eval_particles", eval_cfg.get("n_eval_particles", 384))),
    )


def _builder_metadata(
    run_dir: Path,
    config: dict[str, Any],
    *,
    required_folds: list[str],
    required_seeds: list[int],
) -> ProblemBuilderMetadata:
    split_meta = dict(config.get("split") or {})
    fold_grid = required_folds or [
        f"fold{idx:02d}" for idx in range(int(split_meta.get("n_folds", 0) or 0))
    ]
    split_identity = {
        "split_strategy": split_meta.get("split_strategy"),
        "n_folds": split_meta.get("n_folds"),
        "stratify_cols": split_meta.get("stratify_cols"),
        "fold_grid": fold_grid,
    }
    split_hash = _sha256_text(split_identity)
    gene_path = run_dir / "expression_genes.txt"
    return ProblemBuilderMetadata(
        builder_name="hnscc_full_pipeline_exporter",
        builder_version="1",
        data_path_hash=_sha256_text(config.get("data_path")),
        mass_table_hash=_sha256_text({"mass_scope": config.get("mass_scope"), "mass_mode": config.get("mass_mode")}),
        split_file_hash=split_hash,
        fold_assignment_hash=split_hash,
        latent_source=config.get("latent_source"),
        latent_key=config.get("latent_key") or config.get("config", {}).get("latent", {}).get("key"),
        gene_panel_hash=_sha256_file(gene_path),
        normalization_hash=_sha256_text(config.get("expression_latent", {}).get("target_sum")),
        hvg_preprocessing_hash=config.get("expression_split_manifest_hash"),
        representation_config_sha256=_sha256_text(config.get("expression_latent", {})),
        fold_grid_sha256=_sha256_text(fold_grid),
        seed_grid=",".join(str(seed) for seed in required_seeds) if required_seeds else None,
        split_manifest_sha256=split_hash,
    )


def _thresholds(args: argparse.Namespace):
    if args.profile == "claim_grade":
        if args.claim_control_null_max is None or args.claim_log_mass_error_max is None:
            raise ValueError(
                "claim_grade export requires --claim-control-null-max and "
                "--claim-log-mass-error-max."
            )
        return claim_grade_thresholds(
            control_null_max=args.claim_control_null_max,
            log_mass_error_max=args.claim_log_mass_error_max,
            guide_concordance_max=args.claim_guide_concordance_max,
            require_guide_concordance=args.claim_require_guide_concordance,
        )
    return thresholds_for_profile(args.profile)


def main() -> None:
    args = parse_args()
    cv_root = Path(args.cv_root)
    output_dir = Path(args.output_dir) if args.output_dir else cv_root / "search"
    trials_dir = output_dir / "trials"
    if args.overwrite and trials_dir.exists():
        shutil.rmtree(trials_dir)
    trials_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_folds = _parse_csv_values(args.required_folds)
    required_seeds = _parse_seed_values(args.required_seeds)
    thresholds = _thresholds(args)
    results_paths = sorted(cv_root.rglob("results_summary.json"))
    written = []
    for index, results_path in enumerate(results_paths):
        run_dir = results_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = _read_json(config_path)
        results = _read_json(results_path)
        spec = _spec_from_run(run_dir, config, results)
        endpoint_summary = _endpoint_eval_summary(run_dir / "test_endpoint_metrics.csv")
        endpoint_summary.update(_counterfactual_summary(run_dir / "biology" / "counterfactual_biology_effects.csv"))
        history = _history_dict(run_dir / "training_history.csv")
        metrics = metrics_from_history(
            history,
            eval_summary=endpoint_summary,
            gpu_seconds=_gpu_seconds(results),
            wall_seconds=_gpu_seconds(results),
            converged=True,
            diverged=False,
        )
        output = CREDOTrainOutput(
            metrics=metrics,
            run_dir=str(run_dir),
            checkpoint_path=str(run_dir / "checkpoint_best_ema.pt"),
            history_path=str(run_dir / "training_history.csv"),
            eval_summary_path=str(run_dir / "test_endpoint_metrics.csv"),
            resolved_config_path=str(config_path),
            builder_metadata=_builder_metadata(
                run_dir,
                config,
                required_folds=required_folds,
                required_seeds=required_seeds,
            ),
        )
        result = run_credo_trial(
            spec,
            train_fn=lambda _cfg, _spec, _reporter, out=output: out,
            output_dir=str(run_dir),
            thresholds=thresholds,
        )
        trial_dir = write_trial_dir(
            trials_dir,
            result,
            index=index,
            trial_id=f"{spec.fold_id}_seed{spec.seed}_{run_dir.name}",
            overwrite=args.overwrite,
        )
        written.append(trial_dir)

    jsonl = reduce_trial_dirs(trials_dir, output_dir / "trials.jsonl")
    records = load_trial_records(jsonl)
    objectives = _parse_csv_values(args.objectives)
    select_kwargs: dict[str, Any] = {
        "profile": args.profile,
        "aggregate_by": ("fold_id", "seed"),
        "objectives": objectives or None,
        "sort_by": args.sort_by or None,
        "require_feasible": True,
        "require_heldout": True,
    }
    if args.min_folds > 0:
        select_kwargs["min_folds"] = args.min_folds
    if args.min_seeds > 0:
        select_kwargs["min_seeds"] = args.min_seeds
    if required_folds:
        select_kwargs["required_folds"] = required_folds
    if required_seeds:
        select_kwargs["required_seeds"] = required_seeds

    try:
        candidates = select_final_candidates(records, **select_kwargs)
    except ValueError as exc:
        candidates = []
        (output_dir / "candidate_selection_error.txt").write_text(str(exc) + "\n", encoding="utf-8")

    (output_dir / "final_candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(candidates).to_csv(output_dir / "final_candidates.csv", index=False)
    manifest = {
        "cv_root": str(cv_root),
        "profile": args.profile,
        "n_trial_records": len(records),
        "n_trial_dirs_written": len(written),
        "n_final_candidates": len(candidates),
        "required_folds": required_folds,
        "required_seeds": required_seeds,
        "trials_jsonl": str(jsonl),
    }
    (output_dir / "search_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
