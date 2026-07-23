#!/usr/bin/env python3
"""Aggregate durable counterfactual rows by learned biological identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _measure_metadata(manifest: dict) -> pd.DataFrame:
    resolved = manifest["resolved_config"]
    if resolved.get("data") is not None:
        path = Path(resolved["data"]["measure_meta"]).expanduser()
        return pd.read_parquet(path)

    from credo import get_recipe, open_study
    from credo.data.splits import (
        SplitPlan,
        validate_representation_scope,
        validate_split_plan,
    )
    from credo.io import RunConfig

    config = RunConfig.model_validate(resolved)
    study = open_study(config.study, verify="semantic")
    try:
        view = config.view(study)
        recipe = get_recipe(config.recipe)
        split = (
            SplitPlan.from_dict(manifest["split_plan"])
            if manifest.get("split_plan") is not None
            else recipe.plan_split(view, config.recipe_config)
        )
        validate_split_plan(view, split)
        validate_representation_scope(view, split)
        compiled = recipe.compile_study(view, split, config.recipe_config)
        return compiled.measure_meta.copy()
    finally:
        study.close()


def summarize(run_dir: Path) -> pd.DataFrame:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metadata = _measure_metadata(manifest)
    effects = pd.read_parquet(run_dir / manifest["outputs"]["counterfactuals"])
    if effects.empty:
        return pd.DataFrame(
            columns=[
                "embedding_id",
                "target_gene",
                "time_label",
                "context_policy",
                "n_measures",
                "n_guides",
                "n_samples",
                "median_delta_log_mass",
                "median_mean_shift_l2",
                "median_energy_distance",
            ]
        )
    for column in ("target_gene", "guide_id"):
        if column not in metadata:
            metadata[column] = None
    joined = effects.merge(
        metadata[["measure_id", "embedding_id", "target_gene", "guide_id", "sample_id"]],
        on="measure_id",
        how="left",
        validate="many_to_one",
    )
    grouped = joined.groupby(
        ["embedding_id", "target_gene", "time_label", "context_policy"],
        observed=True,
        dropna=False,
    )
    return grouped.agg(
        n_measures=("measure_id", "nunique"),
        n_guides=("guide_id", "nunique"),
        n_samples=("sample_id", "nunique"),
        median_delta_log_mass=("delta_log_mass", "median"),
        median_mean_shift_l2=("mean_shift_l2", "median"),
        median_energy_distance=("energy_distance", "median"),
    ).reset_index()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        args.run_dir.parent / "analysis" / f"{args.run_dir.name}_counterfactual_summary.parquet"
    )
    frame = summarize(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
