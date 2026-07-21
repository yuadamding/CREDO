#!/usr/bin/env python3
"""Aggregate durable counterfactual rows by learned biological identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(run_dir: Path) -> pd.DataFrame:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata_path = Path(manifest["resolved_config"]["data"]["measure_meta"]).expanduser()
    metadata = pd.read_parquet(metadata_path)
    effects = pd.read_parquet(run_dir / "counterfactuals.parquet")
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
