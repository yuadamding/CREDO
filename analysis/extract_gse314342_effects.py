"""Aggregate GSE314342 trajectory outputs without collapsing guide or donor evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--measure-manifest", required=True)
    parser.add_argument("--rest-effects", default="")
    parser.add_argument("--output-dir", default="results/gse314342")
    return parser.parse_args(argv)


def _read(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    return pd.read_parquet(source) if source.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(source)


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "tau",
        "normalized_tau",
        "physical_time",
        "source_physical_time",
        "interval_physical_duration",
        "n_source_cells",
        "n_target_cells",
    }
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def _gene_aggregate(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    metrics = _metric_columns(frame)
    grouped = frame.groupby(["target_gene", time_col], observed=True)
    median = grouped[metrics].median().add_suffix("_median")
    mean = grouped[metrics].mean().add_suffix("_mean")
    counts = grouped.agg(n_guide_views=("guide_id", "nunique"), n_donors=("sample_id", "nunique"))
    return pd.concat([counts, median, mean], axis=1).reset_index()


def _rest_gene_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = _metric_columns(frame)
    grouped = frame.groupby("target_gene", observed=True)
    counts = grouped.agg(
        n_guide_views=("guide_id", "nunique"),
        n_donors=("sample_id", "nunique"),
    )
    return pd.concat(
        [counts, grouped[metrics].median().add_suffix("_median"), grouped[metrics].mean().add_suffix("_mean")],
        axis=1,
    ).reset_index()


def _concordance(frame: pd.DataFrame, metric: str, time_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    guide_rows: list[dict[str, object]] = []
    donor_rows: list[dict[str, object]] = []
    for (gene, time_label), group in frame.groupby(["target_gene", time_col], observed=True):
        guide_matrix = group.pivot_table(index="sample_id", columns="guide_id", values=metric, aggfunc="mean")
        guide_corr = guide_matrix.corr().to_numpy()
        guide_upper = guide_corr[np.triu_indices_from(guide_corr, k=1)] if guide_corr.size else np.array([])
        guide_rows.append(
            {
                "target_gene": gene,
                time_col: time_label,
                "metric": metric,
                "n_guides": int(guide_matrix.shape[1]),
                "cross_guide_correlation": float(np.nanmean(guide_upper)) if guide_upper.size else np.nan,
                "cross_guide_sign_agreement": float(
                    group.groupby("guide_id", observed=True)[metric].mean().pipe(
                        lambda values: max((values >= 0).mean(), (values <= 0).mean())
                    )
                ),
            }
        )
        donor_effects = group.groupby("sample_id", observed=True)[metric].mean()
        donor_rows.append(
            {
                "target_gene": gene,
                time_col: time_label,
                "metric": metric,
                "n_donors": int(len(donor_effects)),
                "cross_donor_sign_agreement": float(
                    max((donor_effects >= 0).mean(), (donor_effects <= 0).mean())
                ),
                "donor_effect_sd": float(donor_effects.std(ddof=1)) if len(donor_effects) > 1 else np.nan,
            }
        )
    return pd.DataFrame(guide_rows), pd.DataFrame(donor_rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    predicted = _read(args.predicted)
    counterfactual = _read(args.counterfactual)
    manifest = _read(args.measure_manifest)
    required_manifest = {"sample_id", "guide_id", "target_gene", "is_control"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise KeyError(f"Measure manifest is missing columns: {sorted(missing)}")
    metadata_cols = [
        column
        for column in [
            "sample_id",
            "guide_id",
            "target_gene",
            "embedding_id",
            "is_control",
            "guide_effective_rest",
            "guide_effective_8h",
            "guide_effective_48h",
            "offtarget_flag",
        ]
        if column in manifest.columns
    ]
    metadata = manifest[metadata_cols].drop_duplicates(["sample_id", "guide_id"])

    if "target_label" in counterfactual.columns and "time_label" not in counterfactual.columns:
        counterfactual = counterfactual.rename(columns={"target_label": "time_label"})
    for name, frame in (("predicted", predicted), ("counterfactual", counterfactual)):
        if "guide_id" not in frame.columns or "sample_id" not in frame.columns:
            raise KeyError(f"{name} metrics require sample_id and guide_id output fields.")
    predicted = predicted.merge(metadata, on=["sample_id", "guide_id"], how="left", suffixes=("", "_manifest"))
    counterfactual = counterfactual.merge(
        metadata,
        on=["sample_id", "guide_id"],
        how="left",
        suffixes=("", "_manifest"),
    )
    for frame in (predicted, counterfactual):
        if "target_gene_manifest" in frame:
            frame["target_gene"] = frame.get("target_gene", frame["target_gene_manifest"]).fillna(
                frame["target_gene_manifest"]
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(out / "predicted_metrics_by_guide_time.csv", index=False)
    counterfactual.to_csv(out / "counterfactual_metrics_by_guide_time.csv", index=False)
    _gene_aggregate(counterfactual, "time_label").to_csv(
        out / "counterfactual_metrics_by_gene_time.csv", index=False
    )

    primary_metric = next(
        (
            metric
            for metric in [
                "delta_log_mass_fact_vs_ref",
                "energy_distance_fact_vs_ref",
                "weighted_mean_shift_l2_fact_vs_ref",
            ]
            if metric in counterfactual.columns
        ),
        None,
    )
    if primary_metric is None:
        raise KeyError("Counterfactual table has no recognized CREDO effect metric.")
    guide, donor = _concordance(counterfactual, primary_metric, "time_label")
    guide.to_csv(out / "guide_concordance.csv", index=False)
    donor.to_csv(out / "donor_concordance.csv", index=False)
    if args.rest_effects:
        rest = _read(args.rest_effects)
        required_rest = {"guide_id", "target_gene", "sample_id"}
        missing_rest = required_rest - set(rest.columns)
        if missing_rest:
            raise KeyError(f"Rest effects are missing columns: {sorted(missing_rest)}")
        rest.to_csv(out / "rest_baseline_priming_by_guide.csv", index=False)
        _rest_gene_aggregate(rest).to_csv(
            out / "rest_baseline_priming_by_gene.csv",
            index=False,
        )
    counterfactual[counterfactual["is_control"].astype(bool)].to_csv(
        out / "control_null.csv", index=False
    )
    ineffective_col = next(
        (column for column in ["guide_effective_8h", "guide_effective_48h"] if column in counterfactual),
        None,
    )
    ineffective = (
        counterfactual[~counterfactual[ineffective_col].astype(bool)]
        if ineffective_col is not None
        else counterfactual.iloc[0:0]
    )
    ineffective.to_csv(out / "ineffective_guide_null.csv", index=False)

    claim_report = {
        "cohort": "GSE314342",
        "claim_ready": False,
        "primary_metric": primary_metric,
        "n_guides": int(counterfactual["guide_id"].nunique()),
        "n_target_genes": int(counterfactual["target_gene"].nunique()),
        "n_donors": int(counterfactual["sample_id"].nunique()),
        "time_labels": sorted(counterfactual["time_label"].astype(str).unique().tolist()),
        "completed_products": sorted(path.name for path in out.iterdir()),
        "missing_claim_gates": [
            "leave_one_guide_out",
            "leave_one_donor_out",
            "held_out_8h",
            "external_bulk_validation",
            "numerical_convergence",
        ],
        "claim_boundary": (
            "Dynamic effects compare target-gene and shared NTC-reference dynamics from the same "
            "already perturbed Rest population; pre-Rest perturbation effects are excluded."
        ),
    }
    (out / "claim_report.json").write_text(
        json.dumps(claim_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
