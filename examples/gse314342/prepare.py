#!/usr/bin/env python3
"""Adapt the downloaded GSE314342 support release to canonical CREDO files."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from credo.contracts import Axis, MassSemantics
from credo.io import write_canonical_dataset


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("GSE314342 is_control contains nonboolean values.")
    return normalized.isin({"true", "1"})


def prepare(source_dir: Path, output_dir: Path, *, pilot: bool) -> dict[str, Path]:
    prefix = "pilot_d3_" if pilot else ""
    support_name = "gse314342_credo_pilot_d3.h5ad" if pilot else "gse314342_credo_support.h5ad"
    support_source = ad.read_h5ad(source_dir / support_name)
    required_obs = {
        "sample_id",
        "guide_id",
        "time_label",
        "embedding_id",
        "target_gene",
        "is_control",
    }
    if missing := required_obs - set(support_source.obs.columns):
        raise ValueError(f"Downloaded support is missing columns: {sorted(missing)}")
    obs = support_source.obs.copy().reset_index(drop=True)
    obs["measure_id"] = obs["sample_id"].astype(str) + "::" + obs["guide_id"].astype(str)
    all_measure_ids = set(obs["measure_id"])
    source_measure_ids = set(obs.loc[obs["time_label"].astype(str).eq("Rest"), "measure_id"])
    retained = obs["measure_id"].isin(source_measure_ids).to_numpy()
    obs = obs.loc[retained].copy()
    support_obs = obs[["measure_id", "time_label"]].assign(atom_weight=1.0)
    support_obs.index = pd.Index(
        [f"atom_{index}" for index in range(len(support_obs))], dtype=object
    )
    support = ad.AnnData(
        X=np.empty((len(obs), 0), dtype=np.float32),
        obs=support_obs,
    )
    support.obsm["X_credo"] = np.asarray(support_source.obsm["X_credo"], dtype=np.float32)[retained]

    manifest = pd.read_csv(source_dir / f"{prefix}measure_manifest.csv")
    manifest["measure_id"] = (
        manifest["sample_id"].astype(str) + "::" + manifest["guide_id"].astype(str)
    )
    observed_ids = set(obs["measure_id"])
    manifest = manifest[manifest["measure_id"].isin(observed_ids)].copy()
    manifest["is_control"] = _as_bool(manifest["is_control"])
    manifest["perturbation_id"] = manifest["guide_id"].astype(str)
    measure_meta = manifest[
        [
            "measure_id",
            "sample_id",
            "perturbation_id",
            "guide_id",
            "embedding_id",
            "target_gene",
            "context_group_id",
            "is_control",
        ]
    ].drop_duplicates("measure_id")
    if set(measure_meta["measure_id"]) != observed_ids:
        raise ValueError("GSE314342 measure manifest does not cover every support measure.")

    masses = pd.read_csv(source_dir / f"{prefix}guide_counts_and_masses.csv")
    masses["measure_id"] = masses["sample_id"].astype(str) + "::" + masses["guide_id"].astype(str)
    masses["denominator"] = (
        masses["sample_id"].astype(str)
        + "::"
        + masses["time_label"].astype(str)
        + "::eligible_guides"
    )
    observed_pairs = set(zip(obs["measure_id"], obs["time_label"], strict=False))
    masses = masses[["measure_id", "time_label", "mass_value", "denominator"]].rename(
        columns={"mass_value": "mass"}
    )
    masses = masses[
        [
            pair in observed_pairs
            for pair in zip(masses["measure_id"], masses["time_label"], strict=False)
        ]
    ]

    counts = pd.read_csv(source_dir / f"{prefix}guide_count_blocks.csv")
    counts["measure_id"] = (
        counts["context_group_id"].astype(str) + "::" + counts["guide_id"].astype(str)
    )
    counts = counts[counts["measure_id"].isin(observed_ids)][
        ["context_group_id", "time_label", "measure_id", "exposure", "count"]
    ]
    axis = Axis(
        kind="physical",
        source="Rest",
        labels=("Rest", "Stim8hr", "Stim48hr"),
        values=(0.0, 8.0, 48.0),
    )
    return write_canonical_dataset(
        output_dir,
        support=support,
        measure_meta=measure_meta,
        masses=masses,
        counts=counts,
        axis=axis,
        mass_semantics=MassSemantics.RELATIVE_WITHIN_GROUP,
        description="GSE314342 primary human CD4+ T-cell Perturb-seq",
        source={
            "accession": "GSE314342",
            "support_file": support_name,
            "pilot": pilot,
            "source_supported_measure_count": len(source_measure_ids),
            "excluded_without_source_support": len(all_measure_ids - source_measure_ids),
            "late_time_resolution": "Stim48hr follows processed author metadata",
        },
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    default_source = repository.parent / "inputs" / "GSE314342"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.source_dir / ("canonical_pilot" if args.pilot else "canonical")
    paths = prepare(args.source_dir, output, pilot=args.pilot)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
