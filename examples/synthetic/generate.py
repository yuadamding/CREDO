#!/usr/bin/env python3
"""Generate a tiny three-timepoint cohort using the canonical CREDO contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from credo.contracts import Axis, MassSemantics
from credo.io import write_canonical_dataset


def generate(output_dir: Path, seed: int = 0) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    donors = ("D1", "D2")
    guides = (
        ("NTC-1", "__control__", "__control__", True),
        ("NTC-2", "__control__", "__control__", True),
        ("GENE1-1", "GENE1", "GENE1", False),
        ("GENE1-2", "GENE1", "GENE1", False),
        ("GENE2-1", "GENE2", "GENE2", False),
        ("GENE2-2", "GENE2", "GENE2", False),
    )
    axis = Axis(
        kind="physical",
        source="Rest",
        labels=("Rest", "Stim8hr", "Stim48hr"),
        values=(0.0, 8.0, 48.0),
    )
    metadata_rows = []
    support_rows = []
    latent_rows = []
    mass_rows = []
    source_mass: dict[str, float] = {}
    downstream_mass: dict[tuple[str, str], float] = {}
    for donor_index, donor in enumerate(donors):
        raw_source = rng.gamma(shape=3.0, scale=1.0, size=len(guides))
        raw_source /= raw_source.sum()
        for guide_index, (guide, embedding, target, is_control) in enumerate(guides):
            measure_id = f"{donor}::{guide}"
            metadata_rows.append(
                {
                    "measure_id": measure_id,
                    "sample_id": donor,
                    "perturbation_id": guide,
                    "guide_id": guide,
                    "embedding_id": embedding,
                    "target_gene": target,
                    "context_group_id": donor,
                    "is_control": is_control,
                }
            )
            source_mass[measure_id] = float(raw_source[guide_index])
            base = rng.normal(size=4) + donor_index * 0.15
            for time_index, label in enumerate(axis.labels):
                growth = 0.0 if is_control else (0.08 if embedding == "GENE1" else -0.05)
                mass = source_mass[measure_id] * np.exp(growth * time_index)
                downstream_mass[(measure_id, label)] = float(mass)
                if donor == "D2" and guide == "GENE2-2" and label == "Stim48hr":
                    continue
                mass_rows.append(
                    {
                        "measure_id": measure_id,
                        "time_label": label,
                        "mass": mass,
                        "denominator": f"{donor}::{label}::eligible_guides",
                    }
                )
                shift = np.zeros(4)
                if embedding == "GENE1":
                    shift[0] = 0.25 * time_index
                elif embedding == "GENE2":
                    shift[1] = -0.2 * time_index
                for _ in range(8):
                    support_rows.append(
                        {
                            "measure_id": measure_id,
                            "time_label": label,
                            "atom_weight": 1.0,
                        }
                    )
                    latent_rows.append(base + shift + rng.normal(scale=0.3, size=4))

    count_rows = []
    for donor in donors:
        donor_ids = [row["measure_id"] for row in metadata_rows if row["sample_id"] == donor]
        for label in axis.labels[1:]:
            probabilities = np.asarray(
                [
                    downstream_mass.get((measure_id, label), source_mass[measure_id])
                    for measure_id in donor_ids
                ]
            )
            probabilities /= probabilities.sum()
            counts = rng.multinomial(2_000, probabilities)
            for measure_id, count in zip(donor_ids, counts, strict=False):
                count_rows.append(
                    {
                        "context_group_id": donor,
                        "time_label": label,
                        "measure_id": measure_id,
                        "exposure": source_mass[measure_id],
                        "count": int(count),
                    }
                )

    support_obs = pd.DataFrame(support_rows)
    support_obs.index = pd.Index(
        [f"cell_{index}" for index in range(len(support_obs))], dtype=object
    )
    support = ad.AnnData(
        X=np.empty((len(support_rows), 0), dtype=np.float32),
        obs=support_obs,
    )
    support.obsm["X_credo"] = np.asarray(latent_rows, dtype=np.float32)
    denominator_totals = {
        (donor, label): sum(downstream_mass[(f"{donor}::{guide}", label)] for guide, *_ in guides)
        for donor in donors
        for label in axis.labels
    }
    donor_by_measure = {row["measure_id"]: row["sample_id"] for row in metadata_rows}
    masses = pd.DataFrame(mass_rows)
    masses["mass"] = [
        row.mass / denominator_totals[(donor_by_measure[row.measure_id], row.time_label)]
        for row in masses.itertuples(index=False)
    ]
    return write_canonical_dataset(
        output_dir,
        support=support,
        measure_meta=pd.DataFrame(metadata_rows),
        masses=masses,
        counts=pd.DataFrame(count_rows),
        axis=axis,
        mass_semantics=MassSemantics.RELATIVE_WITHIN_GROUP,
        description="Tiny GSE-like deterministic CREDO fixture",
        source={"generator": "examples/synthetic/generate.py", "seed": seed},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = generate(args.output_dir, seed=args.seed)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
