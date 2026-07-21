#!/usr/bin/env python3
"""Adapt the HNSCC P4/P60 AnnData file to the canonical CREDO contract."""

from __future__ import annotations

import argparse
import hashlib
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
        raise ValueError("HNSCC boolean annotation contains unsupported values.")
    return normalized.isin({"true", "1"})


def _stable_subsample(positions: np.ndarray, cap: int, key: str, seed: int) -> np.ndarray:
    if len(positions) <= cap:
        return positions
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    return np.sort(rng.choice(positions, size=cap, replace=False))


def prepare(
    input_path: Path,
    output_dir: Path,
    *,
    latent_key: str,
    support_cap: int,
    seed: int,
) -> dict[str, Path]:
    source = ad.read_h5ad(input_path, backed="r")
    required = {"timepoint", "guide_id", "target_gene", "is_control", "guide_confident"}
    if missing := required - set(source.obs.columns):
        raise ValueError(f"HNSCC obs is missing columns: {sorted(missing)}")
    if latent_key not in source.obsm:
        raise ValueError(f"HNSCC input is missing obsm[{latent_key!r}].")
    obs = source.obs.copy().reset_index(drop=True)
    confident = _as_bool(obs["guide_confident"])
    obs = obs[confident & obs["timepoint"].astype(str).isin({"P4", "P60"})].copy()
    obs["guide_id"] = obs["guide_id"].astype(str)
    obs["timepoint"] = obs["timepoint"].astype(str)
    support_by_guide = (
        obs.groupby(["guide_id", "timepoint"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["P4", "P60"], fill_value=0)
    )
    complete_guides = set(support_by_guide.index[support_by_guide.gt(0).all(axis=1)])
    obs = obs[obs["guide_id"].isin(complete_guides)].copy()
    original_positions = obs.index.to_numpy(dtype=np.int64)
    latent_all = np.asarray(source.obsm[latent_key], dtype=np.float32)

    selected_positions = []
    mass_rows = []
    for (time_label, guide_id), rows in obs.groupby(
        ["timepoint", "guide_id"], observed=True, sort=False
    ):
        positions = rows.index.to_numpy(dtype=np.int64)
        mass_rows.append(
            {
                "measure_id": guide_id,
                "time_label": str(time_label),
                "mass": len(positions),
                "denominator": f"__global__::{time_label}::captured_cells",
            }
        )
        selected_positions.extend(
            _stable_subsample(positions, support_cap, f"{time_label}:{guide_id}", seed).tolist()
        )
    selected_positions = np.asarray(sorted(selected_positions), dtype=np.int64)
    selected_obs = obs.loc[selected_positions]
    support_obs = pd.DataFrame(
        {
            "measure_id": selected_obs["guide_id"].astype(str).to_numpy(),
            "time_label": selected_obs["timepoint"].astype(str).to_numpy(),
            "atom_weight": np.ones(len(selected_positions)),
        }
    )
    support_obs.index = pd.Index(
        [f"atom_{index}" for index in range(len(support_obs))], dtype=object
    )
    support = ad.AnnData(
        X=np.empty((len(selected_positions), 0), dtype=np.float32),
        obs=support_obs,
    )
    support.obsm["X_credo"] = latent_all[selected_positions]

    metadata_rows = []
    for guide_id, rows in obs.groupby("guide_id", observed=True, sort=True):
        control_values = _as_bool(rows["is_control"]).unique()
        target_values = rows["target_gene"].dropna().astype(str).unique()
        if len(control_values) != 1 or len(target_values) != 1:
            raise ValueError(f"Inconsistent identity annotations for guide {guide_id!r}.")
        is_control = bool(control_values[0])
        target = "__control__" if is_control else target_values[0]
        metadata_rows.append(
            {
                "measure_id": str(guide_id),
                "sample_id": "__pooled__",
                "perturbation_id": str(guide_id),
                "guide_id": str(guide_id),
                "embedding_id": target,
                "target_gene": target,
                "context_group_id": "__global__",
                "is_control": is_control,
            }
        )
    axis = Axis(kind="physical", source="P4", labels=("P4", "P60"), values=(4.0, 60.0))
    return write_canonical_dataset(
        output_dir,
        support=support,
        measure_meta=pd.DataFrame(metadata_rows),
        masses=pd.DataFrame(mass_rows),
        axis=axis,
        mass_semantics=MassSemantics.CAPTURED_COUNT,
        description="HNSCC P4/P60 guide-level endpoint cohort",
        source={
            "input": str(input_path),
            "latent_key": latent_key,
            "guide_confident_only": True,
            "support_cap": support_cap,
            "original_selected_rows": int(len(original_positions)),
        },
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    default_input = (
        repository.parent
        / "inputs"
        / "hnscc"
        / "GSE235325_P4P60_allgenes_allcells_latest_states.h5ad"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument(
        "--output-dir", type=Path, default=repository.parent / "inputs" / "hnscc" / "canonical"
    )
    parser.add_argument("--latent-key", default="X_pca_latest_sct")
    parser.add_argument("--support-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = prepare(
        args.input,
        args.output_dir,
        latent_key=args.latent_key,
        support_cap=args.support_cap,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
