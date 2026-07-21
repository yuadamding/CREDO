#!/usr/bin/env python3
"""Adapt the longitudinal LPS cohort to canonical CREDO files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from credo.contracts import Axis, MassSemantics
from credo.io import write_canonical_dataset


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("LPS is_control contains nonboolean values.")
    return normalized.isin({"true", "1"})


def _stable_subsample(positions: np.ndarray, cap: int, key: str, seed: int) -> np.ndarray:
    if cap <= 0 or len(positions) <= cap:
        return positions
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    return np.sort(rng.choice(positions, size=cap, replace=False))


def _svd_latent(
    source: ad.AnnData,
    positions: np.ndarray,
    *,
    counts_layer: str,
    latent_dim: int,
    hvg_count: int,
    seed: int,
) -> np.ndarray:
    subset = source[positions].to_memory()
    if counts_layer == "X":
        matrix = subset.X
    elif counts_layer in subset.layers:
        matrix = subset.layers[counts_layer]
    else:
        raise ValueError(f"LPS input has no counts layer {counts_layer!r}.")
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if np.any(totals <= 0):
        raise ValueError("LPS support contains cells with zero library size.")
    normalized = sp.diags(10_000.0 / totals) @ matrix
    normalized = normalized.tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).reshape(-1)
    second = np.asarray(normalized.power(2).mean(axis=0)).reshape(-1)
    score = np.maximum(second - np.square(mean), 0.0) / np.maximum(mean, 1e-8)
    keep = np.argsort(score)[::-1][: min(hvg_count, normalized.shape[1])]
    matrix_hvg = normalized[:, np.sort(keep)]
    rank = min(latent_dim, min(matrix_hvg.shape) - 1)
    if rank < 2:
        raise ValueError("LPS support is too small for the requested latent representation.")
    left, singular, right = svds(matrix_hvg, k=rank, which="LM", random_state=seed)
    order = np.argsort(singular)[::-1]
    left = left[:, order]
    singular = singular[order]
    right = right[order]
    for component in range(rank):
        anchor = int(np.argmax(np.abs(right[component])))
        if right[component, anchor] < 0:
            left[:, component] *= -1
    latent = left * singular
    latent -= latent.mean(axis=0, keepdims=True)
    latent /= np.maximum(latent.std(axis=0, keepdims=True), 1e-6)
    return latent.astype(np.float32)


def prepare(
    input_path: Path,
    output_dir: Path,
    *,
    latent_key: str,
    counts_layer: str,
    latent_dim: int,
    hvg_count: int,
    support_cap: int,
    seed: int,
) -> dict[str, Path]:
    if latent_dim < 2 or hvg_count < 2 or support_cap < 2:
        raise ValueError("latent_dim, hvg_count, and support_cap must each be at least 2.")
    source = ad.read_h5ad(input_path, backed="r")
    required = {"time_label", "sample_id", "perturbation_id", "is_control"}
    if missing := required - set(source.obs.columns):
        raise ValueError(f"LPS obs is missing columns: {sorted(missing)}")
    obs = source.obs.copy().reset_index(drop=True)
    obs["time_label"] = obs["time_label"].astype(str)
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["perturbation_id"] = obs["perturbation_id"].astype(str)
    obs["is_control"] = _as_bool(obs["is_control"])
    obs = obs[obs["time_label"].isin({"90m", "6h", "10h"})].copy()
    obs["measure_id"] = obs["sample_id"] + "::" + obs["perturbation_id"]
    source_ids = set(obs.loc[obs["time_label"].eq("90m"), "measure_id"])
    obs = obs[obs["measure_id"].isin(source_ids)].copy()
    if not source_ids or not obs["is_control"].any():
        raise ValueError("LPS input requires 90m source support and controls.")

    metadata_rows = []
    for measure_id, rows in obs.groupby("measure_id", observed=True, sort=True):
        samples = rows["sample_id"].unique()
        perturbations = rows["perturbation_id"].unique()
        controls = rows["is_control"].unique()
        if len(samples) != 1 or len(perturbations) != 1 or len(controls) != 1:
            raise ValueError(f"Inconsistent LPS identity for measure {measure_id!r}.")
        is_control = bool(controls[0])
        embedding = "__control__" if is_control else str(perturbations[0])
        scope = "control" if is_control else "lps"
        metadata_rows.append(
            {
                "measure_id": str(measure_id),
                "sample_id": str(samples[0]),
                "perturbation_id": str(perturbations[0]),
                "guide_id": str(perturbations[0]),
                "embedding_id": embedding,
                "target_gene": "__control__" if is_control else "__not_applicable__",
                "context_group_id": f"{samples[0]}::{scope}",
                "is_control": is_control,
            }
        )
    measure_meta = pd.DataFrame(metadata_rows)
    group_by_measure = measure_meta.set_index("measure_id")["context_group_id"].to_dict()

    mass_frame = obs[["measure_id", "time_label"]].copy()
    if "mass_value" in obs:
        mass_frame["mass"] = pd.to_numeric(obs["mass_value"], errors="raise")
        masses = mass_frame.groupby(["measure_id", "time_label"], observed=True, as_index=False)[
            "mass"
        ].sum()
    else:
        masses = mass_frame.value_counts(sort=False).rename("mass").reset_index()
        masses["context_group_id"] = masses["measure_id"].map(group_by_measure)
        totals = masses.groupby(["context_group_id", "time_label"], observed=True)[
            "mass"
        ].transform("sum")
        masses["mass"] /= totals
        masses = masses.drop(columns="context_group_id")
    masses["denominator"] = [
        f"{group_by_measure[measure_id]}::{time_label}"
        for measure_id, time_label in zip(masses["measure_id"], masses["time_label"], strict=True)
    ]

    selected = []
    for (time_label, measure_id), rows in obs.groupby(
        ["time_label", "measure_id"], observed=True, sort=False
    ):
        positions = rows.index.to_numpy(dtype=np.int64)
        selected.extend(
            _stable_subsample(
                positions,
                support_cap,
                f"{time_label}:{measure_id}",
                seed,
            ).tolist()
        )
    selected_positions = np.asarray(sorted(selected), dtype=np.int64)
    selected_obs = obs.loc[selected_positions]
    support_obs = pd.DataFrame(
        {
            "measure_id": selected_obs["measure_id"].to_numpy(),
            "time_label": selected_obs["time_label"].to_numpy(),
            "atom_weight": np.ones(len(selected_positions)),
        }
    )
    support_obs.index = pd.Index(
        [f"support_{index}" for index in range(len(support_obs))], dtype=object
    )
    support = ad.AnnData(
        X=np.empty((len(selected_positions), 0), dtype=np.float32),
        obs=support_obs,
    )
    uses_existing_latent = latent_key in source.obsm
    if uses_existing_latent:
        latent = np.asarray(source.obsm[latent_key][selected_positions], dtype=np.float32)
    else:
        latent = _svd_latent(
            source,
            selected_positions,
            counts_layer=counts_layer,
            latent_dim=latent_dim,
            hvg_count=hvg_count,
            seed=seed,
        )
    support.obsm["X_credo"] = latent
    axis = Axis(
        kind="physical",
        source="90m",
        labels=("90m", "6h", "10h"),
        values=(1.5, 6.0, 10.0),
    )
    return write_canonical_dataset(
        output_dir,
        support=support,
        measure_meta=measure_meta,
        masses=masses,
        axis=axis,
        mass_semantics=MassSemantics.RELATIVE_WITHIN_GROUP,
        description="Longitudinal donor/cell-state response after LPS stimulation",
        source={
            "input": str(input_path),
            "latent_key": latent_key if uses_existing_latent else None,
            "counts_layer": counts_layer if not uses_existing_latent else None,
            "latent_dim": int(latent.shape[1]),
            "hvg_count": hvg_count if not uses_existing_latent else None,
            "support_cap": support_cap,
            "seed": seed,
        },
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repository.parent / "inputs" / "LPS" / "credo_lps_90m_6h_10h_celltype.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository.parent / "inputs" / "LPS" / "canonical",
    )
    parser.add_argument("--latent-key", default="X_credo")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hvg-count", type=int, default=2_000)
    parser.add_argument("--support-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = prepare(
        args.input,
        args.output_dir,
        latent_key=args.latent_key,
        counts_layer=args.counts_layer,
        latent_dim=args.latent_dim,
        hvg_count=args.hvg_count,
        support_cap=args.support_cap,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
