#!/usr/bin/env python
"""Build the three-time LPS trajectory AnnData used by run_credo_lps_3time.py."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def clean_token(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def canon_time(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if text in {"0", "0m", "0h", "baseline", "base", "pre", "prelps"}:
        return "0h"
    if text in {"90m", "90min", "90mins", "1.5h", "1h30m"} or "90" in text:
        return "90m"
    if text in {"6", "6h", "360m", "360min"} or "6h" in text:
        return "6h"
    if text in {"10", "10h", "600m", "600min"} or "10h" in text:
        return "10h"
    return str(value)


def pick_col(obs: pd.DataFrame, requested: str, candidates: list[str], label: str) -> str:
    if requested != "auto":
        if requested not in obs.columns:
            raise KeyError(f"{label} column {requested!r} not found")
        return requested
    lookup = {col.lower(): col for col in obs.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"Could not auto-detect {label} column. Available: {list(obs.columns)}")


def expression_source(a: ad.AnnData, layer: str):
    if layer != "auto":
        if layer == "X":
            return a.X, a.var.copy(), "X"
        if layer not in a.layers:
            raise KeyError(f"Layer {layer!r} not found. Layers: {list(a.layers.keys())}")
        return a.layers[layer], a.var.copy(), f"layer:{layer}"
    for name in ["counts", "raw_counts", "UMI", "umis"]:
        if name in a.layers:
            return a.layers[name], a.var.copy(), f"layer:{name}"
    if a.raw is not None:
        return a.raw.X, a.raw.var.copy(), "raw"
    return a.X, a.var.copy(), "X"


def subset_from_source(a: ad.AnnData, indices: np.ndarray, X_source, var: pd.DataFrame) -> ad.AnnData:
    X = X_source[indices, :]
    if hasattr(X, "copy"):
        X = X.copy()
    return ad.AnnData(X=X, obs=a.obs.iloc[indices].copy(), var=var.copy())


def add_hv_gene(out: ad.AnnData, n_hv: int) -> None:
    if n_hv <= 0 or n_hv >= out.n_vars:
        out.var["hv_gene"] = True
        return
    X = out.X
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        second = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        arr = np.asarray(X)
        mean = arr.mean(axis=0)
        second = (arr ** 2).mean(axis=0)
    var = np.maximum(second - mean ** 2, 0.0)
    score = np.log1p(var / np.maximum(mean, 1e-8))
    keep = np.argsort(score)[::-1][:n_hv]
    mask = np.zeros(out.n_vars, dtype=bool)
    mask[keep] = True
    out.var["hv_gene"] = mask


def downsample(out: ad.AnnData, max_cells: int, seed: int) -> ad.AnnData:
    if max_cells <= 0:
        return out
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    frame = out.obs.reset_index(drop=False)
    for _, sub in frame.groupby(["time_label", "sample_id", "perturbation_id"], observed=True, sort=False):
        idx = sub.index.to_numpy()
        if len(idx) > max_cells:
            idx = rng.choice(idx, size=max_cells, replace=False)
        keep.append(idx)
    merged = np.concatenate(keep)
    merged.sort()
    return out[merged].copy()


def assign_mass_values(out: ad.AnnData, group_mass: dict[tuple[str, str, str, str], float] | None = None) -> dict[tuple[str, str, str, str], float]:
    """Assign cell weights preserving pre-downsample donor/time composition."""
    scope = np.where(out.obs["is_control"].astype(bool).to_numpy(), "ctrl", "lps")
    frame = out.obs.assign(_scope=scope)
    denom_key = list(
        zip(
            frame["sample_id"].astype(str),
            frame["time_label"].astype(str),
            frame["_scope"].astype(str),
        )
    )
    group_key = [
        (*base, str(pid))
        for base, pid in zip(denom_key, frame["perturbation_id"].astype(str))
    ]

    if group_mass is None:
        denom = pd.Series(denom_key).value_counts().to_dict()
        group_counts = pd.Series(group_key).value_counts().to_dict()
        group_mass = {
            key: float(count) / float(denom[key[:3]])
            for key, count in group_counts.items()
        }
    retained_counts = pd.Series(group_key).value_counts().to_dict()
    out.obs["mass_value"] = [
        group_mass[key] / float(retained_counts[key])
        for key in group_key
    ]
    return group_mass


def main() -> None:
    parser = argparse.ArgumentParser(description="Create credo_lps_90m_6h_10h_celltype.h5ad")
    parser.add_argument("--input", default="../LPS/Manuscript/private_lps_data.h5ad")
    parser.add_argument("--output", default="../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad")
    parser.add_argument("--time-col", default="auto")
    parser.add_argument("--donor-col", default="auto")
    parser.add_argument("--celltype-col", default="auto")
    parser.add_argument("--counts-layer", default="auto")
    parser.add_argument("--n-hv-candidate", type=int, default=6000)
    parser.add_argument("--max-cells-per-key-time-sample", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a = ad.read_h5ad(args.input)
    obs = a.obs.copy()
    time_col = pick_col(obs, args.time_col, ["time_after_LPS", "timepoint", "time", "timeline"], "time")
    donor_col = pick_col(obs, args.donor_col, ["individual_id", "donor_id", "donor", "sample_id"], "donor")
    celltype_col = pick_col(obs, args.celltype_col, ["final_anno", "cell_type", "celltype", "annotation"], "cell type")
    X_source, var, source_name = expression_source(a, args.counts_layer)
    print(f"Using time={time_col}, donor={donor_col}, celltype={celltype_col}, expression={source_name}")

    canon = obs[time_col].map(canon_time)
    a.obs["_canon_time"] = canon.values

    blocks: list[ad.AnnData] = []
    for label, physical in [("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]:
        idx = np.flatnonzero(canon.eq(label).to_numpy())
        if len(idx) == 0:
            raise ValueError(f"No cells found for LPS time {label}")
        block = subset_from_source(a, idx, X_source, var)
        ct = block.obs[celltype_col].astype(str).map(clean_token)
        block.obs["time_label"] = label
        block.obs["physical_time"] = physical
        block.obs["sample_id"] = block.obs[donor_col].astype(str)
        block.obs["perturbation_id"] = "LPS__" + ct
        block.obs["embedding_id"] = block.obs["perturbation_id"]
        block.obs["is_control"] = False
        blocks.append(block)

    baseline_idx = np.flatnonzero(canon.eq("0h").to_numpy())
    if len(baseline_idx) == 0:
        raise ValueError("No baseline cells found for static control")
    for label, physical in [("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]:
        ctrl = subset_from_source(a, baseline_idx, X_source, var)
        ctrl.obs["time_label"] = label
        ctrl.obs["physical_time"] = physical
        ctrl.obs["sample_id"] = ctrl.obs[donor_col].astype(str)
        ctrl.obs["perturbation_id"] = "ctrl__baseline_static"
        ctrl.obs["embedding_id"] = "ctrl__baseline_static"
        ctrl.obs["is_control"] = True
        blocks.append(ctrl)

    out = ad.concat(
        blocks,
        join="outer",
        label="trajectory_block",
        keys=["90m", "6h", "10h", "ctrl_90m", "ctrl_6h", "ctrl_10h"],
        index_unique="__",
    )
    out.obs = out.obs.copy()
    out.obs["cell_id"] = out.obs_names.astype(str)
    group_mass = assign_mass_values(out)
    add_hv_gene(out, args.n_hv_candidate)
    out.layers["counts"] = out.X.copy()
    out = downsample(out, args.max_cells_per_key_time_sample, args.seed)
    assign_mass_values(out, group_mass=group_mass)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output, compression="gzip")
    print("Wrote:", args.output)
    print("shape:", out.shape)
    print(out.obs.groupby(["time_label", "perturbation_id"], observed=True).size().unstack(fill_value=0).T.head(30))


if __name__ == "__main__":
    main()
