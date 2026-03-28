"""
Create explicit train/val/test splits for the full LARRY dataset.

The split is deterministic and stratified by:
- Time point
- Cell type annotation

Outputs:
- split-annotated h5ad
- overall split counts CSV
- per-(time, cell type, split) counts CSV
- Markdown summary

Run with:
  conda run -n scdiffeq python make_larry_full_split.py
"""
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DATA = Path(
    os.environ.get(
        "LARRY_COMPARISON_DATA",
        "/home/yding1995/opscc_sc/scDiffeq/KleinLabData/in_vitro/larry_package_like_no_download.h5ad",
    )
)
OUT = Path(
    os.environ.get(
        "LARRY_FULL_SPLIT_OUT",
        "/home/yding1995/opscc_sc/CAPE/outputs/larry_full_splits",
    )
)
SEED = int(os.environ.get("LARRY_SPLIT_SEED", "0"))
TRAIN_FRAC = float(os.environ.get("LARRY_SPLIT_TRAIN", "0.8"))
VAL_FRAC = float(os.environ.get("LARRY_SPLIT_VAL", "0.1"))
TEST_FRAC = float(os.environ.get("LARRY_SPLIT_TEST", "0.1"))
STRATIFY_COLS = ["Time point", "Cell type annotation"]

assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-8, "Split fractions must sum to 1."

OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

print("Loading full LARRY dataset …")
adata = ad.read_h5ad(DATA)
obs = adata.obs.copy()
obs["_row"] = np.arange(adata.n_obs)

missing = [col for col in STRATIFY_COLS if col not in obs.columns]
if missing:
    raise KeyError(f"Missing required stratification columns: {missing}")

split = np.empty(adata.n_obs, dtype=object)

for keys, group in obs.groupby(STRATIFY_COLS, sort=True, observed=False):
    rows = group["_row"].to_numpy(copy=True)
    rng.shuffle(rows)
    n = len(rows)

    train_end = int(round(TRAIN_FRAC * n))
    val_end = train_end + int(round(VAL_FRAC * n))
    if val_end >= n:
        val_end = n - 1

    split[rows[:train_end]] = "train"
    split[rows[train_end:val_end]] = "val"
    split[rows[val_end:]] = "test"

if np.any(pd.isna(split)):
    raise RuntimeError("Some cells were not assigned to a split.")

adata.obs["split"] = pd.Categorical(split, categories=["train", "val", "test"])
adata.obs["train"] = adata.obs["split"] == "train"
adata.obs["val"] = adata.obs["split"] == "val"
adata.obs["test"] = adata.obs["split"] == "test"

overall = (
    adata.obs["split"]
    .value_counts()
    .reindex(["train", "val", "test"])
    .rename_axis("split")
    .reset_index(name="n_cells")
)
overall["fraction"] = overall["n_cells"] / adata.n_obs

by_group = (
    adata.obs.groupby(STRATIFY_COLS + ["split"], observed=False)
    .size()
    .rename("n_cells")
    .reset_index()
    .sort_values(STRATIFY_COLS + ["split"])
)

split_tag = f"{int(TRAIN_FRAC*100):02d}_{int(VAL_FRAC*100):02d}_{int(TEST_FRAC*100):02d}"
split_h5ad = OUT / f"{DATA.stem}.split_{split_tag}.h5ad"
overall_csv = OUT / f"{DATA.stem}.split_{split_tag}.overall.csv"
by_group_csv = OUT / f"{DATA.stem}.split_{split_tag}.by_time_celltype.csv"
summary_md = OUT / f"{DATA.stem}.split_{split_tag}.summary.md"
summary_json = OUT / f"{DATA.stem}.split_{split_tag}.summary.json"

adata.uns["fixed_split"] = {
    "source_data_path": str(DATA),
    "seed": SEED,
    "train_frac": TRAIN_FRAC,
    "val_frac": VAL_FRAC,
    "test_frac": TEST_FRAC,
    "stratify_cols": STRATIFY_COLS,
    "output_h5ad": str(split_h5ad),
}

print("Writing split-annotated h5ad …")
adata.write_h5ad(split_h5ad)
overall.to_csv(overall_csv, index=False)
by_group.to_csv(by_group_csv, index=False)

summary_payload = {
    "source_data_path": str(DATA),
    "output_h5ad": str(split_h5ad),
    "seed": SEED,
    "stratify_cols": STRATIFY_COLS,
    "n_cells": int(adata.n_obs),
    "n_features": int(adata.n_vars),
    "overall_counts": {
        row["split"]: {
            "n_cells": int(row["n_cells"]),
            "fraction": float(row["fraction"]),
        }
        for _, row in overall.iterrows()
    },
}

with open(summary_json, "w") as f:
    json.dump(summary_payload, f, indent=2)

summary_lines = [
    "# Full LARRY Train/Val/Test Split",
    "",
    f"Source data: `{DATA}`",
    f"Output h5ad: `{split_h5ad}`",
    f"Seed: `{SEED}`",
    f"Stratified by: `{', '.join(STRATIFY_COLS)}`",
    "",
    "| Split | Cells | Fraction |",
    "| --- | ---: | ---: |",
]
for _, row in overall.iterrows():
    summary_lines.append(f"| {row['split']} | {int(row['n_cells'])} | {float(row['fraction']):.4f} |")
summary_lines.append("")
summary_lines.append(f"Overall counts CSV: `{overall_csv}`")
summary_lines.append(f"Per-group counts CSV: `{by_group_csv}`")
summary_lines.append("")

with open(summary_md, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print("\n".join(summary_lines))
