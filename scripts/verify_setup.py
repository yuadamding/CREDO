#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

import credo
from credo.models.full_model import FullDynamicsModel
from credo.training.trainer import Trainer


DEFAULT_DATA_CANDIDATES = [
    "../inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad",
]


def default_data_path() -> str:
    for candidate in DEFAULT_DATA_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return DEFAULT_DATA_CANDIDATES[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=default_data_path(),
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    adata = ad.read_h5ad(data_path, backed="r")
    print("data_path", data_path)
    print("shape", adata.shape)
    print("has_X_pca", "X_pca" in adata.obsm)
    print("has_X_umap", "X_umap" in adata.obsm)
    print("has_X_pca_latest_sct", "X_pca_latest_sct" in adata.obsm)
    print("has_X_umap_latest", "X_umap_latest" in adata.obsm)
    print("torch", torch.__version__)
    print("numpy", np.__version__)
    print("pandas", pd.__version__)
    print("credo", credo.__version__)
    print("model_import", FullDynamicsModel.__name__)
    print("trainer_import", Trainer.__name__)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()


if __name__ == "__main__":
    main()
