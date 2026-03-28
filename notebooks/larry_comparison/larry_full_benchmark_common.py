"""
Shared helpers for fair full-LARRY train/val/test benchmarks.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import anndata as ad
import numpy as np
import pandas as pd
import torch


DEFAULT_SPLIT_DATA = (
    "/home/yding1995/opscc_sc/CAPE/outputs/larry_full_splits/"
    "larry_package_like_no_download.split_80_10_10.h5ad"
)
DEFAULT_OUT = "/home/yding1995/opscc_sc/CAPE/outputs/larry_full_split_benchmark"

TIME_KEY = "Time point"
LABEL_KEY = "Cell type annotation"
SPLIT_KEY = "split"
TRAIN_KEY = "train"
VAL_KEY = "val"
TEST_KEY = "test"
USE_KEY = "X_pca"


def get_split_data_path() -> Path:
    return Path(os.environ.get("LARRY_FULL_SPLIT_DATA", DEFAULT_SPLIT_DATA))


def get_output_root() -> Path:
    return Path(os.environ.get("LARRY_FULL_BENCHMARK_OUT", DEFAULT_OUT))


def sync_cuda_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_text(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def load_split_adata(path: Path | None = None) -> ad.AnnData:
    if path is None:
        path = get_split_data_path()
    adata = ad.read_h5ad(path)
    if "ct_pseudotime" in adata.obs.columns:
        adata.obs["ct_pseudotime"] = adata.obs["ct_pseudotime"].astype(float)
    adata.obs.index.name = "index"
    return adata


def strip_unused_obsm(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    for key in ["X_clone", "cell_fate_df"]:
        if key in adata.obsm:
            del adata.obsm[key]
    adata.obs.index = adata.obs.index.astype(str)
    return adata


def split_counts(adata: ad.AnnData) -> Dict[str, int]:
    return {
        str(k): int(v)
        for k, v in adata.obs[SPLIT_KEY].value_counts().reindex(["train", "val", "test"]).items()
    }


def sorted_times(adata: ad.AnnData) -> List[float]:
    return [float(t) for t in sorted(adata.obs[TIME_KEY].unique().tolist())]


def subset_index(adata: ad.AnnData, *, split: str, time_point: float) -> np.ndarray:
    mask = (adata.obs[SPLIT_KEY] == split) & (adata.obs[TIME_KEY] == time_point)
    return adata.obs.index[mask].to_numpy()


def build_eval_plan(adata: ad.AnnData, seed: int = 0, n_eval: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    times = sorted_times(adata)
    t0 = min(times)
    plan = {
        "seed": seed,
        "n_eval": n_eval,
        "time_key": TIME_KEY,
        "label_key": LABEL_KEY,
        "split_key": SPLIT_KEY,
        "t0": t0,
        "times": times,
    }

    test_seed_pool = subset_index(adata, split="test", time_point=t0)
    n_seed = min(n_eval, len(test_seed_pool))
    plan["eval_seed_idx"] = rng.choice(test_seed_pool, size=n_seed, replace=False).tolist()

    target_idx = {}
    for t in times:
        if t == t0:
            continue
        pool = subset_index(adata, split="test", time_point=t)
        n_target = min(n_eval, len(pool))
        target_idx[f"{t:.1f}"] = rng.choice(pool, size=n_target, replace=False).tolist()
    plan["target_idx_by_time"] = target_idx

    plan["train_t6_idx"] = subset_index(adata, split="train", time_point=max(times)).tolist()
    plan["test_t6_idx"] = subset_index(adata, split="test", time_point=max(times)).tolist()
    return plan


def plan_to_tensors(
    adata: ad.AnnData,
    plan: dict,
    device: str,
) -> tuple[torch.Tensor, Dict[float, torch.Tensor]]:
    z0 = torch.tensor(adata[plan["eval_seed_idx"]].obsm[USE_KEY], dtype=torch.float32, device=device)
    targets = {}
    for t_str, idx in plan["target_idx_by_time"].items():
        targets[float(t_str)] = torch.tensor(adata[idx].obsm[USE_KEY], dtype=torch.float32, device=device)
    return z0, targets


def observed_label_counts(adata: ad.AnnData, idx: Iterable[str]) -> Dict[str, int]:
    series = adata[list(idx)].obs[LABEL_KEY]
    return {str(k): int(v) for k, v in series.value_counts().items()}


def normalize_counts(counts: Dict[str, int], labels: List[str]) -> Dict[str, float]:
    total = sum(counts.get(lbl, 0) for lbl in labels)
    if total == 0:
        return {lbl: 0.0 for lbl in labels}
    return {lbl: counts.get(lbl, 0) / total for lbl in labels}


def total_variation_distance(pred_counts: Dict[str, int], true_counts: Dict[str, int], labels: List[str]) -> float:
    pred = normalize_counts(pred_counts, labels)
    true = normalize_counts(true_counts, labels)
    return 0.5 * sum(abs(pred[lbl] - true[lbl]) for lbl in labels)


def counter_to_dict(labels: Iterable[str]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(labels).items()}


def tensor_from_idx(adata: ad.AnnData, idx: Iterable[str], device: str) -> torch.Tensor:
    return torch.tensor(adata[list(idx)].obsm[USE_KEY], dtype=torch.float32, device=device)


def timed_call(fn, device: str):
    sync_cuda_if_needed(device)
    t0 = time.time()
    out = fn()
    sync_cuda_if_needed(device)
    return out, time.time() - t0


def results_markdown(title: str, rows: List[tuple[str, str]]) -> str:
    lines = [f"# {title}", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)
