"""
Train scDiffeq on the full LARRY dataset and persist a local benchmark record.

This script is intended to create a reusable local reference run for future
comparisons. It stores:
- the training metrics
- the checkpoint/log directory
- summary results JSON/Markdown
- a timestamped run manifest

Run with:
  conda run -n scdiffeq python record_scdiffeq_larry_full.py
"""
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import umap
from geomloss import SamplesLoss
from sklearn.neighbors import KNeighborsClassifier


DATA = Path(
    os.environ.get(
        "LARRY_COMPARISON_DATA",
        "/home/yding1995/opscc_sc/scDiffeq/KleinLabData/in_vitro/larry_package_like_no_download.h5ad",
    )
)
OUT_ROOT = Path(
    os.environ.get(
        "LARRY_FULL_SCDIFFEQ_RECORD_ROOT",
        "/home/yding1995/opscc_sc/CAPE/outputs/larry_full_scdiffeq_record",
    )
)
RUN_TAG = os.environ.get("LARRY_FULL_SCDIFFEQ_RUN_TAG", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = OUT_ROOT / RUN_TAG
MODEL_ROOT = RUN_DIR / "models"
RUN_NAME = os.environ.get("LARRY_SCDIFFEQ_RUN_NAME", "larry_full_scdiffeq")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")
N_EPOCHS = int(os.environ.get("LARRY_SCDIFFEQ_EPOCHS", "500"))
N_EVAL = int(os.environ.get("LARRY_SCDIFFEQ_EVAL_CELLS", "1000"))
N_SIM = int(os.environ.get("LARRY_SCDIFFEQ_SIM", "512"))
RANDOM_SEED = int(os.environ.get("LARRY_COMPARISON_SEED", "0"))

OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def safe_write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def safe_write_text(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


print("Loading full LARRY dataset …")
adata = ad.read_h5ad(DATA)
if "ct_pseudotime" in adata.obs.columns:
    adata.obs["ct_pseudotime"] = adata.obs["ct_pseudotime"].astype(float)
adata.obs.index.name = "index"

for key in ["X_clone", "cell_fate_df"]:
    if key in adata.obsm:
        del adata.obsm[key]

adata.obs.index = adata.obs.reset_index(drop=True).index.astype(str)
print(f"Full adata: {adata.shape}")
print(f"Using device={DEVICE} torch={torch.__version__} env={os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

time_points = [float(t) for t in sorted(adata.obs["Time point"].unique())]
t0 = min(time_points)
t6 = max(time_points)
cells_t0 = adata[adata.obs["Time point"] == t0]

print("Fitting UMAP on full PCA space …")
umap_model = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
adata.obsm["X_umap"] = umap_model.fit_transform(adata.obsm["X_pca"])

print("Building scDiffEq model …")
import scdiffeq as sdq

model = sdq.scDiffEq(
    adata,
    name=RUN_NAME,
    working_dir=str(MODEL_ROOT),
)
n_params = sum(p.numel() for p in model.DiffEq.parameters())

print(f"Training {N_EPOCHS} epochs on full LARRY …")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
t0_train = time.time()
model.fit(
    train_epochs=N_EPOCHS,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    deterministic=False,
    ckpt_frequency=100,
    save_last_ckpt=True,
    keep_ckpts=-1,
)
if DEVICE == "cuda":
    torch.cuda.synchronize()
train_time_s = time.time() - t0_train
peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if DEVICE == "cuda" else None
print(f"Training done in {train_time_s:.1f}s")

model_log_dir = Path(model.logger[0].log_dir)
checkpoints_dir = model_log_dir / "checkpoints"
checkpoint_paths = sorted(str(p) for p in checkpoints_dir.glob("*.ckpt"))
last_ckpt_path = checkpoints_dir / "last.ckpt"
last_ckpt_path_str = str(last_ckpt_path) if last_ckpt_path.exists() else None

metrics_df = model.metrics.copy()
metrics_path = RUN_DIR / "scdiffeq_training_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)


def _last_per_epoch(df, col):
    sub = df[["epoch", col]].dropna()
    return sub.groupby("epoch")[col].last().reset_index()


sink4_curve = _last_per_epoch(metrics_df, "sinkhorn_4.0_training")
sink6_curve = _last_per_epoch(metrics_df, "sinkhorn_6.0_training")

print("Computing drift and diffusion embeddings …")
model.drift()
model.diffusion()

sinkhorn_fn = SamplesLoss("sinkhorn", p=2, blur=0.1, scaling=0.7, debias=True)
obs_pca = {}
for t in time_points:
    mask = adata.obs["Time point"] == t
    obs_pca[t] = torch.tensor(adata[mask].obsm["X_pca"], dtype=torch.float32, device=DEVICE)

eval_n = min(N_EVAL, len(cells_t0))
eval_idx = np.random.choice(len(cells_t0), size=eval_n, replace=False)
eval_cells = cells_t0.obs.iloc[eval_idx]

print(f"Simulating {eval_n} full-dataset t={t0} cells for distribution evaluation …")
if DEVICE == "cuda":
    torch.cuda.synchronize()
t0_eval = time.time()
adata_eval = sdq.tl.simulate(
    adata,
    idx=eval_cells.index,
    N=1,
    diffeq=model.DiffEq,
    time_key="Time point",
    device=SIM_DEVICE,
)
if DEVICE == "cuda":
    torch.cuda.synchronize()
eval_time_s = time.time() - t0_eval


def pca_sinkhorn(adata_sim, t_query, n=1000):
    mask = adata_sim.obs["t"] == t_query
    if mask.sum() == 0:
        return float("nan")
    n_use = min(n, int(mask.sum()), len(obs_pca[t_query]))
    z_sim = torch.tensor(np.array(adata_sim[mask].X), dtype=torch.float32, device=DEVICE)[:n_use]
    z_obs = obs_pca[t_query][torch.randperm(len(obs_pca[t_query]), device=DEVICE)[:n_use]]
    return float(sinkhorn_fn(z_sim, z_obs).item())


sink_pca = {
    f"{t:.1f}": pca_sinkhorn(adata_eval, t, n=eval_n)
    for t in time_points
    if t != t0
}
print("Post-training Sinkhorn by time:", sink_pca)

print(f"Training t={t6}-only kNN on full dataset labels …")
t6_mask = adata.obs["Time point"] == t6
t6_pca_np = adata[t6_mask].obsm["X_pca"]
t6_labels_np = adata[t6_mask].obs["Cell type annotation"].values
knn_t6 = KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=4)
knn_t6.fit(t6_pca_np, t6_labels_np)

prog_n = min(3, len(cells_t0))
prog_idx = np.random.choice(len(cells_t0), size=prog_n, replace=False)
progenitors = cells_t0.obs.iloc[prog_idx]

print(f"Simulating {N_SIM} trajectories per t={t0} seed cell for fate labeling …")
if DEVICE == "cuda":
    torch.cuda.synchronize()
t0_fate = time.time()
adata_sim = sdq.tl.simulate(
    adata,
    idx=progenitors.index,
    N=N_SIM,
    diffeq=model.DiffEq,
    time_key="Time point",
    device=SIM_DEVICE,
)
if DEVICE == "cuda":
    torch.cuda.synchronize()
fate_eval_time_s = time.time() - t0_fate

t6_sim_mask = adata_sim.obs["t"] == t6
z6_sim_np = np.array(adata_sim[t6_sim_mask].X)
pred_labels = knn_t6.predict(z6_sim_np)
fate_counts = dict(Counter(pred_labels))
print("Full-dataset t6 fate counts:", fate_counts)

eval_h5ad_path = RUN_DIR / "adata_eval_scdiffeq_full.h5ad"
eval_npz_path = RUN_DIR / "adata_eval_scdiffeq_full_t6.npz"
try:
    adata_eval.obsm["X_umap"] = umap_model.transform(np.array(adata_eval.X))
    adata_eval.write_h5ad(eval_h5ad_path)
    eval_artifact = str(eval_h5ad_path)
except Exception as e:
    print(f"Warning: could not write full h5ad eval artifact: {e}")
    if eval_h5ad_path.exists():
        eval_h5ad_path.unlink()
    eval_t6_mask = adata_eval.obs["t"] == t6
    np.savez(
        eval_npz_path,
        X=np.array(adata_eval[eval_t6_mask].X),
        t=np.array(adata_eval[eval_t6_mask].obs["t"]),
    )
    eval_artifact = str(eval_npz_path)

results = {
    "method": "scDiffeq",
    "scope": "full_larry",
    "data_path": str(DATA),
    "run_tag": RUN_TAG,
    "run_dir": str(RUN_DIR),
    "device": DEVICE,
    "torch_version": torch.__version__,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    "matmul_precision": torch.get_float32_matmul_precision(),
    "tf32_enabled": bool(torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32),
    "n_epochs": N_EPOCHS,
    "n_params": n_params,
    "n_cells": int(adata.n_obs),
    "n_features": int(adata.n_vars),
    "time_points": time_points,
    "n_cells_by_time": {str(k): int(v) for k, v in adata.obs["Time point"].value_counts().sort_index().items()},
    "cell_type_counts": {str(k): int(v) for k, v in adata.obs["Cell type annotation"].value_counts().items()},
    "train_time_s": round(train_time_s, 1),
    "eval_time_s": round(eval_time_s, 1),
    "fate_eval_time_s": round(fate_eval_time_s, 1),
    "peak_gpu_mem_mb": peak_gpu_mem_mb,
    "sinkhorn_training_t4_curve": sink4_curve["sinkhorn_4.0_training"].tolist(),
    "sinkhorn_training_t6_curve": sink6_curve["sinkhorn_6.0_training"].tolist(),
    "epochs_curve": sink4_curve["epoch"].tolist(),
    "sink_pca_by_time": sink_pca,
    "fate_counts_t6_knn": fate_counts,
    "fate_seed_indices": list(map(str, progenitors.index.tolist())),
    "model_log_dir": str(model_log_dir),
    "metrics_path": str(metrics_path),
    "checkpoint_paths": checkpoint_paths,
    "last_checkpoint_path": last_ckpt_path_str,
    "eval_artifact": eval_artifact,
}

results_path = RUN_DIR / "results_scdiffeq_full.json"
safe_write_json(results_path, results)

summary_lines = [
    "# Full LARRY scDiffeq Record",
    "",
    f"Run tag: `{RUN_TAG}`",
    f"Data: `{DATA}`",
    f"Run dir: `{RUN_DIR}`",
    f"Model log dir: `{model_log_dir}`",
    f"Last checkpoint: `{last_ckpt_path_str}`",
    "",
    "| Metric | Value |",
    "| --- | ---: |",
    f"| Cells | {adata.n_obs} |",
    f"| Features | {adata.n_vars} |",
    f"| Train time (s) | {train_time_s:.1f} |",
    f"| Eval time (s) | {eval_time_s:.1f} |",
    f"| Fate sim time (s) | {fate_eval_time_s:.1f} |",
    f"| Peak GPU mem (MB) | {peak_gpu_mem_mb if peak_gpu_mem_mb is not None else 'n/a'} |",
    f"| Sinkhorn t=4 | {sink_pca.get('4.0', float('nan')):.4f} |",
    f"| Sinkhorn t=6 | {sink_pca.get('6.0', float('nan')):.4f} |",
    "",
    "Fate counts at t=6:",
]
for label, count in sorted(fate_counts.items()):
    summary_lines.append(f"- {label}: {count}")

summary_text = "\n".join(summary_lines) + "\n"
summary_path = RUN_DIR / "summary_scdiffeq_full.md"
safe_write_text(summary_path, summary_text)

latest_record = {
    "run_tag": RUN_TAG,
    "run_dir": str(RUN_DIR),
    "results_path": str(results_path),
    "summary_path": str(summary_path),
    "last_checkpoint_path": last_ckpt_path_str,
    "model_log_dir": str(model_log_dir),
    "metrics_path": str(metrics_path),
}
safe_write_json(OUT_ROOT / "latest_record.json", latest_record)
safe_write_text(OUT_ROOT / "latest_run.txt", str(RUN_DIR) + "\n")

print(summary_text)
print("Saved latest record pointer to", OUT_ROOT / "latest_record.json")
