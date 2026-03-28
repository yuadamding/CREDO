"""
Train scDiffeq on the LARRY dataset and save results for comparison with CAPE.
Run with: conda run -n scdiffeq python run_scdiffeq_larry.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import anndata as ad
import umap

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))  # not needed, just in case

DATA = Path(
    os.environ.get(
        "LARRY_COMPARISON_DATA",
        "/home/yding1995/opscc_sc/scDiffeq/KleinLabData/in_vitro/larry_package_like_no_download.h5ad",
    )
)
OUT = Path(
    os.environ.get(
        "LARRY_COMPARISON_OUT",
        "/home/yding1995/opscc_sc/CAPE/outputs/larry_comparison",
    )
)
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS = int(os.environ.get("LARRY_SCDIFFEQ_EPOCHS", "500"))
N_SIM = int(os.environ.get("LARRY_SCDIFFEQ_SIM", "512"))
RANDOM_SEED = int(os.environ.get("LARRY_COMPARISON_SEED", "0"))
SIM_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# 1. Load & filter data  (identical to demo)
# ---------------------------------------------------------------------------
print("Loading data …")
adata_ref = ad.read_h5ad(DATA)
if "ct_pseudotime" in adata_ref.obs.columns:
    adata_ref.obs["ct_pseudotime"] = adata_ref.obs["ct_pseudotime"].astype(float)
adata_ref.obs.index.name = "index"

nm_clones = adata_ref.uns["fate_counts"][["Monocyte", "Neutrophil"]].dropna().index
adata_ref.obs["nm_clones"] = adata_ref.obs["clone_idx"].isin(nm_clones)
MASK = (
    adata_ref.obs["Cell type annotation"].isin(["Monocyte", "Neutrophil", "Undifferentiated"])
    & adata_ref.obs["nm_clones"]
)
adata = adata_ref[MASK].copy()
del adata.obsm["X_clone"]
del adata.obsm["cell_fate_df"]
adata.obs.index = adata.obs.reset_index(drop=True).index.astype(str)

print(f"Filtered adata: {adata.shape}")
print(f"Using device={DEVICE} torch={torch.__version__} env={os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

# Refit UMAP on filtered data PCA
print("Fitting UMAP …")
UMAP_model = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
adata.obsm["X_umap"] = UMAP_model.fit_transform(adata.obsm["X_pca"])

# ---------------------------------------------------------------------------
# 2. Train scDiffeq
# ---------------------------------------------------------------------------
import scdiffeq as sdq

print("Building scDiffEq model …")
model = sdq.scDiffEq(adata)
n_params = sum(p.numel() for p in model.DiffEq.parameters())

print(f"Training {N_EPOCHS} epochs …")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
t0 = time.time()
model.fit(
    train_epochs=N_EPOCHS,
    accelerator="gpu" if DEVICE == "cuda" else "cpu",
    devices=1,
    deterministic=False,
)
if DEVICE == "cuda":
    torch.cuda.synchronize()
train_time = time.time() - t0
train_peak_gpu_mem_mb = (
    round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if DEVICE == "cuda" else None
)
print(f"Training done in {train_time:.1f}s")

# Save training metrics
metrics_df = model.metrics.copy()
metrics_df.to_csv(OUT / "scdiffeq_training_metrics.csv", index=False)

# Extract per-epoch mean sinkhorn (last entry per epoch)
def _last_per_epoch(df, col):
    """Take last row per unique epoch for a given column."""
    sub = df[["epoch", col]].dropna()
    return sub.groupby("epoch")[col].last().reset_index()

sink4 = _last_per_epoch(metrics_df, "sinkhorn_4.0_training")
sink6 = _last_per_epoch(metrics_df, "sinkhorn_6.0_training")

# ---------------------------------------------------------------------------
# 3. Post-training evaluation in PCA space
# ---------------------------------------------------------------------------
from geomloss import SamplesLoss
sinkhorn_fn = SamplesLoss("sinkhorn", p=2, blur=0.1, scaling=0.7, debias=True)

# Get embeddings
model.drift()
model.diffusion()

# Observed PCA at each time point
t_vals = sorted(adata.obs["Time point"].unique())
obs_pca = {}
for t in t_vals:
    mask = adata.obs["Time point"] == t
    obs_pca[float(t)] = torch.tensor(adata[mask].obsm["X_pca"], dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------
# Post-training evaluation: simulate from 1000 random t=2 undifferentiated cells
# (same protocol as CAPE for fair comparison)
# ---------------------------------------------------------------------------
N_EVAL = 1000
t2_und = adata[(adata.obs["Time point"] == 2.0) & (adata.obs["Cell type annotation"] == "Undifferentiated")]
np.random.seed(RANDOM_SEED)
eval_idx = np.random.choice(len(t2_und), size=N_EVAL, replace=False)
eval_cells = t2_und.obs.iloc[eval_idx]

print(f"Simulating {N_EVAL} evaluation trajectories for Sinkhorn metric …")
if DEVICE == "cuda":
    torch.cuda.synchronize()
eval_t0 = time.time()
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
eval_time_s = time.time() - eval_t0

# Compute Sinkhorn at t=4 and t=6 vs observed (PCA space)
def _pca_sinkhorn(adata_sim, t_query, obs_pca_dict, n=1000):
    mask = adata_sim.obs["t"] == t_query
    if mask.sum() == 0:
        return float("nan")
    z_sim = torch.tensor(np.array(adata_sim[mask].X), dtype=torch.float32, device=DEVICE)[:n]
    z_obs = obs_pca_dict[t_query][torch.randperm(len(obs_pca_dict[t_query]), device=DEVICE)[:n]]
    return sinkhorn_fn(z_sim, z_obs).item()

sink_pca_4 = _pca_sinkhorn(adata_eval, 4.0, obs_pca, n=N_EVAL)
sink_pca_6 = _pca_sinkhorn(adata_eval, 6.0, obs_pca, n=N_EVAL)
print(f"scDiffeq post-training Sinkhorn (PCA, 1k cells): t4={sink_pca_4:.4f}, t6={sink_pca_6:.4f}")

# ---------------------------------------------------------------------------
# Fate prediction: simulate from 3 progenitor cells (for comparison with demo)
# Use t=6-only kNN for consistent fate annotation
# ---------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

t6_pca_np = adata[(adata.obs["Time point"] == 6.0)].obsm["X_pca"]
t6_labels_np = adata[(adata.obs["Time point"] == 6.0)].obs["Cell type annotation"].values
knn_t6 = KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=4)
knn_t6.fit(t6_pca_np, t6_labels_np)

prog_idx = np.random.choice(len(t2_und), size=3, replace=False)
progenitor = t2_und.obs.iloc[prog_idx]

print(f"Simulating {N_SIM} trajectories per progenitor for fate prediction …")
if DEVICE == "cuda":
    torch.cuda.synchronize()
fate_t0 = time.time()
adata_sim = sdq.tl.simulate(
    adata,
    idx=progenitor.index,
    N=N_SIM,
    diffeq=model.DiffEq,
    time_key="Time point",
    device=SIM_DEVICE,
)
if DEVICE == "cuda":
    torch.cuda.synchronize()
fate_time_s = time.time() - fate_t0

# Fate prediction at t=6 using t6-only kNN
t6_sim_mask = adata_sim.obs["t"] == 6.0
z6_sim_np = np.array(adata_sim[t6_sim_mask].X)
pred_labels = knn_t6.predict(z6_sim_np)
from collections import Counter
fate_counts_dict = Counter(pred_labels)
print("scDiffeq fate counts (t6-kNN):", dict(fate_counts_dict))

mono = fate_counts_dict.get("Monocyte", 0)
neut = fate_counts_dict.get("Neutrophil", 0)
total_committed = mono + neut
fate_mono_frac = mono / max(total_committed, 1)

# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------
results = {
    "method": "scDiffeq",
    "device": DEVICE,
    "torch_version": torch.__version__,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    "n_epochs": N_EPOCHS,
    "n_params": n_params,
    "train_time_s": round(train_time, 1),
    "eval_time_s": round(eval_time_s, 1),
    "fate_eval_time_s": round(fate_time_s, 1),
    "peak_gpu_mem_mb": train_peak_gpu_mem_mb,
    "matmul_precision": torch.get_float32_matmul_precision(),
    "tf32_enabled": bool(torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32),
    # Training curve (last sinkhorn per epoch in model's latent space)
    "sinkhorn_4_training_curve": sink4["sinkhorn_4.0_training"].tolist(),
    "sinkhorn_6_training_curve": sink6["sinkhorn_6.0_training"].tolist(),
    "epochs_curve": sink4["epoch"].tolist(),
    # Post-training evaluation in PCA space
    "sink_pca_4": sink_pca_4,
    "sink_pca_6": sink_pca_6,
    # Fate prediction
    "fate_monocyte": int(mono),
    "fate_neutrophil": int(neut),
    "fate_undifferentiated": int(fate_counts_dict.get("Undifferentiated", 0)),
    "fate_mono_frac": round(fate_mono_frac, 4),
}

with open(OUT / "results_scdiffeq.json", "w") as f:
    json.dump(results, f, indent=2)

# Save simulated cells for UMAP plot (use eval sim which has 1000 trajectories)
adata_eval.obsm["X_umap"] = UMAP_model.transform(np.array(adata_eval.X))
try:
    adata_eval.write_h5ad(OUT / "adata_sim_scdiffeq.h5ad")
except Exception as e:
    print(f"Warning: could not write h5ad: {e}")
    # Fall back: save as npz
    t6_mask = adata_eval.obs["t"] == 6.0
    np.savez(OUT / "scdiffeq_sim_t6.npz",
             X=np.array(adata_eval[t6_mask].X),
             umap=adata_eval[t6_mask].obsm.get("X_umap", np.zeros((t6_mask.sum(), 2))))

print("Saved results to", OUT)
print(json.dumps(results, indent=2))
