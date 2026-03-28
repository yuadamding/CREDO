"""
Fair scDiffeq benchmark on the fixed full-LARRY train/val/test split.

Run with:
  conda run -n scdiffeq python run_scdiffeq_larry_full_split.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from sklearn.neighbors import KNeighborsClassifier

from larry_full_benchmark_common import (
    LABEL_KEY,
    TIME_KEY,
    USE_KEY,
    build_eval_plan,
    counter_to_dict,
    get_output_root,
    get_split_data_path,
    load_split_adata,
    normalize_counts,
    observed_label_counts,
    plan_to_tensors,
    results_markdown,
    save_json,
    save_text,
    split_counts,
    strip_unused_obsm,
    sync_cuda_if_needed,
    timed_call,
    total_variation_distance,
)


DATA = get_split_data_path()
OUT = get_output_root() / "scdiffeq"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT = OUT / "model_logs"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_DEVICE = torch.device("cuda:0" if DEVICE == "cuda" else "cpu")
N_EPOCHS = int(os.environ.get("LARRY_FULL_BENCHMARK_EPOCHS", "500"))
N_EVAL = int(os.environ.get("LARRY_FULL_BENCHMARK_EVAL", "1000"))
RANDOM_SEED = int(os.environ.get("LARRY_COMPARISON_SEED", "0"))
SKIP_TRAIN = os.environ.get("LARRY_FULL_BENCHMARK_SKIP_TRAIN", "0") == "1"
CKPT_PATH_OVERRIDE = os.environ.get("LARRY_FULL_BENCHMARK_CKPT_PATH")
METRICS_PATH_OVERRIDE = os.environ.get("LARRY_FULL_BENCHMARK_METRICS_PATH")
TRAIN_TIME_OVERRIDE = os.environ.get("LARRY_FULL_BENCHMARK_TRAIN_TIME_S")
PEAK_GPU_MEM_OVERRIDE = os.environ.get("LARRY_FULL_BENCHMARK_PEAK_GPU_MB")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Loading split full LARRY data …")
adata = strip_unused_obsm(load_split_adata(DATA))
print(f"adata: {adata.shape}")
print(f"split_counts: {split_counts(adata)}")
print(f"Using device={DEVICE} torch={torch.__version__} env={os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

# Force scDiffeq to honor the fixed val split instead of re-splitting train.
from torch_adata import LightningAnnDataModule

_original_configure_train_val_split = LightningAnnDataModule.configure_train_val_split


def _fixed_configure_train_val_split(self):
    obs = self.adata.obs
    train_key = self.hparams["train_key"]
    val_key = self.hparams["val_key"]
    if train_key in obs.columns and val_key in obs.columns:
        return
    return _original_configure_train_val_split(self)


LightningAnnDataModule.configure_train_val_split = _fixed_configure_train_val_split

import scdiffeq as sdq

print("Building scDiffEq model …")
model = sdq.scDiffEq(
    adata,
    name="larry_full_split_scdiffeq",
    working_dir=str(MODEL_ROOT),
)
n_params = sum(p.numel() for p in model.DiffEq.parameters())
print(
    "datamodule shapes:",
    model.LitDataModule.train_adata.shape,
    model.LitDataModule.val_adata.shape,
    model.LitDataModule.test_adata.shape,
)

load_time_s = None
train_time_source = "measured_runtime"
peak_gpu_mem_source = "torch_max_memory_allocated"

if SKIP_TRAIN:
    ckpt_path = Path(CKPT_PATH_OVERRIDE) if CKPT_PATH_OVERRIDE else MODEL_ROOT / "latest_missing.ckpt"
    metrics_source_path = (
        Path(METRICS_PATH_OVERRIDE) if METRICS_PATH_OVERRIDE else ckpt_path.parent.parent / "metrics.csv"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for eval-only mode: {ckpt_path}")
    if not metrics_source_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found for eval-only mode: {metrics_source_path}")

    print(f"Skipping training and loading checkpoint: {ckpt_path}")
    sync_cuda_if_needed(DEVICE)
    t0_load = time.time()
    model.DiffEq = model.DiffEq.__class__.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.DiffEq = model.DiffEq.to(SIM_DEVICE)
    model.freeze()
    sync_cuda_if_needed(DEVICE)
    load_time_s = time.time() - t0_load
    print(f"Checkpoint loaded in {load_time_s:.1f}s")

    train_time_s = float(TRAIN_TIME_OVERRIDE) if TRAIN_TIME_OVERRIDE is not None else None
    peak_gpu_mem_mb = float(PEAK_GPU_MEM_OVERRIDE) if PEAK_GPU_MEM_OVERRIDE is not None else None
    train_time_source = "env_override" if TRAIN_TIME_OVERRIDE is not None else "missing"
    peak_gpu_mem_source = "env_override" if PEAK_GPU_MEM_OVERRIDE is not None else "missing"
    metrics_df = pd.read_csv(metrics_source_path)
    model_log_dir = ckpt_path.parent.parent
    checkpoints_dir = ckpt_path.parent
    last_ckpt = checkpoints_dir / "last.ckpt"
else:
    print(f"Training scDiffeq on train split for {N_EPOCHS} epochs …")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sync_cuda_if_needed(DEVICE)
    t0_train = time.time()
    model.fit(
        train_epochs=N_EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        deterministic=False,
    )
    sync_cuda_if_needed(DEVICE)
    train_time_s = time.time() - t0_train
    peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if DEVICE == "cuda" else None
    print(f"Training done in {train_time_s:.1f}s")

    metrics_df = model.metrics.copy()
    model_log_dir = Path(model.logger[0].log_dir)
    checkpoints_dir = model_log_dir / "checkpoints"
    last_ckpt = checkpoints_dir / "last.ckpt"

metrics_path = OUT / "scdiffeq_training_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)


def _last_per_epoch(df, col):
    sub = df[["epoch", col]].dropna()
    return sub.groupby("epoch")[col].last().reset_index()


times = sorted(float(t) for t in adata.obs[TIME_KEY].unique().tolist())
t0 = min(times)
t_last = max(times)

train_curve = {}
val_curve = {}
for t in times[1:]:
    train_col = f"sinkhorn_{t:.1f}_training"
    val_col = f"sinkhorn_{t:.1f}_validation"
    train_curve[f"{t:.1f}"] = _last_per_epoch(metrics_df, train_col)[train_col].tolist()
    val_curve[f"{t:.1f}"] = _last_per_epoch(metrics_df, val_col)[val_col].tolist()
epoch_curve = _last_per_epoch(metrics_df, f"sinkhorn_{times[1]:.1f}_training")["epoch"].tolist()

eval_plan = build_eval_plan(adata, seed=RANDOM_SEED, n_eval=N_EVAL)
save_json(OUT / "evaluation_plan.json", eval_plan)
z0_eval, targets_eval = plan_to_tensors(adata, eval_plan, device=DEVICE)

sinkhorn_fn = SamplesLoss("sinkhorn", p=2, blur=0.1, scaling=0.7, debias=True)


def run_test_simulation():
    return sdq.tl.simulate(
        adata,
        idx=eval_plan["eval_seed_idx"],
        N=1,
        diffeq=model.DiffEq,
        time_key=TIME_KEY,
        device=SIM_DEVICE,
    )


adata_eval, eval_time_s = timed_call(run_test_simulation, DEVICE)

sink_pca_by_time = {}
for t in times[1:]:
    mask = adata_eval.obs["t"] == t
    z_sim = torch.tensor(np.array(adata_eval[mask].X), dtype=torch.float32, device=DEVICE)
    z_obs = targets_eval[t]
    n_use = min(len(z_sim), len(z_obs))
    sink_pca_by_time[f"{t:.1f}"] = float(sinkhorn_fn(z_sim[:n_use], z_obs[:n_use]).item())

train_tlast_mask = (adata.obs["split"] == "train") & (adata.obs[TIME_KEY] == t_last)
train_tlast_pca = adata[train_tlast_mask].obsm[USE_KEY]
train_tlast_labels = adata[train_tlast_mask].obs[LABEL_KEY].values
knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=4)
knn.fit(train_tlast_pca, train_tlast_labels)

tlast_mask = adata_eval.obs["t"] == t_last
pred_labels = knn.predict(np.array(adata_eval[tlast_mask].X))
pred_counts = counter_to_dict(pred_labels)
true_counts = observed_label_counts(adata, eval_plan["test_t6_idx"])
label_set = sorted(set(pred_counts) | set(true_counts))
t6_tv = total_variation_distance(pred_counts, true_counts, label_set)

checkpoint_paths = sorted(str(p) for p in checkpoints_dir.glob("*.ckpt"))

try:
    adata_eval.write_h5ad(OUT / "adata_eval_scdiffeq_full_split.h5ad")
    eval_artifact = str(OUT / "adata_eval_scdiffeq_full_split.h5ad")
except Exception as e:
    print(f"Warning: could not write eval h5ad: {e}")
    np.savez(
        OUT / "adata_eval_scdiffeq_full_split_tlast.npz",
        X=np.array(adata_eval[tlast_mask].X),
        t=np.array(adata_eval[tlast_mask].obs["t"]),
    )
    eval_artifact = str(OUT / "adata_eval_scdiffeq_full_split_tlast.npz")

results = {
    "method": "scDiffeq",
    "scope": "full_larry_fixed_split",
    "data_path": str(DATA),
    "device": DEVICE,
    "torch_version": torch.__version__,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    "matmul_precision": torch.get_float32_matmul_precision(),
    "tf32_enabled": bool(torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32),
    "n_epochs": N_EPOCHS,
    "n_params": n_params,
    "split_counts": split_counts(adata),
    "train_time_s": round(train_time_s, 1) if train_time_s is not None else None,
    "eval_time_s": round(eval_time_s, 1),
    "load_time_s": round(load_time_s, 1) if load_time_s is not None else None,
    "train_time_source": train_time_source,
    "peak_gpu_mem_mb": peak_gpu_mem_mb,
    "peak_gpu_mem_source": peak_gpu_mem_source,
    "epochs_curve": epoch_curve,
    "sinkhorn_train_curve": train_curve,
    "sinkhorn_val_curve": val_curve,
    "sink_pca_by_time": sink_pca_by_time,
    "t6_pred_label_counts": pred_counts,
    "t6_true_label_counts": true_counts,
    "t6_pred_label_fractions": normalize_counts(pred_counts, label_set),
    "t6_true_label_fractions": normalize_counts(true_counts, label_set),
    "t6_label_tv_distance": t6_tv,
    "evaluation_plan_path": str(OUT / "evaluation_plan.json"),
    "model_log_dir": str(model_log_dir),
    "checkpoint_paths": checkpoint_paths,
    "last_checkpoint_path": str(last_ckpt) if last_ckpt.exists() else None,
    "metrics_path": str(metrics_path),
    "eval_artifact": eval_artifact,
}
save_json(OUT / "results_scdiffeq_full_split.json", results)

summary = results_markdown(
    "scDiffeq Full-LARRY Fixed-Split Benchmark",
    [
        ("Train time (s)", f"{train_time_s:.1f}"),
        ("Eval time (s)", f"{eval_time_s:.1f}"),
        ("Peak GPU mem (MB)", str(peak_gpu_mem_mb)),
        ("Sinkhorn t=4", f"{sink_pca_by_time.get('4.0', float('nan')):.4f}"),
        ("Sinkhorn t=6", f"{sink_pca_by_time.get('6.0', float('nan')):.4f}"),
        ("t6 label TV", f"{t6_tv:.4f}"),
        ("Last checkpoint", str(last_ckpt) if last_ckpt.exists() else "missing"),
    ],
)
save_text(OUT / "summary_scdiffeq_full_split.md", summary)
print(summary)
