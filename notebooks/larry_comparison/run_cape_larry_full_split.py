"""
Fair CAPE benchmark on the fixed full-LARRY train/val/test split.

Run with:
  conda run -n ml1 python run_cape_larry_full_split.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
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
    sync_cuda_if_needed,
    timed_call,
    total_variation_distance,
)


sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from cape.models.coefficients import CoefficientNetworks


DATA = get_split_data_path()
OUT = get_output_root() / "cape"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS = int(os.environ.get("LARRY_FULL_BENCHMARK_EPOCHS", "500"))
N_PARTICLES = int(os.environ.get("LARRY_CAPE_PARTICLES", "512"))
N_EVAL = int(os.environ.get("LARRY_FULL_BENCHMARK_EVAL", "1000"))
D_LATENT = int(os.environ.get("LARRY_CAPE_LATENT_DIM", "50"))
RANDOM_SEED = int(os.environ.get("LARRY_COMPARISON_SEED", "0"))
LOG_INTERVAL = 20

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
adata = load_split_adata(DATA)
print(f"adata: {adata.shape}")
print(f"split_counts: {split_counts(adata)}")
print(f"Using device={DEVICE} torch={torch.__version__} env={os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

times = sorted(float(t) for t in adata.obs[TIME_KEY].unique().tolist())
t0 = min(times)
tau_steps = torch.tensor(np.linspace(0.0, 1.0, len(times)), device=DEVICE, dtype=torch.float32)

train_by_t = {}
val_by_t = {}
test_by_t = {}
for t in times:
    train_mask = (adata.obs["split"] == "train") & (adata.obs[TIME_KEY] == t)
    val_mask = (adata.obs["split"] == "val") & (adata.obs[TIME_KEY] == t)
    test_mask = (adata.obs["split"] == "test") & (adata.obs[TIME_KEY] == t)
    train_by_t[t] = torch.tensor(adata[train_mask].obsm[USE_KEY], dtype=torch.float32, device=DEVICE)
    val_by_t[t] = torch.tensor(adata[val_mask].obsm[USE_KEY], dtype=torch.float32, device=DEVICE)
    test_by_t[t] = torch.tensor(adata[test_mask].obsm[USE_KEY], dtype=torch.float32, device=DEVICE)

z0_pool = train_by_t[t0]
print("train sizes by time:", {f"{t:.1f}": int(len(train_by_t[t])) for t in times})
print("val sizes by time:", {f"{t:.1f}": int(len(val_by_t[t])) for t in times})
print("test sizes by time:", {f"{t:.1f}": int(len(test_by_t[t])) for t in times})

coeff_nets = CoefficientNetworks(
    latent_dim=D_LATENT,
    embedding_dim=1,
    context_dim=0,
    hidden_dim=256,
    depth=4,
    n_time_freqs=8,
    sigma_min=1e-3,
    r_max=2.0,
    ecological_growth=False,
).to(DEVICE)

a_g = torch.zeros(1, 1, device=DEVICE)
ctx = torch.zeros(0, device=DEVICE)
n_params = sum(p.numel() for p in coeff_nets.parameters())
print(f"CAPE params: {n_params:,}")


def rollout_em(z0: torch.Tensor, tau_steps: torch.Tensor, return_logw: bool = False):
    z = z0.unsqueeze(0)
    logw = torch.full(
        (1, z0.shape[0]),
        -np.log(z0.shape[0]),
        device=DEVICE,
        dtype=torch.float32,
    )
    zs = [z.squeeze(0)]
    lws = [logw.squeeze(0)]
    for k in range(len(tau_steps) - 1):
        tau = tau_steps[k]
        dtau = tau_steps[k + 1] - tau_steps[k]
        coeff = coeff_nets(z, tau, ctx, a_g)
        dW = torch.randn_like(z) * dtau.sqrt()
        z = z + coeff.drift * dtau + coeff.sigma_diag * dW
        logw = logw + coeff.growth.squeeze(0).unsqueeze(0) * dtau
        zs.append(z.squeeze(0))
        lws.append(logw.squeeze(0))
    if return_logw:
        return zs, lws
    return zs


sinkhorn_fn = SamplesLoss("sinkhorn", p=2, blur=0.1, scaling=0.7, debias=True)
optimizer = torch.optim.Adam(coeff_nets.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-4)

train_curve = {f"{t:.1f}": [] for t in times if t != t0}
val_curve = {f"{t:.1f}": [] for t in times if t != t0}
epoch_curve = []

print(f"Training CAPE on train split for {N_EPOCHS} epochs …")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
sync_cuda_if_needed(DEVICE)
t0_train = time.time()

for epoch in range(N_EPOCHS):
    coeff_nets.train()
    idx0 = torch.randperm(len(z0_pool), device=DEVICE)[: min(N_PARTICLES, len(z0_pool))]
    z0 = z0_pool[idx0]
    zs, lws = rollout_em(z0, tau_steps, return_logw=True)

    losses = []
    for step_i, t in enumerate(times[1:], start=1):
        n_obs = min(N_PARTICLES, len(train_by_t[t]))
        idx_obs = torch.randperm(len(train_by_t[t]), device=DEVICE)[:n_obs]
        z_obs = train_by_t[t][idx_obs]
        z_sim = zs[step_i][:n_obs]
        lw = lws[step_i][:n_obs]
        w_sim = torch.softmax(lw, dim=0).unsqueeze(0)
        w_obs = torch.full((1, n_obs), 1.0 / n_obs, device=DEVICE)
        loss_t = sinkhorn_fn(w_sim, z_sim.unsqueeze(0), w_obs, z_obs.unsqueeze(0))
        losses.append(loss_t)

    loss = sum(losses)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(coeff_nets.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % LOG_INTERVAL == 0 or epoch == N_EPOCHS - 1:
        coeff_nets.eval()
        epoch_curve.append(epoch)
        with torch.inference_mode():
            idx0_eval = torch.randperm(len(z0_pool), device=DEVICE)[: min(N_PARTICLES, len(z0_pool))]
            z0_eval = z0_pool[idx0_eval]
            zs_eval, lws_eval = rollout_em(z0_eval, tau_steps, return_logw=True)
            for step_i, t in enumerate(times[1:], start=1):
                key = f"{t:.1f}"

                n_train_obs = min(N_PARTICLES, len(train_by_t[t]))
                idx_train = torch.randperm(len(train_by_t[t]), device=DEVICE)[:n_train_obs]
                z_train_obs = train_by_t[t][idx_train]
                z_train_sim = zs_eval[step_i][:n_train_obs]
                lw_train = lws_eval[step_i][:n_train_obs]
                w_train_sim = torch.softmax(lw_train, dim=0).unsqueeze(0)
                w_train_obs = torch.full((1, n_train_obs), 1.0 / n_train_obs, device=DEVICE)
                train_curve[key].append(
                    float(sinkhorn_fn(w_train_sim, z_train_sim.unsqueeze(0), w_train_obs, z_train_obs.unsqueeze(0)).item())
                )

                n_val_obs = min(N_PARTICLES, len(val_by_t[t]))
                idx_val = torch.randperm(len(val_by_t[t]), device=DEVICE)[:n_val_obs]
                z_val_obs = val_by_t[t][idx_val]
                z_val_sim = zs_eval[step_i][:n_val_obs]
                lw_val = lws_eval[step_i][:n_val_obs]
                w_val_sim = torch.softmax(lw_val, dim=0).unsqueeze(0)
                w_val_obs = torch.full((1, n_val_obs), 1.0 / n_val_obs, device=DEVICE)
                val_curve[key].append(
                    float(sinkhorn_fn(w_val_sim, z_val_sim.unsqueeze(0), w_val_obs, z_val_obs.unsqueeze(0)).item())
                )

        if epoch % 100 == 0:
            brief = " ".join(f"t{int(float(k))}={train_curve[k][-1]:.2f}" for k in sorted(train_curve))
            print(f"  epoch {epoch:4d} {brief}")

sync_cuda_if_needed(DEVICE)
train_time_s = time.time() - t0_train
peak_gpu_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if DEVICE == "cuda" else None
print(f"Training done in {train_time_s:.1f}s")

eval_plan = build_eval_plan(adata, seed=RANDOM_SEED, n_eval=N_EVAL)
save_json(OUT / "evaluation_plan.json", eval_plan)
z0_eval, targets_eval = plan_to_tensors(adata, eval_plan, device=DEVICE)

coeff_nets.eval()


def run_test_rollout():
    return rollout_em(z0_eval, tau_steps, return_logw=True)


(zs_test, lws_test), eval_time_s = timed_call(run_test_rollout, DEVICE)
sink_pca_by_time = {}
for step_i, t in enumerate(times[1:], start=1):
    z_sim = zs_test[step_i]
    lw_sim = lws_test[step_i]
    w_sim = torch.softmax(lw_sim, dim=0).unsqueeze(0)
    w_obs = torch.full((1, len(targets_eval[t])), 1.0 / len(targets_eval[t]), device=DEVICE)
    sink_pca_by_time[f"{t:.1f}"] = float(
        sinkhorn_fn(w_sim, z_sim.unsqueeze(0), w_obs, targets_eval[t].unsqueeze(0)).item()
    )

train_t6_mask = (adata.obs["split"] == "train") & (adata.obs[TIME_KEY] == max(times))
train_t6_pca = adata[train_t6_mask].obsm[USE_KEY]
train_t6_labels = adata[train_t6_mask].obs[LABEL_KEY].values
knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=4)
knn.fit(train_t6_pca, train_t6_labels)

t6_sim = zs_test[-1].detach().cpu().numpy()
pred_labels = knn.predict(t6_sim)
pred_counts = counter_to_dict(pred_labels)
true_counts = observed_label_counts(adata, eval_plan["test_t6_idx"])
label_set = sorted(set(true_counts) | set(pred_counts))
t6_tv = total_variation_distance(pred_counts, true_counts, label_set)

model_path = OUT / "cape_model_full_split.pt"
torch.save(
    {
        "state_dict": coeff_nets.state_dict(),
        "latent_dim": D_LATENT,
        "n_params": n_params,
        "tau_steps": tau_steps.detach().cpu(),
        "seed": RANDOM_SEED,
        "split_data": str(DATA),
    },
    model_path,
)

history = {
    "epoch": epoch_curve,
    **{f"train_sinkhorn_{k}": v for k, v in train_curve.items()},
    **{f"val_sinkhorn_{k}": v for k, v in val_curve.items()},
}
save_json(OUT / "training_history.json", history)

results = {
    "method": "CAPE",
    "scope": "full_larry_fixed_split",
    "data_path": str(DATA),
    "device": DEVICE,
    "torch_version": torch.__version__,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    "matmul_precision": torch.get_float32_matmul_precision(),
    "tf32_enabled": bool(torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32),
    "n_epochs": N_EPOCHS,
    "n_particles": N_PARTICLES,
    "latent_dim": D_LATENT,
    "n_params": n_params,
    "split_counts": split_counts(adata),
    "train_time_s": round(train_time_s, 1),
    "eval_time_s": round(eval_time_s, 1),
    "peak_gpu_mem_mb": peak_gpu_mem_mb,
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
    "model_path": str(model_path),
}
save_json(OUT / "results_cape_full_split.json", results)

summary = results_markdown(
    "CAPE Full-LARRY Fixed-Split Benchmark",
    [
        ("Train time (s)", f"{train_time_s:.1f}"),
        ("Eval time (s)", f"{eval_time_s:.1f}"),
        ("Peak GPU mem (MB)", str(peak_gpu_mem_mb)),
        ("Sinkhorn t=4", f"{sink_pca_by_time.get('4.0', float('nan')):.4f}"),
        ("Sinkhorn t=6", f"{sink_pca_by_time.get('6.0', float('nan')):.4f}"),
        ("t6 label TV", f"{t6_tv:.4f}"),
        ("Model path", str(model_path)),
    ],
)
save_text(OUT / "summary_cape_full_split.md", summary)
print(summary)
