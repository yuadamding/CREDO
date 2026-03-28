"""
Train CAPE (simplified, no perturbations) on LARRY data and save results for comparison.
Run with: conda run -n ml1 python run_cape_larry.py

Design notes:
- G=1 single group, ctrl embedding a_g = 0 (CAPE's control anchor trivially applies)
- Rollout: Euler-Maruyama with drift + diagonal diffusion + growth (log-weights)
- Training loss: geomloss Sinkhorn at t=4 and t=6 in PCA-50 space
- Fate prediction: kNN trained on t=6 observed cells only (clean comparison)
"""
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.neighbors import KNeighborsClassifier

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
N_EPOCHS = int(os.environ.get("LARRY_CAPE_EPOCHS", "1000"))
N_PARTICLES = int(os.environ.get("LARRY_CAPE_PARTICLES", "512"))
N_SIM = int(os.environ.get("LARRY_CAPE_SIM", "512"))
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

# ---------------------------------------------------------------------------
# 1. Load & filter data  (identical to scDiffeq demo)
# ---------------------------------------------------------------------------
import anndata as ad

print("Loading data …")
adata_ref = ad.read_h5ad(DATA)
nm_clones = adata_ref.uns["fate_counts"][["Monocyte", "Neutrophil"]].dropna().index
adata_ref.obs["nm_clones"] = adata_ref.obs["clone_idx"].isin(nm_clones)
MASK = (
    adata_ref.obs["Cell type annotation"].isin(["Monocyte", "Neutrophil", "Undifferentiated"])
    & adata_ref.obs["nm_clones"]
)
adata = adata_ref[MASK].copy()
print(f"Filtered adata: {adata.shape}")

pca_by_t, label_by_t = {}, {}
for t in [2.0, 4.0, 6.0]:
    mask = adata.obs["Time point"] == t
    pca_by_t[t] = torch.tensor(adata[mask].obsm["X_pca"], dtype=torch.float32, device=DEVICE)
    label_by_t[t] = adata[mask].obs["Cell type annotation"].values

# Starting pool: t=2 Undifferentiated
und_mask = (adata.obs["Time point"] == 2.0) & (adata.obs["Cell type annotation"] == "Undifferentiated")
z0_pool = torch.tensor(adata[und_mask].obsm["X_pca"], dtype=torch.float32, device=DEVICE)
print(f"t=2 undiff: {len(z0_pool)}  t=4: {len(pca_by_t[4.0])}  t=6: {len(pca_by_t[6.0])}")

# kNN classifier: train on t=6 only (clean fate assignment)
knn_clf = KNeighborsClassifier(n_neighbors=15, metric="euclidean", n_jobs=4)
knn_clf.fit(pca_by_t[6.0].cpu().numpy(), label_by_t[6.0])
print(f"kNN trained on t=6: {len(pca_by_t[6.0])} cells")
print(f"Using device={DEVICE} torch={torch.__version__} env={os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")

# ---------------------------------------------------------------------------
# 2. Build CAPE coefficient network
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
from cape.models.coefficients import CoefficientNetworks

coeff_nets = CoefficientNetworks(
    latent_dim=D_LATENT,
    embedding_dim=1,      # ctrl: a_g = 0
    context_dim=0,
    hidden_dim=256,
    depth=4,
    n_time_freqs=8,
    sigma_min=1e-3,
    r_max=2.0,            # growth/death for lineage bias
    ecological_growth=False,
).to(DEVICE)

a_g = torch.zeros(1, 1, device=DEVICE)   # [G=1, r=1] ctrl embedding
ctx = torch.zeros(0, device=DEVICE)       # no context

n_params = sum(p.numel() for p in coeff_nets.parameters())
print(f"CAPE params: {n_params:,}")

# ---------------------------------------------------------------------------
# 3. Euler-Maruyama rollout  (with growth → log-weights)
# ---------------------------------------------------------------------------
tau_steps = torch.tensor([0.0, 0.5, 1.0], device=DEVICE)

def rollout_em(
    z0: torch.Tensor,      # [N, d]
    tau_steps: torch.Tensor,
    return_logw: bool = False,
):
    """Returns list of z at each step; optionally also list of logw."""
    z   = z0.unsqueeze(0)                             # [1, N, d]
    logw = torch.full((1, z0.shape[0]), -np.log(z0.shape[0]),
                      device=DEVICE, dtype=torch.float32)  # [1, N]
    zs   = [z.squeeze(0)]
    lws  = [logw.squeeze(0)]
    for k in range(len(tau_steps) - 1):
        tau  = tau_steps[k]
        dtau = tau_steps[k + 1] - tau_steps[k]
        coeff = coeff_nets(z, tau, ctx, a_g)
        dW   = torch.randn_like(z) * dtau.sqrt()
        z    = z + coeff.drift * dtau + coeff.sigma_diag * dW
        logw = logw + coeff.growth.squeeze(0).unsqueeze(0) * dtau  # [1, N] += [1, N] * scalar
        zs.append(z.squeeze(0))
        lws.append(logw.squeeze(0))
    if return_logw:
        return zs, lws
    return zs

# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
sinkhorn_fn = SamplesLoss("sinkhorn", p=2, blur=0.1, scaling=0.7, debias=True)

optimizer = torch.optim.Adam(coeff_nets.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-4)

sink4_curve, sink6_curve, epoch_curve = [], [], []

print(f"Training CAPE {N_EPOCHS} epochs on {DEVICE} …")
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
t0 = time.time()

for epoch in range(N_EPOCHS):
    coeff_nets.train()

    idx0 = torch.randperm(len(z0_pool), device=DEVICE)[:N_PARTICLES]
    z0 = z0_pool[idx0]

    idx4 = torch.randperm(len(pca_by_t[4.0]), device=DEVICE)[:N_PARTICLES]
    idx6 = torch.randperm(len(pca_by_t[6.0]), device=DEVICE)[:N_PARTICLES]
    z4_obs = pca_by_t[4.0][idx4]
    z6_obs = pca_by_t[6.0][idx6]

    zs, lws = rollout_em(z0, tau_steps, return_logw=True)
    z4_sim, z6_sim   = zs[1], zs[2]
    lw4,   lw6       = lws[1], lws[2]

    # Normalized weights for weighted Sinkhorn
    w4 = torch.softmax(lw4, dim=0).unsqueeze(0)   # [1, N]
    w6 = torch.softmax(lw6, dim=0).unsqueeze(0)
    w_uniform = torch.full((1, N_PARTICLES), 1.0 / N_PARTICLES, device=DEVICE)

    # geomloss weighted: SamplesLoss()(w_x, x, w_y, y)
    loss4 = sinkhorn_fn(w4, z4_sim.unsqueeze(0), w_uniform, z4_obs.unsqueeze(0))
    loss6 = sinkhorn_fn(w6, z6_sim.unsqueeze(0), w_uniform, z6_obs.unsqueeze(0))
    loss  = loss4 + loss6

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(coeff_nets.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % LOG_INTERVAL == 0 or epoch == N_EPOCHS - 1:
        sink4_curve.append(loss4.item())
        sink6_curve.append(loss6.item())
        epoch_curve.append(epoch)
        if epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  sink4={loss4.item():.4f}  sink6={loss6.item():.4f}")

if DEVICE == "cuda":
    torch.cuda.synchronize()
train_time = time.time() - t0
train_peak_gpu_mem_mb = (
    round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if DEVICE == "cuda" else None
)
print(f"Training done in {train_time:.1f}s")

# ---------------------------------------------------------------------------
# 5. Post-training evaluation
# ---------------------------------------------------------------------------
coeff_nets.eval()
N_EVAL = 1000

if DEVICE == "cuda":
    torch.cuda.synchronize()
eval_t0 = time.time()

with torch.inference_mode():
    idx0_eval = torch.randperm(len(z0_pool), device=DEVICE)[:N_EVAL]
    z0_eval = z0_pool[idx0_eval]
    zs_eval, lws_eval = rollout_em(z0_eval, tau_steps, return_logw=True)
    z4_eval, z6_eval = zs_eval[1], zs_eval[2]
    lw4_eval, lw6_eval = lws_eval[1], lws_eval[2]

    w4_eval = torch.softmax(lw4_eval, dim=0).unsqueeze(0)
    w6_eval = torch.softmax(lw6_eval, dim=0).unsqueeze(0)
    w_uni   = torch.full((1, N_EVAL), 1.0 / N_EVAL, device=DEVICE)

    z4_obs_eval = pca_by_t[4.0][torch.randperm(len(pca_by_t[4.0]), device=DEVICE)[:N_EVAL]]
    z6_obs_eval = pca_by_t[6.0][torch.randperm(len(pca_by_t[6.0]), device=DEVICE)[:N_EVAL]]

    sink_pca_4 = sinkhorn_fn(w4_eval, z4_eval.unsqueeze(0), w_uni, z4_obs_eval.unsqueeze(0)).item()
    sink_pca_6 = sinkhorn_fn(w6_eval, z6_eval.unsqueeze(0), w_uni, z6_obs_eval.unsqueeze(0)).item()

if DEVICE == "cuda":
    torch.cuda.synchronize()
eval_time_s = time.time() - eval_t0

print(f"CAPE post-training Sinkhorn (PCA, weighted): t4={sink_pca_4:.4f}, t6={sink_pca_6:.4f}")

# ---------------------------------------------------------------------------
# 6. Fate prediction  (kNN on t=6 only)
# ---------------------------------------------------------------------------
np.random.seed(RANDOM_SEED)
prog_idx = np.random.choice(len(z0_pool), size=3, replace=False)
z_prog = z0_pool[torch.tensor(prog_idx, device=DEVICE)]

sim_cells, sim_weights = [], []
if DEVICE == "cuda":
    torch.cuda.synchronize()
fate_t0 = time.time()

with torch.inference_mode():
    for i in range(3):
        z_start = z_prog[i].unsqueeze(0).expand(N_SIM, -1)
        zs_sim, lws_sim = rollout_em(z_start, tau_steps, return_logw=True)
        sim_cells.append(zs_sim[-1].cpu().numpy())
        # Normalised weights for weighted fate counting
        w_norm = torch.softmax(lws_sim[-1], dim=0).cpu().numpy()
        sim_weights.append(w_norm)

z_sim_terminal = np.vstack(sim_cells)    # [3*N_SIM, d]
w_sim_all      = np.concatenate(sim_weights)  # [3*N_SIM]
# Renormalise
w_sim_all = w_sim_all / w_sim_all.sum()

pred_labels = knn_clf.predict(z_sim_terminal)
if DEVICE == "cuda":
    torch.cuda.synchronize()
fate_time_s = time.time() - fate_t0

# Both unweighted counts and weight-adjusted estimates
fate_counts_uw = Counter(pred_labels)
fate_weighted  = {}
for lbl in ["Monocyte", "Neutrophil", "Undifferentiated"]:
    mask = pred_labels == lbl
    fate_weighted[lbl] = float(w_sim_all[mask].sum())

print("CAPE fate counts (unweighted):", dict(fate_counts_uw))
print("CAPE fate fractions (weighted):", {k: f"{v:.3f}" for k, v in fate_weighted.items()})

mono = fate_counts_uw.get("Monocyte", 0)
neut = fate_counts_uw.get("Neutrophil", 0)
total_committed = mono + neut
fate_mono_frac  = mono / max(total_committed, 1)
# Also compute weighted version
mono_w = fate_weighted.get("Monocyte", 0)
neut_w = fate_weighted.get("Neutrophil", 0)
fate_mono_frac_w = mono_w / max(mono_w + neut_w, 1e-8)

# ---------------------------------------------------------------------------
# 7. Save trajectory for UMAP
# ---------------------------------------------------------------------------
with torch.no_grad():
    idx0_all  = torch.arange(min(2000, len(z0_pool)))
    z0_all    = z0_pool[idx0_all].to(DEVICE)
    zs_all, lws_all = rollout_em(z0_all, tau_steps, return_logw=True)

sim_traj = {
    "t2": zs_all[0].cpu().numpy(),
    "t4": zs_all[1].cpu().numpy(),
    "t6": zs_all[2].cpu().numpy(),
    "lw6": lws_all[2].cpu().numpy(),
}
np.savez(OUT / "cape_simulated_trajectory.npz", **sim_traj)

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------
results = {
    "method": "CAPE",
    "device": DEVICE,
    "torch_version": torch.__version__,
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    "n_epochs": N_EPOCHS,
    "n_particles": N_PARTICLES,
    "latent_dim": D_LATENT,
    "n_params": n_params,
    "train_time_s": round(train_time, 1),
    "eval_time_s": round(eval_time_s, 1),
    "fate_eval_time_s": round(fate_time_s, 1),
    "peak_gpu_mem_mb": train_peak_gpu_mem_mb,
    "matmul_precision": torch.get_float32_matmul_precision(),
    "tf32_enabled": bool(torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32),
    "sinkhorn_4_training_curve": [float(v) for v in sink4_curve],
    "sinkhorn_6_training_curve": [float(v) for v in sink6_curve],
    "epochs_curve": [int(e) for e in epoch_curve],
    "sink_pca_4": float(sink_pca_4),
    "sink_pca_6": float(sink_pca_6),
    "fate_monocyte":       int(fate_counts_uw.get("Monocyte", 0)),
    "fate_neutrophil":     int(fate_counts_uw.get("Neutrophil", 0)),
    "fate_undifferentiated": int(fate_counts_uw.get("Undifferentiated", 0)),
    "fate_mono_frac":      round(fate_mono_frac, 4),
    "fate_mono_frac_weighted": round(fate_mono_frac_w, 4),
}

with open(OUT / "results_cape.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved to", OUT / "results_cape.json")
print(json.dumps({k: v for k, v in results.items() if not k.endswith("_curve")}, indent=2))
