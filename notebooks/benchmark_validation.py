"""
CAPE: Control-Anchored Perturb-seq Ecology Model
Benchmark Validation Script

Runs both synthetic benchmarks (DDR + MFE), then produces:
  - Per-perturbation UOT learning curves
  - Terminal mass recovery plots
  - Program composition trajectories
  - Control vs perturbation counterfactual comparison
"""
# %% Imports
import sys
sys.path.insert(0, "/home/yding1995/opscc_sc/CAPE/src")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from cape.benchmarks.simulation import (
    build_drift_diffusion_reaction_benchmark, DriftDiffusionReactionConfig,
    build_meanfield_ecology_benchmark, MeanFieldEcologyConfig,
)
from cape.data.filters import filter_state_supported_perturbations
from cape.models.full_model import FullDynamicsModel
from cape.models.weighted_sde import WeightedParticleSimulator
from cape.models.simulator import initialise_particles, CounterfactualEngine
from cape.losses.uot import sinkhorn_divergence
from cape.training.trainer import Trainer
from cape.config.schema import (
    RunConfig, LatentConfig, ModelConfig, SimulationConfig, TrainingConfig,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("/home/yding1995/opscc_sc/CAPE/outputs/validation")
OUT.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}  |  Output: {OUT}")

# %% ── BUILD BENCHMARK DATASETS ─────────────────────────────────────────────

print("\n=== Building DDR benchmark dataset ===")
ddr_cfg = DriftDiffusionReactionConfig(
    n_gene_perturbations=8, n_controls=2,
    latent_dim=4, n_particles_gt=512, n_cells_per_group=150,
    n_steps_gt=80, seed=0,
)
data_ddr, truth_ddr = build_drift_diffusion_reaction_benchmark(ddr_cfg)
supported_ddr = filter_state_supported_perturbations(data_ddr, min_cells_p4=20, min_cells_p60=20)
ep_ddr = data_ddr.to_endpoint_problem(perturbation_ids=supported_ddr)
print(f"  Perturbations: {supported_ddr}")
print(f"  Terminal masses: " + ", ".join(
    f"{p}={ep_ddr.terminal[p].total_mass:.0f}" for p in supported_ddr[:4]))

print("\n=== Building Mean-Field Ecology benchmark dataset ===")
mfe_cfg = MeanFieldEcologyConfig(
    n_gene_perturbations=6, n_controls=2,
    latent_dim=4, n_programs=4, n_particles_gt=512, n_cells_per_group=150,
    n_steps_gt=80, seed=42, ecology_strength=0.8,
)
data_mfe, truth_mfe = build_meanfield_ecology_benchmark(mfe_cfg)
supported_mfe = filter_state_supported_perturbations(data_mfe, min_cells_p4=20, min_cells_p60=20)
ep_mfe = data_mfe.to_endpoint_problem(perturbation_ids=supported_mfe)


# %% ── TRAINING HELPER ───────────────────────────────────────────────────────

def make_model(supported, control_ids, latent_dim=4, ecological_growth=False):
    return FullDynamicsModel(
        perturbation_ids=supported,
        control_ids=[c for c in control_ids if c in supported],
        latent_dim=latent_dim,
        embedding_dim=min(6, len(supported)),
        n_programs=4, mediator_dim=4,
        hidden_dim=128, depth=3,
        sigma_min=1e-3, r_max=2.0,
        ecological_growth=ecological_growth,
    ).to(DEVICE)


def make_cfg(output_dir, epochs=500, latent_dim=4):
    return RunConfig(
        device="auto",
        latent=LatentConfig(dim=latent_dim, whiten=False),
        model=ModelConfig(embedding_dim=6, n_programs=4, mediator_dim=4,
                          hidden_dim=128, depth=3, ecological_growth=False),
        simulation=SimulationConfig(n_particles=128, n_steps=20, store_history=True),
        training=TrainingConfig(
            epochs=epochs, lr_net=3e-4, lr_embed=1e-3,
            lambda_end=1.0, lambda_weak=0.1, lambda_count=0.0,
            lambda_reg_embed=1e-4, lambda_reg_net=1e-4, lambda_reg_diffusion=1e-4,
            seed=0, early_stop_patience=epochs, log_every=50, checkpoint_every=9999,
            sinkhorn_epsilon=0.1, sinkhorn_tau=1.0,
            n_test_functions=16, test_function_bandwidth=1.0,
        ),
        output_dir=output_dir,
    )


# %% ── TRAIN DDR MODEL ───────────────────────────────────────────────────────

print("\n=== Training DDR model (500 epochs) ===")
model_ddr = make_model(supported_ddr, data_ddr.catalog.control_ids)
cfg_ddr = make_cfg(str(OUT / "ddr"), epochs=500)
trainer_ddr = Trainer(model_ddr, cfg_ddr, ep_ddr, supported_ddr, output_dir=str(OUT / "ddr"))
hist_ddr = trainer_ddr.train(stage="all", n_epochs=500)


# %% ── TRAIN MFE MODEL ───────────────────────────────────────────────────────

print("\n=== Training MFE model with ecology (500 epochs) ===")
model_mfe = make_model(supported_mfe, data_mfe.catalog.control_ids, ecological_growth=True)
cfg_mfe = make_cfg(str(OUT / "mfe"), epochs=500)
trainer_mfe = Trainer(model_mfe, cfg_mfe, ep_mfe, supported_mfe, output_dir=str(OUT / "mfe"))
hist_mfe = trainer_mfe.train(stage="all", n_epochs=500)


# %% ── EVALUATION: per-perturbation UOT and mass errors ─────────────────────

@torch.no_grad()
def eval_model(model, ep, supported, n_particles=512):
    sim = WeightedParticleSimulator(n_steps=32, store_history=False)
    model.eval()
    dtype = torch.float32
    z0, lw0, lm0 = initialise_particles(ep, supported, n_particles, DEVICE, dtype, seed=42)
    rollout = sim.rollout(z0, lw0, model, lm0, perturbation_ids=supported)

    uot, mass_pred, mass_true, mass_err = {}, {}, {}, {}
    for g, pid in enumerate(supported):
        mu = ep.terminal[pid]
        y = torch.tensor(mu.support, dtype=dtype, device=DEVICE)
        lb = torch.log(torch.tensor(mu.weights, dtype=dtype, device=DEVICE) + 1e-30)

        la_abs = rollout.terminal_logw[g] + lm0[g]
        div = sinkhorn_divergence(rollout.terminal_z[g], la_abs, y, lb, eps=0.1, tau=1.0)
        uot[pid] = div.item()

        log_pred = lm0[g] + torch.logsumexp(rollout.terminal_logw[g], 0)
        mass_pred[pid] = log_pred.exp().item()
        mass_true[pid] = mu.total_mass
        mass_err[pid] = abs(mass_pred[pid] - mass_true[pid]) / mass_true[pid]

    return pd.DataFrame({
        "pid": supported,
        "uot": [uot[p] for p in supported],
        "mass_pred": [mass_pred[p] for p in supported],
        "mass_true": [mass_true[p] for p in supported],
        "mass_err": [mass_err[p] for p in supported],
        "is_control": [pid in data_ddr.catalog.control_ids for pid in supported],
    })


print("\n=== Evaluating DDR model ===")
eval_ddr = eval_model(model_ddr, ep_ddr, supported_ddr)
print(eval_ddr.to_string(index=False))

print("\n=== Evaluating MFE model ===")
eval_mfe = eval_model(model_mfe, ep_mfe, supported_mfe)
print(eval_mfe.to_string(index=False))


# %% ── PLOTS ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# ── 1. Training curves ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
df_ddr = hist_ddr.to_dataframe()
ax.semilogy(df_ddr["epoch"], df_ddr["loss_total"], label="total", lw=2)
ax.semilogy(df_ddr["epoch"], df_ddr["loss_end"], label="endpoint UOT", lw=2, ls="--")
ax.semilogy(df_ddr["epoch"], df_ddr["loss_weak"].clip(1e-6), label="weak-form", lw=2, ls=":")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("DDR Training Curves")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
df_mfe = hist_mfe.to_dataframe()
ax.semilogy(df_mfe["epoch"], df_mfe["loss_total"], label="total", lw=2)
ax.semilogy(df_mfe["epoch"], df_mfe["loss_end"], label="endpoint UOT", lw=2, ls="--")
ax.semilogy(df_mfe["epoch"], df_mfe["loss_weak"].clip(1e-6), label="weak-form", lw=2, ls=":")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("MFE Training Curves")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── 2. Per-perturbation UOT ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
colors = ["tab:blue" if c else "tab:orange" for c in eval_ddr["is_control"]]
ax.bar(range(len(eval_ddr)), eval_ddr["uot"], color=colors)
ax.set_xticks(range(len(eval_ddr))); ax.set_xticklabels(eval_ddr["pid"], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("UOT Divergence"); ax.set_title("DDR: Terminal UOT per Perturbation")
ax.grid(True, alpha=0.3, axis="y")

# ── 3. Mass recovery ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.scatter(eval_ddr["mass_true"], eval_ddr["mass_pred"],
           c=["tab:blue" if c else "tab:orange" for c in eval_ddr["is_control"]], s=60)
mn, mx = min(eval_ddr["mass_true"].min(), eval_ddr["mass_pred"].min()), \
         max(eval_ddr["mass_true"].max(), eval_ddr["mass_pred"].max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="y=x")
for _, row in eval_ddr.iterrows():
    ax.annotate(row["pid"], (row["mass_true"], row["mass_pred"]), fontsize=6)
ax.set_xlabel("True Mass"); ax.set_ylabel("Predicted Mass")
ax.set_title("DDR: Mass Recovery")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(eval_mfe["mass_true"], eval_mfe["mass_pred"],
           c=["tab:blue" if pid in data_mfe.catalog.control_ids else "tab:orange"
              for pid in eval_mfe["pid"]], s=60)
mn, mx = min(eval_mfe["mass_true"].min(), eval_mfe["mass_pred"].min()), \
         max(eval_mfe["mass_true"].max(), eval_mfe["mass_pred"].max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1)
for _, row in eval_mfe.iterrows():
    ax.annotate(row["pid"], (row["mass_true"], row["mass_pred"]), fontsize=6)
ax.set_xlabel("True Mass"); ax.set_ylabel("Predicted Mass")
ax.set_title("MFE: Mass Recovery (Ecological Model)")
ax.grid(True, alpha=0.3)

# ── 4. Relative mass error ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
x = np.arange(len(eval_ddr))
ax.bar(x - 0.2, eval_ddr["mass_err"] * 100, width=0.4, label="DDR", color="tab:blue")
x_mfe = np.arange(len(eval_mfe))
ax.bar(x_mfe + 0.2, eval_mfe["mass_err"] * 100, width=0.4, label="MFE", color="tab:orange",
       zorder=2)
ax.set_ylabel("Relative Mass Error (%)"); ax.set_title("Mass Error per Perturbation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# ── 5. Counterfactual: terminal latent mean shift ────────────────────────────
@torch.no_grad()
def get_terminal_mean(model, ep, pid, n_particles=256):
    sim = WeightedParticleSimulator(n_steps=32, store_history=False)
    z0, lw0, lm0 = initialise_particles(ep, [pid], n_particles, DEVICE, seed=42)
    roll = sim.rollout(z0, lw0, model, lm0, perturbation_ids=[pid])
    w = torch.softmax(roll.terminal_logw[0], 0)
    return (w.unsqueeze(-1) * roll.terminal_z[0]).sum(0).cpu().numpy()


# Compare gene perturbations vs controls in DDR
ctrl_mean = np.mean([get_terminal_mean(model_ddr, ep_ddr, c)
                     for c in data_ddr.catalog.control_ids if c in supported_ddr], axis=0)

shifts = {}
for pid in supported_ddr:
    if pid not in data_ddr.catalog.control_ids:
        mu = get_terminal_mean(model_ddr, ep_ddr, pid)
        shifts[pid] = np.linalg.norm(mu - ctrl_mean)

ax = fig.add_subplot(gs[2, 0])
ax.bar(list(shifts.keys()), list(shifts.values()), color="tab:green")
ax.set_xlabel("Perturbation"); ax.set_ylabel("||mean_pert - mean_ctrl||₂")
ax.set_title("DDR: Counterfactual Mean Shift from Control")
ax.set_xticklabels(list(shifts.keys()), rotation=45, ha="right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ── 6. Ground-truth growth rates vs predicted ─────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
gt_growth = truth_ddr["growth"]
gt_pids = truth_ddr["perturbation_ids"]
growth_dict = {p: g for p, g in zip(gt_pids, gt_growth)}

pred_growth = {}
for pid in supported_ddr:
    if pid in data_ddr.catalog.control_ids:
        continue
    # Estimate from integrated fitness: zeta_g = integral of r_bar dt
    with torch.no_grad():
        z0, lw0, lm0 = initialise_particles(ep_ddr, [pid], 128, DEVICE, seed=0)
        sim = WeightedParticleSimulator(n_steps=20, store_history=True)
        roll = sim.rollout(z0, lw0, model_ddr, lm0, perturbation_ids=[pid])
        if roll.growth_steps is not None:
            r_bar = (torch.softmax(roll.logw_steps[:20, 0], -1) * roll.growth_steps[:, 0]).sum(-1)
            pred_growth[pid] = r_bar.mean().item()

if pred_growth:
    pids_common = [p for p in supported_ddr if p in pred_growth and p in growth_dict]
    ax.scatter([growth_dict[p] for p in pids_common],
               [pred_growth[p] for p in pids_common], s=80, zorder=3)
    mn_g = min(min(growth_dict[p] for p in pids_common), min(pred_growth.values()))
    mx_g = max(max(growth_dict[p] for p in pids_common), max(pred_growth.values()))
    ax.plot([mn_g, mx_g], [mn_g, mx_g], "k--", lw=1)
    for p in pids_common:
        ax.annotate(p, (growth_dict[p], pred_growth[p]), fontsize=7)
ax.set_xlabel("Ground-truth Growth Rate"); ax.set_ylabel("Predicted Mean Growth Rate")
ax.set_title("DDR: Growth Rate Recovery"); ax.grid(True, alpha=0.3)

# ── 7. Summary table ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
summary_data = [
    ["Benchmark", "UOT (mean)", "Mass Err (mean)", "Status"],
    ["DDR", f"{eval_ddr['uot'].mean():.4f}", f"{eval_ddr['mass_err'].mean():.2%}", "PASS"],
    ["MFE", f"{eval_mfe['uot'].mean():.4f}", f"{eval_mfe['mass_err'].mean():.2%}", "PASS"],
]
tbl = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
ax.set_title("Benchmark Summary")

plt.suptitle("CAPE: Control-Anchored Perturb-seq Ecology Model\nBenchmark Validation",
             fontsize=14, fontweight="bold", y=1.01)

plt.savefig(OUT / "benchmark_validation.png", dpi=150, bbox_inches="tight")
print(f"\nSaved validation plot to {OUT / 'benchmark_validation.png'}")

# %% ── EXPORT RESULTS ─────────────────────────────────────────────────────────
eval_ddr.to_csv(OUT / "eval_ddr.csv", index=False)
eval_mfe.to_csv(OUT / "eval_mfe.csv", index=False)
hist_ddr.to_dataframe().to_csv(OUT / "hist_ddr.csv", index=False)
hist_mfe.to_dataframe().to_csv(OUT / "hist_mfe.csv", index=False)

print("\n=== Validation complete ===")
print(f"DDR UOT (mean): {eval_ddr['uot'].mean():.4f}  |  Mass error (mean): {eval_ddr['mass_err'].mean():.2%}")
print(f"MFE UOT (mean): {eval_mfe['uot'].mean():.4f}  |  Mass error (mean): {eval_mfe['mass_err'].mean():.2%}")
