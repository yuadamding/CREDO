"""
Load both result files and generate comparison plots.
Run with: conda run -n ml1 python plot_comparison.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import anndata as ad
import umap as umap_lib

OUT = Path("/home/yding1995/opscc_sc/CAPE/outputs/larry_comparison")
DATA = Path("/home/yding1995/opscc_sc/scDiffeq/KleinLabData/in_vitro/larry_package_like_no_download.h5ad")

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
with open(OUT / "results_scdiffeq.json") as f:
    sdq = json.load(f)
with open(OUT / "results_cape.json") as f:
    cape = json.load(f)

print("scDiffeq:", {k: v for k, v in sdq.items() if not k.endswith("_curve")})
print("CAPE    :", {k: v for k, v in cape.items() if not k.endswith("_curve")})

# ---------------------------------------------------------------------------
# Load observed data for UMAP
# ---------------------------------------------------------------------------
adata_ref = ad.read_h5ad(DATA)
nm_clones = adata_ref.uns["fate_counts"][["Monocyte", "Neutrophil"]].dropna().index
adata_ref.obs["nm_clones"] = adata_ref.obs["clone_idx"].isin(nm_clones)
MASK = (
    adata_ref.obs["Cell type annotation"].isin(["Monocyte", "Neutrophil", "Undifferentiated"])
    & adata_ref.obs["nm_clones"]
)
adata = adata_ref[MASK].copy()
obs_pca = adata.obsm["X_pca"]

CMAP = {"Undifferentiated": "#888888", "Neutrophil": "#023047", "Monocyte": "#F08700"}

print("Fitting UMAP on observed PCA …")
UMAP_model = umap_lib.UMAP(n_components=2, random_state=0)
obs_umap = UMAP_model.fit_transform(obs_pca)
adata.obsm["X_umap"] = obs_umap

# Load CAPE sim trajectory and project
cape_traj = np.load(OUT / "cape_simulated_trajectory.npz")
cape_t6_umap = UMAP_model.transform(cape_traj["t6"])

# Load scDiffeq sim adata
adata_sim_sdq = ad.read_h5ad(OUT / "adata_sim_scdiffeq.h5ad")
sdq_t6_umap = adata_sim_sdq[adata_sim_sdq.obs["t"] == 6.0].obsm.get("X_umap", None)
if sdq_t6_umap is None:
    sdq_t6_pca = np.array(adata_sim_sdq[adata_sim_sdq.obs["t"] == 6.0].X)
    sdq_t6_umap = UMAP_model.transform(sdq_t6_pca)

# ---------------------------------------------------------------------------
# Ground truth fate
# ---------------------------------------------------------------------------
gt_t6 = adata[adata.obs["Time point"] == 6.0]
gt_counts = gt_t6.obs["Cell type annotation"].value_counts()
gt_mono = int(gt_counts.get("Monocyte", 0))
gt_neut = int(gt_counts.get("Neutrophil", 0))
gt_total = gt_mono + gt_neut
gt_frac = gt_mono / max(gt_total, 1)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"scDiffeq": "#e63946", "CAPE": "#457b9d"}
METHODS = [("scDiffeq", sdq, COLORS["scDiffeq"]), ("CAPE", cape, COLORS["CAPE"])]

# --- Row 0: Training curves (Sinkhorn t=4 and t=6) ---
ax_s4 = fig.add_subplot(gs[0, 0])
ax_s6 = fig.add_subplot(gs[0, 1])

for name, res, col in METHODS:
    epochs = res["epochs_curve"]
    ax_s4.plot(epochs, res["sinkhorn_4_training_curve"], color=col, label=name, lw=2)
    ax_s6.plot(epochs, res["sinkhorn_6_training_curve"], color=col, label=name, lw=2)

ax_s4.set_title("Training Sinkhorn  (t = 4 d)", fontsize=11)
ax_s4.set_xlabel("Epoch")
ax_s4.set_ylabel("Sinkhorn divergence")
ax_s4.legend(fontsize=9)
ax_s4.set_yscale("log")

ax_s6.set_title("Training Sinkhorn  (t = 6 d)", fontsize=11)
ax_s6.set_xlabel("Epoch")
ax_s6.set_ylabel("Sinkhorn divergence")
ax_s6.legend(fontsize=9)
ax_s6.set_yscale("log")

note = ("*scDiffeq Sinkhorn in VAE latent space;\n"
        " CAPE Sinkhorn in PCA space (training loss)")
fig.text(0.02, 0.77, note, fontsize=7, color="#555555", style="italic",
         va="top", ha="left")

# --- Row 0: Final Sinkhorn bar chart (PCA space, fair comparison) ---
ax_bar = fig.add_subplot(gs[0, 2:])
labels = ["t = 4 d\n(PCA space)", "t = 6 d\n(PCA space)"]
sdq_vals = [sdq["sink_pca_4"], sdq["sink_pca_6"]]
cape_vals = [cape["sink_pca_4"], cape["sink_pca_6"]]
x = np.arange(len(labels))
w = 0.35
bars_sdq  = ax_bar.bar(x - w/2, sdq_vals, w, label="scDiffeq", color=COLORS["scDiffeq"], alpha=0.85)
bars_cape = ax_bar.bar(x + w/2, cape_vals, w, label="CAPE",     color=COLORS["CAPE"],    alpha=0.85)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels)
ax_bar.set_ylabel("Sinkhorn divergence (PCA space)")
ax_bar.set_title("Post-training Distribution Match\n(PCA space — identical evaluation)", fontsize=10)
ax_bar.legend(fontsize=9)
for bar in list(bars_sdq) + list(bars_cape):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

# --- Row 1: Fate prediction ---
ax_fate = fig.add_subplot(gs[1, 0])
fate_cats = ["Monocyte", "Neutrophil", "Undifferentiated"]
x_f = np.arange(len(fate_cats))
w_f = 0.25

def _fracs(res):
    total = res["fate_monocyte"] + res["fate_neutrophil"] + res["fate_undifferentiated"]
    return [
        res["fate_monocyte"] / max(total, 1),
        res["fate_neutrophil"] / max(total, 1),
        res["fate_undifferentiated"] / max(total, 1),
    ]

gt_total_all = gt_mono + gt_neut + int(gt_counts.get("Undifferentiated", 0))
gt_fracs = [
    gt_mono / max(gt_total_all, 1),
    gt_neut / max(gt_total_all, 1),
    gt_counts.get("Undifferentiated", 0) / max(gt_total_all, 1),
]

ax_fate.bar(x_f - w_f,   gt_fracs,         w_f, label="Observed (t=6)", color="#2d6a4f",  alpha=0.85)
ax_fate.bar(x_f,          _fracs(sdq),     w_f, label="scDiffeq",       color=COLORS["scDiffeq"], alpha=0.85)
ax_fate.bar(x_f + w_f,   _fracs(cape),    w_f, label="CAPE",            color=COLORS["CAPE"],    alpha=0.85)
ax_fate.set_xticks(x_f)
ax_fate.set_xticklabels(fate_cats, fontsize=9)
ax_fate.set_ylabel("Fraction of simulated cells")
ax_fate.set_title("Fate Prediction (from t=2 progenitors)", fontsize=10)
ax_fate.legend(fontsize=8)

# Mono/(Mono+Neut) ratio comparison
ax_mono = fig.add_subplot(gs[1, 1])
methods_mono = ["Observed\n(t=6)", "scDiffeq", "CAPE"]
mono_fracs   = [gt_frac, sdq["fate_mono_frac"], cape["fate_mono_frac"]]
cols_mono    = ["#2d6a4f", COLORS["scDiffeq"], COLORS["CAPE"]]
bars_m = ax_mono.bar(methods_mono, mono_fracs, color=cols_mono, alpha=0.85, width=0.5)
ax_mono.axhline(gt_frac, color="#2d6a4f", ls="--", lw=1.5, label=f"GT={gt_frac:.2f}")
ax_mono.set_ylabel("Monocyte / (Mono + Neutro)")
ax_mono.set_title("Lineage Bias Accuracy", fontsize=10)
ax_mono.set_ylim(0, 1)
for bar in bars_m:
    ax_mono.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

# --- Row 1-2: UMAP plots ---
def _scatter_umap(ax, title):
    for ct, col in CMAP.items():
        mask = adata.obs["Cell type annotation"] == ct
        xu = obs_umap[mask.values]
        ax.scatter(xu[:, 0], xu[:, 1], c=col, s=3, alpha=0.15, rasterized=True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_title(title, fontsize=10)

# UMAP: observed reference
ax_umap_ref = fig.add_subplot(gs[1, 2])
_scatter_umap(ax_umap_ref, "Observed UMAP")
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker="o", ls="", color=c, label=ct, markersize=6)
           for ct, c in CMAP.items()]
ax_umap_ref.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.7)

# UMAP: scDiffeq simulated at t=6
ax_umap_sdq = fig.add_subplot(gs[1, 3])
_scatter_umap(ax_umap_sdq, "scDiffeq simulated (t=6)")
ax_umap_sdq.scatter(sdq_t6_umap[:, 0], sdq_t6_umap[:, 1],
                    c="#e63946", s=5, alpha=0.4, rasterized=True, label="simulated")

# UMAP: CAPE simulated at t=6
ax_umap_cape = fig.add_subplot(gs[2, 0])
_scatter_umap(ax_umap_cape, "CAPE simulated (t=6)")
ax_umap_cape.scatter(cape_t6_umap[:, 0], cape_t6_umap[:, 1],
                     c="#457b9d", s=5, alpha=0.4, rasterized=True, label="simulated")

# --- Row 2: Summary table ---
ax_tbl = fig.add_subplot(gs[2, 1:])
ax_tbl.axis("off")
table_data = [
    ["Metric", "scDiffeq", "CAPE"],
    ["Sinkhorn t=4 (PCA)",
     f"{sdq['sink_pca_4']:.4f}",
     f"{cape['sink_pca_4']:.4f}"],
    ["Sinkhorn t=6 (PCA)",
     f"{sdq['sink_pca_6']:.4f}",
     f"{cape['sink_pca_6']:.4f}"],
    ["Mono/(Mono+Neut) [GT={:.2f}]".format(gt_frac),
     f"{sdq['fate_mono_frac']:.3f}",
     f"{cape['fate_mono_frac']:.3f}"],
    ["Lineage bias |error|",
     f"{abs(sdq['fate_mono_frac'] - gt_frac):.3f}",
     f"{abs(cape['fate_mono_frac'] - gt_frac):.3f}"],
    ["Training time",
     f"{sdq['train_time_s']:.0f}s",
     f"{cape['train_time_s']:.0f}s"],
    ["# Params", "—", f"{sum(1 for _ in range(1)):,}"],
]

tbl = ax_tbl.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.0, 1.0, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#e0e0e0")
        cell.set_text_props(weight="bold")

fig.suptitle(
    "LARRY Hematopoiesis: scDiffeq vs CAPE\n"
    "(Monocyte/Neutrophil lineage, n=40k cells, 500 training epochs)",
    fontsize=13, y=1.01,
)

fig.savefig(OUT / "larry_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT / 'larry_comparison.png'}")

# Also save summary CSV
import pandas as pd
summary = pd.DataFrame({
    "metric": ["sink_pca_4", "sink_pca_6", "fate_mono_frac", "lineage_bias_error"],
    "scDiffeq": [sdq["sink_pca_4"], sdq["sink_pca_6"], sdq["fate_mono_frac"],
                 abs(sdq["fate_mono_frac"] - gt_frac)],
    "CAPE":     [cape["sink_pca_4"], cape["sink_pca_6"], cape["fate_mono_frac"],
                 abs(cape["fate_mono_frac"] - gt_frac)],
})
summary.to_csv(OUT / "larry_comparison_summary.csv", index=False)
print(summary.to_string(index=False))
