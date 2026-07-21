"""Expression-to-latent VAE for CREDO.

This module provides a **representation-learning VAE** over log1p-normalized
scRNA-seq expression.  It is *not* a count-generative model (NB/ZINB); it
learns a nonlinear latent embedding via MSE reconstruction on library-size-
normalized, log1p-transformed features.  For biologically faithful generative
modelling of raw counts, consider an scVI-style decoder instead.

The intended usage is:

1. Select highly-variable genes from raw counts.
2. Library-normalize + log1p the raw count matrix.
3. Fit the VAE on **training cells only** (see ``fit_expression_vae``).
4. Encode train *and* held-out cells with the frozen encoder.
5. Z-score the latent using training-set mean/std (see ``standardize_latent``).
6. Persist the full artifact bundle (see ``VAEArtifactBundle``).
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def log1p_normalize_expression_matrix(
    matrix: sp.spmatrix | np.ndarray,
    *,
    target_sum: float = 1e4,
    library_totals: np.ndarray | None = None,
) -> sp.csr_matrix | np.ndarray:
    """Library-size normalize then log1p transform expression values.

    Expects **raw counts** as input.  Each cell is scaled so that its total
    count equals *target_sum* before applying ``log1p``.
    """
    if library_totals is not None:
        totals = np.asarray(library_totals, dtype=np.float32).reshape(-1)
        if totals.shape[0] != matrix.shape[0]:
            raise ValueError(
                "library_totals must have one entry per cell. "
                f"Got {totals.shape[0]} totals for {matrix.shape[0]} rows."
            )
    else:
        totals = None

    if sp.issparse(matrix):
        norm = matrix.tocsr(copy=True).astype(np.float32)
        if totals is None:
            totals = np.asarray(norm.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(
            target_sum,
            np.maximum(totals, 1.0),
            out=np.ones_like(totals, dtype=np.float32),
            where=totals > 0,
        )
        norm = norm.multiply(scale[:, None]).tocsr()
        norm.data = np.log1p(norm.data)
        return norm

    arr = np.asarray(matrix, dtype=np.float32)
    if totals is None:
        totals = arr.sum(axis=1, keepdims=True)
    else:
        totals = totals.reshape(-1, 1)
    totals = np.maximum(totals, 1.0)
    arr = np.log1p((arr / totals) * target_sum)
    return arr.astype(np.float32, copy=False)


def _dense_batch(matrix: sp.spmatrix | np.ndarray, rows: np.ndarray) -> np.ndarray:
    batch = matrix[rows]
    if sp.issparse(batch):
        batch = batch.toarray()
    return np.asarray(batch, dtype=np.float32)


def maybe_materialize_dense_matrix(
    matrix: sp.spmatrix | np.ndarray,
    *,
    max_gb: float = 4.0,
) -> tuple[sp.spmatrix | np.ndarray, bool]:
    """Materialize a dense float32 panel when it fits comfortably in host RAM."""
    if max_gb <= 0:
        return matrix, False
    n_rows, n_cols = matrix.shape
    needed_bytes = int(n_rows) * int(n_cols) * np.dtype(np.float32).itemsize
    if needed_bytes > int(max_gb * (1024 ** 3)):
        return matrix, False
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float32), True
    return np.asarray(matrix, dtype=np.float32), True


# ---------------------------------------------------------------------------
# VAE architecture
# ---------------------------------------------------------------------------

class ExpressionVAE(nn.Module):
    """Simple Gaussian VAE over log1p-normalized expression features.

    Architecture: MLP encoder -> (mu, logvar) -> reparameterize -> MLP decoder.
    Loss: MSE reconstruction + KL(q(z|x) || N(0, I)).

    This is a *representation-learning* model, not a count-generative model.
    The decoder reconstructs log-normalized expression, not raw counts.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)

        enc_layers: list[nn.Module] = []
        prev = self.input_dim
        for _ in range(max(self.depth, 1)):
            enc_layers.extend(
                [
                    nn.Linear(prev, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev = self.hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_head = nn.Linear(prev, self.latent_dim)
        self.logvar_head = nn.Linear(prev, self.latent_dim)

        dec_layers: list[nn.Module] = []
        prev = self.latent_dim
        for _ in range(max(self.depth, 1)):
            dec_layers.extend(
                [
                    nn.Linear(prev, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            prev = self.hidden_dim
        dec_layers.append(nn.Linear(prev, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

@dataclass
class ExpressionVAETrainingSummary:
    input_dim: int
    latent_dim: int
    n_train_cells: int
    n_val_cells: int
    epochs_trained: int
    max_epochs: int
    batch_size: int
    device: str
    final_train_loss: float
    final_train_recon: float
    final_train_kl: float
    best_val_loss: float
    best_epoch: int
    kl_weight: float
    kl_warmup_epochs: int
    early_stopped: bool
    seed: int


# ---------------------------------------------------------------------------
# Latent standardization
# ---------------------------------------------------------------------------

@dataclass
class LatentStandardization:
    """Training-set mean and std for z-scoring VAE latent dimensions."""
    mean: np.ndarray   # [latent_dim]
    std: np.ndarray    # [latent_dim]

    def transform(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mean) / (self.std + 1e-8)

    def inverse(self, z_std: np.ndarray) -> np.ndarray:
        return z_std * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> LatentStandardization:
        return cls(mean=np.array(d["mean"], dtype=np.float32),
                   std=np.array(d["std"], dtype=np.float32))

    @classmethod
    def fit(cls, z: np.ndarray) -> LatentStandardization:
        return cls(mean=z.mean(axis=0).astype(np.float32),
                   std=z.std(axis=0).astype(np.float32))


# ---------------------------------------------------------------------------
# Full artifact bundle
# ---------------------------------------------------------------------------

@dataclass
class VAEArtifactBundle:
    """Everything needed to reproduce VAE encoding in future sessions.

    Persisting this bundle guarantees that encoding is reproducible even
    if code, data, or environment change.
    """
    gene_names: list[str]
    source_layer: str | None
    target_sum: float
    vae_hyperparams: dict[str, Any]
    train_cell_indices: list[int]
    latent_standardization: LatentStandardization | None
    training_summary: ExpressionVAETrainingSummary
    requested_layer: str | None = None
    selected_gene_indices: list[int] = field(default_factory=list)
    kept_positions: list[int] | None = None
    split_manifest_hash: str | None = None
    commit_sha: str | None = None

    def save(
        self,
        directory: str | Path,
        model: ExpressionVAE,
        *,
        latent_all_std: np.ndarray | None = None,
    ) -> Path:
        """Save bundle to *directory*: metadata JSON + model state_dict."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), d / "vae_state_dict.pt")

        meta = {
            "gene_names": self.gene_names,
            "source_layer": self.source_layer,
            "requested_layer": self.requested_layer,
            "selected_gene_indices": self.selected_gene_indices,
            "target_sum": self.target_sum,
            "vae_hyperparams": self.vae_hyperparams,
            "train_cell_indices": self.train_cell_indices,
            "kept_positions": self.kept_positions,
            "split_manifest_hash": self.split_manifest_hash,
            "latent_standardization": (
                self.latent_standardization.to_dict()
                if self.latent_standardization is not None else None
            ),
            "training_summary": asdict(self.training_summary),
            "commit_sha": self.commit_sha,
        }
        with open(d / "vae_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        if latent_all_std is not None:
            np.save(d / "latent_all_std.npy", np.asarray(latent_all_std, dtype=np.float32))
        return d

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu") -> tuple[VAEArtifactBundle, ExpressionVAE]:
        """Load bundle and reconstruct the model."""
        d = Path(directory)
        with open(d / "vae_metadata.json") as f:
            meta = json.load(f)

        hp = meta["vae_hyperparams"]
        model = ExpressionVAE(
            input_dim=hp["input_dim"],
            latent_dim=hp["latent_dim"],
            hidden_dim=hp.get("hidden_dim", 512),
            depth=hp.get("depth", 2),
            dropout=hp.get("dropout", 0.1),
        )
        model.load_state_dict(torch.load(d / "vae_state_dict.pt", map_location=device, weights_only=True))
        model.eval()

        ls = meta.get("latent_standardization")
        lat_std = LatentStandardization.from_dict(ls) if ls is not None else None

        summary_dict = meta["training_summary"]
        summary = ExpressionVAETrainingSummary(**summary_dict)

        bundle = cls(
            gene_names=meta["gene_names"],
            source_layer=meta.get("source_layer"),
            requested_layer=meta.get("requested_layer"),
            selected_gene_indices=meta.get("selected_gene_indices", []),
            target_sum=meta["target_sum"],
            vae_hyperparams=hp,
            train_cell_indices=meta["train_cell_indices"],
            kept_positions=meta.get("kept_positions"),
            split_manifest_hash=meta.get("split_manifest_hash"),
            latent_standardization=lat_std,
            training_summary=summary,
            commit_sha=meta.get("commit_sha"),
        )
        return bundle, model


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def fit_expression_vae(
    matrix: sp.spmatrix | np.ndarray,
    *,
    latent_dim: int,
    hidden_dim: int = 512,
    depth: int = 2,
    dropout: float = 0.1,
    epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    kl_weight: float = 1e-3,
    kl_warmup_epochs: int = 20,
    val_frac: float = 0.1,
    early_stop_patience: int = 15,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    use_amp: bool = True,
    amp_dtype: str = "bf16",
) -> tuple[ExpressionVAE, pd.DataFrame, ExpressionVAETrainingSummary]:
    """Fit an expression VAE with validation, early stopping, and KL warmup.

    Parameters
    ----------
    matrix : sparse or dense
        Log1p-normalized expression matrix of shape ``[n_cells, n_genes]``.
        **Must** come from training cells only to prevent representation
        leakage into the held-out set.
    kl_warmup_epochs : int
        Linearly ramp KL weight from 0 to *kl_weight* over this many epochs.
    val_frac : float
        Fraction of training cells held out for validation / early stopping.
    early_stop_patience : int
        Stop if validation ELBO does not improve for this many epochs.
    grad_clip : float
        Max gradient norm for clipping.
    seed : int
        Random seed for torch, CUDA, and numpy for reproducibility.
    """
    # --- Reproducibility ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    n_cells, input_dim = matrix.shape
    if n_cells < 1:
        raise ValueError("VAE fitting requires at least one training cell.")

    # --- Train / validation split ---
    n_val = max(1, int(round(val_frac * n_cells))) if val_frac > 0 else 0
    if n_cells <= 1 and n_val > 0:
        raise ValueError(
            "VAE validation split requested but there are not enough training cells. "
            f"n_cells={n_cells}, val_frac={val_frac}."
        )
    if n_cells > 1:
        n_val = min(n_val, n_cells - 1)
    else:
        n_val = 0
    all_idx = rng.permutation(n_cells)
    val_idx = np.sort(all_idx[:n_val]) if n_val > 0 else np.array([], dtype=np.int64)
    train_idx = np.sort(all_idx[n_val:])
    n_train = len(train_idx)
    if n_train < 1:
        raise ValueError(
            "VAE fitting produced an empty training split. "
            f"n_cells={n_cells}, val_frac={val_frac}, n_val={n_val}."
        )

    model = ExpressionVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    amp_enabled = bool(use_amp and str(device).startswith("cuda"))
    amp_torch_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler_enabled = amp_enabled and amp_torch_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    history_rows: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    early_stopped = False
    max_epochs = max(int(epochs), 1)
    selection_start_epoch = min(max_epochs, max(int(kl_warmup_epochs), 1))

    for epoch in range(1, max_epochs + 1):
        # --- KL warmup: linear ramp from 0 to kl_weight ---
        if kl_warmup_epochs > 0 and epoch <= kl_warmup_epochs:
            effective_kl_weight = kl_weight * (epoch / kl_warmup_epochs)
        else:
            effective_kl_weight = kl_weight

        # --- Training ---
        model.train()
        order = rng.permutation(n_train)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        recon_total = torch.zeros((), device=device, dtype=torch.float32)
        kl_total = torch.zeros((), device=device, dtype=torch.float32)
        n_seen = 0

        for start in range(0, n_train, batch_size):
            batch_rows = train_idx[order[start : start + batch_size]]
            x = torch.from_numpy(_dense_batch(matrix, batch_rows)).to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.autocast(device_type="cuda", dtype=amp_torch_dtype)
                if amp_enabled else
                nullcontext()
            ):
                recon, mu, logvar = model(x)
                recon_loss = F.mse_loss(recon, x, reduction="mean")
                kl = -0.5 * torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                )
                loss = recon_loss + effective_kl_weight * kl
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_n = len(batch_rows)
            total_loss = total_loss + loss.detach().float() * batch_n
            recon_total = recon_total + recon_loss.detach().float() * batch_n
            kl_total = kl_total + kl.detach().float() * batch_n
            n_seen += batch_n

        train_metrics = {
            "epoch": epoch,
            "loss_total": float((total_loss / max(n_seen, 1)).cpu()),
            "loss_recon": float((recon_total / max(n_seen, 1)).cpu()),
            "loss_kl": float((kl_total / max(n_seen, 1)).cpu()),
            "kl_weight_eff": effective_kl_weight,
        }

        # --- Validation ---
        val_loss_avg = float("nan")
        should_stop = False
        train_metrics["selection_eligible"] = epoch >= selection_start_epoch
        if n_val > 0:
            model.eval()
            val_loss = torch.zeros((), device=device, dtype=torch.float32)
            val_recon = torch.zeros((), device=device, dtype=torch.float32)
            val_kl = torch.zeros((), device=device, dtype=torch.float32)
            val_seen = 0
            with torch.no_grad():
                for start in range(0, n_val, batch_size):
                    batch_rows = val_idx[start : start + batch_size]
                    x = torch.from_numpy(_dense_batch(matrix, batch_rows)).to(device=device, non_blocking=True)
                    with (
                        torch.autocast(device_type="cuda", dtype=amp_torch_dtype)
                        if amp_enabled else
                        nullcontext()
                    ):
                        recon, mu, logvar = model(x)
                        recon_loss = F.mse_loss(recon, x, reduction="mean")
                        kl = -0.5 * torch.mean(
                            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                        )
                        loss = recon_loss + effective_kl_weight * kl
                    bn = len(batch_rows)
                    val_loss = val_loss + loss.detach().float() * bn
                    val_recon = val_recon + recon_loss.detach().float() * bn
                    val_kl = val_kl + kl.detach().float() * bn
                    val_seen += bn
            val_loss_avg = float((val_loss / max(val_seen, 1)).cpu())
            train_metrics["val_loss"] = val_loss_avg
            train_metrics["val_recon"] = float((val_recon / max(val_seen, 1)).cpu())
            train_metrics["val_kl"] = float((val_kl / max(val_seen, 1)).cpu())

            # Compare checkpoints only once the KL objective has reached its
            # final weight. Comparing losses during warmup favors early,
            # under-regularized checkpoints because the objective itself is
            # still increasing.
            if epoch >= selection_start_epoch:
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    should_stop = patience_counter >= early_stop_patience
        else:
            # No validation — track training loss for best model
            tl = train_metrics["loss_total"]
            if epoch >= selection_start_epoch and tl < best_val_loss:
                best_val_loss = tl
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history_rows.append(train_metrics)
        if should_stop:
            early_stopped = True
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    history = pd.DataFrame(history_rows)
    last_train = history.iloc[-1].to_dict()
    summary = ExpressionVAETrainingSummary(
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        n_train_cells=int(n_train),
        n_val_cells=int(n_val),
        epochs_trained=int(len(history)),
        max_epochs=max_epochs,
        batch_size=int(batch_size),
        device=str(device),
        final_train_loss=float(last_train["loss_total"]),
        final_train_recon=float(last_train["loss_recon"]),
        final_train_kl=float(last_train["loss_kl"]),
        best_val_loss=float(best_val_loss),
        best_epoch=int(best_epoch),
        kl_weight=float(kl_weight),
        kl_warmup_epochs=int(kl_warmup_epochs),
        early_stopped=early_stopped,
        seed=int(seed),
    )
    return model, history, summary


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_expression_vae(
    model: ExpressionVAE,
    matrix: sp.spmatrix | np.ndarray,
    *,
    batch_size: int = 4096,
    device: str = "cpu",
    use_amp: bool = True,
    amp_dtype: str = "bf16",
) -> np.ndarray:
    """Encode expression matrix to latent means using a trained VAE.

    Returns the posterior mean ``mu`` (deterministic encoding).
    """
    n_cells = matrix.shape[0]
    encoded = np.zeros((n_cells, model.latent_dim), dtype=np.float32)
    model = model.to(device)
    model.eval()
    amp_enabled = bool(use_amp and str(device).startswith("cuda"))
    amp_torch_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    for start in range(0, n_cells, batch_size):
        rows = np.arange(start, min(start + batch_size, n_cells), dtype=np.int64)
        x = torch.from_numpy(_dense_batch(matrix, rows)).to(device=device, non_blocking=True)
        with (
            torch.autocast(device_type="cuda", dtype=amp_torch_dtype)
            if amp_enabled else
            nullcontext()
        ):
            mu, _ = model.encode(x)
        encoded[rows] = mu.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return encoded


def standardize_latent(
    z: np.ndarray,
    stats: LatentStandardization | None = None,
) -> tuple[np.ndarray, LatentStandardization]:
    """Z-score latent dimensions.

    If *stats* is None, fit on *z* (training set).  Otherwise apply the
    given training-set statistics (for encoding held-out data).
    """
    if stats is None:
        stats = LatentStandardization.fit(z)
    return stats.transform(z), stats
