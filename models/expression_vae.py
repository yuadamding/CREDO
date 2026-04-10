"""Expression-to-latent VAE helpers for CAPE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def log1p_normalize_expression_matrix(
    matrix: sp.spmatrix | np.ndarray,
    *,
    target_sum: float = 1e4,
) -> sp.csr_matrix | np.ndarray:
    """Library-size normalize then log1p transform expression values."""
    if sp.issparse(matrix):
        norm = matrix.tocsr(copy=True).astype(np.float32)
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
    totals = arr.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1.0)
    arr = np.log1p((arr / totals) * target_sum)
    return arr.astype(np.float32, copy=False)


def _dense_batch(matrix: sp.spmatrix | np.ndarray, rows: np.ndarray) -> np.ndarray:
    batch = matrix[rows]
    if sp.issparse(batch):
        batch = batch.toarray()
    return np.asarray(batch, dtype=np.float32)


class ExpressionVAE(nn.Module):
    """Simple Gaussian VAE over normalized expression features."""

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


@dataclass
class ExpressionVAETrainingSummary:
    input_dim: int
    latent_dim: int
    n_cells: int
    epochs: int
    batch_size: int
    device: str
    final_total_loss: float
    final_recon_loss: float
    final_kl_loss: float
    kl_weight: float


def fit_expression_vae(
    matrix: sp.spmatrix | np.ndarray,
    *,
    latent_dim: int,
    hidden_dim: int = 512,
    depth: int = 2,
    dropout: float = 0.1,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    kl_weight: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[ExpressionVAE, pd.DataFrame, ExpressionVAETrainingSummary]:
    n_cells, input_dim = matrix.shape
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

    rng = np.random.default_rng(seed)
    rows = np.arange(n_cells, dtype=np.int64)
    history_rows: list[dict] = []

    model.train()
    for epoch in range(1, max(int(epochs), 1) + 1):
        order = rng.permutation(rows)
        total_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        n_seen = 0

        for start in range(0, n_cells, batch_size):
            batch_rows = order[start : start + batch_size]
            x = torch.from_numpy(_dense_batch(matrix, batch_rows)).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + (kl_weight * kl)
            loss.backward()
            optimizer.step()

            batch_n = len(batch_rows)
            total_loss += float(loss.detach().cpu()) * batch_n
            recon_loss_total += float(recon_loss.detach().cpu()) * batch_n
            kl_loss_total += float(kl.detach().cpu()) * batch_n
            n_seen += batch_n

        history_rows.append(
            {
                "epoch": epoch,
                "loss_total": total_loss / max(n_seen, 1),
                "loss_recon": recon_loss_total / max(n_seen, 1),
                "loss_kl": kl_loss_total / max(n_seen, 1),
            }
        )

    history = pd.DataFrame(history_rows)
    last = history.iloc[-1].to_dict()
    summary = ExpressionVAETrainingSummary(
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        n_cells=int(n_cells),
        epochs=int(epochs),
        batch_size=int(batch_size),
        device=str(device),
        final_total_loss=float(last["loss_total"]),
        final_recon_loss=float(last["loss_recon"]),
        final_kl_loss=float(last["loss_kl"]),
        kl_weight=float(kl_weight),
    )
    return model, history, summary


@torch.no_grad()
def encode_expression_vae(
    model: ExpressionVAE,
    matrix: sp.spmatrix | np.ndarray,
    *,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    n_cells = matrix.shape[0]
    encoded = np.zeros((n_cells, model.latent_dim), dtype=np.float32)
    model = model.to(device)
    model.eval()

    for start in range(0, n_cells, batch_size):
        rows = np.arange(start, min(start + batch_size, n_cells), dtype=np.int64)
        x = torch.from_numpy(_dense_batch(matrix, rows)).to(device=device)
        mu, _ = model.encode(x)
        encoded[rows] = mu.detach().cpu().numpy().astype(np.float32, copy=False)
    return encoded
