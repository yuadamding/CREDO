"""Sparse first projection and bounded full-gene multinomial decoder."""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from torch import Tensor, nn

from ..errors import ContractError
from ..representation.state_information import validate_count_csr
from .contracts import RepresentationRules


def sparse_tensor(matrix: sparse.csr_matrix, device: torch.device, *, normalize: bool) -> Tensor:
    counts = validate_count_csr(matrix)
    row = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    values = counts.data.astype(np.float32)
    if normalize:
        depth = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
        values = np.log1p(10000 * values / np.maximum(depth[row], 1)).astype(np.float32)
    indices = np.stack((row, counts.indices)).astype(np.int64)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device),
        torch.from_numpy(values).to(device),
        counts.shape,
        device=device,
    ).coalesce()


class SparseCountAutoencoder(nn.Module):
    """No donor/condition/guide inputs; library size conditions observation, not state."""

    center: Tensor
    scale: Tensor

    def __init__(self, features: int, rules: RepresentationRules, *, factor: bool = False) -> None:
        super().__init__()
        self.features, self.rules, self.factor = features, rules, factor
        dimension = rules.factor_rank if factor else rules.latent_dim
        h1, h2 = rules.hidden_dims
        self.first = nn.Linear(features, dimension if factor else h1)
        self.encoder = (
            nn.Identity()
            if factor
            else nn.Sequential(
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Linear(h1, h2),
                nn.LayerNorm(h2),
                nn.GELU(),
                nn.Linear(h2, dimension),
            )
        )
        self.decoder = (
            nn.Linear(dimension, features)
            if factor
            else nn.Sequential(
                nn.Linear(dimension, h2),
                nn.LayerNorm(h2),
                nn.GELU(),
                nn.Linear(h2, h1),
                nn.LayerNorm(h1),
                nn.GELU(),
                nn.Linear(h1, features),
            )
        )
        self.register_buffer("center", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))

    @property
    def device(self) -> torch.device:
        return self.first.weight.device

    def raw_encode(self, counts: sparse.csr_matrix) -> Tensor:
        if counts.shape[1] != self.features:
            raise ContractError("Canonical RNA feature dimension mismatch.")
        values = sparse_tensor(counts, self.device, normalize=True)
        return self.encoder(torch.sparse.mm(values, self.first.weight.T) + self.first.bias)

    def encode(self, counts: sparse.csr_matrix) -> Tensor:
        return (self.raw_encode(counts) - self.center) / self.scale

    def log_probabilities(self, counts: sparse.csr_matrix, *, ablate: bool = False) -> Tensor:
        latent = self.raw_encode(counts)
        if ablate:
            latent = torch.zeros_like(latent) + self.center
        return torch.log_softmax(self.decoder(latent), dim=-1)

    def decode(self, state: Tensor, exposure: Tensor) -> Tensor:
        """Expected counts at caller-declared RNA exposure; no endpoint depth inference."""
        if (
            state.ndim != 2
            or state.shape[1] != len(self.center)
            or exposure.shape != (len(state),)
            or not bool(torch.isfinite(state).all())
            or not bool(torch.isfinite(exposure).all())
            or bool(torch.any(exposure < 0))
        ):
            raise ContractError("Aligned finite states and nonnegative RNA exposures required.")
        return torch.softmax(self.decoder(state * self.scale + self.center), -1) * exposure[:, None]


def count_loss(log_probabilities: Tensor, target: sparse.csr_matrix) -> tuple[Tensor, Tensor]:
    """Equal-cell CE, excluding only explicitly zero-scored-half rows; raw B remains sparse."""
    if tuple(log_probabilities.shape) != target.shape:
        raise ContractError("Count score dimensions differ.")
    tensor = sparse_tensor(target, log_probabilities.device, normalize=False)
    row, column = tensor.indices()
    terms = -tensor.values() * log_probabilities[row, column]
    totals = torch.zeros(len(log_probabilities), device=log_probabilities.device)
    nll = torch.zeros_like(totals).index_add(0, row, terms)
    totals.index_add_(0, row, tensor.values())
    return nll / totals.clamp_min(1), totals


def initialize(
    rules: RepresentationRules, features: int, *, factor: bool = False
) -> SparseCountAutoencoder:
    if rules.device == "cuda" and not torch.cuda.is_available():
        raise ContractError("Explicit CUDA execution requested but unavailable; no CPU fallback.")
    devices = [torch.cuda.current_device()] if rules.device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(rules.seed)
        model = SparseCountAutoencoder(features, rules, factor=factor).to(rules.device)
    return model


def dense_budget(rules: RepresentationRules, features: int, rows: int) -> None:
    # Conservative minibatch activation estimate, not a device/RSS peak guarantee.
    needed = rows * (features * 8 + sum(rules.hidden_dims) * 16 + rules.latent_dim * 8) * 4
    if needed > rules.maximum_dense_working_bytes:
        raise ContractError("Full-gene decoder minibatch exceeds dense working budget.")
