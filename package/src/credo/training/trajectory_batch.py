"""Batch helpers for multi-time trajectory training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..data.core import MeasureKey
from ..data.trajectory_view import TrajectoryLike, embedding_id_for_measure_key


@dataclass(frozen=True)
class TrajectoryBatch:
    measure_keys: list[MeasureKey]
    embedding_ids: list[str]
    source_label: str
    target_labels: list[str]
    tau_grid: torch.Tensor
    checkpoint_indices: dict[str, int]


def initialise_particles_from_trajectory(
    trajectory: TrajectoryLike,
    source_label: str,
    measure_keys: list[MeasureKey],
    n_particles: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample particles from a trajectory source checkpoint.

    Sampling uses the observed finite-measure weights.  The returned
    ``logw0`` is relative within each source measure, while ``log_m0`` stores
    the absolute source mass.  Downstream endpoint losses reconstruct absolute
    weights as ``log_m0[:, None] + logw_steps[idx]``.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")
    if not measure_keys:
        raise ValueError("measure_keys must not be empty")

    first = trajectory.get(source_label, measure_keys[0])
    d = first.latent_dim
    G = len(measure_keys)
    z0 = torch.zeros(G, n_particles, d, dtype=dtype, device=device)
    logw0 = torch.full(
        (G, n_particles),
        -np.log(float(n_particles)),
        dtype=dtype,
        device=device,
    )
    log_m0 = torch.zeros(G, dtype=dtype, device=device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(int(seed))

    for g, key in enumerate(measure_keys):
        mu = trajectory.get(source_label, key)
        support = torch.as_tensor(mu.support, dtype=dtype, device=device)
        weights = torch.as_tensor(mu.weights, dtype=dtype, device=device)
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, n_particles, replacement=True, generator=generator)
        z0[g] = support[idx]
        log_m0[g] = torch.as_tensor(np.log(mu.total_mass), dtype=dtype, device=device)

    return z0, logw0, log_m0


def embedding_ids_for_measure_keys(measure_keys: list[MeasureKey]) -> list[str]:
    return [embedding_id_for_measure_key(key) for key in measure_keys]


__all__ = [
    "TrajectoryBatch",
    "embedding_ids_for_measure_keys",
    "initialise_particles_from_trajectory",
]
