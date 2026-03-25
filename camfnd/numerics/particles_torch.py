from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from camfnd.data.contract import FiniteMeasure


@dataclass(slots=True)
class TorchParticleState:
    z: Tensor
    logw: Tensor
    perturbation_id: str
    sample_id: str
    mass0: float
    particles_per_atom: int = 1

    def validate(self) -> None:
        if self.z.ndim != 2 or self.z.shape[1] <= 0:
            raise ValueError("z must have shape [N, d] with d >= 1.")
        if self.logw.ndim != 1 or self.logw.shape[0] != self.z.shape[0]:
            raise ValueError("logw must have shape [N] matching z rows.")
        if not torch.isfinite(self.z).all() or not torch.isfinite(self.logw).all():
            raise ValueError("Particle state contains non-finite values.")
        if self.mass0 <= 0:
            raise ValueError("mass0 must be positive.")
        if self.particles_per_atom <= 0:
            raise ValueError("particles_per_atom must be positive.")

    @classmethod
    def from_measure(
        cls,
        measure: FiniteMeasure,
        *,
        particles_per_atom: int = 1,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> "TorchParticleState":
        measure.validate()
        if int(particles_per_atom) <= 0:
            raise ValueError("particles_per_atom must be positive.")
        support = torch.as_tensor(measure.support, dtype=dtype, device=device)
        if particles_per_atom > 1:
            support = support.repeat_interleave(particles_per_atom, dim=0)
        logw = torch.zeros(support.shape[0], dtype=dtype, device=device)
        state = cls(
            z=support,
            logw=logw,
            perturbation_id=measure.perturbation_id,
            sample_id=measure.sample_id,
            mass0=float(measure.total_mass),
            particles_per_atom=int(particles_per_atom),
        )
        state.validate()
        return state

    @property
    def n_particles(self) -> int:
        return int(self.z.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.z.shape[1])

    def atom_weights(self) -> Tensor:
        return (self.mass0 / self.n_particles) * torch.exp(self.logw)

    def total_mass(self) -> Tensor:
        return self.atom_weights().sum()

    def normalized_weights(self) -> Tensor:
        weights = self.atom_weights()
        return weights / weights.sum()

    def mean(self) -> Tensor:
        w = self.normalized_weights()[:, None]
        return (w * self.z).sum(dim=0)

    def variance_trace(self) -> Tensor:
        mu = self.mean()
        centered = self.z - mu[None, :]
        w = self.normalized_weights()[:, None]
        cov = centered.T @ (centered * w)
        return torch.trace(cov)

    def summary(self) -> dict:
        summary = {
            "n_particles": self.n_particles,
            "latent_dim": self.latent_dim,
            "mass": float(self.total_mass().detach().cpu()),
            "mean_0": float(self.mean()[0].detach().cpu()),
            "var_trace": float(self.variance_trace().detach().cpu()),
        }
        mean = self.mean().detach().cpu()
        for dim_idx in range(1, self.latent_dim):
            summary[f"mean_{dim_idx}"] = float(mean[dim_idx])
        return summary

    def to_measure(self, *, time_label: str) -> FiniteMeasure:
        support = self.z.detach().cpu().numpy()
        weights = self.atom_weights().detach().cpu().numpy()
        measure = FiniteMeasure(
            support=support.astype(float),
            weights=weights.astype(float),
            total_mass=float(weights.sum()),
            perturbation_id=self.perturbation_id,
            time_label=str(time_label),
            sample_id=self.sample_id,
        )
        measure.validate()
        return measure


def measures_from_terminal_states(states: Dict[object, TorchParticleState], *, time_label: str) -> Dict[object, FiniteMeasure]:
    return {key: state.to_measure(time_label=time_label) for key, state in states.items()}
