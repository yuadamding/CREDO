from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from camfnd.data.contract import FiniteMeasure


Array = np.ndarray


@dataclass(slots=True)
class ParticleState:
    r"""Weighted particle representation of one perturbation/sample population.

    The empirical finite measure represented by the particle cloud is

        \hat{\mu}_t = (mass0 / N) * \sum_i exp(logw_i) \delta_{z_i}.

    Step 2 keeps the initialization simple: particles are placed exactly on the
    observed P4 support atoms, optionally repeated `particles_per_atom` times.
    """

    z: Array  # shape [N, d]
    logw: Array  # shape [N]
    perturbation_id: str
    sample_id: str
    mass0: float
    particles_per_atom: int = 1

    def validate(self) -> None:
        if self.z.ndim != 2:
            raise ValueError("z must have shape [N, d].")
        if self.logw.ndim != 1:
            raise ValueError("logw must have shape [N].")
        if self.z.shape[0] != self.logw.shape[0]:
            raise ValueError("z rows and logw length must match.")
        if self.z.shape[0] == 0:
            raise ValueError("ParticleState must contain at least one particle.")
        if self.mass0 <= 0:
            raise ValueError("mass0 must be strictly positive.")
        if self.particles_per_atom <= 0:
            raise ValueError("particles_per_atom must be positive.")
        if not np.isfinite(self.z).all():
            raise ValueError("z contains non-finite values.")
        if not np.isfinite(self.logw).all():
            raise ValueError("logw contains non-finite values.")

    @property
    def n_particles(self) -> int:
        return int(self.z.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.z.shape[1])

    @classmethod
    def from_measure(
        cls,
        measure: FiniteMeasure,
        *,
        particles_per_atom: int = 1,
    ) -> "ParticleState":
        measure.validate()
        if particles_per_atom <= 0:
            raise ValueError("particles_per_atom must be positive.")
        z = np.repeat(measure.support.astype(float), repeats=particles_per_atom, axis=0)
        logw = np.zeros(z.shape[0], dtype=float)
        out = cls(
            z=z,
            logw=logw,
            perturbation_id=str(measure.perturbation_id),
            sample_id=str(measure.sample_id),
            mass0=float(measure.total_mass),
            particles_per_atom=int(particles_per_atom),
        )
        out.validate()
        return out

    def copy(self) -> "ParticleState":
        return ParticleState(
            z=self.z.copy(),
            logw=self.logw.copy(),
            perturbation_id=self.perturbation_id,
            sample_id=self.sample_id,
            mass0=self.mass0,
            particles_per_atom=self.particles_per_atom,
        )

    def particle_weights(self) -> Array:
        self.validate()
        weights = (self.mass0 / self.n_particles) * np.exp(self.logw)
        if not np.isfinite(weights).all():
            raise ValueError("Particle weights overflowed or became non-finite.")
        return weights

    def total_mass(self) -> float:
        return float(self.particle_weights().sum())

    def normalized_weights(self) -> Array:
        weights = self.particle_weights()
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("Particle weights sum to a non-positive number.")
        return weights / total

    def mean(self) -> Array:
        p = self.normalized_weights()[:, None]
        return np.sum(p * self.z, axis=0)

    def covariance(self) -> Array:
        mu = self.mean()
        centered = self.z - mu
        return centered.T @ (centered * self.normalized_weights()[:, None])

    def variance_trace(self) -> float:
        return float(np.trace(self.covariance()))

    def summary(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "perturbation_id": self.perturbation_id,
            "n_particles": self.n_particles,
            "total_mass": self.total_mass(),
            "mean_0": float(self.mean()[0]),
            "var_trace": self.variance_trace(),
            "all_finite": bool(np.isfinite(self.z).all() and np.isfinite(self.logw).all()),
        }

    def to_measure(self, *, time_label: str) -> FiniteMeasure:
        weights = self.particle_weights()
        measure = FiniteMeasure(
            support=self.z.copy(),
            weights=weights,
            total_mass=float(weights.sum()),
            perturbation_id=self.perturbation_id,
            time_label=str(time_label),
            sample_id=self.sample_id,
        )
        measure.validate()
        return measure


def compare_measures_exact(lhs: FiniteMeasure, rhs: FiniteMeasure) -> Tuple[bool, str]:
    """Exact comparison used for the Step-2 initialization check.

    This helper intentionally requires exact support equality, so it should only
    be used when `particles_per_atom == 1`.
    """

    lhs.validate()
    rhs.validate()
    if lhs.support.shape != rhs.support.shape:
        return False, "support_shape_mismatch"
    if not np.array_equal(lhs.support, rhs.support):
        return False, "support_values_mismatch"
    if not np.array_equal(lhs.weights, rhs.weights):
        return False, "weight_values_mismatch"
    if lhs.total_mass != rhs.total_mass:
        return False, "mass_mismatch"
    if lhs.perturbation_id != rhs.perturbation_id:
        return False, "perturbation_id_mismatch"
    if lhs.sample_id != rhs.sample_id:
        return False, "sample_id_mismatch"
    return True, "ok"
