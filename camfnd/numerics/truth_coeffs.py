from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from camfnd.data.contract import PerturbSeqDynamicsData


@dataclass(frozen=True, slots=True)
class Stage1SDECoefficients:
    """Hard-coded Stage-I truth coefficients for the core benchmark."""

    theta: float
    sigma: float
    rho: float
    kappa: float

    def drift(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 2 or z.shape[1] != 1:
            raise ValueError("Stage1SDECoefficients currently expects z with shape [N, 1].")
        return self.kappa * (self.theta - z)

    def diffusion(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 2 or z.shape[1] != 1:
            raise ValueError("Stage1SDECoefficients currently expects z with shape [N, 1].")
        return np.full_like(z, fill_value=self.sigma, dtype=float)

    def growth(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 2:
            raise ValueError("z must have shape [N, d].")
        return np.full(z.shape[0], fill_value=self.rho, dtype=float)


def build_truth_coefficients(dataset: PerturbSeqDynamicsData) -> Dict[str, Stage1SDECoefficients]:
    """Parse the Stage-I truth coefficients from the synthetic dataset metadata."""

    dataset.validate()
    if dataset.truth is None or dataset.truth.truth_params is None:
        raise ValueError("Dataset does not contain synthetic truth_params required for Step 2.")

    truth_params = dataset.truth.truth_params.copy()
    if "kappa" not in truth_params.columns:
        kappa = dataset.truth.simulator_config.get("kappa")
        if kappa is None:
            raise ValueError("Could not determine kappa from truth metadata.")
        truth_params["kappa"] = float(kappa)

    required = {"perturbation_id", "theta", "sigma", "rho", "kappa"}
    missing = sorted(required - set(truth_params.columns))
    if missing:
        raise ValueError(f"truth_params missing required columns for Step 2: {missing}")

    coeffs: Dict[str, Stage1SDECoefficients] = {}
    for row in truth_params.to_dict(orient="records"):
        coeffs[str(row["perturbation_id"])] = Stage1SDECoefficients(
            theta=float(row["theta"]),
            sigma=float(row["sigma"]),
            rho=float(row["rho"]),
            kappa=float(row["kappa"]),
        )

    expected = set(dataset.catalog.perturbation_ids)
    if set(coeffs) != expected:
        raise ValueError(
            f"Truth coefficients keys {sorted(coeffs)} do not match catalog {sorted(expected)}."
        )
    return coeffs
