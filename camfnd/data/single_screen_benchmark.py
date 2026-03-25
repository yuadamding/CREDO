from __future__ import annotations

"""Single-screen synthetic benchmark generation utilities."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from camfnd.data.contract import (
    CellStateTable,
    LatentTransform,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    SimulationTruth,
    TimeAxis,
)


@dataclass(frozen=True, slots=True)
class Stage1TruthParams:
    """Ground-truth parameters for the minimum useful Stage-I benchmark."""

    theta: float
    sigma: float
    rho: float


@dataclass(slots=True)
class Stage1BenchmarkConfig:
    """Configuration for the Stage-I synthetic finite-measure endpoint benchmark."""

    seed: int = 7
    sample_id: str = "screen1"
    n_obs_p4: int = 256
    n_obs_p60: int = 256
    T: float = 1.0
    m0: float = -1.0
    sd0: float = 0.15
    kappa: float = 2.0
    infer_latent_transform: bool = True

    def validate(self) -> None:
        for name in ("n_obs_p4", "n_obs_p60"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}.")
        if self.sd0 <= 0:
            raise ValueError(f"sd0 must be positive, got {self.sd0}.")
        if self.kappa < 0:
            raise ValueError(f"kappa must be nonnegative, got {self.kappa}.")
        if not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")


def build_stage1_truth_params() -> Dict[str, Stage1TruthParams]:
    """Return the canonical control / drift / diffusion / reaction Stage-I truth."""

    return {
        "ctrl": Stage1TruthParams(theta=0.0, sigma=0.15, rho=0.0),
        "drift": Stage1TruthParams(theta=0.6, sigma=0.15, rho=0.0),
        "diff": Stage1TruthParams(theta=0.0, sigma=0.35, rho=0.0),
        "react": Stage1TruthParams(theta=0.0, sigma=0.15, rho=-0.7),
    }


def ou_terminal_moments(
    *,
    m0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float = 1.0,
) -> tuple[float, float]:
    """Closed-form moments for the 1D OU latent dynamics in Stage I."""

    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if kappa < 0:
        raise ValueError("kappa must be nonnegative.")

    if kappa == 0.0:
        mean_T = m0
        var_T = v0 + sigma**2 * T
        return float(mean_T), float(var_T)

    exp_term = np.exp(-kappa * T)
    mean_T = theta + (m0 - theta) * exp_term
    var_T = v0 * np.exp(-2.0 * kappa * T) + (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * T))
    return float(mean_T), float(var_T)


def _catalog_frame(perturbations: Iterable[str]) -> pd.DataFrame:
    rows = []
    for perturbation_id in perturbations:
        rows.append(
            {
                "perturbation_id": str(perturbation_id),
                "is_control": bool(perturbation_id == "ctrl"),
            }
        )
    return pd.DataFrame(rows)


def _analytic_summary_table(
    *,
    truth_params: Dict[str, Stage1TruthParams],
    config: Stage1BenchmarkConfig,
) -> pd.DataFrame:
    rows = []
    v0 = config.sd0**2
    for perturbation_id, params in truth_params.items():
        mean1, var1 = ou_terminal_moments(
            m0=config.m0,
            v0=v0,
            kappa=config.kappa,
            theta=params.theta,
            sigma=params.sigma,
            T=config.T,
        )
        rows.append(
            {
                "sample_id": config.sample_id,
                "perturbation_id": perturbation_id,
                "initial_mean": float(config.m0),
                "initial_variance": float(v0),
                "terminal_mean": float(mean1),
                "terminal_variance": float(var1),
                "initial_mass": 1.0,
                "terminal_mass": float(np.exp(params.rho * config.T)),
                "theta": float(params.theta),
                "sigma": float(params.sigma),
                "rho": float(params.rho),
                "kappa": float(config.kappa),
                "n_obs_p4": int(config.n_obs_p4),
                "n_obs_p60": int(config.n_obs_p60),
            }
        )
    return pd.DataFrame(rows).sort_values("perturbation_id").reset_index(drop=True)


def generate_stage1_dataset(
    config: Stage1BenchmarkConfig | None = None,
    truth_params: Dict[str, Stage1TruthParams] | None = None,
) -> PerturbSeqDynamicsData:
    """Generate the minimum useful Stage-I finite-measure endpoint dataset.

    The returned object contains only P4 and P60 snapshots plus separate guide-abundance masses.
    It is therefore aligned with the endpoint-only supervision setting.
    """

    config = config or Stage1BenchmarkConfig()
    config.validate()
    truth_params = truth_params or build_stage1_truth_params()
    if set(truth_params) != {"ctrl", "drift", "diff", "react"}:
        raise ValueError(
            "Stage-I truth parameters must contain exactly {'ctrl', 'drift', 'diff', 'react'}."
        )

    rng = np.random.default_rng(config.seed)
    time_axis = TimeAxis()
    catalog = PerturbationCatalog(_catalog_frame(truth_params.keys()))
    catalog.validate()

    obs_rows: list[dict] = []
    Z_rows: list[np.ndarray] = []
    mass_rows: list[dict] = []
    truth_rows: list[dict] = []
    v0 = config.sd0**2

    for perturbation_id, params in truth_params.items():
        p4 = rng.normal(loc=config.m0, scale=config.sd0, size=config.n_obs_p4)
        mean1, var1 = ou_terminal_moments(
            m0=config.m0,
            v0=v0,
            kappa=config.kappa,
            theta=params.theta,
            sigma=params.sigma,
            T=config.T,
        )
        p60 = rng.normal(loc=mean1, scale=np.sqrt(var1), size=config.n_obs_p60)
        M0 = 1.0
        M1 = float(np.exp(params.rho * config.T) * M0)

        for index, value in enumerate(p4):
            obs_rows.append(
                {
                    "cell_id": f"{config.sample_id}_P4_{perturbation_id}_{index}",
                    "perturbation_id": perturbation_id,
                    "time_label": time_axis.initial_label,
                    "sample_id": config.sample_id,
                }
            )
            Z_rows.append(np.array([float(value)], dtype=float))

        for index, value in enumerate(p60):
            obs_rows.append(
                {
                    "cell_id": f"{config.sample_id}_P60_{perturbation_id}_{index}",
                    "perturbation_id": perturbation_id,
                    "time_label": time_axis.terminal_label,
                    "sample_id": config.sample_id,
                }
            )
            Z_rows.append(np.array([float(value)], dtype=float))

        mass_rows.extend(
            [
                {
                    "perturbation_id": perturbation_id,
                    "time_label": time_axis.initial_label,
                    "sample_id": config.sample_id,
                    "mass": M0,
                },
                {
                    "perturbation_id": perturbation_id,
                    "time_label": time_axis.terminal_label,
                    "sample_id": config.sample_id,
                    "mass": M1,
                },
            ]
        )

        truth_rows.append(
            {
                "sample_id": config.sample_id,
                "perturbation_id": perturbation_id,
                **asdict(params),
                "kappa": float(config.kappa),
                "m0": float(config.m0),
                "sd0": float(config.sd0),
                "T": float(config.T),
            }
        )

    obs = pd.DataFrame(obs_rows)
    Z = np.vstack(Z_rows)
    masses = pd.DataFrame(mass_rows)
    truth_params_df = pd.DataFrame(truth_rows).sort_values("perturbation_id").reset_index(drop=True)
    analytic_summary = _analytic_summary_table(truth_params=truth_params, config=config)

    cells = CellStateTable(obs=obs, Z=Z)
    mass_table = MassTable(table=masses)
    truth = SimulationTruth(
        truth_params=truth_params_df,
        analytic_summary=analytic_summary,
        simulator_config=asdict(config),
    )

    latent_transform = LatentTransform.from_array(Z) if config.infer_latent_transform else None

    dataset = PerturbSeqDynamicsData(
        time_axis=time_axis,
        catalog=catalog,
        cells=cells,
        masses=mass_table,
        latent_transform=latent_transform,
        truth=truth,
        metadata={
            "stage": "stage1",
            "generator": "closed_form_ou_endpoint_sampler",
        },
    )
    dataset.validate()
    return dataset


# Semantic aliases for the clearer software-facing API.
SingleScreenTruthParams = Stage1TruthParams
SingleScreenBenchmarkConfig = Stage1BenchmarkConfig
build_single_screen_truth_params = build_stage1_truth_params
generate_single_screen_dataset = generate_stage1_dataset
