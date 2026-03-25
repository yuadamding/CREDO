from __future__ import annotations

"""Multi-screen synthetic benchmark generation utilities."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from camfnd.models.context_map import ContextMapConfig
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
class Stage2TruthParams:
    theta: float
    sigma: float
    rho: float


@dataclass(slots=True)
class Stage2BenchmarkConfig:
    seed: int = 23
    screen_ids: tuple[str, str] = ('screen1', 'screen2')
    n_obs_p4: int = 256
    n_obs_p60: int = 256
    n_truth_particles: int = 4096
    n_steps: int = 64
    T: float = 1.0
    m0: float = -1.0
    sd0: float = 0.15
    kappa: float = 2.0
    eta: float = 0.8
    driver_initial_masses: tuple[float, float] = (0.5, 2.0)
    infer_latent_transform: bool = True
    context_map: ContextMapConfig = ContextMapConfig(sharpness=6.0, bias=0.0, learnable=False)

    def validate(self) -> None:
        if len(self.screen_ids) != 2:
            raise ValueError('screen_ids must contain exactly two screens.')
        if len(set(self.screen_ids)) != 2:
            raise ValueError('screen_ids must be unique.')
        if len(self.driver_initial_masses) != 2:
            raise ValueError('driver_initial_masses must have length 2.')
        for name in ('n_obs_p4', 'n_obs_p60', 'n_truth_particles', 'n_steps'):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive.')
        if self.T <= 0 or self.sd0 <= 0 or self.kappa < 0 or self.eta < 0:
            raise ValueError('Invalid Stage2BenchmarkConfig numeric values.')
        self.context_map.validate()


def build_stage2_truth_params() -> Dict[str, Stage2TruthParams]:
    return {
        'ctrl': Stage2TruthParams(theta=0.0, sigma=0.15, rho=0.0),
        'drift': Stage2TruthParams(theta=0.6, sigma=0.15, rho=0.0),
        'diff': Stage2TruthParams(theta=0.0, sigma=0.35, rho=0.0),
        'react': Stage2TruthParams(theta=0.0, sigma=0.15, rho=-0.7),
        'driver': Stage2TruthParams(theta=1.0, sigma=0.15, rho=float(np.log(2.0))),
    }


def _catalog_frame(perturbations: Iterable[str]) -> pd.DataFrame:
    rows = []
    for perturbation_id in perturbations:
        rows.append({'perturbation_id': str(perturbation_id), 'is_control': bool(perturbation_id == 'ctrl')})
    return pd.DataFrame(rows)


def _occupancy(z: np.ndarray, sharpness: float, bias: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-sharpness * (z - bias)))


def _simulate_screen(
    *,
    rng: np.random.Generator,
    screen_id: str,
    truth_params: Dict[str, Stage2TruthParams],
    initial_masses: Dict[str, float],
    config: Stage2BenchmarkConfig,
) -> tuple[dict, pd.DataFrame]:
    dt = config.T / config.n_steps
    sqrt_dt = float(np.sqrt(dt))
    sharpness = config.context_map.sharpness
    bias = config.context_map.bias
    particles: dict[str, dict] = {}
    for perturbation_id, params in truth_params.items():
        z0 = rng.normal(loc=config.m0, scale=config.sd0, size=(config.n_truth_particles, 1))
        particles[perturbation_id] = {
            'z0': z0.copy(),
            'z': z0.copy(),
            'logw': np.zeros(config.n_truth_particles, dtype=float),
            'mass0': float(initial_masses[perturbation_id]),
            'theta': float(params.theta),
            'sigma': float(params.sigma),
            'rho': float(params.rho),
        }

    context_rows = []
    for step_idx in range(config.n_steps + 1):
        numer = 0.0
        denom = 0.0
        for state in particles.values():
            atom_weights = (state['mass0'] / config.n_truth_particles) * np.exp(state['logw'])
            numer += float((atom_weights[:, None] * _occupancy(state['z'], sharpness, bias)).sum())
            denom += float(atom_weights.sum())
        context = numer / max(denom, 1e-12)
        context_rows.append({'sample_id': screen_id, 'step': step_idx, 'time': step_idx * dt, 'context': float(context), 'total_mass': float(denom)})
        if step_idx == config.n_steps:
            break
        for perturbation_id, state in particles.items():
            noise = rng.normal(loc=0.0, scale=1.0, size=(config.n_truth_particles, 1))
            drift = config.kappa * (state['theta'] - state['z']) + config.eta * context
            state['z'] = state['z'] + drift * dt + state['sigma'] * sqrt_dt * noise
            state['logw'] = state['logw'] + state['rho'] * dt

    terminal = {}
    for perturbation_id, state in particles.items():
        atom_weights = (state['mass0'] / config.n_truth_particles) * np.exp(state['logw'])
        probs = atom_weights / atom_weights.sum()
        terminal[perturbation_id] = {
            'z0': state['z0'],
            'z1': state['z'],
            'weights1': atom_weights,
            'probs1': probs,
            'mass0': state['mass0'],
            'mass1': float(atom_weights.sum()),
        }
    return terminal, pd.DataFrame(context_rows)


def _truth_summary(screen_id: str, perturbation_id: str, terminal: dict) -> dict:
    z1 = terminal['z1'].reshape(-1)
    probs = terminal['probs1'].reshape(-1)
    mean1 = float((probs * z1).sum())
    var1 = float((probs * (z1 - mean1) ** 2).sum())
    return {
        'sample_id': screen_id,
        'perturbation_id': perturbation_id,
        'initial_mean': float(terminal['z0'].mean()),
        'initial_variance': float(terminal['z0'].var()),
        'terminal_mean': mean1,
        'terminal_variance': var1,
        'initial_mass': float(terminal['mass0']),
        'terminal_mass': float(terminal['mass1']),
    }


def generate_stage2_dataset(
    config: Stage2BenchmarkConfig | None = None,
    truth_params: Dict[str, Stage2TruthParams] | None = None,
) -> PerturbSeqDynamicsData:
    config = config or Stage2BenchmarkConfig()
    config.validate()
    truth_params = truth_params or build_stage2_truth_params()
    expected = {'ctrl', 'drift', 'diff', 'react', 'driver'}
    if set(truth_params) != expected:
        raise ValueError(f'Stage-II truth parameters must contain exactly {expected}.')

    rng = np.random.default_rng(config.seed)
    time_axis = TimeAxis()
    catalog = PerturbationCatalog(_catalog_frame(truth_params.keys()))
    catalog.validate()

    obs_rows: list[dict] = []
    Z_rows: list[np.ndarray] = []
    mass_rows: list[dict] = []
    truth_rows: list[dict] = []
    endpoint_rows: list[dict] = []
    context_frames: list[pd.DataFrame] = []

    for screen_id, driver_mass in zip(config.screen_ids, config.driver_initial_masses, strict=True):
        initial_masses = {pid: 1.0 for pid in truth_params}
        initial_masses['driver'] = float(driver_mass)
        terminal_by_pert, context_df = _simulate_screen(
            rng=rng,
            screen_id=screen_id,
            truth_params=truth_params,
            initial_masses=initial_masses,
            config=config,
        )
        context_frames.append(context_df)

        for perturbation_id, params in truth_params.items():
            terminal = terminal_by_pert[perturbation_id]
            p4 = rng.normal(loc=config.m0, scale=config.sd0, size=config.n_obs_p4)
            idx = rng.choice(config.n_truth_particles, size=config.n_obs_p60, replace=True, p=terminal['probs1'])
            p60 = terminal['z1'][idx, 0]

            for i, value in enumerate(p4):
                obs_rows.append({
                    'cell_id': f'{screen_id}_P4_{perturbation_id}_{i}',
                    'perturbation_id': perturbation_id,
                    'time_label': time_axis.initial_label,
                    'sample_id': screen_id,
                })
                Z_rows.append(np.array([float(value)], dtype=float))
            for i, value in enumerate(p60):
                obs_rows.append({
                    'cell_id': f'{screen_id}_P60_{perturbation_id}_{i}',
                    'perturbation_id': perturbation_id,
                    'time_label': time_axis.terminal_label,
                    'sample_id': screen_id,
                })
                Z_rows.append(np.array([float(value)], dtype=float))

            mass_rows.extend([
                {'perturbation_id': perturbation_id, 'time_label': time_axis.initial_label, 'sample_id': screen_id, 'mass': float(initial_masses[perturbation_id])},
                {'perturbation_id': perturbation_id, 'time_label': time_axis.terminal_label, 'sample_id': screen_id, 'mass': float(terminal['mass1'])},
            ])

            truth_rows.append({
                'sample_id': screen_id,
                'perturbation_id': perturbation_id,
                'theta': float(params.theta),
                'sigma': float(params.sigma),
                'rho': float(params.rho),
                'kappa': float(config.kappa),
                'eta': float(config.eta),
                'driver_initial_mass': float(driver_mass),
            })
            endpoint_rows.append(_truth_summary(screen_id, perturbation_id, terminal))

    obs = pd.DataFrame(obs_rows)
    Z = np.vstack(Z_rows)
    masses = pd.DataFrame(mass_rows)
    truth_params_df = pd.DataFrame(truth_rows).sort_values(['sample_id', 'perturbation_id']).reset_index(drop=True)
    analytic_summary = pd.DataFrame(endpoint_rows).sort_values(['sample_id', 'perturbation_id']).reset_index(drop=True)
    context_trajectories = pd.concat(context_frames, ignore_index=True)

    cells = CellStateTable(obs=obs, Z=Z)
    mass_table = MassTable(table=masses)
    truth = SimulationTruth(
        truth_params=truth_params_df,
        analytic_summary=analytic_summary,
        context_trajectories=context_trajectories,
        simulator_config={
            'seed': config.seed,
            'screen_ids': list(config.screen_ids),
            'n_obs_p4': config.n_obs_p4,
            'n_obs_p60': config.n_obs_p60,
            'n_truth_particles': config.n_truth_particles,
            'n_steps': config.n_steps,
            'T': config.T,
            'm0': config.m0,
            'sd0': config.sd0,
            'kappa': config.kappa,
            'eta': config.eta,
            'driver_initial_masses': list(config.driver_initial_masses),
            'context_map': asdict(config.context_map),
        },
    )
    latent_transform = LatentTransform.from_array(Z) if config.infer_latent_transform else None
    dataset = PerturbSeqDynamicsData(
        time_axis=time_axis,
        catalog=catalog,
        cells=cells,
        masses=mass_table,
        latent_transform=latent_transform,
        truth=truth,
        metadata={'stage': 'stage2', 'generator': 'joint_meanfield_particle_sampler'},
    )
    dataset.validate()
    return dataset


# Semantic aliases for the clearer software-facing API.
MultiscreenTruthParams = Stage2TruthParams
MultiscreenBenchmarkConfig = Stage2BenchmarkConfig
build_multiscreen_truth_params = build_stage2_truth_params
generate_multiscreen_dataset = generate_stage2_dataset
