from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from camfnd.data.contract import EndpointProblem, FiniteMeasure, Key, PerturbSeqDynamicsData
from camfnd.numerics.particles_np import ParticleState
from camfnd.numerics.truth_coeffs import Stage1SDECoefficients, build_truth_coefficients


@dataclass(slots=True)
class EulerMaruyamaConfig:
    """Numerical configuration for the Stage-I weighted particle simulator."""

    n_steps: int = 128
    seed: int = 0
    particles_per_atom: int = 1
    store_history: bool = False

    def validate(self) -> None:
        if int(self.n_steps) <= 0:
            raise ValueError("n_steps must be positive.")
        if int(self.particles_per_atom) <= 0:
            raise ValueError("particles_per_atom must be positive.")


@dataclass(slots=True)
class SimulationResult:
    """Structured result of one Stage-I particle simulation run."""

    config: EulerMaruyamaConfig
    initial_particles: Dict[Key, ParticleState]
    terminal_particles: Dict[Key, ParticleState]
    initial_measures: Dict[Key, FiniteMeasure]
    terminal_measures: Dict[Key, FiniteMeasure]
    summary: pd.DataFrame
    history: Optional[Dict[Key, pd.DataFrame]] = None
    stability: Dict[str, bool] = field(default_factory=dict)

    @property
    def stable(self) -> bool:
        return bool(all(self.stability.values()))


class Stage1EulerMaruyamaSimulator:
    """Trusted no-context Stage-I simulator using hard-coded truth coefficients.

    Step 2 intentionally keeps the simulator deterministic in its architecture:
    it reads an `EndpointProblem`, initializes one weighted particle family per
    perturbation/sample key, propagates the particles with Euler-Maruyama, and
    converts the terminal cloud back into finite measures.
    """

    def __init__(
        self,
        *,
        endpoint_problem: EndpointProblem,
        coefficients: Dict[str, Stage1SDECoefficients],
        config: EulerMaruyamaConfig,
    ) -> None:
        endpoint_problem.validate()
        config.validate()
        self.problem = endpoint_problem
        self.coefficients = coefficients
        self.config = config
        self.problem.time_axis.validate()
        self.latent_dim = self.problem.metadata.get("latent_dim")
        if self.latent_dim is None:
            self.latent_dim = next(iter(self.problem.initial.values())).support.shape[1]
        if int(self.latent_dim) != 1:
            raise NotImplementedError("Stage 2 reference simulator currently implements the 1D Stage-I benchmark only.")
        expected = set(self.problem.catalog.perturbation_ids)
        if set(self.coefficients) != expected:
            raise ValueError(
                f"Coefficient keys {sorted(self.coefficients)} do not match endpoint catalog {sorted(expected)}."
            )

    @classmethod
    def from_dataset(
        cls,
        dataset: PerturbSeqDynamicsData,
        *,
        config: Optional[EulerMaruyamaConfig] = None,
    ) -> "Stage1EulerMaruyamaSimulator":
        dataset.validate()
        endpoint_problem = dataset.to_endpoint_problem(by_sample=True)
        coefficients = build_truth_coefficients(dataset)
        return cls(
            endpoint_problem=endpoint_problem,
            coefficients=coefficients,
            config=config or EulerMaruyamaConfig(),
        )

    @property
    def dt(self) -> float:
        t0 = self.problem.time_axis.t(self.problem.time_axis.initial_label)
        t1 = self.problem.time_axis.t(self.problem.time_axis.terminal_label)
        return float((t1 - t0) / self.config.n_steps)

    def initialize_particles(self) -> Dict[Key, ParticleState]:
        states: Dict[Key, ParticleState] = {}
        for key, measure in self.problem.initial.items():
            states[key] = ParticleState.from_measure(
                measure,
                particles_per_atom=self.config.particles_per_atom,
            )
        return states

    def step(self, state: ParticleState, coeffs: Stage1SDECoefficients, *, dt: float, rng: np.random.Generator) -> ParticleState:
        state.validate()
        z = state.z.copy()
        logw = state.logw.copy()
        noise = rng.normal(size=z.shape)
        drift = coeffs.drift(z)
        diffusion = coeffs.diffusion(z)
        z_next = z + drift * dt + diffusion * np.sqrt(dt) * noise
        logw_next = logw + coeffs.growth(z) * dt
        out = ParticleState(
            z=z_next,
            logw=logw_next,
            perturbation_id=state.perturbation_id,
            sample_id=state.sample_id,
            mass0=state.mass0,
            particles_per_atom=state.particles_per_atom,
        )
        out.validate()
        return out

    def run(self) -> SimulationResult:
        rng = np.random.default_rng(self.config.seed)
        dt = self.dt
        initial_particles = self.initialize_particles()
        particles = {key: state.copy() for key, state in initial_particles.items()}

        history: Optional[Dict[Key, list[dict]]] = None
        if self.config.store_history:
            history = {
                key: [
                    {
                        "step": 0,
                        "time": self.problem.time_axis.t(self.problem.time_axis.initial_label),
                        **state.summary(),
                    }
                ]
                for key, state in particles.items()
            }

        for step_idx in range(self.config.n_steps):
            for key in self.problem.keys:
                perturbation_id = key[1] if isinstance(key, tuple) else str(key)
                particles[key] = self.step(
                    particles[key],
                    self.coefficients[perturbation_id],
                    dt=dt,
                    rng=rng,
                )
                if history is not None:
                    history[key].append(
                        {
                            "step": step_idx + 1,
                            "time": (step_idx + 1) * dt,
                            **particles[key].summary(),
                        }
                    )

        initial_measures = {
            key: state.to_measure(time_label=self.problem.time_axis.initial_label)
            for key, state in initial_particles.items()
        }
        terminal_measures = {
            key: state.to_measure(time_label=self.problem.time_axis.terminal_label)
            for key, state in particles.items()
        }

        stability = {
            "all_terminal_particles_finite": bool(all(np.isfinite(state.z).all() and np.isfinite(state.logw).all() for state in particles.values())),
            "all_terminal_masses_positive": bool(all(measure.total_mass > 0 for measure in terminal_measures.values())),
            "all_terminal_measures_valid": True,
        }
        for measure in terminal_measures.values():
            measure.validate()

        summary_rows = []
        for key in self.problem.keys:
            init = initial_particles[key]
            term = particles[key]
            summary_rows.append(
                {
                    "key": key,
                    "sample_id": init.sample_id,
                    "perturbation_id": init.perturbation_id,
                    "n_particles": init.n_particles,
                    "initial_mass": init.total_mass(),
                    "terminal_mass": term.total_mass(),
                    "initial_mean_0": float(init.mean()[0]),
                    "terminal_mean_0": float(term.mean()[0]),
                    "initial_var_trace": init.variance_trace(),
                    "terminal_var_trace": term.variance_trace(),
                    "all_finite": bool(np.isfinite(term.z).all() and np.isfinite(term.logw).all()),
                }
            )
        summary = pd.DataFrame(summary_rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)

        history_df = {key: pd.DataFrame(rows) for key, rows in history.items()} if history is not None else None
        return SimulationResult(
            config=self.config,
            initial_particles=initial_particles,
            terminal_particles=particles,
            initial_measures=initial_measures,
            terminal_measures=terminal_measures,
            summary=summary,
            history=history_df,
            stability=stability,
        )


def simulate_stage1_dataset(
    dataset: PerturbSeqDynamicsData,
    *,
    config: Optional[EulerMaruyamaConfig] = None,
) -> SimulationResult:
    simulator = Stage1EulerMaruyamaSimulator.from_dataset(dataset, config=config)
    return simulator.run()
