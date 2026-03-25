from __future__ import annotations

"""Multi-screen context-aware learned particle simulator."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from camfnd.models.coeff_nets import ControlAnchoredStage2Model
from camfnd.data.contract import EndpointProblem, FiniteMeasure, Key, PerturbSeqDynamicsData
from camfnd.numerics.particles_torch import TorchParticleState, measures_from_terminal_states
from camfnd.simulation.single_screen_sim import LearnedSimulatorConfig


@dataclass(slots=True)
class LearnedStage2SimulationResult:
    config: LearnedSimulatorConfig
    initial_particles: Dict[Key, TorchParticleState]
    terminal_particles: Dict[Key, TorchParticleState]
    terminal_measures: Dict[Key, FiniteMeasure]
    summary: pd.DataFrame
    context_summary: pd.DataFrame
    stability: Dict[str, bool] = field(default_factory=dict)
    history: Optional[Dict[Key, pd.DataFrame]] = None

    @property
    def stable(self) -> bool:
        return bool(all(self.stability.values()))


class LearnedStage2JointSimulator:
    def __init__(self, endpoint_problem: EndpointProblem, config: LearnedSimulatorConfig) -> None:
        endpoint_problem.validate()
        config.validate()
        self.problem = endpoint_problem
        self.config = config
        self.device = torch.device(config.resolved_device)
        self.dtype = config.torch_dtype
        latent_dim = self.problem.metadata.get('latent_dim')
        if latent_dim is None:
            latent_dim = next(iter(self.problem.initial.values())).support.shape[1]
        if int(latent_dim) != 1:
            raise NotImplementedError('Step 4 reference implementation currently supports the 1D Stage-II benchmark only.')

    @classmethod
    def from_dataset(cls, dataset: PerturbSeqDynamicsData, *, config: LearnedSimulatorConfig) -> 'LearnedStage2JointSimulator':
        dataset.validate()
        return cls(dataset.to_endpoint_problem(by_sample=True), config=config)

    @property
    def dt(self) -> float:
        t0 = self.problem.time_axis.t(self.problem.time_axis.initial_label)
        t1 = self.problem.time_axis.t(self.problem.time_axis.terminal_label)
        return float((t1 - t0) / self.config.n_steps)

    def initialize_particles(self) -> Dict[Key, TorchParticleState]:
        states: Dict[Key, TorchParticleState] = {}
        for key, measure in self.problem.initial.items():
            states[key] = TorchParticleState.from_measure(
                measure,
                particles_per_atom=self.config.particles_per_atom,
                dtype=self.dtype,
                device=self.device,
            )
        return states

    def _noise_bank(self, particles: Dict[Key, TorchParticleState]) -> Dict[Key, Tensor]:
        noise: Dict[Key, Tensor] = {}
        for offset, key in enumerate(self.problem.keys):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.config.seed + 997 * offset))
            n_particles = particles[key].n_particles
            noise[key] = torch.randn(
                self.config.n_steps,
                n_particles,
                1,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
        return noise

    def run(self, model: ControlAnchoredStage2Model) -> LearnedStage2SimulationResult:
        model.eval()
        dt = self.dt
        sqrt_dt = float(np.sqrt(dt))
        initial_particles = self.initialize_particles()
        particles = {
            key: TorchParticleState(
                z=state.z.clone(),
                logw=state.logw.clone(),
                perturbation_id=state.perturbation_id,
                sample_id=state.sample_id,
                mass0=state.mass0,
                particles_per_atom=state.particles_per_atom,
            )
            for key, state in initial_particles.items()
        }
        noise_bank = self._noise_bank(initial_particles)

        history = {key: [] for key in self.problem.keys} if self.config.store_history else None
        context_rows = []

        for step_idx in range(self.config.n_steps):
            t_scalar = torch.tensor(step_idx * dt, dtype=self.dtype, device=self.device)
            contexts = model.context_values(particles)
            for sample_id, cval in contexts.items():
                total_mass = 0.0
                for key, state in particles.items():
                    if state.sample_id == sample_id:
                        total_mass += float(state.total_mass().detach().cpu())
                context_rows.append({
                    'sample_id': sample_id,
                    'step': step_idx,
                    'time': float(step_idx * dt),
                    'context': float(cval.detach().cpu()),
                    'total_mass': total_mass,
                })
            for key in self.problem.keys:
                state = particles[key]
                coeffs = model.coefficients(state.z, t_scalar, state.perturbation_id, contexts[state.sample_id])
                z_next = state.z + coeffs['drift'] * dt + coeffs['diffusion'] * sqrt_dt * noise_bank[key][step_idx]
                logw_next = state.logw + coeffs['growth'].reshape(-1) * dt
                particles[key] = TorchParticleState(
                    z=z_next,
                    logw=logw_next,
                    perturbation_id=state.perturbation_id,
                    sample_id=state.sample_id,
                    mass0=state.mass0,
                    particles_per_atom=state.particles_per_atom,
                )
                if history is not None:
                    history[key].append({'step': step_idx + 1, **particles[key].summary()})

        contexts = model.context_values(particles)
        for sample_id, cval in contexts.items():
            total_mass = 0.0
            for key, state in particles.items():
                if state.sample_id == sample_id:
                    total_mass += float(state.total_mass().detach().cpu())
            context_rows.append({
                'sample_id': sample_id,
                'step': self.config.n_steps,
                'time': float(self.config.n_steps * dt),
                'context': float(cval.detach().cpu()),
                'total_mass': total_mass,
            })

        terminal_measures = measures_from_terminal_states(particles, time_label=self.problem.time_axis.terminal_label)
        summary_rows = []
        for key in self.problem.keys:
            init = initial_particles[key]
            term = particles[key]
            summary_rows.append({
                'key': key,
                'sample_id': term.sample_id,
                'perturbation_id': term.perturbation_id,
                'n_particles': term.n_particles,
                'initial_mass': float(init.total_mass().detach().cpu()),
                'terminal_mass': float(term.total_mass().detach().cpu()),
                'initial_mean_0': float(init.mean()[0].detach().cpu()),
                'terminal_mean_0': float(term.mean()[0].detach().cpu()),
                'initial_var_trace': float(init.variance_trace().detach().cpu()),
                'terminal_var_trace': float(term.variance_trace().detach().cpu()),
                'all_finite': bool(torch.isfinite(term.z).all() and torch.isfinite(term.logw).all()),
            })
        summary = pd.DataFrame(summary_rows).sort_values(['sample_id', 'perturbation_id']).reset_index(drop=True)
        context_summary = pd.DataFrame(context_rows).sort_values(['sample_id', 'step']).reset_index(drop=True)

        stability = {
            'all_terminal_particles_finite': bool(all(torch.isfinite(state.z).all() and torch.isfinite(state.logw).all() for state in particles.values())),
            'all_terminal_masses_positive': bool(all(measure.total_mass > 0 for measure in terminal_measures.values())),
            'all_terminal_measures_valid': True,
        }
        for measure in terminal_measures.values():
            measure.validate()

        history_frames = {key: pd.DataFrame(rows) for key, rows in history.items()} if history is not None else None
        return LearnedStage2SimulationResult(
            config=self.config,
            initial_particles=initial_particles,
            terminal_particles=particles,
            terminal_measures=terminal_measures,
            summary=summary,
            context_summary=context_summary,
            stability=stability,
            history=history_frames,
        )


# Semantic aliases for the clearer software-facing API.
MultiscreenContextSimulationResult = LearnedStage2SimulationResult
LearnedMultiscreenContextSimulator = LearnedStage2JointSimulator
