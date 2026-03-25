"""Numerical simulation utilities: numpy particles, truth coefficients, Euler-Maruyama."""

from camfnd.numerics.particles_np import ParticleState, compare_measures_exact
from camfnd.numerics.particles_torch import TorchParticleState, measures_from_terminal_states
from camfnd.numerics.truth_coeffs import Stage1SDECoefficients, build_truth_coefficients
from camfnd.numerics.euler_maruyama import (
    EulerMaruyamaConfig,
    SimulationResult,
    Stage1EulerMaruyamaSimulator,
    simulate_stage1_dataset,
)

__all__ = [
    "ParticleState",
    "compare_measures_exact",
    "TorchParticleState",
    "measures_from_terminal_states",
    "Stage1SDECoefficients",
    "build_truth_coefficients",
    "EulerMaruyamaConfig",
    "SimulationResult",
    "Stage1EulerMaruyamaSimulator",
    "simulate_stage1_dataset",
]
