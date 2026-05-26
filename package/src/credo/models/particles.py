"""Particle rollout and counterfactual utilities.

The implementation is split between ``weighted_sde`` and ``simulator`` for
historical reasons.  This facade provides the public particle API in one place.
"""
from __future__ import annotations

from .simulator import (
    CounterfactualEngine,
    CounterfactualResult,
    initialise_particles,
    initialise_particles_from_measures,
    initialise_particles_from_trajectory,
    rollout_with_clamped_context,
)
from .weighted_sde import ParticleRollout, WeightedParticleSimulator

__all__ = [
    "CounterfactualEngine",
    "CounterfactualResult",
    "ParticleRollout",
    "WeightedParticleSimulator",
    "initialise_particles",
    "initialise_particles_from_measures",
    "initialise_particles_from_trajectory",
    "rollout_with_clamped_context",
]
