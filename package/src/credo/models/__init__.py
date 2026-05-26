"""Model components for CREDO dynamics."""
from __future__ import annotations

from .coefficients import CoefficientNetworks, Coefficients
from .context import ContextAggregator, ContextState, GroupStatistics, ProgramEncoder
from .ecology import EcologicalPayoff
from .embeddings import PerturbationEmbedding, TimeEmbedding
from .expression_vae import (
    ExpressionVAE,
    ExpressionVAETrainingSummary,
    LatentStandardization,
    VAEArtifactBundle,
    encode_expression_vae,
    fit_expression_vae,
    log1p_normalize_expression_matrix,
    maybe_materialize_dense_matrix,
    standardize_latent,
)
from .full_model import FullDynamicsModel
from .particles import (
    CounterfactualEngine,
    CounterfactualResult,
    ParticleRollout,
    WeightedParticleSimulator,
    initialise_particles,
    initialise_particles_from_measures,
    initialise_particles_from_trajectory,
    rollout_with_clamped_context,
)

__all__ = [
    "CoefficientNetworks",
    "Coefficients",
    "ContextAggregator",
    "ContextState",
    "CounterfactualEngine",
    "CounterfactualResult",
    "EcologicalPayoff",
    "ExpressionVAE",
    "ExpressionVAETrainingSummary",
    "FullDynamicsModel",
    "GroupStatistics",
    "LatentStandardization",
    "ParticleRollout",
    "PerturbationEmbedding",
    "ProgramEncoder",
    "TimeEmbedding",
    "VAEArtifactBundle",
    "WeightedParticleSimulator",
    "encode_expression_vae",
    "fit_expression_vae",
    "initialise_particles",
    "initialise_particles_from_measures",
    "initialise_particles_from_trajectory",
    "log1p_normalize_expression_matrix",
    "maybe_materialize_dense_matrix",
    "rollout_with_clamped_context",
    "standardize_latent",
]
