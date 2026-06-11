"""Model components for CREDO dynamics."""
from __future__ import annotations

from .coefficients import CoefficientNetworks, Coefficients
from .context import ContextAggregator, ContextDiagnostics, ContextState, GroupStatistics, ProgramEncoder
from .causal_attention_blocks import MassGraphMaskedCrossAttention
from .causal_context import (
    CausalAttentionDiagnostics,
    CausalContextState,
    CausalEcologicalAttentionContext,
)
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
from .interventions import CausalAttentionIntervention
from .single_time_counterfactual import SingleTimeCounterfactualEngine
from .transformer_blocks import FeedForwardBlock, InducedSetAttentionBlock, MassBiasedCrossAttention
from .transformer_context import MassAwareTransformerContextAggregator
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


def __getattr__(name: str):
    if name in {"TrajectoryCounterfactualEngine", "TrajectoryCounterfactualResult"}:
        from .trajectory_counterfactual import TrajectoryCounterfactualEngine, TrajectoryCounterfactualResult

        return {
            "TrajectoryCounterfactualEngine": TrajectoryCounterfactualEngine,
            "TrajectoryCounterfactualResult": TrajectoryCounterfactualResult,
        }[name]
    raise AttributeError(name)

__all__ = [
    "CoefficientNetworks",
    "Coefficients",
    "CausalAttentionDiagnostics",
    "CausalAttentionIntervention",
    "CausalContextState",
    "CausalEcologicalAttentionContext",
    "ContextAggregator",
    "ContextDiagnostics",
    "ContextState",
    "CounterfactualEngine",
    "CounterfactualResult",
    "EcologicalPayoff",
    "ExpressionVAE",
    "ExpressionVAETrainingSummary",
    "FeedForwardBlock",
    "FullDynamicsModel",
    "GroupStatistics",
    "InducedSetAttentionBlock",
    "LatentStandardization",
    "MassAwareTransformerContextAggregator",
    "MassBiasedCrossAttention",
    "MassGraphMaskedCrossAttention",
    "ParticleRollout",
    "PerturbationEmbedding",
    "ProgramEncoder",
    "SingleTimeCounterfactualEngine",
    "TimeEmbedding",
    "TrajectoryCounterfactualEngine",
    "TrajectoryCounterfactualResult",
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
