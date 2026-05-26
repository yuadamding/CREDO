"""Loss functions and rollout regularizers."""
from __future__ import annotations

from .counts import (
    CountLikelihood,
    DirichletMultinomialLikelihood,
    MultiTimeCountLikelihood,
    count_fractions_from_zeta,
    integrated_fitness,
    integrated_fitness_curve,
)
from .multitime import (
    MultiTimeEndpointLoss,
    build_target_tensors_by_time,
    checkpoint_indices_for_taus,
    checkpoint_indices_for_trajectory,
    make_observed_tau_grid,
)
from .regularizers import (
    RolloutRegularizer,
    diffusion_magnitude_penalty,
    drift_action_penalty,
    growth_action_penalty,
)
from .uot import UOTLoss, sinkhorn_divergence, sinkhorn_divergence_normalized
from .weak_form import GaussianRBFTestFunctions, WeakFormLoss

__all__ = [
    "CountLikelihood",
    "DirichletMultinomialLikelihood",
    "MultiTimeCountLikelihood",
    "MultiTimeEndpointLoss",
    "RolloutRegularizer",
    "UOTLoss",
    "WeakFormLoss",
    "build_target_tensors_by_time",
    "checkpoint_indices_for_taus",
    "checkpoint_indices_for_trajectory",
    "count_fractions_from_zeta",
    "diffusion_magnitude_penalty",
    "drift_action_penalty",
    "GaussianRBFTestFunctions",
    "growth_action_penalty",
    "integrated_fitness",
    "integrated_fitness_curve",
    "make_observed_tau_grid",
    "sinkhorn_divergence",
    "sinkhorn_divergence_normalized",
]
