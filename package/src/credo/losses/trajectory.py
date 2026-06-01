"""Trajectory-level loss utilities.

Two-endpoint code can use :mod:`credo.losses.endpoint` and
:mod:`credo.losses.counts` directly. Multi-time workflows can import the
checkpointed endpoint and cumulative count helpers from this compact facade.
"""
from __future__ import annotations

from .counts import MultiTimeCountLikelihood, count_fractions_from_zeta, integrated_fitness_curve
from .multitime import (
    MultiTimeEndpointLoss,
    build_target_tensors_by_time,
    checkpoint_indices_for_taus,
    checkpoint_indices_for_trajectory,
    make_observed_tau_grid,
)

__all__ = [
    "MultiTimeCountLikelihood",
    "MultiTimeEndpointLoss",
    "build_target_tensors_by_time",
    "checkpoint_indices_for_taus",
    "checkpoint_indices_for_trajectory",
    "count_fractions_from_zeta",
    "integrated_fitness_curve",
    "make_observed_tau_grid",
]
