"""Public endpoint finite-measure loss API.

The default endpoint objective is geometry-plus-log-mass matching, not a full
dynamic unbalanced-OT path objective.  Legacy imports from ``credo.losses.uot``
remain available for compatibility.
"""
from __future__ import annotations

from .uot import (
    EndpointGeometryMassLoss,
    endpoint_geometry_mass_components,
    endpoint_geometry_mass_loss,
    sinkhorn_divergence_normalized,
)

__all__ = [
    "EndpointGeometryMassLoss",
    "endpoint_geometry_mass_components",
    "endpoint_geometry_mass_loss",
    "sinkhorn_divergence_normalized",
]
