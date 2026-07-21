"""CREDO's compact, canonical public API."""

from .contracts import Axis, FiniteMeasure, MassSemantics, TrajectoryData
from .counterfactual import counterfactual
from .io import load_config, load_data
from .model import CREDOModel
from .training import Trainer

__all__ = [
    "Axis",
    "CREDOModel",
    "FiniteMeasure",
    "MassSemantics",
    "Trainer",
    "TrajectoryData",
    "counterfactual",
    "load_config",
    "load_data",
]

__version__ = "3.0.0a1"
