"""CREDO's compact semantic study and recipe execution API."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from . import contracts as _contracts
from .artifacts import bind_run_study, open_run
from .counterfactual import counterfactual
from .data import SelectionSpec, SplitPlan, open_study, write_study
from .evaluation import evaluate
from .io import load_config
from .lps import PerturbSeqSelection, PerturbSeqStudy, PerturbSeqView
from .registry import get_recipe
from .runtime import (
    CounterfactualQuery,
    CounterfactualResult,
    PredictionQuery,
    PredictionResult,
    train,
)

Axis = _contracts.Axis
CapabilitySet = _contracts.CapabilitySet
FiniteMeasure = _contracts.FiniteMeasure
MassSemantics = _contracts.MassSemantics
RepresentationArtifact = _contracts.RepresentationArtifact
SplitSpec = _contracts.SplitSpec

# Domain-specific public names. Schema-v3 ``Study`` remains available from
# ``credo.data`` only for compatibility conversion.
Study = PerturbSeqStudy
StudyView = PerturbSeqView

__all__ = [
    "SelectionSpec",
    "SplitPlan",
    "Study",
    "StudyView",
    "PerturbSeqSelection",
    "PerturbSeqStudy",
    "PerturbSeqView",
    "PredictionQuery",
    "PredictionResult",
    "CounterfactualQuery",
    "CounterfactualResult",
    "bind_run_study",
    "counterfactual",
    "evaluate",
    "get_recipe",
    "load_config",
    "open_study",
    "open_run",
    "train",
    "write_study",
]

__version__ = "3.0.0a5"

_DEPRECATED_EXPORTS = {
    "CREDOModel": ("credo.recipes.compact_sde_v3.model", "CREDOModel"),
    "CREDOStudy": ("credo.contracts", "CREDOStudy"),
    "Trainer": ("credo.recipes.compact_sde_v3.training", "Trainer"),
    "TrajectoryData": ("credo.contracts", "TrajectoryData"),
    "load_data": ("credo.io", "load_data"),
}


def __getattr__(name: str) -> Any:
    target = _DEPRECATED_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"credo.{name} is a compatibility API; use Study/open_study/train/open_run instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value
