"""Recipe-facing longitudinal Perturb-seq compiled problem contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..data.splits import SplitPlan


def _ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique nonempty IDs.")
    return normalized


@dataclass(frozen=True)
class CompiledObservationSet:
    """Stable identities exposed to one compilation partition."""

    observation_ids: tuple[str, ...]
    series_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("observation_ids", "series_ids", "checkpoint_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))


@dataclass(frozen=True)
class CompiledLPSSplit:
    """Outcome-separated source, training, validation, and denominator partitions."""

    plan: SplitPlan
    source: CompiledObservationSet
    training_targets: CompiledObservationSet
    validation_targets: CompiledObservationSet
    composition_background: CompiledObservationSet | None = None

    def __post_init__(self) -> None:
        overlap = set(self.training_targets.observation_ids) & set(
            self.validation_targets.observation_ids
        )
        if overlap and self.plan.source == "held_out":
            raise ValueError(
                "Training and validation target outcomes must be disjoint; "
                f"overlap={sorted(overlap)[:5]}."
            )


@dataclass(frozen=True)
class CompiledLPSProblem:
    """Base metadata shared by all longitudinal Perturb-seq problem families."""

    problem_kind: str
    partition: CompiledLPSSplit
    study_content_hash: str
    selection_hash: str
    problem_hash: str
    problem_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("problem_kind", "study_content_hash", "selection_hash", "problem_hash"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"CompiledLPSProblem.{name} must be nonempty.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "problem_metadata",
            MappingProxyType(dict(self.problem_metadata)),
        )


@dataclass(frozen=True)
class UnbalancedFlowProblem(CompiledLPSProblem):
    """Finite-measure endpoint pairs and coupling/training specifications."""

    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class StateSequencePredictionProblem(CompiledLPSProblem):
    """Source/context state sequence and perturbation-conditioned target query."""

    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class CouplingProblem(CompiledLPSProblem):
    """Adjacent-checkpoint probabilistic coupling inference problem."""

    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


__all__ = [
    "CompiledLPSProblem",
    "CompiledLPSSplit",
    "CompiledObservationSet",
    "CouplingProblem",
    "StateSequencePredictionProblem",
    "UnbalancedFlowProblem",
]
