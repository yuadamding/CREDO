"""Split-scoped finite-measure dynamics problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import TrajectoryData
from .base import CompiledLPSProblem


@dataclass(frozen=True)
class FiniteMeasureDynamicsProblem(CompiledLPSProblem):
    """Outcome-separated finite-measure inputs for SDE/ODE dynamics recipes."""

    training: TrajectoryData | None = None
    validation: TrajectoryData | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training is None or self.validation is None:
            raise ValueError("FiniteMeasureDynamicsProblem requires train and validation data.")
        if self.training.axis != self.validation.axis:
            raise ValueError("Training and validation problems must share one progression axis.")
        if self.training.representation != self.validation.representation:
            raise ValueError("Training and validation problems must share one representation.")
        training_targets = {
            observation_id for observation_id in self.partition.training_targets.observation_ids
        }
        validation_targets = {
            observation_id for observation_id in self.partition.validation_targets.observation_ids
        }
        if self.partition.plan.source == "held_out" and training_targets & validation_targets:
            raise ValueError("Finite-measure target partitions overlap.")

    @property
    def axis(self):
        return self.training.axis

    @property
    def measures(self):
        return self.training.measures

    @property
    def measure_meta(self):
        return self.training.measure_meta

    @property
    def mass_semantics(self):
        return self.training.mass_semantics

    @property
    def count_blocks(self):
        return self.training.count_blocks

    @property
    def representation(self):
        return self.training.representation

    @property
    def measure_ids(self) -> tuple[str, ...]:
        return self.training.measure_ids

    @property
    def embedding_ids(self) -> tuple[str, ...]:
        return self.training.embedding_ids

    @property
    def control_embedding_ids(self) -> tuple[str, ...]:
        return self.training.control_embedding_ids

    @property
    def latent_dim(self) -> int:
        return self.training.latent_dim

    @property
    def claim_policy(self) -> dict[str, str | bool]:
        return self.training.claim_policy

    def __getattr__(self, name: str) -> Any:
        training = object.__getattribute__(self, "training")
        if training is not None:
            return getattr(training, name)
        raise AttributeError(name)


FiniteMeasureTrajectoryProblem = FiniteMeasureDynamicsProblem
EndpointFiniteMeasureProblem = FiniteMeasureDynamicsProblem

__all__ = [
    "EndpointFiniteMeasureProblem",
    "FiniteMeasureDynamicsProblem",
    "FiniteMeasureTrajectoryProblem",
]
