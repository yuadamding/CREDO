"""Population-trajectory semantics separated from lineage evidence."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from .evidence_tiers import EvidenceChannel, EvidenceDescriptor
from .lineage_semantics import (
    LINEAGE_RESULT_MINIMUM_LEVEL,
    LineageEvidenceLevel,
    LineageResultSemantics,
    lineage_level_at_least,
)
from .models import ArtifactRef, StrictModel
from .scientific_scope import ScientificScope

__all__ = ("ObservedCheckpoint", "TrajectoryResult")


class ObservedCheckpoint(StrictModel):
    """One observed checkpoint on a physical-time scale."""

    checkpoint_id: str = Field(min_length=1)
    physical_time: float

    @model_validator(mode="after")
    def validate_time(self) -> ObservedCheckpoint:
        if not math.isfinite(self.physical_time):
            raise ValueError("Checkpoint physical time must be finite.")
        return self


class TrajectoryResult(StrictModel):
    """A transition-law result whose lineage language is evidence-gated."""

    schema_id: Literal["credo.trajectory_result"] = "credo.trajectory_result"
    schema_version: Literal[1] = 1
    trajectory_result_id: str
    study_id: str = Field(min_length=1)
    capability_assessment_id: str = Field(min_length=1)
    scope: ScientificScope
    evidence: tuple[EvidenceDescriptor, ...]
    lineage_level: LineageEvidenceLevel
    semantics: LineageResultSemantics
    observed_checkpoints: tuple[ObservedCheckpoint, ...]
    predicted_finite_measures: ArtifactRef
    transition_kernels: ArtifactRef
    uncertainty: ArtifactRef
    terminal_fate_probabilities: ArtifactRef | None = None
    heldout_time_evaluation: ArtifactRef | None = None
    semigroup_evaluation: ArtifactRef | None = None
    lineage_evaluation: ArtifactRef | None = None
    individual_cell_paths_claimed: bool = False

    @model_validator(mode="after")
    def validate_trajectory(self) -> TrajectoryResult:
        ids = [item.checkpoint_id for item in self.observed_checkpoints]
        times = [item.physical_time for item in self.observed_checkpoints]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("A trajectory requires at least two unique observed checkpoints.")
        if not self.evidence or len(self.evidence) != len(set(self.evidence)):
            raise ValueError("Trajectory evidence must be nonempty and unique.")
        if any(right <= left for left, right in zip(times, times[1:], strict=False)):
            raise ValueError("Trajectory checkpoints must be strictly increasing in time.")
        required_level = LINEAGE_RESULT_MINIMUM_LEVEL[self.semantics]
        if not lineage_level_at_least(self.lineage_level, required_level):
            raise ValueError("Trajectory semantics exceed the observed lineage evidence level.")
        lineage_semantics = self.semantics != LineageResultSemantics.POPULATION_TRANSITION_ONLY
        channels = {item.channel for item in self.evidence}
        required_channel = {
            LineageResultSemantics.CLONE_RESOLVED_FATE: EvidenceChannel.STABLE_CLONE_BARCODE,
            LineageResultSemantics.ANCESTRAL_TREE: EvidenceChannel.EVOLVING_BARCODE,
            LineageResultSemantics.DIRECT_CELL_PATH_VALIDATION: (
                EvidenceChannel.LIVE_CELL_TRACKING
            ),
        }.get(self.semantics)
        if required_channel is not None and required_channel not in channels:
            raise ValueError(f"Trajectory semantics require {required_channel.value} evidence.")
        if not lineage_semantics and channels == {EvidenceChannel.ENGINEERING_TEST}:
            raise ValueError("A population transition requires model or predictive evidence.")
        if lineage_semantics != (self.lineage_evaluation is not None):
            raise ValueError("Lineage-bearing trajectory semantics require lineage evaluation.")
        direct = self.semantics == LineageResultSemantics.DIRECT_CELL_PATH_VALIDATION
        if self.individual_cell_paths_claimed != direct:
            raise ValueError("Individual-cell paths are permitted only with L3 direct evidence.")
        if self.semigroup_evaluation is not None and len(ids) < 3:
            raise ValueError("Semigroup evaluation requires at least three checkpoints.")
        expected = self.identity(id_field="trajectory_result_id")
        if self.trajectory_result_id != expected:
            raise ValueError(f"trajectory_result_id mismatch: expected {expected}.")
        return self
