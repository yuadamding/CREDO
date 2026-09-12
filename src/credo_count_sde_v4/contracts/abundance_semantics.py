"""Factorized abundance scale, entity, process, and calibration contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .evidence_tiers import EvidenceChannel, EvidenceDescriptor
from .models import ArtifactRef, StrictModel
from .scientific_scope import ScientificScope

__all__ = (
    "AbundanceCalibration",
    "AbundanceCapabilityProfile",
    "AbundanceComponentEvidence",
    "AbundanceEntity",
    "AbundanceObservationModel",
    "AbundanceProcess",
    "AbundanceResult",
    "AbundanceScale",
)


class AbundanceScale(StrEnum):
    """Measurement scale, independent of the entity being counted."""

    RELATIVE_WITHIN_POOL = "relative_within_pool"
    CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT = "capture_calibrated_sampled_compartment"
    ABSOLUTE_TISSUE = "absolute_tissue"


class AbundanceEntity(StrEnum):
    """Biological entity whose abundance is estimated."""

    GUIDE = "guide"
    TARGET = "target"
    PERTURBATION = "perturbation"
    CLONE = "clone"
    CELL_STATE = "cell_state"
    CLONE_STATE = "clone_state"


class AbundanceProcess(StrEnum):
    """Process identified by the observation and component evidence."""

    RELATIVE_SELECTION = "relative_selection"
    ABSOLUTE_NET_CHANGE = "absolute_net_change"
    PROLIFERATION = "proliferation"
    DEATH = "death"
    MIGRATION_OR_COMPARTMENT_LOSS = "migration_or_compartment_loss"


class AbundanceObservationModel(StrEnum):
    """Likelihood family attached to the measurement scale."""

    DIRICHLET_MULTINOMIAL_RELATIVE = "dirichlet_multinomial_relative"
    DIRICHLET_MULTINOMIAL_PLUS_CAPTURE_NB = "dirichlet_multinomial_plus_capture_nb"
    ABSOLUTE_TOTAL_NEGATIVE_BINOMIAL = "absolute_total_negative_binomial"


class AbundanceCapabilityProfile(StrictModel):
    """One structurally supported scale with its entity/process resolution."""

    scale: AbundanceScale
    entities: tuple[AbundanceEntity, ...]
    processes: tuple[AbundanceProcess, ...]
    system_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile(self) -> AbundanceCapabilityProfile:
        for name in ("entities", "processes"):
            values = getattr(self, name)
            if not values or values != tuple(sorted(set(values), key=lambda value: value.value)):
                raise ValueError(f"Abundance profile {name} must be unique and sorted.")
        if self.scale == AbundanceScale.RELATIVE_WITHIN_POOL and self.processes != (
            AbundanceProcess.RELATIVE_SELECTION,
        ):
            raise ValueError("Relative abundance identifies only relative selection.")
        return self


class AbundanceCalibration(StrictModel):
    """Exact system boundary and artifacts behind non-relative abundance."""

    system_boundary: str = Field(min_length=1)
    calibration_basis: str = Field(min_length=1)
    calibration_timepoints: tuple[str, ...]
    capture_fraction_model: str | None = None
    total_count_artifact: ArtifactRef
    uncertainty_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_calibration(self) -> AbundanceCalibration:
        if len(self.calibration_timepoints) < 2 or self.calibration_timepoints != tuple(
            sorted(set(self.calibration_timepoints))
        ):
            raise ValueError("Net-change calibration requires at least two unique timepoints.")
        return self


class AbundanceComponentEvidence(StrictModel):
    """One abundance process and its claim-specific measurement descriptors."""

    process: AbundanceProcess
    evidence: tuple[EvidenceDescriptor, ...]
    artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_component_support(self) -> AbundanceComponentEvidence:
        if not self.evidence or len(self.evidence) != len(set(self.evidence)):
            raise ValueError("Abundance component evidence must be nonempty and unique.")
        channels = {item.channel for item in self.evidence}
        required = {
            AbundanceProcess.ABSOLUTE_NET_CHANGE: {EvidenceChannel.ABSOLUTE_CELL_COUNT},
            AbundanceProcess.PROLIFERATION: {
                EvidenceChannel.ABSOLUTE_CELL_COUNT,
                EvidenceChannel.PROLIFERATION_ASSAY,
            },
            AbundanceProcess.DEATH: {
                EvidenceChannel.ABSOLUTE_CELL_COUNT,
                EvidenceChannel.DEATH_ASSAY,
            },
            AbundanceProcess.MIGRATION_OR_COMPARTMENT_LOSS: {
                EvidenceChannel.ABSOLUTE_CELL_COUNT,
                EvidenceChannel.MIGRATION_ASSAY,
            },
        }.get(self.process, set())
        if not required <= channels:
            missing = ", ".join(sorted(item.value for item in required - channels))
            raise ValueError(f"{self.process.value} lacks required evidence: {missing}.")
        if self.process in {
            AbundanceProcess.PROLIFERATION,
            AbundanceProcess.DEATH,
            AbundanceProcess.MIGRATION_OR_COMPARTMENT_LOSS,
        } and not channels & {
            EvidenceChannel.INTERVENTION,
            EvidenceChannel.RESCUE,
            EvidenceChannel.EPISTASIS,
            EvidenceChannel.ORTHOGONAL_COMPONENT_ASSAY,
        }:
            raise ValueError(
                "Separated abundance components require component intervention evidence."
            )
        return self


class AbundanceResult(StrictModel):
    """Hash-bound abundance result with orthogonal scale/entity/process axes."""

    schema_id: Literal["credo.abundance_result"] = "credo.abundance_result"
    schema_version: Literal[1] = 1
    abundance_result_id: str
    study_id: str = Field(min_length=1)
    capability_assessment_id: str = Field(min_length=1)
    scope: ScientificScope
    scale: AbundanceScale
    entity: AbundanceEntity
    system_boundary: str = Field(min_length=1)
    observation_model: AbundanceObservationModel
    components: tuple[AbundanceComponentEvidence, ...]
    calibration: AbundanceCalibration | None = None

    @model_validator(mode="after")
    def validate_abundance_result(self) -> AbundanceResult:
        processes = tuple(item.process for item in self.components)
        if not processes or len(processes) != len(set(processes)):
            raise ValueError("Abundance processes must be nonempty and unique.")
        expected_model = {
            AbundanceScale.RELATIVE_WITHIN_POOL: (
                AbundanceObservationModel.DIRICHLET_MULTINOMIAL_RELATIVE
            ),
            AbundanceScale.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT: (
                AbundanceObservationModel.DIRICHLET_MULTINOMIAL_PLUS_CAPTURE_NB
            ),
            AbundanceScale.ABSOLUTE_TISSUE: (
                AbundanceObservationModel.ABSOLUTE_TOTAL_NEGATIVE_BINOMIAL
            ),
        }[self.scale]
        if self.observation_model != expected_model:
            raise ValueError("Abundance observation model contradicts its scale.")
        relative = self.scale == AbundanceScale.RELATIVE_WITHIN_POOL
        if relative:
            if set(processes) != {AbundanceProcess.RELATIVE_SELECTION}:
                raise ValueError("Relative abundance supports only relative selection.")
            if self.calibration is not None:
                raise ValueError("Relative abundance cannot carry absolute calibration.")
        else:
            if self.calibration is None or self.calibration.system_boundary != self.system_boundary:
                raise ValueError("Non-relative abundance requires boundary-matched calibration.")
        if (
            self.scale == AbundanceScale.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT
            and self.calibration is not None
            and self.calibration.capture_fraction_model is None
        ):
            raise ValueError("Capture-calibrated abundance requires a capture-fraction model.")
        expected = self.identity(id_field="abundance_result_id")
        if self.abundance_result_id != expected:
            raise ValueError(f"abundance_result_id mismatch: expected {expected}.")
        return self
