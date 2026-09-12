"""Orthogonal evidence tier, channel, independence, and claim linkage."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .models import ArtifactRef, StrictModel
from .scientific_scope import ScientificScope

__all__ = (
    "EVIDENCE_CHANNEL_TIER",
    "EvidenceChannel",
    "EvidenceDescriptor",
    "EvidenceExposureStatus",
    "EvidenceIndependence",
    "EvidenceLink",
    "EvidenceSemanticRole",
    "EvidenceTier",
)


class EvidenceTier(StrEnum):
    """Reporting class; deliberately not a universal scalar ordering."""

    E0_ENGINEERING = "E0_engineering"
    E1_IN_SAMPLE = "E1_in_sample"
    E2_HELD_OUT = "E2_held_out"
    E3_INDEPENDENT_COHORT = "E3_independent_cohort"
    E4_ORTHOGONAL_INTERVENTION = "E4_orthogonal_intervention"
    E5_DIRECT_MEASUREMENT = "E5_direct_measurement"


class EvidenceChannel(StrEnum):
    """What was measured or withheld; channels are claim-specific."""

    ENGINEERING_TEST = "engineering_test"
    MODEL_FIT = "model_fit"
    HELDOUT_DONOR = "heldout_donor"
    HELDOUT_GUIDE_TARGET_SHARED = "heldout_guide_target_shared"
    HELDOUT_TARGET = "heldout_target"
    HELDOUT_TIME = "heldout_time"
    INDEPENDENT_COHORT = "independent_cohort"
    ABSOLUTE_CELL_COUNT = "absolute_cell_count"
    STABLE_CLONE_BARCODE = "stable_clone_barcode"
    EVOLVING_BARCODE = "evolving_barcode"
    LIVE_CELL_TRACKING = "live_cell_tracking"
    ORTHOGONAL_COMPONENT_ASSAY = "orthogonal_component_assay"
    PROLIFERATION_ASSAY = "proliferation_assay"
    DEATH_ASSAY = "death_assay"
    MIGRATION_ASSAY = "migration_assay"
    INTERVENTION = "intervention"
    RESCUE = "rescue"
    EPISTASIS = "epistasis"
    SPATIAL_COLOCALIZATION = "spatial_colocalization"


EVIDENCE_CHANNEL_TIER = {
    EvidenceChannel.ENGINEERING_TEST: EvidenceTier.E0_ENGINEERING,
    EvidenceChannel.MODEL_FIT: EvidenceTier.E1_IN_SAMPLE,
    EvidenceChannel.HELDOUT_DONOR: EvidenceTier.E2_HELD_OUT,
    EvidenceChannel.HELDOUT_GUIDE_TARGET_SHARED: EvidenceTier.E2_HELD_OUT,
    EvidenceChannel.HELDOUT_TARGET: EvidenceTier.E2_HELD_OUT,
    EvidenceChannel.HELDOUT_TIME: EvidenceTier.E2_HELD_OUT,
    EvidenceChannel.INDEPENDENT_COHORT: EvidenceTier.E3_INDEPENDENT_COHORT,
    EvidenceChannel.ORTHOGONAL_COMPONENT_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.PROLIFERATION_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.DEATH_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.MIGRATION_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.INTERVENTION: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.RESCUE: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.EPISTASIS: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
    EvidenceChannel.ABSOLUTE_CELL_COUNT: EvidenceTier.E5_DIRECT_MEASUREMENT,
    EvidenceChannel.STABLE_CLONE_BARCODE: EvidenceTier.E5_DIRECT_MEASUREMENT,
    EvidenceChannel.EVOLVING_BARCODE: EvidenceTier.E5_DIRECT_MEASUREMENT,
    EvidenceChannel.LIVE_CELL_TRACKING: EvidenceTier.E5_DIRECT_MEASUREMENT,
    # Spatial colocalization is an association channel, not an intervention.
    EvidenceChannel.SPATIAL_COLOCALIZATION: EvidenceTier.E2_HELD_OUT,
}


class EvidenceIndependence(StrEnum):
    """Independence from the data and decisions used to fit the result."""

    SAME_DATA = "same_data"
    IN_SAMPLE_RESAMPLING = "in_sample_resampling"
    HELDOUT_GUIDE_TARGET_SHARED = "heldout_guide_target_shared"
    HELDOUT_TARGET = "heldout_target"
    HELDOUT_DONOR = "heldout_donor"
    HELDOUT_TIME = "heldout_time"
    INDEPENDENT_COHORT = "independent_cohort"
    ORTHOGONAL_SAME_STUDY = "orthogonal_same_study"
    EXTERNAL_INTERVENTION = "external_intervention"
    DIRECT_OBSERVATION = "direct_observation"


class EvidenceExposureStatus(StrEnum):
    """Whether analysis choices were frozen before evidence was inspected."""

    PROSPECTIVE = "prospective"
    EXPOSED = "exposed"
    NOT_APPLICABLE = "not_applicable"


class EvidenceSemanticRole(StrEnum):
    """Scientific role played by one artifact for one atomic claim."""

    ENGINEERING_QUALIFICATION = "engineering_qualification"
    PROGRAM_COUNT_PREDICTION = "program_count_prediction"
    PROGRAM_STABILITY = "program_stability"
    GUIDE_TARGET_CONSISTENCY = "guide_target_consistency"
    GENE_EFFECT = "gene_effect"
    RELATIVE_ABUNDANCE = "relative_abundance"
    ABSOLUTE_NET_CHANGE = "absolute_net_change"
    PROLIFERATION_COMPONENT = "proliferation_component"
    DEATH_COMPONENT = "death_component"
    MIGRATION_OR_LOSS_COMPONENT = "migration_or_loss_component"
    POPULATION_TRANSITION = "population_transition"
    CLONE_FATE = "clone_fate"
    ANCESTRAL_LINEAGE = "ancestral_lineage"
    DIRECT_CELL_PATH = "direct_cell_path"
    CONTEXT_ASSOCIATION = "context_association"
    ECOLOGICAL_INTERVENTION = "ecological_intervention"
    GENERATOR_ATTRIBUTION = "generator_attribution"
    EXTERNAL_CONCORDANCE = "external_concordance"


class EvidenceDescriptor(StrictModel):
    """Independent evidence axes; no claim is authorized from tier alone."""

    tier: EvidenceTier
    channel: EvidenceChannel
    independence: EvidenceIndependence
    unit_of_replication: str = Field(min_length=1)
    prospective_or_exposed_status: EvidenceExposureStatus
    multiplicity_family: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_descriptor(self) -> EvidenceDescriptor:
        if EVIDENCE_CHANNEL_TIER[self.channel] != self.tier:
            raise ValueError("Evidence tier contradicts its measurement channel.")
        expected_independence = {
            EvidenceChannel.HELDOUT_GUIDE_TARGET_SHARED: (
                EvidenceIndependence.HELDOUT_GUIDE_TARGET_SHARED
            ),
            EvidenceChannel.HELDOUT_TARGET: EvidenceIndependence.HELDOUT_TARGET,
            EvidenceChannel.HELDOUT_DONOR: EvidenceIndependence.HELDOUT_DONOR,
            EvidenceChannel.HELDOUT_TIME: EvidenceIndependence.HELDOUT_TIME,
            EvidenceChannel.INDEPENDENT_COHORT: EvidenceIndependence.INDEPENDENT_COHORT,
        }.get(self.channel)
        if expected_independence is not None and self.independence != expected_independence:
            raise ValueError("Evidence channel contradicts its independence class.")
        return self


class EvidenceLink(StrictModel):
    """Semantic, scope-exact link from one artifact and result to one claim."""

    claim_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    artifact_ref: ArtifactRef
    semantic_role: EvidenceSemanticRole
    study_id: str = Field(min_length=1)
    evidence_study_id: str = Field(min_length=1)
    scope: ScientificScope
    split_id: str | None = None
    metric_or_statistic_id: str = Field(min_length=1)
    descriptor: EvidenceDescriptor

    @model_validator(mode="after")
    def validate_link(self) -> EvidenceLink:
        heldout = self.descriptor.tier == EvidenceTier.E2_HELD_OUT
        if heldout != (self.split_id is not None):
            raise ValueError("Held-out evidence requires exactly one split_id.")
        independent = self.descriptor.channel == EvidenceChannel.INDEPENDENT_COHORT
        if independent == (self.evidence_study_id == self.study_id):
            raise ValueError("Independent-cohort status contradicts evidence study identity.")
        return self
