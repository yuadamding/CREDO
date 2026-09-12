"""Dataset capabilities derived from observed fields and explicit design gates."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, TypeAdapter, model_validator

from .abundance_semantics import (
    AbundanceCapabilityProfile,
    AbundanceEntity,
    AbundanceProcess,
    AbundanceScale,
)
from .lineage_semantics import LineageEvidenceLevel
from .models import ArtifactRef, Sha256, StrictModel

__all__ = (
    "AbsoluteAbundanceDesignContract",
    "AbsoluteCountBasis",
    "CapabilityDecision",
    "CloneObservationContract",
    "DatasetCapabilityAssessment",
    "PhysicalContextDesignContract",
    "ProtectedExpressionAccessContract",
    "RelativeAbundanceDesignContract",
    "ScientificCapability",
    "StudyEntityCounts",
    "StudyEvidenceContract",
    "StudyField",
    "derive_dataset_capabilities",
    "freeze_study_evidence_contract",
    "validate_dataset_capabilities",
)


class StudyField(StrEnum):
    """Physically observed fields that may unlock a scientific result layer."""

    BIOLOGICAL_SAMPLE_ID = "biological_sample_id"
    PHYSICAL_POOL_ID = "physical_pool_id"
    DONOR_ID = "donor_id"
    REPLICATE_ID = "replicate_id"
    TIME = "time"
    PERTURBATION_ID = "perturbation_id"
    GUIDE_ID = "guide_id"
    TARGET_ID = "target_id"
    CELL_ID = "cell_id"
    CELL_STATE = "cell_state"
    CLONE_ID = "clone_id"
    LINEAGE_PARENT_ID = "lineage_parent_id"
    LIVE_CELL_TRACK_ID = "live_cell_track_id"
    INPUT_GUIDE_COUNT = "input_guide_count"
    RECOVERED_GUIDE_COUNT = "recovered_guide_count"
    ABSOLUTE_TOTAL_CELL_COUNT = "absolute_total_cell_count"
    CAPTURE_FRACTION = "capture_fraction"
    SPATIAL_COORDINATES = "spatial_coordinates"
    RNA_COUNTS = "rna_counts"
    PROLIFERATION_ASSAY = "proliferation_assay"
    DEATH_ASSAY = "death_assay"
    MIGRATION_ASSAY = "migration_assay"
    MEDIATOR_INTERVENTION = "mediator_intervention"


class AbsoluteCountBasis(StrEnum):
    """Physical calibration scale, not an inferred biological process."""

    CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT = "capture_calibrated_sampled_compartment"
    ABSOLUTE_TISSUE = "absolute_tissue"


class ProtectedExpressionAccessContract(StrictModel):
    """Leakage firewall for every decision that could inspect protected expression."""

    protected_access_contract_id: str
    protected_checkpoint_ids: tuple[str, ...]
    representation_fit_accessed: Literal[False] = False
    feature_selection_accessed: Literal[False] = False
    normalization_estimation_accessed: Literal[False] = False
    batch_correction_accessed: Literal[False] = False
    model_selection_accessed: Literal[False] = False
    stopping_accessed: Literal[False] = False
    metric_threshold_selection_accessed: Literal[False] = False
    program_or_pathway_discovery_accessed: Literal[False] = False
    audit_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_access(self) -> ProtectedExpressionAccessContract:
        if not self.protected_checkpoint_ids or self.protected_checkpoint_ids != tuple(
            sorted(set(self.protected_checkpoint_ids))
        ):
            raise ValueError("Protected checkpoints must be nonempty, unique, and sorted.")
        expected = self.identity(id_field="protected_access_contract_id")
        if self.protected_access_contract_id != expected:
            raise ValueError(f"protected_access_contract_id mismatch: expected {expected}.")
        return self


class RelativeAbundanceDesignContract(StrictModel):
    """Design gates needed before guide fractions identify relative selection."""

    relative_design_id: str
    pool_or_exposure_alignment: Literal[True] = True
    guide_catalog_sha256: Sha256
    stable_denominator_definition: str = Field(min_length=1)
    replicate_identity_complete: Literal[True] = True
    sampling_uncertainty_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_design(self) -> RelativeAbundanceDesignContract:
        expected = self.identity(id_field="relative_design_id")
        if self.relative_design_id != expected:
            raise ValueError(f"relative_design_id mismatch: expected {expected}.")
        return self


class AbsoluteAbundanceDesignContract(StrictModel):
    """System boundary and repeated calibration for non-relative counts."""

    absolute_design_id: str
    basis: AbsoluteCountBasis
    system_boundary: str = Field(min_length=1)
    calibration_timepoints: tuple[str, ...]
    capture_fraction_model: str | None = None
    total_count_artifact: ArtifactRef
    uncertainty_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_design(self) -> AbsoluteAbundanceDesignContract:
        if len(self.calibration_timepoints) < 2 or self.calibration_timepoints != tuple(
            sorted(set(self.calibration_timepoints))
        ):
            raise ValueError("Absolute net change requires repeated calibrated timepoints.")
        capture = self.basis == AbsoluteCountBasis.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT
        if capture != (self.capture_fraction_model is not None):
            raise ValueError("Capture-fraction model must agree with the calibration basis.")
        expected = self.identity(id_field="absolute_design_id")
        if self.absolute_design_id != expected:
            raise ValueError(f"absolute_design_id mismatch: expected {expected}.")
        return self


class PhysicalContextDesignContract(StrictModel):
    """Separates co-residence, counterfactual modelling, and intervention."""

    context_design_id: str
    populations_co_resident: bool
    perturbations_cultured_separately: bool
    qualified_context_model: bool = False
    mediator_intervention_observed: bool = False
    audit_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_design(self) -> PhysicalContextDesignContract:
        if self.populations_co_resident == self.perturbations_cultured_separately:
            raise ValueError("Context co-residence and separate-culture declarations conflict.")
        if self.qualified_context_model and not self.populations_co_resident:
            raise ValueError("A context counterfactual requires co-resident populations.")
        if self.mediator_intervention_observed and not self.populations_co_resident:
            raise ValueError("An ecological intervention requires co-resident populations.")
        expected = self.identity(id_field="context_design_id")
        if self.context_design_id != expected:
            raise ValueError(f"context_design_id mismatch: expected {expected}.")
        return self


class CloneObservationContract(StrictModel):
    """Quality gates for stable and evolving barcode interpretations."""

    clone_observation_id: str
    cross_time_continuity_verified: bool
    collision_rate: float = Field(ge=0, le=1)
    maximum_collision_rate: float = Field(ge=0, le=1)
    dropout_rate: float = Field(ge=0, le=1)
    maximum_dropout_rate: float = Field(ge=0, le=1)
    minimum_clone_support: int = Field(ge=2)
    assignment_confidence: float = Field(ge=0, le=1)
    minimum_assignment_confidence: float = Field(ge=0, le=1)
    biological_replicates: int = Field(ge=1)
    observation_model: str = Field(min_length=1)
    evolving_barcode_history_observed: bool = False
    parent_relationship_directly_observed: bool = False
    parent_tree_inferred_only: bool = False
    audit_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_clone_observation(self) -> CloneObservationContract:
        if (
            self.parent_relationship_directly_observed
            and not self.evolving_barcode_history_observed
        ):
            raise ValueError("Observed parent relationships require an evolving barcode history.")
        if self.parent_relationship_directly_observed and self.parent_tree_inferred_only:
            raise ValueError(
                "A parent relation cannot be both directly observed and inferred only."
            )
        expected = self.identity(id_field="clone_observation_id")
        if self.clone_observation_id != expected:
            raise ValueError(f"clone_observation_id mismatch: expected {expected}.")
        return self

    @property
    def stable_clone_qualified(self) -> bool:
        return (
            self.cross_time_continuity_verified
            and self.collision_rate <= self.maximum_collision_rate
            and self.dropout_rate <= self.maximum_dropout_rate
            and self.assignment_confidence >= self.minimum_assignment_confidence
            and self.biological_replicates >= 2
        )

    @property
    def ancestry_qualified(self) -> bool:
        return (
            self.stable_clone_qualified
            and self.evolving_barcode_history_observed
            and self.parent_relationship_directly_observed
            and not self.parent_tree_inferred_only
        )


class ScientificCapability(StrEnum):
    """Structurally detectable scientific capability."""

    PERTURBATION_PROGRAMS = "supports_perturbation_programs"
    POPULATION_TRAJECTORY = "supports_population_trajectory"
    STRICT_HELDOUT_TIME = "supports_strict_heldout_time"
    RELATIVE_ABUNDANCE = "supports_relative_abundance"
    ABSOLUTE_ABUNDANCE = "supports_absolute_abundance"
    PHYSICAL_CONTEXT_ASSOCIATION = "supports_physical_context_association"
    ECOLOGICAL_COUNTERFACTUAL = "supports_ecological_counterfactual"
    ECOLOGICAL_INTERVENTION = "supports_ecological_intervention"
    CLONE_LINEAGE = "supports_clone_lineage"
    ANCESTRAL_LINEAGE = "supports_ancestral_lineage"
    DIRECT_CELL_PATH = "supports_direct_cell_path"


class StudyEntityCounts(StrictModel):
    """Cardinalities used for structural capability checks, not outcome claims."""

    biological_samples: int = Field(ge=0)
    physical_pools: int = Field(ge=0)
    donors: int = Field(ge=0)
    replicates: int = Field(ge=0)
    checkpoints: int = Field(ge=1)
    perturbations: int = Field(ge=0)
    guides: int = Field(ge=0)
    targets: int = Field(ge=0)
    cells: int = Field(ge=0)
    clones: int = Field(ge=0)


class StudyEvidenceContract(StrictModel):
    """Observed study design plus auditable non-field gates."""

    schema_id: Literal["credo.study_evidence_contract"] = "credo.study_evidence_contract"
    schema_version: Literal[1] = 1
    study_contract_id: str
    study_id: str = Field(min_length=1)
    observed_fields: tuple[StudyField, ...]
    entity_counts: StudyEntityCounts
    physical_pool_identity_complete: bool = False
    destructive_snapshots_only: bool = True
    protected_expression_access: ProtectedExpressionAccessContract | None = None
    relative_abundance_design: RelativeAbundanceDesignContract | None = None
    absolute_abundance_design: AbsoluteAbundanceDesignContract | None = None
    physical_context_design: PhysicalContextDesignContract | None = None
    clone_observation: CloneObservationContract | None = None

    @model_validator(mode="after")
    def validate_study(self) -> StudyEvidenceContract:
        expected_fields = tuple(sorted(set(self.observed_fields), key=lambda value: value.value))
        if not expected_fields or self.observed_fields != expected_fields:
            raise ValueError("Observed study fields must be nonempty, unique, and sorted.")
        fields = set(self.observed_fields)
        count_fields = {
            StudyField.BIOLOGICAL_SAMPLE_ID: self.entity_counts.biological_samples,
            StudyField.PHYSICAL_POOL_ID: self.entity_counts.physical_pools,
            StudyField.DONOR_ID: self.entity_counts.donors,
            StudyField.REPLICATE_ID: self.entity_counts.replicates,
            StudyField.PERTURBATION_ID: self.entity_counts.perturbations,
            StudyField.GUIDE_ID: self.entity_counts.guides,
            StudyField.TARGET_ID: self.entity_counts.targets,
            StudyField.CELL_ID: self.entity_counts.cells,
            StudyField.CLONE_ID: self.entity_counts.clones,
        }
        for field, count in count_fields.items():
            if (field in fields) != (count > 0):
                raise ValueError(f"{field.value} presence contradicts its entity count.")
        if StudyField.TIME not in fields and self.entity_counts.checkpoints != 1:
            raise ValueError("Multiple checkpoints require an observed time field.")
        if self.physical_pool_identity_complete and StudyField.PHYSICAL_POOL_ID not in fields:
            raise ValueError("Complete physical-pool identity requires physical_pool_id.")
        if StudyField.LIVE_CELL_TRACK_ID in fields and self.destructive_snapshots_only:
            raise ValueError("Live-cell tracks contradict destructive-snapshots-only semantics.")
        if self.protected_expression_access is not None and (
            self.entity_counts.checkpoints < 3
            or StudyField.TIME not in fields
            or StudyField.RNA_COUNTS not in fields
        ):
            raise ValueError("Protected-time access requires at least three count checkpoints.")
        if (
            self.relative_abundance_design is not None
            and not {
                StudyField.PHYSICAL_POOL_ID,
                StudyField.GUIDE_ID,
                StudyField.INPUT_GUIDE_COUNT,
                StudyField.RECOVERED_GUIDE_COUNT,
            }
            <= fields
        ):
            raise ValueError("Relative-abundance design lacks its observed count fields.")
        if self.absolute_abundance_design is not None:
            if StudyField.ABSOLUTE_TOTAL_CELL_COUNT not in fields:
                raise ValueError("Absolute-abundance design requires total cell counts.")
            capture = (
                self.absolute_abundance_design.basis
                == AbsoluteCountBasis.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT
            )
            if capture and StudyField.CAPTURE_FRACTION not in fields:
                raise ValueError("Capture calibration requires an observed capture fraction.")
        if self.physical_context_design is not None and not self.physical_pool_identity_complete:
            raise ValueError("Physical-context design requires complete physical pools.")
        if self.clone_observation is not None and StudyField.CLONE_ID not in fields:
            raise ValueError("Clone-observation design requires clone_id.")
        expected = self.identity(id_field="study_contract_id")
        if self.study_contract_id != expected:
            raise ValueError(f"study_contract_id mismatch: expected {expected}.")
        return self


class CapabilityDecision(StrictModel):
    """One explicit capability decision and every blocking reason."""

    capability: ScientificCapability
    supported: bool
    required_fields: tuple[StudyField, ...]
    missing_fields: tuple[StudyField, ...]
    blocking_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_decision(self) -> CapabilityDecision:
        for name in ("required_fields", "missing_fields"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values), key=lambda value: value.value)):
                raise ValueError(f"Capability {name} must be unique and sorted.")
        if not set(self.missing_fields) <= set(self.required_fields):
            raise ValueError("Capability missing fields must be a subset of required fields.")
        if self.supported == bool(self.missing_fields or self.blocking_reasons):
            raise ValueError("Capability support contradicts its blocking evidence.")
        return self


class DatasetCapabilityAssessment(StrictModel):
    """Content-addressed, fail-closed structural assessment for one study."""

    schema_id: Literal["credo.dataset_capability_assessment"] = (
        "credo.dataset_capability_assessment"
    )
    schema_version: Literal[1] = 1
    capability_assessment_id: str
    study_contract_id: str = Field(min_length=1)
    decisions: tuple[CapabilityDecision, ...]
    abundance_profiles: tuple[AbundanceCapabilityProfile, ...]
    maximum_lineage_level: LineageEvidenceLevel
    structural_capability_only: Literal[True] = True
    scientific_validation_passed: Literal[False] = False

    @model_validator(mode="after")
    def validate_assessment(self) -> DatasetCapabilityAssessment:
        if tuple(item.capability for item in self.decisions) != tuple(ScientificCapability):
            raise ValueError("Capability assessment must cover every capability in frozen order.")
        scales = tuple(item.scale for item in self.abundance_profiles)
        if len(scales) != len(set(scales)) or scales != tuple(
            sorted(scales, key=lambda value: value.value)
        ):
            raise ValueError("Abundance profiles must have unique, sorted scales.")
        expected = self.identity(id_field="capability_assessment_id")
        if self.capability_assessment_id != expected:
            raise ValueError(f"capability_assessment_id mismatch: expected {expected}.")
        return self

    def supports(self, capability: ScientificCapability) -> bool:
        return next(item.supported for item in self.decisions if item.capability == capability)

    def abundance_profile(self, scale: AbundanceScale) -> AbundanceCapabilityProfile | None:
        return next((item for item in self.abundance_profiles if item.scale == scale), None)


_PROGRAM_FIELDS = {
    StudyField.BIOLOGICAL_SAMPLE_ID,
    StudyField.TIME,
    StudyField.PERTURBATION_ID,
    StudyField.CELL_ID,
    StudyField.RNA_COUNTS,
}
_REQUIRED_FIELDS = {
    ScientificCapability.PERTURBATION_PROGRAMS: _PROGRAM_FIELDS,
    ScientificCapability.POPULATION_TRAJECTORY: _PROGRAM_FIELDS,
    ScientificCapability.STRICT_HELDOUT_TIME: _PROGRAM_FIELDS,
    ScientificCapability.RELATIVE_ABUNDANCE: {
        StudyField.PHYSICAL_POOL_ID,
        StudyField.TIME,
        StudyField.GUIDE_ID,
        StudyField.INPUT_GUIDE_COUNT,
        StudyField.RECOVERED_GUIDE_COUNT,
    },
    ScientificCapability.ABSOLUTE_ABUNDANCE: {
        StudyField.BIOLOGICAL_SAMPLE_ID,
        StudyField.TIME,
        StudyField.PERTURBATION_ID,
        StudyField.ABSOLUTE_TOTAL_CELL_COUNT,
    },
    ScientificCapability.PHYSICAL_CONTEXT_ASSOCIATION: _PROGRAM_FIELDS
    | {StudyField.PHYSICAL_POOL_ID},
    ScientificCapability.ECOLOGICAL_COUNTERFACTUAL: _PROGRAM_FIELDS | {StudyField.PHYSICAL_POOL_ID},
    ScientificCapability.ECOLOGICAL_INTERVENTION: _PROGRAM_FIELDS
    | {StudyField.PHYSICAL_POOL_ID, StudyField.MEDIATOR_INTERVENTION},
    ScientificCapability.CLONE_LINEAGE: _PROGRAM_FIELDS | {StudyField.CLONE_ID},
    ScientificCapability.ANCESTRAL_LINEAGE: _PROGRAM_FIELDS
    | {StudyField.CLONE_ID, StudyField.LINEAGE_PARENT_ID},
    ScientificCapability.DIRECT_CELL_PATH: _PROGRAM_FIELDS | {StudyField.LIVE_CELL_TRACK_ID},
}


def _identity(model: type[StrictModel], payload: dict[str, Any], id_field: str) -> StrictModel:
    normalized: dict[str, Any] = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in model.model_fields.items()
        if name in payload
    }
    provisional = model.model_construct(**normalized)
    payload[id_field] = provisional.identity(id_field=id_field)
    return model.model_validate(payload)


def _freeze_nested(model: type[StrictModel], payload: dict[str, Any], id_field: str) -> StrictModel:
    data = dict(payload)
    data[id_field] = "pending"
    return _identity(model, data, id_field)


def freeze_study_evidence_contract(**payload: Any) -> StudyEvidenceContract:
    """Create one identity-bound study contract, including nested design gates."""

    data = dict(payload)
    nested: dict[str, tuple[type[StrictModel], str]] = {
        "protected_expression_access": (
            ProtectedExpressionAccessContract,
            "protected_access_contract_id",
        ),
        "relative_abundance_design": (RelativeAbundanceDesignContract, "relative_design_id"),
        "absolute_abundance_design": (AbsoluteAbundanceDesignContract, "absolute_design_id"),
        "physical_context_design": (PhysicalContextDesignContract, "context_design_id"),
        "clone_observation": (CloneObservationContract, "clone_observation_id"),
    }
    for field, (model, id_field) in nested.items():
        value = data.get(field)
        if isinstance(value, dict):
            data[field] = _freeze_nested(model, value, id_field)
    data["study_contract_id"] = "pending"
    return _identity(StudyEvidenceContract, data, "study_contract_id")  # type: ignore[return-value]


def _entity_resolution(
    study: StudyEvidenceContract, *, clone_qualified: bool
) -> tuple[AbundanceEntity, ...]:
    fields = set(study.observed_fields)
    entities: list[AbundanceEntity] = []
    mapping = {
        StudyField.GUIDE_ID: AbundanceEntity.GUIDE,
        StudyField.TARGET_ID: AbundanceEntity.TARGET,
        StudyField.PERTURBATION_ID: AbundanceEntity.PERTURBATION,
        StudyField.CELL_STATE: AbundanceEntity.CELL_STATE,
    }
    entities.extend(entity for field, entity in mapping.items() if field in fields)
    if clone_qualified:
        entities.append(AbundanceEntity.CLONE)
        if StudyField.CELL_STATE in fields:
            entities.append(AbundanceEntity.CLONE_STATE)
    return tuple(sorted(set(entities), key=lambda value: value.value))


def derive_dataset_capabilities(study: StudyEvidenceContract) -> DatasetCapabilityAssessment:
    """Derive structural capability without reading outcomes or fitting a model."""

    present = set(study.observed_fields)
    decisions: list[CapabilityDecision] = []
    for capability in ScientificCapability:
        required = _REQUIRED_FIELDS[capability]
        missing = required - present
        blockers: list[str] = []
        if (
            capability
            in {
                ScientificCapability.POPULATION_TRAJECTORY,
                ScientificCapability.CLONE_LINEAGE,
                ScientificCapability.ANCESTRAL_LINEAGE,
                ScientificCapability.DIRECT_CELL_PATH,
            }
            and study.entity_counts.checkpoints < 2
        ):
            blockers.append("requires at least two observed checkpoints")
        if capability == ScientificCapability.STRICT_HELDOUT_TIME:
            if study.entity_counts.checkpoints < 3:
                blockers.append("requires at least three observed checkpoints")
            if study.protected_expression_access is None:
                blockers.append("requires a protected-expression access contract")
        if capability == ScientificCapability.RELATIVE_ABUNDANCE:
            if not study.physical_pool_identity_complete:
                blockers.append("requires complete physical_pool_id coverage")
            if study.relative_abundance_design is None:
                blockers.append(
                    "requires aligned pools, guide catalog, denominator, and uncertainty"
                )
        if capability == ScientificCapability.ABSOLUTE_ABUNDANCE and (
            study.absolute_abundance_design is None
        ):
            blockers.append("requires repeated absolute-count calibration and system boundary")
        if capability in {
            ScientificCapability.PHYSICAL_CONTEXT_ASSOCIATION,
            ScientificCapability.ECOLOGICAL_COUNTERFACTUAL,
            ScientificCapability.ECOLOGICAL_INTERVENTION,
        }:
            if not study.physical_pool_identity_complete:
                blockers.append("requires complete physical_pool_id coverage")
            context = study.physical_context_design
            if context is None or not context.populations_co_resident:
                blockers.append("requires verified co-residence in one physical context")
            elif capability == ScientificCapability.ECOLOGICAL_COUNTERFACTUAL and not (
                context.qualified_context_model
            ):
                blockers.append("requires an independently qualified context model")
            elif capability == ScientificCapability.ECOLOGICAL_INTERVENTION and not (
                context.mediator_intervention_observed
            ):
                blockers.append("requires a mediator or context intervention")
        if capability in {
            ScientificCapability.CLONE_LINEAGE,
            ScientificCapability.ANCESTRAL_LINEAGE,
        }:
            clone = study.clone_observation
            if clone is None or not clone.stable_clone_qualified:
                blockers.append(
                    "requires qualified clone continuity, collision, dropout, and replication"
                )
            elif capability == ScientificCapability.ANCESTRAL_LINEAGE and not (
                clone.ancestry_qualified
            ):
                blockers.append("requires directly observed evolving-barcode ancestry")
        if capability == ScientificCapability.DIRECT_CELL_PATH and study.destructive_snapshots_only:
            blockers.append("destructive snapshots cannot validate individual cell paths")
        required_tuple = tuple(sorted(required, key=lambda value: value.value))
        missing_tuple = tuple(sorted(missing, key=lambda value: value.value))
        decisions.append(
            CapabilityDecision(
                capability=capability,
                supported=not missing_tuple and not blockers,
                required_fields=required_tuple,
                missing_fields=missing_tuple,
                blocking_reasons=tuple(sorted(set(blockers))),
            )
        )

    decision_map = {item.capability: item.supported for item in decisions}
    entities = _entity_resolution(
        study, clone_qualified=decision_map[ScientificCapability.CLONE_LINEAGE]
    )
    profiles: list[AbundanceCapabilityProfile] = []
    if decision_map[ScientificCapability.RELATIVE_ABUNDANCE]:
        profiles.append(
            AbundanceCapabilityProfile(
                scale=AbundanceScale.RELATIVE_WITHIN_POOL,
                entities=entities,
                processes=(AbundanceProcess.RELATIVE_SELECTION,),
                system_boundary="declared physical pool denominator",
            )
        )
    if decision_map[ScientificCapability.ABSOLUTE_ABUNDANCE]:
        design = study.absolute_abundance_design
        assert design is not None
        scale = (
            AbundanceScale.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT
            if design.basis == AbsoluteCountBasis.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT
            else AbundanceScale.ABSOLUTE_TISSUE
        )
        profiles.append(
            AbundanceCapabilityProfile(
                scale=scale,
                entities=entities,
                processes=(AbundanceProcess.ABSOLUTE_NET_CHANGE,),
                system_boundary=design.system_boundary,
            )
        )

    if decision_map[ScientificCapability.DIRECT_CELL_PATH]:
        lineage = LineageEvidenceLevel.L3_LIVE_OR_PAIRED_CELLS
    elif decision_map[ScientificCapability.ANCESTRAL_LINEAGE]:
        lineage = LineageEvidenceLevel.L2_HERITABLE_BARCODES
    elif decision_map[ScientificCapability.CLONE_LINEAGE]:
        lineage = LineageEvidenceLevel.L1_STABLE_CLONE_BARCODES
    else:
        lineage = LineageEvidenceLevel.L0_DESTRUCTIVE_SNAPSHOTS

    payload: dict[str, Any] = {
        "schema_id": "credo.dataset_capability_assessment",
        "schema_version": 1,
        "capability_assessment_id": "pending",
        "study_contract_id": study.study_contract_id,
        "decisions": tuple(decisions),
        "abundance_profiles": tuple(sorted(profiles, key=lambda item: item.scale.value)),
        "maximum_lineage_level": lineage,
        "structural_capability_only": True,
        "scientific_validation_passed": False,
    }
    return _identity(  # type: ignore[return-value]
        DatasetCapabilityAssessment, payload, "capability_assessment_id"
    )


def validate_dataset_capabilities(
    study: StudyEvidenceContract,
    assessment: DatasetCapabilityAssessment,
) -> None:
    """Reject a forged or stale capability assessment."""

    if assessment != derive_dataset_capabilities(study):
        raise ValueError("Dataset capability assessment differs from the observed study design.")
