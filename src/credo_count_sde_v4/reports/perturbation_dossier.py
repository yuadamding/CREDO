"""Self-contained perturbation dossier with exact semantic evidence linkage."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from ..claims import adjudicate_scientific_claim
from ..contracts import (
    LINEAGE_LEVEL_RANK,
    LINEAGE_RESULT_MINIMUM_LEVEL,
    AbundanceResult,
    ArtifactRef,
    BiologicalProgramQualificationBundle,
    ClaimAdjudication,
    DatasetCapabilityAssessment,
    EvidenceDescriptor,
    EvidenceTier,
    ProgramQualificationStatus,
    ScientificCapability,
    ScientificClaimRequest,
    ScientificScope,
    StrictModel,
    StudyEvidenceContract,
    TrajectoryResult,
    derive_dataset_capabilities,
    validate_dataset_capabilities,
)


class DossierSectionName(StrEnum):
    """Frozen user-facing sections in every perturbation dossier."""

    GUIDE_TARGET_QC = "guide_target_qc"
    BIOLOGICAL_PROGRAMS = "biological_programs"
    GENE_EFFECT_DECOMPOSITION = "gene_effect_decomposition"
    GENERATOR_ATTRIBUTION = "generator_attribution"
    CONTEXT_EFFECT = "context_effect"
    UNCERTAINTY = "uncertainty"
    EXTERNAL_VALIDATION = "external_validation"


class DossierSectionStatus(StrEnum):
    """Availability is orthogonal to evidence strength."""

    AVAILABLE = "available"
    STRUCTURALLY_UNAVAILABLE = "structurally_unavailable"
    NOT_RUN = "not_run"
    FAILED_VALIDATION = "failed_validation"
    NOT_APPLICABLE = "not_applicable"
    WITHHELD_BY_CLAIM_POLICY = "withheld_by_claim_policy"


class DossierSection(StrictModel):
    """One result binding or an explicit non-result state."""

    section: DossierSectionName
    status: DossierSectionStatus
    result_id: str | None = None
    result_scope: ScientificScope | None = None
    evidence: tuple[EvidenceDescriptor, ...] = ()
    artifact: ArtifactRef | None = None
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_section(self) -> DossierSection:
        available = self.status == DossierSectionStatus.AVAILABLE
        result_fields = (self.result_id, self.result_scope, self.artifact)
        if available != all(item is not None for item in result_fields):
            raise ValueError("Available sections require result ID, scope, and artifact.")
        if available != bool(self.evidence):
            raise ValueError("Only available sections may carry scientific evidence.")
        if not available and any(item is not None for item in result_fields):
            raise ValueError("Unavailable sections cannot carry result bindings.")
        if len(self.evidence) != len(set(self.evidence)):
            raise ValueError("Section evidence descriptors must be unique.")
        return self


class PerturbationDossier(StrictModel):
    """Primary biological result object with embedded capability evidence."""

    schema_id: Literal["credo.perturbation_dossier"] = "credo.perturbation_dossier"
    schema_version: Literal[1] = 1
    dossier_id: str
    study: StudyEvidenceContract
    capabilities: DatasetCapabilityAssessment
    perturbation_id: str = Field(min_length=1)
    guide_ids: tuple[str, ...] = ()
    target_id: str | None = None
    sections: tuple[DossierSection, ...]
    trajectory: TrajectoryResult | None = None
    abundance: AbundanceResult | None = None
    programs: BiologicalProgramQualificationBundle | None = None
    claim_requests: tuple[ScientificClaimRequest, ...]
    claim_adjudications: tuple[ClaimAdjudication, ...]

    @model_validator(mode="after")
    def validate_dossier(self) -> PerturbationDossier:
        validate_dataset_capabilities(self.study, self.capabilities)
        if self.guide_ids != tuple(sorted(set(self.guide_ids))):
            raise ValueError("Dossier guide IDs must be unique and sorted.")
        if tuple(item.section for item in self.sections) != tuple(DossierSectionName):
            raise ValueError("Dossier must contain every section in frozen order.")
        section_map = {item.section: item for item in self.sections}
        for section in (
            DossierSectionName.BIOLOGICAL_PROGRAMS,
            DossierSectionName.GENE_EFFECT_DECOMPOSITION,
        ):
            if (
                section_map[section].status == DossierSectionStatus.AVAILABLE
                and not self.capabilities.supports(ScientificCapability.PERTURBATION_PROGRAMS)
            ):
                raise ValueError(f"{section.value} exceeds the study capability contract.")
        if (
            section_map[DossierSectionName.GENERATOR_ATTRIBUTION].status
            == DossierSectionStatus.AVAILABLE
            and not self.capabilities.supports(ScientificCapability.POPULATION_TRAJECTORY)
        ):
            raise ValueError("Generator attribution requires population-trajectory capability.")
        if (
            section_map[DossierSectionName.CONTEXT_EFFECT].status
            == DossierSectionStatus.AVAILABLE
            and not self.capabilities.supports(
                ScientificCapability.PHYSICAL_CONTEXT_ASSOCIATION
            )
        ):
            raise ValueError("Context effects require verified physical co-residence.")
        external = section_map[DossierSectionName.EXTERNAL_VALIDATION]
        if external.status == DossierSectionStatus.AVAILABLE and not any(
            item.tier
            in {
                EvidenceTier.E3_INDEPENDENT_COHORT,
                EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
                EvidenceTier.E5_DIRECT_MEASUREMENT,
            }
            for item in external.evidence
        ):
            raise ValueError("External validation requires an external or orthogonal channel.")

        if self.trajectory is not None:
            if not self.capabilities.supports(ScientificCapability.POPULATION_TRAJECTORY):
                raise ValueError("Trajectory result exceeds the study capability contract.")
            if self.trajectory.study_id != self.study.study_id:
                raise ValueError("Trajectory result references another study.")
            if self.trajectory.capability_assessment_id != (
                self.capabilities.capability_assessment_id
            ):
                raise ValueError("Trajectory result references another capability assessment.")
            if LINEAGE_LEVEL_RANK[self.trajectory.lineage_level] > LINEAGE_LEVEL_RANK[
                self.capabilities.maximum_lineage_level
            ]:
                raise ValueError("Trajectory lineage level exceeds the study evidence.")
        if self.abundance is not None:
            if self.abundance.study_id != self.study.study_id:
                raise ValueError("Abundance result references another study.")
            if self.abundance.capability_assessment_id != (
                self.capabilities.capability_assessment_id
            ):
                raise ValueError("Abundance result references another capability assessment.")
            profile = self.capabilities.abundance_profile(self.abundance.scale)
            if profile is None or self.abundance.entity not in profile.entities:
                raise ValueError("Abundance result uses a scale/entity unsupported by the study.")
            if self.abundance.system_boundary != profile.system_boundary:
                raise ValueError("Abundance result exceeds the calibrated system boundary.")
        if self.programs is not None:
            if not self.capabilities.supports(ScientificCapability.PERTURBATION_PROGRAMS):
                raise ValueError("Program result exceeds the study capability contract.")
            if self.programs.study_id != self.study.study_id:
                raise ValueError("Program result references another study.")
            if self.programs.capability_assessment_id != (
                self.capabilities.capability_assessment_id
            ):
                raise ValueError("Program result references another capability assessment.")
            if self.programs.qualification.status != ProgramQualificationStatus.PASS_SCIENTIFIC:
                for section_name in (
                    DossierSectionName.BIOLOGICAL_PROGRAMS,
                    DossierSectionName.GENE_EFFECT_DECOMPOSITION,
                ):
                    if section_map[section_name].status == DossierSectionStatus.AVAILABLE:
                        raise ValueError("Failed program qualification cannot populate a dossier.")

        request_ids = tuple(item.claim_request_id for item in self.claim_requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Dossier claim requests must be unique.")
        for request in self.claim_requests:
            semantics = request.estimand.trajectory_semantics
            if semantics is not None and (
                self.trajectory is None
                or LINEAGE_LEVEL_RANK[LINEAGE_RESULT_MINIMUM_LEVEL[semantics]]
                > LINEAGE_LEVEL_RANK[
                    LINEAGE_RESULT_MINIMUM_LEVEL[self.trajectory.semantics]
                ]
            ):
                raise ValueError("Claim trajectory semantics lack a matching dossier result.")
            if request.estimand.abundance_scale is not None and (
                self.abundance is None
                or request.estimand.abundance_scale != self.abundance.scale
                or request.estimand.abundance_entity != self.abundance.entity
                or request.estimand.abundance_process
                not in {item.process for item in self.abundance.components}
                or request.estimand.system_boundary != self.abundance.system_boundary
            ):
                raise ValueError("Claim abundance axes lack a matching dossier result.")
        expected_adjudications = tuple(
            adjudicate_scientific_claim(self.study, self.capabilities, request)
            for request in self.claim_requests
        )
        if self.claim_adjudications != expected_adjudications:
            raise ValueError("Dossier claim adjudications differ from fail-closed recomputation.")

        # Every link must resolve to the exact result, scope, and artifact bytes.
        results: dict[str, tuple[ScientificScope, set[str]]] = {}
        for section_record in self.sections:
            if section_record.status == DossierSectionStatus.AVAILABLE:
                assert section_record.result_id is not None
                assert section_record.result_scope is not None
                assert section_record.artifact is not None
                results[section_record.result_id] = (
                    section_record.result_scope,
                    {section_record.artifact.sha256},
                )
        if self.trajectory is not None:
            artifacts = {
                self.trajectory.predicted_finite_measures.sha256,
                self.trajectory.transition_kernels.sha256,
                self.trajectory.uncertainty.sha256,
            }
            artifacts |= {
                artifact.sha256
                for artifact in (
                    self.trajectory.terminal_fate_probabilities,
                    self.trajectory.heldout_time_evaluation,
                    self.trajectory.semigroup_evaluation,
                    self.trajectory.lineage_evaluation,
                )
                if artifact is not None
            }
            results[self.trajectory.trajectory_result_id] = (self.trajectory.scope, artifacts)
        if self.abundance is not None:
            artifacts = {item.artifact.sha256 for item in self.abundance.components}
            if self.abundance.calibration is not None:
                artifacts |= {
                    self.abundance.calibration.total_count_artifact.sha256,
                    self.abundance.calibration.uncertainty_artifact.sha256,
                }
            results[self.abundance.abundance_result_id] = (self.abundance.scope, artifacts)
        if self.programs is not None:
            bundle_artifacts = {
                self.programs.model_state_artifact.sha256,
                self.programs.qualification.metrics_artifact.sha256,
                self.programs.qualification.training_log_artifact.sha256,
                self.programs.qualification.software_environment_artifact.sha256,
                self.programs.guide_target_consistency.per_target_artifact.sha256,
                self.programs.uncertainty.seed_loading_stability_artifact.sha256,
                self.programs.uncertainty.inclusion_probability_artifact.sha256,
                self.programs.uncertainty.null_inclusion_artifact.sha256,
                *(item.loading_artifact.sha256 for item in self.programs.program_definitions),
            }
            results[self.programs.program_bundle_id] = (self.programs.scope, bundle_artifacts)
            results[self.programs.uncertainty.uncertainty_id] = (
                self.programs.scope,
                {
                    self.programs.uncertainty.seed_loading_stability_artifact.sha256,
                    self.programs.uncertainty.inclusion_probability_artifact.sha256,
                    self.programs.uncertainty.null_inclusion_artifact.sha256,
                },
            )
            results[self.programs.guide_target_consistency.consistency_id] = (
                self.programs.scope,
                {self.programs.guide_target_consistency.per_target_artifact.sha256},
            )
            for effect in self.programs.perturbation_effects:
                results[effect.program_effect_id] = (
                    effect.scope,
                    {
                        effect.reference_activity.sha256,
                        effect.target_activity.sha256,
                        effect.guide_deviation_activity.sha256,
                        effect.guide_efficiency.sha256,
                    },
                )
            for gene_effect in self.programs.gene_effects:
                results[gene_effect.gene_effect_id] = (
                    gene_effect.scope,
                    {
                        gene_effect.log_fold_change_artifact.sha256,
                        gene_effect.standard_error_artifact.sha256,
                        gene_effect.sign_probability_artifact.sha256,
                    },
                )
        for request in self.claim_requests:
            for link in request.evidence_links:
                result = results.get(link.result_id)
                if result is None:
                    raise ValueError("Claim evidence link references a result outside the dossier.")
                scope, hashes = result
                if link.scope != scope or link.scope != request.estimand.scope:
                    raise ValueError("Claim, result, and evidence-link scopes differ.")
                if link.artifact_ref.sha256 not in hashes:
                    raise ValueError("Claim evidence artifact is unrelated to its linked result.")

        expected = self.identity(id_field="dossier_id")
        if self.dossier_id != expected:
            raise ValueError(f"dossier_id mismatch: expected {expected}.")
        return self


def assemble_perturbation_dossier(
    *,
    study: StudyEvidenceContract,
    perturbation_id: str,
    sections: tuple[DossierSection, ...],
    claim_requests: tuple[ScientificClaimRequest, ...],
    guide_ids: tuple[str, ...] = (),
    target_id: str | None = None,
    trajectory: TrajectoryResult | None = None,
    abundance: AbundanceResult | None = None,
    programs: BiologicalProgramQualificationBundle | None = None,
) -> PerturbationDossier:
    """Assemble a dossier and rederive every capability and claim decision."""

    capabilities = derive_dataset_capabilities(study)
    adjudications = tuple(
        adjudicate_scientific_claim(study, capabilities, request)
        for request in claim_requests
    )
    payload: dict[str, Any] = {
        "schema_id": "credo.perturbation_dossier",
        "schema_version": 1,
        "dossier_id": "pending",
        "study": study,
        "capabilities": capabilities,
        "perturbation_id": perturbation_id,
        "guide_ids": tuple(sorted(set(guide_ids))),
        "target_id": target_id,
        "sections": sections,
        "trajectory": trajectory,
        "abundance": abundance,
        "programs": programs,
        "claim_requests": claim_requests,
        "claim_adjudications": adjudications,
    }
    provisional = PerturbationDossier.model_construct(**payload)
    payload["dossier_id"] = provisional.identity(id_field="dossier_id")
    return PerturbationDossier.model_validate(payload)
