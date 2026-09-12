from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.claims import (
    adjudicate_scientific_claim,
    freeze_scientific_claim_request,
)
from credo_count_sde_v4.contracts import (
    AbsoluteCountBasis,
    AbundanceEntity,
    AbundanceProcess,
    AbundanceScale,
    ArtifactRef,
    CausalStatus,
    ClaimDecision,
    DatasetCapabilityAssessment,
    EffectDirection,
    EvidenceChannel,
    EvidenceDescriptor,
    EvidenceExposureStatus,
    EvidenceIndependence,
    EvidenceLink,
    EvidenceSemanticRole,
    EvidenceTier,
    LineageEvidenceLevel,
    ScientificCapability,
    ScientificClaimKind,
    ScientificClaimRequest,
    ScientificEstimand,
    ScientificScope,
    StrictModel,
    StudyEntityCounts,
    StudyEvidenceContract,
    StudyField,
    derive_dataset_capabilities,
    freeze_study_evidence_contract,
    validate_dataset_capabilities,
)
from credo_count_sde_v4.reports import (
    DossierSection,
    DossierSectionName,
    DossierSectionStatus,
    PerturbationDossier,
    assemble_perturbation_dossier,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    provisional = model.model_construct(**payload)
    payload[id_field] = provisional.identity(id_field=id_field)
    return model.model_validate(payload)


def _artifact(character: str, name: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=f"test.{name}",
        schema_version=1,
        sha256=character * 64,
        size_bytes=1,
        media_type="application/json",
        relative_uri=f"evidence/{name}.json",
    )


def _descriptor(channel: EvidenceChannel) -> EvidenceDescriptor:
    tier = {
        EvidenceChannel.ENGINEERING_TEST: EvidenceTier.E0_ENGINEERING,
        EvidenceChannel.MODEL_FIT: EvidenceTier.E1_IN_SAMPLE,
        EvidenceChannel.HELDOUT_DONOR: EvidenceTier.E2_HELD_OUT,
        EvidenceChannel.HELDOUT_GUIDE_TARGET_SHARED: EvidenceTier.E2_HELD_OUT,
        EvidenceChannel.HELDOUT_TARGET: EvidenceTier.E2_HELD_OUT,
        EvidenceChannel.HELDOUT_TIME: EvidenceTier.E2_HELD_OUT,
        EvidenceChannel.SPATIAL_COLOCALIZATION: EvidenceTier.E2_HELD_OUT,
        EvidenceChannel.INDEPENDENT_COHORT: EvidenceTier.E3_INDEPENDENT_COHORT,
        EvidenceChannel.INTERVENTION: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
        EvidenceChannel.RESCUE: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
        EvidenceChannel.EPISTASIS: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
        EvidenceChannel.ORTHOGONAL_COMPONENT_ASSAY: (
            EvidenceTier.E4_ORTHOGONAL_INTERVENTION
        ),
        EvidenceChannel.PROLIFERATION_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
        EvidenceChannel.DEATH_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
        EvidenceChannel.MIGRATION_ASSAY: EvidenceTier.E4_ORTHOGONAL_INTERVENTION,
        EvidenceChannel.ABSOLUTE_CELL_COUNT: EvidenceTier.E5_DIRECT_MEASUREMENT,
        EvidenceChannel.STABLE_CLONE_BARCODE: EvidenceTier.E5_DIRECT_MEASUREMENT,
        EvidenceChannel.EVOLVING_BARCODE: EvidenceTier.E5_DIRECT_MEASUREMENT,
        EvidenceChannel.LIVE_CELL_TRACKING: EvidenceTier.E5_DIRECT_MEASUREMENT,
    }[channel]
    independence = {
        EvidenceChannel.HELDOUT_DONOR: EvidenceIndependence.HELDOUT_DONOR,
        EvidenceChannel.HELDOUT_GUIDE_TARGET_SHARED: (
            EvidenceIndependence.HELDOUT_GUIDE_TARGET_SHARED
        ),
        EvidenceChannel.HELDOUT_TARGET: EvidenceIndependence.HELDOUT_TARGET,
        EvidenceChannel.HELDOUT_TIME: EvidenceIndependence.HELDOUT_TIME,
        EvidenceChannel.INDEPENDENT_COHORT: EvidenceIndependence.INDEPENDENT_COHORT,
    }.get(channel, EvidenceIndependence.SAME_DATA)
    if tier == EvidenceTier.E4_ORTHOGONAL_INTERVENTION:
        independence = EvidenceIndependence.ORTHOGONAL_SAME_STUDY
    if tier == EvidenceTier.E5_DIRECT_MEASUREMENT:
        independence = EvidenceIndependence.DIRECT_OBSERVATION
    return EvidenceDescriptor(
        tier=tier,
        channel=channel,
        independence=independence,
        unit_of_replication="biological_sample",
        prospective_or_exposed_status=EvidenceExposureStatus.PROSPECTIVE,
        multiplicity_family="dev39_test",
    )


def _scope(subject: str = "TNFRSF1A perturbation") -> ScientificScope:
    return ScientificScope(
        subject=subject,
        entity_scope=(subject,),
        sample_scope=("donor_1", "donor_2"),
        time_scope=("0h", "8h"),
        population="epithelial cells",
        comparison="shared control guides",
    )


def _nested_contract_payloads() -> dict[str, dict[str, Any]]:
    return {
        "protected_expression_access": {
            "protected_checkpoint_ids": ("8h",),
            "audit_artifact": _artifact("1", "protected_access"),
        },
        "relative_abundance_design": {
            "guide_catalog_sha256": "2" * 64,
            "stable_denominator_definition": "all retained guides in each physical pool",
            "sampling_uncertainty_artifact": _artifact("3", "relative_uncertainty"),
        },
        "absolute_abundance_design": {
            "basis": AbsoluteCountBasis.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT,
            "system_boundary": "recovered sampled compartment",
            "calibration_timepoints": ("0h", "8h"),
            "capture_fraction_model": "binomial recovery model",
            "total_count_artifact": _artifact("4", "absolute_totals"),
            "uncertainty_artifact": _artifact("5", "absolute_uncertainty"),
        },
        "physical_context_design": {
            "populations_co_resident": True,
            "perturbations_cultured_separately": False,
            "qualified_context_model": True,
            "mediator_intervention_observed": True,
            "audit_artifact": _artifact("6", "context_audit"),
        },
        "clone_observation": {
            "cross_time_continuity_verified": True,
            "collision_rate": 0.01,
            "maximum_collision_rate": 0.05,
            "dropout_rate": 0.05,
            "maximum_dropout_rate": 0.10,
            "minimum_clone_support": 3,
            "assignment_confidence": 0.99,
            "minimum_assignment_confidence": 0.95,
            "biological_replicates": 3,
            "observation_model": "barcode dropout and collision model",
            "evolving_barcode_history_observed": True,
            "parent_relationship_directly_observed": True,
            "parent_tree_inferred_only": False,
            "audit_artifact": _artifact("7", "clone_audit"),
        },
    }


def _rich_study() -> StudyEvidenceContract:
    return freeze_study_evidence_contract(
        schema_id="credo.study_evidence_contract",
        schema_version=1,
        study_id="rich-physical-lineage-study",
        observed_fields=tuple(sorted(StudyField, key=lambda value: value.value)),
        entity_counts=StudyEntityCounts(
            biological_samples=8,
            physical_pools=8,
            donors=4,
            replicates=3,
            checkpoints=3,
            perturbations=10,
            guides=20,
            targets=10,
            cells=1000,
            clones=100,
        ),
        physical_pool_identity_complete=True,
        destructive_snapshots_only=False,
        **_nested_contract_payloads(),
    )


def _snapshot_study(*, checkpoints: int = 2) -> StudyEvidenceContract:
    fields = tuple(
        sorted(
            {
                StudyField.BIOLOGICAL_SAMPLE_ID,
                StudyField.TIME,
                StudyField.PERTURBATION_ID,
                StudyField.CELL_ID,
                StudyField.RNA_COUNTS,
            },
            key=lambda value: value.value,
        )
    )
    return freeze_study_evidence_contract(
        schema_id="credo.study_evidence_contract",
        schema_version=1,
        study_id="destructive-snapshots",
        observed_fields=fields,
        entity_counts=StudyEntityCounts(
            biological_samples=4,
            physical_pools=0,
            donors=0,
            replicates=0,
            checkpoints=checkpoints,
            perturbations=8,
            guides=0,
            targets=0,
            cells=500,
            clones=0,
        ),
        physical_pool_identity_complete=False,
        destructive_snapshots_only=True,
    )


def _link(
    *,
    claim_id: str,
    result_id: str,
    artifact: ArtifactRef,
    scope: ScientificScope,
    channel: EvidenceChannel,
    role: EvidenceSemanticRole,
    study_id: str = "rich-physical-lineage-study",
    evidence_study_id: str | None = None,
) -> EvidenceLink:
    return EvidenceLink(
        claim_id=claim_id,
        result_id=result_id,
        artifact_ref=artifact,
        semantic_role=role,
        study_id=study_id,
        evidence_study_id=evidence_study_id or study_id,
        scope=scope,
        split_id="outer-split" if _descriptor(channel).tier == EvidenceTier.E2_HELD_OUT else None,
        metric_or_statistic_id="heldout_nb_log_likelihood",
        descriptor=_descriptor(channel),
    )


def _claim(
    *,
    claim_id: str,
    kind: ScientificClaimKind,
    result_id: str,
    artifact: ArtifactRef,
    channel: EvidenceChannel,
    role: EvidenceSemanticRole,
    scope: ScientificScope | None = None,
    requested_free_text: str | None = None,
    **estimand_updates: Any,
) -> ScientificClaimRequest:
    claim_scope = scope or _scope()
    estimand_payload: dict[str, Any] = {
        "scope": claim_scope,
        "outcome": "count-linked transcriptional program",
        "estimand": "held-out mean log-likelihood contrast",
        "direction": EffectDirection.INCREASE,
        "effect_measure": "log likelihood per observed count",
        "causal_status": CausalStatus.PREDICTIVE_ASSOCIATION,
        "uncertainty_result_id": result_id,
        **estimand_updates,
    }
    return freeze_scientific_claim_request(
        schema_id="credo.scientific_claim_request",
        schema_version=1,
        claim_id=claim_id,
        kind=kind,
        estimand=ScientificEstimand(**estimand_payload),
        evidence_links=(
            _link(
                claim_id=claim_id,
                result_id=result_id,
                artifact=artifact,
                scope=claim_scope,
                channel=channel,
                role=role,
            ),
        ),
        requested_free_text=requested_free_text,
    )


def _sections(
    *, result_id: str | None = None, artifact: ArtifactRef | None = None
) -> tuple[DossierSection, ...]:
    return tuple(
        DossierSection(
            section=section,
            status=(
                DossierSectionStatus.AVAILABLE
                if section == DossierSectionName.BIOLOGICAL_PROGRAMS and result_id is not None
                else DossierSectionStatus.NOT_RUN
            ),
            result_id=(result_id if section == DossierSectionName.BIOLOGICAL_PROGRAMS else None),
            result_scope=(_scope() if section == DossierSectionName.BIOLOGICAL_PROGRAMS else None),
            evidence=(
                (_descriptor(EvidenceChannel.HELDOUT_TARGET),)
                if section == DossierSectionName.BIOLOGICAL_PROGRAMS
                and result_id is not None
                else ()
            ),
            artifact=(artifact if section == DossierSectionName.BIOLOGICAL_PROGRAMS else None),
            interpretation="Available qualified result." if result_id else "Not run.",
        )
        for section in DossierSectionName
    )


def test_positive_typed_claim_builds_scope_exact_dossier(tmp_path: Path) -> None:
    study = _rich_study()
    capabilities = derive_dataset_capabilities(study)
    assert all(item.supported for item in capabilities.decisions)
    assert capabilities.maximum_lineage_level == LineageEvidenceLevel.L3_LIVE_OR_PAIRED_CELLS
    artifact = _artifact("8", "program")
    request = _claim(
        claim_id="program",
        kind=ScientificClaimKind.PERTURBATION_PROGRAM_ASSOCIATION,
        result_id="program-result",
        artifact=artifact,
        channel=EvidenceChannel.HELDOUT_TARGET,
        role=EvidenceSemanticRole.PROGRAM_COUNT_PREDICTION,
    )
    dossier = assemble_perturbation_dossier(
        study=study,
        perturbation_id="TNFRSF1A_KO",
        guide_ids=("guide_b", "guide_a"),
        target_id="TNFRSF1A",
        sections=_sections(result_id="program-result", artifact=artifact),
        claim_requests=(request,),
    )
    assert isinstance(dossier, PerturbationDossier)
    assert dossier.claim_adjudications[0].decision == ClaimDecision.PERMITTED
    assert dossier.claim_adjudications[0].permitted_wording is not None
    assert dossier.guide_ids == ("guide_a", "guide_b")
    for name, contract, contract_type in (
        ("study", study, "StudyEvidenceContract"),
        ("capabilities", capabilities, "DatasetCapabilityAssessment"),
        ("claim", request, "ScientificClaimRequest"),
        ("adjudication", dossier.claim_adjudications[0], "ClaimAdjudication"),
        ("dossier", dossier, "PerturbationDossier"),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(contract.model_dump_json() + "\n")
        assert validate_contract(path)["contract_type"] == contract_type


def test_abundance_scale_entity_and_process_are_orthogonal() -> None:
    capabilities = derive_dataset_capabilities(_rich_study())
    relative = capabilities.abundance_profile(AbundanceScale.RELATIVE_WITHIN_POOL)
    absolute = capabilities.abundance_profile(
        AbundanceScale.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT
    )
    assert relative is not None and absolute is not None
    assert AbundanceEntity.CLONE in relative.entities
    assert AbundanceEntity.CLONE in absolute.entities
    assert relative.processes == (AbundanceProcess.RELATIVE_SELECTION,)
    assert absolute.processes == (AbundanceProcess.ABSOLUTE_NET_CHANGE,)
    assert absolute.system_boundary == "recovered sampled compartment"


def test_snapshot_and_three_time_without_access_contract_fail_closed() -> None:
    two = derive_dataset_capabilities(_snapshot_study())
    three = derive_dataset_capabilities(_snapshot_study(checkpoints=3))
    for capabilities in (two, three):
        assert capabilities.supports(ScientificCapability.PERTURBATION_PROGRAMS)
        assert capabilities.supports(ScientificCapability.POPULATION_TRAJECTORY)
        assert not capabilities.supports(ScientificCapability.STRICT_HELDOUT_TIME)
        assert not capabilities.supports(ScientificCapability.RELATIVE_ABUNDANCE)
        assert not capabilities.supports(ScientificCapability.ABSOLUTE_ABUNDANCE)
        assert not capabilities.supports(ScientificCapability.CLONE_LINEAGE)


def test_barcode_channel_cannot_authorize_proliferation() -> None:
    study = _rich_study()
    artifact = _artifact("9", "barcode")
    request = _claim(
        claim_id="bad-proliferation",
        kind=ScientificClaimKind.PROLIFERATION_COMPONENT,
        result_id="abundance-result",
        artifact=artifact,
        channel=EvidenceChannel.STABLE_CLONE_BARCODE,
        role=EvidenceSemanticRole.PROLIFERATION_COMPONENT,
        outcome="proliferation",
        estimand="proliferation component",
        abundance_scale=AbundanceScale.CAPTURE_CALIBRATED_SAMPLED_COMPARTMENT,
        abundance_entity=AbundanceEntity.CLONE,
        abundance_process=AbundanceProcess.PROLIFERATION,
        system_boundary="recovered sampled compartment",
    )
    decision = adjudicate_scientific_claim(
        study, derive_dataset_capabilities(study), request
    )
    assert decision.decision == ClaimDecision.BLOCKED
    assert any("component-specific assay" in reason for reason in decision.blocking_reasons)


def test_absolute_counts_at_one_timepoint_are_rejected() -> None:
    payload = _nested_contract_payloads()["absolute_abundance_design"]
    payload["calibration_timepoints"] = ("8h",)
    with pytest.raises(ValidationError, match="repeated calibrated timepoints"):
        freeze_study_evidence_contract(
            schema_id="credo.study_evidence_contract",
            schema_version=1,
            study_id="bad-absolute",
            observed_fields=tuple(sorted(StudyField, key=lambda value: value.value)),
            entity_counts=_rich_study().entity_counts,
            physical_pool_identity_complete=True,
            destructive_snapshots_only=False,
            absolute_abundance_design=payload,
        )


def test_evidence_hash_with_unrelated_semantic_role_is_blocked() -> None:
    study = _rich_study()
    request = _claim(
        claim_id="wrong-role",
        kind=ScientificClaimKind.RELATIVE_SELECTION,
        result_id="relative-result",
        artifact=_artifact("a", "relative"),
        channel=EvidenceChannel.HELDOUT_TARGET,
        role=EvidenceSemanticRole.GENE_EFFECT,
        outcome="relative guide representation",
        estimand="centered relative selection",
        abundance_scale=AbundanceScale.RELATIVE_WITHIN_POOL,
        abundance_entity=AbundanceEntity.GUIDE,
        abundance_process=AbundanceProcess.RELATIVE_SELECTION,
        system_boundary="declared physical pool denominator",
    )
    decision = adjudicate_scientific_claim(
        study, derive_dataset_capabilities(study), request
    )
    assert decision.decision == ClaimDecision.BLOCKED
    assert any("semantic role" in reason for reason in decision.blocking_reasons)


def test_same_study_cannot_be_relabelled_independent() -> None:
    with pytest.raises(ValidationError, match="Independent-cohort"):
        _link(
            claim_id="same-cohort",
            result_id="result",
            artifact=_artifact("b", "same"),
            scope=_scope(),
            channel=EvidenceChannel.INDEPENDENT_COHORT,
            role=EvidenceSemanticRole.EXTERNAL_CONCORDANCE,
        )


def test_separate_culture_and_high_collision_disable_specific_capabilities() -> None:
    study_payload = _rich_study().model_dump(mode="python")
    study_payload.pop("study_contract_id")
    context = study_payload["physical_context_design"]
    context["populations_co_resident"] = False
    context["perturbations_cultured_separately"] = True
    context["qualified_context_model"] = False
    context["mediator_intervention_observed"] = False
    clone = study_payload["clone_observation"]
    clone["collision_rate"] = 0.20
    altered = freeze_study_evidence_contract(**study_payload)
    capabilities = derive_dataset_capabilities(altered)
    assert not capabilities.supports(ScientificCapability.PHYSICAL_CONTEXT_ASSOCIATION)
    assert not capabilities.supports(ScientificCapability.ECOLOGICAL_COUNTERFACTUAL)
    assert not capabilities.supports(ScientificCapability.ECOLOGICAL_INTERVENTION)
    assert not capabilities.supports(ScientificCapability.CLONE_LINEAGE)
    assert capabilities.supports(ScientificCapability.PERTURBATION_PROGRAMS)


def test_requested_unicode_prose_is_never_the_permitted_wording() -> None:
    study = _rich_study()
    request = _claim(
        claim_id="unicode-prose",
        kind=ScientificClaimKind.PERTURBATION_PROGRAM_ASSOCIATION,
        result_id="program-result",
        artifact=_artifact("c", "unicode"),
        channel=EvidenceChannel.HELDOUT_TARGET,
        role=EvidenceSemanticRole.PROGRAM_COUNT_PREDICTION,
        requested_free_text="This саuses prоliferation and mediates lineage.",
    )
    decision = adjudicate_scientific_claim(
        study, derive_dataset_capabilities(study), request
    )
    assert decision.decision == ClaimDecision.PERMITTED
    assert decision.permitted_wording != request.requested_free_text
    assert decision.permitted_wording is not None
    assert decision.permitted_wording.startswith("Predictive association:")


def test_unavailable_section_cannot_carry_evidence() -> None:
    with pytest.raises(ValidationError, match="Only available"):
        DossierSection(
            section=DossierSectionName.BIOLOGICAL_PROGRAMS,
            status=DossierSectionStatus.NOT_RUN,
            evidence=(_descriptor(EvidenceChannel.MODEL_FIT),),
            interpretation="Not run but incorrectly carrying evidence.",
        )


def test_dossier_rejects_hash_present_under_wrong_result_and_scope() -> None:
    study = _rich_study()
    artifact = _artifact("d", "wrong_result")
    request = _claim(
        claim_id="crosswired",
        kind=ScientificClaimKind.PERTURBATION_PROGRAM_ASSOCIATION,
        result_id="different-result",
        artifact=artifact,
        channel=EvidenceChannel.HELDOUT_TARGET,
        role=EvidenceSemanticRole.PROGRAM_COUNT_PREDICTION,
    )
    with pytest.raises(ValidationError, match="outside the dossier"):
        assemble_perturbation_dossier(
            study=study,
            perturbation_id="perturbation",
            sections=_sections(result_id="program-result", artifact=artifact),
            claim_requests=(request,),
        )


def test_capability_rederivation_and_identity_reject_mutation() -> None:
    study = _rich_study()
    assessment = derive_dataset_capabilities(study)
    forged = assessment.model_copy(
        update={
            "decisions": tuple(
                item.model_copy(update={"supported": False, "blocking_reasons": ("forged",)})
                if item.capability == ScientificCapability.PERTURBATION_PROGRAMS
                else item
                for item in assessment.decisions
            )
        }
    )
    with pytest.raises(ValueError, match="differs from the observed study design"):
        validate_dataset_capabilities(study, forged)
    payload = assessment.model_dump(mode="json")
    payload["scientific_validation_passed"] = True
    with pytest.raises(ValidationError):
        DatasetCapabilityAssessment.model_validate(payload)


@pytest.mark.parametrize(
    "removed,capability",
    [
        ("protected_expression_access", ScientificCapability.STRICT_HELDOUT_TIME),
        ("relative_abundance_design", ScientificCapability.RELATIVE_ABUNDANCE),
        ("absolute_abundance_design", ScientificCapability.ABSOLUTE_ABUNDANCE),
        ("physical_context_design", ScientificCapability.PHYSICAL_CONTEXT_ASSOCIATION),
        ("clone_observation", ScientificCapability.CLONE_LINEAGE),
    ],
)
def test_design_gate_ablation_disables_only_dependent_layer(
    removed: str, capability: ScientificCapability
) -> None:
    payload = _rich_study().model_dump(mode="python")
    payload.pop("study_contract_id")
    payload[removed] = None
    ablated = freeze_study_evidence_contract(**payload)
    capabilities = derive_dataset_capabilities(ablated)
    assert not capabilities.supports(capability)
    assert capabilities.supports(ScientificCapability.PERTURBATION_PROGRAMS)
