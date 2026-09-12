"""Conjunctive, typed scientific-claim adjudication and wording rendering."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    AbundanceProcess,
    CausalStatus,
    ClaimAdjudication,
    ClaimDecision,
    DatasetCapabilityAssessment,
    EffectDirection,
    EvidenceChannel,
    EvidenceSemanticRole,
    LineageResultSemantics,
    ScientificCapability,
    ScientificClaimKind,
    ScientificClaimRequest,
    StrictModel,
    StudyEvidenceContract,
    StudyField,
    validate_dataset_capabilities,
)

_CAPABILITY = {
    ScientificClaimKind.PERTURBATION_PROGRAM_ASSOCIATION: (
        ScientificCapability.PERTURBATION_PROGRAMS
    ),
    ScientificClaimKind.POPULATION_TRANSITION: ScientificCapability.POPULATION_TRAJECTORY,
    ScientificClaimKind.RELATIVE_SELECTION: ScientificCapability.RELATIVE_ABUNDANCE,
    ScientificClaimKind.ABSOLUTE_NET_CHANGE: ScientificCapability.ABSOLUTE_ABUNDANCE,
    ScientificClaimKind.PROLIFERATION_COMPONENT: ScientificCapability.ABSOLUTE_ABUNDANCE,
    ScientificClaimKind.DEATH_COMPONENT: ScientificCapability.ABSOLUTE_ABUNDANCE,
    ScientificClaimKind.MIGRATION_OR_COMPARTMENT_LOSS: (ScientificCapability.ABSOLUTE_ABUNDANCE),
    ScientificClaimKind.DIFFUSION_ATTRIBUTION: ScientificCapability.POPULATION_TRAJECTORY,
    ScientificClaimKind.BIOLOGICAL_PLASTICITY: ScientificCapability.POPULATION_TRAJECTORY,
    ScientificClaimKind.PHYSICAL_CONTEXT_ASSOCIATION: (
        ScientificCapability.PHYSICAL_CONTEXT_ASSOCIATION
    ),
    ScientificClaimKind.ECOLOGICAL_COUNTERFACTUAL: (ScientificCapability.ECOLOGICAL_COUNTERFACTUAL),
    ScientificClaimKind.ECOLOGICAL_MECHANISM: ScientificCapability.ECOLOGICAL_INTERVENTION,
    ScientificClaimKind.CLONE_FATE: ScientificCapability.CLONE_LINEAGE,
    ScientificClaimKind.ANCESTRAL_LINEAGE: ScientificCapability.ANCESTRAL_LINEAGE,
    ScientificClaimKind.DIRECT_CELL_PATH: ScientificCapability.DIRECT_CELL_PATH,
    ScientificClaimKind.CAUSAL_MECHANISM: ScientificCapability.PERTURBATION_PROGRAMS,
}

_ROLE = {
    ScientificClaimKind.PERTURBATION_PROGRAM_ASSOCIATION: {
        EvidenceSemanticRole.PROGRAM_COUNT_PREDICTION,
        EvidenceSemanticRole.PROGRAM_STABILITY,
        EvidenceSemanticRole.GUIDE_TARGET_CONSISTENCY,
        EvidenceSemanticRole.GENE_EFFECT,
    },
    ScientificClaimKind.POPULATION_TRANSITION: {EvidenceSemanticRole.POPULATION_TRANSITION},
    ScientificClaimKind.RELATIVE_SELECTION: {EvidenceSemanticRole.RELATIVE_ABUNDANCE},
    ScientificClaimKind.ABSOLUTE_NET_CHANGE: {EvidenceSemanticRole.ABSOLUTE_NET_CHANGE},
    ScientificClaimKind.PROLIFERATION_COMPONENT: {EvidenceSemanticRole.PROLIFERATION_COMPONENT},
    ScientificClaimKind.DEATH_COMPONENT: {EvidenceSemanticRole.DEATH_COMPONENT},
    ScientificClaimKind.MIGRATION_OR_COMPARTMENT_LOSS: {
        EvidenceSemanticRole.MIGRATION_OR_LOSS_COMPONENT
    },
    ScientificClaimKind.DIFFUSION_ATTRIBUTION: {EvidenceSemanticRole.GENERATOR_ATTRIBUTION},
    ScientificClaimKind.BIOLOGICAL_PLASTICITY: {EvidenceSemanticRole.GENERATOR_ATTRIBUTION},
    ScientificClaimKind.PHYSICAL_CONTEXT_ASSOCIATION: {EvidenceSemanticRole.CONTEXT_ASSOCIATION},
    ScientificClaimKind.ECOLOGICAL_COUNTERFACTUAL: {EvidenceSemanticRole.CONTEXT_ASSOCIATION},
    ScientificClaimKind.ECOLOGICAL_MECHANISM: {EvidenceSemanticRole.ECOLOGICAL_INTERVENTION},
    ScientificClaimKind.CLONE_FATE: {EvidenceSemanticRole.CLONE_FATE},
    ScientificClaimKind.ANCESTRAL_LINEAGE: {EvidenceSemanticRole.ANCESTRAL_LINEAGE},
    ScientificClaimKind.DIRECT_CELL_PATH: {EvidenceSemanticRole.DIRECT_CELL_PATH},
    ScientificClaimKind.CAUSAL_MECHANISM: {
        EvidenceSemanticRole.EXTERNAL_CONCORDANCE,
        EvidenceSemanticRole.ECOLOGICAL_INTERVENTION,
    },
}

_MODEL_CHANNELS = {
    EvidenceChannel.MODEL_FIT,
    EvidenceChannel.HELDOUT_DONOR,
    EvidenceChannel.HELDOUT_GUIDE_TARGET_SHARED,
    EvidenceChannel.HELDOUT_TARGET,
    EvidenceChannel.HELDOUT_TIME,
    EvidenceChannel.INDEPENDENT_COHORT,
}
_HELDOUT_CHANNELS = {
    EvidenceChannel.HELDOUT_DONOR,
    EvidenceChannel.HELDOUT_GUIDE_TARGET_SHARED,
    EvidenceChannel.HELDOUT_TARGET,
    EvidenceChannel.HELDOUT_TIME,
    EvidenceChannel.INDEPENDENT_COHORT,
}
_INTERVENTION_CHANNELS = {
    EvidenceChannel.INTERVENTION,
    EvidenceChannel.RESCUE,
    EvidenceChannel.EPISTASIS,
}


def _identity(model: type[StrictModel], payload: dict[str, Any], id_field: str) -> StrictModel:
    provisional = model.model_construct(**payload)
    payload[id_field] = provisional.identity(id_field=id_field)
    return model.model_validate(payload)


def freeze_scientific_claim_request(**payload: Any) -> ScientificClaimRequest:
    """Create one identity-bound atomic claim from typed estimand and links."""

    data = dict(payload)
    data["claim_request_id"] = "pending"
    return _identity(  # type: ignore[return-value]
        ScientificClaimRequest, data, "claim_request_id"
    )


def _require_channels(
    reasons: list[str],
    channels: set[EvidenceChannel],
    required: set[EvidenceChannel],
    message: str,
) -> None:
    if not required <= channels:
        reasons.append(message)


def _render(request: ScientificClaimRequest) -> str:
    """Render controlled wording solely from the adjudicated atomic estimand."""

    estimand = request.estimand
    direction = {
        EffectDirection.INCREASE: "an increase",
        EffectDirection.DECREASE: "a decrease",
        EffectDirection.MIXED: "a mixed-direction effect",
        EffectDirection.NULL_COMPATIBLE: "an effect compatible with the declared null",
    }[estimand.direction]
    times = ", ".join(estimand.scope.time_scope)
    base = (
        f"For {estimand.scope.subject} in {estimand.scope.population}, the "
        f"{estimand.estimand} for {estimand.outcome} showed {direction} relative to "
        f"{estimand.scope.comparison} over {times}, measured as {estimand.effect_measure}."
    )
    if estimand.causal_status == CausalStatus.MODEL_ATTRIBUTION:
        return "Model attribution only: " + base
    if estimand.causal_status == CausalStatus.CAUSAL_UNDER_INTERVENTION:
        return "Under the declared intervention and system boundary: " + base
    return "Predictive association: " + base


def adjudicate_scientific_claim(
    study: StudyEvidenceContract,
    assessment: DatasetCapabilityAssessment,
    request: ScientificClaimRequest,
) -> ClaimAdjudication:
    """Apply capability, semantic-role, channel-conjunction, and scope gates."""

    validate_dataset_capabilities(study, assessment)
    reasons: list[str] = []
    capability = _CAPABILITY[request.kind]
    if not assessment.supports(capability):
        reasons.append(f"dataset lacks {capability.value}")
    if any(link.study_id != study.study_id for link in request.evidence_links):
        reasons.append("evidence link references another study")
    if request.estimand.uncertainty_result_id not in {
        link.result_id for link in request.evidence_links
    }:
        reasons.append("uncertainty result is not linked to the atomic claim")

    channels = set(request.evidence_channels)
    roles = {link.semantic_role for link in request.evidence_links}
    if not roles <= _ROLE[request.kind]:
        reasons.append("evidence semantic role is incompatible with the claim kind")

    if (
        request.kind
        in {
            ScientificClaimKind.PERTURBATION_PROGRAM_ASSOCIATION,
            ScientificClaimKind.POPULATION_TRANSITION,
            ScientificClaimKind.RELATIVE_SELECTION,
            ScientificClaimKind.DIFFUSION_ATTRIBUTION,
        }
        and not channels & _MODEL_CHANNELS
    ):
        reasons.append("claim requires model-fit or predictive evidence")
    if request.kind in {
        ScientificClaimKind.PHYSICAL_CONTEXT_ASSOCIATION,
        ScientificClaimKind.ECOLOGICAL_COUNTERFACTUAL,
    } and not channels & (_HELDOUT_CHANNELS | {EvidenceChannel.SPATIAL_COLOCALIZATION}):
        reasons.append("context claim requires held-out or spatial-colocalization evidence")

    estimand = request.estimand
    expected_process = {
        ScientificClaimKind.RELATIVE_SELECTION: AbundanceProcess.RELATIVE_SELECTION,
        ScientificClaimKind.ABSOLUTE_NET_CHANGE: AbundanceProcess.ABSOLUTE_NET_CHANGE,
        ScientificClaimKind.PROLIFERATION_COMPONENT: AbundanceProcess.PROLIFERATION,
        ScientificClaimKind.DEATH_COMPONENT: AbundanceProcess.DEATH,
        ScientificClaimKind.MIGRATION_OR_COMPARTMENT_LOSS: (
            AbundanceProcess.MIGRATION_OR_COMPARTMENT_LOSS
        ),
    }.get(request.kind)
    if expected_process is not None:
        if estimand.abundance_process != expected_process:
            reasons.append(f"claim requires {expected_process.value} abundance process")
        if estimand.abundance_scale is None or estimand.abundance_entity is None:
            reasons.append("abundance claim lacks scale and entity")
        else:
            profile = assessment.abundance_profile(estimand.abundance_scale)
            if profile is None:
                reasons.append("requested abundance scale is unsupported by the study")
            else:
                if estimand.abundance_entity not in profile.entities:
                    reasons.append("requested abundance entity is unsupported at this scale")
                if expected_process not in profile.processes and expected_process not in {
                    AbundanceProcess.PROLIFERATION,
                    AbundanceProcess.DEATH,
                    AbundanceProcess.MIGRATION_OR_COMPARTMENT_LOSS,
                }:
                    reasons.append("requested abundance process is unsupported at this scale")
                if estimand.system_boundary != profile.system_boundary:
                    reasons.append("claim system boundary differs from the abundance capability")

    component_requirements = {
        ScientificClaimKind.ABSOLUTE_NET_CHANGE: {EvidenceChannel.ABSOLUTE_CELL_COUNT},
        ScientificClaimKind.PROLIFERATION_COMPONENT: {
            EvidenceChannel.ABSOLUTE_CELL_COUNT,
            EvidenceChannel.PROLIFERATION_ASSAY,
        },
        ScientificClaimKind.DEATH_COMPONENT: {
            EvidenceChannel.ABSOLUTE_CELL_COUNT,
            EvidenceChannel.DEATH_ASSAY,
        },
        ScientificClaimKind.MIGRATION_OR_COMPARTMENT_LOSS: {
            EvidenceChannel.ABSOLUTE_CELL_COUNT,
            EvidenceChannel.MIGRATION_ASSAY,
        },
    }
    required = component_requirements.get(request.kind)
    if required is not None:
        _require_channels(reasons, channels, required, "claim lacks its component-specific assay")
    if request.kind in {
        ScientificClaimKind.PROLIFERATION_COMPONENT,
        ScientificClaimKind.DEATH_COMPONENT,
        ScientificClaimKind.MIGRATION_OR_COMPARTMENT_LOSS,
    } and not channels & (_INTERVENTION_CHANNELS | {EvidenceChannel.ORTHOGONAL_COMPONENT_ASSAY}):
        reasons.append("separated abundance component requires intervention/component evidence")

    expected_trajectory = {
        ScientificClaimKind.POPULATION_TRANSITION: (
            LineageResultSemantics.POPULATION_TRANSITION_ONLY
        ),
        ScientificClaimKind.CLONE_FATE: LineageResultSemantics.CLONE_RESOLVED_FATE,
        ScientificClaimKind.ANCESTRAL_LINEAGE: LineageResultSemantics.ANCESTRAL_TREE,
        ScientificClaimKind.DIRECT_CELL_PATH: (LineageResultSemantics.DIRECT_CELL_PATH_VALIDATION),
    }.get(request.kind)
    if expected_trajectory is not None and estimand.trajectory_semantics != expected_trajectory:
        reasons.append(f"claim requires {expected_trajectory.value} trajectory semantics")
    lineage_channel = {
        ScientificClaimKind.CLONE_FATE: EvidenceChannel.STABLE_CLONE_BARCODE,
        ScientificClaimKind.ANCESTRAL_LINEAGE: EvidenceChannel.EVOLVING_BARCODE,
        ScientificClaimKind.DIRECT_CELL_PATH: EvidenceChannel.LIVE_CELL_TRACKING,
    }.get(request.kind)
    if lineage_channel is not None and lineage_channel not in channels:
        reasons.append(f"claim requires {lineage_channel.value} evidence")

    if (
        request.kind
        in {
            ScientificClaimKind.ECOLOGICAL_MECHANISM,
            ScientificClaimKind.CAUSAL_MECHANISM,
        }
        and not channels & _INTERVENTION_CHANNELS
    ):
        reasons.append("mechanism claim requires intervention, rescue, or epistasis evidence")
    if request.kind == ScientificClaimKind.BIOLOGICAL_PLASTICITY and (
        EvidenceChannel.ORTHOGONAL_COMPONENT_ASSAY not in channels
    ):
        reasons.append("plasticity requires an orthogonal component assay")

    required_study_field = {
        ScientificClaimKind.PROLIFERATION_COMPONENT: StudyField.PROLIFERATION_ASSAY,
        ScientificClaimKind.DEATH_COMPONENT: StudyField.DEATH_ASSAY,
        ScientificClaimKind.MIGRATION_OR_COMPARTMENT_LOSS: StudyField.MIGRATION_ASSAY,
        ScientificClaimKind.ECOLOGICAL_MECHANISM: StudyField.MEDIATOR_INTERVENTION,
    }.get(request.kind)
    if required_study_field is not None and required_study_field not in study.observed_fields:
        reasons.append(f"study lacks {required_study_field.value}")

    expected_causal = {
        ScientificClaimKind.DIFFUSION_ATTRIBUTION: CausalStatus.MODEL_ATTRIBUTION,
        ScientificClaimKind.ECOLOGICAL_MECHANISM: CausalStatus.CAUSAL_UNDER_INTERVENTION,
        ScientificClaimKind.CAUSAL_MECHANISM: CausalStatus.CAUSAL_UNDER_INTERVENTION,
    }.get(request.kind, CausalStatus.PREDICTIVE_ASSOCIATION)
    if estimand.causal_status != expected_causal:
        reasons.append(f"claim kind requires {expected_causal.value} causal status")

    if reasons:
        decision = ClaimDecision.BLOCKED
        wording = None
    elif request.kind == ScientificClaimKind.DIFFUSION_ATTRIBUTION:
        decision = ClaimDecision.MODEL_ATTRIBUTION_ONLY
        wording = _render(request)
    else:
        decision = ClaimDecision.PERMITTED
        wording = _render(request)
    payload: dict[str, Any] = {
        "schema_id": "credo.claim_adjudication",
        "schema_version": 1,
        "adjudication_id": "pending",
        "claim_id": request.claim_id,
        "claim_request_id": request.claim_request_id,
        "capability_assessment_id": assessment.capability_assessment_id,
        "decision": decision,
        "evidence_tiers": tuple(
            sorted({link.descriptor.tier for link in request.evidence_links}, key=lambda x: x.value)
        ),
        "evidence_channels": tuple(sorted(channels, key=lambda x: x.value)),
        "permitted_wording": wording,
        "blocking_reasons": tuple(sorted(set(reasons))),
    }
    return _identity(ClaimAdjudication, payload, "adjudication_id")  # type: ignore[return-value]
