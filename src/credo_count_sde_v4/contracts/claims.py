"""Atomic typed scientific estimands and fail-closed adjudication records."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .abundance_semantics import AbundanceEntity, AbundanceProcess, AbundanceScale
from .evidence_tiers import EvidenceChannel, EvidenceLink, EvidenceTier
from .lineage_semantics import LineageResultSemantics
from .models import StrictModel
from .scientific_scope import ScientificScope

__all__ = (
    "CausalStatus",
    "ClaimAdjudication",
    "ClaimDecision",
    "EffectDirection",
    "ScientificClaimKind",
    "ScientificClaimRequest",
    "ScientificEstimand",
)


class ScientificClaimKind(StrEnum):
    """Atomic claim classes with distinct evidence conjunctions."""

    PERTURBATION_PROGRAM_ASSOCIATION = "perturbation_program_association"
    POPULATION_TRANSITION = "population_transition"
    RELATIVE_SELECTION = "relative_selection"
    ABSOLUTE_NET_CHANGE = "absolute_net_change"
    PROLIFERATION_COMPONENT = "proliferation_component"
    DEATH_COMPONENT = "death_component"
    MIGRATION_OR_COMPARTMENT_LOSS = "migration_or_compartment_loss"
    DIFFUSION_ATTRIBUTION = "diffusion_attribution"
    BIOLOGICAL_PLASTICITY = "biological_plasticity"
    PHYSICAL_CONTEXT_ASSOCIATION = "physical_context_association"
    ECOLOGICAL_COUNTERFACTUAL = "ecological_counterfactual"
    ECOLOGICAL_MECHANISM = "ecological_mechanism"
    CLONE_FATE = "clone_fate"
    ANCESTRAL_LINEAGE = "ancestral_lineage"
    DIRECT_CELL_PATH = "direct_cell_path"
    CAUSAL_MECHANISM = "causal_mechanism"


class ClaimDecision(StrEnum):
    """Fail-closed disposition for one typed claim."""

    PERMITTED = "permitted"
    MODEL_ATTRIBUTION_ONLY = "model_attribution_only"
    BLOCKED = "blocked"


class EffectDirection(StrEnum):
    """Direction of one atomic estimand."""

    INCREASE = "increase"
    DECREASE = "decrease"
    MIXED = "mixed"
    NULL_COMPATIBLE = "null_compatible"


class CausalStatus(StrEnum):
    """Claim status fixed independently of prose."""

    PREDICTIVE_ASSOCIATION = "predictive_association"
    MODEL_ATTRIBUTION = "model_attribution"
    CAUSAL_UNDER_INTERVENTION = "causal_under_intervention"


class ScientificEstimand(StrictModel):
    """One atomic estimand from which publishable wording is rendered."""

    scope: ScientificScope
    outcome: str = Field(min_length=1)
    estimand: str = Field(min_length=1)
    direction: EffectDirection
    effect_measure: str = Field(min_length=1)
    causal_status: CausalStatus
    mechanism_type: str | None = None
    abundance_scale: AbundanceScale | None = None
    abundance_entity: AbundanceEntity | None = None
    abundance_process: AbundanceProcess | None = None
    system_boundary: str | None = None
    trajectory_semantics: LineageResultSemantics | None = None
    uncertainty_result_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_estimand(self) -> ScientificEstimand:
        abundance = (
            self.abundance_scale,
            self.abundance_entity,
            self.abundance_process,
            self.system_boundary,
        )
        if any(item is not None for item in abundance) and not all(
            item is not None for item in abundance
        ):
            raise ValueError("Abundance scale, entity, and process must be declared together.")
        if self.causal_status == CausalStatus.CAUSAL_UNDER_INTERVENTION and (
            self.mechanism_type is None
        ):
            raise ValueError("A causal estimand requires an explicit mechanism type.")
        return self


class ScientificClaimRequest(StrictModel):
    """Typed atomic claim plus semantically linked evidence; prose is non-authoritative."""

    schema_id: Literal["credo.scientific_claim_request"] = "credo.scientific_claim_request"
    schema_version: Literal[1] = 1
    claim_request_id: str
    claim_id: str = Field(min_length=1)
    kind: ScientificClaimKind
    estimand: ScientificEstimand
    evidence_links: tuple[EvidenceLink, ...]
    requested_free_text: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> ScientificClaimRequest:
        if not self.evidence_links or len(self.evidence_links) != len(set(self.evidence_links)):
            raise ValueError("A scientific claim requires unique semantic evidence links.")
        for link in self.evidence_links:
            if link.claim_id != self.claim_id:
                raise ValueError("Evidence link references another claim.")
            if link.scope != self.estimand.scope:
                raise ValueError("Claim and evidence-link scopes differ.")
        expected = self.identity(id_field="claim_request_id")
        if self.claim_request_id != expected:
            raise ValueError(f"claim_request_id mismatch: expected {expected}.")
        return self

    @property
    def evidence_channels(self) -> frozenset[EvidenceChannel]:
        return frozenset(link.descriptor.channel for link in self.evidence_links)


class ClaimAdjudication(StrictModel):
    """Machine-readable authorization and deterministic rendered wording."""

    schema_id: Literal["credo.claim_adjudication"] = "credo.claim_adjudication"
    schema_version: Literal[1] = 1
    adjudication_id: str
    claim_id: str = Field(min_length=1)
    claim_request_id: str = Field(min_length=1)
    capability_assessment_id: str = Field(min_length=1)
    decision: ClaimDecision
    evidence_tiers: tuple[EvidenceTier, ...]
    evidence_channels: tuple[EvidenceChannel, ...]
    permitted_wording: str | None = None
    blocking_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def validate_adjudication(self) -> ClaimAdjudication:
        if self.evidence_tiers != tuple(sorted(set(self.evidence_tiers), key=lambda x: x.value)):
            raise ValueError("Evidence tiers must be unique and sorted.")
        if self.evidence_channels != tuple(
            sorted(set(self.evidence_channels), key=lambda x: x.value)
        ):
            raise ValueError("Evidence channels must be unique and sorted.")
        if self.blocking_reasons != tuple(sorted(set(self.blocking_reasons))):
            raise ValueError("Claim blocking reasons must be unique and sorted.")
        blocked = self.decision == ClaimDecision.BLOCKED
        if blocked != bool(self.blocking_reasons):
            raise ValueError("Claim decision contradicts its blocking reasons.")
        if blocked == (self.permitted_wording is not None):
            raise ValueError("Only nonblocked claims may carry rendered wording.")
        expected = self.identity(id_field="adjudication_id")
        if self.adjudication_id != expected:
            raise ValueError(f"adjudication_id mismatch: expected {expected}.")
        return self
