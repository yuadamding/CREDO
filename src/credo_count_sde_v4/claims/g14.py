"""Cross-contract validation for the G14 evidence-only stage."""

from __future__ import annotations

from collections.abc import Mapping

from ..contracts import (
    ClaimAdjudication,
    ClaimDecision,
    ClaimRegistry,
    G14MultiplicityContract,
    G14RobustnessPlan,
    G14SealContract,
)
from ..errors import IntegrityError

_COMPATIBLE_METHODS = {
    "paired_max_statistic": {"paired_max_statistic"},
    "joint_max_statistic": {"joint_max_or_holm"},
    "holm_fwer": {"joint_max_or_holm", "holm_or_westfall_young"},
    "westfall_young": {"holm_or_westfall_young"},
    "bh_fdr": {"bh_fdr_locked_family"},
    "descriptive_only": {"bh_fdr_locked_family"},
}


def validate_g14_contracts(
    registry: ClaimRegistry,
    robustness: G14RobustnessPlan,
    multiplicity: G14MultiplicityContract,
    seal: G14SealContract,
    evidence_adjudications: Mapping[str, ClaimAdjudication] | None = None,
) -> None:
    """Fail closed unless registry, sensitivity, multiplicity, and seal agree."""

    if multiplicity.claim_registry_id != registry.registry_id:
        raise IntegrityError("G14 multiplicity references a different claim registry.")
    if (
        seal.claim_registry_id != registry.registry_id
        or seal.robustness_plan_id != robustness.plan_id
        or seal.multiplicity_contract_id != multiplicity.multiplicity_contract_id
    ):
        raise IntegrityError("G14 seal references an incompatible contract graph.")
    records = {record.claim_id: record for record in registry.records}
    covered: list[str] = []
    for family in multiplicity.families:
        for claim_id in family.claim_ids:
            if claim_id not in records:
                raise IntegrityError(f"G14 multiplicity references unknown claim {claim_id}.")
            if records[claim_id].claim_family != family.family_id:
                raise IntegrityError(f"G14 claim {claim_id} occurs in the wrong family.")
            if family.method not in _COMPATIBLE_METHODS[records[claim_id].multiplicity_method]:
                raise IntegrityError(
                    f"G14 claim {claim_id} uses a multiplicity method incompatible with its family."
                )
            covered.append(claim_id)
    if set(covered) != set(records) or len(covered) != len(records):
        raise IntegrityError("G14 multiplicity must cover every claim exactly once.")
    axis_claims = {axis.axis_id: set(axis.claim_ids) for axis in robustness.axes}
    for axis_id, claim_ids in axis_claims.items():
        unknown = claim_ids - set(records)
        if unknown:
            raise IntegrityError(
                f"G14 robustness axis {axis_id} references unknown claims {sorted(unknown)}."
            )
    for claim_id, record in records.items():
        missing_axes = {
            axis_id
            for axis_id in record.required_robustness_axis_ids
            if axis_id not in axis_claims or claim_id not in axis_claims[axis_id]
        }
        if missing_axes:
            raise IntegrityError(
                f"G14 claim {claim_id} lacks required robustness axes {sorted(missing_axes)}."
            )
    if set(seal.claim_decisions) != set(records):
        raise IntegrityError("G14 seal must bind exactly one decision for every claim.")
    expected_components = {record.component_id for record in records.values()}
    if set(seal.component_receipt_hashes) != expected_components:
        raise IntegrityError("G14 evidence graph must bind every and only claimed components.")
    for claim_id, record in records.items():
        decision = seal.claim_decisions[claim_id]
        if evidence_adjudications is not None and claim_id in evidence_adjudications:
            adjudication = evidence_adjudications[claim_id]
            if adjudication.claim_id != claim_id:
                raise IntegrityError(f"G14 claim {claim_id} has a cross-wired adjudication.")
            if decision == "promoted" and adjudication.decision == ClaimDecision.BLOCKED:
                raise IntegrityError(f"G14 claim {claim_id} has a blocked evidence adjudication.")
        if record.external_independence_class == "unresolved_blocked" and decision == "promoted":
            raise IntegrityError(f"Unresolved external claim {claim_id} cannot be promoted.")
        if record.claim_family == "M" and decision == "promoted":
            failed_parents = [
                parent_id
                for parent_id in record.parent_claim_ids
                if seal.claim_decisions[parent_id] != "promoted"
            ]
            if failed_parents:
                raise IntegrityError(
                    f"Mechanism claim {claim_id} cannot activate before predictive parents pass."
                )
