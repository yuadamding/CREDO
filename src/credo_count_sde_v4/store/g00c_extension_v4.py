"""V4-chain two-million-row extension authority for Dev37."""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDecisionReceiptV5,
    G00CExecutionBundleV5,
    G00CFeatureSelectionResultV3,
    G00CFinalSealV1,
    G00CSamplerPlanV4,
    G00CSampleSizeExtensionFreezeV2,
    G00CSupportAuditContractV2,
)
from ..errors import IntegrityError
from .g00c_publication_v4 import verify_g00c_final_seal_v1
from .g00c_v3 import _read_model


def verify_g00c_extension_freeze_v2(
    root: Path,
    extension: G00CSampleSizeExtensionFreezeV2,
    *,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    feature_result: G00CFeatureSelectionResultV3,
) -> G00CSamplerPlanV4:
    """Require one sealed V5 base stop and one exact 2M sampler/support surface."""

    base = _read_model(root, extension.base_execution_bundle, G00CExecutionBundleV5)
    decision = _read_model(root, extension.base_extension_required_receipt, G00CDecisionReceiptV5)
    seal = _read_model(root, extension.base_final_seal, G00CFinalSealV1)
    seal_root = (root / extension.base_final_seal.relative_uri).parent
    sealed_bundle, _, sealed_decision = verify_g00c_final_seal_v1(seal_root, seal)
    if (
        base.bundle_id != extension.base_execution_bundle_id
        or decision.receipt_id != extension.base_extension_required_receipt_id
        or seal.seal_id != extension.base_final_seal_id
        or seal.execution_bundle_id != base.bundle_id
        or seal.final_decision_id != decision.receipt_id
        or base.terminal_status != "extension_required"
        or decision.terminal_status != "extension_required"
        or decision.may_parent_g00d
        or decision.execution_bundle_id != base.bundle_id
        or sealed_bundle != base
        or sealed_decision != decision
    ):
        raise IntegrityError("Dev37 extension does not parent one sealed V5 base stop.")
    if (
        extension.execution_authority_id != authority.authority_id
        or base.execution_authority_id != authority.authority_id
        or extension.source_binding_id != authority.source_plane.binding_id
        or extension.feature_selection_result_id != feature_result.result_id
        or extension.selected_feature_count != feature_result.selected_feature_count
        or extension.selected_feature_order_sha256 != feature_result.selected_feature_order_sha256
        or extension.seed_schedule_id != authority.seed_schedule_id
        or extension.seed_schedule_artifact != authority.seed_schedule
        or extension.selected_feature_surface != feature_result.ordered_features
    ):
        raise IntegrityError("Dev37 extension changes authority, features, or seed support.")
    plan = _read_model(root, extension.extension_sampler_plan, G00CSamplerPlanV4)
    support = _read_model(
        root, extension.extension_support_audit_contract, G00CSupportAuditContractV2
    )
    if (
        plan.plan_id != extension.extension_sampler_plan_id
        or plan.grid_stage != "extension"
        or plan.execution_authority_namespace != authority.selection_freeze_id
        or plan.seed_schedule_id != authority.seed_schedule_id
        or support.stage != "extension"
        or support.feature_candidate_counts
        or support.cell_candidate_counts != (2_000_000,)
    ):
        raise IntegrityError("Dev37 extension changes the single 2M sampler/support surface.")
    return plan
