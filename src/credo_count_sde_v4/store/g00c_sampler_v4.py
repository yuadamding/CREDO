"""Authority-owned sampler replay and support derivation for Dev37."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    ArtifactRef,
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDurableRestartReceiptV4,
    G00CRefitSeedScheduleV1,
    G00CSamplerEvidenceV4,
    G00CSamplerPlanV4,
    G00CSupportAuditContractV2,
    G00CSupportAuditReceiptV3,
)
from ..errors import IntegrityError
from .g00c_evidence_v4 import verify_g00c_restart_v4
from .g00c_sampler_v3 import (
    derive_support_table_v3,
    replay_sampler_plan_v3,
)
from .g00c_v3 import _path, _read_model


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV3, role: str) -> str:
    matches = [
        item.artifact.sha256
        for item in authority.implementation.implementations
        if item.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev37 authority has no unique {role} implementation.")
    return matches[0]


def verify_g00c_sampler_v4(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    evidence: G00CSamplerEvidenceV4,
    hierarchy: pd.DataFrame,
    *,
    expected_plan_artifact: ArtifactRef | None = None,
    expected_plan_id: str | None = None,
) -> tuple[G00CSamplerPlanV4, np.ndarray[Any, Any], pd.DataFrame]:
    """Regenerate all rows/states from the authority-owned plan and fresh-process evidence."""

    plan_artifact = expected_plan_artifact or authority.base_sampler_plan
    plan_id = expected_plan_id or authority.base_sampler_plan_id
    if (
        evidence.execution_authority_id != authority.authority_id
        or evidence.sampler_plan != plan_artifact
        or evidence.sampler_plan_id != plan_id
        or evidence.row_hierarchy != authority.sampler_row_hierarchy
        or evidence.nested_training_row_order != authority.nested_training_row_order
        or evidence.implementation_sha256 != _implementation_hash(authority, "sampler")
    ):
        raise IntegrityError("Dev37 sampler evidence differs from its pre-access authority.")
    plan = _read_model(root, plan_artifact, G00CSamplerPlanV4)
    if (
        plan.plan_id != plan_id
        or plan.execution_authority_namespace != authority.selection_freeze_id
        or (
            expected_plan_artifact is None
            and plan.expected_total_trace_rows != authority.base_sampler_expected_trace_rows
        )
    ):
        raise IntegrityError("Dev37 authority-owned sampler plan is cross-wired.")
    schedule = _read_model(root, authority.seed_schedule, G00CRefitSeedScheduleV1)
    if plan.seed_schedule_id != schedule.schedule_id:
        raise IntegrityError("Dev37 sampler plan binds another seed schedule.")
    order_frame = pd.read_parquet(_path(root, authority.nested_training_row_order))
    if tuple(order_frame.columns) != ("rank", "row_id"):
        raise IntegrityError("Dev37 nested training order has another schema.")
    order = order_frame["row_id"].to_numpy(dtype=np.int64)
    uninterrupted, uninterrupted_states = replay_sampler_plan_v3(
        plan, hierarchy, order, schedule, resumed=False
    )
    resumed, resumed_states = replay_sampler_plan_v3(plan, hierarchy, order, schedule, resumed=True)
    if len(uninterrupted) != plan.expected_total_trace_rows:
        raise IntegrityError("Dev37 sampler trace has another pre-access-frozen size.")
    observed = (
        pd.read_parquet(_path(root, evidence.uninterrupted_draw_trace)),
        pd.read_parquet(_path(root, evidence.resumed_draw_trace)),
        pd.read_parquet(_path(root, evidence.uninterrupted_state_trace)),
        pd.read_parquet(_path(root, evidence.resumed_state_trace)),
    )
    expected = (uninterrupted, resumed, uninterrupted_states, resumed_states)
    if any(
        tuple(item.columns) != tuple(reference.columns) or not item.equals(reference)
        for item, reference in zip(observed, expected, strict=True)
    ):
        raise IntegrityError("Dev37 sampler evidence differs from executable replay.")
    if not uninterrupted.equals(resumed) or not uninterrupted_states.equals(resumed_states):
        raise IntegrityError("Dev37 resumed sampler output differs from uninterrupted output.")
    restart = _read_model(root, evidence.restart_receipt, G00CDurableRestartReceiptV4)
    if (
        restart.receipt_id != evidence.restart_receipt_id
        or restart.component != "sampler"
        or restart.uninterrupted_outputs
        != (evidence.uninterrupted_draw_trace, evidence.uninterrupted_state_trace)
        or restart.resumed_outputs != (evidence.resumed_draw_trace, evidence.resumed_state_trace)
    ):
        raise IntegrityError("Dev37 sampler restart receipt is cross-wired.")
    verify_g00c_restart_v4(root, restart)
    return plan, order, uninterrupted


def verify_g00c_support_v4(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    evidence: G00CSamplerEvidenceV4,
    evidence_artifact: ArtifactRef,
    receipt: G00CSupportAuditReceiptV3,
    *,
    plan: G00CSamplerPlanV4,
    hierarchy: pd.DataFrame,
    order: np.ndarray[Any, Any],
    trace: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute one stage's support table from the verified Dev37 trace."""

    contract = _read_model(root, receipt.support_contract, G00CSupportAuditContractV2)
    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.support_contract_id != contract.contract_id
        or receipt.sampler_evidence != evidence_artifact
        or receipt.sampler_evidence_id != evidence.evidence_id
        or receipt.implementation_sha256 != _implementation_hash(authority, "support_auditor")
    ):
        raise IntegrityError("Dev37 support receipt is cross-wired.")
    derived = derive_support_table_v3(
        contract,
        plan,  # type: ignore[arg-type]
        hierarchy,
        order,
        trace,
    )
    observed = pd.read_parquet(_path(root, receipt.derived_support_table))
    if tuple(observed.columns) != tuple(derived.columns) or not observed.equals(derived):
        raise IntegrityError("Dev37 support table differs from sampler-derived support.")
    return derived
