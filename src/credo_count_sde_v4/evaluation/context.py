"""Generic context-identification audits; no cohort meaning is inferred here."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..canonical import contract_id
from ..contracts import ContextAuditContract, ContextAuditReceipt


def evaluate_context_audit(
    contract: ContextAuditContract,
    residualized_pool_composition: np.ndarray[Any, Any],
    *,
    selected_rank: int,
    context_parameters: int,
    comparator_parameters: int,
    held_out_access_pass: bool,
    observation_operator_pass: bool,
) -> ContextAuditReceipt:
    matrix = np.asarray(residualized_pool_composition, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Pool-composition audit matrix must be finite and two-dimensional.")
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0:
        effective_rank, condition = 0, None
    else:
        effective_rank = int(np.sum(singular / singular[0] >= contract.singular_ratio_threshold))
        condition = (
            float(singular[0] / singular[selected_rank - 1])
            if 0 < selected_rank <= len(singular) and singular[selected_rank - 1] > 0
            else None
        )
    denominator = max(context_parameters, comparator_parameters, 1)
    parameter_difference = abs(context_parameters - comparator_parameters) / denominator
    eligible = (
        effective_rank >= contract.effective_rank_threshold
        and selected_rank <= min(contract.maximum_interaction_rank, effective_rank)
        and condition is not None
        and condition <= contract.maximum_condition_number
        and parameter_difference <= contract.parameter_match_fraction
        and held_out_access_pass
        and observation_operator_pass
    )
    return ContextAuditReceipt(
        contract_hash=contract_id(contract),
        effective_rank=effective_rank,
        selected_rank=selected_rank,
        condition_number=condition,
        parameter_difference_fraction=parameter_difference,
        held_out_access_pass=held_out_access_pass,
        observation_operator_pass=observation_operator_pass,
        status="eligible" if eligible else "diagnostic_only",
    )
