"""Evaluation gates for claim-grade CREDO outputs."""
from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


ESS_STATUS_ORDER = {
    "not_available": 0,
    "fail": 1,
    "claim_grade_blocked": 2,
    "warn": 3,
    "pass": 4,
}


def ess_gate_status(
    metrics: Mapping[str, float],
    *,
    ess_warn_frac: float = 0.20,
    ess_fail_frac: float = 0.05,
    ess_claim_grade_min_frac: float = 0.10,
    ess_max_weight_frac_fail: float = 0.50,
) -> str:
    """Classify particle-weight degeneracy from rollout ESS diagnostics."""
    terminal_min = float(metrics.get("terminal_ess_frac_min", math.nan))
    max_weight = float(metrics.get("max_weight_frac_mean", math.nan))
    if not math.isfinite(terminal_min) or not math.isfinite(max_weight):
        return "not_available"
    if terminal_min < ess_fail_frac or max_weight > ess_max_weight_frac_fail:
        return "fail"
    if terminal_min < ess_claim_grade_min_frac:
        return "claim_grade_blocked"
    if terminal_min < ess_warn_frac:
        return "warn"
    return "pass"


def ess_claim_gate(
    metrics: Mapping[str, float],
    *,
    ess_warn_frac: float = 0.20,
    ess_fail_frac: float = 0.05,
    ess_claim_grade_min_frac: float = 0.10,
    ess_max_weight_frac_fail: float = 0.50,
) -> dict[str, object]:
    """Return a stable claim-gate payload for one effect or rollout row."""
    status = ess_gate_status(
        metrics,
        ess_warn_frac=ess_warn_frac,
        ess_fail_frac=ess_fail_frac,
        ess_claim_grade_min_frac=ess_claim_grade_min_frac,
        ess_max_weight_frac_fail=ess_max_weight_frac_fail,
    )
    failed: list[str] = []
    terminal_min = float(metrics.get("terminal_ess_frac_min", math.nan))
    max_weight = float(metrics.get("max_weight_frac_mean", math.nan))
    if not math.isfinite(terminal_min):
        failed.append("terminal_ess_frac_min_missing")
    elif terminal_min < ess_claim_grade_min_frac:
        failed.append("terminal_ess_frac_min")
    if not math.isfinite(max_weight):
        failed.append("max_weight_frac_mean_missing")
    elif max_weight > ess_max_weight_frac_fail:
        failed.append("max_weight_frac_mean")
    return {
        "ess_gate_status": status,
        "ess_claim_grade_allowed": status in {"pass", "warn"},
        "ess_failed_gates": ",".join(failed),
    }


def append_ess_claim_gate(
    frame: pd.DataFrame,
    *,
    ess_warn_frac: float = 0.20,
    ess_fail_frac: float = 0.05,
    ess_claim_grade_min_frac: float = 0.10,
    ess_max_weight_frac_fail: float = 0.50,
) -> pd.DataFrame:
    """Append ESS claim-gate columns to an evaluation table."""
    rows = [
        ess_claim_gate(
            row,
            ess_warn_frac=ess_warn_frac,
            ess_fail_frac=ess_fail_frac,
            ess_claim_grade_min_frac=ess_claim_grade_min_frac,
            ess_max_weight_frac_fail=ess_max_weight_frac_fail,
        )
        for row in frame.to_dict(orient="records")
    ]
    gate_frame = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame.copy(), gate_frame], axis=1)


__all__ = [
    "ESS_STATUS_ORDER",
    "append_ess_claim_gate",
    "ess_claim_gate",
    "ess_gate_status",
]
