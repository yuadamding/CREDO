"""Evaluation gates for claim-grade CREDO outputs."""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import pandas as pd


ESS_STATUS_ORDER = {
    "not_available": 0,
    "fail": 1,
    "claim_grade_blocked": 2,
    "warn": 3,
    "pass": 4,
}
ESS_GATE_COLUMNS = (
    "ess_gate_status",
    "ess_claim_grade_allowed",
    "ess_claim_grade_allowed_strict",
    "ess_claim_grade_allowed_lenient",
    "ess_failed_gates",
    "ess_warn_frac_used",
    "ess_fail_frac_used",
    "ess_claim_grade_min_frac_used",
    "ess_max_weight_frac_fail_used",
)


def _finite_metric(metrics: Mapping[str, object], names: Iterable[str]) -> float:
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return math.nan


def ess_gate_status(
    metrics: Mapping[str, float],
    *,
    ess_warn_frac: float = 0.20,
    ess_fail_frac: float = 0.05,
    ess_claim_grade_min_frac: float = 0.10,
    ess_max_weight_frac_fail: float = 0.50,
) -> str:
    """Classify particle-weight degeneracy from rollout ESS diagnostics."""
    terminal_min = _finite_metric(metrics, ("terminal_ess_frac_min", "terminal_ess_frac"))
    min_over_time = _finite_metric(metrics, ("min_ess_frac_over_time", "min_ess_frac_mean"))
    min_over_time_available = math.isfinite(min_over_time)
    max_weight = _finite_metric(metrics, ("max_weight_frac_over_time", "max_weight_frac_mean"))
    if not min_over_time_available:
        min_over_time = terminal_min
    if not math.isfinite(terminal_min) or not math.isfinite(min_over_time) or not math.isfinite(max_weight):
        return "not_available"
    if terminal_min < ess_fail_frac or min_over_time < ess_fail_frac or max_weight > ess_max_weight_frac_fail:
        return "fail"
    if terminal_min < ess_claim_grade_min_frac:
        return "claim_grade_blocked"
    if terminal_min < ess_warn_frac or min_over_time < ess_warn_frac:
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
    terminal_min = _finite_metric(metrics, ("terminal_ess_frac_min", "terminal_ess_frac"))
    min_over_time = _finite_metric(metrics, ("min_ess_frac_over_time", "min_ess_frac_mean"))
    min_over_time_available = math.isfinite(min_over_time)
    max_weight = _finite_metric(metrics, ("max_weight_frac_over_time", "max_weight_frac_mean"))
    if not min_over_time_available:
        min_over_time = terminal_min
    if not math.isfinite(terminal_min):
        failed.append("terminal_ess_frac_min_missing")
    elif terminal_min < ess_claim_grade_min_frac:
        failed.append("terminal_ess_frac_min")
    if not min_over_time_available and not math.isfinite(terminal_min):
        failed.append("min_ess_frac_over_time_missing")
    elif min_over_time_available and min_over_time < ess_fail_frac:
        failed.append("min_ess_frac_over_time")
    if not math.isfinite(max_weight):
        failed.append("max_weight_frac_over_time_missing")
    elif max_weight > ess_max_weight_frac_fail:
        failed.append("max_weight_frac_over_time")
    strict_allowed = status == "pass"
    lenient_allowed = status in {"pass", "warn"}
    return {
        "ess_gate_status": status,
        "ess_claim_grade_allowed": lenient_allowed,
        "ess_claim_grade_allowed_strict": strict_allowed,
        "ess_claim_grade_allowed_lenient": lenient_allowed,
        "ess_failed_gates": ",".join(failed),
        "ess_warn_frac_used": float(ess_warn_frac),
        "ess_fail_frac_used": float(ess_fail_frac),
        "ess_claim_grade_min_frac_used": float(ess_claim_grade_min_frac),
        "ess_max_weight_frac_fail_used": float(ess_max_weight_frac_fail),
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
    base = frame.drop(columns=[column for column in ESS_GATE_COLUMNS if column in frame], errors="ignore")
    rows = [
        ess_claim_gate(
            row,
            ess_warn_frac=ess_warn_frac,
            ess_fail_frac=ess_fail_frac,
            ess_claim_grade_min_frac=ess_claim_grade_min_frac,
            ess_max_weight_frac_fail=ess_max_weight_frac_fail,
        )
        for row in base.to_dict(orient="records")
    ]
    gate_frame = pd.DataFrame(rows, index=base.index)
    return pd.concat([base.copy(), gate_frame], axis=1)


__all__ = [
    "ESS_STATUS_ORDER",
    "ESS_GATE_COLUMNS",
    "append_ess_claim_gate",
    "ess_claim_gate",
    "ess_gate_status",
]
