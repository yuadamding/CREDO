"""Stable metric and result schema for CREDO setting search.

These dataclasses are the contract between the trainer and the optimizer. A
real trial fills :class:`CREDOTrialMetrics` from a trainer's
``TrainingHistory.to_dict()`` (plus an eval summary); the objective layer then
turns metrics into a scalar pruner score, a multi-objective vector, and a set
of hard feasibility constraints.

The metric names track the diagnostics CREDO already emits:
``terminal_ess_frac_min``, ``min_ess_frac_mean`` (intra-trajectory minimum,
surfaced here as ``min_ess_frac_over_time``), ``max_weight_frac_mean``,
``logw_range_max``, and the endpoint geometry/mass losses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


def _last(values: Any) -> float:
    """Return the last finite scalar of a list/sequence, else NaN."""
    if values is None:
        return math.nan
    if isinstance(values, (int, float)):
        return float(values)
    try:
        seq = list(values)
    except TypeError:
        return math.nan
    for item in reversed(seq):
        try:
            out = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return math.nan


def _last_str(values: Any) -> Optional[str]:
    if values is None:
        return None
    if isinstance(values, str):
        return values
    try:
        seq = list(values)
    except TypeError:
        return None
    for item in reversed(seq):
        if item is not None:
            return str(item)
    return None


@dataclass
class CREDOTrialMetrics:
    """Outcome metrics for one CREDO trial (one fold/seed)."""

    # Fit quality. endpoint_geom_mass is the *combined* geometry+log-mass proxy
    # (the training-history loss_end). endpoint_sinkhorn / endpoint_mass_penalty
    # are the decomposed pieces when an evaluator can provide them; log_mass_error
    # is a distinct relative terminal-mass diagnostic, NOT the tau penalty term.
    endpoint_geom_mass: float = math.nan
    endpoint_sinkhorn: Optional[float] = None
    endpoint_mass_penalty: Optional[float] = None
    log_mass_error: float = math.nan
    count_nll: Optional[float] = None
    weak_loss: Optional[float] = None

    # Generalization (only meaningful when held out -- see validation_source)
    heldout_score: Optional[float] = None
    validation_source: Optional[str] = None  # "held_out" | "train_self_eval" | None

    # Counterfactual sanity / claim diagnostics
    control_null_gap: Optional[float] = None
    guide_concordance_gap: Optional[float] = None

    # Particle-weight stability
    terminal_ess_frac_min: float = math.nan
    min_ess_frac_over_time: float = math.nan
    max_weight_frac_mean: float = math.nan
    logw_range_max: float = math.nan

    # Cost / status
    gpu_seconds: float = math.nan
    wall_seconds: float = math.nan
    converged: bool = False
    diverged: bool = False


@dataclass
class CREDOTrialResult:
    """A spec, its metrics, and the derived search quantities."""

    spec: Any  # CREDOTrialSpec (avoid import cycle)
    metrics: CREDOTrialMetrics
    objective_vector: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, bool] = field(default_factory=dict)
    pruner_score: float = math.nan
    feasible: bool = False
    run_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None


def metrics_from_history(
    history: Mapping[str, Any],
    *,
    eval_summary: Optional[Mapping[str, Any]] = None,
    gpu_seconds: float = math.nan,
    wall_seconds: float = math.nan,
    diverged: bool = False,
    converged: bool = True,
) -> CREDOTrialMetrics:
    """Build :class:`CREDOTrialMetrics` from a trainer history dict + eval summary.

    ``history`` is expected to look like ``TrainingHistory.to_dict()`` (lists of
    per-epoch values). ``eval_summary`` is an optional flat mapping such as the
    output of ``credo.eval.hnscc.summarize_eval`` (held-out fit) plus any
    counterfactual-null / guide-concordance gaps. Missing keys degrade to
    NaN/None rather than raising, so partial-fidelity runs still produce a row.
    """
    history = dict(history)
    summary = dict(eval_summary or {})

    # Endpoint fit: prefer the eval summary (held out) over the training loss.
    # loss_end IS the combined geometry+log-mass proxy; the training history has
    # no separate pure-geometry or mass term, so do not invent one from history.
    endpoint = summary.get("mean_endpoint_geom_mass")
    endpoint_from_summary = endpoint is not None
    if endpoint is None:
        endpoint = _last(history.get("loss_end"))
    endpoint_sinkhorn = summary.get("mean_endpoint_sinkhorn")
    endpoint_mass_penalty = summary.get("mean_endpoint_mass_penalty")
    # Relative terminal mass error (distinct from the tau penalty); only the eval
    # summary provides it.
    log_mass_error = summary.get("mean_log_mass_error", summary.get("mean_mass_rel_error"))

    # Provenance follows the source of the headline endpoint metric: if it came
    # from the (held-out) eval summary, trust the summary's validation_source;
    # only fall back to the training history's per-epoch label otherwise. This
    # prevents a stale "train_self_eval" history label from mislabeling a
    # genuinely held-out evaluation.
    validation_source = (
        (summary.get("validation_source") if endpoint_from_summary else None)
        or _last_str(history.get("validation_source"))
        or summary.get("validation_source")
    )

    return CREDOTrialMetrics(
        endpoint_geom_mass=float(endpoint) if endpoint is not None else math.nan,
        endpoint_sinkhorn=float(endpoint_sinkhorn) if endpoint_sinkhorn is not None else None,
        endpoint_mass_penalty=float(endpoint_mass_penalty)
        if endpoint_mass_penalty is not None
        else None,
        log_mass_error=float(log_mass_error) if log_mass_error is not None else math.nan,
        count_nll=_opt(history.get("loss_count")),
        weak_loss=_opt(history.get("loss_weak")),
        heldout_score=_opt(summary.get("heldout_score")),
        validation_source=validation_source,
        control_null_gap=_opt(summary.get("control_null_gap")),
        guide_concordance_gap=_opt(summary.get("guide_concordance_gap")),
        terminal_ess_frac_min=_last(history.get("terminal_ess_frac_min")),
        min_ess_frac_over_time=_last(
            history.get("min_ess_frac_over_time", history.get("min_ess_frac_mean"))
        ),
        max_weight_frac_mean=_last(history.get("max_weight_frac_mean")),
        logw_range_max=_last(history.get("logw_range_max")),
        gpu_seconds=float(gpu_seconds),
        wall_seconds=float(wall_seconds),
        converged=bool(converged),
        diverged=bool(diverged),
    )


def metrics_from_epoch(
    epoch_metrics: Mapping[str, Any],
    *,
    diverged: bool = False,
) -> CREDOTrialMetrics:
    """Build metrics from a trainer's *single-epoch* scalar metrics dict.

    Used by the trainer reporter hook so the optimizer can prune on intermediate
    progress. Treats a non-finite total/endpoint loss as divergence.
    """
    m = dict(epoch_metrics)
    endpoint = _last(m.get("loss_end"))
    total = _last(m.get("loss_total", m.get("loss_end")))
    return CREDOTrialMetrics(
        endpoint_geom_mass=endpoint,
        count_nll=_opt(m.get("loss_count")),
        weak_loss=_opt(m.get("loss_weak")),
        validation_source=_last_str(m.get("validation_source")),
        terminal_ess_frac_min=_last(m.get("terminal_ess_frac_min")),
        min_ess_frac_over_time=_last(m.get("min_ess_frac_over_time", m.get("min_ess_frac_mean"))),
        max_weight_frac_mean=_last(m.get("max_weight_frac_mean")),
        logw_range_max=_last(m.get("logw_range_max")),
        diverged=bool(diverged) or not math.isfinite(total),
    )


def _opt(values: Any) -> Optional[float]:
    out = _last(values)
    return None if math.isnan(out) else out


__all__ = [
    "CREDOTrialMetrics",
    "CREDOTrialResult",
    "metrics_from_epoch",
    "metrics_from_history",
]
