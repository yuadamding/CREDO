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
from typing import Any, Literal, Mapping, Optional


MassErrorKind = Literal["abs_log_residual", "relative_error", "unknown"]


def _last(values: Any) -> float:
    """Return the last finite scalar of a list/sequence (or a finite scalar), else NaN.

    Robust to NumPy scalar types, 0-D arrays, and scalar tensors: a direct
    float() conversion is attempted before falling back to sequence handling.
    """
    if values is None or isinstance(values, str):
        return math.nan
    # Scalars (incl. numpy/torch scalar types and 0-D arrays) convert directly.
    try:
        out = float(values)
    except (TypeError, ValueError):
        pass
    else:
        return out if math.isfinite(out) else math.nan
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
    # are the decomposed pieces when an evaluator can provide them; mass_error_*
    # stores the typed terminal-mass diagnostic, NOT the tau penalty term.
    endpoint_geom_mass: float = math.nan  # best available (held-out if present)
    train_endpoint_geom_mass: Optional[float] = None  # training-loss value, for reference
    endpoint_sinkhorn: Optional[float] = None
    endpoint_mass_penalty: Optional[float] = None
    # Typed finite-measure mass diagnostic. ``abs_log_residual`` is the only
    # kind accepted by claim-grade gates; ``relative_error`` may still be useful
    # for screening, but it is not interchangeable with an absolute log residual.
    mass_error_value: float = math.nan
    mass_error_kind: MassErrorKind = "unknown"
    signed_log_mass_residual: Optional[float] = None
    # Backward-compatible alias for older callers. New code should use
    # ``mass_error_value`` + ``mass_error_kind``.
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
    source_ess_frac: float = math.nan
    factual_terminal_ess_frac: float = math.nan
    reference_terminal_ess_frac: float = math.nan
    factual_min_ess_frac_over_time: float = math.nan
    reference_min_ess_frac_over_time: float = math.nan
    factual_max_weight_frac: float = math.nan
    reference_max_weight_frac: float = math.nan
    factual_logw_range: float = math.nan
    reference_logw_range: float = math.nan

    # Cost / status
    gpu_seconds: float = math.nan
    wall_seconds: float = math.nan
    converged: bool = False
    diverged: bool = False

    def __post_init__(self) -> None:
        if self.mass_error_kind not in ("abs_log_residual", "relative_error", "unknown"):
            raise ValueError(f"Unknown mass_error_kind: {self.mass_error_kind!r}.")
        mass_error = _finite_or_nan(self.mass_error_value)
        legacy = _finite_or_nan(self.log_mass_error)
        if math.isnan(mass_error) and math.isfinite(legacy):
            self.mass_error_value = legacy
        elif math.isfinite(mass_error) and math.isnan(legacy):
            self.log_mass_error = mass_error


@dataclass
class CREDOTrainOutput:
    """Rich return type for a trial's train_fn.

    Lets ``run_credo_trial`` populate a reproducibility-complete result
    (checkpoint, history, eval summary, resolved config, and failure metadata)
    rather than only the metrics. ``train_fn`` may still return a bare
    :class:`CREDOTrialMetrics` for convenience.
    """

    metrics: CREDOTrialMetrics
    run_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    history_path: Optional[str] = None
    eval_summary_path: Optional[str] = None
    resolved_config_path: Optional[str] = None
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None


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
    history_path: Optional[str] = None
    eval_summary_path: Optional[str] = None
    resolved_config_path: Optional[str] = None
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None


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
    train_endpoint = _last(history.get("loss_end"))
    endpoint = _summary_opt(summary, "mean_endpoint_geom_mass")
    endpoint_sinkhorn = _summary_opt(summary, "mean_endpoint_sinkhorn")
    endpoint_mass_penalty = _summary_opt(summary, "mean_endpoint_mass_penalty")
    # The headline endpoint can be supplied by the summary either as the combined
    # proxy OR as the pure-geometry term (objective_vector prefers sinkhorn when
    # finite). Either one means the headline endpoint came from the (held-out)
    # eval summary, so provenance should follow the summary.
    headline_from_summary = endpoint is not None or endpoint_sinkhorn is not None
    if endpoint is None:
        endpoint = train_endpoint
    mass_error_value, mass_error_kind, signed_log_mass_residual = _mass_error_from_summary(summary)

    # Provenance follows the source of the headline endpoint metric: if it came
    # from the (held-out) eval summary, trust the summary's validation_source;
    # only fall back to the training history's per-epoch label otherwise. This
    # prevents a stale "train_self_eval" history label from mislabeling a
    # genuinely held-out evaluation.
    # Provenance follows the SOURCE of the headline endpoint metric. If the
    # endpoint came from the eval summary, use the summary's validation_source;
    # if it came from the training history, it is NOT held out regardless of any
    # validation_source the summary happens to carry (that label belongs to some
    # other summarized quantity, not the training-loss endpoint).
    if headline_from_summary:
        validation_source = summary.get("validation_source")
    else:
        validation_source = _last_str(history.get("validation_source"))
        if validation_source is None and not math.isnan(train_endpoint):
            validation_source = "train_self_eval"

    # Prefer (sanitized) eval-summary values; fall back to training history.
    count_nll = _summary_opt(summary, "mean_count_nll")
    if count_nll is None:
        count_nll = _opt(history.get("loss_count"))
    weak_loss = _summary_opt(summary, "mean_weak_loss")
    if weak_loss is None:
        weak_loss = _opt(history.get("loss_weak"))

    return CREDOTrialMetrics(
        endpoint_geom_mass=float(endpoint) if endpoint is not None else math.nan,
        train_endpoint_geom_mass=None if math.isnan(train_endpoint) else train_endpoint,
        endpoint_sinkhorn=endpoint_sinkhorn,
        endpoint_mass_penalty=endpoint_mass_penalty,
        mass_error_value=mass_error_value,
        mass_error_kind=mass_error_kind,
        signed_log_mass_residual=signed_log_mass_residual,
        log_mass_error=mass_error_value,
        count_nll=count_nll,
        weak_loss=weak_loss,
        heldout_score=_summary_opt(summary, "heldout_score"),
        validation_source=validation_source,
        control_null_gap=_summary_opt(summary, "control_null_gap"),
        guide_concordance_gap=_summary_opt(summary, "guide_concordance_gap"),
        terminal_ess_frac_min=_last(history.get("terminal_ess_frac_min")),
        min_ess_frac_over_time=_last(
            history.get("min_ess_frac_over_time", history.get("min_ess_frac_mean"))
        ),
        max_weight_frac_mean=_last(history.get("max_weight_frac_mean")),
        logw_range_max=_last(history.get("logw_range_max")),
        source_ess_frac=_last(summary.get("source_ess_frac", history.get("source_ess_frac"))),
        factual_terminal_ess_frac=_last(
            summary.get("factual_terminal_ess_frac", history.get("factual_terminal_ess_frac"))
        ),
        reference_terminal_ess_frac=_last(
            summary.get("reference_terminal_ess_frac", history.get("reference_terminal_ess_frac"))
        ),
        factual_min_ess_frac_over_time=_last(
            summary.get(
                "factual_min_ess_frac_over_time",
                history.get("factual_min_ess_frac_over_time"),
            )
        ),
        reference_min_ess_frac_over_time=_last(
            summary.get(
                "reference_min_ess_frac_over_time",
                history.get("reference_min_ess_frac_over_time"),
            )
        ),
        factual_max_weight_frac=_last(
            summary.get("factual_max_weight_frac", history.get("factual_max_weight_frac"))
        ),
        reference_max_weight_frac=_last(
            summary.get("reference_max_weight_frac", history.get("reference_max_weight_frac"))
        ),
        factual_logw_range=_last(summary.get("factual_logw_range", history.get("factual_logw_range"))),
        reference_logw_range=_last(
            summary.get("reference_logw_range", history.get("reference_logw_range"))
        ),
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
    # Prune on the HELD-OUT endpoint loss when the trainer provides a finite one
    # (val_endpoint_loss / eval_endpoint_loss); fall back to the training loss
    # (loss_end) when no finite validation signal exists. Using _first_finite
    # means a present-but-NaN validation key correctly falls through to the next
    # source instead of poisoning the metric.
    train_endpoint = _last(m.get("loss_end"))
    val_endpoint = _first_finite(m.get("val_endpoint_loss"), m.get("eval_endpoint_loss"))
    # Provenance follows the endpoint actually used: only when a finite held-out
    # endpoint is used do we keep the reported validation_source; if we fell back
    # to the training loss, the metric is train_self_eval (never held_out).
    if math.isfinite(val_endpoint):
        endpoint = val_endpoint
        validation_source = _last_str(m.get("validation_source"))
    else:
        endpoint = train_endpoint
        validation_source = "train_self_eval" if math.isfinite(train_endpoint) else None
    total = _last(m.get("loss_total", m.get("loss_end")))
    mass_error_value, mass_error_kind, signed_log_mass_residual = _mass_error_from_epoch(m)
    return CREDOTrialMetrics(
        endpoint_geom_mass=endpoint,
        train_endpoint_geom_mass=None if math.isnan(train_endpoint) else train_endpoint,
        mass_error_value=mass_error_value,
        mass_error_kind=mass_error_kind,
        signed_log_mass_residual=signed_log_mass_residual,
        log_mass_error=mass_error_value,
        count_nll=_opt_value(_first_finite(m.get("val_count_nll"), m.get("loss_count"))),
        weak_loss=_opt_value(_first_finite(m.get("val_weak_loss"), m.get("loss_weak"))),
        validation_source=validation_source,
        terminal_ess_frac_min=_last(m.get("terminal_ess_frac_min")),
        min_ess_frac_over_time=_last(m.get("min_ess_frac_over_time", m.get("min_ess_frac_mean"))),
        max_weight_frac_mean=_last(m.get("max_weight_frac_mean")),
        logw_range_max=_last(m.get("logw_range_max")),
        source_ess_frac=_last(m.get("source_ess_frac")),
        factual_terminal_ess_frac=_last(m.get("factual_terminal_ess_frac")),
        reference_terminal_ess_frac=_last(m.get("reference_terminal_ess_frac")),
        factual_min_ess_frac_over_time=_last(m.get("factual_min_ess_frac_over_time")),
        reference_min_ess_frac_over_time=_last(m.get("reference_min_ess_frac_over_time")),
        factual_max_weight_frac=_last(m.get("factual_max_weight_frac")),
        reference_max_weight_frac=_last(m.get("reference_max_weight_frac")),
        factual_logw_range=_last(m.get("factual_logw_range")),
        reference_logw_range=_last(m.get("reference_logw_range")),
        diverged=bool(diverged) or not math.isfinite(total) or not math.isfinite(endpoint),
    )


def _opt(values: Any) -> Optional[float]:
    out = _last(values)
    return None if math.isnan(out) else out


def _opt_value(value: float) -> Optional[float]:
    return None if value is None or math.isnan(value) else value


def _first_finite(*values: Any) -> float:
    """Return the first finite scalar across the given sources, else NaN.

    Each source is passed through :func:`_last`, so a present-but-NaN value
    correctly falls through to the next source.
    """
    for value in values:
        out = _last(value)
        if math.isfinite(out):
            return out
    return math.nan


def _summary_opt(summary: Mapping[str, Any], key: str) -> Optional[float]:
    """Read a scalar from an eval summary, returning None for missing/NaN/non-numeric
    values so they never enter objectives or constraints as non-finite numbers."""
    if key not in summary:
        return None
    try:
        value = float(summary[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mass_error_from_summary(summary: Mapping[str, Any]) -> tuple[float, MassErrorKind, Optional[float]]:
    signed = _summary_opt(summary, "mean_log_mass_residual")
    if signed is None:
        signed = _summary_opt(summary, "signed_log_mass_residual")
    for key in ("mass_error_value", "mean_abs_log_mass_residual", "mean_log_mass_error", "mean_mass_error"):
        value = _summary_opt(summary, key)
        if value is not None:
            return float(value), "abs_log_residual", signed
    value = _summary_opt(summary, "mean_mass_rel_error")
    if value is not None:
        return float(value), "relative_error", signed
    return math.nan, "unknown", signed


def _mass_error_from_epoch(epoch_metrics: Mapping[str, Any]) -> tuple[float, MassErrorKind, Optional[float]]:
    signed = _opt_value(
        _first_finite(
            epoch_metrics.get("log_mass_residual"),
            epoch_metrics.get("signed_log_mass_residual"),
        )
    )
    for key in ("mass_error_value", "abs_log_mass_residual", "log_mass_error", "mass_error"):
        value = _opt_value(_first_finite(epoch_metrics.get(key)))
        if value is not None:
            return float(value), "abs_log_residual", signed
    value = _opt_value(_first_finite(epoch_metrics.get("mass_rel_error")))
    if value is not None:
        return float(value), "relative_error", signed
    return math.nan, "unknown", signed


__all__ = [
    "CREDOTrainOutput",
    "CREDOTrialMetrics",
    "CREDOTrialResult",
    "MassErrorKind",
    "metrics_from_epoch",
    "metrics_from_history",
]
