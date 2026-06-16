"""Objectives and hard constraints for CREDO setting search.

Two levels, per the recommended design:

* ``pruner_score`` - a cheap, conservative *scalar* used only for early
  stopping / pruning of unpromising trials.
* ``objective_vector`` + ``hard_constraints`` - the constrained multi-objective
  used for final Pareto selection.

The constraints are deliberately wired to the *fixed* diagnostics:

* ESS feasibility uses ``min(terminal_ess_frac_min, min_ess_frac_over_time)``
  so a run that collapses mid-rollout is infeasible even if the terminal step
  recovers (mirrors the claim-grade gate fix in ``credo.eval.gates``).
* Held-out generalization only counts when ``validation_source == "held_out"``;
  a ``train_self_eval`` score must not satisfy a generalization constraint
  (mirrors the trajectory-trainer provenance fix).
* Frozen method semantics must hold (``space.assert_frozen_semantics``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .metrics import CREDOTrialMetrics
from .space import CREDOTrialSpec, assert_frozen_semantics


DIVERGENCE_PENALTY = 1_000.0
MISSING_METRIC_PENALTY = 1_000.0

DEFAULT_PRUNER_WEIGHTS: dict[str, float] = {
    "endpoint_geom_mass": 0.40,
    "log_mass_error": 0.30,
    "count_nll": 0.10,
    "weak_loss": 0.05,
    "gpu_seconds": 0.05,
    "particle_failure": 0.10,
}


@dataclass(frozen=True)
class ConstraintThresholds:
    ess_floor: float = 0.10  # matches ess_claim_grade_min_frac default
    max_weight_ceiling: float = 0.50  # matches ess_max_weight_frac_fail default
    control_null_max: float = math.inf
    guide_concordance_max: float = math.inf
    require_heldout_provenance: bool = True
    # Claim-grade requirements (default off so cheap screening stays permissive).
    # CREDO is a finite-measure model: a claim-grade endpoint run must have a
    # finite mass diagnostic, and counterfactual/guide claims need their
    # diagnostics actually evaluated, not silently absent.
    require_mass_metric: bool = False
    require_control_null: bool = False
    require_guide_concordance: bool = False


DEFAULT_THRESHOLDS = ConstraintThresholds()

# Stricter profile for final, claim-grade biological model selection.
CLAIM_GRADE_THRESHOLDS = ConstraintThresholds(
    require_heldout_provenance=True,
    require_mass_metric=True,
    require_control_null=True,
    require_guide_concordance=True,
)


@dataclass
class Standardizer:
    """Robust per-metric standardizer (median / IQR) fit across trials.

    Defaults to identity so a single trial still produces a usable score; fit it
    on a population of completed trials to make ``pruner_score`` comparable.
    """

    center: Mapping[str, float] | None = None
    scale: Mapping[str, float] | None = None

    def z(self, name: str, value: float) -> float:
        if value is None or math.isnan(value):
            return 0.0
        c = (self.center or {}).get(name, 0.0)
        s = (self.scale or {}).get(name, 1.0)
        s = s if s and abs(s) > 1e-12 else 1.0
        return (value - c) / s

    def z_or_penalty(self, name: str, value: float, *, missing_penalty: float = MISSING_METRIC_PENALTY) -> float:
        """Like :meth:`z`, but a *required* metric that is missing/NaN is a large
        penalty rather than a neutral zero (so a crashed trial with partial logs
        does not look competitive)."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return missing_penalty
        return self.z(name, value)


def _particle_failure_penalty(m: CREDOTrialMetrics) -> float:
    """Soft penalty (>= 0) that grows as particle weights degenerate."""
    penalty = 0.0
    floor = ConstraintThresholds().ess_floor
    ess = min(_nan_to(m.terminal_ess_frac_min, 1.0), _nan_to(m.min_ess_frac_over_time, 1.0))
    if ess < floor:
        penalty += (floor - ess) / floor
    max_w = _nan_to(m.max_weight_frac_mean, 0.0)
    ceiling = ConstraintThresholds().max_weight_ceiling
    if max_w > ceiling:
        penalty += (max_w - ceiling) / max(ceiling, 1e-6)
    return penalty


def pruner_score(
    metrics: CREDOTrialMetrics,
    *,
    weights: Optional[Mapping[str, float]] = None,
    standardizer: Optional[Standardizer] = None,
) -> float:
    """Scalar early-stopping score (lower is better).

    Conservative by design: divergence dominates, particle degeneracy is
    penalized, and cost is a small term. Use only for pruning, not selection.
    """
    if metrics.diverged:
        return DIVERGENCE_PENALTY
    w = dict(DEFAULT_PRUNER_WEIGHTS)
    if weights:
        w.update(weights)
    std = standardizer or Standardizer()

    score = 0.0
    # endpoint_geom_mass is the core fit metric: a missing/NaN value is a large
    # penalty, not a neutral zero (catches trials that crashed with partial logs).
    score += w["endpoint_geom_mass"] * std.z_or_penalty("endpoint_geom_mass", metrics.endpoint_geom_mass)
    score += w["log_mass_error"] * std.z("log_mass_error", metrics.log_mass_error)
    if metrics.count_nll is not None:
        score += w["count_nll"] * std.z("count_nll", metrics.count_nll)
    if metrics.weak_loss is not None:
        score += w["weak_loss"] * std.z("weak_loss", metrics.weak_loss)
    score += w["gpu_seconds"] * std.z("gpu_seconds", metrics.gpu_seconds)
    score += w["particle_failure"] * _particle_failure_penalty(metrics)
    return float(score)


def objective_vector(metrics: CREDOTrialMetrics) -> dict[str, float]:
    """Multi-objective vector for constrained Pareto selection (all minimized).

    ``heldout_generalization`` is only populated when the score is genuinely
    held out; otherwise it is omitted so it cannot silently rank trials on
    training-data fit.
    """
    vector: dict[str, float] = {"gpu_seconds": _nan_to(metrics.gpu_seconds, math.inf)}
    # Only call an axis "endpoint_geometry" when it is genuinely pure geometry.
    # Otherwise expose the honestly-named combined proxy "endpoint_geom_mass",
    # which already couples mass via the log-mass penalty -- avoid double-counting
    # mass under a "geometry" label.
    if metrics.endpoint_sinkhorn is not None:
        vector["endpoint_geometry"] = float(metrics.endpoint_sinkhorn)
        if metrics.endpoint_mass_penalty is not None:
            vector["endpoint_mass_penalty"] = float(metrics.endpoint_mass_penalty)
    else:
        vector["endpoint_geom_mass"] = _nan_to(metrics.endpoint_geom_mass, math.inf)
    if metrics.log_mass_error is not None and not math.isnan(metrics.log_mass_error):
        vector["mass_error"] = float(metrics.log_mass_error)
    if metrics.count_nll is not None:
        vector["count_nll"] = float(metrics.count_nll)
    if metrics.heldout_score is not None and metrics.validation_source == "held_out":
        vector["heldout_generalization"] = float(metrics.heldout_score)
    if metrics.control_null_gap is not None:
        vector["counterfactual_null_gap"] = float(metrics.control_null_gap)
    return vector


def hard_constraints(
    metrics: CREDOTrialMetrics,
    spec: CREDOTrialSpec,
    thresholds: ConstraintThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, bool]:
    """Feasibility constraints; a trial is selectable only if all are True."""
    ess = min(_nan_to(metrics.terminal_ess_frac_min, 0.0), _nan_to(metrics.min_ess_frac_over_time, 0.0))
    constraints = {
        "not_diverged": not metrics.diverged,
        "converged_ok": bool(metrics.converged) and not metrics.diverged,
        # A missing/NaN core fit metric must not pass as feasible (crashed trial).
        "fit_metrics_finite": _finite(metrics.endpoint_geom_mass),
        # Finite-measure claim-grade selection requires a finite mass diagnostic.
        "mass_metric_finite": (not thresholds.require_mass_metric)
        or _finite(metrics.log_mass_error),
        # Intra-trajectory minimum, not just terminal -- a mid-rollout collapse
        # below the floor is infeasible even if the terminal step recovered.
        "ess_ok": ess >= thresholds.ess_floor,
        "max_weight_ok": _nan_to(metrics.max_weight_frac_mean, 1.0) <= thresholds.max_weight_ceiling,
        # When required, a missing diagnostic (gap is None) is infeasible -- a
        # claim cannot rest on an un-evaluated control-null / guide-concordance.
        "control_null_ok": _gap_ok(
            metrics.control_null_gap, thresholds.control_null_max, thresholds.require_control_null
        ),
        "guide_concordance_ok": _gap_ok(
            metrics.guide_concordance_gap,
            thresholds.guide_concordance_max,
            thresholds.require_guide_concordance,
        ),
        "semantic_ok": _frozen_ok(spec),
    }
    if thresholds.require_heldout_provenance:
        # A reported generalization score must be genuinely held out.
        constraints["heldout_provenance_ok"] = (
            metrics.heldout_score is None or metrics.validation_source == "held_out"
        )
    return constraints


def constraints_satisfied(constraints: Mapping[str, bool]) -> bool:
    return all(bool(v) for v in constraints.values())


def feasible_pruner_score(
    metrics: CREDOTrialMetrics,
    spec: CREDOTrialSpec,
    *,
    thresholds: ConstraintThresholds = DEFAULT_THRESHOLDS,
    weights: Optional[Mapping[str, float]] = None,
    standardizer: Optional[Standardizer] = None,
) -> float:
    """Pruner score with a large additive penalty when infeasible.

    Suitable as a single scalar for a constrained single-objective study.
    """
    base = pruner_score(metrics, weights=weights, standardizer=standardizer)
    if not constraints_satisfied(hard_constraints(metrics, spec, thresholds)):
        return base + DIVERGENCE_PENALTY
    return base


def _frozen_ok(spec: CREDOTrialSpec) -> bool:
    try:
        assert_frozen_semantics(spec)
    except ValueError:
        return False
    return True


def _nan_to(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    value = float(value)
    return default if math.isnan(value) else value


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(float(value))


def _gap_ok(gap: Optional[float], max_value: float, required: bool) -> bool:
    """Diagnostic-gap feasibility. When the diagnostic is required, a missing
    value (``None``) is infeasible; otherwise a missing value passes (screening)."""
    if gap is None:
        return not required
    return float(gap) <= max_value


__all__ = [
    "CLAIM_GRADE_THRESHOLDS",
    "ConstraintThresholds",
    "DEFAULT_PRUNER_WEIGHTS",
    "DEFAULT_THRESHOLDS",
    "Standardizer",
    "constraints_satisfied",
    "feasible_pruner_score",
    "hard_constraints",
    "objective_vector",
    "pruner_score",
]
