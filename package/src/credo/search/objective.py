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

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from .metrics import CREDOTrialMetrics, MassErrorKind
from .space import CREDOTrialSpec, assert_frozen_semantics


DIVERGENCE_PENALTY = 1_000.0
MISSING_METRIC_PENALTY = 1_000.0

DEFAULT_PRUNER_WEIGHTS: dict[str, float] = {
    "endpoint_geom_mass": 0.40,
    "mass_error": 0.30,
    "count_nll": 0.10,
    "weak_loss": 0.05,
    "gpu_seconds": 0.05,
    "particle_failure": 0.10,
}


class SearchProfile(str, Enum):
    """Named search phases with different evidentiary standards."""

    LIGHT_SCREEN = "light_screen"
    PARETO_REFIT = "pareto_refit"
    CLAIM_GRADE = "claim_grade"
    ABLATION_ONLY = "ablation_only"


@dataclass(frozen=True)
class ConstraintThresholds:
    ess_floor: float = 0.10  # matches ess_claim_grade_min_frac default
    max_weight_ceiling: float = 0.50  # matches ess_max_weight_frac_fail default
    control_null_max: float = math.inf
    guide_concordance_max: float = math.inf
    # If a heldout_score is reported, its provenance must be held_out.
    require_heldout_provenance: bool = True
    # If True, the headline endpoint metric itself must come from a held-out
    # evaluation (validation_source == "held_out"); a trial selected on training
    # self-evaluation is infeasible. Stricter than require_heldout_provenance,
    # which only constrains an explicitly-reported heldout_score.
    require_heldout_endpoint: bool = False
    # Claim-grade requirements (default off so cheap screening stays permissive).
    # CREDO is a finite-measure model: a claim-grade endpoint run must have a
    # finite mass diagnostic, and counterfactual/guide claims need their
    # diagnostics actually evaluated, not silently absent.
    require_mass_metric: bool = False
    require_control_null: bool = False
    require_guide_concordance: bool = False
    require_branch_particle_diagnostics: bool = False
    required_mass_error_kind: Optional[MassErrorKind] = None
    # Mass *calibration* ceiling (claim-grade): existence of a finite mass error
    # is not enough -- a claim-grade endpoint run must also keep it below this.
    log_mass_error_max: float = math.inf

    def __post_init__(self) -> None:
        # Range validation only. We intentionally do NOT forbid inf ceilings when
        # a diagnostic is "required": the presence profile (require_* with open
        # ceilings) is a legitimate gate. Finite ceilings are enforced by the
        # claim_grade_thresholds(...) factory.
        if not 0.0 <= float(self.ess_floor) <= 1.0:
            raise ValueError(f"ess_floor must be in [0, 1], got {self.ess_floor!r}.")
        if not 0.0 < float(self.max_weight_ceiling) <= 1.0:
            raise ValueError(f"max_weight_ceiling must be in (0, 1], got {self.max_weight_ceiling!r}.")
        if float(self.control_null_max) < 0:
            raise ValueError(f"control_null_max must be non-negative, got {self.control_null_max!r}.")
        if float(self.guide_concordance_max) < 0:
            raise ValueError(
                f"guide_concordance_max must be non-negative, got {self.guide_concordance_max!r}."
            )
        if float(self.log_mass_error_max) < 0:
            raise ValueError(
                f"log_mass_error_max must be non-negative, got {self.log_mass_error_max!r}."
            )
        if self.required_mass_error_kind is not None and self.required_mass_error_kind not in (
            "abs_log_residual",
            "relative_error",
            "unknown",
        ):
            raise ValueError(
                f"required_mass_error_kind must be a known mass kind, got {self.required_mass_error_kind!r}."
            )


DEFAULT_THRESHOLDS = ConstraintThresholds()

# Presence + provenance gate: requires the mass / control-null / guide-concordance
# diagnostics to EXIST and the endpoint metric to be held out, but leaves the gap
# *ceilings* open (inf). This is NOT sufficient for a real claim on its own -- use
# claim_grade_thresholds(...) with null-calibrated finite ceilings (e.g. from the
# practical-null floor profiles) for final selection.
CLAIM_GRADE_PRESENCE_THRESHOLDS = ConstraintThresholds(
    require_heldout_provenance=True,
    require_heldout_endpoint=True,
    require_mass_metric=True,
    require_control_null=True,
    require_guide_concordance=True,
    require_branch_particle_diagnostics=True,
    required_mass_error_kind="abs_log_residual",
)
# Deliberately not an open-ceiling alias. Final biological claims must call
# claim_grade_thresholds(...) with finite calibrated ceilings.
CLAIM_GRADE_THRESHOLDS = None


def claim_grade_thresholds(
    *,
    control_null_max: float,
    log_mass_error_max: Optional[float] = None,
    guide_concordance_max: Optional[float] = None,
    require_guide_concordance: bool = False,
    ess_floor: float = 0.10,
    max_weight_ceiling: float = 0.50,
) -> ConstraintThresholds:
    """Build a claim-grade threshold profile with FINITE diagnostic ceilings.

    ``control_null_max`` is required and must be finite and non-negative, so
    callers cannot silently accept any finite gap by passing inf. When
    ``require_guide_concordance=True``, ``guide_concordance_max`` must also be a
    finite non-negative limit. Enable guide concordance only for guide-level
    claims; perturbation-level claims should leave it off so they are not
    rejected merely because guide-level data are unavailable.
    """
    if not math.isfinite(float(control_null_max)) or float(control_null_max) < 0:
        raise ValueError(
            f"control_null_max must be a finite, non-negative threshold, got {control_null_max!r}."
        )
    if (
        log_mass_error_max is None
        or not math.isfinite(float(log_mass_error_max))
        or float(log_mass_error_max) < 0
    ):
        raise ValueError(
            "log_mass_error_max must be a finite, non-negative abs-log-residual "
            f"threshold, got {log_mass_error_max!r}."
        )
    if require_guide_concordance:
        if (
            guide_concordance_max is None
            or not math.isfinite(float(guide_concordance_max))
            or float(guide_concordance_max) < 0
        ):
            raise ValueError(
                "guide_concordance_max must be a finite, non-negative threshold when "
                f"require_guide_concordance=True, got {guide_concordance_max!r}."
            )
    elif guide_concordance_max is None:
        guide_concordance_max = math.inf
    return ConstraintThresholds(
        ess_floor=ess_floor,
        max_weight_ceiling=max_weight_ceiling,
        control_null_max=control_null_max,
        guide_concordance_max=guide_concordance_max,
        log_mass_error_max=float(log_mass_error_max),
        require_heldout_provenance=True,
        require_heldout_endpoint=True,
        require_mass_metric=True,
        require_control_null=True,
        require_guide_concordance=require_guide_concordance,
        require_branch_particle_diagnostics=True,
        required_mass_error_kind="abs_log_residual",
    )


def thresholds_for_profile(
    profile: SearchProfile | str,
    *,
    claim_thresholds: Optional[ConstraintThresholds] = None,
) -> ConstraintThresholds:
    """Return the default feasibility gate for a named search profile."""
    profile = SearchProfile(profile)
    if profile in (SearchProfile.LIGHT_SCREEN, SearchProfile.ABLATION_ONLY):
        return DEFAULT_THRESHOLDS
    if profile is SearchProfile.PARETO_REFIT:
        return ConstraintThresholds(
            require_heldout_provenance=True,
            require_heldout_endpoint=True,
            require_mass_metric=True,
            require_control_null=True,
            required_mass_error_kind="abs_log_residual",
        )
    if claim_thresholds is None:
        raise ValueError(
            "SearchProfile.CLAIM_GRADE requires explicit finite claim_thresholds "
            "from claim_grade_thresholds(...)."
        )
    return claim_thresholds


def threshold_metadata(thresholds: ConstraintThresholds) -> dict[str, object]:
    """Stable manifest metadata for the feasibility threshold profile."""
    if thresholds is None:
        raise ValueError("threshold_metadata requires a concrete ConstraintThresholds object.")
    payload = dataclasses.asdict(thresholds)
    profile = _threshold_profile(thresholds)
    encoded = json.dumps({"profile": profile, **payload}, sort_keys=True, default=str)
    return {
        "threshold_profile": profile,
        "thresholds_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        **payload,
    }


def _threshold_profile(thresholds: ConstraintThresholds) -> str:
    if thresholds == DEFAULT_THRESHOLDS:
        return "default"
    if _is_finite_claim_grade_thresholds(thresholds):
        return "claim_grade_finite"
    if _is_claim_grade_presence_thresholds(thresholds):
        return "claim_grade_presence"
    return "custom"


def _is_claim_grade_presence_thresholds(thresholds: ConstraintThresholds) -> bool:
    return (
        thresholds.require_heldout_provenance
        and thresholds.require_heldout_endpoint
        and thresholds.require_mass_metric
        and thresholds.require_control_null
        and thresholds.require_branch_particle_diagnostics
        and thresholds.required_mass_error_kind == "abs_log_residual"
    )


def _is_finite_claim_grade_thresholds(thresholds: ConstraintThresholds) -> bool:
    return (
        _is_claim_grade_presence_thresholds(thresholds)
        and math.isfinite(float(thresholds.control_null_max))
        and math.isfinite(float(thresholds.log_mass_error_max))
        and (
            not thresholds.require_guide_concordance
            or math.isfinite(float(thresholds.guide_concordance_max))
        )
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
        if "log_mass_error" in weights and "mass_error" not in weights:
            w["mass_error"] = weights["log_mass_error"]
    std = standardizer or Standardizer()

    score = 0.0
    # The headline endpoint metric is the core fit term: a missing/NaN value is a
    # large penalty, not a neutral zero (catches trials that crashed with partial
    # logs). Use the same combined-or-pure-geometry fallback as hard_constraints,
    # so a feasible decomposed-only trial is not penalized for a missing combined
    # proxy.
    endpoint_name, endpoint_value = _headline_endpoint(metrics)
    score += w["endpoint_geom_mass"] * std.z_or_penalty(endpoint_name, endpoint_value)
    score += w["mass_error"] * std.z_or_penalty("mass_error", metrics.mass_error_value)
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
    # mass under a "geometry" label. _add_if_finite guards against NaN sneaking in
    # from externally-constructed metrics.
    if _finite(metrics.endpoint_sinkhorn):
        vector["endpoint_geometry"] = float(metrics.endpoint_sinkhorn)
        vector["endpoint_geometry_or_proxy"] = float(metrics.endpoint_sinkhorn)
        _add_if_finite(vector, "endpoint_mass_penalty", metrics.endpoint_mass_penalty)
    else:
        # No finite pure-geometry term -> fall back to the combined proxy.
        vector["endpoint_geom_mass"] = _nan_to(metrics.endpoint_geom_mass, math.inf)
        vector["endpoint_geometry_or_proxy"] = _nan_to(metrics.endpoint_geom_mass, math.inf)
    _add_if_finite(vector, "mass_error", metrics.mass_error_value)
    _add_if_finite(vector, "count_nll", metrics.count_nll)
    if metrics.validation_source == "held_out":
        _add_if_finite(vector, "heldout_generalization", metrics.heldout_score)
    _add_if_finite(vector, "counterfactual_null_gap", metrics.control_null_gap)
    _add_if_finite(vector, "guide_concordance_gap", metrics.guide_concordance_gap)
    return vector


def hard_constraints(
    metrics: CREDOTrialMetrics,
    spec: CREDOTrialSpec,
    thresholds: ConstraintThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, bool]:
    """Feasibility constraints; a trial is selectable only if all are True."""
    if thresholds is None:
        raise ValueError(
            "CLAIM_GRADE_THRESHOLDS no longer points to open-ceiling gates. "
            "Use CLAIM_GRADE_PRESENCE_THRESHOLDS for diagnostic presence checks "
            "or claim_grade_thresholds(...) for final claim-grade selection."
        )
    ess = min(_nan_to(metrics.terminal_ess_frac_min, 0.0), _nan_to(metrics.min_ess_frac_over_time, 0.0))
    constraints = {
        "not_diverged": not metrics.diverged,
        "converged_ok": bool(metrics.converged) and not metrics.diverged,
        # A missing/NaN core fit metric must not pass as feasible (crashed trial).
        # Accept the pure-geometry term when the combined proxy is absent, matching
        # objective_vector's headline-endpoint fallback.
        "fit_metrics_finite": _finite(metrics.endpoint_geom_mass)
        or _finite(metrics.endpoint_sinkhorn),
        # Finite-measure claim-grade selection requires a finite mass diagnostic...
        "mass_metric_finite": (not thresholds.require_mass_metric)
        or _finite(metrics.mass_error_value),
        "mass_error_kind_ok": (not thresholds.require_mass_metric)
        or thresholds.required_mass_error_kind is None
        or metrics.mass_error_kind == thresholds.required_mass_error_kind,
        # ...and, when a calibration ceiling is set, the mass error must clear it.
        "mass_error_ok": (not thresholds.require_mass_metric)
        or (
            _finite(metrics.mass_error_value)
            and float(metrics.mass_error_value) <= thresholds.log_mass_error_max
        ),
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
    constraints.update(_branch_particle_constraints(metrics, thresholds))
    if thresholds.require_heldout_provenance:
        # A reported generalization score must be genuinely held out.
        constraints["heldout_provenance_ok"] = (
            metrics.heldout_score is None or metrics.validation_source == "held_out"
        )
    if thresholds.require_heldout_endpoint:
        # The selection metric itself must come from a held-out evaluation.
        constraints["heldout_endpoint_ok"] = metrics.validation_source == "held_out"
    return constraints


def constraints_satisfied(constraints: Mapping[str, bool]) -> bool:
    return all(bool(v) for v in constraints.values())


def constrained_score_from_constraints(
    metrics: CREDOTrialMetrics,
    constraints: Mapping[str, bool],
    *,
    standardizer: Optional[Standardizer] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Pruner score penalized by an already-computed (possibly augmented) constraint
    set. Use this so run_credo_trial and external wrappers share one scoring path
    that honors constraints (e.g. ``no_failure``) added outside ``hard_constraints``."""
    score = pruner_score(metrics, weights=weights, standardizer=standardizer)
    if not constraints_satisfied(constraints):
        score += DIVERGENCE_PENALTY
    return score


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


def _add_if_finite(vector: dict[str, float], name: str, value: Optional[float]) -> None:
    if value is not None and math.isfinite(float(value)):
        vector[name] = float(value)


def _headline_endpoint(metrics: CREDOTrialMetrics) -> tuple[str, Optional[float]]:
    """The endpoint metric used for scoring/feasibility: the combined geom-mass
    proxy when finite, else the pure-geometry term, matching objective_vector and
    hard_constraints. Returns (name, value); value is None when neither is finite."""
    if _finite(metrics.endpoint_geom_mass):
        return "endpoint_geom_mass", float(metrics.endpoint_geom_mass)
    if _finite(metrics.endpoint_sinkhorn):
        return "endpoint_sinkhorn", float(metrics.endpoint_sinkhorn)
    return "endpoint_geom_mass", None


def _gap_ok(gap: Optional[float], max_value: float, required: bool) -> bool:
    """Diagnostic-gap feasibility. When the diagnostic is required, a missing
    value (``None``) is infeasible; otherwise a missing value passes (screening)."""
    if gap is None:
        return not required
    return float(gap) <= max_value


def _branch_particle_constraints(
    metrics: CREDOTrialMetrics,
    thresholds: ConstraintThresholds,
) -> dict[str, bool]:
    required = thresholds.require_branch_particle_diagnostics
    return {
        "source_ess_ok": _optional_floor_ok(metrics.source_ess_frac, thresholds.ess_floor, required),
        "factual_terminal_ess_ok": _optional_floor_ok(
            metrics.factual_terminal_ess_frac, thresholds.ess_floor, required
        ),
        "reference_terminal_ess_ok": _optional_floor_ok(
            metrics.reference_terminal_ess_frac, thresholds.ess_floor, required
        ),
        "factual_min_ess_ok": _optional_floor_ok(
            metrics.factual_min_ess_frac_over_time, thresholds.ess_floor, required
        ),
        "reference_min_ess_ok": _optional_floor_ok(
            metrics.reference_min_ess_frac_over_time, thresholds.ess_floor, required
        ),
        "factual_max_weight_ok": _optional_ceiling_ok(
            metrics.factual_max_weight_frac, thresholds.max_weight_ceiling, required
        ),
        "reference_max_weight_ok": _optional_ceiling_ok(
            metrics.reference_max_weight_frac, thresholds.max_weight_ceiling, required
        ),
        "factual_logw_range_finite": _optional_finite_ok(metrics.factual_logw_range, required),
        "reference_logw_range_finite": _optional_finite_ok(metrics.reference_logw_range, required),
    }


def _optional_floor_ok(value: Optional[float], floor: float, required: bool) -> bool:
    if not _finite(value):
        return not required
    return float(value) >= floor


def _optional_ceiling_ok(value: Optional[float], ceiling: float, required: bool) -> bool:
    if not _finite(value):
        return not required
    return float(value) <= ceiling


def _optional_finite_ok(value: Optional[float], required: bool) -> bool:
    return _finite(value) or not required


__all__ = [
    "CLAIM_GRADE_PRESENCE_THRESHOLDS",
    "CLAIM_GRADE_THRESHOLDS",
    "ConstraintThresholds",
    "DEFAULT_PRUNER_WEIGHTS",
    "DEFAULT_THRESHOLDS",
    "SearchProfile",
    "Standardizer",
    "claim_grade_thresholds",
    "constrained_score_from_constraints",
    "constraints_satisfied",
    "feasible_pruner_score",
    "hard_constraints",
    "objective_vector",
    "pruner_score",
    "threshold_metadata",
    "thresholds_for_profile",
]
