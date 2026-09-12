"""Pure, expression-agnostic helpers for the Dev34 G00C selection freeze."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..contracts import G00CRefitSeedRecordV1, G00CRefitSeedScheduleV1

SEED_STREAMS = (
    "initialization",
    "training_sampler",
    "thinning",
    "validation_evaluation",
    "stochastic_optimizer_or_augmentation",
    "restart_interruption_point",
)
FIXED_OTHER_CANDIDATE_REPLAY_DRAWS = (0, 6, 12, 18, 24, 30, 36, 42, 48, 58)


@dataclass(frozen=True)
class CheckpointMultinomialRefit:
    """Closed-form checkpoint-conditioned multinomial fit summary."""

    final_state_hash: str
    validation_total_count: int
    validation_nll_sum: float
    validation_nll_per_count: float
    validation_excess_nll_per_count: float


@dataclass(frozen=True)
class CommonSupportPrefixFit:
    """Prefix-plus-residual fit with the same 0.5 prior per common-support gene."""

    candidate_probabilities: np.ndarray[Any, Any]
    residual_frequencies: np.ndarray[Any, Any] | None
    expanded_probabilities: np.ndarray[Any, Any]


def checkpoint_multinomial_refit(
    training_counts: np.ndarray[Any, Any],
    validation_counts: np.ndarray[Any, Any],
    *,
    pseudocount: float = 0.5,
) -> CheckpointMultinomialRefit:
    """Fit checkpoint intercepts and score excess NLL on the same count surface."""

    training = np.asarray(training_counts, dtype=np.float64)
    validation = np.asarray(validation_counts, dtype=np.float64)
    if (
        training.ndim != 2
        or training.shape != validation.shape
        or training.shape[0] < 1
        or training.shape[1] < 2
        or not np.isfinite(training).all()
        or not np.isfinite(validation).all()
        or np.any(training < 0.0)
        or np.any(validation < 0.0)
        or pseudocount <= 0.0
    ):
        raise ValueError("Checkpoint multinomial counts or pseudocount are invalid.")
    training_totals = training.sum(axis=1, keepdims=True)
    validation_totals = validation.sum(axis=1, keepdims=True)
    if np.any(training_totals <= 0.0) or np.any(validation_totals <= 0.0):
        raise ValueError("Every checkpoint must contain positive training and validation counts.")
    width = training.shape[1]
    probabilities = (training + pseudocount) / (training_totals + pseudocount * width)
    saturated = (validation + pseudocount) / (validation_totals + pseudocount * width)
    nll = float(-np.sum(validation * np.log(probabilities)))
    saturated_nll = float(-np.sum(validation * np.log(saturated)))
    total = int(validation.sum())
    per_count = nll / total
    excess = max((nll - saturated_nll) / total, 0.0)
    state = np.asarray(probabilities, dtype="<f8").tobytes(order="C")
    return CheckpointMultinomialRefit(
        final_state_hash=hashlib.sha256(state).hexdigest(),
        validation_total_count=total,
        validation_nll_sum=nll,
        validation_nll_per_count=per_count,
        validation_excess_nll_per_count=excess,
    )


def checkpoint_multinomial_refit_common_support(
    training_counts: np.ndarray[Any, Any],
    *,
    modeled_features: int,
    per_feature_pseudocount: float = 0.5,
) -> CommonSupportPrefixFit:
    """Fit one prefix while preserving a 0.5 prior for every one of 4,096 genes."""

    training = np.asarray(training_counts, dtype=np.float64)
    if (
        training.ndim != 2
        or training.shape[0] < 1
        or training.shape[1] != 4096
        or not 1 <= modeled_features <= 4096
        or not np.isfinite(training).all()
        or np.any(training < 0.0)
        or per_feature_pseudocount != 0.5
    ):
        raise ValueError("Common-support training counts or prior are invalid.")
    smoothed = training + per_feature_pseudocount
    denominator = smoothed.sum(axis=1, keepdims=True)
    if modeled_features == 4096:
        expanded = smoothed / denominator
        return CommonSupportPrefixFit(
            candidate_probabilities=expanded.copy(),
            residual_frequencies=None,
            expanded_probabilities=expanded,
        )
    primary = smoothed[:, :modeled_features]
    omitted = smoothed[:, modeled_features:]
    omitted_total = omitted.sum(axis=1, keepdims=True)
    residual_frequencies = omitted / omitted_total
    candidate = np.concatenate((primary, omitted_total), axis=1) / denominator
    expanded = expand_common_support_probabilities(
        candidate,
        residual_frequencies,
        reference_feature_count=4096,
    )
    if np.any(expanded <= 0.0):
        raise ValueError("The per-feature common-support prior must yield positive probabilities.")
    return CommonSupportPrefixFit(
        candidate_probabilities=candidate,
        residual_frequencies=residual_frequencies,
        expanded_probabilities=expanded,
    )


def derive_refit_seed_schedule(
    derivation_namespace_id: str,
    *,
    fold_id: Literal["lodo-D1"] = "lodo-D1",
    stage: Literal["g00c_feature_and_cell_selection"] = ("g00c_feature_and_cell_selection"),
) -> G00CRefitSeedScheduleV1:
    """Expand all 59 paired-draw seed streams before any result is observed."""

    if len(derivation_namespace_id) != 64 or any(
        character not in "0123456789abcdef" for character in derivation_namespace_id
    ):
        raise ValueError("Seed derivation namespace must be a lowercase SHA-256 digest.")
    records: list[G00CRefitSeedRecordV1] = []
    for draw_id in range(59):
        values = {}
        for stream in SEED_STREAMS:
            material = "|".join(
                (derivation_namespace_id, fold_id, stage, str(draw_id), stream)
            ).encode("utf-8")
            values[stream] = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        records.append(G00CRefitSeedRecordV1(draw_id=draw_id, **values))
    payload: dict[str, Any] = {
        "schedule_id": "pending",
        "derivation_namespace_id": derivation_namespace_id,
        "fold_id": fold_id,
        "stage": stage,
        "records": tuple(records),
    }
    provisional = G00CRefitSeedScheduleV1.model_construct(**payload)
    payload["schedule_id"] = provisional.identity(id_field="schedule_id")
    return G00CRefitSeedScheduleV1.model_validate(payload)


def expand_common_support_probabilities(
    candidate_probabilities: np.ndarray[Any, Any],
    residual_frequencies: np.ndarray[Any, Any] | None,
    *,
    reference_feature_count: int = 4096,
) -> np.ndarray[Any, Any]:
    """Expand a prefix-plus-residual distribution onto one common feature support."""

    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if candidate.ndim != 2 or not np.isfinite(candidate).all() or np.any(candidate < 0.0):
        raise ValueError("Candidate probabilities must be a finite nonnegative matrix.")
    if not np.allclose(candidate.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("Candidate probabilities must sum to one per checkpoint.")
    candidate_width = candidate.shape[1]
    if candidate_width == reference_feature_count:
        if residual_frequencies is not None:
            raise ValueError("The reference candidate cannot bind residual frequencies.")
        return candidate.copy()
    modeled_features = candidate_width - 1
    omitted_features = reference_feature_count - modeled_features
    residual = np.asarray(residual_frequencies, dtype=np.float64)
    if (
        modeled_features < 1
        or omitted_features < 1
        or residual.shape != (candidate.shape[0], omitted_features)
        or not np.isfinite(residual).all()
        or np.any(residual <= 0.0)
        or not np.allclose(residual.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Residual frequencies do not match the omitted common support.")
    expanded = np.concatenate(
        (candidate[:, :modeled_features], candidate[:, -1:] * residual), axis=1
    )
    if expanded.shape[1] != reference_feature_count or not np.allclose(
        expanded.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Expanded probabilities do not preserve the common support mass.")
    return expanded


def weighted_multinomial_nll_per_count(
    validation_counts: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any],
    inverse_probability_weights: np.ndarray[Any, Any],
) -> float:
    """Score every candidate on the same weighted count denominator."""

    counts = np.asarray(validation_counts, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    weights = np.asarray(inverse_probability_weights, dtype=np.float64)
    if (
        counts.ndim != 2
        or counts.shape != probs.shape
        or weights.shape != (counts.shape[0],)
        or not np.isfinite(counts).all()
        or not np.isfinite(probs).all()
        or not np.isfinite(weights).all()
        or np.any(counts < 0.0)
        or np.any(probs <= 0.0)
        or np.any(weights <= 0.0)
        or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Common-support counts, probabilities, or weights are invalid.")
    weighted_counts = counts * weights[:, None]
    denominator = float(weighted_counts.sum())
    if denominator <= 0.0:
        raise ValueError("Common-support weighted validation count must be positive.")
    return float(-np.sum(weighted_counts * np.log(probs)) / denominator)


def feature_selection_decision_v3(
    *,
    candidates: tuple[int, ...],
    paired_absolute_difference_q95: np.ndarray[Any, Any],
    support_eligible: tuple[bool, ...],
    epsilon: float = 0.0001,
    reference_zero_tolerance: float = 1e-12,
) -> int:
    """Select the smallest supported prefix after checking the 4,096 reference."""

    expected = (256, 512, 1024, 2048, 4096)
    values = np.asarray(paired_absolute_difference_q95, dtype=np.float64)
    if (
        candidates != expected
        or len(support_eligible) != len(expected)
        or values.shape != (len(expected),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or epsilon != 0.0001
        or not 0.0 < reference_zero_tolerance <= 1e-10
        or values[-1] > reference_zero_tolerance
        or not support_eligible[-1]
    ):
        raise ValueError("Feature-selection evidence differs from the frozen Dev35 contract.")
    qualifying = [
        candidate
        for candidate, value, supported in zip(candidates, values, support_eligible, strict=True)
        if supported and value <= epsilon
    ]
    if not qualifying:
        raise ValueError("No supported feature candidate qualifies, including the reference.")
    return min(qualifying)


def sample_size_decision_v3(
    *,
    grid_stage: Literal["base", "extension"],
    candidates: tuple[int, ...],
    paired_absolute_difference_q95: np.ndarray[Any, Any],
    support_eligible: tuple[bool, ...],
    epsilon: float = 0.0001,
    reference_zero_tolerance: float = 1e-12,
) -> tuple[Literal["selected", "extension_required", "fail_no_saturation"], int | None]:
    """Apply the fail-closed base/extension saturation rule."""

    base = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    expected = base if grid_stage == "base" else (*base, 2_000_000)
    values = np.asarray(paired_absolute_difference_q95, dtype=np.float64)
    if (
        candidates != expected
        or len(support_eligible) != len(expected)
        or values.shape != (len(expected),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or epsilon != 0.0001
        or not 0.0 < reference_zero_tolerance <= 1e-10
        or values[-1] > reference_zero_tolerance
        or not support_eligible[-1]
    ):
        raise ValueError("Sample-size decision inputs differ from the frozen Dev35 contract.")
    qualifying = [
        candidate
        for candidate, value, supported in zip(
            candidates[:-1], values[:-1], support_eligible[:-1], strict=True
        )
        if supported and value <= epsilon
    ]
    if qualifying:
        return "selected", min(qualifying)
    if grid_stage == "base":
        return "extension_required", None
    return "fail_no_saturation", None
