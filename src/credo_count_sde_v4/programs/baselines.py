"""Frozen Dev40-B count-prediction baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import BaselineName
from ..errors import ContractError


@dataclass(frozen=True)
class BaselinePrediction:
    """Mean-count prediction for one frozen baseline."""

    name: BaselineName
    mean: np.ndarray


def _frequency(counts: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    totals = counts.sum(axis=0, dtype=np.float64) + pseudocount
    return totals / totals.sum()


def _conditional_frequencies(
    counts: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_levels: int,
    second_levels: int,
    fallback: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    result = np.empty((first_levels, second_levels, counts.shape[1]), dtype=np.float64)
    for left in range(first_levels):
        for right in range(second_levels):
            mask = (first == left) & (second == right)
            if np.any(mask):
                totals = counts[mask].sum(axis=0, dtype=np.float64)
                result[left, right] = (totals + shrinkage * fallback) / (totals.sum() + shrinkage)
            else:
                result[left, right] = fallback
    return result


def _validate_inputs(
    training_counts: np.ndarray,
    training_library_size: np.ndarray,
    training_checkpoint: np.ndarray,
    training_target: np.ndarray,
    training_guide: np.ndarray,
    evaluation_library_size: np.ndarray,
    evaluation_checkpoint: np.ndarray,
    evaluation_target: np.ndarray,
) -> tuple[int, int]:
    counts = np.asarray(training_counts)
    rows = counts.shape[0]
    if counts.ndim != 2 or rows == 0 or counts.shape[1] < 2:
        raise ContractError("Baseline training counts must be a nonempty matrix.")
    for value in (
        training_library_size,
        training_checkpoint,
        training_target,
        training_guide,
    ):
        if np.asarray(value).shape != (rows,):
            raise ContractError("Baseline training labels do not align with counts.")
    evaluation_rows = len(evaluation_library_size)
    if np.asarray(evaluation_checkpoint).shape != (evaluation_rows,) or np.asarray(
        evaluation_target
    ).shape != (evaluation_rows,):
        raise ContractError("Baseline evaluation labels do not align.")
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ContractError("Baseline counts must be finite and nonnegative.")
    if not np.allclose(counts, np.floor(counts), atol=0, rtol=0):
        raise ContractError("Baseline likelihood requires integer counts.")
    if not np.allclose(counts.sum(axis=1), training_library_size, atol=0, rtol=0):
        raise ContractError("Baseline library offsets must equal raw-count totals.")
    checkpoints = max(int(np.max(training_checkpoint)), int(np.max(evaluation_checkpoint))) + 1
    targets = max(int(np.max(training_target)), int(np.max(evaluation_target))) + 1
    return checkpoints, targets


def fit_frozen_baselines(
    *,
    training_counts: np.ndarray,
    training_library_size: np.ndarray,
    training_checkpoint: np.ndarray,
    training_target: np.ndarray,
    training_guide: np.ndarray,
    control_guide_indices: tuple[int, ...],
    evaluation_library_size: np.ndarray,
    evaluation_checkpoint: np.ndarray,
    evaluation_target: np.ndarray,
    sparse_factor_rank: int,
) -> tuple[BaselinePrediction, ...]:
    """Fit every comparator from training rows and predict evaluation means."""

    counts = np.asarray(training_counts, dtype=np.float64)
    library = np.asarray(evaluation_library_size, dtype=np.float64)
    train_checkpoint = np.asarray(training_checkpoint, dtype=np.int64)
    train_target = np.asarray(training_target, dtype=np.int64)
    train_guide = np.asarray(training_guide, dtype=np.int64)
    eval_checkpoint = np.asarray(evaluation_checkpoint, dtype=np.int64)
    eval_target = np.asarray(evaluation_target, dtype=np.int64)
    checkpoints, targets = _validate_inputs(
        counts,
        np.asarray(training_library_size),
        train_checkpoint,
        train_target,
        train_guide,
        library,
        eval_checkpoint,
        eval_target,
    )
    if sparse_factor_rank <= 0:
        raise ContractError("Sparse-factor baseline rank must be positive.")
    global_frequency = _frequency(counts)
    global_mean = library[:, None] * global_frequency

    control_mask = np.isin(train_guide, np.asarray(control_guide_indices, dtype=np.int64))
    control_checkpoint = _conditional_frequencies(
        counts[control_mask],
        train_checkpoint[control_mask],
        np.zeros(control_mask.sum(), dtype=np.int64),
        first_levels=checkpoints,
        second_levels=1,
        fallback=global_frequency,
        shrinkage=50.0,
    )[:, 0]
    control_mean = library[:, None] * control_checkpoint[eval_checkpoint]

    target_checkpoint = _conditional_frequencies(
        counts,
        train_target,
        train_checkpoint,
        first_levels=targets,
        second_levels=checkpoints,
        fallback=global_frequency,
        shrinkage=100.0,
    )
    pseudobulk_mean = library[:, None] * target_checkpoint[eval_target, eval_checkpoint]

    target_average = _conditional_frequencies(
        counts,
        train_target,
        np.zeros(len(counts), dtype=np.int64),
        first_levels=targets,
        second_levels=1,
        fallback=global_frequency,
        shrinkage=100.0,
    )[:, 0]
    target_average_mean = library[:, None] * target_average[eval_target]

    # Per-gene DE is a shrunk target/checkpoint log ratio against control.
    log_ratio = np.log(target_checkpoint + 1e-12) - np.log(control_checkpoint[None] + 1e-12)
    log_ratio = np.clip(log_ratio, -4.0, 4.0)
    de_frequency = control_checkpoint[None] * np.exp(log_ratio)
    de_frequency /= de_frequency.sum(axis=2, keepdims=True)
    de_mean = library[:, None] * de_frequency[eval_target, eval_checkpoint]

    # GSFA-style comparator: sparse SVD of target/checkpoint log-frequency effects.
    effect_matrix = log_ratio.reshape(targets * checkpoints, counts.shape[1])
    left, singular, right = np.linalg.svd(effect_matrix, full_matrices=False)
    rank = min(sparse_factor_rank, len(singular))
    loadings = right[:rank].copy()
    threshold = np.quantile(np.abs(loadings), 0.7)
    loadings[np.abs(loadings) < threshold] = 0.0
    reconstruction = (left[:, :rank] * singular[:rank]) @ loadings
    gsfa_frequency = control_checkpoint[None] * np.exp(
        np.clip(reconstruction.reshape(targets, checkpoints, -1), -4.0, 4.0)
    )
    gsfa_frequency /= gsfa_frequency.sum(axis=2, keepdims=True)
    gsfa_mean = library[:, None] * gsfa_frequency[eval_target, eval_checkpoint]

    means = {
        BaselineName.EXACT_GLOBAL_GENE_FREQUENCY: global_mean,
        BaselineName.PSEUDOBULK_NEGATIVE_BINOMIAL: pseudobulk_mean,
        BaselineName.PER_GENE_DIFFERENTIAL_EXPRESSION: de_mean,
        BaselineName.GSFA_STYLE_SPARSE_FACTORS: gsfa_mean,
        BaselineName.TARGET_AVERAGE: target_average_mean,
        BaselineName.CONTROL_ONLY: control_mean,
    }
    return tuple(BaselinePrediction(name=name, mean=means[name]) for name in BaselineName)
