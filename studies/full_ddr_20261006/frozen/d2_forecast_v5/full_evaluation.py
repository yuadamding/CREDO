"""Bounded-memory numerical scores for the new full-DDR study.

These are array-level primitives, not an outcome-access or selection authority.
The worker must enforce forecast sealing, phase roles, common support, frozen
feature identities and the full catalog before providing arrays. CPU calls are
for synthetic qualification; production execution requires its own H100 gate.
No routine loads cells, fits a model, changes a forecast, or chooses a winner.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from scipy.special import logsumexp


def _nonnegative(value: float, name: str) -> tuple[float, list[dict]]:
    if not math.isfinite(value) or value < -1e-10:
        raise ValueError(f"{name} is nonfinite or negative beyond the registered numerical tolerance.")
    return (0.0, [dict(metric=name, original=value, canonical=0.0)]) if value < 0 else (value, [])


def _normalized_log_weights(values, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError("Finite log weights must match the particle axis.")
    probability = np.exp(values - logsumexp(values))
    if np.any(probability <= 0) or not np.isfinite(probability).all():
        raise ValueError("Particle underflow is unresolved, not a fabricated zero.")
    return probability


def _support(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} requires finite nonempty [cells, features] support.")
    return values


def _pair_mean(x, p, y, q, *, block_size: int, self_pair: bool):
    """No cell-by-cell host sync and at most one B-by-B distance tensor."""
    total = x.new_zeros(())
    for i in range(0, len(x), block_size):
        for j in range(i if self_pair else 0, len(y), block_size):
            distance = torch.cdist(x[i:i + block_size], y[j:j + block_size], p=2.0,
                                  compute_mode="donot_use_mm_for_euclid_dist")
            term = p[i:i + block_size] @ (distance @ q[j:j + block_size])
            total.add_(term if not self_pair or i == j else 2 * term)
            del distance  # Release the preceding tile before allocating the next.
    return total


def weighted_v_energy(predicted, predicted_log_weight, observed, observed_weights, *,
                      block_size: int = 256, device: str = "cuda") -> dict:
    """Exact weighted biased V-energy using RMS-Euclidean latent distances.

    V self terms include their zero diagonals; this is not the U statistic.
    FP64 direct differences avoid the matrix-product distance shortcut. Tiling
    bounds pairwise storage, not O(P**2 + P*N + N**2) arithmetic. No support
    subsampling or feature projection is used. Tiny negative roundoff is recorded.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("A positive integer distance-tile bound is required.")
    x, y = _support(predicted, "predicted"), _support(observed, "observed")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Predicted and observed state axes disagree.")
    p = _normalized_log_weights(predicted_log_weight, len(x))
    q = np.asarray(observed_weights, dtype=np.float64)
    if q.shape != (len(y),) or not np.isfinite(q).all() or np.any(q < 0) or not q.sum() > 0:
        raise ValueError("Observed weights require finite nonnegative positive-total sampling mass.")
    if not math.isfinite(float(q.sum())):
        raise ValueError("Observed weight normalization overflowed.")
    q = q / q.sum()
    # Remove one shared origin, not distinct centroids, which would erase signal.
    origin = x[0].copy()
    x, y = x - origin, y - origin
    scale = math.sqrt(x.shape[1])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Centered state differences overflowed.")
    with torch.inference_mode():
        tx = torch.tensor(x, dtype=torch.float64, device=device)
        ty = torch.tensor(y, dtype=torch.float64, device=device)
        tp = torch.tensor(p, dtype=torch.float64, device=device)
        tq = torch.tensor(q, dtype=torch.float64, device=device)
        cross = _pair_mean(tx, tp, ty, tq, block_size=block_size, self_pair=False) / scale
        self_predicted = _pair_mean(tx, tp, tx, tp, block_size=block_size, self_pair=True) / scale
        self_observed = _pair_mean(ty, tq, ty, tq, block_size=block_size, self_pair=True) / scale
        raw = float(2 * cross - self_predicted - self_observed)
        value, corrections = _nonnegative(raw, "weighted_V_energy_RMS")
        return dict(energy=value, raw_energy=raw, cross_mean=float(cross),
                    predicted_self_mean=float(self_predicted), observed_self_mean=float(self_observed),
                    corrections=corrections, metric="weighted_biased_v_energy",
                    distance="RMS_Euclidean", dtype="float64", block_size=block_size,
                    predicted_cells=len(x), observed_cells=len(y), state_dimension=x.shape[1],
                    largest_distance_tile_bytes=8 * min(block_size, max(len(x), len(y))) ** 2)


def categorical_scores(catalog_log_probability, raw_endpoint_counts) -> dict:
    """Complete supplied catalog CE, entropy, CE-minus-H and quadratic score.

    Caller binds the exact ordered 26,504 categories. Observed zeros remain
    zeros, and no target-panel normalization or endpoint pseudocount is applied.
    Forecasts must already be normalized, finite and strictly positive in FP64.
    """
    counts = np.asarray(raw_endpoint_counts)
    logp = np.asarray(catalog_log_probability, dtype=np.float64)
    if counts.ndim != 1 or not len(counts) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("A complete nonnegative integer count vector is required.")
    total = sum(int(n) for n in counts)
    if total <= 0 or not math.isfinite(float(total)):
        raise ValueError("A positive finite complete-capture total is required.")
    if logp.shape != counts.shape or not np.isfinite(logp).all():
        raise ValueError("Finite catalog log probabilities must match the complete count axis.")
    if abs(float(logsumexp(logp))) > 1e-12:
        raise ValueError("Forecast is not normalized on the declared complete catalog.")
    forecast = np.exp(logp)
    if np.any(forecast <= 0):
        raise ValueError("Underflowed probability is unavailable, not an observed zero.")
    observed = counts.astype(np.float64) / total
    positive = observed > 0
    entropy = -float(observed[positive] @ np.log(observed[positive]))
    ce = -float(observed @ logp)
    kl, corrections = _nonnegative(ce - entropy, "CE_minus_H")
    return dict(cross_entropy_nats=ce, endpoint_entropy_nats=entropy, ce_minus_entropy_nats=kl,
                quadratic_score=float(1 + forecast @ forecast - 2 * observed @ forecast),
                catalog_size=len(counts), captured_cells=total, endpoint_zero_categories=int((counts == 0).sum()),
                corrections=corrections)


def _composition(values, *, dimension: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or (dimension is not None and len(values) != dimension):
        raise ValueError("Exact one-dimensional gene axis required.")
    if not np.isfinite(values).all() or np.any(values < 0) or not math.isclose(
        float(values.sum()), 1.0, abs_tol=1e-6, rel_tol=0
    ):
        raise ValueError("A finite normalized composition is required; no smoothing or repair.")
    return values


def endpoint_effect_decomposition(predicted_target, predicted_control, observed_target, observed_control, *,
                                  gene_weights=None) -> dict:
    """Separate own-control/target endpoints and their contrast at h(q)=log1p(1e4q)."""
    target = _composition(predicted_target)
    control = _composition(predicted_control, dimension=len(target))
    truth = _composition(observed_target, dimension=len(target))
    reference = _composition(observed_control, dimension=len(target))
    weights = np.ones(len(target), dtype=np.float64) if gene_weights is None else np.asarray(gene_weights, dtype=np.float64)
    if weights.shape != target.shape or not np.isfinite(weights).all() or np.any(weights < 0) or not weights.sum() > 0:
        raise ValueError("Finite nonnegative positive-total gene weights on the exact axis are required.")
    if not math.isfinite(float(weights.sum())):
        raise ValueError("Gene-weight normalization overflowed.")
    weights = weights / weights.sum()
    target_error = np.log1p(1e4 * target) - np.log1p(1e4 * truth)
    control_error = np.log1p(1e4 * control) - np.log1p(1e4 * reference)
    effect_error = target_error - control_error
    common_error = (target_error + control_error) / 2
    observed_effect = np.log1p(1e4 * truth) - np.log1p(1e4 * reference)
    target_mse = float(weights @ np.square(target_error))
    control_mse = float(weights @ np.square(control_error))
    interaction = float(-2 * weights @ (target_error * control_error))
    effect_mse = float(weights @ np.square(effect_error))
    return dict(target_endpoint_mse=target_mse, own_control_endpoint_mse=control_mse,
                interaction=interaction, effect_mse=effect_mse,
                common_endpoint_component_mse=float(weights @ np.square(common_error)),
                zero_effect_mse=float(weights @ np.square(observed_effect)),
                decomposition_residual=effect_mse - (target_mse + control_mse + interaction),
                gene_dimension=len(target))


def decoded_mean_composition(states, log_weights, decode: Callable, *, gene_dimension: int,
                             batch_rows: int = 512, device: str = "cuda") -> np.ndarray:
    """Weighted composition before logarithms, with bounded decoder batches.

    ``decode`` must be the phase-qualified frozen R48 composition forward, not
    transformed expression or a fitted correction. Its input derivatives are
    irrelevant here: this is inference-only scoring, not a training objective.
    This routine does not certify the decoder identity or outcome access.
    """
    states = _support(states, "decoder source")
    if type(gene_dimension) is not int or gene_dimension < 1 or type(batch_rows) is not int or batch_rows < 1:
        raise ValueError("Positive fixed gene dimension and decoder-batch bound required.")
    probability = _normalized_log_weights(log_weights, len(states))
    result = np.zeros(gene_dimension, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(states), batch_rows):
            block = torch.tensor(states[start:start + batch_rows], dtype=torch.float32, device=device)
            decoded = decode(block)
            if not torch.is_tensor(decoded):
                raise ValueError("Frozen decoder must return a composition tensor.")
            values = decoded.detach().double().cpu().numpy()
            if values.shape != (len(block), gene_dimension) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Decoded composition shape or finite/nonnegative requirement failed.")
            if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
                raise ValueError("Decoder output is not normalized composition; do not repair it.")
            result += probability[start:start + len(block)] @ values
    return _composition(result, dimension=gene_dimension)
