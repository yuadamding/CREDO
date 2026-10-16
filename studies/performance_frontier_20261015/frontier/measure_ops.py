"""Differentiable finite-measure algebra and bounded-memory shape primitives.

These functions do not select calibrations, inspect outcomes, or authorize data
access. Each caller must supply the prefix-specific integrals and fixed support.
"""

import math
from typing import Sequence

import torch
from torch.utils.checkpoint import checkpoint


def nonnegative_score(raw, *, component_absolute_sum):
    """Explicit FP64 score-only roundoff record, never a training-loss clamp.

    A nonnegative mathematical statistic can cancel to a tiny negative value.
    Retain that value and its64-epsilon absolute-component bound. Anything below
    the bound fails; positive values are unchanged. This is not a probability
    floor and must be persisted beside the reported risk.
    """
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 0 or raw.dtype != torch.float64
            or not bool(torch.isfinite(raw)) or not math.isfinite(component_absolute_sum)
            or component_absolute_sum < 0):
        raise ValueError("Finite FP64 score and nonnegative component scale required")
    value = float(raw.detach().cpu())
    bound = 64 * torch.finfo(torch.float64).eps * max(1.0, component_absolute_sum)
    if value < -bound:
        raise ValueError("Negative risk exceeds the declared FP64 roundoff bound")
    return {"risk": max(0.0, value), "raw_risk": value,
            "negative_roundoff_corrected": value < 0, "roundoff_bound": bound,
            "training_loss_modified": False}


def _finite(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"Floating tensor required: {name}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")


def particle_integrals(log_multiplier, log_base_weight=None):
    """Return normalized log weights and log mass, preserving gradients.

    The last axis is particles. Uniform source proposal weights are the
    default; importance corrections (including mixture proposal corrections)
    belong in log_multiplier. Never truncate previously normalized weights.
    """
    _finite(log_multiplier, "particle log multipliers")
    if log_multiplier.ndim < 1 or log_multiplier.shape[-1] < 1:
        raise ValueError("Nonempty particle axis required")
    if log_base_weight is None:
        raw = log_multiplier - math.log(log_multiplier.shape[-1])
    else:
        _finite(log_base_weight, "base log weights")
        if log_base_weight.shape != log_multiplier.shape:
            raise ValueError("Base weights must match the full particle shape")
        if not torch.allclose(torch.logsumexp(log_base_weight, -1),
                              torch.zeros_like(log_base_weight[..., 0]), atol=1e-6, rtol=0):
            raise ValueError("Base integration weights must sum to one")
        raw = log_multiplier + log_base_weight
    log_mass = torch.logsumexp(raw, -1)
    return raw - log_mass.unsqueeze(-1), log_mass


def source_log_frequencies(counts, smoothing=0.5):
    if (not isinstance(counts, torch.Tensor) or counts.ndim != 1 or not counts.numel()
            or counts.dtype not in (torch.int32, torch.int64) or bool((counts < 0).any())):
        raise ValueError("Complete nonnegative integer count vector required")
    if not math.isfinite(smoothing) or smoothing <= 0 or counts.sum().item() <= 0:
        raise ValueError("Positive smoothing and eligible captured total required")
    mass = counts.to(torch.float64) + smoothing
    return mass.log() - mass.sum().log()


def catalog_log_frequencies(log_source, modeled_indices, log_mass):
    """Exactly one full-catalog normalizer; complement mass multipliers are1."""
    _check_log_distribution(log_source, "source frequencies")
    _finite(log_mass, "guide log masses")
    if (modeled_indices.dtype != torch.int64 or modeled_indices.ndim != 1
            or log_mass.shape != modeled_indices.shape
            or modeled_indices.device != log_source.device or log_mass.device != log_source.device
            or modeled_indices.unique().numel() != modeled_indices.numel()
            or bool(((modeled_indices < 0) | (modeled_indices >= log_source.numel())).any())):
        raise ValueError("Distinct in-catalog modeled indices and matching log masses required")
    increments = torch.zeros_like(log_source).scatter(0, modeled_indices, log_mass.to(log_source.dtype))
    raw = log_source + increments
    return raw - torch.logsumexp(raw, 0)


def _check_log_distribution(values, name):
    _finite(values, name)
    if values.ndim != 1 or not values.numel():
        raise ValueError(f"One nonempty complete distribution required: {name}")
    tolerance = 1e-10 if values.dtype == torch.float64 else 2e-6
    if abs(torch.logsumexp(values, 0).item()) > tolerance:
        raise ValueError(f"Unnormalized {name}")


def calibrate_abundance(log_source, log_prediction, alpha):
    _check_log_distribution(log_source, "source probabilities")
    _check_log_distribution(log_prediction, "predicted probabilities")
    if (log_prediction.shape != log_source.shape or log_prediction.device != log_source.device
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("Matching catalog and alpha in[0,1] required")
    raw = log_source + alpha * (log_prediction - log_source)
    return raw - torch.logsumexp(raw, 0)


def calibrate_dispersion(states, normalized_log_weights, scale):
    _finite(states, "states")
    _check_log_distribution(normalized_log_weights, "conditional weights")
    if (states.ndim != 2 or states.shape[0] != normalized_log_weights.numel()
            or states.device != normalized_log_weights.device
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("Matching states and positive scale required")
    weights = normalized_log_weights.exp().to(states.dtype)
    mean = torch.sum(states * weights[:, None], 0, keepdim=True)
    return mean + scale * (states - mean)


def round_robin_component_counts(total, components=3):
    if type(total) is not int or type(components) is not int or components < 1 or total < components:
        raise ValueError("Every ensemble component needs a positive particle count")
    return tuple(total // components + int(index < total % components) for index in range(components))


def ensemble_catalog(log_catalogs):
    if not log_catalogs:
        raise ValueError("At least one full catalog required")
    for values in log_catalogs:
        _check_log_distribution(values, "component catalog")
    if len({tuple(values.shape) for values in log_catalogs}) != 1:
        raise ValueError("Component catalogs differ")
    return torch.logsumexp(torch.stack(tuple(log_catalogs)), 0) - math.log(len(log_catalogs))


def ensemble_guide(states: Sequence[torch.Tensor], log_weights: Sequence[torch.Tensor],
                   component_log_frequencies):
    """Equal finite-measure mixture, NOT equal within-guide state mixture.

    Components already have their fixed calibration applied. Their guide
    frequencies must come from complete catalogs recomputed at the respective
    component prefixes; unequal particle counts never change the1/K mixture.
    """
    _finite(component_log_frequencies, "component guide frequencies")
    if (not states or len(states) != len(log_weights)
            or component_log_frequencies.shape != (len(states),)
            or bool((component_log_frequencies > 1e-12).any())):
        raise ValueError("One positive-frequency component per particle law required")
    dimensions = set()
    for x, logw in zip(states, log_weights):
        _finite(x, "component states")
        _check_log_distribution(logw, "component conditional weights")
        if x.ndim != 2 or x.shape[0] != logw.numel():
            raise ValueError("Component law shape mismatch")
        dimensions.add(x.shape[1])
    if len(dimensions) != 1:
        raise ValueError("Component state axes differ")
    log_share = component_log_frequencies - torch.logsumexp(component_log_frequencies, 0)
    combined = torch.cat([logw + log_share[index] for index, logw in enumerate(log_weights)])
    log_frequency = torch.logsumexp(component_log_frequencies, 0) - math.log(len(states))
    return torch.cat(tuple(states)), combined, log_frequency


def weighted_multiscale_kernel(x, wx, y, wy, bandwidths, *, block=128, recompute_backward=True):
    """Gaussian mixture kernel in RMS-Euclidean distance, weighted V estimate.

    Supports autograd and bounded pairwise blocks; no full NxM allocation.
    Training recomputes pure block operations during backward rather than
    retaining every distance/kernel graph. Actual peak memory still requires
    runtime qualification. Scoring should supply FP64 tensors under no_grad.
    Bandwidths must be fitted/frozen on the allowed training subset by caller.
    """
    for tensor, name in ((x, "x"), (y, "y"), (wx, "wx"), (wy, "wy")):
        _finite(tensor, name)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or x.shape[1] == 0
            or wx.shape != (len(x),) or wy.shape != (len(y),) or min(len(x), len(y)) < 1
            or bool((wx < 0).any()) or bool((wy < 0).any())
            or not torch.allclose(wx.sum(), torch.ones_like(wx.sum()), atol=1e-6, rtol=0)
            or not torch.allclose(wy.sum(), torch.ones_like(wy.sum()), atol=1e-6, rtol=0)):
        raise ValueError("Two probability-weighted laws on the same state axis required")
    if type(block) is not int or block < 1:
        raise ValueError("Positive pairwise block size required")
    bandwidths = tuple(float(value) for value in bandwidths)
    if not bandwidths or any(not math.isfinite(value) or value <= 0 for value in bandwidths):
        raise ValueError("Positive finite prebound kernel bandwidths required")

    def reduce_block(xb, wb, yb, vb):
        distance2 = (xb[:, None, :] - yb[None, :, :]).square().mean(-1)
        kernel = sum(torch.exp(-distance2 / (2 * width * width)) for width in bandwidths) / len(bandwidths)
        return torch.sum(kernel * wb[:, None] * vb[None, :])

    result = x.new_zeros(())
    for start in range(0, len(x), block):
        xb, wb = x[start:start + block], wx[start:start + block]
        for stop in range(0, len(y), block):
            yb, vb = y[stop:stop + block], wy[stop:stop + block]
            if recompute_backward and torch.is_grad_enabled() and any(value.requires_grad for value in (xb, wb, yb, vb)):
                term = checkpoint(reduce_block, xb, wb, yb, vb, use_reentrant=False, preserve_rng_state=False)
            else:
                term = reduce_block(xb, wb, yb, vb)
            result = result + term
    return result


def multiscale_mmd2(x, wx, y, wy, bandwidths, *, block=128, recompute_backward=True):
    """Biased weighted MMD²; retain floating-point residuals, never clip losses."""
    return (weighted_multiscale_kernel(x, wx, x, wx, bandwidths, block=block, recompute_backward=recompute_backward)
            + weighted_multiscale_kernel(y, wy, y, wy, bandwidths, block=block, recompute_backward=recompute_backward)
            - 2 * weighted_multiscale_kernel(x, wx, y, wy, bandwidths, block=block, recompute_backward=recompute_backward))
