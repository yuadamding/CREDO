"""New pooled-NTC companion primitives, distinct from historical primary effects.

Observed means are cell-composition means over positive-RNA support, not summed
count pseudobulks, captured-guide frequencies or absolute population quantity.
Pool in composition space before log1p. The runtime owner binds actual target
and pooled-control populations; these numerical functions cannot prove custody.
"""

import numpy as np
from scipy import sparse
import torch


class CellCompositionAccumulator:
    """Streaming sparse mean of per-cell RNA compositions on one fixed axis."""

    def __init__(self, feature_count, *, maximum_batch_rows=2048):
        if type(feature_count) is not int or feature_count < 1 or type(maximum_batch_rows) is not int or maximum_batch_rows < 1:
            raise ValueError("Positive fixed gene axis and bounded row batch required")
        self.feature_count = feature_count
        self.maximum_batch_rows = maximum_batch_rows
        self.weighted_sum = np.zeros(feature_count, dtype=np.float64)
        self.total_positive_weight = 0.0
        self.captured_rows = self.positive_RNA_rows = self.zero_RNA_rows = 0

    def update(self, counts, cell_weights=None):
        if (not sparse.issparse(counts) or counts.format != "csr" or counts.ndim != 2
                or counts.shape[1] != self.feature_count or not 0 < counts.shape[0] <= self.maximum_batch_rows
                or counts.dtype.kind not in "iu" or np.any(counts.data < 0)
                or not counts.has_canonical_format):
            raise ValueError("Bounded canonical nonnegative integer RNA-only CSR batch required")
        weights = np.ones(counts.shape[0], dtype=np.float64) if cell_weights is None else np.asarray(cell_weights, dtype=np.float64)
        if weights.shape != (counts.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Finite nonnegative sampling weights, not guide frequencies, required")
        depth = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
        if not np.isfinite(depth).all():
            raise ValueError("Nonfinite RNA depth")
        positive = depth > 0
        factors = np.zeros_like(depth)
        factors[positive] = weights[positive] / depth[positive]
        contribution = np.asarray(counts.T @ factors).ravel()
        if not np.isfinite(contribution).all():
            raise ValueError("Nonfinite weighted composition")
        self.weighted_sum += contribution
        self.total_positive_weight += float(weights[positive].sum())
        self.captured_rows += len(depth)
        self.positive_RNA_rows += int(positive.sum())
        self.zero_RNA_rows += int((~positive).sum())

    def result(self):
        support = {"captured_rows": self.captured_rows, "positive_RNA_rows": self.positive_RNA_rows,
                   "zero_RNA_rows": self.zero_RNA_rows,
                   "positive_sampling_weight": self.total_positive_weight,
                   "readout": "weighted_mean_positive_cell_RNA_composition"}
        if self.total_positive_weight <= 0:
            return {"composition": None, "available": False, "reason": "no_positive_weight_RNA_support", **support}
        q = self.weighted_sum / self.total_positive_weight
        if not np.isfinite(q).all() or not np.isclose(q.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("Observed full-axis composition did not normalize")
        return {"composition": q.copy(), "available": True, "reason": None, **support}


def _composition(value, *, ndim, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != ndim
            or min(value.shape) < 1 or not bool(torch.isfinite(value).all()) or bool((value < 0).any())):
        raise ValueError(f"Finite nonnegative full-axis {name} composition required")
    tolerance = 2e-6 if value.dtype == torch.float32 else 1e-10
    if not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), atol=tolerance, rtol=0):
        raise ValueError(f"Unnormalized {name} composition; no silent gene-wise renormalization")


def transform_composition(value):
    """Natural log(1+10000q), without a second log or an undeclared floor."""
    _composition(value, ndim=value.ndim, name="RNA")
    return torch.log1p(10000 * value)


def pool_compositions(compositions, fixed_weights):
    """When a declared mixture is used, pool BEFORE the nonlinear transform."""
    _composition(compositions, ndim=2, name="component")
    if (fixed_weights.shape != (len(compositions),) or fixed_weights.device != compositions.device
            or not bool(torch.isfinite(fixed_weights).all()) or bool((fixed_weights < 0).any())
            or not torch.allclose(fixed_weights.sum(), torch.ones_like(fixed_weights.sum()), atol=1e-8, rtol=0)):
        raise ValueError("Original fixed nonnegative mixture weights must sum to one")
    return torch.sum(compositions * fixed_weights[:, None], 0)


def endpoint_effect_terms(predicted_targets, observed_targets, predicted_control, observed_control,
                          *, target_weights=None):
    """Endpoint/contrast decomposition with one pooled control, not N controls.

    Return all-gene means. Target weights are the caller's fixed law reduction;
    the control endpoint is evaluated once. Gradients remain available for the
    prospective molecular-aware training recipes.
    """
    for value, ndim, name in ((predicted_targets, 2, "predicted targets"),
                              (observed_targets, 2, "observed targets"),
                              (predicted_control, 1, "predicted control"),
                              (observed_control, 1, "observed control")):
        _composition(value, ndim=ndim, name=name)
    if (predicted_targets.shape != observed_targets.shape
            or predicted_control.shape != observed_control.shape
            or predicted_targets.shape[1:] != predicted_control.shape
            or len({value.device for value in (predicted_targets, observed_targets, predicted_control, observed_control)}) != 1):
        raise ValueError("Target/control full-gene axes and devices must match")
    n = len(predicted_targets)
    if target_weights is None:
        target_weights = predicted_targets.new_full((n,), 1 / n)
    if (target_weights.shape != (n,) or target_weights.device != predicted_targets.device
            or not bool(torch.isfinite(target_weights).all()) or bool((target_weights < 0).any())
            or not torch.allclose(target_weights.sum(), torch.ones_like(target_weights.sum()), atol=2e-6, rtol=0)):
        raise ValueError("One fixed normalized target-law reduction required")
    error_t = transform_composition(predicted_targets) - transform_composition(observed_targets)
    error_c = transform_composition(predicted_control) - transform_composition(observed_control)
    target = error_t.square().mean(-1)
    control = error_c.square().mean()
    interaction = -2 * (error_t * error_c[None, :]).mean(-1)
    effect = (error_t - error_c[None, :]).square().mean(-1)
    zero = (transform_composition(observed_targets) - transform_composition(observed_control)).square().mean(-1)
    return {"target_mse_per_law": target, "control_mse": control,
            "effect_mse_per_law": effect, "interaction_per_law": interaction,
            "zero_effect_mse_per_law": zero,
            "target_mse": (target_weights * target).sum(),
            "effect_mse": (target_weights * effect).sum(),
            "zero_effect_mse": (target_weights * zero).sum(),
            "endpoint_objective": .5 * ((target_weights * target).sum() + control),
            "control_endpoint_observations": 1}
