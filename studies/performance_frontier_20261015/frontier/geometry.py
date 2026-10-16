"""Retained full-support, debiased Sinkhorn geometry on ragged tensor laws.

Source: credo_full_ddr_gse314342_20261006_v1/fitting_gpu_native_v6/bundle/
gpu_native.py, SHA256 9aa1229b8bba69c6e3e4d49cf3c5d6cbdc23fc1097708d1ef4c92851709d1b12.
The backend resolver, softmin, solver and geometry-width functions below retain
the source equations. The wrapper replaces historical whole-study objects with
explicit tensors. No data, model, sampler or optimizer is opened here.

This is NOT energy distance. It uses p=2, blur=sqrt(0.1), scaling=0.5, debias=True,
and EACH law's own annealing diameter/schedule. Observed weights retain the
historical training clamp_min(1e-30) before log; affected rows are reported. This
is not a catalog-probability floor or an exact-zero observed-weight convention.
Native positive uniform RNA-law weights above the clamp are unchanged. Padding
always has log weight -inf, and no observed row (even a zero-weight row) is dropped.

The explicit element cap bounds the four nominal cost matrices PER kernel call,
not the total CUDA allocator or autograd peak across retained outputs. Runtime
memory/throughput still require hardware qualification. Missing observed laws
are omitted with an explicit denominator, never assigned a scored zero loss.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.metadata import version
import inspect
import math
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


RETAINED_SOURCE_SHA256 = "9aa1229b8bba69c6e3e4d49cf3c5d6cbdc23fc1097708d1ef4c92851709d1b12"
OBSERVED_TRAINING_WEIGHT_CLAMP = 1e-30
SINKHORN_EPSILON = 0.1


@lru_cache(maxsize=1)
def geomloss_backend():
    """Bind internal helpers to the SAME backend as public scalar SamplesLoss.

    GeomLoss 0.3.1 moved the scalar implementation into ``_legacy``; 0.2.6
    exposes it directly. Resolve from the public class, never by catching an
    arbitrary import failure and silently trying a different estimator.
    This metadata-only check is safe before opening any cohort inputs.
    """
    from geomloss import SamplesLoss

    scalar_module = SamplesLoss.__module__
    if scalar_module not in {"geomloss.samples_loss", "geomloss._legacy.samples_loss"}:
        raise RuntimeError("Unqualified GeomLoss scalar backend: " + scalar_module)
    package = scalar_module.rsplit(".", 1)[0]
    scalar = import_module(scalar_module)
    divergence = import_module(package + ".sinkhorn_divergence")
    utilities = import_module(package + ".utils")
    samples = import_module(package + ".sinkhorn_samples")
    if (scalar.SamplesLoss is not SamplesLoss
            or scalar.routines["sinkhorn"]["tensorized"] is not samples.sinkhorn_tensorized):
        raise RuntimeError("GeomLoss public and tensorized backend identities differ")
    helpers = {}
    for name in ("log_weights", "sinkhorn_cost", "sinkhorn_loop", "epsilon_schedule"):
        helper = getattr(divergence, name)
        if not callable(helper) or helper.__module__ != divergence.__name__:
            raise RuntimeError("Unqualified GeomLoss helper: " + name)
        helpers[name] = helper
    for name in ("log_weights", "sinkhorn_cost", "sinkhorn_loop"):
        if samples.sinkhorn_tensorized.__globals__.get(name) is not helpers[name]:
            raise RuntimeError("GeomLoss batched/scalar helper identities differ: " + name)
    squared_distances = utilities.squared_distances
    if not callable(squared_distances) or squared_distances.__module__ != utilities.__name__:
        raise RuntimeError("Unqualified GeomLoss squared-distance helper")
    try:
        inspect.signature(helpers["sinkhorn_loop"]).bind(*(None,) * 9, debias=True)
        inspect.signature(helpers["sinkhorn_cost"]).bind(*(None,) * 8, batch=True, debias=True, potentials=False)
        inspect.signature(helpers["epsilon_schedule"]).bind(*(None,) * 4)
    except TypeError as error:
        raise RuntimeError("GeomLoss internal call signature changed; no fallback") from error
    return SimpleNamespace(**helpers, squared_distances=squared_distances,
        identity=dict(version=version("geomloss"), scalar_module=scalar_module,
                      divergence_module=divergence.__name__, utilities_module=utilities.__name__))


def softmin_batched(eps, cost, log_potential):
    """Per-guide temperatures, not a batch-global annealing temperature."""
    return -eps * (log_potential[:, None, :] - cost / eps[:, :, None]).logsumexp(2)


def sinkhorn_batched(x, log_a, y, log_b, schedules):
    """Upstream tensorized solver with independent scalar-guide schedules.

    Only groups with equal schedule LENGTH are combined. Values remain different
    for each guide. Right-side detach and envelope gradients match GeomLoss.
    The upstream loop changes global grad mode; restore it even on failure.
    """
    backend = geomloss_backend()
    log_weights, sinkhorn_cost, sinkhorn_loop = backend.log_weights, backend.sinkhorn_cost, backend.sinkhorn_loop
    squared_distances = backend.squared_distances

    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[-1] != y.shape[-1]:
        raise ValueError("Matching batched point clouds required")
    if log_a.shape != x.shape[:2] or log_b.shape != y.shape[:2]:
        raise ValueError("Weights and supports disagree")
    if len(schedules) != x.shape[0] or not schedules or len({len(s) for s in schedules}) != 1:
        raise ValueError("One equal-length annealing schedule per guide required")
    if any(any(not math.isfinite(eps) or eps <= 0 for eps in s) for s in schedules):
        raise ValueError("Positive finite temperatures required")
    if len({s[-1] for s in schedules}) != 1:
        raise ValueError("All guides must use the same terminal epsilon")
    a, b = log_a.softmax(-1), log_b.softmax(-1)
    c_xy = squared_distances(x, y.detach()) / 2
    c_yx = squared_distances(y, x.detach()) / 2
    c_xx = squared_distances(x, x.detach()) / 2
    c_yy = squared_distances(y, y.detach()) / 2
    schedule = torch.tensor(np.asarray(schedules).T.copy(), device=x.device, dtype=x.dtype)
    eps_list = [row[:, None] for row in schedule]
    grad_enabled = torch.is_grad_enabled()
    try:
        f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(softmin_batched, log_weights(a), log_weights(b),
            c_xx, c_yy, c_xy, c_yx, eps_list, None, debias=True)
        return sinkhorn_cost(schedules[0][-1], None, a, b, f_aa, g_bb, g_ab, f_ba,
                             batch=True, debias=True, potentials=False)
    finally:
        torch.set_grad_enabled(grad_enabled)


def geometry_width(length, particles, options):
    """Retain full support; padding is optional, the workspace bound is not.

    Most laws share power-of-two buckets. If padding alone would make even a
    singleton inadmissible, use its exact length with the SAME dense solver.
    A truly oversized law still fails closed, before scientific updates.
    """
    if type(length) is not int or length < 1 or type(particles) is not int or particles < 1:
        raise ValueError("Positive integer geometry support and particle counts required")
    padded = 1 << (length - 1).bit_length()
    width = padded if (particles + padded)**2 <= options.maximum_geometry_elements else length
    if (particles + width)**2 > options.maximum_geometry_elements:
        raise RuntimeError("Single full-support geometry exceeds workspace budget: "
            f"support={length}, particles={particles}, exact_elements={(particles + length)**2}, "
            f"maximum={options.maximum_geometry_elements}")
    return width


@dataclass(frozen=True)
class GeometryOptions:
    maximum_geometry_elements: int
    maximum_geometry_guides: int
    expected_geomloss_version: str

    def __post_init__(self):
        for name in ("maximum_geometry_elements", "maximum_geometry_guides"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Explicit positive integer {name} required")
        if self.expected_geomloss_version not in ("0.2.6", "0.3.1"):
            raise ValueError("Explicit retained-layout GeomLoss version required; no latest-version fallback")


@dataclass(frozen=True)
class ObservedLaw:
    law_id: str
    support: Tensor
    weights: Tensor


@dataclass
class GeometryResult:
    per_law_losses: Tensor
    available_indices: Tensor
    unavailable_indices: Tensor
    loss_sum: Tensor
    loss_mean: Tensor | None
    input_law_count: int
    eligible_count: int
    observation_support_counts: tuple[int, ...]
    observed_weight_clamp_counts: tuple[int, ...]
    diameters: tuple[float, ...]
    annealing_schedules: tuple[tuple[float, ...], ...]
    kernel_batches: tuple[dict, ...]
    backend_identity: dict
    dropped_observed_rows: int = 0

    def weighted_sum(self, full_law_weights: Tensor) -> Tensor:
        """Additive numerator; caller supplies the complete logical denominator."""
        if (not isinstance(full_law_weights, Tensor) or full_law_weights.shape != (self.input_law_count,)
                or full_law_weights.device != self.loss_sum.device or full_law_weights.dtype != self.loss_sum.dtype
                or full_law_weights.requires_grad or not bool(torch.isfinite(full_law_weights).all())
                or bool((full_law_weights < 0).any())):
            raise ValueError("Fixed nonnegative full-input law weights on the loss device/dtype required")
        return (self.per_law_losses * full_law_weights[self.available_indices]).sum() + self.loss_sum * 0


class RaggedSinkhornGeometry:
    """Full observed supports, zero-probability padding, independent schedules."""

    def __init__(self, *, options: GeometryOptions):
        if not isinstance(options, GeometryOptions):
            raise ValueError("Explicit geometry workspace and backend options required")
        self.options = options
        backend = geomloss_backend()
        if backend.identity["version"] != options.expected_geomloss_version:
            raise RuntimeError("Installed GeomLoss version differs from the caller-bound version")
        self.backend_identity = {**backend.identity, "torch_version": version("torch"),
            "retained_source_sha256": RETAINED_SOURCE_SHA256, "p": 2,
            "blur": math.sqrt(SINKHORN_EPSILON), "scaling": 0.5, "debias": True,
            "observed_training_weight_clamp": OBSERVED_TRAINING_WEIGHT_CLAMP,
            "workspace_bound": "nominal_four_cost_matrices_per_kernel_not_allocator_peak"}

    def __call__(self, predicted: Tensor, normalized_log_weights: Tensor, *,
                 law_ids: Sequence[str], observations: Sequence[ObservedLaw | None]) -> GeometryResult:
        if (not isinstance(predicted, Tensor) or predicted.ndim != 3 or predicted.shape[-1] != 48
                or min(predicted.shape[:2]) < 1 or predicted.dtype not in (torch.float32, torch.float64)
                or predicted.device.type not in ("cpu", "cuda") or not bool(torch.isfinite(predicted).all())):
            raise ValueError("Finite FP32/FP64 laws x particles x 48 predictions required")
        batch, particles, _ = predicted.shape
        logs = normalized_log_weights
        if (not isinstance(logs, Tensor) or logs.shape != predicted.shape[:2] or logs.device != predicted.device
                or logs.dtype != predicted.dtype or bool(torch.isnan(logs).any()) or bool(torch.isposinf(logs).any())):
            raise ValueError("Aligned normalized log weights (zero probability = -inf) required")
        tolerance = 2e-6 if logs.dtype == torch.float32 else 1e-10
        normalization = torch.logsumexp(logs.detach(), dim=1)
        if not bool(torch.isfinite(normalization).all()) or bool((normalization.abs() > tolerance).any()):
            raise ValueError("Prediction log weights must already be normalized per law")
        ids = tuple(law_ids)
        if len(ids) != batch or any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != batch or len(observations) != batch:
            raise ValueError("One distinct law identity and explicit observed-or-missing entry per prediction required")
        lengths, clamp_counts, widths, selected, missing = [], [], {}, [], []
        for row, law in enumerate(observations):
            if law is None:
                lengths.append(0)
                clamp_counts.append(0)
                missing.append(row)
                continue
            if not isinstance(law, ObservedLaw) or law.law_id != ids[row]:
                raise ValueError("Observed law identity does not match prediction routing")
            y, w = law.support, law.weights
            if (not isinstance(y, Tensor) or y.ndim != 2 or y.shape[1] != 48 or y.shape[0] < 1
                    or y.device != predicted.device or y.dtype != predicted.dtype or y.requires_grad
                    or not bool(torch.isfinite(y).all()) or not isinstance(w, Tensor) or w.shape != y.shape[:1]
                    or w.device != predicted.device or w.dtype != predicted.dtype or w.requires_grad
                    or not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or not bool(w.sum() > 0)):
                raise ValueError("Full finite fixed observed support and nonnegative positive-total weights required")
            lengths.append(len(y))
            clamp_counts.append(int((w < OBSERVED_TRAINING_WEIGHT_CLAMP).sum()))
            widths[row] = geometry_width(len(y), particles, self.options)
            selected.append(row)
        available = torch.tensor(selected, device=predicted.device, dtype=torch.int64)
        unavailable = torch.tensor(missing, device=predicted.device, dtype=torch.int64)
        if not selected:
            zero = predicted.sum() * 0
            return GeometryResult(predicted.new_empty(0), available, unavailable, zero, None,
                batch, 0, tuple(lengths), tuple(clamp_counts), (), (), (), dict(self.backend_identity))

        def padded(rows, width):
            y = predicted.new_zeros((len(rows), width, 48))
            log_b = predicted.new_full((len(rows), width), -torch.inf)
            valid = torch.zeros((len(rows), width), device=predicted.device, dtype=torch.bool)
            for index, row in enumerate(rows):
                law = observations[row]
                length = lengths[row]
                y[index, :length] = law.support
                log_b[index, :length] = law.weights.clamp_min(OBSERVED_TRAINING_WEIGHT_CLAMP).log()
                valid[index, :length] = True
            return y, log_b, valid

        def limit(width):
            return min(self.options.maximum_geometry_guides,
                       self.options.maximum_geometry_elements // (particles + width)**2)

        # First calculate independent diameters, transferring only one small
        # concatenated metadata vector. Padded zeros never enter these bounds.
        buckets = defaultdict(list)
        for row in selected:
            buckets[widths[row]].append(row)
        diameter_parts, diameter_rows = [], []
        for width, members in buckets.items():
            for start in range(0, len(members), limit(width)):
                rows = members[start:start + limit(width)]
                y, _, valid = padded(rows, width)
                x = predicted[rows].detach()
                lower = torch.minimum(x.amin(1), y.masked_fill(~valid[..., None], torch.inf).amin(1))
                upper = torch.maximum(x.amax(1), y.masked_fill(~valid[..., None], -torch.inf).amax(1))
                diameter_parts.append((upper - lower).norm(dim=-1))
                diameter_rows.extend(rows)
        diameter_values = torch.cat(diameter_parts).detach().cpu().tolist()
        if any(not math.isfinite(value) or value <= 0 for value in diameter_values):
            raise RuntimeError("Degenerate Sinkhorn annealing diameter; no silent fallback")
        diameters = dict(zip(diameter_rows, diameter_values))
        epsilon_schedule = geomloss_backend().epsilon_schedule
        schedules = {row: epsilon_schedule(2, diameters[row], math.sqrt(SINKHORN_EPSILON), .5) for row in selected}
        groups = defaultdict(list)
        for row in selected:
            groups[(widths[row], len(schedules[row]))].append(row)
        values, value_rows, batches = [], [], []
        for (width, _), members in groups.items():
            for start in range(0, len(members), limit(width)):
                rows = members[start:start + limit(width)]
                y, log_b, _ = padded(rows, width)
                values.append(sinkhorn_batched(predicted[rows], logs[rows], y, log_b, [schedules[row] for row in rows]))
                value_rows.extend(rows)
                batches.append(dict(law_indices=tuple(rows), padded_width=width,
                    nominal_cost_elements=len(rows) * (particles + width)**2,
                    nominal_cost_bytes=len(rows) * (particles + width)**2 * predicted.element_size()))
        combined = torch.cat(values)
        order = torch.tensor(np.argsort(value_rows).copy(), device=predicted.device)
        per_law = combined[order]
        if not bool(torch.isfinite(per_law.detach()).all()):
            raise RuntimeError("Nonfinite full-support Sinkhorn loss; no fallback")
        return GeometryResult(per_law, available, unavailable, per_law.sum(), per_law.mean(),
            batch, len(selected), tuple(lengths), tuple(clamp_counts),
            tuple(diameters[row] for row in selected), tuple(tuple(schedules[row]) for row in selected),
            tuple(batches), dict(self.backend_identity))
