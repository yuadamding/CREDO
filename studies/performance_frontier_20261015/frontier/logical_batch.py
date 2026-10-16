"""Exact first-order logical-batch accumulation with a two-pass count gradient.

This adapter performs backward calls, never optimizer calls or cache writes.
All sources/noise and logical denominators are caller-bound. Pass one evaluates
current active masses without graphs; the complete catalog supplies dL/d(log A).
Pass two repeats each identical microbatch and backpropagates its local losses
plus that detached mass derivative. Only one microbatch's graph is retained.

This is the gradient of the declared cached estimator, NOT the full current-
parameter catalog gradient. Local loss numerators must be additive over chunks;
the adapter cannot make a cross-guide nonlinear objective additive. Regularizers
on global parameters are applied once. No gradient clearing, clipping, optimizer
step, cache refresh, scheduling or automatic retry belongs to this primitive.
"""

from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .count_objective import ActiveFitnessChunk, CatalogBlock, FitnessCache, cached_count_objective


@dataclass(frozen=True)
class LogicalMicrobatch:
    identity: str
    cache_indices: Tensor
    payload: Mapping[str, Tensor]


@dataclass
class LocalForward:
    log_mass: Tensor
    loss_sums: Mapping[str, Tensor]
    metadata: Mapping[str, str | int | float | bool | None]


@dataclass(frozen=True)
class LossReduction:
    coefficient: float
    logical_denominator: float


@dataclass
class LogicalBatchResult:
    displayed_objective: Tensor
    local_loss_sums: dict[str, Tensor]
    local_weighted_terms: dict[str, Tensor]
    count_ce: Tensor | None
    donor_count_ce: Tensor | None
    count_gradient_coefficients: Tensor | None
    active_indices: Tensor
    pre_update_log_mass: Tensor
    global_regularizer: Tensor
    metadata: tuple[dict, ...]
    first_pass_calls: int
    replay_calls: int
    backward_calls: int
    maximum_replay_mass_absolute_difference: float
    replay_atol: float
    replay_rtol: float
    optimizer_calls: int = 0
    cache_writes: int = 0


def _number(value, *, name, positive=False):
    if (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"Finite {'positive' if positive else 'nonnegative'} {name} required")


def _metadata(values):
    if not isinstance(values, Mapping) or any(not isinstance(key, str) or not key for key in values):
        raise ValueError("Callback metadata must have nonempty string keys")
    for value in values.values():
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Callback metadata must be scalar-only, without tensors or retained graphs")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Callback metadata must be finite")
    return dict(values)


def _tensor_identity(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype,
            value.device, value.requires_grad)


def _module_identity(module):
    return (module.training, getattr(module, "phase", None),
            tuple((name, _tensor_identity(value)) for name, value in module.named_parameters()),
            tuple((name, _tensor_identity(value)) for name, value in module.named_buffers()))


def _input_identity(microbatches):
    return tuple((batch.identity, _tensor_identity(batch.cache_indices),
                  tuple((key, _tensor_identity(value)) for key, value in sorted(batch.payload.items())))
                 for batch in microbatches)


def _cache_identity(cache):
    if cache is None:
        return None
    return (_tensor_identity(cache.values), _tensor_identity(cache.seen), _tensor_identity(cache.age),
            tuple(sorted(cache.binding.items())), cache.last_full_refresh_update, cache.refresh_particles)


def _rng_state(devices):
    return (torch.get_rng_state().clone(), random.getstate(), np.random.get_state(),
            tuple((device, torch.cuda.get_rng_state(device).clone()) for device in devices))


def _same_rng(left, right):
    return (torch.equal(left[0], right[0]) and left[1] == right[1]
            and left[2][0] == right[2][0] and np.array_equal(left[2][1], right[2][1])
            and left[2][2:] == right[2][2:] and len(left[3]) == len(right[3])
            and all(a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(left[3], right[3])))


def _validate_forward(result, batch, reductions, *, include_local_losses, device, dtype):
    if not isinstance(result, LocalForward):
        raise ValueError("Callback must return LocalForward")
    mass = result.log_mass
    if (not isinstance(mass, Tensor) or mass.shape != batch.cache_indices.shape or mass.dtype != dtype
            or mass.device != device or not bool(torch.isfinite(mass).all())):
        raise ValueError("Callback log masses must match the active cache dtype/device/shape and be finite")
    expected = set(reductions) if include_local_losses else set()
    if not isinstance(result.loss_sums, Mapping) or set(result.loss_sums) != expected:
        raise ValueError("Callback must return exact named local numerators, or none for the count-only first pass")
    for name, value in result.loss_sums.items():
        if (not isinstance(value, Tensor) or value.ndim != 0 or value.dtype != dtype
                or value.device != device or not bool(torch.isfinite(value))):
            raise ValueError(f"Finite scalar local numerator on the mass device/dtype required: {name}")
    return _metadata(result.metadata)


def accumulate_logical_batch(
    module: nn.Module,
    microbatches: Sequence[LogicalMicrobatch],
    forward: Callable,
    *,
    phase: str,
    local_reductions: Mapping[str, LossReduction],
    count_coefficient: float,
    sampling_scale: float,
    blocks: Sequence[CatalogBlock] | None,
    cache: FitnessCache | None,
    global_regularizer: Callable[[], Tensor] | None,
    global_regularizer_coefficient: float,
    replay_atol: float,
    replay_rtol: float,
) -> LogicalBatchResult:
    """Accumulate into existing parameter gradients; caller owns their lifecycle.

    Callback signature is forward(batch, *, phase, include_local_losses). It
    receives explicit caller-owned payload tensors and must not sample RNG,
    mutate inputs/parameters/cache, normalize by microbatch size, or retain a graph
    elsewhere. loss_sums are weighted SUMS over the microbatch; fixed logical
    denominators and coefficients are applied here. Returned metadata must be the
    same across the two passes (no outcome or inclusion changes between them).

    A failure can leave partial accumulated gradients. The caller must treat it
    as a failed logical update and MUST NOT apply an optimizer step. The adapter
    does not hide this by clearing gradients or retrying. Returned pre-update
    masses are for a caller-owned cache write only after its accepted optimizer
    step; returning them is not evidence that such a step occurred.
    """
    if not isinstance(module, nn.Module) or phase not in ("state", "joint"):
        raise ValueError("Explicit module and state/joint phase required")
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("Logical gradient accumulation requires ordinary enabled autograd")
    if hasattr(module, "phase") and module.phase != phase:
        raise ValueError("Caller model phase differs from the logical-batch phase")
    for value, name in ((count_coefficient, "count coefficient"),
                        (global_regularizer_coefficient, "global regularizer coefficient"),
                        (replay_atol, "replay absolute tolerance"), (replay_rtol, "replay relative tolerance")):
        _number(value, name=name)
    _number(sampling_scale, name="sampling scale", positive=True)
    if phase == "state" and count_coefficient != 0:
        raise ValueError("State stabilization cannot optimize abundance")
    if global_regularizer is None and global_regularizer_coefficient != 0:
        raise ValueError("Nonzero global regularization needs an explicit callback")
    if not isinstance(local_reductions, Mapping):
        raise ValueError("Explicit named logical loss reductions required")
    for name, reduction in local_reductions.items():
        if not isinstance(name, str) or not name or not isinstance(reduction, LossReduction):
            raise ValueError("Named LossReduction records required")
        _number(reduction.coefficient, name=f"{name} coefficient")
        _number(reduction.logical_denominator, name=f"{name} logical denominator", positive=True)
    if not microbatches or any(not isinstance(batch, LogicalMicrobatch) for batch in microbatches):
        raise ValueError("A nonempty ordered logical microbatch inventory is required")
    identities = [batch.identity for batch in microbatches]
    if any(not isinstance(value, str) or not value for value in identities) or len(set(identities)) != len(identities):
        raise ValueError("Unique explicit microbatch identities required")
    parameters = tuple(module.parameters())
    if not parameters or not any(parameter.requires_grad for parameter in parameters):
        raise ValueError("The module has no trainable parameters")
    device, dtype = parameters[0].device, parameters[0].dtype
    if dtype not in (torch.float32, torch.float64) or any(p.device != device or p.dtype != dtype for p in parameters):
        raise ValueError("One explicit FP32/FP64 parameter device/dtype required")
    for batch in microbatches:
        index = batch.cache_indices
        if (not isinstance(index, Tensor) or index.dtype != torch.int64 or index.ndim != 1
                or not index.numel() or index.device != device or bool((index < 0).any())):
            raise ValueError("Each microbatch requires nonnegative int64 active indices on the model device")
        if (not isinstance(batch.payload, Mapping) or any(not isinstance(key, str) or not key
                or not isinstance(value, Tensor) for key, value in batch.payload.items())):
            raise ValueError("Payload must be explicit named prebound tensors")
    indices = torch.cat([batch.cache_indices for batch in microbatches])
    if indices.unique().numel() != indices.numel():
        raise ValueError("A modeled law cannot occur twice in the logical batch")
    count_enabled = count_coefficient > 0
    if count_enabled and (not isinstance(cache, FitnessCache) or not blocks
                          or cache.values.device != device or cache.values.dtype != dtype):
        raise ValueError("Enabled count objective requires matching complete catalogs and cache")
    module_identity, input_identity, cache_identity = (_module_identity(module), _input_identity(microbatches), _cache_identity(cache))
    cuda_devices = (device.index if device.index is not None else torch.cuda.current_device(),) if device.type == "cuda" else ()
    rng_identity = _rng_state(cuda_devices)

    def unchanged():
        if _module_identity(module) != module_identity or _input_identity(microbatches) != input_identity or _cache_identity(cache) != cache_identity:
            raise RuntimeError("Logical update changed model/input/cache identity; discard partial gradients and do not step")
        if not _same_rng(rng_identity, _rng_state(cuda_devices)):
            raise RuntimeError("Hidden RNG sampling during logical update; discard partial gradients and do not step")

    masses, first_metadata = [], []
    first_calls = replay_calls = backward_calls = 0
    count = None
    if count_enabled:
        with torch.no_grad():
            for batch in microbatches:
                unchanged()
                output = forward(batch, phase=phase, include_local_losses=False)
                metadata = _validate_forward(output, batch, local_reductions, include_local_losses=False, device=device, dtype=dtype)
                unchanged()
                masses.append(output.log_mass.detach().clone())
                first_metadata.append(metadata)
                first_calls += 1
                del output
            count = cached_count_objective(blocks, cache,
                [ActiveFitnessChunk(batch.cache_indices, mass) for batch, mass in zip(microbatches, masses)],
                sampling_scale=sampling_scale)
        count_derivative = count.active_derivative
    else:
        count_derivative = None
    local_sums = {name: parameters[0].new_zeros(()) for name in local_reductions}
    metadata_rows, offset, maximum_difference = [], 0, 0.0
    for position, batch in enumerate(microbatches):
        unchanged()
        output = forward(batch, phase=phase, include_local_losses=True)
        metadata = _validate_forward(output, batch, local_reductions, include_local_losses=True, device=device, dtype=dtype)
        unchanged()
        replay_calls += 1
        if count_enabled:
            if not output.log_mass.requires_grad:
                raise ValueError("Count-gradient replay returned detached masses; do not step")
            difference = float((output.log_mass.detach() - masses[position]).abs().max())
            maximum_difference = max(maximum_difference, difference)
            if not torch.allclose(output.log_mass.detach(), masses[position], atol=replay_atol, rtol=replay_rtol):
                raise RuntimeError("Replay mass differs from the first-pass current catalog; discard partial gradients and do not step")
            if metadata != first_metadata[position]:
                raise RuntimeError("Callback metadata changed between count and gradient passes; do not step")
        else:
            masses.append(output.log_mass.detach().clone())
        proxy = output.log_mass.new_zeros(())
        for name, reduction in local_reductions.items():
            value = output.loss_sums[name]
            local_sums[name] = local_sums[name] + value.detach()
            proxy = proxy + (reduction.coefficient / reduction.logical_denominator) * value
        if count_enabled:
            derivative = count_derivative[offset:offset + len(batch.cache_indices)]
            # Zero scalar value, exact required first derivative. The displayed
            # objective below uses the actual full-catalog CE, not this proxy.
            proxy = proxy + count_coefficient * torch.sum(derivative * (output.log_mass - output.log_mass.detach()))
        if proxy.requires_grad:
            proxy.backward()
            backward_calls += 1
        metadata_rows.append(metadata)
        offset += len(batch.cache_indices)
        del proxy, output
    unchanged()
    global_value = parameters[0].new_zeros(())
    if global_regularizer is not None:
        value = global_regularizer()
        if (not isinstance(value, Tensor) or value.ndim != 0 or value.device != device
                or value.dtype != dtype or not bool(torch.isfinite(value))):
            raise ValueError("Global regularizer must return one finite scalar on the parameter device/dtype")
        unchanged()
        global_value = value.detach()
        if value.requires_grad and global_regularizer_coefficient > 0:
            (global_regularizer_coefficient * value).backward()
            backward_calls += 1
        del value
    unchanged()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
        raise RuntimeError("Nonfinite accumulated gradient; do not apply an optimizer step")
    terms = {name: value * (local_reductions[name].coefficient / local_reductions[name].logical_denominator)
             for name, value in local_sums.items()}
    displayed = sum(terms.values(), parameters[0].new_zeros(())) + global_regularizer_coefficient * global_value
    if count_enabled:
        displayed = displayed + count_coefficient * count.ce.detach()
    return LogicalBatchResult(
        displayed_objective=displayed, local_loss_sums=local_sums, local_weighted_terms=terms,
        count_ce=None if count is None else count.ce.detach(),
        donor_count_ce=None if count is None else count.donor_ce.detach(),
        count_gradient_coefficients=None if count_derivative is None else count_derivative.detach(),
        active_indices=indices.detach().clone(), pre_update_log_mass=torch.cat(masses).detach(),
        global_regularizer=global_value, metadata=tuple(metadata_rows), first_pass_calls=first_calls,
        replay_calls=replay_calls, backward_calls=backward_calls,
        maximum_replay_mass_absolute_difference=maximum_difference,
        replay_atol=float(replay_atol), replay_rtol=float(replay_rtol))
