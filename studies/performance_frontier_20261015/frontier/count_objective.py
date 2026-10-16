"""One logical-batch cached categorical objective; no scheduling or data access.

Source authorities (the completed study, not this new implementation):
* fitting_gpu_native_v6/bundle/architecture_fitting.py:32-54: value-preserving
  population/batch gradient scale and equal-donor complete-catalog loss.
* fitting_gpu_native_v6/bundle/gpu_native.py:285-295: one log_softmax per full
  donor catalog, including fixed complement fitness.
* comparator_implementation_v1/comparator_training.py:114-170: detached inactive
  cache, current active replacement, momentum zero, and pre-update-value writes.
* comparator_qualification_launch_v2/package_v1/payload/comparator_production_v2/
  production_training.py:300-337,508-552: objective and update/write ordering.

For each donor, q=softmax(log_source + modeled_log_mass), with zero complement
increments. Let L be the equal-donor mean CE. The surrogate L+(c-1)*(L-stopgrad L)
has displayed scalar L and active gradient c*dL. Caller supplies c, ordinarily
N_laws/B_logical. All microbatch fitness tensors are assembled BEFORE any catalog
normalization. Independently normalized microbatch CE losses are not equivalent.

The historical training path used FP32. This primitive preserves an explicit
caller FP32/FP64 dtype; it does not silently cast measure_ops' FP64 source logs
to historical FP32. Catalog order is supplied unchanged. Mathematical agreement
with measure_ops.catalog_log_frequencies is required, not a claim that a new
order/dtype is bitwise historical replay. Cache refresh timing, particle counts,
sampling policy and loss coefficients belong to the calling execution contract.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor


CACHE_SCHEMA = "credo.frontier.endpoint_fitness_cache.v1"
CACHE_POLICY = "detached_inactive_current_active_complement_zero_momentum_zero"
_BINDING_KEYS = {"model_recipe", "training_subset", "catalog_layout", "source_sampler"}


def _binding(value):
    if (not isinstance(value, Mapping) or set(value) != _BINDING_KEYS
            or any(not isinstance(x, str) or not x for x in value.values())):
        raise ValueError("Exact nonempty model, training-subset, catalog-layout and sampler bindings required")
    return dict(value)


def _floating(values, name):
    if (not isinstance(values, Tensor) or values.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(values).all())):
        raise ValueError(f"Finite FP32/FP64 tensor required: {name}")


def _indices(values, *, limit, device, name):
    if (not isinstance(values, Tensor) or values.dtype != torch.int64 or values.ndim != 1
            or values.device != device or values.unique().numel() != values.numel()
            or bool(((values < 0) | (values >= limit)).any())):
        raise ValueError(f"Distinct in-range int64 indices on the bound device required: {name}")


def sampling_scale_from_sizes(*, population: int, logical_batch: int) -> float:
    if (type(population) is not int or type(logical_batch) is not int
            or not 1 <= logical_batch <= population):
        raise ValueError("Require 1 <= logical batch <= modeled-law population")
    return population / logical_batch


def value_preserving_surrogate(value: Tensor, *, sampling_scale: float) -> Tensor:
    _floating(value, "categorical loss")
    if value.ndim != 0 or isinstance(sampling_scale, bool) or not math.isfinite(sampling_scale) or sampling_scale <= 0:
        raise ValueError("A scalar loss and explicit finite positive gradient scale are required")
    return value + (sampling_scale - 1.0) * (value - value.detach())


@dataclass(frozen=True)
class CatalogBlock:
    """A complete catalog in caller-bound order; counts are unsmoothed truth."""
    donor_id: str
    catalog_id: str
    log_source: Tensor
    endpoint_counts: Tensor
    modeled_indices: Tensor
    cache_indices: Tensor

    def validate(self, cache):
        if not isinstance(self.donor_id, str) or not self.donor_id or not isinstance(self.catalog_id, str) or not self.catalog_id:
            raise ValueError("Explicit donor and complete catalog identity required")
        _floating(self.log_source, "source log frequencies")
        if (self.log_source.ndim != 1 or not self.log_source.numel()
                or self.log_source.requires_grad or self.log_source.device != cache.values.device
                or self.log_source.dtype != cache.values.dtype):
            raise ValueError("Fixed full source log frequencies must match cache dtype/device")
        tolerance = 1e-10 if self.log_source.dtype == torch.float64 else 2e-6
        if abs(torch.logsumexp(self.log_source, 0).item()) > tolerance:
            raise ValueError("Source log frequencies must be a normalized positive complete catalog")
        counts = self.endpoint_counts
        if (not isinstance(counts, Tensor) or counts.dtype not in (torch.int32, torch.int64)
                or counts.shape != self.log_source.shape or counts.device != self.log_source.device
                or bool((counts < 0).any()) or counts.sum().item() <= 0):
            raise ValueError("Complete nonnegative integer endpoint counts with positive total required")
        _indices(self.modeled_indices, limit=self.log_source.numel(), device=self.log_source.device, name="modeled catalog")
        _indices(self.cache_indices, limit=len(cache.values), device=self.log_source.device, name="cache routing")
        if self.modeled_indices.shape != self.cache_indices.shape:
            raise ValueError("Every modeled catalog entry needs exactly one cached law")


@dataclass(frozen=True)
class ActiveFitnessChunk:
    """Current log mass increments for one microbatch, before an optimizer step."""
    cache_indices: Tensor
    log_mass: Tensor


class FitnessCache:
    """Endpoint-only sufficient cache state, detached and never independently fit.

    Historical code stored source and endpoint rows. Source log mass increments
    are fixed zero here; only the endpoint vector enters categorical CE. No
    unneeded source row, ecological summaries or optimizer state is serialized.
    The caller explicitly chooses when to tick, refresh, or write PRE-update
    active values. These methods do not schedule or perform optimizer updates.
    """

    def __init__(self, values: Tensor, *, binding: Mapping[str, str], seen: Tensor,
                 age: Tensor, last_full_refresh_update: int = -1, refresh_particles: int = 0):
        _floating(values, "cached log masses")
        if values.ndim != 1 or not values.numel():
            raise ValueError("A nonempty modeled-law cache vector is required")
        if (not isinstance(seen, Tensor) or seen.dtype != torch.bool or seen.shape != values.shape
                or seen.device != values.device or not isinstance(age, Tensor) or age.dtype != torch.int64
                or age.shape != values.shape or age.device != values.device or bool((age < 0).any())
                or bool((age[~seen] != 0).any())):
            raise ValueError("Aligned boolean coverage and nonnegative integer ages required")
        if (type(last_full_refresh_update) is not int or last_full_refresh_update < -1
                or type(refresh_particles) is not int or refresh_particles < 0
                or (last_full_refresh_update == -1) != (refresh_particles == 0)):
            raise ValueError("Refresh metadata must consistently describe no refresh or an explicit completed refresh")
        self.values = values.detach().clone()
        self.seen, self.age = seen.detach().clone(), age.detach().clone()
        self.binding = _binding(binding)
        self.last_full_refresh_update = last_full_refresh_update
        self.refresh_particles = refresh_particles

    def assert_complete(self):
        if not bool(self.seen.all()):
            raise ValueError("Every modeled law needs cached coverage before count calculation")

    def mixed_fitness(self, indices, values):
        self.assert_complete()
        _indices(indices, limit=len(self.values), device=self.values.device, name="logical active laws")
        _floating(values, "active log masses")
        if values.shape != indices.shape or values.device != self.values.device or values.dtype != self.values.dtype:
            raise ValueError("Active values must match cache dtype/device and routing shape")
        return self.values.detach().clone().index_copy(0, indices, values)

    @torch.no_grad()
    def tick(self):
        self.age[self.seen] += 1

    @torch.no_grad()
    def update_active(self, indices, pre_update_values):
        """Caller supplies rollouts computed before its applied optimizer update."""
        _indices(indices, limit=len(self.values), device=self.values.device, name="active cache update")
        _floating(pre_update_values, "pre-update fitness")
        if (pre_update_values.shape != indices.shape or pre_update_values.device != self.values.device
                or pre_update_values.dtype != self.values.dtype):
            raise ValueError("Pre-update values must match routing and cache dtype/device")
        self.values.index_copy_(0, indices, pre_update_values.detach())
        self.seen.index_fill_(0, indices, True)
        self.age.index_fill_(0, indices, 0)

    @torch.no_grad()
    def apply_full_refresh(self, values, *, update_index: int, particles: int):
        _floating(values, "full refresh fitness")
        if (values.shape != self.values.shape or values.device != self.values.device or values.dtype != self.values.dtype
                or type(update_index) is not int or update_index < 0 or type(particles) is not int or particles < 1):
            raise ValueError("Full aligned refresh and explicit nonnegative update/positive particle count required")
        self.values.copy_(values.detach())
        self.seen.fill_(True)
        self.age.zero_()
        self.last_full_refresh_update, self.refresh_particles = update_index, particles

    def state_dict(self):
        return dict(schema=CACHE_SCHEMA, policy=CACHE_POLICY, binding=dict(self.binding),
                    dtype=str(self.values.dtype), values=self.values.detach().cpu().clone(),
                    seen=self.seen.detach().cpu().clone(), age=self.age.detach().cpu().clone(),
                    last_full_refresh_update=self.last_full_refresh_update,
                    refresh_particles=self.refresh_particles)

    @classmethod
    def from_state_dict(cls, state, *, expected_binding, expected_size: int, expected_dtype, device="cpu"):
        expected_keys = {"schema", "policy", "binding", "dtype", "values", "seen", "age",
                         "last_full_refresh_update", "refresh_particles"}
        if (not isinstance(state, dict) or set(state) != expected_keys or state["schema"] != CACHE_SCHEMA
                or state["policy"] != CACHE_POLICY or state["binding"] != _binding(expected_binding)
                or state["dtype"] != str(expected_dtype) or type(expected_size) is not int or expected_size < 1):
            raise ValueError("Cache schema, policy, binding or expected shape/dtype differs")
        if (not isinstance(state["values"], Tensor) or state["values"].shape != (expected_size,)
                or state["values"].dtype != expected_dtype):
            raise ValueError("Saved cache values differ from the expected dtype or size")
        if any(not isinstance(state[key], Tensor) or state[key].device.type != "cpu" for key in ("values", "seen", "age")):
            raise ValueError("Restart payload must contain explicit CPU cache tensors")
        return cls(state["values"].to(device), binding=state["binding"], seen=state["seen"].to(device),
                   age=state["age"].to(device), last_full_refresh_update=state["last_full_refresh_update"],
                   refresh_particles=state["refresh_particles"])


@dataclass
class CountObjective:
    ce: Tensor
    surrogate: Tensor
    donor_ce: Tensor
    donor_ids: tuple[str, ...]
    catalog_log_probabilities: tuple[Tensor, ...]
    active_indices: Tensor
    active_log_mass: Tensor
    active_derivative: Tensor
    sampling_scale: float
    modeled_population: int


def cached_count_objective(blocks: Sequence[CatalogBlock], cache: FitnessCache,
                           chunks: Sequence[ActiveFitnessChunk], *, sampling_scale: float) -> CountObjective:
    """Assemble all current chunks, then normalize once per complete donor catalog.

    A logical batch must contain distinct modeled laws. Repeated guides in
    different chunks are rejected rather than silently changing their sampling
    weights. Donors without an active guide still contribute their cached full CE.
    active_derivative is a detached exact derivative of the surrogate with respect
    to concatenated active log masses, useful for a separately qualified two-pass
    gradient implementation; using it does not itself establish such an adapter.
    """
    if not isinstance(cache, FitnessCache) or not blocks or not chunks:
        raise ValueError("A cache, complete donor blocks and one nonempty logical active batch are required")
    if any(not isinstance(block, CatalogBlock) for block in blocks):
        raise ValueError("Every donor requires an explicit CatalogBlock")
    for block in blocks:
        block.validate(cache)
    if (len({block.donor_id for block in blocks}) != len(blocks)
            or len({block.catalog_id for block in blocks}) != 1
            or len({block.log_source.numel() for block in blocks}) != 1):
        raise ValueError("Distinct donors must share one complete catalog identity and dimension")
    routing = torch.cat([block.cache_indices for block in blocks])
    if (routing.numel() != len(cache.values) or routing.unique().numel() != len(cache.values)
            or not torch.equal(routing.sort().values, torch.arange(len(cache.values), device=cache.values.device))):
        raise ValueError("Donor catalog routing must cover every modeled law exactly once")
    for chunk in chunks:
        if not isinstance(chunk, ActiveFitnessChunk):
            raise ValueError("Explicit current-fitness chunks required")
        _indices(chunk.cache_indices, limit=len(cache.values), device=cache.values.device, name="microbatch active laws")
        _floating(chunk.log_mass, "microbatch active log masses")
        if (not chunk.cache_indices.numel() or chunk.log_mass.shape != chunk.cache_indices.shape
                or chunk.log_mass.dtype != cache.values.dtype or chunk.log_mass.device != cache.values.device):
            raise ValueError("Every nonempty active chunk must match the cache dtype/device and routing")
    active_indices = torch.cat([chunk.cache_indices for chunk in chunks])
    active_values = torch.cat([chunk.log_mass for chunk in chunks])
    full = cache.mixed_fitness(active_indices, active_values)  # rejects cross-chunk duplicates
    losses, log_probabilities = [], []
    derivative = torch.zeros_like(full)
    for block in blocks:
        increments = torch.zeros_like(block.log_source).index_copy(0, block.modeled_indices, full[block.cache_indices])
        log_q = torch.log_softmax(block.log_source + increments, dim=0)
        truth = block.endpoint_counts.to(dtype=log_q.dtype)
        truth = truth / truth.sum()
        losses.append(-(truth * log_q).sum())
        log_probabilities.append(log_q)
        derivative = derivative.index_copy(0, block.cache_indices,
                                           (log_q.exp() - truth)[block.modeled_indices] / len(blocks))
    donor_ce = torch.stack(losses)
    ce = donor_ce.mean()
    surrogate = value_preserving_surrogate(ce, sampling_scale=sampling_scale)
    return CountObjective(ce=ce, surrogate=surrogate, donor_ce=donor_ce,
                          donor_ids=tuple(block.donor_id for block in blocks),
                          catalog_log_probabilities=tuple(log_probabilities),
                          active_indices=active_indices, active_log_mass=active_values,
                          active_derivative=(derivative[active_indices] * sampling_scale).detach(),
                          sampling_scale=float(sampling_scale), modeled_population=len(cache.values))
