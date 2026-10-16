"""Stateless, law-scoped CPU sampling with optional explicit GPU transfer.

Source states are supplied already encoded in the admitted fold representation.
This module reads no files and cannot verify the input owner's scientific
provenance. It does not select targets, substitute controls or infer missing laws.

Each source/noise stream uses SHA-256 -> NumPy PCG64DXSM, keyed by the declared
study/fold/seed/update/purpose/generation, complete source-law identity and an
explicit draw occurrence. Configuration stays in receipts but is deliberately
excluded from draw keys for common random numbers across model families.

Noise is drawn particle-major, then transposed to [steps,laws,particles,latent].
Prefix and law-microbatch changes do not change a particle's draws. Requesting a
particle offset regenerates the bounded prefix and slices it; it does not depend
on a previous call, cached mutable RNG state, or a microchunk's execution order.

The sampler is iid within every F2 strided component, not a sorted partition of
source cells. This establishes its sampling law, not guaranteed rare-state
coverage in a finite component. Source importance weights are used ONCE, in the
categorical draw; sampled particles subsequently have equal integration weight.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np
import torch

from .configuration import DONORS, SEEDS, get_configuration


SAMPLER_VERSION = "credo.frontier.particle_major_pcg64dxsm.v1"
PURPOSES = ("training", "cache", "forecast", "qualification", "scales")


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Explicit nonempty {label} required")


def _unsigned(value, label):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(f"Nonnegative 64-bit integer required: {label}")


def _sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SamplingContext:
    study_id: str
    training_donors: tuple
    configuration_id: str
    study_seed: int
    model_seed: int
    accepted_update: int
    purpose: str
    generation: int = 0

    def __post_init__(self):
        _text(self.study_id, "study identity")
        donors = tuple(self.training_donors)
        if len(donors) not in (2, 3) or donors != tuple(d for d in DONORS if d in donors):
            raise ValueError("Canonical two- or three-donor fitting subset required")
        get_configuration(self.configuration_id)
        _unsigned(self.study_seed, "study seed")
        _unsigned(self.accepted_update, "accepted update")
        _unsigned(self.generation, "generation")
        if type(self.model_seed) is not int or self.model_seed not in SEEDS:
            raise ValueError("Registered seed triplet is exactly 0, 1, 2")
        if self.purpose not in PURPOSES:
            raise ValueError("Explicit registered sampling purpose required")
        object.__setattr__(self, "training_donors", donors)

    def draw_binding(self):
        result = asdict(self)
        result.pop("configuration_id")  # Architecture-neutral common draws.
        return result


@dataclass(frozen=True)
class LawIdentity:
    donor: str
    condition: str
    actual_guide_id: str
    source_id: str
    replicate_id: str

    def __post_init__(self):
        if self.donor not in DONORS or self.condition != "Stim8hr":
            raise ValueError("This continuation sampler accepts declared Stim8hr source laws only")
        for field in ("actual_guide_id", "source_id", "replicate_id"):
            _text(getattr(self, field), field)


@dataclass(frozen=True)
class PooledSourceIdentity:
    """RNA-state pool stream identity, never an actual guide/catalog category.

    Field names deliberately differ from LawIdentity, so its existing draw keys
    remain unchanged and a pool cannot collide with a guide-labelled stream.
    The definition hash binds membership/weighting; it is not provenance proof.
    """
    donor: str
    condition: str
    source_id: str
    replicate_id: str
    pool_id: str
    ntc_catalog_sha256: str
    definition_sha256: str

    def __post_init__(self):
        if self.donor not in DONORS or self.condition != "Stim8hr":
            raise ValueError("Pooled source sampling requires declared donor/Stim8hr identity")
        for field in ("source_id", "replicate_id", "pool_id"):
            _text(getattr(self, field), field)
        for field in ("ntc_catalog_sha256", "definition_sha256"):
            value = getattr(self, field)
            if (type(value) is not str or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("Pooled source requires exact NTC catalog/definition identities")


@dataclass(frozen=True)
class SourceLaw:
    identity: LawIdentity | PooledSourceIdentity
    states: object
    weights: object

    def arrays(self, latent_dim):
        if not isinstance(self.identity, (LawIdentity, PooledSourceIdentity)):
            raise ValueError("Fully scoped donor/condition/guide/source/replicate identity required")
        states = self.states
        if isinstance(states, torch.Tensor):
            if states.device.type != "cpu" or states.dtype != torch.float32 or states.requires_grad:
                raise ValueError("Source cache must be constant CPU FP32; no implicit device move/detach")
            states = states.numpy()
        if (not isinstance(states, np.ndarray) or states.dtype != np.float32
                or states.ndim != 2 or states.shape[1] != latent_dim or len(states) == 0
                or not np.isfinite(states).all()):
            raise ValueError("Nonempty finite FP32 source state law on the bound latent axis required")
        weights = self.weights
        if isinstance(weights, torch.Tensor):
            if weights.device.type != "cpu" or not weights.is_floating_point() or weights.requires_grad:
                raise ValueError("Source weights must be constant CPU floating values")
            weights = weights.numpy()
        weights = np.asarray(weights)
        if (weights.dtype.kind != "f" or weights.shape != (len(states),)
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("Finite nonnegative source weights aligned with every source state required")
        weights = weights.astype(np.float64, copy=False)
        total = float(weights.sum(dtype=np.float64))
        if total <= 0 or not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Source integration weights must already sum to one")
        # Explicit floating-summation normalization, not a change in the source
        # weighting policy or a floor for zero-probability source cells.
        return states, weights / total


@dataclass(frozen=True)
class LawRequest:
    law: SourceLaw
    occurrence: int = 0

    def __post_init__(self):
        if not isinstance(self.law, SourceLaw):
            raise ValueError("An explicit source law is required")
        _unsigned(self.occurrence, "logical law occurrence")

    @property
    def identity(self):
        return {"law": asdict(self.law.identity), "occurrence": self.occurrence}


@dataclass(frozen=True)
class SamplerSpecification:
    maximum_particles: int = 8192
    maximum_laws: int = 1024
    latent_dim: int = 48
    steps: int = 8
    numpy_version: str = np.__version__

    def __post_init__(self):
        if (type(self.maximum_particles) is not int or not 1 <= self.maximum_particles <= 8192
                or type(self.maximum_laws) is not int or not 1 <= self.maximum_laws <= 1024
                or self.latent_dim != 48 or self.steps != 8):
            raise ValueError("Bound positive law/particle ceilings, 48 coordinates and eight steps required")
        if self.numpy_version != np.__version__:
            raise ValueError("NumPy runtime differs from the bound sampler transform")


@dataclass(frozen=True)
class ParticleBatch:
    states: torch.Tensor
    noise: torch.Tensor
    source_indices: torch.Tensor
    receipt: dict


@dataclass(frozen=True)
class ScaleSourceBatch:
    """Persistence scale draws: no unused SDE noise tensor is constructed."""
    states: torch.Tensor
    source_indices: torch.Tensor
    receipt: dict


def _generator(context, request, stream):
    if stream not in ("source_categorical", "standard_normal"):
        raise ValueError("Unregistered particle random stream")
    key = {"schema": SAMPLER_VERSION, "context": context.draw_binding(),
           "request": request.identity, "stream": stream}
    digest = _sha256(key)
    return np.random.Generator(np.random.PCG64DXSM(int(digest, 16))), digest


def sample_batch(requests: Sequence[LawRequest], context: SamplingContext, *, particles: int,
                 particle_offset=0, specification: SamplerSpecification | None = None,
                 device="cpu", pin_memory=False):
    """Draw supplied laws in caller order; never infer support or replace a law.

    The source/noise numerical streams do not use batch position, microbatch
    width, GPU worker identity or configuration. An explicit occurrence is
    necessary when the same scoped law appears twice in a logical batch; assign
    occurrences before partitioning, rather than resetting them per microbatch.

    Particle offsets support stream/replay checks. F2's rollout additionally
    needs its global component proposal convention: do not pass arbitrary
    particle chunks as independent locally reindexed mixture rollouts.

    CPU->GPU pinned transfer is an opt-in implementation path, not a claim of
    measured throughput or GPU numerical qualification. Source indices remain
    on CPU for provenance; state/noise tensors move to the explicit destination.
    """
    if not isinstance(context, SamplingContext):
        raise ValueError("A bound sampling context is required")
    spec = SamplerSpecification() if specification is None else specification
    if not isinstance(spec, SamplerSpecification) or spec.numpy_version != np.__version__:
        raise ValueError("A matching sampler specification is required")
    requests = tuple(requests)
    if not 0 < len(requests) <= spec.maximum_laws or not all(isinstance(r, LawRequest) for r in requests):
        raise ValueError("Nonempty bounded list of explicit law requests required")
    if (type(particles) is not int or particles < 1 or type(particle_offset) is not int
            or particle_offset < 0 or particle_offset + particles > spec.maximum_particles):
        raise ValueError("Particle interval exceeds its fixed positive prefix allowance")
    request_keys = [_sha256(request.identity) for request in requests]
    if len(set(request_keys)) != len(request_keys):
        raise ValueError("Duplicate scoped law and occurrence; assign explicit logical occurrences")
    if context.purpose in ("training", "cache", "scales") and any(
            request.law.identity.donor not in context.training_donors for request in requests):
        raise ValueError("Training/cache sampling cannot contain a held-out donor")
    destination = torch.device(device)
    if destination.type not in ("cpu", "cuda") or (destination.type == "cuda" and destination.index is None):
        raise ValueError("Destination must be CPU or an explicit indexed CUDA device")
    if type(pin_memory) is not bool or (pin_memory and destination.type != "cuda"):
        raise ValueError("Pinned staging is an explicit CUDA-transfer option only")
    stop = particle_offset + particles
    states = np.empty((len(requests), particles, spec.latent_dim), dtype=np.float32)
    noise = np.empty((spec.steps, len(requests), particles, spec.latent_dim), dtype=np.float32)
    source_indices = np.empty((len(requests), particles), dtype=np.int64)
    streams = []
    for law_index, request in enumerate(requests):  # No Python particle loop.
        values, weights = request.law.arrays(spec.latent_dim)
        source_generator, source_key = _generator(context, request, "source_categorical")
        noise_generator, noise_key = _generator(context, request, "standard_normal")
        cdf = np.cumsum(weights, dtype=np.float64)
        cdf[-1] = 1.0
        # side=right skips all zero-width CDF intervals, including leading zeros.
        indices = np.searchsorted(cdf, source_generator.random(stop, dtype=np.float64), side="right")[particle_offset:]
        source_indices[law_index] = indices
        states[law_index] = values[indices]
        gaussian = noise_generator.standard_normal((stop, spec.steps, spec.latent_dim), dtype=np.float64)
        noise[:, law_index] = gaussian[particle_offset:].transpose(1, 0, 2).astype(np.float32)
        streams.append({"request": request.identity, "source_stream_sha256": source_key,
                        "noise_stream_sha256": noise_key, "source_support": len(values),
                        "positive_source_weights": int(np.count_nonzero(weights))})
    state_tensor, noise_tensor = torch.from_numpy(states), torch.from_numpy(noise)
    if destination.type == "cuda":
        if pin_memory:
            state_tensor, noise_tensor = state_tensor.pin_memory(), noise_tensor.pin_memory()
        state_tensor = state_tensor.to(destination, non_blocking=pin_memory)
        noise_tensor = noise_tensor.to(destination, non_blocking=pin_memory)
    receipt = {
        "schema": SAMPLER_VERSION, "context": asdict(context), "specification": asdict(spec),
        "configuration_in_draw_key": False, "microbatch_or_worker_in_draw_key": False,
        "bit_generator": "PCG64DXSM", "normal_transform": "NumPy_float64_standard_normal_then_FP32",
        "noise_draw_order": "particle,step,coordinate_then_transpose_step,law,particle,coordinate",
        "source_sampling": "iid_normalized_source_weight_categorical_with_replacement",
        "source_weight_sum_tolerance": 1e-7,
        "post_sampling_integration": "equal_weight_particles_no_second_source_weight_factor",
        "law_count": len(requests), "particles_returned_per_law": particles,
        "particle_offset": particle_offset, "generated_prefix_per_law": stop,
        "source_uniform_draws": len(requests) * stop,
        "standard_normal_draws": len(requests) * stop * spec.steps * spec.latent_dim,
        "source_indices_device": "cpu", "state_noise_device": str(destination),
        "pinned_transfer": pin_memory, "streams": streams,
        "scientific_input_provenance_verified_here": False,
    }
    return ParticleBatch(state_tensor, noise_tensor, torch.from_numpy(source_indices), receipt)


def sample_scale_sources(requests, context, *, particles, specification=None, device="cpu"):
    """Exact source stream of sample_batch at purpose=scales, without noise.

    This is not a different sampler or a new random seed. It skips only the
    independent standard-normal stream that a persistence calculation does not
    use. Existing training/cache/forecast draws remain unchanged.
    """
    if (not isinstance(context, SamplingContext) or context.purpose != "scales"
            or context.model_seed != 0 or context.accepted_update != 0 or context.generation != 0):
        raise ValueError("Fixed fitting-scale purpose/seed/update/generation required")
    spec = SamplerSpecification() if specification is None else specification
    if not isinstance(spec, SamplerSpecification) or spec.numpy_version != np.__version__:
        raise ValueError("Exact sampler specification required")
    requests = tuple(requests)
    if (not 0 < len(requests) <= spec.maximum_laws
            or not all(isinstance(r, LawRequest) for r in requests)
            or type(particles) is not int or not 1 <= particles <= spec.maximum_particles):
        raise ValueError("Explicit bounded law and particle inventory required")
    if len({_sha256(r.identity) for r in requests}) != len(requests):
        raise ValueError("Duplicate scoped scale law and occurrence")
    if any(r.law.identity.donor not in context.training_donors for r in requests):
        raise ValueError("Fitting scales cannot contain a held-out donor")
    destination = torch.device(device)
    if destination.type not in ("cpu", "cuda") or (destination.type == "cuda" and destination.index is None):
        raise ValueError("CPU or explicitly indexed CUDA destination required")
    states = np.empty((len(requests), particles, spec.latent_dim), dtype=np.float32)
    indices = np.empty((len(requests), particles), dtype=np.int64)
    streams = []
    for i, request in enumerate(requests):
        values, weights = request.law.arrays(spec.latent_dim)
        generator, key = _generator(context, request, "source_categorical")
        cdf = np.cumsum(weights, dtype=np.float64)
        cdf[-1] = 1.
        indices[i] = np.searchsorted(cdf, generator.random(particles, dtype=np.float64), side="right")
        states[i] = values[indices[i]]
        streams.append({"request": request.identity, "source_stream_sha256": key,
                        "source_support": len(values), "positive_source_weights": int(np.count_nonzero(weights))})
    tensor = torch.from_numpy(states).to(destination)
    return ScaleSourceBatch(tensor, torch.from_numpy(indices), {
        "schema": SAMPLER_VERSION, "context": asdict(context), "specification": asdict(spec),
        "source_sampling": "iid_normalized_source_weight_categorical_with_replacement",
        "post_sampling_integration": "equal_weight_particles_no_second_source_weight_factor",
        "source_uniform_draws": len(requests) * particles, "standard_normal_draws": 0,
        "unused_noise_omitted": True, "state_device": str(destination), "source_indices_device": "cpu",
        "streams": streams, "scientific_input_provenance_verified_here": False})
