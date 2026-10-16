"""Accepted-update owner for one bound frontier trajectory, not a scheduler.

Scientific use is CUDA-only. Providers supply already admitted fitting-only
inputs, exact observation/catalog identities, and explicit source/noise draws.
This module never opens a biological payload or chooses scales, auxiliary
coefficients, support shrinkage, donors, or an observation model. Adam and its
state are fresh once at joint-phase entry, then preserved across promotions.

The new accepted-update cache policy is explicit: refresh immediately before
updates 257, 321, 385, ... using 32 particles per modeled law. Following each
healthy Adam update, age the cache once and write the active PRE-update values.
No terminal refresh is performed. A failed/incomplete attempt is terminal;
recovery from an earlier checkpoint requires a separately implemented decision,
not an implicit retry by this owner. Checkpoints capture model/optimizer, global
RNG, provider sampler state, cache, phase and counters. They cannot resurrect
uncheckpointed accepted work after a process loss.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Protocol
import uuid

import torch
from torch import Tensor

from .checkpoints import TrajectoryLease, file_sha256, load_checkpoint, save_checkpoint
from .configuration import STAGE_UPDATES, get_configuration
from .count_objective import CatalogBlock, FitnessCache, sampling_scale_from_sizes
from .geometry import RaggedSinkhornGeometry
from .logical_batch import _module_identity
from .models import FrontierModel
from .observation import SparseR48
from .training_objective import (LogicalTrainingInputs, MolecularTrainingInputs,
    ObjectiveExecution, ObjectiveProtocol, accumulate_training_objective)


SCHEMA = "credo.frontier.accepted_update_owner.v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class OwnerSchedule:
    execution_scope: str = "scientific"
    stages: tuple[int, ...] = STAGE_UPDATES
    state_updates: int = 256
    cache_period: int = 64
    cache_particles: int = 32
    checkpoint_period: int = 256
    logical_batch_laws: int = 1024
    training_particles: int = 128
    maximum_refresh_laws: int = 256

    def __post_init__(self):
        names = ("state_updates", "cache_period", "cache_particles", "checkpoint_period",
                 "logical_batch_laws", "training_particles", "maximum_refresh_laws")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in names):
            raise ValueError("Explicit positive integer owner budgets required")
        if (type(self.stages) is not tuple or not self.stages
                or any(type(x) is not int or x < 1 for x in self.stages)
                or self.stages != tuple(sorted(set(self.stages)))
                or self.state_updates >= self.stages[-1]):
            raise ValueError("Increasing accepted-update stages including a joint phase required")
        if self.execution_scope == "scientific":
            expected = (STAGE_UPDATES, 256, 64, 32, 256, 1024, 128, 256)
            if (self.stages, *(getattr(self, key) for key in names)) != expected:
                raise ValueError("Scientific accepted-update schedule is fixed")
        elif self.execution_scope == "synthetic_cpu_only":
            if (self.stages[-1] > 8 or self.logical_batch_laws > 64 or self.training_particles > 128
                    or self.maximum_refresh_laws > 64 or self.cache_particles != 32):
                raise ValueError("Fabricated CPU fixture exceeds its small numerical scope")
        elif self.execution_scope == "synthetic_cuda_qualification":
            # A disposable, full-batch CUDA fixture, never a shortened
            # scientific fit. Actual invocation still needs separate authority.
            expected = (((2, 3, 4), 1, 2, 32, 1, 1024, 128, 256),
                        ((2, 3, 4), 1, 2, 32, 1, 4, 4, 4))
            # The second exact shape is the compact file-backed worker probe.
            # It cannot stand in for the full-axis owner/memory qualification.
            if (self.stages, *(getattr(self, key) for key in names)) not in expected:
                raise ValueError("CUDA qualification schedule/shape is fixed and disposable")
        else:
            raise ValueError("Explicit scientific, CPU-fixture or CUDA-qualification scope required")

    def phase_after(self, accepted_updates):
        return "state" if accepted_updates <= self.state_updates else "joint"

    def refresh_before(self, accepted_updates):
        return (accepted_updates >= self.state_updates
                and (accepted_updates - self.state_updates) % self.cache_period == 0)


@dataclass(frozen=True)
class FittingInventory:
    """Provider's immutable cache-index routing and already admitted input IDs."""
    training_donors: tuple[str, ...]
    law_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    input_sha256: str
    manifest_sha256: str
    sampler_sha256: str
    observation_sha256: str

    def __post_init__(self):
        n = len(self.law_ids)
        if (self.training_donors != tuple(sorted(set(self.training_donors)))
                or len(self.training_donors) not in (2, 3)
                or not set(self.training_donors) <= {"D1", "D2", "D3", "D4"}
                or not n or len(set(self.law_ids)) != n
                or len(self.donor_ids) != n or len(self.target_ids) != n
                or not set(self.donor_ids) <= set(self.training_donors)
                or any(type(values) is not tuple for values in (self.training_donors, self.law_ids, self.donor_ids, self.target_ids))
                or any(not isinstance(value, str) or not value for value in (*self.law_ids, *self.target_ids))
                or any(not _sha(getattr(self, name)) for name in
                       ("input_sha256", "manifest_sha256", "sampler_sha256", "observation_sha256"))):
            raise ValueError("Exact fitting-only donor/law/target routing and input hashes required")


@dataclass(frozen=True)
class PreparedUpdate:
    inputs: LogicalTrainingInputs
    molecular: MolecularTrainingInputs | None
    sampling_scale: float


@dataclass(frozen=True)
class CacheRefreshBatch:
    cache_indices: Tensor
    law_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    source: Tensor
    noise: Tensor


@dataclass(frozen=True)
class ObservationBinding:
    """Artifact identity plus actual module digest, NOT a loader verification.

    The source adapter must first verify the artifact's bytes and load exactly
    that artifact. Supplying an arbitrary SHA here does not prove that step.
    The owner verifies the actual frozen parameters/scaler against state_sha256
    and binds both identities, preventing an unnoticed different decoder.
    """
    artifact_sha256: str
    state_sha256: str

    def __post_init__(self):
        if not _sha(self.artifact_sha256) or not _sha(self.state_sha256):
            raise ValueError("Exact observation artifact and decoder-state SHA-256 required")


def observation_state_digest(decoder):
    """Hash actual frozen weights, feature/fold identity, and all scaler buffers.

    Run once when admitting/restoring the module; per-update mutation guards use
    tensor identities/versions instead of repeatedly copying GPU weights to CPU.
    """
    if not isinstance(decoder, SparseR48) or decoder.training or any(p.requires_grad for p in decoder.parameters()):
        raise ValueError("Actual frozen observation decoder required")
    decoder._check_scaling()
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(fold=asdict(decoder.fold), features=decoder.features,
        latent_dim=decoder.latent_dim, hidden_dims=decoder.hidden_dims), sort_keys=True).encode())
    for name, tensor in sorted(decoder.state_dict().items()):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite frozen observation state")
        value = tensor.detach().to("cpu").contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FittingProvider(Protocol):
    """No mutation of model/cache; no outcomes outside the bound fitting donors.

    ``position`` in state_dict counts committed accepted updates. sample() and
    cache_batches() may advance explicitly saved sampler generators, not that
    counter. accepted() advances it only after a healthy Adam/cache update.
    Global RNG is additionally saved by the checkpoint layer. Providers must
    validate their own input hashes and snapshots; this metadata interface is
    not proof that an external dataset actually has the declared identities.
    """
    inventory: FittingInventory

    def sample(self, *, accepted_updates: int, phase: str) -> PreparedUpdate: ...
    def cache_batches(self, *, accepted_updates: int, particles: int,
                      maximum_laws: int) -> Iterable[CacheRefreshBatch]: ...
    def accepted(self, *, accepted_updates: int) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


def owner_binding(*, model, protocol, execution, geometry, inventory, schedule,
                  decoder=None, observation_binding=None, maximum_checkpoint_bytes=None):
    """Put this exact binding under lease.identity['training_owner'] before use."""
    if (not isinstance(model, FrontierModel) or not isinstance(protocol, ObjectiveProtocol)
            or not isinstance(execution, ObjectiveExecution) or not isinstance(schedule, OwnerSchedule)
            or not isinstance(inventory, FittingInventory) or not isinstance(geometry, RaggedSinkhornGeometry)):
        raise ValueError("Actual owner components are required for identity binding")
    if maximum_checkpoint_bytes is not None and (type(maximum_checkpoint_bytes) is not int
                                                or maximum_checkpoint_bytes < 1):
        raise ValueError("Positive integer checkpoint payload cap required")
    from . import training_objective
    support = model.support_factors.detach().cpu().tolist() if model.config.target_residual else None
    molecular = protocol.endpoint_coefficient > 0 or protocol.effect_coefficient > 0
    observation = None
    if molecular or decoder is not None or observation_binding is not None:
        if (not isinstance(observation_binding, ObservationBinding) or not isinstance(decoder, SparseR48)
                or observation_binding.artifact_sha256 != inventory.observation_sha256
                or decoder.fold.training_donors != protocol.training_donors
                or observation_state_digest(decoder) != observation_binding.state_sha256):
            raise ValueError("Actual decoder/fold/scaler differs from the observation binding")
        observation = asdict(observation_binding)
    return {
        "schema": SCHEMA, "protocol_sha256": _hash(asdict(protocol)),
        "objective_source_sha256": file_sha256(training_objective.__file__),
        "owner_source_sha256": file_sha256(__file__),
        "configuration_sha256": _hash(asdict(model.config)),
        "model_target_order_sha256": _hash([list(model.target_ids), list(model.control_ids)]),
        "model_support_sha256": _hash(support),
        "fitting_inventory_sha256": _hash(asdict(inventory)),
        "execution_sha256": _hash(asdict(execution)),
        "geometry_sha256": _hash(asdict(geometry.options)),
        "schedule_sha256": _hash(asdict(schedule)),
        "observation_binding": observation,
        "maximum_checkpoint_bytes": maximum_checkpoint_bytes,
    }


def _publish_state(path, payload):
    """One atomically replaced owner journal; historical checkpoints stay intact."""
    if path.is_symlink():
        raise ValueError("Owner journal cannot be a symlink")
    temporary = path.parent / (path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TrainingOwner:
    """One leased model trajectory; each run_stage consumes exactly one stage.

    No scheduler, retry, data discovery, selection, forecast or automatic stage
    promotion occurs. The caller retains the lease throughout use. Reopening
    requires an explicit latest complete checkpoint receipt and matching journal.
    A CUDA device alone does not establish scientific launch authorization.
    """

    def __init__(self, *, lease: TrajectoryLease, model: FrontierModel,
                 provider: FittingProvider, protocol: ObjectiveProtocol,
                 execution: ObjectiveExecution, geometry: RaggedSinkhornGeometry,
                 blocks: tuple[CatalogBlock, ...], cache: FitnessCache,
                 schedule: OwnerSchedule, decoder: SparseR48 | None = None,
                 observation_binding: ObservationBinding | None = None,
                 resume: dict | None = None, retention_policy=None, maximum_checkpoint_bytes=None,
                 resource_guard=None):
        lease.assert_active()
        self.lease, self.model, self.provider = lease, model, provider
        self.protocol, self.execution, self.geometry = protocol, execution, geometry
        self.blocks, self.cache, self.schedule, self.decoder = blocks, cache, schedule, decoder
        self.maximum_checkpoint_bytes = maximum_checkpoint_bytes
        self.inventory = provider.inventory
        binding = owner_binding(model=model, protocol=protocol, execution=execution,
            geometry=geometry, inventory=self.inventory, schedule=schedule,
            decoder=decoder, observation_binding=observation_binding,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes)
        if (lease.identity.get("training_owner") != binding
                or lease.identity["input_sha256"] != self.inventory.input_sha256
                or lease.identity["configuration_sha256"] != binding["configuration_sha256"]
                or model.config != get_configuration(model.config.configuration_id)
                or protocol.training_donors != self.inventory.training_donors
                or protocol.family != model.config.family):
            raise ValueError("Lease/protocol/model/fitting provider binding mismatch")
        self.binding = binding
        self.decoder_identity = None if decoder is None else _module_identity(decoder)
        devices = {parameter.device for parameter in model.parameters()}
        if len(devices) != 1 or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("One FP32 model device is required")
        self.device = next(iter(devices))
        if ((schedule.execution_scope in ("scientific", "synthetic_cuda_qualification") and self.device.type != "cuda")
                or (schedule.execution_scope == "synthetic_cpu_only"
                    and (self.device.type != "cpu" or torch.get_num_threads() > 2))):
            raise ValueError("Scientific owner is CUDA-only; fabricated CPU owner requires at most two threads")
        if schedule.execution_scope in ("scientific", "synthetic_cuda_qualification") and maximum_checkpoint_bytes is None:
            raise ValueError("Scientific main fitting requires an admitted checkpoint payload cap")
        if (not isinstance(cache, FitnessCache) or cache.values.device != self.device
                or cache.values.dtype != torch.float32 or len(cache.values) != len(self.inventory.law_ids)
                or schedule.logical_batch_laws > len(cache.values)
                or tuple(block.donor_id for block in blocks) != protocol.training_donors):
            raise ValueError("Complete fitting catalogs/cache and logical population must match")
        expected_cache_binding = dict(model_recipe=model.config.configuration_id,
            training_subset="-".join(protocol.training_donors), catalog_layout=self.inventory.manifest_sha256,
            source_sampler=self.inventory.sampler_sha256)
        if cache.binding != expected_cache_binding:
            raise ValueError("Cache model/catalog/sampler binding differs from the owner inventory")
        routed = []
        for block in blocks:
            block.validate(cache)
            routed.extend(block.cache_indices.tolist())
            if any(self.inventory.donor_ids[index] != block.donor_id for index in block.cache_indices.tolist()):
                raise ValueError("Catalog cache donor routing differs from fitting inventory")
        if sorted(routed) != list(range(len(cache.values))):
            raise ValueError("Every fitting cache law must have exactly one catalog route")
        self.journal = lease.root / "TRAINING_OWNER_STATE.json"
        self.history = lease.root / "TRAINING_HISTORY.jsonl"
        self.retention_policy = retention_policy
        if retention_policy is not None:
            from .retention import RetentionPolicy, bind_policy
            if (not isinstance(retention_policy, RetentionPolicy)
                    or retention_policy.explicit_milestone_updates != schedule.stages):
                raise ValueError("Retention must preserve every registered owner milestone")
            bind_policy(lease, retention_policy)
        self.accepted_updates = self.attempted_updates = self.optimizer_calls = 0
        self.healthy_optimizer_updates = self.joint_entries = 0
        self.refresh_updates = []
        self.last_checkpoint = None
        self.last_metrics = None
        self.failure = None
        self.status = "ready"
        self.resource_guard = resource_guard
        if resource_guard is not None:
            from .fitting_resources import FittingResourceGuard
            if type(resource_guard) is not FittingResourceGuard:
                raise TypeError("Exact fitting resource guard required")
            resource_guard.bind(self)
        elif schedule.execution_scope in ("scientific", "synthetic_cuda_qualification"):
            raise ValueError("Scientific owner requires its admitted resource guard")
        model.set_phase("state")
        model.train()
        self.optimizer = self._new_adam()
        if resume is None:
            if self.journal.exists() or self.history.exists() or any((lease.root / "checkpoints").iterdir()):
                raise ValueError("Existing owner requires explicit latest checkpoint resume; never restart implicitly")
            self._provider_position(0)
            self._write_status()
        else:
            self._restore(resume)

    def _new_adam(self):
        c = self.model.config
        return torch.optim.Adam(self.model.parameters(), lr=c.learning_rate,
            betas=c.adam_betas, eps=c.adam_epsilon, weight_decay=c.weight_decay,
            amsgrad=False, foreach=False, fused=False)

    def _provider_position(self, expected):
        state = self.provider.state_dict()
        if (not isinstance(state, dict) or type(state.get("position")) is not int
                or state["position"] != expected or self.provider.inventory != self.inventory):
            raise ValueError("Provider accepted-update position or fitting inventory changed")
        return state

    def _assert_decoder(self):
        if self.decoder is not None and _module_identity(self.decoder) != self.decoder_identity:
            raise ValueError("Bound observation weights/scaler/frozen state changed")

    def _state(self):
        return dict(schema=SCHEMA, identity=self.lease.identity, status=self.status,
            accepted_updates=self.accepted_updates, attempted_updates=self.attempted_updates,
            optimizer_calls=self.optimizer_calls, healthy_optimizer_updates=self.healthy_optimizer_updates,
            phase=self.model.phase, joint_entries=self.joint_entries, refresh_updates=self.refresh_updates,
            last_checkpoint=self.last_checkpoint, last_metrics=self.last_metrics,
            retention_policy_sha256=None if self.retention_policy is None else self.retention_policy.identity,
            resource_policy_sha256=None if self.resource_guard is None else self.resource_guard.identity,
            resources=None if self.resource_guard is None else self.resource_guard.record(),
            failure=self.failure,
            automatic_retry=False, scientific_updates=self.accepted_updates if self.schedule.execution_scope == "scientific" else 0,
            disposable_optimizer_calls=self.optimizer_calls if self.schedule.execution_scope != "scientific" else 0)

    def _write_status(self):
        self.lease.assert_active()
        _publish_state(self.journal, self._state())

    def _restore(self, record):
        if not self.journal.is_file() or self.journal.is_symlink():
            raise ValueError("Missing exact accepted-boundary owner journal")
        prior = json.loads(self.journal.read_text())
        u = record.get("accepted_updates")
        if (prior.get("identity") != self.lease.identity or prior.get("schema") != SCHEMA
                or prior.get("status") not in ("ready", "stage_complete")
                or type(u) is not int or not 0 <= u <= self.schedule.stages[-1]
                or prior.get("accepted_updates") != u or prior.get("attempted_updates") != u
                or prior.get("optimizer_calls") != u or prior.get("healthy_optimizer_updates") != u
                or prior.get("retention_policy_sha256") !=
                    (None if self.retention_policy is None else self.retention_policy.identity)
                or prior.get("resource_policy_sha256") !=
                    (None if self.resource_guard is None else self.resource_guard.identity)
                or prior.get("last_checkpoint") != record):
            raise ValueError("Failed/incomplete/uncheckpointed work cannot automatically resume or replay")
        state = load_checkpoint(self.lease, record["directory"], expected_sha256=record["checkpoint_sha256"],
            model=self.model, optimizer=self.optimizer, scheduler=None, execution_scope=self.schedule.execution_scope)
        sampler = state["sampler_state"]
        expected_phase = self.schedule.phase_after(u)
        expected_entries = int(u > self.schedule.state_updates)
        if (state["accepted_updates"] != u or state["phase"] != expected_phase
                or sampler.get("position") != u or sampler.get("binding") != self.binding
                or sampler.get("joint_entries") != expected_entries
                or sampler.get("resource_policy_sha256") != prior.get("resource_policy_sha256")
                or sampler.get("refresh_updates") != prior.get("refresh_updates")):
            raise ValueError("Checkpoint owner phase/cache/sampler boundary is inconsistent")
        self.cache = FitnessCache.from_state_dict(state["count_cache"], expected_binding=self.cache.binding,
            expected_size=len(self.cache.values), expected_dtype=self.cache.values.dtype, device=self.device)
        self.provider.load_state_dict(sampler["provider_state"])
        self._provider_position(u)
        self.accepted_updates = self.attempted_updates = self.optimizer_calls = self.healthy_optimizer_updates = u
        self.joint_entries = expected_entries
        self.refresh_updates = list(sampler["refresh_updates"])
        self.last_checkpoint, self.last_metrics = record, prior.get("last_metrics")
        self.status = prior["status"]

    def _routing(self, batch):
        indices = batch.cache_indices
        if (not isinstance(indices, Tensor) or indices.dtype != torch.int64 or indices.ndim != 1
                or indices.device != self.device or not len(indices) or indices.unique().numel() != len(indices)
                or bool(((indices < 0) | (indices >= len(self.cache.values))).any())):
            raise ValueError("Distinct in-range fitting cache indices required")
        rows = indices.tolist()
        for name in ("law_ids", "donor_ids", "target_ids"):
            if getattr(batch, name) != tuple(getattr(self.inventory, name)[row] for row in rows):
                raise ValueError("Sampled law/donor/target routing differs from bound inventory")
        return rows

    @torch.no_grad()
    def _refresh(self):
        values = torch.empty_like(self.cache.values)
        seen = torch.zeros_like(self.cache.seen)
        for batch in self.provider.cache_batches(accepted_updates=self.accepted_updates,
                particles=self.schedule.cache_particles, maximum_laws=self.schedule.maximum_refresh_laws):
            if not isinstance(batch, CacheRefreshBatch):
                raise ValueError("Full cache refresh requires explicit source/noise batches")
            rows = self._routing(batch)
            if (len(rows) > self.schedule.maximum_refresh_laws or bool(seen[batch.cache_indices].any())
                    or batch.source.shape != (len(rows), self.schedule.cache_particles, 48)
                    or batch.noise.shape != (8, *batch.source.shape)
                    or any(t.device != self.device or t.dtype != torch.float32 or t.requires_grad
                           or not bool(torch.isfinite(t).all()) for t in (batch.source, batch.noise))):
                raise ValueError("Refresh source/noise/count/coverage violates bound policy")
            result = self.model(batch.source, batch.noise, target_ids=batch.target_ids, compute_anchor=False)
            if not bool(torch.isfinite(result.log_mass_increment).all()):
                raise ValueError("Nonfinite cache refresh log mass")
            values[batch.cache_indices] = result.log_mass_increment
            seen[batch.cache_indices] = True
        if not bool(seen.all()):
            raise ValueError("Full cache refresh omitted fitting laws")
        self._provider_position(self.accepted_updates)
        self.cache.apply_full_refresh(values, update_index=self.accepted_updates,
            particles=self.schedule.cache_particles)
        self.refresh_updates.append(self.accepted_updates)

    def _finite_optimizer(self):
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise RuntimeError("Nonfinite model after optimizer call")
        for state in self.optimizer.state.values():
            if any(isinstance(value, Tensor) and not bool(torch.isfinite(value).all()) for value in state.values()):
                raise RuntimeError("Nonfinite Adam state after optimizer call")

    def _checkpoint(self, *, milestone):
        if self.maximum_checkpoint_bytes != self.binding["maximum_checkpoint_bytes"]:
            raise ValueError("Bound checkpoint payload cap changed during fitting")
        if self.resource_guard is not None:
            self.resource_guard.check("before_checkpoint")
        if self.history.is_symlink():
            raise ValueError("Training history cannot be a symlink")
        if self.history.exists():
            with self.history.open("rb") as handle:
                os.fsync(handle.fileno())
        sampler = dict(position=self.accepted_updates, provider_state=self._provider_position(self.accepted_updates),
            binding=self.binding, joint_entries=self.joint_entries, refresh_updates=list(self.refresh_updates),
            resource_policy_sha256=None if self.resource_guard is None else self.resource_guard.identity)
        self.last_checkpoint = save_checkpoint(self.lease, model=self.model, optimizer=self.optimizer,
            scheduler=None, accepted_updates=self.accepted_updates, attempted_updates=self.attempted_updates,
            phase=self.model.phase, sampler_state=sampler, count_cache=self.cache.state_dict(),
            execution_scope=self.schedule.execution_scope, milestone=milestone,
            maximum_checkpoint_bytes=self.maximum_checkpoint_bytes)
        if self.retention_policy is not None:
            from .retention import prepare_retirement, commit_retirement
            retirement = prepare_retirement(self.lease, self.retention_policy)
            if retirement is not None:
                commit_retirement(self.lease, self.retention_policy, retirement)
        if self.resource_guard is not None:
            self.resource_guard.check("after_checkpoint")

    def run_stage(self, target_updates: int):
        self.lease.assert_active()
        if self.status not in ("ready", "stage_complete"):
            raise RuntimeError("Failed/incomplete owner is terminal; no automatic retry")
        remaining = [n for n in self.schedule.stages if n > self.accepted_updates]
        if not remaining or target_updates != remaining[0]:
            raise ValueError("Exactly the next registered stage may run; no automatic stage expansion")
        try:
            while self.accepted_updates < target_updates:
                if self.resource_guard is not None:
                    self.resource_guard.check("before_update")
                self._assert_decoder()
                self._provider_position(self.accepted_updates)
                self.attempted_updates += 1
                self.status = "attempt_started"
                self._write_status()  # A crash from here cannot silently replay this attempt.
                phase = self.schedule.phase_after(self.accepted_updates + 1)
                if phase != self.model.phase:
                    if phase != "joint" or self.joint_entries or self.accepted_updates != self.schedule.state_updates:
                        raise RuntimeError("Unexpected optimizer phase transition")
                    self.model.set_phase("joint")
                    self.optimizer = self._new_adam()
                    self.joint_entries += 1
                if phase == "joint" and self.schedule.refresh_before(self.accepted_updates):
                    self._refresh()
                prepared = self.provider.sample(accepted_updates=self.accepted_updates, phase=phase)
                if not isinstance(prepared, PreparedUpdate):
                    raise ValueError("Provider must bind one actual logical objective input")
                self._routing(prepared.inputs)
                if (prepared.inputs.source.shape != (self.schedule.logical_batch_laws, self.schedule.training_particles, 48)
                        or prepared.sampling_scale != sampling_scale_from_sizes(population=len(self.cache.values),
                            logical_batch=self.schedule.logical_batch_laws)):
                    raise ValueError("Logical population/particle/sampling scale differs from contract")
                self._provider_position(self.accepted_updates)
                self._assert_decoder()
                self.optimizer.zero_grad(set_to_none=True)
                result = accumulate_training_objective(self.model, prepared.inputs, protocol=self.protocol,
                    execution=self.execution, geometry=self.geometry, blocks=self.blocks, cache=self.cache,
                    sampling_scale=prepared.sampling_scale, decoder=self.decoder, molecular=prepared.molecular)
                self._assert_decoder()
                if not bool(torch.isfinite(result.displayed_objective)):
                    raise RuntimeError("Nonfinite complete objective")
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.config.gradient_clip,
                    error_if_nonfinite=True, foreach=False)
                self.optimizer_calls += 1  # Count invocation even if Adam fails part way through.
                self.optimizer.step()
                self._finite_optimizer()
                self.healthy_optimizer_updates += 1
                self.cache.tick()
                self.cache.update_active(result.logical_result.active_indices, result.logical_result.pre_update_log_mass)
                self.provider.accepted(accepted_updates=self.accepted_updates + 1)
                self._provider_position(self.accepted_updates + 1)
                self.accepted_updates += 1
                if self.resource_guard is not None:
                    self.resource_guard.check("after_update")
                self.last_metrics = dict(update=self.accepted_updates, phase=phase,
                    objective=float(result.displayed_objective), gradient_norm_before_clip=float(norm),
                    count_ce=None if result.logical_result.count_ce is None else float(result.logical_result.count_ce),
                    local_terms={name: float(value) for name, value in result.logical_result.local_weighted_terms.items()},
                    molecular_terms={name: float(value) for name, value in result.molecular_terms.items()},
                    cache_refresh_update=self.cache.last_full_refresh_update,
                    model_forward_calls=result.model_forward_calls)
                if self.history.is_symlink():
                    raise ValueError("Training history cannot be a symlink")
                with self.history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(self.last_metrics, sort_keys=True, allow_nan=False) + "\n")
                self.status = "stage_complete" if self.accepted_updates == target_updates else "ready"
                if self.accepted_updates % self.schedule.checkpoint_period == 0 or self.accepted_updates == target_updates:
                    self._checkpoint(milestone=self.accepted_updates in self.schedule.stages)
                self._write_status()
            if self.resource_guard is not None:
                self.resource_guard.check("stage_complete")
                self._write_status()
        except BaseException as error:
            self.status = "failed"
            self.failure = dict(error_type=type(error).__name__, message=str(error)[:2000],
                optimizer_invoked_beyond_accepted=self.optimizer_calls > self.accepted_updates)
            try:
                self._write_status()
            except BaseException as publication_error:
                error.add_note("Failure journal publication also failed: " + type(publication_error).__name__)
            raise
        return self._state()
