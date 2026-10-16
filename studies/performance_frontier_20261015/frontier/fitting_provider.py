"""Validated in-memory fold cache -> existing fitting-owner inputs.

No file loader, encoder execution, publication owner or scheduler is provided.
The caller supplies immutable arrays admitted from the exact fitting role and
COMPLETE-selected observation model. This adapter validates identities, array
contents/routing and byte budgets; it cannot establish biological provenance or
that an asserted observation model actually produced supplied latent values.

Raw positive-RNA compositions are representation-independent, source-keyed
references shared across folds. Latents/offsets are fold-owned. Zero-RNA capture
remains in full-catalog counts, never state. Every endpoint atom is retained.
Molecular controls are donor pooled source laws, not new catalog categories.

Logical samples are stateless uniform WITHOUT replacement, distinct at each
update; not historical shuffled epochs. State phase samples paired laws, joint
phase all source-supported laws. No redraw repairs an unavailable objective.
Global molecular macro weights are estimated using the separately bound
Horvitz multiplier, not self-normalized sampled-target macro averaging.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
import torch

from .configuration import DONORS, SEEDS, get_configuration
from .count_objective import CatalogBlock, FitnessCache
from .geometry import ObservedLaw
from .measure_ops import source_log_frequencies
from .native_inputs import FoldInputs
from .objective_registry import ObjectiveRules, RULES_SHA256
from .sampling import (LawIdentity, PooledSourceIdentity, LawRequest, SourceLaw,
                       SamplingContext, SamplerSpecification, sample_batch)
from .training_objective import (LogicalTrainingInputs, MolecularTrainingInputs,
                                 PooledControlInputs)
from .training_owner import (CacheRefreshBatch, FittingInventory, ObservationBinding,
                             OwnerSchedule, PreparedUpdate, observation_state_digest)


SCHEMA = "credo.frontier.fold-fitting-cache.v1"
SAMPLING_POLICY = "uniform_distinct_fixed_batch_per_update_pcg64dxsm_v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Exact lowercase SHA-256 identity required")


def _text(value):
    if type(value) is not str or not value:
        raise ValueError("Explicit nonempty identity required")


def _array_hash(value):
    digest = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode())
    # Read-only contiguous arrays/mmap views; no full-size byte-string copy.
    if value.size:
        # memoryview.cast rejects zero-sized multidimensional shapes. The
        # dtype/shape header still uniquely binds an empty state population.
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _array(value, dtype, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype)
            or value.shape != shape or not value.flags.c_contiguous or value.flags.writeable):
        raise ValueError(f"Read-only contiguous exact {dtype} {shape} array required: {name}")


@dataclass(frozen=True)
class CatalogGuide:
    guide_id: str
    target_id: str
    is_control: bool

    def __post_init__(self):
        _text(self.guide_id); _text(self.target_id)
        if type(self.is_control) is not bool:
            raise ValueError("Explicit actual-guide control flag required")


@dataclass(frozen=True)
class PopulationReference:
    donor: str
    condition: str
    source_id: str
    replicate_id: str
    source_access_sha256: str
    source_input_sha256: str
    row_identity_sha256: str
    raw_summary_sha256: str
    latent_sha256: str

    def __post_init__(self):
        if self.donor not in DONORS or self.condition not in ("Stim8hr", "Stim48hr"):
            raise ValueError("Only exact fitting donor/condition sources are supported")
        _text(self.source_id); _text(self.replicate_id)
        for name in ("source_access_sha256", "source_input_sha256", "row_identity_sha256", "raw_summary_sha256", "latent_sha256"):
            _sha(getattr(self, name))


@dataclass(frozen=True)
class PopulationArrays:
    """External immutable arrays, not a claim of verified disk publication.

    Means have one full-gene row for each complete-catalog category. Unavailable
    rows must be all zero AND have positive_counts==0; they are never returned
    as an observed composition. Counts and means can be shared across folds.
    Offsets partition every positive-RNA latent row in canonical guide order.
    """
    reference: PopulationReference
    captured_counts: np.ndarray
    positive_counts: np.ndarray
    offsets: np.ndarray
    states: np.ndarray
    mean_compositions: np.ndarray

    def raw_identity(self, *, catalog_sha256, feature_axis_sha256):
        ref = self.reference
        return _hash(dict(schema="credo.frontier.source-rna-summary.v1", donor=ref.donor,
            condition=ref.condition, source_id=ref.source_id, source_input_sha256=ref.source_input_sha256,
            row_identity_sha256=ref.row_identity_sha256, catalog_sha256=catalog_sha256,
            feature_axis_sha256=feature_axis_sha256,
            definition="equal_positive_RNA_cell_mean_composition_before_log;zero_RNA_capture_retained",
            arrays={name: _array_hash(getattr(self, name)) for name in
                    ("captured_counts", "positive_counts", "mean_compositions")}))

    def latent_identity(self, *, catalog_sha256, observation_binding):
        ref = self.reference
        return _hash(dict(schema="credo.frontier.fold-source-latents.v1", donor=ref.donor,
            condition=ref.condition, source_id=ref.source_id,
            source_access_sha256=ref.source_access_sha256, row_identity_sha256=ref.row_identity_sha256,
            catalog_sha256=catalog_sha256, observation_binding=asdict(observation_binding),
            definition="all_positive_RNA_cells;guide_order_then_bound_native_row_order;standardized48",
            arrays={name: _array_hash(getattr(self, name)) for name in ("offsets", "states")}))


@dataclass(frozen=True)
class FoldCacheManifest:
    fold: FoldInputs
    catalog: tuple[CatalogGuide, ...]
    feature_axis_sha256: str
    features: int
    observation_binding: ObservationBinding
    source_access_sha256: str
    supervision_access_sha256: str
    populations: tuple[PopulationReference, ...]
    rules_sha256: str = RULES_SHA256
    synthetic_fixture: bool = False

    def __post_init__(self):
        if (not isinstance(self.fold, FoldInputs) or not isinstance(self.observation_binding, ObservationBinding)
                or type(self.catalog) is not tuple or not self.catalog
                or not all(isinstance(g, CatalogGuide) for g in self.catalog)
                or len({g.guide_id for g in self.catalog}) != len(self.catalog)
                or not any(g.is_control for g in self.catalog)
                or type(self.features) is not int or self.features < 1 or type(self.synthetic_fixture) is not bool):
            raise ValueError("Exact fold/catalog/observation and feature-axis binding required")
        for name in ("feature_axis_sha256", "source_access_sha256", "supervision_access_sha256", "rules_sha256"):
            _sha(getattr(self, name))
        if self.rules_sha256 != RULES_SHA256:
            raise ValueError("Cache objective/support definition differs from prospective rules")
        if not self.synthetic_fixture and (len(self.catalog), self.features) != (26504, 18129):
            raise ValueError("Scientific cache requires the complete historical catalog and RNA axis")
        if self.synthetic_fixture and (len(self.catalog) > 64 or self.features > 64):
            raise ValueError("Fabricated cache exceeds its small catalog/feature scope")
        controls = {g.target_id for g in self.catalog if g.is_control}
        if controls & {g.target_id for g in self.catalog if not g.is_control}:
            raise ValueError("Target identities must not mix control and targeting categories")
        expected = tuple((donor, condition) for donor in self.fold.training_donors for condition in ("Stim8hr", "Stim48hr"))
        if (type(self.populations) is not tuple or not all(isinstance(r, PopulationReference) for r in self.populations)
                or tuple((r.donor, r.condition) for r in self.populations) != expected
                or len({r.source_id for r in self.populations}) != len(self.populations)):
            raise ValueError("Exactly ordered source/endpoint references for fitting donors required")
        for ref in self.populations:
            expected_access = self.source_access_sha256 if ref.condition == "Stim8hr" else self.supervision_access_sha256
            if ref.source_access_sha256 != expected_access:
                raise ValueError("Source population belongs to a different fitting role capability")

    @property
    def catalog_sha256(self):
        return _hash([asdict(g) for g in self.catalog])

    @property
    def input_sha256(self):
        return _hash(dict(fold=self.fold.identity, source=self.source_access_sha256,
                          supervision=self.supervision_access_sha256))

    @property
    def identity(self):
        return _hash(dict(schema=SCHEMA, **asdict(self)))


@dataclass(frozen=True)
class ProviderLimits:
    maximum_host_bytes: int
    maximum_prepared_bytes: int
    maximum_pool_cells: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Explicit positive host/prepared/pool resource bounds required")


class FoldFittingCache:
    """Validate pre-admitted arrays once; no implicit opening, encoding or loader."""

    def __init__(self, manifest, populations, *, expected_manifest_sha256, limits):
        if not isinstance(manifest, FoldCacheManifest) or not isinstance(limits, ProviderLimits):
            raise ValueError("Typed manifest and explicit memory bounds required")
        _sha(expected_manifest_sha256)
        if manifest.identity != expected_manifest_sha256:
            raise ValueError("Fitting cache manifest content identity differs")
        populations = tuple(populations)
        if (len(populations) != len(manifest.populations)
                or any(not isinstance(p, PopulationArrays) for p in populations)
                or tuple(p.reference for p in populations) != manifest.populations):
            raise ValueError("Exact ordered fitting population arrays required")
        fields = ("captured_counts", "positive_counts", "offsets", "states", "mean_compositions")
        if any(not isinstance(getattr(p, field), np.ndarray) for p in populations for field in fields):
            raise ValueError("Explicit immutable NumPy arrays/mmap views required")
        # Conservative: count every reference even when array storage is shared.
        host = sum(getattr(p, name).nbytes for p in populations for name in fields)
        if host > limits.maximum_host_bytes:
            raise ValueError("Complete fold cache exceeds explicit host-byte cap")
        self.manifest, self.limits = manifest, limits
        self.populations = MappingProxyType({(p.reference.donor, p.reference.condition): p for p in populations})
        g, j = len(manifest.catalog), manifest.features
        for p in populations:
            _array(p.captured_counts, np.int64, (g,), "captured")
            _array(p.positive_counts, np.int64, (g,), "positive RNA")
            _array(p.offsets, np.int64, (g + 1,), "offsets")
            if (np.any(p.captured_counts < 0) or int(p.captured_counts.sum()) <= 0
                    or np.any(p.positive_counts < 0) or np.any(p.positive_counts > p.captured_counts)
                    or p.offsets[0] != 0 or np.any(np.diff(p.offsets) != p.positive_counts)):
                raise ValueError("Full captured/positive counts and exact latent offsets disagree")
            _array(p.states, np.float32, (int(p.positive_counts.sum()), 48), "latents")
            _array(p.mean_compositions, np.float64, (g, j), "raw RNA means")
            if not np.isfinite(p.states).all():
                raise ValueError("Nonfinite encoded positive-RNA state")
            # Gene rows are scanned in bounded blocks, not a second dense cohort.
            for start in range(0, g, 256):
                stop = min(g, start + 256)
                means, positive = p.mean_compositions[start:stop], p.positive_counts[start:stop] > 0
                if (not np.isfinite(means).all() or np.any(means < 0)
                        or not np.allclose(means[positive].sum(1), 1., rtol=0, atol=1e-10)
                        or np.any(means[~positive] != 0)):
                    raise ValueError("Positive-RNA compositions unavailable/normalized rows disagree")
            if (p.raw_identity(catalog_sha256=manifest.catalog_sha256, feature_axis_sha256=manifest.feature_axis_sha256)
                    != p.reference.raw_summary_sha256
                    or p.latent_identity(catalog_sha256=manifest.catalog_sha256, observation_binding=manifest.observation_binding)
                    != p.reference.latent_sha256):
                raise ValueError("Actual raw summary/latent array content differs from its binding")
        self.laws = tuple((donor, index) for donor in manifest.fold.training_donors
                          for index in range(g) if self.populations[(donor, "Stim8hr")].positive_counts[index] > 0)
        self.paired = tuple(index for index, (donor, guide) in enumerate(self.laws)
                            if self.populations[(donor, "Stim48hr")].positive_counts[guide] > 0)
        self.paired_set = frozenset(self.paired)
        if not self.laws or not self.paired:
            raise ValueError("No source or paired state support; no substitute is permitted")
        self.control_target_ids = tuple(sorted({x.target_id for x in manifest.catalog if x.is_control}))
        control_positions = tuple(i for i, guide in enumerate(manifest.catalog) if guide.is_control)
        self.ntc_catalog_sha256 = _hash([asdict(manifest.catalog[i]) for i in control_positions])
        self.pool_laws, self.pool_truth = [], []
        pool_bytes = 0
        for donor in manifest.fold.training_donors:
            source, endpoint = (self.populations[(donor, c)] for c in ("Stim8hr", "Stim48hr"))
            source_n = sum(int(source.positive_counts[i]) for i in control_positions)
            endpoint_n = sum(int(endpoint.positive_counts[i]) for i in control_positions)
            if not source_n or not endpoint_n:
                raise ValueError("Required pooled-NTC source/endpoint unavailable; no replacement")
            if source_n > limits.maximum_pool_cells:
                raise ValueError("Full NTC pool exceeds explicit support cap; no truncation")
            pool_bytes += source_n * (48 * 4 + 8) + j * 8
        if host + pool_bytes > limits.maximum_host_bytes:
            raise ValueError("Fold cache plus complete NTC pools exceeds host-byte cap")
        for donor in manifest.fold.training_donors:
            source, endpoint = (self.populations[(donor, c)] for c in ("Stim8hr", "Stim48hr"))
            pieces = [source.states[source.offsets[i]:source.offsets[i + 1]] for i in control_positions if source.positive_counts[i]]
            states = np.concatenate(pieces)
            weights = np.full(len(states), 1. / len(states), np.float64)
            states.setflags(write=False); weights.setflags(write=False)
            definition = _hash(dict(policy="all_NTC_positive_RNA_source_cells_equal", source=source.reference.latent_sha256,
                                    raw_summary=source.reference.raw_summary_sha256, catalog=self.ntc_catalog_sha256))
            ref = source.reference
            identity = PooledSourceIdentity(donor, "Stim8hr", ref.source_id, ref.replicate_id,
                f"{donor}::pooled-NTC-RNA", self.ntc_catalog_sha256, definition)
            self.pool_laws.append(SourceLaw(identity, states, weights))
            n = sum(int(endpoint.positive_counts[i]) for i in control_positions)
            mean = sum(endpoint.mean_compositions[i] * (int(endpoint.positive_counts[i]) / n)
                       for i in control_positions if endpoint.positive_counts[i])
            mean.setflags(write=False)
            self.pool_truth.append(mean)
        self.pool_laws, self.pool_truth = tuple(self.pool_laws), tuple(self.pool_truth)
        self.host_bytes = host + pool_bytes
        # Fixed full-dataset target macro measure, independent of batch contents.
        groups = {}
        for cache_index in self.paired:
            donor, guide = self.laws[cache_index]
            entry = manifest.catalog[guide]
            if not entry.is_control:
                groups.setdefault(donor, {}).setdefault(entry.target_id, []).append(cache_index)
        if set(groups) != set(manifest.fold.training_donors):
            raise ValueError("Each fitting donor needs supported targeting molecular laws")
        self.molecular_weights = {}
        for donor, targets in groups.items():
            for indices in targets.values():
                weight = 1. / (len(groups) * len(targets) * len(indices))
                self.molecular_weights.update({index: weight for index in indices})
        self.molecular_weights = MappingProxyType(self.molecular_weights)
        self._array_flags = tuple((id(getattr(p, name)), getattr(p, name).shape, getattr(p, name).dtype.str,
                                  getattr(p, name).ctypes.data) for p in populations for name in fields)
        self._metadata_fields = ("manifest", "populations", "laws", "paired", "paired_set", "molecular_weights",
                                 "pool_laws", "pool_truth", "control_target_ids", "ntc_catalog_sha256")
        self._metadata_ids = tuple(id(getattr(self, name)) for name in self._metadata_fields)

    def assert_unchanged(self):
        # Read-only flags detect accidental ordinary mutation. A production
        # owner must also verify file/input leases; these are not OS isolation.
        fields = ("captured_counts", "positive_counts", "offsets", "states", "mean_compositions")
        if tuple(id(getattr(self, name)) for name in self._metadata_fields) != self._metadata_ids:
            raise ValueError("Bound cache metadata/routing changed")
        values = tuple(self.populations[(r.donor, r.condition)] for r in self.manifest.populations)
        stamp = tuple((id(getattr(p, name)), getattr(p, name).shape, getattr(p, name).dtype.str,
                       getattr(p, name).ctypes.data) for p in values for name in fields)
        if stamp != self._array_flags or any(getattr(p, name).flags.writeable for p in values for name in fields):
            raise ValueError("Bound cache storage or read-only policy changed")
        if any(law.states.flags.writeable or law.weights.flags.writeable for law in self.pool_laws) or any(
                mean.flags.writeable for mean in self.pool_truth):
            raise ValueError("Pooled-control storage is no longer read-only")

    def verify_decoder(self, decoder, observation_binding):
        if (observation_binding != self.manifest.observation_binding
                or decoder.fold.fold_id != self.manifest.fold.fold_id
                or decoder.fold.training_donors != self.manifest.fold.training_donors
                or decoder.fold.feature_axis_sha256 != self.manifest.feature_axis_sha256
                or decoder.features != self.manifest.features
                or observation_state_digest(decoder) != observation_binding.state_sha256):
            raise ValueError("Loaded observation module differs from cache representation")


class FoldFittingProvider:
    """Existing FittingProvider interface; no optimizer/cache-state ownership."""

    def __init__(self, cache, *, rules, configuration_id, model_seed, study_id, study_seed,
                 schedule, device="cpu", pin_memory=False):
        if (not isinstance(cache, FoldFittingCache) or not isinstance(rules, ObjectiveRules)
                or not isinstance(schedule, OwnerSchedule) or rules.identity != cache.manifest.rules_sha256):
            raise ValueError("Bound fold cache, objective rules and owner schedule required")
        config = get_configuration(configuration_id)
        _text(study_id)
        if type(model_seed) is not int or model_seed not in SEEDS or type(study_seed) is not int or not 0 <= study_seed < 2**64:
            raise ValueError("Exact registered model seed and explicit study seed required")
        self.device = torch.device(device)
        if (self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and self.device.index is None)
                or type(pin_memory) is not bool or (pin_memory and self.device.type != "cuda")):
            raise ValueError("Explicit CPU or indexed CUDA destination and transfer policy required")
        if cache.manifest.synthetic_fixture != (schedule.execution_scope in ("synthetic_cpu_only", "synthetic_cuda_qualification")):
            raise ValueError("Synthetic versus scientific cache/schedule scope differs")
        if cache.manifest.synthetic_fixture and schedule.execution_scope == "synthetic_cuda_qualification":
            from .main_qualification_worker_plan import worker_schedule
            if schedule != worker_schedule() or self.device != torch.device("cuda:0") or torch.get_num_threads() != 8:
                raise ValueError("Exact compact CUDA worker fixture and eight threads required")
        elif cache.manifest.synthetic_fixture and self.device.type != "cpu":
            raise ValueError("Fabricated provider tests are CPU-only")
        if min(len(cache.laws), len(cache.paired)) < schedule.logical_batch_laws:
            raise ValueError("Uniform distinct logical batch exceeds full phase support")
        self.cache, self.schedule = cache, schedule
        self.configuration_id, self.model_seed = configuration_id, model_seed
        self.study_id, self.study_seed = study_id, study_seed
        self.pin_memory = pin_memory
        self.position, self._prepared_position = 0, None
        self.specification = SamplerSpecification()
        self.molecular_enabled = rules.document["family_auxiliaries"][config.family]["both_endpoints"] > 0
        manifest = cache.manifest
        sampler_binding = dict(policy=SAMPLING_POLICY, study_id=study_id, study_seed=study_seed,
            model_seed=model_seed, configuration_id=configuration_id, schedule=asdict(schedule),
            particle_specification=asdict(self.specification), rules_sha256=rules.identity,
            cache_manifest_sha256=manifest.identity, common_random_configuration_excluded=True,
            unavailable_logical_objective="fail_no_redraw", numpy_version=np.__version__)
        self.inventory = FittingInventory(manifest.fold.training_donors,
            tuple(f"{d}::guide-index-{g}" for d, g in cache.laws),
            tuple(d for d, _ in cache.laws), tuple(manifest.catalog[g].target_id for _, g in cache.laws),
            manifest.input_sha256, manifest.identity, _hash(sampler_binding), manifest.observation_binding.artifact_sha256)
        self.binding = _hash(dict(inventory=asdict(self.inventory), device=str(self.device), pin_memory=pin_memory,
                                  limits=asdict(cache.limits)))
        self._ordered_laws_sha256 = _hash(self.inventory.law_ids)
        # Bytes-backed routing arrays cannot be made writable. Their identities
        # are checked below without rescanning the catalog at every update.
        self._state_indices = np.frombuffer(np.asarray(cache.paired, dtype=np.int64).tobytes(), dtype=np.int64)
        self._all_indices = np.frombuffer(np.arange(len(cache.laws), dtype=np.int64).tobytes(), dtype=np.int64)
        self._bound_context = self._context_stamp()

    def _context_stamp(self):
        # Only the two accepted-position fields may advance. Frozen inventory
        # tuples and bytes-backed routing are identity checked, while small
        # context/configuration values are checked by content. This avoids a
        # repeated full-law hash or any tensor/device work in the guard.
        return (id(self.cache), id(self.inventory), _hash(dict(
            study_id=self.study_id, study_seed=self.study_seed, model_seed=self.model_seed,
            configuration_id=self.configuration_id, schedule=asdict(self.schedule),
            specification=asdict(self.specification), device=str(self.device),
            device_type=type(self.device).__name__, pin_memory=self.pin_memory,
            molecular_enabled=self.molecular_enabled, limits=asdict(self.cache.limits),
            binding=self.binding, ordered_laws_sha256=self._ordered_laws_sha256)),
            tuple((id(array), array.dtype.str, array.shape, array.ctypes.data, array.flags.writeable)
                  for array in (self._state_indices, self._all_indices)))

    def _assert_context(self):
        try:
            unchanged = self._context_stamp() == self._bound_context
        except (AttributeError, TypeError, ValueError):
            unchanged = False
        if not unchanged:
            raise ValueError("Bound fitting-provider context/routing/transfer settings changed")

    def _context(self, update, purpose):
        return SamplingContext(self.study_id, self.inventory.training_donors, self.configuration_id,
                               self.study_seed, self.model_seed, update, purpose)

    def selected_indices(self, *, accepted_updates, phase):
        self._position(accepted_updates)
        expected = self.schedule.phase_after(accepted_updates + 1)
        if phase != expected:
            raise ValueError("Provider phase differs from the next accepted update")
        population = self._state_indices if phase == "state" else self._all_indices
        # Do not include configuration, worker, microbatch or wall-clock state.
        key = dict(schema=SAMPLING_POLICY, study=self.study_id, fitting_donors=self.inventory.training_donors,
                   study_seed=self.study_seed, model_seed=self.model_seed, update=accepted_updates,
                   phase=phase, ordered_laws_sha256=self._ordered_laws_sha256)
        rng = np.random.Generator(np.random.PCG64DXSM(int(_hash(key), 16)))
        return tuple(int(index) for index in rng.choice(population, self.schedule.logical_batch_laws, replace=False))

    def _position(self, expected):
        self._assert_context()
        if type(expected) is not int or expected != self.position or not 0 <= expected < self.schedule.stages[-1]:
            raise ValueError("Exact committed provider update position required")
        self.cache.assert_unchanged()

    def _requests(self, indices):
        requests = []
        for index in indices:
            donor, guide_index = self.cache.laws[index]
            p = self.cache.populations[(donor, "Stim8hr")]
            start, stop = p.offsets[guide_index:guide_index + 2]
            ref, guide = p.reference, self.cache.manifest.catalog[guide_index]
            law = SourceLaw(LawIdentity(donor, "Stim8hr", guide.guide_id, ref.source_id, ref.replicate_id),
                            p.states[start:stop], np.full(stop - start, 1. / (stop - start), np.float64))
            requests.append(LawRequest(law))
        return requests

    def _transfer(self, value, *, dtype=None):
        # Copy CPU arrays so downstream tensors never share writable aliases.
        return torch.tensor(value, dtype=dtype, device=self.device)

    def _prepared_bound(self, indices, particles, *, molecular, endpoints):
        source_cells = sum(int(self.cache.populations[(d, "Stim8hr")].positive_counts[g]) for d, g in (self.cache.laws[i] for i in indices))
        endpoint_cells = (sum(int(self.cache.populations[(d, "Stim48hr")].positive_counts[g]) for d, g in
                              (self.cache.laws[i] for i in indices)) if endpoints else 0)
        rows = len(indices) + (len(self.inventory.training_donors) if molecular else 0)
        # Conservative CPU+device sampled state/noise/index staging, endpoint
        # tensors, source weights, and molecular truth/weight stacks. Model and
        # decoder backward workspaces are separately guarded by ObjectiveExecution.
        size = rows * particles * (9 * 48 * 4 + 8) * 3 + source_cells * 16 + endpoint_cells * (48 * 4 + 4) * 2
        if molecular:
            size += (len(indices) + len(self.inventory.training_donors)) * self.cache.manifest.features * 8 * 3
        if size > self.cache.limits.maximum_prepared_bytes:
            raise ValueError("Complete prepared-input envelope exceeds byte cap; no truncation")
        return size

    def sample(self, *, accepted_updates, phase):
        # A failed replacement request cannot leave an earlier prepared input
        # eligible for acceptance. The owner separately makes failures terminal.
        self._prepared_position = None
        indices = self.selected_indices(accepted_updates=accepted_updates, phase=phase)
        molecular = self.molecular_enabled and phase == "joint"
        if not any(index in self.cache.paired_set for index in indices):
            raise ValueError("Sampled batch has no observed state law; unavailable, fail without redraw")
        targets = tuple((position, index) for position, index in enumerate(indices) if index in self.cache.molecular_weights)
        if molecular and not targets:
            raise ValueError("Sampled batch has no supported targeting molecular law; unavailable, fail without redraw")
        self._prepared_bound(indices, self.schedule.training_particles, molecular=molecular, endpoints=True)
        sampled = sample_batch(self._requests(indices), self._context(accepted_updates, "training"),
            particles=self.schedule.training_particles, specification=self.specification,
            device=self.device, pin_memory=self.pin_memory)
        observations = []
        for index in indices:
            donor, guide = self.cache.laws[index]
            p = self.cache.populations[(donor, "Stim48hr")]
            start, stop = map(int, p.offsets[guide:guide + 2])
            observations.append(None if start == stop else ObservedLaw(self.inventory.law_ids[index],
                self._transfer(p.states[start:stop]),
                torch.full((stop - start,), 1. / (stop - start), dtype=torch.float32, device=self.device)))
        batch = LogicalTrainingInputs(tuple(self.inventory.law_ids[i] for i in indices),
            tuple(self.inventory.donor_ids[i] for i in indices), tuple(self.inventory.target_ids[i] for i in indices),
            sampled.states, sampled.noise, self._transfer(indices, dtype=torch.int64), tuple(observations),
            torch.ones(len(indices), dtype=torch.float32, device=self.device),
            torch.full((len(indices),), 1. / len(indices), dtype=torch.float32, device=self.device))
        molecular_inputs = None
        if molecular:
            pool = sample_batch(tuple(LawRequest(law) for law in self.cache.pool_laws),
                self._context(accepted_updates, "training"), particles=self.schedule.training_particles,
                specification=self.specification, device=self.device, pin_memory=self.pin_memory)
            donors = self.inventory.training_donors
            control = PooledControlInputs(donors, tuple(law.identity.pool_id for law in self.cache.pool_laws),
                torch.arange(len(donors), dtype=torch.int64, device=self.device),
                (self.cache.control_target_ids[0],) * len(donors), pool.states, pool.noise,
                torch.ones(len(donors), dtype=torch.float64, device=self.device),
                self._transfer(np.stack(self.cache.pool_truth)),
                torch.full((len(donors),), 1. / len(donors), dtype=torch.float64, device=self.device))
            raw_weights = np.asarray([self.cache.molecular_weights[index] for _, index in targets], np.float64)
            mass = float(raw_weights.sum())
            truths = np.stack([self.cache.populations[(self.cache.laws[index][0], "Stim48hr")].mean_compositions[self.cache.laws[index][1]]
                               for _, index in targets])
            molecular_inputs = MolecularTrainingInputs(self._transfer([row for row, _ in targets], dtype=torch.int64),
                self._transfer(truths), self._transfer(raw_weights / mass), control,
                target_estimator_multiplier=(len(self.cache.laws) / len(indices)) * mass)
        self._assert_context()
        self._prepared_position = accepted_updates
        return PreparedUpdate(batch, molecular_inputs, len(self.cache.laws) / len(indices))

    def cache_batches(self, *, accepted_updates, particles, maximum_laws):
        self._position(accepted_updates)
        if (particles != self.schedule.cache_particles or maximum_laws != self.schedule.maximum_refresh_laws
                or not self.schedule.refresh_before(accepted_updates)):
            raise ValueError("Exact declared refresh boundary/particles/chunk bound required")
        for start in range(0, len(self.cache.laws), maximum_laws):
            # A generator can be suspended while its consumer handles a batch.
            # Recheck the complete context on every resumed iteration.
            self._position(accepted_updates)
            indices = tuple(range(start, min(start + maximum_laws, len(self.cache.laws))))
            self._prepared_bound(indices, particles, molecular=False, endpoints=False)
            sampled = sample_batch(self._requests(indices), self._context(accepted_updates, "cache"),
                particles=particles, specification=self.specification, device=self.device, pin_memory=self.pin_memory)
            batch = CacheRefreshBatch(self._transfer(indices, dtype=torch.int64),
                tuple(self.inventory.law_ids[i] for i in indices), tuple(self.inventory.donor_ids[i] for i in indices),
                tuple(self.inventory.target_ids[i] for i in indices), sampled.states, sampled.noise)
            self._assert_context()
            yield batch

    def count_inputs(self):
        """Fresh detached initial cache plus complete donor catalogs; no update."""
        self._assert_context()
        self.cache.assert_unchanged()
        required = len(self.inventory.training_donors) * len(self.cache.manifest.catalog) * 48 + len(self.cache.laws) * 64
        if required > self.cache.limits.maximum_prepared_bytes:
            raise ValueError("Complete catalog/count-cache inputs exceed prepared-byte cap")
        binding = dict(model_recipe=self.configuration_id, training_subset="-".join(self.inventory.training_donors),
                       catalog_layout=self.inventory.manifest_sha256, source_sampler=self.inventory.sampler_sha256)
        n = len(self.cache.laws)
        cache = FitnessCache(torch.zeros(n, dtype=torch.float32, device=self.device), binding=binding,
            seen=torch.zeros(n, dtype=torch.bool, device=self.device), age=torch.zeros(n, dtype=torch.int64, device=self.device))
        blocks = []
        for donor in self.inventory.training_donors:
            source, endpoint = (self.cache.populations[(donor, c)] for c in ("Stim8hr", "Stim48hr"))
            pairs = [(i, guide) for i, (d, guide) in enumerate(self.cache.laws) if d == donor]
            log_source = source_log_frequencies(self._transfer(source.captured_counts)).float()
            blocks.append(CatalogBlock(donor, self.cache.manifest.catalog_sha256, log_source,
                self._transfer(endpoint.captured_counts), self._transfer([g for _, g in pairs], dtype=torch.int64),
                self._transfer([i for i, _ in pairs], dtype=torch.int64)))
        return tuple(blocks), cache

    def accepted(self, *, accepted_updates):
        self._assert_context()
        if (type(accepted_updates) is not int or accepted_updates != self.position + 1
                or self._prepared_position != self.position or accepted_updates > self.schedule.stages[-1]):
            raise ValueError("Commit exactly one prepared healthy update; no duplicate/skip")
        self.position = accepted_updates
        self._prepared_position = None

    def state_dict(self):
        self._assert_context()
        self.cache.assert_unchanged()
        return dict(schema="credo.frontier.fitting-provider-state.v1", binding=self.binding,
                    position=self.position, policy=SAMPLING_POLICY, mutable_rng_state=None)

    def load_state_dict(self, state):
        self._assert_context()
        self.cache.assert_unchanged()
        expected = {"schema", "binding", "position", "policy", "mutable_rng_state"}
        if (self.position != 0 or self._prepared_position is not None or not isinstance(state, dict)
                or set(state) != expected or state["schema"] != "credo.frontier.fitting-provider-state.v1"
                or state["binding"] != self.binding or state["policy"] != SAMPLING_POLICY
                or state["mutable_rng_state"] is not None or type(state["position"]) is not int
                or not 0 <= state["position"] <= self.schedule.stages[-1]):
            raise ValueError("Fresh matching provider and exact committed sampler-state binding required")
        self.position = state["position"]
