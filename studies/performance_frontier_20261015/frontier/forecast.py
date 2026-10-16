"""Source-only forecasts; no loader, truth access or scoring.

Every positive-RNA source guide is forecast irrespective of endpoint or trained
target support. The complete captured catalog is normalized after each particle
prefix, including unit-multiplier source-unavailable categories. Pooled NTC RNA
is a separate cell-equal source law, never an added catalog entry.

Caller-supplied artifact/input hashes do not prove custody, permitted access or
checkpoint loading. This core verifies actual supplied arrays and frozen model
state, and requires explicit output/working estimates. Those conservative byte
estimates are not measured CUDA/RSS qualification or permission to execute.
An optional explicit streaming sink receives bounded laws but is not completed
here. Its caller can publish only after this function returns validated evidence.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import math
import random
from types import MappingProxyType

import numpy as np
import torch

from .configuration import DONORS, SEEDS, STAGE_UPDATES
from .fitting_provider import CatalogGuide, PopulationArrays, PopulationReference, _array_hash, _hash, _sha
from .measure_ops import (calibrate_abundance, calibrate_dispersion, catalog_log_frequencies,
                          source_log_frequencies, round_robin_component_counts)
from .models import FrontierModel
from .native_inputs import FoldInputs
from .sampling import LawIdentity, LawRequest, PooledSourceIdentity, SamplingContext, SamplerSpecification, SourceLaw, sample_batch
from .selector_contract import Recipe
from .training_owner import ObservationBinding


SCIENCE_PREFIXES = (512, 4096, 8192)
ENSEMBLE_TOTALS = (4096, 8192)


def ensemble_prefix_plan(model_seed, totals, single_seed_prefixes, *, synthetic_fixture=False):
    """Registered unequal seed budgets derived from an existing maximum rollout.

    This is metadata arithmetic, not generation authorization. Scientific totals
    are exactly 4096/8192. Tiny explicitly synthetic totals permit CPU proofs but
    each seed still receives >=4 particles, so all F2 components remain present.
    The maximum ordinary prefix must already cover every requested total: this
    option cannot grow the maximum rollout or add stochastic draws.
    """
    ordinary, totals = tuple(single_seed_prefixes), tuple(totals)
    if (type(model_seed) is not int or model_seed not in SEEDS or type(synthetic_fixture) is not bool
            or not ordinary or any(type(p) is not int or p < 1 for p in ordinary)
            or ordinary != tuple(sorted(set(ordinary)))
            or any(type(p) is not int for p in totals) or totals != tuple(sorted(set(totals)))):
        raise ValueError("Exact seed and ascending unique ordinary/ensemble prefix plan required")
    if synthetic_fixture:
        if max(ordinary) > 128 or any(not 12 <= total <= 128 for total in totals):
            raise ValueError("Synthetic ensemble totals must be 12..128 particles")
    elif not set(ordinary) <= set(SCIENCE_PREFIXES) or not set(totals) <= set(ENSEMBLE_TOTALS):
        raise ValueError("Only registered scientific single-seed and ensemble totals are allowed")
    if any(total > max(ordinary) for total in totals):
        raise ValueError("Ensemble prefixes must reuse an already requested maximum rollout")
    return {str(total): round_robin_component_counts(total, 3)[model_seed] for total in totals}


@dataclass(frozen=True)
class ForecastSource:
    fold: FoldInputs
    catalog: tuple[CatalogGuide, ...]
    observation_binding: ObservationBinding
    feature_axis_sha256: str
    reference: PopulationReference
    captured_counts: np.ndarray
    positive_counts: np.ndarray
    offsets: np.ndarray
    states: np.ndarray
    synthetic_fixture: bool = False
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.fold, FoldInputs) or not isinstance(self.reference, PopulationReference)
                or not isinstance(self.observation_binding, ObservationBinding)
                or self.reference.donor not in DONORS or self.reference.donor in self.fold.training_donors
                or self.reference.condition != "Stim8hr" or type(self.synthetic_fixture) is not bool
                or type(self.catalog) is not tuple or not self.catalog
                or not all(isinstance(g, CatalogGuide) for g in self.catalog)
                or len({g.guide_id for g in self.catalog}) != len(self.catalog)):
            raise ValueError("Exact held-out early source, fold, representation and complete catalog required")
        _sha(self.feature_axis_sha256)
        g = len(self.catalog)
        if (not self.synthetic_fixture and g != 26504) or (self.synthetic_fixture and g > 64):
            raise ValueError("Complete scientific catalog or bounded fabricated catalog required")
        for name, dtype, shape in (("captured_counts", np.int64, (g,)), ("positive_counts", np.int64, (g,)),
                                   ("offsets", np.int64, (g + 1,))):
            value = getattr(self, name)
            if (not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape
                    or not value.flags.c_contiguous or value.flags.writeable or np.any(value < 0)):
                raise ValueError("Immutable exact source counts/offsets required")
        total = sum(int(v) for v in self.captured_counts)
        if (not 0 < total <= np.iinfo(np.int64).max or np.any(self.positive_counts > self.captured_counts)
                or not np.array_equal(self.offsets, np.r_[0, np.cumsum(self.positive_counts)])):
            raise ValueError("Complete capture and positive-RNA source support disagree")
        if (not isinstance(self.states, np.ndarray) or self.states.dtype != np.float32
                or self.states.shape != (int(self.offsets[-1]), 48)
                or not self.states.flags.c_contiguous or self.states.flags.writeable
                or not np.isfinite(self.states).all()):
            raise ValueError("Immutable complete guide-ordered finite FP32 source states required")
        controls = {g.target_id for g in self.catalog if g.is_control}
        if not controls or controls & {g.target_id for g in self.catalog if not g.is_control}:
            raise ValueError("A fixed disjoint NTC catalog is required, including absent NTCs")
        # No molecular array or endpoint is needed. This helper's latent hash
        # depends only on offsets/states and the exact observation binding.
        view = PopulationArrays(self.reference, self.captured_counts, self.positive_counts,
                                self.offsets, self.states, None)
        if view.latent_identity(catalog_sha256=self.catalog_sha256,
                               observation_binding=self.observation_binding) != self.reference.latent_sha256:
            raise ValueError("Supplied source states differ from fold representation/content identity")
        object.__setattr__(self, "_identity", self._digest())

    @property
    def catalog_sha256(self):
        return _hash([asdict(g) for g in self.catalog])

    def _digest(self):
        return _hash(dict(fold=asdict(self.fold), catalog_sha256=self.catalog_sha256,
            observation=asdict(self.observation_binding), feature_axis_sha256=self.feature_axis_sha256,
            reference=asdict(self.reference), synthetic_fixture=self.synthetic_fixture,
            arrays={name: _array_hash(getattr(self, name)) for name in
                    ("captured_counts", "positive_counts", "offsets", "states")}))

    @property
    def identity(self):
        return self._identity

    def assert_unchanged(self):
        if any(getattr(self, n).flags.writeable for n in ("captured_counts", "positive_counts", "offsets", "states")) or self._digest() != self.identity:
            raise ValueError("Forecast source arrays changed during computation")


def model_state_digest(model):
    """Actual frozen state/routing digest; not a checkpoint-loader proof."""
    if (not isinstance(model, FrontierModel) or model.training or model.phase != "joint"
            or not model.reaction_enabled or any(p.requires_grad for p in model.parameters())):
        raise ValueError("Pre-frozen eval/joint FrontierModel required; core never changes its phase")
    if model._target_lookup != {key: i for i, key in enumerate(model.target_ids)}:
        raise ValueError("Model target routing differs from its frozen inventory")
    digest = hashlib.sha256(_hash(model.architecture()).encode())
    for name, tensor in sorted(model.state_dict().items()):
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Finite FP32 model weights and buffers required")
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(_array_hash(value).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ForecastModelBinding:
    fold: FoldInputs
    observation_binding: ObservationBinding
    configuration_id: str
    model_artifact_sha256: str
    model_state_sha256: str
    target_support_sha256: str
    numerical_source_sha256: str
    model_seed: int
    accepted_updates: int

    def __post_init__(self):
        if not isinstance(self.fold, FoldInputs) or not isinstance(self.observation_binding, ObservationBinding):
            raise ValueError("Exact model fitting fold and representation required")
        Recipe(self.configuration_id)
        if (type(self.model_seed) is not int or self.model_seed not in SEEDS
                or type(self.accepted_updates) is not int or self.accepted_updates < 0):
            raise ValueError("Registered model seed and explicit accepted-update identity required")
        for name in ("model_artifact_sha256", "model_state_sha256", "target_support_sha256", "numerical_source_sha256"):
            _sha(getattr(self, name))


@dataclass(frozen=True)
class ForecastLimits:
    maximum_host_bytes: int
    maximum_device_working_bytes: int
    maximum_output_bytes: int
    maximum_pool_cells: int
    microbatch_laws: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()) or self.microbatch_laws > 1024:
            raise ValueError("Explicit positive forecast byte/pool/microbatch limits required")


@dataclass(frozen=True)
class ConditionalForecast:
    uncalibrated_states: torch.Tensor  # CPU FP32
    states: torch.Tensor               # CPU FP32, fixed weighted-mean spread calibration
    log_weights: torch.Tensor          # CPU FP64, normalized at THIS prefix
    log_mass_increment: float          # uncalibrated reaction/proposal integral


@dataclass(frozen=True)
class PrefixForecast:
    particles: int
    guide_laws: object                 # catalog index -> ConditionalForecast
    unavailable_guides: tuple[int, ...]
    log_mass_increments: torch.Tensor  # complete catalog; complement is exactly zero
    log_source_frequencies: torch.Tensor
    uncalibrated_log_frequencies: torch.Tensor
    log_frequencies: torch.Tensor
    pooled_control: ConditionalForecast | None
    pooled_control_unavailable_reason: str | None


@dataclass(frozen=True)
class Forecast:
    prefixes: object
    receipt: dict


@dataclass(frozen=True)
class ForecastEmission:
    """Successful streamed computation, not a retained bank or completed seal."""
    receipt: dict


def _rng_digest(device):
    digest = hashlib.sha256(repr(random.getstate()).encode())
    npstate = np.random.get_state()
    digest.update(repr((npstate[0], *npstate[2:])).encode()); digest.update(npstate[1].tobytes())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if device.type == "cuda":
        digest.update(torch.cuda.get_rng_state(device).cpu().numpy().tobytes())
    return digest.hexdigest()


def _resources(model, source, prefixes, limits, *, storage_estimate=None):
    supported = int(np.count_nonzero(source.positive_counts))
    pool_cells = sum(int(source.positive_counts[i]) for i, g in enumerate(source.catalog) if g.is_control)
    if pool_cells > limits.maximum_pool_cells:
        raise ValueError("Full pooled NTC source exceeds support cap; no truncation")
    laws = supported + int(pool_cells > 0)
    p, batch, width = max(prefixes), min(limits.microbatch_laws, max(1, supported)), model.config.width
    # Deliberately count both raw/calibrated banks even at spread=1, plus every
    # prefix separately. No nested-bank storage or alias saving is assumed.
    output = laws * sum(prefixes) * (2 * 48 * 4 + 8)
    output += len(prefixes) * (len(source.catalog) * 4 * 8 + laws * 2048)
    model_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    # Eight-step retained DDR tensors, component direct tensors, FP64 reductions,
    # modulation contractions and sampler copies. Still requires runtime proof.
    working = (2 * model_bytes + batch * p * (96 * 48 + 32 * width) * 4
               + batch * width * 48 * 8 * 4)
    input_bytes = sum(getattr(source, n).nbytes for n in ("captured_counts", "positive_counts", "offsets", "states"))
    # Source categorical sampling validates complete supports and constructs
    # FP64 normalized weights/CDFs. These grow with actual source cells, not
    # with the number of forecast particles. Requests retain each batch's
    # source weights while one law's validation/CDF temporaries are active.
    maximum_batch_support = sum(sorted((int(n) for n in source.positive_counts if n), reverse=True)[:batch])
    maximum_law_support = max((int(n) for n in source.positive_counts), default=0)
    support_working = 16 * max(maximum_batch_support, pool_cells) + 128 * max(maximum_law_support, pool_cells)
    catalog_metadata = sum(1024 + len(g.guide_id.encode()) + len(g.target_id.encode()) for g in source.catalog)
    # Draw receipts persist until the whole run is verified. Streaming the
    # particle arrays does not remove their per-law strings/dicts from memory.
    sampler_metadata = laws * (8192 + 8 * (len(source.reference.source_id.encode())
                                         + len(source.reference.replicate_id.encode()))) + 4 * catalog_metadata
    if storage_estimate is None:
        retained_output = output
        storage_working = 0
    else:
        # The writer stores one raw maximum prefix and the exact per-prefix
        # weights using bounded positional writes. No full output mmap/bank is
        # retained by this core. Resource admission includes its own buffers.
        output = storage_estimate["output_upper_bound_bytes"]
        retained_output = (p * (2 * 48 * 4 + 8)
                           + len(prefixes) * len(source.catalog) * 4 * 8)
        storage_working = storage_estimate["maximum_working_bytes"]
    host = (input_bytes + catalog_metadata + sampler_metadata + retained_output
            + pool_cells * (48 * 4 + 8) + support_working + working + storage_working)
    if output > limits.maximum_output_bytes or working > limits.maximum_device_working_bytes or host > limits.maximum_host_bytes:
        raise ValueError("Forecast output/working estimate exceeds explicit byte cap before rollout")
    return dict(complete_catalog=len(source.catalog), supported_guides=supported, pooled_NTC_cells=pool_cells,
                output_bytes=output, device_working_bytes=working, host_bytes=host,
                retained_output_bytes=retained_output, storage_working_bytes=storage_working,
                retained_sampler_metadata_bytes=sampler_metadata,
                source_support_working_bytes=support_working,
                maximum_batch_source_cells=maximum_batch_support,
                caller_retained_RNA_means_decoder_and_other_banks_included=False,
                streamed_output=storage_estimate is not None,
                estimates_not_measured_peak=True, separately_admitted_runtime_required=True)


def _conditional(sliced, row, spread):
    logs = sliced.log_weights[row].double()
    log_mass = torch.logsumexp(logs, 0)
    normalized = logs - log_mass
    raw = sliced.states[row].detach().cpu().clone()
    weights = normalized.detach().cpu().clone()
    calibrated = calibrate_dispersion(raw, weights, spread)
    if not bool(torch.isfinite(calibrated).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Nonfinite forecast law; no probability floor or replacement")
    return ConditionalForecast(raw, calibrated, weights, float(log_mass))


@torch.no_grad()
def forecast_population(model, source, binding, context, recipe, *, prefixes, limits,
                        ensemble_component_totals=(), sink=None):
    """One query donor, one frozen recipe/seed, all source-supported guides.

    Generate once at max(prefixes), then derive each prefix's conditional law
    and mass anew. Optional registered ensemble component prefixes reuse that
    same rollout; they do not relax the ordinary scientific prefix gate.
    Without a sink, returned tensors remain caller-owned and mutable. With an
    admitted ForecastWriter, emit each law immediately and return only a
    ForecastEmission receipt. Caller invokes writer.finish AFTER that return;
    an exception leaves provisional output incomplete. Neither mode releases
    endpoint truth or provides scheduler/resource authorization.
    """
    if (not isinstance(source, ForecastSource) or not isinstance(binding, ForecastModelBinding)
            or not isinstance(context, SamplingContext) or not isinstance(recipe, Recipe)
            or not isinstance(limits, ForecastLimits) or not isinstance(model, FrontierModel)):
        raise TypeError("Typed source/model/sampler/recipe/resource bindings required")
    if (source.fold != binding.fold or source.observation_binding != binding.observation_binding
            or context.training_donors != source.fold.training_donors or context.purpose != "forecast"
            or context.configuration_id != recipe.configuration_id or recipe.configuration_id != binding.configuration_id
            or model.config.configuration_id != binding.configuration_id
            or context.model_seed != binding.model_seed or context.accepted_update != binding.accepted_updates):
        raise ValueError("Forecast fitting fold, representation, configuration or sampler role differs")
    controls = {g.target_id for g in source.catalog if g.is_control}
    if controls != set(model.control_ids) or controls & set(model.target_ids):
        raise ValueError("Frozen model control/target routing differs from query catalog")
    if model_state_digest(model) != binding.model_state_sha256:
        raise ValueError("Actual model state differs from admitted forecast binding")
    prefixes = tuple(prefixes)
    if (not prefixes or any(type(p) is not int or p < model.config.mixture_components for p in prefixes)
            or prefixes != tuple(sorted(set(prefixes)))):
        raise ValueError("Ascending distinct positive prefixes with all mixture components required")
    device = model.reference_embedding.device
    if source.synthetic_fixture:
        if device.type != "cpu" or max(prefixes) > 128 or len(source.catalog) > 64 or limits.microbatch_laws > 64:
            raise ValueError("Fabricated forecasting is bounded CPU-only work")
    elif (device.type != "cuda" or device.index is None or not set(prefixes) <= set(SCIENCE_PREFIXES)
          or binding.accepted_updates not in STAGE_UPDATES):
        raise ValueError("Scientific forecast requires indexed CUDA and registered particle prefixes")
    single_seed_prefixes = prefixes
    ensemble_component_totals = tuple(ensemble_component_totals)
    component_plan = ensemble_prefix_plan(binding.model_seed, ensemble_component_totals, prefixes,
                                          synthetic_fixture=source.synthetic_fixture)
    if component_plan and (len(binding.fold.training_donors) != 3 or binding.accepted_updates != 9216):
        raise ValueError("Three-seed ensembles require an outer triple-donor 9216-update recipe")
    prefixes = tuple(sorted(set(prefixes) | set(component_plan.values())))
    source.assert_unchanged()
    if sink is not None:
        from .forecast_storage import ForecastStorageSpec, ForecastWriter
        if not isinstance(sink, ForecastWriter):
            raise TypeError("Explicit active ForecastWriter required for streaming")
        sink.assert_active()
        expected_sink = ForecastStorageSpec.from_source(source, binding, context, recipe,
            prefixes=single_seed_prefixes, ensemble_component_totals=ensemble_component_totals)
        if (sink.spec.identity != expected_sink.identity or sink.spec.effective_prefixes != prefixes
                or sink.limits.maximum_batch_laws < limits.microbatch_laws):
            raise ValueError("Streaming output spec or batch bound differs from the forecast")
    resources = _resources(model, source, prefixes, limits,
                           storage_estimate=None if sink is None else sink.estimate)
    rng_before = _rng_digest(device)
    supported = tuple(i for i, n in enumerate(source.positive_counts) if n > 0)
    missing = tuple(i for i, n in enumerate(source.positive_counts) if n == 0)
    outputs = {p: {} for p in prefixes}
    masses = {p: torch.zeros(len(source.catalog), dtype=torch.float64, device="cpu") for p in prefixes}
    spec = SamplerSpecification(maximum_laws=limits.microbatch_laws)
    sampler_receipts = []

    def law(index):
        a, b = map(int, source.offsets[index:index + 2])
        ref = source.reference
        identity = LawIdentity(ref.donor, "Stim8hr", source.catalog[index].guide_id, ref.source_id, ref.replicate_id)
        return SourceLaw(identity, source.states[a:b], np.full(b - a, 1. / (b - a), np.float64))

    for start in range(0, len(supported), limits.microbatch_laws):
        indices = supported[start:start + limits.microbatch_laws]
        sampled = sample_batch(tuple(LawRequest(law(i)) for i in indices), context,
                               particles=max(prefixes), specification=spec, device=device)
        result = model.rollout(sampled.states, sampled.noise,
                               target_ids=tuple(source.catalog[i].target_id for i in indices), compute_anchor=False)
        sampler_receipts.append(sampled.receipt)
        for prefix in prefixes:
            # Recompute F2 proposal counts once for the complete microbatch,
            # not once per row and never by slicing already normalized weights.
            sliced = model.prefix_rollout(result, prefix)
            for row, index in enumerate(indices):
                output = _conditional(sliced, row, recipe.spread_scale)
                masses[prefix][index] = output.log_mass_increment
                if sink is None:
                    outputs[prefix][index] = output
                else:
                    sink.emit(prefix, index, output)
                    del output

    pool = {p: None for p in prefixes}
    pool_identity = None
    if resources["pooled_NTC_cells"]:
        control_indices = tuple(i for i, g in enumerate(source.catalog) if g.is_control)
        states = np.concatenate([source.states[source.offsets[i]:source.offsets[i + 1]]
                                 for i in control_indices if source.positive_counts[i]])
        weights = np.full(len(states), 1. / len(states), np.float64)
        catalog_sha = _hash([asdict(source.catalog[i]) for i in control_indices])
        ref = source.reference
        definition = _hash(dict(policy="all_NTC_positive_RNA_source_cells_equal", source=ref.latent_sha256,
                                raw_summary=ref.raw_summary_sha256, catalog=catalog_sha))
        pool_identity = PooledSourceIdentity(ref.donor, "Stim8hr", ref.source_id, ref.replicate_id,
                                            f"{ref.donor}::pooled-NTC-RNA", catalog_sha, definition)
        sampled = sample_batch((LawRequest(SourceLaw(pool_identity, states, weights)),), context,
                               particles=max(prefixes), specification=spec, device=device)
        result = model.rollout(sampled.states, sampled.noise,
                               indices=torch.full((1,), -1, device=device, dtype=torch.int64), compute_anchor=False)
        sampler_receipts.append(sampled.receipt)
        for prefix in prefixes:
            output = _conditional(model.prefix_rollout(result, prefix), 0, recipe.spread_scale)
            if sink is None:
                pool[prefix] = output
            else:
                sink.emit(prefix, "pooled_control", output)
                del output

    source_logs = source_log_frequencies(torch.tensor(source.captured_counts, dtype=torch.int64, device="cpu"))
    index_tensor = torch.tensor(supported, dtype=torch.int64, device="cpu")
    completed = {}
    for prefix in prefixes:
        raw_logs = catalog_log_frequencies(source_logs, index_tensor, masses[prefix][index_tensor])
        logs = calibrate_abundance(source_logs, raw_logs, recipe.alpha)
        if bool((logs.exp() <= 0).any()):
            raise ValueError("Underflowed catalog frequency; no undeclared floor")
        if sink is None:
            completed[prefix] = PrefixForecast(prefix, MappingProxyType(outputs[prefix]), missing, masses[prefix],
                source_logs.clone(), raw_logs, logs, pool[prefix],
                None if pool[prefix] is not None else "missing_positive_RNA_source_NTC_pool")
        else:
            sink.write_catalog(prefix, log_mass_increments=masses[prefix], log_source_frequencies=source_logs,
                uncalibrated_log_frequencies=raw_logs, log_frequencies=logs)
    source.assert_unchanged()
    if model_state_digest(model) != binding.model_state_sha256 or _rng_digest(device) != rng_before:
        raise ValueError("Frozen model or global RNG changed during forecast")
    receipt = dict(schema="credo.frontier.in-memory-forecast.v1", source_identity=source.identity,
        query_donor=source.reference.donor,
        model_binding=asdict(binding), context=asdict(context), recipe=asdict(recipe),
        prefixes=list(prefixes), resources=resources, limits=asdict(limits),
        single_seed_prefixes=list(single_seed_prefixes),
        ensemble_component_totals=list(ensemble_component_totals),
        ensemble_component_prefixes=component_plan,
        maximum_generated_particles=max(single_seed_prefixes), synthetic_fixture=source.synthetic_fixture,
        supported_guide_indices=list(supported), unavailable_source_guide_indices=list(missing),
        target_ids=list(model.target_ids), control_ids=list(model.control_ids),
        pooled_control_identity=asdict(pool_identity) if pool_identity is not None else None,
        sampler_receipts=sampler_receipts, uncalibrated_mass_reduction="FP64_logsumexp_of_prefix_proposal_corrected_FP32_log_weights",
        output_tensor_storage="CPU_FP32_states_FP64_normalized_logweights_and_catalogs",
        complement_mass_multiplier=1, pooled_control_is_catalog_entry=False,
        source_only_support=True, endpoint_inputs_used=False, optimizer_calls=0,
        input_artifact_custody_verified_here=False, checkpoint_loading_verified_here=False,
        complete_prediction_checks_passed=True,
        law_storage="in_memory" if sink is None else "streamed_to_explicit_sink",
        sink_spec_sha256=None if sink is None else sink.spec.identity,
        publication_claimed=False, scientific_launch_authority=False)
    if sink is not None:
        sink.assert_active()
        return ForecastEmission(receipt)
    return Forecast(MappingProxyType(completed), receipt)
