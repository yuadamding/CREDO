"""Fixed-total equal three-seed finite-measure mixtures, without generation.

Consume the registered seed-specific prefixes already derived from each maximum
rollout by forecast_population. Never truncate normalized weights, draw particles,
call models, average coordinates, or weight seeds by their particle counts.

Raw and calibrated ensembles need DIFFERENT conditional weights: abundance
calibration changes each seed's guide frequency. Calibrate component populations
first, then mix their finite measures. The pooled NTC molecular companion instead
mixes the three normalized control laws equally; its guide-catalog abundance and
reaction integral do not determine the pool's seed weights.

Ensemble arrays concatenate seed blocks. They are NOT simple nested prefixes
across total budgets. No ensemble reaction-mass increment is invented: each seed
has its own complete-catalog normalizer. Custody, streaming, publication and
selected-checkpoint loading remain separate caller-owned gates.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType

import torch

from .forecast import (ConditionalForecast, Forecast, ForecastModelBinding, PrefixForecast,
                       ensemble_prefix_plan)
from .configuration import DONORS
from .fitting_provider import _array_hash, _hash
from .measure_ops import (calibrate_abundance, catalog_log_frequencies, ensemble_catalog,
                          ensemble_guide, round_robin_component_counts)
from .native_inputs import FoldInputs
from .sampling import SamplingContext
from .selector_contract import Recipe
from .training_owner import ObservationBinding


@dataclass(frozen=True)
class EnsembleLimits:
    maximum_host_bytes: int
    maximum_output_bytes: int
    maximum_working_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Explicit positive ensemble host/output/working byte caps required")


@dataclass(frozen=True)
class EnsembleConditionalForecast:
    uncalibrated_states: torch.Tensor
    states: torch.Tensor
    uncalibrated_log_weights: torch.Tensor
    log_weights: torch.Tensor


@dataclass(frozen=True)
class EnsemblePrefixForecast:
    particles: int
    component_particles: tuple[int, int, int]
    guide_laws: object
    unavailable_guides: tuple[int, ...]
    log_source_frequencies: torch.Tensor
    uncalibrated_log_frequencies: torch.Tensor
    log_frequencies: torch.Tensor
    pooled_control: EnsembleConditionalForecast | None
    pooled_control_unavailable_reason: str | None


@dataclass(frozen=True)
class EnsembleForecast:
    prefixes: object
    receipt: dict


def _tensor(value, dtype, shape, label):
    if (not isinstance(value, torch.Tensor) or value.dtype != dtype or value.device.type != "cpu"
            or tuple(value.shape) != tuple(shape) or not value.is_contiguous()
            or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise ValueError("Exact finite constant contiguous CPU tensor required: " + label)
    return value


def _logs(value, size, label):
    _tensor(value, torch.float64, (size,), label)
    if abs(torch.logsumexp(value, 0).item()) > 1e-10:
        raise ValueError("Prefix-specific normalized log probabilities required: " + label)


def _law(value, particles):
    if not isinstance(value, ConditionalForecast):
        raise ValueError("Existing prefix-specific ConditionalForecast required")
    _tensor(value.uncalibrated_states, torch.float32, (particles, 48), "raw state law")
    _tensor(value.states, torch.float32, (particles, 48), "calibrated state law")
    _logs(value.log_weights, particles, "state weights")
    if isinstance(value.log_mass_increment, bool) or not math.isfinite(value.log_mass_increment):
        raise ValueError("Finite prefix-specific uncalibrated mass increment required")


def _binding(receipt):
    raw = dict(receipt["model_binding"])
    fold = dict(raw["fold"])
    fold["training_donors"] = tuple(fold["training_donors"])
    raw["fold"] = FoldInputs(**fold)
    raw["observation_binding"] = ObservationBinding(**raw["observation_binding"])
    return ForecastModelBinding(**raw)


def _prefix(value, count, catalog, supported, missing, recipe):
    if (not isinstance(value, PrefixForecast) or value.particles != count
            or tuple(value.guide_laws) != supported or value.unavailable_guides != missing):
        raise ValueError("Prefix size and entire actual-source guide support must match")
    for name in ("log_source_frequencies", "uncalibrated_log_frequencies", "log_frequencies"):
        _logs(getattr(value, name), catalog, name)
        if bool((getattr(value, name).exp() <= 0).any()):
            raise ValueError("Positive complete-catalog probabilities required; no floor")
    _tensor(value.log_mass_increments, torch.float64, (catalog,), "catalog mass increments")
    if any(float(value.log_mass_increments[i]) != 0. for i in missing):
        raise ValueError("Source-unavailable catalog complement must retain unit multiplier")
    for index, law in value.guide_laws.items():
        _law(law, count)
        if float(value.log_mass_increments[index]) != law.log_mass_increment:
            raise ValueError("Guide and complete-catalog prefix mass increments differ")
    indices = torch.tensor(supported, device="cpu", dtype=torch.int64)
    expected = catalog_log_frequencies(value.log_source_frequencies, indices,
                                       value.log_mass_increments[indices])
    if not torch.allclose(expected, value.uncalibrated_log_frequencies, atol=1e-12, rtol=0):
        raise ValueError("Catalog was not normalized from this component prefix's masses")
    calibrated = calibrate_abundance(value.log_source_frequencies, expected, recipe.alpha)
    if not torch.allclose(calibrated, value.log_frequencies, atol=1e-12, rtol=0):
        raise ValueError("Component abundance calibration differs from fixed recipe")
    if value.pooled_control is not None:
        _law(value.pooled_control, count)
        if value.pooled_control_unavailable_reason is not None:
            raise ValueError("Available pooled NTC cannot have an unavailable reason")
    elif value.pooled_control_unavailable_reason != "missing_positive_RNA_source_NTC_pool":
        raise ValueError("Missing pooled source requires the explicit unchanged reason")


def _tensors(prefix):
    for name in ("log_mass_increments", "log_source_frequencies",
                 "uncalibrated_log_frequencies", "log_frequencies"):
        yield getattr(prefix, name)
    for law in (*prefix.guide_laws.values(), *((prefix.pooled_control,) if prefix.pooled_control is not None else ())):
        yield law.uncalibrated_states
        yield law.states
        yield law.log_weights


def _digest(component, needed):
    # Incremental content binding: do not retain a catalog-sized list of every
    # tensor digest in addition to the input/output particle banks.
    digest = hashlib.sha256(b"credo.frontier.ensemble-component-content.v1\n")
    digest.update(_hash(component.receipt).encode())
    for p in needed:
        value = component.prefixes[p]
        digest.update(_hash(dict(particles=value.particles, guide_indices=list(value.guide_laws),
            unavailable_guides=list(value.unavailable_guides),
            pooled_control_unavailable_reason=value.pooled_control_unavailable_reason)).encode())
        for tensor in _tensors(value):
            digest.update(_array_hash(tensor.numpy()).encode())
        for law in value.guide_laws.values():
            digest.update(_hash(dict(log_mass=law.log_mass_increment)).encode())
        digest.update(_hash(dict(pool_log_mass=value.pooled_control.log_mass_increment
            if value.pooled_control is not None else None)).encode())
    return digest.hexdigest()


def _resources(components, totals, catalog, laws, limits):
    # Charge every supplied bank, even unused ordinary prefixes and aliases.
    inputs, maximum_receipt_bytes = 0, 0
    for component in components:
        receipt_bytes = len(json.dumps(component.receipt, sort_keys=True, allow_nan=False).encode())
        inputs += receipt_bytes
        maximum_receipt_bytes = max(maximum_receipt_bytes, receipt_bytes)
        for value in component.prefixes.values():
            if not isinstance(value, PrefixForecast):
                raise ValueError("Only single-seed prefix forecasts may enter the ensemble")
            inputs += sum(t.numel() * t.element_size() for t in _tensors(value))
            inputs += (len(value.guide_laws) + 1) * 2048
    output = laws * sum(totals) * (2 * 48 * 4 + 2 * 8)
    output += len(totals) * (catalog * 3 * 8 + laws * 2048 + 8192)
    working = (4 * max(totals) * (2 * 48 * 4 + 2 * 8) + catalog * (16 * 8 + 256)
               + 2 * maximum_receipt_bytes + 65536)
    host = inputs + output + working
    if (output > limits.maximum_output_bytes or working > limits.maximum_working_bytes
            or host > limits.maximum_host_bytes):
        raise ValueError("Ensemble retained inputs/output/working estimate exceeds explicit cap")
    return dict(input_bytes=inputs, output_bytes=output, working_bytes=working, host_bytes=host,
                estimates_not_measured_peak=True, input_alias_savings_assumed=False)


def _mix(laws, raw_frequencies, calibrated_frequencies):
    raw_states, raw_weights, _ = ensemble_guide(
        tuple(law.uncalibrated_states for law in laws), tuple(law.log_weights for law in laws), raw_frequencies)
    states, weights, _ = ensemble_guide(
        tuple(law.states for law in laws), tuple(law.log_weights for law in laws), calibrated_frequencies)
    return EnsembleConditionalForecast(raw_states, states, raw_weights, weights)


@torch.no_grad()
def ensemble_forecast(components, *, totals, limits, synthetic_fixture=False):
    """Mix ordered seeds 0,1,2 from ONE fixed outer recipe and query population.

    Each input must already contain its registered component prefix. There is no
    fallback generation, weight slicing, cross-recipe averaging or completion of
    unavailable state laws. Input tensor content hashes are recorded and checked
    again at exit; they do not replace a verified external publication identity.
    """
    components, totals = tuple(components), tuple(totals)
    if (len(components) != 3 or not all(isinstance(c, Forecast) for c in components)
            or not isinstance(limits, EnsembleLimits) or type(synthetic_fixture) is not bool
            or not totals or any(type(t) is not int for t in totals)
            or totals != tuple(sorted(set(totals)))):
        raise ValueError("Three ordered single-seed forecasts, explicit totals and bounds required")
    bindings, recipes, contexts, plans = [], [], [], []
    for seed, component in enumerate(components):
        receipt = component.receipt
        if (receipt.get("schema") != "credo.frontier.in-memory-forecast.v1"
                or receipt.get("synthetic_fixture") is not synthetic_fixture
                or receipt.get("endpoint_inputs_used") is not False
                or receipt.get("source_only_support") is not True):
            raise ValueError("Source-only matching scientific/synthetic forecast receipt required")
        b, recipe = _binding(receipt), Recipe(**receipt["recipe"])
        context = SamplingContext(**receipt["context"])
        if (b.model_seed != seed or context.model_seed != seed or len(b.fold.training_donors) != 3
                or b.accepted_updates != 9216 or context.accepted_update != 9216
                or context.purpose != "forecast" or context.training_donors != b.fold.training_donors
                or context.configuration_id != b.configuration_id or recipe.configuration_id != b.configuration_id):
            raise ValueError("Exactly seeds0,1,2 of the same completed outer recipe are required")
        if receipt.get("query_donor") != next(d for d in DONORS if d not in b.fold.training_donors):
            raise ValueError("The exact held-out outer query donor is required")
        plan = ensemble_prefix_plan(seed, totals, receipt["single_seed_prefixes"], synthetic_fixture=synthetic_fixture)
        declared = ensemble_prefix_plan(seed, receipt["ensemble_component_totals"], receipt["single_seed_prefixes"],
                                       synthetic_fixture=synthetic_fixture)
        if (receipt["ensemble_component_prefixes"] != declared
                or not set(totals) <= set(receipt["ensemble_component_totals"])
                or receipt["maximum_generated_particles"] != max(receipt["single_seed_prefixes"])
                or receipt["prefixes"] != sorted(set(receipt["single_seed_prefixes"]) | set(declared.values()))
                or any(p not in component.prefixes for p in plan.values())):
            raise ValueError("Required component prefixes were not derived from the admitted maximum rollout")
        bindings.append(b); recipes.append(recipe); contexts.append(context); plans.append(plan)
    first, first_receipt = bindings[0], components[0].receipt
    for index in (1, 2):
        b, receipt = bindings[index], components[index].receipt
        common_binding = (b.fold, b.observation_binding, b.configuration_id, b.target_support_sha256,
                          b.numerical_source_sha256, b.accepted_updates)
        expected_binding = (first.fold, first.observation_binding, first.configuration_id, first.target_support_sha256,
                            first.numerical_source_sha256, first.accepted_updates)
        a, c = asdict(contexts[0]), asdict(contexts[index])
        a.pop("model_seed"); c.pop("model_seed")
        if common_binding != expected_binding or recipes[index] != recipes[0] or a != c:
            raise ValueError("Ensemble fold/representation/recipe/support/sampling context differs")
        for name in ("source_identity", "query_donor", "supported_guide_indices", "unavailable_source_guide_indices",
                     "target_ids", "control_ids", "pooled_control_identity"):
            if receipt[name] != first_receipt[name]:
                raise ValueError("Ensemble query population or complete support differs: " + name)
    catalog = first_receipt["resources"]["complete_catalog"]
    supported = tuple(first_receipt["supported_guide_indices"])
    missing = tuple(first_receipt["unavailable_source_guide_indices"])
    if (type(catalog) is not int or catalog < 1 or (synthetic_fixture and catalog > 64)
            or (not synthetic_fixture and catalog != 26504)
            or any(type(i) is not int for i in (*supported, *missing))
            or supported != tuple(sorted(set(supported))) or missing != tuple(sorted(set(missing)))
            or set(supported) & set(missing) or set(supported) | set(missing) != set(range(catalog))
            or any(c.receipt["resources"]["complete_catalog"] != catalog for c in components)):
        raise ValueError("Complete common catalog/source support is required, with no favorable omissions")
    pool_present = first_receipt["pooled_control_identity"] is not None
    if pool_present:
        query = first_receipt["pooled_control_identity"]["donor"]
        if query in first.fold.training_donors:
            raise ValueError("Pooled query donor cannot be a fitting donor")
    for index, component in enumerate(components):
        for p in plans[index].values():
            value = component.prefixes[p]
            _prefix(value, p, catalog, supported, missing, recipes[index])
            if (value.pooled_control is not None) != pool_present:
                raise ValueError("Pooled NTC support differs from the common source identity")
    reference_logs = components[0].prefixes[plans[0][str(totals[0])]].log_source_frequencies
    for index, component in enumerate(components):
        if any(not torch.equal(component.prefixes[p].log_source_frequencies, reference_logs)
               for p in plans[index].values()):
            raise ValueError("Complete source frequencies differ across seeds or prefixes")
    resources = _resources(components, totals, catalog, len(supported) + int(pool_present), limits)
    hashes = tuple(_digest(component, sorted(set(plan.values()))) for component, plan in zip(components, plans))
    outputs = {}
    for total in totals:
        counts = round_robin_component_counts(total, 3)
        prefixes = tuple(component.prefixes[plan[str(total)]] for component, plan in zip(components, plans))
        raw_catalog = ensemble_catalog(tuple(p.uncalibrated_log_frequencies for p in prefixes))
        calibrated_catalog = ensemble_catalog(tuple(p.log_frequencies for p in prefixes))
        laws = {}
        for index in supported:
            laws[index] = _mix(tuple(p.guide_laws[index] for p in prefixes),
                torch.stack(tuple(p.uncalibrated_log_frequencies[index] for p in prefixes)),
                torch.stack(tuple(p.log_frequencies[index] for p in prefixes)))
        pool = None
        if pool_present:
            # Unit component masses: equal normalized pool mixture, never guide
            # count weights, learned pool reaction masses, or n_seed/total.
            equal = torch.zeros(3, device="cpu", dtype=torch.float64)
            pool = _mix(tuple(p.pooled_control for p in prefixes), equal, equal)
        outputs[total] = EnsemblePrefixForecast(total, counts, MappingProxyType(laws), missing,
            reference_logs.clone(), raw_catalog, calibrated_catalog, pool,
            None if pool is not None else "missing_positive_RNA_source_NTC_pool")
    after = tuple(_digest(component, sorted(set(plan.values()))) for component, plan in zip(components, plans))
    if after != hashes:
        raise ValueError("Input forecast content changed during ensemble construction")
    receipt = dict(schema="credo.frontier.in-memory-three-seed-ensemble.v1", seeds=[0, 1, 2],
        model_bindings=[asdict(b) for b in bindings], source_identity=first_receipt["source_identity"],
        query_donor=first_receipt["query_donor"],
        recipe=asdict(recipes[0]), fold=asdict(first.fold), observation_binding=asdict(first.observation_binding),
        target_support_sha256=first.target_support_sha256, synthetic_fixture=synthetic_fixture,
        totals=list(totals), component_counts={str(t): list(round_robin_component_counts(t, 3)) for t in totals},
        component_content_sha256=list(hashes), content_binding_scope="receipts_and_used_prefix_tensors",
        supported_guide_indices=list(supported), unavailable_source_guide_indices=list(missing),
        pooled_control_identity=first_receipt["pooled_control_identity"], resources=resources, limits=asdict(limits),
        rule="equal_mixture_of_calibrated_finite_measures", conditional_seed_weights="guide_frequency_proportional",
        pooled_control_seed_weights="exactly_one_third_each", raw_and_calibrated_conditional_weights_separate=True,
        particle_order="concatenate_seed0_seed1_seed2_blocks_not_nested_across_total_budgets",
        newly_generated_particles=0, model_calls=0, optimizer_calls=0, endpoint_inputs_used=False,
        input_artifact_custody_verified_here=False, publication_claimed=False, scientific_launch_authority=False)
    return EnsembleForecast(MappingProxyType(outputs), receipt)
