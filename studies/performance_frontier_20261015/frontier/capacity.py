"""Matched width128/256 reports from saved inner-score evidence only.

CAPACITY_COMPARISON_PROTOCOL.json is a diagnostic addendum, not a new search
arm. V1 and V2 share learning rate/weight decay and all registered architecture
settings except width. Actual active parameter counts and measured cost evidence
remain mandatory: width alone does not establish the capacity/cost difference.

This module reads no files, models, tensors or outcomes. Evidence hashes are
caller-verified bindings, not proofs manufactured by these dataclasses. No
selector rule changes, fitting requests or global capacity winner are produced.
Raw latent scores from different fold-native representations are not averaged.
Each summary is restricted to one selector's three permitted inner edges.
"""

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from .configuration import ALPHA_GRID, FAMILIES, SPREAD_GRID, get_configuration
from .planner import StagePlan, training_pairs
from .selection import CHANNELS, nondominated
from .selector_contract import CostEvidence, ScoreEvidence, _digest, _identity


def objective_scientific_identity(protocol, *, configuration_id, rules, scales):
    """Common scientific objective, not configuration-specific protocol_id.

    objective_registry.build_objective_protocol hashes configuration_id into
    protocol_id. First reconstruct/verify that exact producer contract, then
    omit ONLY its label. Retain every coefficient, scale, regularizer, family,
    donor, bandwidth, rule and scale-provenance binding in the common identity.
    No model, tensor payload or optimizer is constructed.
    """
    from .objective_registry import build_objective_protocol

    expected = build_objective_protocol(configuration_id, rules=rules, scales=scales,
        fitting_donors=scales.fitting_donors, observation_artifact_sha256=scales.observation_artifact_sha256,
        fitting_input_sha256=scales.fitting_input_sha256)
    if protocol != expected:
        raise ValueError("Supplied protocol differs from its exact registry producer")
    scientific = asdict(protocol)
    del scientific["protocol_id"]
    return _digest(dict(schema="credo.frontier.capacity-scientific-objective.v1",
        scientific_objective=scientific, rules_sha256=rules.identity, scale_binding=scales.as_binding()))


def actual_draw_receipt_identity(receipt, *, sampler_source_identity, source_law_payload_identity,
                                 sampled_output_identity):
    """Configuration-neutral identity of an actual sample_batch receipt.

    SamplingContext.draw_binding excludes configuration_id; the full receipt
    and provider inventory.sampler_sha256 do not. Verify the source/noise stream
    keys against that exact producer equation before deriving a common identity.
    Bind externally verified source states AND weights, plus the actual sampled
    states/noise/source-index payload identity, not their labels or RNG keys alone.
    The complete run's actual_draw_identity must cover its full ordered draw
    inventory, including training/cache/forecast/pool draws as applicable. One
    matching receipt is not proof of all those draws; no inventory is inferred.
    Producers may hash each in-memory payload and accumulate the inventory
    digest incrementally, or reuse an already verified receipt. No retention
    or replay of every training batch is required here. Both callers must bind
    the same admitted work scope; helper success does not establish complete
    trajectory draw reconciliation or authorize more numerical work.
    This helper does not read arrays or generate random numbers.
    """
    from .sampling import SAMPLER_VERSION, LawIdentity, PooledSourceIdentity, SamplingContext, SamplerSpecification

    _identity(sampler_source_identity)
    _identity(source_law_payload_identity)
    _identity(sampled_output_identity)
    expected = dict(schema=SAMPLER_VERSION, configuration_in_draw_key=False,
        microbatch_or_worker_in_draw_key=False, bit_generator="PCG64DXSM",
        normal_transform="NumPy_float64_standard_normal_then_FP32",
        noise_draw_order="particle,step,coordinate_then_transpose_step,law,particle,coordinate",
        source_sampling="iid_normalized_source_weight_categorical_with_replacement",
        source_weight_sum_tolerance=1e-7,
        post_sampling_integration="equal_weight_particles_no_second_source_weight_factor")
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Unrecognized actual sample_batch numerical receipt")
    context = SamplingContext(**receipt["context"])
    specification = SamplerSpecification(**receipt["specification"])
    particles, offset = receipt["particles_returned_per_law"], receipt["particle_offset"]
    if (type(particles) is not int or type(offset) is not int or particles < 1 or offset < 0
            or offset+particles > specification.maximum_particles):
        raise ValueError("Invalid actual particle interval")
    streams = receipt["streams"]
    if not isinstance(streams, list) or not 1 <= len(streams) <= specification.maximum_laws:
        raise ValueError("Explicit nonempty bounded actual draw streams required")
    seen = set()
    for stream in streams:
        request = stream["request"]
        if not isinstance(request, dict) or set(request) != {"law", "occurrence"}:
            raise ValueError("Exact law and occurrence binding required")
        law_type = PooledSourceIdentity if "pool_id" in request["law"] else LawIdentity
        law = law_type(**request["law"])
        occurrence = request["occurrence"]
        if type(occurrence) is not int or not 0 <= occurrence < 2**64:
            raise ValueError("Explicit nonnegative draw occurrence required")
        if context.purpose in ("training", "cache", "scales") and law.donor not in context.training_donors:
            raise ValueError("Actual fitting draw contains a held-out donor")
        request_identity = _digest(request)
        if request_identity in seen:
            raise ValueError("Duplicate law/occurrence in actual draw receipt")
        seen.add(request_identity)
        for kind, field in (("source_categorical", "source_stream_sha256"), ("standard_normal", "noise_stream_sha256")):
            expected_key = _digest(dict(schema=SAMPLER_VERSION, context=context.draw_binding(), request=request, stream=kind))
            if stream.get(field) != expected_key:
                raise ValueError("Actual draw stream differs from the registered draw key")
        support, positive = stream["source_support"], stream["positive_source_weights"]
        if type(support) is not int or type(positive) is not int or not 1 <= positive <= support:
            raise ValueError("Explicit actual source support required")
    numerical_counts = dict(law_count=len(streams), generated_prefix_per_law=offset+particles,
        source_uniform_draws=len(streams)*(offset+particles),
        standard_normal_draws=len(streams)*(offset+particles)*specification.steps*specification.latent_dim)
    if any(type(receipt.get(k)) is not int or receipt[k] != v for k, v in numerical_counts.items()):
        raise ValueError("Actual draw counts do not reconcile")
    # Preserve every receipt field, including runtime/transfer evidence. Only
    # the configuration label is removed; original receipts remain untouched.
    common_receipt = {**receipt, "context": context.draw_binding()}
    return _digest(dict(schema="credo.frontier.capacity-actual-draw.v1",
        sampler_source_identity=sampler_source_identity, source_law_payload_identity=source_law_payload_identity,
        sampled_output_identity=sampled_output_identity, actual_receipt=common_receipt))


@dataclass(frozen=True)
class CapacityCalibration:
    """A fixed matched calibration, chosen without the reported query outcomes.

    The unmodified comparison is alpha=spread_scale=1. No default automatically
    picks a favorable pair from saved calibrated results.
    """
    alpha: float
    spread_scale: float
    binding_identity: str
    used_query_outcomes: bool

    def __post_init__(self):
        _identity(self.binding_identity)
        if (isinstance(self.alpha, bool) or self.alpha not in ALPHA_GRID
                or isinstance(self.spread_scale, bool) or self.spread_scale not in SPREAD_GRID):
            raise ValueError("Explicit registered fixed calibration required")
        if self.used_query_outcomes is not False:
            raise ValueError("Capacity calibration must be fixed without the query outcomes")


@dataclass(frozen=True)
class CapacityRunEvidence:
    """Exact saved score plus measured cost/parameter and training bindings.

    inference_workload_identity binds hardware, units and the common profiling
    workload; CostEvidence.profile_identity binds that model's measurements.
    Optional measured seconds are descriptive, never inferred from updates.
    The publisher must verify each identity against its actual retained record.
    objective_scientific_identity is the common signature from the helper above,
    NOT ObjectiveProtocol.protocol_id or its configuration-specific artifact.
    actual_draw_identity binds the complete verified draw inventory, NOT the
    configuration-specific provider inventory.sampler_sha256. Its receipt-level
    components may be derived with actual_draw_receipt_identity above.
    """
    score: ScoreEvidence
    cost: CostEvidence
    checkpoint_identity: str
    model_source_identity: str
    objective_scientific_identity: str
    normalization_identity: str
    actual_draw_identity: str
    parameter_inventory_identity: str
    inference_workload_identity: str
    inference_cost_unit: str
    measured_fit_seconds: float | None = None
    measured_scoring_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.score, ScoreEvidence) or not isinstance(self.cost, CostEvidence):
            raise ValueError("Actual saved ScoreEvidence and CostEvidence required")
        for field in ("checkpoint_identity", "model_source_identity", "objective_scientific_identity",
                      "normalization_identity", "actual_draw_identity", "parameter_inventory_identity",
                      "inference_workload_identity"):
            _identity(getattr(self, field))
        if (self.cost.configuration_id != self.score.recipe.configuration_id
                or self.cost.runtime_identity != self.score.context.runtime_identity):
            raise ValueError("Measured cost must bind this configuration and runtime")
        if self.cost.inference_cost <= 0:
            raise ValueError("A positive measured inference cost is required, not a default zero")
        if not isinstance(self.inference_cost_unit, str) or not self.inference_cost_unit:
            raise ValueError("Explicit measured inference-cost units required")
        for field in ("measured_fit_seconds", "measured_scoring_seconds"):
            value = getattr(self, field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("Optional measured times must be positive finite seconds")


def _validate_plan(plan, selector):
    if not isinstance(plan, StagePlan) or plan.name not in ("screen", "family", "finalist"):
        raise ValueError("An exact inner-stage plan is required; no outer/global capacity pooling")
    plan.__post_init__()
    return {task.identifier: task for task in plan.scores_for(selector)}


def _validate_run(run, allowed, stage, calibration):
    if not isinstance(run, CapacityRunEvidence) or not isinstance(calibration, CapacityCalibration):
        raise ValueError("Bound capacity run and fixed calibration required")
    # Dataclasses are immutable, but explicit validation rejects incomplete
    # reconstruction instead of accepting an arbitrary anonymous score vector.
    run.__post_init__()
    score = run.score
    if allowed.get(score.task.identifier) != score.task or score.context.stage != stage:
        raise ValueError("Score is not an authorized same-stage edge for this selector")
    if (score.recipe.alpha, score.recipe.spread_scale) != (calibration.alpha, calibration.spread_scale):
        raise ValueError("Score does not use the fixed common capacity calibration")


def _payload(run):
    return dict(task=asdict(run.score.task), recipe=asdict(run.score.recipe),
        context=asdict(run.score.context), coverage={c: asdict(run.score.coverage[c]) for c in CHANNELS},
        risks=dict(run.score.risks), cost=asdict(run.cost), checkpoint_identity=run.checkpoint_identity,
        model_source_identity=run.model_source_identity, objective_scientific_identity=run.objective_scientific_identity,
        normalization_identity=run.normalization_identity, actual_draw_identity=run.actual_draw_identity,
        parameter_inventory_identity=run.parameter_inventory_identity,
        inference_workload_identity=run.inference_workload_identity, inference_cost_unit=run.inference_cost_unit,
        measured_fit_seconds=run.measured_fit_seconds, measured_scoring_seconds=run.measured_scoring_seconds)


def _relation(small, large):
    names = nondominated({"width128": small, "width256": large})
    if names == ["width128"]:
        return "smaller_dominates", names
    if names == ["width256"]:
        return "larger_dominates", names
    return ("equal" if small == large else "trade_off"), names


def compare_capacity_edge(smaller, larger, *, plan, selector, calibration):
    """Nine large-minus-small differences on one fully matched allowed edge.

    This does not average predictions or seeds, apply a new advancement margin,
    or infer equal compute from equal accepted updates.
    """
    allowed = _validate_plan(plan, selector)
    for run in (smaller, larger):
        _validate_run(run, allowed, plan.name, calibration)
    small, large = smaller.score, larger.score
    cs, cl = get_configuration(small.recipe.configuration_id), get_configuration(large.recipe.configuration_id)
    if cs.family != cl.family or cs.variant != "V1" or cl.variant != "V2":
        raise ValueError("Capacity comparison requires same-family V1 versus V2")
    registered_small, registered_large = asdict(cs), asdict(cl)
    for key in ("configuration_id", "variant", "width"):
        del registered_small[key], registered_large[key]
    if registered_small != registered_large or (cs.width, cl.width) != (128, 256):
        raise ValueError("Registered variants differ in more than the declared capacity change")
    if (small.task.key.training_donors != large.task.key.training_donors
            or small.task.key.seed != large.task.key.seed or small.task.updates != large.task.updates
            or small.task.validation_donor != large.task.validation_donor):
        raise ValueError("Training pair, query donor, accepted updates and seed must match")
    if small.context != large.context or small.coverage != large.coverage:
        raise ValueError("Exact representation, runtime, input, scoring prefix and nine-channel support must match")
    for key in ("model_source_identity", "objective_scientific_identity", "normalization_identity", "actual_draw_identity",
                "inference_workload_identity", "inference_cost_unit"):
        if getattr(smaller, key) != getattr(larger, key):
            raise ValueError("Unmatched capacity evidence: " + key)
    if larger.cost.active_parameters <= smaller.cost.active_parameters:
        raise ValueError("Width256 is not larger by the supplied actual active parameter counts")
    if smaller.checkpoint_identity == larger.checkpoint_identity:
        raise ValueError("Distinct capacity trajectories cannot identify the same checkpoint")
    differences = {c: large.risks[c]-small.risks[c] for c in CHANNELS}
    directions = {c: "larger_lower_risk" if differences[c] < 0 else
                  "larger_higher_risk" if differences[c] > 0 else "equal" for c in CHANNELS}
    relationship, frontier = _relation(dict(small.risks), dict(large.risks))
    payload = dict(schema="credo.frontier.capacity-edge.v1", selector=selector,
        family=cs.family, training_donors=list(small.task.key.training_donors),
        validation_donor=small.task.validation_donor, stage=plan.name, accepted_updates=small.task.updates,
        seed=small.task.key.seed, scoring_particles=small.context.scoring_particles,
        calibration=asdict(calibration), smaller=_payload(smaller), larger=_payload(larger),
        paired_differences=differences, direction=directions, nondominated_widths=frontier,
        relationship=relationship, active_parameter_difference=larger.cost.active_parameters-smaller.cost.active_parameters,
        active_parameter_ratio=larger.cost.active_parameters/smaller.cost.active_parameters,
        measured_inference_cost_difference=larger.cost.inference_cost-smaller.cost.inference_cost,
        measured_inference_cost_ratio=larger.cost.inference_cost/smaller.cost.inference_cost,
        interpretation="finite_budget_screening_only" if small.task.updates == 1024 else "matched_longer_budget_comparison",
        equal_compute_claimed=False, longer_budget_inference=False, independent_validation=False,
        changes_selector_eligibility=False, additional_fits=0)
    return {**payload, "content_identity": _digest(payload)}


def report_capacity_selector(plan, selector, family, evidence: Sequence[CapacityRunEvidence], *, calibration):
    """One family, one selector, all three predeclared edges; no raw-risk pool.

    Missing/uncontinued variants remain unavailable. A later selector normally
    retains just one configuration per family, so the longer matched comparison
    is unavailable; another selector's continuation cannot supply the missing arm.
    """
    allowed = _validate_plan(plan, selector)
    if family not in FAMILIES or not isinstance(calibration, CapacityCalibration):
        raise ValueError("Registered family and explicit fixed calibration required")
    wanted = {f"{family}-V1", f"{family}-V2"}
    indexed = {}
    for run in evidence:
        _validate_run(run, allowed, plan.name, calibration)
        if run.score.recipe.configuration_id not in wanted or run.score.task.identifier in indexed:
            raise ValueError("Duplicate or foreign-family/variant capacity evidence")
        indexed[run.score.task.identifier] = run
    edges = []
    for pair in training_pairs(selector):
        tasks = {task.key.configuration_id: task for task in allowed.values()
                 if task.key.training_donors == pair and task.key.configuration_id in wanted}
        unavailable = {}
        for configuration in sorted(wanted):
            if configuration not in tasks:
                unavailable[configuration] = "variant_not_authorized_for_this_selector_stage"
            elif tasks[configuration].identifier not in indexed:
                unavailable[configuration] = "missing_saved_score_evidence"
        if unavailable:
            edges.append(dict(training_donors=list(pair), available=False, unavailable=unavailable,
                              paired_differences=None, relationship=None))
        else:
            row = compare_capacity_edge(indexed[tasks[f"{family}-V1"].identifier],
                indexed[tasks[f"{family}-V2"].identifier], plan=plan, selector=selector, calibration=calibration)
            edges.append(dict(available=True, **row))
    complete = [row for row in edges if row["available"]]
    direction_counts = {channel: {direction: sum(row["direction"][channel] == direction for row in complete)
        for direction in ("larger_lower_risk", "equal", "larger_higher_risk")} for channel in CHANNELS}
    payload = dict(schema="credo.frontier.capacity-selector.v1", selector=selector, family=family, stage=plan.name,
        calibration=asdict(calibration), expected_edges=3, available_edges=len(complete), complete=len(complete) == 3,
        edges=edges, direction_counts=direction_counts, pooled_raw_latent_risks=None,
        global_capacity_winner=None, selection_rule_changed=False, additional_fits=0,
        additional_updates=0, independent_validation=False,
        interpretation="finite_budget_screening_only" if plan.name == "screen" else "only_explicitly_matched_longer_budget_evidence",
        limitation="Missing longer-budget pairs do not show that larger models cannot help; no outcome sharing across selectors")
    return {**payload, "content_identity": _digest(payload)}
