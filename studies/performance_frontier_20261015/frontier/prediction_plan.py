"""Metadata-only prediction/selection inventories for the registered stage graph.

Sharing a fitted trajectory never shares its two different query populations
or enlarges a selector's outcome access. One bank is required per directed score
edge, not per calibration. Outer ensembles reuse the three existing banks and
their explicitly corrected smaller prefixes; they add no generated particles.

This module does not inspect models, inputs, publications, outcomes, resources,
or scheduler state. Completion identifiers are caller-verified metadata, NOT
proof of a seal. Actual artifact admission, sufficient storage and publication
verification remain separate obligations. No launch/deletion authority exists.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .configuration import ALPHA_GRID, DONORS, SEEDS, SPREAD_GRID
from .planner import ScoreTask, StagePlan, TrainingTask, main_budget
from .selector_contract import OuterRecipe, Recipe


SCHEMA = "credo.frontier.prediction-operation-plan.v1"
METHOD = "one_bank_per_directed_edge_fixed_grid_and_equal_three_seed_mixture_v1"
_SOURCE = Path(__file__)
_SOURCE_SHA256 = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Exact lowercase SHA-256 identity required")


def _source_marker():
    if hashlib.sha256(_SOURCE.read_bytes()).hexdigest() != _SOURCE_SHA256:
        raise ValueError("Prediction-plan implementation changed after import")
    return dict(module="frontier/prediction_plan.py", sha256=_SOURCE_SHA256, method=METHOD)


def _stage(value):
    if type(value) is not StagePlan:
        raise TypeError("Exact validated StagePlan required")
    value.__post_init__()
    return value


def _outer_inventory(stage, supplied):
    if stage.name != "outer":
        if supplied is not None:
            raise ValueError("Inner prediction cannot receive frozen outer recipes")
        return ()
    if not isinstance(supplied, Mapping) or set(supplied) != set(DONORS):
        raise ValueError("All four selectors' frozen outer recipe triplets required")
    result = []
    for selector in DONORS:
        rows = supplied[selector]
        if (type(rows) is not tuple or len(rows) != 3 or any(type(r) is not OuterRecipe for r in rows)
                or len({r.configuration_id for r in rows}) != 3
                or {r.configuration_id for r in rows} != set(stage.eligible_for(selector))):
            raise ValueError("Frozen outer triplet differs from the exact eligible configurations")
        for row in rows:
            Recipe(row.configuration_id, row.alpha, row.spread_scale)
            _sha(row.selection_contract_identity); _sha(row.selection_evidence_identity)
            if (row.assessment_donor != selector
                    or row.fitting_donors != tuple(d for d in DONORS if d != selector)
                    or row.seeds != SEEDS or type(row.seeds) is not tuple
                    or type(row.updates) is not int or row.updates != 9216
                    or isinstance(row.inner_quality, bool) or not isinstance(row.inner_quality, (int, float))
                    or not math.isfinite(row.inner_quality) or row.inner_quality < 0):
                raise ValueError("Exact outer donor/fitting fold/seeds/update/selection metadata required")
        if (len({r.selection_contract_identity for r in rows}) != 1
                or len({r.selection_evidence_identity for r in rows}) != 1):
            raise ValueError("One complete inner selection decision must bind the outer triplet")
        result.extend(sorted(rows, key=lambda r: r.configuration_id))
    return tuple(result)


def _component_counts(total):
    # The fixed round-robin plan, not sampling or a new particle-generation call.
    return tuple(total//3 + int(seed < total % 3) for seed in SEEDS)


@dataclass(frozen=True)
class ForecastOperation:
    edge: ScoreTask
    stage: str
    single_seed_prefixes: tuple
    calibration_recipes: tuple
    ensemble_component_totals: tuple = ()

    @property
    def identifier(self):
        return "forecast/" + self.edge.identifier.removeprefix("score/")

    @property
    def component_prefixes(self):
        return tuple((total, _component_counts(total)[self.edge.key.seed])
                     for total in self.ensemble_component_totals)

    @property
    def effective_prefixes(self):
        return tuple(sorted(set(self.single_seed_prefixes) | {p for _, p in self.component_prefixes}))

    def record(self):
        return dict(forecast_id=self.identifier, edge_id=self.edge.identifier, edge=asdict(self.edge),
            stage=self.stage, raw_recipe=asdict(Recipe(self.edge.key.configuration_id)),
            single_seed_prefixes=list(self.single_seed_prefixes),
            ensemble_component_totals=list(self.ensemble_component_totals),
            ensemble_component_prefixes={str(total): count for total, count in self.component_prefixes},
            effective_prefixes=list(self.effective_prefixes), maximum_particles=max(self.single_seed_prefixes),
            calibration_recipes=[asdict(r) for r in self.calibration_recipes])


@dataclass(frozen=True)
class ForecastGroup:
    """Complete intended fold/query seal inventory, not a successful-bank list."""
    stage: str
    fitting_donors: tuple
    query_donor: str
    permitted_selector: str
    forecast_ids: tuple
    edge_ids: tuple


@dataclass(frozen=True)
class EnsembleOperation:
    frozen_recipe: OuterRecipe
    parent_forecast_ids: tuple
    total_particles: tuple = (4096, 8192)

    @property
    def identifier(self):
        return f"ensemble/outer/{self.frozen_recipe.assessment_donor}/{self.frozen_recipe.configuration_id}"

    def record(self):
        return dict(ensemble_id=self.identifier, frozen_recipe=asdict(self.frozen_recipe),
            parent_forecast_ids=list(self.parent_forecast_ids), seeds=list(SEEDS),
            total_particles=list(self.total_particles),
            component_counts={str(t): list(_component_counts(t)) for t in self.total_particles},
            rule="equal_mixture_of_calibrated_finite_measures", newly_generated_particles=0)


@dataclass(frozen=True, init=False)
class PredictionPlan:
    stage: StagePlan
    forecasts: tuple
    groups: tuple
    ensembles: tuple
    outer_recipes: tuple
    _encoded: bytes

    def __init__(self, *args, **kwargs):
        raise TypeError("Use PredictionPlan.from_stage with the complete validated stage")

    @classmethod
    def from_stage(cls, stage, *, outer_recipes=None):
        stage = _stage(stage)
        frozen = _outer_inventory(stage, outer_recipes)
        lookup = {(r.assessment_donor, r.configuration_id): r for r in frozen}
        outer = stage.name == "outer"
        prefixes = (4096, 8192) if outer else ((4096,) if stage.name == "finalist" else (512,))
        forecasts = []
        for edge in sorted(stage.scores, key=lambda s: s.identifier):
            if outer:
                r = lookup[edge.selector, edge.key.configuration_id]
                recipes = (Recipe(r.configuration_id, r.alpha, r.spread_scale),)
            else:
                recipes = tuple(Recipe(edge.key.configuration_id, a, s) for a in ALPHA_GRID for s in SPREAD_GRID)
            forecasts.append(ForecastOperation(edge, stage.name, prefixes, recipes, (4096, 8192) if outer else ()))
        forecasts = tuple(forecasts)
        grouped = {}
        for operation in forecasts:
            edge = operation.edge
            grouped.setdefault((edge.key.training_donors, edge.validation_donor, edge.selector), []).append(operation)
        groups = tuple(ForecastGroup(stage.name, donors, query, selector,
                       tuple(o.identifier for o in operations), tuple(o.edge.identifier for o in operations))
                       for (donors, query, selector), operations in sorted(grouped.items()))
        ensembles = []
        for r in frozen:
            components = sorted((o for o in forecasts if o.edge.selector == r.assessment_donor
                                 and o.edge.key.configuration_id == r.configuration_id), key=lambda o: o.edge.key.seed)
            if tuple(o.edge.key.seed for o in components) != SEEDS:
                raise ValueError("Exact ordered seed triplet missing from outer bank inventory")
            ensembles.append(EnsembleOperation(r, tuple(o.identifier for o in components)))
        ensembles = tuple(ensembles)
        counts = dict(raw_bank_operations=len(forecasts), complete_fold_query_groups=len(groups),
            directed_score_edges=len(stage.scores),
            inner_recipe_edge_assessments=0 if outer else sum(len(o.calibration_recipes) for o in forecasts),
            outer_single_seed_prefix_assessments=sum(len(o.single_seed_prefixes) for o in forecasts) if outer else 0,
            ensemble_operations=len(ensembles), ensemble_total_assessments=sum(len(o.total_particles) for o in ensembles))
        document = dict(schema=SCHEMA, status="metadata_inventory_not_execution_admission",
            implementation=_source_marker(), stage_plan=asdict(stage), stage_plan_sha256=_hash(asdict(stage)),
            forecasts=[o.record() for o in forecasts], groups=[asdict(g) for g in groups],
            ensembles=[o.record() for o in ensembles], counts=counts,
            counts_are_registered_operation_inventory_not_completed_work=True,
            generated_particle_or_neural_forward_budget_not_established_here=True,
            actual_publication_verification_required=True, actual_selection_evidence_verification_required=True,
            actual_source_model_runtime_storage_bindings_required=True,
            scientific_launch_authority=False, deletion_authority=False, automatic_retries=0)
        obj = object.__new__(cls)
        for key, value in dict(stage=stage, forecasts=forecasts, groups=groups, ensembles=ensembles,
                               outer_recipes=frozen, _encoded=_encode(document)).items():
            object.__setattr__(obj, key, value)
        return obj

    @property
    def identity(self):
        _source_marker()
        return hashlib.sha256(self._encoded).hexdigest()

    def record(self):
        _source_marker()
        return json.loads(self._encoded)


def prediction_budget(plans):
    """Reconcile the full registered graph; no operation has been executed."""
    plans = tuple(plans)
    if any(type(plan) is not PredictionPlan for plan in plans):
        raise TypeError("Four complete PredictionPlan instances required")
    fitting = main_budget(tuple(plan.stage for plan in plans))
    counts = [plan.record()["counts"] for plan in plans]
    return dict(schema="credo.frontier.prediction-operation-budget.v1", implementation=_source_marker(),
        stage_plan_identities=[plan.identity for plan in plans], main_training=fitting,
        **{name: sum(row[name] for row in counts) for name in counts[0]},
        automatic_budget_expansion=False, scientific_launch_authority=False,
        operation_counts_are_not_rollout_forward_or_completed_work_counts=True)


def required_before_continuation(previous_stage, next_training_task):
    """All previous directed edges for shared weights, not only promoted views.

    The current forecast loader cannot load an earlier stage after its mutable
    owner journal advances. Each requesting selector's prior query forecast
    must therefore be independently sealed before those shared weights advance.
    This function does not require or reveal the other selector's scores.
    """
    previous_stage = _stage(previous_stage)
    if type(next_training_task) is not TrainingTask:
        raise TypeError("Exact proposed continuation TrainingTask required")
    next_training_task.__post_init__()
    expected = {"screen": (1024, 3072), "family": (3072, 9216)}.get(previous_stage.name)
    if (expected is None or len(next_training_task.key.training_donors) != 2
            or (next_training_task.start_updates, next_training_task.target_updates) != expected):
        raise ValueError("Only the exact next shared inner trajectory transition is supported")
    previous = [t for t in previous_stage.training if t.key == next_training_task.key]
    if (len(previous) != 1 or previous[0].target_updates != next_training_task.start_updates
            or not set(next_training_task.requesting_selectors) <= set(previous[0].requesting_selectors)):
        raise ValueError("Continuation lacks its exact prior shared-trajectory requests")
    required = tuple(sorted(s.identifier for s in previous_stage.scores
                            if s.key == next_training_task.key and s.updates == next_training_task.start_updates))
    if len(required) != len(previous[0].requesting_selectors):
        raise ValueError("Prior complete directed forecast inventory does not reconcile")
    return required


def require_continuation_forecasts(previous_stage, next_training_task, verified_complete_edge_ids):
    """Require caller-verified seal identifiers; never verify or authorize work.

    Supply unique ScoreTask.identifier values from this previous stage only.
    The caller must verify complete actual forecast seals, their model/prefix
    bindings and retention. A fabricated ID list cannot establish publication.
    A separately valid next-stage selection/admission is still required.
    """
    required = required_before_continuation(previous_stage, next_training_task)
    supplied = verified_complete_edge_ids
    if (type(supplied) is not tuple or any(type(x) is not str for x in supplied)
            or len(set(supplied)) != len(supplied)
            or not set(supplied) <= {s.identifier for s in previous_stage.scores}):
        raise ValueError("Unique exact caller-verified previous-stage edge identifiers required")
    missing = tuple(edge for edge in required if edge not in supplied)
    if missing:
        raise ValueError("Shared trajectory still requires previous directed forecasts: " + ", ".join(missing))
    return required
