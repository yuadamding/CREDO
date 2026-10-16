"""Strict metadata adapter between saved score evidence and nested selection.

No outcomes are read here. The supplied identity claims require verification by
the future input owner; syntactically valid hashes do not prove their contents.
All selection arithmetic is conditional on those source-bound claims. This
adapter closes anonymous-vector, incomplete-coverage and calibration-loss gaps;
it does not implement a predictor, a worker or a scheduler.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType

from .configuration import ALPHA_GRID, DONORS, SEEDS, SPREAD_GRID, get_configuration
from .planner import ScoreTask
from .selection import CHANNELS, CandidateQuality, choose, complete_vector, normalized_risks, quality


def _identity(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("Identity must be a lowercase SHA-256 content identity")
    return value


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Recipe:
    configuration_id: str
    alpha: float = 1.0
    spread_scale: float = 1.0

    def __post_init__(self):
        get_configuration(self.configuration_id)
        if isinstance(self.alpha, bool) or self.alpha not in ALPHA_GRID:
            raise ValueError("Abundance calibration is outside the registered grid")
        if isinstance(self.spread_scale, bool) or self.spread_scale not in SPREAD_GRID:
            raise ValueError("Spread calibration is outside the registered grid")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "spread_scale", float(self.spread_scale))


@dataclass(frozen=True)
class ChannelCoverage:
    identity: str
    scored_units: int

    def __post_init__(self):
        _identity(self.identity)
        if type(self.scored_units) is not int or self.scored_units < 1:
            raise ValueError("Every selection channel needs positive fixed support")


@dataclass(frozen=True)
class EvidenceContext:
    stage: str
    runtime_identity: str
    representation_identity: str
    fitting_input_identity: str
    prediction_source_identity: str
    truth_identity: str
    scoring_identity: str
    scoring_particles: int
    representation_fitting_donors: tuple

    def __post_init__(self):
        if self.stage not in ("screen", "family", "finalist"):
            raise ValueError("Selection context cannot use outer outcomes")
        for field in ("runtime_identity", "representation_identity", "fitting_input_identity",
                      "prediction_source_identity", "truth_identity", "scoring_identity"):
            _identity(getattr(self, field))
        expected_particles = 4096 if self.stage == "finalist" else 512
        if type(self.scoring_particles) is not int or self.scoring_particles != expected_particles:
            raise ValueError("Wrong registered scoring resolution for this allocation stage")
        donors = tuple(self.representation_fitting_donors)
        if len(donors) != 2 or donors != tuple(d for d in DONORS if d in donors):
            raise ValueError("Inner representation must bind a canonical two-donor fitting subset")
        object.__setattr__(self, "representation_fitting_donors", donors)


@dataclass(frozen=True)
class TrainingScales:
    identity: str
    fitting_donors: tuple
    fitting_input_identity: str
    representation_identity: str
    values: object

    def __post_init__(self):
        for value in (self.identity, self.fitting_input_identity, self.representation_identity):
            _identity(value)
        donors = tuple(self.fitting_donors)
        if len(donors) != 2 or donors != tuple(d for d in DONORS if d in donors):
            raise ValueError("Scales must bind exactly the canonical fitting pair")
        values = complete_vector(self.values)
        if any(value <= 0 for value in values.values()):
            raise ValueError("Fitting-only fallback scales must be strictly positive")
        object.__setattr__(self, "fitting_donors", donors)
        object.__setattr__(self, "values", MappingProxyType(values))


@dataclass(frozen=True)
class ScoreEvidence:
    task: ScoreTask
    recipe: Recipe
    context: EvidenceContext
    coverage: object
    risks: object

    def __post_init__(self):
        if not isinstance(self.task, ScoreTask) or self.task.phase != "inner":
            raise ValueError("Only exact inner score tasks can enter selection")
        if self.recipe.configuration_id != self.task.key.configuration_id:
            raise ValueError("Recipe and scored trajectory configuration differ")
        if self.context.representation_fitting_donors != self.task.key.training_donors:
            raise ValueError("Representation fitting donors differ from the trajectory subset")
        if set(self.coverage) != set(CHANNELS) or not all(
                isinstance(self.coverage[c], ChannelCoverage) for c in CHANNELS):
            raise ValueError("All nine channel support identities are required")
        object.__setattr__(self, "coverage", MappingProxyType(dict(self.coverage)))
        object.__setattr__(self, "risks", MappingProxyType(complete_vector(self.risks)))


@dataclass(frozen=True)
class CostEvidence:
    configuration_id: str
    runtime_identity: str
    profile_identity: str
    inference_cost: float
    active_parameters: int

    def __post_init__(self):
        get_configuration(self.configuration_id)
        _identity(self.runtime_identity)
        _identity(self.profile_identity)
        if (isinstance(self.inference_cost, bool) or not math.isfinite(self.inference_cost)
                or self.inference_cost < 0):
            raise ValueError("Inference cost must be finite and nonnegative")
        if type(self.active_parameters) is not int or self.active_parameters < 1:
            raise ValueError("Active parameter count must be a positive integer")


@dataclass(frozen=True)
class EvaluatedRecipe:
    recipe: Recipe
    candidate_quality: CandidateQuality
    contract_identity: str
    evidence: tuple
    cost: CostEvidence


@dataclass(frozen=True)
class OuterRecipe:
    """No outer fit yet: exact inner-selected recipe for future deployment."""
    assessment_donor: str
    fitting_donors: tuple
    configuration_id: str
    alpha: float
    spread_scale: float
    selection_contract_identity: str
    selection_evidence_identity: str
    inner_quality: float
    seeds: tuple = SEEDS
    updates: int = 9216


class SelectorContract:
    """One selector/stage with fixed uncalibrated incumbent and training scales."""

    def __init__(self, plan, selector, baseline_evidence, training_scales):
        if plan.name not in ("screen", "family", "finalist") or selector not in DONORS:
            raise ValueError("Expected one registered inner selector and stage")
        # StagePlan validates complete edge coverage at construction; recheck
        # here to prevent a mutated/incompletely deserialized metadata object.
        plan.__post_init__()
        self.plan = plan
        self.selector = selector
        self.baseline_configuration = "F0-V0" if plan.name == "screen" else next(
            c for c in plan.eligible_for(selector) if get_configuration(c).family == "F0")
        self._baseline = self._index_evidence(Recipe(self.baseline_configuration), baseline_evidence)
        scales = tuple(training_scales)
        if len(scales) != 3 or len({item.fitting_donors for item in scales}) != 3:
            raise ValueError("Exactly three distinct fitting-pair scale bindings are required")
        self._scales = MappingProxyType({item.fitting_donors: item for item in scales})
        for evidence in self._baseline.values():
            scale = self._scales.get(evidence.task.key.training_donors)
            if scale is None:
                raise ValueError("Missing fitting-only scale binding for an inner edge")
            self._match_scales(evidence, scale)
        self.identity = _digest({
            "stage": plan.name, "selector": selector,
            "eligible_configurations": plan.eligible_for(selector),
            "baseline_configuration": self.baseline_configuration,
            "baseline": [{"task": item.task.identifier, "context": asdict(item.context),
                          "coverage": {c: asdict(item.coverage[c]) for c in CHANNELS},
                          "risks": dict(item.risks)} for item in self._baseline.values()],
            "scales": [{"identity": item.identity, "fitting_donors": item.fitting_donors,
                        "fitting_input_identity": item.fitting_input_identity,
                        "representation_identity": item.representation_identity,
                        "values": dict(item.values)} for _, item in sorted(self._scales.items())],
        })

    def _index_evidence(self, recipe, evidence):
        if recipe.configuration_id not in self.plan.eligible_for(self.selector):
            raise ValueError("Recipe is not eligible for this outer selector")
        supplied = tuple(evidence)
        expected = {task.identifier: task for task in self.plan.scores_for(self.selector)
                    if task.key.configuration_id == recipe.configuration_id}
        if len(supplied) != 3 or len({item.task.identifier for item in supplied}) != 3:
            raise ValueError("Exactly three distinct authorized inner score keys are required")
        if {item.task.identifier for item in supplied} != set(expected):
            raise ValueError("Score keys are not the three allowed edges for this selector/recipe")
        indexed = {}
        for item in supplied:
            if item.task != expected[item.task.identifier] or item.recipe != recipe:
                raise ValueError("Score evidence does not match the exact recipe and trajectory")
            if item.context.stage != self.plan.name:
                raise ValueError("Score evidence came from a different allocation stage")
            indexed[item.task.key.training_donors] = item
        return MappingProxyType(dict(sorted(indexed.items())))

    @staticmethod
    def _match_scales(evidence, scale):
        context = evidence.context
        if (scale.fitting_donors != evidence.task.key.training_donors
                or scale.fitting_input_identity != context.fitting_input_identity
                or scale.representation_identity != context.representation_identity):
            raise ValueError("Scale identity/provenance is not the same fitting-only fold")

    def evaluate(self, recipe, evidence, cost):
        indexed = self._index_evidence(recipe, evidence)
        vectors = []
        for pair, item in indexed.items():
            baseline = self._baseline[pair]
            if item.context != baseline.context:
                raise ValueError("Runtime/source/representation/truth/scoring context differs from incumbent")
            if item.coverage != baseline.coverage:
                raise ValueError("Candidate and incumbent require identical support in every channel")
            if cost.configuration_id != recipe.configuration_id or cost.runtime_identity != item.context.runtime_identity:
                raise ValueError("Prebound inference-cost evidence has the wrong model/runtime identity")
            scale = self._scales[pair]
            self._match_scales(item, scale)
            vectors.append(normalized_risks(item.risks, baseline.risks, scale.values))
        row = CandidateQuality(recipe.configuration_id, quality(vectors), cost.inference_cost,
                               cost.active_parameters, recipe.alpha, recipe.spread_scale)
        return EvaluatedRecipe(recipe, row, self.identity, tuple(indexed.values()), cost)

    def _complete_recipes(self, rows):
        """Require the frozen full calibration grid, never a favorable subset."""
        rows = tuple(rows)
        expected = {Recipe(c, alpha, spread) for c in self.plan.eligible_for(self.selector)
                    for alpha in ALPHA_GRID for spread in SPREAD_GRID}
        if len(rows) != len(expected) or {row.recipe for row in rows} != expected:
            raise ValueError("Every eligible configuration and all 24 calibrations must be available exactly once")
        result = {}
        for row in rows:
            if row.contract_identity != self.identity or self.evaluate(row.recipe, row.evidence, row.cost) != row:
                raise ValueError("Evaluation belongs to another selector/contract or was altered")
            result[row.recipe] = row
        return result

    @staticmethod
    def _choose(rows):
        rows = tuple(rows)
        best = choose([row.candidate_quality for row in rows])
        return next(row for row in rows if row.candidate_quality == best)

    def family_winners(self, rows):
        if self.plan.name != "screen":
            raise ValueError("Family winners can be selected only after the initial screen")
        checked = self._complete_recipes(rows)
        return tuple(self._choose(row for row in checked.values()
                                 if get_configuration(row.recipe.configuration_id).family == family)
                     for family in ("F0", "F1", "F2", "F3", "F4", "F5"))

    def finalists(self, rows):
        if self.plan.name != "family":
            raise ValueError("Finalist selection requires the family-comparison stage")
        checked = self._complete_recipes(rows)
        return tuple(self._choose(row for row in checked.values()
                                 if get_configuration(row.recipe.configuration_id).family in group)
                     for group in (("F0",), ("F1", "F2", "F3"), ("F4", "F5")))

    def freeze_outer(self, rows):
        if self.plan.name != "finalist":
            raise ValueError("Outer recipes require the completed final inner stage")
        checked = self._complete_recipes(rows)
        decision_evidence = _digest([
            {"recipe": asdict(row.recipe), "quality": asdict(row.candidate_quality),
             "cost": asdict(row.cost),
             "scores": [{"task": item.task.identifier, "context": asdict(item.context),
                         "coverage": {c: asdict(item.coverage[c]) for c in CHANNELS},
                         "risks": dict(item.risks)} for item in row.evidence]}
            for row in sorted(checked.values(), key=lambda item: (
                item.recipe.configuration_id, item.recipe.alpha, item.recipe.spread_scale))])
        frozen = []
        for configuration in self.plan.eligible_for(self.selector):
            row = self._choose(item for item in checked.values() if item.recipe.configuration_id == configuration)
            frozen.append(OuterRecipe(
                self.selector, tuple(d for d in DONORS if d != self.selector), configuration,
                row.recipe.alpha, row.recipe.spread_scale, self.identity, decision_evidence,
                row.candidate_quality.quality))
        return tuple(frozen)
