"""Source-only, bounded calibration views over an unchanged sealed forecast bank.

Inner views expose the registered 24 recipes only on one eligible selector edge.
Outer views expose exactly its previously selected recipe, never an outer grid.
The derivation inventory must be sealed by the study owner before truth release;
this module neither publishes that seal nor admits outcomes or chooses recipes.
No particles, model calls, fitting, full-bank copies or altered parent receipts.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import stat

import torch

from . import measure_ops
from .configuration import ALPHA_GRID, DONORS, SEEDS, SPREAD_GRID
from .fitting_provider import _hash, _sha
from .forecast import ConditionalForecast
from .forecast_storage import ForecastReader, _runtime
from .planner import ScoreTask, StagePlan
from .selector_contract import OuterRecipe, Recipe


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _stamp(info):
    return tuple(getattr(info, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"))


def _source_records():
    records = []
    for path in (Path(__file__), Path(measure_ops.__file__)):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
            raise ValueError("Bound regular local calibration implementation required")
        with path.open("rb") as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) != before.st_size or _stamp(path.lstat()) != _stamp(before):
            raise ValueError("Calibration implementation changed while binding")
        records.append((str(path), hashlib.sha256(raw).hexdigest(), _stamp(before)))
    return tuple(records)


# Bind the source present at import, then recheck actual bytes at construction.
# Access checks retain their stat leases without hashing source per guide.
_IMPORTED_SOURCES = _source_records()


@dataclass(frozen=True, init=False)
class CalibrationView:
    """One live reader, immutable role/recipe inventory, one copied law at a time.

    Returned tensors are disposable caller-owned copies. Retaining them across
    calls is the caller's memory responsibility. No complete bank is retained.
    """
    _reader: object
    _spec: object
    _parent: bytes
    _runtime: bytes
    _sources: tuple
    _recipes: tuple
    _prefixes: tuple
    _manifest: bytes
    _identity: str

    def __init__(self, *args, **kwargs):
        raise TypeError("Use the explicitly role-bound inner or outer factory")

    @classmethod
    def for_inner(cls, reader, *, plan, selector, task, adapter_source_sha256):
        if not isinstance(plan, StagePlan) or not isinstance(task, ScoreTask):
            raise TypeError("Exact stage plan and score edge required")
        plan.__post_init__()
        task.__post_init__()
        if (plan.name not in ("screen", "family", "finalist") or selector not in DONORS
                or task.phase != "inner" or task.selector != selector
                or task not in plan.scores_for(selector)
                or task.key.configuration_id not in plan.eligible_for(selector)):
            raise ValueError("Only this selector's eligible inner edge may use the calibration grid")
        expected_updates = {"screen": 1024, "family": 3072, "finalist": 9216}[plan.name]
        if task.updates != expected_updates:
            raise ValueError("Inner stage and model update prefix differ")
        recipes = tuple(Recipe(task.key.configuration_id, a, s) for a in ALPHA_GRID for s in SPREAD_GRID)
        role = dict(phase="inner", stage=plan.name, selector=selector, task=asdict(task),
                    stage_plan_sha256=_hash(asdict(plan)), eligible_configurations=list(plan.eligible_for(selector)),
                    required_recipe_count=24, outer_outcome_selection_allowed=False)
        return cls._create(reader, task, recipes, role, adapter_source_sha256)

    @classmethod
    def for_outer(cls, reader, *, frozen, task, adapter_source_sha256):
        if not isinstance(frozen, OuterRecipe) or not isinstance(task, ScoreTask):
            raise TypeError("Exact frozen outer recipe and score edge required")
        task.__post_init__()
        recipe = Recipe(frozen.configuration_id, frozen.alpha, frozen.spread_scale)
        _sha(frozen.selection_contract_identity); _sha(frozen.selection_evidence_identity)
        if (frozen.assessment_donor not in DONORS
                or frozen.fitting_donors != tuple(d for d in DONORS if d != frozen.assessment_donor)
                or frozen.seeds != SEEDS or type(frozen.updates) is not int or frozen.updates != 9216
                or isinstance(frozen.inner_quality, bool) or not math.isfinite(frozen.inner_quality)
                or frozen.inner_quality < 0 or task.phase != "outer"
                or task.selector != frozen.assessment_donor or task.validation_donor != frozen.assessment_donor
                or task.key.training_donors != frozen.fitting_donors
                or task.key.configuration_id != frozen.configuration_id
                or task.key.seed not in frozen.seeds or task.updates != frozen.updates):
            raise ValueError("Outer derivation must use exactly the previously frozen recipe and seed context")
        role = dict(phase="outer", stage="outer", selector=frozen.assessment_donor,
                    task=asdict(task), frozen_outer_recipe=asdict(frozen),
                    frozen_outer_recipe_sha256=_hash(asdict(frozen)), required_recipe_count=1,
                    outer_outcome_selection_allowed=False)
        return cls._create(reader, task, (recipe,), role, adapter_source_sha256)

    @classmethod
    def _create(cls, reader, task, recipes, role, expected_adapter):
        if not isinstance(reader, ForecastReader):
            raise TypeError("An active verified ForecastReader is required")
        reader._access()
        _sha(expected_adapter)
        sources = _source_records()
        if sources != _IMPORTED_SOURCES or expected_adapter != sources[0][1]:
            raise ValueError("Actual calibration adapter/measure equations differ from admitted source")
        doc = reader.spec.document
        binding, context = doc["model_binding"], doc["context"]
        if (tuple(binding["fold"]["training_donors"]) != task.key.training_donors
                or binding["configuration_id"] != task.key.configuration_id
                or binding["model_seed"] != task.key.seed or binding["accepted_updates"] != task.updates
                or context["configuration_id"] != task.key.configuration_id
                or tuple(context["training_donors"]) != task.key.training_donors
                or context["model_seed"] != task.key.seed or context["accepted_update"] != task.updates
                or context["purpose"] != "forecast" or doc["source_reference"]["donor"] != task.validation_donor
                or doc["source_reference"]["condition"] != "Stim8hr"):
            raise ValueError("Parent bank differs from the exact model, seed, stage or query edge")
        # Validate even the parent's recipe without relabelling its publication.
        parent_recipe = Recipe(**doc["recipe"])
        if parent_recipe.configuration_id != task.key.configuration_id:
            raise ValueError("Parent recipe belongs to another model configuration")
        prefixes = tuple(doc["prefixes"])
        if not doc["synthetic_fixture"] and task.phase == "inner":
            required = 4096 if role["stage"] == "finalist" else 512
            if required not in doc["single_seed_prefixes"]:
                raise ValueError("Parent bank lacks the registered inner scoring prefix")
            prefixes = (required,)
        elif not doc["synthetic_fixture"] and tuple(doc["single_seed_prefixes"]) != (4096, 8192):
            raise ValueError("Outer bank requires both registered assessment prefixes")
        parent = _encode(dict(reader.publication_reference))
        runtime = _encode(_runtime())
        if _runtime() != doc["reconstruction_runtime"]:
            raise ValueError("Calibration runtime differs from parent readback")
        manifest = dict(schema="credo.frontier.calibration-derivation.v1", status="source_only_inventory_bound",
            parent_publication=json.loads(parent), parent_recipe=asdict(parent_recipe),
            source_identity=doc["source_identity"], model_binding=binding, context=context,
            catalog_sha256=doc["catalog_sha256"], feature_axis_sha256=doc["feature_axis_sha256"],
            support_sha256=_hash({k: doc[k] for k in ("supported_guide_indices", "unavailable_source_guide_indices",
                                                      "positive_counts", "pooled_control_identity")}),
            role=role, recipes=[asdict(r) for r in recipes], permitted_prefixes=list(prefixes),
            implementation_sources=[dict(module=Path(p).name, sha256=h) for p, h, _ in sources],
            reconstruction_runtime=json.loads(runtime), synthetic_fixture=doc["synthetic_fixture"],
            synthetic_resolution_not_scientific=doc["synthetic_fixture"],
            derivation="raw_prefix_states_and_exact_prefix_weights; complete_source_and_raw_catalog_logs",
            parent_bank_modified=False, particle_generation_calls=0, optimizer_calls=0,
            endpoint_inputs_used=False, selection_performed=False, publication_performed=False,
            requires_owner_pretruth_seal=True, selection_evidence_custody_verified_here=False,
            scientific_launch_authority=False)
        encoded = _encode(manifest)
        obj = object.__new__(cls)
        for name, value in dict(_reader=reader, _spec=reader.spec, _parent=parent, _runtime=runtime,
                _sources=sources, _recipes=recipes, _prefixes=prefixes, _manifest=encoded,
                _identity=hashlib.sha256(encoded).hexdigest()).items():
            object.__setattr__(obj, name, value)
        obj._check()
        return obj

    def _check(self):
        self._reader._access()
        if (self._reader.spec is not self._spec or _encode(dict(self._reader.publication_reference)) != self._parent
                or _encode(_runtime()) != self._runtime):
            raise ValueError("Parent publication or calibration runtime changed")
        for path, _, stamp in self._sources:
            if _stamp(Path(path).lstat()) != stamp:
                raise ValueError("Bound calibration implementation changed")

    def _admit(self, recipe, prefix):
        self._check()
        if not isinstance(recipe, Recipe) or recipe not in self._recipes:
            raise ValueError("Recipe is not in this source-bound derivation inventory")
        if type(prefix) is not int or prefix not in self._prefixes:
            raise ValueError("Prefix is not permitted by this derivation role")

    @property
    def recipes(self):
        self._check()
        return self._recipes

    @property
    def prefixes(self):
        self._check()
        return self._prefixes

    @property
    def identity(self):
        self._check()
        return self._identity

    @property
    def derivation_manifest(self):
        self._check()
        return json.loads(self._manifest)

    def recipe_identity(self, recipe):
        self._check()
        if recipe not in self._recipes:
            raise ValueError("Recipe is not in this derivation inventory")
        return _hash(dict(derivation_sha256=self._identity, recipe=asdict(recipe)))

    @torch.no_grad()
    def law(self, recipe, prefix, index):
        self._admit(recipe, prefix)
        original = self._reader.law(prefix, index)
        if original is None:
            self._check()
            return None
        # Apply once to RAW coordinates, never to the parent's calibrated states.
        states = measure_ops.calibrate_dispersion(original.uncalibrated_states, original.log_weights, recipe.spread_scale)
        result = ConditionalForecast(original.uncalibrated_states, states, original.log_weights, original.log_mass_increment)
        self._check()
        return result

    @torch.no_grad()
    def catalog(self, recipe, prefix):
        self._admit(recipe, prefix)
        result = self._reader.catalog(prefix)
        result["log_frequencies"] = measure_ops.calibrate_abundance(
            result["log_source_frequencies"], result["uncalibrated_log_frequencies"], recipe.alpha)
        if bool((result["log_frequencies"].exp() <= 0).any()):
            raise ValueError("Derived catalog underflow; no undeclared probability floor")
        self._check()
        return result
