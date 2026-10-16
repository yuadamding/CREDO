"""Bounded lazy three-seed finite-measure mixtures of sealed outer forecasts.

Only small complete-catalog vectors are cached, as immutable bytes. A request
copies one corrected-prefix law from each live CalibrationView and concatenates
their particles with the existing ensemble arithmetic. No model, sampler,
optimizer, truth, bank publication or whole-bank materialization is involved.
The owner must seal this derivation inventory before releasing outer outcomes.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import torch

from . import ensemble_forecast as arithmetic
from . import forecast, measure_ops
from .calibration_view import CalibrationView
from .fitting_provider import _hash, _sha
from .forecast_storage import POOL, _runtime
from .selector_contract import Recipe


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _stamp(info):
    return tuple(getattr(info, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"))


def _source_records():
    records = []
    for path in (Path(__file__), Path(arithmetic.__file__), Path(measure_ops.__file__), Path(forecast.__file__)):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
            raise ValueError("Regular bounded local ensemble implementation required")
        raw = path.read_bytes()
        if len(raw) != before.st_size or _stamp(path.lstat()) != _stamp(before):
            raise ValueError("Ensemble implementation changed while binding")
        records.append((str(path), hashlib.sha256(raw).hexdigest(), _stamp(before)))
    return tuple(records)


_IMPORTED_SOURCES = _source_records()


def _copy(blob):
    return torch.from_numpy(np.frombuffer(blob, dtype=np.float64).copy())


@dataclass(frozen=True, init=False)
class EnsembleView:
    _components: tuple
    _component_ids: tuple
    _recipe: Recipe
    _totals: tuple
    _counts: tuple
    _catalogs: tuple
    _catalog_size: int
    _supported: frozenset
    _missing: frozenset
    _pool_present: bool
    _runtime: bytes
    _sources: tuple
    _manifest: bytes
    _identity: str

    def __init__(self, *args, **kwargs):
        raise TypeError("Use EnsembleView.bind with three exact outer calibration views")

    @classmethod
    @torch.no_grad()
    def bind(cls, components, *, totals, maximum_working_bytes, adapter_source_sha256):
        if (type(components) is not tuple or len(components) != 3
                or not all(type(v) is CalibrationView for v in components)
                or len({id(v) for v in components}) != 3
                or len({id(v._reader) for v in components}) != 3):
            raise TypeError("Three distinct ordered seed calibration views/readers required")
        if type(maximum_working_bytes) is not int or maximum_working_bytes < 1:
            raise ValueError("Explicit positive ensemble host/working cap required")
        totals = tuple(totals)
        if not totals or any(type(t) is not int for t in totals) or totals != tuple(sorted(set(totals))):
            raise ValueError("Ascending distinct registered fixed-total budgets required")
        for view in components:
            view._check()
        # Readers traverse whole mapped banks at construction. Lazy copying does
        # not imply low RSS: charge every parent publication in full, with no
        # assumed alias, cache eviction or credit from another owner's budget.
        mapped_bank_bytes = sum(v._reader.publication_reference["total_bytes"] for v in components)
        # Admission precedes parsing full catalog metadata.
        metadata_bytes = 12*sum(len(v._spec._encoded) + len(v._manifest) for v in components) + 1024**2
        if mapped_bank_bytes + metadata_bytes > maximum_working_bytes:
            raise ValueError("Ensemble mapped banks/metadata exceed the explicit working cap")
        _sha(adapter_source_sha256)
        sources = _source_records()
        if sources != _IMPORTED_SOURCES or adapter_source_sha256 != sources[0][1]:
            raise ValueError("Actual ensemble implementation differs from admitted source")
        manifests = tuple(v.derivation_manifest for v in components)
        docs = tuple(v._spec.document for v in components)
        first, first_doc = manifests[0], docs[0]
        synthetic = first_doc["synthetic_fixture"]
        if not synthetic and totals != (4096, 8192):
            raise ValueError("Scientific ensemble requires both registered fixed totals")
        if synthetic and maximum_working_bytes > 8*1024**3:
            raise ValueError("Fabricated ensemble is bounded to eight GiB working memory")
        recipe = components[0].recipes[0]
        plans = []
        for seed, (view, manifest, doc) in enumerate(zip(components, manifests, docs)):
            role, binding, context = manifest["role"], manifest["model_binding"], manifest["context"]
            if (role["phase"] != "outer" or role["stage"] != "outer" or role["required_recipe_count"] != 1
                    or view.recipes != (recipe,) or binding["model_seed"] != seed
                    or binding["accepted_updates"] != 9216 or len(binding["fold"]["training_donors"]) != 3
                    or context["model_seed"] != seed or context["accepted_update"] != 9216
                    or doc["synthetic_fixture"] is not synthetic):
                raise ValueError("Exactly outer seeds zero, one and two of one frozen recipe required")
            if role["frozen_outer_recipe"] != first["role"].get("frozen_outer_recipe"):
                raise ValueError("Ensemble frozen selection recipe/evidence differs")
            common_binding = ("fold", "observation_binding", "configuration_id", "target_support_sha256", "numerical_source_sha256")
            if any(binding[k] != first["model_binding"][k] for k in common_binding):
                raise ValueError("Ensemble fold/model family/representation/support differs")
            a, b = dict(context), dict(first["context"])
            a.pop("model_seed"); b.pop("model_seed")
            if a != b:
                raise ValueError("Ensemble source sampling context differs")
            common_source = ("source_identity", "source_reference", "catalog", "catalog_sha256", "feature_axis_sha256",
                             "captured_counts", "positive_counts", "supported_guide_indices",
                             "unavailable_source_guide_indices", "pooled_control_identity", "reconstruction_runtime")
            if any(doc[k] != first_doc[k] for k in common_source):
                raise ValueError("Ensemble query population, complete catalog or source support differs")
            plan = forecast.ensemble_prefix_plan(seed, totals, doc["single_seed_prefixes"], synthetic_fixture=synthetic)
            declared = forecast.ensemble_prefix_plan(seed, doc["ensemble_component_totals"],
                doc["single_seed_prefixes"], synthetic_fixture=synthetic)
            if (not set(totals) <= set(doc["ensemble_component_totals"])
                    or doc["ensemble_component_prefixes"] != declared
                    or any(p not in view.prefixes for p in plan.values())):
                raise ValueError("Exact component prefixes were not included in the sealed parent banks")
            plans.append(plan)
        g = len(first_doc["catalog"])
        if (synthetic and g > 64) or (not synthetic and g != 26504):
            raise ValueError("Complete registered catalog required")
        retained_catalog = len(totals)*9*g*8
        catalog_working = 40*g*8
        law_working = 16*max(totals)*(2*48*4+2*8)
        working = metadata_bytes + retained_catalog + catalog_working + law_working
        if mapped_bank_bytes + working > maximum_working_bytes:
            raise ValueError("Ensemble catalog/law estimate exceeds explicit working cap")
        # Whole-catalog frequency calculation happens once per total, not once
        # per guide. Only byte-backed catalogs survive this constructor.
        catalogs, catalog_hashes = [], []
        source_reference = None
        for total in totals:
            counts = tuple(plan[str(total)] for plan in plans)
            if counts != measure_ops.round_robin_component_counts(total, 3):
                raise ValueError("Unexpected fixed-total component allocation")
            values = tuple(v.catalog(recipe, n) for v, n in zip(components, counts))
            reference = values[0]["log_source_frequencies"]
            if (any(not torch.equal(v["log_source_frequencies"], reference) for v in values)
                    or (source_reference is not None and not torch.equal(reference, source_reference))):
                raise ValueError("Complete source frequencies differ across seeds or totals")
            source_reference = reference
            raw_components = torch.stack(tuple(v["uncalibrated_log_frequencies"] for v in values))
            calibrated_components = torch.stack(tuple(v["log_frequencies"] for v in values))
            raw = measure_ops.ensemble_catalog(tuple(v["uncalibrated_log_frequencies"] for v in values))
            calibrated = measure_ops.ensemble_catalog(tuple(v["log_frequencies"] for v in values))
            if bool((raw.exp() <= 0).any()) or bool((calibrated.exp() <= 0).any()):
                raise ValueError("Ensemble catalog underflow; no probability floor")
            blobs = tuple(t.contiguous().numpy().tobytes() for t in
                          (reference, raw_components, calibrated_components, raw, calibrated))
            catalogs.append((total, blobs))
            catalog_hashes.append(dict(total=total, sha256=[hashlib.sha256(b).hexdigest() for b in blobs]))
        component_ids = tuple(v.identity for v in components)
        runtime = _encode(_runtime())
        manifest = dict(schema="credo.frontier.lazy-ensemble-derivation.v1", status="source_only_inventory_bound",
            component_derivation_sha256=list(component_ids),
            parent_publications=[m["parent_publication"] for m in manifests],
            model_bindings=[m["model_binding"] for m in manifests], seeds=[0, 1, 2],
            frozen_outer_recipe=first["role"]["frozen_outer_recipe"], recipe=asdict(recipe),
            fold=first["model_binding"]["fold"], observation_binding=first["model_binding"]["observation_binding"],
            source_identity=first["source_identity"], query_donor=first_doc["source_reference"]["donor"],
            catalog_sha256=first_doc["catalog_sha256"], feature_axis_sha256=first_doc["feature_axis_sha256"],
            support_sha256=first["support_sha256"], supported_guide_indices=first_doc["supported_guide_indices"],
            unavailable_source_guide_indices=first_doc["unavailable_source_guide_indices"],
            pooled_control_identity=first_doc["pooled_control_identity"],
            totals=list(totals), component_counts={str(t): list(measure_ops.round_robin_component_counts(t, 3)) for t in totals},
            catalog_content_sha256=catalog_hashes, reconstruction_runtime=json.loads(runtime),
            implementation_sources=[dict(module=Path(p).name, sha256=h) for p, h, _ in sources],
            resources=dict(metadata_estimate_bytes=metadata_bytes, retained_catalog_bytes=retained_catalog,
                catalog_working_bytes=catalog_working, one_law_working_bytes=law_working,
                additional_view_host_estimate_bytes=working, mapped_bank_charge_bytes=mapped_bank_bytes,
                host_admission_estimate_bytes=mapped_bank_bytes+working, maximum_working_bytes=maximum_working_bytes,
                whole_parent_publications_charged=True, mapped_page_eviction_assumed=False,
                estimates_not_measured_peak=True,
                caller_retained_return_values_not_included=True),
            rule="equal_mixture_of_calibrated_finite_measures", conditional_seed_weights="guide_frequency_proportional",
            pooled_control_seed_weights="exactly_one_third_each", raw_and_calibrated_conditional_weights_separate=True,
            particle_order="concatenate_seed0_seed1_seed2_blocks_not_nested_across_total_budgets",
            synthetic_fixture=synthetic, newly_generated_particles=0, model_calls=0, optimizer_calls=0,
            endpoint_inputs_used=False, selection_performed=False, publication_performed=False,
            requires_owner_pretruth_seal=True, scientific_launch_authority=False)
        encoded = _encode(manifest)
        obj = object.__new__(cls)
        fields = dict(_components=components, _component_ids=component_ids, _recipe=recipe, _totals=totals,
            _counts=tuple(measure_ops.round_robin_component_counts(t, 3) for t in totals), _catalogs=tuple(catalogs),
            _catalog_size=g, _supported=frozenset(first_doc["supported_guide_indices"]),
            _missing=frozenset(first_doc["unavailable_source_guide_indices"]),
            _pool_present=first_doc["pooled_control_identity"] is not None, _runtime=runtime, _sources=sources,
            _manifest=encoded, _identity=hashlib.sha256(encoded).hexdigest())
        for name, value in fields.items():
            object.__setattr__(obj, name, value)
        obj._check()
        return obj

    def _check(self):
        if tuple(v.identity for v in self._components) != self._component_ids or _encode(_runtime()) != self._runtime:
            raise ValueError("Ensemble parent derivations or runtime changed")
        for path, _, stamp in self._sources:
            if _stamp(Path(path).lstat()) != stamp:
                raise ValueError("Ensemble implementation changed")

    def _admit(self, total):
        self._check()
        if type(total) is not int or total not in self._totals:
            raise ValueError("Fixed total is outside the sealed derivation inventory")
        position = self._totals.index(total)
        return self._counts[position], self._catalogs[position][1]

    @property
    def identity(self):
        self._check()
        return self._identity

    @property
    def derivation_manifest(self):
        self._check()
        return json.loads(self._manifest)

    @property
    def totals(self):
        self._check()
        return self._totals

    @torch.no_grad()
    def catalog(self, total):
        _, blobs = self._admit(total)
        result = dict(log_source_frequencies=_copy(blobs[0]),
                      uncalibrated_log_frequencies=_copy(blobs[3]), log_frequencies=_copy(blobs[4]))
        self._check()
        return result

    @torch.no_grad()
    def law(self, total, index):
        counts, blobs = self._admit(total)
        if type(index) is not int and index != POOL:
            raise ValueError("Exact guide catalog index or pooled-control key required")
        if index == POOL:
            if not self._pool_present:
                return None
        elif index not in self._supported:
            if index in self._missing:
                return None
            raise ValueError("Guide is outside the complete source inventory")
        laws = tuple(v.law(self._recipe, n, index) for v, n in zip(self._components, counts))
        if any(law is None or len(law.states) != n for law, n in zip(laws, counts)):
            raise ValueError("A required seed law is missing; no component renormalization")
        if index == POOL:
            raw = calibrated = torch.zeros(3, dtype=torch.float64, device="cpu")
        else:
            raw_values = np.frombuffer(blobs[1], dtype=np.float64)
            calibrated_values = np.frombuffer(blobs[2], dtype=np.float64)
            raw = torch.tensor([float(raw_values[s*self._catalog_size+index]) for s in range(3)], dtype=torch.float64, device="cpu")
            calibrated = torch.tensor([float(calibrated_values[s*self._catalog_size+index]) for s in range(3)], dtype=torch.float64, device="cpu")
        result = arithmetic._mix(laws, raw, calibrated)
        self._check()
        return result
