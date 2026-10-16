"""Released packed forecasts -> nine-channel scores, one complete law at a time.

All source-only recipes in the admitted view are scored. Geometry and molecular
readout are computed once per spread, not once per abundance alpha. Catalog
risks remain alpha-specific and joint risks depend on both. No state law is
subsampled and no candidate-dependent support is selected. The output sink must
publish/retain the emitted rows under its separately admitted lifecycle.

This is a numerical join, not an operating-system custody boundary, scheduler,
worker resource admission or scientific launch. The owner must hold the exact
seal/plan custody, live parent reader, verified observation loading, and any
external arrays in addition to the explicit accounting below.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .assessment_support import build_assessment_support
from . import evaluation, measure_ops, molecular
from .calibration_view import CalibrationView
from .ensemble_view import EnsembleView
from .evaluation import (ScoreIdentity, reduce_scores, score_catalog, score_joint,
                         score_molecular_law, score_state_law)
from .fitting_provider import _hash
from .forecast import ForecastSource
from .forecast_seal import ForecastSealPlan, freeze_calibration_derivation, freeze_ensemble_derivation
from .observation import SparseR48
from .selection import CHANNELS
from .training_owner import observation_state_digest
from .truth_compiler import CompiledTruth


@dataclass(frozen=True)
class AssessmentLimits:
    maximum_host_bytes: int
    maximum_device_bytes: int
    maximum_output_bytes: int
    maximum_row_bytes: int = 65536
    pair_block: int = 128
    gene_block: int = 1024
    row_block: int = 128
    caller_retained_host_bytes: int = 0

    def __post_init__(self):
        values = asdict(self)
        if (any(type(v) is not int or v < 1 for k, v in values.items()
                if k != "caller_retained_host_bytes")
                or type(self.caller_retained_host_bytes) is not int or self.caller_retained_host_bytes < 0
                or self.maximum_row_bytes > 1024**2 or self.pair_block > 1024
                or self.row_block > 2048):
            raise ValueError("Explicit positive bounded score working/output/row/block limits required")


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def scientific_recipe_identity(recipe):
    """Seed-independent recipe identity; forecast provenance remains separate."""
    return _hash(dict(schema="credo.frontier.scientific-score-recipe.v1", recipe=asdict(recipe)))


def _recipes(view):
    return (view._recipe,) if type(view) is EnsembleView else view.recipes


def _components(view):
    return view._components if type(view) is EnsembleView else (view,)


def _catalog(view, recipe, prefix):
    return view.catalog(prefix) if type(view) is EnsembleView else view.catalog(recipe, prefix)


def _law(view, recipe, prefix, index):
    return view.law(prefix, index) if type(view) is EnsembleView else view.law(recipe, prefix, index)


def _admission(view, plan, source, truth, observation, prefix):
    if (type(view) not in (CalibrationView, EnsembleView) or not isinstance(plan, ForecastSealPlan)
            or type(source) is not ForecastSource or type(truth) is not CompiledTruth
            or not isinstance(observation, SparseR48)):
        raise TypeError("Live forecast view, frozen seal plan, source, released truth and actual decoder required")
    plan.__post_init__()
    if truth.release.expectation.record() != plan.expectation.record():
        raise ValueError("Truth release differs from the exact complete forecast plan")
    manifest = view.derivation_manifest
    ensemble = type(view) is EnsembleView
    if ensemble and freeze_ensemble_derivation(view) not in plan.ensemble_derivations:
        raise ValueError("Exact ensemble derivation must be sealed before truth release")
    for component in _components(view):
        frozen = freeze_calibration_derivation(component)
        entries = [entry for entry in plan.entries if entry.derivation_manifest == frozen]
        if len(entries) != 1:
            raise ValueError("Exactly one sealed calibration derivation required; raw-only banks are not sufficient")
        entry = entries[0]
        parent = dict(component._reader.publication_reference)
        if (entry.spec.identity != component._reader.spec.identity
                or parent != component.derivation_manifest["parent_publication"]
                or parent["complete_sha256"] != entry.expected_complete_sha256
                or parent["manifest_sha256"] != entry.expected_manifest_sha256):
            raise ValueError("Scoring parent differs from sealed derivation")
    prefixes = view.totals if ensemble else view._reader.spec.document["single_seed_prefixes"]
    if (manifest["source_identity"] != source.identity
            or manifest["catalog_sha256"] != source.catalog_sha256
            or manifest["feature_axis_sha256"] != source.feature_axis_sha256
            or type(prefix) is not int or prefix not in prefixes):
        raise ValueError("Scoring bank/source/prefix differs from sealed derivation")
    if (observation.fold.fold_id != source.fold.fold_id
            or observation.fold.training_donors != source.fold.training_donors
            or observation.fold.feature_axis_sha256 != source.feature_axis_sha256
            or observation.latent_dim != 48
            or observation.features != truth.population.mean_compositions.shape[1]
            or observation_state_digest(observation) != source.observation_binding.state_sha256):
        raise ValueError("Actual frozen fold-native decoder/axis differs")
    if source.synthetic_fixture and (prefix > 128 or observation.device.type != "cpu"):
        raise ValueError("Fabricated scoring requires CPU and at most 128 particles")
    if observation.device.type == "cuda" and observation.device.index is None:
        raise ValueError("An explicit CUDA device index is required")
    return manifest, build_assessment_support(source, truth)


def _estimate(source, truth, observation, view, support, prefix, limits):
    p = truth.population
    paired = [int(row.split(":")[1]) for row in support.required_row_ids if row.startswith("guide:")]
    recipes = _recipes(view)
    spreads = len({r.spread_scale for r in recipes})
    n = max((int(p.positive_counts[i]) for i in paired), default=0)
    s = max((int(source.positive_counts[i]) for i in paired), default=0)
    arrays = sum(getattr(source, k).nbytes for k in ("captured_counts", "positive_counts", "offsets", "states"))
    arrays += sum(getattr(p, k).nbytes for k in ("captured_counts", "positive_counts", "offsets", "states", "mean_compositions"))
    model = sum(t.numel()*t.element_size() for t in observation.state_dict().values())
    # Score kernels use FP64 law copies and a coordinatewise MMD difference +
    # square workspace, not just a block-by-block distance matrix.
    laws = (prefix+n+s)*48*8*6 + (prefix+n+s)*8*8
    geometry = limits.pair_block**2*8*(48*3+16)
    hidden = sum(observation.hidden_dims) + 48
    decoding = prefix*48*8*4 + limits.row_block*(min(limits.gene_block, observation.features)*8*8+hidden*4*8)
    decoding += observation.features*8*24
    working = laws+geometry+decoding+model
    rows = len(support.required_row_ids)
    score_array = len(recipes)*rows*len(CHANNELS)*8
    # Conservative Python bookkeeping and per-recipe reduced coverage/results.
    bookkeeping = len(recipes)*rows*len(CHANNELS)*1024
    components = _components(view)
    metadata = 8*(len(support._encoded)+len(view._manifest)
                  +sum(len(v._manifest)+len(v._reader.spec._encoded) for v in components))
    # The reader verifies all mmap bytes on entry. Without separately qualified
    # page eviction, conservative host admission charges the whole bank too.
    mapped_bank = sum(int(v._reader.publication_reference["total_bytes"]) for v in components)
    ensemble_working = (view.derivation_manifest["resources"]["additional_view_host_estimate_bytes"]
                        if type(view) is EnsembleView else 0)
    host = (arrays+mapped_bank+ensemble_working+2*model+score_array+bookkeeping+metadata+4*limits.maximum_row_bytes
            +len(recipes)*len(source.catalog)*8*8+limits.caller_retained_host_bytes
            + (working if observation.device.type == "cpu" else laws))
    if (host > limits.maximum_host_bytes or working > limits.maximum_device_bytes
            or limits.row_block > observation.maximum_rows):
        raise ValueError("Scoring preflight exceeds explicit host/device/decoder-row limits")
    if source.synthetic_fixture and limits.maximum_host_bytes > 8*1024**3:
        raise ValueError("Fabricated score host allowance cannot exceed 8 GiB")
    pairs = spreads*sum(2*(prefix**2+int(p.positive_counts[i])**2+prefix*int(p.positive_counts[i]))
        +int(source.positive_counts[i])**2+int(source.positive_counts[i])*int(p.positive_counts[i]) for i in paired)
    return dict(host_logical_upper_estimate_bytes=host, device_working_estimate_bytes=working,
        retained_source_truth_array_bytes=arrays, reduction_array_bytes=score_array,
        mapped_bank_residency_charge_bytes=mapped_bank,
        additional_ensemble_view_host_estimate_bytes=ensemble_working,
        conservative_python_bookkeeping_bytes=bookkeeping,
        full_law_maximum_source_cells=s, full_law_maximum_endpoint_cells=n,
        geometry_distance_pair_terms_upper_estimate=pairs, mmd_bandwidths_per_pair=4, distinct_spreads=spreads,
        recipe_count=len(recipes), no_full_bank_copy=True, no_law_subsampling=True,
        decoder_and_external_storage_loads_are_caller_owned=True,
        measured_RSS_GPU_peak_or_throughput=False)


def _merge(*records):
    result = dict(records[0])
    result["unavailable"] = {}
    result["roundoff"] = {}
    for record in records:
        result.update({k: v for k, v in record.items() if k not in ("unavailable", "roundoff")})
        result["unavailable"].update(record.get("unavailable", {}))
        result["roundoff"].update(record.get("roundoff", {}))
    return result


def _score_sources():
    paths = [Path(__file__), Path(evaluation.__file__), Path(measure_ops.__file__),
             Path(molecular.__file__)]
    paths += [Path(__file__).with_name(name+".py") for name in
              ("observation", "calibration_view", "ensemble_view", "assessment_support", "forecast_seal")]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@torch.no_grad()
def score_calibration_bank(view, *, plan, source, truth, observation, prefix, limits, row_sink):
    """Emit canonical JSONL rows and return recipe reductions + completion record.

    row_sink accepts bytes and must not silently drop/alter them. Its publication
    and final hash/readback are separate caller responsibilities. Failure raises
    without a completed result; prior rows are an incomplete attempt, not a retry.
    Returned reductions do not claim scientific acceptance or select a recipe.
    """
    if type(view) is not CalibrationView:
        raise TypeError("Single-seed calibration view required")
    return _score_bank(view, plan=plan, source=source, truth=truth, observation=observation,
                       prefix=prefix, limits=limits, row_sink=row_sink)


@torch.no_grad()
def score_ensemble_bank(view, *, plan, source, truth, observation, total, limits, row_sink):
    """Assess one presealed fixed three-seed mixture, never an inner candidate.

    Its one outer-selected recipe is already applied in EnsembleView. This does
    not refit calibration, average particle coordinates or average seed scores.
    The total budget, not three times that budget, enters the scoring kernels.
    """
    if type(view) is not EnsembleView:
        raise TypeError("Source-only fixed three-seed ensemble view required")
    return _score_bank(view, plan=plan, source=source, truth=truth, observation=observation,
                       prefix=total, limits=limits, row_sink=row_sink)


def _score_bank(view, *, plan, source, truth, observation, prefix, limits, row_sink):
    if not isinstance(limits, AssessmentLimits) or not callable(row_sink):
        raise TypeError("Explicit scoring limits and bounded external row sink required")
    limits.__post_init__()
    manifest, support = _admission(view, plan, source, truth, observation, prefix)
    estimate = _estimate(source, truth, observation, view, support, prefix, limits)
    score_sources = _score_sources()
    recipes = _recipes(view)
    ensemble = type(view) is EnsembleView
    components = _components(view)
    bindings = [v.derivation_manifest["model_binding"] for v in components]
    mixture_model = dict(schema="credo.frontier.fixed-three-seed-model-identity.v1",
        components=[dict(seed=b["model_seed"], model_sha256=b["model_artifact_sha256"]) for b in bindings],
        rule="equal_mixture_of_calibrated_finite_measures") if ensemble else None
    recipe_index = {r: i for i, r in enumerate(recipes)}
    weights = support.weights
    required = support.required_row_ids
    row_index = {key: i for i, key in enumerate(required)}
    values = np.full((len(recipes), len(required), len(CHANNELS)), np.nan, dtype=np.float64)
    reasons = {}
    identities = [ScoreIdentity(source.fold.fold_id, source.reference.donor, scientific_recipe_identity(r),
        None if ensemble else bindings[0]["model_seed"], prefix, bindings[0]["numerical_source_sha256"],
        source.observation_binding.artifact_sha256,
        _hash(mixture_model) if ensemble else bindings[0]["model_artifact_sha256"],
        _hash(dict(source=source.identity, endpoint=truth.population.reference.raw_summary_sha256,
                   endpoint_latent=truth.population.reference.latent_sha256, support=support.identity)),
        source.catalog_sha256, source.feature_axis_sha256,
        predictor_kind="three_seed_finite_measure_mixture" if ensemble else "single_seed") for r in recipes]
    device = observation.device
    counts = torch.tensor(truth.population.captured_counts.copy(), dtype=torch.int64, device=device)
    total = sum(int(v) for v in truth.population.captured_counts)
    if total <= 0:
        raise ValueError("No endpoint captured total; no synthetic catalog distribution")
    output_hash = hashlib.sha256()
    output_bytes = emitted_rows = decoder_calls = state_calls = joint_calls = 0
    seen = set()

    def emit(r, key, score):
        nonlocal output_bytes, emitted_rows
        ri = recipe_index[r]
        identity = asdict(identities[ri])
        score = dict(score, identity=identity, arithmetic_only=False)
        address = (ri, key)
        if key not in row_index or address in seen:
            raise ValueError("Missing or duplicated fixed score row")
        for ci, channel in enumerate(CHANNELS):
            if key not in weights[channel]:
                continue
            if channel not in score:
                raise ValueError("Required fixed channel absent from emitted row")
            value = score[channel]
            reason = score.get("unavailable", {}).get(channel)
            if value is None:
                if not isinstance(reason, str) or not reason:
                    raise ValueError("Missing required score needs explicit reason")
                reasons[ri, row_index[key], ci] = reason
            elif reason is not None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid required score is not removable support")
            else:
                values[ri, row_index[key], ci] = value
        record = dict(recipe=asdict(r), recipe_identity=identities[ri].recipe_id,
                      prefix=prefix, row_id=key, score=score)
        payload = _encode(record)
        if len(payload) > limits.maximum_row_bytes or output_bytes+len(payload) > limits.maximum_output_bytes:
            raise ValueError("Per-row or total scoring output allowance exceeded")
        row_sink(payload)
        output_hash.update(payload)
        output_bytes += len(payload); emitted_rows += 1; seen.add(address)

    def decode(law):
        nonlocal decoder_calls
        if law is None:
            return None
        decoder_calls += 1
        return observation.weighted_composition(law.states.to(device), law.log_weights.exp().to(device),
            standardized=True, gene_block=limits.gene_block, row_block=limits.row_block, recompute_backward=False)

    def observed_law(population, i):
        start, stop = map(int, population.offsets[i:i+2])
        state = torch.tensor(population.states[start:stop].copy(), dtype=torch.float64, device=device)
        weight = torch.full((stop-start,), 1.0/(stop-start), dtype=torch.float64, device=device)
        return state, weight

    # Cell-equal observed NTC means: accumulate one gene vector, never gather
    # all control rows into a second control-by-gene matrix.
    controls = [i for i, guide in enumerate(source.catalog) if guide.is_control]
    control_total = sum(int(truth.population.positive_counts[i]) for i in controls)
    control_observed = None
    if control_total:
        control_observed = torch.zeros(observation.features, dtype=torch.float64, device=device)
        for i in controls:
            count = int(truth.population.positive_counts[i])
            if count:
                control_observed += torch.tensor(truth.population.mean_compositions[i].copy(),
                    dtype=torch.float64, device=device)*(count/control_total)

    # Count depends on alpha only. Keep the small vectors; each spread reuses
    # the same exact catalog computation and endpoint denominator.
    catalogs = {}
    count_records = {}
    for r in recipes:
        if r.alpha not in catalogs:
            catalogs[r.alpha] = _catalog(view, r, prefix)["log_frequencies"].to(device)
            count_records[r.alpha] = score_catalog(catalogs[r.alpha], counts,
                expected_catalog_size=len(source.catalog))
        emit(r, "catalog", count_records[r.alpha])

    paired = [int(key.split(":")[1]) for key in required if key.startswith("guide:")]
    groups = {spread: tuple(r for r in recipes if r.spread_scale == spread)
              for spread in dict.fromkeys(r.spread_scale for r in recipes)}
    for group in groups.values():
        representative = group[0]
        control_law = _law(view, representative, prefix, "pooled_control")
        control_predicted = decode(control_law)
        del control_law
        control_record = score_molecular_law(None, None, control_predicted, control_observed)
        for r in group:
            emit(r, "pooled_control", control_record)
        for i in paired:
            law = _law(view, representative, prefix, i)
            target = not source.catalog[i].is_control
            key = f"guide:{i}"
            if law is None:
                channels = [name for name in CHANNELS if key in weights[name]]
                record = dict(dtype="float64", unavailable={name: "candidate_missing_prediction" for name in channels},
                              roundoff={}, **{name: None for name in channels})
                for r in group:
                    emit(r, key, record)
                continue
            x, w = law.states.to(device), law.log_weights.exp().to(device)
            y, v = observed_law(truth.population, i)
            s, u = observed_law(source, i)
            geometry = score_state_law(x, w, y, v, s, u, block=limits.pair_block)
            state_calls += 1
            molecular = {}
            if target:
                predicted = decode(law)
                observed = torch.tensor(truth.population.mean_compositions[i].copy(), dtype=torch.float64, device=device)
                molecular = score_molecular_law(predicted, observed, control_predicted, control_observed)
                del predicted, observed
            for r in group:
                joint = score_joint(x, w, float(catalogs[r.alpha][i].exp()), y, v,
                                    float(int(truth.population.captured_counts[i])/total))
                joint_calls += 1
                emit(r, key, _merge(geometry, joint, molecular))
            del law, x, w, y, v, s, u, geometry, molecular
        del control_predicted, control_record

    if len(seen) != len(recipes)*len(required):
        raise ValueError("Incomplete required recipe/row inventory")
    reductions = []
    for ri, r in enumerate(recipes):
        identity = identities[ri]
        rows = {key: dict(identity=asdict(identity), arithmetic_only=False, unavailable={}) for key in required}
        for ci, channel in enumerate(CHANNELS):
            for key in weights[channel]:
                value = values[ri, row_index[key], ci]
                rows[key][channel] = None if np.isnan(value) else float(value)
                if np.isnan(value):
                    rows[key]["unavailable"][channel] = reasons[ri, row_index[key], ci]
        reductions.append(dict(recipe=asdict(r), scores=reduce_scores(rows, weights, identity=identity)))
        del rows
    # Check actual arrays/weights after the whole operation, not just labels.
    source.assert_unchanged(); truth.assert_unchanged()
    if (view.derivation_manifest != manifest
            or observation_state_digest(observation) != source.observation_binding.state_sha256
            or plan.expectation.record() != truth.release.expectation.record()
            or _score_sources() != score_sources):
        raise ValueError("Scoring source/decoder/release changed during computation")
    receipt = dict(schema=("credo.frontier.streamed-ensemble-assessment.v1" if ensemble
                           else "credo.frontier.streamed-calibration-assessment.v2"), status="numerical_complete_not_published",
        derivation_sha256=view.identity, forecast_seal_sha256=truth.release.manifest_sha256,
        parent_publications=[dict(v._reader.publication_reference) for v in components],
        prediction_derivations={scientific_recipe_identity(r):
            (_hash(dict(ensemble_derivation_sha256=view.identity, recipe=asdict(r))) if ensemble
             else view.recipe_identity(r)) for r in recipes},
        ensemble_model_identity=mixture_model, predictor_kind=identities[0].predictor_kind,
        early_source_identity=source.identity, support_sha256=support.identity,
        observation_binding=asdict(source.observation_binding), prefix=prefix,
        scoring_implementation_sources=score_sources,
        runtime=dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
                     torch_threads=torch.get_num_threads(), score_device=str(device),
                     geometry_and_reduction_dtype="float64", decoder_neural_dtype="float32"),
        score_identity_semantics=dict(source_sha256="producer-bound numerical forecast source",
            input_sha256="hash of early source, released endpoint raw/latent and common support identities",
            recipe_id="seed-independent canonical scientific recipe; prediction_derivations binds each actual view"),
        support_coverage=support.record["coverage"], limits=asdict(limits), estimate=estimate,
        recipe_count=len(recipes), emitted_rows=emitted_rows, rows_bytes=output_bytes,
        rows_sha256=output_hash.hexdigest(), state_score_calls=state_calls,
        decoder_calls=decoder_calls, joint_score_calls=joint_calls,
        distinct_catalog_scores=len(catalogs), geometry_RNA_reused_across_alpha=not ensemble,
        observation_artifact_loading_verified_here=False, external_sink_publication_verified_here=False,
        scientific_operation_inventory_admitted_here=False, optimizer_calls=0, neural_forecast_calls=0,
        scientific_launch_authority=False)
    result = dict(scores=reductions, receipt=receipt)
    summary_bytes = len(_encode(result))
    if output_bytes+summary_bytes > limits.maximum_output_bytes:
        raise ValueError("Complete rows plus assessment summary exceed output allowance")
    receipt["summary_bytes_before_size_field"] = summary_bytes
    # A final fixed-size margin pays for this accounting field itself.
    if output_bytes+len(_encode(result)) > limits.maximum_output_bytes:
        raise ValueError("Complete assessment output allowance exceeded")
    return result
