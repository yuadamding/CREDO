"""Tiny source -> packed forecast -> sealed derivation -> released truth tests.

All counts, particles and decoder weights are fabricated locally. No model
rollout, optimizer, external observation or GPU is used. This exercises actual
native role gates and packed-file readers, not a mocked release token.
"""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frontier import calibration_view
from frontier.assessment import AssessmentLimits, score_calibration_bank, scientific_recipe_identity
from frontier.assessment_support import build_assessment_support
from frontier.calibration_view import CalibrationView
from frontier.evaluation import (ScoreIdentity, reduce_scores, score_catalog,
    score_joint, score_molecular_law, score_state_law)
from frontier.forecast import ConditionalForecast, ForecastModelBinding
from frontier.forecast_seal import (ForecastSealEntry, ForecastSealPlan,
    freeze_calibration_derivation, publish_forecast_seal)
from frontier.forecast_storage import (ForecastReader, ForecastStorageLimits,
    ForecastStorageSpec, ForecastWriter)
from frontier.measure_ops import (calibrate_abundance, calibrate_dispersion,
    catalog_log_frequencies, source_log_frequencies)
from frontier.native_inputs import FoldInputs
from frontier.planner import ScoreTask, TrajectoryKey, screen_plan
from frontier.query_compiler import compile_query_source
from frontier.sampling import SamplingContext
from frontier.selection import CHANNELS
from frontier.selector_contract import OuterRecipe, Recipe
from frontier.training_owner import observation_state_digest
from frontier.truth_compiler import compile_evaluation_truth
from test_assessment_support import package as native_package


PREFIX = 7
MIB = 1024**2


@pytest.fixture(autouse=True)
def bounded_no_optimization(monkeypatch):
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        monkeypatch.setattr(cls, "step", lambda *_a, **_k: pytest.fail("optimizer forbidden"))
    yield
    torch.set_num_threads(original)


def make_bank(path, source, *, offset=0., ensemble_totals=(), seed=0):
    updates = 9216 if len(source.fold.training_donors) == 3 else 1024
    binding = ForecastModelBinding(source.fold, source.observation_binding, "F0-V0",
        hashlib.sha256(f"model-{seed}".encode()).hexdigest(), "c"*64, "d"*64, "e"*64, seed, updates)
    context = SamplingContext("fabricated-assessment", source.fold.training_donors,
                              "F0-V0", 17, seed, updates, "forecast")
    ordinary_prefixes = tuple(sorted({PREFIX, *ensemble_totals}))
    spec = ForecastStorageSpec.from_source(source, binding, context, Recipe("F0-V0"),
        prefixes=ordinary_prefixes, ensemble_component_totals=ensemble_totals)
    doc = spec.document
    receipt = {key: doc[key] for key in ("source_identity", "model_binding", "context", "recipe", "prefixes",
        "single_seed_prefixes", "ensemble_component_totals", "ensemble_component_prefixes",
        "maximum_generated_particles", "synthetic_fixture", "supported_guide_indices",
        "unavailable_source_guide_indices", "pooled_control_identity")}
    receipt.update(schema="credo.frontier.in-memory-forecast.v1", complement_mass_multiplier=1,
        pooled_control_is_catalog_entry=False, source_only_support=True, endpoint_inputs_used=False,
        optimizer_calls=0, input_artifact_custody_verified_here=False, checkpoint_loading_verified_here=False,
        publication_claimed=False, scientific_launch_authority=False, control_ids=["NTC"],
        query_donor="D1", sampler_receipts=[], complete_prediction_checks_passed=True,
        law_storage="in_memory", sink_spec_sha256=None)
    supported = doc["supported_guide_indices"]
    all_mass = torch.zeros(len(source.catalog), dtype=torch.float64)
    maximum = max(ordinary_prefixes)
    states = torch.arange(maximum*48, dtype=torch.float32).reshape(maximum, 48)/200
    with ForecastWriter(path, spec=spec, producer_sha256="f"*64,
                        limits=ForecastStorageLimits(128*MIB, 128*MIB, 2)) as writer:
        for prefix in spec.effective_prefixes:
            logs = torch.log_softmax(torch.linspace(-.4, .7, maximum, dtype=torch.float64)[:prefix], 0)
            for index in supported:
                raw = states[:prefix] + index/10 + offset
                all_mass[index] = (index-2)*(.12+.04*seed)
                writer.emit(prefix, index, ConditionalForecast(raw, calibrate_dispersion(raw, logs, 1.),
                                                             logs, float(all_mass[index])))
            if doc["pooled_control_identity"] is not None:
                raw = states[:prefix]-.2+offset
                writer.emit(prefix, "pooled_control", ConditionalForecast(raw, calibrate_dispersion(raw, logs, 1.), logs, 0.))
            base = source_log_frequencies(torch.from_numpy(source.captured_counts.copy()))
            raw = catalog_log_frequencies(base, torch.tensor(supported), all_mass[supported])
            writer.write_catalog(prefix, log_mass_increments=all_mass, log_source_frequencies=base,
                uncalibrated_log_frequencies=raw, log_frequencies=calibrate_abundance(base, raw, 1.))
        publication = writer.finish(receipt)
    return ForecastSealEntry(f"F0-V0/seed-{seed}", path.name, spec,
                             publication["complete_sha256"], publication["manifest_sha256"])


def open_view(stack, root, entry):
    reader = stack.enter_context(ForecastReader(root/entry.bank_relative_root, spec=entry.spec,
        expected_complete_sha256=entry.expected_complete_sha256,
        expected_manifest_sha256=entry.expected_manifest_sha256,
        maximum_output_bytes=128*MIB, maximum_working_bytes=128*MIB))
    adapter = hashlib.sha256(Path(calibration_view.__file__).read_bytes()).hexdigest()
    donors = tuple(entry.spec.document["model_binding"]["fold"]["training_donors"])
    if len(donors) == 3:
        task = ScoreTask(TrajectoryKey(donors, "F0-V0", 0), "D1", "D1", 9216, "outer")
        frozen = OuterRecipe("D1", donors, "F0-V0", 1., 1., "1"*64, "2"*64, 1.)
        view = CalibrationView.for_outer(reader, frozen=frozen, task=task, adapter_source_sha256=adapter)
    else:
        stage = screen_plan()
        task = next(t for t in stage.scores_for("D2") if t.key.training_donors == ("D3", "D4")
                    and t.key.configuration_id == "F0-V0")
        view = CalibrationView.for_inner(reader, plan=stage, selector="D2", task=task, adapter_source_sha256=adapter)
    return reader, view


def seal_truth(package, *, name, entry, reader, view):
    source, args = package.source, package.args
    if view is not None:
        entry = replace(entry, derivation_manifest=freeze_calibration_derivation(view))
    plan = ForecastSealPlan(source.fold, "D1", source.reference.source_access_sha256, (entry,))
    release = publish_forecast_seal(package.bank_root, name, plan=plan, expectation=plan.expectation,
        readers=(reader,), native_index=package.index, output_cap_bytes=128*1024,
        derivations=None if view is None else (view,))
    endpoint = package.index.open(package.data_root, source.fold, "evaluation_truth",
                                  query_donor="D1", release=release, max_batch_rows=3)
    truth = compile_evaluation_truth(endpoint, **(args | dict(source_id="D1_Stim48hr", release=release)))
    return plan, truth


@pytest.fixture
def fixture(tmp_path, request):
    options = dict(getattr(request, "param", {}))
    outer = options.pop("outer", False)
    data_root, index, args = native_package(tmp_path, **options)
    if outer:
        args["fold"] = FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4"))
        observation = args["observation"]
        observation.fold = replace(observation.fold, fold_id=args["fold"].fold_id,
                                   training_donors=args["fold"].training_donors)
        args["observation_binding"] = replace(args["observation_binding"], state_sha256=observation_state_digest(observation))
    query = index.open(data_root, args["fold"], "query", query_donor="D1", max_batch_rows=3)
    source = compile_query_source(query, **args).source
    bank_root = tmp_path/"banks"; bank_root.mkdir()
    entry = make_bank(bank_root/"forecast", source, ensemble_totals=(12,) if outer else ())
    with ExitStack() as stack:
        reader, view = open_view(stack, bank_root, entry)
        result = SimpleNamespace(data_root=data_root, index=index, args=args, source=source,
            bank_root=bank_root, entry=entry, reader=reader, view=view, stack=stack,
            observation=args["observation"])
        result.plan, result.truth = seal_truth(result, name="SEAL.json", entry=entry, reader=reader, view=view)
        yield result


def limits(**changes):
    return AssessmentLimits(**(dict(maximum_host_bytes=128*MIB, maximum_device_bytes=128*MIB,
        maximum_output_bytes=8*MIB, maximum_row_bytes=65536, pair_block=2,
        gene_block=2, row_block=3, caller_retained_host_bytes=0) | changes))


def score(package, **changes):
    rows = []
    args = dict(plan=package.plan, source=package.source, truth=package.truth,
        observation=package.observation, prefix=PREFIX, limits=limits(), row_sink=rows.append)
    args.update(changes)
    result = score_calibration_bank(package.view, **args)
    return result, rows


def law_tensor(population, index):
    start, stop = population.offsets[index:index+2]
    states = torch.from_numpy(population.states[start:stop].copy())
    return states, torch.full((len(states),), 1/len(states), dtype=torch.float64)


def literal_decode(observation, law):
    """Full-gene eager decoder, not the scorer's blockwise composition method."""
    with torch.no_grad():
        blocks = []
        for start in range(0, len(law.states), 3):
            raw = law.states[start:start+3]*observation.scale+observation.center
            blocks.append(torch.softmax(observation.decoder(raw).double(), -1))
        probability = torch.cat(blocks)
    return (law.log_weights.exp()[:, None]*probability).sum(0)


def test_complete_calibration_grid_streams_fixed_support_and_shared_control_once(fixture):
    result, raw_rows = score(fixture)
    assert len(result["scores"]) == 24
    rows = [json.loads(raw) for raw in raw_rows]
    support = build_assessment_support(fixture.source, fixture.truth)
    assert len(rows) == 24*len(support.required_row_ids)
    assert all(type(raw) is bytes and raw.endswith(b"\n") for raw in raw_rows)
    assert all(raw == (json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
               for raw in raw_rows)
    for recipe in fixture.view.recipes:
        selected = [r for r in rows if r["recipe"] == asdict(recipe)]
        assert {r["row_id"] for r in selected} == set(support.required_row_ids)
        assert len({r["recipe_identity"] for r in selected}) == 1
        assert selected[0]["recipe_identity"] == scientific_recipe_identity(recipe)
        assert result["receipt"]["prediction_derivations"][selected[0]["recipe_identity"]] == fixture.view.recipe_identity(recipe)
        assert all(r["prefix"] == PREFIX for r in selected)
        catalog = next(r["score"] for r in selected if r["row_id"] == "catalog")
        assert catalog["catalog_size"] == 8
        assert catalog["endpoint_zero_categories"] == 2
        assert len([r for r in selected if r["row_id"] == "pooled_control"]) == 1
    assert all(all(channel in row["scores"] for channel in CHANNELS) for row in result["scores"])
    receipt = result["receipt"]
    assert receipt["state_score_calls"] == 4*4
    assert receipt["decoder_calls"] == 4*(3+1)
    assert receipt["joint_score_calls"] == 24*4
    assert receipt["distinct_catalog_scores"] == 6
    assert receipt["emitted_rows"] == len(rows)
    assert receipt["rows_sha256"] == hashlib.sha256(b"".join(raw_rows)).hexdigest()
    assert receipt["rows_bytes"] == sum(map(len, raw_rows))
    assert receipt["optimizer_calls"] == receipt["neural_forecast_calls"] == 0
    assert receipt["forecast_seal_sha256"] == fixture.truth.release.manifest_sha256
    assert receipt["observation_binding"] == asdict(fixture.source.observation_binding)


def test_streamed_all_nine_channels_equal_literal_per_recipe_evaluation(fixture):
    result, raw_rows = score(fixture)
    emitted = [json.loads(raw) for raw in raw_rows]
    support = build_assessment_support(fixture.source, fixture.truth)
    population = fixture.truth.population
    counts = torch.from_numpy(population.captured_counts.copy())
    p = counts.double()/sum(population.captured_counts)
    controls = [i for i, guide in enumerate(fixture.source.catalog) if guide.is_control]
    control_total = sum(int(population.positive_counts[i]) for i in controls)
    observed_control = sum(torch.from_numpy(population.mean_compositions[i].copy())
        *float(population.positive_counts[i]/control_total)
        for i in controls if population.positive_counts[i])
    for recipe in fixture.view.recipes:
        chosen = [row for row in emitted if row["recipe"] == asdict(recipe)]
        identity = ScoreIdentity(**chosen[0]["score"]["identity"])
        catalog = fixture.view.catalog(recipe, PREFIX)["log_frequencies"]
        pool = literal_decode(fixture.observation, fixture.view.law(recipe, PREFIX, "pooled_control"))
        expected = {
            "catalog": score_catalog(catalog, counts, expected_catalog_size=8, identity=identity),
            "pooled_control": score_molecular_law(None, None, pool, observed_control, identity=identity),
        }
        for i in range(4):
            law = fixture.view.law(recipe, PREFIX, i)
            y, v = law_tensor(population, i)
            s, u = law_tensor(fixture.source, i)
            record = score_state_law(law.states, law.log_weights.exp(), y, v, s, u, block=2, identity=identity)
            record.update(score_joint(law.states, law.log_weights.exp(), float(catalog[i].exp()),
                                      y, v, float(p[i]), identity=identity))
            if not fixture.source.catalog[i].is_control:
                record.update(score_molecular_law(literal_decode(fixture.observation, law),
                    torch.from_numpy(population.mean_compositions[i].copy()), pool, observed_control,
                    identity=identity))
            expected[f"guide:{i}"] = record
        reduced = reduce_scores(expected, support.weights, identity=identity)
        actual = next(row["scores"] for row in result["scores"] if row["recipe"] == asdict(recipe))
        for channel in CHANNELS:
            # Full-gene and gene-block FP32 affine calls can round differently;
            # this is numerical parity, not a bitwise BLAS claim.
            tolerance = 3e-6 if "mse" in channel and channel != "joint_mse" else 1e-11
            assert actual[channel] == pytest.approx(reduced[channel], rel=3e-6, abs=tolerance)
            assert actual["coverage"][channel] == reduced["coverage"][channel]


def test_shared_control_and_geometry_are_computed_once_per_spread_not_alpha(fixture, monkeypatch):
    from frontier import assessment
    calls = dict(geometry=0, decoder=0, catalog=0)
    original_geometry = assessment.score_state_law
    original_decoder = fixture.observation.weighted_composition
    original_catalog = assessment.score_catalog
    def geometry(*args, **kwargs):
        calls["geometry"] += 1
        return original_geometry(*args, **kwargs)
    def decoder(*args, **kwargs):
        calls["decoder"] += 1
        return original_decoder(*args, **kwargs)
    def catalog(*args, **kwargs):
        calls["catalog"] += 1
        return original_catalog(*args, **kwargs)
    monkeypatch.setattr(assessment, "score_state_law", geometry)
    monkeypatch.setattr(assessment, "score_catalog", catalog)
    monkeypatch.setattr(fixture.observation, "weighted_composition", decoder)
    score(fixture)
    assert calls == dict(geometry=16, decoder=16, catalog=6)


@pytest.mark.parametrize("cap", ["maximum_host_bytes", "maximum_device_bytes", "maximum_output_bytes", "maximum_row_bytes"])
def test_insufficient_explicit_memory_or_output_limit_never_returns_completion(fixture, cap):
    rows = []
    with pytest.raises((ValueError, MemoryError)):
        score(fixture, limits=limits(**{cap: 1}), row_sink=rows.append)


def test_sink_failure_propagates_without_completion(fixture):
    seen = []
    def fail(raw):
        seen.append(raw)
        raise OSError("fabricated sink failure")
    with pytest.raises(OSError, match="fabricated sink failure"):
        score(fixture, row_sink=fail)
    assert len(seen) == 1


def test_raw_only_release_does_not_admit_calibration_grid(fixture):
    plan, truth = seal_truth(fixture, name="RAW_ONLY.json", entry=fixture.entry,
                            reader=fixture.reader, view=None)
    with pytest.raises(ValueError):
        score(fixture, plan=plan, truth=truth)


def test_foreign_valid_bank_cannot_use_other_banks_release(fixture):
    entry = make_bank(fixture.bank_root/"different", fixture.source, offset=.5)
    _, view = open_view(fixture.stack, fixture.bank_root, entry)
    with pytest.raises(ValueError):
        score_calibration_bank(view, plan=fixture.plan, source=fixture.source, truth=fixture.truth,
            observation=fixture.observation, prefix=PREFIX, limits=limits(), row_sink=lambda _row: None)


def test_seal_for_a_different_inventory_is_rejected_even_with_same_query(fixture):
    raw_plan, _ = seal_truth(fixture, name="OTHER_PLAN.json", entry=fixture.entry,
                             reader=fixture.reader, view=None)
    with pytest.raises(ValueError):
        score(fixture, plan=raw_plan)


@pytest.mark.parametrize("change", ["weights", "scaler", "unfrozen", "training"])
def test_actual_decoder_not_merely_representation_label_is_bound(fixture, change):
    observation = deepcopy(fixture.observation)
    with torch.no_grad():
        if change == "weights": next(observation.decoder.parameters()).add_(.01)
        elif change == "scaler": observation.center.add_(.01)
        elif change == "unfrozen": next(observation.parameters()).requires_grad_(True)
        else: observation.train()
    with pytest.raises(ValueError):
        score(fixture, observation=observation)


def test_source_content_mutation_rejected_before_rows(fixture):
    values = fixture.source.states
    values.setflags(write=True); values[0, 0] += 1; values.setflags(write=False)
    rows = []
    with pytest.raises(ValueError, match="source.*changed|source.*identity"):
        score(fixture, row_sink=rows.append)
    assert rows == []


def test_source_mutation_during_sink_is_detected_without_completion(fixture):
    def mutate(_raw):
        values = fixture.source.captured_counts
        values.setflags(write=True); values[0] += 1; values.setflags(write=False)
    with pytest.raises(ValueError, match="source.*changed|source.*identity"):
        score(fixture, row_sink=mutate)


def test_decoder_mutation_during_sink_is_detected_without_completion(fixture):
    def mutate(_raw):
        with torch.no_grad(): next(fixture.observation.decoder.parameters()).add_(.01)
    with pytest.raises(ValueError):
        score(fixture, row_sink=mutate)


def test_unregistered_prefix_does_not_read_laws(fixture, monkeypatch):
    monkeypatch.setattr(CalibrationView, "law", lambda *_a, **_k: pytest.fail("unregistered law read"))
    with pytest.raises(ValueError): score(fixture, prefix=PREFIX-1)


@pytest.mark.parametrize("fixture", [{"outer": True}], indirect=True)
def test_ensemble_component_prefix_is_not_a_single_seed_assessment(fixture, monkeypatch):
    assert fixture.view.prefixes == (4, PREFIX, 12)
    assert fixture.reader.spec.document["single_seed_prefixes"] == [PREFIX, 12]
    assert fixture.reader.spec.document["ensemble_component_prefixes"] == {"12": 4}
    monkeypatch.setattr(CalibrationView, "law", lambda *_a, **_k: pytest.fail("component-prefix law read"))
    with pytest.raises(ValueError, match="prefix"):
        score(fixture, prefix=4)


def test_forecast_reader_must_remain_open(fixture):
    fixture.reader.close()
    with pytest.raises((ValueError, RuntimeError)): score(fixture)


@pytest.mark.parametrize("fixture", [
    {"source_positive": [2, 1, 1, 0, 1, 0, 0, 0]},
    {"endpoint_positive": [2, 1, 1, 0, 0, 0, 1, 0]},
], indirect=True, ids=["missing-source-NTC", "missing-endpoint-NTC"])
def test_unavailable_control_does_not_remove_valid_target_endpoints(fixture):
    result, raw = score(fixture)
    support = build_assessment_support(fixture.source, fixture.truth)
    for row in result["scores"]:
        scores = row["scores"]
        assert scores["target_endpoint_mse"] is not None
        assert scores["count_quadratic"] is not None
        assert scores["pooled_effect_mse"] is None
        assert scores["pooled_control_endpoint_mse"] is None
        assert scores["coverage"]["target_endpoint_mse"]["available_weight"] == 1
        assert scores["coverage"]["pooled_effect_mse"]["available_weight"] == 0
    assert len(raw) == 24*len(support.required_row_ids)


@pytest.mark.parametrize("fixture", [dict(source_positive=[1,0,0,1,0,0,0,0],
    endpoint_positive=[0,1,0,0,0,1,0,0])], indirect=True)
def test_no_paired_law_keeps_catalog_and_pooled_control_not_zero_state(fixture):
    result, raw = score(fixture)
    for item in result["scores"]:
        scores = item["scores"]
        for channel in ("state_energy", "spread_absolute_error", "shape_mmd2", "joint_mse",
                        "target_endpoint_mse", "pooled_effect_mse"):
            assert scores[channel] is None
        assert scores["count_quadratic"] is not None
        assert scores["pooled_control_endpoint_mse"] is not None
    assert {json.loads(row)["row_id"] for row in raw} == {"catalog", "pooled_control"}
    assert result["receipt"]["state_score_calls"] == 0
    assert result["receipt"]["decoder_calls"] == 4


def test_missing_candidate_law_retains_fixed_denominator_and_other_channels(fixture, monkeypatch):
    original = CalibrationView.law
    def missing(self, recipe, prefix, index):
        return None if index == 0 else original(self, recipe, prefix, index)
    monkeypatch.setattr(CalibrationView, "law", missing)
    result, raw = score(fixture)
    for item in result["scores"]:
        scores = item["scores"]
        assert scores["state_energy"] is None
        assert scores["target_endpoint_mse"] is None
        assert scores["coverage"]["state_energy"]["available_weight"] == .75
        assert scores["coverage"]["target_endpoint_mse"]["available_weight"] == .75
        assert scores["count_quadratic"] is not None
        assert scores["pooled_control_endpoint_mse"] is not None
    affected = [json.loads(row)["score"] for row in raw if json.loads(row)["row_id"] == "guide:0"]
    assert len(affected) == 24
    assert all(row["unavailable"]["state_energy"] == "candidate_missing_prediction" for row in affected)


def test_truth_content_mutation_rejected_before_rows(fixture):
    values = fixture.truth.population.mean_compositions
    values.setflags(write=True); values[0, 0] += .001; values.setflags(write=False)
    rows = []
    with pytest.raises(ValueError, match="population content identity changed"):
        score(fixture, row_sink=rows.append)
    assert rows == []
