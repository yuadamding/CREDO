"""Actual pretruth seals -> fixed-total ensemble scoring, fabricated CPU only."""

from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontier import calibration_view, ensemble_view
from frontier.assessment import score_calibration_bank, score_ensemble_bank
from frontier.assessment_support import build_assessment_support
from frontier.calibration_view import CalibrationView
from frontier.ensemble_view import EnsembleView
from frontier.evaluation import (ScoreIdentity, mean_seed_scores, reduce_scores,
    score_catalog, score_joint, score_molecular_law, score_state_law)
from frontier.forecast_seal import (ForecastSealPlan, freeze_calibration_derivation,
    freeze_ensemble_derivation, publish_forecast_seal)
from frontier.forecast_storage import ForecastReader
from frontier.native_inputs import FoldInputs
from frontier.planner import ScoreTask, TrajectoryKey
from frontier.query_compiler import compile_query_source
from frontier.selection import CHANNELS
from frontier.selector_contract import OuterRecipe
from frontier.training_owner import observation_state_digest
from frontier.truth_compiler import compile_evaluation_truth
from test_assessment import make_bank, limits, law_tensor, literal_decode
from test_assessment_support import package as native_package


MIB = 1024**2


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        monkeypatch.setattr(cls, "step", lambda *_a, **_k: pytest.fail("optimizer forbidden"))
    yield
    torch.set_num_threads(original)


@contextmanager
def package(tmp_path, *, seal_ensemble=True, missing_control=False):
    kwargs = dict(endpoint_positive=[2,1,1,0,0,0,1,0]) if missing_control else {}
    data_root, index, args = native_package(tmp_path, **kwargs)
    args["fold"] = FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4"))
    observation = args["observation"]
    observation.fold = replace(observation.fold, fold_id=args["fold"].fold_id,
        training_donors=args["fold"].training_donors)
    args["observation_binding"] = replace(args["observation_binding"], state_sha256=observation_state_digest(observation))
    query = index.open(data_root, args["fold"], "query", query_donor="D1", max_batch_rows=3)
    source = compile_query_source(query, **args).source
    root = tmp_path/"banks"; root.mkdir()
    frozen = OuterRecipe("D1", source.fold.training_donors, "F0-V0", .25, 1.3, "1"*64, "2"*64, 1.)
    with ExitStack() as stack:
        entries, readers, views = [], [], []
        for seed in range(3):
            entry = make_bank(root/f"seed-{seed}", source, seed=seed, offset=.4*seed, ensemble_totals=(12,13))
            reader = stack.enter_context(ForecastReader(root/entry.bank_relative_root, spec=entry.spec,
                expected_complete_sha256=entry.expected_complete_sha256,
                expected_manifest_sha256=entry.expected_manifest_sha256,
                maximum_output_bytes=128*MIB, maximum_working_bytes=128*MIB))
            task = ScoreTask(TrajectoryKey(source.fold.training_donors, "F0-V0", seed), "D1", "D1", 9216, "outer")
            view = CalibrationView.for_outer(reader, frozen=frozen, task=task,
                adapter_source_sha256=hashlib.sha256(Path(calibration_view.__file__).read_bytes()).hexdigest())
            entries.append(replace(entry, derivation_manifest=freeze_calibration_derivation(view)))
            readers.append(reader); views.append(view)
        ensemble = EnsembleView.bind(tuple(views), totals=(12,13), maximum_working_bytes=128*MIB,
            adapter_source_sha256=hashlib.sha256(Path(ensemble_view.__file__).read_bytes()).hexdigest())
        plan = ForecastSealPlan(source.fold, "D1", source.reference.source_access_sha256, tuple(entries),
            ensemble_derivations=(freeze_ensemble_derivation(ensemble),) if seal_ensemble else ())
        release = publish_forecast_seal(root, "SEAL.json", plan=plan, expectation=plan.expectation,
            readers=tuple(readers), native_index=index, output_cap_bytes=256*1024,
            derivations=tuple(views), ensembles=(ensemble,) if seal_ensemble else ())
        endpoint = index.open(data_root, source.fold, "evaluation_truth", query_donor="D1", release=release, max_batch_rows=3)
        truth = compile_evaluation_truth(endpoint, **(args | dict(source_id="D1_Stim48hr", release=release)))
        yield SimpleNamespace(source=source, truth=truth, observation=observation, plan=plan,
            ensemble=ensemble, views=tuple(views), readers=tuple(readers), root=root)


def score(p, **changes):
    rows = []
    kwargs = dict(plan=p.plan, source=p.source, truth=p.truth, observation=p.observation,
                  total=13, limits=limits(), row_sink=rows.append)
    kwargs.update(changes)
    result = score_ensemble_bank(p.ensemble, **kwargs)
    return result, [json.loads(row) for row in rows]


def test_all_nine_ensemble_channels_match_literal_fixed_total_mixture(tmp_path):
    with package(tmp_path) as p:
        result, rows = score(p)
        support = build_assessment_support(p.source, p.truth)
        identity = ScoreIdentity(**result["scores"][0]["scores"]["identity"])
        assert identity.seed is None and identity.predictor_kind == "three_seed_finite_measure_mixture"
        assert identity.particle_prefix == 13
        assert len(result["scores"]) == 1 and len(rows) == len(support.required_row_ids)
        catalog = p.ensemble.catalog(13)["log_frequencies"]
        counts = torch.from_numpy(p.truth.population.captured_counts.copy())
        control_observed = torch.from_numpy(p.truth.population.mean_compositions[3].copy())
        control_predicted = literal_decode(p.observation, p.ensemble.law(13, "pooled_control"))
        expected = {"catalog": score_catalog(catalog, counts, expected_catalog_size=8, identity=identity),
                    "pooled_control": score_molecular_law(None, None, control_predicted, control_observed, identity=identity)}
        for i in range(4):
            law = p.ensemble.law(13, i)
            assert len(law.states) == 13
            y, v = law_tensor(p.truth.population, i)
            s, u = law_tensor(p.source, i)
            row = score_state_law(law.states, law.log_weights.exp(), y, v, s, u, identity=identity, block=2)
            row.update(score_joint(law.states, law.log_weights.exp(), float(catalog[i].exp()),
                                  y, v, int(counts[i])/int(counts.sum()), identity=identity))
            if not p.source.catalog[i].is_control:
                row.update(score_molecular_law(literal_decode(p.observation, law),
                    torch.from_numpy(p.truth.population.mean_compositions[i].copy()),
                    control_predicted, control_observed, identity=identity))
            expected[f"guide:{i}"] = row
        literal = reduce_scores(expected, support.weights, identity=identity)
        actual = result["scores"][0]["scores"]
        for channel in CHANNELS:
            # Reuse the established single-seed eager-vs-gene-block FP32
            # decoder tolerance; geometry/catalog/joint remain FP64 checks.
            molecular = channel in ("target_endpoint_mse", "pooled_control_endpoint_mse", "pooled_effect_mse")
            assert actual[channel] == pytest.approx(literal[channel],
                rel=3e-6 if molecular else 1e-10, abs=3e-6 if molecular else 1e-12)
        receipt = result["receipt"]
        assert receipt["parent_publications"] == [dict(r.publication_reference) for r in p.readers]
        assert receipt["estimate"]["mapped_bank_residency_charge_bytes"] == sum(r.publication_reference["total_bytes"] for r in p.readers)
        assert receipt["estimate"]["additional_ensemble_view_host_estimate_bytes"] > 0
        assert receipt["state_score_calls"] == receipt["joint_score_calls"] == receipt["decoder_calls"] == 4
        assert receipt["neural_forecast_calls"] == receipt["optimizer_calls"] == 0
        assert receipt["ensemble_model_identity"]["components"] == [
            dict(seed=i, model_sha256=p.readers[i].spec.document["model_binding"]["model_artifact_sha256"]) for i in range(3)]


def test_seed_scores_can_be_averaged_but_ensemble_is_distinct(tmp_path):
    with package(tmp_path) as p:
        records, identities, derivations = {}, {}, []
        for seed, view in enumerate(p.views):
            result = score_calibration_bank(view, plan=p.plan, source=p.source, truth=p.truth,
                observation=p.observation, prefix=13, limits=limits(), row_sink=lambda _raw: None)
            records[seed] = result["scores"][0]["scores"]
            identities[seed] = ScoreIdentity(**records[seed]["identity"])
            derivations.extend(result["receipt"]["prediction_derivations"].values())
        assert len({identity.recipe_id for identity in identities.values()}) == 1
        assert len(set(derivations)) == 3
        mean = mean_seed_scores(records, identities=identities)
        for channel in CHANNELS:
            assert mean[channel] == pytest.approx(sum(record[channel] for record in records.values())/3)
        ensemble, _ = score(p)
        e = ensemble["scores"][0]["scores"]
        assert abs(e["state_energy"]-mean["state_energy"]) > 1e-8
        with pytest.raises(ValueError, match="single-seed"):
            mean_seed_scores({0:e}, identities={0:ScoreIdentity(**e["identity"])})


def test_component_seals_alone_cannot_release_new_ensemble_prediction(tmp_path):
    with package(tmp_path, seal_ensemble=False) as p:
        with pytest.raises(ValueError, match="ensemble derivation"):
            score(p)


def test_unregistered_total_is_not_inferred_from_parent_bank(tmp_path):
    with package(tmp_path) as p:
        with pytest.raises(ValueError, match="prefix"):
            score(p, total=7)


def test_missing_control_preserves_target_endpoints_and_full_catalog(tmp_path):
    with package(tmp_path, missing_control=True) as p:
        result, rows = score(p)
        reduction = result["scores"][0]["scores"]
        assert reduction["target_endpoint_mse"] is not None
        assert reduction["pooled_control_endpoint_mse"] is None
        assert reduction["pooled_effect_mse"] is None
        assert reduction["count_ce_minus_entropy"] is not None
        assert next(row for row in rows if row["row_id"] == "catalog")["score"]["catalog_size"] == 8


def test_parent_mutation_stops_completion(tmp_path):
    with package(tmp_path) as p:
        p.readers[1].close()
        with pytest.raises(RuntimeError, match="context"):
            score(p)


def test_ensemble_memory_admission_charges_all_three_parent_banks(tmp_path):
    with package(tmp_path) as p:
        result, _ = score(p)
        estimate = result["receipt"]["estimate"]["host_logical_upper_estimate_bytes"]
        with pytest.raises(ValueError, match="preflight"):
            score(p, limits=limits(maximum_host_bytes=estimate-1))
