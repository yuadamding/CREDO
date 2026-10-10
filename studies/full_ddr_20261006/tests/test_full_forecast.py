"""Synthetic source-only DDR prediction, exact catalog and seal checks."""
# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import h5py
import numpy as np
import pytest
import torch

from credo.recipes.compact_sde_v3.recipe import recipe
from credo.recipes.compact_sde_v3.particles import ParticleRollout
from encode_sources import publish_json, sha_file
from full_forecast import (
    ForecastCatalog, ForecastSources, full_catalog_probability, iter_forecasts,
    measure_seed, terminal_summary, write_bank, seal_bank,
)
from memory_qualification import qualification_config
from native_study import build_native_study, compile_development
from test_native_study import fixture, config


def setup():
    values = fixture()
    bundle = build_native_study(**values)
    problem, _ = compile_development(bundle, config())
    observation = next(v for v in values["observations"] if v.source_id == "D2_Stim8hr")
    catalog = ForecastCatalog("D2", values["guides"], observation.counts, "a" * 64, "b" * 64, "c" * 64)
    sources = ForecastSources.from_compiled(problem.validation, catalog)
    torch.manual_seed(2)
    model = recipe.build_model(problem, qualification_config({"hidden_dim": 8, "embedding_dim": 2}))
    return problem, sources, model


def rows(sources, model):
    return list(iter_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2, seed=7))


def test_prediction_is_source_only_and_changes_no_parameters():
    problem, sources, model = setup()
    class NoEndpoint:
        def __getitem__(self, key):
            assert key == "Stim8hr", "Forecast opened endpoint observations"
            return problem.validation.measures[key]
    safe = SimpleNamespace(axis=problem.validation.axis, measures=NoEndpoint(),
                           measure_meta=problem.validation.measure_meta, latent_dim=2)
    sources = ForecastSources.from_compiled(safe, sources.catalog)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = rows(sources, model)
    assert len(result) == len(sources.measure_ids)
    assert model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all(row.state.shape == (4, 2) for row in result)
    assert all(row.conditional_features.shape == (2, 3) for row in result)


def test_exact_complete_catalog_preserves_zeros_and_complement_ratios():
    _, sources, model = setup()
    result = rows(sources, model)
    increments = np.stack([row.mass_increment for row in result], axis=1)
    logp, probability = full_catalog_probability(sources.catalog, sources.guide_indices, increments)
    assert probability.shape == (2, 5)
    assert (probability > 0).all()
    assert np.allclose(probability.sum(axis=1), 1, atol=1e-14, rtol=0)
    assert np.allclose(np.exp(logp), probability)
    complement = sorted(set(range(5)) - set(sources.guide_indices))
    # Includes NEW, which has RNA but no fitted target embedding, and raw zero guide2.
    assert 4 in complement and sources.catalog.counts[2] == 0
    assert np.allclose(probability[:, complement[0]] / probability[:, complement[1]],
                       sources.catalog.source_frequency[complement[0]] / sources.catalog.source_frequency[complement[1]])
    expected = sources.catalog.source_frequency[None].repeat(2, axis=0)
    expected[:, sources.guide_indices] *= np.exp(increments)
    expected /= expected.sum(axis=1, keepdims=True)
    assert np.allclose(probability, expected, atol=1e-14, rtol=0)


def test_prefix_mass_and_state_weights_are_recomputed_not_truncated_normalizations():
    initial = torch.full((1, 4), -np.log(4), dtype=torch.float64)
    terminal = initial + torch.tensor([[0., 0., 3., 3.]])
    states = torch.tensor([[[0., 1.], [1., 0.], [2., 1.], [1., 2.]]])
    result = ParticleRollout(torch.stack((states, states)), torch.stack((initial, terminal)),
        torch.tensor([-5.]), torch.tensor([0., 1.]), ("m",), ("e",), ("c",),
        torch.tensor([0]), torch.ones(1), torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2),
        torch.zeros(1, 1, 4), torch.zeros(1, 1, 1), torch.zeros(1, 1, 4, 2))
    row = terminal_summary(result, (2, 4), 1)
    assert row.mass_increment[0] == pytest.approx(0, abs=1e-14)
    assert row.mass_increment[1] == pytest.approx(np.log((1 + np.exp(3)) / 2))
    assert np.allclose(row.conditional_features[0, 1:], np.tanh(states.numpy()[0, :2]).mean(axis=0))
    assert row.ess_fraction[0] == pytest.approx(1)
    assert row.max_weight_fraction[0] == pytest.approx(0.5)
    # A truncation of four-particle normalized weights would sum to <1 here.
    assert row.conditional_features[0, 0] == pytest.approx(1)


def test_random_streams_and_predictions_do_not_depend_on_panel_iteration_order():
    _, sources, model = setup()
    assert measure_seed(7, sources.measure_ids[0], "source") != measure_seed(7, sources.measure_ids[0], "noise")
    baseline = {r.measure_id: r for r in rows(sources, model)}
    reversed_sources = replace(sources, measure_ids=sources.measure_ids[::-1], guide_indices=sources.guide_indices[::-1])
    reordered = {r.measure_id: r for r in rows(reversed_sources, model)}
    for key in baseline:
        assert np.array_equal(baseline[key].state, reordered[key].state)
        assert np.array_equal(baseline[key].log_weight, reordered[key].log_weight)


@pytest.mark.parametrize("change", ["donor", "guide", "embedding", "mass"])
def test_source_substitution_is_rejected(change):
    problem, sources, _ = setup()
    frame = problem.validation.measure_meta.copy()
    measures = dict(problem.validation.measures["Stim8hr"])
    if change == "donor":
        frame.loc[0, "sample_id"] = "D3"
    elif change == "guide":
        frame.loc[0, "guide_id"] = "a-1" if frame.loc[0, "guide_id"] != "a-1" else "NTC-1"
    elif change == "embedding":
        frame.loc[0, "embedding_id"] = "target::B"
    else:
        from credo.contracts import FiniteMeasure
        key = next(iter(measures))
        law = measures[key]
        measures[key] = FiniteMeasure(law.support, law.weights * 2, law.total_mass * 2)
    altered = SimpleNamespace(axis=problem.validation.axis, measure_meta=frame,
        measures={"Stim8hr": measures}, latent_dim=2)
    with pytest.raises(ValueError):
        ForecastSources.from_compiled(altered, sources.catalog)


@pytest.mark.parametrize("invalid", [(-1,), (0, 0), (5,), (True,)])
def test_bad_modeled_catalog_index_rejected(invalid):
    _, sources, _ = setup()
    with pytest.raises(ValueError):
        full_catalog_probability(sources.catalog, invalid, np.zeros((1, len(invalid))))


def test_unfitted_target_or_ecology_cannot_silently_fallback():
    _, sources, model = setup()
    missing = sources.measure_meta.copy()
    missing.loc[0, "embedding_id"] = "target::NEW"
    with pytest.raises(ValueError, match="Unfitted"):
        list(iter_forecasts(model, replace(sources, measure_meta=missing), prefixes=(2, 4)))
    model.context_enabled = True
    with pytest.raises(ValueError, match="forbids ecological"):
        list(iter_forecasts(model, sources, prefixes=(2, 4)))


def test_nonfinite_and_underflow_catalog_values_are_failures_not_fabricated_zeros():
    _, sources, _ = setup()
    for value in (np.nan, np.inf, 1000.):
        with pytest.raises(ValueError):
            full_catalog_probability(sources.catalog, (0,), np.array([[value]]))


def test_full_26504_category_synthetic_catalog_keeps_all_categories():
    from native_study import Guide
    guides = tuple(Guide(f"guide-{i}", "NTC" if i == 0 else f"target-{i}", i == 0) for i in range(26504))
    counts = tuple(0 if i % 3 else i + 1 for i in range(26504))
    catalog = ForecastCatalog("D2", guides, counts, "a" * 64, "b" * 64, "c" * 64)
    _, probability = full_catalog_probability(catalog, (0, 26503), np.array([[0., 0.], [1., -1.]]))
    assert probability.shape == (2, 26504)
    assert np.allclose(probability[0], catalog.source_frequency, rtol=0, atol=1e-16)
    assert (probability[:, np.asarray(counts) == 0] > 0).all()


def test_changed_frozen_parameters_fail_at_end_of_prediction():
    _, sources, model = setup()
    stream = iter_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2)
    next(stream)
    with torch.no_grad():
        model.reference_embedding.add_(1)
    with pytest.raises(RuntimeError, match="changed frozen"):
        list(stream)


def bank(tmp_path):
    _, sources, model = setup()
    root = tmp_path / "bank"
    payload = write_bank(root, sources, iter_forecasts(model, sources, prefixes=(2, 4), steps_per_interval=2, seed=7),
        prefixes=(2, 4), checkpoint_sha256="d" * 64, implementation_sha256="e" * 64,
        seed=7, steps_per_interval=2, maximum_output_bytes=8 * 1024**2)
    return root, payload


def check_receipt(tmp_path, root, payload, **changes):
    proof = dict(schema="credo.all4_full_ddr.forecast_input_check.v1", bank_sha256=sha_file(root / "BANK.json"),
                 inputs_verified_after_prediction=True, parameters_unchanged=True, parameter_updates=0,
                 endpoint_observations_read=False)
    proof.update({k: payload[k] for k in ("source_artifact_sha256", "support_sha256", "representation_sha256", "checkpoint_sha256")})
    proof.update(changes)
    path = tmp_path / "post-prediction-input-check.json"
    publish_json(path, proof)
    return path, sha_file(path)


def test_arrays_require_separate_post_prediction_check_before_seal(tmp_path):
    root, payload = bank(tmp_path)
    assert not (root / "COMPLETE.json").exists()
    assert payload["status"] == "arrays_written_not_sealed"
    assert payload["parameter_updates"] == 0 and not payload["endpoint_observations_read"]
    with h5py.File(root / "forecast.h5", "r") as arrays:
        assert arrays["catalog_probability"].shape == (2, 5)
        assert arrays["terminal_state"].shape[1:] == (4, 2)
        assert arrays["source_counts"][2] == 0
    path, digest = check_receipt(tmp_path, root, payload)
    assert seal_bank(root, input_check_receipt=path, input_check_sha256=digest)["status"] == "sealed_forecast_bank"
    assert json.loads((root / "COMPLETE.json").read_text())["bank_sha256"] == sha_file(root / "BANK.json")
    with pytest.raises(ValueError):
        seal_bank(root, input_check_receipt=path, input_check_sha256=digest)


@pytest.mark.parametrize("change", [dict(inputs_verified_after_prediction=False),
    dict(endpoint_observations_read=True), dict(parameter_updates=1), dict(parameter_updates=False),
    dict(support_sha256="f" * 64), dict(checkpoint_sha256="f" * 64)])
def test_failed_or_wrong_input_checks_never_seal(tmp_path, change):
    root, payload = bank(tmp_path)
    path, digest = check_receipt(tmp_path, root, payload, **change)
    with pytest.raises(ValueError):
        seal_bank(root, input_check_receipt=path, input_check_sha256=digest)
    assert not (root / "COMPLETE.json").exists()


def test_changed_forecast_arrays_reject_seal_and_are_preserved(tmp_path):
    root, payload = bank(tmp_path)
    path, digest = check_receipt(tmp_path, root, payload)
    with h5py.File(root / "forecast.h5", "r+") as arrays:
        arrays["source_counts"][0] = 99
    with pytest.raises(ValueError, match="changed"):
        seal_bank(root, input_check_receipt=path, input_check_sha256=digest)
    assert (root / "forecast.h5").exists() and not (root / "COMPLETE.json").exists()


def test_output_admission_precedes_any_prediction_or_new_directory(tmp_path):
    _, sources, _ = setup()
    def forbidden():
        raise AssertionError("Budget rejection executed numerical prediction")
        yield
    with pytest.raises(ValueError, match="output budget"):
        write_bank(tmp_path / "not-created", sources, forbidden(), prefixes=(4096, 8192),
            checkpoint_sha256="d" * 64, implementation_sha256="e" * 64, seed=0,
            steps_per_interval=8, maximum_output_bytes=100)
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("kind", ["missing", "extra", "relabel", "nonfinite", "seed", "steps"])
def test_invalid_row_inventory_never_publishes_bank(tmp_path, kind):
    _, sources, model = setup()
    results = rows(sources, model)
    if kind == "missing":
        results = results[:-1]
    elif kind == "extra":
        results.append(results[0])
    elif kind == "relabel":
        results[0] = replace(results[0], measure_id="D3::guide-index-0")
    elif kind == "seed":
        results[0] = replace(results[0], seed=99)
    elif kind == "steps":
        results[0] = replace(results[0], steps_per_interval=99)
    else:
        altered = copy.deepcopy(results[0])
        altered.state[0, 0] = np.nan
        results[0] = altered
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        write_bank(root, sources, iter(results), prefixes=(2, 4), checkpoint_sha256="d" * 64,
            implementation_sha256="e" * 64, seed=7, steps_per_interval=2, maximum_output_bytes=8 * 1024**2)
    assert not (root / "BANK.json").exists() and not (root / "COMPLETE.json").exists()
