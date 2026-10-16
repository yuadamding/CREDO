"""Fabricated CPU rollout -> bounded publication -> exact readback, no updates."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from frontier.forecast import ForecastEmission, forecast_population
from frontier.forecast_storage import (ForecastReader, ForecastStorageLimits,
                                      ForecastStorageSpec, ForecastWriter)
from frontier.selector_contract import Recipe
from test_forecast import binding, context, model, source, limits


@pytest.fixture(autouse=True)
def no_steps(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_a, **_k: pytest.fail("optimizer forbidden"))
    yield
    torch.set_num_threads(previous)


def storage_limits():
    return ForecastStorageLimits(output_cap_bytes=128*1024**2,
        working_cap_bytes=256*1024**2, maximum_batch_laws=2)


def core_limits(**changes):
    return limits(**(dict(maximum_host_bytes=512*1024**2,
        maximum_output_bytes=128*1024**2) | changes))


@pytest.mark.parametrize("family", ["F0", "F2", "F3", "F4"])
def test_streaming_and_in_memory_laws_catalogs_are_exact(tmp_path, family):
    m, s = model(family), source()
    b, c, recipe = binding(m, s), context(m, s), Recipe(m.config.configuration_id, .25, 1.3)
    prefixes = (5, 8)
    expected = forecast_population(m, s, b, c, recipe, prefixes=prefixes, limits=core_limits())
    spec = ForecastStorageSpec.from_source(s, b, c, recipe, prefixes=prefixes)
    destination = tmp_path / "forecast"
    with ForecastWriter(destination, spec=spec, producer_sha256="a"*64, limits=storage_limits()) as writer:
        with patch.object(m, "rollout", wraps=m.rollout) as rollout:
            result = forecast_population(m, s, b, c, recipe, prefixes=prefixes,
                                         limits=core_limits(), sink=writer)
        assert rollout.call_count == 3  # two guide batches and one NTC pool
        assert isinstance(result, ForecastEmission) and not hasattr(result, "prefixes")
        assert not (destination / "COMPLETE.json").exists()
        assert result.receipt["law_storage"] == "streamed_to_explicit_sink"
        assert result.receipt["complete_prediction_checks_passed"]
        assert result.receipt["resources"]["streamed_output"]
        publication = writer.finish(result.receipt)
    with ForecastReader(destination, spec=spec,
            expected_complete_sha256=publication["complete_sha256"],
            expected_manifest_sha256=publication["manifest_sha256"],
            maximum_output_bytes=storage_limits().output_cap_bytes,
            maximum_working_bytes=storage_limits().working_cap_bytes) as reader:
        for p in prefixes:
            reference = expected.prefixes[p]
            for index in (*reference.guide_laws, "pooled_control"):
                law = reader.law(p, index)
                wanted = reference.pooled_control if index == "pooled_control" else reference.guide_laws[index]
                for field in ("uncalibrated_states", "states", "log_weights"):
                    assert torch.equal(getattr(law, field), getattr(wanted, field))
                assert law.log_mass_increment == wanted.log_mass_increment
            for index in reference.unavailable_guides:
                assert reader.law(p, index) is None
            for key, value in reader.catalog(p).items():
                assert torch.equal(value, getattr(reference, key))


def test_streaming_model_change_never_returns_success_or_completes(tmp_path):
    m, s = model(), source()
    b, c, recipe = binding(m, s), context(m, s), Recipe(m.config.configuration_id)
    spec = ForecastStorageSpec.from_source(s, b, c, recipe, prefixes=(5, 8))
    destination = tmp_path / "forecast"
    original = m.rollout
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        m.reference_embedding.add_(.01)
        return result
    with pytest.raises(ValueError, match="model or global RNG"):
        with ForecastWriter(destination, spec=spec, producer_sha256="a"*64, limits=storage_limits()) as writer:
            with patch.object(m, "rollout", side_effect=changed):
                forecast_population(m, s, b, c, recipe, prefixes=(5, 8), limits=core_limits(), sink=writer)
    assert not (destination / "COMPLETE.json").exists()


@pytest.mark.parametrize("mismatch", ["recipe", "host_cap", "microbatch"])
def test_streaming_preflight_rejects_before_model_rollout(tmp_path, mismatch):
    m, s = model(), source()
    b, c, recipe = binding(m, s), context(m, s), Recipe(m.config.configuration_id)
    spec = ForecastStorageSpec.from_source(s, b, c, recipe, prefixes=(5, 8))
    destination = tmp_path / "forecast"
    chosen_recipe = replace(recipe, alpha=.25) if mismatch == "recipe" else recipe
    resource = core_limits(maximum_host_bytes=1) if mismatch == "host_cap" else core_limits()
    if mismatch == "microbatch": resource = replace(resource, microbatch_laws=3)
    with pytest.raises(ValueError):
        with ForecastWriter(destination, spec=spec, producer_sha256="a"*64, limits=storage_limits()) as writer:
            with patch.object(m, "rollout", side_effect=AssertionError("must fail before inference")):
                forecast_population(m, s, b, c, chosen_recipe, prefixes=(5, 8), limits=resource, sink=writer)
    assert not (destination / "COMPLETE.json").exists()


def test_invalid_sink_is_rejected_without_inference():
    m, s = model(), source()
    with patch.object(m, "rollout", side_effect=AssertionError("must fail before inference")):
        with pytest.raises(TypeError, match="ForecastWriter"):
            forecast_population(m, s, binding(m, s), context(m, s), Recipe(m.config.configuration_id),
                                prefixes=(5, 8), limits=core_limits(), sink=object())


def test_streamed_catalog_publication_and_readback_ignore_global_default_device(tmp_path):
    m, s = model("F2"), source()
    b, c, recipe = binding(m, s), context(m, s), Recipe(m.config.configuration_id, .25, 1.15)
    spec = ForecastStorageSpec.from_source(s, b, c, recipe, prefixes=(5, 8))
    destination = tmp_path / "forecast"
    with torch.device("meta"):
        with ForecastWriter(destination, spec=spec, producer_sha256="a"*64, limits=storage_limits()) as writer:
            result = forecast_population(m, s, b, c, recipe, prefixes=(5, 8), limits=core_limits(), sink=writer)
            publication = writer.finish(result.receipt)
        with ForecastReader(destination, spec=spec,
                expected_complete_sha256=publication["complete_sha256"],
                expected_manifest_sha256=publication["manifest_sha256"],
                maximum_output_bytes=storage_limits().output_cap_bytes,
                maximum_working_bytes=storage_limits().working_cap_bytes) as reader:
            assert all(v.device.type == "cpu" for v in reader.catalog(5).values())
            assert reader.law(5, 0).states.device.type == "cpu"
