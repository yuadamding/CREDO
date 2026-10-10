"""Synthetic numerical-shape and failure-report tests; no real-data access."""
# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]

import numpy as np
import pandas as pd
import pytest
import torch

from credo.contracts import Axis, FiniteMeasure
from credo.recipes.compact_sde_v3.model import CREDOModel
import d2_forecast as module
import full_forecast
from full_forecast import ForecastCatalog, ForecastSources, GuideForecast
from native_study import Guide, series_id


def synthetic(kind, count=17):
    torch.set_num_threads(1)
    rng = np.random.default_rng(71)
    guides = tuple(Guide(f"guide-{i}", "__NTC__" if i in (0, 7) else f"target-{i}", i in (0, 7))
                   for i in range(count))
    catalog = ForecastCatalog("D2", guides, tuple(31+i for i in range(count)), "a"*64, "b"*64, "c"*64)
    laws, rows = {}, []
    for i, guide in enumerate(guides):
        key = series_id("D2", i)
        support = rng.normal(size=(5+i % 4, 48))
        weights = rng.uniform(.1, 1, len(support))
        weights /= weights.sum()
        mass = float(catalog.source_frequency[i])
        laws[key] = FiniteMeasure(support, weights*mass, mass)
        rows.append(dict(measure_id=key, sample_id="D2", guide_id=guide.guide_id,
                         is_control=guide.is_control, context_group_id="captured-aggregate::D2",
                         embedding_id="shared-NTC-reference" if guide.is_control else f"target::{guide.target_id}"))
    frame = pd.DataFrame(rows)
    sources = ForecastSources.from_compiled(SimpleNamespace(
        axis=Axis("physical", ("Stim8hr", "Stim48hr"), (8., 48.), "Stim8hr"),
        measures={"Stim8hr": laws}, measure_meta=frame, latent_dim=48), catalog)
    torch.manual_seed(93)
    model = CREDOModel(embedding_ids=tuple(dict.fromkeys(frame.embedding_id)),
        control_embedding_ids=("shared-NTC-reference",), latent_dim=48, hidden_dim=32,
        embedding_dim=8, n_programs=8, context_mode="none", trunk_kind=kind, trunk_depth=3)
    model.requires_grad_(False)
    return sources, model


@pytest.mark.parametrize("kind", ["mlp", "residual_mlp"])
@pytest.mark.parametrize("chunk", [1, 2, 16])
def test_fixed_architecture_shape_matches_original_and_preserves_parameters(kind, chunk, monkeypatch):
    sources, model = synthetic(kind)
    saved = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    expected = list(full_forecast.iter_forecasts(model, sources, prefixes=(32, 64), steps_per_interval=8, seed=2))
    sizes = []
    original = module.tensor_rollout
    def tracked(model, dynamics, initial, grid, noise, indices):
        sizes.append(initial.z.shape[0])
        return original(model, dynamics, initial, grid, noise, indices)
    monkeypatch.setattr(module, "tensor_rollout", tracked)
    actual = list(module.iter_batched_forecasts(model, sources, prefixes=(32, 64),
        steps_per_interval=8, seed=2, batch_guides=chunk))
    assert max(sizes) == (1 if kind == "residual_mlp" else chunk)
    assert sum(sizes) == len(sources.measure_ids)
    for a, b in zip(expected, actual, strict=True):
        assert (a.measure_id, a.guide_index, a.seed) == (b.measure_id, b.guide_index, b.seed)
        for field in module._AGREEMENT_FIELDS:
            np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=2e-5, rtol=1e-5)
        if kind == "residual_mlp":
            np.testing.assert_array_equal(a.state, b.state)
    assert model.training
    assert all(torch.equal(value, saved[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_forecasts_are_identical_across_order_and_source_chunk():
    sources, model = synthetic("residual_mlp")
    normal = {r.measure_id: r for r in module.iter_batched_forecasts(model, sources,
        prefixes=(16, 32), steps_per_interval=8, seed=5, batch_guides=16)}
    reversed_sources = replace(sources, measure_ids=sources.measure_ids[::-1], guide_indices=sources.guide_indices[::-1])
    reverse = {r.measure_id: r for r in module.iter_batched_forecasts(model, reversed_sources,
        prefixes=(16, 32), steps_per_interval=8, seed=5, batch_guides=2)}
    for key, row in normal.items():
        for field in module._AGREEMENT_FIELDS:
            np.testing.assert_array_equal(getattr(row, field), getattr(reverse[key], field))


@pytest.mark.parametrize("kind", ["mlp", "residual_mlp"])
def test_legacy_diagnostic_route_is_explicit_and_preserves_mlp_definition(kind, monkeypatch):
    sources, model = synthetic(kind)
    shapes = []
    original = module.tensor_rollout
    def tracked(model, dynamics, initial, grid, noise, indices):
        shapes.append(initial.z.shape[0])
        return original(model, dynamics, initial, grid, noise, indices)
    monkeypatch.setattr(module, "tensor_rollout", tracked)
    legacy = list(module.iter_batched_forecasts(model, sources, prefixes=(16, 32),
        steps_per_interval=8, seed=1, diagnostic_legacy16=True))
    assert shapes == [16, 1]
    actual = list(module.iter_batched_forecasts(model, sources, prefixes=(16, 32), steps_per_interval=8, seed=1))
    if kind == "mlp":
        for a, b in zip(legacy, actual, strict=True):
            for field in module._AGREEMENT_FIELDS:
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    policy = module.forecast_numerical_policy(model, diagnostic_legacy16=True)
    assert policy["diagnostic_only"] and policy["dynamics_maximum_guides"] == 16
    with pytest.raises(ValueError, match="exact 16-guide"):
        list(module.iter_batched_forecasts(model, sources, prefixes=(16, 32),
             batch_guides=1, diagnostic_legacy16=True))


def row(index):
    return GuideForecast(f"synthetic-{index}", index, 3, 8,
        np.zeros((2, 2)), np.zeros(2), np.zeros(2), np.zeros((2, 3)), np.ones(2), np.ones(2)/2)


def closed_stream(values, marker, name):
    try:
        yield from values
    finally:
        marker.append(name)


def test_full_diagnostics_collect_later_guides_and_fields_before_failure():
    expected = [row(0), row(1)]
    actual = [replace(row(0), state=np.full((2, 2), 1e-3)),
              replace(row(1), conditional_features=np.full((2, 3), 2e-3))]
    closed = []
    report = module._compare_forecast_streams(closed_stream(expected, closed, "reference"),
        closed_stream(actual, closed, "candidate"), expected_guides=2)
    assert sorted(closed) == ["candidate", "reference"]
    assert not report["agreement_passed"] and report["compared_guides"] == 2
    assert report["fields"]["state"]["violation_count"] == 4
    field = report["fields"]["conditional_features"]
    assert field["violation_count"] == 6
    assert field["worst_tolerance_error"]["measure_id"] == "synthetic-1"
    assert field["worst_tolerance_error"]["guide_index"] == 1
    assert field["worst_tolerance_error"]["coordinate"] == [0, 0]
    assert field["worst_tolerance_error"]["reference_value"] == 0
    assert field["worst_tolerance_error"]["candidate_value"] == .002
    assert field["maximum_normalized_tolerance_ratio"] == pytest.approx(.002/(2e-5+1e-5*.002))
    assert report["fields"]["log_weight"]["passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", ["identity", "shape", "nonfinite", "empty", "missing", "extra"])
def test_invalid_numerical_or_inventory_results_cannot_pass_and_close(bad):
    expected, actual = [row(0)], [row(0)]
    if bad == "identity": actual[0] = replace(row(0), measure_id="different")
    if bad == "shape": actual[0] = replace(row(0), state=np.zeros((1, 2)))
    if bad == "nonfinite": actual[0] = replace(row(0), state=np.full((2, 2), np.inf))
    if bad == "empty": actual[0] = replace(row(0), state=np.empty((0, 2)))
    if bad == "missing": actual = []
    if bad == "extra": actual.append(row(1))
    closed = []
    report = module._compare_forecast_streams(closed_stream(expected, closed, "reference"),
        closed_stream(actual, closed, "candidate"), expected_guides=1)
    assert not report["agreement_passed"]
    assert sorted(closed) == ["candidate", "reference"]
    json.dumps(report, allow_nan=False)


def test_both_streams_close_when_candidate_raises():
    closed = []
    def broken():
        try:
            yield row(0)
            raise RuntimeError("synthetic read failure")
        finally:
            closed.append("candidate")
    with pytest.raises(RuntimeError, match="synthetic read"):
        module._compare_forecast_streams(closed_stream([row(0), row(1)], closed, "reference"),
                                        broken(), expected_guides=2)
    assert sorted(closed) == ["candidate", "reference"]


def test_fixed_gate_keeps_original_direction_and_tolerances():
    # Use a large reference so reversing np.isclose's tolerance scale matters.
    a, b = np.array([100.]), np.array([100.001020005])
    record = module._field_agreement(a, b, measure_id="s", guide_index=0)
    assert record["passed"] == bool(np.allclose(a, b, atol=2e-5, rtol=1e-5))
    assert module.FORECAST_ATOL == 2e-5 and module.FORECAST_RTOL == 1e-5


@pytest.mark.parametrize("legacy, mismatch", [(False, False), (False, True), (True, False), (True, True)])
def test_mocked_qualification_gate_and_diagnostic_status_not_gpu_evidence(legacy, mismatch, monkeypatch):
    # Control-flow-only mocked device. No CUDA execution/qualification claimed.
    class FakeModel:
        trunk_kind = "residual_mlp"
        training = True
        def parameters(self):
            yield SimpleNamespace(device=torch.device("cuda"))
        def train(self, mode):
            self.training = mode
    model = FakeModel()
    sources = SimpleNamespace()  # dataclass replacement is patched below only for this control-flow unit test
    sources.measure_ids = tuple(f"s{i}" for i in range(32))
    sources.guide_indices = tuple(range(32))
    closed, calls = [], []
    def subset(source, **changes):
        assert source is sources
        assert len(changes["measure_ids"]) == len(changes["guide_indices"]) == 16
        return SimpleNamespace(**changes)
    monkeypatch.setattr(module, "replace", subset)
    def reference(m, source, **kwargs):
        calls.append(("reference", kwargs))
        assert m is model and len(source.measure_ids) == 16
        return closed_stream([row(i) for i in range(16)], closed, "reference")
    def production(m, source, **kwargs):
        calls.append(("candidate", kwargs))
        values = [row(i) for i in range(16)]
        if mismatch:
            values[-1] = replace(values[-1], state=np.ones((2, 2)))
        return closed_stream(values, closed, "candidate")
    monkeypatch.setattr(full_forecast, "iter_forecasts", reference)
    monkeypatch.setattr(module, "iter_batched_forecasts", production)
    if mismatch and not legacy:
        with pytest.raises(module.ForecastAgreementError) as error:
            module.qualify_forecast(model, sources, seed=3, device="cuda", diagnostic_legacy16=legacy)
        report = error.value.report
        assert report["status"] == "failed" and not report["production_qualified"]
        assert report["fields"]["state"]["worst_tolerance_error"]["guide_index"] == 15
    else:
        report = module.qualify_forecast(model, sources, seed=3, device="cuda", diagnostic_legacy16=legacy)
        assert report["status"] == ("diagnostic_only" if legacy else "passed")
        assert report["production_qualified"] == (not legacy)
    assert report["agreement_passed"] == (not mismatch)
    assert len(report["guide_errors"]) == 16
    assert report["atol"] == 2e-5 and report["rtol"] == 1e-5
    assert model.training
    assert sorted(closed) == ["candidate", "reference"]
    assert calls[0][1] == dict(prefixes=(4096, 8192), steps_per_interval=8, seed=3)
    assert calls[1][1] == dict(prefixes=(4096, 8192), steps_per_interval=8, seed=3, diagnostic_legacy16=legacy)
    json.dumps(report, allow_nan=False)


def test_unknown_architecture_fails_closed():
    with pytest.raises(ValueError, match="Unbound"):
        module.forecast_numerical_policy(SimpleNamespace(trunk_kind="unregistered"))
