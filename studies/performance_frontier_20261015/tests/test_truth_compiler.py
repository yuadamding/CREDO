"""Released fabricated endpoint populations; zero training or real-data reads."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from frontier.native_inputs import ForecastExpectation, VerifiedForecastRelease
from frontier.truth_compiler import CompiledTruth, compile_evaluation_truth
from frontier.training_owner import ObservationBinding, observation_state_digest
from test_query_compiler import fixture as query_fixture
from test_native_inputs import digest, publish_seal


def released(root, index, args, *, donor="D1", representation=None):
    expectation = ForecastExpectation(args["fold"].identity,
        index.access(args["fold"], "query", query_donor=donor).identity(),
        ("b"*64,), "c"*64, "d"*64,
        representation or args["observation_binding"].artifact_sha256, ("fabricated-bank",))
    publication, _ = publish_seal(root, expectation)
    return VerifiedForecastRelease.verify(publication, "seal.json",
        manifest_sha256=digest(publication / "seal.json"), expectation=expectation)


@pytest.fixture
def fixture(query_fixture):
    root, index, args = query_fixture
    release = released(root, index, args)
    args = args | dict(source_id="D1_Stim48hr", release=release)
    handle = index.open(root, args["fold"], "evaluation_truth", query_donor="D1", release=release,
                        max_batch_rows=3)
    return root, index, args, handle


def test_only_released_late_rows_all_counts_and_frozen_representation(fixture, monkeypatch):
    _, _, args, handle = fixture
    seen, read = [], handle.read_rows
    def record(addresses):
        seen.extend(addresses)
        return read(addresses)
    monkeypatch.setattr(handle, "read_rows", record)
    before = observation_state_digest(args["observation"])
    rng = torch.get_rng_state().clone()
    result = compile_evaluation_truth(handle, **args)
    assert seen == [("D1_Stim48hr", 0, i) for i in range(3)]
    assert result.population.captured_counts.tolist() == [2, 1, 0]
    assert result.population.positive_counts.tolist() == [1, 1, 0]
    assert result.population.states.shape == (2, 48)
    np.testing.assert_allclose(result.population.mean_compositions[0], [1/3, 0, 2/3])
    np.testing.assert_array_equal(result.population.mean_compositions[2], [0, 0, 0])
    assert result.receipt["compilation"]["schema"] == "credo.frontier.streaming-evaluation-truth.v1"
    assert result.receipt["compilation"]["access_role"] == "evaluation_truth"
    assert result.receipt["forecast_seal_sha256"] == args["release"].manifest_sha256
    assert result.receipt["input_lease_exit"]["status"] == "unchanged"
    assert result.receipt["optimizer_calls"] == 0
    assert observation_state_digest(args["observation"]) == before
    assert torch.equal(rng, torch.get_rng_state())
    result.assert_unchanged()


@pytest.mark.parametrize("change", [dict(release=None), dict(release=True),
    dict(donor="D2"), dict(donor="D3"), dict(source_id="D1_Stim8hr"),
    dict(source_id="D2_Stim48hr"), dict(batch_rows=4), dict(replicate_id="")])
def test_bad_or_missing_release_roles_fail_before_payload(fixture, monkeypatch, change):
    _, _, args, handle = fixture
    monkeypatch.setattr(handle, "capture_publication_identity", lambda: pytest.fail("no payload access"))
    with pytest.raises((ValueError, TypeError)):
        compile_evaluation_truth(handle, **(args | change))


def test_early_query_handle_cannot_be_promoted_by_label(fixture, monkeypatch):
    root, index, args, _ = fixture
    early = index.open(root, args["fold"], "query", query_donor="D1", max_batch_rows=3)
    monkeypatch.setattr(early, "capture_publication_identity", lambda: pytest.fail("no payload access"))
    with pytest.raises(ValueError, match="capability"):
        compile_evaluation_truth(early, **args)


def test_release_for_different_representation_cannot_open_truth(query_fixture, monkeypatch):
    root, index, args = query_fixture
    release = released(root, index, args, representation="e"*64)
    late = index.open(root, args["fold"], "evaluation_truth", query_donor="D1", release=release)
    monkeypatch.setattr(late, "capture_publication_identity", lambda: pytest.fail("no payload access"))
    with pytest.raises(ValueError, match="representation"):
        compile_evaluation_truth(late, **(args | dict(source_id="D1_Stim48hr", release=release)))


def test_changed_fitting_encoder_rejected_before_payload(fixture, monkeypatch):
    _, _, args, handle = fixture
    with torch.no_grad(): args["observation"].scale.add_(.01)
    monkeypatch.setattr(handle, "capture_publication_identity", lambda: pytest.fail("no payload access"))
    with pytest.raises(ValueError, match="representation"):
        compile_evaluation_truth(handle, **args)


def test_native_change_during_endpoint_compile_is_not_returned(fixture, monkeypatch):
    root, _, args, handle = fixture
    encode = args["observation"].encode
    def changed(counts):
        result = encode(counts)
        (root / "source/D1_Stim48hr/population_accounting.parquet").touch()
        return result
    monkeypatch.setattr(args["observation"], "encode", changed)
    with pytest.raises(ValueError, match="publication changed"):
        compile_evaluation_truth(handle, **args)


def test_truth_is_factory_only_and_receipt_cannot_be_relabelled(fixture):
    _, _, args, handle = fixture
    truth = compile_evaluation_truth(handle, **args)
    receipt = truth.receipt
    receipt["query_donor"] = "D2"
    assert truth.receipt["query_donor"] == "D1"
    with pytest.raises(TypeError, match="released endpoint compiler"):
        CompiledTruth(truth.fold, truth.catalog, truth.feature_axis_sha256, truth.observation_binding,
            truth.population, truth.release, receipt)
    states = truth.population.states
    states.setflags(write=True)
    states[0, 0] += 1
    states.setflags(write=False)
    with pytest.raises(ValueError, match="content identity"):
        truth.assert_unchanged()
