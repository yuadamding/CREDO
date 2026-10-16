"""Fabricated native files only, unchanged frozen encoder, zero optimizer calls."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from frontier.cache_compiler import CompilerLimits, QuerySourceCompileSpec, SourceCompileSpec
from frontier.native_inputs import FoldInputs
from frontier.observation import ObservationFold, SparseR48
from frontier.query_compiler import compile_query_source
from frontier.training_owner import ObservationBinding, observation_state_digest
from test_native_inputs import fixture as native_fixture
from test_cache_compiler import fixture as fitting_fixture


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_a, **_k: pytest.fail("optimizer forbidden"))
    root, _, _, index = native_fixture.__wrapped__(tmp_path)
    fold = FoldInputs("inner-D3-D4", ("D3", "D4"))
    observation = SparseR48(ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256),
        features=3, hidden_dims=(8, 6), latent_dim=48, maximum_rows=4,
        maximum_dense_bytes=1024**2, seed=7)
    with torch.no_grad():
        observation.scaling_ready.fill_(True)
    observation.freeze()
    binding = ObservationBinding("a"*64, observation_state_digest(observation))
    args = dict(fold=fold, donor="D1", source_id="D1_Stim8hr", replicate_id="aggregate-unresolved",
        observation=observation, observation_binding=binding,
        limits=CompilerLimits(8*1024**2, 3, 64*1024, 2, 64*1024), batch_rows=3)
    return root, index, args


def handle(fixture, role="query", donor="D1"):
    root, index, args = fixture
    kwargs = dict(query_donor=donor) if role == "query" else {}
    return index.open(root, args["fold"], role, max_batch_rows=3, **kwargs)


def test_query_uses_only_heldout_early_rows_with_frozen_training_representation(fixture, monkeypatch):
    _, _, args = fixture
    source = handle(fixture)
    seen = []
    original = source.read_rows
    def record(addresses):
        seen.extend(addresses)
        return original(addresses)
    monkeypatch.setattr(source, "read_rows", record)
    initial = observation_state_digest(args["observation"])
    rng = torch.get_rng_state().clone()
    result = compile_query_source(source, **args)
    assert seen == [("D1_Stim8hr", 0, i) for i in range(3)]
    assert result.source.fold.training_donors == ("D3", "D4")
    assert result.source.reference.donor == "D1"
    assert result.source.captured_counts.tolist() == [2, 1, 0]
    assert result.source.positive_counts.tolist() == [1, 1, 0]
    assert result.source.states.shape == (2, 48)
    assert result.population.mean_compositions.shape == (3, 3)
    assert result.receipt["retained_early_RNA_summary_bytes"] == 3 * 3 * 8
    assert result.receipt["retained_early_RNA_summary_charged_by_forecast_core"] is False
    assert np.all(result.population.mean_compositions[-1] == 0)
    assert result.receipt["input_lease_exit"]["status"] == "unchanged"
    assert result.receipt["query_endpoint_used"] is False
    assert result.receipt["optimizer_calls"] == 0
    assert result.receipt["compilation"]["access_role"] == "query"
    assert observation_state_digest(args["observation"]) == initial
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("change", [
    dict(donor="D3"), dict(source_id="D1_Stim48hr"), dict(source_id="D2_Stim8hr"),
    dict(fold=FoldInputs("outer-D1", ("D2", "D3", "D4"))),
    dict(batch_rows=4), dict(replicate_id=""),
])
def test_bad_query_identity_fails_before_payload(fixture, monkeypatch, change):
    _, _, args = fixture
    source = handle(fixture)
    monkeypatch.setattr(source, "capture_publication_identity", lambda: pytest.fail("no input access"))
    with pytest.raises(ValueError):
        compile_query_source(source, **(args | change))


def test_fitting_handle_cannot_be_relabelled_as_query(fixture, monkeypatch):
    _, _, args = fixture
    source = handle(fixture, role="dynamics_source")
    monkeypatch.setattr(source, "capture_publication_identity", lambda: pytest.fail("no input access"))
    with pytest.raises(ValueError, match="capability"):
        compile_query_source(source, **args)


def test_query_spec_does_not_weaken_existing_fitting_or_endpoint_gates():
    spec, _, _, _ = fitting_fixture()
    query = QuerySourceCompileSpec(**(asdict(spec) | {
        "fold": spec.fold, "catalog": spec.catalog, "observation_binding": spec.observation_binding,
        "shards": spec.shards, "donor": "D1", "source_id": "D1_Stim8hr", "access_role": "query"}))
    assert query.identity != spec.identity
    for change in (dict(condition="Stim48hr"), dict(access_role="evaluation_truth"),
                   dict(donor="D3"), dict(access_role="dynamics_source")):
        with pytest.raises(ValueError, match="held-out"):
            replace(query, **change)
    with pytest.raises(ValueError, match="fitting donor"):
        replace(spec, donor="D1", access_role="query")
    with pytest.raises(ValueError, match="fitting donor"):
        SourceCompileSpec(**{name: getattr(query, name) for name in asdict(query)})


def test_changed_selected_encoder_rejected_before_input_lease(fixture, monkeypatch):
    _, _, args = fixture
    source = handle(fixture)
    with torch.no_grad(): args["observation"].center.add_(1)
    monkeypatch.setattr(source, "capture_publication_identity", lambda: pytest.fail("no input access"))
    with pytest.raises(ValueError, match="selected observation"):
        compile_query_source(source, **args)


def test_input_change_during_compile_rejects_before_return(fixture, monkeypatch):
    root, _, args = fixture
    source = handle(fixture)
    original = args["observation"].encode
    def mutate(counts):
        result = original(counts)
        path = root / "source/D1_Stim8hr/population_accounting.parquet"
        path.touch()
        return result
    monkeypatch.setattr(args["observation"], "encode", mutate)
    with pytest.raises(ValueError, match="publication changed"):
        compile_query_source(source, **args)


def test_forecast_source_early_means_and_states_do_not_admit_as_fitting_cache(fixture):
    _, _, args = fixture
    result = compile_query_source(handle(fixture), **args)
    assert result.source.reference.source_access_sha256 == result.receipt["query_access_sha256"]
    assert result.source.reference.condition == "Stim8hr"
    assert result.receipt["compilation"]["schema"] == "credo.frontier.streaming-query-source.v1"
    assert not result.receipt["publication_claimed"]
