"""File-backed fabricated metadata, cache and input-scope joins; no CUDA/steps."""

from dataclasses import asdict, replace
import copy
from pathlib import Path

import pytest
import torch

from frontier.fitting_inputs import FittingInputContract, _admit_device, open_fitting_inputs
from frontier.fitting_provider import FoldFittingProvider
from frontier.main_qualification_worker_plan import worker_schedule
from frontier.main_qualification_plan import qualification_schedule
from frontier.objective_registry import ObjectiveRules, RULES_SHA256
from test_fitting_inputs import package, encoded, published
from test_fitting_provider import fixture as provider_fixture
from test_main_worker import no_numerics


def qualified_record(package):
    record = copy.deepcopy(package["record"])
    record["schedule"] = asdict(worker_schedule())
    return record


def test_actual_input_metadata_preserves_selected_observation_cache_and_scalars(package, monkeypatch):
    record = qualified_record(package)
    fixture = FittingInputContract(encoded(record))
    parts = fixture.components()
    assert parts["native_binding"].synthetic_fixture
    assert parts["observation"].specification.synthetic_fixture
    assert parts["schedule"] == worker_schedule()
    for key in ("native", "observation", "preparation", "normalizers", "rules"):
        assert fixture.record()[key] == package["record"][key]
    with pytest.raises(ValueError, match="CUDA worker fixture"):
        _admit_device(fixture, "cpu", False)
    # Metadata admission only: no allocation or CUDA initialization occurs.
    monkeypatch.setattr(torch, "get_num_threads", lambda:8)
    assert _admit_device(fixture, "cuda:0", False) == torch.device("cuda:0")
    with pytest.raises(ValueError): _admit_device(fixture, "cuda:1", False)


@pytest.mark.parametrize("change", ["real_native", "real_observation", "full_shape", "wrong_batch"])
def test_compact_input_scope_cannot_import_scientific_or_full_shape_data(package, change):
    record = qualified_record(package)
    if change == "real_native": record["native"]["binding"]["synthetic_fixture"] = False
    elif change == "real_observation": record["observation"]["specification"]["synthetic_fixture"] = False
    elif change == "full_shape": record["schedule"] = asdict(qualification_schedule())
    else: record["schedule"]["logical_batch_laws"] = 3
    with pytest.raises(ValueError): FittingInputContract(encoded(record))


def test_actual_provider_init_admits_only_compact_eight_thread_cuda_metadata(monkeypatch):
    cache, _ = provider_fixture()
    root = Path(__file__).resolve().parents[1]
    rules = ObjectiveRules((root / "MAIN_OBJECTIVE_RULES.json").read_bytes(), RULES_SHA256)
    arguments = dict(rules=rules, configuration_id="F5-V2", model_seed=0,
        study_id="fabricated-worker-qualification", study_seed=731, schedule=worker_schedule())
    with pytest.raises(ValueError): FoldFittingProvider(cache, device="cpu", **arguments)
    monkeypatch.setattr(torch, "get_num_threads", lambda:8)
    provider = FoldFittingProvider(cache, device="cuda:0", **arguments)
    assert provider.position == 0 and provider.schedule == worker_schedule()
    assert len(provider.inventory.training_donors) == 2
    assert all(x.is_control is False or x.target_id == "NTC" for x in cache.manifest.catalog)
    with pytest.raises(ValueError): FoldFittingProvider(cache, device="cuda:1", **arguments)
    arguments["schedule"] = qualification_schedule()
    with pytest.raises(ValueError): FoldFittingProvider(cache, device="cuda:0", **arguments)


@pytest.mark.parametrize("package", [("D3", "D4"), ("D2", "D3", "D4")], indirect=True)
def test_fixture_sampler_never_needs_to_redraw_an_all_control_molecular_batch(package):
    record = copy.deepcopy(package["record"])
    record["schedule"] = asdict(replace(worker_schedule(), execution_scope="synthetic_cpu_only"))
    record["task"]["key"]["configuration_id"] = "F5-V2"
    # Real fabricated native/cache/decoder loading and source sampling; only
    # simulate sampler commit metadata, with no model optimizer or CUDA calls.
    with open_fitting_inputs(published(package, record)) as opened:
        provider = opened.prepared.arguments["provider"]
        assert len(provider.cache.laws) == 2 * len(provider.inventory.training_donors)
        for update in range(4):
            phase = "state" if update == 0 else "joint"
            batch = provider.sample(accepted_updates=update, phase=phase)
            assert len(batch.inputs.law_ids) == 4
            if phase == "joint":
                assert batch.molecular is not None
                assert batch.molecular.observed_targets.shape[0] >= 1
            provider.accepted(accepted_updates=update + 1)
