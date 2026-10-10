"""Equal-input optimizer qualification remains distinct from backward checks."""
# ruff: noqa: E402
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]

import pytest
import torch

from credo.recipes.compact_sde_v3.recipe import CompactV3Config
from gpu_native import DeviceLawCache
from gpu_native_benchmark import synthetic_data
import gpu_native_qualification as qualification


def test_first_adam_formula_exposes_near_zero_amplification_without_updates():
    actual, expected = {"weight": torch.tensor([-1e-8])}, {"weight": torch.tensor([1e-8])}
    assert qualification.compare(actual, expected, "gradients")["passed"]
    value = qualification.first_adam_sensitivity({"weight": torch.tensor([.1])}, actual, expected)
    assert value["maximum_absolute_parameter_difference"] == pytest.approx(.001)
    assert value["elements_exceeding_parameter_tolerance"] == value["gradient_sign_changes"] == 1
    assert value["additional_optimizer_steps"] == 0
    assert not value["affects_backend_acceptance"]


def test_comparison_failure_publishes_magnitude_and_count_before_raising():
    events = []
    with pytest.raises(RuntimeError, match="allclose failed: weight"):
        qualification.compare({"weight": torch.tensor([0., .1])},
            {"weight": torch.zeros(2)}, "Adam", progress=events.append)
    assert len(events) == 1 and events[0]["event"] == "numerical_agreement_failed"
    assert events[0]["failed_elements"] == 1
    assert events[0]["maximum_absolute_difference"] == pytest.approx(.1)
    assert events[0]["maximum_tolerance_ratio"] == pytest.approx(5000)
    json.dumps(events, allow_nan=False)


def test_exact_snapshot_rejects_approximately_equal_inputs():
    with pytest.raises(RuntimeError, match="exact identity failed"):
        qualification.exact_snapshot({"x": torch.tensor([1e-8])},
            {"x": torch.zeros(1)}, "optimizer input")


@pytest.fixture
def small_protocol():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Each fixture creates unrelated model shapes; do not leak Dynamo's
    # process-global specialization cache into the next numerical test.
    torch.compiler.reset()
    data = synthetic_data(groups=4, latent_dim=2, support=3, seed=8)
    data.context_backgrounds = ()
    problem = SimpleNamespace(training=data, embedding_ids=data.embedding_ids,
        control_embedding_ids=data.control_embedding_ids, latent_dim=data.latent_dim)
    cache = DeviceLawCache(data, device="cpu")
    candidate = dict(candidate_id="synthetic-adam", model=dict(hidden_dim=8, embedding_dim=2,
        trunk_kind="residual_mlp", trunk_depth=2))
    def config(candidate, seed):
        return CompactV3Config.model_validate(dict(model=dict(candidate["model"], context="none"),
            training=dict(seed=seed)))
    normalizers = dict(policy="balanced_fitting_scales_active_coordinate_count_v1",
        fitting_donors=["D3", "D4"], state=1., count=1.)
    contract = json.loads((ROOT / "FITTING_CONTRACT.json").read_text())
    budget = qualification.WorkBudget(full=30, partial=4, steps=6, refresh=1, chunks=1)
    events = []
    def run():
        return qualification.qualify(problem, [candidate], config, contract, normalizers,
            cache, progress=events.append, budget=budget, full_guides=2,
            partial_state=1, partial_mass=1, particles=4, steps=1, compiler_backend="eager")
    try:
        yield run, events, budget
    finally:
        cache.close()
        torch.compiler.reset()
        torch.set_num_threads(old_threads)


def test_complete_protocol_uses_identical_gradients_and_preserves_independent_checks(small_protocol, monkeypatch):
    run, events, budget = small_protocol
    original_call, original_step = qualification.one_call, torch.optim.Adam.step
    updates = []
    def near_zero(model, profile, *args, **kwargs):
        result = original_call(model, profile, *args, **kwargs)
        value = 1e-7 if profile == qualification.PROFILES[0] else -1e-7
        dict(model.named_parameters())["trunk.0.weight"].grad.reshape(-1)[0] = value
        return result
    def step(optimizer, *args, **kwargs):
        updates.append(bool(optimizer.param_groups[0]["foreach"]))
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(qualification, "one_call", near_zero)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    result = run()
    assert len(updates) == budget.used["steps"] == 6
    assert result["scientific_optimizer_updates"] == 0
    accelerated = [r for r in result["profiles"] if r["profile"] != qualification.PROFILES[0]]
    for profile in accelerated:
        row = profile["numerical_agreement"]
        assert all(row[key]["passed"] for key in
            ("loss", "gradients", "clipped", "norm", "Adam", "Adam_state",
             "optimizer_initial_parameters", "optimizer_input_gradients"))
        assert row["optimizer_initial_state"] == dict(passed=True, state="identical_empty_fresh_plain_Adam")
        assert profile["independent_gradient_Adam_sensitivity"]["elements_exceeding_parameter_tolerance"] > 0
    assert len([row for row in events if row["event"] == "qualification_numerical_agreement"]) == 6


@pytest.mark.parametrize("damage", ["parameters", "state"])
def test_bad_optimizer_still_fails_without_an_extra_step(small_protocol, monkeypatch, damage):
    run, events, budget = small_protocol
    original_step = torch.optim.Adam.step
    def corrupted(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        if optimizer.param_groups[0]["foreach"]:
            parameter = optimizer.param_groups[0]["params"][0]
            with torch.no_grad():
                if damage == "parameters":
                    parameter.add_(.1)
                else:
                    optimizer.state[parameter]["exp_avg"].add_(.1)
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", corrupted)
    with pytest.raises(RuntimeError, match="disposable Adam " + damage + " allclose failed"):
        run()
    assert budget.used["steps"] == 3
    assert events[-1]["event"] == "numerical_agreement_failed"
    assert events[-1]["failed_elements"] > 0
    json.dumps(events[-1], allow_nan=False)


def test_bad_backward_still_fails_before_accelerated_step(small_protocol, monkeypatch):
    run, events, budget = small_protocol
    original_call = qualification.one_call
    def corrupted(model, profile, *args, **kwargs):
        result = original_call(model, profile, *args, **kwargs)
        if profile != qualification.PROFILES[0]:
            dict(model.named_parameters())["trunk.0.weight"].grad.add_(.1)
        return result
    monkeypatch.setattr(qualification, "one_call", corrupted)
    with pytest.raises(RuntimeError, match="gradients allclose failed"):
        run()
    assert budget.used["steps"] == 2
    assert events[-1]["comparison"] == "gradients"
