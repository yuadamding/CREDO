"""Compact actual CPU forward/backward only; optimizer and CUDA are forbidden."""

import copy
from pathlib import Path

import pytest
import torch

from frontier.configuration import configurations
from frontier.geometry import GeometryOptions, geomloss_backend
from frontier.main_qualification_plan import cases, plan
from frontier.main_qualification_precision import (PHASES, TOLERANCES, build_precision_fixture,
    compare_evaluations, evaluate_precision_fixture, precision_dimensions, precision_plan,
    run_paired_precision_case, _move)
from frontier.objective_registry import ObjectiveRules, RULES_SHA256


@pytest.fixture(autouse=True)
def zero_updates(monkeypatch, request):
    torch.set_num_threads(2)
    def forbidden(*_a, **_k):
        pytest.fail("No optimizer or CUDA execution in local precision tests")
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.LBFGS):
        monkeypatch.setattr(cls, "step", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    request.node.user_properties.append(("disposable_cpu_optimizer_calls", 0))


@pytest.fixture
def rules():
    return ObjectiveRules((Path(__file__).resolve().parents[1]/"MAIN_OBJECTIVE_RULES.json").read_bytes(), RULES_SHA256)


@pytest.fixture
def geometry():
    return GeometryOptions(4096, 4, geomloss_backend().identity["version"])


def test_exact_plan_all_recipes_both_donor_counts_and_training_phases():
    value = precision_plan()
    assert value["status"] == "proposal_not_launch_authority"
    assert len(value["cases"]) == 96
    assert value["planned_objective_backward_evaluations"] == 288
    assert value["optimizer_calls"] == value["scientific_updates"] == 0
    assert not value["full_worker_entry_verified"] and not value["main_worker_qualification_pass"]
    assert {r["configuration_id"] for r in value["cases"]} == {c.configuration_id for c in configurations()}
    assert {len(r["training_donors"]) for r in value["cases"]} == {2, 3}
    assert {r["phase"] for r in value["cases"]} == set(PHASES)
    assert precision_dimensions().genes == 31  # Compact precision, NOT full RNA-axis qualification.
    assert plan()["paired_numerical_precision"] == value
    assert plan()["owner_optimizer_calls"] == 104
    assert plan()["planned_optimizer_calls"] == 111


LOCAL_CASES = [(c, phase) for c in cases() if len(c.training_donors) == 2 for phase in PHASES]
LOCAL_CASES += [(c, "joint") for c in cases() if len(c.training_donors) == 3 and c.configuration_id.endswith("-V1")]


@pytest.mark.parametrize("case,phase", LOCAL_CASES, ids=[c.identifier+"-"+p for c, p in LOCAL_CASES])
def test_actual_cpu_accumulation_all_recipes_and_both_widths(case, phase, rules, geometry):
    fixture = build_precision_fixture(case, phase, rules=rules, source_sha256="a"*64)
    before = {k: v.clone() for k, v in fixture["model"].state_dict().items()}
    one = evaluate_precision_fixture(fixture, device="cpu", microbatch_laws=1, geometry_options=geometry)
    four = evaluate_precision_fixture(fixture, device="cpu", microbatch_laws=4, geometry_options=geometry)
    report = compare_evaluations(one, four)
    assert report["status"] == "passed", report
    assert report["all_parameter_gradients_checked"] and report["compared_gradient_elements"] > 0
    assert one["optimizer_calls"] == one["cache_writes"] == four["optimizer_calls"] == 0
    assert all(torch.equal(before[k], v) for k, v in fixture["model"].state_dict().items())
    assert all(p.grad is None for p in fixture["model"].parameters())
    assert ("count_ce" in one["values"]) == (phase == "joint")
    molecular = phase == "joint" and case.configuration_id[:2] in ("F3", "F4", "F5")
    assert ("molecular/effect_mse" in one["values"]) == molecular
    assert bool(one["coverage"]["control_component_replays"]) == molecular
    if phase == "state":
        assert report["absent_gradients"]  # Disabled reaction is not invented as a zero gradient.


def test_fixture_copies_truth_weights_noise_and_cache_without_device_rng(rules):
    case = next(c for c in cases() if c.configuration_id == "F3-V1" and len(c.training_donors) == 2)
    fixture = build_precision_fixture(case, "joint", rules=rules, source_sha256="a"*64)
    original = fixture["batch"]
    state = torch.get_rng_state().clone()
    copied = _move(original, "cpu")
    assert torch.equal(state, torch.get_rng_state())
    assert copied is not original and copied.inputs.source.data_ptr() != original.inputs.source.data_ptr()
    assert torch.equal(copied.inputs.noise, original.inputs.noise)
    assert copied.molecular.observed_targets.dtype == torch.float64
    assert any(bool((o.weights == 0).any()) for o in copied.inputs.observations)
    assert len(fixture["blocks"][0].log_source) == 113
    assert len(fixture["blocks"][0].modeled_indices) == 8
    assert bool(fixture["cache"].values.any())


@pytest.fixture
def scalar_evaluation():
    return dict(values={"objective": torch.tensor(1.)}, gradients={"active": torch.tensor([.2, .4]), "disabled": None},
        requires_grad={"active": True, "disabled": False}, parameter_count=3, active_parameter_count=2,
        coverage={"laws": 16}, active_indices=torch.tensor([0, 1]))


@pytest.mark.parametrize("change", ["coverage", "requires_grad", "parameter_count", "active_parameter_count",
    "active_indices", "missing_channel", "missing_parameter", "gradient_participation", "nonfinite"])
def test_pair_rejects_mismatched_population_or_parameter_inventory(scalar_evaluation, change):
    other = copy.deepcopy(scalar_evaluation)
    if change == "coverage": other["coverage"] = {"laws": 15}
    elif change == "requires_grad": other["requires_grad"]["disabled"] = True
    elif change in ("parameter_count", "active_parameter_count"): other[change] += 1
    elif change == "active_indices": other["active_indices"] = torch.tensor([1, 0])
    elif change == "missing_channel": other["values"] = {}
    elif change == "missing_parameter": other["gradients"].pop("active")
    elif change == "gradient_participation": other["gradients"]["disabled"] = torch.tensor(0.)
    else: other["values"]["objective"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError): compare_evaluations(scalar_evaluation, other)


def test_failures_are_retained_not_rescaled_or_floored(scalar_evaluation):
    actual = copy.deepcopy(scalar_evaluation)
    actual["gradients"]["active"][0] += .01
    result = compare_evaluations(scalar_evaluation, actual)
    assert result["status"] == "failed"
    assert result["maximum_tolerance_normalized_error"] > 1
    assert result["absent_gradients"] == ["disabled"]
    assert result["compared_gradient_elements"] == 2
    assert TOLERANCES["gradient_atol"] == 2e-5 and TOLERANCES["gradient_rtol"] == 2e-4


def test_cuda_producer_cannot_claim_cpu_local_test_as_cuda_pass(monkeypatch, rules, geometry):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        run_paired_precision_case(cases()[0], "joint", rules=rules, geometry_options=geometry, source_sha256="a"*64)


def test_width_comparison_uses_same_observations_and_larger_actual_parameter_count(rules):
    def case(identifier): return next(c for c in cases() if c.configuration_id == identifier and len(c.training_donors) == 2)
    first = build_precision_fixture(case("F0-V1"), "joint", rules=rules, source_sha256="a"*64)
    second = build_precision_fixture(case("F0-V2"), "joint", rules=rules, source_sha256="a"*64)
    assert first["input_identity"] == second["input_identity"]
    assert torch.equal(first["batch"].inputs.source, second["batch"].inputs.source)
    assert torch.equal(first["batch"].inputs.noise, second["batch"].inputs.noise)
    assert sum(p.numel() for p in first["model"].parameters()) < sum(p.numel() for p in second["model"].parameters())


def test_precision_rejects_arbitrary_microbatch_without_qualifying_it(rules, geometry):
    fixture = build_precision_fixture(cases()[0], "state", rules=rules, source_sha256="a"*64)
    for bad in (True, 2, 32):
        with pytest.raises(ValueError):
            evaluate_precision_fixture(fixture, device="cpu", microbatch_laws=bad, geometry_options=geometry)


def test_mixed_precision_cannot_masquerade_as_fp32(rules, geometry):
    fixture = build_precision_fixture(cases()[0], "state", rules=rules, source_sha256="a"*64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            build_precision_fixture(cases()[0], "state", rules=rules, source_sha256="a"*64)
        with pytest.raises(ValueError, match="autocast"):
            evaluate_precision_fixture(fixture, device="cpu", microbatch_laws=1, geometry_options=geometry)
