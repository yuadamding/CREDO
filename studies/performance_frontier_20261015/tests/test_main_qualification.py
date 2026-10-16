"""Tiny fabricated input/objective checks: zero optimizer calls, no CUDA."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontier import fitting_resources as resources
from frontier.configuration import configurations
from frontier.count_objective import sampling_scale_from_sizes
from frontier.fitting_resources import FittingResourcePolicy
from frontier.main_qualification_fixture import FixtureDimensions, QualificationProvider
from frontier.main_qualification_owner import DisposableBudget
from frontier.main_qualification_plan import (cases, plan, qualification_schedule,
    QualificationCase, SCOPE, PLANNED_OPTIMIZER_CALLS)
from frontier.training_owner import OwnerSchedule


GIB, MIB = 1024**3, 1024**2


@pytest.fixture(autouse=True)
def zero_steps(monkeypatch, request):
    torch.set_num_threads(2)
    def forbidden(*_a, **_k): pytest.fail("No optimizer/GPU execution allowed")
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.LBFGS):
        monkeypatch.setattr(cls, "step", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    request.node.user_properties.append(("disposable_cpu_optimizer_calls", 0))


def test_finite_workload_and_all_configurations_both_donor_shapes():
    rows = cases()
    assert len(rows) == 48 and len({c.identifier for c in rows}) == 48
    assert {c.configuration_id for c in rows} == {c.configuration_id for c in configurations()}
    assert {len(c.training_donors) for c in rows} == {2, 3}
    assert sum(c.restart_comparison for c in rows) == 2
    assert sum(c.optimizer_calls for c in rows) == PLANNED_OPTIMIZER_CALLS == 104
    assert plan()["scientific_updates"] == 0
    assert plan()["status"] == "proposal_not_launch_authority"


def test_cuda_fixture_schedule_does_not_change_scientific_schedule():
    small = qualification_schedule()
    real = OwnerSchedule()
    assert small.stages == (2, 3, 4) and small.state_updates == 1
    assert real.stages == (1024, 3072, 9216) and real.state_updates == 256
    assert real.cache_period == 64 and real.checkpoint_period == 256
    assert small.logical_batch_laws == real.logical_batch_laws == 1024
    assert small.training_particles == real.training_particles == 128


@pytest.mark.parametrize("field,value", [("stages", (2, 3, 5)), ("state_updates", 2),
    ("cache_period", 1), ("cache_particles", 128), ("logical_batch_laws", 32),
    ("training_particles", 64), ("maximum_refresh_laws", 128), ("checkpoint_period", 2)])
def test_cuda_fixture_schedule_is_not_an_unbounded_short_training_route(field, value):
    with pytest.raises(ValueError, match="qualification"):
        replace(qualification_schedule(), **{field: value})


def test_budget_accounts_planned_cases_and_never_retries():
    budget = DisposableBudget()
    first = cases()[0]
    budget.admit(first)
    with pytest.raises(ValueError): budget.admit(first)
    budget.failed = True
    with pytest.raises(ValueError): budget.admit(cases()[1])
    budget = DisposableBudget(); budget.calls = 111
    with pytest.raises(ValueError): budget.admit(first)


def test_full_dimensions_rejected_on_cpu_before_any_tensor_allocation(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda *_a, **_k: pytest.fail("No full CPU tensors"))
    with pytest.raises(ValueError, match="CUDA-only"):
        QualificationProvider(cases()[0], FixtureDimensions(), device="cpu")


@pytest.mark.parametrize("case", cases())
def test_tiny_fabricated_inputs_use_actual_routing_and_complete_count_denominator(case):
    provider = QualificationProvider(case, FixtureDimensions.tiny(), device="cpu")
    batch = provider.sample(accepted_updates=0, phase="state")
    d = provider.dimensions
    assert batch.inputs.source.shape == (d.logical_laws, d.particles, 48)
    assert batch.inputs.noise.shape == (8, *batch.inputs.source.shape)
    assert len(batch.inputs.cache_indices.unique()) == d.logical_laws
    assert batch.molecular is None
    assert batch.sampling_scale == sampling_scale_from_sizes(population=len(provider.inventory.law_ids), logical_batch=d.logical_laws)
    assert torch.allclose(batch.inputs.geometry_weights.sum(), torch.tensor(1.))
    for i, index in enumerate(batch.inputs.cache_indices.tolist()):
        assert batch.inputs.law_ids[i] == provider.inventory.law_ids[index]
        assert batch.inputs.target_ids[i] == provider.inventory.target_ids[index]
    blocks, cache = provider.count_inputs()
    assert len(blocks) == len(case.training_donors)
    assert all(len(block.log_source) == d.catalog for block in blocks)
    assert all(len(block.modeled_indices) == d.laws_per_donor < d.catalog for block in blocks)
    assert any((block.endpoint_counts == 0).any() for block in blocks)
    for block in blocks: block.validate(cache)
    provider.accepted(accepted_updates=1)
    joint = provider.sample(accepted_updates=1, phase="joint")
    if case.configuration_id[:2] in ("F3", "F4", "F5"):
        mol = joint.molecular
        assert mol is not None
        assert mol.observed_targets.dtype == mol.target_weights.dtype == torch.float64
        assert torch.allclose(mol.observed_targets.sum(-1), torch.ones(len(mol.observed_targets), dtype=torch.float64))
        assert len(mol.controls.donor_ids) == len(case.training_donors)
        assert mol.controls.observed_compositions.shape == (len(case.training_donors), d.genes)
        assert mol.controls.observed_compositions.dtype == mol.controls.reduction_weights.dtype == torch.float64
        assert torch.allclose(mol.controls.mixture_weights, torch.ones(len(case.training_donors), dtype=torch.float64))
    else:
        assert joint.molecular is None


def test_sampler_restore_and_width_pair_get_identical_actual_inputs():
    base = QualificationCase("F0-V1", ("D3", "D4"), False)
    other = QualificationCase("F0-V2", ("D3", "D4"), True)
    first, second = (QualificationProvider(c, FixtureDimensions.tiny(), device="cpu") for c in (base, other))
    a, b = first.sample(accepted_updates=0, phase="state"), second.sample(accepted_updates=0, phase="state")
    assert torch.equal(a.inputs.source, b.inputs.source) and torch.equal(a.inputs.noise, b.inputs.noise)
    assert torch.equal(a.inputs.cache_indices, b.inputs.cache_indices)
    first.accepted(accepted_updates=1)
    refreshed = tuple(first.cache_batches(accepted_updates=1, particles=32, maximum_laws=3))
    assert sum(len(x.law_ids) for x in refreshed) == len(first.inventory.law_ids)
    state = first.state_dict()
    second.load_state_dict(state)
    a, b = first.sample(accepted_updates=1, phase="joint"), second.sample(accepted_updates=1, phase="joint")
    assert torch.equal(a.inputs.source, b.inputs.source) and first.state_dict() == second.state_dict()


def test_cuda_resource_policy_keeps_h100_headroom_and_rejects_cpu(monkeypatch):
    policy = FittingResourcePolicy(SCOPE, 256*MIB, 64*MIB, 256*MIB, 8*MIB, 200*GIB, 8*GIB)
    monkeypatch.setattr(resources, "process_resources", lambda *_a: pytest.fail("No device probe"))
    with pytest.raises(RuntimeError, match="cuda:0"):
        resources.check_process_resources(policy, "cpu")
    monkeypatch.setattr(resources, "process_resources", lambda *_a: dict(maximum_rss_bytes=GIB, threads=8,
        gpu_name="NVIDIA H100 80GB HBM3", total_gpu_bytes=79*GIB, peak_reserved_gpu_bytes=20*GIB))
    assert resources.check_process_resources(policy, "cuda:0")["threads"] == 8
    with pytest.raises(ValueError): replace(policy, output_limit_bytes=17*GIB)
    with pytest.raises(ValueError): replace(policy, minimum_gpu_headroom_bytes=0)


def test_owner_state_separates_disposable_calls_without_running_an_optimizer():
    from frontier.training_owner import TrainingOwner
    owner = object.__new__(TrainingOwner)
    owner.lease = SimpleNamespace(identity={})
    owner.status = "stage_complete"
    owner.accepted_updates = owner.attempted_updates = owner.optimizer_calls = owner.healthy_optimizer_updates = 2
    owner.model = SimpleNamespace(phase="joint")
    owner.joint_entries = 1; owner.refresh_updates = [1]
    owner.last_checkpoint = owner.last_metrics = owner.failure = owner.retention_policy = owner.resource_guard = None
    owner.schedule = qualification_schedule()
    state = owner._state()
    assert state["scientific_updates"] == 0 and state["disposable_optimizer_calls"] == 2
    owner.schedule = OwnerSchedule()
    state = owner._state()
    assert state["scientific_updates"] == 2 and state["disposable_optimizer_calls"] == 0


@pytest.mark.parametrize("family", ["F0", "F1", "F2", "F3", "F4", "F5"])
def test_tiny_provider_drives_real_objective_and_gradients_without_optimizer(family):
    from frontier.geometry import GeometryOptions, RaggedSinkhornGeometry, geomloss_backend
    from frontier.models import build_model
    from frontier.objective_registry import ObjectiveRules, RULES_SHA256, FittingScaleBinding, build_objective_protocol
    from frontier.observation import ObservationFold, SparseR48
    from frontier.training_objective import ObjectiveExecution, accumulate_training_objective
    case = QualificationCase(family+"-V1", ("D3", "D4"), False)
    provider = QualificationProvider(case, FixtureDimensions.tiny(), device="cpu")
    provider.accepted(accepted_updates=1)  # Sampler-position fixture only; no claimed update.
    batch = provider.sample(accepted_updates=1, phase="joint")
    blocks, cache = provider.count_inputs()
    cache.apply_full_refresh(torch.zeros_like(cache.values), update_index=1, particles=32)
    private = dict(target_ids=provider.target_ids, support_factors=[.5]*len(provider.target_ids)) if family in ("F3", "F5") else {}
    model = build_model(case.configuration_id, seed=0, **private)
    model.set_phase("joint")
    decoder = SparseR48(ObservationFold("fabricated", case.training_donors, "a"*64),
        features=3, hidden_dims=(8, 4), maximum_rows=8, seed=0)
    decoder.scaling_ready.fill_(True); decoder.freeze()
    rules = ObjectiveRules((Path(__file__).resolve().parents[1]/"MAIN_OBJECTIVE_RULES.json").read_bytes(), RULES_SHA256)
    scales = FittingScaleBinding(fitting_donors=case.training_donors, rules_sha256=rules.identity,
        observation_artifact_sha256=provider.inventory.observation_sha256,
        fitting_input_sha256=provider.inventory.input_sha256, support_inventory_sha256="a"*64,
        scale_artifact_sha256="b"*64, producer_sha256="c"*64, sampler_sha256="d"*64,
        values={k:1. for k in ("state", "count", "shape", "endpoint", "effect")})
    protocol = build_objective_protocol(case.configuration_id, rules=rules, scales=scales,
        fitting_donors=case.training_donors, observation_artifact_sha256=provider.inventory.observation_sha256,
        fitting_input_sha256=provider.inventory.input_sha256)
    geometry = RaggedSinkhornGeometry(options=GeometryOptions(4096, 2, geomloss_backend().identity["version"]))
    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
    result = accumulate_training_objective(model, batch.inputs, protocol=protocol,
        execution=ObjectiveExecution(1, 1, 3, 4, 4, 100000, 3e-6, 3e-6), geometry=geometry,
        blocks=blocks, cache=cache, sampling_scale=batch.sampling_scale,
        decoder=decoder, molecular=batch.molecular)
    assert torch.isfinite(result.displayed_objective)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert all(torch.equal(before[k], v) for k,v in model.state_dict().items())
    assert result.optimizer_calls == 0
