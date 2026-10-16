"""Zero-update checkpoint integration tests; no optimizer.step is called."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
import torch

from frontier.checkpoints import TrajectoryLease, load_checkpoint, save_checkpoint
from frontier.models import build_model
from frontier.planner import TrajectoryKey


def identity():
    key = TrajectoryKey(("D3", "D4"), "F3-V0", 0)
    digest = sha256(b"synthetic-checkpoint-phase").hexdigest()
    return {"trajectory_id": key.storage_identifier, "logical_trajectory_id": key.identifier,
            "fold_id": "D3-D4", "seed": 0, "input_sha256": digest,
            "configuration_sha256": digest, "source_sha256": digest}


def model(targets=("T1", "T2"), factors=(1., .5), controls=("NTC",)):
    return build_model("F3-V0", seed=0, target_ids=targets, support_factors=factors,
                       control_ids=controls)


def publish(lease, value, phase="joint", sampler=None):
    optimizer = torch.optim.Adam(value.parameters(), lr=.001)
    record = save_checkpoint(lease, model=value, optimizer=optimizer, scheduler=None,
        accepted_updates=0, attempted_updates=0, phase=phase,
        sampler_state={"position": 0} if sampler is None else sampler,
        count_cache={"fitness": torch.zeros(3)}, execution_scope="synthetic_cpu_only")
    return record


def restore(lease, record, value):
    return load_checkpoint(lease, record["directory"], expected_sha256=record["checkpoint_sha256"],
        model=value, optimizer=torch.optim.Adam(value.parameters(), lr=.001), scheduler=None,
        execution_scope="synthetic_cpu_only")


def mask(value):
    return {name: parameter.requires_grad for name, parameter in value.named_parameters()}


def test_joint_checkpoint_restores_fresh_state_phase_and_mask(tmp_path):
    original = model().set_phase("joint")
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        record = publish(lease, original)
    restored = model()
    assert restored.phase == "state" and not restored.reaction_enabled
    assert not restored.target_growth_intercepts.requires_grad
    with TrajectoryLease(tmp_path / "trajectory", identity(), "reader") as lease:
        state = restore(lease, record, restored)
    assert restored.phase == state["phase"] == "joint" and restored.reaction_enabled
    assert restored.target_growth_intercepts.requires_grad
    assert mask(restored) == mask(original)
    assert state["scientific_updates"] == state["disposable_optimizer_calls"] == 0
    for key, tensor in original.state_dict().items():
        assert torch.equal(restored.state_dict()[key], tensor)


def test_state_checkpoint_restores_joint_model_without_mass_training(tmp_path):
    original = model()
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        record = publish(lease, original, phase="state")
    restored = model().set_phase("joint")
    with TrajectoryLease(tmp_path / "trajectory", identity(), "reader") as lease:
        restore(lease, record, restored)
    assert restored.phase == "state" and not restored.reaction_enabled
    assert mask(restored) == mask(original)


def test_phase_mismatch_rejected_before_checkpoint_publication(tmp_path):
    value = model()
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        with pytest.raises(ValueError, match="actual model phase"):
            publish(lease, value, phase="joint")
        assert list((lease.root / "checkpoints").iterdir()) == []
    assert value.phase == "state" and not value.reaction_enabled


def test_inconsistent_reaction_or_gradient_mask_rejected(tmp_path):
    value = model().set_phase("joint")
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        value.reaction_enabled = False
        with pytest.raises(ValueError, match="reaction flag"):
            publish(lease, value)
        value.reaction_enabled = True
        value.target_growth_intercepts.requires_grad_(False)
        before = mask(value)
        with pytest.raises(ValueError, match="requires_grad mask"):
            publish(lease, value)
        assert mask(value) == before
        assert value.phase == "joint" and value.reaction_enabled
        assert list((lease.root / "checkpoints").iterdir()) == []


@pytest.mark.parametrize("targets,factors,controls", [
    (("T2", "T1"), (1., .5), ("NTC",)),
    (("T1", "T2"), (.5, 1.), ("NTC",)),
    (("T1", "T2"), (1., .5), ("other-control",)),
])
def test_same_shape_routing_change_rejected_before_model_mutation(tmp_path, targets, factors, controls):
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        record = publish(lease, model().set_phase("joint"))
    changed = model(targets, factors, controls)
    before = {key: tensor.clone() for key, tensor in changed.state_dict().items()}
    before_mask = mask(changed)
    with TrajectoryLease(tmp_path / "trajectory", identity(), "reader") as lease:
        with pytest.raises(ValueError, match="exact compatible"):
            restore(lease, record, changed)
    assert changed.phase == "state" and not changed.reaction_enabled
    assert mask(changed) == before_mask
    assert all(torch.equal(changed.state_dict()[key], tensor) for key, tensor in before.items())


def test_custom_generators_preserved_then_explicitly_restored_by_caller(tmp_path):
    source_generator = torch.Generator().manual_seed(741)
    numpy_generator = np.random.Generator(np.random.PCG64DXSM(852))
    sampler = {"position": 0, "source_generator_state": source_generator.get_state(),
               "numpy_bit_generator": "PCG64DXSM", "numpy_generator_state": numpy_generator.bit_generator.state}
    expected_torch = torch.randn(5, generator=source_generator)
    expected_numpy = numpy_generator.normal(size=5)
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        record = publish(lease, model().set_phase("joint"), sampler=sampler)
    local_torch = torch.Generator().manual_seed(999)
    local_numpy = np.random.Generator(np.random.PCG64DXSM(999))
    untouched_torch = local_torch.get_state().clone()
    untouched_numpy = local_numpy.bit_generator.state
    with TrajectoryLease(tmp_path / "trajectory", identity(), "reader") as lease:
        result = restore(lease, record, model())
    # Arbitrary caller-owned RNG objects are not discovered or automatically
    # mutated by the loader. The resumed sampler must install these states.
    assert torch.equal(local_torch.get_state(), untouched_torch)
    assert local_numpy.bit_generator.state == untouched_numpy
    returned = result["sampler_state"]
    assert returned["numpy_bit_generator"] == "PCG64DXSM"
    local_torch.set_state(returned["source_generator_state"])
    local_numpy.bit_generator.state = returned["numpy_generator_state"]
    assert torch.equal(torch.randn(5, generator=local_torch), expected_torch)
    assert np.array_equal(local_numpy.normal(size=5), expected_numpy)


def test_generic_linear_phase_and_gradient_mask_compatibility(tmp_path):
    original = torch.nn.Linear(3, 2)
    original.bias.requires_grad_(False)
    with TrajectoryLease(tmp_path / "trajectory", identity(), "writer") as lease:
        record = publish(lease, original, phase="generic")
    restored = torch.nn.Linear(3, 2)
    with TrajectoryLease(tmp_path / "trajectory", identity(), "reader") as lease:
        state = restore(lease, record, restored)
    assert state["phase"] == "generic"
    assert mask(restored) == mask(original)
    assert torch.equal(restored.weight, original.weight)
    assert torch.equal(restored.bias, original.bias)
