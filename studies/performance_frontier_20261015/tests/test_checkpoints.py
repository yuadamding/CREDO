"""Fabricated CPU tests only: five disposable optimizer calls, zero scientific updates."""

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from frontier.checkpoints import (TrajectoryLease, file_sha256, load_checkpoint,
                                  retention_plan, save_checkpoint)


def identity(**changes):
    return {"trajectory_id": "F0-V0-D3D4-seed0", "fold_id": "D3D4", "seed": 0,
            "input_sha256": "1" * 64, "configuration_sha256": "2" * 64,
            "source_sha256": "3" * 64, **changes}


def seed():
    random.seed(72)
    np.random.seed(72)
    torch.manual_seed(72)


def objects():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    return model, optimizer, scheduler


def step(model, optimizer, scheduler, sampler, cache):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(5, 3) + float(np.random.normal()) + random.random()
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    sampler["position"] += 5
    sampler["order"] = torch.randperm(5)
    cache["fitness"] += float(loss.detach())
    cache["refresh_generation"] += 1


def snapshot(lease, values, updates=0, **changes):
    model, optimizer, scheduler = values
    arguments = dict(model=model, optimizer=optimizer, scheduler=scheduler,
                     accepted_updates=updates, attempted_updates=updates, phase="joint",
                     sampler_state={"position": 0, "order": torch.arange(5)},
                     count_cache={"fitness": torch.zeros(6), "refresh_generation": 0},
                     execution_scope="synthetic_cpu_only")
    arguments.update(changes)
    return save_checkpoint(lease, **arguments)


def restore(lease, record, values, **changes):
    model, optimizer, scheduler = values
    arguments = dict(expected_sha256=record["checkpoint_sha256"], model=model,
                     optimizer=optimizer, scheduler=scheduler, execution_scope="synthetic_cpu_only")
    arguments.update(changes)
    return load_checkpoint(lease, record["directory"], **arguments)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_exact_uninterrupted_vs_one_plus_one_resume(tmp_path):
    # Four actual disposable CPU optimizer calls: two reference, one + one restart.
    seed()
    full = objects()
    sampler = {"position": 0, "order": torch.arange(5)}
    cache = {"fitness": torch.zeros(6), "refresh_generation": 0}
    for _ in range(2):
        step(*full, sampler, cache)
    expected_draws = (random.random(), np.random.random(), torch.randn(4))
    seed()
    partial = objects()
    resumed_sampler = {"position": 0, "order": torch.arange(5)}
    resumed_cache = {"fitness": torch.zeros(6), "refresh_generation": 0}
    for _ in range(1):
        step(*partial, resumed_sampler, resumed_cache)
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-a") as lease:
        record = snapshot(lease, partial, 1, sampler_state=resumed_sampler,
                          count_cache=resumed_cache, milestone=True)
    # Disturb every RNG and construct fresh model/optimizer objects before restoration.
    random.random()
    np.random.normal(size=7)
    torch.randn(9)
    resumed = objects()
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-b") as lease:
        state = restore(lease, record, resumed)
        assert state["accepted_updates"] == state["attempted_updates"] == 1
        assert state["scientific_updates"] == 0
        assert state["disposable_optimizer_calls"] == 1
        for _ in range(1):
            step(*resumed, state["sampler_state"], state["count_cache"])
        snapshot(lease, resumed, 2, sampler_state=state["sampler_state"], count_cache=state["count_cache"])
    for a, b in zip(full, resumed):
        assert_tree_equal(a.state_dict(), b.state_dict())
    assert_tree_equal(sampler, state["sampler_state"])
    assert_tree_equal(cache, state["count_cache"])
    assert_tree_equal(expected_draws, (random.random(), np.random.random(), torch.randn(4)))
    assert len(list((tmp_path / "trajectory" / "claims").glob("*.released.json"))) == 2


def test_exclusive_worker_and_changed_identity(tmp_path):
    root = tmp_path / "trajectory"
    with TrajectoryLease(root, identity(), "worker-a"):
        with pytest.raises(BlockingIOError):
            with TrajectoryLease(root, identity(), "worker-b"):
                pytest.fail("Concurrent worker admitted")
    for key, value in (("input_sha256", "4" * 64), ("configuration_sha256", "4" * 64),
                       ("source_sha256", "4" * 64), ("fold_id", "D1D2"),
                       ("trajectory_id", "different"), ("seed", 1)):
        with pytest.raises(ValueError, match="identity changed"):
            with TrajectoryLease(root, identity(**{key: value}), "worker-c"):
                pytest.fail("Changed identity admitted")


def test_no_clobber_hash_and_no_inference_only_loading(tmp_path):
    values = objects()
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-a") as lease:
        record = snapshot(lease, values)
        original_sha = file_sha256(Path(record["directory"]) / "checkpoint.pt")
        with pytest.raises(FileExistsError):
            snapshot(lease, values)
        with pytest.raises(ValueError, match="hash/size"):
            restore(lease, record, values, expected_sha256="0" * 64)
        assert file_sha256(Path(record["directory"]) / "checkpoint.pt") == original_sha
        # Fabricate a historical inference payload with a matching byte receipt;
        # schema/capability validation must reject it even when its hash is supplied.
        path = Path(record["directory"]) / "checkpoint.pt"
        torch.save({"schema": "credo.comparator.terminal_checkpoint.v1", "state_dict": {}}, path)
        receipt_path = path.parent / "COMPLETE.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.update(checkpoint_sha256=file_sha256(path), checkpoint_bytes=path.stat().st_size)
        receipt_path.write_text(json.dumps(receipt))
        record.update(checkpoint_sha256=receipt["checkpoint_sha256"])
        with pytest.raises(ValueError, match="resumable checkpoint"):
            restore(lease, record, values)


@pytest.mark.parametrize("changes", [
    {"attempted_updates": 1}, {"last_attempt_status": "failed"},
    {"sampler_state": {}}, {"count_cache": {"bad": object()}},
    {"count_cache": {"bad": torch.tensor(float("nan"))}},
])
def test_reject_unhealthy_or_unsupported_state_before_publication(tmp_path, changes):
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-a") as lease:
        with pytest.raises((ValueError, TypeError)):
            snapshot(lease, objects(), **changes)
        assert list((lease.root / "checkpoints").iterdir()) == []


def test_latest_previous_milestones_retained_without_deletion(tmp_path):
    # One actual disposable optimizer call for monotone snapshot boundaries.
    values = objects()
    sampler, cache = {"position": 0}, {"fitness": torch.zeros(6), "refresh_generation": 0}
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-a") as lease:
        first = snapshot(lease, values, milestone=True)
        records = [first]
        for number in (1,):
            step(*values, sampler, cache)
            records.append(snapshot(lease, values, number, sampler_state=sampler, count_cache=cache))
        policy = retention_plan(lease)
        assert policy["latest"] == records[1]["directory"]
        assert policy["previous"] == records[0]["directory"]
        assert policy["milestones"] == [first["directory"]]
        assert policy["other_history"] == []
        assert policy["deletion_performed"] is policy["deletion_authorized"] is False
        assert all(Path(record["directory"]).is_dir() for record in records)
        with pytest.raises(ValueError, match="latest"):
            restore(lease, first, values)


def test_partial_publication_is_not_retried_and_symlinks_rejected(tmp_path):
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-a") as lease:
        (lease.root / "checkpoints" / "update-000000001").mkdir()
        with pytest.raises(ValueError, match="Incomplete checkpoint"):
            snapshot(lease, objects(), 1)
    (tmp_path / "alias").symlink_to(tmp_path / "trajectory", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        TrajectoryLease(tmp_path / "alias", identity(), "worker-a")


def test_restore_rejects_changed_scope_structure_and_outside_path(tmp_path, monkeypatch):
    values = objects()
    with TrajectoryLease(tmp_path / "trajectory", identity(), "worker-a") as lease:
        record = snapshot(lease, values)
        original_load = torch.load
        calls = []

        def restricted_load(*args, **kwargs):
            calls.append(kwargs)
            assert kwargs["weights_only"] is True
            return original_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", restricted_load)
        restore(lease, record, objects())
        assert len(calls) == 1
        with pytest.raises(ValueError, match="compatible"):
            restore(lease, record, values, execution_scope="scientific")
        incompatible = list(objects())
        incompatible[0][0] = torch.nn.Linear(3, 9)
        incompatible[1] = torch.optim.Adam(incompatible[0].parameters(), lr=0.01)
        incompatible[2] = torch.optim.lr_scheduler.StepLR(incompatible[1], step_size=2, gamma=0.7)
        with pytest.raises(ValueError, match="structure"):
            restore(lease, record, incompatible)
        with pytest.raises(ValueError, match="latest"):
            load_checkpoint(lease, tmp_path / "outside", expected_sha256=record["checkpoint_sha256"],
                            model=values[0], optimizer=values[1], scheduler=values[2],
                            execution_scope="synthetic_cpu_only")
    with pytest.raises(RuntimeError, match="live"):
        retention_plan(lease)
