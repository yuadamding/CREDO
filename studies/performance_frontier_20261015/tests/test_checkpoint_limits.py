"""Bounded synthetic checkpoint writes and binding checks, zero optimizer steps."""

import io
import json
from pathlib import Path

import pytest
import torch

from frontier.checkpoints import (CheckpointByteLimitExceeded, TrajectoryLease,
    _BoundedCheckpointWriter, load_checkpoint, save_checkpoint)
from frontier.fitting_factory import prepare_fitting_components
from test_checkpoints import identity, objects, snapshot
from test_fitting_factory import package


def test_writer_exact_boundary_and_latched_overflow():
    stream = io.BytesIO()
    writer = _BoundedCheckpointWriter(stream, 4)
    assert writer.write(b"1234") == 4 and writer.write(b"") == 0
    with pytest.raises(CheckpointByteLimitExceeded): writer.write(b"5")
    with pytest.raises(CheckpointByteLimitExceeded): writer.write(b"footer")
    assert writer.exceeded and writer.written == 4 and stream.getvalue() == b"1234"


@pytest.mark.parametrize("kind", ["short", "error"])
def test_io_failure_disallows_archive_finalization_writes(kind):
    class Broken(io.BytesIO):
        def write(self, data):
            if kind == "error":
                super().write(data[:1])
                raise OSError("partial IO error")
            return super().write(data[:1])
    stream = Broken()
    writer = _BoundedCheckpointWriter(stream, 4)
    with pytest.raises(OSError): writer.write(b"abc")
    with pytest.raises(OSError, match="already failed"): writer.write(b"footer")
    assert stream.getvalue() == b"a" and writer.failed


def test_actual_torch_checkpoint_has_bounded_readback_and_restores(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_a, **_k: pytest.fail("Optimizer forbidden"))
    values = objects()
    with TrajectoryLease(tmp_path/"trajectory", identity(), "bounded") as lease:
        record = snapshot(lease, values, maximum_checkpoint_bytes=1024**2)
        assert record["maximum_checkpoint_bytes"] == 1024**2
        assert 0 < record["checkpoint_bytes"] <= record["maximum_checkpoint_bytes"]
        fresh = objects()
        state = load_checkpoint(lease, record["directory"], expected_sha256=record["checkpoint_sha256"],
            model=fresh[0], optimizer=fresh[1], scheduler=fresh[2], execution_scope="synthetic_cpu_only")
        assert state["scientific_updates"] == state["disposable_optimizer_calls"] == 0
        for key, value in values[0].state_dict().items():
            assert torch.equal(value, fresh[0].state_dict()[key])


@pytest.mark.parametrize("cap", [1, 512, 4096])
def test_actual_serialization_overflow_retains_only_bounded_incomplete_output(tmp_path, cap):
    with TrajectoryLease(tmp_path/"trajectory", identity(), "overflow") as lease:
        values = objects()
        with pytest.raises(CheckpointByteLimitExceeded):
            snapshot(lease, values, maximum_checkpoint_bytes=cap)
        directory = lease.root/"checkpoints/update-000000000"
        assert (directory/"checkpoint.pt").stat().st_size <= cap
        assert not (directory/"COMPLETE.json").exists()
        with pytest.raises((ValueError, FileExistsError)):
            snapshot(lease, values, maximum_checkpoint_bytes=1024**2)
        assert (directory/"checkpoint.pt").stat().st_size <= cap


@pytest.mark.parametrize("cap", [False, 0, -1, 1.5, "1000"])
def test_bad_cap_rejected_before_snapshot_copy_and_directory(tmp_path, monkeypatch, cap):
    with TrajectoryLease(tmp_path/"trajectory", identity(), "reject") as lease:
        monkeypatch.setattr("frontier.checkpoints._cpu_tree", lambda *_: pytest.fail("Must reject before copy"))
        with pytest.raises(ValueError, match="payload cap"):
            snapshot(lease, objects(), maximum_checkpoint_bytes=cap)
        assert not list((lease.root/"checkpoints").iterdir())


def test_factory_owner_binding_prevents_changing_payload_cap(package, tmp_path):
    args, _ = package
    prepared = prepare_fitting_components(**dict(args, maximum_checkpoint_bytes=1024**2))
    assert prepared.identity["training_owner"]["maximum_checkpoint_bytes"] == 1024**2
    with TrajectoryLease(tmp_path/"owner", prepared.identity, "bounded") as lease:
        owner = prepared.open_owner(lease)
        assert owner.maximum_checkpoint_bytes == 1024**2 and owner.optimizer_calls == 0
        owner.maximum_checkpoint_bytes *= 2
        with pytest.raises(ValueError, match="cap changed"):
            owner._checkpoint(milestone=False)
        assert not list((lease.root/"checkpoints").iterdir())
    changed = prepare_fitting_components(**dict(args, maximum_checkpoint_bytes=2*1024**2))
    assert changed.identity != prepared.identity
    with pytest.raises(ValueError, match="identity changed"):
        with TrajectoryLease(tmp_path/"owner", changed.identity, "not-a-cap-upgrade"):
            pytest.fail("A budget change is not exact continuation")


def test_factory_bad_cap_rejected_before_loading_selected_observation(package, monkeypatch):
    args, _ = package
    monkeypatch.setattr("frontier.fitting_factory.load_observation", lambda *_: pytest.fail("No decoder before cap validation"))
    with pytest.raises(ValueError, match="payload cap"):
        prepare_fitting_components(**dict(args, maximum_checkpoint_bytes=0))
