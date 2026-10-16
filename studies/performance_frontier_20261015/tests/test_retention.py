"""Disposable checkpoint-file retirement only; zero optimizer.step calls.

Update positions below are fabricated serialization labels, not executed model
updates. No historical files, real inputs, remote files or inference weights are
deleted by these fixtures.
"""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from frontier.checkpoints import (TrajectoryLease, file_sha256, load_checkpoint,
                                  retention_plan, save_checkpoint)
from frontier.retention import (STUDY_ID, RetentionPolicy, bind_policy, commit_retirement,
                                identity_sha256, prepare_retirement)
import frontier.retention as retention


def identity():
    return {"trajectory_id": "synthetic-retention", "fold_id": "D3-D4", "seed": 0,
            "input_sha256": "a" * 64, "configuration_sha256": "b" * 64, "source_sha256": "c" * 64,
            "fixture_note": "fabricated snapshot labels; zero actual optimizer calls"}


def setup(tmp_path):
    study = tmp_path / STUDY_ID
    study.mkdir()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    return study, model, optimizer


def snapshot(lease, model, optimizer, update, *, milestone=False):
    # Fabricated counters qualify retention/resume serialization, not scientific progress.
    return save_checkpoint(lease, model=model, optimizer=optimizer, scheduler=None,
                           accepted_updates=update, attempted_updates=update, phase="synthetic_snapshot",
                           sampler_state={"position": update}, count_cache={"fixture": True},
                           execution_scope="synthetic_cpu_only", milestone=milestone)


def policy(study, lease):
    return RetentionPolicy(str(study), str(lease.root), identity_sha256(lease.identity), (2,))


def populate(lease, model, optimizer):
    return [snapshot(lease, model, optimizer, update, milestone=update == 1) for update in range(6)]


def test_exact_payload_retention_preserves_evidence_and_latest_resume(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        # These files are deliberately outside the eligible checkpoint.pt locations.
        export = lease.root / "observation-candidate-000000001.pt"
        export.write_bytes(b"inference export must survive")
        audit = lease.root / "AUDIT.json"
        audit.write_text('{"keep":true}')
        original = {str(path): file_sha256(path) for path in lease.root.rglob("*.json")}
        declared = policy(study, lease)
        bind_policy(lease, declared)
        prepared = prepare_retirement(lease, declared)
        result = commit_retirement(lease, declared, prepared)
        assert [item["accepted_updates"] for item in result["removed"]] == [0, 3]
        assert result["payload_absence_verified"] is True and result["quota_reclaim_verified"] is False
        for update, record in enumerate(records):
            directory = Path(record["directory"])
            assert (directory / "COMPLETE.json").exists()
            assert (directory / "checkpoint.pt").exists() is (update in (1, 2, 4, 5))
        assert export.read_bytes() == b"inference export must survive"
        assert all(file_sha256(path) == digest for path, digest in original.items())
        resumed = load_checkpoint(lease, records[-1]["directory"], expected_sha256=records[-1]["checkpoint_sha256"],
                                  model=model, optimizer=optimizer, scheduler=None, execution_scope="synthetic_cpu_only")
        assert resumed["accepted_updates"] == 5
        legacy_plan = retention_plan(lease)
        assert legacy_plan["latest"] == records[5]["directory"] and legacy_plan["previous"] == records[4]["directory"]
        assert prepare_retirement(lease, declared) is None
        later = snapshot(lease, model, optimizer, 6)
        follow = prepare_retirement(lease, declared)
        second = commit_retirement(lease, declared, follow)
        assert [item["accepted_updates"] for item in second["removed"]] == [4]
        assert load_checkpoint(lease, later["directory"], expected_sha256=later["checkpoint_sha256"],
                               model=model, optimizer=optimizer, scheduler=None,
                               execution_scope="synthetic_cpu_only")["accepted_updates"] == 6
        with pytest.raises(FileExistsError, match="already completed"):
            commit_retirement(lease, declared, follow)
        assert len(list((lease.root / "claims").iterdir())) == 1  # live claim, preserved


def test_requires_active_exact_bound_policy_and_never_accepts_historical_root(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        declared = policy(study, lease)
        with pytest.raises(FileNotFoundError):
            prepare_retirement(lease, declared)
        bind_policy(lease, declared)
        with pytest.raises(ValueError, match="identity"):
            bind_policy(lease, replace(declared, trajectory_identity_sha256="d" * 64))
        with pytest.raises(ValueError, match="immutable"):
            bind_policy(lease, replace(declared, explicit_milestone_updates=(1, 2)))
        other = tmp_path / "historical_credo"
        other.mkdir()
        with pytest.raises(ValueError, match="new-study"):
            RetentionPolicy(str(other), str(lease.root), identity_sha256(lease.identity), ())
        with pytest.raises(ValueError, match="new-study"):
            RetentionPolicy(str(study), str(study), identity_sha256(lease.identity), ())
    with pytest.raises(RuntimeError, match="live"):
        prepare_retirement(lease, declared)


def test_hash_change_fails_before_preparation_or_deletion(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        declared = policy(study, lease)
        bind_policy(lease, declared)
        path = Path(records[0]["directory"]) / "checkpoint.pt"
        original = bytearray(path.read_bytes())
        original[-1] ^= 1
        path.write_bytes(original)
        with pytest.raises(ValueError, match="SHA-256"):
            prepare_retirement(lease, declared)
        assert not list((lease.root / "retention/transactions").iterdir())
        assert all((Path(record["directory"]) / "checkpoint.pt").exists() for record in records)


def test_changed_inventory_and_forged_plan_cannot_delete_latest_or_milestone(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        declared = policy(study, lease)
        bind_policy(lease, declared)
        prepared = prepare_retirement(lease, declared)
        filename = lease.root / "retention/transactions" / prepared.transaction / "PREPARED.json"
        record = json.loads(filename.read_text())
        record["candidates"][0]["relative_uri"] = "checkpoints/update-000000005/checkpoint.pt"
        filename.write_text(json.dumps(record))  # malicious synthetic alteration
        forged = replace(prepared, prepared_sha256=file_sha256(filename))
        with pytest.raises(ValueError, match="inventory"):
            commit_retirement(lease, declared, forged)
        assert all((Path(item["directory"]) / "checkpoint.pt").exists() for item in records)
        with pytest.raises(RuntimeError, match="explicit recovery"):
            prepare_retirement(lease, declared)


def test_new_checkpoint_after_prepare_invalidates_plan_without_deleting(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        declared = policy(study, lease)
        bind_policy(lease, declared)
        prepared = prepare_retirement(lease, declared)
        snapshot(lease, model, optimizer, 6)
        with pytest.raises(ValueError, match="inventory"):
            commit_retirement(lease, declared, prepared)
        assert all((Path(item["directory"]) / "checkpoint.pt").exists() for item in records)


def test_partial_deletion_is_preserved_and_never_automatically_recovered(tmp_path, monkeypatch):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        declared = policy(study, lease)
        bind_policy(lease, declared)
        prepared = prepare_retirement(lease, declared)
        original = retention._unlink_exact
        calls = []

        def fail_second(root, item):
            calls.append(item["accepted_updates"])
            if len(calls) == 2:
                raise OSError("fabricated unlink failure")
            return original(root, item)

        monkeypatch.setattr(retention, "_unlink_exact", fail_second)
        with pytest.raises(OSError, match="fabricated"):
            commit_retirement(lease, declared, prepared)
        assert calls == [0, 3]
        assert not (Path(records[0]["directory"]) / "checkpoint.pt").exists()
        assert (Path(records[3]["directory"]) / "checkpoint.pt").exists()
        assert all((Path(record["directory"]) / "COMPLETE.json").exists() for record in records)
        transaction = lease.root / "retention/transactions" / prepared.transaction
        assert (transaction / "PREPARED.json").exists() and not (transaction / "DELETED.json").exists()
        with pytest.raises(RuntimeError, match="explicit recovery"):
            prepare_retirement(lease, declared)
        with pytest.raises(FileNotFoundError):
            commit_retirement(lease, declared, prepared)
        assert calls == [0, 3]  # failed replay never invokes unlink again


def test_symlink_payload_and_missing_required_milestone_fail_closed(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        declared = policy(study, lease)
        bind_policy(lease, declared)
        path = Path(records[0]["directory"]) / "checkpoint.pt"
        path.unlink()  # disposable fixture setup, not actual project cleanup
        path.symlink_to(Path(records[5]["directory"]) / "checkpoint.pt")
        with pytest.raises(ValueError, match="Symlink"):
            prepare_retirement(lease, declared)
        assert (Path(records[5]["directory"]) / "checkpoint.pt").exists()


def test_malformed_completion_and_unrecorded_absence_are_rejected(tmp_path):
    study, model, optimizer = setup(tmp_path)
    with TrajectoryLease(study / "trajectory", identity(), "fixture-worker") as lease:
        records = populate(lease, model, optimizer)
        declared = policy(study, lease)
        bind_policy(lease, declared)
        receipt = Path(records[1]["directory"]) / "COMPLETE.json"
        record = json.loads(receipt.read_text())
        record["milestone"] = "false"
        receipt.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="Malformed"):
            prepare_retirement(lease, declared)
