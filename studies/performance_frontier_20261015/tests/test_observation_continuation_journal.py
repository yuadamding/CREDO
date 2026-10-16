"""Fabricated metadata only: no checkpoint tensors, GPU, optimizer or remote I/O."""
import copy
import json

import pytest

from frontier.coordination import AdmissionJournal, WorkBinding, GIB, _digest
from frontier.planner import INNER_PAIRS, OUTER_TRIPLES


def reserve(journal, task, donors, output=4 * GIB):
    return journal.reserve(task, worker_id=task, kind="representation", scientific_updates=20000,
        output_bytes=output, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
        fileset_headroom_bytes=500 * GIB, work_binding=WorkBinding.representation(donors))


def failed(journal, task="parent", donors=INNER_PAIRS[0], with_replacement=False):
    if with_replacement:
        reserve(journal, task + "-v1", donors)
        journal.replace_zero_update_representation(task + "-v1", task,
            authority_sha256="e" * 64, zero_update_evidence_sha256="f" * 64,
            admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
            fileset_headroom_bytes=500 * GIB)
    else:
        reserve(journal, task, donors)
    return journal.reconcile(task, status="failed", exact_scheduler_id=task + "-job",
                             evidence_sha256="c" * 64)


def continuation(journal, old="parent", new="child", **overrides):
    args = dict(accepted_updates=1280, authority_sha256="d" * 64,
        failure_evidence_sha256="c" * 64, checkpoint_sha256="e" * 64,
        recovery_completion_sha256="f" * 64, retained_parent_bytes=GIB,
        retained_inventory_sha256="0" * 64, admission_sha256="1" * 64,
        capacity_receipt_sha256="2" * 64, fileset_headroom_bytes=500 * GIB,
        output_bytes=3 * GIB)
    args.update(overrides)
    return journal.continue_failed_representation(old, new, **args)


def test_four_remaining_continuations_preserve_history_and_total_budget(tmp_path):
    root = tmp_path / "journal"
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        for i, donors in enumerate(INNER_PAIRS[:4]):
            failed(journal, f"parent{i}", donors, with_replacement=True)
        prior_events = {p.name: p.read_bytes() for p in (root / "events").iterdir()}
        for i in range(4):
            state = continuation(journal, f"parent{i}", f"child{i}")
        tasks = state["tasks"]
        counted = [item for item in tasks.values() if "replacement_task_id" not in item]
        assert sum(t["scientific_update_reservation"] for t in counted) == 80000
        assert sum(tasks[f"parent{i}"]["scientific_update_reservation"] for i in range(4)) == 5120
        assert sum(tasks[f"child{i}"]["scientific_update_reservation"] for i in range(4)) == 74880
        assert sum(t["status"] == "reserved" for t in tasks.values()) == 4
        # The historical v1 outputs also remain reserved; no deletion is claimed.
        assert sum(t["storage_reserved_bytes"] for t in tasks.values()) == 32 * GIB
        for i in range(4):
            parent, child = tasks[f"parent{i}"], tasks[f"child{i}"]
            assert parent["status"] == "failed"
            assert parent["scheduler_id"] == f"parent{i}-job"
            assert parent["evidence_sha256"] == "c" * 64
            assert parent["storage_reserved_bytes"] + child["storage_reserved_bytes"] == 4 * GIB
            assert parent["work_binding"]["target_updates"] == child["work_binding"]["start_updates"] == 1280
            assert parent["original_work_binding"] == tasks[f"parent{i}-v1"]["work_binding"]
        assert all((root / "events" / name).read_bytes() == value for name, value in prior_events.items())
        with pytest.raises(ValueError, match="Concurrent"):
            reserve(journal, "eleventh-live", INNER_PAIRS[4])
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        assert journal.snapshot() == state


@pytest.mark.parametrize("override", [
    {"accepted_updates": 0}, {"accepted_updates": 1279}, {"accepted_updates": 1281},
    {"accepted_updates": 1280.0}, {"accepted_updates": True},
    {"authority_sha256": ""}, {"failure_evidence_sha256": "d" * 64},
    {"checkpoint_sha256": "not-a-sha"}, {"recovery_completion_sha256": ""},
    {"retained_inventory_sha256": ""}, {"retained_parent_bytes": 0},
    {"retained_parent_bytes": 4 * GIB}, {"retained_parent_bytes": GIB + 1},
    {"output_bytes": 0}, {"output_bytes": 3 * GIB + 1},
    {"fileset_headroom_bytes": 19 * GIB - 1},
])
def test_scope_and_combined_storage_fail_without_append(tmp_path, override):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal)
        before = journal.snapshot()
        with pytest.raises(ValueError):
            continuation(journal, **override)
        assert journal.snapshot() == before


@pytest.mark.parametrize("status", ["reserved", "submitted", "pending", "running", "uncertain", "complete", "cancelled"])
def test_only_exact_failed_known_scheduler_parent_admitted(tmp_path, status):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        reserve(journal, "parent", INNER_PAIRS[0])
        if status != "reserved":
            journal.reconcile("parent", status=status, exact_scheduler_id="job", evidence_sha256="c" * 64)
        before = journal.snapshot()
        with pytest.raises(ValueError):
            continuation(journal)
        assert journal.snapshot() == before


def test_no_unknown_scheduler_or_qualification_conversion(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        reserve(journal, "parent", INNER_PAIRS[0])
        journal.reconcile("parent", status="failed", exact_scheduler_id=None, evidence_sha256="c" * 64)
        with pytest.raises(ValueError):
            continuation(journal)
        journal.reserve("check", worker_id="check", kind="qualification", scientific_updates=0,
            output_bytes=4 * GIB, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
            fileset_headroom_bytes=500 * GIB, work_binding=WorkBinding.other("qualification", "3" * 64))
        journal.reconcile("check", status="failed", exact_scheduler_id="check-job", evidence_sha256="c" * 64)
        with pytest.raises(ValueError):
            continuation(journal, "check", "illegal-training")


def test_clone_and_further_recovery_are_not_implicit_retries(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal, with_replacement=True)
        continuation(journal)
        before = journal.snapshot()
        for old, new in (("parent", "clone"), ("parent", "child"), ("parent-v1", "old-clone")):
            with pytest.raises(ValueError):
                continuation(journal, old, new)
        with pytest.raises(ValueError):
            reserve(journal, "alias", INNER_PAIRS[0])
        journal.reconcile("child", status="failed", exact_scheduler_id="child-job", evidence_sha256="c" * 64)
        with pytest.raises(ValueError):
            continuation(journal, "child", "second-recovery")
        with pytest.raises(ValueError):
            journal.replace_zero_update_representation("child", "zero-retry",
                authority_sha256="d" * 64, zero_update_evidence_sha256="f" * 64,
                admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
                fileset_headroom_bytes=500 * GIB)
        assert len(journal.snapshot()["tasks"]) == len(before["tasks"])


def test_ten_subsets_still_200000_updates_and_no_extra_fit(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        for i, donors in enumerate(INNER_PAIRS + OUTER_TRIPLES):
            if i == 0:
                failed(journal, "parent", donors, with_replacement=True)
            else:
                reserve(journal, f"done{i}", donors)
                journal.reconcile(f"done{i}", status="complete", exact_scheduler_id=f"j{i}",
                                  evidence_sha256="c" * 64)
        state = continuation(journal)
        assert sum(t["scientific_update_reservation"] for t in state["tasks"].values()
                   if "replacement_task_id" not in t) == 200000
        with pytest.raises(ValueError):
            reserve(journal, "duplicate-subset", INNER_PAIRS[0])


def test_verified_parent_inventory_not_double_charged_against_fresh_headroom(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal)
        state = continuation(journal, fileset_headroom_bytes=19 * GIB)
        assert sum(t["storage_reserved_bytes"] for t in state["tasks"].values()) == 4 * GIB


def test_terminal_child_still_charges_consumed_work_and_retained_outputs(tmp_path):
    root = tmp_path / "journal"
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        failed(journal, with_replacement=True)
        continuation(journal)
        journal.reconcile("child", status="complete", exact_scheduler_id="child-job", evidence_sha256="9" * 64)
        state = journal.release_storage("child", retained_bytes=2 * GIB, inventory_receipt_sha256="8" * 64)
        assert state["tasks"]["parent"]["status"] == "failed"
        assert state["tasks"]["parent"]["storage_reserved_bytes"] == GIB
        assert sum(t["scientific_update_reservation"] for t in state["tasks"].values()
                   if "replacement_task_id" not in t) == 20000
        assert sum(t["storage_reserved_bytes"] for t in state["tasks"].values()) == 7 * GIB
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        assert journal.snapshot() == state


def test_remaining_work_cannot_free_or_expand_study_storage_budget(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal)
        # Other retained artifacts consume the rest of the authorized384GiB.
        journal.reserve("retained-check", worker_id="check", kind="qualification", scientific_updates=0,
            output_bytes=380 * GIB, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
            fileset_headroom_bytes=500 * GIB, work_binding=WorkBinding.other("qualification", "3" * 64))
        journal.reconcile("retained-check", status="complete", exact_scheduler_id="check-job",
                          evidence_sha256="c" * 64)
        state = continuation(journal)
        assert sum(t["storage_reserved_bytes"] for t in state["tasks"].values()) == 384 * GIB
        with pytest.raises(ValueError, match="Aggregate live-output"):
            reserve(journal, "overflow", INNER_PAIRS[1], output=1)


@pytest.mark.parametrize("damage", ["parent_failure", "checkpoint", "original_binding", "child_work", "parent_storage",
                                   "child_storage", "parent_link", "child_link", "unrelated"])
def test_forged_continuation_or_transition_fails_closed(tmp_path, damage):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal, with_replacement=True)
        before = journal.snapshot()
        state = continuation(journal)
        tasks = copy.deepcopy(state["tasks"])
        action = dict(type="authorized_remaining_observation_continuation", task_id="child", predecessor_id="parent",
                      continuation_binding=copy.deepcopy(tasks["parent"]["continuation_binding"]))
        if damage == "parent_failure":
            tasks["parent"]["evidence_sha256"] = "9" * 64
        elif damage == "checkpoint":
            tasks["child"]["continuation_binding"]["checkpoint_sha256"] = "9" * 64
        elif damage == "original_binding":
            tasks["parent"]["original_work_binding"]["target_updates"] = 19000
        elif damage == "child_work":
            tasks["child"]["work_binding"]["start_updates"] = 0
            tasks["child"]["scientific_update_reservation"] = 20000
        elif damage == "parent_storage":
            tasks["parent"]["storage_reserved_bytes"] = GIB + 1
        elif damage == "child_storage":
            tasks["child"]["storage_reserved_bytes"] = 3 * GIB + 1
        elif damage == "parent_link":
            del tasks["parent"]["continuation_task_id"]
        elif damage == "child_link":
            tasks["child"]["continues_task_id"] = "parent-v1"
        else:
            tasks["parent-v1"]["storage_reserved_bytes"] -= 1
        with pytest.raises(ValueError):
            journal._validate(tasks)
            journal._validate_transition(before["tasks"], tasks, action)


def test_replay_rejects_hash_resealed_rewrite_of_original_failure(tmp_path):
    root = tmp_path / "journal"
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        failed(journal)
        state = continuation(journal)
    path = root / "events" / f"{state['sequence']:08d}.json"
    event = json.loads(path.read_text())
    event["tasks"]["parent"]["admission_sha256"] = "9" * 64
    path.write_text(json.dumps(event))  # Deliberate disposable fixture corruption.
    (root / "intents" / path.name).write_text(json.dumps(
        {"sequence": state["sequence"], "event_sha256": _digest(event)}))
    with pytest.raises(ValueError, match="retained failure"):
        with AdmissionJournal(root, study_identity_sha256="a" * 64):
            pass
