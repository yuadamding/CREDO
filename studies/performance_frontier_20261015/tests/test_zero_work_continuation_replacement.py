"""Explicit one-generation replacement accounting; fabricated metadata only."""
from copy import deepcopy
import json

import pytest

from frontier.coordination import AdmissionJournal, GIB, WorkBinding, _digest
from frontier.planner import INNER_PAIRS, OUTER_TRIPLES
from test_observation_continuation_journal import failed, continuation, reserve


def failed_continuation(journal, suffix="", donors=INNER_PAIRS[0]):
    parent, child = "parent" + suffix, "child" + suffix
    failed(journal, parent, donors, with_replacement=True)
    continuation(journal, parent, child)
    return journal.reconcile(child, status="failed", exact_scheduler_id=child + "-job",
                             evidence_sha256="9" * 64)


def replace(journal, old="child", new="replacement", **overrides):
    args = dict(authority_sha256="8" * 64, zero_update_evidence_sha256="9" * 64,
        admission_sha256="1" * 64, capacity_receipt_sha256="7" * 64,
        fileset_headroom_bytes=500 * GIB)
    args.update(overrides)
    return journal.replace_failed_continuation(old, new, **args)


def counted(tasks):
    return sum(item["scientific_update_reservation"] for item in tasks.values()
               if not {"replacement_task_id", "continuation_replacement_task_id"} & set(item))


def test_four_replacements_preserve_parent_failure_history_and_work_storage(tmp_path):
    root = tmp_path / "journal"
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        for i, donors in enumerate(INNER_PAIRS[:4]):
            failed_continuation(journal, str(i), donors)
        before = journal.snapshot()
        events = {path.name: path.read_bytes() for path in (root / "events").iterdir()}
        for i in range(4):
            state = replace(journal, "child" + str(i), "replacement" + str(i))
        tasks = state["tasks"]
        assert counted(tasks) == counted(before["tasks"]) == 80000
        assert sum(tasks["parent" + str(i)]["scientific_update_reservation"] for i in range(4)) == 5120
        assert sum(tasks["replacement" + str(i)]["scientific_update_reservation"] for i in range(4)) == 74880
        assert sum(t["storage_reserved_bytes"] for t in tasks.values()) == sum(
            t["storage_reserved_bytes"] for t in before["tasks"].values()) == 32 * GIB
        assert sum(t["status"] == "reserved" for t in tasks.values()) == 4
        for i in range(4):
            parent, old, new = "parent" + str(i), "child" + str(i), "replacement" + str(i)
            assert tasks[parent] == before["tasks"][parent]
            assert tasks[parent + "-v1"] == before["tasks"][parent + "-v1"]
            assert tasks[parent]["continuation_task_id"] == old
            assert tasks[old]["status"] == "failed"
            assert tasks[old]["evidence_sha256"] == before["tasks"][old]["evidence_sha256"]
            assert tasks[old]["scheduler_id"] == before["tasks"][old]["scheduler_id"]
            assert tasks[old]["work_binding"] == before["tasks"][old]["work_binding"] == tasks[new]["work_binding"]
            assert tasks[old]["scientific_update_reservation"] == 18720  # Historical reservation remains visible.
            assert tasks[old]["storage_reserved_bytes"] == 0
            assert tasks[new]["storage_reserved_bytes"] == 3 * GIB
            assert tasks[new]["continues_task_id"] == parent
            assert tasks[new]["continuation_binding"] == before["tasks"][old]["continuation_binding"]
            assert tasks[new]["zero_update_evidence_sha256"] == tasks[old]["evidence_sha256"]
        assert all((root / "events" / name).read_bytes() == raw for name, raw in events.items())
        with pytest.raises(ValueError, match="Concurrent"):
            reserve(journal, "fifth-live", INNER_PAIRS[4])
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        assert journal.snapshot() == state


@pytest.mark.parametrize("kwargs", [
    {"authority_sha256": ""}, {"authority_sha256": "d" * 64},
    {"zero_update_evidence_sha256": "c" * 64}, {"zero_update_evidence_sha256": ""},
    {"admission_sha256": "2" * 64}, {"capacity_receipt_sha256": ""},
    {"fileset_headroom_bytes": -1}, {"fileset_headroom_bytes": True},
    {"fileset_headroom_bytes": 23 * GIB - 1},
])
def test_bad_authority_failure_identity_or_capacity_is_not_appended(tmp_path, kwargs):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed_continuation(journal)
        before = journal.snapshot()
        with pytest.raises(ValueError):
            replace(journal, **kwargs)
        assert journal.snapshot() == before


@pytest.mark.parametrize("status", ["reserved", "submitted", "pending", "running", "uncertain", "complete", "cancelled"])
def test_only_known_failed_continuation_admitted(tmp_path, status):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal, with_replacement=True)
        continuation(journal)
        if status != "reserved":
            journal.reconcile("child", status=status, exact_scheduler_id="job", evidence_sha256="9" * 64)
        before = journal.snapshot()
        with pytest.raises(ValueError):
            replace(journal)
        assert journal.snapshot() == before


def test_failed_parent_original_or_unknown_scheduler_cannot_be_reclassified(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed(journal, with_replacement=True)
        with pytest.raises(ValueError):
            replace(journal, "parent", "illegal-parent-retry", zero_update_evidence_sha256="c" * 64)
        with pytest.raises(ValueError):
            replace(journal, "parent-v1", "illegal-original-retry")
        continuation(journal)
        journal.reconcile("child", status="failed", exact_scheduler_id=None, evidence_sha256="9" * 64)
        with pytest.raises(ValueError):
            replace(journal)


def test_duplicate_alias_and_second_replacement_fail_closed(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed_continuation(journal)
        replace(journal)
        before = journal.snapshot()
        with pytest.raises(ValueError):
            replace(journal)
        with pytest.raises(ValueError):
            replace(journal, new="second-alias")
        with pytest.raises(ValueError):
            journal.reserve("unlinked-alias", worker_id="alias", kind="representation", scientific_updates=18720,
                output_bytes=3 * GIB, admission_sha256="1" * 64, capacity_receipt_sha256="7" * 64,
                fileset_headroom_bytes=500 * GIB, work_binding=WorkBinding.representation(INNER_PAIRS[0], 20000, 1280))
        assert journal.snapshot() == before
        journal.reconcile("replacement", status="failed", exact_scheduler_id="replacement-job", evidence_sha256="9" * 64)
        with pytest.raises(ValueError):
            replace(journal, "replacement", "third-attempt")


def test_old_output_reservation_cannot_be_reexpanded_after_release(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed_continuation(journal)
        journal.release_storage("child", retained_bytes=100, inventory_receipt_sha256="6" * 64)
        with pytest.raises(ValueError, match="residual-storage"):
            replace(journal)


def test_exact_headroom_and_ten_subsets_keep_original_limits(tmp_path):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed_continuation(journal)
        state = replace(journal, fileset_headroom_bytes=23 * GIB)
        journal.reconcile("replacement", status="complete", exact_scheduler_id="new-job", evidence_sha256="6" * 64)
        for i, donors in enumerate((INNER_PAIRS + OUTER_TRIPLES)[1:], 1):
            reserve(journal, "other" + str(i), donors)
            state = journal.reconcile("other" + str(i), status="complete", exact_scheduler_id="other-job" + str(i),
                                      evidence_sha256="5" * 64)
        assert counted(state["tasks"]) == 200000
        with pytest.raises(ValueError):
            reserve(journal, "eleventh", INNER_PAIRS[0])


@pytest.mark.parametrize("damage", ["failure", "zero_proof", "authority", "source_parent", "binding", "work",
    "retained_output", "child_output", "parent_rewire", "original_parent", "orphan", "second_generation"])
def test_forged_state_or_transition_is_rejected(tmp_path, damage):
    with AdmissionJournal(tmp_path / "journal", study_identity_sha256="a" * 64) as journal:
        failed_continuation(journal)
        before = journal.snapshot()
        state = replace(journal)
        tasks = deepcopy(state["tasks"])
        action = dict(type="authorized_zero_new_update_continuation_replacement", task_id="replacement",
                      predecessor_id="child", authority_sha256="8" * 64, evidence_sha256="9" * 64)
        if damage == "failure":
            tasks["child"]["evidence_sha256"] = "7" * 64
        elif damage == "zero_proof":
            tasks["replacement"]["zero_update_evidence_sha256"] = "7" * 64
        elif damage == "authority":
            tasks["replacement"]["replacement_authority_sha256"] = "7" * 64
        elif damage == "source_parent":
            tasks["replacement"]["continues_task_id"] = "parent-v1"
        elif damage == "binding":
            tasks["replacement"]["continuation_binding"]["checkpoint_sha256"] = "7" * 64
        elif damage == "work":
            tasks["replacement"]["work_binding"] = WorkBinding.representation(INNER_PAIRS[0]).to_dict()
            tasks["replacement"]["scientific_update_reservation"] = 20000
        elif damage == "retained_output":
            tasks["child"]["storage_reserved_bytes"] = 1
        elif damage == "child_output":
            tasks["replacement"]["storage_reserved_bytes"] = 3 * GIB + 1
        elif damage == "parent_rewire":
            tasks["parent"]["continuation_task_id"] = "replacement"
        elif damage == "original_parent":
            tasks["parent"]["admission_sha256"] = "7" * 64
        elif damage == "orphan":
            del tasks["child"]["continuation_replacement_task_id"]
        else:
            tasks["replacement"]["continuation_replacement_task_id"] = "child"
        with pytest.raises(ValueError):
            journal._validate(tasks)
            journal._validate_transition(before["tasks"], tasks, action)


def test_replay_rejects_resealed_history_rewrite(tmp_path):
    root = tmp_path / "journal"
    with AdmissionJournal(root, study_identity_sha256="a" * 64) as journal:
        failed_continuation(journal)
        state = replace(journal)
    path = root / "events" / f"{state['sequence']:08d}.json"
    event = json.loads(path.read_bytes())
    event["tasks"]["parent"]["capacity_receipt_sha256"] = "f" * 64
    path.write_text(json.dumps(event))  # Intentional disposable fixture corruption.
    (root / "intents" / path.name).write_text(json.dumps(
        {"sequence": state["sequence"], "event_sha256": _digest(event)}))
    with pytest.raises(ValueError, match="retained scientific parent"):
        with AdmissionJournal(root, study_identity_sha256="a" * 64):
            pass
