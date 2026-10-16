import json
import hashlib
import pytest

from frontier.coordination import AdmissionJournal, GIB, WorkBinding
from frontier.configuration import configurations
from frontier.planner import INNER_PAIRS, OUTER_TRIPLES, TrainingTask, TrajectoryKey


def binding(task, kind, updates):
    index = int("".join(c for c in task if c.isdigit()) or "1")
    if kind == "main_fit":
        key = TrajectoryKey(("D3", "D4"), configurations()[index % 24].configuration_id, 0)
        return WorkBinding.main(TrainingTask(key, 0, 1024, ("D1",)))
    if kind == "representation":
        return WorkBinding.representation((INNER_PAIRS + OUTER_TRIPLES)[index % 10], updates)
    return WorkBinding.other(kind, hashlib.sha256(task.encode()).hexdigest())


def reserve(journal, task="task1", worker="worker1", kind="main_fit", updates=1024, output=GIB,
            work=None):
    return journal.reserve(task, worker_id=worker, kind=kind, scientific_updates=updates,
                           output_bytes=output, admission_sha256="a" * 64,
                           capacity_receipt_sha256="b" * 64, fileset_headroom_bytes=500 * GIB,
                           work_binding=binding(task, kind, updates) if work is None else work)


def test_four_including_uncertain_and_duplicate_guard(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        for i in range(4):
            reserve(journal, f"task{i}", f"worker{i}")
        journal.reconcile("task0", status="uncertain", exact_scheduler_id=None, evidence_sha256="d" * 64)
        with pytest.raises(ValueError, match="Concurrent"):
            reserve(journal, "task4", "worker4")
        with pytest.raises(ValueError, match="already reserved"):
            reserve(journal, "task0", "worker5")
        assert len(journal.snapshot()["tasks"]) == 4
        journal.reconcile("task0", status="complete", exact_scheduler_id="job0", evidence_sha256="d" * 64)
        reserve(journal, "task4", "worker4")


def test_forecast_two_cap_and_bytes(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal, "f0", "w0", "forecast", 0, 64 * GIB)
        reserve(journal, "f1", "w1", "forecast", 0, 64 * GIB)
        with pytest.raises(ValueError, match="Concurrent"):
            reserve(journal, "f2", "w2", "forecast", 0)
        journal.reconcile("f0", status="complete", exact_scheduler_id="job0", evidence_sha256="d" * 64)
        with pytest.raises(ValueError, match="Forecast worker"):
            reserve(journal, "f2", "w2", "forecast", 0, 65 * GIB)


def test_storage_release_not_automatic_and_reopen(tmp_path):
    root = tmp_path / "ledger"
    with AdmissionJournal(root, study_identity_sha256="c" * 64) as journal:
        reserve(journal)
        journal.reconcile("task1", status="complete", exact_scheduler_id="job1", evidence_sha256="d" * 64)
        assert journal.snapshot()["tasks"]["task1"]["storage_reserved_bytes"] == GIB
        journal.release_storage("task1", retained_bytes=123, inventory_receipt_sha256="e" * 64)
    with AdmissionJournal(root, study_identity_sha256="c" * 64) as journal:
        assert journal.snapshot()["tasks"]["task1"]["storage_reserved_bytes"] == 123
        with pytest.raises(ValueError, match="Terminal"):
            journal.reconcile("task1", status="running", exact_scheduler_id="job1", evidence_sha256="d" * 64)


def test_lock_and_identity_mismatch(tmp_path):
    root = tmp_path / "ledger"
    with AdmissionJournal(root, study_identity_sha256="c" * 64):
        with pytest.raises(BlockingIOError):
            with AdmissionJournal(root, study_identity_sha256="c" * 64):
                pass
    with pytest.raises(ValueError, match="identity changed"):
        with AdmissionJournal(root, study_identity_sha256="d" * 64):
            pass


def test_corrupt_chain_fails_closed(tmp_path):
    root = tmp_path / "ledger"
    with AdmissionJournal(root, study_identity_sha256="c" * 64) as journal:
        reserve(journal)
        journal.reconcile("task1", status="pending", exact_scheduler_id="job1", evidence_sha256="d" * 64)
    first = root / "events" / "00000001.json"
    payload = json.loads(first.read_text())
    payload["action"]["type"] = "corrupted"
    first.write_text(json.dumps(payload))  # Deliberate corruption of disposable fixture.
    with pytest.raises(ValueError, match="chain mismatch"):
        with AdmissionJournal(root, study_identity_sha256="c" * 64):
            pass


def test_headroom_and_worker_and_scheduler_identity(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal)
        with pytest.raises(ValueError, match="Duplicate active"):
            reserve(journal, "task2", "worker1")
        journal.reconcile("task1", status="pending", exact_scheduler_id="job1", evidence_sha256="d" * 64)
        with pytest.raises(ValueError, match="identity changed"):
            journal.reconcile("task1", status="running", exact_scheduler_id="job2", evidence_sha256="d" * 64)
        with pytest.raises(ValueError, match="headroom"):
            journal.reserve("task2", worker_id="w2", kind="main_fit", scientific_updates=1024,
                            output_bytes=GIB, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
                            fileset_headroom_bytes=17 * GIB, work_binding=binding("task2", "main_fit", 1024))


def test_workload_reservations_and_terminal_not_recycled(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        for i in range(10):
            reserve(journal, f"r{i}", "w1", "representation", 20000, 0)
            journal.reconcile(f"r{i}", status="complete", exact_scheduler_id=f"j{i}", evidence_sha256="d" * 64)
        with pytest.raises(ValueError):
            reserve(journal, "r10", "w1", "representation", 20000, 0)
        assert sum(item["scientific_update_reservation"] for item in journal.snapshot()["tasks"].values()) == 200000


def test_cap_cannot_expand(tmp_path):
    with pytest.raises(ValueError, match="cap/reserve"):
        AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64, storage_cap_bytes=385 * GIB)


def test_verified_retained_bytes_count_once_against_fresh_headroom(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal, output=100 * GIB)
        journal.reconcile("task1", status="complete", exact_scheduler_id="job1", evidence_sha256="d" * 64)
        options = dict(worker_id="worker2", kind="main_fit", scientific_updates=1024,
                       output_bytes=GIB, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
                       fileset_headroom_bytes=20 * GIB, work_binding=binding("task2", "main_fit", 1024))
        with pytest.raises(ValueError, match="headroom"):
            journal.reserve("task2", **options)
        journal.release_storage("task1", retained_bytes=100 * GIB, inventory_receipt_sha256="e" * 64)
        journal.reserve("task2", **options)
        assert sum(row["storage_reserved_bytes"] for row in journal.snapshot()["tasks"].values()) == 101 * GIB


def test_canonical_alias_duplicate_and_exact_continuation(tmp_path):
    key = TrajectoryKey(("D3", "D4"), "F0-V0", 0)
    first = WorkBinding.main(TrainingTask(key, 0, 1024, ("D1",)))
    next_work = WorkBinding.main(TrainingTask(key, 1024, 3072, ("D2",)))
    final_work = WorkBinding.main(TrainingTask(key, 3072, 9216, ("D1", "D2")))
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal, "original", "worker0", work=first)
        with pytest.raises(ValueError, match="Duplicate, overlapping"):
            reserve(journal, "alias", "worker1", work=first)
        with pytest.raises(ValueError, match="exact completed predecessor"):
            reserve(journal, "next", "worker1", updates=2048, work=next_work)
        journal.reconcile("original", status="complete", exact_scheduler_id="job0", evidence_sha256="d" * 64)
        with pytest.raises(ValueError, match="unaligned"):
            reserve(journal, "skipped", "worker1", updates=6144, work=final_work)
        reserve(journal, "next", "worker1", updates=2048, work=next_work)
        journal.reconcile("next", status="complete", exact_scheduler_id="job1", evidence_sha256="d" * 64)
        reserve(journal, "last", "worker2", updates=6144, work=final_work)
        assert sum(item["scientific_update_reservation"] for item in journal.snapshot()["tasks"].values()) == 9216


def test_failed_canonical_work_cannot_retry_under_alias(tmp_path):
    first = binding("task0", "main_fit", 1024)
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal, "original", "w0", work=first)
        journal.reconcile("original", status="failed", exact_scheduler_id=None, evidence_sha256="d" * 64)
        with pytest.raises(ValueError, match="Duplicate, overlapping"):
            reserve(journal, "alias", "w1", work=first)
        next_work = WorkBinding.main(TrainingTask(TrajectoryKey(("D3", "D4"), "F0-V0", 0),
                                                  1024, 3072, ("D1",)))
        with pytest.raises(ValueError, match="exact completed predecessor"):
            reserve(journal, "next", "w1", updates=2048, work=next_work)


def test_exact_update_binding_and_operation_alias(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        with pytest.raises(ValueError, match="Reserved workload differs"):
            reserve(journal, updates=1)
        forecast = WorkBinding.other("forecast", "f" * 64)
        reserve(journal, "f0", "w0", "forecast", 0, work=forecast)
        with pytest.raises(ValueError, match="Duplicate canonical"):
            reserve(journal, "f1", "w1", "forecast", 0, work=forecast)


def test_representation_continuation_is_one_fit_not_two(tmp_path):
    first = WorkBinding.representation(("D3", "D4"), 10000)
    second = WorkBinding.representation(("D3", "D4"), 20000, 10000)
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal, "r0", "w0", "representation", 10000, work=first)
        journal.reconcile("r0", status="complete", exact_scheduler_id="job0", evidence_sha256="d" * 64)
        reserve(journal, "r1", "w1", "representation", 10000, work=second)
        assert sum(item["scientific_update_reservation"] for item in journal.snapshot()["tasks"].values()) == 20000


@pytest.mark.parametrize("damage", ["tail_edit", "tail_remove", "intent_only", "event_directory_missing"])
def test_append_damage_fails_closed(tmp_path, damage):
    root = tmp_path / "ledger"
    with AdmissionJournal(root, study_identity_sha256="c" * 64) as journal:
        reserve(journal)
    event = root / "events" / "00000001.json"
    if damage == "tail_edit":
        value = json.loads(event.read_text())
        value["tasks"] = {}
        event.write_text(json.dumps(value))
    elif damage in ("tail_remove", "event_directory_missing"):
        event.unlink()  # Disposable corruption fixture only.
        if damage == "event_directory_missing":
            event.parent.rmdir()
    else:
        (root / "intents" / "00000002.json").write_text("{}")
    with pytest.raises(ValueError, match="intent|Missing committed"):
        with AdmissionJournal(root, study_identity_sha256="c" * 64):
            pass


def test_replay_rejects_semantic_rewrite_even_with_matching_intent(tmp_path):
    from frontier.coordination import _digest
    root = tmp_path / "ledger"
    with AdmissionJournal(root, study_identity_sha256="c" * 64) as journal:
        reserve(journal)
        journal.reconcile("task1", status="running", exact_scheduler_id="job1", evidence_sha256="d" * 64)
    event_path = root / "events" / "00000002.json"
    event = json.loads(event_path.read_text())
    event["tasks"] = {}
    event_path.write_text(json.dumps(event))
    (root / "intents" / event_path.name).write_text(json.dumps({"sequence": 2, "event_sha256": _digest(event)}))
    with pytest.raises(ValueError, match="inventory changed"):
        with AdmissionJournal(root, study_identity_sha256="c" * 64):
            pass


def test_malformed_state_and_work_binding(tmp_path):
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        with pytest.raises(ValueError, match="Malformed task"):
            journal._validate({"bad": {"status": "running"}})
    with pytest.raises(ValueError):
        WorkBinding.main("unbound")
    with pytest.raises(ValueError):
        WorkBinding.representation(("D4", "D3"))
    with pytest.raises(ValueError):
        WorkBinding.other("main_fit", "f" * 64)


def test_global_update_and_storage_caps_remain_charged_after_completion(tmp_path):
    import copy
    with AdmissionJournal(tmp_path / "ledger", study_identity_sha256="c" * 64) as journal:
        reserve(journal)
        template = journal.snapshot()["tasks"]["task1"]
        tasks = {}
        work = [WorkBinding.main(TrainingTask(TrajectoryKey(donors, cfg.configuration_id, seed),
                                              0, 9216, tuple(d for d in ("D1", "D2", "D3", "D4")
                                                             if d not in donors)))
                for donors in OUTER_TRIPLES for cfg in configurations() for seed in (0, 1, 2)]
        for index, binding_value in enumerate(work[:93]):
            item = copy.deepcopy(template)
            item.update(status="complete", scheduler_id=f"j{index}", evidence_sha256="d" * 64,
                        scientific_update_reservation=9216, storage_reserved_bytes=0,
                        work_binding=binding_value.to_dict())
            tasks[f"t{index}"] = item
            if index == 91:
                journal._validate(tasks)  # Exactly 847872 retained charges.
        with pytest.raises(ValueError, match="Scientific update"):
            journal._validate(tasks)
        item = copy.deepcopy(template)
        item["storage_reserved_bytes"] = 385 * GIB
        with pytest.raises(ValueError, match="Aggregate live-output"):
            journal._validate({"storage": item})
