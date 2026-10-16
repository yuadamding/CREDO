"""Replacement orchestration with fabricated journals in disposable directories."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from deployment import common, continuation_build
from frontier.coordination import AdmissionJournal, WorkBinding


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actor_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "operations"))
    spec = importlib.util.spec_from_file_location("replacement_operator_fixture", ROOT / "operations/replace_observation_continuation.py")
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    local = tmp_path / ROOT.name
    local.mkdir()
    files = ("AUTHORITY.json", "OBSERVATION_CONTINUATION_V4_REPLACEMENT_AUTHORITY.json",
        "receipts/OBSERVATION_CONTINUATION_V4_REPLACEMENT_INTAKE_20261016.json",
        "receipts/OBSERVATION_CONTINUATION_V4_CAPACITY_20261016.json",
        "receipts/OBSERVATION_CONTINUATION_V3_FAILURE_CLOSURE_20261016.json")
    for name in files:
        target = local / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    for key, name in (("ROOT", "."), ("BATCH", "observation_continuation_v4_admission"),
        ("AUTHORITY", "OBSERVATION_CONTINUATION_V4_REPLACEMENT_AUTHORITY.json"),
        ("INTAKE", "receipts/OBSERVATION_CONTINUATION_V4_REPLACEMENT_INTAKE_20261016.json"),
        ("CAPACITY", "receipts/OBSERVATION_CONTINUATION_V4_CAPACITY_20261016.json"),
        ("FAILURE", "receipts/OBSERVATION_CONTINUATION_V3_FAILURE_CLOSURE_20261016.json")):
        monkeypatch.setattr(actor, key, local / name)
    proof = json.loads(actor.INTAKE.read_bytes())
    real_package = actor.ops.package
    packages = {pair: real_package(ROOT / ("observation_" + pair.lower() + "_v3_launch")) for pair in actor.PAIRS}
    parent_intake = json.loads((ROOT / "receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json").read_bytes())
    journal_args = dict(study_identity_sha256=actor.ops.sha(local / "AUTHORITY.json"))
    with AdmissionJournal(local / "admission", **journal_args) as journal:
        for pair, (path, build, plan) in packages.items():
            admission = json.loads((path / "bundle/ADMISSION.json").read_bytes())
            evidence = admission["continuation_binding"]
            donors = admission["work_binding"]["identity"][0]
            v2 = admission["continues_task_id"]
            v1 = v2.removesuffix("-v2") + "-v1"
            journal.reserve(v1, worker_id=v1, kind="representation", scientific_updates=20000,
                output_bytes=4 * common.GIB, admission_sha256=build["phase_authority_sha256"],
                capacity_receipt_sha256="b" * 64, fileset_headroom_bytes=500 * common.GIB,
                work_binding=WorkBinding.representation(donors))
            journal.replace_zero_update_representation(v1, v2,
                authority_sha256="e" * 64, zero_update_evidence_sha256="f" * 64,
                admission_sha256=build["phase_authority_sha256"], capacity_receipt_sha256="b" * 64,
                fileset_headroom_bytes=500 * common.GIB)
            journal.reconcile(v2, status="failed", exact_scheduler_id=parent_intake["fits"][pair]["terminal"]["value"]["uid"],
                              evidence_sha256=evidence["failure_evidence_sha256"])
            journal.continue_failed_representation(v2, build["run_id"], accepted_updates=1280,
                authority_sha256=evidence["authority_sha256"], failure_evidence_sha256=evidence["failure_evidence_sha256"],
                checkpoint_sha256=evidence["checkpoint_sha256"], recovery_completion_sha256=evidence["recovery_completion_sha256"],
                retained_parent_bytes=evidence["retained_parent_bytes"], retained_inventory_sha256=evidence["retained_inventory_sha256"],
                admission_sha256=build["phase_authority_sha256"], capacity_receipt_sha256="b" * 64,
                fileset_headroom_bytes=500 * common.GIB, output_bytes=evidence["child_output_cap_bytes"])
            journal.reconcile(build["run_id"], status="failed", exact_scheduler_id=proof["fits"][pair]["uid"],
                              evidence_sha256=actor.ops.sha(actor.FAILURE))
        before = journal.snapshot()
    calls = []
    class RecordingJournal(AdmissionJournal):
        def replace_failed_continuation(self, old, new, **kwargs):
            calls.append((old, new, deepcopy(kwargs)))
            return super().replace_failed_continuation(old, new, **kwargs)
    def forbidden_remote(*args, **kwargs):
        raise AssertionError("Remote operations are outside this metadata-only test")
    monkeypatch.setattr(actor, "AdmissionJournal", RecordingJournal)
    monkeypatch.setattr(actor, "record", common.publish)
    monkeypatch.setattr(actor, "remote", forbidden_remote)
    monkeypatch.setattr(actor, "capacity", forbidden_remote)
    monkeypatch.setattr(actor, "source_manifest", lambda unused: continuation_build.source_manifest(ROOT))
    monkeypatch.setattr(actor.ops, "package", lambda path: real_package(ROOT / path.name))
    return actor, calls, before, journal_args


def test_operator_reserves_four_exact_replacements_and_exports_same_budgets(actor_fixture):
    actor, calls, before, journal_args = actor_fixture
    result = actor.reserve()
    assert len(calls) == len(result["fits"]) == 4
    assert result["credited_parent_updates"] == 5120 and result["new_updates"] == 74880
    assert result["live_h100_reservations"] == 4
    proof = json.loads(actor.INTAKE.read_bytes())
    cap = json.loads(actor.CAPACITY.read_bytes())
    with AdmissionJournal(actor.ROOT / "admission", **journal_args) as journal:
        state = journal.snapshot()
    assert sum(t["storage_reserved_bytes"] for t in state["tasks"].values()) == sum(
        t["storage_reserved_bytes"] for t in before["tasks"].values())
    assert sum(t["scientific_update_reservation"] for t in state["tasks"].values()
        if not {"replacement_task_id", "continuation_replacement_task_id"} & set(t)) == 80000
    for pair, (old, new, kwargs), row in zip(actor.PAIRS, calls, result["fits"]):
        assert new == proof["fits"][pair]["new_run_id"]
        assert kwargs["authority_sha256"] == actor.ops.sha(actor.AUTHORITY)
        assert kwargs["zero_update_evidence_sha256"] == actor.ops.sha(actor.FAILURE)
        assert kwargs["capacity_receipt_sha256"] == actor.ops.sha(actor.CAPACITY)
        assert kwargs["fileset_headroom_bytes"] == min(cap["headroom_bytes"], cap["filesystem_available_bytes"])
        admission = json.loads(Path(row["path"]).read_bytes())
        assert admission["replaces_continuation_task_id"] == old
        assert admission["replacement_authority_sha256"] == kwargs["authority_sha256"]
        assert admission["zero_update_evidence_sha256"] == kwargs["zero_update_evidence_sha256"]
        assert admission["replacement_intake_sha256"] == actor.ops.sha(actor.INTAKE)
        assert admission["continuation_binding"] == state["tasks"][old]["continuation_binding"]
        assert admission["work_binding"] == state["tasks"][new]["work_binding"]
        assert admission["work_binding"]["start_updates"] == 1280
        assert admission["scientific_update_reservation"] == 18720
        assert admission["aggregate_reserved_output_bytes"] == sum(t["storage_reserved_bytes"] for t in state["tasks"].values())
        assert admission["live_h100_reservations"] == 4
        assert row["sha256"] == actor.ops.sha(row["path"])
    with pytest.raises(ValueError, match="admission already exists"):
        actor.reserve()
    assert len(calls) == 4


def test_operator_gate_rejects_changed_child_or_source_link(actor_fixture, monkeypatch):
    actor, _, _, _ = actor_fixture
    result = actor.reserve()
    original = json.loads(Path(result["fits"][0]["path"]).read_bytes())
    current = deepcopy(original)
    monkeypatch.setattr(actor.ops, "admission_gate", lambda *args: current)
    build = dict(run_id=original["run_id"])
    assert actor.gate(Path("/fixture"), build, {}) == original
    for key, value in (("replacement_authority_sha256", "0" * 64), ("replacement_intake_sha256", "0" * 64),
        ("zero_update_evidence_sha256", "0" * 64), ("continues_task_id", "wrong-original"),
        ("replaces_continuation_task_id", "wrong-failed-child"), ("continuation_binding", {})):
        current = dict(original, **{key: value})
        with pytest.raises(ValueError):
            actor.gate(Path("/fixture"), build, {})


def test_nonzero_child_work_blocks_before_new_journal_event(actor_fixture):
    actor, calls, before, journal_args = actor_fixture
    proof = json.loads(actor.INTAKE.read_bytes())
    proof["fits"]["D1D2"]["new_scientific_updates"] = 1
    actor.INTAKE.write_bytes(common.encoded(proof))  # Disposable altered evidence only.
    with pytest.raises(ValueError, match="Zero-work child"):
        actor.reserve()
    assert not calls and not actor.BATCH.exists()
    with AdmissionJournal(actor.ROOT / "admission", **journal_args) as journal:
        assert journal.snapshot() == before
