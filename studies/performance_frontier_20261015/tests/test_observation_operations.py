"""Focused local metadata checks; no SSH, scheduler, GPU, arrays or optimizer."""

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def operations():
    transport = module(ROOT / "operations/transport.py", "frontier_test_transport")
    with patch.dict(sys.modules, {"transport": transport}):
        observation = module(ROOT / "operations/observation.py", "frontier_test_observation_operations")
    return transport, observation


def test_transport_nested_quoting_and_metadata_stdin_no_execution(operations, monkeypatch):
    transport, _ = operations
    arguments = [transport.HOST_PYTHON, "-c", "print('literal $HOME; `x`')"]
    actual = transport.command(arguments)
    assert actual[:1] == ["/usr/bin/ssh"] and actual[-2] == "work-mac"
    second = shlex.split(actual[-1])
    assert second[:-2] == transport.SECOND_HOP and second[-2] == "seadragon"
    assert shlex.split(second[-1]) == arguments
    captured = []

    def no_transport(arguments, **keywords):
        captured.append((arguments, keywords))
        return 'institutional banner\nRECEIPT={"fixture":true}\n'

    monkeypatch.setattr(transport, "call", no_transport)
    assert transport.remote("print('fixture')", {"VALUE": 7}) == {"fixture": True}
    assert captured[0][0] == [transport.HOST_PYTHON, "-"]
    assert captured[0][1]["data"].startswith(b"VALUE=7\n")
    with pytest.raises(ValueError, match="Named explicit"):
        transport.remote("never executed", {"_private": 1})


def test_host_raw_metadata_scripts_are_python36_parseable(operations):
    transport, observation = operations
    for code in (transport.CAPACITY_CODE, observation.PREFLIGHT, observation.ARCHIVE_WRITER):
        ast.parse(code, feature_version=(3, 6))


def test_lifecycle_pins_verified_kubeconfig_without_inherited_environment(operations, tmp_path, monkeypatch):
    _, observation = operations
    for name in ("DEPLOYMENT.json", "BRIDGE_CLOSEOUT.json", "REMOTE_TRANSFER_CLOSEOUT.json"):
        (tmp_path / name).write_text("{}")
    plan_path = tmp_path / "bundle/PLAN.json"
    plan_path.parent.mkdir()
    plan_path.write_text("{}")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = {"bundle_root": str(plan_path.parent), "control_root": str(tmp_path / "control")}
    build = {"run_id": "fixture-only", "plan_sha256": digest}
    recorded, called = [], []
    monkeypatch.delenv("KUBECONFIG", raising=False)
    monkeypatch.setattr(observation, "record", lambda *values: recorded.append(values))
    import os
    actual_lstat = os.lstat

    def metadata_only(path, *args, **kwargs):
        if str(path) == "/rsrch8/home/bcb/yding4/.kube/config":
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=307469)
        return actual_lstat(path, *args, **kwargs)

    def no_controller(command, **kwargs):
        called.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")

    def local_script_only(code, values, **kwargs):
        ast.parse(code, feature_version=(3, 6))
        printed = io.StringIO()
        with patch.object(os, "lstat", metadata_only), patch.object(subprocess, "run", no_controller), redirect_stdout(printed):
            exec(compile(code, "fabricated-host-script", "exec"), dict(values))
        return json.loads(printed.getvalue().strip().split("RECEIPT=", 1)[1])

    monkeypatch.setattr(observation, "remote", local_script_only)
    # The same invocation block serves dryrun/create/launch. Use dryrun here
    # so no real local admission journal can be reconciled by this fixture.
    result = observation.lifecycle(tmp_path, build, plan, "dryrun")
    assert result["status"] == "action_returned" and len(called) == 1
    command, keywords = called[0]
    assert command[:3] == [observation.HOST_PYTHON, str(plan_path.parent / "control.py"), "dryrun"]
    assert keywords["env"]["KUBECONFIG"] == "/rsrch8/home/bcb/yding4/.kube/config"
    assert "timeout" not in keywords  # No scientific job runtime deadline.
    assert len(recorded) == 2


def test_admission_rejects_changed_numerical_source_before_opening_journal(operations, tmp_path):
    _, observation = operations
    (tmp_path / "bundle").mkdir()
    admission = {"source_manifest_sha256": "a" * 64, "phase_authority_sha256": "b" * 64}
    (tmp_path / "bundle/ADMISSION.json").write_text(json.dumps(admission))
    build = {"source_manifest_sha256": "c" * 64, "phase_authority_sha256": "b" * 64}
    with pytest.raises(ValueError, match="exact numerical sources"):
        observation.admission_gate(tmp_path, build, {})


def test_admission_reconciles_aggregate_reservations_and_cited_event(operations, tmp_path, monkeypatch):
    from frontier.coordination import AdmissionJournal, WorkBinding
    _, observation = operations
    monkeypatch.setattr(observation, "ROOT", tmp_path)
    (tmp_path / "AUTHORITY.json").write_text('{"fabricated_metadata_fixture":true}')
    package = tmp_path / "package"
    (package / "bundle").mkdir(parents=True)
    run_id = "fixture-qualification-a"
    binding = WorkBinding.other("qualification", "d" * 64)
    cap = 16 * 1024 ** 3
    journal_options = {"study_identity_sha256": observation.sha(tmp_path / "AUTHORITY.json")}
    with AdmissionJournal(tmp_path / "admission", **journal_options) as journal:
        state = journal.reserve(run_id, worker_id=run_id, kind="qualification", scientific_updates=0,
            output_bytes=cap, admission_sha256="b" * 64, capacity_receipt_sha256="c" * 64,
            fileset_headroom_bytes=100 * 1024 ** 3, work_binding=binding)
    admission = {"source_manifest_sha256": "a" * 64, "phase_authority_sha256": "b" * 64,
        "work_binding": binding.to_dict(), "aggregate_reserved_output_bytes": cap,
        "live_h100_reservations": 1, "journal_event_sha256": state["last_sha256"]}
    path = package / "bundle/ADMISSION.json"
    path.write_text(json.dumps(admission))
    build = {"source_manifest_sha256": "a" * 64, "phase_authority_sha256": "b" * 64, "run_id": run_id}
    plan = {"maximum_output_bytes": cap}
    assert observation.admission_gate(package, build, plan) == admission
    with AdmissionJournal(tmp_path / "admission", **journal_options) as journal:
        journal.reserve("fixture-qualification-b", worker_id="fixture-qualification-b", kind="qualification",
            scientific_updates=0, output_bytes=cap, admission_sha256="b" * 64,
            capacity_receipt_sha256="c" * 64, fileset_headroom_bytes=100 * 1024 ** 3,
            work_binding=WorkBinding.other("qualification", "e" * 64))
    with pytest.raises(ValueError, match="Stale admission"):
        observation.admission_gate(package, build, plan)
    admission.update(aggregate_reserved_output_bytes=2 * cap, live_h100_reservations=2)
    path.write_text(json.dumps(admission))
    assert observation.admission_gate(package, build, plan) == admission
    admission["journal_event_sha256"] = "f" * 64
    path.write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="actual journal event"):
        observation.admission_gate(package, build, plan)


@pytest.mark.parametrize("build_key,plan_key", [
    ("source_manifest_sha256", "source_manifest"),
    ("phase_authority_sha256", "phase_authority"),
])
@pytest.mark.parametrize("changed", ["build_digest", "plan_digest", "plan_path"])
def test_package_binds_build_payload_and_plan_references(operations, tmp_path, monkeypatch,
                                                       build_key, plan_key, changed):
    from deployment.build import archive_bytes, construct, write_package
    from deployment.common import encoded, identity
    _, observation = operations
    monkeypatch.setattr(observation, "ROOT", tmp_path)
    retained = ROOT.parent / "credo_full_ddr_gse314342_20261006_v1/comparator_replay_assessment_launch_v1/package_v1/payload/comparator_replay_assessment_v1/bundle"
    run_id = "credo-frontier-metadata-only-fixture"
    admission = {"schema": "credo.frontier.observation-admission-export.v1", "run_id": run_id,
        "status": "reserved", "storage_reserved_bytes": 16 * 1024 ** 3,
        "scientific_update_reservation": 0, "aggregate_reserved_output_bytes": 16 * 1024 ** 3,
        "live_h100_reservations": 1, "journal_event_sha256": "a" * 64}
    payload, archive, build = construct(ROOT, retained, "observation_metadata_fixture_v1", run_id, admission)
    destination = write_package(tmp_path / "metadata-package", payload, archive, build)
    assert observation.package(destination)[1] == build

    if changed == "build_digest":
        build[build_key] = "f" * 64
    else:
        plan = json.loads(payload["PLAN.json"])
        if changed == "plan_digest":
            # BUILD and PLAN agree with one another but not the actual payload.
            plan[plan_key]["sha256"] = build[build_key] = "f" * 64
        else:
            # A correct digest must not authorize a different named authority.
            plan[plan_key]["path"] += ".unexpected"
        payload["PLAN.json"] = encoded(plan)
        build["plan_sha256"] = identity(payload["PLAN.json"])["sha256"]
        inventory = json.loads(payload["INVENTORY.json"])
        inventory["PLAN.json"] = identity(payload["PLAN.json"])
        payload["INVENTORY.json"] = encoded(inventory)
        build["inventory_sha256"] = identity(payload["INVENTORY.json"])["sha256"]
        archive = archive_bytes(payload)
        build["archive_sha256"] = identity(archive)["sha256"]
        # These are disposable, explicitly fabricated local fixture files.
        (destination / "bundle/PLAN.json").write_bytes(payload["PLAN.json"])
        (destination / "bundle/INVENTORY.json").write_bytes(payload["INVENTORY.json"])
        (destination / "bundle.tar.gz").write_bytes(archive)
    (destination / "BUILD.json").write_bytes(encoded(build))
    with pytest.raises(ValueError, match="BUILD/PLAN/payload binding differs"):
        observation.package(destination)

