"""Capacity packaging/lifecycle tests with synthetic outputs, no remote access."""
# ruff: noqa: E402
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]
import pytest
import qualify_launch as builder
import qualify_operate as operator
import qualify_control as controller
from qualify_worker import validate_plan
from prepare_launch import canonical


@pytest.fixture(scope="module")
def payload():
    return builder.assemble()


def test_exact_capacity_scope_and_worker_imports(payload):
    plan = json.loads(payload["PLAN.json"])
    validate_plan(plan)
    assert "prepare_worker.py" in payload
    assert "preparation_control.py" in payload
    assert "DEVELOPMENT_INPUTS.json" in payload
    assert plan["scientific_optimizer_updates"] == 0
    assert not plan["D1_endpoint_access"]
    assert not plan["D2_scores_computed"]
    assert not plan["optimizer_state_memory_qualified"]
    manifest = json.loads(payload["gpu-job.json"])
    assert manifest["metadata"]["name"] == builder.RUN_ID
    assert manifest["spec"]["suspend"] is True
    assert manifest["spec"]["backoffLimit"] == 0
    assert "activeDeadlineSeconds" not in manifest["spec"]
    pod = manifest["spec"]["template"]["spec"]
    assert "activeDeadlineSeconds" not in pod
    assert pod["nodeSelector"]["gpu-type"] == "H100"
    worker = pod["containers"][0]
    assert worker["command"][-1] == builder.HOME_RUN + "/bundle/worker.py"
    env = {row["name"]: row.get("value") for row in worker["env"]}
    assert env["CREDO_PLAN_SHA256"] == hashlib.sha256(payload["PLAN.json"]).hexdigest()
    assert env["CREDO_RUN_ID"] == builder.RUN_ID
    assert env["PYTHONPATH"] == builder.HOME_RUN + "/bundle/source/src"
    assert not pod["automountServiceAccountToken"]
    for key in ("requests", "limits"):
        assert worker["resources"][key] == {"cpu": "24", "memory": "200Gi", "nvidia.com/gpu": "1",
                                            "ephemeral-storage": "2Gi" if key == "requests" else "8Gi"}


def test_no_historical_scope_or_implementation_drift(payload):
    plan = json.loads(payload["PLAN.json"])
    assert plan["control_root"] == builder.CONTROL_ROOT
    assert plan["worker_output"] == builder.HOME_RUN + "/output"
    assert builder.RUN_ID.encode() in payload["publication.py"]
    assert builder.base.RUN_ID.encode() not in payload["publication.py"]
    hashes = json.loads(payload["IMPLEMENTATION.json"])
    assert hashes == {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()
                      if name.endswith(".py") and name not in {"control.py", "publication.py"}}
    assert plan["worker_sha256"] == hashlib.sha256(payload["worker.py"]).hexdigest()


def test_build_readback_and_no_overwrite(tmp_path):
    destination = tmp_path / "capacity"
    result = builder.build(destination)
    assert result["run_id"] == builder.RUN_ID
    assert result["archive_sha256"] == hashlib.sha256((destination / "bundle.tar.gz").read_bytes()).hexdigest()
    for name, expected in result["files"].items():
        raw = (destination / "bundle" / name).read_bytes()
        assert expected == dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    with pytest.raises(FileExistsError):
        builder.build(destination)


@pytest.mark.parametrize("field,value", [("parameter_updates", 1), ("automatic_retries", 1),
                                        ("D1_endpoint_access", True), ("LARRY_access", True)])
def test_changed_workload_rejected(payload, field, value):
    plan = json.loads(payload["PLAN.json"])
    plan[field] = value
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_capacity_completion_has_its_own_terminal_validator(tmp_path, monkeypatch, payload):
    plan = json.loads(payload["PLAN.json"])
    plan.update(worker_output=str(tmp_path / "output"), maximum_output_bytes=1024**2)
    output = Path(plan["worker_output"])
    output.mkdir()
    (tmp_path / "PLAN.json").write_bytes(canonical(plan))
    (output / "STARTUP.json").write_bytes(canonical({"status": "startup_guard_passed"}))
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    owned = [{"metadata": {"uid": "pod-uid"}}]
    (tmp_path / "startup-validated.json").write_bytes(canonical(dict(uid="job-uid", startup_sha256=sha(output / "STARTUP.json"), actual_pod_audit_passed=True)))
    result = dict(status="qualified_full_ddr_capacity", run_id=plan["run_id"], plan_sha256=sha(tmp_path / "PLAN.json"),
        pod_uid="pod-uid", parameter_updates=0, scientific_optimizer_updates=0, D1_endpoint_access=False,
        LARRY_access=False, D2_scores_computed=False, scientific_training_complete=False,
        results=[dict(candidate_id=row["candidate_id"], probes=[dict(parameter_updates=0, particles=512, steps=8,
                        peak_allocated_bytes=1, peak_reserved_bytes=2)]) for row in plan["candidates"]],
        files=[dict(path="STARTUP.json", sha256=sha(output / "STARTUP.json"), size_bytes=(output / "STARTUP.json").stat().st_size)],
        output_bytes=(output / "STARTUP.json").stat().st_size)
    (output / "COMPLETE.json").write_bytes(canonical(result))
    def checked(path, retain=False):
        raw = Path(path).read_bytes()
        return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)), raw if retain else b""
    monkeypatch.setattr(controller.base, "checked_file", checked)
    monkeypatch.setattr(controller.base, "BASE", tmp_path)
    monkeypatch.setattr(controller.base, "core", SimpleNamespace(sha=sha), raising=False)
    original_stat = Path.lstat
    def owned_stat(path):
        s = original_stat(path)
        return SimpleNamespace(st_uid=307469, st_mode=s.st_mode)
    with patch.object(Path, "lstat", owned_stat):
        assert controller.validate_result(plan, {"uid": "job-uid"}, tmp_path, owned)
        result["parameter_updates"] = 1
        (output / "COMPLETE.json").write_bytes(canonical(result))
        with pytest.raises(RuntimeError, match="scope differs"):
            controller.validate_result(plan, {"uid": "job-uid"}, tmp_path, owned)


def test_operator_source_separates_new_attempt_from_preparation():
    assert operator.BASE.name == "capacity_v1"
    assert operator.TRANSFER.endswith("credo-full-ddr-capacity-20261007-v1-transfer")
    assert operator.HOME_RUN.endswith("/capacity_v1")
    assert operator.CONTROL_ROOT.endswith("/capacity_v1")
    assert copy.deepcopy(operator.RUN_ID) != builder.base.RUN_ID
