"""Synthetic recovery packaging/call-site checks; no network or cohort access."""
# ruff: noqa: E402
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]
import pytest
import capacity_relaunch as builder
import capacity_reoperate as operator
import capacity_recontrol as controller
import qualify_worker as worker
from prepare_launch import canonical


@pytest.fixture(scope="module")
def payload():
    return builder.assemble()


def test_recovery_inherits_budget_roles_hardware_and_new_source(payload):
    plan = json.loads(payload["PLAN.json"])
    worker.validate_plan(plan)
    assert plan["maximum_new_probe_attempts"] == 22
    assert [b["previous_probe_attempts"] + b["maximum_new_probes"]
            for b in plan["candidate_probe_budgets"]] == [10, 10, 10]
    assert plan["recovery_parent_sha256"] == hashlib.sha256(payload["RECOVERY_PARENT.json"]).hexdigest()
    assert plan["scientific_optimizer_updates"] == 0
    assert not plan["D1_endpoint_access"] and not plan["D2_scores_computed"] and not plan["LARRY_access"]
    job = json.loads(payload["gpu-job.json"])
    assert job["metadata"]["name"] == builder.RUN_ID
    assert job["spec"]["suspend"] is True and job["spec"]["backoffLimit"] == 0
    assert "activeDeadlineSeconds" not in job["spec"]
    pod = job["spec"]["template"]["spec"]
    assert "activeDeadlineSeconds" not in pod
    container = pod["containers"][0]
    assert container["resources"]["limits"]["memory"] == "200Gi"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod["nodeSelector"]["gpu-type"] == "H100"
    assert container["command"][-1] == builder.HOME_RUN + "/bundle/worker.py"
    assert builder.RUN_ID.encode() in payload["publication.py"]
    assert builder.prior.RUN_ID.encode() not in payload["publication.py"]
    # Same-image Python 3.12 worker and host-3.6 controller syntax, by role.
    ast.parse(payload["worker.py"], feature_version=(3, 12))
    ast.parse(payload["control.py"], feature_version=(3, 6))
    implementation = json.loads(payload["IMPLEMENTATION.json"])
    assert implementation == {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()
        if name.endswith(".py") and name not in {"control.py", "publication.py"}}


@pytest.mark.parametrize("change", ["reset_budget", "extra_probe", "missing_parent", "training"])
def test_recovery_scope_inflation_rejected(payload, change):
    plan = json.loads(payload["PLAN.json"])
    if change == "reset_budget":
        plan["candidate_probe_budgets"][0]["maximum_new_probes"] = 10
    elif change == "extra_probe":
        plan["maximum_new_probe_attempts"] = 23
    elif change == "missing_parent":
        del plan["recovery_parent_sha256"]
    else:
        plan["scientific_optimizer_updates"] = 1
    with pytest.raises(ValueError):
        worker.validate_plan(plan)


def test_new_exclusive_writer_readback_preserves_parent(tmp_path):
    parent = (ROOT / "capacity_v1/bundle.tar.gz").read_bytes()
    result = builder.build(tmp_path / "capacity_v2")
    assert result["run_id"] == builder.RUN_ID
    assert result["archive_sha256"] == hashlib.sha256((tmp_path / "capacity_v2/bundle.tar.gz").read_bytes()).hexdigest()
    assert (ROOT / "capacity_v1/bundle.tar.gz").read_bytes() == parent
    with pytest.raises(FileExistsError):
        builder.build(tmp_path / "capacity_v2")


def test_top_level_action_passes_inherited_live_binding_without_fallback(tmp_path, monkeypatch):
    # Exercise this new controller's actual action entry, not a helper-only test.
    monkeypatch.setattr(operator, "BASE", tmp_path)
    raw_parent = canonical({"status": "parent_verified"})
    raw_deployment = canonical({"status": "deployed"})
    (tmp_path / "RECOVERY_PARENT_VERIFIED.json").write_bytes(raw_parent)
    (tmp_path / "DEPLOYMENT.json").write_bytes(raw_deployment)
    (tmp_path / "RECOVERY_DEPLOYMENT_VERIFIED.json").write_bytes(canonical(dict(
        status="deployment_verified", plan_sha256="p" * 64,
        previous_receipt_sha256=hashlib.sha256(raw_parent).hexdigest(),
        deployment_receipt_sha256=hashlib.sha256(raw_deployment).hexdigest())))
    monkeypatch.setattr(operator, "verified_build", lambda: ({"plan_sha256": "p" * 64}, {}))
    calls = []
    def actual(name):
        calls.append(name)
        return {"result": {"uid": "accepted-exact-uid", "plan_sha256": "p" * 64}}
    monkeypatch.setattr(operator.shared, "action", actual)
    assert operator.action("create")["result"]["uid"] == "accepted-exact-uid"
    assert calls == ["create"]
    (tmp_path / "DEPLOYMENT.json").write_bytes(canonical({"status": "changed"}))
    with pytest.raises(ValueError, match="chain differs"):
        operator.action("create")
    assert calls == ["create"]


def test_recovery_terminal_audit_rejects_unqualified_claims(tmp_path, monkeypatch, payload):
    plan = json.loads(payload["PLAN.json"])
    plan["worker_output"] = str(tmp_path)
    results = [dict(candidate_id=b["candidate_id"], initial_groups=b["initial_groups"],
        available_groups=4096, measured_groups=b["initial_groups"], probe_attempts=1,
        geometry_backend="geomloss_tensorized", target_vram_reached=False,
        productive_memory_only=True, scientific_optimizer_updates=0,
        probes=[dict(groups=b["initial_groups"], peak_allocated_bytes=100,
            peak_reserved_bytes=200, reserved_admission_ceiling_bytes=78_000_000_000,
            geometry_backend="geomloss_tensorized")]) for b in plan["candidate_probe_budgets"]]
    value = dict(recovery_parent_sha256=plan["recovery_parent_sha256"], results=results)
    monkeypatch.setattr(controller.prior, "validate_result", lambda *args: True)
    monkeypatch.setattr(controller.base, "checked_file", lambda *args: ({}, canonical(value)))
    assert controller.validate_result(plan, {}, tmp_path, [{}])
    value["results"][0]["target_vram_reached"] = True
    with pytest.raises(RuntimeError, match="conclusion"):
        controller.validate_result(plan, {}, tmp_path, [{}])
    value["results"][0]["target_vram_reached"] = False
    value["results"][0]["probe_attempts"] = 3
    with pytest.raises(RuntimeError, match="probes differ"):
        controller.validate_result(plan, {}, tmp_path, [{}])


def test_changed_worker_import_and_plan_are_executable(payload, monkeypatch):
    # Execute the packaged top-level main through plan validation/verification,
    # then stop before any CUDA import/data opening at the exact writer guard.
    plan = json.loads(payload["PLAN.json"])
    monkeypatch.setattr(worker, "load_plan", lambda *args, **kwargs: (
        Path(plan["bundle_root"]), plan, hashlib.sha256(payload["PLAN.json"]).hexdigest()))
    verified = []
    monkeypatch.setattr(worker, "verify_bundle", lambda *args: verified.append(True))
    monkeypatch.setattr(worker.os, "getuid", lambda: 1)
    with pytest.raises(RuntimeError, match="writer differs"):
        worker.main([])
    assert verified == [True]


def test_parent_audit_bound_exact_host_and_identity_not_generic_queue_scan():
    parent = json.loads((ROOT / "RECOVERY_PARENT.json").read_text())
    assert parent["maximum_new_probes_by_candidate"] == [2, 10, 10]
    assert parent["historical_measurements_are_new_source_qualification"] is False
    compile(operator.PARENT_AUDIT, "<parent-audit>", "exec")
    assert "PARENT['supervisor_host']" in operator.PARENT_AUDIT
    assert "PARENT['parent_run_id']" in operator.PARENT_AUDIT
    assert "parent_job_and_pods_absent_twice=True" in operator.PARENT_AUDIT
    assert operator.HOME_RUN.endswith("/capacity_v2")
