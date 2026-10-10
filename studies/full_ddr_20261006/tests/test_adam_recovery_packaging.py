"""Recovery identities and imported evidence, without scheduler or cohort access."""
import ast
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gpu_native_adam_recovery_launch as launch  # noqa: E402
import gpu_native_adam_recovery_operate as operate  # noqa: E402
import gpu_native_compiler_reuse as reuse  # noqa: E402


def test_v5_names_roots_and_audit_are_disjoint():
    assert launch.QUAL_NAME.endswith("-20261009-v5")
    assert launch.FIT_NAME.endswith("-20261009-v5")
    for phase in launch.ROOT_NAMES:
        assert all(path.endswith(launch.ROOT_NAMES[phase]) for path in launch.roots(phase))
    ast.parse(operate.PARENT_AUDIT, feature_version=(3, 6))
    for token in (launch.OLD_QUAL_UID, reuse.OLD_PLAN, "/proc/197662", "190840164",
                  "6ddc45ea3175b61dd90a18476fc767f6899fc728bb83d9901cdac55ac6aed0e4",
                  "successor-intent.json", "successor-create.json", "successor-launch.json", "successor-failure.json"):
        assert token in operate.PARENT_AUDIT
    assert "full=50,partial=6,steps=10,refresh=2,chunks=366" in operate.PARENT_AUDIT
    assert "host!='ldragon5'" in operate.PARENT_AUDIT
    assert "for _ in range(2):" in operate.PARENT_AUDIT


def test_compiler_job_cannot_be_repeated():
    with pytest.raises(RuntimeError, match="no new compiler Job"):
        operate.compiler_check()


def imported_fixture(monkeypatch):
    plan = dict(phase="qualification", image="pinned-image", bundle_root="/new/home/bundle",
        compiler_preflight_import=reuse.import_binding(), compiler_spec_sha256=reuse.OLD_SPEC,
        compiler_preflight_job=reuse.OLD_JOB, compiler_preflight_gate_path=reuse.import_binding()["gate_path"],
        compiler_preflight_qualification_plan_sha256=reuse.OLD_PLAN)
    gate = dict(uid=reuse.OLD_UID, compiler_complete_sha256=reuse.OLD_COMPLETE_SHA,
        cleanup_reference=dict(sha256=reuse.OLD_CLEANUP_SHA))
    complete = dict(image=plan["image"], plan_sha256=reuse.OLD_PLAN, data_free=True)
    cleanup = dict(status="complete", uid=reuse.OLD_UID, reads=[dict(job_absent=True, owned_pods=0)]*2)
    records = {
        plan["compiler_preflight_gate_path"]: (gate, reuse.OLD_GATE_SHA),
        str(Path(plan["compiler_preflight_gate_path"]).parent / "COMPLETE.json"): (complete, reuse.OLD_COMPLETE_SHA),
        "/new/home/bundle/COMPILER_IMPORT_CLEANUP.json": (cleanup, reuse.OLD_CLEANUP_SHA),
    }
    reads = []
    def checked_file(path, expected=None, **kwargs):
        reads.append(str(path))
        record, sha = records[str(path)]
        assert expected == sha
        return json.dumps(record).encode()
    monkeypatch.setattr(reuse.tool, "checked_file", checked_file)
    calls = []
    def original_gate(old):
        calls.append(copy.deepcopy(old))
        return dict(path=old["compiler_preflight_gate_path"], sha256=reuse.OLD_GATE_SHA)
    monkeypatch.setattr(reuse.tool, "verified_compiler_gate", original_gate)
    return plan, records, reads, calls


def test_import_uses_historical_producer_and_worker_visible_cleanup(monkeypatch):
    plan, _, reads, calls = imported_fixture(monkeypatch)
    before = copy.deepcopy(plan)
    result = reuse.verified_imported_compiler_gate(plan)
    assert result["sha256"] == reuse.OLD_GATE_SHA
    assert plan == before
    assert calls[0]["phase"] == "fitting"
    assert calls[0]["compiler_preflight_qualification_plan_sha256"] == reuse.OLD_PLAN
    assert all("scratch" not in path for path in reads)
    assert reads[-1] == "/new/home/bundle/COMPILER_IMPORT_CLEANUP.json"


@pytest.mark.parametrize("key,value", [
    ("compiler_spec_sha256", "changed"), ("compiler_preflight_job", "v5-fabricated"),
    ("compiler_preflight_gate_path", "/new/gate"), ("compiler_preflight_qualification_plan_sha256", "v5")])
def test_import_rejects_changed_producer_key(monkeypatch, key, value):
    plan, _, _, _ = imported_fixture(monkeypatch)
    plan[key] = value
    with pytest.raises(RuntimeError, match="import key differs"):
        reuse.verified_imported_compiler_gate(plan)


@pytest.mark.parametrize("which,key,value", [
    (0, "uid", "wrong"), (0, "compiler_complete_sha256", "wrong"),
    (1, "plan_sha256", "v5"), (1, "image", "wrong"), (1, "data_free", False),
    (2, "uid", "wrong"), (2, "reads", [dict(job_absent=True, owned_pods=0)]), (2, "status", "failed")])
def test_import_rejects_unqualified_receipt(monkeypatch, which, key, value):
    plan, records, _, calls = imported_fixture(monkeypatch)
    list(records.values())[which][0][key] = value
    with pytest.raises(RuntimeError):
        reuse.verified_imported_compiler_gate(plan)
    assert not calls


def test_import_rejects_untyped_or_extended_binding(monkeypatch):
    plan, _, _, _ = imported_fixture(monkeypatch)
    plan["compiler_preflight_import"]["new_job_claim"] = True
    with pytest.raises(RuntimeError, match="Unrecognized"):
        reuse.verified_imported_compiler_gate(plan)


@pytest.fixture(scope="module")
def payloads():
    from frozen_payload import frozen_payload
    return tuple(frozen_payload(ROOT / name) for name in ("gpu_native_qualification_v5", "fitting_gpu_native_v5"))


def test_fresh_bundles_preserve_science_tolerances_and_conditional_chain(payloads):
    old = json.loads((ROOT / "gpu_native_qualification_v4/bundle/GPU_CONTRACT.json").read_bytes())
    for phase, payload in zip(("qualification", "fitting"), payloads):
        plan, contract = (json.loads(payload[name]) for name in ("PLAN.json", "GPU_CONTRACT.json"))
        for key in ("source_sha256", "information_contract", "architectures", "resources"):
            assert contract[key] == old[key]
        assert contract["qualification"]["loss_and_gradient_checks"] == old["qualification"]["loss_and_gradient_checks"]
        assert contract["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] == 194
        assert contract["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] == 32927
        assert plan["equal_input_Adam_qualification_required"] is True
        assert plan["compiler_preflight_import"] == reuse.import_binding()
        assert launch.shared.sha(payload["COMPILER_IMPORT_CLEANUP.json"]) == reuse.OLD_CLEANUP_SHA
        assert "compiler-job.json" not in payload
        assert payload["gpu_native_compiler_reuse.py"] == (ROOT / "gpu_native_compiler_reuse.py").read_bytes()
        assert plan["compiler_preflight_qualification_plan_sha256"] == reuse.OLD_PLAN
        assert plan["run_id"] == (launch.QUAL_NAME if phase == "qualification" else launch.FIT_NAME)
        if phase == "fitting":
            assert plan["qualification_binding"]["plan_sha256"] == launch.shared.sha(payloads[0]["PLAN.json"])
            assert plan["maximum_optimizer_updates"] == 23040 and len(plan["fits"]) == 9
            assert plan["qualification_binding"]["worker_gate_path"] == launch.roots("qualification")[0] + "/QUALIFIED_FITTING_GATE.json"
        job = json.loads(payload["gpu-job.json"])
        assert "activeDeadlineSeconds" not in job["spec"]
        env = {row["name"]: row.get("value") for row in job["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env["CREDO_PLAN_SHA256"] == launch.shared.sha(payload["PLAN.json"])


def test_v5_sources_have_no_scheduler_actions_on_import():
    for name in (*launch.SOURCES, "gpu_native_compiler_reuse.py"):
        ast.parse((ROOT / name).read_bytes())


def test_all_historical_v5_qualification_and_fitting_sources_are_homogeneous(payloads):
    q, f = payloads
    for name, raw in q.items():
        if name.endswith(".py") and name != "publication.py":
            assert raw == f[name], name
    for name, current in (("worker.py", "gpu_native_worker.py"),
                          ("control.py", "gpu_native_control.py"),
                          ("gpu_native_qualification.py", "gpu_native_qualification.py")):
        assert q[name] == (ROOT / "gpu_native_qualification_v5/bundle" / name).read_bytes()
    for payload in payloads:
        implementation = json.loads(payload["IMPLEMENTATION.json"])
        for name, expected in implementation.items():
            assert launch.shared.sha(payload[name]) == expected


def test_old_adam_builder_rejects_new_geometry_source():
    with pytest.raises(ValueError, match="Scientific source changed: gpu_native.py"):
        launch.recovery_contract()


def test_builder_routes_new_roots_and_restores_inherited_globals(monkeypatch):
    original = (launch.inherited.ROOT_NAMES, launch.inherited.QUAL_NAME,
        launch.inherited.FIT_NAME, launch.inherited.assemble_phase)
    def check():
        assert launch.inherited.ROOT_NAMES == launch.ROOT_NAMES
        assert launch.inherited.QUAL_NAME == launch.QUAL_NAME
        assert launch.inherited.FIT_NAME == launch.FIT_NAME
        assert launch.inherited.assemble_phase is launch.assemble_phase
        assert launch.inherited.roots("qualification") == launch.roots("qualification")
        raise RuntimeError("synthetic build boundary")
    monkeypatch.setattr(launch.inherited, "build_pair", check)
    with pytest.raises(RuntimeError, match="synthetic build boundary"):
        launch.build_pair()
    assert original == (launch.inherited.ROOT_NAMES, launch.inherited.QUAL_NAME,
        launch.inherited.FIT_NAME, launch.inherited.assemble_phase)
