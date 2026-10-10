"""New-source recovery and admission gate regression; no remote operations."""
import ast
import copy
import json

import pytest

import gpu_native_geometry_recovery_launch as launch
import gpu_native_geometry_recovery_operate as operate
from geometry_evidence import validate_admission, validate_geometry_qualification
from gpu_native_worker import validate_plan


def admission():
    return dict(status="all_fitting_geometry_support_admitted", particles=512,
        fitting_laws=46809, paired_laws=45014, maximum_geometry_elements=16*1024**2,
        dropped_cells=0, solver="unchanged_geomloss_tensorized", observation_weights_changed=False,
        maximum_support=2948, maximum_singleton_elements=11971600,
        exact_width_laws=[dict(measure_id="D3::synthetic", support=2948, width=2948, pairwise_elements=11971600)])


def test_completion_requires_every_full_shape_to_cover_exact_width_laws():
    record = admission()
    assert validate_admission(record) == ["D3::synthetic"]
    complete = dict(geometry_admission=record, full_support_challenge_qualified=True,
        profiles=[dict(full_support_challenge=dict(exact_width_measure_ids=["D3::synthetic"],
            full_batch_guides=1086, all_required_included=True)) for _ in range(18)])
    validate_geometry_qualification(complete)
    complete["profiles"][17]["full_support_challenge"]["exact_width_measure_ids"] = []
    with pytest.raises(RuntimeError, match="omitted"):
        validate_geometry_qualification(complete)


@pytest.mark.parametrize("key,value", [("dropped_cells", 1), ("particles", 256), ("paired_laws", 45013),
    ("maximum_geometry_elements", 2**25), ("maximum_singleton_elements", 2**25),
    ("maximum_support", 10000), ("observation_weights_changed", True)])
def test_admission_rejects_altered_support_or_budget(key, value):
    record = admission()
    record[key] = value
    with pytest.raises(RuntimeError):
        validate_admission(record)


def test_admission_rejects_duplicate_or_padded_exception():
    record = admission()
    record["exact_width_laws"] *= 2
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_admission(record)
    record = admission()
    record["exact_width_laws"][0]["width"] = 4096
    with pytest.raises(RuntimeError, match="policy"):
        validate_admission(record)


@pytest.fixture(scope="module")
def payloads():
    q = launch.assemble_phase("qualification")
    return q, launch.assemble_phase("fitting", qualification_plan_sha=launch.shared.sha(q["PLAN.json"]))


def test_new_package_is_homogeneous_and_preserves_science(payloads):
    q, f = payloads
    old = json.loads(launch.historical_payload("qualification")["GPU_CONTRACT.json"])
    for payload in payloads:
        plan, contract, authority = (json.loads(payload[name]) for name in (
            "PLAN.json", "GPU_CONTRACT.json", "EXECUTION_AUTHORITY.json"))
        validate_plan(plan, contract, authority)
        for name in ("information_contract", "architectures", "resources"):
            assert contract[name] == old[name]
        assert contract["qualification"]["loss_and_gradient_checks"] == old["qualification"]["loss_and_gradient_checks"]
        assert authority["parent_scientific_updates_applied"] == 18
        assert authority["parent_scientific_updates_attempted"] == 19
        assert authority["maximum_new_scientific_updates"] == 23040
        assert authority["maximum_loss_backward_calls"] == 102
        assert plan["full_support_geometry_admission_required"] is True
        assert plan["equal_input_Adam_qualification_required"] is True
        assert "compiler-job.json" not in payload
        job = json.loads(payload["gpu-job.json"])
        assert "activeDeadlineSeconds" not in job["spec"]
        assert job["spec"]["backoffLimit"] == 0
        worker = job["spec"]["template"]["spec"]["containers"][0]
        assert worker["command"][1] == plan["bundle_root"] + "/worker.py"
        assert next(r["value"] for r in worker["env"] if r["name"] == "CREDO_PLAN_SHA256") == launch.shared.sha(payload["PLAN.json"])
        for name, digest in json.loads(payload["IMPLEMENTATION.json"]).items():
            assert launch.shared.sha(payload[name]) == digest
    for name, raw in q.items():
        if name.endswith(".py") and name != "publication.py":
            assert raw == f[name]
    assert json.loads(f["PLAN.json"])["qualification_binding"]["plan_sha256"] == launch.shared.sha(q["PLAN.json"])


def test_new_plan_cannot_omit_repair_gate(payloads):
    for payload in payloads:
        plan, contract, authority = (copy.deepcopy(json.loads(payload[name])) for name in (
            "PLAN.json", "GPU_CONTRACT.json", "EXECUTION_AUTHORITY.json"))
        plan["full_support_geometry_admission_required"] = False
        with pytest.raises(ValueError, match="Full-support"):
            validate_plan(plan, contract, authority)


def test_parent_failure_and_host_identity_are_exact():
    ast.parse(operate.PARENT_AUDIT, feature_version=(3, 6))
    for token in (launch.OLD_FIT_UID, launch.OLD_FIT_PLAN, launch.OLD_QUAL_UID,
            "677265", "191327801", "593695", "191245791", "host!='ldragon5'",
            "scientific_optimizer_updates')!=18", "attempted_optimizer_updates')!=19"):
        assert token in operate.PARENT_AUDIT
    assert "for _ in range(2):" in operate.PARENT_AUDIT
    assert "previous_successor_was_launched=True" in operate.PARENT_AUDIT


def test_host_geometry_validator_is_python36_compatible():
    ast.parse((launch.ROOT / "geometry_evidence.py").read_bytes(), feature_version=(3, 6))
