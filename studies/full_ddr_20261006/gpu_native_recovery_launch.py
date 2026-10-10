"""Fresh, source-homogeneous dependency recovery; preserve every v1 byte."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import gpu_native_launch as previous
import prepare_launch as shared

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT / "gpu_native_recovery_v2"
QUAL_NAME = "credo-full-ddr-gpu-native-qual-20261009-v2"
FIT_NAME = "credo-full-ddr-fitting-gpu-native-20261009-v2"
ROOT_NAMES = {"qualification": "gpu_native_qualification_v2", "fitting": "fitting_gpu_native_v2"}


def roots(phase):
    suffix = previous.STUDY_SUFFIX + ROOT_NAMES[phase]
    return "/rsrch8/home/bcb/yding4" + suffix, "/rsrch8/scratch/bcb/yding4" + suffix


def recovery_contract():
    original = shared.read_regular(ROOT / "gpu_native_transition_v1/QUALIFICATION_RELAUNCH_CONTRACT.json")
    if shared.sha(original) != "cd8a384611630da8239857c5f7dd57742845789007721a58f82e7c210e1e6235":
        raise ValueError("Historical prepared contract changed")
    contract = json.loads(original)
    for name, expected in contract["source_sha256"].items():
        if name != "gpu_native.py" and shared.sha(shared.read_regular(ROOT / name)) != expected:
            raise ValueError("Unchanged scientific source differs: " + name)
    authority = json.loads(shared.read_regular(RECOVERY / "EXECUTION_AUTHORITY.json"))
    if (authority.get("user_instruction") != "fix all and relaunch"
            or authority.get("parent_prepared_contract_sha256") != shared.sha(original)
            or authority.get("maximum_new_scientific_updates") != 23040
            or authority.get("maximum_loss_backward_calls") != 102
            or authority.get("maximum_disposable_optimizer_steps") != 18
            or authority.get("automatic_retries") != 0):
        raise ValueError("Recovery authority differs")
    contract.update(schema="credo.all4_ddr.gpu_native_qualification_relaunch_contract.v2",
                    status="user_authorized_dependency_recovery")
    contract["source_sha256"]["gpu_native.py"] = shared.sha(shared.read_regular(ROOT / "gpu_native.py"))
    contract["authority"].update(approved_current_actions="One corrected qualification and conditional nine fresh fits.",
        qualification_submission_authorized=True, replacement_training_authorized=True,
        automatic_successor_launch_authorized=True, new_launch_requires_user_approval=False)
    contract["planned_roots"].update(qualification_job=QUAL_NAME, qualification_home=roots("qualification")[0],
        qualification_controller=roots("qualification")[1], replacement_job=FIT_NAME,
        replacement_home=roots("fitting")[0], replacement_controller=roots("fitting")[1])
    consumed = authority["failed_qualification_consumed_work"]
    proof = json.loads(shared.read_regular(RECOVERY / "PARENT_VERIFIED_RUNTIME.json"))
    original_prepared_utc = contract["prepared_utc"]
    contract["prepared_utc"] = proof["utc"]
    contract["recovery"] = dict(failed_run_id=authority["failed_qualification_run_id"],
        failed_job_uid=authority["failed_qualification_job_uid"], failed_plan_sha256=authority["failed_qualification_plan_sha256"],
        preserved_consumed_work=consumed, prior_qualification_scientific_updates=0,
        prior_qualification_models_or_partial_results_reused=False, original_prepared_utc=original_prepared_utc,
        historical_implementation_requirements=contract["implementation_remaining_before_execution"],
        change="Resolve GeomLoss helpers from the public scalar implementation; verify dependency surface before data access; accept lifecycle cleanup history only after two final absence reads.")
    contract["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] += consumed["steps"]
    contract["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] = 102 + consumed["full"] + consumed["partial"]
    contract["implementation_remaining_before_execution"] = [
        "Freeze and publish the corrected dependency/cleanup sources at new disjoint phase identities.",
        "Pass exact new manifest dry runs, suspended admission bindings, startup dependency surface and H100 hardware checks.",
        "Complete the full bounded qualification and input lease exit, then exact-clean before the conditional fitting launch.",
    ]
    return contract, authority


def assemble_phase(phase, *, qualification_plan_sha=None):
    payload = previous.parent_payload()
    contract, authority = recovery_contract()
    contract_raw = shared.canonical(contract)
    authority["prepared_contract_sha256"] = shared.sha(contract_raw)
    authority_raw = shared.canonical(authority)
    for name in ("gpu_native.py", "gpu_native_fitting.py", "architecture_fitting.py", "terminal_fitting.py", "development_inputs.py", "native_study.py",
                 "gpu_native_inputs.py", "gpu_native_qualification.py"):
        payload[name] = shared.read_regular(ROOT / name)
    payload["fitting_worker.py"] = payload["worker.py"]
    payload["worker.py"] = shared.read_regular(ROOT / "gpu_native_worker.py")
    payload["control.py"] = shared.read_regular(ROOT / "gpu_native_control.py")
    for name in ("gpu_native_recovery_launch.py", "gpu_native_recovery_operate.py"):
        payload["controller/" + name] = shared.read_regular(ROOT / name)
    payload["GPU_CONTRACT.json"] = contract_raw
    payload["EXECUTION_AUTHORITY.json"] = authority_raw
    payload["PARENT_ABSENCE.json"] = shared.read_regular(RECOVERY / "PARENT_VERIFIED_RUNTIME.json")
    proof = json.loads(payload["PARENT_ABSENCE.json"])
    if (proof.get("status") != "parent_absence_verified" or proof.get("failed_qualification_run_id") != contract["recovery"]["failed_run_id"]
            or proof.get("failed_qualification_job_uid") != contract["recovery"]["failed_job_uid"]
            or proof.get("failed_qualification_consumed_work") != contract["recovery"]["preserved_consumed_work"]
            or proof.get("failed_qualification_scientific_updates") != 0
            or proof.get("previous_successor_absent") is not True):
        raise ValueError("Exact failed-parent and unused-successor proof required")
    fitting = json.loads(payload["FITTING_CONTRACT.json"])
    fitting.update(maximum_optimizer_updates_per_fit=2560, maximum_optimizer_updates=23040,
        execution_policy="gpu_native_fitting_cache_per_guide_sinkhorn_v1", source_sampler="batched_padded_multinomial_v1",
        normalizers_reused=True, normalizer_source=contract["information_contract"]["normalizers"])
    fitting["optimizer_details"]["foreach"] = True
    payload["FITTING_CONTRACT.json"] = shared.canonical(fitting)
    home, control = roots(phase)
    run_id = QUAL_NAME if phase == "qualification" else FIT_NAME
    payload["publication.py"] = payload["publication.py"].replace(b"credo-full-ddr-fitting-20261007-v1", run_id.encode()).replace(
        ("/rsrch8/home/bcb/yding4" + previous.STUDY_SUFFIX + "fitting_v1").encode(), home.encode()).replace(
        ("/rsrch8/scratch/bcb/yding4" + previous.STUDY_SUFFIX + "fitting_v1").encode(), control.encode())
    payload["IMPLEMENTATION.json"] = shared.canonical({name: shared.sha(raw) for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}})
    plan = json.loads(payload["PLAN.json"])
    plan.update(schema="credo.all4_ddr.gpu_native_" + phase + "_execution.v2", schema_id="credo.all4_full_ddr.gpu_native_" + phase + ".v2",
        phase=phase, run_id=run_id, bundle_root=home + "/bundle", control_root=control, worker_output=home + "/output", output=home + "/output",
        candidates=contract["architectures"], maximum_new_output_bytes=256*1024**2 if phase == "qualification" else 16*1024**3,
        maximum_output_bytes=256*1024**2 if phase == "qualification" else 16*1024**3,
        gpu_contract_sha256=shared.sha(contract_raw), execution_authority_sha256=shared.sha(authority_raw),
        parent_absence_sha256=shared.sha(payload["PARENT_ABSENCE.json"]), implementation_sha256=shared.sha(payload["IMPLEMENTATION.json"]),
        control_sha256=shared.sha(payload["control.py"]), worker_sha256=shared.sha(payload["worker.py"]), fitting_contract_sha256=shared.sha(payload["FITTING_CONTRACT.json"]),
        metadata_path=home + "/bundle/METADATA_PREPARATION.json", authority_path=home + "/bundle/AUTHORITY.json",
        maximum_optimizer_updates=0 if phase == "qualification" else 23040, scientific_optimizer_updates=0 if phase == "qualification" else None,
        fitting_support=dict(source=46809, endpoint=45014, targets=12555), startup_gate_authority="User-authorized corrected GPU-native " + phase + "; no assessment",
        output_completion_status="qualified_gpu_native_full_ddr_execution" if phase == "qualification" else "completed_gpu_native_full_ddr_fitting",
        optimizer_state_memory_qualified=False, normalizer_recomputation=False,
        phase_numerical_roles=["D3", "D4"] if phase == "qualification" else ["D3", "D4", "D2-compilation-only"],
        geomloss_dependency_surface_required=True, failed_qualification_run_id=contract["recovery"]["failed_run_id"])
    plan.pop("capacity_binding_sha256", None)
    if phase == "qualification":
        plan.update(maximum_fits=0, fits=[], scientific_fitting_only=False,
            maximum_disposable_optimizer_steps=18, maximum_loss_backward_calls=102, conditional_successor_run_id=FIT_NAME)
    else:
        if qualification_plan_sha is None:
            raise ValueError("Qualifier identity required")
        qhome, qcontrol = roots("qualification")
        plan.update(maximum_fits=9, scientific_fitting_only=True,
            qualification_binding=dict(plan_sha256=qualification_plan_sha, output=qhome + "/output", control=qcontrol + "/gpu"))
    payload["PLAN.json"] = shared.canonical(plan)
    job = copy.deepcopy(json.loads(payload["gpu-job.json"]))
    job["metadata"]["name"] = run_id
    job["metadata"]["annotations"] = {"seadragon_run_id": run_id}
    worker = job["spec"]["template"]["spec"]["containers"][0]
    worker["command"] = ["/opt/credo/bin/python", home + "/bundle/worker.py"]
    for entry in worker["env"]:
        if entry["name"] == "CREDO_RUN_ID":
            entry["value"] = run_id
        elif entry["name"] == "CREDO_PLAN_SHA256":
            entry["value"] = shared.sha(payload["PLAN.json"])
        elif entry["name"] == "PYTHONPATH":
            entry["value"] = home + "/bundle/source/src"
    payload["gpu-job.json"] = shared.canonical(job)
    return payload


def publish_build(phase, members, archive, receipt):
    # Reuse the tested exclusive publication writer; bind only new roots in
    # this process and restore its globals so historical callers remain intact.
    old = previous.ROOT_NAMES, previous.QUAL_NAME, previous.FIT_NAME
    previous.ROOT_NAMES, previous.QUAL_NAME, previous.FIT_NAME = ROOT_NAMES, QUAL_NAME, FIT_NAME
    try:
        return previous.publish_build(phase, members, archive, receipt)
    finally:
        previous.ROOT_NAMES, previous.QUAL_NAME, previous.FIT_NAME = old


def build_pair():
    for phase in ROOT_NAMES:
        if os.path.lexists(ROOT / ROOT_NAMES[phase]):
            raise FileExistsError("Recovery attempt exists; preserve/reconcile, never overwrite")
    qual = assemble_phase("qualification")
    fitting = assemble_phase("fitting", qualification_plan_sha=shared.sha(qual["PLAN.json"]))
    fit_members, fit_archive, fit_receipt = shared.pack(fitting)
    qual["PHASE_CHAIN.json"] = shared.canonical(dict(schema="credo.gpu_native_conditional_phase_chain.v2",
        qualification_plan_sha256=shared.sha(qual["PLAN.json"]), successor_bundle=roots("fitting")[0] + "/bundle",
        successor_plan_sha256=fit_receipt["plan_sha256"], successor_inventory_sha256=fit_receipt["inventory_sha256"],
        successor_run_id=FIT_NAME, maximum_concurrent_gpu_requests=1, automatic_retries=0))
    qual_members, qual_archive, qual_receipt = shared.pack(qual)
    return dict(fitting=publish_build("fitting", fit_members, fit_archive, fit_receipt),
                qualification=publish_build("qualification", qual_members, qual_archive, qual_receipt))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({phase: {k: v for k, v in value.items() if k != "files"}
        for phase, value in build_pair().items()}, sort_keys=True))
