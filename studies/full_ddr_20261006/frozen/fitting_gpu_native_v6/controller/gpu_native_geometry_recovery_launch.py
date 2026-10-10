"""Fresh full-support padding repair. Immutable v5 is the historical parent."""
from __future__ import annotations

import argparse
import json

import gpu_native_compiler_launch as inherited
import gpu_native_compiler_reuse as reuse
import prepare_launch as shared

ROOT = inherited.ROOT
RECOVERY = ROOT / "gpu_native_geometry_recovery_v6"
ROOT_NAMES = {"qualification": "gpu_native_qualification_v6", "fitting": "fitting_gpu_native_v6"}
QUAL_NAME = "credo-full-ddr-gpu-native-qual-20261009-v6"
FIT_NAME = "credo-full-ddr-fitting-gpu-native-20261009-v6"
CPU_NAME = reuse.OLD_JOB
OLD_QUAL = "credo-full-ddr-gpu-native-qual-20261009-v5"
OLD_QUAL_UID = "ba1c3399-810f-424e-8fb4-54db0e1f21f4"
OLD_QUAL_PLAN = "3fd0d310fc1ebcc465a3d798b591b2d49195900b63c8ce8bcc0f9f082711039b"
OLD_FIT = "credo-full-ddr-fitting-gpu-native-20261009-v5"
OLD_FIT_UID = "eaac6c96-0218-40fb-bab8-d5517c1acec7"
OLD_FIT_PLAN = "384d73610c839fdb1c667b538526d8acb889592e0e2a119d732754abb590d7c7"
SOURCES = ("gpu_native_geometry_recovery_launch.py", "gpu_native_geometry_recovery_operate.py")


def roots(phase):
    suffix = inherited.original.STUDY_SUFFIX + ROOT_NAMES[phase]
    return "/rsrch8/home/bcb/yding4" + suffix, "/rsrch8/scratch/bcb/yding4" + suffix


def historical_payload(phase):
    base = ROOT / ("gpu_native_qualification_v5" if phase == "qualification" else "fitting_gpu_native_v5")
    build = json.loads(shared.read_regular(base / "BUILD.json"))
    expected = OLD_QUAL_PLAN if phase == "qualification" else OLD_FIT_PLAN
    if build["plan_sha256"] != expected:
        raise ValueError("Historical v5 producer differs")
    payload = {}
    for name, entry in build["files"].items():
        raw = shared.read_regular(base / "bundle" / name)
        if entry != dict(sha256=shared.sha(raw), size_bytes=len(raw)):
            raise ValueError("Historical v5 bytes changed: " + name)
        if name not in {"INVENTORY.json", "PHASE_CHAIN.json"}:
            payload[name] = raw
    if shared.sha(payload["PLAN.json"]) != expected:
        raise ValueError("Historical v5 plan changed")
    return payload


def recovery_contract(before):
    contract = json.loads(before)
    for name, expected in contract["source_sha256"].items():
        if name != "gpu_native.py" and shared.sha(shared.read_regular(ROOT / name)) != expected:
            raise ValueError("Unchanged scientific source differs: " + name)
    proof = json.loads(shared.read_regular(RECOVERY / "PARENT_VERIFIED_RUNTIME.json"))
    if (proof.get("status") != "parent_absence_verified" or proof.get("failed_fitting_run_id") != OLD_FIT
            or proof.get("failed_fitting_job_uid") != OLD_FIT_UID
            or proof.get("failed_fitting_scientific_updates") != 18
            or proof.get("failed_fitting_attempted_updates") != 19
            or proof.get("failed_fitting_completed_fits") != 0
            or proof.get("two_read_old_job_and_pod_absence_verified") is not True):
        raise ValueError("Failed-fitting identity, work or absence differs")
    contract.update(schema="credo.all4_ddr.gpu_native_qualification_relaunch_contract.v6",
        status="user_authorized_full_support_geometry_recovery", prepared_utc=proof["utc"])
    contract["source_sha256"]["gpu_native.py"] = shared.sha(shared.read_regular(ROOT / "gpu_native.py"))
    contract["recovery"] = dict(failed_run_id=OLD_FIT, failed_job_uid=OLD_FIT_UID,
        failed_plan_sha256=OLD_FIT_PLAN, scientific_updates_applied=18, scientific_updates_attempted=19,
        completed_fits=0, parent_qualification_passed=True, prior_checkpoints_reused=False,
        previous_contract_sha256=shared.sha(before), prior_history=json.loads(before)["recovery"],
        change="Exact-width dense buckets when padding alone exceeds the unchanged cap; admit every fitting law and challenge every exception at every qualified full shape.")
    contract["planned_roots"].update(qualification_job=QUAL_NAME, qualification_home=roots("qualification")[0],
        qualification_controller=roots("qualification")[1], replacement_job=FIT_NAME,
        replacement_home=roots("fitting")[0], replacement_controller=roots("fitting")[1])
    contract["qualification"]["full_support_challenge_required"] = True
    contract["qualification"]["full_batch_selection"] = "All exact-width exceptions in canonical cache order, then canonical eligible laws without duplicates to 1086; metadata-only, for both stages and every architecture/backend."
    contract["qualification"]["geometry_width_policy"] = "Power-of-two padding only when one padded law fits; otherwise its full exact support. Never increase the workspace cap or drop a cell. Admit the entire fitting support before updates."
    contract["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] = 296
    contract["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] = 32964
    contract["implementation_remaining_before_execution"] = [
        "Admit every fitting law and retain every endpoint cell under the unchanged workspace cap.",
        "Pass the renewed bounded H100 panel with all exact-width exceptions, unchanged tolerances and full input-lease exit.",
        "Exact-clean qualification before conditional fresh nine-fit execution; preserve 18 applied / 19 attempted parent updates."]
    authority = dict(schema="credo.all4_ddr.geometry_recovery_authority.v6",
        status="user_authorized_qualification_and_conditional_fresh_fitting", user_instruction="fix all and relaunch",
        maximum_new_scientific_updates=23040, maximum_loss_backward_calls=102,
        maximum_disposable_optimizer_steps=18, maximum_zero_GPU_compiler_jobs=0, automatic_retries=0,
        parent_scientific_updates_applied=18, parent_scientific_updates_attempted=19,
        parent_completed_fits=0, prior_disposable_steps=34, prior_loss_backward_calls=194,
        historical_original_scientific_attempts_upper=9853, failed_fitting_run_id=OLD_FIT,
        failed_fitting_job_uid=OLD_FIT_UID, failed_fitting_plan_sha256=OLD_FIT_PLAN,
        scientific_architectures_objectives_seeds_unchanged=True, prior_attempts_preserved=True,
        runtime_deadline=None, LARRY_access=False)
    return contract, authority


def assemble_phase(phase, *, qualification_plan_sha=None):
    payload = historical_payload(phase)
    old_plan = json.loads(payload["PLAN.json"])
    contract, authority = recovery_contract(payload["GPU_CONTRACT.json"])
    contract_raw = shared.canonical(contract)
    authority["prepared_contract_sha256"] = shared.sha(contract_raw)
    authority_raw = shared.canonical(authority)
    for name in ("gpu_native.py", "gpu_native_qualification.py", "geometry_evidence.py"):
        payload[name] = shared.read_regular(ROOT / name)
    payload["worker.py"] = shared.read_regular(ROOT / "gpu_native_worker.py")
    payload["control.py"] = shared.read_regular(ROOT / "gpu_native_control.py")
    for name in SOURCES:
        payload["controller/" + name] = shared.read_regular(ROOT / name)
    payload.update({"GPU_CONTRACT.json": contract_raw, "EXECUTION_AUTHORITY.json": authority_raw,
        "PARENT_ABSENCE.json": shared.read_regular(RECOVERY / "PARENT_VERIFIED_RUNTIME.json")})
    home, control = roots(phase)
    run_id = QUAL_NAME if phase == "qualification" else FIT_NAME
    old_home = str(old_plan["bundle_root"]).removesuffix("/bundle")
    payload["publication.py"] = payload["publication.py"].replace(old_plan["run_id"].encode(), run_id.encode()).replace(
        old_home.encode(), home.encode()).replace(old_plan["control_root"].encode(), control.encode())
    payload["IMPLEMENTATION.json"] = shared.canonical({name: shared.sha(raw) for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}})
    plan = dict(old_plan)
    plan.update(schema="credo.all4_ddr.gpu_native_" + phase + "_execution.v6",
        schema_id="credo.all4_full_ddr.gpu_native_" + phase + ".v6", run_id=run_id,
        bundle_root=home + "/bundle", control_root=control, worker_output=home + "/output", output=home + "/output",
        gpu_contract_sha256=shared.sha(contract_raw), execution_authority_sha256=shared.sha(authority_raw),
        parent_absence_sha256=shared.sha(payload["PARENT_ABSENCE.json"]), implementation_sha256=shared.sha(payload["IMPLEMENTATION.json"]),
        control_sha256=shared.sha(payload["control.py"]), worker_sha256=shared.sha(payload["worker.py"]),
        metadata_path=home + "/bundle/METADATA_PREPARATION.json", authority_path=home + "/bundle/AUTHORITY.json",
        full_support_geometry_admission_required=True, failed_fitting_run_id=OLD_FIT,
        startup_gate_authority="User-authorized full-support GPU-native " + phase + " recovery; no assessment")
    plan.pop("failed_qualification_run_id", None)
    if phase == "qualification":
        plan["conditional_successor_run_id"] = FIT_NAME
    else:
        if not qualification_plan_sha:
            raise ValueError("New qualification identity required")
        qhome, qcontrol = roots("qualification")
        plan["qualification_binding"] = dict(plan_sha256=qualification_plan_sha, output=qhome + "/output",
            control=qcontrol + "/gpu", worker_gate_path=qhome + "/QUALIFIED_FITTING_GATE.json")
    payload["PLAN.json"] = shared.canonical(plan)
    job = json.loads(payload["gpu-job.json"])
    job["metadata"]["name"] = run_id
    job["metadata"]["annotations"]["seadragon_run_id"] = run_id
    worker = job["spec"]["template"]["spec"]["containers"][0]
    worker["command"] = ["/opt/credo/bin/python", home + "/bundle/worker.py"]
    for entry in worker["env"]:
        if entry["name"] in {"CREDO_RUN_ID", "CREDO_PLAN_SHA256", "PYTHONPATH"}:
            entry["value"] = {"CREDO_RUN_ID": run_id, "CREDO_PLAN_SHA256": shared.sha(payload["PLAN.json"]),
                "PYTHONPATH": home + "/bundle/source/src"}[entry["name"]]
    payload["gpu-job.json"] = shared.canonical(job)
    return payload


def build_pair():
    old = inherited.ROOT_NAMES, inherited.QUAL_NAME, inherited.FIT_NAME, inherited.assemble_phase
    inherited.ROOT_NAMES, inherited.QUAL_NAME, inherited.FIT_NAME, inherited.assemble_phase = ROOT_NAMES, QUAL_NAME, FIT_NAME, assemble_phase
    try:
        return inherited.build_pair()
    finally:
        inherited.ROOT_NAMES, inherited.QUAL_NAME, inherited.FIT_NAME, inherited.assemble_phase = old


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({phase: {key: value for key, value in row.items() if key != "files"}
        for phase, row in build_pair().items()}, sort_keys=True))
