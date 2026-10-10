"""Fresh bounded Adam qualification correction, reusing the completed compiler."""
from __future__ import annotations

import argparse
import json

import gpu_native_compiler_launch as inherited
import gpu_native_compiler_reuse as reuse
import gpu_native_recovery_launch as prior
import prepare_launch as shared

ROOT = inherited.ROOT
RECOVERY = ROOT / "gpu_native_adam_recovery_v5"
ROOT_NAMES = {"qualification": "gpu_native_qualification_v5", "fitting": "fitting_gpu_native_v5"}
QUAL_NAME = "credo-full-ddr-gpu-native-qual-20261009-v5"
FIT_NAME = "credo-full-ddr-fitting-gpu-native-20261009-v5"
CPU_NAME = reuse.OLD_JOB
OLD_QUAL = "credo-full-ddr-gpu-native-qual-20261009-v4"
OLD_QUAL_UID = "6f0f05f4-bfa3-4bd9-b255-b24d46d748ea"
CONSUMED = dict(full=50, partial=6, steps=10, refresh=2, chunks=366)
EARLIER = [dict(run_id="credo-full-ddr-gpu-native-qual-20261008-v1",
    budget_used=dict(full=11, partial=2, steps=2, refresh=1, chunks=183)),
    dict(run_id="credo-full-ddr-gpu-native-qual-20261009-v2",
    budget_used=dict(full=21, partial=2, steps=4, refresh=1, chunks=183))]
SOURCES = ("gpu_native_adam_recovery_launch.py", "gpu_native_adam_recovery_operate.py")


def roots(phase):
    suffix = inherited.original.STUDY_SUFFIX + ROOT_NAMES[phase]
    return "/rsrch8/home/bcb/yding4" + suffix, "/rsrch8/scratch/bcb/yding4" + suffix


def compiler_payload():
    """Use the exact qualified spec, sources, inventory and archive bytes."""
    base = ROOT / "gpu_native_qualification_v4"
    build = json.loads(shared.read_regular(base / "BUILD.json"))
    payload = {}
    for name in ("COMPILER_SPEC.json", "COMPILER_INVENTORY.json", "toolchain.tar.xz"):
        raw = shared.read_regular(base / "bundle" / name)
        if build["files"][name] != dict(sha256=shared.sha(raw), size_bytes=len(raw)):
            raise ValueError("Historical compiler package changed: " + name)
        payload[name] = raw
    spec = json.loads(payload["COMPILER_SPEC.json"])
    if shared.sha(payload["COMPILER_SPEC.json"]) != reuse.OLD_SPEC:
        raise ValueError("Qualified compiler spec differs")
    if (shared.sha(payload["toolchain.tar.xz"]) != reuse.tool.ARCHIVE_SHA
            or len(payload["toolchain.tar.xz"]) != reuse.tool.ARCHIVE_BYTES
            or shared.sha(payload["COMPILER_INVENTORY.json"]) != spec["inventory_sha256"]):
        raise ValueError("Qualified compiler archive or inventory differs")
    for name, expected in spec["probe_source_sha256"].items():
        if shared.sha(shared.read_regular(ROOT / name)) != expected:
            raise ValueError("Compiler prerequisite source key changed: " + name)
    gate = shared.read_regular(base / "COMPILER_CHECK_RESULT.json")
    if shared.sha(gate) != reuse.OLD_GATE_SHA:
        raise ValueError("Qualified compiler receipt changed")
    cleanup = shared.read_regular(RECOVERY / "COMPILER_IMPORT_CLEANUP.json")
    if shared.sha(cleanup) != reuse.OLD_CLEANUP_SHA:
        raise ValueError("Qualified compiler cleanup changed")
    payload["COMPILER_IMPORT_CLEANUP.json"] = cleanup
    return payload


def recovery_contract():
    before = shared.read_regular(ROOT / "gpu_native_qualification_v4/bundle/GPU_CONTRACT.json")
    build = json.loads(shared.read_regular(ROOT / "gpu_native_qualification_v4/BUILD.json"))
    if shared.sha(before) != build["files"]["GPU_CONTRACT.json"]["sha256"]:
        raise ValueError("Historical v4 contract changed")
    contract = json.loads(before)
    for name, expected in contract["source_sha256"].items():
        if shared.sha(shared.read_regular(ROOT / name)) != expected:
            raise ValueError("Scientific source changed: " + name)
    authority = json.loads(shared.read_regular(RECOVERY / "EXECUTION_AUTHORITY.json"))
    required = dict(user_instruction="fix all and relaunch", failed_qualification_run_id=OLD_QUAL,
        failed_qualification_job_uid=OLD_QUAL_UID, failed_qualification_plan_sha256=reuse.OLD_PLAN,
        maximum_new_scientific_updates=23040, maximum_loss_backward_calls=102,
        maximum_disposable_optimizer_steps=18, automatic_retries=0, maximum_zero_GPU_compiler_jobs=0,
        failed_qualification_consumed_work=CONSUMED, earlier_failed_consumed_work=EARLIER)
    if any(authority.get(key) != value for key, value in required.items()):
        raise ValueError("Adam recovery authority or consumed workload differs")
    proof = json.loads(shared.read_regular(RECOVERY / "PARENT_VERIFIED_RUNTIME.json"))
    contract.update(schema="credo.all4_ddr.gpu_native_qualification_relaunch_contract.v5",
        status="user_authorized_equal_gradient_Adam_qualification_recovery", prepared_utc=proof["utc"])
    contract["recovery"] = dict(failed_run_id=OLD_QUAL, failed_job_uid=OLD_QUAL_UID,
        failed_plan_sha256=reuse.OLD_PLAN, preserved_consumed_work=CONSUMED,
        earlier_preserved_consumed_work=EARLIER, prior_qualification_scientific_updates=0,
        prior_qualification_models_or_partial_results_reused=False, previous_contract_sha256=shared.sha(before),
        change="Compare disposable Adam backends at identical parameters, gradients, clipping and optimizer state, as already registered; preserve separate loss and gradient equivalence checks and tolerances.")
    contract["planned_roots"].update(qualification_job=QUAL_NAME, qualification_home=roots("qualification")[0],
        qualification_controller=roots("qualification")[1], replacement_job=FIT_NAME,
        replacement_home=roots("fitting")[0], replacement_controller=roots("fitting")[1])
    contract["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] = 194
    contract["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] = 32927
    contract["compiler_prerequisite"].update(zero_GPU_preflight_jobs=0, imported_completed_prerequisite=reuse.import_binding())
    contract["implementation_remaining_before_execution"] = [
        "Publish corrected qualification and conditional-fitting bundles under fresh v5 identities.",
        "Validate the exact completed v4 compiler prerequisite and perform native probes in each new worker.",
        "Pass the bounded H100 qualification, input lease exit and exact cleanup before nine fresh fits."]
    return contract, authority


def assemble_phase(phase, *, qualification_plan_sha=None):
    old = prior.RECOVERY, prior.ROOT_NAMES, prior.QUAL_NAME, prior.FIT_NAME, prior.recovery_contract
    prior.RECOVERY, prior.ROOT_NAMES, prior.QUAL_NAME, prior.FIT_NAME, prior.recovery_contract = RECOVERY, ROOT_NAMES, QUAL_NAME, FIT_NAME, recovery_contract
    try:
        payload = prior.assemble_phase(phase, qualification_plan_sha=qualification_plan_sha)
    finally:
        prior.RECOVERY, prior.ROOT_NAMES, prior.QUAL_NAME, prior.FIT_NAME, prior.recovery_contract = old
    payload.update(compiler_payload())
    for name in (*inherited.TOOL_SOURCES, "gpu_native_compiler_reuse.py"):
        payload[name] = shared.read_regular(ROOT / name)
    for name in (*SOURCES, "gpu_native_compiler_launch.py", "gpu_native_compiler_operate.py"):
        payload["controller/" + name] = shared.read_regular(ROOT / name)
    payload["IMPLEMENTATION.json"] = shared.canonical({name: shared.sha(raw) for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}})
    plan = json.loads(payload["PLAN.json"])
    plan.update(schema="credo.all4_ddr.gpu_native_" + phase + "_execution.v5",
        schema_id="credo.all4_full_ddr.gpu_native_" + phase + ".v5", compiler_dependency_surface_required=True,
        compiler_spec_sha256=reuse.OLD_SPEC, compiler_preflight_job=reuse.OLD_JOB,
        compiler_preflight_output=reuse.OLD_HOME + "/compiler_preflight",
        compiler_preflight_gate_path=reuse.import_binding()["gate_path"],
        compiler_preflight_qualification_plan_sha256=reuse.OLD_PLAN,
        compiler_preflight_import=reuse.import_binding(), equal_input_Adam_qualification_required=True,
        implementation_sha256=shared.sha(payload["IMPLEMENTATION.json"]))
    if phase == "fitting":
        plan["qualification_binding"]["worker_gate_path"] = roots("qualification")[0] + "/QUALIFIED_FITTING_GATE.json"
    payload["PLAN.json"] = shared.canonical(plan)
    job = json.loads(payload["gpu-job.json"])
    worker = job["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry for entry in worker["env"]}
    env["CREDO_PLAN_SHA256"]["value"] = shared.sha(payload["PLAN.json"])
    for name, value in dict(CC="/tmp/credo-compiler/cc", CXX="/tmp/credo-compiler/cxx", TORCHINDUCTOR_COMPILE_THREADS="8").items():
        if name in env:
            raise ValueError("Duplicate compiler environment")
        env[name] = dict(name=name, value=value)
    worker["env"] = [env[key] for key in sorted(env)]
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
