"""One new compiler recovery; keep image, estimator and every old bundle."""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path

from compiler_toolchain import ARCHIVE_BYTES, ARCHIVE_SHA, archive_inventory
import gpu_native_launch as original
import gpu_native_recovery_launch as prior
import prepare_launch as shared

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT / "gpu_native_compiler_recovery_v3"
QUAL_NAME = "credo-full-ddr-gpu-native-qual-20261009-v3"
FIT_NAME = "credo-full-ddr-fitting-gpu-native-20261009-v3"
CPU_NAME = "credo-full-ddr-compiler-static-20261009-v3"
ROOT_NAMES = {"qualification": "gpu_native_qualification_v3", "fitting": "fitting_gpu_native_v3"}
TOOL_SOURCES = ("compiler_toolchain.py", "compiler_probe.c", "compiler_probe.cpp", "compiler_preflight_worker.py", "compiler_preflight_control.py")


def roots(phase):
    suffix = original.STUDY_SUFFIX + ROOT_NAMES[phase]
    return "/rsrch8/home/bcb/yding4" + suffix, "/rsrch8/scratch/bcb/yding4" + suffix


@lru_cache(maxsize=1)
def compiler_payload():
    source = json.loads(shared.read_regular(RECOVERY / "COMPILER_SOURCE.json"))
    archive = shared.read_regular(RECOVERY / source["archive_name"])
    if shared.sha(archive) != ARCHIVE_SHA or len(archive) != ARCHIVE_BYTES:
        raise ValueError("Upstream standalone compiler differs")
    inventory = shared.canonical(archive_inventory(archive))
    source.update(inventory_sha256=shared.sha(inventory), probe_source_sha256={
        name: shared.sha(shared.read_regular(ROOT / name)) for name in TOOL_SOURCES})
    return {"toolchain.tar.xz": archive, "COMPILER_INVENTORY.json": inventory,
        "COMPILER_SPEC.json": shared.canonical(source)}


def compiler_contract():
    before = shared.read_regular(ROOT / "gpu_native_qualification_v2/bundle/GPU_CONTRACT.json")
    build = json.loads(shared.read_regular(ROOT / "gpu_native_qualification_v2/BUILD.json"))
    if shared.sha(before) != build["files"]["GPU_CONTRACT.json"]["sha256"]:
        raise ValueError("Previous recovery contract changed")
    contract = json.loads(before)
    for name, expected in contract["source_sha256"].items():
        if shared.sha(shared.read_regular(ROOT / name)) != expected:
            raise ValueError("Unchanged scientific source differs: " + name)
    authority = json.loads(shared.read_regular(RECOVERY / "EXECUTION_AUTHORITY.json"))
    if (authority.get("user_instruction") != "fix all and relaunch"
            or authority.get("failed_qualification_run_id") != "credo-full-ddr-gpu-native-qual-20261009-v2"
            or authority.get("maximum_new_scientific_updates") != 23040
            or authority.get("maximum_loss_backward_calls") != 102
            or authority.get("maximum_disposable_optimizer_steps") != 18
            or authority.get("automatic_retries") != 0):
        raise ValueError("Compiler recovery authority differs")
    proof = json.loads(shared.read_regular(RECOVERY / "PARENT_VERIFIED_RUNTIME.json"))
    consumed = authority["failed_qualification_consumed_work"]
    contract.update(schema="credo.all4_ddr.gpu_native_qualification_relaunch_contract.v3",
        status="user_authorized_compiler_recovery", prepared_utc=proof["utc"])
    contract["recovery"] = dict(failed_run_id=authority["failed_qualification_run_id"],
        failed_job_uid=authority["failed_qualification_job_uid"], failed_plan_sha256=authority["failed_qualification_plan_sha256"],
        preserved_consumed_work=consumed, earlier_preserved_consumed_work=authority["earlier_failed_consumed_work"],
        prior_qualification_scientific_updates=0, prior_qualification_models_or_partial_results_reused=False,
        previous_contract_sha256=shared.sha(before),
        change="Bind a hash-pinned standalone C/C++ compiler; verify its native ABI in the unchanged image before H100 creation and cohort access.")
    contract["planned_roots"].update(qualification_job=QUAL_NAME, qualification_home=roots("qualification")[0],
        qualification_controller=roots("qualification")[1], replacement_job=FIT_NAME,
        replacement_home=roots("fitting")[0], replacement_controller=roots("fitting")[1])
    contract["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] += consumed["steps"]
    contract["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] += consumed["full"] + consumed["partial"]
    contract["compiler_prerequisite"] = dict(spec_sha256=shared.sha(compiler_payload()["COMPILER_SPEC.json"]),
        archive_sha256=ARCHIVE_SHA, image_unchanged=True, zero_GPU_preflight_jobs=1, native_builds_per_worker=2,
        ephemeral_storage_limit_unchanged=True, CPU_job=CPU_NAME, package_installation=False)
    contract["implementation_remaining_before_execution"] = [
        "Publish new immutable compiler and phase bundles; verify failed-owner absence and live admission.",
        "Pass one same-image, zero-GPU compiler/ABI worker and exact cleanup before creating H100 qualification.",
        "Complete the bounded H100 numerical qualification, input lease exit and exact cleanup before nine fresh fits.",
    ]
    return contract, authority


def assemble_phase(phase, *, qualification_plan_sha=None):
    # Reuse the reviewed v2 assembly without modifying its source or bundles.
    previous_values = prior.RECOVERY, prior.ROOT_NAMES, prior.QUAL_NAME, prior.FIT_NAME, prior.recovery_contract
    prior.RECOVERY, prior.ROOT_NAMES, prior.QUAL_NAME, prior.FIT_NAME = RECOVERY, ROOT_NAMES, QUAL_NAME, FIT_NAME
    prior.recovery_contract = compiler_contract
    try:
        payload = prior.assemble_phase(phase, qualification_plan_sha=qualification_plan_sha)
    finally:
        prior.RECOVERY, prior.ROOT_NAMES, prior.QUAL_NAME, prior.FIT_NAME, prior.recovery_contract = previous_values
    payload.update(compiler_payload())
    for name in TOOL_SOURCES:
        payload[name] = shared.read_regular(ROOT / name)
    for name in ("gpu_native_compiler_launch.py", "gpu_native_compiler_operate.py"):
        payload["controller/" + name] = shared.read_regular(ROOT / name)
    payload["IMPLEMENTATION.json"] = shared.canonical({name: shared.sha(raw) for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}})
    plan = json.loads(payload["PLAN.json"])
    qhome, _ = roots("qualification")
    plan.update(schema="credo.all4_ddr.gpu_native_" + phase + "_execution.v3",
        schema_id="credo.all4_full_ddr.gpu_native_" + phase + ".v3",
        compiler_dependency_surface_required=True, compiler_spec_sha256=shared.sha(payload["COMPILER_SPEC.json"]),
        compiler_preflight_job=CPU_NAME, compiler_preflight_output=qhome + "/compiler_preflight",
        compiler_preflight_gate_path=qhome + "/compiler_preflight/COMPILER_PASS.json",
        compiler_preflight_qualification_plan_sha256=qualification_plan_sha,
        implementation_sha256=shared.sha(payload["IMPLEMENTATION.json"]))
    payload["PLAN.json"] = shared.canonical(plan)
    job = json.loads(payload["gpu-job.json"])
    worker = job["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry for entry in worker["env"]}
    env["CREDO_PLAN_SHA256"]["value"] = shared.sha(payload["PLAN.json"])
    for name, value in dict(CC="/tmp/credo-compiler/cc", CXX="/tmp/credo-compiler/cxx", TORCHINDUCTOR_COMPILE_THREADS="8").items():
        if name in env:
            raise ValueError("Duplicate compiler environment identity")
        env[name] = dict(name=name, value=value)
    worker["env"] = [env[key] for key in sorted(env)]
    payload["gpu-job.json"] = shared.canonical(job)
    # Preserve the historical neutral manifest; the changed CPU capability has
    # its own manifest, sources, runtime evidence, and absence/cleanup identity.
    cpu = copy.deepcopy(json.loads(payload["static-job.json"]))
    cpu["metadata"]["name"] = CPU_NAME
    cpu["metadata"]["annotations"] = {"seadragon_run_id": CPU_NAME}
    cpu_worker = cpu["spec"]["template"]["spec"]["containers"][0]
    cpu_worker["command"] = ["/opt/credo/bin/python", qhome + "/bundle/compiler_preflight_worker.py"]
    payload["compiler-job.json"] = shared.canonical(cpu)
    return payload


def publish_build(phase, members, archive, receipt):
    old = original.ROOT_NAMES, original.QUAL_NAME, original.FIT_NAME
    original.ROOT_NAMES, original.QUAL_NAME, original.FIT_NAME = ROOT_NAMES, QUAL_NAME, FIT_NAME
    try:
        return original.publish_build(phase, members, archive, receipt)
    finally:
        original.ROOT_NAMES, original.QUAL_NAME, original.FIT_NAME = old


def build_pair():
    if any(os.path.lexists(ROOT / name) for name in ROOT_NAMES.values()):
        raise FileExistsError("Compiler recovery attempt exists; never overwrite")
    qualifier = assemble_phase("qualification")
    fitting = assemble_phase("fitting", qualification_plan_sha=shared.sha(qualifier["PLAN.json"]))
    fit_members, fit_archive, fit_receipt = shared.pack(fitting)
    qualifier["PHASE_CHAIN.json"] = shared.canonical(dict(schema="credo.gpu_native_conditional_phase_chain.v3",
        qualification_plan_sha256=shared.sha(qualifier["PLAN.json"]), successor_bundle=roots("fitting")[0] + "/bundle",
        successor_plan_sha256=fit_receipt["plan_sha256"], successor_inventory_sha256=fit_receipt["inventory_sha256"],
        successor_run_id=FIT_NAME, maximum_concurrent_gpu_requests=1, automatic_retries=0))
    qual_members, qual_archive, qual_receipt = shared.pack(qualifier)
    return dict(fitting=publish_build("fitting", fit_members, fit_archive, fit_receipt),
        qualification=publish_build("qualification", qual_members, qual_archive, qual_receipt))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps({phase: {k: v for k, v in record.items() if k != "files"}
        for phase, record in build_pair().items()}, sort_keys=True))
