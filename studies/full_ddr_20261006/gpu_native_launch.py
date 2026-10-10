"""Build two new immutable phase bundles from verified parent bytes."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import prepare_launch as shared

ROOT = Path(__file__).resolve().parent
QUAL_NAME = "credo-full-ddr-gpu-native-qual-20261008-v1"
FIT_NAME = "credo-full-ddr-fitting-gpu-native-20261008-v1"
STUDY_SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/"
ROOT_NAMES = {"qualification": "gpu_native_qualification_v1", "fitting": "fitting_gpu_native_v1"}


def roots(phase):
    suffix = STUDY_SUFFIX+ROOT_NAMES[phase]
    return "/rsrch8/home/bcb/yding4"+suffix, "/rsrch8/scratch/bcb/yding4"+suffix


def parent_payload():
    build = json.loads(shared.read_regular(ROOT / "fitting_v1/BUILD.json"))
    if build["plan_sha256"] != "577779938e750b79e074bbc167347b012ded7fb34298b31bd7e95b70e6f3b06c":
        raise ValueError("Exact stopped parent required")
    payload = {}
    for name, expected in build["files"].items():
        raw = shared.read_regular(ROOT / "fitting_v1/bundle" / name)
        if dict(sha256=shared.sha(raw), size_bytes=len(raw)) != expected:
            raise ValueError("Parent bytes differ: "+name)
        if name != "INVENTORY.json":
            payload[name] = raw
    return payload


def assemble_phase(phase, *, qualification_plan_sha=None):
    payload = parent_payload()
    contract_raw = shared.read_regular(ROOT / "gpu_native_transition_v1/QUALIFICATION_RELAUNCH_CONTRACT.json")
    if shared.sha(contract_raw) != "cd8a384611630da8239857c5f7dd57742845789007721a58f82e7c210e1e6235":
        raise ValueError("Approved prepared contract changed")
    contract = json.loads(contract_raw)
    authority_raw = shared.read_regular(ROOT / "gpu_native_transition_v1/EXECUTION_AUTHORITY.json")
    authority = json.loads(authority_raw)
    if authority.get("prepared_contract_sha256") != shared.sha(contract_raw) or authority.get("maximum_new_scientific_updates") != 23040:
        raise ValueError("Execution approval missing")
    for name in ("gpu_native.py", "gpu_native_fitting.py", "architecture_fitting.py", "terminal_fitting.py", "development_inputs.py", "native_study.py"):
        raw = shared.read_regular(ROOT / name)
        if shared.sha(raw) != contract["source_sha256"][name]:
            raise ValueError("Approved numerical source changed: "+name)
        payload[name] = raw
    for name in ("gpu_native_inputs.py", "gpu_native_qualification.py"):
        payload[name] = shared.read_regular(ROOT / name)
    # Retain old worker as pure helper source, never its old entrypoint.
    payload["fitting_worker.py"] = payload["worker.py"]
    payload["worker.py"] = shared.read_regular(ROOT / "gpu_native_worker.py")
    payload["control.py"] = shared.read_regular(ROOT / "gpu_native_control.py")
    for name in ("gpu_native_launch.py", "gpu_native_operate.py"):
        payload["controller/"+name] = shared.read_regular(ROOT / name)
    payload["GPU_CONTRACT.json"] = contract_raw
    payload["EXECUTION_AUTHORITY.json"] = authority_raw
    payload["PARENT_ABSENCE.json"] = shared.read_regular(ROOT / "gpu_native_transition_v1/PARENT_VERIFIED_RUNTIME.json")
    original_contract = json.loads(payload["FITTING_CONTRACT.json"])
    original_contract.update(maximum_optimizer_updates_per_fit=2560, maximum_optimizer_updates=23040,
        execution_policy="gpu_native_fitting_cache_per_guide_sinkhorn_v1", source_sampler="batched_padded_multinomial_v1",
        normalizers_reused=True, normalizer_source=contract["information_contract"]["normalizers"])
    original_contract["optimizer_details"]["foreach"] = True
    payload["FITTING_CONTRACT.json"] = shared.canonical(original_contract)
    home, control = roots(phase)
    run_id = QUAL_NAME if phase == "qualification" else FIT_NAME
    payload["publication.py"] = payload["publication.py"].replace(
        b"credo-full-ddr-fitting-20261007-v1", run_id.encode()).replace(
        ("/rsrch8/home/bcb/yding4"+STUDY_SUFFIX+"fitting_v1").encode(), home.encode()).replace(
        ("/rsrch8/scratch/bcb/yding4"+STUDY_SUFFIX+"fitting_v1").encode(), control.encode())
    implementation = {name: shared.sha(raw) for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}}
    payload["IMPLEMENTATION.json"] = shared.canonical(implementation)
    plan = json.loads(payload["PLAN.json"])
    plan.update(schema="credo.all4_ddr.gpu_native_"+phase+"_execution.v1", schema_id="credo.all4_full_ddr.gpu_native_"+phase+".v1",
        phase=phase, run_id=run_id, bundle_root=home+"/bundle", control_root=control,
        worker_output=home+"/output", output=home+"/output", candidates=contract["architectures"],
        maximum_new_output_bytes=256*1024**2 if phase == "qualification" else 16*1024**3,
        maximum_output_bytes=256*1024**2 if phase == "qualification" else 16*1024**3,
        gpu_contract_sha256=shared.sha(contract_raw), execution_authority_sha256=shared.sha(authority_raw),
        parent_absence_sha256=shared.sha(payload["PARENT_ABSENCE.json"]),
        implementation_sha256=shared.sha(payload["IMPLEMENTATION.json"]),
        control_sha256=shared.sha(payload["control.py"]), worker_sha256=shared.sha(payload["worker.py"]),
        fitting_contract_sha256=shared.sha(payload["FITTING_CONTRACT.json"]),
        metadata_path=home+"/bundle/METADATA_PREPARATION.json", authority_path=home+"/bundle/AUTHORITY.json",
        maximum_optimizer_updates=0 if phase == "qualification" else 23040,
        scientific_optimizer_updates=0 if phase == "qualification" else None,
        fitting_support=dict(source=46809, endpoint=45014, targets=12555),
        startup_gate_authority="User-approved GPU-native "+phase+"; no scientific assessment",
        output_completion_status="qualified_gpu_native_full_ddr_execution" if phase == "qualification" else "completed_gpu_native_full_ddr_fitting",
        optimizer_state_memory_qualified=False, normalizer_recomputation=False,
        phase_numerical_roles=["D3", "D4"] if phase == "qualification" else ["D3", "D4", "D2-compilation-only"])
    plan.pop("capacity_binding_sha256", None)
    if phase == "qualification":
        plan.update(maximum_fits=0, fits=[], scientific_fitting_only=False,
            maximum_disposable_optimizer_steps=18, maximum_loss_backward_calls=102,
            conditional_successor_run_id=FIT_NAME)
    else:
        if qualification_plan_sha is None:
            raise ValueError("Qualifier identity required")
        qhome, qcontrol = roots("qualification")
        plan.update(maximum_fits=9, scientific_fitting_only=True,
            qualification_binding=dict(plan_sha256=qualification_plan_sha, output=qhome+"/output", control=qcontrol+"/gpu"))
    payload["PLAN.json"] = shared.canonical(plan)
    job = copy.deepcopy(json.loads(payload["gpu-job.json"]))
    job["metadata"]["name"] = run_id
    job["metadata"]["annotations"] = {"seadragon_run_id": run_id}
    worker = job["spec"]["template"]["spec"]["containers"][0]
    worker["command"] = ["/opt/credo/bin/python", home+"/bundle/worker.py"]
    for entry in worker["env"]:
        if entry["name"] == "CREDO_RUN_ID":
            entry["value"] = run_id
        elif entry["name"] == "CREDO_PLAN_SHA256":
            entry["value"] = shared.sha(payload["PLAN.json"])
        elif entry["name"] == "PYTHONPATH":
            entry["value"] = home+"/bundle/source/src"
    payload["gpu-job.json"] = shared.canonical(job)
    return payload


def publish_build(phase, members, archive, receipt):
    destination = ROOT / ROOT_NAMES[phase]
    destination.mkdir(mode=0o700)
    receipt.update(run_id=QUAL_NAME if phase == "qualification" else FIT_NAME, phase=phase,
        schema="credo.all4_full_ddr.gpu_native_bundle.v1")
    for name, raw in dict({"bundle/"+name: raw for name, raw in members.items()},
            **{"bundle.tar.gz": archive, "BUILD.json": shared.canonical(receipt)}).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if shared.read_regular(path) != raw:
            raise RuntimeError("Immutable build readback differs")
    for directory in sorted((p for p in (destination / "bundle").rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(0o500)
    (destination / "bundle").chmod(0o500)
    return receipt


def build_pair():
    for phase in ROOT_NAMES:
        if os.path.lexists(ROOT / ROOT_NAMES[phase]):
            raise FileExistsError("Attempt already exists; preserve/reconcile, never overwrite")
    qual = assemble_phase("qualification")
    fitting = assemble_phase("fitting", qualification_plan_sha=shared.sha(qual["PLAN.json"]))
    fit_members, fit_archive, fit_receipt = shared.pack(fitting)
    qual["PHASE_CHAIN.json"] = shared.canonical(dict(schema="credo.gpu_native_conditional_phase_chain.v1",
        qualification_plan_sha256=shared.sha(qual["PLAN.json"]), successor_bundle=roots("fitting")[0]+"/bundle",
        successor_plan_sha256=fit_receipt["plan_sha256"], successor_inventory_sha256=fit_receipt["inventory_sha256"],
        successor_run_id=FIT_NAME, maximum_concurrent_gpu_requests=1, automatic_retries=0))
    qual_members, qual_archive, qual_receipt = shared.pack(qual)
    return dict(fitting=publish_build("fitting", fit_members, fit_archive, fit_receipt),
        qualification=publish_build("qualification", qual_members, qual_archive, qual_receipt))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = build_pair()
    print(json.dumps({phase: {key: value for key, value in record.items() if key != "files"}
        for phase, record in result.items()}, sort_keys=True))
