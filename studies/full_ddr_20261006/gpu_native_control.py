"""Exact-owner phase validation and one conditional successor; Python 3.6.

The inherited per-job supervisor is attach-only. Only this explicitly bound
qualification-success edge may create its distinct fresh-fitting successor.
No retries, elapsed-time deadline, holdout scoring or scientific selection.
"""
import json
import hashlib
import math
from pathlib import Path
import stat
import subprocess

import preparation_control as base

original_monitor = base.monitor


def read(path):
    reference, raw = base.checked_file(path, True)
    return json.loads(raw.decode()), reference


def result_inventory(plan, complete):
    output = Path(plan["worker_output"])
    inventory = {}
    for item in complete.get("files", []):
        relative = Path(item["path"])
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.as_posix() != item["path"] or item["path"] in inventory or item["path"] == "COMPLETE.json"):
            raise RuntimeError("Unsafe output inventory")
        reference, _ = base.checked_file(output / relative)
        if reference != {key: item[key] for key in ("sha256", "size_bytes")}:
            raise RuntimeError("Output byte identity changed")
        inventory[item["path"]] = reference
    names = set()
    for path in output.rglob("*"):
        info = path.lstat()
        if info.st_uid != 307469 or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise RuntimeError("Unsafe output member")
        if stat.S_ISREG(info.st_mode):
            names.add(path.relative_to(output).as_posix())
    size = sum(row["size_bytes"] for row in inventory.values())
    if (names != set(inventory) | {"COMPLETE.json"} or not inventory or size != complete["output_bytes"]
            or size + (output / "COMPLETE.json").stat().st_size > plan["maximum_output_bytes"]):
        raise RuntimeError("Output coverage/cap differs")
    return inventory


def validate_adam_agreement(row):
    """Require the registered equal-input backend test and separate diagnostics."""
    agreement = row.get("numerical_agreement", {})
    initial = agreement.get("optimizer_initial_state", {})
    if (not isinstance(initial, dict) or initial.get("passed") is not True
            or initial.get("state") != "identical_empty_fresh_plain_Adam"):
        raise RuntimeError("Fresh disposable Adam initial state is not proven")
    if row.get("profile") == "scalar_reference":
        if agreement.get("reference") is not True:
            raise RuntimeError("Scalar optimizer reference missing")
        return
    required = {"loss", "gradients", "clipped", "norm", "Adam", "Adam_state",
        "optimizer_initial_parameters", "optimizer_input_gradients", "optimizer_initial_state"}
    if set(agreement) != required or any(
            not isinstance(check, dict) or check.get("passed") is not True for check in agreement.values()):
        raise RuntimeError("Complete numerical and equal-input Adam evidence missing")
    for name in ("optimizer_initial_parameters", "optimizer_input_gradients"):
        check = agreement[name]
        if check.get("comparison") != "exact_tensor_equality" or check.get("tensors", 0) <= 0:
            raise RuntimeError("Adam backend inputs were not verified identical")
    for name in ("loss", "gradients", "clipped", "norm", "Adam", "Adam_state"):
        check = agreement[name]
        if check.get("atol") != 2e-5 or check.get("rtol") != 2e-5:
            raise RuntimeError("Numerical agreement tolerance changed")
    sensitivity = row.get("independent_gradient_Adam_sensitivity", {})
    if (sensitivity.get("status") != "descriptive_independent_gradient_sensitivity"
            or sensitivity.get("affects_backend_acceptance") is not False
            or sensitivity.get("additional_optimizer_steps") != 0):
        raise RuntimeError("Independent-gradient sensitivity evidence missing")


def validate_result(plan, binding, root, owned):
    output = Path(plan["worker_output"])
    complete, reference = read(output / "COMPLETE.json")
    expected = dict(run_id=plan["run_id"], plan_sha256=base.core.sha(base.BASE / "PLAN.json"),
        implementation_sha256=plan["implementation_sha256"], gpu_contract_sha256=plan["gpu_contract_sha256"],
        execution_authority_sha256=plan["execution_authority_sha256"], fitting_contract_sha256=plan["fitting_contract_sha256"],
        input_lease_exit_verified=True, scientific_promotion=False, query_deployment_performed=False,
        assessment_performed=False, D1_endpoint_access=False, D2_scores_computed=False, LARRY_access=False)
    if len(owned) != 1 or complete.get("pod_uid") != owned[0]["metadata"]["uid"] or any(complete.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Application identity/role/lease differs")
    if plan.get("compiler_dependency_surface_required"):
        compiler = complete.get("compiler_identity", {})
        gate = complete.get("compiler_gate_reference", {})
        _, gate_reference = read(plan["compiler_preflight_gate_path"])
        if (complete.get("compiler_spec_sha256") != plan["compiler_spec_sha256"]
                or compiler.get("compiler_spec_sha256") != plan["compiler_spec_sha256"]
                or compiler.get("status") != "compiler_dependency_surface_verified"
                or compiler.get("native_builds") != 2 or compiler.get("python_extension_loaded") is not True
                or compiler.get("cpp_library_loaded") is not True or compiler.get("cohort_access") is not False
                or gate.get("path") != plan["compiler_preflight_gate_path"]
                or {k: gate.get(k) for k in ("sha256", "size_bytes")} != gate_reference):
            raise RuntimeError("Compiler prerequisite evidence differs")
    inventory = result_inventory(plan, complete)
    if plan.get("full_support_geometry_admission_required"):
        from geometry_evidence import validate_admission, validate_geometry_qualification
        admission, _ = read(output / "GEOMETRY_ADMISSION.json")
        validate_admission(admission)
        if plan["phase"] == "qualification":
            if complete.get("geometry_admission") != admission:
                raise RuntimeError("Geometry completion and admission artifact disagree")
            validate_geometry_qualification(complete)
    compiled, _ = read(output / "COMPILED_STUDY.json")
    if any(compiled.get(k) != v for k, v in dict(fitting_series=46809, fitting_state_pairs=45014,
            fitting_targets=12555, exact_optimizer_updates_per_fit=2560,
            guide_exposures_per_fit=2736740).items()):
        raise RuntimeError("Fitting support/workload differs")
    if plan["phase"] == "qualification":
        if (complete.get("status") != "qualified_gpu_native_full_ddr_execution"
                or compiled.get("D2_numerical_access") is not False or complete.get("scientific_optimizer_updates") != 0
                or complete.get("disposable_optimizer_steps") != 18
                or complete.get("budget_used") != dict(full=90, partial=12, steps=18, refresh=3, chunks=549)
                or complete.get("selected_profile") not in {"GPU_native_eager_foreach", "GPU_native_compiled_foreach"}):
            raise RuntimeError("Qualification incomplete or unqualified profile")
        profiles = complete.get("profiles", [])
        expected_profiles = [(candidate["candidate_id"], profile, stage) for candidate in plan["candidates"]
            for profile in ("scalar_reference", "GPU_native_eager_foreach", "GPU_native_compiled_foreach")
            for stage in ("state", "mass")]
        if [(row.get("candidate_id"), row.get("profile"), row.get("stage")) for row in profiles] != expected_profiles:
            raise RuntimeError("Qualification profile inventory differs")
        for row in profiles:
            if plan.get("equal_input_Adam_qualification_required"):
                validate_adam_agreement(row)
            if len(row.get("calls", [])) != 4 or row.get("optimizer_state_bytes", 0) <= 0:
                raise RuntimeError("Cold/warm/optimizer coverage missing")
            for value in row["calls"] + [row["numerical_call"], row]:
                if (not 0 < value["peak_allocated_bytes"] <= 75_000_000_000
                        or not value["peak_allocated_bytes"] <= value["peak_reserved_bytes"] <= 78_000_000_000
                        or value["free_bytes"] < 4_000_000_000):
                    raise RuntimeError("Qualification memory admission failed")
            if row["profile"] != "scalar_reference" and any(
                    check.get("passed") is not True for check in row["numerical_agreement"].values()):
                raise RuntimeError("Numerical/Adam agreement missing")
        partial = complete.get("partial_profiles", [])
        if len(partial) != 12 or any(row["guides"] != {"state": 488, "mass": 111}[row["stage"]] for row in partial):
            raise RuntimeError("Partial-shape coverage missing")
        if any(any(check.get("passed") is not True for check in row["numerical_agreement"].values())
                for row in partial if row["profile"] == "GPU_native_compiled_foreach"):
            raise RuntimeError("Partial compiled agreement missing")
    else:
        if (complete.get("status") != "completed_gpu_native_full_ddr_fitting"
                or complete.get("scientific_optimizer_updates") != 23040
                or complete.get("attempted_optimizer_updates") != 23040
                or complete.get("scientific_acceptance") != "not_assessed_fitting_only"):
            raise RuntimeError("Fresh fitting incomplete")
        fits = complete.get("fits", [])
        if [{k: row.get(k) for k in ("candidate_id", "seed")} for row in fits] != plan["fits"]:
            raise RuntimeError("Fresh architecture/seed inventory differs")
        for row in fits:
            path = "fits/"+row["candidate_id"]+"/seed-"+str(row["seed"])
            saved, _ = read(output / path / "FIT_COMPLETE.json")
            if saved != row or inventory.get(path+"/state/checkpoint.pt") != row.get("checkpoint"):
                raise RuntimeError("Terminal model bytes differ")
            if (row.get("optimizer_updates") != 2560 or row.get("attempted_optimizer_updates") != 2560
                    or row.get("epochs_completed") != 60 or row.get("guide_exposures") != 2736740
                    or row.get("empty_assessment_table_rows") != dict(metrics=0, predictions=0, counterfactuals=0)
                    or row.get("selected_profile") != complete["selected_profile"]):
                raise RuntimeError("Fitting exposure/profile differs")
            admissions = row.get("optimizer_memory_admissions", [])
            if [value.get("stage") for value in admissions] != ["state", "mass"]:
                raise RuntimeError("Missing optimizer-inclusive admissions")
            for value in admissions+[row]:
                if not 0 < value["peak_allocated_bytes"] <= 75_000_000_000 or not value["peak_allocated_bytes"] <= value["peak_reserved_bytes"] <= 78_000_000_000:
                    raise RuntimeError("Fitting memory admission failed")
            if any(value.get("optimizer_state_bytes", 0) <= 0 for value in admissions):
                raise RuntimeError("Optimizer state missing")
            if any(not math.isfinite(v) or v <= 0 for v in row["head_gradient_maxima"].values()):
                raise RuntimeError("Required DDR gradients missing")
    release, _ = read(root / "startup-validated.json")
    if (release.get("uid") != binding["uid"] or release.get("startup_sha256") != inventory["STARTUP.json"]["sha256"]
            or release.get("actual_pod_audit_passed") is not True):
        raise RuntimeError("Actual worker release absent")
    if base.checked_file(output / "COMPLETE.json")[0] != reference:
        raise RuntimeError("Completion changed")
    return True


def publish_qualification_gate(plan, binding, root, successor_plan, chain):
    """Publish host-verified parent receipts where the GPU worker can read them."""
    entry = successor_plan.get("qualification_binding", {})
    destination = Path(plan["worker_output"]).parent / "QUALIFIED_FITTING_GATE.json"
    if (successor_plan.get("phase") != "fitting"
            or successor_plan.get("run_id") != chain["successor_run_id"]
            or successor_plan.get("gpu_contract_sha256") != plan["gpu_contract_sha256"]
            or entry.get("worker_gate_path") != str(destination)
            or entry.get("plan_sha256") != chain["qualification_plan_sha256"]
            or entry.get("output") != plan["worker_output"]
            or entry.get("control") != str(root)):
        raise RuntimeError("Worker-visible qualification gate binding differs")
    artifacts = {}
    for name, path in (("complete", Path(plan["worker_output"]) / "COMPLETE.json"),
            ("terminal", root / "terminal.json"), ("cleanup", root / "cleanup.json")):
        reference, raw = base.checked_file(path, True)
        if (len(raw) != reference["size_bytes"] or len(raw) > 16 * 1024**2
                or hashlib.sha256(raw).hexdigest() != reference["sha256"]):
            raise RuntimeError("Qualification gate source bytes differ")
        artifacts[name] = dict(path=str(path), reference=reference, raw_json=raw.decode("utf-8"))
    complete, terminal, cleanup = (json.loads(artifacts[name]["raw_json"])
        for name in ("complete", "terminal", "cleanup"))
    if (complete.get("status") != "qualified_gpu_native_full_ddr_execution"
            or complete.get("run_id") != plan["run_id"]
            or complete.get("plan_sha256") != chain["qualification_plan_sha256"]
            or complete.get("gpu_contract_sha256") != plan["gpu_contract_sha256"]
            or complete.get("input_lease_exit_verified") is not True
            or complete.get("budget_used") != dict(full=90, partial=12, steps=18, refresh=3, chunks=549)
            or terminal.get("status") != "passed" or terminal.get("uid") != binding["uid"]
            or terminal.get("plan_sha256") != chain["qualification_plan_sha256"]
            or terminal.get("result_complete_sha256") != artifacts["complete"]["reference"]["sha256"]
            or cleanup.get("status") != "complete" or cleanup.get("uid") != binding["uid"]
            or cleanup.get("reads", [])[-2:] != [dict(job_absent=True, owned_pods=0)] * 2):
        raise RuntimeError("Qualification is not complete, passed and cleaned")
    gate = dict(schema="credo.worker_visible_qualification_gate.v1", status="passed_parent_qualification",
        qualification_plan_sha256=chain["qualification_plan_sha256"], parent_run_id=plan["run_id"],
        parent_job_uid=binding["uid"], gpu_contract_sha256=plan["gpu_contract_sha256"],
        successor_plan_sha256=chain["successor_plan_sha256"], successor_run_id=chain["successor_run_id"],
        artifacts=artifacts, no_new_numerical_work=True)
    # This sibling lives outside the already sealed qualification output.
    # core.publish is exclusive: an uncertain or interrupted edge is never retried.
    base.core.publish(destination, gate)
    return gate


def launch_successor(plan, binding, root):
    chain, _ = read(base.BASE / "PHASE_CHAIN.json")
    if chain.get("qualification_plan_sha256") != base.core.sha(base.BASE / "PLAN.json"):
        raise RuntimeError("Conditional phase chain differs")
    terminal, _ = read(root / "terminal.json")
    cleanup, _ = read(root / "cleanup.json")
    if (terminal.get("status") != "passed" or terminal.get("uid") != binding["uid"]
            or cleanup.get("status") != "complete" or cleanup.get("uid") != binding["uid"]
            or not isinstance(cleanup.get("reads"), list)
            or cleanup["reads"][-2:] != [dict(job_absent=True, owned_pods=0)]*2):
        raise RuntimeError("Parent qualifier not qualified/cleaned")
    successor = Path(chain["successor_bundle"])
    if base.core.sha(successor / "INVENTORY.json") != chain["successor_inventory_sha256"] or base.core.sha(successor / "PLAN.json") != chain["successor_plan_sha256"]:
        raise RuntimeError("Immutable successor differs")
    inventory, _ = read(successor / "INVENTORY.json")
    for name, expected in inventory.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or base.checked_file(successor / name)[0] != expected:
            raise RuntimeError("Successor source differs")
    if plan.get("equal_input_Adam_qualification_required"):
        successor_plan, _ = read(successor / "PLAN.json")
        publish_qualification_gate(plan, binding, root, successor_plan, chain)
    base.core.publish(root / "successor-intent.json", chain)
    python = "/usr/libexec/platform-python3.6"
    for action in ("create", "launch"):
        result = subprocess.run([python, str(successor / "control.py"), action, "--plan", str(successor / "PLAN.json")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        if result.returncode:
            raise RuntimeError("Conditional successor "+action+" failed; reconcile, never retry: "+result.stderr.decode()[-2000:])
        record = json.loads(result.stdout.decode().splitlines()[-1])
        if record.get("job") != chain["successor_run_id"] or not record.get("uid"):
            raise RuntimeError("Conditional successor identity differs; preserve/reconcile")
        if action == "launch" and record.get("status") != "healthy_detached_activated":
            raise RuntimeError("Conditional successor not a healthy handoff; no retry")
        base.core.publish(root / ("successor-"+action+".json"), record)


def monitor(plan, binding, root, static=False):
    original_monitor(plan, binding, root, static=static)
    if plan["phase"] == "qualification":
        try:
            launch_successor(plan, binding, root)
        except Exception as error:
            base.core.publish(root / "successor-failure.json", dict(error=repr(error), automatic_retry=False))
            raise


if __name__ == "__main__":
    base.validate_result = validate_result
    base.monitor = monitor
    base.main()
