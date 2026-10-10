"""One data-free zero-GPU compiler check; exact terminal capture and cleanup.

Python 3.6 controller. The existing strict neutral admission profile is reused
with only the explicitly bound command changed. Never install packages or
submit a scientific fitting Job from this controller.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import control
import neutral_admission as neutral
from runtime_policy import validate_job, validate_plan

core = None


def load_core(bundle):
    """The normal CLI initializes lazily; a separate capability must do so too."""
    path = bundle / "lifecycle_core.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != control.base.CORE_SHA:
        raise RuntimeError("Exact no-deadline lifecycle changed")
    spec = importlib.util.spec_from_file_location("credo_compiler_lifecycle", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    control.base.core = module
    return module


def audit(job, manifest):
    validate_job(manifest)
    validate_job(job)
    return neutral.audit_neutral(job, manifest)


def main():
    global core
    bundle = Path(__file__).resolve().parent
    core = load_core(bundle)
    plan = json.loads((bundle / "PLAN.json").read_text())
    validate_plan(plan)
    neutral.NEUTRAL = ["/opt/credo/bin/python", str(bundle / "compiler_preflight_worker.py")]
    core.audit = audit
    root = Path(plan["control_root"]) / "compiler"
    root.mkdir(mode=0o700)
    pid = os.getpid()
    proc = Path("/proc") / str(pid)
    raw_stat = (proc / "stat").read_text()
    core.publish(root / "process.json", dict(pid=pid, host=socket.gethostname(),
        start_ticks=int(raw_stat[raw_stat.rfind(")")+2:].split()[19]),
        cmdline_sha256=hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest(),
        compiler_controller_sha256=core.sha(Path(__file__)), no_retry=True))
    manifest_path = bundle / "compiler-job.json"
    manifest = json.loads(manifest_path.read_text())
    name = manifest["metadata"]["name"]
    if name != plan["compiler_preflight_job"]:
        raise RuntimeError("Compiler-check exact identity differs")
    for _ in range(2):
        if core.get_job(name) is not None or core.pods(name, None):
            raise RuntimeError("Compiler-check exact Job/Pods already exist")
    dry = json.loads(core.kube("create", "--dry-run=server", "-o", "json", "-f", str(manifest_path)))
    core.publish(root / "dryrun.json", dry)
    audit(dry, manifest)
    core.publish(root / "create-intent.json", dict(job=name, manifest_sha256=core.sha(manifest_path)))
    try:
        created = json.loads(core.kube("create", "-o", "json", "-f", str(manifest_path)))
    except Exception as error:
        core.publish(root / "create-uncertain.json", dict(error=repr(error), job=name, no_retry=True))
        raise
    binding = dict(job=name, uid=created["metadata"]["uid"], run_id=name)
    try:
        core.publish(root / "created.json", created)
        audit(created, manifest)
        binding["contract"] = core.job_contract(created)
        for _ in range(2):
            core.verify(core.get_job(name), binding, True)
            if core.pods(name, binding["uid"]):
                raise RuntimeError("Suspended compiler check owns Pods")
        core.publish(root / "live-binding.json", binding)
        core.activate(binding, root)
        while True:
            job = core.get_job(name)
            core.verify(job, binding, False)
            good, bad = core.complete(job)
            if not (good or bad):
                time.sleep(10)
                continue
            owned = core.capture(job, root)
            if not good or bad or len(owned) != 1:
                raise RuntimeError("Same-image compiler worker failed; no automatic retry")
            core.audit_runtime_pod(owned[0], binding, plan)
            value, reference = control.read(Path(plan["compiler_preflight_output"]) / "COMPLETE.json")
            if (value.get("compiler_spec_sha256") != plan["compiler_spec_sha256"]
                    or value.get("plan_sha256") != core.sha(bundle / "PLAN.json")
                    or value.get("image") != plan["image"] or value.get("native_builds") != 2
                    or value.get("status") != "compiler_dependency_surface_verified"
                    or value.get("data_free") is not True or value.get("cohort_access") is not False
                    or value.get("python_extension_loaded") is not True or value.get("cpp_library_loaded") is not True
                    or value.get("python_version") != "3.12.13" or value.get("torch_package_version") != "2.6.0+cu124"
                    or value.get("triton_version") != "3.2.0"
                    or value.get("gpu_requests") != 0 or value.get("gpu_limits") != 0
                    or value.get("visibility_mask") != "none" or value.get("scientific_optimizer_updates") != 0):
                raise RuntimeError("Compiler native/ABI evidence incomplete")
            terminal = dict(status="passed", job=name, uid=binding["uid"],
                manifest_sha256=core.sha(manifest_path), plan_sha256=core.sha(bundle / "PLAN.json"), compiler_complete=reference)
            core.publish(root / "terminal.json", terminal)
            core.cleanup(binding, root)
            cleanup, cleanup_ref = control.read(root / "cleanup.json")
            if cleanup.get("reads", [])[-2:] != [dict(job_absent=True, owned_pods=0)]*2:
                raise RuntimeError("Compiler-check cleanup unresolved")
            result = dict(status="passed_same_image_compiler_prerequisite", job=name, uid=binding["uid"],
                image=plan["image"], qualification_plan_sha256=core.sha(bundle / "PLAN.json"),
                compiler_spec_sha256=plan["compiler_spec_sha256"], compiler_archive_sha256=value["archive_sha256"],
                compiler_complete_sha256=reference["sha256"], cleanup_reads=cleanup["reads"],
                cleanup_reference=cleanup_ref, native_builds=2, data_free=True,
                GPU_requests=0, visibility_mask="none", scientific_optimizer_updates=0)
            core.publish(Path(plan["compiler_preflight_gate_path"]), result)
            print(json.dumps(result, sort_keys=True), flush=True)
            return
    except BaseException as error:
        if not (root / "terminal.json").exists():
            core.publish(root / "terminal.json", dict(status="failed", job=name, uid=binding["uid"], error=repr(error)))
        if not (root / "cleanup.json").exists():
            core.cleanup(binding, root, create_scoped=True)
        raise


if __name__ == "__main__":
    main()
