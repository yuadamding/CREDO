"""New-study preparation validator around unchanged exact-owner lifecycle.

Host Python 3.6 compatible. No scientific fitting, retries, or job deadlines.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

BASE = Path(__file__).resolve().parent
CORE_SHA = "3da1a0a5b0d0d26019918ee48e2ce24cd2e616272de7854f4c82ba2e37e7ebd2"


def checked_file(path, retain=False):
    path = Path(path)
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise RuntimeError("Symlink/non-directory result ancestor")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != 307469:
        raise RuntimeError("Expected exact owned regular result")
    if retain and before.st_size > 64 * 1024**2:
        raise RuntimeError("Metadata exceeds bounded parser budget")
    digest, chunks = hashlib.sha256(), []

    def signature(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_uid,
        )

    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise RuntimeError("Result changed at open")
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
            if retain:
                chunks.append(block)
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise RuntimeError("Result changed during read")
    if signature(path.lstat()) != signature(before):
        raise RuntimeError("Result path changed during read")
    return {"sha256": digest.hexdigest(), "size_bytes": before.st_size}, b"".join(
        chunks
    )


def validate_result(plan, binding, root, owned):
    output = Path(plan["worker_output"])
    complete_ref, raw = checked_file(output / "COMPLETE.json", True)
    complete = json.loads(raw.decode())
    if len(owned) != 1:
        raise RuntimeError("One exact worker Pod required")
    expected = {
        "status": "completed_full_ddr_development_preparation",
        "run_id": plan["run_id"],
        "plan_sha256": core.sha(BASE / "PLAN.json"),
        "pod_uid": owned[0]["metadata"]["uid"],
        "parameter_updates": 0,
        "LARRY_access": False,
        "D1_endpoint_access": False,
        "preparation_only": True,
    }
    if any(complete.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Preparation completion role/workload/identity differs")
    sources = complete.get("sources", [])
    if (
        not isinstance(sources, list)
        or [item.get("source_id") for item in sources] != plan["sources"]
        or any(
            type(item.get("rows")) is not int or item["rows"] <= 0 for item in sources
        )
        or sum(item["rows"] for item in sources) != plan["expected_eligible_rows"]
        or complete.get("eligible_rows") != plan["expected_eligible_rows"]
    ):
        raise RuntimeError("Preparation source/eligible-row coverage differs")
    files = complete.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Complete result inventory required")
    inventory, total = {}, 0
    for item in files:
        name = item.get("path")
        if not isinstance(name, str):
            raise RuntimeError("Result path must be a string")
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.as_posix() != name
            or name == "COMPLETE.json"
            or name in inventory
        ):
            raise RuntimeError("Unsafe or duplicate result path")
        observed, _ = checked_file(output / relative)
        if observed != {
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }:
            raise RuntimeError("Result inventory bytes differ")
        inventory[name] = observed
        total += observed["size_bytes"]
    actual = set()
    for path in output.rglob("*"):
        info = path.lstat()
        if info.st_uid != 307469 or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise RuntimeError("Unexpected output owner/special member")
        if stat.S_ISREG(info.st_mode):
            actual.add(path.relative_to(output).as_posix())
    if (
        actual != set(inventory) | {"COMPLETE.json"}
        or complete.get("output_bytes") != total
        or total + complete_ref["size_bytes"] > plan["maximum_new_output_bytes"]
    ):
        raise RuntimeError("Complete output membership/byte budget differs")
    _, release_raw = checked_file(root / "startup-validated.json", True)
    release = json.loads(release_raw.decode())
    startup_ref, _ = checked_file(output / "STARTUP.json")
    if (
        release.get("uid") != binding["uid"]
        or release.get("startup_sha256") != startup_ref["sha256"]
        or release.get("actual_pod_audit_passed") is not True
    ):
        raise RuntimeError("Preparation did not cross actual-Pod startup release")
    if checked_file(output / "COMPLETE.json")[0] != complete_ref:
        raise RuntimeError("Completion changed during validation")
    return True


def monitor(plan, binding, root, static=False):
    if static:
        raise RuntimeError("This controller does not launch another neutral static job")
    while True:
        current = core.get_job(binding["job"])
        core.verify(current, binding, False)
        passed, failed = core.complete(current)
        if not (root / "startup-validated.json").exists() and not failed:
            try:
                core.release_startup(plan, binding, root, current)
            except Exception as error:
                core.publish(
                    root / "startup-failure.json",
                    {"error": repr(error), "uid": binding["uid"]},
                )
                core.capture(current, root)
                core.cleanup(binding, root)
                raise
        if passed or failed:
            owned = core.capture(current, root)
            application, reason = False, None
            try:
                application = bool(
                    passed
                    and not failed
                    and validate_result(plan, binding, root, owned)
                )
            except Exception as error:
                reason = repr(error)
            core.publish(
                root / "terminal.json",
                {
                    "status": "passed" if application else "failed",
                    "job": binding["job"],
                    "uid": binding["uid"],
                    "run_id": plan["run_id"],
                    "kubernetes_complete": passed,
                    "application_pass": application,
                    "reason": reason,
                    "scientific_promotion": False,
                    "plan_sha256": core.sha(BASE / "PLAN.json"),
                    "result_complete_sha256": core.sha(
                        Path(plan["worker_output"]) / "COMPLETE.json"
                    )
                    if application
                    else None,
                },
            )
            core.cleanup(binding, root)
            if not application:
                raise RuntimeError(
                    "Preparation failed; evidence retained; no automatic retry"
                )
            return
        time.sleep(30)


def main():
    global core
    path = BASE / "lifecycle_core.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORE_SHA:
        raise RuntimeError("Exact no-deadline lifecycle changed")
    spec = importlib.util.spec_from_file_location("ddr_prepare_lifecycle", str(path))
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    plan = json.loads((BASE / "PLAN.json").read_text())
    if len(sys.argv) > 1 and sys.argv[1] in ("create", "launch", "supervise"):
        from gpu_v2_static import validate

        validate(BASE, plan)
    core.monitor = monitor
    core.main()


if __name__ == "__main__":
    main()
