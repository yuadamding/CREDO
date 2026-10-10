"""Capacity-specific terminal audit using unchanged exact-owner lifecycle."""
import json
from pathlib import Path
import stat
import preparation_control as base


def validate_result(plan, binding, root, owned):
    output = Path(plan["worker_output"])
    reference, raw = base.checked_file(output / "COMPLETE.json", True)
    complete = json.loads(raw.decode())
    if len(owned) != 1:
        raise RuntimeError("One exact worker Pod required")
    expected = dict(status="qualified_full_ddr_capacity", run_id=plan["run_id"],
        plan_sha256=base.core.sha(base.BASE / "PLAN.json"), pod_uid=owned[0]["metadata"]["uid"],
        parameter_updates=0, scientific_optimizer_updates=0, D1_endpoint_access=False,
        LARRY_access=False, D2_scores_computed=False, scientific_training_complete=False)
    if any(complete.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Capacity completion scope differs")
    results = complete.get("results", [])
    if [row.get("candidate_id") for row in results] != [row["candidate_id"] for row in plan["candidates"]]:
        raise RuntimeError("Incomplete architecture capacity coverage")
    for result in results:
        if not result.get("probes") or not 1 <= len(result["probes"]) <= plan["max_probes_per_candidate"]:
            raise RuntimeError("Probe workload differs")
        for row in result["probes"]:
            if (row.get("parameter_updates") != 0 or row.get("particles") != 512 or row.get("steps") != 8
                    or not 0 < row.get("peak_allocated_bytes", 0) <= row.get("peak_reserved_bytes", 0) <= 78_000_000_000):
                raise RuntimeError("Probe memory, updates, or execution shape differs")
    inventory = {}
    for item in complete.get("files", []):
        name = item["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or name in inventory:
            raise RuntimeError("Unsafe result inventory")
        actual, _ = base.checked_file(output / relative)
        if actual != dict(sha256=item["sha256"], size_bytes=item["size_bytes"]):
            raise RuntimeError("Result inventory changed")
        inventory[name] = actual
    actual_names = set()
    for path in output.rglob("*"):
        info = path.lstat()
        if info.st_uid != 307469 or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise RuntimeError("Unexpected result member")
        if stat.S_ISREG(info.st_mode):
            actual_names.add(path.relative_to(output).as_posix())
    total = sum(row["size_bytes"] for row in inventory.values())
    if actual_names != set(inventory) | {"COMPLETE.json"} or total != complete.get("output_bytes") or total + reference["size_bytes"] > plan["maximum_output_bytes"]:
        raise RuntimeError("Output membership or budget differs")
    _, raw = base.checked_file(root / "startup-validated.json", True)
    release = json.loads(raw.decode())
    startup, _ = base.checked_file(output / "STARTUP.json")
    if release.get("uid") != binding["uid"] or release.get("startup_sha256") != startup["sha256"] or release.get("actual_pod_audit_passed") is not True:
        raise RuntimeError("Actual-Pod release absent")
    if base.checked_file(output / "COMPLETE.json")[0] != reference:
        raise RuntimeError("Completion changed")
    return True


if __name__ == "__main__":
    base.validate_result = validate_result
    base.main()
