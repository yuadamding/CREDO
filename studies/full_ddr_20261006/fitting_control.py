"""Fitting-only terminal validator; reuse the exact no-deadline owner lifecycle.

Host Python 3.6 compatible. This controller never selects or scores a model.
"""
import json
import math
from pathlib import Path
import stat

import preparation_control as base


def validate_result(plan, binding, root, owned):
    output = Path(plan["worker_output"])
    reference, raw = base.checked_file(output / "COMPLETE.json", True)
    complete = json.loads(raw.decode())
    if len(owned) != 1:
        raise RuntimeError("One exact fitting worker required")
    expected = dict(status="completed_full_ddr_architecture_fitting", run_id=plan["run_id"],
        plan_sha256=base.core.sha(base.BASE / "PLAN.json"), pod_uid=owned[0]["metadata"]["uid"],
        implementation_sha256=plan["implementation_sha256"],
        fitting_contract_sha256=plan["fitting_contract_sha256"],
        capacity_binding_sha256=plan["capacity_binding_sha256"], input_lease_exit_verified=True,
        scientific_training_complete=True, scientific_promotion=False,
        query_deployment_performed=False, assessment_performed=False,
        scientific_acceptance="not_assessed_fitting_only", D1_endpoint_access=False,
        D2_scores_computed=False, LARRY_access=False)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Fitting completion scope or identity differs")
    inventory = {}
    for item in complete.get("files", []):
        name = item["path"]
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or relative.as_posix() != name or name in inventory or name == "COMPLETE.json"):
            raise RuntimeError("Unsafe fitting result inventory")
        actual, _ = base.checked_file(output / relative)
        if actual != dict(sha256=item["sha256"], size_bytes=item["size_bytes"]):
            raise RuntimeError("Fitting result bytes changed")
        inventory[name] = actual
    names = set()
    for path in output.rglob("*"):
        info = path.lstat()
        if info.st_uid != 307469 or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise RuntimeError("Unexpected fitting output member")
        if stat.S_ISREG(info.st_mode):
            names.add(path.relative_to(output).as_posix())
    size = sum(item["size_bytes"] for item in inventory.values())
    if (not inventory or names != set(inventory) | {"COMPLETE.json"}
            or size != complete["output_bytes"] or size+reference["size_bytes"] > plan["maximum_output_bytes"]):
        raise RuntimeError("Fitting output coverage or budget differs")
    _, raw = base.checked_file(output / "COMPILED_STUDY.json", True)
    compiled = json.loads(raw.decode())
    pairs = compiled.get("fitting_state_pairs")
    if (type(pairs) is not int or not 0 < pairs <= 46809 or compiled.get("fitting_series") != 46809
            or compiled.get("fitting_targets") != 12555):
        raise RuntimeError("Compiled fitting support differs")
    expected_updates = 40*int(math.ceil(pairs/1086.0))+20*44
    if (compiled.get("exact_optimizer_updates_per_fit") != expected_updates
            or compiled.get("exact_optimizer_updates_total") != 9*expected_updates
            or complete.get("scientific_optimizer_updates") != 9*expected_updates
            or complete.get("attempted_optimizer_updates") != 9*expected_updates
            or 9*expected_updates > plan["maximum_optimizer_updates"]):
        raise RuntimeError("Fitting scientific update accounting differs")
    _, raw = base.checked_file(output / "NORMALIZERS.json", True)
    scales = json.loads(raw.decode())
    if (scales.get("policy") != "balanced_fitting_scales_active_coordinate_count_v1"
            or scales.get("fitting_donors") != ["D3", "D4"] or scales.get("state_laws") != pairs
            or scales.get("particles") != 512 or scales.get("seed") != 104729
            or scales.get("geometry_backend") != "geomloss_tensorized"
            or scales.get("model_forward_calls") != 0 or scales.get("optimizer_updates") != 0
            or any(not isinstance(scales.get(k), (float, int)) or not math.isfinite(scales[k]) or scales[k] <= 0
                   for k in ("state", "count"))):
        raise RuntimeError("Training normalizer scope or positivity differs")
    fits = complete.get("fits", [])
    if len(fits) != 9 or [{k: row.get(k) for k in ("candidate_id", "seed")} for row in fits] != plan["fits"]:
        raise RuntimeError("Incomplete architecture/seed inventory")
    for row in fits:
        checkpoint = "fits/"+row["candidate_id"]+"/seed-"+str(row["seed"])+"/state/checkpoint.pt"
        fit_complete = checkpoint.rsplit("/state/", 1)[0]+"/FIT_COMPLETE.json"
        _, raw = base.checked_file(output / fit_complete, True)
        if json.loads(raw.decode()) != row or row.get("checkpoint_path") != checkpoint or inventory.get(checkpoint) != row.get("checkpoint"):
            raise RuntimeError("Terminal checkpoint/publication binding differs")
        if (row.get("status") != "terminal_fitting_complete_pending_input_lease_exit"
                or row.get("scientific_assessment") != "not_performed_fitting_only"
                or row.get("optimizer_updates") != expected_updates or row.get("attempted_optimizer_updates") != expected_updates
                or row.get("epochs_completed") != 60 or row.get("guide_exposures") != 40*pairs+20*46809
                or row.get("particles") != 512 or row.get("steps") != 8 or row.get("parameter_count", 0) <= 0
                or row.get("empty_assessment_table_rows") != dict(metrics=0, predictions=0, counterfactuals=0)
                or row.get("fitting_contract_sha256") != plan["fitting_contract_sha256"]
                or row.get("normalizers_sha256") != inventory.get("NORMALIZERS.json", {}).get("sha256")):
            raise RuntimeError("Terminal fit workload or role check failed")
        admissions = row.get("optimizer_memory_admissions", [])
        if len(admissions) != 2 or [v.get("stage") for v in admissions] != ["state", "mass"]:
            raise RuntimeError("Missing stage optimizer-memory admission")
        for value in admissions+[row]:
            if not 0 < value.get("peak_allocated_bytes", 0) <= value.get("peak_reserved_bytes", 0) <= 78_000_000_000 or value["peak_allocated_bytes"] > 75_000_000_000:
                raise RuntimeError("Optimizer-inclusive memory limit exceeded")
        if any(v.get("optimizer_state_bytes", 0) <= 0 for v in admissions):
            raise RuntimeError("Adam state not admitted")
        heads = row.get("head_gradient_maxima", {})
        if set(heads) != {"drift", "diffusion", "reaction"} or any(not math.isfinite(v) or v <= 0 for v in heads.values()):
            raise RuntimeError("Full DDR heads did not receive finite nonzero gradients")
    _, raw = base.checked_file(root / "startup-validated.json", True)
    release = json.loads(raw.decode())
    startup, _ = base.checked_file(output / "STARTUP.json")
    if (release.get("uid") != binding["uid"] or release.get("startup_sha256") != startup["sha256"]
            or release.get("actual_pod_audit_passed") is not True):
        raise RuntimeError("Actual-Pod startup release differs")
    if base.checked_file(output / "COMPLETE.json")[0] != reference:
        raise RuntimeError("Completion changed")
    return True


if __name__ == "__main__":
    base.validate_result = validate_result
    base.main()
