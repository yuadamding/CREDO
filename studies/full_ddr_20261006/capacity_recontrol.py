"""Recovery-specific result audit, then unchanged exact-owner lifecycle."""
import json
from pathlib import Path
import qualify_control as prior
import preparation_control as base


def validate_result(plan, binding, root, owned):
    prior.validate_result(plan, binding, root, owned)
    _, raw = base.checked_file(Path(plan["worker_output"]) / "COMPLETE.json", True)
    complete = json.loads(raw.decode())
    if (complete.get("recovery_parent_sha256") != plan["recovery_parent_sha256"]
            or plan.get("maximum_new_probe_attempts") != 22
            or plan.get("maximum_total_probe_attempts_including_parent") != 30):
        raise RuntimeError("Recovery parent/workload differs")
    total = 0
    for result, budget in zip(complete["results"], plan["candidate_probe_budgets"]):
        rows = result["probes"]
        if (result["candidate_id"] != budget["candidate_id"]
                or not 1 <= len(rows) <= budget["maximum_new_probes"]
                or result.get("probe_attempts") != len(rows)
                or result.get("initial_groups") != budget["initial_groups"]
                or not 16 <= result.get("available_groups", 0) <= 4096
                or rows[0]["groups"] != min(budget["initial_groups"], result["available_groups"])
                or result.get("geometry_backend") not in {
                    "geomloss_tensorized", "frozen_80_iteration_fallback"}):
            raise RuntimeError("Recovery per-candidate probes differ")
        for row in rows:
            if (row.get("geometry_backend") != result["geometry_backend"]
                    or row["peak_allocated_bytes"] > 75_000_000_000
                    or row["peak_reserved_bytes"] > row.get("reserved_admission_ceiling_bytes", 0)):
                raise RuntimeError("Recovery memory/backend audit failed")
        reached = 65_000_000_000 <= rows[-1]["peak_allocated_bytes"] <= 75_000_000_000
        if (result.get("target_vram_reached") != reached
                or result.get("measured_groups") != rows[-1]["groups"]
                or result.get("productive_memory_only") is not True
                or result.get("scientific_optimizer_updates") != 0):
            raise RuntimeError("Recovery capacity conclusion differs from measured probes")
        total += len(rows)
    if total > 22:
        raise RuntimeError("Recovery exceeded inherited probe budget")
    return True


if __name__ == "__main__":
    base.validate_result = validate_result
    base.main()
