"""Import the completed v4 compiler prerequisite without replaying its Job."""
from __future__ import annotations

import json
from pathlib import Path

import compiler_toolchain as tool

OLD_PLAN = "f398af95da33ab9e46420b36aca1f3861c4218207d4bf93c5158a912ecf1c81f"
OLD_JOB = "credo-full-ddr-compiler-static-20261009-v4"
OLD_UID = "59e2df79-79c9-4aa4-a8f2-766ed17e2d91"
OLD_SPEC = "0737f9e14a41a157d4a5ab2bbfe9cabd24b8c671b643dca987ddc6f01c02878d"
OLD_GATE_SHA = "81b44bab73951dffad88d90a702b97ef5023b3acf7ab8e9d7ced8dd57afbeebf"
OLD_COMPLETE_SHA = "4f5c2b6f6afac56a9138c5b71ff4d6fb904c74c8fb6e7487e415bd999224baef"
OLD_CLEANUP_SHA = "0dc6f1b72c10a8e3ecc24189b7126f67b072b130b962c0f702e630846d5a942c"
OLD_SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_qualification_v4"
OLD_HOME = "/rsrch8/home/bcb/yding4" + OLD_SUFFIX
OLD_CONTROL = "/rsrch8/scratch/bcb/yding4" + OLD_SUFFIX


def import_binding():
    return dict(schema="credo.compiler_prerequisite_import.v1", no_new_static_job=True,
        previous_qualification_plan_sha256=OLD_PLAN, job=OLD_JOB, uid=OLD_UID,
        compiler_spec_sha256=OLD_SPEC, gate_path=OLD_HOME + "/compiler_preflight/COMPILER_PASS.json",
        gate_sha256=OLD_GATE_SHA, compiler_complete_sha256=OLD_COMPLETE_SHA,
        cleanup_source_path=OLD_CONTROL + "/compiler/cleanup.json", cleanup_sha256=OLD_CLEANUP_SHA,
        cleanup_bundle_member="COMPILER_IMPORT_CLEANUP.json",
        native_builds_previously_completed=2, native_builds_per_new_worker=2,
        historical_scientific_optimizer_updates=0)


def verified_imported_compiler_gate(plan):
    """Keep producer identity historical and validate the current import key."""
    binding = plan.get("compiler_preflight_import")
    if binding != import_binding():
        raise RuntimeError("Unrecognized compiler prerequisite import")
    if (plan.get("compiler_spec_sha256") != OLD_SPEC
            or plan.get("compiler_preflight_job") != OLD_JOB
            or plan.get("compiler_preflight_gate_path") != binding["gate_path"]
            or plan.get("compiler_preflight_qualification_plan_sha256") != OLD_PLAN):
        raise RuntimeError("Current compiler import key differs")
    gate_raw = tool.checked_file(binding["gate_path"], OLD_GATE_SHA, maximum=16*1024**2)
    gate = json.loads(gate_raw)
    if (gate.get("uid") != OLD_UID or gate.get("compiler_complete_sha256") != OLD_COMPLETE_SHA
            or gate.get("cleanup_reference", {}).get("sha256") != OLD_CLEANUP_SHA):
        raise RuntimeError("Historical compiler producer identity differs")
    complete = json.loads(tool.checked_file(Path(binding["gate_path"]).parent / "COMPLETE.json",
        OLD_COMPLETE_SHA, maximum=16*1024**2))
    if (complete.get("image") != plan["image"] or complete.get("plan_sha256") != OLD_PLAN
            or complete.get("data_free") is not True):
        raise RuntimeError("Historical compiler image or producer plan differs")
    cleanup_path = Path(plan["bundle_root"]) / binding["cleanup_bundle_member"]
    cleanup = json.loads(tool.checked_file(cleanup_path, OLD_CLEANUP_SHA, maximum=16*1024**2))
    if (cleanup.get("status") != "complete" or cleanup.get("uid") != OLD_UID
            or cleanup.get("reads", [])[-2:] != [dict(job_absent=True, owned_pods=0)]*2):
        raise RuntimeError("Historical compiler cleanup differs")
    historical = dict(plan, phase="fitting", compiler_preflight_qualification_plan_sha256=OLD_PLAN)
    return tool.verified_compiler_gate(historical)
