"""Fresh immutable pre-submission loader correction; no extra scientific work."""
import argparse
import json
from pathlib import Path

import gpu_native_compiler_launch as inherited
import prepare_launch as shared

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT / "gpu_native_compiler_recovery_v4"
ROOT_NAMES = {"qualification": "gpu_native_qualification_v4", "fitting": "fitting_gpu_native_v4"}
QUAL_NAME = "credo-full-ddr-gpu-native-qual-20261009-v4"
FIT_NAME = "credo-full-ddr-fitting-gpu-native-20261009-v4"
CPU_NAME = "credo-full-ddr-compiler-static-20261009-v4"


def roots(phase):
    suffix = inherited.original.STUDY_SUFFIX + ROOT_NAMES[phase]
    return "/rsrch8/home/bcb/yding4" + suffix, "/rsrch8/scratch/bcb/yding4" + suffix


original_assemble = inherited.assemble_phase


def assemble_phase(phase, *, qualification_plan_sha=None):
    payload = original_assemble(phase, qualification_plan_sha=qualification_plan_sha)
    for name in ("gpu_native_compiler_recovery_launch.py", "gpu_native_compiler_recovery_operate.py"):
        payload["controller/" + name] = shared.read_regular(ROOT / name)
    payload["IMPLEMENTATION.json"] = shared.canonical({name: shared.sha(raw) for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}})
    plan = json.loads(payload["PLAN.json"])
    plan["implementation_sha256"] = shared.sha(payload["IMPLEMENTATION.json"])
    payload["PLAN.json"] = shared.canonical(plan)
    job = json.loads(payload["gpu-job.json"])
    for entry in job["spec"]["template"]["spec"]["containers"][0]["env"]:
        if entry["name"] == "CREDO_PLAN_SHA256":
            entry["value"] = shared.sha(payload["PLAN.json"])
    payload["gpu-job.json"] = shared.canonical(job)
    return payload


def bind_builder():
    # This is a dedicated CLI process: old on-disk builders and all bundles
    # remain unchanged. Use the reviewed publishing and conditional-chain code.
    inherited.RECOVERY, inherited.ROOT_NAMES = RECOVERY, ROOT_NAMES
    inherited.QUAL_NAME, inherited.FIT_NAME, inherited.CPU_NAME = QUAL_NAME, FIT_NAME, CPU_NAME
    inherited.compiler_payload.cache_clear()
    inherited.assemble_phase = assemble_phase


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    bind_builder()
    print(json.dumps({phase: {key: value for key, value in row.items() if key != "files"}
        for phase, row in inherited.build_pair().items()}, sort_keys=True))
