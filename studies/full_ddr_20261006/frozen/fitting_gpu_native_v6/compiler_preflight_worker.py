"""Zero-GPU compiler validation in the same approved image; no data access."""
import hashlib
import json
import os
from pathlib import Path
import platform

from compiler_toolchain import checked_file, prepare_compiler


def main():
    root = Path(__file__).absolute().parent
    raw = checked_file(root / "PLAN.json")
    plan = json.loads(raw)
    if ((os.getuid(), os.getgid()) != (307469, 600815) or platform.python_version() != "3.12.13"
            or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none" or root != Path(plan["bundle_root"])):
        raise RuntimeError("Same-image compiler worker identity differs")
    inventory = json.loads(checked_file(root / "INVENTORY.json"))
    for name in ("compiler_preflight_worker.py", "compiler_toolchain.py", "compiler_probe.c", "compiler_probe.cpp"):
        checked_file(root / name, inventory[name]["sha256"])
    output = Path(plan["compiler_preflight_output"])
    if os.path.lexists(output):
        raise RuntimeError("Absent compiler-check output required")
    output.mkdir(mode=0o700)
    result = prepare_compiler(root, plan)
    if result["torch_package_version"] != "2.6.0+cu124":
        raise RuntimeError("Compiler-check image package differs")
    result.update(run_id=plan["run_id"], plan_sha256=hashlib.sha256(raw).hexdigest(), image=plan["image"],
        gpu_requests=0, gpu_limits=0, visibility_mask="none", data_free=True)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(output / "COMPLETE.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print("CREDO_COMPILER_READY", flush=True)
    print("CREDO_NEUTRAL_IMAGE_STARTED", flush=True)


if __name__ == "__main__":
    main()
