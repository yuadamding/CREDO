"""Verify reuse of the exact completed v1 neutral gate; host Python 3.6.

This imports historical evidence, not a new static run or a GPU test result.
"""

import hashlib
import json
from pathlib import Path


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Regular static authority/source required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(bundle, plan):
    bundle = Path(bundle)
    if digest(bundle / "STATIC_REUSE.json") != plan.get("static_reuse_sha256"):
        raise RuntimeError("Static reuse authority differs")
    authority = json.loads((bundle / "STATIC_REUSE.json").read_text())
    for name, expected in authority["files"].items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or digest(bundle / name) != expected
        ):
            raise RuntimeError("Static evidence/source differs: " + name)
    previous = json.loads((bundle / "static_proof/PLAN.json").read_text())
    terminal = json.loads((bundle / "static_proof/terminal.json").read_text())
    cleanup = json.loads((bundle / "static_proof/cleanup.json").read_text())
    manifest = json.loads((bundle / "static-job.json").read_text())
    if (
        authority.get("status") != "reuse_completed_neutral_gate_no_new_static_job"
        or authority.get("parent_gpu_job_absent") is not True
        or authority.get("parent_gpu_pods_absent") is not True
        or terminal.get("status") != "passed"
        or cleanup.get("status") != "complete"
        or not terminal.get("uid")
        or terminal["uid"] != cleanup.get("uid")
        or terminal.get("job") != manifest["metadata"]["name"]
        or terminal.get("job") != cleanup.get("job")
        or terminal.get("plan_sha256") != digest(bundle / "static_proof/PLAN.json")
        or terminal.get("manifest_sha256") != digest(bundle / "static-job.json")
        or terminal.get("gpu_requests") != 0
        or terminal.get("gpu_limits") != 0
        or terminal.get("visibility_mask") != "none"
        or terminal.get("unmasked_visibility_claimed") is not False
        or cleanup.get("reads", [])[-2:] != [dict(job_absent=True, owned_pods=0)] * 2
        or previous["image"] != plan["image"]
    ):
        raise RuntimeError("Completed masked static proof is invalid")
    static_pod = manifest["spec"]["template"]["spec"]
    gpu_pod = json.loads((bundle / "gpu-job.json").read_text())["spec"]["template"][
        "spec"
    ]
    static, gpu = static_pod["containers"][0], gpu_pod["containers"][0]
    for key in ("image", "securityContext", "volumeMounts"):
        if static[key] != gpu[key]:
            raise RuntimeError("Static execution surface changed: " + key)
    for key in (
        "volumes",
        "securityContext",
        "automountServiceAccountToken",
        "enableServiceLinks",
    ):
        if static_pod[key] != gpu_pod[key]:
            raise RuntimeError("Static Pod boundary changed: " + key)
    if static["command"] != [
        "/opt/credo/bin/python",
        "-c",
        "print('CREDO_NEUTRAL_IMAGE_STARTED')",
    ]:
        raise RuntimeError("Neutral command changed")
    return True
