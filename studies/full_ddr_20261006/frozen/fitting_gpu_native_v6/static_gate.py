"""One exact zero-GPU, masked same-image neutral startup qualification."""
import json
from pathlib import Path
import time

from control import core
from neutral_admission import audit_neutral, ZERO, LIVE
from runtime_policy import validate_plan, validate_job

def audit(job, manifest):
    validate_job(manifest)
    validate_job(job)
    return audit_neutral(job, manifest)


core.audit = audit


def main():
    bundle = Path(__file__).resolve().parent
    plan = json.loads((bundle / "PLAN.json").read_text())
    validate_plan(plan)
    root = Path(plan["control_root"]) / "static"
    root.mkdir(mode=0o700)
    manifest_path = bundle / "static-job.json"
    manifest = json.loads(manifest_path.read_text())
    name = manifest["metadata"]["name"]
    if core.get_job(name) is not None:
        raise RuntimeError("Static exact identity already exists")
    dry = json.loads(core.kube("create", "--dry-run=server", "-o", "json", "-f", str(manifest_path)))
    core.publish(root / "dryrun.json", dry)
    core.audit(dry, manifest)
    if dry["spec"]["template"]["spec"]["containers"][0]["resources"] not in (ZERO, LIVE):
        raise RuntimeError("Neutral dry-run contract changed")
    core.publish(root / "create-intent.json", dict(name=name, manifest_sha256=core.sha(manifest_path)))
    try:
        created = json.loads(core.kube("create", "-o", "json", "-f", str(manifest_path)))
    except Exception as error:
        core.publish(root / "create-uncertain.json", dict(error=repr(error), job=name, no_retry=True))
        raise
    binding = dict(job=name, uid=created["metadata"]["uid"], run_id=manifest["metadata"]["annotations"]["seadragon_run_id"])
    try:
        core.publish(root / "created.json", created)
        core.audit(created, manifest)
        if created["spec"]["template"]["spec"]["containers"][0]["resources"] not in (ZERO, LIVE):
            raise RuntimeError("Neutral created-Job contract changed")
        binding["contract"] = core.job_contract(created)
        for _ in range(2):
            core.verify(core.get_job(name), binding, True)
            if core.pods(name, binding["uid"]):
                raise RuntimeError("Static suspended job owns pods")
        core.publish(root / "live-binding.json", binding)
        core.activate(binding, root)
        while True:
            job = core.get_job(name)
            core.verify(job, binding, False)
            good, bad = core.complete(job)
            if good or bad:
                owned = core.capture(job, root)
                passed = bool(good and not bad and len(owned) == 1)
                if passed:
                    core.audit_runtime_pod(owned[0], binding, plan)
                    container = owned[0]["spec"]["containers"][0]
                    env = {v["name"]: v.get("value") for v in container["env"]}
                    if env.get("NVIDIA_VISIBLE_DEVICES") != "none" or any(container["resources"][k]["nvidia.com/gpu"] != "0" for k in ("requests", "limits")):
                        raise RuntimeError("Static zero-GPU visibility differs")
                    text = (root / "pod-00.log").read_text()
                    passed = "CREDO_NEUTRAL_IMAGE_STARTED" in text
                core.publish(root / "terminal.json", dict(status="passed" if passed else "failed", job=name, uid=binding["uid"],
                             gpu_requests=0, gpu_limits=0, visibility_mask="none", unmasked_visibility_claimed=False,
                             manifest_sha256=core.sha(manifest_path), plan_sha256=core.sha(bundle / "PLAN.json")))
                core.cleanup(binding, root)
                if not passed:
                    raise RuntimeError("Static qualification failed")
                print("STATIC_PASSED", flush=True)
                return
            time.sleep(10)
    except BaseException as error:
        if not (root / "terminal.json").exists():
            core.publish(root / "failure.json", dict(error=repr(error), uid=binding["uid"]))
        if not (root / "cleanup.json").exists():
            core.cleanup(binding, root, create_scoped=True)
        raise


if __name__ == "__main__":
    main()
