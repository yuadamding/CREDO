"""One new immutable capacity-only recovery; all parent evidence is preserved."""
import argparse
import copy
import json
import os
from pathlib import Path
import qualify_launch as prior
import prepare_launch as base

ROOT = Path(__file__).absolute().parent
RUN_ID = "credo-full-ddr-capacity-20261007-v2"
SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/capacity_v2"
HOME_RUN = "/rsrch8/home/bcb/yding4" + SUFFIX
CONTROL_ROOT = "/rsrch8/scratch/bcb/yding4" + SUFFIX


def assemble():
    payload = prior.assemble()
    for name in ("RECOVERY_PARENT.json", "qualify_control.py"):
        payload[name] = base.read_regular(ROOT / name)
    for name in ("capacity_relaunch.py", "capacity_reoperate.py", "capacity_restatus.py"):
        payload["controller/" + name] = base.read_regular(ROOT / name)
    payload["control.py"] = base.read_regular(ROOT / "capacity_recontrol.py")
    payload["publication.py"] = payload["publication.py"].replace(prior.RUN_ID.encode(), RUN_ID.encode()).replace(
        prior.HOME_RUN.encode(), HOME_RUN.encode()).replace(prior.CONTROL_ROOT.encode(), CONTROL_ROOT.encode())
    implementation = {name: base.sha(raw) for name, raw in sorted(payload.items())
                      if name.endswith(".py") and name not in {"control.py", "publication.py"}}
    payload["IMPLEMENTATION.json"] = base.canonical(implementation)
    plan = json.loads(payload["PLAN.json"])
    plan.update(schema="credo.all4_ddr.capacity_execution.v2", schema_id="credo.all4_full_ddr.capacity.v2",
        run_id=RUN_ID, bundle_root=HOME_RUN + "/bundle", control_root=CONTROL_ROOT,
        worker_output=HOME_RUN + "/output", output=HOME_RUN + "/output",
        metadata_path=HOME_RUN + "/bundle/METADATA_PREPARATION.json",
        authority_path=HOME_RUN + "/bundle/AUTHORITY.json",
        implementation_sha256=base.sha(payload["IMPLEMENTATION.json"]),
        control_sha256=base.sha(payload["control.py"]), worker_sha256=base.sha(payload["worker.py"]),
        recovery_parent_sha256=base.sha(payload["RECOVERY_PARENT.json"]),
        maximum_new_probe_attempts=22, maximum_total_probe_attempts_including_parent=30,
        memory_ladder_policy="quadratic_or_observed_steeper_with_10pct_cushion_and_4GB_live_headroom",
        candidate_probe_budgets=[dict(candidate_id=cid, previous_probe_attempts=used,
            maximum_new_probes=remaining, initial_groups=initial) for cid, used, remaining, initial in [
                ("ddr_mlp128_e8", 8, 2, 1024), ("ddr_res256_e8", 0, 10, 16), ("ddr_res256_e16", 0, 10, 16)]])
    payload["PLAN.json"] = base.canonical(plan)
    job = copy.deepcopy(json.loads(payload["gpu-job.json"]))
    job["metadata"]["name"] = RUN_ID
    job["metadata"]["annotations"] = {"seadragon_run_id": RUN_ID}
    worker = job["spec"]["template"]["spec"]["containers"][0]
    worker["command"] = ["/opt/credo/bin/python", HOME_RUN + "/bundle/worker.py"]
    for entry in worker["env"]:
        if entry["name"] == "CREDO_RUN_ID":
            entry["value"] = RUN_ID
        elif entry["name"] == "CREDO_PLAN_SHA256":
            entry["value"] = base.sha(payload["PLAN.json"])
        elif entry["name"] == "PYTHONPATH":
            entry["value"] = HOME_RUN + "/bundle/source/src"
    payload["gpu-job.json"] = base.canonical(job)
    return payload


def build(destination):
    members, archive, receipt = base.pack(assemble())
    receipt.update(run_id=RUN_ID, schema="credo.all4_full_ddr.capacity_bundle.v2")
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    for name, raw in dict({"bundle/" + name: raw for name, raw in members.items()},
                           **{"bundle.tar.gz": archive, "BUILD.json": base.canonical(receipt)}).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400), "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if base.read_regular(path) != raw:
            raise ValueError("Published bytes differ")
    for path in sorted((p for p in (destination / "bundle").rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o500)
    (destination / "bundle").chmod(0o500)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = build(parser.parse_args().output)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, sort_keys=True))
