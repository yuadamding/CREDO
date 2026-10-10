"""Freeze a new fitting-only lifecycle without modifying completed parents."""
import argparse
import copy
import json
import os
from pathlib import Path

import prepare_launch as base
import qualify_launch as capacity_builder

ROOT = Path(__file__).absolute().parent
RUN_ID = "credo-full-ddr-fitting-20261007-v1"
SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/fitting_v1"
HOME_RUN = "/rsrch8/home/bcb/yding4" + SUFFIX
CONTROL_ROOT = "/rsrch8/scratch/bcb/yding4" + SUFFIX


def assemble():
    payload = capacity_builder.assemble()
    payload.pop("memory_qualification.py")
    for name in ("FITTING_CONTRACT.json", "FIT_CAPACITY_BINDING.json", "terminal_fitting.py",
                 "architecture_fitting.py"):
        payload[name] = base.read_regular(ROOT / name)
    payload["worker.py"] = base.read_regular(ROOT / "fitting_worker.py")
    payload["control.py"] = base.read_regular(ROOT / "fitting_control.py")
    for name in ("fitting_launch.py", "fitting_operate.py", "fitting_status.py"):
        payload["controller/" + name] = base.read_regular(ROOT / name)
    payload["publication.py"] = payload["publication.py"].replace(capacity_builder.RUN_ID.encode(), RUN_ID.encode()).replace(
        capacity_builder.HOME_RUN.encode(), HOME_RUN.encode()).replace(capacity_builder.CONTROL_ROOT.encode(), CONTROL_ROOT.encode())
    implementation = {name: base.sha(raw) for name, raw in sorted(payload.items())
                      if name.endswith(".py") and name not in {"control.py", "publication.py"}}
    payload["IMPLEMENTATION.json"] = base.canonical(implementation)
    plan = json.loads(payload["PLAN.json"])
    for name in ("max_probes_per_candidate", "qualification_particles", "qualification_steps", "parameter_updates",
                 "scientific_optimizer_updates", "expected_eligible_rows", "authorized_new_source_exports",
                 "batch_rows", "training_contract_still_requires_freezing"):
        plan.pop(name, None)
    candidates = [r["candidate_id"] for r in plan["candidates"]]
    plan.update(schema="credo.all4_ddr.architecture_fitting_execution.v1", schema_id="credo.all4_full_ddr.architecture_fitting.v1",
        run_id=RUN_ID, bundle_root=HOME_RUN + "/bundle", control_root=CONTROL_ROOT,
        worker_output=HOME_RUN + "/output", output=HOME_RUN + "/output",
        maximum_new_output_bytes=16*1024**3, maximum_output_bytes=16*1024**3,
        startup_gate_authority="Full DDR architecture fitting only; 9 fresh D3/D4 fits; no assessment",
        implementation_sha256=base.sha(payload["IMPLEMENTATION.json"]),
        control_sha256=base.sha(payload["control.py"]), worker_sha256=base.sha(payload["worker.py"]),
        metadata_path=HOME_RUN + "/bundle/METADATA_PREPARATION.json",
        authority_path=HOME_RUN + "/bundle/AUTHORITY.json",
        fitting_contract_sha256=base.sha(payload["FITTING_CONTRACT.json"]),
        capacity_binding_sha256=base.sha(payload["FIT_CAPACITY_BINDING.json"]),
        fits=[dict(candidate_id=cid, seed=seed) for cid in candidates for seed in (0, 1, 2)],
        maximum_fits=9, maximum_optimizer_updates=23760, scientific_fitting_only=True, preparation_only=False,
        D2_scores_computed=False, D1_endpoint_access=False, LARRY_access=False,
        optimizer_state_memory_qualified=False, capacity_probe_budget_remaining=0,
        fitting_donors=["D3", "D4"], selection_donor="D2", scientific_promotion=False,
        query_deployment_performed=False, assessment_performed=False,
        output_completion_status="completed_full_ddr_architecture_fitting")
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
    if sum(map(len, payload.values())) > base.MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Fitting bundle exceeds admission")
    return payload


def build(destination):
    members, archive, result = base.pack(assemble())
    result.update(run_id=RUN_ID, schema="credo.all4_full_ddr.fitting_bundle.v1")
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    for name, raw in dict({"bundle/"+name: raw for name, raw in members.items()},
                         **{"bundle.tar.gz": archive, "BUILD.json": base.canonical(result)}).items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400), "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if base.read_regular(path) != raw:
            raise ValueError("Published fitting bytes differ")
    for path in sorted((p for p in (destination / "bundle").rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o500)
    (destination / "bundle").chmod(0o500)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    value = build(parser.parse_args().output)
    print(json.dumps({k: v for k, v in value.items() if k != "files"}, sort_keys=True))
