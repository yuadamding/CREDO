"""Build a fresh immutable capacity attempt, preserving completed preparation."""
import argparse
import copy
import json
import os
from pathlib import Path
import prepare_launch as base

ROOT = Path(__file__).absolute().parent
RUN_ID = "credo-full-ddr-capacity-20261007-v1"
SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/capacity_v1"
HOME_RUN = "/rsrch8/home/bcb/yding4" + SUFFIX
CONTROL_ROOT = "/rsrch8/scratch/bcb/yding4" + SUFFIX


def assemble():
    payload = base.assemble(ROOT)
    payload["prepare_worker.py"] = base.read_regular(ROOT / "prepare_worker.py")
    payload["preparation_control.py"] = base.read_regular(ROOT / "preparation_control.py")
    for name in ("memory_qualification.py", "development_inputs.py", "DEVELOPMENT_INPUTS.json"):
        payload[name] = base.read_regular(ROOT / name)
    payload["worker.py"] = base.read_regular(ROOT / "qualify_worker.py")
    payload["control.py"] = base.read_regular(ROOT / "qualify_control.py")
    for name in ("qualify_launch.py", "qualify_operate.py"):
        payload["controller/" + name] = base.read_regular(ROOT / name)
    # Byte-reviewed generic publication, with ONLY fresh task roots substituted.
    payload["publication.py"] = base.publication_wrapper().replace(base.RUN_ID.encode(), RUN_ID.encode()).replace(
        base.HOME_RUN.encode(), HOME_RUN.encode()).replace(base.CONTROL_ROOT.encode(), CONTROL_ROOT.encode())
    implementation = {name: base.sha(raw) for name, raw in sorted(payload.items())
                      if name.endswith(".py") and name not in {"control.py", "publication.py"}}
    payload["IMPLEMENTATION.json"] = base.canonical(implementation)
    plan = json.loads(payload["PLAN.json"])
    architectures = json.loads(payload["STUDY_PLAN.json"])["architecture_candidates"]
    plan.update(schema="credo.all4_ddr.capacity_execution.v1", schema_id="credo.all4_full_ddr.capacity.v1",
        run_id=RUN_ID, bundle_root=HOME_RUN + "/bundle", control_root=CONTROL_ROOT,
        worker_output=HOME_RUN + "/output", output=HOME_RUN + "/output",
        maximum_new_output_bytes=1024**3, maximum_output_bytes=1024**3,
        startup_gate_authority="Full DDR study H100 productive capacity; zero scientific updates",
        implementation_sha256=base.sha(payload["IMPLEMENTATION.json"]),
        control_sha256=base.sha(payload["control.py"]), worker_sha256=base.sha(payload["worker.py"]),
        development_inputs_sha256=base.sha(payload["DEVELOPMENT_INPUTS.json"]),
        metadata_path=HOME_RUN + "/bundle/METADATA_PREPARATION.json",
        authority_path=HOME_RUN + "/bundle/AUTHORITY.json", max_probes_per_candidate=10,
        candidates=[dict(candidate_id=row["id"], model={k: v for k, v in row.items() if k != "id"})
                    for row in architectures],
        scientific_optimizer_updates=0, D2_scores_computed=False,
        qualification_particles=512, qualification_steps=8,
        productive_vram_target_bytes=70_000_000_000,
        measured_memory_target_band_bytes=[65_000_000_000, 75_000_000_000],
        reserved_memory_safety_ceiling_bytes=78_000_000_000,
        optimizer_state_memory_qualified=False, training_contract_still_requires_freezing=True,
        output_completion_status="qualified_full_ddr_capacity")
    payload["PLAN.json"] = base.canonical(plan)
    job = copy.deepcopy(json.loads(payload["gpu-job.json"]))
    job["metadata"]["name"] = RUN_ID
    job["metadata"]["annotations"] = {"seadragon_run_id": RUN_ID}
    worker = job["spec"]["template"]["spec"]["containers"][0]
    worker["command"] = ["/opt/credo/bin/python", HOME_RUN + "/bundle/worker.py"]
    for entry in worker["env"]:
        if entry["name"] == "CREDO_RUN_ID":
            entry["value"] = RUN_ID
        if entry["name"] == "CREDO_PLAN_SHA256":
            entry["value"] = base.sha(payload["PLAN.json"])
        if entry["name"] == "PYTHONPATH":
            entry["value"] = HOME_RUN + "/bundle/source/src"
    payload["gpu-job.json"] = base.canonical(job)
    if sum(map(len, payload.values())) > base.MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Qualification bundle exceeds admission")
    return payload


def build(destination):
    members, archive, receipt = base.pack(assemble())
    receipt.update(run_id=RUN_ID, schema="credo.all4_full_ddr.capacity_bundle.v1")
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
