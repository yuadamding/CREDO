"""One H100 capacity qualification on admitted fitting populations; zero updates."""
from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import time

from prepare_manifest import canonical
from prepare_worker import (
    checked_json, file_identity, load_plan, startup_record, verify_bundle, wait_for_release,
)
from encode_sources import publish_json


def validate_plan(plan):
    recovery = plan.get("schema") == "credo.all4_ddr.capacity_execution.v2"
    expected = dict(schema="credo.all4_ddr.capacity_execution.v1", parameter_updates=0,
                    scientific_optimizer_updates=0, D1_endpoint_access=False, LARRY_access=False,
                    automatic_retries=0, gpu_ceiling=1, max_probes_per_candidate=10,
                    runtime_deadline_policy="no_runtime_deadline_user_20260919")
    if recovery:
        expected["schema"] = "credo.all4_ddr.capacity_execution.v2"
        budgets = plan.get("candidate_probe_budgets")
        if budgets != [dict(candidate_id=cid, previous_probe_attempts=prior,
                maximum_new_probes=remaining, initial_groups=initial)
                for cid, prior, remaining, initial in [
                    ("ddr_mlp128_e8", 8, 2, 1024), ("ddr_res256_e8", 0, 10, 16),
                    ("ddr_res256_e16", 0, 10, 16)]]:
            raise ValueError("Recovery must inherit consumed and remaining probe budgets")
        if (plan.get("maximum_total_probe_attempts_including_parent") != 30
                or plan.get("maximum_new_probe_attempts") != 22
                or len(plan.get("recovery_parent_sha256", "")) != 64):
            raise ValueError("Recovery authority/budget differs")
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Qualification scope/workload differs")
    if [row["candidate_id"] for row in plan["candidates"]] != ["ddr_mlp128_e8", "ddr_res256_e8", "ddr_res256_e16"]:
        raise ValueError("Only the three frozen candidates may be qualified")


def main(argv=None):
    root, plan, plan_sha = load_plan(argv, root=Path(__file__).absolute().parent)
    validate_plan(plan)
    if root != Path(plan["bundle_root"]):
        raise RuntimeError("Worker bundle root differs")
    verify_bundle(root, plan)
    if (os.getuid(), os.getgid()) != (307469, 600815):
        raise RuntimeError("Institutional writer differs")
    output = Path(plan["output"])
    if os.path.lexists(output) or any(p.is_symlink() for p in (output, *output.parents)):
        raise RuntimeError("New absent output required")
    output.mkdir(mode=0o700)
    start = time.perf_counter()
    try:
        import torch
        import pyarrow as pa
        from development_inputs import prepared_development
        from memory_qualification import qualify, qualification_config
        from native_study import compile_development
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("One H100 required, no CPU fallback")
        prop = torch.cuda.get_device_properties(0)
        if "H100" not in prop.name or "MIG" in prop.name or prop.total_memory < 75_000_000_000:
            raise RuntimeError("Full H100 80GB required")
        if torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError("Hardware capability differs")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        pa.set_cpu_count(8)
        pa.set_io_thread_count(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        canary = output / "writer-canary"
        publish_json(canary, {"owner": os.getuid()})
        if canary.stat().st_uid != os.getuid():
            raise RuntimeError("Canary owner differs")
        canary.unlink()
        startup = startup_record(plan, plan_sha, prop, torch)
        publish_json(output / "STARTUP.json", startup)
        wait_for_release(output, startup)
        metadata = checked_json(root / "METADATA_PREPARATION.json", plan["metadata_sha256"])
        inputs = checked_json(root / "DEVELOPMENT_INPUTS.json", plan["development_inputs_sha256"])
        checked_json(root / "AUTHORITY.json", plan["authority_sha256"])
        if "recovery_parent_sha256" in plan:
            checked_json(root / "RECOVERY_PARENT.json", plan["recovery_parent_sha256"])

        def progress(row):
            row = dict(row, utc=datetime.now(timezone.utc).isoformat(), parameter_updates=0)
            with (output / "PROGRESS.jsonl").open("ab") as stream:
                stream.write(canonical(row))
                stream.flush()
            print(canonical(row).decode(), flush=True)

        progress({"phase": "verified_latent_input_opening"})
        results = []
        with prepared_development(inputs, metadata, authority_sha256=plan["authority_sha256"],
                input_sha256=plan["development_inputs_sha256"], metadata_sha256=plan["metadata_sha256"]) as (bundle, observations):
            progress({"phase": "native_study_built", "source_supported_series": len(bundle.source_supported_series)})
            problem, split = compile_development(bundle, qualification_config(plan["candidates"][0]["model"]))
            if set(problem.training.measure_meta.sample_id) != {"D3", "D4"}:
                raise RuntimeError("Qualification fitting split differs")
            support = dict(fitting_series=len(problem.training.measure_ids), selection_series=len(problem.validation.measure_ids),
                fitting_targets=len(problem.training.embedding_ids), selection_split=split.to_dict(),
                complete_guide_catalog=inputs["guide_count"], sources=[dict(source_id=v.source_id,
                    vector_sha256=v.vector_sha256, captured_cells=sum(v.counts)) for v in observations],
                dispositions=list(bundle.dispositions))
            publish_json(output / "SUPPORT.json", support)
            for candidate_index, candidate in enumerate(plan["candidates"]):
                progress({"phase": "capacity_candidate", "candidate_id": candidate["candidate_id"]})
                budget = plan.get("candidate_probe_budgets", [dict(maximum_new_probes=10,
                    initial_groups=16)] * len(plan["candidates"]))[candidate_index]
                result = qualify(problem, candidate["model"], max_probes=budget["maximum_new_probes"],
                    initial_groups=budget["initial_groups"],
                    progress=lambda row, cid=candidate["candidate_id"]: progress(dict(row, candidate_id=cid, phase="capacity_probe")))
                results.append(dict(result, candidate_id=candidate["candidate_id"]))
        # Input snapshot paths are revalidated at context exit, before success.
        files = []
        for path in sorted(output.rglob("*")):
            info = path.lstat()
            if info.st_uid != 307469 or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise RuntimeError("Unexpected output member")
            if stat.S_ISREG(info.st_mode):
                files.append(dict(path=path.relative_to(output).as_posix(), **file_identity(path, owner=307469)))
        complete = dict(status="qualified_full_ddr_capacity", run_id=plan["run_id"], plan_sha256=plan_sha,
            pod_uid=startup["pod_uid"], implementation_sha256=plan["implementation_sha256"],
            parameter_updates=0, scientific_optimizer_updates=0, D1_endpoint_access=False, LARRY_access=False,
            D2_scores_computed=False, scientific_training_complete=False, scientific_promotion=False,
            results=results, files=files, output_bytes=sum(r["size_bytes"] for r in files),
            recovery_parent_sha256=plan.get("recovery_parent_sha256"),
            elapsed_seconds=time.perf_counter() - start, completed_utc=datetime.now(timezone.utc).isoformat())
        if complete["output_bytes"] + len(canonical(complete)) > plan["maximum_output_bytes"]:
            raise RuntimeError("Qualification output cap exceeded")
        publish_json(output / "COMPLETE.json", complete)
    except BaseException as error:
        publish_json(output / "FAILED.json", dict(status="failed", error=repr(error), run_id=plan["run_id"],
            parameter_updates=0, automatic_retry=False, utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    main()
