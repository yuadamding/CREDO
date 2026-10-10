"""Qualification or fresh fitting at disjoint immutable phase identities."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from encode_sources import publish_json
from fitting_worker import failure_accounting, output_inventory, recipe_config
from prepare_manifest import canonical
from prepare_worker import checked_json, file_identity, load_plan, startup_record, verify_bundle, wait_for_release


def validate_plan(plan, contract, authority):
    phase = plan.get("phase")
    if phase not in {"qualification", "fitting"}:
        raise ValueError("Unknown GPU-native phase")
    expected = dict(gpu_ceiling=1, automatic_retries=0, authorized_CPUs=24, authorized_memory_GiB=200,
        threads=8, D1_endpoint_access=False, D2_scores_computed=False, LARRY_access=False,
        runtime_deadline_policy="no_runtime_deadline_user_20260919", geomloss_dependency_surface_required=True)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Execution scope/resources changed")
    if (authority.get("status") != "user_authorized_qualification_and_conditional_fresh_fitting"
            or authority.get("prepared_contract_sha256") != plan["gpu_contract_sha256"]
            or authority.get("maximum_new_scientific_updates") != 23040
            or authority.get("maximum_loss_backward_calls") != 102
            or contract["qualification"]["maximum_total_loss_backward_calls"] != 102
            or contract["proposed_replacement"]["maximum_new_scientific_optimizer_updates"] != 23040):
        raise ValueError("Explicit execution authority/budget differs")
    if plan["candidates"] != contract["architectures"]:
        raise ValueError("Architecture registry changed")
    if phase == "fitting" and (plan.get("maximum_optimizer_updates") != 23040
            or plan.get("fits") != [dict(candidate_id=row["candidate_id"], seed=seed)
                for row in contract["architectures"] for seed in (0, 1, 2)]):
        raise ValueError("Fresh fit inventory/workload changed")
    if authority.get("schema") in {"credo.all4_ddr.compiler_recovery_authority.v3", "credo.all4_ddr.adam_recovery_authority.v5", "credo.all4_ddr.geometry_recovery_authority.v6"} and (
            plan.get("compiler_dependency_surface_required") is not True
            or not plan.get("compiler_spec_sha256") or not plan.get("compiler_preflight_gate_path")):
        raise ValueError("Compiler prerequisite is not bound")
    if authority.get("schema") in {"credo.all4_ddr.adam_recovery_authority.v5", "credo.all4_ddr.geometry_recovery_authority.v6"} and (
            not isinstance(plan.get("compiler_preflight_import"), dict)
            or plan["compiler_preflight_import"].get("schema") != "credo.compiler_prerequisite_import.v1"):
        raise ValueError("Passed historical compiler prerequisite import is not bound")
    if authority.get("schema") in {"credo.all4_ddr.adam_recovery_authority.v5", "credo.all4_ddr.geometry_recovery_authority.v6"}:
        if plan.get("equal_input_Adam_qualification_required") is not True:
            raise ValueError("Equal-input Adam qualification must be required")
        if phase == "fitting":
            entry = plan.get("qualification_binding", {})
            if (not entry.get("output") or entry.get("worker_gate_path")
                    != str(Path(entry["output"]).parent / "QUALIFIED_FITTING_GATE.json")):
                raise ValueError("Worker-visible qualification gate is not bound")
    if authority.get("schema") == "credo.all4_ddr.geometry_recovery_authority.v6" and (
            plan.get("full_support_geometry_admission_required") is not True
            or contract["qualification"].get("full_support_challenge_required") is not True
            or authority.get("parent_scientific_updates_applied") != 18
            or authority.get("parent_scientific_updates_attempted") != 19):
        raise ValueError("Full-support recovery requirement or consumed-work identity differs")


def reuse_normalizers(contract, *, state_pairs):
    from architecture_fitting import validate_normalizers
    entry = contract["information_contract"]["normalizers"]
    scales = checked_json(Path(entry["path"]), entry["sha256"])
    validate_normalizers(scales)
    if (scales.get("state_laws") != state_pairs or scales.get("particles") != 512 or scales.get("seed") != 104729
            or scales.get("geometry_backend") != "geomloss_tensorized" or scales.get("model_forward_calls") != 0
            or scales.get("optimizer_updates") != 0):
        raise RuntimeError("Frozen fitting normalizer scope differs")
    return scales


def qualification_gate(plan):
    """Revalidate qualified worker completion and exact cleanup before fitting."""
    entry = plan["qualification_binding"]
    complete_raw = file_identity(Path(entry["output"]) / "COMPLETE.json")
    complete = checked_json(Path(entry["output"]) / "COMPLETE.json", complete_raw["sha256"])
    terminal_path, cleanup_path = Path(entry["control"]) / "terminal.json", Path(entry["control"]) / "cleanup.json"
    worker_gate = entry.get("worker_gate_path")
    if worker_gate:
        expected_path = Path(entry["output"]).parent / "QUALIFIED_FITTING_GATE.json"
        if worker_gate != str(expected_path):
            raise RuntimeError("Worker-visible qualification gate path differs")
        gate_reference = file_identity(expected_path)
        gate = checked_json(expected_path, gate_reference["sha256"])
        own_plan = file_identity(Path(plan["bundle_root"]) / "PLAN.json")
        if (gate.get("schema") != "credo.worker_visible_qualification_gate.v1"
                or gate.get("status") != "passed_parent_qualification"
                or gate.get("qualification_plan_sha256") != entry["plan_sha256"]
                or gate.get("successor_plan_sha256") != own_plan["sha256"]
                or gate.get("successor_run_id") != plan["run_id"]
                or gate.get("gpu_contract_sha256") != plan["gpu_contract_sha256"]
                or gate.get("no_new_numerical_work") is not True):
            raise RuntimeError("Worker-visible qualification gate identity differs")
        artifacts = gate.get("artifacts", {})
        expected_paths = dict(complete=Path(entry["output"]) / "COMPLETE.json",
            terminal=terminal_path, cleanup=cleanup_path)
        if set(artifacts) != set(expected_paths):
            raise RuntimeError("Qualification gate artifact inventory differs")
        records, references = {}, {}
        for name, path in expected_paths.items():
            artifact = artifacts[name]
            raw = artifact.get("raw_json", "").encode("utf-8")
            reference = dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
            if len(raw) > 16 * 1024**2 or artifact.get("path") != str(path) or artifact.get("reference") != reference:
                raise RuntimeError("Qualification gate embedded receipt identity differs")
            records[name], references[name] = json.loads(raw), reference
        if (references["complete"] != complete_raw or records["complete"] != complete
                or gate.get("parent_run_id") != complete.get("run_id")
                or not gate.get("parent_job_uid")
                or records["terminal"].get("uid") != gate["parent_job_uid"]
                or records["terminal"].get("plan_sha256") != entry["plan_sha256"]):
            raise RuntimeError("Qualification gate parent producer differs")
        terminal, cleanup = records["terminal"], records["cleanup"]
        qualification_refs = dict(complete=complete_raw, terminal=references["terminal"],
            cleanup=references["cleanup"], worker_gate=gate_reference)
    else:
        # Historical fixtures retain their original path; v5 requires the home gate.
        terminal = checked_json(terminal_path, file_identity(terminal_path)["sha256"])
        cleanup = checked_json(cleanup_path, file_identity(cleanup_path)["sha256"])
        qualification_refs = dict(complete=complete_raw, terminal=file_identity(terminal_path), cleanup=file_identity(cleanup_path))
    if (complete.get("status") != "qualified_gpu_native_full_ddr_execution"
            or complete.get("plan_sha256") != entry["plan_sha256"] or not complete.get("input_lease_exit_verified")
            or complete.get("gpu_contract_sha256") != plan["gpu_contract_sha256"]
            or complete.get("budget_used") != dict(full=90, partial=12, steps=18, refresh=3, chunks=549)
            or terminal.get("status") != "passed" or terminal.get("result_complete_sha256") != complete_raw["sha256"]
            or terminal.get("uid") != cleanup.get("uid") or cleanup.get("status") != "complete"
            or not isinstance(cleanup.get("reads"), list)
            or cleanup["reads"][-2:] != [dict(job_absent=True, owned_pods=0)]*2):
        raise RuntimeError("Qualification/input exit/cleanup gate unresolved")
    if complete.get("selected_profile") not in {"GPU_native_eager_foreach", "GPU_native_compiled_foreach"}:
        raise RuntimeError("Unqualified execution profile")
    if plan.get("full_support_geometry_admission_required"):
        from geometry_evidence import validate_geometry_qualification
        validate_geometry_qualification(complete)
    return complete, qualification_refs


def main(argv=None):
    root, plan, plan_sha = load_plan(argv, root=Path(__file__).absolute().parent)
    if root != Path(plan["bundle_root"]):
        raise RuntimeError("Immutable worker root differs")
    verify_bundle(root, plan)
    contract = checked_json(root / "GPU_CONTRACT.json", plan["gpu_contract_sha256"])
    authority = checked_json(root / "EXECUTION_AUTHORITY.json", plan["execution_authority_sha256"])
    validate_plan(plan, contract, authority)
    if (os.getuid(), os.getgid()) != (307469, 600815):
        raise RuntimeError("Institutional writer identity differs")
    output = Path(plan["output"])
    if os.path.lexists(output) or any(p.is_symlink() for p in (output, *output.parents)):
        raise RuntimeError("Absent immutable output required")
    output.mkdir(mode=0o700)
    started = time.perf_counter()
    fits, total_updates, total_attempts = [], 0, 0
    current_fit, current_updates, current_attempts = None, 0, 0
    budget = cache = None
    def progress(value):
        nonlocal current_updates, current_attempts
        current_updates = value.get("optimizer_updates", current_updates)
        current_attempts = value.get("attempted_optimizer_updates", current_attempts)
        rolled = bool(fits and current_fit == {key: fits[-1][key] for key in ("candidate_id", "seed")})
        value = dict(value, utc=datetime.now(timezone.utc).isoformat(), fit=current_fit,
            published_fit_optimizer_updates=total_updates, global_optimizer_updates=total_updates+(0 if rolled else current_updates))
        with (output / "PROGRESS.jsonl").open("ab") as stream:
            stream.write(canonical(value))
            stream.flush()
        print(canonical(value).decode(), flush=True)
    try:
        import pandas as pd
        import pyarrow as pa
        import torch
        from credo.io import RunConfig
        from credo.recipes.compact_sde_v3.recipe import CompactV3Config
        from credo.runtime import TrainingEngine
        from development_inputs import prepared_development
        from gpu_native import DeviceLawCache, GPUOptions, POLICY, SAMPLER, geomloss_backend, geometry_admission
        from gpu_native_fitting import GPUFittingRecipe
        from gpu_native_inputs import compile_fitting_qualification, prepared_fitting
        from gpu_native_qualification import WorkBudget, execution_options, qualify
        from native_study import compile_development

        if (platform.python_version() != "3.12.13" or torch.__version__ != "2.6.0+cu124"
                or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("Pinned same-image CUDA runtime required")
        prop = torch.cuda.get_device_properties(0)
        if "H100" not in prop.name or "MIG" in prop.name or prop.total_memory < 75_000_000_000 or torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError("Full H100 80GB required")
        # Resolve every geometry helper before startup release or cohort access.
        # This checks imports, public/scalar identity and signatures only: no
        # additional loss/backward call or unbudgeted optimizer operation.
        dependency_identity = geomloss_backend().identity
        compiler_identity = compiler_gate_reference = None
        if plan.get("compiler_dependency_surface_required"):
            from compiler_toolchain import prepare_compiler, verified_compiler_gate
            if plan.get("compiler_preflight_import"):
                from gpu_native_compiler_reuse import verified_imported_compiler_gate
                compiler_gate_reference = verified_imported_compiler_gate(plan)
            else:
                compiler_gate_reference = verified_compiler_gate(plan, current_plan_sha256=plan_sha)
            compiler_identity = prepare_compiler(root, plan)
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        pa.set_cpu_count(8)
        pa.set_io_thread_count(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        canary = output / "writer-canary"
        publish_json(canary, dict(uid=os.getuid()))
        canary.unlink()
        startup = startup_record(plan, plan_sha, prop, torch)
        publish_json(output / "STARTUP.json", startup)
        wait_for_release(output, startup)
        progress(dict(event="geomloss_dependency_surface_verified", **dependency_identity))
        if compiler_identity is not None:
            progress(dict(event="compiler_dependency_surface_verified", **{
                key: value for key, value in compiler_identity.items() if key != "status"}))
        fitting_contract = checked_json(root / "FITTING_CONTRACT.json", plan["fitting_contract_sha256"])
        metadata = checked_json(root / "METADATA_PREPARATION.json", plan["metadata_sha256"])
        inputs = checked_json(root / "DEVELOPMENT_INPUTS.json", plan["development_inputs_sha256"])
        checked_json(root / "AUTHORITY.json", plan["authority_sha256"])
        qualification, qualification_refs = (qualification_gate(plan) if plan["phase"] == "fitting" else (None, None))
        options = execution_options(qualification["selected_profile"]) if qualification else GPUOptions()
        loader = prepared_development if plan["phase"] == "fitting" else prepared_fitting
        progress(dict(event="verified_input_opening", phase=plan["phase"]))
        with loader(inputs, metadata, authority_sha256=plan["authority_sha256"], input_sha256=plan["development_inputs_sha256"],
                metadata_sha256=plan["metadata_sha256"]) as (bundle, _):
            def config_for(candidate, seed):
                return CompactV3Config.model_validate(recipe_config(fitting_contract, candidate, seed))
            cfg = config_for(plan["candidates"][0], 0)
            problem = (compile_development(bundle, cfg)[0] if plan["phase"] == "fitting" else compile_fitting_qualification(bundle, cfg))
            data = problem.training
            paired = tuple(key for key in data.measure_ids if key in data.measures["Stim48hr"])
            updates = 40*math.ceil(len(paired)/1086)+20*math.ceil(len(data.measure_ids)/1086)
            if (len(data.measure_ids), len(paired), len(data.embedding_ids), updates) != (46809, 45014, 12555, 2560):
                raise RuntimeError("Exact fitting support/exposure changed")
            scales = reuse_normalizers(contract, state_pairs=len(paired))
            publish_json(output / "NORMALIZERS.json", scales)
            compiled = dict(study_content_hash=problem.study_content_hash, problem_hash=problem.problem_hash,
                selection_hash=problem.selection_hash, fitting_series=len(data.measure_ids), fitting_state_pairs=len(paired),
                fitting_targets=len(data.embedding_ids), exact_optimizer_updates_per_fit=updates,
                exact_optimizer_updates_total=9*updates, guide_exposures_per_fit=40*len(paired)+20*len(data.measure_ids),
                D2_numerical_access=plan["phase"] == "fitting", D2_scores_computed=False, D1_access=False, LARRY_access=False)
            publish_json(output / "COMPILED_STUDY.json", compiled)
            cache = DeviceLawCache(data, device="cuda", options=options)
            if plan.get("full_support_geometry_admission_required"):
                admission = geometry_admission(cache, 512)
                from geometry_evidence import validate_admission
                validate_admission(admission)
                publish_json(output / "GEOMETRY_ADMISSION.json", admission)
                progress(dict(event="full_support_geometry_admission", **admission))
            progress(dict(event="device_laws_ready", cache_bytes=cache.allocated_bytes, **compiled))
            if plan["phase"] == "qualification":
                budget = WorkBudget()
                result = qualify(problem, plan["candidates"], config_for, fitting_contract, scales, cache,
                    progress=progress, budget=budget,
                    require_full_support_challenge=plan.get("full_support_geometry_admission_required", False))
                if plan.get("full_support_geometry_admission_required"):
                    result["geometry_admission"] = admission
                    result["full_support_challenge_qualified"] = True
            else:
                registry = {row["candidate_id"]: row for row in plan["candidates"]}
                for item in plan["fits"]:
                    current_fit, current_updates, current_attempts = item, 0, 0
                    run = output / "fits" / item["candidate_id"] / ("seed-"+str(item["seed"]))
                    config = RunConfig.model_validate(dict(study=output / "COMPILED_STUDY.json", output=run,
                        recipe_config=recipe_config(fitting_contract, registry[item["candidate_id"]], item["seed"]),
                        split=dict(strategy="subject", train_values=("D3", "D4"), validation_values=("D2",), representation_scope="shared")))
                    execution = dict(fitting_contract["memory"], maximum_optimizer_updates_per_fit=2560)
                    fitting_recipe = GPUFittingRecipe(scales, execution, options=options, cache=cache, progress=progress)
                    progress(dict(event="fit_started", **item, selected_profile=qualification["selected_profile"]))
                    trainer = TrainingEngine().fit(fitting_recipe, problem, config, device="cuda")
                    if (trainer.optimizer_updates != 2560 or trainer.attempted_optimizer_updates != 2560
                            or trainer.training_guide_exposures != compiled["guide_exposures_per_fit"]
                            or len(trainer.optimizer_memory_admissions) != 2 or not trainer.metrics.empty
                            or any(not math.isfinite(v) or v <= 0 for v in trainer.head_gradient_maxima.values())):
                        raise RuntimeError("Exact fitting exposure/gradient/memory requirements failed")
                    trainer.save()
                    state = torch.load(run / "state/checkpoint.pt", map_location="cpu", weights_only=True)
                    if state["completed_epochs"] != 60 or state["compiled_problem_hash"] != problem.problem_hash:
                        raise RuntimeError("Terminal checkpoint identity changed")
                    for name, tensor in trainer.model.state_dict().items():
                        if not torch.equal(state["model_state"][name], tensor.detach().cpu()):
                            raise RuntimeError("Terminal checkpoint readback differs")
                    for name in ("metrics", "predictions", "counterfactuals"):
                        if not pd.read_parquet(run / ("tables/"+name+".parquet")).empty:
                            raise RuntimeError("Unexpected outcome scoring")
                    record = dict(**item, status="terminal_fitting_complete_pending_input_lease_exit", optimizer_updates=2560,
                        attempted_optimizer_updates=2560, epochs_completed=60, guide_exposures=trainer.training_guide_exposures,
                        particles=512, steps=8, parameter_count=sum(p.numel() for p in trainer.model.parameters()),
                        checkpoint=file_identity(run / "state/checkpoint.pt", owner=307469),
                        checkpoint_path=(run / "state/checkpoint.pt").relative_to(output).as_posix(),
                        optimizer_memory_admissions=trainer.optimizer_memory_admissions, head_gradient_maxima=trainer.head_gradient_maxima,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        normalizers_sha256=file_identity(output / "NORMALIZERS.json")["sha256"],
                        fitting_contract_sha256=plan["fitting_contract_sha256"], execution_policy=POLICY, sampler=SAMPLER,
                        selected_profile=qualification["selected_profile"], qualification_receipts=qualification_refs,
                        empty_assessment_table_rows=dict(metrics=0, predictions=0, counterfactuals=0),
                        scientific_assessment="not_performed_fitting_only")
                    publish_json(run / "FIT_COMPLETE.json", record)
                    fits.append(record)
                    total_updates += 2560
                    total_attempts += 2560
                    progress(dict(event="fit_published", completed_fit_count=len(fits)))
                    trainer.close()
                    del trainer, state, fitting_recipe, config
                    gc.collect()
                    torch.cuda.empty_cache()
                    output_inventory(output, plan["maximum_output_bytes"])
                current_fit, current_updates, current_attempts = None, 0, 0
                result = dict(status="completed_gpu_native_full_ddr_fitting", fits=fits,
                    scientific_optimizer_updates=total_updates, attempted_optimizer_updates=total_attempts,
                    selected_profile=qualification["selected_profile"], qualification_receipts=qualification_refs,
                    scientific_training_complete=True, scientific_acceptance="not_assessed_fitting_only")
            cache.close()
            cache = None
        # Publication requires successful original-artifact hash checks at lease exit.
        verify_bundle(root, plan)
        files, size = output_inventory(output, plan["maximum_output_bytes"])
        result.update(run_id=plan["run_id"], plan_sha256=plan_sha, implementation_sha256=plan["implementation_sha256"],
            gpu_contract_sha256=plan["gpu_contract_sha256"], fitting_contract_sha256=plan["fitting_contract_sha256"],
            execution_authority_sha256=plan["execution_authority_sha256"], pod_uid=startup["pod_uid"],
            input_lease_exit_verified=True, D2_scores_computed=False, D1_endpoint_access=False, LARRY_access=False,
            query_deployment_performed=False, assessment_performed=False, scientific_promotion=False,
            files=files, output_bytes=size, elapsed_seconds=time.perf_counter()-started,
            completed_utc=datetime.now(timezone.utc).isoformat())
        if compiler_identity is not None:
            result.update(compiler_identity=compiler_identity, compiler_gate_reference=compiler_gate_reference,
                compiler_spec_sha256=plan["compiler_spec_sha256"])
        if size+len(canonical(result)) > plan["maximum_output_bytes"]:
            raise RuntimeError("Completion output cap exceeded")
        publish_json(output / "COMPLETE.json", result)
    except BaseException as error:
        publish_json(output / "FAILED.json", dict(status="failed", error=repr(error), run_id=plan["run_id"],
            current_fit=current_fit, completed_fits=len(fits), budget_used=None if budget is None else budget.used,
            **failure_accounting(fits, current_fit, total_updates, total_attempts, current_updates, current_attempts),
            automatic_retry=False, utc=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    main()
