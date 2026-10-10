"""Nine sequential fresh full-DDR fits; no query, decoder or outcome scoring."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import math
import os
from pathlib import Path
import stat
import time

from encode_sources import publish_json
from prepare_manifest import canonical
from prepare_worker import checked_json, file_identity, load_plan, startup_record, verify_bundle, wait_for_release


def validate_plan(plan, contract=None):
    expected = dict(schema="credo.all4_ddr.architecture_fitting_execution.v1",
        preparation_only=False, scientific_fitting_only=True, D1_endpoint_access=False,
        D2_scores_computed=False, LARRY_access=False, automatic_retries=0, gpu_ceiling=1,
        maximum_fits=9, maximum_optimizer_updates=23760,
        runtime_deadline_policy="no_runtime_deadline_user_20260919")
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Fitting scope/budget differs")
    candidates = ["ddr_mlp128_e8", "ddr_res256_e8", "ddr_res256_e16"]
    if [r["candidate_id"] for r in plan["candidates"]] != candidates:
        raise ValueError("Exactly the registered architectures required")
    if plan.get("fits") != [dict(candidate_id=cid, seed=seed) for cid in candidates for seed in (0, 1, 2)]:
        raise ValueError("Complete exact candidate/seed inventory required")
    if contract is not None:
        required = dict(schema="credo.all4_ddr.architecture_fitting.v1", maximum_fits=9,
            epochs=dict(state=40, mass=20, context=0), measures_per_batch=1086, particles=512,
            steps_per_interval=8, seeds=[0, 1, 2], checkpoint_selection="last", batching="random",
            maximum_optimizer_updates_per_fit=2640, maximum_optimizer_updates=23760,
            query_deployment=False, assessment=False, final_all_four_refit=False, LARRY_access=False)
        if any(contract.get(k) != v for k, v in required.items()):
            raise ValueError("Changed fitting intervention or workload")


def recipe_config(contract, candidate, seed):
    return dict(model=dict(candidate["model"], context="none", n_programs=8, growth_max=3.0),
        training=dict(epochs=contract["epochs"], particles=contract["particles"],
            steps_per_interval=contract["steps_per_interval"], measures_per_batch=contract["measures_per_batch"],
            batching=contract["batching"], learning_rate=contract["learning_rate"],
            checkpoint_selection="last", seed=seed, progress_interval=1,
            catalog_bank_momentum=contract["catalog_bank_momentum"],
            catalog_refresh_particles=contract["catalog_refresh_particles"]),
        evaluation=dict(particles=4096, measures_per_batch=1, steps_per_interval=8),
        validation=dict(fraction=0.0, representation_scope="shared"),
        loss=dict(mass=0.0, count=1.0, count_likelihood="categorical_ce", mass_increment="particle_log_mass",
            sinkhorn_epsilon=0.1, action_drift=contract["loss"]["action_drift"],
            action_diffusion=contract["loss"]["action_diffusion"], action_growth=contract["loss"]["action_growth"],
            parameter_regularization=contract["loss"]["parameter_regularization"]))


def output_inventory(output, maximum_bytes):
    rows = []
    for path in sorted(output.rglob("*")):
        info = path.lstat()
        if info.st_uid != 307469 or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise RuntimeError("Unexpected fitting output member")
        if stat.S_ISREG(info.st_mode):
            rows.append(dict(path=path.relative_to(output).as_posix(), **file_identity(path, owner=307469)))
    size = sum(row["size_bytes"] for row in rows)
    if size > maximum_bytes:
        raise RuntimeError("Fitting publication exceeds output cap")
    return rows, size


def failure_accounting(fits, current_fit, total_updates, total_attempts, current_updates, current_attempts):
    """A published current fit is already included in the cumulative totals."""
    rolled_up = bool(fits and current_fit == {k: fits[-1][k] for k in ("candidate_id", "seed")})
    pending_updates = 0 if rolled_up else current_updates
    pending_attempts = 0 if rolled_up else current_attempts
    return dict(current_fit_already_published=rolled_up,
                published_fit_optimizer_updates=total_updates,
                unpublished_current_fit_optimizer_updates=pending_updates,
                scientific_optimizer_updates=total_updates+pending_updates,
                published_fit_attempted_optimizer_updates=total_attempts,
                unpublished_current_fit_attempted_optimizer_updates=pending_attempts,
                attempted_optimizer_updates=total_attempts+pending_attempts)


def main(argv=None):
    root, plan, plan_sha = load_plan(argv, root=Path(__file__).absolute().parent)
    validate_plan(plan)
    if root != Path(plan["bundle_root"]):
        raise RuntimeError("Fitting bundle root differs")
    verify_bundle(root, plan)
    if (os.getuid(), os.getgid()) != (307469, 600815):
        raise RuntimeError("Institutional fitting writer differs")
    output = Path(plan["output"])
    if os.path.lexists(output) or any(p.is_symlink() for p in (output, *output.parents)):
        raise RuntimeError("New absent fitting output required")
    output.mkdir(mode=0o700)
    started = time.perf_counter()
    fits, total_updates, total_attempts = [], 0, 0
    current_fit, current_updates, current_attempts = None, 0, 0
    try:
        import pandas as pd
        import pyarrow as pa
        import torch
        from architecture_fitting import ArchitectureFittingRecipe, fitting_normalizers
        from credo.io import RunConfig
        from credo.recipes.compact_sde_v3.recipe import CompactV3Config
        from credo.runtime import TrainingEngine
        from development_inputs import prepared_development
        from native_study import compile_development

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("One actual H100 required; no CPU fallback")
        prop = torch.cuda.get_device_properties(0)
        if ("H100" not in prop.name or "MIG" in prop.name or prop.total_memory < 75_000_000_000
                or torch.cuda.get_device_capability(0) != (9, 0)):
            raise RuntimeError("Full H100 80GB required")
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
            raise RuntimeError("Writer canary owner differs")
        canary.unlink()
        startup = startup_record(plan, plan_sha, prop, torch)
        publish_json(output / "STARTUP.json", startup)
        wait_for_release(output, startup)
        contract = checked_json(root / "FITTING_CONTRACT.json", plan["fitting_contract_sha256"])
        validate_plan(plan, contract)
        capacity = checked_json(root / "FIT_CAPACITY_BINDING.json", plan["capacity_binding_sha256"])
        parent = checked_json(Path(capacity["home"]) / "output/COMPLETE.json", capacity["complete_sha256"])
        if parent["status"] != "qualified_full_ddr_capacity" or parent["scientific_optimizer_updates"] != 0:
            raise RuntimeError("Completed capacity prerequisite differs")
        metadata = checked_json(root / "METADATA_PREPARATION.json", plan["metadata_sha256"])
        inputs = checked_json(root / "DEVELOPMENT_INPUTS.json", plan["development_inputs_sha256"])
        checked_json(root / "AUTHORITY.json", plan["authority_sha256"])

        def progress(value):
            nonlocal current_updates, current_attempts
            current_updates = value.get("optimizer_updates", current_updates)
            current_attempts = value.get("attempted_optimizer_updates", current_attempts)
            rolled_up = bool(fits and current_fit == {k: fits[-1][k] for k in ("candidate_id", "seed")})
            value = dict(value, utc=datetime.now(timezone.utc).isoformat(), fit=current_fit,
                         published_fit_optimizer_updates=total_updates,
                         global_optimizer_updates=total_updates+(0 if rolled_up else current_updates))
            with (output / "PROGRESS.jsonl").open("ab") as handle:
                handle.write(canonical(value))
                handle.flush()
            print(canonical(value).decode(), flush=True)

        progress(dict(event="verified_fitting_input_opening"))
        with prepared_development(inputs, metadata, authority_sha256=plan["authority_sha256"],
                input_sha256=plan["development_inputs_sha256"], metadata_sha256=plan["metadata_sha256"]) as (bundle, _):
            config0 = recipe_config(contract, plan["candidates"][0], 0)
            problem, _ = compile_development(bundle, CompactV3Config.model_validate(config0))
            if (len(problem.training.measure_ids) != contract["expected_fitting_source_laws"]
                    or len(problem.training.embedding_ids) != contract["expected_fitting_target_identities"]):
                raise RuntimeError("Fitting support changed from completed capacity")
            paired = tuple(key for key in problem.training.measure_ids if key in problem.training.measures["Stim48hr"])
            exact_updates = 40*math.ceil(len(paired)/1086)+20*math.ceil(len(problem.training.measure_ids)/1086)
            if exact_updates > contract["maximum_optimizer_updates_per_fit"]:
                raise RuntimeError("Exact support exceeds frozen exposure budget")
            # This is a reference to an externally compiled native study, not a
            # duplicate packed RNA/latent export or a standalone native codec.
            compiled = dict(schema="credo.externally_compiled_native_study_reference.v1",
                study_content_hash=problem.study_content_hash, problem_hash=problem.problem_hash,
                selection_hash=problem.selection_hash, development_inputs_sha256=plan["development_inputs_sha256"],
                compiler_sha256=hashlib.sha256((root / "native_study.py").read_bytes()).hexdigest(),
                fitting_series=len(problem.training.measure_ids), fitting_state_pairs=len(paired),
                fitting_targets=len(problem.training.embedding_ids), selection_series=len(problem.validation.measure_ids),
                exact_optimizer_updates_per_fit=exact_updates, exact_optimizer_updates_total=9*exact_updates,
                guide_exposures_per_fit=40*len(paired)+20*len(problem.training.measure_ids),
                D2_scores_computed=False, D1_access=False, LARRY_access=False)
            publish_json(output / "COMPILED_STUDY.json", compiled)
            progress(dict(event="fitting_support_bound", **{k: compiled[k] for k in (
                "fitting_series", "fitting_state_pairs", "fitting_targets", "exact_optimizer_updates_per_fit")}))
            normalizers = fitting_normalizers(problem, particles=512, seed=contract["loss"]["normalizer_seed"],
                                             progress=progress)
            publish_json(output / "NORMALIZERS.json", normalizers)
            progress(dict(event="fitting_scales_frozen", state_normalizer=normalizers["state"],
                          count_normalizer=normalizers["count"]))
            candidates = {r["candidate_id"]: r for r in plan["candidates"]}
            for item in plan["fits"]:
                current_fit = item
                current_updates = current_attempts = 0
                candidate = candidates[item["candidate_id"]]
                run = output / "fits" / item["candidate_id"] / ("seed-" + str(item["seed"]))
                config = RunConfig.model_validate(dict(study=output / "COMPILED_STUDY.json", output=run,
                    recipe_config=recipe_config(contract, candidate, item["seed"]),
                    split=dict(strategy="subject", train_values=("D3", "D4"), validation_values=("D2",),
                               representation_scope="shared")))
                execution = dict(contract["memory"], maximum_optimizer_updates_per_fit=exact_updates)
                recipe = ArchitectureFittingRecipe(normalizers, execution, progress=progress)
                progress(dict(event="fit_started", candidate_id=item["candidate_id"], seed=item["seed"]))
                trainer = TrainingEngine().fit(recipe, problem, config, device="cuda")
                if (trainer.optimizer_updates != exact_updates or trainer.attempted_optimizer_updates != exact_updates
                        or trainer.training_guide_exposures != compiled["guide_exposures_per_fit"]
                        or len(trainer.optimizer_memory_admissions) != 2 or not trainer.metrics.empty
                        or any(not math.isfinite(v) or v <= 0 for v in trainer.head_gradient_maxima.values())):
                    raise RuntimeError("Fit did not complete exact exposure and optimizer admission")
                trainer.save()
                state = torch.load(run / "state/checkpoint.pt", map_location="cpu", weights_only=True)
                if state["completed_epochs"] != 60 or state["compiled_problem_hash"] != problem.problem_hash:
                    raise RuntimeError("Terminal checkpoint contract differs")
                for name, tensor in trainer.model.state_dict().items():
                    if not torch.equal(state["model_state"][name], tensor.detach().cpu()):
                        raise RuntimeError("Terminal checkpoint readback differs")
                for name in ("metrics", "predictions", "counterfactuals"):
                    if not pd.read_parquet(run / ("tables/" + name + ".parquet")).empty:
                        raise RuntimeError("Fitting publication contains outcome scores")
                parameters = sum(p.numel() for p in trainer.model.parameters())
                record = dict(**item, status="terminal_fitting_complete_pending_input_lease_exit",
                    optimizer_updates=exact_updates, attempted_optimizer_updates=exact_updates,
                    epochs_completed=60, guide_exposures=trainer.training_guide_exposures,
                    particles=512, steps=8, parameter_count=parameters,
                    checkpoint=file_identity(run / "state/checkpoint.pt", owner=307469),
                    checkpoint_path=(run / "state/checkpoint.pt").relative_to(output).as_posix(),
                    optimizer_memory_admissions=trainer.optimizer_memory_admissions,
                    head_gradient_maxima=trainer.head_gradient_maxima,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    normalizers_sha256=file_identity(output / "NORMALIZERS.json")["sha256"],
                    fitting_contract_sha256=plan["fitting_contract_sha256"],
                    empty_assessment_table_rows=dict(metrics=0, predictions=0, counterfactuals=0),
                    scientific_assessment="not_performed_fitting_only")
                publish_json(run / "FIT_COMPLETE.json", record)
                fits.append(record)
                total_updates += exact_updates
                total_attempts += exact_updates
                progress(dict(event="fit_published", optimizer_updates=current_updates,
                              completed_fit_count=len(fits), total_completed_optimizer_updates=total_updates))
                trainer.close()
                del trainer, state, recipe, config
                gc.collect()
                torch.cuda.empty_cache()
                output_inventory(output, plan["maximum_output_bytes"])
            current_fit, current_updates, current_attempts = None, 0, 0
        # All original frozen state/count identities are verified at lease EXIT.
        verify_bundle(root, plan)
        files, size = output_inventory(output, plan["maximum_output_bytes"])
        complete = dict(status="completed_full_ddr_architecture_fitting", run_id=plan["run_id"],
            plan_sha256=plan_sha, implementation_sha256=plan["implementation_sha256"], pod_uid=startup["pod_uid"],
            fitting_contract_sha256=plan["fitting_contract_sha256"], capacity_binding_sha256=plan["capacity_binding_sha256"],
            input_lease_exit_verified=True, scientific_training_complete=True,
            scientific_optimizer_updates=total_updates, attempted_optimizer_updates=total_attempts,
            scientific_promotion=False, query_deployment_performed=False, assessment_performed=False,
            scientific_acceptance="not_assessed_fitting_only", D1_endpoint_access=False,
            D2_scores_computed=False, LARRY_access=False, fits=fits, files=files, output_bytes=size,
            elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        if len(fits) != 9 or total_updates != compiled["exact_optimizer_updates_total"] or total_updates > 23760:
            raise RuntimeError("Architecture fitting inventory or update cap differs")
        if size + len(canonical(complete)) > plan["maximum_output_bytes"]:
            raise RuntimeError("Complete publication exceeds output cap")
        publish_json(output / "COMPLETE.json", complete)
    except BaseException as error:
        publish_json(output / "FAILED.json", dict(status="failed", error=repr(error), run_id=plan["run_id"],
            current_fit=current_fit, completed_fits=len(fits),
            **failure_accounting(fits, current_fit, total_updates, total_attempts, current_updates, current_attempts),
            automatic_retry=False, utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    main()
