"""One inference-only worker: all nine terminal forecasts, then a common seal.

This worker cannot assess outcomes or launch a successor. D2 scoring is a
separate phase requiring this completion and exact lifecycle reconciliation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
import stat
import time

from encode_sources import publish_json
from prepare_manifest import canonical
from prepare_worker import checked_json, file_identity, load_plan, startup_record, verify_bundle, wait_for_release


def validate_plan(plan, handoff, authority):
    expected = dict(phase="d2_forecast", gpu_ceiling=1, authorized_CPUs=24,
        authorized_memory_GiB=200, threads=8, parameter_updates=0,
        D1_access=False, D1_endpoint_access=False, D2_scores_computed=False,
        LARRY_access=False, automatic_retries=0, maximum_output_bytes=360*1024**3,
        maximum_new_output_bytes=240*1024**3, combined_phase_output_ceiling_bytes=400*1024**3,
        new_bank_count=6, reused_bank_count=3, all_model_qualification_before_bulk=True,
        later_assessment_requires_fresh_remaining_quota_and_combined_cap_check=True,
        runtime_deadline_policy="no_runtime_deadline_user_20260919")
    if any(plan.get(k) != v or type(plan.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Approved D2 inference resources or role boundary differ")
    if (authority.get("status") != "user_approved_400_GiB_D2_prediction_and_selection"
            or authority.get("maximum_output_bytes") != 400*1024**3
            or authority.get("training_updates") != 0 or authority.get("D1_access") is not False
            or authority.get("automatic_retries") != 0 or authority.get("elapsed_time_deadline") is not None):
        raise ValueError("Explicit D2 phase authority is missing")
    if (handoff.get("query_modeled_guides") != 23421 or handoff.get("query_controls") != 915
            or handoff.get("complete_catalog_size") != 26504 or len(handoff.get("checkpoints", [])) != 9
            or plan.get("sources") != ["D2_Stim8hr"] or plan.get("prefixes") != [4096, 8192]
            or plan.get("steps_per_interval") != 8 or plan.get("batch_guides") != 16):
        raise ValueError("Frozen forecast inventory or numerical workload differs")
    if (plan.get("per_bank_output_bytes") != 40*1024**3
            or 9*plan["per_bank_output_bytes"] > plan["maximum_output_bytes"]):
        raise ValueError("Per-bank output allowance exceeds the phase budget")
    if (plan.get("production_batch_guides_by_architecture") != {"mlp": 16, "residual_mlp": 1}
            or plan.get("legacy_diagnostic") != dict(candidate_id="ddr_res256_e8", seed=0, batch_guides=16)
            or plan.get("qualification_atol") != 2e-5 or plan.get("qualification_rtol") != 1e-5
            or not isinstance(plan.get("external_reuse_manifest_sha256"), str)
            or len(plan["external_reuse_manifest_sha256"]) != 64):
        raise ValueError("Prebound recovery policy or unchanged precision requirement differs")


def label_for(binding):
    return binding["candidate_id"] + "--seed-" + str(binding["seed"])


def publish_qualification(output, label, run):
    """Publish disagreement diagnostics before propagating a failed gate."""
    try:
        result = run()
    except BaseException as error:
        report = getattr(error, "report", None)
        if report is not None:
            publish_json(output / (label + "--qualification.json"), report)
        raise
    publish_json(output / (label + "--qualification.json"), result)
    if result.get("status") != "passed":
        raise RuntimeError("An unresolved numerical qualification cannot authorize publication")
    return result


def check_combined_output(new_bytes, retained_bytes, plan, completion_bytes=0):
    if (new_bytes + completion_bytes > plan["maximum_new_output_bytes"]
            or new_bytes + completion_bytes + retained_bytes > plan["maximum_output_bytes"]
            or new_bytes + completion_bytes + retained_bytes > plan["combined_phase_output_ceiling_bytes"]):
        raise RuntimeError("Retained plus new forecast outputs exceed the bound budget")


def seal_row(row):
    keys = ("candidate_id", "seed", "storage", "path", "seal", "bank_identity")
    result = {key: row[key] for key in keys}
    if row["storage"] == "external_reused":
        result.update(reuse_seal_path=row["reuse_seal_path"], array_identity=row["array_identity"])
    return result


def output_inventory(output, cap, known_arrays):
    """Reuse freshly sealed immutable-array hashes, not a fourth worker reread.

    Array hashes have just been computed twice by write_bank/seal_bank. This
    requires the exact post-seal file signature; the controller independently
    hashes every array again before calling the whole worker successful.
    """
    rows, size = [], 0
    for path in sorted(output.rglob("*")):
        info = path.lstat()
        if info.st_uid != 307469 or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise RuntimeError("Unsafe output member")
        if stat.S_ISDIR(info.st_mode):
            continue
        relative = path.relative_to(output).as_posix()
        if relative == "COMPLETE.json":
            raise RuntimeError("Completion already exists")
        if relative in known_arrays:
            identity, signature = known_arrays[relative]
            if file_signature(path) != signature:
                raise RuntimeError("Sealed array changed before final publication")
        else:
            identity = file_identity(path, owner=307469)
        size += identity["size_bytes"]
        if size+1024**2 > cap:
            raise RuntimeError("Approved combined output cap exceeded")
        rows.append(dict(path=relative, **identity))
    return rows, size


def file_signature(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_uid)


def main():
    root, plan, plan_sha = load_plan(root=Path(__file__).absolute().parent)
    if root != Path(plan["bundle_root"]) or (os.getuid(), os.getgid()) != (307469, 600815):
        raise RuntimeError("Exact worker location and owner required")
    verify_bundle(root, plan)
    handoff = checked_json(root / "D2_HANDOFF.json", plan["handoff_sha256"])
    authority = checked_json(root / "D2_AUTHORITY.json", plan["d2_authority_sha256"])
    validate_plan(plan, handoff, authority)
    output = Path(plan["output"])
    if os.path.lexists(output) or any(p.is_symlink() for p in output.parents):
        raise RuntimeError("Fresh immutable output required; no automatic retry")
    output.mkdir(mode=0o700)
    banks, active = [], None
    started = time.perf_counter()
    def progress(value):
        value = dict(value, utc=datetime.now(timezone.utc).isoformat(), model=active,
                     completed_banks=len(banks), parameter_updates=0, D2_scores_computed=False)
        with (output / "PROGRESS.jsonl").open("ab") as stream:
            stream.write(canonical(value))
            stream.flush()
        print(canonical(value).decode(), flush=True)
    try:
        import pyarrow as pa
        import torch
        from d2_forecast import d2_source_lease, iter_batched_forecasts, qualify_forecast
        from d2_forecast_reuse import validate_reuse, publish_reuse_seals
        from frozen_ddr import load_frozen_model
        from full_forecast import seal_bank, write_bank
        if (platform.python_version() != "3.12.13" or torch.__version__ != "2.6.0+cu124"
                or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("Pinned same-image single-CUDA runtime required")
        prop = torch.cuda.get_device_properties(0)
        if "H100" not in prop.name or "MIG" in prop.name or prop.total_memory < 75_000_000_000 or torch.cuda.get_device_capability(0) != (9, 0):
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
        publish_json(canary, dict(uid=os.getuid()))
        canary.unlink()
        startup = startup_record(plan, plan_sha, prop, torch)
        publish_json(output / "STARTUP.json", startup)
        wait_for_release(output, startup)
        # Read the complete parent once, never Trainer.load or endpoint arrays.
        parent = handoff["parent"]
        complete = checked_json(Path(parent["output_root"]) / "COMPLETE.json", parent["completion_sha256"])
        if (complete.get("run_id") != parent["run_id"] or complete.get("plan_sha256") != parent["plan_sha256"]
                or complete.get("scientific_training_complete") is not True
                or complete.get("scientific_optimizer_updates") != 23040
                or complete.get("assessment_performed") is not False):
            raise RuntimeError("Completed frozen parent lineage differs")
        metadata = checked_json(root / "METADATA_PREPARATION.json", plan["metadata_sha256"])
        inputs = checked_json(root / "DEVELOPMENT_INPUTS.json", plan["development_inputs_sha256"])
        reuse_manifest = checked_json(root / "EXTERNAL_REUSE.json", plan["external_reuse_manifest_sha256"])
        progress(dict(event="opening_D2_source_only"))
        (output / "banks").mkdir()
        with d2_source_lease(inputs, metadata, handoff) as sources:
            if len(sources.measure_ids) != 23421 or len(sources.catalog.guides) != 26504:
                raise RuntimeError("Deployment source support differs")
            # Immutable empirical reference by exact source artifact identity.
            # No forecast noise or query endpoints are needed to define it.
            publish_json(output / "REFERENCES.json", dict(schema="credo.full_ddr.d2_source_references.v1",
                status="frozen_source_only_references", **sources.catalog.identity,
                source_counts=list(sources.catalog.counts), modeled_measure_ids=list(sources.measure_ids),
                modeled_guide_indices=list(sources.guide_indices),
                source_frequency=list(sources.catalog.source_frequency),
                state_persistence="entire_stored_empirical_guide_law_uniform_positive_RNA_cells",
                raw_abundance_persistence="raw_source_counts_divided_by_complete_captured_total",
                regularized_abundance_persistence="(source_count+0.5)/(complete_source_total+0.5*26504)",
                endpoint_observations_read=False, parameter_updates=0))
            # Reuse is an explicit new qualification, not retroactive success of
            # the failed producer. Its arrays remain in their original directory.
            reuse = validate_reuse(reuse_manifest, source_identity=sources.catalog.identity,
                measure_ids=sources.measure_ids, guide_indices=sources.guide_indices,
                checkpoints=handoff["checkpoints"], owner=307469,
                maximum_new_bytes=plan["maximum_new_output_bytes"])
            retained_bytes = reuse["retained_bytes"]
            reused_pairs = {(row["candidate_id"], row["seed"]) for row in reuse["banks"]}
            if len(reused_pairs) != 3:
                raise RuntimeError("Exactly three preserved MLP banks must be requalified")
            check_combined_output(0, retained_bytes, plan)
            qualifications = {}
            # No new HDF5 bank can be opened until every terminal model passes
            # its predeclared numerical path. A legacy diagnostic never qualifies.
            for binding in handoff["checkpoints"]:
                active = {k: binding[k] for k in ("candidate_id", "seed")}
                progress(dict(event="frozen_model_loading"))
                model, loader = load_frozen_model(binding["checkpoint_path"], binding, handoff)
                label = label_for(binding)
                if active == dict(candidate_id="ddr_res256_e8", seed=0):
                    diagnostic = qualify_forecast(model, sources, seed=0, device="cuda", diagnostic_legacy16=True)
                    if diagnostic.get("status") != "diagnostic_only" or diagnostic.get("production_qualified") is not False:
                        raise RuntimeError("Legacy arithmetic diagnostic cannot authorize production")
                    publish_json(output / (label + "--legacy16-diagnostic.json"), diagnostic)
                    progress(dict(event="legacy16_diagnostic_complete", agreement_passed=diagnostic["agreement_passed"]))
                qualification = publish_qualification(output, label,
                    lambda: qualify_forecast(model, sources, seed=binding["seed"], device="cuda"))
                publish_json(output / (label + "--loader.json"), loader)
                if qualification.get("production_qualified") is not True:
                    raise RuntimeError("Production qualification is missing")
                qualifications[label] = dict(qualification=qualification, loader=loader)
                progress(dict(event="source_only_forecast_qualified", status=qualification["status"],
                    production_qualified=True, numerical_policy=qualification["numerical_policy"],
                    maximum_absolute_difference=qualification["maximum_absolute_difference"]))
                del model
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.max_memory_reserved() > 78_000_000_000:
                    raise RuntimeError("CUDA reserved-memory ceiling exceeded")
            if len(qualifications) != 9:
                raise RuntimeError("All nine numerical qualifications must precede bulk publication")
            publish_json(output / "ALL_MODELS_QUALIFIED.json", dict(
                status="all_nine_current_production_paths_qualified_before_new_bulk_arrays",
                checkpoints=[dict(candidate_id=row["candidate_id"], seed=row["seed"],
                    qualification=file_identity(output / (label_for(row)+"--qualification.json")))
                    for row in handoff["checkpoints"]], parameter_updates=0, endpoint_observations_read=False))
            progress(dict(event="all_nine_current_paths_qualified_before_bulk"))
            for binding in handoff["checkpoints"]:
                active = {k: binding[k] for k in ("candidate_id", "seed")}
                label = label_for(binding)
                evidence = qualifications[label]
                if (binding["candidate_id"], binding["seed"]) in reused_pairs:
                    row = next(row for row in reuse["banks"]
                        if (row["candidate_id"], row["seed"]) == (binding["candidate_id"], binding["seed"]))
                    banks.append(dict(row, **evidence, storage="external_reused"))
                    progress(dict(event="preserved_bank_requalified_not_rewritten"))
                    continue
                model, loader = load_frozen_model(binding["checkpoint_path"], binding, handoff)
                if loader != evidence["loader"]:
                    raise RuntimeError("Frozen loader identity changed between qualification and prediction")
                destination = output / "banks" / label
                bank = write_bank(destination, sources,
                    iter_batched_forecasts(model, sources, prefixes=(4096, 8192), steps_per_interval=8,
                        seed=binding["seed"], batch_guides=16, progress=progress),
                    prefixes=(4096, 8192), checkpoint_sha256=binding["sha256"],
                    implementation_sha256=plan["implementation_sha256"], seed=binding["seed"],
                    steps_per_interval=8, maximum_output_bytes=plan["per_bank_output_bytes"])
                banks.append(dict(active, path="banks/" + label, storage="new",
                                  checkpoint_sha256=binding["sha256"], bank=bank, **evidence))
                del model
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.max_memory_reserved() > 78_000_000_000:
                    raise RuntimeError("CUDA reserved-memory ceiling exceeded")
                progress(dict(event="forecast_bank_written_not_sealed"))
        # Leaving the lease verifies all exact source paths after predictions.
        progress(dict(event="source_lease_exit_verified", endpoint_observations_read=False))
        known_arrays = {}
        for row, binding in zip(banks, handoff["checkpoints"], strict=True):
            expected = {k: binding[k] for k in ("sha256", "size_bytes")}
            if file_identity(Path(binding["checkpoint_path"]), owner=307469) != expected:
                raise RuntimeError("Checkpoint changed after prediction")
        # These seals are new local metadata referring to unchanged external
        # arrays, and explicitly record the fresh lease/checkpoint boundary.
        reused_seals = publish_reuse_seals(output, reuse_manifest, reuse, owner=307469,
                                          fresh_lease_exit_verified=True)
        for row in banks:
            if row["storage"] == "external_reused":
                sealed = next(value for value in reused_seals if
                    (value["candidate_id"], value["seed"]) == (row["candidate_id"], row["seed"]))
                row.update(reuse_seal_path=sealed["reuse_seal_path"], reuse_seal=sealed["reuse_seal"],
                           seal=sealed["reuse_seal"])
                row.pop("bank", None)
                continue
            destination = output / row["path"]
            bank = row.pop("bank")
            check_path = output / (destination.name + "--input-check.json")
            publish_json(check_path, dict(schema="credo.all4_full_ddr.forecast_input_check.v1",
                bank_sha256=file_identity(destination / "BANK.json")["sha256"],
                inputs_verified_after_prediction=True, parameters_unchanged=True, parameter_updates=0,
                endpoint_observations_read=False, **{k: bank[k] for k in
                    ("source_artifact_sha256", "support_sha256", "representation_sha256", "checkpoint_sha256")}))
            seal_bank(destination, input_check_receipt=check_path,
                      input_check_sha256=file_identity(check_path)["sha256"])
            array = destination / "forecast.h5"
            reference = bank["artifacts"]["forecast.h5"]
            known_arrays[row["path"] + "/forecast.h5"] = (
                dict(sha256=reference["sha256"], size_bytes=reference["bytes"]), file_signature(array))
            row["seal"] = file_identity(destination / "COMPLETE.json")
            row["bank_identity"] = file_identity(destination / "BANK.json")
        publish_json(output / "PREDICTION_SEAL.json", dict(schema="credo.full_ddr.d2_all_predictions_seal.v1",
            status="all_nine_forecasts_and_source_references_sealed", handoff_sha256=plan["handoff_sha256"],
            banks=[seal_row(row) for row in banks],
            references=file_identity(output / "REFERENCES.json"), input_lease_exit_verified=True,
            parameter_updates=0, endpoint_observations_read=False, D1_access=False, LARRY_access=False))
        progress(dict(event="all_nine_forecasts_sealed_assessment_not_started"))
        files, size = output_inventory(output, plan["maximum_new_output_bytes"], known_arrays)
        check_combined_output(size, retained_bytes, plan)
        result = dict(schema="credo.full_ddr.d2_forecast_completion.v1", status="completed_D2_frozen_forecasts",
            run_id=plan["run_id"], plan_sha256=plan_sha, pod_uid=startup["pod_uid"],
            handoff_sha256=plan["handoff_sha256"], implementation_sha256=plan["implementation_sha256"],
            parent_completion_sha256=parent["completion_sha256"], banks=banks,
            new_bank_count=6, reused_bank_count=3, retained_output_bytes=retained_bytes,
            combined_forecast_output_bytes=size+retained_bytes,
            external_reuse_manifest_sha256=plan["external_reuse_manifest_sha256"],
            all_model_qualification_before_bulk=True,
            input_lease_exit_verified=True, predictions_sealed=True, parameter_updates=0,
            query_deployment_performed=True, assessment_performed=False, D2_scores_computed=False,
            D1_access=False, D1_endpoint_access=False, LARRY_access=False,
            scientific_acceptance="not_assessed_forecasts_only", scientific_promotion=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            files=files, output_bytes=size, elapsed_seconds=time.perf_counter()-started,
            completed_utc=datetime.now(timezone.utc).isoformat())
        check_combined_output(size, retained_bytes, plan, len(canonical(result)))
        publish_json(output / "COMPLETE.json", result)
        print(canonical({k: result[k] for k in ("status", "output_bytes", "completed_utc")}).decode(), flush=True)
    except BaseException as error:
        if not (output / "FAILURE.json").exists():
            publish_json(output / "FAILURE.json", dict(status="failed_no_automatic_retry", error=repr(error),
                model=active, written_banks=len(banks), parameter_updates=0, D2_scores_computed=False,
                newly_written_banks=sum(row.get("storage") == "new" for row in banks),
                external_banks_requalified=sum(row.get("storage") == "external_reused" for row in banks),
                D1_access=False, LARRY_access=False, utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    main()
