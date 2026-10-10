"""Build a new D2 inference bundle from the exact completed v6 source bytes."""
from __future__ import annotations

import argparse
import json

import prepare_launch as shared
from d2_handoff import ROOT, PARENT_PLAN, build_handoff

RUN_ID = "credo-full-ddr-d2-forecast-20261010-v5"
SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/d2_forecast_v5"
HOME_RUN = "/rsrch8/home/bcb/yding4" + SUFFIX
CONTROL_ROOT = "/rsrch8/scratch/bcb/yding4" + SUFFIX
BASE = ROOT / "d2_forecast_v5"
NEW_FILES = ("d2_handoff.py", "frozen_ddr.py", "d2_forecast.py", "d2_forecast_reuse.py")
CONTROLLERS = ("prepare_launch.py", "prepare_operate.py", "qualify_launch.py", "qualify_operate.py",
               "d2_forecast_launch.py", "d2_forecast_operate.py")


def recovery_evidence():
    raw = shared.read_regular(ROOT / "d2_forecast_v3/RECOVERY_RECONCILIATION_20261010.json")
    value = json.loads(raw)
    if (value.get("status") != "failed_v3_reconciled_three_unsealed_banks_no_scores"
            or value.get("job") != "credo-full-ddr-d2-forecast-20261010-v3"
            or value.get("uid") != "59c65683-4390-4c7f-9929-7f109755b082"
            or value.get("plan_sha256") != "2cfbea9a18025757362342fb046ec127155507dd2f73c89f998cb24e2044d40a"
            or value.get("process", {}).get("status") not in
                {"absent", "exited_zombie", "bound_process_absent_pid_reused"}
            or value.get("reads") != [dict(job_absent=True, owned_pods=0)] * 2
            or value.get("failure", {}).get("written_banks") != 3
            or value.get("failure", {}).get("parameter_updates") != 0
            or value.get("failure", {}).get("D2_scores_computed") is not False):
        raise ValueError("Reconciled failed D2 attempt required before a manual replacement")
    return raw


def historical_payload():
    parent = ROOT / "fitting_gpu_native_v6"
    build = json.loads(shared.read_regular(parent / "BUILD.json"))
    if build["plan_sha256"] != PARENT_PLAN:
        raise ValueError("Exact completed fitting producer required")
    payload = {}
    for name, entry in build["files"].items():
        raw = shared.read_regular(parent / "bundle" / name)
        if entry != dict(sha256=shared.sha(raw), size_bytes=len(raw)):
            raise ValueError("Frozen fitting source changed: " + name)
        if name not in {"INVENTORY.json", "PHASE_CHAIN.json"}:
            payload[name] = raw
    if shared.sha(payload["PLAN.json"]) != PARENT_PLAN:
        raise ValueError("Fitting plan changed")
    # No moving source checkout enters the new numerical model.
    return payload


def assemble(*args, **kwargs):
    payload = historical_payload()
    old_plan = json.loads(payload["PLAN.json"])
    for name in NEW_FILES:
        payload[name] = shared.read_regular(ROOT / name)
    for name in ("full_forecast.py", "full_evaluation.py"):
        # Forecast adapters were not part of fitting; bind their new identity.
        payload[name] = shared.read_regular(ROOT / name)
    payload["worker.py"] = shared.read_regular(ROOT / "d2_forecast_worker.py")
    payload["control.py"] = shared.read_regular(ROOT / "d2_forecast_control.py")
    for name in CONTROLLERS:
        payload["controller/" + name] = shared.read_regular(ROOT / name)
    for path in ("d2_forecast_worker.py", "d2_forecast_control.py"):
        payload["controller/" + path] = shared.read_regular(ROOT / path)
    handoff = build_handoff(
        json.loads(shared.read_regular(ROOT / "fitting_gpu_native_v6/STATUS_20261010T174515Z.json")),
        json.loads(shared.read_regular(ROOT / "fitting_gpu_native_v6/D2_HANDOFF_METADATA_20261010.json")),
        json.loads(payload["METADATA_PREPARATION.json"]), json.loads(payload["SELECTION_CONTRACT.json"]))
    handoff.update(status="bound_for_forecast_only_assessment_interfaces_pending")
    handoff["proposed_resources"]["approval"] = "user_approved_400_GiB_D2_prediction_and_selection"
    handoff["remaining"] = [v for v in handoff["remaining"] if v != "400_GiB_output_authority_and_fresh_capacity"]
    authority = dict(schema="credo.full_ddr.d2_execution_authority.v1",
        status="user_approved_400_GiB_D2_prediction_and_selection", user_response="Approve 400 GiB for D2",
        question_item_id='["request_user_input_async","call_5182968ebe3e473f9007e8951ad2732b",0]',
        gpus=1, gpu="H100_80GB", cpus=24, host_memory_GiB=200, threads=8,
        maximum_output_bytes=400*1024**3, training_updates=0, D1_access=False,
        LARRY_access=False, automatic_retries=0, elapsed_time_deadline=None,
        current_worker="source_only_prediction_and_seal", automatic_successor=False,
        assessment="separate_post_seal_phase_within_combined_400_GiB_cap",
        manual_relaunch_user_instruction="fix all and relaunch", manual_replacement_attempt=5,
        repair_scope="residual canonical single-guide arithmetic; all-model prequalification; read-only reuse of three MLP banks",
        unchanged="trained models, particles, source/noise streams, FP32 precision, steps and agreement tolerances")
    payload["D2_HANDOFF.json"] = shared.canonical(handoff)
    payload["D2_AUTHORITY.json"] = shared.canonical(authority)
    # V1 never staged or submitted: preserve its failed full-cap reservation.
    # The user approved a ceiling, not a requirement to reserve unused assessment
    # space during source-only inference. No source/precision/workload changes.
    failed = shared.read_regular(ROOT / "d2_forecast_v1/PREFLIGHT_FAILURE.json")
    failure = json.loads(failed)
    if (failure.get("status") != "blocked_before_staging"
            or failure.get("job_created") is not False or failure.get("remote_staging_performed") is not False):
        raise ValueError("Exact unlaunched capacity-preflight history is required")
    payload["PREVIOUS_PREFLIGHT_FAILURE.json"] = failed
    payload["FAILED_V3_RECONCILIATION.json"] = recovery_evidence()
    local_failure = shared.read_regular(ROOT / "d2_forecast_v4/PREFLIGHT_FAILURE.json")
    if json.loads(local_failure).get("remote_submitted") is not False:
        raise ValueError("Unlaunched v4 admission failure must remain distinct from a job retry")
    payload["UNLAUNCHED_V4_FAILURE.json"] = local_failure
    from d2_forecast_reuse import build_reuse_manifest
    reuse_metadata = json.loads(shared.read_regular(ROOT / "d2_forecast_v3/REUSE_METADATA_20261010.json"))
    for name, descriptor in reuse_metadata["producer_files"].items():
        raw = shared.read_regular(ROOT / "d2_forecast_v3/bundle" / name)
        if (shared.sha(raw), len(raw)) != (descriptor["sha256"], descriptor["size_bytes"]):
            raise ValueError("Retained failed producer differs: " + name)
    reuse = build_reuse_manifest(json.loads(payload["FAILED_V3_RECONCILIATION.json"]), reuse_metadata)
    payload["EXTERNAL_REUSE.json"] = shared.canonical(reuse)
    payload["publication.py"] = shared.publication_wrapper()
    implementation = {name: shared.sha(raw) for name, raw in sorted(payload.items())
                      if name.endswith(".py") and name not in {"control.py", "publication.py"}}
    payload["IMPLEMENTATION.json"] = shared.canonical(implementation)
    keep = ("image", "runtime_deadline_policy", "static_reuse_sha256", "static_reuse_validator_sha256",
        "lifecycle_core_sha256", "legacy_source_binding_sha256", "legacy_source_files", "legacy_source_root",
        "metadata_sha256", "development_inputs_sha256", "authority_sha256", "model_sha256",
        "fitted_complete_sha256", "fitted_root", "representation_identity")
    plan = {k: old_plan[k] for k in keep}
    plan.update(schema="credo.full_ddr.d2_forecast_execution.v1", schema_id="credo.full_ddr.d2_forecast.v1",
        phase="d2_forecast", run_id=RUN_ID, bundle_root=HOME_RUN+"/bundle", control_root=CONTROL_ROOT,
        worker_output=HOME_RUN+"/output", output=HOME_RUN+"/output", gpu_ceiling=1,
        authorized_CPUs=24, authorized_memory_GiB=200, threads=8, torch_threads=8, automatic_retries=0,
        parameter_updates=0, maximum_optimizer_updates=0, D1_access=False, D1_endpoint_access=False,
        D2_scores_computed=False, LARRY_access=False, scientific_promotion=False, assessment_performed=False,
        maximum_new_output_bytes=240*1024**3, maximum_output_bytes=360*1024**3,
        per_bank_output_bytes=40*1024**3, combined_phase_output_ceiling_bytes=400*1024**3,
        previous_unlaunched_preflight_sha256=shared.sha(failed),
        failed_v3_reconciliation_sha256=shared.sha(payload["FAILED_V3_RECONCILIATION.json"]),
        external_reuse_manifest_sha256=shared.sha(payload["EXTERNAL_REUSE.json"]),
        retained_prior_output_bytes=reuse_metadata["retained_output_bytes"],
        new_bank_count=6, reused_bank_count=3, all_model_qualification_before_bulk=True,
        production_batch_guides_by_architecture={"mlp": 16, "residual_mlp": 1},
        legacy_diagnostic={"candidate_id": "ddr_res256_e8", "seed": 0, "batch_guides": 16},
        later_assessment_requires_fresh_remaining_quota_and_combined_cap_check=True,
        maximum_banks=9, sources=["D2_Stim8hr"], prefixes=[4096, 8192], steps_per_interval=8,
        batch_guides=16, scalar_comparison_guides_per_checkpoint=16,
        qualification_atol=2e-5, qualification_rtol=1e-5,
        qualification_scope="fixed_source_only_scalar_vs_architecture_bound_production_not_endpoint_precision",
        precision="strict_FP32_no_TF32_no_AMP", compilation=False,
        handoff_sha256=shared.sha(payload["D2_HANDOFF.json"]),
        d2_authority_sha256=shared.sha(payload["D2_AUTHORITY.json"]),
        implementation_sha256=shared.sha(payload["IMPLEMENTATION.json"]),
        worker_sha256=shared.sha(payload["worker.py"]), control_sha256=shared.sha(payload["control.py"]),
        startup_gate="inline_h100_controller_release_v1",
        startup_gate_authority="User approved D2 frozen-model forecasts: nine banks, no training or outcomes",
        parent_run_id=handoff["parent"]["run_id"], parent_completion_sha256=handoff["parent"]["completion_sha256"])
    payload["PLAN.json"] = shared.canonical(plan)
    payload["gpu-job.json"] = shared.canonical(shared.render_manifest(json.loads(payload["gpu-job.json"]), plan))
    return payload


def bound_shared():
    for name, value in dict(RUN_ID=RUN_ID, HOME_RUN=HOME_RUN, CONTROL_ROOT=CONTROL_ROOT).items():
        setattr(shared, name, value)


def build():
    bound_shared()
    original = shared.assemble
    shared.assemble = assemble
    try:
        return shared.build(BASE)
    finally:
        shared.assemble = original


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = build()
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
