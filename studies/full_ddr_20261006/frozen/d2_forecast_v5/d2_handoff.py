"""Metadata-only handoff from completed v6 fitting to prospective D2 deployment.

No latent arrays, counts, checkpoints, outcomes or scheduler objects are opened.
An output estimate and proposed resource limit are not a launch authorization.
The frozen selection rule and completed fitting artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from encode_sources import publish_json
from prepare_manifest import canonical

ROOT = Path(__file__).resolve().parent
PARENT_JOB = "credo-full-ddr-fitting-gpu-native-20261009-v6"
PARENT_UID = "f7697e94-39b3-4392-ad76-41ab7a0a5672"
PARENT_PLAN = "5e739832d56d7bd5aeda47070d0c80ccd1af55ce9a98939762e5c86eb557461e"
PARENT_COMPLETE = "7341090120ff2df85e525591c9f38e7ac0343cb9e756ad02c6a81dde0498ca26"
PARENT_OUTPUT = ("/rsrch8/home/bcb/yding4/perturbseq/"
                 "credo_full_ddr_gse314342_20261006_v1/fitting_gpu_native_v6/output")
CANDIDATES = ("ddr_mlp128_e8", "ddr_res256_e8", "ddr_res256_e16")
EXPECTED_FITS = tuple((candidate, seed) for candidate in CANDIDATES for seed in (0, 1, 2))
ARCHITECTURES = {
    "ddr_mlp128_e8": dict(hidden_dim=128, embedding_dim=8, trunk_kind="mlp", trunk_depth=2),
    "ddr_res256_e8": dict(hidden_dim=256, embedding_dim=8, trunk_kind="residual_mlp", trunk_depth=3),
    "ddr_res256_e16": dict(hidden_dim=256, embedding_dim=16, trunk_kind="residual_mlp", trunk_depth=3),
}


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("A lowercase SHA256 identity is required")
    return value


def exact_ids(values, donors):
    if not isinstance(values, list) or not values or len(values) != len(set(values)):
        raise ValueError("A complete unique ordered population inventory is required")
    for value in values:
        parts = value.split("::guide-index-") if isinstance(value, str) else ()
        if len(parts) != 2 or parts[0] not in donors or not parts[1].isdigit():
            raise ValueError("Population inventory crosses its donor/guide role")
        index = int(parts[1])
        if not 0 <= index < 26504 or str(index) != parts[1]:
            raise ValueError("Noncanonical guide index")
    return values


def forecast_bytes(laws, *, prefixes=(4096, 8192), dimension=48, catalog_size=26504):
    """Uncompressed writer admission, including both prefixes, not just raw states."""
    if type(laws) is not int or not 1 <= laws <= catalog_size:
        raise ValueError("Invalid modeled-guide count")
    if prefixes != (4096, 8192) or dimension != 48 or catalog_size != 26504:
        raise ValueError("Frozen particle, state and catalog axes cannot be reduced")
    return (4 * 1024**2 + laws * (prefixes[-1] * (4 * dimension + 8)
            + len(prefixes) * (8 * (dimension + 4)))
            + catalog_size * (len(prefixes) * 16 + 64))


def build_handoff(snapshot, metadata_receipt, metadata, selection_contract):
    complete = snapshot["outputs"]["COMPLETE.json"]
    terminal = snapshot["receipts"]["terminal.json"]
    cleanup = snapshot["receipts"]["cleanup.json"]
    if (snapshot.get("job") is not None or snapshot.get("pods") != []
            or terminal.get("status") != "passed" or terminal.get("application_pass") is not True
            or terminal.get("kubernetes_complete") is not True
            or terminal.get("result_complete_sha256") != PARENT_COMPLETE
            or cleanup.get("status") != "complete"
            or any(r.get("uid") != PARENT_UID or r.get("job") != PARENT_JOB for r in (terminal, cleanup))
            or complete.get("run_id") != PARENT_JOB or complete.get("plan_sha256") != PARENT_PLAN
            or terminal.get("plan_sha256") != PARENT_PLAN):
        raise ValueError("Completed exact parent and cleanup must be established")
    required = dict(input_lease_exit_verified=True, scientific_training_complete=True,
                    scientific_optimizer_updates=23040, attempted_optimizer_updates=23040,
                    query_deployment_performed=False, assessment_performed=False,
                    D2_scores_computed=False, D1_endpoint_access=False, LARRY_access=False,
                    scientific_acceptance="not_assessed_fitting_only")
    if any(complete.get(k) != v or type(complete.get(k)) is not type(v) for k, v in required.items()):
        raise ValueError("Parent completion or phase boundary differs")
    if metadata_receipt["complete_identity"]["sha256"] != PARENT_COMPLETE:
        raise ValueError("Metadata read is not bound to the completed parent")
    fits, runs = complete["fits"], metadata_receipt["runs"]
    for rows in (fits, runs):
        if tuple((r["candidate_id"], r["seed"]) for r in rows) != EXPECTED_FITS:
            raise ValueError("All nine distinct candidate/seed identities are required")
    guide_rows = metadata["canonical_guides"]
    if len(guide_rows) != 26504 or [r["guide_index"] for r in guide_rows] != list(range(26504)):
        raise ValueError("Complete ordered guide catalog is required")
    if (selection_contract.get("selection_donor") != "D2"
            or selection_contract.get("seeds") != [0, 1, 2]
            or selection_contract.get("prefix") != 4096
            or selection_contract.get("refinement_prefix") != 8192
            or [r["id"] for r in selection_contract.get("candidates", [])] != list(CANDIDATES)):
        raise ValueError("Frozen D2 decision contract differs")
    train, query = None, None
    bindings = []
    for fit, run in zip(fits, runs, strict=True):
        if (fit.get("optimizer_updates") != 2560 or fit.get("attempted_optimizer_updates") != 2560
                or fit.get("epochs_completed") != 60
                or fit.get("scientific_assessment") != "not_performed_fitting_only"):
            raise ValueError("Incomplete fit cannot enter deployment")
        if (run["checkpoint_sha256"] != fit["checkpoint"]["sha256"]
                or run["compiled_problem"]["problem_hash"] != metadata_receipt["compiled"]["problem_hash"]):
            raise ValueError("Model and data lineage mismatch")
        split = run["validation_split"]
        current_train = exact_ids(split["train_measure_ids"], {"D3", "D4"})
        current_query = exact_ids(split["validation_measure_ids"], {"D2"})
        if (split["train_time_labels"] != ["Stim48hr"]
                or split["validation_time_labels"] != ["Stim48hr"]):
            raise ValueError("Wrong continuation endpoint")
        if train is None:
            train, query = current_train, current_query
        elif train != current_train or query != current_query:
            raise ValueError("Unmatched training or query populations across candidate/seeds")
        if len(train) != 46809 or len(query) != 23421:
            raise ValueError("Parent-bound full support changed")
        relative = f"fits/{fit['candidate_id']}/seed-{fit['seed']}/state/checkpoint.pt"
        if fit["checkpoint_path"] != relative:
            raise ValueError("Checkpoint path was relabeled")
        bindings.append(dict(candidate_id=fit["candidate_id"], seed=fit["seed"],
            checkpoint_path=PARENT_OUTPUT + "/" + relative, **fit["checkpoint"],
            parameter_count=fit["parameter_count"], completed_epochs=60,
            architecture=dict(ARCHITECTURES[fit["candidate_id"]], latent_dim=48, n_programs=8,
                              context_mode="none", sigma_min=0.001, growth_max=3.0),
            compiled_problem_hash=run["compiled_problem"]["problem_hash"],
            run_manifest=dict(path=PARENT_OUTPUT + "/" + relative.rsplit("/state/", 1)[0] + "/run.json",
                              **run["identity"])))
    fitted_targets = set()
    for measure in train:
        row = guide_rows[int(measure.split("::guide-index-")[1])]
        fitted_targets.add("shared-NTC-reference" if row["is_design_ntc"] else "target::" + row["primary_target_id"])
    query_records = []
    for measure in query:
        index = int(measure.split("::guide-index-")[1])
        row = guide_rows[index]
        embedding = "shared-NTC-reference" if row["is_design_ntc"] else "target::" + row["primary_target_id"]
        if embedding not in fitted_targets:
            raise ValueError("Unfitted target cannot be relabeled as a supported forecast")
        query_records.append(dict(measure_id=measure, guide_index=index, guide_id=row["sgRNA"],
            target_id="__NTC__" if row["is_design_ntc"] else row["primary_target_id"],
            is_control=row["is_design_ntc"], embedding_id=embedding))
    if len(fitted_targets) != 12555:
        raise ValueError("Fitted embedding inventory differs")
    allocated = forecast_bytes(len(query))
    nine_bytes = len(bindings) * allocated
    return dict(schema="credo.full_ddr.d2_metadata_handoff.v1",
        status="metadata_bound_not_execution_ready", selection_donor="D2",
        parent=dict(run_id=PARENT_JOB, uid=PARENT_UID, plan_sha256=PARENT_PLAN,
                    completion_sha256=PARENT_COMPLETE, output_root=PARENT_OUTPUT),
        checkpoints=bindings, expected_train_measure_ids=train,
        query_measure_ids=query, query_records=query_records,
        expected_embedding_ids=sorted(fitted_targets), query_inventory_sha256=digest(query_records),
        selection_contract_sha256=digest(selection_contract),
        query_modeled_guides=len(query), query_controls=sum(r["is_control"] for r in query_records),
        query_target_identities=len({r["target_id"] for r in query_records if not r["is_control"]}),
        complete_catalog_size=26504, complementary_categories=26504-len(query),
        endpoint_support_manifest="pending_metadata_binding_not_candidate_score_selection",
        forecast=dict(prefixes=[4096, 8192], steps_per_interval=8, seeds=[0, 1, 2],
                      fit_count=9, guide_model_rollouts=len(query)*9,
                      maximum_prefix_particles=len(query)*9*8192,
                      scalar_particle_steps=len(query)*9*8192*8,
                      reuse_same_largest_prefix_trajectories=True,
                      per_bank_admission_bytes=allocated, nine_bank_admission_bytes=nine_bytes,
                      nine_bank_admission_GiB=nine_bytes/1024**3,
                      previous_256_GiB_proposal_sufficient=nine_bytes <= 256*1024**3),
        proposed_resources=dict(gpus=1, gpu="NVIDIA_H100_80GB", cpus=24, host_memory_GiB=200,
                                threads=8, maximum_output_GiB=400, approval="requested_not_assumed"),
        phase_boundaries=dict(training_updates=0, D1_access=False, LARRY_access=False,
                              scoring_before_prediction_seal=False, retries=0, elapsed_time_deadline=None),
        remaining=["400_GiB_output_authority_and_fresh_capacity",
                   "D2_common_endpoint_support_metadata_and_truth_artifact_binding",
                   "source_only_checkpoint_loader_and_forecast_runtime_qualification",
                   "all_nine_forecasts_and_reference_seal",
                   "R48_decoder_raw_observable_and_separate_endpoint_effect_reporting_binding",
                   "GPU_score_interface_qualification_then_frozen_D2_selection"],
        new_scientific_result=False, neural_inference_performed=False, remote_launch_performed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(snapshot=ROOT / "fitting_gpu_native_v6/STATUS_20261010T174515Z.json",
                 metadata_receipt=ROOT / "fitting_gpu_native_v6/D2_HANDOFF_METADATA_20261010.json",
                 metadata=ROOT / "METADATA_PREPARATION.json",
                 selection_contract=ROOT / "SELECTION_CONTRACT.json")
    values, identities = {}, {}
    for key, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Regular source metadata required")
        raw = path.read_bytes()
        values[key] = json.loads(raw)
        identities[key] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    result = build_handoff(**values)
    result["input_metadata"] = identities
    publish_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "query_modeled_guides", "query_controls",
        "query_target_identities", "complementary_categories", "forecast", "remaining")}, indent=2))


if __name__ == "__main__":
    main()
