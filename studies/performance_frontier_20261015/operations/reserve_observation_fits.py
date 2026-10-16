"""Reserve one complete observation-fit batch; metadata only, never a launcher.

Requires the exact qualification already reconciled COMPLETE in the real
journal, imported successful terminal/readback/cleanup/absence evidence, and a
new capacity receipt collected after that evidence. No elapsed-time expiry is
introduced. Launch staging still performs its own fresh capacity check.

Every requested subset is prevalidated under the journal lock before the first
reservation. All exports cite the final batch event and final aggregate totals.
An interrupted append/export is an explicit reconciliation condition, never an
automatic retry or reason to remove an existing reservation.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.build import NATIVE_ROOT, READER_MEMBERS, SOURCE_MEMBERS, validate_qualified_hardware
from deployment.common import (GIB, IMAGE, PROJECT, STUDY, checked_path, encoded,
                               identity, publish, read, read_json,
                               validate_admission, validate_authority, validate_plan)
from frontier.coordination import AdmissionJournal, LIVE, TERMINAL, WorkBinding


COLLECTOR_SHA = "a847dc3a31716a68f333155a07ca128842d6e04dbd25c8833b1de6608e695bf1"
OUTPUT_BYTES = 4 * GIB
UPDATES = 20000
PAIRED_FP32_TOLERANCES = {"loss_atol": 2e-5, "loss_rtol": 2e-5,
                         "gradient_atol": 2e-5, "gradient_rtol": 2e-4}


def sha(path):
    return identity(read(path))["sha256"]


def _digest(value):
    return identity(encoded(value))["sha256"]


def _timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is None:
            raise ValueError
        return result
    except (TypeError, AttributeError, ValueError) as error:
        raise ValueError("Explicit timezone-aware receipt timestamp required") from error


def _wrapped(snapshot, name):
    wrapped = snapshot.get("records", {}).get(name)
    if not isinstance(wrapped, dict) or set(wrapped) != {"sha256", "value"}:
        raise ValueError("Missing exact imported receipt: " + name)
    value = wrapped["value"]
    # These are the actual retained serializers: worker receipts are indented;
    # lifecycle/controller receipts are compact, both with one final newline.
    raw = ((json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
           if name.startswith("output/") else encoded(value))
    if identity(raw)["sha256"] != wrapped["sha256"]:
        raise ValueError("Imported receipt content/hash mismatch: " + name)
    return value, wrapped["sha256"], len(raw)


def _validate_paired_fp32(paired):
    """Mirror observation_worker.verify_qualification's static paired checks.

    Do not import its numerical module merely to validate metadata. Its exact
    bytes are already source-bound above; tests compare the literal tolerances.
    Runtime/device equality remains the subsequent actual-worker gate.
    """
    required = {"schema": "credo.frontier.observation-paired-fp32.v1", "status": "passed",
                "shape": [16, 18129], "architecture": [18129, 512, 128, 48],
                "tolerances": PAIRED_FP32_TOLERANCES, "optimizer_calls": 0,
                "all_parameter_gradients_checked": True, "same_initial_weights": True,
                "cpu_encoding": "sparse", "cuda_encoding": "bounded_dense"}
    if not isinstance(paired, dict) or any(paired.get(k) != v for k, v in required.items()):
        raise ValueError("Missing or mismatched same-source paired FP32 qualification")
    for name in ("loss", "gradient_summary"):
        measured = paired.get(name, {})
        normalized = measured.get("maximum_tolerance_normalized_error") if isinstance(measured, dict) else None
        if (not isinstance(measured, dict) or measured.get("passed") is not True
                or type(normalized) not in (int, float) or not 0 <= normalized <= 1
                or not math.isfinite(normalized)):
            raise ValueError("Failed same-source paired FP32 tolerance comparison")


def qualification_work(study_root, package, plan, source_sha, protocol_sha, phase_sha):
    """Reconstruct the reservation from bound admission, including v2 approval.

    Metadata validation only: this neither proves numerical completion nor
    authorizes a successor. V2 explicitly approves qualification, not new fits.
    Unknown recovery generations require their own reviewed binding.
    """
    reference = plan.get("admission", {})
    if (set(reference) != {"path", "sha256"}
            or reference["path"] != plan["bundle_root"] + "/ADMISSION.json"):
        raise ValueError("Exact qualification admission reference required")
    admission = read_json(package / "bundle/ADMISSION.json", reference["sha256"])
    validate_admission(plan, admission)
    if (admission.get("source_manifest_sha256") != source_sha
            or admission.get("phase_authority_sha256") != phase_sha):
        raise ValueError("Qualification admission source/authority differs")
    operation = {"run_id": plan["run_id"], "kind": "qualification", "phase_authority_sha256": phase_sha,
                 "source_manifest_sha256": source_sha, "protocol_sha256": protocol_sha}
    recovery_sha = None
    if plan["run_id"] == "credo-frontier-obs-qual-20261015-v1":
        if (plan["phase_directory"] != "observation_qualification_v1"
                or any(k in admission for k in ("requalification_authority", "requalification_authority_sha256"))):
            raise ValueError("Original qualification cannot acquire recovery authority")
    elif plan["run_id"] == "credo-frontier-obs-qual-20261015-v2":
        recovery_sha = admission.get("requalification_authority_sha256")
        if not isinstance(recovery_sha, str) or not re.fullmatch("[0-9a-f]{64}", recovery_sha):
            raise ValueError("Exact recovery authority hash required")
        authority = read_json(study_root / "OBSERVATION_REQUALIFICATION_V2_AUTHORITY.json", recovery_sha)
        expected = {"schema": "credo.frontier.observation-requalification-authority.v1",
                    "user_instruction": "run the corrected qualification", "run_id": plan["run_id"],
                    "phase_directory": "observation_qualification_v2", "phase_contract_sha256": phase_sha,
                    "numerical_protocol_sha256": protocol_sha}
        work = {"planned_disposable_CUDA_optimizer_calls": 13, "maximum_disposable_CUDA_optimizer_calls": 16,
                "scientific_updates": 0, "real_biological_payload_reads": False, "historical_checkpoint_reads": False,
                "LARRY_access": False, "automatic_fit_successor": False, "automatic_retry": False,
                "elapsed_time_deadline": None}
        resources = {"jobs": 1, "H100_per_job": 1, "CPUs": 24, "host_memory_GiB": 200,
                     "application_threads": 8, "output_cap_bytes": 16 * GIB, "aggregate_study_H100_ceiling": 4,
                     "aggregate_study_output_cap_bytes": 384 * GIB, "unallocated_fileset_reserve_bytes": 16 * GIB}
        if (authority != admission.get("requalification_authority")
                or any(authority.get(k) != v for k, v in expected.items())
                or plan["phase_directory"] != expected["phase_directory"]
                or encoded(authority.get("work")) != encoded(work)
                or encoded(authority.get("resources")) != encoded(resources)
                or authority.get("proposal_sha256") != sha(study_root / "OBSERVATION_HARDWARE_RECOVERY_PROPOSAL.json")):
            raise ValueError("Bound qualification-only recovery authority differs")
        _timestamp(authority.get("recorded_at_utc"))
        operation["requalification_authority_sha256"] = recovery_sha
    else:
        raise ValueError("Unregistered qualification generation; explicit recovery binding required")
    work = WorkBinding.other("qualification", _digest(operation))
    if WorkBinding.from_dict(admission.get("work_binding")) != work:
        raise ValueError("Qualification admission work identity differs")
    return work.to_dict(), reference["sha256"], recovery_sha


def qualification_evidence(study_root, package, snapshot_path):
    """Validate locally imported producer evidence; do not query the cluster."""
    study_root, package = checked_path(study_root), checked_path(package)
    if not package.is_relative_to(study_root):
        raise ValueError("Qualification package must belong to this study")
    build, plan = read_json(package / "BUILD.json"), read_json(package / "bundle/PLAN.json")
    validate_plan(plan)
    plan_sha = sha(package / "bundle/PLAN.json")
    source_raw = encoded({"schema": "credo.frontier.observation-worker-source.v1",
                         "files": {name: sha(study_root / name) for name in SOURCE_MEMBERS}})
    source_sha = identity(source_raw)["sha256"]
    protocol_sha, phase_sha = sha(study_root / "OBSERVATION_PROTOCOL.json"), sha(study_root / "OBSERVATION_PHASE_CONTRACT.json")
    validate_authority(read_json(study_root / "OBSERVATION_PHASE_CONTRACT.json"))
    if (plan["mode"] != "qualify" or build.get("mode") != "qualify"
            or build.get("run_id") != plan["run_id"] or build.get("plan_sha256") != plan_sha
            or build.get("source_manifest_sha256") != source_sha
            or build.get("phase_authority_sha256") != phase_sha):
        raise ValueError("Exact same-source qualification package required")
    expected_refs = {"source_manifest": ("SOURCE_MANIFEST.json", source_sha),
                     "protocol": ("OBSERVATION_PROTOCOL.json", protocol_sha),
                     "phase_authority": ("OBSERVATION_PHASE_CONTRACT.json", phase_sha),
                     "parent_authority": ("AUTHORITY.json", sha(study_root / "AUTHORITY.json"))}
    for field, (name, digest) in expected_refs.items():
        if (plan.get(field) != {"path": plan["bundle_root"] + "/" + name, "sha256": digest}
                or sha(package / "bundle" / name) != digest):
            raise ValueError("Qualification source/protocol/authority binding differs")
    work_binding, admission_sha, recovery_sha = qualification_work(
        study_root, package, plan, source_sha, protocol_sha, phase_sha)
    created = read_json(package / "CREATE.json")["records"]["live-binding.json"]
    if (created.get("job") != plan["run_id"] or created.get("run_id") != plan["run_id"]
            or created.get("plan_sha256") != plan_sha or not created.get("uid")):
        raise ValueError("Qualification creation UID/plan binding differs")
    snapshot = read_json(snapshot_path)
    if (snapshot.get("schema") != "credo.frontier.exact-observation-status.v1"
            or snapshot.get("run_id") != plan["run_id"] or snapshot.get("uid") != created["uid"]
            or snapshot.get("plan_sha256") != plan_sha or snapshot.get("remote_writes") is not False
            or snapshot.get("job", "not_checked") is not None or snapshot.get("pods") != []):
        raise ValueError("Exact terminal qualification Job/Pod absence evidence required")
    for name in ("startup-failure.json", "supervisor-failed.json", "handshake/ENTRY_FAILED.json", "output/WORKER_FAILED.json"):
        if snapshot.get("records", {}).get(name) is not None:
            raise ValueError("Qualification failure evidence present")
    terminal, terminal_sha, _ = _wrapped(snapshot, "terminal.json")
    cleanup, cleanup_sha, _ = _wrapped(snapshot, "cleanup.json")
    readback, readback_sha, _ = _wrapped(snapshot, "output-readback.json")
    complete, complete_sha, complete_size = _wrapped(snapshot, "output/QUALIFICATION_COMPLETE.json")
    paired, paired_sha, paired_size = _wrapped(snapshot, "output/PAIRED_FP32_CHECK.json")
    expected_terminal = {"status": "passed", "job": plan["run_id"], "uid": created["uid"],
        "run_id": plan["run_id"], "kubernetes_complete": True, "application_pass": True,
        "reason": None, "mode": "qualify", "lifecycle_plan_sha256": plan_sha,
        "result_complete_sha256": complete_sha, "automatic_successor": False}
    if terminal != expected_terminal:
        raise ValueError("Successful exact qualification terminal required")
    absence = {"job_absent": True, "owned_pods": 0}
    if (cleanup.get("status") != "complete" or cleanup.get("job") != plan["run_id"]
            or cleanup.get("uid") != created["uid"] or len(cleanup.get("reads", [])) < 2
            or cleanup["reads"][-2:] != [absence, absence]):
        raise ValueError("Exact cleanup with two absence reads required")
    expected_complete = {"schema": "credo.frontier.observation-cuda-qualification.v1", "status": "passed",
        "source_manifest_sha256": source_sha, "protocol_sha256": protocol_sha,
        "runtime_image": IMAGE, "scientific_updates": 0, "actual_disposable_optimizer_calls": 13,
        "real_count_payloads_opened": False, "real_input_mount_qualified": False,
        "restart_equivalence": "exact_model_optimizer_sampler_rng", "shape": [2048, 18129], "nonzeros_per_cell": 4096}
    if any(complete.get(k) != v for k, v in expected_complete.items()):
        raise ValueError("Numerical qualification is missing, failed or source-mismatched")
    if (paired.get("schema") != "credo.frontier.observation-paired-fp32.v1" or paired.get("status") != "passed"
            or paired.get("optimizer_calls") != 0 or complete.get("paired_fp32_check") != paired
            or complete.get("paired_fp32_receipt") != {"path": plan["scientific_output"] + "/PAIRED_FP32_CHECK.json",
                                                       "sha256": paired_sha, "size_bytes": paired_size}):
        raise ValueError("Separate paired qualification receipt is absent or changed")
    _validate_paired_fp32(paired)
    qualified_hardware = validate_qualified_hardware(complete.get("hardware"))
    qualified_peaks = []
    for name in ("resources_after_warmup", "resources_all_checks"):
        measured = complete.get(name, {})
        reserved = measured.get("peak_reserved_bytes")
        allocated = measured.get("peak_allocated_bytes")
        headroom = measured.get("physical_headroom_at_peak_reserved_bytes")
        if (any(type(value) is not int for value in (reserved, allocated, headroom))
                or not 0 < allocated <= reserved
                or headroom != qualified_hardware["total_memory_bytes"] - reserved
                or headroom < qualified_hardware["required_headroom_bytes"]):
            raise ValueError("Measured qualification peak plus 8-GiB headroom proof is missing or invalid")
        qualified_peaks.append(reserved)
    if qualified_peaks[1] < qualified_peaks[0]:
        raise ValueError("Measured qualification peak plus 8-GiB headroom proof is nonmonotonic")
    qualified_peak = max(qualified_peaks)
    files = readback.get("files", {})
    if (readback.get("schema") != "credo.frontier.observation-output-readback.v1"
            or readback.get("status") != "passed" or readback.get("run_id") != plan["run_id"]
            or readback.get("no_model_loading") is not True or readback.get("complete_sha256") != complete_sha
            or files.get("QUALIFICATION_COMPLETE.json") != {"sha256": complete_sha, "size_bytes": complete_size}
            or files.get("PAIRED_FP32_CHECK.json") != {"sha256": paired_sha, "size_bytes": paired_size}
            or not files or any(type(row.get("size_bytes")) is not int or row["size_bytes"] < 0 for row in files.values())
            or readback.get("total_bytes") != sum(row["size_bytes"] for row in files.values())
            or not 0 < readback["total_bytes"] <= plan["maximum_output_bytes"]):
        raise ValueError("Exact complete output inventory/readback required")
    return {"run_id": plan["run_id"], "uid": created["uid"], "snapshot_sha256": sha(snapshot_path),
        "snapshot_at_utc": snapshot["at_utc"], "terminal_sha256": terminal_sha, "cleanup_sha256": cleanup_sha,
        "readback_sha256": readback_sha, "numerical_output_bytes": readback["total_bytes"],
        "qualification_storage_reservation_bytes": 16 * GIB,
        "qualification_receipt": plan["scientific_output"] + "/QUALIFICATION_COMPLETE.json",
        "qualification_sha256": complete_sha, "qualified_hardware": qualified_hardware,
        "qualified_peak_reserved_bytes": qualified_peak,
        "source_manifest_sha256": source_sha,
        "protocol_sha256": protocol_sha, "phase_authority_sha256": phase_sha,
        "qualification_admission_sha256": admission_sha,
        "requalification_authority_sha256": recovery_sha,
        "successor_fit_authority_from_qualification": False,
        "work_binding": work_binding}


def _capacity(path, evidence):
    value = read_json(path)
    quota = value.get("quota", {})
    numeric = ("headroom_bytes", "filesystem_available_bytes", "filesystem_available_inodes")
    if (value.get("schema") != "credo.frontier.storage-intake.v1" or value.get("study_id") != STUDY
            or value.get("fileset") != "h_bcb-yding4" or value.get("collector_sha256") != COLLECTOR_SHA
            or value.get("remote_writes") is not False or value.get("numerical_payloads_opened") is not False
            or any(type(value.get(k)) is not int or value[k] < 0 for k in numeric)
            or value["filesystem_available_inodes"] < 20000
            or quota.get("block_grace") != "none" or quota.get("file_grace") != "none"
            or any(type(quota.get(k)) is not int or quota[k] < 0 for k in ("soft_bytes", "hard_bytes", "used_bytes", "in_doubt_bytes"))):
        raise ValueError("Fresh exact metadata-only capacity receipt required")
    limits = [quota[k] for k in ("soft_bytes", "hard_bytes") if quota[k] > 0]
    if (not limits or min(limits) - quota["used_bytes"] - quota["in_doubt_bytes"] != value["headroom_bytes"]
            or _timestamp(value["utc"]) <= _timestamp(evidence["snapshot_at_utc"])):
        raise ValueError("Capacity must reconcile quota and postdate successful cleanup snapshot")
    return min(value["headroom_bytes"], value["filesystem_available_bytes"])


def _requests(root, request_path, evidence):
    request = read_json(request_path)
    if set(request) != {"schema", "requested_fits"} or request["schema"] != "credo.frontier.observation-fit-batch-request.v1":
        raise ValueError("Explicit observation subset batch request required")
    rows = request["requested_fits"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError("One through four explicitly requested fits required")
    protocol = read_json(root / "OBSERVATION_PROTOCOL.json")
    metadata = read_json(root / "OBSERVATION_SUBSET_METADATA_PLAN.json")
    if (metadata.get("status") != "metadata_only_not_admitted_not_launched"
            or metadata.get("observation_protocol_sha256") != evidence["protocol_sha256"]
            or metadata.get("native_inputs_source_sha256") != sha(root / "frontier/native_inputs.py")
            or metadata.get("metadata_index_sha256") != protocol["metadata_index_identity"]
            or metadata.get("rna_axis_sha256") != protocol["RNA_feature_axis_identity"]
            or metadata.get("reader_files") != READER_MEMBERS
            or [r["training_donors"] for r in metadata["subsets"]] != protocol["training_subsets"]):
        raise ValueError("Exact canonical metadata-only subset inventory required")
    subset_rows = {tuple(r["training_donors"]): r for r in metadata["subsets"]}
    seen_runs, seen_phases, seen_subsets, prepared, indexes = set(), set(), set(), [], []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"run_id", "phase", "training_donors"}
                or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", row["run_id"])
                or not re.fullmatch(r"observation_[a-z0-9_]+_v[0-9]+", row["phase"])):
            raise ValueError("Exact new scalar run/phase/subset identities required")
        donors = tuple(row["training_donors"])
        work = WorkBinding.representation(donors)
        if row["run_id"] in seen_runs or row["phase"] in seen_phases or donors in seen_subsets:
            raise ValueError("Duplicate run, phase or canonical subset in batch")
        seen_runs.add(row["run_id"]); seen_phases.add(row["phase"]); seen_subsets.add(donors)
        info = subset_rows[donors]
        fold_id = "subset-" + "-".join(donors)
        fold = {"fold_id": fold_id, "training_donors": donors, "task": "Stim8hr_to_Stim48hr", "Rest": "excluded"}
        fold_sha = hashlib.sha256(json.dumps(fold, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        if (info["fold_id"] != fold_id or info["fold_sha256"] != fold_sha
                or not re.fullmatch("[0-9a-f]{64}", info["input_access_sha256"])):
            raise ValueError("Canonical subset fold/access identity differs")
        indexes.append(protocol["training_subsets"].index(list(donors)))
        bindings = {"metadata_root": NATIVE_ROOT, "dataset_root": NATIVE_ROOT,
            "reader_source_root": PROJECT + "/" + row["phase"] + "/bundle/retained_reader",
            "fold_id": fold_id, "training_donors": list(donors),
            "metadata_index_sha256": metadata["metadata_index_sha256"], "rna_axis_sha256": metadata["rna_axis_sha256"],
            "input_access_sha256": info["input_access_sha256"],
            "qualification_receipt": evidence["qualification_receipt"], "qualification_sha256": evidence["qualification_sha256"]}
        prepared.append({**row, "work_binding": work, "fit_bindings": bindings})
    if indexes != sorted(indexes):
        raise ValueError("Requested subsets must follow canonical protocol order")
    return prepared


def reserve_batch(study_root, qualification_package, snapshot_path, capacity_path, request_path, output):
    """Explicit local mutation only when called; never called by import/tests on real journals."""
    root, output = checked_path(study_root), checked_path(output, exists=False)
    if root.name != STUDY or not output.is_relative_to(root) or not output.parent.is_dir() or output.exists():
        raise ValueError("New prepared study-local batch export destination required; no retry")
    evidence = qualification_evidence(root, qualification_package, snapshot_path)
    capacity_sha = sha(capacity_path)
    headroom = _capacity(capacity_path, evidence)
    rows = _requests(root, request_path, evidence)
    phase_sha = evidence["phase_authority_sha256"]
    if not (root / "admission/IDENTITY.json").is_file():
        raise ValueError("Existing qualification admission journal required; this helper cannot create one")
    with AdmissionJournal(root / "admission", study_identity_sha256=sha(root / "AUTHORITY.json")) as journal:
        state = journal.snapshot()
        qual = state["tasks"].get(evidence["run_id"])
        if (qual is None or qual["status"] != "complete" or qual["scheduler_id"] != evidence["uid"]
                or qual["work_binding"] != evidence["work_binding"] or qual["admission_sha256"] != phase_sha
                or qual["evidence_sha256"] not in (evidence["snapshot_sha256"], evidence["terminal_sha256"])
                or qual["storage_reserved_bytes"] != 16 * GIB or "retained_inventory_sha256" in qual):
            # Numerical output readback omits source, handshake and controller
            # files. It cannot authorize releasing the full phase reservation.
            # A future reduction needs a separate complete-phase inventory.
            raise ValueError("Exact qualification must already be reconciled complete with its unreleased 16-GiB cap")
        if any(t["capacity_receipt_sha256"] == capacity_sha for t in state["tasks"].values()):
            raise ValueError("New batch requires a newly collected, unused capacity receipt")
        planned = deepcopy(state["tasks"])
        prior_trajectories = {WorkBinding.from_dict(t["work_binding"]).trajectory_identity for t in planned.values()}
        for row in rows:
            if row["run_id"] in planned or row["work_binding"].trajectory_identity in prior_trajectories:
                raise ValueError("Subset or task already reserved, even if terminal; no retries")
            planned[row["run_id"]] = {"worker_id": row["run_id"], "kind": "representation",
                "scientific_update_reservation": UPDATES, "storage_reserved_bytes": OUTPUT_BYTES,
                "admission_sha256": phase_sha, "capacity_receipt_sha256": capacity_sha,
                "capacity_headroom_bytes": headroom, "status": "reserved", "scheduler_id": None,
                "work_binding": row["work_binding"].to_dict()}
        # Reuse the journal's complete cap/identity validator before mutation,
        # rather than discovering a later member cannot be admitted mid-batch.
        journal._validate(planned)
        outstanding = sum(t["storage_reserved_bytes"] for t in planned.values()
                          if t["status"] not in TERMINAL or "retained_inventory_sha256" not in t)
        if headroom < outstanding + journal.identity["unallocated_reserve_bytes"]:
            raise ValueError("Insufficient fresh quota headroom for the complete batch and reserve")
        output.mkdir(mode=0o700)
        publish(output / "BATCH_INTENT.json", {"schema": "credo.frontier.observation-fit-batch-intent.v1",
            "request_sha256": sha(request_path), "qualification_evidence": evidence,
            "capacity_receipt_sha256": capacity_sha, "journal_predecessor_sha256": state["last_sha256"],
            "requested_run_ids": [r["run_id"] for r in rows], "automatic_retry": False})
        for row in rows:
            state = journal.reserve(row["run_id"], worker_id=row["run_id"], kind="representation",
                scientific_updates=UPDATES, output_bytes=OUTPUT_BYTES, admission_sha256=phase_sha,
                capacity_receipt_sha256=capacity_sha, fileset_headroom_bytes=headroom, work_binding=row["work_binding"])
        aggregate = sum(t["storage_reserved_bytes"] for t in state["tasks"].values())
        live = sum(t["status"] in LIVE for t in state["tasks"].values())
        exports = []
        for row in rows:
            admission = {"schema": "credo.frontier.observation-admission-export.v1", "run_id": row["run_id"],
                "status": "reserved", "storage_reserved_bytes": OUTPUT_BYTES, "scientific_update_reservation": UPDATES,
                "aggregate_reserved_output_bytes": aggregate, "live_h100_reservations": live,
                "journal_event_sha256": state["last_sha256"], "journal_identity_sha256": sha(root / "admission/IDENTITY.json"),
                "phase_authority_sha256": phase_sha, "source_manifest_sha256": evidence["source_manifest_sha256"],
                "qualification_sha256": evidence["qualification_sha256"], "qualified_hardware": evidence["qualified_hardware"],
                "qualified_peak_reserved_bytes": evidence["qualified_peak_reserved_bytes"],
                "capacity_receipt_sha256": capacity_sha, "work_binding": row["work_binding"].to_dict(),
                "scheduler_submitted": False, "scientific_updates": 0}
            member = output / row["run_id"]
            member.mkdir(mode=0o700)
            a = publish(member / "ADMISSION.json", admission)
            b = publish(member / "FIT_BINDINGS.json", row["fit_bindings"])
            exports.append({"run_id": row["run_id"], "phase": row["phase"], "training_donors": row["training_donors"],
                "admission": {"path": str(member / "ADMISSION.json"), **a},
                "fit_bindings": {"path": str(member / "FIT_BINDINGS.json"), **b}})
        result = {"schema": "credo.frontier.observation-fit-batch-reservation.v1", "status": "reserved_not_launched",
            "fits": exports, "journal_event_sha256": state["last_sha256"], "aggregate_reserved_output_bytes": aggregate,
            "live_h100_reservations": live, "new_scientific_update_reservations": len(rows) * UPDATES,
            "new_output_reservations_bytes": len(rows) * OUTPUT_BYTES, "scheduler_calls": 0,
            "numerical_payloads_opened": False, "optimizer_calls": 0, "qualification_evidence": evidence}
        publish(output / "BATCH_RESERVED.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-package", type=Path, required=True)
    parser.add_argument("--qualification-status", type=Path, required=True)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reserve_batch(ROOT, args.qualification_package, args.qualification_status,
                           args.capacity_receipt, args.request, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
