"""Delegate only fixed same-Pod fixture cases under an external parent release.

No scheduler reservation is made here. Child storage and GPU use are included
in the parent's allocation; these records cannot authorize scientific fitting.
"""

from dataclasses import asdict
from pathlib import Path
import json
import os

from .main_qualification_artifacts import write_json
from .main_qualification_contract import exact
from .main_qualification_sequence import _read
from .main_worker import validate_admission, validate_release, qualification_binding
from .main_worker_contract import decode_contract
from .observation_loader import ObservationFile
from .retention import STUDY_ID, identity_sha256


def issue_case(request_ref, *, parent, references, admission, release):
    """Called only after parent validation, and checked again by the child."""
    exact(release, dict(run_id=parent["run_id"], delegate_fixed_worker_cases=True,
        disposable_qualification_authorized=True, scientific_fit_authorized=False,
        real_biological_payload_access=False, admission_sha256=references["admission"].sha256,
        phase_authority_sha256=references["authority"].sha256))
    request = _read(request_ref)
    exact(request, dict(schema="credo.frontier.main-worker-qualification-request.v1",
        launch_authority=False, optimizer_calls=0))
    fields = request["contract_fields"]
    folder = Path(request_ref.path).parent
    placeholder = ObservationFile(str(folder/"ADMISSION.json"), "0"*64, 1)
    record, parts = decode_contract(json.dumps(dict(fields, admission=asdict(placeholder))).encode(),
                                   qualification=True)
    root = Path(parent["output_root"])/"qualification-fixtures"/parent["run_id"]/STUDY_ID
    case = parts["qualification_case"]
    if (Path(record["study_root"]) != root or record["qualification_parent_id"] != parent["run_id"]
            or folder != root/"worker_requests"/case.stage
            or parts["source_manifest"].sha256 != references["worker_source_manifest"].sha256):
        raise ValueError("Fixed same-parent fixture/source required for delegated case")
    child_admission = dict(schema="credo.frontier.main-worker-qualification-admission.v1",
        run_id=record["worker_id"], parent_run_id=parent["run_id"], status="reserved",
        work_binding=qualification_binding().to_dict(), stage_plan_sha256=identity_sha256(asdict(parts["plan"])),
        source_manifest_sha256=parts["source_manifest"].sha256, input_reference=asdict(parts["input_reference"]),
        resource_policy_sha256=parts["resource_policy"].identity, qualification_sha256=parts["qualification"].sha256,
        scientific_update_reservation=0, disposable_optimizer_calls=case.optimizer_calls, case=asdict(case),
        real_biological_payload_access=False, storage_reserved_bytes=parts["resource_policy"].output_limit_bytes,
        selection_evidence=[], automatic_retry=False, automatic_successor=False,
        allocation_scope="Suballocation of the existing parent; not a new GPU or journal reservation",
        **{k: admission[k] for k in ("journal_event_sha256", "journal_identity_sha256",
            "phase_authority_sha256", "capacity_receipt_sha256", "aggregate_reserved_output_bytes",
            "live_h100_reservations")})
    validate_admission(record, parts, child_admission)
    aref = write_json(folder/"ADMISSION.json", child_admission)
    cref = write_json(folder/"CONTRACT.json", dict(fields, admission=asdict(aref)))
    record, parts = decode_contract(json.dumps(_read(cref)).encode(), qualification=True)
    child_release = dict(schema="credo.frontier.main-worker-qualification-startup-release.v1",
        status="startup_released", run_id=record["worker_id"], parent_run_id=parent["run_id"],
        worker_contract_sha256=cref.sha256, admission_sha256=aref.sha256,
        source_manifest_sha256=parts["source_manifest"].sha256, qualification_sha256=parts["qualification"].sha256,
        qualification_terminal_readback_verified=False, selection_evidence_verified=True,
        scientific_fit_authorized=False, disposable_qualification_authorized=True,
        real_biological_payload_access=False, disposable_optimizer_calls=case.optimizer_calls,
        parent_qualification_authority_verified=True, worker_uid=os.getuid(), worker_gid=os.getgid(),
        **{k: release[k] for k in ("pod_name", "pod_uid", "job_uid", "startup_sha256", "image",
            "journal_event_sha256", "capacity_receipt_sha256", "phase_authority_sha256",
            "actual_pod_audit_passed", "storage_and_fileset_quota_passed", "aggregate_reservation_verified",
            "cgroup_resources_verified", "authorized_cpus", "authorized_memory_bytes", "threads", "gpus",
            "automatic_retry", "automatic_successor")})
    validate_release(record, parts, cref, child_release, child_admission, os.environ)
    return cref, write_json(folder/"RELEASE.json", child_release)
