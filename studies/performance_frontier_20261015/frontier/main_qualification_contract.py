"""Exact prospective main-qualification scope; no execution or self-approval."""

from dataclasses import asdict
import os

from .coordination import WorkBinding, _label, _sha
from .fitting_inputs import _object, _typed
from .main_qualification_plan import plan_identity, plan
from .main_worker_contract import IMAGE, parse
from .observation_loader import ObservationFile, _path
from .retention import STUDY_ID, identity_sha256
from .training_objective import ObjectiveExecution


GIB=1024**3
SCHEMA="credo.frontier.main-qualification-contract.v1"
REFERENCES=("source_manifest","worker_source_manifest","observation_protocol",
    "normalization_protocol","rules","authority","admission")


def settings():
    return dict(owner_execution=asdict(ObjectiveExecution(32,32,1024,1024,128,2**28,3e-6,3e-6)),
        maximum_geometry_elements=2**23,maximum_geometry_guides=32,
        geomloss_binding="Actual public SamplesLoss backend in the pinned image; verified and recorded",
        maximum_output_bytes=16*GIB,maximum_log_bytes_per_process=4*1024**2,
        worker_sequence_output_bytes=4*GIB,scientific_updates=0)


def exact(value,expected):
    if type(value) is not dict or any(k not in value or identity_sha256(value[k])!=identity_sha256(v)
            for k,v in expected.items()):
        raise ValueError("Main qualification metadata scope or identity differs")


def decode(payload):
    record=parse(payload)
    _object(record,("schema","run_id","phase","study_root","output_root","source_root",
        "reader_source_root","runtime_image","expected_runtime","plan_sha256","settings",*REFERENCES))
    if record["schema"]!=SCHEMA or record["runtime_image"]!=IMAGE:
        raise ValueError("Distinct qualification contract and pinned image required")
    _label(record["run_id"]);_label(record["phase"])
    root=_path(record["study_root"]);phase=root/record["phase"]
    if (root.name!=STUDY_ID or _path(record["output_root"])!=phase/"output"
            or _path(record["source_root"])!=phase/"bundle/source"
            or _path(record["reader_source_root"])!=phase/"bundle/retained_reader"):
        raise ValueError("Exact study-specific phase/bundle/output paths required")
    if (record["plan_sha256"]!=plan_identity()
            or identity_sha256(record["settings"])!=identity_sha256(settings())):
        raise ValueError("Qualification plan or numerical/resource settings changed")
    refs={k:_typed(ObservationFile,record[k]) for k in REFERENCES}
    if (len({r.path for r in refs.values()})!=len(refs)
            or any(r.size_bytes>4*1024**2 or _path(r.path).parent!=phase/"bundle" for r in refs.values())):
        raise ValueError("Distinct bounded bundle metadata references required")
    runtime=record["expected_runtime"]
    if (type(runtime) is not dict or set(runtime)!={"python","torch","cuda_runtime","numpy","scipy"}
            or any(type(v) is not str or not v for v in runtime.values())):
        raise ValueError("Explicit qualified-image runtime versions required")
    return record,refs


def validate_parent(record,refs,contract_reference,authority,admission,release,environment,runtime):
    """Consume actual external approval/reservation/Pod release before work."""
    exact(authority,dict(schema="credo.frontier.main-qualification-authority.v1",status="authorized",
        run_id=record["run_id"],plan_sha256=plan_identity(),settings_sha256=identity_sha256(settings()),
        source_manifest_sha256=refs["source_manifest"].sha256,
        worker_source_manifest_sha256=refs["worker_source_manifest"].sha256,
        planned_disposable_optimizer_calls=111,maximum_disposable_optimizer_calls=112,
        scientific_updates=0,real_biological_payload_access=False,main_scientific_fitting=False,
        automatic_retry=False,automatic_successor=False,elapsed_time_deadline=None))
    expected=dict(schema="credo.frontier.main-qualification-admission.v1",status="reserved",
        run_id=record["run_id"],work_binding=WorkBinding.other("qualification",plan_identity()).to_dict(),
        phase_authority_sha256=refs["authority"].sha256,
        source_manifest_sha256=refs["source_manifest"].sha256,
        worker_source_manifest_sha256=refs["worker_source_manifest"].sha256,
        scientific_update_reservation=0,storage_reserved_bytes=16*GIB,
        disposable_optimizer_call_limit=112,automatic_retry=False,automatic_successor=False)
    exact(admission,expected)
    for key in ("journal_event_sha256","journal_identity_sha256","capacity_receipt_sha256"):
        _sha(admission.get(key))
    total,live=admission.get("aggregate_reserved_output_bytes"),admission.get("live_h100_reservations")
    if (type(total) is not int or not 16*GIB<=total<=384*GIB
            or type(live) is not int or not 1<=live<=4):
        raise ValueError("Aggregate journal resource limits exceeded")
    if (environment.get("CREDO_RUN_ID")!=record["run_id"]
            or environment.get("CREDO_V4_EXECUTION_IMAGE_DIGEST")!=IMAGE.split("@")[1]
            or not environment.get("CREDO_POD_NAME") or not environment.get("CREDO_POD_UID")
            or (os.getuid(),os.getgid())!=(307469,600815) or runtime!=record["expected_runtime"]):
        raise ValueError("Actual parent Pod, writer identity or runtime differs")
    exact(release,dict(schema="credo.frontier.main-qualification-startup-release.v1",
        status="startup_released",run_id=record["run_id"],worker_contract_sha256=contract_reference.sha256,
        pod_name=environment["CREDO_POD_NAME"],pod_uid=environment["CREDO_POD_UID"],image=IMAGE,
        admission_sha256=refs["admission"].sha256,source_manifest_sha256=refs["source_manifest"].sha256,
        phase_authority_sha256=refs["authority"].sha256,
        journal_event_sha256=admission["journal_event_sha256"],
        capacity_receipt_sha256=admission["capacity_receipt_sha256"],
        actual_pod_audit_passed=True,storage_and_fileset_quota_passed=True,
        aggregate_reservation_verified=True,cgroup_resources_verified=True,
        delegate_fixed_worker_cases=True,disposable_qualification_authorized=True,
        scientific_fit_authorized=False,real_biological_payload_access=False,
        authorized_cpus=24,authorized_memory_bytes=200*GIB,threads=8,gpus=1,
        automatic_retry=False,automatic_successor=False))
    for key in ("startup_sha256",): _sha(release.get(key))
    if type(release.get("job_uid")) is not str or not release["job_uid"]:
        raise ValueError("Actual scheduler Job UID required")
    if plan()["planned_optimizer_calls"]!=111:
        raise ValueError("Qualification workload differs from explicit parent authority")
