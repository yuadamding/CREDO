"""File-backed disposable worker sequence inside an already admitted GPU job.

The enclosing controller must issue each contract/admission/startup release.
This module cannot grant authority, submit a Job or certify scientific fitting.
It runs the fixed three cases in fresh processes on the same assigned GPU,
checks their publications, and preserves a failed attempt without retrying.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .fitting_inputs import FittingInputContract
from .bounded_process import run_process
from .fitting_resources import FittingResourcePolicy, output_usage
from .main_qualification_artifacts import file_reference, write_json, stage_task
from .main_qualification_custody import publish_custody
from .main_qualification_readback import verify_publication
from .main_qualification_worker_plan import CONTRACT_SCHEMA, SCOPE, cases, identity
from .main_worker_contract import IMAGE, MAXIMUM_BYTES, decode_contract, parse
from .observation_loader import ObservationFile, _path, _verified_file
from .retention import STUDY_ID


GIB, MIB = 1024**3, 1024**2
SEQUENCE_OUTPUT_BYTES = 4*GIB  # Suballocation of the proposed 16-GiB parent.


def _read(reference):
    with _verified_file(reference, maximum=MAXIMUM_BYTES) as stream:
        return parse(stream.read())


def publish_request(root, case, *, parent_id, source_root, source_manifest,
                    input_reference, resume=None, continuation_admissions=()):
    """Create metadata for the external admission owner, never a launch release."""
    root, source_root = _path(root), _path(source_root)
    if (case not in cases() or root.name != STUDY_ID or root.parent.name != parent_id
            or root.parent.parent.name != "qualification-fixtures"
            or source_root == root or not source_root.is_relative_to(root)):
        raise ValueError("Exact isolated qualification root/case/source required")
    for ref in (source_manifest,input_reference):
        if type(ref) is not ObservationFile:
            raise TypeError("Exact immutable source/input references required")
    inputs = FittingInputContract(json.dumps(_read(input_reference)).encode())
    plan, prior, task = stage_task(case)
    iparts, irecord = inputs.components(), inputs.record()
    if (iparts["task"] != task or not iparts["native_binding"].synthetic_fixture
            or irecord["source_sha256"] != source_manifest.sha256):
        raise ValueError("Actual fabricated input does not bind this task/source")
    worker_id = parent_id+"-worker-"+case.stage
    request_root = root/"worker_requests"/case.stage
    request_root.mkdir()  # No reuse of an already attempted case.
    permit = write_json(request_root/"PERMIT.json",dict(
        schema="credo.frontier.main-worker-fixture-permit.v1",
        status="disposable_execution_permitted_not_qualified",case=asdict(case),
        worker_plan_sha256=identity(),source_manifest_sha256=source_manifest.sha256,
        input_reference=asdict(input_reference),scientific_updates=0,
        real_biological_payload_access=False,runtime_image=IMAGE))
    policy = FittingResourcePolicy(SCOPE,GIB,64*MIB,256*MIB,8*MIB,200*GIB,8*GIB)
    record = dict(schema=CONTRACT_SCHEMA,qualification_parent_id=parent_id,worker_id=worker_id,
        study_root=str(root),receipt_root=str(root/"worker_runs"/worker_id),source_root=str(source_root),
        source_manifest=asdict(source_manifest),input_reference=asdict(input_reference),
        qualification=asdict(permit),plan=asdict(plan),task_id=task.identifier,
        previous_stage=None if prior is None else asdict(prior),resume=resume,
        continuation_admissions=[asdict(a) for a in continuation_admissions],
        resource_policy=asdict(policy),pin_memory=False,runtime_image=IMAGE)
    # A placeholder checks structural validity only and is NOT published as an
    # admission. The enclosing controller must supply the actual reference.
    placeholder=ObservationFile(str(request_root/"UNISSUED_ADMISSION.json"),"0"*64,1)
    decode_contract(json.dumps(dict(record,admission=asdict(placeholder))).encode(),qualification=True)
    return write_json(request_root/"REQUEST.json",dict(
        schema="credo.frontier.main-worker-qualification-request.v1",
        status="requires_external_parent_admission_and_release",contract_fields=record,
        source_publication_verified_here=False,launch_authority=False,optimizer_calls=0))


def run_released_case(request_reference, contract_reference, release_reference):
    """One fresh worker process, no timeout, fallback or retry.

    The worker independently checks real Pod/image/source/runtime/release and
    opens CUDA only after those checks. Metadata inspection alone is no pass.
    The parent must retain aggregate capacity and attempted-call accounting.
    """
    request = _read(request_reference)
    if (request.get("schema") != "credo.frontier.main-worker-qualification-request.v1"
            or request.get("launch_authority") is not False):
        raise ValueError("Exact nonauthorizing qualification request required")
    record, parts = decode_contract(json.dumps(_read(contract_reference)).encode(),qualification=True)
    if {k:v for k,v in record.items() if k!="admission"} != request["contract_fields"]:
        raise ValueError("Issued contract differs from the exact requested case")
    _read(release_reference)  # Byte lease/readback; semantic admission is the worker's job.
    if (os.environ.get("CREDO_RUN_ID") != record["qualification_parent_id"]
            or os.environ.get("CREDO_V4_EXECUTION_IMAGE_DIGEST") != IMAGE.split("@")[1]):
        raise ValueError("Only the actually admitted parent Pod may dispatch a CUDA case")
    folder = Path(request_reference.path).parent
    write_json(folder/"ATTEMPT.json",dict(contract=asdict(contract_reference),
        release=asdict(release_reference),case=asdict(parts["qualification_case"]),
        disposable_call_upper_bound=parts["qualification_case"].optimizer_calls,
        scientific_updates=0,automatic_retry=False))
    command = [sys.executable,"-m","frontier.main_worker","run-qualification",
        "--contract",contract_reference.path,"--contract-sha256",contract_reference.sha256,
        "--release",release_reference.path,"--release-sha256",release_reference.sha256]
    env=dict(os.environ,PYTHONPATH=record["source_root"],PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",MKL_NUM_THREADS="8",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    returncode=run_process(command,cwd=record["source_root"],env=env,log_path=folder/"PROCESS.log")
    write_json(folder/"PROCESS_EXIT.json",dict(returncode=returncode,automatic_retry=False,
        actual_optimizer_calls_inferred_from_exit=False))
    if returncode:
        raise RuntimeError("Disposable worker failed; preserve its receipts/logs and do not retry")
    complete=file_reference(Path(record["receipt_root"])/"COMPLETE.json")
    result=verify_publication(contract_reference,complete)
    return write_json(folder/"READBACK.json",result)


def run_sequence(root, *, parent_id, source_root, source_manifest, inputs, admit_case):
    """Consume a parent-owned admission callback; never fabricate its authority.

    ``admit_case(request_ref)`` returns (contract_ref, release_ref) after the
    enclosing controller's actual resource/source/parent-authority checks. Only
    the predeclared same-GPU child processes are started, not new scheduler jobs.
    """
    root=_path(root)
    if set(inputs)!={"screen","family","outer"} or not callable(admit_case):
        raise ValueError("All three exact input references and an external admission owner required")
    if (root.name!=STUDY_ID or root.parent.name!=parent_id
            or root.parent.parent.name!="qualification-fixtures"):
        raise ValueError("Exact isolated parent qualification root required")
    sequence=root/"worker_sequence"; sequence.mkdir()
    for name in ("worker_requests","worker_runs","trajectories"):
        path=root/name
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError("Qualification sequence cannot adopt previous attempts")
        path.mkdir(exist_ok=True)
    results=[]; attempted_upper_bound=0
    try:
        resume=None; custody=()
        for case in cases():
            if output_usage(root)["output_bytes"]+GIB>SEQUENCE_OUTPUT_BYTES:
                raise RuntimeError("Worker sequence lacks its bounded remaining output allowance")
            if case.accepted_start:
                prior=_read(results[-1])["checkpoint_complete"]
                custody,_=publish_custody(root/"opaque-custody",checkpoint_complete=ObservationFile(**prior))
                cp=_read(ObservationFile(**prior))
                resume=dict(directory=str(Path(prior["path"]).parent),**cp)
            else:
                resume=None;custody=()
            request=publish_request(root,case,parent_id=parent_id,source_root=source_root,
                source_manifest=source_manifest,input_reference=inputs[case.stage],
                resume=resume,continuation_admissions=custody)
            contract,release=admit_case(request)
            attempted_upper_bound+=case.optimizer_calls
            readback=run_released_case(request,contract,release)
            results.append(readback)
        if attempted_upper_bound!=7 or output_usage(root)["output_bytes"]>SEQUENCE_OUTPUT_BYTES:
            raise ValueError("Worker sequence workload or output accounting changed")
        return write_json(sequence/"COMPLETE.json",dict(
            schema="credo.frontier.main-qualification-worker-sequence.v1",
            status="three_worker_publications_verified",readbacks=[asdict(r) for r in results],
            disposable_optimizer_calls_reported=7,scientific_updates=0,worker_plan_sha256=identity(),
            same_assigned_gpu_sequential_workers=True,checkpoint_continuation_case_included=True,
            full_shape_owner_cases_verified_here=False,paired_precision_verified_here=False,
            full_main_qualification_pass=False,scientific_launch_authority=False,automatic_retry=False))
    except BaseException as error:
        write_json(sequence/"FAILED.json",dict(exception_type=type(error).__name__,
            completed_case_readbacks=[asdict(r) for r in results],
            attempted_disposable_call_upper_bound=attempted_upper_bound,
            actual_failed_case_call_count_requires_worker_receipts=True,
            scientific_updates=0,automatic_retry=False))
        raise
