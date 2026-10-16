"""Production main-fit CLI consuming, never creating, launch authority.

The separately bound supervisor supplies a hash-identified startup release
after actual Pod, quota, reservation, selection and qualification readback.
This executable performs no scheduler operation, retry, selection, preparation,
forecast or successor launch. No CPU fallback exists. A main-worker CUDA
qualification and controller implementation are still separate dependencies.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

from .coordination import WorkBinding, _sha
from .fitting_execution import run_admitted_fitting_task
from .fitting_inputs import FittingInputContract, _encoded
from .main_worker_contract import (IMAGE, MAXIMUM_BYTES, decode_contract, parse,
                                   qualification_shape)
from .main_worker_source import verified_sources
from .observation_loader import ObservationFile, _path, _verified_file
from .observation_worker import (_atomic_json, _configure_cuda, _qualification_hardware_compatibility,
                                 _read, _runtime, _utc)
from .retention import identity_sha256


GIB = 1024**3


def qualification_mode(parts):
    return parts.get("qualification_case") is not None


def qualification_binding():
    from .main_qualification_plan import plan_identity
    return WorkBinding.other("qualification", plan_identity())


def publish_metadata(output, name, value, allowance):
    """Charge worker receipts to the trajectory guard's reserved metadata bytes."""
    size = len(json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode()) + 1
    used = 0
    for path in output.iterdir():
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Only this worker's regular receipt files are admitted")
        used += path.stat().st_size
    if size + used > allowance:
        raise RuntimeError("Worker receipts exceed the reserved metadata allowance")
    return _atomic_json(output, name, value)


def exact_fields(record, expected):
    """Type-sensitive subset comparison: Boolean pass flags are not integers."""
    if type(record) is not dict or any(k not in record or _encoded(record[k]) != _encoded(v)
                                      for k, v in expected.items()):
        raise ValueError("Required main-worker evidence identity/scope differs")


def validate_admission(record, parts, admission):
    qualifying = qualification_mode(parts)
    expected = dict(schema="credo.frontier.main-fitting-admission.v1", run_id=record["worker_id"],
        status="reserved", work_binding=parts["work_binding"].to_dict(),
        stage_plan_sha256=identity_sha256(asdict(parts["plan"])),
        source_manifest_sha256=parts["source_manifest"].sha256,
        input_reference=asdict(parts["input_reference"]),
        resource_policy_sha256=parts["resource_policy"].identity,
        qualification_sha256=parts["qualification"].sha256,
        scientific_update_reservation=parts["task"].additional_updates,
        storage_reserved_bytes=parts["resource_policy"].output_limit_bytes,
        automatic_retry=False, automatic_successor=False)
    if qualifying:
        expected.update(schema="credo.frontier.main-worker-qualification-admission.v1",
            parent_run_id=record["qualification_parent_id"],
            work_binding=qualification_binding().to_dict(), scientific_update_reservation=0,
            disposable_optimizer_calls=parts["qualification_case"].optimizer_calls,
            case=asdict(parts["qualification_case"]), real_biological_payload_access=False)
    exact_fields(admission, expected)
    for k in ("journal_event_sha256", "journal_identity_sha256", "phase_authority_sha256", "capacity_receipt_sha256"):
        _sha(admission.get(k))
    total, live = admission.get("aggregate_reserved_output_bytes"), admission.get("live_h100_reservations")
    if (type(total) is not int or not expected["storage_reserved_bytes"] <= total <= 384*GIB
            or type(live) is not int or not 1 <= live <= 4):
        raise ValueError("Main worker exceeds aggregate reservation/resource bounds")
    selections = admission.get("selection_evidence")
    if type(selections) is not list or len(selections) != (0 if qualifying or parts["plan"].name == "screen" else 4):
        raise ValueError("Exact prior selection evidence required for promoted/outer fitting")
    refs = tuple(ObservationFile(**r) for r in selections)
    if len({r.path for r in refs}) != len(refs) or sum(r.size_bytes for r in refs) > MAXIMUM_BYTES:
        raise ValueError("Distinct bounded selection evidence required")
    return refs


def validate_release(record, parts, contract_ref, release, admission, environment):
    qualifying = qualification_mode(parts)
    pod_uid = environment.get("CREDO_POD_UID")
    pod_name = environment.get("CREDO_POD_NAME")
    expected_run = record["qualification_parent_id"] if qualifying else record["worker_id"]
    if not pod_uid or not pod_name or environment.get("CREDO_RUN_ID") != expected_run:
        raise ValueError("Actual named Pod identity required")
    if environment.get("CREDO_V4_EXECUTION_IMAGE_DIGEST") != IMAGE.split("@")[1]:
        raise ValueError("Worker image differs from approved pinned image")
    expected = dict(schema="credo.frontier.main-fitting-startup-release.v1",
        status="startup_released", run_id=record["worker_id"], pod_uid=pod_uid, pod_name=pod_name,
        worker_contract_sha256=contract_ref.sha256, admission_sha256=parts["admission"].sha256,
        source_manifest_sha256=parts["source_manifest"].sha256,
        qualification_sha256=parts["qualification"].sha256,
        journal_event_sha256=admission["journal_event_sha256"],
        image=IMAGE, actual_pod_audit_passed=True, storage_and_fileset_quota_passed=True,
        aggregate_reservation_verified=True, cgroup_resources_verified=True,
        qualification_terminal_readback_verified=True, selection_evidence_verified=True,
        scientific_fit_authorized=True, automatic_retry=False, automatic_successor=False,
        authorized_cpus=24, authorized_memory_bytes=200*GIB, threads=8, gpus=1,
        worker_uid=os.getuid(), worker_gid=os.getgid())
    if qualifying:
        expected.update(schema="credo.frontier.main-worker-qualification-startup-release.v1",
            parent_run_id=expected_run, scientific_fit_authorized=False,
            qualification_terminal_readback_verified=False,
            disposable_qualification_authorized=True, real_biological_payload_access=False,
            disposable_optimizer_calls=parts["qualification_case"].optimizer_calls,
            parent_qualification_authority_verified=True)
    exact_fields(release, expected)
    if (os.getuid(), os.getgid()) != (307469, 600815):
        raise ValueError("Main worker must use the approved institutional writer identity")
    for k in ("job_uid", "startup_sha256", "capacity_receipt_sha256", "phase_authority_sha256"):
        if k.endswith("sha256"):
            _sha(release.get(k))
        elif type(release.get(k)) is not str or not release[k]:
            raise ValueError("Actual scheduler Job UID required")
    exact_fields(release, {k: admission[k] for k in ("capacity_receipt_sha256", "phase_authority_sha256")})


def validate_fixture_permit(parts, inputs, permit):
    """Consume a disposable-test permit, never a fabricated 'passed' result."""
    from .main_qualification_worker_plan import worker_schedule, identity
    case = parts["qualification_case"]
    exact_fields(permit, dict(schema="credo.frontier.main-worker-fixture-permit.v1",
        status="disposable_execution_permitted_not_qualified", case=asdict(case),
        worker_plan_sha256=identity(), source_manifest_sha256=parts["source_manifest"].sha256,
        input_reference=asdict(parts["input_reference"]), scientific_updates=0,
        real_biological_payload_access=False, runtime_image=IMAGE))
    iparts = inputs.components()
    if (not iparts["native_binding"].synthetic_fixture
            or not iparts["observation"].specification.synthetic_fixture
            or iparts["schedule"] != worker_schedule()
            or iparts["task"] != parts["task"]):
        raise ValueError("Worker qualification may consume only the exact fabricated input/schedule")
    return dict(status="disposable_test_not_qualification_pass", case=asdict(case),
        worker_plan_sha256=identity(), scientific_fit_authorized=False)


def validate_qualification(record, parts, inputs, qualification, hardware, runtime):
    # Do not accept the completed observation-model qualification under a new
    # label or infer qualification from a syntactically valid shape hash.
    exact_fields(qualification, dict(schema="credo.frontier.main-worker-cuda-qualification.v1",
        status="passed", source_manifest_sha256=parts["source_manifest"].sha256,
        runtime_image=IMAGE, runtime=runtime, scientific_updates=0,
        real_biological_payload_access=False, full_worker_entry_verified=True,
        checkpoint_resume_verified=True, publication_readback_verified=True))
    shape = qualification_shape(inputs, parts["resource_policy"], record["pin_memory"])
    cases = qualification.get("qualified_case_sha256s")
    if type(cases) is not list or not cases or len(cases) != len(set(cases)):
        raise ValueError("Actual main-worker qualified case inventory required")
    for case in cases:
        _sha(case)
    if shape not in cases:
        raise ValueError("This exact numerical shape is not CUDA qualified")
    compatibility = _qualification_hardware_compatibility(qualification, hardware)
    return dict(case_sha256=shape, hardware_compatibility=compatibility)


def execute(contract_ref, release_ref, *, qualification=False):
    """One foreground scientific task; no launch or implicit continuation."""
    output, claimed = None, False
    try:
        with ExitStack() as leases:
            def metadata(ref):
                return parse(leases.enter_context(_verified_file(ref, maximum=MAXIMUM_BYTES)).read())
            raw = leases.enter_context(_verified_file(contract_ref, maximum=MAXIMUM_BYTES)).read()
            record, parts = decode_contract(raw, qualification=qualification)
            output = _path(record["receipt_root"])
            if output.exists():
                raise FileExistsError("Worker receipt root already attempted; no retry")
            if not output.parent.is_dir() or output.parent.stat().st_uid != os.getuid():
                raise ValueError("Prepared owned worker receipt parent required")
            leases.enter_context(verified_sources(record["source_root"], parts["source_manifest"]))
            admission, release = metadata(parts["admission"]), metadata(release_ref)
            selections = validate_admission(record, parts, admission)
            validate_release(record, parts, contract_ref, release, admission, os.environ)
            for ref in selections:
                # The controller verifies the scientific selection semantics;
                # the worker retains those exact byte identities through exit.
                metadata(ref)
            qualification_record = metadata(parts["qualification"])
            input_raw = leases.enter_context(_verified_file(parts["input_reference"], maximum=MAXIMUM_BYTES)).read()
            inputs = FittingInputContract(input_raw)
            irecord, iparts = inputs.record(), inputs.components()
            if (iparts["task"] != parts["task"] or irecord["source_sha256"] != parts["source_manifest"].sha256
                    or irecord["maximum_checkpoint_bytes"] != parts["resource_policy"].checkpoint_limit_bytes):
                raise ValueError("Input contract differs from admitted main task/source/checkpoint size")
            hardware, runtime = _configure_cuda(), _runtime()
            qualified = (validate_fixture_permit(parts, inputs, qualification_record) if qualification else
                validate_qualification(record, parts, inputs, qualification_record, hardware, runtime))
            output.mkdir(mode=0o700)  # Atomic claim; never adopt a competing path.
            claimed = True
            started = publish_metadata(output, "STARTED.json", dict(schema="credo.frontier.main-worker-started.v1",
                started_at_utc=_utc(), worker_id=record["worker_id"], contract=asdict(contract_ref),
                startup_release=asdict(release_ref), qualification=asdict(parts["qualification"]),
                qualification_compatibility=qualified, hardware=hardware, runtime=runtime,
                scientific_updates=0, source_runtime_and_release_verified=True,
                execution_scope=parts["resource_policy"].execution_scope,
                disposable_worker_qualification=qualification),
                parts["resource_policy"].metadata_reserve_bytes)
            result = run_admitted_fitting_task(parts["input_reference"], plan=parts["plan"], task=parts["task"],
                work_binding=parts["work_binding"], source_sha256=parts["source_manifest"].sha256,
                study_root=record["study_root"], worker_id=record["worker_id"],
                resource_policy=parts["resource_policy"], device="cuda:0", pin_memory=record["pin_memory"],
                resume=record["resume"], previous_stage=parts["previous_stage"] if parts["task"].start_updates else None,
                continuation_admissions=parts["continuation_admissions"])
            case = parts["qualification_case"]
            exact_fields(result, dict(status="complete_after_input_stage_and_lease_exit",
                task=asdict(parts["task"]), work_binding=parts["work_binding"].to_dict(),
                source_sha256=parts["source_manifest"].sha256,
                input_contract=asdict(parts["input_reference"]), execution_scope=parts["resource_policy"].execution_scope,
                accepted_start=case.accepted_start if case else parts["task"].start_updates,
                accepted_target=case.accepted_target if case else parts["task"].target_updates,
                scientific_updates_this_task=0 if case else parts["task"].additional_updates,
                optimizer_calls_this_task=case.optimizer_calls if case else parts["task"].additional_updates,
                resource_policy_sha256=parts["resource_policy"].identity,
                full_input_exit_verified=True, automatic_successor=False))
        # In particular, failure of source/release/input metadata exit prevents
        # worker completion, even if the nested owner already published a stage.
        return publish_metadata(output, "COMPLETE.json", dict(schema="credo.frontier.main-worker-completion.v1",
            status="complete", completed_at_utc=_utc(), started=started, execution=result,
            all_worker_input_and_source_exits_verified=True, automatic_successor=False,
            disposable_worker_qualification=qualification, scientific_fit=not qualification),
            parts["resource_policy"].metadata_reserve_bytes)
    except BaseException as error:
        # Do not write into an existing/competing root we did not claim.
        if claimed:
            publish_metadata(output, "FAILED.json", dict(schema="credo.frontier.main-worker-failed.v1",
                recorded_at_utc=_utc(), exception_type=type(error).__name__,
                contract=asdict(contract_ref), automatic_retry=False, automatic_successor=False),
                parts["resource_policy"].metadata_reserve_bytes)
        raise


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "run", "inspect-qualification", "run-qualification"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--release")
    parser.add_argument("--release-sha256")
    args = parser.parse_args(argv)
    raw = _read(args.contract, args.contract_sha256, maximum_bytes=MAXIMUM_BYTES)
    ref = ObservationFile(str(_path(args.contract)), args.contract_sha256, len(raw))
    qualifying = args.mode.endswith("-qualification")
    record, parts = decode_contract(raw, qualification=qualifying)
    if args.mode in ("inspect", "inspect-qualification"):
        if args.release is not None or args.release_sha256 is not None:
            parser.error("Metadata inspection cannot consume a startup release")
        print(json.dumps(dict(status="metadata_valid_not_launch_authority", task_id=record["task_id"],
            stage=parts["plan"].name, scientific_updates_requested=0 if qualifying else parts["task"].additional_updates,
            disposable_optimizer_calls_requested=parts["qualification_case"].optimizer_calls if qualifying else 0,
            source_manifest=asdict(parts["source_manifest"]), numerical_payloads_opened=False,
            optimizer_calls=0), sort_keys=True))
        return 0
    if args.release is None or args.release_sha256 is None:
        parser.error("Scientific execution requires an exact controller startup release")
    release_raw = _read(args.release, args.release_sha256, maximum_bytes=MAXIMUM_BYTES)
    result = execute(ref, ObservationFile(str(_path(args.release)), args.release_sha256, len(release_raw)),
        qualification=qualifying)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
