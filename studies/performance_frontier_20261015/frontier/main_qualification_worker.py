"""One externally admitted main qualification, never scientific launch authority.

Precision and full-shape numerical checks run in one fresh subprocess. It exits
before the file-backed worker sequence starts, releasing its CUDA context. All
children use the one allocated GPU, sequentially, without retries or deadlines.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import sys

import torch

from .bounded_process import run_process
from .fitting_resources import output_usage
from .geometry import GeometryOptions, geomloss_backend
from .main_qualification_artifacts import file_reference, prepare_artifacts, write_json
from .main_qualification_contract import decode, validate_parent, settings, exact
from .main_qualification_dispatch import issue_case
from .main_qualification_owner import DisposableBudget, run_owner_case
from .main_qualification_plan import cases, plan_identity
from .main_qualification_precision import PHASES, run_paired_precision_case
from .main_qualification_sequence import run_sequence
from .main_qualification_terminal import verify as verify_terminal
from .main_worker_contract import MAXIMUM_BYTES, parse
from .main_worker_source import (QUALIFICATION_ENTRY, verified_sources, source_closure,
                                 manifest_payload, MAXIMUM_SOURCE_BYTES)
from .native_inputs import READER_SOURCE_HASHES
from .objective_registry import ObjectiveRules
from .observation_loader import ObservationFile, _path, _verified_file
from .observation_worker import _configure_cuda, _runtime, _resources, _read, _utc
from .retention import STUDY_ID
from .training_objective import ObjectiveExecution


GIB, MIB = 1024**3, 1024**2


def capacity(root, reserve=8*MIB):
    used = output_usage(root)
    if (used["output_bytes"] + reserve > settings()["maximum_output_bytes"]
            or used["filesystem_available_bytes"] < reserve):
        raise RuntimeError("Main qualification output/headroom exhausted; no retry")
    return used


def _metadata(ref):
    with _verified_file(ref, maximum=MAXIMUM_BYTES) as stream:
        return parse(stream.read())


def run_numerics(root, *, references, payloads):
    """Requires the validated external parent admission; no standalone fallback."""
    folder = root/"numerics"
    folder.mkdir()  # A failed attempt must never be adopted or restarted.
    budget = DisposableBudget()
    precision, owners = [], []
    try:
        hardware = _configure_cuda()
        rules = ObjectiveRules(payloads["rules"], references["rules"].sha256)
        backend = geomloss_backend()
        fixed = settings()
        geometry = GeometryOptions(fixed["maximum_geometry_elements"],
            fixed["maximum_geometry_guides"], backend.identity["version"])
        execution = ObjectiveExecution(**fixed["owner_execution"])
        write_json(folder/"STARTED.json", dict(hardware=hardware, runtime=_runtime(),
            backend=backend.identity, plan_sha256=plan_identity(), scientific_updates=0,
            started_at_utc=_utc(), automatic_retry=False))
        for case in cases():
            for phase in PHASES:
                capacity(root, 8*MIB)
                _resources("cuda:0")
                result = run_paired_precision_case(case, phase, rules=rules, geometry_options=geometry,
                    source_sha256=references["source_manifest"].sha256)
                ref = write_json(folder/(case.identifier+"-"+phase+"-precision.json"), result)
                precision.append(ref)
                if result["status"] != "passed":
                    raise RuntimeError("Fixed paired-precision tolerance failed; stop remaining cases without retry")
            capacity(root, 512*MIB)
            run_owner_case(case, folder/case.identifier, budget=budget, rules=rules,
                execution=execution, geometry_options=geometry,
                source_sha256=references["source_manifest"].sha256)
            owners.append(file_reference(folder/case.identifier/"OWNER_CASE_COMPLETE.json"))
            _resources("cuda:0")
            capacity(root)
        if (budget.calls != 104 or budget.failed or len(precision) != 96 or len(owners) != 48
                or budget.completed_cases != {c.identifier for c in cases()}):
            raise ValueError("Main numerical case inventory or disposable accounting differs")
        return write_json(folder/"COMPLETE.json", dict(
            schema="credo.frontier.main-qualification-numerics.v1", status="numerical_components_passed",
            plan_sha256=plan_identity(), source_manifest_sha256=references["source_manifest"].sha256,
            precision=[asdict(r) for r in precision], owners=[asdict(r) for r in owners],
            disposable_optimizer_calls=budget.calls, scientific_updates=0,
            hardware=hardware, runtime=_runtime(), backend=backend.identity, resources=_resources("cuda:0"),
            real_biological_payload_access=False, scientific_launch_authority=False,
            full_worker_entry_verified=False, automatic_retry=False))
    except BaseException as error:
        write_json(folder/"FAILED.json", dict(exception_type=type(error).__name__,
            actual_owner_optimizer_calls=budget.calls, precision_records=[asdict(r) for r in precision],
            completed_owner_records=[asdict(r) for r in owners], scientific_updates=0,
            automatic_retry=False, scientific_launch_authority=False))
        raise


def prepare_sequence(record, references, payloads):
    """Generate small fabricated inputs; do not retain a parent CUDA context."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Parent must not hold a CUDA context during sequential worker dispatch")
    torch.set_num_threads(2)
    root = _path(record["output_root"])/"qualification-fixtures"/record["run_id"]/STUDY_ID
    root.mkdir(parents=True)
    source = source_closure(_path(record["source_root"]))
    manifest = manifest_payload(source)
    if manifest != payloads["worker_source_manifest"]:
        raise ValueError("Default worker closure differs from its independently bound source manifest")
    source_root = root/"source"
    for name, raw in source.items():
        path = source_root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        with _verified_file(ObservationFile(str(path), sha256(raw).hexdigest(), len(raw)),
                            maximum=MAXIMUM_SOURCE_BYTES):
            pass
    with (root/"SOURCE_MANIFEST.json").open("xb") as stream:
        stream.write(manifest); stream.flush(); os.fsync(stream.fileno())
    manifest_ref = file_reference(root/"SOURCE_MANIFEST.json")
    inputs = {}
    for donors in (("D3", "D4"), ("D2", "D3", "D4")):
        ref = prepare_artifacts(root/("fixture-"+"-".join(donors)), donors=donors,
            reader_source_root=record["reader_source_root"], observation_protocol=payloads["observation_protocol"],
            normalization_protocol=payloads["normalization_protocol"], rules_reference=references["rules"],
            source_sha256=manifest_ref.sha256)
        inputs.update({k: ObservationFile(**v) for k,v in _metadata(ref)["inputs"].items()})
    return root, source_root, manifest_ref, inputs


def execute(contract_ref, release_ref, *, numerical_child=False):
    """Fresh immutable parent attempt, or its explicitly requested subprocess."""
    claimed, root = False, None
    try:
        with ExitStack() as leases:
            raw = leases.enter_context(_verified_file(contract_ref, maximum=MAXIMUM_BYTES)).read()
            record, references = decode(raw)
            root = _path(record["output_root"])
            leases.enter_context(verified_sources(record["source_root"], references["source_manifest"],
                                                 entry=QUALIFICATION_ENTRY))
            payloads = {k: leases.enter_context(_verified_file(ref, maximum=MAXIMUM_BYTES)).read()
                        for k, ref in references.items()}
            release = parse(leases.enter_context(_verified_file(release_ref, maximum=MAXIMUM_BYTES)).read())
            authority, admission = parse(payloads["authority"]), parse(payloads["admission"])
            validate_parent(record, references, contract_ref, authority, admission, release, os.environ, _runtime())
            if manifest_payload(source_closure(_path(record["source_root"]))) != payloads["worker_source_manifest"]:
                raise ValueError("Default worker source closure differs before numerical admission")
            # Lease only the four retained code files, never historical data.
            for name, digest in READER_SOURCE_HASHES.items():
                path = _path(record["reader_source_root"])/name
                leases.enter_context(_verified_file(ObservationFile(str(path), digest, path.stat().st_size),
                                                    maximum=2*MIB))
            if numerical_child:
                started = _metadata(file_reference(root/"STARTED.json"))
                exact(started, dict(contract=asdict(contract_ref), release=asdict(release_ref),
                    pid=os.getppid(), run_id=record["run_id"], scientific_updates=0))
                if (root/"FAILED.json").exists() or (root/"COMPLETE.json").exists():
                    raise ValueError("Parent already terminal; numerical child is not admitted")
                result = run_numerics(root, references=references, payloads=payloads)
            else:
                if root.exists() or not root.parent.is_dir() or root.parent.stat().st_uid != os.getuid():
                    raise ValueError("Absent output under exact owned prepared parent required")
                root.mkdir(mode=0o700); claimed = True
                capacity(root, 16*GIB)
                write_json(root/"STARTED.json", dict(schema="credo.frontier.main-qualification-started.v1",
                    run_id=record["run_id"], contract=asdict(contract_ref), release=asdict(release_ref),
                    pid=os.getpid(), started_at_utc=_utc(), scientific_updates=0, automatic_retry=False))
                command = [sys.executable, "-m", "frontier.main_qualification_worker", "numerics",
                    "--contract", contract_ref.path, "--contract-sha256", contract_ref.sha256,
                    "--release", release_ref.path, "--release-sha256", release_ref.sha256]
                env = dict(os.environ, PYTHONPATH=record["source_root"], PYTHONNOUSERSITE="1",
                    OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8")
                code = run_process(command, cwd=record["source_root"], env=env,
                                   log_path=root/"NUMERICS.log")
                write_json(root/"NUMERICS_EXIT.json", dict(returncode=code, automatic_retry=False))
                if code:
                    raise RuntimeError("Numerical child failed; no worker sequence or retry")
                numerical_ref = file_reference(root/"numerics/COMPLETE.json")
                numerical = _metadata(numerical_ref)
                exact(numerical, dict(schema="credo.frontier.main-qualification-numerics.v1",
                    status="numerical_components_passed", plan_sha256=plan_identity(),
                    source_manifest_sha256=references["source_manifest"].sha256,
                    disposable_optimizer_calls=104, scientific_updates=0, runtime=record["expected_runtime"]))
                if len(numerical["precision"]) != 96 or len(numerical["owners"]) != 48:
                    raise ValueError("Numerical child omitted required cases")
                capacity(root, 4*GIB)
                fixture, source_root, source_manifest, inputs = prepare_sequence(record, references, payloads)
                def admit(request):
                    capacity(root)
                    return issue_case(request, parent=record, references=references,
                                      admission=admission, release=release)
                sequence = run_sequence(fixture, parent_id=record["run_id"], source_root=source_root,
                    source_manifest=source_manifest, inputs=inputs, admit_case=admit)
                exact(_metadata(sequence), dict(status="three_worker_publications_verified",
                    disposable_optimizer_calls_reported=7, scientific_updates=0))
                capacity(root)
                result = dict(schema="credo.frontier.main-qualification-execution.v1",
                    status="components_complete_requires_independent_terminal_readback",
                    contract=asdict(contract_ref), release=asdict(release_ref), numerics=asdict(numerical_ref),
                    worker_sequence=asdict(sequence), source_manifest_sha256=references["source_manifest"].sha256,
                    worker_source_manifest_sha256=references["worker_source_manifest"].sha256,
                    reported_disposable_optimizer_calls=111, maximum_disposable_optimizer_calls=112,
                    scientific_updates=0, real_biological_payload_access=False, scientific_launch_authority=False,
                    main_worker_qualification_pass=False, automatic_retry=False, automatic_successor=False)
        # Exit all source/input leases before claiming parent execution complete.
        if numerical_child:
            # A lease-exit failure still makes the subprocess fail; the parent
            # checks exit status before consuming this component's publication.
            return result
        capacity(root)
        result.update(all_parent_source_and_metadata_exits_verified=True, completed_at_utc=_utc())
        eref=write_json(root/"EXECUTION_COMPLETE.json",result)
        capacity(root,16*MIB)  # Reader payload, bounded log, exit and final receipts.
        command=[sys.executable,"-m","frontier.main_qualification_worker","readback",
            "--contract",contract_ref.path,"--contract-sha256",contract_ref.sha256,
            "--release",release_ref.path,"--release-sha256",release_ref.sha256,
            "--execution",eref.path,"--execution-sha256",eref.sha256]
        code=run_process(command,cwd=record["source_root"],env=dict(env,
            OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2"),log_path=root/"READBACK.log")
        write_json(root/"READBACK_EXIT.json",dict(returncode=code,automatic_retry=False))
        if code:raise RuntimeError("Independent terminal readback failed; no final completion or retry")
        readback_ref=file_reference(root/"TERMINAL_READBACK.json")
        exact(_metadata(readback_ref),dict(schema="credo.frontier.main-qualification-terminal-readback.v1",
            status="complete_numerical_publication_inventory_verified",contract=asdict(contract_ref),
            execution=asdict(eref),disposable_optimizer_calls_reported=111,scientific_updates=0,
            scheduler_terminal_verified=False,main_worker_qualification_pass=False,scientific_launch_authority=False))
        capacity(root)
        return write_json(root/"COMPLETE.json",dict(result,
            status="components_complete_after_independent_terminal_readback",execution_record=asdict(eref),
            terminal_readback=asdict(readback_ref)))
    except BaseException as error:
        if claimed:
            write_json(root/"FAILED.json", dict(exception_type=type(error).__name__,
                recorded_at_utc=_utc(), scientific_updates=0, automatic_retry=False,
                actual_disposable_calls_require_component_receipts=True, scientific_launch_authority=False))
        raise


def execute_readback(contract_ref,release_ref,execution_ref):
    """Fresh CPU-only child of the exact released parent; no standalone run."""
    if torch.cuda.is_initialized():raise RuntimeError("Terminal reader must remain CPU-only")
    torch.set_num_threads(2)
    with ExitStack() as leases:
        raw=leases.enter_context(_verified_file(contract_ref,maximum=MAXIMUM_BYTES)).read()
        record,refs=decode(raw)
        root=_path(record["output_root"])
        leases.enter_context(verified_sources(record["source_root"],refs["source_manifest"],entry=QUALIFICATION_ENTRY))
        values={k:parse(leases.enter_context(_verified_file(ref,maximum=MAXIMUM_BYTES)).read())
                for k,ref in refs.items()}
        release=parse(leases.enter_context(_verified_file(release_ref,maximum=MAXIMUM_BYTES)).read())
        validate_parent(record,refs,contract_ref,values["authority"],values["admission"],release,os.environ,_runtime())
        exact(_metadata(file_reference(root/"STARTED.json")),dict(pid=os.getppid(),run_id=record["run_id"],
            contract=asdict(contract_ref),release=asdict(release_ref)))
        if (root/"FAILED.json").exists() or (root/"COMPLETE.json").exists():
            raise ValueError("Parent is already terminal; no repeat readback child")
        result=verify_terminal(contract_ref,execution_ref)
    return write_json(root/"TERMINAL_READBACK.json",result)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "run", "numerics", "readback"))
    for name in ("contract", "contract-sha256"):
        parser.add_argument("--"+name, required=True)
    for name in ("release", "release-sha256", "execution", "execution-sha256"):
        parser.add_argument("--"+name)
    args = parser.parse_args(argv)
    raw = _read(args.contract, args.contract_sha256, maximum_bytes=MAXIMUM_BYTES)
    cref = ObservationFile(str(_path(args.contract)), args.contract_sha256, len(raw))
    record, _ = decode(raw)
    if args.mode!="readback" and (args.execution is not None or args.execution_sha256 is not None):
        parser.error("Execution publication is only an input to terminal readback")
    if args.mode == "inspect":
        if args.release is not None or args.release_sha256 is not None:
            parser.error("Metadata inspection cannot consume a startup release")
        print("Main qualification metadata valid; no launch authority, optimizer or CUDA calls")
        return 0
    if args.release is None or args.release_sha256 is None:
        parser.error("Exact externally issued startup release required")
    release = _read(args.release, args.release_sha256, maximum_bytes=MAXIMUM_BYTES)
    rref = ObservationFile(str(_path(args.release)), args.release_sha256, len(release))
    if args.mode=="readback":
        if args.execution is None or args.execution_sha256 is None:
            parser.error("Exact execution publication required for terminal readback")
        eraw=_read(args.execution,args.execution_sha256,maximum_bytes=MAXIMUM_BYTES)
        result=execute_readback(cref,rref,ObservationFile(str(_path(args.execution)),args.execution_sha256,len(eraw)))
    else:
        result = execute(cref, rref, numerical_child=args.mode == "numerics")
    print(result.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
