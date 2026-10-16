"""Independent byte/identity readback of a completed disposable worker case.

No fitting, decoding, checkpoint deserialization, scheduler operation or
qualification-pass publication. A CUDA run must supply these files; fabricated
metadata tests prove only this reader's checks, not that a GPU ran.
"""

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .main_qualification_artifacts import file_reference
from .main_qualification_worker_plan import worker_schedule
from .configuration import get_configuration
from .main_worker_contract import decode_contract, MAXIMUM_BYTES, parse
from .observation_loader import ObservationFile, _path, _verified_file
from .retention import identity_sha256
from .selection_evidence import _encode


def _require(value, expected):
    if type(value) is not dict or any(k not in value or
            json.dumps(value[k],sort_keys=True,allow_nan=False) != json.dumps(v,sort_keys=True,allow_nan=False)
            for k,v in expected.items()):
        raise ValueError("Disposable worker publication identity/counts differ")


def verify_publication(contract_reference, completion_reference):
    if any(type(r) is not ObservationFile for r in (contract_reference,completion_reference)):
        raise TypeError("Exact contract/completion path, hash and size required")
    with ExitStack() as leases:
        def read(reference, expected_path=None):
            if expected_path is not None and _path(reference.path) != _path(expected_path):
                raise ValueError("Publication reference is outside its exact worker/trajectory path")
            stream=leases.enter_context(_verified_file(reference,maximum=MAXIMUM_BYTES))
            return parse(stream.read())
        raw=leases.enter_context(_verified_file(contract_reference,maximum=MAXIMUM_BYTES)).read()
        record,parts=decode_contract(raw,qualification=True)
        case=parts["qualification_case"]
        receipts=_path(record["receipt_root"])
        trajectory=_path(record["study_root"])/"trajectories"/parts["task"].key.storage_identifier
        for failed in (receipts/"FAILED.json",trajectory/("FITTING_STAGE_"+case.stage+"_FAILED.json")):
            if os.path.lexists(failed):
                raise ValueError("A retained failure prevents qualification publication acceptance")
        complete=read(completion_reference,receipts/"COMPLETE.json")
        _require(complete,dict(schema="credo.frontier.main-worker-completion.v1",status="complete",
            disposable_worker_qualification=True,scientific_fit=False,
            all_worker_input_and_source_exits_verified=True,automatic_successor=False))
        start_ref=ObservationFile(**complete["started"])
        started=read(start_ref,receipts/"STARTED.json")
        _require(started,dict(schema="credo.frontier.main-worker-started.v1",worker_id=record["worker_id"],
            contract=asdict(contract_reference),qualification=asdict(parts["qualification"]),
            execution_scope="synthetic_cuda_qualification",disposable_worker_qualification=True,
            scientific_updates=0,source_runtime_and_release_verified=True))
        _require(started["qualification_compatibility"],dict(status="disposable_test_not_qualification_pass",
            case=asdict(case),scientific_fit_authorized=False))
        execution=complete["execution"]
        _require(execution,dict(schema="credo.frontier.main-fitting-execution.v1",
            status="complete_after_input_stage_and_lease_exit",task=asdict(parts["task"]),
            work_binding=parts["work_binding"].to_dict(),input_contract=asdict(parts["input_reference"]),
            source_sha256=parts["source_manifest"].sha256,worker_id=record["worker_id"],
            plan_sha256=identity_sha256(asdict(parts["plan"])),execution_scope="synthetic_cuda_qualification",
            accepted_start=case.accepted_start,accepted_target=case.accepted_target,
            scientific_updates_this_task=0,optimizer_calls_this_task=case.optimizer_calls,
            resource_policy_sha256=parts["resource_policy"].identity,
            full_input_exit_verified=True,raw_reconstruction=False,normalization_recomputed=False,
            query_outcomes_opened=False,scientific_launch_authority=False,automatic_successor=False))
        checkpoint=execution["checkpoint"]
        directory=trajectory/"checkpoints"/f"update-{case.accepted_target:09d}"
        if _path(checkpoint["directory"]) != directory:
            raise ValueError("Checkpoint directory is not the exact accepted prefix")
        cp_ref=file_reference(directory/"COMPLETE.json")
        cp=read(cp_ref,directory/"COMPLETE.json")
        if cp != {k:v for k,v in checkpoint.items() if k!="directory"}:
            raise ValueError("Checkpoint COMPLETE differs from the worker publication")
        _require(cp,dict(schema="credo.frontier.resumable_checkpoint.v1",
            accepted_updates=case.accepted_target,attempted_updates=case.accepted_target,phase="joint",
            execution_scope="synthetic_cuda_qualification",milestone=True,worker_id=record["worker_id"],
            maximum_checkpoint_bytes=parts["resource_policy"].checkpoint_limit_bytes))
        _require(cp["identity"],dict(trajectory_id=parts["task"].key.storage_identifier,
            fold_id="subset-"+"-".join(case.training_donors),seed=parts["task"].key.seed,
            configuration_sha256=identity_sha256(asdict(get_configuration(case.configuration_id))),
            source_sha256=parts["source_manifest"].sha256))
        _require(cp["identity"]["training_owner"],dict(
            configuration_sha256=cp["identity"]["configuration_sha256"],
            schedule_sha256=identity_sha256(asdict(worker_schedule())),
            maximum_checkpoint_bytes=parts["resource_policy"].checkpoint_limit_bytes))
        tensor_ref=ObservationFile(str(directory/"checkpoint.pt"),cp["checkpoint_sha256"],cp["checkpoint_bytes"])
        leases.enter_context(_verified_file(tensor_ref,maximum=parts["resource_policy"].checkpoint_limit_bytes))
        stage_ref=ObservationFile(**execution["stage_completion"])
        stage=read(stage_ref,trajectory/("FITTING_STAGE_"+case.stage+".json"))
        _require(stage,dict(schema="credo.frontier.fitting-stage-completion.v1",status="complete",
            identity=cp["identity"],task=asdict(parts["task"]),stage=case.stage,
            plan_sha256=identity_sha256(asdict(parts["plan"])),accepted_updates=case.accepted_target,
            attempted_updates=case.accepted_target,optimizer_calls_this_task=case.optimizer_calls,
            execution_scope="synthetic_cuda_qualification",scientific_updates_this_task=0,
            checkpoint=checkpoint,scientific_launch_authority=False,query_outcomes_opened=False,automatic_successor=False))
        _require(stage["cache_verification"],dict(exit_full_hash_verified=True))
        if case.accepted_start:
            custody=stage["continuation_custody"]
            _require(custody,dict(qualification_checkpoint_updates=case.accepted_start,
                fabricated_scalar_metadata=True,prediction_numerics_qualified_here=False,
                optimizer_calls=0,outcome_or_score_reads=0))
            # VerifiedContinuationCustody hashes its newline-terminated payload,
            # not retention's newline-free generic identity serialization.
            if (stage["continuation_custody_sha256"] != hashlib.sha256(_encode(custody)).hexdigest()
                    or execution["continuation_custody_sha256"] != stage["continuation_custody_sha256"]):
                raise ValueError("Actual continuation custody differs across publications")
            _require(custody,dict(schema="credo.frontier.continuation-forecast-custody.v1",
                status="verified_at_continuation_boundary",task=asdict(parts["task"]),
                previous_stage_sha256=identity_sha256(asdict(parts["previous_stage"])),
                checkpoint_identity=cp["identity"],checkpoint_sha256=record["resume"]["checkpoint_sha256"]))
        elif stage["continuation_custody"] is not None or execution["continuation_custody_sha256"] is not None:
            raise ValueError("Fresh worker cannot claim checkpoint continuation")
        release_ref=ObservationFile(**execution["lease_release"])
        released=read(release_ref,trajectory/"claims"/(cp["claim_id"]+".released.json"))
        if released != dict(claim_id=cp["claim_id"],exception_type=None,automatic_restart_authorized=False):
            raise ValueError("Successful exact trajectory release required")
    return dict(schema="credo.frontier.main-qualification-worker-readback.v1",
        status="complete_publication_readback_verified",case=asdict(case),contract=asdict(contract_reference),
        completion=asdict(completion_reference),started=asdict(start_ref),stage=asdict(stage_ref),
        checkpoint_complete=asdict(cp_ref),checkpoint_payload=asdict(tensor_ref),lease_release=asdict(release_ref),
        scientific_updates=0,disposable_optimizer_calls_reported=case.optimizer_calls,
        actual_cuda_execution_proven_by_this_reader=False,checkpoint_deserializations=0,
        full_worker_qualification_pass=False,automatic_successor=False)
