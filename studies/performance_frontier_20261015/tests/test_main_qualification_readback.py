"""Publication reader tests with explicitly opaque, never executed checkpoints."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from frontier.configuration import get_configuration
from frontier.main_qualification_artifacts import file_reference, write_json
from frontier.main_qualification_readback import verify_publication
from frontier.main_qualification_worker_plan import cases, worker_schedule
from frontier.main_worker_contract import decode_contract
from frontier.observation_loader import ObservationFile
from frontier.native_inputs import FoldInputs
from frontier.retention import identity_sha256
from frontier.selection_evidence import _encode
from test_main_worker import contract, encoded, no_numerics
from test_main_worker_qualification import qualify


def publication(base, case):
    record = qualify(base, case)
    root = Path(record["study_root"])
    cref = write_json(root/"WORKER.json", record)
    record, parts = decode_contract(Path(cref.path).read_bytes(), qualification=True)
    return publish_outputs(cref,record,parts)


def publish_outputs(cref,record,parts):
    case=parts["qualification_case"]
    root=Path(record["study_root"])
    receipts = Path(record["receipt_root"]); receipts.mkdir()
    trajectory = root/"trajectories"/parts["task"].key.storage_identifier
    directory = trajectory/"checkpoints"/f"update-{case.accepted_target:09d}"
    directory.mkdir(parents=True); (trajectory/"claims").mkdir(exist_ok=True)
    # Never torch.load or represent these bytes as an actual optimizer run.
    tensor = directory/"checkpoint.pt"
    tensor.write_bytes(b"OPAQUE-CHECKPOINT-METADATA-TEST-NOT-CUDA-OR-TRAINING")
    identity = dict(trajectory_id=parts["task"].key.storage_identifier,
        fold_id="subset-"+"-".join(case.training_donors), seed=0,
        configuration_sha256=identity_sha256(asdict(get_configuration(case.configuration_id))),
        input_sha256="a"*64, source_sha256=parts["source_manifest"].sha256)
    identity["native_roles"]={key:identity_sha256(["opaque-role",key]) for key in
        ("representation_fit","dynamics_source","dynamics_supervision")}
    identity["input_sha256"]=identity_sha256(dict(
        fold=FoldInputs(identity["fold_id"],case.training_donors).identity,
        source=identity["native_roles"]["dynamics_source"],
        supervision=identity["native_roles"]["dynamics_supervision"]))
    identity["training_owner"] = dict(configuration_sha256=identity["configuration_sha256"],
        schedule_sha256=identity_sha256(asdict(worker_schedule())),
        maximum_checkpoint_bytes=parts["resource_policy"].checkpoint_limit_bytes,
        observation_binding=dict(artifact_sha256="a"*64,state_sha256="b"*64))
    cp = dict(schema="credo.frontier.resumable_checkpoint.v1", identity=identity,
        checkpoint_sha256=file_reference(tensor).sha256, checkpoint_bytes=tensor.stat().st_size,
        accepted_updates=case.accepted_target, attempted_updates=case.accepted_target,
        phase="joint", execution_scope="synthetic_cuda_qualification", milestone=True,
        claim_id="opaque-test-claim-"+case.stage, worker_id=record["worker_id"],
        maximum_checkpoint_bytes=parts["resource_policy"].checkpoint_limit_bytes)
    write_json(directory/"COMPLETE.json", cp)
    checkpoint = dict(directory=str(directory), **cp)
    custody = None
    if case.accepted_start:
        custody = dict(schema="credo.frontier.continuation-forecast-custody.v1",
            status="verified_at_continuation_boundary",task=asdict(parts["task"]),
            previous_stage_sha256=identity_sha256(asdict(parts["previous_stage"])),
            checkpoint_identity=identity,checkpoint_sha256=record["resume"]["checkpoint_sha256"],
            qualification_checkpoint_updates=2,fabricated_scalar_metadata=True,
            prediction_numerics_qualified_here=False,optimizer_calls=0,outcome_or_score_reads=0)
    custody_hash = None if custody is None else hashlib.sha256(_encode(custody)).hexdigest()
    stage = dict(schema="credo.frontier.fitting-stage-completion.v1", status="complete",
        identity=identity,task=asdict(parts["task"]),stage=case.stage,
        plan_sha256=identity_sha256(asdict(parts["plan"])),accepted_updates=case.accepted_target,
        attempted_updates=case.accepted_target,optimizer_calls_this_task=case.optimizer_calls,
        execution_scope="synthetic_cuda_qualification",scientific_updates_this_task=0,
        checkpoint=checkpoint,scientific_launch_authority=False,query_outcomes_opened=False,
        automatic_successor=False,cache_verification=dict(exit_full_hash_verified=True),
        continuation_custody=custody,continuation_custody_sha256=custody_hash)
    sref = write_json(trajectory/("FITTING_STAGE_"+case.stage+".json"), stage)
    rref = write_json(trajectory/"claims"/(cp["claim_id"]+".released.json"),
        dict(claim_id=cp["claim_id"],exception_type=None,automatic_restart_authorized=False))
    started = dict(schema="credo.frontier.main-worker-started.v1",worker_id=record["worker_id"],
        contract=asdict(cref),qualification=asdict(parts["qualification"]),
        execution_scope="synthetic_cuda_qualification",disposable_worker_qualification=True,
        scientific_updates=0,source_runtime_and_release_verified=True,
        qualification_compatibility=dict(status="disposable_test_not_qualification_pass",
            case=asdict(case),scientific_fit_authorized=False))
    stref = write_json(receipts/"STARTED.json",started)
    execution = dict(schema="credo.frontier.main-fitting-execution.v1",
        status="complete_after_input_stage_and_lease_exit",task=asdict(parts["task"]),
        work_binding=parts["work_binding"].to_dict(),input_contract=asdict(parts["input_reference"]),
        source_sha256=parts["source_manifest"].sha256,worker_id=record["worker_id"],
        plan_sha256=identity_sha256(asdict(parts["plan"])),execution_scope="synthetic_cuda_qualification",
        accepted_start=case.accepted_start,accepted_target=case.accepted_target,
        scientific_updates_this_task=0,optimizer_calls_this_task=case.optimizer_calls,
        resource_policy_sha256=parts["resource_policy"].identity,full_input_exit_verified=True,
        raw_reconstruction=False,normalization_recomputed=False,query_outcomes_opened=False,
        scientific_launch_authority=False,automatic_successor=False,checkpoint=checkpoint,
        stage_completion=asdict(sref),lease_release=asdict(rref),continuation_custody_sha256=custody_hash)
    complete = dict(schema="credo.frontier.main-worker-completion.v1",status="complete",
        disposable_worker_qualification=True,scientific_fit=False,
        all_worker_input_and_source_exits_verified=True,automatic_successor=False,
        started=asdict(stref),execution=execution)
    done = write_json(receipts/"COMPLETE.json",complete)
    return cref,done,complete,parts


def rewrite(reference, value):
    # Only this test's fabricated files. No production receipt is edited.
    Path(reference.path).write_bytes(encoded(value))
    return file_reference(reference.path)


@pytest.mark.parametrize("case",cases())
def test_readback_retains_actual_counts_and_does_not_claim_cuda_execution(contract,case):
    cref,done,_,_=publication(contract,case)
    result=verify_publication(cref,done)
    assert result["status"]=="complete_publication_readback_verified"
    assert result["scientific_updates"]==result["checkpoint_deserializations"]==0
    assert result["disposable_optimizer_calls_reported"]==case.optimizer_calls
    assert not result["actual_cuda_execution_proven_by_this_reader"]
    assert not result["full_worker_qualification_pass"]


@pytest.mark.parametrize("mutation",["failed","changed_tensor","wrong_scope","nominal_prefix",
    "missing_cache_exit","wrong_claim","wrong_source","wrong_seed","wrong_schedule","foreign_path"])
def test_readback_rejects_invalid_publications(contract,mutation):
    cref,done,value,parts=publication(contract,cases()[0])
    cp=value["execution"]["checkpoint"]; directory=Path(cp["directory"])
    if mutation=="failed": write_json(Path(done.path).parent/"FAILED.json",dict(injected=True))
    elif mutation=="changed_tensor": (directory/"checkpoint.pt").write_bytes(b"changed")
    elif mutation=="wrong_scope": value["scientific_fit"]=True
    elif mutation=="nominal_prefix": value["execution"]["accepted_target"]=1024
    elif mutation=="missing_cache_exit":
        sref=ObservationFile(**value["execution"]["stage_completion"])
        stage=json.loads(Path(sref.path).read_bytes());stage["cache_verification"]["exit_full_hash_verified"]=False
        value["execution"]["stage_completion"]=asdict(rewrite(sref,stage))
    elif mutation=="wrong_claim":
        rref=ObservationFile(**value["execution"]["lease_release"])
        value["execution"]["lease_release"]=asdict(rewrite(rref,dict(claim_id="wrong",exception_type=None,
            automatic_restart_authorized=False)))
    elif mutation=="foreign_path": cp["directory"]=str(directory.parent/"update-000001024")
    else:
        if mutation=="wrong_source": cp["identity"]["source_sha256"]="b"*64
        elif mutation=="wrong_seed": cp["identity"]["seed"]=1
        else: cp["identity"]["training_owner"]["schedule_sha256"]="b"*64
        rewrite(file_reference(directory/"COMPLETE.json"),{k:v for k,v in cp.items() if k!="directory"})
    done=rewrite(done,value)
    with pytest.raises((ValueError,FileNotFoundError)): verify_publication(cref,done)


def test_continuation_uses_newline_terminated_custody_hash(contract):
    cref,done,value,_=publication(contract,cases()[1])
    assert verify_publication(cref,done)["disposable_optimizer_calls_reported"]==1
    sref=ObservationFile(**value["execution"]["stage_completion"])
    stage=json.loads(Path(sref.path).read_bytes())
    stage["continuation_custody_sha256"]=identity_sha256(stage["continuation_custody"])
    value["execution"]["continuation_custody_sha256"]=stage["continuation_custody_sha256"]
    value["execution"]["stage_completion"]=asdict(rewrite(sref,stage))
    with pytest.raises(ValueError): verify_publication(cref,rewrite(done,value))
