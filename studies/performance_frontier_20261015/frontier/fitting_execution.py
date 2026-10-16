"""One already admitted main task: inputs -> guarded fit -> verified stage exit.

This is the numerical worker execution API, not a CLI, scheduler or authority.
The caller must first verify the immutable executing source/runtime, actual GPU
qualification, mount/cgroup/quota and aggregate reservation, and saved selector
decision. A WorkBinding describes that reservation; it does not create/prove it.
No raw reconstruction, scale replay, selection, forecast generation or successor
launch occurs. Scientific continuation verifies actual prior forecast custody.
"""

from contextlib import ExitStack
from dataclasses import asdict
import json
import os

from .checkpoints import TrajectoryLease,file_sha256
from .continuation_custody import ContinuationForecastAdmission,verify_continuation_custody
from .coordination import WorkBinding,_label
from .fitting_inputs import FittingInputContract,MAXIMUM_CONTRACT_BYTES,open_fitting_inputs
from .fitting_resources import FittingResourceGuard,FittingResourcePolicy,check_process_resources
from .fitting_stage import FittingStage,task_boundaries
from .observation_loader import ObservationFile,_sha,_verified_file
from .planner import StagePlan,TrainingTask
from .prediction_plan import required_before_continuation
from .retention import STUDY_ID,_path,identity_sha256


SCHEMA="credo.frontier.main-fitting-execution.v1"


def run_admitted_fitting_task(input_reference,*,plan,task,work_binding,source_sha256,
                             study_root,worker_id,resource_policy,device,pin_memory=False,
                             resume=None,previous_stage=None,continuation_admissions=()):
    """Run exactly one bound task, returning only after all success exits.

    Input-contract/metadata/cache contexts exit before FittingStage can publish
    completion. The trajectory lease exits last. Failures remain terminal and
    propagate; the enclosing worker owns its startup/exit failure receipt and
    publication of this returned execution record. There is no retry here.
    """
    if (type(input_reference) is not ObservationFile or type(plan) is not StagePlan
            or type(task) is not TrainingTask or type(work_binding) is not WorkBinding
            or type(resource_policy) is not FittingResourcePolicy):
        raise TypeError("Exact input, registered plan/task, work and resource bindings required")
    plan.__post_init__();task.__post_init__();resource_policy.__post_init__()
    input_reference.__post_init__();_sha(source_sha256);_label(worker_id)
    if task not in plan.training or work_binding!=WorkBinding.main(task):
        raise ValueError("Task differs from the complete plan or admitted work binding")
    if type(pin_memory) is not bool or type(continuation_admissions) is not tuple:
        raise ValueError("Explicit input-transfer choice and immutable custody inventory required")
    # Freeze resume metadata; never infer a best or latest checkpoint here.
    resume=None if resume is None else json.loads(json.dumps(resume,sort_keys=True,allow_nan=False))
    if task.start_updates:
        required=required_before_continuation(previous_stage,task)
        if (type(resume) is not dict or
                any(type(a) is not ContinuationForecastAdmission for a in continuation_admissions)
                or tuple(sorted(a.edge_id for a in continuation_admissions))!=required):
            raise ValueError("Exact checkpoint and every prior directed forecast required before inputs")
    elif resume is not None or previous_stage is not None or continuation_admissions:
        raise ValueError("Fresh fitting cannot receive resume/previous-forecast state")
    root=_path(study_root,directory=True)
    parent=_path(root/"trajectories",directory=True)
    if root.name!=STUDY_ID or root.stat().st_uid!=os.geteuid() or parent.stat().st_uid!=os.geteuid():
        raise ValueError("Prepared owned new-study trajectory parent required")
    output=_path(parent/task.key.storage_identifier)
    if not task.start_updates and os.path.lexists(output):
        raise FileExistsError("Fresh trajectory already exists; no restart or automatic retry")
    if task.start_updates and not output.is_dir():
        raise ValueError("Continuation requires its existing exact trajectory")
    for name in ("FITTING_STAGE_"+plan.name+".json","FITTING_STAGE_"+plan.name+"_FAILED.json"):
        if os.path.lexists(output/name):
            raise FileExistsError("Fitting stage already attempted; no automatic retry")
    before=check_process_resources(resource_policy,device)
    fs=os.statvfs(parent)
    if fs.f_bavail*fs.f_frsize<resource_policy.metadata_reserve_bytes+resource_policy.checkpoint_limit_bytes:
        raise RuntimeError("Filesystem cannot cover the next bounded checkpoint and metadata")
    # OUTER stack owns stage/lease. INNER stack owns every input, including the
    # small preflight contract. It must exit first, even after numerical success.
    with ExitStack() as owners:
        with ExitStack() as inputs:
            stream=inputs.enter_context(_verified_file(input_reference,maximum=MAXIMUM_CONTRACT_BYTES))
            contract=FittingInputContract(stream.read())
            record,parts=contract.record(),contract.components()
            start,targets=task_boundaries(task,parts["schedule"])
            if (parts["task"]!=task or record["source_sha256"]!=source_sha256
                    or parts["schedule"].execution_scope!=resource_policy.execution_scope
                    or record["maximum_checkpoint_bytes"]!=resource_policy.checkpoint_limit_bytes
                    or (resume is not None and resume.get("accepted_updates")!=start)):
                raise ValueError("Serialized inputs differ from the admitted task/source/resource/prefix")
            opened=inputs.enter_context(open_fitting_inputs(input_reference,device=device,pin_memory=pin_memory))
            prepared=opened.prepared
            if opened.task!=task or opened.input_contract_sha256!=input_reference.sha256:
                raise ValueError("Opened inputs differ from the admitted contract")
            custody=None
            if start:
                custody=verify_continuation_custody(previous_stage,task,continuation_admissions,
                    checkpoint_identity=prepared.identity,checkpoint_sha256=resume["checkpoint_sha256"],
                    qualification_prefix=start if resource_policy.execution_scope=="synthetic_cuda_qualification" else None)
            else:
                # Claim fresh pathname atomically after inputs are admitted.
                # Another worker may have created it since the early check;
                # even an empty/failed competing root must not be adopted.
                output.mkdir(mode=0o700)
            lease=owners.enter_context(TrajectoryLease(output,prepared.identity,worker_id))
            guard=FittingResourceGuard(lease,resource_policy)
            stage=owners.enter_context(FittingStage(prepared=prepared,lease=lease,plan=plan,task=task,
                study_root=root,resume=resume,previous_stage=previous_stage,
                resource_guard=guard,continuation_custody=custody))
            stage.run()
    # No success object exists until both the full input exits and lease release
    # have succeeded. The published stage and failure marker remain authoritative.
    if stage.receipt is None or stage.failure_path.exists() or stage.owner.status!="stage_complete":
        raise RuntimeError("Numerical return is not accepted stage completion")
    released=lease.root/"claims"/(lease.claim_id+".released.json")
    release_ref=ObservationFile(str(released),file_sha256(released),released.stat().st_size)
    with _verified_file(release_ref,maximum=1024**2) as stream:
        release=json.load(stream)
    if release!={"claim_id":lease.claim_id,"exception_type":None,"automatic_restart_authorized":False}:
        raise ValueError("Exact successful trajectory release required")
    return dict(schema=SCHEMA,status="complete_after_input_stage_and_lease_exit",
        task=asdict(task),work_binding=work_binding.to_dict(),plan_sha256=identity_sha256(asdict(plan)),
        input_contract=asdict(input_reference),source_sha256=source_sha256,
        execution_source_sha256=file_sha256(__file__),worker_id=worker_id,
        stage_completion=stage.receipt,checkpoint=stage.owner.last_checkpoint,
        lease_release=asdict(release_ref),execution_scope=resource_policy.execution_scope,
        scientific_updates_this_task=task.additional_updates if resource_policy.execution_scope=="scientific" else 0,
        optimizer_calls_this_task=stage.owner.optimizer_calls-start,
        accepted_start=start,accepted_target=targets[-1],resource_policy=asdict(resource_policy),
        resource_policy_sha256=resource_policy.identity,pre_input_resources=before,last_resources=guard.record(),
        continuation_custody_sha256=None if custody is None else custody.identity,
        full_input_exit_verified=True,raw_reconstruction=False,normalization_recomputed=False,
        query_outcomes_opened=False,scientific_launch_authority=False,automatic_successor=False,
        source_runtime_quota_and_selection_admission="Required from enclosing worker; not proved by this execution API")
