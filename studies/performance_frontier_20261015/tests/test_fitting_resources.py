"""Five actual disposable CPU Adam calls; mocked GPU samples are not CUDA work."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontier import fitting_resources as module
from frontier.cache_storage import load_fold_cache
from frontier.checkpoints import TrajectoryLease
from frontier.fitting_factory import prepare_fitting_components
from frontier.fitting_resources import (FittingResourceGuard,FittingResourcePolicy,
    output_usage,snapshot_tensor_bytes)
from frontier.fitting_stage import FittingStage
from frontier.planner import DONORS,TrajectoryKey,family_plan,screen_plan
from frontier.prediction_plan import required_before_continuation
from frontier.retention import STUDY_ID
from frontier.training_owner import OwnerSchedule
from test_checkpoints import identity
from test_fitting_factory import package as factory_fixture


MIB=1024**2
GIB=1024**3
ORIGINAL_STEP=torch.optim.Adam.step
CALLS=0


def policy():
    return FittingResourcePolicy("synthetic_cpu_only",512*MIB,64*MIB,256*MIB,8*MIB,8*GIB,0)


@pytest.fixture
def package(tmp_path,monkeypatch,request):
    global CALLS
    generator=factory_fixture.__wrapped__(tmp_path/"fixture",monkeypatch)
    args,_=next(generator)
    args=dict(args,schedule=OwnerSchedule("synthetic_cpu_only",(2,4,6),1,2,32,1,2,4,4),
              maximum_checkpoint_bytes=64*MIB)
    root=tmp_path/STUDY_ID;root.mkdir();(root/"trajectories").mkdir()
    seen=[]
    def counted(opt,*a,**kw):
        global CALLS
        CALLS+=1
        if CALLS>5:raise RuntimeError("Five-call resource fixture limit exceeded")
        seen.append(1);return ORIGINAL_STEP(opt,*a,**kw)
    monkeypatch.setattr(torch.optim.Adam,"step",counted)
    try:yield dict(args=args,generator=generator,root=root,calls=seen,
                   publication=dict(args["cache"].publication_reference),limits=args["cache"].limits)
    finally:
        generator.close()
        request.node.user_properties.append(("disposable_cpu_optimizer_calls",len(seen)))


def task(plan):
    return next(t for t in plan.training if t.key==TrajectoryKey(("D3","D4"),"F0-V1",0))


def open_stage(package,outer,plan,*,cache=None,previous=None,resume=None,resource_policy=None):
    t=task(plan)
    args=dict(package["args"],trajectory_id=t.key.storage_identifier)
    if cache is not None:args["cache"]=cache
    prepared=prepare_fitting_components(**args)
    lease=outer.enter_context(TrajectoryLease(package["root"]/"trajectories"/t.key.storage_identifier,
                                               prepared.identity,"resource-fixture"))
    guard=FittingResourceGuard(lease,policy() if resource_policy is None else resource_policy)
    stage=outer.enter_context(FittingStage(prepared=prepared,lease=lease,plan=plan,task=t,
        study_root=package["root"],previous_stage=previous,resume=resume,resource_guard=guard,
        verified_complete_edge_ids=() if previous is None else required_before_continuation(previous,t)))
    return stage,guard


def reopen(package):
    ref=package["publication"]
    return load_fold_cache(ref["root"],expected_manifest_sha256=ref["manifest_sha256"],
        expected_complete_sha256=ref["complete_sha256"],output_cap_bytes=MIB,limits=package["limits"])


def test_actual_guarded_stage_continuation_and_unchanged_resource_identity(package,monkeypatch):
    first=screen_plan()
    with ExitStack() as outer:
        stage,guard=open_stage(package,outer,first)
        stage.run();package["generator"].close()
    assert len(package["calls"])==2 and stage.receipt
    assert guard.record()["event"]=="stage_complete"
    assert guard.record()["fileset_quota_checked_by_this_component"] is False
    assert guard.record()["checks"]==10
    second=family_plan({d:tuple(f"F{i}-V1" for i in range(6)) for d in DONORS})
    resume=stage.owner.last_checkpoint
    with ExitStack() as outer:
        with reopen(package) as cache:
            stage,guard=open_stage(package,outer,second,cache=cache,previous=first,resume=resume)
            stage.run()
    assert len(package["calls"])==4 and stage.owner.accepted_updates==4
    state=json.loads(stage.owner.journal.read_text())
    assert state["resource_policy_sha256"]==policy().identity
    payload=torch.load(Path(stage.owner.last_checkpoint["directory"])/"checkpoint.pt",weights_only=True)
    assert payload["sampler_state"]["resource_policy_sha256"]==policy().identity
    # An unchanged model/input identity does not permit an unrecorded resource
    # policy change on resume; no optimizer call or checkpoint load is allowed.
    monkeypatch.setattr("frontier.training_owner.load_checkpoint",lambda *_a,**_k:pytest.fail("No tensor restore"))
    with reopen(package) as cache:
        args=dict(package["args"],cache=cache,trajectory_id=task(second).key.storage_identifier)
        prepared=prepare_fitting_components(**args)
        with TrajectoryLease(stage.lease.root,prepared.identity,"different-policy") as lease:
            changed=FittingResourceGuard(lease,replace(policy(),output_limit_bytes=513*MIB))
            with pytest.raises(ValueError,match="resume"):
                prepared.open_owner(lease,resume=stage.owner.last_checkpoint,
                    retention_policy=stage.policy,resource_guard=changed)
    assert len(package["calls"])==4


def test_snapshot_capacity_failure_precedes_cpu_copy_and_keeps_applied_count(package,monkeypatch):
    first=screen_plan()
    with pytest.raises(RuntimeError,match="snapshot"):
        with ExitStack() as outer:
            stage,guard=open_stage(package,outer,first)
            monkeypatch.setattr(module,"snapshot_tensor_bytes",lambda _:65*MIB)
            monkeypatch.setattr("frontier.checkpoints._cpu_tree",lambda *_:pytest.fail("No snapshot copies"))
            stage.run()
    assert len(package["calls"])==1
    state=json.loads(stage.owner.journal.read_text())
    assert state["status"]=="failed" and state["accepted_updates"]==state["optimizer_calls"]==1
    assert not list((stage.lease.root/"checkpoints").iterdir())
    assert stage.failure_path.exists() and not stage.path.exists()


@pytest.mark.parametrize("failure",["capacity","RSS","policy"])
def test_runtime_guard_failure_before_update_remains_terminal_and_recorded(package,monkeypatch,failure):
    first=screen_plan()
    with pytest.raises((ValueError,RuntimeError)):
        with ExitStack() as outer:
            stage,guard=open_stage(package,outer,first)
            if failure=="capacity":
                monkeypatch.setattr(module,"output_usage",lambda _:dict(output_bytes=512*MIB,
                    filesystem_available_bytes=GIB,fileset_quota_checked_by_this_component=False))
            elif failure=="RSS":
                monkeypatch.setattr(module,"process_resources",lambda _:dict(maximum_rss_bytes=9*GIB,threads=2))
            else:guard.policy=replace(policy(),output_limit_bytes=513*MIB)
            stage.run()
    assert package["calls"]==[]
    assert stage.owner.status=="failed"
    assert json.loads(stage.owner.journal.read_text())["status"]=="failed"
    assert stage.failure_path.exists()


def test_snapshot_estimate_does_not_clone_cache_or_change_model(package,monkeypatch):
    first=screen_plan()
    # No stage.run: use a plain owner to inspect the estimate without applying
    # a numerical step or publishing a stage success.
    t=task(first)
    args=dict(package["args"],trajectory_id=t.key.storage_identifier)
    prepared=prepare_fitting_components(**args)
    with TrajectoryLease(package["root"]/"trajectories"/t.key.storage_identifier,prepared.identity,"estimate") as lease:
        owner=prepared.open_owner(lease,resource_guard=FittingResourceGuard(lease,policy()))
        monkeypatch.setattr(owner.cache,"state_dict",lambda:pytest.fail("No cache CPU clone"))
        n=snapshot_tensor_bytes(owner)
        assert n>=sum(x.numel()*x.element_size() for x in owner.model.state_dict().values())>0
        assert owner.optimizer_calls==0


@pytest.mark.parametrize("field,value",[
    ("output_limit_bytes",64*MIB),("checkpoint_limit_bytes",True),("snapshot_reserve_bytes",64*MIB),
    ("metadata_reserve_bytes",MIB),("maximum_rss_bytes",9*GIB),("minimum_gpu_headroom_bytes",8*GIB),
    ("execution_scope","other")])
def test_policy_cannot_expand_or_omit_scope(field,value):
    with pytest.raises(ValueError):replace(policy(),**{field:value})


def test_output_measurement_preserves_files_and_rejects_links(tmp_path):
    (tmp_path/"nested").mkdir()
    p=tmp_path/"nested/data";p.write_bytes(b"abc")
    assert output_usage(tmp_path)["output_bytes"]==3
    (tmp_path/"alias").symlink_to(p)
    with pytest.raises(ValueError,match="Symlink"):output_usage(tmp_path)
    assert p.read_bytes()==b"abc"


def test_output_measurement_rejects_unreadable_subtree(tmp_path,monkeypatch):
    def inaccessible(_root,*,followlinks,onerror):
        assert not followlinks
        onerror(PermissionError("Cannot inspect exact trajectory"))
        return iter(())
    monkeypatch.setattr(module.os,"walk",inaccessible)
    with pytest.raises(PermissionError,match="exact trajectory"):
        output_usage(tmp_path)


def test_output_measurement_rejects_changed_root_owner(tmp_path,monkeypatch):
    actual=tmp_path.stat().st_uid
    monkeypatch.setattr(module.os,"geteuid",lambda:actual+1)
    with pytest.raises(ValueError,match="root ownership"):
        output_usage(tmp_path)


def test_mocked_gpu_measurement_is_not_hardware_qualification(tmp_path,monkeypatch):
    root=tmp_path/STUDY_ID;root.mkdir()
    admitted=replace(policy(),execution_scope="scientific",maximum_rss_bytes=200*GIB,
                     minimum_gpu_headroom_bytes=8*GIB)
    sample=dict(maximum_rss_bytes=GIB,threads=8,gpu_name="NVIDIA H100 80GB HBM3",
                total_gpu_bytes=80*GIB,peak_reserved_gpu_bytes=70*GIB)
    monkeypatch.setattr(module,"process_resources",lambda _:dict(sample))
    with TrajectoryLease(root/"mocked",identity(),"mock") as lease:
        owner=SimpleNamespace(lease=lease,schedule=SimpleNamespace(execution_scope="scientific"),
            maximum_checkpoint_bytes=64*MIB,device=torch.device("cuda:0"),accepted_updates=0)
        guard=FittingResourceGuard(lease,admitted);guard.bind(owner)
        sample["peak_reserved_gpu_bytes"]=73*GIB
        with pytest.raises(RuntimeError,match="headroom"):guard.check("before_update")
        assert guard.record()["checks"]==1
