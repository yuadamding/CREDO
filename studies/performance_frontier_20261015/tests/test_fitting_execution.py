"""One-call execution integration; four actual disposable CPU Adam calls.

Two updates complete a tiny DIRECT task; two precede an injected input-exit
failure. No remote/GPU/biological access or scientific model fitting occurs.
"""

from contextlib import contextmanager
from dataclasses import replace
import copy
import json
from pathlib import Path

import pytest
import torch

from frontier import fitting_execution as module
from frontier.coordination import WorkBinding
from frontier.fitting_execution import run_admitted_fitting_task
from frontier.fitting_resources import FittingResourcePolicy,check_process_resources
from frontier.planner import DONORS,TrajectoryKey,family_plan,screen_plan
from test_fitting_inputs import package as input_fixture,published,forbid_recomputation


MIB=1024**2
GIB=1024**3
ORIGINAL_STEP=torch.optim.Adam.step
CALLS=0


@pytest.fixture
def package(tmp_path,monkeypatch,request):
    generator=input_fixture.__wrapped__(tmp_path,monkeypatch,request)
    p=next(generator)
    (p["root"]/"trajectories").mkdir()
    p["reference"]=published(p)
    p["plan"]=screen_plan()
    p["task"]=next(t for t in p["plan"].training if t.key==TrajectoryKey(("D3","D4"),"F0-V1",0))
    try:yield p
    finally:generator.close()


def arguments(p):
    return dict(plan=p["plan"],task=p["task"],work_binding=WorkBinding.main(p["task"]),
        source_sha256=p["record"]["source_sha256"],study_root=p["root"],worker_id="fabricated-execution",
        resource_policy=FittingResourcePolicy("synthetic_cpu_only",512*MIB,64*MIB,256*MIB,8*MIB,8*GIB,0),
        device="cpu")


def count_steps(p,monkeypatch):
    def step(opt,*a,**kw):
        global CALLS
        CALLS+=1
        if CALLS>4:raise RuntimeError("Four-call execution fixture allowance exceeded")
        p["calls"].append(1)
        return ORIGINAL_STEP(opt,*a,**kw)
    monkeypatch.setattr(torch.optim.Adam,"step",step)


def test_one_call_executes_guarded_task_and_returns_after_all_exits(package,monkeypatch):
    count_steps(package,monkeypatch);forbid_recomputation(monkeypatch)
    out=run_admitted_fitting_task(package["reference"],**arguments(package))
    assert len(package["calls"])==2
    assert out["scientific_updates_this_task"]==0 and out["optimizer_calls_this_task"]==2
    assert out["accepted_start"]==0 and out["accepted_target"]==2
    assert out["full_input_exit_verified"] and not out["normalization_recomputed"]
    stage=json.loads(Path(out["stage_completion"]["path"]).read_text())
    assert stage["cache_verification"]["exit_full_hash_verified"]
    assert out["resource_policy_sha256"]==out["last_resources"]["policy_sha256"]
    assert json.loads(Path(out["lease_release"]["path"]).read_text())["exception_type"] is None
    with pytest.raises(FileExistsError,match="already exists"):
        run_admitted_fitting_task(package["reference"],**arguments(package))
    assert len(package["calls"])==2


def test_post_update_input_failure_never_returns_success_and_marks_owner_failed(package,monkeypatch):
    count_steps(package,monkeypatch);forbid_recomputation(monkeypatch)
    original=module.open_fitting_inputs
    @contextmanager
    def changed(*a,**kw):
        with original(*a,**kw) as opened:
            yield opened
            Path(package["reference"].path).touch()
    monkeypatch.setattr(module,"open_fitting_inputs",changed)
    with pytest.raises(ValueError,match="changed"):
        run_admitted_fitting_task(package["reference"],**arguments(package))
    assert len(package["calls"])==2
    output=package["root"]/"trajectories"/package["task"].key.storage_identifier
    assert not (output/"FITTING_STAGE_screen.json").exists()
    assert (output/"FITTING_STAGE_screen_FAILED.json").is_file()
    state=json.loads((output/"TRAINING_OWNER_STATE.json").read_text())
    assert state["status"]=="failed" and state["accepted_updates"]==state["optimizer_calls"]==2
    assert not state["automatic_retry"]


@pytest.mark.parametrize("change",["source","scope","checkpoint_cap","task"])
def test_input_binding_rejection_precedes_prepared_input_open(package,monkeypatch,change):
    record=copy.deepcopy(package["record"])
    if change=="source":record["source_sha256"]="f"*64
    elif change=="scope":
        args=arguments(package)
        args["resource_policy"]=replace(args["resource_policy"],execution_scope="scientific",
            maximum_rss_bytes=200*GIB,minimum_gpu_headroom_bytes=8*GIB)
        monkeypatch.setattr(module,"check_process_resources",lambda *_a,**_k:{})
    elif change=="checkpoint_cap":record["maximum_checkpoint_bytes"]=32*MIB
    else:record["task"]["key"]["configuration_id"]="F1-V1"
    monkeypatch.setattr(module,"open_fitting_inputs",lambda *_a,**_k:pytest.fail("No model/cache open"))
    args=arguments(package) if change!="scope" else args
    with pytest.raises(ValueError,match="admitted task/source/resource/prefix"):
        run_admitted_fitting_task(published(package,record),**args)
    assert not list((package["root"]/"trajectories").iterdir())


@pytest.mark.parametrize("change",["work","plan","resume","prior","custody","worker","root"])
def test_bad_task_admission_rejected_before_any_contract_read(package,monkeypatch,change):
    args=arguments(package)
    if change=="work":args["work_binding"]=WorkBinding.main(package["plan"].training[0])
    elif change=="plan":args["plan"]=family_plan({d:tuple(f"F{i}-V1" for i in range(6)) for d in DONORS})
    elif change=="resume":args["resume"]={"accepted_updates":2}
    elif change=="prior":args["previous_stage"]=package["plan"]
    elif change=="custody":args["continuation_admissions"]=("invented",)
    elif change=="worker":args["worker_id"]="../escape"
    else:args["study_root"]=package["root"].parent
    monkeypatch.setattr(module,"_verified_file",lambda *_a,**_k:pytest.fail("No contract read"))
    with pytest.raises((ValueError,FileNotFoundError)):
        run_admitted_fitting_task(package["reference"],**args)
    assert package["calls"]==[]


def test_missing_continuation_inventory_fails_before_inputs(package,monkeypatch):
    args=arguments(package)
    args["plan"]=family_plan({d:tuple(f"F{i}-V1" for i in range(6)) for d in DONORS})
    args["task"]=next(t for t in args["plan"].training if t.key==package["task"].key)
    args.update(work_binding=WorkBinding.main(args["task"]),previous_stage=package["plan"],
        resume={"accepted_updates":2,"checkpoint_sha256":"a"*64})
    monkeypatch.setattr(module,"_verified_file",lambda *_a,**_k:pytest.fail("No contract read"))
    with pytest.raises(ValueError,match="every prior directed forecast"):
        run_admitted_fitting_task(package["reference"],**args)


def test_wrong_fixture_device_is_rejected_before_cuda_probe(monkeypatch):
    policy=FittingResourcePolicy("synthetic_cpu_only",512*MIB,64*MIB,256*MIB,8*MIB,8*GIB,0)
    monkeypatch.setattr("frontier.fitting_resources.process_resources",lambda *_a,**_k:pytest.fail("No device probe"))
    with pytest.raises(ValueError,match="CPU-only"):
        check_process_resources(policy,"cuda:0")


def test_competing_fresh_path_is_not_adopted_after_input_preparation(package,monkeypatch):
    original=module.open_fitting_inputs
    output=package["root"]/"trajectories"/package["task"].key.storage_identifier
    @contextmanager
    def competing(*a,**kw):
        with original(*a,**kw) as opened:
            output.mkdir()
            yield opened
    monkeypatch.setattr(module,"open_fitting_inputs",competing)
    monkeypatch.setattr(module,"TrajectoryLease",lambda *_a,**_k:pytest.fail("Do not adopt competing output"))
    with pytest.raises(FileExistsError):
        run_admitted_fitting_task(package["reference"],**arguments(package))
    assert output.is_dir() and not list(output.iterdir()) and package["calls"]==[]


def test_continuation_routes_exact_checkpoint_to_custody_before_owner(package,monkeypatch):
    from frontier.continuation_custody import ContinuationForecastAdmission
    from frontier.selection_evidence import SelectionMetadataFile
    from frontier.prediction_plan import required_before_continuation
    args=arguments(package)
    args["plan"]=family_plan({d:tuple(f"F{i}-V1" for i in range(6)) for d in DONORS})
    args["task"]=next(t for t in args["plan"].training if t.key==package["task"].key)
    args.update(work_binding=WorkBinding.main(args["task"]),previous_stage=package["plan"],
        resume={"accepted_updates":2,"checkpoint_sha256":"a"*64})
    base=package["root"]/"unopened-custody"
    refs=[SelectionMetadataFile(str(base/name),"b"*64,1) for name in ("SEAL.json","SEAL.json.plan.json","COMPLETE.json")]
    args["continuation_admissions"]=tuple(ContinuationForecastAdmission(edge,*refs,MIB,MIB,True)
        for edge in required_before_continuation(package["plan"],args["task"]))
    record=copy.deepcopy(package["record"])
    from dataclasses import asdict
    record["task"]=asdict(args["task"])
    output=package["root"]/"trajectories"/args["task"].key.storage_identifier
    output.mkdir()
    def custody(previous,task,admissions,**kw):
        assert previous==package["plan"] and task==args["task"]
        assert admissions==args["continuation_admissions"]
        assert kw["checkpoint_sha256"]=="a"*64
        assert kw["checkpoint_identity"]["trajectory_id"]==task.key.storage_identifier
        raise ValueError("Injected custody refusal before any owner")
    monkeypatch.setattr(module,"verify_continuation_custody",custody)
    monkeypatch.setattr(module,"TrajectoryLease",lambda *_a,**_k:pytest.fail("No owner before custody"))
    with pytest.raises(ValueError,match="custody refusal"):
        run_admitted_fitting_task(published(package,record),**args)
    assert not list(output.iterdir()) and package["calls"]==[]
