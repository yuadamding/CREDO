"""Synthetic receipt graphs only: opaque checkpoints and no numerical replay."""

import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from frontier import main_qualification_worker as producer
from frontier import main_qualification_terminal as terminal
from frontier.configuration import get_configuration
from frontier.main_qualification_artifacts import file_reference,write_json
from frontier.main_qualification_contract import settings
from frontier.main_qualification_fixture import FixtureDimensions
from frontier.main_qualification_plan import cases,plan_identity,qualification_schedule,SCOPE
from frontier.main_qualification_precision import precision_dimensions,precision_plan,TOLERANCES,PHASES
from frontier.main_qualification_readback import verify_publication
from frontier.main_qualification_worker_plan import cases as worker_cases
from frontier.retention import identity_sha256
from frontier.qualification_inventory import OutputInventory
from test_main_qualification_worker import parent,mocked_parent,mocked_numerics
from test_main_worker import no_numerics


GIB,MIB=1024**3,1024**2


def rewrite(path,value):
    # These are new test-owned fabricated files, never retained run evidence.
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
    return file_reference(path)


def read(path):return json.loads(Path(path).read_bytes())


def precision_record(case,phase,source,rules,geometry):
    statistic=dict(elements=1,maximum_absolute_error=0.,maximum_reference_magnitude=1.,
                   maximum_tolerance_normalized_error=0.,passed=True)
    channels={k:dict(statistic) for k in ("objective","global_regularizer","active_log_mass")}
    if phase=="joint":
        channels.update({k:dict(statistic) for k in ("count_ce","donor_count_ce","count_gradient_coefficients")})
    gradients={"fixture.parameter":dict(statistic,elements=4)}
    comparison=dict(status="passed",channels=channels,parameter_gradients=gradients,absent_gradients=[],
        all_parameter_gradients_checked=True,compared_gradient_elements=4,maximum_gradient_absolute_error=0.,
        maximum_channel_absolute_error=0.,maximum_tolerance_normalized_error=0.)
    return dict(schema="credo.frontier.main-paired-precision-case.v1",status="passed",case=asdict(case),phase=phase,
        input_dimensions=asdict(precision_dimensions()),tolerances=dict(TOLERANCES),source_sha256=source,
        rules_sha256=rules,geometry=geometry,input_identity=identity_sha256(dict(
            schema="credo.main-qualification-fabricated-inputs.v1",donors=case.training_donors,
            dimensions=asdict(precision_dimensions()),seed=419)),precision_plan_sha256=identity_sha256(precision_plan()),
        optimizer_calls=0,scientific_updates=0,same_inputs_and_initial_weights=True,real_biological_payload_access=False,
        full_worker_entry_verified=False,full_shape_memory_qualified=False,main_worker_qualification_pass=False,
        parameter_count=4,active_parameter_count=4,cpu_cuda=copy.deepcopy(comparison),
        cuda_microbatch_accumulation=copy.deepcopy(comparison),fixture_only_not_actual_precision=True)


def owner_record(base,case,source,geometry):
    expected=[("primary",2,2)]
    if case.restart_comparison:expected.extend([("primary",3,1),("continuous-reference",3,3)])
    rows=[]
    trajectories=[("primary",3 if case.restart_comparison else 2)]
    if case.restart_comparison:trajectories.append(("continuous-reference",3))
    for label,stop in trajectories:
        trajectory=base/label;(trajectory/"claims").mkdir(parents=True)
        identity=dict(trajectory_id=label,fold_id="fabricated-"+"-".join(case.training_donors),seed=419,
            source_sha256=source,input_sha256="d"*64,
            configuration_sha256=identity_sha256(asdict(get_configuration(case.configuration_id))),
            training_owner=dict(schedule_sha256=identity_sha256(asdict(qualification_schedule())),
                                maximum_checkpoint_bytes=64*MIB))
        for update in range(1,stop+1):
            folder=trajectory/"checkpoints"/f"update-{update:09d}";folder.mkdir(parents=True)
            (folder/"checkpoint.pt").write_bytes(b"OPAQUE-TEST-NOT-A-TENSOR-OR-TRAINED-MODEL")
            tensor=file_reference(folder/"checkpoint.pt")
            claim="opaque-"+str(update)
            write_json(trajectory/"claims"/(claim+".released.json"),dict(
                claim_id=claim,exception_type=None,automatic_restart_authorized=False))
            cp=dict(schema="credo.frontier.resumable_checkpoint.v1",identity=identity,checkpoint_sha256=tensor.sha256,
                checkpoint_bytes=tensor.size_bytes,accepted_updates=update,attempted_updates=update,
                phase="state" if update==1 else "joint",execution_scope=SCOPE,milestone=update>=2,
                claim_id=claim,worker_id=label+"-worker-0",maximum_checkpoint_bytes=64*MIB)
            write_json(folder/"COMPLETE.json",cp)
    for label,updates,calls_ in expected:
        folder=base/label/"checkpoints"/f"update-{updates:09d}"
        rows.append(dict(label=label,accepted=updates,disposable_calls=calls_,scientific_updates=0,
            active_parameters=4,last_metrics=dict(update=updates,phase="joint",gradient_norm_before_clip=1.,
                                                  objective=.5,count_ce=2.),
            checkpoint=dict(directory=str(folder),**read(folder/"COMPLETE.json")),
            resources=dict(threads=8,accepted_updates=updates,event="stage_complete",maximum_rss_bytes=GIB,
                output_bytes=1024,peak_reserved_gpu_bytes=GIB,total_gpu_bytes=80*GIB,gpu_name="NVIDIA H100 80GB HBM3")))
    times=[("primary",2)]+([("primary",3),("continuous-reference",2),("continuous-reference",3)]
                           if case.restart_comparison else [])
    return dict(schema="credo.frontier.main-cuda-owner-case.v1",status="owner_case_passed",case=asdict(case),
        plan_sha256=plan_identity(),source_sha256=source,input_dimensions=asdict(FixtureDimensions()),
        execution=settings()["owner_execution"],geometry=geometry,optimizer_calls=case.optimizer_calls,
        scientific_updates=0,real_biological_payload_access=False,restart_verified=case.restart_comparison,
        full_worker_entry_verified=False,main_worker_qualification_pass=False,automatic_retry=False,automatic_successor=False,
        records=rows,timings=[dict(label=a,accepted_target=b,elapsed_seconds=1.) for a,b in times],
        fixture_only_no_optimizer_or_cuda=True)


@pytest.fixture
def complete(parent,mocked_parent):
    producer.execute(parent.cref,parent.rref)  # GPU branches are mocked by imported fixtures.
    numerical=read(parent.root/"numerics/COMPLETE.json")
    geometry=dict(maximum_geometry_elements=settings()["maximum_geometry_elements"],
        maximum_geometry_guides=settings()["maximum_geometry_guides"],expected_geomloss_version="0.3.1")
    numerical["precision"],numerical["owners"]=[],[]
    for case in cases():
        for phase in PHASES:
            p=parent.root/"numerics"/(case.identifier+"-"+phase+"-precision.json")
            value=precision_record(case,phase,parent.refs["source_manifest"].sha256,parent.refs["rules"].sha256,geometry)
            numerical["precision"].append(asdict(rewrite(p,value)))
        base=parent.root/"numerics"/case.identifier
        value=owner_record(base,case,parent.refs["source_manifest"].sha256,geometry)
        numerical["owners"].append(asdict(rewrite(base/"OWNER_CASE_COMPLETE.json",value)))
    numerical["hardware"]=dict(device="cuda:0",name="NVIDIA H100 80GB HBM3",capability=[9,0],visible_devices=1,
        precision="FP32",tf32=False,deterministic_algorithms=True,required_headroom_bytes=8*GIB,total_memory_bytes=80*GIB)
    numerical["resources"]=dict(maximum_rss_bytes=GIB,allocated_bytes=GIB,reserved_bytes=GIB,
        peak_allocated_bytes=GIB,peak_reserved_bytes=GIB,physical_headroom_at_peak_reserved_bytes=79*GIB)
    numerical["backend"]=dict(version="0.3.1",scalar_module="geomloss._legacy.samples_loss",
        divergence_module="geomloss._legacy.sinkhorn_divergence",utilities_module="geomloss._legacy.utils")
    start=read(parent.root/"numerics/STARTED.json")
    start.update({k:numerical[k] for k in ("hardware","runtime","backend","plan_sha256","scientific_updates","automatic_retry")})
    rewrite(parent.root/"numerics/STARTED.json",start)
    nr=rewrite(parent.root/"numerics/COMPLETE.json",numerical)
    execution=read(parent.root/"EXECUTION_COMPLETE.json");execution["numerics"]=asdict(nr)
    sequence_path=Path(execution["worker_sequence"]["path"])
    sequence=read(sequence_path)
    for index,case in enumerate(worker_cases()):
        readback=sequence["readbacks"][index];saved=read(readback["path"])
        terminal_cp_path=Path(saved["checkpoint_complete"]["path"])
        cp=read(terminal_cp_path)
        for update in range(case.accepted_start+1,case.accepted_target):
            folder=terminal_cp_path.parent.parent/f"update-{update:09d}";folder.mkdir()
            (folder/"checkpoint.pt").write_bytes(Path(saved["checkpoint_payload"]["path"]).read_bytes())
            write_json(folder/"COMPLETE.json",dict(cp,accepted_updates=update,attempted_updates=update,
                phase="state" if update==1 else "joint",milestone=update>=2))
        started=read(saved["started"]["path"])
        request=Path(saved["contract"]["path"]).parent
        started.update(hardware=numerical["hardware"],runtime=parent.runtime,
            startup_release=asdict(file_reference(request/"RELEASE.json")))
        sref=rewrite(saved["started"]["path"],started)
        done=read(saved["completion"]["path"]);done["started"]=asdict(sref)
        dref=rewrite(saved["completion"]["path"],done)
        fresh=verify_publication(file_reference(saved["contract"]["path"]),dref)
        sequence["readbacks"][index]=asdict(rewrite(readback["path"],fresh))
    execution["worker_sequence"]=asdict(rewrite(sequence_path,sequence))
    eref=rewrite(parent.root/"EXECUTION_COMPLETE.json",execution)
    return parent,eref,geometry


def test_complete_inventory_reconciles_all_cases_without_loading_checkpoint(complete):
    parent,eref,_=complete
    result=terminal.verify(parent.cref,eref)
    assert result["precision_cases"]==96 and result["owner_cases"]==48 and result["worker_cases"]==3
    assert result["owner_terminal_checkpoint_payloads"]==52
    assert result["owner_all_checkpoint_payloads"]==104
    assert result["worker_all_checkpoint_payloads"]==7
    assert result["disposable_optimizer_calls_reported"]==111
    assert result["checkpoint_deserializations"]==0 and result["scientific_updates"]==0
    assert not result["scheduler_terminal_verified"] and not result["main_worker_qualification_pass"]
    assert not result["scientific_launch_authority"] and not result["numerical_replay"]
    assert result["total_bytes"]==sum(v["size_bytes"] for v in result["files"].values())


@pytest.mark.parametrize("change",["missing_case","duplicate_case","wrong_calls","hardware","failed_exit","failed_marker","prefix_payload"])
def test_terminal_rejects_incomplete_or_changed_evidence(complete,change):
    parent,eref,_=complete
    if change in ("missing_case","duplicate_case","wrong_calls","hardware"):
        path=parent.root/"numerics/COMPLETE.json";value=read(path)
        if change=="missing_case":value["precision"].pop()
        elif change=="duplicate_case":value["precision"][-1]=value["precision"][0]
        elif change=="wrong_calls":value["disposable_optimizer_calls"]=103
        else:value["hardware"]["name"]="A100"
        nr=rewrite(path,value);execution=read(eref.path);execution["numerics"]=asdict(nr)
        eref=rewrite(eref.path,execution)
    elif change=="failed_exit":rewrite(parent.root/"NUMERICS_EXIT.json",dict(returncode=17,automatic_retry=False))
    elif change=="failed_marker":write_json(parent.root/"numerics/FAILED.json",dict(fixture_only=True))
    else:
        p=parent.root/"numerics"/cases()[0].identifier/"primary/checkpoints/update-000000001/checkpoint.pt"
        p.write_bytes(b"changed opaque fixture")
    with pytest.raises(ValueError):terminal.verify(parent.cref,eref)


@pytest.mark.parametrize("change",["tolerance","summary","gradient_count","channel","phase","source","threshold"])
def test_precision_statistics_checked_not_just_top_level_pass(change):
    case=cases()[0];geometry=dict(expected_geomloss_version="0.3.1")
    value=precision_record(case,"joint","a"*64,"b"*64,geometry)
    if change=="tolerance":value["cpu_cuda"]["parameter_gradients"]["fixture.parameter"]["maximum_tolerance_normalized_error"]=1.01
    elif change=="summary":value["cpu_cuda"]["maximum_gradient_absolute_error"]=1.
    elif change=="gradient_count":value["cpu_cuda"]["compared_gradient_elements"]=5
    elif change=="channel":value["cpu_cuda"]["channels"].pop("objective")
    elif change=="phase":value["phase"]="state"
    elif change=="source":value["source_sha256"]="c"*64
    else:value["tolerances"]["loss_atol"]=.1
    with pytest.raises(ValueError):terminal.verify_precision(value,case,"joint",source="a"*64,rules="b"*64,geometry=geometry)


def test_inventory_detects_additions_and_never_follows_links(tmp_path):
    (tmp_path/"one").write_bytes(b"original")
    inventory=OutputInventory(tmp_path,limit=MIB)
    (tmp_path/"two").write_bytes(b"additional")
    with pytest.raises(ValueError,match="changed"):inventory.verify_exit()
    (tmp_path/"link").symlink_to(tmp_path/"one")
    with pytest.raises(ValueError):OutputInventory(tmp_path,limit=MIB)


def test_inventory_caps_logs_and_total_bytes(tmp_path):
    (tmp_path/"PROCESS.log").write_bytes(b"x"*(4*MIB+1))
    with pytest.raises(ValueError,match="log"):OutputInventory(tmp_path,limit=8*MIB)


def test_parent_does_not_complete_after_readback_failure(parent,mocked_parent,monkeypatch):
    original=producer.run_process
    def fail(command,**kwargs):
        if command[3]=="readback":return 19
        return original(command,**kwargs)
    monkeypatch.setattr(producer,"run_process",fail)
    with pytest.raises(RuntimeError,match="terminal readback failed"):producer.execute(parent.cref,parent.rref)
    assert (parent.root/"EXECUTION_COMPLETE.json").exists()
    assert (parent.root/"FAILED.json").exists() and not (parent.root/"COMPLETE.json").exists()


def test_real_cpu_readback_child_publishes_without_loading_models(complete,monkeypatch):
    parent,eref,_=complete
    # Remove only these newly fabricated test completion records to represent
    # the point just before the real CPU-only readback subprocess is invoked.
    (parent.root/"COMPLETE.json").unlink()
    (parent.root/"TERMINAL_READBACK.json").unlink()
    monkeypatch.setattr(producer.os,"getppid",lambda:read(parent.root/"STARTED.json")["pid"])
    ref=producer.execute_readback(parent.cref,parent.rref,eref)
    value=read(ref.path)
    assert value["checkpoint_deserializations"]==0
    assert value["owner_all_checkpoint_payloads"]+value["worker_all_checkpoint_payloads"]==111
    assert not value["scientific_launch_authority"]
