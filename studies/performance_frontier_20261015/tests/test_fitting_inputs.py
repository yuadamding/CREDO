"""Actual fabricated file -> numerical scales -> worker inputs -> stage joins.

Six disposable CPU Adam calls per complete execution; no biological inputs,
GPU, scheduler, actual forecast or scientific fitting. Other cases forbid Adam.
"""

from contextlib import ExitStack
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from frontier import fitting_inputs as module
from frontier.cache_compiler import CompilerLimits
from frontier.checkpoints import TrajectoryLease
from frontier.fitting_inputs import FittingInputContract, open_fitting_inputs
from frontier.fitting_preparation import (FittingPopulation, PreparationLimits, prepare_fitting_cache)
from frontier.fitting_provider import ProviderLimits
from frontier.fitting_stage import FittingStage
from frontier.native_inputs import FoldInputs, NativeInputIndex
from frontier.normalizer_publication import NormalizationLimits, publish_normalizers
from frontier.observation import ObservationFold
from frontier.observation_loader import ObservationFile, STUDY
from frontier.observation_training import ObservationTrainingSpec
from frontier.planner import (DONORS, INNER_PAIRS, OUTER_TRIPLES, TrajectoryKey,
    family_plan, finalist_plan, outer_plan, screen_plan)
from frontier.training_objective import ObjectiveExecution
from frontier.training_owner import OwnerSchedule
from test_native_inputs import native_package, READER
from test_observation_loader import fixture as observation_fixture
from test_normalization import geometry, PROTOCOL


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
ORIGINAL_STEP = torch.optim.Adam.step
TOTAL_CALLS = 0


def encoded(record):
    return json.dumps(record, sort_keys=True, allow_nan=False).encode()


def reference(path):
    raw = path.read_bytes()
    return ObservationFile(str(path), hashlib.sha256(raw).hexdigest(), len(raw))


def published(package, record=None):
    path = package["root"]/"INPUTS.json"
    path.write_bytes(encoded(package["record"] if record is None else record))
    return reference(path)


@pytest.fixture
def package(tmp_path, monkeypatch, request):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_a, **_k: pytest.fail("Optimizer not admitted"))
    study = tmp_path/STUDY; study.mkdir()
    raw, metadata, binding, index = native_package(study/"native", temporal=True)
    donors = getattr(request,"param",("D3","D4"))
    fold = FoldInputs("subset-"+"-".join(donors), donors)
    admission, _ = observation_fixture(study/"observation",
        specification=replace(ObservationTrainingSpec.fixture(), features=3, latent_dim=48),
        observation_fold=ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256),
        input_sha256=index.access(fold, "representation_fit").identity())
    handles = tuple(index.open(raw, fold, role, max_batch_rows=3, max_uncompressed_bytes=MIB,
        max_cached_bytes=MIB, max_output_bytes=64*1024) for role in ("dynamics_source", "dynamics_supervision"))
    provider = ProviderLimits(8*MIB, 8*MIB, 64)
    owner = prepare_fitting_cache(handles, observation_admission=admission,
        populations=tuple(FittingPopulation(d,c,d+"_"+c,"fabricated-culture-unresolved")
            for d in fold.training_donors for c in ("Stim8hr", "Stim48hr")),
        limits=PreparationLimits(CompilerLimits(8*MIB,3,64*1024,2,64*1024), provider,64*MIB,256*1024,3))
    prep = owner.publish(study/"prepared", output_cap_bytes=96*MIB)
    geom = geometry()
    norms = publish_normalizers(prep["root"], study/"normalizers",
        preparation_complete_sha256=prep["complete_sha256"], preparation_output_cap_bytes=96*MIB,
        limits=NormalizationLimits(provider,2*len(donors),MIB,4*MIB), geometry=geom,
        study_id="fabricated-worker-inputs", study_seed=731, protocol_payload=PROTOCOL,
        source_particles=8, block=4)
    plan = screen_plan()
    if len(donors)==3:
        families=family_plan({d:tuple("F"+str(i)+"-V1" for i in range(6)) for d in DONORS})
        finalists=finalist_plan(families,{d:("F0-V1","F1-V1","F4-V1") for d in DONORS})
        plan=outer_plan(finalists)
    task = next(t for t in plan.training if t.key==TrajectoryKey(donors,"F0-V1",0))
    record = dict(schema=module.SCHEMA, study_id="fabricated-worker-inputs", study_seed=731,
        source_sha256="a"*64, task=asdict(task),
        native=dict(metadata_root=str(metadata), reader_source_root=str(READER),
            binding=asdict(binding), metadata_index_sha256=index.identity), observation=asdict(admission),
        preparation=dict(root=prep["root"], complete_sha256=prep["complete_sha256"],
            output_cap_bytes=96*MIB, provider_limits=asdict(provider)),
        normalizers=dict(root=norms["root"], complete_sha256=norms["complete_sha256"],
            maximum_artifact_bytes=MIB, output_cap_bytes=4*MIB),
        rules=asdict(reference(ROOT/"MAIN_OBJECTIVE_RULES.json")),
        schedule=asdict(OwnerSchedule("synthetic_cpu_only",(2,4,6),1,2,32,1,2,4,4)),
        execution=asdict(ObjectiveExecution(1,1,3,4,4,100000,3e-6,3e-6)),
        geometry=asdict(geom.options), maximum_checkpoint_bytes=64*MIB)
    calls=[]
    try:
        yield dict(root=study, record=json.loads(encoded(record)), calls=calls)
    finally:
        request.node.user_properties.append(("disposable_cpu_optimizer_calls",len(calls)))


def forbid_recomputation(monkeypatch):
    monkeypatch.setattr(NativeInputIndex,"open",lambda *_a,**_k:pytest.fail("No count/cell handles"))
    monkeypatch.setattr("frontier.normalization.compute_normalizers",
                        lambda *_a,**_k:pytest.fail("No numerical scale replay"))


@pytest.mark.parametrize("family",["F0","F1","F2","F3","F4","F5"])
def test_serialized_contract_opens_actual_frozen_inputs_for_every_family(package,monkeypatch,family):
    record=copy.deepcopy(package["record"])
    record["task"]["key"]["configuration_id"]=family+"-V1"
    ref=published(package,record)
    forbid_recomputation(monkeypatch)
    with open_fitting_inputs(ref) as opened:
        prepared=opened.prepared
        assert prepared.arguments["model"].config.configuration_id==family+"-V1"
        assert opened.input_contract_sha256==ref.sha256
        assert len(prepared.selection_scales)==9 and len(prepared.scale_binding.values)==5
        assert all(x>0 for x in prepared.selection_scales.values())
        assert prepared.cache.manifest.fold.training_donors==("D3","D4")
        assert prepared.identity["trajectory_id"]==opened.task.key.storage_identifier
        assert prepared.arguments["maximum_checkpoint_bytes"]==64*MIB
        assert prepared.cache._closed is False
    assert prepared.cache._closed is True
    assert prepared.cache.verification["exit_full_hash_verified"] is True
    assert not (package["root"]/"trajectories").exists()


@pytest.mark.parametrize("package",INNER_PAIRS+OUTER_TRIPLES,indirect=True)
def test_all_ten_registered_fitting_subsets_remain_fold_native(package,monkeypatch):
    ref=published(package)
    forbid_recomputation(monkeypatch)
    with open_fitting_inputs(ref) as opened:
        donors=tuple(package["record"]["task"]["key"]["training_donors"])
        assert opened.task.key.training_donors==donors
        assert opened.prepared.cache.manifest.fold.training_donors==donors
        assert opened.prepared.arguments["decoder"].fold.training_donors==donors
        assert set(d for d,_ in opened.prepared.cache.laws)==set(donors)
        assert not set(opened.task.requesting_selectors)&set(donors)


@pytest.mark.parametrize("start,target",[(1024,3072),(3072,9216)])
def test_continuation_input_contract_preserves_same_fitting_identity(package,start,target):
    first=published(package)
    with open_fitting_inputs(first) as original:
        identity=original.prepared.identity
    record=copy.deepcopy(package["record"])
    record["task"].update(start_updates=start,target_updates=target)
    with open_fitting_inputs(published(package,record)) as continued:
        assert continued.prepared.identity==identity
        assert continued.task.start_updates==start and continued.task.target_updates==target
    # This prepares inputs only. No continuation is admitted without the prior
    # numerical completion, forecast custody, selection and worker reservation.


@pytest.mark.parametrize("mutation",[
    lambda r:r.update(unapproved_optimizer="Adam"),
    lambda r:r["native"].update(dataset_root="/not-a-consumed-count-root"),
    lambda r:r["native"]["binding"].update(synthetic_fixture=1),
    lambda r:r["task"]["key"].update(training_donors=["D4","D3"]),
    lambda r:r["task"]["key"].update(training_donors=["D1","D2"]),
    lambda r:r["task"]["key"].update(seed=1),
    lambda r:r["task"].update(target_updates=2000),
    lambda r:r["observation"]["specification"].update(latent_dim=3),
    lambda r:r["schedule"].update(execution_scope="scientific"),
    lambda r:r["schedule"].update(stages=[2,4]),
    lambda r:r.update(maximum_checkpoint_bytes=True),
    lambda r:r["normalizers"].update(root=r["preparation"]["root"]),
    lambda r:r["preparation"]["provider_limits"].update(maximum_host_bytes=9*1024**3),
    lambda r:r["rules"].update(sha256="b"*64),
])
def test_contract_rejects_ambiguity_or_changed_scope_before_index(package,monkeypatch,mutation):
    record=copy.deepcopy(package["record"]);mutation(record)
    ref=published(package,record)
    monkeypatch.setattr(module,"NativeInputIndex",lambda *_a,**_k:pytest.fail("No metadata admission"))
    with pytest.raises((ValueError,TypeError)):
        with open_fitting_inputs(ref):pytest.fail("Invalid inputs opened")


@pytest.mark.parametrize("device,pin",[("cuda:0",False),("meta",False),("cpu",True)])
def test_device_rejection_precedes_metadata_or_cuda(package,monkeypatch,device,pin):
    ref=published(package)
    monkeypatch.setattr(module,"NativeInputIndex",lambda *_a,**_k:pytest.fail("No input read"))
    with pytest.raises(ValueError,match="CPU-only"):
        with open_fitting_inputs(ref,device=device,pin_memory=pin):pytest.fail("Wrong device")


@pytest.mark.parametrize("field",["metadata_index_sha256","normalizer_sha","cache_sha","observation_sha"])
def test_wrong_identity_fails_before_model_construction(package,monkeypatch,field):
    record=copy.deepcopy(package["record"])
    if field=="metadata_index_sha256":record["native"][field]="b"*64
    elif field=="normalizer_sha":record["normalizers"]["complete_sha256"]="b"*64
    elif field=="cache_sha":record["preparation"]["complete_sha256"]="b"*64
    else:record["observation"]["artifact"]["sha256"]="b"*64
    ref=published(package,record)
    monkeypatch.setattr(module,"prepare_fitting_components",lambda **_k:pytest.fail("No model construction"))
    with pytest.raises(ValueError):
        with open_fitting_inputs(ref):pytest.fail("Wrong identity accepted")


@pytest.mark.parametrize("changed",["contract","native","observation","cache"])
def test_input_leases_remain_active_until_caller_exits(package,changed):
    ref=published(package)
    with pytest.raises(ValueError,match="changed"):
        with open_fitting_inputs(ref) as opened:
            record=package["record"]
            path={"contract":Path(ref.path),
                  "native":Path(record["native"]["metadata_root"])/"source/guides.parquet",
                  "observation":Path(record["observation"]["artifact"]["path"]),
                  "cache":Path(record["preparation"]["root"])/"cache/p000.states.npy"}[changed]
            path.touch()
    assert opened.prepared.cache._closed is True


def counted_steps(package,monkeypatch):
    def counted(opt,*a,**kw):
        global TOTAL_CALLS
        TOTAL_CALLS+=1
        if TOTAL_CALLS>6:raise RuntimeError("Six-call fixture budget exhausted")
        package["calls"].append(1)
        return ORIGINAL_STEP(opt,*a,**kw)
    monkeypatch.setattr(torch.optim.Adam,"step",counted)


def run_stage(package,monkeypatch,*,family="F0",change_input=False):
    record=copy.deepcopy(package["record"])
    record["task"]["key"]["configuration_id"]=family+"-V1"
    ref=published(package,record)
    counted_steps(package,monkeypatch);forbid_recomputation(monkeypatch)
    (package["root"]/"trajectories").mkdir()
    with ExitStack() as outer:
        with open_fitting_inputs(ref) as admitted:
            lease=outer.enter_context(TrajectoryLease(package["root"]/"trajectories"/admitted.task.key.storage_identifier,
                admitted.prepared.identity,"fabricated-input-owner"))
            stage=outer.enter_context(FittingStage(prepared=admitted.prepared,lease=lease,plan=screen_plan(),
                task=admitted.task,study_root=package["root"]))
            stage.run()
            if change_input:Path(ref.path).touch()
    return stage


@pytest.mark.parametrize("family",["F0","F4"])
def test_actual_numerical_normalizers_to_two_update_stage(package,monkeypatch,family):
    stage=run_stage(package,monkeypatch,family=family)
    assert len(package["calls"])==2 and stage.receipt
    result=json.loads(stage.path.read_text())
    assert result["scientific_updates_this_task"]==0
    assert result["optimizer_calls_this_task"]==2
    assert result["cache_verification"]["exit_full_hash_verified"] is True
    assert stage.owner.status=="stage_complete"


def test_post_training_contract_mutation_invalidates_stage(package,monkeypatch):
    with pytest.raises(ValueError,match="changed"):
        run_stage(package,monkeypatch,change_input=True)
    assert len(package["calls"])==2
    folder=package["root"]/"trajectories"/"D3-D4--F0-V1--seed0"
    assert not (folder/"FITTING_STAGE_screen.json").exists()
    assert (folder/"FITTING_STAGE_screen_FAILED.json").exists()
    assert json.loads((folder/"TRAINING_OWNER_STATE.json").read_text())["status"]=="failed"


def test_contract_rejects_duplicate_keys_and_does_not_share_mutable_record(package):
    raw=encoded(package["record"])
    with pytest.raises(ValueError,match="Duplicate"):
        FittingInputContract(b'{"schema":"duplicate",'+raw[1:])
    contract=FittingInputContract(raw)
    record=contract.record();record["task"]["key"]["seed"]=2
    assert contract.components()["task"].key.seed==0
    with pytest.raises(ValueError,match="bounded|Bounded"):
        FittingInputContract(b" "*(module.MAXIMUM_CONTRACT_BYTES+1))
