"""Parent control flow with explicitly mocked GPU work; zero optimizer calls."""

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier import main_qualification_worker as module
from frontier import main_qualification_contract as contract
from frontier import main_qualification_sequence as sequence_module
from frontier.main_qualification_artifacts import file_reference, write_json
from frontier.main_worker_contract import IMAGE, decode_contract
from frontier.main_worker_source import source_closure, manifest_payload, QUALIFICATION_ENTRY
from frontier.coordination import WorkBinding
from frontier.main_qualification_plan import plan_identity, cases
from frontier.retention import STUDY_ID, identity_sha256
from test_main_worker import no_numerics
from test_main_qualification_readback import publish_outputs
from test_native_inputs import READER


ROOT=Path(__file__).resolve().parents[1]
GIB=1024**3


@pytest.fixture
def parent(tmp_path,monkeypatch):
    root=tmp_path/STUDY_ID; bundle=root/"qualification-v1/bundle";bundle.mkdir(parents=True)
    source=source_closure(ROOT,entry=QUALIFICATION_ENTRY)
    for name,raw in source.items():
        path=bundle/"source"/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    refs={}
    for label,raw in (("source_manifest",manifest_payload(source,entry=QUALIFICATION_ENTRY)),
            ("worker_source_manifest",manifest_payload(source_closure(ROOT))),
            ("observation_protocol",(ROOT/"OBSERVATION_PROTOCOL.json").read_bytes()),
            ("normalization_protocol",(ROOT/"NORMALIZATION_PROTOCOL.json").read_bytes()),
            ("rules",(ROOT/"MAIN_OBJECTIVE_RULES.json").read_bytes())):
        path=bundle/(label+".json");path.write_bytes(raw);refs[label]=file_reference(path)
    for name in module.READER_SOURCE_HASHES:
        path=bundle/"retained_reader"/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((Path(READER)/name).read_bytes())
    runtime=dict(python="3.12.13",torch="2.6.0+cu124",cuda_runtime="12.4",numpy="2.5.1",scipy="1.18.0")
    authority=dict(schema="credo.frontier.main-qualification-authority.v1",status="authorized",run_id="test-parent",
        plan_sha256=plan_identity(),settings_sha256=identity_sha256(contract.settings()),
        source_manifest_sha256=refs["source_manifest"].sha256,
        worker_source_manifest_sha256=refs["worker_source_manifest"].sha256,
        planned_disposable_optimizer_calls=111,maximum_disposable_optimizer_calls=112,scientific_updates=0,
        real_biological_payload_access=False,main_scientific_fitting=False,automatic_retry=False,
        automatic_successor=False,elapsed_time_deadline=None,fixture_only_not_real_authority=True)
    refs["authority"]=write_json(bundle/"AUTHORITY.json",authority)
    admission=dict(schema="credo.frontier.main-qualification-admission.v1",status="reserved",run_id="test-parent",
        work_binding=WorkBinding.other("qualification",plan_identity()).to_dict(),
        phase_authority_sha256=refs["authority"].sha256,source_manifest_sha256=refs["source_manifest"].sha256,
        worker_source_manifest_sha256=refs["worker_source_manifest"].sha256,scientific_update_reservation=0,
        storage_reserved_bytes=16*GIB,disposable_optimizer_call_limit=112,automatic_retry=False,automatic_successor=False,
        journal_event_sha256="a"*64,journal_identity_sha256="b"*64,capacity_receipt_sha256="c"*64,
        aggregate_reserved_output_bytes=80*GIB,live_h100_reservations=4)
    refs["admission"]=write_json(bundle/"ADMISSION.json",admission)
    record=dict(schema=contract.SCHEMA,run_id="test-parent",phase="qualification-v1",study_root=str(root),
        output_root=str(bundle.parent/"output"),source_root=str(bundle/"source"),
        reader_source_root=str(bundle/"retained_reader"),runtime_image=IMAGE,expected_runtime=runtime,
        plan_sha256=plan_identity(),settings=contract.settings(),**{k:asdict(v) for k,v in refs.items()})
    cref=write_json(bundle/"CONTRACT.json",record)
    release=dict(schema="credo.frontier.main-qualification-startup-release.v1",status="startup_released",
        run_id=record["run_id"],worker_contract_sha256=cref.sha256,pod_name="test-pod",pod_uid="test-pod-uid",
        image=IMAGE,admission_sha256=refs["admission"].sha256,source_manifest_sha256=refs["source_manifest"].sha256,
        phase_authority_sha256=refs["authority"].sha256,journal_event_sha256=admission["journal_event_sha256"],
        capacity_receipt_sha256=admission["capacity_receipt_sha256"],actual_pod_audit_passed=True,
        storage_and_fileset_quota_passed=True,aggregate_reservation_verified=True,cgroup_resources_verified=True,
        delegate_fixed_worker_cases=True,disposable_qualification_authorized=True,scientific_fit_authorized=False,
        real_biological_payload_access=False,authorized_cpus=24,authorized_memory_bytes=200*GIB,threads=8,gpus=1,
        automatic_retry=False,automatic_successor=False,startup_sha256="d"*64,job_uid="test-job-uid")
    rref=write_json(bundle/"RELEASE.json",release)
    env=dict(CREDO_RUN_ID=record["run_id"],CREDO_POD_NAME="test-pod",CREDO_POD_UID="test-pod-uid",
             CREDO_V4_EXECUTION_IMAGE_DIGEST=IMAGE.split("@")[1])
    for k,v in env.items():monkeypatch.setenv(k,v)
    return SimpleNamespace(record=record,refs=refs,authority=authority,admission=admission,release=release,
        cref=cref,rref=rref,env=env,runtime=runtime,root=Path(record["output_root"]))


def validate(p,**changes):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os,"getuid",lambda:307469);patch.setattr(os,"getgid",lambda:600815)
        arguments=dict(record=p.record,refs=p.refs,contract_reference=p.cref,authority=p.authority,
            admission=p.admission,release=p.release,environment=p.env,runtime=p.runtime)
        arguments.update(changes);contract.validate_parent(**arguments)


def test_exact_parent_scope_and_inspection_are_nonnumerical(parent,capsys):
    record,refs=contract.decode(Path(parent.cref.path).read_bytes())
    assert refs==parent.refs and record==parent.record
    validate(parent)
    assert module.main(["inspect","--contract",parent.cref.path,"--contract-sha256",parent.cref.sha256])==0
    assert "no launch authority" in capsys.readouterr().out
    assert not parent.root.exists()


@pytest.mark.parametrize("section,key,value",[
    ("authority","schema","credo.frontier.observation-authority.v1"),
    ("authority","scientific_updates",1),("authority","planned_disposable_optimizer_calls",112),
    ("authority","real_biological_payload_access",True),("authority","automatic_retry",True),
    ("admission","live_h100_reservations",5),("admission","storage_reserved_bytes",17*GIB),
    ("admission","aggregate_reserved_output_bytes",385*GIB),("admission","scientific_update_reservation",1),
    ("release","delegate_fixed_worker_cases",False),("release","scientific_fit_authorized",True),
    ("release","pod_uid","other"),("release","threads",True),("runtime","torch","unqualified")])
def test_unapproved_scope_rejected(parent,section,key,value):
    with pytest.raises(ValueError):validate(parent,**{section:dict(getattr(parent,section),**{key:value})})


@pytest.mark.parametrize("change",["path","source","plan","image","settings","runtime"])
def test_contract_wrong_paths_and_bindings(parent,change):
    value=json.loads(Path(parent.cref.path).read_bytes())
    if change=="path":value["output_root"]=str(parent.root.parent)
    elif change=="source":value["source_root"]=str(ROOT)
    elif change=="plan":value["plan_sha256"]="0"*64
    elif change=="image":value["runtime_image"]="unqualified"
    elif change=="settings":value["settings"]["scientific_updates"]=1
    else:value["expected_runtime"]["torch"]=""
    with pytest.raises(ValueError):contract.decode(json.dumps(value).encode())


@pytest.fixture
def mocked_numerics(monkeypatch):
    calls=[]
    monkeypatch.setattr(module,"_configure_cuda",lambda:dict(fixture_only_not_cuda=True))
    monkeypatch.setattr(module,"_resources",lambda _device:dict(fixture_only_not_measurement=True))
    def precision(case,phase,**_):
        calls.append((case.identifier,phase))
        return dict(status="passed",fixture_only_no_gpu=True,optimizer_calls=0)
    def owner(case,root,*,budget,**_):
        budget.admit(case);budget.calls+=case.optimizer_calls;budget.completed_cases.add(case.identifier)
        root.mkdir();return write_json(root/"OWNER_CASE_COMPLETE.json",dict(fixture_only_no_gpu=True))
    monkeypatch.setattr(module,"run_paired_precision_case",precision)
    monkeypatch.setattr(module,"run_owner_case",owner)
    return calls


def test_numerical_sequence_accounting_with_mocked_kernels(parent,mocked_numerics):
    parent.root.mkdir()
    result=module.run_numerics(parent.root,references=parent.refs,
        payloads={k:Path(v.path).read_bytes() for k,v in parent.refs.items()})
    saved=json.loads(Path(result.path).read_bytes())
    assert len(mocked_numerics)==96 and len(saved["owners"])==48
    assert saved["disposable_optimizer_calls"]==104 and saved["scientific_updates"]==0
    assert not saved["scientific_launch_authority"] and not saved["full_worker_entry_verified"]
    with pytest.raises(FileExistsError):module.run_numerics(parent.root,references=parent.refs,payloads={})


def test_numerical_precision_failure_prevents_owner_and_retry(parent,mocked_numerics,monkeypatch):
    parent.root.mkdir()
    monkeypatch.setattr(module,"run_paired_precision_case",lambda *_a,**_k:dict(status="failed"))
    monkeypatch.setattr(module,"run_owner_case",lambda *_a,**_k:pytest.fail("No owner after precision failure"))
    with pytest.raises(RuntimeError,match="precision"):
        module.run_numerics(parent.root,references=parent.refs,
            payloads={k:Path(v.path).read_bytes() for k,v in parent.refs.items()})
    failed=json.loads((parent.root/"numerics/FAILED.json").read_bytes())
    assert failed["actual_owner_optimizer_calls"]==0
    assert not (parent.root/"numerics/COMPLETE.json").exists()


@pytest.fixture
def mocked_parent(parent,monkeypatch,mocked_numerics):
    @contextmanager
    def sources(*_a,**_k):yield {}  # Source leases have separate fresh-process tests.
    monkeypatch.setattr(module,"verified_sources",sources)
    original=module.validate_parent
    def parent_check(*args):
        with pytest.MonkeyPatch.context() as p:
            p.setattr(os,"getuid",lambda:307469);p.setattr(os,"getgid",lambda:600815)
            original(*args)
    monkeypatch.setattr(module,"validate_parent",parent_check)
    monkeypatch.setattr(module,"_runtime",lambda:parent.runtime)
    issue=module.issue_case
    def child_admission(*args,**kwargs):
        with pytest.MonkeyPatch.context() as p:
            p.setattr(os,"getuid",lambda:307469);p.setattr(os,"getgid",lambda:600815)
            return issue(*args,**kwargs)
    monkeypatch.setattr(module,"issue_case",child_admission)
    seen=[]
    def numerical_process(command,**kwargs):
        if command[3]=="readback":
            # Parent-flow fixture only; the real semantic reader has its own
            # complete inventory tests. No fake publication is deployable.
            seen.append("readback")
            eref=file_reference(command[command.index("--execution")+1])
            write_json(parent.root/"TERMINAL_READBACK.json",dict(
                schema="credo.frontier.main-qualification-terminal-readback.v1",
                status="complete_numerical_publication_inventory_verified",contract=asdict(parent.cref),
                execution=asdict(eref),disposable_optimizer_calls_reported=111,scientific_updates=0,
                scheduler_terminal_verified=False,main_worker_qualification_pass=False,scientific_launch_authority=False,
                fixture_only_not_real_readback=True))
            return 0
        assert command[1:4]==["-m","frontier.main_qualification_worker","numerics"]
        assert "timeout" not in kwargs
        seen.append("numerics")
        module.run_numerics(parent.root,references=parent.refs,
            payloads={k:Path(v.path).read_bytes() for k,v in parent.refs.items()})
        return 0
    def worker_process(command,**kwargs):
        cref=file_reference(command[command.index("--contract")+1])
        record,parts=decode_contract(Path(cref.path).read_bytes(),qualification=True)
        seen.append(parts["qualification_case"].stage)
        publish_outputs(cref,record,parts)
        return 0
    monkeypatch.setattr(module,"run_process",numerical_process)
    monkeypatch.setattr(sequence_module,"run_process",worker_process)
    return seen


def test_parent_real_file_sequence_mocked_gpu_and_immutable_outputs(parent,mocked_parent):
    result=module.execute(parent.cref,parent.rref)
    value=json.loads(Path(result.path).read_bytes())
    assert mocked_parent==["numerics","screen","family","outer","readback"]
    assert value["reported_disposable_optimizer_calls"]==111 and value["scientific_updates"]==0
    assert not value["main_worker_qualification_pass"] and not value["scientific_launch_authority"]
    assert value["all_parent_source_and_metadata_exits_verified"]
    assert value["worker_source_manifest_sha256"]==parent.refs["worker_source_manifest"].sha256
    with pytest.raises(ValueError,match="Absent output"):module.execute(parent.cref,parent.rref)
    assert not (parent.root/"FAILED.json").exists()


def test_parent_stops_after_child_failure(parent,mocked_parent,monkeypatch):
    monkeypatch.setattr(module,"run_process",lambda *_a,**_k:17)
    with pytest.raises(RuntimeError,match="Numerical child failed"):module.execute(parent.cref,parent.rref)
    assert not (parent.root/"qualification-fixtures").exists()
    assert not (parent.root/"COMPLETE.json").exists()
    assert json.loads((parent.root/"FAILED.json").read_bytes())["actual_disposable_calls_require_component_receipts"]


def test_parent_source_exit_failure_prevents_completion(parent,mocked_parent,monkeypatch):
    @contextmanager
    def changed(*_a,**_k):
        yield {}
        raise ValueError("fabricated source exit change")
    monkeypatch.setattr(module,"verified_sources",changed)
    with pytest.raises(ValueError,match="source exit change"):module.execute(parent.cref,parent.rref)
    assert mocked_parent==["numerics","screen","family","outer"]
    assert (parent.root/"FAILED.json").exists() and not (parent.root/"COMPLETE.json").exists()
