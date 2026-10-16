"""Real fabricated-input/custody sequence with mocked subprocesses; no CUDA."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from frontier import main_qualification_sequence as module
from frontier.main_qualification_artifacts import file_reference,prepare_artifacts,write_json
from frontier.main_qualification_worker_plan import cases
from frontier.main_worker_contract import IMAGE,decode_contract
from frontier.observation_loader import ObservationFile
from frontier.retention import STUDY_ID
from test_main_worker import no_numerics
from test_main_worker_qualification import admission
from test_main_qualification_readback import publish_outputs
from test_native_inputs import READER


ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    parent="fabricated-parent"
    root=tmp_path/"qualification-fixtures"/parent/STUDY_ID;root.mkdir(parents=True)
    source=root/"source";source.mkdir()
    source_ref=write_json(root/"SOURCE_MANIFEST.json",dict(fabricated_source_not_runtime_evidence=True))
    inputs={}
    for donors in (("D3","D4"),("D2","D3","D4")):
        ref=prepare_artifacts(root/("fixture-"+"-".join(donors)),donors=donors,
            reader_source_root=READER,observation_protocol=(ROOT/"OBSERVATION_PROTOCOL.json").read_bytes(),
            normalization_protocol=(ROOT/"NORMALIZATION_PROTOCOL.json").read_bytes(),
            rules_reference=file_reference(ROOT/"MAIN_OBJECTIVE_RULES.json"),source_sha256=source_ref.sha256)
        inputs.update({k:ObservationFile(**v) for k,v in json.loads(Path(ref.path).read_bytes())["inputs"].items()})
    monkeypatch.setenv("CREDO_RUN_ID",parent)
    monkeypatch.setenv("CREDO_V4_EXECUTION_IMAGE_DIGEST",IMAGE.split("@")[1])
    seen=[]
    def admit(request):
        # Explicit fake controller records; never leave the mocked CPU test.
        fields=json.loads(Path(request.path).read_bytes())["contract_fields"]
        folder=Path(request.path).parent
        placeholder=ObservationFile(str(folder/"ADMISSION.json"),"0"*64,1)
        record,parts=decode_contract(json.dumps(dict(fields,admission=asdict(placeholder))).encode(),qualification=True)
        aref=write_json(folder/"ADMISSION.json",admission(record,parts))
        cref=write_json(folder/"CONTRACT.json",dict(fields,admission=asdict(aref)))
        release=write_json(folder/"RELEASE.json",dict(fabricated_release_not_launch_authority=True))
        return cref,release
    def child(command,**kwargs):
        assert command[1:4]==["-m","frontier.main_worker","run-qualification"]
        assert "timeout" not in kwargs
        assert kwargs["cwd"]==str(source)
        assert kwargs["env"]["PYTHONPATH"]==str(source)
        assert kwargs["env"]["OMP_NUM_THREADS"]=="8"
        cref=file_reference(command[command.index("--contract")+1])
        record,parts=decode_contract(Path(cref.path).read_bytes(),qualification=True)
        seen.append(parts["qualification_case"].stage)
        publish_outputs(cref,record,parts)
        return 0
    monkeypatch.setattr(module,"run_process",child)
    return root,dict(parent_id=parent,source_root=source,source_manifest=source_ref,inputs=inputs,admit_case=admit),seen,child


def test_three_cases_use_real_byte_custody_but_mocked_gpu_processes(prepared):
    root,arguments,seen,_=prepared
    ref=module.run_sequence(root,**arguments)
    result=json.loads(Path(ref.path).read_bytes())
    assert seen==["screen","family","outer"]
    assert result["disposable_optimizer_calls_reported"]==7  # Reported fake metadata, not actual calls.
    assert result["scientific_updates"]==0 and not result["full_main_qualification_pass"]
    assert not result["scientific_launch_authority"]
    family=json.loads((root/"worker_requests/family/CONTRACT.json").read_bytes())
    assert family["resume"]["accepted_updates"]==2
    assert family["resume"]["execution_scope"]=="synthetic_cuda_qualification"
    assert len(family["continuation_admissions"])==2
    custody=json.loads((root/"opaque-custody/CUSTODY_COMPLETE.json").read_bytes())
    assert custody["verification"]["selected_artifact_files"]==12
    assert custody["numerical_forecasts"]==0
    with pytest.raises(FileExistsError): module.run_sequence(root,**arguments)
    assert seen==["screen","family","outer"]


def test_failed_child_stops_without_resume_or_retry(prepared,monkeypatch):
    root,arguments,seen,child=prepared
    def failed(command,**kwargs):
        if len(seen)==1:
            seen.append("family-failed");return 17
        return child(command,**kwargs)
    monkeypatch.setattr(module,"run_process",failed)
    with pytest.raises(RuntimeError,match="do not retry"): module.run_sequence(root,**arguments)
    assert seen==["screen","family-failed"]
    failure=json.loads((root/"worker_sequence/FAILED.json").read_bytes())
    assert failure["attempted_disposable_call_upper_bound"]==3
    assert failure["actual_failed_case_call_count_requires_worker_receipts"]
    assert not (root/"worker_sequence/COMPLETE.json").exists()
    assert not (root/"worker_requests/outer").exists()


def test_changed_issued_contract_fails_before_process(prepared):
    root,arguments,seen,_=prepared
    original=arguments["admit_case"]
    def wrong(request):
        cref,rref=original(request)
        record=json.loads(Path(cref.path).read_bytes());record["pin_memory"]=True
        other=write_json(Path(cref.path).parent/"WRONG_CONTRACT.json",record)
        return other,rref
    arguments["admit_case"]=wrong
    with pytest.raises(ValueError,match="differs"): module.run_sequence(root,**arguments)
    assert seen==[]


def test_missing_parent_environment_blocks_dispatch(prepared,monkeypatch):
    root,arguments,seen,_=prepared
    monkeypatch.delenv("CREDO_RUN_ID")
    with pytest.raises(ValueError,match="parent Pod"): module.run_sequence(root,**arguments)
    assert seen==[]
