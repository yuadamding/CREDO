"""Actual compact file preparation; no optimizer, CUDA or biological payloads."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from frontier.main_qualification_artifacts import prepare_artifacts, file_reference, write_json
from frontier.main_qualification_worker_plan import worker_schedule
from frontier.fitting_inputs import FittingInputContract, open_fitting_inputs
from frontier.retention import STUDY_ID
from test_native_inputs import READER


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_updates(monkeypatch,request):
    torch.set_num_threads(2)
    def forbid(*_a,**_k): pytest.fail("No optimizer or CUDA operation authorized")
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD,torch.optim.LBFGS):
        monkeypatch.setattr(cls,"step",forbid)
    monkeypatch.setattr(torch.cuda,"_lazy_init",forbid)
    request.node.user_properties.append(("disposable_cpu_optimizer_calls",0))


@pytest.fixture(params=[("D3","D4"),("D2","D3","D4")])
def artifact(tmp_path,request):
    study=tmp_path / STUDY_ID; study.mkdir()
    root=study / "fixture-fold"
    args=dict(donors=request.param,reader_source_root=READER,
        observation_protocol=(ROOT/"OBSERVATION_PROTOCOL.json").read_bytes(),
        normalization_protocol=(ROOT/"NORMALIZATION_PROTOCOL.json").read_bytes(),
        rules_reference=file_reference(ROOT/"MAIN_OBJECTIVE_RULES.json"),source_sha256="a"*64)
    complete=prepare_artifacts(root,**args)
    return root,args,complete


def test_fabricated_native_to_normalizer_producer_and_actual_consumer(artifact):
    root,args,reference=artifact
    value=json.loads(Path(reference.path).read_bytes())
    assert value["status"]=="fabricated_input_preparation_complete"
    assert value["observation_optimizer_calls"]==value["main_optimizer_calls"]==value["scientific_updates"]==0
    assert not value["real_biological_payload_access"]
    assert value["counts"]==dict(genes=3,catalog=3,source_cells_per_donor=3,endpoint_cells_per_donor=3)
    expected={"screen","family"} if len(args["donors"])==2 else {"outer"}
    assert set(value["inputs"])==expected
    saved=[]
    for stage,ref in value["inputs"].items():
        raw=Path(ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==ref["sha256"]
        contract=FittingInputContract(raw); record=contract.record(); saved.append(record)
        assert contract.components()["schedule"]==worker_schedule()
        assert contract.components()["native_binding"].synthetic_fixture
        assert record["observation"]["trajectory_identity"]["fabricated_observation_without_training"]
        # Exercise the real consumer on local CPU without using the CUDA entry.
        record["schedule"]=asdict(replace(worker_schedule(),execution_scope="synthetic_cpu_only"))
        cpu=write_json(root/("CPU-INPUTS-"+stage+".json"),record)
        with open_fitting_inputs(cpu) as opened:
            p=opened.prepared
            assert p.cache.manifest.fold.training_donors==args["donors"]
            assert len(p.cache.laws)==2*len(args["donors"])
            assert all(x>0 for x in p.selection_scales.values())
            assert p.arguments["decoder"].fold.training_donors==args["donors"]
            assert p.arguments["provider"].position==0
        assert p.cache.verification["exit_full_hash_verified"]
    if len(saved)==2:
        for key in ("observation","native","preparation","normalizers"):
            assert saved[0][key]==saved[1][key]
    assert not (root/"trajectories").exists()


def test_fixture_preparation_cannot_overwrite_or_be_reported_as_training(artifact):
    root,args,reference=artifact
    before=Path(reference.path).read_bytes()
    with pytest.raises(FileExistsError): prepare_artifacts(root,**args)
    assert Path(reference.path).read_bytes()==before
    marker=json.loads((root/"observation/FIXTURE_ONLY.json").read_bytes())
    assert marker["nominal_update_fields_are_fabricated"] and not marker["scientific_selected_model"]
    completion=json.loads((root/"observation/COMPLETE.json").read_bytes())
    assert completion["fixture_only"] and completion["actual_disposable_optimizer_calls"]==0
