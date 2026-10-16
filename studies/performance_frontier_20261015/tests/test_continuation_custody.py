"""Opaque byte-custody fixtures, not numerical forecasts; zero optimizer calls.

Files named .npy below are deliberately labelled opaque byte fixtures. They
test hash/path custody only, not an array decoder or scientific particle count.
Nominal 512-prefix metadata is never presented as a numerical fixture result.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from frontier import continuation_custody as module
from frontier.continuation_custody import (ContinuationForecastAdmission,VerifiedContinuationCustody,
    continuation_edge_ids,verify_continuation_custody)
from frontier.fitting_provider import _hash
from frontier.prediction_plan import required_before_continuation
from test_selection_checkpoint_limit import bounded
from test_selection_evidence import package,stage,write_json


MIB=1024**2


def opaque_custody_files(p):
    plan=json.loads(Path(p.admission.seal_plan.path).read_text())
    seal=json.loads(Path(p.admission.seal.path).read_text())
    for entry,config,row in zip(plan["entries"],plan["configuration_inventory"]["entries"],seal["forecasts"]):
        total=0
        for artifact in row["artifacts"][:6]:
            path=p.root/artifact["relative_uri"];path.parent.mkdir(exist_ok=True)
            raw=("OPAQUE-CUSTODY-FIXTURE-NOT-A-NUMERICAL-FORECAST:"+artifact["relative_uri"]).encode()
            path.write_bytes(raw)
            artifact.update(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
            total+=len(raw)
        entry["complete_sha256"]=row["artifacts"][0]["sha256"]
        entry["manifest_sha256"]=row["artifacts"][1]["sha256"]
        config["derivation_manifest"]["parent_publication"].update(
            complete_sha256=entry["complete_sha256"],manifest_sha256=entry["manifest_sha256"],total_bytes=total)
    plan["expectation"]["configuration_sha256"]=_hash(plan["configuration_inventory"])
    seal["bindings"]=plan["expectation"]
    plan_ref=write_json(Path(p.admission.seal_plan.path),plan)
    seal["forecasts"][0]["artifacts"][-1].update(sha256=plan_ref.sha256,size_bytes=plan_ref.size_bytes)
    seal_ref=write_json(Path(p.admission.seal.path),seal)
    return ContinuationForecastAdmission(p.task.identifier,seal_ref,plan_ref,
        p.admission.checkpoint_complete,4*MIB,4*MIB,True)


@pytest.fixture
def case(tmp_path,request):
    name=getattr(request,"param","screen")
    packages=tuple(package(tmp_path/d,name=name,configuration="F0-V1",selector=d,
        training_pair=("D2","D3"),checkpoint_change=bounded) for d in ("D1","D4"))
    refs=tuple(opaque_custody_files(p) for p in packages)
    # Both exact original checkpoint metadata files have identical content.
    assert refs[0].checkpoint_complete.sha256==refs[1].checkpoint_complete.sha256
    refs=(refs[0],replace(refs[1],checkpoint_complete=refs[0].checkpoint_complete))
    previous=packages[0].plan
    following=stage("family" if name=="screen" else "finalist")
    task=next(t for t in following.training if t.key==packages[0].task.key)
    checkpoint=json.loads(Path(refs[0].checkpoint_complete.path).read_text())
    return dict(previous=previous,task=task,refs=refs,checkpoint=checkpoint,packages=packages)


def verify(case,**changes):
    args=dict(checkpoint_identity=case["checkpoint"]["identity"],
        checkpoint_sha256=case["checkpoint"]["checkpoint_sha256"])
    args.update(changes)
    return verify_continuation_custody(case["previous"],case["task"],case["refs"],**args)


@pytest.mark.parametrize("case",["screen","family"],indirect=True)
@pytest.mark.parametrize("execution_scope", ["synthetic_cpu_only"])
def test_both_directed_banks_are_verified_without_outcomes_or_other_banks(case,monkeypatch,execution_scope):
    original=module._verified_file
    reads=[]
    @contextmanager
    def guarded(ref,**kw):
        path=Path(ref.path)
        assert "assessment-scalars" not in path.parts and path.suffix!=".pt"
        if path.parent.name.startswith("bank-not-created-"):
            assert path.parent.name=="bank-not-created-F0-V1"
        reads.append(ref.path)
        with original(ref,**kw) as stream:yield stream
    monkeypatch.setattr(module,"_verified_file",guarded)
    result=verify(case)
    assert result.edge_ids==required_before_continuation(case["previous"],case["task"])
    r=result.record()
    assert r["selected_artifact_files"]==12 and r["selected_artifact_bytes"]>0
    assert r["outcome_or_score_reads"]==r["checkpoint_tensor_reads"]==r["optimizer_calls"]==0
    assert r["complete_group_metadata_verified"] and not r["unrelated_candidate_payloads_rehashed"]
    assert len(reads)==17  # Two seals, two plans, one checkpoint metadata, twelve bank files.
    assert {x["query_donor"] for x in r["observations"]}=={"D1","D4"}
    assert continuation_edge_ids(result,previous_stage=case["previous"],next_task=case["task"],
        checkpoint_identity=case["checkpoint"]["identity"],checkpoint_sha256=case["checkpoint"]["checkpoint_sha256"],
        execution_scope=execution_scope)==result.edge_ids
    r["edge_ids"].clear()
    assert len(result.edge_ids)==2
    with pytest.raises(TypeError):VerifiedContinuationCustody()


@pytest.mark.parametrize("bad",["missing","duplicate","foreign"])
def test_incomplete_inventory_fails_before_opening_any_file(case,monkeypatch,bad):
    if bad=="missing":case["refs"]=case["refs"][:1]
    elif bad=="duplicate":case["refs"]=(case["refs"][0],)*2
    else:case["refs"]=(replace(case["refs"][0],edge_id="foreign"),case["refs"][1])
    monkeypatch.setattr(module,"_verified_file",lambda *_a,**_k:pytest.fail("No file reads"))
    with pytest.raises(ValueError,match="every required"):verify(case)


@pytest.mark.parametrize("change",["missing","corrupt","symlink"])
def test_changed_selected_artifact_cannot_authorize_continuation(case,change):
    seal=json.loads(Path(case["refs"][1].seal.path).read_text())
    row=next(r for r in seal["forecasts"] if r["forecast_id"]=="fabricated-bank-F0-V1")
    path=Path(case["refs"][1].seal.path).parent/row["artifacts"][2]["relative_uri"]
    if change=="missing":path.unlink()
    elif change=="corrupt":path.write_bytes(b"!"*path.stat().st_size)
    else:
        copy=path.with_suffix(".fixture-copy");path.rename(copy);path.symlink_to(copy)
    with pytest.raises((ValueError,FileNotFoundError)):verify(case)


def test_corruption_during_final_descriptor_exit_invalidates_whole_check(case,monkeypatch):
    original=module._verified_file
    bank_refs=[]
    @contextmanager
    def changed(ref,**kw):
        is_bank=Path(ref.path).parent.name=="bank-not-created-F0-V1"
        if is_bank:bank_refs.append(ref)
        with original(ref,**kw) as stream:
            yield stream
            if is_bank and ref==bank_refs[-1]:
                Path(bank_refs[0].path).touch()
    monkeypatch.setattr(module,"_verified_file",changed)
    with pytest.raises(ValueError,match="changed"):verify(case)


@pytest.mark.parametrize("wrong",["hash","identity","byte_cap"])
def test_checkpoint_and_read_budget_are_checked_before_bank_bytes(case,monkeypatch,wrong):
    original=module._verified_file
    @contextmanager
    def guarded(ref,**kw):
        assert not Path(ref.path).parent.name.startswith("bank-not-created-")
        with original(ref,**kw) as stream:yield stream
    monkeypatch.setattr(module,"_verified_file",guarded)
    if wrong=="hash":args=dict(checkpoint_sha256="f"*64)
    elif wrong=="identity":args=dict(checkpoint_identity=dict(case["checkpoint"]["identity"],seed=1))
    else:
        case["refs"]=tuple(replace(r,maximum_artifact_bytes=1) for r in case["refs"]);args={}
    with pytest.raises(ValueError,match="checkpoint|allowance"):verify(case,**args)


@pytest.mark.parametrize("wrong",["scope","hash","identity","ids"])
def test_custody_cannot_be_reused_for_another_owner_or_scientific_scope(case,wrong):
    custody=verify(case)
    args=dict(previous_stage=case["previous"],next_task=case["task"],
        checkpoint_identity=case["checkpoint"]["identity"],checkpoint_sha256=case["checkpoint"]["checkpoint_sha256"],
        execution_scope="synthetic_cpu_only")
    if wrong=="scope":args["execution_scope"]="scientific"
    elif wrong=="hash":args["checkpoint_sha256"]="f"*64
    elif wrong=="identity":args["checkpoint_identity"]={}
    else:args["supplied_edge_ids"]=("invented",)
    with pytest.raises(ValueError,match="continuing owner"):continuation_edge_ids(custody,**args)
