"""Replacement batch export on disposable metadata only; never real journals."""
import importlib
import json
from pathlib import Path
import sys
import pytest
from frontier.coordination import AdmissionJournal, WorkBinding, GIB


@pytest.fixture
def recovery(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"operations"))
    m=importlib.import_module("recover_observation_fits")
    root=tmp_path/"credo_performance_frontier_20261015_v1"; root.mkdir()
    monkeypatch.setattr(m,"ROOT",root)
    for name in (m.AUTHORITY,m.REQUEST,"AUTHORITY.json","capacity.json"):
        (root/name).write_text("{}")
    e=dict(run_id="qual",phase_authority_sha256="a"*64,source_manifest_sha256="b"*64,
        qualification_sha256="c"*64,qualified_hardware={},qualified_peak_reserved_bytes=123)
    rows=[]; checks=[]
    with AdmissionJournal(root/"admission",study_identity_sha256=m.sha(root/"AUTHORITY.json")) as j:
        j.reserve("qual",worker_id="qual",kind="qualification",scientific_updates=0,output_bytes=16*GIB,
            admission_sha256="a"*64,capacity_receipt_sha256="b"*64,fileset_headroom_bytes=200*GIB,
            work_binding=WorkBinding.other("qualification","f"*64))
        j.reconcile("qual",status="complete",exact_scheduler_id="q-job",evidence_sha256="d"*64)
        for i,donors in enumerate(m.SUBSETS):
            old,new="old"+str(i),"new"+str(i); work=WorkBinding.representation(donors)
            j.reserve(old,worker_id=old,kind="representation",scientific_updates=20000,output_bytes=4*GIB,
                admission_sha256="a"*64,capacity_receipt_sha256="b"*64,fileset_headroom_bytes=200*GIB,work_binding=work)
            if i==0:j.reconcile(old,status="failed",exact_scheduler_id="old-job",evidence_sha256="d"*64)
            row=dict(run_id=new,predecessor_id=old,training_donors=donors,phase="fake"+str(i),
                predecessor_plan_sha256="e"*64,work_binding=work,fit_bindings={"fabricated":True})
            rows.append(row)
            checks.append(dict(predecessor_id=old,replacement_id=new,old_job_absent=True,old_owned_pods=0,
                new_job_absent=True,new_owned_pods=0,old_scientific_output_absent=True,new_roots_absent=True,
                predecessor_plan_sha256="e"*64))
    proof=dict(status="passed",authority_sha256=m.sha(root/m.AUTHORITY),request_sha256=m.sha(root/m.REQUEST),
        remote_writes=False,failure_log_sha256="9"*64,checks=checks)
    (root/"intake.json").write_text(json.dumps(proof))
    monkeypatch.setattr(m,"bindings",lambda:(e,{},rows,[],{"sha256":"9"*64}))
    monkeypatch.setattr(m,"_capacity",lambda *_a:200*GIB)
    monkeypatch.setattr(m,"remote",lambda *_a,**_k:pytest.fail("No remote calls"))
    return m,root


def test_complete_batch_export_and_no_repeated_reservation(recovery):
    m,r=recovery
    result=m.reserve(r/"intake.json",r/"capacity.json",r/"output")
    assert result["live_h100_reservations"]==4 and result["scientific_reservation_transferred"]==80000
    assert result["aggregate_reserved_output_bytes"]==48*GIB  # one qualification in this fixture
    assert result["additional_scientific_trajectories"]==result["scheduler_calls"]==0
    for row in result["fits"]:
        a=json.loads(Path(row["admission"]["path"]).read_text())
        assert a["storage_reserved_bytes"]==4*GIB and a["scientific_update_reservation"]==20000
        assert a["journal_event_sha256"]==result["journal_event_sha256"]
        assert a["replacement_authority_sha256"]==m.sha(r/m.AUTHORITY)
    with pytest.raises(ValueError): m.reserve(r/"intake.json",r/"capacity.json",r/"again")
    assert not (r/"again").exists()


@pytest.mark.parametrize("damage",["authority","old_job","new_root","work","capacity"])
def test_bad_intake_or_capacity_never_mutates_journal(recovery,monkeypatch,damage):
    m,r=recovery
    before={p.name:p.read_bytes() for p in (r/"admission/events").iterdir()}
    p=r/"intake.json"; x=json.loads(p.read_text())
    if damage=="authority":x["authority_sha256"]="1"*64
    elif damage=="old_job":x["checks"][0]["old_job_absent"]=False
    elif damage=="new_root":x["checks"][1]["new_roots_absent"]=False
    elif damage=="work":x["checks"][2]["predecessor_plan_sha256"]="1"*64
    else:monkeypatch.setattr(m,"_capacity",lambda *_a:50*GIB)
    p.write_text(json.dumps(x))
    with pytest.raises(ValueError):m.reserve(p,r/"capacity.json",r/"output")
    assert not (r/"output").exists()
    assert before=={p.name:p.read_bytes() for p in (r/"admission/events").iterdir()}
