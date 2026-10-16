"""Metadata-only recovery ledger checks; no optimizer, GPU or remote work."""
import copy
import pytest
from frontier.coordination import AdmissionJournal, WorkBinding, GIB
from frontier.planner import INNER_PAIRS, OUTER_TRIPLES


def original(j, name, donors):
    return j.reserve(name, worker_id=name, kind="representation", scientific_updates=20000,
        output_bytes=4*GIB, admission_sha256="a"*64, capacity_receipt_sha256="b"*64,
        fileset_headroom_bytes=500*GIB, work_binding=WorkBinding.representation(donors))


def replace(j, old="old0", new="new0", **kwargs):
    args=dict(authority_sha256="e"*64, zero_update_evidence_sha256="f"*64,
        admission_sha256="a"*64, capacity_receipt_sha256="d"*64,
        fileset_headroom_bytes=500*GIB)
    args.update(kwargs)
    return j.replace_zero_update_representation(old,new,**args)


def test_four_replacements_preserve_history_storage_and_only_four_live(tmp_path):
    with AdmissionJournal(tmp_path/"ledger",study_identity_sha256="c"*64) as j:
        for i, donors in enumerate(INNER_PAIRS[:4]): original(j,"old"+str(i),donors)
        j.reconcile("old0",status="failed",exact_scheduler_id="old-job",evidence_sha256="b"*64)
        earlier={p.name:p.read_bytes() for p in (tmp_path/"ledger/events").iterdir()}
        for i in range(4): state=replace(j,"old"+str(i),"new"+str(i))
        assert sum(t["status"]=="reserved" for t in state["tasks"].values())==4
        assert sum(t["storage_reserved_bytes"] for t in state["tasks"].values())==32*GIB
        assert sum(t["scientific_update_reservation"] for t in state["tasks"].values()
                   if "replacement_task_id" not in t)==80000
        assert state["tasks"]["old0"]["status"]=="failed"
        assert state["tasks"]["old0"]["evidence_sha256"]=="b"*64
        assert state["tasks"]["old1"]["status"]=="cancelled"
        assert all((tmp_path/"ledger/events"/n).read_bytes()==v for n,v in earlier.items())
        with pytest.raises(ValueError): replace(j,"new0","third0")
        with pytest.raises(ValueError): replace(j,"old0","third0")
    with AdmissionJournal(tmp_path/"ledger",study_identity_sha256="c"*64) as j:
        assert j.snapshot()==state


@pytest.mark.parametrize("status",["submitted","pending","running","uncertain","complete"])
def test_live_or_successful_work_cannot_use_zero_work_replacement(tmp_path,status):
    with AdmissionJournal(tmp_path/"ledger",study_identity_sha256="c"*64) as j:
        original(j,"old0",INNER_PAIRS[0])
        j.reconcile("old0",status=status,exact_scheduler_id="job0",evidence_sha256="b"*64)
        before=j.snapshot()
        with pytest.raises(ValueError): replace(j)
        assert j.snapshot()==before


def test_only_ten_scientific_subsets_not_eleven_attempts_are_charged(tmp_path):
    with AdmissionJournal(tmp_path/"ledger",study_identity_sha256="c"*64) as j:
        for i, donors in enumerate(INNER_PAIRS+OUTER_TRIPLES):
            original(j,"old"+str(i),donors)
            j.reconcile("old"+str(i),status="failed" if i==0 else "complete",
                exact_scheduler_id="j"+str(i),evidence_sha256="b"*64)
        state=replace(j)
        assert len(state["tasks"])==11
        assert sum(t["scientific_update_reservation"] for t in state["tasks"].values()
                   if "replacement_task_id" not in t)==200000
        with pytest.raises(ValueError): original(j,"duplicate",INNER_PAIRS[0])


def test_replacement_still_charges_old_storage_and_requires_authority_hash(tmp_path):
    with AdmissionJournal(tmp_path/"ledger",study_identity_sha256="c"*64) as j:
        original(j,"old0",INNER_PAIRS[0]); before=j.snapshot()
        for kwargs in (dict(fileset_headroom_bytes=23*GIB),dict(authority_sha256=""),dict(output_bytes=5*GIB)):
            with pytest.raises(ValueError): replace(j,**kwargs)
            assert j.snapshot()==before


@pytest.mark.parametrize("damage",["missing_successor","orphan_authority","different_work","changed_failure","second_replacement"])
def test_forged_link_or_transition_rejected(tmp_path,damage):
    with AdmissionJournal(tmp_path/"ledger",study_identity_sha256="c"*64) as j:
        original(j,"old0",INNER_PAIRS[0]); before=j.snapshot()
        state=replace(j); value=copy.deepcopy(state["tasks"])
        action=dict(type="authorized_zero_update_representation_replacement",task_id="new0",
                    predecessor_id="old0",authority_sha256="e"*64,evidence_sha256="f"*64)
        if damage=="missing_successor": del value["new0"]
        elif damage=="orphan_authority": del value["old0"]["replacement_task_id"]
        elif damage=="different_work": value["new0"]["work_binding"]=WorkBinding.representation(INNER_PAIRS[1]).to_dict()
        elif damage=="changed_failure": value["old0"]["capacity_receipt_sha256"]="1"*64
        else: value["old0"]["replaces_task_id"]="new0"
        with pytest.raises(ValueError):
            j._validate(value); j._validate_transition(before["tasks"],value,action)
