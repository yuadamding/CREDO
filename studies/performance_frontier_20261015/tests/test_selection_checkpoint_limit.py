"""Checkpoint schema integration only; no scientific or disposable updates.

The nominal scientific prefix in existing scalar fixtures is fabricated
metadata, not a small numerical run relabelled as a completed scientific fit.
The separate actual producer check publishes a zero-update CPU checkpoint.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontier import selection_evidence as module
from frontier.checkpoints import TrajectoryLease
from frontier.selection_evidence import read_inner_assessment
from test_checkpoints import identity, objects, snapshot
from test_selection_evidence import package


MIB=1024**2


def bounded(record):
    record["maximum_checkpoint_bytes"]=MIB
    record["identity"]["training_owner"]["maximum_checkpoint_bytes"]=MIB


def test_bounded_checkpoint_metadata_is_consumed_without_tensor_reads(tmp_path):
    p=package(tmp_path,checkpoint_change=bounded)
    result=read_inner_assessment(p.admission,stage_plan=p.plan,task=p.task)
    assert len(result.evidence)==24
    assert result.provenance["checkpoint_identity"]["training_owner"]["maximum_checkpoint_bytes"]==MIB
    assert result.provenance["bank_or_outcome_or_checkpoint_tensor_reads"]==0
    assert not list(tmp_path.rglob("*.pt"))


@pytest.mark.parametrize("change",[
    lambda r:r.update(maximum_checkpoint_bytes=True),
    lambda r:r.update(maximum_checkpoint_bytes=0),
    lambda r:r.update(maximum_checkpoint_bytes=2*MIB),
    lambda r:r.pop("maximum_checkpoint_bytes"),
    lambda r:r.update(checkpoint_bytes=MIB+1),
    lambda r:r["identity"]["training_owner"].update(maximum_checkpoint_bytes=True),
    lambda r:r["identity"]["training_owner"].pop("maximum_checkpoint_bytes"),
    lambda r:r.update(unregistered_resource_field=True),
])
def test_changed_producer_limit_rejected_before_saved_scores(tmp_path,monkeypatch,change):
    def altered(r):
        bounded(r);change(r)
    p=package(tmp_path,checkpoint_change=altered)
    monkeypatch.setattr(module,"verify_assessment",lambda *_a,**_k:pytest.fail("No scalar score reads"))
    with pytest.raises(ValueError,match="Checkpoint|checkpoint"):
        read_inner_assessment(p.admission,stage_plan=p.plan,task=p.task)


def test_scientific_metadata_cannot_omit_checkpoint_bound(tmp_path):
    p=package(tmp_path)
    record=json.loads(Path(p.admission.checkpoint_complete.path).read_text())
    record["execution_scope"]="scientific"
    selected=next(e for e in p.saved_plan["entries"]
        if e["model_binding"]["configuration_id"]==p.task.key.configuration_id)
    with pytest.raises(ValueError,match="checkpoint.*bound|Checkpoint.*bound"):
        module._checkpoint(record,SimpleNamespace(fabricated_scalar_metadata=False),p.task,
            module.FoldInputs(**{**p.saved_plan["fold"],"training_donors":p.task.key.training_donors}),
            selected["model_binding"])
    # This direct schema check does not claim that the fabricated metadata is a
    # scientific checkpoint; it must reject the missing production requirement.


def test_actual_zero_update_producer_has_the_same_bounded_receipt_fields(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.Adam,"step",lambda *_a,**_k:pytest.fail("No optimizer calls"))
    values=objects()
    with TrajectoryLease(tmp_path/"zero-update-producer",identity(),"schema-only") as lease:
        saved=snapshot(lease,values,maximum_checkpoint_bytes=MIB,milestone=True)
    actual=json.loads((Path(saved["directory"])/"COMPLETE.json").read_text())
    assert actual["execution_scope"]=="synthetic_cpu_only"
    assert actual["accepted_updates"]==actual["attempted_updates"]==0
    p=package(tmp_path/"nominal-scalar-metadata",checkpoint_change=bounded)
    nominal=json.loads(Path(p.admission.checkpoint_complete.path).read_text())
    assert set(nominal)==set(actual)
    assert 0<actual["checkpoint_bytes"]<=actual["maximum_checkpoint_bytes"]==MIB
    # Keep actual and fabricated receipts separate; only the key inventory is
    # compared. No producer identity, numerical prefix or payload is relabelled.


def test_legacy_fabricated_metadata_remains_compatible(tmp_path):
    p=package(tmp_path)
    assert len(read_inner_assessment(p.admission,stage_plan=p.plan,task=p.task).evidence)==24
