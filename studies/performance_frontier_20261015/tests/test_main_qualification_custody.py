"""Opaque compact-checkpoint metadata tests, not real CUDA checkpoint claims."""

from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

import pytest

from frontier.continuation_custody import verify_continuation_custody,continuation_edge_ids
from frontier.main_qualification_artifacts import file_reference,write_json,stage_task
from frontier.main_qualification_custody import publish_custody
from frontier.main_qualification_worker_plan import cases
from frontier.selection_evidence import _checkpoint,_checkpoint_prefix
from frontier.native_inputs import FoldInputs
from test_selection_evidence import package
from test_selection_checkpoint_limit import bounded


@pytest.fixture
def checkpoint(tmp_path):
    # Explicit fake byte checkpoint, never deserialized or represented as an
    # actual CUDA run. It exercises byte/metadata joins only.
    p=package(tmp_path/"metadata",configuration="F0-V2",training_pair=("D3","D4"),checkpoint_change=bounded)
    path=Path(p.admission.checkpoint_complete.path)
    value=json.loads(path.read_bytes())
    tensor=path.parent/"checkpoint.pt"; tensor.write_bytes(b"OPAQUE-CHECKPOINT-FIXTURE-NO-OPTIMIZER-RUN")
    value.update(accepted_updates=2,attempted_updates=2,execution_scope="synthetic_cuda_qualification",
        checkpoint_sha256=hashlib.sha256(tensor.read_bytes()).hexdigest(),checkpoint_bytes=tensor.stat().st_size)
    path.unlink()  # Only this test's just-created fabricated metadata, not run evidence.
    reference=write_json(path,value)
    return tmp_path,reference,value


def test_compact_custody_keeps_actual_prefix_and_verifies_both_directed_files(checkpoint):
    root,reference,cp=checkpoint
    admissions,receipt=publish_custody(root/"custody",checkpoint_complete=reference)
    previous,_,_=stage_task(cases()[0]); _,_,task=stage_task(cases()[1])
    verified=verify_continuation_custody(previous,task,admissions,checkpoint_identity=cp["identity"],
        checkpoint_sha256=cp["checkpoint_sha256"],qualification_prefix=2)
    assert verified.record()["qualification_checkpoint_updates"]==2
    assert verified.record()["selected_artifact_files"]==12
    assert verified.record()["optimizer_calls"]==0
    assert continuation_edge_ids(verified,previous_stage=previous,next_task=task,checkpoint_identity=cp["identity"],
        checkpoint_sha256=cp["checkpoint_sha256"],execution_scope="synthetic_cuda_qualification")==verified.edge_ids
    for scope in ("scientific","synthetic_cpu_only"):
        with pytest.raises(ValueError): continuation_edge_ids(verified,previous_stage=previous,next_task=task,
            checkpoint_identity=cp["identity"],checkpoint_sha256=cp["checkpoint_sha256"],execution_scope=scope)
    assert json.loads(Path(receipt.path).read_bytes())["numerical_forecasts"]==0
    assert json.loads(Path(reference.path).read_bytes())["accepted_updates"]==2
    with pytest.raises(ValueError): verify_continuation_custody(previous,task,admissions,
        checkpoint_identity=cp["identity"],checkpoint_sha256=cp["checkpoint_sha256"])


@pytest.mark.parametrize("bad",[True,0,3,1024])
def test_no_arbitrary_prefix_mapping(checkpoint,bad):
    root,reference,cp=checkpoint
    admissions,_=publish_custody(root/"custody",checkpoint_complete=reference)
    previous,_,_=stage_task(cases()[0]); _,_,task=stage_task(cases()[1])
    with pytest.raises(ValueError): verify_continuation_custody(previous,task,admissions,
        checkpoint_identity=cp["identity"],checkpoint_sha256=cp["checkpoint_sha256"],qualification_prefix=bad)


def test_changed_checkpoint_or_forecast_bytes_fail(checkpoint):
    root,reference,cp=checkpoint
    admissions,_=publish_custody(root/"custody",checkpoint_complete=reference)
    first=Path(admissions[0].seal.path).parent/"opaque-custody-F0-V2"/"states.npy"
    # Resolve actual retained array filename from the bound seal, not a guess.
    seal=json.loads(Path(admissions[0].seal.path).read_bytes())
    row=next(r for r in seal["forecasts"] if r["forecast_id"]=="fabricated-bank-F0-V2")
    first=Path(admissions[0].seal.path).parent/row["artifacts"][2]["relative_uri"]
    first.write_bytes(first.read_bytes()+b"changed")
    previous,_,_=stage_task(cases()[0]); _,_,task=stage_task(cases()[1])
    with pytest.raises(ValueError): verify_continuation_custody(previous,task,admissions,
        checkpoint_identity=cp["identity"],checkpoint_sha256=cp["checkpoint_sha256"],qualification_prefix=2)


def test_scientific_metadata_cannot_enable_compact_mapping(checkpoint):
    root,reference,cp=checkpoint
    admissions,_=publish_custody(root/"custody",checkpoint_complete=reference)
    previous,_,_=stage_task(cases()[0])
    edge=next(t for t in previous.scores if t.identifier==admissions[0].edge_id)
    with pytest.raises(ValueError): _checkpoint_prefix(replace(admissions[0],fabricated_scalar_metadata=False),edge,2)
    with pytest.raises(FileExistsError): publish_custody(root/"custody",checkpoint_complete=reference)
