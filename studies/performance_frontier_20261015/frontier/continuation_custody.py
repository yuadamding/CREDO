"""Verify the actual forecast bytes required before shared weights advance.

No scores, outcomes, checkpoint tensors, model, optimizer or scheduler are used.
Complete group metadata is checked, but only the continuing model's six files
per directed query are streamed. Other candidates need not be rehashed for each
continuation. This is a custody boundary, not numerical forecast qualification,
selection approval, future retention or resource/launch authority.
"""

from contextlib import ExitStack
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path

from .prediction_plan import required_before_continuation
from .retention import identity_sha256
from .selection_evidence import (SelectionMetadataFile,_checkpoint,_encode,_parse,
    _positive,_seal_metadata,_sha)
from .observation_loader import _verified_file


SCHEMA="credo.frontier.continuation-forecast-custody.v1"
MIB=1024**2
GIB=1024**3


@dataclass(frozen=True)
class ContinuationForecastAdmission:
    edge_id:str
    seal:SelectionMetadataFile
    seal_plan:SelectionMetadataFile
    checkpoint_complete:SelectionMetadataFile
    maximum_metadata_bytes:int
    maximum_artifact_bytes:int
    fabricated_scalar_metadata:bool=False

    def __post_init__(self):
        if type(self.edge_id) is not str or not self.edge_id:
            raise ValueError("Exact previous directed edge required")
        if type(self.fabricated_scalar_metadata) is not bool:
            raise ValueError("Explicit fabricated/real custody scope required")
        _positive(self.maximum_metadata_bytes);_positive(self.maximum_artifact_bytes)
        if (self.maximum_metadata_bytes>32*MIB or
                self.maximum_artifact_bytes>(GIB if self.fabricated_scalar_metadata else 384*GIB)):
            raise ValueError("Bounded continuation custody byte allowances required")
        refs=(self.seal,self.seal_plan,self.checkpoint_complete)
        if any(type(r) is not SelectionMetadataFile for r in refs):
            raise TypeError("Exact seal, plan and checkpoint-metadata references required")
        for ref in refs:ref.__post_init__()
        if (sum(r.size_bytes for r in refs)>self.maximum_metadata_bytes
                or self.seal_plan.path!=self.seal.path+".plan.json"
                or Path(self.checkpoint_complete.path).name!="COMPLETE.json"
                or len({r.path for r in refs})!=3):
            raise ValueError("Distinct complete bounded continuation metadata required")


@dataclass(frozen=True,init=False)
class VerifiedContinuationCustody:
    _payload:bytes

    def __init__(self,*_a,**_k):
        raise TypeError("Use verify_continuation_custody; edge IDs are not file custody")

    @property
    def identity(self):
        return hashlib.sha256(self._payload).hexdigest()

    @property
    def edge_ids(self):
        return tuple(self.record()["edge_ids"])

    def record(self):
        return json.loads(self._payload)


def continuation_edge_ids(custody,*,previous_stage,next_task,checkpoint_identity,
                          checkpoint_sha256,execution_scope,supplied_edge_ids=()):
    """Bind already verified bytes to the exact owner about to resume."""
    if type(custody) is not VerifiedContinuationCustody:
        raise TypeError("Actual verified continuation custody required")
    if execution_scope not in ("scientific","synthetic_cpu_only","synthetic_cuda_qualification"):
        raise ValueError("Exact continuing-owner execution scope required")
    record=custody.record()
    qualification_prefix = 2 if execution_scope == "synthetic_cuda_qualification" else None
    if (record["task"]!=json.loads(_encode(asdict(next_task)))
            or record["previous_stage_sha256"]!=identity_sha256(asdict(previous_stage))
            or record["checkpoint_identity"]!=checkpoint_identity
            or record["checkpoint_sha256"]!=checkpoint_sha256
            or record["fabricated_scalar_metadata"] is not (execution_scope!="scientific")
            or record.get("qualification_checkpoint_updates") != qualification_prefix
            or (supplied_edge_ids and supplied_edge_ids!=custody.edge_ids)
            or custody.edge_ids!=required_before_continuation(previous_stage,next_task)):
        raise ValueError("Verified forecast custody differs from the continuing owner/task")
    return custody.edge_ids


def verify_continuation_custody(previous_stage,next_task,admissions,*,
                                checkpoint_identity,checkpoint_sha256,qualification_prefix=None):
    """Consume caller-admitted exact references before any owner continuation.

    The expected checkpoint must come from the accepted prior owner boundary,
    not be inferred from whichever forecast happens to be available. Verify
    every prior directed query even if only one selector promotes the model.
    Metadata and selected-bank file leases coexist until the whole check exits.
    """
    required=required_before_continuation(previous_stage,next_task)
    if qualification_prefix is not None:
        from .planner import screen_plan
        if (type(qualification_prefix) is not int or qualification_prefix != 2
                or previous_stage != screen_plan()
                or next_task.key.training_donors != ("D3", "D4")
                or next_task.key.configuration_id != "F0-V2" or next_task.key.seed != 0
                or (next_task.start_updates, next_task.target_updates) != (1024, 3072)):
            raise ValueError("Only the fixed compact worker continuation may map its checkpoint prefix")
    _sha(checkpoint_sha256)
    if (type(admissions) is not tuple or len(admissions)!=len(required)
            or any(type(a) is not ContinuationForecastAdmission for a in admissions)
            or tuple(sorted(a.edge_id for a in admissions))!=required):
        raise ValueError("Exactly every required prior directed forecast must be admitted")
    for item in admissions:item.__post_init__()
    if len({a.fabricated_scalar_metadata for a in admissions})!=1:
        raise ValueError("One actual or fabricated custody scope required")
    if len({a.checkpoint_complete for a in admissions})!=1:
        raise ValueError("Both query forecasts must bind the same prior checkpoint metadata")
    # Freeze the caller's expected identity before opening any supplied file.
    expected_identity=_parse(_encode(checkpoint_identity))
    scopes=admissions[0].fabricated_scalar_metadata
    artifact_cap=sum(a.maximum_artifact_bytes for a in admissions)
    if artifact_cap>(GIB if scopes else 384*GIB):
        raise ValueError("Combined continuation artifact-read allowance exceeded")
    tasks={t.identifier:t for t in previous_stage.scores}
    by_edge={a.edge_id:a for a in admissions}
    observations=[]
    artifact_paths=set()
    with ExitStack() as stack:
        # Hold each distinct small metadata file once, including the shared
        # checkpoint COMPLETE record; never load the checkpoint tensor payload.
        held={}
        def read_metadata(ref,maximum):
            key=(ref.path,ref.sha256,ref.size_bytes)
            if key not in held:
                stream=stack.enter_context(_verified_file(ref,maximum=maximum))
                held[key]=_parse(stream.read())
            return held[key]
        pending=[]
        for edge_id in required:
            item=by_edge[edge_id];task=tasks[edge_id]
            seal,plan,checkpoint=(read_metadata(r,item.maximum_metadata_bytes)
                for r in (item.seal,item.seal_plan,item.checkpoint_complete))
            (entry,config,parent),fold=_seal_metadata(item,seal,plan,previous_stage,task,
                qualification_prefix=qualification_prefix)
            binding=entry["model_binding"]
            actual_identity=_checkpoint(checkpoint,item,task,fold,binding,
                qualification_prefix=qualification_prefix)
            if (actual_identity!=expected_identity
                    or checkpoint["checkpoint_sha256"]!=checkpoint_sha256
                    or binding["model_artifact_sha256"]!=checkpoint_sha256):
                raise ValueError("Prior forecast is not from the exact continuing checkpoint")
            row=next(r for r in seal["forecasts"] if r["forecast_id"]==entry["forecast_id"])
            # _seal_metadata enforces six exact bank files, with the readable
            # group plan referenced only once. That plan is already held above.
            members=row["artifacts"][:6]
            total=sum(a["size_bytes"] for a in members)
            if total>item.maximum_artifact_bytes:
                raise ValueError("Selected forecast exceeds its artifact-read allowance")
            refs=[]
            for a in members:
                path=str(Path(item.seal.path).parent/a["relative_uri"])
                if path in artifact_paths:
                    raise ValueError("Different directed queries cannot reuse the same bank path")
                artifact_paths.add(path)
                refs.append(SelectionMetadataFile(path,a["sha256"],a["size_bytes"]))
            pending.append((item,refs))
            observations.append(dict(edge_id=edge_id,forecast_id=entry["forecast_id"],
                query_donor=task.validation_donor,permitted_selector=task.selector,
                seal=asdict(item.seal),seal_plan=asdict(item.seal_plan),
                checkpoint_complete=asdict(item.checkpoint_complete),
                artifact_files=[asdict(r) for r in refs],artifact_bytes=total,
                parent_publication=parent,model_binding=binding))
        # All metadata and combined sizes must pass before reading bank bytes.
        for item,refs in pending:
            for ref in refs:
                stack.enter_context(_verified_file(ref,maximum=item.maximum_artifact_bytes))
    record=dict(schema=SCHEMA,status="verified_at_continuation_boundary",
        task=asdict(next_task),previous_stage_sha256=identity_sha256(asdict(previous_stage)),
        checkpoint_identity=expected_identity,checkpoint_sha256=checkpoint_sha256,
        edge_ids=list(required),observations=observations,
        metadata_bytes=sum(key[2] for key in held),
        selected_artifact_bytes=sum(r["artifact_bytes"] for r in observations),
        selected_artifact_files=len(artifact_paths),
        fabricated_scalar_metadata=scopes,complete_group_metadata_verified=True,
        qualification_checkpoint_updates=qualification_prefix,
        unrelated_candidate_payloads_rehashed=False,outcome_or_score_reads=0,
        checkpoint_tensor_reads=0,optimizer_calls=0,forecast_regeneration=False,
        prediction_numerics_qualified_here=False,ongoing_retention_guaranteed=False,
        scientific_launch_authority=False,automatic_successor=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result=object.__new__(VerifiedContinuationCustody)
    object.__setattr__(result,"_payload",_encode(record))
    return result
