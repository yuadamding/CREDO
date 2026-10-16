"""Serialized main-worker task, not permission to launch or fit.

Only metadata is decoded here. The executor must separately verify controller
release, actual source/runtime, main-worker qualification and input custody.
The observation-only qualification cannot authorize this different path.
"""

from dataclasses import asdict, fields
import json
from pathlib import Path

from .continuation_custody import ContinuationForecastAdmission
from .coordination import WorkBinding, _label
from .fitting_inputs import _encoded, _object, _typed
from .fitting_resources import FittingResourcePolicy
from .observation_loader import ObservationFile, _path, _unique_object
from .planner import (StagePlan, ScoreTask, TrainingTask, TrajectoryKey,
                      screen_plan, family_plan, finalist_plan, outer_plan)
from .prediction_plan import required_before_continuation
from .retention import STUDY_ID, identity_sha256
from .selection_evidence import SelectionMetadataFile


SCHEMA = "credo.frontier.main-worker-contract.v1"
MAXIMUM_BYTES = 4 * 1024**2
IMAGE = "hpcharbor.mdanderson.edu/yding41/credo_env@sha256:138d7191792ce1139708751f109817a976c5b96ba3a00be3dff88173b62eb21a"


def parse(payload):
    if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_BYTES:
        raise ValueError("Bounded immutable worker metadata required")
    return json.loads(payload, object_pairs_hook=_unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite worker metadata")))


def task_from_record(value, cls=TrainingTask):
    record = dict(_object(value, (f.name for f in fields(cls))))
    record["key"] = _typed(TrajectoryKey, record["key"], tuples=("training_donors",))
    if cls is TrainingTask:
        if type(record["requesting_selectors"]) is not list:
            raise ValueError("Explicit selector list required")
        record["requesting_selectors"] = tuple(record["requesting_selectors"])
    result = cls(**record)
    if _encoded(asdict(result)) != _encoded(value):
        raise ValueError("Canonical task ordering required")
    return result


def plan_from_record(value):
    _object(value, (f.name for f in fields(StagePlan)))
    if any(type(value[k]) is not list for k in ("training", "scores", "eligibility")):
        raise ValueError("Complete serialized stage lists required")
    result = StagePlan(value["name"],
        tuple(task_from_record(t) for t in value["training"]),
        tuple(task_from_record(t, ScoreTask) for t in value["scores"]), value["eligibility"])
    if _encoded(asdict(result)) != _encoded(value):
        raise ValueError("Canonical complete stage required")
    return result


def decode_contract(payload, *, qualification=False):
    if type(qualification) is not bool:
        raise ValueError("Explicit worker execution mode required")
    record = parse(payload)
    expected_fields = ("schema", "worker_id", "study_root", "receipt_root", "source_root",
        "source_manifest", "input_reference", "plan", "task_id", "previous_stage", "resume",
        "continuation_admissions", "resource_policy", "pin_memory", "runtime_image",
        "admission", "qualification")
    from .main_qualification_worker_plan import CONTRACT_SCHEMA, case_for, SCOPE
    _object(record, expected_fields + (("qualification_parent_id",) if qualification else ()))
    if record["schema"] != (CONTRACT_SCHEMA if qualification else SCHEMA) or record["runtime_image"] != IMAGE:
        raise ValueError("Exact main-worker schema and approved image required")
    if qualification:
        _label(record["qualification_parent_id"])
    _label(record["worker_id"])
    root, receipts, source = (_path(record[k]) for k in ("study_root", "receipt_root", "source_root"))
    if (root.name != STUDY_ID or receipts != root / "worker_runs" / record["worker_id"]
            or source == root or not source.is_relative_to(root)
            or receipts.is_relative_to(source) or source.is_relative_to(receipts)):
        raise ValueError("Separate exact study worker/source paths required")
    if qualification and (root.parent.name != record["qualification_parent_id"]
            or root.parent.parent.name != "qualification-fixtures"):
        raise ValueError("Disposable worker requires its isolated qualification-fixtures parent")
    refs = {k: _typed(ObservationFile, record[k]) for k in
            ("source_manifest", "input_reference", "admission", "qualification")}
    if (any(r.size_bytes > MAXIMUM_BYTES for r in refs.values())
            or len({r.path for r in refs.values()}) != 4):
        raise ValueError("Distinct bounded main-worker references required")
    plan = plan_from_record(record["plan"])
    tasks = {t.identifier: t for t in plan.training}
    if type(record["task_id"]) is not str or record["task_id"] not in tasks:
        raise ValueError("Worker must name one exact registered stage task")
    task = tasks[record["task_id"]]
    prior = None if record["previous_stage"] is None else plan_from_record(record["previous_stage"])
    if plan.name == "screen":
        if plan != screen_plan() or prior is not None:
            raise ValueError("Initial screen must contain the complete frozen search")
    else:
        if prior is None:
            raise ValueError("Promoted and outer fits require the prior complete stage")
        if plan.name == "family":
            expected = family_plan(dict(plan.eligibility)) if prior == screen_plan() else None
        elif plan.name == "finalist":
            expected = finalist_plan(prior, dict(plan.eligibility))
        else:
            expected = outer_plan(prior)
        if plan != expected:
            raise ValueError("Worker stage differs from the registered predecessor transition")
    policy = _typed(FittingResourcePolicy, record["resource_policy"])
    if policy.execution_scope != (SCOPE if qualification else "scientific") or type(record["pin_memory"]) is not bool:
        raise ValueError("Production entry is scientific CUDA only with explicit transfer choice")
    case = case_for(plan, task) if qualification else None
    if type(record["continuation_admissions"]) is not list:
        raise ValueError("Exact continuation metadata list required")
    custody = []
    for value in record["continuation_admissions"]:
        value = dict(_object(value, (f.name for f in fields(ContinuationForecastAdmission))))
        for k in ("seal", "seal_plan", "checkpoint_complete"):
            value[k] = _typed(SelectionMetadataFile, value[k])
        item = ContinuationForecastAdmission(**value)
        if item.fabricated_scalar_metadata is not qualification:
            raise ValueError("Forecast custody must match scientific versus fabricated worker scope")
        custody.append(item)
    resume = record["resume"]
    if task.start_updates:
        if (type(resume) is not dict or resume.get("accepted_updates") != (case.accepted_start if case else task.start_updates)
                or tuple(sorted(c.edge_id for c in custody)) != required_before_continuation(prior, task)):
            raise ValueError("Exact prior checkpoint and both directed forecasts required")
    elif resume is not None or custody:
        raise ValueError("Fresh fits cannot resume or consume prior checkpoint custody")
    return record, dict(plan=plan, task=task, previous_stage=prior, resource_policy=policy,
        continuation_admissions=tuple(custody), work_binding=WorkBinding.main(task),
        qualification_case=case, **refs)


def qualification_shape(inputs, policy, pin_memory):
    """Exact numerical case identity, independent of donor names and file paths.

    The future CUDA producer must exercise the actual execution shapes; this
    identifier alone supplies no qualification evidence.
    """
    record, parts = inputs.record(), inputs.components()
    from .configuration import get_configuration
    if parts["schedule"].execution_scope != "scientific" or parts["native_binding"].synthetic_fixture:
        raise ValueError("Scientific input shapes required")
    return identity_sha256(dict(configuration=asdict(get_configuration(parts["task"].key.configuration_id)),
        native_binding_dimensions=dict(rna_features=parts["native_binding"].rna_features),
        observation_specification=asdict(parts["observation"].specification),
        schedule=record["schedule"], execution=record["execution"], geometry=record["geometry"],
        provider_limits=record["preparation"]["provider_limits"], rules_sha256=parts["rules"].sha256,
        resource_policy=asdict(policy), pin_memory=pin_memory))
