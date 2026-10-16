"""Qualification-entry metadata only. Never run Adam or CUDA locally."""

from dataclasses import asdict, replace
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from frontier import main_worker as worker
from frontier.coordination import WorkBinding
from frontier.fitting_resources import FittingResourcePolicy
from frontier.main_qualification_plan import plan as full_plan, qualification_schedule
from frontier.main_qualification_worker_plan import (CONTRACT_SCHEMA, SCOPE, cases,
    worker_schedule, identity, case_for)
from frontier.main_worker_contract import decode_contract
from frontier.planner import DONORS, TrajectoryKey, screen_plan, family_plan, finalist_plan, outer_plan
from frontier.prediction_plan import required_before_continuation
from frontier.retention import STUDY_ID, identity_sha256
from test_main_worker import contract, encoded, ref, no_numerics, release_fixture
from frontier.observation_loader import ObservationFile


GIB, MIB = 1024**3, 1024**2


def chain():
    first = screen_plan()
    family = family_plan({d: tuple(f"F{i}-V2" for i in range(6)) for d in DONORS})
    finalists = finalist_plan(family, {d: ("F0-V2", "F1-V2", "F5-V2") for d in DONORS})
    return {"screen": (first, None), "family": (family, first), "outer": (outer_plan(finalists), finalists)}


def qualify(base, case):
    value = copy.deepcopy(base)
    root = Path(base["study_root"]).parent / "qualification-fixtures" / "main-qual-v1" / STUDY_ID
    root.mkdir(parents=True, exist_ok=True); (root / "worker_runs").mkdir(exist_ok=True)
    current, prior = chain()[case.stage]
    task = next(t for t in current.training if t.key == TrajectoryKey(case.training_donors, case.configuration_id, 0))
    value.update(schema=CONTRACT_SCHEMA, qualification_parent_id="main-qual-v1",
        study_root=str(root), receipt_root=str(root / "worker_runs" / value["worker_id"]),
        source_root=str(root / "bundle/source"), plan=asdict(current), task_id=task.identifier,
        previous_stage=None if prior is None else asdict(prior),
        resource_policy=asdict(FittingResourcePolicy(SCOPE, GIB, 64*MIB, 256*MIB, 8*MIB, 200*GIB, 8*GIB)))
    if case.accepted_start:
        value["resume"] = dict(accepted_updates=case.accepted_start, checkpoint_sha256="a"*64)
        value["continuation_admissions"] = [dict(edge_id=edge, seal=ref(root / f"{i}-SEAL.json"),
            seal_plan=ref(root / f"{i}-SEAL.json.plan.json"), checkpoint_complete=ref(root / "prior/COMPLETE.json"),
            maximum_metadata_bytes=MIB, maximum_artifact_bytes=MIB, fabricated_scalar_metadata=True)
            for i, edge in enumerate(required_before_continuation(prior, task))]
    return value


@pytest.mark.parametrize("case", cases())
def test_exact_three_cases_keep_disposable_and_scientific_work_separate(contract, case):
    value = qualify(contract, case)
    record, parts = decode_contract(encoded(value), qualification=True)
    assert parts["qualification_case"] == case
    assert parts["work_binding"] == WorkBinding.main(parts["task"])
    assert worker.qualification_binding().kind == "qualification"
    assert worker.qualification_binding().target_updates == 0
    assert parts["resource_policy"].execution_scope == SCOPE
    assert bool(parts["continuation_admissions"]) == bool(case.accepted_start)
    assert encoded(record) == encoded(value)
    with pytest.raises(ValueError): decode_contract(encoded(value))
    with pytest.raises(ValueError): decode_contract(encoded(contract), qualification=True)


def test_qualification_workload_and_shapes_are_bounded():
    proposal = full_plan()
    assert sum(c.optimizer_calls for c in cases()) == 7
    assert proposal["owner_optimizer_calls"] == 104
    assert proposal["planned_optimizer_calls"] == 111 <= proposal["maximum_optimizer_calls"] == 112
    assert proposal["scientific_updates"] == 0
    assert worker_schedule().stages == qualification_schedule().stages
    assert worker_schedule().logical_batch_laws == 4
    assert qualification_schedule().logical_batch_laws == 1024
    for field, value in (("logical_batch_laws", 3), ("training_particles", 8), ("maximum_refresh_laws", 8)):
        with pytest.raises(ValueError): replace(worker_schedule(), **{field: value})


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("qualification_parent_id"),
    lambda r: r.update(qualification_parent_id="foreign-parent"),
    lambda r: r.update(schema="credo.frontier.main-worker-contract.v1"),
    lambda r: r["resource_policy"].update(execution_scope="scientific"),
    lambda r: r["resource_policy"].update(execution_scope="synthetic_cpu_only"),
    lambda r: r.update(task_id=screen_plan().training[0].identifier),
])
def test_no_unregistered_or_nonisolated_worker_case(contract, mutation):
    value = qualify(contract, cases()[0]); mutation(value)
    with pytest.raises((ValueError, TypeError)): decode_contract(encoded(value), qualification=True)


@pytest.mark.parametrize("change", ["missing_custody", "scientific_custody", "global_prefix"])
def test_compact_continuation_requires_both_real_verified_metadata_paths(contract, change):
    value = qualify(contract, cases()[1])
    if change == "missing_custody": value["continuation_admissions"].pop()
    elif change == "scientific_custody": value["continuation_admissions"][0]["fabricated_scalar_metadata"] = False
    else: value["resume"]["accepted_updates"] = 1024
    with pytest.raises(ValueError): decode_contract(encoded(value), qualification=True)


def admission(record, parts):
    return dict(schema="credo.frontier.main-worker-qualification-admission.v1",
        run_id=record["worker_id"], parent_run_id=record["qualification_parent_id"], status="reserved",
        work_binding=worker.qualification_binding().to_dict(),
        stage_plan_sha256=identity_sha256(asdict(parts["plan"])),
        source_manifest_sha256=parts["source_manifest"].sha256,
        input_reference=asdict(parts["input_reference"]),
        resource_policy_sha256=parts["resource_policy"].identity,
        qualification_sha256=parts["qualification"].sha256,
        scientific_update_reservation=0, disposable_optimizer_calls=parts["qualification_case"].optimizer_calls,
        case=asdict(parts["qualification_case"]), real_biological_payload_access=False,
        storage_reserved_bytes=GIB, aggregate_reserved_output_bytes=64*GIB, live_h100_reservations=1,
        selection_evidence=[], automatic_retry=False, automatic_successor=False,
        **{k:"a"*64 for k in ("journal_event_sha256", "journal_identity_sha256",
                              "phase_authority_sha256", "capacity_receipt_sha256")})


@pytest.mark.parametrize("case", cases())
def test_qualification_admission_never_reserves_a_study_fit(contract, case):
    record, parts = decode_contract(encoded(qualify(contract, case)), qualification=True)
    allowed = admission(record, parts)
    assert worker.validate_admission(record, parts, allowed) == ()
    for key, value in (("scientific_update_reservation", 1), ("real_biological_payload_access", True),
                       ("work_binding", parts["work_binding"].to_dict()), ("disposable_optimizer_calls", 20)):
        changed = dict(allowed, **{key:value})
        with pytest.raises(ValueError): worker.validate_admission(record, parts, changed)


def fixture_permit(parts):
    return dict(schema="credo.frontier.main-worker-fixture-permit.v1",
        status="disposable_execution_permitted_not_qualified", case=asdict(parts["qualification_case"]),
        worker_plan_sha256=identity(), source_manifest_sha256=parts["source_manifest"].sha256,
        input_reference=asdict(parts["input_reference"]), scientific_updates=0,
        real_biological_payload_access=False, runtime_image=worker.IMAGE)


def test_fixture_permit_cannot_be_a_pass_or_consume_real_data(contract):
    _, parts = decode_contract(encoded(qualify(contract, cases()[0])), qualification=True)
    native = SimpleNamespace(synthetic_fixture=True)
    observed = SimpleNamespace(specification=SimpleNamespace(synthetic_fixture=True))
    values = dict(native_binding=native, observation=observed, schedule=worker_schedule(), task=parts["task"])
    inputs = SimpleNamespace(components=lambda:values)
    permit = fixture_permit(parts)
    assert worker.validate_fixture_permit(parts, inputs, permit)["scientific_fit_authorized"] is False
    with pytest.raises(ValueError): worker.validate_fixture_permit(parts, inputs, dict(permit,status="passed"))
    native.synthetic_fixture = False
    with pytest.raises(ValueError): worker.validate_fixture_permit(parts, inputs, permit)
    native.synthetic_fixture = True; values["schedule"] = qualification_schedule()
    with pytest.raises(ValueError): worker.validate_fixture_permit(parts, inputs, permit)


def test_qualification_inspection_uses_no_numerical_or_scientific_authority(contract, monkeypatch, capsys):
    value = qualify(contract, cases()[0])
    reference = ref(Path(value["study_root"]) / "WORKER.json", value)
    monkeypatch.setattr(worker, "execute", lambda *_a, **_k: pytest.fail("No executor in inspection"))
    assert worker.main(["inspect-qualification", "--contract", reference["path"],
                        "--contract-sha256", reference["sha256"]]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scientific_updates_requested"] == result["optimizer_calls"] == 0
    assert result["disposable_optimizer_calls_requested"] == 2
    assert result["numerical_payloads_opened"] is False


@pytest.mark.parametrize("change", ["none", "scientific", "self_pass", "no_parent_authority", "wrong_parent", "wrong_calls"])
def test_disposable_release_keeps_actual_parent_and_never_self_certifies(contract, monkeypatch, change):
    record, parts = decode_contract(encoded(qualify(contract, cases()[0])), qualification=True)
    allowed = admission(record, parts)
    cref, env, release = release_fixture(record, parts, allowed, monkeypatch)
    env["CREDO_RUN_ID"] = record["qualification_parent_id"]
    release.update(schema="credo.frontier.main-worker-qualification-startup-release.v1",
        parent_run_id=record["qualification_parent_id"], scientific_fit_authorized=False,
        qualification_terminal_readback_verified=False, disposable_qualification_authorized=True,
        real_biological_payload_access=False, disposable_optimizer_calls=2,
        parent_qualification_authority_verified=True)
    if change == "scientific": release["scientific_fit_authorized"] = True
    elif change == "self_pass": release["qualification_terminal_readback_verified"] = True
    elif change == "no_parent_authority": release["parent_qualification_authority_verified"] = False
    elif change == "wrong_parent": env["CREDO_RUN_ID"] = "foreign"
    elif change == "wrong_calls": release["disposable_optimizer_calls"] = 3
    if change == "none": worker.validate_release(record, parts, cref, release, allowed, env)
    else:
        with pytest.raises(ValueError): worker.validate_release(record, parts, cref, release, allowed, env)


@pytest.fixture
def execution(contract, monkeypatch):
    # Real worker control flow and leases; CUDA, source context and numerical
    # executor are mocked. This is NOT full worker-entry CUDA qualification.
    value = qualify(contract, cases()[0]); root = Path(value["study_root"])
    value["source_manifest"] = ref(root / "SOURCES.json", {"fabricated_source": True})
    value["input_reference"] = ref(root / "INPUTS.json", {"fabricated_inputs": True})
    _, parts = decode_contract(encoded(value), qualification=True)
    value["qualification"] = ref(root / "PERMIT.json", fixture_permit(parts))
    record, parts = decode_contract(encoded(value), qualification=True)
    value["admission"] = ref(root / "ADMISSION.json", admission(record, parts))
    record, parts = decode_contract(encoded(value), qualification=True)
    cref = ObservationFile(**ref(root / "WORKER.json", value))
    rref = ObservationFile(**ref(root / "RELEASE.json", {"fabricated_release": True}))
    events = []
    @contextmanager
    def sources(*_a):
        events.append("source_enter"); yield; events.append("source_exit")
    monkeypatch.setattr(worker, "verified_sources", sources)
    monkeypatch.setattr(worker, "validate_release", lambda *_a: events.append("release_mock"))
    monkeypatch.setattr(worker, "_configure_cuda", lambda: events.append("cuda_mock") or {})
    monkeypatch.setattr(worker, "_runtime", lambda: {})
    monkeypatch.setattr(worker, "validate_qualification", lambda *_a: pytest.fail("No synthetic success certificate"))
    class Inputs:
        def __init__(self, _raw): pass
        def record(self):
            return dict(source_sha256=parts["source_manifest"].sha256, maximum_checkpoint_bytes=64*MIB)
        def components(self):
            return dict(task=parts["task"], native_binding=SimpleNamespace(synthetic_fixture=True),
                observation=SimpleNamespace(specification=SimpleNamespace(synthetic_fixture=True)),
                schedule=worker_schedule())
    monkeypatch.setattr(worker, "FittingInputContract", Inputs)
    def run(reference, **kwargs):
        events.append("executor_mock")
        assert reference == parts["input_reference"]
        assert kwargs["resource_policy"].execution_scope == SCOPE and kwargs["device"] == "cuda:0"
        return dict(status="complete_after_input_stage_and_lease_exit", task=asdict(parts["task"]),
            work_binding=parts["work_binding"].to_dict(), source_sha256=parts["source_manifest"].sha256,
            input_contract=asdict(parts["input_reference"]), execution_scope=SCOPE,
            accepted_start=0, accepted_target=2, scientific_updates_this_task=0, optimizer_calls_this_task=2,
            resource_policy_sha256=parts["resource_policy"].identity, full_input_exit_verified=True,
            automatic_successor=False)
    monkeypatch.setattr(worker, "run_admitted_fitting_task", run)
    return cref, rref, record, parts, events


def test_qualification_uses_shared_executor_and_publication_after_exit(execution):
    cref, rref, record, parts, events = execution
    result = worker.execute(cref, rref, qualification=True)
    assert events == ["source_enter", "release_mock", "cuda_mock", "executor_mock", "source_exit"]
    receipt = json.loads(Path(result["path"]).read_text())
    assert receipt["disposable_worker_qualification"] and not receipt["scientific_fit"]
    assert receipt["execution"]["scientific_updates_this_task"] == 0
    assert receipt["execution"]["optimizer_calls_this_task"] == 2  # Mock metadata, no actual calls.
    with pytest.raises(FileExistsError): worker.execute(cref, rref, qualification=True)


@pytest.mark.parametrize("failure", ["source_exit", "input_lease", "scientific_result", "wrong_prefix"])
def test_qualification_cannot_publish_after_failure_or_miscount(execution, monkeypatch, failure):
    cref, rref, record, parts, events = execution
    original = worker.run_admitted_fitting_task
    if failure == "source_exit":
        @contextmanager
        def bad_source(*_a):
            yield
            raise ValueError("Injected source exit failure")
        monkeypatch.setattr(worker, "verified_sources", bad_source)
    else:
        def invalid(*a, **k):
            result = original(*a, **k)
            if failure == "input_lease": Path(parts["input_reference"].path).touch()
            elif failure == "scientific_result": result["scientific_updates_this_task"] = 2
            else: result["accepted_target"] = 1024
            return result
        monkeypatch.setattr(worker, "run_admitted_fitting_task", invalid)
    with pytest.raises(ValueError): worker.execute(cref, rref, qualification=True)
    output = Path(record["receipt_root"])
    assert not (output / "COMPLETE.json").exists()
    assert (output / "FAILED.json").exists()
