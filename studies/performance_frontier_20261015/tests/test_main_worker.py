"""Fabricated metadata/mocked execution only; zero optimizer or GPU calls."""

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import copy
import json
from pathlib import Path

import pytest
import torch

from frontier import main_worker as worker
from frontier.main_worker_contract import (SCHEMA, IMAGE, decode_contract, plan_from_record,
                                           qualification_shape)
from frontier.fitting_resources import FittingResourcePolicy
from frontier.observation_loader import ObservationFile
from frontier.planner import (DONORS, TrajectoryKey, screen_plan, family_plan, finalist_plan, outer_plan)
from frontier.prediction_plan import required_before_continuation
from frontier.retention import STUDY_ID, identity_sha256


GIB, MIB = 1024**3, 1024**2


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def ref(path, value=None):
    if value is None:
        return dict(path=str(path), sha256="a"*64, size_bytes=1)
    raw = encoded(value)
    path.write_bytes(raw)
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture(autouse=True)
def no_numerics(monkeypatch, request):
    def forbid(*_a, **_k):
        pytest.fail("No optimizer or GPU call is authorized by this metadata suite")
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.LBFGS):
        monkeypatch.setattr(cls, "step", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    request.node.user_properties.append(("disposable_cpu_optimizer_calls", 0))


@pytest.fixture
def contract(tmp_path):
    root = tmp_path / STUDY_ID
    root.mkdir(); (root / "worker_runs").mkdir()
    plan = screen_plan()
    task = next(t for t in plan.training if t.key == TrajectoryKey(("D3", "D4"), "F0-V1", 0))
    policy = FittingResourcePolicy("scientific", 2*GIB, 64*MIB, 256*MIB, 8*MIB, 200*GIB, 8*GIB)
    value = dict(schema=SCHEMA, worker_id="fabricated-main", study_root=str(root),
        receipt_root=str(root / "worker_runs/fabricated-main"), source_root=str(root / "bundle/source"),
        source_manifest=ref(root / "SOURCES.json"), input_reference=ref(root / "INPUTS.json"),
        admission=ref(root / "ADMISSION.json"), qualification=ref(root / "QUALIFICATION.json"),
        plan=asdict(plan), task_id=task.identifier, previous_stage=None, resume=None,
        continuation_admissions=[], resource_policy=asdict(policy), pin_memory=False, runtime_image=IMAGE)
    return json.loads(encoded(value))


def plans():
    first = screen_plan()
    family = family_plan({d: tuple("F"+str(i)+"-V1" for i in range(6)) for d in DONORS})
    finalist = finalist_plan(family, {d: ("F0-V1", "F1-V1", "F4-V1") for d in DONORS})
    return first, family, finalist, outer_plan(finalist)


def bind_stage(contract, current, prior=None):
    value = copy.deepcopy(contract)
    task = current.training[0]
    value.update(plan=asdict(current), previous_stage=None if prior is None else asdict(prior), task_id=task.identifier)
    if task.start_updates:
        value["resume"] = dict(accepted_updates=task.start_updates, checkpoint_sha256="a"*64)
        root = Path(value["study_root"])
        value["continuation_admissions"] = [dict(edge_id=edge, seal=ref(root / (str(i)+"SEAL.json")),
            seal_plan=ref(root / (str(i)+"SEAL.json.plan.json")), checkpoint_complete=ref(root / "prior/COMPLETE.json"),
            maximum_metadata_bytes=MIB, maximum_artifact_bytes=MIB, fabricated_scalar_metadata=False)
            for i, edge in enumerate(required_before_continuation(prior, task))]
    return json.loads(encoded(value))


@pytest.mark.parametrize("index", range(4))
def test_all_registered_stages_round_trip(contract, index):
    chain = plans()
    value = bind_stage(contract, chain[index], None if not index else chain[index-1])
    record, parts = decode_contract(encoded(value))
    assert parts["plan"] == chain[index] and record == value
    assert parts["task"] == chain[index].training[0]
    assert len(parts["continuation_admissions"]) == (2 if index in (1, 2) else 0)


def test_all_screen_configurations_and_shared_pairs_keep_exact_tasks(contract):
    first = screen_plan()
    for task in first.training:
        value = dict(contract, task_id=task.identifier)
        _, parts = decode_contract(encoded(value))
        assert parts["task"] == task
        assert set(parts["task"].key.training_donors).isdisjoint(parts["task"].requesting_selectors)
    assert len(first.training) == 144


@pytest.mark.parametrize("change", [
    lambda r: r.update(runtime_image="unapproved"),
    lambda r: r.update(unregistered=True),
    lambda r: r.update(pin_memory=1),
    lambda r: r.update(worker_id="../wrong"),
    lambda r: r.update(receipt_root=r["study_root"]),
    lambda r: r.update(source_root=r["receipt_root"]),
    lambda r: r.update(task_id="missing-task"),
    lambda r: r["plan"]["training"].pop(),
    lambda r: r["plan"]["scores"].pop(),
    lambda r: r["plan"]["training"][0]["key"].update(training_donors=["D2", "D1"]),
    lambda r: r["resource_policy"].update(execution_scope="synthetic_cpu_only"),
    lambda r: r.update(resume={"accepted_updates": 0}),
    lambda r: r["source_manifest"].update(size_bytes=5*MIB),
])
def test_contract_rejects_changed_scope_before_execution(contract, change):
    change(contract)
    with pytest.raises((ValueError, TypeError)):
        decode_contract(encoded(contract))


@pytest.mark.parametrize("change", ["missing_prior", "missing_query", "fabricated", "resume_prefix", "foreign_candidate"])
def test_continuation_rejects_incomplete_prior_evidence(contract, change):
    first, family, *_ = plans()
    value = bind_stage(contract, family, first)
    if change == "missing_prior": value["previous_stage"] = None
    elif change == "missing_query": value["continuation_admissions"].pop()
    elif change == "fabricated": value["continuation_admissions"][0]["fabricated_scalar_metadata"] = True
    elif change == "resume_prefix": value["resume"]["accepted_updates"] = 0
    else: value["continuation_admissions"][0]["edge_id"] = "other"
    with pytest.raises(ValueError): decode_contract(encoded(value))


def admissions(contract):
    record, parts = decode_contract(encoded(contract))
    admission = dict(schema="credo.frontier.main-fitting-admission.v1", run_id=record["worker_id"],
        status="reserved", work_binding=parts["work_binding"].to_dict(),
        stage_plan_sha256=identity_sha256(asdict(parts["plan"])),
        source_manifest_sha256=parts["source_manifest"].sha256,
        input_reference=asdict(parts["input_reference"]), resource_policy_sha256=parts["resource_policy"].identity,
        qualification_sha256=parts["qualification"].sha256,
        scientific_update_reservation=parts["task"].additional_updates,
        storage_reserved_bytes=parts["resource_policy"].output_limit_bytes,
        automatic_retry=False, automatic_successor=False, aggregate_reserved_output_bytes=4*GIB,
        live_h100_reservations=1, selection_evidence=[],
        **{k: "a"*64 for k in ("journal_event_sha256", "journal_identity_sha256", "phase_authority_sha256", "capacity_receipt_sha256")})
    return record, parts, admission


@pytest.mark.parametrize("field,value", [("automatic_retry", 0), ("automatic_successor", True),
    ("live_h100_reservations", 5), ("live_h100_reservations", True),
    ("aggregate_reserved_output_bytes", 385*GIB), ("scientific_update_reservation", 1025),
    ("source_manifest_sha256", "b"*64), ("journal_event_sha256", None),
    ("selection_evidence", [dict(path="/x", sha256="a"*64, size_bytes=1)])])
def test_bad_admission_rejected(contract, field, value):
    record, parts, admission = admissions(contract)
    assert worker.validate_admission(record, parts, admission) == ()
    admission[field] = value
    with pytest.raises((ValueError, TypeError)):
        worker.validate_admission(record, parts, admission)


def test_inspection_is_metadata_only_and_run_requires_release(contract, monkeypatch, capsys):
    path = Path(contract["study_root"]) / "WORKER.json"
    reference = ref(path, contract)
    monkeypatch.setattr(worker, "execute", lambda *_a: pytest.fail("No fitting"))
    args = ["--contract", str(path), "--contract-sha256", reference["sha256"]]
    assert worker.main(["inspect", *args]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["optimizer_calls"] == 0 and not report["numerical_payloads_opened"]
    with pytest.raises(SystemExit): worker.main(["run", *args])


@pytest.fixture
def execution(contract, monkeypatch):
    # Test the REAL wrapper and metadata leases; replace only the CUDA probe,
    # heavy input decoder, and numerical executor. No actual main fit occurs.
    root = Path(contract["study_root"])
    contract["source_manifest"] = ref(root / "SOURCES.json", {"fabricated_source": True})
    contract["input_reference"] = ref(root / "INPUTS.json", {"fabricated_inputs": True})
    contract["qualification"] = ref(root / "QUALIFICATION.json", {"fabricated_qualification": True})
    _, _, admission = admissions(contract)
    contract["admission"] = ref(root / "ADMISSION.json", admission)
    contract_ref = ObservationFile(**ref(root / "WORKER.json", contract))
    release_ref = ObservationFile(**ref(root / "RELEASE.json", {"fabricated_release": True}))
    record, parts = decode_contract(encoded(contract))
    events = []
    @contextmanager
    def sources(*_a):
        events.append("source_enter")
        yield
        events.append("source_exit")
    monkeypatch.setattr(worker, "verified_sources", sources)
    monkeypatch.setattr(worker, "validate_release", lambda *_a: events.append("release"))
    monkeypatch.setattr(worker, "_configure_cuda", lambda: events.append("cuda_mock") or {})
    monkeypatch.setattr(worker, "_runtime", lambda: {})
    monkeypatch.setattr(worker, "validate_qualification", lambda *_a: events.append("qualification_mock") or {})
    class Inputs:
        def __init__(self, _raw): pass
        def record(self):
            return dict(source_sha256=parts["source_manifest"].sha256, maximum_checkpoint_bytes=64*MIB)
        def components(self): return dict(task=parts["task"])
    monkeypatch.setattr(worker, "FittingInputContract", Inputs)
    def run(reference, **kwargs):
        events.append("executor_mock")
        assert reference == parts["input_reference"]
        assert kwargs["plan"] == parts["plan"] and kwargs["task"] == parts["task"]
        assert kwargs["previous_stage"] is None and kwargs["device"] == "cuda:0"
        return dict(status="complete_after_input_stage_and_lease_exit", task=asdict(parts["task"]),
            work_binding=parts["work_binding"].to_dict(), source_sha256=parts["source_manifest"].sha256,
            input_contract=asdict(parts["input_reference"]), execution_scope="scientific",
            accepted_start=0, accepted_target=1024, scientific_updates_this_task=1024,
            optimizer_calls_this_task=1024, resource_policy_sha256=parts["resource_policy"].identity,
            full_input_exit_verified=True, automatic_successor=False)
    monkeypatch.setattr(worker, "run_admitted_fitting_task", run)
    return dict(contract=contract_ref, release=release_ref, parts=parts, record=record, events=events,
        root=root, output=Path(record["receipt_root"]))


def test_wrapper_waits_for_source_exit_before_completion_and_rejects_reentry(execution):
    p = execution
    result = worker.execute(p["contract"], p["release"])
    assert p["events"] == ["source_enter", "release", "cuda_mock", "qualification_mock", "executor_mock", "source_exit"]
    assert json.loads(Path(result["path"]).read_text())["all_worker_input_and_source_exits_verified"]
    with pytest.raises(FileExistsError): worker.execute(p["contract"], p["release"])
    assert not (p["output"] / "FAILED.json").exists()


def test_source_exit_failure_prevents_worker_completion(execution, monkeypatch):
    @contextmanager
    def bad_source(*_a):
        yield
        raise ValueError("Source exit changed")
    monkeypatch.setattr(worker, "verified_sources", bad_source)
    with pytest.raises(ValueError, match="Source exit"):
        worker.execute(execution["contract"], execution["release"])
    assert not (execution["output"] / "COMPLETE.json").exists()
    assert (execution["output"] / "FAILED.json").is_file()


def test_release_change_during_mock_work_prevents_completion(execution, monkeypatch):
    original = worker.run_admitted_fitting_task
    def changed(*a, **k):
        result = original(*a, **k)
        Path(execution["release"].path).touch()
        return result
    monkeypatch.setattr(worker, "run_admitted_fitting_task", changed)
    with pytest.raises(ValueError, match="changed"):
        worker.execute(execution["contract"], execution["release"])
    assert not (execution["output"] / "COMPLETE.json").exists()


def test_execution_result_cannot_underreport_task(execution, monkeypatch):
    original = worker.run_admitted_fitting_task
    monkeypatch.setattr(worker, "run_admitted_fitting_task", lambda *a, **k:
        dict(original(*a, **k), scientific_updates_this_task=1))
    with pytest.raises(ValueError, match="evidence"):
        worker.execute(execution["contract"], execution["release"])
    assert (execution["output"] / "FAILED.json").is_file()


def test_observation_qualification_is_rejected_before_shape_or_runtime_use(contract):
    record, parts = decode_contract(encoded(contract))
    with pytest.raises(ValueError, match="evidence"):
        worker.validate_qualification(record, parts, None,
            dict(schema="credo.frontier.observation-cuda-qualification.v1", status="passed"), {}, {})


def test_required_pass_flags_do_not_accept_integer_one():
    with pytest.raises(ValueError): worker.exact_fields({"passed": 1}, {"passed": True})


def release_fixture(record, parts, admission, monkeypatch):
    monkeypatch.setattr(worker.os, "getuid", lambda: 307469)
    monkeypatch.setattr(worker.os, "getgid", lambda: 600815)
    contract_ref = ObservationFile("/unopened-contract.json", "c"*64, 1)
    environment = dict(CREDO_POD_UID="pod-uid", CREDO_POD_NAME="pod-name", CREDO_RUN_ID=record["worker_id"],
        CREDO_V4_EXECUTION_IMAGE_DIGEST=IMAGE.split("@")[1])
    release = dict(schema="credo.frontier.main-fitting-startup-release.v1", status="startup_released",
        run_id=record["worker_id"], pod_uid="pod-uid", pod_name="pod-name", job_uid="job-uid",
        worker_contract_sha256=contract_ref.sha256, admission_sha256=parts["admission"].sha256,
        source_manifest_sha256=parts["source_manifest"].sha256, qualification_sha256=parts["qualification"].sha256,
        journal_event_sha256=admission["journal_event_sha256"], image=IMAGE,
        actual_pod_audit_passed=True, storage_and_fileset_quota_passed=True, aggregate_reservation_verified=True,
        cgroup_resources_verified=True, qualification_terminal_readback_verified=True,
        selection_evidence_verified=True, scientific_fit_authorized=True,
        automatic_retry=False, automatic_successor=False, authorized_cpus=24,
        authorized_memory_bytes=200*GIB, threads=8, gpus=1, worker_uid=307469, worker_gid=600815,
        startup_sha256="b"*64, capacity_receipt_sha256=admission["capacity_receipt_sha256"],
        phase_authority_sha256=admission["phase_authority_sha256"])
    return contract_ref, environment, release


@pytest.mark.parametrize("change", ["none", "pod", "image", "owner", "quota", "selection", "qualification", "authority", "bool"])
def test_controller_release_is_bound_to_actual_worker_and_required_checks(contract, monkeypatch, change):
    record, parts, admission = admissions(contract)
    cref, environment, release = release_fixture(record, parts, admission, monkeypatch)
    if change == "pod": environment["CREDO_POD_UID"] = "other"
    elif change == "image": environment["CREDO_V4_EXECUTION_IMAGE_DIGEST"] = "other"
    elif change == "owner": monkeypatch.setattr(worker.os, "getuid", lambda: 1)
    elif change == "quota": release["storage_and_fileset_quota_passed"] = False
    elif change == "selection": release["selection_evidence_verified"] = False
    elif change == "qualification": release["qualification_terminal_readback_verified"] = False
    elif change == "authority": release["phase_authority_sha256"] = "e"*64
    elif change == "bool": release["scientific_fit_authorized"] = 1
    if change == "none": worker.validate_release(record, parts, cref, release, admission, environment)
    else:
        with pytest.raises(ValueError): worker.validate_release(record, parts, cref, release, admission, environment)


@pytest.mark.parametrize("change", ["none", "source", "runtime", "shape", "reconstruction", "no_headroom"])
def test_main_qualification_requires_exact_source_runtime_case_and_memory(contract, monkeypatch, change):
    record, parts = decode_contract(encoded(contract))
    monkeypatch.setattr(worker, "qualification_shape", lambda *_a: "b"*64)
    hardware = dict(device="cuda:0", visible_devices=1, precision="FP32", tf32=False,
        deterministic_algorithms=True, required_headroom_bytes=8*GIB,
        name="NVIDIA H100 80GB HBM3", capability=[9, 0], total_memory_bytes=79*GIB)
    qualification = dict(schema="credo.frontier.main-worker-cuda-qualification.v1", status="passed",
        source_manifest_sha256=parts["source_manifest"].sha256, runtime_image=IMAGE,
        runtime={"fabricated": "runtime"}, scientific_updates=0, real_biological_payload_access=False,
        full_worker_entry_verified=True, checkpoint_resume_verified=True, publication_readback_verified=True,
        qualified_case_sha256s=["b"*64], hardware=hardware,
        resources_all_checks=dict(peak_reserved_bytes=20*GIB, physical_headroom_at_peak_reserved_bytes=59*GIB))
    if change == "source": qualification["source_manifest_sha256"] = "f"*64
    elif change == "runtime": qualification["runtime"] = {"fabricated": "different"}
    elif change == "shape": qualification["qualified_case_sha256s"] = ["c"*64]
    elif change == "reconstruction": qualification["full_worker_entry_verified"] = False
    elif change == "no_headroom": qualification["resources_all_checks"] = dict(
        peak_reserved_bytes=78*GIB, physical_headroom_at_peak_reserved_bytes=GIB)
    if change == "none":
        out = worker.validate_qualification(record, parts, None, qualification, hardware, {"fabricated": "runtime"})
        assert out["case_sha256"] == "b"*64
    else:
        with pytest.raises(ValueError):
            worker.validate_qualification(record, parts, None, qualification, hardware, {"fabricated": "runtime"})


def test_worker_receipts_cannot_consume_unreserved_output(tmp_path):
    with pytest.raises(RuntimeError, match="metadata allowance"):
        worker.publish_metadata(tmp_path, "oversized.json", {"x": "z"*100}, 10)
    assert not list(tmp_path.iterdir())
