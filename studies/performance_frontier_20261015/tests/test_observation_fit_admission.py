"""Fabricated temporary metadata evidence only: no arrays, remote or optimizer."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from deployment.build import FIT_FIELDS, SOURCE_MEMBERS
from deployment.common import GIB, encoded, identity, read_json
from frontier.coordination import AdmissionJournal, WorkBinding
from operations import reserve_observation_fits as helper


REAL_ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def wrapped(value, worker=False):
    raw = ((json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
           if worker else encoded(value))
    return {"sha256": identity(raw)["sha256"], "value": value}, len(raw)


@pytest.fixture
def fixture(tmp_path, request):
    root = tmp_path / helper.STUDY
    root.mkdir()
    for name in (*SOURCE_MEMBERS, "AUTHORITY.json", "OBSERVATION_PROTOCOL.json",
                 "OBSERVATION_PHASE_CONTRACT.json", "OBSERVATION_SUBSET_METADATA_PLAN.json"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_ROOT / name, destination)
    # The source/plan metadata are copied, not executed. Every numerical and
    # scheduler-success claim below is deliberately fabricated inside tmp_path.
    package = root / "fabricated_qualification_package"
    package.mkdir()
    recovery = getattr(request, "param", False)
    original = REAL_ROOT / ("observation_qualification_launch_v2" if recovery else "observation_qualification_launch_v1")
    for name in ("BUILD.json", "CREATE.json", "bundle/PLAN.json", "bundle/SOURCE_MANIFEST.json",
                 "bundle/AUTHORITY.json", "bundle/OBSERVATION_PROTOCOL.json", "bundle/OBSERVATION_PHASE_CONTRACT.json",
                 "bundle/ADMISSION.json"):
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, destination)
    if recovery:
        for name in ("OBSERVATION_REQUALIFICATION_V2_AUTHORITY.json", "OBSERVATION_HARDWARE_RECOVERY_PROPOSAL.json"):
            shutil.copyfile(REAL_ROOT / name, root / name)
    # Rebind ONLY these fabricated copies to the current copied numerical
    # source. An older immutable real qualification is not evidence for a new
    # source revision, and no historical receipt is edited by this fixture.
    source_manifest = {"schema": "credo.frontier.observation-worker-source.v1",
                       "files": {name: helper.sha(root / name) for name in SOURCE_MEMBERS}}
    write(package / "bundle/SOURCE_MANIFEST.json", source_manifest)
    plan = read_json(package / "bundle/PLAN.json")
    plan["source_manifest"]["sha256"] = helper.sha(package / "bundle/SOURCE_MANIFEST.json")
    admission = read_json(package / "bundle/ADMISSION.json")
    admission["source_manifest_sha256"] = plan["source_manifest"]["sha256"]
    operation = {"run_id": plan["run_id"], "kind": "qualification",
                 "phase_authority_sha256": plan["phase_authority"]["sha256"],
                 "source_manifest_sha256": plan["source_manifest"]["sha256"],
                 "protocol_sha256": plan["protocol"]["sha256"]}
    if recovery:
        operation["requalification_authority_sha256"] = admission["requalification_authority_sha256"]
    admission["work_binding"] = WorkBinding.other("qualification", helper._digest(operation)).to_dict()
    write(package / "bundle/ADMISSION.json", admission)
    plan["admission"]["sha256"] = helper.sha(package / "bundle/ADMISSION.json")
    write(package / "bundle/PLAN.json", plan)
    build = read_json(package / "BUILD.json")
    build["source_manifest_sha256"] = plan["source_manifest"]["sha256"]
    build["plan_sha256"] = helper.sha(package / "bundle/PLAN.json")
    write(package / "BUILD.json", build)
    created = read_json(package / "CREATE.json")
    created["records"]["live-binding.json"]["uid"] = "fixture-only-qualification-uid"
    created["records"]["live-binding.json"]["plan_sha256"] = build["plan_sha256"]
    write(package / "CREATE.json", created)
    plan, build = read_json(package / "bundle/PLAN.json"), read_json(package / "BUILD.json")
    uid, run_id = "fixture-only-qualification-uid", plan["run_id"]
    paired = {"schema": "credo.frontier.observation-paired-fp32.v1", "status": "passed", "optimizer_calls": 0,
              "shape": [16, 18129], "architecture": [18129, 512, 128, 48],
              "tolerances": dict(helper.PAIRED_FP32_TOLERANCES), "all_parameter_gradients_checked": True,
              "same_initial_weights": True, "cpu_encoding": "sparse", "cuda_encoding": "bounded_dense",
              "loss": {"passed": True, "maximum_tolerance_normalized_error": 0.25},
              "gradient_summary": {"passed": True, "maximum_tolerance_normalized_error": 0.5},
              "fixture_only": True}
    paired_wrapper, paired_size = wrapped(paired, True)
    complete = {"schema": "credo.frontier.observation-cuda-qualification.v1", "status": "passed",
        "source_manifest_sha256": build["source_manifest_sha256"], "protocol_sha256": plan["protocol"]["sha256"],
        "runtime_image": helper.IMAGE, "scientific_updates": 0, "actual_disposable_optimizer_calls": 13,
        "hardware": {"device": "cuda:0", "name": "NVIDIA H100 80GB HBM3", "total_memory_bytes": 85028372480,
                     "capability": [9, 0], "visible_devices": 1, "precision": "FP32", "tf32": False,
                     "deterministic_algorithms": True, "required_headroom_bytes": 8 * GIB},
        "resources_after_warmup": {"peak_reserved_bytes": 4 * GIB, "peak_allocated_bytes": 3 * GIB,
                                   "physical_headroom_at_peak_reserved_bytes": 85028372480 - 4 * GIB},
        "resources_all_checks": {"peak_reserved_bytes": 5 * GIB, "peak_allocated_bytes": 4 * GIB,
                                 "physical_headroom_at_peak_reserved_bytes": 85028372480 - 5 * GIB},
        "real_count_payloads_opened": False, "real_input_mount_qualified": False,
        "restart_equivalence": "exact_model_optimizer_sampler_rng", "shape": [2048, 18129], "nonzeros_per_cell": 4096,
        "paired_fp32_check": paired, "paired_fp32_receipt": {"path": plan["scientific_output"] + "/PAIRED_FP32_CHECK.json",
                                                          "sha256": paired_wrapper["sha256"], "size_bytes": paired_size}, "fixture_only": True}
    complete_wrapper, complete_size = wrapped(complete, True)
    files = {"QUALIFICATION_COMPLETE.json": {"sha256": complete_wrapper["sha256"], "size_bytes": complete_size},
             "PAIRED_FP32_CHECK.json": {"sha256": paired_wrapper["sha256"], "size_bytes": paired_size},
             "fabricated_not_a_real_checkpoint.pt": {"sha256": "f" * 64, "size_bytes": 12345}}
    readback = {"schema": "credo.frontier.observation-output-readback.v1", "status": "passed", "run_id": run_id,
                "files": files, "total_bytes": sum(r["size_bytes"] for r in files.values()),
                "complete_sha256": complete_wrapper["sha256"], "no_model_loading": True}
    terminal = {"status": "passed", "job": run_id, "uid": uid, "run_id": run_id, "kubernetes_complete": True,
                "application_pass": True, "reason": None, "mode": "qualify", "lifecycle_plan_sha256": build["plan_sha256"],
                "result_complete_sha256": complete_wrapper["sha256"], "automatic_successor": False}
    cleanup = {"status": "complete", "job": run_id, "uid": uid,
               "reads": [{"job_absent": True, "owned_pods": 0}] * 2}
    snapshot = {"schema": "credo.frontier.exact-observation-status.v1", "run_id": run_id, "uid": uid,
        "plan_sha256": build["plan_sha256"], "at_utc": "2026-10-16T00:00:00+00:00", "remote_writes": False,
        "job": None, "pods": [], "records": {"terminal.json": wrapped(terminal)[0], "cleanup.json": wrapped(cleanup)[0],
            "output-readback.json": wrapped(readback)[0], "output/QUALIFICATION_COMPLETE.json": complete_wrapper,
            "output/PAIRED_FP32_CHECK.json": paired_wrapper}}
    snapshot_path = root / "fabricated_status.json"
    write(snapshot_path, snapshot)
    capacity = {"schema": "credo.frontier.storage-intake.v1", "study_id": helper.STUDY, "fileset": "h_bcb-yding4",
        "collector_sha256": helper.COLLECTOR_SHA, "remote_writes": False, "numerical_payloads_opened": False,
        "filesystem_available_inodes": 30000, "headroom_bytes": 100 * GIB, "filesystem_available_bytes": 200 * GIB,
        "utc": "2026-10-16T00:00:01+00:00", "quota": {"block_grace": "none", "file_grace": "none",
            "soft_bytes": 1000 * GIB, "hard_bytes": 2000 * GIB, "used_bytes": 900 * GIB, "in_doubt_bytes": 0}}
    capacity_path = root / "fabricated_capacity.json"
    write(capacity_path, capacity)
    request_path = root / "fabricated_request.json"
    subsets = read_json(root / "OBSERVATION_PROTOCOL.json")["training_subsets"][:4]
    requests = [{"run_id": "fixture-obs-" + "-".join(d.lower() for d in donors),
                 "phase": "observation_fixture_" + "_".join(d.lower() for d in donors) + "_v1",
                 "training_donors": donors} for donors in subsets]
    write(request_path, {"schema": "credo.frontier.observation-fit-batch-request.v1", "requested_fits": requests})
    evidence = helper.qualification_evidence(root, package, snapshot_path)
    options = {"study_identity_sha256": helper.sha(root / "AUTHORITY.json")}
    with AdmissionJournal(root / "admission", **options) as journal:
        journal.reserve(run_id, worker_id=run_id, kind="qualification", scientific_updates=0, output_bytes=16 * GIB,
            admission_sha256=evidence["phase_authority_sha256"], capacity_receipt_sha256="c" * 64,
            fileset_headroom_bytes=100 * GIB, work_binding=WorkBinding.from_dict(evidence["work_binding"]))
        journal.reconcile(run_id, status="submitted", exact_scheduler_id=uid, evidence_sha256="d" * 64)
    return {"root": root, "package": package, "snapshot": snapshot_path, "capacity": capacity_path,
            "request": request_path, "output": root / "fabricated_exports", "evidence": evidence,
            "options": options, "rows": requests}


def finish_qualification(f, release=False):
    e = f["evidence"]
    with AdmissionJournal(f["root"] / "admission", **f["options"]) as journal:
        journal.reconcile(e["run_id"], status="complete", exact_scheduler_id=e["uid"], evidence_sha256=e["snapshot_sha256"])
        if release:
            journal.release_storage(e["run_id"], retained_bytes=e["numerical_output_bytes"], inventory_receipt_sha256=e["readback_sha256"])


def state(f):
    with AdmissionJournal(f["root"] / "admission", **f["options"]) as journal:
        return journal.snapshot()


def run(f):
    return helper.reserve_batch(f["root"], f["package"], f["snapshot"], f["capacity"], f["request"], f["output"])


def test_four_reservations_share_final_event_and_preserve_retained_bytes(fixture, monkeypatch):
    f = fixture
    finish_qualification(f)
    before = state(f)
    result = run(f)
    after = state(f)
    assert after["sequence"] == before["sequence"] + 4
    assert result["status"] == "reserved_not_launched"
    assert result["new_scientific_update_reservations"] == 80000
    assert result["new_output_reservations_bytes"] == 16 * GIB
    assert result["aggregate_reserved_output_bytes"] == 32 * GIB
    assert result["live_h100_reservations"] == 4
    assert result["optimizer_calls"] == result["scheduler_calls"] == 0
    monkeypatch.syspath_prepend(str(REAL_ROOT / "operations"))
    import observation as operations
    monkeypatch.setattr(operations, "ROOT", f["root"])
    for row, expected in zip(result["fits"], f["rows"]):
        admission = read_json(row["admission"]["path"])
        bindings = read_json(row["fit_bindings"]["path"])
        assert admission["journal_event_sha256"] == after["last_sha256"]
        assert admission["aggregate_reserved_output_bytes"] == result["aggregate_reserved_output_bytes"]
        assert admission["live_h100_reservations"] == 4
        assert admission["work_binding"] == WorkBinding.representation(expected["training_donors"]).to_dict()
        assert admission["qualification_sha256"] == f["evidence"]["qualification_sha256"]
        assert admission["qualified_hardware"] == f["evidence"]["qualified_hardware"]
        assert admission["qualified_peak_reserved_bytes"] == f["evidence"]["qualified_peak_reserved_bytes"] == 5 * GIB
        assert admission["qualified_hardware"] == read_json(f["snapshot"])["records"][
            "output/QUALIFICATION_COMPLETE.json"]["value"]["hardware"]
        assert set(bindings) == FIT_FIELDS
        assert bindings["fold_id"] == "subset-" + "-".join(expected["training_donors"])
        assert bindings["metadata_root"] == bindings["dataset_root"] == helper.NATIVE_ROOT
        assert bindings["reader_source_root"] == helper.PROJECT + "/" + expected["phase"] + "/bundle/retained_reader"
        gate_package = f["root"] / (row["run_id"] + "-gate-fixture")
        write(gate_package / "bundle/ADMISSION.json", admission)
        assert operations.admission_gate(gate_package, {"run_id": row["run_id"],
            "source_manifest_sha256": admission["source_manifest_sha256"],
            "phase_authority_sha256": admission["phase_authority_sha256"]},
            {"maximum_output_bytes": 4 * GIB}) == admission
    with pytest.raises(ValueError, match="no retry"):
        run(f)
    assert state(f) == after


def test_passed_numerical_evidence_does_not_free_submitted_qualification_slot(fixture):
    before = state(fixture)
    with pytest.raises(ValueError, match="already be reconciled complete"):
        run(fixture)
    assert state(fixture) == before
    assert not fixture["output"].exists()


def replace_fixture_paired(snapshot, paired):
    """Rebind every fake producer hash so failures test semantics, not stale SHA."""
    records = snapshot["records"]
    paired_wrapper, paired_size = wrapped(paired, True)
    records["output/PAIRED_FP32_CHECK.json"] = paired_wrapper
    complete = records["output/QUALIFICATION_COMPLETE.json"]["value"]
    complete["paired_fp32_check"] = paired
    complete["paired_fp32_receipt"].update(sha256=paired_wrapper["sha256"], size_bytes=paired_size)
    complete_wrapper, complete_size = wrapped(complete, True)
    records["output/QUALIFICATION_COMPLETE.json"] = complete_wrapper
    terminal = records["terminal.json"]["value"]
    terminal["result_complete_sha256"] = complete_wrapper["sha256"]
    records["terminal.json"] = wrapped(terminal)[0]
    readback = records["output-readback.json"]["value"]
    readback["complete_sha256"] = complete_wrapper["sha256"]
    for name, wrapper, size in (("QUALIFICATION_COMPLETE.json", complete_wrapper, complete_size),
                                ("PAIRED_FP32_CHECK.json", paired_wrapper, paired_size)):
        readback["files"][name] = {"sha256": wrapper["sha256"], "size_bytes": size}
    readback["total_bytes"] = sum(row["size_bytes"] for row in readback["files"].values())
    records["output-readback.json"] = wrapped(readback)[0]


@pytest.mark.parametrize("change", ["missing", "name", "memory", "capability"])
def test_qualification_hardware_bound_before_any_reservation(fixture, change):
    f = fixture
    finish_qualification(f)
    snapshot = read_json(f["snapshot"])
    complete = snapshot["records"]["output/QUALIFICATION_COMPLETE.json"]["value"]
    if change == "missing":
        complete.pop("hardware")
    elif change == "name":
        complete["hardware"]["name"] = "NVIDIA H100 PCIe"
    elif change == "memory":
        complete["hardware"]["total_memory_bytes"] = 75 * GIB - 1
    else:
        complete["hardware"]["capability"] = [8, 0]
    replace_fixture_paired(snapshot, complete["paired_fp32_check"])
    write(f["snapshot"], snapshot)
    before = state(f)
    with pytest.raises(ValueError, match="qualification hardware"):
        run(f)
    assert state(f) == before
    assert not f["output"].exists()


@pytest.mark.parametrize("total", [75 * GIB, 85028372479, 82 * GIB])
def test_qualification_preserves_different_valid_full_device_memory_total(fixture, total):
    f = fixture
    snapshot = read_json(f["snapshot"])
    complete = snapshot["records"]["output/QUALIFICATION_COMPLETE.json"]["value"]
    complete["hardware"]["total_memory_bytes"] = total
    for name in ("resources_after_warmup", "resources_all_checks"):
        complete[name]["physical_headroom_at_peak_reserved_bytes"] = total - complete[name]["peak_reserved_bytes"]
    replace_fixture_paired(snapshot, complete["paired_fp32_check"])
    write(f["snapshot"], snapshot)
    before = state(f)
    evidence = helper.qualification_evidence(f["root"], f["package"], f["snapshot"])
    assert evidence["qualified_hardware"] == complete["hardware"]
    assert evidence["qualified_hardware"]["total_memory_bytes"] == total
    assert evidence["qualification_sha256"] == snapshot["records"]["output/QUALIFICATION_COMPLETE.json"]["sha256"]
    assert evidence["qualified_peak_reserved_bytes"] == 5 * GIB
    assert state(f) == before


@pytest.mark.parametrize("change", ["missing", "boolean_peak", "allocated_above_reserved", "headroom_mismatch", "headroom_short", "nonmonotonic"])
def test_qualification_measured_peak_headroom_proof_required(fixture, change):
    f = fixture
    finish_qualification(f)
    snapshot = read_json(f["snapshot"])
    complete = snapshot["records"]["output/QUALIFICATION_COMPLETE.json"]["value"]
    measured = complete["resources_all_checks"]
    if change == "missing":
        complete.pop("resources_all_checks")
    elif change == "boolean_peak":
        measured["peak_reserved_bytes"] = True
    elif change == "allocated_above_reserved":
        measured["peak_allocated_bytes"] = measured["peak_reserved_bytes"] + 1
    elif change == "headroom_mismatch":
        measured["physical_headroom_at_peak_reserved_bytes"] -= 1
    elif change == "nonmonotonic":
        measured["peak_allocated_bytes"] = 2 * GIB
        measured["peak_reserved_bytes"] = 3 * GIB
        measured["physical_headroom_at_peak_reserved_bytes"] = complete["hardware"]["total_memory_bytes"] - 3 * GIB
    else:
        measured["peak_reserved_bytes"] = complete["hardware"]["total_memory_bytes"] - 8 * GIB + 1
        measured["physical_headroom_at_peak_reserved_bytes"] = 8 * GIB - 1
    replace_fixture_paired(snapshot, complete["paired_fp32_check"])
    write(f["snapshot"], snapshot)
    before = state(f)
    with pytest.raises(ValueError, match="8-GiB headroom proof"):
        run(f)
    assert state(f) == before
    assert not f["output"].exists()


@pytest.mark.parametrize("change", ["missing_gradient", "gradient_not_all", "bad_tolerance", "failing_loss",
                                    "failing_gradient", "wrong_shape", "wrong_architecture"])
def test_paired_fp32_semantics_checked_before_any_reservation(fixture, change):
    f = fixture
    finish_qualification(f)
    snapshot = read_json(f["snapshot"])
    paired = deepcopy(snapshot["records"]["output/PAIRED_FP32_CHECK.json"]["value"])
    if change == "missing_gradient":
        paired.pop("gradient_summary")
    elif change == "gradient_not_all":
        paired["all_parameter_gradients_checked"] = False
    elif change == "bad_tolerance":
        paired["tolerances"]["gradient_rtol"] = 2e-3
    elif change == "failing_loss":
        paired["loss"]["maximum_tolerance_normalized_error"] = 1.01
    elif change == "failing_gradient":
        paired["gradient_summary"]["passed"] = False
    elif change == "wrong_shape":
        paired["shape"] = [8, 18129]
    else:
        paired["architecture"] = [18129, 128, 48]
    replace_fixture_paired(snapshot, paired)
    write(f["snapshot"], snapshot)
    before = state(f)
    with pytest.raises(ValueError, match="paired FP32"):
        run(f)
    assert state(f) == before
    assert not f["output"].exists()


@pytest.mark.parametrize("error", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, None, True])
def test_paired_fp32_rejects_nonfinite_or_invalid_normalized_error(fixture, error):
    paired = read_json(fixture["snapshot"])["records"]["output/PAIRED_FP32_CHECK.json"]["value"]
    paired["gradient_summary"]["maximum_tolerance_normalized_error"] = error
    with pytest.raises(ValueError, match="paired FP32 tolerance"):
        helper._validate_paired_fp32(paired)


def test_admission_tolerances_match_bound_worker_without_importing_numerical_module():
    tree = ast.parse((REAL_ROOT / "frontier/observation_worker.py").read_text())
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "PAIRED_FP32_TOLERANCES" for target in node.targets)]
    assert values == [helper.PAIRED_FP32_TOLERANCES]


@pytest.mark.parametrize("change", ["no_cleanup", "job_present", "pod_present", "wrong_uid", "failed_terminal", "altered_receipt_hash", "wrong_source"])
def test_incomplete_or_mismatched_success_evidence_blocks_without_mutation(fixture, change):
    f = fixture
    finish_qualification(f)
    snapshot = read_json(f["snapshot"])
    if change == "no_cleanup":
        snapshot["records"]["cleanup.json"] = None
    elif change == "job_present":
        snapshot["job"] = {"uid": f["evidence"]["uid"]}
    elif change == "pod_present":
        snapshot["pods"] = [{"uid": "fixture-pod"}]
    elif change == "wrong_uid":
        snapshot["uid"] = "different-fixture-uid"
    elif change == "failed_terminal":
        terminal = snapshot["records"]["terminal.json"]["value"]
        terminal["application_pass"] = False
        snapshot["records"]["terminal.json"] = wrapped(terminal)[0]
    elif change == "altered_receipt_hash":
        snapshot["records"]["output/QUALIFICATION_COMPLETE.json"]["sha256"] = "b" * 64
    else:
        (f["root"] / "frontier/observation.py").write_text("# fabricated changed source\n")
    write(f["snapshot"], snapshot)
    before = state(f)
    with pytest.raises(ValueError):
        run(f)
    assert state(f) == before
    assert not f["output"].exists()


@pytest.mark.parametrize("change", ["one_existing_live", "low_headroom", "stale_capacity", "duplicate_subset", "noncanonical_order", "already_reserved_subset"])
def test_entire_batch_rejected_before_first_reservation(fixture, change):
    f = fixture
    finish_qualification(f)
    if change in ("one_existing_live", "already_reserved_subset"):
        work = (WorkBinding.other("forecast", "e" * 64) if change == "one_existing_live"
                else WorkBinding.representation(("D1", "D2")))
        with AdmissionJournal(f["root"] / "admission", **f["options"]) as journal:
            journal.reserve("fixture-prior", worker_id="fixture-prior", kind=work.kind, scientific_updates=work.updates,
                output_bytes=GIB, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
                fileset_headroom_bytes=100 * GIB, work_binding=work)
            if change == "already_reserved_subset":
                journal.reconcile("fixture-prior", status="complete", exact_scheduler_id="fixture-old-uid", evidence_sha256="d" * 64)
    elif change in ("low_headroom", "stale_capacity"):
        capacity = read_json(f["capacity"])
        if change == "low_headroom":
            capacity["headroom_bytes"] = 30 * GIB  # Qualification + four outputs + reserve need 48 GiB.
            capacity["quota"]["used_bytes"] = 970 * GIB
        else:
            capacity["utc"] = "2026-10-15T23:59:59+00:00"
        write(f["capacity"], capacity)
    else:
        request = read_json(f["request"])
        if change == "duplicate_subset":
            request["requested_fits"][-1]["training_donors"] = ["D1", "D2"]
        else:
            request["requested_fits"].reverse()
        write(f["request"], request)
    before = state(f)
    with pytest.raises(ValueError):
        run(f)
    assert state(f) == before
    assert not f["output"].exists()


def test_numerical_output_only_inventory_cannot_release_whole_phase_storage(fixture):
    f = fixture
    finish_qualification(f, release=True)
    before = state(f)
    with pytest.raises(ValueError, match="unreleased 16-GiB cap"):
        run(f)
    assert state(f) == before
    assert not f["output"].exists()


def test_existing_live_slot_allows_three_not_four_new_fits(fixture):
    f = fixture
    finish_qualification(f)
    with AdmissionJournal(f["root"] / "admission", **f["options"]) as journal:
        journal.reserve("fixture-other", worker_id="fixture-other", kind="forecast", scientific_updates=0,
            output_bytes=GIB, admission_sha256="a" * 64, capacity_receipt_sha256="b" * 64,
            fileset_headroom_bytes=100 * GIB, work_binding=WorkBinding.other("forecast", "e" * 64))
    request = read_json(f["request"])
    request["requested_fits"] = request["requested_fits"][:3]
    write(f["request"], request)
    result = run(f)
    assert result["live_h100_reservations"] == 4
    assert result["new_scientific_update_reservations"] == 60000


def test_ten_subset_manifest_is_canonical_and_totals_reconcile():
    metadata = read_json(REAL_ROOT / "OBSERVATION_SUBSET_METADATA_PLAN.json")
    protocol = read_json(REAL_ROOT / "OBSERVATION_PROTOCOL.json")
    assert [r["training_donors"] for r in metadata["subsets"]] == protocol["training_subsets"]
    assert len(metadata["subsets"]) == 10
    assert metadata["payloads_opened"] is metadata["worker_visibility_established"] is False
    for row in metadata["subsets"]:
        assert row["fold_id"] == "subset-" + "-".join(row["training_donors"])
        expected_sources = sorted(d + "_" + t for d in row["training_donors"] for t in ("Stim8hr", "Stim48hr"))
        assert row["source_ids"] == expected_sources
        for field in ("shards", "captured_rows_metadata"):
            assert row[field] == sum(metadata["source_inventory"][sid][field] for sid in expected_sources)


def qualification_work(f, plan=None):
    plan = plan or read_json(f["package"] / "bundle/PLAN.json")
    return helper.qualification_work(f["root"], f["package"], plan,
        plan["source_manifest"]["sha256"], plan["protocol"]["sha256"], plan["phase_authority"]["sha256"])


def test_actual_v2_package_work_metadata_preserves_authority_without_outcome_access():
    # Read only already published local metadata, not the running Job or outputs.
    package = REAL_ROOT / "observation_qualification_launch_v2"
    f = {"root": REAL_ROOT, "package": package}
    before = helper.sha(package / "bundle/ADMISSION.json")
    work, admission_sha, recovery_sha = qualification_work(f)
    admission = read_json(package / "bundle/ADMISSION.json")
    assert work == admission["work_binding"]
    assert admission_sha == before == helper.sha(package / "bundle/ADMISSION.json")
    assert recovery_sha == helper.sha(REAL_ROOT / "OBSERVATION_REQUALIFICATION_V2_AUTHORITY.json")
    assert admission["requalification_authority"]["work"]["automatic_fit_successor"] is False


@pytest.mark.parametrize("fixture", [True], indirect=True)
def test_complete_fabricated_requalification_preserves_journal_identity(fixture):
    f = fixture
    before = state(f)
    evidence = helper.qualification_evidence(f["root"], f["package"], f["snapshot"])
    admission = read_json(f["package"] / "bundle/ADMISSION.json")
    assert evidence["work_binding"] == admission["work_binding"] == before["tasks"][evidence["run_id"]]["work_binding"]
    assert evidence["requalification_authority_sha256"] == admission["requalification_authority_sha256"]
    assert evidence["successor_fit_authority_from_qualification"] is False
    assert state(f) == before
    # Fabricated numerical pass is still insufficient until exact terminal
    # state has been reconciled in this temporary journal.
    with pytest.raises(ValueError, match="already be reconciled complete"):
        run(f)
    assert state(f) == before and not f["output"].exists()


@pytest.mark.parametrize("fixture", [True], indirect=True)
@pytest.mark.parametrize("change", ["no_authority", "wrong_hash", "inline_changed", "source_changed",
    "wrong_work", "unknown_run", "wrong_phase", "permission_text", "scientific_updates",
    "fit_successor", "retry", "deadline", "GPU_count", "output_cap", "boolean_steps", "proposal"])
def test_requalification_scope_or_provenance_mismatch_is_read_only(fixture, change):
    f = fixture
    plan = read_json(f["package"] / "bundle/PLAN.json")
    admission = read_json(f["package"] / "bundle/ADMISSION.json")
    authority_path = f["root"] / "OBSERVATION_REQUALIFICATION_V2_AUTHORITY.json"
    authority = read_json(authority_path)
    if change == "no_authority":
        admission.pop("requalification_authority_sha256")
    elif change == "wrong_hash":
        admission["requalification_authority_sha256"] = "a" * 64
    elif change == "inline_changed":
        admission["requalification_authority"]["interpretation"] = "fabricated changed approval"
    elif change == "source_changed":
        admission["source_manifest_sha256"] = "a" * 64
    elif change == "wrong_work":
        admission["work_binding"] = WorkBinding.other("qualification", "b" * 64).to_dict()
    elif change == "unknown_run":
        plan["run_id"] = admission["run_id"] = "credo-frontier-obs-qual-20261015-v99"
    elif change == "wrong_phase":
        plan["phase_directory"] = "observation_qualification_v99"
    else:
        if change == "permission_text": authority["user_instruction"] = "check status"
        elif change == "scientific_updates": authority["work"]["scientific_updates"] = 1
        elif change == "fit_successor": authority["work"]["automatic_fit_successor"] = True
        elif change == "retry": authority["work"]["automatic_retry"] = True
        elif change == "deadline": authority["work"]["elapsed_time_deadline"] = 86400
        elif change == "GPU_count": authority["resources"]["H100_per_job"] = 4
        elif change == "output_cap": authority["resources"]["output_cap_bytes"] = 32 * GIB
        elif change == "boolean_steps": authority["work"]["scientific_updates"] = False
        elif change == "proposal": authority["proposal_sha256"] = "c" * 64
        write(authority_path, authority)
        admission["requalification_authority"] = authority
        admission["requalification_authority_sha256"] = helper.sha(authority_path)
    # Rebind the *fabricated* admission hash so rejection tests its contents.
    write(f["package"] / "bundle/ADMISSION.json", admission)
    plan["admission"]["sha256"] = helper.sha(f["package"] / "bundle/ADMISSION.json")
    before = state(f)
    with pytest.raises(ValueError):
        qualification_work(f, plan)
    assert state(f) == before and not f["output"].exists()


@pytest.mark.parametrize("fixture", [True], indirect=True)
def test_requalification_cannot_recycle_failed_scientific_subset(fixture):
    f = fixture
    finish_qualification(f)
    with AdmissionJournal(f["root"] / "admission", **f["options"]) as journal:
        journal.reserve("fixture-failed-fit", worker_id="fixture-failed-fit", kind="representation",
            scientific_updates=20000, output_bytes=4 * GIB, admission_sha256="a" * 64,
            capacity_receipt_sha256="b" * 64, fileset_headroom_bytes=100 * GIB,
            work_binding=WorkBinding.representation(("D1", "D2")))
        journal.reconcile("fixture-failed-fit", status="failed", exact_scheduler_id="fixture-failed-uid",
                          evidence_sha256="c" * 64)
    before = state(f)
    with pytest.raises(ValueError, match="no retries"):
        run(f)
    assert state(f) == before and not f["output"].exists()
