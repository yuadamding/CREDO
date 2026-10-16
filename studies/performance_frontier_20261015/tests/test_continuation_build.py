"""Metadata-only continuation packaging: no journal mutation, tensors or jobs."""
import ast
from copy import deepcopy
import io
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from deployment import common, continuation_build as build
from frontier.coordination import WorkBinding


ROOT = Path(__file__).resolve().parents[1]


def fabricated_admission(pair):
    old = ROOT / ("observation_" + pair.lower() + "_v2_launch") / "bundle"
    worker = json.loads((old / "WORKER_CONTRACT.json").read_bytes())
    inventory = json.loads((ROOT / "receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json").read_bytes())
    _, source = build.source_manifest(ROOT)
    return dict(schema="credo.frontier.observation-admission-export.v1",
        run_id="credo-frontier-obs-" + pair.lower() + "-20261016-v3", status="reserved",
        work_binding=WorkBinding.representation(worker["training_donors"], 20000, 1280).to_dict(),
        storage_reserved_bytes=4 * common.GIB - inventory["fits"][pair]["parent_output_bytes"],
        scientific_update_reservation=18720, aggregate_reserved_output_bytes=64 * common.GIB,
        live_h100_reservations=4, journal_event_sha256="a" * 64, journal_identity_sha256="b" * 64,
        capacity_receipt_sha256="c" * 64,
        phase_authority_sha256=common.identity((old / "OBSERVATION_PHASE_CONTRACT.json").read_bytes())["sha256"],
        continuation_authority_sha256=common.identity((ROOT / "OBSERVATION_CONTINUATION_V3_AUTHORITY.json").read_bytes())["sha256"],
        source_manifest_sha256=common.identity(source)["sha256"], fixture_only_not_launch_authority=True)


@pytest.fixture
def package_factory(tmp_path, monkeypatch):
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    for module in (build, common):
        monkeypatch.setattr(module, "PROJECT", str(home))
        monkeypatch.setattr(module, "CONTROL", str(scratch))

    def factory(pair="D1D2", admission=None):
        admission = fabricated_admission(pair) if admission is None else admission
        payload, archive, report = build.construct(ROOT, pair, admission)
        plan = json.loads(payload["PLAN.json"])
        bundle = Path(plan["bundle_root"])
        for name, raw in payload.items():
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return payload, archive, report, bundle
    return factory


@pytest.mark.parametrize("pair", build.PAIRS)
def test_all_four_builds_bind_remaining_work_and_preserve_science(package_factory, pair):
    payload, archive, report, bundle = package_factory(pair)
    plan = json.loads(payload["PLAN.json"])
    worker, admission = common.verify_bundle(plan, bundle)
    recovery = json.loads(payload["CONTINUATION_CONTRACT.json"])
    old_path = ROOT / ("observation_" + pair.lower() + "_v2_launch") / "bundle"
    old = json.loads((old_path / "WORKER_CONTRACT.json").read_bytes())
    assert admission["work_binding"] == WorkBinding.representation(old["training_donors"], 20000, 1280).to_dict()
    assert plan["maximum_new_scientific_updates"] == report["maximum_new_scientific_updates"] == 18720
    assert recovery["consumed_updates"] == report["consumed_updates"] == 1280
    assert recovery["maximum_total_updates"] == 20000
    assert plan["maximum_output_bytes"] + recovery["parent_retained_output_bytes"] == 4 * common.GIB
    assert worker["output_limit_bytes"] == plan["maximum_output_bytes"]
    for name in ("metadata_root", "dataset_root", "fold_id", "training_donors", "metadata_index_sha256",
                 "rna_axis_sha256", "input_access_sha256", "protocol_sha256", "runtime_image",
                 "qualification_receipt", "qualification_sha256"):
        assert worker[name] == old[name]
    assert worker["trajectory_id"] != old["trajectory_id"]
    assert recovery["parent_fit_contract"]["sha256"] == common.identity((old_path / "WORKER_CONTRACT.json").read_bytes())["sha256"]
    assert payload["OBSERVATION_PROTOCOL.json"] == (old_path / "OBSERVATION_PROTOCOL.json").read_bytes()
    assert not any(name.endswith((".pt", ".npz", ".parquet")) for name in payload)
    assert report["mixed_source_lineage"] is True and report["remote_submission"] is False
    assert common.identity(archive)["sha256"] == report["archive_sha256"]


def test_runtime_contract_schema_and_exact_reference_sizes(package_factory):
    payload, _, _, _ = package_factory()
    recovery = json.loads(payload["CONTINUATION_CONTRACT.json"])
    tree = ast.parse(payload["source/frontier/observation_continuation.py"])
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_recovery")
    required = next(node.value for node in function.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "required" for target in node.targets))
    assert set(recovery) == ast.literal_eval(required)
    for name in ("fit_contract", "authority", "parent_intake", "parent_inventory", "parent_fit_contract",
                 "parent_source_manifest", "parent_input_lease", "recovery_check_worker_contract",
                 "recovery_check_completion", "recovery_check_restore", "recovery_check_readback",
                 "recovery_check_terminal"):
        reference = recovery[name]
        assert set(reference) == {"path", "sha256", "size_bytes"}, name
        assert type(reference["size_bytes"]) is int and reference["size_bytes"] > 0
        assert Path(reference["path"]).is_absolute()
        assert len(reference["sha256"]) == 64


@pytest.mark.parametrize("pair", build.PAIRS)
def test_controller_evidence_is_byte_identical_and_worker_visible(package_factory, pair):
    payload, _, _, bundle = package_factory(pair)
    recovery = json.loads(payload["CONTINUATION_CONTRACT.json"])
    origins = json.loads(payload["RECOVERY_CHECK_METADATA_ORIGINS.json"])
    originals = json.loads((ROOT / "receipts/OBSERVATION_CONTINUATION_V3_CHECK_REFERENCES.json").read_bytes())
    snapshot = json.loads((ROOT / "receipts/RETENTION_CHECK_V2_STATUS_20261016T161210Z.json").read_bytes())
    assert set(origins["copies"]) == {"output-readback.json", "terminal.json"}
    for field, key, name in (
        ("recovery_check_readback", "output-readback.json", "RECOVERY_CHECK_OUTPUT_READBACK.json"),
        ("recovery_check_terminal", "terminal.json", "RECOVERY_CHECK_TERMINAL.json"),
    ):
        reference = recovery[field]
        assert reference["path"] == str(bundle / name)
        assert origins["copies"][key]["origin"] == originals[field]
        assert origins["copies"][key]["worker_visible_copy"] == reference
        assert payload[name] == common.encoded(snapshot["records"][key]["value"])
        assert common.identity(payload[name]) == {k: originals[field][k] for k in ("sha256", "size_bytes")}
        assert common.read(reference["path"], reference["sha256"]) == payload[name]
    # Only provenance records may retain controller-scratch origins. Every
    # reference opened directly by the continuation worker must be visible.
    for field, value in recovery.items():
        if isinstance(value, dict) and "path" in value:
            assert not value["path"].startswith("/rsrch8/scratch/"), field
            assert value["path"].startswith((str(bundle) + "/", "/rsrch8/home/")), field


@pytest.mark.parametrize("damage", ["changed_readback", "changed_terminal", "missing_value", "wrong_size"])
def test_controller_copy_rejects_nonidentical_snapshot_or_size(package_factory, monkeypatch, damage):
    original_read = build.read
    def changed_read(path, *args, **kwargs):
        raw = original_read(path, *args, **kwargs)
        if Path(path).name == "RETENTION_CHECK_V2_STATUS_20261016T161210Z.json" and damage != "wrong_size":
            snapshot = json.loads(raw)
            key = "terminal.json" if damage == "changed_terminal" else "output-readback.json"
            if damage == "missing_value":
                del snapshot["records"][key]["value"]
            else:
                snapshot["records"][key]["value"]["fixture_corruption"] = True
            return common.encoded(snapshot)
        if Path(path).name == "OBSERVATION_CONTINUATION_V3_CHECK_REFERENCES.json" and damage == "wrong_size":
            references = json.loads(raw)
            references["recovery_check_readback"]["size_bytes"] += 1
            return common.encoded(references)
        return raw
    monkeypatch.setattr(build, "read", changed_read)
    with pytest.raises(ValueError, match="metadata (snapshot|bytes)"):
        package_factory()


@pytest.mark.parametrize("pair", build.PAIRS)
def test_failed_v3_package_keeps_original_evidence_and_rejects_worker_references(pair):
    frozen = ROOT / ("observation_" + pair.lower() + "_v3_launch")
    raw = (frozen / "bundle/CONTINUATION_CONTRACT.json").read_bytes()
    report = json.loads((frozen / "BUILD.json").read_bytes())
    assert common.identity(raw)["sha256"] == report["recovery_contract_sha256"]
    historical = json.loads(raw)
    assert historical["recovery_check_readback"]["path"].startswith("/rsrch8/scratch/")
    assert historical["recovery_check_terminal"]["path"].startswith("/rsrch8/scratch/")
    plan = json.loads((frozen / "bundle/PLAN.json").read_bytes())
    bundle = Path(plan["bundle_root"])
    with pytest.raises(ValueError, match="outside the visible home surface"):
        build.validate_worker_visible_references(historical, bundle, bundle.parent.parent)


def test_archive_inventory_hardware_and_no_automatic_launch(package_factory):
    payload, archive, _, _ = package_factory()
    inventory = json.loads(payload["INVENTORY.json"])
    assert set(inventory) == set(payload) - {"INVENTORY.json"}
    assert all(common.identity(payload[name]) == record for name, record in inventory.items())
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        assert {item.name for item in tar.getmembers()} == {"bundle/" + name for name in payload}
        assert all(item.isfile() and item.uid == item.gid == item.mtime == 0 for item in tar.getmembers())
    job = json.loads(payload["gpu-job.json"])
    spec, pod = job["spec"], job["spec"]["template"]["spec"]
    assert spec["suspend"] is True and spec["backoffLimit"] == 0
    assert spec["parallelism"] == spec["completions"] == 1
    assert "activeDeadlineSeconds" not in spec and "activeDeadlineSeconds" not in pod
    assert len(pod["containers"]) == 1
    for term in pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]:
        products = next(row for row in term["matchExpressions"] if row["key"] == "nvidia.com/gpu.product")
        assert products["values"] == ["NVIDIA-H100-80GB-HBM3"]
    for name, raw in payload.items():
        if name.endswith(".py") and not name.startswith(("source/", "retained_reader/")):
            ast.parse(raw, feature_version=(3, 6))


@pytest.mark.parametrize("field,value", [
    ("continuation_authority_sha256", "0" * 64), ("source_manifest_sha256", "0" * 64),
    ("storage_reserved_bytes", 4 * common.GIB), ("scientific_update_reservation", 20000),
    ("live_h100_reservations", 5), ("status", "complete"),
])
def test_builder_rejects_unbound_scope(package_factory, field, value):
    admission = fabricated_admission("D1D2")
    admission[field] = value
    with pytest.raises(ValueError):
        package_factory(admission=admission)


def test_builder_rejects_zero_restart_other_subset_and_other_pair(package_factory):
    for work in (WorkBinding.representation(("D1", "D2")),
                 WorkBinding.representation(("D1", "D3"), 20000, 1280)):
        admission = fabricated_admission("D1D2")
        admission["work_binding"] = work.to_dict()
        with pytest.raises(ValueError):
            package_factory(admission=admission)
    with pytest.raises(ValueError, match="four consumed-work"):
        build.construct(ROOT, "D3D4", fabricated_admission("D1D2"))


@pytest.mark.parametrize("name", ["frontier/observation_continuation.py", "frontier/observation_recovery.py"])
def test_extra_source_mutation_fails_before_worker_launch(package_factory, name):
    payload, _, _, bundle = package_factory()
    path = bundle / "source" / name
    path.write_bytes(path.read_bytes() + b"\n# disposable fixture mutation\n")
    with pytest.raises(ValueError, match="hash differs"):
        common.verify_bundle(json.loads(payload["PLAN.json"]), bundle)


@pytest.mark.parametrize("field,value", [
    ("maximum_new_scientific_updates", 20000), ("combined_output_limit_bytes", 8 * common.GIB),
    ("maximum_output_bytes", 4 * common.GIB), ("automatic_successor", True),
    ("automatic_retries", 1), ("elapsed_time_deadline_seconds", 86400),
])
def test_lifecycle_cannot_expand_continuation(package_factory, field, value):
    payload, _, _, _ = package_factory()
    plan = deepcopy(json.loads(payload["PLAN.json"]))
    plan[field] = value
    with pytest.raises(ValueError):
        common.validate_plan(plan)


@pytest.fixture
def mocked_actor(tmp_path, monkeypatch):
    """Run orchestration against disposable metadata and an in-memory journal."""
    monkeypatch.syspath_prepend(str(ROOT / "operations"))
    spec = importlib.util.spec_from_file_location("frontier_test_continuation_actor", ROOT / "operations/continue_observation.py")
    actor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor)
    local = tmp_path / ROOT.name
    local.mkdir()
    files = ("AUTHORITY.json", "OBSERVATION_CONTINUATION_V3_AUTHORITY.json",
        "receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json",
        "receipts/OBSERVATION_CONTINUATION_V3_CAPACITY_20261016.json",
        "receipts/RETENTION_CHECK_V2_STATUS_20261016T161210Z.json",
        "receipts/OBSERVATION_RETENTION_RECOVERY_BYTE_INTAKE_20261016_V1.json",
        "admission/IDENTITY.json")
    for name in files:
        destination = local / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    for key, relative in (("ROOT", "."), ("BATCH", "observation_continuation_v3_admission"),
        ("AUTHORITY", "OBSERVATION_CONTINUATION_V3_AUTHORITY.json"),
        ("INTAKE", "receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json"),
        ("CAPACITY", "receipts/OBSERVATION_CONTINUATION_V3_CAPACITY_20261016.json"),
        ("CHECK", "receipts/RETENTION_CHECK_V2_STATUS_20261016T161210Z.json")):
        monkeypatch.setattr(actor, key, local / relative)
    intake = json.loads(actor.INTAKE.read_bytes())
    check = json.loads(actor.CHECK.read_bytes())
    tasks = {check["run_id"]: dict(status="complete", storage_reserved_bytes=4 * common.GIB,
                                 scientific_update_reservation=0)}
    real_package = actor.ops.package
    parents = {}
    for pair in build.PAIRS:
        parents[pair] = real_package(ROOT / ("observation_" + pair.lower() + "_v2_launch"))
        run = parents[pair][1]["run_id"]
        tasks[run] = dict(status="failed", scheduler_id=intake["fits"][pair]["terminal"]["value"]["uid"],
            evidence_sha256="c" * 64, scientific_update_reservation=20000, storage_reserved_bytes=4 * common.GIB)
    calls = []
    state = dict(tasks=tasks, last_sha256="a" * 64)

    class FakeJournal:
        def __init__(self, path, **kwargs):
            assert path == local / "admission"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def snapshot(self):
            return deepcopy(state)
        def continue_failed_representation(self, old, new, **kwargs):
            calls.append((old, new, deepcopy(kwargs)))
            pair = next(p for p in build.PAIRS if parents[p][1]["run_id"] == old)
            donors = json.loads((parents[pair][0] / "bundle/WORKER_CONTRACT.json").read_bytes())["training_donors"]
            tasks[old].update(scientific_update_reservation=1280, storage_reserved_bytes=kwargs["retained_parent_bytes"])
            tasks[new] = dict(status="reserved", scientific_update_reservation=18720,
                storage_reserved_bytes=kwargs["output_bytes"], continues_task_id=old,
                work_binding=WorkBinding.representation(donors, 20000, 1280).to_dict(),
                continuation_binding=dict(authority_sha256=kwargs["authority_sha256"],
                                          checkpoint_sha256=kwargs["checkpoint_sha256"]))
            return self.snapshot()

    monkeypatch.setattr(actor, "AdmissionJournal", FakeJournal)
    monkeypatch.setattr(actor, "record", common.publish)
    monkeypatch.setattr(actor, "source_manifest", lambda unused: build.source_manifest(ROOT))
    monkeypatch.setattr(actor.ops, "package", lambda path: real_package(ROOT / path.name))
    return actor, calls, state, intake


def test_actor_reserve_passes_exact_remaining_bindings_without_real_journal_mutation(mocked_actor):
    actor, calls, state, intake = mocked_actor
    result = actor.reserve()
    assert result["new_updates"] == 74880 and result["consumed_updates"] == 5120
    assert len(calls) == 4
    for pair, (_, _, kwargs), row in zip(build.PAIRS, calls, result["fits"]):
        assert kwargs["accepted_updates"] == 1280
        assert kwargs["retained_parent_bytes"] == intake["fits"][pair]["parent_output_bytes"]
        assert kwargs["output_bytes"] + kwargs["retained_parent_bytes"] == 4 * common.GIB
        assert kwargs["retained_inventory_sha256"] == intake["fits"][pair]["inventory_sha256"]
        admission = json.loads(Path(row["path"]).read_bytes())
        assert admission["scientific_update_reservation"] == 18720
        assert admission["live_h100_reservations"] == 4
        assert admission["work_binding"]["start_updates"] == 1280
        assert common.identity(Path(row["path"]).read_bytes())["sha256"] == row["sha256"]
    assert sum(t["scientific_update_reservation"] for t in state["tasks"].values()) == 80000
    with pytest.raises(ValueError, match="already has admission"):
        actor.reserve()
    assert len(calls) == 4


def test_actor_gate_requires_exact_consumed_parent_and_binding(mocked_actor, monkeypatch):
    actor, _, state, _ = mocked_actor
    result = actor.reserve()
    admission = json.loads(Path(result["fits"][0]["path"]).read_bytes())
    monkeypatch.setattr(actor.ops, "admission_gate", lambda *unused: admission)
    report = dict(run_id=admission["run_id"])
    assert actor.gate(Path("/fixture"), report, {}) == admission
    state["tasks"][admission["run_id"]]["continues_task_id"] = "wrong-parent"
    with pytest.raises(ValueError, match="continuation reservation"):
        actor.gate(Path("/fixture"), report, {})
