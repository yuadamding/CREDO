"""A qualified successor reads home receipts without assuming scratch visibility."""
# ruff: noqa: E402
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]

import pytest

import gpu_native_control as controller
import gpu_native_worker as worker
from prepare_manifest import canonical


def reference(path):
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


@pytest.fixture
def completed_parent(tmp_path, monkeypatch):
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    output, control = home / "qual/output", scratch / "qual/gpu"
    qbundle, successor = home / "qual/bundle", home / "fit/bundle"
    qplan = dict(phase="qualification", run_id="qual", gpu_contract_sha256="c"*64,
        worker_output=str(output), equal_input_Adam_qualification_required=True)
    publish(qbundle / "PLAN.json", qplan)
    qsha = reference(qbundle / "PLAN.json")["sha256"]
    fplan = dict(phase="fitting", run_id="fit", gpu_contract_sha256="c"*64,
        bundle_root=str(successor), qualification_binding=dict(plan_sha256=qsha,
            output=str(output), control=str(control), worker_gate_path=str(home / "qual/QUALIFIED_FITTING_GATE.json")))
    publish(successor / "PLAN.json", fplan)
    publish(successor / "control.py", {})
    publish(successor / "INVENTORY.json", {name: reference(successor / name) for name in ("PLAN.json", "control.py")})
    complete = dict(status="qualified_gpu_native_full_ddr_execution", run_id="qual",
        plan_sha256=qsha, gpu_contract_sha256="c"*64, input_lease_exit_verified=True,
        budget_used=dict(full=90, partial=12, steps=18, refresh=3, chunks=549),
        selected_profile="GPU_native_eager_foreach")
    publish(output / "COMPLETE.json", complete)
    publish(control / "terminal.json", dict(status="passed", uid="parent-uid", plan_sha256=qsha,
        result_complete_sha256=reference(output / "COMPLETE.json")["sha256"]))
    publish(control / "cleanup.json", dict(status="complete", uid="parent-uid",
        reads=[dict(job_absent=True, owned_pods=0)]*2))
    chain = dict(qualification_plan_sha256=qsha, successor_bundle=str(successor),
        successor_plan_sha256=reference(successor / "PLAN.json")["sha256"],
        successor_inventory_sha256=reference(successor / "INVENTORY.json")["sha256"], successor_run_id="fit")
    publish(qbundle / "PHASE_CHAIN.json", chain)
    def checked_file(path, retain=False):
        path = Path(path)
        return reference(path), path.read_bytes() if retain else b""
    monkeypatch.setattr(controller.base, "checked_file", checked_file)
    monkeypatch.setattr(controller.base, "BASE", qbundle)
    monkeypatch.setattr(controller.base, "core", SimpleNamespace(
        sha=lambda path: reference(Path(path))["sha256"], publish=publish), raising=False)
    launches = []
    def run(command, **kwargs):
        # The proof must already exist when the sole create edge is reached.
        assert Path(fplan["qualification_binding"]["worker_gate_path"]).is_file()
        launches.append(command[2])
        return SimpleNamespace(returncode=0,
            stdout=b'{"status":"healthy_detached_activated","job":"fit","uid":"fit-uid"}\n', stderr=b"")
    monkeypatch.setattr(controller.subprocess, "run", run)
    return SimpleNamespace(qplan=qplan, fplan=fplan, root=control, output=output,
        scratch=scratch, chain=chain, launches=launches, complete=complete)


def block_worker_scratch(monkeypatch, scratch):
    identity, checked = worker.file_identity, worker.checked_json
    seen = []
    def reject(path):
        path = Path(path)
        assert scratch not in path.parents, "GPU worker attempted a scratch read"
        seen.append(path)
    def checked_identity(path, **kwargs):
        reject(path)
        return identity(path, **kwargs)
    def checked_json(path, expected):
        reject(path)
        return checked(path, expected)
    monkeypatch.setattr(worker, "file_identity", checked_identity)
    monkeypatch.setattr(worker, "checked_json", checked_json)
    return seen


def test_real_successor_edge_publishes_home_gate_and_worker_never_reads_scratch(completed_parent, monkeypatch):
    case = completed_parent
    controller.launch_successor(case.qplan, dict(uid="parent-uid"), case.root)
    assert case.launches == ["create", "launch"]
    seen = block_worker_scratch(monkeypatch, case.scratch)
    complete, refs = worker.qualification_gate(case.fplan)
    assert complete == case.complete
    assert refs["terminal"] == reference(case.root / "terminal.json")
    assert refs["cleanup"] == reference(case.root / "cleanup.json")
    assert "worker_gate" in refs and seen
    # A second owner cannot overwrite the completed handoff or create again.
    with pytest.raises(FileExistsError):
        controller.launch_successor(case.qplan, dict(uid="parent-uid"), case.root)
    assert case.launches == ["create", "launch"]


@pytest.mark.parametrize("damage", ["terminal_hash", "failed_terminal", "wrong_parent", "wrong_successor", "complete_changed"])
def test_worker_rejects_corrupt_or_mismatched_embedded_receipts(completed_parent, monkeypatch, damage):
    case = completed_parent
    controller.launch_successor(case.qplan, dict(uid="parent-uid"), case.root)
    path = Path(case.fplan["qualification_binding"]["worker_gate_path"])
    gate = json.loads(path.read_bytes())
    if damage == "terminal_hash":
        gate["artifacts"]["terminal"]["reference"]["sha256"] = "0"*64
    elif damage == "failed_terminal":
        artifact = gate["artifacts"]["terminal"]
        value = json.loads(artifact["raw_json"])
        value["status"] = "failed"
        raw = canonical(value)
        artifact.update(raw_json=raw.decode(), reference=dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
    elif damage == "wrong_parent":
        gate["parent_job_uid"] = "other-uid"
    elif damage == "wrong_successor":
        gate["successor_plan_sha256"] = "0"*64
    else:
        changed = copy.deepcopy(case.complete)
        changed["selected_profile"] = "GPU_native_compiled_foreach"
        (case.output / "COMPLETE.json").write_bytes(canonical(changed))
    path.write_bytes(canonical(gate))
    block_worker_scratch(monkeypatch, case.scratch)
    with pytest.raises(RuntimeError):
        worker.qualification_gate(case.fplan)


def test_failed_parent_never_publishes_gate_or_creates_successor(completed_parent):
    case = completed_parent
    path = case.root / "terminal.json"
    value = json.loads(path.read_bytes())
    value["status"] = "failed"
    path.write_bytes(canonical(value))
    with pytest.raises(RuntimeError, match="not qualified"):
        controller.launch_successor(case.qplan, dict(uid="parent-uid"), case.root)
    assert not Path(case.fplan["qualification_binding"]["worker_gate_path"]).exists()
    assert case.launches == []
