"""CPU-only operation guards; all remote calls mocked."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_operate as ops


def fixture_build(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "BASE", tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    plan = dict(run_id=ops.RUN_ID, control_root=ops.CONTROL_ROOT,
                bundle_root=ops.HOME_RUN + "/bundle")
    raw = ops.canonical(plan)
    (bundle / "PLAN.json").write_bytes(raw)
    (tmp_path / "bundle.tar.gz").write_bytes(b"fixture-archive")
    build = dict(run_id=ops.RUN_ID, plan_sha256=ops.digest(raw),
                 archive_sha256=ops.digest(b"fixture-archive"),
                 files={"PLAN.json": dict(sha256=ops.digest(raw), size_bytes=len(raw))})
    (bundle / "controller").mkdir()
    for name in ("prepare_operate.py", "prepare_launch.py"):
        code = (ops.ROOT / name).read_bytes()
        (bundle / "controller" / name).write_bytes(code)
        build["files"]["controller/" + name] = dict(sha256=ops.digest(code), size_bytes=len(code))
    (tmp_path / "BUILD.json").write_bytes(ops.canonical(build))
    return build, plan


def test_exact_build_and_tamper(tmp_path, monkeypatch):
    expected = fixture_build(tmp_path, monkeypatch)
    assert ops.verified_build() == expected
    (tmp_path / "bundle/PLAN.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        ops.verified_build()


def test_inventory_traversal_rejected(tmp_path, monkeypatch):
    build, _ = fixture_build(tmp_path, monkeypatch)
    build["files"]["../escape"] = {}
    (tmp_path / "BUILD.json").write_bytes(ops.canonical(build))
    with pytest.raises(ValueError, match="Unsafe"):
        ops.verified_build()


def test_receipts_are_no_clobber(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "BASE", tmp_path)
    ops.record("INTENT.json", {"value": 1})
    with pytest.raises(FileExistsError):
        ops.record("INTENT.json", {"value": 2})
    assert json.loads((tmp_path / "INTENT.json").read_text()) == {"value": 1}
    with pytest.raises(ValueError):
        ops.record("../escape", {})


def test_strict_chain_no_backend_override():
    command = ops.ssh_command(["/usr/bin/python", "-c", "print('safe')"])
    assert "StrictHostKeyChecking=yes" in command
    assert "ForwardAgent=no" in command
    assert command[-2] == "work-mac"
    assert "company-seadragon.conf" in command[-1]
    assert "HostName=" not in command[-1]


def test_remote_receipt_and_ambiguous_response(monkeypatch):
    monkeypatch.setattr(ops, "call", lambda *a, **k: 'banner\nRECEIPT={"ok":true}\n')
    assert ops.remote("pass") == {"ok": True}
    monkeypatch.setattr(ops, "call", lambda *a, **k: 'RECEIPT={}\nRECEIPT={}\n')
    with pytest.raises(RuntimeError, match="Ambiguous"):
        ops.remote("pass")


def test_no_automatic_retry(monkeypatch):
    calls = []

    def failure(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, b"", b"failure")

    monkeypatch.setattr(ops.subprocess, "run", failure)
    with pytest.raises(RuntimeError, match="without retry"):
        ops.call(["true"])
    assert len(calls) == 1


@pytest.mark.parametrize("deny", ["status", "mode", "namespace", "required"])
def test_session_rejects_unqualified(tmp_path, monkeypatch, deny):
    monkeypatch.setattr(ops, "BASE", tmp_path)
    value = dict(status="ready", mode="mutate", namespace="yn-gpu-workload",
                 context="yding4_yn-gpu-workload@kubernetes-admin@kubernetes",
                 required={"jobs": {"allowed": True}})
    value[deny] = {"jobs": {"allowed": False}} if deny == "required" else "incorrect"
    (tmp_path / "SESSION_PREFLIGHT.json").write_bytes(ops.canonical(value))
    with pytest.raises(ValueError):
        ops.session_gate()


def test_host_programs_parse_as_python36():
    import ast
    module = ast.parse(Path(ops.__file__).read_text())
    scripts = [node.value for node in ast.walk(module) if isinstance(node, ast.Constant)
               and isinstance(node.value, str) and "RECEIPT=" in node.value and "\nimport " in node.value]
    assert len(scripts) >= 5
    for script in scripts:
        ast.parse(script, feature_version=(3, 6))
