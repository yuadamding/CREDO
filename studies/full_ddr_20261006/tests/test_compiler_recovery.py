"""Compiler-only recovery regressions; synthetic CPU, no remote actions."""
# ruff: noqa: E402
import ast
import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]

import pytest

import compiler_toolchain as tool
import gpu_native_compiler_launch as launch
import gpu_native_compiler_operate as operate
import gpu_native_worker
from prepare_launch import canonical, sha


def archive(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:xz") as handle:
        for name, kind, raw in entries:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = len(raw) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                entry.linkname = "outside"
            handle.addfile(entry, io.BytesIO(raw) if kind == tarfile.REGTYPE else None)
    return output.getvalue()


def minimal(extra=()):
    return archive([(tool.ROOT_NAME + "/zig", tarfile.REGTYPE, b"binary"),
                    (tool.ROOT_NAME + "/LICENSE", tarfile.REGTYPE, b"license"), *extra])


@pytest.mark.parametrize("name,kind", [
    ("/absolute", tarfile.REGTYPE), (tool.ROOT_NAME + "/../outside", tarfile.REGTYPE),
    ("other/zig", tarfile.REGTYPE), (tool.ROOT_NAME + "/./zig", tarfile.REGTYPE),
    (tool.ROOT_NAME + "/zig", tarfile.REGTYPE), (tool.ROOT_NAME + "/link", tarfile.SYMTYPE),
    (tool.ROOT_NAME + "/link", tarfile.LNKTYPE), (tool.ROOT_NAME + "/device", tarfile.CHRTYPE),
])
def test_compiler_archive_rejects_unsafe_members(name, kind):
    with pytest.raises(RuntimeError, match="Unsafe"):
        tool.archive_inventory(minimal([(name, kind, b"bad")]))


def test_compiler_archive_caps_and_required_license():
    with pytest.raises(RuntimeError, match="cap exceeded"):
        tool.archive_inventory(minimal(), maximum=5)
    with pytest.raises(RuntimeError, match="Incomplete"):
        tool.archive_inventory(archive([(tool.ROOT_NAME + "/zig", tarfile.REGTYPE, b"binary")]))


def test_compiler_extract_is_exclusive_and_preserves_bytes(tmp_path):
    raw = minimal()
    inventory = tool.archive_inventory(raw)
    root = tmp_path / "compiler"
    binary = tool.unpack(raw, inventory, root)
    assert binary.read_bytes() == b"binary"
    assert binary.stat().st_mode & 0o777 == 0o500
    assert (binary.parent / "LICENSE").stat().st_mode & 0o777 == 0o400
    with pytest.raises(RuntimeError, match="no overwrite or retry"):
        tool.unpack(raw, inventory, root)
    altered = copy.deepcopy(inventory)
    altered[tool.ROOT_NAME + "/zig"]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="inventory differs"):
        tool.unpack(raw, altered, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def test_compiler_read_refuses_symlinks_hash_and_oversize(tmp_path):
    actual = tmp_path / "actual"
    actual.write_bytes(b"small")
    link = tmp_path / "link"
    link.symlink_to(actual)
    with pytest.raises(RuntimeError, match="regular"):
        tool.checked_file(link)
    with pytest.raises(RuntimeError, match="hash differs"):
        tool.checked_file(actual, "wrong")
    with pytest.raises(RuntimeError, match="regular"):
        tool.checked_file(actual, maximum=1)
    assert tool.checked_file(actual, tool.digest(b"small")) == b"small"


def test_compiler_wrappers_use_absolute_exec_no_shell_and_eight_cores(tmp_path, monkeypatch):
    root = tmp_path / "compiler"
    root.mkdir()
    monkeypatch.setattr(tool.os, "environ", dict())
    refs = tool.bind_wrappers(root / "zig", root, python_executable=sys.executable)
    assert tool.os.environ["CC"] == str(root / "cc")
    assert tool.os.environ["CXX"] == str(root / "cxx")
    assert tool.os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "8"
    for name, mode in (("cc", "cc"), ("cxx", "c++")):
        raw = (root / name).read_bytes()
        assert refs[name]["sha256"] == tool.digest(raw)
        ast.parse(raw)
        assert "os.execv(" in raw.decode() and repr(mode) in raw.decode()
        assert "[:8]" in raw.decode() and "shell" not in raw.decode()
        assert tool.TARGET in raw.decode()
    with pytest.raises(FileExistsError):
        tool.bind_wrappers(root / "zig", root)


def test_unbound_triton_version_fails_before_any_native_build(tmp_path, monkeypatch):
    import triton
    monkeypatch.setattr(triton, "__version__", "unqualified")
    with pytest.raises(RuntimeError, match="Pinned-image"):
        tool.native_probes(ROOT, tmp_path)
    assert not (tmp_path / "probe").exists()


def gate_fixture(tmp_path):
    complete = dict(status="compiler_dependency_surface_verified", compiler_spec_sha256="spec",
        archive_sha256=tool.ARCHIVE_SHA, native_builds=2, python_extension_loaded=True, cpp_library_loaded=True,
        cohort_access=False, scientific_optimizer_updates=0, python_version="3.12.13",
        torch_package_version="2.6.0+cu124", triton_version="3.2.0")
    gate = dict(status="passed_same_image_compiler_prerequisite", image="image", qualification_plan_sha256="q",
        compiler_spec_sha256="spec", compiler_archive_sha256=tool.ARCHIVE_SHA, job="cpu", uid="owned",
        native_builds=2, data_free=True, GPU_requests=0, visibility_mask="none", scientific_optimizer_updates=0,
        cleanup_reads=[dict(job_absent=True, owned_pods=0)]*2,
        compiler_complete_sha256=sha(canonical(complete)))
    plan = dict(phase="qualification", image="image", compiler_spec_sha256="spec", compiler_preflight_job="cpu",
        compiler_preflight_gate_path=str(tmp_path / "COMPILER_PASS.json"), compiler_preflight_qualification_plan_sha256="q")
    (tmp_path / "COMPLETE.json").write_bytes(canonical(complete))
    (tmp_path / "COMPILER_PASS.json").write_bytes(canonical(gate))
    return plan, gate, complete


@pytest.mark.parametrize("phase", ["qualification", "fitting"])
def test_complete_same_image_compiler_gate_works_for_both_phases(tmp_path, phase):
    plan, gate, _ = gate_fixture(tmp_path)
    plan["phase"] = phase
    ref = tool.verified_compiler_gate(plan, current_plan_sha256="q")
    assert ref["sha256"] == sha(canonical(gate))


@pytest.mark.parametrize("key,value", [
    ("image", "changed"), ("qualification_plan_sha256", "changed"), ("native_builds", 1),
    ("GPU_requests", 1), ("data_free", False), ("scientific_optimizer_updates", 1),
    ("cleanup_reads", [dict(job_absent=True, owned_pods=0)]), ("compiler_complete_sha256", "changed"),
])
def test_compiler_gate_rejects_missing_image_identity_build_or_cleanup(tmp_path, key, value):
    plan, gate, _ = gate_fixture(tmp_path)
    gate[key] = value
    (tmp_path / "COMPILER_PASS.json").write_bytes(canonical(gate))
    with pytest.raises(RuntimeError):
        tool.verified_compiler_gate(plan, current_plan_sha256="q")


@pytest.mark.parametrize("key,value", [
    ("python_version", "3.13"), ("torch_package_version", "2.9"), ("triton_version", "3.5"),
    ("cpp_library_loaded", False), ("cohort_access", True),
])
def test_compiler_gate_rejects_unqualified_native_abi(tmp_path, key, value):
    plan, gate, complete = gate_fixture(tmp_path)
    complete[key] = value
    gate["compiler_complete_sha256"] = sha(canonical(complete))
    (tmp_path / "COMPLETE.json").write_bytes(canonical(complete))
    (tmp_path / "COMPILER_PASS.json").write_bytes(canonical(gate))
    with pytest.raises(RuntimeError, match="native/ABI"):
        tool.verified_compiler_gate(plan, current_plan_sha256="q")


@pytest.fixture(scope="module")
def payloads():
    from frozen_payload import frozen_payload
    return tuple(frozen_payload(ROOT / name) for name in ("gpu_native_qualification_v3", "fitting_gpu_native_v3"))


def test_compiler_recovery_keeps_science_image_budget_and_historical_neutral_identity(payloads):
    q, f = payloads
    previous = json.loads((ROOT / "gpu_native_qualification_v2/bundle/GPU_CONTRACT.json").read_bytes())
    for phase, payload in (("qualification", q), ("fitting", f)):
        plan, contract, authority = (json.loads(payload[name]) for name in ("PLAN.json", "GPU_CONTRACT.json", "EXECUTION_AUTHORITY.json"))
        gpu_native_worker.validate_plan(plan, contract, authority)
        for key in ("information_contract", "architectures", "resources", "source_sha256"):
            assert contract[key] == previous[key]
        assert contract["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] == 138
        assert contract["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] == 32917
        assert payload["static-job.json"] == (ROOT / "gpu_native_qualification_v2/bundle/static-job.json").read_bytes()
        assert payload["toolchain.tar.xz"] == (launch.RECOVERY / "zig-linux-x86_64-0.13.0.tar.xz").read_bytes()
        job = json.loads(payload["gpu-job.json"])
        worker = job["spec"]["template"]["spec"]["containers"][0]
        env = {entry["name"]: entry["value"] for entry in worker["env"] if "value" in entry}
        assert env["CREDO_PLAN_SHA256"] == sha(payload["PLAN.json"])
        assert env["CC"] == "/tmp/credo-compiler/cc" and env["CXX"] == "/tmp/credo-compiler/cxx"
        original_job = json.loads((ROOT / "gpu_native_qualification_v2/bundle/gpu-job.json").read_bytes())
        assert worker["resources"] == original_job["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert worker["resources"]["requests"]["nvidia.com/gpu"] == "1"
        assert plan["compiler_dependency_surface_required"] is True
        assert "activeDeadlineSeconds" not in job["spec"]
        if phase == "fitting":
            assert plan["qualification_binding"]["plan_sha256"] == sha(q["PLAN.json"])
            assert plan["compiler_preflight_qualification_plan_sha256"] == sha(q["PLAN.json"])
            assert plan["maximum_optimizer_updates"] == 23040 and len(plan["fits"]) == 9


def test_new_compiler_cpu_command_has_its_own_strict_zero_gpu_admission(payloads):
    namespace = {}
    exec((ROOT / "gpu_native_qualification_v2/bundle/neutral_admission.py").read_bytes(), namespace)
    q, _ = payloads
    manifest = json.loads(q["compiler-job.json"])
    namespace["NEUTRAL"] = manifest["spec"]["template"]["spec"]["containers"][0]["command"]
    live = copy.deepcopy(manifest)
    spec = live["spec"]["template"]["spec"]
    spec["nodeSelector"] = {"gpu-type": "A100"}
    spec["containers"][0]["resources"] = copy.deepcopy(namespace["LIVE"])
    namespace["audit_neutral"](live, manifest)
    spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "1"
    with pytest.raises(RuntimeError, match="resource mapping"):
        namespace["audit_neutral"](live, manifest)


def test_compiler_gate_and_native_probe_precede_release_and_cohort_access():
    tree = ast.parse(inspect.getsource(gpu_native_worker.main))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)
    assert max(calls["verified_compiler_gate"]) < min(calls["prepare_compiler"])
    assert max(calls["prepare_compiler"]) < min(calls["wait_for_release"]) < min(calls["loader"])


def test_compiler_host_controller_is_python36_syntax_and_no_deadline():
    raw = (ROOT / "compiler_preflight_control.py").read_bytes()
    ast.parse(raw, feature_version=(3, 6))
    ast.parse(operate.LIVE_COMPILER_GATE, feature_version=(3, 6))
    assert b"timeout=" not in raw
    assert b"core.cleanup(binding" in raw and b"core.pods(name, None)" in raw


def test_real_compiler_lifecycle_import_initializes_without_cli_or_scheduler():
    code = """
import importlib.util,json
from pathlib import Path
import control
assert not hasattr(control.base,'core')
spec=importlib.util.spec_from_file_location('new_compiler_controller',SOURCE)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.core is None
core=module.load_core(Path(BUNDLE))
assert control.base.core is core
assert core.MODE is None
assert callable(core.get_job) and callable(core.cleanup) and callable(core.audit_runtime_pod)
print(json.dumps(dict(status='actual_lifecycle_initialized_no_scheduler_calls')))
"""
    bundle = str(ROOT / "gpu_native_qualification_v3/bundle")
    prefix = "import sys\nsys.path.insert(0," + repr(bundle) + ")\nSOURCE=" + repr(str(ROOT / "compiler_preflight_control.py")) + "\nBUNDLE=" + repr(bundle) + "\n"
    result = subprocess.run([sys.executable, "-c", prefix + code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "actual_lifecycle_initialized_no_scheduler_calls"


def test_historical_compiler_correction_both_phase_bindings_in_isolated_process():
    code = """
import json,sys
import gpu_native_compiler_recovery_launch as new
import gpu_native_worker
from prepare_launch import sha
sys.path.insert(0,str(new.ROOT/'tests'))
from frozen_payload import frozen_payload
q=frozen_payload(new.ROOT/'gpu_native_qualification_v4')
f=frozen_payload(new.ROOT/'fitting_gpu_native_v4')
for phase,payload,run in [('qualification',q,new.QUAL_NAME),('fitting',f,new.FIT_NAME)]:
 plan=json.loads(payload['PLAN.json'])
 gpu_native_worker.validate_plan(plan,json.loads(payload['GPU_CONTRACT.json']),json.loads(payload['EXECUTION_AUTHORITY.json']))
 assert plan['run_id']==run and plan['bundle_root']==new.roots(phase)[0]+'/bundle'
 assert plan['compiler_preflight_job']==new.CPU_NAME
 assert json.loads(payload['compiler-job.json'])['metadata']['name']==new.CPU_NAME
 assert payload['compiler_preflight_control.py']==(new.ROOT/'compiler_preflight_control.py').read_bytes()
 assert 'controller/gpu_native_compiler_recovery_operate.py' in payload
 assert plan['implementation_sha256']==sha(payload['IMPLEMENTATION.json'])
 if phase=='fitting':assert plan['qualification_binding']['plan_sha256']==sha(q['PLAN.json'])
print(json.dumps(dict(status='corrected_source_homogeneous_phases_verified_no_submission')))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "corrected_source_homogeneous_phases_verified_no_submission"


@pytest.mark.parametrize("success", [True, False])
def test_compiler_controller_pass_or_failure_captures_and_cleans_without_retry(tmp_path, monkeypatch, success):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    control_root = tmp_path / "control"
    control_root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    plan = dict(control_root=str(control_root), compiler_preflight_job="cpu", compiler_preflight_output=str(output),
        compiler_preflight_gate_path=str(output / "COMPILER_PASS.json"), compiler_spec_sha256="spec", image="image")
    manifest = dict(metadata=dict(name="cpu", uid="owned"))
    (bundle / "PLAN.json").write_bytes(canonical(plan))
    (bundle / "compiler-job.json").write_bytes(canonical(manifest))
    complete = dict(compiler_spec_sha256="spec", plan_sha256="hash", image="image", native_builds=2,
        status="compiler_dependency_surface_verified", data_free=True, cohort_access=False,
        python_extension_loaded=True, cpp_library_loaded=True, python_version="3.12.13",
        torch_package_version="2.6.0+cu124", triton_version="3.2.0", gpu_requests=0, gpu_limits=0,
        visibility_mask="none", scientific_optimizer_updates=0, archive_sha256=tool.ARCHIVE_SHA)
    (output / "COMPLETE.json").write_bytes(canonical(complete))
    calls, reads = [], 0
    def get_job(name):
        nonlocal reads
        reads += 1
        return None if reads <= 2 else manifest
    def kube(*args):
        calls.append(tuple(args))
        return json.dumps(manifest)
    def publish(path, value):
        path.write_bytes(canonical(value))
    def read(path):
        return json.loads(path.read_bytes()), dict(sha256="hash", size_bytes=1)
    def cleanup(binding, root, **kwargs):
        calls.append(("cleanup",))
        publish(root / "cleanup.json", dict(reads=[dict(job_absent=True, owned_pods=0)]*2))
    core = SimpleNamespace(get_job=get_job, pods=lambda *args: [], kube=kube, publish=publish,
        job_contract=lambda *args: {}, verify=lambda *args: None, activate=lambda *args: None,
        complete=lambda *args: (success, not success), capture=lambda *args: [{}],
        audit_runtime_pod=lambda *args: None, cleanup=cleanup, sha=lambda *args: "hash")
    monkeypatch.setitem(sys.modules, "control", SimpleNamespace(base=SimpleNamespace(core=core), read=read))
    monkeypatch.setitem(sys.modules, "neutral_admission", SimpleNamespace(audit_neutral=lambda *args: None))
    monkeypatch.setitem(sys.modules, "runtime_policy", SimpleNamespace(validate_job=lambda *args: None, validate_plan=lambda *args: None))
    spec = importlib.util.spec_from_file_location("synthetic_compiler_controller", ROOT / "compiler_preflight_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(bundle / "compiler_preflight_control.py"))
    monkeypatch.setattr(module, "load_core", lambda *args: core)
    if success:
        module.main()
        assert json.loads((output / "COMPILER_PASS.json").read_bytes())["GPU_requests"] == 0
    else:
        with pytest.raises(RuntimeError, match="worker failed"):
            module.main()
        assert not (output / "COMPILER_PASS.json").exists()
    assert sum(row[0] == "create" and "--dry-run=server" not in row for row in calls) == 1
    assert sum(row[0] == "cleanup" for row in calls) == 1
    assert (control_root / "compiler/process.json").is_file()
