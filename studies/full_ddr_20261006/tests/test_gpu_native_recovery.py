"""Dependency-layout and cleanup-history regressions; synthetic CPU only."""
# ruff: noqa: E402
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]

import pytest

import gpu_native
import gpu_native_worker
from gpu_native_recovery_launch import QUAL_NAME, FIT_NAME, assemble_phase, recovery_contract, roots
from prepare_launch import sha


@pytest.fixture(autouse=True)
def clear_dependency_cache():
    gpu_native.geomloss_backend.cache_clear()
    yield
    gpu_native.geomloss_backend.cache_clear()


def test_helpers_are_the_exact_public_scalar_tensorized_backend():
    from geomloss import SamplesLoss
    surface = gpu_native.geomloss_backend()
    samples = importlib.import_module(SamplesLoss.__module__.rsplit(".", 1)[0] + ".sinkhorn_samples")
    assert surface.identity["scalar_module"] == SamplesLoss.__module__
    assert surface.identity["version"] in {"0.2.6", "0.3.1"}
    assert gpu_native.geomloss_backend() is surface
    for name in ("sinkhorn_loop", "sinkhorn_cost", "log_weights"):
        assert getattr(surface, name) is samples.sinkhorn_tensorized.__globals__[name]


def test_unknown_layout_is_not_an_import_fallback(monkeypatch):
    import geomloss
    class Unknown:
        pass
    Unknown.__module__ = "geomloss.unqualified.samples_loss"
    monkeypatch.setattr(geomloss, "SamplesLoss", Unknown)
    with pytest.raises(RuntimeError, match="Unqualified GeomLoss scalar backend"):
        gpu_native.geomloss_backend()


def test_mixed_scalar_and_batched_helpers_fail_closed(monkeypatch):
    from geomloss import SamplesLoss
    samples = importlib.import_module(SamplesLoss.__module__.rsplit(".", 1)[0] + ".sinkhorn_samples")
    monkeypatch.setitem(samples.sinkhorn_tensorized.__globals__, "sinkhorn_cost", lambda *args: None)
    with pytest.raises(RuntimeError, match="helper identities differ"):
        gpu_native.geomloss_backend()


def test_incompatible_helper_signature_fails_before_numerical_work(monkeypatch):
    from geomloss import SamplesLoss
    package = SamplesLoss.__module__.rsplit(".", 1)[0]
    module = importlib.import_module(package + ".sinkhorn_divergence")
    samples = importlib.import_module(package + ".sinkhorn_samples")
    def changed():
        raise AssertionError("must not execute a compatibility probe")
    changed.__module__ = module.__name__
    monkeypatch.setattr(module, "sinkhorn_cost", changed)
    monkeypatch.setitem(samples.sinkhorn_tensorized.__globals__, "sinkhorn_cost", changed)
    with pytest.raises(RuntimeError, match="call signature changed"):
        gpu_native.geomloss_backend()


def test_worker_dependency_check_precedes_release_and_cohort_open():
    tree = ast.parse(inspect.getsource(gpu_native_worker.main))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)
    assert max(calls["geomloss_backend"]) < min(calls["wait_for_release"]) < min(calls["loader"])


@pytest.mark.parametrize("history,passed", [
    ([dict(job_absent=True, owned_pods=0)]*2, True),
    ([dict(job_absent=False, owned_pods=0)] + [dict(job_absent=True, owned_pods=0)]*2, True),
    ([dict(job_absent=True, owned_pods=0)], False),
    ([dict(job_absent=True, owned_pods=0), dict(job_absent=False, owned_pods=0)], False),
    ([dict(job_absent=True, owned_pods=0), dict(job_absent=True, owned_pods=1)], False),
])
def test_actual_fitting_qualification_gate_requires_two_final_absence_reads(monkeypatch, history, passed):
    complete = dict(status="qualified_gpu_native_full_ddr_execution", plan_sha256="q", input_lease_exit_verified=True,
        gpu_contract_sha256="g", budget_used=dict(full=90, partial=12, steps=18, refresh=3, chunks=549),
        selected_profile="GPU_native_eager_foreach")
    terminal = dict(status="passed", result_complete_sha256="c", uid="qualification")
    cleanup = dict(status="complete", uid="qualification", reads=history)
    values = {"COMPLETE.json": complete, "terminal.json": terminal, "cleanup.json": cleanup}
    monkeypatch.setattr(gpu_native_worker, "file_identity", lambda path: dict(sha256="c", size_bytes=1))
    monkeypatch.setattr(gpu_native_worker, "checked_json", lambda path, expected: values[Path(path).name])
    plan = dict(gpu_contract_sha256="g", qualification_binding=dict(output="/synthetic/output", control="/synthetic/control/gpu", plan_sha256="q"))
    if passed:
        result, _ = gpu_native_worker.qualification_gate(plan)
        assert result is complete
    else:
        with pytest.raises(RuntimeError, match="cleanup gate unresolved"):
            gpu_native_worker.qualification_gate(plan)


def test_recovery_changes_only_execution_and_preserves_scientific_contract():
    original = json.loads((ROOT / "gpu_native_transition_v1/QUALIFICATION_RELAUNCH_CONTRACT.json").read_bytes())
    current, authority = recovery_contract()
    assert current["information_contract"] == original["information_contract"]
    assert current["architectures"] == original["architectures"]
    assert current["resources"] == original["resources"]
    assert current["qualification"]["loss_and_gradient_checks"] == original["qualification"]["loss_and_gradient_checks"]
    assert current["proposed_replacement"]["maximum_new_scientific_optimizer_updates"] == 23040
    assert current["proposed_replacement"]["maximum_combined_optimizer_operations_including_disposable_steps"] == 32913
    assert current["qualification"]["maximum_combined_loss_backward_calls_including_failed_attempt"] == 115
    assert current["recovery"]["preserved_consumed_work"] == dict(full=11, partial=2, steps=2, refresh=1, chunks=183)
    assert not current["recovery"]["prior_qualification_models_or_partial_results_reused"]
    assert authority["user_instruction"] == "fix all and relaunch"


def test_recovery_both_real_phase_bindings_are_disjoint_from_failed_generation():
    q = assemble_phase("qualification")
    f = assemble_phase("fitting", qualification_plan_sha=sha(q["PLAN.json"]))
    for phase, payload, name in (("qualification", q, QUAL_NAME), ("fitting", f, FIT_NAME)):
        plan, contract, authority = (json.loads(payload[key]) for key in ("PLAN.json", "GPU_CONTRACT.json", "EXECUTION_AUTHORITY.json"))
        gpu_native_worker.validate_plan(plan, contract, authority)
        assert plan["bundle_root"] == roots(phase)[0] + "/bundle"
        assert plan["run_id"] == name and "20261009-v2" in name
        assert plan["geomloss_dependency_surface_required"] is True
        assert plan["gpu_contract_sha256"] == sha(payload["GPU_CONTRACT.json"])
        assert payload["gpu_native.py"] == (ROOT / "gpu_native.py").read_bytes()
        assert payload["worker.py"] == (ROOT / "gpu_native_worker.py").read_bytes()
        assert payload["control.py"] == (ROOT / "gpu_native_control.py").read_bytes()
    assert json.loads(f["PLAN.json"])["qualification_binding"]["control"] == roots("qualification")[1] + "/gpu"
