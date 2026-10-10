"""Small synthetic benchmark and receipt-safety checks, never cohort or CUDA work."""
# ruff: noqa: E402
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src")]

import pytest
import torch

import gpu_native_benchmark as benchmark
from gpu_native import GPUOptions


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_small_synthetic_measurement_preserves_scope_and_count_catalog():
    result = benchmark.run_benchmark(groups=4, particles=4, steps=1, support=4,
                                     hidden_dim=8, embedding_dim=2, repetitions=1)
    assert result["numerical_agreement"] and result["finite_gradients"] and result["finite_objectives"]
    assert result["gradient_parameter_set_agreement"]
    assert result["maximum_absolute_gradient_error"] < 2e-5
    assert result["optimizer_steps"] == result["scientific_optimizer_updates"] == result["real_data_reads"] == 0
    assert not result["full_cohort_training_qualified"]
    assert result["hardware"]["device_type"] == "cpu"
    assert not result["hardware"]["required_hardware_identity_matches"]
    assert not result["sampler_bitwise_scalar_replay"]
    for block in result["count_catalogs"]:
        assert block["donor"] in {"D3", "D4"}
        assert block["categories"] == 26504 and block["observed_zero_categories"] > 0
    for key in ("reference", "optimized", "sampling_reference", "sampling_optimized"):
        assert result[key]["cold_seconds"] > 0 and len(result[key]["warm_seconds"]) == 1
    assert all(len(value) == 64 for value in result["sources"].values())
    assert result["environment"]["torch"] == str(torch.__version__)


@pytest.mark.parametrize("kwargs", [dict(groups=1087), dict(groups=65), dict(particles=129),
    dict(steps=5), dict(support=129), dict(repetitions=6), dict(hidden_dim=513),
    dict(trunk_kind="unknown"), dict(trunk_depth=7), dict(embedding_dim=33),
    dict(options=GPUOptions(precision="tf32")), dict(device="meta")])
def test_workload_and_policy_rejected_before_any_synthetic_allocation(kwargs, monkeypatch):
    def forbidden(**unused):
        raise AssertionError("Data allocation preceded admission")
    monkeypatch.setattr(benchmark, "synthetic_data", forbidden)
    with pytest.raises(ValueError):
        benchmark.run_benchmark(**kwargs)


def test_nonfinite_measurement_remains_failure_with_json_safe_diagnostics(monkeypatch):
    original = benchmark.tensor_rollout
    def damaged(*args, **kwargs):
        output = original(*args, **kwargs)
        return replace(output, drift_steps=output.drift_steps*float("nan"))
    monkeypatch.setattr(benchmark, "tensor_rollout", damaged)
    result = benchmark.run_benchmark(groups=4, particles=4, steps=1, support=4,
                                     hidden_dim=8, embedding_dim=2, repetitions=1)
    assert not result["numerical_agreement"] and not result["finite_objectives"]
    assert not result["finite_gradients"]
    assert result["optimized_objective"] is result["absolute_objective_error"] is None
    assert result["maximum_absolute_gradient_error"] is None
    assert json.loads(json.dumps(result, allow_nan=False))["numerical_agreement"] is False


def test_invalid_coordinates_fail_closed_without_a_nominal_benchmark_receipt(monkeypatch):
    original = benchmark.tensor_rollout
    def damaged(*args, **kwargs):
        output = original(*args, **kwargs)
        return replace(output, z_steps=output.z_steps*float("nan"))
    monkeypatch.setattr(benchmark, "tensor_rollout", damaged)
    with pytest.raises(RuntimeError, match="diameter"):
        benchmark.run_benchmark(groups=4, particles=4, steps=1, support=4,
                                hidden_dim=8, embedding_dim=2, repetitions=1)


def test_global_numerical_flags_restored_on_failure(monkeypatch):
    threads = torch.get_num_threads()
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    def failed(**kwargs):
        raise RuntimeError("synthetic construction failure")
    monkeypatch.setattr(benchmark, "synthetic_data", failed)
    try:
        with pytest.raises(RuntimeError, match="construction"):
            benchmark.run_benchmark()
        assert torch.get_num_threads() == threads
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn


@pytest.mark.parametrize("agreement", [True, False])
def test_cli_exclusive_receipt_and_numerical_failure_exit(tmp_path, monkeypatch, capsys, agreement):
    output = tmp_path / "new.json"
    result = dict(numerical_agreement=agreement, optimizer_steps=0, real_data_reads=0)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda **kwargs: result)
    assert benchmark.main(["--output", str(output)]) == (0 if agreement else 1)
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == result
    with pytest.raises(FileExistsError):
        benchmark.main(["--output", str(output)])
    assert json.loads(output.read_text()) == result


def test_existing_output_and_symlink_rejected_before_measurement(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Measurement ran despite existing output")
    monkeypatch.setattr(benchmark, "run_benchmark", forbidden)
    retained = tmp_path / "retained.json"
    retained.write_text("retained")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(tmp_path / "missing.json")
    for path in (retained, symlink, tmp_path / "missing-directory/new.json"):
        with pytest.raises(FileExistsError):
            benchmark.main(["--output", str(path)])
    assert retained.read_text() == "retained"
