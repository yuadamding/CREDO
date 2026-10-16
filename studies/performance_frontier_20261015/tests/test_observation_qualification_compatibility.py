"""Prospective fabricated receipt regressions; zero optimizer/GPU/data calls.

No existing optimization fixture is imported or executed. The byte totals in
the principal case reproduce the observed admission mismatch, but these local
receipts are deliberately fabricated tests, not new runtime qualification.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from frontier import observation_worker as worker


GIB = 1024 ** 3
QUALIFIED_BYTES = 85028372480
ACTUAL_BYTES = 84929478656
MEASURED_PEAK = 1927282688


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def bound(tmp_path):
    hardware = {"device": "cuda:0", "name": "NVIDIA H100 80GB HBM3", "total_memory_bytes": QUALIFIED_BYTES,
                "capability": [9, 0], "visible_devices": 1, "precision": "FP32", "tf32": False,
                "deterministic_algorithms": True, "required_headroom_bytes": 8 * GIB}
    contract = {"mode": "fit", "qualification_receipt": str(tmp_path / "fabricated_qualification.json"),
                "source_manifest_sha256": "a" * 64, "protocol_sha256": "b" * 64,
                "runtime_image": "fabricated/never-launched@sha256:" + "c" * 64,
                "study_root": str(tmp_path), "output_limit_bytes": 4 * GIB}
    record = {"schema": "credo.frontier.observation-cuda-qualification.v1", "status": "passed",
        "source_manifest_sha256": contract["source_manifest_sha256"], "protocol_sha256": contract["protocol_sha256"],
        "runtime_image": contract["runtime_image"], "runtime": worker._runtime(), "hardware": hardware,
        "specification_sha256": worker.ObservationTrainingSpec().identity, "scientific_updates": 0,
        "actual_disposable_optimizer_calls": 13, "real_count_payloads_opened": False, "real_input_mount_qualified": False,
        "restart_equivalence": "exact_model_optimizer_sampler_rng", "shape": [2048, 18129], "nonzeros_per_cell": 4096,
        "resources_all_checks": {"peak_reserved_bytes": MEASURED_PEAK,
                                  "physical_headroom_at_peak_reserved_bytes": QUALIFIED_BYTES - MEASURED_PEAK},
        "paired_fp32_check": {"schema": "credo.frontier.observation-paired-fp32.v1", "status": "passed",
            "shape": [16, 18129], "architecture": [18129, 512, 128, 48], "tolerances": dict(worker.PAIRED_FP32_TOLERANCES),
            "optimizer_calls": 0, "all_parameter_gradients_checked": True, "same_initial_weights": True,
            "cpu_encoding": "sparse", "cuda_encoding": "bounded_dense",
            "loss": {"passed": True, "maximum_tolerance_normalized_error": 0.1},
            "gradient_summary": {"passed": True, "maximum_tolerance_normalized_error": 0.2}}}
    contract["qualification_sha256"] = write(Path(contract["qualification_receipt"]), record)
    actual = dict(hardware, total_memory_bytes=ACTUAL_BYTES)
    return contract, record, actual


def replace_receipt(contract, record):
    contract["qualification_sha256"] = write(Path(contract["qualification_receipt"]), record)


def test_observed_same_h100_byte_difference_is_compatible_and_preserved(bound):
    contract, record, actual = bound
    with patch.object(worker.torch.cuda, "get_device_properties", side_effect=AssertionError("No GPU call allowed")):
        verified = worker.verify_qualification(contract, actual)
        report = worker._qualification_hardware_compatibility(verified, actual)
    assert verified == record  # Immutable qualification observation is not rewritten.
    assert report["qualified_hardware"]["total_memory_bytes"] == QUALIFIED_BYTES
    assert report["actual_hardware"]["total_memory_bytes"] == ACTUAL_BYTES
    assert report["qualification_peak_reserved_bytes"] == MEASURED_PEAK
    assert report["actual_minus_qualified_total_memory_bytes"] == -98893824
    assert report["actual_headroom_over_qualification_peak_bytes"] == ACTUAL_BYTES - MEASURED_PEAK
    assert report["actual_worker_peak_qualified_by_this_comparison"] is False
    assert record["hardware"]["total_memory_bytes"] == QUALIFIED_BYTES


@pytest.mark.parametrize("field,value", [
    ("name", "NVIDIA H100 PCIe"), ("capability", [8, 0]), ("precision", "BF16"),
    ("tf32", True), ("deterministic_algorithms", False), ("visible_devices", 2),
    ("device", "cuda:1"), ("required_headroom_bytes", 7 * GIB),
    ("total_memory_bytes", 74 * GIB), ("total_memory_bytes", 83 * GIB),
])
def test_other_hardware_mismatches_still_fail_with_key_name(bound, field, value):
    contract, _, actual = bound
    actual[field] = value
    with pytest.raises(ValueError, match="hardware\\." + field):
        worker.verify_qualification(contract, actual)


@pytest.mark.parametrize("field", ["name", "total_memory_bytes", "required_headroom_bytes"])
def test_missing_hardware_fields_are_not_silently_defaulted(bound, field):
    contract, _, actual = bound
    actual.pop(field)
    with pytest.raises(ValueError, match="hardware\\." + field):
        worker.verify_qualification(contract, actual)


@pytest.mark.parametrize("total", [75 * GIB, 82 * GIB])
def test_full_class_endpoints_with_peak_headroom_are_allowed(bound, total):
    contract, _, actual = bound
    actual["total_memory_bytes"] = total
    assert worker.verify_qualification(contract, actual)["status"] == "passed"


def test_measured_peak_plus_exact_eight_gib_boundary_and_one_byte_failure(bound):
    contract, record, actual = bound
    actual["total_memory_bytes"] = 75 * GIB
    record["resources_all_checks"].update(peak_reserved_bytes=67 * GIB,
        physical_headroom_at_peak_reserved_bytes=QUALIFIED_BYTES - 67 * GIB)
    replace_receipt(contract, record)
    assert worker.verify_qualification(contract, actual)["status"] == "passed"
    record["resources_all_checks"].update(peak_reserved_bytes=67 * GIB + 1,
        physical_headroom_at_peak_reserved_bytes=QUALIFIED_BYTES - 67 * GIB - 1)
    replace_receipt(contract, record)
    with pytest.raises(ValueError, match="hardware.total_memory_bytes:peak_plus_headroom"):
        worker.verify_qualification(contract, actual)


@pytest.mark.parametrize("change", ["missing_peak", "zero_peak", "float_peak", "inconsistent_headroom", "qualification_too_small"])
def test_invalid_qualification_memory_evidence_fails(bound, change):
    contract, record, actual = bound
    if change == "missing_peak":
        record.pop("resources_all_checks")
    elif change == "zero_peak":
        record["resources_all_checks"]["peak_reserved_bytes"] = 0
    elif change == "float_peak":
        record["resources_all_checks"]["peak_reserved_bytes"] = float(MEASURED_PEAK)
    elif change == "inconsistent_headroom":
        record["resources_all_checks"]["physical_headroom_at_peak_reserved_bytes"] -= 1
    else:
        record["hardware"]["total_memory_bytes"] = 74 * GIB
    replace_receipt(contract, record)
    with pytest.raises(ValueError, match="(resources_all_checks|qualification.hardware.total_memory_bytes)"):
        worker.verify_qualification(contract, actual)


def test_source_change_still_requires_new_qualification_and_runtime_remains_exact(bound):
    contract, record, actual = bound
    contract["source_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="source_manifest_sha256"):
        worker.verify_qualification(contract, actual)
    contract["source_manifest_sha256"] = record["source_manifest_sha256"]
    record["runtime"]["torch"] = "different-fixture-runtime"
    replace_receipt(contract, record)
    with pytest.raises(ValueError, match="runtime.torch"):
        worker.verify_qualification(contract, actual)
    record["runtime"] = worker._runtime()
    record["runtime"]["unexpected"] = None
    replace_receipt(contract, record)
    with pytest.raises(ValueError, match="runtime.unexpected"):
        worker.verify_qualification(contract, actual)


def test_paired_tolerances_and_gradient_checks_remain_exact(bound):
    contract, record, actual = bound
    original = deepcopy(record["paired_fp32_check"])
    for field, value in (("tolerances", {**worker.PAIRED_FP32_TOLERANCES, "gradient_rtol": .01}),
                         ("all_parameter_gradients_checked", False),
                         ("gradient_summary", {"passed": True, "maximum_tolerance_normalized_error": 1.01})):
        record["paired_fp32_check"] = {**original, field: value}
        replace_receipt(contract, record)
        with pytest.raises(ValueError, match="paired_fp32_check\\." + field):
            worker.verify_qualification(contract, actual)


def test_new_fit_receipt_records_actual_memory_before_any_fitter_or_optimizer(bound, tmp_path):
    contract, qualification, actual = bound
    output = tmp_path / "fabricated_fit_output"
    output.mkdir()
    lease = SimpleNamespace(root=output, identity={"fixture_only": True}, assert_active=lambda: None)
    spec = worker.ObservationTrainingSpec()
    stopped = RuntimeError("fabricated stop before fitter construction or optimizer")
    with patch.object(worker, "_input_lease_receipt", return_value=None), \
            patch.object(worker, "RetentionPolicy", return_value=object()), \
            patch.object(worker, "bind_policy"), \
            patch.object(worker, "ObservationFitter", side_effect=stopped) as fitter:
        with pytest.raises(RuntimeError, match="fabricated stop"):
            worker.run_fitting_core(lease, object(), spec, contract, device="cuda:0",
                                    qualification=qualification, hardware=actual)
    assert fitter.call_count == 1  # Stub raises; no fitter/optimizer is constructed.
    report = json.loads((output / "QUALIFICATION_COMPATIBILITY.json").read_text())
    started = json.loads((output / "WORKER_STARTED.json").read_text())
    assert report["qualified_hardware"]["total_memory_bytes"] == QUALIFIED_BYTES
    assert report["actual_hardware"]["total_memory_bytes"] == ACTUAL_BYTES
    assert report["qualification_receipt_sha256"] == contract["qualification_sha256"]
    assert report["source_manifest_sha256"] == contract["source_manifest_sha256"]
    assert started["qualification_compatibility"] == report
    failed = json.loads((output / "WORKER_FAILED.json").read_text())
    assert failed["counters"] is None


def test_production_core_cannot_substitute_qualification_hardware_for_actual_worker(bound, tmp_path):
    contract, qualification, _ = bound
    lease = SimpleNamespace(root=tmp_path, assert_active=lambda: None)
    with patch.object(worker, "ObservationFitter", side_effect=AssertionError("No fitter allowed")):
        with pytest.raises(ValueError, match="Actual configured worker hardware"):
            worker.run_fitting_core(lease, object(), worker.ObservationTrainingSpec(), contract,
                                    device="cuda:0", qualification=qualification)
