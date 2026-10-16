"""CPU-only worker checks: eight toy optimizer calls per complete file.

CUDA qualification itself is NOT executed. Snapshot labels for the scope guard
are fabricated and involve no optimizer.step calls.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from frontier.checkpoints import TrajectoryLease, save_checkpoint
from frontier.native_inputs import NativeBatch
from frontier.observation import ObservationFold
from frontier.observation_training import ObservationFitter, ObservationRows, ObservationTrainingSpec
from frontier.observation_worker import (PAIRED_FP32_TOLERANCES, SOURCE_MEMBERS, _atomic_json,
                                         _fp32_comparison, _input_exit_receipt, _input_lease_receipt,
                                         _runtime, load_contract, run_fitting_core,
                                         verify_qualification, verify_sources)
from frontier.retention import STUDY_ID


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))
    return digest(path)


def rows():
    fold = ObservationFold("D3-D4", ("D3", "D4"), "f" * 64)
    batches = []
    for start in range(0, 64, 4):
        matrix = sparse.csr_matrix(np.asarray([[3, 0, 2, 1, 0, 4, 1]] * 4, dtype=np.uint32))
        cells = pd.DataFrame({"source_id": ["D3_Stim8hr"] * 4, "donor_id": ["D3"] * 4,
                              "condition": ["Stim8hr"] * 4})
        batches.append(NativeBatch(matrix, cells, tuple(f"D3_Stim8hr:cell-{i}" for i in range(start, start + 4)),
                                   np.full(4, 11), np.full(4, True)))
    return ObservationRows.fabricated(batches, fold)


def case(tmp_path, name):
    study = tmp_path / STUDY_ID
    study.mkdir(exist_ok=True)
    source = rows()
    spec = ObservationTrainingSpec.fixture(checkpoint_interval=1)
    identity = {"trajectory_id": name, "fold_id": source.fold.fold_id, "seed": 0,
                "input_sha256": source.identity, "configuration_sha256": spec.identity, "source_sha256": "a" * 64}
    contract = {"study_root": str(study), "mode": "synthetic_cpu_test", "output_limit_bytes": 16 * 1024 ** 2}
    return study, source, spec, identity, contract


def test_complete_cpu_worker_lifecycle_retention_and_immutable_progress(tmp_path):
    # Four toy optimizer calls.
    study, source, spec, identity, contract = case(tmp_path, "complete")
    with TrajectoryLease(study / "complete", identity, "fixture-worker") as lease:
        final, receipt = run_fitting_core(lease, source, spec, contract, device="cpu", qualification={})
        assert final["accepted_updates"] == final["actual_disposable_optimizer_calls"] == 4
        assert final["scientific_updates"] == 0 and final["scope"] == "synthetic_cpu_only"
        assert final["resume_from_accepted_updates"] == 0 and final["timing_scope"] == "current_owner_only"
        assert final["input_lease"] is None and final["input_exit"] is None
        assert digest(receipt["path"]) == receipt["sha256"]
        progress = [json.loads(path.read_text()) for path in sorted((lease.root / "progress").iterdir())]
        assert [row["accepted_updates"] for row in progress] == [1, 2, 3, 4]
        assert all(row["training_and_input_seconds"] >= 0 for row in progress)
        assert all(row["capacity"]["fileset_quota_checked_by_worker"] is False for row in progress)
        assert not (lease.root / "checkpoints/update-000000001/checkpoint.pt").exists()
        assert (lease.root / "checkpoints/update-000000001/COMPLETE.json").exists()
        assert (lease.root / "checkpoints/update-000000002/checkpoint.pt").exists()  # audit milestone
        assert (lease.root / "SELECTED_OBSERVATION.pt").exists()
        with pytest.raises(RuntimeError, match="Closed"):
            run_fitting_core(lease, source, spec, contract, device="cpu", qualification={})
        before = (lease.root / "COMPLETE.json").read_bytes()
        with pytest.raises(FileExistsError, match="Immutable"):
            _atomic_json(lease.root, "COMPLETE.json", {"overwrite": True})
        assert (lease.root / "COMPLETE.json").read_bytes() == before


def test_explicit_cpu_resume_finishes_remaining_updates_only(tmp_path):
    # Two fixture preparation calls plus two resumed calls = four actual calls.
    study, source, spec, identity, contract = case(tmp_path, "resume")
    with TrajectoryLease(study / "resume", identity, "fixture-first") as lease:
        first = ObservationFitter(lease, source, spec, device="cpu")
        first.run_until(2)
        checkpoint = first.last_checkpoint
    with TrajectoryLease(study / "resume", identity, "fixture-second") as lease:
        final, _ = run_fitting_core(lease, source, spec, contract, device="cpu", qualification={},
                                    resume={"directory": checkpoint["directory"], "sha256": checkpoint["checkpoint_sha256"]})
        assert final["accepted_updates"] == 4 and final["resume_from_accepted_updates"] == 2
        assert sorted(path.name for path in (lease.root / "progress").iterdir()) == ["update-000000003.json", "update-000000004.json"]
        assert final["fitting_row_exposures"] == 32


def test_cuda_qualification_checkpoint_scope_rejects_cpu_without_optimizer_call(tmp_path):
    study, source, spec, identity, contract = case(tmp_path, "scope")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    with TrajectoryLease(study / "scope", identity, "fixture-scope") as lease:
        with pytest.raises(ValueError, match="actual CUDA"):
            save_checkpoint(lease, model=model, optimizer=optimizer, scheduler=None,
                accepted_updates=0, attempted_updates=0, phase="qualification", sampler_state={"position": 0},
                count_cache={}, execution_scope="synthetic_cuda_qualification")
        assert not list((lease.root / "checkpoints").iterdir())


def bound_contract(tmp_path):
    study = tmp_path / STUDY_ID
    study.mkdir(exist_ok=True)
    source_manifest = study / "source.json"
    source_sha = write(source_manifest, {"schema": "credo.frontier.observation-worker-source.v1",
        "files": {name: digest(ROOT / name) for name in SOURCE_MEMBERS}})
    protocol = ROOT / "OBSERVATION_PROTOCOL.json"
    contract = {"schema": "credo.frontier.observation-worker-contract.v1", "mode": "qualify",
        "study_root": str(study), "output_root": str(study / "qualification"), "worker_id": "test-owner",
        "trajectory_id": "test-qualification", "source_root": str(ROOT), "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_sha, "protocol_path": str(protocol), "protocol_sha256": digest(protocol),
        "runtime_image": "registry.example/approved@sha256:" + "b" * 64, "output_limit_bytes": 16 * 1024 ** 3}
    path = study / "contract.json"
    write(path, contract)
    return path, contract


def test_exact_source_protocol_contract_hash_and_unrecognized_fields(tmp_path):
    path, contract = bound_contract(tmp_path)
    loaded, spec = load_contract(path, digest(path), "qualify")
    assert loaded == contract and spec.maximum_updates == 20000
    assert set(verify_sources(contract)["files"]) == set(SOURCE_MEMBERS)
    with pytest.raises(ValueError, match="SHA-256"):
        load_contract(path, "a" * 64, "qualify")
    write(path, {**contract, "elapsed_time_limit": 5})
    with pytest.raises(ValueError, match="mode-specific"):
        load_contract(path, digest(path), "qualify")
    changed = json.loads(Path(contract["source_manifest"]).read_text())
    changed["files"]["frontier/observation.py"] = "a" * 64
    write(contract["source_manifest"], changed)
    contract["source_manifest_sha256"] = digest(contract["source_manifest"])
    with pytest.raises(ValueError, match="SHA-256"):
        verify_sources(contract)


def test_same_source_qualification_receipt_not_a_boolean_or_small_cpu_fixture(tmp_path):
    path, contract = bound_contract(tmp_path)
    receipt_path = path.parent / "fabricated_receipt.json"
    # This only tests receipt comparison; it is not CUDA evidence or a production launch.
    hardware = {"device": "cuda:0", "name": "NVIDIA H100 80GB HBM3", "total_memory_bytes": 80 * 1024 ** 3,
        "capability": [9, 0], "visible_devices": 1, "precision": "FP32", "tf32": False,
        "deterministic_algorithms": True, "required_headroom_bytes": 8 * 1024 ** 3}
    record = {"schema": "credo.frontier.observation-cuda-qualification.v1", "status": "passed",
        "source_manifest_sha256": contract["source_manifest_sha256"], "protocol_sha256": contract["protocol_sha256"],
        "runtime_image": contract["runtime_image"], "runtime": _runtime(), "hardware": hardware,
        "specification_sha256": ObservationTrainingSpec().identity, "scientific_updates": 0,
        "actual_disposable_optimizer_calls": 13, "real_count_payloads_opened": False, "real_input_mount_qualified": False,
        "restart_equivalence": "exact_model_optimizer_sampler_rng", "shape": [2048, 18129], "nonzeros_per_cell": 4096,
        "resources_all_checks": {"peak_reserved_bytes": 2 * 1024 ** 3,
                                  "physical_headroom_at_peak_reserved_bytes": 78 * 1024 ** 3},
        "paired_fp32_check": {"schema": "credo.frontier.observation-paired-fp32.v1", "status": "passed",
            "shape": [16, 18129], "architecture": [18129, 512, 128, 48], "tolerances": dict(PAIRED_FP32_TOLERANCES),
            "optimizer_calls": 0, "all_parameter_gradients_checked": True, "same_initial_weights": True,
            "cpu_encoding": "sparse", "cuda_encoding": "bounded_dense",
            "loss": {"passed": True, "maximum_tolerance_normalized_error": 0.1},
            "gradient_summary": {"passed": True, "maximum_tolerance_normalized_error": 0.2}}}
    contract["qualification_receipt"] = str(receipt_path)
    contract["qualification_sha256"] = write(receipt_path, record)
    assert verify_qualification(contract, hardware)["status"] == "passed"
    for changed in ({**record, "source_manifest_sha256": "e" * 64}, {**record, "shape": [8, 7]},
                    {**record, "status": "failed"}, {**record, "actual_disposable_optimizer_calls": 17}):
        contract["qualification_sha256"] = write(receipt_path, changed)
        with pytest.raises(ValueError, match="same-source"):
            verify_qualification(contract, hardware)
    for changed_pair in ({**record["paired_fp32_check"], "tolerances": {**PAIRED_FP32_TOLERANCES, "gradient_rtol": 0.01}},
                         {**record["paired_fp32_check"], "gradient_summary": {"passed": True, "maximum_tolerance_normalized_error": 1.01}},
                         {**record["paired_fp32_check"], "shape": [2, 7]}):
        contract["qualification_sha256"] = write(receipt_path, {**record, "paired_fp32_check": changed_pair})
        with pytest.raises(ValueError, match="same-source"):
            verify_qualification(contract, hardware)


def test_worker_capacity_failure_is_durable_and_prevents_retry_without_steps(tmp_path):
    study, source, spec, identity, contract = case(tmp_path, "capacity")
    contract["output_limit_bytes"] = 1
    with TrajectoryLease(study / "capacity", identity, "fixture-capacity") as lease:
        with pytest.raises(RuntimeError, match="capacity"):
            run_fitting_core(lease, source, spec, contract, device="cpu", qualification={})
        failed = json.loads((lease.root / "WORKER_FAILED.json").read_text())
        assert failed["counters"]["optimizer_calls"] == 0
        with pytest.raises(RuntimeError, match="Closed"):
            run_fitting_core(lease, source, spec, contract, device="cpu", qualification={})


def test_fixed_fp32_comparison_tiny_arrays_without_forward_or_optimizer():
    reference = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    good = _fp32_comparison(reference, reference + 1e-5, atol=2e-5, rtol=2e-5)
    assert good["passed"] and 0 < good["maximum_tolerance_normalized_error"] <= 1
    bad = _fp32_comparison(reference, reference + 1e-3, atol=2e-5, rtol=2e-5)
    assert not bad["passed"] and bad["maximum_tolerance_normalized_error"] > 1
    with pytest.raises(ValueError, match="Nonfinite"):
        _fp32_comparison(reference, torch.tensor([0.0, float("nan"), 1.0]), atol=2e-5, rtol=2e-5)


def test_production_core_rejects_bare_qualification_flag_before_cuda_or_steps(tmp_path):
    study, source, _, identity, contract = case(tmp_path, "bare-flag")
    with TrajectoryLease(study / "bare-flag", identity, "fixture-guard") as lease:
        with pytest.raises(ValueError, match="not a passed flag"):
            run_fitting_core(lease, source, SimpleNamespace(synthetic_fixture=False), contract,
                             device="cuda:0", qualification={"status": "passed"})


def test_input_lease_receipts_are_immutable_and_exit_checks_exact_snapshot(tmp_path):
    # Metadata-receipt protocol fixture only; no actual NativeHandle/payload use.
    study, _, _, identity, _ = case(tmp_path, "input-receipts")
    snapshot = {"schema": "credo.frontier.native-input-lease.v1", "access_sha256": "a" * 64,
                "dataset_root": "/fabricated/never-opened", "metadata_index_sha256": "b" * 64,
                "source_ids": ["D3_Stim8hr"], "payload_hashes_verified_on_consumption": True,
                "all_payloads_consumed_claimed": False,
                "members": {"fixture": {"sha256": "c" * 64, "file_identity": {"st_ino": 12, "st_size": 9}}}}
    seen = []

    def verify(value):
        seen.append(value)
        assert value is snapshot
        return {"schema": "credo.frontier.native-input-lease-exit.v1", "status": "unchanged"}

    source = SimpleNamespace(synthetic_fixture=False, identity="a" * 64,
                             handle=SimpleNamespace(verify_publication_identity=verify))
    with TrajectoryLease(study / "input-receipts", identity, "receipt-test") as lease:
        entered = _input_lease_receipt(lease, source, snapshot)
        assert _input_lease_receipt(lease, source, snapshot) == entered
        changed = {**snapshot, "members": {}}
        with pytest.raises(ValueError, match="SHA-256|Bounded regular"):
            _input_lease_receipt(lease, source, changed)
        exit_receipt = _input_exit_receipt(lease, source, snapshot)
        assert seen == [snapshot] and digest(exit_receipt["path"]) == exit_receipt["sha256"]
        assert not (lease.root / "COMPLETE.json").exists()
        source.handle.verify_publication_identity = lambda value: (_ for _ in ()).throw(ValueError("input changed"))
        with pytest.raises(ValueError, match="input changed"):
            _input_exit_receipt(lease, source, snapshot)
        with pytest.raises(ValueError, match="Fabricated"):
            _input_lease_receipt(lease, SimpleNamespace(synthetic_fixture=True), snapshot)
    with TrajectoryLease(study / "input-receipts", identity, "explicit-resume-test") as lease:
        fresh = {**snapshot, "members": {"fixture": {"sha256": "c" * 64,
                                                     "file_identity": {"st_ino": 144, "st_dev": 27, "st_size": 9}}}}
        entered_again = _input_lease_receipt(lease, source, fresh, resume=True)
        assert "/input-leases/" in entered_again["path"]
        assert digest(entered["path"]) == entered["sha256"]
        changed_sha = {**fresh, "members": {"fixture": {**fresh["members"]["fixture"], "sha256": "d" * 64}}}
        with pytest.raises(ValueError, match="hashes/sizes/roles"):
            _input_lease_receipt(lease, source, changed_sha, resume=True)
