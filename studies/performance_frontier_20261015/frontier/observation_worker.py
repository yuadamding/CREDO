"""Executable independent observation phase: qualify, fit, or explicit resume.

Example: ``python -m frontier.observation_worker qualify --contract FILE
--contract-sha256 SHA``. Production uses ``fit`` with its separately bound
contract; optional --resume-directory/--resume-sha256 request a specific healthy
checkpoint, never an automatic retry. No scheduler, network or deadline logic
is present. Qualification fabricates full-shape sparse data and never opens the
real dataset. Its evidence is not a real-input/mount qualification.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import re
import resource
import stat
import sys
import time
import uuid

import numpy as np
import scipy
from scipy import sparse
import torch

from .checkpoints import TrajectoryLease, load_checkpoint, save_checkpoint
from .native_inputs import FoldInputs, NativeInputIndex
from .observation import ObservationFold, SparseR48
from .observation_training import ObservationFitter, ObservationRows, ObservationTrainingSpec
from .retention import (STUDY_ID, RetentionPolicy, bind_policy, commit_retirement,
                        identity_sha256, prepare_retirement)


SOURCE_MEMBERS = tuple("frontier/" + name for name in (
    "__init__.py", "checkpoints.py", "native_inputs.py", "observation.py",
    "observation_training.py", "retention.py", "observation_worker.py"))
VRAM_HEADROOM_BYTES = 8 * 1024 ** 3
QUALIFICATION_OPTIMIZER_CALLS = 13
PAIRED_FP32_TOLERANCES = {"loss_atol": 2e-5, "loss_rtol": 2e-5,
                         "gradient_atol": 2e-5, "gradient_rtol": 2e-4}


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Exact lowercase SHA-256 required")
    return value


def _path(value, *, exists=True):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Explicit absolute nonsymlink path required")
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path


def _read(path, expected_sha, *, maximum_bytes=32 * 1024 ** 2):
    path = _path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
        raise ValueError("Bounded regular receipt/source file required")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Receipt/source changed while opening")
        payload = stream.read(maximum_bytes + 1)
        after = path.lstat()
        if len(payload) > maximum_bytes or any(getattr(info, key) != getattr(after, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("Receipt/source changed during reading")
    if hashlib.sha256(payload).hexdigest() != _sha(expected_sha):
        raise ValueError("Receipt/source SHA-256 mismatch")
    return payload


def _atomic_json(root, name, record):
    root = _path(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?", name):
        raise ValueError("Explicit bounded receipt name required")
    path = _path(root / name, exists=False)
    if path.exists():
        raise FileExistsError("Immutable worker receipt already exists")
    path.parent.mkdir(exist_ok=True)
    payload = json.dumps(record, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"
    temporary = path.parent / f".publishing-{uuid.uuid4().hex}"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)  # Atomic no-clobber publication.
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(payload).hexdigest()
    if _read(path, digest, maximum_bytes=len(payload)) != payload:
        raise ValueError("Immutable worker receipt failed readback")
    return {"path": str(path), "sha256": digest, "size_bytes": len(payload)}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _runtime():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "numpy": str(np.__version__), "scipy": str(scipy.__version__)}


def _configure_cuda():
    if torch.cuda.is_initialized():
        raise RuntimeError("Worker CUDA configuration must precede context initialization")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
        raise ValueError("Bound deterministic CuBLAS workspace configuration changed")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible H100 is required before data work")
    device = torch.cuda.get_device_properties(0)
    if "H100" not in device.name or not 75 * 1024 ** 3 <= device.total_memory <= 82 * 1024 ** 3:
        raise RuntimeError("One full 80-GB-class H100, not a MIG slice or another GPU, is required")
    free, total = torch.cuda.mem_get_info(0)
    if free < total - 2 * 1024 ** 3:
        raise RuntimeError("H100 is not sufficiently idle for bound single-worker admission")
    return {"device": "cuda:0", "name": device.name, "total_memory_bytes": device.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)), "visible_devices": 1,
            "precision": "FP32", "tf32": False, "deterministic_algorithms": True,
            "required_headroom_bytes": VRAM_HEADROOM_BYTES}


def _resources(device):
    result = {"maximum_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.device(device).type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        peak = torch.cuda.max_memory_reserved(0)
        result.update(allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=peak,
                      physical_headroom_at_peak_reserved_bytes=total - peak)
        if total - peak < VRAM_HEADROOM_BYTES:
            raise RuntimeError("Measured H100 reservation violates the required 8-GiB physical headroom")
    if result["maximum_rss_bytes"] > 200 * 1024 ** 3:
        raise RuntimeError("Worker exceeds the bound 200-GiB host-memory allocation")
    return result


def _output_bytes(root):
    total = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in (*folders, *files):
            path = _path(Path(directory) / name)
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("Unexpected output-tree member type")
    return total


def _capacity(root, limit, reserve=0):
    used = _output_bytes(root)
    available = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    if used + reserve > limit or available < reserve:
        raise RuntimeError("Bound output allowance or filesystem capacity is insufficient")
    return {"output_bytes": used, "output_limit_bytes": limit, "reserved_next_write_bytes": reserve,
            "filesystem_available_bytes": available, "fileset_quota_checked_by_worker": False}


def verify_sources(contract):
    root = _path(contract["source_root"])
    payload = _read(contract["source_manifest"], contract["source_manifest_sha256"])
    manifest = json.loads(payload)
    if manifest.get("schema") != "credo.frontier.observation-worker-source.v1" or set(manifest.get("files", {})) != set(SOURCE_MEMBERS):
        raise ValueError("Exact observation worker source inventory required")
    for name, digest in manifest["files"].items():
        _read(root / name, digest)
        if name == "frontier/observation_worker.py":
            loaded = Path(__file__).absolute()
        else:
            module_name = "frontier" if name.endswith("__init__.py") else name[:-3].replace("/", ".")
            loaded = Path(importlib.import_module(module_name).__file__).absolute()
        if loaded != root / name:
            raise ValueError("Imported observation code is not the bound source tree")
    return manifest


def load_contract(path, digest, mode):
    contract = json.loads(_read(path, digest))
    common = {"schema", "mode", "study_root", "output_root", "worker_id", "trajectory_id", "source_root",
              "source_manifest", "source_manifest_sha256", "protocol_path", "protocol_sha256",
              "runtime_image", "output_limit_bytes"}
    fit = {"metadata_root", "reader_source_root", "dataset_root", "fold_id", "training_donors",
           "metadata_index_sha256", "rna_axis_sha256", "input_access_sha256", "qualification_receipt", "qualification_sha256"}
    if (set(contract) != common | (fit if mode == "fit" else set())
            or contract["schema"] != "credo.frontier.observation-worker-contract.v1" or contract["mode"] != mode):
        raise ValueError("Exact mode-specific observation worker contract required")
    study = _path(contract["study_root"])
    output = _path(contract["output_root"], exists=False)
    if study.name != STUDY_ID or output == study or not output.is_relative_to(study) or not output.parent.is_dir():
        raise ValueError("Exact prepared new-study output parent required")
    for name in ("worker_id", "trajectory_id"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", contract[name]):
            raise ValueError("Safe worker and trajectory identities required")
    if (type(contract["output_limit_bytes"]) is not int or contract["output_limit_bytes"] < 1024 ** 3
            or (mode == "qualify" and contract["output_limit_bytes"] > 16 * 1024 ** 3)):
        raise ValueError("Positive bound output allowance required; qualification maximum is 16 GiB")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", contract["runtime_image"]):
        raise ValueError("Pinned runtime image identity required")
    verify_sources(contract)
    protocol = json.loads(_read(contract["protocol_path"], contract["protocol_sha256"]))
    spec = ObservationTrainingSpec()
    checks = {"schema": "credo.frontier.observation_protocol.v1", "maximum_accepted_updates_each": 20000,
              "logical_batch_positive_training_cells": 2048, "RNA_feature_count": 18129,
              "architecture_dimensions": [18129, 512, 128, 48], "audit_candidates_accepted_updates": list(spec.audit_updates),
              "checkpoint_interval_accepted_updates": 256, "maximum_dense_working_estimate_bytes": 4 * 1024 ** 3,
              "historical_checkpoint_initialization": False, "early_stopping": False, "automatic_retries": 0,
              "elapsed_time_limit": None, "initialization_seed": 0, "maximum_fits": 10,
              "parameters": 18729473, "fitting_conditions": ["Stim8hr", "Stim48hr"],
              "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                            "betas": [0.9, 0.999], "epsilon": 1e-8, "gradient_clip": 1.0}}
    if any(protocol.get(name) != expected for name, expected in checks.items()):
        raise ValueError("Worker contract and exact observation protocol disagree")
    if any(protocol.get("technical_holdout", {}).get(key) != value for key, value in {
            "seed": 0, "expected_fraction": 0.05, "exclude_from_optimizer": True,
            "exclude_from_center_scale": True, "biological_independence_claimed": False}.items()):
        raise ValueError("Technical holdout protocol changed")
    if mode == "fit":
        fold = FoldInputs(contract["fold_id"], tuple(contract["training_donors"]))
        if list(fold.training_donors) not in protocol["training_subsets"]:
            raise ValueError("Fold is not one of the ten registered training subsets")
        for name in ("metadata_index_sha256", "rna_axis_sha256", "input_access_sha256", "qualification_sha256"):
            _sha(contract[name])
        if (contract["metadata_index_sha256"] != protocol["metadata_index_identity"]
                or contract["rna_axis_sha256"] != protocol["RNA_feature_axis_identity"]):
            raise ValueError("Input metadata axis differs from protocol")
    return contract, spec


def _qualification_hardware_compatibility(receipt, hardware):
    """Same H100/runtime class, with measured—not byte-identical—VRAM admission.

    Full H100s of the same name/capability can expose slightly different usable
    byte totals. Retain both observations and require the qualification's peak
    reservation plus the unchanged 8-GiB margin to fit the actual worker.
    This does not relax current-worker resource monitoring or source identity.
    """
    qualified = receipt.get("hardware")
    fixed = {"device": "cuda:0", "visible_devices": 1, "precision": "FP32", "tf32": False,
             "deterministic_algorithms": True, "required_headroom_bytes": VRAM_HEADROOM_BYTES}
    keys = set(fixed) | {"name", "capability", "total_memory_bytes"}
    mismatches = []
    for label, record in (("qualification.hardware", qualified), ("hardware", hardware)):
        if not isinstance(record, dict):
            mismatches.append(label)
            continue
        mismatches.extend(label + "." + key for key in sorted(keys ^ set(record)))
        mismatches.extend(label + "." + key for key, expected in fixed.items() if record.get(key) != expected)
        if not isinstance(record.get("name"), str) or "H100" not in record["name"]:
            mismatches.append(label + ".name")
        if record.get("capability") != [9, 0]:
            mismatches.append(label + ".capability")
        total = record.get("total_memory_bytes")
        if type(total) is not int or not 75 * 1024 ** 3 <= total <= 82 * 1024 ** 3:
            mismatches.append(label + ".total_memory_bytes")
    if isinstance(qualified, dict) and isinstance(hardware, dict):
        mismatches.extend("hardware." + key for key in sorted(keys - {"total_memory_bytes"})
                          if qualified.get(key) != hardware.get(key))
    if mismatches:
        raise ValueError("Mismatched same-source H100 qualification keys: " + ", ".join(sorted(set(mismatches))))
    measured = receipt.get("resources_all_checks", {})
    peak = measured.get("peak_reserved_bytes") if isinstance(measured, dict) else None
    if type(peak) is not int or peak < 1:
        mismatches.append("resources_all_checks.peak_reserved_bytes")
    else:
        if measured.get("physical_headroom_at_peak_reserved_bytes") != qualified["total_memory_bytes"] - peak:
            mismatches.append("resources_all_checks.physical_headroom_at_peak_reserved_bytes")
        if peak + VRAM_HEADROOM_BYTES > qualified["total_memory_bytes"]:
            mismatches.append("qualification.hardware.total_memory_bytes:peak_plus_headroom")
        if peak + VRAM_HEADROOM_BYTES > hardware["total_memory_bytes"]:
            mismatches.append("hardware.total_memory_bytes:peak_plus_headroom")
    if mismatches:
        raise ValueError("Mismatched same-source H100 qualification keys: " + ", ".join(mismatches))
    return {"schema": "credo.frontier.observation-qualification-compatibility.v1", "status": "compatible",
            "policy": "same_h100_class_measured_peak_headroom_v1", "qualified_hardware": deepcopy(qualified),
            "actual_hardware": deepcopy(hardware), "qualification_peak_reserved_bytes": peak,
            "required_headroom_bytes": VRAM_HEADROOM_BYTES,
            "actual_headroom_over_qualification_peak_bytes": hardware["total_memory_bytes"] - peak,
            "actual_minus_qualified_total_memory_bytes": hardware["total_memory_bytes"] - qualified["total_memory_bytes"],
            "actual_worker_peak_qualified_by_this_comparison": False}


def verify_qualification(contract, hardware):
    receipt = json.loads(_read(contract["qualification_receipt"], contract["qualification_sha256"]))
    required = {"schema": "credo.frontier.observation-cuda-qualification.v1", "status": "passed",
                "source_manifest_sha256": contract["source_manifest_sha256"], "protocol_sha256": contract["protocol_sha256"],
                "runtime_image": contract["runtime_image"], "runtime": _runtime(),
                "specification_sha256": ObservationTrainingSpec().identity, "scientific_updates": 0,
                "actual_disposable_optimizer_calls": QUALIFICATION_OPTIMIZER_CALLS,
                "real_count_payloads_opened": False, "real_input_mount_qualified": False,
                "restart_equivalence": "exact_model_optimizer_sampler_rng", "shape": [2048, 18129],
                "nonzeros_per_cell": 4096}
    mismatches = [name for name, expected in required.items() if receipt.get(name) != expected]
    if "runtime" in mismatches and isinstance(receipt.get("runtime"), dict):
        mismatches.remove("runtime")
        mismatches.extend("runtime." + key for key in sorted(set(receipt["runtime"]) | set(required["runtime"]))
                          if key not in receipt["runtime"] or key not in required["runtime"]
                          or receipt["runtime"][key] != required["runtime"][key])
    if mismatches:
        raise ValueError("Missing, failed or mismatched same-source H100 qualification keys: " + ", ".join(mismatches))
    _qualification_hardware_compatibility(receipt, hardware)
    paired = receipt.get("paired_fp32_check", {})
    paired_required = {"schema": "credo.frontier.observation-paired-fp32.v1", "status": "passed",
                       "shape": [16, 18129], "architecture": [18129, 512, 128, 48],
                       "tolerances": PAIRED_FP32_TOLERANCES, "optimizer_calls": 0,
                       "all_parameter_gradients_checked": True, "same_initial_weights": True,
                       "cpu_encoding": "sparse", "cuda_encoding": "bounded_dense"}
    mismatches = ["paired_fp32_check." + name for name, expected in paired_required.items() if paired.get(name) != expected]
    if mismatches:
        raise ValueError("Missing or mismatched same-source paired FP32 qualification keys: " + ", ".join(mismatches))
    for name in ("loss", "gradient_summary"):
        measured = paired.get(name, {})
        normalized = measured.get("maximum_tolerance_normalized_error")
        if (measured.get("passed") is not True or not isinstance(normalized, (int, float))
                or not np.isfinite(normalized) or not 0 <= normalized <= 1):
            raise ValueError("Failed same-source paired FP32 tolerance comparison keys: paired_fp32_check." + name)
    return receipt


def _lease_identity(contract, spec, input_sha, fold_id, contract_sha):
    return {"trajectory_id": contract["trajectory_id"], "fold_id": fold_id, "seed": 0,
            "input_sha256": input_sha, "configuration_sha256": spec.identity,
            "source_sha256": contract["source_manifest_sha256"], "worker_contract_sha256": contract_sha}


def _fail(lease, contract, error, *, counters=None):
    record = {"schema": "credo.frontier.observation-worker-failure.v1", "at_utc": _utc(),
              "identity": lease.identity, "mode": contract["mode"], "exception_type": type(error).__name__,
              "exception_message": str(error)[:4096],
              "automatic_retry": False, "counters": counters}
    try:
        _atomic_json(lease.root, "WORKER_FAILED.json", record)
    except BaseException as publication_error:
        error.add_note(f"Worker failure receipt unavailable: {type(publication_error).__name__}")


def _input_binding(snapshot):
    """Immutable input bindings, excluding worker-local GPFS inode/device IDs."""
    return {**{key: snapshot[key] for key in ("schema", "dataset_root", "access_sha256",
             "metadata_index_sha256", "source_ids", "payload_hashes_verified_on_consumption", "all_payloads_consumed_claimed")},
            "members": {name: {"sha256": item["sha256"], "size_bytes": item["file_identity"]["st_size"]}
                        for name, item in snapshot["members"].items()}}


def _input_lease_receipt(lease, source, snapshot, *, resume=False):
    """Preserve the entered lease on explicit resume; never overwrite its evidence."""
    if source.synthetic_fixture:
        if snapshot is not None:
            raise ValueError("Fabricated fixture cannot claim a real native-input lease")
        return None
    if (not isinstance(snapshot, dict) or snapshot.get("schema") != "credo.frontier.native-input-lease.v1"
            or snapshot.get("access_sha256") != source.identity):
        raise ValueError("Exact entered native input lease required before fitting")
    name = "INPUT_LEASE.json"
    path = lease.root / name
    if not path.exists():
        return _atomic_json(lease.root, name, snapshot)
    if resume:
        # This is a prior owned metadata receipt, not an untrusted model pickle.
        # Its immutable projection must equal the freshly source-bound snapshot;
        # worker-local stat IDs are checked only on their respective workers.
        _path(path)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 32 * 1024 ** 2:
                raise ValueError("Bounded prior input lease receipt required")
            payload = stream.read(32 * 1024 ** 2 + 1)
        previous = json.loads(_read(path, hashlib.sha256(payload).hexdigest()))
        if _input_binding(previous) != _input_binding(snapshot):
            raise ValueError("Explicit resume input hashes/sizes/roles differ from the initial input lease")
        return _atomic_json(lease.root, f"input-leases/{lease.claim_id}.json", snapshot)
    payload = json.dumps(snapshot, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    _read(path, digest, maximum_bytes=len(payload))
    return {"path": str(path), "sha256": digest, "size_bytes": len(payload)}


def _input_exit_receipt(lease, source, snapshot, *, resume=False):
    if source.synthetic_fixture:
        if snapshot is not None:
            raise ValueError("Fabricated fixture cannot claim a real native-input exit check")
        return None
    result = source.handle.verify_publication_identity(snapshot)
    name = f"input-exits/{lease.claim_id}.json" if resume else "INPUT_EXIT.json"
    return _atomic_json(lease.root, name, result)


def run_fitting_core(lease, source, spec, contract, *, device, qualification, resume=None, input_snapshot=None, hardware=None,
                     continuation=None):
    """Shared production/test lifecycle; fabricated CPU scope is visibly distinct."""
    lease.assert_active()
    if (not spec.synthetic_fixture and torch.device(device).type != "cuda") or (spec.synthetic_fixture and torch.device(device).type != "cpu"):
        raise ValueError("CPU fixture and CUDA production scopes cannot be interchanged")
    compatibility = None
    if not spec.synthetic_fixture:
        if (qualification.get("status") != "passed" or "hardware" not in qualification
                or "qualification_receipt" not in contract or "qualification_sha256" not in contract):
            raise ValueError("Production requires the exact same-source qualification receipt, not a passed flag")
        if hardware is None:
            raise ValueError("Actual configured worker hardware is required for production qualification")
        if continuation is None:
            if qualification != verify_qualification(contract, hardware):
                raise ValueError("Production qualification differs from its immutable receipt")
            compatibility = _qualification_hardware_compatibility(qualification, hardware)
            compatibility.update(qualification_receipt_sha256=contract["qualification_sha256"],
                                 source_manifest_sha256=contract["source_manifest_sha256"],
                                 protocol_sha256=contract["protocol_sha256"], runtime=_runtime())
        else:
            from .observation_continuation import verify_for_execution
            compatibility = verify_for_execution(lease, source, spec, contract, qualification,
                resume, input_snapshot, hardware, continuation)
    elif continuation is not None:
        raise ValueError("Source-changing scientific continuation cannot use a CPU fixture")
    if (lease.root / "COMPLETE.json").exists() or (lease.root / "WORKER_FAILED.json").exists():
        raise RuntimeError("Closed worker cannot be rerun or automatically retried")
    started = lease.root / "WORKER_STARTED.json"
    if resume is None and started.exists():
        raise RuntimeError("Existing worker requires explicit checkpoint reconciliation/resume")
    owner = None
    try:
        def capacity(reserve=0):
            if continuation is None:
                return _capacity(lease.root, contract["output_limit_bytes"], reserve)
            from .observation_continuation import combined_capacity
            return combined_capacity(lease.root, contract, continuation, reserve)

        input_lease = _input_lease_receipt(lease, source, input_snapshot, resume=resume is not None)
        policy = RetentionPolicy(contract["study_root"], str(lease.root), identity_sha256(lease.identity), spec.audit_updates)
        bind_policy(lease, policy)
        if not started.exists():
            if compatibility is not None:
                _atomic_json(lease.root, "QUALIFICATION_COMPATIBILITY.json", compatibility)
            _atomic_json(lease.root, "WORKER_STARTED.json", {"schema": "credo.frontier.observation-worker-start.v1",
                "identity": lease.identity, "at_utc": _utc(), "scope": "synthetic_cpu_only" if spec.synthetic_fixture else "scientific",
                "qualification_receipt_sha256": contract.get("qualification_sha256"), "input_lease": input_lease,
                "runtime": _runtime(), "qualification_compatibility": compatibility})
        if resume is None:
            owner = ObservationFitter(lease, source, spec, device=device)
        else:
            owner = ObservationFitter.resume(lease, source, spec, device=device,
                directory=resume["directory"], expected_sha256=resume["sha256"])
        initial_updates = owner.accepted_updates
        if continuation is not None and (initial_updates, owner.attempted_updates, owner.optimizer_calls) != (1280, 1280, 1280):
            raise ValueError("Source-changing continuation must retain exactly 1280 consumed updates")
        times = {"audit_seconds": 0.0, "checkpoint_seconds": 0.0}
        for attribute, key in (("_audit_and_publish", "audit_seconds"), ("_checkpoint", "checkpoint_seconds")):
            operation = getattr(owner, attribute)

            def measured(*args, _operation=operation, _key=key, **kwargs):
                start = time.perf_counter()
                try:
                    return _operation(*args, **kwargs)
                finally:
                    times[_key] += time.perf_counter() - start

            setattr(owner, attribute, measured)
        milestones = sorted(set(range(spec.checkpoint_interval, spec.maximum_updates + 1, spec.checkpoint_interval))
                            | set(spec.audit_updates))
        reserve = max(1024 ** 2, sum(p.numel() * p.element_size() for p in owner.model.parameters()) * 5)
        for target in milestones:
            if target <= owner.accepted_updates:
                continue
            capacity(reserve)
            before_updates, before_times = owner.accepted_updates, dict(times)
            begin = time.perf_counter()
            owner.run_until(target)
            block_seconds = time.perf_counter() - begin
            audit_seconds = times["audit_seconds"] - before_times["audit_seconds"]
            checkpoint_seconds = times["checkpoint_seconds"] - before_times["checkpoint_seconds"]
            retention_start = time.perf_counter()
            prepared = prepare_retirement(lease, policy)
            retirement = None if prepared is None else commit_retirement(lease, policy, prepared)
            state = {"schema": "credo.frontier.observation-worker-progress.v1", "identity": lease.identity,
                "at_utc": _utc(), "accepted_updates": owner.accepted_updates, "attempted_updates": owner.attempted_updates,
                "actual_optimizer_calls": owner.optimizer_calls, "fitting_row_exposures": sum(owner.exposure_by_source.values()),
                "block_updates": target - before_updates, "block_total_seconds": block_seconds,
                "audit_seconds": audit_seconds, "checkpoint_publication_seconds": checkpoint_seconds,
                "training_and_input_seconds": block_seconds - audit_seconds - checkpoint_seconds,
                "retention_seconds": time.perf_counter() - retention_start,
                "latest_checkpoint": owner.last_checkpoint, "selected_candidate": owner.selected_candidate,
                "retired_payload_bytes": 0 if retirement is None else retirement["removed_payload_bytes"],
                "resources": _resources(device), "capacity": capacity()}
            if continuation is not None:
                state.update(consumed_parent_updates=initial_updates,
                             new_scientific_updates_this_owner=owner.accepted_updates-initial_updates,
                             source_continuation_contract=continuation)
            _atomic_json(lease.root, f"progress/update-{target:09d}.json", state)
            print(json.dumps({key: state[key] for key in ("at_utc", "accepted_updates", "attempted_updates", "block_total_seconds", "audit_seconds")}), flush=True)
        capacity(reserve)
        _, selected = owner.finalize_selected()
        input_exit = _input_exit_receipt(lease, source, input_snapshot, resume=resume is not None)
        final = {"schema": "credo.frontier.observation-worker-completion.v1", "status": "complete", "at_utc": _utc(),
                 "identity": lease.identity, "scope": "synthetic_cpu_only" if spec.synthetic_fixture else "scientific",
                 "accepted_updates": owner.accepted_updates, "attempted_updates": owner.attempted_updates,
                 "scientific_updates": 0 if spec.synthetic_fixture else owner.accepted_updates,
                 "actual_disposable_optimizer_calls": owner.optimizer_calls if spec.synthetic_fixture else 0,
                 "selected_export": selected, "audit_history": owner.audit_history,
                 "fitting_row_exposures": sum(owner.exposure_by_source.values()), "timings": times,
                 "timing_scope": "current_owner_only", "resume_from_accepted_updates": initial_updates,
                 "resources": _resources(device), "capacity": capacity(),
                 "input_lease": input_lease, "input_exit": input_exit, "qualification_compatibility": compatibility,
                 "main_dynamics_fitted": False, "held_out_donor_outcomes_opened": False}
        if continuation is not None:
            if owner.accepted_updates-initial_updates != 18720:
                raise ValueError("Source-changing continuation completion changed the remaining update budget")
            final.update(consumed_parent_updates=initial_updates,
                         new_scientific_updates_this_owner=owner.accepted_updates-initial_updates,
                         source_continuation_contract=continuation,
                         mixed_source_continuation=True)
        receipt = _atomic_json(lease.root, "COMPLETE.json", final)
        return final, receipt
    except BaseException as error:
        counters = None if owner is None else {"accepted_updates": owner.accepted_updates,
            "attempted_updates": owner.attempted_updates, "optimizer_calls": owner.optimizer_calls}
        _fail(lease, contract, error, counters=counters)
        raise


def _synthetic_full_counts():
    generator = np.random.Generator(np.random.PCG64DXSM(104729))
    rows, features, per_row = 2048, 18129, 4096
    indices = np.empty(rows * per_row, dtype=np.int32)
    for row in range(rows):
        indices[row * per_row:(row + 1) * per_row] = np.sort(generator.choice(features, per_row, replace=False))
    values = generator.integers(1, 8, len(indices), dtype=np.uint32)
    return sparse.csr_matrix((values, indices, np.arange(rows + 1, dtype=np.int32) * per_row), shape=(rows, features))


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(child) for child in value)
    return value


def _exact_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_exact_tree(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_exact_tree(a, b) for a, b in zip(left, right))
    return left == right


def _fp32_comparison(reference, actual, *, atol, rtol):
    """Elementwise prebound tolerance; diagnostics never relax the threshold."""
    reference, actual = reference.detach().cpu().double(), actual.detach().cpu().double()
    if reference.shape != actual.shape or reference.numel() == 0:
        raise ValueError("Paired FP32 arrays must have the same nonempty shape")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("Nonfinite value in paired FP32 comparison")
    error = torch.abs(actual - reference)
    normalized = error / (atol + rtol * torch.abs(reference))
    return {"elements": reference.numel(), "maximum_absolute_error": float(error.max()),
            "maximum_reference_magnitude": float(reference.abs().max()),
            "maximum_tolerance_normalized_error": float(normalized.max()),
            "passed": bool((normalized <= 1).all())}


def _paired_fp32_check(counts, row_ids, fold):
    """Full-axis first-16-row CPU/CUDA objective/gradient check, no optimizer."""
    if counts.shape != (2048, 18129) or len(row_ids) != 2048:
        raise ValueError("Paired check requires the bound full-shape fabricated fixture")
    mini = counts[:16].copy()
    ids = row_ids[:16]
    cpu = SparseR48(fold, maximum_rows=2048, maximum_dense_bytes=4 * 1024 ** 3, seed=0)
    cuda = SparseR48(fold, maximum_rows=2048, maximum_dense_bytes=4 * 1024 ** 3, seed=0).cuda()
    if not _exact_tree(cpu.state_dict(), _cpu_tree(cuda.state_dict())):
        raise ValueError("Paired CPU/CUDA initialization differs")
    arguments = {"row_ids": ids, "donors": ["D3"] * 16, "conditions": ["Stim8hr"] * 16, "thinning_seed": 0}
    cpu_result = cpu.symmetric_count_loss(mini, **arguments)
    cuda_result = cuda.symmetric_count_loss(mini, **arguments)
    if cpu_result["scored_cell_directions"] != cuda_result["scored_cell_directions"]:
        raise ValueError("Paired CPU/CUDA scored directions differ")
    cpu_result["loss"].backward()
    cuda_result["loss"].backward()
    loss = _fp32_comparison(cpu_result["loss"], cuda_result["loss"],
                           atol=PAIRED_FP32_TOLERANCES["loss_atol"], rtol=PAIRED_FP32_TOLERANCES["loss_rtol"])
    gradients = {}
    cpu_parameters, cuda_parameters = dict(cpu.named_parameters()), dict(cuda.named_parameters())
    if cpu_parameters.keys() != cuda_parameters.keys():
        raise ValueError("Paired CPU/CUDA parameter inventories differ")
    for name, parameter in cpu_parameters.items():
        other = cuda_parameters[name]
        if parameter.grad is None or other.grad is None:
            raise ValueError(f"Missing paired parameter gradient: {name}")
        gradients[name] = _fp32_comparison(parameter.grad, other.grad,
            atol=PAIRED_FP32_TOLERANCES["gradient_atol"], rtol=PAIRED_FP32_TOLERANCES["gradient_rtol"])
    summary = {"elements": sum(item["elements"] for item in gradients.values()),
               "maximum_absolute_error": max(item["maximum_absolute_error"] for item in gradients.values()),
               "maximum_reference_magnitude": max(item["maximum_reference_magnitude"] for item in gradients.values()),
               "maximum_tolerance_normalized_error": max(item["maximum_tolerance_normalized_error"] for item in gradients.values()),
               "passed": all(item["passed"] for item in gradients.values())}
    result = {"schema": "credo.frontier.observation-paired-fp32.v1", "status": "passed" if loss["passed"] and summary["passed"] else "failed",
              "shape": [16, 18129], "architecture": [18129, 512, 128, 48], "tolerances": dict(PAIRED_FP32_TOLERANCES),
              "optimizer_calls": 0, "all_parameter_gradients_checked": True, "same_initial_weights": True,
              "cpu_encoding": "sparse", "cuda_encoding": "bounded_dense", "scored_cell_directions": cpu_result["scored_cell_directions"],
              "loss": loss, "gradient_summary": summary, "parameter_gradients": gradients}
    del cpu, cuda, cpu_result, cuda_result, cpu_parameters, cuda_parameters, parameter, other
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_qualification(lease, contract, hardware):
    """Thirteen disposable full-shape CUDA calls; never a real observation fit."""
    calls = 0
    if any((lease.root / name).exists() for name in ("WORKER_STARTED.json", "WORKER_FAILED.json", "QUALIFICATION_COMPLETE.json")):
        raise RuntimeError("Existing qualification cannot be rerun or retried")
    try:
        _atomic_json(lease.root, "WORKER_STARTED.json", {"mode": "fabricated_cuda_qualification", "at_utc": _utc(),
                                                       "identity": lease.identity, "maximum_optimizer_calls": 16})
        counts = _synthetic_full_counts()
        row_ids = tuple(f"fabricated:D3:Stim8hr:row-{row}" for row in range(2048))
        fold = ObservationFold("qualification-D3-D4", ("D3", "D4"), "f" * 64)
        paired = _paired_fp32_check(counts, row_ids, fold)
        paired_receipt = _atomic_json(lease.root, "PAIRED_FP32_CHECK.json", paired)
        if paired["status"] != "passed":
            raise ValueError("Paired CPU/CUDA FP32 loss or gradient exceeds its fixed tolerance")

        def objects():
            random.seed(67)
            np.random.seed(67)
            torch.manual_seed(67)
            torch.cuda.manual_seed_all(67)
            model = SparseR48(fold, maximum_rows=2048, maximum_dense_bytes=4 * 1024 ** 3, seed=0).cuda()
            return model, torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

        def step(model, optimizer, position):
            nonlocal calls
            if calls >= 16:
                raise RuntimeError("Qualification optimizer-call cap reached")
            random.random(), np.random.random(), torch.rand(3), torch.rand(3, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            result = model.symmetric_count_loss(counts, row_ids=row_ids, donors=["D3"] * 2048,
                                                conditions=["Stim8hr"] * 2048, thinning_seed=position)
            if result["loss"] is None or not bool(torch.isfinite(result["loss"])):
                raise ValueError("Nonfinite qualification objective")
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            calls += 1
            optimizer.step()
            if not all(bool(torch.isfinite(value).all()) for value in model.parameters()):
                raise ValueError("Nonfinite qualification parameter")
            return float(result["loss"].detach())

        model, optimizer = objects()
        for update in range(2):
            step(model, optimizer, update)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings, losses = [], []
        for update in range(2, 5):
            start = time.perf_counter()
            losses.append(step(model, optimizer, update))
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        measured = _resources("cuda")
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        full, full_optimizer = objects()
        for update in range(4):
            step(full, full_optimizer, update)
        expected_model, expected_optimizer = _cpu_tree(full.state_dict()), _cpu_tree(full_optimizer.state_dict())
        expected_rng = (random.random(), float(np.random.random()), torch.rand(3), torch.rand(3, device="cuda").cpu())
        del full, full_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        partial, partial_optimizer = objects()
        for update in range(2):
            step(partial, partial_optimizer, update)
        _capacity(lease.root, contract["output_limit_bytes"], 1024 ** 3)
        checkpoint = save_checkpoint(lease, model=partial, optimizer=partial_optimizer, scheduler=None,
            accepted_updates=2, attempted_updates=2, phase="observation_cuda_qualification",
            sampler_state={"position": 2, "next_thinning_seed": 2, "row_ids_sha256": identity_sha256(row_ids)},
            count_cache={"active": False}, execution_scope="synthetic_cuda_qualification", milestone=True)
        del partial, partial_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        resumed, resumed_optimizer = objects()
        restored = load_checkpoint(lease, checkpoint["directory"], expected_sha256=checkpoint["checkpoint_sha256"],
            model=resumed, optimizer=resumed_optimizer, scheduler=None, execution_scope="synthetic_cuda_qualification")
        if restored["scientific_updates"] != 0 or restored["count_cache"] != {"active": False} or restored["sampler_state"] != {
                "position": 2, "next_thinning_seed": 2, "row_ids_sha256": identity_sha256(row_ids)}:
            raise ValueError("CUDA qualification sampler/scope restore mismatch")
        for update in range(restored["sampler_state"]["position"], 4):
            step(resumed, resumed_optimizer, update)
        actual_rng = (random.random(), float(np.random.random()), torch.rand(3), torch.rand(3, device="cuda").cpu())
        if not (_exact_tree(expected_model, _cpu_tree(resumed.state_dict()))
                and _exact_tree(expected_optimizer, _cpu_tree(resumed_optimizer.state_dict()))
                and _exact_tree(expected_rng, actual_rng)):
            raise ValueError("CUDA restart is not exact for model/optimizer/sampler/RNG")
        if calls != QUALIFICATION_OPTIMIZER_CALLS:
            raise ValueError("Unexpected qualification optimizer accounting")
        final = {"schema": "credo.frontier.observation-cuda-qualification.v1", "status": "passed", "at_utc": _utc(),
            "source_manifest_sha256": contract["source_manifest_sha256"], "protocol_sha256": contract["protocol_sha256"],
            "runtime_image": contract["runtime_image"], "runtime": _runtime(), "hardware": hardware,
            "specification_sha256": ObservationTrainingSpec().identity, "scientific_updates": 0,
            "actual_disposable_optimizer_calls": calls, "real_count_payloads_opened": False,
            "real_input_mount_qualified": False, "restart_equivalence": "exact_model_optimizer_sampler_rng",
            "shape": [2048, 18129], "nonzeros_per_cell": 4096, "architecture": [18129, 512, 128, 48],
            "warmup_optimizer_calls": 2, "timed_optimizer_calls": 3, "timed_seconds": timings,
            "timed_losses": losses, "positive_cells_per_second": 2048 / float(np.mean(timings)),
            "resources_after_warmup": measured, "resources_all_checks": _resources("cuda"),
            "capacity": _capacity(lease.root, contract["output_limit_bytes"]),
            "audit_or_real_input_throughput_measured": False}
        final.update(paired_fp32_check=paired, paired_fp32_receipt=paired_receipt)
        return final, _atomic_json(lease.root, "QUALIFICATION_COMPLETE.json", final)
    except BaseException as error:
        _fail(lease, contract, error, counters={"actual_disposable_optimizer_calls": calls, "scientific_updates": 0})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "fit"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--resume-directory")
    parser.add_argument("--resume-sha256")
    arguments = parser.parse_args(argv)
    if bool(arguments.resume_directory) != bool(arguments.resume_sha256) or (arguments.mode == "qualify" and arguments.resume_directory):
        parser.error("Explicit resume needs both checkpoint directory and SHA, and is never a qualification retry")
    contract, spec = load_contract(arguments.contract, arguments.contract_sha256, arguments.mode)
    hardware = _configure_cuda()  # Before metadata intake or any actual count read.
    if arguments.mode == "qualify":
        identity = _lease_identity(contract, spec, identity_sha256({"fabricated_shape": [2048, 18129], "nnz_per_cell": 4096}),
                                   "qualification-D3-D4", arguments.contract_sha256)
        with TrajectoryLease(contract["output_root"], identity, contract["worker_id"]) as lease:
            _capacity(lease.root, contract["output_limit_bytes"], 1024 ** 3)
            result, receipt = run_qualification(lease, contract, hardware)
    else:
        qualification = verify_qualification(contract, hardware)
        index = NativeInputIndex(contract["metadata_root"], contract["reader_source_root"])
        fold = FoldInputs(contract["fold_id"], tuple(contract["training_donors"]))
        access = index.access(fold, "representation_fit")
        if (index.identity != contract["metadata_index_sha256"] or index.rna_axis_sha256 != contract["rna_axis_sha256"]
                or access.identity() != contract["input_access_sha256"]):
            raise ValueError("Fold-native input identities differ from the frozen contract")
        handle = index.open(contract["dataset_root"], fold, "representation_fit", max_batch_rows=2048)
        input_snapshot = handle.capture_publication_identity()
        rows = ObservationRows(handle, ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256))
        identity = _lease_identity(contract, spec, rows.identity, fold.fold_id, arguments.contract_sha256)
        with TrajectoryLease(contract["output_root"], identity, contract["worker_id"]) as lease:
            resume = None if arguments.resume_directory is None else {
                "directory": str(_path(arguments.resume_directory)), "sha256": _sha(arguments.resume_sha256)}
            result, receipt = run_fitting_core(lease, rows, spec, contract, device="cuda:0", qualification=qualification,
                                               resume=resume, input_snapshot=input_snapshot, hardware=hardware)
    print(json.dumps({"status": result["status"], "receipt": receipt}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
