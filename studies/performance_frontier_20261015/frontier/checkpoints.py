"""Resumable trajectory snapshots; historical inference checkpoints are rejected.

No training, remote access, deletion policy, or automatic restart is implemented
here. Callers own admission and update accounting. A live exclusive lease is
required for publication and restoration. Only tensors and primitive containers
are serialized, and loading always uses PyTorch's restricted ``weights_only``
loader. Runtime and scientific bindings must match exactly.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import uuid

import numpy as np
import torch


SCHEMA = "credo.frontier.resumable_checkpoint.v1"
IDENTITY_KEYS = {
    "trajectory_id", "fold_id", "seed", "input_sha256", "configuration_sha256", "source_sha256"
}


class CheckpointByteLimitExceeded(RuntimeError):
    """The declared payload cap was reached; partial output is not resumable."""


class _BoundedCheckpointWriter:
    """Sequential file-like target whose unsuccessful writes add no bytes.

    torch.save's ZIP writer may attempt to emit its footer after an exception.
    Latch the first overflow so that those finalization writes cannot bypass
    the cap. The caller retains/flushes the actual file; no retry or deletion.
    This bounds serialized bytes, not CPU snapshot-copy memory or total output.
    """
    def __init__(self, stream, maximum_bytes):
        if type(maximum_bytes) is not int or maximum_bytes < 1:
            raise ValueError("Positive integer checkpoint payload cap required")
        if stream.tell() != 0:
            raise ValueError("Bounded checkpoint writer requires a fresh empty file")
        self.stream, self.maximum_bytes = stream, maximum_bytes
        self.written = 0
        self.exceeded = False
        self.failed = False

    def write(self, data):
        if self.failed:
            raise OSError("Checkpoint writer already failed; no finalization writes")
        size = memoryview(data).nbytes
        if self.exceeded or size > self.maximum_bytes-self.written:
            self.exceeded = True
            raise CheckpointByteLimitExceeded("Serialized checkpoint exceeds declared payload cap")
        try:
            count = self.stream.write(data)
        except BaseException:
            self.failed = True
            raise
        if count != size:
            self.failed = True
            raise OSError("Short checkpoint write; publication is incomplete")
        self.written += count
        return count

    def flush(self):
        return self.stream.flush()


def _identity(value):
    if not isinstance(value, dict) or not IDENTITY_KEYS.issubset(value):
        raise ValueError("Exact trajectory, fold, seed, input, configuration and source bindings required")
    for key in ("trajectory_id", "fold_id"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value[key]):
            raise ValueError(f"Invalid {key}")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("Nonnegative integer seed required")
    for key in ("input_sha256", "configuration_sha256", "source_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError(f"Exact lowercase SHA-256 required for {key}")
    # Additional declared scientific bindings are retained, never discarded.
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return json.loads(encoded)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink paths are not admitted")
    return path


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_once(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(Path(path).parent)


class TrajectoryLease(AbstractContextManager):
    """One local/POSIX writer per exact trajectory, with immutable claim records.

    A second process (or second open in this process) cannot acquire the same
    root while this lease lives. GPFS/worker visibility and remote locking still
    require actual-runtime qualification. This is not scheduler admission.
    """

    def __init__(self, root, identity, worker_id):
        self.root = _safe_path(root)
        self.identity = _identity(identity)
        if not isinstance(worker_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", worker_id):
            raise ValueError("An explicit safe worker identity is required")
        self.worker_id = worker_id
        self._lock = None
        self.claim_id = None
        self.pid = None

    def __enter__(self):
        if self._lock is not None:
            raise RuntimeError("Lease is already active")
        self.root.mkdir(exist_ok=True)  # Parent must already be explicitly prepared.
        descriptor = os.open(self.root / "OWNER.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        lock = os.fdopen(descriptor, "r+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity_path = _safe_path(self.root / "IDENTITY.json")
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != self.identity:
                    raise ValueError("Trajectory identity changed")
            else:
                if set(entry.name for entry in self.root.iterdir()) != {"OWNER.lock"}:
                    raise ValueError("Unbound existing trajectory contents")
                _write_json_once(identity_path, self.identity)
            for name in ("claims", "checkpoints"):
                _safe_path(self.root / name).mkdir(exist_ok=True)
            self.claim_id = uuid.uuid4().hex
            self.pid = os.getpid()
            _write_json_once(self.root / "claims" / f"{self.claim_id}.json", {
                "worker_id": self.worker_id, "pid": self.pid, "claim_id": self.claim_id,
                "trajectory_id": self.identity["trajectory_id"], "exclusive_lock": "flock_nonblocking",
            })
            self._lock = lock
            return self
        except BaseException:
            lock.close()
            raise

    def assert_active(self):
        if self._lock is None or self._lock.closed or self.pid != os.getpid():
            raise RuntimeError("A live, same-process exclusive trajectory lease is required")
        if json.loads(_safe_path(self.root / "IDENTITY.json").read_text()) != self.identity:
            raise ValueError("Active trajectory binding changed")

    def __exit__(self, exc_type, exc, traceback):
        if self._lock is not None:
            try:
                _write_json_once(self.root / "claims" / f"{self.claim_id}.released.json", {
                    "claim_id": self.claim_id, "exception_type": None if exc_type is None else exc_type.__name__,
                    "automatic_restart_authorized": False,
                })
            finally:
                self._lock.close()
                self._lock = None
        return False


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError("Nonfinite checkpoint tensor")
        return tensor
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise TypeError("Checkpoint mapping keys must be strings or integers")
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else [],
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
    }


def _runtime():
    return {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)}


def _class_name(value):
    return None if value is None else f"{type(value).__module__}.{type(value).__qualname__}"


def _optimizer_layout(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("Optimizer contains a parameter outside the bound model") from error


def _constructor_metadata(model):
    """Bind non-state_dict scientific routing without deserializing objects.

    Generic torch modules retain the model-class/state-layout checks below.
    Frontier generators additionally bind ordered identities and support, while
    the observation module binds its fold and feature-axis configuration.
    """
    result = {"model_class": _class_name(model)}
    configuration = getattr(model, "config", None)
    if is_dataclass(configuration):
        result["configuration"] = asdict(configuration)
        result["ordered_target_ids"] = list(model.target_ids)
        result["ordered_control_ids"] = list(model.control_ids)
        factors = getattr(model, "support_factors", None)
        result["target_support_factors"] = None if factors is None else factors.detach().cpu().tolist()
    fold = getattr(model, "fold", None)
    if is_dataclass(fold):
        result["observation_fold"] = asdict(fold)
        for name in ("features", "latent_dim", "hidden_dims", "maximum_rows", "maximum_dense_bytes"):
            result[name] = getattr(model, name)
    # Canonical primitive-only representation, independent of checkpoint pickle.
    return json.loads(json.dumps(_cpu_tree(result), sort_keys=True, allow_nan=False))


def _gradient_mask(model):
    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def _check_phase_label(model, phase):
    if hasattr(model, "phase"):
        if model.phase != phase or not callable(getattr(model, "set_phase", None)):
            raise ValueError("Checkpoint phase differs from the actual model phase/setter")
        if hasattr(model, "reaction_enabled") and phase in ("state", "joint"):
            if model.reaction_enabled is not (phase == "joint"):
                raise ValueError("Model reaction flag is inconsistent with its phase")


def _check_gradient_mask(model, mask, phase):
    current = _gradient_mask(model)
    if (not isinstance(mask, dict) or set(mask) != set(current)
            or any(type(value) is not bool for value in mask.values())):
        raise ValueError("Checkpoint requires_grad mask does not match named parameters")
    if not hasattr(model, "phase"):
        return  # Generic modules have no phase-defined canonical mask.
    setter = getattr(model, "set_phase", None)
    if not callable(setter):
        raise ValueError("Phase-dependent model requires an explicit phase setter")
    original_phase = model.phase
    original_reaction = getattr(model, "reaction_enabled", None)
    try:
        setter(phase)
        _check_phase_label(model, phase)
        if _gradient_mask(model) != mask:
            raise ValueError("Checkpoint requires_grad mask is inconsistent with its phase")
    finally:
        setter(original_phase)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(current[name])
        if hasattr(model, "reaction_enabled"):
            model.reaction_enabled = original_reaction


def _restore_phase_and_mask(model, phase, mask):
    if hasattr(model, "phase"):
        model.set_phase(phase)
        _check_phase_label(model, phase)
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(mask[name])
    if _gradient_mask(model) != mask:
        raise ValueError("Restored model phase/gradient mask does not match the checkpoint")


def _validate_rng(rng):
    # Validate using private generators, before changing model or global RNG state.
    random.Random().setstate(rng["python"])
    state = rng["numpy"]
    np.random.RandomState().set_state((state["algorithm"], np.asarray(state["keys"], dtype=np.uint32),
                                      state["position"], state["has_gauss"], state["cached_gaussian"]))
    torch.Generator(device="cpu").set_state(rng["torch_cpu"])
    for value in rng["torch_cuda"]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Malformed CUDA RNG state")


def _completed(lease):
    lease.assert_active()
    entries = []
    for directory in sorted((lease.root / "checkpoints").iterdir()):
        _safe_path(directory)
        receipt_path = _safe_path(directory / "COMPLETE.json")
        if not receipt_path.exists():
            raise ValueError("Incomplete checkpoint publication requires explicit recovery; no automatic retry")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != SCHEMA or receipt.get("identity") != lease.identity:
            raise ValueError("Checkpoint inventory identity mismatch")
        entries.append((directory, receipt))
    return sorted(entries, key=lambda item: item[1]["accepted_updates"])


def retention_plan(lease):
    """Return latest/previous/milestones without deleting or rewriting anything."""
    entries = _completed(lease)
    paths = [str(path) for path, _ in entries]
    milestones = [str(path) for path, receipt in entries if receipt["milestone"]]
    retained = set(milestones + paths[-2:])
    return {"latest": paths[-1] if paths else None, "previous": paths[-2] if len(paths) > 1 else None,
            "milestones": milestones, "retain": sorted(retained),
            "other_history": [path for path in paths if path not in retained],
            "deletion_performed": False, "deletion_authorized": False}


def save_checkpoint(lease, *, model, optimizer, scheduler, accepted_updates, attempted_updates,
                    phase, sampler_state, count_cache, execution_scope, milestone=False,
                    last_attempt_status="accepted", maximum_checkpoint_bytes=None):
    """Publish one accepted-update boundary, with exact hash and no clobber.

    A supplied payload limit is enforced DURING serialization, including archive
    finalization. Aggregate quota/metadata and CPU snapshot-memory admission are
    separate worker obligations. None is inferred from this per-file limit.
    """
    lease.assert_active()
    if maximum_checkpoint_bytes is not None and (type(maximum_checkpoint_bytes) is not int
                                                or maximum_checkpoint_bytes < 1):
        raise ValueError("Positive integer checkpoint payload cap required")
    if (type(accepted_updates) is not int or type(attempted_updates) is not int
            or accepted_updates < 0 or attempted_updates != accepted_updates
            or last_attempt_status != "accepted"):
        raise ValueError("Failed/unaccepted optimizer attempts cannot publish resumable checkpoints")
    if execution_scope not in ("synthetic_cpu_only", "synthetic_cuda_qualification", "scientific") or not isinstance(phase, str) or not phase:
        raise ValueError("Explicit execution scope and phase required")
    if not isinstance(sampler_state, dict) or not isinstance(count_cache, dict):
        raise ValueError("Explicit sampler position/state and count-cache mappings are required")
    if "position" not in sampler_state:
        raise ValueError("Sampler position is required")
    _check_phase_label(model, phase)
    gradient_mask = _gradient_mask(model)
    _check_gradient_mask(model, gradient_mask, phase)
    constructor_metadata = _constructor_metadata(model)
    device_types = sorted({parameter.device.type for parameter in model.parameters()})
    if execution_scope == "synthetic_cpu_only" and device_types != ["cpu"]:
        raise ValueError("Synthetic checkpoint scope is CPU-only")
    if execution_scope == "synthetic_cuda_qualification" and device_types != ["cuda"]:
        raise ValueError("Synthetic CUDA qualification requires an actual CUDA model")
    entries = _completed(lease)
    if entries and accepted_updates <= entries[-1][1]["accepted_updates"]:
        raise FileExistsError("Checkpoint update must increase; history cannot be overwritten")
    model_state = model.state_dict()
    payload = _cpu_tree({
        "schema": SCHEMA, "resume_capable": True, "identity": lease.identity,
        "runtime": _runtime(), "execution_scope": execution_scope,
        "model_class": _class_name(model), "optimizer_class": _class_name(optimizer),
        "model_constructor_metadata": constructor_metadata, "requires_grad_mask": gradient_mask,
        "scheduler_class": _class_name(scheduler), "model_device_types": device_types,
        "model_state": model_state, "model_state_metadata": getattr(model_state, "_metadata", None),
        "model_training": bool(model.training), "optimizer_parameter_layout": _optimizer_layout(model, optimizer),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng": _rng_state(), "sampler_state": sampler_state, "count_cache": count_cache,
        "phase": phase, "accepted_updates": accepted_updates, "attempted_updates": attempted_updates,
        "last_attempt_status": last_attempt_status,
        "scientific_updates": accepted_updates if execution_scope == "scientific" else 0,
        "disposable_optimizer_calls": accepted_updates if execution_scope in ("synthetic_cpu_only", "synthetic_cuda_qualification") else 0,
    })
    directory = lease.root / "checkpoints" / f"update-{accepted_updates:09d}"
    directory.mkdir()  # A failed publication remains visible and cannot be retried implicitly.
    checkpoint = directory / "checkpoint.pt"
    with checkpoint.open("xb") as handle:
        writer = handle if maximum_checkpoint_bytes is None else _BoundedCheckpointWriter(handle, maximum_checkpoint_bytes)
        try:
            torch.save(payload, writer)
        except BaseException as error:
            if maximum_checkpoint_bytes is not None and writer.exceeded:
                raise CheckpointByteLimitExceeded("Serialized checkpoint exceeds declared payload cap; no retry") from error
            raise
        finally:
            handle.flush()
            os.fsync(handle.fileno())
    receipt = {"schema": SCHEMA, "identity": lease.identity, "checkpoint_sha256": file_sha256(checkpoint),
               "checkpoint_bytes": checkpoint.stat().st_size, "accepted_updates": accepted_updates,
               "attempted_updates": attempted_updates, "phase": phase, "execution_scope": execution_scope,
               "milestone": bool(milestone), "claim_id": lease.claim_id, "worker_id": lease.worker_id}
    if maximum_checkpoint_bytes is not None:
        if receipt["checkpoint_bytes"] > maximum_checkpoint_bytes:
            raise ValueError("Checkpoint byte-limit readback failed")
        receipt["maximum_checkpoint_bytes"] = maximum_checkpoint_bytes
    _write_json_once(directory / "COMPLETE.json", receipt)
    _sync_directory(directory.parent)
    return {"directory": str(directory), **receipt}


def load_checkpoint(lease, directory, *, expected_sha256, model, optimizer, scheduler, execution_scope):
    """Restore the latest complete checkpoint; never use arbitrary pickle loading.

    Phase/reaction flags and the named requires_grad mask are restored and
    verified here. The returned sampler state and count cache must still be
    installed by the owning trainer before it continues. Custom RNG generators
    belong in those explicit state mappings; global RNG capture cannot discover
    arbitrary generator objects owned by a caller.
    """
    lease.assert_active()
    directory = _safe_path(directory)
    entries = _completed(lease)
    if not entries or directory != entries[-1][0]:
        raise ValueError("Only this trajectory's latest complete checkpoint may resume")
    receipt = entries[-1][1]
    path = _safe_path(directory / "checkpoint.pt")
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or receipt["checkpoint_sha256"] != expected_sha256
            or file_sha256(path) != expected_sha256 or path.stat().st_size != receipt["checkpoint_bytes"]):
        raise ValueError("Checkpoint payload hash/size mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("resume_capable") is not True
            or payload.get("identity") != lease.identity or payload.get("runtime") != _runtime()
            or payload.get("execution_scope") != execution_scope
            or payload.get("model_class") != _class_name(model)
            or payload.get("model_constructor_metadata") != _constructor_metadata(model)
            or payload.get("optimizer_class") != _class_name(optimizer)
            or payload.get("scheduler_class") != _class_name(scheduler)
            or payload.get("optimizer_parameter_layout") != _optimizer_layout(model, optimizer)
            or payload.get("model_device_types") != sorted({p.device.type for p in model.parameters()})
            or payload.get("accepted_updates") != receipt["accepted_updates"]
            or payload.get("attempted_updates") != payload.get("accepted_updates")
            or payload.get("last_attempt_status") != "accepted" or payload.get("phase") != receipt["phase"]):
        raise ValueError("Not an exact compatible healthy resumable checkpoint")
    _cpu_tree(payload)  # Reject unexpected objects and nonfinite tensors before mutation.
    _check_gradient_mask(model, payload.get("requires_grad_mask"), payload["phase"])
    current = model.state_dict()
    saved = payload["model_state"]
    if set(current) != set(saved) or any(current[key].shape != saved[key].shape
            or current[key].dtype != saved[key].dtype for key in current):
        raise ValueError("Model state structure changed")
    groups = optimizer.state_dict()["param_groups"]
    saved_groups = payload["optimizer_state"]["param_groups"]
    if len(groups) != len(saved_groups) or any(len(a["params"]) != len(b["params"])
                                             for a, b in zip(groups, saved_groups)):
        raise ValueError("Optimizer parameter-group structure changed")
    cuda_rng = payload["rng"]["torch_cuda"]
    if cuda_rng and (not torch.cuda.is_initialized() or len(cuda_rng) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG topology differs from checkpoint")
    _validate_rng(payload["rng"])
    saved = OrderedDict(saved)
    if payload["model_state_metadata"] is not None:
        saved._metadata = payload["model_state_metadata"]
    _restore_phase_and_mask(model, payload["phase"], payload["requires_grad_mask"])
    model.load_state_dict(saved, strict=True)
    model.train(payload["model_training"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    numpy_state = payload["rng"]["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    random.setstate(payload["rng"]["python"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    return {key: payload[key] for key in ("accepted_updates", "attempted_updates", "phase", "sampler_state",
                                         "count_cache", "scientific_updates", "disposable_optimizer_calls")}
