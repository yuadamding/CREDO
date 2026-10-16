"""Verified selected-observation export -> frozen CPU module and owner binding.

No artifact discovery, best-score re-selection, historical initialization,
optimizer restoration, count access or GPU operation is performed here. The
caller supplies separately admitted immutable references. Their numerical and
lifecycle acceptance is not inferred merely from a directory or a hash string.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import torch

from .observation import ObservationFold, SparseR48
from .observation_training import ObservationTrainingSpec
from .training_owner import ObservationBinding, observation_state_digest


STUDY = "credo_performance_frontier_20261015_v1"
MAXIMUM_ARTIFACT_BYTES = 256 * 1024 ** 2
MAXIMUM_METADATA_BYTES = 32 * 1024 ** 2


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Exact lowercase SHA-256 required")
    return value


def _path(value):
    path = Path(value)
    if (not path.is_absolute() or ".." in path.parts
            or any(part.is_symlink() for part in (path, *path.parents))):
        raise ValueError("Explicit absolute nonsymlink artifact/metadata path required")
    return path


@dataclass(frozen=True)
class ObservationFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        _path(self.path)
        _sha(self.sha256)
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= MAXIMUM_ARTIFACT_BYTES:
            raise ValueError("Positive bounded exact file size required")


@dataclass(frozen=True)
class ObservationArtifactAdmission:
    artifact: ObservationFile
    receipt: ObservationFile
    completion: ObservationFile
    worker_contract: ObservationFile
    protocol: ObservationFile
    fold: ObservationFold
    specification: ObservationTrainingSpec
    trajectory_identity: dict

    def __post_init__(self):
        if (not all(isinstance(item, ObservationFile) for item in
                (self.artifact, self.receipt, self.completion, self.worker_contract, self.protocol))
                or not isinstance(self.fold, ObservationFold)
                or not isinstance(self.specification, ObservationTrainingSpec)
                or not isinstance(self.trajectory_identity, dict)):
            raise TypeError("Explicit file, fold, protocol and trajectory admissions required")
        root = _path(self.artifact.path).parent
        if (STUDY not in root.parts or Path(self.artifact.path).name != "SELECTED_OBSERVATION.pt"
                or _path(self.receipt.path) != root / "SELECTED_OBSERVATION.pt.json"
                or _path(self.completion.path) != root / "COMPLETE.json"):
            raise ValueError("Exact selected export and receipts from this new study required")
        identity = self.trajectory_identity
        required = {"trajectory_id", "fold_id", "seed", "input_sha256", "configuration_sha256",
                    "source_sha256", "worker_contract_sha256"}
        if not required <= set(identity):
            raise ValueError("Complete admitted observation trajectory identity required")
        for name in ("input_sha256", "configuration_sha256", "source_sha256", "worker_contract_sha256"):
            _sha(identity[name])
        if (identity["fold_id"] != self.fold.fold_id or identity["seed"] != 0
                or identity["configuration_sha256"] != self.specification.identity
                or identity["worker_contract_sha256"] != self.worker_contract.sha256):
            raise ValueError("Admitted fold, protocol or worker contract differs from trajectory")
        # Reject non-JSON or nonfinite admission metadata before any load.
        json.dumps(identity, sort_keys=True, allow_nan=False)


def _stat_identity(info):
    return tuple(getattr(info, name) for name in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink"))


@contextmanager
def _verified_file(reference, *, maximum):
    path = _path(reference.path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != reference.size_bytes or before.st_size > maximum):
        raise ValueError("Exact regular single-link file size required")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if _stat_identity(os.fstat(stream.fileno())) != _stat_identity(before):
            raise ValueError("Observation file changed while opening")
        digest = hashlib.sha256()
        while block := stream.read(8 * 1024 ** 2):
            digest.update(block)
        if digest.hexdigest() != reference.sha256:
            raise ValueError("Observation file SHA-256 mismatch before loading")
        if (_stat_identity(os.fstat(stream.fileno())) != _stat_identity(before)
                or _stat_identity(path.lstat()) != _stat_identity(before)):
            raise ValueError("Observation file changed during verification")
        stream.seek(0)
        yield stream
        if (_stat_identity(os.fstat(stream.fileno())) != _stat_identity(before)
                or _stat_identity(path.lstat()) != _stat_identity(before)):
            raise ValueError("Observation file changed during loading")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in observation metadata")
        result[key] = value
    return result


def _json(reference):
    with _verified_file(reference, maximum=MAXIMUM_METADATA_BYTES) as stream:
        result = json.load(stream, object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite observation metadata")))
    if not isinstance(result, dict):
        raise ValueError("Observation metadata must be an object")
    return result


def _metadata(admission):
    spec, fold, identity = admission.specification, admission.fold, admission.trajectory_identity
    root = Path(admission.artifact.path).parent
    for name in ("OBSERVATION_FAILED.json", "WORKER_FAILED.json"):
        if os.path.lexists(root / name):
            raise ValueError("Failed observation trajectory cannot supply an inference export")
    contract, protocol = _json(admission.worker_contract), _json(admission.protocol)
    expected_contract = {"schema": "credo.frontier.observation-worker-contract.v1", "mode": "fit",
        "output_root": str(root), "trajectory_id": identity["trajectory_id"], "fold_id": fold.fold_id,
        "training_donors": list(fold.training_donors), "rna_axis_sha256": fold.feature_axis_sha256,
        "input_access_sha256": identity["input_sha256"], "source_manifest_sha256": identity["source_sha256"],
        "protocol_path": admission.protocol.path, "protocol_sha256": admission.protocol.sha256}
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Worker contract does not admit this exact fold, feature axis and input/source identity")
    _sha(contract.get("metadata_index_sha256"))
    expected_protocol = {"schema": "credo.frontier.observation_protocol.v1",
        "RNA_feature_axis_identity": fold.feature_axis_sha256, "RNA_feature_count": spec.features,
        "architecture_dimensions": [spec.features, *spec.hidden_dims, spec.latent_dim],
        "maximum_accepted_updates_each": spec.maximum_updates,
        "logical_batch_positive_training_cells": spec.logical_batch,
        "audit_candidates_accepted_updates": list(spec.audit_updates),
        "checkpoint_interval_accepted_updates": spec.checkpoint_interval,
        "maximum_dense_working_estimate_bytes": spec.maximum_dense_bytes,
        "fitting_conditions": ["Stim8hr", "Stim48hr"], "historical_checkpoint_initialization": False,
        "initialization_seed": 0, "early_stopping": False,
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                      "betas": [0.9, 0.999], "epsilon": 1e-8, "gradient_clip": 1.0}}
    if (any(protocol.get(key) != value for key, value in expected_protocol.items())
            or list(fold.training_donors) not in protocol.get("training_subsets", [])
            or contract.get("metadata_index_sha256") != protocol.get("metadata_index_identity")):
        raise ValueError("Observation protocol/feature admission differs")
    for key, value in {"seed": 0, "expected_fraction": 0.05, "exclude_from_optimizer": True,
                       "exclude_from_center_scale": True, "biological_independence_claimed": False}.items():
        if protocol.get("technical_holdout", {}).get(key) != value:
            raise ValueError("Observation technical holdout protocol differs")
    reference = {"relative_uri": "SELECTED_OBSERVATION.pt", "sha256": admission.artifact.sha256,
                 "size_bytes": admission.artifact.size_bytes}
    if _json(admission.receipt) != reference:
        raise ValueError("Selected artifact sidecar differs from the exact admitted bytes")
    completion = _json(admission.completion)
    expected_completion = {"schema": "credo.frontier.observation-worker-completion.v1", "status": "complete",
        "identity": identity, "scope": "synthetic_cpu_only" if spec.synthetic_fixture else "scientific",
        "accepted_updates": spec.maximum_updates, "attempted_updates": spec.maximum_updates,
        "scientific_updates": 0 if spec.synthetic_fixture else spec.maximum_updates,
        "main_dynamics_fitted": False, "held_out_donor_outcomes_opened": False}
    if any(completion.get(key) != value for key, value in expected_completion.items()):
        raise ValueError("Complete observation worker identity/scope differs")
    if not spec.synthetic_fixture and completion.get("actual_disposable_optimizer_calls") != 0:
        raise ValueError("Scientific observation completion contains an unexpected disposable workload")
    selected = completion.get("selected_export", {})
    if (selected.get("publication") != reference or selected.get("completed_updates") != spec.maximum_updates
            or selected.get("selected_updates") not in spec.audit_updates
            or selected.get("optimizer_resume") is not False
            or selected.get("technical_holdout_excluded_from_scaling") is not True):
        raise ValueError("Artifact is not the COMPLETE-selected, scaling-qualified inference export")
    if not spec.synthetic_fixture and (not completion.get("input_lease") or not completion.get("input_exit")):
        raise ValueError("Scientific observation completion lacks admitted input exit evidence")
    return completion, selected


def load_observation(admission):
    """Return ``(frozen_cpu_SparseR48, actual_ObservationBinding)``.

    This accepts only the exact COMPLETE-selected artifact. Candidate history
    is checked for provenance; risk values are never used to select new weights.
    The downstream qualified owner may move the returned frozen module to its
    admitted device; no device movement other than CPU loading occurs here.
    """
    if not isinstance(admission, ObservationArtifactAdmission):
        raise TypeError("Explicit observation artifact admission required")
    # Revalidate a mutable caller-supplied identity before opening any artifact.
    admission.__post_init__()
    completion, selected = _metadata(admission)
    with _verified_file(admission.artifact, maximum=MAXIMUM_ARTIFACT_BYTES) as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    spec = admission.specification
    keys = {"schema", "inference_only", "identity", "selected_candidate", "scaling", "completed_updates", "model_state"}
    if (not isinstance(payload, dict) or set(payload) != keys
            or payload["schema"] != "credo.frontier.selected-observation.v1" or payload["inference_only"] is not True
            or payload["identity"] != admission.trajectory_identity or payload["completed_updates"] != spec.maximum_updates
            or payload["scaling"] != selected["scaling"]):
        raise ValueError("Selected export payload is historical, incomplete or differently bound")
    candidate = payload["selected_candidate"]
    if (not isinstance(candidate, dict) or candidate.get("accepted_updates") != selected["selected_updates"]
            or sum(item == candidate for item in completion.get("audit_history", [])) != 1):
        raise ValueError("Selected candidate does not match retained COMPLETE audit identity")
    scaling = payload["scaling"]
    if (not isinstance(scaling, dict) or scaling.get("fold_id") != admission.fold.fold_id
            or type(scaling.get("positive_RNA_cells")) is not int or scaling["positive_RNA_cells"] < 2
            or scaling.get("population_variance_ddof") != 0
            or scaling.get("scale_floor_rule") != "max(1e-6,fraction*median_positive_fitting_std)"
            or not isinstance(scaling.get("scale_floor"), (int, float))
            or not math.isfinite(scaling["scale_floor"]) or scaling["scale_floor"] < 1e-6):
        raise ValueError("Selected fitting-only scaling metadata differs")
    model = SparseR48(admission.fold, features=spec.features, hidden_dims=spec.hidden_dims,
        latent_dim=spec.latent_dim, maximum_rows=spec.logical_batch,
        maximum_dense_bytes=spec.maximum_dense_bytes, seed=0)
    expected, actual = model.state_dict(), payload["model_state"]
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("Observation state inventory differs")
    for name, reference in expected.items():
        value = actual[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != reference.dtype
                or value.shape != reference.shape or value.layout != torch.strided
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Observation state shape/dtype/finite-value mismatch: " + name)
    model.load_state_dict(actual, strict=True)
    model.freeze()
    model._check_scaling()
    # fit_scaling writes FP32 buffers, so compare to the FP32 representation of
    # its recorded FP64 floor rather than incorrectly rejecting rounding down.
    floor = torch.tensor(scaling["scale_floor"], dtype=model.scale.dtype)
    if bool((model.scale < floor).any()):
        raise ValueError("Selected scaler falls below its recorded fitting floor")
    binding = ObservationBinding(admission.artifact.sha256, observation_state_digest(model))
    return model, binding
