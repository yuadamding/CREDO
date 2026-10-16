"""Tiny fabricated exports only: zero optimizer calls, GPU, remote or real data."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from frontier.observation import ObservationFold, SparseR48
from frontier.observation_loader import (ObservationArtifactAdmission, ObservationFile, load_observation)
from frontier.observation_training import ObservationTrainingSpec, _publish_tensor
from frontier.training_owner import observation_state_digest


ROOT = Path(__file__).resolve().parents[1]


def reference(path):
    raw = path.read_bytes()
    return ObservationFile(str(path), hashlib.sha256(raw).hexdigest(), len(raw))


def json_file(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return reference(path)


def fixture(tmp_path, *, payload_change=None, contract_change=None, protocol_change=None, completion_change=None,
            specification=None, observation_fold=None, input_sha256="b" * 64):
    # Every scalar "completed update" below is fabricated receipt content, not
    # a claim that fitting ran. No optimizer is instantiated or stepped.
    root = tmp_path / "credo_performance_frontier_20261015_v1" / "fabricated-output"
    root.mkdir(parents=True)
    spec = ObservationTrainingSpec.fixture() if specification is None else specification
    fold = ObservationFold("inner-D3-D4", ("D3", "D4"), "f" * 64) if observation_fold is None else observation_fold
    protocol = json.loads((ROOT / "OBSERVATION_PROTOCOL.json").read_text())
    protocol.update(RNA_feature_axis_identity=fold.feature_axis_sha256, RNA_feature_count=spec.features,
        architecture_dimensions=[spec.features, *spec.hidden_dims, spec.latent_dim],
        maximum_accepted_updates_each=spec.maximum_updates, logical_batch_positive_training_cells=spec.logical_batch,
        audit_candidates_accepted_updates=list(spec.audit_updates), checkpoint_interval_accepted_updates=spec.checkpoint_interval,
        maximum_dense_working_estimate_bytes=spec.maximum_dense_bytes)
    if protocol_change:
        protocol_change(protocol)
    protocol_ref = json_file(root / "OBSERVATION_PROTOCOL.json", protocol)
    contract = {"schema": "credo.frontier.observation-worker-contract.v1", "mode": "fit",
        "output_root": str(root), "trajectory_id": "fabricated-observation", "fold_id": fold.fold_id,
        "training_donors": list(fold.training_donors), "rna_axis_sha256": fold.feature_axis_sha256,
        "input_access_sha256": input_sha256, "source_manifest_sha256": "a" * 64,
        "protocol_path": protocol_ref.path, "protocol_sha256": protocol_ref.sha256,
        "metadata_index_sha256": protocol["metadata_index_identity"]}
    if contract_change:
        contract_change(contract)
    contract_ref = json_file(root / "WORKER_CONTRACT.json", contract)
    identity = {"trajectory_id": "fabricated-observation", "fold_id": fold.fold_id, "seed": 0,
        "input_sha256": input_sha256, "configuration_sha256": spec.identity, "source_sha256": "a" * 64,
        "worker_contract_sha256": contract_ref.sha256}
    model = SparseR48(fold, features=spec.features, hidden_dims=spec.hidden_dims, latent_dim=spec.latent_dim,
        maximum_rows=spec.logical_batch, maximum_dense_bytes=spec.maximum_dense_bytes, seed=0)
    with torch.no_grad():
        model.center.copy_(torch.arange(spec.latent_dim) / 10)
        model.scale.copy_(torch.arange(spec.latent_dim) / 10 + 0.5)
        model.scaling_ready.fill_(True)
    model.freeze()
    scaling = {"positive_RNA_cells": 12, "population_variance_ddof": 0, "scale_floor": 0.01,
               "fold_id": fold.fold_id, "scale_floor_rule": "max(1e-6,fraction*median_positive_fitting_std)"}
    candidate = {"accepted_updates": 2, "risk": 1.0, "positive_cell_directions": 4,
                 "sum_normalized_cross_entropy": 4.0, "positive_RNA_cells": 2, "thinning_seed": 1,
                 "independent_donors": False, "artifact": {"relative_uri": "observation-candidate-000000002.pt",
                     "sha256": "d" * 64, "size_bytes": 128}}
    payload = {"schema": "credo.frontier.selected-observation.v1", "inference_only": True,
        "identity": identity, "selected_candidate": candidate, "scaling": scaling,
        "completed_updates": spec.maximum_updates,
        "model_state": {name: value.clone() for name, value in model.state_dict().items()}}
    if payload_change:
        payload_change(payload)
    publication = _publish_tensor(root, "SELECTED_OBSERVATION.pt", payload)
    complete = {"schema": "credo.frontier.observation-worker-completion.v1", "status": "complete",
        "identity": identity, "scope": "synthetic_cpu_only", "accepted_updates": spec.maximum_updates,
        "attempted_updates": spec.maximum_updates, "scientific_updates": 0,
        "main_dynamics_fitted": False, "held_out_donor_outcomes_opened": False,
        "selected_export": {"publication": publication, "scaling": scaling, "selected_updates": 2,
            "completed_updates": spec.maximum_updates, "optimizer_resume": False,
            "technical_holdout_excluded_from_scaling": True}, "audit_history": [candidate]}
    if completion_change:
        completion_change(complete)
    complete_ref = json_file(root / "COMPLETE.json", complete)
    admission = ObservationArtifactAdmission(reference(root / "SELECTED_OBSERVATION.pt"),
        reference(root / "SELECTED_OBSERVATION.pt.json"), complete_ref, contract_ref, protocol_ref, fold, spec, identity)
    return admission, model


def test_verified_selected_export_returns_frozen_actual_state_binding(tmp_path):
    admission, expected = fixture(tmp_path)
    loaded, binding = load_observation(admission)
    assert loaded.device.type == "cpu" and not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert binding.artifact_sha256 == admission.artifact.sha256
    assert binding.state_sha256 == observation_state_digest(loaded) == observation_state_digest(expected)
    assert loaded.fold == admission.fold and bool(loaded.scaling_ready)
    for name, value in expected.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    # Frozen decoder parameters still permit gradients to a downstream input.
    state = torch.zeros((2, admission.specification.latent_dim), requires_grad=True)
    probabilities = loaded.probabilities(state)
    (probabilities * torch.arange(admission.specification.features)).sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(parameter.grad is None for parameter in loaded.parameters())


def test_hash_and_size_failure_precedes_torch_deserialization(tmp_path, monkeypatch):
    admission, _ = fixture(tmp_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not run"))
    path = Path(admission.artifact.path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch before"):
        load_observation(admission)


@pytest.mark.parametrize("change", [
    lambda value: value.update(training_donors=["D1", "D2"]),
    lambda value: value.update(rna_axis_sha256="e" * 64),
    lambda value: value.update(input_access_sha256="e" * 64),
    lambda value: value.update(source_manifest_sha256="e" * 64),
])
def test_contract_provenance_rejected_before_deserialization(tmp_path, monkeypatch, change):
    admission, _ = fixture(tmp_path, contract_change=change)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not run"))
    with pytest.raises(ValueError, match="exact fold, feature axis"):
        load_observation(admission)


def test_different_protocol_and_complete_selection_are_rejected(tmp_path, monkeypatch):
    admission, _ = fixture(tmp_path / "protocol", protocol_change=lambda value: value.update(fitting_conditions=["Rest", "Stim48hr"]))
    other, _ = fixture(tmp_path / "selection", completion_change=lambda value: value["selected_export"]["publication"].update(sha256="e" * 64))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not run"))
    with pytest.raises(ValueError, match="protocol/feature"):
        load_observation(admission)
    with pytest.raises(ValueError, match="COMPLETE-selected"):
        load_observation(other)


@pytest.mark.parametrize("change,match", [
    (lambda value: value.update(schema="historical.checkpoint"), "historical"),
    (lambda value: value["model_state"]["scale"].fill_(0), "positive scale"),
    (lambda value: value["model_state"]["scaling_ready"].fill_(False), "finalized"),
    (lambda value: value["model_state"].update(center=value["model_state"]["center"].double()), "shape/dtype"),
    (lambda value: value["model_state"]["first.weight"].fill_(float("nan")), "finite-value"),
])
def test_historical_payload_or_invalid_selected_state_cannot_load(tmp_path, change, match):
    admission, _ = fixture(tmp_path, payload_change=change)
    with pytest.raises(ValueError, match=match):
        load_observation(admission)


def test_exact_sidecar_and_regular_nonsymlink_file_required(tmp_path, monkeypatch):
    admission, _ = fixture(tmp_path)
    sidecar = json.loads(Path(admission.receipt.path).read_text())
    sidecar["size_bytes"] += 1
    changed = replace(admission, receipt=json_file(Path(admission.receipt.path), sidecar))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not run"))
    with pytest.raises(ValueError, match="sidecar"):
        load_observation(changed)
    original = Path(admission.artifact.path)
    copy = original.with_name("fixture-copy.pt")
    original.rename(copy)
    original.symlink_to(copy)
    with pytest.raises(ValueError, match="nonsymlink"):
        load_observation(admission)


def test_restricted_load_same_descriptor_and_post_read_change_rejected(tmp_path, monkeypatch):
    admission, _ = fixture(tmp_path)
    real_load = torch.load
    calls = []

    def modify_after_load(stream, **kwargs):
        calls.append(kwargs)
        assert not isinstance(stream, (str, Path)) and stream.tell() == 0
        result = real_load(stream, **kwargs)
        path = Path(admission.artifact.path)
        with path.open("ab") as writer:
            writer.write(b"fixture-change")
        return result

    monkeypatch.setattr(torch, "load", modify_after_load)
    with pytest.raises(ValueError, match="changed during loading"):
        load_observation(admission)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_retained_failure_receipt_rejects_before_torch_load(tmp_path, monkeypatch):
    admission, _ = fixture(tmp_path)
    Path(admission.artifact.path).with_name("WORKER_FAILED.json").write_text("{}")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("torch.load must not run"))
    with pytest.raises(ValueError, match="Failed observation"):
        load_observation(admission)
