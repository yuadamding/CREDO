"""Fabricated main-stage files only; no training/optimizer steps or real data.

Accepted-update counters below are fabricated receipt metadata. The producer's
save_checkpoint writer is used without optimizer.step to test the real format.
"""

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from frontier import configuration as configuration_implementation, models as model_implementation
from frontier.checkpoints import TrajectoryLease, save_checkpoint
from frontier.fitting_factory import prepare_fitting_components
from frontier.forecast import model_state_digest
from frontier.forecast_loader import ForecastArtifactAdmission, ForecastFile, load_forecast_model
from frontier.training_owner import SCHEMA as OWNER_SCHEMA
from test_fitting_factory import package as fitting_package


def reference(path):
    data = path.read_bytes()
    return ForecastFile(str(path), hashlib.sha256(data).hexdigest(), len(data))


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return reference(path)


@pytest.fixture
def checkpoint_package(fitting_package, tmp_path, request):
    args, _ = fitting_package
    family = getattr(request, "param", "F3")
    cap = 64 * 1024**2
    if isinstance(family, tuple):
        family, cap = family
    prepared = prepare_fitting_components(**{**args, "configuration_id": family + "-V1",
                                            "maximum_checkpoint_bytes": cap})
    model = prepared.arguments["model"]
    model.set_phase("joint")
    model.train()
    # Real tensor format, synthetic completion metadata, zero actual updates.
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
    root = tmp_path / "credo_performance_frontier_20261015_v1" / "fabricated-main"
    root.parent.mkdir(exist_ok=True)
    u = 4
    provider_state = {**prepared.arguments["provider"].state_dict(), "position": u}
    sampler = dict(position=u, provider_state=provider_state, binding=prepared.identity["training_owner"],
                   joint_entries=1, refresh_updates=[1, 3])
    with TrajectoryLease(root, prepared.identity, "fabricated-loader-producer") as lease:
        publication = save_checkpoint(lease, model=model, optimizer=optimizer, scheduler=None,
            accepted_updates=u, attempted_updates=u, phase="joint", sampler_state=sampler,
            count_cache=prepared.arguments["cache"].state_dict(), execution_scope="synthetic_cpu_only", milestone=True,
            maximum_checkpoint_bytes=cap)
    status = dict(schema=OWNER_SCHEMA, identity=prepared.identity, status="stage_complete",
        accepted_updates=u, attempted_updates=u, optimizer_calls=u, healthy_optimizer_updates=u,
        phase="joint", joint_entries=1, refresh_updates=[1, 3], last_checkpoint=publication,
        last_metrics={"update": u}, failure=None, automatic_retry=False, scientific_updates=0)
    status_ref = write_json(root / "TRAINING_OWNER_STATE.json", status)
    directory = Path(publication["directory"])
    admission = ForecastArtifactAdmission(reference(directory / "checkpoint.pt"), reference(directory / "COMPLETE.json"),
        status_ref, prepared.identity, args["cache"].manifest.fold, args["cache"].manifest.observation_binding,
        prepared.support_factors, model.config.configuration_id, 0, u, args["schedule"], 64*1024**2, 4*1024**2)
    return admission, model


def changed_payload(admission, change):
    path = Path(admission.checkpoint.path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    change(payload)
    torch.save(payload, path)
    checkpoint = reference(path)
    receipt_path = Path(admission.completion.path)
    receipt = json.loads(receipt_path.read_text())
    receipt.update(checkpoint_sha256=checkpoint.sha256, checkpoint_bytes=checkpoint.size_bytes)
    receipt_ref = write_json(receipt_path, receipt)
    status_path = Path(admission.owner_status.path)
    status = json.loads(status_path.read_text())
    status["last_checkpoint"] = {"directory": str(path.parent), **receipt}
    return replace(admission, checkpoint=checkpoint, completion=receipt_ref,
                   owner_status=write_json(status_path, status))


@pytest.mark.parametrize("checkpoint_package", ["F0", "F1", "F2", "F3", "F4", "F5", ("F0", None)], indirect=True)
def test_actual_main_checkpoint_to_frozen_forecast_without_optimizer_or_rng_restore(checkpoint_package, monkeypatch):
    admission, expected = checkpoint_package
    original = {k: v.clone() for k, v in expected.state_dict().items()}
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    monkeypatch.setattr(torch.optim.Adam, "__init__", lambda *_a, **_k: pytest.fail("No optimizer construction"))
    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", lambda *_a, **_k: pytest.fail("No optimizer restoration"))
    model, binding = load_forecast_model(admission)
    assert not model.training and model.phase == "joint" and model.reaction_enabled
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert binding.model_state_sha256 == model_state_digest(model)
    assert binding.model_artifact_sha256 == admission.checkpoint.sha256
    assert binding.target_support_sha256 == admission.target_support.identity
    assert binding.numerical_source_sha256 == admission.trajectory_identity["source_sha256"]
    assert binding.observation_binding == admission.observation_binding
    assert binding.accepted_updates == 4 and binding.model_seed == 0
    assert random.getstate() == python and torch.equal(torch.get_rng_state(), cpu)
    after = np.random.get_state()
    assert after[0] == numpy[0] and np.array_equal(after[1], numpy[1]) and after[2:] == numpy[2:]


@pytest.mark.parametrize("field,value", [("status", "failed"), ("status", "ready"),
    ("accepted_updates", 2), ("attempted_updates", 5), ("healthy_optimizer_updates", 3),
    ("failure", {"message": "failed"}), ("refresh_updates", [1])])
def test_owner_must_be_exact_current_healthy_completed_stage(checkpoint_package, monkeypatch, field, value):
    admission, _ = checkpoint_package
    path = Path(admission.owner_status.path)
    status = json.loads(path.read_text()); status[field] = value
    admission = replace(admission, owner_status=write_json(path, status))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("Must reject before deserialization"))
    with pytest.raises(ValueError, match="Current owner"):
        load_forecast_model(admission)


@pytest.mark.parametrize("field,value", [("model_seed", 1), ("configuration_id", "F3-V2"),
    ("accepted_updates", 3), ("maximum_checkpoint_bytes", 1)])
def test_explicit_admission_mismatch_or_budget_rejected(checkpoint_package, field, value):
    admission, _ = checkpoint_package
    with pytest.raises(ValueError):
        replace(admission, **{field: value})


@pytest.mark.parametrize("cap", [False, 0, -1, 1.5, "67108864", 1])
def test_producer_cap_must_be_valid_and_cover_actual_archive(checkpoint_package, cap):
    admission, _ = checkpoint_package
    identity = copy.deepcopy(admission.trajectory_identity)
    identity["training_owner"]["maximum_checkpoint_bytes"] = cap
    with pytest.raises(ValueError, match="checkpoint write cap"):
        replace(admission, trajectory_identity=identity)


@pytest.mark.parametrize("change", ["missing", "changed", "null", "boolean"])
def test_completion_must_retain_exact_bound_producer_cap(checkpoint_package, monkeypatch, change):
    admission, _ = checkpoint_package
    path = Path(admission.completion.path)
    receipt = json.loads(path.read_text())
    if change == "missing":
        del receipt["maximum_checkpoint_bytes"]
    else:
        receipt["maximum_checkpoint_bytes"] = {"changed": 128*1024**2, "null": None, "boolean": True}[change]
    admission = replace(admission, completion=write_json(path, receipt))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("Cap rejection must precede tensor loading"))
    with pytest.raises(ValueError, match="completion"):
        load_forecast_model(admission)


def test_consumer_cap_can_be_tighter_than_producer_without_changing_identity(checkpoint_package):
    admission, expected = checkpoint_package
    smaller = replace(admission, maximum_checkpoint_bytes=admission.checkpoint.size_bytes)
    assert smaller.maximum_checkpoint_bytes < smaller.trajectory_identity["training_owner"]["maximum_checkpoint_bytes"]
    model, _ = load_forecast_model(smaller)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in expected.state_dict().items())
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["phase", "identity", "constructor", "support", "dtype", "shape", "nonfinite", "gradient", "sampler", "resource_policy"])
def test_rehashed_payload_cannot_substitute_main_model_identity(checkpoint_package, kind):
    admission, _ = checkpoint_package
    def change(payload):
        name = next(iter(payload["model_state"]))
        if kind == "phase": payload["phase"] = "state"
        elif kind == "identity": payload["identity"]["seed"] = 1
        elif kind == "constructor": payload["model_constructor_metadata"]["ordered_target_ids"] = ["other-target"]
        elif kind == "support": payload["model_state"]["support_factors"].add_(.01)
        elif kind == "dtype": payload["model_state"][name] = payload["model_state"][name].double()
        elif kind == "shape": payload["model_state"][name] = payload["model_state"][name].reshape(-1)[:1]
        elif kind == "nonfinite": payload["model_state"][name].reshape(-1)[0] = float("nan")
        elif kind == "gradient": payload["requires_grad_mask"][next(iter(payload["requires_grad_mask"]))] = False
        elif kind == "sampler": payload["sampler_state"]["provider_state"]["position"] = 3
        elif kind == "resource_policy": payload["sampler_state"]["resource_policy_sha256"] = "b"*64
    admission = changed_payload(admission, change)
    with pytest.raises(ValueError):
        load_forecast_model(admission)


def test_exact_file_hash_checked_before_restricted_load(checkpoint_package, monkeypatch):
    admission, _ = checkpoint_package
    path = Path(admission.checkpoint.path)
    content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("Hash verification must precede load"))
    with pytest.raises(ValueError, match="SHA-256"):
        load_forecast_model(admission)


def test_same_descriptor_restricted_load_and_changed_journal_rejected(checkpoint_package, monkeypatch):
    admission, _ = checkpoint_package
    original = torch.load
    seen = []
    def checked(stream, **kwargs):
        assert hasattr(stream, "fileno") and not isinstance(stream, (str, Path))
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        seen.append(stream.fileno())
        result = original(stream, **kwargs)
        path = Path(admission.owner_status.path)
        # Keep readable bytes but change the verified file during deserialization.
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(torch, "load", checked)
    with pytest.raises(ValueError, match="changed during loading"):
        load_forecast_model(admission)
    assert len(seen) == 1


def test_nonmilestone_completion_does_not_become_forecast(checkpoint_package, monkeypatch):
    admission, _ = checkpoint_package
    path = Path(admission.completion.path)
    receipt = json.loads(path.read_text()); receipt["milestone"] = False
    admission = replace(admission, completion=write_json(path, receipt))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("No deserialization"))
    with pytest.raises(ValueError, match="completion"):
        load_forecast_model(admission)


@pytest.mark.parametrize("kind", ["missing", "schema", "producer", "missing_file", "extra_file",
                                  "extra_field", "file_fields", "size"])
def test_producer_record_required_without_inventing_source_provenance(checkpoint_package, kind):
    admission, _ = checkpoint_package
    identity = copy.deepcopy(admission.trajectory_identity)
    sources = identity["forecast_model_sources"]
    if kind == "missing": del identity["forecast_model_sources"]
    elif kind == "schema": sources["schema"] = "other"
    elif kind == "producer": sources["producer_source_sha256"] = "0" * 64
    elif kind == "missing_file": del sources["files"]["frontier/models.py"]
    elif kind == "extra_file": sources["files"]["frontier/forecast.py"] = sources["files"]["frontier/models.py"]
    elif kind == "extra_field": sources["current_inference_mapping"] = True
    elif kind == "file_fields": sources["files"]["frontier/models.py"]["path"] = "/ignored"
    elif kind == "size": sources["files"]["frontier/models.py"]["size_bytes"] = True
    with pytest.raises(ValueError, match="identity required|source"):
        replace(admission, trajectory_identity=identity)


@pytest.mark.parametrize("module", [model_implementation, configuration_implementation])
@pytest.mark.parametrize("change", ["bytes", "size"])
def test_changed_model_equations_fail_before_checkpoint_deserialization(
        checkpoint_package, monkeypatch, tmp_path, module, change):
    admission, _ = checkpoint_package
    original = Path(module.__file__).read_bytes()
    candidate = tmp_path / Path(module.__file__).name
    if change == "bytes":
        changed = bytearray(original); changed[-1] ^= 1
        candidate.write_bytes(changed)
    else:
        candidate.write_bytes(original + b"\n")
    # Change only a fabricated current-source file; never edit the implementation.
    monkeypatch.setattr(module, "__file__", str(candidate))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: pytest.fail("Source guard must precede tensor load"))
    with pytest.raises(ValueError, match="SHA-256|file size"):
        load_forecast_model(admission)


@pytest.mark.parametrize("module", [model_implementation, configuration_implementation])
def test_model_sources_rechecked_on_same_descriptor_before_return(
        checkpoint_package, monkeypatch, tmp_path, module):
    admission, _ = checkpoint_package
    candidate = tmp_path / Path(module.__file__).name
    original_bytes = Path(module.__file__).read_bytes()
    candidate.write_bytes(original_bytes)
    monkeypatch.setattr(module, "__file__", str(candidate))
    original_load = torch.load
    calls = []
    def changed_source(stream, **kwargs):
        calls.append(1)
        result = original_load(stream, **kwargs)
        modified = bytearray(original_bytes); modified[-1] ^= 1
        candidate.write_bytes(modified)
        return result
    monkeypatch.setattr(torch, "load", changed_source)
    with pytest.raises(ValueError, match="source SHA-256 changed during loading"):
        load_forecast_model(admission)
    assert len(calls) == 1
