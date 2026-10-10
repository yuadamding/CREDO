"""Load only a receipt-bound terminal dynamics model, never a Trainer.

There is no optimizer, catalog refresh, source/endpoint reader or evaluation
call here. CPU execution is useful for synthetic tests; production admission
and the post-prediction lease are enforced separately by the phase worker.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import stat

import torch

from credo.artifacts import CheckpointEnvelope, tensor_state_sha256
from credo.recipes.compact_sde_v3.model import CREDOModel
from prepare_worker import file_identity


def load_frozen_model(path, binding, handoff, *, device="cuda", expected_owner=307469):
    path = Path(path)
    expected = {key: binding[key] for key in ("sha256", "size_bytes")}
    if expected["size_bytes"] > 64 * 1024**2 or file_identity(path, owner=expected_owner) != expected:
        raise ValueError("Exact bounded terminal checkpoint identity is required")
    # Load from the bytes just validated rather than reopening a mutable path
    # inside torch.load. Revalidate those bytes and the path before returning.
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Regular checkpoint required")
        raw = handle.read(expected["size_bytes"] + 1)
    import hashlib
    if len(raw) != expected["size_bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("Checkpoint changed before deserialization")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (payload.get("schema_version") != 2
            or payload.get("completed_epochs") != binding["completed_epochs"]
            or payload.get("compiled_problem_hash") != binding["compiled_problem_hash"]
            or list(payload.get("train_measure_ids", [])) != handoff["expected_train_measure_ids"]
            or list(payload.get("validation_measure_ids", [])) != handoff["query_measure_ids"]
            or list(payload.get("train_time_labels", [])) != ["Stim48hr"]
            or list(payload.get("validation_time_labels", [])) != ["Stim48hr"]
            or payload.get("training_plan", {}).get("seed") != binding["seed"]):
        raise ValueError("Checkpoint phase, seed, terminal exposure or population identity differs")
    envelope = CheckpointEnvelope.from_dict(payload["envelope"])
    if (envelope.mode.value != "inference_only" or envelope.recipe.get("id") != "credo.compact_sde_v3"
            or envelope.recipe.get("version") != "3.0"
            or envelope.study_contract != payload.get("run_contract")):
        raise ValueError("Checkpoint inference envelope differs")
    for name in ("training_plan", "objective_descriptors", "execution_trace"):
        if envelope.training.get(name) != payload.get(name):
            raise ValueError("Checkpoint training provenance differs from its envelope")
    for field, name in (("train_measure_ids", "expected_train_measure_ids"),
                        ("validation_measure_ids", "query_measure_ids")):
        if list(envelope.split_contract.get(field, [])) != handoff[name]:
            raise ValueError("Checkpoint split envelope differs")
    state = payload["model_state"]
    model_contract = envelope.state["model"]
    if (model_contract.get("tensor_count") != len(state)
            or model_contract.get("semantic_hash") != tensor_state_sha256(state)
            or any(not torch.is_tensor(t) or t.dtype != torch.float32
                   or not bool(t.isfinite().all()) for t in state.values())):
        raise ValueError("Nonfinite, changed or non-FP32 model state")
    objective = {"log_count_concentration": payload["log_count_concentration"]}
    if envelope.state["objective"].get("semantic_hash") != tensor_state_sha256(objective):
        raise ValueError("Objective state provenance differs")
    architecture = payload["architecture"]
    if (sorted(architecture["embedding_ids"]) != handoff["expected_embedding_ids"]
            or architecture["control_embedding_ids"] != ["shared-NTC-reference"]
            or architecture["context_mode"] != "none"):
        raise ValueError("Fitted target/control identity or no-context policy differs")
    for key, expected_value in binding["architecture"].items():
        actual = architecture.get(key, {"trunk_kind": "mlp", "trunk_depth": 2}.get(key))
        if actual != expected_value:
            raise ValueError("Candidate architecture was relabeled: " + key)
    # Preserve caller RNG state: random constructor initialization is immediately
    # replaced by every exact saved tensor and is not a new fitted start.
    with torch.random.fork_rng(devices=[]):
        model = CREDOModel(**architecture)
    model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != binding["parameter_count"]:
        raise ValueError("Frozen parameter count differs")
    if tensor_state_sha256(model.state_dict()) != model_contract["semantic_hash"]:
        raise ValueError("Model reconstruction changed terminal parameters")
    model.set_phase("mass")
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    model.assert_soft_reference()
    if file_identity(path, owner=expected_owner) != expected:
        raise ValueError("Checkpoint path changed during model loading")
    return model, dict(checkpoint_sha256=binding["sha256"],
        model_semantic_sha256=model_contract["semantic_hash"], parameter_count=binding["parameter_count"],
        candidate_id=binding["candidate_id"], seed=binding["seed"],
        parameter_updates=0, optimizer_constructed=False, catalog_refresh_performed=False,
        endpoint_observations_read=False, evaluation_performed=False)
