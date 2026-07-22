"""Versioned checkpoint envelopes shared by native and imported recipes."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and values with one canonical encoding."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


class CheckpointMode(StrEnum):
    INFERENCE_ONLY = "inference_only"
    RESUME_CAPABLE = "resume_capable"
    TRAINING_RECIPE_ONLY = "training_recipe_only"


@dataclass(frozen=True)
class CheckpointEnvelope:
    recipe: Mapping[str, Any]
    study_contract: Mapping[str, Any]
    representation_contract: Mapping[str, Any]
    split_contract: Mapping[str, Any]
    state: Mapping[str, Any]
    training: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    mode: CheckpointMode
    import_provenance: Mapping[str, Any] | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CheckpointMode(self.mode))
        if self.schema_version != 2:
            raise ValueError("CheckpointEnvelope schema_version must be 2.")
        mappings = {
            "recipe": self.recipe,
            "study_contract": self.study_contract,
            "representation_contract": self.representation_contract,
            "split_contract": self.split_contract,
            "state": self.state,
            "training": self.training,
            "capabilities": self.capabilities,
        }
        invalid = [name for name, value in mappings.items() if not isinstance(value, Mapping)]
        if invalid:
            raise ValueError(f"Checkpoint envelope fields must be mappings: {invalid}")
        if self.import_provenance is not None and not isinstance(self.import_provenance, Mapping):
            raise ValueError("Checkpoint envelope import_provenance must be a mapping or null.")
        recipe_id = str(self.recipe.get("id", ""))
        recipe_version = str(self.recipe.get("version", ""))
        if not recipe_id or not recipe_version:
            raise ValueError("Checkpoint envelope requires recipe id and version.")
        implementation_hash = str(self.recipe.get("implementation_hash", "")).lower()
        if len(implementation_hash) != 64 or any(
            value not in "0123456789abcdef" for value in implementation_hash
        ):
            raise ValueError("Checkpoint recipe requires a SHA-256 implementation_hash.")
        if self.mode is not CheckpointMode.TRAINING_RECIPE_ONLY and not self.state.get("model"):
            raise ValueError("Inference checkpoints require model state provenance.")
        if self.mode is CheckpointMode.RESUME_CAPABLE:
            missing = [
                name
                for name in ("model", "optimizer", "scheduler", "rng")
                if not self.state.get(name)
            ]
            if missing:
                raise ValueError(f"Resume-capable checkpoint is missing state: {missing}")
            if not bool(self.capabilities.get("checkpoint_resume_supported")):
                raise ValueError("A resume-capable checkpoint requires recipe resume capability.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe": copy.deepcopy(dict(self.recipe)),
            "study_contract": copy.deepcopy(dict(self.study_contract)),
            "representation_contract": copy.deepcopy(dict(self.representation_contract)),
            "split_contract": copy.deepcopy(dict(self.split_contract)),
            "state": copy.deepcopy(dict(self.state)),
            "training": copy.deepcopy(dict(self.training)),
            "capabilities": copy.deepcopy(dict(self.capabilities)),
            "mode": self.mode.value,
            "import_provenance": (
                None
                if self.import_provenance is None
                else copy.deepcopy(dict(self.import_provenance))
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointEnvelope:
        allowed = {
            "schema_version",
            "recipe",
            "study_contract",
            "representation_contract",
            "split_contract",
            "state",
            "training",
            "capabilities",
            "mode",
            "import_provenance",
        }
        required = allowed - {"import_provenance"}
        unknown = set(payload) - allowed
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                "Checkpoint envelope has invalid fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        return cls(
            schema_version=int(payload.get("schema_version", -1)),
            recipe=payload["recipe"],
            study_contract=payload["study_contract"],
            representation_contract=payload["representation_contract"],
            split_contract=payload["split_contract"],
            state=payload["state"],
            training=payload["training"],
            capabilities=payload["capabilities"],
            mode=CheckpointMode(payload["mode"]),
            import_provenance=payload.get("import_provenance"),
        )

    def require_resume(self) -> None:
        if self.mode is not CheckpointMode.RESUME_CAPABLE:
            raise RuntimeError(f"Checkpoint mode {self.mode.value!r} cannot resume training.")


class NativeCheckpointCodec:
    """Minimal codec for already canonical schema-v2 checkpoints."""

    def encode(self, **parts: Any) -> Mapping[str, Any]:
        return CheckpointEnvelope(**parts).to_dict()

    def decode(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return CheckpointEnvelope.from_dict(payload).to_dict()
