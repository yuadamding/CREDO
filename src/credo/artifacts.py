"""Versioned checkpoint envelopes shared by native and imported recipes."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_contract_hash(payload: Mapping[str, Any]) -> str:
    contract = {
        key: value for key, value in payload.items() if key not in {"run_id", "run_contract_hash"}
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_run_json(
    directory: str | Path,
    payload: Mapping[str, Any],
    *,
    artifacts: Mapping[str, str | Path],
) -> Path:
    """Finalize one content-addressed generic run manifest."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(dict(payload))
    manifest["format"] = "credo.run"
    manifest["run_schema_version"] = 1
    artifact_payload = {}
    for name, path in sorted(artifacts.items()):
        relative = str(name)
        candidate = (root / relative).resolve()
        source = Path(path).expanduser().resolve()
        if Path(relative).is_absolute() or not candidate.is_relative_to(root):
            raise ValueError(f"Run artifact escapes its root: {relative!r}.")
        if source != candidate:
            raise ValueError(f"Run artifact {relative!r} must be stored inside the run directory.")
        if not source.is_file():
            raise FileNotFoundError(source)
        artifact_payload[relative] = {
            "sha256": _file_sha256(source),
            "size_bytes": source.stat().st_size,
        }
    manifest["artifacts"] = artifact_payload
    digest = _run_contract_hash(manifest)
    manifest.setdefault("run_id", f"sha256:{digest}")
    manifest["run_contract_hash"] = digest
    path = root / "run.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def write_compact_run_json(run: Any) -> Path:
    """Persist the generic wrapper around one compact-v3 runtime."""
    root = Path(run.config.output).expanduser().resolve()
    legacy = run._manifest()
    resolved = legacy["resolved_config"]
    source = {
        "kind": "native_study" if resolved.get("study") is not None else "run_config",
        "uri": resolved.get("study"),
        "study_content_hash": run.data.metadata.get("study_content_hash"),
        "selection_hash": run.data.metadata.get("selection_hash"),
        "compiled_problem_hash": run.data.metadata.get("compiled_problem_hash"),
    }
    payload = {
        **legacy,
        "schema_version": 3,
        "mode": "inference_only",
        "run_id": f"sha256:{run.checkpoint_sha256}",
        "state": {
            "codec": "credo.compact_v3_checkpoint",
            "path": "state/checkpoint.pt",
        },
        "study_binding": source,
        "split_plan": run.data.metadata.get("split_plan"),
        "outputs": {
            "history": "tables/history.parquet",
            "predictions": "tables/predictions.parquet",
            "metrics": "tables/metrics.parquet",
            "diagnostics": "tables/diagnostics.parquet",
            "counterfactuals": "tables/counterfactuals.parquet",
        },
    }
    artifact_paths = {
        name: root / name
        for name in (
            "state/checkpoint.pt",
            "tables/history.parquet",
            "tables/predictions.parquet",
            "tables/metrics.parquet",
            "tables/diagnostics.parquet",
            "tables/counterfactuals.parquet",
        )
    }
    return write_run_json(root, payload, artifacts=artifact_paths)


def write_imported_run_json(directory: str | Path, envelope: CheckpointEnvelope) -> Path:
    """Add a generic, initially unbound manifest to a portable imported bundle."""
    root = Path(directory).expanduser().resolve()
    artifact_paths = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and path.name not in {"run.json"}
    }
    payload = {
        "schema_version": 3,
        "recipe": dict(envelope.recipe),
        "capabilities": dict(envelope.capabilities),
        "mode": envelope.mode.value,
        "state": {"codec": "credo.transformer_v2_bundle", "path": "."},
        "study_binding": None,
        "split_plan": dict(envelope.split_contract),
        "training": dict(envelope.training),
        "study_contract": dict(envelope.study_contract),
        "representation_contract": dict(envelope.representation_contract),
        "import_provenance": (
            None if envelope.import_provenance is None else dict(envelope.import_provenance)
        ),
        "outputs": {},
    }
    return write_run_json(root, payload, artifacts=artifact_paths)


def _validate_imported_study_contract(payload: Mapping[str, Any], compiled: Any) -> None:
    representation = payload.get("representation_contract")
    if representation is not None and compiled.representation.to_dict() != representation:
        raise ValueError("Bound study representation disagrees with the imported run.")

    study_contract = payload.get("study_contract")
    if isinstance(study_contract, Mapping):
        axis = study_contract.get("axis")
        if isinstance(axis, Mapping):
            actual_axis = compiled.axis
            if (
                actual_axis.kind != axis.get("kind")
                or actual_axis.source != axis.get("source")
                or list(actual_axis.labels) != list(axis.get("labels", ()))
                or (
                    axis.get("value_source") != "ordered_checkpoint_fallback"
                    and list(actual_axis.values) != list(axis.get("values", ()))
                )
            ):
                raise ValueError("Bound study axis disagrees with the imported run.")
        expected_mass = study_contract.get("mass_semantics")
        if expected_mass is not None and compiled.mass_semantics.value != expected_mass:
            raise ValueError("Bound study mass semantics disagree with the imported run.")

    provenance = payload.get("import_provenance")
    if not isinstance(provenance, Mapping):
        return
    model_effects = {str(value) for value in provenance.get("perturbation_ids", ())}
    model_controls = {str(value) for value in provenance.get("control_ids", ())}
    actual_effects = set(compiled.embedding_ids)
    actual_controls = set(compiled.control_embedding_ids)
    unknown_effects = actual_effects - model_effects
    invalid_controls = actual_controls - model_controls
    control_role_mismatch = (actual_effects & model_controls) ^ actual_controls
    if unknown_effects or invalid_controls or control_role_mismatch:
        raise ValueError(
            "Bound study effect catalog disagrees with the imported model; "
            f"unknown={sorted(unknown_effects)[:5]}, "
            f"invalid_controls={sorted(invalid_controls)[:5]}, "
            f"role_mismatch={sorted(control_role_mismatch)[:5]}."
        )


def _verified_run_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    target = Path(path).expanduser().resolve()
    manifest_path = target / "run.json" if target.is_dir() else target
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "credo.run" or payload.get("run_schema_version") != 1:
        raise ValueError("Unsupported CREDO run manifest.")
    if _run_contract_hash(payload) != payload.get("run_contract_hash"):
        raise ValueError("Run contract hash mismatch.")
    run_id = str(payload.get("run_id", ""))
    if not run_id.startswith("sha256:") or len(run_id) != 71:
        raise ValueError("Run identifier must be a SHA-256 identity.")
    root = manifest_path.parent
    for relative, artifact in payload.get("artifacts", {}).items():
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Run artifact escapes its root: {relative!r}.")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"Run artifact size mismatch: {relative}")
        if _file_sha256(candidate) != artifact["sha256"]:
            raise ValueError(f"Run artifact hash mismatch: {relative}")
    return payload, root


def _run_state_path(payload: Mapping[str, Any], root: Path) -> Path:
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Run manifest has no state contract.")
    relative = str(state.get("path", ""))
    candidate = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or not candidate.is_relative_to(root):
        raise ValueError(f"Run state path escapes its root: {relative!r}.")
    codec = state.get("codec")
    if codec == "credo.transformer_v2_bundle":
        if candidate != root or not candidate.is_dir():
            raise ValueError("Transformer bundle state path must be the run directory.")
    elif relative not in payload.get("artifacts", {}):
        raise ValueError("Run state is absent from the verified artifact catalog.")
    return candidate


def open_run(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    evaluation_overrides: dict[str, Any] | None = None,
) -> Any:
    """Open any released run through its recipe-owned checkpoint loader."""
    payload, root = _verified_run_manifest(path)
    from .data import open_study
    from .data.splits import (
        SplitPlan,
        validate_representation_scope,
        validate_split_plan,
    )
    from .io import RunConfig
    from .registry import get_recipe
    from .runtime import validate_view_for_recipe

    recipe_contract = payload["recipe"]
    recipe = get_recipe(f"{recipe_contract['id']}@{recipe_contract['version']}")
    state = payload["state"]
    codec = state["codec"]
    state_path = _run_state_path(payload, root)
    if codec == "credo.transformer_v2_bundle":
        compiled = None
        owner = None
        config = None
        if payload.get("study_binding") is not None:
            from .contracts import SplitSpec

            config = RunConfig.model_validate(payload["resolved_config"])
            owner = open_study(config if config.study is None else config.study, verify="semantic")
            try:
                view = config.view(owner)
                binding = payload["study_binding"]
                if owner.content_hash() != binding["study_content_hash"]:
                    raise ValueError("Run study content hash disagrees with the bound study.")
                if view.semantic_hash() != binding["selection_hash"]:
                    raise ValueError("Run selection hash disagrees with the bound StudyView.")
                split = SplitSpec(**dict(payload["split_plan"]))
                validate_view_for_recipe(
                    view, split, recipe.requirements(config.recipe_config)
                ).raise_for_errors()
                compiled = recipe.compile_study(view, split, config.recipe_config)
                _validate_imported_study_contract(payload, compiled)
                if compiled.metadata.get("compiled_problem_hash") != binding.get(
                    "compiled_problem_hash"
                ):
                    raise ValueError("Run compiled problem hash disagrees with the bound study.")
            except Exception:
                owner.close()
                raise
        run = recipe.load_checkpoint(state_path, compiled, config, device=device)
        if owner is not None:
            run._semantic_owner = owner
        run.run_manifest = payload
        return run
    if codec != "credo.compact_v3_checkpoint":
        raise ValueError(f"Unknown run state codec {codec!r}.")

    config = RunConfig.model_validate(payload["resolved_config"])
    source = config if config.study is None else config.study
    owner = open_study(source, verify="semantic")
    try:
        view = config.view(owner)
        binding = payload.get("study_binding") or {}
        if binding.get("study_content_hash") not in {None, owner.content_hash()}:
            raise ValueError("Run study content hash disagrees with the bound study.")
        if binding.get("selection_hash") not in {None, view.semantic_hash()}:
            raise ValueError("Run selection hash disagrees with the bound StudyView.")
        split = (
            SplitPlan.from_dict(payload["split_plan"])
            if payload.get("split_plan") is not None
            else recipe.plan_split(view, config.recipe_config)
        )
        validate_split_plan(view, split)
        validate_representation_scope(view, split)
        validate_view_for_recipe(
            view, split, recipe.requirements(config.recipe_config)
        ).raise_for_errors()
        compiled = recipe.compile_study(view, split, config.recipe_config)
        if binding.get("compiled_problem_hash") not in {
            None,
            compiled.metadata.get("compiled_problem_hash"),
        }:
            raise ValueError("Run compiled problem hash disagrees with the bound study.")
        run = recipe.load_checkpoint(
            state_path,
            compiled,
            config,
            device=device,
            evaluation_overrides=evaluation_overrides,
        )
        run._semantic_owner = owner
        run.run_manifest = payload
        return run
    except Exception:
        owner.close()
        raise


def bind_run_study(run_path: str | Path, config_source: Any) -> Path:
    """Bind an unbound imported run to one canonical semantic Study selection."""
    payload, root = _verified_run_manifest(run_path)
    if payload["state"]["codec"] != "credo.transformer_v2_bundle":
        raise ValueError("Only imported transformer bundles require post-import study binding.")
    from .contracts import SplitSpec
    from .data import open_study
    from .io import RunConfig, load_config
    from .registry import get_recipe
    from .runtime import validate_view_for_recipe

    config = load_config(config_source) if isinstance(config_source, (str, Path)) else config_source
    if not isinstance(config, RunConfig):
        raise TypeError("Run binding requires a CREDO RunConfig or its YAML path.")
    recipe = get_recipe(config.recipe)
    expected = payload["recipe"]
    if (recipe.recipe_id, recipe.recipe_version) != (expected["id"], expected["version"]):
        raise ValueError("Binding config recipe disagrees with the imported run.")
    owner = open_study(config if config.study is None else config.study, verify="semantic")
    try:
        view = config.view(owner)
        split = SplitSpec(**dict(payload["split_plan"]))
        validate_view_for_recipe(
            view, split, recipe.requirements(config.recipe_config)
        ).raise_for_errors()
        compiled = recipe.compile_study(view, split, config.recipe_config)
        _validate_imported_study_contract(payload, compiled)
        payload["resolved_config"] = config.model_dump(mode="json")
        payload["study_binding"] = {
            "kind": "native_study" if config.study is not None else "run_config",
            "uri": None if config.study is None else str(config.study),
            "study_content_hash": owner.content_hash(),
            "selection_hash": view.semantic_hash(),
            "compiled_problem_hash": compiled.metadata.get("compiled_problem_hash"),
        }
    finally:
        owner.close()
    artifacts = {relative: root / relative for relative in payload["artifacts"]}
    payload.pop("artifacts", None)
    payload.pop("run_contract_hash", None)
    return write_run_json(root, payload, artifacts=artifacts)


__all__ = [
    "CheckpointEnvelope",
    "CheckpointMode",
    "NativeCheckpointCodec",
    "bind_run_study",
    "open_run",
    "tensor_state_sha256",
    "write_compact_run_json",
    "write_imported_run_json",
    "write_run_json",
]
