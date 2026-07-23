"""Importer for inference-only historical transformer-v2 checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from ...artifacts import CheckpointEnvelope, CheckpointMode, tensor_state_sha256
from ...contracts import Axis, RepresentationArtifact, SplitSpec
from ...runtime import validate_training_contract
from .model import FullDynamicsModel
from .recipe import TransformerSDEV2Recipe, TransformerV2RecipeConfig, recipe
from .vae import ExpressionVAE

LEGACY_REQUIRED_KEYS = frozenset(
    {
        "model_state_dict",
        "embedding_ids",
        "measure_keys",
        "source_label",
        "target_labels",
        "epoch",
        "history",
    }
)
LEGACY_COUNT_KEYS = ("count_lik_state_dict", "count_likelihood_state_dict")
HISTORICAL_SOURCE_COMMIT = "4d4fc6a834b59c1c72ab35ba3d298e0f21130aaf"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _ordered_ids_sha256(identifiers: Sequence[str]) -> str:
    payload = json.dumps(list(identifiers), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validated_latent_shape(path: Path) -> tuple[int, int]:
    latent = np.load(path, mmap_mode="r")
    if latent.ndim != 2 or latent.shape[0] < 1 or latent.shape[1] < 1:
        raise ValueError("Preserved latent cache must be a nonempty two-dimensional array.")
    if not np.issubdtype(latent.dtype, np.floating):
        raise ValueError("Preserved latent cache must have a floating-point dtype.")
    row_bytes = max(1, int(latent.shape[1]) * int(latent.dtype.itemsize))
    rows_per_block = max(1, (8 * 1024 * 1024) // row_bytes)
    for start in range(0, int(latent.shape[0]), rows_per_block):
        if not np.isfinite(latent[start : start + rows_per_block]).all():
            raise ValueError("Preserved latent cache contains non-finite values.")
    return int(latent.shape[0]), int(latent.shape[1])


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    credo_root = root.parents[1]
    files = [
        *(root / name for name in ("model.py", "recipe.py", "vae.py", "weak_form.py")),
        credo_root / "contracts.py",
        credo_root / "data/splits.py",
        credo_root / "recipes/compact_sde_v3/particles.py",
        credo_root / "recipes/compact_sde_v3/objective.py",
        credo_root / "artifacts.py",
        credo_root / "runtime.py",
        credo_root / "evaluation.py",
        credo_root / "counterfactual.py",
        credo_root / "recipes/trajectory_compiler.py",
        root / "inference.py",
        root / "importer.py",
    ]
    for path in files:
        digest.update(path.relative_to(credo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _catalog_ids(
    checkpoint: Path,
    catalog: str | Path | Sequence[str] | None,
) -> tuple[list[str], str, str]:
    source = catalog
    if source is None and (checkpoint.parent / "mass_table.csv").is_file():
        source = checkpoint.parent / "mass_table.csv"
    if source is None:
        raise ValueError(
            "Legacy checkpoint omits the full embedding catalog; provide catalog IDs "
            "or a mass_table.csv."
        )
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        frame = pd.read_csv(path, dtype={"perturbation_id": str})
        if "perturbation_id" not in frame:
            raise ValueError("Legacy catalog table requires a perturbation_id column.")
        if "embedding_index" in frame:
            ordered = (
                frame[["perturbation_id", "embedding_index"]]
                .drop_duplicates("perturbation_id")
                .sort_values("embedding_index")
            )
            indices = ordered["embedding_index"].astype(int).tolist()
            if indices != list(range(len(indices))):
                raise ValueError("Catalog embedding_index must be contiguous from zero.")
            identifiers = ordered["perturbation_id"].astype(str).tolist()
            ordering = "explicit_embedding_index"
        else:
            identifiers = sorted(frame["perturbation_id"].astype(str).unique().tolist())
            ordering = "legacy_v2_lps_sorted_catalog"
        provenance = str(path)
    else:
        identifiers = [str(value) for value in source]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Explicit legacy perturbation catalog contains duplicates.")
        provenance = "explicit_identifiers"
        ordering = "explicit_sequence"
    if not identifiers:
        raise ValueError("Legacy perturbation catalog is empty.")
    return identifiers, provenance, ordering


def _control_ids(
    perturbation_ids: Sequence[str], controls: Sequence[str] | None
) -> tuple[list[str], str]:
    if controls is not None:
        values = [str(value) for value in controls]
        source = "explicit"
    else:
        values = [value for value in perturbation_ids if value.lower().startswith("ctrl__")]
        source = "unique_ctrl_prefix"
        if len(values) != 1:
            raise ValueError(
                "Legacy checkpoint omits control IDs; pass controls explicitly because "
                "the catalog does not contain exactly one ctrl__ identifier."
            )
    if len(values) != len(set(values)):
        raise ValueError("Control IDs must be unique.")
    if not values or not set(values) <= set(perturbation_ids):
        raise ValueError("Control IDs must be a nonempty subset of the perturbation catalog.")
    return values, source


def _canonical_recipe_config(
    config: Mapping[str, Any],
    perturbation_ids: Sequence[str],
    control_ids: Sequence[str],
) -> TransformerV2RecipeConfig:
    payload: dict[str, Any] = {
        "perturbation_ids": list(perturbation_ids),
        "control_ids": list(control_ids),
    }
    for section in ("model", "training", "simulation", "trajectory_training"):
        section_type = TransformerV2RecipeConfig.model_fields[section].annotation
        allowed = set(section_type.model_fields)
        raw = config.get(section, {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"Legacy run config section {section!r} must be a mapping.")
        payload[section] = {name: value for name, value in raw.items() if name in allowed}
    return TransformerV2RecipeConfig.model_validate(payload)


def _load_vae(
    state_path: Path,
    metadata_path: Path,
    device: torch.device,
) -> tuple[ExpressionVAE, dict[str, Any], Mapping[str, torch.Tensor]]:
    metadata = _json(metadata_path)
    hyperparameters = dict(metadata.get("vae_hyperparams", {}))
    required = {"input_dim", "latent_dim", "hidden_dim", "depth", "dropout"}
    missing = required - set(hyperparameters)
    if missing:
        raise ValueError(f"VAE metadata is missing hyperparameters: {sorted(missing)}")
    model = ExpressionVAE(
        int(hyperparameters["input_dim"]),
        int(hyperparameters["latent_dim"]),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        depth=int(hyperparameters["depth"]),
        dropout=float(hyperparameters["dropout"]),
    ).to(device)
    state = torch.load(state_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata, state


def _representation_artifact(
    vae_state: Path,
    vae_metadata: Path,
    latents: Path,
    metadata: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    gene_names: Path | None,
    gene_mask: Path | None,
    included_samples: Sequence[str],
    source_label: str,
) -> RepresentationArtifact:
    latent_hash = sha256_file(latents)
    state_hash = sha256_file(vae_state)
    encoder_state = {
        name: value
        for name, value in state.items()
        if name.startswith(("encoder.", "mu_head.", "logvar_head."))
    }
    decoder_state = {name: value for name, value in state.items() if name.startswith("decoder.")}
    if not encoder_state or not decoder_state:
        raise ValueError("VAE state does not expose distinct encoder and decoder components.")
    normalization = metadata.get("latent_standardization")
    normalization_hash = None
    if normalization is not None:
        payload = json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode()
        normalization_hash = hashlib.sha256(payload).hexdigest()
    latent_shape = _validated_latent_shape(latents)
    latent_dim = int(metadata["vae_hyperparams"]["latent_dim"])
    if len(latent_shape) != 2 or latent_shape[1] != latent_dim:
        raise ValueError("Preserved latent cache shape disagrees with VAE metadata.")
    return RepresentationArtifact(
        representation_id=f"expression-vae-v2:{state_hash[:12]}",
        backend="expression_vae_v2",
        latent_dim=latent_dim,
        gene_names_hash=None if gene_names is None else sha256_file(gene_names),
        gene_mask_hash=None if gene_mask is None else sha256_file(gene_mask),
        encoder_state_hash=tensor_state_sha256(encoder_state),
        decoder_state_hash=tensor_state_sha256(decoder_state),
        latent_cache_hash=latent_hash,
        normalization_hash=normalization_hash,
        fit_scope="all_source_samples",
        included_samples=tuple(str(value) for value in included_samples),
        included_time_labels=(str(source_label),),
        producer={
            "format": "expression_vae_v2",
            "state_file_sha256": state_hash,
            "metadata_sha256": sha256_file(vae_metadata),
            "training_summary": metadata.get("training_summary", {}),
            "latent_rows": int(latent_shape[0]),
            "producer_exactly_known": False,
        },
    )


@dataclass
class ImportedTransformerV2Run:
    recipe: TransformerSDEV2Recipe
    model: FullDynamicsModel
    vae: ExpressionVAE
    envelope: CheckpointEnvelope
    representation: RepresentationArtifact
    split: SplitSpec
    checkpoint_path: Path
    latents_path: Path
    model_state: Literal["raw", "ema"]
    study: Any = None
    bundle_path: Path | None = None
    run_manifest: Mapping[str, Any] | None = None

    @property
    def recipe_id(self) -> str:
        return self.recipe.recipe_id

    @property
    def recipe_version(self) -> str:
        return self.recipe.recipe_version

    @property
    def capabilities(self):
        return self.recipe.capabilities

    def require(self, operation: str) -> None:
        self.capabilities.require(operation)

    def evaluate_runtime(
        self,
        *,
        study: Any = None,
        particles: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ):
        from .inference import evaluate_replay

        selected_study = self.study if study is None else study
        if selected_study is None:
            raise ValueError("Imported transformer-v2 evaluation requires a CREDOStudy.")
        frame, _ = evaluate_replay(
            self,
            selected_study,
            particles=640 if particles is None else particles,
            seed=0 if seed is None else seed,
            **kwargs,
        )
        return frame

    def counterfactual_runtime(self, measure_id: str, *, study: Any = None, **kwargs: Any):
        from .inference import counterfactual_replay

        selected_study = self.study if study is None else study
        if selected_study is None:
            raise ValueError("Imported transformer-v2 counterfactuals require a CREDOStudy.")
        return counterfactual_replay(
            self,
            selected_study,
            measure_id,
            **kwargs,
        )

    def close(self) -> None:
        owner = getattr(self, "_semantic_owner", None)
        if owner is not None:
            owner.close()
            self._semantic_owner = None


def _cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in state.items()}


def _write_portable_bundle(
    destination: Path,
    *,
    envelope: CheckpointEnvelope,
    model_state: Mapping[str, torch.Tensor],
    vae_state: Mapping[str, torch.Tensor],
    vae_metadata: Mapping[str, Any],
    representation: RepresentationArtifact,
    latents: Path,
    perturbation_ids: Sequence[str],
    control_ids: Sequence[str],
    latent_dim: int,
    model_config: Mapping[str, Any],
    model_state_selection: str,
    source_artifacts: Mapping[str, Path],
    gene_names: Path | None,
    gene_mask: Path | None,
) -> Path:
    expected_names = {
        "checkpoint.pt",
        "representation.pt",
        "representation.json",
        "envelope.json",
        "latents.npy",
        "source_manifest.json",
        "artifact_manifest.json",
        "run.json",
    }
    if gene_names is not None:
        expected_names.add("gene_names.txt")
    if gene_mask is not None:
        expected_names.add("gene_mask.npy")
    if destination.exists():
        unknown = sorted(
            path.name for path in destination.iterdir() if path.name not in expected_names
        )
        if unknown:
            raise FileExistsError(f"Portable bundle contains files outside its contract: {unknown}")
    destination.mkdir(parents=True, exist_ok=True)
    envelope_payload = json.loads(json.dumps(envelope.to_dict()))
    checkpoint_path = destination / "checkpoint.pt"
    representation_path = destination / "representation.pt"
    representation_metadata_path = destination / "representation.json"
    envelope_path = destination / "envelope.json"
    latents_path = destination / "latents.npy"
    source_manifest_path = destination / "source_manifest.json"
    artifact_manifest_path = destination / "artifact_manifest.json"

    torch.save(
        {
            "schema_version": 2,
            "envelope": envelope_payload,
            "architecture": {
                "perturbation_ids": list(perturbation_ids),
                "control_ids": list(control_ids),
                "latent_dim": int(latent_dim),
                "model": dict(model_config),
            },
            "model_state_selection": str(model_state_selection),
            "model_state": _cpu_state(model_state),
        },
        checkpoint_path,
    )
    torch.save(
        {"schema_version": 1, "state_dict": _cpu_state(vae_state)},
        representation_path,
    )
    representation_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "representation": representation.to_dict(),
                "vae_metadata": dict(vae_metadata),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    envelope_path.write_text(
        json.dumps(envelope_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(latents, latents_path)
    optional_paths: dict[str, Path] = {}
    if gene_names is not None:
        optional_paths["gene_names.txt"] = gene_names
    if gene_mask is not None:
        optional_paths["gene_mask.npy"] = gene_mask
    for relative, source in optional_paths.items():
        shutil.copy2(source, destination / relative)

    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_artifacts": {
                    name: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for name, path in sorted(source_artifacts.items())
                },
                "portable_artifacts_do_not_depend_on_source_paths": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_files = [
        checkpoint_path,
        representation_path,
        representation_metadata_path,
        envelope_path,
        latents_path,
        source_manifest_path,
        *(destination / name for name in optional_paths),
    ]
    artifact_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    path.name: {
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in artifact_files
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    from ...artifacts import write_imported_run_json

    write_imported_run_json(destination, envelope)
    return destination


def load_imported_bundle(
    bundle: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ImportedTransformerV2Run:
    """Load and hash-verify a portable transformer-v2 inference bundle."""
    destination = Path(bundle).expanduser().resolve()
    manifest = _json(destination / "artifact_manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported portable artifact manifest schema.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Portable bundle artifact_manifest.json is invalid.")
    actual_names = {path.name for path in destination.iterdir()}
    unknown_files = actual_names - set(artifacts) - {"artifact_manifest.json", "run.json"}
    if unknown_files:
        raise ValueError(f"Portable bundle contains unmanifested files: {sorted(unknown_files)}")
    required_artifacts = {
        "checkpoint.pt",
        "representation.pt",
        "representation.json",
        "envelope.json",
        "latents.npy",
        "source_manifest.json",
    }
    missing_artifacts = required_artifacts - set(artifacts)
    if missing_artifacts:
        raise ValueError(
            f"Portable bundle manifest omits required artifacts: {sorted(missing_artifacts)}"
        )
    for name, contract in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Portable bundle artifact name is unsafe: {name!r}")
        path = destination / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(contract, dict) or contract.get("sha256") != sha256_file(path):
            raise ValueError(f"Portable bundle artifact hash mismatch: {name}")
        declared_size = contract.get("bytes")
        if not _is_nonnegative_int(declared_size) or declared_size != path.stat().st_size:
            raise ValueError(f"Portable bundle artifact size mismatch: {name}")

    selected_device = torch.device(device)
    checkpoint = torch.load(
        destination / "checkpoint.pt",
        map_location=selected_device,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Portable transformer-v2 checkpoint must contain a mapping.")
    if checkpoint.get("schema_version") != 2:
        raise ValueError("Unsupported portable transformer-v2 checkpoint schema.")
    envelope = CheckpointEnvelope.from_dict(checkpoint["envelope"])
    standalone_envelope = CheckpointEnvelope.from_dict(_json(destination / "envelope.json"))
    if json.loads(json.dumps(standalone_envelope.to_dict())) != json.loads(
        json.dumps(envelope.to_dict())
    ):
        raise ValueError("Portable standalone envelope disagrees with checkpoint.pt.")
    if (
        envelope.recipe.get("id") != recipe.recipe_id
        or envelope.recipe.get("version") != recipe.recipe_version
    ):
        raise ValueError("Portable checkpoint recipe disagrees with transformer-v2.")
    if envelope.recipe.get("implementation_hash") != _implementation_hash():
        raise ValueError("Portable checkpoint implementation hash disagrees with this release.")
    if envelope.mode is not CheckpointMode.INFERENCE_ONLY:
        raise ValueError("Portable transformer-v2 checkpoints must be inference-only.")
    expected_capabilities = json.loads(json.dumps(asdict(recipe.capabilities)))
    if envelope.capabilities != expected_capabilities:
        raise ValueError("Portable checkpoint capabilities disagree with transformer-v2.")
    training_contract = envelope.training
    if (
        training_contract.get("training_plan_status") != "typed_reconstruction_not_executable"
        or training_contract.get("fresh_training_executor_available") is not False
    ):
        raise ValueError("Portable transformer-v2 training provenance is invalid.")
    canonical_config = TransformerV2RecipeConfig.model_validate(
        training_contract.get("recipe_config")
    )
    axis_payload = envelope.study_contract.get("axis")
    if not isinstance(axis_payload, Mapping):
        raise ValueError("Portable transformer-v2 study axis is invalid.")
    contract_axis = Axis(
        kind=str(axis_payload["kind"]),
        source=str(axis_payload["source"]),
        labels=tuple(str(value) for value in axis_payload["labels"]),
        values=tuple(float(value) for value in axis_payload["values"]),
    )
    contract_study = SimpleNamespace(axis=contract_axis)
    expected_plan = recipe.training_plan(contract_study, canonical_config)
    expected_objectives = recipe.build_objectives(contract_study, canonical_config)
    validate_training_contract(recipe, expected_objectives, expected_plan)
    if training_contract.get("training_plan") != expected_plan.to_dict():
        raise ValueError("Portable transformer-v2 training plan disagrees with its recipe config.")
    if training_contract.get("objective_descriptors") != [
        objective.to_dict() for objective in expected_objectives
    ]:
        raise ValueError("Portable transformer-v2 objectives disagree with its recipe config.")
    if not isinstance(training_contract.get("source_run_config"), Mapping):
        raise ValueError("Portable transformer-v2 source run config is missing.")
    provenance = envelope.import_provenance
    if provenance is None:
        raise ValueError("Portable transformer-v2 checkpoint lacks import provenance.")
    architecture = checkpoint["architecture"]
    if not isinstance(architecture, Mapping) or not isinstance(architecture.get("model"), Mapping):
        raise ValueError("Portable transformer-v2 architecture is invalid.")
    perturbation_ids = [str(value) for value in architecture["perturbation_ids"]]
    control_ids = [str(value) for value in architecture["control_ids"]]
    if (
        not perturbation_ids
        or len(perturbation_ids) != len(set(perturbation_ids))
        or not control_ids
        or not set(control_ids) <= set(perturbation_ids)
    ):
        raise ValueError("Portable transformer-v2 perturbation catalog is invalid.")
    expected_order_hash = provenance.get("catalog_order_sha256")
    if expected_order_hash != _ordered_ids_sha256(perturbation_ids):
        raise ValueError("Portable checkpoint perturbation order hash mismatch.")
    if provenance.get("perturbation_ids") != perturbation_ids:
        raise ValueError("Portable checkpoint perturbation order disagrees with provenance.")
    if provenance.get("control_ids") != control_ids:
        raise ValueError("Portable checkpoint controls disagree with provenance.")
    latent_dim = int(architecture["latent_dim"])
    if latent_dim < 1:
        raise ValueError("Portable transformer-v2 latent dimension must be positive.")
    dynamics = recipe._construct_model(
        perturbation_ids,
        control_ids,
        latent_dim,
        architecture["model"],
    ).to(selected_device)
    model_state = checkpoint["model_state"]
    if not isinstance(model_state, Mapping):
        raise ValueError("Portable transformer-v2 model state is invalid.")
    dynamics.load_state_dict(model_state, strict=True)
    model_contract = envelope.state["model"]
    if model_contract.get("tensor_count") != len(model_state) or tensor_state_sha256(
        model_state
    ) != model_contract.get("semantic_hash"):
        raise ValueError("Portable transformer-v2 model state hash mismatch.")
    selection = checkpoint.get("model_state_selection")
    if selection not in {"raw", "ema"} or model_contract.get("selection") != selection:
        raise ValueError("Portable transformer-v2 state selection is invalid.")
    dynamics.assert_soft_reference()
    dynamics.eval()

    representation_payload = torch.load(
        destination / "representation.pt",
        map_location=selected_device,
        weights_only=True,
    )
    if not isinstance(representation_payload, Mapping):
        raise ValueError("Portable representation state must contain a mapping.")
    representation_metadata = _json(destination / "representation.json")
    if representation_payload.get("schema_version") != 1:
        raise ValueError("Unsupported portable representation state schema.")
    if representation_metadata.get("schema_version") != 1:
        raise ValueError("Unsupported portable representation metadata schema.")
    vae_metadata = representation_metadata["vae_metadata"]
    if not isinstance(vae_metadata, Mapping):
        raise ValueError("Portable VAE metadata is invalid.")
    hyperparameters = vae_metadata["vae_hyperparams"]
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Portable VAE hyperparameters are invalid.")
    vae = ExpressionVAE(
        int(hyperparameters["input_dim"]),
        int(hyperparameters["latent_dim"]),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        depth=int(hyperparameters["depth"]),
        dropout=float(hyperparameters["dropout"]),
    ).to(selected_device)
    vae_state = representation_payload["state_dict"]
    if not isinstance(vae_state, Mapping):
        raise ValueError("Portable VAE state is invalid.")
    vae.load_state_dict(vae_state, strict=True)
    representation_state_contract = envelope.state["representation"]
    if representation_state_contract.get("tensor_count") != len(vae_state) or tensor_state_sha256(
        vae_state
    ) != representation_state_contract.get("semantic_hash"):
        raise ValueError("Portable transformer-v2 representation state hash mismatch.")
    vae.eval()
    representation = RepresentationArtifact.from_dict(representation_metadata["representation"])
    if representation.to_dict() != envelope.representation_contract:
        raise ValueError("Portable representation metadata disagrees with its envelope.")
    if representation.latent_dim != latent_dim or int(hyperparameters["latent_dim"]) != latent_dim:
        raise ValueError("Portable model, VAE, and representation latent dimensions disagree.")
    optional_representation_files = {
        "gene_names_hash": "gene_names.txt",
        "gene_mask_hash": "gene_mask.npy",
    }
    for field, name in optional_representation_files.items():
        expected_hash = getattr(representation, field)
        if expected_hash is not None:
            if name not in artifacts:
                raise ValueError(f"Portable bundle omits representation artifact {name!r}.")
            if sha256_file(destination / name) != expected_hash:
                raise ValueError(f"Portable representation artifact hash mismatch: {name}")
    latents_path = destination / "latents.npy"
    if sha256_file(latents_path) != representation.latent_cache_hash:
        raise ValueError("Portable latent cache hash mismatch.")
    latent_shape = _validated_latent_shape(latents_path)
    expected_rows = representation.producer.get("latent_rows")
    if (
        len(latent_shape) != 2
        or latent_shape[1] != latent_dim
        or (expected_rows is not None and latent_shape[0] != int(expected_rows))
    ):
        raise ValueError("Portable latent cache shape disagrees with its representation contract.")
    source_manifest = _json(destination / "source_manifest.json")
    if (
        source_manifest.get("schema_version") != 1
        or source_manifest.get("portable_artifacts_do_not_depend_on_source_paths") is not True
        or not isinstance(source_manifest.get("source_artifacts"), dict)
    ):
        raise ValueError("Portable source manifest is invalid.")
    source_artifacts = source_manifest["source_artifacts"]
    allowed_sources = {
        "checkpoint",
        "run_config",
        "representation",
        "representation_metadata",
        "latents",
        "catalog",
        "gene_names",
        "gene_mask",
    }
    if unknown_sources := set(source_artifacts) - allowed_sources:
        raise ValueError(
            f"Portable source manifest has unknown artifacts: {sorted(unknown_sources)}"
        )
    expected_source_hashes = {
        "checkpoint": provenance.get("source_checkpoint_sha256"),
        "run_config": envelope.study_contract.get("input_hashes", {}).get("run_config"),
        "representation": representation.producer.get("state_file_sha256"),
        "representation_metadata": representation.producer.get("metadata_sha256"),
        "latents": representation.latent_cache_hash,
    }
    if representation.gene_names_hash is not None:
        expected_source_hashes["gene_names"] = representation.gene_names_hash
    if representation.gene_mask_hash is not None:
        expected_source_hashes["gene_mask"] = representation.gene_mask_hash
    for name, expected_hash in expected_source_hashes.items():
        contract = source_artifacts.get(name)
        if (
            not isinstance(contract, Mapping)
            or contract.get("sha256") != expected_hash
            or not isinstance(contract.get("path"), str)
            or not contract.get("path")
            or not _is_nonnegative_int(contract.get("bytes"))
        ):
            raise ValueError(
                f"Portable source manifest disagrees with checkpoint provenance: {name}"
            )
    split = SplitSpec(**dict(envelope.split_contract))
    return ImportedTransformerV2Run(
        recipe=recipe,
        model=dynamics,
        vae=vae,
        envelope=envelope,
        representation=representation,
        split=split,
        checkpoint_path=destination / "checkpoint.pt",
        latents_path=latents_path,
        model_state=selection,
        bundle_path=destination,
    )


def import_legacy_checkpoint(
    checkpoint: str | Path,
    run_config: str | Path,
    representation: str | Path,
    latents: str | Path,
    *,
    output: str | Path | None = None,
    vae_metadata: str | Path | None = None,
    gene_names: str | Path | None = None,
    gene_mask: str | Path | None = None,
    catalog: str | Path | Sequence[str] | None = None,
    controls: Sequence[str] | None = None,
    study_source: str | Path | None = None,
    model_state: Literal["raw", "ema"] = "raw",
    device: str | torch.device = "cpu",
) -> ImportedTransformerV2Run:
    """Strict-load a historical wrapper and emit an inference-only envelope."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = Path(run_config).expanduser().resolve()
    vae_state_path = Path(representation).expanduser().resolve()
    latents_path = Path(latents).expanduser().resolve()
    metadata_path = (
        Path(vae_metadata).expanduser().resolve()
        if vae_metadata is not None
        else vae_state_path.parent / "vae_metadata.json"
    )
    names_path = (
        Path(gene_names).expanduser().resolve()
        if gene_names is not None
        else vae_state_path.parent / "vae_gene_names.txt"
    )
    mask_path = (
        Path(gene_mask).expanduser().resolve()
        if gene_mask is not None
        else vae_state_path.parent / "vae_gene_mask.npy"
    )
    for path in (checkpoint_path, config_path, vae_state_path, latents_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not names_path.is_file():
        names_path = None
    if not mask_path.is_file():
        mask_path = None

    original_hash = sha256_file(checkpoint_path)
    selected_device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Legacy checkpoint must contain a mapping wrapper.")
    missing = LEGACY_REQUIRED_KEYS - set(payload)
    if missing:
        raise ValueError(f"Legacy checkpoint is missing wrapper keys: {sorted(missing)}")
    count_keys = [name for name in LEGACY_COUNT_KEYS if name in payload]
    if len(count_keys) > 1:
        raise ValueError("Legacy checkpoint contains ambiguous count-likelihood spellings.")

    config = _json(config_path)
    model_config = dict(config.get("model", {}))
    identifiers, catalog_source, catalog_ordering = _catalog_ids(checkpoint_path, catalog)
    if catalog_ordering == "legacy_v2_lps_sorted_catalog" and (
        str(payload["source_label"]) != "90m"
        or [str(value) for value in payload["target_labels"]] != ["6h", "10h"]
    ):
        raise ValueError(
            "Sorted catalog inference is verified only for the legacy LPS codec; "
            "provide an explicit sequence or embedding_index catalog."
        )
    control_ids, control_source = _control_ids(identifiers, controls)
    training_embedding_ids = [str(value) for value in payload["embedding_ids"]]
    unknown_training_ids = set(training_embedding_ids) - set(identifiers)
    if unknown_training_ids:
        raise ValueError(
            "Legacy training embeddings are absent from the ordered model catalog: "
            f"{sorted(unknown_training_ids)[:5]}"
        )
    dynamics = recipe._construct_model(
        identifiers,
        control_ids,
        int(config.get("latent", {}).get("dim", 50)),
        model_config,
    ).to(selected_device)
    ema = payload.get("ema_state_dict")
    recipe.load_model_state(
        dynamics,
        payload["model_state_dict"],
        ema,
        state=model_state,
    )
    state = dynamics.state_dict()
    if len(state) != recipe.state_tensor_count:
        raise ValueError("Loaded transformer-v2 state tensor count disagrees with the recipe.")
    if sum(value.numel() for value in state.values()) != recipe.state_element_count:
        raise ValueError("Loaded transformer-v2 state element count disagrees with the recipe.")
    residual_rows = int(state["embedding.embeddings"].shape[0])
    if residual_rows != len(identifiers) - len(control_ids):
        raise ValueError("Ordered perturbation catalog does not match historical residual rows.")
    dynamics.eval()

    vae, vae_meta, vae_state = _load_vae(vae_state_path, metadata_path, selected_device)
    if (
        len(vae_state) != recipe.vae_tensor_count
        or sum(value.numel() for value in vae_state.values()) != recipe.vae_element_count
    ):
        raise ValueError("Loaded VAE tensor contract disagrees with transformer-v2.")

    bundle_metadata_path = checkpoint_path.parent / "metadata.json"
    bundle_metadata = _json(bundle_metadata_path) if bundle_metadata_path.is_file() else {}
    raw_validation_samples = bundle_metadata.get("validation_sample_ids", ())
    if isinstance(raw_validation_samples, str):
        validation_samples = tuple(
            value.strip() for value in raw_validation_samples.split(",") if value.strip()
        )
    else:
        validation_samples = tuple(str(value) for value in raw_validation_samples)
    mass_table_path = checkpoint_path.parent / "mass_table.csv"
    training_samples: tuple[str, ...] = ()
    if mass_table_path.is_file():
        masses = pd.read_csv(mass_table_path, dtype={"sample_id": str})
        training_samples = tuple(sorted(masses["sample_id"].astype(str).unique()))
    included_samples = tuple(dict.fromkeys(training_samples + validation_samples))
    if study_source is not None:
        adata = None
        try:
            import anndata as ad

            adata = ad.read_h5ad(Path(study_source).expanduser().resolve(), backed="r")
            included_samples = tuple(sorted(adata.obs["sample_id"].astype(str).unique()))
        except (ImportError, KeyError, OSError, ValueError) as exc:
            raise ValueError("Unable to resolve representation samples from study_source.") from exc
        finally:
            if adata is not None:
                adata.file.close()
    artifact = _representation_artifact(
        vae_state_path,
        metadata_path,
        latents_path,
        vae_meta,
        vae_state,
        gene_names=names_path,
        gene_mask=mask_path,
        included_samples=included_samples,
        source_label=str(payload["source_label"]),
    )

    fold = bundle_metadata.get("fold")
    split = SplitSpec(
        strategy="sample",
        train_values=training_samples or None,
        validation_values=validation_samples or None,
        fold=None if fold is None else int(fold),
        folds=None if fold is None else 4,
        representation_scope="shared",
        split_id=None if fold is None else f"lps-oof-fold{int(fold):02d}",
    )
    trajectory_path = checkpoint_path.parent / "trajectory_config.json"
    trajectory = _json(trajectory_path) if trajectory_path.is_file() else {}
    axis_labels = [str(payload["source_label"]), *map(str, payload["target_labels"])]
    physical_times = trajectory.get("physical_times", {})
    raw_axis_values = (
        [physical_times.get(label) for label in axis_labels]
        if isinstance(physical_times, Mapping)
        else []
    )
    if len(raw_axis_values) == len(axis_labels) and all(
        isinstance(value, (int, float)) for value in raw_axis_values
    ):
        axis_values = [float(value) for value in raw_axis_values]
        axis_value_source = "trajectory_config"
    else:
        axis_values = [float(index) for index in range(len(axis_labels))]
        axis_value_source = "ordered_checkpoint_fallback"
    contract_axis = Axis(
        kind="physical",
        source=axis_labels[0],
        labels=axis_labels,
        values=axis_values,
    )
    canonical_config = _canonical_recipe_config(config, identifiers, control_ids)
    contract_study = SimpleNamespace(axis=contract_axis)
    training_plan = recipe.training_plan(contract_study, canonical_config)
    objectives = recipe.build_objectives(contract_study, canonical_config)
    validate_training_contract(recipe, objectives, training_plan)
    measure_manifest = checkpoint_path.parent / "measure_key_manifest.csv"
    capabilities = asdict(recipe.capabilities)
    envelope = CheckpointEnvelope(
        recipe={
            "id": recipe.recipe_id,
            "version": recipe.recipe_version,
            "implementation_hash": _implementation_hash(),
        },
        study_contract={
            "axis": {
                "kind": "physical",
                "source": contract_axis.source,
                "labels": list(contract_axis.labels),
                "values": list(contract_axis.values),
                "value_source": axis_value_source,
            },
            "mass_semantics": "relative_within_group",
            "measure_meta_hash": (
                None if not measure_manifest.is_file() else sha256_file(measure_manifest)
            ),
            "input_hashes": {
                "checkpoint": original_hash,
                "run_config": sha256_file(config_path),
                "latents": sha256_file(latents_path),
            },
        },
        representation_contract=artifact.to_dict(),
        split_contract=asdict(split),
        state={
            "model": {
                "source_key": (
                    "model_state_dict"
                    if model_state == "raw"
                    else "model_state_dict+ema_state_dict"
                ),
                "selection": model_state,
                "tensor_count": len(state),
                "semantic_hash": tensor_state_sha256(state),
                "raw_semantic_hash": tensor_state_sha256(payload["model_state_dict"]),
            },
            "ema": {
                "source_key": "ema_state_dict",
                "available": bool(ema),
                "tensor_count": 0 if not ema else len(ema),
                "semantic_hash": None if not ema else tensor_state_sha256(ema),
            },
            "representation": {
                "bundle_path": "representation.pt",
                "source_file_sha256": sha256_file(vae_state_path),
                "tensor_count": len(vae_state),
                "semantic_hash": tensor_state_sha256(vae_state),
            },
            "objective": {
                "source_key": count_keys[0] if count_keys else None,
                "available": bool(count_keys),
            },
            "optimizer": None,
            "scheduler": None,
            "rng": None,
        },
        training={
            "completed_epoch": int(payload["epoch"]),
            "history_available": bool(payload["history"]),
            "training_plan_status": "typed_reconstruction_not_executable",
            "recipe_config": canonical_config.model_dump(mode="json"),
            "training_plan": training_plan.to_dict(),
            "objective_descriptors": [objective.to_dict() for objective in objectives],
            "source_run_config": config,
            "fresh_training_executor_available": False,
            "resume_supported": False,
            "deterministic_cpu_fresh_fit_tested": False,
            "bitwise_retraining_demonstrated": False,
        },
        capabilities=capabilities,
        mode=CheckpointMode.INFERENCE_ONLY,
        import_provenance={
            "format": "legacy_v2_transformer",
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": original_hash,
            "candidate_source_commit": HISTORICAL_SOURCE_COMMIT,
            "producer_exactly_known": False,
            "catalog_source": catalog_source,
            "catalog_ordering": catalog_ordering,
            "catalog_order_sha256": _ordered_ids_sha256(identifiers),
            "control_source": control_source,
            "perturbation_ids": identifiers,
            "control_ids": control_ids,
            "training_measure_keys": list(payload["measure_keys"]),
            "training_embedding_ids": training_embedding_ids,
            "training_embedding_ids_role": "per_measure_metadata_not_model_row_order",
            "missing_resume_state": ["optimizer", "scheduler", "rng", "terminal_state"],
        },
    )
    if sha256_file(checkpoint_path) != original_hash:
        raise RuntimeError("Legacy source checkpoint changed during import.")
    bundle_path = None
    if output is not None:
        destination = Path(output).expanduser().resolve()
        source_artifacts = {
            "checkpoint": checkpoint_path,
            "run_config": config_path,
            "representation": vae_state_path,
            "representation_metadata": metadata_path,
            "latents": latents_path,
        }
        if isinstance(catalog_source, str) and Path(catalog_source).is_file():
            source_artifacts["catalog"] = Path(catalog_source)
        if names_path is not None:
            source_artifacts["gene_names"] = names_path
        if mask_path is not None:
            source_artifacts["gene_mask"] = mask_path
        bundle_path = _write_portable_bundle(
            destination,
            envelope=envelope,
            model_state=state,
            vae_state=vae_state,
            vae_metadata=vae_meta,
            representation=artifact,
            latents=latents_path,
            perturbation_ids=identifiers,
            control_ids=control_ids,
            latent_dim=int(config.get("latent", {}).get("dim", 50)),
            model_config=model_config,
            model_state_selection=model_state,
            source_artifacts=source_artifacts,
            gene_names=names_path,
            gene_mask=mask_path,
        )
    return ImportedTransformerV2Run(
        recipe=recipe,
        model=dynamics,
        vae=vae,
        envelope=envelope,
        representation=artifact,
        split=split,
        checkpoint_path=checkpoint_path,
        latents_path=latents_path,
        model_state=model_state,
        bundle_path=bundle_path,
    )


__all__ = [
    "ImportedTransformerV2Run",
    "import_legacy_checkpoint",
    "load_imported_bundle",
    "sha256_file",
]
