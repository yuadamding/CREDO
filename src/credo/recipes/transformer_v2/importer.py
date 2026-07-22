"""Importer for inference-only historical transformer-v2 checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from ...artifacts import CheckpointEnvelope, CheckpointMode
from ...contracts import RepresentationArtifact, SplitSpec
from .model import FullDynamicsModel
from .recipe import TransformerSDEV2Recipe, recipe
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


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    credo_root = root.parents[1]
    files = [
        *(root / name for name in ("model.py", "recipe.py", "vae.py", "weak_form.py")),
        credo_root / "contracts.py",
        credo_root / "particles.py",
        credo_root / "objective.py",
        credo_root / "artifacts.py",
        credo_root / "evaluation.py",
        credo_root / "counterfactual.py",
        root / "replay.py",
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
) -> tuple[list[str], str]:
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
        identifiers = sorted(frame["perturbation_id"].astype(str).unique().tolist())
        provenance = str(path)
    else:
        identifiers = sorted({str(value) for value in source})
        provenance = "explicit_identifiers"
    if not identifiers:
        raise ValueError("Legacy perturbation catalog is empty.")
    return identifiers, provenance


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
    if not values or not set(values) <= set(perturbation_ids):
        raise ValueError("Control IDs must be a nonempty subset of the perturbation catalog.")
    return values, source


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
    *,
    gene_names: Path | None,
    gene_mask: Path | None,
    included_samples: Sequence[str],
) -> RepresentationArtifact:
    latent_hash = sha256_file(latents)
    state_hash = sha256_file(vae_state)
    normalization = metadata.get("latent_standardization")
    normalization_hash = None
    if normalization is not None:
        payload = json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode()
        normalization_hash = hashlib.sha256(payload).hexdigest()
    latent_shape = np.load(latents, mmap_mode="r").shape
    latent_dim = int(metadata["vae_hyperparams"]["latent_dim"])
    if len(latent_shape) != 2 or latent_shape[1] != latent_dim:
        raise ValueError("Preserved latent cache shape disagrees with VAE metadata.")
    return RepresentationArtifact(
        representation_id=f"lps-shared-source-vae:{state_hash[:12]}",
        backend="expression_vae_v2",
        latent_dim=latent_dim,
        gene_names_hash=None if gene_names is None else sha256_file(gene_names),
        gene_mask_hash=None if gene_mask is None else sha256_file(gene_mask),
        encoder_state_hash=state_hash,
        decoder_state_hash=state_hash,
        latent_cache_hash=latent_hash,
        normalization_hash=normalization_hash,
        fit_scope="all_source_samples",
        included_samples=tuple(str(value) for value in included_samples),
        included_time_labels=("90m",),
        producer={
            "format": "expression_vae_v2",
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
    run_config_path: Path
    latents_path: Path
    model_state: Literal["raw", "ema"]
    source_payload: Mapping[str, Any]
    study: Any = None

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
        from .replay import evaluate_replay

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
        from .replay import counterfactual_replay

        selected_study = self.study if study is None else study
        if selected_study is None:
            raise ValueError("Imported transformer-v2 counterfactuals require a CREDOStudy.")
        return counterfactual_replay(
            self,
            selected_study,
            measure_id,
            **kwargs,
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
    identifiers, catalog_source = _catalog_ids(checkpoint_path, catalog)
    control_ids, control_source = _control_ids(identifiers, controls)
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
        try:
            import anndata as ad

            adata = ad.read_h5ad(Path(study_source).expanduser().resolve(), backed="r")
            included_samples = tuple(sorted(adata.obs["sample_id"].astype(str).unique()))
        except (ImportError, KeyError, OSError, ValueError) as exc:
            raise ValueError("Unable to resolve representation samples from study_source.") from exc
    artifact = _representation_artifact(
        vae_state_path,
        metadata_path,
        latents_path,
        vae_meta,
        gene_names=names_path,
        gene_mask=mask_path,
        included_samples=included_samples,
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
                "source": payload["source_label"],
                "labels": [payload["source_label"], *payload["target_labels"]],
                "values": [
                    trajectory.get("physical_times", {}).get(label)
                    for label in [payload["source_label"], *payload["target_labels"]]
                ],
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
                "semantic_hash": _state_hash(state),
                "raw_semantic_hash": _state_hash(payload["model_state_dict"]),
            },
            "ema": {
                "source_key": "ema_state_dict",
                "available": bool(ema),
                "tensor_count": 0 if not ema else len(ema),
                "semantic_hash": None if not ema else _state_hash(ema),
            },
            "representation": {
                "source_path": str(vae_state_path),
                "tensor_count": len(vae_state),
                "semantic_hash": _state_hash(vae_state),
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
            "training_recipe_available": True,
            "resume_supported": False,
            "exact_retraining": False,
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
            "control_source": control_source,
            "perturbation_ids": identifiers,
            "control_ids": control_ids,
            "training_measure_keys": list(payload["measure_keys"]),
            "training_embedding_ids": list(payload["embedding_ids"]),
            "missing_resume_state": ["optimizer", "scheduler", "rng", "terminal_state"],
        },
    )
    if sha256_file(checkpoint_path) != original_hash:
        raise RuntimeError("Legacy source checkpoint changed during import.")
    if output is not None:
        destination = Path(output).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        envelope_path = destination / "envelope.json"
        envelope_path.write_text(
            json.dumps(envelope.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return ImportedTransformerV2Run(
        recipe=recipe,
        model=dynamics,
        vae=vae,
        envelope=envelope,
        representation=artifact,
        split=split,
        checkpoint_path=checkpoint_path,
        run_config_path=config_path,
        latents_path=latents_path,
        model_state=model_state,
        source_payload=payload,
    )


__all__ = ["ImportedTransformerV2Run", "import_legacy_checkpoint", "sha256_file"]
