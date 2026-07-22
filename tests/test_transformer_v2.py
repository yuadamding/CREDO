from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from credo import counterfactual, evaluate
from credo.artifacts import CheckpointMode, tensor_state_sha256
from credo.io import RunConfig
from credo.recipes.transformer_v2.importer import (
    import_legacy_checkpoint,
    load_imported_bundle,
    sha256_file,
)
from credo.recipes.transformer_v2.model import PerturbationEmbedding
from credo.recipes.transformer_v2.replay import load_lps_replay_study
from credo.recipes.transformer_v2.vae import ExpressionVAE
from credo.recipes.transformer_v2.weak_form import WeakFormLoss
from credo.registry import get_recipe
from credo.runtime import TrainingEngine

ARCHIVE = Path(
    "/home/yding1995/opscc_sc/trained_models/LPS_checkpoints/"
    "lps_oof_tx_ind32_refine500_vae40/oof_inference_bundle"
)
LPS_STUDY = Path("/home/yding1995/opscc_sc/inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad")
REPLAY = ARCHIVE.parent / "credo_replay"


def _paths(fold: str) -> dict[str, Path]:
    root = ARCHIVE / fold
    if not root.is_dir():
        pytest.skip(f"Archived LPS bundle is unavailable: {root}")
    return {
        "checkpoint": root / "checkpoint_best.pt",
        "run_config": root / "run_config.json",
        "representation": root / "vae_artifact/vae_state_dict.pt",
        "latents": root / "vae_artifact/latent_all_std.npy",
    }


def _public_legacy_fixture(root: Path) -> dict[str, object]:
    source = root / "source"
    source.mkdir()
    identifiers = ["ctrl__reference", *(f"perturbation_{index:02d}" for index in range(30))]
    recipe = get_recipe("credo.transformer_sde_v2@2.0")
    torch.manual_seed(13)
    model = recipe._construct_model(identifiers, ["ctrl__reference"], 50, {})
    checkpoint = source / "checkpoint_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_ids": identifiers,
            "measure_keys": [f"measure_{index:02d}" for index in range(len(identifiers))],
            "source_label": "90m",
            "target_labels": ["6h", "10h"],
            "epoch": 7,
            "history": [{"epoch": 7, "loss": 1.0}],
        },
        checkpoint,
    )
    config = source / "run_config.json"
    config.write_text(
        json.dumps(
            {
                "latent": {"dim": 50},
                "model": {},
                "training": {
                    "optimizer": "adamw",
                    "precision": "bf16",
                    "lambda_end": 1.0,
                    "lambda_weak": 0.12,
                },
                "simulation": {"n_particles": 128, "n_steps": 24},
                "trajectory_training": {
                    "steps_per_interval": 24,
                    "endpoint_time_weights": {"6h": 0.5, "10h": 1.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    vae = ExpressionVAE(2_500, 50, hidden_dim=512, depth=2, dropout=0.1)
    representation = source / "vae_state_dict.pt"
    torch.save(vae.state_dict(), representation)
    (source / "vae_metadata.json").write_text(
        json.dumps(
            {
                "vae_hyperparams": {
                    "input_dim": 2_500,
                    "latent_dim": 50,
                    "hidden_dim": 512,
                    "depth": 2,
                    "dropout": 0.1,
                },
                "training_summary": {"fixture": "generated-public-contract"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "vae_gene_names.txt").write_text("GENE_A\n", encoding="utf-8")
    np.save(source / "vae_gene_mask.npy", np.array([True], dtype=bool))
    latents = source / "latent_all_std.npy"
    np.save(latents, np.zeros((4, 50), dtype=np.float32))
    return {
        "checkpoint": checkpoint,
        "run_config": config,
        "representation": representation,
        "latents": latents,
        "catalog": identifiers,
    }


@pytest.mark.parametrize(
    ("fold", "validation_samples"),
    [
        ("fold00", ("02",)),
        ("fold01", ("06",)),
        ("fold02", ("03", "12")),
        ("fold03", ("01", "04")),
    ],
)
def test_all_archived_v2_dynamics_and_vae_states_load_strictly(
    fold: str, validation_samples: tuple[str, ...]
) -> None:
    run = import_legacy_checkpoint(**_paths(fold))
    assert len(run.model.state_dict()) == 146
    assert sum(value.numel() for value in run.model.state_dict().values()) == 5_634_421
    assert len(run.vae.state_dict()) == 14
    assert sum(value.numel() for value in run.vae.state_dict().values()) == 3_165_736
    assert run.split.validation_values == validation_samples
    assert run.split.representation_scope == "shared"
    assert run.representation.fit_scope == "all_source_samples"
    assert run.representation.representation_id.startswith("expression-vae-v2:")
    assert run.representation.included_time_labels == ("90m",)
    assert run.representation.encoder_state_hash != run.representation.decoder_state_hash
    assert len(run.representation.producer["metadata_sha256"]) == 64


def test_v2_importer_selects_embedded_ema_preserves_hash_and_rejects_resume(
    tmp_path: Path,
) -> None:
    paths = _paths("fold00")
    source_hash = sha256_file(paths["checkpoint"])
    raw = import_legacy_checkpoint(**paths, model_state="raw")
    ema = import_legacy_checkpoint(
        **paths,
        model_state="ema",
        output=tmp_path / "imported",
    )
    assert source_hash == sha256_file(paths["checkpoint"])
    assert raw.envelope.mode is CheckpointMode.INFERENCE_ONLY
    with pytest.raises(RuntimeError, match="cannot resume"):
        raw.envelope.require_resume()
    assert not torch.equal(
        raw.model.embedding.reference_embedding,
        ema.model.embedding.reference_embedding,
    )
    envelope = json.loads((tmp_path / "imported/envelope.json").read_text(encoding="utf-8"))
    assert envelope["import_provenance"]["source_checkpoint_sha256"] == source_hash
    assert envelope["state"]["optimizer"] is None
    assert envelope["state"]["rng"] is None
    assert envelope["training"]["resume_supported"] is False
    assert envelope["training"]["training_plan_status"] == ("typed_reconstruction_not_executable")
    assert envelope["training"]["training_plan"]["particles"] == 128
    assert envelope["training"]["objective_descriptors"]
    assert envelope["state"]["model"]["selection"] == "ema"
    assert (
        envelope["state"]["model"]["semantic_hash"]
        != envelope["state"]["model"]["raw_semantic_hash"]
    )


def test_transformer_v2_recipe_is_registered() -> None:
    recipe = get_recipe("credo.transformer_sde_v2@2.0")
    assert recipe.recipe_id == "credo.transformer_sde_v2"
    assert recipe.capabilities.context == "full_population"
    assert recipe.capabilities.context_affects == ("drift", "diffusion", "growth")
    assert recipe.capabilities.checkpoint_resume_supported is False

    identifiers = ["ctrl__reference", *(f"perturbation_{index:02d}" for index in range(30))]
    model = recipe._construct_model(identifiers, ["ctrl__reference"], 50, {})
    assert len(model.state_dict()) == recipe.state_tensor_count == 146
    assert sum(value.numel() for value in model.state_dict().values()) == 5_634_421
    vae = ExpressionVAE(2_500, 50, hidden_dim=512, depth=2, dropout=0.1)
    assert len(vae.state_dict()) == recipe.vae_tensor_count == 14
    assert sum(value.numel() for value in vae.state_dict().values()) == 3_165_736


def test_transformer_v2_uses_the_normal_recipe_config_path(tiny_config, tiny_data) -> None:
    raw = tiny_config.model_dump()
    raw["recipe"] = "credo.transformer_sde_v2@2.0"
    raw["data"]["counts"] = None
    raw["recipe_config"] = {}
    config = RunConfig.model_validate(raw)
    assert config.recipe_config.model.embedding_dim == 48
    assert config.recipe_config.training.precision == "bf16"
    with pytest.raises(RuntimeError, match="does not support 'train'"):
        TrainingEngine().fit(
            get_recipe(config.recipe),
            replace(tiny_data, count_blocks=()),
            config,
            device="cpu",
        )


def test_transformer_v2_weak_form_is_finite_on_a_portable_fixture() -> None:
    generator = torch.Generator().manual_seed(17)
    z_steps = torch.randn(3, 1, 4, 2, generator=generator)
    logw_steps = torch.randn(3, 1, 4, generator=generator)
    drift_steps = torch.randn(2, 1, 4, 2, generator=generator)
    diffusion_steps = torch.rand(2, 1, 4, 2, generator=generator)
    growth_steps = torch.randn(2, 1, 4, generator=generator)
    loss = WeakFormLoss(n_test_functions=3, bandwidth=1.0, latent_dim=2)(
        z_steps,
        logw_steps,
        drift_steps,
        diffusion_steps,
        growth_steps,
        torch.tensor([0.0, 0.5, 1.0]),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_v2_residual_override_is_functional() -> None:
    embedding = PerturbationEmbedding(
        ["ctrl__reference", "GENE1", "GENE2"],
        ["ctrl__reference"],
        4,
    )
    with torch.no_grad():
        embedding.reference_embedding.fill_(0.25)
        embedding.embeddings.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        embedding.growth_bias.copy_(torch.tensor([1.5, -0.5]))
    before = tensor_state_sha256(embedding.state_dict())
    ids = ["ctrl__reference", "GENE1", "GENE2"]
    factual = embedding(ids)
    factual_growth = embedding.growth_intercepts(ids)
    scale = torch.tensor([1.0, 0.0, 1.0])
    reference = embedding(ids, scale)
    reference_growth = embedding.growth_intercepts(ids, scale)
    assert torch.equal(reference[0], factual[0])
    assert torch.equal(reference[1], embedding.reference_embedding)
    assert torch.equal(reference[2], factual[2])
    assert reference_growth[1] == 0
    assert reference_growth[2] == factual_growth[2]
    assert tensor_state_sha256(embedding.state_dict()) == before


def test_generated_legacy_bundle_is_portable_and_hash_verified(tmp_path: Path) -> None:
    paths = _public_legacy_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    imported = import_legacy_checkpoint(**paths, output=bundle)
    expected_model_hash = tensor_state_sha256(imported.model.state_dict())
    expected_vae_hash = tensor_state_sha256(imported.vae.state_dict())
    shutil.rmtree(tmp_path / "source")

    loaded = load_imported_bundle(bundle)
    assert tensor_state_sha256(loaded.model.state_dict()) == expected_model_hash
    assert tensor_state_sha256(loaded.vae.state_dict()) == expected_vae_hash
    assert loaded.model.perturbation_ids == imported.model.perturbation_ids
    assert loaded.envelope.mode is CheckpointMode.INFERENCE_ONLY
    assert loaded.bundle_path == bundle.resolve()
    assert loaded.envelope.training["source_run_config"]["training"]["optimizer"] == "adamw"
    assert (bundle / "gene_names.txt").is_file()
    assert (bundle / "gene_mask.npy").is_file()

    source_manifest_path = bundle / "source_manifest.json"
    artifact_manifest_path = bundle / "artifact_manifest.json"
    original_source_manifest = source_manifest_path.read_text(encoding="utf-8")
    original_artifact_manifest = artifact_manifest_path.read_text(encoding="utf-8")

    artifact_manifest = json.loads(original_artifact_manifest)
    artifact_manifest["artifacts"]["checkpoint.pt"]["bytes"] = None
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact size mismatch: checkpoint.pt"):
        load_imported_bundle(bundle)
    artifact_manifest_path.write_text(original_artifact_manifest, encoding="utf-8")

    for field, value in (("sha256", "0" * 64), ("bytes", None)):
        source_manifest = json.loads(original_source_manifest)
        source_manifest["source_artifacts"]["checkpoint"][field] = value
        source_manifest_path.write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_manifest = json.loads(original_artifact_manifest)
        artifact_manifest["artifacts"]["source_manifest.json"] = {
            "sha256": sha256_file(source_manifest_path),
            "bytes": source_manifest_path.stat().st_size,
        }
        artifact_manifest_path.write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="disagrees with checkpoint provenance: checkpoint"):
            load_imported_bundle(bundle)
    source_manifest_path.write_text(original_source_manifest, encoding="utf-8")
    artifact_manifest_path.write_text(original_artifact_manifest, encoding="utf-8")

    envelope_path = bundle / "envelope.json"
    envelope_path.write_text(
        envelope_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_imported_bundle(bundle)


def test_v2_import_rejects_nonfinite_latent_cache(tmp_path: Path) -> None:
    paths = _public_legacy_fixture(tmp_path)
    latent_path = paths["latents"]
    latent = np.load(latent_path)
    latent[0, 0] = np.nan
    np.save(latent_path, latent)
    with pytest.raises(ValueError, match="latent cache contains non-finite"):
        import_legacy_checkpoint(**paths)


def test_v2_common_evaluation_and_counterfactual_runtime_contracts() -> None:
    if not LPS_STUDY.is_file():
        pytest.skip(f"Archived LPS study is unavailable: {LPS_STUDY}")
    run = import_legacy_checkpoint(**_paths("fold00"))
    study = load_lps_replay_study(run, LPS_STUDY)
    archived_config = run.envelope.training["source_run_config"]
    objectives = run.recipe.build_objectives(study, archived_config)
    plan = run.recipe.training_plan(study, archived_config)
    assert tuple(term.name for term in objectives) == (
        "checkpoint_geometry_mass",
        "weak_form_residual",
        "rollout_regularization",
        "embedding_regularization",
        "growth_intercept_regularization",
        "ecological_payoff_regularization",
    )
    assert objectives[1].weight == pytest.approx(0.12)
    assert objectives[2].config["diffusion_weight"] == pytest.approx(2e-4)
    assert objectives[3].config["residual_weight"] == pytest.approx(1e-4)
    assert objectives[3].config["control_reference_weight"] == pytest.approx(5e-4)
    assert len(plan.stages) == 1
    assert plan.stages[0].precision == "bf16"
    assert plan.stages[0].batching.mode == "all_keys"
    assert plan.seed == 0
    assert plan.particles == 128
    assert plan.steps_per_interval == 24
    assert plan.early_stopping_patience == 50
    assert plan.gradient_clip_norm == 1.0
    assert plan.stages[0].optimizer.parameter_learning_rates == {
        "dynamics": 3e-4,
        "embeddings": 1e-3,
        "transformer": 5e-5,
    }
    assert plan.stages[0].optimizer.parameter_weight_decays == {
        "dynamics": 1e-6,
        "embeddings": 1e-6,
        "transformer": 1e-4,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate(
        run,
        study=study,
        particles=8,
        seed=17,
        noise_seed=23,
        device=device,
    )
    with pytest.raises(ValueError, match="noise seed must be nonnegative"):
        evaluate(run, study=study, particles=4, seed=17, noise_seed=-1, device=device)
    archived = pytest.importorskip("pandas").read_csv(
        ARCHIVE / "fold00/predicted_metrics_by_key_time.csv"
    )
    assert len(metrics) == len(archived) == 49
    assert metrics["measure_id"].tolist() == archived["measure_key"].tolist()
    before = tensor_state_sha256(run.model.state_dict())
    counterfactual(run, study.measure_ids[0], particles=4, seed=29)
    assert tensor_state_sha256(run.model.state_dict()) == before
    control = counterfactual(run, study.measure_ids[-1], n_particles=4, seed=29)
    assert control["delta_log_mass"].eq(0).all()
    assert control["mean_shift_l2"].eq(0).all()


def test_generated_four_fold_replay_contract() -> None:
    manifest_path = REPLAY / "replay_manifest.json"
    metrics_path = REPLAY / "oof_metrics.parquet"
    if not manifest_path.is_file() or not metrics_path.is_file():
        pytest.skip(f"Generated transformer-v2 replay is unavailable: {REPLAY}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pytest.importorskip("pandas").read_parquet(metrics_path)
    assert manifest["folds"] == ["fold00", "fold01", "fold02", "fold03"]
    assert manifest["oof_rows"] == manifest["archived_oof_rows"] == len(metrics) == 268
    assert manifest["oof_row_count_match"] is True
    assert manifest["all_measure_orders_match"] is True
    assert len(manifest["study_source_sha256"]) == 64
    assert set(manifest["agreement"].values()) <= {"exact", "tolerance-level"}
    for report in manifest["fold_reports"].values():
        assert not set(report["dynamics_train_samples"]) & set(report["held_out_samples"])
        assert report["representation_scope"] == "shared"
        assert report["representation_fit_scope"] == "all_source_samples"
