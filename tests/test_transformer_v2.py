from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from credo import counterfactual, evaluate
from credo.artifacts import CheckpointMode
from credo.recipes.transformer_v2.importer import import_legacy_checkpoint, sha256_file
from credo.recipes.transformer_v2.replay import load_lps_replay_study
from credo.registry import get_recipe

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
    assert envelope["state"]["model"]["selection"] == "ema"
    assert (
        envelope["state"]["model"]["semantic_hash"]
        != envelope["state"]["model"]["raw_semantic_hash"]
    )


def test_transformer_v2_recipe_is_registered() -> None:
    recipe = get_recipe("credo.transformer_sde_v2@2.0")
    assert recipe.recipe_id == "credo.transformer_sde_v2"
    assert recipe.capabilities.context == "full_population"
    assert recipe.capabilities.resume_training is False


def test_v2_common_evaluation_and_counterfactual_runtime_contracts() -> None:
    if not LPS_STUDY.is_file():
        pytest.skip(f"Archived LPS study is unavailable: {LPS_STUDY}")
    run = import_legacy_checkpoint(**_paths("fold00"))
    study = load_lps_replay_study(run, LPS_STUDY)
    archived_config = json.loads(run.run_config_path.read_text(encoding="utf-8"))
    objectives = run.recipe.build_objectives(study, archived_config)
    plan = run.recipe.training_plan(study, archived_config)
    assert tuple(term.name for term in objectives) == (
        "checkpoint_geometry_mass",
        "weak_form_residual",
    )
    assert objectives[1].weight == pytest.approx(0.12)
    assert len(plan.stages) == 1
    assert plan.stages[0].precision == "bf16"
    assert plan.stages[0].batching.mode == "all_keys"
    assert plan.stages[0].optimizer.parameter_learning_rates == {
        "dynamics": 3e-4,
        "embeddings": 1e-3,
        "transformer": 5e-5,
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
    archived = pytest.importorskip("pandas").read_csv(
        ARCHIVE / "fold00/predicted_metrics_by_key_time.csv"
    )
    assert len(metrics) == len(archived) == 49
    assert metrics["measure_id"].tolist() == archived["measure_key"].tolist()
    before = run.model.embedding.embeddings.detach().clone()
    before_growth = run.model.embedding.growth_bias.detach().clone()
    counterfactual(run, study.measure_ids[0], particles=4, seed=29)
    assert torch.equal(run.model.embedding.embeddings, before)
    assert torch.equal(run.model.embedding.growth_bias, before_growth)
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
