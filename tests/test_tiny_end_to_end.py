from __future__ import annotations

import json
import pkgutil
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError

import credo
from credo.cli import main
from credo.counterfactual import counterfactual
from credo.io import RunConfig, load_config, load_data, write_canonical_dataset
from credo.model import CREDOModel
from credo.registry import get_recipe
from credo.runtime import TrainingEngine
from credo.training import Trainer


def _fit(config, data):
    return TrainingEngine().fit(get_recipe(config.recipe), data, config, device="cpu")


def test_tiny_gse_like_run_writes_five_artifacts(trained_run) -> None:
    output = trained_run.config.output
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [
            "manifest.json",
            "checkpoint.pt",
            "history.parquet",
            "metrics.parquet",
            "counterfactuals.parquet",
        ]
    )
    history = pd.read_parquet(output / "history.parquet")
    assert history["phase"].tolist() == ["state", "mass", "context"]
    assert history.loc[history["phase"].eq("state"), "bank_seen_fraction"].eq(0.0).all()
    assert (
        history.loc[history["phase"].isin({"mass", "context"}), "bank_seen_fraction"].eq(1.0).all()
    )
    assert history.loc[history["phase"].eq("state"), "validation_count_blocks"].eq(0).all()
    assert (
        history.loc[history["phase"].isin({"mass", "context"}), "validation_count_blocks"]
        .eq(2)
        .all()
    )
    assert history["validation_count_loss"].map(pd.notna).all()
    metrics = pd.read_parquet(output / "metrics.parquet")
    assert metrics.columns.tolist() == [
        "measure_id",
        "time_label",
        "endpoint_role",
        "validation_source",
        "geometry",
        "log_mass_error",
        "predicted_log_mass",
        "observed_log_mass",
        "ess_fraction",
        "max_weight_fraction",
    ]
    assert set(metrics["validation_source"]) == {"held_out"}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mass_semantics"] == "relative_within_group"
    assert manifest["mass_denominators"] == [
        "D1::Rest::eligible_guides",
        "D1::Stim48hr::eligible_guides",
        "D1::Stim8hr::eligible_guides",
        "D2::Rest::eligible_guides",
        "D2::Stim48hr::eligible_guides",
        "D2::Stim8hr::eligible_guides",
    ]
    assert manifest["claim_policy"]["absolute_growth"] is False
    assert manifest["claim_policy"]["abundance_claim"] == "relative_only"
    assert manifest["runtime"] == {
        "device": "cpu",
        "dtype": "float32",
        "accelerator": {"type": "cpu", "name": "CPU", "cuda_runtime": None},
    }
    assert manifest["bank_initialization"]["bank_seen_fraction"] == 1.0
    assert manifest["validation_split"]["strategy"] == "context_group_holdout"


def test_checkpoint_roundtrip_reproduces_predictions(trained_run, tiny_data, tmp_path) -> None:
    loaded = Trainer.load(
        trained_run.config.output / "checkpoint.pt",
        tiny_data,
        trained_run.config,
        device="cpu",
    )
    before = trained_run.metrics.reset_index(drop=True)
    after = loaded.metrics.reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)

    higher_resolution = Trainer.load(
        trained_run.config.output / "checkpoint.pt",
        tiny_data,
        trained_run.config,
        device="cpu",
        evaluation_overrides={"particles": 10, "measures_per_batch": 6},
    )
    assert higher_resolution.settings.evaluation.particles == 10
    assert higher_resolution.settings.evaluation.measures_per_batch == 6

    settings = trained_run.settings
    wrong_model = settings.model.model_copy(update={"hidden_dim": settings.model.hidden_dim + 8})
    wrong_settings = settings.model_copy(update={"model": wrong_model})
    wrong_config = trained_run.config.model_copy(update={"recipe_config": wrong_settings})
    with pytest.raises(ValueError, match="architecture disagrees"):
        Trainer.load(
            trained_run.config.output / "checkpoint.pt",
            tiny_data,
            wrong_config,
            device="cpu",
        )

    wrong_training = settings.training.model_copy(
        update={"steps_per_interval": settings.training.steps_per_interval + 1}
    )
    wrong_settings = settings.model_copy(update={"training": wrong_training})
    wrong_config = trained_run.config.model_copy(update={"recipe_config": wrong_settings})
    with pytest.raises(ValueError, match="run contract disagrees"):
        Trainer.load(
            trained_run.config.output / "checkpoint.pt",
            tiny_data,
            wrong_config,
            device="cpu",
        )

    changed_meta = tiny_data.measure_meta.copy()
    changed_meta.loc[0, "sample_id"] = "changed-sample"
    changed_data = replace(tiny_data, measure_meta=changed_meta)
    with pytest.raises(ValueError, match="run contract disagrees"):
        Trainer.load(
            trained_run.config.output / "checkpoint.pt",
            changed_data,
            trained_run.config,
            device="cpu",
        )

    payload = torch.load(
        trained_run.config.output / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    payload.pop("envelope")
    missing_envelope = tmp_path / "missing-envelope.pt"
    torch.save(payload, missing_envelope)
    with pytest.raises(ValueError, match="missing its envelope"):
        Trainer.load(missing_envelope, tiny_data, trained_run.config, device="cpu")

    corrupted = torch.load(
        trained_run.config.output / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    state_name = next(iter(corrupted["model_state"]))
    corrupted["model_state"][state_name] = corrupted["model_state"][state_name].clone()
    corrupted["model_state"][state_name].reshape(-1)[0] += 1
    corrupted_checkpoint = tmp_path / "corrupted-model.pt"
    torch.save(corrupted, corrupted_checkpoint)
    with pytest.raises(ValueError, match="semantic hash"):
        Trainer.load(corrupted_checkpoint, tiny_data, trained_run.config, device="cpu")


def test_trainer_rejects_noncanonical_model_architecture(tiny_config, tiny_data) -> None:
    settings = tiny_config.recipe_config
    model = CREDOModel(
        embedding_ids=tiny_data.embedding_ids,
        control_embedding_ids=tiny_data.control_embedding_ids,
        latent_dim=tiny_data.latent_dim,
        embedding_dim=settings.model.embedding_dim,
        n_programs=settings.model.n_programs,
        hidden_dim=settings.model.hidden_dim,
        context_mode=settings.model.context,
        growth_max=2.0,
    )
    recipe = get_recipe(tiny_config.recipe)
    with pytest.raises(ValueError, match="architecture disagrees"):
        Trainer.from_plan(
            tiny_data,
            model,
            tiny_config,
            recipe.training_plan(tiny_data, settings),
            recipe.build_objectives(tiny_data, settings),
            device="cpu",
        )


def test_growth_bound_is_configured_in_the_model(tiny_config, tiny_data) -> None:
    settings = tiny_config.recipe_config
    model_config = settings.model.model_copy(update={"growth_max": 7.5})
    epochs = settings.training.epochs.model_copy(update={"state": 1, "mass": 0, "context": 0})
    training = settings.training.model_copy(update={"epochs": epochs})
    loss = settings.loss.model_copy(update={"mass": 0.0, "count": 0.0})
    settings = settings.model_copy(
        update={"model": model_config, "training": training, "loss": loss}
    )
    config = tiny_config.model_copy(update={"recipe_config": settings})
    trainer = _fit(config, tiny_data)
    assert trainer.model.growth_max == 7.5


def test_checkpoint_holdout_masks_training_and_evaluation_times(tiny_config, tiny_data) -> None:
    raw = tiny_config.model_dump()
    recipe_config = raw["recipe_config"]
    recipe_config["training"]["epochs"] = {"state": 1, "mass": 0, "context": 0}
    recipe_config["training"]["particles"] = 4
    recipe_config["evaluation"]["particles"] = 4
    recipe_config["validation"] = {
        "strategy": "checkpoint",
        "values": ["Stim8hr"],
        "fraction": 0,
        "representation_scope": "shared",
    }
    recipe_config["loss"] = {"mass": 0.0, "count": 0.0, "sinkhorn_epsilon": 0.1}
    config = RunConfig.model_validate(raw)
    trainer = _fit(config, tiny_data)
    assert trainer.train_time_labels == ("Stim48hr",)
    assert trainer.validation_time_labels == ("Stim8hr",)
    assert set(trainer.metrics["time_label"]) == {"Stim8hr"}
    expected_train_observations = len(tiny_data.measures["Stim48hr"])
    assert trainer.history_rows[0]["train_observations"] == expected_train_observations


def test_seed_reproduces_independent_fits(tiny_config, tiny_data, tmp_path) -> None:
    raw = tiny_config.model_dump()
    recipe_config = raw["recipe_config"]
    recipe_config["training"]["epochs"] = {"state": 1, "mass": 0, "context": 0}
    recipe_config["training"].update({"particles": 4})
    recipe_config["evaluation"].update({"particles": 4})
    recipe_config["loss"] = {"mass": 0.0, "count": 0.0}
    raw["output"] = tmp_path / "unused"
    config = RunConfig.model_validate(raw)
    first = _fit(config, tiny_data)
    second = _fit(config, tiny_data)
    for name, value in first.model.state_dict().items():
        assert value.equal(second.model.state_dict()[name])
    pd.testing.assert_frame_equal(first.metrics, second.metrics, check_exact=True)


def test_save_refuses_files_outside_the_artifact_contract(trained_run) -> None:
    extra = trained_run.config.output / "extra.csv"
    extra.write_text("stale\n", encoding="utf-8")
    try:
        with pytest.raises(FileExistsError, match="five-artifact contract"):
            trained_run.save()
    finally:
        extra.unlink()


def test_canonical_writer_persists_validated_counts_and_rejects_stale_files(
    tiny_config, tiny_data, tmp_path
) -> None:
    support = ad.read_h5ad(tiny_config.data.support)
    support.obs["atom_weight"] = support.obs["atom_weight"].astype(str)
    support.obsm["X_credo"] = support.obsm["X_credo"].astype("float64")
    measure_meta = pd.read_parquet(tiny_config.data.measure_meta)
    masses = pd.read_parquet(tiny_config.data.masses)
    counts = pd.read_parquet(tiny_config.data.counts)
    counts["exposure"] = counts["exposure"].astype(str)
    counts["count"] = counts["count"].astype(str)
    output = tmp_path / "canonical"
    write_canonical_dataset(
        output,
        support=support,
        measure_meta=measure_meta,
        masses=masses,
        axis=tiny_data.axis,
        mass_semantics=tiny_data.mass_semantics,
        counts=counts,
    )
    written = pd.read_parquet(output / "counts.parquet")
    assert pd.api.types.is_numeric_dtype(written["exposure"])
    assert pd.api.types.is_numeric_dtype(written["count"])
    written_support = ad.read_h5ad(output / "support.h5ad")
    assert pd.api.types.is_numeric_dtype(written_support.obs["atom_weight"])
    assert written_support.obsm["X_credo"].dtype.name == "float32"
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["representation"]["backend"] == "frozen_latent"
    assert manifest["representation"]["fit_scope"] == "external"
    assert manifest["representation"]["included_samples"] == []
    assert manifest["representation"]["included_time_labels"] == []
    data_config = tiny_config.data.model_copy(
        update={
            "support": output / "support.h5ad",
            "measure_meta": output / "measure_meta.parquet",
            "masses": output / "masses.parquet",
            "counts": output / "counts.parquet",
            "dataset": output / "dataset.json",
        }
    )
    loaded = load_data(tiny_config.model_copy(update={"data": data_config}))
    assert loaded.representation.to_dict() == manifest["representation"]

    with pytest.raises(FileExistsError, match="outside its contract"):
        write_canonical_dataset(
            output,
            support=support,
            measure_meta=measure_meta,
            masses=masses,
            axis=tiny_data.axis,
            mass_semantics=tiny_data.mass_semantics,
            counts=None,
        )


def test_lazy_support_matches_eager_loading_and_bounds_cache(tiny_config) -> None:
    lazy_data_config = tiny_config.data.model_copy(update={"support_cache_size": 2})
    lazy = load_data(tiny_config.model_copy(update={"data": lazy_data_config}))
    eager_data_config = tiny_config.data.model_copy(update={"lazy_support": False})
    eager = load_data(tiny_config.model_copy(update={"data": eager_data_config}))

    assert getattr(lazy.measures, "is_lazy", False) is True
    assert len(lazy.measures._cache) == 0
    support_view = lazy.support
    assert len(lazy.measures._cache) == 0
    source_id = next(iter(support_view[lazy.axis.source]))
    np.testing.assert_array_equal(
        support_view[lazy.axis.source][source_id],
        eager.measures[lazy.axis.source][source_id].support,
    )
    assert len(lazy.measures._cache) == 1
    for label in lazy.axis.labels:
        for measure_id in tuple(lazy.measures[label])[:3]:
            lazy_measure = lazy.measures[label][measure_id]
            eager_measure = eager.measures[label][measure_id]
            np.testing.assert_array_equal(lazy_measure.support, eager_measure.support)
            np.testing.assert_allclose(lazy_measure.weights, eager_measure.weights)
            assert lazy_measure.total_mass == eager_measure.total_mass
    assert len(lazy.measures._cache) <= 2
    lazy.measures.close()
    assert lazy.measures._handle is None


def test_lazy_support_validation_scans_for_nonfinite_values(tiny_config, tmp_path) -> None:
    support = ad.read_h5ad(tiny_config.data.support)
    latent = np.asarray(support.obsm[tiny_config.data.latent_key]).copy()
    latent[-1, -1] = np.nan
    support.obsm[tiny_config.data.latent_key] = latent
    corrupted_path = tmp_path / "nonfinite-support.h5ad"
    support.write_h5ad(corrupted_path)
    data_config = tiny_config.data.model_copy(update={"support": corrupted_path})

    with pytest.raises(ValueError, match="contains non-finite values"):
        load_data(tiny_config.model_copy(update={"data": data_config}))


def test_counterfactual_persistence_preserves_prior_rows(trained_run, tiny_data) -> None:
    path = trained_run.config.output / "counterfactuals.parquet"
    manifest_path = trained_run.config.output / "manifest.json"
    original_artifact = path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    noncontrols = tiny_data.measure_meta.loc[
        ~tiny_data.measure_meta["is_control"], "measure_id"
    ].tolist()
    columns = [
        "measure_id",
        "time_label",
        "context_policy",
        "delta_log_mass",
        "mean_shift_l2",
        "energy_distance",
        "context_dependence_shift",
        "factual_ess",
        "reference_ess",
    ]
    prior = {
        "measure_id": noncontrols[0],
        "time_label": tiny_data.axis.labels[1],
        "context_policy": "clamped",
        "delta_log_mass": 0.0,
        "mean_shift_l2": 0.0,
        "energy_distance": 0.0,
        "context_dependence_shift": 0.0,
        "factual_ess": 1.0,
        "reference_ess": 1.0,
    }
    try:
        pd.DataFrame([prior], columns=columns).to_parquet(path, index=False)
        loaded = Trainer.load(
            trained_run.config.output / "checkpoint.pt",
            tiny_data,
            trained_run.config,
            device="cpu",
        )
        counterfactual(loaded, noncontrols[1], context_policy="self_consistent")
        persisted = pd.read_parquet(path)
        keys = set(
            zip(
                persisted["measure_id"],
                persisted["time_label"],
                persisted["context_policy"],
                strict=False,
            )
        )
        assert (prior["measure_id"], prior["time_label"], prior["context_policy"]) in keys
        assert any(key[0] == noncontrols[1] for key in keys)
    finally:
        path.write_bytes(original_artifact)
        manifest_path.write_bytes(original_manifest)


def test_installed_surface_is_compact_and_recipe_runtime_is_explicit() -> None:
    modules = {module.name for module in pkgutil.iter_modules(credo.__path__)}
    assert modules == {
        "artifacts",
        "cli",
        "contracts",
        "counterfactual",
        "data",
        "evaluation",
        "io",
        "model",
        "objective",
        "particles",
        "recipes",
        "registry",
        "runtime",
        "training",
    }
    assert len(credo.__all__) == 18


def test_cli_validate_and_summarize(tiny_config, trained_run, capsys, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config.model_dump(mode="json")), encoding="utf-8")
    assert main(["validate", str(config_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["measure_count"] == 12
    assert main(["summarize", str(trained_run.config.output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["metric_rows"] == len(trained_run.metrics)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        main(["run", str(config_path), "--seed", "-1"])


def test_config_rejects_duplicate_yaml_keys(tmp_path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text("output: first\noutput: second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key 'output'"):
        load_config(config_path)
