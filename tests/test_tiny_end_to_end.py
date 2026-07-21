from __future__ import annotations

import json
import pkgutil
from dataclasses import replace

import anndata as ad
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

import credo
from credo.cli import main
from credo.counterfactual import counterfactual
from credo.io import RunConfig, load_config, write_canonical_dataset
from credo.model import CREDOModel
from credo.training import Trainer


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
    assert manifest["bank_initialization"]["bank_seen_fraction"] == 1.0
    assert manifest["validation_split"]["strategy"] == "context_group_holdout"


def test_checkpoint_roundtrip_reproduces_predictions(trained_run, tiny_data) -> None:
    loaded = Trainer.load(
        trained_run.config.output / "checkpoint.pt",
        tiny_data,
        trained_run.config,
        device="cpu",
    )
    before = trained_run.metrics.reset_index(drop=True)
    after = loaded.metrics.reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)

    wrong_model = trained_run.config.model.model_copy(
        update={"hidden_dim": trained_run.config.model.hidden_dim + 8}
    )
    wrong_config = trained_run.config.model_copy(update={"model": wrong_model})
    with pytest.raises(ValueError, match="architecture disagrees"):
        Trainer.load(
            trained_run.config.output / "checkpoint.pt",
            tiny_data,
            wrong_config,
            device="cpu",
        )

    wrong_training = trained_run.config.training.model_copy(
        update={"steps_per_interval": trained_run.config.training.steps_per_interval + 1}
    )
    wrong_config = trained_run.config.model_copy(update={"training": wrong_training})
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


def test_trainer_rejects_noncanonical_model_architecture(tiny_config, tiny_data) -> None:
    model = CREDOModel(
        embedding_ids=tiny_data.embedding_ids,
        control_embedding_ids=tiny_data.control_embedding_ids,
        latent_dim=tiny_data.latent_dim,
        embedding_dim=tiny_config.model.embedding_dim,
        n_programs=tiny_config.model.n_programs,
        hidden_dim=tiny_config.model.hidden_dim,
        context_mode=tiny_config.model.context,
        growth_max=2.0,
    )
    with pytest.raises(ValueError, match="architecture disagrees"):
        Trainer.fit(tiny_data, model, tiny_config, device="cpu")


def test_seed_reproduces_independent_fits(tiny_config, tiny_data, tmp_path) -> None:
    raw = tiny_config.model_dump()
    raw["training"]["epochs"] = {"state": 1, "mass": 0, "context": 0}
    raw["training"].update({"particles": 4, "eval_particles": 4})
    raw["loss"] = {"mass": 0.0, "count": 0.0}
    raw["output"] = tmp_path / "unused"
    config = RunConfig.model_validate(raw)
    first = Trainer.fit(tiny_data, None, config, device="cpu")
    second = Trainer.fit(tiny_data, None, config, device="cpu")
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


def test_installed_surface_has_fewer_than_ten_modules() -> None:
    modules = {module.name for module in pkgutil.iter_modules(credo.__path__)}
    assert modules == {
        "cli",
        "contracts",
        "counterfactual",
        "io",
        "model",
        "objective",
        "particles",
        "training",
    }
    assert len(credo.__all__) == 9


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
