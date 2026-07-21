from __future__ import annotations

import json
import pkgutil

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

import credo
from credo.cli import main
from credo.io import RunConfig, load_config
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
