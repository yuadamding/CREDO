"""Synthetic-only actual fits proving the terminal/no-assessment boundary."""
# ruff: noqa: E402
from dataclasses import replace
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source/src"), str(ROOT / "tests")]

import numpy as np
import pandas as pd
import pytest
import torch

from credo.data import write_study
from credo.io import RunConfig
from credo.recipes.compact_sde_v3.training import Trainer
from credo.runtime import TrainingEngine
from native_study import build_native_study, compile_development
from test_native_study import fixture
from terminal_fitting import POLICY_ID, TerminalFittingTrainer, _validate_architecture_fit, recipe


def setup_run(tmp_path):
    bundle = build_native_study(**fixture())
    manifest = write_study(bundle.study, tmp_path / "native")
    config = RunConfig.model_validate(dict(
        study=manifest,
        output=tmp_path / "run",
        recipe_config=dict(
            model=dict(context="none", hidden_dim=16, embedding_dim=4, n_programs=2,
                       trunk_kind="residual_mlp", trunk_depth=3),
            training=dict(epochs=dict(state=1, mass=1, context=0), particles=4,
                          catalog_refresh_particles=4, measures_per_batch=2,
                          steps_per_interval=2, checkpoint_selection="last", seed=7),
            evaluation=dict(particles=4, measures_per_batch=2),
            loss=dict(mass=0, count=1, count_likelihood="categorical_ce", mass_increment="particle_log_mass"),
        ),
        split=dict(strategy="subject", train_values=("D3", "D4"), validation_values=("D2",), representation_scope="shared"),
    ))
    problem, _ = compile_development(bundle, config.recipe_config)
    return problem, config


def test_real_terminal_fit_saves_without_scoring_holdout(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("The inherited holdout scoring path was entered.")

    monkeypatch.setattr(Trainer, "_evaluate_ids", forbidden)
    monkeypatch.setattr(Trainer, "_validation_count_loss", forbidden)
    trainer = TrainingEngine().fit(recipe, problem, config, device="cpu")
    assert isinstance(trainer, TerminalFittingTrainer)
    assert trainer.completed_epochs == 2
    assert trainer._deferred_terminal_calls == 1
    assert trainer.metrics.empty
    assert len(trainer.history_rows) == 2
    assert all(not row["validation_monitored"] for row in trainer.history_rows)
    assert all(row["validation_observations"] == row["validation_count_blocks"] == 0 for row in trainer.history_rows)
    assert all(np.isnan(row["validation_objective"]) for row in trainer.history_rows)
    assert trainer.bank.mass_increment == "particle_log_mass"
    assert not trainer.log_count_concentration.requires_grad
    assert trainer.execution_trace[-1]["event"] == POLICY_ID
    assert trainer.execution_trace[-1]["assessment_performed"] is False
    trainer.bank.assert_complete()
    # A genuine synthetic optimization ran, not just an empty interface test.
    for head in ("drift_", "sigma_", "growth_"):
        assert any(parameter.grad is not None and float(parameter.grad.abs().max()) > 0
                   for name, parameter in trainer.model.named_parameters() if name.startswith(head))
    output = trainer.save()
    checkpoint = torch.load(output / "state/checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["completed_epochs"] == 2
    assert checkpoint["execution_trace"][-1]["terminal_evaluation_deferred"] is True
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(checkpoint["model_state"][name], value, rtol=0, atol=0)
    for name in ("predictions", "metrics", "counterfactuals"):
        assert pd.read_parquet(output / f"tables/{name}.parquet").empty
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["scientific_assessment"] == "not_performed_fitting_only"
    assert manifest["study_execution_policy"] == POLICY_ID
    with pytest.raises(FileExistsError, match="never overwrite"):
        trainer.save()
    for operation in (trainer.evaluate, trainer.evaluate_runtime, trainer._evaluate_ids,
                      trainer._validation_count_loss, trainer._validation_score, trainer.load):
        with pytest.raises(RuntimeError):
            operation()


def test_refresh_is_bounded_fitting_only_and_restores_training_mode(tmp_path, monkeypatch):
    import terminal_fitting

    problem, config = setup_run(tmp_path)
    observed = []
    original = terminal_fitting.sample_initial_particles

    def tracked(data, ids, *args, **kwargs):
        assert data is problem.training
        observed.append(tuple(ids))
        return original(data, ids, *args, **kwargs)

    monkeypatch.setattr(terminal_fitting, "REFRESH_GUIDES", 1)
    monkeypatch.setattr(terminal_fitting, "sample_initial_particles", tracked)
    trainer = TrainingEngine().fit(recipe, problem, config, device="cpu")
    assert observed and all(len(ids) == 1 for ids in observed)
    assert set(key for ids in observed for key in ids) == set(problem.training.measure_ids)
    assert all(not key.startswith("D2::") for ids in observed for key in ids)
    trainer.model.train()
    trainer._refresh_bank_for(trainer.data, trainer.bank, epoch=2)
    assert trainer.model.training
    trainer.model.eval()
    trainer._refresh_bank_for(trainer.data, trainer.bank, epoch=2)
    assert not trainer.model.training
    trainer.bank.assert_complete()
    with pytest.raises(ValueError, match="Only the fitting bank"):
        trainer._refresh_bank_for(trainer.validation_data, trainer.validation_bank, epoch=2)


@pytest.mark.parametrize("section,key,value", [
    ("training", "checkpoint_selection", "validation_best"),
    ("model", "context", "catalog_bank"),
    ("loss", "mass", 1),
    ("loss", "count", 0),
    ("loss", "count_likelihood", "dirichlet_multinomial"),
    ("loss", "mass_increment", "integrated_mean_growth"),
])
def test_policy_changes_rejected_before_fitting(tmp_path, section, key, value):
    problem, config = setup_run(tmp_path)
    raw = config.model_dump()
    raw["recipe_config"][section][key] = value
    bad = RunConfig.model_validate(raw)
    with pytest.raises(ValueError):
        _validate_architecture_fit(problem, bad)
    assert not Path(config.output).exists()


def test_donor_and_measure_relabeling_rejected(tmp_path):
    problem, config = setup_run(tmp_path)
    metadata = problem.training.measure_meta.copy()
    metadata.loc[metadata.sample_id == "D3", "sample_id"] = "D1"
    altered = replace(problem, training=replace(problem.training, measure_meta=metadata))
    with pytest.raises(ValueError, match="D3/D4"):
        _validate_architecture_fit(altered, config)
    # A prefix cannot be rewritten merely by changing a donor metadata column.
    metadata = problem.training.measure_meta.copy()
    metadata["sample_id"] = metadata.sample_id.map({"D3": "D4", "D4": "D3"})
    altered = replace(problem, training=replace(problem.training, measure_meta=metadata))
    with pytest.raises(ValueError, match="Measure identity"):
        _validate_architecture_fit(altered, config)


def test_preexisting_output_fails_before_fitting(tmp_path):
    problem, config = setup_run(tmp_path)
    config.output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        TrainingEngine().fit(recipe, problem, config, device="cpu")


def test_unfitted_empty_results_cannot_publish():
    trainer = object.__new__(TerminalFittingTrainer)
    with pytest.raises(RuntimeError, match="complete terminal fit"):
        trainer.save()


def test_unexpected_early_evaluation_fails_instead_of_returning_empty_success(tmp_path, monkeypatch):
    problem, config = setup_run(tmp_path)

    def changed_core(self):
        self.evaluate()

    monkeypatch.setattr(Trainer, "_fit", changed_core)
    with pytest.raises(RuntimeError, match="deferred"):
        TrainingEngine().fit(recipe, problem, config, device="cpu")
    assert not config.output.exists()


def test_failed_refresh_restores_model_mode_and_cannot_publish_fit(tmp_path, monkeypatch):
    import terminal_fitting

    problem, config = setup_run(tmp_path)
    models = []
    original_rollout = terminal_fitting.rollout

    def failed_refresh(model, *args, **kwargs):
        models.append(model)
        raise RuntimeError("synthetic refresh failure")

    monkeypatch.setattr(terminal_fitting, "rollout", failed_refresh)
    with pytest.raises(RuntimeError, match="synthetic refresh failure"):
        TrainingEngine().fit(recipe, problem, config, device="cpu")
    assert len(models) == 1 and models[0].training
    assert not config.output.exists()
    monkeypatch.setattr(terminal_fitting, "rollout", original_rollout)
