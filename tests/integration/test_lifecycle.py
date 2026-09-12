from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from credo_count_sde_v4 import api
from credo_count_sde_v4.compile import load_compiled_problem
from credo_count_sde_v4.contracts import LifecycleState, RunIntent
from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.inference import open_inference_run
from credo_count_sde_v4.model import CountSDEModel
from credo_count_sde_v4.persistence import LifecycleLedger, load_tensor_file
from credo_count_sde_v4.prepare.pipeline import load_config
from credo_count_sde_v4.store import CountStore
from credo_count_sde_v4.synthetic import create_synthetic_project
from credo_count_sde_v4.training import load_training_state
from credo_count_sde_v4.training.calibration import run_state_selection_calibration
from credo_count_sde_v4.training.trainer import (
    _all_state_split,
    _complete_state_objective_values,
    _post_selection_refit,
    _tensor_problem,
    train_model,
)


def _mark_pooled_pilot(payload: dict[str, object]) -> None:
    payload["pooled_estimand"] = "pooled_known_target_heldout_guide"
    payload["outer_fold_id"] = "synthetic-outer-fold-0"
    payload["inner_split_id"] = "synthetic-inner-split-0"
    payload["pooled_outer_fold_ids"] = ["synthetic-outer-fold-0", "synthetic-outer-fold-1"]
    payload["pooled_inner_split_ids"] = ["synthetic-inner-split-0", "synthetic-inner-split-1"]
    model = payload["model"]
    assert isinstance(model, dict)
    model["pool_count"] = 1
    training = payload["training"]
    assert isinstance(training, dict)
    seed = int(training["seed"])
    payload["pooled_optimization_seeds"] = [seed, seed + 1, seed + 2]
    training.update(
        {
            "state_full_batch": True,
            "pilot_device_type": "cpu",
            "state_split_seed": 7001,
            "initialization_seed": 7002,
        }
    )


def _configure_pilot_calibration(
    config: Path,
    *,
    updates: list[int],
    target_margin: float,
    interaction_margin: float,
) -> None:
    payload = yaml.safe_load(config.read_text())
    _mark_pooled_pilot(payload)
    payload["state_selection_calibration"] = (
        "work/input/state-calibration/state-selection-calibration.json"
    )
    payload["training"]["state_checkpoint_updates"] = updates
    payload["training"]["state_validation_target_minimum_improvement"] = target_margin
    payload["training"]["state_validation_interaction_minimum_improvement"] = interaction_margin
    config.write_text(yaml.safe_dump(payload, sort_keys=False))


def _run_pilot_calibration(config: Path) -> None:
    run_state_selection_calibration(
        config,
        config.parent / "work/input/state-calibration",
        repeats_per_null=119,
        permutation_seed_start=100_000,
        optimizer_seed_start=200_000,
        initialization_seed_start=300_000,
    )


@pytest.mark.parametrize("intent", list(RunIntent))
def test_complete_synthetic_lifecycle(tmp_path: Path, intent: RunIntent) -> None:
    config = create_synthetic_project(tmp_path / intent.value, intent=intent, updates=6)
    api.prepare(config)
    api.compile_run(config)
    api.train(config, device="cpu")
    api.finalize(config)
    api.evaluate(config)
    sealed = api.seal(config)
    assert (
        LifecycleLedger(sealed.parent / "ledger" / "events.jsonl").state() is LifecycleState.SEALED
    )
    result = api.verify(sealed, level="full")
    assert result["reload"]["finite_mass"]
    metrics = json.loads((sealed.parent / "evaluation" / "metrics.json").read_text())
    assert metrics["evaluable_series"] == metrics["total_series"] == 6
    inference = json.loads((sealed.parent / "inference" / "inference.json").read_text())
    audit = json.loads((sealed.parent / "evaluation" / "audit.json").read_text())
    assert inference["selected_family"] == "configured_checkpoint"
    assert inference["selection"]["sha256"]
    assert audit["selected_family"] == "configured_checkpoint"


def test_prepare_rejects_reordered_feature_contract(tmp_path: Path) -> None:
    config = create_synthetic_project(tmp_path / "feature-order", updates=2)
    feature_path = config.parent / "work/input/features.json"
    features = json.loads(feature_path.read_text())
    feature_path.write_text(json.dumps(list(reversed(features))))
    with pytest.raises(ContractError, match="Ordered feature contract"):
        api.prepare(config)


def test_compiled_npz_explicit_keywords_preserve_archive_payload(tmp_path, monkeypatch):
    import io

    from credo_count_sde_v4.compile import compiler

    config = create_synthetic_project(tmp_path / "npz-keywords", updates=1)
    api.prepare(config)
    original = np.savez
    captured = {}

    def capture(handle, **arrays):
        captured.update({key: value.copy() for key, value in arrays.items()})
        original(handle, **arrays)

    monkeypatch.setattr(compiler.np, "savez", capture)
    api.compile_run(config)
    expected_names = [
        "source_z",
        "terminal_z",
        "target_index",
        "pool_index",
        "is_control",
        "duration",
        "grid_steps",
        "grid_step_size",
        "source_counts",
        "terminal_counts",
        "series_ids",
    ]
    assert list(captured) == expected_names
    # Historical dynamic expansion and new fixed keywords have identical members,
    # values and dtypes. ZIP timestamps are deliberately not a byte-equality claim.
    historical = io.BytesIO()
    original(historical, **captured)
    historical.seek(0)
    contract, reloaded = load_compiled_problem(config.parent / "work")
    assert contract.compiled_problem_hash == compiler._problem_hash(captured)
    with np.load(historical, allow_pickle=False) as old:
        assert old.files == expected_names
        assert list(reloaded) == expected_names
        for key in expected_names:
            np.testing.assert_array_equal(reloaded[key], old[key])
            assert reloaded[key].dtype == old[key].dtype


def test_trained_gene_decoder_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = create_synthetic_project(tmp_path / "decoder", intent=RunIntent.COUNT_STATE, updates=3)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["gene_decoder_features"] = 12
    payload["model"]["gene_decoder_hidden_dim"] = 8
    payload["model"]["terminal_anchor_drift"] = True
    payload["model"]["source_carryover_alpha"] = 0.0
    payload["model"]["target_anchor_weight"] = 0.0
    payload["training"]["gene_decoder_batch_size"] = 64
    payload["training"]["gene_decoder_loss_weight"] = 0.05
    payload["training"]["selected_update"] = None
    payload["training"]["checkpoint_every"] = 1
    payload["training"]["gene_decoder_validation_fraction"] = 0.2
    payload["training"]["gene_decoder_validation_max_rows"] = 16
    payload["training"]["checkpoint_selection"] = "minimum_gene_decoder_validation"
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    calls = 0
    original_rows = CountStore.rows

    def tracked_rows(self: CountStore, row_ids: np.ndarray):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original_rows(self, row_ids)

    monkeypatch.setattr(CountStore, "rows", tracked_rows)
    api.train(config, device="cpu")
    assert calls == 1
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["policy"] == "minimum_gene_decoder_validation"
    assert selection["selected_update"] in {1, 2, 3}
    assert np.isfinite(selection["score"])
    _, arrays, model, _ = load_training_state(config, device="cpu")
    np.testing.assert_allclose(
        model.terminal_anchor.detach().numpy(), arrays["terminal_z"].mean(axis=0), atol=1e-7
    )
    api.finalize(config)
    run = open_inference_run(config.parent / "work/inference", device="cpu", verify="full")
    mean, _, _ = run.terminal(particles=2)
    composition = run.decode_composition(mean)
    assert composition.shape == (6, 12)
    np.testing.assert_allclose(composition.sum(axis=1), 1.0, atol=1e-6)


def test_state_validation_selects_a_scientific_checkpoint_and_excludes_capacity_probe(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(tmp_path / "state-selection", updates=4)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["state_dependent_drift"] = True
    payload["training"].update(
        {
            "checkpoint_every": 2,
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_validation_max_per_target": 1,
            "support_weight_power": 0.5,
            "target_drift_penalty": 0.25,
            "source_drift_penalty": 0.1,
            "checkpoint_selection": "minimum_state_validation",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    train_model(config, device="cpu", stop_after=1)
    api.resume(config, device="cpu")
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["policy"] == "minimum_state_validation"
    assert selection["candidate_updates"] == [2, 4]
    assert selection["selected_update"] in {2, 4}
    assert np.isfinite(selection["score"])


def test_joint_decoder_training_updates_null_nested_state_channel(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "joint-state-decoder", intent=RunIntent.COUNT_STATE, updates=4
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_conditioned_anchor": True,
            "source_anchor_residual_scale": 0.5,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
            "gene_decoder_features": 12,
            "gene_decoder_hidden_dim": 8,
        }
    )
    payload["training"].update(
        {
            "checkpoint_every": 2,
            "selected_update": None,
            "gene_decoder_batch_size": 64,
            "gene_decoder_loss_weight": 0.05,
            "gene_decoder_validation_fraction": 0.2,
            "train_state_with_gene_decoder": True,
            "state_validation_fraction": 0.34,
            "state_validation_max_per_target": 1,
            "source_drift_penalty": 0.01,
            "checkpoint_selection": "minimum_state_validation",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    api.prepare(config)
    api.compile_run(config)
    api.train(config, device="cpu")
    _, _, model, _ = load_training_state(config, device="cpu")
    assert model.source_anchor_output is not None
    assert torch.count_nonzero(model.source_anchor_output.weight).item() > 0
    selection = json.loads((config.parent / "work/training/selection.json").read_text())
    assert selection["policy"] == "minimum_state_validation"
    assert selection["candidate_updates"] == [2, 4]


def test_null_guarded_interaction_persists_zero_and_refits_selected_null(
    tmp_path: Path,
) -> None:
    config = create_synthetic_project(
        tmp_path / "null-guarded-interaction",
        intent=RunIntent.COUNT_STATE,
        updates=4,
        pooled=True,
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "source_target_interaction_scale": 0.25,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "checkpoint_every": 2,
            "state_checkpoint_updates": [1, 2],
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_validation_max_per_target": 1,
            "source_target_main_penalty": 1.0,
            "source_target_interaction_penalty": 1.0,
            "state_validation_target_minimum_improvement": 1_000.0,
            "state_validation_interaction_minimum_improvement": 1_000.0,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    _configure_pilot_calibration(
        config, updates=[1, 2], target_margin=1_000.0, interaction_margin=1_000.0
    )
    api.prepare(config)
    _run_pilot_calibration(config)
    calibration = json.loads(
        (
            config.parent / "work/input/state-calibration/state-selection-calibration.json"
        ).read_text()
    )
    results = json.loads(
        (
            config.parent / "work/input/state-calibration/state-selection-calibration-results.json"
        ).read_text()
    )
    assert calibration["repeated_per_null"] == 119
    assert len(results["rows"]) == 3 * calibration["repeated_per_null"]
    assert calibration["false_target_main_count"] == 0
    assert calibration["false_interaction_count"] == 0
    assert calibration["false_joint_interaction_count"] == 0
    assert calibration["false_interaction_rate_upper_bound"] < 0.05
    assert all(not row["false_interaction_selected"] for row in results["rows"])
    api.compile_run(config)
    api.train(config, device="cpu")
    training = config.parent / "work/training"
    selection = json.loads((training / "selection.json").read_text())
    assert selection["candidate_updates"] == [0, 1, 2]
    assert selection["inner_selected_checkpoint_id"] != selection["selected_checkpoint_id"]
    assert selection["selected_update"] == 0
    assert selection["selected_family"] == "global_terminal_null"
    assert selection["post_selection_refit"] is True
    assert selection["selected_checkpoint_relative_uri"].endswith(
        "refit/checkpoints/generation-000000000"
    )
    _, arrays, model, checkpoint = load_training_state(config, device="cpu")
    assert checkpoint.checkpoint_id == selection["selected_checkpoint_id"]
    assert checkpoint.update == 0
    target = arrays["target_index"]
    control = arrays["is_control"].astype(bool)
    expected_anchor = np.mean(
        [
            arrays["terminal_z"][(target == value) & ~control].mean(axis=0)
            for value in np.unique(target[~control])
        ],
        axis=0,
    )
    np.testing.assert_allclose(model.terminal_anchor.detach().numpy(), expected_anchor, atol=1e-7)
    assert model.source_target_output is not None
    assert torch.count_nonzero(model.source_target_output.weight).item() == 0
    api.finalize(config)
    inference = json.loads((config.parent / "work/inference/inference.json").read_text())
    assert inference["selected_checkpoint_id"] == selection["selected_checkpoint_id"]
    assert inference["selected_family"] == "global_terminal_null"
    assert inference["selection"]["sha256"]
    checkpoint_manifest = json.loads(
        (
            training
            / selection["selected_checkpoint_relative_uri"].removeprefix("training/")
            / "checkpoint.json"
        ).read_text()
    )
    assert checkpoint_manifest["parent_checkpoint_id"] is None
    assert (
        checkpoint_manifest["selection_source_checkpoint_id"]
        == selection["inner_selected_checkpoint_id"]
    )


def test_source_target_pilot_requires_a_positive_selection_margin(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "zero-selection-margin",
        intent=RunIntent.COUNT_STATE,
        updates=2,
        pooled=True,
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["state_selection_calibration"] = "work/input/unused-calibration.json"
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_checkpoint_updates": [1, 2],
            "source_target_main_penalty": 1.0,
            "source_target_interaction_penalty": 1.0,
            "state_validation_target_minimum_improvement": 0.01,
            "state_validation_interaction_minimum_improvement": 0.0,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="positive interaction margin"):
        api.prepare(config)


def test_source_target_pilot_rejects_untrained_decoder_architecture(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "random-decoder-pilot",
        intent=RunIntent.COUNT_STATE,
        updates=2,
        pooled=True,
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
            "gene_decoder_features": 12,
            "gene_decoder_hidden_dim": 8,
        }
    )
    payload["state_selection_calibration"] = "work/input/unused-calibration.json"
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_checkpoint_updates": [1, 2],
            "source_target_main_penalty": 1.0,
            "source_target_interaction_penalty": 1.0,
            "state_validation_target_minimum_improvement": 0.01,
            "state_validation_interaction_minimum_improvement": 0.01,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    _mark_pooled_pilot(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="disable the gene decoder entirely"):
        api.prepare(config)


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        (
            "model",
            {"shared_diffusion": True, "shared_diffusion_inner_validation_pass": True},
            "forbid uncalibrated channels",
        ),
        (
            "model",
            {"centered_selection": True, "selection_inner_validation_pass": True},
            "forbid uncalibrated channels",
        ),
        ("training", {"support_weight_power": 0.5}, "support_weight_power=0"),
    ],
)
def test_source_target_pilot_rejects_uncalibrated_channels(
    tmp_path: Path,
    section: str,
    changes: dict[str, object],
    message: str,
) -> None:
    config = create_synthetic_project(
        tmp_path / f"forbidden-{next(iter(changes))}",
        intent=RunIntent.COUNT_STATE,
        updates=2,
        pooled=True,
    )
    payload = yaml.safe_load(config.read_text())
    payload["state_selection_calibration"] = "work/input/unused-calibration.json"
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_checkpoint_updates": [1, 2],
            "source_target_main_penalty": 1.0,
            "source_target_interaction_penalty": 1.0,
            "state_validation_target_minimum_improvement": 0.01,
            "state_validation_interaction_minimum_improvement": 0.01,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    payload[section].update(changes)
    _mark_pooled_pilot(payload)
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match=message):
        api.prepare(config)


def test_source_target_pilot_rejects_margin_not_bound_by_calibration(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "mismatched-calibration",
        intent=RunIntent.COUNT_STATE,
        updates=2,
        pooled=True,
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_checkpoint_updates": [1, 2],
            "source_target_main_penalty": 1.0,
            "source_target_interaction_penalty": 1.0,
            "state_validation_target_minimum_improvement": 0.01,
            "state_validation_interaction_minimum_improvement": 0.02,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    _configure_pilot_calibration(
        config, updates=[1, 2], target_margin=1_000.0, interaction_margin=1_000.0
    )
    api.prepare(config)
    _run_pilot_calibration(config)
    payload = yaml.safe_load(config.read_text())
    payload["training"]["state_validation_interaction_minimum_improvement"] = 1_001.0
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ContractError, match="Interaction selection margin differs"):
        api.compile_run(config)


def test_target_only_candidate_is_materialized_by_post_selection_refit(tmp_path: Path) -> None:
    config = create_synthetic_project(
        tmp_path / "target-only-refit",
        intent=RunIntent.COUNT_STATE,
        updates=2,
        pooled=True,
    )
    payload = yaml.safe_load(config.read_text())
    payload["model"].update(
        {
            "terminal_anchor_drift": True,
            "source_carryover_alpha": 0.0,
            "source_target_interaction_rank": 2,
            "trainable_terminal_anchor": False,
            "trainable_target_anchor": False,
        }
    )
    payload["training"].update(
        {
            "selected_update": None,
            "state_validation_fraction": 0.34,
            "state_checkpoint_updates": [1, 2],
            "source_target_main_penalty": 1.0,
            "source_target_interaction_penalty": 1.0,
            "state_validation_target_minimum_improvement": 0.01,
            "state_validation_interaction_minimum_improvement": 0.01,
            "post_selection_state_refit": True,
            "checkpoint_selection": "minimum_state_validation_null_guarded",
        }
    )
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    _configure_pilot_calibration(
        config, updates=[1, 2], target_margin=1_000.0, interaction_margin=1_000.0
    )
    api.prepare(config)
    _run_pilot_calibration(config)
    api.compile_run(config)
    workspace = config.parent / "work"
    contract, arrays = load_compiled_problem(workspace)
    training = workspace / "training"
    training.mkdir()
    selection = {
        "schema_version": 1,
        "selection_id": "pending",
        "compiled_run_id": contract.compiled_run_id,
        "policy": "minimum_state_validation_null_guarded",
        "metric": "state_validation_full_interaction_rmse",
        "score": 0.5,
        "selected_update": 0,
        "selected_checkpoint_id": "0" * 64,
        "selected_checkpoint_relative_uri": "training/checkpoints/generation-000000000",
        "candidate_updates": [0, 1, 2],
        "selected_family": "selected_training_only_target_main",
        "global_null_score": 0.6,
        "shrunk_target_only_score": 0.5,
        "best_target_only_score": 0.5,
        "best_target_only_baseline": "shrunk_target_only",
        "best_noninteraction_score": 0.5,
        "best_noninteraction_baseline": "shrunk_target_only",
        "interaction_score": 0.5,
        "interaction_incremental_gain": 0.0,
        "target_incremental_gain": 0.1,
        "target_minimum_required_improvement": 1_000.0,
        "interaction_minimum_required_improvement": 1_000.0,
        "selection_calibration_hash": contract.state_selection_calibration_hash,
    }
    _post_selection_refit(
        workspace,
        training,
        contract,
        arrays,
        load_config(config),
        torch.device("cpu"),
        selection,
    )
    state = load_tensor_file(training / "refit/checkpoints/generation-000000000/model.safetensors")
    assert 0.0 < float(state["source_target_main_weight"]) <= 1.0
    assert torch.count_nonzero(state["source_target_main_offset"]).item() > 0
    assert torch.count_nonzero(state["source_target_output.weight"]).item() == 0
    training_state = json.loads(
        (training / "refit/checkpoints/generation-000000000/training-state.json").read_text()
    )
    resolved = load_config(config)
    refit_model = CountSDEModel(resolved.model, resolved.intent)
    refit_model.load_state_dict(state)
    tensor_problem = _tensor_problem(arrays, torch.device("cpu"))
    refit_split = _all_state_split(arrays, resolved, torch.device("cpu"))
    empirical, regularized = _complete_state_objective_values(
        refit_model, tensor_problem, refit_split, resolved
    )
    assert training_state["loss"] == pytest.approx(regularized)
    refit = json.loads((training / "refit/refit.json").read_text())
    assert refit["unregularized_target_balanced_state_mse"] == pytest.approx(empirical)
    assert refit["regularized_optimization_objective"] == pytest.approx(regularized)
