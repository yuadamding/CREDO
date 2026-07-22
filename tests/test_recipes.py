from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from credo import evaluate
from credo.artifacts import CheckpointEnvelope, CheckpointMode, tensor_state_sha256
from credo.particles import (
    CatalogContextProvider,
    NoContextProvider,
    SelfConsistentContextProvider,
)
from credo.recipes.compact_v3 import CompactSDEV3Recipe
from credo.registry import RecipeUnavailableError, available_recipes, get_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_compact_v3_golden_run_is_unchanged(trained_run) -> None:
    golden = json.loads((ROOT / "tests/golden/compact_v3.json").read_text(encoding="utf-8"))
    metrics = trained_run.metrics.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    state_hash = tensor_state_sha256(trained_run.model.state_dict())
    assert state_hash == golden["model_state_sha256"]
    assert hashlib.sha256(metrics).hexdigest() == golden["metrics_sha256"]
    payload = torch.load(
        trained_run.config.output / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    envelope = CheckpointEnvelope.from_dict(payload["envelope"])
    assert payload["schema_version"] == 2
    assert envelope.mode is CheckpointMode.INFERENCE_ONLY
    assert envelope.recipe["id"] == "credo.compact_sde_v3"
    assert envelope.recipe["version"] == "3.0"
    assert envelope.study_contract == payload["run_contract"]
    assert envelope.state["model"]["semantic_hash"] == state_hash
    assert len(trained_run.metrics) == golden["metric_rows"]


def test_compact_recipe_is_registered_and_builds_the_canonical_model(
    tiny_config, tiny_data
) -> None:
    assert "credo.compact_sde_v3@3.0" in available_recipes()
    recipe = get_recipe("credo.compact_sde_v3@3.0")
    assert isinstance(recipe, CompactSDEV3Recipe)
    model = recipe.build_model(tiny_data, tiny_config.recipe_config)
    assert model.architecture()["latent_dim"] == tiny_data.latent_dim
    assert recipe.build_representation(tiny_data, None, {}) is tiny_data.representation
    plan = recipe.training_plan(tiny_data, tiny_config.recipe_config)
    assert tuple(stage.name for stage in plan.stages) == ("state", "mass", "context")
    assert all(stage.precision == "fp32" for stage in plan.stages)
    assert all(stage.optimizer.kind == "adam" for stage in plan.stages)
    assert all(stage.optimizer.weight_decay == 0 for stage in plan.stages)
    assert plan.seed == tiny_config.recipe_config.training.seed
    assert plan.particles == tiny_config.recipe_config.training.particles
    assert plan.steps_per_interval == tiny_config.recipe_config.training.steps_per_interval
    default_config = recipe.config_schema()()
    default_plan = recipe.training_plan(tiny_data, {})
    assert default_plan.stages[0].batching.measures_per_batch == (
        default_config.training.measures_per_batch
    )
    assert default_plan.stages[0].optimizer.learning_rate == (default_config.training.learning_rate)


def test_compact_execution_trace_matches_the_immutable_plan(trained_run) -> None:
    plan = trained_run.training_plan
    assert len(trained_run.execution_trace) == 3
    for stage, trace in zip(plan.stages, trained_run.execution_trace, strict=True):
        assert trace["stage"] == stage.name
        assert trace["optimizer"] == "Adam"
        assert trace["optimizer_kind"] == stage.optimizer.kind
        assert trace["learning_rate"] == stage.optimizer.learning_rate
        assert trace["weight_decay"] == stage.optimizer.weight_decay
        assert trace["trainable_tags"] == list(stage.trainable_tags)
        assert trace["objective_weights"] == {
            name: trained_run.objective_map[name].weight for name in stage.active_objectives
        }
        assert trace["objective_configs"] == {
            name: dict(trained_run.objective_map[name].config) for name in stage.active_objectives
        }
        assert trace["checkpoint_metric"] == stage.checkpoint_metric
        assert trace["epochs_completed"] == 1
        assert trace["selected_checkpoint_score"] >= 0


def test_common_evaluator_preserves_compact_metrics(trained_run) -> None:
    common = evaluate(trained_run)
    assert set(common["recipe_id"]) == {"credo.compact_sde_v3"}
    assert set(common["recipe_version"]) == {"3.0"}
    assert common["representation_id"].nunique() == 1
    assert common["split_id"].nunique() == 1
    assert common["split_id"].iloc[0].startswith(f"compact-v3:{trained_run.validation_strategy}:")
    assert len(common["split_id"].iloc[0].rsplit(":", 1)[1]) == 12
    assert common["evaluation_particles"].eq(8).all()
    original = trained_run.metrics.reset_index(drop=True)
    for column in original:
        assert common[column].reset_index(drop=True).equals(original[column])


def test_context_providers_declare_population_scope() -> None:
    assert NoContextProvider.requires_complete_catalog is False
    assert NoContextProvider.requires_full_group_rollout is False
    assert SelfConsistentContextProvider.requires_complete_catalog is False
    assert SelfConsistentContextProvider.requires_full_group_rollout is True
    assert CatalogContextProvider.requires_complete_catalog is True
    assert CatalogContextProvider.requires_full_group_rollout is False


def test_unavailable_recipe_has_a_clear_registry_error() -> None:
    with pytest.raises(RecipeUnavailableError, match="Installed recipes"):
        get_recipe("credo.not_installed@9.9")


def test_tensor_state_hash_supports_bfloat16() -> None:
    state = {"bf16": torch.tensor([1.0, -2.0], dtype=torch.bfloat16)}
    assert len(tensor_state_sha256(state)) == 64
