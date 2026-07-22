from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from credo import evaluate
from credo.particles import (
    CatalogContextProvider,
    NoContextProvider,
    SelfConsistentContextProvider,
)
from credo.recipes.compact_v3 import CompactSDEV3Recipe
from credo.registry import RecipeUnavailableError, available_recipes, get_recipe

ROOT = Path(__file__).resolve().parents[1]


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def test_compact_v3_golden_run_is_unchanged(trained_run) -> None:
    golden = json.loads((ROOT / "tests/golden/compact_v3.json").read_text(encoding="utf-8"))
    metrics = trained_run.metrics.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    ).encode("utf-8")
    assert _state_hash(trained_run.model.state_dict()) == golden["model_state_sha256"]
    assert hashlib.sha256(metrics).hexdigest() == golden["metrics_sha256"]
    assert _file_hash(trained_run.config.output / "checkpoint.pt") == golden["checkpoint_sha256"]
    assert len(trained_run.metrics) == golden["metric_rows"]


def test_compact_recipe_is_registered_and_builds_the_canonical_model(
    tiny_config, tiny_data
) -> None:
    assert "credo.compact_sde_v3@3.0" in available_recipes()
    recipe = get_recipe("credo.compact_sde_v3@3.0")
    assert isinstance(recipe, CompactSDEV3Recipe)
    model = recipe.build_model(tiny_data, tiny_config)
    assert model.architecture()["latent_dim"] == tiny_data.latent_dim
    assert recipe.build_representation(tiny_data, None, {}) is tiny_data.representation
    plan = recipe.training_plan(tiny_data, tiny_config)
    assert tuple(stage.name for stage in plan.stages) == ("state", "mass", "context")
    assert all(stage.precision == "fp32" for stage in plan.stages)


def test_common_evaluator_preserves_compact_metrics(trained_run) -> None:
    common = evaluate(trained_run)
    assert set(common["recipe_id"]) == {"credo.compact_sde_v3"}
    assert set(common["recipe_version"]) == {"3.0"}
    assert common["representation_id"].nunique() == 1
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
