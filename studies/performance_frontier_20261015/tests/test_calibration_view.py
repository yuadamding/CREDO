"""Tiny fabricated CPU bank derivations; zero optimizer steps or outcomes."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import frontier.calibration_view as module
from frontier.calibration_view import CalibrationView
from frontier.configuration import ALPHA_GRID, DONORS, FAMILIES, SPREAD_GRID
from frontier.forecast import forecast_population
from frontier.forecast_storage import ForecastStorageSpec, POOL
from frontier.measure_ops import calibrate_abundance, calibrate_dispersion
from frontier.native_inputs import FoldInputs
from frontier.planner import ScoreTask, TrajectoryKey, family_plan, screen_plan
from frontier.selector_contract import OuterRecipe, Recipe
from test_forecast import source, model, binding, context, limits as core_limits
from test_forecast_storage import publish, reader


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def adapter_hash():
    return sha256(Path(module.__file__).read_bytes()).hexdigest()


def case(tmp_path, *, outer=False, seed=0, missing_pool=False):
    s, m = source(missing_pool=missing_pool), model("F2")
    if outer:
        s = replace(s, fold=FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4")))
    updates = 9216 if outer else 1024
    b = replace(binding(m, s), model_seed=seed, accepted_updates=updates)
    c = replace(context(m, s), model_seed=seed, accepted_update=updates)
    base = Recipe("F2-V0", .25, 1.3)
    prefixes = (5, 8)
    spec = ForecastStorageSpec.from_source(s, b, c, base, prefixes=prefixes)
    result = forecast_population(m, s, b, c, base, prefixes=prefixes, limits=core_limits())
    root = tmp_path / "bank"
    receipt = publish(root, spec, result)
    task = ScoreTask(TrajectoryKey(s.fold.training_donors, "F2-V0", seed),
                     "D1" if outer else "D2", "D1", updates, "outer" if outer else "inner")
    return m, result, root, spec, receipt, task


def inner(loaded, task, **kwargs):
    return CalibrationView.for_inner(loaded, plan=screen_plan(), selector="D2", task=task,
                                    adapter_source_sha256=adapter_hash(), **kwargs)


def frozen_recipe(**kwargs):
    return OuterRecipe(**(dict(assessment_donor="D1", fitting_donors=("D2", "D3", "D4"),
        configuration_id="F2-V0", alpha=.5, spread_scale=1.15,
        selection_contract_identity="c"*64, selection_evidence_identity="d"*64, inner_quality=.8) | kwargs))


def test_all_24_inner_recipes_exact_raw_prefix_derivation_without_rollout(tmp_path):
    m, result, root, spec, receipt, task = case(tmp_path)
    before = {p.name: sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    with reader(root, spec, receipt) as loaded:
        original_receipt = loaded.forecast_receipt
        with patch.object(type(m), "rollout", side_effect=AssertionError("No regeneration")), \
             patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
            view = inner(loaded, task)
            expected_recipes = tuple(Recipe("F2-V0", a, s) for a in ALPHA_GRID for s in SPREAD_GRID)
            assert view.recipes == expected_recipes and len(view.recipes) == 24
            identities = {view.recipe_identity(r) for r in view.recipes}
            assert len(identities) == 24
            for recipe in view.recipes:
                for prefix in view.prefixes:
                    original = result.prefixes[prefix]
                    for key in (*original.guide_laws, POOL):
                        expected = original.pooled_control if key == POOL else original.guide_laws[key]
                        actual = view.law(recipe, prefix, key)
                        assert torch.equal(actual.uncalibrated_states, expected.uncalibrated_states)
                        assert torch.equal(actual.log_weights, expected.log_weights)
                        assert actual.log_mass_increment == expected.log_mass_increment
                        assert torch.equal(actual.states, calibrate_dispersion(
                            expected.uncalibrated_states, expected.log_weights, recipe.spread_scale))
                    catalog = view.catalog(recipe, prefix)
                    for field in ("log_mass_increments", "log_source_frequencies", "uncalibrated_log_frequencies"):
                        assert torch.equal(catalog[field], getattr(original, field))
                    assert torch.equal(catalog["log_frequencies"], calibrate_abundance(
                        original.log_source_frequencies, original.uncalibrated_log_frequencies, recipe.alpha))
                    assert view.law(recipe, prefix, 4) is None
                    assert view.law(recipe, prefix, 5) is None
            manifest = view.derivation_manifest
            assert manifest["parent_recipe"] == {"configuration_id": "F2-V0", "alpha": .25, "spread_scale": 1.3}
            assert manifest["parent_publication"]["complete_sha256"] == receipt["complete_sha256"]
            assert manifest["parent_publication"]["manifest_sha256"] == receipt["manifest_sha256"]
            assert manifest["role"]["selector"] == "D2" and manifest["role"]["task"]["validation_donor"] == "D1"
            assert manifest["requires_owner_pretruth_seal"] is True
            assert manifest["endpoint_inputs_used"] is False and manifest["optimizer_calls"] == 0
            assert manifest["particle_generation_calls"] == 0 and manifest["publication_performed"] is False
            assert manifest["scientific_launch_authority"] is False
            assert {r["module"] for r in manifest["implementation_sources"]} == {"calibration_view.py", "measure_ops.py"}
            assert loaded.forecast_receipt == original_receipt
            assert loaded.spec is spec
    assert before == {p.name: sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


def test_nonmaximum_mixture_prefix_weights_are_not_truncated(tmp_path):
    _, result, root, spec, receipt, task = case(tmp_path)
    with reader(root, spec, receipt) as loaded:
        view = inner(loaded, task)
        short = view.law(Recipe("F2-V0", 1., 1.5), 5, 0)
        full = view.law(Recipe("F2-V0", 1., 1.5), 8, 0)
        assert torch.equal(short.log_weights, result.prefixes[5].guide_laws[0].log_weights)
        truncated = full.log_weights[:5] - torch.logsumexp(full.log_weights[:5], 0)
        assert not torch.equal(short.log_weights, truncated)
        assert torch.equal(short.uncalibrated_states, full.uncalibrated_states[:5])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_outer_accepts_only_exact_frozen_recipe_and_seed_context(tmp_path, seed):
    _, _, root, spec, receipt, task = case(tmp_path, outer=True, seed=seed)
    frozen = frozen_recipe()
    with reader(root, spec, receipt) as loaded:
        view = CalibrationView.for_outer(loaded, frozen=frozen, task=task, adapter_source_sha256=adapter_hash())
        expected = Recipe("F2-V0", .5, 1.15)
        assert view.recipes == (expected,)
        assert view.derivation_manifest["role"]["frozen_outer_recipe"]["selection_evidence_identity"] == "d"*64
        assert view.law(expected, 5, 0) is not None
        with pytest.raises(ValueError, match="inventory"):
            view.catalog(Recipe("F2-V0", .25, 1.15), 5)
        with pytest.raises(ValueError, match="inventory"):
            view.law(Recipe("F2-V0", .5, 1.3), 5, 0)
        wrong_seed = replace(task, key=replace(task.key, seed=(seed+1) % 3))
        with pytest.raises(ValueError, match="Parent bank"):
            CalibrationView.for_outer(loaded, frozen=frozen, task=wrong_seed, adapter_source_sha256=adapter_hash())


@pytest.mark.parametrize("change", ["selector", "query", "stage", "eligibility"])
def test_inner_role_cannot_widen_to_other_selector_outcome_or_stage(tmp_path, change):
    _, _, root, spec, receipt, task = case(tmp_path)
    plan, selector = screen_plan(), "D2"
    if change == "selector": selector = "D1"
    elif change == "query":
        selector = "D1"; task = replace(task, selector="D1", validation_donor="D2")
    else:
        choices = {d: tuple(f"{f}-V0" if change == "stage" else f"{f}-V1" for f in FAMILIES) for d in DONORS}
        plan = family_plan(choices)
        if change == "stage": task = replace(task, updates=3072)
    with reader(root, spec, receipt) as loaded:
        with pytest.raises(ValueError):
            CalibrationView.for_inner(loaded, plan=plan, selector=selector, task=task,
                                      adapter_source_sha256=adapter_hash())


@pytest.mark.parametrize("change", ["donors", "updates", "seeds", "evidence", "quality", "configuration"])
def test_outer_unvalidated_dataclass_fields_cannot_bypass_contract(tmp_path, change):
    _, _, root, spec, receipt, task = case(tmp_path, outer=True)
    kwargs = {"donors": {"fitting_donors": ("D3", "D4")}, "updates": {"updates": 3072},
              "seeds": {"seeds": (0,)}, "evidence": {"selection_evidence_identity": "bad"},
              "quality": {"inner_quality": float("nan")}, "configuration": {"configuration_id": "F0-V0"}}[change]
    with reader(root, spec, receipt) as loaded:
        with pytest.raises(ValueError):
            CalibrationView.for_outer(loaded, frozen=frozen_recipe(**kwargs), task=task,
                                      adapter_source_sha256=adapter_hash())


def test_missing_pool_preserved_and_returned_values_do_not_mutate_bank(tmp_path):
    _, _, root, spec, receipt, task = case(tmp_path, missing_pool=True)
    with reader(root, spec, receipt) as loaded:
        view = inner(loaded, task)
        recipe = Recipe("F2-V0", .1, 1.5)
        assert view.law(recipe, 5, POOL) is None
        law = view.law(recipe, 5, 0)
        original = law.states.clone()
        law.states.fill_(999); law.uncalibrated_states.fill_(999); law.log_weights.fill_(999)
        assert torch.equal(view.law(recipe, 5, 0).states, original)
        values = view.catalog(recipe, 5)
        original_q = values["log_frequencies"].clone()
        values["log_frequencies"].fill_(999)
        assert torch.equal(view.catalog(recipe, 5)["log_frequencies"], original_q)
        manifest = view.derivation_manifest
        manifest["recipes"].clear()
        assert len(view.derivation_manifest["recipes"]) == 24
        with pytest.raises(FrozenInstanceError): view._recipes = ()
        with pytest.raises(ValueError): view.law(Recipe("F0-V0"), 5, 0)
        with pytest.raises(ValueError): view.law(recipe, 7, 0)


def test_parent_lease_runtime_and_actual_source_identity_enforced(tmp_path):
    _, _, root, spec, receipt, task = case(tmp_path)
    loaded = reader(root, spec, receipt)
    with pytest.raises(RuntimeError): inner(loaded, task)
    with loaded:
        with pytest.raises(ValueError, match="source"):
            CalibrationView.for_inner(loaded, plan=screen_plan(), selector="D2", task=task,
                                      adapter_source_sha256="f"*64)
        view = inner(loaded, task)
        with patch.object(module, "_source_records", return_value=module._IMPORTED_SOURCES[:-1]):
            with pytest.raises(ValueError, match="source"): inner(loaded, task)
        with patch.object(module, "_stamp", return_value=(0,)):
            with pytest.raises(ValueError, match="implementation"): view.catalog(Recipe("F2-V0"), 5)
        saved = loaded.publication_reference
        loaded.publication_reference = dict(saved, complete_sha256="f"*64)
        with pytest.raises(ValueError, match="publication"): view.derivation_manifest
        loaded.publication_reference = saved
        torch.set_num_threads(1)
        try:
            with pytest.raises(ValueError, match="runtime"): view.catalog(Recipe("F2-V0"), 5)
        finally:
            torch.set_num_threads(2)
    with pytest.raises(RuntimeError): view.catalog(Recipe("F2-V0"), 5)


def test_changed_payload_rejected_before_derivation(tmp_path):
    _, _, root, spec, receipt, task = case(tmp_path)
    loaded = reader(root, spec, receipt)
    loaded.__enter__()
    try:
        view = inner(loaded, task)
        with (root / "log_weights.npy").open("r+b") as stream:
            stream.seek(-1, 2); stream.write(b"\xff")
        with pytest.raises(ValueError, match="changed"): view.law(Recipe("F2-V0"), 5, 0)
    finally:
        loaded.close(verify_hashes=False)


def test_direct_constructor_cannot_create_unbound_view():
    with pytest.raises(TypeError, match="role-bound"): CalibrationView()
