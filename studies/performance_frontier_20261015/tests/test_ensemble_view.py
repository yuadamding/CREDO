"""Tiny CPU lazy mixtures versus existing arithmetic; zero optimizer steps."""

from contextlib import contextmanager, ExitStack
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from frontier import calibration_view, ensemble_view
from frontier.calibration_view import CalibrationView
from frontier.ensemble_forecast import _binding
from frontier.ensemble_view import EnsembleView
from frontier.forecast_storage import ForecastStorageSpec, POOL
from frontier.native_inputs import FoldInputs
from frontier.planner import ScoreTask, TrajectoryKey
from frontier.sampling import SamplingContext
from frontier.selector_contract import OuterRecipe, Recipe
from test_ensemble_forecast import components, combine
from test_forecast import source
from test_forecast_storage import publish, reader


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def adapter_hash(module=ensemble_view):
    return sha256(Path(module.__file__).read_bytes()).hexdigest()


@contextmanager
def package(tmp_path, family="F2", *, alpha=.25, spread=1.3, missing_pool=False, all_missing=False, derived=True):
    values, models = components(family, alpha=alpha, spread=spread, missing_pool=missing_pool,
                                all_missing=all_missing, derived=derived)
    early = replace(source(missing_pool=missing_pool, all_missing=all_missing),
                    fold=FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4")))
    frozen = OuterRecipe("D1", early.fold.training_donors, family+"-V0", alpha, spread, "c"*64, "d"*64, .8)
    with ExitStack() as stack:
        views, readers = [], []
        for seed, value in enumerate(values):
            receipt = value.receipt
            spec = ForecastStorageSpec.from_source(early, _binding(receipt), SamplingContext(**receipt["context"]),
                Recipe(**receipt["recipe"]), prefixes=tuple(receipt["single_seed_prefixes"]),
                ensemble_component_totals=tuple(receipt["ensemble_component_totals"]))
            path = tmp_path/f"seed-{seed}"
            publication = publish(path, spec, value)
            loaded = stack.enter_context(reader(path, spec, publication))
            readers.append(loaded)
            task = ScoreTask(TrajectoryKey(early.fold.training_donors, family+"-V0", seed), "D1", "D1", 9216, "outer")
            views.append(CalibrationView.for_outer(loaded, frozen=frozen, task=task,
                         adapter_source_sha256=adapter_hash(calibration_view)))
        yield values, models, tuple(views), tuple(readers), frozen


def bind(views, **changes):
    args = dict(totals=(16, 23), maximum_working_bytes=128*1024**2, adapter_source_sha256=adapter_hash())
    args.update(changes)
    return EnsembleView.bind(views, **args)


@pytest.mark.parametrize("family", ["F0", "F1", "F2", "F3", "F4", "F5"])
def test_lazy_all_family_laws_catalogs_equal_existing_ensemble(tmp_path, family):
    with package(tmp_path, family) as (values, _, views, _, _):
        expected = combine(values)
        view = bind(views)
        for total, original in expected.prefixes.items():
            for index in (*original.guide_laws, POOL):
                law = view.law(total, index)
                target = original.pooled_control if index == POOL else original.guide_laws[index]
                for name in ("states", "uncalibrated_states", "log_weights", "uncalibrated_log_weights"):
                    assert torch.equal(getattr(law, name), getattr(target, name))
            for name, tensor in view.catalog(total).items():
                assert torch.equal(tensor, getattr(original, name))
            assert view.law(total, 4) is None and view.law(total, 5) is None
        manifest = view.derivation_manifest
        assert manifest["component_derivation_sha256"] == [v.identity for v in views]
        assert manifest["parent_publications"] == [v.derivation_manifest["parent_publication"] for v in views]
        assert manifest["newly_generated_particles"] == manifest["model_calls"] == manifest["optimizer_calls"] == 0
        assert manifest["endpoint_inputs_used"] is False and manifest["publication_performed"] is False
        assert manifest["requires_owner_pretruth_seal"] is True
        assert manifest["scientific_launch_authority"] is False
        resources = manifest["resources"]
        bank_bytes = sum(v._reader.publication_reference["total_bytes"] for v in views)
        assert resources["mapped_bank_charge_bytes"] == bank_bytes
        assert resources["host_admission_estimate_bytes"] == bank_bytes + resources["additional_view_host_estimate_bytes"]
        assert resources["mapped_page_eviction_assumed"] is False
        assert {r["module"] for r in manifest["implementation_sources"]} == {
            "ensemble_view.py", "ensemble_forecast.py", "measure_ops.py", "forecast.py"}


def test_mass_weighted_raw_calibrated_shares_and_equal_pooled_control(tmp_path):
    with package(tmp_path, alpha=0., spread=1.5) as (values, _, views, _, _):
        view = bind(views)
        total = 16
        counts = view.derivation_manifest["component_counts"][str(total)]
        assert counts == [6, 5, 5]
        law, pool = view.law(total, 0), view.law(total, POOL)
        raw_q = torch.stack([v.prefixes[n].uncalibrated_log_frequencies[0].exp() for v, n in zip(values, counts)])
        assert not torch.equal(law.log_weights, law.uncalibrated_log_weights)
        start = 0
        for seed, n in enumerate(counts):
            # Alpha zero restores the same q0 for all seeds; uneven particle
            # allocations still receive exactly 1/3 finite-measure weight.
            assert abs(float(law.log_weights[start:start+n].exp().sum())-1/3) < 1e-12
            assert abs(float(law.uncalibrated_log_weights[start:start+n].exp().sum())-
                       float(raw_q[seed]/raw_q.sum())) < 1e-12
            assert abs(float(pool.log_weights[start:start+n].exp().sum())-1/3) < 1e-12
            assert torch.equal(law.states[start:start+n], values[seed].prefixes[n].guide_laws[0].states)
            assert torch.equal(pool.states[start:start+n], values[seed].prefixes[n].pooled_control.states)
            start += n
        assert start == total
        assert torch.equal(pool.log_weights, pool.uncalibrated_log_weights)


def test_no_models_particles_full_bank_reads_or_repeated_catalog_construction(tmp_path):
    with package(tmp_path) as (_, models, views, readers, _):
        before = [{p.name: sha256(p.read_bytes()).hexdigest() for p in r.root.iterdir()} for r in readers]
        with patch.object(type(models[0]), "rollout", side_effect=AssertionError("No models")), \
             patch("frontier.sampling.sample_batch", side_effect=AssertionError("No draws")), \
             patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
            with patch.object(CalibrationView, "law", side_effect=AssertionError("No law read during bind")):
                view = bind(views)
            original = CalibrationView.law
            calls = []
            def one_law(self, recipe, prefix, index):
                calls.append((self, prefix, index))
                return original(self, recipe, prefix, index)
            with patch.object(CalibrationView, "catalog", side_effect=AssertionError("Catalogs cached once")), \
                 patch.object(CalibrationView, "law", one_law):
                result = view.law(16, 0)
                assert result.states.shape == (16, 48)
                assert calls == [(views[0], 6, 0), (views[1], 5, 0), (views[2], 5, 0)]
                assert view.catalog(23)["log_frequencies"].shape == (6,)
        assert before == [{p.name: sha256(p.read_bytes()).hexdigest() for p in r.root.iterdir()} for r in readers]


@pytest.mark.parametrize("all_missing", [False, True])
def test_absent_states_and_pool_are_never_synthesized(tmp_path, all_missing):
    with package(tmp_path, missing_pool=True, all_missing=all_missing) as (_, _, views, _, _):
        view = bind(views)
        assert view.law(16, POOL) is None and view.law(23, 2) is None
        if all_missing:
            assert view.law(16, 0) is None
            output = view.catalog(16)
            torch.testing.assert_close(output["log_frequencies"], output["log_source_frequencies"], atol=1e-15, rtol=0)


@pytest.mark.parametrize("failure", ["reordered", "duplicate", "recipe", "selection", "missing_prefix", "source_hash", "cap", "total"])
def test_foreign_or_unbound_component_inventory_rejected(tmp_path, failure):
    with package(tmp_path, derived=failure != "missing_prefix") as (_, _, views, _, frozen):
        changes = {}
        if failure == "reordered": views = tuple(reversed(views))
        elif failure == "duplicate": views = (views[0], views[0], views[2])
        elif failure in ("recipe", "selection"):
            new_frozen = replace(frozen, **({"alpha": .5} if failure == "recipe" else {"selection_evidence_identity": "e"*64}))
            task = ScoreTask(TrajectoryKey(frozen.fitting_donors, frozen.configuration_id, 2), "D1", "D1", 9216, "outer")
            changed = CalibrationView.for_outer(views[2]._reader, frozen=new_frozen, task=task,
                adapter_source_sha256=adapter_hash(calibration_view))
            views = (*views[:2], changed)
        elif failure == "source_hash": changes["adapter_source_sha256"] = "f"*64
        elif failure == "cap": changes["maximum_working_bytes"] = 1
        elif failure == "total": changes["totals"] = (17, 23)
        with pytest.raises((TypeError, ValueError)):
            bind(views, **changes)


def test_inner_derivation_cannot_be_used_for_outer_ensemble(tmp_path):
    from test_calibration_view import case, inner
    inner_root = tmp_path/"inner"; inner_root.mkdir()
    _, _, root, spec, publication, task = case(inner_root)
    with reader(root, spec, publication) as loaded:
        inner_view = inner(loaded, task)
        with package(tmp_path) as (_, _, views, _, _):
            with pytest.raises(ValueError, match="outer"):
                bind((inner_view, *views[1:]))


def test_cached_catalogs_and_returned_values_do_not_expose_mutable_state(tmp_path):
    with package(tmp_path) as (_, _, views, _, _):
        view = bind(views)
        original = view.catalog(16)["log_frequencies"]
        changed = view.catalog(16)
        for value in changed.values(): value.fill_(99)
        assert torch.equal(view.catalog(16)["log_frequencies"], original)
        original_law = view.law(16, 0).states
        changed_law = view.law(16, 0)
        changed_law.states.fill_(99); changed_law.log_weights.fill_(99)
        assert torch.equal(view.law(16, 0).states, original_law)
        record = view.derivation_manifest; record["parent_publications"].clear()
        assert len(view.derivation_manifest["parent_publications"]) == 3
        assert all(type(blob) is bytes for _, blobs in view._catalogs for blob in blobs)
        with pytest.raises(FrozenInstanceError): view._totals = (99,)
        with pytest.raises(ValueError): view.law(16, 99)
        with pytest.raises(ValueError): view.law(17, 0)
        with pytest.raises(ValueError): view.law(16, True)


def test_full_mapped_publications_are_charged_not_just_lazy_law_buffers(tmp_path):
    with package(tmp_path) as (_, _, views, _, _):
        view = bind(views)
        resources = view.derivation_manifest["resources"]
        # Enough for every additional buffer, but not the already-open mapped
        # publications, is insufficient: no implicit owner budget credit.
        with pytest.raises(ValueError, match="cap"):
            bind(views, maximum_working_bytes=resources["additional_view_host_estimate_bytes"])
        with pytest.raises(ValueError, match="cap"):
            bind(views, maximum_working_bytes=resources["host_admission_estimate_bytes"]-1)


def test_parent_runtime_and_implementation_leases_remain_active(tmp_path):
    with package(tmp_path) as (_, _, views, readers, _):
        view = bind(views)
        with patch.object(ensemble_view, "_stamp", return_value=(0,)):
            with pytest.raises(ValueError, match="implementation"): view.catalog(16)
        torch.set_num_threads(1)
        try:
            with pytest.raises(ValueError, match="runtime"): view.law(16, 0)
        finally:
            torch.set_num_threads(2)
        readers[2].close()
        with pytest.raises(RuntimeError): view.law(16, 0)


def test_factory_only():
    with pytest.raises(TypeError): EnsembleView()
