"""Tiny fabricated CPU mixtures only: zero optimizer, real data or GPU calls."""

import copy
from dataclasses import asdict, replace
import math
from types import MappingProxyType
from unittest.mock import patch

import pytest
import torch

from frontier.ensemble_forecast import EnsembleForecast, EnsembleLimits, ensemble_forecast
from frontier.forecast import (Forecast, ForecastModelBinding, ensemble_prefix_plan,
                              forecast_population, model_state_digest)
from frontier.measure_ops import round_robin_component_counts
from frontier.models import FrontierModel, build_model
from frontier.native_inputs import FoldInputs
from frontier.sampling import SamplingContext
from frontier.selector_contract import Recipe
from test_forecast import limits as forecast_limits, source


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def limits(**changes):
    return EnsembleLimits(**(dict(maximum_host_bytes=128 * 1024**2,
        maximum_output_bytes=16 * 1024**2, maximum_working_bytes=16 * 1024**2) | changes))


def components(family="F2", *, alpha=.25, spread=1.3, missing_pool=False,
               all_missing=False, derived=True):
    s = replace(source(missing_pool=missing_pool, all_missing=all_missing),
                fold=FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4")))
    results, models = [], []
    for seed in (0, 1, 2):
        kwargs = dict(target_ids=("KNOWN",), support_factors=[.7]) if family in ("F3", "F5") else {}
        model = build_model(family + "-V0", seed=17 + seed, **kwargs).set_phase("joint")
        # Different constant reaction integrals ensure the test distinguishes
        # finite-measure seed weights from equal state/particle-count mixtures.
        with torch.no_grad():
            if model.config.dynamics == "direct":
                for index, mapping in enumerate(model.components):
                    mapping.weight_reference.weight.zero_()
                    mapping.weight_reference.bias.fill_(.2 * (seed + 1) + .1 * index)
                    mapping.weight_modulation.weight.zero_(); mapping.weight_modulation.bias.zero_()
                if family == "F2":
                    model.mixture_logits.copy_(torch.tensor([-2., -.5, 1., 2.]) + seed * torch.arange(4) * .1)
            else:
                model.fields.growth_reference.weight.zero_()
                model.fields.growth_reference.bias.fill_(.2 * (seed + 1))
                model.fields.growth_residual.weight.zero_(); model.fields.growth_residual.bias.zero_()
        model.eval().requires_grad_(False)
        b = ForecastModelBinding(s.fold, s.observation_binding, model.config.configuration_id,
            str(seed + 1) * 64, model_state_digest(model), "9" * 64, "a" * 64, seed, 9216)
        context = SamplingContext("fabricated-ensemble", s.fold.training_donors,
            model.config.configuration_id, 731, seed, 9216, "forecast")
        result = forecast_population(model, s, b, context, Recipe(model.config.configuration_id, alpha, spread),
            prefixes=(16, 23), limits=forecast_limits(), ensemble_component_totals=(16, 23) if derived else ())
        models.append(model); results.append(result)
    return tuple(results), tuple(models)


def combine(values, **kwargs):
    options = dict(totals=(16, 23), limits=limits(), synthetic_fixture=True)
    options.update(kwargs)
    return ensemble_forecast(values, **options)


def changed_receipt(component, change):
    receipt = copy.deepcopy(component.receipt)
    change(receipt)
    return Forecast(component.prefixes, receipt)


def changed_prefix(component, particles, **changes):
    prefixes = dict(component.prefixes)
    prefixes[particles] = replace(prefixes[particles], **changes)
    return Forecast(MappingProxyType(prefixes), component.receipt)


@pytest.mark.parametrize("family", ["F0", "F1", "F2", "F3", "F4", "F5"])
def test_each_family_fixed_total_finite_measure_and_full_available_support(family):
    values, _ = components(family)
    result = combine(values)
    assert isinstance(result, EnsembleForecast)
    assert result.receipt["seeds"] == [0, 1, 2]
    assert result.receipt["newly_generated_particles"] == result.receipt["optimizer_calls"] == 0
    assert result.receipt["query_donor"] == "D1"
    for total, output in result.prefixes.items():
        counts = round_robin_component_counts(total)
        assert output.component_particles == counts
        assert tuple(output.guide_laws) == (0, 1, 2, 3)  # Includes unseen targeting source.
        assert output.unavailable_guides == (4, 5)
        assert output.log_frequencies.shape == (6,)
        assert abs(torch.logsumexp(output.log_frequencies, 0).item()) < 1e-12
        selected = tuple(value.prefixes[n] for value, n in zip(values, counts))
        for index, law in output.guide_laws.items():
            assert law.states.shape == law.uncalibrated_states.shape == (total, 48)
            assert law.states.device.type == "cpu" and law.states.dtype == torch.float32
            assert abs(torch.logsumexp(law.log_weights, 0).item()) < 1e-12
            assert abs(torch.logsumexp(law.uncalibrated_log_weights, 0).item()) < 1e-12
            # Joint first moment must equal 1/3 sum(q_seed * E_seed[Z]).
            expected = sum(p.log_frequencies[index].exp() *
                (p.guide_laws[index].states.double() * p.guide_laws[index].log_weights.exp()[:, None]).sum(0)
                for p in selected) / 3
            actual = (output.log_frequencies[index].exp() *
                (law.states.double() * law.log_weights.exp()[:, None]).sum(0))
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        expected_catalog = sum(p.log_frequencies.exp() for p in selected) / 3
        torch.testing.assert_close(output.log_frequencies.exp(), expected_catalog, atol=1e-14, rtol=0)
        # Complement conditional frequencies preserve the same source ratio.
        torch.testing.assert_close(output.log_frequencies[4] - output.log_frequencies[5],
                                   output.log_source_frequencies[4] - output.log_source_frequencies[5])


def test_calibrated_and_raw_seed_shares_differ_and_coordinates_are_never_averaged():
    values, _ = components(alpha=0., spread=1.5)
    output = combine(values).prefixes[16]
    counts = output.component_particles
    law = output.guide_laws[0]
    assert not torch.allclose(law.uncalibrated_log_weights, law.log_weights, atol=1e-6, rtol=0)
    start = 0
    raw_q = torch.stack([value.prefixes[n].uncalibrated_log_frequencies[0].exp()
                         for value, n in zip(values, counts)])
    for seed, (value, n) in enumerate(zip(values, counts)):
        p = value.prefixes[n]
        torch.testing.assert_close(law.states[start:start+n], p.guide_laws[0].states, atol=0, rtol=0)
        torch.testing.assert_close(law.uncalibrated_states[start:start+n],
                                   p.guide_laws[0].uncalibrated_states, atol=0, rtol=0)
        assert abs(law.log_weights[start:start+n].exp().sum().item() - 1 / 3) < 1e-12
        assert abs(law.uncalibrated_log_weights[start:start+n].exp().sum().item()
                   - (raw_q[seed] / raw_q.sum()).item()) < 1e-12
        start += n
    assert start == 16  # Not a coordinatewise mean of three shorter populations.
    torch.testing.assert_close(output.log_frequencies, output.log_source_frequencies, atol=1e-12, rtol=0)


def test_pooled_ntc_is_equal_seed_law_mixture_not_mass_or_particle_count_weighted():
    values, _ = components()
    output = combine(values).prefixes[16]
    counts = output.component_particles
    masses = [values[s].prefixes[n].pooled_control.log_mass_increment for s, n in enumerate(counts)]
    assert len(set(masses)) == 3
    assert counts == (6, 5, 5)
    pool = output.pooled_control
    assert torch.equal(pool.log_weights, pool.uncalibrated_log_weights)
    start = 0
    for seed, n in enumerate(counts):
        assert abs(pool.log_weights[start:start+n].exp().sum().item() - 1 / 3) < 1e-12
        torch.testing.assert_close(pool.states[start:start+n], values[seed].prefixes[n].pooled_control.states)
        start += n
    assert output.log_frequencies.numel() == 6  # No extra pooled-control catalog entry.


def test_mixture_component_prefixes_reuse_maximum_states_and_preserve_F2_weights():
    values, _ = components()
    result = combine(values)
    for seed, value in enumerate(values):
        assert value.receipt["maximum_generated_particles"] == 23
        assert all(r["generated_prefix_per_law"] == 23 for r in value.receipt["sampler_receipts"])
        for total in (16, 23):
            n = round_robin_component_counts(total)[seed]
            short, full = value.prefixes[n].guide_laws[0], value.prefixes[23].guide_laws[0]
            torch.testing.assert_close(short.uncalibrated_states, full.uncalibrated_states[:n], atol=0, rtol=0)
            # For nondivisible component counts, truncating old normalized F2
            # weights is not the qualified proposal-corrected prefix law.
            if n in (5, 6, 7):
                wrong = full.log_weights[:n] - torch.logsumexp(full.log_weights[:n], 0)
                assert not torch.allclose(short.log_weights, wrong, atol=1e-4, rtol=0)
    assert result.receipt["particle_order"].endswith("not_nested_across_total_budgets")


def test_ensemble_never_calls_model_sampler_optimizer_or_mutates_input_banks():
    values, _ = components()
    before = [[t.clone() for p in value.prefixes.values() for law in p.guide_laws.values()
               for t in (law.states, law.uncalibrated_states, law.log_weights)] for value in values]
    with patch.object(FrontierModel, "rollout", side_effect=AssertionError("no generation")), \
         patch("frontier.forecast.sample_batch", side_effect=AssertionError("no sampling")), \
         patch.object(torch.optim.Optimizer, "step", side_effect=AssertionError("no optimizer")):
        result = combine(values)
    for value, old in zip(values, before):
        current = [t for p in value.prefixes.values() for law in p.guide_laws.values()
                   for t in (law.states, law.uncalibrated_states, law.log_weights)]
        assert all(torch.equal(a, b) for a, b in zip(current, old))
    # Returned coordinates own new storage, not a mutable view of a seed bank.
    result.prefixes[16].guide_laws[0].states.zero_()
    assert all(torch.equal(a, b) for a, b in zip(
        [t for p in values[0].prefixes.values() for law in p.guide_laws.values()
         for t in (law.states, law.uncalibrated_states, law.log_weights)], before[0]))


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_source_is_unavailable_and_catalog_is_not_removed(all_missing):
    values, _ = components(missing_pool=True, all_missing=all_missing)
    result = combine(values)
    for output in result.prefixes.values():
        assert output.pooled_control is None
        assert output.pooled_control_unavailable_reason == "missing_positive_RNA_source_NTC_pool"
        assert output.log_frequencies.numel() == 6
        if all_missing:
            assert not output.guide_laws
            assert output.unavailable_guides == tuple(range(6))
            torch.testing.assert_close(output.log_frequencies, output.log_source_frequencies, atol=1e-12, rtol=0)


def test_registered_large_allocations_are_metadata_only_and_scientific_gate_stays_fixed():
    plans = [ensemble_prefix_plan(s, (4096, 8192), (4096, 8192)) for s in (0, 1, 2)]
    assert tuple(p["4096"] for p in plans) == (1366, 1365, 1365)
    assert tuple(p["8192"] for p in plans) == (2731, 2731, 2730)
    for args in ((0, (4096,), (512,)), (0, (4096,), (1366,)), (0, (8000,), (8192,))):
        with pytest.raises(ValueError):
            ensemble_prefix_plan(*args)
    with pytest.raises(ValueError):
        ensemble_prefix_plan(0, (129,), (129,), synthetic_fixture=True)


@pytest.mark.parametrize("damage", ["seed_order", "seed_duplicate", "recipe", "representation", "fold",
    "target_support", "numerical_source", "context", "query", "source", "support", "inner", "stage"])
def test_incompatible_ensembles_rejected(damage):
    values, _ = components()
    if damage == "seed_order":
        values = (values[1], values[0], values[2])
    elif damage == "seed_duplicate":
        values = (values[0], values[0], values[2])
    else:
        def change(r):
            if damage == "recipe": r["recipe"]["alpha"] = .5
            elif damage == "representation": r["model_binding"]["observation_binding"]["state_sha256"] = "b" * 64
            elif damage == "fold": r["model_binding"]["fold"]["fold_id"] = "different-fold"
            elif damage == "target_support": r["model_binding"]["target_support_sha256"] = "b" * 64
            elif damage == "numerical_source": r["model_binding"]["numerical_source_sha256"] = "b" * 64
            elif damage == "context": r["context"]["study_seed"] += 1
            elif damage == "query": r["query_donor"] = "D2"
            elif damage == "source": r["source_identity"] = "b" * 64
            elif damage == "support": r["supported_guide_indices"] = [0, 2, 3]
            elif damage == "inner": r["model_binding"]["fold"]["training_donors"] = ["D3", "D4"]
            elif damage == "stage": r["model_binding"]["accepted_updates"] = 3072
        values = (values[0], changed_receipt(values[1], change), values[2])
    with pytest.raises(ValueError):
        combine(values)


def test_missing_component_prefix_has_no_regenerate_or_truncate_fallback():
    values, _ = components(derived=False)
    with pytest.raises(ValueError, match="not derived"):
        combine(values)
    values, _ = components()
    prefixes = dict(values[1].prefixes); prefixes.pop(5)
    values = (values[0], Forecast(MappingProxyType(prefixes), values[1].receipt), values[2])
    with pytest.raises(ValueError, match="not derived"):
        combine(values)


def test_mismatched_catalog_prefix_masses_and_missing_pool_are_rejected():
    values, _ = components()
    mass = values[0].prefixes[6].log_mass_increments.clone(); mass[0] += .5
    with pytest.raises(ValueError, match="mass increments differ"):
        combine((changed_prefix(values[0], 6, log_mass_increments=mass), *values[1:]))
    with pytest.raises(ValueError, match="Pooled NTC support"):
        combine((changed_prefix(values[0], 6, pooled_control=None,
                 pooled_control_unavailable_reason="missing_positive_RNA_source_NTC_pool"), *values[1:]))


@pytest.mark.parametrize("cap", ["maximum_host_bytes", "maximum_output_bytes", "maximum_working_bytes"])
def test_memory_admission_precedes_mixture_concatenation(cap):
    values, _ = components()
    with patch("frontier.ensemble_forecast._mix", side_effect=AssertionError("no output allocation")):
        with pytest.raises(ValueError, match="explicit cap"):
            combine(values, limits=limits(**{cap: 1}))


def test_changed_input_during_combination_is_detected_before_return():
    from frontier import ensemble_forecast as module
    values, _ = components()
    original = module._mix
    calls = []
    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        if not calls:
            values[0].prefixes[6].guide_laws[0].states.add_(.01)
        calls.append(1)
        return value
    with patch.object(module, "_mix", side_effect=changed):
        with pytest.raises(ValueError, match="content changed"):
            combine(values)


def test_inputs_cannot_be_relabelled_as_scientific_or_exceed_fixture_scope():
    values, _ = components()
    with pytest.raises(ValueError):
        combine(values, synthetic_fixture=False)
    with pytest.raises(ValueError):
        combine(values, totals=(129,))


def test_prefix_generation_rejects_ensemble_on_inner_or_preterminal_recipe():
    from test_forecast import binding, context, model
    s, m = source(), model()
    with patch.object(m, "rollout", side_effect=AssertionError("no generation")):
        with pytest.raises(ValueError, match="outer triple"):
            forecast_population(m, s, binding(m, s), context(m, s), Recipe("F0-V0"),
                prefixes=(16,), limits=forecast_limits(), ensemble_component_totals=(16,))
