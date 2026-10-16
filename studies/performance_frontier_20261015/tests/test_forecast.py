"""Fabricated CPU inference only; no optimizer, checkpoint or data access."""

from dataclasses import replace
import math
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from frontier.forecast import (ForecastLimits, ForecastModelBinding, ForecastSource,
                              forecast_population, model_state_digest)
from frontier.evaluation import score_catalog, score_joint, score_state_law
from frontier.fitting_provider import CatalogGuide, PopulationArrays, PopulationReference, _hash
from frontier.measure_ops import calibrate_dispersion
from frontier.models import build_model
from frontier.native_inputs import FoldInputs
from frontier.sampling import SamplingContext, sample_batch
from frontier.selector_contract import Recipe
from frontier.training_owner import ObservationBinding


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def ro(value, dtype):
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


def source(*, missing_pool=False, all_missing=False):
    fold = FoldInputs("subset-D3-D4", ("D3", "D4"))
    catalog = tuple(CatalogGuide(*row) for row in (
        ("g0", "KNOWN", False), ("g1", "UNSEEN", False), ("n0", "NTC", True),
        ("n1", "NTC", True), ("noRNA", "KNOWN", False), ("zeroCapture", "UNSEEN", False)))
    positive = [2, 1, 1, 3, 0, 0]
    if missing_pool: positive[2:4] = [0, 0]
    if all_missing: positive = [0] * 6
    counts, live = ro([20, 4, 100, 3, 7, 0], np.int64), ro(positive, np.int64)
    offsets = ro(np.r_[0, np.cumsum(live)], np.int64)
    states = ro(np.arange(int(live.sum()) * 48).reshape(-1, 48) / 100., np.float32)
    observation = ObservationBinding("1" * 64, "2" * 64)
    ref = PopulationReference("D1", "Stim8hr", "D1_Stim8hr", "aggregate-unresolved", "3" * 64,
                              "4" * 64, "5" * 64, "6" * 64, "0" * 64)
    population = PopulationArrays(ref, counts, live, offsets, states, None)
    ref = replace(ref, latent_sha256=population.latent_identity(
        catalog_sha256=_hash([dict(guide_id=g.guide_id, target_id=g.target_id, is_control=g.is_control)
                              for g in catalog]), observation_binding=observation))
    return ForecastSource(fold, catalog, observation, "7" * 64, ref, counts, live, offsets, states, True)


def model(family="F0"):
    kwargs = dict(target_ids=("KNOWN",), support_factors=[.7]) if family in ("F3", "F5") else {}
    result = build_model(family + "-V0", seed=17, **kwargs).set_phase("joint")
    return result.eval().requires_grad_(False)


def binding(m, s):
    return ForecastModelBinding(s.fold, s.observation_binding, m.config.configuration_id,
                                "8" * 64, model_state_digest(m), "9" * 64, "a" * 64, 0, 1024)


def context(m, s):
    return SamplingContext("fabricated-forecast", s.fold.training_donors, m.config.configuration_id,
                           731, 0, 1024, "forecast")


def limits(**kwargs):
    return ForecastLimits(**(dict(maximum_host_bytes=128 * 1024**2,
        maximum_device_working_bytes=128 * 1024**2, maximum_output_bytes=16 * 1024**2,
        maximum_pool_cells=64, microbatch_laws=2) | kwargs))


def run(m=None, s=None, *, prefixes=(5, 8), alpha=1., spread=1., resource=None):
    m, s = (m or model()), (s or source())
    return forecast_population(m, s, binding(m, s), context(m, s),
        Recipe(m.config.configuration_id, alpha, spread), prefixes=prefixes, limits=resource or limits())


@pytest.mark.parametrize("family", ["F0", "F1", "F2", "F3", "F4", "F5"])
def test_each_family_complete_catalog_source_only_and_score_ready(family):
    m, s = model(family), source()
    result = run(m, s)
    assert result.receipt["optimizer_calls"] == 0
    assert result.receipt["endpoint_inputs_used"] is False
    assert result.receipt["publication_claimed"] is False
    for p, output in result.prefixes.items():
        assert tuple(output.guide_laws) == (0, 1, 2, 3)
        assert output.unavailable_guides == (4, 5)
        assert output.log_frequencies.shape == (6,)
        assert torch.isfinite(output.log_frequencies).all()
        assert output.log_mass_increments[4:].tolist() == [0., 0.]
        torch.testing.assert_close(output.log_frequencies.exp().sum(), torch.tensor(1., dtype=torch.float64))
        for law in (*output.guide_laws.values(), output.pooled_control):
            assert law.states.shape == (p, 48) and law.states.dtype == torch.float32
            assert law.log_weights.dtype == torch.float64
            assert abs(torch.logsumexp(law.log_weights, 0).item()) < 1e-12
            assert not law.states.requires_grad and law.states.device.type == "cpu"
        # Complement relative composition remains persistence, though its total
        # allocation may change through the catalog's one normalizer.
        torch.testing.assert_close(output.log_frequencies[4] - output.log_frequencies[5],
                                   output.log_source_frequencies[4] - output.log_source_frequencies[5])


def test_fixed_calibrations_and_weighted_mean_preservation():
    m, s = model(), source()
    original, calibrated = run(m, s), run(m, s, alpha=0., spread=1.3)
    for p, out in calibrated.prefixes.items():
        torch.testing.assert_close(out.log_frequencies, out.log_source_frequencies, atol=1e-14, rtol=0)
        torch.testing.assert_close(out.uncalibrated_log_frequencies, original.prefixes[p].log_frequencies)
        for index, law in out.guide_laws.items():
            raw = original.prefixes[p].guide_laws[index]
            torch.testing.assert_close(law.uncalibrated_states, raw.states, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(law.states, calibrate_dispersion(law.uncalibrated_states, law.log_weights, 1.3))
            w = law.log_weights.exp().float()[:, None]
            torch.testing.assert_close((w * law.states).sum(0), (w * law.uncalibrated_states).sum(0), atol=5e-7, rtol=1e-6)
        pool = out.pooled_control
        torch.testing.assert_close(pool.states, calibrate_dispersion(pool.uncalibrated_states, pool.log_weights, 1.3))


def test_output_catalog_device_does_not_follow_global_default():
    # All inputs/model are constructed first. A changed caller default must not
    # alter the explicitly CPU publication interface or trigger a GPU transfer.
    m, s = model("F2"), source()
    with torch.device("meta"):
        result = run(m, s)
    for output in result.prefixes.values():
        for value in (output.log_mass_increments, output.log_source_frequencies,
                      output.uncalibrated_log_frequencies, output.log_frequencies):
            assert value.device.type == "cpu"


def test_outputs_feed_existing_score_arithmetic_without_hidden_normalization():
    s = source()
    output = run(model("F2"), s, alpha=.25, spread=1.15).prefixes[8]
    law = output.guide_laws[0]
    observed = torch.tensor(s.states[:2], dtype=torch.float64)
    weights = torch.full((2,), .5, dtype=torch.float64)
    state = score_state_law(law.states, law.log_weights.exp(), observed, weights,
                            observed, weights, block=4)
    count = score_catalog(output.log_frequencies, torch.tensor([3, 4, 1, 2, 0, 0]), expected_catalog_size=6)
    joint = score_joint(law.states, law.log_weights.exp(), float(output.log_frequencies[0].exp()),
                        observed, weights, .3)
    assert all(math.isfinite(value) and value >= 0 for value in
               (state["state_energy"], state["shape_mmd2"], count["count_ce_minus_entropy"], joint["joint_mse"]))


def test_mixture_prefix_recomputes_component_proposal_not_truncated_weights():
    m, s = model("F2"), source()
    with torch.no_grad():
        m.mixture_logits.copy_(torch.tensor([-3., -1., 1., 2.]))
        for component, mapping in enumerate(m.components):
            mapping.weight_reference.weight.zero_()
            mapping.weight_reference.bias.fill_(math.atanh((component + 1.) / 6.))
            mapping.weight_modulation.weight.zero_(); mapping.weight_modulation.bias.zero_()
    result = run(m, s)
    short, full = result.prefixes[5].guide_laws[0], result.prefixes[8].guide_laws[0]
    expected_mass = torch.logsumexp(torch.log_softmax(m.mixture_logits.double(), 0) + torch.tensor([.5, 1., 1.5, 2.], dtype=torch.float64), 0)
    assert abs(short.log_mass_increment - expected_mass.item()) < 5e-7
    assert abs(full.log_mass_increment - expected_mass.item()) < 5e-7
    wrongly_truncated = full.log_weights[:5] - torch.logsumexp(full.log_weights[:5], 0)
    assert not torch.allclose(short.log_weights, wrongly_truncated, atol=1e-3, rtol=0)
    torch.testing.assert_close(short.uncalibrated_states, full.uncalibrated_states[:5], atol=0, rtol=0)


def test_source_draws_are_actual_and_pooled_control_is_cell_equal_not_capture_weighted():
    m, s = model("F3"), source()
    captured = []
    original = sample_batch
    def record(requests, *args, **kwargs):
        captured.extend(request.law for request in requests)
        return original(requests, *args, **kwargs)
    with patch("frontier.forecast.sample_batch", side_effect=record), patch.object(m, "rollout", wraps=m.rollout) as rollout:
        result = run(m, s)
    assert rollout.call_args_list[-1].kwargs["indices"].tolist() == [-1]
    pool = captured[-1]
    assert len(pool.states) == 4  # NTC source depths are1 and3, capture counts100 and3.
    np.testing.assert_array_equal(pool.states, s.states[s.offsets[2]:s.offsets[4]])
    np.testing.assert_array_equal(pool.weights, np.full(4, .25))
    assert pool.identity.pool_id == "D1::pooled-NTC-RNA"
    assert result.receipt["pooled_control_is_catalog_entry"] is False
    assert len(result.prefixes[8].log_frequencies) == 6
    assert any(l.identity.actual_guide_id == "g1" for l in captured[:-1])  # unseen target is not dropped


def test_one_maximum_rollout_per_batch_and_one_prefix_reduction_per_batch():
    m, s = model("F2"), source()
    with patch.object(m, "rollout", wraps=m.rollout) as rollout, \
         patch.object(m, "prefix_rollout", wraps=m.prefix_rollout) as prefix:
        run(m, s)
    assert rollout.call_count == 3  # two actual-guide batches plus one pool
    assert prefix.call_count == 6   # two prefixes per batch, not per row
    assert all(call.args[0].shape[1] == 8 for call in rollout.call_args_list)


def test_missing_source_pool_and_all_unavailable_state_are_not_imputed():
    for s in (source(missing_pool=True), source(all_missing=True)):
        result = run(s=s)
        for out in result.prefixes.values():
            assert out.pooled_control is None
            assert out.pooled_control_unavailable_reason == "missing_positive_RNA_source_NTC_pool"
            assert len(out.log_frequencies) == 6
        if not s.positive_counts.any():
            assert not result.prefixes[8].guide_laws
            torch.testing.assert_close(result.prefixes[8].log_frequencies, result.prefixes[8].log_source_frequencies)


def test_microbatch_change_preserves_common_draws_and_close_fp32_results():
    m, s = model("F2"), source()
    a, b = run(m, s, resource=limits(microbatch_laws=1)), run(m, s, resource=limits(microbatch_laws=4))
    for p in a.prefixes:
        for i in a.prefixes[p].guide_laws:
            torch.testing.assert_close(a.prefixes[p].guide_laws[i].states, b.prefixes[p].guide_laws[i].states, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(a.prefixes[p].log_frequencies, b.prefixes[p].log_frequencies, atol=2e-6, rtol=2e-6)


def test_frozen_state_global_rng_unchanged_and_no_optimizer_calls():
    m, s = model("F5"), source()
    state, random_before, np_before, torch_before = model_state_digest(m), random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with patch.object(torch.optim.Optimizer, "step", side_effect=AssertionError("optimizer forbidden")):
        run(m, s)
    assert model_state_digest(m) == state and random.getstate() == random_before
    now = np.random.get_state()
    assert np_before[0] == now[0] and np.array_equal(np_before[1], now[1]) and np_before[2:] == now[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    s.assert_unchanged()


def test_model_mutation_during_generation_invalidates_result():
    m, s = model(), source()
    original = m.rollout
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with torch.no_grad(): m.reference_embedding.add_(.001)
        return result
    with patch.object(m, "rollout", side_effect=changed), pytest.raises(ValueError, match="changed during forecast"):
        run(m, s)


@pytest.mark.parametrize("damage", ["state", "representation", "fold", "purpose", "model_binding", "not_frozen", "seed", "stage"])
def test_changed_bindings_or_unfrozen_model_rejected_before_rollout(damage):
    m, s = model(), source()
    b, c = binding(m, s), context(m, s)
    if damage == "state":
        with torch.no_grad(): m.reference_embedding.add_(.1)
    elif damage == "representation": b = replace(b, observation_binding=ObservationBinding("b" * 64, "c" * 64))
    elif damage == "fold": b = replace(b, fold=FoldInputs("subset-D2-D4", ("D2", "D4")))
    elif damage == "purpose": c = replace(c, purpose="training")
    elif damage == "model_binding": b = replace(b, model_state_sha256="f" * 64)
    elif damage == "seed": c = replace(c, model_seed=1)
    elif damage == "stage": c = replace(c, accepted_update=3072)
    else: m.train()
    with patch.object(m, "rollout", side_effect=AssertionError("rollout before admission")), pytest.raises(ValueError):
        forecast_population(m, s, b, c, Recipe("F0-V0"), prefixes=(8,), limits=limits())


@pytest.mark.parametrize("cap", ["maximum_host_bytes", "maximum_output_bytes", "maximum_device_working_bytes", "maximum_pool_cells"])
def test_resource_failure_precedes_sampling_or_rollout(cap):
    with patch("frontier.forecast.sample_batch", side_effect=AssertionError("sampling before budget")), pytest.raises(ValueError):
        run(resource=limits(**{cap: 1}))


def test_multiple_prefix_output_is_explicitly_charged_and_fixture_bounds_fixed():
    single, multiple = run(prefixes=(8,)), run()
    assert multiple.receipt["resources"]["output_bytes"] > single.receipt["resources"]["output_bytes"]
    with pytest.raises(ValueError, match="bounded CPU"):
        run(prefixes=(129,))
    with pytest.raises(ValueError, match="Ascending"):
        run(prefixes=(8, 5))
    with pytest.raises(ValueError):
        Recipe("F0-V0", alpha=.2)


def test_memory_estimate_charges_complete_source_support_not_particle_depth():
    narrow = run(prefixes=(5,)).receipt["resources"]
    deeper = run(prefixes=(8,)).receipt["resources"]
    wider = run(prefixes=(8,), resource=limits(microbatch_laws=4)).receipt["resources"]
    # Individual supports are 2,1,1,3; the actual NTC source pool has 4 cells.
    assert narrow["maximum_batch_source_cells"] == 5
    assert wider["maximum_batch_source_cells"] == 7
    assert narrow["source_support_working_bytes"] == 16 * 5 + 128 * 4
    assert deeper["source_support_working_bytes"] == narrow["source_support_working_bytes"]
    assert wider["source_support_working_bytes"] == 16 * 7 + 128 * 4
    assert narrow["retained_sampler_metadata_bytes"] > 0
    assert narrow["caller_retained_RNA_means_decoder_and_other_banks_included"] is False
    assert narrow["estimates_not_measured_peak"] is True


def test_damaged_source_hash_and_fitting_donor_query_rejected():
    s = source()
    with pytest.raises(ValueError, match="content identity"):
        replace(s, states=ro(s.states + 1., np.float32))
    with pytest.raises(ValueError, match="held-out"):
        replace(s, reference=replace(s.reference, donor="D3"))
