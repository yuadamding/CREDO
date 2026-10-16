"""Fabricated CPU arithmetic only: no optimizer, data reader or GPU calls."""

from dataclasses import asdict, replace
import math
from unittest.mock import patch

import pytest
import torch

from frontier import evaluation
from frontier.evaluation import (MMD_BANDWIDTHS, ScoreIdentity, mean_seed_scores,
    reduce_scores, score_catalog, score_joint, score_molecular_law, score_state_law)
from frontier.measure_ops import nonnegative_score
from frontier.selection import CHANNELS


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def identity(seed=0, prefix=3):
    return ScoreIdentity("subset-D3-D4", "D2", "F0-V0", seed, prefix,
        "1"*64, "2"*64, str(seed+3)*64, "6"*64, "7"*64, "8"*64)


def law(values, weights):
    return torch.tensor(values, dtype=torch.float64), torch.tensor(weights, dtype=torch.float64)


def reference_pair(x, w, y, v):
    distance = ((x[:, None, :]-y[None, :, :]).square().mean(-1)).sqrt()
    return w @ distance @ v


def fixture_laws():
    return (law([[0., 1.], [2., 1.], [3., 4.]], [.2, .3, .5]),
            law([[1., 0.], [2., 5.]], [.4, .6]),
            law([[-1., -2.], [0., 1.], [3., 1.], [2., 2.]], [.1, .2, .3, .4]))


def test_state_energy_spread_and_mmd_match_explicit_weighted_v_formulas():
    (x, w), (y, v), (s, u) = fixture_laws()
    row = score_state_law(x, w, y, v, s, u, identity=identity(), block=1)
    xx, yy, ss = reference_pair(x, w, x, w), reference_pair(y, v, y, v), reference_pair(s, u, s, u)
    xy, sy = reference_pair(x, w, y, v), reference_pair(s, u, y, v)
    assert row["state_energy"] == pytest.approx(float(2*xy-xx-yy), abs=2e-15)
    assert row["source_energy"] == pytest.approx(float(2*sy-ss-yy), abs=2e-15)
    assert row["predicted_self_mean"] == pytest.approx(float(xx))
    assert row["observed_self_mean"] == pytest.approx(float(yy))
    assert row["source_self_mean"] == pytest.approx(float(ss))
    assert row["spread_absolute_error"] == pytest.approx(float(abs(xx-yy)))
    assert row["raw_energy_difference"] == pytest.approx(row["cross_contribution"]+row["spread_contribution"])
    def kernel(a, wa, b, wb):
        d2 = (a[:, None, :]-b[None, :, :]).square().mean(-1)
        k = sum(torch.exp(-d2/(2*h*h)) for h in MMD_BANDWIDTHS)/4
        return wa @ k @ wb
    expected = kernel(x, w, x, w)+kernel(y, v, y, v)-2*kernel(x, w, y, v)
    assert row["shape_mmd2"] == pytest.approx(float(expected), abs=2e-15)
    assert row["dtype"] == "float64" and row["observed_cells"] == 2
    assert row["roundoff"]["state_energy"]["raw_risk"] == pytest.approx(float(2*xy-xx-yy))


def test_state_block_and_global_translation_invariant_no_support_dropped():
    (x, w), (y, v), (s, u) = fixture_laws()
    # A zero-weight cell remains part of the complete observed support.
    y = torch.cat((y, torch.tensor([[88., -33.]], dtype=torch.float64)))
    v = torch.cat((v, torch.zeros(1, dtype=torch.float64)))
    rows = [score_state_law(x+a, w, y+a, v, s+a, u, block=b) for a, b in ((0, 1), (0, 128), (32, 2))]
    for name in ("state_energy", "spread_absolute_error", "shape_mmd2", "source_energy"):
        assert rows[0][name] == pytest.approx(rows[1][name], abs=2e-15)
        assert rows[0][name] == pytest.approx(rows[2][name], abs=2e-15)
    assert rows[0]["observed_cells"] == 3


def test_source_diagnostics_optional_preserves_scores_and_reduces_pair_calls():
    (x, w), (y, v), (s, u) = fixture_laws()
    with patch.object(evaluation, "_pair_mean", wraps=evaluation._pair_mean) as calls:
        complete = score_state_law(x, w, y, v, s, u, block=2)
        assert calls.call_count == 5
    with patch.object(evaluation, "_pair_mean", wraps=evaluation._pair_mean) as calls:
        principal = score_state_law(x, w, y, v, None, None, block=2, include_source_diagnostics=False)
        assert calls.call_count == 3
    for name in ("state_energy", "spread_absolute_error", "shape_mmd2", "predicted_self_mean", "observed_self_mean"):
        assert principal[name] == complete[name]
    assert complete["include_source_diagnostics"] is True
    assert principal["include_source_diagnostics"] is False and principal["source_cells"] is None
    assert "source_energy" not in principal and "source_energy" not in principal["roundoff"]
    for first, second in ((s, u), (s, None), (None, u)):
        with pytest.raises(ValueError, match="both source arguments"):
            score_state_law(x, w, y, v, first, second, include_source_diagnostics=False)
    with pytest.raises(ValueError, match="boolean"):
        score_state_law(x, w, y, v, s, u, include_source_diagnostics=1)


def test_identical_law_zero_and_missing_truth_unavailable_not_zero():
    (x, w), _, (s, u) = fixture_laws()
    row = score_state_law(x, w, x, w, s, u, block=2)
    assert row["state_energy"] == pytest.approx(0, abs=2e-15)
    assert row["shape_mmd2"] == pytest.approx(0, abs=2e-15)
    assert row["spread_absolute_error"] == 0
    missing = score_state_law(x, w, None, None, s, u)
    for key in ("state_energy", "spread_absolute_error", "shape_mmd2"):
        assert missing[key] is None and missing["unavailable"][key] == "missing_observed_state_law"
    assert missing["arithmetic_only"] is True


def test_invalid_weights_prefix_and_empty_law_are_not_repaired():
    (x, w), (y, v), (s, u) = fixture_laws()
    for replacement in (w*2, torch.tensor([.2, -.3, 1.1], dtype=torch.float64)):
        with pytest.raises(ValueError, match="normalized"):
            score_state_law(x, replacement, y, v, s, u)
    with pytest.raises(ValueError, match="prefix"):
        score_state_law(x, w, y, v, s, u, identity=identity(prefix=2))
    with pytest.raises(ValueError, match="nonempty"):
        score_state_law(x, w, y[:0], v[:0], s, u)
    with pytest.raises(ValueError, match="both"):
        score_state_law(x, w, y, None, s, u)


def test_full_catalog_ce_entropy_quadratic_with_observed_zeros():
    pcount = torch.tensor([4, 0, 2, 0, 3])
    q = torch.tensor([.2, .1, .3, .15, .25], dtype=torch.float64)
    row = score_catalog(q.log(), pcount, expected_catalog_size=5, identity=identity())
    p = pcount.to(torch.float64)/9
    ce = -(p*q.log()).sum()
    entropy = -(p[p>0]*p[p>0].log()).sum()
    assert row["count_ce_minus_entropy"] == pytest.approx(float(ce-entropy))
    assert row["count_quadratic"] == pytest.approx(float(1+q.square().sum()-2*(p*q).sum()))
    assert row["count_quadratic"] != pytest.approx(float((p-q).square().sum()))
    assert row["count_quadratic_excess"] == pytest.approx(float((p-q).square().sum()))
    assert row["count_quadratic"] == pytest.approx(row["count_quadratic_empirical_minimum"]+row["count_quadratic_excess"])
    assert row["captured_cells"] == 9 and row["endpoint_zero_categories"] == 2
    assert row["catalog_size"] == 5
    full = torch.zeros(26504, dtype=torch.int64)
    full[3], full[20000] = 2, 1
    fullrow = score_catalog(torch.full((26504,), -math.log(26504), dtype=torch.float64), full, expected_catalog_size=26504)
    assert fullrow["catalog_size"] == 26504 and fullrow["endpoint_zero_categories"] == 26502


def test_catalog_fails_wrong_denominator_underflow_and_zero_total():
    counts = torch.tensor([2, 0, 4])
    logq = torch.log_softmax(torch.tensor([1., 2., 3.], dtype=torch.float64), 0)
    with pytest.raises(ValueError, match="complete"):
        score_catalog(logq, counts, expected_catalog_size=4)
    with pytest.raises(ValueError, match="normalized"):
        score_catalog(logq+1, counts, expected_catalog_size=3)
    with pytest.raises(ValueError, match="Underflowed"):
        score_catalog(torch.log_softmax(torch.tensor([0., -2000., -1.], dtype=torch.float64), 0), counts, expected_catalog_size=3)
    with pytest.raises(ValueError, match="total"):
        score_catalog(logq, torch.zeros(3, dtype=torch.int64), expected_catalog_size=3)


def test_joint_constant_and_nonconstant_parts_match_49_feature_identity():
    generator = torch.Generator().manual_seed(31)
    x, y = torch.randn((3, 48), generator=generator, dtype=torch.float64), torch.randn((5, 48), generator=generator, dtype=torch.float64)
    w, v = torch.tensor([.2, .3, .5], dtype=torch.float64), torch.full((5,), .2, dtype=torch.float64)
    row = score_joint(x, w, .08, y, v, .12, identity=identity())
    phi = torch.cat((torch.ones(1, dtype=torch.float64), w@x.tanh()))
    truth = torch.cat((torch.ones(1, dtype=torch.float64), v@y.tanh()))
    expected = (.08*phi-.12*truth).square()
    assert row["joint_mse"] == pytest.approx(float(expected.mean()))
    assert row["joint_mass_mse"] == pytest.approx(.04**2)
    assert 49*row["joint_mse"] == pytest.approx(row["joint_mass_mse"]+48*row["joint_nonconstant_mse"])
    missing = score_joint(x, w, .08, None, None, 0)
    assert missing["joint_mse"] is None and missing["joint_mass_mse"] == pytest.approx(.08**2)
    with pytest.raises(ValueError, match="48-state"):
        score_joint(x[:, :2], w, .08, y[:, :2], v, .12)


def test_molecular_endpoints_effect_and_common_error_decomposition():
    pt, ot, pc, oc = (torch.tensor(v, dtype=torch.float64) for v in
        ([.2, .3, .5], [.1, .5, .4], [.3, .4, .3], [.2, .6, .2]))
    row = score_molecular_law(pt, ot, pc, oc)
    et, ec = torch.log1p(10000*pt)-torch.log1p(10000*ot), torch.log1p(10000*pc)-torch.log1p(10000*oc)
    assert row["target_endpoint_mse"] == pytest.approx(float(et.square().mean()))
    assert row["pooled_control_endpoint_mse"] == pytest.approx(float(ec.square().mean()))
    assert row["pooled_effect_mse"] == pytest.approx(float((et-ec).square().mean()))
    assert row["effect_decomposition_residual"] == pytest.approx(0, abs=2e-15)
    assert row["endpoint_error_interaction"] == pytest.approx(float(-2*(et*ec).mean()))
    assert row["control_endpoint_observations"] == 1


def test_molecular_valid_endpoint_survives_missing_partner():
    q = torch.tensor([.2, .3, .5], dtype=torch.float64)
    target = score_molecular_law(q, q, q, None)
    assert target["target_endpoint_mse"] == 0
    assert target["pooled_control_endpoint_mse"] is None and target["pooled_effect_mse"] is None
    control = score_molecular_law(None, q, q, q)
    assert control["target_endpoint_mse"] is None and control["pooled_control_endpoint_mse"] == 0
    assert control["zero_effect_mse"] == 0
    with pytest.raises(ValueError, match="Unnormalized"):
        score_molecular_law(q*2, q, q, q)


def synthetic_rows(binding, first=1., second=3.):
    return {key: dict(identity=asdict(binding), arithmetic_only=False, unavailable={},
                      **{name: value for name in CHANNELS}) for key, value in (("guide-a", first), ("guide-b", second))}


def fixed_weights():
    weights = {name: {"guide-a": .25, "guide-b": .75} for name in CHANNELS}
    # One physical pooled control is deliberately counted once.
    weights["pooled_control_endpoint_mse"] = {"guide-a": 1.}
    return weights


def test_fixed_reducer_respects_supplied_weights_and_one_control():
    binding = identity()
    result = reduce_scores(synthetic_rows(binding), fixed_weights(), identity=binding)
    assert result["state_energy"] == 2.5
    assert result["pooled_control_endpoint_mse"] == 1.
    assert result["coverage"]["pooled_control_endpoint_mse"]["expected_rows"] == 1
    assert result["selection_available"] is True


def test_fixed_reducer_reports_missing_weight_not_complete_case_mean():
    binding = identity()
    rows = synthetic_rows(binding)
    rows["guide-b"]["state_energy"] = None
    rows["guide-b"]["unavailable"]["state_energy"] = "missing_endpoint_RNA"
    result = reduce_scores(rows, fixed_weights(), identity=binding)
    assert result["state_energy"] is None and result["spread_absolute_error"] == 2.5
    assert result["coverage"]["state_energy"]["available_weight"] == .25
    assert result["selection_available"] is False
    with pytest.raises(ValueError, match="union"):
        reduce_scores({"guide-a": rows["guide-a"]}, fixed_weights(), identity=binding)
    with pytest.raises(ValueError, match="Foreign"):
        reduce_scores(rows, fixed_weights(), identity=replace(binding, particle_prefix=4))
    bad = fixed_weights()
    bad["state_energy"] = {"guide-a": .25}
    with pytest.raises(ValueError, match="sum to one"):
        reduce_scores(rows, bad, identity=binding)


def test_seed_score_mean_is_not_an_ensemble_or_favorable_seed_selection():
    ids = {seed: identity(seed) for seed in (0, 1, 2)}
    records = {seed: reduce_scores(synthetic_rows(ids[seed], seed+1., seed+3.), fixed_weights(), identity=ids[seed]) for seed in ids}
    result = mean_seed_scores(records, identities=ids)
    assert result["state_energy"] == 3.5
    assert result["aggregation"] == "equal_mean_seed_scores_not_prediction_ensemble"
    assert result["biological_replicates_added"] == 0
    with pytest.raises(ValueError, match="inventory"):
        mean_seed_scores({0: records[0]}, identities=ids)
    bad_ids = {**ids, 2: replace(ids[2], representation_sha256="f"*64)}
    with pytest.raises(ValueError, match="mismatched"):
        mean_seed_scores(records, identities=bad_ids)


def test_unavailable_seed_keeps_other_channels_and_reports_coverage():
    ids = {seed: identity(seed) for seed in (0, 1, 2)}
    rows = {seed: synthetic_rows(ids[seed]) for seed in ids}
    rows[2]["guide-a"]["state_energy"] = None
    rows[2]["guide-a"]["unavailable"]["state_energy"] = "forecast_missing"
    records = {seed: reduce_scores(rows[seed], fixed_weights(), identity=ids[seed]) for seed in ids}
    result = mean_seed_scores(records, identities=ids)
    assert result["state_energy"] is None and result["spread_absolute_error"] == 2.5
    assert result["seed_coverage"][2]["state_energy"]["available_weight"] == .75
    assert result["selection_available"] is False


def test_roundoff_is_preserved_and_large_negative_not_floored():
    record = nonnegative_score(torch.tensor(-1e-15, dtype=torch.float64), component_absolute_sum=2.)
    assert record["risk"] == 0 and record["raw_risk"] == -1e-15
    assert record["negative_roundoff_corrected"] is True
    assert record["roundoff_bound"] == 64*torch.finfo(torch.float64).eps*2
    with pytest.raises(ValueError, match="Negative"):
        nonnegative_score(torch.tensor(-1e-5, dtype=torch.float64), component_absolute_sum=2.)


def test_score_identity_requires_explicit_hashes_and_prefix():
    with pytest.raises(ValueError, match="SHA256"):
        replace(identity(), source_sha256="some-source")
    with pytest.raises(ValueError, match="prefix"):
        replace(identity(), particle_prefix=0)


def test_score_identity_preserves_positional_constructor_and_explicit_predictor_kind():
    single = identity()
    assert single.predictor_kind == "single_seed"
    assert single == replace(single, predictor_kind="single_seed")
    mixture = replace(single, seed=None, predictor_kind="three_seed_finite_measure_mixture")
    assert mixture.seed is None
    assert asdict(mixture)["predictor_kind"] == "three_seed_finite_measure_mixture"
    assert mixture.model_sha256 == single.model_sha256
    assert mixture.particle_prefix == single.particle_prefix


@pytest.mark.parametrize("seed", [None, -1, True, 0.0])
def test_single_seed_identity_rejects_noninteger_or_missing_seed(seed):
    with pytest.raises(ValueError, match="nonnegative integer seed"):
        replace(identity(), seed=seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, -1, False])
def test_ensemble_identity_never_masquerades_as_a_seed(seed):
    with pytest.raises(ValueError, match="seed=None"):
        replace(identity(), seed=seed, predictor_kind="three_seed_finite_measure_mixture")


@pytest.mark.parametrize("kind", ["ensemble", "", None])
def test_score_identity_rejects_unknown_predictor_kind(kind):
    with pytest.raises(ValueError, match="predictor kind"):
        replace(identity(), predictor_kind=kind)


def test_fixed_reducer_preserves_separate_ensemble_identity():
    binding = replace(identity(), seed=None, predictor_kind="three_seed_finite_measure_mixture")
    result = reduce_scores(synthetic_rows(binding), fixed_weights(), identity=binding)
    assert result["state_energy"] == 2.5
    assert result["identity"]["seed"] is None
    assert result["identity"]["predictor_kind"] == "three_seed_finite_measure_mixture"
    with pytest.raises(ValueError, match="Foreign"):
        reduce_scores(synthetic_rows(binding), fixed_weights(), identity=identity())


@pytest.mark.parametrize("mixed", [False, True])
def test_seed_score_mean_explicitly_rejects_ensemble_identity(mixed):
    ensemble = replace(identity(), seed=None, predictor_kind="three_seed_finite_measure_mixture")
    ids = {0: ensemble} if not mixed else {0: identity(), 1: ensemble}
    records = {seed: reduce_scores(synthetic_rows(binding), fixed_weights(), identity=binding)
               for seed, binding in ids.items()}
    with pytest.raises(ValueError, match="single-seed.*ensemble"):
        mean_seed_scores(records, identities=ids)
