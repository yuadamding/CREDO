"""Synthetic dense-reference agreement and changed scoring paths only."""
# ruff: noqa: E402
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from scipy.special import softmax

from full_evaluation import (
    weighted_v_energy, categorical_scores, endpoint_effect_decomposition,
    decoded_mean_composition,
)


def empirical():
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(13, 48)), rng.normal(size=(9, 48)) + .2
    logp, q = rng.normal(size=len(x)), rng.uniform(.1, 2, size=len(y))
    return x, logp, y, q


def dense_reference(x, logp, y, q):
    p, q = softmax(logp), q / q.sum()
    def mean(a, b, w, v):
        distance = np.sqrt(np.mean(np.square(a[:, None] - b[None, :]), axis=-1))
        return float(w @ distance @ v)
    return 2 * mean(x, y, p, q) - mean(x, x, p, p) - mean(y, y, q, q)


@pytest.mark.parametrize("tile", [1, 2, 4, 16, 256])
def test_tiled_weighted_v_energy_matches_independent_numpy_dense_reference(tile):
    values = empirical()
    row = weighted_v_energy(*values, block_size=tile, device="cpu")
    assert row["energy"] == pytest.approx(dense_reference(*values), abs=1e-12, rel=0)
    assert row["distance"] == "RMS_Euclidean"
    assert row["dtype"] == "float64"


def test_zero_and_single_atom_analytic_limits():
    x, logp, _, _ = empirical()
    assert weighted_v_energy(x, logp, x, softmax(logp), device="cpu")["energy"] == pytest.approx(0, abs=1e-12)
    assert weighted_v_energy(np.zeros((1, 48)), [0], np.full((1, 48), 2), [1], device="cpu")["energy"] == pytest.approx(4)


def test_v_statistic_does_not_exclude_diagonal_or_use_u_divisors():
    assert weighted_v_energy([[0], [2]], [0, 0], [[1]], [1], device="cpu")["energy"] == pytest.approx(1)


def test_permutation_weight_scale_and_translation_invariance():
    x, logp, y, q = empirical()
    baseline = weighted_v_energy(x, logp, y, q, block_size=3, device="cpu")["energy"]
    translated = weighted_v_energy(x[::-1] + 1e5, logp[::-1], y[::-1] + 1e5, q[::-1] * 100, block_size=4, device="cpu")
    assert translated["energy"] == pytest.approx(baseline, abs=1e-11, rel=0)


def test_pairwise_workspace_never_uses_full_support_matrices(monkeypatch):
    calls = []
    original = torch.cdist
    def checked(x, y, **kwargs):
        calls.append((len(x), len(y)))
        assert len(x) <= 4 and len(y) <= 4
        assert x.dtype == y.dtype == torch.float64
        return original(x, y, **kwargs)
    monkeypatch.setattr(torch, "cdist", checked)
    row = weighted_v_energy(*empirical(), block_size=4, device="cpu")
    assert len(calls) > 3
    assert row["largest_distance_tile_bytes"] == 4 * 4 * 8


@pytest.mark.parametrize("kind", ["axis", "nonfinite", "missing", "weights", "tile", "underflow"])
def test_bad_energy_inputs_rejected(kind):
    x, logp, y, q = empirical()
    options = dict(block_size=4, device="cpu")
    if kind == "axis":
        y = y[:, :-1]
    elif kind == "nonfinite":
        x[0, 0] = np.inf
    elif kind == "missing":
        y, q = y[:0], q[:0]
    elif kind == "weights":
        q[0] = -1
    elif kind == "tile":
        options["block_size"] = True
    else:
        logp[0] = -1000
    with pytest.raises(ValueError):
        weighted_v_energy(x, logp, y, q, **options)


def test_complete_catalog_count_scores_preserve_raw_zeros():
    counts = np.arange(26504, dtype=np.int64) % 4
    frequency = (counts + .5) / (counts.sum() + .5 * len(counts))
    row = categorical_scores(np.log(frequency), counts)
    observed = counts / counts.sum()
    positive = observed > 0
    assert row["catalog_size"] == 26504 and row["endpoint_zero_categories"] == 6626
    assert row["cross_entropy_nats"] == pytest.approx(-sum(p * logq for p, logq in zip(observed, np.log(frequency), strict=True)))
    assert row["endpoint_entropy_nats"] == pytest.approx(-(observed[positive] * np.log(observed[positive])).sum())
    assert row["ce_minus_entropy_nats"] > 0
    assert counts[0] == 0


def test_exact_count_forecast_kl_is_zero_and_capture_depth_does_not_rescale_scores():
    counts = np.array([2, 1, 4])
    expected = categorical_scores(np.log(counts / counts.sum()), counts)
    scaled = categorical_scores(np.log(counts / counts.sum()), counts * 100)
    assert expected["ce_minus_entropy_nats"] == pytest.approx(0, abs=1e-14)
    for key in ("cross_entropy_nats", "endpoint_entropy_nats", "quadratic_score"):
        assert expected[key] == scaled[key]


@pytest.mark.parametrize("counts,logp", [([0, 0], np.log([.5, .5])),
    ([.5, 1.5], np.log([.5, .5])), ([-1, 2], np.log([.5, .5])),
    ([1, 2], np.log([.3, .3])), ([1, 2], [0, -np.inf]),
    ([1, 2, 3], np.log([.5, .5]))])
def test_bad_count_axes_totals_normalization_and_zeros_rejected(counts, logp):
    with pytest.raises(ValueError):
        categorical_scores(logp, counts)


def test_legitimate_endpoint_cancellation_is_reported_not_erased():
    truth = np.array([.2, .3, .5])
    forecast = np.array([.3, .2, .5])
    row = endpoint_effect_decomposition(forecast, forecast, truth, truth)
    assert row["target_endpoint_mse"] > 0 and row["own_control_endpoint_mse"] > 0
    assert row["effect_mse"] == 0 and row["interaction"] < 0
    assert abs(row["decomposition_residual"]) < 1e-14
    assert row["target_endpoint_mse"] + row["own_control_endpoint_mse"] == pytest.approx(
        2 * row["common_endpoint_component_mse"] + .5 * row["effect_mse"])


def test_endpoint_gene_weights_and_before_log_composition_definition():
    q = [[.1, .2, .7], [.2, .2, .6], [.1, .3, .6], [.3, .2, .5]]
    row = endpoint_effect_decomposition(*q, gene_weights=[1, 2, 0])
    eg = np.log1p(1e4 * np.array(q[0])) - np.log1p(1e4 * np.array(q[2]))
    e0 = np.log1p(1e4 * np.array(q[1])) - np.log1p(1e4 * np.array(q[3]))
    assert row["effect_mse"] == pytest.approx(np.array([1/3, 2/3, 0]) @ np.square(eg - e0))


def test_decoder_batches_match_dense_weighted_composition_not_mean_log():
    states = np.arange(30, dtype=np.float32).reshape(10, 3) / 10
    weights = np.arange(10, dtype=np.float64) / 10
    sizes = []
    def decode(z):
        sizes.append(len(z))
        assert not torch.is_grad_enabled()
        return torch.softmax(torch.cat((z, z[:, :2]), dim=1), dim=1)
    result = decoded_mean_composition(states, weights, decode, gene_dimension=5, batch_rows=3, device="cpu")
    dense = torch.softmax(torch.cat((torch.tensor(states), torch.tensor(states[:, :2])), dim=1), dim=1)
    expected = softmax(weights) @ dense.double().numpy()
    assert np.allclose(result, expected, atol=1e-12, rtol=0)
    assert sizes == [3, 3, 3, 1]
    assert result.sum() == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("kind", ["wrong_axis", "nonfinite", "negative", "unnormalized"])
def test_invalid_decoder_outputs_are_not_repaired(kind):
    def decode(z):
        result = torch.ones(len(z), 3) / 3
        if kind == "wrong_axis":
            result = result[:, :2]
        elif kind == "nonfinite":
            result[0, 0] = np.nan
        elif kind == "negative":
            result[0, 0] = -.1
        else:
            result *= .9
        return result
    with pytest.raises(ValueError):
        decoded_mean_composition(np.ones((4, 2)), np.zeros(4), decode, gene_dimension=3, batch_rows=2, device="cpu")
