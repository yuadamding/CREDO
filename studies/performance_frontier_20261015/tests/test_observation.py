"""Fabricated CPU observation checks; zero optimizer calls or historical loads."""

import numpy as np
import pytest
from scipy import sparse
import torch

from frontier.observation import (ObservationFold, SparseR48, symmetric_umi_halves,
                                  technical_holdout, validate_counts)


def fold():
    return ObservationFold("D3D4", ("D3", "D4"), "a" * 64)


def model(**kwargs):
    return SparseR48(fold(), features=7, hidden_dims=(9, 5), latent_dim=3,
                     maximum_rows=8, **kwargs)


def counts():
    return sparse.csr_matrix(np.asarray([[1, 0, 3, 7, 0, 2, 0], [0, 0, 0, 0, 0, 0, 0],
                                         [5, 1, 0, 0, 2, 1, 4], [0, 2, 4, 1, 0, 3, 5]], dtype=np.int32))


def fit_scale(value):
    return value.fit_scaling([(counts(), ["D3", "D3", "D4", "D4"],
                              ["Stim8hr", "Stim8hr", "Stim48hr", "Stim48hr"])])


def test_full_architecture_fresh_initialization_and_no_rng_change():
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()
    full = SparseR48(fold())
    assert torch.equal(before, torch.get_rng_state())
    linears = [layer for layer in full.modules() if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [
        (18129, 512), (512, 128), (128, 48), (48, 128), (128, 512), (512, 18129)]
    expected = sum((a + 1) * b for a, b in [(18129, 512), (512, 128), (128, 48),
                                          (48, 128), (128, 512), (512, 18129)]) + 4 * (512 + 128)
    assert sum(parameter.numel() for parameter in full.parameters()) == expected
    assert sum(isinstance(layer, torch.nn.LayerNorm) for layer in full.modules()) == 4
    assert sum(isinstance(layer, torch.nn.GELU) for layer in full.modules()) == 4
    assert not bool(full.scaling_ready)
    assert torch.equal(full.center, torch.zeros(48))
    assert torch.equal(full.scale, torch.ones(48))
    a, b = model(seed=4), model(seed=4)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    c = model(seed=5)
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters()))


def test_cpu_sparse_projection_matches_dense_formula_without_densifying(monkeypatch):
    value, matrix = model(), counts()
    dense = torch.as_tensor(matrix.toarray(), dtype=torch.float32)
    normalized = torch.log1p(10000 * dense / dense.sum(1, keepdim=True).clamp_min(1))
    expected = value.encoder(value.first(normalized))

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU sparse first layer densified")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    actual = value.raw_encode(matrix)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual.square().sum().backward()
    assert value.first.weight.grad is not None
    assert bool(torch.isfinite(value.first.weight.grad).all())


def test_fold_training_guard_rest_and_heldout_denied():
    value = model()
    for donors, conditions in [(["D1"] * 4, ["Stim8hr"] * 4),
                               (["D3"] * 4, ["Rest"] * 4),
                               (["D3"] * 3, ["Stim8hr"] * 4)]:
        with pytest.raises(ValueError):
            value.symmetric_count_loss(counts(), row_ids=["a", "b", "c", "d"],
                                       donors=donors, conditions=conditions)
        with pytest.raises(ValueError):
            value.fit_scaling([(counts(), donors, conditions)])
        assert not bool(value.scaling_ready)
    with pytest.raises(ValueError):
        ObservationFold("all", ("D1", "D2", "D3", "D4"), "a" * 64)
    with pytest.raises(ValueError):
        ObservationFold("pair", ("D4", "D3"), "a" * 64)


def test_row_hash_split_is_order_invariant_and_outcome_free():
    ids = [f"D3:Stim8hr:cell-{index}" for index in range(64)]
    result = technical_holdout(ids, fold=fold(), seed=8, fraction=0.3)
    assert result.dtype == bool and 0 < result.sum() < len(ids)
    assert np.array_equal(result[::-1], technical_holdout(ids[::-1], fold=fold(), seed=8, fraction=0.3))
    assert np.array_equal(result[:20], technical_holdout(ids[:20], fold=fold(), seed=8, fraction=0.3))
    with pytest.raises(ValueError, match="Duplicate"):
        technical_holdout(["same", "same"], fold=fold())


def test_symmetric_thinning_conserves_counts_and_batch_identity():
    matrix, ids = counts(), ["D3:8:a", "D3:8:b", "D4:48:c", "D4:48:d"]
    first, second = symmetric_umi_halves(matrix, ids, seed=7)
    assert (first + second != matrix).nnz == 0
    assert first.dtype == second.dtype == np.int32
    assert np.all(first.data >= 0) and np.all(second.data >= 0)
    assert first[1].nnz == second[1].nnz == 0
    order = np.array([3, 1, 0, 2])
    reordered = symmetric_umi_halves(matrix[order], [ids[index] for index in order], seed=7)
    assert (first[order] != reordered[0]).nnz == (second[order] != reordered[1]).nnz == 0
    separate = symmetric_umi_halves(matrix[:2], ids[:2], seed=7)
    assert (first[:2] != separate[0]).nnz == 0


def test_symmetric_count_loss_uses_own_half_depth_and_equal_directions():
    value, matrix = model(), counts()
    result = value.symmetric_count_loss(matrix, row_ids=["a", "b", "c", "d"],
                                       donors=["D3"] * 4, conditions=["Stim8hr"] * 4, thinning_seed=3)
    numerator, denominator = 0, 0
    a, b = result["halves"]
    for inputs, targets in ((a, b), (b, a)):
        log_probability = torch.log_softmax(value.decoder(value.raw_encode(inputs)), -1)
        target = torch.as_tensor(targets.toarray(), dtype=torch.float32)
        depth = target.sum(1)
        valid = depth > 0
        numerator = numerator + (-(target * log_probability).sum(1)[valid] / depth[valid]).sum()
        denominator += int(valid.sum())
    assert result["scored_cell_directions"] == denominator
    torch.testing.assert_close(result["loss"], numerator / denominator)
    result["loss"].backward()
    assert value.first.weight.grad is not None and value.decoder[-1].weight.grad is not None
    empty = value.symmetric_count_loss(sparse.csr_matrix((2, 7), dtype=np.int32), row_ids=["x", "y"],
                                      donors=["D3"] * 2, conditions=["Stim8hr"] * 2)
    assert empty["loss"] is None and empty["scored_cell_directions"] == 0
    assert empty["zero_scored_directions"] == 4


def test_streaming_fold_scaling_matches_population_moments_and_is_once_only():
    value, matrix = model(), counts()
    raw = value.raw_encode(matrix).detach().double()[[0, 2, 3]]
    expected_mean, expected_std = raw.mean(0), raw.std(0, correction=0)
    result = value.fit_scaling([(matrix[:2], ["D3"] * 2, ["Stim8hr"] * 2),
                               (matrix[2:], ["D4"] * 2, ["Stim48hr"] * 2)])
    assert result["positive_RNA_cells"] == 3
    torch.testing.assert_close(value.center.double(), expected_mean, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(value.scale.double(), expected_std.clamp_min(result["scale_floor"]), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(value.encode(matrix), (value.raw_encode(matrix) - value.center) / value.scale)
    with pytest.raises(ValueError, match="once-only"):
        fit_scale(value)
    unready = model()
    with pytest.raises(ValueError, match="not finalized"):
        unready.encode(matrix)


@pytest.mark.parametrize("gene_block", [1, 3, 7])
@pytest.mark.parametrize("recompute", [True, False])
def test_streamed_allgene_global_softmax_values_and_input_weight_gradients(gene_block, recompute):
    value = model()
    fit_scale(value)
    value.freeze()
    states = torch.arange(21, dtype=torch.float32).reshape(7, 3).div(17).requires_grad_()
    weights = torch.tensor([1, 2, 0, 4, 2, 1, 3], dtype=torch.float64, requires_grad=True)
    streamed = value.weighted_composition(states, weights, gene_block=gene_block, row_block=2,
                                         recompute_backward=recompute)
    raw = states * value.scale + value.center
    expected = (torch.softmax(value.decoder(raw).double(), -1) * (weights / weights.sum())[:, None]).sum(0)
    torch.testing.assert_close(streamed, expected, rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(streamed.sum(), torch.tensor(1.0, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    readout = torch.arange(7, dtype=torch.float64).square()
    actual_grad = torch.autograd.grad((streamed * readout).sum(), (states, weights), retain_graph=True)
    expected_grad = torch.autograd.grad((expected * readout).sum(), (states, weights))
    for actual, desired in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, desired, rtol=5e-5, atol=2e-7)
        assert bool(torch.isfinite(actual).all()) and bool((actual != 0).any())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in value.parameters())


def test_invalid_count_and_working_budget_guards():
    matrix = counts()
    with pytest.raises(TypeError, match="Sparse"):
        model().raw_encode(matrix.toarray())
    with pytest.raises(ValueError, match="integer"):
        model().raw_encode(matrix.astype(np.float32))
    bad = matrix.copy()
    bad.data[0] = -1
    with pytest.raises(ValueError, match="Nonnegative"):
        model().raw_encode(bad)
    with pytest.raises(ValueError, match="bounded minibatch"):
        validate_counts(matrix, features=7, maximum_rows=2)
    with pytest.raises(ValueError, match="budget"):
        model(maximum_dense_bytes=1).raw_encode(matrix)
    value = model()
    fit_scale(value)
    with pytest.raises(ValueError, match="frozen"):
        value.weighted_composition(torch.zeros(2, 3), torch.ones(2))
    value.freeze()
    with pytest.raises(ValueError, match="positive total"):
        value.weighted_composition(torch.zeros(2, 3), torch.zeros(2))
    with pytest.raises(ValueError, match="bounded"):
        value.weighted_composition(torch.zeros(2, 3), torch.ones(2), row_block=9)
