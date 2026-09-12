"""Synthetic E2 primitives/architectures only; no cohort or GPU model fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.representation.learned_state import (
    CountCompositionModel,
    hierarchical_cell_weights,
    score_matched_composition,
    smoothed_reference_delta_components,
    strict_seed_aggregate,
)
from credo_count_sde_v4.representation.state_information import (
    conditional_composition_cross_entropy,
)


def test_seed_summary_complete_order_invariant_and_explicit_counts():
    expected = [11, 22, 33]
    result = strict_seed_aggregate([33, 11, 22], [6, 2, 4], expected_seeds=expected)
    assert result == strict_seed_aggregate(expected, [2, 4, 6], expected_seeds=expected)
    assert result["expected_count"] == result["present_count"] == result["defined_count"] == 3
    assert result["all_defined"]
    assert (result["mean"], result["std"], result["min"], result["max"]) == (4, 2, 2, 6)


@pytest.mark.parametrize("undefined", [None, np.nan, np.inf, -np.inf])
def test_seed_summary_never_skips_undefined_seed(undefined):
    result = strict_seed_aggregate([11, 22, 33], [1, undefined, 3], expected_seeds=[11, 22, 33])
    assert result["expected_count"] == result["present_count"] == 3
    assert result["defined_count"] == 2
    assert result["undefined_seeds"] == [22] and result["missing_seeds"] == []
    assert not result["all_defined"]
    assert all(result[key] is None for key in ("mean", "std", "min", "max"))


def test_seed_summary_never_skips_missing_seed_and_handles_empty_or_singleton():
    result = strict_seed_aggregate([11, 33], [1, 3], expected_seeds=[11, 22, 33])
    assert result["missing_seeds"] == [22] and result["undefined_seeds"] == []
    assert result["expected_count"] == 3 and result["present_count"] == 2
    assert all(result[key] is None for key in ("mean", "std", "min", "max"))
    empty = strict_seed_aggregate([], [], expected_seeds=[1])
    assert empty["defined_count"] == 0 and empty["mean"] is None
    singleton = strict_seed_aggregate([1], [5], expected_seeds=[1])
    assert singleton["all_defined"] and singleton["mean"] == 5 and singleton["std"] is None


@pytest.mark.parametrize(
    "seeds,values,expected",
    [
        ([1, 1], [2, 2], [1]),
        ([2], [2], [1]),
        ([1], [True], [1]),
        ([True], [1], [1]),
        ([1], [1], [1, 1]),
        ([1], [], [1]),
    ],
)
def test_bad_seed_identity_or_score_is_rejected(seeds, values, expected):
    with pytest.raises(ContractError):
        strict_seed_aggregate(seeds, values, expected_seeds=expected)


def _hierarchy():
    return dict(
        donor_ids=["D1"] * 6 + ["D2"] * 2,
        checkpoint_ids=["Rest"] * 5 + ["Late", "Rest", "Rest"],
        target_ids=["X"] * 4 + ["Y", "X", "X", "X"],
        guide_ids=["g1"] * 3 + ["g2", "g3", "g1", "g1", "g1"],
    )


def test_weights_equal_donor_time_target_guide_then_cell_not_umi_or_population_size():
    data = _hierarchy()
    weights = hierarchical_cell_weights(**data)
    expected = np.asarray([1 / 48, 1 / 48, 1 / 48, 1 / 16, 1 / 8, 1 / 4, 1 / 4, 1 / 4])
    np.testing.assert_allclose(weights, expected, atol=1e-15, rtol=0)
    assert weights.sum() == 1
    order = [5, 2, 7, 1, 4, 3, 6, 0]
    permuted = {name: np.asarray(values)[order] for name, values in data.items()}
    np.testing.assert_allclose(hierarchical_cell_weights(**permuted), weights[order], atol=1e-15)


def test_controls_receive_zero_training_weight_and_separate_control_hierarchy():
    data = _hierarchy()
    original = hierarchical_cell_weights(**data)
    extras = dict(
        donor_ids=["D1"] * 5 + ["D2"],
        checkpoint_ids=["Rest"] * 6,
        target_ids=["NTC"] * 6,
        guide_ids=["c1"] * 5 + ["c2"],
    )
    augmented = {name: values + extras[name] for name, values in data.items()}
    controls = np.asarray([False] * 8 + [True] * 6)
    weights = hierarchical_cell_weights(**augmented, is_control=controls)
    np.testing.assert_allclose(weights[:8], original, atol=1e-15)
    np.testing.assert_array_equal(weights[8:], np.zeros(6))
    control_weights = hierarchical_cell_weights(**augmented, is_control=controls, role="controls")
    np.testing.assert_array_equal(control_weights[:8], np.zeros(8))
    np.testing.assert_allclose(control_weights[8:], [0.1] * 5 + [0.5])


def test_invalid_role_flags_missing_labels_and_crosswired_guides_fail():
    data = _hierarchy()
    with pytest.raises(ContractError):
        hierarchical_cell_weights(**data, is_control=np.zeros(8, dtype=int))
    with pytest.raises(ContractError, match="no cells"):
        hierarchical_cell_weights(**data, role="controls")
    with pytest.raises(ContractError, match="contradictory"):
        hierarchical_cell_weights(**(data | {"target_ids": ["Y"] + data["target_ids"][1:]}))
    with pytest.raises(ContractError, match="nonempty string"):
        hierarchical_cell_weights(**(data | {"donor_ids": [None] + data["donor_ids"][1:]}))


def test_score_matched_constant_is_mean_cell_composition_not_pooled_umi_frequency():
    counts = sparse.csr_matrix([[1000, 0, 0], [0, 1, 0]], dtype=np.int32)
    mixture = 1e-8
    probability = score_matched_composition(counts, [0.5, 0.5], uniform_mixture=mixture)
    expected = (1 - mixture) * np.asarray([0.5, 0.5, 0]) + mixture / 3
    np.testing.assert_allclose(probability, expected, atol=1e-16, rtol=0)
    assert probability.min() > 0
    pooled = np.asarray(counts.sum(axis=0)).reshape(-1) / counts.sum()
    assert np.max(np.abs(probability - pooled)) > 0.49
    # Its analytic optimum matches the uniformly mixed target objective.
    for alternative in ([0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [1 / 3] * 3):
        assert -expected @ np.log(probability) < -expected @ np.log(alternative)


def test_zero_weight_counts_and_row_permutation_cannot_change_matched_reference():
    counts = sparse.csr_matrix([[3, 1], [1, 3], [0, 0]], dtype=np.int32)
    weights = np.asarray([1, 3, 0])
    before = score_matched_composition(counts, weights)
    altered = sparse.csr_matrix([[3, 1], [1, 3], [1000000, 2]], dtype=np.int32)
    np.testing.assert_array_equal(score_matched_composition(altered, weights), before)
    order = [2, 0, 1]
    np.testing.assert_allclose(score_matched_composition(counts[order], weights[order]), before)
    with pytest.raises(ContractError, match="zero RNA depth"):
        score_matched_composition(counts, [1, 1, 1])


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1], [1, np.nan]])
def test_matched_reference_bad_weights_fail(weights):
    with pytest.raises(ContractError):
        score_matched_composition(sparse.csr_matrix([[1, 1], [2, 2]]), weights)


def test_missing_a_decomposition_matches_direct_cross_entropy_and_zero_depth():
    first = sparse.csr_matrix([[6000, 0], [2, 5], [0, 0], [4, 0]], dtype=np.int32)
    heldback = sparse.csr_matrix([[0, 3], [3, 2], [4, 1], [0, 0]], dtype=np.int32)
    prior = np.asarray([0.2, 0.8])
    result = smoothed_reference_delta_components(first, heldback, prior)
    probability = (first.toarray() + 64 * prior) / (np.asarray(first.sum(axis=1)) + 64)
    direct = conditional_composition_cross_entropy(heldback, probability)
    direct -= conditional_composition_cross_entropy(heldback, np.tile(prior, (4, 1)))
    np.testing.assert_allclose(result["total_delta"], direct, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result["decomposition_residual"][:3], 0, atol=1e-12)
    assert result["missing_a_b_fraction"][0] == 1
    assert result["missing_a_contribution"][0] == np.log1p(6000 / 64)
    assert result["detected_a_contribution"][0] == 0
    assert result["missing_a_contribution"][1] == 0
    assert result["detected_a_contribution"][1] < 0
    assert result["total_delta"][2] == 0
    assert np.isnan(result["total_delta"][3])
    np.testing.assert_array_equal(
        result["missing_a_b_count"] + result["detected_a_b_count"], [3, 5, 5, 0]
    )


def test_decomposition_uses_count_zero_not_sparse_entry_presence():
    first = sparse.csr_matrix(([4, 0], [0, 1], [0, 2]), shape=(1, 2))
    second = sparse.csr_matrix([[0, 3]])
    result = smoothed_reference_delta_components(first, second, [0.5, 0.5])
    assert result["missing_a_b_count"][0] == 3
    assert result["detected_a_b_count"][0] == 0


@pytest.mark.parametrize("family", ["M0", "M1", "M2"])
def test_architectures_exactly_start_at_analytic_reference_and_zero_depth_is_defined(family):
    base = np.asarray([0.2, 0.3, 0.5])
    model = CountCompositionModel(base, family=family)
    counts = torch.tensor([[3, 0, 1], [1, 5, 3], [0, 0, 0]], dtype=torch.int64)
    baseline = CountCompositionModel(base, family="M0")
    torch.testing.assert_close(model(counts), baseline(counts), atol=0, rtol=0)
    torch.testing.assert_close(model(counts).sum(dim=1), torch.ones(3))
    assert model.encode(counts).shape == (3, 0 if family == "M0" else 8)
    assert model.configuration()["identity_inputs"] == []
    assert model.configuration()["observation_parameters"] == []
    if family == "M0":
        assert model.optimizer_parameter_groups() == []
        assert not any(p.requires_grad for p in model.parameters())


def test_m1_is_rank_eight_linear_count_composition_and_m2_has_prescribed_width():
    model = CountCompositionModel(np.full(17, 1 / 17), family="M1").double()
    assert isinstance(model.encoder, nn.Linear) and isinstance(model.decoder, nn.Linear)
    assert model.encoder.out_features == model.decoder.in_features == 8
    assert model.encoder.bias is None and model.decoder.bias is None
    with torch.no_grad():
        model.decoder.weight.normal_(std=0.1)
    counts = torch.tensor(np.random.default_rng(11).integers(1, 30, size=(24, 17)))
    residual = (model.logits(counts) - model.logits(counts, ablate_latent=True)).detach().numpy()
    assert np.linalg.matrix_rank(residual, tol=1e-10) <= 8
    nonlinear = CountCompositionModel(np.full(17, 1 / 17), family="M2")
    assert [
        (m.in_features, m.out_features) for m in nonlinear.modules() if isinstance(m, nn.Linear)
    ] == [(17, 128), (128, 8), (8, 128), (128, 17)]
    assert sum(isinstance(m, nn.GELU) for m in nonlinear.modules()) == 2


@pytest.mark.parametrize("family", ["M1", "M2"])
def test_ablation_preserves_fitted_intercept_is_frozen_and_zero_a_falls_back(family):
    model = CountCompositionModel([0.2, 0.3, 0.5], family=family)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
        model.intercept_offset.copy_(torch.tensor([0.1, 0.7, -0.4]))
    counts = torch.tensor([[5, 1, 2], [1, 5, 2], [0, 0, 0]])
    expected = model.intercept_probabilities().detach().expand(3, -1)
    torch.testing.assert_close(model(counts, ablate_latent=True), expected, atol=0, rtol=0)
    torch.testing.assert_close(model(counts)[2], expected[2], atol=0, rtol=0)
    assert not torch.equal(model(counts)[:2], expected[:2])
    frozen = model.frozen_latent_ablation()
    assert not any(p.requires_grad for p in frozen.parameters()) and not frozen.training
    torch.testing.assert_close(frozen(counts), expected, atol=0, rtol=0)
    with torch.no_grad():
        model.intercept_offset[0] += 1
    torch.testing.assert_close(frozen(counts), expected, atol=0, rtol=0)
    assert frozen.configuration()["force_latent_ablation"]


@pytest.mark.parametrize("family", ["M1", "M2"])
def test_intercept_weight_decay_is_zero_and_every_other_parameter_is_regularized(family):
    model = CountCompositionModel([0.4, 0.6], family=family)
    groups = model.optimizer_parameter_groups(1e-4)
    allocated = {id(p): group["weight_decay"] for group in groups for p in group["params"]}
    assert len(allocated) == sum(len(group["params"]) for group in groups)
    assert allocated[id(model.intercept_offset)] == 0
    assert set(allocated) == {id(p) for p in model.parameters()}
    assert all(
        allocated[id(p)] == 1e-4
        for name, p in model.named_parameters()
        if name != "intercept_offset"
    )


@pytest.mark.parametrize(
    "counts",
    [
        torch.tensor([[1.5, 2]]),
        torch.tensor([[-1, 2]]),
        torch.tensor([[True, False]]),
        torch.tensor([[float("nan"), 1]]),
        torch.tensor([[1, 2, 3]]),
    ],
)
def test_non_count_and_misaligned_model_inputs_fail(counts):
    model = CountCompositionModel([0.4, 0.6], family="M2")
    with pytest.raises(ContractError):
        model(counts)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_exact_two_to_31_count_rejected_without_float32_scalar_rounding(dtype):
    model = CountCompositionModel([0.4, 0.6], family="M0")
    counts = torch.tensor([[2**31, 1]], dtype=dtype)
    with pytest.raises(ContractError, match="int32 value contract"):
        model(counts)


def test_representable_count_below_int32_cap_and_integer_exact_cap_are_accepted():
    model = CountCompositionModel([0.4, 0.6], family="M0")
    below = torch.nextafter(torch.tensor(float(2**31)), torch.tensor(float("-inf")))
    for counts in (
        torch.stack([below, torch.tensor(1.0)]).reshape(1, 2),
        torch.tensor([[2**31 - 1, 1]], dtype=torch.int32),
    ):
        torch.testing.assert_close(model(counts)[0], model.intercept_probabilities())


@pytest.mark.parametrize("family", ["M1", "M2"])
def test_positive_synthetic_cell_dependent_signal_can_improve_on_constant(family):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            first = torch.tensor([[40, 3, 2, 1], [1, 2, 3, 40]] * 8, dtype=torch.float32)
            second = torch.tensor([[50, 2, 3, 1], [1, 3, 2, 50]] * 8, dtype=torch.float32)
            target = second / second.sum(dim=1, keepdim=True)
            base = target.mean(dim=0).numpy().astype(np.float64)
            model = CountCompositionModel(base / base.sum(), family=family)
            initial = float(
                -(target * F.log_softmax(model.logits(first), dim=1)).sum(dim=1).mean().detach()
            )
            optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(), lr=0.01)
            for _ in range(60):
                optimizer.zero_grad()
                loss = -(target * F.log_softmax(model.logits(first), dim=1)).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
            query = torch.tensor([[30, 2, 1, 2], [2, 1, 2, 30]], dtype=torch.float32)
            truth = target[:2]
            measured = float(
                -(truth * F.log_softmax(model.logits(query), dim=1)).sum(dim=1).mean().detach()
            )
            assert measured < 0.8 * initial
            assert model(query)[0, 0] > model(query)[0, 3]
            assert model(query)[1, 3] > model(query)[1, 0]
    finally:
        torch.set_num_threads(previous_threads)


def test_constant_counts_have_no_artificial_cell_state_variation():
    model = CountCompositionModel([0.25, 0.75], family="M2")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    counts = torch.tensor([[1, 3], [10, 30], [100, 300]])
    predictions = model(counts)
    torch.testing.assert_close(predictions, predictions[0].expand(3, -1), atol=1e-7, rtol=1e-6)
