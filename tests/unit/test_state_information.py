"""E2 count/readout primitives: positive, null, adversarial and ablation tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.representation.state_information import (
    compare_readout_distributions,
    conditional_composition_cross_entropy,
    count_expression_readouts,
    fixed_bin_occupancies,
    thin_counts_by_cell,
    validate_count_csr,
    weighted_error_decomposition,
)


def _counts():
    return sparse.csr_matrix(
        np.asarray([[20, 0, 40], [0, 0, 0], [15, 25, 0], [3, 8, 12]], dtype=np.int32)
    )


def test_count_validation_preserves_integer_data_zero_depth_and_identity():
    counts = _counts()
    original = counts.copy()
    validated = validate_count_csr(
        counts, cell_ids=["A", "B", "C", "D"], feature_ids=["x", "y", "z"]
    )
    np.testing.assert_array_equal(validated.toarray(), original.toarray())
    np.testing.assert_array_equal(counts.data, original.data)
    np.testing.assert_array_equal(counts.indptr, original.indptr)
    assert validated.dtype == np.int32


@pytest.mark.parametrize(
    "bad",
    [
        sparse.csr_matrix([[True, False]]),
        sparse.csr_matrix([[1.0, 0.0]]),
        sparse.csr_matrix([[1.5, 0.0]]),
        sparse.csr_matrix([[-1, 0]]),
        sparse.csr_matrix([[np.iinfo(np.int32).max + 1, 0]], dtype=np.int64),
        sparse.csc_matrix([[1, 0]]),
        np.asarray([[1, 0]]),
    ],
)
def test_non_csr_noninteger_boolean_negative_and_overflow_counts_rejected(bad):
    with pytest.raises(ContractError):
        validate_count_csr(bad)


@pytest.mark.parametrize("indices", [[0, 0], [1, 0]])
def test_duplicate_or_unsorted_columns_are_not_silently_repaired(indices):
    counts = sparse.csr_matrix(([2, 3], indices, [0, 2]), shape=(1, 2), dtype=np.int32)
    with pytest.raises(ContractError, match="sorted and duplicate-free"):
        validate_count_csr(counts)


@pytest.mark.parametrize(
    "ids", [["A", "A", "C", "D"], ["A", 1, "C", "D"], ["A"], ["A", " ", "C", "D"]]
)
def test_duplicate_mixed_missing_or_empty_cell_identities_rejected(ids):
    with pytest.raises(ContractError, match="cell_ids"):
        validate_count_csr(_counts(), cell_ids=ids)


def test_misaligned_and_mixed_feature_identities_rejected():
    for ids in (["x", "y"], ["x", "x", "z"], ["x", 1, "z"]):
        with pytest.raises(ContractError, match="feature_ids"):
            validate_count_csr(_counts(), feature_ids=ids)


@pytest.mark.parametrize(
    "ids", [{"A", "B", "C", "D"}, dict.fromkeys("ABCD"), frozenset("ABCD"), iter("ABCD")]
)
def test_unordered_or_single_pass_identity_containers_rejected(ids):
    with pytest.raises(ContractError, match="ordered sequence"):
        validate_count_csr(_counts(), cell_ids=ids)


@pytest.mark.parametrize("name", ["indices", "indptr"])
@pytest.mark.parametrize("dtype", [np.float64, np.bool_])
def test_malformed_noninteger_csr_index_arrays_rejected(name, dtype):
    counts = _counts()
    setattr(counts, name, getattr(counts, name).astype(dtype))
    with pytest.raises(ContractError, match="integer dtypes"):
        validate_count_csr(counts)


def test_cell_thinning_conserves_exact_counts_and_ignores_batch_and_row_order():
    counts = _counts()
    original = counts.copy()
    cells = np.asarray(["A", "B", "C", "D"])
    a, b = thin_counts_by_cell(counts, cells, seed=98)
    # Golden stream for the declared v1 thinning scheme; algorithm changes
    # require an explicit new scheme rather than silent receipt reuse.
    np.testing.assert_array_equal(a.toarray(), [[10, 0, 18], [0, 0, 0], [5, 12, 0], [2, 4, 6]])
    np.testing.assert_array_equal((a + b).toarray(), counts.toarray())
    assert a.dtype == b.dtype == np.int32
    assert (a.data >= 0).all() and (b.data >= 0).all()
    parts = [
        thin_counts_by_cell(counts[i:j], cells[i:j], seed=98) for i, j in ((0, 1), (1, 3), (3, 4))
    ]
    np.testing.assert_array_equal(sparse.vstack([part[0] for part in parts]).toarray(), a.toarray())
    np.testing.assert_array_equal(sparse.vstack([part[1] for part in parts]).toarray(), b.toarray())
    order = np.asarray([3, 1, 0, 2])
    perm_a, perm_b = thin_counts_by_cell(counts[order], cells[order], seed=98)
    np.testing.assert_array_equal(perm_a.toarray(), a[order].toarray())
    np.testing.assert_array_equal(perm_b.toarray(), b[order].toarray())
    np.testing.assert_array_equal(counts.toarray(), original.toarray())
    assert a.getrow(1).nnz == b.getrow(1).nnz == 0


def test_thinning_stream_changes_with_seed_and_cell_identity_not_explicit_zeros():
    counts = sparse.csr_matrix([[10000, 0, 10000]], dtype=np.int32)
    a, _ = thin_counts_by_cell(counts, ["A"], seed=7)
    other_seed, _ = thin_counts_by_cell(counts, ["A"], seed=8)
    other_cell, _ = thin_counts_by_cell(counts, ["B"], seed=7)
    assert not np.array_equal(a.toarray(), other_seed.toarray())
    assert not np.array_equal(a.toarray(), other_cell.toarray())
    explicit = sparse.csr_matrix(([10000, 0, 10000], [0, 1, 2], [0, 3]), shape=(1, 3))
    explicit_a, _ = thin_counts_by_cell(explicit, ["A"], seed=7)
    np.testing.assert_array_equal(explicit_a.toarray(), a.toarray())


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5])
def test_thinning_requires_exact_integer_seed(seed):
    with pytest.raises(ContractError, match="seed"):
        thin_counts_by_cell(_counts(), ["A", "B", "C", "D"], seed=seed)


@pytest.mark.parametrize("probability", [0, 1, -0.2, np.nan, True, "half"])
def test_thinning_probability_is_strict(probability):
    with pytest.raises(ContractError, match="probability"):
        thin_counts_by_cell(_counts(), ["A", "B", "C", "D"], seed=1, probability=probability)


def test_conditional_cross_entropy_matches_manual_score_and_reports_zero_depth():
    counts = sparse.csr_matrix([[1, 3], [0, 0], [2, 6]], dtype=np.int32)
    probabilities = np.tile([0.25, 0.75], (3, 1))
    result = conditional_composition_cross_entropy(counts, probabilities)
    expected = -(np.log(0.25) + 3 * np.log(0.75)) / 4
    np.testing.assert_allclose(result[[0, 2]], expected)
    assert np.isnan(result[1])
    # The score conditions on depth: doubling held-back counts is not new exposure input.
    assert result[0] == result[2]


def test_zero_probability_is_infinite_only_when_that_gene_is_observed():
    counts = sparse.csr_matrix([[0, 4], [1, 3]], dtype=np.int32)
    result = conditional_composition_cross_entropy(counts, np.asarray([[0, 1], [0, 1]]))
    assert result[0] == 0
    assert np.isposinf(result[1])


@pytest.mark.parametrize("probabilities", [[[0.4, 0.4]], [[-0.1, 1.1]], [[np.nan, 0]], [[1]]])
def test_unaligned_nonfinite_or_unnormalized_probabilities_rejected(probabilities):
    with pytest.raises(ContractError):
        conditional_composition_cross_entropy(sparse.csr_matrix([[1, 2]]), probabilities)


def test_selected_gene_readouts_preserve_full_rna_denominator_and_zero_coverage():
    counts = sparse.csr_matrix([[2, 3], [0, 0], [0, 0]], dtype=np.int32)
    totals = np.asarray([10, 0, 100])
    fractions = count_expression_readouts(counts, scale=1, library_sizes=totals)
    np.testing.assert_array_equal(fractions[0], [0.2, 0.3])
    assert np.isnan(fractions[1]).all()
    np.testing.assert_array_equal(fractions[2], [0, 0])
    log_cpm = count_expression_readouts(counts, log1p=True, library_sizes=totals)
    np.testing.assert_allclose(log_cpm[0], np.log1p([200000, 300000]))
    full = count_expression_readouts(counts[:1], scale=1)
    np.testing.assert_array_equal(full, [[0.4, 0.6]])


@pytest.mark.parametrize("totals", [[4], [-1], [np.nan], [np.inf], [[10]], [10, 11]])
def test_invalid_library_denominator_rejected(totals):
    with pytest.raises(ContractError):
        count_expression_readouts(sparse.csr_matrix([[2, 3]]), library_sizes=totals)


def test_error_decomposition_uses_normalized_weights_and_prediction_minus_truth():
    result = weighted_error_decomposition(
        np.zeros(3), np.asarray([1, 3, 100]), weights=np.asarray([1, 3, 0])
    )
    assert result["signed_bias"] == 2.5
    assert result["weighted_mse"] == 7
    assert result["squared_bias"] == 6.25
    assert result["centered_error_variance"] == 0.75
    assert result["decomposition_residual"] == 0
    rescaled = weighted_error_decomposition(
        np.zeros(3), np.asarray([1, 3, 100]), weights=np.asarray([1e300, 3e300, 0])
    )
    assert result == rescaled


def test_null_and_common_offset_error_are_not_target_specific_variation():
    exact = weighted_error_decomposition(np.arange(3), np.arange(3))
    assert all(value == 0 for value in exact.values())
    shifted = weighted_error_decomposition(np.arange(3), np.arange(3) - 2)
    assert shifted["signed_bias"] == -2
    assert shifted["squared_bias"] == shifted["weighted_mse"] == 4
    assert shifted["centered_error_variance"] == 0


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1], [1, np.nan]])
def test_invalid_weights_fail_instead_of_silently_changing_population(weights):
    with pytest.raises(ContractError):
        weighted_error_decomposition(np.zeros(2), np.ones(2), weights=weights)


def test_fixed_bin_boundary_convention_and_tail_coverage():
    values = np.asarray([-100, 0, 1, 100])
    occupancy = fixed_bin_occupancies(values, [-np.inf, 0, 1, np.inf], weights=[1, 2, 3, 4])
    np.testing.assert_allclose(occupancy, [[0.1, 0.2, 0.7]])
    np.testing.assert_allclose(occupancy.sum(axis=1), 1)


def test_readout_specific_fixed_bins_are_not_fitted_to_values():
    values = np.asarray([[-1, 10], [0, 20], [1, 30]])
    edges = np.asarray([[-np.inf, 0, np.inf], [-np.inf, 25, np.inf]])
    np.testing.assert_allclose(
        fixed_bin_occupancies(values, edges), [[1 / 3, 2 / 3], [2 / 3, 1 / 3]]
    )


@pytest.mark.parametrize(
    "edges",
    [[0, 1, 2], [-np.inf, 1, 0, np.inf], [-np.inf, 0, 0, np.inf], [-np.inf, np.nan, np.inf]],
)
def test_invalid_or_nonexhaustive_bins_fail(edges):
    with pytest.raises(ContractError, match="bin edges"):
        fixed_bin_occupancies([0, 1], edges)


def test_distribution_comparison_detects_same_mean_different_mixtures():
    observed = np.asarray([-2, -2, 2, 2])
    predicted = np.asarray([-1, -1, 1, 1])
    result = compare_readout_distributions(observed, predicted, [-np.inf, -1.5, 0, 1.5, np.inf])
    np.testing.assert_array_equal(result["mean_error"], [0])
    np.testing.assert_array_equal(result["observed_variance"], [4])
    np.testing.assert_array_equal(result["predicted_variance"], [1])
    np.testing.assert_array_equal(result["occupancy_total_variation"], [1])


def test_distribution_null_and_centroid_collapse_ablation():
    observed = np.asarray([-2, -1, 1, 2])
    edges = [-np.inf, -0.5, 0.5, np.inf]
    null = compare_readout_distributions(observed, observed[::-1], edges)
    for key in ("mean_error", "variance_error", "occupancy_total_variation"):
        np.testing.assert_array_equal(null[key], [0])
    collapsed = compare_readout_distributions(observed, [0, 0], edges)
    np.testing.assert_array_equal(collapsed["mean_error"], [0])
    np.testing.assert_array_equal(collapsed["variance_error"], [-2.5])
    np.testing.assert_array_equal(collapsed["occupancy_total_variation"], [1])


def test_unequal_distribution_sample_counts_use_separate_weights():
    result = compare_readout_distributions(
        [-1, 1],
        [-1, 1, 1],
        [-np.inf, 0, np.inf],
        observed_weights=[1, 1],
        predicted_weights=[2, 1, 1],
    )
    for key in ("mean_error", "variance_error", "occupancy_total_variation"):
        np.testing.assert_array_equal(result[key], [0])


def test_undefined_readouts_and_overflow_fail_without_dropping_rows():
    with pytest.raises(ContractError, match="undefined rows"):
        fixed_bin_occupancies([0, np.nan], [-np.inf, 0, np.inf])
    with pytest.raises(ContractError, match="overflowed"):
        weighted_error_decomposition([-1e308], [1e308])
    with pytest.raises(ContractError, match="overflowed"):
        compare_readout_distributions([-1e308], [1e308], [-np.inf, 0, np.inf])
