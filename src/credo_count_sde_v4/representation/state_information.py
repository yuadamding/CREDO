"""Pure count/readout measurements for E2; no representation fitting or forecasting.

The caller owns donor partitions, fixed readout definitions, and artifact hashes.
These functions never learn normalization, bins, or thresholds from evaluation
rows. Thinning is a technical diagnostic, not an independence guarantee under
overdispersed counts. Readout variability includes measurement noise and is not
an estimate of biological diffusion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from ..errors import ContractError

THINNING_SCHEME = "credo.e2.cell_binomial.pcg64dxsm.v1"


def _identities(values: Sequence[str], size: int, name: str) -> tuple[str, ...]:
    if not isinstance(values, (Sequence, np.ndarray)) or isinstance(values, (str, bytes)):
        raise ContractError(f"{name} must be an ordered sequence of string identities.")
    result = tuple(values)
    if (
        len(result) != size
        or any(not isinstance(value, str) or not value.strip() for value in result)
        or len(set(result)) != size
    ):
        raise ContractError(f"{name} must be aligned, unique, nonempty string identities.")
    return result


def validate_count_csr(
    counts: sparse.csr_matrix,
    *,
    cell_ids: Sequence[str] | None = None,
    feature_ids: Sequence[str] | None = None,
) -> sparse.csr_matrix:
    """Validate integer, canonical CSR without changing counts or feature order.

    Each stored count must fit the existing CountStore int32 value contract.
    Explicit zeros and zero-depth rows are allowed. Duplicate/unsorted columns
    are rejected rather than silently merged or reordered. The returned matrix
    may share input storage; callers must not treat validation as an immutable
    copy or as a persisted source identity.
    """
    if not sparse.issparse(counts) or counts.format != "csr" or counts.ndim != 2:
        raise ContractError("Counts must be a two-dimensional CSR matrix.")
    if not np.issubdtype(counts.indptr.dtype, np.integer) or not np.issubdtype(
        counts.indices.dtype, np.integer
    ):
        raise ContractError("CSR pointers and column indices require integer dtypes.")
    matrix = sparse.csr_matrix(counts, copy=False)
    rows, genes = matrix.shape
    if genes == 0 or not np.issubdtype(matrix.dtype, np.integer):
        raise ContractError("Counts require nonempty features and an integer, non-boolean dtype.")
    if (
        matrix.indptr.ndim != 1
        or matrix.indices.ndim != 1
        or matrix.data.ndim != 1
        or len(matrix.indptr) != rows + 1
        or matrix.indptr[0] != 0
        or np.any(matrix.indptr[1:] < matrix.indptr[:-1])
        or matrix.indptr[-1] != len(matrix.data)
        or len(matrix.indices) != len(matrix.data)
        or np.any(matrix.indices < 0)
        or np.any(matrix.indices >= genes)
    ):
        raise ContractError("CSR pointers or column indices are invalid.")
    if np.any(matrix.data < 0) or np.any(matrix.data > np.iinfo(np.int32).max):
        raise ContractError("Counts must be nonnegative integers within the int32 value contract.")
    if len(matrix.indices) > 1:
        adjacent = np.ones(len(matrix.indices) - 1, dtype=bool)
        boundaries = matrix.indptr[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(matrix.indices))]
        adjacent[boundaries - 1] = False
        if np.any((matrix.indices[1:] <= matrix.indices[:-1]) & adjacent):
            raise ContractError("CSR columns must be sorted and duplicate-free within each row.")
    if cell_ids is not None:
        _identities(cell_ids, rows, "cell_ids")
    if feature_ids is not None:
        _identities(feature_ids, genes, "feature_ids")
    return matrix


def thin_counts_by_cell(
    counts: sparse.csr_matrix,
    cell_ids: Sequence[str],
    *,
    seed: int,
    probability: float = 0.5,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Return integer A, B with A+B=X, stable under row order and batching.

    Each cell has its own SHA-256-derived PCG64DXSM stream. Within a cell,
    positive counts are drawn in canonical feature order. The caller must bind
    that feature order and this scheme identifier in its run protocol. Column
    permutation is not part of this invariance guarantee. Explicit CSR zeros
    consume no random variates. No input matrix or global RNG is modified.
    """
    matrix = validate_count_csr(counts, cell_ids=cell_ids)
    identities = _identities(cell_ids, matrix.shape[0], "cell_ids")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**64
    ):
        raise ContractError("Thinning seed must be an unsigned 64-bit integer.")
    if (
        not isinstance(probability, (int, float, np.integer, np.floating))
        or isinstance(probability, (bool, np.bool_))
        or not np.isfinite(probability)
        or not 0 < probability < 1
    ):
        raise ContractError(
            "Thinning probability must be finite and strictly between zero and one."
        )
    first = np.zeros(len(matrix.data), dtype=np.int32)
    for row, identity in enumerate(identities):
        left, right = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        payload = json.dumps(
            [THINNING_SCHEME, int(seed), identity], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        cell_seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")
        rng = np.random.Generator(np.random.PCG64DXSM(cell_seed))
        values = matrix.data[left:right].astype(np.int64, copy=False)
        positive = values > 0
        first[left:right][positive] = rng.binomial(values[positive], probability).astype(np.int32)
    second = matrix.data.astype(np.int32) - first
    result = []
    for data in (first, second):
        view = sparse.csr_matrix(
            (data, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
        )
        view.eliminate_zeros()
        result.append(view)
    return result[0], result[1]


def _real_array(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if not (np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)):
        raise ContractError(f"{name} must contain real, non-boolean numbers.")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError(
            f"{name} must be finite; undefined rows need explicit coverage handling."
        )
    return result


def conditional_composition_cross_entropy(
    heldback_counts: sparse.csr_matrix, probabilities: np.ndarray
) -> np.ndarray:
    """Per-row held-back cross-entropy, in natural-log units per observed UMI.

    The held-back total only normalizes the score; it is not an encoder input.
    This is multinomial cross-entropy, not full NB or multinomial log likelihood.
    Zero held-back totals return NaN; positive counts at zero probability return
    positive infinity. Neither kind of row is silently discarded.
    """
    matrix = validate_count_csr(heldback_counts)
    probs = _real_array(probabilities, "probabilities")
    if (
        probs.shape != matrix.shape
        or np.any(probs < 0)
        or np.any(probs > 1)
        or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-10, rtol=0)
    ):
        raise ContractError("Probabilities must align with counts and sum to one in every row.")
    result = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    for row in range(matrix.shape[0]):
        left, right = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        values = matrix.data[left:right].astype(np.float64)
        positive = values > 0
        total = values.sum()
        if total > 0:
            with np.errstate(divide="ignore"):
                log_probs = np.log(probs[row, matrix.indices[left:right][positive]])
            result[row] = -float(np.dot(values[positive], log_probs)) / total
    return result


def count_expression_readouts(
    counts: sparse.csr_matrix,
    *,
    scale: float = 1e6,
    log1p: bool = False,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Return fixed-scale fractions or log1p fractions, without learned scaling.

    With selected readout genes, ``library_sizes`` must supply the original full
    RNA totals. It cannot be smaller than the selected count sum. A zero total
    gives an all-NaN row, including if the selected matrix has no expressed gene.
    ``scale=1`` gives fractions; defaults give CPM. This function densifies only
    the supplied matrix, so callers should pass readout columns or minibatches.
    """
    matrix = validate_count_csr(counts)
    if (
        not isinstance(scale, (int, float, np.integer, np.floating))
        or isinstance(scale, (bool, np.bool_))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ContractError("Readout scale must be finite and positive.")
    if not isinstance(log1p, (bool, np.bool_)):
        raise ContractError("log1p must be boolean.")
    totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    library = totals if library_sizes is None else _real_array(library_sizes, "library_sizes")
    if library.shape != totals.shape or np.any(library < totals) or np.any(library < 0):
        raise ContractError("Library sizes must align and cover the complete selected count sum.")
    normalized = np.full(matrix.shape, np.nan, dtype=np.float64)
    np.divide(
        matrix.toarray().astype(np.float64),
        library[:, None],
        out=normalized,
        where=library[:, None] > 0,
    )
    normalized *= float(scale)
    return np.log1p(normalized) if log1p else normalized


def _weights(weights: np.ndarray | None, rows: int) -> np.ndarray:
    if rows == 0:
        raise ContractError("A measurement requires at least one row.")
    values = np.ones(rows) if weights is None else _real_array(weights, "weights")
    if values.shape != (rows,) or np.any(values < 0) or not np.any(values > 0):
        raise ContractError("Weights must align, be nonnegative, and have positive total weight.")
    values = values / values.max()
    return values / values.sum()


def weighted_error_decomposition(
    observed: np.ndarray, predicted: np.ndarray, *, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Decompose weighted MSE into squared signed bias plus centered error variance.

    Bias is predicted minus observed. Removing this observed bias is not a
    forecast correction and this function returns no corrected predictions.
    """
    actual = _real_array(observed, "observed")
    estimate = _real_array(predicted, "predicted")
    if actual.ndim != 1 or estimate.shape != actual.shape:
        raise ContractError("Error decomposition requires aligned one-dimensional readouts.")
    weight = _weights(weights, len(actual))
    with np.errstate(over="ignore", invalid="ignore"):
        error = estimate - actual
        bias = float(np.dot(weight, error))
        squared_bias = bias * bias
        mse = float(np.dot(weight, error * error))
        centered = float(np.dot(weight, (error - bias) ** 2))
    if not np.isfinite([bias, squared_bias, mse, centered]).all():
        raise ContractError("Error decomposition overflowed finite float64 arithmetic.")
    return {
        "weighted_mse": mse,
        "signed_bias": bias,
        "squared_bias": squared_bias,
        "centered_error_variance": centered,
        "decomposition_residual": mse - squared_bias - centered,
    }


def _readout_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = _real_array(values, name)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ContractError(f"{name} requires nonempty rows and readout columns.")
    return array


def fixed_bin_occupancies(
    values: np.ndarray, bin_edges: np.ndarray, *, weights: np.ndarray | None = None
) -> np.ndarray:
    """Return readout-by-bin occupancy using explicit, never-fitted boundaries.

    Edges may be shared (one vector) or readout-specific (one row per readout),
    with equal bin count. They must start at -inf and end at +inf, so every
    finite observation is counted. Internal-edge equality enters the right bin.
    NaNs are rejected rather than silently altering population denominators.
    """
    array = _readout_matrix(values, "values")
    raw_edges = np.asarray(bin_edges)
    if not (
        np.issubdtype(raw_edges.dtype, np.floating) or np.issubdtype(raw_edges.dtype, np.integer)
    ):
        raise ContractError("Bin edges require real numeric boundaries.")
    edges = np.asarray(raw_edges, dtype=np.float64)
    if edges.ndim == 1:
        edges = np.broadcast_to(edges, (array.shape[1], len(edges)))
    if (
        edges.ndim != 2
        or edges.shape[0] != array.shape[1]
        or edges.shape[1] < 2
        or not np.isneginf(edges[:, 0]).all()
        or not np.isposinf(edges[:, -1]).all()
        or not np.isfinite(edges[:, 1:-1]).all()
        or not np.all(edges[:, 1:] > edges[:, :-1])
    ):
        raise ContractError("Fixed bin edges must be ordered, aligned, and cover -inf to +inf.")
    weight = _weights(weights, len(array))
    result = np.empty((array.shape[1], edges.shape[1] - 1), dtype=np.float64)
    for column in range(array.shape[1]):
        labels = np.searchsorted(edges[column, 1:-1], array[:, column], side="right")
        result[column] = np.bincount(labels, weights=weight, minlength=result.shape[1])
    return result


def compare_readout_distributions(
    observed: np.ndarray,
    predicted: np.ndarray,
    bin_edges: np.ndarray,
    *,
    observed_weights: np.ndarray | None = None,
    predicted_weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compare marginal means, population variances, and fixed-bin occupancies.

    Samples may have unequal row counts but must share identical readout axes.
    These marginal checks can detect centroid-preserving collapse, but do not
    identify every joint-distribution discrepancy. No latent-space distances,
    biological-program labels, or noise corrections are inferred here.
    """
    actual = _readout_matrix(observed, "observed")
    estimate = _readout_matrix(predicted, "predicted")
    if actual.shape[1] != estimate.shape[1]:
        raise ContractError("Distribution samples must share the same readout columns.")
    actual_w = _weights(observed_weights, len(actual))
    estimate_w = _weights(predicted_weights, len(estimate))
    with np.errstate(over="ignore", invalid="ignore"):
        actual_mean = actual_w @ actual
        estimate_mean = estimate_w @ estimate
        actual_var = actual_w @ ((actual - actual_mean) ** 2)
        estimate_var = estimate_w @ ((estimate - estimate_mean) ** 2)
        mean_error = estimate_mean - actual_mean
        variance_error = estimate_var - actual_var
    if not all(
        np.isfinite(value).all()
        for value in (
            actual_mean,
            estimate_mean,
            actual_var,
            estimate_var,
            mean_error,
            variance_error,
        )
    ):
        raise ContractError("Readout moments overflowed finite float64 arithmetic.")
    actual_bins = fixed_bin_occupancies(actual, bin_edges, weights=actual_w)
    estimate_bins = fixed_bin_occupancies(estimate, bin_edges, weights=estimate_w)
    return {
        "observed_mean": actual_mean,
        "predicted_mean": estimate_mean,
        "mean_error": mean_error,
        "observed_variance": actual_var,
        "predicted_variance": estimate_var,
        "variance_error": variance_error,
        "observed_occupancy": actual_bins,
        "predicted_occupancy": estimate_bins,
        "occupancy_total_variation": 0.5 * np.abs(estimate_bins - actual_bins).sum(axis=1),
    }
