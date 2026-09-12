from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from scipy import sparse

from credo_count_sde_v4.store.g00c_refit_cuda import (
    DEFAULT_MAX_DEVICE_BYTES,
    accumulate_checkpoint_counts_cpu_compact,
    accumulate_checkpoint_counts_cuda,
    accumulate_checkpoint_counts_cuda_dev37,
    compact_sampled_rows,
    prepare_resident_checkpoint_counts_cuda,
)

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_compact_refit_is_integer_exact_and_bounded() -> None:
    rng = np.random.default_rng(44)
    dense = rng.poisson(1.5, size=(257, 4096)).astype(np.int64)
    dense[rng.random(dense.shape) < 0.82] = 0
    matrix = sparse.csr_matrix(dense)
    row_ids = np.resize(np.arange(257, dtype=np.int64), 16_384)
    weights = np.repeat(72.0, len(row_ids))
    compact = compact_sampled_rows(row_ids, weights)
    checkpoints = np.arange(257, dtype=np.int64) % 3
    reference, reference_hash = accumulate_checkpoint_counts_cpu_compact(
        matrix, checkpoints, compact, seed=9817
    )
    physical = torch.cuda.get_device_properties(0).total_memory
    ceiling = min(DEFAULT_MAX_DEVICE_BYTES, physical - 1024**3)
    observed, receipt = accumulate_checkpoint_counts_cuda(
        matrix,
        checkpoints,
        compact,
        seed=9817,
        maximum_device_bytes=ceiling,
    )
    assert np.array_equal(observed, reference)
    assert receipt.thinning_sha256 == reference_hash
    assert receipt.sampled_rows == 16_384
    assert receipt.unique_rows == 257
    assert 0 < receipt.peak_allocated_bytes <= ceiling
    assert receipt.cuda_version != "None"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_dev37_expanded_refit_preserves_pcg_variate_order() -> None:
    rng = np.random.default_rng(19)
    dense = rng.poisson(1.3, size=(1024, 256)).astype(np.int64)
    dense[rng.random(dense.shape) < 0.8] = 0
    matrix = sparse.csr_matrix(dense)
    checkpoints = np.arange(matrix.shape[0], dtype=np.int64) % 3
    weights = np.full(matrix.shape[0], 36.0)
    pcg = np.random.Generator(np.random.PCG64DXSM(7744))
    thinned = pcg.binomial(matrix.data.astype(np.int64), 0.5).astype(np.float64)
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
    reference = np.zeros((3, matrix.shape[1]), dtype=np.float64)
    for checkpoint in range(3):
        selected = checkpoints[rows] == checkpoint
        reference[checkpoint] = np.bincount(
            matrix.indices[selected],
            weights=thinned[selected] * weights[rows[selected]],
            minlength=matrix.shape[1],
        )
    physical = torch.cuda.get_device_properties(0).total_memory
    observed, receipt = accumulate_checkpoint_counts_cuda_dev37(
        matrix,
        checkpoints,
        weights,
        seed=7744,
        maximum_device_bytes=min(DEFAULT_MAX_DEVICE_BYTES, physical - 1024**3),
    )
    assert np.array_equal(observed.astype(np.float64), reference)
    assert (
        receipt.thinning_sha256
        == hashlib.sha256(np.asarray(thinned, dtype="<f8").tobytes(order="C")).hexdigest()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_resident_refit_is_restart_exact_and_mean_faithful() -> None:
    rng = np.random.default_rng(81)
    matrix = sparse.random(
        2048,
        512,
        density=0.1,
        format="csr",
        random_state=rng,
        data_rvs=lambda size: rng.integers(1, 8, size=size),
    ).astype(np.int64)
    row_ids = np.resize(np.arange(matrix.shape[0], dtype=np.int64), 65_536)
    compact = compact_sampled_rows(row_ids, np.full(len(row_ids), 18.0))
    checkpoints = np.arange(matrix.shape[0], dtype=np.int64) % 3
    physical = torch.cuda.get_device_properties(0).total_memory
    plan = prepare_resident_checkpoint_counts_cuda(
        matrix,
        checkpoints,
        compact,
        maximum_device_bytes=min(DEFAULT_MAX_DEVICE_BYTES, physical - 1024**3),
    )
    first = plan.accumulate(seed=901)
    resumed = plan.accumulate(seed=901)
    assert np.array_equal(first, resumed)
    totals = np.asarray([plan.accumulate(seed=seed).sum() for seed in range(32)])
    expected = float(
        np.sum(
            matrix.data.astype(np.int64)
            * np.repeat(compact.multiplicities, np.diff(matrix.indptr))
            * np.repeat(compact.inverse_probability_weights, np.diff(matrix.indptr))
        )
        * 0.5
    )
    assert abs(float(totals.mean()) - expected) / expected < 0.01
