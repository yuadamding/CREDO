"""CUDA-backed integer accumulation for G00C refit statistics.

The expanded-row path preserves Dev37 PCG64DXSM variate order. Compact paths
collapse duplicate draws before thinning; they preserve the binomial law but
change RNG consumption and therefore require a new scientific authority.
Integer accumulation makes every path independent of CUDA atomic order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import sparse

from ..errors import IntegrityError

DEFAULT_MAX_DEVICE_BYTES = 30 * 1024**3


@dataclass(frozen=True)
class CompactSampledRows:
    """Stable row multiplicities replacing duplicate source reads."""

    row_ids: np.ndarray[Any, Any]
    multiplicities: np.ndarray[Any, Any]
    inverse_probability_weights: np.ndarray[Any, Any]


@dataclass(frozen=True)
class CudaAccumulationReceipt:
    """Non-secret device and resource evidence for one reduction."""

    device_name: str
    torch_version: str
    cuda_version: str
    physical_device_bytes: int
    maximum_device_bytes: int
    peak_allocated_bytes: int
    sampled_rows: int
    unique_rows: int
    matrix_nonzeros: int
    thinning_sha256: str


@dataclass(frozen=True)
class Dev37CudaAccumulationReceipt:
    """Device evidence for a variate-order-preserving Dev37 reduction."""

    device_name: str
    torch_version: str
    cuda_version: str
    physical_device_bytes: int
    maximum_device_bytes: int
    peak_allocated_bytes: int
    expanded_rows: int
    matrix_nonzeros: int
    thinning_sha256: str


@dataclass
class CudaResidentCompactReduction:
    """GPU-resident experimental reduction using CUDA-native binomial draws.

    This path is fast across many refit seeds, but its Philox stream is not the
    frozen PCG64DXSM stream.  It therefore requires a new scientific authority
    and must not be substituted into a Dev37 execution.
    """

    device: torch.device
    flat_indices: torch.Tensor
    effective_counts: torch.Tensor
    probabilities: torch.Tensor
    weights: torch.Tensor
    output: torch.Tensor
    maximum_device_bytes: int

    def accumulate(self, *, seed: int) -> np.ndarray[Any, Any]:
        """Run one restart-exact CUDA-native thinning and integer reduction."""

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        thinned = torch.binomial(
            self.effective_counts,
            self.probabilities,
            generator=generator,
        ).to(dtype=torch.int64)
        self.output.zero_()
        self.output.index_add_(0, self.flat_indices, thinned * self.weights)
        torch.cuda.synchronize(self.device)
        if torch.cuda.max_memory_allocated(self.device) > self.maximum_device_bytes:
            raise RuntimeError("G00C resident CUDA reduction exceeded its allocator ceiling.")
        return self.output.reshape(3, -1).cpu().numpy().copy()


def compact_sampled_rows(
    row_ids: np.ndarray[Any, Any],
    inverse_probability_weights: np.ndarray[Any, Any],
) -> CompactSampledRows:
    """Collapse repeated draws while retaining exact integer draw mass."""

    rows = np.asarray(row_ids, dtype=np.int64)
    weights = np.asarray(inverse_probability_weights, dtype=np.float64)
    if rows.ndim != 1 or weights.shape != rows.shape or len(rows) == 0:
        raise IntegrityError("CUDA refit draws must be nonempty aligned vectors.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise IntegrityError("CUDA refit inverse-probability weights must be finite and positive.")
    integer_weights = np.rint(weights)
    if not np.array_equal(weights, integer_weights):
        raise IntegrityError("CUDA integer reduction requires exact integral sampling weights.")

    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    sorted_weights = integer_weights[order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(sorted_rows[1:] != sorted_rows[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_rows)]
    for start, end in zip(starts, ends, strict=True):
        if np.any(sorted_weights[start:end] != sorted_weights[start]):
            raise IntegrityError("One sampled row has inconsistent hierarchical weights.")
    return CompactSampledRows(
        row_ids=sorted_rows[starts],
        multiplicities=(ends - starts).astype(np.int64),
        inverse_probability_weights=sorted_weights[starts],
    )


def _validate_inputs(
    matrix: sparse.csr_matrix,
    checkpoint_codes: np.ndarray[Any, Any],
    compact: CompactSampledRows,
) -> tuple[sparse.csr_matrix, np.ndarray[Any, Any]]:
    canonical = matrix.copy().tocsr()
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.eliminate_zeros()
    checkpoints = np.asarray(checkpoint_codes, dtype=np.int64)
    if (
        canonical.shape[0] != len(compact.row_ids)
        or checkpoints.shape != compact.row_ids.shape
        or canonical.shape[1] <= 0
        or np.any((checkpoints < 0) | (checkpoints > 2))
        or np.any(canonical.data < 0)
        or not np.equal(canonical.data, np.rint(canonical.data)).all()
    ):
        raise IntegrityError("CUDA refit matrix or checkpoint codes violate the count contract.")
    return canonical, checkpoints


def _thinned_blocks(
    matrix: sparse.csr_matrix,
    compact: CompactSampledRows,
    *,
    seed: int,
    row_block_size: int,
) -> Iterator[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], str]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    digest = hashlib.sha256()
    for row_start in range(0, matrix.shape[0], row_block_size):
        row_end = min(row_start + row_block_size, matrix.shape[0])
        data_start = int(matrix.indptr[row_start])
        data_end = int(matrix.indptr[row_end])
        row_counts = np.diff(matrix.indptr[row_start : row_end + 1])
        local_rows = np.repeat(np.arange(row_start, row_end, dtype=np.int64), row_counts)
        base_counts = matrix.data[data_start:data_end].astype(np.int64, copy=False)
        effective_counts = base_counts * compact.multiplicities[local_rows]
        if np.any(effective_counts < 0):
            raise IntegrityError("CUDA refit effective counts overflow int64.")
        thinned = rng.binomial(effective_counts, 0.5).astype(np.int64)
        digest.update(np.asarray(thinned, dtype="<i8").tobytes(order="C"))
        yield (
            local_rows,
            matrix.indices[data_start:data_end].astype(np.int64, copy=False),
            thinned,
            digest.hexdigest(),
        )


def accumulate_checkpoint_counts_cpu_compact(
    matrix: sparse.csr_matrix,
    checkpoint_codes: np.ndarray[Any, Any],
    compact: CompactSampledRows,
    *,
    seed: int,
    row_block_size: int = 4096,
) -> tuple[np.ndarray[Any, Any], str]:
    """Reference implementation for CUDA equivalence tests."""

    canonical, checkpoints = _validate_inputs(matrix, checkpoint_codes, compact)
    answer = np.zeros((3, canonical.shape[1]), dtype=np.int64)
    thinning_sha256 = hashlib.sha256().hexdigest()
    for local_rows, features, thinned, digest_sha256 in _thinned_blocks(
        canonical, compact, seed=seed, row_block_size=row_block_size
    ):
        contributions = thinned * compact.inverse_probability_weights[local_rows]
        flat = checkpoints[local_rows] * canonical.shape[1] + features
        np.add.at(answer.reshape(-1), flat, contributions)
        thinning_sha256 = digest_sha256
    return answer, thinning_sha256


def accumulate_checkpoint_counts_cuda(
    matrix: sparse.csr_matrix,
    checkpoint_codes: np.ndarray[Any, Any],
    compact: CompactSampledRows,
    *,
    seed: int,
    device: str | torch.device = "cuda",
    maximum_device_bytes: int = DEFAULT_MAX_DEVICE_BYTES,
    row_block_size: int = 4096,
) -> tuple[np.ndarray[Any, Any], CudaAccumulationReceipt]:
    """Reduce compact thinned counts on CUDA under a new RNG-stream contract."""

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G00C CUDA refit requires an available CUDA device.")
    if maximum_device_bytes <= 0:
        raise ValueError("maximum_device_bytes must be positive.")
    canonical, checkpoints = _validate_inputs(matrix, checkpoint_codes, compact)
    properties = torch.cuda.get_device_properties(selected)
    if maximum_device_bytes >= properties.total_memory:
        raise RuntimeError("G00C CUDA ceiling must retain physical-device headroom.")

    torch.cuda.reset_peak_memory_stats(selected)
    output = torch.zeros(3 * canonical.shape[1], dtype=torch.int64, device=selected)
    thinning_sha256 = hashlib.sha256().hexdigest()
    for local_rows, features, thinned, digest_sha256 in _thinned_blocks(
        canonical, compact, seed=seed, row_block_size=row_block_size
    ):
        contributions = thinned * compact.inverse_probability_weights[local_rows]
        if np.any(contributions < 0):
            raise IntegrityError("CUDA refit weighted counts overflow int64.")
        flat = checkpoints[local_rows] * canonical.shape[1] + features
        # Three int64 vectors are live on device for each block.  Bound the
        # transfer even when a source row is much denser than expected.
        chunk_items = max(1, (maximum_device_bytes - output.numel() * 8) // 24)
        for start in range(0, len(flat), chunk_items):
            end = min(start + chunk_items, len(flat))
            index = torch.as_tensor(flat[start:end], dtype=torch.int64, device=selected)
            values = torch.as_tensor(contributions[start:end], dtype=torch.int64, device=selected)
            output.index_add_(0, index, values)
            del index, values
        if torch.cuda.max_memory_allocated(selected) > maximum_device_bytes:
            raise RuntimeError("G00C CUDA accumulation exceeded its allocator ceiling.")
        thinning_sha256 = digest_sha256
    torch.cuda.synchronize(selected)
    counts = output.reshape(3, canonical.shape[1]).cpu().numpy()
    peak = int(torch.cuda.max_memory_allocated(selected))
    receipt = CudaAccumulationReceipt(
        device_name=properties.name,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        physical_device_bytes=int(properties.total_memory),
        maximum_device_bytes=int(maximum_device_bytes),
        peak_allocated_bytes=peak,
        sampled_rows=int(compact.multiplicities.sum()),
        unique_rows=len(compact.row_ids),
        matrix_nonzeros=int(canonical.nnz),
        thinning_sha256=thinning_sha256,
    )
    return counts, receipt


def accumulate_checkpoint_counts_cuda_dev37(
    matrix: sparse.csr_matrix,
    checkpoint_codes: np.ndarray[Any, Any],
    inverse_probability_weights: np.ndarray[Any, Any],
    *,
    seed: int,
    device: str | torch.device = "cuda",
    maximum_device_bytes: int = DEFAULT_MAX_DEVICE_BYTES,
    row_block_size: int = 4096,
) -> tuple[np.ndarray[Any, Any], Dev37CudaAccumulationReceipt]:
    """Preserve Dev37 PCG64DXSM variate order and offload only int64 reduction."""

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Dev37 CUDA refit requires an available CUDA device.")
    canonical = matrix.copy().tocsr()
    checkpoints = np.asarray(checkpoint_codes, dtype=np.int64)
    weights = np.asarray(inverse_probability_weights, dtype=np.float64)
    integer_weights = np.rint(weights)
    if (
        canonical.shape[0] != len(checkpoints)
        or weights.shape != checkpoints.shape
        or canonical.shape[1] <= 0
        or not canonical.has_canonical_format
        or not canonical.has_sorted_indices
        or np.any(canonical.data == 0)
        or np.any(canonical.data < 0)
        or not np.equal(canonical.data, np.rint(canonical.data)).all()
        or np.any((checkpoints < 0) | (checkpoints > 2))
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.array_equal(weights, integer_weights)
    ):
        raise IntegrityError("Dev37 expanded CUDA inputs violate the exact count contract.")
    properties = torch.cuda.get_device_properties(selected)
    if maximum_device_bytes <= 0 or maximum_device_bytes >= properties.total_memory:
        raise RuntimeError("Dev37 CUDA ceiling must retain physical-device headroom.")

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    thinned = rng.binomial(canonical.data.astype(np.int64), 0.5).astype(np.float64)
    thinning_sha256 = hashlib.sha256(
        np.asarray(thinned, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    thinned_integer = thinned.astype(np.int64)

    torch.cuda.reset_peak_memory_stats(selected)
    output = torch.zeros(3 * canonical.shape[1], dtype=torch.int64, device=selected)
    for row_start in range(0, canonical.shape[0], row_block_size):
        row_end = min(row_start + row_block_size, canonical.shape[0])
        data_start = int(canonical.indptr[row_start])
        data_end = int(canonical.indptr[row_end])
        local_rows = np.repeat(
            np.arange(row_start, row_end, dtype=np.int64),
            np.diff(canonical.indptr[row_start : row_end + 1]),
        )
        flat = checkpoints[local_rows] * canonical.shape[1] + canonical.indices[
            data_start:data_end
        ].astype(np.int64, copy=False)
        contributions = thinned_integer[data_start:data_end] * integer_weights[local_rows].astype(
            np.int64
        )
        index = torch.as_tensor(flat, dtype=torch.int64, device=selected)
        values = torch.as_tensor(contributions, dtype=torch.int64, device=selected)
        output.index_add_(0, index, values)
        del index, values
        if torch.cuda.max_memory_allocated(selected) > maximum_device_bytes:
            raise RuntimeError("Dev37 CUDA accumulation exceeded its allocator ceiling.")
    torch.cuda.synchronize(selected)
    counts = output.reshape(3, canonical.shape[1]).cpu().numpy()
    receipt = Dev37CudaAccumulationReceipt(
        device_name=properties.name,
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        physical_device_bytes=int(properties.total_memory),
        maximum_device_bytes=int(maximum_device_bytes),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(selected)),
        expanded_rows=canonical.shape[0],
        matrix_nonzeros=int(canonical.nnz),
        thinning_sha256=thinning_sha256,
    )
    return counts, receipt


def prepare_resident_checkpoint_counts_cuda(
    matrix: sparse.csr_matrix,
    checkpoint_codes: np.ndarray[Any, Any],
    compact: CompactSampledRows,
    *,
    device: str | torch.device = "cuda",
    maximum_device_bytes: int = DEFAULT_MAX_DEVICE_BYTES,
) -> CudaResidentCompactReduction:
    """Upload one compact reduction surface for repeated CUDA-native refits."""

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G00C resident refit requires an available CUDA device.")
    canonical, checkpoints = _validate_inputs(matrix, checkpoint_codes, compact)
    properties = torch.cuda.get_device_properties(selected)
    if maximum_device_bytes <= 0 or maximum_device_bytes >= properties.total_memory:
        raise RuntimeError("G00C resident CUDA ceiling must retain device headroom.")

    row_counts = np.diff(canonical.indptr)
    rows = np.repeat(np.arange(canonical.shape[0], dtype=np.int64), row_counts)
    base = canonical.data.astype(np.int64, copy=False)
    effective = base * compact.multiplicities[rows]
    weights = compact.inverse_probability_weights[rows]
    if np.any(effective < 0) or np.any(weights <= 0):
        raise IntegrityError("G00C resident CUDA inputs overflowed or have nonpositive weight.")
    flat = checkpoints[rows] * canonical.shape[1] + canonical.indices.astype(np.int64)

    torch.cuda.reset_peak_memory_stats(selected)
    effective_tensor = torch.as_tensor(effective, dtype=torch.float64, device=selected)
    plan = CudaResidentCompactReduction(
        device=selected,
        flat_indices=torch.as_tensor(flat, dtype=torch.int64, device=selected),
        effective_counts=effective_tensor,
        probabilities=torch.full_like(effective_tensor, 0.5),
        weights=torch.as_tensor(weights, dtype=torch.int64, device=selected),
        output=torch.zeros(3 * canonical.shape[1], dtype=torch.int64, device=selected),
        maximum_device_bytes=maximum_device_bytes,
    )
    if torch.cuda.max_memory_allocated(selected) > maximum_device_bytes:
        raise RuntimeError("G00C resident CUDA preparation exceeded its allocator ceiling.")
    return plan
