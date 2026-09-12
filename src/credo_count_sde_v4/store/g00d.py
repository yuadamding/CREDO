"""Artifact-derived verification for the G00D integrated-loader gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..canonical import sha256_file
from ..contracts import (
    ArtifactRef,
    G00DParityGateEvidence,
    IntegratedLoaderQualificationContractV2,
    IntegratedLoaderQualificationReceiptV2,
)
from ..errors import IntegrityError
from .qualification import validate_integrated_loader_qualification


def _artifact_path(root: Path, artifact: ArtifactRef) -> Path:
    path = root / artifact.relative_uri
    if (
        not path.is_file()
        or path.stat().st_size != artifact.size_bytes
        or sha256_file(path) != artifact.sha256
    ):
        raise IntegrityError(f"G00D artifact failed verification: {artifact.relative_uri}.")
    return path


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)


def _slope_interval(
    times: np.ndarray[Any, Any], values: np.ndarray[Any, Any]
) -> tuple[float, float]:
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise IntegrityError("G00D memory trace requires at least three increasing timestamps.")
    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise IntegrityError("G00D memory trace cannot estimate a slope.")
    slope = float(np.dot(centered, values - values.mean()) / denominator)
    fitted = values.mean() + slope * centered
    residual = values - fitted
    variance = float(np.dot(residual, residual) / (len(times) - 2))
    standard_error = math.sqrt(max(variance, 0.0) / denominator)
    upper = slope + float(stats.t.ppf(0.975, len(times) - 2)) * standard_error
    return slope, upper


def verify_integrated_loader_qualification(
    root: Path,
    contract: IntegratedLoaderQualificationContractV2,
    receipt: IntegratedLoaderQualificationReceiptV2,
) -> None:
    """Verify G00D files and recompute every persisted performance summary."""

    measurement = json.loads(_artifact_path(root, receipt.measurement_evidence).read_text())
    expected_measurement = {
        "measurement_protocol_sha256": receipt.measurement_protocol_sha256,
        "gpu_name": receipt.gpu_name,
        "gpu_uuid": receipt.gpu_uuid,
        "gpu_count": receipt.gpu_count,
        "cuda_version": receipt.cuda_version,
        "torch_version": receipt.torch_version,
        "container_digest": receipt.container_digest,
        "worker_count": receipt.worker_count,
        "cpu_count": receipt.cpu_count,
        "storage_authority_hash": receipt.storage_authority_hash,
        "microbatch_cells": receipt.microbatch_cells,
        "microbatches_per_update": receipt.microbatches_per_update,
        "macrobatch_cells": receipt.macrobatch_cells,
        "prefetch_depth": receipt.prefetch_depth,
        "warmup_updates": receipt.warmup_updates,
        "measured_updates": receipt.measured_updates,
        "cold_start_measured": receipt.cold_start_measured,
        "steady_state_measured": receipt.steady_state_measured,
        "cache_policy": receipt.cache_policy,
        "telemetry_interval_seconds": receipt.telemetry_interval_seconds,
        "loader_error_count": receipt.loader_error_count,
        "cuda_error_count": receipt.cuda_error_count,
        "monitor_error_count": receipt.monitor_error_count,
    }
    if measurement != expected_measurement:
        raise IntegrityError("G00D measurement evidence differs from its receipt.")

    telemetry = pd.read_parquet(_artifact_path(root, receipt.telemetry_artifact))
    expected_telemetry_columns = (
        "update",
        "compute_seconds",
        "data_wait_seconds",
        "batch_ready_seconds",
        "gpu_utilization",
    )
    if (
        tuple(telemetry.columns) != expected_telemetry_columns
        or len(telemetry) != receipt.measured_updates
        or telemetry["update"].duplicated().any()
    ):
        raise IntegrityError("G00D telemetry has an invalid schema or update universe.")
    numeric = telemetry[list(expected_telemetry_columns[1:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any(numeric < 0):
        raise IntegrityError("G00D telemetry contains invalid numerical values.")
    compute = telemetry["compute_seconds"].to_numpy(dtype=float)
    wait = telemetry["data_wait_seconds"].to_numpy(dtype=float)
    derived_telemetry = (
        (receipt.median_compute_seconds, float(np.median(compute))),
        (receipt.p95_compute_seconds, float(np.quantile(compute, 0.95))),
        (receipt.median_data_wait_seconds, float(np.median(wait))),
        (
            receipt.p95_batch_ready_seconds,
            float(np.quantile(telemetry["batch_ready_seconds"], 0.95)),
        ),
        (receipt.data_wait_fraction, float(wait.sum() / (wait.sum() + compute.sum()))),
        (
            receipt.steady_state_gpu_utilization,
            float(telemetry["gpu_utilization"].mean()),
        ),
    )
    if any(not _close(observed, expected) for observed, expected in derived_telemetry):
        raise IntegrityError("G00D telemetry summaries are not artifact-derived.")

    parity = pd.read_parquet(_artifact_path(root, receipt.parity_artifact))
    parity_columns = (
        "gate",
        "comparison_mode",
        "reference_sha256",
        "observed_sha256",
        "maximum_absolute_error",
        "maximum_relative_error",
    )
    if tuple(parity.columns) != parity_columns:
        raise IntegrityError("G00D parity artifact has an invalid schema.")
    observed_gates = tuple(
        G00DParityGateEvidence.model_validate(record) for record in parity.to_dict(orient="records")
    )
    if observed_gates != receipt.parity_gates:
        raise IntegrityError("G00D parity evidence differs from its receipt.")

    memory = pd.read_parquet(_artifact_path(root, receipt.memory_trace_artifact))
    memory_columns = (
        "sample_time_seconds",
        "loader_rss_bytes",
        "process_loader_rss_bytes",
        "aggregate_worker_rss_bytes",
        "open_shards",
        "open_file_handles",
    )
    if tuple(memory.columns) != memory_columns:
        raise IntegrityError("G00D memory trace has an invalid schema.")
    memory_values = memory[list(memory_columns)].to_numpy(dtype=float)
    if not np.isfinite(memory_values).all() or np.any(memory_values < 0):
        raise IntegrityError("G00D memory trace contains invalid numerical values.")
    times = memory["sample_time_seconds"].to_numpy(dtype=float)
    aggregate = memory["aggregate_worker_rss_bytes"].to_numpy(dtype=float)
    slope, upper = _slope_interval(times, aggregate)
    memory_summaries = (
        (receipt.peak_loader_rss_bytes, int(memory["loader_rss_bytes"].max())),
        (
            receipt.peak_process_loader_rss_bytes,
            int(memory["process_loader_rss_bytes"].max()),
        ),
        (
            receipt.peak_aggregate_worker_rss_bytes,
            int(memory["aggregate_worker_rss_bytes"].max()),
        ),
        (receipt.peak_open_shards, int(memory["open_shards"].max())),
        (receipt.peak_open_file_handles, int(memory["open_file_handles"].max())),
        (receipt.maximum_rss_excursion_bytes, int(aggregate.max() - aggregate.min())),
    )
    if (
        any(observed != expected for observed, expected in memory_summaries)
        or not _close(receipt.rss_slope_bytes_per_second, slope)
        or not _close(receipt.rss_slope_upper_ci_bytes_per_second, upper)
    ):
        raise IntegrityError("G00D memory summaries are not artifact-derived.")
    if receipt.lru_bound_pass != (
        receipt.peak_open_shards <= contract.maximum_open_shards
        and receipt.peak_open_file_handles <= contract.maximum_open_file_handles
    ):
        raise IntegrityError("G00D LRU-bound flag differs from its trace.")
    validate_integrated_loader_qualification(contract, receipt)
