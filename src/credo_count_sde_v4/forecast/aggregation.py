"""Complete-source sufficient statistics without per-shard dense disk expansion."""

from __future__ import annotations

import resource
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse

from ..canonical import contract_id
from ..data.prepared_shards import (
    PreparedAccess,
    PreparedEvaluationAccess,
    PreparedShardReader,
    RowAddress,
)
from ..errors import ContractError
from ..runtime_identity import environment_identity, implementation_tree_hash
from .artifacts import publish_bundle, verify_bundle
from .contracts import ForecastSpec, process_environment


def verify_runtime(spec: ForecastSpec) -> None:
    if (
        implementation_tree_hash() != spec.implementation_sha256
        or environment_identity() != spec.environment
        or process_environment() != spec.process_environment
    ):
        raise ContractError("Frozen forecast source or runtime environment changed.")


def _source_task(arguments: tuple[Any, ...]) -> dict[str, Any]:
    root_value, access_value, spec_value, source_id, output_value = arguments
    spec = ForecastSpec.model_validate(spec_value)
    verify_runtime(spec)
    access = (
        PreparedEvaluationAccess if access_value["role"] == "evaluation_truth" else PreparedAccess
    ).model_validate(access_value)
    output = Path(output_value)
    records = sorted((s for s in access.shards if s.source_id == source_id), key=lambda s: s.shard)
    parents = {"access": access.identity()}
    for record in records:
        parents[f"counts_{record.shard}"] = record.counts.sha256
        parents[f"cells_{record.shard}"] = record.cells.sha256
    if output.exists():
        old = verify_bundle(output, stage="source_summary", specification=spec.identity())
        if old.parents != parents:
            raise ContractError("Source restart authority changed.")
        return {**old.facts, "reused": True, "path": str(output), "identity": old.identity()}
    started = time.monotonic()
    reader = PreparedShardReader(Path(root_value), access)
    catalog = reader.guide_catalog()
    g, f = len(catalog), len(spec.rna_positions)
    sums = np.zeros((g, f), dtype=np.float64)
    raw_sums = (
        np.zeros((g, f), dtype=np.int64) if isinstance(access, PreparedEvaluationAccess) else None
    )
    cells = np.zeros(g, dtype=np.int64)
    positive = np.zeros(g, dtype=np.int64)
    for record in records:
        physical = (
            np.arange(record.rows)
            if record.allowed_rows is None
            else np.asarray(record.allowed_rows)
        )
        for start in range(0, len(physical), spec.batch_rows):
            addresses = [
                RowAddress(source_id=source_id, shard=record.shard, row=int(row))
                for row in physical[start : start + spec.batch_rows]
            ]
            batch = reader.read_rows(addresses)
            guides = batch.cells.guide_index.to_numpy(dtype=np.int64)
            if np.any(guides < 0) or np.any(guides >= g):
                raise ContractError("Cell guide index is outside the bound catalog.")
            for column, expected in (
                ("guide_id", catalog.guide_id),
                ("primary_target_id", catalog.target_id),
                ("is_control", catalog.is_control),
            ):
                if column in batch.cells and not np.array_equal(
                    batch.cells[column], expected.iloc[guides]
                ):
                    raise ContractError("Cell biological labels disagree with the guide catalog.")
            rna = batch.matrix[:, spec.rna_positions]
            depth = np.asarray(rna.sum(axis=1, dtype=np.int64)).ravel()
            if np.any(depth < 0):
                raise ContractError("RNA depth integer overflow.")
            if "RNA_UMIs" in batch.cells and not np.array_equal(depth, batch.cells.RNA_UMIs):
                raise ContractError("RNA depth disagrees with prepared cell metadata.")
            if "all_feature_UMIs" in batch.cells:
                total = np.asarray(batch.matrix.sum(axis=1, dtype=np.int64)).ravel()
                if not np.array_equal(total, batch.cells.all_feature_UMIs):
                    raise ContractError("Full-feature depth disagrees with prepared cell metadata.")
                if "technical_PuroR_UMIs" in batch.cells and not np.array_equal(
                    total - depth, batch.cells.technical_PuroR_UMIs
                ):
                    raise ContractError("Technical counts cannot enter RNA normalization.")
            unique, inverse = np.unique(guides, return_inverse=True)
            weights = np.divide(1.0, depth, out=np.zeros(len(depth)), where=depth > 0)
            grouping = sparse.csr_matrix(
                (weights, (inverse, np.arange(len(guides)))), shape=(len(unique), len(guides))
            )
            normalized = (grouping @ rna).toarray()
            sums[unique] += normalized
            cells += np.bincount(guides, minlength=g)
            positive += np.bincount(guides[depth > 0], minlength=g)
            if raw_sums is not None:
                grouping.data = np.ones(grouping.nnz, dtype=np.int64)
                raw_sums[unique] += (grouping @ rna).toarray()
        if np.any(cells < 0) or raw_sums is not None and np.any(raw_sums < 0):
            raise ContractError("Integer sufficient-statistic overflow.")
    expected_rows = sum(r.selected_rows for r in records)
    if int(cells.sum()) != expected_rows or not np.allclose(
        sums.sum(1), positive, rtol=1e-12, atol=1e-9
    ):
        raise ContractError("Source statistics lost cell coverage or RNA mass.")

    def write(path: Path) -> dict[str, Any]:
        with h5py.File(path / "statistics.h5", "x") as handle:
            handle.create_dataset("composition_sum", data=sums, chunks=(min(32, g), f))
            if raw_sums is not None:
                handle.create_dataset("count_sum", data=raw_sums, chunks=(min(32, g), f))
            handle.create_dataset("n_cells", data=cells)
            handle.create_dataset("n_rna_cells", data=positive)
            handle.attrs["guide_catalog_sha256"] = spec.guide_catalog_sha256
            handle.attrs["rna_order_sha256"] = spec.rna_order_sha256
        return dict(
            source_id=source_id,
            cells=int(cells.sum()),
            rna_cells=int(positive.sum()),
            guides=g,
            rna_features=f,
            shards=len(records),
            zero_guides=int(np.sum(cells == 0)),
            access_sha256=access.identity(),
            complete_denominator=True,
            includes_counts=raw_sums is not None,
            elapsed_seconds=time.monotonic() - started,
            worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            recovery_unit="complete_source_replay_incomplete_source",
        )

    result = publish_bundle(
        output, stage="source_summary", specification=spec.identity(), parents=parents, writer=write
    )
    return {**result.facts, "reused": False, "path": str(output), "identity": result.identity()}


def aggregate_view(
    package_root: Path,
    access: PreparedAccess | PreparedEvaluationAccess,
    spec: ForecastSpec,
    output: Path,
    *,
    prediction: Path | None = None,
    progress: Any = None,
    stop_after_new_sources: int | None = None,
) -> dict[str, Path]:
    """Visit all authorized rows; recover only complete, hash-verified source units.

    Incomplete sources are recomputed in canonical shard/row order. This is
    sufficient-statistic recovery, not optimizer-level resume qualification.
    No per-shard dense files, copied matrices, or latent caches are created.
    """
    verify_runtime(spec)
    if isinstance(access, PreparedAccess) and access.identity() not in {
        spec.fitting.identity(),
        spec.query.identity(),
    }:
        raise ContractError("Aggregation view is outside the frozen fitting/query specification.")
    if isinstance(access, PreparedEvaluationAccess):
        from .evaluation import verify_evaluation_access

        if prediction is None:
            raise ContractError("Endpoint reads require a published prediction bundle.")
        verify_evaluation_access(prediction, access, spec)
    guides = len(PreparedShardReader(package_root, access).guide_catalog())
    raw_multiplier = 2 if isinstance(access, PreparedEvaluationAccess) else 1
    retained = guides * len(spec.rna_positions) * 8 * raw_multiplier
    resident = min(spec.worker_count, len(access.source_ids)) * (
        retained + spec.batch_rows * len(spec.rna_positions) * 8 * 5 + 2 * 1024**3
    )
    if resident > spec.maximum_process_tree_rss_bytes:
        raise ContractError("Aggregation resident-memory estimate exceeds the execution budget.")
    if retained * len(access.source_ids) > spec.maximum_output_bytes:
        raise ContractError("Aggregation output estimate exceeds the execution budget.")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(package_root),
            access.model_dump(mode="json"),
            spec.model_dump(mode="json"),
            source,
            str(output / contract_id(source)[:16]),
        )
        for source in access.source_ids
    ]
    results = []
    new = 0
    if spec.worker_count == 1:
        for result in map(_source_task, jobs):
            results.append(result)
            new += not result["reused"]
            if progress is not None:
                progress(result)
            if stop_after_new_sources is not None and new >= stop_after_new_sources:
                raise InterruptedError("Intentional stop after committed source summaries.")
    else:
        if stop_after_new_sources is not None:
            raise ContractError("Fault injection requires serial canary mode.")
        with ProcessPoolExecutor(
            max_workers=spec.worker_count, mp_context=get_context("spawn")
        ) as executor:
            for result in executor.map(_source_task, jobs, chunksize=1):
                results.append(result)
                if progress is not None:
                    progress(result)
    return {r["source_id"]: Path(r["path"]) for r in results}
