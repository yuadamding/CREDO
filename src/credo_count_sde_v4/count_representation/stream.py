"""Exact stratified audit selection and bounded count-only training batches."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from ..canonical import canonical_json_bytes, contract_id
from ..data.prepared_shards import PreparedShardReader, RowAddress
from ..errors import ContractError
from ..representation.state_information import thin_counts_by_cell
from .contracts import CountRepresentationSpec
from .network import dense_budget

Partition = dict[str, list[int]]


def shard_key(source: str, shard: int) -> str:
    return contract_id([source, shard])


def reader_for(
    root: Path, spec: CountRepresentationSpec, *, query: bool = False
) -> PreparedShardReader:
    return PreparedShardReader(
        root, spec.query if query else spec.fitting, max_batch_rows=spec.rules.batch_rows
    )


def check_metadata(
    cells: pd.DataFrame, catalog: pd.DataFrame, spec: CountRepresentationSpec
) -> None:
    groups = cells.guide_index.to_numpy()
    if groups.dtype.kind not in "iu" or np.any(groups < 0) or np.any(groups >= len(catalog)):
        raise ContractError("Cell guide index outside complete catalog.")
    for column, field in (
        ("guide_id", "guide_id"),
        ("primary_target_id", "target_id"),
        ("is_control", "is_control"),
    ):
        if column in cells and not np.array_equal(cells[column], catalog[field].to_numpy()[groups]):
            raise ContractError("Cell guide/target/control annotation contradicts catalog.")
    roles = {r.source_id: r for r in spec.source_roles}
    for source, block in cells.groupby("source_id", sort=False):
        role = roles[str(source)]
        condition = (
            spec.source_condition if role.condition_role == "source" else spec.destination_condition
        )
        if "donor_id" in block and not block.donor_id.eq(role.donor_id).all():
            raise ContractError("Cell donor contradicts authorized source role.")
        if "condition" in block and not block.condition.eq(condition).all():
            raise ContractError("Cell condition contradicts authorized source role.")


def partition_audit(root: Path, spec: CountRepresentationSpec) -> tuple[Partition, dict[str, Any]]:
    """Lowest cell-identity hashes per source/guide; singleton strata remain training-only.

    Two metadata-only passes. Store only audit coordinates, never expression or
    a cohort-wide metadata frame. At least one training row per stratum remains.
    Source roles explicitly bind donor/condition; no parsing of source names.
    """
    total = sum(s.selected_rows for s in spec.fitting.shards)
    if total > spec.rules.maximum_partition_rows:
        raise ContractError("Audit partition exceeds declared row budget.")
    reader = reader_for(root, spec)
    catalog = reader.guide_catalog()
    sizes: Counter[tuple[str, int]] = Counter()
    for record in spec.fitting.shards:
        cells = reader.metadata(record.source_id, record.shard)
        check_metadata(cells, catalog, spec)
        sizes.update(
            {
                (record.source_id, int(g)): int(n)
                for g, n in cells.guide_index.value_counts().items()
            }
        )
    reserved = {
        key: min(n - 1, max(1, int(np.floor(n * spec.rules.audit_fraction)))) if n > 1 else 0
        for key, n in sizes.items()
    }
    heaps: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
    for record in spec.fitting.shards:
        cells = reader.metadata(record.source_id, record.shard)
        for row in cells.itertuples(index=False):
            group = (record.source_id, int(row.guide_index))
            count = reserved[group]
            if not count:
                continue
            identity = [
                "representation-audit-v1",
                spec.rules.seed,
                record.source_id,
                str(row.cell_id),
            ]
            rank = int(contract_id(identity), 16)
            candidate = (-rank, record.shard, int(row.row_in_shard))
            heap = heaps[group]
            if len(heap) < count:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    partition: Partition = {shard_key(s.source_id, s.shard): [] for s in spec.fitting.shards}
    for (source, _), heap in heaps.items():
        for _, shard, row in heap:
            partition[shard_key(source, shard)].append(row)
    for rows in partition.values():
        rows.sort()
    audit = sum(map(len, partition.values()))
    if not 0 < audit < total:
        raise ContractError("Nonempty disjoint training/audit populations required.")
    return partition, dict(
        fitting_rows=total,
        calibration_training_rows=total - audit,
        audit_rows=audit,
        requested_fraction=spec.rules.audit_fraction,
        realized_fraction=audit / total,
        strata=len(sizes),
        singleton_training_only_strata=sum(n == 1 for n in sizes.values()),
        selection="source_guide_stratified_lowest_cell_hash_floor_fraction_min_one_if_n_ge_two",
        audit_coordinate_sha256=contract_id(partition),
        authority=spec.fitting.identity(),
    )


def iter_counts(
    root: Path,
    spec: CountRepresentationSpec,
    *,
    partition: Partition | None = None,
    split: Literal["train", "audit", "all", "query"] = "all",
    epoch: int = 0,
) -> Iterator[tuple[sparse.csr_matrix, pd.DataFrame]]:
    reader = reader_for(root, spec, query=split == "query")
    catalog = reader.guide_catalog()
    if split in {"train", "audit"} and partition is None:
        raise ContractError("Calibration stream requires its prespecified audit partition.")
    for record in sorted(reader.access.shards, key=lambda s: (s.source_id, s.shard)):
        rows = (
            np.arange(record.rows)
            if record.allowed_rows is None
            else np.asarray(record.allowed_rows)
        )
        if split in {"train", "audit"}:
            assert partition is not None
            selected = np.isin(rows, partition[shard_key(record.source_id, record.shard)])
            rows = rows[selected if split == "audit" else ~selected]
        # Traversal depends on authorized row identities, not unrelated package/protected hashes.
        seed = int(contract_id([spec.rules.seed, epoch, record.source_id, record.shard])[:32], 16)
        rows = np.random.Generator(np.random.PCG64DXSM(seed)).permutation(rows)
        for start in range(0, len(rows), spec.rules.batch_rows):
            chosen = rows[start : start + spec.rules.batch_rows]
            dense_budget(spec.rules, len(spec.rna_positions), len(chosen))
            block = reader.read_rows(
                [
                    RowAddress(source_id=record.source_id, shard=record.shard, row=int(r))
                    for r in chosen
                ]
            )
            cells = block.cells.copy()
            check_metadata(cells, catalog, spec)
            counts = block.matrix[:, list(spec.rna_positions)].tocsr()
            depth = np.asarray(counts.sum(axis=1), dtype=np.int64).ravel()
            all_depth = np.asarray(block.matrix.sum(axis=1), dtype=np.int64).ravel()
            for column, expected in (("RNA_UMIs", depth), ("all_feature_UMIs", all_depth)):
                if column in cells and not np.array_equal(cells[column], expected):
                    raise ContractError("Declared library depth differs from raw integer counts.")
            cells["shard"] = record.shard
            cells["RNA_UMIs"] = depth
            cells["condition_role"] = next(
                r.condition_role for r in spec.source_roles if r.source_id == record.source_id
            )
            yield counts, cells


def halves(
    counts: sparse.csr_matrix, cells: pd.DataFrame, seed: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    # Scoped cell identity; no query/protected data or B library depth enters A.
    ids = [
        contract_id([str(s), str(c)]) for s, c in zip(cells.source_id, cells.cell_id, strict=True)
    ]
    return thin_counts_by_cell(counts, ids, seed=seed, probability=0.5)


class Exposure:
    """Replayable row-address digest and complete consumed/usable counts, not unique donors."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.by_source: Counter[str] = Counter()
        self.rows = self.positive_rows = self.umis = 0

    def update(self, cells: pd.DataFrame) -> None:
        self.rows += len(cells)
        self.positive_rows += int((cells.RNA_UMIs > 0).sum())
        self.umis += int(cells.RNA_UMIs.sum())
        self.by_source.update(map(str, cells.source_id))
        for row in cells.itertuples(index=False):
            self.digest.update(
                canonical_json_bytes(
                    [str(row.source_id), int(row.shard), int(row.row_in_shard), str(row.cell_id)]
                )
                + b"\n"
            )

    def record(self) -> dict[str, Any]:
        return dict(
            rows=self.rows,
            positive_RNA_rows=self.positive_rows,
            RNA_UMIs=self.umis,
            rows_by_source=dict(self.by_source),
            ordered_row_address_sha256=self.digest.hexdigest(),
        )
