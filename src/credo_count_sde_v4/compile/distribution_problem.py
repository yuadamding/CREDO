"""Count-backed empirical population supports; no latent-centroid compilation.

This index binds ALL rows in one authorized source to the complete guide catalog.
It is a reader primitive, not a fitted latent artifact or a compiled trainer run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..canonical import contract_id, sha256_bytes
from ..data.prepared_shards import PreparedShardReader, RowAddress
from ..errors import ContractError


@dataclass(frozen=True)
class PopulationRowIndex:
    access_sha256: str
    source_id: str
    guide_ids: tuple[str, ...]
    offsets: np.ndarray
    coordinates: np.ndarray

    def addresses(self, guide_index: int, positions: np.ndarray) -> tuple[RowAddress, ...]:
        if type(guide_index) is not int or not 0 <= guide_index < len(self.guide_ids):
            raise ContractError("Unknown guide index.")
        positions = np.asarray(positions)
        count = int(self.offsets[guide_index + 1] - self.offsets[guide_index])
        if positions.ndim != 1 or positions.dtype.kind not in "iu":
            raise ContractError("Integer population positions required.")
        if np.any(positions < 0) or np.any(positions >= count):
            raise ContractError(
                "Population position unavailable; zero support requires abstention."
            )
        rows = self.coordinates[self.offsets[guide_index] + positions.astype(np.int64)]
        return tuple(
            RowAddress(source_id=self.source_id, shard=int(shard), row=int(row))
            for shard, row in rows
        )

    def identity(self) -> str:
        return contract_id(
            {
                "schema_version": 1,
                "access_sha256": self.access_sha256,
                "source_id": self.source_id,
                "guide_ids": self.guide_ids,
                "offsets_sha256": sha256_bytes(self.offsets.astype("<i8").tobytes()),
                "coordinates_sha256": sha256_bytes(self.coordinates.astype("<i4").tobytes()),
            }
        )


def compile_population_rows(
    reader: PreparedShardReader,
    source_id: str,
    guide_ids: tuple[str, ...],
    counts: np.ndarray,
    *,
    max_index_bytes: int = 512 * 1024**2,
) -> PopulationRowIndex:
    """One bounded metadata pass; offsets retain observed zero-count categories."""
    if source_id not in reader.access.source_ids:
        raise ContractError("Unauthorized population source.")
    values = np.asarray(counts)
    if (
        values.shape != (len(guide_ids),)
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or len(set(guide_ids)) != len(guide_ids)
        or not guide_ids
    ):
        raise ContractError("Invalid complete population catalog.")
    total = sum(map(int, values))
    records = [s for s in reader.access.shards if s.source_id == source_id]
    if total != sum(s.rows for s in records):
        raise ContractError("Population counts do not cover the complete authorized source.")
    if total * 8 + (len(guide_ids) + 1) * 8 > max_index_bytes:
        raise ContractError("Population index exceeds memory budget.")
    if any(max(s.shard, s.rows) >= 2**31 for s in records):
        raise ContractError("Population coordinate exceeds int32 representation.")
    offsets = np.r_[0, np.cumsum(values, dtype=np.int64)]
    coordinates = np.empty((total, 2), dtype=np.int32)
    cursor = offsets[:-1].copy()
    for record in records:
        cells = reader.metadata(source_id, record.shard)
        groups = cells.guide_index.to_numpy()
        if groups.dtype.kind not in "iu" or np.any(groups < 0) or np.any(groups >= len(guide_ids)):
            raise ContractError("Cell guide index outside the complete catalog.")
        ordered = np.argsort(groups, kind="stable")
        unique, starts, sizes = np.unique(groups[ordered], return_index=True, return_counts=True)
        for group, start, size in zip(unique, starts, sizes, strict=True):
            stop = cursor[group] + size
            if stop > offsets[group + 1]:
                raise ContractError("Cell metadata exceeds declared guide support.")
            coordinates[cursor[group] : stop, 0] = record.shard
            coordinates[cursor[group] : stop, 1] = ordered[start : start + size]
            cursor[group] = stop
    if not np.array_equal(cursor, offsets[1:]):
        raise ContractError("Cell metadata does not cover declared guide support.")
    offsets.flags.writeable = False
    coordinates.flags.writeable = False
    return PopulationRowIndex(reader.access.identity(), source_id, guide_ids, offsets, coordinates)
