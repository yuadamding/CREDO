"""Bounded reads of immutable, already-canonical integer CSR packages.

The allowlist is an application access guard, NOT an OS sandbox. A production
predictor must additionally run without filesystem access to protected sources.
No cohort identity, split derivation, fitting, or artifact publication lives here.
"""

from __future__ import annotations

import hashlib
import os
import stat
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from scipy import sparse

from ..canonical import contract_id
from ..contracts.models import ArtifactRef, Sha256, StrictModel
from ..errors import ContractError


class PreparedShard(StrictModel):
    source_id: str = Field(min_length=1)
    shard: int = Field(ge=0, strict=True)
    rows: int = Field(gt=0, strict=True)
    nnz: int = Field(ge=0, strict=True)
    counts: ArtifactRef
    cells: ArtifactRef


class PreparedAccess(StrictModel):
    """Resolved, content-bound subset; changing any permission changes identity."""

    schema_version: Literal[1] = 1
    package_completion_sha256: Sha256
    package_inventory_sha256: Sha256
    parent_view_sha256: Sha256
    amendment_sha256: Sha256
    feature_order_sha256: Sha256
    n_features: int = Field(gt=0, strict=True)
    task_id: str = Field(min_length=1)
    role: Literal[
        "representation_fit", "dynamics_source", "dynamics_supervision", "baseline_fit", "query"
    ]
    source_ids: tuple[str, ...]
    shards: tuple[PreparedShard, ...]
    storage_isolation_qualified: Literal[False] = False

    @model_validator(mode="after")
    def valid_allowlist(self) -> PreparedAccess:
        if not self.source_ids or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("Unique, nonempty source allowlist required.")
        keys = [(s.source_id, s.shard) for s in self.shards]
        paths = [a.relative_uri for s in self.shards for a in (s.counts, s.cells)]
        if len(set(keys)) != len(keys) or len(set(paths)) != len(paths):
            raise ValueError("Duplicate shard identity or file alias.")
        if {s.source_id for s in self.shards} != set(self.source_ids):
            raise ValueError("Shard sources must exactly cover the authorized source allowlist.")
        return self


class RowAddress(StrictModel):
    source_id: str = Field(min_length=1)
    shard: int = Field(ge=0, strict=True)
    row: int = Field(ge=0, strict=True)


class StreamCursor(StrictModel):
    schema_version: Literal[1] = 1
    access_sha256: Sha256
    seed: int = Field(ge=0, lt=2**64, strict=True)
    epoch: int = Field(ge=0, strict=True)
    shard_position: int = Field(ge=0, strict=True)
    row_position: int = Field(ge=0, strict=True)


@dataclass(frozen=True)
class CountRows:
    matrix: sparse.csr_matrix
    cells: pd.DataFrame


def _signature(path: Path) -> tuple[int, ...]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ContractError("Input is not a regular non-symlink file.")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def regular_member(root: Path, relative_uri: str) -> Path:
    """Reject symlink components and traversal; a mounted read-only root is still required."""
    from ..canonical import validate_relative_uri

    validate_relative_uri(relative_uri)
    path = root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ContractError("Package root must be an absolute non-symlink directory.")
    for part in Path(relative_uri).parts:
        path = path / part
        if path.is_symlink():
            raise ContractError("Symlink component in package member.")
    _signature(path)
    return path


@contextmanager
def verified_member(root: Path, artifact: ArtifactRef) -> Iterator[BinaryIO]:
    """Hash and consume the same open file, then reject changes during the read."""
    path = regular_member(root, artifact.relative_uri)
    before = _signature(path)
    if before[2] != artifact.size_bytes:
        raise ContractError("Artifact size mismatch.")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise ContractError("Artifact identity changed during open.")
        digest = hashlib.sha256()
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != artifact.sha256:
            raise ContractError("Artifact hash mismatch.")
        handle.seek(0)
        yield handle
        if _signature(path) != before:
            raise ContractError("Artifact changed during read.")


class PreparedShardReader:
    """One cached shard, bounded decompression and bounded sparse output batches.

    Returned rows preserve order and duplicates. Streaming visits every authorized
    row once per epoch, with identity-derived shard/row permutations and a replay
    cursor. This is a single-reader stream, not yet a multi-worker training loader.
    """

    def __init__(
        self,
        root: Path,
        access: PreparedAccess,
        *,
        max_uncompressed_bytes: int = 1024**3,
        max_cached_bytes: int = 768 * 1024**2,
        max_output_bytes: int = 512 * 1024**2,
        max_batch_rows: int = 8192,
    ) -> None:
        limits = (max_uncompressed_bytes, max_cached_bytes, max_output_bytes, max_batch_rows)
        if any(type(n) is not int or n <= 0 for n in limits):
            raise ContractError("Reader limits must be positive integers.")
        self.root, self.access = Path(root), access
        self.max_uncompressed_bytes, self.max_cached_bytes = limits[:2]
        self.max_output_bytes, self.max_batch_rows = limits[2:]
        self._records = {(s.source_id, s.shard): s for s in access.shards}
        self._cache: tuple | None = None
        self._pid = os.getpid()

    def _record(self, source_id: str, shard: int) -> PreparedShard:
        if os.getpid() != self._pid:
            raise ContractError("Reader crossed a process boundary; construct a worker reader.")
        try:
            return self._records[(source_id, shard)]
        except KeyError as error:
            raise ContractError("Unauthorized source or shard.") from error

    def metadata(self, source_id: str, shard: int) -> pd.DataFrame:
        record = self._record(source_id, shard)
        if record.cells.size_bytes > self.max_uncompressed_bytes:
            raise ContractError("Cell metadata file exceeds reader budget.")
        with verified_member(self.root, record.cells) as handle:
            cells = pd.read_parquet(handle)
        if cells.memory_usage(deep=True).sum() > self.max_cached_bytes:
            raise ContractError("Cell metadata exceeds resident budget.")
        required = {"source_id", "row_in_shard", "cell_id", "guide_index"}
        if not required.issubset(cells.columns) or len(cells) != record.rows:
            raise ContractError("Cell metadata shape/columns mismatch.")
        if (
            not cells.source_id.eq(source_id).all()
            or not np.array_equal(cells.row_in_shard, np.arange(record.rows))
            or not cells.cell_id.is_unique
        ):
            raise ContractError("Cell metadata identity/order mismatch.")
        return cells

    def _load(self, source_id: str, shard: int) -> CountRows:
        record = self._record(source_id, shard)
        signatures = tuple(
            _signature(regular_member(self.root, a.relative_uri))
            for a in (record.counts, record.cells)
        )
        if self._cache is not None and self._cache[:2] == (source_id, shard):
            if signatures != self._cache[2]:
                raise ContractError("Cached artifact changed.")
            return self._cache[3]
        self._cache = None  # release before allocating another decompressed shard
        with verified_member(self.root, record.counts) as handle:
            with zipfile.ZipFile(handle) as archive:
                entries = archive.infolist()
                if (
                    {e.filename for e in entries}
                    != {"data.npy", "indices.npy", "indptr.npy", "shape.npy", "format.npy"}
                    or len(entries) != 5
                    or sum(e.file_size for e in entries) > self.max_uncompressed_bytes
                ):
                    raise ContractError("CSR archive members/decompression budget mismatch.")
            handle.seek(0)
            matrix = sparse.load_npz(handle)
        if (
            not sparse.isspmatrix_csr(matrix)
            or matrix.dtype != np.uint32
            or matrix.shape != (record.rows, self.access.n_features)
            or matrix.nnz != record.nnz
        ):
            raise ContractError("Count matrix type/shape mismatch.")
        matrix.check_format(full_check=True)
        if not matrix.has_canonical_format:
            raise ContractError("Prepared counts must already have canonical feature order.")
        cells = self.metadata(source_id, shard)
        resident = sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))
        resident += int(cells.memory_usage(deep=True).sum())
        if resident > self.max_cached_bytes:
            raise ContractError("Decompressed shard exceeds resident cache budget.")
        result = CountRows(matrix, cells)
        self._cache = (source_id, shard, signatures, result)
        return result

    def read_rows(self, addresses: Sequence[RowAddress]) -> CountRows:
        if len(addresses) > self.max_batch_rows:
            raise ContractError("Requested rows exceed batch budget.")
        groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
        # Authorize the ENTIRE request before opening any count file.
        for index, address in enumerate(addresses):
            record = self._record(address.source_id, address.shard)
            if address.row >= record.rows:
                raise ContractError("Requested row is outside shard.")
            groups.setdefault((address.source_id, address.shard), []).append((index, address.row))
        if not addresses:
            return CountRows(
                sparse.csr_matrix((0, self.access.n_features), dtype=np.uint32), pd.DataFrame()
            )
        matrices, metadata, positions = [], [], []
        output_bytes = 0
        for key, selections in groups.items():
            block = self._load(*key)
            rows = np.asarray([r for _, r in selections], dtype=np.int64)
            nonzeros = int((block.matrix.indptr[rows + 1] - block.matrix.indptr[rows]).sum())
            output_bytes += nonzeros * 12 + (len(rows) + 1) * 8
            if output_bytes > self.max_output_bytes:
                raise ContractError("Requested sparse output exceeds byte budget.")
            matrices.append(block.matrix[rows])
            metadata.append(block.cells.iloc[rows])
            positions.extend(i for i, _ in selections)
        restore = np.argsort(positions)
        return CountRows(
            sparse.vstack(matrices, format="csr")[restore],
            pd.concat(metadata, ignore_index=True).iloc[restore].reset_index(drop=True),
        )

    def iterate_rows(
        self, *, batch_size: int, seed: int = 0, epoch: int = 0, cursor: StreamCursor | None = None
    ) -> Iterator[tuple[StreamCursor, CountRows]]:
        if type(batch_size) is not int or not 0 < batch_size <= self.max_batch_rows:
            raise ContractError("Invalid streaming batch size.")
        identity = self.access.identity()
        current = cursor or StreamCursor(
            access_sha256=identity, seed=seed, epoch=epoch, shard_position=0, row_position=0
        )
        if (current.access_sha256, current.seed, current.epoch) != (identity, seed, epoch):
            raise ContractError("Resume cursor differs from access/seed/epoch.")

        def generator(label: str) -> np.random.Generator:
            key = contract_id([identity, seed, epoch, label])
            return np.random.Generator(np.random.PCG64DXSM(int(key[:32], 16)))

        order = generator("shards").permutation(len(self.access.shards))
        if current.shard_position > len(order) or (
            current.shard_position == len(order) and current.row_position != 0
        ):
            raise ContractError("Resume cursor is beyond the stream.")
        for position in range(current.shard_position, len(order)):
            record = self.access.shards[int(order[position])]
            rows = generator(f"rows:{record.source_id}:{record.shard}").permutation(record.rows)
            start = current.row_position if position == current.shard_position else 0
            if start >= record.rows:
                raise ContractError("Resume cursor is beyond the shard.")
            for left in range(start, record.rows, batch_size):
                right = min(left + batch_size, record.rows)
                following = StreamCursor(
                    access_sha256=identity,
                    seed=seed,
                    epoch=epoch,
                    shard_position=position + (right == record.rows),
                    row_position=0 if right == record.rows else right,
                )
                batch = self.read_rows(
                    [
                        RowAddress(source_id=record.source_id, shard=record.shard, row=int(row))
                        for row in rows[left:right]
                    ]
                )
                yield following, batch
