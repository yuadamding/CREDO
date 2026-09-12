from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import sparse

from credo_count_sde_v4.canonical import sha256_file
from credo_count_sde_v4.compile.distribution_problem import compile_population_rows
from credo_count_sde_v4.contracts.models import ArtifactRef
from credo_count_sde_v4.data.prepared_shards import (
    PreparedAccess,
    PreparedShard,
    PreparedShardReader,
    RowAddress,
    StreamCursor,
)
from credo_count_sde_v4.errors import ContractError


def artifact(root: Path, name: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id="fixture",
        schema_version=1,
        sha256=sha256_file(root / name),
        size_bytes=(root / name).stat().st_size,
        media_type="application/octet-stream",
        relative_uri=name,
    )


@pytest.fixture
def prepared(tmp_path):
    records = []
    for index, values in enumerate(([[2, 0, 4], [0, 2, 8]], [[3, 1, 0], [1, 3, 2]])):
        matrix = sparse.csr_matrix(np.array(values, dtype=np.uint32))
        sparse.save_npz(tmp_path / f"{index}.npz", matrix)
        pd.DataFrame(
            {
                "source_id": ["source"] * 2,
                "row_in_shard": [0, 1],
                "cell_id": [f"cell{2 * index}", f"cell{2 * index + 1}"],
                "guide_index": [0, 1] if index == 0 else [0, 0],
            }
        ).to_parquet(tmp_path / f"{index}.parquet", index=False)
        records.append(
            PreparedShard(
                source_id="source",
                shard=index,
                rows=2,
                nnz=matrix.nnz,
                counts=artifact(tmp_path, f"{index}.npz"),
                cells=artifact(tmp_path, f"{index}.parquet"),
            )
        )
    access = PreparedAccess(
        package_completion_sha256="a" * 64,
        package_inventory_sha256="b" * 64,
        parent_view_sha256="c" * 64,
        amendment_sha256="d" * 64,
        feature_order_sha256="e" * 64,
        n_features=3,
        task_id="endpoint",
        role="representation_fit",
        source_ids=("source",),
        shards=tuple(records),
    )
    return tmp_path, access


def address(shard, row, source="source"):
    return RowAddress(source_id=source, shard=shard, row=row)


def test_reordered_repeated_cross_shard_rows(prepared):
    root, access = prepared
    reader = PreparedShardReader(root, access)
    batch = reader.read_rows([address(1, 1), address(0, 0), address(1, 1), address(0, 1)])
    assert batch.matrix.toarray().tolist() == [[1, 3, 2], [2, 0, 4], [1, 3, 2], [0, 2, 8]]
    assert batch.cells.cell_id.tolist() == ["cell3", "cell0", "cell3", "cell1"]
    assert batch.matrix.dtype == np.uint32


def test_unauthorized_request_rejected_before_any_count_read(prepared, monkeypatch):
    root, access = prepared
    reader = PreparedShardReader(root, access)

    def forbidden(*args):
        pytest.fail("No file should have been opened")

    monkeypatch.setattr(reader, "_load", forbidden)
    with pytest.raises(ContractError, match="Unauthorized"):
        reader.read_rows([address(0, 0), address(0, 0, "protected")])
    with pytest.raises(ContractError, match="outside"):
        reader.read_rows([address(0, 5)])
    with pytest.raises(ContractError, match="Unauthorized"):
        reader.metadata("protected", 0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_batch_rows", 1),
        ("max_uncompressed_bytes", 1),
        ("max_cached_bytes", 1),
        ("max_output_bytes", 1),
    ],
)
def test_reader_budgets_fail_closed(prepared, name, value):
    root, access = prepared
    reader = PreparedShardReader(root, access, **{name: value})
    with pytest.raises(ContractError, match="budget"):
        reader.read_rows([address(0, 0), address(0, 1)])


def test_empty_rows_are_sparse(prepared):
    root, access = prepared
    batch = PreparedShardReader(root, access).read_rows([])
    assert batch.matrix.shape == (0, 3) and batch.cells.empty


def test_same_size_corruption_rejected_before_decoding(prepared):
    root, access = prepared
    path = root / "0.npz"
    payload = bytearray(path.read_bytes())
    payload[30] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ContractError, match="hash mismatch"):
        PreparedShardReader(root, access).read_rows([address(0, 0)])


def test_mutation_of_cached_shard_rejected(prepared):
    root, access = prepared
    reader = PreparedShardReader(root, access)
    reader.read_rows([address(0, 0)])
    with (root / "0.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ContractError, match="Cached artifact changed"):
        reader.read_rows([address(0, 0)])


def test_symlink_rejected(prepared):
    root, access = prepared
    (root / "0.npz").rename(root / "saved.npz")
    (root / "0.npz").symlink_to(root / "saved.npz")
    with pytest.raises(ContractError, match="Symlink"):
        PreparedShardReader(root, access).read_rows([address(0, 0)])


def test_stream_full_coverage_and_resume(prepared):
    root, access = prepared
    reader = PreparedShardReader(root, access)
    full = list(reader.iterate_rows(batch_size=1, seed=123, epoch=2))
    assert sorted(b.cells.cell_id.iloc[0] for _, b in full) == [f"cell{i}" for i in range(4)]
    cursor = StreamCursor.model_validate_json(full[0][0].model_dump_json())
    resumed = list(
        PreparedShardReader(root, access).iterate_rows(
            batch_size=1, seed=123, epoch=2, cursor=cursor
        )
    )
    assert [c for c, _ in resumed] == [c for c, _ in full[1:]]
    for (_, actual), (_, expected) in zip(resumed, full[1:], strict=True):
        np.testing.assert_array_equal(actual.matrix.toarray(), expected.matrix.toarray())
        pd.testing.assert_frame_equal(actual.cells, expected.cells)
    assert list(reader.iterate_rows(batch_size=1, seed=123, epoch=2, cursor=full[-1][0])) == []
    with pytest.raises(ContractError, match="differs"):
        list(reader.iterate_rows(batch_size=1, seed=124, epoch=2, cursor=cursor))


def test_invalid_cursor_and_process_transition(prepared, monkeypatch):
    root, access = prepared
    reader = PreparedShardReader(root, access)
    cursor = StreamCursor(
        access_sha256=access.identity(), seed=0, epoch=0, shard_position=99, row_position=0
    )
    with pytest.raises(ContractError, match="beyond"):
        list(reader.iterate_rows(batch_size=2, cursor=cursor))
    monkeypatch.setattr("credo_count_sde_v4.data.prepared_shards.os.getpid", lambda: -1)
    with pytest.raises(ContractError, match="process boundary"):
        reader.read_rows([address(0, 0)])


def test_catalog_index_preserves_empirical_support_and_zeros(prepared):
    root, access = prepared
    reader = PreparedShardReader(root, access)
    index = compile_population_rows(reader, "source", ("g0", "g1", "absent"), np.array([3, 1, 0]))
    assert index.offsets.tolist() == [0, 3, 4, 4]
    assert index.coordinates.shape == (4, 2)
    assert len(index.identity()) == 64
    rows = index.addresses(0, np.array([2, 0, 2]))
    assert reader.read_rows(rows).cells.cell_id.tolist() == ["cell3", "cell0", "cell3"]
    assert index.addresses(2, np.array([], dtype=int)) == ()
    with pytest.raises(ContractError, match="abstention"):
        index.addresses(2, np.array([0]))
    with pytest.raises(ValueError):
        index.coordinates[0] = 0


def test_population_index_denominator_and_capacity(prepared):
    root, access = prepared
    reader = PreparedShardReader(root, access)
    with pytest.raises(ContractError, match="complete authorized source"):
        compile_population_rows(reader, "source", ("g",), np.array([2]))
    with pytest.raises(ContractError, match="guide support"):
        compile_population_rows(reader, "source", ("g0", "g1", "g2"), np.array([1, 3, 0]))
    with pytest.raises(ContractError, match="memory budget"):
        compile_population_rows(reader, "source", ("g0", "g1"), np.array([3, 1]), max_index_bytes=1)


@pytest.mark.parametrize(
    "updates",
    [
        {"role": "truth"},
        {"storage_isolation_qualified": True},
        {"unexpected": 1},
        {"source_ids": ["source", "protected"]},
        {"source_ids": ["source", "source"]},
    ],
)
def test_strict_contracts_reject_false_permissions(prepared, updates):
    _, access = prepared
    with pytest.raises(ValidationError):
        PreparedAccess.model_validate({**access.model_dump(), **updates})


@pytest.mark.parametrize("row", [-1, True, 1.2])
def test_address_requires_exact_nonnegative_integer(row):
    with pytest.raises(ValidationError):
        address(0, row)
