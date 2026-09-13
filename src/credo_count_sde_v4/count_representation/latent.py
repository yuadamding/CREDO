"""Frozen individual-cell laws, with explicit unknown geometry and complete support."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..canonical import contract_id, sha256_file
from ..data.prepared_shards import PreparedAccess
from ..errors import ContractError
from .artifacts import output_check, publish, runtime_check, verify
from .contracts import RepresentationManifest
from .stream import Exposure, iter_counts, reader_for
from .workflow import load_representation


def compile_latent_cells(
    root: Path,
    fitted: Path,
    access: PreparedAccess,
    destination: Path,
    *,
    role: Literal["fitting", "query"],
) -> RepresentationManifest:
    output_check(root, destination)
    model, spec, fitted_record = load_representation(fitted)
    expected = spec.fitting if role == "fitting" else spec.query if role == "query" else None
    if expected is None or access != expected:
        raise ContractError("Latent compilation requires the exact frozen fitting/query access.")
    catalog = reader_for(root, spec, query=role == "query").guide_catalog()
    total = sum(s.selected_rows for s in access.shards)
    if (
        total * 8 + (len(access.source_ids) * len(catalog) + 1) * 24
        > spec.rules.maximum_latent_index_bytes
    ):
        raise ContractError("Complete latent population index exceeds byte budget.")

    def writer(path: Path) -> dict[str, Any]:
        exposure = Exposure()
        support = {
            source: np.zeros((len(catalog), 2), dtype=np.int64) for source in access.source_ids
        }
        offset = 0
        metadata_writer = None
        try:
            with h5py.File(path / "latent.h5", "x") as handle, torch.inference_mode():
                states = handle.create_dataset(
                    "state",
                    shape=(total, spec.rules.latent_dim),
                    dtype="float32",
                    chunks=(min(total, spec.rules.batch_rows), spec.rules.latent_dim),
                )
                for counts, cells in iter_counts(
                    root, spec, split="all" if role == "fitting" else "query"
                ):
                    exposure.update(cells)
                    latent = model.encode(counts).cpu().numpy()
                    valid = cells.RNA_UMIs.to_numpy() > 0
                    if not np.isfinite(latent[valid]).all():
                        raise ContractError("Nonfinite encoded state.")
                    # Retain captured cells without inventing geometry for zero-RNA cells.
                    latent[~valid] = 0
                    states[offset : offset + len(cells)] = latent
                    metadata = cells[
                        ["source_id", "shard", "row_in_shard", "cell_id", "guide_index", "RNA_UMIs"]
                    ].copy()
                    metadata["latent_row"] = np.arange(offset, offset + len(cells), dtype=np.int64)
                    metadata["geometry_available"] = valid
                    table = pa.Table.from_pandas(metadata, preserve_index=False)
                    if metadata_writer is None:
                        metadata_writer = pq.ParquetWriter(path / "cells.parquet", table.schema)
                    metadata_writer.write_table(table)
                    for source, indices in cells.groupby("source_id", sort=False).indices.items():
                        groups = cells.guide_index.to_numpy()[indices]
                        np.add.at(support[str(source)][:, 0], groups, 1)
                        np.add.at(
                            support[str(source)][:, 1], groups, valid[indices].astype(np.int64)
                        )
                    offset += len(cells)
        finally:
            if metadata_writer is not None:
                metadata_writer.close()
        if offset != total:
            raise ContractError("Latent compilation omitted authorized cells.")
        summaries = []
        for source, counts in support.items():
            table = catalog.copy()
            table["source_id"] = source
            table["captured_cells"] = counts[:, 0]
            table["geometry_cells"] = counts[:, 1]
            table["geometry_absent_cells"] = counts[:, 0] - counts[:, 1]
            summaries.append(table)
        pd.concat(summaries, ignore_index=True).to_parquet(path / "support.parquet", index=False)
        # One compact all-cell integer index, not a full metadata scan per guide.
        sizes = np.concatenate([support[s][:, 1] for s in access.source_ids])
        offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
        coordinates = np.empty(int(offsets[-1]), dtype=np.int64)
        cursors = offsets[:-1].copy()
        source_indices = {s: i for i, s in enumerate(access.source_ids)}
        for batch in pq.ParquetFile(path / "cells.parquet").iter_batches(
            batch_size=spec.rules.batch_rows
        ):
            frame = batch.to_pandas()
            frame = frame[frame.geometry_available]
            keys = frame.source_id.map(source_indices).to_numpy(dtype=np.int64) * len(
                catalog
            ) + frame.guide_index.to_numpy(dtype=np.int64)
            for key in np.unique(keys):
                rows = frame.latent_row.to_numpy()[keys == key]
                stop = cursors[key] + len(rows)
                coordinates[cursors[key] : stop] = rows
                cursors[key] = stop
        if not np.array_equal(cursors, offsets[1:]):
            raise ContractError("Latent population indexing omitted geometry support.")
        with h5py.File(path / "latent.h5", "r+") as handle:
            handle.create_dataset("population_offsets", data=offsets)
            handle.create_dataset("population_rows", data=coordinates)
        return dict(
            role=role,
            exposure=exposure.record(),
            retained_individual_cells=total,
            geometry_cells=sum(int(a[:, 1].sum()) for a in support.values()),
            latent_dim=spec.rules.latent_dim,
            complete_catalog_categories=len(catalog),
            source_ids=list(access.source_ids),
            latent_payload_sha256=sha256_file(path / "latent.h5"),
            encoded_weight_updates=0,
            raw_count_payload_copied=False,
            law_scope="empirical_positive_RNA_cells; captured_zero_RNA_support_retained_separately",
            zero_RNA_coordinate_policy="zero_placeholder_geometry_available_false",
            representation_qualified=False,
            scientific_promotion=False,
        )

    before = fitted_record.facts["model_numerical_sha256"]
    runtime_check(spec)
    result = publish(
        destination,
        spec,
        stage="latent_cells",
        parents={"representation": fitted_record.identity(), "access": access.identity()},
        writer=writer,
    )
    if before != sha256_file(fitted / "model.safetensors"):
        raise ContractError("Encoding changed frozen representation weights.")
    return result


class EmpiricalLatentStore:
    """Read a guide law in bounded chunks; never replace it with its centroid.

    Every positive-RNA cell receives weight 1 / positive-RNA guide support.
    Empty geometry yields no samples, not NTC replacement. The complete support
    table separately retains captured cells with unavailable geometry and zeros.
    """

    def __init__(self, root: Path, *, representation_sha256: str) -> None:
        self.root = root
        self.record = verify(root, stage="latent_cells")
        if self.record.parents["representation"] != representation_sha256:
            raise ContractError("Latent store belongs to another representation.")
        self.support = pd.read_parquet(root / "support.parquet")
        self._signatures = self._stat()

    def _stat(self) -> tuple[tuple[int, ...], ...]:
        result = []
        for name in ("latent.h5", "cells.parquet", "support.parquet"):
            path = self.root / name
            if path.is_symlink():
                raise ContractError("Latent file alias forbidden.")
            info = path.stat()
            result.append(
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            )
        return tuple(result)

    def population(
        self, source_id: str, guide_index: int, *, batch_rows: int = 1024
    ) -> Iterator[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], pd.DataFrame]]:
        if type(batch_rows) is not int or not 0 < batch_rows <= 8192:
            raise ContractError("Invalid latent batch budget.")
        if self._stat() != self._signatures:
            raise ContractError("Verified latent files changed.")
        rows = self.support[
            self.support.source_id.eq(source_id) & self.support.guide_index.eq(guide_index)
        ]
        if len(rows) != 1:
            raise ContractError("Unknown source/guide population.")
        count = int(rows.geometry_cells.iloc[0])
        found = 0
        with h5py.File(self.root / "latent.h5", "r") as handle:
            source_index = self.record.facts["source_ids"].index(source_id)
            key = source_index * self.record.facts["complete_catalog_categories"] + guide_index
            left, right = map(int, handle["population_offsets"][key : key + 2])
            if right - left != count:
                raise ContractError("Latent index contradicts complete support counts.")
            for start in range(left, right, batch_rows):
                positions = handle["population_rows"][start : min(start + batch_rows, right)]
                states = handle["state"][positions]
                found += len(positions)
                selected = pd.DataFrame(
                    dict(latent_row=positions, source_id=source_id, guide_index=guide_index)
                )
                yield states, np.full(len(positions), 1 / count, dtype=np.float64), selected
        if found != count or self._stat() != self._signatures:
            raise ContractError("Latent population support or verified files changed.")

    def identity(self) -> str:
        return contract_id(
            dict(bundle=self.record.identity(), law="equal_positive_RNA_cell_empirical")
        )
