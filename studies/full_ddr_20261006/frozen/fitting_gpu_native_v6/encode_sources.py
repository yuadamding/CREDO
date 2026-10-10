"""New phase-scoped encoder publication; reuse the frozen R48 forward model.

The original count preparation and representation specifications are immutable.
This adapter consumes a NEW source allowlist; it never edits their old roles.
Output retains individual cells and observed sampling zeros, not centroids.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from prepare_manifest import canonical


class EncodingError(ValueError):
    pass


EARLY_SOURCES = frozenset({"D3_Stim8hr", "D4_Stim8hr", "D2_Stim8hr", "D2_Stim48hr"})
SOURCE_INFERENCE = frozenset({"D1_Stim8hr"})
FINAL_SOURCES = frozenset(f"D{d}_{c}" for d in range(1, 5)
                          for c in ("Rest", "Stim8hr", "Stim48hr"))


def authorize_sources(sources: tuple[str, ...], phase: str, release: dict | None = None) -> None:
    """Check the entire request before opening any input, including payload hashes."""
    if not sources or len(set(sources)) != len(sources):
        raise EncodingError("Unique nonempty sources required.")
    allowed = EARLY_SOURCES if phase == "development" else (
        SOURCE_INFERENCE if phase == "forecast_source" else set())
    if phase in {"assessment", "final_refit"}:
        required = {"selected_architecture_sha256", "terminal_checkpoints_sha256",
                    "D1_forecast_seal_sha256"}
        if phase == "final_refit":
            required.add("D1_assessment_complete_sha256")
        if release is None or not required <= set(release):
            raise EncodingError("Bound forecast release required before endpoint access.")
        for name in required:
            value = release[name]
            if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise EncodingError("Invalid release artifact identity.")
        # The caller must verify these artifact bytes/cross-bindings; a supplied
        # hash string alone is not custody evidence.
        if release.get("verified_by_lifecycle") is not True:
            raise EncodingError("Release identities have not been verified by lifecycle.")
        allowed = {"D1_Stim48hr"} if phase == "assessment" else FINAL_SOURCES
    if not set(sources) <= allowed:
        raise EncodingError("Source request crosses phase boundary.")


def sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EncodingError("Expected regular non-symlink artifact.")
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            value.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise EncodingError("Artifact changed during identity read.")
    return value.hexdigest()


def publish_json(path: Path, value: dict) -> None:
    raw = canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise EncodingError("Published JSON readback mismatch.")


def export_source(
    destination: Path, *, source_id: str, phase: str, expected_rows: int,
    guides: pd.DataFrame, latent_dim: int, rna_features: int,
    blocks: Iterable[tuple[sparse.csr_matrix, pd.DataFrame]],
    encode: Callable[[sparse.csr_matrix], np.ndarray],
    lineage: dict, release: dict | None = None, maximum_output_bytes: int = 2 * 1024**3,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Stream one all-cell source into v4-layout latent support and a new receipt.

    `blocks` must be the explicitly bound PreparedShardReader stream, including
    validated source_id/shard/row_in_shard/cell_id/guide_index/RNA_UMIs. The
    production caller owns full input hash validation and GPU/model admission.
    This writer intentionally has no retry or partial-output promotion path.
    """
    authorize_sources((source_id,), phase, release)
    if type(expected_rows) is not int or expected_rows <= 0:
        raise EncodingError("Positive expected row count required.")
    if latent_dim < 1 or rna_features < 1 or not len(guides):
        raise EncodingError("Positive axes required.")
    required = {"guide_index", "guide_id", "target_id", "is_control"}
    if set(guides) != required or not np.array_equal(guides.guide_index, np.arange(len(guides))):
        raise EncodingError("Canonical complete guide catalog required.")
    if not guides.guide_id.is_unique or guides.is_control.dtype != bool:
        raise EncodingError("Guide catalog identity/control flags invalid.")
    if expected_rows * (latent_dim * 4 + 128) > maximum_output_bytes:
        raise EncodingError("Per-source publication estimate exceeds byte budget.")
    # No existing destination is merged or overwritten, including failed output.
    destination.mkdir(mode=0o700)
    required_lineage = {"input_access_sha256", "representation_sha256", "model_sha256",
                        "scaler_sha256", "gene_axis_sha256", "implementation_sha256"}
    if not required_lineage <= set(lineage):
        raise EncodingError("Complete input/representation/source lineage required.")
    for key in required_lineage:
        value = lineage[key]
        if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
            raise EncodingError("Invalid lineage digest.")

    n_guides = len(guides)
    counts = np.zeros(n_guides, dtype=np.int64)
    geometry = np.zeros(n_guides, dtype=np.int64)
    group_index = np.full(expected_rows, -1, dtype=np.int32)
    valid_index = np.zeros(expected_rows, dtype=bool)
    row_hash = hashlib.sha256()
    writer = None
    seen = 0
    # Native addresses must advance in source/shard order; a duplicate cell ID
    # is additionally rejected across all batches, not only within one batch.
    previous_address = (-1, -1)
    seen_cell_ids: set[str] = set()
    try:
        with h5py.File(destination / "latent.h5", "x") as handle:
            states = handle.create_dataset("state", (expected_rows, latent_dim), dtype="float32",
                                           chunks=(min(expected_rows, 2048), latent_dim))
            for matrix, original in blocks:
                cells = original.copy()
                n = len(cells)
                fields = {"source_id", "shard", "row_in_shard", "cell_id", "guide_index", "RNA_UMIs"}
                if not fields <= set(cells) or not n or seen + n > expected_rows:
                    raise EncodingError("Metadata axes or source row count invalid.")
                if not cells.source_id.eq(source_id).all():
                    raise EncodingError("Source mismatch before encoder access.")
                if not sparse.isspmatrix_csr(matrix) or matrix.shape != (n, rna_features):
                    raise EncodingError("Canonical RNA CSR dimensions required.")
                if matrix.dtype.kind not in "iu" or (matrix.data < 0).any():
                    raise EncodingError("Raw integer nonnegative counts required.")
                matrix.check_format(full_check=True)
                if not matrix.has_canonical_format:
                    raise EncodingError("RNA matrix columns must already be canonical.")
                if cells.guide_index.dtype.kind not in "iu":
                    raise EncodingError("Integer guide indices required.")
                group = cells.guide_index.to_numpy(dtype=np.int64)
                if np.any(group < 0) or np.any(group >= n_guides):
                    raise EncodingError("Guide index outside complete catalog.")
                depth = np.asarray(matrix.sum(axis=1), dtype=np.int64).ravel()
                if not np.array_equal(depth, cells.RNA_UMIs.to_numpy()):
                    raise EncodingError("Declared RNA depth differs from counts.")
                for row in cells.itertuples(index=False):
                    if not isinstance(row.cell_id, str) or not row.cell_id or row.cell_id in seen_cell_ids:
                        raise EncodingError("Missing or duplicate canonical cell identity.")
                    seen_cell_ids.add(row.cell_id)
                    address = (int(row.shard), int(row.row_in_shard))
                    if min(address) < 0 or address <= previous_address:
                        raise EncodingError("Input addresses must advance without duplicates.")
                    previous_address = address
                    row_hash.update(canonical([source_id, *address, row.cell_id, int(row.guide_index)]))
                valid = depth > 0
                encoded = np.asarray(encode(matrix))
                if encoded.shape != (n, latent_dim) or encoded.dtype != np.float32:
                    raise EncodingError("Frozen encoder must return aligned float32 states.")
                if not np.isfinite(encoded[valid]).all():
                    raise EncodingError("Nonfinite available geometry.")
                encoded = encoded.copy()
                encoded[~valid] = 0  # placeholder only, explicitly unavailable
                states[seen:seen + n] = encoded
                cells = cells[list(sorted(fields))]
                cells["latent_row"] = np.arange(seen, seen + n, dtype=np.int64)
                cells["geometry_available"] = valid
                table = pa.Table.from_pandas(cells, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination / "cells.parquet", table.schema)
                writer.write_table(table)
                counts += np.bincount(group, minlength=n_guides)
                geometry += np.bincount(group[valid], minlength=n_guides)
                group_index[seen:seen + n] = group
                valid_index[seen:seen + n] = valid
                seen += n
                if progress:
                    progress(dict(source_id=source_id, completed_rows=seen, planned_rows=expected_rows,
                                  optimizer_updates=0, encoded_geometry_cells=int(geometry.sum())))
            if seen != expected_rows:
                raise EncodingError("Stream omitted authorized cells.")
            offsets = np.r_[0, np.cumsum(geometry, dtype=np.int64)]
            rows = np.flatnonzero(valid_index)
            rows = rows[np.argsort(group_index[rows], kind="stable")]
            handle.create_dataset("population_offsets", data=offsets)
            handle.create_dataset("population_rows", data=rows)
            handle.flush()
    finally:
        if writer is not None:
            writer.close()
    support = guides.copy()
    support["source_id"] = source_id
    support["captured_cells"] = counts
    support["geometry_cells"] = geometry
    support["geometry_absent_cells"] = counts - geometry
    support.to_parquet(destination / "support.parquet", index=False)
    files = {name: {"sha256": sha_file(destination / name), "bytes": (destination / name).stat().st_size}
             for name in ("latent.h5", "cells.parquet", "support.parquet")}
    if sum(item["bytes"] for item in files.values()) > maximum_output_bytes:
        raise EncodingError("Published source exceeds output byte cap; not complete.")
    result = dict(schema="credo.all4_ddr.latent_source.v1", status="encoded_frozen_representation",
                  source_ids=[source_id], phase=phase, expected_rows=expected_rows,
                  geometry_cells=int(geometry.sum()), complete_catalog_categories=n_guides,
                  latent_dim=latent_dim, files=files, lineage=lineage,
                  ordered_cell_address_sha256=row_hash.hexdigest(), optimizer_updates=0,
                  zero_RNA_coordinate_policy="zero_placeholder_geometry_available_false",
                  count_vector_scope="all_captured_rows_same_encoding_pass",
                  raw_counts_modified=False, raw_counts_copied=False, training_complete=False)
    publish_json(destination / "COMPLETE.json", result)
    return result
