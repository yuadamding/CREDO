"""Metadata-only canonical access over immutable source HDF5 CSR matrices."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from ..canonical import canonical_json_bytes, sha256_file
from ..contracts import (
    G00SourcePlaneV2Amendment,
    SourcePlaneDerivationReceipt,
    VirtualCanonicalCountStoreManifest,
    VirtualCanonicalCountStoreManifestV2,
    VirtualCountSource,
    VirtualCountSourceV2,
)
from ..errors import ContractError, IntegrityError
from .csr import SparseCountBatch


def _permutation_hash(values: np.ndarray[Any, Any]) -> str:
    permutation = np.asarray(values, dtype="<i4")
    return hashlib.sha256(permutation.tobytes(order="C")).hexdigest()


def _int64_hash(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _source_row_pairs_hash(source_index: int, source_rows: np.ndarray[Any, Any]) -> str:
    digest = hashlib.sha256()
    rows = np.asarray(source_rows, dtype=np.int64)
    for start in range(0, len(rows), 1_000_000):
        chunk = rows[start : start + 1_000_000]
        pairs = np.empty((len(chunk), 2), dtype="<i8")
        pairs[:, 0] = source_index
        pairs[:, 1] = chunk
        digest.update(pairs.tobytes(order="C"))
    return digest.hexdigest()


def _csr_rows(
    path: Path, source: VirtualCountSourceV2 | VirtualCountSource, rows: np.ndarray[Any, Any]
) -> sparse.csr_matrix:
    requested = np.asarray(rows, dtype=np.int64)
    if np.any(requested < 0) or np.any(requested >= source.rows):
        raise KeyError(f"Source row is outside {source.source_id} bounds.")
    unique_rows, inverse = np.unique(requested, return_inverse=True)
    with h5py.File(path, "r") as handle:
        group = handle[source.dataset_path]
        indptr = group["indptr"]
        data = group["data"]
        indices = group["indices"]
        blocks: list[sparse.csr_matrix] = []
        run_breaks = np.where(np.diff(unique_rows) != 1)[0] + 1
        for run in np.split(unique_rows, run_breaks):
            first = int(run[0])
            last = int(run[-1]) + 1
            offsets = np.asarray(indptr[first : last + 1], dtype=np.int64)
            start, end = int(offsets[0]), int(offsets[-1])
            blocks.append(
                sparse.csr_matrix(
                    (
                        data[start:end],
                        indices[start:end],
                        offsets - start,
                    ),
                    shape=(last - first, source.features),
                )
            )
    unique_matrix = sparse.vstack(blocks, format="csr")
    return cast(sparse.csr_matrix, unique_matrix[inverse].tocsr())


class VirtualCanonicalCountStore:
    """Forensic/sequential source plane; never the direct H100 training backend."""

    def __init__(self, path: Path, *, source_root: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise IntegrityError(f"Virtual canonical store is not a regular directory: {path}.")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise IntegrityError("Virtual canonical store lacks manifest.json.")
        self.path = path
        self.source_root = source_root
        payload = manifest_path.read_text()
        version = int(json.loads(payload).get("schema_version", 0))
        self.manifest: VirtualCanonicalCountStoreManifest | VirtualCanonicalCountStoreManifestV2
        if version == 1:
            self.manifest = VirtualCanonicalCountStoreManifest.model_validate_json(payload)
        elif version == 2:
            self.manifest = VirtualCanonicalCountStoreManifestV2.model_validate_json(payload)
        else:
            raise IntegrityError(f"Unsupported virtual canonical schema version: {version}.")
        self._locator_cache: (
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]] | None
        ) = None
        self._permutations: tuple[np.ndarray[Any, Any], ...] | None = None

    @property
    def locator_cache_bytes(self) -> int:
        if self._locator_cache is None:
            return 0
        return sum(array.nbytes for array in self._locator_cache)

    def _sources(self) -> tuple[VirtualCountSource | VirtualCountSourceV2, ...]:
        return cast(tuple[VirtualCountSource | VirtualCountSourceV2, ...], self.manifest.sources)

    def _locator(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if self._locator_cache is None:
            locator_path = self.path / self.manifest.row_locator.relative_uri
            with h5py.File(locator_path, "r") as handle:
                self._locator_cache = (
                    np.asarray(handle["row_ids_sorted"][:], dtype=np.int64),
                    np.asarray(handle["source_indices_sorted"][:], dtype=np.int16),
                    np.asarray(handle["source_rows_sorted"][:], dtype=np.int64),
                )
        return self._locator_cache

    def _locator_catalog_hashes(self) -> tuple[str, str]:
        locator_path = self.path / self.manifest.row_locator.relative_uri
        with h5py.File(locator_path, "r") as handle:
            required = {
                "guide_codes_sorted",
                "target_codes_sorted",
                "guide_ids",
                "target_ids",
            }
            if not required <= set(handle):
                raise IntegrityError("Virtual row locator lacks guide/target identity datasets.")
            guide_ids = handle["guide_ids"].asstr()[:].tolist()
            target_ids = handle["target_ids"].asstr()[:].tolist()
            guide_codes = np.asarray(handle["guide_codes_sorted"][:], dtype=np.int32)
            target_codes = np.asarray(handle["target_codes_sorted"][:], dtype=np.int32)
            if (
                len(guide_codes) != self.manifest.eligible_rows
                or len(target_codes) != self.manifest.eligible_rows
                or np.any(guide_codes < 0)
                or np.any(guide_codes >= len(guide_ids))
                or np.any(target_codes < 0)
                or np.any(target_codes >= len(target_ids))
            ):
                raise IntegrityError("Virtual row locator guide/target codes are invalid.")
        guide_hash = hashlib.sha256(canonical_json_bytes(guide_ids)).hexdigest()
        target_hash = hashlib.sha256(canonical_json_bytes(target_ids)).hexdigest()
        return guide_hash, target_hash

    def _verify_dev30_authority_artifacts(
        self,
        row_ids: np.ndarray[Any, Any],
        source_indices: np.ndarray[Any, Any],
        source_rows: np.ndarray[Any, Any],
    ) -> None:
        if not isinstance(self.manifest, VirtualCanonicalCountStoreManifestV2):
            return
        crosswalk_path = self.path / self.manifest.guide_target_crosswalk.relative_uri
        numeric_path = self.path / self.manifest.source_numeric_audit.relative_uri
        derivation_path = self.path / self.manifest.source_derivation_receipt.relative_uri
        amendment_path = self.path / self.manifest.source_plane_amendment.relative_uri
        if (
            sha256_file(crosswalk_path) != self.manifest.guide_target_crosswalk.sha256
            or sha256_file(numeric_path) != self.manifest.source_numeric_audit.sha256
            or sha256_file(derivation_path) != self.manifest.source_derivation_receipt.sha256
            or sha256_file(amendment_path) != self.manifest.source_plane_amendment.sha256
        ):
            raise IntegrityError("Virtual canonical authority-artifact hash mismatch.")
        amendment = G00SourcePlaneV2Amendment.model_validate_json(amendment_path.read_text())
        if amendment.amendment_id != self.manifest.source_plane_amendment_id:
            raise IntegrityError("Virtual canonical amendment ID differs from its manifest.")
        amendment_wiring = (
            (amendment.v2_guide_target_crosswalk, self.manifest.guide_target_crosswalk),
            (amendment.v2_numerical_audit, self.manifest.source_numeric_audit),
            (
                amendment.v2_source_derivation_receipt,
                self.manifest.source_derivation_receipt,
            ),
            (amendment.v2_row_locator, self.manifest.row_locator),
        )
        if any(expected != observed for expected, observed in amendment_wiring):
            raise IntegrityError("Virtual canonical amendment artifact wiring differs.")
        crosswalk = pd.read_parquet(crosswalk_path)
        expected_column_order = (
            "guide_id",
            "target_id",
            "is_control",
            "raw_guide_group",
            "eligible_cell_count",
            "observed_source_count",
        )
        if tuple(crosswalk.columns) != expected_column_order:
            raise IntegrityError("Guide-target crosswalk has an unexpected schema.")
        if (
            len(crosswalk) != self.manifest.guide_count
            or crosswalk["guide_id"].duplicated().any()
            or crosswalk["guide_id"].isna().any()
            or crosswalk["target_id"].isna().any()
            or (crosswalk["eligible_cell_count"] <= 0).any()
            or (crosswalk["observed_source_count"] <= 0).any()
            or crosswalk["raw_guide_group"].astype(str).str.contains("multi_sgRNA").any()
            or crosswalk["target_id"].nunique() != self.manifest.target_control_count
            or not pd.api.types.is_bool_dtype(crosswalk["is_control"])
            or not pd.api.types.is_integer_dtype(crosswalk["eligible_cell_count"])
            or not pd.api.types.is_integer_dtype(crosswalk["observed_source_count"])
        ):
            raise IntegrityError("Guide-target crosswalk invariants failed.")
        control_targets = set(crosswalk.loc[crosswalk["is_control"], "target_id"].astype(str))
        targeting_targets = set(crosswalk.loc[~crosswalk["is_control"], "target_id"].astype(str))
        if control_targets & targeting_targets:
            raise IntegrityError("Control and targeting crosswalk target sets overlap.")
        locator_path = self.path / self.manifest.row_locator.relative_uri
        with h5py.File(locator_path, "r") as handle:
            guide_ids = handle["guide_ids"].asstr()[:]
            target_ids = handle["target_ids"].asstr()[:]
            guide_codes = np.asarray(handle["guide_codes_sorted"][:], dtype=np.int64)
            target_codes = np.asarray(handle["target_codes_sorted"][:], dtype=np.int64)
        mapping = crosswalk.set_index("guide_id")["target_id"].to_dict()
        if set(mapping) != set(guide_ids.tolist()):
            raise IntegrityError("Crosswalk and row-locator guide catalogs differ.")
        mapped_targets = np.asarray([mapping[str(guide_ids[code])] for code in guide_codes])
        locator_targets = target_ids[target_codes]
        if not np.array_equal(mapped_targets, locator_targets):
            raise IntegrityError("Eligible row guide-target assignments violate the crosswalk.")
        if set(crosswalk["target_id"].astype(str)) != set(target_ids.tolist()):
            raise IntegrityError("Crosswalk and row-locator target catalogs differ.")
        eligible_counts = np.bincount(guide_codes, minlength=len(guide_ids))
        source_guide_pairs = np.unique(
            guide_codes.astype(np.int64) * len(self.manifest.sources)
            + source_indices.astype(np.int64)
        )
        observed_source_counts = np.bincount(
            source_guide_pairs // len(self.manifest.sources), minlength=len(guide_ids)
        )
        crosswalk_by_guide = crosswalk.set_index("guide_id")
        if not np.array_equal(
            eligible_counts,
            crosswalk_by_guide.loc[guide_ids, "eligible_cell_count"].to_numpy(dtype=np.int64),
        ) or not np.array_equal(
            observed_source_counts,
            crosswalk_by_guide.loc[guide_ids, "observed_source_count"].to_numpy(dtype=np.int64),
        ):
            raise IntegrityError("Crosswalk support counts differ from the row locator.")
        numeric = pd.read_parquet(numeric_path)
        numeric_columns = {
            "source_id",
            "matrix_encoding",
            "storage_value_dtype",
            "indices_dtype",
            "indptr_dtype",
            "counts_nonnegative_verified",
            "counts_integral_verified",
            "counts_finite_verified",
            "maximum_observed_count",
            "csr_indices_in_bounds_verified",
            "csr_indptr_monotonic_verified",
            "csr_terminal_offset_matches_nnz",
        }
        if set(numeric.columns) != numeric_columns or numeric["source_id"].duplicated().any():
            raise IntegrityError("Source numeric audit has an unexpected schema.")
        observed = {
            str(row.source_id): {key: getattr(row, key) for key in numeric_columns - {"source_id"}}
            for row in numeric.itertuples(index=False)
        }
        expected = {
            source.source_id: source.numeric_integrity.model_dump(mode="python")
            for source in self.manifest.sources
        }
        if observed != expected:
            raise IntegrityError("Source numeric audit differs from source records.")
        derivation = SourcePlaneDerivationReceipt.model_validate_json(derivation_path.read_text())
        records = {record.source_id: record for record in derivation.records}
        if set(records) != {source.source_id for source in self.manifest.sources}:
            raise IntegrityError("Source derivation receipt has a different source catalog.")
        for source_index, source in enumerate(self.manifest.sources):
            selected = source_indices == source_index
            selected_rows = source_rows[selected]
            selected_ids = row_ids[selected]
            record = records[source.source_id]
            if (
                record.selected_row_count != int(selected.sum())
                or record.selected_row_count != source.eligible_rows
                or record.selected_nnz != source.eligible_nnz
                or record.selected_row_ids_hash != _int64_hash(selected_ids)
                or record.source_row_pairs_hash
                != _source_row_pairs_hash(source_index, selected_rows)
                or record.source_file_sha256 != source.source_file_sha256
            ):
                raise IntegrityError(f"Source derivation receipt differs for {source.source_id}.")

    def _feature_permutations(self) -> tuple[np.ndarray[Any, Any], ...]:
        if self._permutations is None:
            path = self.path / self.manifest.feature_permutations.relative_uri
            with np.load(path, allow_pickle=False) as archive:
                values = tuple(
                    np.asarray(archive[f"source_{index:06d}"], dtype=np.int32)
                    for index in range(len(self.manifest.sources))
                )
            for source, permutation in zip(self._sources(), values, strict=True):
                if (
                    len(permutation) != self.manifest.features
                    or len(np.unique(permutation)) != len(permutation)
                    or np.any(permutation < 0)
                    or np.any(permutation >= source.features)
                    or _permutation_hash(permutation) != source.canonical_permutation_hash
                ):
                    raise IntegrityError(f"Invalid canonical permutation for {source.source_id}.")
            self._permutations = values
        return self._permutations

    def verify(
        self, *, full: bool = True
    ) -> VirtualCanonicalCountStoreManifestV2 | VirtualCanonicalCountStoreManifest:
        locator_path = self.path / self.manifest.row_locator.relative_uri
        permutation_path = self.path / self.manifest.feature_permutations.relative_uri
        if (
            sha256_file(locator_path) != self.manifest.row_locator.sha256
            or sha256_file(permutation_path) != self.manifest.feature_permutations.sha256
        ):
            raise IntegrityError("Virtual canonical metadata artifact hash mismatch.")
        row_ids, source_indices, source_rows = self._locator()
        if (
            len(row_ids) != self.manifest.eligible_rows
            or len(source_indices) != len(row_ids)
            or len(source_rows) != len(row_ids)
            or np.any(np.diff(row_ids) <= 0)
            or np.any(source_indices < 0)
            or np.any(source_indices >= len(self.manifest.sources))
        ):
            raise IntegrityError("Virtual canonical row locator is invalid.")
        if isinstance(self.manifest, VirtualCanonicalCountStoreManifestV2):
            if _int64_hash(row_ids) != self.manifest.eligible_row_ids_hash:
                raise IntegrityError("Virtual canonical eligible-row hash mismatch.")
        for index, source in enumerate(self._sources()):
            selected = source_indices == index
            selected_rows = source_rows[selected]
            if int(selected.sum()) != source.eligible_rows:
                raise IntegrityError(f"Virtual source row count differs for {source.source_id}.")
            sorted_source_rows = np.sort(selected_rows, kind="stable")
            if len(sorted_source_rows) > 1 and np.any(np.diff(sorted_source_rows) == 0):
                raise IntegrityError(f"Virtual source rows are duplicated for {source.source_id}.")
            if np.any(source_rows[selected] < 0) or np.any(source_rows[selected] >= source.rows):
                raise IntegrityError(f"Virtual source rows exceed {source.source_id} bounds.")
            source_path = self.source_root / source.relative_uri
            if not source_path.is_file():
                raise IntegrityError(f"Virtual source is missing: {source.source_id}.")
            if full and sha256_file(source_path) != source.source_file_sha256:
                raise IntegrityError(f"Virtual source hash mismatch: {source.source_id}.")
        guide_hash, target_hash = self._locator_catalog_hashes()
        if (
            guide_hash != self.manifest.guide_catalog_hash
            or target_hash != self.manifest.target_catalog_hash
        ):
            raise IntegrityError("Virtual row locator guide/target catalog hash mismatch.")
        self._verify_dev30_authority_artifacts(row_ids, source_indices, source_rows)
        self._feature_permutations()
        return self.manifest

    def rows(self, row_ids: np.ndarray[Any, Any]) -> SparseCountBatch:
        requested = np.asarray(row_ids, dtype=np.int64)
        if not len(requested):
            return SparseCountBatch(
                matrix=sparse.csr_matrix((0, self.manifest.features), dtype=np.int32),
                row_ids=requested.copy(),
                feature_index_hash=self.manifest.canonical_feature_index_hash,
            )
        sorted_ids, locator_sources, locator_rows = self._locator()
        found = np.searchsorted(sorted_ids, requested)
        valid = found < len(sorted_ids)
        candidates = np.where(valid)[0]
        valid[candidates] = sorted_ids[found[candidates]] == requested[candidates]
        if not np.all(valid):
            raise KeyError(f"Unknown virtual row IDs: {requested[~valid][:10].tolist()}.")
        source_indices = locator_sources[found]
        source_rows = locator_rows[found]
        permutations = self._feature_permutations()
        blocks: list[sparse.csr_matrix] = []
        output_positions: list[np.ndarray[Any, Any]] = []
        for source_index in np.unique(source_indices):
            positions = np.where(source_indices == source_index)[0]
            source = self.manifest.sources[int(source_index)]
            source_path = self.source_root / source.relative_uri
            matrix = _csr_rows(source_path, source, source_rows[positions])
            blocks.append(matrix[:, permutations[int(source_index)]].tocsr())
            output_positions.append(positions)
        stacked = sparse.vstack(blocks, format="csr")
        block_order = np.concatenate(output_positions)
        output = stacked[np.argsort(block_order, kind="stable")].tocsr()
        if output.shape != (len(requested), self.manifest.features):
            raise ContractError("Virtual canonical read returned an invalid shape.")
        return SparseCountBatch(
            matrix=output,
            row_ids=requested.copy(),
            feature_index_hash=self.manifest.canonical_feature_index_hash,
        )

    def iter_batches(
        self, ordered_row_ids: np.ndarray[Any, Any], *, batch_size: int, cursor: int = 0
    ) -> Iterator[tuple[int, SparseCountBatch]]:
        if batch_size <= 0 or cursor < 0:
            raise ValueError("batch_size must be positive and cursor nonnegative.")
        ids = np.asarray(ordered_row_ids, dtype=np.int64)
        while cursor < len(ids):
            end = min(cursor + batch_size, len(ids))
            yield end, self.rows(ids[cursor:end])
            cursor = end
