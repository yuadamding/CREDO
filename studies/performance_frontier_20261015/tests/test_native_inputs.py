"""Tiny fabricated prepared packages; no real count/latent reads or optimizer calls."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from frontier.native_inputs import (FoldInputs, ForecastExpectation, NativeInputIndex,
                                    PackageBinding, VerifiedForecastRelease, _load_reader)
from frontier.observation import ObservationFold, technical_holdout


WORKSPACE = Path(__file__).resolve().parents[2]
READER = WORKSPACE / "credo_full_cohort_20260912/r48_full_pipeline_20260914_a2/bundle/repo/src/credo_count_sde_v4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def native_package(tmp_path, *, temporal=False):
    root = tmp_path / "data"
    metadata = tmp_path / "metadata"
    (root / "source").mkdir(parents=True)
    (metadata / "source").mkdir(parents=True)
    guides = pd.DataFrame({"sgRNA": ["g0", "g1", "g2"], "primary_target_id": ["T0", "NTC", "T2"],
                           "is_design_ntc": [False, True, False], "experimental_library_member": [True] * 3})
    guides.to_parquet(root / "source/guides.parquet", index=False)
    samples, features, shards = [], [], []
    for donor in ("D1", "D2", "D3", "D4"):
        for condition in ("Rest", "Stim8hr", "Stim48hr"):
            source = f"{donor}_{condition}"
            samples.append({"source_id": source, "donor_id": donor, "condition": condition})
            # Technical feature at position zero: RNA extraction must be indexed, not a prefix.
            for index, feature in enumerate(("PuroR", "ENSG1", "ENSG2", "ENSG3")):
                features.append({"source_id": source, "source_position": 3 - index,
                                 "canonical_position": index, "feature_id": feature, "is_RNA": index > 0})
            matrix = sparse.csr_matrix(np.asarray([[9, 1, 0, 2], [7, 0, 0, 0], [2, 0, 3, 0]], dtype=np.uint32))
            if temporal:
                # Distinct fabricated early/late distributions for normalizer
                # integration, not an alteration of the original reader fixture.
                rows = ([[9, 0, 1, 2], [7, 2, 0, 1], [2, 2, 1, 0]] if condition == "Stim48hr"
                        else [[9, 1, 0, 2], [7, 2, 1, 0], [2, 0, 3, 0]])
                matrix = sparse.csr_matrix(np.asarray(rows, dtype=np.uint32))
            counts_name, cells_name = f"source/{source}/counts/00000.csr.npz", f"source/{source}/cells/00000.parquet"
            (root / counts_name).parent.mkdir(parents=True)
            (root / cells_name).parent.mkdir(parents=True)
            sparse.save_npz(root / counts_name, matrix)
            cells = pd.DataFrame({"source_id": [source] * 3, "row_in_shard": np.arange(3),
                                  "cell_id": ["c0", "c1", "c2"], "guide_index": [0, 0, 1],
                                  "guide_id": ["g0", "g0", "g1"], "primary_target_id": ["T0", "T0", "NTC"],
                                  "is_control": [False, False, True], "RNA_UMIs": np.array([3, 0, 3], dtype=np.uint64),
                                  "donor_id": [donor] * 3, "condition": [condition] * 3})
            if temporal:
                cells["RNA_UMIs"] = np.array([3, 3, 3], dtype=np.uint64)
            cells.to_parquet(root / cells_name, index=False)
            shards.append({"source_id": source, "shard": 0, "rows": 3, "nnz": matrix.nnz,
                           "path": counts_name, "sha256": digest(root / counts_name),
                           "bytes": (root / counts_name).stat().st_size, "cells_path": cells_name,
                           "cells_sha256": digest(root / cells_name), "cell_table_bytes": (root / cells_name).stat().st_size,
                           "exact_reload": True})
            accounting = pd.DataFrame({"source_id": [source] * 3, "guide_index": [0, 1, 2],
                                       "guide_id": ["g0", "g1", "g2"], "observed": [True] * 3,
                                       "n_cells": [2, 1, 0], "captured_cell_denominator": [3] * 3,
                                       "guide_universe_size": [3] * 3, "jeffreys_frequency": np.array([2.5, 1.5, 0.5]) / 4.5})
            accounting.to_parquet(root / f"source/{source}/population_accounting.parquet", index=False)
    for name, records in (("samples_and_culture_branches", samples), ("features_and_source_positions", features),
                          ("shard_manifest", shards)):
        pd.DataFrame(records).to_parquet(root / f"source/{name}.parquet", index=False)
    members = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in members))
    write_json(root / "INPUT_DATA_COMPLETE.json", {"inventory_sha256": digest(root / "SHA256SUMS"),
                                                 "RNA_features": 3, "technical_features": 1})
    for name in ("INPUT_DATA_COMPLETE.json", "SHA256SUMS", "source/guides.parquet",
                 "source/samples_and_culture_branches.parquet", "source/features_and_source_positions.parquet",
                 "source/shard_manifest.parquet"):
        (metadata / name).write_bytes((root / name).read_bytes())
    binding = PackageBinding(digest(root / "INPUT_DATA_COMPLETE.json"), digest(root / "SHA256SUMS"), 3, 3, True)
    index = NativeInputIndex(metadata, READER, binding=binding)
    return root, metadata, binding, index


@pytest.fixture
def fixture(tmp_path):
    return native_package(tmp_path)


def test_metadata_only_index_and_all_ten_folds(fixture, monkeypatch):
    from itertools import combinations
    root, metadata, binding, index = fixture
    assert not list(metadata.rglob("*.npz")) and not list(metadata.rglob("cells"))
    assert index.rna_positions == (1, 2, 3)
    assert index.rna_feature_ids == ("ENSG1", "ENSG2", "ENSG3")
    assert len(index.shards) == 8 and all("Rest" not in row.source_id for row in index.shards)
    count = 0
    for size in (2, 3):
        for donors in combinations(("D1", "D2", "D3", "D4"), size):
            fold = FoldInputs("-".join(donors), donors)
            fitting = index.access(fold, "representation_fit")
            assert len(fitting.source_ids) == 2 * size
            assert fitting.n_features == 4 and fitting.storage_isolation_qualified is False
            assert all(index.source_roles[source][0] in donors for source in fitting.source_ids)
            assert all(row.allowed_rows is None for row in fitting.shards)
            for donor in set(("D1", "D2", "D3", "D4")) - set(donors):
                query = index.access(fold, "query", query_donor=donor)
                assert query.source_ids == (f"{donor}_Stim8hr",)
            count += 1
    assert count == 10
    with pytest.raises(ValueError, match="subset"):
        FoldInputs("all", ("D1", "D2", "D3", "D4"))
    with pytest.raises(ValueError, match="excluded"):
        index.access(FoldInputs("pair", ("D3", "D4")), "query", query_donor="D3")
    with pytest.raises(ValueError, match="Unknown"):
        index.access(FoldInputs("pair", ("D3", "D4")), "Rest")


def test_sparse_rows_rna_zero_capture_and_source_scoped_split_ids(fixture, monkeypatch):
    root, _, _, index = fixture
    fold = FoldInputs("D3-D4", ("D3", "D4"))
    handle = index.open(root, fold, "dynamics_source", max_batch_rows=3)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", lambda *a, **k: pytest.fail("cohort/minibatch densification"))
    batch = handle.read_rows([("D3_Stim8hr", 0, 0), ("D3_Stim8hr", 0, 1), ("D4_Stim8hr", 0, 0)])
    assert sparse.isspmatrix_csr(batch.rna_counts) and batch.rna_counts.shape == (3, 3)
    assert batch.rna_depth.tolist() == [3, 0, 3] and batch.positive_rna.tolist() == [True, False, True]
    assert batch.rna_counts[0, 0] == 1 and batch.rna_counts[0, 2] == 2
    assert batch.row_ids[0] != batch.row_ids[2]
    observed = ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256)
    split = technical_holdout(batch.row_ids, fold=observed)
    reordered = handle.read_rows([("D4_Stim8hr", 0, 0), ("D3_Stim8hr", 0, 1), ("D3_Stim8hr", 0, 0)])
    assert np.array_equal(split[::-1], technical_holdout(reordered.row_ids, fold=observed))
    captured = handle.captured_counts("D3_Stim8hr")
    support = handle.support_counts("D3_Stim8hr")
    assert captured.n_cells.tolist() == support.captured_cells.tolist() == [2, 1, 0]
    assert support.positive_RNA_cells.tolist() == [1, 1, 0]
    assert captured.jeffreys_frequency.iloc[2] > 0  # zero category retained
    for forbidden in ("D1_Stim8hr", "D3_Stim48hr", "D3_Rest"):
        with pytest.raises(ValueError, match="outside"):
            handle.captured_counts(forbidden)
        with pytest.raises(Exception, match="Unauthorized"):
            handle.read_rows([(forbidden, 0, 0)])


def test_query_cannot_open_late_and_request_is_prevalidated(fixture, monkeypatch):
    root, _, _, index = fixture
    handle = index.open(root, FoldInputs("pair", ("D3", "D4")), "query", query_donor="D1")
    monkeypatch.setattr(handle._reader, "_load", lambda *a: pytest.fail("opened count before complete authorization"))
    with pytest.raises(Exception, match="Unauthorized"):
        handle.read_rows([("D1_Stim8hr", 0, 0), ("D1_Stim48hr", 0, 0)])
    with pytest.raises(TypeError, match="verified complete"):
        index.access(FoldInputs("pair", ("D3", "D4")), "evaluation_truth", query_donor="D1", release=True)


def publish_seal(root, expectation):
    publication = root / "forecasts"
    publication.mkdir()
    (publication / "bank.bin").write_bytes(b"fabricated forecast payload")
    record = {"schema": "credo.frontier.forecast-seal.v1", "status": "complete", "bindings": expectation.record(),
              "forecasts": [{"forecast_id": expectation.forecast_ids[0], "artifacts": [{"relative_uri": "bank.bin",
                            "sha256": digest(publication / "bank.bin"), "size_bytes": (publication / "bank.bin").stat().st_size}]}]}
    write_json(publication / "seal.json", record)
    return publication, record


def expectation(index, fold):
    query = index.access(fold, "query", query_donor="D1")
    return ForecastExpectation(fold.identity, query.identity(), ("a" * 64,), "b" * 64, "c" * 64,
                               "d" * 64, ("model-a/seed0/D1/prefix4096/all-channels",))


def test_full_forecast_publication_releases_only_bound_query_truth(fixture):
    root, _, _, index = fixture
    fold = FoldInputs("pair", ("D3", "D4"))
    wanted = expectation(index, fold)
    publication, _ = publish_seal(root, wanted)
    release = VerifiedForecastRelease.verify(publication, "seal.json", manifest_sha256=digest(publication / "seal.json"), expectation=wanted)
    with pytest.raises(FrozenInstanceError):
        release.manifest_sha256 = "f" * 64
    truth = index.open(root, fold, "evaluation_truth", query_donor="D1", release=release)
    assert truth.access.source_ids == ("D1_Stim48hr",)
    assert truth.read_rows([("D1_Stim48hr", 0, 0)]).rna_depth.tolist() == [3]
    with pytest.raises(ValueError, match="different fold/query"):
        index.open(root, fold, "evaluation_truth", query_donor="D2", release=release)
    with pytest.raises(ValueError, match="changed bindings"):
        VerifiedForecastRelease.verify(publication, "seal.json", manifest_sha256=digest(publication / "seal.json"),
                                        expectation=replace(wanted, configuration_sha256="e" * 64))
    (publication / "bank.bin").write_bytes(b"corrupt payload same owner")
    with pytest.raises(ValueError):
        VerifiedForecastRelease.verify(publication, "seal.json", manifest_sha256=digest(publication / "seal.json"), expectation=wanted)


def test_incomplete_seal_and_missing_forecast_rejected(fixture):
    root, _, _, index = fixture
    wanted = expectation(index, FoldInputs("pair", ("D3", "D4")))
    publication, record = publish_seal(root, wanted)
    for changed in ({**record, "status": "in_progress"}, {**record, "forecasts": []}):
        write_json(publication / "seal.json", changed)
        with pytest.raises(ValueError):
            VerifiedForecastRelease.verify(publication, "seal.json", manifest_sha256=digest(publication / "seal.json"), expectation=wanted)


def test_hash_symlink_and_budget_guards(fixture, tmp_path):
    root, metadata, binding, index = fixture
    fold = FoldInputs("pair", ("D3", "D4"))
    with pytest.raises(ValueError, match="caps"):
        index.open(root, fold, "query", query_donor="D1", max_batch_rows=8193)
    linked = tmp_path / "linked"
    linked.symlink_to(metadata, target_is_directory=True)
    with pytest.raises(ValueError, match="nonsymlink"):
        NativeInputIndex(linked, READER, binding=binding)
    original = metadata / "source/guides.parquet"
    original.write_bytes(original.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        NativeInputIndex(metadata, READER, binding=binding)
    count_path = root / "source/D1_Stim8hr/counts/00000.csr.npz"
    count_path.write_bytes(count_path.read_bytes() + b"changed")
    handle = index.open(root, fold, "query", query_donor="D1")
    with pytest.raises(Exception, match="size"):
        handle.read_rows([("D1_Stim8hr", 0, 0)])


def test_stream_resume_uses_unchanged_reader_cursor_and_caps(fixture):
    root, _, _, index = fixture
    fold = FoldInputs("pair", ("D3", "D4"))
    handle = index.open(root, fold, "dynamics_source", max_batch_rows=2)
    all_batches = list(handle.iterate_rows(batch_size=2, seed=17, epoch=0))
    cursor, first = all_batches[0]
    resumed = list(handle.iterate_rows(batch_size=2, seed=17, epoch=0, cursor=cursor))
    assert [batch.row_ids for _, batch in resumed] == [batch.row_ids for _, batch in all_batches[1:]]
    assert sum(len(batch.row_ids) for _, batch in all_batches) == 6
    with pytest.raises(Exception, match="batch"):
        handle.read_rows([("D3_Stim8hr", 0, row) for row in range(3)])
    with pytest.raises(Exception, match="differs"):
        list(handle.iterate_rows(batch_size=2, seed=18, epoch=0, cursor=cursor))


def test_changed_retained_source_is_rejected_before_import(tmp_path):
    source = tmp_path / "reader"
    source.mkdir()
    (source / "errors.py").write_text("raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="SHA-256"):
        _load_reader(source)


def test_input_exit_lease_is_role_scoped_and_does_not_read_arrays(fixture, monkeypatch):
    root, _, _, index = fixture
    handle = index.open(root, FoldInputs("D3-D4", ("D3", "D4")), "representation_fit")
    monkeypatch.setattr(handle._reader, "_load", lambda *a: pytest.fail("lease read a count array"))
    snapshot = handle.capture_publication_identity()
    assert snapshot["source_ids"] == ["D3_Stim48hr", "D3_Stim8hr", "D4_Stim48hr", "D4_Stim8hr"]
    assert all("D1_" not in name and "D2_" not in name and "Rest" not in name for name in snapshot["members"])
    report = handle.verify_publication_identity(snapshot)
    assert report["checked_members"] == 18 and report["additional_count_array_scans"] == 0
    assert report["status"] == "unchanged"
    with pytest.raises(ValueError, match="captured once"):
        handle.capture_publication_identity()
    other = index.open(root, FoldInputs("D3-D4", ("D3", "D4")), "representation_fit")
    with pytest.raises(ValueError, match="same-worker"):
        other.verify_publication_identity(snapshot)


@pytest.mark.parametrize("change", ("payload", "index", "snapshot", "symlink"))
def test_input_exit_lease_rejects_mutated_publication(fixture, change):
    root, _, _, index = fixture
    handle = index.open(root, FoldInputs("D3-D4", ("D3", "D4")), "representation_fit")
    snapshot = handle.capture_publication_identity()
    path = root / "source/D3_Stim8hr/counts/00000.csr.npz"
    if change == "payload":
        path.write_bytes(path.read_bytes() + b"changed after consumption")
    elif change == "index":
        meta = root / "source/shard_manifest.parquet"
        meta.write_bytes(meta.read_bytes() + b"changed index")
    elif change == "snapshot":
        snapshot["members"].pop("source/D3_Stim8hr/counts/00000.csr.npz")
    else:
        replacement = path.with_suffix(".replacement")
        path.rename(replacement)
        path.symlink_to(replacement)
    with pytest.raises(ValueError):
        handle.verify_publication_identity(snapshot)
