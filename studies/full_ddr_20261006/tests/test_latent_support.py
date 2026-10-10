"""Synthetic-only qualification of the read-only native support adapter."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/src"))
import latent_support as adapter
from credo.data.support import SupportRef, SupportStore, SupportStoreRegistry


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def identity(path):
    return adapter.FileIdentity(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)


def fixture(root, *, source="D3_Stim8hr", mutate=None, guide_count=4, dimension=2):
    """Five captured cells: two + one valid states; one absent and one zero guide."""
    latent = root / "latent"
    fitted = root / "fitted"
    latent.mkdir()
    fitted.mkdir()
    data = {
        "state": np.array([[1, 2], [3, 4], [0, 0], [5, 6], [0, 0]], dtype=np.float32),
        "offsets": np.array([0, 2, 3, 3, 3], dtype=np.int64),
        "rows": np.array([0, 1, 3], dtype=np.int64),
        "cells": [{"source_id": source, "guide_index": guide, "RNA_UMIs": umi,
                   "latent_row": i, "geometry_available": umi > 0, "cell_id": f"cell{i}"}
                  for i, (guide, umi) in enumerate(((0, 10), (0, 20), (0, 0), (1, 30), (2, 0)))],
        "support": [{"source_id": source, "guide_index": i, "captured_cells": captured,
                     "geometry_cells": geometry, "geometry_absent_cells": captured - geometry}
                    for i, (captured, geometry) in enumerate(((3, 2), (1, 1), (1, 0), (0, 0)))],
    }
    assert guide_count >= 4 and dimension >= 2
    data["state"] = np.pad(data["state"], ((0, 0), (0, dimension - 2)))
    data["offsets"] = np.pad(data["offsets"], (0, guide_count - 4), constant_values=3)
    data["support"].extend(dict(source_id=source, guide_index=i, captured_cells=0,
                                geometry_cells=0, geometry_absent_cells=0)
                           for i in range(4, guide_count))
    if mutate:
        mutate(data)
    with h5py.File(latent / "latent.h5", "x") as handle:
        handle.create_dataset("state", data=data["state"])
        handle.create_dataset("population_offsets", data=data["offsets"])
        handle.create_dataset("population_rows", data=data["rows"])
    pq.write_table(pa.Table.from_pylist(data["cells"]), latent / "cells.parquet")
    pq.write_table(pa.Table.from_pylist(data["support"]), latent / "support.parquet")
    model = "a" * 64
    representation_record = {"schema_id": "credo.fold_count_representation_bundle", "schema_version": 2,
        "stage": "fitted_representation", "facts": {
        "model_numerical_sha256": model, "latent_dim": dimension, "frozen_fitting_geometry": True},
        "artifacts": [{"relative_uri": "model.safetensors", "sha256": model}]}
    write_json(fitted / "COMPLETE.json", representation_record)
    representation = adapter.RepresentationIdentity("frozen_r48", adapter._contract_id(representation_record), model, model)
    artifacts = [identity(latent / name) for name in ("latent.h5", "cells.parquet", "support.parquet")]
    record = {"schema_id": "credo.fold_count_representation_bundle", "schema_version": 2,
        "stage": "latent_cells", "parents": {"representation": representation.artifact_sha256},
        "artifacts": [{"relative_uri": item.path.name, "sha256": item.sha256, "size_bytes": item.size_bytes} for item in artifacts],
        "facts": {"source_ids": [source], "complete_catalog_categories": guide_count,
                  "retained_individual_cells": 5, "geometry_cells": 3, "latent_dim": dimension,
                  "latent_payload_sha256": artifacts[0].sha256, "encoded_weight_updates": 0,
                  "law_scope": "empirical_positive_RNA_cells; captured_zero_RNA_support_retained_separately",
                  "zero_RNA_coordinate_policy": "zero_placeholder_geometry_available_false"}}
    write_json(latent / "COMPLETE.json", record)
    return adapter.FrozenLatentBinding("frozen_store", *artifacts,
        completion=identity(latent / "COMPLETE.json"), representation_record=identity(fitted / "COMPLETE.json"),
        representation=representation, source_ids=(source,), guide_count=guide_count, dimension=dimension,
        retained_cells=5, geometry_cells=3, batch_rows=1)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def open(self, binding):
        return adapter.FrozenLatentSupportStore(binding, adapter.PhaseAccess("fitting"),
                                              expected_representation=binding.representation)

    def reject_mutation(self, mutate):
        with self.assertRaises(adapter.LatentSupportError):
            self.open(fixture(self.root, mutate=mutate))

    def test_native_store_returns_individual_law_not_mean(self):
        store = self.open(fixture(self.root))
        self.assertIsInstance(store, SupportStore)
        ref = store.ref("D3_Stim8hr", 0)
        law = SupportStoreRegistry([store]).read(ref)
        np.testing.assert_array_equal(law.coordinates, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(law.probabilities, [0.5, 0.5])
        self.assertFalse(law.coordinates.flags.writeable)
        self.assertFalse(law.probabilities.flags.writeable)
        self.assertEqual(store.dimension("frozen_r48"), 2)
        self.assertTrue(store.validate(full_scan=True).valid)

    def test_positive_capture_missing_state_distinct_from_observed_zero(self):
        store = self.open(fixture(self.root))
        absent = store.support_record("D3_Stim8hr", 2)
        zero = store.support_record("D3_Stim8hr", 3)
        self.assertEqual((absent["captured_cells"], absent["geometry_cells"]), (1, 0))
        self.assertEqual((zero["captured_cells"], zero["geometry_cells"]), (0, 0))
        for guide in (2, 3):
            ref = store.ref("D3_Stim8hr", guide)
            self.assertFalse(store.contains(ref))
            with self.assertRaises(adapter.LatentSupportError):
                store.read(ref)

    def test_zero_rna_placeholder_is_not_a_particle_even_if_nonfinite(self):
        binding = fixture(self.root, mutate=lambda d: d["state"].__setitem__(2, [np.nan, np.nan]))
        store = self.open(binding)
        self.assertTrue(store.validate(full_scan=True).valid)
        self.assertEqual(len(store.read(store.ref("D3_Stim8hr", 0)).coordinates), 2)

    def test_nonfinite_positive_rna_coordinate_rejected_on_law_read(self):
        binding = fixture(self.root, mutate=lambda d: d["state"].__setitem__(0, [np.nan, 2]))
        store = self.open(binding)
        with self.assertRaises(adapter.LatentSupportError):
            store.read(store.ref("D3_Stim8hr", 0))
        self.assertFalse(store.validate(full_scan=True).valid)

    def test_support_record_is_a_copy(self):
        store = self.open(fixture(self.root))
        record = store.support_record("D3_Stim8hr", 0)
        record["geometry_cells"] = 999
        self.assertEqual(store.support_record("D3_Stim8hr", 0)["geometry_cells"], 2)

    def test_read_many_and_no_mutable_mapping(self):
        store = self.open(fixture(self.root))
        refs = [store.ref("D3_Stim8hr", i) for i in (0, 1)]
        laws = store.read_many(refs)
        self.assertEqual(len(laws), 2)
        with self.assertRaises(TypeError):
            laws[refs[0]] = laws[refs[1]]

    def test_cached_snapshot_matches_laws_without_per_guide_file_io(self):
        store = self.open(fixture(self.root))
        refs = [store.ref("D3_Stim8hr", i) for i in (0, 1)]
        expected = store.read_many(refs)
        with patch.object(adapter.h5py, "File", wraps=h5py.File) as opened:
            with store.cached_snapshot(maximum_bytes=store.snapshot_bytes):
                with patch.object(adapter, "_signature", side_effect=AssertionError("Per-law stat")):
                    for _ in range(4):
                        for ref in refs:
                            self.assertTrue(store.contains(ref))
                            actual = store.read(ref)
                            np.testing.assert_array_equal(actual.coordinates, expected[ref].coordinates)
                            np.testing.assert_array_equal(actual.probabilities, expected[ref].probabilities)
                            self.assertFalse(actual.coordinates.flags.writeable)
            self.assertEqual(opened.call_count, 1)
        self.assertIsNone(store._snapshot_cache)

    def test_snapshot_budget_and_nested_lease_fail_before_extra_read(self):
        store = self.open(fixture(self.root))
        with patch.object(adapter.h5py, "File", side_effect=AssertionError("Unadmitted read")):
            with self.assertRaises(adapter.LatentSupportError):
                with store.cached_snapshot(maximum_bytes=store.snapshot_bytes - 1):
                    pass
        with store.cached_snapshot(maximum_bytes=store.snapshot_bytes):
            with self.assertRaises(adapter.LatentSupportError):
                with store.cached_snapshot(maximum_bytes=store.snapshot_bytes):
                    pass

    def test_snapshot_detects_changed_inputs_at_exit(self):
        store = self.open(fixture(self.root))
        with self.assertRaises(adapter.LatentSupportError):
            with store.cached_snapshot(maximum_bytes=store.snapshot_bytes):
                with store.binding.support.path.open("ab") as stream:
                    stream.write(b"changed")
                # Laws are the previously verified snapshot, not changed files.
                np.testing.assert_array_equal(store.read(store.ref("D3_Stim8hr", 0)).coordinates,
                                              [[1, 2], [3, 4]])
        self.assertIsNone(store._snapshot_cache)

    def test_snapshot_exception_releases_cache(self):
        store = self.open(fixture(self.root))
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with store.cached_snapshot(maximum_bytes=store.snapshot_bytes):
                raise RuntimeError("caller failure")
        self.assertIsNone(store._snapshot_cache)
        self.assertTrue(store.validate().valid)

    def test_cached_snapshot_zero_rna_and_nonfinite_rules_preserved(self):
        binding = fixture(self.root, mutate=lambda d: d["state"].__setitem__(2, [np.nan, np.nan]))
        store = self.open(binding)
        with store.cached_snapshot(maximum_bytes=store.snapshot_bytes):
            self.assertTrue(store.validate(full_scan=True).valid)
            with self.assertRaises(adapter.LatentSupportError):
                store.read(store.ref("D3_Stim8hr", 2))

    def test_closed_store_rejects_reads(self):
        store = self.open(fixture(self.root))
        ref = store.ref("D3_Stim8hr", 0)
        store.close()
        self.assertFalse(store.contains(ref))
        self.assertFalse(store.validate().valid)
        with self.assertRaises(adapter.LatentSupportError):
            store.read(ref)

    def test_wrong_reference_and_noncanonical_indices(self):
        store = self.open(fixture(self.root))
        for ref in (SupportRef("other", "frozen_r48", "D3_Stim8hr::0"),
                    SupportRef("frozen_store", "other", "D3_Stim8hr::0"),
                    SupportRef("frozen_store", "frozen_r48", "D3_Stim8hr::00"),
                    SupportRef("frozen_store", "frozen_r48", "D3_Stim8hr::-1")):
            self.assertFalse(store.contains(ref))
            with self.assertRaises(KeyError):
                store.read(ref)
        with self.assertRaises(adapter.LatentSupportError):
            store.ref("D3_Stim8hr", True)

    def test_unknown_phase_source_and_larry_rejected_without_io(self):
        binding = fixture(self.root)
        with patch.object(adapter, "_verify_file", side_effect=AssertionError("unauthorized I/O")):
            for source in ("D1_Stim48hr", "D2_Stim8hr", "LARRY_Stim8hr"):
                with self.assertRaises(adapter.LatentSupportError):
                    self.open(replace(binding, source_ids=(source,)))

    def test_d1_forecast_requires_selection_and_checkpoint(self):
        phase = adapter.PhaseAccess("forecast")
        with self.assertRaises(adapter.LatentSupportError):
            phase.admit(("D1_Stim8hr",))
        phase = replace(phase, architecture_selection_sha256="a"*64, terminal_checkpoints_sha256="b"*64,
                        verified_by_lifecycle=True)
        phase.admit(("D1_Stim8hr",))
        with self.assertRaises(adapter.LatentSupportError):
            phase.admit(("D1_Stim48hr",))

    def test_d1_endpoint_requires_seal_and_can_open_after_release(self):
        binding = fixture(self.root, source="D1_Stim48hr")
        phase = adapter.PhaseAccess("assessment", "a"*64, "b"*64)
        with patch.object(adapter, "_verify_file", side_effect=AssertionError("premature I/O")):
            with self.assertRaises(adapter.LatentSupportError):
                adapter.FrozenLatentSupportStore(binding, phase, expected_representation=binding.representation)
        phase = replace(phase, forecast_seal_sha256="c"*64, verified_by_lifecycle=True)
        store = adapter.FrozenLatentSupportStore(binding, phase, expected_representation=binding.representation)
        self.assertEqual(len(store.read(store.ref("D1_Stim48hr", 0)).coordinates), 2)

    def test_final_refit_requires_completed_assessment(self):
        phase = adapter.PhaseAccess("final_refit", "a"*64, "b"*64, "c"*64)
        with self.assertRaises(adapter.LatentSupportError):
            phase.admit(("D1_Stim48hr", "D4_Stim48hr"))
        replace(phase, assessment_sha256="d"*64, verified_by_lifecycle=True).admit(("D1_Stim48hr", "D4_Stim48hr"))

    def test_bare_release_hashes_do_not_establish_lifecycle_authority(self):
        phase = adapter.PhaseAccess("assessment", "a"*64, "b"*64, "c"*64)
        with self.assertRaises(adapter.LatentSupportError):
            phase.admit(("D1_Stim48hr",))

    def test_d2_selection_does_not_become_training(self):
        binding = fixture(self.root, source="D2_Stim48hr")
        with self.assertRaises(adapter.LatentSupportError):
            self.open(binding)
        store = adapter.FrozenLatentSupportStore(binding, adapter.PhaseAccess("selection"),
                                              expected_representation=binding.representation)
        self.assertTrue(store.contains(store.ref("D2_Stim48hr", 0)))

    def test_representation_mismatch_before_io(self):
        binding = fixture(self.root)
        expected = replace(binding.representation, model_sha256="b"*64, scaler_sha256="b"*64)
        with patch.object(adapter, "_verify_file", side_effect=AssertionError("mismatched I/O")):
            with self.assertRaises(adapter.LatentSupportError):
                adapter.FrozenLatentSupportStore(binding, adapter.PhaseAccess("fitting"), expected_representation=expected)

    def test_external_or_mismatched_scaler_rejected(self):
        binding = fixture(self.root)
        with self.assertRaises(adapter.LatentSupportError):
            replace(binding.representation, scaler_sha256="c"*64)
        with self.assertRaises(adapter.LatentSupportError):
            replace(binding.representation, scaler_location="separate_scaler.json")

    def test_latent_byte_identity_mismatch(self):
        binding = fixture(self.root)
        with binding.latent.path.open("ab") as handle:
            handle.write(b"unexpected")
        with self.assertRaises(adapter.LatentSupportError):
            self.open(binding)

    def test_file_changed_after_admission_rejected(self):
        binding = fixture(self.root)
        store = self.open(binding)
        ref = store.ref("D3_Stim8hr", 0)
        with binding.support.path.open("ab") as handle:
            handle.write(b"unexpected")
        with self.assertRaises(adapter.LatentSupportError):
            store.read(ref)

    def test_symlink_rejected(self):
        binding = fixture(self.root)
        linkroot = self.root / "alias"
        linkroot.symlink_to(binding.latent.path.parent, target_is_directory=True)
        binding = replace(binding, latent=replace(binding.latent, path=linkroot/"latent.h5"),
                          cells=replace(binding.cells, path=linkroot/"cells.parquet"),
                          support=replace(binding.support, path=linkroot/"support.parquet"),
                          completion=replace(binding.completion, path=linkroot/"COMPLETE.json"))
        with self.assertRaises(adapter.LatentSupportError):
            self.open(binding)

    def test_index_memory_admission_before_file_io(self):
        binding = replace(fixture(self.root), maximum_index_bytes=1)
        with patch.object(adapter, "_verify_file", side_effect=AssertionError("overbudget I/O")):
            with self.assertRaises(adapter.LatentSupportError):
                self.open(binding)

    def test_law_materialization_budget(self):
        store = self.open(replace(fixture(self.root), maximum_law_bytes=1))
        with self.assertRaises(adapter.LatentSupportError):
            store.read(store.ref("D3_Stim8hr", 0))

    def test_read_many_combined_budget(self):
        store = self.open(replace(fixture(self.root), maximum_law_bytes=100))
        store.read(store.ref("D3_Stim8hr", 0))
        with self.assertRaises(adapter.LatentSupportError):
            store.read_many([store.ref("D3_Stim8hr", i) for i in (0, 1)])

    def test_missing_catalog_category_rejected(self):
        self.reject_mutation(lambda d: d["support"].pop())

    def test_duplicate_catalog_category_rejected(self):
        self.reject_mutation(lambda d: d["support"].append(dict(d["support"][0])))

    def test_negative_or_fractional_counts_rejected(self):
        self.reject_mutation(lambda d: d["support"][0].__setitem__("captured_cells", -1))

    def test_geometry_absent_accounting_rejected(self):
        self.reject_mutation(lambda d: d["support"][0].__setitem__("geometry_absent_cells", 0))

    def test_offsets_disagree_with_geometry(self):
        self.reject_mutation(lambda d: d["offsets"].__setitem__(1, 1))

    def test_duplicate_population_rows_rejected(self):
        self.reject_mutation(lambda d: d["rows"].__setitem__(1, 0))

    def test_unordered_population_rows_rejected(self):
        self.reject_mutation(lambda d: d["rows"].__setitem__(slice(0, 2), [1, 0]))

    def test_out_of_bounds_population_rows_rejected(self):
        self.reject_mutation(lambda d: d["rows"].__setitem__(2, 5))

    def test_zero_rna_indexed_as_geometry_rejected(self):
        self.reject_mutation(lambda d: d["rows"].__setitem__(1, 2))

    def test_wrong_guide_indexed_as_population_rejected(self):
        self.reject_mutation(lambda d: d["cells"][1].__setitem__("guide_index", 1))

    def test_wrong_source_cell_rejected(self):
        self.reject_mutation(lambda d: d["cells"][1].__setitem__("source_id", "D4_Stim8hr"))

    def test_geometry_flag_contradicts_rna_depth(self):
        self.reject_mutation(lambda d: d["cells"][1].__setitem__("geometry_available", False))

    def test_reordered_latent_metadata_rejected(self):
        self.reject_mutation(lambda d: d["cells"][1].__setitem__("latent_row", 0))

    def test_missing_cell_metadata_rejected(self):
        self.reject_mutation(lambda d: d["cells"].pop())

    def test_wrong_state_dimensions_rejected(self):
        self.reject_mutation(lambda d: d.__setitem__("state", d["state"][:, :1]))

    def test_wrong_state_dtype_rejected(self):
        self.reject_mutation(lambda d: d.__setitem__("state", d["state"].astype(np.float64)))

    def test_wrong_index_dtype_rejected(self):
        self.reject_mutation(lambda d: d.__setitem__("rows", d["rows"].astype(np.float32)))

    def test_native_completion_parent_mismatch(self):
        binding = fixture(self.root)
        record = json.loads(binding.completion.path.read_text())
        record["parents"]["representation"] = "b"*64
        write_json(binding.completion.path, record)
        with self.assertRaises(adapter.LatentSupportError):
            self.open(replace(binding, completion=identity(binding.completion.path)))

    def test_declared_source_order_must_match_before_payload_hash(self):
        binding = fixture(self.root)
        with patch.object(adapter, "_verify_file", wraps=adapter._verify_file) as verify:
            with self.assertRaises(adapter.LatentSupportError):
                self.open(replace(binding, source_ids=("D4_Stim8hr",)))
        self.assertEqual([call.args[0].path for call in verify.call_args_list],
                         [binding.completion.path, binding.representation_record.path])

    def test_representation_record_hash_mismatch(self):
        binding = fixture(self.root)
        record = json.loads(binding.representation_record.path.read_text())
        record["facts"]["model_numerical_sha256"] = "b"*64
        write_json(binding.representation_record.path, record)
        with self.assertRaises(adapter.LatentSupportError):
            self.open(replace(binding, representation_record=identity(binding.representation_record.path)))


class NewExportAndCompositeTests(unittest.TestCase):
    """Runs the actual new export writer on small synthetic count blocks."""

    setUp = AdapterTests.setUp
    open = AdapterTests.open

    def new_export(self):
        from encode_sources import export_source
        import pandas as pd
        from scipy import sparse
        old = fixture(self.root)
        guides = pd.DataFrame({"guide_index": [0, 1, 2, 3], "guide_id": ["a", "b", "c", "NTC"],
                               "target_id": ["A", "B", "C", "NTC"], "is_control": [False, False, False, True]})
        matrix = sparse.csr_matrix(np.array([[1, 2], [3, 4], [0, 0], [5, 6], [0, 0]], dtype=np.uint32))
        cells = pd.DataFrame({"source_id": ["D3_Stim8hr"]*5, "shard": [0]*5,
                              "row_in_shard": list(range(5)), "cell_id": [f"n{i}" for i in range(5)],
                              "guide_index": [0, 0, 0, 1, 2], "RNA_UMIs": [3, 7, 0, 11, 0]})
        lineage = {"input_access_sha256": "d"*64, "gene_axis_sha256": "e"*64,
                   "implementation_sha256": "f"*64,
                   "representation_sha256": old.representation.artifact_sha256,
                   "model_sha256": old.representation.model_sha256,
                   "scaler_sha256": old.representation.scaler_sha256}
        destination = self.root / "new_export"
        result = export_source(destination, source_id="D3_Stim8hr", phase="development",
            expected_rows=5, guides=guides, latent_dim=2, rna_features=2,
            blocks=[(matrix[:2], cells.iloc[:2]), (matrix[2:], cells.iloc[2:])],
            encode=lambda counts: counts.toarray().astype(np.float32), lineage=lineage)
        bound = replace(old, latent=identity(destination/"latent.h5"), cells=identity(destination/"cells.parquet"),
                        support=identity(destination/"support.parquet"), completion=identity(destination/"COMPLETE.json"))
        return bound, result

    def test_actual_export_to_native_individual_law_integration(self):
        binding, receipt = self.new_export()
        self.assertEqual(receipt["schema"], "credo.all4_ddr.latent_source.v1")
        self.assertNotIn("stage", receipt)
        store = self.open(binding)
        law = SupportStoreRegistry([store]).read(store.ref("D3_Stim8hr", 0))
        np.testing.assert_array_equal(law.coordinates, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(law.probabilities, [0.5, 0.5])
        self.assertEqual(store.support_record("D3_Stim8hr", 2)["geometry_absent_cells"], 1)
        self.assertEqual(store.support_record("D3_Stim8hr", 3)["captured_cells"], 0)
        self.assertTrue(store.validate(full_scan=True).valid)

    def reject_new_record(self, mutate):
        binding, receipt = self.new_export()
        mutate(receipt)
        write_json(binding.completion.path, receipt)
        with self.assertRaises(adapter.LatentSupportError):
            self.open(replace(binding, completion=identity(binding.completion.path)))

    def test_new_export_wrong_scaler_lineage(self):
        self.reject_new_record(lambda r: r["lineage"].__setitem__("scaler_sha256", "b"*64))

    def test_new_export_wrong_bytes_inventory(self):
        self.reject_new_record(lambda r: r["files"]["latent.h5"].__setitem__("bytes", 1))

    def test_new_export_wrong_ordered_cell_identity(self):
        self.reject_new_record(lambda r: r.__setitem__("ordered_cell_address_sha256", "b"*64))

    def test_new_export_wrong_phase_cannot_be_backdated(self):
        self.reject_new_record(lambda r: r.__setitem__("phase", "final_refit"))

    def test_unknown_export_schema_not_treated_as_old(self):
        self.reject_new_record(lambda r: r.__setitem__("schema", "unrecognized"))

    def two_stores(self):
        roots = [self.root / "train", self.root / "select"]
        for root in roots:
            root.mkdir()
        training = self.open(fixture(roots[0]))
        binding = replace(fixture(roots[1], source="D2_Stim48hr"), store_id="selection_store")
        selection = adapter.FrozenLatentSupportStore(binding, adapter.PhaseAccess("selection"),
                                                    expected_representation=binding.representation)
        return training, selection

    def test_native_development_composite_preserves_phase_specific_stores(self):
        training, selection = self.two_stores()
        store = adapter.DevelopmentLatentSupportStore("development", [training, selection])
        self.assertIsInstance(store, SupportStore)
        self.assertEqual(store.source_ids, ("D3_Stim8hr", "D2_Stim48hr"))
        self.assertEqual(selection.phase.phase_role, "selection")
        refs = [store.ref(source, 0) for source in store.source_ids]
        self.assertEqual({r.store_id for r in refs}, {"development"})
        self.assertEqual(len(store.read_many(refs)), 2)
        self.assertTrue(store.validate(full_scan=True).valid)
        store.close()
        self.assertFalse(store.contains(refs[0]))
        self.assertFalse(training.validate().valid)

    def test_composite_overlapping_sources_rejected(self):
        training, _ = self.two_stores()
        with self.assertRaises(adapter.LatentSupportError):
            adapter.DevelopmentLatentSupportStore("development", [training, training])

    def test_composite_rejects_d1_even_if_individually_released(self):
        binding = fixture(self.root, source="D1_Stim48hr")
        phase = adapter.PhaseAccess("assessment", "a"*64, "b"*64, "c"*64, verified_by_lifecycle=True)
        assessment = adapter.FrozenLatentSupportStore(binding, phase, expected_representation=binding.representation)
        with self.assertRaises(adapter.LatentSupportError):
            adapter.DevelopmentLatentSupportStore("development", [assessment])


if __name__ == "__main__":
    unittest.main()
