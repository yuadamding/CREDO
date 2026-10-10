"""Small deterministic metadata fixtures; no cohort payload or model access."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_manifest as module


def fixture():
    tables = {key: [] for key in module.TABLES}
    tables["guides"] = [
        {"sgRNA": "g1", "seq": "AAAA", "primary_target_id": "target1",
         "designed_target_gene_id": "target1", "target_gene_id": "curated1",
         "is_design_ntc": False, "design_curated_disagreement": True,
         "experimental_library_member": True, "target_partition": "train"},
        {"sgRNA": "control", "seq": "CCCC", "primary_target_id": "__NTC__",
         "designed_target_gene_id": "", "target_gene_id": "NTC",
         "is_design_ntc": True, "design_curated_disagreement": False,
         "experimental_library_member": True, "target_partition": "control"},
    ]
    inventory = {}
    for donor in module.DONORS:
        for condition in module.CONDITIONS:
            source = f"{donor}_{condition}"
            elapsed, stimulated, branch = module.TIMING[condition]
            tables["samples"].append({"source_id": source, "donor_id": donor,
                "condition": condition, "collection_elapsed_hours": elapsed,
                "restimulation_hours": stimulated, "branch": branch,
                "primary_methods_verified": True, "table_S1_verified": False,
                "common_parent_culture": f"{donor}:day12_split"})
            for index, (feature, rna) in enumerate((("gA", True), ("gB", True), ("PuroR", False))):
                tables["features"].append({"source_id": source, "canonical_position": index,
                    "source_position": (index + 1) % 3, "feature_id": feature, "is_RNA": rna})
            row = {"source_id": source, "shard": 0, "rows": 5, "nnz": 10,
                   "bytes": 200, "cell_table_bytes": 300, "first_source_row": 2,
                   "last_source_row": 8, "exact_reload": True,
                   "path": f"source/{source}/counts/00000.csr.npz", "sha256": "a" * 64,
                   "cells_path": f"source/{source}/cells/00000.parquet", "cells_sha256": "b" * 64}
            tables["shards"].append(row)
            inventory[row["path"]] = row["sha256"]
            inventory[row["cells_path"]] = row["cells_sha256"]
    completion = {"RNA_features": 2, "technical_features": 1, "cells": 60, "sources": 12}
    return tables, inventory, completion


def write_fixture(directory):
    """Only synthetic test file output, never a real native-store writer."""
    tables, inventory, completion = fixture()
    root = directory / "native"
    (root / "source").mkdir(parents=True)
    files = {}
    for key, name in module.TABLES.items():
        stream = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(tables[key]), stream)
        value = stream.getvalue()
        (root / name).write_bytes(value)
        inventory[name] = module.digest(value)
        files[name] = {"sha256": module.digest(value), "size_bytes": len(value)}
    sums = "".join(f"{sha}  {name}\n" for name, sha in sorted(inventory.items())).encode()
    completion["inventory_sha256"] = module.digest(sums)
    complete = module.canonical(completion)
    for name, value in (("SHA256SUMS", sums), ("INPUT_DATA_COMPLETE.json", complete)):
        (root / name).write_bytes(value)
        files[name] = {"sha256": module.digest(value), "size_bytes": len(value)}
    receipt = {"status": "verified_metadata_only_not_numerical_admission", "files": files,
               "completion_sha256": module.digest(complete), "inventory_sha256": module.digest(sums)}
    receipt_path = directory / "COMPLETE.json"
    receipt_bytes = module.canonical(receipt)
    receipt_path.write_bytes(receipt_bytes)
    return root, receipt_path, module.digest(receipt_bytes)


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tables, self.inventory, self.complete = fixture()

    def build(self):
        return module.assemble_metadata(self.tables, self.inventory, self.complete)

    def test_complete_grid_catalog_axis_and_unknown_support(self):
        output = self.build()
        self.assertEqual(output["counts"]["sources"], 12)
        self.assertEqual(output["counts"]["producer_eligible_cells"], 60)
        self.assertEqual(output["counts"]["guides"], 2)
        self.assertEqual(output["counts"]["ntc_guides"], 1)
        self.assertFalse(output["admission"]["training_ready"])
        self.assertFalse(output["admission"]["complete_count_vectors_bound"])
        self.assertTrue(all("unknown" in r["state_support"] for r in output["sources"]))
        self.assertFalse(output["representation"]["all_twelve_source_export_established"])

    def test_contemporaneous_rest_not_temporal_zero(self):
        output = self.build()
        timing = output["temporal_contract"]
        self.assertFalse(timing["rest_to_stim8_temporal_edge"])
        self.assertEqual(timing["stimulated_edge"]["elapsed_hours"], 40)
        self.assertEqual(output["sources"][0]["collection_elapsed_hours"], 8)

    def test_d1_endpoint_preparation_delayed_until_selection_and_forecast_seal(self):
        output = self.build()
        policy = output["prospective_role_policy"]
        self.assertEqual(policy["architecture_fitting_donors"], ["D3", "D4"])
        self.assertEqual(policy["architecture_selection_donors"], ["D2"])
        self.assertEqual(policy["reserved_internal_assessment_donors"], ["D1"])
        self.assertIn("D1_forecasts_sealed", policy["D1_Stim48hr_numerical_preparation_prerequisites"])
        delayed = [row["source_id"] for row in output["sources"] if row["delayed_endpoint_preparation"]]
        self.assertEqual(delayed, ["D1_Stim48hr"])
        self.assertTrue(all(row["numerical_access_status"] == "not_admitted_by_metadata_manifest"
                            for row in output["sources"]))

    def test_annotation_disagreement_and_historical_partition_preserved(self):
        output = self.build()["canonical_guides"][0]
        self.assertEqual(output["target_gene_id"], "curated1")
        self.assertEqual(output["primary_target_id"], "target1")
        self.assertTrue(output["design_curated_disagreement"])
        self.assertEqual(output["target_partition_role"], "historical_annotation_not_new_split")

    def test_empty_ntc_gene_annotation_is_not_fabricated(self):
        row = self.build()["canonical_guides"][1]
        self.assertEqual(row["designed_target_gene_id"], "")
        self.assertEqual(row["primary_target_id"], "__NTC__")

    def test_missing_targeting_gene_annotation_rejected(self):
        self.tables["guides"][0]["designed_target_gene_id"] = ""
        with self.assertRaises(module.PreparationError):
            self.build()

    def test_source_permutation_bound_not_applied_again(self):
        output = self.build()
        self.assertEqual(output["source_feature_positions"]["D1_Rest"], [1, 2, 0])
        self.assertEqual([r["feature_id"] for r in output["canonical_features"]], ["gA", "gB", "PuroR"])
        self.assertIn("do_not_reapply", output["native_count_columns"])

    def test_no_input_mutation(self):
        before = copy.deepcopy(self.tables)
        self.build()
        self.assertEqual(self.tables, before)

    def test_missing_source_rejected(self):
        self.tables["samples"].pop()
        with self.assertRaises(module.PreparationError): self.build()

    def test_duplicate_source_rejected(self):
        self.tables["samples"][-1] = self.tables["samples"][0]
        with self.assertRaises(module.PreparationError): self.build()

    def test_mismatched_donor_rejected(self):
        self.tables["samples"][0]["donor_id"] = "D2"
        with self.assertRaises(module.PreparationError): self.build()

    def test_fictitious_rest_time_zero_rejected(self):
        self.tables["samples"][0]["collection_elapsed_hours"] = 0
        with self.assertRaises(module.PreparationError): self.build()

    def test_unverified_timing_rejected(self):
        self.tables["samples"][0]["primary_methods_verified"] = False
        with self.assertRaises(module.PreparationError): self.build()

    def test_duplicate_guide_rejected(self):
        self.tables["guides"].append(copy.deepcopy(self.tables["guides"][0]))
        with self.assertRaises(module.PreparationError): self.build()

    def test_design_nonmember_rejected(self):
        self.tables["guides"][0]["experimental_library_member"] = False
        with self.assertRaises(module.PreparationError): self.build()

    def test_invalid_ntc_flag_rejected(self):
        self.tables["guides"][0]["is_design_ntc"] = 1
        with self.assertRaises(module.PreparationError): self.build()

    def test_feature_axis_disagreement_rejected(self):
        self.tables["features"][3]["feature_id"] = "other"
        with self.assertRaises(module.PreparationError): self.build()

    def test_duplicate_canonical_position_rejected(self):
        self.tables["features"][1]["canonical_position"] = 0
        with self.assertRaises(module.PreparationError): self.build()

    def test_nonbijective_source_permutation_rejected(self):
        self.tables["features"][1]["source_position"] = 1
        with self.assertRaises(module.PreparationError): self.build()

    def test_duplicate_feature_id_rejected(self):
        self.tables["features"][1]["feature_id"] = "gA"
        with self.assertRaises(module.PreparationError): self.build()

    def test_feature_count_mismatch_rejected(self):
        self.complete["RNA_features"] = 5
        with self.assertRaises(module.PreparationError): self.build()

    def test_unknown_shard_source_rejected(self):
        self.tables["shards"][0]["source_id"] = "D5_Rest"
        with self.assertRaises(module.PreparationError): self.build()

    def test_shard_hash_substitution_rejected(self):
        self.tables["shards"][0]["sha256"] = "c" * 64
        with self.assertRaises(module.PreparationError): self.build()

    def test_unsafe_or_wrong_source_shard_path_rejected(self):
        for path in ("../outside", "source/D4_Rest/counts/00000.csr.npz"):
            self.tables["shards"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(module.PreparationError): self.build()

    def test_noncontiguous_shards_rejected(self):
        row = self.tables["shards"][0]
        row["shard"] = 1
        for key, hash_key in (("path", "sha256"), ("cells_path", "cells_sha256")):
            row[key] = row[key].replace("00000", "00001")
            self.inventory[row[key]] = row[hash_key]
        with self.assertRaises(module.PreparationError): self.build()

    def test_overlapping_intervals_rejected(self):
        row = copy.deepcopy(self.tables["shards"][0])
        row["shard"] = 1
        for key, hash_key in (("path", "sha256"), ("cells_path", "cells_sha256")):
            row[key] = row[key].replace("00000", "00001")
            self.inventory[row[key]] = row[hash_key]
        self.tables["shards"].append(row)
        self.complete["cells"] += 5
        with self.assertRaises(module.PreparationError): self.build()

    def test_total_mismatch_rejected(self):
        self.complete["cells"] += 1
        with self.assertRaises(module.PreparationError): self.build()

    def test_cannot_self_promote_metadata_to_numerical_admission(self):
        output = self.build()
        output["admission"]["training_ready"] = True
        with self.assertRaises(module.PreparationError): module.require_numerical_admission(output)


class IdentityAndPublicationTests(unittest.TestCase):
    def test_exact_fixture_import_without_any_count_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt, checksum = write_fixture(Path(temporary))
            result = module.build_manifest(root, receipt, checksum)
            self.assertEqual(result["counts"]["producer_eligible_cells"], 60)
            self.assertEqual(result["reads"]["count_payloads"], 0)
            self.assertEqual(len(result["input_binding"]["metadata_tables"]), 4)
            self.assertFalse((root / "source/D1_Rest/counts").exists())

    def test_receipt_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt, _ = write_fixture(Path(temporary))
            with self.assertRaises(module.PreparationError): module.build_manifest(root, receipt, "0" * 64)

    def test_changed_table_rejected_before_parquet_parse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt, checksum = write_fixture(Path(temporary))
            (root / module.TABLES["guides"]).write_bytes(b"broken")
            with self.assertRaises(module.PreparationError): module.build_manifest(root, receipt, checksum)

    def test_inventory_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, receipt, checksum = write_fixture(Path(temporary))
            (root / "SHA256SUMS").write_bytes(b"changed")
            with self.assertRaises(module.PreparationError): module.build_manifest(root, receipt, checksum)

    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "real").write_bytes(b"{}"); (directory / "alias").symlink_to(directory / "real")
            with self.assertRaises(module.PreparationError): module._read_regular(directory / "alias")

    def test_oversized_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "fixture"; file.write_bytes(b"12345")
            with self.assertRaises(module.PreparationError): module._read_regular(file, limit=4)

    def test_duplicate_or_escaping_inventory_rejected(self):
        for content in (("a" * 64 + "  file\n") * 2, "a" * 64 + "  ../escape\n"):
            with self.subTest(content=content), self.assertRaises(module.PreparationError): module._inventory(content.encode())

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(module.PreparationError): module._json(b'{"a":1,"a":2}')

    def test_publish_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest.json"
            value = {"schema": module.SCHEMA, "status": "metadata_only"}
            checksum = module.publish_manifest(value, output)
            self.assertEqual(module.digest(output.read_bytes()), checksum)
            self.assertEqual(json.loads(output.read_bytes()), value)
            with self.assertRaises(FileExistsError): module.publish_manifest(value, output)

    def test_publish_does_not_create_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent" / "manifest.json"
            with self.assertRaises(module.PreparationError): module.publish_manifest({}, output)
            self.assertFalse(output.parent.exists())

    def test_publish_rejects_symlink_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); (directory / "actual").mkdir()
            (directory / "alias").symlink_to(directory / "actual", target_is_directory=True)
            with self.assertRaises(module.PreparationError): module.publish_manifest({}, directory / "alias/out.json")

    def test_publish_enforces_size_cap(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(module, "OUTPUT_LIMIT", 4):
            with self.assertRaises(module.PreparationError): module.publish_manifest({"x": "y"}, Path(temporary) / "output.json")


if __name__ == "__main__":
    unittest.main()
