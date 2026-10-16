"""Fabricated bounded temporary files only; zero model/optimizer/GPU calls."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from frontier.cache_storage import (ARRAY_FIELDS, MappedFoldCache, load_fold_cache,
                                   publish_fold_cache)
from frontier.fitting_provider import ProviderLimits
from test_fitting_provider import fixture


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class CacheStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "fold"
        self.cache, self.populations = fixture()
        self.cap = 2 * 1024**2
        self.limits = ProviderLimits(32 * 1024**2, 32 * 1024**2, 100)

    def publish(self, **kwargs):
        arguments = dict(expected_manifest_sha256=self.cache.manifest.identity,
                         producer_sha256="d" * 64, output_cap_bytes=self.cap)
        arguments.update(kwargs)
        return publish_fold_cache(self.cache, self.root, **arguments)

    def load(self, receipt, **kwargs):
        arguments = dict(expected_manifest_sha256=self.cache.manifest.identity,
                         expected_complete_sha256=receipt["complete_sha256"],
                         output_cap_bytes=self.cap, limits=self.limits)
        arguments.update(kwargs)
        return load_fold_cache(self.root, **arguments)

    def rewrite_metadata(self, receipt, transform):
        """Create explicitly re-hashed MALFORMED metadata in a fabricated folder."""
        metadata = json.loads((self.root / "MANIFEST.json").read_bytes())
        transform(metadata)
        raw = encoded(metadata)
        (self.root / "MANIFEST.json").write_bytes(raw)
        complete = json.loads((self.root / "COMPLETE.json").read_bytes())
        complete["manifest_file"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        array_bytes = sum(p.stat().st_size for p in self.root.iterdir() if p.suffix == ".npy")
        while True:
            completion = encoded(complete)
            total = array_bytes + len(raw) + len(completion)
            if complete["total_bytes"] == total:
                break
            complete["total_bytes"] = total
        (self.root / "COMPLETE.json").write_bytes(completion)
        return dict(receipt, complete_sha256=hashlib.sha256(completion).hexdigest())

    def test_roundtrip_exact_fold_support_readonly_mmap_and_full_exit_rehash(self):
        receipt = self.publish()
        self.assertEqual(receipt["total_bytes"], sum(p.stat().st_size for p in self.root.iterdir()))
        self.assertEqual(receipt["array_files"], 20)
        self.assertFalse(receipt["scientific_launch_authority"])
        self.assertFalse((self.root / ".COMPLETE.pending").exists())
        with self.load(receipt) as mapped:
            self.assertIsInstance(mapped, MappedFoldCache)
            self.assertEqual(mapped.publication_identity, receipt["complete_sha256"])
            self.assertEqual(mapped.publication_reference["total_bytes"], receipt["total_bytes"])
            self.assertEqual(mapped.manifest, self.cache.manifest)
            self.assertEqual(mapped.laws, self.cache.laws)
            self.assertEqual(mapped.paired, self.cache.paired)
            self.assertEqual(mapped.molecular_weights, self.cache.molecular_weights)
            for key, original in self.cache.populations.items():
                recovered = mapped.populations[key]
                for field in ARRAY_FIELDS:
                    value = getattr(recovered, field)
                    self.assertIsInstance(value, np.memmap)
                    self.assertFalse(value.flags.writeable)
                    np.testing.assert_array_equal(value, getattr(original, field))
                # Complete catalog preserves both zero-source and count-only cases.
                self.assertEqual(len(recovered.captured_counts), 7)
            for actual, expected in zip(mapped.pool_truth, self.cache.pool_truth):
                np.testing.assert_array_equal(actual, expected)
            mapped.assert_unchanged()
            self.assertFalse(mapped.verification["exit_full_hash_verified"])
            with self.assertRaises(TypeError):
                mapped.publication_reference["total_bytes"] = 0
            with self.assertRaises(ValueError):
                mapped.populations["D3", "Stim8hr"].states[0, 0] = 0
        self.assertTrue(mapped.verification["exit_full_hash_verified"])
        self.assertEqual(mapped.verification["exit_file_hash_bytes"], receipt["total_bytes"])
        self.assertTrue(all(file.stream.closed for file in mapped._leases))
        self.assertEqual(len(mapped.populations), 0)
        with self.assertRaises(RuntimeError):
            mapped.assert_unchanged()

    def test_context_is_required_and_cannot_be_entered_twice_or_after_close(self):
        receipt = self.publish()
        mapped = self.load(receipt)
        with self.assertRaises(RuntimeError):
            mapped.assert_unchanged()
        with mapped:
            with self.assertRaises(RuntimeError):
                mapped.__enter__()
        with self.assertRaises(RuntimeError):
            mapped.__enter__()

    def test_exclusive_new_destination_no_overwrite(self):
        receipt = self.publish()
        before = (self.root / "COMPLETE.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual((self.root / "COMPLETE.json").read_bytes(), before)
        self.assertEqual(hashlib.sha256(before).hexdigest(), receipt["complete_sha256"])

    def test_cap_includes_headers_and_metadata_and_preflight_makes_no_directory(self):
        raw_bytes = sum(getattr(p, field).nbytes for p in self.populations for field in ARRAY_FIELDS)
        with self.assertRaisesRegex(ValueError, "headers/metadata"):
            self.publish(output_cap_bytes=raw_bytes)
        self.assertFalse(self.root.exists())

    def test_failed_write_leaves_incomplete_output_and_never_complete(self):
        from frontier import cache_storage
        real = cache_storage._write_once
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise OSError("fabricated I/O failure")
            return real(*args, **kwargs)
        with patch.object(cache_storage, "_write_once", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "fabricated"):
                self.publish()
        self.assertTrue((self.root / "p000.captured_counts.npy").exists())
        self.assertFalse((self.root / "COMPLETE.json").exists())
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_publication_is_last_and_prior_file_mutation_prevents_completion(self):
        from frontier import cache_storage
        real = cache_storage._write_once
        names = []
        def mutate_after_manifest(*args, **kwargs):
            result = real(*args, **kwargs)
            names.append(args[1])
            if args[1] == "MANIFEST.json":
                path = self.root / "p000.captured_counts.npy"
                path.write_bytes(path.read_bytes())
            return result
        with patch.object(cache_storage, "_write_once", side_effect=mutate_after_manifest):
            with self.assertRaisesRegex(ValueError, "changed before completion"):
                self.publish()
        self.assertEqual(names[-1], "MANIFEST.json")
        self.assertEqual(len(names), 21)
        self.assertFalse((self.root / "COMPLETE.json").exists())
        self.assertFalse((self.root / ".COMPLETE.pending").exists())

    def test_changed_source_content_cannot_be_published(self):
        array = self.populations[0].captured_counts
        array.setflags(write=True)
        array[0] += 1
        array.setflags(write=False)
        with self.assertRaisesRegex(ValueError, "Source array content"):
            self.publish()
        self.assertFalse(self.root.exists())

    def test_manifest_and_complete_expectations_are_required_and_exact(self):
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            self.publish(expected_manifest_sha256="f" * 64)
        receipt = self.publish()
        with self.assertRaises(ValueError):
            self.load(receipt, expected_complete_sha256="f" * 64)
        with self.assertRaises(ValueError):
            self.load(receipt, expected_manifest_sha256="f" * 64)

    def test_duplicate_json_keys_are_rejected_even_with_bound_complete_hash(self):
        receipt = self.publish()
        path = self.root / "COMPLETE.json"
        value = path.read_bytes().replace(b'"status":"complete"', b'"status":"complete","status":"complete"')
        path.write_bytes(value)
        revised = dict(receipt, complete_sha256=hashlib.sha256(value).hexdigest())
        with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.load(revised)

    def test_load_storage_and_host_caps_before_mmap(self):
        receipt = self.publish()
        with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
            with self.assertRaises(ValueError):
                self.load(receipt, output_cap_bytes=receipt["total_bytes"] - 1)
            with self.assertRaisesRegex(ValueError, "host cap"):
                self.load(receipt, limits=ProviderLimits(1, 1, 100))

    def test_unexpected_file_is_rejected_before_mmap(self):
        receipt = self.publish()
        (self.root / "unbound.txt").write_text("fabricated")
        with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
            with self.assertRaisesRegex(ValueError, "Unexpected or missing"):
                self.load(receipt)

    def test_truncated_and_appended_array_bytes_rejected_before_mmap(self):
        receipt = self.publish()
        path = self.root / "p000.captured_counts.npy"
        original = path.read_bytes()
        for changed in (original[:-1], original + b"extra"):
            path.write_bytes(changed)
            with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
                with self.assertRaisesRegex(ValueError, "file size"):
                    self.load(receipt)

    def test_payload_hash_mismatch_rejected_before_mmap(self):
        receipt = self.publish()
        path = self.root / "p000.captured_counts.npy"
        value = bytearray(path.read_bytes()); value[-1] ^= 1
        path.write_bytes(value)
        with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.load(receipt)

    def test_fixed_numeric_dtype_shape_and_inventory_rejects_malicious_metadata(self):
        receipt = self.publish()
        original_manifest = (self.root / "MANIFEST.json").read_bytes()
        original_complete = (self.root / "COMPLETE.json").read_bytes()
        changes = ({"dtype": "|O"}, {"relative_path": "../outside.npy"},
                   {"shape": [True]}, {"population_index": True}, {"fortran_order": True},
                   {"data_offset": 0}, {"file_bytes": 1}, {"dtype": ">i8"})
        for changeset in changes:
            with self.subTest(changeset=changeset):
                (self.root / "MANIFEST.json").write_bytes(original_manifest)
                (self.root / "COMPLETE.json").write_bytes(original_complete)
                revised = self.rewrite_metadata(receipt, lambda meta: meta["arrays"][0].update(changeset))
                with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
                    with self.assertRaises(ValueError):
                        self.load(revised)

    def test_actual_canonical_header_is_checked_even_with_matching_file_hash(self):
        receipt = self.publish()
        path = self.root / "p000.captured_counts.npy"
        value = bytearray(path.read_bytes())
        value[0] ^= 1
        path.write_bytes(value)
        revised = self.rewrite_metadata(receipt, lambda meta: meta["arrays"][0].update(
            file_sha256=hashlib.sha256(value).hexdigest()))
        with patch("frontier.cache_storage.np.memmap", side_effect=AssertionError("mmap must not run")):
            with self.assertRaisesRegex(ValueError, "canonical numeric"):
                self.load(revised)

    def test_array_content_binding_is_checked_beyond_file_checksum(self):
        receipt = self.publish()
        revised = self.rewrite_metadata(receipt, lambda meta: meta["arrays"][0].update(array_sha256="f" * 64))
        with self.assertRaisesRegex(ValueError, "array content hash"):
            self.load(revised)

    def test_no_root_symlink_parent_symlink_traversal_or_relative_path(self):
        receipt = self.publish()
        alias = Path(self.tmp.name) / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for path in (alias, self.root / ".." / "fold", Path("relative")):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    load_fold_cache(path, expected_manifest_sha256=self.cache.manifest.identity,
                        expected_complete_sha256=receipt["complete_sha256"], output_cap_bytes=self.cap, limits=self.limits)
        with self.assertRaises(ValueError):
            publish_fold_cache(self.cache, alias / "new", expected_manifest_sha256=self.cache.manifest.identity,
                producer_sha256="d" * 64, output_cap_bytes=self.cap)

    def test_symlink_and_hardlinked_files_are_not_admitted(self):
        receipt = self.publish()
        path = self.root / "p000.captured_counts.npy"
        backup = Path(self.tmp.name) / "fabricated-array.npy"
        path.rename(backup)
        path.symlink_to(backup)
        with self.assertRaisesRegex(ValueError, "single-link"):
            self.load(receipt)
        path.unlink()  # only the fabricated symlink in this temporary fixture
        os.link(backup, path)
        with self.assertRaisesRegex(ValueError, "single-link"):
            self.load(receipt)

    def test_changed_in_place_array_detected_at_consumption_and_exit(self):
        receipt = self.publish()
        mapped = self.load(receipt)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            with mapped:
                path = self.root / "p000.states.npy"
                with path.open("r+b") as stream:
                    stream.seek(-1, 2); stream.write(b"\x00")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    mapped.assert_unchanged()
        self.assertTrue(all(file.stream.closed for file in mapped._leases))

    def test_file_replacement_detected_even_when_bytes_identical(self):
        receipt = self.publish()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            with self.load(receipt) as mapped:
                path = self.root / "p000.states.npy"
                replacement = Path(self.tmp.name) / "replacement.npy"
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
                mapped.assert_unchanged()

    def test_failed_context_entry_closes_descriptors(self):
        receipt = self.publish()
        mapped = self.load(receipt)
        (self.root / "unexpected.txt").write_text("fixture")
        with self.assertRaises(ValueError):
            mapped.__enter__()
        self.assertTrue(mapped._closed)
        self.assertTrue(all(file.stream.closed for file in mapped._leases))

    def test_same_process_lease_is_required(self):
        receipt = self.publish()
        mapped = self.load(receipt)
        real_pid = os.getpid()
        with self.assertRaisesRegex(RuntimeError, "same-process"):
            with mapped:
                with patch("frontier.cache_storage.os.getpid", return_value=real_pid + 1):
                    mapped.assert_unchanged()
        self.assertTrue(mapped._closed)


if __name__ == "__main__":
    unittest.main()
