"""Verified count-native metadata bridge for a NEW all-donor CREDO study.

This does not read RNA, counts, cell tables, latent arrays or checkpoints.
Its result is deliberately NOT a compiled numerical study or launch authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any


SCHEMA = "credo.gse314342.full_ddr.metadata_preparation.v1"
WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_IMPORT = WORKSPACE / "credo_full_cohort_20260912/R48_downstream_v1/main_stim8_to48_v1/metadata_import"
IMPORT_SHA256 = "8d21431edd597e4324ac1c21a0384d4f3adfbc5c395305b50c68a034e4993e9d"
COMPLETION_SHA256 = "77d4fab700449ac323ff750cef6946a3185c06653891c41123fb4c76a4d5f057"
INVENTORY_SHA256 = "9b6a046958ee587ad600dd3cae0cf4aaa6f719b38c436b25a6c8c5159f7dcc63"
REMOTE_NATIVE_ROOT = "/rsrch8/home/bcb/yding4/perturbseq/inputs/GSE314342_count_native_v1"
DONORS = ("D1", "D2", "D3", "D4")
CONDITIONS = ("Rest", "Stim8hr", "Stim48hr")
TIMING = {"Rest": (8, 0, "unstimulated"), "Stim8hr": (8, 8, "stimulated_early"),
          "Stim48hr": (48, 48, "stimulated_late")}
TABLES = {
    "guides": "source/guides.parquet",
    "features": "source/features_and_source_positions.parquet",
    "samples": "source/samples_and_culture_branches.parquet",
    "shards": "source/shard_manifest.parquet",
}
METADATA_LIMIT = 4 * 1024**2
OUTPUT_LIMIT = 32 * 1024**2


class PreparationError(ValueError):
    """Metadata or admission does not satisfy this bridge's contract."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PreparationError("Invalid SHA-256 identity.")
    return value


def _relative(name: Any) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise PreparationError("Invalid relative metadata path.")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise PreparationError("Noncanonical or escaping metadata path.")
    return name


def _read_regular(path: Path, *, limit: int = METADATA_LIMIT) -> bytes:
    # Reject aliases at every existing component, not only the leaf.
    absolute = path.absolute()
    for parent in (absolute, *absolute.parents):
        if parent.is_symlink():
            raise PreparationError(f"Symlink forbidden: {parent}")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise PreparationError("Nonregular or oversized metadata input.")
        content = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if identity(before) != identity(after) or len(content) != before.st_size:
            raise PreparationError("Metadata changed during read.")
    return content


def _json(data: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PreparationError("Duplicate JSON key.")
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(PreparationError("Nonfinite JSON.")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError("Invalid metadata JSON.") from error


def _inventory(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = data.decode("utf-8").splitlines()
        for line in lines:
            checksum, name = line.split("  ", 1)
            _sha(checksum)
            _relative(name)
            if name in result:
                raise PreparationError("Duplicate package inventory member.")
            result[name] = checksum
    except (UnicodeError, ValueError) as error:
        raise PreparationError("Invalid package inventory.") from error
    if not result:
        raise PreparationError("Empty package inventory.")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PreparationError(f"Missing text field: {field}")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PreparationError(f"Invalid integer field: {field}")
    return value


def assemble_metadata(tables: dict[str, list[dict[str, Any]]], inventory: dict[str, str],
                      completion: dict[str, Any]) -> dict[str, Any]:
    """Pure semantic assembly; callers still need the hash-verified loading API."""
    if set(tables) != set(TABLES):
        raise PreparationError("Exact four metadata tables required.")
    sources = {f"{d}_{c}" for d in DONORS for c in CONDITIONS}
    sample_rows = tables["samples"]
    if len(sample_rows) != 12 or {r["source_id"] for r in sample_rows} != sources:
        raise PreparationError("Exactly twelve unique donor-condition sources required.")
    samples = {r["source_id"]: dict(r) for r in sample_rows}
    for source, row in samples.items():
        donor, condition = row["donor_id"], row["condition"]
        if donor not in DONORS or condition not in CONDITIONS or source != f"{donor}_{condition}":
            raise PreparationError("Source donor/condition identity mismatch.")
        elapsed, stimulation, branch = TIMING[condition]
        if (row["collection_elapsed_hours"], row["restimulation_hours"], row["branch"]) != (elapsed, stimulation, branch):
            raise PreparationError("Biological timing/branch disagreement.")
        if row["primary_methods_verified"] is not True:
            raise PreparationError("Timing methods metadata is not verified.")

    guides = []
    seen_guides: set[str] = set()
    for index, original in enumerate(tables["guides"]):
        row = dict(original)
        guide = _text(row["sgRNA"], "sgRNA")
        if guide in seen_guides or row["experimental_library_member"] is not True:
            raise PreparationError("Duplicate guide or incomplete design catalog.")
        seen_guides.add(guide)
        for name in ("primary_target_id", "seq"):
            _text(row[name], name)
        if type(row["is_design_ntc"]) is not bool or type(row["design_curated_disagreement"]) is not bool:
            raise PreparationError("Invalid guide control/annotation flags.")
        # Canonical NTC rows have an empty designed gene ID and a distinct
        # primary control sentinel. Do not manufacture a gene for them.
        if row["is_design_ntc"]:
            if not isinstance(row["designed_target_gene_id"], str):
                raise PreparationError("NTC designed gene annotation must preserve its string field.")
        else:
            _text(row["designed_target_gene_id"], "designed_target_gene_id")
        row.update(guide_index=index, target_partition_role="historical_annotation_not_new_split",
                   numerical_source_support="unknown_not_cellwise_compiled",
                   numerical_endpoint_support="unknown_not_cellwise_compiled")
        guides.append(row)
    if not guides:
        raise PreparationError("Empty guide catalog.")

    by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
    for row in tables["features"]:
        if row["source_id"] not in by_source:
            raise PreparationError("Feature metadata references an unknown source.")
        by_source[row["source_id"]].append(row)
    axis = None
    permutations = {}
    for source in sorted(sources):
        rows = sorted(by_source[source], key=lambda r: r["canonical_position"])
        n = len(rows)
        if not n or [r["canonical_position"] for r in rows] != list(range(n)):
            raise PreparationError("Missing, duplicate or noncontiguous canonical feature position.")
        if any(type(r["canonical_position"]) is not int or type(r["source_position"]) is not int
               or type(r["is_RNA"]) is not bool for r in rows):
            raise PreparationError("Invalid feature index or RNA flag.")
        if sorted(r["source_position"] for r in rows) != list(range(n)):
            raise PreparationError("Source feature permutation is not bijective.")
        if len({r["feature_id"] for r in rows}) != n:
            raise PreparationError("Duplicate feature ID.")
        signature = [{"canonical_position": r["canonical_position"],
                      "feature_id": _text(r["feature_id"], "feature_id"), "is_RNA": r["is_RNA"]}
                     for r in rows]
        if axis is not None and signature != axis:
            raise PreparationError("Canonical feature axes disagree across sources.")
        axis = signature
        permutations[source] = [r["source_position"] for r in rows]
    assert axis is not None
    if sum(r["is_RNA"] for r in axis) != completion["RNA_features"] or sum(not r["is_RNA"] for r in axis) != completion["technical_features"]:
        raise PreparationError("Feature counts disagree with producer completion.")

    shard_rows = []
    seen_shards: set[tuple[str, int]] = set()
    path_names: set[str] = set()
    total_cells = 0
    for original in tables["shards"]:
        row = dict(original)
        source = row["source_id"]
        shard = _integer(row["shard"], "shard")
        if source not in sources or (source, shard) in seen_shards:
            raise PreparationError("Unknown source or duplicate shard.")
        seen_shards.add((source, shard))
        for key, hash_key, folder, suffix in (("path", "sha256", "counts", ".csr.npz"),
                                               ("cells_path", "cells_sha256", "cells", ".parquet")):
            name = _relative(row[key])
            if name != f"source/{source}/{folder}/{shard:05d}{suffix}" or name in path_names:
                raise PreparationError("Shard path does not match source/index or is reused.")
            path_names.add(name)
            if inventory.get(name) != _sha(row[hash_key]):
                raise PreparationError("Shard metadata differs from authoritative inventory.")
        rows = _integer(row["rows"], "rows", minimum=1)
        for name in ("bytes", "cell_table_bytes", "nnz"):
            _integer(row[name], name, minimum=1)
        first = _integer(row["first_source_row"], "first_source_row")
        last = _integer(row["last_source_row"], "last_source_row")
        if last - first + 1 < rows or row["exact_reload"] is not True:
            raise PreparationError("Invalid source-row interval or producer reload flag.")
        total_cells += rows
        shard_rows.append(row)
    for source in sorted(sources):
        records = sorted((r for r in shard_rows if r["source_id"] == source), key=lambda r: r["shard"])
        if not records or [r["shard"] for r in records] != list(range(len(records))):
            raise PreparationError("Source has missing or noncontiguous shards.")
        if any(a["last_source_row"] >= b["first_source_row"] for a, b in zip(records, records[1:])):
            raise PreparationError("Overlapping or unordered native source-row intervals.")
        samples[source].update(
            producer_eligible_cells=sum(r["rows"] for r in records),
            producer_shards=len(records), source_row_axis="native_source_row_not_latent_row",
            new_study_donor_role={"D1": "reserved_internal_assessment_then_final_refit",
                                  "D2": "architecture_selection_then_final_refit",
                                  "D3": "development_fitting", "D4": "development_fitting"}[samples[source]["donor_id"]],
            numerical_access_status="not_admitted_by_metadata_manifest",
            delayed_endpoint_preparation=(source == "D1_Stim48hr"),
            denominator={"scope": "captured_donor_condition_aggregate",
                         "category_count": len(guides), "count_vector_status": "not_read_or_bound_by_this_manifest",
                         "denominator_total_status": "producer_cell_total_only_not_numerical_count_reconciliation"},
            state_support="unknown_until_per_cell_support_compilation",
            culture_capture_identity_status="unresolved_not_inferred_from_source_label")
    if total_cells != completion["cells"] or completion["sources"] != 12:
        raise PreparationError("Producer total/source count differs from shard metadata.")
    return {
        "schema": SCHEMA, "status": "metadata_bound_not_numerical_study",
        "counts": {"donors": 4, "sources": 12, "guides": len(guides),
                   "ntc_guides": sum(r["is_design_ntc"] for r in guides),
                   "designed_target_ids": len({r["designed_target_gene_id"] for r in guides if not r["is_design_ntc"]}),
                   "RNA_features": completion["RNA_features"], "technical_features": completion["technical_features"],
                   "producer_eligible_cells": total_cells, "producer_shards": len(shard_rows)},
        "sources": [samples[f"{d}_{c}"] for d in DONORS for c in CONDITIONS],
        "canonical_guides": guides, "canonical_features": axis,
        "source_feature_positions": permutations,
        "native_count_columns": "already_canonical_do_not_reapply_source_permutation",
        "shards": sorted(shard_rows, key=lambda r: (r["source_id"], r["shard"])),
        "temporal_contract": {
            "stimulated_edge": {"source": "Stim8hr", "endpoint": "Stim48hr", "elapsed_hours": 40},
            "Rest_role": "contemporaneous_unstimulated_sister_branch_not_time_zero",
            "same_cell_tracking": False, "rest_to_stim8_temporal_edge": False,
            "continuity": "destructive_sister_populations_not_verified_maintained_culture_ids"},
        "prospective_role_policy": {
            "authority": "separate_AUTHORITY_and_STUDY_PLAN_not_this_metadata_product",
            "architecture_fitting_donors": ["D3", "D4"],
            "architecture_selection_donors": ["D2"],
            "reserved_internal_assessment_donors": ["D1"],
            "D1_Stim48hr_numerical_preparation_prerequisites": [
                "architecture_selected", "terminal_checkpoints_fixed", "D1_forecasts_sealed"],
            "final_all_four_donor_refit_prerequisite": "D1_assessment_completed",
            "missing_source_encodings_may_not_be_launched_as_one_unrestricted_batch": True,
            "independent_new_donor_validation_claim": False},
        "representation": {
            "selection_status": "unbound_before_architecture_selection",
            "fresh_refit_candidate_sha256": "200d38d0e84b89ba7c5b84ca2d28b8c854c2e4ca33cfc705e12d6e84f98955c2",
            "fresh_refit_fit_sources": ["D3_Rest", "D3_Stim48hr", "D4_Rest", "D4_Stim48hr"],
            "historical_export_cells": 9400093,
            "all_twelve_source_export_established": False,
            "scope_warning": "D3_D4_endpoint_exposure_is_not_nested_inductive_validation",
            "numerical_checkpoint_or_state_read": False},
        "admission": {"numerical_study_compiled": False, "training_ready": False,
                      "donor_model_selection_roles_frozen": False,
                      "full_catalog_guidewise_state_support_verified": False,
                      "complete_count_vectors_bound": False,
                      "data_access_authority": "separate_new_study_authority_required_not_supplied_by_metadata",
                      "LARRY_access": False},
        "reads": {"count_payloads": 0, "RNA_payloads": 0, "cell_table_bodies": 0,
                  "states": 0, "checkpoints": 0, "remote_calls": 0},
    }


def build_manifest(native_root: Path = DEFAULT_IMPORT / "native",
                   import_receipt: Path = DEFAULT_IMPORT / "COMPLETE.json",
                   expected_import_sha256: str = IMPORT_SHA256) -> dict[str, Any]:
    """Read only six small exact metadata files and their import receipt."""
    receipt_bytes = _read_regular(import_receipt)
    if digest(receipt_bytes) != _sha(expected_import_sha256):
        raise PreparationError("Imported metadata receipt identity mismatch.")
    receipt = _json(receipt_bytes)
    if receipt["status"] != "verified_metadata_only_not_numerical_admission":
        raise PreparationError("Metadata import was not verified.")
    complete_bytes = _read_regular(native_root / "INPUT_DATA_COMPLETE.json")
    inventory_bytes = _read_regular(native_root / "SHA256SUMS")
    pinned = {"INPUT_DATA_COMPLETE.json": complete_bytes, "SHA256SUMS": inventory_bytes}
    for name, content in pinned.items():
        binding = receipt["files"][name]
        if digest(content) != binding["sha256"] or len(content) != binding["size_bytes"]:
            raise PreparationError("Native metadata/import identity mismatch.")
    if digest(complete_bytes) != receipt["completion_sha256"] or digest(inventory_bytes) != receipt["inventory_sha256"]:
        raise PreparationError("Receipt top-level identities disagree.")
    complete = _json(complete_bytes)
    if complete["inventory_sha256"] != digest(inventory_bytes):
        raise PreparationError("Completion/inventory identity mismatch.")
    inventory = _inventory(inventory_bytes)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tables = {}
    bindings = {}
    for label, name in TABLES.items():
        content = _read_regular(native_root / name)
        binding = receipt["files"][name]
        checksum = digest(content)
        if checksum != inventory.get(name) or checksum != binding["sha256"] or len(content) != binding["size_bytes"]:
            raise PreparationError(f"Metadata table identity mismatch: {name}")
        tables[label] = pq.read_table(io.BytesIO(content), use_threads=False).to_pylist()
        bindings[name] = {"sha256": checksum, "size_bytes": len(content)}
    result = assemble_metadata(tables, inventory, complete)
    # Production identities are fixed; alternate inputs are fixture-only API use.
    if expected_import_sha256 == IMPORT_SHA256:
        expected = {"donors": 4, "sources": 12, "guides": 26504, "ntc_guides": 992,
                    "designed_target_ids": 12779, "RNA_features": 18129, "technical_features": 1,
                    "producer_eligible_cells": 21996842, "producer_shards": 1826}
        if result["counts"] != expected or digest(complete_bytes) != COMPLETION_SHA256 or digest(inventory_bytes) != INVENTORY_SHA256:
            raise PreparationError("Production native package differs from frozen expected metadata.")
    result["input_binding"] = {
        "metadata_root": str(native_root.absolute()), "import_receipt": str(import_receipt.absolute()),
        "import_receipt_sha256": digest(receipt_bytes), "native_completion_sha256": digest(complete_bytes),
        "native_inventory_sha256": digest(inventory_bytes), "metadata_tables": bindings,
        "historical_native_payload_root": REMOTE_NATIVE_ROOT,
        "payload_root_live_availability_verified_here": False,
        "native_inventory_is_not_local_payload_availability": True}
    return result


def require_numerical_admission(manifest: dict[str, Any]) -> None:
    """A metadata product can NEVER stand in for a separately compiled study."""
    raise PreparationError("Metadata bridge is not numerical training admission; bind source laws, counts, representation and splits in a separately qualified native study.")


def publish_manifest(manifest: dict[str, Any], output: Path) -> str:
    content = canonical(manifest)
    if len(content) > OUTPUT_LIMIT:
        raise PreparationError("Metadata output exceeds 32 MiB.")
    if not output.parent.is_dir() or any(p.is_symlink() for p in (output, *output.absolute().parents)):
        raise PreparationError("Output requires an existing non-symlink parent and path.")
    with output.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if _read_regular(output, limit=OUTPUT_LIMIT) != content:
        raise PreparationError("Published metadata readback differs.")
    return digest(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New absent JSON output; no overwrite or parent creation.")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise PreparationError("Refuse existing output before reading metadata.")
    manifest = build_manifest()
    checksum = publish_manifest(manifest, args.output)
    print(json.dumps({"output": str(args.output), "sha256": checksum,
                      "status": manifest["status"], "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
