"""Exact derived-state/count inputs; no encoder, raw scan, or D1 access."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

from credo.data import ArtifactRef, RepresentationSpec
from latent_support import (
    DevelopmentLatentSupportStore, FileIdentity, FrozenLatentBinding,
    FrozenLatentSupportStore, PhaseAccess, RepresentationIdentity, _json, _verify_file,
)
from native_study import CountObservation, Guide, build_native_study, count_vector_sha256


def checked_metadata(path, expected):
    path = Path(path)
    info = path.lstat()
    if info.st_size > 64 * 1024**2:
        raise ValueError("Bounded metadata required")
    _verify_file(FileIdentity(path, expected, info.st_size))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Metadata changed")
    return _json(raw)


def validate_contract(contract):
    if (contract.get("schema") != "credo.all4_ddr.development_inputs.v1"
            or contract.get("D1_access") is not False or contract.get("LARRY_access") is not False
            or contract.get("source_frequency_pseudocount") != 0.5
            or contract.get("dimension") != 48 or contract.get("guide_count") != 26504):
        raise ValueError("Incorrect development input contract")
    sources = []
    for store in contract["stores"]:
        PhaseAccess(store["phase"]).admit(tuple(store["source_ids"]))
        sources.extend(store["source_ids"])
    expected = {f"{d}_{c}" for d in ("D2", "D3", "D4") for c in ("Stim8hr", "Stim48hr")}
    if len(sources) != len(set(sources)) or set(sources) != expected | {"D3_Rest", "D4_Rest"}:
        raise ValueError("Exact development sources required, without role relabeling")


def frozen_binding(contract, entry):
    root = Path(entry["root"])
    complete = checked_metadata(root / "COMPLETE.json", entry["completion_sha256"])
    if complete.get("schema") == "credo.all4_ddr.latent_source.v1":
        artifacts = {name: dict(sha256=row["sha256"], size_bytes=row["bytes"])
                     for name, row in complete["files"].items()}
    else:
        artifacts = {row["relative_uri"]: row for row in complete["artifacts"]}
    files = [FileIdentity(root / name, artifacts[name]["sha256"], artifacts[name]["size_bytes"])
             for name in ("latent.h5", "cells.parquet", "support.parquet")]
    fitted = Path(contract["fitted_root"]) / "COMPLETE.json"
    representation = RepresentationIdentity(contract["representation_id"], contract["representation_sha256"],
                                             contract["model_sha256"], contract["model_sha256"])
    return FrozenLatentBinding(entry["store_id"], *files,
        completion=FileIdentity(root / "COMPLETE.json", entry["completion_sha256"], (root / "COMPLETE.json").stat().st_size),
        representation_record=FileIdentity(fitted, contract["fitted_complete_sha256"], fitted.stat().st_size),
        representation=representation, source_ids=tuple(entry["source_ids"]),
        guide_count=contract["guide_count"], dimension=contract["dimension"],
        retained_cells=entry["retained_cells"], geometry_cells=entry["geometry_cells"])


@contextmanager
def prepared_development(contract, metadata, *, authority_sha256, input_sha256, metadata_sha256):
    """Hash/validate once, then compile all-guide laws from bounded CPU snapshots.

    Yielded objects remain provisional until EXIT verifies all snapshot paths.
    A caller must publish its completion only after leaving this context.
    """
    validate_contract(contract)
    stores = []
    try:
        for entry in contract["stores"]:
            binding = frozen_binding(contract, entry)
            stores.append(FrozenLatentSupportStore(binding, PhaseAccess(entry["phase"]),
                                                   expected_representation=binding.representation))
        union = DevelopmentLatentSupportStore("gse314342_development_frozen", stores)
        with union.cached_snapshot(maximum_bytes=contract["maximum_snapshot_bytes"]):
            rows = metadata["canonical_guides"]
            if len(rows) != contract["guide_count"] or [r["guide_index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Canonical guide order differs")
            guides = tuple(Guide(r["sgRNA"], "__NTC__" if r["is_design_ntc"] else r["primary_target_id"],
                                 r["is_design_ntc"]) for r in rows)
            observations, support = [], {}
            by_source = {source: store for store in stores for source in store.binding.source_ids}
            for donor in ("D2", "D3", "D4"):
                for checkpoint in ("Stim8hr", "Stim48hr"):
                    source = f"{donor}_{checkpoint}"
                    records = [union.support_record(source, i) for i in range(len(guides))]
                    counts = tuple(r["captured_cells"] for r in records)
                    observations.append(CountObservation(source, donor, checkpoint, counts,
                        count_vector_sha256(counts), by_source[source].binding.completion.sha256))
                    support.update({(source, i): r for i, r in enumerate(records)})
            model = ArtifactRef("frozen-r48-model", str(Path(contract["fitted_root"]) / "model.safetensors"),
                                contract["model_sha256"], None, "application/octet-stream")
            inputs = ArtifactRef("full-ddr-development-inputs", "DEVELOPMENT_INPUTS.json", input_sha256,
                                 None, "application/json")
            features = ArtifactRef("canonical-gene-guide-metadata", "METADATA_PREPARATION.json", metadata_sha256,
                                   None, "application/json")
            rep = RepresentationSpec(contract["representation_id"], "frozen-r48", "latent", contract["dimension"],
                union.store_id, encoder_artifact=model, normalization_artifact=model, feature_artifact=features,
                support_artifact=inputs, fit_subject_ids=("D3", "D4"), fit_checkpoint_ids=("Stim48hr",),
                fit_observation_scope="custom")
            bundle = build_native_study(study_id="gse314342_all4_full_ddr_20261006_v1", guides=guides,
                observations=observations, support_records=support, supports=union, representation=rep,
                authority_sha256=authority_sha256, representation_provenance={"fit_subjects": ["D3", "D4"],
                    "fit_conditions": ["Rest", "Stim48hr"], "frozen_checkpoint": contract["model_sha256"]})
            yield bundle, observations
    finally:
        for store in stores:
            store.close()
