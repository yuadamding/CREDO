"""Fitting-only projection for device qualification; no D2 numerical opens.

The existing endpoint container also contains Rest. Its complete byte identity
and indexing are verified, but only Stim8hr/Stim48hr laws are compiled or used.
Original containers, counts, representations and role labels are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from credo.contracts import SplitSpec
from credo.data import ArtifactRef, RepresentationSpec, SelectionSpec
from credo.registry import get_recipe
from credo.runtime import validate_view_for_recipe
from development_inputs import frozen_binding, validate_contract
from latent_support import DevelopmentLatentSupportStore, FrozenLatentSupportStore, PhaseAccess
from native_study import CountObservation, Guide, build_native_study, count_vector_sha256


def fitting_entries(contract):
    validate_contract(contract)  # Metadata only, including excluded-role labels.
    entries = tuple(row for row in contract["stores"] if row["phase"] == "fitting")
    sources = [source for row in entries for source in row["source_ids"]]
    if set(sources) != {f"{d}_{c}" for d in ("D3", "D4") for c in ("Rest", "Stim8hr", "Stim48hr")}:
        raise ValueError("Exact fitting container projection required")
    for row in entries:
        PhaseAccess("fitting").admit(row["source_ids"])
    return entries


@contextmanager
def prepared_fitting(contract, metadata, *, authority_sha256, input_sha256, metadata_sha256):
    stores = []
    try:
        for entry in fitting_entries(contract):
            binding = frozen_binding(contract, entry)
            stores.append(FrozenLatentSupportStore(binding, PhaseAccess("fitting"),
                expected_representation=binding.representation))
        union = DevelopmentLatentSupportStore("gse314342_fitting_only_frozen", stores)
        with union.cached_snapshot(maximum_bytes=contract["maximum_snapshot_bytes"]):
            rows = metadata["canonical_guides"]
            if len(rows) != contract["guide_count"] or [r["guide_index"] for r in rows] != list(range(len(rows))):
                raise ValueError("Canonical guide order changed")
            guides = tuple(Guide(r["sgRNA"], "__NTC__" if r["is_design_ntc"] else r["primary_target_id"],
                r["is_design_ntc"]) for r in rows)
            observations, support = [], {}
            by_source = {source: store for store in stores for source in store.binding.source_ids}
            for donor in ("D3", "D4"):
                for condition in ("Stim8hr", "Stim48hr"):
                    source = f"{donor}_{condition}"
                    records = [union.support_record(source, i) for i in range(len(guides))]
                    counts = tuple(r["captured_cells"] for r in records)
                    observations.append(CountObservation(source, donor, condition, counts,
                        count_vector_sha256(counts), by_source[source].binding.completion.sha256))
                    support.update({(source, i): row for i, row in enumerate(records)})
            model = ArtifactRef("frozen-r48-model", str(Path(contract["fitted_root"]) / "model.safetensors"),
                contract["model_sha256"], None, "application/octet-stream")
            inputs = ArtifactRef("full-ddr-development-inputs", "DEVELOPMENT_INPUTS.json", input_sha256,
                None, "application/json")
            features = ArtifactRef("canonical-gene-guide-metadata", "METADATA_PREPARATION.json", metadata_sha256,
                None, "application/json")
            rep = RepresentationSpec(contract["representation_id"], "frozen-r48", "latent", 48,
                union.store_id, encoder_artifact=model, normalization_artifact=model, feature_artifact=features,
                support_artifact=inputs, fit_subject_ids=("D3", "D4"), fit_checkpoint_ids=("Stim48hr",),
                fit_observation_scope="custom")
            bundle = build_native_study(study_id="gse314342_gpu_native_fitting_qualification_v1", guides=guides,
                observations=observations, support_records=support, supports=union, representation=rep,
                authority_sha256=authority_sha256, representation_provenance={"fit_subjects": ["D3", "D4"],
                    "fit_conditions": ["Rest", "Stim48hr"], "frozen_checkpoint": contract["model_sha256"]})
            yield bundle, observations
    finally:
        for store in stores:
            store.close()


def compile_fitting_qualification(bundle, config):
    """Self-eval partition is a compiler necessity, never evaluated or trained."""
    if config.model.context != "none" or config.validation.fraction != 0:
        raise ValueError("Qualification cannot activate context or a holdout")
    view = bundle.study.view(SelectionSpec(series_ids=bundle.source_supported_series,
        composition_policy="preserve_background"))
    recipe = get_recipe("credo.compact_sde_v3@3.0")
    split = recipe.plan_split(view, config, SplitSpec(strategy="none", representation_scope="shared"))
    if split.source != "train_self_eval":
        raise ValueError("No new split/search in qualification")
    validate_view_for_recipe(view, split, recipe.requirements(config)).raise_for_errors()
    problem = recipe.compile(view, split, config)
    if set(problem.training.measure_meta.sample_id.astype(str)) != {"D3", "D4"}:
        raise ValueError("Non-fitting donor in device qualification")
    return problem
