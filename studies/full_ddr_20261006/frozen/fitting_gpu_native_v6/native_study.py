"""External GSE314342 adapter into the actual main-CREDO native-v4 contract.

Inputs are already authorized, hash-bound count vectors and frozen latent laws.
This module does not discover files, read count payloads, train a representation,
or grant outcome access.  Unprovided observations remain missing, never zero.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from credo.contracts import SplitSpec
from credo.data import (
    AbundanceChannelSpec,
    AbundanceSemantics,
    AbundanceTable,
    AxisSpec,
    Checkpoint,
    ContextTable,
    InterventionEventTable,
    LPSCompositionTable,
    LongitudinalDesign,
    PerturbationComponentTable,
    PerturbationEffectBindingTable,
    PerturbationReferenceBindingTable,
    PerturbationTable,
    PerturbSeqStudyBuilder,
    PopulationSeriesTable,
    RepresentationCatalog,
    RepresentationSpec,
    SelectionSpec,
    SnapshotObservationTable,
    SupportIndexTable,
    SupportRef,
    Transition,
)
from credo.lps.study import PerturbSeqManifest
from credo.registry import get_recipe
from credo.runtime import validate_view_for_recipe

DONORS = ("D1", "D2", "D3", "D4")
CHECKPOINTS = ("Stim8hr", "Stim48hr")
MODEL_CHANNEL = "regularized_captured_frequency"
RAW_CHANNEL = "captured_count"
EFFECT_BINDING = "actual_guide_to_target"
REFERENCE_BINDING = "donor_checkpoint_ntc"


def _hash(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256.")
    return value


def count_vector_sha256(counts: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(counts), separators=(",", ":")).encode()
    ).hexdigest()


def series_id(donor: str, guide_index: int) -> str:
    return f"{donor}::guide-index-{guide_index}"


def observation_id(donor: str, guide_index: int, checkpoint: str) -> str:
    return f"{series_id(donor, guide_index)}::{checkpoint}"


@dataclass(frozen=True)
class Guide:
    guide_id: str
    target_id: str
    is_control: bool


@dataclass(frozen=True)
class CountObservation:
    source_id: str
    donor: str
    checkpoint: str
    counts: tuple[int, ...]
    vector_sha256: str
    source_artifact_sha256: str


@dataclass(frozen=True)
class NativeStudyBundle:
    study: Any
    source_supported_series: tuple[str, ...]
    dispositions: tuple[dict[str, Any], ...]


def build_native_study(
    *,
    study_id: str,
    guides: Sequence[Guide],
    observations: Sequence[CountObservation],
    support_records: Mapping[tuple[str, int], Mapping[str, Any]],
    supports: Any,
    representation: RepresentationSpec,
    authority_sha256: str,
    representation_provenance: Mapping[str, Any],
) -> NativeStudyBundle:
    """Build and semantically validate one exact all-catalog native study.

    The caller must apply PhaseAccess before constructing observations/store.
    Missing D1 endpoints can be represented without passing their count vector
    or creating a latent reference. No outcome-derived eligibility is used.
    """
    _hash(authority_sha256, "authority_sha256")
    guides = tuple(guides)
    if not guides or len({g.guide_id for g in guides}) != len(guides):
        raise ValueError("Catalog must be nonempty with unique actual guide IDs.")
    if any(
        not g.guide_id or not g.target_id or type(g.is_control) is not bool
        for g in guides
    ):
        raise ValueError(
            "Guide identities, targets and boolean control status are required."
        )
    if not any(g.is_control for g in guides):
        raise ValueError("Actual NTC guides must be retained.")
    if (
        representation.space_kind != "latent"
        or representation.support_store_id != supports.store_id
    ):
        raise ValueError("Frozen latent representation/store identity mismatch.")
    if any(
        getattr(representation, name) is None
        for name in (
            "encoder_artifact",
            "normalization_artifact",
            "feature_artifact",
            "support_artifact",
        )
    ):
        raise ValueError(
            "Frozen encoder, normalization, feature and latent-support artifact identities are required."
        )
    if set(representation_provenance.get("fit_subjects", ())) != {"D3", "D4"} or set(
        representation_provenance.get("fit_conditions", ())
    ) != {"Rest", "Stim48hr"}:
        raise ValueError(
            "This study reuses the explicitly bound D3/D4 Rest/Stim48hr frozen representation."
        )
    if set(representation.fit_subject_ids) != {"D3", "D4"} or set(
        representation.fit_checkpoint_ids
    ) != {"Stim48hr"}:
        raise ValueError(
            "Representation within-study fit identities disagree with original provenance."
        )
    if supports.dimension(representation.representation_id) != representation.dimension:
        raise ValueError("Frozen latent dimension mismatch.")
    observed = {}
    source_ids = set()
    frequency_denominators = {}
    for value in observations:
        key = (value.donor, value.checkpoint)
        if (
            value.donor not in DONORS
            or value.checkpoint not in CHECKPOINTS
            or key in observed
        ):
            raise ValueError(
                "Duplicate or unsupported donor/checkpoint count observation."
            )
        if value.source_id != f"{value.donor}_{value.checkpoint}":
            raise ValueError(
                "Canonical source identity must match its exact donor/checkpoint; role relabeling is forbidden."
            )
        if value.source_id in source_ids:
            raise ValueError(
                "Each observation requires a distinct bound source identity."
            )
        if len(value.counts) != len(guides) or any(
            type(n) is not int or n < 0 for n in value.counts
        ):
            raise ValueError(
                "Complete-catalog nonnegative exact integer counts are required."
            )
        total = sum(value.counts)
        if total == 0:
            raise ValueError(
                "An entirely empty capture is not a frequency observation."
            )
        _hash(value.source_artifact_sha256, "source_artifact_sha256")
        if count_vector_sha256(value.counts) != _hash(
            value.vector_sha256, "vector_sha256"
        ):
            raise ValueError("Count vector content does not match its bound identity.")
        observed[key] = value
        source_ids.add(value.source_id)
        # Complete-catalog totals are computed once per source, not inside G
        # guide loops: the previous repeated reductions required O(G**2) work.
        frequency_denominators[value.source_id] = total + 0.5 * len(guides)
    expected_support_keys = {
        (value.source_id, index)
        for value in observations
        for index in range(len(guides))
    }
    if set(support_records) != expected_support_keys:
        raise ValueError(
            "Support metadata must cover exactly the admitted source/catalog records."
        )

    perturbations, components, effects, references = [], [], [], []
    for index, guide in enumerate(guides):
        pid = f"actual-guide-{index}"
        effect = (
            "shared-NTC-reference" if guide.is_control else f"target::{guide.target_id}"
        )
        perturbations.append(
            dict(
                perturbation_id=pid,
                perturbation_kind="crispr_i",
                is_control=guide.is_control,
                control_kind="non_targeting" if guide.is_control else "none",
            )
        )
        components.append(
            dict(
                perturbation_id=pid,
                component_id=f"component-{index}",
                construct_id=guide.guide_id,
                target_id=guide.target_id,
                component_kind="crispr_i",
                dose=None,
                dose_unit=None,
                component_order=0,
            )
        )
        effects.append(
            dict(
                binding_id=EFFECT_BINDING,
                perturbation_id=pid,
                effect_id=effect,
                parameterization_kind="target_shared",
            )
        )
        references.append(
            dict(
                binding_id=REFERENCE_BINDING,
                perturbation_id=pid,
                reference_pool_id="shared-model-NTC-reference",
                scope_kind="subject",
                match_keys='["subject_id","checkpoint_id"]',
                counterfactual_effect_id="shared-NTC-reference",
            )
        )

    series, events, obs_rows, support_rows, abundance, compositions, dispositions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    active = []
    for donor in DONORS:
        for index, guide in enumerate(guides):
            sid = series_id(donor, index)
            series.append(
                dict(
                    series_id=sid,
                    subject_id=donor,
                    experimental_unit_id=f"unresolved-culture::{donor}",
                    perturbation_id=f"actual-guide-{index}",
                    context_trajectory_id=f"captured-aggregate::{donor}",
                    biological_replicate_id=f"donor::{donor}",
                    continuity_kind="matched_subject_parallel",
                )
            )
            for role, agent, coordinate in (
                ("primary_perturbation", f"actual-guide-{index}", None),
                ("background_stimulus", "restimulation", 0.0),
            ):
                events.append(
                    dict(
                        event_id=f"{sid}::{role}",
                        series_id=sid,
                        agent_id=agent,
                        event_kind="crispr_i"
                        if coordinate is None
                        else "restimulation",
                        modeled_role=role,
                        start_coordinate=coordinate,
                        end_coordinate=None,
                        start_relation="before_source",
                        persistent=True,
                        dose=None,
                        dose_unit=None,
                    )
                )
            source = observed.get((donor, "Stim8hr"))
            source_frequency = (
                None
                if source is None
                else (source.counts[index] + 0.5)
                / frequency_denominators[source.source_id]
            )
            for checkpoint in CHECKPOINTS:
                oid = observation_id(donor, index, checkpoint)
                value = observed.get((donor, checkpoint))
                record = (
                    None if value is None else support_records[(value.source_id, index)]
                )
                geometry = False
                ref = None
                if record is not None:
                    if (
                        record.get("source_id") != value.source_id
                        or type(record.get("guide_index")) is not int
                        or record.get("guide_index") != index
                    ):
                        raise ValueError("Support source/guide metadata mismatch.")
                    if record.get("support_key") != f"{value.source_id}::{index}":
                        raise ValueError(
                            "Support key must identify this exact canonical source/guide population."
                        )
                    captured, n_geometry = (
                        record.get("captured_cells"),
                        record.get("geometry_cells"),
                    )
                    if (
                        type(captured) is not int
                        or type(n_geometry) is not int
                        or captured != value.counts[index]
                        or not 0 <= n_geometry <= captured
                    ):
                        raise ValueError(
                            "Support geometry/captured counts disagree with the count population."
                        )
                    if (type(record.get("geometry_absent_cells")) is not int
                            or record["geometry_absent_cells"] != captured - n_geometry):
                        raise ValueError("Unavailable geometry count differs from captured support.")
                    geometry = n_geometry > 0
                    if record.get("available") is not geometry:
                        raise ValueError(
                            "Support availability must follow positive geometry count."
                        )
                    if geometry:
                        ref = SupportRef(
                            supports.store_id,
                            representation.representation_id,
                            record["support_key"],
                        )
                        if not supports.contains(ref):
                            raise ValueError(
                                "Declared state law is absent from the frozen store."
                            )
                if checkpoint == "Stim8hr" and geometry:
                    active.append(sid)
                denominator = f"captured::{donor}::{checkpoint}"
                obs_rows.append(
                    dict(
                        observation_id=oid,
                        series_id=sid,
                        checkpoint_id=checkpoint,
                        sample_id=value.source_id
                        if value
                        else f"unavailable::{donor}::{checkpoint}",
                        geometry_observed=geometry,
                        abundance_observed=value is not None,
                        context_id=f"captured-aggregate::{donor}",
                        assay_id="guide_resolved_RNA",
                        assignment_qc="source_QC_exactly_one_guide",
                    )
                )
                support_rows.append(
                    dict(
                        observation_id=oid,
                        representation_id=representation.representation_id,
                        store_id=None if ref is None else ref.store_id,
                        support_key=None if ref is None else ref.support_key,
                        available=geometry,
                    )
                )
                for channel in (RAW_CHANNEL, MODEL_CHANNEL):
                    number = None if value is None else value.counts[index]
                    if value is not None and channel == MODEL_CHANNEL:
                        number = (value.counts[index] + 0.5) / frequency_denominators[value.source_id]
                    abundance.append(
                        dict(
                            observation_id=oid,
                            channel_id=channel,
                            value=number,
                            observed=value is not None,
                            denominator_id=denominator,
                            transform_id="half_count_catalog_regularization_v1"
                            if channel == MODEL_CHANNEL
                            else None,
                            source_artifact_id=None
                            if value is None
                            else value.source_artifact_sha256,
                        )
                    )
                if (
                    checkpoint == "Stim48hr"
                    and value is not None
                    and source is not None
                ):
                    compositions.append(
                        dict(
                            composition_block_id=denominator,
                            checkpoint_id=checkpoint,
                            context_id=f"captured-aggregate::{donor}",
                            series_id=sid,
                            observation_id=oid,
                            exposure=source_frequency,
                            count=value.counts[index],
                            denominator_id=denominator,
                            block_kind="capture_stratum",
                            background_fitness=0.0,
                        )
                    )
                dispositions.append(
                    dict(
                        donor=donor,
                        checkpoint=checkpoint,
                        guide_index=index,
                        guide_id=guide.guide_id,
                        target_id=guide.target_id,
                        count_observed=value is not None,
                        geometry_observed=geometry,
                        geometry_status="available"
                        if geometry
                        else (
                            "unavailable_observation"
                            if value is None
                            else "no_positive_RNA_state"
                        ),
                        source_modeled_mass=source_frequency,
                    )
                )

    channels = [
        AbundanceChannelSpec(
            RAW_CHANNEL,
            AbundanceSemantics.CAPTURE_COUNT,
            unit="captured_cells",
            denominator_scope="custom",
        ),
        AbundanceChannelSpec(
            MODEL_CHANNEL,
            AbundanceSemantics.RELATIVE,
            unit="regularized_captured_fraction",
            denominator_scope="custom",
            zero_policy="forbidden",
            transform_id="half_count_catalog_regularization_v1",
            input_channel_id=RAW_CHANNEL,
            transform_parameters={"alpha": 0.5, "catalog_size": len(guides)},
        ),
    ]
    study = PerturbSeqStudyBuilder(
        manifest=PerturbSeqManifest(
            4,
            study_id,
            "bound_count_native_and_frozen_latents_v1",
            representation.representation_id,
            MODEL_CHANNEL,
            "Captured-relative sister-population continuation; no absolute growth/ecology claim.",
            EFFECT_BINDING,
            REFERENCE_BINDING,
        ),
        design=LongitudinalDesign(
            AxisSpec("hours", "physical_time", "hours"),
            (
                Checkpoint("Stim8hr", {"hours": 8.0}, "source"),
                Checkpoint("Stim48hr", {"hours": 48.0}, "target"),
            ),
            (Transition("stimulated_continuation", "Stim8hr", "Stim48hr"),),
        ),
        perturbations=PerturbationTable(pd.DataFrame(perturbations)),
        perturbation_components=PerturbationComponentTable(pd.DataFrame(components)),
        intervention_events=InterventionEventTable(pd.DataFrame(events)),
        series=PopulationSeriesTable(pd.DataFrame(series)),
        observations=SnapshotObservationTable(pd.DataFrame(obs_rows)),
        contexts=ContextTable(
            pd.DataFrame(
                [
                    dict(
                        context_id=f"captured-aggregate::{d}",
                        context_kind="captured_aggregate_not_physical_pool",
                        subject_id=d,
                    )
                    for d in DONORS
                ]
            )
        ),
        representations=RepresentationCatalog((representation,)),
        supports=supports,
        support_index=SupportIndexTable(pd.DataFrame(support_rows)),
        abundance=AbundanceTable(pd.DataFrame(abundance), channels),
        compositions=LPSCompositionTable(
            pd.DataFrame(
                compositions,
                columns=(*LPSCompositionTable.required_columns, "background_fitness"),
            )
        ),
        effect_bindings=PerturbationEffectBindingTable(pd.DataFrame(effects)),
        reference_bindings=PerturbationReferenceBindingTable(pd.DataFrame(references)),
        provenance={
            "adapter": "gse314342_all4_native_v1",
            "authority_sha256": authority_sha256,
            "representation_original_fit_scope": dict(representation_provenance),
            "counts": [
                dict(
                    source_id=v.source_id,
                    donor=v.donor,
                    checkpoint=v.checkpoint,
                    vector_sha256=v.vector_sha256,
                    source_artifact_sha256=v.source_artifact_sha256,
                )
                for v in observations
            ],
            "physical_unit_status": "donor_aggregate_culture_identity_unresolved",
            "ecology_enabled": False,
            "absolute_mass_loss_permitted": False,
            "source_count_exposure_equals_initial_model_mass": True,
            "count_only_complement_policy": "regularized_source_persistence",
            "outcome_access_granted_by_adapter": False,
        },
    ).build(verify="semantic")
    return NativeStudyBundle(study, tuple(active), tuple(dispositions))


def compile_development(bundle: NativeStudyBundle, config: Any):
    """Compile D3/D4 fitting and D2 selection only; D1 is never selected here."""
    if config.model.context != "none" or config.training.epochs.context != 0:
        raise ValueError("Observed-aggregate study does not admit ecology.")
    if (
        config.loss.mass != 0
        or config.loss.mass_increment != "particle_log_mass"
        or config.loss.count_likelihood != "categorical_ce"
    ):
        raise ValueError(
            "Study requires categorical relative counts, exact particle mass and zero absolute mass loss."
        )
    frame = bundle.study.series.to_pandas()
    effects = bundle.study.effect_bindings.to_pandas().set_index("perturbation_id")[
        "effect_id"
    ]
    supported = frame.loc[frame.series_id.isin(bundle.source_supported_series)].copy()
    supported["effect_id"] = supported.perturbation_id.map(effects)
    fitted_effects = set(
        supported.loc[supported.subject_id.isin(("D3", "D4")), "effect_id"]
    )
    selected = tuple(
        supported.loc[
            supported.subject_id.isin(("D2", "D3", "D4"))
            & supported.effect_id.isin(fitted_effects),
            "series_id",
        ]
    )
    view = bundle.study.view(
        SelectionSpec(series_ids=selected, composition_policy="preserve_background")
    )
    selected_recipe = get_recipe("credo.compact_sde_v3@3.0")
    split = selected_recipe.plan_split(
        view,
        config,
        SplitSpec(
            strategy="subject",
            train_values=("D3", "D4"),
            validation_values=("D2",),
            representation_scope="shared",
        ),
    )
    validate_view_for_recipe(
        view, split, selected_recipe.requirements(config)
    ).raise_for_errors()
    return selected_recipe.compile(view, split, config), split
