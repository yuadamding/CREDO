"""Recipe-owned compilation from semantic Study views to trajectory compatibility data."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..contracts import (
    Axis,
    FiniteMeasure,
    FixedContextBackground,
    MassSemantics,
    RepresentationArtifact,
    TrajectoryData,
)
from ..data.splits import SplitPlan
from ..data.study import StudyView
from ..data.support import SupportRef
from ..data.tables import AbundanceSemantics
from ..problems import (
    CompiledLPSSplit,
    CompiledObservationSet,
    FiniteMeasureDynamicsProblem,
)
from .compact_sde_v3.objective import CountBlock


def _pooled_observation_id(
    series_id: str,
    checkpoint_id: str,
    observation_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha256("|".join(observation_ids).encode()).hexdigest()[:12]
    return f"pooled::{series_id}@{checkpoint_id}:{digest}"


def _axis(view: StudyView) -> Axis:
    design = view.study.design
    if len(design.axes) != 1:
        raise ValueError("Trajectory recipes require exactly one study axis.")
    axis_spec = design.axes[0]
    kind = {"physical_time": "physical", "effect": "effect"}.get(axis_spec.kind)
    if kind is None:
        raise ValueError(f"Trajectory recipes cannot compile axis kind {axis_spec.kind!r}.")
    labels = design.ordered_checkpoint_ids
    values = tuple(
        float(design.checkpoint(label).coordinates[axis_spec.axis_id]) for label in labels
    )
    return Axis(kind=kind, source=labels[0], labels=labels, values=values)


def _mass_semantics(view: StudyView) -> MassSemantics:
    if view.abundance_channel is None or view.study.abundance is None:
        return MassSemantics.UNIT
    semantics = view.study.abundance.channels[view.abundance_channel].semantics
    try:
        return {
            AbundanceSemantics.ABSOLUTE: MassSemantics.ABSOLUTE,
            AbundanceSemantics.RELATIVE: MassSemantics.RELATIVE_WITHIN_GROUP,
            AbundanceSemantics.CAPTURE_COUNT: MassSemantics.CAPTURED_COUNT,
            AbundanceSemantics.UNIT: MassSemantics.UNIT,
        }[semantics]
    except KeyError as exc:
        raise ValueError(
            f"Trajectory recipes cannot compile abundance semantics {semantics.value!r}."
        ) from exc


class _CompiledCheckpointMeasures(Mapping[str, FiniteMeasure]):
    def __init__(self, owner: _CompiledMeasures, checkpoint_id: str) -> None:
        self._owner = owner
        self._checkpoint_id = checkpoint_id

    def __getitem__(self, series_id: str) -> FiniteMeasure:
        return self._owner.measure(self._checkpoint_id, str(series_id))

    def __iter__(self) -> Iterator[str]:
        return iter(self._owner.series_by_checkpoint[self._checkpoint_id])

    def __len__(self) -> int:
        return len(self._owner.series_by_checkpoint[self._checkpoint_id])


class _CompiledMeasures(Mapping[str, Mapping[str, FiniteMeasure]]):
    """Lazy finite-measure adapter over one representation and abundance channel.

    Measures are materialized at most once per compiled dataset.  The cached
    arrays are immutable and own no writable alias, so returning the same
    ``FiniteMeasure`` on later epochs cannot let a caller corrupt subsequent
    training batches.
    """

    is_lazy = True

    def __init__(self, view: StudyView, axis: Axis, mass_semantics: MassSemantics) -> None:
        self.view = view
        self.axis = axis
        self.mass_semantics = mass_semantics
        self.latent_dim = view.representation.dimension
        observations = view.observations()
        support_index = view.study.support_index._unsafe_view()
        support_index = support_index.loc[
            support_index["representation_id"].eq(view.representation_id)
            & support_index["observation_id"].isin(observations["observation_id"])
        ]
        available = support_index.loc[support_index["available"]]
        available_ids = set(available["observation_id"])
        observations = observations.loc[observations["observation_id"].isin(available_ids)]
        self._observation = observations.set_index(["checkpoint_id", "series_id"]).sort_index()
        self._support = available.set_index("observation_id")
        abundance = view.abundance()
        self._abundance = (
            abundance.set_index("observation_id") if len(abundance) else pd.DataFrame()
        )
        order = {series_id: index for index, series_id in enumerate(view.series_ids)}
        self.series_by_checkpoint = {
            checkpoint_id: tuple(
                sorted(
                    set(
                        observations.loc[
                            observations["checkpoint_id"].eq(checkpoint_id), "series_id"
                        ].astype(str)
                    ),
                    key=order.__getitem__,
                )
            )
            for checkpoint_id in axis.labels
        }
        self._views = {
            checkpoint_id: _CompiledCheckpointMeasures(self, checkpoint_id)
            for checkpoint_id in axis.labels
        }
        self._measure_cache: dict[tuple[str, str], FiniteMeasure] = {}
        self._cache_lock = threading.RLock()

    def __getitem__(self, checkpoint_id: str) -> Mapping[str, FiniteMeasure]:
        return self._views[str(checkpoint_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._views)

    def __len__(self) -> int:
        return len(self._views)

    def measure(self, checkpoint_id: str, series_id: str) -> FiniteMeasure:
        key = (str(checkpoint_id), str(series_id))
        with self._cache_lock:
            cached = self._measure_cache.get(key)
            if cached is not None:
                return cached
            measure = self._load_measure(*key)
            cached = self._immutable_measure(measure)
            self._measure_cache[key] = cached
            return cached

    @staticmethod
    def _immutable_measure(measure: FiniteMeasure) -> FiniteMeasure:
        """Copy one validated measure into immutable, non-aliased cache storage."""
        cached = FiniteMeasure(measure.support, measure.weights, measure.total_mass)
        support = np.frombuffer(cached.support.tobytes(order="C"), dtype=cached.support.dtype)
        weights = np.frombuffer(cached.weights.tobytes(order="C"), dtype=cached.weights.dtype)
        object.__setattr__(cached, "support", support.reshape(cached.support.shape))
        object.__setattr__(cached, "weights", weights.reshape(cached.weights.shape))
        return cached

    def _load_measure(self, checkpoint_id: str, series_id: str) -> FiniteMeasure:
        try:
            observation = self._observation.loc[(checkpoint_id, series_id)]
        except KeyError as exc:
            raise KeyError(series_id) from exc
        rows = observation if isinstance(observation, pd.DataFrame) else observation.to_frame().T
        rows = rows.sort_values("observation_id")
        observation_ids = tuple(rows["observation_id"].astype(str))
        if len(rows) > 1 and self.view.selection.replicate_policy.mode != "pool":
            raise ValueError(
                "Trajectory compilation requires replicate mode='select' or mode='pool' for "
                f"{series_id!r}/{checkpoint_id!r}."
            )
        laws = []
        masses = []
        refs = []
        for observation_id in observation_ids:
            support = self._support.loc[observation_id]
            ref = SupportRef(
                str(support["store_id"]),
                self.view.representation_id,
                str(support["support_key"]),
            )
            total_mass = 1.0
            if self.view.abundance_channel is not None:
                if observation_id not in self._abundance.index:
                    raise ValueError(
                        f"Observation {observation_id!r} lacks selected abundance channel "
                        f"{self.view.abundance_channel!r}."
                    )
                abundance = self._abundance.loc[observation_id]
                if isinstance(abundance, pd.DataFrame):
                    raise ValueError(
                        f"Observation {observation_id!r} has duplicate abundance rows."
                    )
                if not bool(abundance["observed"]):
                    raise ValueError(f"Observation {observation_id!r} has unobserved abundance.")
                total_mass = float(abundance["value"])
            if not np.isfinite(total_mass) or total_mass <= 0:
                raise ValueError(
                    f"Compact trajectory mass must be positive for {observation_id!r}; select "
                    "an explicit positive transformed abundance channel."
                )
            refs.append(ref)
            masses.append(total_mass)
            laws.append(self.view.study.supports.read(ref))
        if len(rows) == 1:
            ref = refs[0]
            total_mass = masses[0]
            store = self.view.study.supports[ref.store_id]
            finite_reader = getattr(store, "finite_measure", None)
            if callable(finite_reader):
                measure = finite_reader(ref)
                if np.isclose(measure.total_mass, total_mass, rtol=0, atol=0):
                    return measure
            law = laws[0]
            return FiniteMeasure(law.coordinates, law.probabilities * total_mass, total_mass)

        policy = self.view.selection.replicate_policy
        if policy.geometry_pooling != "concatenate":
            raise ValueError("Compact-v3 replicate pooling supports geometry concatenation only.")
        mass_array = np.asarray(masses, dtype=np.float64)
        if self.view.abundance_channel is None:
            scales = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
        elif policy.abundance_pooling == "sum":
            scales = mass_array
        elif policy.abundance_pooling == "mean":
            scales = mass_array / len(rows)
        elif policy.abundance_pooling == "exposure_weighted":
            if "replicate_exposure" not in rows:
                raise ValueError(
                    "Exposure-weighted replicate pooling requires observation replicate_exposure."
                )
            exposure = pd.to_numeric(rows["replicate_exposure"], errors="raise").to_numpy(float)
            if not np.isfinite(exposure).all() or np.any(exposure <= 0):
                raise ValueError("Replicate exposures must be positive and finite.")
            scales = mass_array * exposure / exposure.sum()
        else:  # pragma: no cover - ReplicatePolicy validates this contract.
            raise ValueError(f"Unsupported abundance pooling {policy.abundance_pooling!r}.")
        total_mass = float(scales.sum())
        coordinates = np.concatenate([law.coordinates for law in laws], axis=0)
        weights = np.concatenate(
            [law.probabilities * scale for law, scale in zip(laws, scales, strict=True)]
        )
        return FiniteMeasure(coordinates, weights, total_mass)


def _validate_static_trajectory_semantics(view: StudyView) -> None:
    is_lps = hasattr(view.study, "perturbations")
    perturbation_key = "perturbation_id" if is_lps else "condition_id"
    perturbations = (
        view.study.perturbations._unsafe_view() if is_lps else view.study.conditions._unsafe_view()
    )
    series = view.study.series._unsafe_view()
    effect_binding = view.effect_binding()
    if effect_binding.empty:
        raise ValueError("Released trajectory recipes require a selected effect binding catalog.")
    hierarchical_columns = [
        column
        for column in ("parent_effect_id", "shrinkage_group_id")
        if column in effect_binding and effect_binding[column].notna().any()
    ]
    if hierarchical_columns:
        raise ValueError(
            "Released trajectory recipes support flat effect bindings only; "
            f"hierarchical columns are populated: {hierarchical_columns}."
        )
    selected_perturbations = set(
        series.loc[series["series_id"].isin(view.series_ids), perturbation_key].astype(str)
    )
    control_column = "is_control" if is_lps else "is_reference"
    references = perturbations.loc[
        perturbations[perturbation_key].isin(selected_perturbations) & perturbations[control_column]
    ]
    binding = view.reference_binding()
    if binding.empty:
        raise ValueError(
            "Released trajectory recipes require a selected reference binding catalog."
        )
    pools = set(
        binding.loc[
            binding[perturbation_key].isin(references[perturbation_key]),
            "reference_pool_id",
        ].astype(str)
    )
    if len(pools) > 1:
        raise ValueError(
            "Released trajectory recipes have one global soft reference and cannot compile "
            f"multiple reference pools: {sorted(pools)}."
        )


def _measure_meta(view: StudyView, axis: Axis) -> pd.DataFrame:
    series = view.study.series._unsafe_view()
    series = series.loc[series["series_id"].isin(view.series_ids)].copy()
    is_lps = hasattr(view.study, "perturbations")
    perturbation_key = "perturbation_id" if is_lps else "condition_id"
    perturbations = (
        view.study.perturbations._unsafe_view().set_index("perturbation_id")
        if is_lps
        else view.study.conditions._unsafe_view().set_index("condition_id")
    )
    observations = view.observations()
    source = observations.loc[observations["checkpoint_id"].eq(axis.source)]
    if source.duplicated("series_id").any():
        if view.selection.replicate_policy.mode != "pool":
            duplicate = source.loc[source.duplicated("series_id"), "series_id"].iloc[0]
            raise ValueError(
                "Trajectory compilation requires replicate mode='select' or mode='pool' for "
                f"source series {duplicate!r}."
            )
        source = source.drop_duplicates("series_id")
    source = source.set_index("series_id")
    effect_binding = view.effect_binding()
    if effect_binding.empty:
        raise ValueError("Trajectory compilation requires a selected effect binding catalog.")
    effect_by_perturbation = (
        effect_binding.set_index(perturbation_key)["effect_id"].astype(str).to_dict()
    )
    rows: list[dict[str, Any]] = []
    legacy_binding = view.study.provenance.get("codec") == "credo.current_five_file"
    native_lps = is_lps and "schema_v3_conversion" not in view.study.provenance
    semantic_columns = {
        "series_id",
        "condition_id",
        "perturbation_id",
        "subject_id",
        "experimental_unit_id",
        "context_trajectory_id",
        "biological_replicate_id",
        "continuity_kind",
        "embedding_id",
        "reference_role",
    }
    for row in series.itertuples(index=False):
        values = row._asdict()
        perturbation_id = str(values[perturbation_key])
        perturbation = perturbations.loc[perturbation_id]
        source_observation = source.loc[row.series_id]
        raw_context = source_observation.get("context_id")
        context_id = (
            str(values["context_trajectory_id"])
            if native_lps
            else (str(row.subject_id) if pd.isna(raw_context) else str(raw_context))
        )
        compiled = {key: value for key, value in values.items() if key not in semantic_columns}
        compiled.update(
            {
                "measure_id": str(row.series_id),
                "sample_id": str(row.subject_id),
                "perturbation_id": perturbation_id,
                "embedding_id": effect_by_perturbation[perturbation_id],
                "context_group_id": context_id,
                "is_control": bool(
                    perturbation["is_control"] if is_lps else perturbation["is_reference"]
                ),
            }
        )
        if is_lps:
            compiled["experimental_unit_id"] = str(values["experimental_unit_id"])
            compiled["continuity_kind"] = str(values["continuity_kind"])
        elif "perturbation_id" not in compiled and legacy_binding:
            compiled["perturbation_id"] = perturbation_id
        for column in ("guide_id", "target_gene"):
            if (
                column not in compiled
                and column in perturbation.index
                and pd.notna(perturbation[column])
            ):
                compiled[column] = str(perturbation[column])
        components = getattr(view.study, "perturbation_components", None)
        if is_lps and components is not None:
            component_rows = components._unsafe_view()
            component_rows = component_rows.loc[
                component_rows["perturbation_id"].eq(perturbation_id)
            ]
            constructs = tuple(dict.fromkeys(component_rows["construct_id"].astype(str)))
            targets = tuple(dict.fromkeys(component_rows["target_id"].astype(str)))
            if len(constructs) == 1:
                compiled.setdefault("guide_id", constructs[0])
                compiled["construct_id"] = constructs[0]
            if len(targets) == 1:
                compiled.setdefault("target_gene", targets[0])
                compiled["target_id"] = targets[0]
        rows.append(compiled)
    frame = pd.DataFrame(rows)
    preferred = [
        "measure_id",
        "sample_id",
        "perturbation_id",
        "guide_id",
        "embedding_id",
        "target_gene",
        "context_group_id",
        "is_control",
    ]
    return frame.loc[
        :,
        [column for column in preferred if column in frame]
        + [column for column in frame if column not in preferred],
    ]


def _count_blocks(view: StudyView, measure_meta: pd.DataFrame) -> tuple[CountBlock, ...]:
    frame = view.compositions()
    if frame.empty:
        return ()
    index = {measure_id: position for position, measure_id in enumerate(measure_meta["measure_id"])}
    context_by_series = measure_meta.set_index("measure_id")["context_group_id"].astype(str)
    blocks: list[CountBlock] = []
    for _block_id, rows in frame.groupby("composition_block_id", observed=True, sort=False):
        rows = rows.sort_values(["series_id", "observation_id"])
        active = rows.loc[rows["series_id"].isin(index)].copy()
        background = rows.loc[~rows["series_id"].isin(index)].copy()
        if active.empty:
            continue
        context_groups = set(active["series_id"].map(context_by_series).astype(str))
        if len(context_groups) != 1:
            raise ValueError(
                "One composition block must map to one population-series context trajectory; "
                f"block={_block_id!r}, contexts={sorted(context_groups)}."
            )
        context_group_id = next(iter(context_groups))
        if active["series_id"].duplicated().any():
            active = (
                active.groupby("series_id", observed=True, sort=False)
                .agg(
                    {
                        "context_id": "first",
                        "checkpoint_id": "first",
                        "denominator_id": "first",
                        "exposure": "sum",
                        "count": "sum",
                    }
                )
                .reset_index()
            )
        policy = view.selection.composition_policy
        source_denominator = str(rows["denominator_id"].iloc[0])
        modeled_denominator = source_denominator
        if policy == "condition_on_selection":
            modeled_denominator = f"{source_denominator}|conditioned:{view.semantic_hash()[:16]}"
        if policy != "preserve_background" and not background.empty:
            raise ValueError(
                f"Composition policy {policy!r} retained unmodeled background unexpectedly."
            )
        if not background.empty and background["series_id"].duplicated().any():
            aggregations: dict[str, str] = {"exposure": "sum", "count": "sum"}
            if "background_fitness" in background:
                aggregations["background_fitness"] = "mean"
            background = (
                background.groupby("series_id", observed=True, sort=False)
                .agg(aggregations)
                .reset_index()
            )
        background_fitness = (
            background["background_fitness"].to_numpy(float)
            if "background_fitness" in background
            else np.zeros(len(background), dtype=np.float32)
        )
        blocks.append(
            CountBlock(
                context_group_id=context_group_id,
                time_label=str(active["checkpoint_id"].iloc[0]),
                measure_indices=np.asarray([index[value] for value in active["series_id"]]),
                exposure=active["exposure"].to_numpy(),
                counts=active["count"].to_numpy(),
                background_series_ids=tuple(background["series_id"].astype(str)),
                background_fitness=background_fitness,
                background_exposure=background["exposure"].to_numpy(),
                background_counts=background["count"].to_numpy(),
                source_denominator_id=source_denominator,
                modeled_denominator_id=modeled_denominator,
                conditioning_policy=policy,
            )
        )
    return tuple(blocks)


def _representation(view: StudyView) -> RepresentationArtifact:
    legacy = view.study.provenance.get("legacy_representation")
    if isinstance(legacy, Mapping):
        return RepresentationArtifact.from_dict(legacy)
    spec = view.representation
    support_hash = (
        spec.support_artifact.sha256
        if spec.support_artifact is not None
        else hashlib.sha256(
            f"{view.study.content_hash()}:{spec.support_store_id}:{spec.representation_id}".encode()
        ).hexdigest()
    )
    series = view.study.series._unsafe_view().set_index("series_id")
    included_samples = tuple(
        dict.fromkeys(
            str(series.loc[series_id, "subject_id"])
            for series_id in spec.included_series
            if series_id in series.index
        )
    )
    fit_scope = {
        "external_frozen": "external",
        "shared_source_only": "all_source_samples",
        "shared_all_observations": "all_checkpoints",
        "nested_by_subject": "training_split",
        "nested_by_perturbation": "training_split",
        "nested_by_checkpoint": "training_fold_source",
        "fully_nested": "training_split",
    }[spec.scope_mode]
    return RepresentationArtifact(
        representation_id=spec.representation_id,
        backend=spec.backend,
        latent_dim=spec.dimension,
        latent_cache_hash=support_hash,
        fit_scope=fit_scope,
        gene_names_hash=(None if spec.feature_artifact is None else spec.feature_artifact.sha256),
        encoder_state_hash=(
            None if spec.encoder_artifact is None else spec.encoder_artifact.sha256
        ),
        decoder_state_hash=(
            None if spec.decoder_artifact is None else spec.decoder_artifact.sha256
        ),
        normalization_hash=(
            None if spec.normalization_artifact is None else spec.normalization_artifact.sha256
        ),
        included_samples=included_samples,
        included_time_labels=spec.included_checkpoints,
        producer={
            "source": "PerturbSeqStudy",
            "study_content_hash": view.study.content_hash(),
            "scope_mode": spec.scope_mode,
            "fit_split_id": spec.fit_split_id,
            "fit_selection_hash": spec.fit_selection_hash,
            "fit_observation_scope": spec.fit_observation_scope,
        },
    )


def _source_observed_context_backgrounds(
    view: StudyView,
    axis: Axis,
    measure_meta: pd.DataFrame,
    *,
    particles: int,
    minimum_mass_coverage: float,
) -> tuple[FixedContextBackground, ...]:
    """Compress omitted source-pool members into fixed donor-local measures."""
    if view.selection.composition_policy != "preserve_background":
        raise ValueError(
            "source_observed_aggregate context requires "
            "selection.composition_policy='preserve_background'."
        )
    if view.abundance_channel is None or view.study.abundance is None:
        raise ValueError(
            "source_observed_aggregate context requires an explicit abundance channel."
        )
    composition = view.compositions()
    if composition.empty:
        raise ValueError(
            "source_observed_aggregate context requires pooled composition blocks."
        )
    active_series = set(measure_meta["measure_id"].astype(str))
    active_rows = composition.loc[composition["series_id"].isin(active_series)]
    if active_rows.empty:
        raise ValueError("No active rows were found in pooled composition blocks.")
    touched_blocks = tuple(dict.fromkeys(active_rows["composition_block_id"].astype(str)))
    composition = composition.loc[
        composition["composition_block_id"].isin(touched_blocks)
    ].copy()
    context_by_series = (
        measure_meta.set_index("measure_id")["context_group_id"].astype(str).to_dict()
    )
    series_frame = view.study.series._unsafe_view().set_index("series_id")
    abundance = view.study.abundance._unsafe_view()
    abundance = abundance.loc[
        abundance["channel_id"].eq(view.abundance_channel),
        ["observation_id", "value", "observed", "denominator_id"],
    ].copy()
    if abundance["observation_id"].duplicated().any():
        raise ValueError("Background abundance channel contains duplicate observations.")
    abundance = abundance.set_index("observation_id")
    support = view.study.support_index._unsafe_view()
    support = support.loc[
        support["representation_id"].eq(view.representation_id)
    ].set_index("observation_id")
    if support.index.duplicated().any():
        raise ValueError("Background support index contains duplicate observations.")
    source_observations = view.study.observations._unsafe_view()
    source_observations = source_observations.loc[
        source_observations["checkpoint_id"].eq(axis.source),
        ["observation_id", "series_id", "context_id"],
    ].copy()
    if source_observations["series_id"].duplicated().any():
        raise ValueError(
            "Fixed source context currently requires one source observation per series."
        )
    if "schema_v3_conversion" not in view.study.provenance:
        series_context = series_frame["context_trajectory_id"].astype(str).to_dict()
    else:
        source_by_series = source_observations.set_index("series_id")
        series_context = {
            str(series_id): (
                str(source_by_series.loc[series_id, "context_id"])
                if pd.notna(source_by_series.loc[series_id, "context_id"])
                else str(series_frame.loc[series_id, "subject_id"])
            )
            for series_id in source_by_series.index
        }

    backgrounds: list[FixedContextBackground] = []
    blocks_by_group: dict[str, list[str]] = {}
    for block_id in touched_blocks:
        rows = composition.loc[composition["composition_block_id"].eq(block_id)]
        active = rows.loc[rows["series_id"].isin(active_series)]
        groups = {
            context_by_series[str(series_id)] for series_id in active["series_id"].astype(str)
        }
        if len(groups) != 1:
            raise ValueError(
                "One source composition block must map to one modeled context group; "
                f"block={block_id!r}, groups={sorted(groups)}."
            )
        blocks_by_group.setdefault(next(iter(groups)), []).append(block_id)

    for context_group_id, group_blocks in blocks_by_group.items():
        background_memberships: list[set[str]] = []
        for block_id in group_blocks:
            rows = composition.loc[composition["composition_block_id"].eq(block_id)]
            background_memberships.append(
                set(
                    rows.loc[~rows["series_id"].isin(active_series), "series_id"].astype(str)
                )
            )
        if not background_memberships[0]:
            raise ValueError(
                "source_observed_aggregate context requires omitted source-pool members; "
                f"group={context_group_id!r} has none."
            )
        if any(
            membership != background_memberships[0]
            for membership in background_memberships[1:]
        ):
            raise ValueError(
                "Omitted population membership must be stable across checkpoint "
                f"composition blocks for group={context_group_id!r}."
            )
        background_series = background_memberships[0]
        background_contexts = {
            series_context.get(str(series_id))
            for series_id in background_series
        }
        if background_contexts != {context_group_id}:
            raise ValueError(
                "An omitted source-pool denominator crosses ecological context groups; "
                f"group={context_group_id!r}, "
                f"contexts={sorted(str(v) for v in background_contexts)}."
            )
        background = source_observations.loc[
            source_observations["series_id"].isin(background_series)
        ].copy()
        if set(background["series_id"].astype(str)) != background_series:
            missing = sorted(background_series - set(background["series_id"].astype(str)))[:5]
            raise ValueError(
                "Fixed source context background lacks source observations for "
                f"series={missing}."
            )
        observation_ids = background["observation_id"].astype(str)
        missing_abundance = sorted(set(observation_ids) - set(abundance.index))[:5]
        if missing_abundance:
            raise ValueError(
                "Fixed source context background lacks modeling abundance for "
                f"observations={missing_abundance}."
            )
        mass_rows = abundance.loc[list(observation_ids)].copy()
        mass_values = pd.to_numeric(mass_rows["value"], errors="raise").to_numpy(float)
        if (
            not mass_rows["observed"].astype(bool).all()
            or not np.isfinite(mass_values).all()
            or np.any(mass_values <= 0)
        ):
            raise ValueError(
                "Fixed source context background requires positive observed abundance."
            )
        total_mass = float(mass_values.sum())
        background["_background_mass"] = mass_values
        background["_support_available"] = background["observation_id"].map(
            support["available"].astype(bool)
        )
        supported = background.loc[background["_support_available"].eq(True)].copy()
        if supported.empty:
            raise ValueError(
                f"Fixed source context background {context_group_id!r} has no latent support."
            )
        supported_mass = float(supported["_background_mass"].sum())
        coverage = supported_mass / total_mass
        if coverage < float(minimum_mass_coverage):
            raise ValueError(
                "Fixed source context background support-mass coverage is below the "
                f"configured minimum: group={context_group_id!r}, "
                f"coverage={coverage:.6f}, minimum={minimum_mass_coverage:.6f}."
            )

        supported = supported.sort_values("observation_id").reset_index(drop=True)
        mixture = supported["_background_mass"].to_numpy(float, copy=True)
        mixture = mixture / mixture.sum()
        seed_material = (
            f"{view.semantic_hash()}:{axis.source}:{context_group_id}:"
            f"{','.join(sorted(group_blocks))}:"
            f"{particles}:source_observed_aggregate"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
        generator = np.random.default_rng(seed)
        selected = generator.choice(
            len(supported),
            size=int(particles),
            replace=True,
            p=mixture,
        )
        coordinates = np.empty(
            (int(particles), view.representation.dimension),
            dtype=np.float32,
        )
        for row_index in np.unique(selected):
            positions = np.flatnonzero(selected == row_index)
            support_row = support.loc[str(supported.loc[row_index, "observation_id"])]
            ref = SupportRef(
                str(support_row["store_id"]),
                view.representation_id,
                str(support_row["support_key"]),
            )
            law = view.study.supports.read(ref)
            atom_indices = generator.choice(
                len(law.coordinates),
                size=len(positions),
                replace=True,
                p=law.probabilities,
            )
            coordinates[positions] = law.coordinates[atom_indices]
        measure = FiniteMeasure(
            coordinates,
            np.full(int(particles), total_mass / int(particles), dtype=np.float64),
            total_mass,
        )
        denominator_ids = set(mass_rows["denominator_id"].dropna().astype(str))
        if len(denominator_ids) != 1:
            raise ValueError(
                "Fixed source context background must have one source denominator; "
                f"group={context_group_id!r}, denominators={sorted(denominator_ids)}."
            )
        backgrounds.append(
            FixedContextBackground(
                context_group_id=context_group_id,
                source_checkpoint_id=axis.source,
                measure=measure,
                source_denominator_id=next(iter(denominator_ids)),
                observed_series_count=int(background["series_id"].nunique()),
                supported_series_count=int(supported["series_id"].nunique()),
                support_mass_fraction=coverage,
                compression_method="deterministic_mixture_sample",
                compression_seed=seed,
            )
        )
    return tuple(backgrounds)


def compile_trajectory_view(
    view: StudyView,
    *,
    split_plan: SplitPlan | None = None,
    context_background_mode: str = "none",
    context_background_particles: int = 2048,
    context_background_min_mass_coverage: float = 0.99,
) -> TrajectoryData:
    """Compile one semantically validated view for the legacy trajectory executor."""
    axis = _axis(view)
    mass_semantics = _mass_semantics(view)
    if view.abundance_channel is not None:
        support_index = view.study.support_index._unsafe_view()
        available = set(
            support_index.loc[
                support_index["representation_id"].eq(view.representation_id)
                & support_index["available"],
                "observation_id",
            ]
        ) & set(view.observation_ids)
        abundance = view.abundance().set_index("observation_id")
        invalid = [
            observation_id
            for observation_id in available
            if observation_id not in abundance.index
            or not bool(abundance.loc[observation_id, "observed"])
            or not np.isfinite(float(abundance.loc[observation_id, "value"]))
            or float(abundance.loc[observation_id, "value"]) <= 0
        ]
        if invalid:
            raise ValueError(
                "Trajectory geometry requires an explicit positive modeling abundance; "
                f"invalid observations={sorted(invalid)[:5]}."
            )
    _validate_static_trajectory_semantics(view)
    metadata = _measure_meta(view, axis)
    measures = _CompiledMeasures(view, axis, mass_semantics)
    count_blocks = _count_blocks(view, metadata)
    if context_background_mode == "none":
        context_backgrounds: tuple[FixedContextBackground, ...] = ()
    elif context_background_mode == "source_observed_aggregate":
        context_backgrounds = _source_observed_context_backgrounds(
            view,
            axis,
            metadata,
            particles=int(context_background_particles),
            minimum_mass_coverage=float(context_background_min_mass_coverage),
        )
    else:
        raise ValueError(f"Unknown context background mode {context_background_mode!r}.")
    context_background_contract = [
        {
            "context_group_id": background.context_group_id,
            "source_checkpoint_id": background.source_checkpoint_id,
            "source_denominator_id": background.source_denominator_id,
            "observed_series_count": background.observed_series_count,
            "supported_series_count": background.supported_series_count,
            "support_mass_fraction": background.support_mass_fraction,
            "aggregate_atoms": len(background.measure.support),
            "total_mass": background.measure.total_mass,
            "compression_method": background.compression_method,
            "compression_seed": background.compression_seed,
            "content_hash": background.content_hash(),
            "trajectory_role": "observed_fixed_background",
        }
        for background in context_backgrounds
    ]
    observation_map: dict[str, str] = {}
    pooled_observations: dict[str, tuple[str, ...]] = {}
    for (checkpoint_id, series_id), rows in view.observations().groupby(
        ["checkpoint_id", "series_id"], observed=True, sort=False
    ):
        observation_ids = tuple(sorted(rows["observation_id"].astype(str)))
        if len(observation_ids) == 1:
            observation_id = observation_ids[0]
        elif view.selection.replicate_policy.mode == "pool":
            observation_id = _pooled_observation_id(
                str(series_id), str(checkpoint_id), observation_ids
            )
            pooled_observations[observation_id] = observation_ids
        else:
            continue
        observation_map[f"{series_id}\0{checkpoint_id}"] = observation_id
    provenance = view.study.provenance
    mass_denominators = (
        sorted(view.abundance()["denominator_id"].dropna().astype(str).unique().tolist())
        if "denominator_id" in view.abundance()
        else []
    )
    runtime_metadata = {
        "input_paths": dict(provenance.get("input_paths", {})),
        "input_hashes": dict(provenance.get("input_hashes", {})),
        "dataset": dict(provenance.get("legacy_dataset", {})),
        "mass_denominators": list(provenance.get("mass_denominators", mass_denominators)),
        "study_content_hash": view.study.content_hash(),
        "selection_hash": view.semantic_hash(),
        "compiled_problem_hash": hashlib.sha256(
            (
                view.semantic_hash()
                + ":"
                + ("unplanned" if split_plan is None else split_plan.split_id)
                + ":trajectory-v1:"
                + hashlib.sha256(
                    json.dumps(
                        context_background_contract,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            ).encode()
        ).hexdigest(),
        "replicate_transform": {
            "policy": view.selection.replicate_policy.to_dict(),
            "pooled_observations": pooled_observations,
        },
        "composition_transform": [
            {
                "source_denominator_id": block.source_denominator_id,
                "modeled_denominator_id": block.modeled_denominator_id,
                "conditioning_policy": block.conditioning_policy,
                "background_series_ids": list(block.background_series_ids),
                "background_fitness_source": (
                    "none" if not block.background_series_ids else "table_or_neutral_zero"
                ),
            }
            for block in count_blocks
        ],
        "context_background": {
            "mode": context_background_mode,
            "temporal_policy": (
                "absent"
                if not context_backgrounds
                else "source_observed_geometry_and_mass_fixed_across_rollout"
            ),
            "modeled_active_trajectories": True,
            "background_trajectories_modeled": False,
            "groups": context_background_contract,
        },
        "observation_id_by_series_checkpoint": observation_map,
    }
    if split_plan is not None:
        runtime_metadata["split_plan"] = split_plan.to_dict()
    return TrajectoryData(
        axis=axis,
        measures=measures,
        measure_meta=metadata,
        mass_semantics=mass_semantics,
        count_blocks=count_blocks,
        context_backgrounds=context_backgrounds,
        metadata=runtime_metadata,
        representation=_representation(view),
    )


def _observation_set(
    observations: pd.DataFrame,
    observation_ids: tuple[str, ...],
) -> CompiledObservationSet:
    selected = observations.loc[observations["observation_id"].isin(observation_ids)]
    selected_ids = set(selected["observation_id"].astype(str))
    ordered_ids = tuple(value for value in observation_ids if value in selected_ids)
    return CompiledObservationSet(
        observation_ids=ordered_ids,
        series_ids=tuple(dict.fromkeys(selected["series_id"].astype(str))),
        checkpoint_ids=tuple(dict.fromkeys(selected["checkpoint_id"].astype(str))),
    )


def _subject_local_composition_blocks(
    view: StudyView,
    composition_rows: pd.DataFrame,
    selected_observation_ids: set[str],
    *,
    held_out_subject_ids: set[str],
    partition: Literal["training", "validation"],
) -> bool:
    """Return whether every touched denominator stays within selected subjects."""
    selected_rows = composition_rows.loc[
        composition_rows["observation_id"].isin(selected_observation_ids)
    ]
    if selected_rows.empty:
        return False
    touched_blocks = set(selected_rows["composition_block_id"])
    denominator_rows = composition_rows.loc[
        composition_rows["composition_block_id"].isin(touched_blocks)
    ]
    subject_by_series = (
        view.study.series._unsafe_view().set_index("series_id")["subject_id"].astype(str)
    )
    selected_subjects = selected_rows["series_id"].map(subject_by_series)
    denominator_subjects = denominator_rows["series_id"].map(subject_by_series)
    if selected_subjects.isna().any() or denominator_subjects.isna().any():
        return False
    selected_subject_set = set(selected_subjects.astype(str))
    if partition == "training":
        if selected_subject_set & held_out_subject_ids:
            return False
    elif partition == "validation":
        if not selected_subject_set <= held_out_subject_ids:
            return False
    else:
        raise ValueError(f"Unknown composition partition {partition!r}.")
    return set(denominator_subjects.astype(str)) <= selected_subject_set


def compile_finite_measure_problem(
    view: StudyView,
    split_plan: SplitPlan,
    *,
    config: Any | None = None,
) -> FiniteMeasureDynamicsProblem:
    """Compile target-outcome-separated train and validation finite measures."""
    observations = view.observations()
    source_checkpoint = view.study.design.source_checkpoint_id
    source_ids = tuple(
        observations.loc[
            observations["checkpoint_id"].eq(source_checkpoint), "observation_id"
        ].astype(str)
    )
    source_id_set = set(source_ids)
    training_target_ids = tuple(
        value for value in split_plan.train_observation_ids if value not in source_id_set
    )
    validation_target_ids = tuple(
        value for value in split_plan.validation_observation_ids if value not in source_id_set
    )
    training_target_id_set = set(training_target_ids)
    validation_target_id_set = set(validation_target_ids)
    training_observation_ids = tuple(
        value
        for value in split_plan.train_observation_ids
        if value in source_id_set or value in training_target_id_set
    )
    validation_observation_ids = tuple(
        value
        for value in split_plan.validation_observation_ids
        if value in source_id_set or value in validation_target_id_set
    )

    def partition_selection(
        selection: Any,
        observation_ids: tuple[str, ...],
        *,
        partition: Literal["training", "validation"],
    ) -> Any:
        compiled = replace(selection, observation_ids=observation_ids)
        compositions = getattr(view.study, "compositions", None)
        if (
            compositions is None
            or split_plan.source != "held_out"
            or compiled.composition_policy == "drop"
        ):
            return compiled
        frame = compositions._unsafe_view()
        selected_ids = set(observation_ids)
        touched = set(frame.loc[frame["observation_id"].isin(selected_ids), "composition_block_id"])
        denominator_ids = set(
            frame.loc[frame["composition_block_id"].isin(touched), "observation_id"].astype(str)
        )
        if denominator_ids - selected_ids:
            # A restricted guide panel may still use the full count denominator in
            # donor-held-out CV when each physical pool is wholly donor-local.
            if (
                compiled.composition_policy == "preserve_background"
                and split_plan.task_kind == "subject_generalization"
                and split_plan.held_out_subject_ids
                and _subject_local_composition_blocks(
                    view,
                    frame,
                    selected_ids,
                    held_out_subject_ids=set(split_plan.held_out_subject_ids),
                    partition=partition,
                )
            ):
                return compiled
            # A held-out outcome may never enter fitting as denominator background.
            return replace(compiled, composition_policy="condition_on_selection")
        return compiled

    training_selection = partition_selection(
        split_plan.train_selection,
        training_observation_ids,
        partition="training",
    )
    validation_selection = partition_selection(
        split_plan.validation_selection,
        validation_observation_ids,
        partition="validation",
    )
    training_view = view.study.view(
        training_selection,
        representation_id=view.representation_id,
        abundance_channel=view.abundance_channel,
    )
    validation_view = view.study.view(
        validation_selection,
        representation_id=view.representation_id,
        abundance_channel=view.abundance_channel,
    )
    model_config = None if config is None else getattr(config, "model", None)
    background_mode = (
        "none" if model_config is None else str(model_config.context_background)
    )
    background_particles = (
        2048 if model_config is None else int(model_config.context_background_particles)
    )
    background_coverage = (
        0.99
        if model_config is None
        else float(model_config.context_background_min_mass_coverage)
    )
    compile_options = {
        "split_plan": split_plan,
        "context_background_mode": background_mode,
        "context_background_particles": background_particles,
        "context_background_min_mass_coverage": background_coverage,
    }
    training = compile_trajectory_view(training_view, **compile_options)
    validation = compile_trajectory_view(validation_view, **compile_options)

    background = None
    if getattr(view.study, "compositions", None) is not None:
        composition = view.study.compositions._unsafe_view()
        active_ids = set(training_observation_ids) | set(validation_observation_ids)
        touched_blocks = set(
            composition.loc[composition["observation_id"].isin(active_ids), "composition_block_id"]
        )
        detached = composition.loc[
            composition["composition_block_id"].isin(touched_blocks)
            & ~composition["observation_id"].isin(active_ids)
        ]
        if len(detached):
            background = CompiledObservationSet(
                observation_ids=tuple(detached["observation_id"].astype(str)),
                series_ids=tuple(dict.fromkeys(detached["series_id"].astype(str))),
                checkpoint_ids=tuple(dict.fromkeys(detached["checkpoint_id"].astype(str))),
            )
    partition = CompiledLPSSplit(
        plan=split_plan,
        source=_observation_set(observations, source_ids),
        training_targets=_observation_set(observations, training_target_ids),
        validation_targets=_observation_set(observations, validation_target_ids),
        composition_background=background,
    )
    payload = {
        "problem_kind": "finite_measure_dynamics",
        "study_content_hash": view.study.content_hash(),
        "selection_hash": view.semantic_hash(),
        "split_id": split_plan.split_id,
        "source_observation_ids": list(partition.source.observation_ids),
        "training_target_observation_ids": list(training_target_ids),
        "validation_target_observation_ids": list(validation_target_ids),
        "training_composition_policy": training_selection.composition_policy,
        "validation_composition_policy": validation_selection.composition_policy,
        "context_background_mode": background_mode,
        "training_context_background_hashes": [
            value.content_hash() for value in training.context_backgrounds
        ],
        "validation_context_background_hashes": [
            value.content_hash() for value in validation.context_backgrounds
        ],
        "compiler": "finite_measure_lps_v1",
    }
    problem_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FiniteMeasureDynamicsProblem(
        problem_kind="finite_measure_dynamics",
        partition=partition,
        study_content_hash=view.study.content_hash(),
        selection_hash=view.semantic_hash(),
        problem_hash=problem_hash,
        problem_metadata={
            "compiler": "finite_measure_lps_v1",
            "training_composition_policy": training_selection.composition_policy,
            "validation_composition_policy": validation_selection.composition_policy,
            "context_background_mode": background_mode,
        },
        training=training,
        validation=validation,
    )


__all__ = ["compile_finite_measure_problem", "compile_trajectory_view"]
