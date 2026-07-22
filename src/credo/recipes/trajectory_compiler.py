"""Recipe-owned compilation from semantic Study views to trajectory compatibility data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import Axis, FiniteMeasure, MassSemantics, RepresentationArtifact, TrajectoryData
from ..data.study import StudyView
from ..data.support import SupportRef
from ..data.tables import AbundanceSemantics
from ..objective import CountBlock


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
    """Lazy finite-measure adapter over one representation and abundance channel."""

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
        self._observation = observations.set_index(["checkpoint_id", "series_id"])
        self._support = available.set_index("observation_id")
        abundance = view.abundance()
        self._abundance = (
            abundance.set_index("observation_id") if len(abundance) else pd.DataFrame()
        )
        order = {series_id: index for index, series_id in enumerate(view.series_ids)}
        self.series_by_checkpoint = {
            checkpoint_id: tuple(
                sorted(
                    observations.loc[
                        observations["checkpoint_id"].eq(checkpoint_id), "series_id"
                    ].astype(str),
                    key=order.__getitem__,
                )
            )
            for checkpoint_id in axis.labels
        }
        self._views = {
            checkpoint_id: _CompiledCheckpointMeasures(self, checkpoint_id)
            for checkpoint_id in axis.labels
        }

    def __getitem__(self, checkpoint_id: str) -> Mapping[str, FiniteMeasure]:
        return self._views[str(checkpoint_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._views)

    def __len__(self) -> int:
        return len(self._views)

    def measure(self, checkpoint_id: str, series_id: str) -> FiniteMeasure:
        try:
            observation = self._observation.loc[(checkpoint_id, series_id)]
        except KeyError as exc:
            raise KeyError(series_id) from exc
        if isinstance(observation, pd.DataFrame):
            raise ValueError(
                "Trajectory compilation does not support replicate observations; select or pool "
                f"{series_id!r}/{checkpoint_id!r} first."
            )
        observation_id = str(observation["observation_id"])
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
                raise ValueError(f"Observation {observation_id!r} has duplicate abundance rows.")
            if not bool(abundance["observed"]):
                raise ValueError(f"Observation {observation_id!r} has unobserved abundance.")
            total_mass = float(abundance["value"])
        if not np.isfinite(total_mass) or total_mass <= 0:
            raise ValueError(
                f"Compact trajectory mass must be positive for {observation_id!r}; select an "
                "explicit positive transformed abundance channel."
            )
        store = self.view.study.supports[ref.store_id]
        finite_reader = getattr(store, "finite_measure", None)
        if callable(finite_reader):
            measure = finite_reader(ref)
            if np.isclose(measure.total_mass, total_mass, rtol=0, atol=0):
                return measure
        law = store.read(ref)
        return FiniteMeasure(law.coordinates, law.probabilities * total_mass, total_mass)


def _measure_meta(view: StudyView, axis: Axis) -> pd.DataFrame:
    series = view.study.series._unsafe_view()
    series = series.loc[series["series_id"].isin(view.series_ids)].copy()
    conditions = view.study.conditions._unsafe_view().set_index("condition_id")
    observations = view.observations()
    source = observations.loc[observations["checkpoint_id"].eq(axis.source)]
    if source.duplicated("series_id").any():
        duplicate = source.loc[source.duplicated("series_id"), "series_id"].iloc[0]
        raise ValueError(
            f"Trajectory compilation requires a replicate policy for source series {duplicate!r}."
        )
    source = source.set_index("series_id")
    rows: list[dict[str, Any]] = []
    legacy_binding = view.study.provenance.get("codec") == "credo.current_five_file"
    semantic_columns = {
        "series_id",
        "condition_id",
        "subject_id",
        "embedding_id",
        "reference_role",
    }
    for row in series.itertuples(index=False):
        values = row._asdict()
        condition = conditions.loc[row.condition_id]
        source_observation = source.loc[row.series_id]
        raw_context = source_observation.get("context_id")
        context_id = str(row.subject_id) if pd.isna(raw_context) else str(raw_context)
        compiled = {key: value for key, value in values.items() if key not in semantic_columns}
        compiled.update(
            {
                "measure_id": str(row.series_id),
                "sample_id": str(row.subject_id),
                "embedding_id": str(row.embedding_id),
                "context_group_id": context_id,
                "is_control": str(row.reference_role) == "reference",
            }
        )
        if "perturbation_id" not in compiled and legacy_binding:
            compiled["perturbation_id"] = str(row.condition_id)
        for column in ("guide_id", "target_gene"):
            if column not in compiled and column in condition.index and pd.notna(condition[column]):
                compiled[column] = str(condition[column])
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
    blocks: list[CountBlock] = []
    for block_id, rows in frame.groupby("composition_block_id", observed=True, sort=False):
        del block_id
        rows = rows.sort_values("series_id")
        unknown = set(rows["series_id"]) - set(index)
        if unknown:
            raise ValueError(
                "Composition policy retained series outside the compiled trajectory; "
                f"unknown={sorted(unknown)[:5]}."
            )
        blocks.append(
            CountBlock(
                context_group_id=str(rows["context_id"].iloc[0]),
                time_label=str(rows["checkpoint_id"].iloc[0]),
                measure_indices=np.asarray([index[value] for value in rows["series_id"]]),
                exposure=rows["exposure"].to_numpy(),
                counts=rows["count"].to_numpy(),
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
        else hashlib.sha256(f"{view.semantic_hash()}:{spec.support_store_id}".encode()).hexdigest()
    )
    series = view.study.series._unsafe_view().set_index("series_id")
    included_samples = tuple(
        dict.fromkeys(
            str(series.loc[series_id, "subject_id"])
            for series_id in spec.included_series
            if series_id in series.index
        )
    )
    return RepresentationArtifact(
        representation_id=spec.representation_id,
        backend=spec.backend,
        latent_dim=spec.dimension,
        latent_cache_hash=support_hash,
        fit_scope="training_split" if spec.fit_split_id else "external",
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
        producer={"source": "StudyView", "study_hash": view.semantic_hash()},
    )


def compile_trajectory_view(view: StudyView) -> TrajectoryData:
    """Compile one semantically validated view for the legacy trajectory executor."""
    axis = _axis(view)
    if tuple(view.checkpoint_ids) != tuple(axis.labels):
        raise ValueError("Trajectory compilation currently requires the complete chain design.")
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
    metadata = _measure_meta(view, axis)
    measures = _CompiledMeasures(view, axis, mass_semantics)
    count_blocks = _count_blocks(view, metadata)
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
    }
    return TrajectoryData(
        axis=axis,
        measures=measures,
        measure_meta=metadata,
        mass_semantics=mass_semantics,
        count_blocks=count_blocks,
        metadata=runtime_metadata,
        representation=_representation(view),
    )


__all__ = ["compile_trajectory_view"]
