"""Recipe-visible split planning and representation leakage checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

from ..contracts import SplitSpec, TrajectoryData
from .study import SelectionSpec, StudyView, _canonical_value, _frame_digest

SplitSource = Literal["held_out", "train_self_eval"]
SplitStrategy = Literal[
    "context_group_holdout",
    "checkpoint_holdout",
    "within_embedding_holdout",
    "train_self_eval",
]
SplitTaskKind = Literal[
    "series_generalization",
    "subject_generalization",
    "experimental_unit_generalization",
    "guide_within_target_generalization",
    "target_generalization",
    "perturbation_generalization",
    "context_generalization",
    "checkpoint_interpolation",
    "checkpoint_extrapolation",
    "combination_generalization",
    "train_self_evaluation",
]
RepresentationProtocol = Literal[
    "external_frozen",
    "shared_all_observations",
    "shared_source_only",
    "nested_by_subject",
    "nested_by_perturbation",
    "nested_by_checkpoint",
    "fully_nested",
]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_dict(selection: SelectionSpec) -> dict[str, Any]:
    return {
        "series_ids": None if selection.series_ids is None else list(selection.series_ids),
        "observation_ids": (
            None if selection.observation_ids is None else list(selection.observation_ids)
        ),
        "checkpoint_ids": (
            None if selection.checkpoint_ids is None else list(selection.checkpoint_ids)
        ),
        "subject_ids": None if selection.subject_ids is None else list(selection.subject_ids),
        "experimental_unit_ids": (
            None
            if selection.experimental_unit_ids is None
            else list(selection.experimental_unit_ids)
        ),
        "perturbation_ids": (
            None if selection.perturbation_ids is None else list(selection.perturbation_ids)
        ),
        "construct_ids": (
            None if selection.construct_ids is None else list(selection.construct_ids)
        ),
        "target_ids": None if selection.target_ids is None else list(selection.target_ids),
        "context_ids": None if selection.context_ids is None else list(selection.context_ids),
        "control_kinds": (
            None if selection.control_kinds is None else list(selection.control_kinds)
        ),
        "qc_tiers": None if selection.qc_tiers is None else list(selection.qc_tiers),
        "perturbation_filter": (
            None if selection.perturbation_filter is None else dict(selection.perturbation_filter)
        ),
        "condition_filter": (
            None if selection.condition_filter is None else dict(selection.condition_filter)
        ),
        "observation_filter": (
            None if selection.observation_filter is None else dict(selection.observation_filter)
        ),
        "representation_id": selection.representation_id,
        "abundance_channel_id": selection.abundance_channel_id,
        "effect_binding_id": selection.effect_binding_id,
        "reference_binding_id": selection.reference_binding_id,
        "composition_policy": selection.composition_policy,
        "replicate_policy": selection.replicate_policy.to_dict(),
    }


@dataclass(frozen=True)
class SplitPlan:
    """One content-addressed split used by compilation, training, and evaluation."""

    split_id: str
    train_selection: SelectionSpec
    validation_selection: SelectionSpec
    train_series_ids: tuple[str, ...]
    validation_series_ids: tuple[str, ...]
    train_checkpoint_ids: tuple[str, ...]
    validation_checkpoint_ids: tuple[str, ...]
    train_observation_ids: tuple[str, ...]
    validation_observation_ids: tuple[str, ...]
    held_out_series: tuple[str, ...]
    held_out_checkpoints: tuple[str, ...]
    held_out_observations: tuple[str, ...]
    source: SplitSource
    strategy: SplitStrategy
    representation_scope: Literal["shared", "nested"]
    representation_evaluation: Literal["transductive", "inductive"]
    task_kind: SplitTaskKind = "train_self_evaluation"
    representation_protocol: RepresentationProtocol = "shared_all_observations"
    held_out_subject_ids: tuple[str, ...] = ()
    held_out_experimental_unit_ids: tuple[str, ...] = ()
    held_out_perturbation_ids: tuple[str, ...] = ()
    held_out_construct_ids: tuple[str, ...] = ()
    held_out_target_ids: tuple[str, ...] = ()
    held_out_context_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "train_series_ids",
            "validation_series_ids",
            "train_checkpoint_ids",
            "validation_checkpoint_ids",
            "train_observation_ids",
            "validation_observation_ids",
            "held_out_series",
            "held_out_checkpoints",
            "held_out_observations",
            "held_out_subject_ids",
            "held_out_experimental_unit_ids",
            "held_out_perturbation_ids",
            "held_out_construct_ids",
            "held_out_target_ids",
            "held_out_context_ids",
        ):
            values = tuple(str(value) for value in getattr(self, name))
            if any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(f"SplitPlan.{name} must contain unique nonempty IDs.")
            object.__setattr__(self, name, values)
        split_id = str(self.split_id)
        if not split_id:
            raise ValueError("SplitPlan.split_id must be nonempty.")
        object.__setattr__(self, "split_id", split_id)
        if self.source not in {"held_out", "train_self_eval"}:
            raise ValueError(f"Unknown split source {self.source!r}.")
        if self.strategy not in {
            "context_group_holdout",
            "checkpoint_holdout",
            "within_embedding_holdout",
            "train_self_eval",
        }:
            raise ValueError(f"Unknown split strategy {self.strategy!r}.")
        if self.task_kind not in {
            "series_generalization",
            "subject_generalization",
            "experimental_unit_generalization",
            "guide_within_target_generalization",
            "target_generalization",
            "perturbation_generalization",
            "context_generalization",
            "checkpoint_interpolation",
            "checkpoint_extrapolation",
            "combination_generalization",
            "train_self_evaluation",
        }:
            raise ValueError(f"Unknown longitudinal Perturb-seq task kind {self.task_kind!r}.")
        protocols = {
            "external_frozen",
            "shared_all_observations",
            "shared_source_only",
            "nested_by_subject",
            "nested_by_perturbation",
            "nested_by_checkpoint",
            "fully_nested",
        }
        if self.representation_protocol not in protocols:
            raise ValueError(f"Unknown representation protocol {self.representation_protocol!r}.")
        if (
            self.representation_scope == "nested"
            and self.representation_protocol == "shared_all_observations"
        ):
            object.__setattr__(self, "representation_protocol", "fully_nested")
        protocol_scope = (
            "nested"
            if self.representation_protocol.startswith("nested_")
            or self.representation_protocol == "fully_nested"
            else "shared"
        )
        if self.representation_scope != protocol_scope:
            raise ValueError(
                "SplitPlan representation_scope disagrees with representation_protocol."
            )

    @property
    def train_measure_ids(self) -> tuple[str, ...]:
        """Compatibility name used by the compact-v3 numerical runtime."""
        return self.train_series_ids

    @property
    def validation_measure_ids(self) -> tuple[str, ...]:
        """Compatibility name used by the compact-v3 numerical runtime."""
        return self.validation_series_ids

    @property
    def train_time_labels(self) -> tuple[str, ...]:
        """Compatibility name used by the compact-v3 numerical runtime."""
        return self.train_checkpoint_ids

    @property
    def validation_time_labels(self) -> tuple[str, ...]:
        """Compatibility name used by the compact-v3 numerical runtime."""
        return self.validation_checkpoint_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "train_selection": _selection_dict(self.train_selection),
            "validation_selection": _selection_dict(self.validation_selection),
            "train_series_ids": list(self.train_series_ids),
            "validation_series_ids": list(self.validation_series_ids),
            "train_checkpoint_ids": list(self.train_checkpoint_ids),
            "validation_checkpoint_ids": list(self.validation_checkpoint_ids),
            "train_observation_ids": list(self.train_observation_ids),
            "validation_observation_ids": list(self.validation_observation_ids),
            "held_out_series": list(self.held_out_series),
            "held_out_checkpoints": list(self.held_out_checkpoints),
            "held_out_observations": list(self.held_out_observations),
            "source": self.source,
            "strategy": self.strategy,
            "representation_scope": self.representation_scope,
            "representation_evaluation": self.representation_evaluation,
            "task_kind": self.task_kind,
            "representation_protocol": self.representation_protocol,
            "held_out_subject_ids": list(self.held_out_subject_ids),
            "held_out_experimental_unit_ids": list(self.held_out_experimental_unit_ids),
            "held_out_perturbation_ids": list(self.held_out_perturbation_ids),
            "held_out_construct_ids": list(self.held_out_construct_ids),
            "held_out_target_ids": list(self.held_out_target_ids),
            "held_out_context_ids": list(self.held_out_context_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SplitPlan:
        def selection(value: Mapping[str, Any]) -> SelectionSpec:
            from .study import ReplicatePolicy

            raw = dict(value)
            raw["replicate_policy"] = ReplicatePolicy.from_dict(
                raw.get("replicate_policy", {"mode": "reject"})
            )
            return SelectionSpec(**raw)

        return cls(
            split_id=str(payload["split_id"]),
            train_selection=selection(payload["train_selection"]),
            validation_selection=selection(payload["validation_selection"]),
            train_series_ids=tuple(payload["train_series_ids"]),
            validation_series_ids=tuple(payload["validation_series_ids"]),
            train_checkpoint_ids=tuple(payload["train_checkpoint_ids"]),
            validation_checkpoint_ids=tuple(payload["validation_checkpoint_ids"]),
            train_observation_ids=tuple(payload.get("train_observation_ids", ())),
            validation_observation_ids=tuple(payload.get("validation_observation_ids", ())),
            held_out_series=tuple(payload.get("held_out_series", ())),
            held_out_checkpoints=tuple(payload.get("held_out_checkpoints", ())),
            held_out_observations=tuple(payload.get("held_out_observations", ())),
            source=payload["source"],
            strategy=payload["strategy"],
            representation_scope=payload["representation_scope"],
            representation_evaluation=payload.get(
                "representation_evaluation",
                "transductive" if payload["representation_scope"] == "shared" else "inductive",
            ),
            task_kind=payload.get(
                "task_kind",
                "train_self_evaluation"
                if payload["source"] == "train_self_eval"
                else (
                    "checkpoint_interpolation"
                    if payload["strategy"] == "checkpoint_holdout"
                    else "perturbation_generalization"
                ),
            ),
            representation_protocol=payload.get(
                "representation_protocol",
                "shared_all_observations"
                if payload["representation_scope"] == "shared"
                else "fully_nested",
            ),
            held_out_subject_ids=tuple(payload.get("held_out_subject_ids", ())),
            held_out_experimental_unit_ids=tuple(payload.get("held_out_experimental_unit_ids", ())),
            held_out_perturbation_ids=tuple(payload.get("held_out_perturbation_ids", ())),
            held_out_construct_ids=tuple(payload.get("held_out_construct_ids", ())),
            held_out_target_ids=tuple(payload.get("held_out_target_ids", ())),
            held_out_context_ids=tuple(payload.get("held_out_context_ids", ())),
        )


@dataclass(frozen=True)
class _SplitInputs:
    series_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    source_checkpoint_id: str
    observed_by_checkpoint: Mapping[str, tuple[str, ...]]
    observation_ids_by_pair: Mapping[tuple[str, str], tuple[str, ...]]
    metadata: pd.DataFrame
    has_compositions: bool
    semantic_hash: str
    selection: SelectionSpec
    representation_protocol: RepresentationProtocol
    enforce_representation_protocol: bool


def _view_inputs(view: StudyView) -> _SplitInputs:
    design = view.study.design
    checkpoints = design.ordered_checkpoint_ids
    observations = view.observations()
    support = view.study.support_index._unsafe_view()
    available_ids = set(
        support.loc[
            support["representation_id"].eq(view.representation_id) & support["available"],
            "observation_id",
        ].astype(str)
    )
    observed = observations.loc[observations["observation_id"].isin(available_ids)]
    order = {series_id: index for index, series_id in enumerate(view.series_ids)}
    observed_by_checkpoint = {
        checkpoint_id: tuple(
            sorted(
                set(
                    observed.loc[observed["checkpoint_id"].eq(checkpoint_id), "series_id"].astype(
                        str
                    )
                ),
                key=order.__getitem__,
            )
        )
        for checkpoint_id in checkpoints
    }
    ids_by_pair = {
        (str(checkpoint_id), str(series_id)): tuple(rows["observation_id"].astype(str))
        for (checkpoint_id, series_id), rows in observations.groupby(
            ["checkpoint_id", "series_id"], observed=True, sort=False
        )
    }

    series = view.study.series._unsafe_view()
    series = series.loc[series["series_id"].isin(view.series_ids)].copy()
    is_lps = hasattr(view.study, "perturbations")
    perturbation_key = "perturbation_id" if is_lps else "condition_id"
    perturbations = (
        view.study.perturbations._unsafe_view().set_index("perturbation_id")
        if is_lps
        else view.study.conditions._unsafe_view().set_index("condition_id")
    )
    effect_binding = view.effect_binding()
    if effect_binding.empty:
        raise ValueError("Compact split planning requires a selected effect binding catalog.")
    effect_by_perturbation = (
        effect_binding.set_index(perturbation_key)["effect_id"].astype(str).to_dict()
    )
    component_by_perturbation: dict[str, dict[str, str]] = {}
    components = getattr(view.study, "perturbation_components", None)
    if components is not None:
        for perturbation_id, rows in components._unsafe_view().groupby(
            "perturbation_id", observed=True, sort=False
        ):
            record: dict[str, str] = {}
            constructs = tuple(dict.fromkeys(rows["construct_id"].astype(str)))
            targets = tuple(dict.fromkeys(rows["target_id"].astype(str)))
            if len(constructs) == 1:
                record["construct_id"] = constructs[0]
                record["guide_id"] = constructs[0]
            if len(targets) == 1:
                record["target_id"] = targets[0]
                record["target_gene"] = targets[0]
            component_by_perturbation[str(perturbation_id)] = record
    source = (
        observations.loc[observations["checkpoint_id"].eq(design.source_checkpoint_id)]
        .drop_duplicates("series_id")
        .set_index("series_id")
    )
    rows: list[dict[str, Any]] = []
    for row in series.itertuples(index=False):
        values = row._asdict()
        perturbation_id = str(values[perturbation_key])
        perturbation = perturbations.loc[perturbation_id]
        source_observation = source.loc[row.series_id]
        raw_context = source_observation.get("context_id")
        record: dict[str, Any] = {
            "measure_id": str(row.series_id),
            "sample_id": str(row.subject_id),
            "subject_id": str(row.subject_id),
            "perturbation_id": perturbation_id,
            "embedding_id": effect_by_perturbation[perturbation_id],
            "context_group_id": (str(row.subject_id) if pd.isna(raw_context) else str(raw_context)),
        }
        for column in ("experimental_unit_id", "biological_replicate_id"):
            if column in values and pd.notna(values[column]):
                record[column] = str(values[column])
        if not pd.isna(raw_context):
            record["context_id"] = str(raw_context)
        record.update(component_by_perturbation.get(perturbation_id, {}))
        for column in ("guide_id", "target_gene"):
            if column in perturbation.index and pd.notna(perturbation[column]):
                record[column] = str(perturbation[column])
        rows.append(record)
    metadata = pd.DataFrame(rows).set_index("measure_id", drop=False)
    compositions = view.compositions()
    selected_support = support.loc[
        support["representation_id"].eq(view.representation_id)
        & support["observation_id"].isin(view.observation_ids),
        ["observation_id", "representation_id", "available"],
    ]
    perturbation_frame = (
        view.perturbations()
        if is_lps
        else view.study.conditions._unsafe_view().loc[
            view.study.conditions._unsafe_view()["condition_id"].isin(series[perturbation_key])
        ]
    )
    split_basis = _canonical_hash(
        {
            "study_id": view.study.manifest.study_id,
            "design": _canonical_value(view.study.design),
            "perturbations": _frame_digest(perturbation_frame),
            "series": _frame_digest(series),
            "observations": _frame_digest(observations),
            "support_availability": _frame_digest(selected_support),
            "effect_binding": _frame_digest(effect_binding),
            "reference_binding": _frame_digest(view.reference_binding()),
            "compositions": None if compositions.empty else _frame_digest(compositions),
            "selection": _selection_dict(view.selection),
            "representation_id": view.representation_id,
        }
    )
    return _SplitInputs(
        series_ids=view.series_ids,
        checkpoint_ids=checkpoints,
        source_checkpoint_id=design.source_checkpoint_id,
        observed_by_checkpoint=MappingProxyType(observed_by_checkpoint),
        observation_ids_by_pair=MappingProxyType(ids_by_pair),
        metadata=metadata,
        has_compositions=not compositions.empty,
        semantic_hash=split_basis,
        selection=view.selection,
        representation_protocol=getattr(view.representation, "scope_mode", "external_frozen"),
        enforce_representation_protocol=hasattr(view.study, "perturbations"),
    )


def _trajectory_inputs(data: TrajectoryData) -> _SplitInputs:
    ids_by_pair = {
        (str(checkpoint_id), str(series_id)): (f"{series_id}@{checkpoint_id}",)
        for checkpoint_id in data.axis.labels
        for series_id in data.measures[checkpoint_id]
    }
    metadata = data.measure_meta.set_index("measure_id", drop=False)
    semantic_payload = {
        "series_ids": list(data.measure_ids),
        "checkpoints": list(data.axis.labels),
        "metadata": data.measure_meta.to_dict(orient="records"),
        "representation": data.representation.to_dict(),
    }
    return _SplitInputs(
        series_ids=data.measure_ids,
        checkpoint_ids=data.axis.labels,
        source_checkpoint_id=data.axis.source,
        observed_by_checkpoint=MappingProxyType(
            {label: tuple(data.measures[label]) for label in data.axis.labels}
        ),
        observation_ids_by_pair=MappingProxyType(ids_by_pair),
        metadata=metadata,
        has_compositions=bool(data.count_blocks),
        semantic_hash=_canonical_hash(semantic_payload),
        selection=SelectionSpec(),
        representation_protocol=(
            "fully_nested"
            if data.representation.fit_scope in {"training_fold_source", "training_split"}
            else "external_frozen"
        ),
        enforce_representation_protocol=False,
    )


def _validate_holdout_embeddings(
    metadata: pd.DataFrame,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> None:
    if not train_ids or not validation_ids:
        raise ValueError(
            "Explicit context-group validation requires nonempty train and holdout sets."
        )
    train_embeddings = set(metadata.loc[list(train_ids), "embedding_id"])
    validation_embeddings = set(metadata.loc[list(validation_ids), "embedding_id"])
    missing = validation_embeddings - train_embeddings
    if missing:
        raise ValueError(
            "Validation embeddings must be represented in training; "
            f"missing={sorted(map(str, missing))[:5]}."
        )


def _selection_for(
    inputs: _SplitInputs,
    series_ids: tuple[str, ...],
    target_checkpoint_ids: tuple[str, ...],
) -> SelectionSpec:
    checkpoints = tuple(
        checkpoint_id
        for checkpoint_id in inputs.checkpoint_ids
        if checkpoint_id == inputs.source_checkpoint_id or checkpoint_id in target_checkpoint_ids
    )
    return replace(
        inputs.selection,
        series_ids=series_ids,
        observation_ids=None,
        checkpoint_ids=checkpoints,
    )


def _observation_ids(
    inputs: _SplitInputs,
    series_ids: tuple[str, ...],
    target_checkpoint_ids: tuple[str, ...],
) -> tuple[str, ...]:
    checkpoints = (inputs.source_checkpoint_id, *target_checkpoint_ids)
    return tuple(
        observation_id
        for checkpoint_id in checkpoints
        for series_id in series_ids
        for observation_id in inputs.observation_ids_by_pair.get((checkpoint_id, series_id), ())
    )


def _finalize_plan(
    inputs: _SplitInputs,
    *,
    train: tuple[str, ...],
    validation: tuple[str, ...],
    train_checkpoints: tuple[str, ...],
    validation_checkpoints: tuple[str, ...],
    source: SplitSource,
    strategy: SplitStrategy,
    representation_scope: Literal["shared", "nested"],
    task_kind: SplitTaskKind | None = None,
) -> SplitPlan:
    held_out_series = (
        validation if source == "held_out" and strategy != "checkpoint_holdout" else ()
    )
    held_out_checkpoints = validation_checkpoints if strategy == "checkpoint_holdout" else ()
    train_observations = _observation_ids(inputs, train, train_checkpoints)
    validation_observations = _observation_ids(inputs, validation, validation_checkpoints)
    source_ids = set(_observation_ids(inputs, validation, ()))
    held_out_observations = (
        tuple(value for value in validation_observations if value not in source_ids)
        if source == "held_out"
        else ()
    )
    train_metadata = inputs.metadata.loc[list(train)]
    validation_metadata = inputs.metadata.loc[list(validation)]

    def held_out_values(column: str) -> tuple[str, ...]:
        if column not in inputs.metadata:
            return ()
        training = set(train_metadata[column].dropna().astype(str))
        values = tuple(dict.fromkeys(validation_metadata[column].dropna().astype(str)))
        return tuple(value for value in values if value not in training)

    held_out_subject_ids = held_out_values("subject_id")
    held_out_experimental_unit_ids = held_out_values("experimental_unit_id")
    held_out_perturbation_ids = held_out_values("perturbation_id")
    held_out_construct_ids = held_out_values("construct_id")
    held_out_target_ids = held_out_values("target_id")
    held_out_context_ids = held_out_values("context_id")
    if task_kind is None:
        if source == "train_self_eval":
            task_kind = "train_self_evaluation"
        elif strategy == "checkpoint_holdout":
            final_checkpoint = inputs.checkpoint_ids[-1]
            task_kind = (
                "checkpoint_extrapolation"
                if final_checkpoint in validation_checkpoints
                else "checkpoint_interpolation"
            )
        elif held_out_subject_ids:
            task_kind = "subject_generalization"
        elif held_out_experimental_unit_ids:
            task_kind = "experimental_unit_generalization"
        elif held_out_target_ids:
            task_kind = "target_generalization"
        elif held_out_construct_ids:
            task_kind = "guide_within_target_generalization"
        elif held_out_perturbation_ids:
            task_kind = "perturbation_generalization"
        elif held_out_context_ids:
            task_kind = "context_generalization"
        else:
            task_kind = "series_generalization"

    protocol = inputs.representation_protocol
    protocol_scope: Literal["shared", "nested"] = (
        "nested" if protocol.startswith("nested_") or protocol == "fully_nested" else "shared"
    )
    if representation_scope != protocol_scope:
        if inputs.enforce_representation_protocol:
            raise ValueError(
                f"validation.representation_scope={representation_scope!r} disagrees with "
                f"representation protocol {protocol!r}."
            )
        protocol = "fully_nested" if representation_scope == "nested" else "shared_all_observations"
    if protocol == "external_frozen" or protocol == "fully_nested":
        representation_evaluation = "inductive"
    elif protocol == "shared_all_observations":
        representation_evaluation = "transductive"
    elif protocol == "shared_source_only":
        representation_evaluation = (
            "inductive" if task_kind.startswith("checkpoint_") else "transductive"
        )
    elif protocol == "nested_by_subject":
        representation_evaluation = (
            "inductive" if task_kind == "subject_generalization" else "transductive"
        )
    elif protocol == "nested_by_perturbation":
        representation_evaluation = (
            "inductive"
            if task_kind
            in {
                "guide_within_target_generalization",
                "target_generalization",
                "perturbation_generalization",
                "combination_generalization",
            }
            else "transductive"
        )
    else:
        representation_evaluation = (
            "inductive" if task_kind.startswith("checkpoint_") else "transductive"
        )
    payload = {
        "study": inputs.semantic_hash,
        "train_series_ids": list(train),
        "validation_series_ids": list(validation),
        "train_checkpoint_ids": list(train_checkpoints),
        "validation_checkpoint_ids": list(validation_checkpoints),
        "train_observation_ids": list(train_observations),
        "validation_observation_ids": list(validation_observations),
        "source": source,
        "strategy": strategy,
        "representation_scope": representation_scope,
        "representation_protocol": protocol,
        "representation_evaluation": representation_evaluation,
        "task_kind": task_kind,
        "held_out_subject_ids": list(held_out_subject_ids),
        "held_out_experimental_unit_ids": list(held_out_experimental_unit_ids),
        "held_out_perturbation_ids": list(held_out_perturbation_ids),
        "held_out_construct_ids": list(held_out_construct_ids),
        "held_out_target_ids": list(held_out_target_ids),
        "held_out_context_ids": list(held_out_context_ids),
    }
    split_id = f"sha256:{_canonical_hash(payload)}"
    return SplitPlan(
        split_id=split_id,
        train_selection=_selection_for(inputs, train, train_checkpoints),
        validation_selection=_selection_for(inputs, validation, validation_checkpoints),
        train_series_ids=train,
        validation_series_ids=validation,
        train_checkpoint_ids=train_checkpoints,
        validation_checkpoint_ids=validation_checkpoints,
        train_observation_ids=train_observations,
        validation_observation_ids=validation_observations,
        held_out_series=held_out_series,
        held_out_checkpoints=held_out_checkpoints,
        held_out_observations=held_out_observations,
        source=source,
        strategy=strategy,
        representation_scope=representation_scope,
        representation_evaluation=representation_evaluation,
        task_kind=task_kind,
        representation_protocol=protocol,
        held_out_subject_ids=held_out_subject_ids,
        held_out_experimental_unit_ids=held_out_experimental_unit_ids,
        held_out_perturbation_ids=held_out_perturbation_ids,
        held_out_construct_ids=held_out_construct_ids,
        held_out_target_ids=held_out_target_ids,
        held_out_context_ids=held_out_context_ids,
    )


def _plan(
    inputs: _SplitInputs,
    validation_config: Any,
    *,
    seed: int,
    requested: SplitSpec | None = None,
) -> SplitPlan:
    downstream = tuple(
        checkpoint_id
        for checkpoint_id in inputs.checkpoint_ids
        if checkpoint_id != inputs.source_checkpoint_id
    )
    eligible = tuple(
        series_id
        for series_id in inputs.series_ids
        if any(series_id in inputs.observed_by_checkpoint[label] for label in downstream)
    )
    if not eligible:
        raise ValueError("No source series has a downstream observation.")
    eligible_set = set(eligible)
    metadata = inputs.metadata
    strategy = str(validation_config.strategy)
    values = tuple(str(value) for value in validation_config.values)
    fraction = float(validation_config.fraction)
    representation_scope = validation_config.representation_scope

    if requested is not None:
        representation_scope = requested.representation_scope
        if requested.strategy != "none":
            if requested.strategy in {"context_group", "checkpoint"}:
                strategy = requested.strategy
                values = tuple(requested.validation_values or ())
                fraction = 0.0
            elif requested.strategy in {
                "measure",
                "sample",
                "subject",
                "experimental_unit",
                "guide",
                "construct",
                "target",
                "perturbation",
                "combination",
                "context",
                "embedding",
            }:
                column = {
                    "measure": "measure_id",
                    "sample": "sample_id",
                    "subject": "subject_id",
                    "experimental_unit": "experimental_unit_id",
                    "guide": "guide_id",
                    "construct": "construct_id",
                    "target": "target_id",
                    "perturbation": "perturbation_id",
                    "combination": "perturbation_id",
                    "context": "context_id",
                    "embedding": "embedding_id",
                }[requested.strategy]
                if column not in metadata:
                    raise ValueError(f"Split strategy {requested.strategy!r} requires {column!r}.")
                selected = set(requested.validation_values or ())
                unknown = selected - set(metadata[column].astype(str))
                if unknown:
                    raise ValueError(
                        f"Unknown validation {requested.strategy} values: {sorted(unknown)}"
                    )
                validation_ids = tuple(
                    series_id
                    for series_id in inputs.series_ids
                    if str(metadata.loc[series_id, column]) in selected
                )
                validation_outcomes = tuple(
                    series_id for series_id in validation_ids if series_id in eligible_set
                )
                train_ids = tuple(
                    series_id for series_id in inputs.series_ids if series_id not in validation_ids
                )
                if not train_ids or not validation_outcomes:
                    raise ValueError(
                        f"Explicit {requested.strategy} validation requires nonempty train "
                        "and observed holdout sets."
                    )
                if requested.strategy not in {"target", "perturbation", "combination"}:
                    _validate_holdout_embeddings(metadata, train_ids, validation_outcomes)
                explicit_task: SplitTaskKind | None = {
                    "sample": "subject_generalization",
                    "subject": "subject_generalization",
                    "experimental_unit": "experimental_unit_generalization",
                    "guide": "guide_within_target_generalization",
                    "construct": "guide_within_target_generalization",
                    "target": "target_generalization",
                    "perturbation": "perturbation_generalization",
                    "combination": "combination_generalization",
                    "context": "context_generalization",
                }.get(requested.strategy)
                return _finalize_plan(
                    inputs,
                    train=train_ids,
                    validation=validation_ids,
                    train_checkpoints=downstream,
                    validation_checkpoints=downstream,
                    source="held_out",
                    strategy="within_embedding_holdout",
                    representation_scope=representation_scope,
                    task_kind=explicit_task,
                )
            else:
                raise ValueError(f"Compact split planning does not support {requested.strategy!r}.")

    if strategy == "checkpoint":
        requested_values = set(values)
        validation_times = tuple(label for label in downstream if label in requested_values)
        unknown = requested_values - set(downstream)
        if unknown:
            raise ValueError(f"Unknown validation checkpoints: {sorted(unknown)}")
        train_times = tuple(label for label in downstream if label not in requested_values)
        validation_ids = tuple(
            series_id
            for series_id in inputs.series_ids
            if any(series_id in inputs.observed_by_checkpoint[label] for label in validation_times)
        )
        if not validation_ids:
            raise ValueError("Explicit checkpoint validation has no observed series.")
        return _finalize_plan(
            inputs,
            train=inputs.series_ids,
            validation=validation_ids,
            train_checkpoints=train_times,
            validation_checkpoints=validation_times,
            source="held_out",
            strategy="checkpoint_holdout",
            representation_scope=representation_scope,
        )

    if strategy == "context_group":
        available = set(metadata["context_group_id"].astype(str))
        selected = set(values)
        unknown = selected - available
        if unknown:
            raise ValueError(f"Unknown validation context groups: {sorted(unknown)}")
        validation_ids = tuple(
            series_id
            for series_id in eligible
            if str(metadata.loc[series_id, "context_group_id"]) in selected
        )
        train_ids = tuple(
            series_id
            for series_id in inputs.series_ids
            if str(metadata.loc[series_id, "context_group_id"]) not in selected
        )
        _validate_holdout_embeddings(metadata, train_ids, validation_ids)
        return _finalize_plan(
            inputs,
            train=train_ids,
            validation=validation_ids,
            train_checkpoints=downstream,
            validation_checkpoints=downstream,
            source="held_out",
            strategy="context_group_holdout",
            representation_scope=representation_scope,
        )

    if strategy == "train_self_eval" or fraction <= 0 or len(eligible) < 2:
        return _finalize_plan(
            inputs,
            train=inputs.series_ids,
            validation=eligible,
            train_checkpoints=downstream,
            validation_checkpoints=downstream,
            source="train_self_eval",
            strategy="train_self_eval",
            representation_scope=representation_scope,
        )

    context_groups = tuple(
        dict.fromkeys(metadata.loc[list(inputs.series_ids), "context_group_id"].tolist())
    )
    if len(context_groups) > 1:
        holdout_count = min(
            max(1, int(round(len(context_groups) * fraction))),
            len(context_groups) - 1,
        )
        ordered_groups = sorted(
            context_groups,
            key=lambda value: hashlib.sha256(f"{seed}:group:{value}".encode()).hexdigest(),
        )
        for offset in range(len(ordered_groups)):
            held_out: set[str] = set()
            rotated = ordered_groups[offset:] + ordered_groups[:offset]
            for candidate in rotated:
                trial = held_out | {candidate}
                validation_ids = tuple(
                    series_id
                    for series_id in eligible
                    if metadata.loc[series_id, "context_group_id"] in trial
                )
                train_ids = tuple(
                    series_id
                    for series_id in inputs.series_ids
                    if metadata.loc[series_id, "context_group_id"] not in trial
                )
                train_embeddings = {
                    metadata.loc[series_id, "embedding_id"] for series_id in train_ids
                }
                validation_embeddings = {
                    metadata.loc[series_id, "embedding_id"] for series_id in validation_ids
                }
                if train_ids and validation_ids and validation_embeddings <= train_embeddings:
                    held_out = trial
                if len(held_out) == holdout_count:
                    return _finalize_plan(
                        inputs,
                        train=train_ids,
                        validation=validation_ids,
                        train_checkpoints=downstream,
                        validation_checkpoints=downstream,
                        source="held_out",
                        strategy="context_group_holdout",
                        representation_scope=representation_scope,
                    )

    if inputs.has_compositions:
        return _finalize_plan(
            inputs,
            train=inputs.series_ids,
            validation=eligible,
            train_checkpoints=downstream,
            validation_checkpoints=downstream,
            source="train_self_eval",
            strategy="train_self_eval",
            representation_scope=representation_scope,
        )

    validation_values: list[str] = []
    for embedding_id, rows in metadata.loc[list(eligible)].groupby("embedding_id", observed=True):
        ids = rows.index.tolist()
        guides: dict[str, list[str]] = {}
        for series_id in ids:
            guide_id = (
                str(metadata.loc[series_id, "guide_id"])
                if "guide_id" in metadata
                else str(series_id)
            )
            guides.setdefault(guide_id, []).append(series_id)
        if len(guides) > 1:
            holdout_count = min(max(1, int(round(len(guides) * fraction))), len(guides) - 1)
            ordered_guides = sorted(
                guides,
                key=lambda value: hashlib.sha256(
                    f"{seed}:guide:{embedding_id}:{value}".encode()
                ).hexdigest(),
            )
            for guide_id in ordered_guides[:holdout_count]:
                validation_values.extend(guides[guide_id])
        elif len(ids) > 1:
            holdout_count = min(max(1, int(round(len(ids) * fraction))), len(ids) - 1)
            ordered_ids = sorted(
                ids,
                key=lambda value: hashlib.sha256(
                    f"{seed}:measure:{embedding_id}:{value}".encode()
                ).hexdigest(),
            )
            validation_values.extend(ordered_ids[:holdout_count])
    validation_set = set(validation_values)
    validation_ids = tuple(value for value in eligible if value in validation_set)
    if not validation_ids:
        return _finalize_plan(
            inputs,
            train=inputs.series_ids,
            validation=eligible,
            train_checkpoints=downstream,
            validation_checkpoints=downstream,
            source="train_self_eval",
            strategy="train_self_eval",
            representation_scope=representation_scope,
        )
    train_ids = tuple(value for value in inputs.series_ids if value not in validation_set)
    return _finalize_plan(
        inputs,
        train=train_ids,
        validation=validation_ids,
        train_checkpoints=downstream,
        validation_checkpoints=downstream,
        source="held_out",
        strategy="within_embedding_holdout",
        representation_scope=representation_scope,
    )


def plan_compact_split(
    view: StudyView,
    config: Any,
    requested: SplitSpec | None = None,
) -> SplitPlan:
    """Plan compact-v3's exact split directly from semantic study tables."""
    return _plan(
        _view_inputs(view),
        config.validation,
        seed=int(config.training.seed),
        requested=requested,
    )


def plan_compact_trajectory_split(
    data: TrajectoryData,
    config: Any,
    *,
    seed: int | None = None,
) -> SplitPlan:
    """Compatibility planner for callers that still supply ``TrajectoryData``."""
    settings = config.recipe_config
    return _plan(
        _trajectory_inputs(data),
        settings.validation,
        seed=int(settings.training.seed if seed is None else seed),
    )


def validate_representation_scope(view: StudyView, split: SplitPlan) -> None:
    """Reject representation provenance that leaks across the declared LPS task."""
    representation = view.representation
    protocol = getattr(representation, "scope_mode", None)
    if protocol is None:
        protocol = "fully_nested" if representation.fit_split_id is not None else "external_frozen"
    if protocol != split.representation_protocol:
        raise ValueError(
            f"Split representation protocol {split.representation_protocol!r} disagrees with "
            f"representation {representation.representation_id!r} ({protocol!r})."
        )
    if not hasattr(view.study, "perturbations"):
        if split.representation_scope == "shared" or split.source != "held_out":
            return
        if split.strategy == "checkpoint_holdout":
            fit_checkpoints = set(representation.included_checkpoints)
            if not fit_checkpoints:
                raise ValueError(
                    "Nested checkpoint validation requires recorded representation checkpoints."
                )
            leaked = set(split.held_out_checkpoints) & fit_checkpoints
        else:
            if not representation.included_series:
                raise ValueError(
                    "Nested series validation requires recorded representation series."
                )
            leaked = set(split.held_out_series) & set(representation.included_series)
        if leaked:
            raise ValueError(
                f"Nested validation includes held-out representation series: {sorted(leaked)[:5]}"
            )
        return
    if split.source != "held_out" or protocol == "external_frozen":
        return

    fit_checkpoints = set(
        getattr(representation, "fit_checkpoint_ids", ()) or representation.included_checkpoints
    )
    fit_subjects = set(getattr(representation, "fit_subject_ids", ()))
    fit_perturbations = set(getattr(representation, "fit_perturbation_ids", ()))
    if protocol == "shared_all_observations":
        if split.representation_evaluation != "transductive":
            raise ValueError("A shared-all-observations representation is transductive.")
        return
    if protocol == "shared_source_only":
        if not fit_checkpoints:
            raise ValueError("shared_source_only requires recorded representation fit checkpoints.")
        non_source = fit_checkpoints - {view.study.design.source_checkpoint_id}
        if non_source:
            raise ValueError(
                f"shared_source_only includes non-source checkpoints: {sorted(non_source)}."
            )
        return

    if representation.fit_split_id != split.split_id:
        raise ValueError(
            "Nested representation fit_split_id must equal the exact content-addressed split."
        )
    expected_selection_hash = _canonical_hash(_selection_dict(split.train_selection))
    if (
        getattr(representation, "fit_selection_hash", None) is not None
        and representation.fit_selection_hash != expected_selection_hash
    ):
        raise ValueError("Representation fit_selection_hash disagrees with training selection.")
    if split.task_kind.startswith("checkpoint_"):
        if not fit_checkpoints:
            raise ValueError("Nested checkpoint evaluation requires recorded fit checkpoints.")
        leaked = set(split.held_out_checkpoints) & fit_checkpoints
        if leaked:
            raise ValueError(
                "Nested checkpoint validation includes held-out representation checkpoints: "
                f"{sorted(leaked)}"
            )
        return
    if split.task_kind == "subject_generalization":
        if not fit_subjects:
            raise ValueError("Nested subject evaluation requires recorded fit_subject_ids.")
        leaked = set(split.held_out_subject_ids) & fit_subjects
    elif split.task_kind in {
        "guide_within_target_generalization",
        "target_generalization",
        "perturbation_generalization",
        "combination_generalization",
    }:
        if not fit_perturbations:
            raise ValueError(
                "Nested perturbation evaluation requires recorded fit_perturbation_ids."
            )
        leaked = set(split.held_out_perturbation_ids) & fit_perturbations
    else:
        if not representation.included_series:
            raise ValueError("Nested evaluation requires recorded fit series.")
        leaked = set(split.held_out_series) & set(representation.included_series)
    if leaked:
        raise ValueError(
            f"Nested representation includes held-out identities: {sorted(leaked)[:5]}"
        )


def validate_split_plan(view: StudyView, split: SplitPlan) -> None:
    """Verify that a persisted plan is complete and content-bound to this view."""
    inputs = _view_inputs(view)
    downstream = set(inputs.checkpoint_ids) - {inputs.source_checkpoint_id}
    unknown_series = (set(split.train_series_ids) | set(split.validation_series_ids)) - set(
        inputs.series_ids
    )
    unknown_checkpoints = (
        set(split.train_checkpoint_ids) | set(split.validation_checkpoint_ids)
    ) - downstream
    if unknown_series or unknown_checkpoints:
        raise ValueError(
            "Split plan references identities outside the selected StudyView; "
            f"series={sorted(unknown_series)[:5]}, "
            f"checkpoints={sorted(unknown_checkpoints)[:5]}."
        )
    if not split.train_series_ids or not split.validation_series_ids:
        raise ValueError("Split plan requires nonempty training and validation series.")

    all_series = set(inputs.series_ids)
    all_downstream = set(inputs.checkpoint_ids) - {inputs.source_checkpoint_id}
    eligible = {
        series_id
        for series_id in inputs.series_ids
        if any(series_id in inputs.observed_by_checkpoint[label] for label in all_downstream)
    }
    train_series = set(split.train_series_ids)
    validation_series = set(split.validation_series_ids)
    train_checkpoints = set(split.train_checkpoint_ids)
    validation_checkpoints = set(split.validation_checkpoint_ids)
    if split.strategy == "checkpoint_holdout":
        expected_validation = {
            series_id
            for series_id in inputs.series_ids
            if any(
                series_id in inputs.observed_by_checkpoint[label]
                for label in validation_checkpoints
            )
        }
        valid_shape = (
            train_series == all_series
            and validation_series == expected_validation
            and not (train_checkpoints & validation_checkpoints)
            and train_checkpoints | validation_checkpoints == all_downstream
        )
    elif split.strategy == "train_self_eval":
        valid_shape = (
            train_series == all_series
            and validation_series == eligible
            and train_checkpoints == all_downstream
            and validation_checkpoints == all_downstream
        )
    else:
        valid_shape = (
            not (train_series & validation_series)
            and train_series | validation_series == all_series
            and train_checkpoints == all_downstream
            and validation_checkpoints == all_downstream
        )
    if not valid_shape:
        raise ValueError("Split plan partitions do not match its declared strategy.")
    if split.strategy not in {"checkpoint_holdout", "train_self_eval"} and split.task_kind not in {
        "target_generalization",
        "perturbation_generalization",
        "combination_generalization",
    }:
        validation_outcomes = tuple(
            series_id for series_id in split.validation_series_ids if series_id in eligible
        )
        _validate_holdout_embeddings(
            inputs.metadata,
            split.train_series_ids,
            validation_outcomes,
        )
    required_holdout = {
        "series_generalization": split.held_out_series,
        "subject_generalization": split.held_out_subject_ids,
        "experimental_unit_generalization": split.held_out_experimental_unit_ids,
        "guide_within_target_generalization": split.held_out_construct_ids,
        "target_generalization": split.held_out_target_ids,
        "perturbation_generalization": split.held_out_perturbation_ids,
        "context_generalization": split.held_out_context_ids,
        "combination_generalization": split.held_out_perturbation_ids,
        "checkpoint_interpolation": split.held_out_checkpoints,
        "checkpoint_extrapolation": split.held_out_checkpoints,
    }.get(split.task_kind)
    if split.source == "held_out" and required_holdout is not None and not required_holdout:
        raise ValueError(
            f"Split task {split.task_kind!r} has no corresponding held-out identities."
        )

    expected = _finalize_plan(
        inputs,
        train=split.train_series_ids,
        validation=split.validation_series_ids,
        train_checkpoints=split.train_checkpoint_ids,
        validation_checkpoints=split.validation_checkpoint_ids,
        source=split.source,
        strategy=split.strategy,
        representation_scope=split.representation_scope,
        task_kind=split.task_kind,
    )
    if expected != split:
        raise ValueError("Split plan is not content-bound to the selected StudyView.")


__all__ = [
    "RepresentationProtocol",
    "SplitPlan",
    "SplitTaskKind",
    "plan_compact_split",
    "plan_compact_trajectory_split",
    "validate_representation_scope",
    "validate_split_plan",
]
