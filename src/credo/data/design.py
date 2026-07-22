"""Storage-independent experimental design contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

AxisKind = Literal[
    "physical_time",
    "developmental_stage",
    "effect",
    "dose",
    "ordered_condition",
]
CheckpointRole = Literal["source", "intermediate", "target"]
Topology = Literal["chain", "star", "dag"]


def _identifier(value: str, field_name: str) -> str:
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{field_name} must be nonempty.")
    return normalized


@dataclass(frozen=True)
class AxisSpec:
    """One named coordinate axis in an experimental design."""

    axis_id: str
    kind: AxisKind
    unit: str | None = None
    ordered: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_id", _identifier(self.axis_id, "axis_id"))
        if self.kind not in {
            "physical_time",
            "developmental_stage",
            "effect",
            "dose",
            "ordered_condition",
        }:
            raise ValueError(f"Unsupported axis kind {self.kind!r}.")
        if self.unit is not None:
            object.__setattr__(self, "unit", _identifier(self.unit, "axis unit"))


@dataclass(frozen=True)
class Checkpoint:
    """One observed or modeled point in a study design."""

    checkpoint_id: str
    coordinates: Mapping[str, float | str]
    role: CheckpointRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", _identifier(self.checkpoint_id, "checkpoint_id"))
        coordinates = {str(key): value for key, value in self.coordinates.items()}
        if not coordinates or any(not key for key in coordinates):
            raise ValueError("Checkpoint coordinates must have nonempty axis identifiers.")
        if self.role not in {"source", "intermediate", "target"}:
            raise ValueError(f"Unsupported checkpoint role {self.role!r}.")
        object.__setattr__(self, "coordinates", MappingProxyType(coordinates))


@dataclass(frozen=True)
class Transition:
    """One directed edge between checkpoints."""

    transition_id: str
    source_checkpoint_id: str
    target_checkpoint_id: str

    def __post_init__(self) -> None:
        for name in ("transition_id", "source_checkpoint_id", "target_checkpoint_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.source_checkpoint_id == self.target_checkpoint_id:
            raise ValueError("A transition cannot connect a checkpoint to itself.")


@dataclass(frozen=True)
class StudyDesign:
    """Axes, checkpoints, and transition topology for one study."""

    axes: tuple[AxisSpec, ...]
    checkpoints: tuple[Checkpoint, ...]
    transitions: tuple[Transition, ...]
    topology: Topology = "chain"
    _checkpoint_by_id: Mapping[str, Checkpoint] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        checkpoints = tuple(self.checkpoints)
        transitions = tuple(self.transitions)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "transitions", transitions)
        if self.topology not in {"chain", "star", "dag"}:
            raise ValueError(f"Unsupported study topology {self.topology!r}.")
        if not axes:
            raise ValueError("StudyDesign requires at least one axis.")
        if len(checkpoints) < 2:
            raise ValueError("StudyDesign requires at least two checkpoints.")
        axis_ids = [axis.axis_id for axis in axes]
        checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in checkpoints]
        transition_ids = [transition.transition_id for transition in transitions]
        if len(axis_ids) != len(set(axis_ids)):
            raise ValueError("StudyDesign axis identifiers must be unique.")
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError("StudyDesign checkpoint identifiers must be unique.")
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("StudyDesign transition identifiers must be unique.")
        expected_coordinates = set(axis_ids)
        for checkpoint in checkpoints:
            if set(checkpoint.coordinates) != expected_coordinates:
                raise ValueError(
                    f"Checkpoint {checkpoint.checkpoint_id!r} coordinates must match all axes."
                )
        sources = [checkpoint for checkpoint in checkpoints if checkpoint.role == "source"]
        if len(sources) != 1:
            raise ValueError("StudyDesign requires exactly one source checkpoint.")
        if not any(checkpoint.role == "target" for checkpoint in checkpoints):
            raise ValueError("StudyDesign requires at least one target checkpoint.")
        known = set(checkpoint_ids)
        pairs: set[tuple[str, str]] = set()
        adjacency: dict[str, list[str]] = {checkpoint_id: [] for checkpoint_id in checkpoint_ids}
        indegree = {checkpoint_id: 0 for checkpoint_id in checkpoint_ids}
        for transition in transitions:
            pair = (transition.source_checkpoint_id, transition.target_checkpoint_id)
            if not set(pair) <= known:
                raise ValueError(
                    f"Transition {transition.transition_id!r} references an unknown checkpoint."
                )
            if pair in pairs:
                raise ValueError(f"Duplicate transition edge {pair!r}.")
            pairs.add(pair)
            adjacency[pair[0]].append(pair[1])
            indegree[pair[1]] += 1
        self._validate_acyclic(adjacency, indegree)
        if self.topology == "chain":
            self._validate_chain(adjacency, indegree, sources[0].checkpoint_id)
        object.__setattr__(
            self,
            "_checkpoint_by_id",
            MappingProxyType({checkpoint.checkpoint_id: checkpoint for checkpoint in checkpoints}),
        )

    @staticmethod
    def _validate_acyclic(adjacency: Mapping[str, list[str]], indegree: Mapping[str, int]) -> None:
        remaining = dict(indegree)
        queue = [node for node, degree in remaining.items() if degree == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for target in adjacency[node]:
                remaining[target] -= 1
                if remaining[target] == 0:
                    queue.append(target)
        if visited != len(remaining):
            raise ValueError("StudyDesign transitions must form an acyclic graph.")

    def _validate_chain(
        self,
        adjacency: Mapping[str, list[str]],
        indegree: Mapping[str, int],
        source: str,
    ) -> None:
        if len(self.transitions) != len(self.checkpoints) - 1:
            raise ValueError("A chain design requires exactly n_checkpoints - 1 transitions.")
        if indegree[source] != 0:
            raise ValueError("The source checkpoint cannot have an incoming transition.")
        if any(len(targets) > 1 for targets in adjacency.values()):
            raise ValueError("A chain checkpoint cannot have multiple outgoing transitions.")
        if any(degree > 1 for degree in indegree.values()):
            raise ValueError("A chain checkpoint cannot have multiple incoming transitions.")
        visited: set[str] = set()
        current = source
        while current not in visited:
            visited.add(current)
            targets = adjacency[current]
            if not targets:
                break
            current = targets[0]
        if len(visited) != len(self.checkpoints):
            raise ValueError("A chain design must connect every checkpoint from the source.")

    @property
    def axis_ids(self) -> tuple[str, ...]:
        return tuple(axis.axis_id for axis in self.axes)

    @property
    def checkpoint_ids(self) -> tuple[str, ...]:
        return tuple(checkpoint.checkpoint_id for checkpoint in self.checkpoints)

    @property
    def source_checkpoint_id(self) -> str:
        return next(
            checkpoint.checkpoint_id
            for checkpoint in self.checkpoints
            if checkpoint.role == "source"
        )

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        try:
            return self._checkpoint_by_id[str(checkpoint_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown checkpoint_id {checkpoint_id!r}.") from exc


__all__ = ["AxisSpec", "Checkpoint", "StudyDesign", "Transition"]
