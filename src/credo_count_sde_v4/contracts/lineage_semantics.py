"""Lineage evidence levels and their permitted result semantics."""

from __future__ import annotations

from enum import StrEnum

__all__ = (
    "LINEAGE_LEVEL_RANK",
    "LINEAGE_RESULT_MINIMUM_LEVEL",
    "LineageEvidenceLevel",
    "LineageResultSemantics",
    "lineage_level_at_least",
)


class LineageEvidenceLevel(StrEnum):
    """Strongest lineage observation physically present in a study."""

    L0_DESTRUCTIVE_SNAPSHOTS = "L0_destructive_snapshots"
    L1_STABLE_CLONE_BARCODES = "L1_stable_clone_barcodes"
    L2_HERITABLE_BARCODES = "L2_heritable_barcodes"
    L3_LIVE_OR_PAIRED_CELLS = "L3_live_or_paired_cells"


LINEAGE_LEVEL_RANK = {level: index for index, level in enumerate(LineageEvidenceLevel)}


class LineageResultSemantics(StrEnum):
    """Result language authorized by each lineage evidence level."""

    POPULATION_TRANSITION_ONLY = "population_transition_only"
    CLONE_RESOLVED_FATE = "clone_resolved_fate"
    ANCESTRAL_TREE = "ancestral_tree"
    DIRECT_CELL_PATH_VALIDATION = "direct_cell_path_validation"


LINEAGE_RESULT_MINIMUM_LEVEL = {
    LineageResultSemantics.POPULATION_TRANSITION_ONLY: (
        LineageEvidenceLevel.L0_DESTRUCTIVE_SNAPSHOTS
    ),
    LineageResultSemantics.CLONE_RESOLVED_FATE: LineageEvidenceLevel.L1_STABLE_CLONE_BARCODES,
    LineageResultSemantics.ANCESTRAL_TREE: LineageEvidenceLevel.L2_HERITABLE_BARCODES,
    LineageResultSemantics.DIRECT_CELL_PATH_VALIDATION: (
        LineageEvidenceLevel.L3_LIVE_OR_PAIRED_CELLS
    ),
}


def lineage_level_at_least(
    observed: LineageEvidenceLevel,
    required: LineageEvidenceLevel,
) -> bool:
    """Return whether the observed lineage channel supports a requested level."""

    return LINEAGE_LEVEL_RANK[observed] >= LINEAGE_LEVEL_RANK[required]
