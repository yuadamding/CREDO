"""Evaluator-only, donor/condition/guide-keyed effects (metric revision 2).

Observed controls enter truth contrasts only. The prediction-side control mean
must be supplied independently by the predictor. Missing matched controls and
non-informative effects remain explicit coverage losses, never automatic passes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..errors import ContractError

Array = NDArray[Any]


def _composition(values: Array) -> Array:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape):
        raise ContractError("Effect composition requires a nonempty cell-by-gene matrix.")
    depth = x.sum(axis=1)
    if not np.isfinite(x).all() or (x < 0).any() or (depth <= 0).any():
        raise ContractError("Effect composition requires finite positive-depth rows.")
    return x / depth[:, None]


def _validate_keys(
    rows: int,
    donor: Array,
    condition: Array,
    guide: Array,
    guide_to_target: Array,
    control_guides: tuple[int, ...],
) -> None:
    if any(
        np.asarray(x).shape != (rows,) or np.asarray(x).dtype.kind not in "iu" or np.any(x < 0)
        for x in (donor, condition, guide)
    ):
        raise ContractError("Nonnegative integer effect keys must align with rows.")
    mapping = np.asarray(guide_to_target)
    if (
        mapping.ndim != 1
        or not len(mapping)
        or mapping.dtype.kind not in "iu"
        or (mapping < 0).any()
    ):
        raise ContractError("Guide target catalog must be a nonnegative integer vector.")
    if np.any(guide >= len(mapping)):
        raise ContractError("Guide keys exceed the explicit target catalog.")
    if (
        not control_guides
        or len(set(control_guides)) != len(control_guides)
        or any(type(g) is not int or g < 0 or g >= len(mapping) for g in control_guides)
    ):
        raise ContractError("Explicit control guides must belong to the target catalog.")


@dataclass(frozen=True)
class GeneSignReport:
    accuracy: float | None
    units: tuple[dict[str, Any], ...]
    informative_gene_effects: int
    candidate_gene_effects: int
    supported_units: int
    total_units: int
    minimum_abs_log_effect: float
    log_pseudocount: float
    metric_semantics: str = "donor_condition_guide_gene_sign_v2_equal_cell_composition"


def gene_sign_report(
    *,
    counts: Array,
    predicted_mean: Array,
    predicted_reference_mean: Array,
    donor: Array,
    condition: Array,
    guide: Array,
    guide_to_target: Array,
    control_guides: tuple[int, ...],
    minimum_abs_log_effect: float = 0.05,
    log_pseudocount: float = 1e-8,
) -> GeneSignReport:
    if not np.isfinite(minimum_abs_log_effect) or minimum_abs_log_effect < 0:
        raise ContractError("Informative-effect threshold must be fixed and nonnegative.")
    if not np.isfinite(log_pseudocount) or log_pseudocount <= 0:
        raise ContractError("Log pseudocount must be fixed and positive.")
    observed, predicted, reference = map(
        _composition, (counts, predicted_mean, predicted_reference_mean)
    )
    if observed.shape != predicted.shape or observed.shape != reference.shape:
        raise ContractError("Observed and predicted effect rows/features must align.")
    _validate_keys(len(observed), donor, condition, guide, guide_to_target, control_guides)
    controls = np.isin(guide, control_guides)
    units: list[dict[str, Any]] = []
    for d, c in sorted(set(zip(donor.tolist(), condition.tolist(), strict=True))):
        stratum = (donor == d) & (condition == c)
        ntc = stratum & controls
        for g in sorted(set(guide[stratum & ~controls].tolist())):
            selected = stratum & (guide == g)
            row: dict[str, Any] = {
                "donor": int(d),
                "condition": int(c),
                "guide": int(g),
                "target": int(guide_to_target[g]),
                "cells": int(selected.sum()),
                "control_cells": int(ntc.sum()),
                "genes": observed.shape[1],
                "informative_genes": 0,
                "accuracy": None,
                "status": "missing_donor_condition_controls",
            }
            if ntc.any():
                truth = np.log(observed[selected].mean(axis=0) + log_pseudocount)
                truth -= np.log(observed[ntc].mean(axis=0) + log_pseudocount)
                estimate = np.log(predicted[selected].mean(axis=0) + log_pseudocount)
                estimate -= np.log(reference[selected].mean(axis=0) + log_pseudocount)
                informative = np.abs(truth) >= minimum_abs_log_effect
                row["informative_genes"] = int(informative.sum())
                row["status"] = "no_informative_effects"
                if informative.any():
                    row["accuracy"] = float(
                        np.mean(np.sign(truth[informative]) == np.sign(estimate[informative]))
                    )
                    row["status"] = "supported"
            units.append(row)
    # Conditions within guide, guides within target, targets within donor, donors.
    by_guide: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in units:
        if row["accuracy"] is not None:
            by_guide[(row["donor"], row["target"], row["guide"])].append(row["accuracy"])
    by_target: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (d, t, _), values in by_guide.items():
        by_target[(d, t)].append(float(np.mean(values)))
    by_donor: dict[int, list[float]] = defaultdict(list)
    for (d, _), values in by_target.items():
        by_donor[d].append(float(np.mean(values)))
    accuracy = float(np.mean([np.mean(x) for x in by_donor.values()])) if by_donor else None
    return GeneSignReport(
        accuracy,
        tuple(units),
        sum(r["informative_genes"] for r in units),
        len(units) * observed.shape[1],
        sum(r["accuracy"] is not None for r in units),
        len(units),
        minimum_abs_log_effect,
        log_pseudocount,
    )


def sister_guide_reports(
    *,
    counts: Array,
    donor: Array,
    condition: Array,
    guide: Array,
    guide_to_target: Array,
    control_guides: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    observed = _composition(counts)
    _validate_keys(len(observed), donor, condition, guide, guide_to_target, control_guides)
    controls = np.isin(guide, control_guides)
    effects: dict[int, dict[tuple[int, int, int], float]] = defaultdict(dict)
    support: dict[tuple[int, int, int], tuple[int, int]] = {}
    for d, c in sorted(set(zip(donor.tolist(), condition.tolist(), strict=True))):
        selected = (donor == d) & (condition == c)
        ntc = selected & controls
        if not ntc.any():
            continue
        reference = np.log(observed[ntc].mean(axis=0) + 1e-8)
        for g in sorted(set(guide[selected & ~controls].tolist())):
            rows = selected & (guide == g)
            delta = np.log(observed[rows].mean(axis=0) + 1e-8) - reference
            effects[g].update({(int(d), int(c), j): float(v) for j, v in enumerate(delta)})
            support[(g, int(d), int(c))] = (int(rows.sum()), int(ntc.sum()))
    reports = []
    present_guides = set(guide[~controls].tolist())
    for target in sorted({int(guide_to_target[g]) for g in present_guides}):
        guides = sorted(g for g in present_guides if guide_to_target[g] == target)
        pairs = []
        for left, right in combinations(guides, 2):
            keys = sorted(set(effects[left]) & set(effects[right]))
            a = np.asarray([effects[left][key] for key in keys])
            b = np.asarray([effects[right][key] for key in keys])
            supported = len(keys) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0
            correlation = float(np.corrcoef(a, b)[0, 1]) if supported else None
            shared = sorted({(d, c) for d, c, _ in keys})
            pairs.append(
                {
                    "guides": (int(left), int(right)),
                    "shared_donor_conditions": shared,
                    "compared_gene_keys": len(keys),
                    "correlation": correlation,
                    "status": "supported"
                    if correlation is not None
                    else "insufficient_shared_support",
                    "support": [
                        {
                            "donor": d,
                            "condition": c,
                            "left_cells": support[(left, d, c)][0],
                            "right_cells": support[(right, d, c)][0],
                            "control_cells": support[(left, d, c)][1],
                        }
                        for d, c in shared
                    ],
                    "within_variance": float(np.mean((a - b) ** 2) / 4) if keys else None,
                }
            )
        reports.append({"target_index": target, "guide_indices": tuple(guides), "pairs": pairs})
    return tuple(reports)
