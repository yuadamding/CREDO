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


class GeneSignAccumulator:
    """Evaluator-only sums of cell compositions; no retained cell-by-gene panel."""

    def __init__(
        self,
        *,
        genes: int,
        guide_to_target: Array,
        control_guides: tuple[int, ...],
        maximum_output_bytes: int,
        minimum_abs_log_effect: float = 0.05,
        log_pseudocount: float = 1e-8,
    ) -> None:
        if not np.isfinite(minimum_abs_log_effect) or minimum_abs_log_effect < 0:
            raise ContractError("Informative-effect threshold must be fixed and nonnegative.")
        if not np.isfinite(log_pseudocount) or log_pseudocount <= 0:
            raise ContractError("Log pseudocount must be fixed and positive.")
        if min(genes, maximum_output_bytes) <= 0:
            raise ContractError("Effect output dimensions and budget must be positive.")
        empty = np.empty(0, dtype=np.int64)
        _validate_keys(0, empty, empty, empty, guide_to_target, control_guides)
        self.genes = genes
        self.guide_to_target = guide_to_target
        self.control_guides = control_guides
        self.maximum_output_bytes = maximum_output_bytes
        self.minimum_abs_log_effect = minimum_abs_log_effect
        self.log_pseudocount = log_pseudocount
        self.targets: dict[tuple[int, int, int], tuple[int, Array, Array, Array]] = {}
        self.controls: dict[tuple[int, int], tuple[int, Array]] = {}

    @property
    def output_bytes(self) -> int:
        return (3 * len(self.targets) + len(self.controls)) * self.genes * 8

    def _reserve(self, arrays: int) -> None:
        if self.output_bytes + arrays * self.genes * 8 > self.maximum_output_bytes:
            raise ContractError("Keyed effects exceed their evaluation-output budget.")

    def add_controls(self, *, counts: Array, donor: Array, condition: Array, guide: Array) -> None:
        observed = _composition(counts)
        _validate_keys(
            len(observed), donor, condition, guide, self.guide_to_target, self.control_guides
        )
        if observed.shape[1] != self.genes or not np.isin(guide, self.control_guides).all():
            raise ContractError("Evaluation references must be aligned control observations.")
        for d, c in sorted(set(zip(donor.tolist(), condition.tolist(), strict=True))):
            selected = (donor == d) & (condition == c)
            if (d, c) not in self.controls:
                self._reserve(1)
                self.controls[d, c] = (0, np.zeros(self.genes))
            n, total = self.controls[d, c]
            total += observed[selected].sum(axis=0)
            self.controls[d, c] = (n + int(selected.sum()), total)

    def add_targets(
        self,
        *,
        counts: Array,
        predicted_mean: Array,
        predicted_reference_mean: Array,
        donor: Array,
        condition: Array,
        guide: Array,
    ) -> None:
        observed, predicted, reference = map(
            _composition, (counts, predicted_mean, predicted_reference_mean)
        )
        if (
            observed.shape != predicted.shape
            or observed.shape != reference.shape
            or observed.shape[1] != self.genes
        ):
            raise ContractError("Observed and predicted effect rows/features must align.")
        _validate_keys(
            len(observed), donor, condition, guide, self.guide_to_target, self.control_guides
        )
        if np.isin(guide, self.control_guides).any():
            raise ContractError("Evaluation targets must be perturbation observations.")
        for key in sorted(
            set(zip(donor.tolist(), condition.tolist(), guide.tolist(), strict=True))
        ):
            d, c, g = key
            selected = (donor == d) & (condition == c) & (guide == g)
            if key not in self.targets:
                self._reserve(3)
                self.targets[key] = (
                    0,
                    np.zeros(self.genes),
                    np.zeros(self.genes),
                    np.zeros(self.genes),
                )
            n, obs, pred, ref = self.targets[key]
            obs += observed[selected].sum(axis=0)
            pred += predicted[selected].sum(axis=0)
            ref += reference[selected].sum(axis=0)
            self.targets[key] = (n + int(selected.sum()), obs, pred, ref)

    def report(self) -> GeneSignReport:
        units: list[dict[str, Any]] = []
        for (d, c, g), (n, observed, predicted, reference) in sorted(self.targets.items()):
            control = self.controls.get((d, c))
            row: dict[str, Any] = {
                "donor": int(d),
                "condition": int(c),
                "guide": int(g),
                "target": int(self.guide_to_target[g]),
                "cells": n,
                "control_cells": control[0] if control is not None else 0,
                "genes": self.genes,
                "informative_genes": 0,
                "accuracy": None,
                "status": "missing_donor_condition_controls",
            }
            if control is not None:
                truth = np.log(observed / n + self.log_pseudocount)
                truth -= np.log(control[1] / control[0] + self.log_pseudocount)
                estimate = np.log(predicted / n + self.log_pseudocount)
                estimate -= np.log(reference / n + self.log_pseudocount)
                informative = np.abs(truth) >= self.minimum_abs_log_effect
                row["informative_genes"] = int(informative.sum())
                row["status"] = "no_informative_effects"
                if informative.any():
                    row["accuracy"] = float(
                        np.mean(np.sign(truth[informative]) == np.sign(estimate[informative]))
                    )
                    row["status"] = "supported"
            units.append(row)
        return _summarize_sign_units(
            units, self.genes, self.minimum_abs_log_effect, self.log_pseudocount
        )


def _summarize_sign_units(
    units: list[dict[str, Any]], genes: int, minimum_abs_log_effect: float, log_pseudocount: float
) -> GeneSignReport:
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
        len(units) * genes,
        sum(r["accuracy"] is not None for r in units),
        len(units),
        minimum_abs_log_effect,
        log_pseudocount,
    )


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
    # Preserve the public dense helper's validation, while sharing aggregation.
    observed, predicted, reference = map(
        _composition, (counts, predicted_mean, predicted_reference_mean)
    )
    if observed.shape != predicted.shape or observed.shape != reference.shape:
        raise ContractError("Observed and predicted effect rows/features must align.")
    _validate_keys(len(observed), donor, condition, guide, guide_to_target, control_guides)
    accumulator = GeneSignAccumulator(
        genes=observed.shape[1],
        guide_to_target=guide_to_target,
        control_guides=control_guides,
        maximum_output_bytes=observed.size * 3 * 8,
        minimum_abs_log_effect=minimum_abs_log_effect,
        log_pseudocount=log_pseudocount,
    )
    controls = np.isin(guide, control_guides)
    if controls.any():
        accumulator.add_controls(
            counts=counts[controls],
            donor=donor[controls],
            condition=condition[controls],
            guide=guide[controls],
        )
    if (~controls).any():
        accumulator.add_targets(
            counts=counts[~controls],
            predicted_mean=predicted_mean[~controls],
            predicted_reference_mean=predicted_reference_mean[~controls],
            donor=donor[~controls],
            condition=condition[~controls],
            guide=guide[~controls],
        )
    return accumulator.report()


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
