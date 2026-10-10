"""Prospective D2-only, score-averaged balanced DDR architecture selection.

This module reads already computed scores. It cannot load cells, train models,
construct an outcome-dependent panel, or use the reserved D1 assessment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "credo.all4_full_ddr.balanced_selection.v1"
STATE_REFERENCE = "source_population_persistence"
COUNT_REFERENCE = "regularized_persistence_alpha0p5"
EXPECTED_CANDIDATES = ("ddr_mlp128_e8", "ddr_res256_e8", "ddr_res256_e16")
IDENTITIES = (
    "support_id",
    "catalog_id",
    "state_axis_id",
    "truth_id",
    "prefix_qualification_id",
    "prefix",
)


class SelectionError(ValueError):
    """The complete registered comparison cannot be selected from these inputs."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{name} must be a finite numeric score.")
    value = float(value)
    if not math.isfinite(value):
        raise SelectionError(f"{name} must be finite.")
    return value


def _nonnegative(
    value: Any, name: str, tolerance: float, corrections: list[dict[str, Any]]
) -> float:
    result = _number(value, name)
    if result < -tolerance:
        raise SelectionError(
            f"{name} is negative beyond numerical tolerance: {result}."
        )
    if result < 0:
        corrections.append({"score": name, "original": result, "canonical": 0.0})
        return 0.0
    return result


def _validate_contract(
    contract: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    if contract.get("schema") != SCHEMA or contract.get("selection_donor") != "D2":
        raise SelectionError("Only the registered D2 selection contract is supported.")
    if contract.get("seeds") != [0, 1, 2]:
        raise SelectionError("Every registered seed 0, 1, 2 is required.")
    if contract.get("state_estimator") != "weighted_biased_v_energy":
        raise SelectionError(
            "State scores must use the nonnegative weighted V-energy estimator."
        )
    if contract.get("count_score") != "full_catalog_categorical_ce_nats":
        raise SelectionError(
            "Count scores must be complete-catalog categorical CE in nats."
        )
    if contract.get("catalog_size") != 26504:
        raise SelectionError("The complete 26,504-category catalog is required.")
    if contract.get("prefix") != 4096:
        raise SelectionError(
            "Architecture selection uses only the primary 4,096-particle prefix."
        )
    for name in IDENTITIES[:-1]:
        value = contract.get(name)
        if not isinstance(value, str) or not value or value.startswith("UNBOUND"):
            raise SelectionError(f"The exact {name} must be bound before selection.")
    if contract.get("nonnegative_atol") != 1e-10 or contract.get("tie_atol") != 1e-12:
        raise SelectionError(
            "The prospective numerical tolerances cannot be changed here."
        )
    support: dict[str, str] = {}
    for item in contract.get("state_support", []):
        record, target = item.get("record_id"), item.get("target_id")
        if (
            not isinstance(record, str)
            or not record
            or not isinstance(target, str)
            or not target
        ):
            raise SelectionError(
                "State support requires nonempty record and target IDs."
            )
        if record in support:
            raise SelectionError("State support records must be unique.")
        support[record] = target
    if not support:
        raise SelectionError(
            "A nonempty prospectively fixed state support is required."
        )
    candidates = contract.get("candidates", [])
    if [item.get("id") for item in candidates] != list(EXPECTED_CANDIDATES):
        raise SelectionError(
            "Candidate inventory differs from the three registered architectures."
        )
    entries = {item["id"]: item for item in candidates}
    parameter_counts = []
    for rank, candidate in enumerate(EXPECTED_CANDIDATES):
        item = entries[candidate]
        if item.get("complexity_rank") != rank:
            raise SelectionError(
                "Complexity tie-breaking ranks must remain prospectively fixed."
            )
        count = item.get("parameter_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SelectionError(
                "Actual qualified candidate parameter counts must be bound."
            )
        parameter_counts.append(count)
    if parameter_counts != sorted(parameter_counts):
        raise SelectionError(
            "Bound parameter counts contradict the registered complexity ranks."
        )
    entries[STATE_REFERENCE] = contract.get("state_reference", {})
    entries[COUNT_REFERENCE] = contract.get("count_reference", {})
    for candidate, item in entries.items():
        identity = item.get("qualification_id")
        if (
            not isinstance(identity, str)
            or not identity
            or identity.startswith("UNBOUND")
        ):
            raise SelectionError(
                f"A qualified procedure identity is required for {candidate}."
            )
        if item.get("qualified") is not True:
            raise SelectionError(
                f"Procedure {candidate} has not qualified; selection remains incomplete."
            )
    return support, entries


def select_architecture(
    contract: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    count_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select one architecture only from a complete, identity-bound D2 score bank.

    Reductions: equal records within each target, equal targets, then equal seed
    scores. Counts use one full-catalog CE per seed. Entropy is subtracted before
    normalization, so unavoidable catalog entropy cannot dominate the state term.
    A missing or unqualified candidate is an incomplete comparison, never silently
    dropped. Tiny negative roundoff is recorded; it is not a preservation margin.
    """
    support, procedures = _validate_contract(contract)
    tolerance = contract["nonnegative_atol"]
    corrections: list[dict[str, Any]] = []
    seeds = contract["seeds"]
    state_procedures = (*EXPECTED_CANDIDATES, STATE_REFERENCE)
    count_procedures = (*EXPECTED_CANDIDATES, COUNT_REFERENCE)

    def identity(row: Mapping[str, Any], allowed: tuple[str, ...]) -> tuple[str, int]:
        candidate, seed = row.get("procedure_id"), row.get("seed")
        if (
            candidate not in allowed
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed not in seeds
        ):
            raise SelectionError("An undeclared procedure or seed entered selection.")
        if row.get("donor") != "D2":
            raise SelectionError(
                "D1 or other donor outcomes cannot influence D2 selection."
            )
        if row.get("qualified") is not True:
            raise SelectionError("An unqualified score cannot enter selection.")
        if row.get("qualification_id") != procedures[candidate]["qualification_id"]:
            raise SelectionError("Score and qualified procedure identities disagree.")
        for name in IDENTITIES:
            if row.get(name) != contract[name]:
                raise SelectionError(f"Score and frozen {name} disagree.")
        return candidate, seed

    state: dict[tuple[str, int, str], float] = {}
    for row in state_rows:
        candidate, seed = identity(row, state_procedures)
        record = row.get("record_id")
        if record not in support or row.get("target_id") != support[record]:
            raise SelectionError(
                "State score changes the frozen common-support/target mapping."
            )
        key = candidate, seed, record
        if key in state:
            raise SelectionError(f"Duplicate state score {key}.")
        state[key] = _nonnegative(
            row.get("energy"), f"state:{key}", tolerance, corrections
        )
    expected_state = {
        (a, s, r) for a in state_procedures for s in seeds for r in support
    }
    if set(state) != expected_state:
        raise SelectionError(
            "State-score inventory is incomplete; no missing-output winner."
        )

    counts: dict[tuple[str, int], tuple[float, float]] = {}
    entropy: float | None = None
    for row in count_rows:
        key = identity(row, count_procedures)
        if key in counts:
            raise SelectionError(f"Duplicate count score {key}.")
        current_entropy = _number(row.get("endpoint_entropy_nats"), "endpoint entropy")
        if (
            current_entropy < 0
            or current_entropy > math.log(contract["catalog_size"]) + tolerance
        ):
            raise SelectionError(
                "Endpoint entropy is outside the catalog probability bounds."
            )
        if entropy is not None and current_entropy != entropy:
            raise SelectionError(
                "Count rows must use the identical frozen endpoint entropy."
            )
        entropy = current_entropy
        ce = _number(row.get("cross_entropy_nats"), f"CE:{key}")
        if ce < 0:
            raise SelectionError("Categorical cross-entropy must be nonnegative.")
        kl = _nonnegative(ce - entropy, f"CE_minus_H:{key}", tolerance, corrections)
        counts[key] = ce, kl
    if set(counts) != {(a, s) for a in count_procedures for s in seeds}:
        raise SelectionError("Full-catalog count-score inventory is incomplete.")
    # Count persistence is deterministic and must not drift with model seed.
    if len({counts[COUNT_REFERENCE, seed] for seed in seeds}) != 1:
        raise SelectionError(
            "Regularized count persistence cannot vary across model seeds."
        )

    targets = sorted(set(support.values()))
    target_records = {
        target: [r for r, t in support.items() if t == target] for target in targets
    }
    state_seed = {
        (candidate, seed): math.fsum(
            math.fsum(state[candidate, seed, r] for r in target_records[t])
            / len(target_records[t])
            for t in targets
        )
        / len(targets)
        for candidate in state_procedures
        for seed in seeds
    }
    state_mean = {
        candidate: math.fsum(state_seed[candidate, s] for s in seeds) / len(seeds)
        for candidate in state_procedures
    }
    count_mean = {
        candidate: math.fsum(counts[candidate, s][1] for s in seeds) / len(seeds)
        for candidate in count_procedures
    }
    e0, k0 = state_mean[STATE_REFERENCE], count_mean[COUNT_REFERENCE]
    if e0 <= 0 or k0 <= 0:
        raise SelectionError(
            "A persistence normalizer is zero; balanced selection is unresolved."
        )

    candidates = []
    for candidate in EXPECTED_CANDIDATES:
        energy, kl = state_mean[candidate], count_mean[candidate]
        state_ratio, count_ratio = energy / e0, kl / k0
        balanced = 0.5 * (state_ratio + count_ratio)
        if not math.isfinite(balanced):
            raise SelectionError("Normalized balanced score is not finite.")
        candidates.append(
            {
                "id": candidate,
                "complexity_rank": procedures[candidate]["complexity_rank"],
                "parameter_count": procedures[candidate]["parameter_count"],
                "state_energy": energy,
                "count_ce_nats": math.fsum(counts[candidate, s][0] for s in seeds)
                / len(seeds),
                "count_ce_minus_entropy": kl,
                "state_ratio_to_persistence": state_ratio,
                "count_kl_ratio_to_regularized_persistence": count_ratio,
                "balanced_risk": balanced,
                "state_difference_from_persistence": energy - e0,
                "count_ce_difference_from_persistence": kl - k0,
                "state_preserved": energy <= e0,
                "count_preserved": kl <= k0,
                "balanced_developmental_advancement": energy <= e0
                and kl <= k0
                and balanced < 1,
                "seed_scores": [
                    {
                        "seed": seed,
                        "state_energy": state_seed[candidate, seed],
                        "count_ce_nats": counts[candidate, seed][0],
                        "count_ce_minus_entropy": counts[candidate, seed][1],
                    }
                    for seed in seeds
                ],
            }
        )
    minimum = min(row["balanced_risk"] for row in candidates)
    tied = [
        row
        for row in candidates
        if row["balanced_risk"] <= minimum + contract["tie_atol"]
    ]
    winner = min(tied, key=lambda row: (row["complexity_rank"], row["id"]))
    return {
        "schema": SCHEMA,
        "status": "complete_defined_balanced_selection",
        "selection_donor": "D2",
        **{name: contract[name] for name in IDENTITIES},
        "selected_architecture": winner["id"],
        "selected_seed": None,
        "prediction_ensembling": False,
        "selected_balanced_developmental_advancement": winner[
            "balanced_developmental_advancement"
        ],
        "any_candidate_balanced_developmental_advancement": any(
            row["balanced_developmental_advancement"] for row in candidates
        ),
        "independent_validation": False,
        "perturbation_effect_advancement_assessed": False,
        "state_record_count": len(support),
        "target_count": len(targets),
        "reference_state_energy": e0,
        "reference_count_ce_minus_entropy": k0,
        "endpoint_entropy_nats": entropy,
        "candidates": candidates,
        "numerical_roundoff_corrections": corrections,
        "tie_candidates": [row["id"] for row in tied],
    }
