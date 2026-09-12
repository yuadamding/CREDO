"""Deterministic hierarchical sampler replay and derived support for Dev36."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    G00CD1ExecutionAuthorityFreezeV2,
    G00CRefitSeedScheduleV1,
    G00CSamplerEvidenceV3,
    G00CSamplerPlanEntryV3,
    G00CSamplerPlanV3,
    G00CSupportAuditContractV2,
    G00CSupportAuditReceiptV3,
)
from ..errors import IntegrityError
from .g00c_v3 import _path, _read_model

TRACE_COLUMNS = (
    "entry_index",
    "draw_index",
    "macro_update",
    "microbatch",
    "microbatch_offset",
    "row_id",
    "source_index",
    "target_code",
    "guide_code",
    "is_control",
    "inverse_probability_weight",
    "thinning_draw",
)
STATE_COLUMNS = (
    "entry_index",
    "macro_cursor",
    "sampler_state_sha256",
    "thinning_state_sha256",
)
SUPPORT_COLUMNS = (
    "candidate_kind",
    "candidate_value",
    "dimension",
    "stratum_id",
    "cells",
    "minimum_cells_per_substratum",
    "maximum_cells_per_substratum",
    "weighted_effective_sample_size",
    "maximum_to_median_weight_ratio",
    "zero_support",
    "selection_eligible",
)


def _state_hash(generator: np.random.Generator) -> str:
    payload = json.dumps(generator.bit_generator.state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV2, role: str) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev36 authority has no unique {role} implementation.")
    return matches[0]


def _entry_rows(
    entry: G00CSamplerPlanEntryV3,
    order: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    count = 1_000_000 if entry.candidate_kind == "feature_count" else entry.candidate_value
    if count > len(order):
        raise IntegrityError("Dev36 sampler candidate exceeds the frozen nested row order.")
    return order[:count]


@dataclass(frozen=True)
class _PreparedHierarchy:
    """Immutable lookup tables shared by every draw over one row prefix."""

    sources: np.ndarray[Any, Any]
    targets: dict[int, np.ndarray[Any, Any]]
    guides: dict[tuple[int, int], np.ndarray[Any, Any]]
    rows: dict[tuple[int, int, int], np.ndarray[Any, Any]]
    control_row_ids: np.ndarray[Any, Any]
    control_values: np.ndarray[Any, Any]


def _prepare_hierarchy(
    hierarchy: pd.DataFrame,
    allowed_rows: np.ndarray[Any, Any],
) -> _PreparedHierarchy:
    """Index one frozen prefix once instead of rescanning it for every seed."""

    selected = hierarchy.loc[hierarchy["row_id"].isin(allowed_rows)].copy()
    if len(selected) != len(allowed_rows) or selected["row_id"].duplicated().any():
        raise IntegrityError("Dev36 sampler rows are not one exact frozen prefix.")
    source_groups = {
        int(source): frame for source, frame in selected.groupby("source_index", sort=True)
    }
    if not source_groups:
        raise IntegrityError("Dev36 sampler has no training-fit source strata.")

    targets: dict[int, np.ndarray[Any, Any]] = {}
    guides: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    rows: dict[tuple[int, int, int], np.ndarray[Any, Any]] = {}
    for source, source_frame in source_groups.items():
        target_groups = {
            int(target): frame for target, frame in source_frame.groupby("target_code", sort=True)
        }
        targets[source] = np.asarray(sorted(target_groups), dtype=np.int64)
        for target, target_frame in target_groups.items():
            guide_groups = {
                int(guide): frame for guide, frame in target_frame.groupby("guide_code", sort=True)
            }
            guides[(source, target)] = np.asarray(sorted(guide_groups), dtype=np.int64)
            for guide, guide_frame in guide_groups.items():
                rows[(source, target, guide)] = guide_frame["row_id"].to_numpy(dtype=np.int64)

    control = selected[["row_id", "is_control"]].sort_values("row_id", kind="stable")
    return _PreparedHierarchy(
        sources=np.asarray(sorted(source_groups), dtype=np.int64),
        targets=targets,
        guides=guides,
        rows=rows,
        control_row_ids=control["row_id"].to_numpy(dtype=np.int64),
        control_values=control["is_control"].to_numpy(dtype=np.bool_),
    )


def _draw_prepared_entry(
    entry_index: int,
    entry: G00CSamplerPlanEntryV3,
    prepared: _PreparedHierarchy,
    schedule: G00CRefitSeedScheduleV1,
    *,
    resume_after_macro_update: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = schedule.records[entry.refit_draw_id]
    sampler = np.random.Generator(np.random.PCG64DXSM(seed.training_sampler))
    thinning = np.random.Generator(np.random.PCG64DXSM(seed.thinning))
    sources = prepared.sources
    states: list[dict[str, object]] = []
    total = entry.macro_updates * 4096
    row_ids = np.empty(total, dtype=np.int64)
    source_indices = np.empty(total, dtype=np.int64)
    target_codes = np.empty(total, dtype=np.int64)
    guide_codes = np.empty(total, dtype=np.int64)
    weights = np.empty(total, dtype=np.float64)
    thinning_draws = np.empty(total, dtype=np.uint64)
    for draw_index in range(total):
        source = int(sources[sampler.integers(len(sources))])
        targets = prepared.targets[source]
        target = int(targets[sampler.integers(len(targets))])
        guides = prepared.guides[(source, target)]
        guide = int(guides[sampler.integers(len(guides))])
        candidates = prepared.rows[(source, target, guide)]
        row_id = int(candidates[sampler.integers(len(candidates))])
        probability = 1.0 / (len(sources) * len(targets) * len(guides) * len(candidates))
        row_ids[draw_index] = row_id
        source_indices[draw_index] = source
        target_codes[draw_index] = target
        guide_codes[draw_index] = guide
        weights[draw_index] = 1.0 / probability
        thinning_draws[draw_index] = thinning.bit_generator.random_raw()
        cursor = draw_index + 1
        if cursor % 4096 == 0:
            states.append(
                {
                    "entry_index": entry_index,
                    "macro_cursor": cursor // 4096,
                    "sampler_state_sha256": _state_hash(sampler),
                    "thinning_state_sha256": _state_hash(thinning),
                }
            )
            if resume_after_macro_update == cursor // 4096:
                sampler_state = json.loads(json.dumps(sampler.bit_generator.state))
                thinning_state = json.loads(json.dumps(thinning.bit_generator.state))
                sampler = np.random.Generator(np.random.PCG64DXSM())
                thinning = np.random.Generator(np.random.PCG64DXSM())
                sampler.bit_generator.state = sampler_state
                thinning.bit_generator.state = thinning_state
    control_positions = np.searchsorted(prepared.control_row_ids, row_ids)
    if not np.array_equal(prepared.control_row_ids[control_positions], row_ids):
        raise IntegrityError("Dev36 sampled row is absent from its prepared hierarchy.")
    draw_indices = np.arange(total, dtype=np.int64)
    return (
        pd.DataFrame(
            {
                "entry_index": np.full(total, entry_index, dtype=np.int64),
                "draw_index": draw_indices,
                "macro_update": draw_indices // 4096,
                "microbatch": (draw_indices % 4096) // 512,
                "microbatch_offset": draw_indices % 512,
                "row_id": row_ids,
                "source_index": source_indices,
                "target_code": target_codes,
                "guide_code": guide_codes,
                "is_control": prepared.control_values[control_positions],
                "inverse_probability_weight": weights,
                "thinning_draw": thinning_draws,
            },
            columns=TRACE_COLUMNS,
        ),
        pd.DataFrame(states, columns=STATE_COLUMNS),
    )


def _draw_entry(
    entry_index: int,
    entry: G00CSamplerPlanEntryV3,
    hierarchy: pd.DataFrame,
    allowed_rows: np.ndarray[Any, Any],
    schedule: G00CRefitSeedScheduleV1,
    *,
    resume_after_macro_update: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper for one exact entry."""

    return _draw_prepared_entry(
        entry_index,
        entry,
        _prepare_hierarchy(hierarchy, allowed_rows),
        schedule,
        resume_after_macro_update=resume_after_macro_update,
    )


def replay_sampler_plan_v3(
    plan: G00CSamplerPlanV3,
    hierarchy: pd.DataFrame,
    order: np.ndarray[Any, Any],
    schedule: G00CRefitSeedScheduleV1,
    *,
    resumed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize the exact trace from frozen metadata and seed streams."""

    traces: list[pd.DataFrame] = []
    states: list[pd.DataFrame] = []
    prepared_prefixes: dict[int, _PreparedHierarchy] = {}
    for index, entry in enumerate(plan.entries):
        allowed_rows = _entry_rows(entry, order)
        prepared = prepared_prefixes.get(len(allowed_rows))
        if prepared is None:
            prepared = _prepare_hierarchy(hierarchy, allowed_rows)
            prepared_prefixes[len(allowed_rows)] = prepared
        trace, state = _draw_prepared_entry(
            index,
            entry,
            prepared,
            schedule,
            resume_after_macro_update=(plan.resume_after_macro_update if resumed else None),
        )
        traces.append(trace)
        states.append(state)
    return pd.concat(traces, ignore_index=True), pd.concat(states, ignore_index=True)


def _read_hierarchy(root: Path, authority: G00CD1ExecutionAuthorityFreezeV2) -> pd.DataFrame:
    hierarchy = pd.read_parquet(_path(root, authority.sampler_row_hierarchy))
    expected = ("row_id", "source_index", "target_code", "guide_code", "is_control")
    if (
        tuple(hierarchy.columns) != expected
        or len(hierarchy) != authority.sampler_hierarchy_rows
        or hierarchy["row_id"].duplicated().any()
    ):
        raise IntegrityError("Dev36 sampler hierarchy is incomplete or malformed.")
    return hierarchy


def verify_g00c_sampler_v3(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV2,
    evidence: G00CSamplerEvidenceV3,
    *,
    expected_candidate_values: dict[str, tuple[int, ...]] | None = None,
) -> tuple[G00CSamplerPlanV3, pd.DataFrame, np.ndarray[Any, Any], pd.DataFrame]:
    """Independently regenerate every draw, weight, thinning value, RNG state, and cursor."""

    if (
        evidence.execution_authority_id != authority.authority_id
        or evidence.row_hierarchy != authority.sampler_row_hierarchy
        or evidence.nested_training_row_order != authority.nested_training_row_order
        or evidence.implementation_sha256 != _implementation_hash(authority, "sampler")
    ):
        raise IntegrityError("Dev36 sampler evidence is cross-wired to another authority.")
    plan = _read_model(root, evidence.sampler_plan, G00CSamplerPlanV3)
    if (
        plan.plan_id != evidence.sampler_plan_id
        or plan.execution_authority_id != authority.authority_id
    ):
        raise IntegrityError("Dev36 sampler plan identity differs from its evidence.")
    if expected_candidate_values is not None:
        expected_keys = {
            (kind, value, draw)
            for kind, values in expected_candidate_values.items()
            for value in values
            for draw in range(59)
        }
        observed_keys = {
            (entry.candidate_kind, entry.candidate_value, entry.refit_draw_id)
            for entry in plan.entries
        }
        if observed_keys != expected_keys:
            raise IntegrityError("Dev36 sampler plan does not cover the exact frozen grids.")
    schedule = _read_model(root, authority.seed_schedule, G00CRefitSeedScheduleV1)
    if plan.seed_schedule_id != schedule.schedule_id:
        raise IntegrityError("Dev36 sampler plan uses another six-stream schedule.")
    hierarchy = _read_hierarchy(root, authority)
    order_frame = pd.read_parquet(_path(root, authority.nested_training_row_order))
    if tuple(order_frame.columns) != ("rank", "row_id"):
        raise IntegrityError("Dev36 sampler nested order has an invalid schema.")
    order = order_frame["row_id"].to_numpy(dtype=np.int64)
    uninterrupted, uninterrupted_states = replay_sampler_plan_v3(
        plan, hierarchy, order, schedule, resumed=False
    )
    resumed, resumed_states = replay_sampler_plan_v3(plan, hierarchy, order, schedule, resumed=True)
    observed_uninterrupted = pd.read_parquet(_path(root, evidence.uninterrupted_draw_trace))
    observed_resumed = pd.read_parquet(_path(root, evidence.resumed_draw_trace))
    observed_uninterrupted_states = pd.read_parquet(_path(root, evidence.uninterrupted_state_trace))
    observed_resumed_states = pd.read_parquet(_path(root, evidence.resumed_state_trace))
    for observed, expected, label in (
        (observed_uninterrupted, uninterrupted, "uninterrupted draw"),
        (observed_resumed, resumed, "resumed draw"),
        (observed_uninterrupted_states, uninterrupted_states, "uninterrupted state"),
        (observed_resumed_states, resumed_states, "resumed state"),
    ):
        if tuple(observed.columns) != tuple(expected.columns) or not observed.equals(expected):
            raise IntegrityError(f"Dev36 {label} evidence differs from executable replay.")
    if not uninterrupted.equals(resumed) or not uninterrupted_states.equals(resumed_states):
        raise IntegrityError(
            "Dev36 interrupted sampler replay differs from uninterrupted execution."
        )
    return plan, hierarchy, order, uninterrupted


def derive_support_table_v3(
    contract: G00CSupportAuditContractV2,
    plan: G00CSamplerPlanV3,
    hierarchy: pd.DataFrame,
    order: np.ndarray[Any, Any],
    trace: pd.DataFrame,
) -> pd.DataFrame:
    """Derive support from verified sampled rows, weights, and frozen hierarchy."""

    candidate_keys = [
        *(("feature_count", value) for value in contract.feature_candidate_counts),
        *(("training_cells", value) for value in contract.cell_candidate_counts),
    ]
    rows: list[dict[str, object]] = []
    for kind, value in candidate_keys:
        matching = [
            (index, item)
            for index, item in enumerate(plan.entries)
            if item.candidate_kind == kind and item.candidate_value == value
        ]
        if not matching:
            raise IntegrityError("Dev36 sampler plan omits a support-audit candidate.")
        entry_indices = [index for index, _ in matching]
        entry = matching[0][1]
        allowed = _entry_rows(entry, order)
        complete = hierarchy.loc[hierarchy["row_id"].isin(allowed)].copy()
        selected = trace.loc[trace["entry_index"].isin(entry_indices)].copy()
        if selected.empty:
            raise IntegrityError("Dev36 verified sampler trace has no support observations.")
        selected["donor_checkpoint"] = selected["source_index"].astype(str)
        selected["target"] = selected["target_code"].astype(str)
        selected["guide"] = selected["guide_code"].astype(str)
        selected["control_vs_targeting"] = np.where(selected["is_control"], "control", "targeting")
        selected["sampler_stratum"] = (
            selected["source_index"].astype(str)
            + ":"
            + selected["target_code"].astype(str)
            + ":"
            + selected["guide_code"].astype(str)
        )
        complete["donor_checkpoint"] = complete["source_index"].astype(str)
        complete["target"] = complete["target_code"].astype(str)
        complete["guide"] = complete["guide_code"].astype(str)
        complete["control_vs_targeting"] = np.where(complete["is_control"], "control", "targeting")
        complete["sampler_stratum"] = (
            complete["source_index"].astype(str)
            + ":"
            + complete["target_code"].astype(str)
            + ":"
            + complete["guide_code"].astype(str)
        )
        for dimension in contract.dimensions:
            expected_strata = complete[dimension].drop_duplicates().sort_values()
            counts = (
                selected.groupby(dimension, sort=True).size().reindex(expected_strata, fill_value=0)
            )
            for stratum, cells in counts.items():
                cells = int(cells)
                zero = cells == 0
                stratum_weights = selected.loc[
                    selected[dimension].astype(str) == str(stratum),
                    "inverse_probability_weight",
                ].to_numpy(dtype=np.float64)
                if zero:
                    ess = 0.0
                    ratio = 0.0
                else:
                    ess = float(stratum_weights.sum() ** 2 / np.square(stratum_weights).sum())
                    ratio = float(stratum_weights.max() / np.median(stratum_weights))
                rows.append(
                    {
                        "candidate_kind": kind,
                        "candidate_value": value,
                        "dimension": dimension,
                        "stratum_id": str(stratum),
                        "cells": cells,
                        "minimum_cells_per_substratum": int(counts.min()),
                        "maximum_cells_per_substratum": int(counts.max()),
                        "weighted_effective_sample_size": ess,
                        "maximum_to_median_weight_ratio": ratio,
                        "zero_support": zero,
                        "selection_eligible": not zero,
                    }
                )
    return pd.DataFrame(rows, columns=SUPPORT_COLUMNS)


def verify_g00c_support_v3(
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV2,
    sampler: G00CSamplerEvidenceV3,
    receipt: G00CSupportAuditReceiptV3,
    *,
    plan: G00CSamplerPlanV3,
    hierarchy: pd.DataFrame,
    order: np.ndarray[Any, Any],
    trace: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute support; never trust eligibility columns supplied by an execution."""

    contract = _read_model(root, receipt.support_contract, G00CSupportAuditContractV2)
    if (
        receipt.execution_authority_id != authority.authority_id
        or receipt.support_contract_id != contract.contract_id
        or receipt.sampler_evidence_id != sampler.evidence_id
        or receipt.implementation_sha256 != _implementation_hash(authority, "support_auditor")
    ):
        raise IntegrityError("Dev36 support receipt is cross-wired.")
    derived = derive_support_table_v3(contract, plan, hierarchy, order, trace)
    observed = pd.read_parquet(_path(root, receipt.derived_support_table))
    if tuple(observed.columns) != SUPPORT_COLUMNS or not observed.equals(derived):
        raise IntegrityError("Dev36 support table differs from independently derived strata.")
    return derived
