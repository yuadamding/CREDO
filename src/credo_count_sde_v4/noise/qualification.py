"""T02A raw-count split-half and relative-mass sampling noise floors.

This module deliberately has no model, representation, optimizer, or external
biological-annotation dependency.  It consumes the passed T00 pooled population
and its observed endpoint CSR counts, then freezes conditional measurement-noise
tolerances before learned components are inspected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import rel_entr, xlogy
from scipy.stats import rankdata, spearmanr

from ..canonical import canonical_json_bytes, contract_id, path_manifest, sha256_bytes, sha256_file
from ..contracts import (
    ComponentTestContract,
    ComponentTestReceipt,
    ComponentTestReceiptV2,
    RawCountMassNoiseAmendment,
    RawCountMassNoiseAmendmentReceipt,
    RawCountMassNoiseBundle,
    RawCountMassNoiseReceipt,
)
from ..data import verify_pooled_finite_measures
from ..errors import ContractError, IntegrityError
from ..persistence import artifact_ref, publish_directory, verify_directory
from ..runtime_identity import environment_identity, environment_lock_hash
from ..store import CountStore

_DEFAULT_REPEATS = 100
_DEFAULT_VARIABLE_GENES = 2_000
_DEFAULT_TOP_GENES = 50
_DEFAULT_RANK_TOP_K = 20
_RAW_METRICS = (
    "hellinger",
    "jensen_shannon",
    "multinomial_deviance_per_count",
    "pseudobulk_gene_spearman",
    "top_variable_gene_overlap",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _implementation_identity() -> tuple[str, dict[str, str]]:
    package = Path(__file__).resolve().parents[1]
    relative_paths = (
        "contracts/models.py",
        "data/pooling.py",
        "noise/qualification.py",
        "runtime_identity.py",
        "store/csr.py",
    )
    files = {relative: sha256_file(package / relative) for relative in relative_paths}
    return sha256_bytes(canonical_json_bytes(files)), files


def _quantile(values: pd.Series | np.ndarray[Any, Any], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ContractError("T02A quantiles require a nonempty finite sample.")
    return float(np.quantile(array, probability, method="linear"))


def _rowwise_correlation(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
    return np.clip(result, -1.0, 1.0)


def _top_overlap(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any], *, count: int
) -> np.ndarray[Any, Any]:
    left_top = np.argpartition(left, -count, axis=1)[:, -count:]
    right_top = np.argpartition(right, -count, axis=1)[:, -count:]
    overlaps = np.empty(len(left_top), dtype=np.float64)
    for index, (left_row, right_row) in enumerate(zip(left_top, right_top, strict=True)):
        overlaps[index] = len(set(left_row.tolist()) & set(right_row.tolist())) / count
    return overlaps


def _target_balanced_repeat_summary(frame: pd.DataFrame) -> pd.DataFrame:
    targeting = frame[~frame.is_control].copy()
    controls = frame[frame.is_control].copy()
    target_means = targeting.groupby(
        ["repeat", "seed", "checkpoint", "target_id"], observed=True, as_index=False
    )[list(_RAW_METRICS)].mean()
    target_balanced = target_means.groupby(
        ["repeat", "seed", "checkpoint"], observed=True, as_index=False
    )[list(_RAW_METRICS)].mean()
    target_balanced["population"] = "perturbation_target_balanced"
    control_means = controls.groupby(
        ["repeat", "seed", "checkpoint"], observed=True, as_index=False
    )[list(_RAW_METRICS)].mean()
    control_means["population"] = "control_guide_mean"
    return (
        pd.concat([target_balanced, control_means], ignore_index=True)
        .sort_values(["repeat", "checkpoint", "population"], kind="stable")
        .reset_index(drop=True)
    )


def _raw_target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["checkpoint", "target_id", "is_control"], observed=True)[list(_RAW_METRICS)]
        .agg(["mean", "median", lambda values: np.quantile(values, 0.95)])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary.rename(
        columns={column: column.replace("_<lambda_0>", "_q95") for column in summary}
    )


def _raw_split_half_metrics(
    *,
    cells: pd.DataFrame,
    catalog: pd.DataFrame,
    store: CountStore,
    checkpoints: tuple[str, str],
    repeats: int,
    seed_start: int,
    variable_gene_count: int,
    top_gene_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[pd.DataFrame] = []
    variable_rows: list[pd.DataFrame] = []
    ordered_catalog = catalog.sort_values("guide_id", kind="stable").reset_index(drop=True)
    guides = ordered_catalog.guide_id.astype(str).tolist()
    targets = ordered_catalog.target_id.astype(str).to_numpy()
    controls = ordered_catalog.is_control.astype(bool).to_numpy()
    guide_index = {guide: index for index, guide in enumerate(guides)}
    feature_count = store.manifest.features
    selected_count = min(variable_gene_count, feature_count)
    if top_gene_count > selected_count:
        raise ContractError("T02A top-gene count exceeds the available variable-gene universe.")

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        local = cells[cells.checkpoint == checkpoint].copy().reset_index(drop=True)
        if set(local.guide_id) != set(guides):
            raise ContractError(f"T02A checkpoint {checkpoint!r} does not cover every T00 guide.")
        batch = store.rows(local.row_id.to_numpy(dtype=np.int64))
        group_rows = local.guide_id.map(guide_index).to_numpy(dtype=np.int32)
        group_matrix = sparse.csr_matrix(
            (
                np.ones(len(local), dtype=np.int8),
                (group_rows, np.arange(len(local), dtype=np.int64)),
            ),
            shape=(len(guides), len(local)),
        )
        full = (group_matrix @ batch.matrix).toarray().astype(np.float64, copy=False)
        library = full.sum(axis=1, keepdims=True)
        if np.any(library <= 0):
            raise ContractError("Every T02A guide/checkpoint pseudobulk must contain counts.")
        log_cpm = np.log1p(10_000.0 * full / library)
        variance = np.var(log_cpm, axis=0)
        feature_indices = np.arange(feature_count, dtype=np.int64)
        variable = np.lexsort((feature_indices, -variance))[:selected_count]
        variable_rows.append(
            pd.DataFrame(
                {
                    "checkpoint": checkpoint,
                    "variance_rank": np.arange(1, selected_count + 1, dtype=np.int64),
                    "feature_index": variable,
                    "log_cpm_variance": variance[variable],
                }
            )
        )
        background = full.sum(axis=0)
        background_probability = (background + 0.5) / (
            float(background.sum()) + 0.5 * feature_count
        )
        local_groups = {
            guide: np.asarray(indices, dtype=np.int64)
            for guide, indices in local.groupby("guide_id", sort=True).indices.items()
        }

        for repeat in range(repeats):
            seed = seed_start + checkpoint_index * repeats + repeat
            rng = np.random.default_rng(seed)
            half_group = np.empty(len(local), dtype=np.int32)
            for guide in guides:
                positions = local_groups[guide]
                shuffled = rng.permutation(positions)
                cut = len(shuffled) // 2
                index = guide_index[guide]
                half_group[shuffled[:cut]] = 2 * index
                half_group[shuffled[cut:]] = 2 * index + 1
            half_matrix = sparse.csr_matrix(
                (
                    np.ones(len(local), dtype=np.int8),
                    (half_group, np.arange(len(local), dtype=np.int64)),
                ),
                shape=(2 * len(guides), len(local)),
            )
            aggregate = (half_matrix @ batch.matrix).toarray().astype(np.float64, copy=False)
            left = aggregate[0::2]
            right = aggregate[1::2]
            left_total = left.sum(axis=1, keepdims=True)
            right_total = right.sum(axis=1, keepdims=True)
            if np.any(left_total <= 0) or np.any(right_total <= 0):
                raise ContractError("A balanced T02A cell half contains zero total counts.")
            left_probability = left / left_total
            right_probability = right / right_total
            hellinger = np.sqrt(
                0.5
                * np.sum(
                    (np.sqrt(left_probability) - np.sqrt(right_probability)) ** 2,
                    axis=1,
                )
            )
            midpoint = 0.5 * (left_probability + right_probability)
            js = 0.5 * (
                np.sum(rel_entr(left_probability, midpoint), axis=1)
                + np.sum(rel_entr(right_probability, midpoint), axis=1)
            )
            pooled = (left + right) / (left_total + right_total)
            expected_left = left_total * pooled
            expected_right = right_total * pooled
            left_ratio = np.divide(
                left,
                expected_left,
                out=np.ones_like(left),
                where=left > 0,
            )
            right_ratio = np.divide(
                right,
                expected_right,
                out=np.ones_like(right),
                where=right > 0,
            )
            deviance = (
                2.0
                * (
                    np.sum(xlogy(left, left_ratio), axis=1)
                    + np.sum(xlogy(right, right_ratio), axis=1)
                )
                / (left_total[:, 0] + right_total[:, 0])
            )
            ranked = rankdata(aggregate[:, variable], method="average", axis=1)
            gene_spearman = _rowwise_correlation(ranked[0::2], ranked[1::2])
            left_score = np.abs(
                np.log((left[:, variable] + 0.5) / (left_total + 0.5 * feature_count))
                - np.log(background_probability[variable])[None, :]
            )
            right_score = np.abs(
                np.log((right[:, variable] + 0.5) / (right_total + 0.5 * feature_count))
                - np.log(background_probability[variable])[None, :]
            )
            top_overlap = _top_overlap(left_score, right_score, count=top_gene_count)
            half_counts = np.bincount(half_group, minlength=2 * len(guides)).reshape(-1, 2)
            frame = pd.DataFrame(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "guide_id": guides,
                    "target_id": targets,
                    "is_control": controls,
                    "left_cells": half_counts[:, 0],
                    "right_cells": half_counts[:, 1],
                    "left_umis": left_total[:, 0],
                    "right_umis": right_total[:, 0],
                    "hellinger": hellinger,
                    "jensen_shannon": js,
                    "multinomial_deviance_per_count": deviance,
                    "pseudobulk_gene_spearman": gene_spearman,
                    "top_variable_gene_overlap": top_overlap,
                }
            )
            all_rows.append(frame)

    raw = (
        pd.concat(all_rows, ignore_index=True)
        .sort_values(["checkpoint", "repeat", "guide_id"], kind="stable")
        .reset_index(drop=True)
    )
    variables = pd.concat(variable_rows, ignore_index=True)
    repeat_summary = _target_balanced_repeat_summary(raw)
    target_summary = _raw_target_summary(raw)
    return raw, repeat_summary, target_summary, variables


def _rank_correlation(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def _set_overlap(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any], *, k: int, largest: bool
) -> float:
    if largest:
        left_index = np.argpartition(left, -k)[-k:]
        right_index = np.argpartition(right, -k)[-k:]
    else:
        left_index = np.argpartition(left, k - 1)[:k]
        right_index = np.argpartition(right, k - 1)[:k]
    return len(set(left_index.tolist()) & set(right_index.tolist())) / k


def _mass_noise_metrics(
    *,
    measures: pd.DataFrame,
    catalog: pd.DataFrame,
    source_checkpoint: str,
    terminal_checkpoint: str,
    repeats: int,
    seed_start: int,
    pseudocount: float,
    rank_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = catalog.sort_values("guide_id", kind="stable").reset_index(drop=True)
    guides = ordered.guide_id.astype(str).to_numpy()
    targets = ordered.target_id.astype(str).to_numpy()
    controls = ordered.is_control.astype(bool).to_numpy()
    pivot = measures.pivot(index="guide_id", columns="checkpoint", values="cell_count").reindex(
        guides
    )
    counts_source = pivot[source_checkpoint].to_numpy(dtype=np.int64)
    counts_terminal = pivot[terminal_checkpoint].to_numpy(dtype=np.int64)
    if np.any(counts_source <= 0) or np.any(counts_terminal <= 0):
        raise ContractError("T02A mass bootstrap requires nonempty retained measures.")
    count_pair = (counts_source, counts_terminal)
    total_pair = (int(counts_source.sum()), int(counts_terminal.sum()))
    probability_pair = tuple(
        (counts + pseudocount) / (int(counts.sum()) + pseudocount * len(counts))
        for counts in count_pair
    )
    observed = np.log(probability_pair[1]) - np.log(probability_pair[0])
    targeting_indices = np.flatnonzero(~controls)
    targeting_targets = sorted(set(targets[targeting_indices].tolist()))
    target_indices = {
        target: np.flatnonzero((targets == target) & ~controls) for target in targeting_targets
    }
    observed_target = np.asarray(
        [
            math.log(probability_pair[1][target_indices[target]].sum())
            - math.log(probability_pair[0][target_indices[target]].sum())
            for target in targeting_targets
        ],
        dtype=np.float64,
    )
    k = min(rank_top_k, len(targeting_indices) // 2)
    if k < 1:
        raise ContractError("T02A rank stability requires at least two targeting guides.")
    errors = np.empty((repeats, len(guides)), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    target_errors: dict[str, list[float]] = {target: [] for target in targeting_targets}
    for repeat in range(repeats):
        seed = seed_start + repeat
        rng = np.random.default_rng(seed)
        sampled = tuple(
            rng.multinomial(total, probability)
            for total, probability in zip(total_pair, probability_pair, strict=True)
        )
        sampled_probability = tuple(
            (counts + pseudocount) / (int(counts.sum()) + pseudocount * len(counts))
            for counts in sampled
        )
        effect = np.log(sampled_probability[1]) - np.log(sampled_probability[0])
        error = effect - observed
        errors[repeat] = error
        per_target_mse = np.asarray(
            [np.mean(error[target_indices[target]] ** 2) for target in targeting_targets]
        )
        target_effect = np.asarray(
            [
                math.log(sampled_probability[1][target_indices[target]].sum())
                - math.log(sampled_probability[0][target_indices[target]].sum())
                for target in targeting_targets
            ]
        )
        for target, value in zip(targeting_targets, target_effect - observed_target, strict=True):
            target_errors[target].append(float(value))
        rows.append(
            {
                "repeat": repeat,
                "seed": seed,
                "interval_log_mass_rmse_target_balanced": float(np.sqrt(per_target_mse.mean())),
                "expansion_sign_accuracy": float(
                    np.mean(
                        np.sign(effect[targeting_indices]) == np.sign(observed[targeting_indices])
                    )
                ),
                "guide_rank_spearman": _rank_correlation(
                    observed[targeting_indices], effect[targeting_indices]
                ),
                "target_rank_spearman": _rank_correlation(observed_target, target_effect),
                "top_k_overlap": _set_overlap(
                    observed[targeting_indices], effect[targeting_indices], k=k, largest=True
                ),
                "bottom_k_overlap": _set_overlap(
                    observed[targeting_indices], effect[targeting_indices], k=k, largest=False
                ),
            }
        )
    bootstrap = pd.DataFrame(rows)
    guide_rows: list[dict[str, Any]] = []
    for index, guide in enumerate(guides):
        guide_error = errors[:, index]
        guide_rows.append(
            {
                "guide_id": guide,
                "target_id": targets[index],
                "is_control": controls[index],
                "observed_interval_log_mass": observed[index],
                "bootstrap_error_sd": float(np.std(guide_error, ddof=1)),
                "absolute_error_q95": _quantile(np.abs(guide_error), 0.95),
                "sign_stability": float(
                    np.mean(np.sign(observed[index] + guide_error) == np.sign(observed[index]))
                ),
            }
        )
    guide_noise = pd.DataFrame(guide_rows)
    target_rows: list[dict[str, Any]] = []
    for target, observed_value in zip(targeting_targets, observed_target, strict=True):
        values = np.asarray(target_errors[target], dtype=np.float64)
        target_rows.append(
            {
                "target_id": target,
                "guide_count": len(target_indices[target]),
                "observed_interval_log_mass": observed_value,
                "bootstrap_error_sd": float(np.std(values, ddof=1)),
                "absolute_error_q95": _quantile(np.abs(values), 0.95),
                "sign_stability": float(
                    np.mean(np.sign(observed_value + values) == np.sign(observed_value))
                ),
            }
        )
    return bootstrap, guide_noise, pd.DataFrame(target_rows)


def _seed_ranges_disjoint(
    *, split_seed_start: int, split_repeats: int, mass_seed_start: int, mass_repeats: int
) -> bool:
    split = range(split_seed_start, split_seed_start + 2 * split_repeats)
    mass = range(mass_seed_start, mass_seed_start + mass_repeats)
    return split.stop <= mass.start or mass.stop <= split.start


def _raw_invariants(
    raw: pd.DataFrame,
    *,
    guides: int,
    checkpoints: int,
    repeats: int,
    catalog: pd.DataFrame | None = None,
    measures: pd.DataFrame | None = None,
    checkpoint_names: tuple[str, str] | None = None,
    seed_start: int | None = None,
) -> bool:
    expected = guides * checkpoints * repeats
    required_columns = {
        "repeat",
        "seed",
        "checkpoint",
        "guide_id",
        "target_id",
        "is_control",
        "left_cells",
        "right_cells",
        "left_umis",
        "right_umis",
        *_RAW_METRICS,
    }
    if set(raw.columns) != required_columns:
        return False
    finite = np.isfinite(raw.loc[:, _RAW_METRICS].to_numpy(dtype=np.float64)).all()
    bounds = bool(
        raw.hellinger.between(0, 1 + 1e-12).all()
        and raw.jensen_shannon.between(0, math.log(2) + 1e-12).all()
        and (raw.multinomial_deviance_per_count >= 0).all()
        and raw.pseudobulk_gene_spearman.between(-1, 1).all()
        and raw.top_variable_gene_overlap.between(0, 1).all()
    )
    balanced = bool((raw.left_cells - raw.right_cells).abs().le(1).all())
    base = len(raw) == expected and finite and bounds and balanced
    if not base or any(
        value is None for value in (catalog, measures, checkpoint_names, seed_start)
    ):
        return base
    assert catalog is not None
    assert measures is not None
    assert checkpoint_names is not None
    assert seed_start is not None
    if raw.duplicated(["repeat", "checkpoint", "guide_id"]).any():
        return False
    expected_guides = set(catalog.guide_id.astype(str))
    if set(raw.guide_id.astype(str)) != expected_guides:
        return False
    expected_keys = {
        (repeat, checkpoint, guide)
        for checkpoint in checkpoint_names
        for repeat in range(repeats)
        for guide in expected_guides
    }
    observed_keys = set(
        raw[["repeat", "checkpoint", "guide_id"]].itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        return False
    checkpoint_index = {checkpoint: index for index, checkpoint in enumerate(checkpoint_names)}
    expected_seed = raw.apply(
        lambda row: (
            seed_start + checkpoint_index[str(row.checkpoint)] * repeats + int(row["repeat"])
        ),
        axis=1,
    )
    if not np.array_equal(
        raw.seed.to_numpy(dtype=np.int64), expected_seed.to_numpy(dtype=np.int64)
    ):
        return False
    mapping = catalog.set_index("guide_id")[["target_id", "is_control"]]
    mapped = mapping.reindex(raw.guide_id.astype(str))
    if mapped.isna().any().any():
        return False
    if not np.array_equal(raw.target_id.astype(str).to_numpy(), mapped.target_id.astype(str)):
        return False
    if not np.array_equal(raw.is_control.astype(bool).to_numpy(), mapped.is_control.astype(bool)):
        return False
    cell_counts = measures.set_index(["checkpoint", "guide_id"]).cell_count
    expected_cells = np.asarray(
        [
            cell_counts.loc[(checkpoint, guide)]
            for checkpoint, guide in raw[["checkpoint", "guide_id"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=np.int64,
    )
    return bool(
        np.array_equal(
            raw.left_cells.to_numpy(dtype=np.int64) + raw.right_cells.to_numpy(dtype=np.int64),
            expected_cells,
        )
    )


def _mass_invariants(
    bootstrap: pd.DataFrame, *, repeats: int, seed_start: int | None = None
) -> bool:
    required_columns = {
        "repeat",
        "seed",
        "interval_log_mass_rmse_target_balanced",
        "expansion_sign_accuracy",
        "guide_rank_spearman",
        "target_rank_spearman",
        "top_k_overlap",
        "bottom_k_overlap",
    }
    if set(bootstrap.columns) != required_columns:
        return False
    finite = np.isfinite(bootstrap.select_dtypes(include=[np.number]).to_numpy()).all()
    bounded = bool(
        (bootstrap.interval_log_mass_rmse_target_balanced >= 0).all()
        and bootstrap.expansion_sign_accuracy.between(0, 1).all()
        and bootstrap.guide_rank_spearman.between(-1, 1).all()
        and bootstrap.target_rank_spearman.between(-1, 1).all()
        and bootstrap.top_k_overlap.between(0, 1).all()
        and bootstrap.bottom_k_overlap.between(0, 1).all()
    )
    base = (
        len(bootstrap) == repeats
        and finite
        and bounded
        and not bootstrap.repeat.duplicated().any()
        and set(bootstrap.repeat.astype(int)) == set(range(repeats))
    )
    if not base or seed_start is None:
        return base
    ordered = bootstrap.sort_values("repeat", kind="stable")
    return np.array_equal(
        ordered.seed.to_numpy(dtype=np.int64),
        np.arange(seed_start, seed_start + repeats, dtype=np.int64),
    )


def _noise_summary_invariants(
    guide_noise: pd.DataFrame, target_noise: pd.DataFrame, *, catalog: pd.DataFrame
) -> bool:
    guide_columns = {
        "guide_id",
        "target_id",
        "is_control",
        "observed_interval_log_mass",
        "bootstrap_error_sd",
        "absolute_error_q95",
        "sign_stability",
    }
    target_columns = {
        "target_id",
        "guide_count",
        "observed_interval_log_mass",
        "bootstrap_error_sd",
        "absolute_error_q95",
        "sign_stability",
    }
    if set(guide_noise.columns) != guide_columns or set(target_noise.columns) != target_columns:
        return False
    if guide_noise.guide_id.duplicated().any() or target_noise.target_id.duplicated().any():
        return False
    mapping = catalog.set_index("guide_id")[["target_id", "is_control"]]
    observed = guide_noise.set_index("guide_id")
    if set(observed.index.astype(str)) != set(mapping.index.astype(str)):
        return False
    observed = observed.reindex(mapping.index)
    if not np.array_equal(observed.target_id.astype(str), mapping.target_id.astype(str)):
        return False
    if not np.array_equal(observed.is_control.astype(bool), mapping.is_control.astype(bool)):
        return False
    targeting = catalog[~catalog.is_control.astype(bool)]
    expected_counts = targeting.groupby("target_id", observed=True).guide_id.size().sort_index()
    target_indexed = target_noise.set_index("target_id").sort_index()
    if set(target_indexed.index.astype(str)) != set(expected_counts.index.astype(str)):
        return False
    if not np.array_equal(
        target_indexed.guide_count.to_numpy(dtype=np.int64),
        expected_counts.to_numpy(dtype=np.int64),
    ):
        return False
    numeric = pd.concat(
        [
            guide_noise[
                [
                    "observed_interval_log_mass",
                    "bootstrap_error_sd",
                    "absolute_error_q95",
                    "sign_stability",
                ]
            ],
            target_noise[
                [
                    "observed_interval_log_mass",
                    "bootstrap_error_sd",
                    "absolute_error_q95",
                    "sign_stability",
                ]
            ],
        ],
        ignore_index=True,
    )
    return bool(
        np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        and (numeric.bootstrap_error_sd >= 0).all()
        and (numeric.absolute_error_q95 >= 0).all()
        and numeric.sign_stability.between(0, 1).all()
    )


def _legacy_thresholds(
    *, raw: pd.DataFrame, raw_repeat: pd.DataFrame, mass: pd.DataFrame, mass_guide: pd.DataFrame
) -> dict[str, Any]:
    target_repeat = raw_repeat[raw_repeat.population == "perturbation_target_balanced"]
    control_raw = raw[raw.is_control]
    targeting_guide_noise = mass_guide[~mass_guide.is_control]
    target_median_detection = targeting_guide_noise.groupby(
        "target_id", observed=True
    ).absolute_error_q95.median()
    return {
        "schema_version": 1,
        "status": "frozen_before_learned_model_inspection",
        "control_dispersion_hellinger_q95": _quantile(control_raw.hellinger, 0.95),
        "mass_improvement_margin_interval_log_rmse_q95": _quantile(
            mass.interval_log_mass_rmse_target_balanced, 0.95
        ),
        "minimum_detectable_abs_interval_log_mass_effect": _quantile(target_median_detection, 0.95),
        "protected_raw_count_metrics": {
            "target_balanced_hellinger_q95": _quantile(target_repeat.hellinger, 0.95),
            "target_balanced_jensen_shannon_q95": _quantile(target_repeat.jensen_shannon, 0.95),
            "target_balanced_deviance_per_count_q95": _quantile(
                target_repeat.multinomial_deviance_per_count, 0.95
            ),
            "target_balanced_pseudobulk_spearman_q05": _quantile(
                target_repeat.pseudobulk_gene_spearman, 0.05
            ),
            "target_balanced_top_gene_overlap_q05": _quantile(
                target_repeat.top_variable_gene_overlap, 0.05
            ),
        },
        "mass_stability_metrics": {
            "expansion_sign_accuracy_q05": _quantile(mass.expansion_sign_accuracy, 0.05),
            "guide_rank_spearman_q05": _quantile(mass.guide_rank_spearman, 0.05),
            "target_rank_spearman_q05": _quantile(mass.target_rank_spearman, 0.05),
            "top_k_overlap_q05": _quantile(mass.top_k_overlap, 0.05),
            "bottom_k_overlap_q05": _quantile(mass.bottom_k_overlap, 0.05),
        },
        "quantile_method": "numpy_linear",
        "target_balance": "mean_per_perturbation_target_then_mean_targets",
        "controls": "reported_separately_not_one_perturbation_target",
    }


def _component_receipt_invariants(
    component: ComponentTestReceipt | ComponentTestReceiptV2,
    *,
    contract: ComponentTestContract,
    detailed: RawCountMassNoiseReceipt,
    pooled_data_id: str,
    count_store_sha256: str,
    raw_repeat: pd.DataFrame,
) -> bool:
    """Check the generic component surface against the detailed T02A receipt."""

    expected_inputs = {
        "pooled_data": pooled_data_id,
        "count_store": count_store_sha256,
        "environment": detailed.environment_hash,
    }
    common = bool(
        component.test_id == contract.test_id
        and component.status == detailed.status
        and component.primary_metric == contract.primary_metric
        and component.primary_baseline == contract.primary_baseline
        and component.protected_metrics_pass
        == (detailed.raw_invariants_pass and detailed.mass_invariants_pass)
        and component.selected_update == 0
        and component.input_hashes == expected_inputs
        and component.config_hash == detailed.config_hash
        and component.implementation_hash == detailed.implementation_hash
    )
    if not common:
        return False
    if isinstance(component, ComponentTestReceiptV2):
        return bool(
            component.receipt_role == "calibration"
            and component.estimand == "target_balanced_split_half_hellinger"
            and component.quantile_probability == 0.95
            and component.quantile_value == detailed.target_balanced_hellinger_q95
            and component.repeat_count == detailed.split_repeats
            and component.sampling_method == "balanced_cell_split_half_by_checkpoint"
        )
    targeting = raw_repeat[raw_repeat.population == "perturbation_target_balanced"]
    return bool(
        component.point_delta == detailed.target_balanced_hellinger_q95
        and component.bootstrap_interval
        == (
            float(targeting.hellinger.min()),
            detailed.target_balanced_hellinger_q95,
        )
        and component.required_margin == contract.required_margin
        and component.channel_activity == 0.0
    )


def _interpreted_thresholds(
    *,
    raw: pd.DataFrame,
    raw_repeat: pd.DataFrame,
    mass: pd.DataFrame,
    mass_guide: pd.DataFrame,
    mass_target: pd.DataFrame,
) -> dict[str, Any]:
    legacy = _legacy_thresholds(raw=raw, raw_repeat=raw_repeat, mass=mass, mass_guide=mass_guide)
    targeting_guide_noise = mass_guide[~mass_guide.is_control]
    target_medians = targeting_guide_noise.groupby(
        "target_id", observed=True
    ).absolute_error_q95.median()
    return {
        "schema_version": 1,
        "status": "derived_interpretation_of_immutable_t02a",
        "observed_endpoint_sampling_rmse_q95": _quantile(
            mass.interval_log_mass_rmse_target_balanced, 0.95
        ),
        "guide_abs_error_q95_target_median_q95": _quantile(target_medians, 0.95),
        "target_abs_error_q95_across_targets_q95": _quantile(mass_target.absolute_error_q95, 0.95),
        "formal_minimum_detectable_effect_status": "not_estimated",
        "model_comparison_improvement_margin_status": "not_estimated",
        "pooled_raw_count_descriptive": legacy["protected_raw_count_metrics"],
        "pooled_control_hellinger_q95_descriptive": legacy["control_dispersion_hellinger_q95"],
        "mass_stability_metrics": {
            "expansion_sign_accuracy_q05": _quantile(mass.expansion_sign_accuracy, 0.05),
            "guide_rank_spearman_q05": _quantile(mass.guide_rank_spearman, 0.05),
            "target_rank_spearman_q05": _quantile(mass.target_rank_spearman, 0.05),
            "guide_top_k_overlap_q05": _quantile(mass.top_k_overlap, 0.05),
            "guide_bottom_k_overlap_q05": _quantile(mass.bottom_k_overlap, 0.05),
        },
        "quantile_method": "numpy_linear",
        "target_balance": legacy["target_balance"],
        "controls": legacy["controls"],
    }


def _checkpoint_thresholds(
    *, raw: pd.DataFrame, raw_repeat: pd.DataFrame, checkpoints: tuple[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        targeting = raw_repeat[
            (raw_repeat.checkpoint == checkpoint)
            & (raw_repeat.population == "perturbation_target_balanced")
        ]
        controls = raw[(raw.checkpoint == checkpoint) & raw.is_control]
        for population, frame, unit in (
            ("targeting", targeting, "target_balanced_repeat"),
            ("controls", controls, "control_guide_repeat"),
        ):
            if frame.empty:
                raise IntegrityError(
                    f"T02A checkpoint {checkpoint!r} lacks {population} threshold rows."
                )
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "population": population,
                    "sampling_unit": unit,
                    "observations": len(frame),
                    "hellinger_q95": _quantile(frame.hellinger, 0.95),
                    "jensen_shannon_q95": _quantile(frame.jensen_shannon, 0.95),
                    "multinomial_deviance_per_count_q95": _quantile(
                        frame.multinomial_deviance_per_count, 0.95
                    ),
                    "pseudobulk_gene_spearman_q05": _quantile(frame.pseudobulk_gene_spearman, 0.05),
                    "top_variable_gene_overlap_q05": _quantile(
                        frame.top_variable_gene_overlap, 0.05
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["checkpoint", "population"], kind="stable")
        .reset_index(drop=True)
    )


def _target_rank_stability(mass_bootstrap: pd.DataFrame) -> pd.DataFrame:
    return (
        mass_bootstrap[["repeat", "seed", "target_rank_spearman"]]
        .sort_values("repeat", kind="stable")
        .reset_index(drop=True)
    )


def qualify_raw_count_mass_noise(
    destination: Path,
    *,
    pooled_bundle: Path,
    count_store: Path,
    split_repeats: int = _DEFAULT_REPEATS,
    mass_bootstrap_repeats: int = _DEFAULT_REPEATS,
    split_seed_start: int = 20_260_821,
    mass_seed_start: int = 21_260_821,
    variable_gene_count: int = _DEFAULT_VARIABLE_GENES,
    top_gene_count: int = _DEFAULT_TOP_GENES,
    rank_top_k: int = _DEFAULT_RANK_TOP_K,
) -> Path:
    """Run and atomically publish the independent T02A noise calibration."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    if split_repeats < 100 or mass_bootstrap_repeats < 100:
        raise ContractError("T02A requires at least 100 split and mass repeats.")
    if min(split_seed_start, mass_seed_start) < 0 or not _seed_ranges_disjoint(
        split_seed_start=split_seed_start,
        split_repeats=split_repeats,
        mass_seed_start=mass_seed_start,
        mass_repeats=mass_bootstrap_repeats,
    ):
        raise ContractError("T02A seed ranges must be disjoint and nonnegative.")
    if min(variable_gene_count, top_gene_count, rank_top_k) < 1:
        raise ContractError("T02A feature and rank counts must be positive.")
    pooled = verify_pooled_finite_measures(pooled_bundle)
    store = CountStore(count_store)
    store_manifest = store.verify(full=True)
    cells = pd.read_parquet(pooled_bundle / pooled.cells.relative_uri)
    catalog = pd.read_parquet(pooled_bundle / pooled.guide_catalog.relative_uri)
    measures = pd.read_parquet(pooled_bundle / pooled.finite_measures.relative_uri)
    if not set(cells.row_id.astype(int)).issubset(set(store.row_ids().astype(int))):
        raise ContractError("T02A CountStore does not cover every retained T00 cell.")
    checkpoints = (pooled.source_checkpoint, pooled.terminal_checkpoint)
    raw, raw_repeat, raw_target, variable_genes = _raw_split_half_metrics(
        cells=cells,
        catalog=catalog,
        store=store,
        checkpoints=checkpoints,
        repeats=split_repeats,
        seed_start=split_seed_start,
        variable_gene_count=variable_gene_count,
        top_gene_count=top_gene_count,
    )
    mass, mass_guide, mass_target = _mass_noise_metrics(
        measures=measures,
        catalog=catalog,
        source_checkpoint=pooled.source_checkpoint,
        terminal_checkpoint=pooled.terminal_checkpoint,
        repeats=mass_bootstrap_repeats,
        seed_start=mass_seed_start,
        pseudocount=pooled.mass_pseudocount,
        rank_top_k=rank_top_k,
    )
    raw_pass = _raw_invariants(
        raw,
        guides=pooled.retained_guides,
        checkpoints=2,
        repeats=split_repeats,
        catalog=catalog,
        measures=measures,
        checkpoint_names=checkpoints,
        seed_start=split_seed_start,
    )
    mass_pass = _mass_invariants(
        mass, repeats=mass_bootstrap_repeats, seed_start=mass_seed_start
    ) and _noise_summary_invariants(mass_guide, mass_target, catalog=catalog)
    thresholds = _legacy_thresholds(
        raw=raw,
        raw_repeat=raw_repeat,
        mass=mass,
        mass_guide=mass_guide,
    )
    environment = environment_identity()
    environment_hash = environment_lock_hash()
    implementation_hash, implementation_files = _implementation_identity()
    config = {
        "method": "cell_split_half_and_multinomial_catalog_bootstrap_v1",
        "split_repeats": split_repeats,
        "mass_bootstrap_repeats": mass_bootstrap_repeats,
        "split_seed_start": split_seed_start,
        "mass_seed_start": mass_seed_start,
        "variable_gene_count": min(variable_gene_count, store_manifest.features),
        "variable_gene_method": "within_checkpoint_guide_log1p_cpm_variance",
        "spearman_gene_universe": "checkpoint_specific_frozen_variable_genes",
        "top_gene_count": top_gene_count,
        "top_gene_score": "absolute_smoothed_log_enrichment_from_checkpoint_pool",
        "rank_top_k": rank_top_k,
        "mass_pseudocount": pooled.mass_pseudocount,
        "target_balance": thresholds["target_balance"],
        "environment": environment,
    }
    config_hash = sha256_bytes(canonical_json_bytes(config))
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": "T02A_RAW_COUNT_MASS_NOISE",
        "component": "raw_count_and_relative_mass_noise",
        "primary_metric": "target_balanced_split_half_hellinger_q95",
        "primary_baseline": "independent_cell_halves_and_multinomial_catalog_bootstrap",
        "required_margin": 0.0,
        "drift": "off",
        "diffusion": "off",
        "reaction": "off",
        "ecology": "off",
        "decoder": "off",
        "update_zero_selectable": True,
        "post_selection_refit_required": False,
    }
    contract_payload["test_contract_id"] = contract_id(
        contract_payload, id_field="test_contract_id"
    )
    test_contract = ComponentTestContract.model_validate(contract_payload)

    def writer(temp: Path) -> None:
        raw.to_parquet(temp / "RAW_SPLIT_HALF_METRICS.parquet", index=False)
        raw_repeat.to_parquet(temp / "RAW_REPEAT_SUMMARY.parquet", index=False)
        raw_target.to_parquet(temp / "RAW_TARGET_SUMMARY.parquet", index=False)
        variable_genes.to_parquet(temp / "VARIABLE_GENES.parquet", index=False)
        mass.to_parquet(temp / "MASS_BOOTSTRAP_METRICS.parquet", index=False)
        mass_guide.to_parquet(temp / "MASS_GUIDE_NOISE.parquet", index=False)
        mass_target.to_parquet(temp / "MASS_TARGET_NOISE.parquet", index=False)
        _write_json(temp / "FROZEN_THRESHOLDS.json", thresholds)
        _write_json(temp / "CONFIG.json", config)
        _write_json(temp / "TEST_CONTRACT.json", test_contract.model_dump(mode="json"))
        _write_json(
            temp / "INPUTS.sha256",
            {
                "schema_version": 1,
                "pooled_data_id": pooled.pooled_data_id,
                "pooled_bundle_sha256": sha256_file(pooled_bundle / "pooled-data.json"),
                "count_store_sha256": store_manifest.content_sha256,
                "feature_index_hash": store_manifest.feature_index_hash,
                "retained_cell_universe_hash": pooled.retained_cell_universe_hash,
            },
        )
        _write_json(
            temp / "IMPLEMENTATION.sha256",
            {
                "schema_version": 1,
                "implementation_hash": implementation_hash,
                "files": implementation_files,
            },
        )
        _write_json(
            temp / "NULL_CALIBRATION.json",
            {
                "schema_version": 1,
                "status": "not_applicable_measurement_noise_characterization",
            },
        )
        _write_json(
            temp / "BOOTSTRAP_RESULTS.json",
            {
                "schema_version": 1,
                "split_half_repeats": split_repeats,
                "mass_bootstrap_repeats": mass_bootstrap_repeats,
                "thresholds": thresholds,
            },
        )
        _write_json(
            temp / "CHANNEL_ACTIVITY.json",
            {
                "schema_version": 1,
                "drift": 0.0,
                "diffusion": 0.0,
                "reaction": 0.0,
                "ecology": 0.0,
                "decoder": 0.0,
            },
        )
        _write_json(
            temp / "SELECTED_MODEL.json",
            {"schema_version": 1, "selected_model": None, "selected_update": 0},
        )
        detail_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "test_contract_id": test_contract.test_contract_id,
            "status": "pass" if raw_pass and mass_pass else "fail_retired",
            "retained_guides": pooled.retained_guides,
            "perturbation_targets": pooled.perturbation_targets,
            "split_repeats": split_repeats,
            "mass_bootstrap_repeats": mass_bootstrap_repeats,
            "target_balanced_hellinger_q95": thresholds["protected_raw_count_metrics"][
                "target_balanced_hellinger_q95"
            ],
            "control_hellinger_q95": thresholds["control_dispersion_hellinger_q95"],
            "target_balanced_js_q95": thresholds["protected_raw_count_metrics"][
                "target_balanced_jensen_shannon_q95"
            ],
            "target_balanced_deviance_q95": thresholds["protected_raw_count_metrics"][
                "target_balanced_deviance_per_count_q95"
            ],
            "target_balanced_spearman_q05": thresholds["protected_raw_count_metrics"][
                "target_balanced_pseudobulk_spearman_q05"
            ],
            "target_balanced_top_gene_overlap_q05": thresholds["protected_raw_count_metrics"][
                "target_balanced_top_gene_overlap_q05"
            ],
            "interval_log_mass_rmse_q95": thresholds[
                "mass_improvement_margin_interval_log_rmse_q95"
            ],
            "expansion_sign_accuracy_q05": thresholds["mass_stability_metrics"][
                "expansion_sign_accuracy_q05"
            ],
            "guide_rank_spearman_q05": thresholds["mass_stability_metrics"][
                "guide_rank_spearman_q05"
            ],
            "target_rank_spearman_q05": thresholds["mass_stability_metrics"][
                "target_rank_spearman_q05"
            ],
            "top_k_overlap_q05": thresholds["mass_stability_metrics"]["top_k_overlap_q05"],
            "bottom_k_overlap_q05": thresholds["mass_stability_metrics"]["bottom_k_overlap_q05"],
            "raw_invariants_pass": raw_pass,
            "mass_invariants_pass": mass_pass,
            "protected_metrics_frozen": True,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
        }
        detail_payload["receipt_id"] = contract_id(detail_payload, id_field="receipt_id")
        receipt = RawCountMassNoiseReceipt.model_validate(detail_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        generic_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_id": test_contract.test_id,
            "receipt_role": "calibration",
            "status": receipt.status,
            "primary_metric": test_contract.primary_metric,
            "primary_baseline": test_contract.primary_baseline,
            "point_delta": None,
            "bootstrap_interval": None,
            "required_margin": None,
            "channel_activity": None,
            "estimand": "target_balanced_split_half_hellinger",
            "quantile_probability": 0.95,
            "quantile_value": receipt.target_balanced_hellinger_q95,
            "repeat_count": split_repeats,
            "sampling_method": "balanced_cell_split_half_by_checkpoint",
            "protected_metrics_pass": raw_pass and mass_pass,
            "selected_update": 0,
            "input_hashes": {
                "pooled_data": pooled.pooled_data_id,
                "count_store": store_manifest.content_sha256,
                "environment": environment_hash,
            },
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
        }
        generic_payload["receipt_id"] = contract_id(generic_payload, id_field="receipt_id")
        component_receipt = ComponentTestReceiptV2.model_validate(generic_payload)
        _write_json(temp / "COMPONENT_RECEIPT.json", component_receipt.model_dump(mode="json"))
        bundle_payload = {
            "schema_version": 1,
            "noise_id": "pending",
            "test_contract_id": test_contract.test_contract_id,
            "pooled_data_id": pooled.pooled_data_id,
            "count_store_sha256": store_manifest.content_sha256,
            "feature_index_hash": store_manifest.feature_index_hash,
            "environment_hash": environment_hash,
            "source_checkpoint": pooled.source_checkpoint,
            "terminal_checkpoint": pooled.terminal_checkpoint,
            "split_repeats": split_repeats,
            "mass_bootstrap_repeats": mass_bootstrap_repeats,
            "split_seed_start": split_seed_start,
            "mass_seed_start": mass_seed_start,
            "variable_gene_count": min(variable_gene_count, store_manifest.features),
            "top_gene_count": top_gene_count,
            "rank_top_k": rank_top_k,
            "variable_genes": artifact_ref(
                temp,
                temp / "VARIABLE_GENES.parquet",
                schema_id="credo.t02a_variable_genes",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "raw_split_metrics": artifact_ref(
                temp,
                temp / "RAW_SPLIT_HALF_METRICS.parquet",
                schema_id="credo.t02a_raw_split_half",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "raw_repeat_summary": artifact_ref(
                temp,
                temp / "RAW_REPEAT_SUMMARY.parquet",
                schema_id="credo.t02a_raw_repeat_summary",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "raw_target_summary": artifact_ref(
                temp,
                temp / "RAW_TARGET_SUMMARY.parquet",
                schema_id="credo.t02a_raw_target_summary",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "mass_bootstrap": artifact_ref(
                temp,
                temp / "MASS_BOOTSTRAP_METRICS.parquet",
                schema_id="credo.t02a_mass_bootstrap",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "mass_guide_noise": artifact_ref(
                temp,
                temp / "MASS_GUIDE_NOISE.parquet",
                schema_id="credo.t02a_mass_guide_noise",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "mass_target_noise": artifact_ref(
                temp,
                temp / "MASS_TARGET_NOISE.parquet",
                schema_id="credo.t02a_mass_target_noise",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "frozen_thresholds": artifact_ref(
                temp,
                temp / "FROZEN_THRESHOLDS.json",
                schema_id="credo.t02a_frozen_thresholds",
                media_type="application/json",
            ).model_dump(mode="json"),
            "test_receipt": artifact_ref(
                temp,
                temp / "TEST_RECEIPT.json",
                schema_id="credo.t02a_test_receipt",
                media_type="application/json",
            ).model_dump(mode="json"),
        }
        bundle_payload["noise_id"] = contract_id(bundle_payload, id_field="noise_id")
        bundle = RawCountMassNoiseBundle.model_validate(bundle_payload)
        _write_json(temp / "raw-count-mass-noise.json", bundle.model_dump(mode="json"))
        _write_json(
            temp / "QUALIFICATION_LINK.json",
            {
                "schema_version": 1,
                "noise_id": bundle.noise_id,
                "receipt_id": receipt.receipt_id,
            },
        )
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_raw_count_mass_noise(destination, pooled_bundle=pooled_bundle, count_store=count_store)
    return destination


def verify_raw_count_mass_noise(
    path: Path, *, pooled_bundle: Path, count_store: Path
) -> RawCountMassNoiseBundle:
    """Verify the T02A bundle, its immutable parents, and frozen metrics."""

    verify_directory(path)
    bundle = RawCountMassNoiseBundle.model_validate_json(
        (path / "raw-count-mass-noise.json").read_text()
    )
    receipt = RawCountMassNoiseReceipt.model_validate_json((path / "TEST_RECEIPT.json").read_text())
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    component_payload = json.loads((path / "COMPONENT_RECEIPT.json").read_text())
    component: ComponentTestReceipt | ComponentTestReceiptV2
    if component_payload.get("schema_version") == 2:
        component = ComponentTestReceiptV2.model_validate(component_payload)
    else:
        component = ComponentTestReceipt.model_validate(component_payload)
    pooled = verify_pooled_finite_measures(pooled_bundle)
    store = CountStore(count_store).verify(full=True)
    if (
        contract.test_id not in {"T02_RAW_COUNT_MASS_NOISE", "T02A_RAW_COUNT_MASS_NOISE"}
        or bundle.test_contract_id != contract.test_contract_id
        or receipt.test_contract_id != contract.test_contract_id
        or component.test_id != contract.test_id
        or component.status != receipt.status
        or bundle.pooled_data_id != pooled.pooled_data_id
        or bundle.count_store_sha256 != store.content_sha256
        or bundle.feature_index_hash != store.feature_index_hash
    ):
        raise IntegrityError("T02A contract, receipts, bundle, or parents are inconsistent.")
    for reference in (
        bundle.variable_genes,
        bundle.raw_split_metrics,
        bundle.raw_repeat_summary,
        bundle.raw_target_summary,
        bundle.mass_bootstrap,
        bundle.mass_guide_noise,
        bundle.mass_target_noise,
        bundle.frozen_thresholds,
        bundle.test_receipt,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"T02A artifact differs from its reference: {reference.relative_uri}."
            )
    config = json.loads((path / "CONFIG.json").read_text())
    if (
        sha256_bytes(canonical_json_bytes(config)) != receipt.config_hash
        or sha256_bytes(canonical_json_bytes(config.get("environment"))) != receipt.environment_hash
        or receipt.environment_hash != bundle.environment_hash
    ):
        raise IntegrityError("T02A configuration or environment identity differs.")
    implementation = json.loads((path / "IMPLEMENTATION.sha256").read_text())
    if (
        implementation.get("implementation_hash") != receipt.implementation_hash
        or sha256_bytes(canonical_json_bytes(implementation.get("files")))
        != receipt.implementation_hash
    ):
        raise IntegrityError("T02A implementation identity differs from its receipt.")
    raw = pd.read_parquet(path / bundle.raw_split_metrics.relative_uri)
    raw_repeat = pd.read_parquet(path / bundle.raw_repeat_summary.relative_uri)
    raw_target = pd.read_parquet(path / bundle.raw_target_summary.relative_uri)
    variable_genes = pd.read_parquet(path / bundle.variable_genes.relative_uri)
    mass = pd.read_parquet(path / bundle.mass_bootstrap.relative_uri)
    mass_guide = pd.read_parquet(path / bundle.mass_guide_noise.relative_uri)
    mass_target = pd.read_parquet(path / bundle.mass_target_noise.relative_uri)
    catalog = pd.read_parquet(pooled_bundle / pooled.guide_catalog.relative_uri)
    measures = pd.read_parquet(pooled_bundle / pooled.finite_measures.relative_uri)
    checkpoints = (pooled.source_checkpoint, pooled.terminal_checkpoint)
    if (
        bundle.source_checkpoint != pooled.source_checkpoint
        or bundle.terminal_checkpoint != pooled.terminal_checkpoint
        or config.get("split_repeats") != bundle.split_repeats
        or config.get("mass_bootstrap_repeats") != bundle.mass_bootstrap_repeats
        or config.get("split_seed_start") != bundle.split_seed_start
        or config.get("mass_seed_start") != bundle.mass_seed_start
        or config.get("variable_gene_count") != bundle.variable_gene_count
        or config.get("top_gene_count") != bundle.top_gene_count
        or config.get("rank_top_k") != bundle.rank_top_k
        or not _seed_ranges_disjoint(
            split_seed_start=bundle.split_seed_start,
            split_repeats=bundle.split_repeats,
            mass_seed_start=bundle.mass_seed_start,
            mass_repeats=bundle.mass_bootstrap_repeats,
        )
    ):
        raise IntegrityError("T02A bundle configuration differs from its table contract.")
    raw_pass = _raw_invariants(
        raw,
        guides=pooled.retained_guides,
        checkpoints=2,
        repeats=bundle.split_repeats,
        catalog=catalog,
        measures=measures,
        checkpoint_names=checkpoints,
        seed_start=bundle.split_seed_start,
    )
    mass_pass = _mass_invariants(
        mass,
        repeats=bundle.mass_bootstrap_repeats,
        seed_start=bundle.mass_seed_start,
    ) and _noise_summary_invariants(mass_guide, mass_target, catalog=catalog)
    expected_variable_keys = {
        (checkpoint, rank)
        for checkpoint in checkpoints
        for rank in range(1, bundle.variable_gene_count + 1)
    }
    observed_variable_keys = set(
        variable_genes[["checkpoint", "variance_rank"]].itertuples(index=False, name=None)
    )
    variable_pass = bool(
        set(variable_genes.columns)
        == {"checkpoint", "variance_rank", "feature_index", "log_cpm_variance"}
        and observed_variable_keys == expected_variable_keys
        and not variable_genes.duplicated(["checkpoint", "variance_rank"]).any()
        and not variable_genes.duplicated(["checkpoint", "feature_index"]).any()
        and variable_genes.feature_index.between(0, store.features - 1).all()
        and np.isfinite(variable_genes.log_cpm_variance.to_numpy(dtype=np.float64)).all()
    )
    try:
        pd.testing.assert_frame_equal(
            raw_repeat.reset_index(drop=True),
            _target_balanced_repeat_summary(raw),
            check_exact=True,
            check_dtype=True,
        )
        pd.testing.assert_frame_equal(
            raw_target.reset_index(drop=True),
            _raw_target_summary(raw),
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as exc:
        raise IntegrityError(
            "T02A stored raw summaries differ from the guide-level table."
        ) from exc
    thresholds = json.loads((path / bundle.frozen_thresholds.relative_uri).read_text())
    recomputed_thresholds = _legacy_thresholds(
        raw=raw,
        raw_repeat=raw_repeat,
        mass=mass,
        mass_guide=mass_guide,
    )
    if (
        raw_pass != receipt.raw_invariants_pass
        or mass_pass != receipt.mass_invariants_pass
        or not variable_pass
        or receipt.status != ("pass" if raw_pass and mass_pass else "fail_retired")
        or thresholds != recomputed_thresholds
        or thresholds["control_dispersion_hellinger_q95"] != receipt.control_hellinger_q95
        or thresholds["mass_improvement_margin_interval_log_rmse_q95"]
        != receipt.interval_log_mass_rmse_q95
        or thresholds["protected_raw_count_metrics"]["target_balanced_hellinger_q95"]
        != receipt.target_balanced_hellinger_q95
        or thresholds["protected_raw_count_metrics"]["target_balanced_jensen_shannon_q95"]
        != receipt.target_balanced_js_q95
        or thresholds["protected_raw_count_metrics"]["target_balanced_deviance_per_count_q95"]
        != receipt.target_balanced_deviance_q95
        or thresholds["protected_raw_count_metrics"]["target_balanced_pseudobulk_spearman_q05"]
        != receipt.target_balanced_spearman_q05
        or thresholds["protected_raw_count_metrics"]["target_balanced_top_gene_overlap_q05"]
        != receipt.target_balanced_top_gene_overlap_q05
        or thresholds["mass_stability_metrics"]["expansion_sign_accuracy_q05"]
        != receipt.expansion_sign_accuracy_q05
        or thresholds["mass_stability_metrics"]["guide_rank_spearman_q05"]
        != receipt.guide_rank_spearman_q05
        or thresholds["mass_stability_metrics"]["target_rank_spearman_q05"]
        != receipt.target_rank_spearman_q05
        or thresholds["mass_stability_metrics"]["top_k_overlap_q05"] != receipt.top_k_overlap_q05
        or thresholds["mass_stability_metrics"]["bottom_k_overlap_q05"]
        != receipt.bottom_k_overlap_q05
        or not _component_receipt_invariants(
            component,
            contract=contract,
            detailed=receipt,
            pooled_data_id=pooled.pooled_data_id,
            count_store_sha256=store.content_sha256,
            raw_repeat=raw_repeat,
        )
    ):
        raise IntegrityError("T02A stored gates differ from recomputed invariants.")
    link = json.loads((path / "QUALIFICATION_LINK.json").read_text())
    if link != {
        "schema_version": 1,
        "noise_id": bundle.noise_id,
        "receipt_id": receipt.receipt_id,
    }:
        raise IntegrityError("T02A qualification-to-receipt link differs.")
    for expected, relative in (
        line.split(maxsplit=1) for line in (path / "SHA256SUMS").read_text().splitlines()
    ):
        if sha256_file(path / relative.strip()) != expected:
            raise IntegrityError(f"T02A SHA256SUMS mismatch: {relative.strip()}.")
    return bundle


def _threshold_semantics_payload(*, parent_noise_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "parent_noise_id": parent_noise_id,
        "calibration_scope": "conditional_pooled_endpoint_sampling_noise",
        "renamed_estimands": {
            "mass_improvement_margin_interval_log_rmse_q95": {
                "replacement": "observed_endpoint_sampling_rmse_q95",
                "allowed_use": "absolute_adequacy_and_preprocessing_tolerance_reference",
                "forbidden_use": "direct_model_versus_baseline_improvement_margin",
            },
            "minimum_detectable_abs_interval_log_mass_effect": {
                "replacement": "guide_abs_error_q95_target_median_q95",
                "allowed_use": "descriptive_conservative_guide_noise_summary",
                "forbidden_use": "formal_minimum_detectable_effect_or_power_claim",
            },
            "top_k_overlap": {
                "replacement": "guide_top_k_overlap",
                "allowed_use": "guide_level_rank_stability",
                "forbidden_use": "target_level_top_k_overlap_claim",
            },
            "bottom_k_overlap": {
                "replacement": "guide_bottom_k_overlap",
                "allowed_use": "guide_level_rank_stability",
                "forbidden_use": "target_level_bottom_k_overlap_claim",
            },
        },
        "checkpoint_use": {
            "source_representation": "use_source_checkpoint_rows",
            "terminal_reconstruction": "use_terminal_checkpoint_rows",
            "pooled_rows": "descriptive_only",
        },
        "model_comparison": {
            "required_statistic": "paired_loss_difference_on_shared_bootstrap_catalog",
            "promotion_rule": "upper_q95_delta_below_negative_preregistered_epsilon",
            "epsilon_source": "separate_synthetic_or_model_comparison_null",
        },
        "formal_detectable_effect": {
            "status": "not_estimated",
            "required_addition": "prespecified_alpha_power_estimand_and_detection_rule",
        },
        "information_boundary": (
            "learned_model_outputs_and_external_biological_annotations_not_read;"
            "observed_pooled_endpoint_counts_used_to_characterize_conditional_sampling_noise"
        ),
    }


def derive_raw_count_mass_noise_amendment(
    destination: Path,
    *,
    t02a_bundle: Path,
    pooled_bundle: Path,
    count_store: Path,
) -> Path:
    """Publish a semantic amendment derived from one immutable T02A bundle."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    parent = verify_raw_count_mass_noise(
        t02a_bundle, pooled_bundle=pooled_bundle, count_store=count_store
    )
    raw = pd.read_parquet(t02a_bundle / parent.raw_split_metrics.relative_uri)
    raw_repeat = pd.read_parquet(t02a_bundle / parent.raw_repeat_summary.relative_uri)
    mass = pd.read_parquet(t02a_bundle / parent.mass_bootstrap.relative_uri)
    mass_guide = pd.read_parquet(t02a_bundle / parent.mass_guide_noise.relative_uri)
    mass_target = pd.read_parquet(t02a_bundle / parent.mass_target_noise.relative_uri)
    checkpoints = (parent.source_checkpoint, parent.terminal_checkpoint)
    checkpoint_thresholds = _checkpoint_thresholds(
        raw=raw, raw_repeat=raw_repeat, checkpoints=checkpoints
    )
    recomputed = _interpreted_thresholds(
        raw=raw,
        raw_repeat=raw_repeat,
        mass=mass,
        mass_guide=mass_guide,
        mass_target=mass_target,
    )
    rank_stability = _target_rank_stability(mass)
    semantics = _threshold_semantics_payload(parent_noise_id=parent.noise_id)
    parent_bundle_sha256 = sha256_file(t02a_bundle / "raw-count-mass-noise.json")
    implementation_hash, implementation_files = _implementation_identity()
    environment = environment_identity()
    environment_hash = environment_lock_hash()

    def writer(temp: Path) -> None:
        checkpoint_thresholds.to_parquet(temp / "THRESHOLDS_BY_CHECKPOINT.parquet", index=False)
        _write_json(temp / "RECOMPUTED_THRESHOLDS.json", recomputed)
        rank_stability.to_parquet(temp / "TARGET_RANK_STABILITY.parquet", index=False)
        _write_json(temp / "THRESHOLD_SEMANTICS.json", semantics)
        _write_json(
            temp / "IMPLEMENTATION.sha256",
            {
                "schema_version": 1,
                "implementation_hash": implementation_hash,
                "files": implementation_files,
            },
        )
        _write_json(temp / "ENVIRONMENT.json", environment)
        bundle_payload = {
            "schema_version": 1,
            "amendment_id": "pending",
            "parent_noise_id": parent.noise_id,
            "parent_bundle_sha256": parent_bundle_sha256,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
            "source_checkpoint": parent.source_checkpoint,
            "terminal_checkpoint": parent.terminal_checkpoint,
            "thresholds_by_checkpoint": artifact_ref(
                temp,
                temp / "THRESHOLDS_BY_CHECKPOINT.parquet",
                schema_id="credo.t02a_thresholds_by_checkpoint",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "recomputed_thresholds": artifact_ref(
                temp,
                temp / "RECOMPUTED_THRESHOLDS.json",
                schema_id="credo.t02a_recomputed_thresholds",
                media_type="application/json",
            ).model_dump(mode="json"),
            "target_rank_stability": artifact_ref(
                temp,
                temp / "TARGET_RANK_STABILITY.parquet",
                schema_id="credo.t02a_target_rank_stability",
                media_type="application/x-parquet",
            ).model_dump(mode="json"),
            "threshold_semantics": artifact_ref(
                temp,
                temp / "THRESHOLD_SEMANTICS.json",
                schema_id="credo.t02a_threshold_semantics",
                media_type="application/json",
            ).model_dump(mode="json"),
            "implementation_identity": artifact_ref(
                temp,
                temp / "IMPLEMENTATION.sha256",
                schema_id="credo.t02a_amendment_implementation_identity",
                media_type="application/json",
            ).model_dump(mode="json"),
            "environment_identity": artifact_ref(
                temp,
                temp / "ENVIRONMENT.json",
                schema_id="credo.t02a_amendment_environment_identity",
                media_type="application/json",
            ).model_dump(mode="json"),
        }
        bundle_payload["amendment_id"] = contract_id(bundle_payload, id_field="amendment_id")
        amendment = RawCountMassNoiseAmendment.model_validate(bundle_payload)
        _write_json(
            temp / "raw-count-mass-noise-amendment.json",
            amendment.model_dump(mode="json"),
        )
        receipt_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "amendment_id": amendment.amendment_id,
            "parent_noise_id": parent.noise_id,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
            "status": "pass",
            "parent_bundle_verified": True,
            "table_invariants_pass": True,
            "thresholds_recomputed": True,
            "checkpoint_thresholds_recomputed": True,
            "misleading_labels_retired": True,
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = RawCountMassNoiseAmendmentReceipt.model_validate(receipt_payload)
        _write_json(temp / "VERIFICATION_RECEIPT.json", receipt.model_dump(mode="json"))
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_raw_count_mass_noise_amendment(
        destination,
        t02a_bundle=t02a_bundle,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
        _verified_parent=parent,
    )
    return destination


def verify_raw_count_mass_noise_amendment(
    path: Path,
    *,
    t02a_bundle: Path,
    pooled_bundle: Path,
    count_store: Path,
    _verified_parent: RawCountMassNoiseBundle | None = None,
) -> RawCountMassNoiseAmendment:
    """Verify a T02A amendment against its exact immutable parent tables."""

    verify_directory(path)
    amendment = RawCountMassNoiseAmendment.model_validate_json(
        (path / "raw-count-mass-noise-amendment.json").read_text()
    )
    receipt = RawCountMassNoiseAmendmentReceipt.model_validate_json(
        (path / "VERIFICATION_RECEIPT.json").read_text()
    )
    parent = _verified_parent or verify_raw_count_mass_noise(
        t02a_bundle, pooled_bundle=pooled_bundle, count_store=count_store
    )
    if (
        amendment.parent_noise_id != parent.noise_id
        or receipt.parent_noise_id != parent.noise_id
        or receipt.amendment_id != amendment.amendment_id
        or amendment.parent_bundle_sha256 != sha256_file(t02a_bundle / "raw-count-mass-noise.json")
        or amendment.source_checkpoint != parent.source_checkpoint
        or amendment.terminal_checkpoint != parent.terminal_checkpoint
        or receipt.status != "pass"
    ):
        raise IntegrityError("T02A amendment, receipt, or immutable parent differs.")
    for reference in (
        amendment.thresholds_by_checkpoint,
        amendment.recomputed_thresholds,
        amendment.target_rank_stability,
        amendment.threshold_semantics,
        amendment.implementation_identity,
        amendment.environment_identity,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(f"T02A amendment artifact differs: {reference.relative_uri}.")
    raw = pd.read_parquet(t02a_bundle / parent.raw_split_metrics.relative_uri)
    raw_repeat = pd.read_parquet(t02a_bundle / parent.raw_repeat_summary.relative_uri)
    mass = pd.read_parquet(t02a_bundle / parent.mass_bootstrap.relative_uri)
    mass_guide = pd.read_parquet(t02a_bundle / parent.mass_guide_noise.relative_uri)
    mass_target = pd.read_parquet(t02a_bundle / parent.mass_target_noise.relative_uri)
    expected_checkpoint = _checkpoint_thresholds(
        raw=raw,
        raw_repeat=raw_repeat,
        checkpoints=(parent.source_checkpoint, parent.terminal_checkpoint),
    )
    expected_rank = _target_rank_stability(mass)
    try:
        pd.testing.assert_frame_equal(
            pd.read_parquet(path / amendment.thresholds_by_checkpoint.relative_uri),
            expected_checkpoint,
            check_exact=True,
            check_dtype=True,
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(path / amendment.target_rank_stability.relative_uri),
            expected_rank,
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as exc:
        raise IntegrityError("T02A amendment parquet tables do not recompute exactly.") from exc
    expected_thresholds = _interpreted_thresholds(
        raw=raw,
        raw_repeat=raw_repeat,
        mass=mass,
        mass_guide=mass_guide,
        mass_target=mass_target,
    )
    expected_semantics = _threshold_semantics_payload(parent_noise_id=parent.noise_id)
    implementation = json.loads((path / amendment.implementation_identity.relative_uri).read_text())
    environment = json.loads((path / amendment.environment_identity.relative_uri).read_text())
    if (
        json.loads((path / amendment.recomputed_thresholds.relative_uri).read_text())
        != expected_thresholds
        or json.loads((path / amendment.threshold_semantics.relative_uri).read_text())
        != expected_semantics
        or implementation.get("implementation_hash") != amendment.implementation_hash
        or sha256_bytes(canonical_json_bytes(implementation.get("files")))
        != amendment.implementation_hash
        or receipt.implementation_hash != amendment.implementation_hash
        or sha256_bytes(canonical_json_bytes(environment)) != amendment.environment_hash
        or receipt.environment_hash != amendment.environment_hash
    ):
        raise IntegrityError("T02A amendment JSON does not match recomputed semantics.")
    for expected, relative in (
        line.split(maxsplit=1) for line in (path / "SHA256SUMS").read_text().splitlines()
    ):
        if sha256_file(path / relative.strip()) != expected:
            raise IntegrityError(f"T02A amendment SHA256SUMS mismatch: {relative.strip()}.")
    return amendment
