"""One-shot evaluation, baseline audit, and final sealing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from ..canonical import canonical_json_bytes, contract_id, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    BaselineRegistry,
    EvaluationBundleManifest,
    InferenceBundleManifest,
    SealedRunManifest,
    SelectionManifest,
    SemanticStudySnapshot,
    SeriesRecord,
)
from ..errors import ContractError
from ..inference import open_inference_run
from ..numerics import rollout
from ..persistence import publish_directory, verify_directory
from ..store import CountStore


def _future_ref(workspace: Path, final: Path, temp: Path, schema: str, media: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema,
        schema_version=1,
        sha256=sha256_file(temp),
        size_bytes=temp.stat().st_size,
        media_type=media,
        relative_uri=final.relative_to(workspace).as_posix(),
    )


def _hash_rows(rows: tuple[int, ...]) -> str:
    return sha256_bytes(np.asarray(rows, dtype="<i8").tobytes())


def _training_row_order(records: tuple[SeriesRecord, ...]) -> tuple[int, ...]:
    """Canonical baseline order: every source row, then every endpoint row."""

    source = tuple(row for record in records for row in record.source_rows)
    terminal = tuple(row for record in records for row in record.terminal_rows)
    return source + terminal


def _interaction_advancement_pass(
    *,
    selected_family: str,
    interaction_bootstrap_upper: float,
    overall_bootstrap_upper: float,
    required_interaction_improvement: float,
    required_overall_improvement: float,
    target_main_bootstrap_upper: float,
    required_target_main_improvement: float,
    interaction_displacement_rms: float,
    minimum_interaction_displacement_rms: float,
    target_win_fraction: float,
    minimum_target_win_fraction: float,
    maximum_leave_one_target_out_delta: float,
    top_target_absolute_contribution_fraction: float,
    maximum_single_target_contribution_fraction: float,
) -> bool:
    """Fail closed unless the deployed family and both outer gates pass."""

    return bool(
        selected_family == "target_plus_source_target_interaction"
        and interaction_bootstrap_upper < -required_interaction_improvement
        and overall_bootstrap_upper < -required_overall_improvement
        and target_main_bootstrap_upper < -required_target_main_improvement
        and interaction_displacement_rms >= minimum_interaction_displacement_rms
        and target_win_fraction >= minimum_target_win_fraction
        and maximum_leave_one_target_out_delta < 0.0
        and top_target_absolute_contribution_fraction <= maximum_single_target_contribution_fraction
    )


def _series_means(
    records: tuple[SeriesRecord, ...], row_ids: np.ndarray[Any, Any], latents: np.ndarray[Any, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    lookup = {int(row): index for index, row in enumerate(row_ids)}
    source = np.asarray(
        [latents[[lookup[row] for row in record.source_rows]].mean(axis=0) for record in records],
        dtype=np.float32,
    )
    terminal = np.asarray(
        [latents[[lookup[row] for row in record.terminal_rows]].mean(axis=0) for record in records],
        dtype=np.float32,
    )
    return source, terminal


def _observed_gene_compositions(
    records: tuple[SeriesRecord, ...], store: CountStore
) -> np.ndarray[Any, Any]:
    row_ids = np.asarray(
        [row for record in records for row in record.terminal_rows], dtype=np.int64
    )
    batch = store.rows(row_ids).matrix
    compositions: list[np.ndarray[Any, Any]] = []
    cursor = 0
    for record in records:
        end = cursor + len(record.terminal_rows)
        counts = np.asarray(batch[cursor:end].sum(axis=0)).reshape(-1).astype(np.float64)
        total = counts.sum()
        compositions.append(counts / total if total > 0 else np.zeros_like(counts))
        cursor = end
    return np.asarray(compositions, dtype=np.float32)


def _rmse(predicted: np.ndarray[Any, Any], terminal: np.ndarray[Any, Any]) -> float:
    return float(np.sqrt(np.mean(np.square(predicted - terminal))))


def _safe_correlation(
    kind: str, left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]
) -> float | None:
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    value = (
        pearsonr(left, right).statistic if kind == "pearson" else spearmanr(left, right).statistic
    )
    return float(value) if np.isfinite(value) else None


def _population_metrics(
    frame: pd.DataFrame,
    terminal: np.ndarray[Any, Any],
    source: np.ndarray[Any, Any],
    predictions: dict[str, np.ndarray[Any, Any]],
    mask: np.ndarray[Any, Any],
) -> dict[str, Any]:
    if not np.any(mask):
        return {"series": 0, "evaluable": False}
    result: dict[str, Any] = {"series": int(mask.sum())}
    target = frame.loc[mask, "target_index"].to_numpy(dtype=np.int64)
    for name, values in predictions.items():
        result[f"{name}_rmse"] = _rmse(values[mask], terminal[mask])
        target_mse = [
            float(
                np.mean(np.square(values[mask][target == value] - terminal[mask][target == value]))
            )
            for value in np.unique(target)
        ]
        result[f"{name}_target_balanced_rmse"] = float(np.sqrt(np.mean(target_mse)))
    observed_delta = terminal[mask] - source[mask]
    model_delta = predictions["v4"][mask] - source[mask]
    result["v4_delta_flat_pearson"] = _safe_correlation(
        "pearson", model_delta.reshape(-1), observed_delta.reshape(-1)
    )
    result["v4_delta_norm_spearman"] = _safe_correlation(
        "spearman", np.linalg.norm(model_delta, axis=1), np.linalg.norm(observed_delta, axis=1)
    )
    cosine_denominator = np.linalg.norm(model_delta, axis=1) * np.linalg.norm(
        observed_delta, axis=1
    )
    valid_cosine = cosine_denominator > 0
    cosine = np.full(len(model_delta), np.nan, dtype=np.float64)
    cosine[valid_cosine] = (
        np.sum(model_delta[valid_cosine] * observed_delta[valid_cosine], axis=1)
        / cosine_denominator[valid_cosine]
    )
    result["v4_delta_cosine_mean"] = float(np.nanmean(cosine)) if valid_cosine.any() else None
    result["v4_delta_cosine_median"] = float(np.nanmedian(cosine)) if valid_cosine.any() else None
    observed_residuals: list[np.ndarray[Any, Any]] = []
    predicted_residuals: list[np.ndarray[Any, Any]] = []
    for value in np.unique(target):
        local = target == value
        if local.sum() < 2:
            continue
        observed_local = terminal[mask][local]
        predicted_local = predictions["v4"][mask][local]
        observed_residuals.append(observed_local - observed_local.mean(axis=0))
        predicted_residuals.append(predicted_local - predicted_local.mean(axis=0))
    result["sister_guide_residual_flat_pearson"] = (
        _safe_correlation(
            "pearson",
            np.concatenate(predicted_residuals).reshape(-1),
            np.concatenate(observed_residuals).reshape(-1),
        )
        if observed_residuals
        else None
    )
    return result


def _target_balanced_bootstrap_differences(
    model: np.ndarray[Any, Any],
    baseline: np.ndarray[Any, Any],
    terminal: np.ndarray[Any, Any],
    target_indices: np.ndarray[Any, Any],
    *,
    seed: int,
    draws: int,
) -> np.ndarray[Any, Any]:
    """Bootstrap targets while preserving the target-balanced RMSE estimand."""

    unique_targets = np.unique(target_indices)
    model_target_mse = np.asarray(
        [
            np.mean(np.square(model[target_indices == value] - terminal[target_indices == value]))
            for value in unique_targets
        ],
        dtype=np.float64,
    )
    baseline_target_mse = np.asarray(
        [
            np.mean(
                np.square(baseline[target_indices == value] - terminal[target_indices == value])
            )
            for value in unique_targets
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(unique_targets), size=(draws, len(unique_targets)))
    return np.sqrt(model_target_mse[sampled].mean(axis=1)) - np.sqrt(
        baseline_target_mse[sampled].mean(axis=1)
    )


def _target_balanced_rms(
    displacement: np.ndarray[Any, Any], target_indices: np.ndarray[Any, Any]
) -> float:
    target_mse = [
        float(np.mean(np.square(displacement[target_indices == value])))
        for value in np.unique(target_indices)
    ]
    return float(np.sqrt(np.mean(target_mse)))


def _target_balanced_mean(
    values: np.ndarray[Any, Any],
    target_indices: np.ndarray[Any, Any],
    controls: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    targeting = ~controls
    selected_targets = np.unique(target_indices[targeting])
    if not len(selected_targets):
        return values.mean(axis=0)
    return np.asarray(
        [values[(target_indices == value) & targeting].mean(axis=0) for value in selected_targets]
    ).mean(axis=0)


def _independent_shrunk_target_prediction(
    *,
    train_terminal: np.ndarray[Any, Any],
    train_target: np.ndarray[Any, Any],
    train_control: np.ndarray[Any, Any],
    evaluation_target: np.ndarray[Any, Any],
    evaluation_control: np.ndarray[Any, Any],
    maximum_weight: float,
    scalar_ridge: float,
) -> tuple[np.ndarray[Any, Any], float]:
    """Materialize M1 independently of whichever family was deployed."""

    global_terminal = _target_balanced_mean(train_terminal, train_target, train_control)
    numerators: list[float] = []
    denominators: list[float] = []
    offsets: dict[int, np.ndarray[Any, Any]] = {}
    for target_value in np.unique(train_target[~train_control]):
        local = (train_target == target_value) & ~train_control
        local_terminal = train_terminal[local]
        offsets[int(target_value)] = local_terminal.mean(axis=0) - global_terminal
        if len(local_terminal) < 2:
            continue
        residual = local_terminal - global_terminal
        other_mean = (local_terminal.sum(axis=0) - local_terminal) / (len(local_terminal) - 1)
        candidate = other_mean - global_terminal
        numerators.append(float(np.mean(residual * candidate)))
        denominators.append(float(np.mean(np.square(candidate))))
    alpha = (
        float(
            np.clip(
                np.mean(numerators) / (np.mean(denominators) + scalar_ridge),
                0,
                maximum_weight,
            )
        )
        if numerators
        else 0.0
    )
    prediction = np.broadcast_to(
        global_terminal, (len(evaluation_target), len(global_terminal))
    ).copy()
    for index, target_value in enumerate(evaluation_target):
        if not evaluation_control[index] and int(target_value) in offsets:
            prediction[index] += alpha * offsets[int(target_value)]
    return prediction.astype(np.float32), alpha


def _empirical_bayes_target_prediction(
    *,
    train_terminal: np.ndarray[Any, Any],
    train_target: np.ndarray[Any, Any],
    train_control: np.ndarray[Any, Any],
    evaluation_target: np.ndarray[Any, Any],
    evaluation_control: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], dict[int, float]]:
    """Training-only multiplicity/dispersion-aware target shrinkage."""

    global_terminal = _target_balanced_mean(train_terminal, train_target, train_control)
    target_values = np.unique(train_target[~train_control])
    means = {
        int(value): train_terminal[(train_target == value) & ~train_control].mean(axis=0)
        for value in target_values
    }
    tau2 = float(np.mean([np.mean(np.square(value - global_terminal)) for value in means.values()]))
    gates: dict[int, float] = {}
    for value in target_values:
        local = train_terminal[(train_target == value) & ~train_control]
        sigma2 = float(np.mean(np.square(local - local.mean(axis=0))))
        gates[int(value)] = tau2 / (tau2 + sigma2 / len(local) + 1e-12)
    prediction = np.broadcast_to(
        global_terminal, (len(evaluation_target), len(global_terminal))
    ).copy()
    for index, target_value in enumerate(evaluation_target):
        key = int(target_value)
        if not evaluation_control[index] and key in means:
            prediction[index] += gates[key] * (means[key] - global_terminal)
    return prediction.astype(np.float32), gates


def _linear_source_target_prediction(
    *,
    train_source: np.ndarray[Any, Any],
    train_terminal: np.ndarray[Any, Any],
    train_target: np.ndarray[Any, Any],
    train_control: np.ndarray[Any, Any],
    evaluation_source: np.ndarray[Any, Any],
    evaluation_target: np.ndarray[Any, Any],
    ridge: float,
) -> np.ndarray[Any, Any]:
    """Regularized additive source-plus-target comparator with no interaction."""

    primary = ~train_control
    train_source = train_source[primary]
    train_terminal = train_terminal[primary]
    train_target = train_target[primary]
    target_values = tuple(sorted(set(map(int, train_target))))
    lookup = {value: index for index, value in enumerate(target_values)}
    train_one_hot = np.zeros((len(train_target), len(target_values)), dtype=np.float64)
    train_one_hot[np.arange(len(train_target)), [lookup[int(value)] for value in train_target]] = 1
    design = np.concatenate(
        [np.ones((len(train_source), 1)), train_source.astype(np.float64), train_one_hot], axis=1
    )
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    multiplicity = np.asarray(
        [np.sum(train_target == value) for value in train_target], dtype=np.float64
    )
    weights = 1.0 / multiplicity
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_terminal = train_terminal * np.sqrt(weights)[:, None]
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_terminal,
    )
    evaluation_one_hot = np.zeros((len(evaluation_target), len(target_values)), dtype=np.float64)
    for row, value in enumerate(evaluation_target):
        if int(value) in lookup:
            evaluation_one_hot[row, lookup[int(value)]] = 1.0
    evaluation_design = np.concatenate(
        [
            np.ones((len(evaluation_source), 1)),
            evaluation_source.astype(np.float64),
            evaluation_one_hot,
        ],
        axis=1,
    )
    return np.asarray(evaluation_design @ coefficients, dtype=np.float32)


def _evaluate_bound_outer(
    workspace: Path, run: Any, destination: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    input_root = workspace / "input"
    outer_path = input_root / "outer-evaluation.json"
    plan_path = input_root / "evaluation-plan.json"
    source_manifest_path = input_root / "source-manifest.json"
    if sha256_file(source_manifest_path) != run.contract.source_manifest_hash:
        raise ContractError("Bound source manifest differs from the compiled run.")
    source_manifest = json.loads(source_manifest_path.read_text())
    for key, path in (
        ("outer_evaluation_sha256", outer_path),
        ("evaluation_plan_sha256", plan_path),
    ):
        if source_manifest.get(key) != sha256_file(path):
            raise ContractError(f"Bound source manifest does not match {path.name}.")
    outer = json.loads(outer_path.read_text())
    frozen_plan = json.loads(plan_path.read_text())
    records = tuple(SeriesRecord.model_validate(row) for row in outer["series"])
    if not records:
        raise ContractError("Outer evaluation catalog is empty.")
    baseline_registry = BaselineRegistry.model_validate_json(
        (input_root / "baseline-registry.json").read_text()
    )
    logical_baseline_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "registry": baseline_registry.model_dump(mode="json"),
                "representation_id": run.contract.representation_id,
            }
        )
    )
    if logical_baseline_hash != run.contract.baseline_registry_hash:
        raise ContractError("Baseline registry differs from the compiled run.")
    evaluator_hash = sha256_file(Path(__file__))
    if any(row.code_hash != evaluator_hash for row in baseline_registry.baselines):
        raise ContractError("Baseline implementation hash differs from the frozen evaluator.")
    train_snapshot = SemanticStudySnapshot.model_validate_json(
        (workspace / "compiled" / "snapshot.json").read_text()
    )
    train_rows = _training_row_order(train_snapshot.series)
    test_source = tuple(row for record in records for row in record.source_rows)
    test_terminal = tuple(row for record in records for row in record.terminal_rows)
    for baseline in baseline_registry.baselines:
        if (
            baseline.allowed_training_rows_hash != _hash_rows(train_rows)
            or baseline.allowed_test_source_rows_hash != _hash_rows(test_source)
            or baseline.forbidden_endpoint_rows_hash != _hash_rows(test_terminal)
        ):
            raise ContractError(f"Baseline information set is false: {baseline.baseline_id}.")

    with h5py.File(workspace / "prepared" / "latents.h5", "r") as handle:
        row_ids = handle["row_ids"][:]
        latents = handle["z"][:]
    source, terminal = _series_means(records, row_ids, latents)
    train_source = run.arrays["source_z"]
    train_terminal = run.arrays["terminal_z"]
    train_target = run.arrays["target_index"].astype(np.int64)
    train_control = run.arrays["is_control"].astype(bool)
    train_change = train_terminal - train_source
    global_delta = _target_balanced_mean(train_change, train_target, train_control)
    control_delta = (
        train_change[train_control].mean(axis=0) if train_control.any() else global_delta
    )
    global_terminal = _target_balanced_mean(train_terminal, train_target, train_control)
    target_indices = np.asarray([record.target_index for record in records], dtype=np.int64)
    evaluation_control = np.asarray([record.is_control for record in records], dtype=bool)
    target_delta_rows: list[np.ndarray[Any, Any]] = []
    target_terminal_rows: list[np.ndarray[Any, Any]] = []
    for row, target_value in zip(source, target_indices, strict=True):
        local = train_target == target_value
        target_delta_rows.append(
            row
            + ((train_terminal - train_source)[local].mean(axis=0) if local.any() else global_delta)
        )
        target_terminal_rows.append(
            train_terminal[local].mean(axis=0) if local.any() else global_terminal
        )
    predictions: dict[str, np.ndarray[Any, Any]] = {
        "persistence": source,
        "global_delta": source + global_delta,
        "control_delta": source + control_delta,
        "target_delta": np.asarray(target_delta_rows, dtype=np.float32),
        "global_terminal": np.broadcast_to(global_terminal, terminal.shape).copy(),
        "target_terminal": np.asarray(target_terminal_rows, dtype=np.float32),
    }
    interaction_pilot = bool(run.config.model.source_target_interaction_rank)
    shrunk_target_alpha: float | None = None
    if interaction_pilot:
        predictions["shrunk_target_only"], shrunk_target_alpha = (
            _independent_shrunk_target_prediction(
                train_terminal=train_terminal,
                train_target=train_target,
                train_control=train_control,
                evaluation_target=target_indices,
                evaluation_control=evaluation_control,
                maximum_weight=run.config.model.source_target_main_max_weight,
                scalar_ridge=run.config.training.source_target_main_penalty,
            )
        )
        predictions["empirical_bayes_target"], empirical_bayes_gates = (
            _empirical_bayes_target_prediction(
                train_terminal=train_terminal,
                train_target=train_target,
                train_control=train_control,
                evaluation_target=target_indices,
                evaluation_control=evaluation_control,
            )
        )
        if "linear_source_target_ridge" not in frozen_plan:
            raise ContractError("Interaction evaluation requires a frozen linear ridge.")
        if not np.isclose(
            float(frozen_plan["linear_source_target_ridge"]),
            run.config.training.noninteraction_linear_ridge,
        ):
            raise ContractError("Outer linear baseline ridge differs from training selection.")
        predictions["linear_source_plus_target"] = _linear_source_target_prediction(
            train_source=train_source,
            train_terminal=train_terminal,
            train_target=train_target,
            train_control=train_control,
            evaluation_source=source,
            evaluation_target=target_indices,
            ridge=float(frozen_plan["linear_source_target_ridge"]),
        )
    else:
        empirical_bayes_gates = {}
    device = run.device
    source_tensor = torch.from_numpy(source).to(device)
    target_tensor = torch.from_numpy(target_indices).to(device)
    control_tensor = torch.tensor([record.is_control for record in records], device=device)
    states, weights, mass = rollout(
        run.model,
        source_tensor,
        torch.tensor([record.duration for record in records], device=device),
        target_tensor,
        torch.zeros_like(target_tensor),
        control_tensor,
        torch.tensor([record.source_count + 0.5 for record in records], device=device),
        particles=run.config.evaluation.particles,
        steps=run.config.evaluation.steps,
        seed=run.config.evaluation.seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    predictions["v4"] = (states * weights.unsqueeze(-1)).sum(dim=1).cpu().numpy()
    if interaction_pilot:
        target_only_states, target_only_weights, _ = rollout(
            run.model,
            source_tensor,
            torch.tensor([record.duration for record in records], device=device),
            target_tensor,
            torch.zeros_like(target_tensor),
            control_tensor,
            torch.tensor([record.source_count + 0.5 for record in records], device=device),
            particles=run.config.evaluation.particles,
            steps=run.config.evaluation.steps,
            seed=run.config.evaluation.seed,
            effect_mode="target_only",
        )
        predictions["deployed_target_only"] = (
            (target_only_states * target_only_weights.unsqueeze(-1)).sum(dim=1).cpu().numpy()
        )
    mass_np = mass.cpu().numpy()
    weights_np = weights.cpu().numpy()
    numerical_pass = bool(
        all(np.isfinite(value).all() for value in predictions.values())
        and np.isfinite(mass_np).all()
        and (mass_np > 0).all()
        and np.isfinite(weights_np).all()
        and np.max(np.abs(weights_np.sum(axis=1) - 1.0)) <= 1e-6
    )
    frame = pd.DataFrame(
        {
            "series_id": [record.series_id for record in records],
            "target_index": target_indices,
            "is_control": [record.is_control for record in records],
            "source_cells": [record.source_count for record in records],
            "terminal_cells": [record.terminal_count for record in records],
            "outer_fold_id": run.config.outer_fold_id,
            "inner_split_id": run.config.inner_split_id,
            "optimization_seed": run.config.training.seed,
        }
    )
    if interaction_pilot:
        frame["interaction_displacement_rms"] = np.sqrt(
            np.mean(
                np.square(predictions["v4"] - predictions["deployed_target_only"]),
                axis=1,
            )
        )
    for name, values in predictions.items():
        frame[f"{name}_rmse"] = np.sqrt(np.mean(np.square(values - terminal), axis=1))
    all_mask = np.ones(len(frame), dtype=bool)
    targeting = ~frame.is_control.to_numpy(dtype=bool)
    control = ~targeting
    train_target_multiplicity = {
        int(value): int(np.sum((train_target == value) & ~train_control))
        for value in np.unique(train_target[~train_control])
    }
    interaction_eligible = np.asarray(
        [
            (not bool(is_control)) and train_target_multiplicity.get(int(value), 0) >= 2
            for value, is_control in zip(target_indices, evaluation_control, strict=True)
        ],
        dtype=bool,
    )
    frame["interaction_eligible"] = interaction_eligible
    if interaction_pilot and not interaction_eligible.any():
        raise ContractError("Outer catalog has no known targets with at least two training guides.")
    population_masks = {
        "all_series": all_mask,
        "targeting_series": targeting,
        "control_series": control,
        "interaction_eligible_targets": interaction_eligible,
    }
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "all_series": _population_metrics(frame, terminal, source, predictions, all_mask),
        "targeting_series": _population_metrics(frame, terminal, source, predictions, targeting),
        "control_series": _population_metrics(frame, terminal, source, predictions, control),
    }
    if interaction_eligible.any():
        metrics["interaction_eligible_targets"] = _population_metrics(
            frame, terminal, source, predictions, interaction_eligible
        )
    if run.capabilities.decode_gene_composition:
        observed_composition = _observed_gene_compositions(
            records, CountStore(workspace / "input/counts.h5")
        )
        gene_metrics: dict[str, Any] = {}
        for name in ("v4", "global_terminal"):
            composition = run.decode_composition(predictions[name])
            cross_entropy = -np.sum(
                observed_composition * np.log(composition.clip(min=1e-12)), axis=1
            )
            frame[f"{name}_gene_cross_entropy"] = cross_entropy
            target_means = [
                float(cross_entropy[target_indices == value].mean())
                for value in np.unique(target_indices)
            ]
            gene_metrics[f"{name}_target_balanced_cross_entropy"] = float(np.mean(target_means))
            gene_metrics[f"{name}_median_effective_genes"] = float(
                np.median(1.0 / np.square(composition).sum(axis=1))
            )
        gene_metrics["v4_minus_global_terminal_cross_entropy"] = (
            gene_metrics["v4_target_balanced_cross_entropy"]
            - gene_metrics["global_terminal_target_balanced_cross_entropy"]
        )
        gene_metrics["observed_median_effective_genes"] = float(
            np.median(1.0 / np.square(observed_composition).sum(axis=1))
        )
        metrics["gene_composition_diagnostic"] = gene_metrics
    primary_metric = str(frozen_plan.get("primary_metric", "target_balanced_rmse"))
    model_key = f"v4_{primary_metric}"
    primary_population = str(frozen_plan["primary_population"])
    if interaction_pilot and primary_population != "interaction_eligible_targets":
        raise ContractError(
            "Pooled source-target pilots require the known-target multi-guide population."
        )
    if primary_population not in metrics:
        raise ValueError(
            f"Primary population {primary_population!r} is absent from evaluation metrics."
        )
    primary_metrics = metrics[primary_population]
    declared_primary = str(frozen_plan["primary_baseline"])
    if interaction_pilot:
        if declared_primary != "best_preregistered_noninteraction":
            raise ContractError(
                "Source-target interaction evaluation requires the strongest frozen "
                "noninteraction baseline."
            )
        required_baselines = {
            "shrunk_target_only",
            "empirical_bayes_target",
            "target_terminal",
            "target_delta",
            "linear_source_plus_target",
        }
        declared_baselines = tuple(frozen_plan.get("noninteraction_baselines", ()))
        if set(declared_baselines) != required_baselines:
            raise ContractError("The pooled noninteraction baseline set is incomplete.")
        registered = {row.baseline_id for row in baseline_registry.baselines}
        if not required_baselines <= registered:
            raise ContractError(
                "The baseline registry does not bind every pooled noninteraction comparator."
            )
        primary = min(
            declared_baselines,
            key=lambda name: primary_metrics[f"{name}_{primary_metric}"],
        )
    else:
        primary = declared_primary
    key = f"{primary}_{primary_metric}"
    if key not in primary_metrics or model_key not in primary_metrics:
        raise ValueError("Primary metric or baseline is absent from the frozen population.")
    numerical_tolerance = float(frozen_plan.get("numerical_tolerance", 0.0))
    scientific_minimum = float(frozen_plan.get("scientific_minimum_improvement", 0.0))
    delta = primary_metrics[model_key] - primary_metrics[key]
    metrics["primary_comparison"] = {
        "population": primary_population,
        "baseline": primary,
        "baseline_policy": declared_primary,
        "metric": primary_metric,
        "v4": primary_metrics[model_key],
        "baseline_value": primary_metrics[key],
        "delta": delta,
        "gate_type": "numerical_tie_or_better",
        "numerical_tolerance": numerical_tolerance,
        "pass": bool(delta <= numerical_tolerance),
    }
    differences = _target_balanced_bootstrap_differences(
        predictions["v4"][population_masks[primary_population]],
        predictions[primary][population_masks[primary_population]],
        terminal[population_masks[primary_population]],
        target_indices[population_masks[primary_population]],
        seed=int(frozen_plan["bootstrap_seed"]),
        draws=int(frozen_plan["bootstrap_draws"]),
    )
    interval = [float(value) for value in np.quantile(differences, [0.025, 0.975])]
    metrics["conditional_target_bootstrap"] = {
        "seed": int(frozen_plan["bootstrap_seed"]),
        "draws": int(frozen_plan["bootstrap_draws"]),
        "interval_95": interval,
        "aggregation": "target_balanced_RMSE",
        "biological_replicate_interval": False,
    }
    plan = {
        **frozen_plan,
        "run_id": run.manifest.run_id,
        "outer_evaluation_sha256": sha256_file(outer_path),
        "baseline_registry_hash": run.contract.baseline_registry_hash,
        "one_shot": bool(not frozen_plan.get("historically_exposed", False)),
    }
    scientific_threshold = scientific_minimum + numerical_tolerance
    minimum_interaction_rms = float(frozen_plan.get("minimum_interaction_displacement_rms", 0.0))
    if interaction_pilot and minimum_interaction_rms <= 0.0:
        raise ContractError(
            "Source-target interaction evaluation requires a positive frozen effect-size floor."
        )
    targeting_interaction_rms = (
        _target_balanced_rms(
            predictions["v4"][interaction_eligible]
            - predictions["deployed_target_only"][interaction_eligible],
            target_indices[interaction_eligible],
        )
        if interaction_pilot and interaction_eligible.any()
        else 0.0
    )
    family_eligible = run.manifest.selected_family == "target_plus_source_target_interaction"
    overall_interval: list[float] | None = None
    if interaction_pilot:
        if (
            "interaction_scientific_minimum_improvement" not in frozen_plan
            or "overall_scientific_minimum_improvement" not in frozen_plan
            or "target_main_scientific_minimum_improvement" not in frozen_plan
            or "minimum_target_win_fraction" not in frozen_plan
            or "maximum_single_target_contribution_fraction" not in frozen_plan
        ):
            raise ContractError(
                "Interaction evaluation requires frozen nested margins and breadth gates."
            )
        overall_differences = _target_balanced_bootstrap_differences(
            predictions["v4"][population_masks[primary_population]],
            predictions["global_terminal"][population_masks[primary_population]],
            terminal[population_masks[primary_population]],
            target_indices[population_masks[primary_population]],
            seed=int(frozen_plan["bootstrap_seed"]),
            draws=int(frozen_plan["bootstrap_draws"]),
        )
        overall_interval = [
            float(value) for value in np.quantile(overall_differences, [0.025, 0.975])
        ]
        selection = SelectionManifest.model_validate_json(
            (workspace / "inference/selection-manifest.json").read_text()
        )
        if selection.best_target_only_baseline is None:
            raise ContractError("Interaction inference lacks the selected M1 identity.")
        selected_target_only = selection.best_target_only_baseline
        target_main_differences = _target_balanced_bootstrap_differences(
            predictions["deployed_target_only"][population_masks[primary_population]],
            predictions["global_terminal"][population_masks[primary_population]],
            terminal[population_masks[primary_population]],
            target_indices[population_masks[primary_population]],
            seed=int(frozen_plan["bootstrap_seed"]),
            draws=int(frozen_plan["bootstrap_draws"]),
        )
        target_main_interval = [
            float(value) for value in np.quantile(target_main_differences, [0.025, 0.975])
        ]
        interaction_threshold = (
            float(frozen_plan["interaction_scientific_minimum_improvement"]) + numerical_tolerance
        )
        overall_threshold = (
            float(frozen_plan["overall_scientific_minimum_improvement"]) + numerical_tolerance
        )
        target_main_threshold = (
            float(frozen_plan["target_main_scientific_minimum_improvement"]) + numerical_tolerance
        )
        primary_mask = population_masks[primary_population]
        primary_targets = target_indices[primary_mask]
        if len(np.unique(primary_targets)) < 2:
            raise ContractError("Pooled interaction inference requires at least two targets.")
        per_target_differences = {
            int(value): float(
                np.sqrt(
                    np.mean(
                        np.square(
                            predictions["v4"][primary_mask][primary_targets == value]
                            - terminal[primary_mask][primary_targets == value]
                        )
                    )
                )
                - np.sqrt(
                    np.mean(
                        np.square(
                            predictions[primary][primary_mask][primary_targets == value]
                            - terminal[primary_mask][primary_targets == value]
                        )
                    )
                )
            )
            for value in np.unique(primary_targets)
        }
        target_win_fraction = float(
            np.mean(np.asarray(list(per_target_differences.values())) < 0.0)
        )
        absolute_target_differences = np.abs(
            np.asarray(list(per_target_differences.values()), dtype=np.float64)
        )
        top_target_absolute_contribution_fraction = float(
            absolute_target_differences.max() / max(float(absolute_target_differences.sum()), 1e-12)
        )
        leave_one_target_out = []
        unique_primary_targets = np.unique(primary_targets)
        for omitted in unique_primary_targets:
            keep = primary_mask & (target_indices != omitted)
            leave_one_target_out.append(
                _target_balanced_rms(predictions["v4"][keep] - terminal[keep], target_indices[keep])
                - _target_balanced_rms(
                    predictions[primary][keep] - terminal[keep], target_indices[keep]
                )
            )
        maximum_leave_one_target_out_delta = float(max(leave_one_target_out))
        minimum_target_win_fraction = float(frozen_plan["minimum_target_win_fraction"])
        maximum_single_target_contribution_fraction = float(
            frozen_plan["maximum_single_target_contribution_fraction"]
        )
        metrics["interaction_outer_gate"] = {
            "m2_minus_m1_interval_95": interval,
            "m2_minus_m0_interval_95": overall_interval,
            "m1_minus_m0_interval_95": target_main_interval,
            "shrunk_target_alpha": shrunk_target_alpha,
            "empirical_bayes_gate_min": min(empirical_bayes_gates.values(), default=0.0),
            "empirical_bayes_gate_max": max(empirical_bayes_gates.values(), default=0.0),
            "best_noninteraction_baseline": primary,
            "selected_target_only_baseline": selected_target_only,
            "m0_global_null_rmse": primary_metrics[f"global_terminal_{primary_metric}"],
            "m1_deployed_target_only_rmse": primary_metrics[
                f"deployed_target_only_{primary_metric}"
            ],
            "m2_interaction_rmse": primary_metrics[model_key],
            "m1_minus_m0": (
                primary_metrics[f"deployed_target_only_{primary_metric}"]
                - primary_metrics[f"global_terminal_{primary_metric}"]
            ),
            "interaction_threshold_including_numerical_tolerance": interaction_threshold,
            "overall_threshold_including_numerical_tolerance": overall_threshold,
            "target_main_threshold_including_numerical_tolerance": target_main_threshold,
            "target_win_fraction": target_win_fraction,
            "minimum_target_win_fraction": minimum_target_win_fraction,
            "median_target_rmse_difference": float(
                np.median(np.asarray(list(per_target_differences.values())))
            ),
            "top_target_absolute_contribution_fraction": (
                top_target_absolute_contribution_fraction
            ),
            "maximum_single_target_contribution_fraction": (
                maximum_single_target_contribution_fraction
            ),
            "maximum_leave_one_target_out_delta": maximum_leave_one_target_out_delta,
            "per_target_rmse_differences": per_target_differences,
            "aggregation": "target_balanced_RMSE",
        }
        scientific_gate_pass = bool(
            numerical_pass
            and _interaction_advancement_pass(
                selected_family=run.manifest.selected_family,
                interaction_bootstrap_upper=interval[1],
                overall_bootstrap_upper=overall_interval[1],
                required_interaction_improvement=interaction_threshold,
                required_overall_improvement=overall_threshold,
                target_main_bootstrap_upper=target_main_interval[1],
                required_target_main_improvement=target_main_threshold,
                interaction_displacement_rms=targeting_interaction_rms,
                minimum_interaction_displacement_rms=minimum_interaction_rms,
                target_win_fraction=target_win_fraction,
                minimum_target_win_fraction=minimum_target_win_fraction,
                maximum_leave_one_target_out_delta=maximum_leave_one_target_out_delta,
                top_target_absolute_contribution_fraction=(
                    top_target_absolute_contribution_fraction
                ),
                maximum_single_target_contribution_fraction=(
                    maximum_single_target_contribution_fraction
                ),
            )
        )
    else:
        scientific_gate_pass = bool(numerical_pass and interval[1] < -scientific_threshold)
    audit = {
        "schema_version": 1,
        "status": "engineering_complete" if numerical_pass else "numerical_failure",
        "qualified": False,
        "numerical_pass": numerical_pass,
        "scientific_gate_pass": scientific_gate_pass,
        "selected_family": run.manifest.selected_family,
        "interaction_family_eligible": family_eligible,
        "interaction_displacement_rms": targeting_interaction_rms,
        "minimum_interaction_displacement_rms": minimum_interaction_rms,
        "scientific_gate": (
            "nested_M2_beats_M1_and_M0_with_target_bootstrap"
            if interaction_pilot
            else "conditional_target_bootstrap_upper_below_negative_minimum"
        ),
        "interaction_vs_target_only_interval_95": interval if interaction_pilot else None,
        "interaction_vs_global_null_interval_95": overall_interval,
        "shrunk_target_main_weight": shrunk_target_alpha,
        "pooled_estimand": run.config.pooled_estimand,
        "outer_fold_id": run.config.outer_fold_id,
        "inner_split_id": run.config.inner_split_id,
        "pooled_outer_fold_ids": run.config.pooled_outer_fold_ids,
        "pooled_inner_split_ids": run.config.pooled_inner_split_ids,
        "pooled_optimization_seeds": run.config.pooled_optimization_seeds,
        "optimization_seed": run.config.training.seed,
        "state_selection_calibration_stage": (run.contract.state_selection_calibration_stage),
        "scientific_minimum_improvement": scientific_minimum,
        "scientific_improvement_threshold_including_numerical_tolerance": scientific_threshold,
        "outer_evaluation_access": (
            "current_lifecycle_bound_before_compile_but_historically_exposed"
            if frozen_plan.get("historically_exposed", False)
            else "one_shot_bound_before_compile"
        ),
        "checkpoint_selected_before_evaluation": True,
        "endpoint_used_for_eligibility": bool(
            frozen_plan.get("endpoint_used_for_eligibility", False)
        ),
        "eligibility_rule": frozen_plan.get("eligibility_rule"),
        "claim_status": "engineering_only_historical_endpoint_exposure",
        "scientific_scope": "one_pooled_context_no_library_or_pool_predictors",
        "biological_replication": "unavailable",
        "evaluator_sha256": evaluator_hash,
        "device": str(device),
    }
    return metrics, plan, audit, frame


def evaluate_run(config_path: Path, *, device: str = "cpu") -> Path:
    import yaml

    root = config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text())
    workspace = (root / raw["workspace"]).resolve()
    run = open_inference_run(workspace / "inference", device=device, verify="full")
    destination = workspace / "evaluation"
    if (workspace / "input" / "outer-evaluation.json").is_file():
        metrics, plan, audit, frame = _evaluate_bound_outer(workspace, run, destination)
        predicted_mass = np.full(len(frame), np.nan)
        observed_counts = np.full(len(frame), np.nan)
        evaluable = np.ones(len(frame), dtype=bool)
    else:
        predicted_state, predicted_mass, _ = run.terminal()
        target_state = run.arrays["terminal_z"]
        evaluable = np.isfinite(target_state).all(axis=1)
        state_rmse = float(
            np.sqrt(np.mean((predicted_state[evaluable] - target_state[evaluable]) ** 2))
        )
        persistence_rmse = float(
            np.sqrt(np.mean((run.arrays["source_z"][evaluable] - target_state[evaluable]) ** 2))
        )
        observed_counts = run.arrays["terminal_counts"].astype(float)
        predicted_abundance = np.log(predicted_mass + 1e-12)
        observed_abundance = np.log(observed_counts + 0.5) - np.log(
            run.arrays["source_counts"] + 0.5
        )
        abundance_rho = (
            float(spearmanr(predicted_abundance, observed_abundance).statistic)
            if run.capabilities.predict_relative_mass and len(observed_counts) > 1
            else None
        )
        if abundance_rho is not None and not np.isfinite(abundance_rho):
            abundance_rho = None
        metrics = {
            "schema_version": 1,
            "state_rmse": state_rmse,
            "persistence_rmse": persistence_rmse,
            "state_delta_vs_persistence": state_rmse - persistence_rmse,
            "relative_abundance_spearman": abundance_rho,
            "evaluable_series": int(evaluable.sum()),
            "total_series": int(len(evaluable)),
        }
        plan = {
            "schema_version": 1,
            "run_id": run.manifest.run_id,
            "information_set": run.contract.information_set_hash,
            "one_shot": True,
            "baseline_registry_hash": run.contract.baseline_registry_hash,
            "multiplicity_plan_hash": run.contract.multiplicity_plan_hash,
            "selected_family": run.manifest.selected_family,
        }
        audit = {
            "schema_version": 1,
            "outer_evaluation_access": "one_shot",
            "checkpoint_selected_before_evaluation": True,
            "endpoint_used_for_eligibility": False,
            "claim_status": "engineering_only",
            "selected_family": run.manifest.selected_family,
            "interaction_family_eligible": False,
            "scientific_gate_pass": False,
        }
        frame = pd.DataFrame(
            {
                "series_id": run.arrays["series_ids"].astype(str),
                "target_index": run.arrays["target_index"],
                "pool_index": run.arrays["pool_index"],
                "state_evaluable": evaluable,
                "predicted_relative_mass": predicted_mass,
                "observed_terminal_count": observed_counts,
            }
        )

    def writer(temp: Path) -> None:
        (temp / "plan.json").write_bytes(canonical_json_bytes(plan) + b"\n")
        (temp / "metrics.json").write_bytes(canonical_json_bytes(metrics) + b"\n")
        (temp / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
        frame.to_parquet(temp / "predictions.parquet", index=False)
        final = destination
        payload = {
            "schema_version": 1,
            "evaluation_id": "pending",
            "run_id": run.manifest.run_id,
            "plan": _future_ref(
                workspace,
                final / "plan.json",
                temp / "plan.json",
                "credo.evaluation_plan",
                "application/json",
            ).model_dump(mode="json"),
            "metrics": _future_ref(
                workspace,
                final / "metrics.json",
                temp / "metrics.json",
                "credo.metrics",
                "application/json",
            ).model_dump(mode="json"),
            "predictions": _future_ref(
                workspace,
                final / "predictions.parquet",
                temp / "predictions.parquet",
                "credo.predictions",
                "application/x-parquet",
            ).model_dump(mode="json"),
            "audit": _future_ref(
                workspace,
                final / "audit.json",
                temp / "audit.json",
                "credo.evaluation_audit",
                "application/json",
            ).model_dump(mode="json"),
        }
        payload["evaluation_id"] = contract_id(payload, id_field="evaluation_id")
        manifest = EvaluationBundleManifest.model_validate(payload)
        (temp / "evaluation.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(destination, writer)
    return destination


def seal_run(config_path: Path) -> Path:
    import yaml

    root = config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text())
    workspace = (root / raw["workspace"]).resolve()
    inference_root = workspace / "inference"
    evaluation_root = workspace / "evaluation"
    verify_directory(inference_root)
    verify_directory(evaluation_root)
    inference = InferenceBundleManifest.model_validate_json(
        (inference_root / "inference.json").read_text()
    )
    evaluation = EvaluationBundleManifest.model_validate_json(
        (evaluation_root / "evaluation.json").read_text()
    )
    if evaluation.run_id != inference.run_id:
        raise ValueError("Evaluation does not belong to inference bundle.")
    destination = workspace / "sealed"

    def writer(temp: Path) -> None:
        claim = json.loads((evaluation_root / "audit.json").read_text())
        (temp / "claim-audit.json").write_bytes(canonical_json_bytes(claim) + b"\n")
        inference_ref = ArtifactRef(
            schema_id="credo.inference_bundle",
            schema_version=1,
            sha256=sha256_file(inference_root / "inference.json"),
            size_bytes=(inference_root / "inference.json").stat().st_size,
            media_type="application/json",
            relative_uri="../inference/inference.json".replace("../", "inference/"),
        )
        # Sealed refs are logical bundle IDs rather than navigable filesystem
        # paths; verification resolves parents from the sibling workspace.
        inference_ref = inference_ref.model_copy(
            update={"relative_uri": "inference/inference.json"}
        )
        evaluation_ref = ArtifactRef(
            schema_id="credo.evaluation_bundle",
            schema_version=1,
            sha256=sha256_file(evaluation_root / "evaluation.json"),
            size_bytes=(evaluation_root / "evaluation.json").stat().st_size,
            media_type="application/json",
            relative_uri="evaluation/evaluation.json",
        )
        claim_ref = _future_ref(
            workspace,
            destination / "claim-audit.json",
            temp / "claim-audit.json",
            "credo.claim_audit",
            "application/json",
        )
        payload = {
            "schema_version": 1,
            "sealed_id": "pending",
            "inference": inference_ref.model_dump(mode="json"),
            "evaluations": [evaluation_ref.model_dump(mode="json")],
            "claim_audit": claim_ref.model_dump(mode="json"),
        }
        payload["sealed_id"] = contract_id(payload, id_field="sealed_id")
        manifest = SealedRunManifest.model_validate(payload)
        (temp / "sealed.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
        )

    publish_directory(destination, writer)
    return destination
