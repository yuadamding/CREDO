"""Evidence-bearing pooled calibration for nested state-family selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from ..canonical import atomic_json, canonical_json_bytes, sha256_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    SemanticStudySnapshot,
    StateCalibrationCheckpointScore,
    StateSelectionCalibration,
    StateSelectionCalibrationResults,
    StateSelectionCalibrationRow,
)
from ..prepare.pipeline import load_config, load_prepared_arrays
from ..runtime_identity import environment_lock_hash, implementation_tree_hash
from .trainer import (
    _initialize_state_channels,
    _loss,
    _materialize_target_main_baseline,
    _new_model_optimizer,
    _state_split,
    _tensor_problem,
    _training_diagnostics,
)

NullFamily = Literal["global_target_main", "conditional_interaction", "joint_nested"]
TargetOnlyBaseline = Literal["shrunk_target_only", "empirical_bayes_target", "target_terminal"]
NoninteractionBaseline = Literal[
    "shrunk_target_only",
    "empirical_bayes_target",
    "target_terminal",
    "target_delta",
    "linear_source_plus_target",
]
SelectedStateFamily = Literal[
    "global_terminal_null",
    "shrunk_sister_guide_target_terminal",
    "selected_training_only_target_main",
    "target_plus_source_target_interaction",
]


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def calibration_population_hashes(snapshot: SemanticStudySnapshot) -> tuple[str, str]:
    """Bind target multiplicity and the exact source/terminal support structure."""

    target_values = sorted(
        {series.target_index for series in snapshot.series if not series.is_control}
    )
    guide_hash = _hash(
        {
            "guide_counts_per_target": sorted(
                sum(
                    int(series.target_index == target and not series.is_control)
                    for series in snapshot.series
                )
                for target in target_values
            )
        }
    )
    support_hash = _hash(
        {
            "rows": [
                {
                    "series_id": series.series_id,
                    "target_index": series.target_index,
                    "is_control": series.is_control,
                    "source_count": series.source_count,
                    "terminal_count": series.terminal_count,
                }
                for series in sorted(snapshot.series, key=lambda item: item.series_id)
            ]
        }
    )
    return guide_hash, support_hash


def calibration_protocol_payload(results: StateSelectionCalibrationResults) -> dict[str, Any]:
    return results.model_dump(mode="json", exclude={"rows"})


def optimizer_fingerprint(config: Any) -> str:
    return _hash(
        {
            "class": config.training.optimizer_name,
            "learning_rate": config.training.learning_rate,
            "betas": config.training.optimizer_betas,
            "epsilon": config.training.optimizer_epsilon,
            "weight_decay": config.training.optimizer_weight_decay,
            "deterministic_algorithms": config.training.deterministic,
        }
    )


def _prepared_problem(
    config_path: Path,
) -> tuple[Any, Any, SemanticStudySnapshot, dict[str, np.ndarray[Any, Any]]]:
    from ..compile.compiler import _lookup_means

    config = load_config(config_path)
    root = config_path.parent.resolve()
    workspace = (root / config.workspace).resolve()
    prepared, row_ids, latents = load_prepared_arrays(workspace)
    snapshot = SemanticStudySnapshot.model_validate_json(
        (root / config.semantic_snapshot).resolve().read_text()
    )
    source_z, terminal_z = _lookup_means(snapshot, row_ids, latents)
    duration = np.asarray([series.duration for series in snapshot.series], dtype=np.float32)
    grid_steps = np.ceil(duration / config.evaluation.max_step_duration).astype(np.int64)
    arrays = {
        "source_z": source_z,
        "terminal_z": terminal_z,
        "target_index": np.asarray(
            [series.target_index for series in snapshot.series], dtype=np.int64
        ),
        "pool_index": np.asarray([series.pool_index for series in snapshot.series], dtype=np.int64),
        "is_control": np.asarray([series.is_control for series in snapshot.series], dtype=np.uint8),
        "duration": duration,
        "grid_steps": grid_steps,
        "grid_step_size": duration / grid_steps,
        "source_counts": np.asarray(
            [series.source_count for series in snapshot.series], dtype=np.int64
        ),
        "terminal_counts": np.asarray(
            [series.terminal_count for series in snapshot.series], dtype=np.int64
        ),
        "series_ids": np.asarray([series.series_id for series in snapshot.series]),
    }
    return config, prepared, snapshot, arrays


def _nonidentity_permutation(size: int, generator: np.random.Generator) -> np.ndarray[Any, Any]:
    if size <= 1:
        return np.arange(size)
    identity = np.arange(size)
    for _ in range(64):
        candidate = generator.permutation(size)
        if not np.array_equal(candidate, identity):
            return candidate
    return np.roll(identity, 1)


def _global_target_main_null(
    arrays: dict[str, np.ndarray[Any, Any]], seed: int
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """Permute terminal target blocks within equal-multiplicity strata."""

    result = {name: value.copy() for name, value in arrays.items()}
    generator = np.random.default_rng(seed)
    target = arrays["target_index"].astype(np.int64, copy=False)
    control = arrays["is_control"].astype(bool, copy=False)
    groups: dict[int, list[int]] = {}
    target_values = sorted(set(map(int, target[~control])))
    target_means = [
        arrays["terminal_z"][(target == value) & ~control].mean(axis=0) for value in target_values
    ]
    global_terminal = np.asarray(target_means).mean(axis=0)
    for value in target_values:
        groups.setdefault(int(np.sum((target == value) & ~control)), []).append(value)
    mapping: list[dict[str, Any]] = []
    for multiplicity, values in sorted(groups.items()):
        if len(values) == 1:
            destination_target = values[0]
            destination = np.where((target == destination_target) & ~control)[0]
            block = arrays["terminal_z"][destination]
            result["terminal_z"][destination] = block - block.mean(axis=0) + global_terminal
            mapping.append(
                {
                    "multiplicity": multiplicity,
                    "destination_target": destination_target,
                    "source_target": -1,
                    "fallback": "pooled_centered_residual",
                }
            )
            continue
        permutation = _nonidentity_permutation(len(values), generator)
        source_terminal = arrays["terminal_z"].copy()
        for destination_position, source_position in enumerate(permutation.tolist()):
            destination_target = values[destination_position]
            source_target = values[source_position]
            destination = np.where((target == destination_target) & ~control)[0]
            source = np.where((target == source_target) & ~control)[0]
            result["terminal_z"][destination] = source_terminal[source]
            mapping.append(
                {
                    "multiplicity": multiplicity,
                    "destination_target": destination_target,
                    "source_target": source_target,
                }
            )
    return result, {"null": "global_target_main", "seed": seed, "mapping": mapping}


def _conditional_interaction_null(
    arrays: dict[str, np.ndarray[Any, Any]], seed: int
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """Permute guide source states within target while retaining target means."""

    result = {name: value.copy() for name, value in arrays.items()}
    generator = np.random.default_rng(seed)
    target = arrays["target_index"].astype(np.int64, copy=False)
    control = arrays["is_control"].astype(bool, copy=False)
    mapping: list[dict[str, Any]] = []
    for target_value, is_control in sorted(set(zip(list(target), list(control), strict=True))):
        local = np.where((target == target_value) & (control == is_control))[0]
        permutation = _nonidentity_permutation(len(local), generator)
        result["source_z"][local] = arrays["source_z"][local[permutation]]
        mapping.append(
            {
                "target": int(target_value),
                "is_control": bool(is_control),
                "destination": local.tolist(),
                "source": local[permutation].tolist(),
            }
        )
    return result, {"null": "conditional_interaction", "seed": seed, "mapping": mapping}


def _null_problem(
    arrays: dict[str, np.ndarray[Any, Any]], family: NullFamily, seed: int
) -> tuple[dict[str, np.ndarray[Any, Any]], str]:
    if family == "global_target_main":
        result, mapping = _global_target_main_null(arrays, seed)
    elif family == "conditional_interaction":
        result, mapping = _conditional_interaction_null(arrays, seed)
    else:
        terminal_null, terminal_mapping = _global_target_main_null(arrays, seed)
        result, source_mapping = _conditional_interaction_null(terminal_null, seed + 1_000_003)
        mapping = {
            "null": "joint_nested",
            "terminal": terminal_mapping,
            "source": source_mapping,
        }
    return result, _hash(mapping)


def _fit_one_null(
    arrays: dict[str, np.ndarray[Any, Any]],
    config: Any,
    *,
    family: NullFamily,
    replicate_index: int,
    permutation_seed: int,
    optimizer_seed: int,
    initialization_seed: int,
    device: torch.device,
) -> StateSelectionCalibrationRow:
    replicate_config = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={"seed": optimizer_seed, "initialization_seed": initialization_seed}
            )
        }
    )
    null_arrays, permutation_hash = _null_problem(arrays, family, permutation_seed)
    model, optimizer = _new_model_optimizer(replicate_config, device)
    problem = _tensor_problem(null_arrays, device)
    split = _state_split(null_arrays, replicate_config, device)
    _initialize_state_channels(model, problem, split.fit_indices, replicate_config)
    diagnostics = _training_diagnostics(model, problem, split, replicate_config)
    null_score = float(diagnostics["state_validation_global_null_rmse"])
    target_score = float(diagnostics["state_validation_best_target_only_rmse"])
    target_baseline = str(diagnostics["state_validation_best_target_only_baseline"])
    noninteraction_score = float(diagnostics["state_validation_best_noninteraction_rmse"])
    _materialize_target_main_baseline(
        model,
        problem,
        split.fit_indices,
        replicate_config,
        target_baseline,
    )

    def checkpoint_score(
        update: int, row: dict[str, float | str], interaction_score: float
    ) -> StateCalibrationCheckpointScore:
        return StateCalibrationCheckpointScore(
            update=update,
            global_null_score=float(row["state_validation_global_null_rmse"]),
            shrunk_target_only_score=float(row["state_validation_shrunk_target_only_rmse"]),
            empirical_bayes_target_score=float(
                row["state_validation_noninteraction_empirical_bayes_target_rmse"]
            ),
            target_terminal_score=float(
                row["state_validation_noninteraction_target_terminal_rmse"]
            ),
            target_delta_score=float(row["state_validation_noninteraction_target_delta_rmse"]),
            linear_source_plus_target_score=float(
                row["state_validation_noninteraction_linear_source_plus_target_rmse"]
            ),
            best_target_only_score=float(row["state_validation_best_target_only_rmse"]),
            best_target_only_baseline=cast(
                TargetOnlyBaseline, row["state_validation_best_target_only_baseline"]
            ),
            best_noninteraction_score=float(row["state_validation_best_noninteraction_rmse"]),
            best_noninteraction_baseline=cast(
                NoninteractionBaseline, row["state_validation_best_noninteraction_baseline"]
            ),
            interaction_score=interaction_score,
        )

    scores = [checkpoint_score(0, diagnostics, target_score)]
    maximum_update = max(replicate_config.training.state_checkpoint_updates)
    checkpoint_set = set(replicate_config.training.state_checkpoint_updates)
    for update in range(1, maximum_update + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, problem, replicate_config, update, None, split)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite calibration loss at {family}, permutation {permutation_seed}, "
                f"update {update}."
            )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if update in checkpoint_set:
            diagnostics = _training_diagnostics(model, problem, split, replicate_config)
            scores.append(
                checkpoint_score(
                    update,
                    diagnostics,
                    float(diagnostics["state_validation_full_interaction_rmse"]),
                )
            )
    best = min(scores[1:], key=lambda item: (item.interaction_score, item.update))
    interaction_gain = noninteraction_score - best.interaction_score
    overall_gain = null_score - best.interaction_score
    target_gain = null_score - target_score
    selected_family: SelectedStateFamily
    if (
        interaction_gain
        >= replicate_config.training.state_validation_interaction_minimum_improvement
        and overall_gain >= replicate_config.training.state_validation_target_minimum_improvement
    ):
        selected_family = "target_plus_source_target_interaction"
        selected_update = best.update
    elif target_gain >= replicate_config.training.state_validation_target_minimum_improvement:
        selected_family = "selected_training_only_target_main"
        selected_update = 0
    else:
        selected_family = "global_terminal_null"
        selected_update = 0
    selected_m2 = selected_family == "target_plus_source_target_interaction"
    return StateSelectionCalibrationRow(
        replicate_index=replicate_index,
        null_family=family,
        outer_fold_id=replicate_config.outer_fold_id,
        inner_split_id=replicate_config.inner_split_id,
        permutation_seed=permutation_seed,
        optimizer_seed=optimizer_seed,
        initialization_seed=initialization_seed,
        permutation_sha256=permutation_hash,
        fit_series_sha256=split.fit_series_hash,
        validation_series_sha256=split.validation_series_hash,
        checkpoint_scores=tuple(scores),
        selected_update=selected_update,
        selected_family=selected_family,
        maximum_target_main_gain=target_gain,
        maximum_interaction_gain=max(
            item.best_noninteraction_score - item.interaction_score for item in scores[1:]
        ),
        false_target_main_selected=(
            family == "global_target_main" and selected_family != "global_terminal_null"
        ),
        false_interaction_selected=family == "conditional_interaction" and selected_m2,
        false_joint_interaction_selected=family == "joint_nested" and selected_m2,
    )


def run_state_selection_calibration(
    config_path: Path,
    output_root: Path,
    *,
    repeats_per_null: int,
    permutation_seed_start: int,
    optimizer_seed_start: int,
    initialization_seed_start: int,
    device: str = "cpu",
    calibration_stage: Literal["development", "locked_audit"] = "development",
    development_calibration: Path | None = None,
) -> tuple[Path, Path]:
    """Run three null families on one fixed split and publish strict evidence."""

    required = 199 if calibration_stage == "locked_audit" else 119
    if repeats_per_null < required:
        raise ValueError(
            f"{calibration_stage} calibration requires at least {required} fits per null."
        )
    development_sha256: str | None = None
    development_results: StateSelectionCalibrationResults | None = None
    if calibration_stage == "locked_audit":
        if development_calibration is None:
            raise ValueError("Locked audit calibration requires a development receipt.")
        development_receipt = StateSelectionCalibration.model_validate_json(
            development_calibration.read_text()
        )
        if development_receipt.calibration_stage != "development":
            raise ValueError("Locked audit parent must be a development calibration.")
        development_results_path = (
            development_calibration.parent / development_receipt.results_artifact.relative_uri
        ).resolve()
        if sha256_file(development_results_path) != development_receipt.results_artifact.sha256:
            raise ValueError("Development calibration result bytes differ from its receipt.")
        development_results = StateSelectionCalibrationResults.model_validate_json(
            development_results_path.read_text()
        )
        development_sha256 = sha256_file(development_calibration)
    elif development_calibration is not None:
        raise ValueError("Development calibration cannot bind an earlier calibration receipt.")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    config, prepared, snapshot, arrays = _prepared_problem(config_path)
    if not config.model.source_target_interaction_rank:
        raise ValueError("State-selection calibration requires an interaction pilot config.")
    selected_device = torch.device(device)
    if selected_device.type != config.training.pilot_device_type:
        raise ValueError("Calibration device differs from the frozen pilot device type.")
    rows: list[StateSelectionCalibrationRow] = []
    families: tuple[NullFamily, ...] = (
        "global_target_main",
        "conditional_interaction",
        "joint_nested",
    )
    if development_results is not None:
        used = {
            "permutation": {row.permutation_seed for row in development_results.rows},
            "optimizer": {row.optimizer_seed for row in development_results.rows},
            "initialization": {row.initialization_seed for row in development_results.rows},
        }
        proposed = {
            "permutation": {
                permutation_seed_start + family * 10_000_000 + index
                for family in range(3)
                for index in range(repeats_per_null)
            },
            "optimizer": {
                optimizer_seed_start + family * 10_000_000 + index
                for family in range(3)
                for index in range(repeats_per_null)
            },
            "initialization": {
                initialization_seed_start + family * 10_000_000 + index
                for family in range(3)
                for index in range(repeats_per_null)
            },
        }
        if any(used[name] & proposed[name] for name in used):
            raise ValueError("Locked audit seeds overlap the development calibration.")
    for family_index, family in enumerate(families):
        family_offset = family_index * 10_000_000
        for local_index in range(repeats_per_null):
            rows.append(
                _fit_one_null(
                    arrays,
                    config,
                    family=family,
                    replicate_index=len(rows),
                    permutation_seed=permutation_seed_start + family_offset + local_index,
                    optimizer_seed=optimizer_seed_start + family_offset + local_index,
                    initialization_seed=initialization_seed_start + family_offset + local_index,
                    device=selected_device,
                )
            )
    from ..compile.compiler import _problem_hash

    root = config_path.parent.resolve()
    guide_hash, support_hash = calibration_population_hashes(snapshot)
    calibration_id = _hash(
        {
            "config": config.model_dump(mode="json"),
            "prepared_id": prepared.prepared_id,
            "split_hash": sha256_file((root / config.split_contract).resolve()),
            "stage": calibration_stage,
            "rows": [
                {
                    "null_family": row.null_family,
                    "permutation_seed": row.permutation_seed,
                    "optimizer_seed": row.optimizer_seed,
                    "initialization_seed": row.initialization_seed,
                }
                for row in rows
            ],
        }
    )
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported calibration device: {selected_device.type}.")
    device_type = cast(Literal["cpu", "cuda"], selected_device.type)
    results = StateSelectionCalibrationResults(
        calibration_id=calibration_id,
        calibration_stage=calibration_stage,
        development_calibration_sha256=development_sha256,
        repeated_per_null=repeats_per_null,
        pooled_estimand=config.pooled_estimand,
        outer_fold_id=config.outer_fold_id,
        inner_split_id=config.inner_split_id,
        pooled_outer_fold_ids=config.pooled_outer_fold_ids,
        pooled_inner_split_ids=config.pooled_inner_split_ids,
        pooled_optimization_seeds=config.pooled_optimization_seeds,
        state_split_seed=config.training.state_split_seed,
        implementation_tree_hash=implementation_tree_hash(),
        calibration_code_hash=sha256_file(Path(__file__)),
        environment_lock_hash=environment_lock_hash(),
        optimizer_fingerprint=optimizer_fingerprint(config),
        device_type=device_type,
        dtype=config.training.dtype,
        deterministic_algorithms=config.training.deterministic,
        representation_id=prepared.prepared_id,
        split_manifest_hash=sha256_file((root / config.split_contract).resolve()),
        compiled_problem_hash=_problem_hash(arrays),
        interaction_rank=config.model.source_target_interaction_rank,
        interaction_scale=config.model.source_target_interaction_scale,
        learning_rate=config.training.learning_rate,
        state_batch_size=config.training.state_batch_size,
        state_full_batch=True,
        noninteraction_linear_ridge=config.training.noninteraction_linear_ridge,
        source_target_main_penalty=config.training.source_target_main_penalty,
        source_target_interaction_penalty=config.training.source_target_interaction_penalty,
        target_minimum_improvement=config.training.state_validation_target_minimum_improvement,
        interaction_minimum_improvement=(
            config.training.state_validation_interaction_minimum_improvement
        ),
        checkpoint_updates=config.training.state_checkpoint_updates,
        guide_per_target_distribution_hash=guide_hash,
        support_distribution_hash=support_hash,
        rows=tuple(rows),
    )
    results_path = output_root / "state-selection-calibration-results.json"
    atomic_json(results_path, results.model_dump(mode="json"))
    false_target = sum(row.false_target_main_selected for row in rows)
    false_interaction = sum(row.false_interaction_selected for row in rows)
    false_joint = sum(row.false_joint_interaction_selected for row in rows)
    if false_target or false_interaction or false_joint:
        raise RuntimeError(
            "Nested null calibration failed: "
            f"target={false_target}, interaction={false_interaction}, joint={false_joint}."
        )
    upper = 1.0 - 0.05 ** (1.0 / repeats_per_null)
    artifact = ArtifactRef(
        schema_id="credo.state_selection_calibration_results",
        schema_version=1,
        sha256=sha256_file(results_path),
        size_bytes=results_path.stat().st_size,
        media_type="application/json",
        relative_uri=results_path.name,
    )
    receipt = StateSelectionCalibration(
        calibration_id=calibration_id,
        calibration_stage=calibration_stage,
        development_calibration_sha256=development_sha256,
        repeated_per_null=repeats_per_null,
        false_target_main_count=0,
        false_interaction_count=0,
        false_joint_interaction_count=0,
        false_target_main_rate_upper_bound=upper,
        false_interaction_rate_upper_bound=upper,
        false_joint_interaction_rate_upper_bound=upper,
        target_minimum_improvement=config.training.state_validation_target_minimum_improvement,
        interaction_minimum_improvement=(
            config.training.state_validation_interaction_minimum_improvement
        ),
        checkpoint_updates=config.training.state_checkpoint_updates,
        results_artifact=artifact,
        calibration_protocol_hash=_hash(calibration_protocol_payload(results)),
    )
    receipt_path = output_root / "state-selection-calibration.json"
    atomic_json(receipt_path, receipt.model_dump(mode="json"))
    return results_path, receipt_path
