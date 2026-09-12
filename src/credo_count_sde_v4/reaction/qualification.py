"""T07S learned constant-reaction recovery on complete synthetic catalogs."""

from __future__ import annotations

import json
import platform
import shutil
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from ..canonical import canonical_json_bytes, contract_id, path_manifest, sha256_bytes, sha256_file
from ..contracts import (
    ComponentTestContract,
    ComponentTestReceiptV2,
    ModelConfig,
    ReactionRecoveryMetricAmendment,
    ReactionRecoveryQualificationBundle,
    ReactionRecoveryTestReceipt,
    ReactionRecoveryTestReceiptV1,
    ReactionRecoveryTestReceiptV3,
    RunIntent,
)
from ..errors import IntegrityError
from ..model import CountSDEModel
from ..numerics import rollout
from ..objectives import count_probabilities, exact_count_loss
from ..persistence import (
    artifact_ref,
    load_tensor_file,
    publish_directory,
    save_tensor_file,
    verify_directory,
)

_TEST_ID = "T07S_REACTION_RECOVERY"
_METHOD = "complete_denominator_dm_reaction_recovery_v2"
_AMENDMENT_METHOD = "t07s_interval_metric_amendment_v2"
_TARGET_EFFECTS = (
    0.0,
    -1.10,
    -0.90,
    -0.70,
    -0.50,
    -0.30,
    -0.10,
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
    1.10,
)
_CANDIDATE_UPDATES = (0, 25, 50, 100, 200)
_LEARNING_RATE = 0.05
_FIXED_CONCENTRATION_PARAMETER = 1000.0
_NULL_CALIBRATION_REPEATS = 59
_NULL_AUDIT_REPEATS = 60
_NULL_SEED_START = 20_260_823
_RECOVERY_SEED_START = 21_260_823
_BOOTSTRAP_SEED = 22_260_823
_BOOTSTRAP_DRAWS = 4000
_FALSE_PROMOTION_UPPER_LIMIT = 0.05
_MINIMUM_CHANNEL_ACTIVITY = 0.20
_REACTION_RMSE_LIMIT = 0.10
_GAUGE_TOLERANCE = 1e-12
_ROLLOUT_TOLERANCE = 1e-12


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _implementation_identity() -> tuple[str, dict[str, str]]:
    package = Path(__file__).resolve().parents[1]
    relative_paths = (
        "contracts/models.py",
        "model/count_sde.py",
        "numerics/particles.py",
        "objectives/counts.py",
        "persistence/artifacts.py",
        "reaction/qualification.py",
    )
    files = {relative: sha256_file(package / relative) for relative in relative_paths}
    return sha256_bytes(canonical_json_bytes(files)), files


def _environment_identity() -> tuple[str, dict[str, str]]:
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": metadata.version("pyarrow"),
        "scipy": metadata.version("scipy"),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "deterministic_algorithms": True,
    }
    return sha256_bytes(canonical_json_bytes(environment)), environment


def _new_model(pool_count: int) -> CountSDEModel:
    # Non-reaction tensors are initialized reproducibly, frozen, and protected.
    torch.manual_seed(20_260_823)
    model = CountSDEModel(
        ModelConfig(
            state_dim=1,
            target_count=len(_TARGET_EFFECTS),
            pool_count=pool_count,
            shared_diffusion=False,
            centered_selection=False,
        ),
        RunIntent.COUNT_MEASURE,
    ).to(dtype=torch.float64)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.target_fitness.requires_grad_(True)
    with torch.no_grad():
        model.target_fitness.zero_()
        model.pool_reference_fitness.zero_()
        model.log_concentration.fill_(_FIXED_CONCENTRATION_PARAMETER)
    return model


def _catalog(effects: tuple[float, ...], *, pools: int, seed: int, catalog: str) -> pd.DataFrame:
    """Generate one complete-denominator synthetic count catalog."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    series_index = 0
    for pool_index in range(pools):
        duration = (0.75, 1.0, 1.25)[pool_index % 3]
        pending: list[dict[str, Any]] = []
        for target_index, effect in enumerate(effects):
            replicas = 3 if target_index == 0 else 2
            for replicate in range(replicas):
                pending.append(
                    {
                        "catalog": catalog,
                        "pool_index": pool_index,
                        "series_index": series_index,
                        "target_index": target_index,
                        "guide_id": f"target{target_index:02d}_guide{replicate}",
                        "is_control": target_index == 0,
                        "duration": duration,
                        "source_count": int(rng.integers(500, 2001)),
                        "truth_raw_fitness": 0.0 if target_index == 0 else effect,
                    }
                )
                series_index += 1
        source = np.asarray([row["source_count"] for row in pending], dtype=np.float64)
        raw = np.asarray([row["truth_raw_fitness"] for row in pending], dtype=np.float64)
        logits = np.log(source + 0.5) + duration * raw
        probability = np.exp(logits - np.max(logits))
        probability /= probability.sum()
        terminal = rng.multinomial(60_000, probability)
        for row, terminal_count in zip(pending, terminal, strict=True):
            row["terminal_count"] = int(terminal_count)
            rows.append(row)
    return pd.DataFrame(rows)


def _concatenate_catalogs(*frames: pd.DataFrame) -> pd.DataFrame:
    combined: list[pd.DataFrame] = []
    pool_offset = 0
    series_offset = 0
    for frame in frames:
        copy = frame.copy()
        copy["pool_index"] = copy["pool_index"].astype(np.int64) + pool_offset
        copy["series_index"] = np.arange(series_offset, series_offset + len(copy))
        combined.append(copy)
        pool_offset += int(frame.pool_index.max()) + 1
        series_offset += len(copy)
    return pd.concat(combined, ignore_index=True)


def _tensors(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    ordered = frame.sort_values("series_index", kind="stable")
    return {
        "target": torch.tensor(ordered.target_index.to_numpy(), dtype=torch.int64),
        "pool": torch.tensor(ordered.pool_index.to_numpy(), dtype=torch.int64),
        "control": torch.tensor(ordered.is_control.to_numpy(), dtype=torch.bool),
        "duration": torch.tensor(ordered.duration.to_numpy(), dtype=torch.float64),
        "source": torch.tensor(ordered.source_count.to_numpy(), dtype=torch.int64),
        "terminal": torch.tensor(ordered.terminal_count.to_numpy(), dtype=torch.int64),
    }


def _count_loss(model: CountSDEModel, frame: pd.DataFrame) -> torch.Tensor:
    values = _tensors(frame)
    raw = model.raw_fitness(values["target"], values["pool"], values["control"])
    return exact_count_loss(
        values["terminal"],
        values["source"],
        raw,
        values["duration"],
        values["pool"],
        model.log_concentration,
    )


def _fit_and_select(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    model = _new_model(int(train.pool_index.max()) + 1)
    optimizer = torch.optim.Adam([model.target_fitness], lr=_LEARNING_RATE)
    candidates = set(_CANDIDATE_UPDATES)
    rows = [
        {
            "update": 0,
            "training_loss": float(_count_loss(model, train).detach()),
            "validation_loss": float(_count_loss(model, validation).detach()),
        }
    ]
    for update in range(1, max(_CANDIDATE_UPDATES) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _count_loss(model, train)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if update in candidates:
            rows.append(
                {
                    "update": update,
                    "training_loss": float(_count_loss(model, train).detach()),
                    "validation_loss": float(_count_loss(model, validation).detach()),
                }
            )
    curve = pd.DataFrame(rows)
    selected = min(
        ((float(row.validation_loss), int(row.update)) for row in curve.itertuples()),
        key=lambda value: (value[0], value[1]),
    )[1]
    curve["selected"] = curve["update"] == selected
    return selected, curve


def _fit_fixed_updates(frame: pd.DataFrame, updates: int) -> tuple[CountSDEModel, float]:
    model = _new_model(int(frame.pool_index.max()) + 1)
    protected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name != "target_fitness"
    }
    if updates:
        optimizer = torch.optim.Adam([model.target_fitness], lr=_LEARNING_RATE)
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = _count_loss(model, frame)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    changes = [
        float(torch.max(torch.abs(model.state_dict()[name] - original)))
        for name, original in protected.items()
        if original.numel()
    ]
    maximum = max(changes, default=0.0)
    return model, maximum


def _observed_relative_effect(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    ordered = frame.sort_values("series_index", kind="stable")
    result = np.empty(len(ordered), dtype=np.float64)
    for pool_index, positions in ordered.groupby("pool_index", sort=True).indices.items():
        del pool_index
        index = np.asarray(positions, dtype=np.int64)
        source = ordered.iloc[index].source_count.to_numpy(dtype=np.float64) + 0.5
        terminal = ordered.iloc[index].terminal_count.to_numpy(dtype=np.float64) + 0.5
        source_probability = source / source.sum()
        terminal_probability = terminal / terminal.sum()
        ratio = np.log(terminal_probability) - np.log(source_probability)
        result[index] = ratio - np.sum(source_probability * ratio)
    return result


def _recovery_rows(model: CountSDEModel, frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("series_index", kind="stable").reset_index(drop=True)
    values = _tensors(ordered)
    exposure = values["source"].to(torch.float64) + 0.5
    with torch.no_grad():
        predicted_rate = model.relative_fitness(
            values["target"], values["pool"], values["control"], exposure
        ).numpy()
    observed_interval_change = _observed_relative_effect(ordered)
    duration = ordered.duration.to_numpy(dtype=np.float64)
    predicted_interval_change = duration * predicted_rate
    result = ordered.copy()
    result["observed_centered_interval_log_frequency_change"] = observed_interval_change
    result["predicted_centered_interval_log_frequency_change"] = predicted_interval_change
    result["baseline_centered_interval_log_frequency_change"] = 0.0
    result["predicted_centered_relative_fitness_rate"] = predicted_rate
    result["new_squared_error"] = np.square(predicted_interval_change - observed_interval_change)
    result["baseline_squared_error"] = np.square(observed_interval_change)
    return result


def _target_metrics(series: pd.DataFrame) -> pd.DataFrame:
    targeting = series.loc[~series.is_control].copy()
    rows: list[dict[str, Any]] = []
    for target_index, group in targeting.groupby("target_index", sort=True):
        rows.append(
            {
                "target_index": int(target_index),
                "series_count": len(group),
                "new_mse": float(group.new_squared_error.mean()),
                "baseline_mse": float(group.baseline_squared_error.mean()),
                "truth_raw_fitness": float(group.truth_raw_fitness.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _losses(targets: pd.DataFrame) -> tuple[float, float, float]:
    new = float(np.sqrt(targets.new_mse.mean()))
    baseline = float(np.sqrt(targets.baseline_mse.mean()))
    return new, baseline, new - baseline


def _null_refit(repeat: int, *, partition: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = _NULL_SEED_START + repeat * 10
    zero = tuple(0.0 for _ in _TARGET_EFFECTS)
    train = _catalog(zero, pools=2, seed=seed, catalog="null_train")
    validation = _catalog(zero, pools=2, seed=seed + 1, catalog="null_validation")
    test = _catalog(zero, pools=3, seed=seed + 2, catalog="null_test")
    selected, _ = _fit_and_select(train, validation)
    model, _ = _fit_fixed_updates(_concatenate_catalogs(train, validation), selected)
    recovery = _recovery_rows(model, test)
    target = _target_metrics(recovery)
    new, baseline, delta = _losses(target)
    effects = [
        {
            "partition": partition,
            "repeat": repeat,
            "seed": seed,
            "selected_update": selected,
            "target_index": target_index,
            "fitted_raw_fitness": float(model.target_fitness[target_index].detach()),
        }
        for target_index in range(len(_TARGET_EFFECTS))
    ]
    return (
        {
            "partition": partition,
            "repeat": repeat,
            "seed": seed,
            "selected_update": selected,
            "new_target_balanced_rmse": new,
            "baseline_target_balanced_rmse": baseline,
            "delta": delta,
        },
        effects,
    )


def _false_promotion_upper(successes: int, trials: int) -> float:
    if successes == trials:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, trials - successes))


def _bootstrap(
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[float, float], list[float]]:
    target_ids = target.target_index.to_numpy(dtype=np.int64)
    lookup = target.set_index("target_index")
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    rows: list[dict[str, int]] = []
    deltas: list[float] = []
    for draw in range(_BOOTSTRAP_DRAWS):
        sampled = rng.choice(target_ids, size=len(target_ids), replace=True)
        for position, target_index in enumerate(sampled):
            rows.append(
                {
                    "draw": draw,
                    "position": position,
                    "target_index": int(target_index),
                }
            )
        metrics = lookup.loc[sampled]
        deltas.append(float(np.sqrt(metrics.new_mse.mean()) - np.sqrt(metrics.baseline_mse.mean())))
    interval = tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))
    return pd.DataFrame(rows), (interval[0], interval[1]), deltas


def _bootstrap_from_draws(
    target: pd.DataFrame, draws: pd.DataFrame
) -> tuple[tuple[float, float], list[float]]:
    """Recompute paired target-bootstrap deltas from persisted selections."""

    lookup = target.set_index("target_index")
    deltas: list[float] = []
    for _, group in draws.groupby("draw", sort=True):
        sampled = group.sort_values("position").target_index.to_numpy(dtype=np.int64)
        metrics = lookup.loc[sampled]
        deltas.append(float(np.sqrt(metrics.new_mse.mean()) - np.sqrt(metrics.baseline_mse.mean())))
    interval = tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))
    return (interval[0], interval[1]), deltas


def _protected_metrics(model: CountSDEModel, frame: pd.DataFrame) -> dict[str, float]:
    """Recompute fixed-channel, gauge, probability, and rollout invariants."""

    values = _tensors(frame)
    exposure = values["source"].to(torch.float64) + 0.5
    with torch.no_grad():
        relative = model.relative_fitness(
            values["target"], values["pool"], values["control"], exposure
        )
        gauge_errors: list[float] = []
        probability_errors: list[float] = []
        for pool_index in torch.unique(values["pool"]):
            mask = values["pool"] == pool_index
            weight = exposure[mask] / exposure[mask].sum()
            gauge_errors.append(float(torch.abs((weight * relative[mask]).sum())))
            raw = model.raw_fitness(
                values["target"][mask], values["pool"][mask], values["control"][mask]
            )
            probability = count_probabilities(values["source"][mask], raw, values["duration"][mask])
            probability_errors.append(float(torch.abs(probability.sum() - 1.0)))
        _, weights, mass = rollout(
            model,
            torch.zeros((len(frame), 1), dtype=torch.float64),
            values["duration"],
            values["target"],
            values["pool"],
            values["control"],
            exposure,
            particles=8,
            steps=64,
            seed=23_260_823,
        )
        expected_mass = torch.exp(values["duration"] * relative)
        rollout_error = float(
            torch.max(torch.abs(mass - expected_mass) / expected_mass.clamp_min(1e-15))
        )
        weight_error = float(torch.max(torch.abs(weights.sum(dim=1) - 1.0)))
    baseline = _new_model(model.config.pool_count)
    changes = [
        float(torch.max(torch.abs(value - baseline.state_dict()[name])))
        for name, value in model.state_dict().items()
        if name != "target_fitness" and value.numel()
    ]
    return {
        "gauge_error": max(gauge_errors),
        "probability_error": max(probability_errors),
        "rollout_error": rollout_error,
        "control_error": float(abs(model.target_fitness[0].detach())),
        "weight_error": weight_error,
        "fixed_change": max(changes, default=0.0),
    }


def _run_qualification() -> dict[str, Any]:
    null_rows: list[dict[str, Any]] = []
    null_effects: list[dict[str, Any]] = []
    for repeat in range(_NULL_CALIBRATION_REPEATS):
        row, effects = _null_refit(repeat, partition="calibration")
        null_rows.append(row)
        null_effects.extend(effects)
    for audit_repeat in range(_NULL_AUDIT_REPEATS):
        repeat = _NULL_CALIBRATION_REPEATS + audit_repeat
        row, effects = _null_refit(repeat, partition="audit")
        null_rows.append(row)
        null_effects.extend(effects)
    null = pd.DataFrame(null_rows)
    calibration = null.loc[null.partition == "calibration", "delta"].to_numpy()
    required_margin = max(0.0, -float(np.quantile(calibration, 0.05, method="lower")))
    audit = null.loc[null.partition == "audit", "delta"].to_numpy()
    false_promotions = int(np.sum(audit < -required_margin))
    false_upper = _false_promotion_upper(false_promotions, len(audit))
    calibration_nonzero_selections = int(
        (null.loc[null.partition == "calibration", "selected_update"] != 0).sum()
    )
    audit_nonzero_selections = int(
        (null.loc[null.partition == "audit", "selected_update"] != 0).sum()
    )

    train = _catalog(_TARGET_EFFECTS, pools=3, seed=_RECOVERY_SEED_START, catalog="recovery_train")
    validation = _catalog(
        _TARGET_EFFECTS,
        pools=2,
        seed=_RECOVERY_SEED_START + 1,
        catalog="recovery_validation",
    )
    test = _catalog(
        _TARGET_EFFECTS, pools=3, seed=_RECOVERY_SEED_START + 2, catalog="recovery_test"
    )
    selected_update, curve = _fit_and_select(train, validation)
    model, fixed_change = _fit_fixed_updates(
        _concatenate_catalogs(train, validation), selected_update
    )
    recovery = _recovery_rows(model, test)
    target = _target_metrics(recovery)
    new_loss, baseline_loss, point_delta = _losses(target)
    bootstrap_draws, interval, bootstrap_deltas = _bootstrap(target)

    fitted = model.target_fitness.detach().numpy()
    truth = np.asarray(_TARGET_EFFECTS)
    reaction_rmse = float(np.sqrt(np.mean(np.square(fitted[1:] - truth[1:]))))
    sign_accuracy = float(np.mean(np.sign(fitted[1:]) == np.sign(truth[1:])))
    channel_activity = float(np.sqrt(np.mean(np.square(fitted[1:]))))
    protected = _protected_metrics(model, test)

    return {
        "null": null,
        "null_effects": pd.DataFrame(null_effects),
        "required_margin": required_margin,
        "false_promotions": false_promotions,
        "false_promotion_upper": false_upper,
        "calibration_nonzero_selections": calibration_nonzero_selections,
        "audit_nonzero_selections": audit_nonzero_selections,
        "curve": curve,
        "selected_update": selected_update,
        "model": model,
        "fixed_change": max(fixed_change, protected["fixed_change"]),
        "recovery": recovery,
        "target": target,
        "new_loss": new_loss,
        "baseline_loss": baseline_loss,
        "point_delta": point_delta,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_interval": interval,
        "bootstrap_deltas": bootstrap_deltas,
        "reaction_rmse": reaction_rmse,
        "sign_accuracy": sign_accuracy,
        "channel_activity": channel_activity,
        "gauge_error": protected["gauge_error"],
        "probability_error": protected["probability_error"],
        "rollout_error": protected["rollout_error"],
        "control_error": protected["control_error"],
        "weight_error": protected["weight_error"],
    }


def _config(environment: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": _METHOD,
        "device": "cpu",
        "dtype": "float64",
        "target_effects": list(_TARGET_EFFECTS),
        "candidate_updates": list(_CANDIDATE_UPDATES),
        "learning_rate": _LEARNING_RATE,
        "fixed_concentration_parameter": _FIXED_CONCENTRATION_PARAMETER,
        "null_calibration_repeats": _NULL_CALIBRATION_REPEATS,
        "null_audit_repeats": _NULL_AUDIT_REPEATS,
        "null_seed_start": _NULL_SEED_START,
        "recovery_seed_start": _RECOVERY_SEED_START,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "bootstrap_draws": _BOOTSTRAP_DRAWS,
        "false_promotion_upper_limit": _FALSE_PROMOTION_UPPER_LIMIT,
        "primary_metric_estimand": "centered_interval_log_frequency_change",
        "predicted_interval_effect": "duration_times_centered_relative_fitness_rate",
        "minimum_channel_activity": _MINIMUM_CHANNEL_ACTIVITY,
        "reaction_rmse_limit": _REACTION_RMSE_LIMIT,
        "gauge_tolerance": _GAUGE_TOLERANCE,
        "rollout_tolerance": _ROLLOUT_TOLERANCE,
        "source_smoothing": 0.5,
        "train_pools": 3,
        "validation_pools": 2,
        "test_pools": 3,
        "selected_model_pool_count": 5,
        "terminal_count_per_pool": 60_000,
        "controls_per_pool": 3,
        "guides_per_target_per_pool": 2,
        "environment": environment,
    }


def qualify_reaction_recovery(destination: Path) -> Path:
    """Run and atomically publish the T07S R0/R1 CPU qualification."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        result = _run_qualification()
    finally:
        torch.use_deterministic_algorithms(previous)
    environment_hash, environment = _environment_identity()
    config = _config(environment)
    config_hash = sha256_bytes(canonical_json_bytes(config))
    implementation_hash, implementation_files = _implementation_identity()
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": _TEST_ID,
        "component": "trainable_constant_relative_fitness",
        "primary_metric": "interval_effect_target_balanced_rmse",
        "primary_baseline": "zero_reaction",
        "required_margin": result["required_margin"],
        "drift": "fixed",
        "diffusion": "fixed",
        "reaction": "trainable",
        "ecology": "off",
        "decoder": "off",
        "update_zero_selectable": True,
        "post_selection_refit_required": True,
    }
    contract_payload["test_contract_id"] = contract_id(
        contract_payload, id_field="test_contract_id"
    )
    contract = ComponentTestContract.model_validate(contract_payload)
    r0_pass = result["false_promotion_upper"] <= _FALSE_PROMOTION_UPPER_LIMIT
    r1_margin_pass = result["bootstrap_interval"][1] < -result["required_margin"]
    channel_pass = (
        result["channel_activity"] >= _MINIMUM_CHANNEL_ACTIVITY
        and result["reaction_rmse"] <= _REACTION_RMSE_LIMIT
        and result["sign_accuracy"] == 1.0
    )
    protected = (
        result["fixed_change"] == 0.0
        and result["gauge_error"] <= _GAUGE_TOLERANCE
        and result["probability_error"] <= _GAUGE_TOLERANCE
        and result["rollout_error"] <= _ROLLOUT_TOLERANCE
        and result["control_error"] == 0.0
        and result["weight_error"] <= _GAUGE_TOLERANCE
    )
    status = "pass" if r0_pass and r1_margin_pass and channel_pass and protected else "fail_retired"

    def writer(temp: Path) -> None:
        result["null"].to_parquet(temp / "NULL_REFITS.parquet", index=False)
        result["null_effects"].to_parquet(temp / "NULL_MODEL_EFFECTS.parquet", index=False)
        result["curve"].to_parquet(temp / "RECOVERY_CURVE.parquet", index=False)
        result["recovery"].to_parquet(temp / "RECOVERY_SERIES.parquet", index=False)
        result["target"].to_parquet(temp / "TARGET_METRICS.parquet", index=False)
        result["bootstrap_draws"].to_parquet(temp / "BOOTSTRAP_TARGET_DRAWS.parquet", index=False)
        save_tensor_file(temp / "SELECTED_MODEL.safetensors", result["model"].state_dict())
        _write_json(temp / "CONFIG.json", config)
        _write_json(temp / "TEST_CONTRACT.json", contract.model_dump(mode="json"))
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
                "calibration_repeats": _NULL_CALIBRATION_REPEATS,
                "audit_repeats": _NULL_AUDIT_REPEATS,
                "required_margin": result["required_margin"],
                "audit_false_promotions": result["false_promotions"],
                "audit_false_promotion_upper_95": result["false_promotion_upper"],
                "calibration_nonzero_checkpoint_selections": result[
                    "calibration_nonzero_selections"
                ],
                "audit_nonzero_checkpoint_selections": result["audit_nonzero_selections"],
                "upper_limit": _FALSE_PROMOTION_UPPER_LIMIT,
            },
        )
        _write_json(
            temp / "BOOTSTRAP_RESULTS.json",
            {
                "schema_version": 1,
                "unit": "synthetic_target_centered_interval_log_frequency_change",
                "draws": _BOOTSTRAP_DRAWS,
                "seed": _BOOTSTRAP_SEED,
                "point_delta": result["point_delta"],
                "interval": result["bootstrap_interval"],
            },
        )
        _write_json(
            temp / "CHANNEL_ACTIVITY.json",
            {
                "schema_version": 1,
                "reaction_rms": result["channel_activity"],
                "minimum": _MINIMUM_CHANNEL_ACTIVITY,
                "reaction_rmse": result["reaction_rmse"],
                "sign_accuracy": result["sign_accuracy"],
                "fixed_channel_max_abs_change": result["fixed_change"],
            },
        )
        _write_json(
            temp / "MODEL_CARD.json",
            {
                "schema_version": 1,
                "family": "constant_target_relative_fitness",
                "intent": "count_measure",
                "trained_parameters": ["target_fitness"],
                "fixed_parameters": [
                    "drift",
                    "diffusion",
                    "selection",
                    "pool_reference_fitness",
                    "concentration",
                ],
                "controls": "exact-zero target reaction before pool centering",
                "post_selection_refit": True,
                "biological_claims": False,
            },
        )
        receipt_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "parent_qualification_id": None,
            "status": status,
            "r0_calibration_repeats": _NULL_CALIBRATION_REPEATS,
            "r0_audit_repeats": _NULL_AUDIT_REPEATS,
            "r0_required_margin": result["required_margin"],
            "r0_calibration_nonzero_checkpoint_selections": result[
                "calibration_nonzero_selections"
            ],
            "r0_audit_nonzero_checkpoint_selections": result["audit_nonzero_selections"],
            "r0_audit_nonzero_checkpoint_selection_rate": result["audit_nonzero_selections"]
            / _NULL_AUDIT_REPEATS,
            "r0_audit_false_promotions": result["false_promotions"],
            "r0_audit_false_promotion_upper_95": result["false_promotion_upper"],
            "r0_false_promotion_guard_pass": r0_pass,
            "r1_metric_estimand": "centered_interval_log_frequency_change",
            "r1_selected_update": result["selected_update"],
            "r1_post_selection_refit_pass": True,
            "r1_interval_effect_target_balanced_rmse": result["new_loss"],
            "r1_zero_baseline_interval_effect_target_balanced_rmse": result["baseline_loss"],
            "r1_point_delta": result["point_delta"],
            "r1_target_bootstrap_interval": result["bootstrap_interval"],
            "r1_margin_pass": r1_margin_pass,
            "r1_reaction_rmse": result["reaction_rmse"],
            "r1_sign_accuracy": result["sign_accuracy"],
            "r1_channel_activity": result["channel_activity"],
            "r1_channel_activity_pass": channel_pass,
            "weighted_gauge_max_abs_error": result["gauge_error"],
            "rollout_mass_max_relative_error": result["rollout_error"],
            "probability_normalization_max_abs_error": result["probability_error"],
            "control_target_mask_max_abs_error": result["control_error"],
            "fixed_channel_max_abs_change": result["fixed_change"],
            "protected_metrics_pass": protected,
            "update_zero_selectable": True,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = ReactionRecoveryTestReceipt.model_validate(receipt_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        component_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_id": _TEST_ID,
            "receipt_role": "model_comparison",
            "status": status,
            "primary_metric": contract.primary_metric,
            "primary_baseline": contract.primary_baseline,
            "point_delta": result["point_delta"],
            "bootstrap_interval": result["bootstrap_interval"],
            "required_margin": result["required_margin"],
            "channel_activity": result["channel_activity"],
            "estimand": None,
            "quantile_probability": None,
            "quantile_value": None,
            "repeat_count": None,
            "sampling_method": None,
            "protected_metrics_pass": protected,
            "selected_update": result["selected_update"],
            "input_hashes": {"synthetic_protocol": config_hash, "environment": environment_hash},
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
        }
        component_payload["receipt_id"] = contract_id(component_payload, id_field="receipt_id")
        component = ComponentTestReceiptV2.model_validate(component_payload)
        _write_json(temp / "COMPONENT_RECEIPT.json", component.model_dump(mode="json"))
        refs = {
            "null_refits": (
                "NULL_REFITS.parquet",
                "credo.t07s_null_refits",
                "application/x-parquet",
            ),
            "null_model_effects": (
                "NULL_MODEL_EFFECTS.parquet",
                "credo.t07s_null_effects",
                "application/x-parquet",
            ),
            "recovery_curve": (
                "RECOVERY_CURVE.parquet",
                "credo.t07s_curve",
                "application/x-parquet",
            ),
            "recovery_series": (
                "RECOVERY_SERIES.parquet",
                "credo.t07s_series",
                "application/x-parquet",
            ),
            "target_metrics": (
                "TARGET_METRICS.parquet",
                "credo.t07s_target_metrics",
                "application/x-parquet",
            ),
            "bootstrap_target_draws": (
                "BOOTSTRAP_TARGET_DRAWS.parquet",
                "credo.t07s_bootstrap_draws",
                "application/x-parquet",
            ),
            "selected_model": (
                "SELECTED_MODEL.safetensors",
                "credo.t07s_model",
                "application/x-safetensors",
            ),
            "test_receipt": ("TEST_RECEIPT.json", "credo.t07s_test_receipt", "application/json"),
        }
        bundle_payload: dict[str, Any] = {
            "schema_version": 1,
            "qualification_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "method": _METHOD,
            "environment_hash": environment_hash,
            "null_calibration_repeats": _NULL_CALIBRATION_REPEATS,
            "null_audit_repeats": _NULL_AUDIT_REPEATS,
            "candidate_updates": _CANDIDATE_UPDATES,
            "target_count": len(_TARGET_EFFECTS),
            "pool_count": 5,
        }
        for field, (name, schema_id, media_type) in refs.items():
            bundle_payload[field] = artifact_ref(
                temp, temp / name, schema_id=schema_id, media_type=media_type
            ).model_dump(mode="json")
        bundle_payload["qualification_id"] = contract_id(
            bundle_payload, id_field="qualification_id"
        )
        bundle = ReactionRecoveryQualificationBundle.model_validate(bundle_payload)
        _write_json(temp / "reaction-recovery.json", bundle.model_dump(mode="json"))
        _write_json(
            temp / "QUALIFICATION_LINK.json",
            {
                "schema_version": 1,
                "qualification_id": bundle.qualification_id,
                "receipt_id": receipt.receipt_id,
            },
        )
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_reaction_recovery_qualification(destination)
    return destination


_DEV23_METRIC_COLUMNS = (
    "observed_relative_fitness",
    "predicted_relative_fitness",
    "baseline_relative_fitness",
    "new_squared_error",
    "baseline_squared_error",
)
_DEV24_METRIC_COLUMNS = (
    "observed_centered_interval_log_frequency_change",
    "predicted_centered_interval_log_frequency_change",
    "baseline_centered_interval_log_frequency_change",
    "predicted_centered_relative_fitness_rate",
    "new_squared_error",
    "baseline_squared_error",
)


def _verified_dev23_parent(
    parent: Path,
) -> tuple[ReactionRecoveryQualificationBundle, ReactionRecoveryTestReceiptV1]:
    """Verify and bind the immutable dev23 parent without invoking its old evaluator."""

    verify_directory(parent)
    bundle = ReactionRecoveryQualificationBundle.model_validate_json(
        (parent / "reaction-recovery.json").read_text()
    )
    receipt = ReactionRecoveryTestReceiptV1.model_validate_json(
        (parent / "TEST_RECEIPT.json").read_text()
    )
    if bundle.method != "complete_denominator_dm_reaction_recovery_v1":
        raise IntegrityError("The metric amendment requires an immutable dev23 T07S-v1 parent.")
    if receipt.test_contract_id != bundle.test_contract_id or receipt.status != "pass":
        raise IntegrityError("The dev23 T07S parent receipt does not bind a passing parent bundle.")
    link = json.loads((parent / "QUALIFICATION_LINK.json").read_text())
    if link != {
        "schema_version": 1,
        "qualification_id": bundle.qualification_id,
        "receipt_id": receipt.receipt_id,
    }:
        raise IntegrityError("The dev23 T07S parent link is inconsistent.")
    for reference in (
        bundle.null_refits,
        bundle.null_model_effects,
        bundle.recovery_curve,
        bundle.recovery_series,
        bundle.target_metrics,
        bundle.bootstrap_target_draws,
        bundle.selected_model,
        bundle.test_receipt,
    ):
        artifact = parent / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"The dev23 parent artifact differs from its reference: {reference.relative_uri}."
            )
    return bundle, receipt


def _amended_result(parent: Path) -> dict[str, Any]:
    bundle, parent_receipt = _verified_dev23_parent(parent)
    null = pd.read_parquet(parent / bundle.null_refits.relative_uri)
    effects = pd.read_parquet(parent / bundle.null_model_effects.relative_uri)
    curve = pd.read_parquet(parent / bundle.recovery_curve.relative_uri)
    draws = pd.read_parquet(parent / bundle.bootstrap_target_draws.relative_uri)
    legacy_series = pd.read_parquet(parent / bundle.recovery_series.relative_uri)
    missing = set(_DEV23_METRIC_COLUMNS) - set(legacy_series.columns)
    if missing:
        raise IntegrityError(f"The dev23 recovery series lacks legacy metric columns: {missing}.")
    raw_series = legacy_series.drop(columns=list(_DEV23_METRIC_COLUMNS))
    model = _new_model(bundle.pool_count)
    model.load_state_dict(
        load_tensor_file(parent / bundle.selected_model.relative_uri), strict=True
    )
    recovery = _recovery_rows(model, raw_series)
    target = _target_metrics(recovery)
    new_loss, baseline_loss, point_delta = _losses(target)
    interval, bootstrap_deltas = _bootstrap_from_draws(target, draws)

    corrected_null = _derive_interval_null_refits(null, effects)
    calibration = corrected_null.loc[
        corrected_null.partition == "calibration", "interval_effect_rmse_delta"
    ].to_numpy()
    margin = max(0.0, -float(np.quantile(calibration, 0.05, method="lower")))
    audit = corrected_null.loc[
        corrected_null.partition == "audit", "interval_effect_rmse_delta"
    ].to_numpy()
    false_promotions = int(np.sum(audit < -margin))
    false_upper = _false_promotion_upper(false_promotions, len(audit))
    calibration_nonzero = int(
        (null.loc[null.partition == "calibration", "selected_update"] != 0).sum()
    )
    audit_nonzero = int((null.loc[null.partition == "audit", "selected_update"] != 0).sum())
    selected_update = int(curve.loc[curve.selected, "update"].iloc[0])
    if selected_update != parent_receipt.r1_selected_update:
        raise IntegrityError("The dev23 selected update differs from its immutable receipt.")

    fitted = model.target_fitness.detach().numpy()
    truth = np.asarray(_TARGET_EFFECTS)
    reaction_rmse = float(np.sqrt(np.mean(np.square(fitted[1:] - truth[1:]))))
    sign_accuracy = float(np.mean(np.sign(fitted[1:]) == np.sign(truth[1:])))
    channel_activity = float(np.sqrt(np.mean(np.square(fitted[1:]))))
    protected = _protected_metrics(model, raw_series)
    return {
        "parent_bundle": bundle,
        "parent_receipt": parent_receipt,
        "null": null,
        "corrected_null": corrected_null,
        "null_effects": effects,
        "curve": curve,
        "draws": draws,
        "model": model,
        "raw_series": raw_series,
        "recovery": recovery,
        "target": target,
        "new_loss": new_loss,
        "baseline_loss": baseline_loss,
        "point_delta": point_delta,
        "bootstrap_interval": interval,
        "bootstrap_deltas": bootstrap_deltas,
        "required_margin": margin,
        "false_promotions": false_promotions,
        "false_promotion_upper": false_upper,
        "calibration_nonzero_selections": calibration_nonzero,
        "audit_nonzero_selections": audit_nonzero,
        "selected_update": selected_update,
        "reaction_rmse": reaction_rmse,
        "sign_accuracy": sign_accuracy,
        "channel_activity": channel_activity,
        "gauge_error": protected["gauge_error"],
        "probability_error": protected["probability_error"],
        "rollout_error": protected["rollout_error"],
        "control_error": protected["control_error"],
        "weight_error": protected["weight_error"],
        "fixed_change": protected["fixed_change"],
    }


def _derive_interval_null_refits(legacy_null: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """Re-evaluate persisted dev23 null coefficients in interval-effect units.

    This is intentionally evaluation-only: the selected update and every fitted
    coefficient come from the immutable parent. No optimizer or selection code
    is invoked.
    """

    required_null = {"partition", "repeat", "seed", "selected_update"}
    required_effects = required_null | {"target_index", "fitted_raw_fitness"}
    if missing := required_null - set(legacy_null.columns):
        raise IntegrityError(f"The dev23 null table lacks required columns: {missing}.")
    if missing := required_effects - set(effects.columns):
        raise IntegrityError(f"The dev23 null-effect table lacks required columns: {missing}.")
    if legacy_null.duplicated(["partition", "repeat"]).any():
        raise IntegrityError("The dev23 null table contains duplicate repeat identities.")

    rows: list[dict[str, Any]] = []
    zero = tuple(0.0 for _ in _TARGET_EFFECTS)
    grouped = effects.groupby(["partition", "repeat"], sort=False)
    for summary in legacy_null.sort_values(["partition", "repeat"], kind="stable").itertuples():
        key = (str(summary.partition), int(summary.repeat))
        try:
            fitted = grouped.get_group(key).sort_values("target_index", kind="stable")
        except KeyError as exc:
            raise IntegrityError(f"The dev23 null coefficients lack repeat {key}.") from exc
        if tuple(fitted.target_index.astype(int)) != tuple(range(len(_TARGET_EFFECTS))):
            raise IntegrityError(f"The dev23 null coefficients are incomplete for repeat {key}.")
        if not (
            fitted.seed.astype(int).eq(int(summary.seed)).all()
            and fitted.selected_update.astype(int).eq(int(summary.selected_update)).all()
        ):
            raise IntegrityError(f"The dev23 null coefficient metadata differs for repeat {key}.")

        model = _new_model(pool_count=3)
        with torch.no_grad():
            model.target_fitness.copy_(
                torch.tensor(fitted.fitted_raw_fitness.to_numpy(), dtype=torch.float64)
            )
        test = _catalog(
            zero,
            pools=3,
            seed=int(summary.seed) + 2,
            catalog="null_test",
        )
        recovery = _recovery_rows(model, test)
        legacy_recovery = recovery.copy()
        legacy_recovery["new_squared_error"] = np.square(
            legacy_recovery.predicted_centered_relative_fitness_rate
            - legacy_recovery.observed_centered_interval_log_frequency_change
        )
        legacy_recovery["baseline_squared_error"] = np.square(
            legacy_recovery.observed_centered_interval_log_frequency_change
        )
        legacy_target = _target_metrics(legacy_recovery)
        _, _, legacy_delta = _losses(legacy_target)
        if abs(legacy_delta - float(summary.delta)) > 1e-12:
            raise IntegrityError(
                "The recreated null catalog does not reproduce the parent legacy metric "
                f"for repeat {key}."
            )
        target = _target_metrics(recovery)
        new, baseline, delta = _losses(target)
        rows.append(
            {
                "partition": str(summary.partition),
                "repeat": int(summary.repeat),
                "seed": int(summary.seed),
                "selected_update": int(summary.selected_update),
                "interval_effect_target_balanced_rmse": new,
                "zero_reaction_interval_effect_target_balanced_rmse": baseline,
                "interval_effect_rmse_delta": delta,
                "metric_estimand": "centered_interval_log_frequency_change",
            }
        )
    return pd.DataFrame(rows)


def amend_reaction_recovery_metrics(destination: Path, *, parent: Path) -> Path:
    """Publish the unified duration-correct dev25 authority without retraining."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    result = _amended_result(parent)
    parent_bundle = result["parent_bundle"]
    environment_hash, environment = _environment_identity()
    implementation_hash, implementation_files = _implementation_identity()
    parent_bundle_sha = sha256_file(parent / "reaction-recovery.json")
    parent_manifest_sha = sha256_file(parent / "artifacts.json")
    config = {
        "schema_version": 1,
        "method": _AMENDMENT_METHOD,
        "device": "cpu",
        "dtype": "float64",
        "parent_qualification_id": parent_bundle.qualification_id,
        "parent_bundle_sha256": parent_bundle_sha,
        "parent_artifacts_manifest_sha256": parent_manifest_sha,
        "selected_model_reused_without_optimizer": True,
        "primary_metric_estimand": "centered_interval_log_frequency_change",
        "r0_metric_estimand": "centered_interval_log_frequency_change",
        "r0_derivation": "persisted_coefficients_replayed_on_seed_plus_2_null_catalog",
        "predicted_interval_effect": "duration_times_centered_relative_fitness_rate",
        "false_promotion_upper_limit": _FALSE_PROMOTION_UPPER_LIMIT,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "bootstrap_draws": _BOOTSTRAP_DRAWS,
        "environment": environment,
    }
    config_hash = sha256_bytes(canonical_json_bytes(config))
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": _TEST_ID,
        "component": "constant_target_reaction_interval_metric_amendment",
        "primary_metric": "interval_effect_target_balanced_rmse",
        "primary_baseline": "zero_reaction",
        "required_margin": result["required_margin"],
        "drift": "fixed",
        "diffusion": "fixed",
        "reaction": "trainable",
        "ecology": "off",
        "decoder": "off",
        "update_zero_selectable": True,
        "post_selection_refit_required": True,
    }
    contract_payload["test_contract_id"] = contract_id(
        contract_payload, id_field="test_contract_id"
    )
    contract = ComponentTestContract.model_validate(contract_payload)
    r0_pass = result["false_promotion_upper"] <= _FALSE_PROMOTION_UPPER_LIMIT
    r1_margin_pass = result["bootstrap_interval"][1] < -result["required_margin"]
    channel_pass = (
        result["channel_activity"] >= _MINIMUM_CHANNEL_ACTIVITY
        and result["reaction_rmse"] <= _REACTION_RMSE_LIMIT
        and result["sign_accuracy"] == 1.0
    )
    protected_pass = (
        result["fixed_change"] == 0.0
        and result["gauge_error"] <= _GAUGE_TOLERANCE
        and result["probability_error"] <= _GAUGE_TOLERANCE
        and result["rollout_error"] <= _ROLLOUT_TOLERANCE
        and result["control_error"] == 0.0
        and result["weight_error"] <= _GAUGE_TOLERANCE
    )
    status = (
        "pass" if r0_pass and r1_margin_pass and channel_pass and protected_pass else "fail_retired"
    )

    def writer(temp: Path) -> None:
        copy_names = (
            "NULL_REFITS.parquet",
            "NULL_MODEL_EFFECTS.parquet",
            "RECOVERY_CURVE.parquet",
            "BOOTSTRAP_TARGET_DRAWS.parquet",
            "SELECTED_MODEL.safetensors",
        )
        for name in copy_names:
            shutil.copyfile(parent / name, temp / name)
        result["corrected_null"].to_parquet(temp / "NULL_INTERVAL_REFITS.parquet", index=False)
        result["recovery"].to_parquet(temp / "RECOVERY_SERIES.parquet", index=False)
        result["target"].to_parquet(temp / "TARGET_METRICS.parquet", index=False)
        _write_json(temp / "CONFIG.json", config)
        _write_json(temp / "TEST_CONTRACT.json", contract.model_dump(mode="json"))
        _write_json(
            temp / "PARENT_LINK.json",
            {
                "schema_version": 2,
                "parent_qualification_id": parent_bundle.qualification_id,
                "parent_bundle_sha256": parent_bundle_sha,
                "parent_artifacts_manifest_sha256": parent_manifest_sha,
                "selected_model_sha256": sha256_file(parent / "SELECTED_MODEL.safetensors"),
                "legacy_null_refits_sha256": sha256_file(temp / "NULL_REFITS.parquet"),
                "corrected_null_interval_refits_sha256": sha256_file(
                    temp / "NULL_INTERVAL_REFITS.parquet"
                ),
                "optimizer_rerun": False,
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
                "schema_version": 3,
                "metric_estimand": "centered_interval_log_frequency_change",
                "calibration_repeats": _NULL_CALIBRATION_REPEATS,
                "audit_repeats": _NULL_AUDIT_REPEATS,
                "required_margin": result["required_margin"],
                "calibration_nonzero_checkpoint_selections": result[
                    "calibration_nonzero_selections"
                ],
                "audit_nonzero_checkpoint_selections": result["audit_nonzero_selections"],
                "audit_false_promotions": result["false_promotions"],
                "audit_false_promotion_upper_95": result["false_promotion_upper"],
                "upper_limit": _FALSE_PROMOTION_UPPER_LIMIT,
                "legacy_parent_null_rows_preserved": True,
                "corrected_null_interval_rows_derived": True,
                "optimizer_rerun": False,
            },
        )
        _write_json(
            temp / "BOOTSTRAP_RESULTS.json",
            {
                "schema_version": 2,
                "unit": "synthetic_target_centered_interval_log_frequency_change",
                "draws": _BOOTSTRAP_DRAWS,
                "seed": _BOOTSTRAP_SEED,
                "new_target_balanced_rmse": result["new_loss"],
                "zero_baseline_target_balanced_rmse": result["baseline_loss"],
                "point_delta": result["point_delta"],
                "interval": result["bootstrap_interval"],
            },
        )
        _write_json(
            temp / "CHANNEL_ACTIVITY.json",
            {
                "schema_version": 1,
                "reaction_rms": result["channel_activity"],
                "minimum": _MINIMUM_CHANNEL_ACTIVITY,
                "reaction_rmse": result["reaction_rmse"],
                "sign_accuracy": result["sign_accuracy"],
                "fixed_channel_max_abs_change": result["fixed_change"],
            },
        )
        _write_json(
            temp / "MODEL_CARD.json",
            {
                "schema_version": 2,
                "family": "constant_target_average_relative_reaction",
                "intent": "count_measure",
                "trained_parameters": ["target_fitness"],
                "metric_estimand": "centered_interval_log_frequency_change",
                "state_dependent_centered_reaction": "not_qualified",
                "selected_model_reused_without_optimizer": True,
                "biological_claims": False,
            },
        )
        legacy_null_ref = artifact_ref(
            temp,
            temp / "NULL_REFITS.parquet",
            schema_id="credo.t07s_legacy_null_refits",
            media_type="application/x-parquet",
        )
        corrected_null_ref = artifact_ref(
            temp,
            temp / "NULL_INTERVAL_REFITS.parquet",
            schema_id="credo.t07s_interval_null_refits",
            media_type="application/x-parquet",
        )
        receipt_payload = {
            "schema_version": 3,
            "receipt_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "parent_qualification_id": parent_bundle.qualification_id,
            "status": status,
            "r0_calibration_repeats": _NULL_CALIBRATION_REPEATS,
            "r0_audit_repeats": _NULL_AUDIT_REPEATS,
            "r0_required_margin": result["required_margin"],
            "r0_calibration_nonzero_checkpoint_selections": result[
                "calibration_nonzero_selections"
            ],
            "r0_audit_nonzero_checkpoint_selections": result["audit_nonzero_selections"],
            "r0_audit_nonzero_checkpoint_selection_rate": result["audit_nonzero_selections"]
            / _NULL_AUDIT_REPEATS,
            "r0_audit_false_promotions": result["false_promotions"],
            "r0_audit_false_promotion_upper_95": result["false_promotion_upper"],
            "r0_false_promotion_guard_pass": r0_pass,
            "r0_metric_estimand": "centered_interval_log_frequency_change",
            "legacy_null_refits": legacy_null_ref.model_dump(mode="json"),
            "corrected_null_interval_refits": corrected_null_ref.model_dump(mode="json"),
            "optimizer_rerun": False,
            "r1_metric_estimand": "centered_interval_log_frequency_change",
            "r1_selected_update": result["selected_update"],
            "r1_post_selection_refit_pass": True,
            "r1_interval_effect_target_balanced_rmse": result["new_loss"],
            "r1_zero_baseline_interval_effect_target_balanced_rmse": result["baseline_loss"],
            "r1_point_delta": result["point_delta"],
            "r1_target_bootstrap_interval": result["bootstrap_interval"],
            "r1_margin_pass": r1_margin_pass,
            "r1_reaction_rmse": result["reaction_rmse"],
            "r1_sign_accuracy": result["sign_accuracy"],
            "r1_channel_activity": result["channel_activity"],
            "r1_channel_activity_pass": channel_pass,
            "weighted_gauge_max_abs_error": result["gauge_error"],
            "rollout_mass_max_relative_error": result["rollout_error"],
            "probability_normalization_max_abs_error": result["probability_error"],
            "control_target_mask_max_abs_error": result["control_error"],
            "fixed_channel_max_abs_change": result["fixed_change"],
            "protected_metrics_pass": protected_pass,
            "update_zero_selectable": True,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = ReactionRecoveryTestReceiptV3.model_validate(receipt_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        component_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_id": _TEST_ID,
            "receipt_role": "model_comparison",
            "status": status,
            "primary_metric": contract.primary_metric,
            "primary_baseline": contract.primary_baseline,
            "point_delta": result["point_delta"],
            "bootstrap_interval": result["bootstrap_interval"],
            "required_margin": result["required_margin"],
            "channel_activity": result["channel_activity"],
            "estimand": None,
            "quantile_probability": None,
            "quantile_value": None,
            "repeat_count": None,
            "sampling_method": None,
            "protected_metrics_pass": protected_pass,
            "selected_update": result["selected_update"],
            "input_hashes": {
                "parent_qualification": parent_bundle_sha,
                "parent_artifacts_manifest": parent_manifest_sha,
                "legacy_null_refits": legacy_null_ref.sha256,
                "corrected_null_interval_refits": corrected_null_ref.sha256,
                "environment": environment_hash,
            },
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
        }
        component_payload["receipt_id"] = contract_id(component_payload, id_field="receipt_id")
        component = ComponentTestReceiptV2.model_validate(component_payload)
        _write_json(temp / "COMPONENT_RECEIPT.json", component.model_dump(mode="json"))

        refs = {
            "legacy_null_refits": (
                "NULL_REFITS.parquet",
                "credo.t07s_legacy_null_refits",
            ),
            "corrected_null_interval_refits": (
                "NULL_INTERVAL_REFITS.parquet",
                "credo.t07s_interval_null_refits",
            ),
            "null_model_effects": ("NULL_MODEL_EFFECTS.parquet", "credo.t07s_null_effects"),
            "recovery_curve": ("RECOVERY_CURVE.parquet", "credo.t07s_curve"),
            "recovery_series": ("RECOVERY_SERIES.parquet", "credo.t07s_interval_series"),
            "target_metrics": ("TARGET_METRICS.parquet", "credo.t07s_interval_targets"),
            "bootstrap_target_draws": (
                "BOOTSTRAP_TARGET_DRAWS.parquet",
                "credo.t07s_bootstrap_draws",
            ),
            "selected_model": ("SELECTED_MODEL.safetensors", "credo.t07s_model"),
            "test_receipt": ("TEST_RECEIPT.json", "credo.t07s_test_receipt_v3"),
            "component_receipt": ("COMPONENT_RECEIPT.json", "credo.component_receipt_v2"),
        }
        amendment_payload: dict[str, Any] = {
            "schema_version": 2,
            "amendment_id": "pending",
            "method": _AMENDMENT_METHOD,
            "parent_qualification_id": parent_bundle.qualification_id,
            "parent_bundle_sha256": parent_bundle_sha,
            "parent_artifacts_manifest_sha256": parent_manifest_sha,
            "metric_estimand": "centered_interval_log_frequency_change",
            "r0_metric_estimand": "centered_interval_log_frequency_change",
            "false_promotion_upper_limit": _FALSE_PROMOTION_UPPER_LIMIT,
            "optimizer_rerun": False,
            "environment_hash": environment_hash,
        }
        for field, (name, schema_id) in refs.items():
            media_type = "application/x-parquet"
            if name.endswith(".json"):
                media_type = "application/json"
            elif name.endswith(".safetensors"):
                media_type = "application/x-safetensors"
            amendment_payload[field] = artifact_ref(
                temp, temp / name, schema_id=schema_id, media_type=media_type
            ).model_dump(mode="json")
        amendment_payload["amendment_id"] = contract_id(amendment_payload, id_field="amendment_id")
        amendment = ReactionRecoveryMetricAmendment.model_validate(amendment_payload)
        _write_json(temp / "reaction-recovery-amendment.json", amendment.model_dump(mode="json"))
        _write_json(
            temp / "QUALIFICATION_LINK.json",
            {
                "schema_version": 3,
                "amendment_id": amendment.amendment_id,
                "parent_qualification_id": parent_bundle.qualification_id,
                "receipt_id": receipt.receipt_id,
            },
        )
        checksums = path_manifest(temp)
        (temp / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksums)
        )

    publish_directory(destination, writer)
    verify_reaction_recovery_metric_amendment(destination, parent=parent)
    return destination


def _assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    except AssertionError as exc:
        raise IntegrityError(f"T07S {name} differs from recomputation.") from exc


def verify_reaction_recovery_qualification(path: Path) -> ReactionRecoveryQualificationBundle:
    """Verify T07S bytes and recompute every published decision statistic."""

    verify_directory(path)
    bundle = ReactionRecoveryQualificationBundle.model_validate_json(
        (path / "reaction-recovery.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    receipt = ReactionRecoveryTestReceipt.model_validate_json(
        (path / "TEST_RECEIPT.json").read_text()
    )
    component = ComponentTestReceiptV2.model_validate_json(
        (path / "COMPONENT_RECEIPT.json").read_text()
    )
    if (
        contract.test_id != _TEST_ID
        or contract.test_contract_id != bundle.test_contract_id
        or receipt.test_contract_id != bundle.test_contract_id
        or component.test_id != _TEST_ID
        or component.status != receipt.status
    ):
        raise IntegrityError("T07S contract, bundle, and receipts are inconsistent.")
    for reference in (
        bundle.null_refits,
        bundle.null_model_effects,
        bundle.recovery_curve,
        bundle.recovery_series,
        bundle.target_metrics,
        bundle.bootstrap_target_draws,
        bundle.selected_model,
        bundle.test_receipt,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"T07S artifact differs from its reference: {reference.relative_uri}."
            )
    link = json.loads((path / "QUALIFICATION_LINK.json").read_text())
    if link != {
        "schema_version": 1,
        "qualification_id": bundle.qualification_id,
        "receipt_id": receipt.receipt_id,
    }:
        raise IntegrityError("T07S qualification-to-receipt link differs from its bundle.")
    config = json.loads((path / "CONFIG.json").read_text())
    if sha256_bytes(canonical_json_bytes(config)) != receipt.config_hash:
        raise IntegrityError("T07S configuration hash differs from its receipt.")
    implementation = json.loads((path / "IMPLEMENTATION.sha256").read_text())
    current_implementation, current_files = _implementation_identity()
    if (
        implementation.get("implementation_hash") != receipt.implementation_hash
        or implementation.get("files") != current_files
        or current_implementation != receipt.implementation_hash
    ):
        raise IntegrityError("T07S implementation identity differs from current source.")
    current_environment, environment = _environment_identity()
    if (
        current_environment != receipt.environment_hash
        or current_environment != bundle.environment_hash
        or config.get("environment") != environment
    ):
        raise IntegrityError("T07S numerical environment differs from its receipt.")

    null = pd.read_parquet(path / bundle.null_refits.relative_uri).sort_values(
        ["partition", "repeat"], kind="stable"
    )
    effects = pd.read_parquet(path / bundle.null_model_effects.relative_uri)
    if len(null) != _NULL_CALIBRATION_REPEATS + _NULL_AUDIT_REPEATS:
        raise IntegrityError("T07S null-refit count is incomplete.")
    if set(null.partition) != {"calibration", "audit"} or null.repeat.nunique() != len(null):
        raise IntegrityError("T07S null partitions or repeat identities are invalid.")
    if len(effects) != len(null) * len(_TARGET_EFFECTS):
        raise IntegrityError("T07S null fitted-effect rows are incomplete.")
    if not set(effects.selected_update).issubset(_CANDIDATE_UPDATES):
        raise IntegrityError("T07S null refit selected outside the frozen update grid.")
    expected_effect_keys = {
        (str(row.partition), int(row.repeat), target_index)
        for row in null.itertuples()
        for target_index in range(len(_TARGET_EFFECTS))
    }
    observed_effect_keys = set(
        zip(
            effects.partition.astype(str),
            effects.repeat.astype(int),
            effects.target_index.astype(int),
            strict=True,
        )
    )
    if observed_effect_keys != expected_effect_keys:
        raise IntegrityError("T07S null fitted-effect identities are incomplete.")
    if not effects.loc[effects.target_index == 0, "fitted_raw_fitness"].eq(0.0).all():
        raise IntegrityError("T07S null controls do not retain exact-zero target reaction.")
    metadata = effects.merge(
        null[["partition", "repeat", "seed", "selected_update"]],
        on=["partition", "repeat"],
        suffixes=("_effect", "_summary"),
        validate="many_to_one",
    )
    if not (
        metadata.seed_effect.eq(metadata.seed_summary).all()
        and metadata.selected_update_effect.eq(metadata.selected_update_summary).all()
    ):
        raise IntegrityError("T07S null effect metadata differs from its refit summary.")
    calibration = null.loc[null.partition == "calibration", "delta"].to_numpy()
    margin = max(0.0, -float(np.quantile(calibration, 0.05, method="lower")))
    audit = null.loc[null.partition == "audit", "delta"].to_numpy()
    false_promotions = int(np.sum(audit < -margin))
    false_upper = _false_promotion_upper(false_promotions, len(audit))
    calibration_nonzero_selections = int(
        (null.loc[null.partition == "calibration", "selected_update"] != 0).sum()
    )
    audit_nonzero_selections = int(
        (null.loc[null.partition == "audit", "selected_update"] != 0).sum()
    )

    curve = pd.read_parquet(path / bundle.recovery_curve.relative_uri).sort_values("update")
    if (
        tuple(curve["update"].astype(int)) != _CANDIDATE_UPDATES
        or int(curve["selected"].sum()) != 1
    ):
        raise IntegrityError("T07S recovery selection grid is invalid.")
    selected = min(
        ((float(row.validation_loss), int(row.update)) for row in curve.itertuples()),
        key=lambda value: (value[0], value[1]),
    )[1]
    if selected != int(curve.loc[curve.selected, "update"].iloc[0]):
        raise IntegrityError("T07S selected update is not the frozen validation minimum.")
    model = _new_model(bundle.pool_count)
    state = load_tensor_file(path / bundle.selected_model.relative_uri)
    model.load_state_dict(state, strict=True)
    series = pd.read_parquet(path / bundle.recovery_series.relative_uri).sort_values(
        "series_index", kind="stable"
    )
    raw_series = series.drop(
        columns=[
            "observed_centered_interval_log_frequency_change",
            "predicted_centered_interval_log_frequency_change",
            "baseline_centered_interval_log_frequency_change",
            "predicted_centered_relative_fitness_rate",
            "new_squared_error",
            "baseline_squared_error",
        ]
    )
    expected_series = _recovery_rows(model, raw_series)
    _assert_frame_close(series, expected_series, name="series metrics")
    target = pd.read_parquet(path / bundle.target_metrics.relative_uri).sort_values("target_index")
    expected_target = _target_metrics(expected_series)
    _assert_frame_close(target, expected_target, name="target metrics")
    new_loss, baseline_loss, point_delta = _losses(target)
    draws = pd.read_parquet(path / bundle.bootstrap_target_draws.relative_uri).sort_values(
        ["draw", "position"], kind="stable"
    )
    if len(draws) != _BOOTSTRAP_DRAWS * len(target):
        raise IntegrityError("T07S target-bootstrap draw table is incomplete.")
    lookup = target.set_index("target_index")
    deltas: list[float] = []
    for _, group in draws.groupby("draw", sort=True):
        sampled = group.target_index.to_numpy(dtype=np.int64)
        metrics = lookup.loc[sampled]
        deltas.append(float(np.sqrt(metrics.new_mse.mean()) - np.sqrt(metrics.baseline_mse.mean())))
    interval = tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))
    r0_pass = false_upper <= _FALSE_PROMOTION_UPPER_LIMIT
    r1_margin_pass = interval[1] < -margin
    fitted = model.target_fitness.detach().numpy()
    truth = np.asarray(_TARGET_EFFECTS)
    reaction_rmse = float(np.sqrt(np.mean(np.square(fitted[1:] - truth[1:]))))
    sign_accuracy = float(np.mean(np.sign(fitted[1:]) == np.sign(truth[1:])))
    activity = float(np.sqrt(np.mean(np.square(fitted[1:]))))
    channel_pass = (
        activity >= _MINIMUM_CHANNEL_ACTIVITY
        and reaction_rmse <= _REACTION_RMSE_LIMIT
        and sign_accuracy == 1.0
    )
    protected_values = _protected_metrics(model, raw_series)
    protected = (
        protected_values["fixed_change"] == 0.0
        and protected_values["gauge_error"] <= _GAUGE_TOLERANCE
        and protected_values["probability_error"] <= _GAUGE_TOLERANCE
        and protected_values["rollout_error"] <= _ROLLOUT_TOLERANCE
        and protected_values["control_error"] == 0.0
        and protected_values["weight_error"] <= _GAUGE_TOLERANCE
    )
    recomputed_status = (
        "pass" if r0_pass and r1_margin_pass and channel_pass and protected else "fail_retired"
    )
    numeric = (
        abs(receipt.r0_required_margin - margin) <= 1e-12
        and receipt.r0_calibration_nonzero_checkpoint_selections == calibration_nonzero_selections
        and receipt.r0_audit_nonzero_checkpoint_selections == audit_nonzero_selections
        and abs(
            receipt.r0_audit_nonzero_checkpoint_selection_rate
            - audit_nonzero_selections / len(audit)
        )
        <= 1e-15
        and receipt.r0_audit_false_promotions == false_promotions
        and abs(receipt.r0_audit_false_promotion_upper_95 - false_upper) <= 1e-12
        and receipt.r0_false_promotion_guard_pass == r0_pass
        and receipt.r1_metric_estimand == "centered_interval_log_frequency_change"
        and receipt.r1_selected_update == selected
        and abs(receipt.r1_interval_effect_target_balanced_rmse - new_loss) <= 1e-12
        and abs(receipt.r1_zero_baseline_interval_effect_target_balanced_rmse - baseline_loss)
        <= 1e-12
        and abs(receipt.r1_point_delta - point_delta) <= 1e-12
        and np.allclose(receipt.r1_target_bootstrap_interval, interval, atol=1e-12, rtol=1e-12)
        and abs(receipt.r1_reaction_rmse - reaction_rmse) <= 1e-12
        and abs(receipt.r1_sign_accuracy - sign_accuracy) <= 1e-12
        and abs(receipt.r1_channel_activity - activity) <= 1e-12
        and abs(receipt.fixed_channel_max_abs_change - protected_values["fixed_change"]) <= 1e-12
        and abs(receipt.weighted_gauge_max_abs_error - protected_values["gauge_error"]) <= 1e-12
        and abs(
            receipt.probability_normalization_max_abs_error - protected_values["probability_error"]
        )
        <= 1e-12
        and abs(receipt.rollout_mass_max_relative_error - protected_values["rollout_error"])
        <= 1e-12
        and abs(receipt.control_target_mask_max_abs_error - protected_values["control_error"])
        <= 1e-12
    )
    if not numeric or receipt.status != recomputed_status:
        raise IntegrityError("T07S receipt differs from recomputed decision statistics.")
    if (
        component.point_delta != receipt.r1_point_delta
        or component.bootstrap_interval != receipt.r1_target_bootstrap_interval
        or component.required_margin != receipt.r0_required_margin
        or component.channel_activity != receipt.r1_channel_activity
    ):
        raise IntegrityError("T07S generic component receipt differs from detailed evidence.")
    return bundle


def verify_reaction_recovery_metric_amendment(
    path: Path, *, parent: Path
) -> ReactionRecoveryMetricAmendment:
    """Verify dev25 bytes and recompute corrected R0/R1 metrics from the parent."""

    verify_directory(path)
    parent_bundle, _ = _verified_dev23_parent(parent)
    amendment = ReactionRecoveryMetricAmendment.model_validate_json(
        (path / "reaction-recovery-amendment.json").read_text()
    )
    receipt = ReactionRecoveryTestReceiptV3.model_validate_json(
        (path / "TEST_RECEIPT.json").read_text()
    )
    component = ComponentTestReceiptV2.model_validate_json(
        (path / "COMPONENT_RECEIPT.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    parent_bundle_sha = sha256_file(parent / "reaction-recovery.json")
    parent_manifest_sha = sha256_file(parent / "artifacts.json")
    if (
        amendment.parent_qualification_id != parent_bundle.qualification_id
        or amendment.parent_bundle_sha256 != parent_bundle_sha
        or amendment.parent_artifacts_manifest_sha256 != parent_manifest_sha
        or receipt.parent_qualification_id != parent_bundle.qualification_id
    ):
        raise IntegrityError("The T07S metric amendment differs from its immutable parent.")
    if (
        contract.test_id != _TEST_ID
        or receipt.test_contract_id != contract.test_contract_id
        or component.test_id != _TEST_ID
        or component.status != receipt.status
    ):
        raise IntegrityError("The T07S metric-amendment contracts and receipts are inconsistent.")
    for reference in (
        amendment.legacy_null_refits,
        amendment.corrected_null_interval_refits,
        amendment.null_model_effects,
        amendment.recovery_curve,
        amendment.recovery_series,
        amendment.target_metrics,
        amendment.bootstrap_target_draws,
        amendment.selected_model,
        amendment.test_receipt,
        amendment.component_receipt,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"T07S amended artifact differs from its reference: {reference.relative_uri}."
            )
    copied = {
        "NULL_REFITS.parquet": parent_bundle.null_refits,
        "NULL_MODEL_EFFECTS.parquet": parent_bundle.null_model_effects,
        "RECOVERY_CURVE.parquet": parent_bundle.recovery_curve,
        "BOOTSTRAP_TARGET_DRAWS.parquet": parent_bundle.bootstrap_target_draws,
        "SELECTED_MODEL.safetensors": parent_bundle.selected_model,
    }
    for name, parent_reference in copied.items():
        if sha256_file(path / name) != parent_reference.sha256:
            raise IntegrityError(f"T07S amendment did not preserve parent bytes: {name}.")
    parent_link = json.loads((path / "PARENT_LINK.json").read_text())
    if parent_link != {
        "schema_version": 2,
        "parent_qualification_id": parent_bundle.qualification_id,
        "parent_bundle_sha256": parent_bundle_sha,
        "parent_artifacts_manifest_sha256": parent_manifest_sha,
        "selected_model_sha256": parent_bundle.selected_model.sha256,
        "legacy_null_refits_sha256": amendment.legacy_null_refits.sha256,
        "corrected_null_interval_refits_sha256": (amendment.corrected_null_interval_refits.sha256),
        "optimizer_rerun": False,
    }:
        raise IntegrityError("The T07S parent link does not prove no-retraining reuse.")
    link = json.loads((path / "QUALIFICATION_LINK.json").read_text())
    if link != {
        "schema_version": 3,
        "amendment_id": amendment.amendment_id,
        "parent_qualification_id": parent_bundle.qualification_id,
        "receipt_id": receipt.receipt_id,
    }:
        raise IntegrityError("The T07S amendment-to-receipt link is inconsistent.")

    config = json.loads((path / "CONFIG.json").read_text())
    if sha256_bytes(canonical_json_bytes(config)) != receipt.config_hash:
        raise IntegrityError("The T07S amendment configuration hash differs from its receipt.")
    implementation = json.loads((path / "IMPLEMENTATION.sha256").read_text())
    current_implementation, current_files = _implementation_identity()
    if (
        implementation.get("implementation_hash") != current_implementation
        or implementation.get("files") != current_files
        or receipt.implementation_hash != current_implementation
    ):
        raise IntegrityError("The T07S amendment implementation identity differs from source.")
    current_environment, environment = _environment_identity()
    if (
        current_environment != amendment.environment_hash
        or current_environment != receipt.environment_hash
        or config.get("environment") != environment
    ):
        raise IntegrityError("The T07S amendment numerical environment differs from its receipt.")

    expected = _amended_result(parent)
    corrected_null = pd.read_parquet(path / amendment.corrected_null_interval_refits.relative_uri)
    series = pd.read_parquet(path / amendment.recovery_series.relative_uri)
    target = pd.read_parquet(path / amendment.target_metrics.relative_uri)
    draws = pd.read_parquet(path / amendment.bootstrap_target_draws.relative_uri)
    _assert_frame_close(series, expected["recovery"], name="amended series metrics")
    _assert_frame_close(target, expected["target"], name="amended target metrics")
    _assert_frame_close(
        corrected_null,
        expected["corrected_null"],
        name="duration-correct null interval metrics",
    )
    _assert_frame_close(
        draws.sort_values(["draw", "position"]),
        expected["draws"].sort_values(["draw", "position"]),
        name="preserved bootstrap selections",
    )
    r0_pass = expected["false_promotion_upper"] <= _FALSE_PROMOTION_UPPER_LIMIT
    r1_margin_pass = expected["bootstrap_interval"][1] < -expected["required_margin"]
    channel_pass = (
        expected["channel_activity"] >= _MINIMUM_CHANNEL_ACTIVITY
        and expected["reaction_rmse"] <= _REACTION_RMSE_LIMIT
        and expected["sign_accuracy"] == 1.0
    )
    protected_pass = (
        expected["fixed_change"] == 0.0
        and expected["gauge_error"] <= _GAUGE_TOLERANCE
        and expected["probability_error"] <= _GAUGE_TOLERANCE
        and expected["rollout_error"] <= _ROLLOUT_TOLERANCE
        and expected["control_error"] == 0.0
        and expected["weight_error"] <= _GAUGE_TOLERANCE
    )
    expected_status = (
        "pass" if r0_pass and r1_margin_pass and channel_pass and protected_pass else "fail_retired"
    )
    numeric = (
        receipt.r0_calibration_repeats == _NULL_CALIBRATION_REPEATS
        and receipt.r0_audit_repeats == _NULL_AUDIT_REPEATS
        and abs(receipt.r0_required_margin - expected["required_margin"]) <= 1e-12
        and receipt.r0_calibration_nonzero_checkpoint_selections
        == expected["calibration_nonzero_selections"]
        and receipt.r0_audit_nonzero_checkpoint_selections == expected["audit_nonzero_selections"]
        and abs(
            receipt.r0_audit_nonzero_checkpoint_selection_rate
            - expected["audit_nonzero_selections"] / _NULL_AUDIT_REPEATS
        )
        <= 1e-15
        and receipt.r0_audit_false_promotions == expected["false_promotions"]
        and abs(receipt.r0_audit_false_promotion_upper_95 - expected["false_promotion_upper"])
        <= 1e-12
        and receipt.r0_false_promotion_guard_pass == r0_pass
        and receipt.r0_metric_estimand == "centered_interval_log_frequency_change"
        and receipt.legacy_null_refits == amendment.legacy_null_refits
        and receipt.corrected_null_interval_refits == amendment.corrected_null_interval_refits
        and receipt.optimizer_rerun is False
        and receipt.r1_metric_estimand == "centered_interval_log_frequency_change"
        and receipt.r1_selected_update == expected["selected_update"]
        and abs(receipt.r1_interval_effect_target_balanced_rmse - expected["new_loss"]) <= 1e-12
        and abs(
            receipt.r1_zero_baseline_interval_effect_target_balanced_rmse
            - expected["baseline_loss"]
        )
        <= 1e-12
        and abs(receipt.r1_point_delta - expected["point_delta"]) <= 1e-12
        and np.allclose(
            receipt.r1_target_bootstrap_interval,
            expected["bootstrap_interval"],
            atol=1e-12,
            rtol=1e-12,
        )
        and receipt.r1_margin_pass == r1_margin_pass
        and abs(receipt.r1_reaction_rmse - expected["reaction_rmse"]) <= 1e-12
        and abs(receipt.r1_sign_accuracy - expected["sign_accuracy"]) <= 1e-12
        and abs(receipt.r1_channel_activity - expected["channel_activity"]) <= 1e-12
        and receipt.r1_channel_activity_pass == channel_pass
        and abs(receipt.weighted_gauge_max_abs_error - expected["gauge_error"]) <= 1e-12
        and abs(receipt.rollout_mass_max_relative_error - expected["rollout_error"]) <= 1e-12
        and abs(receipt.probability_normalization_max_abs_error - expected["probability_error"])
        <= 1e-12
        and abs(receipt.control_target_mask_max_abs_error - expected["control_error"]) <= 1e-12
        and abs(receipt.fixed_channel_max_abs_change - expected["fixed_change"]) <= 1e-12
        and receipt.protected_metrics_pass == protected_pass
    )
    if not numeric or receipt.status != expected_status:
        raise IntegrityError("The T07S amended receipt differs from recomputed statistics.")
    if (
        component.point_delta != receipt.r1_point_delta
        or component.bootstrap_interval != receipt.r1_target_bootstrap_interval
        or component.required_margin != receipt.r0_required_margin
        or component.channel_activity != receipt.r1_channel_activity
    ):
        raise IntegrityError("The generic component receipt differs from amended evidence.")
    return amendment
