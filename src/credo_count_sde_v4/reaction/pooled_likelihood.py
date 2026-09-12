"""T07R-A0 pooled relative-guide reaction likelihood qualification."""

from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

from ..canonical import canonical_json_bytes, contract_id, sha256_bytes, sha256_file
from ..contracts import (
    ComponentTestContract,
    ComponentTestReceiptV2,
    ModelConfig,
    PooledFiniteMeasureBundle,
    PooledReactionLikelihoodBundle,
    PooledReactionLikelihoodReceipt,
    RawCountMassNoiseAmendment,
    RawCountMassNoiseAmendmentReceipt,
    ReactionRecoveryMetricAmendment,
    ReactionRecoveryTestReceiptV3,
    RunIntent,
)
from ..data import verify_pooled_finite_measures
from ..errors import IntegrityError
from ..model import CountSDEModel
from ..objectives import count_probabilities, exact_count_loss
from ..persistence import (
    artifact_ref,
    load_tensor_file,
    publish_directory,
    save_tensor_file,
    verify_directory,
)

_TEST_ID = "T07R_POOLED_LIKELIHOOD_A0"
_METHOD = "pooled_target_reaction_dm_likelihood_v1"
_CANDIDATE_UPDATES = (0, 25, 50, 100, 200)
_OUTER_FOLD = 0
_INNER_VALIDATION_FOLD = 1
_FIT_FOLDS = (2, 3)
_CONCENTRATION = 1000.0
_RIDGE = 0.05
_SOURCE_SMOOTHING = 0.5
_PROBABILITY_TOLERANCE = 1e-6
_EFFECT_TOLERANCE = 5e-6
_PARITY_SEED = 20_260_826
_CALIBRATION_DRAWS = 200
_BOOTSTRAP_SEED = 21_260_826
_BOOTSTRAP_DRAWS = 4000


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _implementation_identity() -> tuple[str, dict[str, str]]:
    package = Path(__file__).resolve().parents[1]
    relative_paths = (
        "contracts/models.py",
        "model/count_sde.py",
        "objectives/counts.py",
        "persistence/artifacts.py",
        "reaction/pooled_likelihood.py",
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
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }
    return sha256_bytes(canonical_json_bytes(environment)), environment


def _model(target_count: int) -> CountSDEModel:
    torch.manual_seed(_PARITY_SEED)
    model = CountSDEModel(
        ModelConfig(
            state_dim=1,
            target_count=target_count,
            pool_count=1,
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
        model.log_concentration.fill_(_CONCENTRATION)
    return model


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("guide_id", kind="stable").reset_index(drop=True)


def _tensors(frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    ordered = _ordered(frame)
    return {
        "target": torch.tensor(ordered.target_index.to_numpy(), dtype=torch.int64),
        "control": torch.tensor(ordered.is_control.to_numpy(), dtype=torch.bool),
        "source": torch.tensor(ordered.source_count.to_numpy(), dtype=torch.int64),
        "terminal": torch.tensor(ordered.terminal_count.to_numpy(), dtype=torch.int64),
        "duration": torch.ones(len(ordered), dtype=torch.float64),
        "pool": torch.zeros(len(ordered), dtype=torch.int64),
    }


def _likelihood_loss(model: CountSDEModel, frame: pd.DataFrame) -> torch.Tensor:
    values = _tensors(frame)
    raw = model.raw_fitness(values["target"], values["pool"], values["control"])
    loss = exact_count_loss(
        values["terminal"],
        values["source"],
        raw,
        values["duration"],
        values["pool"],
        model.log_concentration,
    )
    return loss / values["terminal"].sum().to(torch.float64)


def _training_loss(model: CountSDEModel, frame: pd.DataFrame) -> torch.Tensor:
    active = sorted(set(frame.loc[~frame.is_control, "target_index"].astype(int)))
    penalty = torch.mean(torch.square(model.target_fitness[active]))
    return _likelihood_loss(model, frame) + 0.5 * _RIDGE * penalty


def _fit_production(frame: pd.DataFrame, updates: int) -> CountSDEModel:
    target_count = int(frame.target_index.max()) + 1
    model = _model(target_count)
    if updates == 0:
        return model
    optimizer = torch.optim.LBFGS(
        [model.target_fitness],
        lr=1.0,
        max_iter=updates,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = _training_loss(model, frame)
        loss.backward()  # type: ignore[no-untyped-call]
        assert model.target_fitness.grad is not None
        model.target_fitness.grad[0] = 0.0
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        model.target_fitness[0] = 0.0
    return model


def _fit_and_select(fit: pd.DataFrame, validation: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, float | int | bool]] = []
    for update in _CANDIDATE_UPDATES:
        model = _fit_production(fit, update)
        rows.append(
            {
                "update": update,
                "fit_penalized_nll_per_count": float(_training_loss(model, fit).detach()),
                "validation_nll_per_count": float(_likelihood_loss(model, validation).detach()),
                "selected": False,
            }
        )
    curve = pd.DataFrame(rows)
    selected = min(
        ((float(row.validation_nll_per_count), int(row.update)) for row in curve.itertuples()),
        key=lambda item: (item[0], item[1]),
    )[1]
    curve.loc[curve["update"] == selected, "selected"] = True
    return selected, curve


def _direct_reference(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Independently solve the same penalized DM target hierarchy with SciPy."""

    ordered = _ordered(frame)
    source = ordered.source_count.to_numpy(dtype=np.float64) + _SOURCE_SMOOTHING
    terminal = ordered.terminal_count.to_numpy(dtype=np.float64)
    target = ordered.target_index.to_numpy(dtype=np.int64)
    target_count = int(target.max()) + 1
    active = np.asarray(sorted(set(target) - {0}), dtype=np.int64)

    def objective(values: np.ndarray[Any, Any]) -> tuple[float, np.ndarray[Any, Any]]:
        raw = np.zeros(target_count, dtype=np.float64)
        raw[active] = values
        logits = np.log(source) + raw[target]
        probability = np.exp(logits - np.max(logits))
        probability /= probability.sum()
        alpha = _CONCENTRATION * probability
        total = terminal.sum()
        log_probability = (
            gammaln(total + 1.0)
            - gammaln(terminal + 1.0).sum()
            + gammaln(_CONCENTRATION)
            - gammaln(total + _CONCENTRATION)
            + (gammaln(terminal + alpha) - gammaln(alpha)).sum()
        )
        alpha_score = digamma(terminal + alpha) - digamma(alpha)
        logits_score = (
            _CONCENTRATION * probability * (alpha_score - np.sum(probability * alpha_score))
        )
        target_gradient = np.bincount(target, weights=-logits_score, minlength=target_count)[active]
        target_gradient = target_gradient / total + _RIDGE * values / len(active)
        loss = -log_probability / total + 0.5 * _RIDGE * float(np.mean(np.square(values)))
        return float(loss), target_gradient

    result = minimize(
        objective,
        np.zeros(len(active), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": 0.0,
            "gtol": 1e-12,
            "maxiter": 2000,
            "maxls": 100,
            "maxcor": 50,
        },
    )
    if not result.success and float(np.max(np.abs(result.jac))) > 1e-9:
        raise IntegrityError(f"The independent T07R reference did not converge: {result.message}")
    effects = np.zeros(target_count, dtype=np.float64)
    effects[active] = result.x
    return effects


def _direct_probabilities(
    frame: pd.DataFrame, effects: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    ordered = _ordered(frame)
    logits = np.log(ordered.source_count.to_numpy(dtype=np.float64) + _SOURCE_SMOOTHING)
    logits += effects[ordered.target_index.to_numpy(dtype=np.int64)]
    probabilities = np.exp(logits - np.max(logits))
    return probabilities / probabilities.sum()


def _production_probabilities(model: CountSDEModel, frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    values = _tensors(frame)
    with torch.no_grad():
        raw = model.raw_fitness(values["target"], values["pool"], values["control"])
        probability = count_probabilities(
            values["source"], raw, values["duration"], source_smoothing=_SOURCE_SMOOTHING
        )
    return probability.numpy()


def _dm_nll_per_count(counts: np.ndarray[Any, Any], probability: np.ndarray[Any, Any]) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    alpha = _CONCENTRATION * np.asarray(probability, dtype=np.float64)
    total = counts.sum()
    value = (
        gammaln(total + 1.0)
        - gammaln(counts + 1.0).sum()
        + gammaln(_CONCENTRATION)
        - gammaln(total + _CONCENTRATION)
        + (gammaln(counts + alpha) - gammaln(alpha)).sum()
    )
    return float(-value / total)


def _load_catalog(
    pooled_bundle: Path, fold_assignment: Path
) -> tuple[PooledFiniteMeasureBundle, pd.DataFrame]:
    pooled = verify_pooled_finite_measures(pooled_bundle)
    measures = pd.read_parquet(pooled_bundle / pooled.finite_measures.relative_uri)
    catalog = pd.read_parquet(pooled_bundle / pooled.guide_catalog.relative_uri)
    counts = measures.pivot(index="guide_id", columns="checkpoint", values="cell_count")
    required_checkpoints = {pooled.source_checkpoint, pooled.terminal_checkpoint}
    if set(counts.columns) != required_checkpoints or counts.isna().any().any():
        raise IntegrityError("T07R requires one complete source/terminal count per retained guide.")
    folds = pd.read_parquet(fold_assignment)
    if list(folds.columns) != ["guide_id", "held_out_fold"] or folds.guide_id.duplicated().any():
        raise IntegrityError("T07R fold assignment has an invalid schema or duplicate guide.")
    frame = catalog.merge(counts.reset_index(), on="guide_id", validate="one_to_one").merge(
        folds, on="guide_id", validate="one_to_one"
    )
    if len(frame) != pooled.retained_guides or set(frame.held_out_fold) != {0, 1, 2, 3}:
        raise IntegrityError("T07R folds do not cover the exact retained T00 guide catalog.")
    frame = frame.rename(
        columns={
            pooled.source_checkpoint: "source_count",
            pooled.terminal_checkpoint: "terminal_count",
        }
    )
    targets = ["__control__", *sorted(frame.loc[~frame.is_control, "target_id"].unique())]
    target_index = {target: index for index, target in enumerate(targets)}
    frame["target_index"] = np.where(
        frame.is_control,
        0,
        frame.target_id.map(target_index),
    ).astype(np.int64)
    frame["role"] = np.select(
        [
            frame.held_out_fold.eq(_OUTER_FOLD),
            frame.held_out_fold.eq(_INNER_VALIDATION_FOLD),
        ],
        ["outer_evaluation", "inner_validation"],
        default="fit",
    )
    outer_targets = set(
        frame.loc[(frame.role == "outer_evaluation") & ~frame.is_control, "target_id"]
    )
    training_targets = set(
        frame.loc[(frame.role != "outer_evaluation") & ~frame.is_control, "target_id"]
    )
    if outer_targets - training_targets:
        raise IntegrityError("A T07R held-out target has no training sister guide.")
    return pooled, _ordered(frame)


def _verify_parents(
    pooled_bundle: Path, t02a_amendment: Path, t07s_amendment: Path
) -> tuple[PooledFiniteMeasureBundle, RawCountMassNoiseAmendment, ReactionRecoveryMetricAmendment]:
    pooled = verify_pooled_finite_measures(pooled_bundle)
    verify_directory(t02a_amendment)
    noise = RawCountMassNoiseAmendment.model_validate_json(
        (t02a_amendment / "raw-count-mass-noise-amendment.json").read_text()
    )
    noise_receipt = RawCountMassNoiseAmendmentReceipt.model_validate_json(
        (t02a_amendment / "VERIFICATION_RECEIPT.json").read_text()
    )
    verify_directory(t07s_amendment)
    reaction = ReactionRecoveryMetricAmendment.model_validate_json(
        (t07s_amendment / "reaction-recovery-amendment.json").read_text()
    )
    reaction_receipt = ReactionRecoveryTestReceiptV3.model_validate_json(
        (t07s_amendment / "TEST_RECEIPT.json").read_text()
    )
    if (
        not noise_receipt.parent_bundle_verified
        or noise_receipt.status != "pass"
        or reaction_receipt.status != "pass"
        or reaction_receipt.optimizer_rerun
    ):
        raise IntegrityError("T07R requires passed immutable T02A and dev25 T07S parents.")
    return pooled, noise, reaction


def _result(
    pooled_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> dict[str, Any]:
    pooled, noise, reaction = _verify_parents(pooled_bundle, t02a_amendment, t07s_amendment)
    _, catalog = _load_catalog(pooled_bundle, fold_assignment)
    fit = catalog.loc[catalog.held_out_fold.isin(_FIT_FOLDS)].copy()
    validation = catalog.loc[catalog.held_out_fold == _INNER_VALIDATION_FOLD].copy()
    training = catalog.loc[catalog.held_out_fold != _OUTER_FOLD].copy()
    outer = catalog.loc[catalog.held_out_fold == _OUTER_FOLD].copy()

    selected_update, curve = _fit_and_select(fit, validation)
    production = _fit_production(training, selected_update)
    reference_effects = _direct_reference(training)
    production_effects = production.target_fitness.detach().numpy()

    reference_probability = _direct_probabilities(outer, reference_effects)
    production_probability = _production_probabilities(production, outer)
    source_probability = _direct_probabilities(outer, np.zeros_like(reference_effects))
    observed = outer.terminal_count.to_numpy(dtype=np.int64)
    m2_loss = _dm_nll_per_count(observed, production_probability)
    m1_loss = _dm_nll_per_count(observed, reference_probability)
    point_delta = m2_loss - m1_loss

    synthetic = training.copy()
    generating_probability = _direct_probabilities(training, reference_effects)
    rng = np.random.default_rng(_PARITY_SEED)
    synthetic["terminal_count"] = rng.multinomial(250_000, generating_probability)
    synthetic_reference = _direct_reference(synthetic)
    synthetic_production = _fit_production(synthetic, max(_CANDIDATE_UPDATES))
    synthetic_reference_probability = _direct_probabilities(synthetic, synthetic_reference)
    synthetic_production_probability = _production_probabilities(synthetic_production, synthetic)
    probability_error = float(
        np.max(np.abs(synthetic_production_probability - synthetic_reference_probability))
    )
    effect_error = float(
        np.max(np.abs(synthetic_production.target_fitness.detach().numpy() - synthetic_reference))
    )
    parity_pass = probability_error < _PROBABILITY_TOLERANCE and effect_error < _EFFECT_TOLERANCE

    calibration_rows: list[dict[str, float | int]] = []
    calibration_rng = np.random.default_rng(_PARITY_SEED + 1)
    for draw in range(_CALIBRATION_DRAWS):
        terminal = calibration_rng.multinomial(250_000, synthetic_reference_probability)
        delta = _dm_nll_per_count(terminal, synthetic_production_probability) - _dm_nll_per_count(
            terminal, synthetic_reference_probability
        )
        calibration_rows.append({"draw": draw, "paired_nll_delta": delta})
    calibration = pd.DataFrame(calibration_rows)
    noninferiority_margin = max(
        1e-8,
        float(np.quantile(np.abs(calibration.paired_nll_delta), 0.95)),
    )

    bootstrap_rows: list[dict[str, float | int]] = []
    bootstrap_rng = np.random.default_rng(_BOOTSTRAP_SEED)
    observed_probability = (observed.astype(np.float64) + 0.5) / (
        observed.sum() + 0.5 * len(observed)
    )
    for draw in range(_BOOTSTRAP_DRAWS):
        terminal = bootstrap_rng.multinomial(int(observed.sum()), observed_probability)
        delta = _dm_nll_per_count(terminal, production_probability) - _dm_nll_per_count(
            terminal, reference_probability
        )
        bootstrap_rows.append({"draw": draw, "paired_nll_delta": delta})
    bootstrap = pd.DataFrame(bootstrap_rows)
    interval = tuple(
        float(value) for value in np.quantile(bootstrap.paired_nll_delta, [0.025, 0.975])
    )
    upper = float(np.quantile(bootstrap.paired_nll_delta, 0.95))
    noninferiority_pass = upper < noninferiority_margin

    outer_metrics = (
        outer[
            [
                "guide_id",
                "target_id",
                "target_index",
                "is_control",
                "source_count",
                "terminal_count",
                "held_out_fold",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )
    outer_metrics["observed_terminal_probability"] = (observed + 0.5) / (
        observed.sum() + 0.5 * len(observed)
    )
    outer_metrics["m0_source_probability"] = source_probability
    outer_metrics["m1_sister_target_probability"] = reference_probability
    outer_metrics["m2_production_probability"] = production_probability

    refit_effects = pd.DataFrame(
        {
            "target_index": np.arange(len(reference_effects)),
            "m1_direct_effect": reference_effects,
            "m2_production_effect": production_effects,
            "is_control": np.arange(len(reference_effects)) == 0,
        }
    )
    probability_normalization_error = float(abs(production_probability.sum() - 1.0))
    control_error = float(abs(production_effects[0]))
    protected_pass = probability_normalization_error <= 1e-12 and control_error == 0.0
    status = "pass" if parity_pass and noninferiority_pass and protected_pass else "fail_retired"
    return {
        "pooled": pooled,
        "noise": noise,
        "reaction": reaction,
        "catalog": catalog,
        "curve": curve,
        "production": production,
        "refit_effects": refit_effects,
        "outer_metrics": outer_metrics,
        "calibration": calibration,
        "bootstrap": bootstrap,
        "selected_update": selected_update,
        "probability_error": probability_error,
        "effect_error": effect_error,
        "parity_pass": parity_pass,
        "noninferiority_margin": noninferiority_margin,
        "m2_loss": m2_loss,
        "m1_loss": m1_loss,
        "point_delta": point_delta,
        "interval": (interval[0], interval[1]),
        "upper": upper,
        "noninferiority_pass": noninferiority_pass,
        "probability_normalization_error": probability_normalization_error,
        "control_error": control_error,
        "protected_pass": protected_pass,
        "status": status,
    }


def qualify_pooled_reaction_likelihood(
    destination: Path,
    *,
    pooled_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> Path:
    """Run and atomically publish the frozen one-fold CPU T07R-A0 pilot."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    result = _result(pooled_bundle, t02a_amendment, t07s_amendment, fold_assignment)
    environment_hash, environment = _environment_identity()
    implementation_hash, implementation_files = _implementation_identity()
    parent_hashes = {
        "t00_bundle": sha256_file(pooled_bundle / "pooled-data.json"),
        "t00_manifest": sha256_file(pooled_bundle / "artifacts.json"),
        "t02a_amendment": sha256_file(t02a_amendment / "raw-count-mass-noise-amendment.json"),
        "t02a_manifest": sha256_file(t02a_amendment / "artifacts.json"),
        "t07s_amendment": sha256_file(t07s_amendment / "reaction-recovery-amendment.json"),
        "t07s_manifest": sha256_file(t07s_amendment / "artifacts.json"),
        "fold_assignment": sha256_file(fold_assignment),
    }
    config = {
        "schema_version": 1,
        "test_id": _TEST_ID,
        "method": _METHOD,
        "device": "cpu",
        "dtype": "float64",
        "outer_fold": _OUTER_FOLD,
        "inner_validation_fold": _INNER_VALIDATION_FOLD,
        "fit_folds": list(_FIT_FOLDS),
        "candidate_updates": list(_CANDIDATE_UPDATES),
        "concentration": _CONCENTRATION,
        "ridge": _RIDGE,
        "source_smoothing": _SOURCE_SMOOTHING,
        "probability_tolerance": _PROBABILITY_TOLERANCE,
        "effect_tolerance": _EFFECT_TOLERANCE,
        "calibration_draws": _CALIBRATION_DRAWS,
        "bootstrap_draws": _BOOTSTRAP_DRAWS,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "environment": environment,
        "parent_hashes": parent_hashes,
        "interpretation": "implementation_noninferiority_not_predictive_advancement",
    }
    config_hash = sha256_bytes(canonical_json_bytes(config))
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": _TEST_ID,
        "component": "pooled_relative_guide_reaction_likelihood",
        "primary_metric": "paired_p60_dm_nll_per_count_delta",
        "primary_baseline": "leave_one_guide_out_shrunk_target_reference",
        "required_margin": result["noninferiority_margin"],
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

    def writer(temp: Path) -> None:
        result["catalog"].to_parquet(temp / "INPUT_CATALOG.parquet", index=False)
        result["curve"].to_parquet(temp / "SELECTION_CURVE.parquet", index=False)
        result["refit_effects"].to_parquet(temp / "REFIT_EFFECTS.parquet", index=False)
        result["outer_metrics"].to_parquet(temp / "OUTER_GUIDE_METRICS.parquet", index=False)
        result["calibration"].to_parquet(temp / "NULL_CALIBRATION.parquet", index=False)
        result["bootstrap"].to_parquet(temp / "BOOTSTRAP_DELTAS.parquet", index=False)
        save_tensor_file(temp / "SELECTED_MODEL.safetensors", result["production"].state_dict())
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
            temp / "PARENT_LINK.json",
            {
                "schema_version": 1,
                "pooled_data_id": result["pooled"].pooled_data_id,
                "t02a_amendment_id": result["noise"].amendment_id,
                "t07s_amendment_id": result["reaction"].amendment_id,
                "hashes": parent_hashes,
            },
        )
        _write_json(
            temp / "ESTIMATOR_PARITY.json",
            {
                "schema_version": 1,
                "training_only_synthetic_catalog": True,
                "direct_reference": "independent_scipy_penalized_dm",
                "production_path": "count_sde_model_plus_count_probabilities",
                "probability_max_abs_error": result["probability_error"],
                "probability_tolerance": _PROBABILITY_TOLERANCE,
                "effect_max_abs_error": result["effect_error"],
                "effect_tolerance": _EFFECT_TOLERANCE,
                "pass": result["parity_pass"],
            },
        )
        receipt_payload = {
            "schema_version": 1,
            "receipt_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "status": result["status"],
            "evidence_role": "development",
            "metric_estimand": "pooled_p60_dirichlet_multinomial_nll_per_count",
            "outer_fold": _OUTER_FOLD,
            "selected_update": result["selected_update"],
            "post_selection_zero_initialized_refit": True,
            "estimator_probability_max_abs_error": result["probability_error"],
            "estimator_probability_tolerance": _PROBABILITY_TOLERANCE,
            "estimator_effect_max_abs_error": result["effect_error"],
            "estimator_effect_tolerance": _EFFECT_TOLERANCE,
            "estimator_parity_pass": result["parity_pass"],
            "noninferiority_margin": result["noninferiority_margin"],
            "m2_production_nll": result["m2_loss"],
            "m1_sister_target_nll": result["m1_loss"],
            "point_delta": result["point_delta"],
            "paired_bootstrap_interval": result["interval"],
            "paired_bootstrap_upper_95": result["upper"],
            "paired_bootstrap_draws": _BOOTSTRAP_DRAWS,
            "real_pooled_noninferiority_pass": result["noninferiority_pass"],
            "protected_channels_pass": result["protected_pass"],
            "control_residual_max_abs_error": result["control_error"],
            "probability_normalization_max_abs_error": result["probability_normalization_error"],
            "parent_components_verified": True,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = PooledReactionLikelihoodReceipt.model_validate(receipt_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        component_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_id": _TEST_ID,
            "receipt_role": "model_comparison",
            "status": result["status"],
            "primary_metric": contract.primary_metric,
            "primary_baseline": contract.primary_baseline,
            "point_delta": result["point_delta"],
            "bootstrap_interval": result["interval"],
            "required_margin": result["noninferiority_margin"],
            "channel_activity": float(
                np.sqrt(
                    np.mean(np.square(result["production"].target_fitness.detach().numpy()[1:]))
                )
            ),
            "estimand": None,
            "quantile_probability": None,
            "quantile_value": None,
            "repeat_count": None,
            "sampling_method": None,
            "protected_metrics_pass": result["protected_pass"],
            "selected_update": result["selected_update"],
            "input_hashes": {**parent_hashes, "environment": environment_hash},
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
        }
        component_payload["receipt_id"] = contract_id(component_payload, id_field="receipt_id")
        component = ComponentTestReceiptV2.model_validate(component_payload)
        _write_json(temp / "COMPONENT_RECEIPT.json", component.model_dump(mode="json"))

        refs = {
            "input_catalog": artifact_ref(
                temp,
                temp / "INPUT_CATALOG.parquet",
                schema_id="credo.t07r_input_catalog",
                media_type="application/x-parquet",
            ),
            "selection_curve": artifact_ref(
                temp,
                temp / "SELECTION_CURVE.parquet",
                schema_id="credo.t07r_selection_curve",
                media_type="application/x-parquet",
            ),
            "refit_effects": artifact_ref(
                temp,
                temp / "REFIT_EFFECTS.parquet",
                schema_id="credo.t07r_refit_effects",
                media_type="application/x-parquet",
            ),
            "estimator_parity": artifact_ref(
                temp,
                temp / "ESTIMATOR_PARITY.json",
                schema_id="credo.t07r_estimator_parity",
                media_type="application/json",
            ),
            "outer_guide_metrics": artifact_ref(
                temp,
                temp / "OUTER_GUIDE_METRICS.parquet",
                schema_id="credo.t07r_outer_guide_metrics",
                media_type="application/x-parquet",
            ),
            "null_calibration": artifact_ref(
                temp,
                temp / "NULL_CALIBRATION.parquet",
                schema_id="credo.t07r_null_calibration",
                media_type="application/x-parquet",
            ),
            "bootstrap_deltas": artifact_ref(
                temp,
                temp / "BOOTSTRAP_DELTAS.parquet",
                schema_id="credo.t07r_bootstrap_deltas",
                media_type="application/x-parquet",
            ),
            "selected_model": artifact_ref(
                temp,
                temp / "SELECTED_MODEL.safetensors",
                schema_id="credo.t07r_model",
                media_type="application/x-safetensors",
            ),
            "test_receipt": artifact_ref(
                temp,
                temp / "TEST_RECEIPT.json",
                schema_id="credo.t07r_test_receipt",
                media_type="application/json",
            ),
            "component_receipt": artifact_ref(
                temp,
                temp / "COMPONENT_RECEIPT.json",
                schema_id="credo.component_receipt_v2",
                media_type="application/json",
            ),
        }
        bundle_payload = {
            "schema_version": 1,
            "qualification_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "method": _METHOD,
            "evidence_role": "development",
            "pooled_data_id": result["pooled"].pooled_data_id,
            "t02a_amendment_id": result["noise"].amendment_id,
            "t07s_amendment_id": result["reaction"].amendment_id,
            "fold_assignment_sha256": parent_hashes["fold_assignment"],
            "outer_fold": _OUTER_FOLD,
            "inner_validation_fold": _INNER_VALIDATION_FOLD,
            "candidate_updates": _CANDIDATE_UPDATES,
            "retained_guides": len(result["catalog"]),
            "outer_guides": int((result["catalog"].held_out_fold == _OUTER_FOLD).sum()),
            "target_count": int(result["catalog"].target_index.max()) + 1,
            **{name: ref.model_dump(mode="json") for name, ref in refs.items()},
        }
        bundle_payload["qualification_id"] = contract_id(
            bundle_payload, id_field="qualification_id"
        )
        bundle = PooledReactionLikelihoodBundle.model_validate(bundle_payload)
        _write_json(temp / "pooled-reaction-likelihood.json", bundle.model_dump(mode="json"))

    return publish_directory(destination, writer)


def verify_pooled_reaction_likelihood(
    path: Path,
    *,
    pooled_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> PooledReactionLikelihoodBundle:
    """Verify bytes, immutable parents, and every T07R-A0 decision statistic."""

    verify_directory(path)
    bundle = PooledReactionLikelihoodBundle.model_validate_json(
        (path / "pooled-reaction-likelihood.json").read_text()
    )
    receipt = PooledReactionLikelihoodReceipt.model_validate_json(
        (path / "TEST_RECEIPT.json").read_text()
    )
    component = ComponentTestReceiptV2.model_validate_json(
        (path / "COMPONENT_RECEIPT.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    if (
        receipt.test_contract_id != bundle.test_contract_id
        or contract.test_contract_id != bundle.test_contract_id
        or component.test_id != _TEST_ID
    ):
        raise IntegrityError("T07R contracts and receipts are inconsistent.")
    for reference in (
        bundle.input_catalog,
        bundle.selection_curve,
        bundle.refit_effects,
        bundle.estimator_parity,
        bundle.outer_guide_metrics,
        bundle.null_calibration,
        bundle.bootstrap_deltas,
        bundle.selected_model,
        bundle.test_receipt,
        bundle.component_receipt,
    ):
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(
                f"T07R artifact differs from its reference: {reference.relative_uri}."
            )
    expected = _result(pooled_bundle, t02a_amendment, t07s_amendment, fold_assignment)
    config = json.loads((path / "CONFIG.json").read_text())
    if sha256_bytes(canonical_json_bytes(config)) != receipt.config_hash:
        raise IntegrityError("T07R configuration hash differs from its receipt.")
    current_implementation, current_files = _implementation_identity()
    implementation = json.loads((path / "IMPLEMENTATION.sha256").read_text())
    if (
        receipt.implementation_hash != current_implementation
        or implementation.get("implementation_hash") != current_implementation
        or implementation.get("files") != current_files
    ):
        raise IntegrityError("T07R implementation identity differs from current source.")
    current_environment, _ = _environment_identity()
    if receipt.environment_hash != current_environment:
        raise IntegrityError("T07R numerical environment differs from its receipt.")
    frames = (
        (bundle.input_catalog.relative_uri, expected["catalog"]),
        (bundle.selection_curve.relative_uri, expected["curve"]),
        (bundle.refit_effects.relative_uri, expected["refit_effects"]),
        (bundle.outer_guide_metrics.relative_uri, expected["outer_metrics"]),
        (bundle.null_calibration.relative_uri, expected["calibration"]),
        (bundle.bootstrap_deltas.relative_uri, expected["bootstrap"]),
    )
    for name, frame in frames:
        actual = pd.read_parquet(path / name)
        pd.testing.assert_frame_equal(actual, frame, check_exact=False, atol=1e-12, rtol=1e-12)
    saved = load_tensor_file(path / bundle.selected_model.relative_uri)
    for name, value in expected["production"].state_dict().items():
        if not torch.equal(saved[name], value):
            raise IntegrityError(f"T07R selected-model tensor differs: {name}.")
    numeric = (
        receipt.selected_update == expected["selected_update"]
        and abs(receipt.estimator_probability_max_abs_error - expected["probability_error"])
        <= 1e-12
        and abs(receipt.estimator_effect_max_abs_error - expected["effect_error"]) <= 1e-12
        and receipt.estimator_parity_pass == expected["parity_pass"]
        and abs(receipt.noninferiority_margin - expected["noninferiority_margin"]) <= 1e-12
        and abs(receipt.m2_production_nll - expected["m2_loss"]) <= 1e-12
        and abs(receipt.m1_sister_target_nll - expected["m1_loss"]) <= 1e-12
        and abs(receipt.point_delta - expected["point_delta"]) <= 1e-12
        and np.allclose(receipt.paired_bootstrap_interval, expected["interval"], atol=1e-12)
        and abs(receipt.paired_bootstrap_upper_95 - expected["upper"]) <= 1e-12
        and receipt.real_pooled_noninferiority_pass == expected["noninferiority_pass"]
        and receipt.protected_channels_pass == expected["protected_pass"]
        and receipt.status == expected["status"]
    )
    if not numeric:
        raise IntegrityError("T07R receipt differs from full recomputation.")
    return bundle
