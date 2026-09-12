"""CPU-only forensic correction of the T07R physical-pool DM estimand."""

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
    PhysicalPoolConditionalReactionBundle,
    PhysicalPoolConditionalReactionReceipt,
    PooledFiniteMeasureBundle,
    RawCountMassNoiseAmendment,
    RawCountMassNoiseAmendmentReceipt,
    RawCountMassNoiseBundle,
    ReactionRecoveryMetricAmendment,
    ReactionRecoveryTestReceiptV3,
    RunIntent,
)
from ..data import verify_pooled_finite_measures
from ..errors import IntegrityError
from ..model import CountSDEModel
from ..objectives import (
    conditional_dirichlet_multinomial_log_prob,
    count_probabilities,
)
from ..persistence import (
    artifact_ref,
    load_tensor_file,
    publish_directory,
    save_tensor_file,
    verify_directory,
)

_TEST_ID = "T07R_POOLED_LIKELIHOOD_A0"
_METHOD = "physical_pool_conditional_dm_likelihood_v2"
_PREDECESSOR = "fold_subcomposition_fixed_concentration_dm_v1"
_CANDIDATE_UPDATES = (0, 25, 50, 100, 200)
_OUTER_FOLD = 0
_INNER_VALIDATION_FOLD = 1
_FIT_FOLDS = (2, 3)
_CONCENTRATION = 1000.0
_RIDGE = 0.05
_SOURCE_SMOOTHING = 0.5
_PROBABILITY_TOLERANCE = 1e-6
_EFFECT_TOLERANCE = 5e-6
_FACTORIZATION_TOLERANCE = 1e-12
_PARITY_SEED = 20_260_827
_BOOTSTRAP_SEED = 21_260_827
_BOOTSTRAP_DRAWS = 4000


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _implementation_identity() -> tuple[str, dict[str, str]]:
    package = Path(__file__).resolve().parents[1]
    paths = (
        "contracts/models.py",
        "model/count_sde.py",
        "objectives/counts.py",
        "persistence/artifacts.py",
        "reaction/physical_pool_conditional.py",
    )
    files = {relative: sha256_file(package / relative) for relative in paths}
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


def _full_tensors(catalog: pd.DataFrame) -> dict[str, torch.Tensor]:
    ordered = _ordered(catalog)
    return {
        "target": torch.tensor(ordered.target_index.to_numpy(), dtype=torch.int64),
        "control": torch.tensor(ordered.is_control.to_numpy(), dtype=torch.bool),
        "source": torch.tensor(ordered.source_count.to_numpy(), dtype=torch.int64),
        "terminal": torch.tensor(ordered.terminal_count.to_numpy(), dtype=torch.int64),
        "duration": torch.ones(len(ordered), dtype=torch.float64),
        "pool": torch.zeros(len(ordered), dtype=torch.int64),
    }


def _active_mask(catalog: pd.DataFrame, folds: tuple[int, ...]) -> np.ndarray[Any, Any]:
    return _ordered(catalog).held_out_fold.isin(folds).to_numpy(dtype=bool)


def _full_probabilities(model: CountSDEModel, catalog: pd.DataFrame) -> torch.Tensor:
    values = _full_tensors(catalog)
    raw = model.raw_fitness(values["target"], values["pool"], values["control"])
    return count_probabilities(
        values["source"], raw, values["duration"], source_smoothing=_SOURCE_SMOOTHING
    )


def _conditional_loss(
    model: CountSDEModel,
    catalog: pd.DataFrame,
    folds: tuple[int, ...],
) -> torch.Tensor:
    values = _full_tensors(catalog)
    mask = torch.tensor(_active_mask(catalog, folds), dtype=torch.bool)
    probability = _full_probabilities(model, catalog)
    log_probability = conditional_dirichlet_multinomial_log_prob(
        values["terminal"][mask], probability, mask, _CONCENTRATION
    )
    return -log_probability / values["terminal"][mask].sum().to(torch.float64)


def _training_loss(
    model: CountSDEModel,
    catalog: pd.DataFrame,
    folds: tuple[int, ...],
) -> torch.Tensor:
    active_rows = catalog.held_out_fold.isin(folds) & ~catalog.is_control
    targets = sorted(set(catalog.loc[active_rows, "target_index"].astype(int)))
    penalty = torch.mean(torch.square(model.target_fitness[targets]))
    return _conditional_loss(model, catalog, folds) + 0.5 * _RIDGE * penalty


def _fit_production(
    catalog: pd.DataFrame,
    folds: tuple[int, ...],
    updates: int,
) -> CountSDEModel:
    model = _model(int(catalog.target_index.max()) + 1)
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
        loss = _training_loss(model, catalog, folds)
        loss.backward()  # type: ignore[no-untyped-call]
        assert model.target_fitness.grad is not None
        model.target_fitness.grad[0] = 0.0
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        model.target_fitness[0] = 0.0
    return model


def _fit_and_select(catalog: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, float | int | bool]] = []
    for update in _CANDIDATE_UPDATES:
        model = _fit_production(catalog, _FIT_FOLDS, update)
        rows.append(
            {
                "update": update,
                "fit_penalized_nll_per_count": float(
                    _training_loss(model, catalog, _FIT_FOLDS).detach()
                ),
                "validation_nll_per_count": float(
                    _conditional_loss(model, catalog, (_INNER_VALIDATION_FOLD,)).detach()
                ),
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


def _direct_reference(
    catalog: pd.DataFrame,
    folds: tuple[int, ...],
) -> np.ndarray[Any, Any]:
    """Independently solve the conditional physical-pool objective with SciPy."""

    ordered = _ordered(catalog)
    source = ordered.source_count.to_numpy(dtype=np.float64) + _SOURCE_SMOOTHING
    terminal_full = ordered.terminal_count.to_numpy(dtype=np.float64)
    target = ordered.target_index.to_numpy(dtype=np.int64)
    mask = ordered.held_out_fold.isin(folds).to_numpy(dtype=bool)
    terminal = terminal_full[mask]
    target_count = int(target.max()) + 1
    active_targets = np.asarray(sorted(set(target[mask]) - {0}), dtype=np.int64)

    def objective(values: np.ndarray[Any, Any]) -> tuple[float, np.ndarray[Any, Any]]:
        raw = np.zeros(target_count, dtype=np.float64)
        raw[active_targets] = values
        logits = np.log(source) + raw[target]
        probability = np.exp(logits - np.max(logits))
        probability /= probability.sum()
        alpha_active = _CONCENTRATION * probability[mask]
        concentration_active = float(alpha_active.sum())
        total = float(terminal.sum())
        log_probability = (
            gammaln(total + 1.0)
            - gammaln(terminal + 1.0).sum()
            + gammaln(concentration_active)
            - gammaln(total + concentration_active)
            + (gammaln(terminal + alpha_active) - gammaln(alpha_active)).sum()
        )
        alpha_score = (
            digamma(concentration_active)
            - digamma(total + concentration_active)
            + digamma(terminal + alpha_active)
            - digamma(alpha_active)
        )
        weighted_score = _CONCENTRATION * probability[mask] * alpha_score
        score_sum = float(weighted_score.sum())
        logits_score = -probability * score_sum
        logits_score[mask] += weighted_score
        target_gradient: np.ndarray[Any, Any] = np.bincount(
            target, weights=-logits_score, minlength=target_count
        )[active_targets]
        target_gradient = target_gradient / total + _RIDGE * values / len(active_targets)
        loss = -log_probability / total + 0.5 * _RIDGE * float(np.mean(np.square(values)))
        return float(loss), target_gradient

    result = minimize(
        objective,
        np.zeros(len(active_targets), dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1e-12, "maxiter": 2000, "maxls": 100},
    )
    if not result.success and float(np.max(np.abs(result.jac))) > 1e-9:
        raise IntegrityError(f"Independent T07R-v2 reference failed: {result.message}")
    effects = np.zeros(target_count, dtype=np.float64)
    effects[active_targets] = result.x
    return effects


def _probabilities(catalog: pd.DataFrame, effects: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    ordered = _ordered(catalog)
    logits = np.log(ordered.source_count.to_numpy(dtype=np.float64) + _SOURCE_SMOOTHING)
    logits += effects[ordered.target_index.to_numpy(dtype=np.int64)]
    probability = np.exp(logits - np.max(logits))
    return probability / probability.sum()


def _production_probabilities(model: CountSDEModel, catalog: pd.DataFrame) -> np.ndarray[Any, Any]:
    with torch.no_grad():
        return _full_probabilities(model, catalog).numpy()


def _dm_nll_from_alpha(counts: np.ndarray[Any, Any], alpha: np.ndarray[Any, Any]) -> float:
    counts64 = np.asarray(counts, dtype=np.float64)
    alpha64 = np.asarray(alpha, dtype=np.float64)
    total = float(counts64.sum())
    concentration = float(alpha64.sum())
    value = (
        gammaln(total + 1.0)
        - gammaln(counts64 + 1.0).sum()
        + gammaln(concentration)
        - gammaln(total + concentration)
        + (gammaln(counts64 + alpha64) - gammaln(alpha64)).sum()
    )
    return float(-value / total)


def _score(
    catalog: pd.DataFrame, folds: tuple[int, ...], probability: np.ndarray[Any, Any]
) -> float:
    ordered = _ordered(catalog)
    mask = ordered.held_out_fold.isin(folds).to_numpy(dtype=bool)
    return _dm_nll_from_alpha(
        ordered.loc[mask, "terminal_count"].to_numpy(dtype=np.float64),
        _CONCENTRATION * probability[mask],
    )


def _sister_support(catalog: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = _ordered(catalog).copy()
    fit_counts = frame.loc[
        frame.held_out_fold.isin(_FIT_FOLDS) & ~frame.is_control, "target_id"
    ].value_counts()
    nonouter_counts = frame.loc[
        frame.held_out_fold.ne(_OUTER_FOLD) & ~frame.is_control, "target_id"
    ].value_counts()
    frame["fit_sister_count"] = frame.target_id.map(fit_counts).fillna(0).astype(int)
    frame["nonouter_sister_count"] = frame.target_id.map(nonouter_counts).fillna(0).astype(int)
    inner = frame.loc[frame.held_out_fold.eq(_INNER_VALIDATION_FOLD) & ~frame.is_control].copy()
    outer = frame.loc[frame.held_out_fold.eq(_OUTER_FOLD) & ~frame.is_control].copy()
    summary = {
        "minimum_inner_fit_sisters": int(inner.fit_sister_count.min()),
        "zero_inner_fit_sister_guides": int(inner.fit_sister_count.eq(0).sum()),
        "minimum_outer_nonouter_sisters": int(outer.nonouter_sister_count.min()),
        "zero_outer_nonouter_sister_guides": int(outer.nonouter_sister_count.eq(0).sum()),
    }
    if summary["zero_inner_fit_sister_guides"] or summary["zero_outer_nonouter_sister_guides"]:
        raise IntegrityError(
            "T07R-v2 requires at least one permitted sister for every scored guide."
        )
    audit = pd.concat(
        [
            inner.assign(audit_role="inner_validation").rename(
                columns={"fit_sister_count": "sister_count"}
            )[["audit_role", "sister_count"]],
            outer.assign(audit_role="outer_evaluation").rename(
                columns={"nonouter_sister_count": "sister_count"}
            )[["audit_role", "sister_count"]],
        ],
        ignore_index=True,
    )
    audit = (
        audit.value_counts(["audit_role", "sister_count"])
        .rename("guide_count")
        .reset_index()
        .sort_values(["audit_role", "sister_count"], kind="stable")
        .reset_index(drop=True)
    )
    return audit, summary


def _load_catalog(
    pooled_bundle: Path, fold_assignment: Path
) -> tuple[PooledFiniteMeasureBundle, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    pooled = verify_pooled_finite_measures(pooled_bundle)
    measures = pd.read_parquet(pooled_bundle / pooled.finite_measures.relative_uri)
    guides = pd.read_parquet(pooled_bundle / pooled.guide_catalog.relative_uri)
    counts = measures.pivot(index="guide_id", columns="checkpoint", values="cell_count")
    if set(counts.columns) != {pooled.source_checkpoint, pooled.terminal_checkpoint}:
        raise IntegrityError("T07R-v2 requires exact source and terminal counts.")
    folds = pd.read_parquet(fold_assignment)
    if list(folds.columns) != ["guide_id", "held_out_fold"] or folds.guide_id.duplicated().any():
        raise IntegrityError("T07R-v2 fold assignment is invalid.")
    frame = guides.merge(counts.reset_index(), on="guide_id", validate="one_to_one").merge(
        folds, on="guide_id", validate="one_to_one"
    )
    if len(frame) != 495 or set(frame.held_out_fold) != {0, 1, 2, 3}:
        raise IntegrityError("T07R-v2 requires the exact 495-guide/four-fold population.")
    frame = frame.rename(
        columns={
            pooled.source_checkpoint: "source_count",
            pooled.terminal_checkpoint: "terminal_count",
        }
    )
    targets = ["__control__", *sorted(frame.loc[~frame.is_control, "target_id"].unique())]
    target_index = {target: index for index, target in enumerate(targets)}
    frame["target_index"] = np.where(frame.is_control, 0, frame.target_id.map(target_index)).astype(
        np.int64
    )
    frame["role"] = np.select(
        [frame.held_out_fold.eq(_OUTER_FOLD), frame.held_out_fold.eq(_INNER_VALIDATION_FOLD)],
        ["outer_evaluation", "inner_validation"],
        default="fit",
    )
    audit, summary = _sister_support(frame)
    return pooled, _ordered(frame), audit, summary


def _verify_parents(
    pooled_bundle: Path,
    t02a_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
) -> tuple[
    PooledFiniteMeasureBundle,
    RawCountMassNoiseBundle,
    RawCountMassNoiseAmendment,
    ReactionRecoveryMetricAmendment,
]:
    pooled = verify_pooled_finite_measures(pooled_bundle)
    verify_directory(t02a_bundle)
    raw_noise = RawCountMassNoiseBundle.model_validate_json(
        (t02a_bundle / "raw-count-mass-noise.json").read_text()
    )
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
    reaction_receipt_path = t07s_amendment / reaction.test_receipt.relative_uri
    reaction_receipt = ReactionRecoveryTestReceiptV3.model_validate_json(
        reaction_receipt_path.read_text()
    )
    checks = (
        noise_receipt.status == "pass",
        noise_receipt.parent_bundle_verified,
        noise_receipt.amendment_id == noise.amendment_id,
        noise_receipt.parent_noise_id == raw_noise.noise_id == noise.parent_noise_id,
        noise.parent_bundle_sha256 == sha256_file(t02a_bundle / "raw-count-mass-noise.json"),
        raw_noise.pooled_data_id == pooled.pooled_data_id,
        (raw_noise.source_checkpoint, raw_noise.terminal_checkpoint)
        == (pooled.source_checkpoint, pooled.terminal_checkpoint)
        == (noise.source_checkpoint, noise.terminal_checkpoint),
        reaction.method == "t07s_interval_metric_amendment_v2",
        reaction_receipt.status == "pass",
        not reaction_receipt.optimizer_rerun,
        reaction_receipt.parent_qualification_id == reaction.parent_qualification_id,
        reaction.test_receipt.size_bytes == reaction_receipt_path.stat().st_size,
        reaction.test_receipt.sha256 == sha256_file(reaction_receipt_path),
    )
    if not all(checks):
        raise IntegrityError("T07R-v2 parent contracts are not exactly cross-linked.")
    return pooled, raw_noise, noise, reaction


def _factorization_check() -> float:
    probability = torch.tensor([0.08, 0.17, 0.11, 0.29, 0.35], dtype=torch.float64)
    mask = torch.tensor([True, False, True, True, False])
    counts = torch.tensor([7.0, 13.0, 5.0], dtype=torch.float64)
    actual = (
        -conditional_dirichlet_multinomial_log_prob(
            counts, probability, mask, _CONCENTRATION
        ).item()
        / counts.sum().item()
    )
    expected = _dm_nll_from_alpha(
        counts.numpy(), _CONCENTRATION * probability.numpy()[mask.numpy()]
    )
    return float(abs(actual - expected))


def _result(
    pooled_bundle: Path,
    t02a_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> dict[str, Any]:
    pooled, raw_noise, noise, reaction = _verify_parents(
        pooled_bundle, t02a_bundle, t02a_amendment, t07s_amendment
    )
    _, catalog, support_audit, support = _load_catalog(pooled_bundle, fold_assignment)
    selected_update, curve = _fit_and_select(catalog)
    production = _fit_production(catalog, (1, 2, 3), selected_update)
    reference_effects = _direct_reference(catalog, (1, 2, 3))
    production_effects = production.target_fitness.detach().numpy()
    m1_probability = _probabilities(catalog, reference_effects)
    m2_probability = _production_probabilities(production, catalog)
    m0_probability = _probabilities(catalog, np.zeros_like(reference_effects))
    outer_mask = _active_mask(catalog, (_OUTER_FOLD,))
    observed = _ordered(catalog).loc[outer_mask, "terminal_count"].to_numpy(dtype=np.int64)
    m1_loss = _score(catalog, (_OUTER_FOLD,), m1_probability)
    m2_loss = _score(catalog, (_OUTER_FOLD,), m2_probability)
    point_delta = m2_loss - m1_loss

    synthetic = catalog.copy()
    generating_effects = _direct_reference(catalog, (1, 2, 3))
    generating_probability = _probabilities(catalog, generating_effects)
    training_mask = _active_mask(catalog, (1, 2, 3))
    conditional = generating_probability[training_mask]
    conditional /= conditional.sum()
    synthetic_terminal = np.zeros(len(synthetic), dtype=np.int64)
    synthetic_terminal[training_mask] = np.random.default_rng(_PARITY_SEED).multinomial(
        250_000, conditional
    )
    synthetic["terminal_count"] = synthetic_terminal
    synthetic_reference = _direct_reference(synthetic, (1, 2, 3))
    synthetic_production = _fit_production(synthetic, (1, 2, 3), 200)
    reference_probability = _probabilities(synthetic, synthetic_reference)
    production_probability = _production_probabilities(synthetic_production, synthetic)
    probability_error = float(np.max(np.abs(production_probability - reference_probability)))
    effect_error = float(
        np.max(np.abs(synthetic_production.target_fitness.detach().numpy() - synthetic_reference))
    )
    parity_pass = probability_error < _PROBABILITY_TOLERANCE and effect_error < _EFFECT_TOLERANCE
    factorization_error = _factorization_check()

    ordered = _ordered(catalog)
    observed_full_probability = (
        ordered.terminal_count.to_numpy(dtype=np.float64) + _SOURCE_SMOOTHING
    )
    observed_full_probability /= observed_full_probability.sum()
    observed_outer_probability = observed_full_probability[outer_mask]
    observed_outer_probability /= observed_outer_probability.sum()
    alpha_outer = _CONCENTRATION * observed_full_probability[outer_mask]
    m1_alpha = _CONCENTRATION * m1_probability[outer_mask]
    m2_alpha = _CONCENTRATION * m2_probability[outer_mask]
    multinomial_rows: list[dict[str, float | int]] = []
    dm_rows: list[dict[str, float | int]] = []
    multinomial_rng = np.random.default_rng(_BOOTSTRAP_SEED)
    dm_rng = np.random.default_rng(_BOOTSTRAP_SEED + 1)
    for draw in range(_BOOTSTRAP_DRAWS):
        multinomial_counts = multinomial_rng.multinomial(
            int(observed.sum()), observed_outer_probability
        )
        multinomial_delta = _dm_nll_from_alpha(multinomial_counts, m2_alpha) - _dm_nll_from_alpha(
            multinomial_counts, m1_alpha
        )
        multinomial_rows.append({"draw": draw, "paired_nll_delta_m2_minus_m1": multinomial_delta})
        theta = dm_rng.dirichlet(alpha_outer)
        dm_counts = dm_rng.multinomial(int(observed.sum()), theta)
        dm_delta = _dm_nll_from_alpha(dm_counts, m2_alpha) - _dm_nll_from_alpha(dm_counts, m1_alpha)
        dm_rows.append({"draw": draw, "paired_nll_delta_m2_minus_m1": dm_delta})
    multinomial_bootstrap = pd.DataFrame(multinomial_rows)
    dm_bootstrap = pd.DataFrame(dm_rows)
    multinomial_interval = tuple(
        float(value)
        for value in np.quantile(multinomial_bootstrap.paired_nll_delta_m2_minus_m1, [0.025, 0.975])
    )
    dm_interval = tuple(
        float(value)
        for value in np.quantile(dm_bootstrap.paired_nll_delta_m2_minus_m1, [0.025, 0.975])
    )
    decision = (
        "m2_superior"
        if multinomial_interval[1] < 0
        else "m1_superior"
        if multinomial_interval[0] > 0
        else "inconclusive"
    )
    outer = ordered.loc[outer_mask].copy().reset_index(drop=True)
    outer["observed_terminal_conditional_probability"] = observed_outer_probability
    outer["m0_physical_pool_probability"] = m0_probability[outer_mask]
    outer["m1_physical_pool_probability"] = m1_probability[outer_mask]
    outer["m2_physical_pool_probability"] = m2_probability[outer_mask]
    outer["m1_conditional_probability"] = (
        m1_probability[outer_mask] / m1_probability[outer_mask].sum()
    )
    outer["m2_conditional_probability"] = (
        m2_probability[outer_mask] / m2_probability[outer_mask].sum()
    )
    outer["m1_conditional_concentration"] = float(m1_alpha.sum())
    outer["m2_conditional_concentration"] = float(m2_alpha.sum())
    refit_effects = pd.DataFrame(
        {
            "target_index": np.arange(len(reference_effects)),
            "m1_direct_effect": reference_effects,
            "m2_selected_policy_effect": production_effects,
            "is_control": np.arange(len(reference_effects)) == 0,
        }
    )
    protected_pass = bool(abs(m2_probability.sum() - 1.0) <= 1e-12 and production_effects[0] == 0.0)
    return {
        "pooled": pooled,
        "raw_noise": raw_noise,
        "noise": noise,
        "reaction": reaction,
        "catalog": catalog,
        "support_audit": support_audit,
        "support": support,
        "curve": curve,
        "production": production,
        "refit_effects": refit_effects,
        "outer": outer,
        "multinomial_bootstrap": multinomial_bootstrap,
        "dm_bootstrap": dm_bootstrap,
        "selected_update": selected_update,
        "probability_error": probability_error,
        "effect_error": effect_error,
        "parity_pass": parity_pass,
        "factorization_error": factorization_error,
        "factorization_pass": factorization_error < _FACTORIZATION_TOLERANCE,
        "m1_loss": m1_loss,
        "m2_loss": m2_loss,
        "point_delta": point_delta,
        "multinomial_interval": multinomial_interval,
        "dm_interval": dm_interval,
        "decision": decision,
        "protected_pass": protected_pass,
    }


def qualify_physical_pool_conditional_reaction(
    destination: Path,
    *,
    pooled_bundle: Path,
    t02a_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> Path:
    """Run and atomically publish the one permitted CPU forensic correction."""

    if destination.exists():
        raise FileExistsError(f"Committed destination already exists: {destination}.")
    result = _result(pooled_bundle, t02a_bundle, t02a_amendment, t07s_amendment, fold_assignment)
    environment_hash, environment = _environment_identity()
    implementation_hash, implementation_files = _implementation_identity()
    parent_hashes = {
        "t00_bundle": sha256_file(pooled_bundle / "pooled-data.json"),
        "t00_manifest": sha256_file(pooled_bundle / "artifacts.json"),
        "t02a_bundle": sha256_file(t02a_bundle / "raw-count-mass-noise.json"),
        "t02a_bundle_manifest": sha256_file(t02a_bundle / "artifacts.json"),
        "t02a_amendment": sha256_file(t02a_amendment / "raw-count-mass-noise-amendment.json"),
        "t02a_amendment_manifest": sha256_file(t02a_amendment / "artifacts.json"),
        "t07s_amendment": sha256_file(t07s_amendment / "reaction-recovery-amendment.json"),
        "t07s_manifest": sha256_file(t07s_amendment / "artifacts.json"),
        "fold_assignment": sha256_file(fold_assignment),
    }
    config = {
        "schema_version": 2,
        "test_id": _TEST_ID,
        "method": _METHOD,
        "evidence_role": "forensic_estimand_correction",
        "exposure_status": "historically_exposed_development",
        "device": "cpu",
        "dtype": "float64",
        "physical_pool_id": "pooled_P4_to_P60",
        "full_category_count": 495,
        "fit_terminal_category_mask": [2, 3],
        "validation_terminal_category_mask": [1],
        "outer_terminal_category_mask": [0],
        "source_denominator_scope": "all_495_retained_guides",
        "terminal_likelihood_scope": "conditional_subcomposition",
        "full_concentration": _CONCENTRATION,
        "conditional_concentration_rule": "sum_full_alpha_over_active_categories",
        "candidate_updates": list(_CANDIDATE_UPDATES),
        "ridge": _RIDGE,
        "source_smoothing": _SOURCE_SMOOTHING,
        "bootstrap_draws_each": _BOOTSTRAP_DRAWS,
        "conditional_multinomial_bootstrap_seed": _BOOTSTRAP_SEED,
        "conditional_dm_bootstrap_seed": _BOOTSTRAP_SEED + 1,
        "predictive_decision_rule": {
            "m2_superior": "primary_interval_upper_lt_zero",
            "m1_superior": "primary_interval_lower_gt_zero",
            "otherwise": "inconclusive",
        },
        "environment": environment,
        "parent_hashes": parent_hashes,
    }
    config_hash = sha256_bytes(canonical_json_bytes(config))
    contract_payload = {
        "schema_version": 1,
        "test_contract_id": "pending",
        "test_id": _TEST_ID,
        "component": "physical_pool_conditional_relative_guide_reaction",
        "primary_metric": "conditional_dm_nll_per_count_delta_m2_minus_m1",
        "primary_baseline": "fully_fitted_nonouter_sister_target_reference",
        "required_margin": 0.0,
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
        result["support_audit"].to_parquet(temp / "ROLE_SISTER_SUPPORT.parquet", index=False)
        result["curve"].to_parquet(temp / "SELECTION_CURVE.parquet", index=False)
        result["refit_effects"].to_parquet(temp / "REFIT_EFFECTS.parquet", index=False)
        result["outer"].to_parquet(temp / "OUTER_GUIDE_METRICS.parquet", index=False)
        result["multinomial_bootstrap"].to_parquet(
            temp / "CONDITIONAL_MULTINOMIAL_BOOTSTRAP.parquet", index=False
        )
        result["dm_bootstrap"].to_parquet(temp / "CONDITIONAL_DM_BOOTSTRAP.parquet", index=False)
        save_tensor_file(temp / "SELECTED_MODEL.safetensors", result["production"].state_dict())
        _write_json(temp / "CONFIG.json", config)
        _write_json(temp / "TEST_CONTRACT.json", contract.model_dump(mode="json"))
        _write_json(
            temp / "IMPLEMENTATION.sha256",
            {
                "schema_version": 2,
                "implementation_hash": implementation_hash,
                "files": implementation_files,
            },
        )
        parent_link = {
            "schema_version": 2,
            "pooled_data_id": result["pooled"].pooled_data_id,
            "source_checkpoint": result["pooled"].source_checkpoint,
            "terminal_checkpoint": result["pooled"].terminal_checkpoint,
            "t02a_noise_id": result["raw_noise"].noise_id,
            "t02a_amendment_id": result["noise"].amendment_id,
            "t07s_amendment_id": result["reaction"].amendment_id,
            "t07s_method": result["reaction"].method,
            "hashes": parent_hashes,
        }
        _write_json(temp / "PARENT_LINK.json", parent_link)
        parity = {
            "schema_version": 2,
            "training_only_synthetic_catalog": True,
            "direct_reference": "independent_scipy_physical_pool_conditional_dm",
            "production_path": "count_sde_model_plus_physical_pool_conditional_dm",
            "probability_max_abs_error": result["probability_error"],
            "probability_tolerance": _PROBABILITY_TOLERANCE,
            "effect_max_abs_error": result["effect_error"],
            "effect_tolerance": _EFFECT_TOLERANCE,
            "pass": result["parity_pass"],
        }
        _write_json(temp / "ESTIMATOR_PARITY.json", parity)
        factorization = {
            "schema_version": 2,
            "identity": "DM_full_implies_DM_active_given_active_total",
            "conditional_concentration": "sum_full_alpha_over_active_categories",
            "max_abs_error": result["factorization_error"],
            "tolerance": _FACTORIZATION_TOLERANCE,
            "pass": result["factorization_pass"],
        }
        _write_json(temp / "FACTORIZATION_CHECK.json", factorization)
        receipt_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "status": f"forensic_{result['decision']}",
            "evidence_role": "forensic_estimand_correction",
            "exposure_status": "historically_exposed_development",
            "metric_estimand": "physical_pool_conditional_p60_dm_nll_per_count",
            "selected_update": result["selected_update"],
            "estimator_probability_max_abs_error": result["probability_error"],
            "estimator_probability_tolerance": _PROBABILITY_TOLERANCE,
            "estimator_effect_max_abs_error": result["effect_error"],
            "estimator_effect_tolerance": _EFFECT_TOLERANCE,
            "estimator_parity_pass": result["parity_pass"],
            "factorization_max_abs_error": result["factorization_error"],
            "factorization_tolerance": _FACTORIZATION_TOLERANCE,
            "factorization_pass": result["factorization_pass"],
            "m2_selected_policy_nll": result["m2_loss"],
            "m1_sister_target_nll": result["m1_loss"],
            "point_delta_m2_minus_m1": result["point_delta"],
            "predictive_decision": result["decision"],
            "conditional_multinomial_interval_95": result["multinomial_interval"],
            "conditional_dm_interval_95": result["dm_interval"],
            "bootstrap_draws_each": _BOOTSTRAP_DRAWS,
            **result["support"],
            "protected_channels_pass": result["protected_pass"],
            "parent_components_verified": True,
            "config_hash": config_hash,
            "implementation_hash": implementation_hash,
            "environment_hash": environment_hash,
        }
        receipt_payload["receipt_id"] = contract_id(receipt_payload, id_field="receipt_id")
        receipt = PhysicalPoolConditionalReactionReceipt.model_validate(receipt_payload)
        _write_json(temp / "TEST_RECEIPT.json", receipt.model_dump(mode="json"))
        component_payload = {
            "schema_version": 2,
            "receipt_id": "pending",
            "test_id": _TEST_ID,
            "receipt_role": "model_comparison",
            "status": "fail_retired",
            "primary_metric": contract.primary_metric,
            "primary_baseline": contract.primary_baseline,
            "point_delta": result["point_delta"],
            "bootstrap_interval": result["multinomial_interval"],
            "required_margin": 0.0,
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
        files = {
            "input_catalog": (
                "INPUT_CATALOG.parquet",
                "credo.t07r_v2_input_catalog",
                "application/x-parquet",
            ),
            "role_support_audit": (
                "ROLE_SISTER_SUPPORT.parquet",
                "credo.t07r_v2_role_support",
                "application/x-parquet",
            ),
            "selection_curve": (
                "SELECTION_CURVE.parquet",
                "credo.t07r_v2_selection_curve",
                "application/x-parquet",
            ),
            "refit_effects": (
                "REFIT_EFFECTS.parquet",
                "credo.t07r_v2_refit_effects",
                "application/x-parquet",
            ),
            "estimator_parity": (
                "ESTIMATOR_PARITY.json",
                "credo.t07r_v2_estimator_parity",
                "application/json",
            ),
            "factorization_check": (
                "FACTORIZATION_CHECK.json",
                "credo.t07r_v2_factorization",
                "application/json",
            ),
            "outer_guide_metrics": (
                "OUTER_GUIDE_METRICS.parquet",
                "credo.t07r_v2_outer_metrics",
                "application/x-parquet",
            ),
            "conditional_multinomial_bootstrap": (
                "CONDITIONAL_MULTINOMIAL_BOOTSTRAP.parquet",
                "credo.t07r_v2_multinomial_bootstrap",
                "application/x-parquet",
            ),
            "conditional_dm_bootstrap": (
                "CONDITIONAL_DM_BOOTSTRAP.parquet",
                "credo.t07r_v2_dm_bootstrap",
                "application/x-parquet",
            ),
            "selected_model": (
                "SELECTED_MODEL.safetensors",
                "credo.t07r_v2_model",
                "application/x-safetensors",
            ),
            "parent_link": ("PARENT_LINK.json", "credo.t07r_v2_parent_link", "application/json"),
            "test_receipt": ("TEST_RECEIPT.json", "credo.t07r_v2_test_receipt", "application/json"),
            "component_receipt": (
                "COMPONENT_RECEIPT.json",
                "credo.component_receipt_v2",
                "application/json",
            ),
        }
        refs = {
            key: artifact_ref(temp, temp / name, schema_id=schema, media_type=media)
            for key, (name, schema, media) in files.items()
        }
        bundle_payload = {
            "schema_version": 2,
            "qualification_id": "pending",
            "test_contract_id": contract.test_contract_id,
            "method": _METHOD,
            "evidence_role": "forensic_estimand_correction",
            "exposure_status": "historically_exposed_development",
            "predecessor_method": _PREDECESSOR,
            "pooled_data_id": result["pooled"].pooled_data_id,
            "t02a_noise_id": result["raw_noise"].noise_id,
            "t02a_amendment_id": result["noise"].amendment_id,
            "t07s_amendment_id": result["reaction"].amendment_id,
            "fold_assignment_sha256": parent_hashes["fold_assignment"],
            "physical_pool_id": "pooled_P4_to_P60",
            "full_category_count": 495,
            "source_denominator_scope": "all_495_retained_guides",
            "terminal_likelihood_scope": "conditional_subcomposition",
            "full_concentration": _CONCENTRATION,
            "conditional_concentration_rule": "sum_full_alpha_over_active_categories",
            "outer_fold": _OUTER_FOLD,
            "inner_validation_fold": _INNER_VALIDATION_FOLD,
            "candidate_updates": _CANDIDATE_UPDATES,
            "minimum_inner_fit_sisters": result["support"]["minimum_inner_fit_sisters"],
            "minimum_outer_nonouter_sisters": result["support"]["minimum_outer_nonouter_sisters"],
            **{name: ref.model_dump(mode="json") for name, ref in refs.items()},
        }
        bundle_payload["qualification_id"] = contract_id(
            bundle_payload, id_field="qualification_id"
        )
        bundle = PhysicalPoolConditionalReactionBundle.model_validate(bundle_payload)
        _write_json(
            temp / "physical-pool-conditional-reaction.json", bundle.model_dump(mode="json")
        )

    return publish_directory(destination, writer)


def verify_physical_pool_conditional_reaction(
    path: Path,
    *,
    pooled_bundle: Path,
    t02a_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> PhysicalPoolConditionalReactionBundle:
    """Verify bytes, exact parents, and full deterministic CPU recomputation."""

    verify_directory(path)
    bundle = PhysicalPoolConditionalReactionBundle.model_validate_json(
        (path / "physical-pool-conditional-reaction.json").read_text()
    )
    receipt = PhysicalPoolConditionalReactionReceipt.model_validate_json(
        (path / "TEST_RECEIPT.json").read_text()
    )
    contract = ComponentTestContract.model_validate_json((path / "TEST_CONTRACT.json").read_text())
    component = ComponentTestReceiptV2.model_validate_json(
        (path / "COMPONENT_RECEIPT.json").read_text()
    )
    if not (
        receipt.test_contract_id == bundle.test_contract_id == contract.test_contract_id
        and component.test_id == _TEST_ID
    ):
        raise IntegrityError("T07R-v2 contracts and receipts are inconsistent.")
    references = [
        value
        for name, value in bundle.__dict__.items()
        if name
        in {
            "input_catalog",
            "role_support_audit",
            "selection_curve",
            "refit_effects",
            "estimator_parity",
            "factorization_check",
            "outer_guide_metrics",
            "conditional_multinomial_bootstrap",
            "conditional_dm_bootstrap",
            "selected_model",
            "parent_link",
            "test_receipt",
            "component_receipt",
        }
    ]
    for reference in references:
        artifact = path / reference.relative_uri
        if (
            artifact.stat().st_size != reference.size_bytes
            or sha256_file(artifact) != reference.sha256
        ):
            raise IntegrityError(f"T07R-v2 artifact differs: {reference.relative_uri}.")
    expected = _result(pooled_bundle, t02a_bundle, t02a_amendment, t07s_amendment, fold_assignment)
    config = json.loads((path / "CONFIG.json").read_text())
    if sha256_bytes(canonical_json_bytes(config)) != receipt.config_hash:
        raise IntegrityError("T07R-v2 configuration hash differs.")
    implementation_hash, implementation_files = _implementation_identity()
    implementation = json.loads((path / "IMPLEMENTATION.sha256").read_text())
    if not (
        receipt.implementation_hash == implementation_hash
        and implementation["implementation_hash"] == implementation_hash
        and implementation["files"] == implementation_files
    ):
        raise IntegrityError("T07R-v2 implementation identity differs.")
    environment_hash, _ = _environment_identity()
    if receipt.environment_hash != environment_hash:
        raise IntegrityError("T07R-v2 environment identity differs.")
    frames = (
        (bundle.input_catalog.relative_uri, expected["catalog"]),
        (bundle.role_support_audit.relative_uri, expected["support_audit"]),
        (bundle.selection_curve.relative_uri, expected["curve"]),
        (bundle.refit_effects.relative_uri, expected["refit_effects"]),
        (bundle.outer_guide_metrics.relative_uri, expected["outer"]),
        (bundle.conditional_multinomial_bootstrap.relative_uri, expected["multinomial_bootstrap"]),
        (bundle.conditional_dm_bootstrap.relative_uri, expected["dm_bootstrap"]),
    )
    for relative, expected_frame in frames:
        pd.testing.assert_frame_equal(
            pd.read_parquet(path / relative),
            expected_frame,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    saved = load_tensor_file(path / bundle.selected_model.relative_uri)
    for name, value in expected["production"].state_dict().items():
        if not torch.equal(saved[name], value):
            raise IntegrityError(f"T07R-v2 model tensor differs: {name}.")
    numeric = (
        receipt.selected_update == expected["selected_update"]
        and abs(receipt.estimator_probability_max_abs_error - expected["probability_error"])
        <= 1e-12
        and abs(receipt.estimator_effect_max_abs_error - expected["effect_error"]) <= 1e-12
        and receipt.estimator_parity_pass == expected["parity_pass"]
        and abs(receipt.factorization_max_abs_error - expected["factorization_error"]) <= 1e-12
        and receipt.factorization_pass == expected["factorization_pass"]
        and abs(receipt.m2_selected_policy_nll - expected["m2_loss"]) <= 1e-12
        and abs(receipt.m1_sister_target_nll - expected["m1_loss"]) <= 1e-12
        and abs(receipt.point_delta_m2_minus_m1 - expected["point_delta"]) <= 1e-12
        and np.allclose(
            receipt.conditional_multinomial_interval_95,
            expected["multinomial_interval"],
            atol=1e-12,
        )
        and np.allclose(receipt.conditional_dm_interval_95, expected["dm_interval"], atol=1e-12)
        and receipt.predictive_decision == expected["decision"]
        and receipt.protected_channels_pass == expected["protected_pass"]
    )
    if not numeric:
        raise IntegrityError("T07R-v2 receipt differs from full recomputation.")
    return bundle
