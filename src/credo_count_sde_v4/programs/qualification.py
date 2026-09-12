"""Dev40-B split, baseline, stability, null, and sign qualification."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..contracts import BaselineName, ProgramSplitKind
from ..errors import ContractError
from .baselines import fit_frozen_baselines
from .head import (
    CountLinkedProgramHead,
    ProgramBatch,
    ProgramFitConfig,
    ProgramFitResult,
    fit_program_head,
    negative_binomial_log_prob,
)


@dataclass(frozen=True)
class ProgramDataset:
    """Cohort-neutral raw-count inputs for independent program qualification."""

    counts: np.ndarray
    state: np.ndarray
    donor_index: np.ndarray
    sample_index: np.ndarray
    checkpoint_index: np.ndarray
    target_index: np.ndarray
    guide_index: np.ndarray
    guide_to_target: np.ndarray
    control_guide_indices: tuple[int, ...]
    checkpoint_times: tuple[float, ...]
    target_descriptors: np.ndarray | None = None
    protected_expression_access_contract_id: str | None = None
    heldout_donor_eligible: bool = True
    heldout_donor_ineligibility_reason: str | None = None

    def validate(self) -> None:
        counts = np.asarray(self.counts)
        rows = counts.shape[0]
        if counts.ndim != 2 or rows < 20 or counts.shape[1] < 2:
            raise ContractError("Program dataset requires a nontrivial cell-by-gene matrix.")
        if (
            np.any(counts < 0)
            or not np.isfinite(counts).all()
            or not np.allclose(counts, np.floor(counts), atol=0, rtol=0)
        ):
            raise ContractError("Program dataset requires finite raw nonnegative integer counts.")
        state = np.asarray(self.state)
        if state.ndim != 2 or state.shape[0] != rows or not np.isfinite(state).all():
            raise ContractError("Program state matrix does not align with counts.")
        for name in (
            "donor_index",
            "sample_index",
            "checkpoint_index",
            "target_index",
            "guide_index",
        ):
            value = np.asarray(getattr(self, name))
            if (
                value.shape != (rows,)
                or not np.issubdtype(value.dtype, np.integer)
                or np.any(value < 0)
            ):
                raise ContractError(f"Program {name} must be an aligned integer vector.")
        mapping = np.asarray(self.guide_to_target)
        if (
            mapping.ndim != 1
            or np.any(mapping < 0)
            or not np.array_equal(np.unique(mapping), np.arange(np.max(mapping) + 1))
            or np.max(self.guide_index) >= len(mapping)
            or not np.array_equal(mapping[self.guide_index], self.target_index)
        ):
            raise ContractError("Program guide/target labels contradict the frozen hierarchy.")
        if (
            not self.control_guide_indices
            or min(self.control_guide_indices) < 0
            or max(self.control_guide_indices) >= len(mapping)
            or tuple(sorted(set(self.control_guide_indices))) != self.control_guide_indices
        ):
            raise ContractError("Program control-guide indices are invalid.")
        if len(self.checkpoint_times) != int(np.max(self.checkpoint_index)) + 1 or any(
            right <= left
            for left, right in zip(self.checkpoint_times, self.checkpoint_times[1:], strict=False)
        ):
            raise ContractError("Program physical checkpoint times are incomplete or unordered.")
        if self.target_descriptors is not None and (
            self.target_descriptors.ndim != 2
            or self.target_descriptors.shape[0] != len(np.unique(mapping))
            or not np.isfinite(self.target_descriptors).all()
        ):
            raise ContractError("Program target descriptors are invalid.")
        if self.heldout_donor_eligible == (
            self.heldout_donor_ineligibility_reason is not None
        ):
            raise ContractError(
                "Held-out-donor eligibility and its ineligibility reason contradict."
            )
        if self.heldout_donor_eligible and len(np.unique(self.donor_index)) < 2:
            raise ContractError("Held-out-donor evaluation requires at least two donors.")


@dataclass(frozen=True)
class ProgramQualificationConfig:
    """Thresholds frozen before outer-split or null outcomes are inspected."""

    programs: int = 6
    fit: ProgramFitConfig = ProgramFitConfig()
    stability_seeds: tuple[int, ...] = (11, 29, 47)
    null_replicates: int = 20
    null_fit_epochs: int = 80
    sparse_factor_rank: int = 6
    minimum_log_likelihood_improvement: float = 0.0
    minimum_gene_sign_accuracy: float = 0.55
    minimum_seed_loading_stability: float = 0.60
    minimum_donor_loading_stability: float = 0.60
    minimum_sister_guide_correlation: float = 0.30
    maximum_null_inclusion_rate: float = 0.10
    effect_inclusion_threshold: float = 0.15
    loading_inclusion_threshold: float = 0.05
    inner_validation_fraction: float = 0.15
    seed: int = 0


@dataclass(frozen=True)
class RuntimeBaselineMetric:
    name: BaselineName
    mean_log_likelihood_per_count: float
    mean_negative_log_likelihood_per_cell: float


@dataclass(frozen=True)
class RuntimeSplitMetric:
    kind: ProgramSplitKind
    eligible: bool
    passed: bool
    reason: str | None
    fit_units: tuple[str, ...]
    evaluation_units: tuple[str, ...]
    model_mean_log_likelihood_per_count: float | None
    model_mean_negative_log_likelihood_per_cell: float | None
    baselines: tuple[RuntimeBaselineMetric, ...]
    improvement_over_best_baseline: float | None
    gene_sign_accuracy: float | None


@dataclass(frozen=True)
class RuntimeTargetGuideMetric:
    """Observed sister-guide agreement for one target in the modeled panel."""

    target_index: int
    guide_indices: tuple[int, ...]
    pair_count: int
    median_correlation: float
    within_target_variance: float


@dataclass(frozen=True)
class ProgramQualificationMetrics:
    """In-memory metrics from the complete Dev40-B qualification surface."""

    splits: tuple[RuntimeSplitMetric, ...]
    median_seed_loading_correlation: float
    median_donor_loading_correlation: float | None
    median_sister_guide_correlation: float
    target_variance_fraction: float
    inconsistent_target_fraction: float
    per_target_guide_metrics: tuple[RuntimeTargetGuideMetric, ...]
    null_program_inclusion_rate: float
    seed_stability_pass: bool
    donor_stability_pass: bool
    sister_guide_consistency_pass: bool
    null_inclusion_calibrated: bool
    heldout_target_performance_pass: bool
    gene_sign_calibration_pass: bool
    scientific_pass: bool
    seed_loadings: tuple[np.ndarray, ...]
    donor_loadings: tuple[np.ndarray, ...]
    stability_fits: tuple[ProgramFitResult, ...]
    donor_fits: tuple[ProgramFitResult, ...]
    null_replicate_inclusion_rates: tuple[float, ...]
    null_replicate_families: tuple[str, ...]
    reference_fit: ProgramFitResult


def _batch(dataset: ProgramDataset, indices: np.ndarray) -> ProgramBatch:
    counts = torch.as_tensor(dataset.counts[indices], dtype=torch.float32)
    return ProgramBatch(
        counts=counts,
        library_size=counts.sum(dim=1),
        state=torch.as_tensor(dataset.state[indices], dtype=torch.float32),
        sample_index=torch.as_tensor(dataset.sample_index[indices], dtype=torch.long),
        checkpoint_index=torch.as_tensor(dataset.checkpoint_index[indices], dtype=torch.long),
        target_index=torch.as_tensor(dataset.target_index[indices], dtype=torch.long),
        guide_index=torch.as_tensor(dataset.guide_index[indices], dtype=torch.long),
    )


def _new_model(dataset: ProgramDataset, programs: int, *, seed: int) -> CountLinkedProgramHead:
    torch.manual_seed(seed)
    descriptors = (
        None
        if dataset.target_descriptors is None
        else torch.as_tensor(dataset.target_descriptors, dtype=torch.float32)
    )
    return CountLinkedProgramHead(
        genes=dataset.counts.shape[1],
        programs=programs,
        state_dimension=dataset.state.shape[1],
        samples=int(np.max(dataset.sample_index)) + 1,
        checkpoints=len(dataset.checkpoint_times),
        targets=len(dataset.guide_to_target) and int(np.max(dataset.guide_to_target)) + 1,
        guides=len(dataset.guide_to_target),
        guide_to_target=torch.as_tensor(dataset.guide_to_target, dtype=torch.long),
        control_guide_indices=dataset.control_guide_indices,
        checkpoint_times=torch.as_tensor(dataset.checkpoint_times, dtype=torch.float32),
        target_descriptors=descriptors,
    )


def _initialize_intercept(model: CountLinkedProgramHead, counts: np.ndarray) -> None:
    frequency = counts.sum(axis=0, dtype=np.float64) + 0.5
    frequency /= frequency.sum()
    with torch.no_grad():
        model.gene_intercept.copy_(torch.as_tensor(np.log(frequency), dtype=torch.float32))


def _split_masks(
    dataset: ProgramDataset,
) -> tuple[tuple[ProgramSplitKind, np.ndarray, np.ndarray, str | None], ...]:
    rows = len(dataset.counts)
    all_rows = np.arange(rows)
    donor = int(np.max(dataset.donor_index))
    donor_eval = dataset.donor_index == donor
    donor_reason = (
        None
        if dataset.heldout_donor_eligible
        else dataset.heldout_donor_ineligibility_reason
    )

    heldout_guides: list[int] = []
    controls = set(dataset.control_guide_indices)
    for target in sorted(set(dataset.target_index) - {0}):
        guides = sorted(
            set(dataset.guide_index[dataset.target_index == target].tolist()) - controls
        )
        if len(guides) >= 2:
            heldout_guides.append(guides[-1])
    guide_eval = np.isin(dataset.guide_index, heldout_guides)

    target = int(np.max(dataset.target_index))
    target_eval = dataset.target_index == target
    target_reason = (
        None
        if dataset.target_descriptors is not None
        else "identifier-only target effects cannot predict an unseen target"
    )

    checkpoint = int(np.max(dataset.checkpoint_index))
    time_eval = dataset.checkpoint_index == checkpoint
    time_blockers: list[str] = []
    if len(dataset.checkpoint_times) < 3:
        time_blockers.append(
            "held-out time requires at least two fit checkpoints and one evaluation checkpoint"
        )
    if dataset.protected_expression_access_contract_id is None:
        time_blockers.append(
            "no protected-expression access contract for the held-out checkpoint"
        )
    time_reason = "; ".join(time_blockers) or None
    return (
        (
            ProgramSplitKind.HELDOUT_DONOR,
            all_rows[~donor_eval],
            all_rows[donor_eval],
            donor_reason,
        ),
        (
            ProgramSplitKind.HELDOUT_GUIDE_TARGET_SHARED,
            all_rows[~guide_eval],
            all_rows[guide_eval],
            None if heldout_guides else "no target has two observed sister guides",
        ),
        (
            ProgramSplitKind.HELDOUT_TARGET,
            all_rows[~target_eval],
            all_rows[target_eval],
            target_reason,
        ),
        (
            ProgramSplitKind.HELDOUT_TIME,
            all_rows[~time_eval],
            all_rows[time_eval],
            time_reason,
        ),
    )


def _inner_split(
    indices: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 0.5:
        raise ContractError("Inner validation fraction must lie between zero and one half.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    validation_count = max(1, int(round(len(shuffled) * fraction)))
    if validation_count >= len(shuffled):
        raise ContractError("Outer training set is too small for inner validation.")
    return shuffled[validation_count:], shuffled[:validation_count]


def _dispersion(counts: np.ndarray) -> np.ndarray:
    mean = counts.mean(axis=0, dtype=np.float64) + 1e-6
    variance = counts.var(axis=0, dtype=np.float64) + 1e-6
    theta = mean**2 / np.maximum(variance - mean, 1e-3)
    return np.clip(theta, 0.1, 1e4)


def _likelihood_metrics(
    counts: np.ndarray,
    mean: np.ndarray,
    dispersion: np.ndarray,
) -> tuple[float, float]:
    count_tensor = torch.as_tensor(counts, dtype=torch.float64)
    mean_tensor = torch.as_tensor(np.maximum(mean, 1e-8), dtype=torch.float64)
    dispersion_tensor = torch.as_tensor(dispersion, dtype=torch.float64)
    with torch.no_grad():
        log_prob = negative_binomial_log_prob(count_tensor, mean_tensor, dispersion_tensor).sum(
            dim=1
        )
    return (
        float(log_prob.sum() / count_tensor.sum().clamp_min(1.0)),
        float((-log_prob).mean()),
    )


def _gene_sign_accuracy(
    *,
    dataset: ProgramDataset,
    evaluation_indices: np.ndarray,
    predicted_mean: np.ndarray,
    training_indices: np.ndarray,
) -> float:
    observed = dataset.counts[evaluation_indices].sum(axis=0, dtype=np.float64) + 0.5
    observed /= observed.sum()
    controls = np.isin(
        dataset.guide_index[training_indices], np.asarray(dataset.control_guide_indices)
    )
    control_counts = dataset.counts[training_indices][controls]
    if not len(control_counts):
        return 0.0
    control = control_counts.sum(axis=0, dtype=np.float64) + 0.5
    control /= control.sum()
    predicted = predicted_mean.sum(axis=0, dtype=np.float64) + 1e-8
    predicted /= predicted.sum()
    truth_effect = np.log(observed) - np.log(control)
    predicted_effect = np.log(predicted) - np.log(control)
    informative = np.abs(truth_effect) >= 0.05
    if not np.any(informative):
        return 0.5
    return float(
        np.mean(np.sign(truth_effect[informative]) == np.sign(predicted_effect[informative]))
    )


def _evaluate_split(
    dataset: ProgramDataset,
    *,
    kind: ProgramSplitKind,
    training_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    ineligibility_reason: str | None,
    config: ProgramQualificationConfig,
    device: torch.device | str,
) -> tuple[RuntimeSplitMetric, ProgramFitResult | None]:
    if ineligibility_reason is not None or not len(training_indices) or not len(evaluation_indices):
        reason = ineligibility_reason or "split has no fit or evaluation rows"
        return (
            RuntimeSplitMetric(
                kind=kind,
                eligible=False,
                passed=False,
                reason=reason,
                fit_units=(),
                evaluation_units=(),
                model_mean_log_likelihood_per_count=None,
                model_mean_negative_log_likelihood_per_cell=None,
                baselines=(),
                improvement_over_best_baseline=None,
                gene_sign_accuracy=None,
            ),
            None,
        )
    fit_indices, validation_indices = _inner_split(
        training_indices,
        fraction=config.inner_validation_fraction,
        seed=config.seed + list(ProgramSplitKind).index(kind),
    )
    model = _new_model(dataset, config.programs, seed=config.fit.seed)
    _initialize_intercept(model, dataset.counts[fit_indices])
    fit = fit_program_head(
        model,
        training=_batch(dataset, fit_indices),
        validation=_batch(dataset, validation_indices),
        config=config.fit,
        device=device,
    )
    evaluation_batch = _batch(dataset, evaluation_indices).to(device)
    with torch.no_grad():
        predicted_mean = fit.model.mean(evaluation_batch).cpu().numpy()
    dispersion = _dispersion(dataset.counts[fit_indices])
    model_ll, model_nll = _likelihood_metrics(
        dataset.counts[evaluation_indices], predicted_mean, dispersion
    )
    baseline_predictions = fit_frozen_baselines(
        training_counts=dataset.counts[fit_indices],
        training_library_size=dataset.counts[fit_indices].sum(axis=1),
        training_checkpoint=dataset.checkpoint_index[fit_indices],
        training_target=dataset.target_index[fit_indices],
        training_guide=dataset.guide_index[fit_indices],
        control_guide_indices=dataset.control_guide_indices,
        evaluation_library_size=dataset.counts[evaluation_indices].sum(axis=1),
        evaluation_checkpoint=dataset.checkpoint_index[evaluation_indices],
        evaluation_target=dataset.target_index[evaluation_indices],
        sparse_factor_rank=config.sparse_factor_rank,
    )
    baseline_metrics = tuple(
        RuntimeBaselineMetric(
            prediction.name,
            *_likelihood_metrics(dataset.counts[evaluation_indices], prediction.mean, dispersion),
        )
        for prediction in baseline_predictions
    )
    best_baseline = max(item.mean_log_likelihood_per_count for item in baseline_metrics)
    improvement = model_ll - best_baseline
    sign_accuracy = _gene_sign_accuracy(
        dataset=dataset,
        evaluation_indices=evaluation_indices,
        predicted_mean=predicted_mean,
        training_indices=fit_indices,
    )
    passed = (
        improvement > config.minimum_log_likelihood_improvement
        and sign_accuracy >= config.minimum_gene_sign_accuracy
    )
    unit_values = {
        ProgramSplitKind.HELDOUT_DONOR: dataset.donor_index,
        ProgramSplitKind.HELDOUT_GUIDE_TARGET_SHARED: dataset.guide_index,
        ProgramSplitKind.HELDOUT_TARGET: dataset.target_index,
        ProgramSplitKind.HELDOUT_TIME: dataset.checkpoint_index,
    }[kind]
    fit_units = tuple(str(value) for value in sorted(set(unit_values[training_indices].tolist())))
    evaluation_units = tuple(
        str(value) for value in sorted(set(unit_values[evaluation_indices].tolist()))
    )
    return (
        RuntimeSplitMetric(
            kind=kind,
            eligible=True,
            passed=passed,
            reason=None,
            fit_units=fit_units,
            evaluation_units=evaluation_units,
            model_mean_log_likelihood_per_count=model_ll,
            model_mean_negative_log_likelihood_per_cell=model_nll,
            baselines=baseline_metrics,
            improvement_over_best_baseline=improvement,
            gene_sign_accuracy=sign_accuracy,
        ),
        fit,
    )


def _align_loading(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float, np.ndarray]:
    """Align a candidate basis to the reference by permutation and sign."""

    correlation = np.corrcoef(reference.T, candidate.T)[: reference.shape[1], reference.shape[1] :]
    correlation = np.nan_to_num(correlation)
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    aligned = np.empty_like(candidate)
    for reference_column, candidate_column in zip(rows, columns, strict=True):
        sign = 1.0 if correlation[reference_column, candidate_column] >= 0 else -1.0
        aligned[:, reference_column] = candidate[:, candidate_column] * sign
    return float(np.median(np.abs(correlation[rows, columns]))), aligned


def _fit_reference(
    dataset: ProgramDataset,
    config: ProgramQualificationConfig,
    *,
    seed: int,
    device: torch.device | str,
    indices: np.ndarray | None = None,
) -> ProgramFitResult:
    indices = np.arange(len(dataset.counts)) if indices is None else indices
    fit_indices, validation_indices = _inner_split(
        indices, fraction=config.inner_validation_fraction, seed=seed
    )
    model = _new_model(dataset, config.programs, seed=seed)
    _initialize_intercept(model, dataset.counts[fit_indices])
    return fit_program_head(
        model,
        training=_batch(dataset, fit_indices),
        validation=_batch(dataset, validation_indices),
        config=replace(config.fit, seed=seed),
        device=device,
    )


def _guide_target_consistency(
    dataset: ProgramDataset, *, inconsistent_threshold: float
) -> tuple[float, float, float, tuple[RuntimeTargetGuideMetric, ...]]:
    """Measure sister-guide agreement from observed counts, not fitted shrinkage."""

    controls = np.isin(
        dataset.guide_index, np.asarray(dataset.control_guide_indices, dtype=np.int64)
    )
    observed_guides = set(dataset.guide_index.tolist())
    per_target: list[RuntimeTargetGuideMetric] = []
    target_means: list[np.ndarray] = []
    within_variances: list[float] = []
    all_correlations: list[float] = []
    for target in sorted(set(dataset.target_index.tolist()) - {0}):
        guides = tuple(
            guide
            for guide in sorted(observed_guides)
            if dataset.guide_to_target[guide] == target
            and guide not in dataset.control_guide_indices
        )
        if len(guides) < 2:
            continue
        effects: list[np.ndarray] = []
        for guide in guides:
            checkpoint_effects: list[np.ndarray] = []
            for checkpoint in range(len(dataset.checkpoint_times)):
                guide_rows = (dataset.guide_index == guide) & (
                    dataset.checkpoint_index == checkpoint
                )
                control_rows = controls & (dataset.checkpoint_index == checkpoint)
                if not np.any(guide_rows) or not np.any(control_rows):
                    continue
                guide_frequency = (
                    dataset.counts[guide_rows].sum(axis=0, dtype=np.float64) + 0.5
                )
                guide_frequency /= guide_frequency.sum()
                control_frequency = (
                    dataset.counts[control_rows].sum(axis=0, dtype=np.float64) + 0.5
                )
                control_frequency /= control_frequency.sum()
                checkpoint_effects.append(np.log(guide_frequency) - np.log(control_frequency))
            if checkpoint_effects:
                effects.append(np.concatenate(checkpoint_effects))
        if len(effects) < 2 or any(value.shape != effects[0].shape for value in effects):
            continue
        pair_correlations: list[float] = []
        for left in range(len(effects)):
            for right in range(left + 1, len(effects)):
                value = float(np.corrcoef(effects[left], effects[right])[0, 1])
                if np.isfinite(value):
                    pair_correlations.append(value)
        if not pair_correlations:
            continue
        effect_matrix = np.stack(effects)
        target_mean = effect_matrix.mean(axis=0)
        within_variance = float(np.mean((effect_matrix - target_mean) ** 2))
        median_correlation = float(np.median(pair_correlations))
        all_correlations.extend(pair_correlations)
        target_means.append(target_mean)
        within_variances.append(within_variance)
        per_target.append(
            RuntimeTargetGuideMetric(
                target_index=target,
                guide_indices=guides,
                pair_count=len(pair_correlations),
                median_correlation=median_correlation,
                within_target_variance=within_variance,
            )
        )
    if not per_target:
        return -1.0, 0.0, 1.0, ()
    between_variance = (
        float(np.mean(np.var(np.stack(target_means), axis=0)))
        if len(target_means) > 1
        else 0.0
    )
    within_variance = float(np.mean(within_variances))
    total_variance = between_variance + within_variance
    target_variance_fraction = between_variance / total_variance if total_variance else 0.0
    inconsistent_fraction = float(
        np.mean(
            [item.median_correlation < inconsistent_threshold for item in per_target]
        )
    )
    return (
        float(np.median(all_correlations)),
        target_variance_fraction,
        inconsistent_fraction,
        tuple(per_target),
    )


def _null_inclusion_rate(
    dataset: ProgramDataset,
    config: ProgramQualificationConfig,
    *,
    device: torch.device | str,
) -> tuple[float, tuple[float, ...], tuple[str, ...]]:
    if config.null_replicates <= 0:
        raise ContractError("Null calibration requires at least one replicate.")
    rng = np.random.default_rng(config.seed + 991)
    included = 0
    total = 0
    replicate_rates: list[float] = []
    replicate_families: list[str] = []
    for replicate in range(config.null_replicates):
        family_index = replicate % 3
        if family_index == 0:
            family = "control_label_permutation"
            counts = dataset.counts.copy()
            for checkpoint in np.unique(dataset.checkpoint_index):
                positions = np.where(dataset.checkpoint_index == checkpoint)[0]
                counts[positions] = counts[rng.permutation(positions)]
            null_dataset = replace(
                dataset,
                counts=counts,
                protected_expression_access_contract_id=None,
            )
        elif family_index == 1:
            family = "target_within_checkpoint_permutation"
            guide = dataset.guide_index.copy()
            controls = np.isin(guide, np.asarray(dataset.control_guide_indices))
            for checkpoint in np.unique(dataset.checkpoint_index):
                positions = np.where((dataset.checkpoint_index == checkpoint) & ~controls)[0]
                guide[positions] = rng.permutation(guide[positions])
            null_dataset = replace(
                dataset,
                guide_index=guide,
                target_index=dataset.guide_to_target[guide],
                protected_expression_access_contract_id=None,
            )
        else:
            family = "negative_binomial_no_program"
            counts = np.empty_like(dataset.counts)
            for checkpoint in np.unique(dataset.checkpoint_index):
                positions = np.where(dataset.checkpoint_index == checkpoint)[0]
                frequency = dataset.counts[positions].sum(axis=0, dtype=np.float64) + 0.5
                frequency /= frequency.sum()
                library = dataset.counts[positions].sum(axis=1)
                mean = library[:, None] * frequency
                dispersion = _dispersion(dataset.counts[positions])
                gamma_rate = rng.gamma(shape=dispersion, scale=mean / dispersion)
                counts[positions] = rng.poisson(gamma_rate)
            zero_rows = np.where(counts.sum(axis=1) == 0)[0]
            counts[zero_rows, 0] = 1
            null_dataset = replace(
                dataset,
                counts=counts,
                protected_expression_access_contract_id=None,
            )
        null_config = replace(
            config,
            fit=replace(
                config.fit,
                max_epochs=min(config.fit.max_epochs, config.null_fit_epochs),
                patience=min(config.fit.patience, max(5, config.null_fit_epochs // 4)),
            ),
        )
        fitted = _fit_reference(
            null_dataset,
            null_config,
            seed=config.seed + 1000 + replicate,
            device=device,
        ).model
        parameter = (
            fitted.target_effect
            if fitted.target_effect is not None
            else fitted.target_descriptor_weight
        )
        assert parameter is not None
        magnitude = parameter.detach().abs().cpu().numpy()
        replicate_included = int(np.sum(magnitude >= config.effect_inclusion_threshold))
        included += replicate_included
        total += magnitude.size
        replicate_rates.append(replicate_included / magnitude.size)
        replicate_families.append(family)
    return included / max(total, 1), tuple(replicate_rates), tuple(replicate_families)


def qualify_program_model(
    dataset: ProgramDataset,
    *,
    config: ProgramQualificationConfig | None = None,
    device: torch.device | str = "cpu",
) -> ProgramQualificationMetrics:
    """Run all Dev40-B splits and promotion gates without touching SDE dynamics."""

    dataset.validate()
    config = config or ProgramQualificationConfig()
    if len(config.stability_seeds) < 3 or len(set(config.stability_seeds)) != len(
        config.stability_seeds
    ):
        raise ContractError("Program stability requires at least three unique seeds.")
    split_metrics: list[RuntimeSplitMetric] = []
    for kind, training, evaluation, reason in _split_masks(dataset):
        metric, _ = _evaluate_split(
            dataset,
            kind=kind,
            training_indices=training,
            evaluation_indices=evaluation,
            ineligibility_reason=reason,
            config=config,
            device=device,
        )
        split_metrics.append(metric)

    fits = tuple(
        _fit_reference(dataset, config, seed=seed, device=device) for seed in config.stability_seeds
    )
    reference_loading = fits[0].model.normalized_loadings.detach().cpu().numpy()
    seed_alignments = tuple(
        _align_loading(
            reference_loading,
            fit.model.normalized_loadings.detach().cpu().numpy(),
        )
        for fit in fits[1:]
    )
    stability = float(np.median([item[0] for item in seed_alignments]))
    aligned_seed_loadings = (reference_loading, *(item[1] for item in seed_alignments))
    donor_fits: tuple[ProgramFitResult, ...] = ()
    donor_stability: float | None = None
    aligned_donor_loadings: tuple[np.ndarray, ...] = ()
    if dataset.heldout_donor_eligible:
        donor_fits = tuple(
            _fit_reference(
                dataset,
                config,
                seed=config.seed + 500 + donor,
                device=device,
                indices=np.where(dataset.donor_index != donor)[0],
            )
            for donor in sorted(np.unique(dataset.donor_index).tolist())
        )
        donor_alignments = tuple(
            _align_loading(
                reference_loading,
                fit.model.normalized_loadings.detach().cpu().numpy(),
            )
            for fit in donor_fits
        )
        donor_stability = float(np.median([item[0] for item in donor_alignments]))
        aligned_donor_loadings = tuple(item[1] for item in donor_alignments)
    sister, target_fraction, inconsistent_fraction, per_target = _guide_target_consistency(
        dataset,
        inconsistent_threshold=config.minimum_sister_guide_correlation,
    )
    null_rate, null_replicates, null_families = _null_inclusion_rate(dataset, config, device=device)
    target_metric = next(
        item for item in split_metrics if item.kind == ProgramSplitKind.HELDOUT_TARGET
    )
    eligible_metrics = [item for item in split_metrics if item.eligible]
    gene_sign_pass = bool(eligible_metrics) and all(
        item.gene_sign_accuracy is not None
        and item.gene_sign_accuracy >= config.minimum_gene_sign_accuracy
        for item in eligible_metrics
    )
    gates = (
        stability >= config.minimum_seed_loading_stability,
        donor_stability is not None
        and donor_stability >= config.minimum_donor_loading_stability,
        sister >= config.minimum_sister_guide_correlation,
        null_rate <= config.maximum_null_inclusion_rate,
        target_metric.eligible and target_metric.passed,
        gene_sign_pass,
        all(item.eligible and item.passed for item in split_metrics),
    )
    return ProgramQualificationMetrics(
        splits=tuple(split_metrics),
        median_seed_loading_correlation=stability,
        median_donor_loading_correlation=donor_stability,
        median_sister_guide_correlation=sister,
        target_variance_fraction=target_fraction,
        inconsistent_target_fraction=inconsistent_fraction,
        per_target_guide_metrics=per_target,
        null_program_inclusion_rate=null_rate,
        seed_stability_pass=gates[0],
        donor_stability_pass=gates[1],
        sister_guide_consistency_pass=gates[2],
        null_inclusion_calibrated=gates[3],
        heldout_target_performance_pass=gates[4],
        gene_sign_calibration_pass=gates[5],
        scientific_pass=all(gates),
        seed_loadings=aligned_seed_loadings,
        donor_loadings=aligned_donor_loadings,
        stability_fits=fits,
        donor_fits=donor_fits,
        null_replicate_inclusion_rates=null_replicates,
        null_replicate_families=null_families,
        reference_fit=fits[0],
    )
