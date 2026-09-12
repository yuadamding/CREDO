"""Dev40-B split, baseline, stability, null, and sign qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..canonical import contract_id
from ..contracts import BaselineName, ProgramSplitKind
from ..errors import ContractError
from .baselines import fit_frozen_baselines
from .head import (
    CountLinkedProgramHead,
    ProgramBatch,
    ProgramFitConfig,
    ProgramFitResult,
    fit_program_head,
    iter_program_predictions,
    negative_binomial_log_prob,
)
from .keyed_metrics import GeneSignAccumulator, gene_sign_report, sister_guide_reports


@dataclass(frozen=True)
class ProgramDataset:
    """Cohort-neutral raw-count inputs for independent program qualification."""

    counts: np.ndarray[Any, Any]
    state: np.ndarray[Any, Any]
    donor_index: np.ndarray[Any, Any]
    sample_index: np.ndarray[Any, Any]
    checkpoint_index: np.ndarray[Any, Any]
    target_index: np.ndarray[Any, Any]
    guide_index: np.ndarray[Any, Any]
    guide_to_target: np.ndarray[Any, Any]
    control_guide_indices: tuple[int, ...]
    checkpoint_times: tuple[float, ...]
    target_descriptors: np.ndarray[Any, Any] | None = None
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
        if self.heldout_donor_eligible == (self.heldout_donor_ineligibility_reason is not None):
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
    evaluation_control_fraction: float = 0.25
    maximum_evaluation_output_bytes: int = 64 * 1024**2
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
    predictive_nb_log_likelihood: float | None = None
    common_dispersion_mean_prediction_score: float | None = None
    gene_sign_coverage: dict[str, Any] | None = None
    evaluation_reference: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeTargetGuideMetric:
    """Observed sister-guide agreement for one target in the modeled panel."""

    target_index: int
    guide_indices: tuple[int, ...]
    pair_count: int
    median_correlation: float
    within_target_variance: float
    shared_support: tuple[dict[str, Any], ...] = ()
    support_complete: bool = True


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
    seed_loadings: tuple[np.ndarray[Any, Any], ...]
    donor_loadings: tuple[np.ndarray[Any, Any], ...]
    stability_fits: tuple[ProgramFitResult, ...]
    donor_fits: tuple[ProgramFitResult, ...]
    null_replicate_inclusion_rates: tuple[float, ...]
    null_replicate_families: tuple[str, ...]
    reference_fit: ProgramFitResult
    qualification_scope: str = "legacy_four_split_component_diagnostic_not_nested_donor_forecast"
    metric_revision: int = 3
    null_calibration_semantics: str = "legacy_coefficient_exceedance_not_discovery_calibration"
    biological_efficiency_identified: bool = False


def _batch(dataset: ProgramDataset, indices: np.ndarray[Any, Any]) -> ProgramBatch:
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


def _initialize_intercept(model: CountLinkedProgramHead, counts: np.ndarray[Any, Any]) -> None:
    frequency = counts.sum(axis=0, dtype=np.float64) + 0.5
    frequency /= frequency.sum()
    with torch.no_grad():
        model.gene_intercept.copy_(torch.as_tensor(np.log(frequency), dtype=torch.float32))


def _split_masks(
    dataset: ProgramDataset,
    *,
    evaluation_control_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[
    tuple[
        ProgramSplitKind,
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        str | None,
    ],
    ...,
]:
    """Separate target predictions from prespecified evaluator-only controls.

    Guide/target splits reserve a seeded fraction per donor/condition before
    fitting; no outcomes are inspected. At least one control stays available for
    fitting. A singleton control is not borrowed as independent test evidence.
    Donor/time references come only from their already held-out observations.
    """
    if not 0 < evaluation_control_fraction < 1:
        raise ContractError("Evaluation control fraction must lie between zero and one.")
    rows = len(dataset.counts)
    all_rows = np.arange(rows)
    donor = int(np.max(dataset.donor_index))
    donor_eval = dataset.donor_index == donor
    donor_reason = (
        None if dataset.heldout_donor_eligible else dataset.heldout_donor_ineligibility_reason
    )

    heldout_guides: list[int] = []
    controls = set(dataset.control_guide_indices)
    control_mask = np.isin(dataset.guide_index, tuple(controls))
    reserved = np.zeros(rows, dtype=bool)
    rng = np.random.default_rng(seed)
    for d, c in sorted(
        set(zip(dataset.donor_index.tolist(), dataset.checkpoint_index.tolist(), strict=True))
    ):
        candidates = all_rows[
            control_mask & (dataset.donor_index == d) & (dataset.checkpoint_index == c)
        ]
        if len(candidates) >= 2:
            number = min(
                len(candidates) - 1,
                max(1, int(np.ceil(len(candidates) * evaluation_control_fraction))),
            )
            reserved[rng.permutation(candidates)[:number]] = True
    for target in sorted(set(dataset.target_index)):
        guides = sorted(
            set(dataset.guide_index[dataset.target_index == target].tolist()) - controls
        )
        if len(guides) >= 2:
            heldout_guides.append(guides[-1])
    guide_eval = np.isin(dataset.guide_index, heldout_guides)

    perturbation_rows = ~np.isin(dataset.guide_index, tuple(controls))
    observed_targets = dataset.target_index[perturbation_rows]
    target = int(np.max(observed_targets)) if len(observed_targets) else -1
    target_eval = (dataset.target_index == target) & perturbation_rows
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
        time_blockers.append("no protected-expression access contract for the held-out checkpoint")
    time_reason = "; ".join(time_blockers) or None
    return (
        (
            ProgramSplitKind.HELDOUT_DONOR,
            all_rows[~donor_eval],
            all_rows[donor_eval & ~control_mask],
            all_rows[donor_eval & control_mask],
            donor_reason,
        ),
        (
            ProgramSplitKind.HELDOUT_GUIDE_TARGET_SHARED,
            all_rows[~guide_eval & ~reserved],
            all_rows[guide_eval],
            all_rows[reserved],
            None if heldout_guides else "no target has two observed sister guides",
        ),
        (
            ProgramSplitKind.HELDOUT_TARGET,
            all_rows[~target_eval & ~reserved],
            all_rows[target_eval],
            all_rows[reserved],
            target_reason,
        ),
        (
            ProgramSplitKind.HELDOUT_TIME,
            all_rows[~time_eval],
            all_rows[time_eval & ~control_mask],
            all_rows[time_eval & control_mask],
            time_reason,
        ),
    )


def _inner_split(
    indices: np.ndarray[Any, Any], *, fraction: float, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if not 0 < fraction < 0.5:
        raise ContractError("Inner validation fraction must lie between zero and one half.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    validation_count = max(1, int(round(len(shuffled) * fraction)))
    if validation_count >= len(shuffled):
        raise ContractError("Outer training set is too small for inner validation.")
    return shuffled[validation_count:], shuffled[:validation_count]


def _dispersion(counts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    mean = counts.mean(axis=0, dtype=np.float64) + 1e-6
    variance = counts.var(axis=0, dtype=np.float64) + 1e-6
    theta = mean**2 / np.maximum(variance - mean, 1e-3)
    return np.clip(theta, 0.1, 1e4)


def _likelihood_metrics(
    counts: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    dispersion: np.ndarray[Any, Any],
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
    evaluation_indices: np.ndarray[Any, Any],
    predicted_mean: np.ndarray[Any, Any],
    predicted_reference_mean: np.ndarray[Any, Any],
    training_indices: np.ndarray[Any, Any],
) -> float | None:
    """Revision 2; training counts cannot substitute for matched endpoint controls."""
    return gene_sign_report(
        counts=dataset.counts[evaluation_indices],
        predicted_mean=predicted_mean,
        predicted_reference_mean=predicted_reference_mean,
        donor=dataset.donor_index[evaluation_indices],
        condition=dataset.checkpoint_index[evaluation_indices],
        guide=dataset.guide_index[evaluation_indices],
        guide_to_target=dataset.guide_to_target,
        control_guides=dataset.control_guide_indices,
    ).accuracy


def _evaluate_split(
    dataset: ProgramDataset,
    *,
    kind: ProgramSplitKind,
    training_indices: np.ndarray[Any, Any],
    evaluation_indices: np.ndarray[Any, Any],
    evaluation_reference_indices: np.ndarray[Any, Any],
    ineligibility_reason: str | None,
    config: ProgramQualificationConfig,
    device: torch.device | str,
) -> tuple[RuntimeSplitMetric, ProgramFitResult | None]:
    partitions = (training_indices, evaluation_indices, evaluation_reference_indices)
    for indices in partitions:
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or np.any(indices >= len(dataset.counts))
            or np.any(indices < 0)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ContractError("Evaluation row partitions must contain unique valid row indices.")
    if any(np.intersect1d(partitions[a], partitions[b]).size for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ContractError(
            "Fitting, evaluation targets and evaluation references must be disjoint."
        )
    if (
        not np.isin(
            dataset.guide_index[evaluation_reference_indices], dataset.control_guide_indices
        ).all()
        or np.isin(dataset.guide_index[evaluation_indices], dataset.control_guide_indices).any()
    ):
        raise ContractError("Evaluation target/reference roles contradict control identities.")
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
    genes = dataset.counts.shape[1]
    if config.maximum_evaluation_output_bytes <= 0 or config.fit.minibatch_size <= 0:
        raise ContractError("Evaluation batch and output limits must be positive.")
    target_keys = set(
        zip(
            dataset.donor_index[evaluation_indices].tolist(),
            dataset.checkpoint_index[evaluation_indices].tolist(),
            dataset.guide_index[evaluation_indices].tolist(),
            strict=True,
        )
    )
    reference_keys = set(
        zip(
            dataset.donor_index[evaluation_reference_indices].tolist(),
            dataset.checkpoint_index[evaluation_reference_indices].tolist(),
            strict=True,
        )
    )
    baseline_keys, baseline_inverse = np.unique(
        np.stack(
            (
                dataset.checkpoint_index[evaluation_indices],
                dataset.target_index[evaluation_indices],
            ),
            axis=1,
        ),
        axis=0,
        return_inverse=True,
    )
    summary_bytes = (3 * len(target_keys) + len(reference_keys)) * genes * 8
    baseline_bytes = len(BaselineName) * len(baseline_keys) * genes * 8
    chunk_bytes = min(config.fit.minibatch_size, len(evaluation_indices)) * genes * 2 * 4
    required_bytes = summary_bytes + baseline_bytes + chunk_bytes
    if (
        genes > config.fit.maximum_panel_genes
        or required_bytes > config.maximum_evaluation_output_bytes
    ):
        raise ContractError("Outer evaluation exceeds its small-panel evaluation-output budget.")
    reference_record = {
        "policy": "outcome_blind_seeded_control_reservation_or_outer_donor_time_controls_v1",
        "independence_unit": "cell_observations_not_independent_donor_or_culture_replicates",
        "reserved_fraction": config.evaluation_control_fraction,
        "reservation_seed": config.seed,
        "fit_candidate_rows": len(training_indices),
        "evaluation_target_rows": len(evaluation_indices),
        "evaluation_reference_rows": len(evaluation_reference_indices),
        "partition_sha256": contract_id(
            {
                name: rows.tolist()
                for name, rows in zip(("fit", "target", "reference"), partitions, strict=True)
            }
        ),
        "reference_outcomes": "evaluator_only_excluded_from_fit_and_prediction",
        "likelihood_population": "evaluation_target_rows_only",
        "estimated_retained_array_and_prediction_bytes": required_bytes,
    }
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
    dispersion = _dispersion(dataset.counts[fit_indices])
    fitted_dispersion = fit.model.dispersion.detach().cpu().numpy()
    # Fit once at unit depth on unique evaluation metadata keys. Only bounded
    # chunks of cell-level baseline predictions are ever materialized.
    baseline_tables = fit_frozen_baselines(
        training_counts=dataset.counts[fit_indices],
        training_library_size=dataset.counts[fit_indices].sum(axis=1),
        training_checkpoint=dataset.checkpoint_index[fit_indices],
        training_target=dataset.target_index[fit_indices],
        training_guide=dataset.guide_index[fit_indices],
        control_guide_indices=dataset.control_guide_indices,
        evaluation_library_size=np.ones(len(baseline_keys)),
        evaluation_checkpoint=baseline_keys[:, 0],
        evaluation_target=baseline_keys[:, 1],
        sparse_factor_rank=config.sparse_factor_rank,
    )
    accumulator = GeneSignAccumulator(
        genes=genes,
        guide_to_target=dataset.guide_to_target,
        control_guides=dataset.control_guide_indices,
        maximum_output_bytes=config.maximum_evaluation_output_bytes - baseline_bytes - chunk_bytes,
    )
    # References are summarized only here, never passed to the model/baselines.
    for start in range(0, len(evaluation_reference_indices), config.fit.minibatch_size):
        indices = evaluation_reference_indices[start : start + config.fit.minibatch_size]
        accumulator.add_controls(
            counts=dataset.counts[indices],
            donor=dataset.donor_index[indices],
            condition=dataset.checkpoint_index[indices],
            guide=dataset.guide_index[indices],
        )
    batches = (
        _batch(dataset, evaluation_indices[start : start + config.fit.minibatch_size])
        for start in range(0, len(evaluation_indices), config.fit.minibatch_size)
    )
    totals = np.zeros((2 + len(baseline_tables), 2), dtype=np.float64)
    total_counts = 0.0
    start = 0
    for batch, prediction, reference in iter_program_predictions(
        fit.model,
        batches,
        device=device,
        maximum_batch_rows=config.fit.minibatch_size,
        maximum_output_bytes=config.maximum_evaluation_output_bytes
        - summary_bytes
        - baseline_bytes,
    ):
        assert reference is not None
        stop = start + len(batch.counts)
        indices = evaluation_indices[start:stop]
        counts = dataset.counts[indices]
        library = counts.sum(axis=1, dtype=np.float64)
        predicted = prediction.cpu().numpy()
        reference_mean = reference.cpu().numpy()
        weight = np.array([library.sum(), len(indices)])
        total_counts += float(library.sum())
        totals[0] += np.asarray(_likelihood_metrics(counts, predicted, dispersion)) * weight
        totals[1] += np.asarray(_likelihood_metrics(counts, predicted, fitted_dispersion)) * weight
        for number, table in enumerate(baseline_tables, start=2):
            mean = library[:, None] * table.mean[baseline_inverse[start:stop]]
            totals[number] += np.asarray(_likelihood_metrics(counts, mean, dispersion)) * weight
        accumulator.add_targets(
            counts=counts,
            predicted_mean=predicted,
            predicted_reference_mean=reference_mean,
            donor=dataset.donor_index[indices],
            condition=dataset.checkpoint_index[indices],
            guide=dataset.guide_index[indices],
        )
        start = stop
    totals /= np.array([max(total_counts, 1.0), len(evaluation_indices)])
    model_ll, model_nll = map(float, totals[0])
    predictive_ll = float(totals[1, 0])
    baseline_metrics = tuple(
        RuntimeBaselineMetric(table.name, float(values[0]), float(values[1]))
        for table, values in zip(baseline_tables, totals[2:], strict=True)
    )
    best_baseline = max(item.mean_log_likelihood_per_count for item in baseline_metrics)
    improvement = model_ll - best_baseline
    sign_report = accumulator.report()
    sign_accuracy = sign_report.accuracy
    passed = (
        improvement > config.minimum_log_likelihood_improvement
        and sign_accuracy is not None
        and sign_accuracy >= config.minimum_gene_sign_accuracy
        and sign_report.supported_units == sign_report.total_units
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
            predictive_nb_log_likelihood=predictive_ll,
            common_dispersion_mean_prediction_score=model_ll,
            gene_sign_coverage=asdict(sign_report),
            evaluation_reference=reference_record,
        ),
        fit,
    )


def _align_loading(
    reference: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]
) -> tuple[float, np.ndarray[Any, Any]]:
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
    indices: np.ndarray[Any, Any] | None = None,
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
    reports = sister_guide_reports(
        counts=dataset.counts,
        donor=dataset.donor_index,
        condition=dataset.checkpoint_index,
        guide=dataset.guide_index,
        guide_to_target=dataset.guide_to_target,
        control_guides=dataset.control_guide_indices,
    )
    targets = []
    all_correlations = []
    for report in reports:
        pairs = [p for p in report["pairs"] if p["correlation"] is not None]
        correlations = [float(p["correlation"]) for p in pairs]
        all_correlations.extend(correlations)
        targets.append(
            RuntimeTargetGuideMetric(
                target_index=report["target_index"],
                guide_indices=report["guide_indices"],
                pair_count=len(pairs),
                median_correlation=float(np.median(correlations)) if correlations else -1.0,
                within_target_variance=(
                    float(np.mean([p["within_variance"] for p in pairs])) if pairs else 0.0
                ),
                shared_support=tuple(report["pairs"]),
                support_complete=bool(pairs) and len(pairs) == len(report["pairs"]),
            )
        )
    if not targets:
        return -1.0, 0.0, 1.0, ()
    # No between-target variance is inferred from differently keyed supports.
    # The historical scalar is retained at zero; revision-2 reports mark it unestimated.
    return (
        float(np.median(all_correlations)) if all_correlations else -1.0,
        0.0,
        float(
            np.mean(
                [
                    not t.support_complete or t.median_correlation < inconsistent_threshold
                    for t in targets
                ]
            )
        ),
        tuple(targets),
    )


def _legacy_parameter_exceedance_rate(
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
    for kind, training, evaluation, references, reason in _split_masks(
        dataset, evaluation_control_fraction=config.evaluation_control_fraction, seed=config.seed
    ):
        metric, _ = _evaluate_split(
            dataset,
            kind=kind,
            training_indices=training,
            evaluation_indices=evaluation,
            evaluation_reference_indices=references,
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
    aligned_donor_loadings: tuple[np.ndarray[Any, Any], ...] = ()
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
    null_rate, null_replicates, null_families = _legacy_parameter_exceedance_rate(
        dataset, config, device=device
    )
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
        donor_stability is not None and donor_stability >= config.minimum_donor_loading_stability,
        sister >= config.minimum_sister_guide_correlation
        and bool(per_target)
        and all(t.support_complete for t in per_target),
        False,  # Coefficient thresholds/shortened null fits do not calibrate discoveries.
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
