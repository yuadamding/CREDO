"""Immutable Dev40 qualification bundle publication and verification."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from scipy.special import ndtr

from ..canonical import contract_id, sha256_file
from ..contracts import (
    ArtifactRef,
    BaselineMetric,
    BaselineName,
    BiologicalProgramQualificationBundle,
    CountLinkedProgramContract,
    GeneLevelEffect,
    GuideTargetConsistency,
    HeldoutTargetMode,
    LibrarySizeSemantics,
    NegativeBinomialSemantics,
    PerturbationProgramEffect,
    ProgramDefinition,
    ProgramNullContract,
    ProgramQualificationProtocol,
    ProgramQualificationReceipt,
    ProgramQualificationStatus,
    ProgramSplitEvaluation,
    ProgramSplitKind,
    ProgramUncertainty,
    ScientificScope,
    SparseLoadingPrior,
    StrictModel,
)
from ..errors import ContractError, IntegrityError
from .qualification import (
    ProgramDataset,
    ProgramQualificationConfig,
    ProgramQualificationMetrics,
    qualify_program_model,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    provisional = model.model_construct(**payload)
    payload[id_field] = provisional.identity(id_field=id_field)
    return model.model_validate(payload)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_npy(path: Path, value: np.ndarray[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value), allow_pickle=False)


def _artifact(root: Path, path: Path, schema_id: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        schema_id=schema_id,
        schema_version=1,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
        relative_uri=path.relative_to(root).as_posix(),
    )


def _program_contract(
    *,
    root: Path,
    dataset: ProgramDataset,
    config: ProgramQualificationConfig,
    feature_order_sha256: str,
    guide_target_map_sha256: str,
    fit_row_ids_sha256: str,
) -> CountLinkedProgramContract:
    descriptor_artifact = None
    if dataset.target_descriptors is not None:
        path = root / "artifacts" / "target_descriptors.npy"
        _write_npy(path, dataset.target_descriptors)
        descriptor_artifact = _artifact(root, path, "credo.target_descriptors", "application/x-npy")
    return _identified(
        CountLinkedProgramContract,
        {
            "schema_id": "credo.count_linked_program_contract",
            "schema_version": 1,
            "count_likelihood": NegativeBinomialSemantics.NB2_GENE_DISPERSION,
            "library_size_semantics": LibrarySizeSemantics.OBSERVED_TOTAL_COUNT_OFFSET,
            "sparse_loading_prior": SparseLoadingPrior.L1_WITH_UNIT_NORM_COLUMNS,
            "state_dependence_enabled": dataset.state.shape[1] > 0,
            "checkpoint_dependence_enabled": True,
            "donor_or_sample_effects_enabled": True,
            "heldout_target_mode": (
                HeldoutTargetMode.PREDECLARED_TARGET_DESCRIPTORS
                if descriptor_artifact is not None
                else HeldoutTargetMode.IDENTIFIER_ONLY_NOT_ELIGIBLE
            ),
            "target_descriptor_artifact": descriptor_artifact,
            "feature_order_sha256": feature_order_sha256,
            "guide_target_map_sha256": guide_target_map_sha256,
            "fit_row_ids_sha256": fit_row_ids_sha256,
            "genes": dataset.counts.shape[1],
            "programs": config.programs,
            "state_dimension": dataset.state.shape[1],
            "samples": int(np.max(dataset.sample_index)) + 1,
            "checkpoints": len(dataset.checkpoint_times),
            "checkpoint_time_values": dataset.checkpoint_times,
            "targets": int(np.max(dataset.target_index)) + 1,
            "guides": len(dataset.guide_to_target),
            "control_guide_indices": dataset.control_guide_indices,
            "loading_l1_weight": config.fit.loading_l1_weight,
            "guide_deviation_l2_weight": config.fit.guide_deviation_l2_weight,
            "target_effect_l2_weight": config.fit.target_effect_l2_weight,
        },
        "program_contract_id",
    )


def _split_contracts(
    metrics: ProgramQualificationMetrics,
    dataset: ProgramDataset,
) -> tuple[ProgramSplitEvaluation, ...]:
    results: list[ProgramSplitEvaluation] = []
    for item in metrics.splits:
        # The frozen V1 summary requires every metric for an eligible split.
        # Preserve unsupported contrasts as undefined in the detailed V2
        # artifact, rather than fabricating a score to satisfy that old shape.
        supported = item.eligible and item.gene_sign_accuracy is not None
        results.append(
            ProgramSplitEvaluation(
                split_id=contract_id(
                    {
                        "kind": item.kind,
                        "fit_units": item.fit_units,
                        "evaluation_units": item.evaluation_units,
                    }
                ),
                kind=item.kind,
                eligible=supported,
                passed=item.passed if supported else False,
                ineligibility_reason=(
                    None
                    if supported
                    else item.reason or "Matched donor-condition gene-sign truth is unavailable."
                ),
                fit_units=item.fit_units,
                evaluation_units=item.evaluation_units,
                protected_expression_access_contract_id=(
                    dataset.protected_expression_access_contract_id
                    if item.kind == ProgramSplitKind.HELDOUT_TIME and supported
                    else None
                ),
                model_mean_log_likelihood_per_count=(
                    item.model_mean_log_likelihood_per_count if supported else None
                ),
                model_mean_negative_log_likelihood_per_cell=(
                    item.model_mean_negative_log_likelihood_per_cell if supported else None
                ),
                baselines=tuple(
                    BaselineMetric(
                        baseline=baseline.name,
                        mean_log_likelihood_per_count=(baseline.mean_log_likelihood_per_count),
                        mean_negative_log_likelihood_per_cell=(
                            baseline.mean_negative_log_likelihood_per_cell
                        ),
                    )
                    for baseline in item.baselines
                    if supported
                ),
                improvement_over_best_baseline=(
                    item.improvement_over_best_baseline if supported else None
                ),
                gene_sign_accuracy=item.gene_sign_accuracy if supported else None,
            )
        )
    return tuple(results)


def _qualification_protocol(
    config: ProgramQualificationConfig,
) -> ProgramQualificationProtocol:
    """Persist every setting that can change a Dev40-B qualification decision."""

    fit = config.fit
    return _identified(
        ProgramQualificationProtocol,
        {
            "split_kinds": tuple(ProgramSplitKind),
            "baselines": tuple(BaselineName),
            "stability_seeds": config.stability_seeds,
            "sparse_factor_rank": config.sparse_factor_rank,
            "null_replicates": config.null_replicates,
            "null_fit_epochs": config.null_fit_epochs,
            "minimum_log_likelihood_improvement": (config.minimum_log_likelihood_improvement),
            "minimum_gene_sign_accuracy": config.minimum_gene_sign_accuracy,
            "minimum_seed_loading_stability": config.minimum_seed_loading_stability,
            "minimum_donor_loading_stability": (config.minimum_donor_loading_stability),
            "minimum_sister_guide_correlation": (config.minimum_sister_guide_correlation),
            "maximum_null_inclusion_rate": config.maximum_null_inclusion_rate,
            "effect_inclusion_threshold": config.effect_inclusion_threshold,
            "loading_inclusion_threshold": config.loading_inclusion_threshold,
            "inner_validation_fraction": config.inner_validation_fraction,
            "seed": config.seed,
            "learning_rate": fit.learning_rate,
            "weight_decay": fit.weight_decay,
            "max_epochs": fit.max_epochs,
            "patience": fit.patience,
            "minimum_delta": fit.minimum_delta,
            "gradient_clip_norm": fit.gradient_clip_norm,
            "minibatch_size": fit.minibatch_size,
            "loading_l1_weight": fit.loading_l1_weight,
            "guide_deviation_l2_weight": fit.guide_deviation_l2_weight,
            "target_effect_l2_weight": fit.target_effect_l2_weight,
        },
        "qualification_protocol_id",
    )


def _model_artifacts(
    root: Path,
    metrics: ProgramQualificationMetrics,
    *,
    target_index: int,
    guide_indices: tuple[int, ...],
    loading_inclusion_threshold: float,
) -> dict[str, ArtifactRef]:
    model = metrics.reference_fit.model.cpu()
    state_path = root / "artifacts" / "model_state.npz"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in model.state_dict().items()
        if value is not None
    }
    np.savez_compressed(state_path, **arrays)
    references: dict[str, ArtifactRef] = {
        "model_state": _artifact(
            root, state_path, "credo.count_linked_program_state", "application/x-npz"
        )
    }
    times = torch.stack(
        (
            torch.ones_like(model.normalized_checkpoint_times),
            model.normalized_checkpoint_times,
        ),
        dim=1,
    )
    with torch.no_grad():
        reference = times @ model.checkpoint_reference
        targets = torch.full((len(times),), target_index, dtype=torch.long)
        target = model._target_activity(targets, times)
        deviations = torch.stack(
            [
                torch.einsum("nb,bk->nk", times, model.guide_deviation[index])
                for index in guide_indices
            ]
        )
        efficiency = model.latent_guide_scale[list(guide_indices)]
        combined = target.unsqueeze(0) + efficiency[:, None, None] * deviations
        gene_effect = combined.mean(dim=0) @ model.normalized_loadings.T
        stability_gene_effects: list[torch.Tensor] = []
        for fit in metrics.stability_fits:
            replicate = fit.model.cpu()
            replicate_times = torch.stack(
                (
                    torch.ones_like(replicate.normalized_checkpoint_times),
                    replicate.normalized_checkpoint_times,
                ),
                dim=1,
            )
            replicate_targets = torch.full((len(replicate_times),), target_index, dtype=torch.long)
            replicate_target = replicate._target_activity(replicate_targets, replicate_times)
            replicate_deviations = torch.stack(
                [
                    torch.einsum("nb,bk->nk", replicate_times, replicate.guide_deviation[index])
                    for index in guide_indices
                ]
            )
            replicate_efficiency = replicate.latent_guide_scale[list(guide_indices)]
            replicate_combined = (
                replicate_target.unsqueeze(0)
                + replicate_efficiency[:, None, None] * replicate_deviations
            )
            stability_gene_effects.append(
                replicate_combined.mean(dim=0) @ replicate.normalized_loadings.T
            )
        replicate_stack = torch.stack(stability_gene_effects)
        standard_error = replicate_stack.std(dim=0, correction=1) / np.sqrt(
            len(stability_gene_effects)
        )
        sign_probability = torch.as_tensor(
            ndtr((gene_effect.abs() / standard_error.clamp_min(1e-6)).cpu().numpy())
        )
    values = {
        "reference_activity": reference.numpy(),
        "target_activity": target.numpy(),
        "guide_deviation_activity": deviations.numpy(),
        "guide_efficiency": efficiency.numpy(),
        "gene_log_fold_change": gene_effect.numpy(),
        "gene_standard_error": standard_error.numpy(),
        "gene_sign_probability": sign_probability.numpy(),
        "seed_aligned_loadings": np.stack(metrics.seed_loadings),
        "loading_inclusion": (
            np.abs(np.stack(metrics.seed_loadings).astype(np.float64))
            >= loading_inclusion_threshold
        ).mean(axis=0),
        "null_inclusion": np.asarray(metrics.null_replicate_inclusion_rates),
    }
    if metrics.donor_loadings:
        values["donor_aligned_loadings"] = np.stack(metrics.donor_loadings)
    for name, value in values.items():
        path = root / "artifacts" / f"{name}.npy"
        _write_npy(path, value)
        references[name] = _artifact(root, path, f"credo.{name}", "application/x-npy")
    return references


def _publish_bundle(
    root: Path,
    *,
    dataset: ProgramDataset,
    metrics: ProgramQualificationMetrics,
    config: ProgramQualificationConfig,
    study_id: str,
    capability_assessment_id: str,
    scope: ScientificScope,
    perturbation_id: str,
    target_id: str,
    target_index: int,
    guide_ids: tuple[str, ...],
    guide_indices: tuple[int, ...],
    feature_order_sha256: str,
    guide_target_map_sha256: str,
    fit_row_ids_sha256: str,
) -> BiologicalProgramQualificationBundle:
    if config.null_replicates < 20:
        raise ContractError("Persisted qualification requires at least 20 null replicates.")
    contract = _program_contract(
        root=root,
        dataset=dataset,
        config=config,
        feature_order_sha256=feature_order_sha256,
        guide_target_map_sha256=guide_target_map_sha256,
        fit_row_ids_sha256=fit_row_ids_sha256,
    )
    qualification_protocol = _qualification_protocol(config)
    training_device = str(next(metrics.reference_fit.model.parameters()).device)
    artifacts = _model_artifacts(
        root,
        metrics,
        target_index=target_index,
        guide_indices=guide_indices,
        loading_inclusion_threshold=config.loading_inclusion_threshold,
    )
    model = metrics.reference_fit.model.cpu()
    definitions: list[ProgramDefinition] = []
    loadings = model.normalized_loadings.detach().numpy()
    for index in range(config.programs):
        path = root / "artifacts" / f"program_{index:03d}_loading.npy"
        _write_npy(path, loadings[:, index])
        artifact = _artifact(root, path, "credo.program_loading", "application/x-npy")
        selected = np.abs(loadings[:, index]) >= config.loading_inclusion_threshold
        definitions.append(
            _identified(
                ProgramDefinition,
                {
                    "program_contract_id": contract.program_contract_id,
                    "program_index": index,
                    "loading_artifact": artifact,
                    "loading_threshold": config.loading_inclusion_threshold,
                    "nonzero_genes": int(selected.sum()),
                    "positive_genes": int(np.sum(loadings[selected, index] > 0)),
                    "negative_genes": int(np.sum(loadings[selected, index] < 0)),
                    "l1_norm": float(np.abs(loadings[:, index]).sum()),
                    "l2_norm": float(np.linalg.norm(loadings[:, index])),
                },
                "program_definition_id",
            )
        )
    effect = _identified(
        PerturbationProgramEffect,
        {
            "program_contract_id": contract.program_contract_id,
            "scope": scope,
            "perturbation_id": perturbation_id,
            "target_id": target_id,
            "guide_ids": tuple(sorted(guide_ids)),
            "reference_activity": artifacts["reference_activity"],
            "target_activity": artifacts["target_activity"],
            "guide_deviation_activity": artifacts["guide_deviation_activity"],
            "guide_efficiency": artifacts["guide_efficiency"],
        },
        "program_effect_id",
    )
    gene_effect = _identified(
        GeneLevelEffect,
        {
            "program_contract_id": contract.program_contract_id,
            "program_effect_id": effect.program_effect_id,
            "scope": scope,
            "log_fold_change_artifact": artifacts["gene_log_fold_change"],
            "standard_error_artifact": artifacts["gene_standard_error"],
            "sign_probability_artifact": artifacts["gene_sign_probability"],
            "decomposition_identity_max_abs_error": 0.0,
        },
        "gene_effect_id",
    )
    consistency_path = root / "artifacts" / "guide_target_consistency.json"
    _write_json(
        consistency_path,
        {
            "metric_revision": metrics.metric_revision,
            "median_sister_guide_correlation": metrics.median_sister_guide_correlation,
            "target_variance_fraction": metrics.target_variance_fraction,
            "target_variance_fraction_status": "not_estimated_by_keyed_evaluator_legacy_slot_zero",
            "inconsistent_target_fraction": metrics.inconsistent_target_fraction,
            "per_target": [
                {
                    "target_index": item.target_index,
                    "guide_indices": item.guide_indices,
                    "pair_count": item.pair_count,
                    "median_correlation": item.median_correlation,
                    "within_target_variance": item.within_target_variance,
                    "shared_support": item.shared_support,
                    "support_complete": item.support_complete,
                    "unsupported_legacy_slots": "minus_one_correlation_zero_variance_if_no_pairs",
                }
                for item in metrics.per_target_guide_metrics
            ],
        },
    )
    consistency = _identified(
        GuideTargetConsistency,
        {
            "program_contract_id": contract.program_contract_id,
            "scope": scope,
            "per_target_artifact": _artifact(
                root, consistency_path, "credo.guide_target_consistency", "application/json"
            ),
            "median_sister_guide_correlation": max(
                -1.0, min(1.0, metrics.median_sister_guide_correlation)
            ),
            "target_variance_fraction": metrics.target_variance_fraction,
            "inconsistent_target_fraction": metrics.inconsistent_target_fraction,
            "minimum_sister_guides": 2,
        },
        "consistency_id",
    )
    donor_artifact = artifacts.get("donor_aligned_loadings")
    donor_count = len(np.unique(dataset.donor_index)) if dataset.heldout_donor_eligible else 0
    uncertainty = _identified(
        ProgramUncertainty,
        {
            "program_contract_id": contract.program_contract_id,
            "scope": scope,
            "seed_count": len(config.stability_seeds),
            "donor_count": donor_count,
            "seed_loading_stability_artifact": artifacts["seed_aligned_loadings"],
            "donor_loading_stability_artifact": donor_artifact,
            "inclusion_probability_artifact": artifacts["loading_inclusion"],
            "null_inclusion_artifact": artifacts["null_inclusion"],
            "median_seed_loading_correlation": metrics.median_seed_loading_correlation,
            "median_donor_loading_correlation": (metrics.median_donor_loading_correlation),
            "donor_stability_available": donor_artifact is not None,
            "null_inclusion_rate": metrics.null_program_inclusion_rate,
        },
        "uncertainty_id",
    )
    metrics_path = root / "artifacts" / "qualification_metrics.json"
    _write_json(
        metrics_path,
        {
            "metric_revision": metrics.metric_revision,
            "qualification_scope": metrics.qualification_scope,
            "scientific_pass": metrics.scientific_pass,
            "biological_efficiency_identified": metrics.biological_efficiency_identified,
            "guide_efficiency_artifact_semantics": "legacy_name_unidentified_latent_guide_scale",
            "null_calibration_semantics": metrics.null_calibration_semantics,
            "null_inclusion_calibrated": metrics.null_inclusion_calibrated,
            "legacy_parameter_exceedance_rate": metrics.null_program_inclusion_rate,
            "execution_limits": {
                "maximum_panel_genes": config.fit.maximum_panel_genes,
                "maximum_host_payload_bytes": config.fit.maximum_host_payload_bytes,
                "memory_semantics": "input_tensor_payload_not_peak_process_or_device_memory",
            },
            "seed_loading_stability": metrics.median_seed_loading_correlation,
            "donor_loading_stability": metrics.median_donor_loading_correlation,
            "sister_guide_correlation": metrics.median_sister_guide_correlation,
            "target_variance_fraction": metrics.target_variance_fraction,
            "inconsistent_target_fraction": metrics.inconsistent_target_fraction,
            "null_inclusion_rate": metrics.null_program_inclusion_rate,
            "null_replicate_families": metrics.null_replicate_families,
            "splits": [
                {
                    "kind": item.kind.value,
                    "eligible": item.eligible,
                    "passed": item.passed,
                    "improvement": item.improvement_over_best_baseline,
                    "gene_sign_accuracy": item.gene_sign_accuracy,
                    "gene_sign_coverage": item.gene_sign_coverage,
                    "predictive_nb_log_likelihood": item.predictive_nb_log_likelihood,
                    "common_dispersion_mean_prediction_score": (
                        item.common_dispersion_mean_prediction_score
                    ),
                }
                for item in metrics.splits
            ],
        },
    )
    log_path = root / "artifacts" / "training_log.json"
    _write_json(
        log_path,
        {
            "training_loss": metrics.reference_fit.training_loss,
            "validation_loss": metrics.reference_fit.validation_loss,
            "best_epoch": metrics.reference_fit.best_epoch,
            "stopped_early": metrics.reference_fit.stopped_early,
        },
    )
    environment_path = root / "artifacts" / "software_environment.json"
    _write_json(
        environment_path,
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "training_device": training_device,
        },
    )
    qualification = _identified(
        ProgramQualificationReceipt,
        {
            "schema_id": "credo.program_qualification_receipt",
            "schema_version": 1,
            "program_contract_id": contract.program_contract_id,
            "qualification_protocol_id": (qualification_protocol.qualification_protocol_id),
            "scope": scope,
            "status": (
                ProgramQualificationStatus.PASS_SCIENTIFIC
                if metrics.scientific_pass
                else ProgramQualificationStatus.FAIL_QUALIFICATION
            ),
            "split_evaluations": _split_contracts(metrics, dataset),
            "seed_stability_pass": metrics.seed_stability_pass,
            "donor_stability_pass": metrics.donor_stability_pass,
            "null_inclusion_calibrated": metrics.null_inclusion_calibrated,
            "sister_guide_consistency_pass": metrics.sister_guide_consistency_pass,
            "heldout_target_performance_pass": (metrics.heldout_target_performance_pass),
            "gene_sign_calibration_pass": metrics.gene_sign_calibration_pass,
            "external_pathway_concordance_available": False,
            "external_pathway_concordance_pass": None,
            "qualified_program_definition_ids": (
                tuple(
                    sorted(
                        item.program_definition_id for item in definitions if item.nonzero_genes > 0
                    )
                )
                if metrics.scientific_pass
                else ()
            ),
            "metrics_artifact": _artifact(
                root, metrics_path, "credo.program_qualification_metrics", "application/json"
            ),
            "training_log_artifact": _artifact(
                root, log_path, "credo.program_training_log", "application/json"
            ),
            "software_environment_artifact": _artifact(
                root, environment_path, "credo.software_environment", "application/json"
            ),
        },
        "qualification_receipt_id",
    )
    null_contract = _identified(
        ProgramNullContract,
        {
            "schema_id": "credo.program_null_contract",
            "schema_version": 1,
            "null_families": (
                "control_label_permutation",
                "target_within_checkpoint_permutation",
                "negative_binomial_no_program",
            ),
            "replicate_count": config.null_replicates,
            "maximum_program_inclusion_rate": config.maximum_null_inclusion_rate,
        },
        "null_contract_id",
    )
    return _identified(
        BiologicalProgramQualificationBundle,
        {
            "schema_id": "credo.biological_program_qualification_bundle",
            "schema_version": 1,
            "study_id": study_id,
            "capability_assessment_id": capability_assessment_id,
            "contract": contract,
            "qualification_protocol": qualification_protocol,
            "scope": scope,
            "program_definitions": tuple(definitions),
            "perturbation_effects": (effect,),
            "gene_effects": (gene_effect,),
            "guide_target_consistency": consistency,
            "uncertainty": uncertainty,
            "qualification": qualification,
            "null_contract": null_contract,
            "simulation_contract": None,
            "model_state_artifact": artifacts["model_state"],
        },
        "program_bundle_id",
    )


def qualify_and_publish_program_model(
    destination: Path,
    *,
    dataset: ProgramDataset,
    config: ProgramQualificationConfig,
    study_id: str,
    capability_assessment_id: str,
    scope: ScientificScope,
    perturbation_id: str,
    target_id: str,
    target_index: int,
    guide_ids: tuple[str, ...],
    guide_indices: tuple[int, ...],
    feature_order_sha256: str,
    guide_target_map_sha256: str,
    fit_row_ids_sha256: str,
    device: torch.device | str = "cpu",
) -> Path:
    """Run all qualification gates and atomically publish immutable artifacts."""

    if destination.exists():
        raise ContractError(f"Program qualification destination already exists: {destination}")
    if len(guide_ids) != len(guide_indices) or not guide_ids:
        raise ContractError("Guide IDs and indices must be nonempty and aligned.")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        metrics = qualify_program_model(dataset, config=config, device=device)
        bundle = _publish_bundle(
            staging,
            dataset=dataset,
            metrics=metrics,
            config=config,
            study_id=study_id,
            capability_assessment_id=capability_assessment_id,
            scope=scope,
            perturbation_id=perturbation_id,
            target_id=target_id,
            target_index=target_index,
            guide_ids=guide_ids,
            guide_indices=guide_indices,
            feature_order_sha256=feature_order_sha256,
            guide_target_map_sha256=guide_target_map_sha256,
            fit_row_ids_sha256=fit_row_ids_sha256,
        )
        bundle_path = staging / "program_qualification_bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n")
        verify_program_qualification(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def verify_program_qualification(path: Path) -> BiologicalProgramQualificationBundle:
    """Verify schema, identities, and every referenced artifact byte."""

    bundle_path = path / "program_qualification_bundle.json"
    if not bundle_path.is_file():
        raise IntegrityError("Program qualification bundle is missing.")
    bundle = BiologicalProgramQualificationBundle.model_validate_json(bundle_path.read_text())
    artifacts: set[ArtifactRef] = {
        bundle.model_state_artifact,
        bundle.qualification.metrics_artifact,
        bundle.qualification.training_log_artifact,
        bundle.qualification.software_environment_artifact,
        bundle.guide_target_consistency.per_target_artifact,
        bundle.uncertainty.seed_loading_stability_artifact,
        bundle.uncertainty.inclusion_probability_artifact,
        bundle.uncertainty.null_inclusion_artifact,
    }
    if bundle.contract.target_descriptor_artifact is not None:
        artifacts.add(bundle.contract.target_descriptor_artifact)
    if bundle.uncertainty.donor_loading_stability_artifact is not None:
        artifacts.add(bundle.uncertainty.donor_loading_stability_artifact)
    artifacts |= {item.loading_artifact for item in bundle.program_definitions}
    for effect_record in bundle.perturbation_effects:
        artifacts |= {
            effect_record.reference_activity,
            effect_record.target_activity,
            effect_record.guide_deviation_activity,
            effect_record.guide_efficiency,
        }
    for gene_record in bundle.gene_effects:
        artifacts |= {
            gene_record.log_fold_change_artifact,
            gene_record.standard_error_artifact,
            gene_record.sign_probability_artifact,
        }
    for artifact in artifacts:
        artifact_path = path / artifact.relative_uri
        if not artifact_path.is_file():
            raise IntegrityError(f"Program artifact is missing: {artifact.relative_uri}")
        if artifact_path.stat().st_size != artifact.size_bytes:
            raise IntegrityError(f"Program artifact size mismatch: {artifact.relative_uri}")
        if sha256_file(artifact_path) != artifact.sha256:
            raise IntegrityError(f"Program artifact digest mismatch: {artifact.relative_uri}")
    return bundle
