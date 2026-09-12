"""Canonical Python lifecycle API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel

from .canonical import sha256_file
from .compat.credo3 import verify_frozen_credo
from .compile import compile_problem
from .contracts import (
    AbundanceResult,
    BiologicalProgramQualificationBundle,
    CheckpointMultinomialDecoderContract,
    CheckpointMultinomialDecoderContractV1,
    ClaimAdjudication,
    ClaimRegistry,
    ClaimRegistryV1,
    CompiledRunContract,
    ComponentTestContract,
    ComponentTestReceipt,
    ComponentTestReceiptV2,
    CountLinkedProgramContract,
    CountRepresentationBundle,
    CountStoreManifest,
    DatasetCapabilityAssessment,
    EvaluationBundleManifest,
    FoldNativeCompactViewContractV1,
    FoldNativeCompactViewContractV2,
    G00CDecisionReceipt,
    G00CExecutionBundle,
    G00SourceAuthorityV1,
    G00SourceAuthorityV2,
    G00SourcePlaneV2Amendment,
    G00SourcePlaneV2AmendmentReceipt,
    G14MultiplicityContract,
    G14MultiplicityContractV1,
    G14RobustnessPlan,
    G14RobustnessPlanV1,
    G14SealContract,
    G14SealContractV1,
    GeneLevelEffect,
    GuideTargetConsistency,
    InferenceBundleManifest,
    IntegratedLoaderQualificationContractV1,
    IntegratedLoaderQualificationContractV2,
    IntegratedLoaderQualificationReceiptV1,
    IntegratedLoaderQualificationReceiptV2,
    LifecycleState,
    ParticleEngineQualificationBundle,
    ParticleEngineTestReceipt,
    PerturbationProgramEffect,
    PhysicalPoolConditionalReactionBundle,
    PhysicalPoolConditionalReactionReceipt,
    PooledFiniteMeasureBundle,
    PooledReactionLikelihoodBundle,
    PooledReactionLikelihoodReceipt,
    PreparedRepresentation,
    ProgramDefinition,
    ProgramNullContract,
    ProgramQualificationProtocol,
    ProgramQualificationReceipt,
    ProgramSimulationContract,
    ProgramUncertainty,
    RawCountMassNoiseAmendment,
    RawCountMassNoiseAmendmentReceipt,
    RawCountMassNoiseBundle,
    RawCountMassNoiseReceipt,
    ReactionRecoveryMetricAmendment,
    ReactionRecoveryMetricAmendmentV1,
    ReactionRecoveryQualificationBundle,
    ReactionRecoveryTestReceipt,
    ReactionRecoveryTestReceiptV1,
    ReactionRecoveryTestReceiptV3,
    ResolvedConfig,
    ScientificClaimRequest,
    SealedRunManifest,
    SelectionManifest,
    SemanticStudySnapshot,
    ShardedCountStoreManifest,
    SourcePlaneDerivationReceipt,
    StateSelectionCalibration,
    StateSelectionCalibrationResults,
    StudyEvidenceContract,
    TrajectoryResult,
    VerifyLevel,
    VirtualCanonicalCountStoreManifestV1,
    VirtualCanonicalCountStoreManifestV2,
)
from .data import build_pooled_finite_measures, verify_pooled_finite_measures
from .errors import ContractError, IntegrityError
from .evaluation import evaluate_run, seal_run
from .inference import V4Run, finalize_inference, open_inference_run
from .noise import (
    derive_raw_count_mass_noise_amendment,
    qualify_raw_count_mass_noise,
    verify_raw_count_mass_noise,
    verify_raw_count_mass_noise_amendment,
)
from .numerics import (
    qualify_particle_engine as run_particle_engine_qualification,
)
from .numerics import (
    verify_particle_engine_qualification,
)
from .persistence import LifecycleLedger, verify_directory
from .prepare import prepare_representation
from .reaction import (
    amend_reaction_recovery_metrics,
    qualify_physical_pool_conditional_reaction,
    qualify_pooled_reaction_likelihood,
    qualify_reaction_recovery,
    verify_physical_pool_conditional_reaction,
    verify_pooled_reaction_likelihood,
    verify_reaction_recovery_metric_amendment,
    verify_reaction_recovery_qualification,
)
from .reports import PerturbationDossier
from .representation import qualify_count_representation, verify_count_representation
from .store import CountStore
from .training import resume_training, train_model
from .training.calibration import run_state_selection_calibration


def _supported_preflight() -> None:
    """Bind every supported lifecycle call to the exact frozen dependency."""

    verify_frozen_credo()


def resolve_config(config_path: Path) -> ResolvedConfig:
    """Parse and fully default the strict run configuration."""

    _supported_preflight()
    from .prepare.pipeline import load_config

    return load_config(config_path)


def pool_finite_measures(
    destination: Path,
    *,
    cells: pd.DataFrame,
    guide_catalog: pd.DataFrame,
    source_checkpoint: str,
    terminal_checkpoint: str,
    feature_order_hashes: Mapping[str, str],
    minimum_source_cells: int = 1,
    mass_pseudocount: float = 0.5,
) -> Path:
    """Build and fully verify the cohort-neutral T00 pooled data bundle."""

    _supported_preflight()
    result = build_pooled_finite_measures(
        destination,
        cells=cells,
        guide_catalog=guide_catalog,
        source_checkpoint=source_checkpoint,
        terminal_checkpoint=terminal_checkpoint,
        feature_order_hashes=feature_order_hashes,
        minimum_source_cells=minimum_source_cells,
        mass_pseudocount=mass_pseudocount,
    )
    verify_pooled_finite_measures(result)
    return result


def qualify_representation(
    destination: Path,
    *,
    pooled_bundle: Path,
    count_store: Path,
    outer_folds: pd.DataFrame,
    **settings: Any,
) -> Path:
    """Fit, adjudicate, and fully verify the independent T01 component."""

    _supported_preflight()
    result = qualify_count_representation(
        destination,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
        outer_folds=outer_folds,
        **settings,
    )
    verify_count_representation(result, pooled_bundle=pooled_bundle, count_store=count_store)
    return result


def qualify_particle_engine(destination: Path) -> Path:
    """Run and fully verify the independent T04 fixed-truth qualification."""

    _supported_preflight()
    result = run_particle_engine_qualification(destination)
    verify_particle_engine_qualification(result)
    return result


def qualify_reaction(destination: Path) -> Path:
    """Run and fully verify the independent T07S learned-reaction test."""

    _supported_preflight()
    result = qualify_reaction_recovery(destination)
    verify_reaction_recovery_qualification(result)
    return result


def amend_reaction_metrics(destination: Path, *, parent: Path) -> Path:
    """Correct T07S interval units while preserving the immutable parent tensors."""

    _supported_preflight()
    result = amend_reaction_recovery_metrics(destination, parent=parent)
    verify_reaction_recovery_metric_amendment(result, parent=parent)
    return result


def qualify_pooled_reaction(
    destination: Path,
    *,
    pooled_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> Path:
    """Run and fully verify the frozen one-fold CPU T07R-A0 qualification."""

    _supported_preflight()
    settings = {
        "pooled_bundle": pooled_bundle,
        "t02a_amendment": t02a_amendment,
        "t07s_amendment": t07s_amendment,
        "fold_assignment": fold_assignment,
    }
    result = qualify_pooled_reaction_likelihood(destination, **settings)
    verify_pooled_reaction_likelihood(result, **settings)
    return result


def correct_physical_pool_reaction(
    destination: Path,
    *,
    pooled_bundle: Path,
    t02a_bundle: Path,
    t02a_amendment: Path,
    t07s_amendment: Path,
    fold_assignment: Path,
) -> Path:
    """Run and verify the one permitted exposed CPU denominator correction."""

    _supported_preflight()
    settings = {
        "pooled_bundle": pooled_bundle,
        "t02a_bundle": t02a_bundle,
        "t02a_amendment": t02a_amendment,
        "t07s_amendment": t07s_amendment,
        "fold_assignment": fold_assignment,
    }
    result = qualify_physical_pool_conditional_reaction(destination, **settings)
    verify_physical_pool_conditional_reaction(result, **settings)
    return result


def qualify_raw_noise(
    destination: Path,
    *,
    pooled_bundle: Path,
    count_store: Path,
    **settings: Any,
) -> Path:
    """Run and fully verify the independent T02A noise qualification."""

    _supported_preflight()
    result = qualify_raw_count_mass_noise(
        destination,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
        **settings,
    )
    verify_raw_count_mass_noise(
        result,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
    )
    return result


def verify_raw_noise(
    path: Path, *, pooled_bundle: Path, count_store: Path
) -> RawCountMassNoiseBundle:
    """Fully verify an existing T02A bundle and both immutable parents."""

    _supported_preflight()
    return verify_raw_count_mass_noise(
        path,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
    )


def amend_raw_noise_interpretation(
    destination: Path,
    *,
    t02a_bundle: Path,
    pooled_bundle: Path,
    count_store: Path,
) -> Path:
    """Derive and fully verify a semantic amendment without rerunning T02A."""

    _supported_preflight()
    return derive_raw_count_mass_noise_amendment(
        destination,
        t02a_bundle=t02a_bundle,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
    )


def verify_raw_noise_amendment(
    path: Path,
    *,
    t02a_bundle: Path,
    pooled_bundle: Path,
    count_store: Path,
) -> RawCountMassNoiseAmendment:
    """Verify a derived T02A interpretation amendment and its parent bundle."""

    _supported_preflight()
    return verify_raw_count_mass_noise_amendment(
        path,
        t02a_bundle=t02a_bundle,
        pooled_bundle=pooled_bundle,
        count_store=count_store,
    )


def validate_contract(path: Path) -> dict[str, Any]:
    """Validate canonical JSON syntax and reject non-object contracts."""

    _supported_preflight()
    from .canonical import canonical_json_bytes, sha256_file

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError("A contract must be a JSON object with schema_version.")
    canonical_json_bytes(payload)
    selected: type[BaseModel] | None
    evidence_contracts: dict[str, type[BaseModel]] = {
        "credo.study_evidence_contract": StudyEvidenceContract,
        "credo.dataset_capability_assessment": DatasetCapabilityAssessment,
        "credo.abundance_result": AbundanceResult,
        "credo.trajectory_result": TrajectoryResult,
        "credo.scientific_claim_request": ScientificClaimRequest,
        "credo.claim_adjudication": ClaimAdjudication,
        "credo.perturbation_dossier": PerturbationDossier,
        "credo.count_linked_program_contract": CountLinkedProgramContract,
        "credo.program_definition": ProgramDefinition,
        "credo.perturbation_program_effect": PerturbationProgramEffect,
        "credo.gene_level_effect": GeneLevelEffect,
        "credo.guide_target_consistency": GuideTargetConsistency,
        "credo.program_uncertainty": ProgramUncertainty,
        "credo.program_qualification_receipt": ProgramQualificationReceipt,
        "credo.program_null_contract": ProgramNullContract,
        "credo.program_qualification_protocol": ProgramQualificationProtocol,
        "credo.program_simulation_contract": ProgramSimulationContract,
        "credo.biological_program_qualification_bundle": (BiologicalProgramQualificationBundle),
    }
    if payload.get("schema_id") in evidence_contracts:
        selected = evidence_contracts[str(payload["schema_id"])]
    elif "calibration_id" in payload and "rows" in payload:
        selected = StateSelectionCalibrationResults
    else:
        if payload.get("backend") == "csr_hdf5_sharded":
            selected = ShardedCountStoreManifest
        elif payload.get("backend") in {
            "virtual_canonical_h5ad_csr_v1",
            "virtual_canonical_h5ad_csr_v2",
        }:
            selected = (
                VirtualCanonicalCountStoreManifestV1
                if payload.get("schema_version") == 1
                else VirtualCanonicalCountStoreManifestV2
            )
        elif "authority_id" in payload and "source_reconciliation_pass" in payload:
            selected = (
                G00SourceAuthorityV1 if payload.get("schema_version") == 1 else G00SourceAuthorityV2
            )
        elif "parent_g00a_v1_authority_id" in payload:
            selected = G00SourcePlaneV2Amendment
        elif "derived_g00a_v2_authority_id" in payload:
            selected = G00SourcePlaneV2AmendmentReceipt
        elif (
            "receipt_id" in payload
            and "records" in payload
            and "model_fitting_performed" in payload
        ):
            selected = SourcePlaneDerivationReceipt
        elif "bundle_id" in payload and "compact_payload" in payload:
            selected = G00CExecutionBundle
        elif "execution_bundle_id" in payload:
            selected = G00CDecisionReceipt
        elif "qualification_contract_id" in payload and "raw_rows_per_second_gate" in payload:
            selected = IntegratedLoaderQualificationContractV1
        elif "qualification_contract_id" in payload and "measurement_protocol_sha256" in payload:
            selected = IntegratedLoaderQualificationContractV2
        elif "receipt_id" in payload and "data_wait_fraction" in payload:
            selected = (
                IntegratedLoaderQualificationReceiptV1
                if payload.get("schema_version") == 1
                else IntegratedLoaderQualificationReceiptV2
            )
        elif "fold_view_id" in payload:
            selected = (
                FoldNativeCompactViewContractV1
                if payload.get("schema_version") == 1
                else FoldNativeCompactViewContractV2
            )
        elif "decoder_contract_id" in payload:
            selected = (
                CheckpointMultinomialDecoderContractV1
                if payload.get("schema_version") == 1
                else CheckpointMultinomialDecoderContract
            )
        elif "registry_id" in payload and "records" in payload:
            selected = ClaimRegistryV1 if payload.get("schema_version") == 1 else ClaimRegistry
        elif "plan_id" in payload and "axes" in payload:
            selected = (
                G14RobustnessPlanV1 if payload.get("schema_version") == 1 else G14RobustnessPlan
            )
        elif "g14_contract_id" in payload:
            selected = G14SealContractV1 if payload.get("schema_version") == 1 else G14SealContract
        elif "multiplicity_contract_id" in payload:
            selected = (
                G14MultiplicityContractV1
                if payload.get("schema_version") == 1
                else G14MultiplicityContract
            )
        elif payload.get("method") in {
            "complete_denominator_dm_reaction_recovery_v1",
            "complete_denominator_dm_reaction_recovery_v2",
        }:
            selected = ReactionRecoveryQualificationBundle
        elif payload.get("method") == "pooled_target_reaction_dm_likelihood_v1":
            selected = PooledReactionLikelihoodBundle
        elif payload.get("method") == "physical_pool_conditional_dm_likelihood_v2":
            selected = PhysicalPoolConditionalReactionBundle
        elif payload.get("method") == "t07s_interval_metric_amendment_v2":
            selected = ReactionRecoveryMetricAmendment
        elif payload.get("method") == "t07s_interval_metric_amendment_v1":
            selected = ReactionRecoveryMetricAmendmentV1
        elif payload.get("schema_version") == 3 and "r0_false_promotion_guard_pass" in payload:
            selected = ReactionRecoveryTestReceiptV3
        elif "r0_false_promotion_guard_pass" in payload:
            selected = ReactionRecoveryTestReceipt
        elif "r0_false_selection_guard_pass" in payload:
            selected = ReactionRecoveryTestReceiptV1
        elif "estimator_parity_pass" in payload and "real_pooled_noninferiority_pass" in payload:
            selected = PooledReactionLikelihoodReceipt
        elif "factorization_pass" in payload and "predictive_decision" in payload:
            selected = PhysicalPoolConditionalReactionReceipt
        else:
            discriminators = (
                ("compiled_run_id", CompiledRunContract),
                ("qualification_id", ParticleEngineQualificationBundle),
                ("noise_id", RawCountMassNoiseBundle),
                ("parent_bundle_verified", RawCountMassNoiseAmendmentReceipt),
                ("amendment_id", RawCountMassNoiseAmendment),
                ("representation_id", CountRepresentationBundle),
                ("pooled_data_id", PooledFiniteMeasureBundle),
                ("test_contract_id", ComponentTestContract),
                (
                    "ou_largest_grid_variance_relative_error",
                    ParticleEngineTestReceipt,
                ),
                ("interval_log_mass_rmse_q95", RawCountMassNoiseReceipt),
                ("receipt_role", ComponentTestReceiptV2),
                ("receipt_id", ComponentTestReceipt),
                ("prepared_id", PreparedRepresentation),
                ("evaluation_id", EvaluationBundleManifest),
                ("run_id", InferenceBundleManifest),
                ("sealed_id", SealedRunManifest),
                ("selection_id", SelectionManifest),
                ("calibration_id", StateSelectionCalibration),
                ("store_id", CountStoreManifest),
                ("study_id", SemanticStudySnapshot),
            )
            selected = next((model for field, model in discriminators if field in payload), None)
    if selected is None:
        raise ValueError("Unknown contract discriminator.")
    selected.model_validate(payload)
    public_name = {
        G00SourceAuthorityV1: "G00SourceAuthority",
        VirtualCanonicalCountStoreManifestV1: "VirtualCanonicalCountStoreManifest",
        FoldNativeCompactViewContractV1: "FoldNativeCompactViewContract",
        IntegratedLoaderQualificationContractV1: "IntegratedLoaderQualificationContract",
        IntegratedLoaderQualificationReceiptV1: "IntegratedLoaderQualificationReceipt",
        G00SourceAuthorityV2: "G00SourceAuthority",
        VirtualCanonicalCountStoreManifestV2: "VirtualCanonicalCountStoreManifest",
        FoldNativeCompactViewContractV2: "FoldNativeCompactViewContract",
        IntegratedLoaderQualificationContractV2: "IntegratedLoaderQualificationContract",
        IntegratedLoaderQualificationReceiptV2: "IntegratedLoaderQualificationReceipt",
        G00SourcePlaneV2Amendment: "G00SourcePlaneV2Amendment",
        G00SourcePlaneV2AmendmentReceipt: "G00SourcePlaneV2AmendmentReceipt",
        G00CExecutionBundle: "G00CExecutionBundle",
        G00CDecisionReceipt: "G00CDecisionReceipt",
        SourcePlaneDerivationReceipt: "SourcePlaneDerivationReceipt",
    }.get(selected, selected.__name__)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload["schema_version"],
        "contract_type": public_name,
    }


def estimate(config_path: Path) -> dict[str, Any]:
    """Return a conservative plan-size receipt without reading endpoints."""

    config = resolve_config(config_path)
    model = config.model
    state_parameters = model.state_dim * (1 + model.target_count)
    measure_parameters = model.pool_count + model.target_count + model.pool_count
    context_parameters = (
        model.context_rank * (2 * model.pool_count + model.target_count + model.state_dim)
        if config.intent.value == "count_context"
        else 0
    )
    return {
        "schema_version": 1,
        "intent": config.intent.value,
        "approximate_trainable_parameters": state_parameters
        + measure_parameters
        + context_parameters,
        "training_updates": config.training.max_updates,
        "evaluation_particle_steps_per_series": config.evaluation.particles
        * config.evaluation.steps,
        "hard_output_bytes": config.evaluation.output_bytes_limit,
    }


def _workspace(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text())
    return (config_path.parent.resolve() / payload["workspace"]).resolve()


def _ledger(config_path: Path) -> LifecycleLedger:
    return LifecycleLedger(_workspace(config_path) / "ledger" / "events.jsonl")


def _trained_artifact_id(training_root: Path) -> str:
    selection = training_root / "selection.json"
    if selection.exists():
        return str(json.loads(selection.read_text())["selected_checkpoint_id"])
    latest = json.loads((training_root / "checkpoints" / "latest.json").read_text())
    return str(latest["checkpoint_id"])


def prepare(config_path: Path) -> Path:
    _supported_preflight()
    result = prepare_representation(config_path)
    manifest = json.loads((result / "prepared.json").read_text())
    _ledger(config_path).transition(LifecycleState.PREPARED, artifact_id=manifest["prepared_id"])
    return result


def compile_run(config_path: Path) -> Path:
    _supported_preflight()
    result = compile_problem(config_path)
    manifest = json.loads((result / "contract.json").read_text())
    _ledger(config_path).transition(
        LifecycleState.COMPILED, artifact_id=manifest["compiled_run_id"]
    )
    return result


def calibrate_state_selection(
    config_path: Path,
    output_root: Path,
    *,
    permutation_seed_start: int,
    optimizer_seed_start: int,
    initialization_seed_start: int,
    repeats_per_null: int,
    device: str = "cpu",
    calibration_stage: Literal["development", "locked_audit"] = "development",
    development_calibration: Path | None = None,
) -> tuple[Path, Path]:
    """Run and publish fixed-split pooled null fits before pilot compilation."""

    _supported_preflight()
    return run_state_selection_calibration(
        config_path,
        output_root,
        repeats_per_null=repeats_per_null,
        permutation_seed_start=permutation_seed_start,
        optimizer_seed_start=optimizer_seed_start,
        initialization_seed_start=initialization_seed_start,
        device=device,
        calibration_stage=calibration_stage,
        development_calibration=development_calibration,
    )


def train(config_path: Path, *, device: str | None = None) -> Path:
    _supported_preflight()
    result = train_model(config_path, device=device)
    _ledger(config_path).transition(
        LifecycleState.TRAINED, artifact_id=_trained_artifact_id(result)
    )
    return result


def fork(config_path: Path, *, from_checkpoint: Path, device: str | None = None) -> Path:
    """Start a new compiled attempt from compatible inference weights only."""

    _supported_preflight()
    from .prepare.pipeline import load_config

    if load_config(config_path).model.source_target_interaction_rank:
        raise ContractError(
            "Null-guarded source-target pilots prohibit checkpoint forking; "
            "update 0 must be a fresh exact null."
        )
    result = train_model(config_path, device=device, initial_checkpoint=from_checkpoint)
    _ledger(config_path).transition(
        LifecycleState.TRAINED,
        artifact_id=_trained_artifact_id(result),
        details={"fork_parent": str(from_checkpoint)},
    )
    return result


def resume(config_path: Path, *, device: str | None = None) -> Path:
    _supported_preflight()
    result = resume_training(config_path, device=device)
    if _ledger(config_path).state() is LifecycleState.COMPILED:
        _ledger(config_path).transition(
            LifecycleState.TRAINED, artifact_id=_trained_artifact_id(result)
        )
    return result


def finalize(config_path: Path) -> Path:
    _supported_preflight()
    result = finalize_inference(config_path)
    manifest = json.loads((result / "inference.json").read_text())
    _ledger(config_path).transition(LifecycleState.FINALIZED, artifact_id=manifest["run_id"])
    return result


def open_run(path: Path, *, device: str = "cpu", verify: str = "full") -> V4Run:
    _supported_preflight()
    return open_inference_run(path, device=device, verify=verify)


def evaluate(config_path: Path, *, device: str = "cpu") -> Path:
    _supported_preflight()
    result = evaluate_run(config_path, device=device)
    manifest = json.loads((result / "evaluation.json").read_text())
    _ledger(config_path).transition(LifecycleState.EVALUATED, artifact_id=manifest["evaluation_id"])
    return result


def seal(config_path: Path) -> Path:
    _supported_preflight()
    result = seal_run(config_path)
    manifest = json.loads((result / "sealed.json").read_text())
    _ledger(config_path).transition(LifecycleState.SEALED, artifact_id=manifest["sealed_id"])
    return result


def verify(path: Path, *, level: str | VerifyLevel = VerifyLevel.CONTENT) -> dict[str, Any]:
    _supported_preflight()
    selected = VerifyLevel(level)
    result: dict[str, Any] = {"root": str(path), "level": selected.value}
    result["manifest"] = verify_directory(path)
    workspace = path.parent
    if (path / "sealed.json").exists():
        sealed = SealedRunManifest.model_validate_json((path / "sealed.json").read_text())
        inference_root = workspace / "inference"
        evaluation_root = workspace / "evaluation"
        verify_directory(inference_root)
        verify_directory(evaluation_root)
        if sha256_file(inference_root / "inference.json") != sealed.inference.sha256:
            raise IntegrityError("Sealed inference reference does not match sibling bundle.")
        if len(sealed.evaluations) != 1 or (
            sha256_file(evaluation_root / "evaluation.json") != sealed.evaluations[0].sha256
        ):
            raise IntegrityError("Sealed evaluation reference does not match sibling bundle.")
        if sha256_file(path / "claim-audit.json") != sealed.claim_audit.sha256:
            raise IntegrityError("Sealed claim-audit reference mismatch.")
    if selected in {VerifyLevel.CONTENT, VerifyLevel.RELOAD, VerifyLevel.RESUME, VerifyLevel.FULL}:
        if (workspace / "compiled" / "config.json").exists():
            config = json.loads((workspace / "compiled" / "config.json").read_text())
            config_root = Path(config.get("_config_root", workspace.parent))
            count_path = config_root / config["count_store"]
            if not count_path.exists():
                # Normal resolved configs use a path relative to the original
                # config root, which is the workspace parent in synthetic runs.
                count_path = workspace.parent / config["count_store"]
            if count_path.exists():
                result["count_store"] = (
                    CountStore(count_path).verify(full=True).model_dump(mode="json")
                )
    if selected in {VerifyLevel.RELOAD, VerifyLevel.FULL} and (workspace / "inference").exists():
        run = open_inference_run(workspace / "inference", verify="full")
        mean, mass, _ = run.terminal(particles=8, steps=2)
        result["reload"] = {"series": len(mean), "finite_mass": bool((mass > 0).all())}
    return result
