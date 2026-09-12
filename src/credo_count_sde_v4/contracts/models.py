"""Strict v4 contract models.

All contracts reject unknown fields and resolve defaults before identity is
computed. Paths are portable relative URIs; exact bytes remain authoritative.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..canonical import contract_id, validate_relative_uri
from ..errors import ContractError

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    def identity(self, id_field: str | None = None) -> str:
        return contract_id(self, id_field=id_field)


class RunIntent(StrEnum):
    COUNT_STATE = "count_state"
    COUNT_MEASURE = "count_measure"
    COUNT_CONTEXT = "count_context"


class VerifyLevel(StrEnum):
    MANIFEST = "manifest"
    CONTENT = "content"
    RELOAD = "reload"
    RESUME = "resume"
    FULL = "full"


class LifecycleState(StrEnum):
    RESOLVED = "resolved"
    PREPARED = "prepared"
    COMPILED = "compiled"
    TRAINED = "trained"
    FINALIZED = "finalized"
    EVALUATED = "evaluated"
    SEALED = "sealed"


class EvidenceRole(StrEnum):
    DEVELOPMENT = "development"
    SEALED = "sealed"
    EXTERNAL_CONFIRMATION = "external_confirmation"


class ArtifactRef(StrictModel):
    schema_id: str
    schema_version: int = Field(ge=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: str
    relative_uri: str

    @field_validator("relative_uri")
    @classmethod
    def safe_uri(cls, value: str) -> str:
        return validate_relative_uri(value)


class FeatureKey(StrictModel):
    namespace: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    namespace_version: str = Field(min_length=1)


class InformationSet(StrictModel):
    information_set_id: str
    fit_rows: tuple[int, ...]
    validation_rows: tuple[int, ...] = ()
    query_rows: tuple[int, ...] = ()
    protected_rows: tuple[int, ...] = ()

    @model_validator(mode="after")
    def disjoint(self) -> InformationSet:
        fit, validation, protected = map(
            set, (self.fit_rows, self.validation_rows, self.protected_rows)
        )
        if fit & protected or validation & protected:
            raise ValueError("Fit/validation rows overlap protected rows.")
        if fit & validation:
            raise ValueError("Fit and validation rows must be disjoint.")
        return self


class SplitContract(StrictModel):
    schema_version: int = 1
    split_id: str
    training_units: tuple[str, ...]
    inner_validation_units: tuple[str, ...] = ()
    outer_evaluation_units: tuple[str, ...]
    grouping_unit: str

    @model_validator(mode="after")
    def unit_disjointness(self) -> SplitContract:
        groups = [
            set(self.training_units),
            set(self.inner_validation_units),
            set(self.outer_evaluation_units),
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Split unit sets must be pairwise disjoint.")
        return self


class RowIndex(StrictModel):
    schema_version: int = 1
    row_ids_hash: Sha256
    row_count: int = Field(gt=0)
    dtype: Literal["int64"] = "int64"
    unique: Literal[True] = True


class FeatureIndex(StrictModel):
    schema_version: int = 1
    features: tuple[FeatureKey, ...]
    ordered_hash: Sha256

    @model_validator(mode="after")
    def unique_features(self) -> FeatureIndex:
        keys = [(item.namespace, item.namespace_version, item.feature_id) for item in self.features]
        if len(keys) != len(set(keys)):
            raise ValueError("Composite feature keys must be unique.")
        return self


class PooledFiniteMeasureBundle(StrictModel):
    """Immutable T00 pooled guide-by-checkpoint finite-measure data contract."""

    schema_version: int = 1
    pooled_data_id: str
    sample_id: Literal["pooled"] = "pooled"
    source_checkpoint: str = Field(min_length=1)
    terminal_checkpoint: str = Field(min_length=1)
    feature_order_hash: Sha256
    input_cell_universe_hash: Sha256
    retained_cell_universe_hash: Sha256
    excluded_cell_universe_hash: Sha256
    source_eligibility_min_cells: int = Field(ge=1)
    eligibility_uses_terminal_counts: Literal[False] = False
    mass_pseudocount: float = Field(default=0.5, gt=0)
    retained_cells: int = Field(ge=1)
    retained_guides: int = Field(ge=1)
    targeting_guides: int = Field(ge=1)
    control_guides: int = Field(ge=1)
    perturbation_targets: int = Field(ge=1)
    cells: ArtifactRef
    guide_catalog: ArtifactRef
    eligibility: ArtifactRef
    finite_measures: ArtifactRef
    per_guide_metrics: ArtifactRef
    per_target_metrics: ArtifactRef

    @model_validator(mode="after")
    def validate_identity(self) -> PooledFiniteMeasureBundle:
        expected = self.identity(id_field="pooled_data_id")
        if self.pooled_data_id != expected:
            raise ValueError(f"pooled_data_id mismatch: expected {expected}.")
        if self.source_checkpoint == self.terminal_checkpoint:
            raise ValueError("Source and terminal checkpoints must differ.")
        if self.targeting_guides + self.control_guides != self.retained_guides:
            raise ValueError("Targeting/control guide counts do not cover the retained catalog.")
        return self


class CountRepresentationBundle(StrictModel):
    """Immutable T01 count-native representation qualification bundle."""

    schema_version: int = 1
    representation_id: str
    pooled_data_id: str
    method: Literal["multinomial_hellinger_pca_v1", "multinomial_centered_hellinger_pca_v2"]
    feature_index_hash: Sha256
    count_store_sha256: Sha256
    dimensions: tuple[int, ...]
    selected_dimensions: dict[str, int]
    outer_folds: tuple[str, ...]
    selection_uses_terminal_outcomes: Literal[False] = False
    dynamics_gradients_enabled: Literal[False] = False
    fit_checkpoint: str
    protected_checkpoint: str
    fold_index: ArtifactRef
    encoder_state: ArtifactRef
    candidate_metrics: ArtifactRef
    per_guide_metrics: ArtifactRef
    per_target_metrics: ArtifactRef
    support_metrics: ArtifactRef
    null_calibration: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_representation(self) -> CountRepresentationBundle:
        expected = self.identity(id_field="representation_id")
        if self.representation_id != expected:
            raise ValueError(f"representation_id mismatch: expected {expected}.")
        if not self.dimensions or tuple(sorted(set(self.dimensions))) != self.dimensions:
            raise ValueError("Representation dimensions must be unique and increasing.")
        if any(value <= 0 for value in self.dimensions):
            raise ValueError("Representation dimensions must be positive.")
        if set(self.selected_dimensions) != set(self.outer_folds):
            raise ValueError("Every outer fold must have exactly one selected dimension.")
        if any(value not in {0, *self.dimensions} for value in self.selected_dimensions.values()):
            raise ValueError("A selected dimension is outside the frozen candidate set.")
        if self.fit_checkpoint == self.protected_checkpoint:
            raise ValueError("Fit and protected checkpoints must differ.")
        return self


class CheckpointMultinomialDecoderContractV1(StrictModel):
    """Read-only decoder contract retained for immutable dev28 evidence."""

    schema_version: Literal[1] = 1
    decoder_contract_id: str
    equation: Literal["softmax(checkpoint_intercept + latent_weights @ z)"] = (
        "softmax(checkpoint_intercept + latent_weights @ z)"
    )
    intercept_axis: Literal["checkpoint"] = "checkpoint"
    dimension_zero_null: Literal["checkpoint_global_frequency"] = "checkpoint_global_frequency"
    checkpoints: tuple[str, ...]
    features: int = Field(gt=0)
    latent_dimension: int = Field(ge=0)
    pseudocount: float = Field(default=0.5, gt=0)
    fit_row_ids_hash: Sha256
    heldout_donor_outcomes_used: Literal[False] = False
    guide_parameters: Literal[False] = False
    target_parameters: Literal[False] = False
    heldout_donor_parameters: Literal[False] = False

    @model_validator(mode="after")
    def validate_checkpoint_decoder_v1(self) -> CheckpointMultinomialDecoderContractV1:
        if not self.checkpoints or tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("Checkpoint decoder checkpoints must be unique and sorted.")
        if any(not value for value in self.checkpoints):
            raise ValueError("Checkpoint decoder identifiers cannot be empty.")
        expected = self.identity(id_field="decoder_contract_id")
        if self.decoder_contract_id != expected:
            raise ValueError(f"decoder_contract_id mismatch: expected {expected}.")
        return self


class CheckpointMultinomialDecoderContract(StrictModel):
    """Leakage-safe G04 decoder and its dimension-zero null."""

    schema_version: Literal[2] = 2
    decoder_contract_id: str
    equation: Literal["softmax(checkpoint_intercept + latent_weights @ z)"] = (
        "softmax(checkpoint_intercept + latent_weights @ z)"
    )
    intercept_axis: Literal["checkpoint"] = "checkpoint"
    dimension_zero_null: Literal["checkpoint_global_frequency"] = "checkpoint_global_frequency"
    checkpoints: tuple[str, ...]
    physical_time_hours: tuple[float, ...]
    checkpoint_order_hash: Sha256
    features: int = Field(gt=0)
    latent_dimension: int = Field(ge=0)
    pseudocount: float = Field(default=0.5, gt=0)
    fit_row_ids_hash: Sha256
    heldout_donor_outcomes_used: Literal[False] = False
    guide_parameters: Literal[False] = False
    target_parameters: Literal[False] = False
    heldout_donor_parameters: Literal[False] = False

    @model_validator(mode="after")
    def validate_checkpoint_decoder(self) -> CheckpointMultinomialDecoderContract:
        if not self.checkpoints or len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("Checkpoint decoder checkpoints must be unique.")
        if any(not value for value in self.checkpoints):
            raise ValueError("Checkpoint decoder identifiers cannot be empty.")
        if len(self.physical_time_hours) != len(self.checkpoints):
            raise ValueError("Every checkpoint requires one physical time.")
        if any(not math.isfinite(value) for value in self.physical_time_hours) or any(
            right <= left
            for left, right in zip(
                self.physical_time_hours, self.physical_time_hours[1:], strict=False
            )
        ):
            raise ValueError("Checkpoint physical times must be finite and strictly increasing.")
        expected_order_hash = contract_id(
            {
                "checkpoints": self.checkpoints,
                "physical_time_hours": self.physical_time_hours,
            }
        )
        if self.checkpoint_order_hash != expected_order_hash:
            raise ValueError("Checkpoint chronology hash mismatch.")
        expected = self.identity(id_field="decoder_contract_id")
        if self.decoder_contract_id != expected:
            raise ValueError(f"decoder_contract_id mismatch: expected {expected}.")
        return self


class ParticleEngineQualificationBundle(StrictModel):
    """Immutable T04 numerical particle-engine qualification bundle."""

    schema_version: int = 1
    qualification_id: str
    test_contract_id: str
    method: Literal["streaming_euler_maruyama_v1"]
    environment_hash: Sha256
    particle_grid: tuple[int, ...]
    step_grid: tuple[int, ...]
    seed_count: int = Field(ge=1)
    deterministic_drift: ArtifactRef
    ou_grid: ArtifactRef
    reaction_mass: ArtifactRef
    ecology: ArtifactRef
    lifecycle: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_qualification(self) -> ParticleEngineQualificationBundle:
        expected = self.identity(id_field="qualification_id")
        if self.qualification_id != expected:
            raise ValueError(f"qualification_id mismatch: expected {expected}.")
        if tuple(sorted(set(self.particle_grid))) != self.particle_grid:
            raise ValueError("Particle qualification grid must be increasing and unique.")
        if tuple(sorted(set(self.step_grid))) != self.step_grid:
            raise ValueError("Step qualification grid must be increasing and unique.")
        if any(value <= 0 for value in (*self.particle_grid, *self.step_grid)):
            raise ValueError("Particle and step qualification grids must be positive.")
        return self


class ParticleEngineTestReceipt(StrictModel):
    """Complete T04 fixed-truth decision surface."""

    schema_version: int = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    deterministic_drift_max_abs_error: float = Field(ge=0)
    drift_refinement_pass: bool
    ou_mean_within_two_standard_errors: bool
    ou_largest_grid_variance_relative_error: float = Field(ge=0)
    ou_convergence_pass: bool
    reaction_max_relative_error: float = Field(ge=0)
    ecology_absolute_weight_max_error: float = Field(ge=0)
    normalized_context_negative_control_detected: bool
    stabilized_log_weight_pass: bool
    deterministic_replay_pass: bool
    interrupted_resume_pass: bool
    no_guide_switching_pass: bool
    normalized_particle_weights_pass: bool
    declared_mass_pass: bool
    capacity_probe_exclusion_pass: bool
    protected_metrics_pass: bool
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_t04_receipt(self) -> ParticleEngineTestReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        gates = (
            self.drift_refinement_pass,
            self.ou_mean_within_two_standard_errors,
            self.ou_convergence_pass,
            self.normalized_context_negative_control_detected,
            self.stabilized_log_weight_pass,
            self.deterministic_replay_pass,
            self.interrupted_resume_pass,
            self.no_guide_switching_pass,
            self.normalized_particle_weights_pass,
            self.declared_mass_pass,
            self.capacity_probe_exclusion_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T04 receipt must satisfy every fixed numerical gate.")
        return self

    @property
    def stabilized_absolute_log_mass_pass(self) -> bool:
        """Correct interpretation of the frozen legacy wire-field name."""

        return self.stabilized_log_weight_pass


class ReactionRecoveryQualificationBundle(StrictModel):
    """Immutable T07S learned constant-reaction qualification bundle."""

    schema_version: int = 1
    qualification_id: str
    test_contract_id: str
    method: Literal[
        "complete_denominator_dm_reaction_recovery_v1",
        "complete_denominator_dm_reaction_recovery_v2",
    ]
    environment_hash: Sha256
    null_calibration_repeats: int = Field(ge=59)
    null_audit_repeats: int = Field(ge=59)
    candidate_updates: tuple[int, ...]
    target_count: int = Field(ge=4)
    pool_count: int = Field(ge=2)
    null_refits: ArtifactRef
    null_model_effects: ArtifactRef
    recovery_curve: ArtifactRef
    recovery_series: ArtifactRef
    target_metrics: ArtifactRef
    bootstrap_target_draws: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_reaction_qualification(self) -> ReactionRecoveryQualificationBundle:
        expected = self.identity(id_field="qualification_id")
        if self.qualification_id != expected:
            raise ValueError(f"qualification_id mismatch: expected {expected}.")
        if not self.candidate_updates or self.candidate_updates[0] != 0:
            raise ValueError("Reaction recovery must retain update 0 as a candidate.")
        if tuple(sorted(set(self.candidate_updates))) != self.candidate_updates:
            raise ValueError("Reaction candidate updates must be increasing and unique.")
        return self


class ReactionRecoveryTestReceiptV1(StrictModel):
    """Legacy dev23 T07S receipt retained for immutable-parent verification."""

    schema_version: int = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    r0_calibration_repeats: int = Field(ge=59)
    r0_audit_repeats: int = Field(ge=59)
    r0_required_margin: float = Field(ge=0)
    r0_audit_false_promotions: int = Field(ge=0)
    r0_audit_false_promotion_upper_95: float = Field(ge=0, le=1)
    r0_false_selection_guard_pass: bool
    r1_selected_update: int = Field(ge=0)
    r1_post_selection_refit_pass: bool
    r1_point_delta: float
    r1_target_bootstrap_interval: tuple[float, float]
    r1_margin_pass: bool
    r1_reaction_rmse: float = Field(ge=0)
    r1_sign_accuracy: float = Field(ge=0, le=1)
    r1_channel_activity: float = Field(ge=0)
    r1_channel_activity_pass: bool
    weighted_gauge_max_abs_error: float = Field(ge=0)
    rollout_mass_max_relative_error: float = Field(ge=0)
    probability_normalization_max_abs_error: float = Field(ge=0)
    control_target_mask_max_abs_error: float = Field(ge=0)
    fixed_channel_max_abs_change: float = Field(ge=0)
    protected_metrics_pass: bool
    update_zero_selectable: Literal[True] = True
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_reaction_receipt(self) -> ReactionRecoveryTestReceiptV1:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.r1_target_bootstrap_interval
        if lower > upper:
            raise ValueError("Reaction target-bootstrap interval must be ordered.")
        gates = (
            self.r0_false_selection_guard_pass,
            self.r1_post_selection_refit_pass,
            self.r1_margin_pass,
            self.r1_channel_activity_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T07S receipt must satisfy every recovery gate.")
        return self


class ReactionRecoveryTestReceipt(StrictModel):
    """Duration-correct dev24 T07S R0-null and R1-nonzero decision surface."""

    schema_version: Literal[2] = 2
    receipt_id: str
    test_contract_id: str
    parent_qualification_id: str | None = None
    status: Literal["pass", "fail_retired"]
    r0_calibration_repeats: int = Field(ge=59)
    r0_audit_repeats: int = Field(ge=59)
    r0_required_margin: float = Field(ge=0)
    r0_calibration_nonzero_checkpoint_selections: int = Field(ge=0)
    r0_audit_nonzero_checkpoint_selections: int = Field(ge=0)
    r0_audit_nonzero_checkpoint_selection_rate: float = Field(ge=0, le=1)
    r0_audit_false_promotions: int = Field(ge=0)
    r0_audit_false_promotion_upper_95: float = Field(ge=0, le=1)
    r0_false_promotion_guard_pass: bool
    r1_metric_estimand: Literal["centered_interval_log_frequency_change"]
    r1_selected_update: int = Field(ge=0)
    r1_post_selection_refit_pass: bool
    r1_interval_effect_target_balanced_rmse: float = Field(ge=0)
    r1_zero_baseline_interval_effect_target_balanced_rmse: float = Field(ge=0)
    r1_point_delta: float
    r1_target_bootstrap_interval: tuple[float, float]
    r1_margin_pass: bool
    r1_reaction_rmse: float = Field(ge=0)
    r1_sign_accuracy: float = Field(ge=0, le=1)
    r1_channel_activity: float = Field(ge=0)
    r1_channel_activity_pass: bool
    weighted_gauge_max_abs_error: float = Field(ge=0)
    rollout_mass_max_relative_error: float = Field(ge=0)
    probability_normalization_max_abs_error: float = Field(ge=0)
    control_target_mask_max_abs_error: float = Field(ge=0)
    fixed_channel_max_abs_change: float = Field(ge=0)
    protected_metrics_pass: bool
    update_zero_selectable: Literal[True] = True
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_reaction_receipt(self) -> ReactionRecoveryTestReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.r1_target_bootstrap_interval
        if lower > upper:
            raise ValueError("Reaction target-bootstrap interval must be ordered.")
        if self.r0_audit_nonzero_checkpoint_selections > self.r0_audit_repeats:
            raise ValueError("Audit nonzero-checkpoint selections exceed the audit repeats.")
        expected_rate = self.r0_audit_nonzero_checkpoint_selections / self.r0_audit_repeats
        if abs(self.r0_audit_nonzero_checkpoint_selection_rate - expected_rate) > 1e-15:
            raise ValueError("Audit nonzero-checkpoint selection rate is inconsistent.")
        gates = (
            self.r0_false_promotion_guard_pass,
            self.r1_post_selection_refit_pass,
            self.r1_margin_pass,
            self.r1_channel_activity_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T07S receipt must satisfy every recovery gate.")
        return self


class ReactionRecoveryMetricAmendmentV1(StrictModel):
    """Legacy dev24 R1-only metric amendment retained for contract validation."""

    schema_version: Literal[1] = 1
    amendment_id: str
    method: Literal["t07s_interval_metric_amendment_v1"]
    parent_qualification_id: str
    parent_bundle_sha256: Sha256
    parent_artifacts_manifest_sha256: Sha256
    metric_estimand: Literal["centered_interval_log_frequency_change"]
    false_promotion_upper_limit: float = Field(ge=0.05, le=0.05)
    environment_hash: Sha256
    null_refits: ArtifactRef
    null_model_effects: ArtifactRef
    recovery_curve: ArtifactRef
    recovery_series: ArtifactRef
    target_metrics: ArtifactRef
    bootstrap_target_draws: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef
    component_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_reaction_metric_amendment(self) -> ReactionRecoveryMetricAmendmentV1:
        expected = self.identity(id_field="amendment_id")
        if self.amendment_id != expected:
            raise ValueError(f"amendment_id mismatch: expected {expected}.")
        return self


class ReactionRecoveryTestReceiptV3(StrictModel):
    """Unified dev25 T07S decision surface with duration-correct R0 and R1."""

    schema_version: Literal[3] = 3
    receipt_id: str
    test_contract_id: str
    parent_qualification_id: str
    status: Literal["pass", "fail_retired"]
    r0_calibration_repeats: int = Field(ge=59)
    r0_audit_repeats: int = Field(ge=59)
    r0_required_margin: float = Field(ge=0)
    r0_calibration_nonzero_checkpoint_selections: int = Field(ge=0)
    r0_audit_nonzero_checkpoint_selections: int = Field(ge=0)
    r0_audit_nonzero_checkpoint_selection_rate: float = Field(ge=0, le=1)
    r0_audit_false_promotions: int = Field(ge=0)
    r0_audit_false_promotion_upper_95: float = Field(ge=0, le=1)
    r0_false_promotion_guard_pass: bool
    r0_metric_estimand: Literal["centered_interval_log_frequency_change"]
    legacy_null_refits: ArtifactRef
    corrected_null_interval_refits: ArtifactRef
    optimizer_rerun: Literal[False] = False
    r1_metric_estimand: Literal["centered_interval_log_frequency_change"]
    r1_selected_update: int = Field(ge=0)
    r1_post_selection_refit_pass: bool
    r1_interval_effect_target_balanced_rmse: float = Field(ge=0)
    r1_zero_baseline_interval_effect_target_balanced_rmse: float = Field(ge=0)
    r1_point_delta: float
    r1_target_bootstrap_interval: tuple[float, float]
    r1_margin_pass: bool
    r1_reaction_rmse: float = Field(ge=0)
    r1_sign_accuracy: float = Field(ge=0, le=1)
    r1_channel_activity: float = Field(ge=0)
    r1_channel_activity_pass: bool
    weighted_gauge_max_abs_error: float = Field(ge=0)
    rollout_mass_max_relative_error: float = Field(ge=0)
    probability_normalization_max_abs_error: float = Field(ge=0)
    control_target_mask_max_abs_error: float = Field(ge=0)
    fixed_channel_max_abs_change: float = Field(ge=0)
    protected_metrics_pass: bool
    update_zero_selectable: Literal[True] = True
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_reaction_receipt(self) -> ReactionRecoveryTestReceiptV3:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.r1_target_bootstrap_interval
        if lower > upper:
            raise ValueError("Reaction target-bootstrap interval must be ordered.")
        if self.r0_audit_nonzero_checkpoint_selections > self.r0_audit_repeats:
            raise ValueError("Audit nonzero-checkpoint selections exceed the audit repeats.")
        expected_rate = self.r0_audit_nonzero_checkpoint_selections / self.r0_audit_repeats
        if abs(self.r0_audit_nonzero_checkpoint_selection_rate - expected_rate) > 1e-15:
            raise ValueError("Audit nonzero-checkpoint selection rate is inconsistent.")
        gates = (
            self.r0_false_promotion_guard_pass,
            self.r1_post_selection_refit_pass,
            self.r1_margin_pass,
            self.r1_channel_activity_pass,
            self.protected_metrics_pass,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T07S receipt must satisfy every recovery gate.")
        return self


class ReactionRecoveryMetricAmendment(StrictModel):
    """Immutable duration-correct R0/R1 evidence amendment over dev23 T07S."""

    schema_version: Literal[2] = 2
    amendment_id: str
    method: Literal["t07s_interval_metric_amendment_v2"]
    parent_qualification_id: str
    parent_bundle_sha256: Sha256
    parent_artifacts_manifest_sha256: Sha256
    metric_estimand: Literal["centered_interval_log_frequency_change"]
    r0_metric_estimand: Literal["centered_interval_log_frequency_change"]
    false_promotion_upper_limit: float = Field(ge=0.05, le=0.05)
    optimizer_rerun: Literal[False] = False
    environment_hash: Sha256
    legacy_null_refits: ArtifactRef
    corrected_null_interval_refits: ArtifactRef
    null_model_effects: ArtifactRef
    recovery_curve: ArtifactRef
    recovery_series: ArtifactRef
    target_metrics: ArtifactRef
    bootstrap_target_draws: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef
    component_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_reaction_metric_amendment(self) -> ReactionRecoveryMetricAmendment:
        expected = self.identity(id_field="amendment_id")
        if self.amendment_id != expected:
            raise ValueError(f"amendment_id mismatch: expected {expected}.")
        return self


class PooledReactionLikelihoodBundle(StrictModel):
    """Immutable T07R-A0 pooled relative-guide likelihood qualification."""

    schema_version: Literal[1] = 1
    qualification_id: str
    test_contract_id: str
    method: Literal["pooled_target_reaction_dm_likelihood_v1"]
    evidence_role: Literal["development"] = "development"
    pooled_data_id: str
    t02a_amendment_id: str
    t07s_amendment_id: str
    fold_assignment_sha256: Sha256
    outer_fold: Literal[0] = 0
    inner_validation_fold: Literal[1] = 1
    candidate_updates: tuple[int, ...]
    retained_guides: int = Field(ge=1)
    outer_guides: int = Field(ge=1)
    target_count: int = Field(ge=2)
    input_catalog: ArtifactRef
    selection_curve: ArtifactRef
    refit_effects: ArtifactRef
    estimator_parity: ArtifactRef
    outer_guide_metrics: ArtifactRef
    null_calibration: ArtifactRef
    bootstrap_deltas: ArtifactRef
    selected_model: ArtifactRef
    test_receipt: ArtifactRef
    component_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_pooled_reaction_likelihood(self) -> PooledReactionLikelihoodBundle:
        expected = self.identity(id_field="qualification_id")
        if self.qualification_id != expected:
            raise ValueError(f"qualification_id mismatch: expected {expected}.")
        if self.candidate_updates != (0, 25, 50, 100, 200):
            raise ValueError("T07R-A0 requires the frozen update grid 0,25,50,100,200.")
        if self.outer_guides >= self.retained_guides:
            raise ValueError("The T07R outer fold must be smaller than the retained catalog.")
        return self


class PooledReactionLikelihoodReceipt(StrictModel):
    """Complete T07R-A0 parity and real pooled noninferiority decision surface."""

    schema_version: Literal[1] = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    evidence_role: Literal["development"] = "development"
    metric_estimand: Literal["pooled_p60_dirichlet_multinomial_nll_per_count"]
    outer_fold: Literal[0] = 0
    selected_update: int = Field(ge=0)
    post_selection_zero_initialized_refit: Literal[True] = True
    estimator_probability_max_abs_error: float = Field(ge=0)
    estimator_probability_tolerance: float = Field(gt=0)
    estimator_effect_max_abs_error: float = Field(ge=0)
    estimator_effect_tolerance: float = Field(gt=0)
    estimator_parity_pass: bool
    noninferiority_margin: float = Field(ge=0)
    m2_production_nll: float = Field(ge=0)
    m1_sister_target_nll: float = Field(ge=0)
    point_delta: float
    paired_bootstrap_interval: tuple[float, float]
    paired_bootstrap_upper_95: float
    paired_bootstrap_draws: int = Field(ge=1000)
    real_pooled_noninferiority_pass: bool
    protected_channels_pass: bool
    control_residual_max_abs_error: float = Field(ge=0)
    probability_normalization_max_abs_error: float = Field(ge=0)
    parent_components_verified: bool
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_pooled_reaction_receipt(self) -> PooledReactionLikelihoodReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.paired_bootstrap_interval
        if lower > upper:
            raise ValueError("T07R paired bootstrap interval must be ordered.")
        parity = (
            self.estimator_probability_max_abs_error < self.estimator_probability_tolerance
            and self.estimator_effect_max_abs_error < self.estimator_effect_tolerance
        )
        if self.estimator_parity_pass != parity:
            raise ValueError("T07R estimator-parity flag differs from its numerical metrics.")
        gates = (
            self.estimator_parity_pass,
            self.real_pooled_noninferiority_pass,
            self.protected_channels_pass,
            self.parent_components_verified,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T07R-A0 receipt must satisfy every frozen gate.")
        return self


class PhysicalPoolConditionalReactionBundle(StrictModel):
    """Immutable CPU forensic correction of the T07R physical-pool estimand."""

    schema_version: Literal[2] = 2
    qualification_id: str
    test_contract_id: str
    method: Literal["physical_pool_conditional_dm_likelihood_v2"]
    evidence_role: Literal["forensic_estimand_correction"]
    exposure_status: Literal["historically_exposed_development"]
    predecessor_method: Literal["fold_subcomposition_fixed_concentration_dm_v1"]
    pooled_data_id: str
    t02a_noise_id: str
    t02a_amendment_id: str
    t07s_amendment_id: str
    fold_assignment_sha256: Sha256
    physical_pool_id: Literal["pooled_P4_to_P60"]
    full_category_count: Literal[495]
    source_denominator_scope: Literal["all_495_retained_guides"]
    terminal_likelihood_scope: Literal["conditional_subcomposition"]
    full_concentration: float = Field(ge=1000.0, le=1000.0)
    conditional_concentration_rule: Literal["sum_full_alpha_over_active_categories"]
    outer_fold: Literal[0] = 0
    inner_validation_fold: Literal[1] = 1
    candidate_updates: tuple[int, ...]
    minimum_inner_fit_sisters: int = Field(ge=1)
    minimum_outer_nonouter_sisters: int = Field(ge=1)
    input_catalog: ArtifactRef
    role_support_audit: ArtifactRef
    selection_curve: ArtifactRef
    refit_effects: ArtifactRef
    estimator_parity: ArtifactRef
    factorization_check: ArtifactRef
    outer_guide_metrics: ArtifactRef
    conditional_multinomial_bootstrap: ArtifactRef
    conditional_dm_bootstrap: ArtifactRef
    selected_model: ArtifactRef
    parent_link: ArtifactRef
    test_receipt: ArtifactRef
    component_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_physical_pool_conditional_reaction(
        self,
    ) -> PhysicalPoolConditionalReactionBundle:
        expected = self.identity(id_field="qualification_id")
        if self.qualification_id != expected:
            raise ValueError(f"qualification_id mismatch: expected {expected}.")
        if self.candidate_updates != (0, 25, 50, 100, 200):
            raise ValueError("T07R-A0-v2 requires the frozen update grid.")
        return self


class PhysicalPoolConditionalReactionReceipt(StrictModel):
    """Numerical parity and exposed predictive decision for T07R-A0-v2."""

    schema_version: Literal[2] = 2
    receipt_id: str
    test_contract_id: str
    status: Literal["forensic_m1_superior", "forensic_m2_superior", "forensic_inconclusive"]
    evidence_role: Literal["forensic_estimand_correction"]
    exposure_status: Literal["historically_exposed_development"]
    metric_estimand: Literal["physical_pool_conditional_p60_dm_nll_per_count"]
    selected_update: int = Field(ge=0)
    estimator_probability_max_abs_error: float = Field(ge=0)
    estimator_probability_tolerance: float = Field(gt=0)
    estimator_effect_max_abs_error: float = Field(ge=0)
    estimator_effect_tolerance: float = Field(gt=0)
    estimator_parity_pass: bool
    factorization_max_abs_error: float = Field(ge=0)
    factorization_tolerance: float = Field(gt=0)
    factorization_pass: bool
    m2_selected_policy_nll: float = Field(ge=0)
    m1_sister_target_nll: float = Field(ge=0)
    point_delta_m2_minus_m1: float
    predictive_decision: Literal["m1_superior", "m2_superior", "inconclusive"]
    conditional_multinomial_interval_95: tuple[float, float]
    conditional_dm_interval_95: tuple[float, float]
    bootstrap_draws_each: int = Field(ge=1000)
    minimum_inner_fit_sisters: int = Field(ge=1)
    minimum_outer_nonouter_sisters: int = Field(ge=1)
    zero_inner_fit_sister_guides: Literal[0] = 0
    zero_outer_nonouter_sister_guides: Literal[0] = 0
    protected_channels_pass: bool
    parent_components_verified: Literal[True] = True
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_physical_pool_conditional_receipt(
        self,
    ) -> PhysicalPoolConditionalReactionReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        for interval in (
            self.conditional_multinomial_interval_95,
            self.conditional_dm_interval_95,
        ):
            if interval[0] > interval[1]:
                raise ValueError("T07R-A0-v2 intervals must be ordered.")
        parity = (
            self.estimator_probability_max_abs_error < self.estimator_probability_tolerance
            and self.estimator_effect_max_abs_error < self.estimator_effect_tolerance
        )
        if self.estimator_parity_pass != parity:
            raise ValueError("Estimator-parity flag differs from its numerical metrics.")
        if self.factorization_pass != (
            self.factorization_max_abs_error < self.factorization_tolerance
        ):
            raise ValueError("DM factorization flag differs from its numerical metric.")
        primary = self.conditional_multinomial_interval_95
        decision = (
            "m2_superior" if primary[1] < 0 else "m1_superior" if primary[0] > 0 else "inconclusive"
        )
        if self.predictive_decision != decision or self.status != f"forensic_{decision}":
            raise ValueError("Predictive decision must follow the sign of the primary interval.")
        return self


class RawCountMassNoiseBundle(StrictModel):
    """Immutable T02A raw-count and relative-mass noise-floor bundle."""

    schema_version: int = 1
    noise_id: str
    test_contract_id: str
    pooled_data_id: str
    count_store_sha256: Sha256
    feature_index_hash: Sha256
    environment_hash: Sha256
    source_checkpoint: str = Field(min_length=1)
    terminal_checkpoint: str = Field(min_length=1)
    split_repeats: int = Field(ge=100)
    mass_bootstrap_repeats: int = Field(ge=100)
    split_seed_start: int = Field(ge=0)
    mass_seed_start: int = Field(ge=0)
    variable_gene_count: int = Field(ge=2)
    top_gene_count: int = Field(ge=1)
    rank_top_k: int = Field(ge=1)
    variable_genes: ArtifactRef
    raw_split_metrics: ArtifactRef
    raw_repeat_summary: ArtifactRef
    raw_target_summary: ArtifactRef
    mass_bootstrap: ArtifactRef
    mass_guide_noise: ArtifactRef
    mass_target_noise: ArtifactRef
    frozen_thresholds: ArtifactRef
    test_receipt: ArtifactRef

    @model_validator(mode="after")
    def validate_noise_bundle(self) -> RawCountMassNoiseBundle:
        expected = self.identity(id_field="noise_id")
        if self.noise_id != expected:
            raise ValueError(f"noise_id mismatch: expected {expected}.")
        if self.source_checkpoint == self.terminal_checkpoint:
            raise ValueError("T02A source and terminal checkpoints must differ.")
        if self.top_gene_count > self.variable_gene_count:
            raise ValueError("Top-gene overlap cannot exceed its frozen gene universe.")
        return self


class RawCountMassNoiseReceipt(StrictModel):
    """Complete T02A calibration decision and frozen noise floors."""

    schema_version: int = 1
    receipt_id: str
    test_contract_id: str
    status: Literal["pass", "fail_retired"]
    retained_guides: int = Field(ge=1)
    perturbation_targets: int = Field(ge=1)
    split_repeats: int = Field(ge=100)
    mass_bootstrap_repeats: int = Field(ge=100)
    target_balanced_hellinger_q95: float = Field(ge=0)
    control_hellinger_q95: float = Field(ge=0)
    target_balanced_js_q95: float = Field(ge=0)
    target_balanced_deviance_q95: float = Field(ge=0)
    target_balanced_spearman_q05: float = Field(ge=-1, le=1)
    target_balanced_top_gene_overlap_q05: float = Field(ge=0, le=1)
    interval_log_mass_rmse_q95: float = Field(ge=0)
    expansion_sign_accuracy_q05: float = Field(ge=0, le=1)
    guide_rank_spearman_q05: float = Field(ge=-1, le=1)
    target_rank_spearman_q05: float = Field(ge=-1, le=1)
    top_k_overlap_q05: float = Field(ge=0, le=1)
    bottom_k_overlap_q05: float = Field(ge=0, le=1)
    raw_invariants_pass: bool
    mass_invariants_pass: bool
    protected_metrics_frozen: bool
    config_hash: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256

    @model_validator(mode="after")
    def validate_noise_receipt(self) -> RawCountMassNoiseReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        if self.status == "pass" and not (
            self.raw_invariants_pass and self.mass_invariants_pass and self.protected_metrics_frozen
        ):
            raise ValueError("A passing T02A receipt must freeze both complete noise floors.")
        return self


class RawCountMassNoiseAmendment(StrictModel):
    """Derived T02A interpretation amendment bound to one immutable bundle."""

    schema_version: Literal[1] = 1
    amendment_id: str
    parent_noise_id: str
    parent_bundle_sha256: Sha256
    implementation_hash: Sha256
    environment_hash: Sha256
    source_checkpoint: str = Field(min_length=1)
    terminal_checkpoint: str = Field(min_length=1)
    thresholds_by_checkpoint: ArtifactRef
    recomputed_thresholds: ArtifactRef
    target_rank_stability: ArtifactRef
    threshold_semantics: ArtifactRef
    implementation_identity: ArtifactRef
    environment_identity: ArtifactRef

    @model_validator(mode="after")
    def validate_amendment(self) -> RawCountMassNoiseAmendment:
        expected = self.identity(id_field="amendment_id")
        if self.amendment_id != expected:
            raise ValueError(f"amendment_id mismatch: expected {expected}.")
        if self.source_checkpoint == self.terminal_checkpoint:
            raise ValueError("T02A amendment checkpoints must differ.")
        return self


class RawCountMassNoiseAmendmentReceipt(StrictModel):
    """Fail-closed verification result for a derived T02A amendment."""

    schema_version: Literal[1] = 1
    receipt_id: str
    amendment_id: str
    parent_noise_id: str
    implementation_hash: Sha256
    environment_hash: Sha256
    status: Literal["pass", "fail_retired"]
    parent_bundle_verified: bool
    table_invariants_pass: bool
    thresholds_recomputed: bool
    checkpoint_thresholds_recomputed: bool
    misleading_labels_retired: bool

    @model_validator(mode="after")
    def validate_amendment_receipt(self) -> RawCountMassNoiseAmendmentReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        gates = (
            self.parent_bundle_verified,
            self.table_invariants_pass,
            self.thresholds_recomputed,
            self.checkpoint_thresholds_recomputed,
            self.misleading_labels_retired,
        )
        if self.status == "pass" and not all(gates):
            raise ValueError("A passing T02A amendment must satisfy every verification gate.")
        return self


class ComponentTestContract(StrictModel):
    """Frozen component-wise qualification contract."""

    schema_version: int = 1
    test_contract_id: str
    test_id: str = Field(pattern=r"^T\d{2}[A-Z]?_[A-Z0-9_]+$")
    component: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    primary_baseline: str = Field(min_length=1)
    required_margin: float = Field(ge=0)
    drift: Literal["off", "fixed", "trainable"] = "off"
    diffusion: Literal["off", "fixed", "trainable"] = "off"
    reaction: Literal["off", "fixed", "trainable"] = "off"
    ecology: Literal["off", "fixed", "trainable"] = "off"
    decoder: Literal["off", "fixed", "trainable"] = "off"
    update_zero_selectable: bool
    post_selection_refit_required: bool

    @model_validator(mode="after")
    def validate_contract_identity(self) -> ComponentTestContract:
        expected = self.identity(id_field="test_contract_id")
        if self.test_contract_id != expected:
            raise ValueError(f"test_contract_id mismatch: expected {expected}.")
        stage = self.test_id[:3]
        exact: dict[str, tuple[str, str, str, str, str]] = {
            "T00": ("off", "off", "off", "off", "off"),
            "T01": ("off", "off", "off", "off", "off"),
            "T02": ("off", "off", "off", "off", "off"),
            "T03": ("off", "off", "off", "off", "off"),
            "T04": ("fixed", "fixed", "fixed", "off", "off"),
            "T05": ("trainable", "fixed", "off", "off", "off"),
            "T06": ("fixed", "trainable", "off", "off", "off"),
            "T07": ("fixed", "fixed", "trainable", "off", "off"),
            "T08": ("trainable", "trainable", "trainable", "off", "off"),
            "T09": ("fixed", "fixed", "fixed", "trainable", "off"),
            "T11": ("fixed", "fixed", "fixed", "fixed", "off"),
            "T12": ("fixed", "fixed", "fixed", "fixed", "trainable"),
            "T13": ("fixed", "fixed", "fixed", "fixed", "fixed"),
        }
        observed = (self.drift, self.diffusion, self.reaction, self.ecology, self.decoder)
        if stage in exact and observed != exact[stage]:
            raise ValueError(f"{stage} channel-isolation matrix mismatch.")
        if stage == "T10" and (
            observed[:3] != ("fixed", "fixed", "fixed")
            or self.ecology not in {"off", "fixed"}
            or self.decoder != "off"
        ):
            raise ValueError("T10 channel-isolation matrix mismatch.")
        return self


class ComponentTestReceipt(StrictModel):
    """Fail-closed result for one independently qualified component."""

    schema_version: int = 1
    receipt_id: str
    test_id: str = Field(pattern=r"^T\d{2}[A-Z]?_[A-Z0-9_]+$")
    status: Literal["pass", "fail_retired", "not_run"]
    primary_metric: str
    primary_baseline: str
    point_delta: float
    bootstrap_interval: tuple[float, float]
    required_margin: float = Field(ge=0)
    channel_activity: float = Field(ge=0)
    protected_metrics_pass: bool
    selected_update: int = Field(ge=0)
    input_hashes: dict[str, Sha256]
    config_hash: Sha256
    implementation_hash: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> ComponentTestReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        lower, upper = self.bootstrap_interval
        if lower > upper:
            raise ValueError("Bootstrap interval must be ordered.")
        if self.status == "pass" and not self.protected_metrics_pass:
            raise ValueError("A passing component must preserve protected metrics.")
        return self


class ComponentTestReceiptV2(StrictModel):
    """Role-aware component receipt without overloaded comparison fields."""

    schema_version: Literal[2] = 2
    receipt_id: str
    test_id: str = Field(pattern=r"^T\d{2}[A-Z]?_[A-Z0-9_]+$")
    receipt_role: Literal["model_comparison", "calibration", "invariant"]
    status: Literal["pass", "fail_retired", "not_run"]
    primary_metric: str
    primary_baseline: str
    point_delta: float | None = None
    bootstrap_interval: tuple[float, float] | None = None
    required_margin: float | None = Field(default=None, ge=0)
    channel_activity: float | None = Field(default=None, ge=0)
    estimand: str | None = None
    quantile_probability: float | None = Field(default=None, ge=0, le=1)
    quantile_value: float | None = None
    repeat_count: int | None = Field(default=None, ge=1)
    sampling_method: str | None = None
    protected_metrics_pass: bool
    selected_update: int = Field(ge=0)
    input_hashes: dict[str, Sha256]
    config_hash: Sha256
    implementation_hash: Sha256

    @model_validator(mode="after")
    def validate_role_receipt(self) -> ComponentTestReceiptV2:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        numeric = (
            self.point_delta,
            *(self.bootstrap_interval or ()),
            self.required_margin,
            self.channel_activity,
            self.quantile_probability,
            self.quantile_value,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("Component receipt numerical fields must be finite.")
        comparison = (
            self.point_delta,
            self.bootstrap_interval,
            self.required_margin,
            self.channel_activity,
        )
        calibration = (
            self.estimand,
            self.quantile_probability,
            self.quantile_value,
            self.repeat_count,
            self.sampling_method,
        )
        if self.receipt_role == "model_comparison":
            if any(value is None for value in comparison) or any(
                value is not None for value in calibration
            ):
                raise ValueError("Model-comparison receipts require only comparison fields.")
            assert self.bootstrap_interval is not None
            if self.bootstrap_interval[0] > self.bootstrap_interval[1]:
                raise ValueError("Bootstrap interval must be ordered.")
        elif self.receipt_role == "calibration":
            if any(value is not None for value in comparison) or any(
                value is None for value in calibration
            ):
                raise ValueError("Calibration receipts require only calibration fields.")
        elif any(value is not None for value in (*comparison, *calibration)):
            raise ValueError("Invariant receipts do not carry comparison or calibration fields.")
        if self.status == "pass" and not self.protected_metrics_pass:
            raise ValueError("A passing component must preserve protected metrics.")
        return self


class InputViewArtifact(StrictModel):
    schema_version: int = 1
    view_id: str
    method: Literal["identity_library_normalized", "matched_control_offset_v1"]
    raw_parent_hash: Sha256
    fit_rows_hash: Sha256
    parameter_artifact: ArtifactRef | None
    zero_preserving: Literal[True] = True
    estimability_pass: bool
    inner_validation_pass: bool
    selected: bool


class EffectHierarchyRow(StrictModel):
    perturbation_id: str
    target_id: str
    target_index: int = Field(ge=0)
    is_control: bool = False
    source_efficacy: float | None = Field(default=None, gt=0)


class EffectHierarchyContract(StrictModel):
    schema_version: int = 1
    hierarchy_id: str
    rows: tuple[EffectHierarchyRow, ...]
    guide_specific_drift: Literal[False] = False
    guide_specific_diffusion: Literal[False] = False


class TopologySupport(StrictModel):
    series_id: str
    partition_id: str
    source_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)


class TransportTopologyContract(StrictModel):
    schema_version: int = 1
    topology_id: str
    partitions: tuple[str, ...]
    allowed_edges: tuple[tuple[str, str], ...]
    support: tuple[TopologySupport, ...] = ()
    immigration_enabled: bool = False

    @model_validator(mode="after")
    def supported_terminal_states(self) -> TransportTopologyContract:
        allowed = set(self.partitions)
        if any(
            source not in allowed or target not in allowed for source, target in self.allowed_edges
        ):
            raise ValueError("Topology edge references an undeclared partition.")
        if not self.immigration_enabled:
            invalid = [
                row for row in self.support if row.source_count == 0 and row.terminal_count > 0
            ]
            if invalid:
                raise ValueError(
                    "Positive terminal support from a zero-source partition requires an "
                    "explicit immigration mechanism."
                )
        return self


class DenominatorBlock(StrictModel):
    block_id: str
    category_ids: tuple[str, ...]
    explicit_zero_categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def complete_zeros(self) -> DenominatorBlock:
        if not self.category_ids or len(self.category_ids) != len(set(self.category_ids)):
            raise ValueError("Denominator categories must be nonempty and unique.")
        if not set(self.explicit_zero_categories) <= set(self.category_ids):
            raise ValueError("Explicit zero categories must remain in the denominator.")
        return self


class DenominatorContract(StrictModel):
    schema_version: int = 1
    denominator_id: str
    blocks: tuple[DenominatorBlock, ...]
    relative_within_group: Literal[True] = True
    complete_categories: Literal[True] = True
    source_smoothing: float = 0.5

    @field_validator("source_smoothing")
    @classmethod
    def smoothing_frozen(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("v4.0 source smoothing is frozen at 0.5.")
        return value


class PoolContributor(StrictModel):
    pool_id: str
    series_id: str
    contributor_role: Literal["modeled", "background"] = "modeled"


class PoolContract(StrictModel):
    schema_version: int = 1
    pool_contract_id: str
    physical_pool_ids: tuple[str, ...]
    contributors: tuple[PoolContributor, ...]
    complete_contributor_policy: Literal[True] = True

    @model_validator(mode="after")
    def contributors_cover_pools(self) -> PoolContract:
        declared = set(self.physical_pool_ids)
        observed = {item.pool_id for item in self.contributors}
        if observed != declared:
            raise ValueError("Every declared physical pool needs at least one contributor.")
        return self


class ExposureRecord(StrictModel):
    unit_type: str
    unit_id: str
    endpoint_seen: bool
    used_for_architecture: bool = False
    used_for_hyperparameters: bool = False
    used_for_thresholds: bool = False
    used_for_biological_story: bool = False
    first_exposure_date: str | None = None
    source_artifact_hash: Sha256 | None = None
    exposure_role: EvidenceRole = EvidenceRole.DEVELOPMENT


class ExposureRegistry(StrictModel):
    schema_version: int = 1
    records: tuple[ExposureRecord, ...]


class EligibilityManifest(StrictModel):
    schema_version: int = 1
    manifest_id: str
    abundance_eligible_units: tuple[str, ...]
    state_evaluable_units: tuple[str, ...]
    endpoint_existence_used_for_abundance: Literal[False] = False
    source_information_set_hash: Sha256


class PreregistrationRef(StrictModel):
    schema_version: int = 1
    preregistration_id: str
    artifact: ArtifactRef
    frozen_before_evaluation: bool


class SeriesRecord(StrictModel):
    series_id: str
    target_index: int = Field(ge=0)
    pool_index: int = Field(ge=0)
    is_control: bool = False
    source_rows: tuple[int, ...]
    terminal_rows: tuple[int, ...]
    source_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)
    duration: float = Field(gt=0)


class SemanticStudySnapshot(StrictModel):
    schema_version: int = 1
    study_id: str
    series: tuple[SeriesRecord, ...]
    observed_edges: tuple[tuple[str, str], ...]
    feature_index_hash: Sha256
    row_universe_hash: Sha256
    exposure_registry: ExposureRegistry

    @model_validator(mode="after")
    def validate_series(self) -> SemanticStudySnapshot:
        if not self.series:
            raise ValueError("Semantic snapshot needs at least one series.")
        ids = [series.series_id for series in self.series]
        if len(ids) != len(set(ids)):
            raise ValueError("Series IDs must be unique.")
        return self


class CountStoreManifest(StrictModel):
    schema_version: int = 1
    store_id: str
    backend: Literal["csr_hdf5"] = "csr_hdf5"
    rows: int = Field(gt=0)
    features: int = Field(gt=0)
    nnz: int = Field(ge=0)
    value_dtype: str
    row_ids_hash: Sha256
    feature_index_hash: Sha256
    content_sha256: Sha256
    relative_uri: str

    _safe_uri = field_validator("relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )


class CountStoreShard(StrictModel):
    """One immutable row-contiguous member of a sharded count store."""

    shard_id: str = Field(min_length=1)
    relative_uri: str
    rows: int = Field(gt=0)
    features: int = Field(gt=0)
    nnz: int = Field(ge=0)
    row_id_min: int
    row_id_max: int
    row_ids_hash: Sha256
    content_sha256: Sha256

    _safe_uri = field_validator("relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @model_validator(mode="after")
    def valid_row_interval(self) -> CountStoreShard:
        if self.row_id_max < self.row_id_min:
            raise ValueError("Shard row interval is reversed.")
        return self


class ShardedCountStoreManifest(StrictModel):
    """Catalog-independent sparse store assembled from ordered CSR shards."""

    schema_version: int = 1
    store_id: str
    backend: Literal["csr_hdf5_sharded"] = "csr_hdf5_sharded"
    rows: int = Field(gt=0)
    features: int = Field(gt=0)
    nnz: int = Field(ge=0)
    value_dtype: Literal["int32"] = "int32"
    row_ids_hash: Sha256
    feature_index_hash: Sha256
    row_locator_sha256: Sha256
    row_locator_relative_uri: str = "row-locator.h5"
    merkle_root: Sha256
    shards: tuple[CountStoreShard, ...]

    _safe_locator_uri = field_validator("row_locator_relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @model_validator(mode="after")
    def validate_store(self) -> ShardedCountStoreManifest:
        if not self.shards:
            raise ValueError("Sharded count store needs at least one shard.")
        if sum(item.rows for item in self.shards) != self.rows:
            raise ValueError("Shard rows do not sum to store rows.")
        if sum(item.nnz for item in self.shards) != self.nnz:
            raise ValueError("Shard nonzeros do not sum to store nonzeros.")
        if any(item.features != self.features for item in self.shards):
            raise ValueError("Every shard must use the store feature width.")
        expected = self.identity(id_field="store_id")
        if self.store_id != expected:
            raise ValueError(f"store_id mismatch: expected {expected}.")
        return self


class VirtualCountSource(StrictModel):
    """One immutable source matrix behind the metadata-only canonical plane."""

    source_id: str = Field(min_length=1)
    donor_id: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    physical_time_hours: float
    relative_uri: str
    source_file_sha256: Sha256
    dataset_path: str = Field(default="X", min_length=1)
    rows: int = Field(gt=0)
    features: int = Field(gt=0)
    nnz: int = Field(ge=0)
    eligible_rows: int = Field(gt=0)
    eligible_nnz: int = Field(ge=0)
    source_feature_order_hash: Sha256
    canonical_permutation_hash: Sha256

    _safe_uri = field_validator("relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @field_validator("physical_time_hours")
    @classmethod
    def finite_physical_time(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Source physical time must be finite and nonnegative.")
        return value


class G00SourceAuthority(StrictModel):
    """G00A source/feature authority; it cannot expose model-facing outcomes."""

    schema_version: Literal[1] = 1
    authority_id: str
    sources: tuple[VirtualCountSource, ...]
    canonical_feature_index_hash: Sha256
    guide_catalog_hash: Sha256
    target_catalog_hash: Sha256
    eligibility_rule: Literal["guide_group == targeting single sgRNA AND low_quality == false"]
    eligible_row_ids_hash: Sha256
    eligible_rows: int = Field(gt=0)
    eligible_nnz: int = Field(ge=0)
    canonical_row_id_rule: Literal["(sample_index << 32) | source_row_index"] = (
        "(sample_index << 32) | source_row_index"
    )
    full_source_hashes_verified: Literal[True] = True
    source_reconciliation_pass: Literal[True] = True
    model_facing_output: Literal[False] = False

    @model_validator(mode="after")
    def validate_source_authority(self) -> G00SourceAuthority:
        ids = [source.source_id for source in self.sources]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("G00A sources must be nonempty and uniquely identified.")
        if sum(source.eligible_rows for source in self.sources) != self.eligible_rows:
            raise ValueError("G00A source eligible rows do not reconcile to the authority total.")
        if sum(source.eligible_nnz for source in self.sources) != self.eligible_nnz:
            raise ValueError(
                "G00A source eligible nonzeros do not reconcile to the authority total."
            )
        expected = self.identity(id_field="authority_id")
        if self.authority_id != expected:
            raise ValueError(f"authority_id mismatch: expected {expected}.")
        return self


class VirtualCanonicalCountStoreManifest(StrictModel):
    """G00B metadata-only canonical access over immutable source matrices."""

    schema_version: Literal[1] = 1
    virtual_store_id: str
    backend: Literal["virtual_canonical_h5ad_csr_v1"] = "virtual_canonical_h5ad_csr_v1"
    source_authority_id: str
    canonical_feature_index_hash: Sha256
    guide_catalog_hash: Sha256
    target_catalog_hash: Sha256
    eligibility_rule: Literal["guide_group == targeting single sgRNA AND low_quality == false"]
    eligible_rows: int = Field(gt=0)
    features: int = Field(gt=0)
    row_locator: ArtifactRef
    feature_permutations: ArtifactRef
    row_locator_schema: Literal["canonical_row_source_row_guide_target_v1"] = (
        "canonical_row_source_row_guide_target_v1"
    )
    sources: tuple[VirtualCountSource, ...]
    raw_counts_materialized: Literal[False] = False
    intended_use: Literal["sequential_statistics_and_fold_materialization"] = (
        "sequential_statistics_and_fold_materialization"
    )
    direct_h100_training_backend: Literal[False] = False
    protected_outer_endpoints_read: Literal[False] = False
    archival_shard_plan: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_virtual_store(self) -> VirtualCanonicalCountStoreManifest:
        ids = [source.source_id for source in self.sources]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Virtual canonical sources must be nonempty and unique.")
        if any(source.features != self.features for source in self.sources):
            raise ValueError("Every virtual source must expose the canonical feature width.")
        if sum(source.eligible_rows for source in self.sources) != self.eligible_rows:
            raise ValueError("Virtual source eligible rows do not reconcile to the manifest.")
        expected = self.identity(id_field="virtual_store_id")
        if self.virtual_store_id != expected:
            raise ValueError(f"virtual_store_id mismatch: expected {expected}.")
        return self


class FoldNativeCompactViewContract(StrictModel):
    """G00C training-only feature/sample materialization for one outer fold."""

    schema_version: Literal[1] = 1
    fold_view_id: str
    parent_source_authority_id: str
    parent_virtual_store_id: str
    outer_split_id: str
    training_donor_ids: tuple[str, ...]
    heldout_donor_id: str
    training_only_feature_selection_hash: Sha256
    ordered_selected_feature_ids_hash: Sha256
    selected_features: int = Field(gt=0, le=4096)
    sample_size_candidates: tuple[int, ...]
    selected_training_cells: int = Field(gt=0)
    training_rows_hash: Sha256
    heldout_source_rows_hash: Sha256
    protected_outer_endpoint_rows_hash: Sha256
    protected_outer_endpoints_materialized: Literal[False] = False
    count_dtype: Literal["uint16", "int32"]
    index_dtype: Literal["uint16", "int32"]
    count_maximum_audit_pass: bool
    physical_layout: Literal["donor_checkpoint_target_guide_row"]
    microbatch_cells: int = Field(ge=1)
    microbatches_per_update: int = Field(ge=1)
    gradient_accumulation: Literal[True] = True

    @model_validator(mode="after")
    def validate_fold_view(self) -> FoldNativeCompactViewContract:
        if (
            not self.training_donor_ids
            or len(self.training_donor_ids) != len(set(self.training_donor_ids))
            or self.heldout_donor_id in self.training_donor_ids
        ):
            raise ValueError("Fold-native donor roles must be nonempty, unique, and disjoint.")
        if (
            not self.sample_size_candidates
            or tuple(sorted(set(self.sample_size_candidates))) != self.sample_size_candidates
            or self.selected_training_cells not in self.sample_size_candidates
        ):
            raise ValueError("Fold-native sample-size candidates must be increasing and selected.")
        if self.count_dtype == "uint16" and not self.count_maximum_audit_pass:
            raise ValueError("uint16 counts require a passed maximum-count audit.")
        expected = self.identity(id_field="fold_view_id")
        if self.fold_view_id != expected:
            raise ValueError(f"fold_view_id mismatch: expected {expected}.")
        return self


class IntegratedLoaderQualificationContract(StrictModel):
    """G00D end-to-end loader/compute qualification on one compact view."""

    schema_version: Literal[1] = 1
    qualification_contract_id: str
    fold_view_id: str
    maximum_data_wait_fraction: float = Field(default=0.10, ge=0, le=0.10)
    minimum_steady_state_gpu_utilization: float = Field(default=0.85, ge=0.85, le=1)
    prefetch_depth: int = Field(ge=1)
    maximum_loader_rss_bytes: int = Field(gt=0)
    maximum_open_shards: int = Field(ge=1)
    training_metric_absolute_tolerance: float = Field(default=1e-6, gt=0)
    training_metric_relative_tolerance: float = Field(default=1e-5, gt=0)
    require_zero_row_count_order_errors: Literal[True] = True
    require_bounded_loader_memory: Literal[True] = True
    raw_rows_per_second_gate: Literal[False] = False

    @model_validator(mode="after")
    def validate_loader_contract(self) -> IntegratedLoaderQualificationContract:
        expected = self.identity(id_field="qualification_contract_id")
        if self.qualification_contract_id != expected:
            raise ValueError(f"qualification_contract_id mismatch: expected {expected}.")
        return self


class IntegratedLoaderQualificationReceipt(StrictModel):
    """Observed G00D wait/utilization, parity, and memory result."""

    schema_version: Literal[1] = 1
    receipt_id: str
    qualification_contract_id: str
    data_wait_fraction: float = Field(ge=0, le=1)
    steady_state_gpu_utilization: float = Field(ge=0, le=1)
    p95_batch_ready_seconds: float = Field(ge=0)
    covered_compute_seconds: float = Field(ge=0)
    peak_loader_rss_bytes: int = Field(ge=0)
    peak_open_shards: int = Field(ge=0)
    row_count_order_errors: int = Field(ge=0)
    unbounded_memory_growth_detected: bool
    training_metric_max_absolute_error: float = Field(ge=0)
    training_metric_max_relative_error: float = Field(ge=0)
    training_metric_parity_pass: bool
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_loader_receipt(self) -> IntegratedLoaderQualificationReceipt:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


# Expose explicit aliases for accepted dev29 v1 evidence.
VirtualCountSourceV1 = VirtualCountSource
G00SourceAuthorityV1 = G00SourceAuthority
VirtualCanonicalCountStoreManifestV1 = VirtualCanonicalCountStoreManifest
FoldNativeCompactViewContractV1 = FoldNativeCompactViewContract
IntegratedLoaderQualificationContractV1 = IntegratedLoaderQualificationContract
IntegratedLoaderQualificationReceiptV1 = IntegratedLoaderQualificationReceipt


class ProtectedSourceAccessSemantics(StrictModel):
    """Exact distinction between source-byte authority reads and outcome use."""

    protected_source_bytes_hashed: Literal[True] = True
    protected_obs_metadata_read_for_authority: Literal[True] = True
    protected_csr_structure_scanned_for_authority: Literal[True] = True
    protected_csr_values_scanned_for_numeric_authority: Literal[True] = True
    protected_expression_values_used_for_feature_selection: Literal[False] = False
    protected_expression_values_used_for_model_fitting: Literal[False] = False
    protected_expression_values_used_for_model_selection: Literal[False] = False
    protected_expression_values_used_for_evaluation: Literal[False] = False


class SourceNumericIntegrity(StrictModel):
    """Observed sparse-storage invariants for one source matrix."""

    matrix_encoding: Literal["csr"] = "csr"
    storage_value_dtype: str = Field(min_length=1)
    indices_dtype: str = Field(min_length=1)
    indptr_dtype: str = Field(min_length=1)
    counts_nonnegative_verified: Literal[True] = True
    counts_integral_verified: Literal[True] = True
    counts_finite_verified: Literal[True] = True
    maximum_observed_count: int = Field(ge=0)
    csr_indices_in_bounds_verified: Literal[True] = True
    csr_indptr_monotonic_verified: Literal[True] = True
    csr_terminal_offset_matches_nnz: Literal[True] = True


class SourcePlaneDerivationRecord(StrictModel):
    """Construction-time evidence for one locator-selected immutable source."""

    source_id: str = Field(min_length=1)
    selected_row_count: int = Field(gt=0)
    selected_nnz: int = Field(ge=0)
    selected_row_ids_hash: Sha256
    source_row_pairs_hash: Sha256
    scanner_implementation_sha256: Sha256
    source_file_sha256: Sha256


class SourcePlaneDerivationReceipt(StrictModel):
    """Content-addressed construction scan behind routine G00B verification."""

    schema_version: Literal[1] = 1
    receipt_id: str
    records: tuple[SourcePlaneDerivationRecord, ...]
    model_fitting_performed: Literal[False] = False
    protected_outcome_scientific_use: Literal[False] = False

    @model_validator(mode="after")
    def validate_derivation(self) -> SourcePlaneDerivationReceipt:
        source_ids = [record.source_id for record in self.records]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError("Source-plane derivation records must be nonempty and unique.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class SourceHashBinding(StrictModel):
    """One immutable source identity in a format-upgrade amendment."""

    source_id: str = Field(min_length=1)
    source_file_sha256: Sha256


class LegacyChecksumManifestBinding(StrictModel):
    """Byte identity of the historical checksum boundary inside its parent."""

    relative_uri: Literal["SHA256SUMS"] = "SHA256SUMS"
    sha256: Sha256
    size_bytes: int = Field(gt=0)


class LegacyParentDescriptor(StrictModel):
    """Accepted semantic identities behind one historical parent directory."""

    logical_name: str = Field(min_length=1)
    g00a_v1_authority_id: str = Field(min_length=1)
    g00b_v1_virtual_store_id: str = Field(min_length=1)
    sha256sums: LegacyChecksumManifestBinding


class LegacyPublicationSemantics(StrictModel):
    """Statements that a non-retroactive wrapper may make about its parent."""

    sha256sums_present: Literal[True] = True
    artifacts_manifest_present: Literal[False] = False
    committed_marker_present: Literal[False] = False
    manifest_last_publication_proven: Literal[False] = False
    atomic_publication_proven: Literal[False] = False
    transactional_publication_proven: Literal[False] = False


class LegacyChecksumVerificationSummary(StrictModel):
    """Fail-closed syntax, file-set, and byte verification summary."""

    listed_files: int = Field(gt=0)
    matched_files: int = Field(gt=0)
    missing_files: Literal[0] = 0
    mismatched_files: Literal[0] = 0
    duplicate_manifest_paths: Literal[0] = 0
    path_traversal_entries: Literal[0] = 0
    symlinks: Literal[0] = 0
    special_files: Literal[0] = 0
    uncovered_authoritative_files: Literal[0] = 0

    @model_validator(mode="after")
    def validate_summary(self) -> LegacyChecksumVerificationSummary:
        if self.matched_files != self.listed_files:
            raise ValueError("Every legacy checksum entry must match exactly.")
        return self


class LegacyAttestationArtifacts(StrictModel):
    """Wrapper-owned evidence used to derive the attestation."""

    directory_inventory: ArtifactRef
    checksum_verification: ArtifactRef
    parent_link: ArtifactRef


class LegacyExecutionBoundary(StrictModel):
    """Explicitly non-scientific and non-mutating wrapper execution boundary."""

    parent_writes_performed: Literal[False] = False
    source_matrix_access_performed: Literal[False] = False
    model_fitting_performed: Literal[False] = False
    protected_expression_scientifically_used: Literal[False] = False


class LegacyAttestationBuilder(StrictModel):
    """Exact package and component identity that built the wrapper."""

    git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    distribution_sha256: Sha256
    implementation_sha256: Sha256
    environment_sha256: Sha256


class LegacyParentAttestationV1(StrictModel):
    """Non-retroactive wrapper around one accepted checksum-only parent."""

    schema_id: Literal["credo.legacy_parent_attestation"] = "credo.legacy_parent_attestation"
    schema_version: Literal[1] = 1
    attestation_id: str
    evidence_role: Literal["nonretroactive_legacy_checksum_parent_attestation"] = (
        "nonretroactive_legacy_checksum_parent_attestation"
    )
    parent: LegacyParentDescriptor
    legacy_publication_semantics: LegacyPublicationSemantics
    verification: LegacyChecksumVerificationSummary
    artifacts: LegacyAttestationArtifacts
    execution_boundary: LegacyExecutionBoundary
    builder: LegacyAttestationBuilder

    @model_validator(mode="after")
    def validate_attestation(self) -> LegacyParentAttestationV1:
        expected = self.identity(id_field="attestation_id")
        if self.attestation_id != expected:
            raise ValueError(f"attestation_id mismatch: expected {expected}.")
        return self


class LegacyParentAttestationTestContractV1(StrictModel):
    """Frozen gate list for one non-retroactive wrapper execution."""

    schema_version: Literal[1] = 1
    test_contract_id: str
    parent_logical_name: str = Field(min_length=1)
    require_strict_manifest_syntax: Literal[True] = True
    require_complete_regular_file_coverage: Literal[True] = True
    require_no_symlinks_or_special_files: Literal[True] = True
    require_exact_parent_semantic_ids: Literal[True] = True
    require_parent_relationship_verification: Literal[True] = True
    require_no_parent_writes: Literal[True] = True
    require_no_source_matrix_access: Literal[True] = True

    @model_validator(mode="after")
    def validate_contract(self) -> LegacyParentAttestationTestContractV1:
        expected = self.identity(id_field="test_contract_id")
        if self.test_contract_id != expected:
            raise ValueError(f"test_contract_id mismatch: expected {expected}.")
        return self


class LegacyParentAttestationReceiptV1(StrictModel):
    """Passed decision receipt for one exact wrapper and historical parent."""

    schema_version: Literal[1] = 1
    receipt_id: str
    test_contract_id: str = Field(min_length=1)
    attestation_id: str = Field(min_length=1)
    attestation: ArtifactRef
    legacy_sha256sums_verified: bool
    all_authoritative_files_covered: bool
    parent_g00a_v1_verified: bool
    parent_g00b_v1_verified: bool
    parent_g00a_g00b_relationship_verified: bool
    original_atomicity_not_claimed: bool
    parent_writes_performed: Literal[False] = False
    source_matrix_access_performed: Literal[False] = False
    model_fitting_performed: Literal[False] = False
    protected_expression_scientifically_used: Literal[False] = False
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_receipt(self) -> LegacyParentAttestationReceiptV1:
        passed = all(
            (
                self.legacy_sha256sums_verified,
                self.all_authoritative_files_covered,
                self.parent_g00a_v1_verified,
                self.parent_g00b_v1_verified,
                self.parent_g00a_g00b_relationship_verified,
                self.original_atomicity_not_claimed,
            )
        )
        expected_status = "pass" if passed else "fail"
        if self.status != expected_status:
            raise ValueError(f"Legacy-parent attestation status must be {expected_status}.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class NativeManifestLastBoundary(StrictModel):
    """Native publication type; legacy dispatch may never construct this type."""

    boundary_kind: Literal["native_manifest_last_v1"] = "native_manifest_last_v1"
    parent_logical_name: str = Field(min_length=1)
    g00a_v1_authority_id: str = Field(min_length=1)
    g00b_v1_virtual_store_id: str = Field(min_length=1)
    artifacts_manifest: ArtifactRef
    committed_marker: ArtifactRef
    sha256sums: ArtifactRef


class LegacyChecksumAttestedBoundary(StrictModel):
    """Distinct parent type admitted only by one passed sibling wrapper."""

    boundary_kind: Literal["legacy_checksum_attested_v1"] = "legacy_checksum_attested_v1"
    parent_logical_name: str = Field(min_length=1)
    g00a_v1_authority_id: str = Field(min_length=1)
    g00b_v1_virtual_store_id: str = Field(min_length=1)
    original_sha256sums: ArtifactRef
    attestation_id: str = Field(min_length=1)
    attestation: ArtifactRef
    attestation_receipt_id: str = Field(min_length=1)
    attestation_receipt: ArtifactRef


ParentPublicationBoundary = Annotated[
    NativeManifestLastBoundary | LegacyChecksumAttestedBoundary,
    Field(discriminator="boundary_kind"),
]


class B0ParentResolutionReceiptV1(StrictModel):
    """Fail-closed B0.1 result for exactly one discriminated parent type."""

    schema_version: Literal[1] = 1
    receipt_id: str
    boundary: ParentPublicationBoundary
    parent_checksum_boundary: bool
    all_consumed_artifacts_covered: bool
    g00a_v1_valid: bool
    g00b_v1_valid: bool
    g00a_g00b_relationship_valid: bool
    historical_semantics: LegacyPublicationSemantics | None = None
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_resolution(self) -> B0ParentResolutionReceiptV1:
        if isinstance(self.boundary, LegacyChecksumAttestedBoundary):
            if self.historical_semantics is None:
                raise ValueError("Legacy parent resolution requires historical semantics.")
        elif self.historical_semantics is not None:
            raise ValueError("Native parent resolution cannot carry legacy semantics.")
        passed = all(
            (
                self.parent_checksum_boundary,
                self.all_consumed_artifacts_covered,
                self.g00a_v1_valid,
                self.g00b_v1_valid,
                self.g00a_g00b_relationship_valid,
            )
        )
        expected_status = "pass" if passed else "fail"
        if self.status != expected_status:
            raise ValueError(f"B0 parent-resolution status must be {expected_status}.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00SourcePlaneV2Amendment(StrictModel):
    """No-model provenance bridge from accepted Dev29 v1 evidence to v2."""

    schema_version: Literal[1] = 1
    amendment_id: str
    parent_g00a_v1_authority_id: str = Field(min_length=1)
    parent_g00a_v1: ArtifactRef
    parent_g00b_v1_virtual_store_id: str = Field(min_length=1)
    parent_g00b_v1_manifest: ArtifactRef
    immutable_source_hashes: tuple[SourceHashBinding, ...]
    v2_guide_target_crosswalk: ArtifactRef
    v2_numerical_audit: ArtifactRef
    v2_source_derivation_receipt: ArtifactRef
    v2_row_locator: ArtifactRef
    builder_implementation_sha256: Sha256
    environment_hash: Sha256
    model_fitting_performed: Literal[False] = False
    protected_outcome_scientific_use: Literal[False] = False

    @model_validator(mode="after")
    def validate_amendment(self) -> G00SourcePlaneV2Amendment:
        source_ids = [source.source_id for source in self.immutable_source_hashes]
        if len(source_ids) != 12 or len(source_ids) != len(set(source_ids)):
            raise ValueError("The GSE314342 amendment must bind exactly 12 unique sources.")
        expected = self.identity(id_field="amendment_id")
        if self.amendment_id != expected:
            raise ValueError(f"amendment_id mismatch: expected {expected}.")
        return self


class G00SourcePlaneV2AmendmentReceipt(StrictModel):
    """Decision receipt for the no-model v1-to-v2 authority upgrade."""

    schema_version: Literal[1] = 1
    receipt_id: str
    amendment_id: str = Field(min_length=1)
    derived_g00a_v2_authority_id: str = Field(min_length=1)
    derived_g00a_v2: ArtifactRef
    derived_g00b_v2_virtual_store_id: str = Field(min_length=1)
    derived_g00b_v2: ArtifactRef
    parent_files_verified: bool
    all_source_hashes_verified: bool
    derivation_receipt_verified: bool
    v2_parent_equality_verified: bool
    model_fitting_performed: Literal[False] = False
    protected_outcome_scientific_use: Literal[False] = False
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_amendment_receipt(self) -> G00SourcePlaneV2AmendmentReceipt:
        passed = (
            self.parent_files_verified
            and self.all_source_hashes_verified
            and self.derivation_receipt_verified
            and self.v2_parent_equality_verified
        )
        expected_status = "pass" if passed else "fail"
        if self.status != expected_status:
            raise ValueError(f"Source-plane amendment status must be {expected_status}.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class VirtualCountSourceV2(StrictModel):
    """Dev31 immutable source record with numerical CSR authority."""

    source_id: str = Field(min_length=1)
    donor_id: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    physical_time_hours: float
    relative_uri: str
    source_file_sha256: Sha256
    dataset_path: str = Field(default="X", min_length=1)
    rows: int = Field(gt=0)
    features: int = Field(gt=0)
    nnz: int = Field(ge=0)
    eligible_rows: int = Field(gt=0)
    eligible_nnz: int = Field(ge=0)
    source_feature_order_hash: Sha256
    canonical_permutation_hash: Sha256
    numeric_integrity: SourceNumericIntegrity

    _safe_uri = field_validator("relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @field_validator("physical_time_hours")
    @classmethod
    def finite_physical_time(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Source physical time must be finite and nonnegative.")
        return value


class G00SourceAuthorityV2(StrictModel):
    """Dev31 G00A source, numeric, feature, and guide-target authority."""

    schema_version: Literal[2] = 2
    authority_id: str
    source_plane_amendment_id: str = Field(min_length=1)
    source_plane_amendment: ArtifactRef
    sources: tuple[VirtualCountSourceV2, ...]
    canonical_feature_index_hash: Sha256
    guide_catalog_hash: Sha256
    target_catalog_hash: Sha256
    guide_target_crosswalk_hash: Sha256
    guide_target_crosswalk: ArtifactRef
    source_numeric_audit: ArtifactRef
    source_derivation_receipt: ArtifactRef
    guide_count: int = Field(gt=0)
    target_control_count: int = Field(gt=0)
    eligibility_rule: Literal["guide_group == targeting single sgRNA AND low_quality == false"]
    eligibility_uses_heldout_stimulated_outcomes: Literal[False] = False
    eligible_row_ids_hash: Sha256
    eligible_rows: int = Field(gt=0)
    eligible_nnz: int = Field(ge=0)
    canonical_row_id_rule: Literal["(sample_index << 32) | source_row_index"] = (
        "(sample_index << 32) | source_row_index"
    )
    access_semantics: ProtectedSourceAccessSemantics
    full_source_hashes_verified: Literal[True] = True
    source_reconciliation_pass: Literal[True] = True
    guide_target_crosswalk_invariants_pass: Literal[True] = True
    model_facing_output: Literal[False] = False

    @model_validator(mode="after")
    def validate_source_authority(self) -> G00SourceAuthorityV2:
        ids = [source.source_id for source in self.sources]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("G00A sources must be nonempty and uniquely identified.")
        if sum(source.eligible_rows for source in self.sources) != self.eligible_rows:
            raise ValueError("G00A source eligible rows do not reconcile to the authority total.")
        if sum(source.eligible_nnz for source in self.sources) != self.eligible_nnz:
            raise ValueError(
                "G00A source eligible nonzeros do not reconcile to the authority total."
            )
        if self.guide_target_crosswalk.sha256 != self.guide_target_crosswalk_hash:
            raise ValueError("Guide-target crosswalk hash must equal its ArtifactRef hash.")
        expected = self.identity(id_field="authority_id")
        if self.authority_id != expected:
            raise ValueError(f"authority_id mismatch: expected {expected}.")
        return self


class VirtualCanonicalCountStoreManifestV2(StrictModel):
    """Dev31 G00B metadata-only access with exact G00A parent binding."""

    schema_version: Literal[2] = 2
    virtual_store_id: str
    backend: Literal["virtual_canonical_h5ad_csr_v2"] = "virtual_canonical_h5ad_csr_v2"
    source_authority_id: str
    source_plane_amendment_id: str = Field(min_length=1)
    source_plane_amendment: ArtifactRef
    canonical_feature_index_hash: Sha256
    guide_catalog_hash: Sha256
    target_catalog_hash: Sha256
    guide_target_crosswalk_hash: Sha256
    guide_target_crosswalk: ArtifactRef
    source_numeric_audit: ArtifactRef
    source_derivation_receipt: ArtifactRef
    guide_count: int = Field(gt=0)
    target_control_count: int = Field(gt=0)
    eligibility_rule: Literal["guide_group == targeting single sgRNA AND low_quality == false"]
    eligibility_uses_heldout_stimulated_outcomes: Literal[False] = False
    eligible_row_ids_hash: Sha256
    eligible_rows: int = Field(gt=0)
    eligible_nnz: int = Field(ge=0)
    features: int = Field(gt=0)
    row_locator: ArtifactRef
    feature_permutations: ArtifactRef
    row_locator_schema: Literal["canonical_row_source_row_guide_target_v2"] = (
        "canonical_row_source_row_guide_target_v2"
    )
    sources: tuple[VirtualCountSourceV2, ...]
    access_semantics: ProtectedSourceAccessSemantics
    raw_counts_materialized: Literal[False] = False
    intended_use: Literal["sequential_statistics_and_fold_materialization"] = (
        "sequential_statistics_and_fold_materialization"
    )
    direct_h100_training_backend: Literal[False] = False
    archival_shard_plan: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_virtual_store(self) -> VirtualCanonicalCountStoreManifestV2:
        ids = [source.source_id for source in self.sources]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Virtual canonical sources must be nonempty and unique.")
        if any(source.features != self.features for source in self.sources):
            raise ValueError("Every virtual source must expose the canonical feature width.")
        if sum(source.eligible_rows for source in self.sources) != self.eligible_rows:
            raise ValueError("Virtual source eligible rows do not reconcile to the manifest.")
        if sum(source.eligible_nnz for source in self.sources) != self.eligible_nnz:
            raise ValueError("Virtual source eligible nonzeros do not reconcile to the manifest.")
        if self.guide_target_crosswalk.sha256 != self.guide_target_crosswalk_hash:
            raise ValueError("Guide-target crosswalk hash must equal its ArtifactRef hash.")
        expected = self.identity(id_field="virtual_store_id")
        if self.virtual_store_id != expected:
            raise ValueError(f"virtual_store_id mismatch: expected {expected}.")
        return self


class FoldRowRoleRecord(StrictModel):
    """One frozen row role in a fold-native extraction."""

    role: Literal[
        "training_fit",
        "training_validation",
        "heldout_source_query",
        "protected_heldout_stimulated",
    ]
    rows: int = Field(gt=0)
    row_ids_hash: Sha256


class TrainingOnlyFeatureSelectionContract(StrictModel):
    """No-outer-outcome ranked-prefix feature selection protocol."""

    method: Literal["training_only_ranked_prefix_v1"] = "training_only_ranked_prefix_v1"
    implementation_sha256: Sha256
    fit_rows_hash: Sha256
    validation_rows_hash: Sha256
    candidate_feature_counts: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    selection_metric: Literal["training_validation_negative_log_likelihood"] = (
        "training_validation_negative_log_likelihood"
    )
    minimum_improvement_margin: float = Field(ge=0)
    paired_refit_draws: Literal[59] = 59
    maximum_feature_count: Literal[4096] = 4096
    paired_statistic: Literal["p95_absolute_paired_nll_difference_to_4096"] = (
        "p95_absolute_paired_nll_difference_to_4096"
    )
    selection_rule: Literal["smallest_prefix_with_p95_difference_le_margin"] = (
        "smallest_prefix_with_p95_difference_le_margin"
    )
    ordered_feature_table: ArtifactRef
    custom001_puror_role: Literal["technical_assay_feature"] = "technical_assay_feature"
    custom001_puror_in_primary_biological_metric: Literal[False] = False
    custom001_puror_sidecar_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_candidates(self) -> TrainingOnlyFeatureSelectionContract:
        if self.candidate_feature_counts != (256, 512, 1024, 2048, 4096):
            raise ValueError("Feature candidates must equal the frozen positive prefix grid.")
        return self


class TrainingOnlySampleSizeSelectionContract(StrictModel):
    """Paired-refit saturation rule for the training-only cell budget."""

    comparison: Literal["NLL_N_minus_NLL_Nmax"] = "NLL_N_minus_NLL_Nmax"
    equivalence_statistic: Literal["p95_absolute_paired_nll_difference"] = (
        "p95_absolute_paired_nll_difference"
    )
    saturation_rule: Literal["smallest_N_with_p95_absolute_difference_le_epsilon"] = (
        "smallest_N_with_p95_absolute_difference_le_epsilon"
    )
    paired_refit_draws: Literal[59] = 59
    equivalence_epsilon: float = Field(gt=0)
    candidate_cells: tuple[int, ...]
    two_million_extension_triggered_by_no_saturation: bool = False

    @model_validator(mode="after")
    def validate_grid(self) -> TrainingOnlySampleSizeSelectionContract:
        base = (50_000, 100_000, 250_000, 500_000, 1_000_000)
        allowed = {base, (*base, 2_000_000)}
        if self.candidate_cells not in allowed:
            raise ValueError("Sample-size grid must be the frozen G00C grid.")
        includes_extension = 2_000_000 in self.candidate_cells
        if includes_extension != self.two_million_extension_triggered_by_no_saturation:
            raise ValueError("The two-million-cell extension requires documented nonsaturation.")
        return self


class CompactSamplerContract(StrictModel):
    """Exact weighted, resumable fold-native sampler semantics."""

    implementation_sha256: Sha256
    microbatch_cells: Literal[512] = 512
    microbatches_per_update: Literal[8] = 8
    macrobatch_cells: Literal[4096] = 4096
    donor_weighting: Literal["equal"] = "equal"
    checkpoint_weighting: Literal["equal"] = "equal"
    target_weighting: Literal["equal"] = "equal"
    guide_weighting_within_target: Literal["equal"] = "equal"
    rng_algorithm: str = Field(min_length=1)
    rng_seed: int = Field(ge=0)
    thinning_rule: str = Field(min_length=1)
    resume_cursor_schema: str = Field(min_length=1)
    resume_cursor_initial_hash: Sha256

    @model_validator(mode="after")
    def validate_batch(self) -> CompactSamplerContract:
        if self.microbatch_cells * self.microbatches_per_update != self.macrobatch_cells:
            raise ValueError("Sampler microbatches must multiply to the macrobatch size.")
        return self


class FoldNativeCompactViewContractV2(StrictModel):
    """Dev31 G00C training-only selection and materialization contract."""

    schema_version: Literal[2] = 2
    fold_view_id: str
    parent_source_authority_id: str
    parent_virtual_store_id: str
    source_plane_amendment_id: str = Field(min_length=1)
    source_plane_amendment: ArtifactRef
    source_plane_amendment_receipt_id: str = Field(min_length=1)
    source_plane_amendment_receipt: ArtifactRef
    parent_eligible_row_ids_hash: Sha256
    parent_guide_target_crosswalk_hash: Sha256
    outer_split_id: str
    training_donor_ids: tuple[str, ...]
    heldout_donor_id: str
    row_roles: tuple[FoldRowRoleRecord, ...]
    row_role_audit: ArtifactRef
    row_role_assignment_implementation_sha256: Sha256
    feature_selection: TrainingOnlyFeatureSelectionContract
    sample_size_selection: TrainingOnlySampleSizeSelectionContract
    sampler: CompactSamplerContract
    count_dtype: Literal["uint16", "int32"]
    index_dtype: Literal["uint16", "int32"]
    maximum_observed_count: int = Field(ge=0)
    physical_layout: Literal["donor_checkpoint_target_guide_row"]
    protected_outer_stimulated_expression_values_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_fold_view(self) -> FoldNativeCompactViewContractV2:
        if (
            not self.training_donor_ids
            or len(self.training_donor_ids) != len(set(self.training_donor_ids))
            or self.heldout_donor_id in self.training_donor_ids
        ):
            raise ValueError("Fold-native donor roles must be nonempty, unique, and disjoint.")
        roles = [record.role for record in self.row_roles]
        expected_roles = {
            "training_fit",
            "training_validation",
            "heldout_source_query",
            "protected_heldout_stimulated",
        }
        if len(roles) != len(expected_roles) or set(roles) != expected_roles:
            raise ValueError("Fold-native row-role audit must contain every role exactly once.")
        uint16_safe = self.maximum_observed_count <= 65_535
        if self.count_dtype == "uint16" and not uint16_safe:
            raise ValueError("uint16 counts require maximum_observed_count <= 65535.")
        expected = self.identity(id_field="fold_view_id")
        if self.fold_view_id != expected:
            raise ValueError(f"fold_view_id mismatch: expected {expected}.")
        return self


class G00CFeatureSelectionResult(StrictModel):
    """Hash-bound evidence and derived decision for one ranked-prefix curve."""

    curve: ArtifactRef
    paired_refit_draws: ArtifactRef
    ordered_features: ArtifactRef
    fit_rows_hash: Sha256
    validation_rows_hash: Sha256
    selected_feature_count: int = Field(gt=0, le=4096)


class G00CSampleSizeSelectionResult(StrictModel):
    """Hash-bound evidence and derived decision for one training-cell scale curve."""

    curve: ArtifactRef
    paired_refit_draws: ArtifactRef
    selected_training_rows: ArtifactRef
    training_scale_row_order: ArtifactRef
    selected_training_cells: int = Field(gt=0)
    two_million_extension_opened: bool


class G00CSamplerEvidence(StrictModel):
    """Compact deterministic row/weight/RNG/resume sequence evidence."""

    sampler_plan: ArtifactRef
    epoch_index: ArtifactRef
    uninterrupted_draw_trace: ArtifactRef
    resumed_draw_trace: ArtifactRef
    uninterrupted_state_trace: ArtifactRef
    resumed_state_trace: ArtifactRef
    resume_test: ArtifactRef


class G00CExecutionBundle(StrictModel):
    """Complete non-model G00C materialization evidence."""

    schema_version: Literal[1] = 1
    bundle_id: str
    fold_view_id: str = Field(min_length=1)
    row_roles: ArtifactRef
    feature_selection: G00CFeatureSelectionResult
    sample_size_selection: G00CSampleSizeSelectionResult
    sampler: G00CSamplerEvidence
    compact_payload: ArtifactRef
    publication_manifest: ArtifactRef
    reload_receipt: ArtifactRef
    compact_payload_schema: Literal["g00c_compact_csr_hdf5_v1"] = "g00c_compact_csr_hdf5_v1"
    compact_payload_rows: int = Field(gt=0)
    compact_payload_features: int = Field(gt=0, le=4096)
    compact_payload_row_ids_hash: Sha256
    compact_payload_feature_order_hash: Sha256
    compact_payload_counts_sha256: Sha256
    protected_rows_in_compact_payload: Literal[0] = 0
    maximum_observed_count: int = Field(ge=0)
    count_dtype: Literal["uint16", "int32"]
    index_dtype: Literal["uint16", "int32"]

    @model_validator(mode="after")
    def validate_bundle(self) -> G00CExecutionBundle:
        if self.count_dtype == "uint16" and self.maximum_observed_count > 65_535:
            raise ValueError("G00C uint16 payload exceeds the observed-count limit.")
        expected = self.identity(id_field="bundle_id")
        if self.bundle_id != expected:
            raise ValueError(f"bundle_id mismatch: expected {expected}.")
        return self


class G00CDecisionReceipt(StrictModel):
    """Derived G00C execution decision; no biological outcome is represented."""

    schema_version: Literal[1] = 1
    receipt_id: str
    fold_view_id: str = Field(min_length=1)
    execution_bundle_id: str = Field(min_length=1)
    row_roles_verified: bool
    feature_selection_verified: bool
    sample_size_selection_verified: bool
    sampler_sequence_verified: bool
    compact_counts_verified: bool
    protected_rows_absent: bool
    immutable_publication_verified: bool
    full_reload_verified: bool
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_decision(self) -> G00CDecisionReceipt:
        passed = all(
            (
                self.row_roles_verified,
                self.feature_selection_verified,
                self.sample_size_selection_verified,
                self.sampler_sequence_verified,
                self.compact_counts_verified,
                self.protected_rows_absent,
                self.immutable_publication_verified,
                self.full_reload_verified,
            )
        )
        expected_status = "pass" if passed else "fail"
        if self.status != expected_status:
            raise ValueError(f"G00C decision status must be {expected_status}.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class RefitReplayPolicyV1(StrictModel):
    """Pre-result replay sample and implementation for one paired-refit family."""

    implementation_sha256: Sha256
    preregistered_audit_draw_ids: tuple[int, ...]
    selected_candidate_all_draws: Literal[True] = True
    reference_candidate_all_draws: Literal[True] = True
    exact_nll_replay_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_draws(self) -> RefitReplayPolicyV1:
        if (
            not self.preregistered_audit_draw_ids
            or tuple(sorted(set(self.preregistered_audit_draw_ids)))
            != self.preregistered_audit_draw_ids
            or self.preregistered_audit_draw_ids[0] < 0
            or self.preregistered_audit_draw_ids[-1] >= 59
        ):
            raise ValueError("Refit replay audit draws must be unique, sorted, and inside 0..58.")
        return self


class TrainingOnlyFeatureSelectionContractV2(TrainingOnlyFeatureSelectionContract):
    """Dev33 ranked-prefix contract with preregistered executable refit replay."""

    # Intentional schema-version discriminator narrowing for the additive contract.
    method: Literal["training_only_ranked_prefix_v2"] = "training_only_ranked_prefix_v2"  # type: ignore[assignment]
    refit_replay: RefitReplayPolicyV1


class TrainingOnlySampleSizeSelectionContractV2(TrainingOnlySampleSizeSelectionContract):
    """Dev33 two-stage saturation contract that cannot pass at the base maximum."""

    # Intentional schema-version discriminator narrowing for the additive contract.
    saturation_rule: Literal["submaximum_or_extension_required_v2"] = (
        "submaximum_or_extension_required_v2"  # type: ignore[assignment]
    )
    grid_stage: Literal["base", "extension"] = "base"
    base_grid_extension_required_receipt: ArtifactRef | None = None
    refit_replay: RefitReplayPolicyV1

    @model_validator(mode="after")
    def validate_extension_parent(self) -> TrainingOnlySampleSizeSelectionContractV2:
        if self.grid_stage == "base":
            if self.candidate_cells != (50_000, 100_000, 250_000, 500_000, 1_000_000):
                raise ValueError("The base sample-size stage must use the frozen base grid.")
            if self.base_grid_extension_required_receipt is not None:
                raise ValueError("The base stage cannot bind an extension-parent receipt.")
        else:
            if self.candidate_cells != (
                50_000,
                100_000,
                250_000,
                500_000,
                1_000_000,
                2_000_000,
            ):
                raise ValueError("The extension stage must add exactly two million cells.")
            if self.base_grid_extension_required_receipt is None:
                raise ValueError("The extension stage requires the passed base-grid stop receipt.")
        return self


class FoldNativeCompactViewContractV3(FoldNativeCompactViewContractV2):
    """Dev33 fail-closed fold contract for ordered, bounded, replayed G00C evidence."""

    # Intentional schema-version discriminator narrowing for the additive contract.
    schema_version: Literal[3] = 3  # type: ignore[assignment]
    feature_selection: TrainingOnlyFeatureSelectionContractV2
    sample_size_selection: TrainingOnlySampleSizeSelectionContractV2
    physical_order_fields: Literal[
        "donor_id,physical_time_hours,checkpoint,target_id,guide_id,source_row,row_id"
    ] = "donor_id,physical_time_hours,checkpoint,target_id,guide_id,source_row,row_id"
    compact_verification_block_rows: int = Field(default=8192, ge=1, le=65_536)
    compact_verifier_implementation_sha256: Sha256
    maximum_verifier_rss_bytes: int = Field(gt=0)
    maximum_source_handles: int = Field(ge=1)


class G00CFeatureSelectionResultV2(StrictModel):
    """Dev33 feature curve plus full refit provenance."""

    curve: ArtifactRef
    refit_records: ArtifactRef
    ordered_features: ArtifactRef
    fit_rows_hash: Sha256
    validation_rows_hash: Sha256
    selected_feature_count: int = Field(gt=0, le=4096)


class G00CSampleSizeSelectionResultV2(StrictModel):
    """Dev33 sample-size decision, including the mandatory extension stop state."""

    curve: ArtifactRef
    refit_records: ArtifactRef
    selected_training_rows: ArtifactRef | None = None
    training_scale_row_order: ArtifactRef
    selection_status: Literal["selected", "extension_required"]
    selected_training_cells: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_selection_state(self) -> G00CSampleSizeSelectionResultV2:
        if self.selection_status == "selected":
            if self.selected_training_cells is None or self.selected_training_rows is None:
                raise ValueError("A selected sample size requires its exact row artifact.")
        elif self.selected_training_cells is not None or self.selected_training_rows is not None:
            raise ValueError("An extension-required decision cannot select rows or a cell count.")
        return self


class G00CCompactVerificationReceiptV2(StrictModel):
    """Observed bounded-memory, ordered, exact compact-payload verification."""

    schema_version: Literal[2] = 2
    receipt_id: str
    fold_view_id: str = Field(min_length=1)
    compact_payload_sha256: Sha256
    verifier_implementation_sha256: Sha256
    verifier_block_rows: int = Field(ge=1)
    maximum_loaded_nonzeros: int = Field(ge=0)
    total_rows_checked: int = Field(gt=0)
    total_nonzeros_checked: int = Field(ge=0)
    total_count_sum: int = Field(ge=0)
    peak_process_rss_bytes: int = Field(gt=0)
    maximum_open_source_handles: int = Field(ge=1)
    compact_row_set_sha256: Sha256
    compact_ordered_row_ids_sha256: Sha256
    contiguous_physical_blocks: int = Field(gt=0)
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_receipt(self) -> G00CCompactVerificationReceiptV2:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CExecutionBundleV2(StrictModel):
    """Dev33 non-model materialization evidence with unambiguous row identities."""

    schema_version: Literal[2] = 2
    bundle_id: str
    fold_view_id: str = Field(min_length=1)
    row_roles: ArtifactRef
    feature_selection: G00CFeatureSelectionResultV2
    sample_size_selection: G00CSampleSizeSelectionResultV2
    sampler: G00CSamplerEvidence
    compact_payload: ArtifactRef
    compact_verification_receipt: ArtifactRef
    publication_manifest: ArtifactRef
    reload_receipt: ArtifactRef
    compact_payload_schema: Literal["g00c_compact_csr_hdf5_v2"] = "g00c_compact_csr_hdf5_v2"
    compact_payload_rows: int = Field(gt=0)
    compact_payload_features: int = Field(gt=0, le=4096)
    compact_row_set_sha256: Sha256
    compact_ordered_row_ids_sha256: Sha256
    compact_payload_feature_order_hash: Sha256
    compact_payload_counts_sha256: Sha256
    protected_rows_in_compact_payload: Literal[0] = 0
    maximum_observed_count: int = Field(ge=0)
    count_dtype: Literal["uint16", "int32"]
    index_dtype: Literal["uint16", "int32"]

    @model_validator(mode="after")
    def validate_bundle(self) -> G00CExecutionBundleV2:
        if self.count_dtype == "uint16" and self.maximum_observed_count > 65_535:
            raise ValueError("G00C uint16 payload exceeds the observed-count limit.")
        expected = self.identity(id_field="bundle_id")
        if self.bundle_id != expected:
            raise ValueError(f"bundle_id mismatch: expected {expected}.")
        return self


class G00CDecisionReceiptV2(StrictModel):
    """Dev33 G00C decision with replay, order, and streaming gates."""

    schema_version: Literal[2] = 2
    receipt_id: str
    fold_view_id: str = Field(min_length=1)
    execution_bundle_id: str = Field(min_length=1)
    row_roles_verified: bool
    feature_selection_verified: bool
    sample_size_selection_verified: bool
    refit_replay_verified: bool
    sampler_sequence_verified: bool
    physical_order_verified: bool
    streaming_verification_verified: bool
    compact_counts_verified: bool
    protected_rows_absent: bool
    immutable_publication_verified: bool
    full_reload_verified: bool
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_decision(self) -> G00CDecisionReceiptV2:
        passed = all(
            (
                self.row_roles_verified,
                self.feature_selection_verified,
                self.sample_size_selection_verified,
                self.refit_replay_verified,
                self.sampler_sequence_verified,
                self.physical_order_verified,
                self.streaming_verification_verified,
                self.compact_counts_verified,
                self.protected_rows_absent,
                self.immutable_publication_verified,
                self.full_reload_verified,
            )
        )
        expected_status = "pass" if passed else "fail"
        if self.status != expected_status:
            raise ValueError(f"G00C decision status must be {expected_status}.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CParentBindingV1(StrictModel):
    """One immutable B0-A2 parent used by the Dev34 G00C selection freeze."""

    role: Literal[
        "g00a_v2",
        "g00b_v2",
        "source_plane_amendment",
        "source_plane_amendment_receipt",
        "source_plane_decision_receipt",
        "b0_a2_execution_amendment",
    ]
    identity_field: str = Field(min_length=1)
    identity_value: str = Field(min_length=1)
    artifact: ArtifactRef


class G00CCanaryPrerequisiteV1(StrictModel):
    """Finalized Dev33-B prerequisite that is forbidden as a claim parent."""

    dev33_code_commit: GitCommit
    dev33_wheel_sha256: Sha256
    canary_contract_id: Sha256
    execution_record_commit: GitCommit
    independent_verification_id: Sha256
    final_audit_id: Sha256
    authority_archive_uri: str = Field(min_length=1)
    authority_archive_sha256: Sha256
    evidence_role: Literal["engineering_canary"] = "engineering_canary"
    prerequisite_satisfied: Literal[True] = True
    promotion_eligible: Literal[False] = False
    parent_eligible: Literal[False] = False
    feature_artifact_reuse_permitted: Literal[False] = False
    row_artifact_reuse_permitted: Literal[False] = False
    sampler_artifact_reuse_permitted: Literal[False] = False
    compact_payload_reuse_permitted: Literal[False] = False


class G00CRefitSeedRecordV1(StrictModel):
    """Expanded deterministic seed streams for one paired refit draw."""

    draw_id: int = Field(ge=0, le=58)
    initialization: int = Field(ge=0, le=2**64 - 1)
    training_sampler: int = Field(ge=0, le=2**64 - 1)
    thinning: int = Field(ge=0, le=2**64 - 1)
    validation_evaluation: int = Field(ge=0, le=2**64 - 1)
    stochastic_optimizer_or_augmentation: int = Field(ge=0, le=2**64 - 1)
    restart_interruption_point: int = Field(ge=0, le=2**64 - 1)


class G00CRefitSeedScheduleV1(StrictModel):
    """Hash-bound, pre-result seed schedule shared by every paired candidate."""

    schema_version: Literal[1] = 1
    schedule_id: str
    derivation: Literal["sha256_namespace_fold_stage_draw_stream_uint64_be_v1"] = (
        "sha256_namespace_fold_stage_draw_stream_uint64_be_v1"
    )
    derivation_namespace_id: Sha256
    fold_id: Literal["lodo-D1"] = "lodo-D1"
    stage: Literal["g00c_feature_and_cell_selection"] = "g00c_feature_and_cell_selection"
    records: tuple[G00CRefitSeedRecordV1, ...]
    paired_across_candidates: Literal[True] = True
    selected_candidate_replay_draw_ids: tuple[int, ...] = tuple(range(59))
    reference_candidate_replay_draw_ids: tuple[int, ...] = tuple(range(59))
    other_candidate_replay_draw_ids: tuple[int, ...] = (
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        58,
    )

    @model_validator(mode="after")
    def validate_schedule(self) -> G00CRefitSeedScheduleV1:
        if tuple(record.draw_id for record in self.records) != tuple(range(59)):
            raise ValueError("Dev34 seed schedule requires ordered draw IDs 0..58.")
        values = [
            seed
            for record in self.records
            for seed in (
                record.initialization,
                record.training_sampler,
                record.thinning,
                record.validation_evaluation,
                record.stochastic_optimizer_or_augmentation,
                record.restart_interruption_point,
            )
        ]
        if len(values) != len(set(values)):
            raise ValueError("Dev34 expanded seed streams must not collide.")
        if self.selected_candidate_replay_draw_ids != tuple(range(59)):
            raise ValueError("Dev34 must replay all selected-candidate draws.")
        if self.reference_candidate_replay_draw_ids != tuple(range(59)):
            raise ValueError("Dev34 must replay all reference-candidate draws.")
        if self.other_candidate_replay_draw_ids != (0, 6, 12, 18, 24, 30, 36, 42, 48, 58):
            raise ValueError("Dev34 nonselected replay draws differ from the frozen subset.")
        expected = self.identity(id_field="schedule_id")
        if self.schedule_id != expected:
            raise ValueError(f"schedule_id mismatch: expected {expected}.")
        return self


class G00CFeatureRankingFreezeV1(StrictModel):
    """Training-only 4,096-feature ranking at the frozen one-million-row scale."""

    method: Literal["checkpoint_conditioned_poisson_deviance_v1"] = (
        "checkpoint_conditioned_poisson_deviance_v1"
    )
    library_size_offset: Literal["log_total_primary_umi"] = "log_total_primary_umi"
    checkpoint_effect: Literal["fixed_intercept_per_checkpoint"] = "fixed_intercept_per_checkpoint"
    score: Literal["summed_poisson_deviance_from_checkpoint_null"] = (
        "summed_poisson_deviance_from_checkpoint_null"
    )
    tie_breaks: tuple[Literal["total_umi_desc", "detection_count_desc", "feature_id_utf8_asc"], ...]
    fit_reference_cells: Literal[1_000_000] = 1_000_000
    fit_reference_rows_hash: Sha256
    validation_rows_hash: Sha256
    candidate_feature_counts: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    reference_feature_count: Literal[4096] = 4096
    custom001_puror_is_sidecar: Literal[True] = True
    custom001_puror_in_primary_metric: Literal[False] = False

    @model_validator(mode="after")
    def validate_ranking(self) -> G00CFeatureRankingFreezeV1:
        if self.candidate_feature_counts != (256, 512, 1024, 2048, 4096):
            raise ValueError("Dev34 feature candidates must equal the frozen prefix grid.")
        if self.tie_breaks != (
            "total_umi_desc",
            "detection_count_desc",
            "feature_id_utf8_asc",
        ):
            raise ValueError("Dev34 feature ranking requires the complete frozen tie-break order.")
        return self


class G00CCommonSupportFeatureMetricV1(StrictModel):
    """Common 4,096-feature scoring surface for every prefix candidate."""

    reference_feature_count: Literal[4096] = 4096
    candidate_feature_counts: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    candidate_distribution: Literal["prefix_plus_residual_category_v1"] = (
        "prefix_plus_residual_category_v1"
    )
    omitted_mass_expansion: Literal[
        "frozen_checkpoint_specific_training_only_frequency_vector_v1"
    ] = "frozen_checkpoint_specific_training_only_frequency_vector_v1"
    checkpoint_ids: tuple[Literal["Rest", "Stim8hr", "Stim48hr"], ...] = (
        "Rest",
        "Stim8hr",
        "Stim48hr",
    )
    residual_frequency_artifact: ArtifactRef
    residual_frequency_fit_rows_hash: Sha256
    validation_support: Literal["same_frozen_4096_features_for_every_candidate"] = (
        "same_frozen_4096_features_for_every_candidate"
    )
    validation_count_total: Literal["identical_across_feature_candidates"] = (
        "identical_across_feature_candidates"
    )
    metric_unit: Literal["nats_per_weighted_validation_count"] = (
        "nats_per_weighted_validation_count"
    )
    inverse_probability_weights_paired: Literal[True] = True
    custom001_puror_excluded: Literal[True] = True
    comparison: Literal["q95_absolute_paired_nll_difference_to_4096"] = (
        "q95_absolute_paired_nll_difference_to_4096"
    )
    feature_equivalence_epsilon: float = Field(default=0.0001, json_schema_extra={"const": 0.0001})
    selection_rule: Literal["smallest_qualifying_prefix_v1"] = "smallest_qualifying_prefix_v1"

    @model_validator(mode="after")
    def validate_metric(self) -> G00CCommonSupportFeatureMetricV1:
        if self.candidate_feature_counts != (256, 512, 1024, 2048, 4096):
            raise ValueError("Dev34 common-support candidates must equal the frozen prefix grid.")
        if self.checkpoint_ids != ("Rest", "Stim8hr", "Stim48hr"):
            raise ValueError("Dev34 common-support checkpoints must follow physical chronology.")
        if self.feature_equivalence_epsilon != 0.0001:
            raise ValueError("Dev34 feature equivalence epsilon must remain 1e-4.")
        return self


class G00CSerialSelectionFreezeV1(StrictModel):
    """Feature selection must become an immutable parent of cell selection."""

    serial_dependency: Literal["feature_then_cell_budget_v1"] = "feature_then_cell_budget_v1"
    frozen_nested_training_row_order_hash: Sha256
    feature_selection_reference_rows_hash: Sha256
    feature_selection_training_cells: Literal[1_000_000] = 1_000_000
    feature_selection_uses_first_rows_of_frozen_nested_order: Literal[True] = True
    feature_result_frozen_before_cell_curve_access: Literal[True] = True
    ranking_recomputed_per_cell_candidate: Literal[False] = False
    feature_width_changed_per_cell_candidate: Literal[False] = False
    sample_result_binds_parent_feature_selection_result_sha256: Literal[True] = True
    sample_result_binds_selected_feature_count: Literal[True] = True
    sample_result_binds_selected_feature_order_sha256: Literal[True] = True
    smaller_cell_budget_interpretation: Literal[
        "model_fit_budget_conditional_on_one_million_cell_feature_selection"
    ] = "model_fit_budget_conditional_on_one_million_cell_feature_selection"


class G00CSelectionMarginFreezeV1(StrictModel):
    """Fixed, pre-result feature and cell equivalence margins."""

    metric_unit: Literal["nats_per_weighted_validation_count"] = (
        "nats_per_weighted_validation_count"
    )
    feature_equivalence_epsilon: float = Field(default=0.0001, json_schema_extra={"const": 0.0001})
    cell_equivalence_epsilon: float = Field(default=0.0001, json_schema_extra={"const": 0.0001})
    candidate_curve_values_accessed: Literal[False] = False
    cohort_expression_values_accessed: Literal[False] = False
    heldout_stimulated_expression_values_accessed: Literal[False] = False

    @model_validator(mode="after")
    def validate_margins(self) -> G00CSelectionMarginFreezeV1:
        if self.feature_equivalence_epsilon != 0.0001 or self.cell_equivalence_epsilon != 0.0001:
            raise ValueError("Dev34 feature and cell equivalence margins must remain 1e-4.")
        return self


class G00CRefitFreezeV1(StrictModel):
    """Exact paired-refit identity shared by feature and cell selection."""

    seed_schedule_id: Sha256
    seed_schedule_artifact: ArtifactRef
    count_thinning_method: Literal["paired_binomial_half_count_v1"] = (
        "paired_binomial_half_count_v1"
    )
    model_family: Literal["checkpoint_conditioned_multinomial_intercept_v1"] = (
        "checkpoint_conditioned_multinomial_intercept_v1"
    )
    initial_state: Literal["closed_form_zero_state"] = "closed_form_zero_state"
    optimizer: Literal["closed_form_no_optimizer"] = "closed_form_no_optimizer"
    maximum_updates: Literal[0] = 0
    pseudocount: float = Field(default=0.5, gt=0.0)
    model_config_hash: Sha256
    replay_implementation_sha256: Sha256
    exact_replay_required: Literal[True] = True


class G00CFeatureSelectionResultV3(StrictModel):
    """Common-support feature result that can parent sample-size selection."""

    schema_version: Literal[3] = 3
    result_id: str
    curve: ArtifactRef
    refit_records: ArtifactRef
    ordered_features: ArtifactRef
    common_support_metric_receipt: ArtifactRef
    fit_rows_hash: Sha256
    validation_rows_hash: Sha256
    selected_feature_count: Literal[256, 512, 1024, 2048, 4096]
    selected_feature_order_sha256: Sha256
    metric_unit: Literal["nats_per_weighted_validation_count"] = (
        "nats_per_weighted_validation_count"
    )
    validation_count_total_identical_across_candidates: Literal[True] = True
    selected_at_reference: bool

    @model_validator(mode="after")
    def validate_feature_result(self) -> G00CFeatureSelectionResultV3:
        if self.selected_at_reference != (self.selected_feature_count == 4096):
            raise ValueError("selected_at_reference disagrees with selected feature count.")
        expected = self.identity(id_field="result_id")
        if self.result_id != expected:
            raise ValueError(f"result_id mismatch: expected {expected}.")
        return self


class G00CSampleSizeSelectionResultV3(StrictModel):
    """Serially bound sample result with a terminal no-saturation state."""

    schema_version: Literal[3] = 3
    result_id: str
    grid_stage: Literal["base", "extension"]
    curve: ArtifactRef
    refit_records: ArtifactRef
    training_scale_row_order: ArtifactRef
    parent_feature_selection_result_sha256: Sha256
    selected_feature_count: Literal[256, 512, 1024, 2048, 4096]
    selected_feature_order_sha256: Sha256
    serial_dependency: Literal["feature_then_cell_budget_v1"] = "feature_then_cell_budget_v1"
    base_grid_extension_required_receipt: ArtifactRef | None = None
    selection_status: Literal["selected", "extension_required", "fail_no_saturation"]
    selected_training_rows: ArtifactRef | None = None
    selected_training_cells: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_sample_result(self) -> G00CSampleSizeSelectionResultV3:
        if self.grid_stage == "base":
            if self.base_grid_extension_required_receipt is not None:
                raise ValueError("The base sample-size result cannot bind an extension parent.")
            if self.selection_status == "fail_no_saturation":
                raise ValueError("Base-grid nonsaturation must emit extension_required.")
            if self.selection_status == "selected" and self.selected_training_cells not in (
                50_000,
                100_000,
                250_000,
                500_000,
            ):
                raise ValueError("A base selection must be a frozen sub-million candidate.")
        else:
            if self.base_grid_extension_required_receipt is None:
                raise ValueError("The extension result requires the base stop receipt.")
            if self.selection_status == "extension_required":
                raise ValueError("A third sample-size extension is forbidden.")
            if self.selection_status == "selected" and self.selected_training_cells not in (
                50_000,
                100_000,
                250_000,
                500_000,
                1_000_000,
            ):
                raise ValueError("The extension reference alone is fail_no_saturation.")
        if self.selection_status == "selected":
            if self.selected_training_cells is None or self.selected_training_rows is None:
                raise ValueError("A selected sample size requires its exact row artifact.")
        elif self.selected_training_cells is not None or self.selected_training_rows is not None:
            raise ValueError("A nonselected sample-size result cannot publish selected rows.")
        expected = self.identity(id_field="result_id")
        if self.result_id != expected:
            raise ValueError(f"result_id mismatch: expected {expected}.")
        return self


class G00CSupportAuditFreezeV1(StrictModel):
    """Mandatory support diagnostics for every candidate row budget."""

    dimensions: tuple[
        Literal[
            "donor_checkpoint",
            "target",
            "guide",
            "control_vs_targeting",
            "sampler_stratum",
        ],
        ...,
    ]
    candidate_cell_counts: tuple[int, ...] = (
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
    )
    report_zero_support_strata: Literal[True] = True
    reject_declared_zero_support_strata: Literal[True] = True
    silent_weight_renormalization_permitted: Literal[False] = False
    report_minimum_and_maximum_cells_per_stratum: Literal[True] = True
    report_weight_distribution: Literal[True] = True
    report_weighted_effective_sample_size: Literal[True] = True
    report_maximum_to_median_weight_ratio: Literal[True] = True

    @model_validator(mode="after")
    def validate_support(self) -> G00CSupportAuditFreezeV1:
        expected = (
            "donor_checkpoint",
            "target",
            "guide",
            "control_vs_targeting",
            "sampler_stratum",
        )
        if self.dimensions != expected:
            raise ValueError("Dev34 support audit dimensions must equal the frozen ordered set.")
        return self


class G00CSupportAuditContractV2(StrictModel):
    """Stage-scoped support authority; two-million rows exist only after extension freeze."""

    schema_version: Literal[2] = 2
    contract_id: Sha256
    stage: Literal["base", "extension"]
    dimensions: tuple[str, ...] = (
        "donor_checkpoint",
        "target",
        "guide",
        "control_vs_targeting",
        "sampler_stratum",
    )
    feature_candidate_counts: tuple[int, ...]
    cell_candidate_counts: tuple[int, ...]
    report_zero_support_strata: Literal[True] = True
    reject_zero_support_candidate: Literal[True] = True
    silent_weight_renormalization_permitted: Literal[False] = False

    @model_validator(mode="after")
    def validate_stage(self) -> G00CSupportAuditContractV2:
        expected_dimensions = (
            "donor_checkpoint",
            "target",
            "guide",
            "control_vs_targeting",
            "sampler_stratum",
        )
        if self.dimensions != expected_dimensions:
            raise ValueError("Dev35 support dimensions differ from the frozen ordered set.")
        if self.stage == "base":
            if self.feature_candidate_counts != (256, 512, 1024, 2048, 4096) or (
                self.cell_candidate_counts != (50_000, 100_000, 250_000, 500_000, 1_000_000)
            ):
                raise ValueError("Dev35 base support surface changed its candidate grids.")
        elif self.feature_candidate_counts or self.cell_candidate_counts != (2_000_000,):
            raise ValueError("Dev35 extension support surface must contain only two million.")
        expected = self.identity(id_field="contract_id")
        if self.contract_id != expected:
            raise ValueError(f"contract_id mismatch: expected {expected}.")
        return self


class G00CMonitorFreezeV1(StrictModel):
    """Process-tree monitor completeness and bounded-memory contract."""

    implementation_sha256: Sha256
    polling_interval_milliseconds: int = Field(ge=10, le=1000)
    maximum_unreadable_samples: int = Field(ge=0)
    maximum_unreadable_fraction: float = Field(ge=0.0, le=0.01)
    maximum_consecutive_unreadable_samples: int = Field(ge=0, le=10)
    maximum_temporal_gap_milliseconds: int = Field(ge=10, le=5000)
    process_tree_coverage_required: Literal[True] = True
    child_process_aggregation_required: Literal[True] = True
    expression_access_start_covered: Literal[True] = True
    expression_access_end_covered: Literal[True] = True
    maximum_process_tree_rss_bytes: int = Field(gt=0)


class G00CPublicationFreezeV1(StrictModel):
    """Fresh-workspace publication and status-dependent evidence surface."""

    fresh_workspace_required: Literal[True] = True
    candidate_workspace_rename_permitted: Literal[False] = False
    canary_payload_reuse_permitted: Literal[False] = False
    canary_selection_artifact_reuse_permitted: Literal[False] = False
    selected_view_rebuilt_from_g00b: Literal[True] = True
    manifest_last: Literal[True] = True
    no_clobber: Literal[True] = True
    always_required_artifacts: tuple[str, ...]
    pass_only_required_artifacts: tuple[str, ...]
    extension_required_artifacts: tuple[str, ...]
    fail_no_saturation_required_artifacts: tuple[str, ...]
    selected_artifacts_forbidden_on_nonpass: Literal[True] = True
    terminal_statuses: tuple[
        Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"], ...
    ] = ("pass", "extension_required", "fail_no_saturation", "failed_integrity")
    stop_before_g00d: Literal[True] = True

    @model_validator(mode="after")
    def validate_publication(self) -> G00CPublicationFreezeV1:
        always = {
            "G00C_REFIT_SEED_SCHEDULE.json",
            "ROW_HASHES.json",
            "FEATURE_SELECTION_CURVE.parquet",
            "CELL_SELECTION_CURVE.parquet",
            "REFIT_PROVENANCE.parquet",
            "REPLAY_AUDIT.json",
            "SUPPORT_AUDIT.parquet",
            "SAMPLER_RESTART.json",
            "DECISION_RECEIPT.json",
            "artifacts.json",
            "COMMITTED",
            "SHA256SUMS",
        }
        passed = {
            "SELECTED_FEATURES.parquet",
            "SELECTED_ROWS.parquet",
            "compact.h5",
            "puroR-sidecar.h5",
            "PHYSICAL_RUNS.parquet",
            "COMPACT_VERIFICATION.json",
            "WRITER_RESTART.json",
        }
        extension = {"BASE_GRID_STOP_RECEIPT.json"}
        nonsaturation = {"BASE_GRID_STOP_RECEIPT.json", "EXTENSION_RESULT.json"}
        observed = (
            self.always_required_artifacts,
            self.pass_only_required_artifacts,
            self.extension_required_artifacts,
            self.fail_no_saturation_required_artifacts,
        )
        expected = (always, passed, extension, nonsaturation)
        if any(
            set(items) != required or len(items) != len(required)
            for items, required in zip(observed, expected, strict=True)
        ):
            raise ValueError("Dev34 publication artifacts differ from the frozen status surfaces.")
        return self


class G00CSelectionFreezeContractV1(StrictModel):
    """Dev34-A no-expression contract for one promotion-eligible G00C fold."""

    schema_version: Literal[1] = 1
    freeze_id: str
    parent_bindings: tuple[G00CParentBindingV1, ...]
    dev33_canary: G00CCanaryPrerequisiteV1
    outer_split_id: Literal["lodo-D1"] = "lodo-D1"
    training_donor_ids: tuple[Literal["D2", "D3", "D4"], ...] = ("D2", "D3", "D4")
    heldout_donor_id: Literal["D1"] = "D1"
    parent_eligible_rows: Literal[21_996_842] = 21_996_842
    row_roles: tuple[FoldRowRoleRecord, ...]
    row_role_freeze: ArtifactRef
    training_scale_row_order_hash: Sha256
    seed_schedule: G00CRefitSeedScheduleV1
    seed_schedule_artifact: ArtifactRef
    feature_ranking: G00CFeatureRankingFreezeV1
    common_support_metric: G00CCommonSupportFeatureMetricV1
    serial_selection: G00CSerialSelectionFreezeV1
    selection_margins: G00CSelectionMarginFreezeV1
    refits: G00CRefitFreezeV1
    base_cell_grid: tuple[int, ...] = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    extension_additional_cell_grid: tuple[int, ...] = (2_000_000,)
    extension_requires_new_contract: Literal[True] = True
    third_extension_permitted: Literal[False] = False
    support_audit: G00CSupportAuditFreezeV1
    monitor: G00CMonitorFreezeV1
    publication: G00CPublicationFreezeV1
    protected_heldout_stimulated_expression_values_accessed_during_freeze: Literal[False] = False
    any_cohort_expression_values_accessed_during_freeze: Literal[False] = False
    execution_backend: Literal["cpu_only"] = "cpu_only"
    g00c_status: Literal["contract_frozen_not_run"] = "contract_frozen_not_run"
    g00d_status: Literal["blocked"] = "blocked"
    g04_g07_g08_status: Literal["blocked"] = "blocked"
    biological_claims: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> G00CSelectionFreezeContractV1:
        expected_roles = (
            "g00a_v2",
            "g00b_v2",
            "source_plane_amendment",
            "source_plane_amendment_receipt",
            "source_plane_decision_receipt",
            "b0_a2_execution_amendment",
        )
        if tuple(binding.role for binding in self.parent_bindings) != expected_roles:
            raise ValueError("Dev34 requires the six ordered B0-A2 parent bindings.")
        if self.training_donor_ids != ("D2", "D3", "D4"):
            raise ValueError("Dev34 fold 0 requires ordered training donors D2, D3, D4.")
        roles = tuple(record.role for record in self.row_roles)
        expected_row_roles = (
            "training_fit",
            "training_validation",
            "heldout_source_query",
            "protected_heldout_stimulated",
        )
        if (
            roles != expected_row_roles
            or sum(record.rows for record in self.row_roles) != self.parent_eligible_rows
        ):
            raise ValueError("Dev34 row roles must be ordered, complete, and reconcile to G00B.")
        if self.base_cell_grid != (50_000, 100_000, 250_000, 500_000, 1_000_000):
            raise ValueError("Dev34 base cell grid differs from the frozen grid.")
        if self.extension_additional_cell_grid != (2_000_000,):
            raise ValueError("Dev34 extension must add exactly two million cells.")
        validation_hash = next(
            record.row_ids_hash for record in self.row_roles if record.role == "training_validation"
        )
        if self.feature_ranking.validation_rows_hash != validation_hash:
            raise ValueError("Dev34 feature validation rows differ from the frozen role.")
        if (
            self.serial_selection.frozen_nested_training_row_order_hash
            != self.training_scale_row_order_hash
        ):
            raise ValueError("Dev34 serial selection must bind the frozen nested row order.")
        if (
            self.serial_selection.feature_selection_reference_rows_hash
            != self.feature_ranking.fit_reference_rows_hash
        ):
            raise ValueError("Dev34 serial selection must bind the one-million-row prefix.")
        if (
            self.common_support_metric.residual_frequency_fit_rows_hash
            != self.feature_ranking.fit_reference_rows_hash
        ):
            raise ValueError("Dev34 residual frequencies must use the frozen ranking rows.")
        if self.refits.seed_schedule_id != self.seed_schedule.schedule_id:
            raise ValueError("Dev34 refits must bind the expanded seed schedule.")
        expected = self.identity(id_field="freeze_id")
        if self.freeze_id != expected:
            raise ValueError(f"freeze_id mismatch: expected {expected}.")
        return self


class G00CCommonSupportPriorV2(StrictModel):
    """Dev35 prior that is identical per primary gene on the 4,096-gene support."""

    schema_version: Literal[2] = 2
    reference_feature_count: Literal[4096] = 4096
    per_feature_pseudocount: float = Field(default=0.5, json_schema_extra={"const": 0.5})
    modeled_feature_prior: Literal["0.5_per_primary_feature"] = "0.5_per_primary_feature"
    residual_category_prior: Literal["0.5_times_omitted_feature_count"] = (
        "0.5_times_omitted_feature_count"
    )
    residual_frequency_estimator: Literal[
        "checkpoint_training_counts_plus_0.5_per_omitted_feature_v1"
    ] = "checkpoint_training_counts_plus_0.5_per_omitted_feature_v1"
    residual_frequencies_strictly_positive: Literal[True] = True
    common_support_probabilities_strictly_positive: Literal[True] = True
    simulation_sensitivity_required_before_expression_access: Literal[False] = False
    selection_threshold: float = Field(default=0.0001, json_schema_extra={"const": 0.0001})

    @model_validator(mode="after")
    def validate_prior(self) -> G00CCommonSupportPriorV2:
        if self.per_feature_pseudocount != 0.5 or self.selection_threshold != 0.0001:
            raise ValueError("Dev35 prior and selection threshold must remain exactly frozen.")
        return self


class G00CCommonSupportMetricReceiptV1(StrictModel):
    """Hash-bound proof that every feature candidate used one scoring alphabet."""

    schema_version: Literal[1] = 1
    receipt_id: Sha256
    selection_freeze_id: Sha256
    seed_schedule_id: Sha256
    seed_schedule_artifact: ArtifactRef
    refit_records: ArtifactRef
    validation_counts: ArtifactRef
    residual_frequencies: ArtifactRef
    prior_sensitivity: ArtifactRef
    prior: G00CCommonSupportPriorV2
    candidate_feature_counts: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    validation_total_count_hash: Sha256
    validation_total_identical_across_candidates: Literal[True] = True
    reference_draw_differences_zero_tolerance: float = Field(default=1e-12, gt=0.0, le=1e-10)
    probabilities_finite_and_strictly_positive: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_receipt(self) -> G00CCommonSupportMetricReceiptV1:
        if self.candidate_feature_counts != (256, 512, 1024, 2048, 4096):
            raise ValueError("Dev35 common-support receipt has a different feature grid.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSampleSizeExtensionFreezeV1(StrictModel):
    """Pre-access authority for the single permitted two-million-row extension."""

    schema_version: Literal[1] = 1
    extension_freeze_id: Sha256
    base_selection_freeze_id: Sha256
    base_execution_bundle: ArtifactRef
    base_extension_required_receipt: ArtifactRef
    base_feature_selection_result: ArtifactRef
    selected_feature_count: Literal[256, 512, 1024, 2048, 4096]
    selected_feature_order_sha256: Sha256
    selected_feature_surface: ArtifactRef
    seed_schedule_id: Sha256
    seed_schedule_artifact: ArtifactRef
    seed_schedule_policy: Literal["unchanged_from_base"] = "unchanged_from_base"
    base_candidate_cells: tuple[int, ...] = (50_000, 100_000, 250_000, 500_000, 1_000_000)
    added_candidate_cells: tuple[int, ...] = (2_000_000,)
    base_support_audit: ArtifactRef
    extension_support_audit_contract: ArtifactRef
    third_extension_permitted: Literal[False] = False
    fresh_attempt_id: str = Field(min_length=1)
    publication_root_uri: str = Field(min_length=1)
    expression_values_accessed_during_freeze: Literal[False] = False
    status: Literal["extension_frozen_not_run"] = "extension_frozen_not_run"

    @field_validator("publication_root_uri")
    @classmethod
    def safe_publication_root(cls, value: str) -> str:
        return validate_relative_uri(value)

    @model_validator(mode="after")
    def validate_extension(self) -> G00CSampleSizeExtensionFreezeV1:
        if self.base_candidate_cells != (50_000, 100_000, 250_000, 500_000, 1_000_000):
            raise ValueError("Dev35 extension changed the frozen base grid.")
        if self.added_candidate_cells != (2_000_000,):
            raise ValueError("Dev35 permits exactly one two-million-row extension.")
        expected = self.identity(id_field="extension_freeze_id")
        if self.extension_freeze_id != expected:
            raise ValueError(f"extension_freeze_id mismatch: expected {expected}.")
        return self


class G00CImplementationBindingV1(StrictModel):
    """One role-labelled implementation artifact in the pre-access authority."""

    role: Literal[
        "feature_ranking",
        "residual_frequency",
        "refit",
        "sampler",
        "monitor",
        "execution_verifier",
        "decision_verifier",
    ]
    artifact: ArtifactRef


class G00CImplementationAuthorityV1(StrictModel):
    """Exact code and runtime identities required before G00C expression access."""

    dev35_code_commit: GitCommit
    wheel: ArtifactRef
    normalized_sdist: ArtifactRef
    implementation_tree_sha256: Sha256
    environment_lock: ArtifactRef
    environment_kind: Literal["exact_local_lock", "oci_container"]
    execution_environment_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    implementations: tuple[G00CImplementationBindingV1, ...]

    @model_validator(mode="after")
    def validate_implementations(self) -> G00CImplementationAuthorityV1:
        expected = (
            "feature_ranking",
            "residual_frequency",
            "refit",
            "sampler",
            "monitor",
            "execution_verifier",
            "decision_verifier",
        )
        if tuple(binding.role for binding in self.implementations) != expected:
            raise ValueError("Dev35 implementation roles must equal the frozen ordered set.")
        return self


class G00CD1ExecutionAuthorityFreezeV1(StrictModel):
    """Concrete metadata-only D1 authority layered over the Dev34 design freeze."""

    schema_version: Literal[1] = 1
    authority_id: Sha256
    selection_freeze: ArtifactRef
    selection_freeze_id: Sha256
    outer_split_id: Literal["lodo-D1"] = "lodo-D1"
    row_role_freeze: ArtifactRef
    row_roles: tuple[FoldRowRoleRecord, ...]
    nested_training_row_order: ArtifactRef
    nested_training_row_order_hash: Sha256
    feature_reference_rows: ArtifactRef
    feature_reference_rows_hash: Sha256
    feature_reference_rows_are_first_million: Literal[True] = True
    seed_schedule: ArtifactRef
    seed_schedule_id: Sha256
    common_support_prior: G00CCommonSupportPriorV2
    base_support_audit_contract: ArtifactRef
    implementation: G00CImplementationAuthorityV1
    fresh_attempt_id: str = Field(min_length=1)
    publication_root_uri: str = Field(min_length=1)
    prior_attempt_artifact_reuse_permitted: Literal[False] = False
    expression_values_accessed_during_freeze: Literal[False] = False
    protected_heldout_expression_values_accessed_during_freeze: Literal[False] = False
    status: Literal["finalized_preaccess_not_executed"] = "finalized_preaccess_not_executed"
    expression_access_authorized_by_this_record: Literal[False] = False
    biological_claims: Literal[False] = False

    @field_validator("publication_root_uri")
    @classmethod
    def safe_authority_publication_root(cls, value: str) -> str:
        return validate_relative_uri(value)

    @model_validator(mode="after")
    def validate_authority(self) -> G00CD1ExecutionAuthorityFreezeV1:
        expected_roles = (
            "training_fit",
            "training_validation",
            "heldout_source_query",
            "protected_heldout_stimulated",
        )
        if tuple(record.role for record in self.row_roles) != expected_roles:
            raise ValueError("Dev35 D1 authority requires the four ordered row roles.")
        expected = self.identity(id_field="authority_id")
        if self.authority_id != expected:
            raise ValueError(f"authority_id mismatch: expected {expected}.")
        return self


class G00CExecutionBundleV3(StrictModel):
    """Dev35 authoritative chain from freeze through V3 selections and publication."""

    schema_version: Literal[3] = 3
    bundle_id: Sha256
    execution_authority: ArtifactRef
    execution_authority_id: Sha256
    selection_freeze: ArtifactRef
    selection_freeze_id: Sha256
    seed_schedule: ArtifactRef
    seed_schedule_id: Sha256
    row_roles: ArtifactRef
    feature_selection_result: ArtifactRef
    feature_selection_result_id: Sha256
    sample_size_selection_result: ArtifactRef
    sample_size_selection_result_id: Sha256
    common_support_receipt: ArtifactRef
    base_support_audit_contract: ArtifactRef
    base_support_audit: ArtifactRef
    extension_support_audit_contract: ArtifactRef | None = None
    extension_support_audit: ArtifactRef | None = None
    sampler_evidence: ArtifactRef
    publication_manifest: ArtifactRef
    extension_freeze: ArtifactRef | None = None
    compact_payload: ArtifactRef | None = None
    compact_verification_receipt: ArtifactRef | None = None
    reload_receipt: ArtifactRef | None = None
    failure_receipt: ArtifactRef | None = None
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]

    @model_validator(mode="after")
    def validate_bundle(self) -> G00CExecutionBundleV3:
        pass_artifacts = (
            self.compact_payload,
            self.compact_verification_receipt,
            self.reload_receipt,
        )
        if self.terminal_status == "pass":
            if (
                any(artifact is None for artifact in pass_artifacts)
                or self.failure_receipt is not None
            ):
                raise ValueError(
                    "A Dev35 pass requires all materialization evidence and no failure."
                )
        elif any(artifact is not None for artifact in pass_artifacts):
            raise ValueError("A nonpass Dev35 bundle cannot publish selected compact artifacts.")
        if self.terminal_status == "fail_no_saturation" and self.extension_freeze is None:
            raise ValueError("Terminal nonsaturation requires its extension authority.")
        if self.terminal_status == "extension_required" and self.extension_freeze is not None:
            raise ValueError("The base stop precedes creation of the extension authority.")
        if self.terminal_status == "failed_integrity":
            if self.failure_receipt is None:
                raise ValueError("An integrity failure requires a bound failure receipt.")
        elif self.failure_receipt is not None:
            raise ValueError("Only failed_integrity may bind a failure receipt.")
        extension_support = (
            self.extension_support_audit_contract,
            self.extension_support_audit,
        )
        if self.extension_freeze is None and any(
            artifact is not None for artifact in extension_support
        ):
            raise ValueError("Two-million support evidence requires a frozen extension.")
        if self.extension_freeze is not None and any(
            artifact is None for artifact in extension_support
        ):
            raise ValueError("A frozen extension requires its separate support evidence.")
        expected = self.identity(id_field="bundle_id")
        if self.bundle_id != expected:
            raise ValueError(f"bundle_id mismatch: expected {expected}.")
        return self


class G00CDecisionReceiptV3(StrictModel):
    """Artifact-derived Dev35 terminal decision and sole G00D parent gate."""

    schema_version: Literal[3] = 3
    receipt_id: Sha256
    execution_bundle_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    feature_selection_result_id: Sha256
    sample_size_selection_result_id: Sha256
    seed_schedule_id: Sha256
    common_support_receipt_sha256: Sha256
    support_audit_sha256s: tuple[Sha256, ...]
    verifier_implementation_sha256: Sha256
    verified_artifact_sha256s: tuple[Sha256, ...]
    grid_stage: Literal["base", "extension"]
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]
    verification_completed: Literal[True] = True
    may_parent_g00d: bool
    biological_claims: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> G00CDecisionReceiptV3:
        if not self.verified_artifact_sha256s or len(self.verified_artifact_sha256s) != len(
            set(self.verified_artifact_sha256s)
        ):
            raise ValueError("Dev35 verified artifact hashes must be nonempty and unique.")
        expected_support_hashes = 2 if self.grid_stage == "extension" else 1
        if (
            len(self.support_audit_sha256s) != expected_support_hashes
            or len(set(self.support_audit_sha256s)) != expected_support_hashes
        ):
            raise ValueError("Dev35 decision has the wrong stage-scoped support hashes.")
        if self.may_parent_g00d != (self.terminal_status == "pass"):
            raise ValueError("Only a fully verified Dev35 pass may parent G00D.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CImplementationBindingV2(StrictModel):
    """One role-labelled Dev36 executable whose bytes are part of authority."""

    role: Literal[
        "feature_ranking",
        "residual_frequency",
        "refit",
        "sampler",
        "support_auditor",
        "monitor",
        "materialization_verifier",
        "refit_replay_verifier",
        "publication_verifier",
        "execution_verifier",
        "decision_verifier",
    ]
    artifact: ArtifactRef


class G00CImplementationAuthorityV2(StrictModel):
    """Dev36 code/runtime authority with explicit seal implementations."""

    schema_version: Literal[2] = 2
    dev36_code_commit: GitCommit
    wheel: ArtifactRef
    normalized_sdist: ArtifactRef
    implementation_tree_sha256: Sha256
    environment_lock: ArtifactRef
    environment_kind: Literal["exact_local_lock", "oci_container"]
    execution_environment_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    implementations: tuple[G00CImplementationBindingV2, ...]

    @model_validator(mode="after")
    def validate_implementations(self) -> G00CImplementationAuthorityV2:
        expected = (
            "feature_ranking",
            "residual_frequency",
            "refit",
            "sampler",
            "support_auditor",
            "monitor",
            "materialization_verifier",
            "refit_replay_verifier",
            "publication_verifier",
            "execution_verifier",
            "decision_verifier",
        )
        if tuple(binding.role for binding in self.implementations) != expected:
            raise ValueError("Dev36 implementation roles must equal the sealed ordered set.")
        by_role = {binding.role: binding.artifact.sha256 for binding in self.implementations}
        if by_role["materialization_verifier"] == by_role["execution_verifier"]:
            raise ValueError(
                "Dev36 materialization verification must be a distinct source artifact."
            )
        if by_role["sampler"] == by_role["execution_verifier"]:
            raise ValueError("Dev36 sampler must be an independently bound implementation.")
        return self


class G00CD1ExecutionAuthorityFreezeV2(StrictModel):
    """Concrete Dev36 metadata-only D1 authority with every seal implementation."""

    schema_version: Literal[2] = 2
    authority_id: Sha256
    selection_freeze: ArtifactRef
    selection_freeze_id: Sha256
    outer_split_id: Literal["lodo-D1"] = "lodo-D1"
    row_role_freeze: ArtifactRef
    row_roles: tuple[FoldRowRoleRecord, ...]
    nested_training_row_order: ArtifactRef
    nested_training_row_order_hash: Sha256
    feature_reference_rows: ArtifactRef
    feature_reference_rows_hash: Sha256
    feature_reference_rows_are_first_million: Literal[True] = True
    sampler_row_hierarchy: ArtifactRef
    sampler_hierarchy_rows: Literal[16_924_672] = 16_924_672
    seed_schedule: ArtifactRef
    seed_schedule_id: Sha256
    common_support_prior: G00CCommonSupportPriorV2
    base_support_audit_contract: ArtifactRef
    implementation: G00CImplementationAuthorityV2
    monitored_process_tree_ceiling_bytes: Literal[68719476736] = 68_719_476_736
    monitored_process_tree_ceiling_semantics: Literal[
        "claim_bearing_full_grid_process_tree_rss_ceiling"
    ] = "claim_bearing_full_grid_process_tree_rss_ceiling"
    fresh_attempt_id: str = Field(min_length=1)
    publication_root_uri: str = Field(min_length=1)
    authority_archive_required: Literal[True] = True
    prior_attempt_artifact_reuse_permitted: Literal[False] = False
    expression_values_accessed_during_freeze: Literal[False] = False
    protected_heldout_expression_values_accessed_during_freeze: Literal[False] = False
    status: Literal["finalized_preaccess_not_executed"] = "finalized_preaccess_not_executed"
    expression_access_authorized_by_this_record: Literal[False] = False
    biological_claims: Literal[False] = False

    @field_validator("publication_root_uri")
    @classmethod
    def safe_dev36_publication_root(cls, value: str) -> str:
        return validate_relative_uri(value)

    @model_validator(mode="after")
    def validate_authority(self) -> G00CD1ExecutionAuthorityFreezeV2:
        expected_roles = (
            "training_fit",
            "training_validation",
            "heldout_source_query",
            "protected_heldout_stimulated",
        )
        if tuple(record.role for record in self.row_roles) != expected_roles:
            raise ValueError("Dev36 D1 authority requires the four ordered row roles.")
        training_rows = next(
            record.rows for record in self.row_roles if record.role == "training_fit"
        )
        if training_rows != self.sampler_hierarchy_rows:
            raise ValueError("Dev36 sampler hierarchy must cover every training-fit row.")
        expected = self.identity(id_field="authority_id")
        if self.authority_id != expected:
            raise ValueError(f"authority_id mismatch: expected {expected}.")
        return self


class G00CSamplerPlanEntryV3(StrictModel):
    """One exact hierarchical-sampler replay surface."""

    candidate_kind: Literal["feature_count", "training_cells"]
    candidate_value: int = Field(gt=0)
    refit_draw_id: int = Field(ge=0, le=58)
    macro_updates: int = Field(ge=1)


class G00CSamplerPlanV3(StrictModel):
    """Frozen micro/macro boundaries and six-stream sampler schedule."""

    schema_version: Literal[3] = 3
    plan_id: Sha256
    execution_authority_id: Sha256
    seed_schedule_id: Sha256
    entries: tuple[G00CSamplerPlanEntryV3, ...]
    microbatch_cells: Literal[512] = 512
    microbatches_per_macro_update: Literal[8] = 8
    macrobatch_cells: Literal[4096] = 4096
    hierarchy: Literal["donor_checkpoint_then_target_then_guide_then_row_equal_v1"] = (
        "donor_checkpoint_then_target_then_guide_then_row_equal_v1"
    )
    inverse_probability_weights: Literal["exact_raw_inverse_draw_probability"] = (
        "exact_raw_inverse_draw_probability"
    )
    thinning: Literal["binomial_half_count_from_frozen_stream"] = (
        "binomial_half_count_from_frozen_stream"
    )
    resume_after_macro_update: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_plan(self) -> G00CSamplerPlanV3:
        keys = [
            (entry.candidate_kind, entry.candidate_value, entry.refit_draw_id)
            for entry in self.entries
        ]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Dev36 sampler-plan entries must be nonempty and unique.")
        if self.resume_after_macro_update >= min(entry.macro_updates for entry in self.entries):
            raise ValueError("Dev36 sampler resume point must precede every terminal cursor.")
        expected = self.identity(id_field="plan_id")
        if self.plan_id != expected:
            raise ValueError(f"plan_id mismatch: expected {expected}.")
        return self


class G00CSamplerEvidenceV3(StrictModel):
    """Exact uninterrupted/resumed trace and state evidence for the frozen sampler."""

    schema_version: Literal[3] = 3
    evidence_id: Sha256
    execution_authority_id: Sha256
    sampler_plan: ArtifactRef
    sampler_plan_id: Sha256
    row_hierarchy: ArtifactRef
    nested_training_row_order: ArtifactRef
    uninterrupted_draw_trace: ArtifactRef
    resumed_draw_trace: ArtifactRef
    uninterrupted_state_trace: ArtifactRef
    resumed_state_trace: ArtifactRef
    implementation_sha256: Sha256
    ordered_draws_identical: Literal[True] = True
    rng_states_identical: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_evidence(self) -> G00CSamplerEvidenceV3:
        expected = self.identity(id_field="evidence_id")
        if self.evidence_id != expected:
            raise ValueError(f"evidence_id mismatch: expected {expected}.")
        return self


class G00CRefitReplayReceiptV3(StrictModel):
    """Executable closed-form replay evidence, not a self-declared refit table."""

    schema_version: Literal[3] = 3
    receipt_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    seed_schedule_id: Sha256
    candidate_kind: Literal["feature_count", "training_cells"]
    selected_candidate: int = Field(gt=0)
    reference_candidate: int = Field(gt=0)
    modeled_feature_count: Literal[256, 512, 1024, 2048, 4096]
    refit_records: ArtifactRef
    sufficient_statistics: ArtifactRef
    replayed_rows: ArtifactRef
    replay_implementation_sha256: Sha256
    preregistered_audit_draw_ids: tuple[int, ...] = (
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        58,
    )
    selected_and_reference_all_59_draws: Literal[True] = True
    exact_nll_and_state_hash_equality: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_replay(self) -> G00CRefitReplayReceiptV3:
        if self.preregistered_audit_draw_ids != (0, 6, 12, 18, 24, 30, 36, 42, 48, 58):
            raise ValueError("Dev36 refit replay changed the preregistered audit draws.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSupportAuditReceiptV3(StrictModel):
    """Support metrics independently derived from a verified sampler trace."""

    schema_version: Literal[3] = 3
    receipt_id: Sha256
    execution_authority_id: Sha256
    support_contract: ArtifactRef
    support_contract_id: Sha256
    sampler_evidence: ArtifactRef
    sampler_evidence_id: Sha256
    derived_support_table: ArtifactRef
    implementation_sha256: Sha256
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_support_receipt(self) -> G00CSupportAuditReceiptV3:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CMaterializationReceiptV3(StrictModel):
    """Complete direct source-byte verification of a selected compact view."""

    schema_version: Literal[3] = 3
    receipt_id: Sha256
    execution_authority_id: Sha256
    compact_payload: ArtifactRef
    puro_r_sidecar: ArtifactRef
    selected_features: ArtifactRef
    selected_rows: ArtifactRef
    physical_runs: ArtifactRef
    selected_feature_count: int = Field(gt=0, le=4096)
    compact_row_count: int = Field(gt=0)
    puro_r_feature_id: Literal["CUSTOM001_PuroR"] = "CUSTOM001_PuroR"
    puro_r_canonical_index: int = Field(ge=0)
    compact_verification_block_rows: int = Field(ge=1, le=65536)
    compact_data_dtype: Literal["uint16", "int32"]
    compact_index_dtype: Literal["uint16", "int32"]
    maximum_observed_count: int = Field(ge=0)
    selected_row_set_sha256: Sha256
    compact_ordered_row_ids_sha256: Sha256
    protected_row_ids_sha256: Sha256
    uninterrupted_compact_payload: ArtifactRef
    resumed_compact_payload: ArtifactRef
    uninterrupted_puro_r_sidecar: ArtifactRef
    resumed_puro_r_sidecar: ArtifactRef
    protected_access_receipt: ArtifactRef
    reload_receipt: ArtifactRef
    verifier_implementation_sha256: Sha256
    bounded_source_equality: Literal[True] = True
    primary_sidecar_separation_verified: Literal[True] = True
    writer_restart_byte_identical: Literal[True] = True
    physical_order_verified: Literal[True] = True
    protected_expression_reads: Literal[0] = 0
    immutable_publication_verified: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_materialization(self) -> G00CMaterializationReceiptV3:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CPublicationArtifactV3(StrictModel):
    """One immutable payload in the status-dependent publication inventory."""

    name: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def safe_publication_name(cls, value: str) -> str:
        return validate_relative_uri(value)


class G00CPublicationManifestV3(StrictModel):
    """Manifest-last exact publication inventory for one terminal status."""

    schema_version: Literal[3] = 3
    manifest_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    feature_selection_result_id: Sha256
    sample_size_selection_result_id: Sha256
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]
    artifacts: tuple[G00CPublicationArtifactV3, ...]
    sha256sums: ArtifactRef
    committed: ArtifactRef
    publication_event_receipt: ArtifactRef
    publisher_implementation_sha256: Sha256
    manifest_written_last: Literal[True] = True
    destination_preexisted: Literal[False] = False
    no_clobber: Literal[True] = True

    @model_validator(mode="after")
    def validate_publication_manifest(self) -> G00CPublicationManifestV3:
        names = tuple(item.name for item in self.artifacts)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("Dev36 publication artifacts must be uniquely sorted by name.")
        expected = self.identity(id_field="manifest_id")
        if self.manifest_id != expected:
            raise ValueError(f"manifest_id mismatch: expected {expected}.")
        return self


class G00CExecutionBundleV4(StrictModel):
    """Dev36 sealed chain; every opaque Dev35 gate is replaced by typed evidence."""

    schema_version: Literal[4] = 4
    bundle_id: Sha256
    execution_authority: ArtifactRef
    execution_authority_id: Sha256
    selection_freeze: ArtifactRef
    selection_freeze_id: Sha256
    seed_schedule: ArtifactRef
    seed_schedule_id: Sha256
    row_roles: ArtifactRef
    feature_selection_result: ArtifactRef
    feature_selection_result_id: Sha256
    sample_size_selection_result: ArtifactRef
    sample_size_selection_result_id: Sha256
    common_support_receipt: ArtifactRef
    feature_refit_replay_receipt: ArtifactRef
    sample_refit_replay_receipt: ArtifactRef
    sampler_evidence: ArtifactRef
    base_support_audit_contract: ArtifactRef
    base_support_audit_receipt: ArtifactRef
    extension_freeze: ArtifactRef | None = None
    extension_support_audit_contract: ArtifactRef | None = None
    extension_support_audit_receipt: ArtifactRef | None = None
    materialization_receipt: ArtifactRef | None = None
    publication_manifest: ArtifactRef
    failure_receipt: ArtifactRef | None = None
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]

    @model_validator(mode="after")
    def validate_bundle(self) -> G00CExecutionBundleV4:
        if self.terminal_status == "pass":
            if self.materialization_receipt is None or self.failure_receipt is not None:
                raise ValueError("A Dev36 pass requires typed materialization and no failure.")
        elif self.materialization_receipt is not None:
            raise ValueError("A nonpass Dev36 bundle cannot bind materialization evidence.")
        if self.terminal_status == "failed_integrity":
            if self.failure_receipt is None:
                raise ValueError("A Dev36 integrity failure requires its typed failure receipt.")
        elif self.failure_receipt is not None:
            raise ValueError("Only failed_integrity may bind a failure receipt.")
        extension = (
            self.extension_freeze,
            self.extension_support_audit_contract,
            self.extension_support_audit_receipt,
        )
        if self.terminal_status == "fail_no_saturation" and any(item is None for item in extension):
            raise ValueError("Dev36 nonsaturation requires its complete extension chain.")
        if self.extension_freeze is None and any(item is not None for item in extension[1:]):
            raise ValueError("Dev36 extension evidence requires a frozen extension.")
        expected = self.identity(id_field="bundle_id")
        if self.bundle_id != expected:
            raise ValueError(f"bundle_id mismatch: expected {expected}.")
        return self


class G00CDecisionReceiptV4(StrictModel):
    """Artifact-derived Dev36 decision and sole successor G00D parent gate."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    execution_bundle_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    feature_selection_result_id: Sha256
    sample_size_selection_result_id: Sha256
    sampler_evidence_id: Sha256
    feature_refit_replay_receipt_id: Sha256
    sample_refit_replay_receipt_id: Sha256
    support_audit_receipt_ids: tuple[Sha256, ...]
    materialization_receipt_id: Sha256 | None = None
    publication_manifest_id: Sha256
    verified_artifact_sha256s: tuple[Sha256, ...]
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]
    may_parent_g00d: bool
    verification_completed: Literal[True] = True
    biological_claims: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> G00CDecisionReceiptV4:
        if not self.verified_artifact_sha256s or len(self.verified_artifact_sha256s) != len(
            set(self.verified_artifact_sha256s)
        ):
            raise ValueError("Dev36 verified artifact hashes must be nonempty and unique.")
        passed = self.terminal_status == "pass"
        if (
            self.may_parent_g00d != passed
            or (self.materialization_receipt_id is not None) != passed
        ):
            raise ValueError("Only a sealed Dev36 pass may parent G00D.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSourceFileBindingV4(StrictModel):
    """One ordered immutable source in the accepted G00B plane."""

    source_id: str = Field(min_length=1)
    checkpoint: Literal["Rest", "Stim8hr", "Stim48hr"]
    source_file_sha256: Sha256


class G00CSourcePlaneBindingV4(StrictModel):
    """Dev37 descriptor from which the verifier constructs the source store itself."""

    schema_version: Literal[4] = 4
    binding_id: Sha256
    accepted_g00b_parent: ArtifactRef
    accepted_g00b_manifest_sha256: Sha256
    virtual_store_id: Sha256
    source_authority_id: Sha256
    virtual_store_relative_uri: str = Field(min_length=1)
    canonical_feature_index: ArtifactRef
    canonical_feature_index_hash: Sha256
    row_locator_sha256: Sha256
    feature_permutations_sha256: Sha256
    guide_target_crosswalk_sha256: Sha256
    source_files: tuple[G00CSourceFileBindingV4, ...]
    feature_count: int = Field(gt=1)
    puro_r_feature_id: Literal["CUSTOM001_PuroR"] = "CUSTOM001_PuroR"
    puro_r_canonical_index: int = Field(ge=0)
    full_store_verification_required: Literal[True] = True

    _safe_store_uri = field_validator("virtual_store_relative_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @model_validator(mode="after")
    def validate_binding(self) -> G00CSourcePlaneBindingV4:
        if self.accepted_g00b_parent.sha256 != self.accepted_g00b_manifest_sha256:
            raise ValueError("Dev37 G00B parent and manifest hashes must be identical.")
        if self.puro_r_canonical_index >= self.feature_count:
            raise ValueError("Dev37 PuroR index lies outside the canonical feature index.")
        keys = [(item.source_id, item.checkpoint) for item in self.source_files]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Dev37 source bindings must be nonempty and unique.")
        expected = self.identity(id_field="binding_id")
        if self.binding_id != expected:
            raise ValueError(f"binding_id mismatch: expected {expected}.")
        return self


class G00CSamplerPlanEntryV4(StrictModel):
    """One pre-access-fixed draw surface and its exact terminal cursor."""

    candidate_kind: Literal["feature_count", "training_cells"]
    candidate_value: int = Field(gt=0)
    refit_draw_id: int = Field(ge=0, le=58)
    macro_updates: int = Field(ge=2)
    expected_trace_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_entry(self) -> G00CSamplerPlanEntryV4:
        if self.expected_trace_rows != self.macro_updates * 4096:
            raise ValueError("Dev37 sampler trace size must equal macro_updates * 4096.")
        return self


class G00CSamplerPlanV4(StrictModel):
    """Complete base or extension sampler plan frozen before expression access."""

    schema_version: Literal[4] = 4
    plan_id: Sha256
    execution_authority_namespace: Sha256
    seed_schedule_id: Sha256
    grid_stage: Literal["base", "extension"]
    entries: tuple[G00CSamplerPlanEntryV4, ...]
    resume_after_macro_update: int = Field(ge=1)
    microbatch_cells: Literal[512] = 512
    microbatches_per_macro_update: Literal[8] = 8
    macrobatch_cells: Literal[4096] = 4096
    hierarchy: Literal["donor_checkpoint_then_target_then_guide_then_row_equal_v1"] = (
        "donor_checkpoint_then_target_then_guide_then_row_equal_v1"
    )
    inverse_probability_weights: Literal["exact_raw_inverse_draw_probability"] = (
        "exact_raw_inverse_draw_probability"
    )
    thinning: Literal["ordered_entry_stream_binomial_half_count_v2"] = (
        "ordered_entry_stream_binomial_half_count_v2"
    )
    expected_total_trace_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_plan(self) -> G00CSamplerPlanV4:
        keys = [
            (entry.candidate_kind, entry.candidate_value, entry.refit_draw_id)
            for entry in self.entries
        ]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Dev37 sampler entries must be nonempty and unique.")
        expected_candidates = (
            {
                "feature_count": (256, 512, 1024, 2048, 4096),
                "training_cells": (50_000, 100_000, 250_000, 500_000, 1_000_000),
            }
            if self.grid_stage == "base"
            else {"training_cells": (2_000_000,)}
        )
        expected_keys = {
            (kind, value, draw)
            for kind, values in expected_candidates.items()
            for value in values
            for draw in range(59)
        }
        if set(keys) != expected_keys:
            raise ValueError("Dev37 sampler plan does not equal the stage's frozen grid.")
        if self.resume_after_macro_update >= min(item.macro_updates for item in self.entries):
            raise ValueError("Dev37 resume cursor must precede every terminal cursor.")
        if self.expected_total_trace_rows != sum(item.expected_trace_rows for item in self.entries):
            raise ValueError("Dev37 sampler total trace size does not reconcile.")
        expected = self.identity(id_field="plan_id")
        if self.plan_id != expected:
            raise ValueError(f"plan_id mismatch: expected {expected}.")
        return self


class G00CHierarchyDerivationReceiptV4(StrictModel):
    """Independent reconstruction of every hierarchy field from G00B metadata."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    source_binding_id: Sha256
    hierarchy: ArtifactRef
    row_count: int = Field(gt=0)
    ordered_row_ids_sha256: Sha256
    source_index_sha256: Sha256
    target_code_sha256: Sha256
    guide_code_sha256: Sha256
    is_control_sha256: Sha256
    locator_and_crosswalk_rederived: Literal[True] = True
    expression_values_accessed: Literal[False] = False
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_receipt(self) -> G00CHierarchyDerivationReceiptV4:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CImplementationBindingV3(StrictModel):
    """One exact Dev37 implementation surface."""

    role: Literal[
        "feature_ranking",
        "source_authority",
        "hierarchy_derivation",
        "refit",
        "sampler",
        "support_auditor",
        "monitor",
        "source_access_auditor",
        "materialization_verifier",
        "refit_replay_verifier",
        "publication_verifier",
        "restart_verifier",
        "extension_verifier",
        "execution_verifier",
        "decision_verifier",
    ]
    artifact: ArtifactRef


class G00CImplementationAuthorityV3(StrictModel):
    """Dev37 release and complete verifier implementation authority."""

    schema_version: Literal[3] = 3
    dev37_code_commit: GitCommit
    wheel: ArtifactRef
    normalized_sdist: ArtifactRef
    implementation_tree_sha256: Sha256
    environment_lock: ArtifactRef
    environment_kind: Literal["exact_local_lock", "oci_container"]
    execution_environment_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    implementations: tuple[G00CImplementationBindingV3, ...]

    @model_validator(mode="after")
    def validate_implementations(self) -> G00CImplementationAuthorityV3:
        expected = (
            "feature_ranking",
            "source_authority",
            "hierarchy_derivation",
            "refit",
            "sampler",
            "support_auditor",
            "monitor",
            "source_access_auditor",
            "materialization_verifier",
            "refit_replay_verifier",
            "publication_verifier",
            "restart_verifier",
            "extension_verifier",
            "execution_verifier",
            "decision_verifier",
        )
        observed = tuple(item.role for item in self.implementations)
        if observed != expected:
            raise ValueError("Dev37 implementation roles must equal the frozen ordered set.")
        return self


class G00CD1ExecutionAuthorityFreezeV3(StrictModel):
    """Dev37 expression-free D1 authority with source and sampler closure."""

    schema_version: Literal[3] = 3
    authority_id: Sha256
    selection_freeze: ArtifactRef
    selection_freeze_id: Sha256
    outer_split_id: Literal["lodo-D1"] = "lodo-D1"
    row_role_freeze: ArtifactRef
    row_roles: tuple[FoldRowRoleRecord, ...]
    nested_training_row_order: ArtifactRef
    nested_training_row_order_hash: Sha256
    feature_reference_rows: ArtifactRef
    feature_reference_rows_hash: Sha256
    feature_reference_rows_are_first_million: Literal[True] = True
    sampler_row_hierarchy: ArtifactRef
    sampler_hierarchy_rows: int = Field(gt=0)
    hierarchy_derivation_receipt: ArtifactRef
    source_plane: G00CSourcePlaneBindingV4
    seed_schedule: ArtifactRef
    seed_schedule_id: Sha256
    base_sampler_plan: ArtifactRef
    base_sampler_plan_id: Sha256
    base_sampler_expected_trace_rows: int = Field(gt=0)
    common_support_prior: G00CCommonSupportPriorV2
    base_support_audit_contract: ArtifactRef
    implementation: G00CImplementationAuthorityV3
    monitored_process_tree_ceiling_bytes: Literal[68719476736] = 68_719_476_736
    fresh_attempt_id: str = Field(min_length=1)
    publication_root_uri: str = Field(min_length=1)
    authority_archive_required: Literal[True] = True
    prior_attempt_artifact_reuse_permitted: Literal[False] = False
    expression_values_accessed_during_freeze: Literal[False] = False
    protected_heldout_expression_values_accessed_during_freeze: Literal[False] = False
    status: Literal["finalized_preaccess_not_executed"] = "finalized_preaccess_not_executed"
    expression_access_authorized_by_this_record: Literal[False] = False
    biological_claims: Literal[False] = False

    _safe_publication_uri = field_validator("publication_root_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @model_validator(mode="after")
    def validate_authority(self) -> G00CD1ExecutionAuthorityFreezeV3:
        expected_roles = (
            "training_fit",
            "training_validation",
            "heldout_source_query",
            "protected_heldout_stimulated",
        )
        if tuple(item.role for item in self.row_roles) != expected_roles:
            raise ValueError("Dev37 D1 authority requires the four ordered row roles.")
        training_rows = next(item.rows for item in self.row_roles if item.role == "training_fit")
        if training_rows != self.sampler_hierarchy_rows:
            raise ValueError("Dev37 hierarchy must contain every training-fit row.")
        expected = self.identity(id_field="authority_id")
        if self.authority_id != expected:
            raise ValueError(f"authority_id mismatch: expected {expected}.")
        return self


class G00CFeatureRankingReceiptV4(StrictModel):
    """Source-derived checkpoint-conditioned Poisson-deviance feature ranking."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    execution_authority_id: Sha256
    source_binding_id: Sha256
    fit_rows_hash: Sha256
    fit_row_count: int = Field(gt=0)
    checkpoint_ids: tuple[Literal["Rest", "Stim8hr", "Stim48hr"], ...] = (
        "Rest",
        "Stim8hr",
        "Stim48hr",
    )
    complete_ranking: ArtifactRef
    ordered_feature_hash: Sha256
    selected_prefix_hash: Sha256
    selected_prefix_count: Literal[4096] = 4096
    canonical_feature_index_hash: Sha256
    puro_r_canonical_index: int = Field(ge=0)
    method: Literal["checkpoint_conditioned_poisson_deviance_v1"] = (
        "checkpoint_conditioned_poisson_deviance_v1"
    )
    scores_recomputed_from_source: Literal[True] = True
    exact_id_index_mapping_verified: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_ranking(self) -> G00CFeatureRankingReceiptV4:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSourceAccessLedgerReceiptV4(StrictModel):
    """Role-labelled ledger from which protected-expression access is derived."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    execution_authority_id: Sha256
    source_binding_id: Sha256
    ledger: ArtifactRef
    access_rows: int = Field(ge=0)
    role_row_hashes: dict[str, Sha256]
    protected_row_ids_sha256: Sha256
    protected_expression_reads: Literal[0] = 0
    zero_protected_reads_derived_from_full_ledger: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_ledger(self) -> G00CSourceAccessLedgerReceiptV4:
        if not self.role_row_hashes:
            raise ValueError("Dev37 source-access ledger must bind at least one role.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CProcessTreeMonitorReceiptV4(StrictModel):
    """Out-of-process process-tree RSS and temporal-coverage evidence."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    execution_authority_id: Sha256
    attempt_id: str = Field(min_length=1)
    monitor_pid: int = Field(gt=0)
    monitored_root_pid: int = Field(gt=0)
    trace: ArtifactRef
    samples: int = Field(gt=1)
    maximum_process_tree_rss_bytes: int = Field(ge=0)
    unreadable_samples: int = Field(ge=0)
    maximum_consecutive_unreadable_samples: int = Field(ge=0)
    maximum_temporal_gap_milliseconds: float = Field(ge=0)
    descendants_observed: int = Field(ge=0)
    access_start_monotonic_ns: int = Field(ge=0)
    access_end_monotonic_ns: int = Field(ge=0)
    monitor_start_monotonic_ns: int = Field(ge=0)
    monitor_end_monotonic_ns: int = Field(ge=0)
    monitor_ran_out_of_process: Literal[True] = True
    process_tree_coverage_verified: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_monitor(self) -> G00CProcessTreeMonitorReceiptV4:
        if self.monitor_pid == self.monitored_root_pid:
            raise ValueError("Dev37 monitor must execute outside the monitored process.")
        if not (
            self.monitor_start_monotonic_ns
            <= self.access_start_monotonic_ns
            <= self.access_end_monotonic_ns
            <= self.monitor_end_monotonic_ns
        ):
            raise ValueError("Dev37 monitor does not cover the complete access interval.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CDurableRestartReceiptV4(StrictModel):
    """Fresh-process restart evidence with distinct attempt paths and equal bytes."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    component: Literal["sampler", "writer"]
    uninterrupted_attempt_id: str = Field(min_length=1)
    resumed_attempt_id: str = Field(min_length=1)
    durable_checkpoint: ArtifactRef
    interrupted_no_final_publication_receipt: ArtifactRef
    uninterrupted_outputs: tuple[ArtifactRef, ...]
    resumed_outputs: tuple[ArtifactRef, ...]
    uninterrupted_process_receipt: ArtifactRef
    resumed_process_receipt: ArtifactRef
    fresh_process_restart: Literal[True] = True
    interrupted_attempt_published_final_payload: Literal[False] = False
    output_hash_and_size_equality: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_restart(self) -> G00CDurableRestartReceiptV4:
        if self.uninterrupted_attempt_id == self.resumed_attempt_id:
            raise ValueError("Dev37 restart attempts must have distinct identities.")
        if not self.uninterrupted_outputs or len(self.uninterrupted_outputs) != len(
            self.resumed_outputs
        ):
            raise ValueError("Dev37 restart outputs must be nonempty and paired.")
        for uninterrupted, resumed in zip(
            self.uninterrupted_outputs, self.resumed_outputs, strict=True
        ):
            if uninterrupted.relative_uri == resumed.relative_uri:
                raise ValueError("Dev37 restart outputs must use distinct paths.")
            if (uninterrupted.sha256, uninterrupted.size_bytes) != (
                resumed.sha256,
                resumed.size_bytes,
            ):
                raise ValueError("Dev37 restart output bytes differ.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSamplerEvidenceV4(StrictModel):
    """Dev37 trace evidence tied to the pre-access plan and fresh-process restart."""

    schema_version: Literal[4] = 4
    evidence_id: Sha256
    execution_authority_id: Sha256
    sampler_plan: ArtifactRef
    sampler_plan_id: Sha256
    row_hierarchy: ArtifactRef
    nested_training_row_order: ArtifactRef
    uninterrupted_draw_trace: ArtifactRef
    resumed_draw_trace: ArtifactRef
    uninterrupted_state_trace: ArtifactRef
    resumed_state_trace: ArtifactRef
    restart_receipt: ArtifactRef
    restart_receipt_id: Sha256
    implementation_sha256: Sha256
    ordered_draws_identical: Literal[True] = True
    rng_states_identical: Literal[True] = True
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_evidence(self) -> G00CSamplerEvidenceV4:
        if (
            self.uninterrupted_draw_trace.relative_uri == self.resumed_draw_trace.relative_uri
            or self.uninterrupted_state_trace.relative_uri == self.resumed_state_trace.relative_uri
        ):
            raise ValueError("Dev37 sampler attempts must publish to distinct paths.")
        expected = self.identity(id_field="evidence_id")
        if self.evidence_id != expected:
            raise ValueError(f"evidence_id mismatch: expected {expected}.")
        return self


class G00CRefitReplayReceiptV4(StrictModel):
    """Checkpoint-indexed sufficient statistics independently derived from source bytes."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    seed_schedule_id: Sha256
    source_binding_id: Sha256
    sampler_evidence_id: Sha256
    source_access_ledger_receipt_id: Sha256
    candidate_kind: Literal["feature_count", "training_cells"]
    selected_candidate: int = Field(gt=0)
    reference_candidate: int = Field(gt=0)
    modeled_feature_count: Literal[256, 512, 1024, 2048, 4096]
    candidate_values: tuple[int, ...]
    checkpoint_ids: tuple[Literal["Rest", "Stim8hr", "Stim48hr"], ...] = (
        "Rest",
        "Stim8hr",
        "Stim48hr",
    )
    refit_records: ArtifactRef
    source_derived_statistics: ArtifactRef
    replayed_rows: ArtifactRef
    fit_row_hashes: tuple[Sha256, ...]
    validation_row_hash: Sha256
    training_count_hash: Sha256
    validation_count_vector_hash: Sha256
    thinning_trace_hash: Sha256
    training_shape: tuple[int, int, int, Literal[4096]]
    validation_shape: tuple[int, Literal[3], Literal[4096]]
    validation_vectors_identical_across_candidates: Literal[True] = True
    feature_statistics_identical_before_prefix: Literal[True] = True
    nested_cell_prefixes_verified: Literal[True] = True
    selected_and_reference_all_59_draws: Literal[True] = True
    preregistered_audit_draw_ids: tuple[int, ...] = (
        0,
        6,
        12,
        18,
        24,
        30,
        36,
        42,
        48,
        58,
    )
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_replay(self) -> G00CRefitReplayReceiptV4:
        if not self.candidate_values or len(self.candidate_values) != self.training_shape[1]:
            raise ValueError("Dev37 candidate axis does not match the training statistics.")
        if self.training_shape[0] != 59 or self.training_shape[2:] != (3, 4096):
            raise ValueError("Dev37 training statistics must be [59,candidate,3,4096].")
        if self.validation_shape != (59, 3, 4096):
            raise ValueError("Dev37 validation statistics must be [59,3,4096].")
        if len(self.fit_row_hashes) != len(self.candidate_values):
            raise ValueError("Dev37 fit-row hashes must cover every candidate.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSampleSizeExtensionFreezeV2(StrictModel):
    """One V4-chain two-million-row extension and no third access."""

    schema_version: Literal[2] = 2
    extension_freeze_id: Sha256
    base_execution_bundle: ArtifactRef
    base_execution_bundle_id: Sha256
    base_extension_required_receipt: ArtifactRef
    base_extension_required_receipt_id: Sha256
    base_final_seal: ArtifactRef
    base_final_seal_id: Sha256
    execution_authority_id: Sha256
    source_binding_id: Sha256
    feature_selection_result_id: Sha256
    selected_feature_count: Literal[256, 512, 1024, 2048, 4096]
    selected_feature_order_sha256: Sha256
    selected_feature_surface: ArtifactRef
    seed_schedule_id: Sha256
    seed_schedule_artifact: ArtifactRef
    base_support_audit_receipt: ArtifactRef
    extension_sampler_plan: ArtifactRef
    extension_sampler_plan_id: Sha256
    extension_support_audit_contract: ArtifactRef
    added_candidate_cells: tuple[Literal[2_000_000], ...] = (2_000_000,)
    third_extension_permitted: Literal[False] = False
    fresh_attempt_id: str = Field(min_length=1)
    publication_root_uri: str = Field(min_length=1)
    expression_values_accessed_during_freeze: Literal[False] = False
    status: Literal["extension_frozen_not_run"] = "extension_frozen_not_run"

    _safe_extension_uri = field_validator("publication_root_uri")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @model_validator(mode="after")
    def validate_extension(self) -> G00CSampleSizeExtensionFreezeV2:
        if self.added_candidate_cells != (2_000_000,):
            raise ValueError("Dev37 permits exactly one two-million-row extension.")
        expected = self.identity(id_field="extension_freeze_id")
        if self.extension_freeze_id != expected:
            raise ValueError(f"extension_freeze_id mismatch: expected {expected}.")
        return self


class G00CMaterializationReceiptV4(StrictModel):
    """Dev37 materialization wrapper with monitor, ledger, and durable writer restart."""

    schema_version: Literal[4] = 4
    receipt_id: Sha256
    execution_authority_id: Sha256
    source_binding_id: Sha256
    base_materialization_receipt: ArtifactRef
    base_materialization_receipt_id: Sha256
    source_access_ledger_receipt: ArtifactRef
    source_access_ledger_receipt_id: Sha256
    monitor_receipt: ArtifactRef
    monitor_receipt_id: Sha256
    writer_restart_receipt: ArtifactRef
    writer_restart_receipt_id: Sha256
    status: Literal["pass"] = "pass"

    @model_validator(mode="after")
    def validate_materialization(self) -> G00CMaterializationReceiptV4:
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CSemanticPublicationArtifactV4(StrictModel):
    """One semantic role mapped to the exact bundle ArtifactRef and published filename."""

    role: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    source_artifact: ArtifactRef
    published_sha256: Sha256
    published_size_bytes: int = Field(ge=0)

    _safe_filename = field_validator("filename")(
        classmethod(lambda cls, value: validate_relative_uri(value))
    )

    @model_validator(mode="after")
    def validate_mapping(self) -> G00CSemanticPublicationArtifactV4:
        if (self.published_sha256, self.published_size_bytes) != (
            self.source_artifact.sha256,
            self.source_artifact.size_bytes,
        ):
            raise ValueError("Dev37 published bytes must equal the bound source artifact.")
        return self


class G00CInnerPublicationInventoryV4(StrictModel):
    """Decision-free semantic inventory written as inner artifacts.json."""

    schema_version: Literal[4] = 4
    inventory_id: Sha256
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]
    artifacts: tuple[G00CSemanticPublicationArtifactV4, ...]
    final_decision_included: Literal[False] = False

    @model_validator(mode="after")
    def validate_inventory(self) -> G00CInnerPublicationInventoryV4:
        roles = tuple(item.role for item in self.artifacts)
        names = tuple(item.filename for item in self.artifacts)
        if roles != tuple(sorted(roles)) or len(roles) != len(set(roles)):
            raise ValueError("Dev37 semantic roles must be uniquely sorted.")
        if len(names) != len(set(names)) or any(
            item.role == "final_decision" for item in self.artifacts
        ):
            raise ValueError("Dev37 inner publication filenames must be unique and decision-free.")
        expected = self.identity(id_field="inventory_id")
        if self.inventory_id != expected:
            raise ValueError(f"inventory_id mismatch: expected {expected}.")
        return self


class G00CInnerPublicationManifestV4(StrictModel):
    """Cycle-free inner execution publication; the final decision is deliberately absent."""

    schema_version: Literal[4] = 4
    manifest_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    feature_selection_result_id: Sha256
    sample_size_selection_result_id: Sha256
    sampler_evidence_id: Sha256
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]
    artifact_inventory: ArtifactRef
    artifact_inventory_id: Sha256
    sha256sums: ArtifactRef
    committed: ArtifactRef
    publication_event_receipt: ArtifactRef
    publisher_implementation_sha256: Sha256
    manifest_written_last: Literal[True] = True
    destination_preexisted: Literal[False] = False
    no_clobber: Literal[True] = True

    @model_validator(mode="after")
    def validate_manifest(self) -> G00CInnerPublicationManifestV4:
        expected = self.identity(id_field="manifest_id")
        if self.manifest_id != expected:
            raise ValueError(f"manifest_id mismatch: expected {expected}.")
        return self


class G00CExecutionBundleV5(StrictModel):
    """Dev37 source-derived execution bundle, parent of a later outer decision seal."""

    schema_version: Literal[5] = 5
    bundle_id: Sha256
    execution_authority: ArtifactRef
    execution_authority_id: Sha256
    selection_freeze: ArtifactRef
    selection_freeze_id: Sha256
    seed_schedule: ArtifactRef
    seed_schedule_id: Sha256
    row_roles: ArtifactRef
    feature_ranking_receipt: ArtifactRef
    feature_selection_result: ArtifactRef
    feature_selection_result_id: Sha256
    sample_size_selection_result: ArtifactRef
    sample_size_selection_result_id: Sha256
    common_support_receipt: ArtifactRef
    feature_refit_replay_receipt: ArtifactRef
    sample_refit_replay_receipt: ArtifactRef
    sampler_evidence: ArtifactRef
    sampler_restart_receipt: ArtifactRef
    source_access_ledger_receipt: ArtifactRef
    monitor_receipt: ArtifactRef
    base_support_audit_contract: ArtifactRef
    base_support_audit_receipt: ArtifactRef
    extension_freeze: ArtifactRef | None = None
    extension_sampler_evidence: ArtifactRef | None = None
    extension_sampler_restart_receipt: ArtifactRef | None = None
    extension_support_audit_contract: ArtifactRef | None = None
    extension_support_audit_receipt: ArtifactRef | None = None
    materialization_receipt: ArtifactRef | None = None
    inner_publication_manifest: ArtifactRef
    failure_receipt: ArtifactRef | None = None
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]

    @model_validator(mode="after")
    def validate_bundle(self) -> G00CExecutionBundleV5:
        if self.terminal_status == "pass":
            if self.materialization_receipt is None or self.failure_receipt is not None:
                raise ValueError("A Dev37 pass requires typed materialization and no failure.")
        elif self.materialization_receipt is not None:
            raise ValueError("A nonpass Dev37 bundle cannot bind materialization evidence.")
        if self.terminal_status == "failed_integrity":
            if self.failure_receipt is None:
                raise ValueError("A Dev37 integrity failure requires its receipt.")
        elif self.failure_receipt is not None:
            raise ValueError("Only failed_integrity may bind a failure receipt.")
        extension = (
            self.extension_freeze,
            self.extension_sampler_evidence,
            self.extension_sampler_restart_receipt,
            self.extension_support_audit_contract,
            self.extension_support_audit_receipt,
        )
        if self.terminal_status == "fail_no_saturation" and any(item is None for item in extension):
            raise ValueError("Dev37 nonsaturation requires its complete V4 extension chain.")
        if self.extension_freeze is None and any(item is not None for item in extension[1:]):
            raise ValueError("Dev37 extension evidence requires its V4 extension freeze.")
        expected = self.identity(id_field="bundle_id")
        if self.bundle_id != expected:
            raise ValueError(f"bundle_id mismatch: expected {expected}.")
        return self


class G00CDecisionReceiptV5(StrictModel):
    """Dev37 terminal decision created after the immutable inner publication."""

    schema_version: Literal[5] = 5
    receipt_id: Sha256
    execution_bundle_id: Sha256
    execution_authority_id: Sha256
    selection_freeze_id: Sha256
    feature_selection_result_id: Sha256
    sample_size_selection_result_id: Sha256
    feature_ranking_receipt_id: Sha256
    sampler_evidence_id: Sha256
    sampler_restart_receipt_id: Sha256
    feature_refit_replay_receipt_id: Sha256
    sample_refit_replay_receipt_id: Sha256
    support_audit_receipt_ids: tuple[Sha256, ...]
    source_access_ledger_receipt_id: Sha256
    monitor_receipt_id: Sha256
    materialization_receipt_id: Sha256 | None = None
    inner_publication_manifest_id: Sha256
    verified_artifact_sha256s: tuple[Sha256, ...]
    terminal_status: Literal["pass", "extension_required", "fail_no_saturation", "failed_integrity"]
    may_parent_g00d: bool
    verification_completed: Literal[True] = True
    biological_claims: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> G00CDecisionReceiptV5:
        passed = self.terminal_status == "pass"
        if (
            self.may_parent_g00d != passed
            or (self.materialization_receipt_id is not None) != passed
        ):
            raise ValueError("Only a source-derived, sealed Dev37 pass may parent G00D.")
        if not self.verified_artifact_sha256s or len(self.verified_artifact_sha256s) != len(
            set(self.verified_artifact_sha256s)
        ):
            raise ValueError("Dev37 verified artifact hashes must be nonempty and unique.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class G00CFinalSealV1(StrictModel):
    """Outer no-cycle seal binding bundle, inner publication, and final decision."""

    schema_version: Literal[1] = 1
    seal_id: Sha256
    execution_bundle: ArtifactRef
    execution_bundle_id: Sha256
    inner_publication_manifest: ArtifactRef
    inner_publication_manifest_id: Sha256
    final_decision: ArtifactRef
    final_decision_id: Sha256
    outer_artifact_inventory: ArtifactRef
    sha256sums: ArtifactRef
    committed: ArtifactRef
    no_clobber: Literal[True] = True
    manifest_last: Literal[True] = True
    status: Literal["sealed"] = "sealed"

    @model_validator(mode="after")
    def validate_seal(self) -> G00CFinalSealV1:
        expected = self.identity(id_field="seal_id")
        if self.seal_id != expected:
            raise ValueError(f"seal_id mismatch: expected {expected}.")
        return self


class G00DParityGateContract(StrictModel):
    """Frozen comparison semantics for one integrated-loader parity gate."""

    gate: Literal[
        "row_ids",
        "raw_counts",
        "sample_weights",
        "thinning_rng",
        "loss",
        "gradient",
        "parameter",
        "interrupted_resume",
    ]
    comparison_mode: Literal["exact_hash", "exact_or_numerical", "numerical_tolerance"]


class IntegratedLoaderQualificationContractV2(StrictModel):
    """Dev31 G00D benchmark, parity, and bounded-memory protocol."""

    schema_version: Literal[2] = 2
    qualification_contract_id: str
    fold_view_id: str
    expected_gpu_name: str = Field(min_length=1)
    expected_gpu_count: Literal[1] = 1
    expected_cuda_version: str = Field(min_length=1)
    expected_torch_version: str = Field(min_length=1)
    expected_container_digest: Sha256
    worker_count: int = Field(ge=1)
    cpu_count: int = Field(ge=1)
    storage_authority_hash: Sha256
    microbatch_cells: Literal[512] = 512
    microbatches_per_update: Literal[8] = 8
    macrobatch_cells: Literal[4096] = 4096
    prefetch_depth: int = Field(ge=1)
    warmup_updates: int = Field(gt=0)
    measured_updates: int = Field(gt=0)
    require_cold_start_measurement: Literal[True] = True
    require_steady_state_measurement: Literal[True] = True
    cache_policy: str = Field(min_length=1)
    measurement_protocol_sha256: Sha256
    telemetry_interval_seconds: float = Field(gt=0)
    maximum_data_wait_fraction: float = Field(default=0.10, ge=0, le=0.10)
    minimum_steady_state_gpu_utilization: float = Field(default=0.85, ge=0.85, le=1)
    maximum_p95_batch_ready_seconds: float = Field(gt=0)
    maximum_loader_rss_bytes: int = Field(gt=0)
    maximum_process_loader_rss_bytes: int = Field(gt=0)
    maximum_aggregate_worker_rss_bytes: int = Field(gt=0)
    maximum_open_shards: int = Field(ge=1)
    maximum_open_file_handles: int = Field(ge=1)
    maximum_rss_slope_upper_bytes_per_second: float = Field(ge=0)
    maximum_rss_excursion_fraction: float = Field(gt=0, le=1)
    parity_absolute_tolerance: float = Field(default=1e-6, gt=0)
    parity_relative_tolerance: float = Field(default=1e-5, gt=0)
    parity_gates: tuple[G00DParityGateContract, ...]

    @model_validator(mode="after")
    def validate_loader_contract(self) -> IntegratedLoaderQualificationContractV2:
        modes = {gate.gate: gate.comparison_mode for gate in self.parity_gates}
        expected_modes = {
            "row_ids": "exact_hash",
            "raw_counts": "exact_hash",
            "sample_weights": "exact_or_numerical",
            "thinning_rng": "exact_hash",
            "loss": "numerical_tolerance",
            "gradient": "numerical_tolerance",
            "parameter": "numerical_tolerance",
            "interrupted_resume": "exact_hash",
        }
        if len(self.parity_gates) != len(expected_modes) or modes != expected_modes:
            raise ValueError("G00D parity comparison modes must equal the frozen gate policy.")
        expected = self.identity(id_field="qualification_contract_id")
        if self.qualification_contract_id != expected:
            raise ValueError(f"qualification_contract_id mismatch: expected {expected}.")
        return self


class G00DParityGateEvidence(StrictModel):
    """Per-gate exact or numerical parity evidence."""

    gate: Literal[
        "row_ids",
        "raw_counts",
        "sample_weights",
        "thinning_rng",
        "loss",
        "gradient",
        "parameter",
        "interrupted_resume",
    ]
    comparison_mode: Literal["exact_hash", "exact_or_numerical", "numerical_tolerance"]
    reference_sha256: Sha256
    observed_sha256: Sha256
    maximum_absolute_error: float = Field(ge=0)
    maximum_relative_error: float = Field(ge=0)


class IntegratedLoaderQualificationReceiptV2(StrictModel):
    """Dev31 observed G00D performance, parity, and memory evidence."""

    schema_version: Literal[2] = 2
    receipt_id: str
    qualification_contract_id: str
    gpu_name: str = Field(min_length=1)
    gpu_uuid: str = Field(min_length=1)
    gpu_count: int = Field(ge=1)
    cuda_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    container_digest: Sha256
    worker_count: int = Field(ge=1)
    cpu_count: int = Field(ge=1)
    storage_authority_hash: Sha256
    microbatch_cells: int = Field(gt=0)
    microbatches_per_update: int = Field(gt=0)
    macrobatch_cells: int = Field(gt=0)
    prefetch_depth: int = Field(ge=1)
    warmup_updates: int = Field(gt=0)
    measured_updates: int = Field(gt=0)
    cold_start_measured: bool
    steady_state_measured: bool
    cache_policy: str = Field(min_length=1)
    measurement_protocol_sha256: Sha256
    measurement_evidence: ArtifactRef
    telemetry_artifact: ArtifactRef
    parity_artifact: ArtifactRef
    memory_trace_artifact: ArtifactRef
    telemetry_interval_seconds: float = Field(gt=0)
    median_compute_seconds: float = Field(ge=0)
    p95_compute_seconds: float = Field(ge=0)
    median_data_wait_seconds: float = Field(ge=0)
    p95_batch_ready_seconds: float = Field(ge=0)
    data_wait_fraction: float = Field(ge=0, le=1)
    steady_state_gpu_utilization: float = Field(ge=0, le=1)
    peak_loader_rss_bytes: int = Field(ge=0)
    peak_process_loader_rss_bytes: int = Field(ge=0)
    peak_aggregate_worker_rss_bytes: int = Field(ge=0)
    peak_open_shards: int = Field(ge=0)
    peak_open_file_handles: int = Field(ge=0)
    rss_slope_bytes_per_second: float
    rss_slope_upper_ci_bytes_per_second: float
    maximum_rss_excursion_bytes: int = Field(ge=0)
    parity_gates: tuple[G00DParityGateEvidence, ...]
    lru_bound_pass: bool
    loader_error_count: int = Field(ge=0)
    cuda_error_count: int = Field(ge=0)
    monitor_error_count: int = Field(ge=0)
    maximum_parity_absolute_error: float = Field(ge=0)
    maximum_parity_relative_error: float = Field(ge=0)
    memory_growth_pass: bool
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_loader_receipt(self) -> IntegratedLoaderQualificationReceiptV2:
        gate_names = [gate.gate for gate in self.parity_gates]
        expected_gates = {
            "row_ids",
            "raw_counts",
            "sample_weights",
            "thinning_rng",
            "loss",
            "gradient",
            "parameter",
            "interrupted_resume",
        }
        if len(gate_names) != len(expected_gates) or set(gate_names) != expected_gates:
            raise ValueError("G00D receipt must contain every parity gate exactly once.")
        observed_absolute = max(gate.maximum_absolute_error for gate in self.parity_gates)
        observed_relative = max(gate.maximum_relative_error for gate in self.parity_gates)
        if (
            self.maximum_parity_absolute_error != observed_absolute
            or self.maximum_parity_relative_error != observed_relative
        ):
            raise ValueError("Aggregate parity errors must be derived from per-gate evidence.")
        expected = self.identity(id_field="receipt_id")
        if self.receipt_id != expected:
            raise ValueError(f"receipt_id mismatch: expected {expected}.")
        return self


class PreparedRepresentation(StrictModel):
    schema_version: int = 1
    prepared_id: str
    cache_generation_id: str
    count_store: ArtifactRef
    information_set: ArtifactRef
    fit_selection: ArtifactRef
    input_view: ArtifactRef
    feature_index: ArtifactRef
    encoder_state: ArtifactRef
    decoder_state: ArtifactRef
    latent_cache: ArtifactRef
    fit_rows_hash: Sha256
    validation_rows_hash: Sha256
    state_dim: int = Field(gt=0)


class ResolvedRunCapabilities(StrictModel):
    predict_state: bool = True
    # The development SVD representation has no calibrated count decoder.
    # Recipes must opt in only after a reconstruction/likelihood gate passes.
    decode_gene_composition: bool = False
    predict_relative_mass: bool = False
    resume_training: bool = True
    target_reference_counterfactual: bool = True
    fixed_context_counterfactual: bool = False
    dynamic_context_counterfactual: bool = False
    stream_terminal_particles: bool = True

    @classmethod
    def for_intent(cls, intent: RunIntent) -> ResolvedRunCapabilities:
        measure = intent in {RunIntent.COUNT_MEASURE, RunIntent.COUNT_CONTEXT}
        context = intent is RunIntent.COUNT_CONTEXT
        return cls(
            predict_relative_mass=measure,
            fixed_context_counterfactual=context,
            dynamic_context_counterfactual=context,
        )


class ImplementationCapabilities(StrictModel):
    schema_version: int = 1
    count_state: Literal[True] = True
    count_measure: Literal[True] = True
    count_context: Literal[True] = True
    sparse_count_store: Literal[True] = True
    exact_complete_block_dm: Literal[True] = True
    typed_resume: Literal[True] = True
    canonical_loader: Literal["credo-v4 open-run"] = "credo-v4 open-run"


class ModelConfig(StrictModel):
    state_dim: int = Field(default=8, gt=0)
    target_count: int = Field(gt=0)
    pool_count: int = Field(default=1, gt=0)
    hidden_dim: int = Field(default=32, gt=0)
    gene_decoder_features: int = Field(default=0, ge=0)
    gene_decoder_hidden_dim: int = Field(default=0, ge=0)
    state_dependent_drift: bool = False
    freeze_closed_form_drift: bool = False
    terminal_anchor_drift: bool = False
    source_carryover_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    source_conditioned_anchor: bool = False
    source_anchor_residual_scale: float = Field(default=0.25, gt=0.0)
    source_target_interaction_rank: int = Field(default=0, ge=0, le=8)
    source_target_interaction_scale: float = Field(default=0.25, gt=0.0)
    source_target_whitening_ridge: float = Field(default=1e-3, gt=0.0)
    source_target_main_max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    trainable_terminal_anchor: bool = True
    trainable_target_anchor: bool = True
    target_anchor_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptive_target_anchor: bool = False
    adaptive_target_anchor_max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    adaptive_target_anchor_min_guide_gain: float = Field(default=0.0, ge=0.0)
    shared_diffusion: bool = False
    shared_diffusion_inner_validation_pass: bool = False
    centered_selection: bool = False
    selection_inner_validation_pass: bool = False
    source_efficacy_sensitivity: bool = False
    context_rank: int = Field(default=2, ge=1, le=4)

    @model_validator(mode="after")
    def ablation_gates(self) -> ModelConfig:
        if self.gene_decoder_hidden_dim and not self.gene_decoder_features:
            raise ValueError("A hidden gene decoder requires gene_decoder_features.")
        if self.terminal_anchor_drift and self.state_dependent_drift:
            raise ValueError("Terminal-anchor and neural state drift are mutually exclusive.")
        if not self.terminal_anchor_drift and (
            self.source_carryover_alpha != 1.0
            or self.source_conditioned_anchor
            or not self.trainable_terminal_anchor
            or not self.trainable_target_anchor
            or self.target_anchor_weight != 0.0
        ):
            raise ValueError("Anchor shrinkage parameters require terminal_anchor_drift.")
        if self.source_conditioned_anchor and self.source_carryover_alpha != 0.0:
            raise ValueError(
                "A source-conditioned terminal anchor requires zero recurrent carryover."
            )
        if self.source_target_interaction_rank and (
            not self.terminal_anchor_drift or self.source_carryover_alpha != 0.0
        ):
            raise ValueError(
                "A source-target interaction requires a zero-carryover terminal anchor."
            )
        if self.source_target_interaction_rank and self.source_conditioned_anchor:
            raise ValueError(
                "Source-only and source-target anchor residuals are mutually exclusive."
            )
        if self.adaptive_target_anchor and (
            not self.terminal_anchor_drift
            or self.source_carryover_alpha != 0.0
            or self.target_anchor_weight != 0.0
        ):
            raise ValueError(
                "Adaptive target anchors require a zero-carryover terminal anchor "
                "with no fixed target weight."
            )
        if self.freeze_closed_form_drift and not self.state_dependent_drift:
            raise ValueError("A frozen closed-form drift requires a state-dependent residual.")
        if self.shared_diffusion and not self.shared_diffusion_inner_validation_pass:
            raise ValueError("Shared diffusion requires a passing inner-validation ablation.")
        if self.centered_selection and not self.selection_inner_validation_pass:
            raise ValueError("State selection requires a passing inner-validation ablation.")
        return self


class TrainingConfig(StrictModel):
    max_updates: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=1e-2, gt=0)
    state_batch_size: int = Field(default=16, gt=0)
    checkpoint_every: int = Field(default=25, gt=0)
    seed: int = Field(default=0, ge=0)
    initialization_seed: int = Field(default=0, ge=0)
    state_split_seed: int = Field(default=0, ge=0)
    dtype: Literal["float32", "float64"] = "float32"
    deterministic: bool = True
    state_full_batch: bool = False
    optimizer_name: Literal["AdamW"] = "AdamW"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = Field(default=1e-8, gt=0.0)
    optimizer_weight_decay: float = Field(default=0.01, ge=0.0)
    pilot_device_type: Literal["cpu", "cuda"] | None = None
    selected_update: int | None = None
    analytic_fit: bool = False
    gene_decoder_batch_size: int = Field(default=0, ge=0)
    gene_decoder_loss_weight: float = Field(default=0.0, ge=0.0)
    gene_decoder_validation_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    gene_decoder_validation_max_rows: int = Field(default=8_192, gt=0)
    train_state_with_gene_decoder: bool = False
    state_validation_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    state_validation_max_per_target: int = Field(default=1, gt=0)
    support_weight_power: float = Field(default=0.0, ge=0.0, le=1.0)
    support_weight_cap: int = Field(default=1_000, gt=0)
    target_drift_penalty: float = Field(default=0.0, ge=0.0)
    source_drift_penalty: float = Field(default=0.0, ge=0.0)
    source_target_main_penalty: float = Field(default=0.0, ge=0.0)
    source_target_interaction_penalty: float = Field(default=0.0, ge=0.0)
    noninteraction_linear_ridge: float = Field(default=1.0, gt=0.0)
    state_checkpoint_updates: tuple[int, ...] = ()
    state_validation_target_minimum_improvement: float = Field(default=0.0, ge=0.0)
    state_validation_interaction_minimum_improvement: float = Field(default=0.0, ge=0.0)
    post_selection_state_refit: bool = False
    checkpoint_selection: Literal[
        "final",
        "minimum_gene_decoder_validation",
        "minimum_state_validation",
        "minimum_state_validation_null_guarded",
    ] = "final"

    @model_validator(mode="after")
    def selected_in_budget(self) -> TrainingConfig:
        beta1, beta2 = self.optimizer_betas
        if not (0.0 <= beta1 < beta2 < 1.0):
            raise ValueError("optimizer_betas must satisfy 0 <= beta1 < beta2 < 1.")
        if (self.gene_decoder_batch_size == 0) != (self.gene_decoder_loss_weight == 0.0):
            raise ValueError("Gene decoder batch size and loss weight must be enabled together.")
        if self.gene_decoder_validation_fraction and not self.gene_decoder_batch_size:
            raise ValueError("Gene decoder validation requires decoder training.")
        if self.train_state_with_gene_decoder and not self.gene_decoder_batch_size:
            raise ValueError("Joint state/decoder training requires an enabled gene decoder.")
        if self.checkpoint_selection == "minimum_gene_decoder_validation":
            if not self.gene_decoder_validation_fraction:
                raise ValueError("Validation checkpoint selection requires a validation split.")
            if self.selected_update is not None:
                raise ValueError(
                    "Validation checkpoint selection resolves selected_update after training."
                )
        if self.checkpoint_selection in {
            "minimum_state_validation",
            "minimum_state_validation_null_guarded",
        }:
            if not self.state_validation_fraction:
                raise ValueError("State validation selection requires a validation split.")
            if self.selected_update is not None:
                raise ValueError(
                    "State-validation checkpoint selection resolves selected_update after training."
                )
        if (
            self.checkpoint_selection == "minimum_state_validation_null_guarded"
            and not self.post_selection_state_refit
        ):
            raise ValueError("Null-guarded state selection requires post-selection refitting.")
        if tuple(sorted(set(self.state_checkpoint_updates))) != self.state_checkpoint_updates:
            raise ValueError("state_checkpoint_updates must be strictly increasing and unique.")
        if any(
            update <= 0 or update > self.max_updates for update in self.state_checkpoint_updates
        ):
            raise ValueError("state_checkpoint_updates must lie within 1..max_updates.")
        if self.analytic_fit and self.gene_decoder_batch_size:
            raise ValueError("Analytic fitting cannot train a gene decoder.")
        if self.analytic_fit and (self.max_updates != 1 or self.selected_update not in {None, 1}):
            raise ValueError("Analytic fitting uses exactly one selected checkpoint generation.")
        if self.selected_update is not None and self.selected_update > self.max_updates:
            raise ValueError("selected_update exceeds max_updates.")
        if (
            self.selected_update is not None
            and self.selected_update != self.max_updates
            and self.selected_update % self.checkpoint_every != 0
        ):
            raise ValueError("selected_update must be a persisted checkpoint generation.")
        return self


class EvaluationConfig(StrictModel):
    particles: int = Field(default=128, gt=0)
    steps: int = Field(default=8, gt=0)
    seed: int = Field(default=10_000, ge=0)
    output_bytes_limit: int = Field(default=1_000_000_000, gt=0)
    max_step_duration: float = Field(default=1.0, gt=0)


class PhysicalGrid(StrictModel):
    schema_version: int = 1
    axis_unit: str = Field(min_length=1)
    duration: float = Field(gt=0)
    maximum_step: float = Field(gt=0)
    steps: int = Field(gt=0)
    step_size: float = Field(gt=0)

    @model_validator(mode="after")
    def exact_grid(self) -> PhysicalGrid:
        expected_steps = math.ceil(self.duration / self.maximum_step)
        expected_size = self.duration / expected_steps
        if self.steps != expected_steps or not math.isclose(
            self.step_size, expected_size, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("Physical grid does not match ceil(duration / maximum_step).")
        return self

    @classmethod
    def resolve(cls, *, axis_unit: str, duration: float, maximum_step: float) -> PhysicalGrid:
        steps = math.ceil(duration / maximum_step)
        return cls(
            axis_unit=axis_unit,
            duration=duration,
            maximum_step=maximum_step,
            steps=steps,
            step_size=duration / steps,
        )


class ResolvedConfig(StrictModel):
    schema_version: int = 1
    workspace: str
    semantic_snapshot: str
    count_store: str
    information_set: str
    split_contract: str
    eligibility_manifest: str
    effect_hierarchy: str
    topology_contract: str
    denominator_contract: str | None = None
    pool_contract: str | None = None
    preregistration: str
    multiplicity_plan: str
    candidate_selection_plan: str
    state_selection_calibration: str | None = None
    pooled_estimand: Literal["pooled_known_target_heldout_guide"] | None = None
    outer_fold_id: str | None = None
    inner_split_id: str | None = None
    pooled_outer_fold_ids: tuple[str, ...] = ()
    pooled_inner_split_ids: tuple[str, ...] = ()
    pooled_optimization_seeds: tuple[int, ...] = ()
    baseline_registry: str
    intent: RunIntent
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    feature_index: str
    source_manifest: str
    feature_permutation: str
    input_view: Literal["identity_library_normalized", "matched_control_offset_v1"] = (
        "identity_library_normalized"
    )
    correction_metadata: str | None = None
    source_smoothing: float = 0.5
    representation_fit_max_rows: int = Field(default=250_000, gt=0)
    representation_encode_batch_size: int = Field(default=8_192, gt=0)
    representation_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def correction_dependency(self) -> ResolvedConfig:
        if self.source_smoothing != 0.5:
            raise ValueError("v4.0 source smoothing is frozen at 0.5.")
        if self.input_view == "matched_control_offset_v1" and not self.correction_metadata:
            raise ValueError("matched_control_offset_v1 requires correction_metadata.")
        if (
            self.intent in {RunIntent.COUNT_MEASURE, RunIntent.COUNT_CONTEXT}
            and not self.denominator_contract
        ):
            raise ValueError(f"{self.intent.value} requires a complete denominator contract.")
        if self.intent is RunIntent.COUNT_CONTEXT and not self.pool_contract:
            raise ValueError("count_context requires a physical pool contract.")
        if self.intent is RunIntent.COUNT_STATE and (
            self.denominator_contract is not None or self.pool_contract is not None
        ):
            raise ValueError("count_state forbids denominator and pool contracts.")
        if self.model.terminal_anchor_drift and self.intent is not RunIntent.COUNT_STATE:
            raise ValueError("Development terminal-anchor drift is count_state-only.")
        if self.model.source_target_interaction_rank:
            if self.intent is not RunIntent.COUNT_STATE:
                raise ValueError("Source-target state interactions are count_state-only.")
            if (
                self.model.trainable_terminal_anchor
                or self.model.trainable_target_anchor
                or self.model.target_anchor_weight != 0.0
                or self.model.adaptive_target_anchor
            ):
                raise ValueError("Source-target pilots require frozen null and target anchors.")
            if self.training.source_target_main_penalty <= 0.0:
                raise ValueError("Source-target pilots require positive target-main shrinkage.")
            if self.training.source_target_interaction_penalty <= 0.0:
                raise ValueError(
                    "Source-target interactions require positive interaction shrinkage."
                )
            if self.training.state_validation_target_minimum_improvement <= 0.0:
                raise ValueError("Source-target pilots require a positive target-only margin.")
            if self.training.state_validation_interaction_minimum_improvement <= 0.0:
                raise ValueError("Source-target pilots require a positive interaction margin.")
            if self.training.checkpoint_selection != "minimum_state_validation_null_guarded":
                raise ValueError("Source-target interactions require null-guarded selection.")
            if not self.training.state_checkpoint_updates:
                raise ValueError(
                    "Source-target pilots require an explicit early checkpoint schedule."
                )
            if self.training.state_checkpoint_updates[0] > 25:
                raise ValueError("Source-target pilot checkpointing must begin by update 25.")
            if self.training.max_updates > 500:
                raise ValueError("The first pooled source-target pilot is capped at 500 updates.")
            if self.state_selection_calibration is None:
                raise ValueError("Source-target pilots require a bound selection calibration.")
            if self.pooled_estimand != "pooled_known_target_heldout_guide":
                raise ValueError(
                    "Source-target pilots require the pooled known-target held-out-guide estimand."
                )
            if not self.outer_fold_id or not self.inner_split_id:
                raise ValueError(
                    "Source-target pilots require fixed outer-fold and inner-split IDs."
                )
            if (
                len(set(self.pooled_outer_fold_ids)) < 2
                or self.outer_fold_id not in self.pooled_outer_fold_ids
            ):
                raise ValueError("Pooled pilots require at least two frozen outer guide folds.")
            if (
                len(self.pooled_inner_split_ids) != len(self.pooled_outer_fold_ids)
                or self.inner_split_id not in self.pooled_inner_split_ids
            ):
                raise ValueError("Pooled pilots require one frozen inner split per outer fold.")
            if (
                len(set(self.pooled_optimization_seeds)) < 3
                or self.training.seed not in self.pooled_optimization_seeds
            ):
                raise ValueError("Pooled pilots require a frozen plan of at least three seeds.")
            if self.model.pool_count != 1:
                raise ValueError(
                    "Pooled source-target pilots require exactly one model-facing pool."
                )
            if not self.training.state_full_batch:
                raise ValueError(
                    "Pooled source-target pilots require full-batch state optimization."
                )
            if self.training.pilot_device_type is None:
                raise ValueError("Source-target pilots require a frozen device type.")
            forbidden_channels = {
                "shared_diffusion": self.model.shared_diffusion,
                "centered_selection": self.model.centered_selection,
                "state_dependent_drift": self.model.state_dependent_drift,
                "source_conditioned_anchor": self.model.source_conditioned_anchor,
                "analytic_fit": self.training.analytic_fit,
            }
            enabled = sorted(name for name, value in forbidden_channels.items() if value)
            if enabled:
                raise ValueError(
                    "Source-target pilots forbid uncalibrated channels: " + ", ".join(enabled)
                )
            if self.training.support_weight_power != 0.0:
                raise ValueError(
                    "Source-target pilots freeze support_weight_power=0 until weighted "
                    "calibration is implemented."
                )
            if any(
                (
                    self.model.gene_decoder_features,
                    self.model.gene_decoder_hidden_dim,
                    self.training.gene_decoder_batch_size,
                    self.training.gene_decoder_loss_weight,
                    self.training.gene_decoder_validation_fraction,
                )
            ):
                raise ValueError(
                    "Source-target state pilots must disable the gene decoder entirely."
                )
        return self


class CompiledRunContract(StrictModel):
    schema_version: int = 1
    compiled_run_id: str
    recipe_id: Literal["credo.count_sde_v4"] = "credo.count_sde_v4"
    recipe_version: Literal["4.0.dev36", "4.0.dev37"] = "4.0.dev37"
    recipe_wheel_hash: Sha256
    frozen_credo_artifact_hash: Sha256
    environment_lock_hash: Sha256
    source_manifest_hash: Sha256
    count_store_merkle_root: Sha256
    row_universe_hash: Sha256
    feature_index_hash: Sha256
    feature_permutation_hash: Sha256
    exposure_registry_hash: Sha256
    split_manifest_hash: Sha256
    information_set_hash: Sha256
    eligibility_manifest_hash: Sha256
    target_hierarchy_hash: Sha256
    denominator_manifest_hash: Sha256 | None
    experimental_topology_hash: Sha256
    physical_pool_manifest_hash: Sha256 | None
    preregistration_hash: Sha256
    multiplicity_plan_hash: Sha256
    candidate_selection_plan_hash: Sha256
    state_selection_calibration_hash: Sha256 | None
    state_selection_calibration_stage: Literal["development", "locked_audit"] | None
    correction_contract_hash: Sha256
    representation_id: str
    latent_cache_index_hash: Sha256
    compiled_problem_hash: Sha256
    resolved_config_hash: Sha256
    mathematical_contract_hash: Sha256
    count_estimator_contract_hash: Sha256 | None
    loss_scale_hash: Sha256
    baseline_registry_hash: Sha256
    compute_budget_hash: Sha256
    output_quota_hash: Sha256
    implementation_tree_hash: Sha256
    capabilities: ResolvedRunCapabilities

    @model_validator(mode="after")
    def validate_compiled_id(self) -> CompiledRunContract:
        expected = self.identity(id_field="compiled_run_id")
        if self.compiled_run_id != expected:
            raise ValueError(f"compiled_run_id mismatch: expected {expected}.")
        return self


class CheckpointManifest(StrictModel):
    schema_version: int = 1
    checkpoint_id: str
    compiled_run_id: str
    generation: int = Field(ge=0)
    update: int = Field(ge=0)
    stage: str
    parent_checkpoint_id: str | None = None
    selection_source_checkpoint_id: str | None = None
    model: ArtifactRef
    optimizer: ArtifactRef
    optimizer_tree: ArtifactRef
    rng: ArtifactRef
    sampler: ArtifactRef
    training_state: ArtifactRef


class InferenceBundleManifest(StrictModel):
    schema_version: int = 1
    run_id: str
    compiled_run_id: str
    selected_checkpoint_id: str
    recipe_id: Literal["credo.count_sde_v4"] = "credo.count_sde_v4"
    recipe_version: Literal["4.0.dev36", "4.0.dev37"] = "4.0.dev37"
    selected_family: Literal[
        "configured_checkpoint",
        "gene_decoder_selected",
        "state_validation_selected",
        "global_terminal_null",
        "shrunk_sister_guide_target_terminal",
        "selected_training_only_target_main",
        "target_plus_source_target_interaction",
    ]
    selection: ArtifactRef
    model: ArtifactRef
    run_contract: ArtifactRef
    evaluation_seed_plan: ArtifactRef
    output_schema: ArtifactRef
    capabilities: ResolvedRunCapabilities


class EvaluationBundleManifest(StrictModel):
    schema_version: int = 1
    evaluation_id: str
    run_id: str
    plan: ArtifactRef
    metrics: ArtifactRef
    predictions: ArtifactRef
    audit: ArtifactRef


class SealedRunManifest(StrictModel):
    schema_version: int = 1
    sealed_id: str
    inference: ArtifactRef
    evaluations: tuple[ArtifactRef, ...]
    claim_audit: ArtifactRef


class CounterfactualBranch(StrictModel):
    branch_id: str
    effect_mode: Literal["factual", "reference"]
    context_mode: Literal["source_fixed", "pool_dynamic", "reference_dynamic"]


class CounterfactualDesign(StrictModel):
    series_index: int = Field(ge=0)
    branches: tuple[CounterfactualBranch, ...]
    particles: int = Field(default=128, gt=0)
    steps: int = Field(default=8, gt=0)
    seed: int = Field(default=0, ge=0)

    @field_validator("branches")
    @classmethod
    def unique_branches(
        cls, value: tuple[CounterfactualBranch, ...]
    ) -> tuple[CounterfactualBranch, ...]:
        ids = [branch.branch_id for branch in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Counterfactual branch IDs must be unique.")
        return value


class PredictionQuery(StrictModel):
    schema_version: int = 1
    series_indices: tuple[int, ...]
    particles: int = Field(gt=0)
    physical_grid: PhysicalGrid
    seed: int = Field(ge=0)
    include_terminal_particles: bool = False
    output_bytes_limit: int = Field(gt=0)


class EvaluationPlan(StrictModel):
    schema_version: int = 1
    evaluation_plan_id: str
    run_id: str
    information_set_hash: Sha256
    seed_plan_hash: Sha256
    baseline_registry_hash: Sha256
    multiplicity_plan_hash: Sha256
    one_shot_outer_access: Literal[True] = True


class BaselineRegistry(StrictModel):
    schema_version: int = 1
    registry_id: str
    baselines: tuple[BaselineInformationSet, ...]


class MultiplicityPlan(StrictModel):
    schema_version: int = 1
    plan_id: str
    procedure: Literal["BH", "hierarchical_BH"] = "BH"
    q: float = Field(default=0.05, gt=0, lt=1)
    families: tuple[str, ...]
    reporting_universe: str


class ClaimRecordV1(StrictModel):
    """Read-only dev28 claim row retained for immutable evidence."""

    claim_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    parent_claim_ids: tuple[str, ...] = ()
    endpoint: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    direction: Literal["lower", "higher"]
    primary_or_secondary: Literal["primary", "secondary"]
    claim_family: Literal["P", "M", "E", "G"]
    resampling_unit: str = Field(min_length=1)
    multiplicity_method: Literal[
        "paired_max_statistic",
        "joint_max_statistic",
        "holm_fwer",
        "westfall_young",
        "bh_fdr",
        "descriptive_only",
    ]
    external_independence_class: Literal[
        "not_external",
        "unresolved_blocked",
        "same_data_reexpression",
        "shared_samples_distinct_assay",
        "distinct_samples_same_study",
        "independent_study",
    ] = "not_external"
    eligible_wording: str = Field(min_length=1)
    forbidden_wording: tuple[str, ...]

    @model_validator(mode="after")
    def validate_claim_v1(self) -> ClaimRecordV1:
        if self.claim_id in self.parent_claim_ids:
            raise ValueError("A claim cannot be its own parent.")
        if not self.forbidden_wording or any(not value for value in self.forbidden_wording):
            raise ValueError("Every claim requires explicit forbidden wording.")
        return self


class ClaimRegistryV1(StrictModel):
    """Read-only dev28 registry; v2 is required for new evidence."""

    schema_version: Literal[1] = 1
    registry_id: str
    frozen_before_first_relevant_outer_evaluation: Literal[True] = True
    records: tuple[ClaimRecordV1, ...]

    @model_validator(mode="after")
    def validate_registry_v1(self) -> ClaimRegistryV1:
        ids = [record.claim_id for record in self.records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Claim registry IDs must be nonempty and unique.")
        known = set(ids)
        if any(not set(record.parent_claim_ids) <= known for record in self.records):
            raise ValueError("Claim registry contains an unknown parent claim.")
        expected = self.identity(id_field="registry_id")
        if self.registry_id != expected:
            raise ValueError(f"registry_id mismatch: expected {expected}.")
        return self


class RobustnessAxisV1(StrictModel):
    axis_id: str = Field(min_length=1)
    values: tuple[str, ...]
    frozen_before_corresponding_outer_outcomes: Literal[True] = True

    @model_validator(mode="after")
    def validate_axis_v1(self) -> RobustnessAxisV1:
        if len(self.values) < 2 or len(self.values) != len(set(self.values)):
            raise ValueError("A robustness axis requires at least two unique values.")
        return self


class G14RobustnessPlanV1(StrictModel):
    schema_version: Literal[1] = 1
    plan_id: str
    axes: tuple[RobustnessAxisV1, ...]
    base_contract_hashes: tuple[Sha256, ...]
    model_fitting: Literal[False] = False
    model_selection: Literal[False] = False
    threshold_adjustment: Literal[False] = False
    target_discovery: Literal[False] = False
    sensitivities_select_reported_model: Literal[False] = False

    @model_validator(mode="after")
    def validate_robustness_plan_v1(self) -> G14RobustnessPlanV1:
        axis_ids = [axis.axis_id for axis in self.axes]
        if not axis_ids or len(axis_ids) != len(set(axis_ids)):
            raise ValueError("G14 robustness axes must be nonempty and unique.")
        if not self.base_contract_hashes:
            raise ValueError("G14 robustness must bind at least one base contract.")
        expected = self.identity(id_field="plan_id")
        if self.plan_id != expected:
            raise ValueError(f"plan_id mismatch: expected {expected}.")
        return self


class G14MultiplicityFamilyV1(StrictModel):
    family_id: Literal["P", "M", "E", "G"]
    claim_ids: tuple[str, ...]
    method: Literal[
        "paired_max_statistic",
        "joint_max_or_holm",
        "holm_or_westfall_young",
        "bh_fdr_locked_family",
    ]
    error_rate: float = Field(default=0.05, gt=0, le=0.05)
    conditional_on_observed_donors: bool = False

    @model_validator(mode="after")
    def validate_family_v1(self) -> G14MultiplicityFamilyV1:
        if not self.claim_ids or len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("Multiplicity family claim IDs must be nonempty and unique.")
        if self.family_id == "P" and not self.conditional_on_observed_donors:
            raise ValueError("Family P inference must be conditional on observed donors.")
        return self


class G14MultiplicityContractV1(StrictModel):
    schema_version: Literal[1] = 1
    multiplicity_contract_id: str
    claim_registry_id: str
    families: tuple[G14MultiplicityFamilyV1, ...]
    general_population_donor_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_g14_multiplicity_v1(self) -> G14MultiplicityContractV1:
        family_ids = [family.family_id for family in self.families]
        if not family_ids or len(family_ids) != len(set(family_ids)):
            raise ValueError("G14 multiplicity families must be nonempty and unique.")
        claim_ids = [claim for family in self.families for claim in family.claim_ids]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("One claim cannot occur in multiple multiplicity families.")
        expected = self.identity(id_field="multiplicity_contract_id")
        if self.multiplicity_contract_id != expected:
            raise ValueError(f"multiplicity_contract_id mismatch: expected {expected}.")
        return self


class G14SealContractV1(StrictModel):
    """Read-only dev28 evidence-only seal skeleton."""

    schema_version: Literal[1] = 1
    g14_contract_id: str
    claim_registry_id: str
    robustness_plan_id: str
    multiplicity_contract_id: str
    purpose: Literal["robustness_multiplicity_and_claim_sealing"] = (
        "robustness_multiplicity_and_claim_sealing"
    )
    model_fitting: Literal[False] = False
    model_selection: Literal[False] = False
    threshold_adjustment: Literal[False] = False
    target_discovery: Literal[False] = False
    required_outputs: tuple[
        Literal[
            "ROBUSTNESS_MATRIX.parquet",
            "SIMULTANEOUS_INTERVALS.parquet",
            "MULTIPLICITY_DECISION.json",
            "CLAIM_LEDGER.parquet",
            "FINAL_EVIDENCE_GRAPH.json",
            "FINAL_CLAIM_SEAL_RECEIPT.json",
            "SHA256SUMS",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_g14_seal_v1(self) -> G14SealContractV1:
        expected_outputs = {
            "ROBUSTNESS_MATRIX.parquet",
            "SIMULTANEOUS_INTERVALS.parquet",
            "MULTIPLICITY_DECISION.json",
            "CLAIM_LEDGER.parquet",
            "FINAL_EVIDENCE_GRAPH.json",
            "FINAL_CLAIM_SEAL_RECEIPT.json",
            "SHA256SUMS",
        }
        if set(self.required_outputs) != expected_outputs or len(self.required_outputs) != len(
            expected_outputs
        ):
            raise ValueError("G14 seal output surface is incomplete or duplicated.")
        expected = self.identity(id_field="g14_contract_id")
        if self.g14_contract_id != expected:
            raise ValueError(f"g14_contract_id mismatch: expected {expected}.")
        return self


class ClaimRecord(StrictModel):
    """One preregistered G14 claim key and its allowable interpretation."""

    claim_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    parent_claim_ids: tuple[str, ...] = ()
    endpoint: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    direction: Literal["lower", "higher"]
    primary_or_secondary: Literal["primary", "secondary"]
    claim_family: Literal["P", "M", "E", "G"]
    resampling_unit: str = Field(min_length=1)
    multiplicity_method: Literal[
        "paired_max_statistic",
        "joint_max_statistic",
        "holm_fwer",
        "westfall_young",
        "bh_fdr",
        "descriptive_only",
    ]
    external_independence_class: Literal[
        "not_external",
        "unresolved_blocked",
        "same_data_reexpression",
        "shared_samples_distinct_assay",
        "distinct_samples_same_study",
        "independent_study",
    ] = "not_external"
    required_robustness_axis_ids: tuple[str, ...]
    eligible_wording: str = Field(min_length=1)
    forbidden_wording: tuple[str, ...]

    @model_validator(mode="after")
    def validate_claim(self) -> ClaimRecord:
        if self.claim_id in self.parent_claim_ids:
            raise ValueError("A claim cannot be its own parent.")
        if not self.forbidden_wording or any(not value for value in self.forbidden_wording):
            raise ValueError("Every claim requires explicit forbidden wording.")
        if not self.required_robustness_axis_ids or len(self.required_robustness_axis_ids) != len(
            set(self.required_robustness_axis_ids)
        ):
            raise ValueError("Every claim requires unique robustness-axis identities.")
        return self


class ClaimRegistry(StrictModel):
    """Immutable preregistered claim universe for G14."""

    schema_version: Literal[2] = 2
    registry_id: str
    frozen_before_first_relevant_outer_evaluation: Literal[True] = True
    records: tuple[ClaimRecord, ...]

    @model_validator(mode="after")
    def validate_registry(self) -> ClaimRegistry:
        ids = [record.claim_id for record in self.records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Claim registry IDs must be nonempty and unique.")
        known = set(ids)
        for record in self.records:
            if not set(record.parent_claim_ids) <= known:
                raise ValueError("Claim registry contains an unknown parent claim.")
        visiting: set[str] = set()
        visited: set[str] = set()
        parent_map = {record.claim_id: record.parent_claim_ids for record in self.records}

        def visit(claim_id: str) -> None:
            if claim_id in visiting:
                raise ValueError("Claim registry parent graph must be acyclic.")
            if claim_id in visited:
                return
            visiting.add(claim_id)
            for parent_id in parent_map[claim_id]:
                visit(parent_id)
            visiting.remove(claim_id)
            visited.add(claim_id)

        for claim_id in ids:
            visit(claim_id)
        expected = self.identity(id_field="registry_id")
        if self.registry_id != expected:
            raise ValueError(f"registry_id mismatch: expected {expected}.")
        return self


class RobustnessAxis(StrictModel):
    axis_id: str = Field(min_length=1)
    values: tuple[str, ...]
    claim_ids: tuple[str, ...]
    frozen_before_corresponding_outer_outcomes: Literal[True] = True

    @model_validator(mode="after")
    def validate_axis(self) -> RobustnessAxis:
        if len(self.values) < 2 or len(self.values) != len(set(self.values)):
            raise ValueError("A robustness axis requires at least two unique values.")
        if not self.claim_ids or len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("A robustness axis requires unique applicable claim IDs.")
        return self


class G14RobustnessPlan(StrictModel):
    schema_version: Literal[2] = 2
    plan_id: str
    axes: tuple[RobustnessAxis, ...]
    base_contract_hashes: tuple[Sha256, ...]
    model_fitting: Literal[False] = False
    model_selection: Literal[False] = False
    threshold_adjustment: Literal[False] = False
    target_discovery: Literal[False] = False
    sensitivities_select_reported_model: Literal[False] = False

    @model_validator(mode="after")
    def validate_robustness_plan(self) -> G14RobustnessPlan:
        axis_ids = [axis.axis_id for axis in self.axes]
        if not axis_ids or len(axis_ids) != len(set(axis_ids)):
            raise ValueError("G14 robustness axes must be nonempty and unique.")
        if not self.base_contract_hashes:
            raise ValueError("G14 robustness must bind at least one base contract.")
        expected = self.identity(id_field="plan_id")
        if self.plan_id != expected:
            raise ValueError(f"plan_id mismatch: expected {expected}.")
        return self


class G14MultiplicityFamily(StrictModel):
    family_id: Literal["P", "M", "E", "G"]
    claim_ids: tuple[str, ...]
    method: Literal[
        "paired_max_statistic",
        "joint_max_or_holm",
        "holm_or_westfall_young",
        "bh_fdr_locked_family",
    ]
    error_rate: float = Field(default=0.05, gt=0, le=0.05)
    conditional_on_observed_donors: bool = False

    @model_validator(mode="after")
    def validate_family(self) -> G14MultiplicityFamily:
        if not self.claim_ids or len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("Multiplicity family claim IDs must be nonempty and unique.")
        if self.family_id == "P" and not self.conditional_on_observed_donors:
            raise ValueError("Family P inference must be conditional on observed donors.")
        return self


class G14MultiplicityContract(StrictModel):
    schema_version: Literal[2] = 2
    multiplicity_contract_id: str
    claim_registry_id: str
    families: tuple[G14MultiplicityFamily, ...]
    general_population_donor_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_g14_multiplicity(self) -> G14MultiplicityContract:
        family_ids = [family.family_id for family in self.families]
        if not family_ids or len(family_ids) != len(set(family_ids)):
            raise ValueError("G14 multiplicity families must be nonempty and unique.")
        claim_ids = [claim for family in self.families for claim in family.claim_ids]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("One claim cannot occur in multiple multiplicity families.")
        expected = self.identity(id_field="multiplicity_contract_id")
        if self.multiplicity_contract_id != expected:
            raise ValueError(f"multiplicity_contract_id mismatch: expected {expected}.")
        return self


class G14SealContract(StrictModel):
    """Final evidence-only stage; it cannot train, select, or discover."""

    schema_version: Literal[2] = 2
    g14_contract_id: str
    claim_registry_id: str
    robustness_plan_id: str
    multiplicity_contract_id: str
    purpose: Literal["robustness_multiplicity_and_claim_sealing"] = (
        "robustness_multiplicity_and_claim_sealing"
    )
    model_fitting: Literal[False] = False
    model_selection: Literal[False] = False
    threshold_adjustment: Literal[False] = False
    target_discovery: Literal[False] = False
    claim_decisions: dict[str, Literal["promoted", "not_promoted", "blocked", "not_run"]]
    component_receipt_hashes: dict[str, Sha256]
    required_outputs: tuple[
        Literal[
            "ROBUSTNESS_MATRIX.parquet",
            "SIMULTANEOUS_INTERVALS.parquet",
            "MULTIPLICITY_DECISION.json",
            "CLAIM_LEDGER.parquet",
            "FINAL_EVIDENCE_GRAPH.json",
            "FINAL_CLAIM_SEAL_RECEIPT.json",
            "SHA256SUMS",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_g14_seal(self) -> G14SealContract:
        expected_outputs = {
            "ROBUSTNESS_MATRIX.parquet",
            "SIMULTANEOUS_INTERVALS.parquet",
            "MULTIPLICITY_DECISION.json",
            "CLAIM_LEDGER.parquet",
            "FINAL_EVIDENCE_GRAPH.json",
            "FINAL_CLAIM_SEAL_RECEIPT.json",
            "SHA256SUMS",
        }
        if set(self.required_outputs) != expected_outputs or len(self.required_outputs) != len(
            expected_outputs
        ):
            raise ValueError("G14 seal output surface is incomplete or duplicated.")
        if not self.claim_decisions:
            raise ValueError("G14 seal must bind every final claim decision.")
        if not self.component_receipt_hashes:
            raise ValueError("G14 evidence graph must bind actual component receipt hashes.")
        expected = self.identity(id_field="g14_contract_id")
        if self.g14_contract_id != expected:
            raise ValueError(f"g14_contract_id mismatch: expected {expected}.")
        return self


class CandidateSelectionPlan(StrictModel):
    schema_version: int = 1
    plan_id: str
    information_set_hash: Sha256
    score: str
    direction: Literal["higher", "lower"]
    tie_rule: str
    maximum_candidates: int = Field(gt=0)
    minimum_support: int = Field(ge=1)
    protected_endpoint_selection: Literal[False] = False


class StateSelectionCalibration(StrictModel):
    """Frozen pooled calibration for nested state-family selection."""

    schema_version: int = 1
    calibration_id: str
    method: Literal["pooled_nested_two_null"] = "pooled_nested_two_null"
    calibration_stage: Literal["development", "locked_audit"]
    development_calibration_sha256: Sha256 | None = None
    repeated_per_null: int = Field(ge=119)
    false_target_main_count: Literal[0] = 0
    false_interaction_count: Literal[0] = 0
    false_joint_interaction_count: Literal[0] = 0
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    confidence_method: Literal["clopper_pearson_one_sided"] = "clopper_pearson_one_sided"
    familywise_error_target: float = Field(default=0.05, gt=0.0, le=0.05)
    false_target_main_rate_upper_bound: float = Field(ge=0.0, le=0.05)
    false_interaction_rate_upper_bound: float = Field(ge=0.0, le=0.05)
    false_joint_interaction_rate_upper_bound: float = Field(ge=0.0, le=0.05)
    target_minimum_improvement: float = Field(gt=0.0)
    interaction_minimum_improvement: float = Field(gt=0.0)
    checkpoint_updates: tuple[int, ...]
    results_artifact: ArtifactRef
    calibration_protocol_hash: Sha256

    @model_validator(mode="after")
    def valid_schedule(self) -> StateSelectionCalibration:
        if not self.checkpoint_updates:
            raise ValueError("Calibration must bind a nonempty checkpoint schedule.")
        if tuple(sorted(set(self.checkpoint_updates))) != self.checkpoint_updates:
            raise ValueError("Calibration checkpoint updates must be increasing and unique.")
        if self.calibration_stage == "locked_audit" and self.repeated_per_null < 199:
            raise ValueError("Locked audit calibration requires at least 199 fits per null.")
        if (self.calibration_stage == "locked_audit") != (
            self.development_calibration_sha256 is not None
        ):
            raise ValueError("Only locked audits must bind the threshold-development calibration.")
        exact_upper = 1.0 - (1.0 - self.confidence_level) ** (1.0 / self.repeated_per_null)
        observed_bounds = (
            self.false_target_main_rate_upper_bound,
            self.false_interaction_rate_upper_bound,
            self.false_joint_interaction_rate_upper_bound,
        )
        if any(
            not math.isclose(value, exact_upper, rel_tol=0.0, abs_tol=1e-12)
            for value in observed_bounds
        ):
            raise ValueError(
                "A null-family upper bound is inconsistent with zero-failure one-sided "
                "Clopper-Pearson calibration."
            )
        return self


class StateCalibrationCheckpointScore(StrictModel):
    update: int = Field(ge=0)
    global_null_score: float = Field(ge=0.0)
    shrunk_target_only_score: float = Field(ge=0.0)
    empirical_bayes_target_score: float = Field(ge=0.0)
    target_terminal_score: float = Field(ge=0.0)
    target_delta_score: float = Field(ge=0.0)
    linear_source_plus_target_score: float = Field(ge=0.0)
    best_target_only_score: float = Field(ge=0.0)
    best_target_only_baseline: Literal[
        "shrunk_target_only", "empirical_bayes_target", "target_terminal"
    ]
    best_noninteraction_score: float = Field(ge=0.0)
    best_noninteraction_baseline: Literal[
        "shrunk_target_only",
        "empirical_bayes_target",
        "target_terminal",
        "target_delta",
        "linear_source_plus_target",
    ]
    interaction_score: float = Field(ge=0.0)

    @model_validator(mode="after")
    def exact_best_baselines(self) -> StateCalibrationCheckpointScore:
        target_scores = {
            "shrunk_target_only": self.shrunk_target_only_score,
            "empirical_bayes_target": self.empirical_bayes_target_score,
            "target_terminal": self.target_terminal_score,
        }
        all_scores = {
            **target_scores,
            "target_delta": self.target_delta_score,
            "linear_source_plus_target": self.linear_source_plus_target_score,
        }
        if not math.isclose(
            self.best_target_only_score,
            target_scores[self.best_target_only_baseline],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or self.best_target_only_score != min(target_scores.values()):
            raise ValueError("Best target-only calibration score is inconsistent.")
        if not math.isclose(
            self.best_noninteraction_score,
            all_scores[self.best_noninteraction_baseline],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or self.best_noninteraction_score != min(all_scores.values()):
            raise ValueError("Best noninteraction calibration score is inconsistent.")
        return self


class StateSelectionCalibrationRow(StrictModel):
    """One randomization fit conditional on a fixed pooled training split."""

    replicate_index: int = Field(ge=0)
    null_family: Literal["global_target_main", "conditional_interaction", "joint_nested"]
    outer_fold_id: str
    inner_split_id: str
    permutation_seed: int = Field(ge=0)
    optimizer_seed: int = Field(ge=0)
    initialization_seed: int = Field(ge=0)
    permutation_sha256: Sha256
    fit_series_sha256: Sha256
    validation_series_sha256: Sha256
    checkpoint_scores: tuple[StateCalibrationCheckpointScore, ...]
    selected_update: int = Field(ge=0)
    selected_family: Literal[
        "global_terminal_null",
        "shrunk_sister_guide_target_terminal",
        "selected_training_only_target_main",
        "target_plus_source_target_interaction",
    ]
    maximum_target_main_gain: float
    maximum_interaction_gain: float
    false_target_main_selected: bool
    false_interaction_selected: bool
    false_joint_interaction_selected: bool

    @model_validator(mode="after")
    def exact_false_selection_label(self) -> StateSelectionCalibrationRow:
        if len({self.permutation_seed, self.optimizer_seed, self.initialization_seed}) != 3:
            raise ValueError(
                "Calibration randomization, optimizer, and initialization seeds must differ."
            )
        m1_or_m2 = self.selected_family != "global_terminal_null"
        m2 = self.selected_family == "target_plus_source_target_interaction"
        expected_target = self.null_family == "global_target_main" and m1_or_m2
        expected_interaction = self.null_family == "conditional_interaction" and m2
        expected_joint = self.null_family == "joint_nested" and m2
        if (
            self.false_target_main_selected != expected_target
            or self.false_interaction_selected != expected_interaction
            or self.false_joint_interaction_selected != expected_joint
        ):
            raise ValueError("Calibration false-selection labels disagree with the null family.")
        updates = tuple(item.update for item in self.checkpoint_scores)
        if not updates or tuple(sorted(set(updates))) != updates:
            raise ValueError("Calibration checkpoint scores must be ordered and unique.")
        if self.selected_family == "target_plus_source_target_interaction":
            if self.selected_update == 0 or self.selected_update not in updates:
                raise ValueError("Selected interaction update is absent from checkpoint scores.")
        elif self.selected_update != 0:
            raise ValueError("A noninteraction calibration family must select update zero.")
        return self


class StateSelectionCalibrationResults(StrictModel):
    """Row-level evidence and exact execution surface for null calibration."""

    schema_version: int = 1
    calibration_id: str
    method: Literal["pooled_nested_two_null"] = "pooled_nested_two_null"
    calibration_stage: Literal["development", "locked_audit"]
    development_calibration_sha256: Sha256 | None = None
    repeated_per_null: int = Field(ge=119)
    pooled_estimand: Literal["pooled_known_target_heldout_guide"]
    outer_fold_id: str
    inner_split_id: str
    pooled_outer_fold_ids: tuple[str, ...]
    pooled_inner_split_ids: tuple[str, ...]
    pooled_optimization_seeds: tuple[int, ...]
    state_split_seed: int = Field(ge=0)
    implementation_tree_hash: Sha256
    calibration_code_hash: Sha256
    environment_lock_hash: Sha256
    optimizer_fingerprint: Sha256
    device_type: Literal["cpu", "cuda"]
    dtype: Literal["float32", "float64"]
    deterministic_algorithms: bool
    representation_id: str
    split_manifest_hash: Sha256
    compiled_problem_hash: Sha256
    interaction_rank: int = Field(ge=1, le=8)
    interaction_scale: float = Field(gt=0.0)
    learning_rate: float = Field(gt=0.0)
    state_batch_size: int = Field(gt=0)
    state_full_batch: Literal[True] = True
    noninteraction_linear_ridge: float = Field(gt=0.0)
    source_target_main_penalty: float = Field(gt=0.0)
    source_target_interaction_penalty: float = Field(gt=0.0)
    target_minimum_improvement: float = Field(gt=0.0)
    interaction_minimum_improvement: float = Field(gt=0.0)
    checkpoint_updates: tuple[int, ...]
    guide_per_target_distribution_hash: Sha256
    support_distribution_hash: Sha256
    rows: tuple[StateSelectionCalibrationRow, ...]

    @model_validator(mode="after")
    def complete_rows(self) -> StateSelectionCalibrationResults:
        if tuple(sorted(set(self.checkpoint_updates))) != self.checkpoint_updates:
            raise ValueError("Result checkpoint updates must be increasing and unique.")
        if self.calibration_stage == "locked_audit" and self.repeated_per_null < 199:
            raise ValueError("Locked audit results require at least 199 fits per null.")
        if (self.calibration_stage == "locked_audit") != (
            self.development_calibration_sha256 is not None
        ):
            raise ValueError("Only locked audit results bind a development calibration.")
        if (
            len(set(self.pooled_outer_fold_ids)) < 2
            or len(self.pooled_inner_split_ids) != len(self.pooled_outer_fold_ids)
            or self.outer_fold_id not in self.pooled_outer_fold_ids
            or self.inner_split_id not in self.pooled_inner_split_ids
            or len(set(self.pooled_optimization_seeds)) < 3
        ):
            raise ValueError("Calibration results have an incomplete pooled validation plan.")
        if len(self.rows) != 3 * self.repeated_per_null:
            raise ValueError("Every null family must contain repeated_per_null result rows.")
        if tuple(row.replicate_index for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("Calibration replicate indices must be contiguous and ordered.")
        if (
            len({row.outer_fold_id for row in self.rows}) != 1
            or len({row.inner_split_id for row in self.rows}) != 1
        ):
            raise ValueError("Calibration rows must use one fixed pooled split identity.")
        if any(
            row.outer_fold_id != self.outer_fold_id or row.inner_split_id != self.inner_split_id
            for row in self.rows
        ):
            raise ValueError("Calibration row split identity differs from the result contract.")
        if (
            len({row.fit_series_sha256 for row in self.rows}) != 1
            or len({row.validation_series_sha256 for row in self.rows}) != 1
        ):
            raise ValueError("Calibration rows must use one fixed fit/validation series split.")
        expected_updates = (0, *self.checkpoint_updates)
        if any(
            tuple(score.update for score in row.checkpoint_scores) != expected_updates
            for row in self.rows
        ):
            raise ValueError("Calibration row checkpoint grids differ from the result contract.")
        for family in ("global_target_main", "conditional_interaction", "joint_nested"):
            local = [row for row in self.rows if row.null_family == family]
            if len(local) != self.repeated_per_null:
                raise ValueError(f"Calibration null family {family} is incomplete.")
            if len({row.permutation_seed for row in local}) != len(local):
                raise ValueError(f"Calibration null family {family} has duplicate permutations.")
        return self


class SelectionManifest(StrictModel):
    """Immutable nested-family decision and post-selection refit binding."""

    schema_version: int = 1
    selection_id: str
    compiled_run_id: str
    policy: str
    metric: str
    score: float | None
    selected_update: int = Field(ge=0)
    selected_checkpoint_id: str
    selected_checkpoint_relative_uri: str
    candidate_updates: tuple[int, ...]
    selected_family: Literal[
        "configured_checkpoint",
        "gene_decoder_selected",
        "state_validation_selected",
        "global_terminal_null",
        "shrunk_sister_guide_target_terminal",
        "selected_training_only_target_main",
        "target_plus_source_target_interaction",
    ]
    inner_selected_update: int | None = Field(default=None, ge=0)
    inner_selected_checkpoint_id: str | None = None
    refit_checkpoint_id: str | None = None
    post_selection_refit: bool = False
    refit_series_hash: Sha256 | None = None
    global_null_score: float | None = None
    shrunk_target_only_score: float | None = None
    best_target_only_score: float | None = None
    best_target_only_baseline: (
        Literal["shrunk_target_only", "empirical_bayes_target", "target_terminal"] | None
    ) = None
    best_noninteraction_score: float | None = None
    best_noninteraction_baseline: (
        Literal[
            "shrunk_target_only",
            "empirical_bayes_target",
            "target_terminal",
            "target_delta",
            "linear_source_plus_target",
        ]
        | None
    ) = None
    interaction_score: float | None = None
    interaction_incremental_gain: float | None = None
    target_incremental_gain: float | None = None
    target_minimum_required_improvement: float | None = None
    interaction_minimum_required_improvement: float | None = None
    selection_calibration_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> SelectionManifest:
        expected = self.identity(id_field="selection_id")
        if self.selection_id != expected:
            raise ValueError(f"selection_id mismatch: expected {expected}.")
        if self.policy == "minimum_state_validation_null_guarded":
            required = (
                self.global_null_score,
                self.shrunk_target_only_score,
                self.best_target_only_score,
                self.best_target_only_baseline,
                self.best_noninteraction_score,
                self.best_noninteraction_baseline,
                self.interaction_score,
                self.interaction_incremental_gain,
                self.target_incremental_gain,
                self.target_minimum_required_improvement,
                self.interaction_minimum_required_improvement,
                self.selection_calibration_hash,
            )
            if any(value is None for value in required):
                raise ValueError("Null-guarded selections require complete nested-family evidence.")
        if self.post_selection_refit:
            if self.inner_selected_checkpoint_id is None or self.refit_checkpoint_id is None:
                raise ValueError("Refit selections must bind inner and refit checkpoint IDs.")
            if self.selected_checkpoint_id != self.refit_checkpoint_id:
                raise ValueError("Selected checkpoint must be the committed refit checkpoint.")
        return self


class BaselineInformationSet(StrictModel):
    schema_version: int = 1
    baseline_id: str
    code_hash: Sha256
    allowed_training_rows_hash: Sha256
    allowed_test_source_rows_hash: Sha256
    forbidden_endpoint_rows_hash: Sha256
    split_hash: Sha256
    representation_id: str
    denominator_hash: Sha256 | None = None
    aggregation_hash: Sha256
    seed_plan_hash: Sha256


class ContextAuditContract(StrictModel):
    schema_version: int = 1
    effective_rank_threshold: int = Field(default=2, ge=2)
    singular_ratio_threshold: float = Field(default=0.05, gt=0, lt=1)
    maximum_interaction_rank: int = Field(default=4, ge=1, le=4)
    maximum_condition_number: float = Field(default=50.0, gt=0)
    parameter_match_fraction: float = Field(default=0.01, ge=0, le=0.01)
    held_out_pool_information_set: Sha256
    observation_operator: ArtifactRef


class ContextAuditReceipt(StrictModel):
    schema_version: int = 1
    contract_hash: Sha256
    effective_rank: int = Field(ge=0)
    selected_rank: int = Field(ge=0)
    condition_number: float | None
    parameter_difference_fraction: float = Field(ge=0)
    held_out_access_pass: bool
    observation_operator_pass: bool
    status: Literal["eligible", "diagnostic_only"]


class ShardAppendContract(StrictModel):
    """Exact next P2 chunk identity required before any shard append."""

    schema_version: Literal[1] = 1
    chunk_contract_id: str
    build_plan_id: str = Field(min_length=1)
    shard_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    source_id: str = Field(min_length=1)
    source_cursor_start: int = Field(ge=0)
    source_cursor_end: int = Field(gt=0)
    row_ids_hash: Sha256
    guide_run_id: str = Field(min_length=1)
    chunk_rows: int = Field(gt=0)
    chunk_nnz: int = Field(ge=0)
    cumulative_row_identity_hash: Sha256

    @model_validator(mode="after")
    def validate_append_contract(self) -> ShardAppendContract:
        if self.source_cursor_end <= self.source_cursor_start:
            raise ValueError("Append source cursor interval must advance.")
        if self.source_cursor_end - self.source_cursor_start != self.chunk_rows:
            raise ValueError("Append cursor span must equal the frozen chunk row count.")
        expected = self.identity(id_field="chunk_contract_id")
        if self.chunk_contract_id != expected:
            raise ValueError(f"chunk_contract_id mismatch: expected {expected}.")
        return self


class ShardBuildCheckpoint(StrictModel):
    schema_version: int = 1
    build_plan_id: str
    phase: Literal["BUILDING", "FINALIZING", "FINALIZED"] = "BUILDING"
    append_plan_hash: Sha256
    source_cursor: int = Field(ge=0)
    completed_shard_hashes: tuple[Sha256, ...]
    temporary_shard_ids: tuple[str, ...] = ()
    row_count: int = Field(ge=0)
    nonzero_count: int = Field(ge=0)
    active_shard_index: int | None = Field(default=None, ge=0)
    active_shard_rows: int = Field(default=0, ge=0)
    active_shard_nonzeros: int = Field(default=0, ge=0)
    committed_chunks: int = Field(default=0, ge=0)
    source_id: str | None = None
    active_offsets_sha256: Sha256 | None = None
    expected_chunk_index: int | None = Field(default=None, ge=0)
    expected_source_cursor_start: int | None = Field(default=None, ge=0)
    expected_source_cursor_end: int | None = Field(default=None, gt=0)
    expected_row_ids_hash: Sha256 | None = None
    expected_guide_run_id: str | None = None
    expected_chunk_nnz: int | None = Field(default=None, ge=0)
    cumulative_row_identity_hash: Sha256
    finalizing_content_sha256: Sha256 | None = None
    finalizing_manifest: CountStoreManifest | None = None
    destination_name: str | None = None
    rng_state: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_shard_build_checkpoint(self) -> ShardBuildCheckpoint:
        if not self.build_plan_id:
            raise ValueError("Shard-build checkpoint requires a build-plan identity.")
        if len(self.completed_shard_hashes) != len(set(self.completed_shard_hashes)):
            raise ValueError("Completed shard hashes must be unique.")
        if len(self.temporary_shard_ids) != len(set(self.temporary_shard_ids)):
            raise ValueError("Temporary shard identities must be unique.")
        expected_fields = (
            self.expected_chunk_index,
            self.expected_source_cursor_start,
            self.expected_source_cursor_end,
            self.expected_row_ids_hash,
            self.expected_guide_run_id,
            self.expected_chunk_nnz,
        )
        if any(value is None for value in expected_fields) and any(
            value is not None for value in expected_fields
        ):
            raise ValueError("The complete next append identity must be present or absent.")
        finalizing_fields = (
            self.finalizing_content_sha256,
            self.finalizing_manifest,
            self.destination_name,
        )
        if self.phase == "BUILDING" and any(value is not None for value in finalizing_fields):
            raise ValueError("BUILDING cannot contain finalization identities.")
        if self.phase in {"FINALIZING", "FINALIZED"} and any(
            value is None for value in finalizing_fields
        ):
            raise ValueError("Finalization phases require content, manifest, and destination.")
        if self.destination_name is not None and (
            not self.destination_name
            or self.destination_name in {".", ".."}
            or "/" in self.destination_name
        ):
            raise ValueError("Shard destination must be one safe path component.")
        if self.phase == "FINALIZED" and self.active_shard_index is not None:
            raise ValueError("FINALIZED cannot retain an active shard index.")
        if self.active_shard_index is None:
            if self.active_shard_rows or self.active_shard_nonzeros:
                raise ValueError("A finalized checkpoint cannot retain active shard counts.")
            if self.active_offsets_sha256 is not None:
                raise ValueError("A finalized checkpoint cannot retain active offsets.")
        else:
            if not self.source_id or not self.temporary_shard_ids:
                raise ValueError("An active shard requires source and temporary-file identities.")
            if self.active_offsets_sha256 is None:
                raise ValueError("An active shard requires an offsets hash.")
            if self.active_shard_rows > self.row_count:
                raise ValueError("Active shard rows exceed total committed rows.")
            if self.active_shard_nonzeros > self.nonzero_count:
                raise ValueError("Active shard nonzeros exceed total committed nonzeros.")
        if self.phase != "BUILDING" and any(value is not None for value in expected_fields):
            raise ValueError("Finalization cannot retain an expected append identity.")
        return self


class RepresentationCheckpoint(StrictModel):
    schema_version: int = 1
    representation_contract_id: str
    update: int = Field(ge=0)
    model: ArtifactRef
    optimizer: ArtifactRef
    rng: ArtifactRef
    cell_sampler_cursor: int = Field(ge=0)
    loss_scale_hash: Sha256
    selected_inner_state: ArtifactRef | None = None


class DynamicsCheckpoint(StrictModel):
    schema_version: int = 1
    compiled_run_id: str
    update: int = Field(ge=0)
    model: ArtifactRef
    optimizer: ArtifactRef
    rng: ArtifactRef
    series_sampler_cursor: int = Field(ge=0)
    exact_count_cursor: int = Field(ge=0)
    brownian_plan_hash: Sha256
    bank_state: ArtifactRef | None = None
    compute_used: float = Field(ge=0)
    output_bytes_used: int = Field(ge=0)


class EvaluationCheckpoint(StrictModel):
    schema_version: int = 1
    run_id: str
    evaluation_id: str
    completed_chunks: tuple[Sha256, ...]
    particle_plan_hash: Sha256
    noise_plan_hash: Sha256
    output_bytes_used: int = Field(ge=0)
    compute_used: float = Field(ge=0)


def checked_contract(model: type[StrictModel], payload: dict[str, Any]) -> StrictModel:
    try:
        return model.model_validate(payload)
    except Exception as exc:  # pydantic aggregates the useful field diagnostics
        raise ContractError(str(exc)) from exc
