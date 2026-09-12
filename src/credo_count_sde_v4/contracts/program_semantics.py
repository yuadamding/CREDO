"""Dev40 count-linked biological-program contracts and qualification records."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .models import ArtifactRef, Sha256, StrictModel
from .scientific_scope import ScientificScope

__all__ = (
    "BaselineMetric",
    "BaselineName",
    "BiologicalProgramQualificationBundle",
    "CountLinkedProgramContract",
    "GeneLevelEffect",
    "GuideTargetConsistency",
    "HeldoutTargetMode",
    "LibrarySizeSemantics",
    "NegativeBinomialSemantics",
    "PerturbationProgramEffect",
    "ProgramDefinition",
    "ProgramNullContract",
    "ProgramQualificationProtocol",
    "ProgramQualificationReceipt",
    "ProgramQualificationStatus",
    "ProgramSimulationContract",
    "ProgramSplitEvaluation",
    "ProgramSplitKind",
    "ProgramUncertainty",
    "SparseLoadingPrior",
)


class NegativeBinomialSemantics(StrEnum):
    """Frozen NB parameterization used by the program head."""

    NB2_GENE_DISPERSION = "nb2_gene_dispersion_var_mu_plus_mu_squared_over_theta"


class LibrarySizeSemantics(StrEnum):
    """The library term is observed exposure, never a fitted biological effect."""

    OBSERVED_TOTAL_COUNT_OFFSET = "observed_total_count_log_offset"


class SparseLoadingPrior(StrEnum):
    """Qualified optimization surrogate for sparse gene-program loadings."""

    L1_WITH_UNIT_NORM_COLUMNS = "l1_with_unit_norm_columns"


class HeldoutTargetMode(StrEnum):
    """Whether an unseen target has legal predictive information."""

    IDENTIFIER_ONLY_NOT_ELIGIBLE = "identifier_only_not_eligible"
    PREDECLARED_TARGET_DESCRIPTORS = "predeclared_target_descriptors"


class CountLinkedProgramContract(StrictModel):
    """Static/checkpoint-conditional head, deliberately uncoupled from dynamics."""

    schema_id: Literal["credo.count_linked_program_contract"] = (
        "credo.count_linked_program_contract"
    )
    schema_version: Literal[1] = 1
    program_contract_id: str
    equation: Literal[
        "log_mu=log_library+sample_gene+state_gene+W@(reference+target+efficacy*guide)"
    ] = "log_mu=log_library+sample_gene+state_gene+W@(reference+target+efficacy*guide)"
    hierarchy_equation: Literal["a_ref+a_target+q_guide*a_guide"] = "a_ref+a_target+q_guide*a_guide"
    count_likelihood: NegativeBinomialSemantics
    library_size_semantics: LibrarySizeSemantics
    sparse_loading_prior: SparseLoadingPrior
    raw_nonnegative_integer_counts_required: Literal[True] = True
    gene_specific_dispersion: Literal[True] = True
    control_reference_centered: Literal[True] = True
    target_guide_hierarchy_enabled: Literal[True] = True
    state_dependence_enabled: bool
    checkpoint_dependence_enabled: bool
    checkpoint_time_semantics: Literal["linear_continuous_physical_time"] = (
        "linear_continuous_physical_time"
    )
    checkpoint_time_values: tuple[float, ...]
    donor_or_sample_effects_enabled: bool
    heldout_target_mode: HeldoutTargetMode
    target_descriptor_artifact: ArtifactRef | None = None
    feature_order_sha256: Sha256
    guide_target_map_sha256: Sha256
    fit_row_ids_sha256: Sha256
    genes: int = Field(gt=1)
    programs: int = Field(gt=0)
    state_dimension: int = Field(ge=0)
    samples: int = Field(gt=0)
    checkpoints: int = Field(gt=0)
    targets: int = Field(gt=0)
    guides: int = Field(gt=0)
    control_guide_indices: tuple[int, ...]
    loading_l1_weight: float = Field(gt=0)
    guide_deviation_l2_weight: float = Field(gt=0)
    target_effect_l2_weight: float = Field(gt=0)
    dynamics_coupled: Literal[False] = False
    checkpoint_schema_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_contract(self) -> CountLinkedProgramContract:
        if (
            not self.control_guide_indices
            or self.control_guide_indices != tuple(sorted(set(self.control_guide_indices)))
            or any(index < 0 or index >= self.guides for index in self.control_guide_indices)
        ):
            raise ValueError("Control-guide indices must be nonempty, unique, sorted, and valid.")
        descriptor_mode = self.heldout_target_mode == (
            HeldoutTargetMode.PREDECLARED_TARGET_DESCRIPTORS
        )
        if descriptor_mode != (self.target_descriptor_artifact is not None):
            raise ValueError("Held-out-target mode contradicts its descriptor artifact.")
        if self.state_dependence_enabled != (self.state_dimension > 0):
            raise ValueError("State dependence contradicts the declared state dimension.")
        if len(self.checkpoint_time_values) != self.checkpoints or any(
            right <= left
            for left, right in zip(
                self.checkpoint_time_values,
                self.checkpoint_time_values[1:],
                strict=False,
            )
        ):
            raise ValueError("Checkpoint physical times must be complete and increasing.")
        expected = self.identity(id_field="program_contract_id")
        if self.program_contract_id != expected:
            raise ValueError(f"program_contract_id mismatch: expected {expected}.")
        return self


class ProgramDefinition(StrictModel):
    """One sparse signed gene program, without a biological-name claim."""

    schema_id: Literal["credo.program_definition"] = "credo.program_definition"
    schema_version: Literal[1] = 1
    program_definition_id: str
    program_contract_id: str = Field(min_length=1)
    program_index: int = Field(ge=0)
    loading_artifact: ArtifactRef
    loading_threshold: float = Field(gt=0)
    nonzero_genes: int = Field(ge=0)
    positive_genes: int = Field(ge=0)
    negative_genes: int = Field(ge=0)
    l1_norm: float = Field(gt=0)
    l2_norm: float = Field(gt=0)
    training_derived_label: str | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> ProgramDefinition:
        if self.positive_genes + self.negative_genes != self.nonzero_genes:
            raise ValueError("Signed loading counts do not cover the nonzero genes.")
        expected = self.identity(id_field="program_definition_id")
        if self.program_definition_id != expected:
            raise ValueError(f"program_definition_id mismatch: expected {expected}.")
        return self


class PerturbationProgramEffect(StrictModel):
    """Control, target, and guide-deviation program effects for one scope."""

    schema_id: Literal["credo.perturbation_program_effect"] = "credo.perturbation_program_effect"
    schema_version: Literal[1] = 1
    program_effect_id: str
    program_contract_id: str = Field(min_length=1)
    scope: ScientificScope
    perturbation_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    guide_ids: tuple[str, ...]
    reference_activity: ArtifactRef
    target_activity: ArtifactRef
    guide_deviation_activity: ArtifactRef
    guide_efficiency: ArtifactRef

    @model_validator(mode="after")
    def validate_effect(self) -> PerturbationProgramEffect:
        if not self.guide_ids or self.guide_ids != tuple(sorted(set(self.guide_ids))):
            raise ValueError("Program-effect guides must be nonempty, unique, and sorted.")
        expected = self.identity(id_field="program_effect_id")
        if self.program_effect_id != expected:
            raise ValueError(f"program_effect_id mismatch: expected {expected}.")
        return self


class GeneLevelEffect(StrictModel):
    """Signed gene effect reconstructed through the sparse program hierarchy."""

    schema_id: Literal["credo.gene_level_effect"] = "credo.gene_level_effect"
    schema_version: Literal[1] = 1
    gene_effect_id: str
    program_contract_id: str = Field(min_length=1)
    program_effect_id: str = Field(min_length=1)
    scope: ScientificScope
    log_fold_change_artifact: ArtifactRef
    standard_error_artifact: ArtifactRef
    sign_probability_artifact: ArtifactRef
    decomposition_identity_max_abs_error: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_gene_effect(self) -> GeneLevelEffect:
        expected = self.identity(id_field="gene_effect_id")
        if self.gene_effect_id != expected:
            raise ValueError(f"gene_effect_id mismatch: expected {expected}.")
        return self


class GuideTargetConsistency(StrictModel):
    """Sister-guide agreement and target/guide variance decomposition."""

    schema_id: Literal["credo.guide_target_consistency"] = "credo.guide_target_consistency"
    schema_version: Literal[1] = 1
    consistency_id: str
    program_contract_id: str = Field(min_length=1)
    scope: ScientificScope
    per_target_artifact: ArtifactRef
    median_sister_guide_correlation: float = Field(ge=-1, le=1)
    target_variance_fraction: float = Field(ge=0, le=1)
    inconsistent_target_fraction: float = Field(ge=0, le=1)
    minimum_sister_guides: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_consistency(self) -> GuideTargetConsistency:
        expected = self.identity(id_field="consistency_id")
        if self.consistency_id != expected:
            raise ValueError(f"consistency_id mismatch: expected {expected}.")
        return self


class ProgramUncertainty(StrictModel):
    """Seed/donor stability and calibrated inclusion uncertainty."""

    schema_id: Literal["credo.program_uncertainty"] = "credo.program_uncertainty"
    schema_version: Literal[1] = 1
    uncertainty_id: str
    program_contract_id: str = Field(min_length=1)
    scope: ScientificScope
    seed_count: int = Field(ge=3)
    donor_count: int = Field(ge=0)
    seed_loading_stability_artifact: ArtifactRef
    donor_loading_stability_artifact: ArtifactRef | None = None
    inclusion_probability_artifact: ArtifactRef
    null_inclusion_artifact: ArtifactRef
    median_seed_loading_correlation: float = Field(ge=-1, le=1)
    median_donor_loading_correlation: float | None = Field(default=None, ge=-1, le=1)
    donor_stability_available: bool
    null_inclusion_rate: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_uncertainty(self) -> ProgramUncertainty:
        available = (
            self.donor_loading_stability_artifact is not None
            and self.median_donor_loading_correlation is not None
            and self.donor_count >= 2
        )
        if self.donor_stability_available != available:
            raise ValueError("Donor-stability availability contradicts its evidence fields.")
        expected = self.identity(id_field="uncertainty_id")
        if self.uncertainty_id != expected:
            raise ValueError(f"uncertainty_id mismatch: expected {expected}.")
        return self


class BaselineName(StrEnum):
    """Frozen Dev40-B comparator set."""

    EXACT_GLOBAL_GENE_FREQUENCY = "exact_global_gene_frequency"
    PSEUDOBULK_NEGATIVE_BINOMIAL = "pseudobulk_negative_binomial"
    PER_GENE_DIFFERENTIAL_EXPRESSION = "per_gene_differential_expression"
    GSFA_STYLE_SPARSE_FACTORS = "gsfa_style_sparse_factors"
    TARGET_AVERAGE = "target_average"
    CONTROL_ONLY = "control_only"


class BaselineMetric(StrictModel):
    """Comparable held-out count metric for one method."""

    baseline: BaselineName
    mean_log_likelihood_per_count: float
    mean_negative_log_likelihood_per_cell: float = Field(ge=0)


class ProgramSplitKind(StrEnum):
    """Non-interchangeable outer validation splits."""

    HELDOUT_DONOR = "heldout_donor"
    HELDOUT_GUIDE_TARGET_SHARED = "heldout_guide_target_shared"
    HELDOUT_TARGET = "heldout_target"
    HELDOUT_TIME = "heldout_time"


class ProgramQualificationProtocol(StrictModel):
    """Content-addressed Dev40-B optimization, split, and threshold freeze."""

    schema_id: Literal["credo.program_qualification_protocol"] = (
        "credo.program_qualification_protocol"
    )
    schema_version: Literal[1] = 1
    qualification_protocol_id: str
    split_kinds: tuple[ProgramSplitKind, ...]
    baselines: tuple[BaselineName, ...]
    evaluation_dispersion_semantics: Literal[
        "training_only_method_of_moments_shared_across_candidates"
    ] = "training_only_method_of_moments_shared_across_candidates"
    stability_seeds: tuple[int, ...]
    sparse_factor_rank: int = Field(gt=0)
    null_replicates: int = Field(ge=20)
    null_fit_epochs: int = Field(gt=0)
    minimum_log_likelihood_improvement: float
    minimum_gene_sign_accuracy: float = Field(ge=0, le=1)
    minimum_seed_loading_stability: float = Field(ge=-1, le=1)
    minimum_donor_loading_stability: float = Field(ge=-1, le=1)
    minimum_sister_guide_correlation: float = Field(ge=-1, le=1)
    maximum_null_inclusion_rate: float = Field(gt=0, lt=1)
    effect_inclusion_threshold: float = Field(gt=0)
    loading_inclusion_threshold: float = Field(gt=0)
    inner_validation_fraction: float = Field(gt=0, lt=0.5)
    seed: int = Field(ge=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    minimum_delta: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    loading_l1_weight: float = Field(gt=0)
    guide_deviation_l2_weight: float = Field(gt=0)
    target_effect_l2_weight: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_protocol(self) -> ProgramQualificationProtocol:
        if self.split_kinds != tuple(ProgramSplitKind):
            raise ValueError("Qualification split kinds must match the frozen sequence.")
        if self.baselines != tuple(BaselineName):
            raise ValueError("Qualification baselines must match the frozen sequence.")
        if len(self.stability_seeds) < 3 or self.stability_seeds != tuple(
            sorted(set(self.stability_seeds))
        ):
            raise ValueError("Stability seeds must contain at least three unique sorted values.")
        expected = self.identity(id_field="qualification_protocol_id")
        if self.qualification_protocol_id != expected:
            raise ValueError(f"qualification_protocol_id mismatch: expected {expected}.")
        return self


class ProgramSplitEvaluation(StrictModel):
    """One leakage-audited split and every frozen comparator."""

    split_id: str = Field(min_length=1)
    kind: ProgramSplitKind
    eligible: bool
    passed: bool
    ineligibility_reason: str | None = None
    fit_units: tuple[str, ...]
    evaluation_units: tuple[str, ...]
    protected_expression_access_contract_id: str | None = None
    model_mean_log_likelihood_per_count: float | None = None
    model_mean_negative_log_likelihood_per_cell: float | None = Field(default=None, ge=0)
    baselines: tuple[BaselineMetric, ...] = ()
    improvement_over_best_baseline: float | None = None
    gene_sign_accuracy: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_split(self) -> ProgramSplitEvaluation:
        if set(self.fit_units) & set(self.evaluation_units):
            raise ValueError("Program split fit and evaluation units overlap.")
        if self.eligible:
            if self.ineligibility_reason is not None:
                raise ValueError("Eligible split cannot carry an ineligibility reason.")
            if not self.fit_units or not self.evaluation_units:
                raise ValueError("Eligible split requires fit and evaluation units.")
            if tuple(item.baseline for item in self.baselines) != tuple(BaselineName):
                raise ValueError("Eligible split must report every baseline in frozen order.")
            if None in {
                self.model_mean_log_likelihood_per_count,
                self.model_mean_negative_log_likelihood_per_cell,
                self.improvement_over_best_baseline,
                self.gene_sign_accuracy,
            }:
                raise ValueError("Eligible split is missing a required predictive metric.")
        else:
            if self.passed or self.ineligibility_reason is None:
                raise ValueError("Ineligible split must be unpassed with an explicit reason.")
            if self.baselines or any(
                value is not None
                for value in (
                    self.model_mean_log_likelihood_per_count,
                    self.model_mean_negative_log_likelihood_per_cell,
                    self.improvement_over_best_baseline,
                    self.gene_sign_accuracy,
                )
            ):
                raise ValueError("Ineligible split cannot carry fabricated metrics.")
        if self.kind == ProgramSplitKind.HELDOUT_TIME and self.eligible != (
            self.protected_expression_access_contract_id is not None
        ):
            raise ValueError("Held-out time requires a protected-expression access contract.")
        return self


class ProgramQualificationStatus(StrEnum):
    """Scientific result gate; engineering execution is not a program result."""

    PASS_SCIENTIFIC = "pass_scientific"
    FAIL_QUALIFICATION = "fail_qualification"
    ENGINEERING_ONLY = "engineering_only"


class ProgramQualificationReceipt(StrictModel):
    """Complete Dev40-B promotion gate."""

    schema_id: Literal["credo.program_qualification_receipt"] = (
        "credo.program_qualification_receipt"
    )
    schema_version: Literal[1] = 1
    qualification_receipt_id: str
    program_contract_id: str = Field(min_length=1)
    qualification_protocol_id: str = Field(min_length=1)
    scope: ScientificScope
    status: ProgramQualificationStatus
    split_evaluations: tuple[ProgramSplitEvaluation, ...]
    seed_stability_pass: bool
    donor_stability_pass: bool
    null_inclusion_calibrated: bool
    sister_guide_consistency_pass: bool
    heldout_target_performance_pass: bool
    gene_sign_calibration_pass: bool
    external_pathway_concordance_available: bool
    external_pathway_concordance_pass: bool | None = None
    qualified_program_definition_ids: tuple[str, ...]
    metrics_artifact: ArtifactRef
    training_log_artifact: ArtifactRef
    software_environment_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_receipt(self) -> ProgramQualificationReceipt:
        kinds = tuple(item.kind for item in self.split_evaluations)
        if kinds != tuple(ProgramSplitKind):
            raise ValueError("Qualification must report every split kind in frozen order.")
        if self.external_pathway_concordance_available != (
            self.external_pathway_concordance_pass is not None
        ):
            raise ValueError("Pathway availability contradicts its decision.")
        gates = (
            self.seed_stability_pass,
            self.donor_stability_pass,
            self.null_inclusion_calibrated,
            self.sister_guide_consistency_pass,
            self.heldout_target_performance_pass,
            self.gene_sign_calibration_pass,
            all(item.eligible and item.passed for item in self.split_evaluations),
            self.external_pathway_concordance_pass is not False,
        )
        passed = all(gates)
        if (self.status == ProgramQualificationStatus.PASS_SCIENTIFIC) != passed:
            raise ValueError("Program qualification status contradicts its promotion gates.")
        if passed != bool(self.qualified_program_definition_ids):
            raise ValueError("Only a passed qualification may expose qualified programs.")
        if self.qualified_program_definition_ids != tuple(
            sorted(set(self.qualified_program_definition_ids))
        ):
            raise ValueError("Qualified program IDs must be unique and sorted.")
        expected = self.identity(id_field="qualification_receipt_id")
        if self.qualification_receipt_id != expected:
            raise ValueError(f"qualification_receipt_id mismatch: expected {expected}.")
        return self


class ProgramNullContract(StrictModel):
    """Frozen negative controls for false-program inclusion."""

    schema_id: Literal["credo.program_null_contract"] = "credo.program_null_contract"
    schema_version: Literal[1] = 1
    null_contract_id: str
    null_families: tuple[
        Literal[
            "control_label_permutation",
            "target_within_checkpoint_permutation",
            "negative_binomial_no_program",
        ],
        ...,
    ]
    replicate_count: int = Field(ge=20)
    maximum_program_inclusion_rate: float = Field(gt=0, lt=1)
    selection_threshold_frozen_before_nulls: Literal[True] = True

    @model_validator(mode="after")
    def validate_null(self) -> ProgramNullContract:
        expected_families = (
            "control_label_permutation",
            "target_within_checkpoint_permutation",
            "negative_binomial_no_program",
        )
        if self.null_families != expected_families:
            raise ValueError("Program null families must match the frozen sequence.")
        expected = self.identity(id_field="null_contract_id")
        if self.null_contract_id != expected:
            raise ValueError(f"null_contract_id mismatch: expected {expected}.")
        return self


class ProgramSimulationContract(StrictModel):
    """Fixed-truth count simulation for positive, null, and ablation tests."""

    schema_id: Literal["credo.program_simulation_contract"] = "credo.program_simulation_contract"
    schema_version: Literal[1] = 1
    simulation_contract_id: str
    seed: int = Field(ge=0)
    cells: int = Field(ge=100)
    genes: int = Field(ge=20)
    programs: int = Field(ge=2)
    donors: int = Field(ge=3)
    checkpoints: int = Field(ge=2)
    targets: int = Field(ge=3)
    guides_per_target: int = Field(ge=2)
    controls: int = Field(ge=2)
    null_program_indices: tuple[int, ...]
    truth_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_simulation(self) -> ProgramSimulationContract:
        if self.null_program_indices != tuple(sorted(set(self.null_program_indices))) or any(
            index < 0 or index >= self.programs for index in self.null_program_indices
        ):
            raise ValueError("Null program indices must be unique, sorted, and valid.")
        expected = self.identity(id_field="simulation_contract_id")
        if self.simulation_contract_id != expected:
            raise ValueError(f"simulation_contract_id mismatch: expected {expected}.")
        return self


class BiologicalProgramQualificationBundle(StrictModel):
    """Complete program result surface admitted to a perturbation dossier."""

    schema_id: Literal["credo.biological_program_qualification_bundle"] = (
        "credo.biological_program_qualification_bundle"
    )
    schema_version: Literal[1] = 1
    program_bundle_id: str
    study_id: str = Field(min_length=1)
    capability_assessment_id: str = Field(min_length=1)
    contract: CountLinkedProgramContract
    qualification_protocol: ProgramQualificationProtocol
    scope: ScientificScope
    program_definitions: tuple[ProgramDefinition, ...]
    perturbation_effects: tuple[PerturbationProgramEffect, ...]
    gene_effects: tuple[GeneLevelEffect, ...]
    guide_target_consistency: GuideTargetConsistency
    uncertainty: ProgramUncertainty
    qualification: ProgramQualificationReceipt
    null_contract: ProgramNullContract
    simulation_contract: ProgramSimulationContract | None = None
    model_state_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_bundle(self) -> BiologicalProgramQualificationBundle:
        contract_id = self.contract.program_contract_id
        contract_ids = (
            *(item.program_contract_id for item in self.program_definitions),
            *(item.program_contract_id for item in self.perturbation_effects),
            *(item.program_contract_id for item in self.gene_effects),
            self.guide_target_consistency.program_contract_id,
            self.uncertainty.program_contract_id,
            self.qualification.program_contract_id,
        )
        if any(item != contract_id for item in contract_ids):
            raise ValueError("Program bundle contains a cross-wired contract reference.")
        if (
            self.qualification.qualification_protocol_id
            != self.qualification_protocol.qualification_protocol_id
        ):
            raise ValueError("Program bundle contains a cross-wired qualification protocol.")
        if self.qualification_protocol.null_replicates != self.null_contract.replicate_count:
            raise ValueError("Qualification and null replicate counts differ.")
        scopes = (
            *(item.scope for item in self.perturbation_effects),
            *(item.scope for item in self.gene_effects),
            self.guide_target_consistency.scope,
            self.uncertainty.scope,
            self.qualification.scope,
        )
        if any(item != self.scope for item in scopes):
            raise ValueError("Program bundle result scopes differ.")
        definition_ids = {item.program_definition_id for item in self.program_definitions}
        if not set(self.qualification.qualified_program_definition_ids) <= definition_ids:
            raise ValueError("Qualification references an absent program definition.")
        effect_ids = {item.program_effect_id for item in self.perturbation_effects}
        if any(item.program_effect_id not in effect_ids for item in self.gene_effects):
            raise ValueError("Gene-level effect references an absent program effect.")
        expected = self.identity(id_field="program_bundle_id")
        if self.program_bundle_id != expected:
            raise ValueError(f"program_bundle_id mismatch: expected {expected}.")
        return self
