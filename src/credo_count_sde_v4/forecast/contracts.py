"""Versioned execution and publication contracts for population-mean forecasts."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field, model_validator

from ..canonical import contract_id
from ..contracts.models import ArtifactRef, GitCommit, Sha256, StrictModel
from ..data.prepared_shards import PreparedAccess

FAMILIES = (
    "source_persistence",
    "fitting_control_response",
    "guide_endpoint_transfer",
    "target_endpoint_transfer",
    "hierarchical_source_response",
)

PROCESS_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMPY_MADVISE_HUGEPAGE",
)


def process_environment() -> dict[str, str | None]:
    """Only explicitly allowlisted numerical/device knobs, never arbitrary secrets."""
    return {key: os.environ.get(key) for key in PROCESS_ENV_KEYS}


class SourceRole(StrictModel):
    source_id: str
    donor_id: str
    condition_role: Literal["source", "destination"]


class BaselineRules(StrictModel):
    schema_version: Literal[1] = 1
    families: tuple[str, ...] = FAMILIES
    normalization: Literal["equal_cell_RNA_composition"] = "equal_cell_RNA_composition"
    exclude_technical_features: Literal[True] = True
    expression_source_minimum_cells: Literal[1] = 1
    expression_pseudocount: float = Field(default=1e-8, gt=0)
    informative_log_effect: float = Field(default=0.05, ge=0)
    shrinkage_cells: float = Field(default=32.0, gt=0)
    abundance_pseudocount: float = Field(default=0.5, gt=0)
    abundance_denominator: Literal["complete_bound_guide_catalog"] = "complete_bound_guide_catalog"
    expression_fit_aggregation: Literal["equal_available_donor_then_equal_guide_target"] = (
        "equal_available_donor_then_equal_guide_target"
    )
    unknown_source_expression: Literal["abstain"] = "abstain"
    unknown_source_abundance: Literal["mass_only_source_prior"] = "mass_only_source_prior"
    hierarchical_unknown_pair: Literal["target_then_fitting_control_response"] = (
        "target_then_fitting_control_response"
    )

    @model_validator(mode="after")
    def fixed_families(self) -> BaselineRules:
        if self.abundance_pseudocount != 0.5:
            raise ValueError("The complete-catalog Jeffreys pseudocount is fixed at 0.5.")
        if self.families != FAMILIES:
            raise ValueError("The baseline family order is fixed before evaluation.")
        return self


class ForecastSpec(StrictModel):
    schema_id: Literal["credo.paired_condition_baseline_spec"] = (
        "credo.paired_condition_baseline_spec"
    )
    schema_version: Literal[2] = 2
    base_git_commit: GitCommit
    implementation_sha256: Sha256
    external_code_sha256: dict[str, Sha256]
    environment: dict[str, Any]
    process_environment: dict[str, str | None]
    task_id: str
    source_condition: str
    destination_condition: str
    fitting_donors: tuple[str, ...]
    query_donor: str
    protected_donors: tuple[str, ...]
    source_roles: tuple[SourceRole, ...]
    fitting: PreparedAccess
    query: PreparedAccess
    # Opaque endpoint-view authority only: no endpoint observations in this object.
    evaluation_view: ArtifactRef
    ordered_rna_features: tuple[str, ...]
    rna_positions: tuple[int, ...]
    rules: BaselineRules = BaselineRules()
    seed: Literal[0] = 0
    context_enabled: Literal[False] = False
    expression_output: Literal["population_mean_not_cell_matched_or_full_law"] = (
        "population_mean_not_cell_matched_or_full_law"
    )
    storage_isolation_qualified: Literal[False] = False
    worker_count: int = Field(default=4, ge=1, le=10, strict=True)
    batch_rows: int = Field(default=2048, ge=1, le=8192, strict=True)
    recovery_unit: Literal["complete_source_replay_incomplete_source"] = (
        "complete_source_replay_incomplete_source"
    )
    maximum_process_tree_rss_bytes: int = Field(default=64 * 1024**3, gt=0)
    maximum_output_bytes: int = Field(default=128 * 1024**3, gt=0)

    @model_validator(mode="after")
    def roles_and_orders(self) -> ForecastSpec:
        if set(self.process_environment) != set(PROCESS_ENV_KEYS):
            raise ValueError("Every allowlisted numerical process setting must be frozen.")
        if self.source_condition == self.destination_condition:
            raise ValueError("Source and destination conditions must differ.")
        if self.fitting.role != "baseline_fit" or self.query.role != "query":
            raise ValueError("Predictors require fitting/query capabilities, never endpoint truth.")
        if not self.fitting_donors or len(set(self.fitting_donors)) != len(self.fitting_donors):
            raise ValueError("Distinct fitting donors required.")
        if (
            set(self.fitting_donors) & {self.query_donor, *self.protected_donors}
            or self.query_donor in self.protected_donors
        ):
            raise ValueError("Donor roles overlap.")
        if self.fitting.task_id != self.task_id or self.query.task_id != self.task_id:
            raise ValueError("Prepared task identities disagree.")
        if (
            self.fitting.guide_catalog != self.query.guide_catalog
            or self.fitting.feature_order_sha256 != self.query.feature_order_sha256
        ):
            raise ValueError("Prepared feature/guide authorities disagree.")
        for key in (
            "package_completion_sha256",
            "package_inventory_sha256",
            "amendment_sha256",
            "n_features",
        ):
            if getattr(self.fitting, key) != getattr(self.query, key):
                raise ValueError("Fitting/query package or amendment identities disagree.")
        if (
            len(self.ordered_rna_features) != len(self.rna_positions)
            or not self.rna_positions
            or len(set(self.ordered_rna_features)) != len(self.ordered_rna_features)
        ):
            raise ValueError("RNA feature identities must be complete, unique and ordered.")
        if (
            tuple(sorted(set(self.rna_positions))) != self.rna_positions
            or min(self.rna_positions) < 0
            or max(self.rna_positions) >= self.fitting.n_features
        ):
            raise ValueError("Invalid canonical RNA positions.")
        roles = {row.source_id: row for row in self.source_roles}
        if len(roles) != len(self.source_roles) or set(roles) != set(self.fitting.source_ids) | set(
            self.query.source_ids
        ):
            raise ValueError("Source roles must exactly cover permitted input views.")
        actual = {(roles[s].donor_id, roles[s].condition_role) for s in self.fitting.source_ids}
        expected = {(d, c) for d in self.fitting_donors for c in ("source", "destination")}
        if actual != expected or len(self.fitting.source_ids) != len(expected):
            raise ValueError("Fitting sources must form exact donor/condition pairs.")
        if len(self.query.source_ids) != 1 or any(
            roles[s].donor_id != self.query_donor or roles[s].condition_role != "source"
            for s in self.query.source_ids
        ):
            raise ValueError("Query access must contain only the declared donor source.")
        return self

    @property
    def guide_catalog_sha256(self) -> str:
        return self.fitting.guide_catalog.ordered_catalog_sha256

    @property
    def rna_order_sha256(self) -> str:
        return contract_id(list(zip(self.rna_positions, self.ordered_rna_features, strict=True)))


class BundleManifest(StrictModel):
    schema_id: Literal["credo.population_baseline_bundle"] = "credo.population_baseline_bundle"
    schema_version: Literal[1] = 1
    stage: Literal[
        "shard_summary", "source_summary", "fitted_baselines", "prediction", "evaluation"
    ]
    specification_sha256: Sha256
    parents: dict[str, Sha256]
    artifacts: tuple[ArtifactRef, ...]
    facts: dict[str, Any]

    @model_validator(mode="after")
    def unique_files(self) -> BundleManifest:
        paths = [a.relative_uri for a in self.artifacts]
        if len(set(paths)) != len(paths) or "COMPLETE.json" in paths:
            raise ValueError("Bundle inventory must be unique and cannot include its own marker.")
        return self
