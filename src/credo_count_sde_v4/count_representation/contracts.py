"""New representation lineage; no existing checkpoint or baseline schema changes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ..canonical import contract_id
from ..contracts.models import ArtifactRef, GitCommit, Sha256, StrictModel
from ..data.prepared_shards import PreparedAccess
from ..forecast.contracts import PROCESS_ENV_KEYS, SourceRole

PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class RNAFeature(StrictModel):
    feature_id: str = Field(min_length=1)
    is_RNA: bool = Field(strict=True)


class RepresentationRules(StrictModel):
    latent_dim: PositiveInt = 48
    hidden_dims: tuple[PositiveInt, PositiveInt] = (512, 128)
    factor_rank: PositiveInt = 8
    candidate_epochs: tuple[PositiveInt, ...] = (1, 2, 4, 8)
    audit_fraction: float = Field(default=0.05, gt=0, lt=0.5, allow_inf_nan=False)
    batch_rows: int = Field(default=256, gt=0, le=8192, strict=True)
    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    gradient_clip: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    minimum_count_improvement: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    scale_floor_fraction: float = Field(default=0.01, gt=0, le=1, allow_inf_nan=False)
    batch_correction: Literal["identity_unqualified"] = "identity_unqualified"
    observation: Literal["library_conditioned_multinomial"] = "library_conditioned_multinomial"
    thinning_probability: float = Field(default=0.5, strict=True)
    seed: int = Field(default=0, ge=0, lt=2**32, strict=True)
    device: Literal["cpu", "cuda"] = "cpu"
    maximum_dense_working_bytes: PositiveInt = 512 * 1024**2
    maximum_partition_rows: PositiveInt = 30_000_000
    maximum_latent_index_bytes: PositiveInt = 512 * 1024**2

    @model_validator(mode="after")
    def ordered_exposures(self) -> RepresentationRules:
        if self.thinning_probability != 0.5:
            raise ValueError("The initial symmetric molecule split is fixed at one half.")
        if (
            not self.candidate_epochs
            or tuple(sorted(set(self.candidate_epochs))) != self.candidate_epochs
        ):
            raise ValueError("Prespecify unique increasing full-epoch exposures.")
        return self


class CountRepresentationSpec(StrictModel):
    schema_id: Literal["credo.fold_count_representation_spec"] = (
        "credo.fold_count_representation_spec"
    )
    schema_version: Literal[1] = 1
    base_git_commit: GitCommit
    implementation_sha256: Sha256
    environment: dict[str, Any]
    process_environment: dict[str, str | None]
    fitting: PreparedAccess
    query: PreparedAccess
    source_roles: tuple[SourceRole, ...]
    fitting_donors: tuple[str, ...]
    query_donor: str
    protected_donors: tuple[str, ...]
    source_condition: str = Field(min_length=1)
    destination_condition: str = Field(min_length=1)
    features: tuple[RNAFeature, ...]
    rules: RepresentationRules = RepresentationRules()
    context_enabled: Literal[False] = False
    protected_endpoint_access: Literal[False] = False

    @model_validator(mode="after")
    def information_boundaries(self) -> CountRepresentationSpec:
        if self.source_condition == self.destination_condition:
            raise ValueError("Source and destination conditions must differ.")
        if self.fitting.role != "representation_fit" or self.query.role != "query":
            raise ValueError("Exact representation-fitting and query capabilities required.")
        if set(self.process_environment) != set(PROCESS_ENV_KEYS):
            raise ValueError("Freeze every allowlisted numerical process setting.")
        if (
            not self.fitting_donors
            or len(set(self.fitting_donors)) != len(self.fitting_donors)
            or set(self.fitting_donors) & {self.query_donor, *self.protected_donors}
            or self.query_donor in self.protected_donors
        ):
            raise ValueError("Fitting, query and protected donor roles must be disjoint.")
        for key in (
            "package_completion_sha256",
            "package_inventory_sha256",
            "amendment_sha256",
            "feature_order_sha256",
            "guide_catalog",
            "n_features",
            "task_id",
        ):
            if getattr(self.fitting, key) != getattr(self.query, key):
                raise ValueError("Fitting/query package authorities differ.")
        feature_rows = [f.model_dump() for f in self.features]
        if (
            len(self.features) != self.fitting.n_features
            or len({f.feature_id for f in self.features}) != len(self.features)
            or not self.rna_positions
            or contract_id(feature_rows) != self.fitting.feature_order_sha256
        ):
            raise ValueError("Complete canonical feature/RNA mask differs from authority.")
        roles = {r.source_id: r for r in self.source_roles}
        if len(roles) != len(self.source_roles) or set(roles) != set(self.fitting.source_ids) | set(
            self.query.source_ids
        ):
            raise ValueError("Source roles must exactly cover fitting/query sources.")
        expected = {(d, c) for d in self.fitting_donors for c in ("source", "destination")}
        actual = {(roles[s].donor_id, roles[s].condition_role) for s in self.fitting.source_ids}
        if actual != expected or len(self.fitting.source_ids) != len(expected):
            raise ValueError("Only exact fitting donor/source-destination pairs are permitted.")
        if len(self.query.source_ids) != 1 or any(
            roles[s].donor_id != self.query_donor or roles[s].condition_role != "source"
            for s in self.query.source_ids
        ):
            raise ValueError("Query must be the declared source, never its endpoint.")
        fit_files = {a.relative_uri for s in self.fitting.shards for a in (s.counts, s.cells)}
        if fit_files & {a.relative_uri for s in self.query.shards for a in (s.counts, s.cells)}:
            raise ValueError("Fitting/query artifact alias is forbidden.")
        return self

    @property
    def rna_positions(self) -> tuple[int, ...]:
        return tuple(i for i, feature in enumerate(self.features) if feature.is_RNA)


class RepresentationManifest(StrictModel):
    schema_id: Literal["credo.fold_count_representation_bundle"] = (
        "credo.fold_count_representation_bundle"
    )
    schema_version: Literal[1] = 1
    stage: Literal["calibration", "fitted_representation", "latent_cells"]
    specification_sha256: Sha256
    parents: dict[str, Sha256]
    artifacts: tuple[ArtifactRef, ...]
    facts: dict[str, Any]

    @model_validator(mode="after")
    def unique_payloads(self) -> RepresentationManifest:
        paths = [a.relative_uri for a in self.artifacts]
        if len(paths) != len(set(paths)) or "COMPLETE.json" in paths:
            raise ValueError("Unique payloads excluding the completion marker required.")
        return self
