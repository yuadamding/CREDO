"""Deterministic non-biological fixture used by tests and smoke runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import sparse

from .canonical import canonical_json_bytes, sha256_bytes
from .contracts import (
    ArtifactRef,
    BaselineInformationSet,
    BaselineRegistry,
    CandidateSelectionPlan,
    DenominatorBlock,
    DenominatorContract,
    EffectHierarchyContract,
    EffectHierarchyRow,
    EligibilityManifest,
    EvidenceRole,
    ExposureRecord,
    ExposureRegistry,
    FeatureKey,
    InformationSet,
    MultiplicityPlan,
    PoolContract,
    PoolContributor,
    PreregistrationRef,
    RunIntent,
    SemanticStudySnapshot,
    SeriesRecord,
    SplitContract,
    TopologySupport,
    TransportTopologyContract,
)
from .store import build_count_store


def create_synthetic_project(
    output: Path,
    *,
    intent: RunIntent = RunIntent.COUNT_CONTEXT,
    seed: int = 7,
    updates: int = 40,
    pooled: bool = False,
) -> Path:
    """Create a small known-signal two-pool count-SDE fixture."""

    if output.exists():
        raise FileExistsError(output)
    input_root = output / "work" / "input"
    input_root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    features = tuple(
        FeatureKey(
            namespace="synthetic_feature", feature_id=f"f{index:02d}", namespace_version="v1"
        )
        for index in range(12)
    )
    target_indices = [0, 1, 2, 0, 1, 2]
    pool_indices = [0, 0, 0, 0, 0, 0] if pooled else [0, 0, 0, 1, 1, 1]
    pool_count = 1 if pooled else 2
    source_rows: list[int] = []
    terminal_rows: list[int] = []
    matrices: list[np.ndarray[Any, Any]] = []
    series_records: list[SeriesRecord] = []
    row_cursor = 10_000
    source_abundance = [60, 60, 60, 50, 50, 50]
    terminal_abundance = [58, 92, 36, 51, 72, 27]
    base = np.asarray([2, 4, 3, 5, 2, 4, 3, 2, 5, 3, 4, 2], dtype=float)
    shifts = {
        0: np.zeros(12),
        1: np.asarray([4, 3, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0], dtype=float),
        2: np.asarray([0, 0, 0, 3, 4, 2, 0, 0, 1, 2, 0, 0], dtype=float),
    }
    for series_index, (target, pool) in enumerate(zip(target_indices, pool_indices, strict=True)):
        source_ids = tuple(range(row_cursor, row_cursor + 16))
        row_cursor += 16
        terminal_ids = tuple(range(row_cursor, row_cursor + 16))
        row_cursor += 16
        source_rows.extend(source_ids)
        terminal_rows.extend(terminal_ids)
        source_matrix = rng.poisson(base + 1.0, size=(16, len(features)))
        pool_shift = 0.5 * pool
        terminal_matrix = rng.poisson(
            base + shifts[target] + pool_shift + 1.0, size=(16, len(features))
        )
        matrices.extend([source_matrix, terminal_matrix])
        series_records.append(
            SeriesRecord(
                series_id=f"series-{series_index}",
                target_index=target,
                pool_index=pool,
                is_control=target == 0,
                source_rows=source_ids,
                terminal_rows=terminal_ids,
                source_count=source_abundance[series_index],
                terminal_count=terminal_abundance[series_index],
                duration=1.0,
            )
        )
    row_ids = np.asarray(source_rows + terminal_rows, dtype=np.int64)
    # Matrices were appended as source/terminal pairs. Reorder them to match
    # the global row order used above (all source rows, then all terminal rows).
    pair_arrays = [np.asarray(matrix) for matrix in matrices]
    source_arrays = pair_arrays[0::2]
    terminal_arrays = pair_arrays[1::2]
    matrix = sparse.csr_matrix(np.vstack(source_arrays + terminal_arrays).astype(np.int32))
    store_path = input_root / "counts.h5"
    store_manifest = build_count_store(store_path, matrix, row_ids=row_ids, features=features)
    (input_root / "count-store.json").write_bytes(
        canonical_json_bytes(store_manifest.model_dump(mode="json")) + b"\n"
    )
    feature_payload = [feature.model_dump(mode="json") for feature in features]
    (input_root / "features.json").write_bytes(canonical_json_bytes(feature_payload) + b"\n")
    information = InformationSet(
        information_set_id="synthetic-source-only",
        fit_rows=tuple(source_rows),
        query_rows=tuple(terminal_rows),
        protected_rows=tuple(terminal_rows),
    )
    (input_root / "information-set.json").write_bytes(
        canonical_json_bytes(information.model_dump(mode="json")) + b"\n"
    )
    information_hash = sha256_bytes((input_root / "information-set.json").read_bytes())
    exposure = ExposureRegistry(
        records=tuple(
            ExposureRecord(
                unit_type="series",
                unit_id=record.series_id,
                endpoint_seen=False,
                exposure_role=EvidenceRole.DEVELOPMENT,
            )
            for record in series_records
        )
    )
    row_hash = sha256_bytes(np.asarray(row_ids, dtype="<i8").tobytes())
    feature_hash = sha256_bytes(canonical_json_bytes(feature_payload))
    snapshot = SemanticStudySnapshot(
        study_id="synthetic-pooled-v1" if pooled else "synthetic-two-pool-v1",
        series=tuple(series_records),
        observed_edges=(("source", "terminal"),),
        feature_index_hash=feature_hash,
        row_universe_hash=row_hash,
        exposure_registry=exposure,
    )
    (input_root / "snapshot.json").write_bytes(
        canonical_json_bytes(snapshot.model_dump(mode="json")) + b"\n"
    )
    split = SplitContract(
        split_id="synthetic-source-endpoint-split-v1",
        training_units=("source",),
        outer_evaluation_units=("terminal",),
        grouping_unit="observation_role",
    )
    eligibility = EligibilityManifest(
        manifest_id="synthetic-source-eligible-v1",
        abundance_eligible_units=tuple(record.series_id for record in series_records),
        state_evaluable_units=tuple(record.series_id for record in series_records),
        source_information_set_hash=information_hash,
    )
    hierarchy = EffectHierarchyContract(
        hierarchy_id="synthetic-target-hierarchy-v1",
        rows=tuple(
            EffectHierarchyRow(
                perturbation_id=record.series_id,
                target_id=f"target-{record.target_index}",
                target_index=record.target_index,
                is_control=record.is_control,
            )
            for record in series_records
        ),
    )
    topology = TransportTopologyContract(
        topology_id="synthetic-single-partition-v1",
        partitions=("all",),
        allowed_edges=(("all", "all"),),
        support=tuple(
            TopologySupport(
                series_id=record.series_id,
                partition_id="all",
                source_count=len(record.source_rows),
                terminal_count=len(record.terminal_rows),
            )
            for record in series_records
        ),
    )
    denominator = DenominatorContract(
        denominator_id="synthetic-complete-pools-v1",
        blocks=tuple(
            DenominatorBlock(
                block_id=f"pool-{pool}",
                category_ids=tuple(
                    record.series_id for record in series_records if record.pool_index == pool
                ),
            )
            for pool in range(pool_count)
        ),
    )
    pools = PoolContract(
        pool_contract_id="synthetic-physical-pools-v1",
        physical_pool_ids=tuple(f"pool-{pool}" for pool in range(pool_count)),
        contributors=tuple(
            PoolContributor(pool_id=f"pool-{record.pool_index}", series_id=record.series_id)
            for record in series_records
        ),
    )
    protocol_path = input_root / "preregistered-protocol.json"
    protocol_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "purpose": "non-biological deterministic software validation",
                "outer_access": "one_shot",
            }
        )
        + b"\n"
    )
    preregistration = PreregistrationRef(
        preregistration_id="synthetic-protocol-v1",
        artifact=ArtifactRef(
            schema_id="credo.preregistered_protocol",
            schema_version=1,
            sha256=sha256_bytes(protocol_path.read_bytes()),
            size_bytes=protocol_path.stat().st_size,
            media_type="application/json",
            relative_uri="preregistered-protocol.json",
        ),
        frozen_before_evaluation=True,
    )
    multiplicity = MultiplicityPlan(
        plan_id="synthetic-bh-v1",
        families=("engineering_targets",),
        reporting_universe="all synthetic targets",
    )
    candidates = CandidateSelectionPlan(
        plan_id="synthetic-source-only-candidates-v1",
        information_set_hash=information_hash,
        score="source_support",
        direction="higher",
        tie_rule="series_id ascending",
        maximum_candidates=6,
        minimum_support=1,
    )
    baseline = BaselineInformationSet(
        baseline_id="persistence-v1",
        code_hash=sha256_bytes(b"synthetic-persistence-v1"),
        allowed_training_rows_hash=sha256_bytes(
            np.asarray(information.fit_rows, dtype="<i8").tobytes()
        ),
        allowed_test_source_rows_hash=sha256_bytes(
            np.asarray(information.fit_rows, dtype="<i8").tobytes()
        ),
        forbidden_endpoint_rows_hash=sha256_bytes(
            np.asarray(information.protected_rows, dtype="<i8").tobytes()
        ),
        split_hash=sha256_bytes(canonical_json_bytes(split.model_dump(mode="json"))),
        representation_id="resolved-during-compile",
        aggregation_hash=sha256_bytes(b"series-mean-v1"),
        seed_plan_hash=sha256_bytes(b"deterministic-zero-v1"),
    )
    baseline_registry = BaselineRegistry(
        registry_id="synthetic-baselines-v1", baselines=(baseline,)
    )

    def write_contract(name: str, value: object) -> None:
        assert hasattr(value, "model_dump")
        (input_root / name).write_bytes(canonical_json_bytes(value.model_dump(mode="json")) + b"\n")

    write_contract("split.json", split)
    write_contract("eligibility.json", eligibility)
    write_contract("effect-hierarchy.json", hierarchy)
    write_contract("topology.json", topology)
    write_contract("denominator.json", denominator)
    write_contract("pools.json", pools)
    write_contract("preregistration.json", preregistration)
    write_contract("multiplicity.json", multiplicity)
    write_contract("candidate-selection.json", candidates)
    write_contract("baseline-registry.json", baseline_registry)
    (input_root / "source-manifest.json").write_bytes(
        canonical_json_bytes(
            {"schema_version": 1, "source": "generated_synthetic_fixture", "seed": seed}
        )
        + b"\n"
    )
    (input_root / "feature-permutation.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "policy": "identity",
                "ordered_feature_hash": snapshot.feature_index_hash,
            }
        )
        + b"\n"
    )
    config = {
        "schema_version": 1,
        "workspace": "work",
        "semantic_snapshot": "work/input/snapshot.json",
        "count_store": "work/input/counts.h5",
        "information_set": "work/input/information-set.json",
        "split_contract": "work/input/split.json",
        "eligibility_manifest": "work/input/eligibility.json",
        "effect_hierarchy": "work/input/effect-hierarchy.json",
        "topology_contract": "work/input/topology.json",
        "denominator_contract": (
            None if intent is RunIntent.COUNT_STATE else "work/input/denominator.json"
        ),
        "pool_contract": "work/input/pools.json" if intent is RunIntent.COUNT_CONTEXT else None,
        "preregistration": "work/input/preregistration.json",
        "multiplicity_plan": "work/input/multiplicity.json",
        "candidate_selection_plan": "work/input/candidate-selection.json",
        "state_selection_calibration": None,
        "baseline_registry": "work/input/baseline-registry.json",
        "intent": intent.value,
        "model": {
            "state_dim": 4,
            "target_count": 3,
            "pool_count": pool_count,
            "hidden_dim": 16,
            "shared_diffusion": False,
            "shared_diffusion_inner_validation_pass": False,
            "centered_selection": False,
            "selection_inner_validation_pass": False,
            "source_efficacy_sensitivity": False,
            "context_rank": 2,
        },
        "training": {
            "max_updates": updates,
            "learning_rate": 0.03,
            "state_batch_size": 6,
            "checkpoint_every": max(1, updates // 2),
            "seed": seed,
            "dtype": "float32",
            "deterministic": True,
            "selected_update": updates,
        },
        "evaluation": {
            "particles": 64,
            "steps": 8,
            "seed": seed + 10_000,
            "output_bytes_limit": 100_000_000,
        },
        "feature_index": "work/input/features.json",
        "source_manifest": "work/input/source-manifest.json",
        "feature_permutation": "work/input/feature-permutation.json",
        "input_view": "identity_library_normalized",
        "correction_metadata": None,
        "source_smoothing": 0.5,
    }
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return output / "config.yaml"
