from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import sparse

from credo_count_sde_v4.canonical import contract_id, sha256_file
from credo_count_sde_v4.contracts.models import ArtifactRef
from credo_count_sde_v4.data.prepared_shards import (
    GuideCatalogBinding,
    PreparedAccess,
    PreparedEvaluationAccess,
    PreparedShard,
)
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.forecast.aggregation import aggregate_view
from credo_count_sde_v4.forecast.artifacts import publish_bundle, verify_bundle
from credo_count_sde_v4.forecast.baselines import fit_baselines, predict_baselines
from credo_count_sde_v4.forecast.contracts import (
    FAMILIES,
    BaselineRules,
    ForecastSpec,
    SourceRole,
    process_environment,
)
from credo_count_sde_v4.forecast.evaluation import evaluate_baselines, prediction_barrier
from credo_count_sde_v4.runtime_identity import environment_identity, implementation_tree_hash


def artifact(root, name):
    return ArtifactRef(
        schema_id="synthetic",
        schema_version=1,
        sha256=sha256_file(root / name),
        size_bytes=(root / name).stat().st_size,
        media_type="application/octet-stream",
        relative_uri=name,
    )


def make_fixture(root: Path, *, endpoint_variant=False, null=False, workers=1):
    root.mkdir()
    catalog = pd.DataFrame(
        dict(
            guide_index=list(range(6)),
            guide_id=["control", "up", "down", "source-absent", "fit-absent", "zero-rna"],
            target_id=["control", "A", "B", "A", "C", "D"],
            is_control=[True, False, False, False, False, False],
        )
    )
    catalog.to_parquet(root / "catalog.parquet", index=False)
    records = {}
    roles = []
    for donor in ("fit-a", "fit-b", "query"):
        for condition in ("source", "destination"):
            sid = f"{donor}-{condition}"
            if donor != "query" or condition == "source":
                roles.append(SourceRole(source_id=sid, donor_id=donor, condition_role=condition))
            guides = (
                [0, 1, 2, 3, 5]
                if donor != "query"
                else [0, 1, 2, 4, 5]
                if condition == "source"
                else [0, 1, 1, 2, 3, 4, 5]
            )
            if endpoint_variant and sid == "query-destination":
                guides = [0, 0, 0, 1, 2, 3, 4, 5, 5]
            values = []
            for guide in guides:
                rna = (
                    [5, 5]
                    if null or condition == "source" or guide in (0, 4, 5)
                    else [9, 1]
                    if guide in (1, 3)
                    else [1, 9]
                )
                if endpoint_variant and sid == "query-destination":
                    rna = [2, 8] if guide == 0 else list(reversed(rna))
                if guide == 5 and donor == "query":
                    rna = [0, 0]
                values.append([17 + guide, *rna])
            sid_records = []
            # Two shards exercise complete source accumulation and metadata checks.
            for number, rows in enumerate(np.array_split(np.arange(len(guides)), 2)):
                array = np.asarray(values, dtype=np.uint32)[rows]
                matrix = sparse.csr_matrix(array)
                counts_name, cells_name = f"{sid}-{number}.npz", f"{sid}-{number}.parquet"
                sparse.save_npz(root / counts_name, matrix)
                chosen = np.asarray(guides)[rows]
                pd.DataFrame(
                    dict(
                        source_id=[sid] * len(rows),
                        row_in_shard=np.arange(len(rows)),
                        cell_id=[f"{sid}:{i}" for i in rows],
                        guide_index=chosen,
                        guide_id=catalog.guide_id.iloc[chosen].to_numpy(),
                        primary_target_id=catalog.target_id.iloc[chosen].to_numpy(),
                        is_control=catalog.is_control.iloc[chosen].to_numpy(),
                        RNA_UMIs=array[:, 1:].sum(1),
                        all_feature_UMIs=array.sum(1),
                        technical_PuroR_UMIs=array[:, 0],
                    )
                ).to_parquet(root / cells_name, index=False)
                sid_records.append(
                    PreparedShard(
                        source_id=sid,
                        shard=number,
                        rows=len(rows),
                        nnz=matrix.nnz,
                        counts=artifact(root, counts_name),
                        cells=artifact(root, cells_name),
                    )
                )
            records[sid] = sid_records
    (root / "endpoint-view.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in records["query-destination"]])
    )
    common = dict(
        package_completion_sha256=contract_id({"variant": endpoint_variant, "null": null}),
        package_inventory_sha256="b" * 64,
        amendment_sha256="c" * 64,
        feature_order_sha256=contract_id(["technical", "A", "B"]),
        guide_catalog=GuideCatalogBinding(
            artifact=artifact(root, "catalog.parquet"),
            ordered_catalog_sha256=contract_id(catalog.to_dict("records")),
        ),
        n_features=3,
        task_id="paired",
    )
    fit_ids = tuple(s for s in records if s.startswith("fit-"))
    fitting = PreparedAccess(
        **common,
        parent_view_sha256="d" * 64,
        role="baseline_fit",
        source_ids=fit_ids,
        shards=tuple(s for sid in fit_ids for s in records[sid]),
    )
    query = PreparedAccess(
        **common,
        parent_view_sha256="e" * 64,
        role="query",
        source_ids=("query-source",),
        shards=tuple(records["query-source"]),
    )
    spec = ForecastSpec(
        base_git_commit="f" * 40,
        implementation_sha256=implementation_tree_hash(),
        external_code_sha256={},
        environment=environment_identity(),
        process_environment=process_environment(),
        task_id="paired",
        source_condition="rest",
        destination_condition="endpoint",
        fitting_donors=("fit-a", "fit-b"),
        query_donor="query",
        protected_donors=("protected",),
        source_roles=tuple(roles),
        fitting=fitting,
        query=query,
        evaluation_view=artifact(root, "endpoint-view.json"),
        ordered_rna_features=("A", "B"),
        rna_positions=(1, 2),
        worker_count=workers,
        batch_rows=2,
    )
    return root, spec, catalog, records["query-destination"]


def publish(fixture, output, *, recovery=False):
    root, spec, catalog, endpoint_records = fixture
    if recovery:
        with pytest.raises(InterruptedError):
            aggregate_view(
                root, spec.fitting, spec, output / "fit-summary", stop_after_new_sources=1
            )
    events = []
    summaries = aggregate_view(
        root, spec.fitting, spec, output / "fit-summary", progress=events.append
    )
    if recovery:
        assert sum(e["reused"] for e in events) == 1
    fit_baselines(spec, catalog, summaries, output / "fit")
    queries = aggregate_view(root, spec.query, spec, output / "query-summary")
    predict_baselines(output / "fit", queries, output / "prediction")
    seal, _ = prediction_barrier(output / "prediction")
    access = PreparedEvaluationAccess.model_validate(
        {
            **spec.query.model_dump(
                exclude={"role", "schema_version", "source_ids", "shards", "parent_view_sha256"}
            ),
            "source_ids": ["query-destination"],
            "shards": endpoint_records,
            "parent_view_sha256": spec.evaluation_view.sha256,
            "prediction_seal_sha256": seal.identity(),
        }
    )
    return access


def evaluate(fixture, output, access):
    root, spec, _, _ = fixture
    sources = aggregate_view(
        root, access, spec, output / "endpoint-summary", prediction=output / "prediction"
    )
    evaluate_baselines(
        output / "prediction", access, sources["query-destination"], output / "evaluation"
    )
    return json.loads((output / "evaluation/metrics.json").read_text())


def test_real_interfaces_canary_mean_effect_mass_missing_and_recovery(tmp_path):
    fixture = make_fixture(tmp_path / "input")
    first = tmp_path / "first"
    access = publish(fixture, first, recovery=True)
    before = verify_bundle(first / "prediction").identity()
    metrics = evaluate(fixture, first, access)
    assert verify_bundle(first / "prediction").identity() == before
    perfect = metrics["guide_endpoint_transfer"]
    assert perfect["targeting"]["macro_target"]["composition_mse"] < 1e-12
    assert perfect["targeting"]["macro_target"]["effect_sign_accuracy"] == 1
    assert perfect["abundance"]["guides"] == 6
    assert perfect["abundance"]["observed_cells"] == 7
    coverage = pd.read_parquet(first / "prediction/coverage.parquet")
    assert set(coverage.loc[coverage.guide_index.eq(3), "expression_status"]) == {"source_absent"}
    assert set(coverage.loc[coverage.guide_index.eq(5), "expression_status"]) == {"zero_RNA_source"}
    assert coverage[
        coverage.family.eq("hierarchical_source_response") & coverage.guide_index.eq(4)
    ].expression_available.all()
    mass = pd.read_parquet(first / "evaluation/abundance_by_guide.parquet")
    assert (mass.prediction_frequency > 0).all()
    assert np.allclose(mass.groupby("family").prediction_frequency.sum(), 1)
    expression = pd.read_parquet(first / "evaluation/expression_by_guide.parquet")
    assert not expression[expression.guide_index.eq(3)].scored.any()
    assert len(expression) == 6 * len(FAMILIES)
    assert (
        metrics["source_persistence"]["targeting_common_support"]["macro_target"]["composition_mse"]
        > perfect["targeting_common_support"]["macro_target"]["composition_mse"]
    )
    uninterrupted = tmp_path / "uninterrupted"
    other_access = publish(fixture, uninterrupted)
    other_metrics = evaluate(fixture, uninterrupted, other_access)
    assert metrics == other_metrics
    for stage in ("fit", "prediction"):
        assert (
            verify_bundle(first / stage).facts["numerical_sha256"]
            == verify_bundle(uninterrupted / stage).facts["numerical_sha256"]
        )


def test_endpoint_control_expression_and_abundance_changes_cannot_change_predictions(tmp_path):
    original = make_fixture(tmp_path / "original-input")
    changed = make_fixture(tmp_path / "changed-input", endpoint_variant=True)
    first, second = tmp_path / "first", tmp_path / "second"
    a = publish(original, first)
    b = publish(changed, second)
    assert original[1].identity() != changed[1].identity()
    for stage in ("fit", "prediction"):
        assert (
            verify_bundle(first / stage).facts["numerical_sha256"]
            == verify_bundle(second / stage).facts["numerical_sha256"]
        )
    baseline = evaluate(original, first, a)
    altered = evaluate(changed, second, b)
    assert (
        altered["guide_endpoint_transfer"]["targeting"]["macro_target"]["composition_mse"]
        > baseline["guide_endpoint_transfer"]["targeting"]["macro_target"]["composition_mse"]
    )
    assert altered["guide_endpoint_transfer"]["abundance"]["observed_cells"] == 9


def test_null_and_worker_parallelism(tmp_path):
    fixture = make_fixture(tmp_path / "input", null=True, workers=2)
    out = tmp_path / "run"
    access = publish(fixture, out)
    metrics = evaluate(fixture, out, access)
    for family in FAMILIES:
        assert metrics[family]["targeting"]["macro_target"]["composition_mse"] < 1e-12
        assert metrics[family]["targeting"]["macro_target"]["effect_sign_accuracy"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"query_donor": "fit-a"},
        {"source_condition": "endpoint"},
        {"fitting_donors": ()},
        {"rna_positions": (2, 1)},
        {"ordered_rna_features": ("A", "A")},
        {"source_roles": ()},
        {"seed": 1},
        {"context_enabled": True},
    ],
)
def test_invalid_execution_specifications_rejected(tmp_path, change):
    _, spec, _, _ = make_fixture(tmp_path / "input")
    with pytest.raises(ValidationError):
        ForecastSpec.model_validate({**spec.model_dump(), **change})


def test_fixed_baseline_rules_and_runtime_reuse(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path / "input")
    root, spec, _, _ = fixture
    legacy = spec.model_dump()
    legacy["schema_version"] = 1
    legacy.pop("process_environment")
    with pytest.raises(ValidationError):
        ForecastSpec.model_validate(legacy)
    for change in ({"families": ("unknown",)}, {"abundance_pseudocount": 1.0}):
        with pytest.raises(ValidationError):
            BaselineRules.model_validate(change)
    wrong = ForecastSpec.model_validate({**spec.model_dump(), "implementation_sha256": "0" * 64})
    with pytest.raises(ContractError, match="runtime"):
        aggregate_view(root, wrong.fitting, wrong, tmp_path / "wrong")
    altered = spec.query.model_copy(update={"parent_view_sha256": "0" * 64})
    with pytest.raises(ContractError, match="outside"):
        aggregate_view(root, altered, spec, tmp_path / "outside")
    for key in ("maximum_process_tree_rss_bytes", "maximum_output_bytes"):
        limited = ForecastSpec.model_validate({**spec.model_dump(), key: 1})
        with pytest.raises(ContractError, match="budget"):
            aggregate_view(root, limited.fitting, limited, tmp_path / key)
    monkeypatch.setenv("NUMPY_MADVISE_HUGEPAGE", "changed")
    with pytest.raises(ContractError, match="runtime"):
        aggregate_view(root, spec.fitting, spec, tmp_path / "process-setting-change")


def test_endpoint_access_requires_valid_published_prediction(tmp_path):
    fixture = make_fixture(tmp_path / "input")
    root, spec, _, records = fixture
    with pytest.raises((ContractError, IntegrityError, FileNotFoundError)):
        prediction_barrier(tmp_path / "unpublished")
    out = tmp_path / "run"
    access = publish(fixture, out)
    with pytest.raises(ContractError, match="published prediction"):
        aggregate_view(root, access, spec, tmp_path / "forbidden")
    wrong = access.model_copy(update={"prediction_seal_sha256": "0" * 64})
    with pytest.raises(ContractError, match="bound"):
        aggregate_view(root, wrong, spec, tmp_path / "wrong", prediction=out / "prediction")
    with pytest.raises(FileExistsError):
        predict_baselines(
            out / "fit",
            {"query-source": next((out / "query-summary").iterdir())},
            out / "prediction",
        )
    with (out / "prediction/predictions.h5").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises((ContractError, IntegrityError)):
        aggregate_view(root, access, spec, tmp_path / "corrupt", prediction=out / "prediction")


def test_label_permutation_harms_keyed_score_without_cell_correspondence(tmp_path):
    fixture = make_fixture(tmp_path / "input")
    out = tmp_path / "run"
    access = publish(fixture, out)
    metrics = evaluate(fixture, out, access)
    expression = pd.read_parquet(out / "evaluation/expression_by_guide.parquet")
    assert set(expression.endpoint_cells) == {1, 2}
    # Publish an intentionally wrong but explicitly labeled new prediction generation.
    original = verify_bundle(out / "prediction")

    def writer(path):
        for item in original.artifacts:
            source = out / "prediction" / item.relative_uri
            (path / item.relative_uri).write_bytes(source.read_bytes())
        with h5py.File(path / "predictions.h5", "r+") as handle:
            for family in FAMILIES:
                array = handle[family]["mean_composition"]
                exchanged = array[[1, 2]][::-1]
                array[[1, 2]] = exchanged
        return original.facts

    shuffled = out / "shuffled"
    new = publish_bundle(
        shuffled,
        stage="prediction",
        specification=original.specification_sha256,
        parents=original.parents,
        writer=writer,
    )
    revised = access.model_copy(update={"prediction_seal_sha256": new.identity()})
    truth = aggregate_view(
        fixture[0], revised, fixture[1], out / "shuffled-truth", prediction=shuffled
    )
    evaluate_baselines(shuffled, revised, truth["query-destination"], out / "shuffled-evaluation")
    bad = json.loads((out / "shuffled-evaluation/metrics.json").read_text())
    assert (
        bad["guide_endpoint_transfer"]["targeting_common_support"]["macro_target"][
            "effect_sign_accuracy"
        ]
        == 0
    )
    assert (
        bad["guide_endpoint_transfer"]["targeting_common_support"]["macro_target"][
            "composition_mse"
        ]
        > metrics["guide_endpoint_transfer"]["targeting_common_support"]["macro_target"][
            "composition_mse"
        ]
    )


def test_role_catalog_order_and_rules_invalidate_artifact_reuse(tmp_path):
    fixture = make_fixture(tmp_path / "input")
    root, spec, catalog, _ = fixture
    summaries = aggregate_view(root, spec.fitting, spec, tmp_path / "summaries")
    with pytest.raises(ContractError, match="catalog identity"):
        fit_baselines(spec, catalog.iloc[::-1], summaries, tmp_path / "bad-catalog")
    changed = ForecastSpec.model_validate(
        {**spec.model_dump(), "rules": {**spec.rules.model_dump(), "shrinkage_cells": 64.0}}
    )
    with pytest.raises(ContractError, match="specification"):
        fit_baselines(changed, catalog, summaries, tmp_path / "bad-transform")
    fit_baselines(spec, catalog, summaries, tmp_path / "fit")
    with pytest.raises(ContractError, match="authorized role"):
        predict_baselines(tmp_path / "fit", summaries, tmp_path / "bad-query")
