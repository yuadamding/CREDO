from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from scipy import sparse

from credo_count_sde_v4.canonical import contract_id, sha256_file
from credo_count_sde_v4.contracts.models import ArtifactRef
from credo_count_sde_v4.count_representation.artifacts import verify
from credo_count_sde_v4.count_representation.contracts import (
    CountRepresentationSpec,
    RepresentationRules,
    RNAFeature,
)
from credo_count_sde_v4.count_representation.latent import (
    EmpiricalLatentStore,
    compile_latent_cells,
)
from credo_count_sde_v4.count_representation.network import count_loss, initialize, sparse_tensor
from credo_count_sde_v4.count_representation.stream import halves, iter_counts, partition_audit
from credo_count_sde_v4.count_representation.workflow import (
    calibrate_representation,
    load_representation,
    refit_representation,
)
from credo_count_sde_v4.data.prepared_shards import (
    GuideCatalogBinding,
    PreparedAccess,
    PreparedShard,
    PreparedShardReader,
)
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.forecast.contracts import SourceRole, process_environment
from credo_count_sde_v4.runtime_identity import environment_identity, implementation_tree_hash


def artifact(root, name):
    return ArtifactRef(
        schema_id="fixture",
        schema_version=1,
        relative_uri=name,
        sha256=sha256_file(root / name),
        size_bytes=(root / name).stat().st_size,
        media_type="application/octet-stream",
    )


def fixture(
    root: Path,
    *,
    null=False,
    epochs=(2, 8),
    zero=True,
    source_shift=False,
    depth_imbalance=False,
    rna_features=6,
    rows_per_shard=32,
    rules=None,
):
    root.mkdir()
    catalog = pd.DataFrame(
        dict(
            guide_index=[0, 1, 2, 3],
            guide_id=["g0", "g1", "ntc", "absent"],
            target_id=["target", "target", "control", "missing"],
            is_control=[False, False, True, False],
        )
    )
    catalog.to_parquet(root / "catalog.parquet", index=False)
    features = tuple(
        RNAFeature(feature_id=f, is_RNA=f != "technical")
        for f in ("technical", *[f"rna-{i}" for i in range(rna_features)])
    )
    records, roles = [], []
    patterns = np.array(
        [
            [1700, 200, 50, 50, 0, 0],
            [200, 1700, 50, 50, 0, 0],
            [50, 50, 1700, 200, 0, 0],
            [50, 50, 200, 1700, 0, 0],
        ],
        dtype=np.uint32,
    )
    for donor in ("fit-a", "fit-b", "query"):
        for condition in ("source", "destination"):
            if donor == "query" and condition == "destination":
                continue
            sid = f"{donor}-{condition}"
            roles.append(SourceRole(source_id=sid, donor_id=donor, condition_role=condition))
            for shard in range(2):
                n = rows_per_shard
                rna = (
                    np.tile(np.array([500, 500, 500, 500, 0, 0], dtype=np.uint32), (n, 1))
                    if null
                    else patterns[np.arange(n) % 4]
                )
                rna = rna.copy()
                if depth_imbalance:
                    rna = np.tile(
                        np.array([[900, 100, 0, 0, 0, 0], [1, 9, 0, 0, 0, 0]], dtype=np.uint32),
                        (n // 2, 1),
                    )
                if source_shift:
                    index = 2 * (donor == "fit-b") + (condition == "destination")
                    rna[:, index] += (index + 1) * 2500
                if rna_features > 6:
                    # Bounded synthetic width/sparsity fixture, not a real cohort sample.
                    rng = np.random.default_rng(734 + shard)
                    extra = (
                        sparse.random(
                            n,
                            rna_features - 6,
                            density=0.30,
                            format="csr",
                            random_state=rng,
                            data_rvs=lambda size, generator=rng: generator.integers(
                                1, 5, size=size
                            ),
                        )
                        .toarray()
                        .astype(np.uint32)
                    )
                    rna = np.column_stack((rna, extra))
                if zero and shard == 0:
                    rna[-1] = 0
                counts = sparse.csr_matrix(
                    np.column_stack((np.full(n, 900000, dtype=np.uint32), rna))
                )
                name = f"{sid}-{shard}"
                sparse.save_npz(root / f"{name}.npz", counts)
                pd.DataFrame(
                    dict(
                        source_id=[sid] * n,
                        row_in_shard=np.arange(n),
                        cell_id=[f"cell-{shard}-{i}" for i in range(n)],
                        guide_index=np.arange(n) % 3,
                        donor_id=[donor] * n,
                        condition=[condition] * n,
                        RNA_UMIs=rna.sum(axis=1),
                        all_feature_UMIs=np.asarray(counts.sum(axis=1)).ravel(),
                    )
                ).to_parquet(root / f"{name}.parquet", index=False)
                records.append(
                    PreparedShard(
                        source_id=sid,
                        shard=shard,
                        rows=n,
                        nnz=counts.nnz,
                        counts=artifact(root, f"{name}.npz"),
                        cells=artifact(root, f"{name}.parquet"),
                    )
                )
    common = dict(
        package_completion_sha256="a" * 64,
        package_inventory_sha256="b" * 64,
        parent_view_sha256="c" * 64,
        amendment_sha256="d" * 64,
        feature_order_sha256=contract_id([f.model_dump() for f in features]),
        guide_catalog=GuideCatalogBinding(
            artifact=artifact(root, "catalog.parquet"),
            ordered_catalog_sha256=contract_id(catalog.to_dict("records")),
        ),
        n_features=len(features),
        task_id="paired",
    )
    fit = PreparedAccess(
        **common,
        role="representation_fit",
        source_ids=tuple(r.source_id for r in roles if r.donor_id != "query"),
        shards=tuple(s for s in records if not s.source_id.startswith("query")),
    )
    query = PreparedAccess(
        **common,
        role="query",
        source_ids=("query-source",),
        shards=tuple(s for s in records if s.source_id.startswith("query")),
    )
    spec = CountRepresentationSpec(
        base_git_commit="1" * 40,
        implementation_sha256=implementation_tree_hash(),
        environment=environment_identity(),
        process_environment=process_environment(),
        fitting=fit,
        query=query,
        source_roles=tuple(roles),
        fitting_donors=("fit-a", "fit-b"),
        query_donor="query",
        protected_donors=("outer",),
        source_condition="source",
        destination_condition="destination",
        features=features,
        rules=rules
        or RepresentationRules(
            hidden_dims=(24, 12),
            factor_rank=1,
            candidate_epochs=epochs,
            learning_rate=0.01,
            batch_rows=32,
        ),
    )
    return root, spec


@pytest.fixture
def small(tmp_path):
    return fixture(tmp_path / "inputs", epochs=(1, 2))


def test_real_stream_calibration_fresh_refit_and_frozen_individual_laws(tmp_path, monkeypatch):
    root, spec = fixture(tmp_path / "inputs")
    original_load = PreparedShardReader._load

    def only_fit(self, source, shard):
        assert not source.startswith("query"), "Query data must never enter fitting"
        return original_load(self, source, shard)

    with monkeypatch.context() as context:
        context.setattr(PreparedShardReader, "_load", only_fit)
        calibrated = calibrate_representation(root, spec, tmp_path / "calibration")
        fitted = refit_representation(root, tmp_path / "calibration", tmp_path / "fit")
    report = json.loads((tmp_path / "calibration/calibration.json").read_text())
    selected = next(
        r for r in report["candidates"] if r["epoch"] == report["selected_epochs"]["autoencoder"]
    )
    score = selected["models"]
    assert (
        score["autoencoder"]["mean_cell_direction_CE"]
        < score["condition_composition"]["mean_cell_direction_CE"] - 0.1
    )
    assert calibrated.facts["heldout_count_gate"]
    assert fitted.facts["fitting_rows"] == 256 and fitted.facts["query_cells_used"] == 0
    audit = report["partition"]["audit_rows"]
    assert all(r["exposure"]["rows"] == 256 - audit for r in report["epochs"])
    exposure = json.loads((tmp_path / "fit/exposure.json").read_text())
    assert all(r["exposure"]["rows"] == 256 for r in exposure["epochs"])
    assert exposure["scaling"]["positive_RNA_cells"] == 252
    assert not fitted.facts["representation_qualified"]
    model, saved_spec, saved = load_representation(tmp_path / "fit")
    assert saved_spec == spec and saved.identity() == fitted.identity()
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    before = sha256_file(tmp_path / "fit/model.safetensors")
    latent = compile_latent_cells(
        root, tmp_path / "fit", spec.query, tmp_path / "query", role="query"
    )
    assert latent.facts["retained_individual_cells"] == 64 and latent.facts["geometry_cells"] == 63
    assert sha256_file(tmp_path / "fit/model.safetensors") == before
    store = EmpiricalLatentStore(tmp_path / "query", representation_sha256=fitted.identity())
    parts = list(store.population("query-source", 0, batch_rows=5))
    states = np.concatenate([p[0] for p in parts])
    weights = np.concatenate([p[1] for p in parts])
    assert len(states) == 22 and states.shape[1] == 48
    assert np.max(np.var(states, axis=0)) > 0.05
    assert weights.sum() == pytest.approx(1) and np.unique(states, axis=0).shape[0] > 1
    assert list(store.population("query-source", 3)) == [] and len(store.identity()) == 64
    expected = model.decode(torch.from_numpy(states[:2]), torch.tensor([100.0, 200.0]))
    assert torch.allclose(expected.sum(1), torch.tensor([100.0, 200.0]), atol=1e-4)
    with pytest.raises(FileExistsError):
        compile_latent_cells(root, tmp_path / "fit", spec.query, tmp_path / "query", role="query")
    with pytest.raises(ContractError, match="exact"):
        compile_latent_cells(root, tmp_path / "fit", spec.fitting, tmp_path / "bad", role="query")
    with pytest.raises(ContractError, match="another"):
        EmpiricalLatentStore(tmp_path / "query", representation_sha256="0" * 64)
    with pytest.raises(ContractError, match="Unknown"):
        list(store.population("query-source", 99))
    with pytest.raises(ContractError, match="budget"):
        list(store.population("query-source", 0, batch_rows=0))
    refit = refit_representation(root, tmp_path / "calibration", tmp_path / "repeat-fit")
    assert refit.facts["model_numerical_sha256"] == before
    fitting_latent = compile_latent_cells(
        root, tmp_path / "fit", spec.fitting, tmp_path / "fitting-latent", role="fitting"
    )
    assert fitting_latent.facts["retained_individual_cells"] == 256
    assert fitting_latent.facts["geometry_cells"] == 252
    # Population access uses the compiled integer index, not a full parquet scan.
    with monkeypatch.context() as context:
        context.setattr(
            "credo_count_sde_v4.count_representation.latent.pq.ParquetFile",
            lambda *a, **k: pytest.fail("population access must be indexed"),
        )
        assert sum(len(part[0]) for part in store.population("query-source", 0)) == 22
    with h5py.File(tmp_path / "query/latent.h5", "r+") as handle:
        handle["state"][0, 0] += 1
    with pytest.raises(ContractError, match="changed"):
        list(store.population("query-source", 0))


def test_null_does_not_advance_on_training_reconstruction(tmp_path):
    root, spec = fixture(tmp_path / "input", null=True, zero=False, epochs=(1, 2))
    record = calibrate_representation(root, spec, tmp_path / "calibration")
    assert not record.facts["heldout_count_gate"] and not record.facts["representation_qualified"]


def test_stratified_partition_is_metadata_only_disjoint_complete_and_stable(small, monkeypatch):
    root, spec = small

    def forbidden(*args, **kwargs):
        pytest.fail("Partition construction must not open counts")

    with monkeypatch.context() as context:
        context.setattr(PreparedShardReader, "_load", forbidden)
        partition, report = partition_audit(root, spec)
        assert partition_audit(root, spec)[0] == partition
    seen = {}
    for split in ("train", "audit"):
        seen[split] = {
            (r.source_id, r.shard, r.row_in_shard)
            for _, cells in iter_counts(root, spec, partition=partition, split=split)
            for r in cells.itertuples(index=False)
        }
    assert not seen["train"] & seen["audit"]
    assert len(seen["train"] | seen["audit"]) == 256 and len(seen["audit"]) == report["audit_rows"]
    counts, cells = next(iter_counts(root, spec))
    assert counts.shape[1] == 6 and counts.max() < 900000
    first, second = halves(counts, cells, 0)
    assert (first + second != counts).nnz == 0
    a, b = halves(counts[::-1].tocsr(), cells.iloc[::-1], 0)
    assert (a[::-1] != first).nnz == 0 and (b[::-1] != second).nnz == 0
    with pytest.raises(ContractError, match="partition"):
        next(iter_counts(root, spec, split="audit"))


def test_authorized_numerics_independent_of_unopened_query_and_package_provenance(small, tmp_path):
    root, spec = small
    first = calibrate_representation(root, spec, tmp_path / "calibration-a")
    payload = spec.model_dump(mode="json")
    for name in ("fitting", "query"):
        payload[name]["package_completion_sha256"] = "e" * 64
    for shard in payload["query"]["shards"]:
        shard["counts"]["sha256"] = "0" * 64
    alternative = CountRepresentationSpec.model_validate(payload)
    second = calibrate_representation(root, alternative, tmp_path / "calibration-b")
    assert first.identity() != second.identity()
    for name in ("autoencoder", "count_factor"):
        assert sha256_file(
            tmp_path / f"calibration-a/calibration_{name}.safetensors"
        ) == sha256_file(tmp_path / f"calibration-b/calibration_{name}.safetensors")


@pytest.mark.parametrize(
    "change", ["fit_role", "query_role", "outer_overlap", "features", "source_roles", "package"]
)
def test_bad_information_contracts_rejected_before_reads(small, change):
    _, spec = small
    payload = spec.model_dump(mode="json")
    if change == "fit_role":
        payload["fitting"]["role"] = "query"
    elif change == "query_role":
        payload["query"]["role"] = "dynamics_supervision"
    elif change == "outer_overlap":
        payload["protected_donors"] = ["fit-a"]
    elif change == "features":
        payload["features"][0]["is_RNA"] = True
    elif change == "source_roles":
        payload["source_roles"][-1]["condition_role"] = "destination"
    else:
        payload["query"]["package_inventory_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        CountRepresentationSpec.model_validate(payload)


def test_sparse_first_projection_count_score_ablation_and_budgets(small):
    root, spec = small
    counts, _ = next(iter_counts(root, spec))
    model = initialize(spec.rules, counts.shape[1])
    values = sparse_tensor(counts, model.device, normalize=True)
    assert values.layout == torch.sparse_coo
    actual = model.raw_encode(counts)
    assert torch.allclose(actual, model.encoder(model.first(values.to_dense())), atol=1e-5)
    logits = model.log_probabilities(counts)
    score, depth = count_loss(logits, counts)
    assert torch.isfinite(score).all() and bool((depth >= 0).all())
    score.mean().backward()
    assert model.first.weight.grad is not None and bool(model.first.weight.grad.abs().sum() > 0)
    ablated = model.log_probabilities(counts, ablate=True)
    assert torch.allclose(ablated[0], ablated[-1])
    with pytest.raises(ContractError, match="dimension"):
        model.raw_encode(counts[:, :2])
    with pytest.raises(ContractError, match="dimensions"):
        count_loss(logits, counts[:, :2])
    with pytest.raises(ContractError, match="Aligned"):
        model.decode(actual, torch.tensor([-1.0] * len(actual)))
    tiny = spec.model_copy(
        update={"rules": spec.rules.model_copy(update={"maximum_dense_working_bytes": 1})}
    )
    with pytest.raises(ContractError, match="budget"):
        next(iter_counts(root, tiny))
    tiny = spec.model_copy(
        update={"rules": spec.rules.model_copy(update={"maximum_partition_rows": 1})}
    )
    with pytest.raises(ContractError, match="budget"):
        partition_audit(root, tiny)


def test_corrupt_inputs_and_runtime_fail_closed(small, tmp_path):
    root, spec = small
    changed = spec.model_copy(update={"implementation_sha256": "0" * 64})
    with pytest.raises(ContractError, match="runtime"):
        calibrate_representation(root, changed, tmp_path / "bad")
    original = root / spec.fitting.shards[0].counts.relative_uri
    raw = bytearray(original.read_bytes())
    raw[30] ^= 1
    original.write_bytes(raw)
    with pytest.raises(ContractError, match="hash"):
        list(iter_counts(root, spec))
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("epochs", [(), (2, 1), (1, 1), (0,), (True,)])
def test_prespecified_exposures_are_strict(epochs):
    with pytest.raises(ValidationError):
        RepresentationRules(candidate_epochs=epochs)


def test_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ContractError, match="CUDA"):
        initialize(RepresentationRules(device="cuda"), 6)
    with pytest.raises(ValidationError, match="one half"):
        RepresentationRules(thinning_probability=0.7)


def test_selected_fitting_rows_are_the_exact_authority(small):
    root, spec = small
    payload = spec.model_dump(mode="json")
    for shard in payload["fitting"]["shards"]:
        shard["allowed_rows"] = [0, 1, 2, 3]
    narrowed = CountRepresentationSpec.model_validate(payload)
    batches = list(iter_counts(root, narrowed))
    assert sum(len(cells) for _, cells in batches) == 32
    assert all(set(cells.row_in_shard).issubset({0, 1, 2, 3}) for _, cells in batches)
    partition, report = partition_audit(root, narrowed)
    assert report["fitting_rows"] == 32
    assert sum(map(len, partition.values())) == report["audit_rows"]


def test_empty_audit_and_input_tree_publication_are_rejected(small, tmp_path):
    root, spec = small
    with pytest.raises(ContractError, match="outside"):
        calibrate_representation(root, spec, root / "forbidden")
    payload = spec.model_dump(mode="json")
    payload["fitting"]["shards"] = [s for s in payload["fitting"]["shards"] if s["shard"] == 0]
    for shard in payload["fitting"]["shards"]:
        shard["allowed_rows"] = [0]
    one_cell = CountRepresentationSpec.model_validate(payload)
    with pytest.raises(ContractError, match="Nonempty"):
        partition_audit(root, one_cell)


def test_module_cli_runs_explicit_stages(small, tmp_path, capsys):
    from credo_count_sde_v4.count_representation.__main__ import main

    root, spec = small
    path = tmp_path / "spec.json"
    path.write_text(spec.model_dump_json())
    common = ["--input-root", str(root)]
    main(["calibrate", *common, "--specification", str(path), "--output", str(tmp_path / "cal")])
    main(
        [
            "refit",
            *common,
            "--calibration",
            str(tmp_path / "cal"),
            "--output",
            str(tmp_path / "fit"),
        ]
    )
    main(
        [
            "encode",
            *common,
            "--fitted",
            str(tmp_path / "fit"),
            "--role",
            "query",
            "--output",
            str(tmp_path / "encoded"),
        ]
    )
    assert '"latent_cells"' in capsys.readouterr().out
    assert verify(tmp_path / "encoded").facts["retained_individual_cells"] == 64


def test_zero_rna_preserved_and_calibration_never_claims_empty_success(small, tmp_path):
    root, spec = small
    partition, _ = partition_audit(root, spec)
    assert all(rows == sorted(set(rows)) for rows in partition.values())
    record = calibrate_representation(root, spec, tmp_path / "calibration")
    assert record.facts["audit_rows"] > 0
    with pytest.raises(FileExistsError):
        calibrate_representation(root, spec, tmp_path / "calibration")
    with pytest.raises(ContractError, match="stage"):
        verify(tmp_path / "calibration", stage="latent_cells")
    path = tmp_path / "calibration/calibration.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntegrityError):
        refit_representation(root, tmp_path / "calibration", tmp_path / "fit")
