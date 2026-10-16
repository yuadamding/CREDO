"""Complete tiny native sources and released truth; no optimizer/GPU/real data."""

from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from frontier.assessment_support import AssessmentSupport, PROTOCOL_SHA256, build_assessment_support
from frontier.cache_compiler import CompilerLimits
from frontier.evaluation import ScoreIdentity, reduce_scores
from frontier.native_inputs import FoldInputs, ForecastExpectation, NativeInputIndex, PackageBinding, VerifiedForecastRelease
from frontier.observation import ObservationFold, SparseR48
from frontier.query_compiler import compile_query_source
from frontier.selection import CHANNELS
from frontier.training_owner import ObservationBinding, observation_state_digest
from frontier.truth_compiler import compile_evaluation_truth
from test_native_inputs import READER, digest, write_json


@pytest.fixture(autouse=True)
def bounded_no_optimizer(monkeypatch):
    torch.set_num_threads(2)
    for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        monkeypatch.setattr(cls, "step", lambda *_a, **_k: pytest.fail("optimizer forbidden"))


def package(tmp_path, *, source_positive=None, endpoint_positive=None):
    """Eight guides, two sister guides, an absent law and mismatched NTC support."""
    source_positive = [2, 1, 1, 3, 1, 2, 0, 0] if source_positive is None else list(source_positive)
    endpoint_positive = [2, 1, 1, 2, 0, 0, 1, 0] if endpoint_positive is None else list(endpoint_positive)
    captures = {"Stim8hr": [max(n, base) for n, base in zip(source_positive, [3, 1, 1, 3, 1, 2, 0, 1])],
                "Stim48hr": [max(n, base) for n, base in zip(endpoint_positive, [2, 1, 1, 2, 1, 0, 1, 0])]}
    positives = {"Stim8hr": source_positive, "Stim48hr": endpoint_positive}
    root, metadata = tmp_path / "data", tmp_path / "metadata"
    (root / "source").mkdir(parents=True)
    (metadata / "source").mkdir(parents=True)
    guide_ids = ["a0", "a1", "b0", "ntc0", "c0", "ntc1", "d0", "z0"]
    targets = ["A", "A", "B", "NTC", "C", "NTC", "D", "Z"]
    controls = [False, False, False, True, False, True, False, False]
    pd.DataFrame(dict(sgRNA=guide_ids, primary_target_id=targets, is_design_ntc=controls,
        experimental_library_member=[True]*8)).to_parquet(root / "source/guides.parquet", index=False)
    samples, features, shards = [], [], []
    for donor in ("D1", "D2", "D3", "D4"):
        for condition in ("Stim8hr", "Stim48hr"):
            source_id = donor + "_" + condition
            samples.append(dict(source_id=source_id, donor_id=donor, condition=condition))
            for j, feature in enumerate(("PuroR", "ENSG1", "ENSG2", "ENSG3")):
                features.append(dict(source_id=source_id, source_position=j, canonical_position=j,
                                     feature_id=feature, is_RNA=j > 0))
            gids, values = [], []
            for i in range(8):
                for r in range(captures[condition][i]):
                    gids.append(i)
                    values.append([1, i+1, 1, 0] if r < positives[condition][i] else [1, 0, 0, 0])
            matrix = sparse.csr_matrix(np.array(values, dtype=np.uint32))
            n = len(gids)
            counts_name = f"source/{source_id}/counts/00000.csr.npz"
            cells_name = f"source/{source_id}/cells/00000.parquet"
            (root / counts_name).parent.mkdir(parents=True)
            (root / cells_name).parent.mkdir(parents=True)
            sparse.save_npz(root / counts_name, matrix)
            pd.DataFrame(dict(source_id=[source_id]*n, row_in_shard=np.arange(n),
                cell_id=["cell-"+str(i) for i in range(n)], guide_index=gids,
                guide_id=[guide_ids[i] for i in gids], primary_target_id=[targets[i] for i in gids],
                is_control=[controls[i] for i in gids], RNA_UMIs=np.array(matrix[:, 1:].sum(axis=1)).ravel().astype(np.uint64),
                donor_id=[donor]*n, condition=[condition]*n)).to_parquet(root / cells_name, index=False)
            shards.append(dict(source_id=source_id, shard=0, rows=n, nnz=matrix.nnz,
                path=counts_name, sha256=digest(root / counts_name), bytes=(root / counts_name).stat().st_size,
                cells_path=cells_name, cells_sha256=digest(root / cells_name),
                cell_table_bytes=(root / cells_name).stat().st_size, exact_reload=True))
            pd.DataFrame(dict(source_id=[source_id]*8, guide_index=list(range(8)), guide_id=guide_ids,
                observed=[True]*8, n_cells=captures[condition], captured_cell_denominator=[n]*8,
                guide_universe_size=[8]*8, jeffreys_frequency=(np.array(captures[condition])+.5)/(n+4)
                )).to_parquet(root / f"source/{source_id}/population_accounting.parquet", index=False)
    for name, rows in (("samples_and_culture_branches", samples), ("features_and_source_positions", features),
                        ("shard_manifest", shards)):
        pd.DataFrame(rows).to_parquet(root / f"source/{name}.parquet", index=False)
    members = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(root).as_posix()}\n" for p in members))
    write_json(root / "INPUT_DATA_COMPLETE.json", dict(inventory_sha256=digest(root / "SHA256SUMS"),
                                                    RNA_features=3, technical_features=1))
    for name in ("INPUT_DATA_COMPLETE.json", "SHA256SUMS", "source/guides.parquet",
                 "source/samples_and_culture_branches.parquet", "source/features_and_source_positions.parquet",
                 "source/shard_manifest.parquet"):
        (metadata / name).write_bytes((root / name).read_bytes())
    index = NativeInputIndex(metadata, READER,
        binding=PackageBinding(digest(root / "INPUT_DATA_COMPLETE.json"), digest(root / "SHA256SUMS"),
                               rna_features=3, guide_categories=8, synthetic_fixture=True))
    fold = FoldInputs("inner-D3-D4", ("D3", "D4"))
    observation = SparseR48(ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256),
        features=3, hidden_dims=(8, 6), latent_dim=48, maximum_rows=4, maximum_dense_bytes=1024**2, seed=7)
    with torch.no_grad():
        observation.scaling_ready.fill_(True)
    observation.freeze()
    args = dict(fold=fold, donor="D1", source_id="D1_Stim8hr", replicate_id="aggregate-unresolved",
        observation=observation, observation_binding=ObservationBinding("a"*64, observation_state_digest(observation)),
        limits=CompilerLimits(8*1024**2, 3, 64*1024, 2, 64*1024), batch_rows=3)
    return root, index, args


def compile_pair(built, *, release_name="release-0", model_sha="b"*64):
    root, index, args = built
    query = index.open(root, args["fold"], "query", query_donor="D1", max_batch_rows=3)
    source = compile_query_source(query, **args).source
    expectation = ForecastExpectation(args["fold"].identity, query.access.identity(), (model_sha,),
        "c"*64, "d"*64, args["observation_binding"].artifact_sha256, (release_name,))
    publication = root / release_name
    publication.mkdir()
    (publication / "bank.bin").write_bytes(b"fabricated " + release_name.encode())
    write_json(publication / "seal.json", dict(schema="credo.frontier.forecast-seal.v1", status="complete",
        bindings=expectation.record(), forecasts=[dict(forecast_id=release_name,
            artifacts=[dict(relative_uri="bank.bin", sha256=digest(publication / "bank.bin"),
                            size_bytes=(publication / "bank.bin").stat().st_size)])]))
    release = VerifiedForecastRelease.verify(publication, "seal.json",
        manifest_sha256=digest(publication / "seal.json"), expectation=expectation)
    endpoint = index.open(root, args["fold"], "evaluation_truth", query_donor="D1", release=release, max_batch_rows=3)
    truth = compile_evaluation_truth(endpoint, **(args | dict(source_id="D1_Stim48hr", release=release)))
    return source, truth


@pytest.fixture
def pair(tmp_path):
    return compile_pair(package(tmp_path))


def test_nine_weights_common_support_ntc_macro_targets_and_all_catalog(pair):
    source, truth = pair
    rng = torch.get_rng_state().clone()
    support = build_assessment_support(source, truth)
    assert set(support.weights) == set(CHANNELS)
    assert support.required_row_ids == ("guide:0", "guide:1", "guide:2", "guide:3", "catalog", "pooled_control")
    for channel in ("state_energy", "spread_absolute_error", "shape_mmd2", "joint_mse"):
        assert dict(support.weights[channel]) == {f"guide:{i}": .25 for i in range(4)}
    for channel in ("target_endpoint_mse", "pooled_effect_mse"):
        assert dict(support.weights[channel]) == {"guide:0": .25, "guide:1": .25, "guide:2": .5}
    for channel in ("count_ce_minus_entropy", "count_quadratic"):
        assert dict(support.weights[channel]) == {"catalog": 1.0}
    assert dict(support.weights["pooled_control_endpoint_mse"]) == {"pooled_control": 1.0}
    assert support.record["rule_binding"]["protocol_sha256"] == PROTOCOL_SHA256
    assert torch.equal(rng, torch.get_rng_state())
    assert support.record["optimizer_calls"] == 0


def test_full_inventory_missing_support_count_zeros_and_control_weights(pair):
    support = build_assessment_support(*pair)
    record, rows = support.record, support.inventory
    assert len(rows) == 8
    assert [r["support_category"] for r in rows] == ["paired"]*4 + ["source_only", "source_only", "endpoint_only", "neither"]
    assert record["coverage"]["source_only_indices"] == [4, 5]
    assert record["coverage"]["endpoint_only_indices"] == [6]
    assert record["coverage"]["source_capture_zero_indices"] == [6]
    assert record["coverage"]["endpoint_capture_zero_indices"] == [5, 7]
    assert rows[7]["source_captured_count"] == 1 and rows[7]["source_positive_rna_count"] == 0
    assert rows[4]["observation_unavailable_reason"]["state_energy"] == "missing_endpoint_positive_rna"
    assert rows[6]["observation_unavailable_reason"]["state_energy"] == "missing_source_positive_rna"
    assert all(r["eligible_for_channel"]["count_quadratic"] for r in rows)
    assert record["control_pool"]["source_positive_rna_cells"] == 5
    assert record["control_pool"]["endpoint_positive_rna_cells"] == 2
    assert record["control_pool"]["source_weight_without_endpoint_rna"] == .4
    assert record["control_pool"]["common_pool_available"] is True
    assert rows[3]["source_control_positive_rna_weight"] == .6
    assert rows[5]["source_control_positive_rna_weight"] == .4
    assert rows[3]["endpoint_control_positive_rna_weight"] == 1
    assert rows[3]["observation_unavailable_reason"]["target_endpoint_mse"] == "control_guide_not_target_endpoint"


def score_rows(support):
    identity = ScoreIdentity("inner-D3-D4", "D1", "fabricated", 0, 16,
                            *[str(i)*64 for i in range(1, 7)])
    rows = {name: dict(identity=asdict(identity), arithmetic_only=False, unavailable={})
            for name in support.required_row_ids}
    for channel, weights in support.weights.items():
        for name in weights:
            rows[name][channel] = 2.0
    return identity, rows


def test_missing_candidate_keeps_required_weight_and_valid_endpoints(pair):
    support = build_assessment_support(*pair)
    identity, rows = score_rows(support)
    rows["guide:0"]["state_energy"] = None
    rows["guide:0"]["unavailable"]["state_energy"] = "candidate_missing_prediction"
    result = reduce_scores(rows, support.weights, identity=identity)
    assert result["state_energy"] is None
    assert result["target_endpoint_mse"] == 2
    assert result["count_quadratic"] == 2
    assert result["coverage"]["state_energy"]["available_weight"] == .75
    assert dict(support.weights["state_energy"])["guide:0"] == .25


@pytest.mark.parametrize("source_ntc,endpoint_ntc", [(False, True), (True, False), (False, False)])
def test_missing_control_does_not_drop_target_endpoint_or_effect_weights(tmp_path, source_ntc, endpoint_ntc):
    early = [2, 1, 1, 3 if source_ntc else 0, 1, 2 if source_ntc else 0, 0, 0]
    late = [2, 1, 1, 2 if endpoint_ntc else 0, 0, 0, 1, 0]
    support = build_assessment_support(*compile_pair(package(tmp_path, source_positive=early, endpoint_positive=late)))
    assert len(support.weights["target_endpoint_mse"]) == 3
    assert support.weights["pooled_effect_mse"] == support.weights["target_endpoint_mse"]
    assert support.weights["pooled_control_endpoint_mse"] == {"pooled_control": 1.0}
    missing = support.record["unavailable_channels_by_observation_support"]
    assert "target_endpoint_mse" not in missing and "pooled_effect_mse" in missing
    identity, rows = score_rows(support)
    for channel in ("pooled_control_endpoint_mse", "pooled_effect_mse"):
        for name in support.weights[channel]:
            rows[name][channel] = None
            rows[name]["unavailable"][channel] = missing[channel]
    result = reduce_scores(rows, support.weights, identity=identity)
    assert result["target_endpoint_mse"] == 2
    assert result["pooled_control_endpoint_mse"] is None
    assert result["pooled_effect_mse"] is None


def test_no_paired_guides_are_unavailable_not_zero(tmp_path):
    source, truth = compile_pair(package(tmp_path, source_positive=[1,0,0,1,0,0,0,0],
                                        endpoint_positive=[0,1,0,0,0,1,0,0]))
    support = build_assessment_support(source, truth)
    assert support.required_row_ids == ("catalog", "pooled_control")
    assert support.weights["state_energy"] == {} and support.weights["target_endpoint_mse"] == {}
    assert support.record["control_pool"]["common_pool_available"]  # Different NTCs do not invalidate cell pools.
    identity, rows = score_rows(support)
    result = reduce_scores(rows, support.weights, identity=identity)
    assert result["state_energy"] is None and result["target_endpoint_mse"] is None
    assert result["count_quadratic"] == result["pooled_control_endpoint_mse"] == 2


def test_support_digest_is_candidate_release_independent_and_immutable(tmp_path):
    built = package(tmp_path)
    one = build_assessment_support(*compile_pair(built))
    two = build_assessment_support(*compile_pair(built, release_name="different-bank", model_sha="e"*64))
    assert one.identity == two.identity and one.record == two.record
    assert not any(k in json.dumps(one.record) for k in ("model_sha256", "forecast_seal_sha256", "model_seed", "particle_prefix"))
    changed = one.record
    changed["channel_weights"]["state_energy"].clear()
    assert one.weights["state_energy"]
    changed_weights = one.weights
    changed_weights["state_energy"]["guide:0"] = 1
    assert one.weights["state_energy"]["guide:0"] == .25
    with pytest.raises(TypeError, match="released observations"):
        AssessmentSupport(changed)


def test_missing_or_forged_truth_and_changed_source_rejected(pair):
    source, truth = pair
    for bad in (None, True, truth.population, truth.receipt):
        with pytest.raises(TypeError):
            build_assessment_support(source, bad)
    values = source.captured_counts
    values.setflags(write=True)
    values[0] += 1
    values.setflags(write=False)
    with pytest.raises(ValueError, match="source arrays changed"):
        build_assessment_support(source, truth)


def test_release_for_other_query_is_not_support_authority(pair):
    source, truth = pair
    # Creating a fresh otherwise valid source with a different access binding
    # requires refreshing its source latent identity, not bypassing its hash.
    from frontier.fitting_provider import PopulationArrays
    ref = replace(source.reference, source_access_sha256="f"*64)
    view = PopulationArrays(ref, source.captured_counts, source.positive_counts, source.offsets, source.states, None)
    ref = replace(ref, latent_sha256=view.latent_identity(catalog_sha256=source.catalog_sha256,
                                                       observation_binding=source.observation_binding))
    changed = replace(source, reference=ref)
    with pytest.raises(ValueError, match="release differ"):
        build_assessment_support(changed, truth)


@pytest.mark.parametrize("field", ["fold", "catalog", "observation_binding", "feature_axis_sha256"])
def test_other_valid_but_incompatible_source_bindings_rejected(pair, field):
    from frontier.fitting_provider import PopulationArrays, _hash
    source, truth = pair
    changes = {
        "fold": FoldInputs("different-fold", ("D3", "D4")),
        "catalog": (replace(source.catalog[0], target_id="OTHER"),) + source.catalog[1:],
        "observation_binding": replace(source.observation_binding, artifact_sha256="f"*64),
        "feature_axis_sha256": "f"*64,
    }
    value = changes[field]
    catalog = value if field == "catalog" else source.catalog
    observation = value if field == "observation_binding" else source.observation_binding
    view = PopulationArrays(source.reference, source.captured_counts, source.positive_counts,
                            source.offsets, source.states, None)
    ref = replace(source.reference, latent_sha256=view.latent_identity(
        catalog_sha256=_hash([asdict(g) for g in catalog]), observation_binding=observation))
    changed = replace(source, **{field: value}, reference=ref)
    with pytest.raises(ValueError, match="release differ"):
        build_assessment_support(changed, truth)


def test_only_control_paired_does_not_become_a_molecular_target(tmp_path):
    support = build_assessment_support(*compile_pair(package(tmp_path,
        source_positive=[0,0,0,2,0,0,0,0], endpoint_positive=[0,0,0,1,0,0,0,0])))
    assert support.weights["state_energy"] == {"guide:3": 1.0}
    assert support.weights["target_endpoint_mse"] == support.weights["pooled_effect_mse"] == {}
    assert support.weights["pooled_control_endpoint_mse"] == {"pooled_control": 1.0}
    assert support.record["coverage"]["represented_targets"] == 0
