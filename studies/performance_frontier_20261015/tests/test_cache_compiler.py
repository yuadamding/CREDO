"""Tiny fabricated sparse rows and fresh frozen CPU encoders; zero optimizers."""

from dataclasses import replace
import json
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from frontier.cache_compiler import (ArraySink, CompilerLimits, NativeChunk,
    NativeShard, SourceCompileSpec, compile_population, native_source_input_identity, ordered_native_chunks)
from frontier.fitting_provider import CatalogGuide, FoldCacheManifest, FoldFittingCache, ProviderLimits, _array_hash
from frontier.native_inputs import FoldInputs, NativeBatch, NativeHandle, PackageBinding
from frontier.observation import ObservationFold, SparseR48
from frontier.training_owner import ObservationBinding, observation_state_digest


def fixture(*, condition="Stim8hr", donor="D3", fold=None, seed=7, counts=None):
    fold = fold or FoldInputs("subset-D3-D4", ("D3", "D4"))
    catalog = tuple(CatalogGuide(g, t, control) for g, t, control in (
        ("g0", "T0", False), ("g1", "T0", False), ("ntc", "NTC", True),
        ("countOnly", "T3", False), ("absent", "T4", False)))
    counts = np.asarray(counts if counts is not None else [
        [2, 0, 0, 0], [1, 3, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0],
        [0, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 4], [1, 1, 0, 0],
        [0, 2, 2, 0], [0, 0, 0, 0]], dtype=np.uint32)
    guides = np.array([2, 0, 1, 2, 3, 0, 1, 2, 0, 3], np.int64)
    source = donor + "_" + condition
    cells = pd.DataFrame(dict(source_id=[source] * 10, cell_id=[f"c{i}" for i in range(10)],
        row_in_shard=np.r_[np.arange(6), np.arange(4)], guide_index=guides,
        donor_id=[donor] * 10, condition=[condition] * 10,
        RNA_UMIs=counts.sum(1, dtype=np.uint64),
        guide_id=[catalog[g].guide_id for g in guides],
        primary_target_id=[catalog[g].target_id for g in guides],
        is_control=[catalog[g].is_control for g in guides]))
    model = SparseR48(ObservationFold(fold.fold_id, fold.training_donors, "f" * 64),
        features=4, hidden_dims=(8, 6), latent_dim=48, maximum_rows=8, maximum_dense_bytes=1024**2, seed=seed)
    with torch.no_grad():
        model.center.copy_(torch.arange(48, dtype=torch.float32) / 100)
        model.scale.copy_(1 + torch.arange(48, dtype=torch.float32) / 200)
        model.scaling_ready.fill_(True)
    model.freeze()
    binding = ObservationBinding("a" * 64, observation_state_digest(model))
    spec = SourceCompileSpec(fold, catalog, 4, "f" * 64, binding, donor, condition, source,
        "aggregate-culture-unresolved", "dynamics_source" if condition == "Stim8hr" else "dynamics_supervision",
        "1" * 64 if condition == "Stim8hr" else "2" * 64, "3" * 64,
        (NativeShard(0, 6), NativeShard(4, 4)),
        tuple(int(x) for x in np.bincount(guides, minlength=5)),
        tuple(int(x) for x in np.bincount(guides[counts.sum(1) > 0], minlength=5)), True)
    return spec, model, counts, cells


def stream(counts, cells, *, size=3):
    for shard, begin, end in ((0, 0, 6), (4, 6, 10)):
        for start in range(begin, end, size):
            stop = min(start + size, end)
            frame = cells.iloc[start:stop].copy().reset_index(drop=True)
            matrix = sparse.csr_matrix(counts[start:stop])
            depths = np.asarray(matrix.sum(1, dtype=np.uint64)).ravel()
            ids = tuple(json.dumps([s, c], separators=(",", ":")) for s, c in zip(frame.source_id, frame.cell_id))
            yield NativeChunk(shard, start - begin, NativeBatch(matrix, frame, ids, depths, depths > 0))


def limits(**kwargs):
    return CompilerLimits(**(dict(maximum_host_bytes=16 * 1024**2, maximum_chunk_rows=6,
        maximum_chunk_bytes=64 * 1024, encoding_rows=3, maximum_identity_bytes=65536) | kwargs))


def run(f=None, **kwargs):
    spec, model, counts, cells = f or fixture()
    return compile_population(spec, stream(counts, cells), model, limits=limits(), **kwargs)


def test_complete_counts_raw_equal_cell_means_and_canonical_full_state_support():
    spec, model, counts, cells = fixture()
    compiled = run((spec, model, counts, cells))
    p = compiled.population
    assert p.captured_counts.tolist() == [3, 2, 3, 2, 0]
    assert p.positive_counts.tolist() == [3, 2, 2, 0, 0]
    assert p.offsets.tolist() == [0, 3, 5, 7, 7, 7]
    # Equal-cell composition, NOT summed-count pseudobulk: depths 4,8,4.
    np.testing.assert_allclose(p.mean_compositions[0], [5 / 12, 5 / 12, 1 / 6, 0], atol=1e-16)
    np.testing.assert_allclose(p.mean_compositions[1], [0, 0, .5, .5])
    assert np.array_equal(p.mean_compositions[3:], np.zeros((2, 4)))
    positive = counts.sum(1) > 0
    with torch.no_grad():
        expected = model.encode(sparse.csr_matrix(counts[positive])).numpy()
    ordered = np.argsort(cells.guide_index.to_numpy()[positive], kind="stable")
    np.testing.assert_allclose(p.states, expected[ordered], atol=5e-7, rtol=1e-6)
    for field in (p.states, p.mean_compositions, p.captured_counts, p.positive_counts, p.offsets):
        assert not field.flags.writeable
    assert compiled.receipt["optimizer_calls"] == 0
    assert compiled.receipt["encoding_batches"] == 3
    assert compiled.receipt["publication_claimed"] is False
    assert compiled.receipt["observation_technical_holdout_filter_applied"] is False


def test_native_chunk_partition_does_not_change_states_means_or_identities():
    spec, model, counts, cells = fixture()
    one = compile_population(spec, stream(counts, cells, size=1), model, limits=limits())
    six = compile_population(spec, stream(counts, cells, size=6), model, limits=limits())
    assert one.population.reference == six.population.reference
    np.testing.assert_array_equal(one.population.states, six.population.states)
    np.testing.assert_array_equal(one.population.mean_compositions, six.population.mean_compositions)


def test_raw_summary_reused_across_folds_but_latents_and_role_binding_are_not():
    first = run()
    spec, model, counts, cells = fixture(fold=FoldInputs("subset-D1-D3", ("D1", "D3")), seed=11)
    spec = replace(spec, source_access_sha256="9" * 64)
    second = compile_population(spec, stream(counts, cells), model, limits=limits(), reuse_raw=first.population)
    p, q = first.population, second.population
    assert p.mean_compositions is q.mean_compositions
    assert p.captured_counts is q.captured_counts
    assert p.reference.raw_summary_sha256 == q.reference.raw_summary_sha256
    assert p.reference.latent_sha256 != q.reference.latent_sha256
    assert not np.array_equal(p.states, q.states)
    assert second.receipt["reused_raw_summary"] is True


def test_endpoint_keeps_laws_without_any_source_support_and_is_provider_compatible():
    populations = []
    observation = None
    for donor in ("D3", "D4"):
        for condition in ("Stim8hr", "Stim48hr"):
            f = fixture(donor=donor, condition=condition)
            if condition == "Stim48hr":
                spec, model, counts, cells = f
                counts[4, 1] = 1  # count-only early guide has observed endpoint state
                cells.loc[4, "RNA_UMIs"] = 1
                spec = replace(spec, positive_counts=(3, 2, 2, 1, 0))
                f = spec, model, counts, cells
            compiled = run(f)
            populations.append(compiled.population)
            observation = f[0].observation_binding
    spec = f[0]
    manifest = FoldCacheManifest(spec.fold, spec.catalog, spec.feature_axis_sha256, 4, observation,
        "1" * 64, "2" * 64, tuple(p.reference for p in populations), synthetic_fixture=True)
    cache = FoldFittingCache(manifest, populations, expected_manifest_sha256=manifest.identity,
        limits=ProviderLimits(16 * 1024**2, 16 * 1024**2, 128))
    assert cache.populations[("D3", "Stim48hr")].positive_counts[3] == 1
    assert ("D3", 3) not in cache.laws


@pytest.mark.parametrize("damage", ["drop", "extra", "swap", "wrong_start", "wrong_shard", "row_order"])
def test_incomplete_duplicate_or_reordered_native_stream_fails(damage):
    spec, model, counts, cells = fixture()
    chunks = list(stream(counts, cells))
    if damage == "drop":
        chunks.pop()
    elif damage == "extra":
        chunks.append(chunks[-1])
    elif damage == "swap":
        chunks[0], chunks[1] = chunks[1], chunks[0]
    elif damage == "wrong_start":
        chunks[0] = replace(chunks[0], start_row=1)
    elif damage == "wrong_shard":
        chunks[0] = replace(chunks[0], shard=1)
    else:
        chunks[0].batch.cells.loc[0, "row_in_shard"] = 1
    with pytest.raises(ValueError):
        compile_population(spec, chunks, model, limits=limits())


def test_duplicate_cell_across_shards_and_wrong_catalog_are_rejected():
    spec, model, counts, cells = fixture()
    cells.loc[8, "cell_id"] = cells.loc[0, "cell_id"]
    with pytest.raises(ValueError, match="Duplicate cell"):
        compile_population(spec, stream(counts, cells), model, limits=limits())
    spec, model, counts, cells = fixture()
    cells.loc[1, "guide_id"] = "ntc"
    with pytest.raises(ValueError, match="canonical catalog"):
        compile_population(spec, stream(counts, cells), model, limits=limits())


@pytest.mark.parametrize("damage", ["donor", "condition", "axis", "model", "scaler", "support", "depth"])
def test_source_model_axis_scaler_and_support_are_exact(damage):
    spec, model, counts, cells = fixture()
    if damage == "donor":
        cells.loc[1, "donor_id"] = "D1"
    elif damage == "condition":
        cells.loc[1, "condition"] = "Rest"
    elif damage == "axis":
        spec = replace(spec, feature_axis_sha256="8" * 64)
    elif damage == "model":
        with torch.no_grad():
            model.first.weight.add_(.001)
    elif damage == "scaler":
        model.scale[0] = 2
    elif damage == "support":
        spec = replace(spec, positive_counts=(2, 2, 2, 0, 0))
    else:
        cells.loc[0, "RNA_UMIs"] = 3
    with pytest.raises(ValueError):
        compile_population(spec, stream(counts, cells), model, limits=limits())


def test_model_mutation_during_stream_invalidates_whole_compilation():
    spec, model, counts, cells = fixture()
    def corrupt():
        yield from stream(counts, cells)
        model.center[0] += 1
    with pytest.raises(ValueError, match="changed during"):
        compile_population(spec, corrupt(), model, limits=limits())


def test_budget_checks_before_iteration_and_runtime_chunk_cap():
    spec, model, counts, cells = fixture()
    def forbidden():
        raise AssertionError("stream consumed before admission")
        yield
    for cap in (limits(maximum_host_bytes=1), limits(maximum_identity_bytes=1)):
        with pytest.raises(ValueError, match="cap"):
            compile_population(spec, forbidden(), model, limits=cap)
    with pytest.raises(ValueError, match="byte cap"):
        compile_population(spec, stream(counts, cells), model, limits=limits(maximum_chunk_bytes=1))
    with pytest.raises(ValueError, match="minibatch"):
        compile_population(spec, stream(counts, cells), model, limits=limits(maximum_chunk_rows=1))


def test_raw_reuse_budget_includes_retained_old_latents_and_exact_cap():
    spec, model, counts, cells = fixture()
    first = compile_population(spec, stream(counts, cells), model, limits=limits())
    base_bytes = first.receipt["conservative_host_estimate_bytes"]
    retained_bytes = first.population.states.nbytes + first.population.offsets.nbytes
    required_bytes = base_bytes + retained_bytes

    # A fresh compilation still fits its original cap. Reusing raw means keeps
    # the old PopulationArrays' latent storage alive alongside the new sink.
    fresh = compile_population(spec, stream(counts, cells), model,
                               limits=limits(maximum_host_bytes=base_bytes))
    assert fresh.receipt["conservative_host_estimate_bytes"] == base_bytes

    def forbidden():
        raise AssertionError("stream consumed before reuse budget admission")
        yield

    with pytest.raises(ValueError, match="host-byte cap"):
        compile_population(spec, forbidden(), model,
            limits=limits(maximum_host_bytes=required_bytes - 1), reuse_raw=first.population)
    reused = compile_population(spec, stream(counts, cells), model,
        limits=limits(maximum_host_bytes=required_bytes), reuse_raw=first.population)
    assert reused.receipt["conservative_host_estimate_bytes"] == required_bytes
    assert reused.population.mean_compositions is first.population.mean_compositions
    np.testing.assert_array_equal(reused.population.states, fresh.population.states)
    assert reused.population.reference == fresh.population.reference


def test_explicit_sink_and_raw_reuse_identity_guard():
    spec, model, counts, cells = fixture()
    sink = ArraySink(np.empty((7, 48), np.float32), np.empty((5, 4), np.float64))
    first = compile_population(spec, stream(counts, cells), model, limits=limits(), sink=sink)
    assert first.population.states is sink.states
    assert first.population.mean_compositions is sink.mean_compositions
    with pytest.raises(ValueError, match="another source"):
        compile_population(replace(spec, source_input_sha256="c" * 64), stream(counts, cells), model,
            limits=limits(), reuse_raw=first.population)
    cells.loc[0, "cell_id"] = "different-source-row"
    with pytest.raises(ValueError, match="row inventory"):
        compile_population(spec, stream(counts, cells), model, limits=limits(), reuse_raw=first.population)


def test_no_dense_cohort_and_no_optimizer_or_global_rng_change():
    spec, model, counts, cells = fixture()
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    with patch.object(torch.optim.Optimizer, "step", side_effect=AssertionError("optimizer forbidden")), \
         patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense cohort forbidden")):
        run((spec, model, counts, cells))
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_scientific_axis_cannot_silently_use_fixture_catalog():
    spec, _, _, _ = fixture()
    with pytest.raises(ValueError, match="26,504"):
        replace(spec, synthetic_fixture=False)
    with pytest.raises(ValueError, match="dynamics role"):
        replace(spec, access_role="representation_fit")


def test_all_zero_rna_source_retains_captured_counts_without_fabricated_state():
    spec, model, counts, cells = fixture(counts=np.zeros((10, 4), np.uint32))
    with patch.object(model, "encode", side_effect=AssertionError("zero RNA must not be encoded")):
        result = run((spec, model, counts, cells))
    assert result.population.states.shape == (0, 48)
    assert result.population.captured_counts.sum() == 10
    assert result.population.positive_counts.sum() == 0
    assert result.receipt["encoding_batches"] == 0
    assert _array_hash(np.empty((0, 48), np.float32)) != _array_hash(np.empty((0, 47), np.float32))
    assert _array_hash(np.empty((0, 48), np.float32)) != _array_hash(np.empty((0, 48), np.float64))


def test_native_adapter_validates_capability_then_reads_canonical_bounded_rows():
    spec, model, counts, cells = fixture()
    shards = []
    for shard in spec.shards:
        record = SimpleNamespace(source_id=spec.source_id, shard=shard.shard, rows=shard.rows, allowed_rows=None)
        record.model_dump = lambda mode, s=shard: dict(source_id=spec.source_id, shard=s.shard, rows=s.rows)
        shards.append(record)
    access = SimpleNamespace(identity=lambda: spec.source_access_sha256, source_ids=(spec.source_id,), shards=tuple(shards))
    index = SimpleNamespace(source_roles={spec.source_id: (spec.donor, spec.condition)}, shards=shards,
        binding=PackageBinding("0" * 64, "1" * 64, 4, 5, True), rna_axis_sha256=spec.feature_axis_sha256,
        inventory={f"source/{spec.source_id}/population_accounting.parquet": "4" * 64},
        access=lambda fold, role: access,
        catalog=lambda: pd.DataFrame([dict(guide_index=i, guide_id=g.guide_id, target_id=g.target_id,
                                          is_control=g.is_control) for i, g in enumerate(spec.catalog)]))
    spec = replace(spec, source_input_sha256=native_source_input_identity(index, spec.source_id))
    handle = object.__new__(NativeHandle)
    handle.index, handle.access = index, access
    called = []
    expected = list(stream(counts, cells, size=2))
    def read_rows(addresses):
        chunk = expected[len(called)]
        assert addresses == [(spec.source_id, chunk.shard, row)
                             for row in range(chunk.start_row, chunk.start_row + len(chunk.batch.cells))]
        called.append(addresses)
        return chunk.batch
    handle.read_rows = read_rows
    iterator = ordered_native_chunks(handle, spec, batch_rows=2)
    assert called == []
    compiled = compile_population(spec, iterator, model, limits=limits())
    assert len(called) == 5
    assert compiled.receipt["captured_rows"] == 10
    called.clear()
    with pytest.raises(ValueError, match="identity differs"):
        list(ordered_native_chunks(handle, replace(spec, source_access_sha256="5" * 64), batch_rows=2))
    assert called == []


def test_synthetic_compiler_rejects_cuda_identity_without_allocating_cuda():
    spec, model, counts, cells = fixture()
    with patch.object(SparseR48, "device", new_callable=lambda: property(lambda self: torch.device("cuda:0"))):
        with pytest.raises(ValueError, match="CPU-only"):
            compile_population(spec, stream(counts, cells), model, limits=limits())
