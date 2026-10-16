"""Fabricated source laws only, <=128 particles/law and zero optimizer calls."""

from dataclasses import asdict, replace
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from frontier.models import build_model
from frontier.sampling import (LawIdentity, PooledSourceIdentity, LawRequest, SamplingContext, SAMPLER_VERSION,
                               SamplerSpecification, SourceLaw, sample_batch)


def context(**changes):
    value = SamplingContext("fabricated-frontier", ("D3", "D4"), "F0-V0", 321, 0, 256, "training")
    return replace(value, **changes)


def request(guide="g1", donor="D3", source="source-a", replicate="replicate-a", occurrence=0,
            weights=(.1, .2, .3, .4)):
    states = np.arange(4 * 48, dtype=np.float32).reshape(4, 48) / 100
    law = SourceLaw(LawIdentity(donor, "Stim8hr", guide, source, replicate), states,
                    np.asarray(weights, dtype=np.float64))
    return LawRequest(law, occurrence)


def test_exact_prefix_and_particle_offset_invariance():
    requests = [request(), request("g2")]
    full = sample_batch(requests, context(), particles=128)
    for size in (1, 7, 32, 64, 127):
        prefix = sample_batch(requests, context(), particles=size)
        assert torch.equal(prefix.states, full.states[:, :size])
        assert torch.equal(prefix.noise, full.noise[:, :, :size])
        assert torch.equal(prefix.source_indices, full.source_indices[:, :size])
    pieces = [sample_batch(requests, context(), particles=32, particle_offset=i) for i in (0, 32, 64, 96)]
    assert torch.equal(torch.cat([part.states for part in pieces], 1), full.states)
    assert torch.equal(torch.cat([part.noise for part in pieces], 2), full.noise)


def test_law_order_and_microbatch_partition_are_invariant():
    requests = [request("g1"), request("g2"), request("g3", donor="D4")]
    full = sample_batch(requests, context(), particles=32)
    reverse = sample_batch(requests[::-1], context(), particles=32)
    assert torch.equal(reverse.states.flip(0), full.states)
    assert torch.equal(reverse.noise.flip(1), full.noise)
    pieces = [sample_batch([value], context(), particles=32) for value in requests]
    assert torch.equal(torch.cat([part.states for part in pieces]), full.states)
    assert torch.equal(torch.cat([part.noise for part in pieces], 1), full.noise)


def test_common_draws_across_all_families_keep_configuration_in_receipt():
    requests = [request()]
    original = sample_batch(requests, context(), particles=32)
    for family in range(6):
        candidate = sample_batch(requests, context(configuration_id=f"F{family}-V2"), particles=32)
        assert torch.equal(candidate.states, original.states)
        assert torch.equal(candidate.noise, original.noise)
        assert candidate.receipt["context"]["configuration_id"] == f"F{family}-V2"
        assert candidate.receipt["configuration_in_draw_key"] is False


def test_purpose_seed_update_generation_and_scoped_source_keys_separate_streams():
    original = sample_batch([request()], context(), particles=32)
    for changes in ({"purpose": "cache"}, {"purpose": "forecast"}, {"model_seed": 1},
                    {"model_seed": 2}, {"accepted_update": 257}, {"generation": 1},
                    {"study_seed": 322}, {"training_donors": ("D2", "D3")}):
        other = sample_batch([request()], context(**changes), particles=32)
        assert not torch.equal(other.noise, original.noise)
    for changed in (request(donor="D4"), request(source="source-b"),
                    request(replicate="replicate-b"), request(occurrence=1)):
        other = sample_batch([changed], context(), particles=32)
        assert not torch.equal(other.noise, original.noise)
    stream = original.receipt["streams"][0]
    assert stream["source_stream_sha256"] != stream["noise_stream_sha256"]


def test_duplicate_identity_rejected_but_explicit_occurrence_stable():
    with pytest.raises(ValueError, match="Duplicate scoped"):
        sample_batch([request(), request()], context(), particles=16)
    first, second = request(), request(occurrence=1)
    batch = sample_batch([first, second], context(), particles=16)
    assert not torch.equal(batch.noise[:, 0], batch.noise[:, 1])
    separate = sample_batch([second], context(), particles=16)
    assert torch.equal(batch.noise[:, 1:2], separate.noise)


def test_zero_weight_source_cells_never_drawn_and_weights_not_applied_twice():
    value = request(weights=(0., .25, 0., .75))
    drawn = sample_batch([value], context(), particles=128)
    assert set(drawn.source_indices.flatten().tolist()) == {1, 3}
    assert drawn.receipt["post_sampling_integration"] == "equal_weight_particles_no_second_source_weight_factor"
    point = sample_batch([request(weights=(0., 0., 1., 0.))], context(), particles=128)
    assert torch.equal(point.source_indices, torch.full((1, 128), 2))
    assert torch.equal(point.states[0], torch.from_numpy(value.law.states[2]).expand(128, -1))


def test_f2_strides_draw_the_same_law_not_sorted_source_partitions():
    # A deterministic check of these fabricated draws, not a guarantee of
    # representative finite samples for every future rare-state source law.
    batch = sample_batch([request(weights=(.25, .25, .25, .25))], context(configuration_id="F2-V0"), particles=128)
    for component in range(4):
        indices = batch.source_indices[0, component::4]
        assert set(indices.tolist()) == {0, 1, 2, 3}
    assert not torch.equal(batch.source_indices[0], batch.source_indices[0].sort().values)


def test_no_global_rng_mutation():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    sample_batch([request()], context(), particles=32)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_training_and_cache_roles_and_no_unknown_law_fallback():
    for purpose in ("training", "cache"):
        with pytest.raises(ValueError, match="held-out donor"):
            sample_batch([request(donor="D1")], context(purpose=purpose), particles=8)
    assert sample_batch([request(donor="D1")], context(purpose="forecast"), particles=8).states.shape == (1, 8, 48)
    with pytest.raises(ValueError, match="Stim8hr"):
        LawIdentity("D3", "Stim48hr", "g", "source", "replicate")
    empty = replace(request(), law=replace(request().law, states=np.empty((0, 48), np.float32), weights=np.empty(0)))
    with pytest.raises(ValueError, match="Nonempty"):
        sample_batch([empty], context(), particles=8)


def test_actual_generator_forward_reuses_bound_draws_without_optimizer():
    ctx = context(configuration_id="F2-V0")
    batch = sample_batch([request()], ctx, particles=32)
    model = build_model("F2-V0", seed=0).set_phase("joint")
    with torch.no_grad():
        result = model.rollout(batch.states, batch.noise, compute_anchor=False)
        prefix = model.prefix_rollout(result, 13)
        separately_sampled = sample_batch([request()], ctx, particles=13)
        separate = model.rollout(separately_sampled.states, separately_sampled.noise, compute_anchor=False)
    # Forward batching may change FP32 GEMM rounding; source/noise equality is
    # bitwise, while forecast publication still derives the prefix from max.
    assert torch.equal(separately_sampled.states, batch.states[:, :13])
    assert torch.equal(separately_sampled.noise, batch.noise[:, :, :13])
    torch.testing.assert_close(prefix.states, separate.states, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(prefix.log_mass_increment, separate.log_mass_increment, rtol=2e-6, atol=2e-6)


def test_sampler_guards_and_runtime_binding():
    for weights in ((0., 0., 0., 0.), (.1, .2, .3, .5), (.1, -.2, .3, .8), (.1, np.nan, .3, .6)):
        with pytest.raises(ValueError):
            sample_batch([request(weights=weights)], context(), particles=8)
    with pytest.raises(ValueError):
        sample_batch([request()], context(), particles=0)
    with pytest.raises(ValueError):
        sample_batch([request()], context(), particles=2, particle_offset=8191)
    with pytest.raises(ValueError):
        sample_batch([request()], context(), particles=8, device="cuda")
    with pytest.raises(ValueError):
        sample_batch([request()], context(), particles=8, pin_memory=True)
    with pytest.raises(ValueError):
        SamplerSpecification(numpy_version="unbound-runtime")
    with pytest.raises(ValueError):
        context(model_seed=3)
    with pytest.raises(ValueError):
        context(configuration_id="UNKNOWN")
    with pytest.raises(ValueError):
        sample_batch([], context(), particles=8)


def test_typed_pooled_ntc_stream_is_separate_and_prefix_stable():
    original = request()
    pooled = LawRequest(SourceLaw(PooledSourceIdentity("D3", "Stim8hr", "source-a", "replicate-a",
        "D3-NTC-pool", "1" * 64, "2" * 64), original.law.states, original.law.weights))
    full = sample_batch([pooled], context(), particles=32)
    prefix = sample_batch([pooled], context(), particles=8)
    guide = sample_batch([original], context(), particles=32)
    # Exact pre-extension key schema: typed pool support must not domain-tag or
    # otherwise change any existing actual-guide source/noise random stream.
    for stream, field in (("source_categorical", "source_stream_sha256"), ("standard_normal", "noise_stream_sha256")):
        payload = {"schema": SAMPLER_VERSION, "context": context().draw_binding(),
                   "request": {"law": asdict(original.law.identity), "occurrence": 0}, "stream": stream}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert guide.receipt["streams"][0][field] == expected
    assert torch.equal(prefix.states, full.states[:, :8])
    assert torch.equal(prefix.noise, full.noise[:, :, :8])
    assert not torch.equal(full.noise, guide.noise)
    assert "actual_guide_id" not in full.receipt["streams"][0]["request"]["law"]
    assert full.receipt["streams"][0]["request"]["law"]["pool_id"] == "D3-NTC-pool"
    with pytest.raises(ValueError, match="held-out donor"):
        sample_batch([replace(pooled, law=replace(pooled.law, identity=replace(pooled.law.identity, donor="D1")))],
                     context(), particles=8)


def test_pool_definition_binding_changes_draws_and_rejects_missing_identity():
    value = request()
    identity = PooledSourceIdentity("D3", "Stim8hr", "source-a", "replicate-a", "pool", "1" * 64, "2" * 64)
    law = SourceLaw(identity, value.law.states, value.law.weights)
    first = sample_batch([LawRequest(law)], context(), particles=8)
    second = sample_batch([LawRequest(replace(law, identity=replace(identity, definition_sha256="3" * 64)))], context(), particles=8)
    assert not torch.equal(first.noise, second.noise)
    for changes in ({"condition": "Stim48hr"}, {"pool_id": ""}, {"definition_sha256": "unknown"}):
        with pytest.raises(ValueError):
            replace(identity, **changes)
