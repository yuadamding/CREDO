"""Source-only scale-stream parity; fabricated CPU and zero optimizer calls."""
from unittest.mock import patch

import pytest
import torch

from frontier import sampling
from frontier.sampling import sample_batch, sample_scale_sources
from test_sampling import context, request


def fixed(**kwargs):
    return context(**(dict(purpose="scales", accepted_update=0) | kwargs))


def test_exact_source_parity_without_a_noise_generator():
    inputs = (request(), request("g2", donor="D4", weights=(0, .5, .5, 0)))
    expected = sample_batch(inputs, fixed(), particles=128)
    with patch.object(sampling, "_generator", wraps=sampling._generator) as generator:
        actual = sample_scale_sources(inputs, fixed(), particles=128)
    assert torch.equal(actual.states, expected.states)
    assert torch.equal(actual.source_indices, expected.source_indices)
    assert generator.call_count == 2
    assert all(call.args[2] == "source_categorical" for call in generator.call_args_list)
    assert actual.receipt["standard_normal_draws"] == 0
    assert expected.receipt["standard_normal_draws"] == 2 * 128 * 8 * 48


def test_prefix_order_and_configuration_parity():
    inputs = (request(), request("g2", donor="D4"))
    full = sample_scale_sources(inputs, fixed(), particles=128)
    other = sample_scale_sources(inputs[::-1], fixed(configuration_id="F5-V2"), particles=32)
    assert torch.equal(full.states[:, :32], other.states.flip(0))
    assert torch.equal(full.source_indices[:, :32], other.source_indices.flip(0))


@pytest.mark.parametrize("changes", [{"purpose": "forecast"}, {"accepted_update": 1},
                                      {"generation": 1}, {"model_seed": 1}])
def test_fixed_normalization_draw_identity(changes):
    with pytest.raises(ValueError, match="Fixed fitting-scale"):
        sample_scale_sources((request(),), fixed(**changes), particles=8)


def test_no_heldout_donor_or_duplicate_or_oversized_request():
    with pytest.raises(ValueError, match="held-out"):
        sample_scale_sources((request(donor="D1"),), fixed(), particles=8)
    with pytest.raises(ValueError, match="held-out"):
        sample_batch((request(donor="D1"),), fixed(), particles=8)
    with pytest.raises(ValueError, match="Duplicate"):
        sample_scale_sources((request(), request()), fixed(), particles=8)
    with pytest.raises(ValueError, match="bounded"):
        sample_scale_sources((request(),), fixed(), particles=8193)
