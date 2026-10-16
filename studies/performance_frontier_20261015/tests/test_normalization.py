"""Small fabricated CPU normalizers; no optimizer, data or historical models."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from frontier.geometry import GeometryOptions, RaggedSinkhornGeometry, geomloss_backend
from frontier.normalization import compute_normalizers, reduce_normalizers, _sinkhorn
from frontier.selection import CHANNELS
from test_fitting_provider import fixture


torch.set_num_threads(2)
PROTOCOL = (Path(__file__).resolve().parents[1] / "NORMALIZATION_PROTOCOL.json").read_bytes()


def rows():
    cache = fixture()[0]
    laws = []
    for index in cache.paired:
        donor, guide = cache.laws[index]
        entry = cache.manifest.catalog[guide]
        donor_value = 1. if donor == "D3" else 3.
        row = dict(donor=donor, guide_index=guide, guide_id=entry.guide_id,
            target_id=entry.target_id, is_control=entry.is_control,
            state_energy=donor_value, spread_absolute_error=2. * donor_value,
            shape_mmd2=3. * donor_value, joint_mse=4. * donor_value,
            sinkhorn=9. * donor_value)
        if not entry.is_control:
            row.update(target_endpoint_mse=(2. if entry.target_id == "A" else 8.) * donor_value,
                       pooled_effect_mse=(6. if entry.target_id == "A" else 10.) * donor_value)
        laws.append(row)
    donors = [dict(donor=d, catalog_categories=7, count_ce_minus_entropy=v,
                   count_quadratic=2. * v, pooled_control_endpoint_mse=3. * v)
              for d, v in (("D3", 2.), ("D4", 6.))]
    return cache, laws, donors


def reduce(cache, laws, donors):
    return reduce_normalizers(cache, laws, donors, sampler_binding={"fabricated": True},
                              producer_sha256="a" * 64)


def test_distinct_training_and_selection_scales_and_target_macro():
    cache, laws, donors = rows()
    result = reduce(cache, laws, donors)
    assert set(result["selection_values"]) == set(CHANNELS)
    assert result["training_values"] == dict(state=18., count=4., shape=6., endpoint=11., effect=16.)
    assert result["selection_values"]["state_energy"] == 2.
    assert result["selection_values"]["target_endpoint_mse"] == 10.
    assert result["selection_values"]["pooled_control_endpoint_mse"] == 12.
    assert result["scientific_updates"] == 0
    assert result["query_outcomes_used"] is False
    assert result["scientific_launch_authority"] is False


def test_input_order_does_not_change_result_or_support_identity():
    cache, laws, donors = rows()
    assert reduce(cache, laws, donors) == reduce(cache, list(reversed(laws)), list(reversed(donors)))


@pytest.mark.parametrize("mutation", ["missing_law", "duplicate_law", "foreign_donor", "wrong_guide",
                                     "missing_target_score", "missing_donor", "short_catalog"])
def test_no_subset_or_unavailable_channel_repair(mutation):
    cache, laws, donors = rows()
    if mutation == "missing_law": laws.pop()
    elif mutation == "duplicate_law": laws.append(deepcopy(laws[0]))
    elif mutation == "foreign_donor": laws[0]["donor"] = "D1"
    elif mutation == "wrong_guide": laws[0]["guide_id"] = "wrong"
    elif mutation == "missing_target_score": laws[0]["pooled_effect_mse"] = None
    elif mutation == "missing_donor": donors.pop()
    elif mutation == "short_catalog": donors[0]["catalog_categories"] = 6
    with pytest.raises(ValueError):
        reduce(cache, laws, donors)


@pytest.mark.parametrize("value", [0., float("nan"), -1., True])
def test_nonpositive_or_invalid_scale_is_not_floored(value):
    cache, laws, donors = rows()
    for law in laws:
        law["state_energy"] = value
    with pytest.raises(ValueError):
        reduce(cache, laws, donors)


def test_sinkhorn_roundoff_preserved_and_larger_negative_rejected():
    assert _sinkhorn(-1e-6) == dict(raw_risk=-1e-6, risk=0., negative_roundoff_corrected=True,
                                  negative_tolerance=1e-5, training_loss_modified=False)
    with pytest.raises(ValueError):
        _sinkhorn(-1.0001e-5)


def geometry():
    return RaggedSinkhornGeometry(options=GeometryOptions(100000, 1, geomloss_backend().identity["version"]))


def compute(cache, **kwargs):
    return compute_normalizers(cache, geometry(), study_id="fabricated-normalization", study_seed=731,
        producer_sha256="a" * 64, sampler_source_sha256="b" * 64,
        protocol_payload=PROTOCOL, source_particles=8, block=4, **kwargs)


def test_actual_fabricated_calculation_retains_full_support_and_repeatability():
    cache = fixture()[0]
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
    first, second = compute(cache), compute(cache)
    assert first == second
    assert len(first["per_law"]) == len(cache.paired) == 10
    assert all(row["catalog_categories"] == 7 for row in first["per_donor"])
    assert all(v > 0 for v in first["training_values"].values())
    assert all(v > 0 for v in first["selection_values"].values())
    assert first["sampler_binding"]["context"]["purpose"] == "scales"
    assert first["sampler_binding"]["context"]["model_seed"] == 0
    assert random.getstate() == python_state
    now = np.random.get_state()
    assert now[0] == numpy_state[0] and np.array_equal(now[1], numpy_state[1]) and now[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    cache.assert_unchanged()


def test_fixture_cannot_route_to_gpu_or_exceed_particle_scope():
    cache = fixture()[0]
    with pytest.raises(ValueError, match="bounded CPU"):
        compute(cache, device="cuda:0")
    with pytest.raises(ValueError, match="Exact cache"):
        compute_normalizers(cache, geometry(), study_id="x", study_seed=0,
            producer_sha256="a" * 64, sampler_source_sha256="b" * 64, protocol_payload=b"{}",
            source_particles=8, block=4)
