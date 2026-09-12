"""Positive, null, adversarial and ablation tests for static known-target effects."""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.static_effects import TargetSourceResidual, fit_target_source_residual


def _data(*, features: int = 2) -> dict:
    donors, targets, rows = [], [], []
    for donor in range(3):
        for target in range(2):
            for guide in range(4):
                donors.append(f"D{donor}")
                targets.append(f"T{target}")
                rows.append(f"D{donor}:T{target}:G{guide}")
    rng = np.random.default_rng(42)
    source = rng.normal(size=(len(rows), features))
    return dict(
        source_features=source,
        observed_effects=np.asarray([int(t[-1]) for t in targets]),
        donor_ids=donors,
        target_ids=targets,
        row_ids=rows,
        feature_names=[f"source_{j}" for j in range(features)],
        ridge_alpha=10.0,
    )


def _resign(payload):
    payload["model_id"] = contract_id(payload, id_field="model_id")
    return payload


def test_positive_source_signal_improves_target_baseline():
    data = _data(features=1)
    # All donors contain both positive and negative guide states: the target
    # baseline contains no guide-specific state signal, even during cross-fit.
    x = np.tile([-1.5, -0.5, 0.5, 1.5], 6)[:, None]
    data["source_features"] = x
    data["observed_effects"] = data["observed_effects"] + 2 * x[:, 0]
    data["ridge_alpha"] = 0.1
    fit = fit_target_source_residual(**data)
    baseline = fit_target_source_residual(**data, residual_scale=0)
    # Evaluate fresh feature values, not the training feature grid.
    new_x = np.asarray([[-1.0], [1.0], [-1.0], [1.0]])
    new_targets = ["T0", "T0", "T1", "T1"]
    actual = np.asarray([-2.0, 2.0, -1.0, 3.0])
    prediction = fit.predict(new_x, new_targets, data["feature_names"])
    null = baseline.predict(new_x, new_targets, data["feature_names"])
    assert np.mean((prediction - actual) ** 2) < np.mean((null - actual) ** 2) * 0.001
    assert fit.to_payload()["include_target_indicators"] is True


def test_null_target_only_truth_has_exact_zero_residual():
    data = _data()
    model = fit_target_source_residual(**data)
    record = model.fit_record
    assert np.count_nonzero(record["residual_targets"]) == 0
    assert np.count_nonzero(model.to_payload()["coefficients"]) == 0
    np.testing.assert_array_equal(
        model.predict(data["source_features"], data["target_ids"], data["feature_names"]),
        data["observed_effects"],
    )


def test_scale_zero_is_exact_target_baseline_even_for_extreme_finite_sources():
    data = _data()
    data["observed_effects"] = data["observed_effects"] + data["source_features"][:, 0]
    model = fit_target_source_residual(**data, residual_scale=0)
    expected = [model.to_payload()["target_baseline"][t] for t in data["target_ids"]]
    prediction = model.predict(
        np.full_like(data["source_features"], 1e300), data["target_ids"], data["feature_names"]
    )
    np.testing.assert_array_equal(prediction, expected)


def test_residual_scaling_and_target_indicator_ablation():
    data = _data()
    data["observed_effects"] = data["observed_effects"] + data["source_features"][:, 0]
    full = fit_target_source_residual(**data, residual_scale=1)
    half = fit_target_source_residual(**data, residual_scale=0.5)
    zero = fit_target_source_residual(**data, residual_scale=0)
    args = data["source_features"], data["target_ids"], data["feature_names"]
    np.testing.assert_allclose(half.predict(*args), (full.predict(*args) + zero.predict(*args)) / 2)
    ablation = fit_target_source_residual(**data, include_target_indicators=False)
    assert ablation.fit_record["design_feature_names"] == data["feature_names"]
    assert len(full.fit_record["design_feature_names"]) == len(data["feature_names"]) + 2
    assert not any(name.startswith("donor") for name in full.fit_record["design_feature_names"])


def test_each_residual_baseline_excludes_whole_donor_not_just_own_row():
    data = _data()
    data["observed_effects"] = np.asarray([10 * int(d[-1]) for d in data["donor_ids"]], float)
    model = fit_target_source_residual(**data)
    record = model.fit_record
    expected = {"D0": 15, "D1": 10, "D2": 5}
    assert record["donor_crossfit_baselines"] == [expected[d] for d in record["donor_ids"]]
    changed = copy.deepcopy(data)
    changed["observed_effects"][np.asarray(data["donor_ids"]) == "D0"] += 999
    altered = fit_target_source_residual(**changed).fit_record
    d0 = np.asarray(record["donor_ids"]) == "D0"
    np.testing.assert_array_equal(
        np.asarray(record["donor_crossfit_baselines"])[d0],
        np.asarray(altered["donor_crossfit_baselines"])[d0],
    )


def test_two_training_donors_are_supported_and_per_target_donor_coverage_is_required():
    data = _data()
    keep = np.asarray(data["donor_ids"]) != "D2"
    for key in ("source_features", "observed_effects", "donor_ids", "target_ids", "row_ids"):
        data[key] = np.asarray(data[key])[keep]
    fit_target_source_residual(**data)
    bad = copy.deepcopy(data)
    bad["target_ids"][0] = "TX"
    with pytest.raises(ContractError, match="at least two donors"):
        fit_target_source_residual(**bad)
    bad = copy.deepcopy(data)
    bad["donor_ids"] = ["D0"] * len(data["row_ids"])
    with pytest.raises(ContractError, match="At least two training donors"):
        fit_target_source_residual(**bad)


def test_unequal_donor_guide_counts_use_balanced_baselines_and_training_only_scaling():
    data = dict(
        source_features=np.asarray([[0.0], [0.0], [0.0], [12.0], [2.0], [4.0]]),
        observed_effects=np.asarray([0.0, 0.0, 0.0, 12.0, 2.0, 4.0]),
        donor_ids=["A", "A", "A", "B", "A", "B"],
        target_ids=["X", "X", "X", "X", "Y", "Y"],
        row_ids=["a", "b", "c", "d", "e", "f"],
        feature_names=["source"],
        ridge_alpha=2.0,
    )
    model = fit_target_source_residual(**data)
    payload, record = model.to_payload(), model.fit_record
    assert payload["target_baseline"] == {"X": 6.0, "Y": 3.0}
    np.testing.assert_allclose(record["row_weights"], [0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    assert np.mean(record["row_weights"]) == 1
    assert payload["design_mean"][0] == 4.5
    original = model.to_payload()
    model.predict([[10000.0]], ["X"], ["source"])
    assert model.to_payload() == original


@pytest.mark.parametrize("width", [2, 80])
def test_primal_and_dual_match_explicit_weighted_unpenalized_intercept_solution(width):
    data = _data(features=width)
    data["observed_effects"] = data["observed_effects"] + data["source_features"][:, 0]
    model = fit_target_source_residual(**data)
    payload, record = model.to_payload(), model.fit_record
    one_hot = np.asarray(data["target_ids"])[:, None] == np.asarray(payload["target_ids"])[None, :]
    raw = np.column_stack([data["source_features"], one_hot.astype(float)])
    z = (raw - payload["design_mean"]) / payload["design_scale"]
    w = np.asarray(record["row_weights"])
    residual = np.asarray(record["residual_targets"])
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.diag([0.0] + [data["ridge_alpha"]] * z.shape[1])
    expected = np.linalg.solve(
        design.T @ (w[:, None] * design) + penalty, design.T @ (w * residual)
    )
    np.testing.assert_allclose(payload["intercept"], expected[0], atol=1e-12)
    np.testing.assert_allclose(payload["coefficients"], expected[1:], atol=1e-12)
    assert payload["solver"] == ("weighted_dual" if width == 80 else "weighted_primal")


def test_canonical_row_permutation_identity_and_json_roundtrip():
    data = _data()
    data["observed_effects"] = data["observed_effects"] + data["source_features"][:, 0]
    original = fit_target_source_residual(**data)
    order = np.random.default_rng(9).permutation(len(data["row_ids"]))
    changed = dict(data)
    for name in ("source_features", "observed_effects", "donor_ids", "target_ids", "row_ids"):
        changed[name] = np.asarray(data[name])[order]
    assert fit_target_source_residual(**changed).model_id == original.model_id
    payload = json.loads(json.dumps(original.to_payload(), allow_nan=False))
    restored = TargetSourceResidual.from_payload(payload)
    args = data["source_features"], data["target_ids"], data["feature_names"]
    np.testing.assert_array_equal(original.predict(*args), restored.predict(*args))
    restored.fit_record["residual_targets"][0] = 999
    payload["coefficients"][0] = 999
    assert restored.model_id == original.model_id
    with pytest.raises(FrozenInstanceError):
        restored._payload_json = b"{}"


def test_serialized_hash_tampering_and_rehashed_semantic_tampering_fail():
    model = fit_target_source_residual(**_data())
    payload = model.to_payload()
    payload["coefficients"][0] += 1
    with pytest.raises(IntegrityError):
        TargetSourceResidual.from_payload(payload)
    for mutation in (
        lambda p: p.update(extra=True),
        lambda p: p.update(schema_version=True),
        lambda p: p.update(design_scale=[0] * len(p["design_scale"])),
        lambda p: p["fit_record"]["donor_crossfit_baselines"].__setitem__(0, 999),
        lambda p: p["fit_record"]["input_identity"].update(extra="a" * 64),
        lambda p: p["target_baseline"].update(T0=999),
    ):
        corrupted = model.to_payload()
        mutation(corrupted)
        with pytest.raises(ContractError):
            TargetSourceResidual.from_payload(_resign(corrupted))


@pytest.mark.parametrize("alpha", [0, -1, True, np.inf, np.nan, "1"])
def test_invalid_regularization_fails(alpha):
    data = _data()
    data["ridge_alpha"] = alpha
    with pytest.raises(ContractError):
        fit_target_source_residual(**data)


@pytest.mark.parametrize("scale", [-0.1, 1.1, True, np.nan])
def test_invalid_residual_scale_fails(scale):
    with pytest.raises(ContractError):
        fit_target_source_residual(**_data(), residual_scale=scale)


def test_alignment_finiteness_and_unknown_target_fail_closed():
    data = _data()
    for name, value in (
        ("row_ids", ["duplicate"] * 24),
        ("source_features", np.full((24, 2), np.nan)),
        ("source_features", np.full((24, 2), "1")),
        ("observed_effects", np.zeros(23)),
        ("feature_names", ["same", "same"]),
        ("feature_names", {"source_0", "source_1"}),
        ("feature_names", {"source_0": 0, "source_1": 1}),
    ):
        changed = {**data, name: value}
        with pytest.raises(ContractError):
            fit_target_source_residual(**changed)
    model = fit_target_source_residual(**data, residual_scale=0)
    with pytest.raises(ContractError, match="Unknown target"):
        model.predict([[0, 0]], ["unseen"], data["feature_names"])
    with pytest.raises(ContractError, match="feature order"):
        model.predict([[0, 0]], ["T0"], list(reversed(data["feature_names"])))
    with pytest.raises(ContractError, match="misaligned"):
        model.predict([[0, 0]], ["T0", "T1"], data["feature_names"])


def test_intercept_only_model_and_input_arrays_are_not_mutated():
    data = _data(features=0)
    before = copy.deepcopy(data)
    model = fit_target_source_residual(**data, include_target_indicators=False)
    assert model.to_payload()["solver"] == "intercept_only"
    np.testing.assert_array_equal(model.predict(np.empty((2, 0)), ["T0", "T1"], []), [0.0, 1.0])
    for name in ("source_features", "observed_effects"):
        np.testing.assert_array_equal(data[name], before[name])
