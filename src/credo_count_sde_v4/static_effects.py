"""Known-target, source-conditioned static effect qualification baseline.

This is a standalone statistical object, not an SDE or a checkpoint format.
Effects, feature construction, held-out partitions and biological reference
centering are supplied by an external adapter. Donor IDs are used only to
cross-fit training baselines and balance rows; they are never predictor inputs.
The caller must exclude future/evaluation information from source features.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import canonical_json_bytes, contract_id
from .errors import ContractError, IntegrityError

_SCHEMA = "credo.target_source_residual"
_INDICATOR = "__target_indicator__::"
_WEIGHTING = "equal_target_then_equal_donor_within_target_then_equal_row;mean_weight=1"
_BASELINE = "mean_rows_within_target_donor_then_mean_donors"
_CROSSFIT = "each_training_row_baseline_excludes_its_entire_donor"
_STANDARDIZATION = "weighted_training_only_full_design;std_le_1e-12_scale=1"
_OBJECTIVE = "sum(w*(residual-intercept-standardized_design@beta)^2)+alpha*sum(beta^2)"
_KEYS = {
    "schema_id",
    "schema_version",
    "model_id",
    "feature_names",
    "target_ids",
    "include_target_indicators",
    "ridge_alpha",
    "residual_scale",
    "target_baseline",
    "design_mean",
    "design_scale",
    "coefficients",
    "intercept",
    "solver",
    "fit_record",
}
_RECORD_KEYS = {
    "row_ids",
    "donor_ids",
    "target_ids",
    "observed_effects",
    "donor_crossfit_baselines",
    "residual_targets",
    "row_weights",
    "design_feature_names",
    "input_identity",
    "n_rows",
    "n_source_features",
    "weighting_policy",
    "baseline_policy",
    "residual_target_policy",
    "standardization_policy",
    "solver_objective",
}
_IDENTITY_KEYS = {
    "source_features_sha256",
    "observed_effects_sha256",
    "donor_ids_sha256",
    "target_ids_sha256",
    "row_ids_sha256",
    "feature_names_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _identifiers(value: Sequence[str], name: str, *, unique: bool = False) -> tuple[str, ...]:
    _require(
        not isinstance(value, (str, bytes, set, frozenset, Mapping)),
        f"{name} must be an ordered sequence.",
    )
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ContractError(f"{name} must be an ordered sequence.") from exc
    _require(
        all(isinstance(x, str) and x and x.strip() == x for x in result),
        f"{name} requires nonempty, unpadded string identities.",
    )
    if unique:
        _require(len(set(result)) == len(result), f"{name} must be unique.")
    return result


def _array(value: Any, name: str, ndim: int) -> np.ndarray[Any, Any]:
    try:
        raw = np.asarray(value)
        _require(raw.dtype.kind in "iuf", f"{name} must contain real numeric values.")
        array = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} is not a real numeric array.") from exc
    _require(array.ndim == ndim, f"{name} must have {ndim} dimensions.")
    _require(bool(np.isfinite(array).all()), f"{name} must be finite.")
    return array


def _scalar(value: Any, name: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_)),
        f"{name} must be numeric.",
    )
    number = float(value)
    _require(bool(np.isfinite(number)), f"{name} must be finite.")
    if positive:
        _require(number > 0, f"{name} must be strictly positive.")
    return number


def _array_id(array: np.ndarray[Any, Any]) -> str:
    value = np.asarray(array, dtype="<f8", order="C")
    header = canonical_json_bytes({"dtype": "<f8", "shape": list(value.shape)})
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def _baseline_contract(
    observed: np.ndarray[Any, Any], donors: tuple[str, ...], targets: tuple[str, ...]
) -> tuple[dict[str, float], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    _require(len(set(donors)) >= 2, "At least two training donors are required.")
    target_array, donor_array = np.asarray(targets), np.asarray(donors)
    baseline, crossfit = {}, np.empty(len(observed), dtype=np.float64)
    weights = np.empty(len(observed), dtype=np.float64)
    target_order = sorted(set(targets))
    for target in target_order:
        donor_order = sorted(set(donor_array[target_array == target]))
        _require(
            len(donor_order) >= 2, f"Target {target!r} needs observations from at least two donors."
        )
        donor_means = {
            donor: float(observed[(target_array == target) & (donor_array == donor)].mean())
            for donor in donor_order
        }
        baseline[target] = float(np.mean(list(donor_means.values())))
        for donor in donor_order:
            mask = (target_array == target) & (donor_array == donor)
            crossfit[mask] = float(np.mean([v for d, v in donor_means.items() if d != donor]))
            weights[mask] = len(observed) / (len(target_order) * len(donor_order) * mask.sum())
    _require(bool(np.isfinite(crossfit).all()), "Cross-fitted target baselines overflowed.")
    return baseline, crossfit, weights


def _design(
    source: np.ndarray[Any, Any],
    targets: tuple[str, ...],
    target_order: tuple[str, ...],
    indicators: bool,
) -> np.ndarray[Any, Any]:
    _require(
        set(targets) <= set(target_order),
        "Unknown target: unseen-target prediction is unsupported.",
    )
    if not indicators:
        return source
    one_hot = np.asarray(targets)[:, None] == np.asarray(target_order)[None, :]
    return np.column_stack((source, one_hot.astype(np.float64)))


def _validate_payload(payload: Mapping[str, Any]) -> None:
    _require(
        isinstance(payload, Mapping) and set(payload) == _KEYS,
        "Static-effect payload has missing or unknown fields.",
    )
    _require(
        payload["schema_id"] == _SCHEMA
        and type(payload["schema_version"]) is int
        and payload["schema_version"] == 1,
        "Unsupported static-effect schema.",
    )
    if payload["model_id"] != contract_id(dict(payload), id_field="model_id"):
        raise IntegrityError("Static-effect model content hash differs.")
    names = _identifiers(payload["feature_names"], "feature_names", unique=True)
    _require(not any(x.startswith(_INDICATOR) for x in names), "Reserved feature-name namespace.")
    targets = _identifiers(payload["target_ids"], "target_ids", unique=True)
    _require(bool(targets) and targets == tuple(sorted(targets)), "Target catalog must be sorted.")
    indicators = payload["include_target_indicators"]
    _require(type(indicators) is bool, "include_target_indicators must be Boolean.")
    _scalar(payload["ridge_alpha"], "ridge_alpha", positive=True)
    scale = _scalar(payload["residual_scale"], "residual_scale")
    _require(0 <= scale <= 1, "residual_scale must be between zero and one.")
    baseline = payload["target_baseline"]
    _require(
        isinstance(baseline, Mapping) and set(baseline) == set(targets),
        "Target baseline catalog differs.",
    )
    for value in baseline.values():
        _scalar(value, "target_baseline")
    p = len(names) + (len(targets) if indicators else 0)
    for name in ("design_mean", "design_scale", "coefficients"):
        value = _array(payload[name], name, 1)
        _require(len(value) == p, f"{name} dimension differs from the design.")
        if name == "design_scale":
            _require(bool((value > 0).all()), "Design scales must be strictly positive.")
    intercept = _scalar(payload["intercept"], "intercept")
    record = payload["fit_record"]
    _require(
        isinstance(record, Mapping) and set(record) == _RECORD_KEYS,
        "Fit record has missing or unknown fields.",
    )
    row_ids = _identifiers(record["row_ids"], "row_ids", unique=True)
    donors = _identifiers(record["donor_ids"], "donor_ids")
    fit_targets = _identifiers(record["target_ids"], "fit target_ids")
    n = len(row_ids)
    _require(n > 0 and row_ids == tuple(sorted(row_ids)), "Fit rows must be sorted and nonempty.")
    _require(
        len(donors) == len(fit_targets) == n and set(fit_targets) == set(targets),
        "Fit row identities are misaligned.",
    )
    _require(type(record["n_rows"]) is int and record["n_rows"] == n, "Fit row count differs.")
    _require(
        type(record["n_source_features"]) is int and record["n_source_features"] == len(names),
        "Source feature count differs.",
    )
    expected_names = names + (
        tuple(_INDICATOR + target for target in targets) if indicators else ()
    )
    _require(
        tuple(record["design_feature_names"]) == expected_names, "Design feature order differs."
    )
    policies = {
        "weighting_policy": _WEIGHTING,
        "baseline_policy": _BASELINE,
        "residual_target_policy": _CROSSFIT,
        "standardization_policy": _STANDARDIZATION,
        "solver_objective": _OBJECTIVE,
    }
    _require(all(record[k] == v for k, v in policies.items()), "Fit policy differs from schema.")
    arrays = {
        key: _array(record[key], key, 1)
        for key in (
            "observed_effects",
            "donor_crossfit_baselines",
            "residual_targets",
            "row_weights",
        )
    }
    _require(all(len(value) == n for value in arrays.values()), "Fit numeric rows are misaligned.")
    expected_baseline, crossfit, weights = _baseline_contract(
        arrays["observed_effects"], donors, fit_targets
    )
    _require(
        dict(baseline) == expected_baseline, "Persisted target baselines disagree with fit rows."
    )
    _require(
        np.array_equal(crossfit, arrays["donor_crossfit_baselines"]),
        "Donor-crossfitted baselines disagree with fit rows.",
    )
    _require(np.array_equal(weights, arrays["row_weights"]), "Fit row weights disagree.")
    residual = arrays["observed_effects"] - crossfit
    _require(np.array_equal(residual, arrays["residual_targets"]), "Fit residual targets disagree.")
    _require(
        intercept == float(np.average(residual, weights=weights)), "Residual intercept differs."
    )
    expected_solver = (
        "intercept_only" if p == 0 else "weighted_dual" if p > n else "weighted_primal"
    )
    _require(payload["solver"] == expected_solver, "Solver does not match the recorded design.")
    identity = record["input_identity"]
    _require(
        isinstance(identity, Mapping) and set(identity) == _IDENTITY_KEYS,
        "Input identity has missing or unknown fields.",
    )
    _require(
        all(
            isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
            for x in identity.values()
        ),
        "Input identities must be lowercase SHA-256 digests.",
    )
    expected_ids = {"observed_effects_sha256": _array_id(arrays["observed_effects"])}
    for key, value in (
        ("row_ids", row_ids),
        ("donor_ids", donors),
        ("target_ids", fit_targets),
        ("feature_names", names),
    ):
        expected_ids[key + "_sha256"] = contract_id(list(value))
    _require(
        all(identity[key] == value for key, value in expected_ids.items()),
        "Input identities disagree with fit record.",
    )


@dataclass(frozen=True, slots=True)
class TargetSourceResidual:
    """Immutable, strict, content-addressed known-target statistical predictor.

    Use ``fit_target_source_residual`` or ``from_payload`` to create a model.
    ``to_payload`` and ``fit_record`` return detached copies, never live state.
    """

    _payload_json: bytes

    def __post_init__(self) -> None:
        _require(
            type(self._payload_json) is bytes, "Persisted model state must be immutable bytes."
        )
        try:
            payload = json.loads(self._payload_json)
        except (ValueError, UnicodeError) as exc:
            raise ContractError("Invalid static-effect JSON payload.") from exc
        _validate_payload(payload)
        _require(
            self._payload_json == canonical_json_bytes(payload), "Payload JSON is not canonical."
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TargetSourceResidual:
        """Reject unknown fields, schema changes, altered hashes and inconsistent fit records."""
        _validate_payload(payload)
        return cls(canonical_json_bytes(dict(payload)))

    def to_payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    @property
    def fit_record(self) -> dict[str, Any]:
        return self.to_payload()["fit_record"]

    @property
    def model_id(self) -> str:
        return self.to_payload()["model_id"]

    def predict(
        self, source_features: Any, target_ids: Sequence[str], feature_names: Sequence[str]
    ) -> np.ndarray[Any, Any]:
        """Predict an already-defined effect; no donor identity or future outcome is accepted."""
        payload = self.to_payload()
        names = _identifiers(feature_names, "feature_names", unique=True)
        _require(names == tuple(payload["feature_names"]), "Prediction feature order differs.")
        source = _array(source_features, "source_features", 2)
        targets = _identifiers(target_ids, "target_ids")
        _require(
            source.shape == (len(targets), len(names)), "Prediction source rows are misaligned."
        )
        design = _design(
            source, targets, tuple(payload["target_ids"]), payload["include_target_indicators"]
        )
        baseline = np.asarray(
            [payload["target_baseline"][target] for target in targets], dtype=float
        )
        if payload["residual_scale"] == 0:
            return baseline  # Exact baseline ablation, not baseline + 0 * numerical noise.
        z = (design - np.asarray(payload["design_mean"])) / np.asarray(payload["design_scale"])
        result = baseline + payload["residual_scale"] * (
            payload["intercept"] + z @ np.asarray(payload["coefficients"])
        )
        _require(bool(np.isfinite(result).all()), "Static-effect prediction overflowed.")
        return result


def fit_target_source_residual(
    source_features: Any,
    observed_effects: Any,
    donor_ids: Sequence[str],
    target_ids: Sequence[str],
    row_ids: Sequence[str],
    feature_names: Sequence[str],
    ridge_alpha: float,
    residual_scale: float = 1.0,
    *,
    include_target_indicators: bool = True,
) -> TargetSourceResidual:
    """Fit target baseline + donor-crossfitted, weighted ridge source residual.

    Canonical row ordering makes a jointly permuted input identical. Weights
    give equal mass to targets, then donors within each target, then rows within
    each target/donor, normalized to mean one. Each row's residual target uses
    a baseline that excludes its *whole donor*. All source and optional target
    indicator design columns are standardized using these training weights.
    The intercept is unpenalized; alpha multiplies squared coefficient norm
    alongside the *sum* of weighted squared errors, not their mean.

    No checkpoint or representation contract is changed. This API cannot detect
    mislabeled donor IDs or outcome-derived features: the adapter must bind their
    meaning and perform donor-nested preprocessing and model selection.
    """
    source = _array(source_features, "source_features", 2)
    observed = _array(observed_effects, "observed_effects", 1)
    donors = _identifiers(donor_ids, "donor_ids")
    targets = _identifiers(target_ids, "target_ids")
    rows = _identifiers(row_ids, "row_ids", unique=True)
    names = _identifiers(feature_names, "feature_names", unique=True)
    _require(
        source.shape == (len(rows), len(names))
        and len(rows) > 0
        and len(observed) == len(donors) == len(targets) == len(rows),
        "Training arrays and identities are misaligned or empty.",
    )
    _require(not any(name.startswith(_INDICATOR) for name in names), "Reserved feature namespace.")
    _require(type(include_target_indicators) is bool, "include_target_indicators must be Boolean.")
    alpha = _scalar(ridge_alpha, "ridge_alpha", positive=True)
    scale = _scalar(residual_scale, "residual_scale")
    _require(0 <= scale <= 1, "residual_scale must be between zero and one.")
    order = np.argsort(np.asarray(rows), kind="stable")
    source, observed = source[order], observed[order]
    rows, donors, targets = (tuple(value[i] for i in order) for value in (rows, donors, targets))
    baseline, crossfit, weights = _baseline_contract(observed, donors, targets)
    residual = observed - crossfit
    target_order = tuple(sorted(baseline))
    design = _design(source, targets, target_order, include_target_indicators)
    mean = np.average(design, axis=0, weights=weights)
    variance = np.average((design - mean) ** 2, axis=0, weights=weights)
    _require(
        bool(np.isfinite(mean).all() and np.isfinite(variance).all()),
        "Training design standardization overflowed.",
    )
    deviation = np.sqrt(variance)
    deviation[deviation <= 1e-12] = 1.0
    standardized = (design - mean) / deviation
    intercept = float(np.average(residual, weights=weights))
    a = np.sqrt(weights)[:, None] * standardized
    y = np.sqrt(weights) * (residual - intercept)
    n, p = a.shape
    try:
        if p == 0:
            coefficients, solver = np.empty(0), "intercept_only"
        elif p > n:
            coefficients = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(n), y)
            solver = "weighted_dual"
        else:
            coefficients = np.linalg.solve(a.T @ a + alpha * np.eye(p), a.T @ y)
            solver = "weighted_primal"
    except np.linalg.LinAlgError as exc:
        raise ContractError("Weighted ridge solve failed.") from exc
    _require(bool(np.isfinite(coefficients).all()), "Weighted ridge coefficients are nonfinite.")
    identity = {
        "source_features_sha256": _array_id(source),
        "observed_effects_sha256": _array_id(observed),
    }
    for key, value in (
        ("row_ids", rows),
        ("donor_ids", donors),
        ("target_ids", targets),
        ("feature_names", names),
    ):
        identity[key + "_sha256"] = contract_id(list(value))
    payload = {
        "schema_id": _SCHEMA,
        "schema_version": 1,
        "feature_names": list(names),
        "target_ids": list(target_order),
        "include_target_indicators": include_target_indicators,
        "ridge_alpha": alpha,
        "residual_scale": scale,
        "target_baseline": baseline,
        "design_mean": mean.tolist(),
        "design_scale": deviation.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "solver": solver,
        "fit_record": {
            "row_ids": list(rows),
            "donor_ids": list(donors),
            "target_ids": list(targets),
            "observed_effects": observed.tolist(),
            "donor_crossfit_baselines": crossfit.tolist(),
            "residual_targets": residual.tolist(),
            "row_weights": weights.tolist(),
            "design_feature_names": list(names)
            + (
                [_INDICATOR + target for target in target_order]
                if include_target_indicators
                else []
            ),
            "input_identity": identity,
            "n_rows": n,
            "n_source_features": len(names),
            "weighting_policy": _WEIGHTING,
            "baseline_policy": _BASELINE,
            "residual_target_policy": _CROSSFIT,
            "standardization_policy": _STANDARDIZATION,
            "solver_objective": _OBJECTIVE,
        },
    }
    payload["model_id"] = contract_id(payload)
    return TargetSourceResidual.from_payload(payload)
