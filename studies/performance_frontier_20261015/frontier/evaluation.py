"""Bounded FP64 score arithmetic, not data access or assessment admission.

Energy/spread and (1,tanh(z)) joint equations follow the retained
comparator_implementation_v1/comparator_scores.py and
d2_forecast_v3/bundle/full_evaluation.py. Their historical numerical tolerance
is NOT reused: this new study records the bound specified in measure_ops.
Shape uses MAIN_OBJECTIVE_RULES' fixed weighted Gaussian V-statistic MMD2.
The quadratic score is 1+sum(q*q)-2*sum(p*q), not squared composition error.

Inputs are complete supplied laws/catalogs on caller-bound axes. No support
sampling, automatic renormalization, probability floors, decoder calls or
outcome reads occur. All computation is FP64, including converted coordinates.
Weights/log frequencies must already be normalized to FP64 score precision.
Identity tags detect incompatible records; they do not verify custody or bytes.
Arithmetic-only calls may omit identity; production reductions may not.

Reduction weights are supplied explicitly per channel. Their scientific policy
belongs to the prospective measurement contract, not this module. An unavailable
required score makes the aggregate unavailable: no favorable complete-case
renormalization. Independent valid target/control endpoint scores are retained.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Mapping

import torch

from .measure_ops import nonnegative_score, weighted_multiscale_kernel
from .molecular import transform_composition
from .selection import CHANNELS


MMD_BANDWIDTHS = (0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ScoreIdentity:
    fold_id: str
    query_donor: str
    recipe_id: str
    seed: int | None
    particle_prefix: int
    source_sha256: str
    representation_sha256: str
    model_sha256: str
    input_sha256: str
    catalog_sha256: str
    gene_axis_sha256: str
    predictor_kind: str = "single_seed"

    def __post_init__(self):
        for key in ("fold_id", "query_donor", "recipe_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError("Nonempty score identity required: " + key)
        if self.predictor_kind == "single_seed":
            if type(self.seed) is not int or self.seed < 0:
                raise ValueError("Single-seed predictor requires a nonnegative integer seed")
        elif self.predictor_kind == "three_seed_finite_measure_mixture":
            if self.seed is not None:
                raise ValueError("Three-seed finite-measure mixture requires seed=None")
        else:
            raise ValueError("Unknown predictor kind")
        if type(self.particle_prefix) is not int or self.particle_prefix < 1:
            raise ValueError("Positive integer particle prefix required")
        for key, value in asdict(self).items():
            if key.endswith("sha256") and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None):
                raise ValueError("Exact lowercase SHA256 required: " + key)


def _record(identity):
    if identity is not None and not isinstance(identity, ScoreIdentity):
        raise ValueError("ScoreIdentity or explicitly arithmetic-only None required")
    return {"identity": asdict(identity) if identity is not None else None,
            "arithmetic_only": identity is None, "dtype": "float64", "unavailable": {}, "roundoff": {}}


def _missing(record, channels, reason):
    for channel in channels:
        record[channel] = None
        record["unavailable"][channel] = reason


def _tensor(value, name, ndim):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != ndim or min(value.shape, default=1) < 1 or not bool(torch.isfinite(value).all())):
        raise ValueError("Finite nonempty floating tensor required: " + name)
    return value.detach().to(dtype=torch.float64)


def _law(states, weights, name):
    x, w = _tensor(states, name + " states", 2), _tensor(weights, name + " weights", 1)
    if (w.shape != (len(x),) or w.device != x.device or bool((w < 0).any())
            or abs(w.sum().item() - 1.) > 1e-10):
        raise ValueError("Matching normalized nonnegative FP64 law weights required: " + name)
    return x, w


def _matching(*values):
    if len({(v.device, v.shape[1]) for v in values}) != 1:
        raise ValueError("All state laws must have the same coordinate axis and device")


def _check_block(block):
    if type(block) is not int or block < 1:
        raise ValueError("Explicit positive integer pairwise block bound required")


def _risk(record, name, raw, absolute_components):
    evidence = nonnegative_score(raw, component_absolute_sum=float(absolute_components))
    record[name] = evidence["risk"]
    record["roundoff"][name] = evidence


def _pair_mean(x, w, y, v, block, *, self_pair=False):
    total = x.new_zeros(())
    scale = math.sqrt(x.shape[1])
    for i in range(0, len(x), block):
        for j in range(i if self_pair else 0, len(y), block):
            # Direct differences, not the matrix-product Euclidean shortcut.
            distances = torch.cdist(x[i:i+block], y[j:j+block], p=2,
                                   compute_mode="donot_use_mm_for_euclid_dist") / scale
            term = w[i:i+block] @ (distances @ v[j:j+block])
            total += term if not self_pair or i == j else 2 * term
            del distances
    return total


@torch.no_grad()
def score_state_law(prediction, prediction_weights, observed, observed_weights, source, source_weights,
                    *, identity=None, block=128, include_source_diagnostics=True):
    """One complete law; predicted weights must be recomputed for this prefix.

    Missing endpoint support yields unavailable metrics, not an empty law or
    a zero. Zero-weight support rows are retained in support accounting.
    The default also scores the complete source-reference law. Callers needing
    only principal prediction metrics may explicitly disable those diagnostics;
    both source arguments must then be None and no source law is copied/scored.
    """
    _check_block(block)
    record = _record(identity)
    x, w = _law(prediction, prediction_weights, "prediction")
    if type(include_source_diagnostics) is not bool:
        raise ValueError("Explicit boolean include_source_diagnostics required")
    s = u = None
    if include_source_diagnostics:
        s, u = _law(source, source_weights, "source")
        _matching(x, s)
    elif source is not None or source_weights is not None:
        raise ValueError("Disabled source diagnostics require both source arguments to be None")
    if identity is not None and len(x) != identity.particle_prefix:
        raise ValueError("Prediction particle axis differs from the bound prefix")
    record.update(predicted_cells=len(x), source_cells=len(s) if s is not None else None,
                  include_source_diagnostics=include_source_diagnostics, state_dimension=x.shape[1], block=block)
    if observed is None or observed_weights is None:
        if observed is not None or observed_weights is not None:
            raise ValueError("Missing observed law requires both states and weights absent")
        record["observed_cells"] = 0
        _missing(record, ("state_energy", "spread_absolute_error", "shape_mmd2"), "missing_observed_state_law")
        return record
    y, v = _law(observed, observed_weights, "observed")
    _matching(x, y)
    record["observed_cells"] = len(y)
    xx = _pair_mean(x, w, x, w, block, self_pair=True)
    yy = _pair_mean(y, v, y, v, block, self_pair=True)
    xy = _pair_mean(x, w, y, v, block)
    _risk(record, "state_energy", 2*xy-xx-yy, 2*xy+xx+yy)
    record.update(cross_mean=float(xy), predicted_self_mean=float(xx), observed_self_mean=float(yy),
        spread_absolute_error=float(abs(xx-yy)),
        distance="RMS_Euclidean", statistic="weighted_biased_V_energy_including_self_diagonals",
        largest_distance_tile_bytes=8*min(block, max(len(x), len(y), len(s) if s is not None else 0))**2)
    if include_source_diagnostics:
        ss = _pair_mean(s, u, s, u, block, self_pair=True)
        sy = _pair_mean(s, u, y, v, block)
        _risk(record, "source_energy", 2*sy-ss-yy, 2*sy+ss+yy)
        record.update(source_cross_mean=float(sy), source_self_mean=float(ss),
            source_spread_absolute_error=float(abs(ss-yy)),
            energy_difference=record["state_energy"]-record["source_energy"],
            raw_energy_difference=float(2*(xy-sy)-(xx-ss)),
            cross_contribution=float(2*(xy-sy)), spread_contribution=float(-(xx-ss)))
    kernel = lambda a, wa, b, wb: weighted_multiscale_kernel(
        a, wa, b, wb, MMD_BANDWIDTHS, block=block, recompute_backward=False)
    kxx, kyy, kxy = kernel(x, w, x, w), kernel(y, v, y, v), kernel(x, w, y, v)
    _risk(record, "shape_mmd2", kxx+kyy-2*kxy, kxx+kyy+2*kxy)
    record.update(mmd_kernel_xx=float(kxx), mmd_kernel_yy=float(kyy), mmd_kernel_xy=float(kxy),
                  mmd_bandwidths=list(MMD_BANDWIDTHS), mmd_estimator="weighted_biased_V_statistic")
    return record


@torch.no_grad()
def score_catalog(logq, endpoint_counts, *, expected_catalog_size, identity=None):
    """One complete catalog, including zero counts and non-state-modeled rows.

    Caller binds ordered IDs; matching length alone is not an identity proof.
    Finite log probabilities with underflow on exponentiation are rejected,
    preserving the retained strictly-positive forecast scoring contract.
    """
    record = _record(identity)
    qlog = _tensor(logq, "catalog log probabilities", 1)
    if (type(expected_catalog_size) is not int or expected_catalog_size < 1
            or len(qlog) != expected_catalog_size or not isinstance(endpoint_counts, torch.Tensor)
            or endpoint_counts.shape != qlog.shape or endpoint_counts.device != qlog.device
            or endpoint_counts.dtype not in (torch.int32, torch.int64) or bool((endpoint_counts < 0).any())):
        raise ValueError("Exact complete matching catalog and nonnegative integer counts required")
    if abs(torch.logsumexp(qlog, 0).item()) > 1e-10:
        raise ValueError("Complete-catalog log probabilities must already be normalized")
    q = qlog.exp()
    if bool((q <= 0).any()):
        raise ValueError("Underflowed forecast probability is not an observed zero; no floor applied")
    # Python integer accumulation avoids silent int64 total overflow.
    total = sum(endpoint_counts.detach().cpu().tolist())
    if total <= 0 or not math.isfinite(float(total)):
        raise ValueError("Positive finite complete-catalog endpoint total required")
    p = endpoint_counts.to(torch.float64) / total
    positive = p > 0
    entropy = -(p[positive] * p[positive].log()).sum()
    ce = -(p*qlog).sum()
    _risk(record, "count_ce_minus_entropy", ce-entropy, abs(ce)+abs(entropy))
    qq, pq = q.square().sum(), (p*q).sum()
    _risk(record, "count_quadratic", 1+qq-2*pq, 1+qq+2*abs(pq))
    quadratic_excess = (q-p).square().sum()
    _risk(record, "count_quadratic_excess", quadratic_excess, quadratic_excess)
    record.update(cross_entropy_nats=float(ce), endpoint_entropy_nats=float(entropy), catalog_size=len(p),
        captured_cells=total, endpoint_zero_categories=int((endpoint_counts == 0).sum()),
        count_quadratic_empirical_minimum=float(1-p.square().sum()),
        quadratic_definition="1+sum(q_squared)-2*sum(p*q)")
    return record


@torch.no_grad()
def score_joint(prediction, prediction_weights, q, observed, observed_weights, p, *, identity=None):
    """49 moments of the finite measure; the constant mass term stays visible."""
    record = _record(identity)
    if (isinstance(q, bool) or isinstance(p, bool) or not math.isfinite(float(q))
            or not math.isfinite(float(p)) or not 0 < q <= 1 or not 0 <= p <= 1):
        raise ValueError("Full-catalog positive predicted and nonnegative observed frequencies required")
    q, p = float(q), float(p)
    x, w = _law(prediction, prediction_weights, "prediction")
    if x.shape[1] != 48 or (identity is not None and len(x) != identity.particle_prefix):
        raise ValueError("Exact 48-state axis and bound prediction prefix required")
    predicted = torch.cat((x.new_ones(1), (w[:, None]*x.tanh()).sum(0)))
    record.update(joint_mass_mse=float((q-p)**2), features=49, predicted_features=predicted.cpu().tolist())
    if observed is None or observed_weights is None:
        if observed is not None or observed_weights is not None:
            raise ValueError("Missing observed law requires both states and weights absent")
        _missing(record, ("joint_mse", "joint_nonconstant_mse"), "missing_observed_state_law")
        return record
    y, v = _law(observed, observed_weights, "observed")
    _matching(x, y)
    truth = torch.cat((y.new_ones(1), (v[:, None]*y.tanh()).sum(0)))
    squared = (float(q)*predicted-float(p)*truth).square()
    _risk(record, "joint_mse", squared.mean(), squared.mean())
    record.update(joint_nonconstant_mse=float(squared[1:].mean()), observed_features=truth.cpu().tolist())
    return record


@torch.no_grad()
def score_molecular_law(predicted_target, observed_target, predicted_control, observed_control, *, identity=None):
    """One target and one already-defined pooled NTC; no implicit pooling.

    A control score is reusable evidence for one control, not an additional
    independent observation per target. The explicit reducer binds its scope.
    """
    record = _record(identity)
    values = {}
    for name, value in (("predicted_target", predicted_target), ("observed_target", observed_target),
                         ("predicted_control", predicted_control), ("observed_control", observed_control)):
        if value is not None:
            q = _tensor(value, name, 1)
            values[name] = transform_composition(q)
    if len({(v.device, v.shape) for v in values.values()}) > 1:
        raise ValueError("Molecular full-gene axes and devices disagree")
    record["genes"] = next(iter(values.values())).numel() if values else None
    et = ec = None
    if "predicted_target" in values and "observed_target" in values:
        et = values["predicted_target"]-values["observed_target"]
        _risk(record, "target_endpoint_mse", et.square().mean(), et.square().mean())
    else:
        _missing(record, ("target_endpoint_mse",), "missing_target_prediction_or_observation")
    if "predicted_control" in values and "observed_control" in values:
        ec = values["predicted_control"]-values["observed_control"]
        _risk(record, "pooled_control_endpoint_mse", ec.square().mean(), ec.square().mean())
        record["control_endpoint_observations"] = 1
    else:
        _missing(record, ("pooled_control_endpoint_mse",), "missing_pooled_control_prediction_or_observation")
        record["control_endpoint_observations"] = 0
    if et is not None and ec is not None:
        effect = (et-ec).square().mean()
        _risk(record, "pooled_effect_mse", effect, effect)
        record["endpoint_error_interaction"] = float(-2*(et*ec).mean())
        record["effect_decomposition_residual"] = record["pooled_effect_mse"]-(
            record["target_endpoint_mse"]+record["pooled_control_endpoint_mse"]+record["endpoint_error_interaction"])
    else:
        _missing(record, ("pooled_effect_mse",), "missing_complete_target_and_pooled_control_endpoints")
    if "observed_target" in values and "observed_control" in values:
        zero = (values["observed_target"]-values["observed_control"]).square().mean()
        record["zero_effect_mse"] = float(zero)
    else:
        record["zero_effect_mse"] = None
    return record


def reduce_scores(rows: Mapping[str, Mapping], weights: Mapping[str, Mapping[str, float]], *, identity: ScoreIdentity):
    """Fixed per-channel row weights, with complete explicit coverage accounting.

    Empty channel support or unavailable required rows yields None. Missing rows,
    duplicated bindings (not representable in this mapping), foreign identities
    and unnormalized weights are errors. Zero-weight rows remain inventoried but
    do not make an otherwise available channel unavailable.
    """
    result = _record(identity)
    if identity is None or set(weights) != set(CHANNELS):
        raise ValueError("Bound identity and exactly nine explicit channel reductions required")
    used = set().union(*(set(item) for item in weights.values()))
    if set(rows) != used or any(not isinstance(k, str) or not k for k in used):
        raise ValueError("Rows must exactly match the union of fixed channel supports")
    for row in rows.values():
        if row.get("identity") != asdict(identity) or row.get("arithmetic_only") is not False:
            raise ValueError("Foreign or unbound source/fold/prefix/model score identity")
    coverage = {}
    for channel in CHANNELS:
        declared = weights[channel]
        if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in declared.values()):
            raise ValueError("Finite nonnegative explicit reduction weights required")
        if declared and abs(math.fsum(declared.values())-1.) > 1e-10:
            raise ValueError("Fixed channel weights must sum to one; no inferred renormalization")
        missing, scored = {}, {}
        for key, weight in declared.items():
            row = rows[key]
            if channel not in row:
                raise ValueError("Required score key absent: " + channel)
            value = row[channel]
            reason = row.get("unavailable", {}).get(channel)
            if value is None:
                if not isinstance(reason, str) or not reason:
                    raise ValueError("Unavailable channel requires a specific reason")
                missing[key] = reason
            elif reason is not None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Available risks must be finite nonnegative and not labeled unavailable")
            else:
                scored[key] = value
        available_weight = math.fsum(declared[key] for key in scored)
        coverage[channel] = dict(expected_rows=len(declared), available_rows=len(scored),
            available_weight=available_weight, missing_rows=missing, fixed_weights=dict(declared))
        if not declared or any(declared[key] > 0 for key in missing):
            _missing(result, (channel,), "no_declared_support" if not declared else "incomplete_fixed_support")
        else:
            result[channel] = math.fsum(declared[key]*value for key, value in scored.items())
    result["coverage"] = coverage
    result["reduction_sha256"] = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result["selection_available"] = not result["unavailable"]
    return result


def mean_seed_scores(seed_records: Mapping[int, Mapping], *, identities: Mapping[int, ScoreIdentity]):
    """Arithmetic mean of required seed scores; this is NOT an ensemble.

    Exact per-seed identities and identical fixed coverage are mandatory. Seeds
    are numerical repeats, not independent donors. No seed may be discarded.
    """
    if not identities or set(seed_records) != set(identities):
        raise ValueError("Exactly the declared seed inventory is required")
    comparable = []
    for seed, identity in identities.items():
        if isinstance(identity, ScoreIdentity) and identity.predictor_kind != "single_seed":
            raise ValueError("Only single-seed identities may enter seed-score averaging; ensemble scores remain separate")
        if type(seed) is not int or not isinstance(identity, ScoreIdentity) or identity.seed != seed:
            raise ValueError("Declared seed and score identity disagree")
        row = seed_records[seed]
        if row.get("identity") != asdict(identity) or row.get("arithmetic_only") is not False:
            raise ValueError("Unbound or mismatched seed score")
        if any(channel not in row for channel in CHANNELS) or set(row.get("coverage", {})) != set(CHANNELS):
            raise ValueError("Complete explicit seed scores and coverage required")
        common = asdict(identity)
        del common["seed"], common["model_sha256"]
        fixed_support = {channel: (row["coverage"][channel]["expected_rows"],
            row["coverage"][channel]["fixed_weights"]) for channel in CHANNELS}
        comparable.append((common, row.get("reduction_sha256"), fixed_support))
    if any(value != comparable[0] for value in comparable[1:]) or not comparable[0][1] or comparable[0][2] is None:
        raise ValueError("Seeds require identical fold, source, representation, prefix and fixed coverage")
    result = {"seed_identities": {seed: asdict(value) for seed, value in identities.items()},
              "seed_scores": {seed: {key: value.get(key) for key in CHANNELS} for seed, value in seed_records.items()},
              "seed_coverage": {seed: value["coverage"] for seed, value in seed_records.items()},
              "unavailable": {}, "aggregation": "equal_mean_seed_scores_not_prediction_ensemble",
              "biological_replicates_added": 0, "reduction_sha256": comparable[0][1]}
    for channel in CHANNELS:
        values = [seed_records[seed].get(channel) for seed in identities]
        for seed in identities:
            reason = seed_records[seed].get("unavailable", {}).get(channel)
            if (seed_records[seed][channel] is None and (not isinstance(reason, str) or not reason)) or (seed_records[seed][channel] is not None and reason is not None):
                raise ValueError("Seed availability and specific missingness reason disagree")
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0) for v in values):
            raise ValueError("Invalid seed risk")
        if any(value is None for value in values):
            _missing(result, (channel,), "unavailable_required_seed_score")
        else:
            result[channel] = math.fsum(values)/len(values)
    result["selection_available"] = not result["unavailable"]
    return result
