"""Fixed-risk selection arithmetic; no data access or candidate-set expansion.

The caller supplies one complete, support-matched score vector per permitted
inner edge. Raw latent risks from distinct representations are never pooled.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .configuration import ALPHA_GRID, SPREAD_GRID, get_configuration


CHANNELS = (
    "state_energy", "spread_absolute_error", "shape_mmd2",
    "count_ce_minus_entropy", "count_quadratic", "joint_mse",
    "target_endpoint_mse", "pooled_control_endpoint_mse", "pooled_effect_mse",
)
FALLBACK_EPSILON = 1e-6
QUALITY_TIE_ATOL = 1e-6


class UnavailableSelection(ValueError):
    pass


def complete_vector(values: Mapping[str, float]):
    if set(values) != set(CHANNELS):
        raise UnavailableSelection("Exactly the nine registered channels are required")
    result = {}
    for name in CHANNELS:
        value = values[name]
        if isinstance(value, bool) or value is None or not math.isfinite(float(value)) or value < 0:
            raise UnavailableSelection(f"Unavailable/invalid risk: {name}")
        result[name] = float(value)
    return result


def normalized_risks(candidate, incumbent, training_scales):
    """Incumbent=1; near-zero baseline uses a positive fitting-only scale.

    Scales are explicit caller-bound positive empirical risks in the channel's
    own units, not outer-donor statistics or a post hoc numerical floor.
    """
    candidate, incumbent = complete_vector(candidate), complete_vector(incumbent)
    scales = complete_vector(training_scales)
    if any(value <= 0 for value in scales.values()):
        raise UnavailableSelection("All fitting-only fallback scales must be positive")
    return {name: 1 + (candidate[name] - incumbent[name]) /
            max(incumbent[name], FALLBACK_EPSILON * scales[name]) for name in CHANNELS}


def normalization_metadata(incumbent, training_scales):
    """Carry absolute-fallback flags alongside allocation-score arithmetic."""
    incumbent, scales = complete_vector(incumbent), complete_vector(training_scales)
    if any(value <= 0 for value in scales.values()):
        raise UnavailableSelection("All fitting-only fallback scales must be positive")
    return {name: {"absolute_fallback": incumbent[name] < FALLBACK_EPSILON * scales[name],
                   "denominator": max(incumbent[name], FALLBACK_EPSILON * scales[name]),
                   "incumbent_risk": incumbent[name], "fitting_scale": scales[name]}
            for name in CHANNELS}


def quality(edge_vectors: Sequence[Mapping[str, float]]):
    if len(edge_vectors) != 3:
        raise UnavailableSelection("Exactly three allowed donor-held-out inner edges required")
    values = []
    for vector in edge_vectors:
        if set(vector) != set(CHANNELS):
            raise UnavailableSelection("Missing normalized channel")
        for value in vector.values():
            if not math.isfinite(float(value)):
                raise UnavailableSelection("Nonfinite normalized risk")
            values.append(float(value))
    return .5 * math.fsum(values) / len(values) + .5 * max(values)


@dataclass(frozen=True)
class CandidateQuality:
    configuration_id: str
    quality: float
    inference_cost: float
    active_parameters: int
    alpha: float = 1.0
    spread_scale: float = 1.0


def choose(rows: Sequence[CandidateQuality]):
    if not rows:
        raise UnavailableSelection("No eligible candidates")
    identities = set()
    for row in rows:
        get_configuration(row.configuration_id)
        if (type(row.alpha) not in (int, float) or row.alpha not in ALPHA_GRID
                or type(row.spread_scale) not in (int, float) or row.spread_scale not in SPREAD_GRID):
            raise UnavailableSelection("Calibration must belong to the frozen grids")
        identity = (row.configuration_id, row.alpha, row.spread_scale)
        if identity in identities or not math.isfinite(row.quality):
            raise UnavailableSelection("Duplicate candidate recipe or invalid quality")
        if (not math.isfinite(row.inference_cost) or row.inference_cost < 0
                or type(row.active_parameters) is not int or row.active_parameters < 1):
            raise UnavailableSelection("Prebound cost/complexity required for tie-breaking")
        identities.add(identity)
    optimum = min(row.quality for row in rows)
    tied = [row for row in rows if row.quality <= optimum + QUALITY_TIE_ATOL]
    # After the requested cost/complexity rules, deterministic identity resolves
    # exact remaining ties. No outcome-dependent tie window or additional seed.
    return min(tied, key=lambda row: (row.inference_cost, row.active_parameters,
                                     row.configuration_id, row.alpha, row.spread_scale))


def nondominated(vectors: Mapping[str, Mapping[str, float]]):
    """Exact Pareto frontier on a common normalized aggregation/support."""
    checked = {name: complete_vector(vector) for name, vector in vectors.items()}
    result = []
    for name, vector in checked.items():
        dominated = any(other != name and
                        all(competitor[channel] <= vector[channel] for channel in CHANNELS) and
                        any(competitor[channel] < vector[channel] for channel in CHANNELS)
                        for other, competitor in checked.items())
        if not dominated:
            result.append(name)
    return sorted(result)


def developmental_margins(donor_normalized_vectors, *, fallback_flags):
    """Descriptive proposed5%/2% rule, not an independent-validation claim.

    Ratios must already be matched inside each donor's representation. Report
    donor-specific vectors too; no cell/guide resampling creates more donors.
    """
    donors = {"D1", "D2", "D3", "D4"}
    if set(donor_normalized_vectors) != donors or set(fallback_flags) != donors:
        raise UnavailableSelection("All four outer donor results required")
    if any(set(flags) != set(CHANNELS) or any(type(value) is not bool for value in flags.values())
           for flags in fallback_flags.values()):
        raise UnavailableSelection("Explicit denominator-fallback flag required for every donor/channel")
    vectors = [complete_vector(values) for values in donor_normalized_vectors.values()]
    mean = {channel: math.fsum(v[channel] for v in vectors) / 4 for channel in CHANNELS}
    unavailable = [channel for channel in CHANNELS if any(flags[channel] for flags in fallback_flags.values())]
    improvement = [channel for channel, risk in mean.items() if channel not in unavailable and risk <= .95]
    adverse = [channel for channel, risk in mean.items() if channel not in unavailable and risk > 1.02]
    return {"mean_normalized_risks": mean, "improved_5percent_channels": improvement,
            "worsened_over_2percent_channels": adverse,
            "percentage_rule_unavailable_channels": unavailable,
            "developmental_rule_met": None if unavailable else bool(improvement) and not adverse,
            "independent_validation": False}
