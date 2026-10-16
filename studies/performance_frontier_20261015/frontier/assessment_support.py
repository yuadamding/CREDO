"""Candidate-independent, released paired-support assessment weights.

This is the prospective support policy, not a scorer, outcome opener, model
selector or scientific launch authority. The full captured catalog is retained
even when a guide has no RNA law. Missing predictions never alter these weights.
The returned digest deliberately excludes model/score/seed/prefix identities and
the candidate-dependent forecast seal: the actual release is checked on entry,
but identical observations must yield identical support for every candidate.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from .forecast import ForecastSource
from .selection import CHANNELS
from .truth_compiler import CompiledTruth


PROTOCOL_SHA256 = "2cb6ffc1cbe4931d35c8f03aa85501fb69a3e0b0892c2ce8533f089b5d92d485"
_FACTORY = object()
_STATE = ("state_energy", "spread_absolute_error", "shape_mmd2", "joint_mse")
_MOLECULAR = ("target_endpoint_mse", "pooled_effect_mse")
_COUNT = ("count_ce_minus_entropy", "count_quadratic")


def _rule_binding():
    path = Path(__file__).resolve().parents[1] / "ASSESSMENT_SUPPORT_PROTOCOL.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Exact regular prospective assessment protocol required")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Prospective assessment protocol changed")
    protocol = json.loads(payload)
    if tuple(protocol["implementation"]["required_channels"]) != tuple(CHANNELS):
        raise ValueError("Assessment protocol channel axis differs from scoring")
    return dict(protocol_sha256=PROTOCOL_SHA256,
                implementation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@dataclass(frozen=True, init=False)
class AssessmentSupport:
    """Immutable support inventory; public properties return detached copies."""
    _encoded: bytes

    def __init__(self, document, *, _factory=None):
        if _factory is not _FACTORY:
            raise TypeError("Use build_assessment_support with released observations")
        object.__setattr__(self, "_encoded", json.dumps(document, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode())

    @property
    def identity(self):
        return hashlib.sha256(self._encoded).hexdigest()

    @property
    def record(self):
        return json.loads(self._encoded)

    @property
    def inventory(self):
        return self.record["guide_inventory"]

    @property
    def weights(self):
        # reduce_scores canonically JSON-encodes this exact dict-of-dicts. The
        # returned copy can be serialized without exposing stored mutable state.
        return self.record["channel_weights"]

    @property
    def required_row_ids(self):
        return tuple(self.record["required_row_ids"])


def _physical_reference(reference):
    # Evaluation access/latent identities contain the release. Preserve actual
    # population/row/input identity, not a model-specific custody digest.
    return {name: getattr(reference, name) for name in (
        "donor", "condition", "source_id", "replicate_id", "source_input_sha256",
        "row_identity_sha256")}


def build_assessment_support(source, truth):
    """Bind the nine registered reductions on one released donor/fold.

    No prediction or score argument exists. State and molecular score support
    depends only on positive-RNA observations. Captured count zeros are genuine
    catalog observations, never missing law imputations. A missing NTC pool
    leaves control/effect weights intact, so the evaluator reports unavailable.
    """
    if type(source) is not ForecastSource or type(truth) is not CompiledTruth:
        raise TypeError("Validated ForecastSource and factory CompiledTruth required")
    rule = _rule_binding()
    source.assert_unchanged()
    truth.assert_unchanged()
    endpoint = truth.population
    release = truth.release.expectation
    if (source.fold != truth.fold or source.catalog != truth.catalog
            or source.observation_binding != truth.observation_binding
            or source.feature_axis_sha256 != truth.feature_axis_sha256
            or source.reference.donor != endpoint.reference.donor
            or source.reference.condition != "Stim8hr" or endpoint.reference.condition != "Stim48hr"
            or source.reference.source_access_sha256 != release.query_access_sha256
            or source.fold.identity != release.fold_sha256
            or source.observation_binding.artifact_sha256 != release.representation_sha256):
        raise ValueError("Source/truth fold, query, catalog, representation or release differ")
    catalog = source.catalog
    g = len(catalog)
    if endpoint.mean_compositions.shape[0] != g or endpoint.positive_counts.shape != (g,):
        raise ValueError("Complete endpoint observation support must use the same catalog")
    source_counts = [int(v) for v in source.captured_counts]
    endpoint_counts = [int(v) for v in endpoint.captured_counts]
    source_positive = [int(v) for v in source.positive_counts]
    endpoint_positive = [int(v) for v in endpoint.positive_counts]
    paired = tuple(i for i in range(g) if source_positive[i] > 0 and endpoint_positive[i] > 0)
    targets = {}
    for i in paired:
        if not catalog[i].is_control:
            targets.setdefault(catalog[i].target_id, []).append(i)
    state_weights = {f"guide:{i}": 1.0/len(paired) for i in paired}
    molecular_weights = {f"guide:{i}": 1.0/(len(targets)*len(indices))
                         for indices in targets.values() for i in indices}
    weights = {channel: dict(state_weights) for channel in _STATE}
    weights.update({channel: dict(molecular_weights) for channel in _MOLECULAR})
    weights.update({channel: {"catalog": 1.0} for channel in _COUNT})
    weights["pooled_control_endpoint_mse"] = {"pooled_control": 1.0}
    if set(weights) != set(CHANNELS):
        raise ValueError("Exactly nine registered assessment channels required")

    controls = tuple(i for i, guide in enumerate(catalog) if guide.is_control)
    source_ntc = sum(source_positive[i] for i in controls)
    endpoint_ntc = sum(endpoint_positive[i] for i in controls)
    pool_reason = ("missing_source_and_endpoint_positive_rna_control_pools" if not source_ntc and not endpoint_ntc
                   else "missing_source_positive_rna_control_pool" if not source_ntc
                   else "missing_endpoint_positive_rna_control_pool" if not endpoint_ntc else None)
    inventory = []
    for i, guide in enumerate(catalog):
        has_source, has_endpoint = source_positive[i] > 0, endpoint_positive[i] > 0
        common = has_source and has_endpoint
        category = "paired" if common else "source_only" if has_source else "endpoint_only" if has_endpoint else "neither"
        state_reason = (None if common else "missing_endpoint_positive_rna" if has_source
                        else "missing_source_positive_rna" if has_endpoint
                        else "missing_source_and_endpoint_positive_rna")
        target_reason = "control_guide_not_target_endpoint" if guide.is_control else state_reason
        eligible = {channel: common for channel in _STATE}
        eligible.update({channel: common and not guide.is_control for channel in _MOLECULAR})
        eligible.update({channel: True for channel in _COUNT})
        eligible["pooled_control_endpoint_mse"] = False
        unavailable = {channel: state_reason for channel in _STATE}
        unavailable.update(target_endpoint_mse=target_reason,
            pooled_effect_mse=target_reason or pool_reason,
            pooled_control_endpoint_mse="standalone_pooled_control_not_guide_score")
        unavailable.update({channel: None for channel in _COUNT})
        inventory.append(dict(catalog_index=i, row_id=f"guide:{i}", **asdict(guide),
            source_captured_count=source_counts[i], endpoint_captured_count=endpoint_counts[i],
            source_positive_rna_count=source_positive[i], endpoint_positive_rna_count=endpoint_positive[i],
            source_capture_zero=source_counts[i] == 0, endpoint_capture_zero=endpoint_counts[i] == 0,
            forecast_expected_from_source=has_source, support_category=category,
            paired=common, source_only=has_source and not has_endpoint,
            endpoint_only=has_endpoint and not has_source, neither=not has_source and not has_endpoint,
            source_control_positive_rna_weight=(source_positive[i]/source_ntc if guide.is_control and source_ntc else 0.0),
            endpoint_control_positive_rna_weight=(endpoint_positive[i]/endpoint_ntc if guide.is_control and endpoint_ntc else 0.0),
            eligible_for_channel=eligible, observation_unavailable_reason=unavailable))

    unavailable = {channel: "no_paired_guides" for channel in _STATE if not paired}
    unavailable.update({channel: "no_paired_targeting_guides" for channel in _MOLECULAR if not targets})
    if pool_reason:
        unavailable["pooled_control_endpoint_mse"] = pool_reason
        if targets:
            unavailable["pooled_effect_mse"] = pool_reason
    if not sum(endpoint_counts):
        unavailable.update({channel: "no_endpoint_captured_total" for channel in _COUNT})
    required_rows = [f"guide:{i}" for i in paired] + ["catalog", "pooled_control"]
    document = dict(schema="credo.frontier.common-assessment-support.v1", rule_binding=rule,
        bindings=dict(fold=asdict(source.fold), fold_sha256=source.fold.identity,
            query_donor=source.reference.donor, query_access_sha256=release.query_access_sha256,
            catalog_sha256=source.catalog_sha256, feature_axis_sha256=source.feature_axis_sha256,
            observation_binding=asdict(source.observation_binding),
            source_population=_physical_reference(source.reference),
            endpoint_population=_physical_reference(endpoint.reference), synthetic_fixture=source.synthetic_fixture),
        complete_forecast_release_verified=True, candidate_outputs_used=False,
        guide_inventory=inventory, required_row_ids=required_rows, channel_weights=weights,
        unavailable_channels_by_observation_support=unavailable,
        coverage=dict(catalog_guides=g, forecast_source_guides=sum(v > 0 for v in source_positive),
            endpoint_positive_rna_guides=sum(v > 0 for v in endpoint_positive), paired_guides=len(paired),
            paired_targeting_guides=sum(len(v) for v in targets.values()), represented_targets=len(targets),
            source_only_indices=[i for i, row in enumerate(inventory) if row["source_only"]],
            endpoint_only_indices=[i for i, row in enumerate(inventory) if row["endpoint_only"]],
            neither_indices=[i for i, row in enumerate(inventory) if row["neither"]],
            source_captured_total=sum(source_counts), endpoint_captured_total=sum(endpoint_counts),
            source_capture_zero_indices=[i for i, n in enumerate(source_counts) if n == 0],
            endpoint_capture_zero_indices=[i for i, n in enumerate(endpoint_counts) if n == 0]),
        control_pool=dict(row_id="pooled_control", catalog_indices=list(controls),
            source_positive_rna_cells=source_ntc, endpoint_positive_rna_cells=endpoint_ntc,
            source_positive_rna_guides=sum(source_positive[i] > 0 for i in controls),
            endpoint_positive_rna_guides=sum(endpoint_positive[i] > 0 for i in controls),
            expected_forecast_available=source_ntc > 0, observed_endpoint_available=endpoint_ntc > 0,
            common_pool_available=source_ntc > 0 and endpoint_ntc > 0, unavailable_reason=pool_reason,
            source_weight_without_endpoint_rna=sum(row["source_control_positive_rna_weight"]
                for row in inventory if row["is_control"] and not row["endpoint_positive_rna_count"]),
            endpoint_weight_without_source_rna=sum(row["endpoint_control_positive_rna_weight"]
                for row in inventory if row["is_control"] and not row["source_positive_rna_count"]),
            weighting="equal_positive_RNA_cells_at_each_time_separately_not_captured_counts",
            missing_individual_ntc_does_not_replace_or_renormalize_a_fixed_guide_mixture=True),
        reduction_policy=dict(candidate_missing_prediction="required_score_unavailable_fixed_weight_retained",
            empty_support="unavailable_not_zero", count_normalization="complete_catalog_never_paired_panel",
            target_reduction="equal_guides_within_target_then_equal_targets",
            state_reduction="equal_paired_guide_laws_including_NTC", control_reduction="one_standalone_pool",
            molecular_companion="new_pooled_NTC_analysis_historical_primary_unchanged"),
        optimizer_calls=0, scientific_launch_authority=False)
    return AssessmentSupport(document, _factory=_FACTORY)
