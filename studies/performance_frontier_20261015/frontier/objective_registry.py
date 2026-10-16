"""Pure bindings for the prospectively frozen frontier objective.

No files, populations, GPU devices or optimizers are accessed. The caller must
verify the supplied artifact bytes, source roles and numerical producer before
scientific admission. Hash syntax and positive supplied scales DO NOT establish
that scales were computed correctly, that an inventory is complete, or that a
job may launch. The returned protocol is only a metadata-bound construction.

The exact rules bytes are pinned deliberately: changing their scientific rules
requires a separately reviewed registry revision, not a caller-selected hash.
Support factors use only explicit fitting-source count metadata. They are not
calculated from query support, endpoint support or target molecular outcomes.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType

from .configuration import DONORS, get_configuration


RULES_SHA256 = "d490943dbd05a4a18e01bf4853f7a463ea71e6d5aeaff8aa7fae7c383a6e9ecf"
SCALE_NAMES = ("state", "count", "shape", "endpoint", "effect")


def _identity(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("An exact lowercase SHA-256 identity is required")
    return value


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _donors(value):
    if (type(value) is not tuple or len(value) not in (2, 3)
            or value != tuple(donor for donor in DONORS if donor in value)):
        raise ValueError("Canonical two/three-donor fitting subset required")
    return value


def _ids(value, *, label):
    if (type(value) is not tuple or not value
            or any(type(item) is not str or not item for item in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Distinct explicit {label} identifiers required")
    return value


@dataclass(frozen=True)
class ObjectiveRules:
    """Exact immutable JSON bytes; parsing does not grant execution authority."""

    payload: bytes
    expected_sha256: str

    def __post_init__(self):
        _identity(self.expected_sha256)
        if (type(self.payload) is not bytes
                or self.expected_sha256 != RULES_SHA256
                or sha256(self.payload).hexdigest() != RULES_SHA256):
            raise ValueError("Objective rules differ from the prospectively frozen registry")
        if self.document["schema"] != "credo.frontier.main-objective-rules.v1":
            raise ValueError("Wrong objective-rule schema")

    @property
    def document(self):
        # A fresh object prevents mutations of a returned dict from changing
        # later construction. There is no automatic read of a local rule file.
        return json.loads(self.payload)

    @property
    def identity(self):
        return self.expected_sha256


@dataclass(frozen=True)
class FittingScaleBinding:
    """Supplied numerical values and provenance claims, not numerical proof.

    All five values are required even for F0: one complete scale package is
    shared by every family/seed for the exact fitting subset. Inventory and
    artifact hashes must be checked by the numerical input owner. The observation
    technical holdout's permitted post-selection use is defined in the rules;
    it does not permit query-donor observations in these values.
    """

    fitting_donors: tuple[str, ...]
    rules_sha256: str
    observation_artifact_sha256: str
    fitting_input_sha256: str
    support_inventory_sha256: str
    scale_artifact_sha256: str
    producer_sha256: str
    sampler_sha256: str
    values: Mapping

    def __post_init__(self):
        _donors(self.fitting_donors)
        for name in ("rules_sha256", "observation_artifact_sha256", "fitting_input_sha256",
                     "support_inventory_sha256", "scale_artifact_sha256", "producer_sha256",
                     "sampler_sha256"):
            _identity(getattr(self, name))
        if self.rules_sha256 != RULES_SHA256:
            raise ValueError("Scale producer rules differ from this objective registry")
        if not isinstance(self.values, Mapping) or set(self.values) != set(SCALE_NAMES):
            raise ValueError("Exactly all five fitting-only normalization scales are required")
        values = {}
        for name in SCALE_NAMES:
            value = self.values[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Supplied scales must be finite and strictly positive; no floor")
            values[name] = float(value)
        object.__setattr__(self, "values", MappingProxyType(values))

    def as_binding(self):
        return dict(schema="credo.frontier.fitting-scale-binding.v1",
                    fitting_donors=list(self.fitting_donors),
                    rules_sha256=self.rules_sha256,
                    observation_artifact_sha256=self.observation_artifact_sha256,
                    fitting_input_sha256=self.fitting_input_sha256,
                    support_inventory_sha256=self.support_inventory_sha256,
                    scale_artifact_sha256=self.scale_artifact_sha256,
                    producer_sha256=self.producer_sha256, sampler_sha256=self.sampler_sha256,
                    source_condition="Stim8hr", endpoint_condition="Stim48hr",
                    values=dict(self.values), numerical_values_verified_by_this_helper=False,
                    scientific_launch_authority=False)

    @property
    def identity(self):
        return _digest(self.as_binding())


def build_objective_protocol(configuration_id, *, rules, scales, fitting_donors,
                             observation_artifact_sha256, fitting_input_sha256):
    """Construct the existing ObjectiveProtocol after exact context matching.

    Importing its class occurs only here; support/identity helpers are stdlib
    only. No tensor, model, optimizer or decoder is constructed.
    """
    if not isinstance(rules, ObjectiveRules) or not isinstance(scales, FittingScaleBinding):
        raise ValueError("Registered rules and explicit fitting-only scale binding required")
    _donors(fitting_donors)
    _identity(observation_artifact_sha256)
    _identity(fitting_input_sha256)
    if (scales.rules_sha256 != rules.identity or scales.fitting_donors != fitting_donors
            or scales.observation_artifact_sha256 != observation_artifact_sha256
            or scales.fitting_input_sha256 != fitting_input_sha256):
        raise ValueError("Scale fold, selected observation artifact or fitting inputs differ")
    configuration = get_configuration(configuration_id)
    document = rules.document
    core = document["core"]
    auxiliary = document["family_auxiliaries"][configuration.family]
    regularizers = document["regularizers"]
    from .training_objective import ObjectiveProtocol
    return ObjectiveProtocol(
        protocol_id="frontier-main-objective-v1:" + _digest(dict(
            configuration_id=configuration.configuration_id, rules_sha256=rules.identity,
            scale_binding=scales.as_binding())),
        family=configuration.family, training_donors=fitting_donors,
        state_coefficient=core["state_coefficient"], count_coefficient=core["count_coefficient"],
        shape_coefficient=auxiliary["shape"], endpoint_coefficient=auxiliary["both_endpoints"],
        effect_coefficient=auxiliary["contrast"],
        **{f"{key}_scale": value for key, value in scales.values.items()},
        deterministic_action=regularizers["deterministic_action"],
        stochastic_action=regularizers["stochastic_action"], reaction_action=regularizers["reaction_action"],
        reference_embedding_penalty=regularizers["reference_embedding"],
        target_embedding_penalty=regularizers["target_embedding_F3_F5"] if configuration.target_residual else 0.,
        target_growth_penalty=regularizers["target_growth_F3_F5"] if configuration.target_residual else 0.,
        shape_bandwidths=tuple(document["shape"]["Gaussian_bandwidths"]))


@dataclass(frozen=True)
class SourceSupportRecord:
    donor_id: str
    condition: str
    guide_id: str
    target_id: str
    positive_rna_cells: int
    is_control: bool

    def __post_init__(self):
        if (self.donor_id not in DONORS or self.condition != "Stim8hr"
                or type(self.guide_id) is not str or not self.guide_id
                or type(self.target_id) is not str or not self.target_id
                or type(self.positive_rna_cells) is not int or self.positive_rna_cells < 0
                or type(self.is_control) is not bool):
            raise ValueError("Exact source-only donor/guide/target/control and nonnegative RNA count required")


@dataclass(frozen=True)
class TargetSupportFactors:
    fitting_donors: tuple[str, ...]
    source_inventory_sha256: str
    source_rows_sha256: str
    rules_sha256: str
    control_target_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    support_scores: tuple[int, ...]
    factors: tuple[float, ...]

    def __post_init__(self):
        _donors(self.fitting_donors)
        _identity(self.source_inventory_sha256)
        _identity(self.source_rows_sha256)
        if self.rules_sha256 != RULES_SHA256:
            raise ValueError("Target support rules differ from the bound registry")
        _ids(self.control_target_ids, label="control-target")
        if (type(self.target_ids) is not tuple
                or self.target_ids != tuple(sorted(set(self.target_ids)))
                or any(type(target) is not str or not target for target in self.target_ids)
                or set(self.target_ids) & set(self.control_target_ids)
                or type(self.support_scores) is not tuple or type(self.factors) is not tuple
                or len(self.target_ids) != len(self.support_scores) or len(self.target_ids) != len(self.factors)):
            raise ValueError("Canonical noncontrol target/support factor alignment required")
        for score, factor in zip(self.support_scores, self.factors):
            if (type(score) is not int or score < 1 or type(factor) is not float
                    or factor != math.sqrt(score / (score + 256))):
                raise ValueError("Support factor must equal the fixed source-support equation")

    def factor_for(self, target_id):
        if type(target_id) is not str or not target_id:
            raise ValueError("Explicit target identity required")
        try:
            return self.factors[self.target_ids.index(target_id)]
        except ValueError:
            return 0.0  # Exact zero for controls and unseen/zero-source targets.

    @property
    def identity(self):
        return _digest(asdict(self))


def source_support_factors(records, *, fitting_donors, control_target_ids,
                           source_inventory_sha256, rules):
    """Count each fitting donor-guide once, cap at128, then sqrt(n/(n+256)).

    Input order has no effect. Zero-source targets get no private parameter row;
    controls are excluded, and factor_for returns exact zero for both classes.
    The caller must verify inventory completeness against the bound source
    manifest. This helper checks supplied row consistency, not missing rows in
    an artifact it deliberately does not open.
    """
    _donors(fitting_donors)
    _ids(control_target_ids, label="control-target")
    _identity(source_inventory_sha256)
    if not isinstance(rules, ObjectiveRules):
        raise ValueError("Exact frozen objective rules required")
    records = tuple(records)
    if not records or not all(isinstance(row, SourceSupportRecord) for row in records):
        raise ValueError("Explicit nonempty fitting-source support metadata required")
    if {row.donor_id for row in records} != set(fitting_donors):
        raise ValueError("Every fitting donor, and no query donor, must supply source metadata")
    seen, guide_metadata, scores = set(), {}, {}
    cap = rules.document["target_support"]["per_donor_guide_cell_cap"]
    controls = set(control_target_ids)
    for row in records:
        key = (row.donor_id, row.guide_id)
        if key in seen:
            raise ValueError("Repeated donor-guide support would double count a source law")
        seen.add(key)
        if row.is_control != (row.target_id in controls):
            raise ValueError("Source control flag disagrees with the bound control-target inventory")
        metadata = (row.target_id, row.is_control)
        if row.guide_id in guide_metadata and guide_metadata[row.guide_id] != metadata:
            raise ValueError("An actual guide changed target/control identity across fitting donors")
        guide_metadata[row.guide_id] = metadata
        if not row.is_control and row.positive_rna_cells:
            scores[row.target_id] = scores.get(row.target_id, 0) + min(row.positive_rna_cells, cap)
    target_ids = tuple(sorted(scores))
    ordered = sorted(records, key=lambda row: (row.donor_id, row.guide_id))
    return TargetSupportFactors(
        fitting_donors, source_inventory_sha256, _digest([asdict(row) for row in ordered]),
        rules.identity, tuple(sorted(control_target_ids)), target_ids,
        tuple(scores[target] for target in target_ids),
        tuple(math.sqrt(scores[target] / (scores[target] + 256)) for target in target_ids))
