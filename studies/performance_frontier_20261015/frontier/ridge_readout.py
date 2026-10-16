"""Decoder-bypassing molecular ridge primitives; no file access or launch owner.

These routines do not grant input access, fit admission or prediction custody.
The caller must bind fitting-only arrays, exact query sources and output limits.
The deterministic solve is fitting work, even though it uses no optimizer.
Tests which mock that solve do not qualify numerical fitting.
"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .configuration import DONORS
from .planner import inner_edges


FEATURES = 194
LAMBDA_GRID = (0.001, 0.01, 0.1, 1.0)
MAX_OBSERVED_READOUT = math.log1p(10000)


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase SHA-256 identity required')
    return value


def _ids(values):
    if (type(values) is not tuple or not values or len(set(values)) != len(values)
            or any(not isinstance(v, str) or not v.strip() for v in values)):
        raise ValueError('Nonempty ordered distinct string identities required')
    return values


def _positive_int(value):
    if type(value) is not int or value < 1:
        raise ValueError('Explicit positive integer resource bound required')
    return value


def _array(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype.kind != 'f'
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError(f'Finite floating array with exact {name} shape required')
    return value


def _observed(value, shape):
    _array(value, shape, 'observed readout')
    if np.any(value < 0) or np.any(value > MAX_OBSERVED_READOUT + 1e-12):
        raise ValueError('Observed readout must be log1p(10000 * mean composition)')


@dataclass(frozen=True)
class RidgeBinding:
    fitting_donors: tuple
    gene_ids: tuple
    representation_sha256: str
    fitting_input_sha256: str
    synthetic_fixture: bool = False

    def __post_init__(self):
        if (type(self.fitting_donors) is not tuple or len(self.fitting_donors) not in (2, 3)
                or self.fitting_donors != tuple(d for d in DONORS if d in self.fitting_donors)):
            raise ValueError('Canonical registered fitting pair or triple required')
        _ids(self.gene_ids)
        _sha(self.representation_sha256); _sha(self.fitting_input_sha256)
        if type(self.synthetic_fixture) is not bool:
            raise ValueError('Explicit fixture classification required')
        if (not self.synthetic_fixture and len(self.gene_ids) != 18129
                or self.synthetic_fixture and len(self.gene_ids) > 64):
            raise ValueError('Retain full scientific gene axis or bounded fabricated axis')

    @property
    def identity(self):
        return sha256(json.dumps(self.__dict__, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class TrainingRow:
    donor: str
    row_id: str
    target_id: str | None
    role: str = 'target'

    def __post_init__(self):
        if self.donor not in DONORS or not isinstance(self.row_id, str) or not self.row_id.strip():
            raise ValueError('Registered donor and explicit row identity required')
        if self.role == 'control':
            if self.row_id != 'pooled_control' or self.target_id is not None:
                raise ValueError('One explicitly pooled NTC row, not repeated guide controls')
        elif self.role != 'target' or not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError('Targeting rows require their target identity')
        elif self.row_id == 'pooled_control':
            raise ValueError('Pooled-control identity cannot label a targeting row')


def training_weights(binding, rows):
    """Equal donor, equal target, equal sister guide; control has half per donor."""
    if not isinstance(binding, RidgeBinding) or type(rows) is not tuple or not rows:
        raise ValueError('Exact fitting binding and ordered row manifest required')
    if not all(isinstance(row, TrainingRow) for row in rows):
        raise ValueError('Explicit training rows required')
    keys = [(r.donor, r.row_id) for r in rows]
    if len(set(keys)) != len(keys) or set(r.donor for r in rows) != set(binding.fitting_donors):
        raise ValueError('Duplicate rows or fitting donor exposure differs')
    if binding.synthetic_fixture and len(rows) > 64:
        raise ValueError('Fabricated law allowance exceeded')
    result = np.empty(len(rows), dtype=np.float64)
    for donor in binding.fitting_donors:
        target = [i for i, r in enumerate(rows) if r.donor == donor and r.role == 'target']
        control = [i for i, r in enumerate(rows) if r.donor == donor and r.role == 'control']
        if len(control) != 1 or not target:
            raise ValueError('Each donor needs targeting support and exactly one pooled control')
        groups = Counter(rows[i].target_id for i in target)
        half = .5 / len(binding.fitting_donors)
        result[control[0]] = half
        for i in target:
            result[i] = half / (len(groups) * groups[rows[i].target_id])
    return result


def source_moments(states, *, block_rows, maximum_working_bytes):
    """Full supplied positive-RNA support, equal-cell bounded tanh moments."""
    block_rows = _positive_int(block_rows); cap = _positive_int(maximum_working_bytes)
    if (not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[1] != 48
            or not len(states) or states.dtype.kind != 'f'):
        raise ValueError('Nonempty source states on exact 48-coordinate axis required')
    if states.nbytes + 48 * (40 * min(len(states), block_rows) + 64) > cap:
        raise ValueError('Source moment workspace exceeds bound before allocation')
    total = np.zeros(48, dtype=np.float64); second = np.zeros(48, dtype=np.float64)
    for start in range(0, len(states), block_rows):
        x = states[start:start+block_rows]
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite source state')
        x = np.tanh(np.asarray(x, dtype=np.float64))
        total += x.sum(0); second += (x*x).sum(0)
    mean = total / len(states)
    variance = second / len(states) - mean*mean
    tolerance = 64 * np.finfo(np.float64).eps * max(1, math.ceil(len(states)/block_rows))
    if np.any(variance < -tolerance):
        raise ValueError('Invalid bounded-feature variance')
    return np.concatenate((mean, np.maximum(variance, 0)))


def source_features(guide_moments, pooled_control_moments, *, source_frequency, positive_rna_support):
    """No late data or trainable identity. Pool control CELLS before moments."""
    for value in (guide_moments, pooled_control_moments):
        _array(value, (96,), 'source moments')
        if (np.any(np.abs(value[:48]) > 1+1e-12) or np.any(value[48:] < 0)
                or np.any(value[48:] > 1-value[:48]**2 + 1e-12)):
            raise ValueError('Invalid bounded-feature moments')
    _positive_int(positive_rna_support)
    if isinstance(source_frequency, bool) or not math.isfinite(source_frequency) or not 0 < source_frequency <= 1:
        raise ValueError('Positive complete-catalog source frequency required')
    return np.concatenate((guide_moments, pooled_control_moments,
                           [math.log(source_frequency), math.log1p(positive_rna_support)]))


@dataclass(frozen=True)
class RidgeDesign:
    binding: RidgeBinding
    rows: tuple
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    standardized: np.ndarray
    gram: np.ndarray


def prepare_design(binding, rows, features, *, maximum_rows, maximum_working_bytes):
    n = len(rows)
    if n > _positive_int(maximum_rows):
        raise ValueError('Fitting row allowance exceeded')
    cap = _positive_int(maximum_working_bytes)
    estimate = 64*n*FEATURES + 64*FEATURES**2 + 128*n + 128*FEATURES
    if estimate > cap:
        raise ValueError('Ridge design workspace exceeds bound before allocation')
    _array(features, (n, FEATURES), 'fitting features')
    w = training_weights(binding, rows)
    mean = w @ features
    centered = np.asarray(features, dtype=np.float64) - mean
    constant = np.all(features == features[0], axis=0)
    mean[constant] = features[0, constant]
    centered[:, constant] = 0
    variance = w @ (centered*centered)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    standardized = centered / scale
    gram = standardized.T @ (w[:, None]*standardized)
    for value in (mean, scale, standardized, gram):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite fitting-only standardization or Gram matrix')
        value.setflags(write=False)
    w.setflags(write=False)
    return RidgeDesign(binding, rows, w, mean, scale, standardized, gram)


def penalty_system(design, regularization):
    if not isinstance(design, RidgeDesign) or isinstance(regularization, bool) or regularization not in LAMBDA_GRID:
        raise ValueError('Prepared design and registered positive regularization required')
    return design.gram + regularization*np.eye(FEATURES, dtype=np.float64)


@dataclass(frozen=True)
class GeneBlock:
    start: int
    gene_ids: tuple
    row_keys: tuple
    source_readout: np.ndarray
    endpoint_readout: np.ndarray


@dataclass(frozen=True)
class RidgeReadout:
    binding: RidgeBinding
    regularization: float
    mean: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        if not isinstance(self.binding, RidgeBinding) or isinstance(self.regularization, bool) or self.regularization not in LAMBDA_GRID:
            raise ValueError('Bound ridge readout required')
        g = len(self.binding.gene_ids)
        for value, shape in ((self.mean, (FEATURES,)), (self.scale, (FEATURES,)),
                             (self.intercept, (g,)), (self.coefficients, (FEATURES, g))):
            _array(value, shape, 'ridge parameter')
        if np.any(self.scale <= 0):
            raise ValueError('Positive fitting-only feature scale required')


def fit_gene_blocks(design, blocks, *, regularization, maximum_gene_block, maximum_working_bytes):
    """Actual fitting: requires separate admission; caller supplies ordered blocks.

    Weighted square loss is averaged over rows, not over output genes, for each
    independently regularized gene. Gene blocking cannot alter its penalty.
    No whole-study approval is inferred from this numerical function.
    """
    if not isinstance(design, RidgeDesign):
        raise ValueError('Prepared fitting-only design required')
    n, g = len(design.rows), len(design.binding.gene_ids)
    block = min(g, _positive_int(maximum_gene_block)); cap = _positive_int(maximum_working_bytes)
    # Includes design, two input blocks, transformed temporaries, output and LAPACK workspace.
    estimate = (64*n*FEATURES + 128*FEATURES**2 + 64*n*block
                + 64*FEATURES*block + 32*FEATURES*g + 128*(n+g+FEATURES))
    if estimate > cap:
        raise ValueError('Ridge solve workspace exceeds bound before allocation')
    system = penalty_system(design, regularization)
    factor = cho_factor(system, lower=True, check_finite=True)
    coef = np.empty((FEATURES, g), dtype=np.float64); intercept = np.empty(g, dtype=np.float64)
    keys = tuple((r.donor, r.row_id) for r in design.rows)
    next_gene = 0
    for part in blocks:
        if (not isinstance(part, GeneBlock) or type(part.start) is not int or part.start != next_gene
                or type(part.gene_ids) is not tuple or not 0 < len(part.gene_ids) <= block
                or part.row_keys != keys
                or part.gene_ids != design.binding.gene_ids[next_gene:next_gene+len(part.gene_ids)]):
            raise ValueError('Gene blocks must exhaust exact gene and fitting-row order once')
        b = len(part.gene_ids)
        for value in (part.source_readout, part.endpoint_readout):
            _observed(value, (n, b))
        delta = np.asarray(part.endpoint_readout, dtype=np.float64) - part.source_readout
        mu = design.weights @ delta
        rhs = design.standardized.T @ (design.weights[:, None]*(delta-mu))
        solved = cho_solve(factor, rhs, check_finite=True)
        _array(solved, (FEATURES, b), 'solved coefficients')
        residual = system @ solved - rhs
        scale = np.abs(system) @ np.abs(solved) + np.abs(rhs)
        tolerance = 256*FEATURES*np.finfo(np.float64).eps*np.maximum(1., scale)
        if not np.isfinite(residual).all() or np.any(np.abs(residual) > tolerance):
            raise ValueError('Ridge solver residual exceeds fixed arithmetic bound')
        coef[:, next_gene:next_gene+b] = solved
        intercept[next_gene:next_gene+b] = mu
        next_gene += b
    if next_gene != g:
        raise ValueError('Incomplete all-gene response inventory')
    for value in (coef, intercept): value.setflags(write=False)
    return RidgeReadout(design.binding, float(regularization), design.mean, design.scale, intercept, coef)


def predict_block(model, features, source_readout, *, query_donor, condition,
                  representation_sha256, start, gene_ids, maximum_rows, maximum_working_bytes):
    """Source-only transformed endpoint prediction; never clip to a composition."""
    if (not isinstance(model, RidgeReadout) or query_donor not in DONORS
            or query_donor in model.binding.fitting_donors or condition != 'Stim8hr'
            or representation_sha256 != model.binding.representation_sha256):
        raise ValueError('Exact held-out early source and fitting representation required')
    if (type(start) is not int or start < 0 or type(gene_ids) is not tuple or not gene_ids
            or gene_ids != model.binding.gene_ids[start:start+len(gene_ids)]):
        raise ValueError('Prediction gene axis/order differs')
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        raise ValueError('Exact source feature matrix required')
    n, b = len(features), len(gene_ids)
    if not 0 < n <= _positive_int(maximum_rows):
        raise ValueError('Prediction row allowance exceeded')
    estimate = (64*n*FEATURES + 64*n*b + 16*FEATURES*len(model.binding.gene_ids)
                + 128*(FEATURES+b))
    if estimate > _positive_int(maximum_working_bytes):
        raise ValueError('Prediction workspace exceeds bound before allocation')
    _array(features, (n, FEATURES), 'query feature')
    _observed(source_readout, (n, b))
    result = (np.asarray(source_readout, dtype=np.float64) + model.intercept[start:start+b]
              + ((features-model.mean)/model.scale) @ model.coefficients[:, start:start+b])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite ridge molecular forecast')
    return result


def transformed_endpoint_scores(predicted_target, observed_target, predicted_control, observed_control):
    """One law's direct transformed-scale errors; missing control does not erase target."""
    values = [x for x in (predicted_target, observed_target, predicted_control, observed_control) if x is not None]
    if not values or not isinstance(values[0], np.ndarray) or values[0].ndim != 1 or not len(values[0]):
        raise ValueError('At least one nonempty all-gene readout required')
    shape = values[0].shape
    for value in values: _array(value, shape, 'molecular readout')
    for value in (observed_target, observed_control):
        if value is not None: _observed(value, shape)
    pt, ot, pc, oc = (None if v is None else np.asarray(v, dtype=np.float64)
                      for v in (predicted_target, observed_target, predicted_control, observed_control))
    et = None if pt is None or ot is None else pt-ot
    ec = None if pc is None or oc is None else pc-oc
    def mse(value):
        if value is None: return None
        result = float(np.mean(np.asarray(value, dtype=np.float64)**2))
        if not math.isfinite(result): raise ValueError('Nonfinite molecular squared error')
        return result
    interaction = None if et is None or ec is None else float(-2*np.mean(et*ec, dtype=np.float64))
    if interaction is not None and not math.isfinite(interaction):
        raise ValueError('Nonfinite endpoint interaction')
    return dict(target_endpoint_mse=mse(et), pooled_control_endpoint_mse=mse(ec),
                pooled_effect_mse=mse(None if et is None or ec is None else et-ec),
                interaction=interaction, zero_effect_mse=mse(None if ot is None or oc is None else ot-oc),
                control_reduction='one pooled control per donor, not independent per target')


@dataclass(frozen=True)
class RidgeInnerScore:
    selector: str
    fitting_donors: tuple
    validation_donor: str
    regularization: float
    target_mse: float
    control_mse: float
    support_sha256: str
    fitting_input_sha256: str
    representation_sha256: str
    truth_sha256: str
    readout_sha256: str

    def __post_init__(self):
        if (self.selector, self.fitting_donors, self.validation_donor) not in inner_edges():
            raise ValueError('Only the exact permitted inner donor edge can select ridge')
        if isinstance(self.regularization, bool) or self.regularization not in LAMBDA_GRID:
            raise ValueError('Unregistered ridge strength')
        for value in (self.target_mse, self.control_mse):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Both molecular endpoint channels required and finite')
        for value in (self.support_sha256, self.fitting_input_sha256, self.representation_sha256,
                      self.truth_sha256, self.readout_sha256): _sha(value)


def select_regularization(selector, records):
    """One frozen donor-safe choice; no partial edges, penalty grids or supports."""
    if selector not in DONORS:
        raise ValueError('Registered outer selector required')
    records = tuple(records)
    expected = {(pair, donor, lam) for s, pair, donor in inner_edges() if s == selector for lam in LAMBDA_GRID}
    if (not all(isinstance(r, RidgeInnerScore) and r.selector == selector for r in records)
            or len(records) != len(expected)
            or {(r.fitting_donors, r.validation_donor, r.regularization) for r in records} != expected):
        raise ValueError('Complete unique three-edge four-strength inner evidence required')
    for _, pair, donor in (e for e in inner_edges() if e[0] == selector):
        group = [r for r in records if r.fitting_donors == pair and r.validation_donor == donor]
        if len({(r.support_sha256, r.fitting_input_sha256, r.representation_sha256,
                 r.truth_sha256, r.readout_sha256) for r in group}) != 1:
            raise ValueError('Ridge strengths must share exact support, inputs, representation and readout')
    risks = {lam: math.fsum(.5*r.target_mse+.5*r.control_mse for r in records if r.regularization == lam)/3
             for lam in LAMBDA_GRID}
    return dict(regularization=min(LAMBDA_GRID, key=lambda lam: (risks[lam], -lam)),
                inner_endpoint_risks=risks, selection_donors=tuple(d for d in DONORS if d != selector),
                outer_donor_used=False, capability='molecular_readout_only')
