"""Exact full-catalog linear abundance comparator, without I/O or launch authority.

No target/state support filter removes a catalog entry. Every objective call
uses current parameters for every category. This is a separate count-only
procedure, not a change to the neural cached estimator. Data custody, actual
resource measurements and publication remain the admitted worker's obligation.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .configuration import DONORS
from .fitting_provider import CatalogGuide
from .planner import inner_edges


FEATURES = 100
REGULARIZATION_GRID = (.001, .01, .1, 1.)
MAX_ITERATIONS = 500
MAX_EVALUATIONS = 12501


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase SHA-256 required')


def _bound(value):
    if type(value) is not int or value < 1:
        raise ValueError('Explicit positive integer memory bound required')
    return value


def _array(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError(f'Finite FP32/FP64 exact {name} shape required')


def _counts(value, size, *, require_total):
    if (not isinstance(value, np.ndarray) or value.dtype != np.int64
            or value.shape != (size,) or np.any(value < 0)):
        raise ValueError('Full nonnegative int64 captured-count vector required')
    total = sum(int(v) for v in value)
    if require_total and total <= 0:
        raise ValueError('Positive complete captured population required')
    return total


@dataclass(frozen=True)
class HeadBinding:
    fitting_donors: tuple
    catalog: tuple
    representation_sha256: str
    fitting_input_sha256: str
    synthetic_fixture: bool = False

    def __post_init__(self):
        if (type(self.fitting_donors) is not tuple or len(self.fitting_donors) not in (2, 3)
                or self.fitting_donors != tuple(d for d in DONORS if d in self.fitting_donors)):
            raise ValueError('Canonical fitting pair or triple required')
        if (type(self.catalog) is not tuple or not self.catalog
                or not all(isinstance(g, CatalogGuide) for g in self.catalog)
                or len({g.guide_id for g in self.catalog}) != len(self.catalog)):
            raise ValueError('Distinct complete ordered catalog required')
        if type(self.synthetic_fixture) is not bool:
            raise ValueError('Explicit fixture classification required')
        if ((not self.synthetic_fixture and len(self.catalog) != 26504)
                or (self.synthetic_fixture and len(self.catalog) > 64)):
            raise ValueError('Full scientific catalog or small fabricated inventory required')
        _sha(self.representation_sha256); _sha(self.fitting_input_sha256)

    @property
    def catalog_sha256(self):
        return sha256(json.dumps([asdict(g) for g in self.catalog],sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class SourceCatalog:
    donor: str
    condition: str
    catalog_sha256: str
    representation_sha256: str
    input_sha256: str
    captured_counts: np.ndarray
    positive_rna_counts: np.ndarray
    moments: np.ndarray

    def validate(self, binding):
        if (not isinstance(binding, HeadBinding) or self.donor not in DONORS or self.condition != 'Stim8hr'
                or self.catalog_sha256 != binding.catalog_sha256
                or self.representation_sha256 != binding.representation_sha256):
            raise ValueError('Exact early source, complete catalog and fold representation required')
        _sha(self.input_sha256)
        n=len(binding.catalog)
        _counts(self.captured_counts,n,require_total=True)
        _counts(self.positive_rna_counts,n,require_total=False)
        if np.any(self.positive_rna_counts > self.captured_counts):
            raise ValueError('Positive-RNA support exceeds captured population')
        _array(self.moments,(n,96),'source moment')
        absent=self.positive_rna_counts == 0
        if np.any(self.moments[absent] != 0):
            raise ValueError('Missing state requires explicit neutral placeholders, not fabricated moments')
        mean,var=self.moments[:,:48],self.moments[:,48:]
        if (np.any(np.abs(mean)>1+1e-12) or np.any(var<0)
                or np.any(var>1-mean*mean+1e-12)):
            raise ValueError('Invalid bounded source moments')


def source_features(binding, source, *, maximum_working_bytes):
    n=len(binding.catalog)
    if 64*n*FEATURES + 128*n > _bound(maximum_working_bytes):
        raise ValueError('Source feature workspace exceeds bound before allocation')
    source.validate(binding)
    total=sum(int(v) for v in source.captured_counts)
    logq0=np.log(source.captured_counts.astype(np.float64)+.5)-math.log(total+.5*n)
    features=np.column_stack((source.moments,logq0,np.log1p(source.captured_counts.astype(np.float64)),
                              np.log1p(source.positive_rna_counts.astype(np.float64)),
                              (source.positive_rna_counts>0).astype(np.float64)))
    return features,logq0


@dataclass(frozen=True)
class FittingCatalog:
    source: SourceCatalog
    endpoint_counts: np.ndarray
    endpoint_input_sha256: str


@dataclass(frozen=True)
class HeadDesign:
    binding: HeadBinding
    mean: np.ndarray
    scale: np.ndarray
    target_ids: tuple
    target_index: np.ndarray
    penalty: np.ndarray
    standardized_features: tuple
    source_log_frequencies: tuple
    observed_frequencies: tuple
    fitting_catalogs: tuple

    @property
    def parameter_count(self):
        return FEATURES+len(self.target_ids)


def prepare_design(binding, catalogs, *, maximum_working_bytes):
    """Fitting-only scaling and target membership; endpoints affect only p."""
    if not isinstance(binding, HeadBinding) or type(catalogs) is not tuple:
        raise ValueError('Bound fitting catalogs required')
    if (not all(isinstance(c,FittingCatalog) for c in catalogs)
            or tuple(c.source.donor for c in catalogs) != binding.fitting_donors):
        raise ValueError('Exact fitting donor order required; no validation donor')
    n,d=len(binding.catalog),len(catalogs)
    cap=_bound(maximum_working_bytes)
    if 96*d*n*FEATURES + 512*n + 128*FEATURES > cap:
        raise ValueError('Full-catalog design workspace exceeds bound before allocation')
    features=[];logs=[];observed=[];support={}
    for c in catalogs:
        x,logq=source_features(binding,c.source,maximum_working_bytes=cap)
        _sha(c.endpoint_input_sha256)
        total=_counts(c.endpoint_counts,n,require_total=True)
        features.append(x);logs.append(logq);observed.append(c.endpoint_counts.astype(np.float64)/total)
        for g,count in zip(binding.catalog,c.source.captured_counts):
            if not g.is_control and count>0:
                support[g.target_id]=support.get(g.target_id,0)+int(count)
    # Only 2 or 3 complete catalog matrices; no whole-cohort RNA densification.
    mean=sum(x.mean(0) for x in features)/d
    constant=np.ones(FEATURES,dtype=bool)
    first=features[0][0]
    for x in features: constant &= np.all(x == first,axis=0)
    mean[constant]=first[constant]
    centered=[x-mean for x in features]
    for x in centered: x[:,constant]=0
    variance=sum((x*x).mean(0) for x in centered)/d
    scale=np.sqrt(variance);scale[scale==0]=1
    standardized=tuple(x/scale for x in centered)
    targets=tuple(sorted(support));lookup={t:i for i,t in enumerate(targets)}
    indices=np.array([-1 if g.is_control else lookup.get(g.target_id,-1) for g in binding.catalog],dtype=np.int64)
    penalty=np.array([1.]*FEATURES+[1.+1./support[t] for t in targets],dtype=np.float64)
    for x in (mean,scale,penalty,*standardized,*logs,*observed):
        if not np.isfinite(x).all(): raise ValueError('Nonfinite count design')
        x.setflags(write=False)
    indices.setflags(write=False)
    return HeadDesign(binding,mean,scale,targets,indices,penalty,standardized,tuple(logs),tuple(observed),catalogs)


def catalog_prediction(parameters, features, source_logs, target_index):
    """One full current-parameter normalization; retain finite log probabilities."""
    _array(parameters,(len(parameters),),'parameter')
    if len(parameters)<FEATURES or features.ndim!=2:
        raise ValueError('Full abundance parameter vector and catalog features required')
    n=len(features);nt=len(parameters)-FEATURES
    _array(features,(n,FEATURES),'standardized feature');_array(source_logs,(n,),'source log frequency')
    if (not n or abs(float(logsumexp(source_logs)))>1e-10
            or not isinstance(target_index,np.ndarray) or target_index.dtype!=np.int64 or target_index.shape!=(n,)
            or np.any(target_index < -1) or np.any(target_index >= nt)):
        raise ValueError('Complete normalized source logs and target routing required')
    logits=source_logs+features@parameters[:FEATURES]
    active=target_index>=0
    logits[active]+=parameters[FEATURES+target_index[active]]
    if not np.isfinite(logits).all(): raise ValueError('Nonfinite log abundance')
    logq=logits-logsumexp(logits)
    q=np.exp(logq)
    if not np.isfinite(logq).all() or np.any(q<=0):
        raise ValueError('Underflowed forecast is not an observed zero; no floor')
    return logq,q


def objective_and_gradient(parameters, design, regularization):
    """Exact equal-donor CE gradient through every catalog category, no optimizer."""
    if not isinstance(design,HeadDesign) or isinstance(regularization,bool) or regularization not in REGULARIZATION_GRID:
        raise ValueError('Full fitting design and registered regularization required')
    _array(parameters,(design.parameter_count,),'parameter')
    p64=np.asarray(parameters,dtype=np.float64)
    loss=0.;gradient=np.zeros(design.parameter_count,dtype=np.float64);d=len(design.binding.fitting_donors)
    active=design.target_index>=0
    for x,logs,p in zip(design.standardized_features,design.source_log_frequencies,design.observed_frequencies):
        logq,q=catalog_prediction(p64,x,logs,design.target_index)
        loss-=float(p@logq)/d
        residual=(q-p)/d
        gradient[:FEATURES]+=x.T@residual
        gradient[FEATURES:]+=np.bincount(design.target_index[active],weights=residual[active],minlength=len(design.target_ids))
    loss+=.5*regularization*float(np.dot(design.penalty,p64*p64))
    gradient+=regularization*design.penalty*p64
    if not math.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite exact count objective or gradient')
    return loss,gradient


@dataclass(frozen=True)
class AbundanceHead:
    binding: HeadBinding
    mean: np.ndarray
    scale: np.ndarray
    target_ids: tuple
    target_index: np.ndarray
    parameters: np.ndarray
    regularization: float
    diagnostics: dict


def fit_head(design, regularization, *, maximum_working_bytes):
    """Scientific/disposable fitting, NOT authorized by implementation tests.

    Caller owns exact fit admission and failure receipts. No resume, retry or
    fallback occurs. Iteration-limited finite results are labeled explicitly.
    """
    if (not isinstance(design,HeadDesign) or isinstance(regularization,bool)
            or regularization not in REGULARIZATION_GRID):
        raise ValueError('Prepared count design and registered regularization required')
    n=len(design.binding.catalog);d=len(design.binding.fitting_donors);k=design.parameter_count
    if 96*d*n*FEATURES+1024*k+512*n > _bound(maximum_working_bytes):
        raise ValueError('Count fitting workspace exceeds bound before allocation')
    calls=0
    def objective(parameters):
        nonlocal calls
        if calls>=MAX_EVALUATIONS: raise ValueError('Registered function-evaluation cap exhausted')
        calls+=1
        return objective_and_gradient(parameters,design,regularization)
    result=minimize(objective,np.zeros(k,dtype=np.float64),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=MAX_ITERATIONS,maxfun=MAX_EVALUATIONS-1,maxls=25,maxcor=10,ftol=1e-12,gtol=1e-8))
    if (type(result.nit) is not int or not 0<=result.nit<=MAX_ITERATIONS
            or type(result.nfev) is not int or result.nfev!=calls):
        raise ValueError('Optimizer counter mismatch')
    if not result.success and not (result.status==1 and result.nit==MAX_ITERATIONS):
        raise ValueError('Count optimizer failed; no automatic retry or fallback')
    final_value,final_gradient=objective(result.x)
    if not math.isfinite(float(result.fun)) or not math.isclose(float(result.fun),final_value,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Final count objective does not reconcile')
    parameters=np.array(result.x,dtype=np.float64,copy=True);parameters.setflags(write=False)
    return AbundanceHead(design.binding,design.mean,design.scale,design.target_ids,design.target_index,
        parameters,float(regularization),dict(converged=bool(result.success),iterations=result.nit,
            function_gradient_evaluations=calls,final_objective=final_value,
            final_gradient_infinity_norm=float(np.max(np.abs(final_gradient))),
            terminal_reason='converged' if result.success else 'iteration_cap',solver_message=str(result.message)[:512]))


def predict_catalog(model, source, *, maximum_working_bytes):
    if (not isinstance(model,AbundanceHead) or not isinstance(source,SourceCatalog)
            or source.donor in model.binding.fitting_donors):
        raise ValueError('Fitted comparator and held-out early source required')
    features,logs=source_features(model.binding,source,maximum_working_bytes=maximum_working_bytes)
    _array(model.mean,(FEATURES,),'fitting feature mean');_array(model.scale,(FEATURES,),'fitting feature scale')
    if np.any(model.scale<=0): raise ValueError('Positive fitting-only feature scaling required')
    if (type(model.target_ids) is not tuple or tuple(sorted(set(model.target_ids)))!=model.target_ids
            or len(model.parameters)!=FEATURES+len(model.target_ids)):
        raise ValueError('Fitted target intercept routing changed')
    lookup={t:i for i,t in enumerate(model.target_ids)}
    if not set(lookup).issubset({g.target_id for g in model.binding.catalog if not g.is_control}):
        raise ValueError('Fitted target outside the catalog')
    expected=np.array([-1 if g.is_control else lookup.get(g.target_id,-1) for g in model.binding.catalog],dtype=np.int64)
    if not np.array_equal(model.target_index,expected):
        raise ValueError('Fitted target intercept routing changed')
    logq,_=catalog_prediction(model.parameters,(features-model.mean)/model.scale,logs,model.target_index)
    return logq


@dataclass(frozen=True)
class InnerCountScore:
    selector: str
    fitting_donors: tuple
    validation_donor: str
    regularization: float
    cross_entropy: float
    catalog_sha256: str
    source_sha256: str
    fitting_input_sha256: str
    truth_sha256: str
    representation_sha256: str
    scoring_sha256: str

    def __post_init__(self):
        if (self.selector,self.fitting_donors,self.validation_donor) not in inner_edges():
            raise ValueError('Exact permitted inner donor edge required')
        if isinstance(self.regularization,bool) or self.regularization not in REGULARIZATION_GRID:
            raise ValueError('Unregistered count penalty')
        if isinstance(self.cross_entropy,bool) or not math.isfinite(self.cross_entropy) or self.cross_entropy<0:
            raise ValueError('Finite complete-catalog CE required')
        for value in (self.catalog_sha256,self.source_sha256,self.fitting_input_sha256,self.truth_sha256,
                      self.representation_sha256,self.scoring_sha256): _sha(value)


def select_regularization(selector, records):
    records=tuple(records)
    expected={(p,d,lam) for s,p,d in inner_edges() if s==selector for lam in REGULARIZATION_GRID}
    if (selector not in DONORS or len(records)!=12
            or not all(isinstance(r,InnerCountScore) and r.selector==selector for r in records)
            or {(r.fitting_donors,r.validation_donor,r.regularization) for r in records}!=expected):
        raise ValueError('Exactly three permitted donor edges and four strengths required')
    for _,pair,donor in (e for e in inner_edges() if e[0]==selector):
        group=[r for r in records if r.fitting_donors==pair and r.validation_donor==donor]
        if len({(r.catalog_sha256,r.source_sha256,r.fitting_input_sha256,r.truth_sha256,
                 r.representation_sha256,r.scoring_sha256) for r in group})!=1:
            raise ValueError('Every strength must use the same complete catalog, source, truth and readout')
    risks={lam:math.fsum(r.cross_entropy/3 for r in records if r.regularization==lam) for lam in REGULARIZATION_GRID}
    return dict(regularization=min(REGULARIZATION_GRID,key=lambda lam:(risks[lam],-lam)),inner_ce=risks,
                outer_donor_used=False,capability='complete_catalog_abundance_only')
