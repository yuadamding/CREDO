"""Conditional split diagnostics on supplied observations, not an outcome opener.

No file/network/model/optimizer access occurs here. The scientific owner must
bind input custody, exact populations, operations and publication separately.
Balanced splitting is GLOBAL across captured units, never guide-stratified:
balancing each guide would artificially suppress guide-frequency uncertainty.
Cross-errors remain descriptive; disjointness alone does not establish the
independence/unbiasedness assumptions needed for noise-removal claims.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy import sparse


REPETITIONS = tuple(range(20))


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Explicit nonempty identity required')
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact SHA-256 identity required')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SplitAssignment:
    """A caller-owned row-aligned assignment; it contains no outcome values."""
    row_ids: tuple
    half: tuple
    population_sha256: str
    repetition: int
    unit_kind: str
    unit_provenance_sha256: str | None
    unit_count_a: int
    unit_count_b: int

    def __post_init__(self):
        _sha(self.population_sha256)
        if (type(self.row_ids) is not tuple or type(self.half) is not tuple
                or len(self.row_ids)<2 or len(set(self.row_ids))!=len(self.row_ids)
                or any(not isinstance(r,str) or not r for r in self.row_ids)
                or len(self.row_ids)!=len(self.half) or any(type(h) is not int or h not in (0,1) for h in self.half)
                or set(self.half)!={0,1} or type(self.repetition) is not int or self.repetition not in REPETITIONS
                or any(type(n) is not int or n<1 for n in (self.unit_count_a,self.unit_count_b))
                or self.unit_count_b-self.unit_count_a not in (0,1)):
            raise ValueError('Exact balanced-unit, disjoint captured assignment required')
        if self.unit_kind=='conditional_cell_sampling':
            if (self.unit_provenance_sha256 is not None or self.half.count(0)!=self.unit_count_a
                    or self.half.count(1)!=self.unit_count_b):
                raise ValueError('Cell assignment unit counts differ')
        elif self.unit_kind=='declared_capture_units':
            _sha(self.unit_provenance_sha256)
            if self.unit_count_a+self.unit_count_b>len(self.row_ids):
                raise ValueError('Capture-unit count exceeds captured rows')
        else:
            raise ValueError('Unknown splitting unit')

    @property
    def identity(self):
        # Order-independent assignment identity; returned rows preserve caller order.
        return _digest([self.population_sha256, self.repetition, self.unit_kind,
                        self.unit_provenance_sha256, sorted(zip(self.row_ids, self.half))])


def balanced_assignment(row_ids, *, population_sha256, repetition,
                        capture_units=None, unit_provenance_sha256=None):
    """One of twenty deterministic splits; no expression/support stratification.

    Caller must account for O(n log n) metadata sorting and memory. This is not
    a streaming all-cohort intake or a proof that supplied unit labels are real.
    """
    _sha(population_sha256)
    rows = tuple(row_ids)
    if (len(rows) < 2 or any(not isinstance(v, str) or not v for v in rows)
            or len(set(rows)) != len(rows) or type(repetition) is not int or repetition not in REPETITIONS):
        raise ValueError('Unique captured row identities and one registered repetition required')
    if capture_units is None:
        if unit_provenance_sha256 is not None:
            raise ValueError('Cell splitting cannot claim capture-unit provenance')
        units, kind = rows, 'conditional_cell_sampling'
    else:
        units, kind = tuple(capture_units), 'declared_capture_units'
        _sha(unit_provenance_sha256)
        if len(units) != len(rows) or any(not isinstance(u, str) or not u for u in units):
            raise ValueError('Every captured row requires one declared capture unit')
    unique = set(units)
    if len(unique) < 2:
        raise ValueError('At least two splitting units required; no fabricated second unit')
    ranked = sorted(unique, key=lambda unit: (
        _digest(['frontier-reliability-v1', population_sha256, repetition, unit]), unit))
    first = set(ranked[:len(ranked)//2])
    return SplitAssignment(rows, tuple(0 if u in first else 1 for u in units),
        population_sha256, repetition, kind, unit_provenance_sha256, len(first), len(unique)-len(first))


def support_tier(source_count):
    if type(source_count) is not int or source_count < 0:
        raise ValueError('Explicit nonnegative source support count required')
    return '0-15' if source_count < 16 else '16-63' if source_count < 64 else '64-255' if source_count < 256 else '256+'


class SplitCompositionAccumulator:
    """One pass over row-ordered sparse RNA chunks for ONE registered split.

    Captured zeros remain in catalog counts; only positive-RNA cells contribute
    compositions. Rows must exhaust the exact assignment once and in order.
    No per-guide balancing, hidden sampling weights or gene selection occurs.
    Array RAM and maximum sparse chunk workspace are explicitly charged before
    allocation. Scientific publication/whole-worker peak remains external.
    """
    def __init__(self, assignment, *, catalog_size, features, control_indices,
                 maximum_rows, maximum_nnz, maximum_working_bytes):
        if not isinstance(assignment, SplitAssignment):
            raise TypeError('Exact captured split assignment required')
        for v in (catalog_size, features, maximum_rows, maximum_nnz, maximum_working_bytes):
            if type(v) is not int or v < 1:
                raise ValueError('Positive dimensions and explicit memory/chunk limits required')
        controls = tuple(control_indices)
        if len(set(controls)) != len(controls) or any(type(g) is not int or not 0 <= g < catalog_size for g in controls):
            raise ValueError('Unique catalog NTC indices required')
        # Persistent sums, result/accessor vectors, sparse-index/factor temporaries,
        # copied caller sparse arrays and row/guide/half integer workspaces.
        required = 16*catalog_size*features + 128*catalog_size + 64*features + 64*maximum_nnz + 128*(maximum_rows+1)
        if required > maximum_working_bytes:
            raise ValueError('Split accumulator exceeds declared working memory before allocation')
        if (len(assignment.half) != len(assignment.row_ids) or set(assignment.half) != {0, 1}
                or len(set(assignment.row_ids)) != len(assignment.row_ids)):
            raise ValueError('Disjoint exhaustive two-half assignment required')
        self.assignment, self.catalog_size, self.features = assignment, catalog_size, features
        self.controls, self.maximum_rows, self.maximum_nnz = controls, maximum_rows, maximum_nnz
        self.maximum_working_bytes, self.accounted_array_workspace_bytes = maximum_working_bytes, required
        self.captured = np.zeros((2,catalog_size),dtype=np.int64)
        self.positive = np.zeros((2,catalog_size),dtype=np.int64)
        self.sums = np.zeros((2,catalog_size,features),dtype=np.float64)
        self.position, self.finalized, self.failed = 0, False, False

    def update(self, row_ids, guide_indices, counts):
        if self.finalized or self.failed:
            raise ValueError('Closed/failed accumulator cannot accept more rows')
        try:
            rows = tuple(row_ids)
            if (not 0 < len(rows) <= self.maximum_rows
                    or rows != self.assignment.row_ids[self.position:self.position+len(rows)]):
                raise ValueError('Exact next captured row identities required')
            guides = np.asarray(guide_indices)
            if (guides.dtype.kind not in 'iu' or guides.shape != (len(rows),)
                    or np.any(guides < 0) or np.any(guides >= self.catalog_size)):
                raise ValueError('Complete catalog guide index axis required')
            if (not sparse.isspmatrix_csr(counts) or counts.shape != (len(rows),self.features)
                    or counts.dtype.kind not in 'iu' or counts.nnz > self.maximum_nnz
                    or not counts.has_canonical_format or np.any(counts.data < 0)):
                raise ValueError('Bounded canonical nonnegative integer RNA-only CSR required')
            halves = np.asarray(self.assignment.half[self.position:self.position+len(rows)],dtype=np.int64)
            depth = np.asarray(counts.sum(axis=1,dtype=np.float64)).ravel()
            if not np.isfinite(depth).all():
                raise ValueError('Nonfinite RNA depth')
            positive = depth > 0
            row = np.repeat(np.arange(len(rows)),np.diff(counts.indptr))
            factors = np.zeros(len(rows),dtype=np.float64)
            factors[positive] = 1/depth[positive]
            contribution=counts.data*factors[row]
            if not np.isfinite(contribution).all():
                raise ValueError('Nonfinite chunk composition')
            np.add.at(self.captured,(halves,guides),1)
            np.add.at(self.positive,(halves[positive],guides[positive]),1)
            # Nonnegative contributions are bounded by one per captured cell.
            # Do not rescan/densify the G x J accumulator after every RNA chunk.
            np.add.at(self.sums,(halves[row],guides[row],counts.indices),contribution)
            self.position += len(rows)
        except BaseException:
            self.failed = True
            raise

    def finish(self):
        if self.finalized or self.failed or self.position != len(self.assignment.row_ids):
            raise ValueError('Exact complete captured inventory required before finalization')
        if not np.allclose(self.sums.sum(axis=2),self.positive,rtol=1e-10,atol=1e-10):
            raise ValueError('Positive-RNA composition accounting failed')
        self.finalized = True
        return dict(assignment_sha256=self.assignment.identity, unit_kind=self.assignment.unit_kind,
            repetition=self.assignment.repetition, captured_rows=int(self.captured.sum()),
            positive_RNA_rows=int(self.positive.sum()), captured_by_half=self.captured.sum(1).tolist(),
            positive_RNA_by_half=self.positive.sum(1).tolist(), catalog_size=self.catalog_size,
            features=self.features, independent_biological_replicates=0,
            accounted_array_workspace_bytes=self.accounted_array_workspace_bytes,
            assignment_metadata_and_worker_overhead_included=False)

    def _closed(self):
        if not self.finalized or self.failed:
            raise ValueError('Finalized successful population required')

    def guide_compositions(self, index):
        self._closed()
        if type(index) is not int or not 0 <= index < self.catalog_size:
            raise ValueError('Exact catalog index required')
        return tuple(None if self.positive[h,index]==0 else self.sums[h,index]/self.positive[h,index] for h in (0,1))

    def control_compositions(self):
        self._closed()
        # Loop over controls to avoid a copy of a G_control x J dense submatrix.
        sums = np.zeros((2,self.features),dtype=np.float64)
        totals = np.zeros(2,dtype=np.int64)
        for index in self.controls:
            sums += self.sums[:,index,:]
            totals += self.positive[:,index]
        return tuple(None if totals[h]==0 else sums[h]/totals[h] for h in (0,1))

    def catalog_counts(self):
        self._closed()
        return self.captured.copy()


def empirical_catalog_pair(counts_a, counts_b):
    """Both directions, full catalog, no smoothing or zero-category removal."""
    arrays = [np.asarray(c) for c in (counts_a,counts_b)]
    if (any(c.ndim!=1 or c.dtype.kind not in 'iu' or not len(c) or np.any(c<0) for c in arrays)
            or arrays[0].shape!=arrays[1].shape):
        raise ValueError('Matching complete nonnegative integer catalogs required')
    totals = [sum(map(int,c)) for c in arrays]
    if min(totals)<=0:
        return dict(available=False, reason='missing_captured_half', totals=totals)
    p = [c.astype(np.float64)/n for c,n in zip(arrays,totals)]
    result=dict(available=True,totals=totals,catalog_size=len(p[0]),
        zero_categories=[int((c==0).sum()) for c in arrays],squared_frequency_difference=float(np.square(p[0]-p[1]).sum()))
    for label,q,truth in (('A_predicts_B',p[0],p[1]),('B_predicts_A',p[1],p[0])):
        missing=(q==0)&(truth>0)
        ce=None if missing.any() else float(-np.sum(truth[truth>0]*np.log(q[truth>0])))
        result[label]=dict(cross_entropy_nats=ce,extended_cross_entropy='+infinity' if missing.any() else 'finite',
            positive_truth_zero_forecast_categories=int(missing.sum()),
            quadratic_score=float(1+np.square(q).sum()-2*np.dot(q,truth)))
    return result


def split_cross_error(prediction, half_a, half_b):
    """Exact descriptive cross-error on one supplied fixed-scale gene vector.

    No independence/unbiasedness claim is inferred, even for disjoint halves.
    Negative values are valid and retained. Missing partners do not discard an
    independently valid single-half endpoint comparison.
    """
    values={}
    for name, value in (('prediction',prediction),('half_a',half_a),('half_b',half_b)):
        if value is not None:
            x=np.asarray(value)
            if x.ndim!=1 or not len(x) or x.dtype.kind!='f' or not np.isfinite(x).all():
                raise ValueError('Finite nonempty floating readout vectors required')
            values[name]=x.astype(np.float64)
    if len({x.shape for x in values.values()})>1:
        raise ValueError('Fixed readout axes must match')
    result=dict(cross_error=None,mse_a=None,mse_b=None,half_discrepancy_mse=None,
        available=False,reason='missing_prediction_or_half',noise_corrected_risk_claimed=False,
        assumptions_verified=False,negative_values_clipped=False)
    for name,key in (('half_a','mse_a'),('half_b','mse_b')):
        if 'prediction' in values and name in values:
            result[key]=float(np.square(values['prediction']-values[name]).mean())
    if 'half_a' in values and 'half_b' in values:
        result['half_discrepancy_mse']=float(np.square(values['half_a']-values['half_b']).mean())
    if len(values)==3:
        cross=float(np.mean((values['prediction']-values['half_a'])*(values['prediction']-values['half_b'])))
        result.update(available=True,reason=None,cross_error=cross,
            arithmetic_residual=cross-0.5*(result['mse_a']+result['mse_b']-result['half_discrepancy_mse']))
    if any(isinstance(v,float) and not math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite split-error arithmetic')
    return result


def molecular_split_errors(predicted_target, target_a, target_b, predicted_control, control_a, control_b):
    """Separate target/control/contrast errors on raw and transformed readouts.

    Controls must already be cell-pooled compositions, not pooled after log.
    The caller reports the standalone control once per split population.
    """
    values=[]
    for value in (predicted_target,target_a,target_b,predicted_control,control_a,control_b):
        if value is None:
            values.append(None)
        else:
            x=np.asarray(value,dtype=np.float64)
            if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<0) or abs(x.sum()-1)>1e-10:
                raise ValueError('Full-gene normalized nonnegative compositions required')
            values.append(x)
    if len({x.shape for x in values if x is not None})>1:
        raise ValueError('Molecular axes differ')
    result=dict(control_reduction_scope='one_pooled_control_per_split_population',noise_corrected_risk_claimed=False)
    for scale in ('mean_composition','log1p_10000_mean_composition'):
        v=values if scale=='mean_composition' else [None if x is None else np.log1p(10000*x) for x in values]
        t,a,b,c,ca,cb=v
        contrast=[None if x is None or y is None else x-y for x,y in ((t,c),(a,ca),(b,cb))]
        target,control,effect=split_cross_error(t,a,b),split_cross_error(c,ca,cb),split_cross_error(*contrast)
        result[scale]=dict(target=target,control=control,effect=effect)
        if all(x is not None for x in v):
            interaction=-float(np.mean((t-a)*(c-cb)+(c-ca)*(t-b)))
            result[scale].update(endpoint_cross_error_interaction=interaction,
                decomposition_residual=effect['cross_error']-target['cross_error']-control['cross_error']-interaction)
    return result


def state_half_discrepancy(states_a, states_b, *, block, maximum_working_bytes):
    """Reuse existing full-support state equations; CPU diagnostic, not a forecast."""
    import torch
    from .evaluation import score_state_law
    if type(block) is not int or block<1 or type(maximum_working_bytes) is not int or maximum_working_bytes<1:
        raise ValueError('Explicit positive state block/memory bounds required')
    arrays=[]
    for values in (states_a,states_b):
        if values is None:
            arrays.append(None)
            continue
        x=np.asarray(values)
        if x.ndim!=2 or x.shape[1]!=48 or x.dtype.kind!='f' or not len(x) or not np.isfinite(x).all():
            raise ValueError('Complete positive-RNA half states on 48-feature axis required')
        arrays.append(x)
    if any(x is None for x in arrays):
        return dict(available=False,reason='missing_positive_rna_half',
                    cells=[0 if x is None else len(x) for x in arrays])
    # MMD uses explicit b x b x 48 differences, unlike the distance-only tile.
    tile=min(block,max(len(x) for x in arrays))
    required=sum(x.nbytes+8*x.size+8*len(x) for x in arrays)+(32*48+128)*tile**2+32*tile*48
    if required>maximum_working_bytes:
        raise ValueError('Full half-support state calculation exceeds working allowance')
    tensors=[torch.as_tensor(x,dtype=torch.float64,device='cpu') for x in arrays]
    weights=[torch.full((len(x),),1/len(x),dtype=torch.float64,device='cpu') for x in arrays]
    score=score_state_law(tensors[0],weights[0],tensors[1],weights[1],None,None,
                          block=block,include_source_diagnostics=False)
    return dict(available=True,reason=None,cells=[len(x) for x in arrays],
        state_energy=score['state_energy'],spread_absolute_error=score['spread_absolute_error'],
        shape_mmd2=score['shape_mmd2'],half_a_spread=score['predicted_self_mean'],
        half_b_spread=score['observed_self_mean'],roundoff=score['roundoff'],
        accounted_array_workspace_bytes=required,independent_biological_replicates=0,
        forecasting_ceiling_claimed=False)


def joint_half_discrepancy(states_a, states_b, counts_a, counts_b, *, guide_index,
                           block, maximum_working_bytes):
    """49-moment discrepancy with actual FULL-catalog half denominators.

    The owner must ensure state rows and counts use the SAME split population.
    Captured frequencies and state support counts are checked, but labels alone
    cannot prove that provenance. Missing RNA retains the mass diagnostic only.
    """
    counts=[np.asarray(v) for v in (counts_a,counts_b)]
    if (any(c.ndim!=1 or not len(c) or c.dtype.kind not in 'iu' or np.any(c<0) for c in counts)
            or counts[0].shape!=counts[1].shape or type(guide_index) is not int
            or not 0<=guide_index<len(counts[0]) or type(block) is not int or block<1
            or type(maximum_working_bytes) is not int or maximum_working_bytes<1):
        raise ValueError('Full captured catalogs, exact guide and positive resource bounds required')
    totals=[sum(map(int,c)) for c in counts]
    if min(totals)<=0:
        return dict(available=False,reason='missing_captured_half',joint_mse=None,joint_mass_mse=None,
                    joint_nonconstant_mse=None,captured_totals=totals)
    frequencies=[int(c[guide_index])/n for c,n in zip(counts,totals)]
    arrays=[]
    for x,c in zip((states_a,states_b),counts):
        if x is None:
            arrays.append(None)
        else:
            x=np.asarray(x)
            if (x.ndim!=2 or x.shape[1]!=48 or not len(x) or x.dtype.kind!='f'
                    or not np.isfinite(x).all() or len(x)>int(c[guide_index])):
                raise ValueError('Full positive-RNA half support must fit captured guide counts')
            arrays.append(x)
    required=sum(c.nbytes for c in counts)+sum(x.nbytes for x in arrays if x is not None)+32*block*48+8*4*49
    if required>maximum_working_bytes:
        raise ValueError('Full joint-support calculation exceeds working allowance')
    result=dict(available=False,reason='missing_positive_rna_half',joint_mse=None,joint_nonconstant_mse=None,
        joint_mass_mse=float((frequencies[0]-frequencies[1])**2),captured_totals=totals,
        guide_frequencies=frequencies,positive_RNA_support=[0 if x is None else len(x) for x in arrays],
        state_count_population_matching_required=True,independent_biological_replicates=0,
        accounted_array_workspace_bytes=required)
    if any(x is None for x in arrays):return result
    moments=[]
    for x in arrays:
        total=np.zeros(48,dtype=np.float64)
        for start in range(0,len(x),block):
            total+=np.tanh(x[start:start+block].astype(np.float64)).sum(axis=0)
        moments.append(np.concatenate(([1.],total/len(x))))
    difference=frequencies[0]*moments[0]-frequencies[1]*moments[1]
    squared=np.square(difference)
    result.update(available=True,reason=None,joint_mse=float(squared.mean()),joint_nonconstant_mse=float(squared[1:].mean()))
    return result
