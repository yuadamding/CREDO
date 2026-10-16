"""Existing fold caches/query populations -> the two simple comparator inputs.

No files, encoders, decoders, optimizers, releases or scheduler are opened here.
The caller must already hold source-bound cache/query admission and leases.
Array identity checks do not create permission or prediction custody. Fitting
and query paths are deliberately separate; the latter has no endpoint argument.
"""

from dataclasses import dataclass
import math

import numpy as np

from . import abundance_head as ah
from . import ridge_readout as rr
from .fitting_provider import FoldFittingCache, PopulationArrays, _hash
from .forecast import ForecastSource
from .native_inputs import FoldInputs, _hash as native_hash
from .query_compiler import CompiledQuery
from .training_owner import ObservationBinding


@dataclass(frozen=True)
class ReferenceLimits:
    maximum_host_bytes: int
    state_block_rows: int
    gene_block: int

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in self.__dict__.values()):
            raise ValueError('Explicit positive reference host/block bounds required')


@dataclass(frozen=True)
class ReferenceBindings:
    fold: FoldInputs
    feature_axis_sha256: str
    observation_binding: ObservationBinding
    ridge: rr.RidgeBinding
    abundance: ah.HeadBinding

    def __post_init__(self):
        if (not isinstance(self.fold,FoldInputs) or not isinstance(self.observation_binding,ObservationBinding)
                or not isinstance(self.ridge,rr.RidgeBinding)
                or not isinstance(self.abundance,ah.HeadBinding)
                or self.fold.training_donors!=self.ridge.fitting_donors
                or self.ridge.fitting_donors!=self.abundance.fitting_donors
                or native_hash(list(self.ridge.gene_ids))!=self.feature_axis_sha256
                or self.ridge.representation_sha256!=self.abundance.representation_sha256
                or self.ridge.representation_sha256!=self.observation_binding.artifact_sha256
                or self.ridge.fitting_input_sha256!=self.abundance.fitting_input_sha256
                or self.ridge.synthetic_fixture!=self.abundance.synthetic_fixture):
            raise ValueError('Exact common fold, ordered gene axis, representation and input bindings required')


def _admit(limits, *, host, donors, guides, rows):
    if not isinstance(limits,ReferenceLimits): raise ValueError('Explicit reference limits required')
    # Conservative array workspace, not measured peak process RSS or scheduler admission.
    extra=(64*donors*guides*(96+194+100) + 64*limits.state_block_rows*48
           + 64*rows*limits.gene_block + 128*(rows+guides))
    if host+extra>limits.maximum_host_bytes:
        raise ValueError('Cache plus comparator workspace exceeds host bound before allocation')
    return extra


def _early_summary(population, binding, limits):
    g=len(binding.catalog)
    moments=np.zeros((g,96),dtype=np.float64)
    for i,n in enumerate(population.positive_counts):
        if n:
            if binding.synthetic_fixture and n>128:
                raise ValueError('Fabricated per-law particle allowance exceeded')
            moments[i]=rr.source_moments(population.states[population.offsets[i]:population.offsets[i+1]],
                block_rows=limits.state_block_rows,maximum_working_bytes=limits.maximum_host_bytes)
    moments.setflags(write=False)
    ref=population.reference
    result=ah.SourceCatalog(ref.donor,ref.condition,binding.catalog_sha256,binding.representation_sha256,
        _hash(dict(raw=ref.raw_summary_sha256,latent=ref.latent_sha256,input=ref.source_input_sha256)),
        population.captured_counts,population.positive_counts,moments)
    result.validate(binding)
    return result


def _pool_moments(summary, controls):
    n=sum(int(summary.positive_rna_counts[i]) for i in controls)
    if not n: return None
    mean=np.zeros(48);second=np.zeros(48)
    for i in controls:
        if summary.positive_rna_counts[i]:
            w=int(summary.positive_rna_counts[i])/n
            m=summary.moments[i,:48]
            mean+=w*m;second+=w*(summary.moments[i,48:]+m*m)
    variance=second-mean*mean
    tolerance=64*np.finfo(float).eps*max(1,len(controls))
    if np.any(variance < -tolerance): raise ValueError('Invalid pooled bounded variance')
    return np.concatenate((mean,np.maximum(variance,0)))


def _log_sources(summary):
    counts=summary.captured_counts
    return np.log(counts.astype(np.float64)+.5)-math.log(sum(int(n) for n in counts)+.5*len(counts))


def _ridge_features(summary, controls, route):
    pool=_pool_moments(summary,controls)
    if pool is None: return None
    logs=_log_sources(summary)
    features=np.empty((len(route),rr.FEATURES),dtype=np.float64)
    for row,i in enumerate(route):
        if i is None:
            moments=pool
            frequency=math.fsum(math.exp(logs[c]) for c in controls)
            support=sum(int(summary.positive_rna_counts[c]) for c in controls)
        else:
            moments=summary.moments[i];frequency=math.exp(logs[i]);support=int(summary.positive_rna_counts[i])
        features[row]=rr.source_features(moments,pool,source_frequency=frequency,positive_rna_support=support)
    features.setflags(write=False)
    return features


def _readout_block(population, controls, route, start, stop):
    result=np.empty((len(route),stop-start),dtype=np.float64)
    for row,i in enumerate(route):
        if i is not None:
            if not population.positive_counts[i]: raise ValueError('Missing guide RNA is not a zero readout')
            composition=population.mean_compositions[i,start:stop]
        else:
            n=sum(int(population.positive_counts[c]) for c in controls)
            if not n: raise ValueError('Missing pooled control cannot be imputed')
            composition=np.zeros(stop-start,dtype=np.float64)
            for c in controls:
                if population.positive_counts[c]:
                    composition+=population.mean_compositions[c,start:stop]*(int(population.positive_counts[c])/n)
        result[row]=np.log1p(10000*composition)
    return result


@dataclass(frozen=True)
class FittingReferenceInputs:
    bindings: ReferenceBindings
    cache: FoldFittingCache
    abundance_catalogs: tuple
    ridge_rows: tuple
    ridge_features: np.ndarray
    routes: tuple
    controls: tuple
    limits: ReferenceLimits
    additional_workspace_estimate: int

    def gene_blocks(self):
        """Bounded full-gene responses, preserving both donor and gene order."""
        genes=self.bindings.ridge.gene_ids
        keys=tuple((r.donor,r.row_id) for r in self.ridge_rows)
        for start in range(0,len(genes),self.limits.gene_block):
            self.cache.assert_unchanged()
            stop=min(start+self.limits.gene_block,len(genes))
            source=[];endpoint=[]
            for donor,route in self.routes:
                source.append(_readout_block(self.cache.populations[(donor,'Stim8hr')],self.controls,route,start,stop))
                endpoint.append(_readout_block(self.cache.populations[(donor,'Stim48hr')],self.controls,route,start,stop))
            yield rr.GeneBlock(start,genes[start:stop],keys,np.concatenate(source),np.concatenate(endpoint))
        self.cache.assert_unchanged()


def prepare_fitting_inputs(cache, gene_ids, *, limits):
    if not isinstance(cache,FoldFittingCache): raise ValueError('Verified fitting cache required')
    cache.assert_unchanged();m=cache.manifest
    if type(gene_ids) is not tuple or len(gene_ids)!=m.features or native_hash(list(gene_ids))!=m.feature_axis_sha256:
        raise ValueError('Actual ordered native RNA feature axis required before arrays')
    bindings=ReferenceBindings(m.fold,m.feature_axis_sha256,m.observation_binding,
        rr.RidgeBinding(m.fold.training_donors,gene_ids,m.observation_binding.artifact_sha256,m.identity,m.synthetic_fixture),
        ah.HeadBinding(m.fold.training_donors,m.catalog,m.observation_binding.artifact_sha256,m.identity,m.synthetic_fixture))
    g=len(m.catalog);d=len(m.fold.training_donors)
    extra=_admit(limits,host=cache.host_bytes,donors=d,guides=g,rows=d*(g+1))
    controls=tuple(i for i,c in enumerate(m.catalog) if c.is_control)
    rows=[];routes=[];features=[];catalogs=[]
    for donor in m.fold.training_donors:
        source=cache.populations[(donor,'Stim8hr')];endpoint=cache.populations[(donor,'Stim48hr')]
        summary=_early_summary(source,bindings.abundance,limits)
        catalogs.append(ah.FittingCatalog(summary,endpoint.captured_counts,endpoint.reference.raw_summary_sha256))
        targets=tuple(i for i,c in enumerate(m.catalog) if not c.is_control and source.positive_counts[i]>0 and endpoint.positive_counts[i]>0)
        if not targets: raise ValueError('Missing paired targeting support for ridge fitting')
        route=targets+(None,);routes.append((donor,route))
        rows.extend(rr.TrainingRow(donor,f'guide:{i}',m.catalog[i].target_id) for i in targets)
        rows.append(rr.TrainingRow(donor,'pooled_control',None,'control'))
        f=_ridge_features(summary,controls,route)
        if f is None: raise ValueError('Missing fitting source control pool')
        features.append(f)
    all_features=np.concatenate(features);all_features.setflags(write=False)
    cache.assert_unchanged()
    return FittingReferenceInputs(bindings,cache,tuple(catalogs),tuple(rows),all_features,
                                   tuple(routes),controls,limits,extra)


@dataclass(frozen=True)
class QueryReferenceInputs:
    bindings: ReferenceBindings
    query: CompiledQuery
    abundance_source: ah.SourceCatalog
    row_ids: tuple
    route: tuple
    ridge_features: np.ndarray | None
    ridge_unavailable_reason: str | None
    controls: tuple
    limits: ReferenceLimits
    additional_workspace_estimate: int

    def early_gene_blocks(self):
        """Early RNA only. No query endpoint or paired-support filter exists here."""
        if self.ridge_features is None:
            raise ValueError(self.ridge_unavailable_reason)
        self.query.source.assert_unchanged()
        genes=self.bindings.ridge.gene_ids
        for start in range(0,len(genes),self.limits.gene_block):
            if self.query.population.mean_compositions.flags.writeable:
                raise ValueError('Query early RNA storage is no longer read-only')
            stop=min(start+self.limits.gene_block,len(genes))
            yield start,genes[start:stop],_readout_block(self.query.population,self.controls,self.route,start,stop)
        self.query.source.assert_unchanged()
        p=self.query.population;s=self.query.source
        if p.raw_identity(catalog_sha256=s.catalog_sha256,feature_axis_sha256=s.feature_axis_sha256)!=p.reference.raw_summary_sha256:
            raise ValueError('Query early RNA changed during iteration')


def prepare_query_inputs(query, bindings, *, limits):
    if not isinstance(query,CompiledQuery) or not isinstance(bindings,ReferenceBindings):
        raise ValueError('Compiled early query and common reference bindings required')
    source,pop=query.source,query.population
    if (not isinstance(source,ForecastSource) or not isinstance(pop,PopulationArrays) or source.fold!=bindings.fold
            or source.catalog!=bindings.abundance.catalog
            or source.observation_binding!=bindings.observation_binding
            or source.feature_axis_sha256!=bindings.feature_axis_sha256
            or source.synthetic_fixture!=bindings.ridge.synthetic_fixture
            or pop.reference!=source.reference):
        raise ValueError('Held-out source population, representation or axis differs')
    fields=('captured_counts','positive_counts','offsets','states')
    if any(getattr(pop,f) is not getattr(source,f) for f in fields):
        raise ValueError('Compiled query must share its actual immutable source arrays')
    g=len(source.catalog);j=len(bindings.ridge.gene_ids)
    means=pop.mean_compositions
    if (not isinstance(means,np.ndarray) or means.dtype!=np.float64 or means.shape!=(g,j)
            or not means.flags.c_contiguous or means.flags.writeable):
        raise ValueError('Immutable complete early RNA summary matrix required')
    host=sum(getattr(pop,f).nbytes for f in (*fields,'mean_compositions'))
    extra=_admit(limits,host=host,donors=1,guides=g,rows=g+1)
    source.assert_unchanged()
    # Match the producer's raw identity; do not substitute a row-order/symbol guess.
    if pop.raw_identity(catalog_sha256=source.catalog_sha256,feature_axis_sha256=source.feature_axis_sha256)!=pop.reference.raw_summary_sha256:
        raise ValueError('Query raw mean identity differs')
    for start in range(0,g,128):
        q=means[start:start+128];present=pop.positive_counts[start:start+128]>0
        if (not np.isfinite(q).all() or np.any(q<0) or np.any(q[~present]!=0)
                or not np.allclose(q[present].sum(1),1.,atol=1e-10,rtol=0)):
            raise ValueError('Query RNA support and normalized means disagree')
    summary=_early_summary(pop,bindings.abundance,limits)
    controls=tuple(i for i,c in enumerate(source.catalog) if c.is_control)
    targets=tuple(i for i,c in enumerate(source.catalog) if not c.is_control and source.positive_counts[i]>0)
    route=targets+(None,)
    features=_ridge_features(summary,controls,route)
    rows=tuple(f'guide:{i}' for i in targets)+('pooled_control',)
    source.assert_unchanged()
    return QueryReferenceInputs(bindings,query,summary,rows,route,features,
        None if features is not None else 'missing_positive_RNA_source_control_pool',controls,limits,extra)
