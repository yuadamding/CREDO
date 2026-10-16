"""Fabricated cache/query integration; no file payloads, models, fits or GPU."""

from dataclasses import replace

import numpy as np
import pytest

from frontier import abundance_head as ah
from frontier import ridge_readout as rr
from frontier import simple_reference_inputs as sri
from frontier.fitting_provider import FoldFittingCache
from frontier.forecast import ForecastSource
from frontier.native_inputs import _hash as native_hash
from frontier.query_compiler import CompiledQuery
from test_fitting_provider import fixture as cache_fixture, readonly
from test_main_worker import no_numerics
from test_ridge_readout import forbid_fit_solvers
from test_abundance_head import no_count_optimizer


CAP=64*1024**2
LIMITS=sri.ReferenceLimits(CAP,4,2)


def reidentify(p, catalog_sha, feature_sha, observation):
    ref=replace(p.reference,raw_summary_sha256=p.raw_identity(catalog_sha256=catalog_sha,feature_axis_sha256=feature_sha))
    p=replace(p,reference=ref)
    return replace(p,reference=replace(ref,latent_sha256=p.latent_identity(catalog_sha256=catalog_sha,observation_binding=observation)))


@pytest.fixture
def cache():
    old,populations=cache_fixture()
    genes=('gene0','gene1','gene2','gene3','gene_β')
    axis=native_hash(list(genes))
    p=tuple(reidentify(x,old.manifest.catalog_sha256,axis,old.manifest.observation_binding) for x in populations)
    manifest=replace(old.manifest,feature_axis_sha256=axis,populations=tuple(x.reference for x in p))
    result=FoldFittingCache(manifest,p,expected_manifest_sha256=manifest.identity,limits=old.limits)
    return result,genes


def query_from(cache, *, no_controls=False):
    m=cache.manifest;p=cache.populations[('D3','Stim8hr')]
    ref=replace(p.reference,donor='D1',source_id='D1_Stim8hr',source_access_sha256='f'*64)
    p=replace(p,reference=ref)
    if no_controls:
        positive=p.positive_counts.copy();controls=[i for i,g in enumerate(m.catalog) if g.is_control]
        positive[controls]=0
        parts=[p.states[p.offsets[i]:p.offsets[i+1]] for i in range(len(m.catalog)) if positive[i]]
        means=p.mean_compositions.copy();means[controls]=0
        p=replace(p,positive_counts=readonly(positive),offsets=readonly(np.r_[0,np.cumsum(positive)]),
                  states=readonly(np.concatenate(parts)),mean_compositions=readonly(means))
    p=reidentify(p,m.catalog_sha256,m.feature_axis_sha256,m.observation_binding)
    source=ForecastSource(m.fold,m.catalog,m.observation_binding,m.feature_axis_sha256,p.reference,
                          p.captured_counts,p.positive_counts,p.offsets,p.states,True)
    return CompiledQuery(source,p,{'fabricated_metadata':True,'scientific_admission':False})


def test_fitting_adapters_share_actual_fold_and_preserve_whole_catalog(cache):
    original,genes=cache
    inputs=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    assert inputs.bindings.ridge.fitting_input_sha256==original.manifest.identity
    assert inputs.bindings.abundance.fitting_input_sha256==original.manifest.identity
    assert inputs.bindings.ridge.gene_ids==genes
    assert len(inputs.abundance_catalogs)==2
    for c in inputs.abundance_catalogs:
        assert c.source.captured_counts is original.populations[(c.source.donor,'Stim8hr')].captured_counts
        assert c.endpoint_counts is original.populations[(c.source.donor,'Stim48hr')].captured_counts
        assert len(c.endpoint_counts)==7
        assert np.all(c.source.moments[c.source.positive_rna_counts==0]==0)
    assert ('D3','guide:5') not in [(r.donor,r.row_id) for r in inputs.ridge_rows]
    design=ah.prepare_design(inputs.bindings.abundance,inputs.abundance_catalogs,maximum_working_bytes=CAP)
    value,gradient=ah.objective_and_gradient(np.zeros(design.parameter_count),design,.1)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert len(design.observed_frequencies[0])==7


def test_ridge_rows_and_weights_match_existing_target_macro_measure(cache):
    original,genes=cache;inputs=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    weights=rr.training_weights(inputs.bindings.ridge,inputs.ridge_rows)
    lookup={law:i for i,law in enumerate(original.laws)}
    for row,weight in zip(inputs.ridge_rows,weights):
        if row.role=='control': assert weight==.25
        else:
            i=lookup[(row.donor,int(row.row_id.split(':')[1]))]
            assert weight==pytest.approx(original.molecular_weights[i]/2)
    d=rr.prepare_design(inputs.bindings.ridge,inputs.ridge_rows,inputs.ridge_features,
                        maximum_rows=64,maximum_working_bytes=CAP)
    assert d.standardized.shape==(8,194)


def test_gene_blocks_pool_positive_rna_cells_before_transform(cache):
    original,genes=cache;inputs=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    blocks=list(inputs.gene_blocks())
    assert [b.start for b in blocks]==[0,2,4]
    assert sum((b.gene_ids for b in blocks),())==genes
    source=np.concatenate([b.source_readout for b in blocks],axis=1)
    endpoint=np.concatenate([b.endpoint_readout for b in blocks],axis=1)
    for row_id,row in enumerate(inputs.ridge_rows):
        for condition,actual in (('Stim8hr',source),('Stim48hr',endpoint)):
            p=original.populations[(row.donor,condition)]
            if row.role=='control':
                indices=inputs.controls;n=sum(int(p.positive_counts[i]) for i in indices)
                expected=sum(p.mean_compositions[i]*int(p.positive_counts[i])/n for i in indices)
                if condition=='Stim8hr':
                    wrong=sum(p.mean_compositions[i] for i in indices)/len(indices)
                    assert not np.allclose(expected,wrong,atol=1e-12,rtol=0)
            else:expected=p.mean_compositions[int(row.row_id.split(':')[1])]
            assert actual[row_id]==pytest.approx(np.log1p(10000*expected))


def test_control_moments_are_cell_pool_moments_not_guide_average(cache):
    original,genes=cache;inputs=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    for donor in original.manifest.fold.training_donors:
        p=original.populations[(donor,'Stim8hr')]
        cells=np.concatenate([p.states[p.offsets[i]:p.offsets[i+1]] for i in inputs.controls])
        x=np.tanh(cells.astype(np.float64))
        expected=np.r_[x.mean(0),x.var(0)]
        row=next(i for i,r in enumerate(inputs.ridge_rows) if r.donor==donor and r.role=='control')
        assert inputs.ridge_features[row,:96]==pytest.approx(expected,abs=1e-14)
        assert inputs.ridge_features[row,96:192]==pytest.approx(expected,abs=1e-14)


def test_query_has_source_only_support_no_late_panel_filter(cache):
    original,genes=cache;fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=query_from(original)
    inputs=sri.prepare_query_inputs(query,fitting.bindings,limits=LIMITS)
    assert inputs.row_ids==('guide:0','guide:1','guide:2','guide:5','pooled_control')
    assert inputs.ridge_unavailable_reason is None
    assert inputs.ridge_features.shape==(5,194)
    assert len(inputs.abundance_source.captured_counts)==7
    blocks=list(inputs.early_gene_blocks())
    assert sum((b[1] for b in blocks),())==genes
    all_readout=np.concatenate([b[2] for b in blocks],axis=1)
    assert all_readout[3]==pytest.approx(np.log1p(10000*query.population.mean_compositions[5]))
    model=rr.RidgeReadout(fitting.bindings.ridge,.1,np.zeros(194),np.ones(194),np.zeros(5),np.zeros((194,5)))
    for start,axis,source_readout in blocks:
        predicted=rr.predict_block(model,inputs.ridge_features,source_readout,query_donor='D1',condition='Stim8hr',
            representation_sha256=model.binding.representation_sha256,start=start,gene_ids=axis,maximum_rows=64,maximum_working_bytes=CAP)
        assert predicted==pytest.approx(source_readout)


def test_count_query_adapter_to_forecast_retains_unmodeled_and_zero_categories(cache):
    original,genes=cache;fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=sri.prepare_query_inputs(query_from(original),fitting.bindings,limits=LIMITS)
    d=ah.prepare_design(fitting.bindings.abundance,fitting.abundance_catalogs,maximum_working_bytes=CAP)
    model=ah.AbundanceHead(d.binding,d.mean,d.scale,d.target_ids,d.target_index,
                          np.zeros(d.parameter_count),.1,{'literal_zero_parameters_not_fit':True})
    logs=ah.predict_catalog(model,query.abundance_source,maximum_working_bytes=CAP)
    counts=query.abundance_source.captured_counts.astype(float)
    assert np.exp(logs)==pytest.approx((counts+.5)/(sum(counts)+.5*len(counts)))
    assert len(logs)==len(original.manifest.catalog)==7


def test_query_memory_and_full_observation_binding_fail_before_state_scan(cache,monkeypatch):
    original,genes=cache;fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=query_from(original)
    monkeypatch.setattr(ForecastSource,'assert_unchanged',lambda *_a:pytest.fail('State scan before admission'))
    with pytest.raises(ValueError,match='host bound'):
        sri.prepare_query_inputs(query,fitting.bindings,limits=replace(LIMITS,maximum_host_bytes=1))
    changed=replace(fitting.bindings,observation_binding=replace(fitting.bindings.observation_binding,state_sha256='e'*64))
    with pytest.raises(ValueError,match='representation'):
        sri.prepare_query_inputs(query,changed,limits=LIMITS)


def test_missing_source_control_leaves_count_input_valid_not_imputed(cache):
    original,genes=cache;fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=query_from(original,no_controls=True)
    inputs=sri.prepare_query_inputs(query,fitting.bindings,limits=LIMITS)
    assert inputs.ridge_features is None
    assert inputs.ridge_unavailable_reason=='missing_positive_RNA_source_control_pool'
    assert len(inputs.abundance_source.captured_counts)==7
    ah.source_features(fitting.bindings.abundance,inputs.abundance_source,maximum_working_bytes=CAP)
    with pytest.raises(ValueError,match='control_pool'):list(inputs.early_gene_blocks())


def test_axis_and_memory_fail_before_moment_computation(cache,monkeypatch):
    original,genes=cache
    monkeypatch.setattr(sri,'_early_summary',lambda *_a:pytest.fail('Moment work not admitted'))
    with pytest.raises(ValueError,match='axis'):
        sri.prepare_fitting_inputs(original,genes[::-1],limits=LIMITS)
    with pytest.raises(ValueError,match='host bound'):
        sri.prepare_fitting_inputs(original,genes,limits=replace(LIMITS,maximum_host_bytes=1))


@pytest.mark.parametrize('problem',['raw_identity','late_population','different_arrays','wrong_fold','wrong_gene_axis'])
def test_query_rejects_bad_bridge(cache,problem):
    original,genes=cache;fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=query_from(original);bindings=fitting.bindings
    if problem=='raw_identity':query=replace(query,population=replace(query.population,mean_compositions=readonly(query.population.mean_compositions*.5)))
    elif problem=='late_population':query=replace(query,population=original.populations[('D3','Stim48hr')])
    elif problem=='different_arrays':query=replace(query,population=replace(query.population,states=readonly(query.population.states)))
    elif problem=='wrong_fold':
        ridge=replace(bindings.ridge,fitting_donors=('D2','D4'))
        count=replace(bindings.abundance,fitting_donors=('D2','D4'))
        bindings=replace(bindings,fold=replace(bindings.fold,training_donors=('D2','D4')),ridge=ridge,abundance=count)
    else:
        with pytest.raises(ValueError):replace(bindings,ridge=replace(bindings.ridge,gene_ids=genes[::-1]))
        return
    with pytest.raises(ValueError):sri.prepare_query_inputs(query,bindings,limits=LIMITS)


def test_query_source_not_rehashed_for_every_gene_block_and_early_mutation_detected(cache,monkeypatch):
    original,genes=cache;fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=query_from(original);inputs=sri.prepare_query_inputs(query,fitting.bindings,limits=replace(LIMITS,gene_block=1))
    original_check=ForecastSource.assert_unchanged;calls=[]
    def check(self):calls.append(self.identity);return original_check(self)
    monkeypatch.setattr(ForecastSource,'assert_unchanged',check)
    assert len(list(inputs.early_gene_blocks()))==5
    assert len(calls)==2
    iterator=inputs.early_gene_blocks();next(iterator)
    query.population.mean_compositions.setflags(write=True)
    try:
        with pytest.raises(ValueError,match='read-only'):next(iterator)
    finally:query.population.mean_compositions.setflags(write=False)
