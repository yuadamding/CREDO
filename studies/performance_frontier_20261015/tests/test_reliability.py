"""Tiny fabricated repeatability calculations; no training/outcome access."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
import torch

from test_main_worker import no_numerics
from frontier.reliability import (REPETITIONS,balanced_assignment,support_tier,
    SplitCompositionAccumulator,empirical_catalog_pair,split_cross_error,
    molecular_split_errors,state_half_discrepancy,joint_half_discrepancy)
from frontier.evaluation import score_state_law,score_catalog,score_joint


def assignment(n=12,repeat=0):
    return balanced_assignment(tuple('cell-%03d'%i for i in range(n)),population_sha256='a'*64,repetition=repeat)


def test_twenty_global_splits_are_reproducible_order_invariant_and_not_replicates():
    rows=tuple('cell-%03d'%i for i in range(31))
    identities=[]
    for repetition in REPETITIONS:
        a=balanced_assignment(rows,population_sha256='a'*64,repetition=repetition)
        b=balanced_assignment(rows[::-1],population_sha256='a'*64,repetition=repetition)
        assert a.identity==b.identity and dict(zip(a.row_ids,a.half))==dict(zip(b.row_ids,b.half))
        assert a.half.count(0)==15 and a.half.count(1)==16
        assert a.unit_kind=='conditional_cell_sampling'
        identities.append(a.identity)
    assert len(set(identities))==20
    # Assign guides AFTER splitting: the algorithm cannot balance each guide.
    a=assignment(12)
    guides=np.array(a.half)
    assert np.bincount(guides[np.array(a.half)==0],minlength=2).tolist()==[6,0]


def test_capture_units_are_never_split_and_need_exact_provenance():
    rows=tuple('r%d'%i for i in range(9));units=('A',)*7+('B','C')
    a=balanced_assignment(rows,population_sha256='a'*64,repetition=1,
                          capture_units=units,unit_provenance_sha256='b'*64)
    for unit in set(units):assert len({h for u,h in zip(units,a.half) if u==unit})==1
    assert (a.unit_count_a,a.unit_count_b)==(1,2)
    assert sorted((a.half.count(0),a.half.count(1)))!=[4,5] # Units, not cells, balanced.
    with pytest.raises(ValueError,match='SHA'):balanced_assignment(rows,population_sha256='a'*64,repetition=0,capture_units=units)
    with pytest.raises(ValueError,match='two splitting'):balanced_assignment(rows,population_sha256='a'*64,repetition=0,
        capture_units=('A',)*9,unit_provenance_sha256='b'*64)


@pytest.mark.parametrize('kwargs',[
 dict(row_ids=('x','x'),population_sha256='a'*64,repetition=0),
 dict(row_ids=('x',),population_sha256='a'*64,repetition=0),
 dict(row_ids=('x','y'),population_sha256='a'*64,repetition=20),
 dict(row_ids=('x','y'),population_sha256='a'*64,repetition=True),
 dict(row_ids=('x','y'),population_sha256='bad',repetition=0),
 dict(row_ids=('x','y'),population_sha256='a'*64,repetition=0,unit_provenance_sha256='b'*64),
])
def test_bad_assignment_inputs_fail(kwargs):
    with pytest.raises(ValueError):balanced_assignment(**kwargs)


def fixture():
    a=assignment(12)
    counts=sparse.csr_matrix(np.array([[0,0,0],[9,1,0],[0,2,2],[0,0,3],[4,0,0],[0,8,0],
                                     [0,0,0],[1,1,1],[0,2,8],[3,0,1],[2,2,0],[0,0,1]],dtype=np.int64))
    guides=np.array([0,0,1,1,2,2,0,1,2,0,1,2],dtype=np.int64)
    return a,counts,guides


def accumulator(a,**kw):
    settings=dict(catalog_size=4,features=3,control_indices=(0,1),maximum_rows=12,
                  maximum_nnz=36,maximum_working_bytes=1024**2)
    settings.update(kw)
    return SplitCompositionAccumulator(a,**settings)


def test_streaming_compositions_match_literal_cell_equal_pooling_and_capture_zeros():
    a,counts,guides=fixture()
    one,chunked=accumulator(a),accumulator(a)
    one.update(a.row_ids,guides,counts)
    for start in range(0,len(guides),3):chunked.update(a.row_ids[start:start+3],guides[start:start+3],counts[start:start+3])
    report=one.finish();chunked.finish()
    assert report['captured_rows']==12 and report['positive_RNA_rows']==10
    assert report['independent_biological_replicates']==0
    np.testing.assert_array_equal(one.catalog_counts(),chunked.catalog_counts())
    np.testing.assert_allclose(one.sums,chunked.sums,rtol=0,atol=1e-15)
    dense=counts.toarray();depth=dense.sum(1);half=np.array(a.half)
    for h in (0,1):
        np.testing.assert_array_equal(one.catalog_counts()[h],np.bincount(guides[half==h],minlength=4))
        for guide in range(4):
            selected=(half==h)&(guides==guide)&(depth>0)
            expected=(dense[selected]/depth[selected,None]).mean(0) if selected.any() else None
            actual=one.guide_compositions(guide)[h]
            if expected is None:assert actual is None
            else:np.testing.assert_allclose(actual,expected)
        selected=(half==h)&np.isin(guides,[0,1])&(depth>0)
        np.testing.assert_allclose(one.control_compositions()[h],(dense[selected]/depth[selected,None]).mean(0))
    assert one.guide_compositions(3)==(None,None)
    assert one.catalog_counts()[:,3].tolist()==[0,0]
    with pytest.raises(ValueError,match='Closed'):one.update(a.row_ids,guides,counts)


def test_global_missing_guide_halves_are_preserved_not_rebalanced():
    a=assignment(8);guides=np.array(a.half)
    value=accumulator(a)
    value.update(a.row_ids,guides,sparse.csr_matrix(np.ones((8,3),dtype=np.int64)))
    value.finish()
    assert value.guide_compositions(0)[1] is None
    assert value.guide_compositions(1)[0] is None
    score=empirical_catalog_pair(*value.catalog_counts())
    assert score['catalog_size']==4 and score['A_predicts_B']['extended_cross_entropy']=='+infinity'


def test_memory_guard_precedes_allocation(monkeypatch):
    monkeypatch.setattr(np,'zeros',lambda *_a,**_k:pytest.fail('Should not allocate'))
    with pytest.raises(ValueError,match='memory before allocation'):accumulator(assignment(),maximum_working_bytes=1)


@pytest.mark.parametrize('kind',['wrong_rows','wrong_guides','float_counts','too_many_nnz','wrong_gene_axis'])
def test_bad_chunk_closes_accumulator(kind):
    a,counts,guides=fixture();value=accumulator(a)
    rows=a.row_ids
    if kind=='wrong_rows':rows=rows[::-1]
    elif kind=='wrong_guides':guides[0]=4
    elif kind=='float_counts':counts=counts.astype(float)
    elif kind=='too_many_nnz':value=accumulator(a,maximum_nnz=1)
    elif kind=='wrong_gene_axis':counts=counts[:,:2]
    with pytest.raises(ValueError):value.update(rows,guides,counts)
    with pytest.raises(ValueError,match='failed'):value.update(a.row_ids,guides,counts)
    with pytest.raises(ValueError):value.finish()


def test_incomplete_inventory_and_missing_controls_are_explicit():
    a,counts,guides=fixture();value=accumulator(a,control_indices=())
    value.update(a.row_ids[:3],guides[:3],counts[:3])
    with pytest.raises(ValueError,match='complete'):value.finish()
    value.update(a.row_ids[3:],guides[3:],counts[3:]);value.finish()
    assert value.control_compositions()==(None,None)


def test_catalog_full_axes_and_existing_score_agree_without_pseudocounts():
    a=np.array([3,2,1],dtype=np.int64);b=np.array([4,1,0],dtype=np.int64)
    score=empirical_catalog_pair(a,b)
    assert score['B_predicts_A']['extended_cross_entropy']=='+infinity'
    reference=score_catalog(torch.tensor(np.log(a/a.sum()),dtype=torch.float64),torch.tensor(b),expected_catalog_size=3)
    assert score['A_predicts_B']['cross_entropy_nats']==pytest.approx(reference['cross_entropy_nats'])
    assert score['A_predicts_B']['quadratic_score']==pytest.approx(reference['count_quadratic'])
    assert empirical_catalog_pair(a,np.zeros(3,dtype=np.int64))['reason']=='missing_captured_half'
    json.dumps(score,allow_nan=False) # Infinity encoded explicitly, not nonstandard JSON.


def test_cross_error_retains_negative_values_and_available_single_half_scores():
    p=np.array([0.,0.]);a=np.array([1.,-1.]);b=-a
    result=split_cross_error(p,a,b)
    assert result['cross_error']==-1 and result['arithmetic_residual']==0
    assert not result['noise_corrected_risk_claimed'] and not result['negative_values_clipped']
    missing=split_cross_error(p,a,None)
    assert missing['mse_a']==1 and missing['cross_error'] is None
    assert split_cross_error(None,a,b)['half_discrepancy_mse']==4


def test_fixed_population_half_cross_error_is_not_its_full_mean_error():
    population=np.array([[0.,0.],[2.,2.],[4.,4.],[6.,6.]])
    pred=population.mean(0)
    half_a,half_b=population[:2].mean(0),population[2:].mean(0)
    result=split_cross_error(pred,half_a,half_b)
    assert np.square(pred-population.mean(0)).mean()==0
    assert result['cross_error']==-4
    assert not result['assumptions_verified']


def test_molecular_shared_control_decomposition_and_pool_before_transform():
    t,a,b,c,ca,cb=[np.array(v,dtype=np.float64) for v in
        ([.2,.3,.5],[.1,.4,.5],[.4,.4,.2],[.3,.3,.4],[.1,.1,.8],[.6,.2,.2])]
    report=molecular_split_errors(t,a,b,c,ca,cb)
    for scale in ('mean_composition','log1p_10000_mean_composition'):
        row=report[scale]
        assert row['decomposition_residual']==pytest.approx(0,abs=1e-14)
        assert row['effect']['arithmetic_residual']==pytest.approx(0,abs=1e-14)
    assert report['control_reduction_scope']=='one_pooled_control_per_split_population'
    missing=molecular_split_errors(t,a,b,c,None,cb)
    assert missing['mean_composition']['target']['available']
    assert not missing['mean_composition']['effect']['available']
    with pytest.raises(ValueError):molecular_split_errors(t,a,b,c,ca,np.array([.2,.2,.2]))


def test_state_half_discrepancy_matches_existing_full_support_geometry_and_tiling():
    a=np.arange(5*48,dtype=np.float64).reshape(5,48)/100
    b=np.arange(7*48,dtype=np.float64).reshape(7,48)/80
    first=state_half_discrepancy(a,b,block=2,maximum_working_bytes=1024**2)
    second=state_half_discrepancy(a,b,block=4,maximum_working_bytes=1024**2)
    reference=score_state_law(torch.tensor(a),torch.full((5,),.2,dtype=torch.float64),
        torch.tensor(b),torch.full((7,),1/7,dtype=torch.float64),None,None,block=2,include_source_diagnostics=False)
    for key in ('state_energy','spread_absolute_error','shape_mmd2'):
        assert first[key]==pytest.approx(reference[key],abs=1e-14)
        assert first[key]==pytest.approx(second[key],abs=1e-14)
    assert first['cells']==[5,7] and not first['forecasting_ceiling_claimed']
    assert state_half_discrepancy(a,None,block=2,maximum_working_bytes=1024**2)['reason']=='missing_positive_rna_half'
    with pytest.raises(ValueError,match='allowance'):state_half_discrepancy(a,b,block=2,maximum_working_bytes=1)


def test_state_cpu_placement_ignores_default_device():
    values=np.ones((3,48),dtype=np.float32)
    with torch.device('meta'):
        result=state_half_discrepancy(values,values,block=2,maximum_working_bytes=1024**2)
    assert result['state_energy']==0


def test_joint_reuses_complete_catalog_denominator_and_existing_49_moment_metric():
    a=np.arange(3*48,dtype=float).reshape(3,48)/70
    b=np.arange(4*48,dtype=float).reshape(4,48)/50
    ca=np.array([3,7,90],dtype=np.int64);cb=np.array([4,6,190],dtype=np.int64)
    result=joint_half_discrepancy(a,b,ca,cb,guide_index=0,block=2,maximum_working_bytes=1024**2)
    assert result['guide_frequencies']==[.03,.02] # Not .3 and .4 panel-normalized.
    reference=score_joint(torch.tensor(a),torch.full((3,),1/3,dtype=torch.float64),.03,
        torch.tensor(b),torch.full((4,),.25,dtype=torch.float64),.02)
    for key in ('joint_mse','joint_mass_mse','joint_nonconstant_mse'):
        assert result[key]==pytest.approx(reference[key],abs=1e-15)
    missing=joint_half_discrepancy(a,None,ca,cb,guide_index=0,block=2,maximum_working_bytes=1024**2)
    assert missing['joint_mse'] is None and missing['joint_mass_mse']==pytest.approx(.0001)
    with pytest.raises(ValueError,match='support'):joint_half_discrepancy(a,b,np.array([2,7,90]),cb,
        guide_index=0,block=2,maximum_working_bytes=1024**2)


@pytest.mark.parametrize('count,tier',[(0,'0-15'),(15,'0-15'),(16,'16-63'),(63,'16-63'),(64,'64-255'),(255,'64-255'),(256,'256+')])
def test_source_support_tier_boundaries(count,tier):assert support_tier(count)==tier


def test_protocol_freezes_twenty_splits_but_not_scientific_execution():
    doc=json.loads((Path(__file__).resolve().parents[1]/'RELIABILITY_PROTOCOL.json').read_text())
    assert tuple(doc['repetition_ids'])==REPETITIONS and doc['repetitions']==20
    assert doc['execution']['scientific_launch_authorized'] is False
    assert doc['custody']['method_revision_from_outer_diagnostics'] is False
    assert doc['cross_error']['default_claim']=='Descriptive split cross-error only'
