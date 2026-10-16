"""Synthetic arithmetic only. Every actual coefficient solver is forbidden.

Mocked solve tests verify block orchestration, NOT the Cholesky implementation.
No local fit allowance is created by using a deterministic linear solve.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import torch

from frontier import ridge_readout as rr
from frontier.evaluation import score_molecular_law
from frontier.planner import inner_edges
from test_main_worker import no_numerics


CAP = 8*1024**2


@pytest.fixture(autouse=True)
def forbid_fit_solvers(monkeypatch, request):
    def forbid(*_args, **_kwargs):
        pytest.fail('No numerical ridge fit is admitted by this arithmetic-only suite')
    for module, names in ((np.linalg, ('solve', 'lstsq', 'pinv', 'inv', 'cholesky')),
                          (scipy.linalg, ('solve', 'lstsq', 'pinv', 'inv', 'cho_factor', 'cho_solve')),
                          (rr, ('cho_factor', 'cho_solve'))):
        for name in names: monkeypatch.setattr(module, name, forbid)
    request.node.user_properties.append(('numerical_linear_fits', 0))


def binding():
    return rr.RidgeBinding(('D3', 'D4'), ('geneA', 'geneB', 'geneC'), 'a'*64, 'b'*64, True)


def rows():
    return (rr.TrainingRow('D3', 'g1', 't1'), rr.TrainingRow('D3', 'g2', 't1'),
            rr.TrainingRow('D3', 'g3', 't2'), rr.TrainingRow('D3', 'pooled_control', None, 'control'),
            rr.TrainingRow('D4', 'g1', 't1'), rr.TrainingRow('D4', 'pooled_control', None, 'control'))


def design():
    features = np.zeros((6, rr.FEATURES))
    features[:, 0] = [-2., -1., 0., 1., 2., 3.]
    features[:, 1] = [1., 1., 2., 2., 3., 3.]
    features[:, 2] = .3
    return rr.prepare_design(binding(), rows(), features, maximum_rows=64, maximum_working_bytes=CAP)


def test_donor_target_guide_reduction_and_single_control():
    weights = rr.training_weights(binding(), rows())
    assert weights == pytest.approx([1/16, 1/16, 1/8, 1/4, 1/4, 1/4])
    assert sum(weights[:4]) == sum(weights[4:]) == .5
    for changed in (rows()[:-1], rows()+rows()[:1], rows()[:4],
                    (replace(rows()[0], donor='D1'),)+rows()[1:]):
        with pytest.raises(ValueError): rr.training_weights(binding(), changed)


@pytest.mark.parametrize('change', [
    dict(fitting_donors=('D4', 'D3')), dict(fitting_donors=('D1', 'D2', 'D3', 'D4')),
    dict(fitting_donors=('D3', 'D3')), dict(gene_ids=('g', 'g')),
    dict(representation_sha256='wrong'), dict(synthetic_fixture=False),
])
def test_binding_rejects_wrong_scope(change):
    with pytest.raises(ValueError): replace(binding(), **change)


def test_full_support_source_moments_and_early_only_features():
    states = np.arange(7*48, dtype=float).reshape(7,48)/100
    m = rr.source_moments(states, block_rows=2, maximum_working_bytes=CAP)
    assert m[:48] == pytest.approx(np.tanh(states).mean(0))
    assert m[48:] == pytest.approx(np.tanh(states).var(0))
    features = rr.source_features(m, m, source_frequency=.01, positive_rna_support=7)
    assert len(features) == 194
    assert features[-2:] == pytest.approx([np.log(.01), np.log1p(7)])
    with pytest.raises(ValueError): rr.source_moments(states, block_rows=2, maximum_working_bytes=1)
    with pytest.raises(ValueError): rr.source_features(m, m, source_frequency=0, positive_rna_support=7)
    with pytest.raises(ValueError): rr.source_features(m, m, source_frequency=.1, positive_rna_support=0)
    bad = m.copy(); bad[48:] = 1
    with pytest.raises(ValueError): rr.source_features(bad, m, source_frequency=.1, positive_rna_support=7)


def test_fitting_only_scaling_constant_columns_and_penalty_system():
    d = design()
    assert d.weights @ d.standardized == pytest.approx(np.zeros(rr.FEATURES), abs=1e-14)
    assert d.scale[2] == 1 and np.all(d.standardized[:,2:] == 0)
    expected = sum(w*np.outer(x,x) for w,x in zip(d.weights, d.standardized))
    assert d.gram == pytest.approx(expected)
    assert rr.penalty_system(d, .1) == pytest.approx(expected+.1*np.eye(rr.FEATURES))
    assert not d.standardized.flags.writeable
    with pytest.raises(ValueError): rr.penalty_system(d, 0)


def test_design_memory_bound_before_allocation(monkeypatch):
    monkeypatch.setattr(rr, 'training_weights', lambda *_a: pytest.fail('Memory admission must come first'))
    with pytest.raises(ValueError, match='workspace'):
        rr.prepare_design(binding(), rows(), np.zeros((6,194)), maximum_rows=64, maximum_working_bytes=1)


def blocks(d):
    keys = tuple((r.donor, r.row_id) for r in d.rows)
    for start, genes in ((0, d.binding.gene_ids[:2]), (2, d.binding.gene_ids[2:])):
        source = np.full((6, len(genes)), .5)
        yield rr.GeneBlock(start, genes, keys, source, source+.25)


def mock_zero_solver(monkeypatch):
    calls = []
    def factor(system, **kwargs):
        calls.append(('factor', system.copy(), kwargs)); return 'mock-factor'
    def solve(factor, rhs, **kwargs):
        assert factor == 'mock-factor'
        calls.append(('solve', rhs.copy(), kwargs))
        assert np.all(rhs == 0)
        return np.zeros_like(rhs)
    monkeypatch.setattr(rr, 'cho_factor', factor)
    monkeypatch.setattr(rr, 'cho_solve', solve)
    return calls


def test_mocked_fit_block_wiring_and_unpenalized_intercept(monkeypatch):
    d=design(); calls=mock_zero_solver(monkeypatch)
    model=rr.fit_gene_blocks(d, blocks(d), regularization=.01, maximum_gene_block=2, maximum_working_bytes=CAP)
    assert model.intercept == pytest.approx(np.full(3,.25))
    assert np.all(model.coefficients == 0)
    assert [c[0] for c in calls] == ['factor','solve','solve']
    assert calls[0][1] == pytest.approx(d.gram+.01*np.eye(194))


def test_mocked_nonzero_solver_checks_weighted_rhs_and_gene_block_invariance(monkeypatch):
    features=np.zeros((6,194)); features[:,0]=np.arange(6)
    d=rr.prepare_design(binding(),rows(),features,maximum_rows=64,maximum_working_bytes=CAP)
    fixed=np.zeros((194,3)); fixed[0]=[.01,.02,.03]
    lam=.1
    # Prespecified generative coefficients, not coefficients estimated by any solver.
    source=np.full((6,3),1.)
    endpoint=source+.25+(1+lam)*d.standardized@fixed
    keys=tuple((r.donor,r.row_id) for r in d.rows)
    results=[]
    for width in (1,3):
        cursor=[0]
        monkeypatch.setattr(rr,'cho_factor',lambda system,**_kw: system)
        def mock_solve(system,rhs,**_kw):
            start=cursor[0]; stop=start+rhs.shape[1]; expected=fixed[:,start:stop]
            assert rhs == pytest.approx(system@expected,abs=1e-14)
            cursor[0]=stop
            return expected.copy()
        monkeypatch.setattr(rr,'cho_solve',mock_solve)
        parts=[rr.GeneBlock(start,d.binding.gene_ids[start:start+width],keys,
                           source[:,start:start+width],endpoint[:,start:start+width]) for start in range(0,3,width)]
        results.append(rr.fit_gene_blocks(d,parts,regularization=lam,maximum_gene_block=width,maximum_working_bytes=CAP))
    assert results[0].coefficients == pytest.approx(fixed)
    assert results[0].coefficients == pytest.approx(results[1].coefficients)
    assert results[0].intercept == pytest.approx(np.full(3,.25))


def test_mocked_solver_failure_has_no_retry_or_fallback(monkeypatch):
    d=design(); calls=[]
    def failed_factor(*_a,**_kw):
        calls.append(1); raise np.linalg.LinAlgError('fabricated failure')
    monkeypatch.setattr(rr,'cho_factor',failed_factor)
    with pytest.raises(np.linalg.LinAlgError,match='fabricated failure'):
        rr.fit_gene_blocks(d,blocks(d),regularization=.1,maximum_gene_block=2,maximum_working_bytes=CAP)
    assert calls==[1]


@pytest.mark.parametrize('problem', ['gene_order','row_order','missing_tail','repeat_block','oversized','bad_readout'])
def test_mocked_fit_rejects_incomplete_or_misaligned_response(monkeypatch, problem):
    d=design(); mock_zero_solver(monkeypatch); parts=list(blocks(d))
    if problem=='gene_order': parts[0]=replace(parts[0],gene_ids=parts[0].gene_ids[::-1])
    elif problem=='row_order': parts[0]=replace(parts[0],row_keys=parts[0].row_keys[::-1])
    elif problem=='missing_tail': parts.pop()
    elif problem=='repeat_block': parts.append(parts[0])
    elif problem=='oversized': parts[0]=replace(parts[0],gene_ids=d.binding.gene_ids)
    else: parts[0]=replace(parts[0],source_readout=np.full((6,2),-1.))
    with pytest.raises(ValueError):
        rr.fit_gene_blocks(d,parts,regularization=.01,maximum_gene_block=2,maximum_working_bytes=CAP)


def test_inaccurate_solver_rejected_without_fallback(monkeypatch):
    d=design(); mock_zero_solver(monkeypatch)
    monkeypatch.setattr(rr,'cho_solve',lambda _f,rhs,**_kw: np.ones_like(rhs))
    with pytest.raises(ValueError,match='residual'):
        rr.fit_gene_blocks(d,blocks(d),regularization=.1,maximum_gene_block=2,maximum_working_bytes=CAP)


def model():
    # Literal parameters, not estimated from fabricated outcomes.
    coef=np.zeros((194,3)); coef[0]=[1.,2.,3.]
    return rr.RidgeReadout(binding(),.1,np.zeros(194),np.ones(194),np.array([-4.,0.,10.]),coef)


def predict(value=None, **changes):
    kwargs=dict(query_donor='D1',condition='Stim8hr',representation_sha256='a'*64,
                start=0,gene_ids=binding().gene_ids,maximum_rows=64,maximum_working_bytes=CAP)
    kwargs.update(changes)
    return rr.predict_block(model() if value is None else value,np.zeros((2,194)),np.full((2,3),.5),**kwargs)


def test_prediction_direct_readout_does_not_clip_or_decode():
    result=predict()
    assert result == pytest.approx(np.array([[-3.5,.5,10.5]]*2))
    value=model();features=np.zeros((2,194));features[:,0]=[1.,2.]
    result=rr.predict_block(value,features,np.full((2,3),.5),query_donor='D2',condition='Stim8hr',
        representation_sha256='a'*64,start=0,gene_ids=value.binding.gene_ids,maximum_rows=64,maximum_working_bytes=CAP)
    assert result == pytest.approx(np.array([[-2.5,2.5,13.5],[-1.5,4.5,16.5]]))


@pytest.mark.parametrize('change', [dict(query_donor='D3'),dict(condition='Stim48hr'),
    dict(representation_sha256='c'*64),dict(gene_ids=('geneC','geneB','geneA')),
    dict(maximum_working_bytes=1),dict(maximum_rows=1)])
def test_prediction_rejects_wrong_information_and_resources(change):
    with pytest.raises(ValueError): predict(**change)


def test_transformed_readout_scoring_matches_existing_molecular_equations():
    comps=[np.array(a,dtype=np.float64) for a in ([.2,.3,.5],[.1,.4,.5],[.4,.2,.4],[.3,.4,.3])]
    transformed=[np.log1p(10000*q) for q in comps]
    value=rr.transformed_endpoint_scores(*transformed)
    existing=score_molecular_law(*(torch.from_numpy(q) for q in comps))
    for key in ('target_endpoint_mse','pooled_control_endpoint_mse','pooled_effect_mse','zero_effect_mse'):
        assert value[key] == pytest.approx(existing[key])
    assert value['interaction'] == pytest.approx(existing['endpoint_error_interaction'])
    et=transformed[0]-transformed[1]; ec=transformed[2]-transformed[3]
    assert value['pooled_effect_mse'] == pytest.approx(np.mean((et-ec)**2))
    assert value['pooled_effect_mse'] == pytest.approx(value['target_endpoint_mse']+value['pooled_control_endpoint_mse']+value['interaction'])
    missing=rr.transformed_endpoint_scores(transformed[0],transformed[1],None,transformed[3])
    assert missing['target_endpoint_mse']==value['target_endpoint_mse']
    assert missing['pooled_effect_mse'] is None and missing['pooled_control_endpoint_mse'] is None


def test_scoring_subtracts_in_float64_and_preserves_unconstrained_prediction():
    p=np.array([-100.,20.,.123456],dtype=np.float32)
    observed=np.array([9.,0.,.012345],dtype=np.float32)
    score=rr.transformed_endpoint_scores(p,observed,None,None)
    expected=np.mean((p.astype(np.float64)-observed.astype(np.float64))**2)
    assert score['target_endpoint_mse']==expected
    assert score['pooled_effect_mse'] is None


def evidence(selector='D1'):
    return [rr.RidgeInnerScore(s,p,d,lam,1.+lam,2.,'a'*64,'b'*64,'c'*64,'d'*64,'e'*64)
            for s,p,d in inner_edges() if s==selector for lam in rr.LAMBDA_GRID]


def test_selector_complete_inner_only_and_stronger_tie():
    result=rr.select_regularization('D1',evidence())
    assert result['regularization']==.001 and not result['outer_donor_used']
    assert result['selection_donors']==('D2','D3','D4')
    assert rr.select_regularization('D1',[replace(r,target_mse=1.) for r in evidence()])['regularization']==1.
    with pytest.raises(ValueError): replace(evidence()[0],validation_donor='D1')


@pytest.mark.parametrize('problem',['missing','duplicate','other_selector','support','representation','truth','readout'])
def test_selector_refuses_incomplete_or_unmatched_comparison(problem):
    values=evidence()
    if problem=='missing': values.pop()
    elif problem=='duplicate': values[-1]=values[0]
    elif problem=='other_selector': values=evidence('D2')
    else:
        key={'support':'support_sha256','representation':'representation_sha256',
             'truth':'truth_sha256','readout':'readout_sha256'}[problem]
        values[0]=replace(values[0],**{key:'f'*64})
    with pytest.raises(ValueError): rr.select_regularization('D1',values)


def test_protocol_is_not_launch_authority_and_full_gene_readout_remains_distinct():
    protocol=json.loads((Path(__file__).resolve().parents[1]/'RIDGE_READOUT_PROTOCOL.json').read_text())
    assert protocol['status']=='implementation_design_not_scientific_launch_admission'
    assert protocol['source_features']['count']==rr.FEATURES
    assert tuple(protocol['fitting']['regularization_grid'])==rr.LAMBDA_GRID
    assert protocol['selection']['maximum_proposed_fits']==28
    assert 'not the fold-native encoder' in protocol['purpose']
