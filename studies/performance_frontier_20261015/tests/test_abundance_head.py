"""Fabricated arithmetic/gradient checks; optimizer execution is forbidden."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize
import torch

from frontier import abundance_head as ah
from frontier.fitting_provider import CatalogGuide
from frontier.measure_ops import source_log_frequencies
from frontier.evaluation import score_catalog
from frontier.planner import inner_edges
from test_main_worker import no_numerics


CAP=8*1024**2


@pytest.fixture(autouse=True)
def no_count_optimizer(monkeypatch,request):
    def forbid(*_a,**_kw): pytest.fail('No abundance optimizer is admitted by this test suite')
    monkeypatch.setattr(ah,'minimize',forbid)
    monkeypatch.setattr(scipy.optimize,'minimize',forbid)
    request.node.user_properties.append(('abundance_optimizer_iterations',0))


def binding():
    catalog=tuple(CatalogGuide('g'+str(i),target,control) for i,(target,control) in enumerate(
        [('T1',False),('T1',False),('T2',False),('NTC',True),('T3',False),('T4',False)]))
    return ah.HeadBinding(('D3','D4'),catalog,'a'*64,'b'*64,True)


def source(donor='D3'):
    b=binding()
    counts=np.array([10,0,4,5,0,2] if donor=='D3' else [2,3,0,1,0,1],dtype=np.int64)
    positive=np.array([8,0,2,4,0,0] if donor=='D3' else [2,3,0,1,0,0],dtype=np.int64)
    m=np.zeros((6,96));m[positive>0,:48]=np.arange(48)/100;m[positive>0,48:]=.05
    return ah.SourceCatalog(donor,'Stim8hr',b.catalog_sha256,b.representation_sha256,'c'*64,counts,positive,m)


def catalogs():
    return (ah.FittingCatalog(source(),np.array([6,0,2,5,4,0],dtype=np.int64),'d'*64),
            ah.FittingCatalog(source('D4'),np.array([0,4,3,1,0,2],dtype=np.int64),'e'*64))


def design(): return ah.prepare_design(binding(),catalogs(),maximum_working_bytes=CAP)


def test_full_catalog_source_reference_and_missing_state_indicator():
    s=source();x,logs=ah.source_features(binding(),s,maximum_working_bytes=CAP)
    expected=source_log_frequencies(torch.from_numpy(s.captured_counts)).numpy()
    assert logs==pytest.approx(expected)
    assert x.shape==(6,100)
    assert x[:,-1].tolist()==[1,0,1,1,0,0]
    assert np.all(x[5,:96]==0) and s.captured_counts[5]>0
    assert np.all(np.exp(logs)>0)
    assert x[:,97]==pytest.approx(np.log1p(s.captured_counts))


@pytest.mark.parametrize('change',[dict(fitting_donors=('D4','D3')),dict(fitting_donors=('D3','D3')),
    dict(synthetic_fixture=False),dict(representation_sha256='bad')])
def test_binding_rejects_wrong_scientific_scope(change):
    with pytest.raises(ValueError): replace(binding(),**change)


@pytest.mark.parametrize('problem',['late','catalog','representation','negative','no_capture','support','fabricated_state'])
def test_source_rejects_invalid_or_unavailable_inputs(problem):
    s=source()
    if problem=='late':s=replace(s,condition='Stim48hr')
    elif problem=='catalog':s=replace(s,catalog_sha256='f'*64)
    elif problem=='representation':s=replace(s,representation_sha256='f'*64)
    elif problem=='negative':s.captured_counts[0]=-1
    elif problem=='no_capture':s.captured_counts[:]=0
    elif problem=='support':s.positive_rna_counts[0]=11
    else:s.moments[1,0]=.1
    with pytest.raises(ValueError):ah.source_features(binding(),s,maximum_working_bytes=CAP)


def test_full_catalog_design_has_no_endpoint_selected_target_membership():
    d=design()
    assert d.target_ids==('T1','T2','T4')
    assert d.target_index.tolist()==[0,0,1,-1,-1,2]
    assert d.penalty[100:]==pytest.approx([1+1/15,1+1/4,1+1/3])
    assert sum(x.mean(0) for x in d.standardized_features)/2==pytest.approx(np.zeros(100),abs=1e-14)
    changed=tuple(replace(c,endpoint_counts=c.endpoint_counts[::-1].copy()) for c in catalogs())
    other=ah.prepare_design(binding(),changed,maximum_working_bytes=CAP)
    assert other.target_ids==d.target_ids
    assert other.mean==pytest.approx(d.mean) and other.scale==pytest.approx(d.scale)
    with pytest.raises(ValueError):ah.prepare_design(binding(),catalogs()[::-1],maximum_working_bytes=CAP)
    changed=(replace(catalogs()[0],source=source('D1')),catalogs()[1])
    with pytest.raises(ValueError):ah.prepare_design(binding(),changed,maximum_working_bytes=CAP)


def test_memory_gate_precedes_feature_allocation(monkeypatch):
    monkeypatch.setattr(ah,'source_features',lambda *_a,**_kw:pytest.fail('Too early'))
    with pytest.raises(ValueError,match='workspace'):
        ah.prepare_design(binding(),catalogs(),maximum_working_bytes=1)


def test_zero_parameters_are_exact_persistence_with_unsmoothed_truth():
    d=design();theta=np.zeros(d.parameter_count)
    loss,gradient=ah.objective_and_gradient(theta,d,.01)
    expected=[]
    for x,logs,c in zip(d.standardized_features,d.source_log_frequencies,catalogs()):
        predicted,_=ah.catalog_prediction(theta,x,logs,d.target_index)
        assert predicted==pytest.approx(logs)
        score=score_catalog(torch.from_numpy(predicted),torch.from_numpy(c.endpoint_counts),expected_catalog_size=6)
        expected.append(score['cross_entropy_nats'])
        assert score['endpoint_zero_categories']>0 and score['catalog_size']==6
    assert loss==pytest.approx(np.mean(expected))
    assert np.isfinite(gradient).all()


def test_full_current_catalog_gradient_matches_independent_autograd():
    d=design();theta=np.linspace(-.03,.03,d.parameter_count)
    loss,gradient=ah.objective_and_gradient(theta,d,.1)
    t=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    terms=[]
    route=d.target_index;active=np.flatnonzero(route>=0)
    for x,logs,p in zip(d.standardized_features,d.source_log_frequencies,d.observed_frequencies):
        delta=torch.tensor(x)@t[:100]
        extra=torch.zeros(6,dtype=torch.float64).index_add(0,torch.tensor(active),t[100+torch.tensor(route[active])])
        logq=torch.log_softmax(torch.tensor(logs)+delta+extra,0)
        terms.append(-(torch.tensor(p)*logq).sum())
    expected=torch.stack(terms).mean()+.05*(torch.tensor(d.penalty)*t*t).sum()
    g,=torch.autograd.grad(expected,t)
    assert loss==pytest.approx(expected.item(),abs=1e-12)
    assert gradient==pytest.approx(g.detach().numpy(),abs=1e-12)
    # This includes the RNA-unsupported category and the late-positive source-zero target.
    assert d.observed_frequencies[0][4]>0 and d.target_index[4]==-1


def test_finite_difference_gradients_and_equal_donor_not_pooled_count_weighting():
    d=design();theta=np.linspace(.01,.02,d.parameter_count)
    loss,g=ah.objective_and_gradient(theta,d,.01)
    for j in (0,48,96,97,98,99,100,101,102):
        h=np.zeros(d.parameter_count);h[j]=1e-6
        finite=(ah.objective_and_gradient(theta+h,d,.01)[0]-ah.objective_and_gradient(theta-h,d,.01)[0])/2e-6
        assert g[j]==pytest.approx(finite,abs=5e-9)
    cc=catalogs();scaled=(replace(cc[0],endpoint_counts=cc[0].endpoint_counts*17),cc[1])
    other=ah.prepare_design(binding(),scaled,maximum_working_bytes=CAP)
    value,other_g=ah.objective_and_gradient(theta,other,.01)
    assert value==pytest.approx(loss) and other_g==pytest.approx(g)


def test_current_inactive_category_affects_normalizer_and_gradient():
    d=design();base=np.zeros(d.parameter_count);changed=base.copy();changed[-1]=.7
    q0=np.exp(ah.catalog_prediction(base,d.standardized_features[0],d.source_log_frequencies[0],d.target_index)[0])
    q1=np.exp(ah.catalog_prediction(changed,d.standardized_features[0],d.source_log_frequencies[0],d.target_index)[0])
    assert q1[5]>q0[5] and np.all(q1[:5]<q0[:5])
    assert not np.array_equal(ah.objective_and_gradient(base,d,.01)[1],ah.objective_and_gradient(changed,d,.01)[1])


def test_underflow_fails_instead_of_becoming_an_observed_zero():
    d=design();theta=np.zeros(d.parameter_count);theta[100]=2000
    with pytest.raises(ValueError,match='Underflowed'):
        ah.objective_and_gradient(theta,d,.1)


def mock_optimizer(monkeypatch, *, iterations=0, success=True,status=0):
    observed=[]
    def mock(fun,x0,**kwargs):
        observed.append(kwargs)
        value=None
        for _ in range(iterations+1):value,_=fun(x0)
        return SimpleNamespace(x=x0,fun=value,nit=iterations,nfev=iterations+1,
                               success=success,status=status,message='fabricated terminal')
    monkeypatch.setattr(ah,'minimize',mock)
    return observed


def test_mocked_optimizer_contract_and_terminal_reconciliation(monkeypatch):
    d=design();calls=mock_optimizer(monkeypatch)
    m=ah.fit_head(d,.01,maximum_working_bytes=CAP)
    assert calls[0]['method']=='L-BFGS-B' and calls[0]['jac'] is True
    assert calls[0]['options']['maxiter']==500
    assert m.diagnostics['function_gradient_evaluations']==2 and m.diagnostics['converged']
    assert np.all(m.parameters==0)


def test_iteration_cap_is_not_convergence_and_failure_is_not_retried(monkeypatch):
    d=design();monkeypatch.setattr(ah,'MAX_ITERATIONS',2)
    calls=mock_optimizer(monkeypatch,iterations=2,success=False,status=1)
    model=ah.fit_head(d,.1,maximum_working_bytes=CAP)
    assert not model.diagnostics['converged'] and model.diagnostics['terminal_reason']=='iteration_cap'
    assert len(calls)==1
    calls=mock_optimizer(monkeypatch,success=False,status=2)
    with pytest.raises(ValueError,match='failed'):ah.fit_head(d,.1,maximum_working_bytes=CAP)
    assert len(calls)==1


def test_function_cap_and_memory_gate_do_not_expand_work(monkeypatch):
    d=design();monkeypatch.setattr(ah,'MAX_EVALUATIONS',2)
    calls=mock_optimizer(monkeypatch,iterations=2)
    with pytest.raises(ValueError,match='cap exhausted'):ah.fit_head(d,.1,maximum_working_bytes=CAP)
    assert len(calls)==1
    with pytest.raises(ValueError,match='workspace'):ah.fit_head(d,.1,maximum_working_bytes=1)
    assert len(calls)==1


def literal_model():
    d=design();p=np.zeros(d.parameter_count);p[100:]=[.1,.2,.3]
    return ah.AbundanceHead(d.binding,d.mean,d.scale,d.target_ids,d.target_index,p,.1,{'fabricated':True})


def test_query_source_only_and_control_unseen_intercepts_are_zero():
    m=literal_model();s=source('D1')
    q=ah.predict_catalog(m,s,maximum_working_bytes=CAP)
    _,base=ah.source_features(binding(),s,maximum_working_bytes=CAP)
    raw=base+np.array([.1,.1,.2,0,0,.3])
    expected=raw-np.log(np.exp(raw).sum())
    assert q==pytest.approx(expected)
    assert len(q)==6 and np.exp(q).sum()==pytest.approx(1)
    for bad in (source('D3'),replace(s,condition='Stim48hr'),replace(s,representation_sha256='f'*64)):
        with pytest.raises(ValueError):ah.predict_catalog(m,bad,maximum_working_bytes=CAP)
    bad=replace(m,target_index=np.zeros(6,dtype=np.int64))
    with pytest.raises(ValueError):ah.predict_catalog(bad,s,maximum_working_bytes=CAP)


def evidence(selector='D1'):
    return [ah.InnerCountScore(s,p,d,lam,2.+lam,*[c*64 for c in 'abcdef'])
            for s,p,d in inner_edges() if s==selector for lam in ah.REGULARIZATION_GRID]


def test_inner_penalty_selection_complete_and_donor_separated():
    result=ah.select_regularization('D1',evidence())
    assert result['regularization']==.001 and not result['outer_donor_used']
    assert ah.select_regularization('D1',[replace(r,cross_entropy=2.) for r in evidence()])['regularization']==1.
    with pytest.raises(ValueError):replace(evidence()[0],validation_donor='D1')


@pytest.mark.parametrize('problem',['missing','duplicate','outer','catalog','source','fitting_input','truth','representation','scoring'])
def test_selector_refuses_wrong_support_or_provenance(problem):
    rows=evidence()
    if problem=='missing':rows.pop()
    elif problem=='duplicate':rows[-1]=rows[0]
    elif problem=='outer':rows=evidence('D2')
    else:rows[0]=replace(rows[0],**{problem+'_sha256':'f'*64 if problem!='scoring' else 'a'*64})
    with pytest.raises(ValueError):ah.select_regularization('D1',rows)


def test_protocol_preserves_28_fit_and_500_iteration_caps():
    path=Path(__file__).resolve().parents[1]/'ABUNDANCE_HEAD_PROTOCOL.json'
    def unique_pairs(pairs):
        result={}
        for k,v in pairs:
            assert k not in result
            result[k]=v
        return result
    data=json.loads(path.read_text(),object_pairs_hook=unique_pairs)
    assert data['status']=='implementation_design_not_scientific_launch_admission'
    assert data['nested_selection']['maximum_fits']==28
    assert tuple(data['model']['regularization_grid'])==ah.REGULARIZATION_GRID
    assert ah.MAX_ITERATIONS==500 and ah.MAX_EVALUATIONS==12501
