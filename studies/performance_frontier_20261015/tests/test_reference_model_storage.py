"""Temporary fabricated artifacts; literal parameters, no fitting or GPU calls."""

from dataclasses import replace
import hashlib
import json
import os

import numpy as np
import pytest

from frontier import abundance_head as ah
from frontier import ridge_readout as rr
from frontier import simple_reference_inputs as sri
from frontier import reference_model_storage as storage
from test_simple_reference_inputs import cache, query_from, readonly, LIMITS
from test_main_worker import no_numerics
from test_ridge_readout import forbid_fit_solvers
from test_abundance_head import no_count_optimizer


CAP=2*1024**2
PRODUCER='a'*64
FIT='b'*64


@pytest.fixture(params=['ridge','abundance'])
def artifact(cache,request):
    original,genes=cache
    fitting=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    query=sri.prepare_query_inputs(query_from(original),fitting.bindings,limits=LIMITS)
    if request.param=='ridge':
        model=rr.RidgeReadout(fitting.bindings.ridge,.1,readonly(np.linspace(-.2,.2,194)),
            readonly(np.ones(194)*1.1),readonly(np.arange(5)/100),
            readonly(np.arange(194*5).reshape(194,5)/100000))
    else:
        d=ah.prepare_design(fitting.bindings.abundance,fitting.abundance_catalogs,maximum_working_bytes=LIMITS.maximum_host_bytes)
        model=ah.AbundanceHead(d.binding,readonly(d.mean),readonly(d.scale),d.target_ids,readonly(d.target_index),
            readonly(np.linspace(-.002,.002,d.parameter_count)),.1,{'literal_parameters_not_fitted':True})
    return request.param,model,fitting.bindings,query


def publish(artifact,root,**changes):
    kind,model,bindings,_=artifact
    options=dict(producer_sha256=PRODUCER,fitting_receipt_sha256=FIT,output_cap_bytes=CAP)
    options.update(changes)
    return storage.publish_reference(model,bindings,root,**options)


def load(artifact,root,receipt,**changes):
    kind,_,bindings,_=artifact
    options=dict(kind=kind,expected_complete_sha256=receipt['complete_sha256'],producer_sha256=PRODUCER,
        fitting_receipt_sha256=FIT,output_cap_bytes=CAP,maximum_host_bytes=64*1024**2)
    options.update(changes)
    return storage.load_reference(root,bindings,**options)


def prediction(model,kind,query):
    if kind=='abundance':
        return ah.predict_catalog(model,query.abundance_source,maximum_working_bytes=LIMITS.maximum_host_bytes)
    return np.concatenate([rr.predict_block(model,query.ridge_features,early,
        query_donor='D1',condition='Stim8hr',representation_sha256=model.binding.representation_sha256,
        start=start,gene_ids=genes,maximum_rows=64,maximum_working_bytes=LIMITS.maximum_host_bytes)
        for start,genes,early in query.early_gene_blocks()],axis=1)


def rewrite_manifest(root,receipt,change):
    """Malformed tiny fixture with recomputed hashes; not production publication."""
    manifest=json.loads((root/'MANIFEST.json').read_bytes())
    change(manifest)
    raw=storage._json_bytes(manifest)
    (root/'MANIFEST.json').write_bytes(raw)
    complete=json.loads((root/'COMPLETE.json').read_bytes())
    complete.update(manifest_sha256=hashlib.sha256(raw).hexdigest(),manifest_bytes=len(raw))
    array_bytes=sum(p.stat().st_size for p in root.iterdir() if p.suffix=='.npy')
    while True:
        encoded=storage._json_bytes(complete)
        total=array_bytes+len(raw)+len(encoded)
        if complete['total_bytes']==total:break
        complete['total_bytes']=total
    (root/'COMPLETE.json').write_bytes(encoded)
    return dict(receipt,complete_sha256=hashlib.sha256(encoded).hexdigest())


def test_roundtrip_preserves_prediction_binding_and_private_readonly_arrays(artifact,tmp_path):
    root=tmp_path/'model';receipt=publish(artifact,root);loaded=load(artifact,root,receipt)
    kind,original,bindings,query=artifact
    assert loaded.binding==original.binding
    assert receipt['total_bytes']==sum(p.stat().st_size for p in root.iterdir())
    assert len(list(root.iterdir()))==6
    assert receipt['fitting_completion_verified'] is False
    assert receipt['scientific_launch_authority'] is False
    assert receipt['inference_only'] is True
    assert not (root/'.COMPLETE.pending').exists()
    assert np.array_equal(prediction(loaded,kind,query),prediction(original,kind,query))
    names=('mean','scale','intercept','coefficients') if kind=='ridge' else ('mean','scale','target_index','parameters')
    for name in names:
        before,after=getattr(original,name),getattr(loaded,name)
        assert np.array_equal(before,after)
        assert after.flags.owndata and not after.flags.writeable
        assert not np.shares_memory(before,after)
    # The returned model owns its bytes rather than borrowing a live mmap.
    path=root/'p000.mean.npy';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
    assert np.array_equal(loaded.mean,original.mean)
    with pytest.raises(ValueError):load(artifact,root,receipt)


def test_existing_artifact_is_never_overwritten(artifact,tmp_path):
    root=tmp_path/'model';receipt=publish(artifact,root)
    with pytest.raises(FileExistsError):publish(artifact,root)
    assert hashlib.sha256((root/'COMPLETE.json').read_bytes()).hexdigest()==receipt['complete_sha256']


def test_resource_limits_fail_without_creating_output_or_reading_parameters(artifact,tmp_path,monkeypatch):
    root=tmp_path/'model'
    with pytest.raises(ValueError,match='before writes'):publish(artifact,root,output_cap_bytes=1)
    assert not root.exists()
    receipt=publish(artifact,root)
    real=storage._open_verified;opened=[]
    def check(fd,name,*a,**kw):
        opened.append(name)
        assert not name.endswith('.npy')
        return real(fd,name,*a,**kw)
    monkeypatch.setattr(storage,'_open_verified',check)
    with pytest.raises(ValueError,match='host bound'):load(artifact,root,receipt,maximum_host_bytes=1)
    assert opened==['COMPLETE.json']


@pytest.mark.parametrize('field',['producer_sha256','fitting_receipt_sha256','expected_complete_sha256'])
def test_exact_external_hashes_required(artifact,tmp_path,field):
    root=tmp_path/'model';receipt=publish(artifact,root)
    with pytest.raises(ValueError):load(artifact,root,receipt,**{field:'e'*64})


def test_other_fold_observation_state_cannot_be_substituted(artifact,tmp_path):
    root=tmp_path/'model';receipt=publish(artifact,root)
    kind,model,binding,query=artifact
    changed=replace(binding,observation_binding=replace(binding.observation_binding,state_sha256='e'*64))
    with pytest.raises(ValueError,match='bindings'):
        load((kind,model,changed,query),root,receipt)


@pytest.mark.parametrize('problem',['dtype','shape','path','boolean_offset','fortran'])
def test_rehashed_malformed_layout_is_rejected_before_array_read(artifact,tmp_path,monkeypatch,problem):
    root=tmp_path/'model';receipt=publish(artifact,root)
    def change(m):
        r=m['arrays'][0]
        if problem=='dtype':r['dtype']='|O'
        elif problem=='shape':r['shape']=[1,194]
        elif problem=='path':r['relative_path']='../outside.npy'
        elif problem=='boolean_offset':r['population_index']=False
        else:r['fortran_order']=0
    receipt=rewrite_manifest(root,receipt,change)
    real=storage._open_verified
    def no_arrays(fd,name,*a,**kw):
        assert not name.endswith('.npy')
        return real(fd,name,*a,**kw)
    monkeypatch.setattr(storage,'_open_verified',no_arrays)
    with pytest.raises(ValueError,match='layout'):load(artifact,root,receipt)


@pytest.mark.parametrize('problem',['symlink','hardlink','extra_member'])
def test_noncanonical_file_inventory_is_rejected(artifact,tmp_path,problem):
    root=tmp_path/'model';receipt=publish(artifact,root);path=root/'p000.mean.npy'
    if problem=='symlink':
        other=tmp_path/'outside.npy';path.rename(other);path.symlink_to(other)
    elif problem=='hardlink':os.link(path,tmp_path/'alias.npy')
    else:(root/'EXTRA.json').write_text('{}')
    with pytest.raises(ValueError):load(artifact,root,receipt)


def test_partial_publication_remains_incomplete_and_cannot_retry(artifact,tmp_path,monkeypatch):
    root=tmp_path/'model';real=storage._write_once;calls=[]
    def fail(fd,name,*a,**kw):
        calls.append(name)
        if len(calls)==2:raise OSError('fabricated write failure')
        return real(fd,name,*a,**kw)
    monkeypatch.setattr(storage,'_write_once',fail)
    with pytest.raises(OSError,match='fabricated'):publish(artifact,root)
    assert set(p.name for p in root.iterdir())=={'p000.mean.npy'}
    assert not (root/'COMPLETE.json').exists()
    with pytest.raises(FileExistsError):publish(artifact,root)


def test_parameter_change_during_publication_prevents_completion(artifact,tmp_path,monkeypatch):
    root=tmp_path/'model';real=storage._write_once;model=artifact[1]
    def change(fd,name,*a,**kw):
        result=real(fd,name,*a,**kw)
        if name=='MANIFEST.json':
            model.mean.setflags(write=True);model.mean[0]+=.1;model.mean.setflags(write=False)
        return result
    monkeypatch.setattr(storage,'_write_once',change)
    with pytest.raises(ValueError,match='model changed'):publish(artifact,root)
    assert not (root/'COMPLETE.json').exists()


def test_late_write_during_final_load_hash_is_detected(artifact,tmp_path,monkeypatch):
    root=tmp_path/'model';receipt=publish(artifact,root);real=storage._digest_stream
    def change(stream):
        result=real(stream)
        # The helper's open-time hashes use its own module; this function is
        # reached only by the loader's final rehash pass.
        path=root/'p000.mean.npy';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        return result
    monkeypatch.setattr(storage,'_digest_stream',change)
    with pytest.raises(ValueError,match='changed'):load(artifact,root,receipt)


def test_abundance_target_route_and_finite_readonly_parameters_are_required(cache,tmp_path):
    original,genes=cache;inputs=sri.prepare_fitting_inputs(original,genes,limits=LIMITS)
    d=ah.prepare_design(inputs.bindings.abundance,inputs.abundance_catalogs,maximum_working_bytes=LIMITS.maximum_host_bytes)
    model=ah.AbundanceHead(d.binding,readonly(d.mean),readonly(d.scale),d.target_ids,readonly(d.target_index),
        readonly(np.zeros(d.parameter_count)),.1,{'literal_parameters_not_fitted':True})
    for i,changed in enumerate((replace(model,target_index=readonly(np.zeros_like(d.target_index))),
                               replace(model,parameters=np.zeros(d.parameter_count)),
                               replace(model,parameters=readonly(np.full(d.parameter_count,np.nan))),
                               replace(model,regularization=True))):
        with pytest.raises(ValueError):
            publish(('abundance',changed,inputs.bindings,None),tmp_path/f'bad-{i}')
        assert not (tmp_path/f'bad-{i}').exists()
