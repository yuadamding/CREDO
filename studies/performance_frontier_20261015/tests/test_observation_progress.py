"""Fabricated metadata only: no remote calls, checkpoints, arrays or optimizer."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'operations/observation_progress.py'
SPEC = importlib.util.spec_from_file_location('observation_progress_test', SOURCE)
API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API)


def record(value):
    raw = json.dumps(value, sort_keys=True).encode()
    return dict(value=value, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def metadata():
    output, bundle = '/fixture/output', '/fixture/bundle'
    source = record(dict(schema='credo.frontier.observation-worker-source.v1', files={}))
    worker = record(dict(schema='credo.frontier.observation-worker-contract.v1', worker_id='fixture-v4',
        mode='fit', output_root=output, source_manifest=bundle+'/SOURCE_MANIFEST.json',
        source_manifest_sha256=source['sha256'], trajectory_id='trajectory-v4',
        fold_id='subset-D1-D2', input_access_sha256='a'*64))
    reference = lambda name, value: dict(path=bundle+'/'+name,
        sha256=value['sha256'], size_bytes=value['size_bytes'])
    plan = dict(run_id='fixture-v4', mode='fit', scientific_output=output, bundle_root=bundle,
        maximum_new_scientific_updates=18720, worker_contract=reference('WORKER_CONTRACT.json',worker),
        source_manifest=reference('SOURCE_MANIFEST.json',source),
        recovery_contract=dict(path=bundle+'/CONTINUATION_CONTRACT.json',sha256='b'*64,size_bytes=123))
    identity = dict(trajectory_id='trajectory-v4',fold_id='subset-D1-D2',seed=0,
        input_sha256='a'*64,source_sha256=source['sha256'],worker_contract_sha256=worker['sha256'],
        configuration_sha256='c'*64,recovery_parent_checkpoint_sha256='d'*64)
    files = {bundle+'/WORKER_CONTRACT.json':worker,bundle+'/SOURCE_MANIFEST.json':source,
             output+'/IDENTITY.json':record(identity)}
    def progress(updates):
        path = output+'/progress/update-%09d.json' % updates
        files[path] = record(dict(schema='credo.frontier.observation-worker-progress.v1',identity=identity,
            accepted_updates=updates,attempted_updates=updates,actual_optimizer_calls=updates,
            consumed_parent_updates=1280,new_scientific_updates_this_owner=updates-1280,
            source_continuation_contract=plan['recovery_contract'],at_utc='2026-10-16T19:00:00Z'))
        return path
    calls = []
    def read(path,maximum):
        calls.append(path)
        assert maximum == API.MAXIMUM_JSON_BYTES
        assert path in files or '/progress/update-' in path
        return files.get(path)
    return plan,files,progress,read,calls


def test_latest_milestone_descends_exact_paths_and_never_opens_payload(metadata):
    plan,files,publish,read,calls = metadata
    publish(1536)
    latest = publish(2048)
    result = API.observation_progress(plan,read)
    assert result['accepted_updates'] == 2048
    assert result['new_scientific_updates_this_owner'] == 768
    assert result['remaining_updates_to_target'] == 17952
    assert result['terminal_acceptance'] is False
    expected = ['/fixture/output/progress/update-%09d.json' % count
                for count in API.MILESTONES if count >= 2048]
    assert calls[3:] == expected
    assert calls[-1] == latest


def test_terminal_milestone_is_not_terminal_acceptance(metadata):
    plan,files,publish,read,calls = metadata
    publish(20000)
    result = API.observation_progress(plan,read)
    assert result['accepted_updates'] == 20000
    assert result['new_scientific_updates_this_owner'] == 18720
    assert result['terminal_acceptance'] is False
    assert len(calls) == 4


@pytest.mark.parametrize('missing', ['identity','progress'])
def test_missing_evidence_is_unavailable_not_zero(metadata,missing):
    plan,files,publish,read,calls = metadata
    if missing == 'identity':
        files.pop('/fixture/output/IDENTITY.json')
        def read(path,maximum):
            return files.get(path)
    result = API.observation_progress(plan,read)
    assert result['status'] == 'unavailable'
    assert result['accepted_updates'] is None
    assert result['new_scientific_updates_this_owner'] is None


@pytest.mark.parametrize('key,value', [
    ('schema','wrong'),('identity',{}),('consumed_parent_updates',0),
    ('accepted_updates',1537),('attempted_updates',1537),('actual_optimizer_calls',1537),
    ('new_scientific_updates_this_owner',0),('source_continuation_contract',{}),
    ('accepted_updates',1536.0)])
def test_invalid_latest_never_falls_back(metadata,key,value):
    plan,files,publish,read,calls = metadata
    path = publish(1536)
    files[path]['value'][key] = value
    with pytest.raises(ValueError):
        API.observation_progress(plan,read)


@pytest.mark.parametrize('name', ['WORKER_CONTRACT.json','SOURCE_MANIFEST.json'])
def test_bound_metadata_hash_mismatch(metadata,name):
    plan,files,publish,read,calls = metadata
    files['/fixture/bundle/'+name]['sha256'] = 'e'*64
    with pytest.raises(ValueError,match='bytes differ'):
        API.observation_progress(plan,read)


def test_worker_and_output_identity_mismatch(metadata):
    plan,files,publish,read,calls = metadata
    files['/fixture/output/IDENTITY.json']['value']['trajectory_id'] = 'other'
    with pytest.raises(ValueError,match='Output identity'):
        API.observation_progress(plan,read)


def test_bounded_owned_nonsymlink_reader(tmp_path):
    path = tmp_path/'metadata.json'
    path.write_text('{"ok":true}')
    assert API.read_progress_json(str(path),owner_uid=os.getuid())['value'] == {'ok':True}
    assert API.read_progress_json(str(tmp_path/'missing'),owner_uid=os.getuid()) is None
    with pytest.raises(ValueError,match='owner'):
        API.read_progress_json(str(path),owner_uid=os.getuid()+1)
    with pytest.raises(ValueError,match='size'):
        API.read_progress_json(str(path),maximum=1,owner_uid=os.getuid())
    link = tmp_path/'link.json'
    link.symlink_to(path)
    with pytest.raises(ValueError,match='type'):
        API.read_progress_json(str(link),owner_uid=os.getuid())
    with pytest.raises(ValueError,match='type'):
        API.read_progress_json(str(tmp_path),owner_uid=os.getuid())


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}','{"a":NaN}','[]','bad'])
def test_malformed_json_rejected(tmp_path,raw):
    path = tmp_path/'metadata.json'
    path.write_text(raw)
    with pytest.raises(ValueError):
        API.read_progress_json(str(path),owner_uid=os.getuid())


def test_remote_python36_source_and_execution(metadata):
    source = SOURCE.read_text()
    ast.parse(source,feature_version=(3,6))
    namespace = {}
    exec(compile(source,'fabricated-remote-helper','exec'),namespace)
    plan,files,publish,read,calls = metadata
    publish(1536)
    assert namespace['observation_progress'](plan,read)['accepted_updates'] == 1536
