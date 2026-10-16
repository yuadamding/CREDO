"""Explicit v4 replacement packaging; no journal mutation or remote operation.

Reuse the corrected continuation builder in memory, then bind a fresh identity.
Frozen v3 packages and all numerical worker sources remain unchanged. Controller
receipts are exact worker-visible copies with separately preserved origin refs.
"""
import json
from pathlib import Path

from . import continuation_build as prior_builder
from .common import PROJECT, CONTROL, GIB, encoded, identity, read, validate_plan, validate_admission
from .build import archive_bytes, manifest_from_template


PAIRS = prior_builder.PAIRS
AUTHORITY = 'OBSERVATION_CONTINUATION_V4_REPLACEMENT_AUTHORITY.json'
INTAKE = 'receipts/OBSERVATION_CONTINUATION_V4_REPLACEMENT_INTAKE_20261016.json'


def ref(path, raw):
    return dict(path=str(path), **identity(raw))


def _inside(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def _rebase(value, old, new):
    if isinstance(value, dict):
        return {key:_rebase(item, old, new) for key,item in value.items()}
    if isinstance(value, list):
        return [_rebase(item, old, new) for item in value]
    if isinstance(value, str) and (value == str(old) or value.startswith(str(old)+'/')):
        return str(new/Path(value).relative_to(old))
    return value


def validate_replacement_authority(authority, admission, pair):
    expected = dict(schema='credo.frontier.observation-continuation-replacement-authority.v1',status='approved',
        pairs=list(PAIRS),failed_generation='v3',replacement_generation='v4',per_pair_explicit_replacement_attempts=1,
        source_continuation_authority='OBSERVATION_CONTINUATION_V3_AUTHORITY.json',consumed_updates_each=1280,
        maximum_new_updates_each=18720,maximum_total_updates_each=20000,consumed_updates_all_four=5120,
        maximum_new_updates_all_four=74880,required_failed_child_new_updates=0,
        required_failed_child_scientific_output='absent',combined_output_limit_bytes_each=4*GIB,
        maximum_concurrent_h100=4,resources_each=dict(H100=1,CPUs=24,host_memory_GiB=200,application_threads=8),
        automatic_retries=0,automatic_successor=False,elapsed_time_deadline_seconds=None,
        original_workload_expanded=False,main_model_fitting=False,historical_parents_and_failed_children_read_only=True,
        numerical_worker_sources_changed=False,local_optimizer_calls_authorized=0,LARRY_access=False)
    if any(type(authority.get(key)) is not type(value) or authority.get(key)!=value for key,value in expected.items()):
        raise ValueError('Exact bounded v4 replacement authority required')
    if admission.get('replaces_continuation_task_id') != 'credo-frontier-obs-'+pair.lower()+'-20261016-v3':
        raise ValueError('Exact failed v3 continuation predecessor required')
    for name in ('replacement_authority_sha256','zero_update_evidence_sha256','replacement_intake_sha256'):
        value=admission.get(name)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Exact replacement evidence binding required: '+name)


def worker_dependency_closure(root, payload, plan):
    """Enumerate worker-read study artifacts without opening biological payloads.

    Native input payloads remain verified by their existing consumption leases;
    this manifest must not cause an eager raw-data or checkpoint hash replay.
    Controller-only provenance paths are explicitly not runtime read edges.
    """
    root, bundle = Path(root), Path(plan['bundle_root'])
    worker=json.loads(payload['WORKER_CONTRACT.json'])
    recovery=json.loads(payload['CONTINUATION_CONTRACT.json'])
    original_path=root/('observation_'+recovery['pair'].lower()+'_v2_launch')/'bundle'
    original=json.loads(read(original_path/'WORKER_CONTRACT.json'))
    old_project=Path(original['study_root'])
    # Disposable tests may relocate only the new prepared project. Production
    # admits exactly this study and its original bound native-input root.
    visible_roots=tuple(dict.fromkeys((Path(PROJECT),old_project,Path(original['dataset_root']))))
    metadata, tensors, inputs = {}, {}, {}
    roots, locks, absent = {}, {}, {}

    def visible(path, label):
        path=Path(path)
        if not path.is_absolute() or '..' in path.parts or not any(_inside(path,allowed) for allowed in visible_roots):
            raise ValueError('Worker dependency outside visible home surfaces: '+label)
        return str(path)

    def add(table, label, reference):
        item={key:reference[key] for key in ('path','sha256','size_bytes') if key in reference}
        if set(item) not in ({'path','sha256'}, {'path','sha256','size_bytes'}):
            raise ValueError('Bound worker dependency requires path/hash: '+label)
        visible(item['path'],label)
        if not isinstance(item['sha256'],str) or len(item['sha256'])!=64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise ValueError('Worker dependency hash differs: '+label)
        if 'size_bytes' in item and (type(item['size_bytes']) is not int or item['size_bytes']<0):
            raise ValueError('Worker dependency size differs: '+label)
        if _inside(item['path'],bundle):
            name=Path(item['path']).relative_to(bundle).as_posix()
            if name not in payload or identity(payload[name])['sha256']!=item['sha256']:
                raise ValueError('Worker dependency absent or unbound in new bundle: '+label)
            if 'size_bytes' in item and len(payload[name])!=item['size_bytes']:
                raise ValueError('Worker dependency payload size differs: '+label)
        table[label]=item

    for key in ('worker_contract','source_manifest','protocol','phase_authority','parent_authority','admission',
                'recovery_contract','replacement_authority','replacement_intake','recovery_metadata_origins'):
        add(metadata,'plan.'+key,plan[key])
    for name,sha in plan['controller_sources'].items():
        add(metadata,'controller_source.'+name,dict(path=str(bundle/name),sha256=sha,size_bytes=len(payload[name])))
    for field,value in recovery.items():
        if isinstance(value,dict) and 'path' in value:
            add(metadata,'recovery.'+field,value)
    source=json.loads(payload['SOURCE_MANIFEST.json'])
    for name,sha in {**source['files'],**recovery['extra_sources']}.items():
        add(metadata,'worker_source.'+name,dict(path=str(Path(worker['source_root'])/name),sha256=sha,
                                               size_bytes=len(payload['source/'+name])))
    for name,raw in payload.items():
        if name.startswith('retained_reader/'):
            add(metadata,'reader_source.'+name,ref(bundle/name,raw))
    for field in ('study_root','source_root','reader_source_root','metadata_root','dataset_root','output_root'):
        roots['worker.'+field]=visible(worker[field],'worker.'+field)
    add(metadata,'original.qualification',dict(path=original['qualification_receipt'],sha256=original['qualification_sha256']))
    original_sources=json.loads(read(original_path/'SOURCE_MANIFEST.json'))
    name='frontier/observation_worker.py'
    add(metadata,'original.worker_source',dict(path=str(Path(original['source_root'])/name),
        sha256=original_sources['files'][name],size_bytes=len(read(original_path/'source'/name))))
    check_worker=json.loads(read(root/'observation_retention_check_v2_launch/bundle/WORKER_CONTRACT.json'))
    add(metadata,'checked_worker.source_manifest',check_worker['source_manifest'])
    parent=json.loads(payload['PARENT_INTAKE.json'])['fits'][recovery['pair']]
    for key in ('identity','failure','checkpoint_completion','incomplete_retirement'):
        add(metadata,'parent.'+key,parent[key])
    add(tensors,'parent.checkpoint_payload',parent['checkpoint_payload'])
    for index,value in enumerate(parent['candidate_payloads']):
        add(tensors,'parent.candidate.'+str(index),value)
    parent_root=Path(original['output_root'])
    roots['parent.output_root']=visible(parent_root,'parent.output_root')
    locks['parent.owner']=visible(parent_root/'OWNER.lock','parent.owner')
    absent['parent.retired_payload']=visible(parent_root/'checkpoints/update-000000768/checkpoint.pt','parent.retired_payload')
    absent['parent.retirement_publication']=visible(parent_root/'retention/transactions/after-update-000001280/DELETED.json',
                                                   'parent.retirement_publication')
    parent_inventory=json.loads(payload['PARENT_INVENTORY.json'])['fits'][recovery['pair']]
    snapshot=parent_inventory['parent_input_lease']['value']
    if snapshot['dataset_root']!=worker['dataset_root']:
        raise ValueError('Declared native input root differs')
    initial_metadata={'INPUT_DATA_COMPLETE.json','SHA256SUMS',
        'source/samples_and_culture_branches.parquet','source/features_and_source_positions.parquet',
        'source/guides.parquet','source/shard_manifest.parquet'}
    if not initial_metadata <= set(snapshot['members']):
        raise ValueError('Complete native metadata startup closure is required')
    for name,item in snapshot['members'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Native input member escapes declared source root')
        target=metadata if name in initial_metadata else inputs
        label='native_metadata.'+name if name in initial_metadata else name
        add(target,label,dict(path=str(Path(snapshot['dataset_root'])/relative),sha256=item['sha256'],
                             size_bytes=item['file_identity']['st_size']))
    origins=json.loads(payload['RECOVERY_CHECK_METADATA_ORIGINS.json'])
    for key,name in (('output-readback.json','RECOVERY_CHECK_OUTPUT_READBACK.json'),
                     ('terminal.json','RECOVERY_CHECK_TERMINAL.json')):
        record=origins['copies'][key]
        if ({field:record['origin'][field] for field in ('sha256','size_bytes')} != identity(payload[name])
                or record['worker_visible_copy']!=ref(bundle/name,payload[name])):
            raise ValueError('Recovery metadata origin/copy bytes differ')
    return dict(schema='credo.frontier.worker-read-dependency-closure.v1',
        policy='home_visible_metadata_preflight_payload_verification_on_existing_consumption_path',
        run_id=plan['run_id'],worker_visible_roots=[str(path) for path in visible_roots],
        metadata_references=metadata,checkpoint_payload_references=tensors,input_payload_references=inputs,
        stat_and_read_roots=roots,lock_paths=locks,expected_absent_paths=absent,
        provenance_only_origins=origins['copies'],runtime_image_libraries_outside_study_closure=True,
        eager_checkpoint_or_native_payload_hashing_authorized=False,optimizer_calls=0)


def construct(root, pair, admission):
    root=Path(root)
    if pair not in PAIRS:
        raise ValueError('Exactly the four approved v4 replacements are supported')
    approval_raw=read(root/AUTHORITY)
    approval=json.loads(approval_raw)
    validate_replacement_authority(approval,admission,pair)
    if admission['replacement_authority_sha256']!=identity(approval_raw)['sha256']:
        raise ValueError('Replacement authority hash differs from admission')
    intake_raw=read(root/INTAKE)
    intake=json.loads(intake_raw)
    if (identity(intake_raw)['sha256']!=admission['replacement_intake_sha256']
            or intake.get('schema')!='credo.frontier.observation-continuation-replacement-intake.v1'
            or intake.get('status')!='passed' or intake.get('authority_sha256')!=identity(approval_raw)['sha256']
            or set(intake.get('fits',{}))!=set(PAIRS)):
        raise ValueError('Fresh replacement absence/zero-work intake differs')
    run='credo-frontier-obs-'+pair.lower()+'-20261016-v4'
    if admission.get('run_id')!=run:
        raise ValueError('Exact fresh v4 run identity required')
    replacement_row=intake['fits'][pair]
    expected_row=dict(run_id=admission['replaces_continuation_task_id'],new_run_id=run,
        failed_job_absent=True,failed_owned_pods_absent=True,failed_scientific_output_absent=True,
        new_job_absent=True,new_roots_absent=True,new_scientific_updates=0,parent_inventory_unchanged=True,
        parent_output_bytes=4*GIB-admission['storage_reserved_bytes'])
    if any(type(replacement_row.get(k)) is not type(v) or replacement_row.get(k)!=v for k,v in expected_row.items()):
        raise ValueError('Replacement predecessor is not the exact closed zero-work child')
    # This projection is used only for the retained builder's in-memory schema
    # checks. It is never staged, published, or entered into the journal.
    projected=dict(admission,run_id='credo-frontier-obs-'+pair.lower()+'-20261016-v3')
    payload,_,_=prior_builder.construct(root,pair,projected)
    old_plan=json.loads(payload['PLAN.json'])
    old_bundle=Path(old_plan['bundle_root'])
    phase='observation_'+pair.lower()+'_v4'
    bundle=Path(PROJECT)/phase/'bundle'
    worker=_rebase(json.loads(payload['WORKER_CONTRACT.json']),old_bundle.parent,bundle.parent)
    worker['worker_id']=run
    worker['trajectory_id']=worker['trajectory_id'].removesuffix('-v3-recovery')+'-v4-recovery'
    worker_raw=encoded(worker)
    recovery=_rebase(json.loads(payload['CONTINUATION_CONTRACT.json']),old_bundle.parent,bundle.parent)
    recovery['fit_contract']=ref(bundle/'WORKER_CONTRACT.json',worker_raw)
    recovery_raw=encoded(recovery)
    origins=_rebase(json.loads(payload['RECOVERY_CHECK_METADATA_ORIGINS.json']),old_bundle.parent,bundle.parent)
    payload.update({'WORKER_CONTRACT.json':worker_raw,'CONTINUATION_CONTRACT.json':recovery_raw,
        'ADMISSION.json':encoded(admission),'REPLACEMENT_AUTHORITY.json':approval_raw,
        'REPLACEMENT_INTAKE.json':intake_raw,
        'RECOVERY_CHECK_METADATA_ORIGINS.json':encoded(origins)})
    plan=_rebase(old_plan,old_bundle.parent,bundle.parent)
    plan.update(run_id=run,phase_directory=phase,control_root=str(Path(CONTROL)/phase),
        worker_contract=ref(bundle/'WORKER_CONTRACT.json',worker_raw),
        recovery_contract=ref(bundle/'CONTINUATION_CONTRACT.json',recovery_raw),
        admission=ref(bundle/'ADMISSION.json',payload['ADMISSION.json']),
        replacement_authority=ref(bundle/'REPLACEMENT_AUTHORITY.json',approval_raw),
        replacement_intake=ref(bundle/'REPLACEMENT_INTAKE.json',intake_raw),
        recovery_metadata_origins=ref(bundle/'RECOVERY_CHECK_METADATA_ORIGINS.json',payload['RECOVERY_CHECK_METADATA_ORIGINS.json']))
    prior_builder.validate_worker_visible_references(recovery,bundle,
        Path(recovery['parent_fit_contract']['path']).parent.parent.parent)
    dependencies=worker_dependency_closure(root,payload,plan)
    payload['WORKER_READ_DEPENDENCIES.json']=encoded(dependencies)
    plan['worker_read_dependencies']=ref(bundle/'WORKER_READ_DEPENDENCIES.json',payload['WORKER_READ_DEPENDENCIES.json'])
    validate_plan(plan);validate_admission(plan,admission)
    plan_raw=encoded(plan)
    payload['PLAN.json']=plan_raw
    payload['gpu-job.json']=encoded(manifest_from_template(json.loads(payload['gpu-job.json']),plan,plan_raw))
    payload.pop('INVENTORY.json',None)
    payload['INVENTORY.json']=encoded({name:identity(raw) for name,raw in payload.items()})
    archive=archive_bytes(payload)
    return payload,archive,dict(schema='credo.frontier.observation-build.v1',run_id=run,phase=phase,mode='fit',
        archive=identity(archive),archive_sha256=identity(archive)['sha256'],
        inventory_sha256=identity(payload['INVENTORY.json'])['sha256'],plan_sha256=identity(plan_raw)['sha256'],
        source_manifest_sha256=plan['source_manifest']['sha256'],phase_authority_sha256=plan['phase_authority']['sha256'],
        continuation_authority_sha256=recovery['authority']['sha256'],replacement_authority_sha256=identity(approval_raw)['sha256'],
        recovery_contract_sha256=identity(recovery_raw)['sha256'],worker_read_dependencies_sha256=plan['worker_read_dependencies']['sha256'],
        maximum_new_scientific_updates=18720,consumed_updates=1280,mixed_source_lineage=True,
        automatic_retry=False,automatic_successor=False,remote_submission=False)
