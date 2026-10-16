"""Build a fresh, explicitly mixed-source observation continuation package."""
import json
from pathlib import Path
from .common import (PROJECT, CONTROL, GIB, encoded, identity, read, validate_plan,
                     validate_admission)
from .build import SOURCE_MEMBERS, ADAPTER_MEMBERS, archive_bytes, manifest_from_template

PAIRS = ('D1D2', 'D1D3', 'D1D4', 'D2D3')
EXTRAS = ('frontier/observation_continuation.py', 'frontier/observation_recovery.py')


def ref(path, raw):
    return dict(path=str(path), **identity(raw))


def source_manifest(root):
    source = {name: read(Path(root)/name) for name in SOURCE_MEMBERS}
    manifest = encoded(dict(schema='credo.frontier.observation-worker-source.v1',
                            files={name:identity(raw)['sha256'] for name,raw in source.items()}))
    return source, manifest


def validate_worker_visible_references(recovery, bundle, retained_project):
    """All directly opened references must stay on the admitted home surfaces.

    This is a lexical contract guard, not a worker-mount availability check.
    Controller-only origins belong in the separate provenance record.
    """
    roots = (Path(bundle), Path(retained_project))
    for name, value in recovery.items():
        if isinstance(value, dict) and 'path' in value:
            path = Path(value['path'])
            if (not path.is_absolute() or '..' in path.parts
                    or not any(path.is_relative_to(root) for root in roots)):
                raise ValueError('Worker evidence reference is outside the visible home surface: '+name)


def construct(root, pair, admission):
    from frontier.coordination import WorkBinding
    root = Path(root)
    if pair not in PAIRS: raise ValueError('Only the four consumed-work continuations are admitted')
    prior = root/('observation_'+pair.lower()+'_v2_launch')/'bundle'
    original_inventory = json.loads(read(prior/'INVENTORY.json'))
    payload = {name:read(prior/name, item['sha256']) for name,item in original_inventory.items()}
    old_plan = json.loads(payload['PLAN.json'])
    old_worker = json.loads(payload['WORKER_CONTRACT.json'])
    authority = read(root/'OBSERVATION_CONTINUATION_V3_AUTHORITY.json')
    intake = read(root/'receipts/OBSERVATION_RETENTION_RECOVERY_BYTE_INTAKE_20261016_V1.json')
    inventory_raw = read(root/'receipts/OBSERVATION_CONTINUATION_V3_INTAKE_20261016.json')
    inventory = json.loads(inventory_raw)['fits'][pair]
    retained = inventory['parent_output_bytes']
    cap = 4*GIB-retained
    phase = 'observation_'+pair.lower()+'_v3'
    run = 'credo-frontier-obs-'+pair.lower()+'-20261016-v3'
    bundle = Path(PROJECT)/phase/'bundle'
    source, manifest = source_manifest(root)
    actual = WorkBinding.from_dict(admission['work_binding'])
    expected = WorkBinding.representation(old_worker['training_donors'],start_updates=1280,target_updates=20000)
    if (actual != expected or admission['continuation_authority_sha256'] != identity(authority)['sha256']
            or admission['source_manifest_sha256'] != identity(manifest)['sha256']
            or admission['storage_reserved_bytes'] != cap):
        raise ValueError('Continuation admission differs from scientific/source/storage bindings')
    worker = dict(old_worker, worker_id=run, trajectory_id=old_worker['trajectory_id']+'-v3-recovery',
        source_root=str(bundle/'source'), source_manifest=str(bundle/'SOURCE_MANIFEST.json'),
        source_manifest_sha256=identity(manifest)['sha256'],protocol_path=str(bundle/'OBSERVATION_PROTOCOL.json'),
        reader_source_root=str(bundle/'retained_reader'),output_root=str(bundle.parent/'output'),output_limit_bytes=cap)
    worker_raw = encoded(worker)
    check_snapshot = json.loads(read(root/'receipts/RETENTION_CHECK_V2_STATUS_20261016T161210Z.json'))
    check_plan = json.loads(read(root/'observation_retention_check_v2_launch/bundle/PLAN.json'))
    check_references = json.loads(read(root/'receipts/OBSERVATION_CONTINUATION_V3_CHECK_REFERENCES.json'))
    def check_ref(key, path):
        record = check_snapshot['records'][key]
        if record is None: raise ValueError('Required recovery evidence missing')
        matches = [item for item in check_references.values() if item['path'] == str(path)]
        if len(matches) != 1 or matches[0]['sha256'] != record['sha256']:
            raise ValueError('Fresh exact recovery metadata reference differs')
        return matches[0]
    check_output = Path(check_plan['scientific_output'])
    check_control = Path(check_plan['control_root'])/'gpu'
    # GPU workers see the approved home mount, not controller scratch. Preserve
    # the controller's exact metadata bytes in the worker-visible bundle. A
    # parsed snapshot is usable only when its canonical encoding reproduces
    # both the already bound SHA-256 and byte count; never silently reserialize
    # a different receipt or relabel the origin as worker-local evidence.
    copied_evidence = {}
    def copied_check_ref(key, path, name):
        origin = check_ref(key, path)
        record = check_snapshot['records'][key]
        if not isinstance(record.get('value'), dict):
            raise ValueError('Complete controller metadata snapshot required for exact copy')
        raw = encoded(record['value'])
        if identity(raw) != {field:origin[field] for field in ('sha256', 'size_bytes')}:
            raise ValueError('Canonical controller metadata bytes differ from bound original')
        publication = ref(bundle/name, raw)
        payload[name] = raw
        copied_evidence[key] = dict(origin=origin, worker_visible_copy=publication)
        return publication
    readback_ref = copied_check_ref('output-readback.json', check_control/'output-readback.json',
                                   'RECOVERY_CHECK_OUTPUT_READBACK.json')
    terminal_ref = copied_check_ref('terminal.json', check_control/'terminal.json',
                                   'RECOVERY_CHECK_TERMINAL.json')
    payload['RECOVERY_CHECK_METADATA_ORIGINS.json'] = encoded(dict(
        schema='credo.frontier.recovery-check-metadata-copies.v1',
        publication_policy='byte_identical_copy_from_hash_and_size_verified_snapshot',
        copies=copied_evidence))
    recovery = dict(schema='credo.frontier.observation-continuation-contract.v1',
        mode='explicit_source_change_continuation',pair=pair,
        fit_contract=ref(bundle/'WORKER_CONTRACT.json',worker_raw),
        authority=ref(bundle/'CONTINUATION_AUTHORITY.json',authority),
        parent_intake=ref(bundle/'PARENT_INTAKE.json',intake),
        parent_inventory=ref(bundle/'PARENT_INVENTORY.json',inventory_raw),
        parent_fit_contract=ref(Path(old_plan['bundle_root'])/'WORKER_CONTRACT.json',payload['WORKER_CONTRACT.json']),
        parent_source_manifest=ref(Path(old_plan['bundle_root'])/'SOURCE_MANIFEST.json',payload['SOURCE_MANIFEST.json']),
        parent_input_lease={k:v for k,v in inventory['parent_input_lease'].items() if k!='value'},
        recovery_check_worker_contract=check_plan['worker_contract'],
        recovery_check_completion=check_ref('output/COMPLETE.json',check_output/'COMPLETE.json'),
        recovery_check_restore=check_ref('output/'+pair+'/RESTORE_VERIFIED.json',check_output/pair/'RESTORE_VERIFIED.json'),
        recovery_check_readback=readback_ref,
        recovery_check_terminal=terminal_ref,
        parent_retained_output_bytes=retained,combined_output_limit_bytes=4*GIB,
        consumed_updates=1280,maximum_new_updates=18720,maximum_total_updates=20000,
        extra_sources={name:identity(read(root/name))['sha256'] for name in EXTRAS})
    validate_worker_visible_references(recovery, bundle, Path(old_plan['bundle_root']).parent.parent)
    recovery_raw = encoded(recovery)
    payload.update({name:read(root/'deployment'/name) for name in ADAPTER_MEMBERS})
    payload.update({'source/'+name:raw for name,raw in source.items()})
    payload.update({'source/'+name:read(root/name) for name in EXTRAS})
    plan = dict(old_plan,run_id=run,phase_directory=phase,bundle_root=str(bundle),
        worker_output=str(bundle.parent/'handshake'),scientific_output=worker['output_root'],
        control_root=str(Path(CONTROL)/phase),maximum_output_bytes=cap,maximum_new_scientific_updates=18720,
        parent_retained_output_bytes=retained,combined_output_limit_bytes=4*GIB,
        worker_contract=ref(bundle/'WORKER_CONTRACT.json',worker_raw),
        source_manifest=ref(bundle/'SOURCE_MANIFEST.json',manifest),
        recovery_contract=ref(bundle/'CONTINUATION_CONTRACT.json',recovery_raw))
    payload.update({'WORKER_CONTRACT.json':worker_raw,'SOURCE_MANIFEST.json':manifest,
        'CONTINUATION_CONTRACT.json':recovery_raw,'CONTINUATION_AUTHORITY.json':authority,
        'PARENT_INTAKE.json':intake,'PARENT_INVENTORY.json':inventory_raw,'ADMISSION.json':encoded(admission)})
    for field,name in (('protocol','OBSERVATION_PROTOCOL.json'),('phase_authority','OBSERVATION_PHASE_CONTRACT.json'),
                       ('parent_authority','AUTHORITY.json'),('admission','ADMISSION.json')):
        plan[field]=ref(bundle/name,payload[name])
    plan['controller_sources']={name:identity(payload[name])['sha256'] for name in old_plan['controller_sources']}
    plan['control_sha256']=plan['controller_sources']['control.py']
    validate_plan(plan);validate_admission(plan,admission)
    plan_raw=encoded(plan)
    payload['PLAN.json']=plan_raw
    payload['gpu-job.json']=encoded(manifest_from_template(json.loads(payload['gpu-job.json']),plan,plan_raw))
    payload['INVENTORY.json']=encoded({name:identity(raw) for name,raw in payload.items()})
    archive=archive_bytes(payload)
    report=dict(schema='credo.frontier.observation-build.v1',run_id=run,phase=phase,mode='fit',
        archive=identity(archive),archive_sha256=identity(archive)['sha256'],
        inventory_sha256=identity(payload['INVENTORY.json'])['sha256'],plan_sha256=identity(plan_raw)['sha256'],
        source_manifest_sha256=identity(manifest)['sha256'],phase_authority_sha256=plan['phase_authority']['sha256'],
        continuation_authority_sha256=identity(authority)['sha256'],recovery_contract_sha256=identity(recovery_raw)['sha256'],
        maximum_new_scientific_updates=18720,consumed_updates=1280,mixed_source_lineage=True,
        automatic_retry=False,automatic_successor=False,remote_submission=False)
    return payload,archive,report
