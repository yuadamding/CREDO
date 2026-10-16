"""Read-only terminal intake for the four explicitly mixed-source v3 fits.

This is not the fresh-fit adapter, a recovery owner, or a journal transition.
It reconciles already published metadata after exact terminal cleanup. No
checkpoint, count array, model, or optimizer is loaded. A successful intake
still requires downstream byte/tensor validation before observation reuse.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from transport import ROOT, remote, record
from observation import package, sha
from status_observation import STATUS_CODE
from accept_observation import (GIB, CONTROL_METADATA, COLLECT_CODE, digest,
    strict_json, reference, specification, terminal_snapshot)
from deployment.common import read


NUMERICAL = ['frontier/'+name for name in ('__init__.py', 'checkpoints.py',
    'native_inputs.py', 'observation.py', 'observation_training.py')]
STATIC_OUTPUT = ('COMPLETE.json', 'WORKER_STARTED.json', 'INPUT_LEASE.json',
                 'SELECTED_OBSERVATION.pt.json')


def require(value, expected, label):
    if not isinstance(value, dict) or any(k not in value or type(value[k]) is not type(v)
            or value[k] != v for k, v in expected.items()):
        raise ValueError(label+' differs')


def ref_shape(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'sha256', 'size_bytes'}
            or not isinstance(value['path'], str) or not value['path'].startswith('/')
            or re.fullmatch(r'[0-9a-f]{64}', value['sha256']) is None
            or type(value['size_bytes']) is not int or value['size_bytes'] <= 0):
        raise ValueError('Exact metadata reference required')


def output_names(plan, complete):
    ref = complete.get('input_exit')
    ref_shape(ref)
    root, path = Path(plan['scientific_output']), Path(ref['path'])
    if (path.parent != root/'input-exits'
            or re.fullmatch(r'[0-9a-f]{32}\.json', path.name) is None):
        raise ValueError('Exact claim-specific input exit required')
    claim = path.stem
    return STATIC_OUTPUT + ('input-exits/'+path.name, 'claims/'+claim+'.json',
                            'claims/'+claim+'.released.json')


def metadata_paths(plan, snapshot):
    complete = snapshot['records']['output/COMPLETE.json']['value']
    return ([plan['scientific_output']+'/'+name for name in output_names(plan, complete)]
        + [plan['control_root']+'/gpu/'+name for name in CONTROL_METADATA])


def input_projection(value):
    return {**{k:value[k] for k in ('schema', 'dataset_root', 'access_sha256',
        'metadata_index_sha256', 'source_ids', 'payload_hashes_verified_on_consumption',
        'all_payloads_consumed_claimed')}, 'members': {name:dict(sha256=item['sha256'],
        size_bytes=item['file_identity']['st_size']) for name,item in value['members'].items()}}


def validate_selected(complete, files, selected_receipt, worker, spec):
    """Verify the historical-plus-child audit selection, without reselecting."""
    selected = complete.get('selected_export', {})
    publication = selected.get('publication', {})
    if (publication != selected_receipt or publication.get('relative_uri') != 'SELECTED_OBSERVATION.pt'
            or files.get('SELECTED_OBSERVATION.pt') != {k:publication.get(k) for k in ('sha256','size_bytes')}
            or type(publication.get('size_bytes')) is not int
            or not 0 < publication['size_bytes'] <= 256*1024**2):
        raise ValueError('Selected inference export byte identity differs')
    require(selected, dict(completed_updates=20000, optimizer_resume=False,
        technical_holdout_excluded_from_scaling=True), 'Selected inference scope')
    audits = complete.get('audit_history', [])
    if [row.get('accepted_updates') for row in audits] != spec['audit_updates']:
        raise ValueError('Complete parent-plus-child fixed audit inventory required')
    for row in audits:
        risk, numerator, directions = (row.get(k) for k in
            ('risk', 'sum_normalized_cross_entropy', 'positive_cell_directions'))
        if (type(row.get('accepted_updates')) is not int or type(risk) not in (int,float)
                or not math.isfinite(risk) or risk < 0 or type(numerator) not in (int,float)
                or not math.isfinite(numerator) or type(directions) is not int or directions <= 0
                or type(row.get('positive_RNA_cells')) is not int or row['positive_RNA_cells'] < 1
                or directions > 2*row['positive_RNA_cells']
                or not math.isclose(risk, numerator/directions, rel_tol=1e-12, abs_tol=1e-12)
                or row.get('independent_donors') is not False or row.get('thinning_seed') != 1):
            raise ValueError('Technical audit arithmetic/scope differs')
        candidate = row.get('artifact', {})
        name = 'observation-candidate-%09d.pt' % row['accepted_updates']
        if (candidate.get('relative_uri') != name
                or files.get(name) != {k:candidate.get(k) for k in ('sha256','size_bytes')}):
            raise ValueError('Retained audit candidate byte identity differs')
    if (type(selected.get('selected_updates')) is not int or selected['selected_updates'] !=
            min(audits, key=lambda row:(row['risk'],row['accepted_updates']))['accepted_updates']):
        raise ValueError('Frozen technical selection rule differs')
    scale = selected.get('scaling', {})
    require(scale, dict(fold_id=worker['fold_id'], population_variance_ddof=0,
        scale_floor_rule='max(1e-6,fraction*median_positive_fitting_std)'), 'Fitting-only scale')
    if (type(scale.get('positive_RNA_cells')) is not int or scale['positive_RNA_cells'] < 2
            or type(scale.get('scale_floor')) not in (int,float)
            or not math.isfinite(scale['scale_floor']) or scale['scale_floor'] < 1e-6):
        raise ValueError('Fitting-only scale population/floor differs')
    return selected


def validate(build, plan, worker, protocol, recovery, parents, retained, uid, snapshot, collected):
    """Pure reconciliation of bytes from the bound package and terminal producer."""
    terminal_snapshot(snapshot, build, uid)
    require(plan, dict(mode='fit', maximum_new_scientific_updates=18720,
        combined_output_limit_bytes=4*GIB), 'Continuation lifecycle plan')
    require(worker, dict(mode='fit', output_root=plan['scientific_output'],
        source_manifest_sha256=build['source_manifest_sha256'], protocol_sha256=plan['protocol']['sha256'],
        protocol_path=plan['protocol']['path'], rna_axis_sha256=protocol['RNA_feature_axis_identity'],
        metadata_index_sha256=protocol['metadata_index_identity'], output_limit_bytes=plan['maximum_output_bytes']),
        'Continuation worker')
    require(recovery, dict(schema='credo.frontier.observation-continuation-contract.v1',
        mode='explicit_source_change_continuation', consumed_updates=1280, maximum_new_updates=18720,
        maximum_total_updates=20000, combined_output_limit_bytes=4*GIB,
        fit_contract=plan['worker_contract'], parent_retained_output_bytes=plan['parent_retained_output_bytes']),
        'Bound continuation contract')
    pair = recovery['pair']
    if (pair not in ('D1D2','D1D3','D1D4','D2D3') or ''.join(worker['training_donors']) != pair
            or worker['training_donors'] not in protocol['training_subsets']):
        raise ValueError('Continuation pair/fitting donors differ')
    require(retained, dict(schema='credo.frontier.observation-continuation-intake.v1', status='passed',
        authority_sha256=recovery['authority']['sha256']), 'Retained parent inventory')
    parent, prior = parents['fits'][pair], retained['fits'][pair]
    parent_bytes = plan['parent_retained_output_bytes']
    if (type(parent_bytes) is not int or not 0 < parent_bytes < 4*GIB
            or type(plan['maximum_output_bytes']) is not int
            or parent_bytes + plan['maximum_output_bytes'] != 4*GIB):
        raise ValueError('Combined parent-plus-child storage reservation differs')
    require(prior, dict(parent_output_bytes=parent_bytes, old_job_absent=True,
        new_job_absent=True, owned_pods_absent=True, new_roots_absent=True), 'Prelaunch parent closure')
    inventory_raw = (json.dumps(prior['inventory'],sort_keys=True,separators=(',',':'))+'\n').encode()
    if (hashlib.sha256(inventory_raw).hexdigest() != prior['inventory_sha256']
            or sum(row['st_size'] for row in prior['inventory'].values()) != parent_bytes
            or {k:prior['parent_input_lease'][k] for k in ('path','sha256','size_bytes')} != recovery['parent_input_lease']
            or recovery['parent_input_lease']['path'] != str(Path(parent['identity']['path']).parent/'INPUT_LEASE.json')):
        raise ValueError('Parent inventory/original input binding differs')
    if (collected.get('schema') != 'credo.frontier.observation-terminal-metadata.v1'
            or any(collected.get(k) is not False for k in
                ('remote_writes','model_payloads_opened','count_payloads_opened'))
            or set(collected.get('files',{})) != set(metadata_paths(plan,snapshot))):
        raise ValueError('Exact metadata-only continuation collection required')
    values, refs = {}, {}
    for path,item in collected['files'].items():
        raw = item['text'].encode('utf-8')
        ref = reference(raw,path)
        if (not 0 < len(raw) <= 32*1024**2 or set(item) != set(ref)|{'text'}
                or any(item[k] != v for k,v in ref.items())):
            raise ValueError('Collected metadata bytes differ')
        name = (Path(path).name if path.startswith(plan['control_root']+'/gpu/')
                else Path(path).relative_to(plan['scientific_output']).as_posix())
        values[name],refs[name] = strict_json(raw),ref
    for name in CONTROL_METADATA+('COMPLETE.json','WORKER_STARTED.json'):
        key = name if name in CONTROL_METADATA else 'output/'+name
        item = snapshot['records'].get(key)
        if not item or item.get('sha256') != refs[name]['sha256'] or item.get('value') != values[name]:
            raise ValueError('Metadata changed after exact terminal snapshot')
    complete, readback = values['COMPLETE.json'], values['output-readback.json']
    names = output_names(plan,complete)
    files = readback.get('files',{})
    require(readback, dict(schema='credo.frontier.observation-output-readback.v1', status='passed',
        run_id=build['run_id'], no_model_loading=True, parent_output_bytes=parent_bytes,
        combined_output_limit_bytes=4*GIB, source_continuation_contract=plan['recovery_contract'],
        new_scientific_updates_this_owner=18720, cumulative_scientific_updates=20000,
        parent_payloads_opened=False), 'Continuation producer readback')
    if not isinstance(files,dict) or not 0 < len(files) <= 20000:
        raise ValueError('Bounded producer inventory required')
    for name,item in files.items():
        path = PurePosixPath(name)
        if (not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != name
                or '\\' in name or any(p.startswith('.') for p in path.parts) or 'FAILED' in name.upper()
                or set(item) != {'sha256','size_bytes'} or re.fullmatch(r'[0-9a-f]{64}',item.get('sha256','')) is None
                or type(item.get('size_bytes')) is not int or item['size_bytes'] < 0):
            raise ValueError('Unsafe or malformed producer inventory')
    total = sum(item['size_bytes'] for item in files.values())
    require(readback, dict(total_bytes=total, child_output_bytes=total,
        combined_output_bytes=parent_bytes+total), 'Producer byte accounting')
    if not 0 < total <= plan['maximum_output_bytes']:
        raise ValueError('Combined output allowance exceeded')
    for name in names:
        if files.get(name) != {k:refs[name][k] for k in ('sha256','size_bytes')}:
            raise ValueError('Continuation metadata differs from producer readback')
    if (readback.get('complete_sha256') != refs['COMPLETE.json']['sha256']
            or values['terminal.json'].get('result_complete_sha256') != refs['COMPLETE.json']['sha256']):
        raise ValueError('Completion byte identity differs')
    spec,spec_sha = specification(protocol)
    original = parent['identity']['value']
    require(original, dict(fold_id=worker['fold_id'],seed=0,input_sha256=worker['input_access_sha256'],
        configuration_sha256=spec_sha,source_sha256=recovery['parent_source_manifest']['sha256'],
        worker_contract_sha256=recovery['parent_fit_contract']['sha256']), 'Original trajectory identity')
    identity = dict(original,trajectory_id=worker['trajectory_id'],source_sha256=worker['source_manifest_sha256'],
        worker_contract_sha256=plan['worker_contract']['sha256'],
        recovery_parent_checkpoint_sha256=parent['checkpoint_payload']['sha256'],
        recovery_authority_sha256=recovery['authority']['sha256'])
    require(complete, dict(schema='credo.frontier.observation-worker-completion.v1',status='complete',
        identity=identity,scope='scientific',accepted_updates=20000,attempted_updates=20000,
        scientific_updates=20000,actual_disposable_optimizer_calls=0,main_dynamics_fitted=False,
        held_out_donor_outcomes_opened=False,resume_from_accepted_updates=1280,consumed_parent_updates=1280,
        new_scientific_updates_this_owner=18720,source_continuation_contract=plan['recovery_contract'],
        mixed_source_continuation=True), 'Mixed-source scientific completion')
    compatibility = complete.get('qualification_compatibility',{})
    require(compatibility, dict(schema='credo.frontier.observation-source-continuation-compatibility.v1',
        status='compatible',policy='unchanged_numerical_sources_plus_actual_mount_retirement_and_exact_cuda_restore_v1',
        original_qualification_sha256=worker['qualification_sha256'],original_source_manifest_sha256=original['source_sha256'],
        continuation_source_manifest_sha256=worker['source_manifest_sha256'],
        recovery_check_sha256=recovery['recovery_check_completion']['sha256'],
        continuation_contract_sha256=plan['recovery_contract']['sha256'],numerical_sources_unchanged=NUMERICAL,
        mixed_source_continuation=True,fresh_numerical_qualification_performed=False), 'Qualified source-changing reuse')
    start = values['WORKER_STARTED.json']
    require(start,dict(identity=identity,scope='scientific',qualification_compatibility=compatibility,
        input_lease=refs['INPUT_LEASE.json']), 'Started continuation')
    release = values['startup-validated.json']
    require(release,dict(status='startup_released',uid=uid,run_id=build['run_id'],
        worker_contract_sha256=plan['worker_contract']['sha256'],lifecycle_plan_sha256=build['plan_sha256'],
        recovery_contract_sha256=plan['recovery_contract']['sha256'],actual_pod_audit_passed=True,
        storage_and_fileset_quota_passed=True), 'Actual startup release')
    if snapshot['records'].get('handshake/WORKER_EXIT.json',{}).get('value') != dict(returncode=0,run_id=build['run_id']):
        raise ValueError('Successful exact worker exit required')
    exit_name = names[-3]
    require(complete,dict(input_lease=refs['INPUT_LEASE.json'],input_exit=refs[exit_name]),'Input evidence references')
    claim = Path(exit_name).stem
    require(values[names[-2]],dict(claim_id=claim,worker_id=worker['worker_id'],
        trajectory_id=worker['trajectory_id']), 'Continuation claim owner')
    if values[names[-1]] != dict(claim_id=claim,exception_type=None,automatic_restart_authorized=False):
        raise ValueError('Successful exact claim release required')
    lease = values['INPUT_LEASE.json']
    require(lease,dict(schema='credo.frontier.native-input-lease.v1',dataset_root=worker['dataset_root'],
        access_sha256=worker['input_access_sha256'],metadata_index_sha256=worker['metadata_index_sha256'],
        payload_hashes_verified_on_consumption=True,all_payloads_consumed_claimed=False), 'Fitting input lease')
    if (not isinstance(lease.get('members'),dict) or not lease['members'] or not lease.get('source_ids')
            or input_projection(lease) != input_projection(prior['parent_input_lease']['value'])):
        raise ValueError('Continuation inputs differ from original fitting inputs')
    if values[exit_name] != dict(schema='credo.frontier.native-input-lease-exit.v1',status='unchanged',
            input_lease_sha256=digest(lease,ensure_ascii=False),access_sha256=worker['input_access_sha256'],
            checked_members=len(lease['members']),additional_count_array_scans=0,same_worker_identity_check=True):
        raise ValueError('Same-worker input lease exit differs')
    selected = validate_selected(complete,files,values['SELECTED_OBSERVATION.pt.json'],worker,spec)
    publication = selected['publication']
    return dict(schema='credo.frontier.observation-continuation-terminal-acceptance.v1',status='metadata_accepted',
        run_id=build['run_id'],job_uid=uid,plan_sha256=build['plan_sha256'],scientific_updates=20000,
        consumed_parent_updates=1280,new_scientific_updates_this_owner=18720,mixed_source_continuation=True,
        source_continuation_contract=plan['recovery_contract'],parent_checkpoint=parent['checkpoint_payload'],
        trajectory_identity=identity,specification=spec,fold=dict(fold_id=worker['fold_id'],
        training_donors=worker['training_donors'],feature_axis_sha256=worker['rna_axis_sha256']),
        artifact=dict(path=plan['scientific_output']+'/SELECTED_OBSERVATION.pt',
            sha256=publication['sha256'],size_bytes=publication['size_bytes']),
        receipt=refs['SELECTED_OBSERVATION.pt.json'],completion=refs['COMPLETE.json'],input_lease=refs['INPUT_LEASE.json'],
        input_exit=refs[exit_name],producer_readback=refs['output-readback.json'],selected_updates=selected['selected_updates'],
        producer_output_bytes=total,storage_reservation_bytes=plan['maximum_output_bytes'],parent_retained_output_bytes=parent_bytes,
        combined_output_bytes=parent_bytes+total,combined_output_limit_bytes=4*GIB,storage_reclamation_claimed=False,
        tensors_loaded=False,numerical_replay=False,downstream_loader_byte_and_tensor_checks_required=True,
        main_model_launch_authorized=False,journal_reconciled=False)


def local_context(path):
    path,build,plan = package(path)
    if plan.get('mode') != 'fit' or 'recovery_contract' not in plan:
        raise ValueError('Explicit v3 scientific continuation package required')
    created = strict_json(read(path/'CREATE.json'))['records']['live-binding.json']
    require(created,dict(job=build['run_id'],plan_sha256=build['plan_sha256']), 'Exact creation binding')
    def bound(ref,name):
        ref_shape(ref)
        raw = read(path/'bundle'/name,ref['sha256'])
        if ref != reference(raw,plan['bundle_root']+'/'+name):
            raise ValueError('Local frozen metadata binding differs')
        return strict_json(raw)
    worker = bound(plan['worker_contract'],'WORKER_CONTRACT.json')
    protocol = bound(plan['protocol'],'OBSERVATION_PROTOCOL.json')
    recovery = bound(plan['recovery_contract'],'CONTINUATION_CONTRACT.json')
    parents = bound(recovery['parent_intake'],'PARENT_INTAKE.json')
    retained = bound(recovery['parent_inventory'],'PARENT_INVENTORY.json')
    return path,build,plan,worker,protocol,recovery,parents,retained,created['uid']


def collect(path,output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(ROOT) or not output.parent.is_dir():
        raise ValueError('New prepared study receipt required')
    context = local_context(path)
    path,build,plan,*_,uid = context
    snapshot = remote(STATUS_CODE,dict(BUNDLE=plan['bundle_root'],PLAN_SHA=build['plan_sha256'],JOB=build['run_id'],UID=uid))
    terminal_snapshot(snapshot,build,uid)
    collected = remote(COLLECT_CODE,dict(PATHS=metadata_paths(plan,snapshot)))
    accepted = validate(*context[1:],snapshot,collected)
    accepted.update(worker_contract=plan['worker_contract'],protocol=plan['protocol'])
    record(output,dict(schema='credo.frontier.observation-continuation-terminal-intake.v1',
        package=str(path),at_utc=datetime.now(timezone.utc).isoformat(),validator_sha256=sha(__file__),
        snapshot=snapshot,collected=collected,accepted=accepted,journal_reconciled=False,remote_writes=False))
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='Exact completed v3 launch package')
    parser.add_argument('--output',required=True,help='New local terminal-intake receipt')
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.input,args.output),sort_keys=True))


if __name__ == '__main__':
    main()
