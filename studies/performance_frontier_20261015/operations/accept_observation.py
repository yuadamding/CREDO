"""Accept a completed observation fit from exact metadata; never run a model.

Collection is explicit and read-only remotely, not a polling or retry owner.
The existing worker/lifecycle byte readback is retained as producer evidence.
The downstream observation loader still hashes/deserializes the selected export
on its admitted worker. This command cannot authorize main-model execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

from transport import ROOT, remote, record
from observation import package, sha
from status_observation import STATUS_CODE

sys.path.insert(0, str(ROOT))
from deployment.common import read
from frontier.coordination import AdmissionJournal, WorkBinding, LIVE


GIB = 1024 ** 3
OUTPUT_METADATA = ('COMPLETE.json', 'WORKER_STARTED.json', 'INPUT_LEASE.json',
                   'INPUT_EXIT.json', 'SELECTED_OBSERVATION.pt.json')
CONTROL_METADATA = ('terminal.json', 'cleanup.json', 'output-readback.json', 'startup-validated.json')


def digest(value, *, ensure_ascii=True):
    # Native-input/specification semantic identity deliberately has no newline.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=ensure_ascii, allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate metadata key')
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))
    if not isinstance(result, dict):
        raise ValueError('Metadata object required')
    return result


def reference(raw, path):
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def specification(protocol):
    shape = protocol['architecture_dimensions']
    spec = dict(logical_batch=protocol['logical_batch_positive_training_cells'],
        maximum_updates=protocol['maximum_accepted_updates_each'],
        audit_updates=protocol['audit_candidates_accepted_updates'],
        checkpoint_interval=protocol['checkpoint_interval_accepted_updates'], features=shape[0],
        hidden_dims=shape[1:3], latent_dim=shape[3],
        maximum_dense_bytes=protocol['maximum_dense_working_estimate_bytes'], synthetic_fixture=False)
    identity = digest(dict(specification=spec, optimizer='AdamW', lr=1e-3, weight_decay=1e-4,
        betas=[0.9, 0.999], epsilon=1e-8, gradient_clip=1.0, initialization_seed=0,
        technical_holdout=['row_hash', 0, 0.05], audit_thinning_seed=1,
        training_thinning_seed='accepted_update_index', scale_excludes_holdout=True))
    return spec, identity


def terminal_snapshot(snapshot, build, uid):
    if (snapshot.get('schema') != 'credo.frontier.exact-observation-status.v1'
            or snapshot.get('run_id') != build['run_id'] or snapshot.get('uid') != uid
            or snapshot.get('plan_sha256') != build['plan_sha256']
            or snapshot.get('remote_writes') is not False
            or snapshot.get('job', 'missing') is not None or snapshot.get('pods') != []):
        raise ValueError('Exact cleaned Job/Pod absence snapshot required')
    records = snapshot['records']
    for name in ('supervisor-failed.json', 'startup-failure.json', 'handshake/ENTRY_FAILED.json',
                 'output/WORKER_FAILED.json'):
        if records.get(name, 'missing') is not None:
            raise ValueError('Failure or missing lifecycle evidence: ' + name)
    for name in CONTROL_METADATA:
        if not isinstance(records.get(name), dict):
            raise ValueError('Completed lifecycle evidence required: ' + name)
    terminal = records['terminal.json']['value']
    expected = dict(status='passed', job=build['run_id'], uid=uid, run_id=build['run_id'],
        kubernetes_complete=True, application_pass=True, reason=None, mode='fit',
        lifecycle_plan_sha256=build['plan_sha256'], automatic_successor=False)
    if any(terminal.get(k) != v for k, v in expected.items()):
        raise ValueError('Successful exact observation terminal required')
    cleanup = records['cleanup.json']['value']
    if (any(cleanup.get(k) != v for k, v in dict(status='complete', job=build['run_id'], uid=uid).items())
            or len(cleanup.get('reads', [])) < 2
            or cleanup['reads'][-2:] != [dict(job_absent=True, owned_pods=0)] * 2):
        raise ValueError('Two-read exact cleanup evidence required')


COLLECT_CODE = r'''
import hashlib,json,os,stat
from pathlib import Path
def read_one(path):
 p=Path(path)
 for ancestor in (p.parent,)+tuple(p.parent.parents):
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise ValueError('Unsafe receipt ancestor')
 before=p.lstat()
 if (not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_uid!=307469
     or not 0<before.st_size<=32*1024**2):raise ValueError('Receipt type/owner/size differs')
 signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
 with os.fdopen(os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
  if signature(os.fstat(stream.fileno()))!=signature(before):raise ValueError('Receipt open race')
  raw=stream.read(32*1024**2+1)
  if signature(os.fstat(stream.fileno()))!=signature(before):raise ValueError('Receipt changed')
 if signature(p.lstat())!=signature(before) or len(raw)!=before.st_size:raise ValueError('Receipt changed')
 return dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw),text=raw.decode('utf-8'))
files={path:read_one(path) for path in PATHS}
print('RECEIPT='+json.dumps(dict(schema='credo.frontier.observation-terminal-metadata.v1',
 files=files,remote_writes=False,model_payloads_opened=False,count_payloads_opened=False)))
'''


def metadata_paths(plan):
    return ([plan['scientific_output'] + '/' + name for name in OUTPUT_METADATA]
        + [plan['control_root'] + '/gpu/' + name for name in CONTROL_METADATA])


def validate(build, plan, worker, protocol, uid, snapshot, collected):
    """Pure metadata reconciliation; callers bind the local package first."""
    terminal_snapshot(snapshot, build, uid)
    if (plan.get('mode') != 'fit' or plan.get('maximum_new_scientific_updates') != 20000
            or plan.get('maximum_output_bytes') != 4*GIB
            or worker.get('mode') != 'fit' or worker.get('output_root') != plan['scientific_output']
            or worker.get('source_manifest_sha256') != build['source_manifest_sha256']
            or worker.get('protocol_sha256') != plan['protocol']['sha256']
            or worker.get('protocol_path') != plan['protocol']['path']
            or worker.get('training_donors') not in protocol.get('training_subsets', [])
            or worker.get('rna_axis_sha256') != protocol.get('RNA_feature_axis_identity')
            or worker.get('metadata_index_sha256') != protocol.get('metadata_index_identity')):
        raise ValueError('Exact scientific observation package required')
    if (collected.get('schema') != 'credo.frontier.observation-terminal-metadata.v1'
            or any(collected.get(k) is not False for k in
                   ('remote_writes', 'model_payloads_opened', 'count_payloads_opened'))
            or set(collected.get('files', {})) != set(metadata_paths(plan))):
        raise ValueError('Exact metadata-only collection required')
    values, refs = {}, {}
    for path, item in collected['files'].items():
        raw = item['text'].encode('utf-8')
        actual = reference(raw, path)
        if (not 0 < len(raw) <= 32*1024**2 or set(item) != set(actual) | {'text'}
                or any(item[k] != v for k, v in actual.items())):
            raise ValueError('Collected metadata bytes differ')
        values[Path(path).name], refs[Path(path).name] = strict_json(raw), actual
    for name in CONTROL_METADATA + ('COMPLETE.json', 'WORKER_STARTED.json'):
        key = name if name in CONTROL_METADATA else 'output/' + name
        previous = snapshot['records'].get(key)
        if (not isinstance(previous, dict) or previous.get('sha256') != refs[name]['sha256']
                or previous.get('value') != values[name]):
            raise ValueError('Terminal metadata changed after exact status snapshot')
    complete, readback, lease, exit_record = (values[k] for k in
        ('COMPLETE.json', 'output-readback.json', 'INPUT_LEASE.json', 'INPUT_EXIT.json'))
    files = readback.get('files', {})
    if (readback.get('schema') != 'credo.frontier.observation-output-readback.v1'
            or readback.get('status') != 'passed' or readback.get('run_id') != build['run_id']
            or readback.get('no_model_loading') is not True or not isinstance(files, dict) or not files
            or len(files) > 20000):
        raise ValueError('Bounded producer output readback required')
    for name, entry in files.items():
        parts = PurePosixPath(name)
        if (not parts.parts or parts.is_absolute() or '..' in parts.parts or parts.as_posix() != name
                or '\\' in name or any(p.startswith('.') for p in parts.parts)
                or 'FAILED' in name.upper() or set(entry) != {'sha256', 'size_bytes'}
                or re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')) is None
                or type(entry.get('size_bytes')) is not int or entry['size_bytes'] < 0):
            raise ValueError('Unsafe or malformed producer inventory')
    if (readback.get('total_bytes') != sum(e['size_bytes'] for e in files.values())
            or not 0 < readback['total_bytes'] <= plan['maximum_output_bytes']):
        raise ValueError('Producer inventory byte accounting differs')
    for name in OUTPUT_METADATA:
        expected = {k: refs[name][k] for k in ('sha256', 'size_bytes')}
        if files.get(name) != expected:
            raise ValueError('Output metadata differs from producer byte readback')
    complete_sha = refs['COMPLETE.json']['sha256']
    if (readback.get('complete_sha256') != complete_sha
            or values['terminal.json'].get('result_complete_sha256') != complete_sha):
        raise ValueError('Completion byte identity differs')
    spec, spec_sha = specification(protocol)
    identity = dict(trajectory_id=worker['trajectory_id'], fold_id=worker['fold_id'], seed=0,
        input_sha256=worker['input_access_sha256'], configuration_sha256=spec_sha,
        source_sha256=worker['source_manifest_sha256'], worker_contract_sha256=plan['worker_contract']['sha256'])
    expected = dict(schema='credo.frontier.observation-worker-completion.v1', status='complete',
        identity=identity, scope='scientific', accepted_updates=20000, attempted_updates=20000,
        scientific_updates=20000, actual_disposable_optimizer_calls=0,
        main_dynamics_fitted=False, held_out_donor_outcomes_opened=False, resume_from_accepted_updates=0)
    if (any(complete.get(k) != v for k, v in expected.items())
            or any(type(complete.get(k)) is not int for k in
                   ('accepted_updates', 'attempted_updates', 'scientific_updates',
                    'actual_disposable_optimizer_calls', 'resume_from_accepted_updates'))):
        raise ValueError('Scientific completion/trajectory differs')
    start = values['WORKER_STARTED.json']
    compatibility = complete.get('qualification_compatibility', {})
    if (start.get('identity') != identity or start.get('scope') != 'scientific'
            or start.get('qualification_compatibility') != compatibility
            or any(compatibility.get(k) != v for k, v in dict(status='compatible',
                source_manifest_sha256=worker['source_manifest_sha256'],
                protocol_sha256=worker['protocol_sha256'],
                qualification_receipt_sha256=worker['qualification_sha256']).items())):
        raise ValueError('Started/completed qualification identity differs')
    release = values['startup-validated.json']
    worker_exit = snapshot['records'].get('handshake/WORKER_EXIT.json')
    if (release.get('status') != 'startup_released' or release.get('uid') != uid
            or release.get('run_id') != build['run_id']
            or release.get('worker_contract_sha256') != plan['worker_contract']['sha256']
            or release.get('lifecycle_plan_sha256') != build['plan_sha256']
            or release.get('actual_pod_audit_passed') is not True
            or release.get('storage_and_fileset_quota_passed') is not True
            or not worker_exit or worker_exit['value'] != dict(returncode=0, run_id=build['run_id'])):
        raise ValueError('Actual startup release and successful worker exit required')
    for key, name in (('input_lease', 'INPUT_LEASE.json'), ('input_exit', 'INPUT_EXIT.json')):
        if complete.get(key) != refs[name]:
            raise ValueError('Completion input reference differs')
    if (start.get('input_lease') != refs['INPUT_LEASE.json']
            or lease.get('schema') != 'credo.frontier.native-input-lease.v1'
            or lease.get('dataset_root') != worker['dataset_root']
            or lease.get('access_sha256') != worker['input_access_sha256']
            or lease.get('metadata_index_sha256') != worker['metadata_index_sha256']
            or lease.get('payload_hashes_verified_on_consumption') is not True
            or lease.get('all_payloads_consumed_claimed') is not False
            or not isinstance(lease.get('members'), dict) or not lease['members']
            or not lease.get('source_ids')):
        raise ValueError('Exact fitting input lease required')
    expected_exit = dict(schema='credo.frontier.native-input-lease-exit.v1', status='unchanged',
        input_lease_sha256=digest(lease, ensure_ascii=False), access_sha256=worker['input_access_sha256'],
        checked_members=len(lease['members']), additional_count_array_scans=0, same_worker_identity_check=True)
    if exit_record != expected_exit:
        raise ValueError('Same-worker semantic input lease exit differs')
    selected = complete.get('selected_export', {})
    selected_ref = selected.get('publication', {})
    if (selected_ref != values['SELECTED_OBSERVATION.pt.json']
            or selected_ref.get('relative_uri') != 'SELECTED_OBSERVATION.pt'
            or files.get('SELECTED_OBSERVATION.pt') != {k: selected_ref.get(k) for k in ('sha256', 'size_bytes')}
            or not 0 < selected_ref.get('size_bytes', 0) <= 256*1024**2
            or selected.get('completed_updates') != 20000 or selected.get('optimizer_resume') is not False
            or selected.get('technical_holdout_excluded_from_scaling') is not True):
        raise ValueError('Selected inference export differs')
    audits = complete.get('audit_history', [])
    if [a.get('accepted_updates') for a in audits] != spec['audit_updates']:
        raise ValueError('Complete fixed technical audit inventory required')
    for row in audits:
        risk, numerator, directions = (row.get(k) for k in ('risk', 'sum_normalized_cross_entropy', 'positive_cell_directions'))
        if (type(risk) not in (int, float) or not math.isfinite(risk) or risk < 0
                or type(numerator) not in (int, float) or not math.isfinite(numerator)
                or type(directions) is not int or directions <= 0
                or type(row.get('positive_RNA_cells')) is not int or row['positive_RNA_cells'] < 1
                or directions > 2*row['positive_RNA_cells']
                or not math.isclose(risk, numerator / directions, rel_tol=1e-12, abs_tol=1e-12)
                or row.get('independent_donors') is not False or row.get('thinning_seed') != 1):
            raise ValueError('Invalid technical audit arithmetic/scope')
        candidate = row.get('artifact', {})
        if (candidate.get('relative_uri') != 'observation-candidate-%09d.pt' % row['accepted_updates']
                or files.get(candidate['relative_uri']) != {k: candidate.get(k) for k in ('sha256', 'size_bytes')}):
            raise ValueError('Retained audit candidate byte identity differs')
    # Verify the already selected candidate; do not replace it or load candidates.
    if selected.get('selected_updates') != min(audits, key=lambda a: (a['risk'], a['accepted_updates']))['accepted_updates']:
        raise ValueError('Export contradicts frozen technical selection rule')
    scale = selected.get('scaling', {})
    if (scale.get('fold_id') != worker['fold_id'] or scale.get('population_variance_ddof') != 0
            or type(scale.get('positive_RNA_cells')) is not int or scale['positive_RNA_cells'] < 2
            or scale.get('scale_floor_rule') != 'max(1e-6,fraction*median_positive_fitting_std)'
            or type(scale.get('scale_floor')) not in (int, float)
            or not math.isfinite(scale['scale_floor']) or scale['scale_floor'] < 1e-6):
        raise ValueError('Fitting-only scale identity differs')
    return dict(schema='credo.frontier.observation-terminal-acceptance.v1', status='metadata_accepted',
        run_id=build['run_id'], job_uid=uid, plan_sha256=build['plan_sha256'], scientific_updates=20000,
        trajectory_identity=identity, specification=spec,
        fold=dict(fold_id=worker['fold_id'], training_donors=worker['training_donors'],
                  feature_axis_sha256=worker['rna_axis_sha256']),
        artifact=dict(path=plan['scientific_output']+'/SELECTED_OBSERVATION.pt',
                      sha256=selected_ref['sha256'], size_bytes=selected_ref['size_bytes']),
        receipt=refs['SELECTED_OBSERVATION.pt.json'], completion=refs['COMPLETE.json'],
        input_lease=refs['INPUT_LEASE.json'], input_exit=refs['INPUT_EXIT.json'],
        producer_readback=refs['output-readback.json'], selected_updates=selected['selected_updates'],
        producer_output_bytes=readback['total_bytes'], storage_reservation_bytes=4*GIB,
        storage_reclamation_claimed=False, tensors_loaded=False, numerical_replay=False,
        downstream_loader_byte_and_tensor_checks_required=True, main_model_launch_authorized=False)


def local_context(path):
    path, build, plan = package(path)
    if plan.get('mode') != 'fit':
        raise ValueError('Only a scientific observation fit can enter terminal intake')
    created = strict_json(read(path/'CREATE.json'))['records']['live-binding.json']
    if created.get('job') != build['run_id'] or created.get('plan_sha256') != build['plan_sha256']:
        raise ValueError('Exact creation binding differs')
    worker_raw = read(path/'bundle/WORKER_CONTRACT.json', plan['worker_contract']['sha256'])
    protocol_raw = read(path/'bundle/OBSERVATION_PROTOCOL.json', plan['protocol']['sha256'])
    return path, build, plan, created['uid'], strict_json(worker_raw), strict_json(protocol_raw)


def collect(path, output):
    output = Path(output).absolute()
    if output.exists() or not output.is_relative_to(ROOT) or not output.parent.is_dir():
        raise ValueError('New prepared study receipt required')
    path, build, plan, uid, worker, protocol = local_context(path)
    snapshot = remote(STATUS_CODE, dict(BUNDLE=plan['bundle_root'], PLAN_SHA=build['plan_sha256'],
                                      JOB=build['run_id'], UID=uid))
    # No detailed reads for a running/pending job. Caller verifies chain first.
    terminal_snapshot(snapshot, build, uid)
    collected = remote(COLLECT_CODE, dict(PATHS=metadata_paths(plan)))
    accepted = validate(build, plan, worker, protocol, uid, snapshot, collected)
    accepted.update(worker_contract=reference(read(path/'bundle/WORKER_CONTRACT.json'), plan['worker_contract']['path']),
                    protocol=reference(read(path/'bundle/OBSERVATION_PROTOCOL.json'), plan['protocol']['path']))
    evidence = dict(schema='credo.frontier.observation-intake.v1', package=str(path),
        at_utc=datetime.now(timezone.utc).isoformat(), validator_sha256=sha(__file__),
        snapshot=snapshot, collected=collected, accepted=accepted,
        journal_reconciled=False, remote_writes=False)
    record(output, evidence)
    return accepted


def reconcile(intake_path, output):
    """Free only the completed H100 slot; retain the entire storage reservation.

    If publication interrupted after a committed journal event, explicit replay
    accepts only that same event/evidence. It never retries numerical work.
    """
    intake_path, output = map(lambda p: Path(p).absolute(), (intake_path, output))
    if (output.exists() or not output.is_relative_to(ROOT) or not output.parent.is_dir()
            or not intake_path.is_relative_to(ROOT)):
        raise ValueError('New local reconciliation receipt and study intake required')
    raw = read(intake_path)
    evidence = strict_json(raw)
    if (evidence.get('schema') != 'credo.frontier.observation-intake.v1'
            or evidence.get('validator_sha256') != sha(__file__) or evidence.get('remote_writes') is not False):
        raise ValueError('Exact versioned intake evidence required')
    path, build, plan, uid, worker, protocol = local_context(evidence['package'])
    accepted = validate(build, plan, worker, protocol, uid, evidence['snapshot'], evidence['collected'])
    accepted.update(worker_contract=reference(read(path/'bundle/WORKER_CONTRACT.json'), plan['worker_contract']['path']),
                    protocol=reference(read(path/'bundle/OBSERVATION_PROTOCOL.json'), plan['protocol']['path']))
    if accepted != evidence.get('accepted'):
        raise ValueError('Intake acceptance differs on independent local reconciliation')
    admission = strict_json(read(path/'bundle/ADMISSION.json', plan['admission']['sha256']))
    if admission.get('work_binding') != WorkBinding.representation(tuple(worker['training_donors'])).to_dict():
        raise ValueError('Reservation donor subset differs from accepted observation model')
    evidence_sha = hashlib.sha256(raw).hexdigest()
    with AdmissionJournal(ROOT/'admission', study_identity_sha256=sha(ROOT/'AUTHORITY.json')) as journal:
        state = journal.snapshot()
        task = state['tasks'].get(build['run_id'], {})
        if (task.get('worker_id') != build['run_id'] or task.get('kind') != 'representation'
                or task.get('scheduler_id') != uid or task.get('scientific_update_reservation') != 20000
                or task.get('storage_reserved_bytes') != 4*GIB
                or task.get('work_binding') != admission.get('work_binding')
                or task.get('admission_sha256') != admission.get('phase_authority_sha256')):
            raise ValueError('Exact observation reservation differs')
        if task['status'] == 'complete':
            if task.get('evidence_sha256') != evidence_sha:
                raise ValueError('Completed reservation cites another intake')
            action = 'reused_exact_committed_terminal_event'
        elif task['status'] in LIVE:
            state = journal.reconcile(build['run_id'], status='complete', exact_scheduler_id=uid,
                                      evidence_sha256=evidence_sha)
            action = 'completed_observation_reservation'
        else:
            raise ValueError('Failed/cancelled observation reservation cannot be accepted')
        result = dict(schema='credo.frontier.observation-terminal-reconciliation.v1', status='complete',
            run_id=build['run_id'], job_uid=uid, action=action, intake=reference(raw, intake_path),
            journal_event_sha256=state['last_sha256'], live_h100_reservations=sum(
                t['status'] in LIVE for t in state['tasks'].values()),
            storage_reserved_bytes=4*GIB, storage_reclaimed_bytes=0,
            automatic_successor=False, remote_writes=False, scientific_updates_applied=0)
        record(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('collect', 'reconcile'))
    parser.add_argument('--input', required=True, help='Exact launch package or previously collected intake')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = collect(args.input, args.output) if args.action == 'collect' else reconcile(args.input, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
