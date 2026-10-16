"""Metadata-only contract for the separately approved zero-update check."""
import json
from pathlib import Path
try:
    from .common import CONTROL, CORE_SHA, GIB, IMAGE, PROJECT, STUDY, read, read_json
    from .main_qualification_common import same, digest, reference, read_reference, validate_gpu, validate_resources
except ImportError:
    from common import CONTROL, CORE_SHA, GIB, IMAGE, PROJECT, STUDY, read, read_json
    from main_qualification_common import same, digest, reference, read_reference, validate_gpu, validate_resources

RUN = 'credo-frontier-retention-check-20261016-v2'
PHASE = 'observation_retention_check_v2'
# Keep the exact failed attempt readable by the status adapter. This is not
# permission to submit it again; the new builder accepts only RUN/PHASE above.
ATTEMPTS = {
    'observation_retention_check_v1': 'credo-frontier-retention-check-20261016-v1',
    PHASE: RUN,
}
PAIRS = ('D1D2', 'D1D3', 'D1D4', 'D2D3')
RUNTIME = dict(python='3.12.13', torch='2.6.0+cu124', cuda_runtime='12.4', numpy='2.5.1', scipy='1.18.0')


def validate_authority(authority):
    phase = authority.get('phase')
    if phase not in ATTEMPTS:
        raise ValueError('Unknown explicit recovery attempt')
    same(authority, dict(schema='credo.frontier.retention-check-authority.v1', status='authorized',
        run_id=ATTEMPTS[phase], phase=phase, study_id=STUDY, optimizer_calls=0, scientific_updates=0,
        checkpoint_pairs=list(PAIRS), output_limit_bytes=4*GIB, gpus=1, cpus=24,
        host_memory_gib=200, threads=8, runtime_image=IMAGE, raw_data_access=False,
        parent_writes=False, automatic_retry=False, automatic_successor=False,
        scientific_relaunch_authorized=False, elapsed_time_deadline=None, LARRY_access=False))
    digest(authority['parent_intake_sha256'])
    return authority


def validate_plan(plan, bundle=None):
    phase = plan.get('phase_directory')
    if phase not in ATTEMPTS:
        raise ValueError('Unknown explicit recovery attempt')
    base = Path(PROJECT)/phase
    same(plan, dict(schema='credo.frontier.retention-check-lifecycle.v1', mode='retention-check',
        run_id=ATTEMPTS[phase], phase_directory=phase, image=IMAGE, bundle_root=str(base/'bundle'),
        worker_output=str(base/'handshake'), scientific_output=str(base/'output'),
        control_root=str(Path(CONTROL)/phase), authorized_CPUs=24, authorized_memory_GiB=200,
        threads=8, gpu_ceiling=1, aggregate_h100_ceiling=4, automatic_retries=0,
        automatic_successor=False, elapsed_time_deadline_seconds=None,
        runtime_deadline_policy='no_runtime_deadline_user_20260919',
        startup_gate='inline_h100_controller_release_v1', maximum_output_bytes=4*GIB,
        maximum_disposable_optimizer_steps=0, planned_disposable_optimizer_steps=0,
        maximum_new_scientific_updates=0, raw_data_access=False, LARRY_access=False))
    if bundle is not None and str(bundle) != plan['bundle_root']:
        raise ValueError('Exact recovery-check bundle required')
    return plan


def validate_admission(worker, admission):
    same(admission, dict(schema='credo.frontier.retention-check-admission.v1', status='reserved',
        run_id=worker['run_id'], work_binding=dict(kind='qualification', identity=[worker['authority']['sha256']],
            start_updates=0, target_updates=0), phase_authority_sha256=worker['authority']['sha256'],
        source_manifest_sha256=worker['source_manifest']['sha256'], scientific_update_reservation=0,
        storage_reserved_bytes=4*GIB, disposable_optimizer_call_limit=0,
        automatic_retry=False, automatic_successor=False))
    for key in ('journal_event_sha256', 'journal_identity_sha256', 'capacity_receipt_sha256'):
        digest(admission.get(key))
    for key, lower, upper in (('live_h100_reservations',1,4),('aggregate_reserved_output_bytes',4*GIB,384*GIB)):
        if type(admission.get(key)) is not int or not lower <= admission[key] <= upper:
            raise ValueError('Recovery aggregate reservation differs')


def verify_bundle(plan, bundle):
    validate_plan(plan, bundle)
    bundle = Path(bundle)
    if plan['controller_sources'].get('lifecycle_core.py') != CORE_SHA:
        raise ValueError('Pinned lifecycle changed')
    for name, sha in plan['controller_sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe controller member')
        read(bundle/name, digest(sha))
    worker = json.loads(read_reference(plan['worker_contract']))
    same(worker, dict(schema='credo.frontier.retention-check-worker.v1',run_id=plan['run_id'],
        phase=plan['phase_directory'], source_root=str(bundle/'source'), output_root=plan['scientific_output'],
        runtime_image=IMAGE, expected_runtime=RUNTIME, optimizer_calls=0, output_limit_bytes=4*GIB))
    for key in ('source_manifest','authority','admission','parent_intake'):
        if Path(worker[key]['path']).parent != bundle:
            raise ValueError('Bound recovery metadata escaped bundle')
        read_reference(worker[key])
    for key, wk in (('source_manifest','source_manifest'),('phase_authority','authority'),('admission','admission')):
        if plan[key] != worker[wk]:
            raise ValueError('Worker/controller binding differs')
    source = json.loads(read_reference(worker['source_manifest']))
    same(source, dict(entry='frontier/observation_recovery_check.py'))
    for name, ref in source['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe source member')
        if len(read(bundle/'source'/name,ref['sha256'])) != ref['size_bytes']:
            raise ValueError('Source size differs')
    authority = validate_authority(json.loads(read_reference(worker['authority'])))
    same(authority, dict(run_id=worker['run_id'], phase=worker['phase']))
    if authority['parent_intake_sha256'] != worker['parent_intake']['sha256']:
        raise ValueError('Checkpoint intake not authorized')
    admission = json.loads(read_reference(worker['admission']))
    validate_admission(worker,admission)
    return worker,admission
