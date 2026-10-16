"""Main-qualification lifecycle metadata only; compatible with host Python 3.6.

No imports of numerical modules, scheduler calls, or authority construction.
The existing observation adapter and its active bundles remain unchanged.
"""
import json
from pathlib import Path
import re

try:
    from .common import CONTROL, CORE_SHA, GIB, IMAGE, PROJECT, encoded, identity, read, read_json
except ImportError:  # Standalone immutable deployment bundle.
    from common import CONTROL, CORE_SHA, GIB, IMAGE, PROJECT, encoded, identity, read, read_json


def same(value, expected):
    if type(value) is not dict or any(k not in value or encoded(value[k]) != encoded(v)
                                       for k, v in expected.items()):
        raise ValueError('Main qualification metadata identity differs')


def semantic_sha(value):
    # Numerical identity deliberately omits the serialization newline. File and
    # admitted-Pod hashes retain it; these are different, named identities.
    return identity(encoded(value)[:-1])['sha256']


def digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Exact lowercase SHA-256 required')
    return value


def reference(path, raw):
    return dict(path=str(path), **identity(raw))


def read_reference(ref):
    if type(ref) is not dict or set(ref) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('Exact path/hash/size reference required')
    digest(ref['sha256'])
    if type(ref['size_bytes']) is not int or not 0 < ref['size_bytes'] <= 4 * 1024**2:
        raise ValueError('Bounded nonempty metadata reference required')
    raw = read(ref['path'], ref['sha256'], maximum=4 * 1024**2)
    if len(raw) != ref['size_bytes']:
        raise ValueError('Metadata size differs')
    return raw


def validate_plan(plan, bundle=None):
    same(plan, dict(schema='credo.frontier.main-qualification-lifecycle.v1',
        mode='main-qualification', image=IMAGE, authorized_CPUs=24,
        authorized_memory_GiB=200, threads=8, gpu_ceiling=1, aggregate_h100_ceiling=4,
        automatic_retries=0, automatic_successor=False, elapsed_time_deadline_seconds=None,
        runtime_deadline_policy='no_runtime_deadline_user_20260919',
        startup_gate='inline_h100_controller_release_v1', maximum_output_bytes=16*GIB,
        maximum_disposable_optimizer_steps=112, planned_disposable_optimizer_steps=111,
        maximum_new_scientific_updates=0, real_biological_payload_access=False, LARRY_access=False))
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', plan.get('run_id', '')):
        raise ValueError('Exact scalar qualification Job name required')
    phase = plan.get('phase_directory', '')
    if not re.fullmatch(r'main_qualification_v[1-9][0-9]*', phase):
        raise ValueError('Distinct main-qualification phase required')
    base = Path(PROJECT)/phase
    same(plan, dict(bundle_root=str(base/'bundle'), worker_output=str(base/'handshake'),
                    scientific_output=str(base/'output'), control_root=str(Path(CONTROL)/phase)))
    if bundle is not None and str(bundle) != plan['bundle_root']:
        raise ValueError('Loaded bundle path differs')
    return plan


def validate_authority(worker, authority):
    same(authority, dict(schema='credo.frontier.main-qualification-authority.v1', status='authorized',
        run_id=worker['run_id'], plan_sha256=worker['plan_sha256'],
        settings_sha256=semantic_sha(worker['settings']),
        source_manifest_sha256=worker['source_manifest']['sha256'],
        worker_source_manifest_sha256=worker['worker_source_manifest']['sha256'],
        planned_disposable_optimizer_calls=111, maximum_disposable_optimizer_calls=112,
        scientific_updates=0, real_biological_payload_access=False, main_scientific_fitting=False,
        automatic_retry=False, automatic_successor=False, elapsed_time_deadline=None))


def validate_admission(worker, admission):
    same(admission, dict(schema='credo.frontier.main-qualification-admission.v1', status='reserved',
        run_id=worker['run_id'], work_binding=dict(kind='qualification',
            identity=[worker['plan_sha256']], start_updates=0, target_updates=0),
        phase_authority_sha256=worker['authority']['sha256'],
        source_manifest_sha256=worker['source_manifest']['sha256'],
        worker_source_manifest_sha256=worker['worker_source_manifest']['sha256'],
        scientific_update_reservation=0, storage_reserved_bytes=16*GIB,
        disposable_optimizer_call_limit=112, automatic_retry=False, automatic_successor=False))
    for key in ('journal_event_sha256', 'journal_identity_sha256', 'capacity_receipt_sha256'):
        digest(admission.get(key))
    for key, minimum, maximum in (('live_h100_reservations', 1, 4),
                                  ('aggregate_reserved_output_bytes', 16*GIB, 384*GIB)):
        if type(admission.get(key)) is not int or not minimum <= admission[key] <= maximum:
            raise ValueError('Aggregate resource reservation exceeds bound')


def verify_bundle(plan, bundle):
    validate_plan(plan, bundle)
    bundle = Path(bundle)
    if plan['controller_sources'].get('lifecycle_core.py') != CORE_SHA:
        raise ValueError('Pinned scheduler lifecycle differs')
    for name, sha in plan['controller_sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe controller member')
        read(bundle/name, digest(sha))
    worker = json.loads(read_reference(plan['worker_contract']))
    same(worker, dict(schema='credo.frontier.main-qualification-contract.v1',
        run_id=plan['run_id'], phase=plan['phase_directory'], study_root=PROJECT,
        source_root=str(bundle/'source'), reader_source_root=str(bundle/'retained_reader'),
        output_root=plan['scientific_output'], runtime_image=IMAGE))
    for key in ('source_manifest', 'worker_source_manifest', 'observation_protocol',
                'normalization_protocol', 'rules', 'authority', 'admission'):
        if Path(worker[key]['path']).parent != bundle:
            raise ValueError('Numerical metadata must be bound inside this bundle')
        read_reference(worker[key])
    for plan_key, worker_key in (('source_manifest', 'source_manifest'),
                                  ('phase_authority', 'authority'), ('admission', 'admission')):
        if plan[plan_key] != worker[worker_key]:
            raise ValueError('Lifecycle/numerical metadata binding differs')
    if (worker['plan_sha256'] != semantic_sha(read_json(bundle/'QUALIFICATION_PLAN.json'))
            or worker['settings'] != read_json(bundle/'QUALIFICATION_SETTINGS.json')):
        raise ValueError('Numerical workload differs from packaged plan/settings')
    sources = json.loads(read_reference(worker['source_manifest']))
    if sources.get('entry') != 'frontier/main_qualification_worker.py':
        raise ValueError('Distinct main qualification entry required')
    for name, ref in sources['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe numerical source member')
        raw = read(bundle/'source'/name, ref['sha256'])
        if len(raw) != ref['size_bytes']:
            raise ValueError('Numerical source size differs')
    authority = json.loads(read_reference(worker['authority']))
    admission = json.loads(read_reference(worker['admission']))
    validate_authority(worker, authority)
    validate_admission(worker, admission)
    return worker, admission


def validate_resources(value):
    same(value, dict(cpu_limit=24, memory_limit_bytes=200*GIB, application_threads=8))
    if value.get('cgroup_version') not in (1, 2):
        raise ValueError('Measured cgroup version required')
    if type(value.get('affinity_cpus')) is not int or value['affinity_cpus'] < 24:
        raise ValueError('Insufficient actual CPU affinity')


def validate_gpu(value):
    same(value, dict(count=1, name='NVIDIA H100 80GB HBM3', compute_capability=[9, 0]))
    if type(value.get('total_memory_bytes')) is not int or not 75*GIB <= value['total_memory_bytes'] <= 82*GIB:
        raise ValueError('Full H100 memory range differs')
