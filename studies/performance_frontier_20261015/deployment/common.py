"""Small Python-3.6-compatible metadata helpers, not a scheduler implementation."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

STUDY = 'credo_performance_frontier_20261015_v1'
PROJECT = '/rsrch8/home/bcb/yding4/perturbseq/' + STUDY
CONTROL = '/rsrch8/scratch/bcb/yding4/perturbseq/' + STUDY
IMAGE = 'hpcharbor.mdanderson.edu/yding41/credo_env@sha256:138d7191792ce1139708751f109817a976c5b96ba3a00be3dff88173b62eb21a'
CORE_SHA = '3da1a0a5b0d0d26019918ee48e2ce24cd2e616272de7854f4c82ba2e37e7ebd2'
GIB = 1024 ** 3
VISIBLE_HOME = '/rsrch8/home/bcb/yding4/perturbseq'


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def identity(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def checked_path(value, exists=True):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Explicit absolute path required')
    for item in (path,) + tuple(path.parents):
        if item.is_symlink():
            raise ValueError('Symlink path forbidden')
    if exists and not path.exists():
        raise FileNotFoundError(str(path))
    return path


def read(path, expected=None, maximum=32 * 1024 ** 2):
    path = checked_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ValueError('Bounded regular metadata/source required')
    with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if signature(before) != signature(opened):
            raise ValueError('Metadata open race')
        raw = stream.read(maximum + 1)
        if signature(opened) != signature(os.fstat(stream.fileno())) or signature(before) != signature(path.lstat()):
            raise ValueError('Metadata changed while reading')
    if len(raw) > maximum or (expected is not None and identity(raw)['sha256'] != expected):
        raise ValueError('Metadata size/hash differs')
    return raw


def read_json(path, expected=None):
    return json.loads(read(path, expected))


def publish(path, value):
    path = checked_path(path, exists=False)
    raw = encoded(value)
    temporary = path.parent / ('.publishing-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(str(temporary), str(path), follow_symlinks=False)
    finally:
        temporary.unlink()
    descriptor = os.open(str(path.parent), os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if read(path, identity(raw)['sha256']) != raw:
        raise ValueError('Publication readback differs')
    return identity(raw)


def validate_plan(plan, bundle=None):
    required = dict(schema='credo.frontier.observation-lifecycle.v1', image=IMAGE,
                    authorized_CPUs=24, authorized_memory_GiB=200, threads=8,
                    gpu_ceiling=1, aggregate_h100_ceiling=4, automatic_retries=0,
                    automatic_successor=False, elapsed_time_deadline_seconds=None,
                    runtime_deadline_policy='no_runtime_deadline_user_20260919',
                    startup_gate='inline_h100_controller_release_v1', LARRY_access=False)
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError('Observation lifecycle contract differs')
    if plan.get('mode') not in ('qualify', 'fit') or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', plan.get('run_id', '')):
        raise ValueError('Exact scalar observation identity required')
    phase = plan.get('phase_directory', '')
    if not re.fullmatch(r'observation_[a-z0-9_]+_v[0-9]+', phase):
        raise ValueError('New observation phase directory required')
    expected = Path(PROJECT) / phase
    paths = dict(bundle_root=str(expected / 'bundle'), worker_output=str(expected / 'handshake'),
                 scientific_output=str(expected / 'output'), control_root=str(Path(CONTROL) / phase))
    if any(plan.get(key) != value for key, value in paths.items()):
        raise ValueError('Exact new-study phase paths required')
    if bundle is not None and str(bundle) != plan['bundle_root']:
        raise ValueError('Loaded bundle differs')
    recovery = plan.get('recovery_contract')
    cap = 16 * GIB if plan['mode'] == 'qualify' else 4 * GIB
    if recovery is not None:
        retained = plan.get('parent_retained_output_bytes')
        if (plan['mode'] != 'fit' or type(retained) is not int or not 0 < retained < 2 * GIB
                or plan.get('combined_output_limit_bytes') != 4 * GIB
                or not isinstance(recovery, dict)
                or recovery.get('path') != str(expected / 'bundle' / 'CONTINUATION_CONTRACT.json')
                or not re.fullmatch('[0-9a-f]{64}', recovery.get('sha256', ''))):
            raise ValueError('Explicit bounded continuation contract required')
        cap -= retained
    if (plan.get('maximum_output_bytes') != cap
            or plan.get('maximum_disposable_optimizer_steps') != (16 if plan['mode'] == 'qualify' else 0)
            or plan.get('planned_disposable_optimizer_steps') != (13 if plan['mode'] == 'qualify' else 0)):
        raise ValueError('Bound observation phase resource limit differs')
    updates = 18720 if recovery is not None else (0 if plan['mode'] == 'qualify' else 20000)
    if plan.get('maximum_new_scientific_updates') != updates:
        raise ValueError('Observation update limit differs')
    return plan


def validate_authority(authority):
    expected_authority = dict(schema='credo.frontier.observation-phase-authority.v1', study_id=STUDY,
                             maximum_concurrent_h100=4, gpus_per_worker=1, cpus_per_worker=24,
                             host_memory_gib_per_worker=200, application_threads=8, runtime_image=IMAGE,
                             automatic_retries=0, elapsed_time_deadline=None, LARRY_access=False,
                             main_dynamics_training_authorized_by_this_phase=False)
    if (any(authority.get(key) != value for key, value in expected_authority.items())
            or authority['qualification']['planned_disposable_optimizer_calls'] != 13
            or authority['qualification']['maximum_disposable_optimizer_calls'] != 16
            or authority['qualification']['output_limit_bytes'] != 16 * GIB
            or authority['fitting']['output_limit_bytes_per_subset'] != 4 * GIB
            or authority['fitting']['maximum_accepted_updates_per_subset'] != 20000
            or authority['storage']['whole_study_live_output_cap_bytes'] != 384 * GIB
            or authority['storage']['additional_unallocated_fileset_reserve_bytes'] != 16 * GIB):
        raise ValueError('Phase authority scientific/resource scope differs')
    return authority


def validate_admission(plan, admission):
    if (admission.get('schema') != 'credo.frontier.observation-admission-export.v1'
            or admission.get('run_id') != plan['run_id'] or admission.get('status') != 'reserved'
            or admission.get('storage_reserved_bytes') != plan['maximum_output_bytes']
            or admission.get('scientific_update_reservation') != plan['maximum_new_scientific_updates']
            or type(admission.get('aggregate_reserved_output_bytes')) is not int
            or not plan['maximum_output_bytes'] <= admission['aggregate_reserved_output_bytes'] <= 384 * GIB
            or type(admission.get('live_h100_reservations')) is not int
            or not 1 <= admission['live_h100_reservations'] <= 4
            or not re.fullmatch('[0-9a-f]{64}', admission.get('journal_event_sha256', ''))):
        raise ValueError('Exact modern-Python journal reservation export required')
    return admission


def _inside(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def _dependency_reference(value, require_size=False):
    allowed = ({'path', 'sha256', 'size_bytes'},) if require_size else (
        {'path', 'sha256'}, {'path', 'sha256', 'size_bytes'})
    if (not isinstance(value, dict) or set(value) not in allowed
            or not isinstance(value.get('path'), str)
            or not isinstance(value.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', value['sha256']) is None
            or ('size_bytes' in value and (type(value['size_bytes']) is not int or value['size_bytes'] < 0))):
        raise ValueError('Exact bounded worker dependency reference required')
    return value


def validate_replacement(plan, admission, worker):
    ref = _dependency_reference(plan['replacement_authority'], require_size=True)
    if (ref['path'] != str(Path(plan['bundle_root'])/'REPLACEMENT_AUTHORITY.json')
            or ref['sha256'] != admission.get('replacement_authority_sha256')):
        raise ValueError('Replacement authority/admission binding differs')
    raw = read(ref['path'], ref['sha256'])
    if len(raw) != ref['size_bytes']:
        raise ValueError('Replacement authority size differs')
    authority = json.loads(raw)
    expected = dict(schema='credo.frontier.observation-continuation-replacement-authority.v1',
        status='approved', user_instruction='yes', study_id=STUDY,
        pairs=['D1D2','D1D3','D1D4','D2D3'], failed_generation='v3', replacement_generation='v4',
        per_pair_explicit_replacement_attempts=1,
        source_continuation_authority='OBSERVATION_CONTINUATION_V3_AUTHORITY.json',
        consumed_updates_each=1280, maximum_new_updates_each=18720, maximum_total_updates_each=20000,
        consumed_updates_all_four=5120, maximum_new_updates_all_four=74880,
        required_failed_child_new_updates=0, required_failed_child_scientific_output='absent',
        combined_output_limit_bytes_each=4*GIB, maximum_concurrent_h100=4,
        resources_each=dict(H100=1, CPUs=24, host_memory_GiB=200, application_threads=8),
        automatic_retries=0, automatic_successor=False, elapsed_time_deadline_seconds=None,
        original_workload_expanded=False, main_model_fitting=False,
        historical_parents_and_failed_children_read_only=True, numerical_worker_sources_changed=False,
        local_optimizer_calls_authorized=0, LARRY_access=False)
    if any(type(authority.get(k)) is not type(v) or authority.get(k) != v for k,v in expected.items()):
        raise ValueError('Exact v4 replacement authority scope differs')
    pair = ''.join(worker['training_donors'])
    if (pair not in authority['pairs'] or plan.get('mode') != 'fit'
            or plan['run_id'] != 'credo-frontier-obs-'+pair.lower()+'-20261016-v4'
            or admission.get('replaces_continuation_task_id') != 'credo-frontier-obs-'+pair.lower()+'-20261016-v3'):
        raise ValueError('Exact four-pair v3-to-v4 replacement required')
    intake_ref = _dependency_reference(plan['replacement_intake'], require_size=True)
    if (intake_ref['path'] != str(Path(plan['bundle_root'])/'REPLACEMENT_INTAKE.json')
            or intake_ref['sha256'] != admission.get('replacement_intake_sha256')):
        raise ValueError('Replacement intake/admission binding differs')
    intake_raw = read(intake_ref['path'],intake_ref['sha256'])
    if len(intake_raw) != intake_ref['size_bytes']:
        raise ValueError('Replacement intake size differs')
    intake = json.loads(intake_raw)
    if (intake.get('schema') != 'credo.frontier.observation-continuation-replacement-intake.v1'
            or intake.get('status') != 'passed' or intake.get('authority_sha256') != ref['sha256']
            or intake.get('prior_failure_closure_sha256') != admission.get('zero_update_evidence_sha256')):
        raise ValueError('Replacement closure authority/evidence differs')
    return authority


def verify_worker_read_dependencies(plan, worker):
    """Read/hash small worker dependencies before release, never real payloads.

    Invoked by verify_bundle on both controller and actual worker. The manifest
    is not a promise that raw arrays/checkpoints were consumed or qualified.
    Their references, roots, locks and expected absences are lexical checks;
    existing scientific leases retain their actual payload validation duties.
    """
    ref = _dependency_reference(plan['worker_read_dependencies'], require_size=True)
    bundle = Path(plan['bundle_root'])
    if ref['path'] != str(bundle/'WORKER_READ_DEPENDENCIES.json'):
        raise ValueError('Exact bundle worker dependency manifest required')
    raw = read(ref['path'], ref['sha256'])
    if len(raw) != ref['size_bytes']:
        raise ValueError('Worker dependency manifest size differs')
    inventory = read_json(bundle/'INVENTORY.json')
    if inventory.get('WORKER_READ_DEPENDENCIES.json') != identity(raw):
        raise ValueError('Bundle inventory omits or changes worker dependency manifest')
    closure = json.loads(raw)
    required = {'schema','policy','run_id','worker_visible_roots','metadata_references',
        'checkpoint_payload_references','input_payload_references','stat_and_read_roots',
        'lock_paths','expected_absent_paths','provenance_only_origins',
        'runtime_image_libraries_outside_study_closure',
        'eager_checkpoint_or_native_payload_hashing_authorized','optimizer_calls'}
    expected = dict(schema='credo.frontier.worker-read-dependency-closure.v1',
        policy='home_visible_metadata_preflight_payload_verification_on_existing_consumption_path',
        run_id=plan['run_id'],runtime_image_libraries_outside_study_closure=True,
        eager_checkpoint_or_native_payload_hashing_authorized=False,optimizer_calls=0)
    roots = list(dict.fromkeys((PROJECT,str(Path(VISIBLE_HOME)/STUDY),worker['dataset_root'])))
    if (not _inside(worker['dataset_root'],Path(VISIBLE_HOME)/'inputs')
            or set(closure) != required or any(type(closure.get(k)) is not type(v) or closure.get(k) != v
            for k,v in expected.items()) or closure['worker_visible_roots'] != roots):
        raise ValueError('Worker dependency closure scope differs')
    def visible(value):
        if not isinstance(value,str):
            raise ValueError('Explicit worker-visible path required')
        path = Path(value)
        if (not path.is_absolute() or '..' in path.parts or path.as_posix() != value
                or not any(_inside(path,root) for root in roots)):
            raise ValueError('Worker dependency outside visible home surfaces')
        return path
    metadata = closure['metadata_references']
    if not isinstance(metadata,dict) or not 0 < len(metadata) <= 4096:
        raise ValueError('Bounded nonempty worker metadata closure required')
    for table_name in ('metadata_references','checkpoint_payload_references','input_payload_references'):
        table = closure[table_name]
        if not isinstance(table,dict) or not table or len(table) > 20000:
            raise ValueError('Bounded declared dependency table required')
        for label,item in table.items():
            if not isinstance(label,str) or not label:
                raise ValueError('Named worker dependency required')
            visible(_dependency_reference(item)['path'])
    for name in ('stat_and_read_roots','lock_paths','expected_absent_paths'):
        table = closure[name]
        if not isinstance(table,dict) or not table or len(table)>128:
            raise ValueError('Bounded lexical worker path table required')
        for path in table.values():
            visible(path)
    # Ensure directly opened contracts cannot simply be left off the manifest.
    for key in ('worker_contract','source_manifest','protocol','phase_authority','parent_authority',
                'admission','recovery_contract','replacement_authority','replacement_intake','recovery_metadata_origins'):
        if metadata.get('plan.'+key) != plan[key]:
            raise ValueError('Worker metadata closure omits plan dependency: '+key)
    # Deduplicate repeated references while retaining exact size consistency.
    contents, seen, total = {}, {}, 0
    for label,item in metadata.items():
        path = item['path']
        if path in seen:
            previous = seen[path]
            if item['sha256'] != previous['sha256']:
                raise ValueError('Conflicting metadata hashes for one dependency')
            data = contents[path]
        else:
            data = read(path,item['sha256'])
            total += len(data)
            if total > 128*1024**2:
                raise ValueError('Worker metadata read allowance exceeded')
            contents[path],seen[path] = data,item
        if 'size_bytes' in item and len(data) != item['size_bytes']:
            raise ValueError('Worker metadata dependency size differs')
        if _inside(path,bundle):
            name = Path(path).relative_to(bundle).as_posix()
            if inventory.get(name) != identity(data):
                raise ValueError('Worker dependency differs from bundle inventory')
    recovery = json.loads(contents[plan['recovery_contract']['path']])
    for label,item in recovery.items():
        if isinstance(item,dict) and 'path' in item and metadata.get('recovery.'+label) != item:
            raise ValueError('Worker metadata closure omits recovery dependency: '+label)
    for name in ('study_root','source_root','reader_source_root','metadata_root','dataset_root','output_root'):
        if closure['stat_and_read_roots'].get('worker.'+name) != worker[name]:
            raise ValueError('Worker dependency root differs: '+name)
    # Origins deliberately retain controller-scratch paths. Only their exact
    # copies are runtime read dependencies; the origins themselves are not read.
    origins = json.loads(contents[plan['recovery_metadata_origins']['path']])
    if closure['provenance_only_origins'] != origins.get('copies'):
        raise ValueError('Recovery metadata origin/copy custody differs')
    for item in closure['provenance_only_origins'].values():
        origin = _dependency_reference(item['origin'], require_size=True)
        copy = _dependency_reference(item['worker_visible_copy'], require_size=True)
        visible(copy['path'])
        if (copy['path'] not in contents or identity(contents[copy['path']]) !=
                {k:origin[k] for k in ('sha256','size_bytes')}
                or {k:copy[k] for k in ('sha256','size_bytes')} != identity(contents[copy['path']])):
            raise ValueError('Worker-visible receipt is not byte-identical to its origin')
    return closure


def verify_bundle(plan, bundle):
    validate_plan(plan, bundle)
    for name, digest in plan['controller_sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe controller source member')
        read(Path(bundle) / name, digest)
    if plan['controller_sources'].get('lifecycle_core.py') != CORE_SHA:
        raise ValueError('Pinned lifecycle core differs')
    for field in ('worker_contract', 'source_manifest', 'protocol', 'phase_authority', 'parent_authority', 'admission'):
        ref = plan[field]
        read(ref['path'], ref['sha256'])
    validate_authority(read_json(plan['phase_authority']['path'], plan['phase_authority']['sha256']))
    worker = read_json(plan['worker_contract']['path'], plan['worker_contract']['sha256'])
    if (worker.get('mode') != plan['mode'] or worker.get('worker_id') != plan['run_id']
            or worker.get('output_root') != plan['scientific_output'] or worker.get('runtime_image') != IMAGE
            or worker.get('output_limit_bytes') != plan['maximum_output_bytes']):
        raise ValueError('Worker/lifecycle binding differs')
    source = read_json(plan['source_manifest']['path'], plan['source_manifest']['sha256'])
    if (worker['source_manifest'] != plan['source_manifest']['path']
            or worker['source_manifest_sha256'] != plan['source_manifest']['sha256']
            or worker['protocol_path'] != plan['protocol']['path']
            or worker['protocol_sha256'] != plan['protocol']['sha256']):
        raise ValueError('Worker source/protocol binding differs')
    for name, digest in source['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe worker source member')
        read(Path(worker['source_root']) / name, digest)
    admission = read_json(plan['admission']['path'], plan['admission']['sha256'])
    validate_admission(plan, admission)
    if plan.get('recovery_contract') is not None:
        ref = plan['recovery_contract']
        recovery = read_json(ref['path'], ref['sha256'])
        expected = dict(schema='credo.frontier.observation-continuation-contract.v1',
                        mode='explicit_source_change_continuation', consumed_updates=1280,
                        maximum_new_updates=18720, maximum_total_updates=20000,
                        combined_output_limit_bytes=4 * GIB,
                        parent_retained_output_bytes=plan['parent_retained_output_bytes'])
        if any(recovery.get(k) != v for k, v in expected.items()):
            raise ValueError('Continuation scientific/accounting scope differs')
        if (recovery['fit_contract']['path'] != plan['worker_contract']['path']
                or recovery['fit_contract']['sha256'] != plan['worker_contract']['sha256']
                or recovery['authority']['sha256'] != admission.get('continuation_authority_sha256')
                or recovery['pair'] != ''.join(worker['training_donors'])):
            raise ValueError('Continuation worker/admission binding differs')
        extras = recovery.get('extra_sources', {})
        if set(extras) != {'frontier/observation_continuation.py', 'frontier/observation_recovery.py'}:
            raise ValueError('Exact continuation source closure required')
        for name, digest in extras.items():
            read(Path(worker['source_root']) / name, digest)
        authority = read_json(recovery['authority']['path'], recovery['authority']['sha256'])
        if (authority.get('schema') != 'credo.frontier.observation-continuation-authority.v1'
                or authority.get('status') != 'approved' or recovery['pair'] not in authority.get('pairs', [])
                or authority.get('maximum_new_updates_each') != 18720
                or authority.get('original_workload_expanded') is not False):
            raise ValueError('Explicit continuation authority missing')
    if 'replacement_authority' in plan or 'worker_read_dependencies' in plan:
        if not all(name in plan for name in ('replacement_authority','replacement_intake','worker_read_dependencies','recovery_metadata_origins','recovery_contract')):
            raise ValueError('Complete replacement authority and worker dependency gate required')
        validate_replacement(plan, admission, worker)
        verify_worker_read_dependencies(plan, worker)
    return worker, admission
