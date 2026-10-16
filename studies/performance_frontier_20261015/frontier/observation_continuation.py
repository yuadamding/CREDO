"""Explicit remaining-work continuation after the qualified retention recovery.

This is a distinct entry point. Ordinary same-source qualification and resume
remain unchanged. It imports the original 1280-update checkpoint into a fresh
mixed-lineage child and permits only its remaining 18720 updates. No scheduler,
automatic retry, source selection, or additional fit is implemented here.
"""
import argparse
import ast
import importlib
import hashlib
import json
import os
from pathlib import Path
import stat

from . import observation_worker as worker
from .checkpoints import TrajectoryLease
from .native_inputs import FoldInputs, NativeInputIndex
from .observation import ObservationFold
from .observation_training import ObservationRows, ObservationTrainingSpec
from .observation_recovery import import_parent, CONSUMED, MAXIMUM


SCHEMA = 'credo.frontier.observation-continuation-contract.v1'
PAIRS = ('D1D2', 'D1D3', 'D1D4', 'D2D3')
NUMERICAL = tuple('frontier/'+name for name in (
    '__init__.py', 'checkpoints.py', 'native_inputs.py', 'observation.py', 'observation_training.py'))
EXTRA = ('frontier/observation_continuation.py', 'frontier/observation_recovery.py')
COMBINED_CAP = 4*1024**3


def read_reference(reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'size_bytes'}:
        raise ValueError('Exact path/hash/size evidence reference required')
    raw = worker._read(reference['path'], reference['sha256'])
    if type(reference['size_bytes']) is not int or len(raw) != reference['size_bytes']:
        raise ValueError('Referenced evidence size differs')
    return json.loads(raw)


def require(value, expected, label):
    if any(key not in value or type(value[key]) is not type(item) or value[key] != item
           for key, item in expected.items()):
        raise ValueError(label+' differs from the authorized continuation')


def _worker_change_scope(old_source, current_source):
    """Reject incidental worker changes outside the reviewed lifecycle bridge."""
    old, current = ast.parse(old_source), ast.parse(current_source)
    def without_core(tree):
        return ast.dump(ast.Module(body=[n for n in tree.body
            if not isinstance(n, ast.FunctionDef) or n.name != 'run_fitting_core'], type_ignores=[]))
    if without_core(old) != without_core(current):
        raise ValueError('Observation worker changed outside the explicit continuation bridge')
    def core(tree):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_fitting_core')
    protected_calls = {'ObservationFitter', 'ObservationFitter.resume', 'owner.run_until',
                       'owner.finalize_selected', 'bind_policy', 'prepare_retirement', 'commit_retirement'}
    def calls(tree):
        return [ast.dump(n) for n in ast.walk(core(tree)) if isinstance(n, ast.Call)
                and ast.unparse(n.func) in protected_calls]
    if calls(old) != calls(current):
        raise ValueError('Scientific fitter/retention operations changed during source continuation')
    def schedule(tree):
        return [ast.dump(n) for n in ast.walk(core(tree)) if isinstance(n, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in ('milestones', 'reserve')
                        for target in n.targets)]
    if schedule(old) != schedule(current):
        raise ValueError('Scientific schedule or checkpoint reserve changed')


def verify_parent_inventory(recovery, parent_root):
    intake = read_reference(recovery['parent_inventory'])
    require(intake, dict(schema='credo.frontier.observation-continuation-intake.v1', status='passed',
                        authority_sha256=recovery['authority']['sha256']), 'Parent inventory intake')
    row = intake['fits'][recovery['pair']]
    require(row, dict(old_job_absent=True, new_job_absent=True, owned_pods_absent=True,
                      new_roots_absent=True, parent_output_bytes=recovery['parent_retained_output_bytes']),
            'Parent closure/intake')
    recorded = row['inventory']
    raw = (json.dumps(recorded, sort_keys=True, separators=(',', ':'))+'\n').encode()
    if hashlib.sha256(raw).hexdigest() != row['inventory_sha256']:
        raise ValueError('Parent inventory digest differs')
    if ({k:row['parent_input_lease'][k] for k in ('path','sha256','size_bytes')} != recovery['parent_input_lease']
            or read_reference(recovery['parent_input_lease']) != row['parent_input_lease']['value']):
        raise ValueError('Parent inventory and original input lease differ')
    terminal, cleanup = row['terminal']['value'], row['cleanup']['value']
    if terminal['status'] != 'failed' or cleanup['status'] != 'complete' or terminal['uid'] != cleanup['uid']:
        raise ValueError('Parent terminal/cleanup identity differs')
    fields = ('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns')
    def scan():
        result = {}
        worker._path(parent_root)
        for directory, folders, files in os.walk(parent_root, followlinks=False):
            for name in folders+files:
                path = worker._path(Path(directory)/name)
                info = path.lstat()
                if info.st_uid != os.geteuid() or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    raise ValueError('Owned regular parent inventory required')
                if stat.S_ISREG(info.st_mode):
                    if info.st_nlink != 1 or len(result) >= 20000:
                        raise ValueError('Parent inventory link/count limit differs')
                    result[path.relative_to(parent_root).as_posix()] = {key:getattr(info,key) for key in fields}
        return result
    first = scan()
    # Device/inode numbers are worker-local across GPFS/NFS access paths. Keep
    # full identity stability within this worker; compare portable metadata to
    # the controller's retained inventory without pretending identical devices.
    portable = lambda rows:{name:{key:value for key,value in info.items() if key not in ('st_dev','st_ino')}
                           for name,info in rows.items()}
    if portable(first) != portable(recorded) or scan() != first:
        raise ValueError('Parent inventory changed since authorized intake')
    if sum(info['st_size'] for info in first.values()) != recovery['parent_retained_output_bytes']:
        raise ValueError('Parent byte accounting differs from inventory')


def load_recovery(reference, contract, hardware):
    recovery = read_reference(reference)
    required = {'schema', 'mode', 'fit_contract', 'authority', 'pair', 'parent_intake', 'parent_inventory',
                'parent_fit_contract', 'parent_source_manifest', 'parent_input_lease',
                'recovery_check_worker_contract', 'recovery_check_completion', 'recovery_check_restore',
                'recovery_check_readback', 'recovery_check_terminal', 'parent_retained_output_bytes',
                'combined_output_limit_bytes', 'consumed_updates', 'maximum_new_updates',
                'maximum_total_updates', 'extra_sources'}
    if set(recovery) != required or recovery['pair'] not in PAIRS:
        raise ValueError('Exact source-continuation contract fields and approved pair required')
    require(recovery, dict(schema=SCHEMA, mode='explicit_source_change_continuation',
        combined_output_limit_bytes=COMBINED_CAP, consumed_updates=CONSUMED,
        maximum_new_updates=MAXIMUM-CONSUMED, maximum_total_updates=MAXIMUM), 'Workload')
    if read_reference(recovery['fit_contract']) != contract:
        raise ValueError('Continuation and actual fitting contract differ')
    authority = read_reference(recovery['authority'])
    require(authority, dict(schema='credo.frontier.observation-continuation-authority.v1',
        status='approved', pairs=list(PAIRS), consumed_updates_each=CONSUMED,
        maximum_new_updates_each=MAXIMUM-CONSUMED, maximum_total_updates_each=MAXIMUM,
        combined_output_limit_bytes_each=COMBINED_CAP, maximum_concurrent_h100=4,
        automatic_retries=0, elapsed_time_deadline_seconds=None, original_workload_expanded=False,
        main_model_fitting=False), 'Authority')
    original = read_reference(recovery['parent_fit_contract'])
    for name in ('metadata_root', 'dataset_root', 'fold_id', 'training_donors', 'metadata_index_sha256',
                 'rna_axis_sha256', 'input_access_sha256', 'protocol_sha256', 'runtime_image',
                 'qualification_receipt', 'qualification_sha256'):
        if contract.get(name) != original.get(name):
            raise ValueError('Source-continuation scientific contract changed: '+name)
    if (original.get('mode') != 'fit' or original.get('output_limit_bytes') != COMBINED_CAP
            or ''.join(original['training_donors']) != recovery['pair']):
        raise ValueError('Parent is not the approved original observation fit')
    manifest_ref = recovery['parent_source_manifest']
    if (manifest_ref['path'], manifest_ref['sha256']) != (original['source_manifest'], original['source_manifest_sha256']):
        raise ValueError('Original source manifest binding differs')
    prior_sources = read_reference(manifest_ref)
    current_sources = worker.verify_sources(contract)
    if (prior_sources.get('schema') != 'credo.frontier.observation-worker-source.v1'
            or set(prior_sources.get('files', {})) != set(worker.SOURCE_MEMBERS)):
        raise ValueError('Exact original seven-file observation source manifest required')
    for name in NUMERICAL:
        if current_sources['files'][name] != prior_sources['files'][name]:
            raise ValueError('Numerical observation source changed: '+name)
    old_worker = worker._read(Path(original['source_root'])/'frontier/observation_worker.py',
                              prior_sources['files']['frontier/observation_worker.py'])
    new_worker = worker._read(Path(contract['source_root'])/'frontier/observation_worker.py',
                              current_sources['files']['frontier/observation_worker.py'])
    _worker_change_scope(old_worker, new_worker)
    if set(recovery['extra_sources']) != set(EXTRA):
        raise ValueError('Exact continuation/importer source bindings required')
    for name, sha in recovery['extra_sources'].items():
        worker._read(Path(contract['source_root'])/name, sha)
        imported = importlib.import_module(name[:-3].replace('/', '.'))
        if Path(imported.__file__).absolute() != Path(contract['source_root'])/name:
            raise ValueError('Continuation helper is not the bound imported source')

    intake = read_reference(recovery['parent_intake'])
    parent = intake['fits'][recovery['pair']]
    parent_root = Path(original['output_root'])
    if (Path(parent['identity']['path']).parent != parent_root
            or parent['identity']['value']['source_sha256'] != original['source_manifest_sha256']
            or parent['identity']['value']['worker_contract_sha256'] != recovery['parent_fit_contract']['sha256']):
        raise ValueError('Original parent checkpoint lineage differs')
    expected_input = parent_root/'INPUT_LEASE.json'
    if recovery['parent_input_lease']['path'] != str(expected_input):
        raise ValueError('Original input lease path differs')
    verify_parent_inventory(recovery, parent_root)

    checked_worker = read_reference(recovery['recovery_check_worker_contract'])
    check_sources = read_reference(checked_worker['source_manifest'])
    complete = read_reference(recovery['recovery_check_completion'])
    restore = read_reference(recovery['recovery_check_restore'])
    readback = read_reference(recovery['recovery_check_readback'])
    terminal = read_reference(recovery['recovery_check_terminal'])
    require(complete, dict(schema='credo.frontier.retention-check-completion.v1', status='passed',
        run_id=checked_worker['run_id'], contract=recovery['recovery_check_worker_contract'],
        source_manifest_sha256=checked_worker['source_manifest']['sha256'],
        parent_intake_sha256=recovery['parent_intake']['sha256'], optimizer_calls=0, scientific_updates=0,
        canary_passed=True, parent_writes=False, training_continuation_qualified=False), 'Recovery check')
    if (checked_worker['runtime_image'] != contract['runtime_image'] or checked_worker['expected_runtime'] != worker._runtime()
            or checked_worker['parent_intake']['sha256'] != recovery['parent_intake']['sha256']):
        raise ValueError('Qualified recovery runtime/intake differs')
    for name in NUMERICAL+('frontier/retention.py',):
        if check_sources['files'][name]['sha256'] != current_sources['files'][name]:
            raise ValueError('Recovered numerical/retention source differs: '+name)
    if (check_sources['files']['frontier/observation_recovery.py']['sha256'] != recovery['extra_sources']['frontier/observation_recovery.py']
            or check_sources['files']['frontier/observation_worker.py']['sha256'] != prior_sources['files']['frontier/observation_worker.py']):
        raise ValueError('Recovered importer/original worker differs')
    require(restore, dict(pair=recovery['pair'], status='exact_state_restoration_verified_on_cuda',
        parent_checkpoint=parent['checkpoint_payload'], consumed_updates=CONSUMED, remaining_update_cap=MAXIMUM-CONSUMED,
        optimizer_calls=0, scientific_updates=0, parent_changed=False, training_data_opened=False,
        model_optimizer_rng_sampler_and_audit_values_equal=True, training_continuation_qualified=False), 'Parent restore')
    if [row for row in complete['parents'] if row['pair'] == recovery['pair']] != [restore]:
        raise ValueError('Parent restore does not belong to completed recovery check')
    require(readback, dict(status='passed', run_id=checked_worker['run_id'],
        complete_sha256=recovery['recovery_check_completion']['sha256']), 'Recovery readback')
    require(terminal, dict(status='passed', application_pass=True, kubernetes_complete=True,
        run_id=checked_worker['run_id'], result_complete_sha256=recovery['recovery_check_completion']['sha256']), 'Recovery terminal')
    check_root = Path(checked_worker['output_root'])
    for name, relative in (('recovery_check_completion', 'COMPLETE.json'),
                           ('recovery_check_restore', recovery['pair']+'/RESTORE_VERIFIED.json')):
        reference_item = recovery[name]
        if (reference_item['path'] != str(check_root/relative)
                or readback['files'][relative] != {k:reference_item[k] for k in ('sha256', 'size_bytes')}):
            raise ValueError('Recovery check publication inventory differs')
    retained = recovery['parent_retained_output_bytes']
    if (type(retained) is not int or not 0 < retained < COMBINED_CAP
            or contract['output_limit_bytes'] != COMBINED_CAP-retained
            or worker._output_bytes(parent_root) != retained):
        raise ValueError('Parent plus child output reservation differs')
    qualification = worker.verify_qualification(original, hardware)
    compatibility = worker._qualification_hardware_compatibility(qualification, hardware)
    compatibility.update(schema='credo.frontier.observation-source-continuation-compatibility.v1',
        policy='unchanged_numerical_sources_plus_actual_mount_retirement_and_exact_cuda_restore_v1',
        original_qualification_sha256=original['qualification_sha256'],
        original_source_manifest_sha256=original['source_manifest_sha256'],
        continuation_source_manifest_sha256=contract['source_manifest_sha256'],
        recovery_check_sha256=recovery['recovery_check_completion']['sha256'],
        continuation_contract_sha256=reference['sha256'], numerical_sources_unchanged=list(NUMERICAL),
        mixed_source_continuation=True, fresh_numerical_qualification_performed=False)
    return recovery, parent, qualification, compatibility


def verify_input_snapshot(recovery, snapshot):
    original = read_reference(recovery['parent_input_lease'])
    if worker._input_binding(original) != worker._input_binding(snapshot):
        raise ValueError('Parent and continuation input hashes/sizes/roles differ')


def combined_capacity(root, contract, continuation, reserve=0):
    recovery = read_reference(continuation)
    original = read_reference(recovery['parent_fit_contract'])
    retained = worker._output_bytes(original['output_root'])
    if (retained != recovery['parent_retained_output_bytes']
            or contract['output_limit_bytes']+retained != recovery['combined_output_limit_bytes']):
        raise ValueError('Retained parent bytes or combined output cap changed')
    result = worker._capacity(root, contract['output_limit_bytes'], reserve)
    result.update(retained_parent_output_bytes=retained, combined_output_bytes=retained+result['output_bytes'],
                  combined_output_limit_bytes=recovery['combined_output_limit_bytes'])
    return result


def verify_for_execution(lease, source, spec, contract, qualification, resume, snapshot, hardware, continuation):
    recovery, parent, verified_qualification, compatibility = load_recovery(continuation, contract, hardware)
    if qualification != verified_qualification or spec != ObservationTrainingSpec() or source.synthetic_fixture:
        raise ValueError('Continuation requires the original qualified scientific procedure')
    expected_identity = worker._lease_identity(contract, spec, source.identity, contract['fold_id'],
                                               recovery['fit_contract']['sha256'])
    expected_identity.update(recovery_parent_checkpoint_sha256=parent['checkpoint_payload']['sha256'],
                             recovery_authority_sha256=recovery['authority']['sha256'])
    if lease.identity != expected_identity:
        raise ValueError('Current lease differs from the exact source-changing contract')
    verify_input_snapshot(recovery, snapshot)
    imported_path = lease.root/'IMPORT_COMPLETE.json'
    imported = json.loads(worker._read(imported_path, hashlib.sha256(imported_path.read_bytes()).hexdigest()))
    require(imported, dict(status='imported_not_numerically_resumed', parent_identity=parent['identity']['value'],
        child_identity=lease.identity, authority_sha256=recovery['authority']['sha256'],
        consumed_updates=CONSUMED, remaining_update_cap=MAXIMUM-CONSUMED, maximum_updates_total=MAXIMUM,
        parent_checkpoint=parent['checkpoint_payload'], parent_changed=False, scientific_optimizer_calls=0), 'Imported parent')
    if resume != dict(directory=imported['checkpoint_directory'], sha256=imported['checkpoint_sha256']):
        raise ValueError('Explicit resume is not the bound imported checkpoint')
    combined_capacity(lease.root, contract, continuation)
    return compatibility


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contract', 'contract-sha256', 'recovery-contract', 'recovery-contract-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    contract, spec = worker.load_contract(args.contract, args.contract_sha256, 'fit')
    recovery_raw = worker._read(args.recovery_contract, args.recovery_contract_sha256)
    continuation = dict(path=args.recovery_contract, sha256=args.recovery_contract_sha256, size_bytes=len(recovery_raw))
    hardware = worker._configure_cuda()
    recovery, parent, qualification, _ = load_recovery(continuation, contract, hardware)
    if (recovery['fit_contract']['path'], recovery['fit_contract']['sha256']) != (args.contract, args.contract_sha256):
        raise ValueError('CLI fitting contract differs from the explicit continuation binding')
    index = NativeInputIndex(contract['metadata_root'], contract['reader_source_root'])
    fold = FoldInputs(contract['fold_id'], tuple(contract['training_donors']))
    access = index.access(fold, 'representation_fit')
    if (index.identity != contract['metadata_index_sha256'] or index.rna_axis_sha256 != contract['rna_axis_sha256']
            or access.identity() != contract['input_access_sha256']):
        raise ValueError('Fold-native continuation input identities differ')
    handle = index.open(contract['dataset_root'], fold, 'representation_fit', max_batch_rows=2048)
    snapshot = handle.capture_publication_identity()
    verify_input_snapshot(recovery, snapshot)
    rows = ObservationRows(handle, ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256))
    identity = worker._lease_identity(contract, spec, rows.identity, fold.fold_id, args.contract_sha256)
    identity.update(recovery_parent_checkpoint_sha256=parent['checkpoint_payload']['sha256'],
                    recovery_authority_sha256=recovery['authority']['sha256'])
    with TrajectoryLease(contract['output_root'], identity, contract['worker_id']) as lease:
        try:
            combined_capacity(lease.root, contract, continuation, reserve=512*1024**2)
            imported = import_parent(lease, parent, authority_sha256=recovery['authority']['sha256'])
            resume = dict(directory=imported['checkpoint_directory'], sha256=imported['checkpoint_sha256'])
            result, receipt = worker.run_fitting_core(lease, rows, spec, contract, device='cuda:0',
                qualification=qualification, resume=resume, input_snapshot=snapshot, hardware=hardware,
                continuation=continuation)
        except BaseException as error:
            if not (lease.root/'WORKER_FAILED.json').exists():
                worker._fail(lease, contract, error, counters=dict(accepted_updates=CONSUMED,
                    attempted_updates=CONSUMED, optimizer_calls=CONSUMED))
            raise
    print(json.dumps(dict(status=result['status'], receipt=receipt)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
