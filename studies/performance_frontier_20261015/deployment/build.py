"""Deterministic allowlisted observation bundle builder; no remote operations.

Run from the study root with ``python -m deployment.build --help``. A real
journal reservation export is required; this builder never fabricates one.
"""
import argparse
from copy import deepcopy
import gzip
import io
import json
import os
from pathlib import Path
import re
import tarfile

from .common import (CONTROL, CORE_SHA, GIB, IMAGE, PROJECT, STUDY, encoded, identity,
                     read, read_json, validate_admission, validate_authority, validate_plan)

SOURCE_MEMBERS = tuple('frontier/' + name for name in (
    '__init__.py', 'checkpoints.py', 'native_inputs.py', 'observation.py',
    'observation_training.py', 'retention.py', 'observation_worker.py'))
ADAPTER_MEMBERS = ('common.py', 'capacity.py', 'control.py', 'worker_entry.py')
PINNED = {
    'lifecycle_core.py': CORE_SHA,
    'runtime_policy.py': 'bbe70b8caf28f398fd939a887822df2462f5077b9cdc70ebc5a443a719485fa5',
    'gpu_v2_static.py': '0d4d597ba46815b3e146735e62d4383c004c558a057377563c70e9cd20d8c061',
    'STATIC_REUSE.json': '21eb4a54a1f3c1e905463c6f162d782773e7e18c0fe81976434d26cafea7ceda',
    'capacity_metadata.py': 'a847dc3a31716a68f333155a07ca128842d6e04dbd25c8833b1de6608e695bf1',
    'gpu-job.json': '73c62f3fd478403dc9539099f7ee4839ce07497e92a2c53851f7149a89c56775',
}
FIT_FIELDS = frozenset(('metadata_root', 'reader_source_root', 'dataset_root', 'fold_id', 'training_donors',
    'metadata_index_sha256', 'rna_axis_sha256', 'input_access_sha256', 'qualification_receipt', 'qualification_sha256'))
NATIVE_ROOT = '/rsrch8/home/bcb/yding4/perturbseq/inputs/GSE314342_count_native_v1'
READER_RELATIVE_ROOT = 'credo_full_cohort_20260912/r48_full_pipeline_20260914_a2/bundle/repo/src/credo_count_sde_v4'
READER_MEMBERS = {
    'errors.py': 'f7450783f31f429c68bb03c94fe65c476bbf459d7835c3eb829b310bdc191c53',
    'canonical.py': '835abb437c71b0859c7396f115f74a3296c77354b730f9d607953297909c6a8d',
    'contracts/models.py': 'cb1aae49413c74de4851c24c489f10ef22dae196665326123a90b1dadff4d235',
    'data/prepared_shards.py': 'a18dcd3ac918a434697cfaf4c5a0c82716d0e47b8f544139acdb4cc99f3180d8',
}
QUALIFIED_GPU_PRODUCT = 'NVIDIA-H100-80GB-HBM3'


def validate_qualified_hardware(value):
    """Validate, then retain the hardware copied from the verified qualification.

    This prospective restriction uses a product label already present in the
    institutional template. CUDA-reported usable memory can differ between
    devices of this same class; preserve the actual measured total but validate
    its full-device range rather than equating it to one earlier device's total.
    Labels do not replace the worker's compatibility and measured-capacity checks.
    """
    expected = dict(device='cuda:0', name='NVIDIA H100 80GB HBM3',
                    capability=[9, 0], visible_devices=1,
                    precision='FP32', tf32=False, deterministic_algorithms=True,
                    required_headroom_bytes=8 * GIB)
    if (not isinstance(value, dict) or set(value) != set(expected) | {'total_memory_bytes'}
            or any(type(value[key]) is not type(target) or value[key] != target
                   for key, target in expected.items())
            or type(value['total_memory_bytes']) is not int
            or not 75 * GIB <= value['total_memory_bytes'] <= 82 * GIB
            or any(type(item) is not int for item in value['capability'])):
        raise ValueError('Exact verified H100 80GB HBM3 qualification hardware required')
    return deepcopy(value)


def reference(path, raw):
    return dict(path=str(path), sha256=identity(raw)['sha256'])


def validate_fit_binding(worker, admission, registered_subsets):
    """Bind the launched fold to the exact work reserved by the journal.

    Admission and input exports are separate files. Both may be individually
    valid while describing different fits; never permit their recombination to
    change which canonical subset consumes a scientific update reservation.
    This is a local builder check, not a new journal query or reservation.
    """
    from frontier.coordination import WorkBinding

    donors = worker.get('training_donors')
    if not isinstance(donors, list) or donors not in registered_subsets:
        raise ValueError('Fit subset is not registered in canonical donor order')
    expected = WorkBinding.representation(donors, start_updates=0, target_updates=20000)
    actual = WorkBinding.from_dict(admission.get('work_binding'))
    if actual != expected:
        raise ValueError('Fit subset/update interval differs from the exact admission work binding')
    if (worker.get('fold_id') != 'subset-' + '-'.join(donors)
            or worker.get('trajectory_id') != 'observation-' + '-'.join(donors) + '-seed0'):
        raise ValueError('Fit fold/trajectory differs from the canonical admitted subset')
    qualified_sha = admission.get('qualification_sha256')
    if (not isinstance(qualified_sha, str) or not re.fullmatch('[0-9a-f]{64}', qualified_sha)
            or worker.get('qualification_sha256') != qualified_sha):
        raise ValueError('Fit qualification SHA differs from its exact admission evidence')
    hardware = validate_qualified_hardware(admission.get('qualified_hardware'))
    peak = admission.get('qualified_peak_reserved_bytes')
    if (type(peak) is not int or peak < 1
            or peak + hardware['required_headroom_bytes'] > hardware['total_memory_bytes']):
        raise ValueError('Measured qualification peak plus required headroom exceeds qualified device capacity')
    return actual


def archive_bytes(payload):
    """Stable bytes: sorted members, fixed metadata and gzip mtime, no symlinks."""
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for name, raw in sorted(payload.items()):
                path = Path(name)
                if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
                    raise ValueError('Unsafe bundle member')
                record = tarfile.TarInfo('bundle/' + name)
                record.size, record.mode, record.uid, record.gid, record.mtime = len(raw), 0o400, 0, 0, 0
                record.uname = record.gname = ''
                archive.addfile(record, io.BytesIO(raw))
    return stream.getvalue()


def manifest_from_template(template, plan, plan_raw):
    manifest = deepcopy(template)
    manifest['metadata']['name'] = plan['run_id']
    manifest['metadata']['annotations'] = {'seadragon_run_id': plan['run_id']}
    spec = manifest['spec']
    spec.update(suspend=True, completions=1, parallelism=1, backoffLimit=0)
    if 'activeDeadlineSeconds' in spec or 'activeDeadlineSeconds' in spec['template']['spec']:
        raise ValueError('Unexpected runtime deadline in retained template')
    spec['template']['metadata']['labels'] = {'app.kubernetes.io/name': 'credo-frontier-observation'}
    if plan['mode'] == 'fit':
        validate_qualified_hardware(plan.get('qualified_hardware'))
        if plan.get('qualified_gpu_product') != QUALIFIED_GPU_PRODUCT:
            raise ValueError('Fit GPU-product restriction differs from qualified hardware')
        terms = spec['template']['spec'].get('affinity', {}).get('nodeAffinity', {}).get(
            'requiredDuringSchedulingIgnoredDuringExecution', {}).get('nodeSelectorTerms', [])
        if not terms:
            raise ValueError('Retained required GPU-product affinity is missing')
        for term in terms:
            expressions = [row for row in term.get('matchExpressions', [])
                           if row.get('key') == 'nvidia.com/gpu.product']
            if (len(expressions) != 1 or expressions[0].get('operator') != 'In'
                    or QUALIFIED_GPU_PRODUCT not in expressions[0].get('values', [])):
                raise ValueError('Qualified GPU product must already be allowed in every affinity branch')
            expressions[0]['values'] = [QUALIFIED_GPU_PRODUCT]
    container = spec['template']['spec']['containers'][0]
    container['command'] = ['/opt/credo/bin/python', plan['bundle_root'] + '/worker_entry.py',
                            '--plan', plan['bundle_root'] + '/PLAN.json', '--plan-sha256', identity(plan_raw)['sha256']]
    values = {'CREDO_PLAN_PATH': plan['bundle_root'] + '/PLAN.json',
              'CREDO_PLAN_SHA256': identity(plan_raw)['sha256'], 'CREDO_RUN_ID': plan['run_id'],
              'PYTHONPATH': plan['bundle_root']}
    for item in container['env']:
        if item['name'] in values:
            item['value'] = values[item['name']]
    return manifest


def construct(study_root, retained_bundle, phase, run_id, admission, mode='qualify', fit_bindings=None):
    study_root, retained_bundle = Path(study_root), Path(retained_bundle)
    if not study_root.is_absolute() or study_root.name != STUDY or mode not in ('qualify', 'fit'):
        raise ValueError('Exact new-study source root/mode required')
    source = {name: read(study_root / name) for name in SOURCE_MEMBERS}
    source_manifest = encoded(dict(schema='credo.frontier.observation-worker-source.v1',
                                  files={name: identity(raw)['sha256'] for name, raw in source.items()}))
    inherited = {name: read(retained_bundle / name, digest) for name, digest in PINNED.items()}
    static = json.loads(inherited['STATIC_REUSE.json'])
    for name, digest in static['files'].items():
        inherited[name] = read(retained_bundle / name, digest)
    if inherited['lifecycle_core.py'] != read(retained_bundle / 'lifecycle_core.py', CORE_SHA):
        raise ValueError('Conflicting retained core identity')
    adapters = {name: read(study_root / 'deployment' / name) for name in ADAPTER_MEMBERS}
    protocol = read(study_root / 'OBSERVATION_PROTOCOL.json')
    authority = read(study_root / 'OBSERVATION_PHASE_CONTRACT.json')
    validate_authority(json.loads(authority))
    parent_authority = read(study_root / 'AUTHORITY.json')
    admission_raw = encoded(admission)
    phase_root = Path(PROJECT) / phase
    bundle = phase_root / 'bundle'
    cap = (16 if mode == 'qualify' else 4) * GIB
    worker = dict(schema='credo.frontier.observation-worker-contract.v1', mode=mode,
                  study_root=PROJECT, output_root=str(phase_root / 'output'), worker_id=run_id,
                  trajectory_id='observation-qualification-v1' if mode == 'qualify' else 'observation-' + '-'.join((fit_bindings or {}).get('training_donors', ())) + '-seed0',
                  source_root=str(bundle / 'source'), source_manifest=str(bundle / 'SOURCE_MANIFEST.json'),
                  source_manifest_sha256=identity(source_manifest)['sha256'], protocol_path=str(bundle / 'OBSERVATION_PROTOCOL.json'),
                  protocol_sha256=identity(protocol)['sha256'], runtime_image=IMAGE, output_limit_bytes=cap)
    if mode == 'fit':
        if fit_bindings is None or set(fit_bindings) != FIT_FIELDS:
            raise ValueError('Complete exact fit-only input and qualification bindings required')
        registered = json.loads(protocol)['training_subsets']
        if fit_bindings['training_donors'] not in registered:
            raise ValueError('Fit subset is not registered')
        if (fit_bindings['metadata_root'] != NATIVE_ROOT or fit_bindings['dataset_root'] != NATIVE_ROOT
                or fit_bindings['reader_source_root'] != str(bundle / 'retained_reader')):
            raise ValueError('Fit must reuse the exact native publication and packaged pinned reader')
        worker.update(fit_bindings)
        validate_fit_binding(worker, admission, registered)
    elif fit_bindings is not None:
        raise ValueError('Qualification cannot bind real input paths')
    worker_raw = encoded(worker)
    controls = dict(inherited)
    controls.pop('gpu-job.json')
    controls.update(adapters)
    controller_sources = {name: identity(raw)['sha256'] for name, raw in controls.items() if name.endswith('.py')}
    plan = dict(schema='credo.frontier.observation-lifecycle.v1', schema_id='credo.frontier.observation-lifecycle.v1',
                run_id=run_id, phase_directory=phase, mode=mode, image=IMAGE,
                bundle_root=str(bundle), worker_output=str(phase_root / 'handshake'), scientific_output=str(phase_root / 'output'),
                control_root=str(Path(CONTROL) / phase), control_sha256=controller_sources['control.py'],
                controller_sources=controller_sources, authorized_CPUs=24, authorized_memory_GiB=200,
                threads=8, gpu_ceiling=1, aggregate_h100_ceiling=4, automatic_retries=0, automatic_successor=False,
                elapsed_time_deadline_seconds=None, runtime_deadline_policy='no_runtime_deadline_user_20260919',
                maximum_output_bytes=cap, maximum_disposable_optimizer_steps=16 if mode == 'qualify' else 0,
                planned_disposable_optimizer_steps=13 if mode == 'qualify' else 0,
                maximum_new_scientific_updates=0 if mode == 'qualify' else 20000,
                startup_gate='inline_h100_controller_release_v1',
                startup_gate_authority='Source-bound observation phase plus exact journal reservation; no automatic successor',
                LARRY_access=False, static_reuse_sha256=PINNED['STATIC_REUSE.json'],
                worker_contract=reference(bundle / 'WORKER_CONTRACT.json', worker_raw),
                source_manifest=reference(bundle / 'SOURCE_MANIFEST.json', source_manifest),
                protocol=reference(bundle / 'OBSERVATION_PROTOCOL.json', protocol),
                phase_authority=reference(bundle / 'OBSERVATION_PHASE_CONTRACT.json', authority),
                parent_authority=reference(bundle / 'AUTHORITY.json', parent_authority),
                admission=reference(bundle / 'ADMISSION.json', admission_raw))
    if mode == 'fit':
        plan.update(qualified_hardware=validate_qualified_hardware(admission['qualified_hardware']),
                    qualification_sha256=worker['qualification_sha256'], qualified_gpu_product=QUALIFIED_GPU_PRODUCT,
                    qualified_peak_reserved_bytes=admission['qualified_peak_reserved_bytes'])
    validate_plan(plan)
    # Validate reservation before creating any output artifacts. Live journal
    # reconciliation remains the launch controller's responsibility.
    validate_admission(plan, admission)
    plan_raw = encoded(plan)
    controls.update({'PLAN.json': plan_raw, 'WORKER_CONTRACT.json': worker_raw, 'SOURCE_MANIFEST.json': source_manifest,
                     'OBSERVATION_PROTOCOL.json': protocol, 'OBSERVATION_PHASE_CONTRACT.json': authority,
                     'AUTHORITY.json': parent_authority, 'ADMISSION.json': admission_raw,
                     'gpu-job.json': encoded(manifest_from_template(json.loads(inherited['gpu-job.json']), plan, plan_raw))})
    payload = dict(controls)
    payload.update({'source/' + name: raw for name, raw in source.items()})
    if mode == 'fit':
        # Reuse the existing six metadata files in the native publication. Only
        # four small hash-bound reader sources are staged: no count, cell,
        # historical model or metadata-copy reconstruction is performed.
        reader_root = study_root.parent / READER_RELATIVE_ROOT
        payload.update({'retained_reader/' + name: read(reader_root / name, digest)
                        for name, digest in READER_MEMBERS.items()})
    inventory_raw = encoded({name: identity(raw) for name, raw in sorted(payload.items())})
    payload['INVENTORY.json'] = inventory_raw
    archive = archive_bytes(payload)
    build = dict(schema='credo.frontier.observation-build.v1', phase=phase, run_id=run_id, mode=mode,
                 archive=identity(archive), archive_top_level='bundle', payload_inventory={name: identity(raw) for name, raw in sorted(payload.items())},
                 archive_sha256=identity(archive)['sha256'], inventory_sha256=identity(inventory_raw)['sha256'],
                 plan_sha256=identity(plan_raw)['sha256'],
                 lifecycle_plan_sha256=identity(plan_raw)['sha256'], worker_contract_sha256=identity(worker_raw)['sha256'],
                 source_manifest_sha256=identity(source_manifest)['sha256'], phase_authority_sha256=identity(authority)['sha256'],
                 manifest_sha256=identity(controls['gpu-job.json'])['sha256'], remote_phase_root=str(phase_root),
                 remote_control_root=plan['control_root'], static_proof='reuse_verified_neutral_only',
                 newly_staged_real_inputs=False, scheduler_calls=0, numerical_calls=0,
                 fitting_reader_sources={} if mode == 'qualify' else dict(READER_MEMBERS),
                 runtime_qualification_established=False, automatic_successor=False)
    build['builder_source_sha256'] = identity(read(Path(__file__).absolute()))['sha256']
    return payload, archive, build


def write_package(output, payload, archive, build):
    output = Path(output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError('New explicit local package directory required')
    if any(path.is_symlink() for path in output.parents):
        raise ValueError('Unsafe package ancestor')
    output.mkdir(mode=0o700)
    for name, raw in sorted(payload.items()):
        target = output / 'bundle' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(raw)
    for name, raw in (('bundle.tar.gz', archive), ('BUILD.json', encoded(build))):
        with (output / name).open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    if read(output / 'bundle.tar.gz', build['archive']['sha256']) != archive:
        raise ValueError('Built archive readback differs')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True)
    parser.add_argument('--retained-bundle', required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--admission-json', required=True)
    parser.add_argument('--mode', choices=('qualify', 'fit'), default='qualify')
    parser.add_argument('--fit-bindings')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    payload, archive, build = construct(args.study_root, args.retained_bundle, args.phase, args.run_id,
        read_json(args.admission_json), args.mode, None if args.fit_bindings is None else read_json(args.fit_bindings))
    write_package(args.output, payload, archive, build)
    print(json.dumps(dict(status='local_package_created_not_staged', output=args.output, archive=build['archive'],
                          lifecycle_plan_sha256=build['lifecycle_plan_sha256']), sort_keys=True))


if __name__ == '__main__':
    main()
