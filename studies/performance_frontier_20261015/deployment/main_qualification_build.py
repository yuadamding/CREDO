"""Prepare one exact main-qualification bundle from external approval/admission.

No journal reservation, remote operation or launch happens in this builder.
The operator must separately reconcile the live aggregate journal at launch.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .common import CONTROL, GIB, IMAGE, PROJECT, STUDY, encoded, identity, read, publish
from .build import PINNED, archive_bytes
from .main_qualification_source import construct as construct_source
from .main_qualification_common import reference, validate_plan, validate_authority, validate_admission
from frontier.main_qualification_contract import decode, settings
from frontier.main_qualification_plan import plan_identity


ADAPTERS = {'common.py': 'common.py', 'capacity.py': 'capacity.py',
    'main_qualification_common.py': 'main_qualification_common.py',
    'worker_entry.py': 'main_qualification_entry.py', 'control.py': 'main_qualification_control.py'}
RUNTIME = dict(python='3.12.13', torch='2.6.0+cu124', cuda_runtime='12.4', numpy='2.5.1', scipy='1.18.0')


def construct(study_root, reader_root, retained_bundle, phase, run_id, authority, admission):
    study_root, retained_bundle = Path(study_root), Path(retained_bundle)
    if not study_root.is_absolute() or study_root.name != STUDY:
        raise ValueError('Exact study root required')
    authority_raw = authority if isinstance(authority, bytes) else encoded(authority)
    authority = json.loads(authority_raw)
    _, source_report, source = construct_source(study_root, reader_root)
    inherited = {n: read(retained_bundle/n, sha) for n, sha in PINNED.items()}
    for name, sha in json.loads(inherited['STATIC_REUSE.json'])['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe retained static-proof source')
        inherited[name] = read(retained_bundle/name, sha)
    template = json.loads(inherited.pop('gpu-job.json'))
    adapters = {dest: read(study_root/'deployment'/src) for dest, src in ADAPTERS.items()}
    if set(source) & (set(inherited) | set(adapters)):
        raise ValueError('Numerical and lifecycle payload members overlap')
    controls = dict(inherited, **adapters)
    bundle = Path(PROJECT)/phase/'bundle'
    source.update({'MAIN_QUALIFICATION_AUTHORITY.json': authority_raw, 'ADMISSION.json': encoded(admission)})
    names = dict(source_manifest='SOURCE_MANIFEST.json', worker_source_manifest='WORKER_SOURCE_MANIFEST.json',
        observation_protocol='OBSERVATION_PROTOCOL.json', normalization_protocol='NORMALIZATION_PROTOCOL.json',
        rules='MAIN_OBJECTIVE_RULES.json', authority='MAIN_QUALIFICATION_AUTHORITY.json', admission='ADMISSION.json')
    worker = dict(schema='credo.frontier.main-qualification-contract.v1', run_id=run_id, phase=phase,
        study_root=PROJECT, output_root=str(bundle.parent/'output'), source_root=str(bundle/'source'),
        reader_source_root=str(bundle/'retained_reader'), runtime_image=IMAGE, expected_runtime=RUNTIME,
        plan_sha256=plan_identity(), settings=settings(),
        **{key: reference(bundle/name, source[name]) for key, name in names.items()})
    validate_authority(worker, authority)
    validate_admission(worker, admission)
    worker_raw = encoded(worker)
    decode(worker_raw)
    controllers = {n: identity(raw)['sha256'] for n, raw in controls.items() if n.endswith('.py')}
    plan = dict(schema='credo.frontier.main-qualification-lifecycle.v1',
        schema_id='credo.frontier.main-qualification-lifecycle.v1', mode='main-qualification',
        phase_directory=phase, run_id=run_id, image=IMAGE, bundle_root=str(bundle),
        worker_output=str(bundle.parent/'handshake'), scientific_output=worker['output_root'],
        control_root=str(Path(CONTROL)/phase), control_sha256=controllers['control.py'],
        controller_sources=controllers, authorized_CPUs=24, authorized_memory_GiB=200,
        threads=8, gpu_ceiling=1, aggregate_h100_ceiling=4, automatic_retries=0, automatic_successor=False,
        elapsed_time_deadline_seconds=None, runtime_deadline_policy='no_runtime_deadline_user_20260919',
        startup_gate='inline_h100_controller_release_v1',
        startup_gate_authority='Distinct external main qualification authority and exact journal reservation',
        maximum_output_bytes=16*GIB, maximum_disposable_optimizer_steps=112,
        planned_disposable_optimizer_steps=111, maximum_new_scientific_updates=0,
        real_biological_payload_access=False, LARRY_access=False,
        static_reuse_sha256=PINNED['STATIC_REUSE.json'], worker_contract=reference(bundle/'WORKER_CONTRACT.json', worker_raw),
        source_manifest=worker['source_manifest'], phase_authority=worker['authority'], admission=worker['admission'])
    validate_plan(plan)
    plan_raw = encoded(plan)
    manifest = deepcopy(template)
    manifest['metadata'].update(name=run_id, annotations={'seadragon_run_id': run_id})
    spec = manifest['spec']
    spec.update(suspend=True, completions=1, parallelism=1, backoffLimit=0)
    pod = spec['template']['spec']
    if 'activeDeadlineSeconds' in spec or 'activeDeadlineSeconds' in pod:
        raise ValueError('Unexpected deadline in retained template')
    spec['template']['metadata']['labels'] = {'app.kubernetes.io/name': 'credo-frontier-main-qualification'}
    terms = pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']
    if not terms:
        raise ValueError('Retained GPU affinity missing')
    for term in terms:
        rows = [r for r in term['matchExpressions'] if r['key'] == 'nvidia.com/gpu.product']
        if len(rows) != 1 or rows[0]['operator'] != 'In' or 'NVIDIA-H100-80GB-HBM3' not in rows[0]['values']:
            raise ValueError('Required HBM3 product not present in institutional template')
        rows[0]['values'] = ['NVIDIA-H100-80GB-HBM3']
    container = pod['containers'][0]
    container['command'] = ['/opt/credo/bin/python', str(bundle/'worker_entry.py'), '--plan', str(bundle/'PLAN.json'),
                            '--plan-sha256', identity(plan_raw)['sha256']]
    values = dict(CREDO_PLAN_PATH=str(bundle/'PLAN.json'), CREDO_PLAN_SHA256=identity(plan_raw)['sha256'],
                  CREDO_RUN_ID=run_id, PYTHONPATH=str(bundle))
    names = [r['name'] for r in container['env']]
    if len(names) != len(set(names)) or not set(values) <= set(names):
        raise ValueError('Protected environment identity missing or duplicated')
    for row in container['env']:
        if row['name'] in values:
            row.pop('valueFrom', None)
            row['value'] = values[row['name']]
    payload = dict(controls, **source)
    payload.update({'PLAN.json': plan_raw, 'WORKER_CONTRACT.json': worker_raw, 'gpu-job.json': encoded(manifest)})
    inventory = {n: identity(raw) for n, raw in payload.items()}
    payload['INVENTORY.json'] = encoded(inventory)
    archive = archive_bytes(payload)
    report = dict(schema='credo.frontier.main-qualification-build.v1', phase=phase, run_id=run_id,
        plan_sha256=identity(plan_raw)['sha256'], archive_sha256=identity(archive)['sha256'],
        archive_bytes=len(archive), inventory_sha256=identity(payload['INVENTORY.json'])['sha256'],
        source_manifest_sha256=source_report['source_manifest_sha256'],
        worker_source_manifest_sha256=source_report['worker_source_manifest_sha256'],
        phase_authority_sha256=worker['authority']['sha256'],
        disposable_optimizer_calls=0, scientific_updates=0, numerical_execution=False,
        cuda_qualification=False, scientific_launch_authority=False, scheduler_submitted=False)
    return payload, archive, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-root', 'reader-root', 'retained-bundle', 'phase', 'run-id', 'authority', 'admission', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    payload, archive, report = construct(args.study_root, args.reader_root, args.retained_bundle,
        args.phase, args.run_id, read(args.authority), json.loads(read(args.admission)))
    output = Path(args.output)
    if not output.is_absolute() or any(p.is_symlink() for p in (output,) + tuple(output.parents)):
        raise ValueError('Explicit nonsymlink new output root required')
    output.mkdir(mode=0o700)
    for name, raw in dict({'bundle/'+n: raw for n, raw in payload.items()}, **{'bundle.tar.gz': archive}).items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if read(path, identity(raw)['sha256']) != raw:
            raise ValueError('Bundle publication readback differs')
    publish(output/'BUILD.json', report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
