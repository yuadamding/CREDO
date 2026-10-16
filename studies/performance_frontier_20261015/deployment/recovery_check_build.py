"""Build the one approved recovery check; no submission or numerical work."""
import json
import os
from pathlib import Path
from .common import CONTROL, CORE_SHA, GIB, IMAGE, PROJECT, encoded, identity, read, publish
from .build import PINNED, SOURCE_MEMBERS, archive_bytes, manifest_from_template
from .recovery_check_common import RUN, PHASE, RUNTIME, reference, same, validate_authority, validate_admission, validate_plan

MEMBERS=SOURCE_MEMBERS+tuple('frontier/'+n for n in (
    'observation_recovery.py','retention_canary.py','observation_recovery_check.py'))
ADAPTERS={'common.py':'common.py','capacity.py':'capacity.py',
    'main_qualification_common.py':'main_qualification_common.py',
    'recovery_check_common.py':'recovery_check_common.py',
    'worker_entry.py':'recovery_check_entry.py','control.py':'recovery_check_control.py'}


def source_payload(root):
    sources={n:read(Path(root)/n) for n in MEMBERS}
    manifest=encoded(dict(schema='credo.frontier.retention-check-source.v1',
        entry='frontier/observation_recovery_check.py',files={n:identity(raw) for n,raw in sources.items()}))
    return sources,manifest


def construct(root,retained,authority_raw,intake_raw,admission):
    root,retained=Path(root),Path(retained)
    authority=validate_authority(json.loads(authority_raw))
    same(authority,dict(run_id=RUN,phase=PHASE))
    if authority['parent_intake_sha256']!=identity(intake_raw)['sha256']:
        raise ValueError('Bound parent intake changed')
    inherited={n:read(retained/n,digest) for n,digest in PINNED.items()}
    for n,digest in json.loads(inherited['STATIC_REUSE.json'])['files'].items():
        if Path(n).is_absolute() or '..' in Path(n).parts: raise ValueError('Unsafe inherited member')
        inherited[n]=read(retained/n,digest)
    template=json.loads(inherited.pop('gpu-job.json'))
    controls=dict(inherited,**{n:read(root/'deployment'/src) for n,src in ADAPTERS.items()})
    sources,manifest=source_payload(root)
    bundle=Path(PROJECT)/PHASE/'bundle'
    payload={**controls,**{'source/'+n:raw for n,raw in sources.items()},
        'SOURCE_MANIFEST.json':manifest,'RECOVERY_CHECK_AUTHORITY.json':authority_raw,
        'PARENT_INTAKE.json':intake_raw,'ADMISSION.json':encoded(admission)}
    names=dict(source_manifest='SOURCE_MANIFEST.json',authority='RECOVERY_CHECK_AUTHORITY.json',
        parent_intake='PARENT_INTAKE.json',admission='ADMISSION.json')
    worker=dict(schema='credo.frontier.retention-check-worker.v1',run_id=RUN,phase=PHASE,
        source_root=str(bundle/'source'),output_root=str(bundle.parent/'output'),runtime_image=IMAGE,
        expected_runtime=RUNTIME,optimizer_calls=0,output_limit_bytes=4*GIB,
        **{key:reference(bundle/n,payload[n]) for key,n in names.items()})
    validate_admission(worker,admission)
    worker_raw=encoded(worker)
    controllers={n:identity(raw)['sha256'] for n,raw in controls.items() if n.endswith('.py')}
    plan=dict(schema='credo.frontier.retention-check-lifecycle.v1',schema_id='credo.frontier.retention-check-lifecycle.v1',
        mode='retention-check',run_id=RUN,phase_directory=PHASE,image=IMAGE,
        bundle_root=str(bundle),worker_output=str(bundle.parent/'handshake'),scientific_output=worker['output_root'],
        control_root=str(Path(CONTROL)/PHASE),control_sha256=controllers['control.py'],controller_sources=controllers,
        authorized_CPUs=24,authorized_memory_GiB=200,threads=8,gpu_ceiling=1,aggregate_h100_ceiling=4,
        automatic_retries=0,automatic_successor=False,elapsed_time_deadline_seconds=None,
        runtime_deadline_policy='no_runtime_deadline_user_20260919',startup_gate='inline_h100_controller_release_v1',
        startup_gate_authority='Explicit zero-update recovery check approval; no training successor',
        maximum_output_bytes=4*GIB,maximum_disposable_optimizer_steps=0,planned_disposable_optimizer_steps=0,
        maximum_new_scientific_updates=0,raw_data_access=False,LARRY_access=False,
        static_reuse_sha256=PINNED['STATIC_REUSE.json'],worker_contract=reference(bundle/'WORKER_CONTRACT.json',worker_raw),
        source_manifest=worker['source_manifest'],phase_authority=worker['authority'],admission=worker['admission'])
    validate_plan(plan)
    plan_raw=encoded(plan)
    manifest_job=manifest_from_template(template,plan,plan_raw)
    pod=manifest_job['spec']['template']['spec']
    for term in pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']:
        rows=[r for r in term['matchExpressions'] if r['key']=='nvidia.com/gpu.product']
        if len(rows)!=1 or 'NVIDIA-H100-80GB-HBM3' not in rows[0]['values']:
            raise ValueError('Required H100 product missing')
        rows[0]['values']=['NVIDIA-H100-80GB-HBM3']
    payload.update({'PLAN.json':plan_raw,'WORKER_CONTRACT.json':worker_raw,'gpu-job.json':encoded(manifest_job)})
    payload['INVENTORY.json']=encoded({n:identity(raw) for n,raw in payload.items()})
    archive=archive_bytes(payload)
    report=dict(schema='credo.frontier.retention-check-build.v1',run_id=RUN,phase=PHASE,
        plan_sha256=identity(plan_raw)['sha256'],archive_sha256=identity(archive)['sha256'],archive_bytes=len(archive),
        inventory_sha256=identity(payload['INVENTORY.json'])['sha256'],
        source_manifest_sha256=worker['source_manifest']['sha256'],phase_authority_sha256=worker['authority']['sha256'],
        optimizer_calls=0,scientific_updates=0,remote_submission=False,automatic_successor=False)
    return payload,archive,report


def publish_package(output,payload,archive,report):
    output=Path(output)
    if not output.is_absolute() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Absolute new nonsymlink package root required')
    output.mkdir(mode=0o700)
    for n,raw in {**{'bundle/'+n:raw for n,raw in payload.items()},'bundle.tar.gz':archive}.items():
        path=output/n;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if read(path,identity(raw)['sha256'])!=raw:raise ValueError('Package readback differs')
    publish(output/'BUILD.json',report)
