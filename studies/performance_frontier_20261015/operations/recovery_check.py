"""Operate only the approved zero-update recovery check using retained lifecycle."""
import argparse
import json
from pathlib import Path
import sys
from transport import ROOT
import observation as retained
sys.path.insert(0,str(ROOT))
from deployment.recovery_check_common import validate_plan, validate_authority, validate_admission


def package(path):
    path=Path(path).absolute()
    if not path.is_relative_to(ROOT) or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Exact new study-owned package required')
    build=json.loads((path/'BUILD.json').read_bytes())
    plan=validate_plan(json.loads((path/'bundle/PLAN.json').read_bytes()))
    if (build.get('schema')!='credo.frontier.retention-check-build.v1'
            or build['plan_sha256']!=retained.sha(path/'bundle/PLAN.json')
            or build['archive_sha256']!=retained.sha(path/'bundle.tar.gz')
            or build['inventory_sha256']!=retained.sha(path/'bundle/INVENTORY.json')
            or build['run_id']!=plan['run_id'] or build['phase']!=plan['phase_directory']):
        raise ValueError('Recovery package changed')
    inventory=json.loads((path/'bundle/INVENTORY.json').read_bytes())
    found=set()
    for p in (path/'bundle').rglob('*'):
        if p.is_symlink() or not (p.is_dir() or p.is_file()): raise ValueError('Unsafe package member')
        if p.is_file(): found.add(p.relative_to(path/'bundle').as_posix())
    if found!=set(inventory)|{'INVENTORY.json'}: raise ValueError('Package inventory differs')
    for n,ref in inventory.items():
        if Path(n).is_absolute() or '..' in Path(n).parts: raise ValueError('Unsafe package path')
        p=path/'bundle'/n
        if retained.sha(p)!=ref['sha256'] or p.stat().st_size!=ref['size_bytes']:
            raise ValueError('Packaged member differs')
    worker=json.loads((path/'bundle/WORKER_CONTRACT.json').read_bytes())
    for ref in [plan['worker_contract'],*[worker[k] for k in ('source_manifest','authority','admission','parent_intake')]]:
        if Path(ref['path']).parent!=Path(plan['bundle_root']): raise ValueError('Metadata escaped package')
        p=path/'bundle'/Path(ref['path']).name
        if retained.sha(p)!=ref['sha256'] or p.stat().st_size!=ref['size_bytes']:
            raise ValueError('Bound metadata differs')
    authority=validate_authority(json.loads((path/'bundle/RECOVERY_CHECK_AUTHORITY.json').read_bytes()))
    validate_admission(worker,json.loads((path/'bundle/ADMISSION.json').read_bytes()))
    if (build['source_manifest_sha256']!=worker['source_manifest']['sha256']
            or build['phase_authority_sha256']!=worker['authority']['sha256']
            or authority['parent_intake_sha256']!=worker['parent_intake']['sha256']):
        raise ValueError('Recovery input/source authority differs')
    return path,build,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','stage','closeout','dryrun','create','launch'))
    parser.add_argument('--package',required=True);parser.add_argument('--session-receipt',required=True)
    args=parser.parse_args()
    path,build,plan=package(args.package)
    retained.session_gate(args.session_receipt)
    admission=retained.admission_gate(path,build,plan)
    if args.action=='preflight': result=retained.preflight(path,build,plan,admission)
    elif args.action=='stage': result=retained.stage(path,build,plan)
    elif args.action=='closeout': result=retained.closeout(path,build)
    else: result=retained.lifecycle(path,build,plan,args.action)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
