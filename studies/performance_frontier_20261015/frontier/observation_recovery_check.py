"""One actual-mount canary and four exact checkpoint restores; no optimizer step.

This check does not continue a scientific trainer, consume a new data batch,
qualify a later optimizer update, or authorize a successor.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from .checkpoints import TrajectoryLease, load_checkpoint, _rng_state
from .observation import ObservationFold, SparseR48
from .observation_recovery import import_parent, _tensor_record, CONSUMED
from .observation_worker import _configure_cuda, _runtime, _resources, _output_bytes
from .retention import _write_once
from .retention_canary import run_canary


def exact_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual,torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not torch.equal(actual.detach().cpu(),expected.detach().cpu())):
            raise ValueError('Restored tensor differs')
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual) != set(expected):
            raise ValueError('Restored mapping differs')
        for key in expected: exact_tree(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError('Restored sequence differs')
        for a,b in zip(actual,expected): exact_tree(a,b)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError('Restored value differs')


def forbidden_step(*args,**kwargs):
    raise RuntimeError('Optimizer calls are forbidden in the zero-update recovery check')


def check_parent(parent, pair, worker, contract_sha, root):
    original = parent['identity']['value']
    child = dict(original,trajectory_id=worker['run_id']+'-'+pair.lower(),
        source_sha256=worker['source_manifest']['sha256'],worker_contract_sha256=contract_sha,
        recovery_parent_checkpoint_sha256=parent['checkpoint_payload']['sha256'],
        recovery_authority_sha256=worker['authority']['sha256'])
    with TrajectoryLease(root/pair,child,worker['run_id']) as lease:
        imported = import_parent(lease,parent,authority_sha256=worker['authority']['sha256'])
        saved = _tensor_record(parent['checkpoint_payload'],Path(parent['checkpoint_payload']['path']))
        metadata = saved['model_constructor_metadata']
        fold = dict(metadata['observation_fold'])
        fold['training_donors'] = tuple(fold['training_donors'])
        if ''.join(fold['training_donors']) != pair or fold['fold_id'] != original['fold_id']:
            raise ValueError('Parent model constructor fold differs from intake')
        model = SparseR48(ObservationFold(**fold),seed=0,**{
            name:tuple(metadata[name]) if name=='hidden_dims' else metadata[name]
            for name in ('features','hidden_dims','latent_dim','maximum_rows','maximum_dense_bytes')}).to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,betas=(0.9,0.999),eps=1e-8)
        optimizer.step = forbidden_step
        state = load_checkpoint(lease,imported['checkpoint_directory'],expected_sha256=imported['checkpoint_sha256'],
            model=model,optimizer=optimizer,scheduler=None,execution_scope='scientific')
        exact_tree(model.state_dict(),saved['model_state'])
        exact_tree(optimizer.state_dict(),saved['optimizer_state'])
        exact_tree(_rng_state(),saved['rng'])
        exact_tree(state['sampler_state'],saved['sampler_state'])
        for key in ('accepted_updates','attempted_updates','phase','scientific_updates','disposable_optimizer_calls'):
            exact_tree(state[key],saved[key])
        for key in saved['count_cache']:
            if key not in ('audit_history','selected_candidate'):
                exact_tree(state['count_cache'][key],saved['count_cache'][key])
        for a,b in zip(state['count_cache']['audit_history'],saved['count_cache']['audit_history']):
            exact_tree({k:v for k,v in a.items() if k!='artifact'},{k:v for k,v in b.items() if k!='artifact'})
        exact_tree({k:v for k,v in state['count_cache']['selected_candidate'].items() if k!='artifact'},
                   {k:v for k,v in saved['count_cache']['selected_candidate'].items() if k!='artifact'})
        result = dict(pair=pair,status='exact_state_restoration_verified_on_cuda',
            parent_checkpoint=parent['checkpoint_payload'],child_checkpoint_sha256=imported['checkpoint_sha256'],
            consumed_updates=CONSUMED,remaining_update_cap=20000-CONSUMED,optimizer_calls=0,
            scientific_updates=0,parent_changed=False,training_data_opened=False,
            model_optimizer_rng_sampler_and_audit_values_equal=True,training_continuation_qualified=False,
            imported_payload_bytes=imported['payload_bytes_written'])
        _write_once(lease.root/'RESTORE_VERIFIED.json',result)
        del model,optimizer,saved,state
    gc.collect(); torch.cuda.empty_cache()
    return result


def main(argv=None):
    from recovery_check_common import verify_bundle, same, PAIRS
    from common import read, read_json, identity
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','release','release-sha256'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    plan=read_json(args.plan,args.plan_sha256)
    worker,_=verify_bundle(plan,Path(args.plan).parent)
    release=read_json(args.release,args.release_sha256)
    same(release,dict(status='startup_released',run_id=worker['run_id'],
        worker_contract_sha256=plan['worker_contract']['sha256'],scientific_fit_authorized=False,
        optimizer_calls_authorized=0,checkpoint_intake_sha256=worker['parent_intake']['sha256']))
    if _runtime()!=worker['expected_runtime']: raise ValueError('Qualified runtime differs')
    gpu=_configure_cuda()
    # This also blocks an accidental optimizer call introduced in a helper.
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD): cls.step=forbidden_step
    root=Path(worker['output_root']); root.mkdir(mode=0o700)
    try:
        intake=read_json(worker['parent_intake']['path'],worker['parent_intake']['sha256'])
        parents=intake['fits']
        if set(parents)!=set(PAIRS): raise ValueError('Exactly four approved parent checkpoints required')
        canary=run_canary(root/'storage_canary')
        rows=[]
        for pair in PAIRS:
            rows.append(check_parent(parents[pair],pair,worker,plan['worker_contract']['sha256'],root))
            if _output_bytes(root)>worker['output_limit_bytes']-4*1024**2:
                raise ValueError('Combined recovery check output cap exceeded')
        verify_bundle(plan,Path(args.plan).parent)
        result=dict(schema='credo.frontier.retention-check-completion.v1',status='passed',
            run_id=worker['run_id'],contract=plan['worker_contract'],
            release=dict(path=args.release,**identity(read(args.release,args.release_sha256))),
            source_manifest_sha256=worker['source_manifest']['sha256'],
            parent_intake_sha256=worker['parent_intake']['sha256'],optimizer_calls=0,scientific_updates=0,
            consumed_updates_preserved=5120,remaining_scientific_updates_not_authorized=74880,
            parent_writes=False,raw_data_access=False,training_continuation_qualified=False,
            scientific_relaunch_authorized=False,automatic_retry=False,automatic_successor=False,
            canary_passed=canary['status']=='passed',parents=rows,gpu=gpu,resources=_resources('cuda'),
            output_bytes_before_completion=_output_bytes(root))
        _write_once(root/'COMPLETE.json',result)
    except BaseException as error:
        _write_once(root/'CHECK_FAILED.json',dict(exception_type=type(error).__name__,message=str(error),
            optimizer_calls=0,scientific_updates=0,automatic_retry=False))
        raise
    return 0


if __name__=='__main__': raise SystemExit(main())
