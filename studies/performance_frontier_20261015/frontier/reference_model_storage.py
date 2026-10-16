"""Exclusive inference-only publication for the two simple reference models.

Reuse the tested canonical NPY writer and byte-reader. No pickle, compressed
archive, optimizer state, model discovery or refit occurs. A completion here
means artifact publication, not acceptance of its fitting experiment. The
caller separately verifies the source-bound fitting receipt, numerical
qualification, admission and selector evidence. Loaded arrays are private
read-only copies: unlike a mapped population cache, they do not borrow storage
from a file after this function returns.
"""

from dataclasses import asdict
import hashlib
import json
import os
import stat

import numpy as np

from . import abundance_head as ah
from . import ridge_readout as rr
from .cache_storage import (_array_record, _blocks, _digest_stream, _header,
    _json_bytes, _open_verified, _parse, _path, _stamp, _write_once)
from .simple_reference_inputs import ReferenceBindings


SCHEMA='credo.frontier.simple-reference-model.v1'
COMPLETE_SCHEMA='credo.frontier.simple-reference-model-complete.v1'
MAXIMUM_BYTES=128*1024**2
MAXIMUM_METADATA_BYTES=4*1024**2


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase SHA-256 required')


def _positive(value, maximum=None):
    if type(value) is not int or value<1 or maximum is not None and value>maximum:
        raise ValueError('Explicit bounded positive resource size required')


def _metadata(value):
    payload=_json_bytes(value)
    if len(payload)>MAXIMUM_METADATA_BYTES: raise ValueError('Reference model metadata exceeds bound')
    return payload


def _binding_record(bindings):
    if not isinstance(bindings,ReferenceBindings): raise ValueError('Exact shared reference bindings required')
    bindings.__post_init__()
    return json.loads(_metadata(asdict(bindings)))


def _parts(model, bindings):
    if type(model) is rr.RidgeReadout:
        if model.binding!=bindings.ridge: raise ValueError('Ridge fitting binding differs')
        model.__post_init__()
        kind='ridge';metadata=dict(regularization=model.regularization)
        arrays={name:getattr(model,name) for name in ('mean','scale','intercept','coefficients')}
    elif type(model) is ah.AbundanceHead:
        if model.binding!=bindings.abundance: raise ValueError('Abundance fitting binding differs')
        kind='abundance'
        if (type(model.target_ids) is not tuple or tuple(sorted(set(model.target_ids)))!=model.target_ids
                or not set(model.target_ids).issubset({g.target_id for g in model.binding.catalog if not g.is_control})
                or isinstance(model.regularization,bool) or model.regularization not in ah.REGULARIZATION_GRID
                or type(model.diagnostics) is not dict):
            raise ValueError('Abundance target inventory, penalty or diagnostics differs')
        lookup={t:i for i,t in enumerate(model.target_ids)}
        route=np.array([-1 if g.is_control else lookup.get(g.target_id,-1) for g in model.binding.catalog],dtype=np.int64)
        if not np.array_equal(model.target_index,route): raise ValueError('Abundance target routing differs')
        for name,shape in (('mean',(100,)),('scale',(100,)),('parameters',(100+len(model.target_ids),))):
            ah._array(getattr(model,name),shape,name)
        if np.any(model.scale<=0): raise ValueError('Positive fitting feature scales required')
        metadata=dict(regularization=model.regularization,target_ids=list(model.target_ids),diagnostics=model.diagnostics)
        arrays={name:getattr(model,name) for name in ('mean','scale','target_index','parameters')}
    else: raise TypeError('Only the two exact inference reference types can be published')
    for name,array in arrays.items():
        dtype=np.int64 if name=='target_index' else np.float64
        if (not isinstance(array,np.ndarray) or array.dtype!=dtype or not array.flags.c_contiguous
                or array.flags.writeable or not np.isfinite(array).all()):
            raise ValueError('Read-only contiguous finite canonical reference parameters required')
    # Capture dictionary contents independently of a mutable diagnostics object.
    return kind,json.loads(_metadata(metadata)),arrays


def _completion(kind, metadata, records, output_cap_bytes):
    result=dict(schema=COMPLETE_SCHEMA,status='publication_complete',kind=kind,
        manifest_sha256=hashlib.sha256(metadata).hexdigest(),manifest_bytes=len(metadata),
        array_files=len(records),output_cap_bytes=output_cap_bytes,total_bytes=0,
        fitting_completion_verified=False,scientific_launch_authority=False)
    while True:
        raw=_metadata(result)
        total=len(raw)+len(metadata)+sum(r['file_bytes'] for r in records)
        if total==result['total_bytes']: return result,raw
        result['total_bytes']=total


def publish_reference(model, bindings, destination, *, producer_sha256, fitting_receipt_sha256, output_cap_bytes):
    """Publish once in a NEW owned directory; failures remain incomplete."""
    _sha(producer_sha256);_sha(fitting_receipt_sha256);_positive(output_cap_bytes,MAXIMUM_BYTES)
    binding_record=_binding_record(bindings)
    kind,model_metadata,arrays=_parts(model,bindings)
    records=[_array_record(0,name,array) for name,array in arrays.items()]
    document=dict(schema=SCHEMA,kind=kind,bindings=binding_record,model=model_metadata,
        producer_sha256=producer_sha256,fitting_receipt_sha256=fitting_receipt_sha256,
        arrays=records,inference_only=True,optimizer_resume_supported=False)
    metadata=_metadata(document)
    complete,complete_bytes=_completion(kind,metadata,records,output_cap_bytes)
    if complete['total_bytes']>output_cap_bytes: raise ValueError('Reference publication exceeds output cap before writes')
    root=_path(destination)
    if os.path.lexists(root): raise FileExistsError('Reference artifact exists; no overwrite or retry')
    parent=_path(root.parent)
    if parent.stat().st_uid!=os.geteuid(): raise ValueError('Publication parent must be owned by writer')
    parent_fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);root_fd=None
    try:
        if _stamp(os.fstat(parent_fd))!=_stamp(parent.lstat()): raise ValueError('Parent changed during opening')
        os.mkdir(root.name,0o700,dir_fd=parent_fd)
        root_fd=os.open(root.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
        os.fsync(parent_fd)
        stamps={}
        for record in records:
            array=arrays[record['field']]
            def chunks():
                yield _header(array.dtype,array.shape)
                yield from _blocks(array)
            stamps[record['relative_path']]=_write_once(root_fd,record['relative_path'],chunks(),
                expected_sha256=record['file_sha256'],expected_bytes=record['file_bytes'])
        stamps['MANIFEST.json']=_write_once(root_fd,'MANIFEST.json',(metadata,),
            expected_sha256=complete['manifest_sha256'],expected_bytes=len(metadata))
        kind_after,metadata_after,arrays_after=_parts(model,bindings)
        if (kind_after!=kind or metadata_after!=model_metadata or _binding_record(bindings)!=binding_record
                or [_array_record(0,n,a) for n,a in arrays_after.items()]!=records):
            raise ValueError('Reference model changed during publication')
        expected=set(stamps)
        if set(os.listdir(root_fd))!=expected: raise ValueError('Unexpected pre-completion reference member')
        for name,stamp in stamps.items():
            if _stamp(os.stat(name,dir_fd=root_fd,follow_symlinks=False))!=stamp:
                raise ValueError('Published reference file changed')
        digest=hashlib.sha256(complete_bytes).hexdigest()
        _write_once(root_fd,'.COMPLETE.pending',(complete_bytes,),expected_sha256=digest,expected_bytes=len(complete_bytes))
        os.link('.COMPLETE.pending','COMPLETE.json',src_dir_fd=root_fd,dst_dir_fd=root_fd,follow_symlinks=False)
        os.unlink('.COMPLETE.pending',dir_fd=root_fd)
        os.fsync(root_fd)
        if set(os.listdir(root_fd))!=expected|{'COMPLETE.json'}: raise ValueError('Unexpected completed reference member')
        for name,stamp in stamps.items():
            if _stamp(os.stat(name,dir_fd=root_fd,follow_symlinks=False))!=stamp:
                raise ValueError('Published reference file changed at completion')
        total=sum(os.stat(n,dir_fd=root_fd,follow_symlinks=False).st_size for n in os.listdir(root_fd))
        if total!=complete['total_bytes'] or _stamp(os.fstat(root_fd))!=_stamp(root.lstat()):
            raise ValueError('Completed reference byte or directory identity differs')
        return dict(root=str(root),kind=kind,complete_sha256=digest,total_bytes=total,
            output_cap_bytes=output_cap_bytes,producer_sha256=producer_sha256,fitting_receipt_sha256=fitting_receipt_sha256,
            inference_only=True,fitting_completion_verified=False,scientific_launch_authority=False)
    finally:
        if root_fd is not None: os.close(root_fd)
        os.close(parent_fd)


def _layout(kind, model, bindings):
    if type(model) is not dict: raise ValueError('Exact model metadata required')
    if kind=='ridge':
        if set(model)!={'regularization'}: raise ValueError('Exact ridge model metadata required')
        g=len(bindings.ridge.gene_ids)
        return [('mean',np.dtype('float64'),(194,)),('scale',np.dtype('float64'),(194,)),
                ('intercept',np.dtype('float64'),(g,)),('coefficients',np.dtype('float64'),(194,g))]
    if kind!='abundance' or set(model)!={'regularization','target_ids','diagnostics'}:
        raise ValueError('Exact abundance model metadata required')
    targets=model['target_ids']
    if (not isinstance(targets,list) or not all(isinstance(t,str) and t for t in targets)
            or targets!=sorted(set(targets)) or len(targets)>len(bindings.abundance.catalog)):
        raise ValueError('Exact bounded target intercept inventory required')
    return [('mean',np.dtype('float64'),(100,)),('scale',np.dtype('float64'),(100,)),
            ('target_index',np.dtype('int64'),(len(bindings.abundance.catalog),)),
            ('parameters',np.dtype('float64'),(100+len(targets),))]


def load_reference(root, bindings, *, kind, expected_complete_sha256, producer_sha256,
                   fitting_receipt_sha256, output_cap_bytes, maximum_host_bytes):
    """Return private read-only parameters after exact identity and byte checks.

    Loading does not validate the external fit receipt's scientific claims.
    Expected bindings and all three content hashes must come from the admitted
    experiment, not be discovered by accepting whatever this directory says.
    """
    for value in (expected_complete_sha256,producer_sha256,fitting_receipt_sha256): _sha(value)
    _positive(output_cap_bytes,MAXIMUM_BYTES);_positive(maximum_host_bytes)
    expected_binding=_binding_record(bindings)
    if kind not in ('ridge','abundance'): raise ValueError('Exact reference kind required')
    root=_path(root);before=root.lstat()
    if not stat.S_ISDIR(before.st_mode) or before.st_uid!=os.geteuid(): raise ValueError('Owned reference directory required')
    root_fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);leases=[]
    try:
        if _stamp(os.fstat(root_fd))!=_stamp(before): raise ValueError('Reference directory changed while opening')
        cfile=_open_verified(root_fd,'COMPLETE.json',expected_complete_sha256,maximum=min(output_cap_bytes,MAXIMUM_METADATA_BYTES))
        leases.append(cfile);completion=_parse(cfile.stream.read())
        required={'schema','status','kind','manifest_sha256','manifest_bytes','array_files','output_cap_bytes','total_bytes',
                  'fitting_completion_verified','scientific_launch_authority'}
        if (set(completion)!=required or completion['schema']!=COMPLETE_SCHEMA or completion['status']!='publication_complete'
                or completion['kind']!=kind or type(completion['array_files']) is not int or completion['array_files']!=4
                or completion['fitting_completion_verified'] is not False or completion['scientific_launch_authority'] is not False):
            raise ValueError('Reference completion schema or scope differs')
        for key in ('manifest_bytes','total_bytes','output_cap_bytes'): _positive(completion[key],MAXIMUM_BYTES)
        if completion['total_bytes']>min(output_cap_bytes,completion['output_cap_bytes']): raise ValueError('Reference artifact exceeds bound')
        # Includes copied numeric arrays, hash buffers and conservative JSON-object overhead.
        if 4*completion['total_bytes']+16*completion['manifest_bytes']>maximum_host_bytes:
            raise ValueError('Reference load exceeds host bound before parameter allocation')
        mfile=_open_verified(root_fd,'MANIFEST.json',completion['manifest_sha256'],size=completion['manifest_bytes'],
                             maximum=min(output_cap_bytes,MAXIMUM_METADATA_BYTES))
        leases.append(mfile);metadata=_parse(mfile.stream.read())
        keys={'schema','kind','bindings','model','producer_sha256','fitting_receipt_sha256','arrays','inference_only','optimizer_resume_supported'}
        if (set(metadata)!=keys or metadata['schema']!=SCHEMA or metadata['kind']!=kind
                or metadata['bindings']!=expected_binding or metadata['producer_sha256']!=producer_sha256
                or metadata['fitting_receipt_sha256']!=fitting_receipt_sha256 or metadata['inference_only'] is not True
                or metadata['optimizer_resume_supported'] is not False):
            raise ValueError('Reference bindings, producer or fitting receipt differ')
        layout=_layout(kind,metadata['model'],bindings);records=metadata['arrays']
        if not isinstance(records,list) or len(records)!=4: raise ValueError('Complete reference array inventory required')
        names={'MANIFEST.json','COMPLETE.json'};plans=[]
        for record,(field,dtype,shape) in zip(records,layout):
            header=_header(dtype,shape);nbytes=int(np.prod(shape))*dtype.itemsize
            fixed=dict(population_index=0,field=field,relative_path='p000.'+field+'.npy',dtype=dtype.str,
                shape=list(shape),fortran_order=False,data_offset=len(header),nbytes=nbytes,file_bytes=len(header)+nbytes,
                header_sha256=hashlib.sha256(header).hexdigest())
            if (not isinstance(record,dict) or set(record)!=set(fixed)|{'array_sha256','file_sha256'}
                    or any(type(record[k]) is not int for k in ('population_index','data_offset','nbytes','file_bytes'))
                    or type(record['shape']) is not list or any(type(n) is not int for n in record['shape'])
                    or record['fortran_order'] is not False
                    or any(record[k]!=v for k,v in fixed.items())):
                raise ValueError('Reference numeric layout differs before parameter read')
            _sha(record['array_sha256']);_sha(record['file_sha256'])
            names.add(record['relative_path']);plans.append((record,field,dtype,shape,header,nbytes))
        total=cfile.size+mfile.size+sum(r['file_bytes'] for r in records)
        if total!=completion['total_bytes'] or set(os.listdir(root_fd))!=names:
            raise ValueError('Reference total bytes or exact directory inventory differs')
        arrays={}
        for record,field,dtype,shape,header,nbytes in plans:
            lease=_open_verified(root_fd,record['relative_path'],record['file_sha256'],size=record['file_bytes'],maximum=output_cap_bytes)
            leases.append(lease)
            if os.fstat(lease.stream.fileno()).st_uid!=os.geteuid(): raise ValueError('Reference parameter owner differs')
            if lease.stream.read(len(header))!=header: raise ValueError('Canonical non-pickle reference header differs')
            payload=lease.stream.read(nbytes)
            if len(payload)!=nbytes or lease.stream.read(1): raise ValueError('Reference numeric payload length differs')
            array=np.frombuffer(payload,dtype=dtype).reshape(shape).copy(order='C');array.setflags(write=False)
            if _array_record(0,field,array)!=record: raise ValueError('Decoded reference parameter identity differs')
            arrays[field]=array
        model_meta=metadata['model']
        if kind=='ridge':
            model=rr.RidgeReadout(bindings.ridge,model_meta['regularization'],arrays['mean'],arrays['scale'],arrays['intercept'],arrays['coefficients'])
        else:
            model=ah.AbundanceHead(bindings.abundance,arrays['mean'],arrays['scale'],tuple(model_meta['target_ids']),
                arrays['target_index'],arrays['parameters'],model_meta['regularization'],model_meta['diagnostics'])
        _parts(model,bindings)
        for lease in leases:
            if (os.fstat(lease.stream.fileno()).st_uid!=os.geteuid()
                    or _stamp(os.fstat(lease.stream.fileno()))!=lease.stamp
                    or _stamp(os.stat(lease.name,dir_fd=root_fd,follow_symlinks=False))!=lease.stamp
                    or _digest_stream(lease.stream)!=lease.sha256):
                raise ValueError('Reference files changed during loading')
        # Recheck all descriptors and names after hashing the last file as well.
        # Loaded arrays are independent copies; no file-backed views escape.
        for lease in leases:
            if (_stamp(os.fstat(lease.stream.fileno()))!=lease.stamp
                    or _stamp(os.stat(lease.name,dir_fd=root_fd,follow_symlinks=False))!=lease.stamp):
                raise ValueError('Reference files changed during final readback')
        if (_stamp(root.lstat())!=_stamp(before) or _stamp(os.fstat(root_fd))!=_stamp(before)
                or set(os.listdir(root_fd))!=names): raise ValueError('Reference root changed during loading')
        return model
    finally:
        for lease in leases: lease.stream.close()
        os.close(root_fd)
