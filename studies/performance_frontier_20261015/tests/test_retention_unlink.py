"""Literal-file retirement regressions. No models, data or optimizer updates."""

import os
from pathlib import Path
import ast
import hashlib
from types import SimpleNamespace

import pytest

from frontier import retention


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / 'checkpoints/update-000000768/checkpoint.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'disposable-literal-checkpoint')
    (path.parent/'COMPLETE.json').write_bytes(b'{}\n')
    digest, signature, _ = retention._read_file(path)
    return tmp_path, path, {'relative_uri':str(path.relative_to(tmp_path)),
        'sha256':digest, 'size_bytes':path.stat().st_size, 'file_signature':signature}


def track_payload_descriptors(monkeypatch):
    opened = set()
    original_open, original_close = os.open, os.close
    def open_file(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        # Track the explicit directory-relative descriptor. The earlier hash
        # reader uses fdopen(), whose close is internal to the Python stream.
        if path == 'checkpoint.pt':
            opened.add(fd)
        return fd
    def close_file(fd):
        original_close(fd)
        opened.discard(fd)
    monkeypatch.setattr(os, 'open', open_file)
    monkeypatch.setattr(os, 'close', close_file)
    return opened


def test_retirement_closes_own_file_handles_before_unlink(payload, monkeypatch):
    root, path, item = payload
    opened = track_payload_descriptors(monkeypatch)
    original = os.unlink
    calls = []
    def guarded_unlink(name, *args, **kwargs):
        # An open-unlink backend may keep a temporary link. The corrected
        # implementation does not trigger that backend-specific path itself.
        assert not opened, 'payload still open at unlink (old implementation)'
        calls.append(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(os, 'unlink', guarded_unlink)
    retention._unlink_exact(root, item)
    assert calls == ['checkpoint.pt'] and not opened
    assert not path.exists() and (path.parent/'COMPLETE.json').read_bytes() == b'{}\n'


@pytest.mark.parametrize('damage', ['content', 'size', 'hardlink', 'symlink', 'replacement'])
def test_preunlink_identity_and_integrity_guards_remain(payload, monkeypatch, damage):
    root, path, item = payload
    if damage == 'content':
        path.write_bytes(b'x'*item['size_bytes'])
    elif damage == 'size':
        path.write_bytes(b'x')
    elif damage == 'hardlink':
        os.link(path, path.parent/'alias')
    elif damage == 'symlink':
        path.rename(path.parent/'other')
        path.symlink_to(path.parent/'other')
    else:
        path.rename(path.parent/'other')
        path.write_bytes((path.parent/'other').read_bytes())
    monkeypatch.setattr(os, 'unlink', lambda *a, **k: pytest.fail('Must fail before deletion'))
    with pytest.raises(ValueError):
        retention._unlink_exact(root, item)


def test_rechecks_name_after_payload_handle_closes(payload, monkeypatch):
    root, path, item = payload
    original_open, original_close = os.open, os.close
    payload_fd = []
    def open_file(name, flags, *args, **kwargs):
        fd = original_open(name, flags, *args, **kwargs)
        if name == 'checkpoint.pt':
            payload_fd.append(fd)
        return fd
    def close_file(fd):
        original_close(fd)
        if payload_fd and fd == payload_fd[-1]:
            payload_fd.clear()
            path.rename(path.parent/'original')
            path.write_bytes(b'replacement')
    monkeypatch.setattr(os, 'open', open_file)
    monkeypatch.setattr(os, 'close', close_file)
    monkeypatch.setattr(os, 'unlink', lambda *a, **k: pytest.fail('Replacement must survive'))
    with pytest.raises(ValueError, match='after validation handle closed'):
        retention._unlink_exact(root, item)
    assert path.read_bytes() == b'replacement'


@pytest.mark.parametrize('replacement', ['file', 'broken_symlink', 'same_parent_renamed'])
def test_postunlink_drift_never_reports_success(payload, monkeypatch, replacement):
    root, path, item = payload
    original = os.unlink
    def changed(name, *args, **kwargs):
        original(name, *args, **kwargs)
        if replacement == 'file':
            path.write_bytes(b'unexpected')
        elif replacement == 'broken_symlink':
            path.symlink_to(path.parent/'nonexistent')
        else:
            path.parent.rename(path.parent.with_name('displaced'))
            path.parent.mkdir()
    monkeypatch.setattr(os, 'unlink', changed)
    with pytest.raises(ValueError, match='absence|directory changed'):
        retention._unlink_exact(root, item)


def test_failed_unlink_closes_descriptors_and_preserves_file(payload, monkeypatch):
    root, path, item = payload
    opened = track_payload_descriptors(monkeypatch)
    def denied(*args, **kwargs):
        raise PermissionError('fabricated permission denial')
    monkeypatch.setattr(os, 'unlink', denied)
    with pytest.raises(PermissionError):
        retention._unlink_exact(root, item)
    assert not opened and path.exists()


def test_wrong_member_cannot_delete_other_path(payload, monkeypatch):
    root, path, item = payload
    monkeypatch.setattr(os, 'unlink', lambda *a, **k: pytest.fail('No deletion'))
    with pytest.raises(ValueError, match='exact checkpoint'):
        retention._unlink_exact(root, {**item, 'relative_uri':'COMPLETE.json'})


@pytest.mark.parametrize('legacy', [True, False])
def test_open_unlink_linkcount_negative_control(payload, monkeypatch, legacy):
    """Simulated remote-filesystem result, not a claim about the actual mount."""
    root,path,item = payload
    original_open, original_close, original_fstat = os.open, os.close, os.fstat
    tracked = set()
    def open_file(name, flags, *args, **kwargs):
        fd = original_open(name,flags,*args,**kwargs)
        if name=='checkpoint.pt':tracked.add(fd)
        return fd
    def close_file(fd):
        original_close(fd)
        tracked.discard(fd)
    def fstat(fd):
        value = original_fstat(fd)
        if fd in tracked and not path.exists():
            # Retain inode/size and report a remaining link while open.
            fields = {k:getattr(value,k) for k in dir(value) if k.startswith('st_')}
            fields['st_nlink']=1
            return SimpleNamespace(**fields)
        return value
    monkeypatch.setattr(os,'open',open_file)
    monkeypatch.setattr(os,'close',close_file)
    monkeypatch.setattr(os,'fstat',fstat)
    operation = retention._unlink_exact
    if legacy:
        source = (Path(__file__).resolve().parents[1]/
                  'observation_d1d2_v2_launch/bundle/source/frontier/retention.py').read_bytes()
        assert hashlib.sha256(source).hexdigest()=='e721ba8f81635de0e04aece8e3afd9fafa3d0bd0511d34c2e7526552b719c98a'
        node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_unlink_exact')
        namespace = dict(vars(retention))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-retirement-function>','exec'),namespace)
        operation = namespace['_unlink_exact']
        with pytest.raises(ValueError,match='Unlinked payload inode readback failed'):
            operation(root,item)
    else:
        operation(root,item)
    assert not path.exists() and not tracked
