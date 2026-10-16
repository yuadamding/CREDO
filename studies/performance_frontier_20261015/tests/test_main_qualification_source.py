"""Immutable code-only bundle; source-entry checks do not execute CUDA."""

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from deployment.main_qualification_source import construct,main
from frontier.main_worker_source import source_closure,manifest_payload,QUALIFICATION_ENTRY
from test_native_inputs import READER


ROOT=Path(__file__).resolve().parents[1]


def test_distinct_parent_closure_includes_unchanged_child_and_no_extra_data():
    source=source_closure(ROOT,entry=QUALIFICATION_ENTRY)
    child=source_closure(ROOT)
    assert all(source[k]==v for k,v in child.items())
    assert "frontier/bounded_process.py" in source and QUALIFICATION_ENTRY in source
    assert QUALIFICATION_ENTRY not in child
    assert json.loads(manifest_payload(child))["schema"]=="credo.frontier.main-worker-source.v1"
    assert json.loads(manifest_payload(source,entry=QUALIFICATION_ENTRY))["schema"]=="credo.frontier.main-qualification-source.v1"
    with pytest.raises(ValueError):source_closure(ROOT,entry="frontier/models.py")
    archive,record,payload=construct(ROOT,READER)
    assert construct(ROOT,READER)[0]==archive
    assert record["optimizer_calls"]==0 and not record["disposable_launch_authority"]
    assert not record["biological_payloads_included"]
    with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as files:
        assert set(files.getnames())=={"bundle/"+n for n in payload}
        for entry in files:
            assert entry.isfile() and files.extractfile(entry).read()==payload[entry.name[len("bundle/"):]]


def test_bundle_publication_and_no_overwrite(tmp_path,capsys):
    args=["--study-root",str(ROOT),"--reader-source-root",str(READER),"--output",str(tmp_path/"package")]
    assert main(args)==0
    reported=json.loads(capsys.readouterr().out)
    output=tmp_path/"package"
    assert sha256((output/"main-qualification-source.tar.gz").read_bytes()).hexdigest()==reported["archive_sha256"]
    assert reported["worker_source_manifest_sha256"]==sha256(manifest_payload(source_closure(ROOT))).hexdigest()
    with pytest.raises(FileExistsError):main(args)


def test_actual_parent_entry_source_lease_in_fresh_metadata_process(tmp_path):
    source=source_closure(ROOT,entry=QUALIFICATION_ENTRY)
    for name,raw in source.items():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
    manifest=manifest_payload(source,entry=QUALIFICATION_ENTRY)
    (tmp_path/"SOURCE_MANIFEST.json").write_bytes(manifest)
    program="""
import sys
from pathlib import Path
import frontier.main_qualification_worker
from frontier.main_worker_source import verified_sources, QUALIFICATION_ENTRY
from frontier.observation_loader import ObservationFile
root=Path(sys.argv[1]);sys.modules['__main__'].__file__=str(root/QUALIFICATION_ENTRY)
ref=ObservationFile(str(root/'SOURCE_MANIFEST.json'),sys.argv[2],int(sys.argv[3]))
with verified_sources(root,ref,entry=QUALIFICATION_ENTRY): pass
try:
    with verified_sources(root,ref,entry=QUALIFICATION_ENTRY):
        (root/QUALIFICATION_ENTRY).touch()
except ValueError as error: assert 'changed' in str(error)
else: raise AssertionError('Source exit change accepted')
print('metadata-only-parent-source-entry-exit-passed')
"""
    result=subprocess.run([sys.executable,"-c",program,str(tmp_path),sha256(manifest).hexdigest(),str(len(manifest))],
        cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path)),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="metadata-only-parent-source-entry-exit-passed"
