"""Local source/package checks only; no numerical execution or remote work."""

from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import pytest

from deployment.main_source import construct, main
from frontier.main_worker_source import source_closure, manifest_payload


ROOT = Path(__file__).resolve().parents[1]


def test_actual_complete_main_import_closure_and_stable_archive():
    source = source_closure(ROOT)
    required = ("main_worker", "main_worker_contract", "main_worker_source", "fitting_execution",
        "fitting_inputs", "fitting_resources", "fitting_stage", "models", "training_owner",
        "training_objective", "continuation_custody", "observation_worker")
    assert all("frontier/" + name + ".py" in source for name in required)
    assert all(name.startswith("frontier/") and name.endswith(".py") for name in source)
    first, receipt, payload = construct(ROOT)
    assert construct(ROOT)[0] == first
    assert receipt["source_manifest_sha256"] == sha256(manifest_payload(source)).hexdigest()
    assert not receipt["scientific_launch_authority"] and not receipt["cuda_qualification"]
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        assert set(archive.getnames()) == {"bundle/" + name for name in payload}
        for member in archive:
            assert member.isfile()
            assert archive.extractfile(member).read() == payload[member.name[len("bundle/"):]]


def test_local_package_publishes_matching_manifest_and_never_clobbers(tmp_path, capsys):
    output = tmp_path / "main-source"
    args = ["--study-root", str(ROOT), "--output", str(output)]
    assert main(args) == 0
    reported = json.loads(capsys.readouterr().out)
    assert sha256((output / "SOURCE_MANIFEST.json").read_bytes()).hexdigest() == reported["source_manifest_sha256"]
    assert sha256((output / "main-source.tar.gz").read_bytes()).hexdigest() == reported["archive_sha256"]
    with pytest.raises(FileExistsError): main(args)


def test_discovery_rejects_linked_source_before_archiving(tmp_path):
    (tmp_path / "frontier").mkdir()
    (tmp_path / "frontier/__init__.py").write_text("")
    (tmp_path / "frontier/main_worker.py").symlink_to(ROOT / "frontier/main_worker.py")
    with pytest.raises(ValueError, match="single-link"):
        source_closure(tmp_path)


def test_deferred_relative_import_is_included(tmp_path):
    package = tmp_path / "frontier"; package.mkdir()
    (package / "__init__.py").write_text("# fixture\n")
    (package / "main_worker.py").write_text("def later():\n    from .child import value\n")
    (package / "child.py").write_text("value = 1\n")
    assert set(source_closure(tmp_path)) == {"frontier/__init__.py", "frontier/main_worker.py", "frontier/child.py"}


def test_actual_source_leases_reject_post_entry_file_change(tmp_path, monkeypatch):
    # Run a fresh metadata-only interpreter to avoid pytest's unrelated imports.
    # Entry pathname is simulated; this is not CUDA/production-entry proof.
    import os
    import subprocess
    import sys
    source = source_closure(ROOT)
    for name, raw in source.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    manifest = manifest_payload(source)
    (tmp_path / "SOURCE_MANIFEST.json").write_bytes(manifest)
    program = """
import sys
from pathlib import Path
from frontier.main_worker_source import verified_sources
from frontier.observation_loader import ObservationFile
root = Path(sys.argv[1])
sys.modules['__main__'].__file__ = str(root/'frontier/main_worker.py')
ref = ObservationFile(str(root/'SOURCE_MANIFEST.json'), sys.argv[2], int(sys.argv[3]))
with verified_sources(root, ref):
    pass
try:
    with verified_sources(root, ref):
        (root/'frontier/main_worker.py').touch()
except ValueError as error:
    assert 'changed' in str(error)
else:
    raise AssertionError('Changed source accepted')
print('metadata-only-source-entry-exit-passed')
"""
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path), sha256(manifest).hexdigest(), str(len(manifest))],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(tmp_path)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "metadata-only-source-entry-exit-passed"
