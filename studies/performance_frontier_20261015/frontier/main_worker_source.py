"""Exact import closure for the main worker, without importing numerical code.

The allowlist follows literal in-package Python imports, including deferred
imports. Pinned external readers remain separately verified by the input owner.
No data, tests, historical repositories, or remote paths are searched.
"""

import ast
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys


SCHEMA = "credo.frontier.main-worker-source.v1"
ENTRY = "frontier/main_worker.py"
QUALIFICATION_ENTRY = "frontier/main_qualification_worker.py"
MAXIMUM_SOURCE_BYTES = 2 * 1024**2
MAXIMUM_TOTAL_BYTES = 16 * 1024**2


def _entry_schema(entry):
    if entry == ENTRY:
        return SCHEMA
    if entry == QUALIFICATION_ENTRY:
        return "credo.frontier.main-qualification-source.v1"
    raise ValueError("Only the fixed main or qualification entry is admitted")


def source_closure(root, *, entry=ENTRY):
    _entry_schema(entry)
    root = Path(root)
    pending, source = {entry, "frontier/__init__.py"}, {}
    while pending:
        name = min(pending); pending.remove(name)
        if name in source:
            continue
        path = root / name
        if (not root.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
                or not path.is_file() or path.stat().st_size > MAXIMUM_SOURCE_BYTES
                or path.stat().st_nlink != 1):
            raise ValueError("Bounded regular single-link worker source required")
        raw = path.read_bytes()
        if len(raw) > MAXIMUM_SOURCE_BYTES:
            raise ValueError("Source changed beyond per-file allowance")
        source[name] = raw
        if sum(map(len, source.values())) > MAXIMUM_TOTAL_BYTES:
            raise ValueError("Main worker source closure exceeds package allowance")
        for node in ast.walk(ast.parse(raw, filename=name)):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level != 1 or "/" in name[len("frontier/"):]:
                        raise ValueError("Unregistered relative package shape")
                    modules = ["frontier." + node.module] if node.module else ["frontier." + a.name for a in node.names]
                elif node.module and node.module.startswith("frontier"):
                    modules = [node.module]
                    if node.module == "frontier":
                        modules += ["frontier." + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names if a.name == "frontier" or a.name.startswith("frontier.")]
            for module in modules:
                member = "frontier/__init__.py" if module == "frontier" else module.replace(".", "/") + ".py"
                pending.add(member)
    return dict(sorted(source.items()))


def manifest_payload(source, *, entry=ENTRY):
    return (json.dumps(dict(schema=_entry_schema(entry), entry=entry,
        files={k: dict(sha256=sha256(v).hexdigest(), size_bytes=len(v)) for k, v in sorted(source.items())}),
        sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


@contextmanager
def verified_sources(root, reference, *, entry=ENTRY):
    # These deferred imports are part of the closure but do not affect the
    # builder's CPU-only discovery. Leases cover the complete worker execution.
    from .observation_loader import ObservationFile, _verified_file
    from .main_worker_contract import parse
    root = Path(root)
    with ExitStack() as stack:
        payload = stack.enter_context(_verified_file(reference, maximum=MAXIMUM_SOURCE_BYTES)).read()
        manifest = parse(payload)
        if (set(manifest) != {"schema", "entry", "files"} or manifest["schema"] != _entry_schema(entry)
                or manifest["entry"] != entry or type(manifest["files"]) is not dict):
            raise ValueError("Exact main-worker source manifest required")
        source = source_closure(root, entry=entry)
        if payload != manifest_payload(source, entry=entry):
            raise ValueError("Main worker source manifest differs from complete actual import closure")
        for name, raw in source.items():
            ref = ObservationFile(str(root / name), sha256(raw).hexdigest(), len(raw))
            stack.enter_context(_verified_file(ref, maximum=MAXIMUM_SOURCE_BYTES))
        def check_loaded():
            for name, module in tuple(sys.modules.items()):
                if name != "frontier" and not name.startswith("frontier."):
                    continue
                relative = "frontier/__init__.py" if name == "frontier" else name.replace(".", "/") + ".py"
                # Other imported frontier modules are not acceptable in this
                # fresh production process, even when they are not yet called.
                if relative not in source or Path(getattr(module, "__file__", "")).absolute() != root / relative:
                    raise ValueError("Imported frontier module is outside the bound worker closure")
            loaded_entry = Path(sys.modules["__main__"].__file__).absolute()
            if loaded_entry != root / entry:
                raise ValueError("Main worker must run as its bound module entry")
        check_loaded()
        yield manifest
        check_loaded()
