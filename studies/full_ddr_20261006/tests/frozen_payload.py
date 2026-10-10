"""Historical packaging tests read sealed bytes, never rebuild an old launch."""
import hashlib
import json
from pathlib import Path


def frozen_payload(root):
    root = Path(root)
    build = json.loads((root / "BUILD.json").read_bytes())
    payload = {}
    for name, expected in build["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        raw = (root / "bundle" / relative).read_bytes()
        assert expected == dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        payload[name] = raw
    assert hashlib.sha256(payload["PLAN.json"]).hexdigest() == build["plan_sha256"]
    return payload
