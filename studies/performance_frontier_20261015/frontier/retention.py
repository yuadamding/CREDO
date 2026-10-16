"""Exact new-study rolling checkpoint-payload retention, under a live lease.

No directories, receipts, candidate exports or historical inputs are deleted.
Latest, previous, receipt-marked milestones and policy-marked milestones keep
their complete payloads. Immutable PREPARED/DELETED transaction records retain
the evidence of intentionally retired older payloads. Any partial transaction
requires an explicit external recovery decision; this module has no retry or
recovery operation. Logical bytes removed are not a live quota measurement.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .checkpoints import SCHEMA, TrajectoryLease


STUDY_ID = "credo_performance_frontier_20261015_v1"
POLICY_SCHEMA = "credo.frontier.checkpoint-retention-policy.v1"
PREPARED_SCHEMA = "credo.frontier.checkpoint-retirement-prepared.v1"
DELETED_SCHEMA = "credo.frontier.checkpoint-retirement-deleted.v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity_sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Exact lowercase SHA-256 required")
    return value


def _path(value, *, directory=False):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Explicit absolute non-escaping path required")
    for entry in (*reversed(path.parents), path):
        if entry.is_symlink():
            raise ValueError("Symlink retention paths are forbidden")
    if directory and not path.is_dir():
        raise ValueError("Existing regular directory required")
    return path


def _signature(info):
    return {name: getattr(info, name) for name in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_nlink")}


def _read_file(path, *, expected_sha=None, size=None, maximum_bytes=None):
    path = _path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid != os.geteuid()):
        raise ValueError("Owned regular single-link retention artifact required")
    if size is not None and before.st_size != size:
        raise ValueError("Retention artifact size mismatch")
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise ValueError("Retention receipt exceeds read budget")
    result = bytearray() if maximum_bytes is not None else None
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError("Retention artifact changed while opening")
        while block := stream.read(1024 * 1024):
            digest.update(block)
            if result is not None:
                result.extend(block)
        if _signature(path.lstat()) != _signature(before):
            raise ValueError("Retention artifact changed while reading")
    actual = digest.hexdigest()
    if expected_sha is not None and actual != _sha(expected_sha):
        raise ValueError("Retention artifact SHA-256 mismatch")
    return actual, _signature(before), None if result is None else bytes(result)


def _json(path):
    digest, _, payload = _read_file(path, maximum_bytes=16 * 1024 * 1024)
    return json.loads(payload), digest


def _sync(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path, value):
    path = _path(path)
    payload = _canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _sync(path.parent)
    digest = hashlib.sha256(payload).hexdigest()
    _, _, actual = _read_file(path, expected_sha=digest, size=len(payload), maximum_bytes=len(payload))
    if actual != payload or json.loads(actual) != value:
        raise ValueError("Published retention receipt failed readback")
    return digest


@dataclass(frozen=True)
class RetentionPolicy:
    study_root: str
    trajectory_root: str
    trajectory_identity_sha256: str
    explicit_milestone_updates: tuple[int, ...]
    study_id: str = STUDY_ID
    preserve_latest_payloads: int = 2

    def __post_init__(self):
        study = _path(self.study_root, directory=True)
        root = _path(self.trajectory_root, directory=True)
        if (self.study_id != STUDY_ID or study.name != STUDY_ID or root == study
                or not root.is_relative_to(study) or root.name in ("checkpoints", "claims", "retention")
                or type(self.preserve_latest_payloads) is not int or self.preserve_latest_payloads != 2):
            raise ValueError("Retention is restricted to an exact new-study trajectory and latest two payloads")
        _sha(self.trajectory_identity_sha256)
        if (type(self.explicit_milestone_updates) is not tuple
                or tuple(sorted(set(self.explicit_milestone_updates))) != self.explicit_milestone_updates
                or any(type(n) is not int or n < 0 for n in self.explicit_milestone_updates)):
            raise ValueError("Explicit sorted unique nonnegative milestone updates required")

    def record(self):
        result = asdict(self)
        result["explicit_milestone_updates"] = list(self.explicit_milestone_updates)
        return {"schema": POLICY_SCHEMA, **result}

    @property
    def identity(self):
        return identity_sha256(self.record())


def _authority(lease, policy, *, require_bound=True):
    if not isinstance(lease, TrajectoryLease) or not isinstance(policy, RetentionPolicy):
        raise TypeError("An active trajectory lease and explicit retention policy are required")
    lease.assert_active()
    root = _path(policy.trajectory_root, directory=True)
    _path(policy.study_root, directory=True)
    if root != lease.root or identity_sha256(lease.identity) != policy.trajectory_identity_sha256:
        raise ValueError("Retention root/trajectory identity differs from the active lease")
    # Revalidate the exact path boundary at every operation, not only at creation.
    RetentionPolicy(**asdict(policy))
    if require_bound:
        record, _ = _json(root / "retention" / "POLICY.json")
        if record != {"policy": policy.record(), "policy_sha256": policy.identity, "identity": lease.identity}:
            raise ValueError("Prebound retention policy changed")
    return root


def bind_policy(lease, policy):
    """Publish the policy before proposing payload retirement; never overwrite."""
    root = _authority(lease, policy, require_bound=False)
    directory = root / "retention"
    if not directory.exists():
        directory.mkdir()
        _sync(root)
    _path(directory, directory=True)
    record = {"policy": policy.record(), "policy_sha256": policy.identity, "identity": lease.identity}
    filename = directory / "POLICY.json"
    if filename.exists():
        saved, digest = _json(filename)
        if saved != record:
            raise ValueError("Existing retention policy is immutable")
    else:
        if any(directory.iterdir()):
            raise ValueError("Unbound retention contents require investigation")
        digest = _write_once(filename, record)
    transactions = directory / "transactions"
    if not transactions.exists():
        transactions.mkdir()
        _sync(directory)
    _path(transactions, directory=True)
    return {"policy_sha256": policy.identity, "receipt_sha256": digest}


def _history(lease, policy, *, permitted_prepared=None):
    """Validate all existing retirement evidence; unresolved state is a hard stop."""
    root = _authority(lease, policy)
    retired = {}
    for directory in sorted((root / "retention" / "transactions").iterdir()):
        _path(directory, directory=True)
        if not re.fullmatch(r"after-update-[0-9]{9}", directory.name):
            raise ValueError("Malformed retention transaction directory")
        prepared_path, deleted_path = directory / "PREPARED.json", directory / "DELETED.json"
        if not prepared_path.exists():
            raise RuntimeError("Incomplete retention transaction; no automatic recovery")
        prepared, prepared_sha = _json(prepared_path)
        if (prepared.get("schema") != PREPARED_SCHEMA or prepared.get("policy_sha256") != policy.identity
                or prepared.get("identity") != lease.identity or prepared.get("transaction") != directory.name):
            raise ValueError("Retention transaction binding changed")
        if not deleted_path.exists():
            if directory.name != permitted_prepared:
                raise RuntimeError("Prepared/partial retirement requires explicit recovery; no automatic continuation")
            continue
        deleted, _ = _json(deleted_path)
        if (deleted != {"schema": DELETED_SCHEMA, "policy_sha256": policy.identity,
                       "identity": lease.identity, "prepared_sha256": prepared_sha,
                       "transaction": directory.name, "removed": prepared["candidates"],
                       "removed_payload_bytes": sum(item["size_bytes"] for item in prepared["candidates"]),
                       "payload_absence_verified": True, "quota_reclaim_verified": False}):
            raise ValueError("Deletion receipt does not reconcile exactly with its preparation")
        for item in prepared["candidates"]:
            relative = item["relative_uri"]
            if relative in retired or not re.fullmatch(r"checkpoints/update-[0-9]{9}/checkpoint\.pt", relative):
                raise ValueError("Repeated or non-checkpoint retirement member")
            path = _path(root / relative)
            if path.exists() or path.is_symlink():
                raise ValueError("A recorded retired payload unexpectedly exists")
            retired[relative] = item
    return retired


def _inventory(lease, policy, *, permitted_prepared=None):
    root = _authority(lease, policy)
    retired = _history(lease, policy, permitted_prepared=permitted_prepared)
    records = []
    for directory in sorted((root / "checkpoints").iterdir()):
        _path(directory, directory=True)
        if not re.fullmatch(r"update-[0-9]{9}", directory.name):
            raise ValueError("Malformed checkpoint directory; nothing may be retired")
        receipt, receipt_sha = _json(directory / "COMPLETE.json")
        update = receipt.get("accepted_updates")
        if (receipt.get("schema") != SCHEMA or receipt.get("identity") != lease.identity
                or type(update) is not int or update < 0 or directory.name != f"update-{update:09d}"
                or receipt.get("attempted_updates") != update or type(receipt.get("milestone")) is not bool
                or type(receipt.get("checkpoint_bytes")) is not int or receipt["checkpoint_bytes"] < 1
                or receipt.get("execution_scope") not in ("scientific", "synthetic_cpu_only", "synthetic_cuda_qualification")
                or not isinstance(receipt.get("phase"), str) or not receipt["phase"]):
            raise ValueError("Malformed or changed checkpoint completion receipt")
        _sha(receipt["checkpoint_sha256"])
        relative = f"checkpoints/{directory.name}/checkpoint.pt"
        item = {"relative_uri": relative, "accepted_updates": update, "sha256": receipt["checkpoint_sha256"],
                "size_bytes": receipt["checkpoint_bytes"], "complete_receipt_sha256": receipt_sha}
        if relative in retired:
            expected = {key: value for key, value in retired[relative].items() if key != "file_signature"}
            if item != expected:
                raise ValueError("Retired payload completion evidence changed")
            item["retired"] = True
        else:
            _, signature, _ = _read_file(root / relative, expected_sha=item["sha256"], size=item["size_bytes"])
            item["file_signature"] = signature
            item["retired"] = False
        item["milestone"] = receipt["milestone"] or update in policy.explicit_milestone_updates
        records.append(item)
    records.sort(key=lambda row: row["accepted_updates"])
    if set(retired) - {item["relative_uri"] for item in records}:
        raise ValueError("Retired checkpoint completion receipt was removed")
    keep = {row["accepted_updates"] for row in records[-2:]} | {row["accepted_updates"] for row in records if row["milestone"]}
    if any(row["retired"] and row["accepted_updates"] in keep for row in records):
        raise ValueError("Required latest/previous/milestone payload has been retired")
    candidates = [{key: value for key, value in row.items() if key not in ("milestone", "retired")}
                  for row in records if not row["retired"] and row["accepted_updates"] not in keep]
    snapshot = [{key: value for key, value in row.items() if key != "file_signature"} for row in records]
    return {"inventory": snapshot, "retained_updates": sorted(keep), "candidates": candidates,
            "latest_update": records[-1]["accepted_updates"] if records else None}


@dataclass(frozen=True)
class PreparedRetirement:
    transaction: str
    prepared_sha256: str
    policy_sha256: str


def prepare_retirement(lease, policy):
    """Verify retained and proposed payloads, then seal an exact deletion plan."""
    current = _inventory(lease, policy)
    if not current["candidates"]:
        return None
    transaction = f"after-update-{current['latest_update']:09d}"
    directory = lease.root / "retention" / "transactions" / transaction
    directory.mkdir()  # An existing/incomplete generation is never silently reused.
    _sync(directory.parent)
    record = {"schema": PREPARED_SCHEMA, "policy_sha256": policy.identity, "identity": lease.identity,
              "transaction": transaction, **current}
    digest = _write_once(directory / "PREPARED.json", record)
    return PreparedRetirement(transaction, digest, policy.identity)


def _unlink_exact(root, item):
    """Retire one validated name, with no payload descriptor held at unlink.

    An open-unlinked file need not report nlink=0 on a remote filesystem (for
    example, NFS can temporarily rename it). The retirement contract is exact
    pathname absence, not an assertion about physical quota reclamation. Close
    our payload handle before unlink, then revalidate the name and parent and
    verify absence through both the pinned directory and the canonical path.
    The live trajectory lease serializes cooperative writers. As with the
    original stat/unlink path, this is not protection against a hostile process
    running as the same UID between the final stat and unlink.
    """
    relative = item["relative_uri"]
    if not re.fullmatch(r"checkpoints/update-[0-9]{9}/checkpoint\.pt", relative):
        raise ValueError("Only exact checkpoint.pt payload members may be removed")
    path = _path(root / relative)
    _, signature, _ = _read_file(path, expected_sha=item["sha256"], size=item["size_bytes"])
    if signature != item["file_signature"]:
        raise ValueError("Prepared payload inode/signature changed")
    parent_info = path.parent.stat()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened_parent = os.fstat(descriptor)
        if (parent_info.st_dev, parent_info.st_ino) != (opened_parent.st_dev, opened_parent.st_ino):
            raise ValueError("Checkpoint directory changed during retirement")
        payload_fd = os.open("checkpoint.pt", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            if (_signature(os.fstat(payload_fd)) != signature
                    or _signature(os.stat("checkpoint.pt", dir_fd=descriptor, follow_symlinks=False)) != signature):
                raise ValueError("Payload changed immediately before retirement")
        finally:
            os.close(payload_fd)
        # No open payload descriptor remains in this routine. Keep the complete
        # pre-unlink fingerprint, including owner and single-link requirement.
        if _signature(os.stat("checkpoint.pt", dir_fd=descriptor, follow_symlinks=False)) != signature:
            raise ValueError("Payload changed after validation handle closed")
        expected_parent = (opened_parent.st_dev, opened_parent.st_ino, opened_parent.st_uid)
        def parent_unchanged():
            current = _path(path.parent, directory=True).stat()
            if (current.st_dev, current.st_ino, current.st_uid) != expected_parent:
                raise ValueError("Checkpoint directory changed at retirement boundary")
        parent_unchanged()
        os.unlink("checkpoint.pt", dir_fd=descriptor)
        os.fsync(descriptor)
        parent_unchanged()
        try:
            os.stat("checkpoint.pt", dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Payload absence readback failed through pinned directory")
        if os.path.lexists(path):
            raise ValueError("Payload absence readback failed through canonical path")
    finally:
        os.close(descriptor)


def commit_retirement(lease, policy, prepared):
    """Execute one prepared exact plan; any partial result blocks future calls."""
    root = _authority(lease, policy)
    if (not isinstance(prepared, PreparedRetirement) or prepared.policy_sha256 != policy.identity
            or not re.fullmatch(r"after-update-[0-9]{9}", prepared.transaction)):
        raise ValueError("Exact bound prepared-retirement identity required")
    directory = root / "retention" / "transactions" / prepared.transaction
    _path(directory, directory=True)
    if (directory / "DELETED.json").exists():
        raise FileExistsError("Retirement already completed; no deletion replay")
    record, digest = _json(directory / "PREPARED.json")
    if digest != _sha(prepared.prepared_sha256):
        raise ValueError("Prepared retirement receipt hash mismatch")
    current = _inventory(lease, policy, permitted_prepared=prepared.transaction)
    expected = {"schema": PREPARED_SCHEMA, "policy_sha256": policy.identity, "identity": lease.identity,
                "transaction": prepared.transaction, **current}
    if record != expected or not record["candidates"]:
        raise ValueError("Prepared inventory/retention set changed; no payload deleted")
    for item in record["candidates"]:
        _authority(lease, policy)
        _unlink_exact(root, item)
    deleted = {"schema": DELETED_SCHEMA, "policy_sha256": policy.identity, "identity": lease.identity,
               "prepared_sha256": digest, "transaction": prepared.transaction, "removed": record["candidates"],
               "removed_payload_bytes": sum(item["size_bytes"] for item in record["candidates"]),
               "payload_absence_verified": True, "quota_reclaim_verified": False}
    deleted_sha = _write_once(directory / "DELETED.json", deleted)
    # Reconcile the immutable evidence and absence once more after publication.
    _history(lease, policy)
    return {"transaction": prepared.transaction, "deleted_receipt_sha256": deleted_sha, **deleted}
