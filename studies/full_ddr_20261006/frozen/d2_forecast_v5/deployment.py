"""Stdlib-only B1 archive validation and no-overwrite publication.

This module performs no mutation on import. The controller explicitly calls
publish_archive only after approved-chain/session preflight and authorization.
"""
import hashlib
import ctypes
import errno
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tarfile


def rename_noreplace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("renameat2 unavailable; no overwrite-capable publication fallback")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def publication_canary(root):
    """Qualify actual fileset no-replace and exact owned-empty removability."""
    source, destination = root / "rename-canary-source", root / "rename-canary-destination"
    try:
        source.mkdir(mode=0o700)
        rename_noreplace(source, destination)
        source.mkdir(mode=0o700)
        try:
            rename_noreplace(source, destination)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
        else:
            raise RuntimeError("Fileset rename unexpectedly replaced an existing destination")
        for path in (source, destination):
            if path.is_symlink() or path.stat().st_uid != os.getuid() or list(path.iterdir()):
                raise RuntimeError("Owned rename canary changed")
            path.rmdir()
    except BaseException as error:
        with (root / "PUBLICATION_FAILURE.json").open("x") as stream:
            json.dump(dict(status="failed_before_bundle_write", error=repr(error),
                source_exists=source.exists(), destination_exists=destination.exists(),
                original_inputs_unchanged=True, no_fallback_attempted=True), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        raise


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_file(path, raw):
    with os.fdopen(os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fchmod(stream.fileno(), 0o400)
        os.fsync(stream.fileno())
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Exclusive file descriptor type/owner differs: " + str(path))
        if stat.S_IMODE(info.st_mode) != 0o400:
            raise ValueError("Exclusive file fchmod did not establish mode0400: " + str(path))
        if info.st_size != len(raw):
            raise ValueError("Exclusive file descriptor size differs: " + str(path))


def set_directory_mode(path, mode):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode:
            raise ValueError("Directory fchmod/type/owner verification differs: " + str(path))
    finally:
        os.close(descriptor)


def permission_canary(root):
    """Test explicit post-creation file and directory mode changes before data."""
    directory = root / "publication-permissions-canary"
    directory.mkdir(mode=0o700)
    path = directory / "payload"
    raw = b"B1r3-permission-readback\n"
    exclusive_file(path, raw)
    set_directory_mode(directory, 0o500)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o400 or path.read_bytes() != raw:
        raise ValueError("Read-only permission canary final readback differs")
    if list(directory.iterdir()) != [path]:
        raise ValueError("Permission canary exact membership differs")
    # Restore only this newly created, exactly verified owned canary directory.
    set_directory_mode(directory, 0o700)
    path.unlink()
    directory.rmdir()
    fsync_directory(root)


def atomic_receipt(path, raw):
    """Publish fully flushed bytes with atomic, no-replace hard-link creation."""
    temporary = path.with_name(path.name + ".publishing")
    exclusive_file(temporary, raw)
    os.link(str(temporary), str(path), follow_symlinks=False)
    fsync_directory(path.parent)
    temporary.unlink()
    fsync_directory(path.parent)


def receipt_link_canary(root):
    """Qualify actual same-fileset link/no-replace/readback/removability."""
    source = root / "receipt-link-canary-source"
    destination = root / "receipt-link-canary-destination"
    raw = b"B1r2-exclusive-receipt-canary\n"
    exclusive_file(source, raw)
    os.link(str(source), str(destination), follow_symlinks=False)
    fsync_directory(root)
    try:
        os.link(str(source), str(destination), follow_symlinks=False)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
    else:
        raise RuntimeError("Hard-link publication did not reject collision")
    for path in (source, destination):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o400 or path.read_bytes() != raw:
            raise RuntimeError("Receipt canary owner/type/readback differs")
    if source.stat().st_ino != destination.stat().st_ino:
        raise RuntimeError("Receipt canary did not create one linked inode")
    source.unlink()
    destination.unlink()
    fsync_directory(root)


def publish_reserved_bundle(root, payload, build, *, root_canary_qualified=False):
    """Skill-authorized GPFS fallback: exclusive final directory and files.

    No rename is attempted. The final path alone is never a readiness signal;
    DEPLOYMENT.json is linked only after complete durable inventory readback.
    Any failure preserves the reserved root and no success receipt is issued.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.stat().st_uid != os.getuid():
        raise ValueError("Reserved publication owner/type differs")
    if not root_canary_qualified:
        permission_canary(root)
        receipt_link_canary(root)
    final = root / "bundle"
    final.mkdir(mode=0o700)
    directories = {final}
    for name, raw in sorted(payload.items()):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name or name in ("", "."):
            raise ValueError("Noncanonical publication member")
        parent = final
        for part in relative.parts[:-1]:
            parent = parent / part
            if parent not in directories:
                parent.mkdir(mode=0o700)
                directories.add(parent)
        exclusive_file(final / relative, raw)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        set_directory_mode(directory, 0o500)
    seen_files, seen_directories = set(), {final}
    for path in final.rglob("*"):
        info = path.lstat()
        if info.st_uid != os.getuid():
            raise ValueError("Published owner differs")
        if stat.S_ISDIR(info.st_mode):
            if stat.S_IMODE(info.st_mode) != 0o500:
                raise ValueError("Final directory mode differs: " + str(path))
            seen_directories.add(path)
        elif stat.S_ISREG(info.st_mode):
            name = str(path.relative_to(final))
            if name not in payload:
                raise ValueError("Unexpected final inventory member: " + name)
            if path.read_bytes() != payload[name]:
                raise ValueError("Final inventory bytes differ: " + name)
            if stat.S_IMODE(info.st_mode) != 0o400:
                raise ValueError("Final file mode differs: " + name)
            seen_files.add(name)
        else:
            raise ValueError("Unexpected special member in reserved final bundle")
    if seen_files != set(payload) or seen_directories != directories:
        raise ValueError("Final exact file/directory inventory differs")
    fsync_directory(root)
    receipt = dict(status="published_verified_B1_bundle", root=str(root),
        inventory_sha256=build["inventory_sha256"], plan_sha256=build["plan_sha256"],
        archive_sha256=build["archive_sha256"], files=len(payload) - 1,
        original_data_retained=True, raw_counts_copied=False,
        exclusive_final_directory=True, exclusive_nofollow_files=True,
        complete_inventory_readback=True, receipt_link_canary_passed=True,
        explicit_file_fchmod_verified=True, directory_mode_verified=True,
        publication="exclusive_reserved_directory_manifest_last_hardlink",
        overwrite_capable_rename_used=False, incomplete_directory_is_not_ready=True)
    atomic_receipt(root / "DEPLOYMENT.json", (json.dumps(receipt, sort_keys=True) + "\n").encode())
    return receipt


def publish_both_roots(roots, payload, build):
    """All root canaries must pass before either root receives bundle bytes."""
    for root in roots:
        root.mkdir(mode=0o700)
        permission_canary(root)
        receipt_link_canary(root)
    return [publish_reserved_bundle(root, payload, build, root_canary_qualified=True) for root in roots]


def archive_payload(raw, archive_sha256, inventory_sha256):
    if sha(raw) != archive_sha256 or len(raw) > 100_000_000:
        raise ValueError("Archive hash/size differs")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len({item.name for item in members}):
            raise ValueError("Duplicate archive members")
        for item in members:
            path = Path(item.name)
            if (not item.isfile() or path.is_absolute() or ".." in path.parts
                    or len(path.parts) < 2 or path.parts[0] != "bundle"
                    or str(path) != item.name):
                raise ValueError("Unsafe archive member")
        if sum(item.size for item in members) > 100_000_000:
            raise ValueError("Unpacked bundle exceeds fixed source bound")
        payload = {str(Path(item.name).relative_to("bundle")): archive.extractfile(item).read()
                   for item in members}
    inventory_raw = payload["INVENTORY.json"]
    if sha(inventory_raw) != inventory_sha256:
        raise ValueError("Bundle inventory identity differs")
    inventory = json.loads(inventory_raw)
    if set(payload) != set(inventory) | {"INVENTORY.json"}:
        raise ValueError("Archive inventory coverage differs")
    for name, binding in inventory.items():
        if len(payload[name]) != binding["size_bytes"] or sha(payload[name]) != binding["sha256"]:
            raise ValueError("Bundle payload hash/size differs")
    return payload, inventory


def _amount(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]?)", value)
    if not match:
        raise ValueError("Unknown GPFS quota quantity")
    return float(match.group(1)) * 1024 ** (" KMGTPE".index(match.group(2)) if match.group(2) else 0)


def capacity(parent, fileset, minimum_bytes=15_000_000_000):
    minimum_bytes = max(minimum_bytes, 15_000_000_000)
    fs = os.statvfs(str(parent))
    if fs.f_bavail * fs.f_frsize < minimum_bytes or fs.f_favail < 20000:
        raise ValueError("Insufficient filesystem bytes/inodes")
    command = ["/usr/lpp/mmfs/bin/mmlsattr", "-L", str(parent)]
    mapped = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    names = [line.split(":", 1)[1].strip() for line in mapped.stdout.decode().splitlines()
             if line.startswith("fileset name:")]
    if mapped.returncode or names != [fileset]:
        raise ValueError("Exact project fileset mapping differs")
    result = subprocess.run(["/usr/lpp/mmfs/bin/mmlsquota", "-j", fileset,
                             "--block-size", "auto", "rsrch8"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    rows = [line.split() for line in result.stdout.decode().splitlines() if line.split()[:1] == ["rsrch8"]]
    if result.returncode or len(rows) != 1:
        raise ValueError("Exact fileset quota unavailable")
    row = rows[0]
    if row[1] != "FILESET" or row[6] != "none" or row[12] != "none":
        raise ValueError("Fileset quota/grace is not qualified")
    used = _amount(row[2]) + _amount(row[5])
    limits = [_amount(x) for x in row[3:5] if _amount(x) > 0]
    file_used = int(row[8]) + int(row[11])
    file_limits = [int(x) for x in row[9:11] if int(x) > 0]
    if not limits or min(limits) - used < minimum_bytes or (file_limits and min(file_limits) - file_used < 20000):
        raise ValueError("Insufficient task fileset bytes/files")
    return dict(fileset=fileset, available_bytes=fs.f_bavail * fs.f_frsize,
                quota_output=result.stdout.decode(), minimum_bytes=minimum_bytes)


def validated_roots(plan):
    """Pure exact-scope check, tested against the real prepared specification."""
    roots = (Path(plan["control_root"]), Path(plan["bundle_root"]).parent)
    expected_suffix = "/perturbseq/R48_downstream_v1/main_stim8_preparation_20260930_v1"
    if (str(roots[0]) != "/rsrch8/scratch/bcb/yding4" + expected_suffix
            or str(roots[1]) != "/rsrch8/home/bcb/yding4" + expected_suffix
            or plan["bundle_root"] != str(roots[1] / "bundle")
            or plan["worker_output"] != str(roots[1] / "output")
            or plan["run_id"] != "credo-stim8-prepare-20260930-v1"):
        raise ValueError("B1 staging roots differ from exact approved scope")
    return roots


def publish_archive(raw, build, *, capacity_check=capacity):
    if (os.getuid(), os.getgid()) != (307469, 600815):
        raise ValueError("Institutional publisher UID/GID differs")
    payload, inventory = archive_payload(raw, build["archive_sha256"], build["inventory_sha256"])
    if sha(payload["PLAN.json"]) != build["plan_sha256"]:
        raise ValueError("Staging plan digest differs")
    plan = json.loads(payload["PLAN.json"])
    roots = validated_roots(plan)
    quotas = []
    for root, fileset in zip(roots, ("s_bcb-yding4", "h_bcb-yding4")):
        if os.path.lexists(root):
            raise FileExistsError("B1 stage is no-overwrite; reconcile existing exact root")
        for parent in root.parents:
            if not stat.S_ISDIR(parent.lstat().st_mode):
                raise ValueError("Unsafe staging ancestor")
        if root.parent.stat().st_uid != 307469 or not os.access(str(root.parent), os.W_OK):
            raise ValueError("Project parent ownership/writability differs")
        quotas.append(capacity_check(root.parent, fileset))
    publications = publish_both_roots(roots, payload, build)
    return dict(status="published_both_new_roots", roots=list(map(str, roots)),
                plan_sha256=build["plan_sha256"], inventory_sha256=build["inventory_sha256"],
                archive_sha256=build["archive_sha256"], quotas=quotas,
                original_data_retained=True, raw_counts_copied=False,
                both_root_canaries_passed_before_any_bundle=True, publications=publications)
