"""Reserve the first observation qualification in the actual four-H100 journal.

Pure local metadata operation, not a scheduler submission. It has no retry or
replacement path: an existing task or output receipt must be reconciled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from transport import ROOT, record

sys.path.insert(0, str(ROOT))
from deployment.build import SOURCE_MEMBERS
from deployment.common import encoded
from frontier.coordination import AdmissionJournal, WorkBinding, LIVE


RUN_ID = "credo-frontier-obs-qual-20261015-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve(capacity_path, output):
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("Admission export exists; do not reserve twice")
    capacity_path = Path(capacity_path).absolute()
    receipt = json.loads(capacity_path.read_bytes())
    if (receipt.get("schema") != "credo.frontier.storage-intake.v1"
            or receipt.get("study_id") != ROOT.name or receipt.get("fileset") != "h_bcb-yding4"
            or receipt.get("remote_writes") is not False or receipt.get("numerical_payloads_opened") is not False
            or receipt.get("filesystem_available_inodes", 0) < 20000):
        raise ValueError("Exact current metadata-only frontier capacity receipt required")
    source_manifest = encoded(dict(schema="credo.frontier.observation-worker-source.v1",
                                   files={name: digest(ROOT / name) for name in SOURCE_MEMBERS}))
    source_sha = hashlib.sha256(source_manifest).hexdigest()
    phase_sha = digest(ROOT / "OBSERVATION_PHASE_CONTRACT.json")
    operation = {"run_id": RUN_ID, "kind": "qualification", "phase_authority_sha256": phase_sha,
                 "source_manifest_sha256": source_sha, "protocol_sha256": digest(ROOT / "OBSERVATION_PROTOCOL.json")}
    work = WorkBinding.other("qualification", hashlib.sha256(encoded(operation)).hexdigest())
    with AdmissionJournal(ROOT / "admission", study_identity_sha256=digest(ROOT / "AUTHORITY.json")) as journal:
        state = journal.reserve(RUN_ID, worker_id=RUN_ID, kind="qualification", scientific_updates=0,
            output_bytes=16 * 1024 ** 3, admission_sha256=phase_sha, capacity_receipt_sha256=digest(capacity_path),
            fileset_headroom_bytes=min(receipt["headroom_bytes"], receipt["filesystem_available_bytes"]), work_binding=work)
        result = {"schema": "credo.frontier.observation-admission-export.v1", "run_id": RUN_ID,
                  "status": "reserved", "storage_reserved_bytes": 16 * 1024 ** 3, "scientific_update_reservation": 0,
                  "aggregate_reserved_output_bytes": sum(item["storage_reserved_bytes"] for item in state["tasks"].values()),
                  "live_h100_reservations": sum(item["status"] in LIVE for item in state["tasks"].values()),
                  "journal_event_sha256": state["last_sha256"], "journal_identity_sha256": digest(ROOT / "admission/IDENTITY.json"),
                  "phase_authority_sha256": phase_sha, "source_manifest_sha256": source_sha,
                  "capacity_receipt_sha256": digest(capacity_path), "work_binding": work.to_dict(),
                  "scheduler_submitted": False, "scientific_updates": 0}
        record(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reserve(args.capacity_receipt, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
