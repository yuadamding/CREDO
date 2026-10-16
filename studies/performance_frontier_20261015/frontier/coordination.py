"""Durable admission accounting, not a scheduler client or launch authority.

An owner reserves before submission. Pending, running and uncertain submissions
all occupy a slot; there is no timeout, retry or queue-bypass transition. Every
mutation appends a hash-linked immutable event under one POSIX lock. The actual
worker filesystem still requires lock/publication qualification on that mount.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from .planner import INNER_PAIRS, OUTER_TRIPLES, TrainingTask, TrajectoryKey


GIB = 1024 ** 3
LIVE = frozenset(("reserved", "submitted", "pending", "running", "uncertain"))
TERMINAL = frozenset(("complete", "failed", "cancelled"))
KINDS = frozenset(("representation", "main_fit", "forecast", "qualification"))


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Exact lowercase SHA-256 binding required")
    return value


def _label(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Safe exact task/worker identity required")
    return value


def _encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


def _continuation_binding(value):
    """The one approved 1,280 -> 20,000 observation recovery generation."""
    hashes = {"authority_sha256", "failure_evidence_sha256", "checkpoint_sha256",
              "recovery_completion_sha256", "retained_inventory_sha256"}
    numbers = {"accepted_updates", "target_updates", "retained_parent_bytes",
               "child_output_cap_bytes", "combined_output_cap_bytes"}
    if not isinstance(value, dict) or set(value) != hashes | numbers:
        raise ValueError("Exact continuation evidence binding required")
    for field in hashes:
        _sha(value[field])
    if (any(type(value[field]) is not int for field in numbers)
            or value["accepted_updates"] != 1280 or value["target_updates"] != 20000
            or value["combined_output_cap_bytes"] != 4 * GIB
            or not 0 < value["retained_parent_bytes"] < 4 * GIB
            or not 0 < value["child_output_cap_bytes"] <= 4 * GIB - value["retained_parent_bytes"]):
        raise ValueError("Exact remaining-update and combined four-GiB recovery limits required")
    return value


def _path(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink paths are not admitted")
    return path


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path, value):
    with path.open("xb") as handle:
        handle.write(_encode(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _sync(path.parent)


@dataclass(frozen=True)
class WorkBinding:
    """Canonical numerical work, separate from its scheduler/task alias.

    Main-fit transitions reuse the planner's scientific constraints. Other
    work's explicit identity must be the producer's complete operation digest;
    this journal cannot establish that an arbitrary supplied hash is complete.
    """

    kind: str
    identity: tuple
    start_updates: int = 0
    target_updates: int = 0

    def __post_init__(self):
        if self.kind not in KINDS or not isinstance(self.identity, tuple):
            raise ValueError("Typed canonical scientific work binding required")
        if (type(self.start_updates) is not int or type(self.target_updates) is not int
                or self.start_updates < 0 or self.target_updates < self.start_updates):
            raise ValueError("Exact nonnegative accepted-update interval required")
        if self.kind == "main_fit":
            if len(self.identity) != 3:
                raise ValueError("Main work requires subset/configuration/seed identity")
            donors, config, seed = self.identity
            key = TrajectoryKey(tuple(donors), config, seed)
            if tuple(donors) != key.training_donors:
                raise ValueError("Canonical fitting donor order required")
            # Selection eligibility remains bound by the separate admission
            # receipt. Do not rebuild or silently broaden any selector graph.
            from .configuration import DONORS
            selectors = tuple(d for d in DONORS if d not in key.training_donors)
            TrainingTask(key, self.start_updates, self.target_updates, selectors)
            object.__setattr__(self, "identity", (key.training_donors, config, seed))
        elif self.kind == "representation":
            if (len(self.identity) != 1 or tuple(self.identity[0]) not in INNER_PAIRS + OUTER_TRIPLES
                    or not self.start_updates < self.target_updates <= 20000):
                raise ValueError("One canonical subset-native observation fit and bounded interval required")
            object.__setattr__(self, "identity", (tuple(self.identity[0]),))
        else:
            if len(self.identity) != 1 or self.start_updates or self.target_updates:
                raise ValueError("Nontraining work requires an explicit operation identity and zero updates")
            _sha(self.identity[0])

    @classmethod
    def main(cls, task):
        if not isinstance(task, TrainingTask):
            raise ValueError("Main binding requires the registered planner TrainingTask")
        return cls("main_fit", (task.key.training_donors, task.key.configuration_id, task.key.seed),
                   task.start_updates, task.target_updates)

    @classmethod
    def representation(cls, training_donors, target_updates=20000, start_updates=0):
        return cls("representation", (tuple(training_donors),), start_updates, target_updates)

    @classmethod
    def other(cls, kind, operation_identity_sha256):
        if kind not in ("forecast", "qualification"):
            raise ValueError("Explicit nontraining operation kind required")
        return cls(kind, (_sha(operation_identity_sha256),))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"kind", "identity", "start_updates", "target_updates"}:
            raise ValueError("Malformed work binding schema")
        if not isinstance(value["identity"], list):
            raise ValueError("Serialized work identity must be a list")
        identity = list(value["identity"])
        if value["kind"] in ("main_fit", "representation") and identity:
            identity[0] = tuple(identity[0])
        return cls(value["kind"], tuple(identity), value["start_updates"], value["target_updates"])

    def to_dict(self):
        return json.loads(_encode({"kind": self.kind, "identity": self.identity,
                                  "start_updates": self.start_updates, "target_updates": self.target_updates}))

    @property
    def trajectory_identity(self):
        return _digest({"kind": self.kind, "identity": self.identity})

    @property
    def updates(self):
        return self.target_updates - self.start_updates


class AdmissionJournal(AbstractContextManager):
    """One study's bounded GPU/work/storage ledger, independent of wall time.

    Source/runtime/scientific admission receipts remain separate. This class
    validates their identities, not their substantive claims. Callers must
    reconcile exact scheduler objects before a terminal transition; supplying
    an arbitrary receipt hash is not operational verification.
    """

    def __init__(self, root, *, study_identity_sha256, storage_cap_bytes=384 * GIB,
                 unallocated_reserve_bytes=16 * GIB):
        self.root = _path(root)
        if (type(storage_cap_bytes) is not int or not 0 < storage_cap_bytes <= 384 * GIB
                or type(unallocated_reserve_bytes) is not int or unallocated_reserve_bytes < 16 * GIB):
            raise ValueError("Storage cap/reserve exceeds the declared design")
        self.identity = {"schema": "credo.frontier.admission.v2",
                         "study_identity_sha256": _sha(study_identity_sha256),
                         "h100_cap": 4, "forecast_worker_cap": 2,
                         "main_update_cap": 847872, "representation_update_cap": 200000,
                         "representation_fit_cap": 10, "storage_cap_bytes": storage_cap_bytes,
                         "unallocated_reserve_bytes": unallocated_reserve_bytes}
        self.lock = None
        self.pid = None

    def __enter__(self):
        if self.lock is not None:
            raise RuntimeError("Journal context is already active")
        self.root.mkdir(exist_ok=True)  # Parent prepared by the exact study owner.
        descriptor = os.open(self.root / "OWNER.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        lock = os.fdopen(descriptor, "r+b")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = _path(self.root / "IDENTITY.json")
            if identity.exists():
                if json.loads(identity.read_text()) != self.identity:
                    raise ValueError("Admission journal identity changed")
                if any(not _path(self.root / name).is_dir() for name in ("events", "intents")):
                    raise ValueError("Missing committed journal directory; explicit reconciliation required")
            else:
                if {item.name for item in self.root.iterdir()} != {"OWNER.lock"}:
                    raise ValueError("Existing unbound admission contents")
                for name in ("events", "intents"):
                    _path(self.root / name).mkdir()
                _sync(self.root)
                _write_once(identity, self.identity)
            self.lock, self.pid = lock, os.getpid()
            self.snapshot()
            return self
        except BaseException:
            lock.close()
            self.lock = None
            raise

    def __exit__(self, *exc):
        self.lock.close()
        self.lock = None
        return False

    def _active(self):
        if self.lock is None or self.lock.closed or self.pid != os.getpid():
            raise RuntimeError("Same-process exclusive journal owner required")
        if json.loads(_path(self.root / "IDENTITY.json").read_text()) != self.identity:
            raise ValueError("Active journal identity changed")

    def snapshot(self):
        self._active()
        state = {"tasks": {}, "sequence": 0, "last_sha256": "0" * 64}
        events = sorted(_path(self.root / "events").iterdir())
        intents = sorted(_path(self.root / "intents").iterdir())
        if [path.name for path in events] != [path.name for path in intents]:
            raise ValueError("Incomplete append intent/event inventory; explicit reconciliation required")
        for number, (path, intent_path) in enumerate(zip(events, intents), 1):
            if path.name != f"{number:08d}.json" or not _path(path).is_file():
                raise ValueError("Incomplete or unexpected journal inventory; explicit reconciliation required")
            event = json.loads(path.read_text())
            intent = json.loads(_path(intent_path).read_text())
            if (not isinstance(intent, dict) or set(intent) != {"sequence", "event_sha256"}
                    or intent["sequence"] != number or intent["event_sha256"] != _digest(event)):
                raise ValueError("Admission append intent/hash chain mismatch")
            if (not isinstance(event, dict)
                    or set(event) != {"sequence", "previous_sha256", "study_identity", "action", "tasks"}
                    or event["sequence"] != number or event["previous_sha256"] != state["last_sha256"]
                    or event["study_identity"] != self.identity):
                raise ValueError("Admission event chain mismatch")
            self._validate(event["tasks"])
            self._validate_transition(state["tasks"], event["tasks"], event["action"])
            state = {"tasks": event["tasks"], "sequence": number, "last_sha256": _digest(event)}
        return state

    def _validate(self, tasks):
        if not isinstance(tasks, dict):
            raise ValueError("Task registry must be an exact mapping")
        required = {"worker_id", "kind", "scientific_update_reservation", "storage_reserved_bytes",
                    "admission_sha256", "capacity_receipt_sha256", "capacity_headroom_bytes", "status",
                    "scheduler_id", "work_binding"}
        hashes = {"evidence_sha256", "retained_inventory_sha256", "replacement_authority_sha256",
                  "zero_update_evidence_sha256"}
        continuation_fields = {"continuation_task_id", "continues_task_id",
                               "original_work_binding", "continuation_binding"}
        continuation_replacement_fields = {"continuation_replacement_task_id", "replaces_continuation_task_id"}
        optional = hashes | {"replacement_task_id", "replaces_task_id"} | continuation_fields | continuation_replacement_fields
        groups = {}
        for key, item in tasks.items():
            if not isinstance(item, dict) or not required <= set(item) or set(item) - required - optional:
                raise ValueError("Malformed task reservation schema")
            _label(key)
            _label(item["worker_id"])
            for field in ("admission_sha256", "capacity_receipt_sha256"):
                _sha(item[field])
            for field in hashes & set(item):
                _sha(item[field])
            if item["scheduler_id"] is not None:
                _label(item["scheduler_id"])
            if item["kind"] not in KINDS or item["status"] not in LIVE | TERMINAL:
                raise ValueError("Unknown task kind or state")
            if any(type(item[k]) is not int or item[k] < 0 for k in
                   ("scientific_update_reservation", "storage_reserved_bytes", "capacity_headroom_bytes")):
                raise ValueError("Nonnegative exact workload and storage reservations required")
            binding = WorkBinding.from_dict(item["work_binding"])
            if binding.kind != item["kind"] or binding.updates != item["scientific_update_reservation"]:
                raise ValueError("Reserved workload differs from the canonical work binding")
            if item["status"] == "reserved":
                if item["scheduler_id"] is not None or "evidence_sha256" in item:
                    raise ValueError("A fresh reservation cannot carry scheduler state")
            elif "evidence_sha256" not in item or (item["scheduler_id"] is None and item["status"] not in
                                                       ("uncertain", "failed", "cancelled")):
                raise ValueError("Reconciled status lacks exact scheduler/evidence identity")
            if "retained_inventory_sha256" in item and item["status"] not in TERMINAL:
                raise ValueError("Live work cannot release storage")
            if "replacement_task_id" in item:
                successor = tasks.get(_label(item["replacement_task_id"]))
                if (item["kind"] != "representation" or item["status"] not in ("failed", "cancelled")
                        or "replaces_task_id" in item or "zero_update_evidence_sha256" not in item
                        or not isinstance(successor, dict) or successor.get("replaces_task_id") != key
                        or successor.get("original_work_binding", successor.get("work_binding")) != item["work_binding"]
                        or successor.get("replacement_authority_sha256") != item.get("replacement_authority_sha256")
                        or "replacement_authority_sha256" not in item):
                    raise ValueError("Exact zero-update replacement linkage required")
            elif "continuation_replacement_task_id" not in item:
                groups.setdefault(binding.trajectory_identity, []).append((binding, item))
            if "replaces_task_id" in item:
                predecessor = tasks.get(_label(item["replaces_task_id"]))
                if (item["kind"] != "representation" or not isinstance(predecessor, dict)
                        or predecessor.get("replacement_task_id") != key
                        or "replacement_authority_sha256" not in item):
                    raise ValueError("Replacement requires retained predecessor and explicit authority")
            if ("replacement_authority_sha256" in item and not
                    ({"replacement_task_id", "replaces_task_id"} | continuation_replacement_fields) & set(item)
                    or "zero_update_evidence_sha256" in item and not
                    ({"replacement_task_id"} | continuation_replacement_fields) & set(item)):
                raise ValueError("Orphan replacement evidence")
            if continuation_fields & set(item):
                evidence = _continuation_binding(item.get("continuation_binding"))
                if ("continuation_task_id" in item) == ("continues_task_id" in item):
                    raise ValueError("One exact continuation linkage required")
                if "continuation_task_id" in item:
                    parent_id, parent = key, item
                    child_id = _label(item["continuation_task_id"])
                    child = tasks.get(child_id)
                else:
                    child_id, child = key, item
                    parent_id = _label(item["continues_task_id"])
                    parent = tasks.get(parent_id)
                original_child = None if not isinstance(parent, dict) else tasks.get(parent.get("continuation_task_id"))
                linked_child = isinstance(parent, dict) and isinstance(child, dict) and (
                    parent.get("continuation_task_id") == child_id
                    or isinstance(original_child, dict)
                    and original_child.get("continuation_replacement_task_id") == child_id
                    and child.get("replaces_continuation_task_id") == parent.get("continuation_task_id"))
                if (not isinstance(parent, dict) or not isinstance(child, dict) or not linked_child
                        or child.get("continues_task_id") != parent_id
                        or parent.get("continuation_binding") != evidence
                        or child.get("continuation_binding") != evidence
                        or parent.get("kind") != "representation" or parent.get("status") != "failed"
                        or parent.get("scheduler_id") is None
                        or parent.get("evidence_sha256") != evidence["failure_evidence_sha256"]
                        or child.get("kind") != "representation"
                        or {"continues_task_id", "replacement_task_id"} & set(parent)
                        or {"continuation_task_id", "original_work_binding", "replacement_task_id",
                            "replaces_task_id"} & set(child)):
                    raise ValueError("Exact failed-parent remaining-work continuation linkage required")
                original = WorkBinding.from_dict(parent.get("original_work_binding"))
                consumed = WorkBinding.from_dict(parent["work_binding"])
                remaining = WorkBinding.from_dict(child["work_binding"])
                if (original.kind != "representation" or original.start_updates != 0
                        or original.target_updates != 20000
                        or consumed != WorkBinding.representation(original.identity[0], 1280)
                        or remaining != WorkBinding.representation(original.identity[0], 20000, 1280)
                        or "retained_inventory_sha256" not in parent
                        or parent["storage_reserved_bytes"] > evidence["retained_parent_bytes"]
                        or child["storage_reserved_bytes"] > evidence["child_output_cap_bytes"]):
                    raise ValueError("Continuation must retain consumed work and original combined output budget")
            if continuation_replacement_fields & set(item):
                if ("continuation_replacement_task_id" in item) == ("replaces_continuation_task_id" in item):
                    raise ValueError("Exactly one zero-work continuation replacement generation required")
                if "continuation_replacement_task_id" in item:
                    old_id, old = key, item
                    new_id = _label(item["continuation_replacement_task_id"])
                    new = tasks.get(new_id)
                else:
                    new_id, new = key, item
                    old_id = _label(item["replaces_continuation_task_id"])
                    old = tasks.get(old_id)
                if (not isinstance(old, dict) or not isinstance(new, dict)
                        or old.get("continuation_replacement_task_id") != new_id
                        or new.get("replaces_continuation_task_id") != old_id
                        or old.get("status") != "failed" or old.get("scheduler_id") is None
                        or old.get("kind") != "representation" or new.get("kind") != "representation"
                        or "continues_task_id" not in old
                        or new.get("continues_task_id") != old["continues_task_id"]
                        or new.get("continuation_binding") != old.get("continuation_binding")
                        or new.get("work_binding") != old["work_binding"]
                        or old.get("storage_reserved_bytes") != 0
                        or old.get("retained_inventory_sha256") != old.get("evidence_sha256")
                        or old.get("zero_update_evidence_sha256") != old.get("evidence_sha256")
                        or new.get("zero_update_evidence_sha256") != old.get("evidence_sha256")
                        or not isinstance(old.get("replacement_authority_sha256"), str)
                        or new.get("replacement_authority_sha256") != old["replacement_authority_sha256"]
                        or old["replacement_authority_sha256"] == old["continuation_binding"]["authority_sha256"]
                        or {"replaces_continuation_task_id", "replacement_task_id", "replaces_task_id"} & set(old)
                        or {"continuation_replacement_task_id", "replacement_task_id", "replaces_task_id"} & set(new)):
                    raise ValueError("Exact failed zero-work continuation replacement linkage required")
        for sequence in groups.values():
            sequence.sort(key=lambda pair: pair[0].start_updates)
            if sequence[0][0].kind not in ("main_fit", "representation"):
                if len(sequence) != 1:
                    raise ValueError("Duplicate canonical scientific operation, regardless of task alias")
                continue
            if len({(binding.start_updates, binding.target_updates) for binding, _ in sequence}) != len(sequence):
                raise ValueError("Duplicate, overlapping or unaligned trajectory interval")
            prior_end = 0
            for index, (binding, item) in enumerate(sequence):
                if binding.start_updates != prior_end:
                    raise ValueError("Duplicate, overlapping or unaligned trajectory interval")
                if (index < len(sequence) - 1 and item["status"] != "complete"
                        and not (item["status"] == "failed" and "continuation_task_id" in item)):
                    raise ValueError("Continuation requires the exact completed predecessor; no retry")
                prior_end = binding.target_updates
        active = [item for item in tasks.values() if item["status"] in LIVE]
        if len(active) > 4 or sum(item["kind"] == "forecast" for item in active) > 2:
            raise ValueError("Concurrent H100/forecast reservation cap exceeded")
        worker_ids = [item["worker_id"] for item in active]
        scheduler_ids = [item["scheduler_id"] for item in tasks.values() if item["scheduler_id"] is not None]
        if len(set(worker_ids)) != len(worker_ids) or len(set(scheduler_ids)) != len(scheduler_ids):
            raise ValueError("Duplicate active worker or scheduler object")
        if sum(item["storage_reserved_bytes"] for item in tasks.values()) > self.identity["storage_cap_bytes"]:
            raise ValueError("Aggregate live-output reservation exceeds study cap")
        for kind, cap in (("main_fit", 847872), ("representation", 200000)):
            if sum(item["scientific_update_reservation"] for item in tasks.values()
                   if item["kind"] == kind and not
                   {"replacement_task_id", "continuation_replacement_task_id"} & set(item)) > cap:
                raise ValueError("Scientific update reservation ceiling exceeded")
        if sum(sequence[0][0].kind == "representation" for sequence in groups.values()) > 10:
            raise ValueError("Observation fit inventory exceeded")
        for key, item in tasks.items():
            if item["kind"] in ("forecast", "qualification") and item["scientific_update_reservation"] != 0:
                raise ValueError("Forecast/qualification cannot reserve scientific training")
            if item["kind"] == "representation" and item["scientific_update_reservation"] > 20000:
                raise ValueError("One observation fit exceeds update ceiling")
            if item["kind"] == "main_fit" and item["scientific_update_reservation"] > 9216:
                raise ValueError("One main fit exceeds update ceiling")
            if item["kind"] == "forecast" and item["storage_reserved_bytes"] > 64 * GIB:
                raise ValueError("Forecast worker storage ceiling exceeded")

    @staticmethod
    def _validate_transition(previous, current, action):
        if not isinstance(action, dict) or not {"type", "task_id"} <= set(action):
            raise ValueError("Malformed journal action")
        key = action["task_id"]
        _label(key)
        kind = action["type"]
        if kind == "authorized_zero_new_update_continuation_replacement":
            if set(action) != {"type", "task_id", "predecessor_id", "authority_sha256", "evidence_sha256"}:
                raise ValueError("Exact zero-work continuation replacement action required")
            old = _label(action["predecessor_id"])
            authority, evidence = _sha(action["authority_sha256"]), _sha(action["evidence_sha256"])
            if old not in previous or key in previous or set(current) != set(previous) | {key}:
                raise ValueError("Fresh replacement and retained failed continuation required")
            before, after, replacement = previous[old], current[old], current[key]
            if (before["kind"] != "representation" or before["status"] != "failed"
                    or before["scheduler_id"] is None or before["evidence_sha256"] != evidence
                    or "continues_task_id" not in before
                    or {"continuation_replacement_task_id", "replaces_continuation_task_id",
                        "replacement_task_id", "replaces_task_id"} & set(before)
                    or authority == before["continuation_binding"]["authority_sha256"]
                    or before["storage_reserved_bytes"] != before["continuation_binding"]["child_output_cap_bytes"]
                    or replacement["status"] != "reserved"
                    or replacement["work_binding"] != before["work_binding"]
                    or replacement["scientific_update_reservation"] != before["scientific_update_reservation"]
                    or replacement["storage_reserved_bytes"] != before["storage_reserved_bytes"]
                    or replacement["admission_sha256"] != before["admission_sha256"]
                    or replacement.get("continues_task_id") != before["continues_task_id"]
                    or replacement.get("continuation_binding") != before["continuation_binding"]
                    or replacement.get("replaces_continuation_task_id") != old
                    or replacement.get("replacement_authority_sha256") != authority
                    or replacement.get("zero_update_evidence_sha256") != evidence):
                raise ValueError("Only the explicitly authorized zero-work failed continuation may be replaced")
            expected = dict(before, continuation_replacement_task_id=key,
                replacement_authority_sha256=authority, zero_update_evidence_sha256=evidence,
                storage_reserved_bytes=0, retained_inventory_sha256=evidence)
            if after != expected or any(current[k] != v for k, v in previous.items() if k != old):
                raise ValueError("Continuation replacement altered retained scientific parent or failure evidence")
            return
        if kind == "authorized_remaining_observation_continuation":
            if set(action) != {"type", "task_id", "predecessor_id", "continuation_binding"}:
                raise ValueError("Exact remaining-work continuation action required")
            old = _label(action["predecessor_id"])
            evidence = _continuation_binding(action["continuation_binding"])
            if old not in previous or key in previous or set(current) != set(previous) | {key}:
                raise ValueError("Fresh continuation and retained predecessor required")
            before, after, child = previous[old], current[old], current[key]
            original = WorkBinding.from_dict(before["work_binding"])
            if (before["kind"] != "representation" or before["status"] != "failed"
                    or before["scheduler_id"] is None
                    or before["evidence_sha256"] != evidence["failure_evidence_sha256"]
                    or {"continuation_task_id", "continues_task_id", "continuation_binding",
                        "original_work_binding", "replacement_task_id"} & set(before)
                    or original.start_updates != 0 or original.target_updates != 20000
                    or evidence["retained_parent_bytes"] > before["storage_reserved_bytes"]
                    or before["storage_reserved_bytes"] > 4 * GIB
                    or child["status"] != "reserved"
                    or child.get("continuation_binding") != evidence
                    or child.get("continues_task_id") != old):
                raise ValueError("Only the explicitly approved failed observation generation may continue")
            expected = dict(before, scientific_update_reservation=1280,
                work_binding=WorkBinding.representation(original.identity[0], 1280).to_dict(),
                original_work_binding=before["work_binding"], continuation_task_id=key,
                continuation_binding=evidence, storage_reserved_bytes=evidence["retained_parent_bytes"],
                retained_inventory_sha256=evidence["retained_inventory_sha256"])
            if (after != expected
                    or child["storage_reserved_bytes"] != evidence["child_output_cap_bytes"]
                    or any(current[k] != v for k, v in previous.items() if k != old)):
                raise ValueError("Continuation altered unrelated or retained failure evidence")
            return
        if kind == "authorized_zero_update_representation_replacement":
            if set(action) != {"type", "task_id", "predecessor_id", "authority_sha256", "evidence_sha256"}:
                raise ValueError("Exact replacement action required")
            old = _label(action["predecessor_id"])
            authority, evidence = _sha(action["authority_sha256"]), _sha(action["evidence_sha256"])
            if old not in previous or key in previous or set(current) != set(previous) | {key}:
                raise ValueError("Fresh replacement and retained predecessor required")
            before, after, replacement = previous[old], current[old], current[key]
            if (before["kind"] != "representation" or before["status"] not in ("reserved", "failed", "cancelled")
                    or {"replacement_task_id", "replaces_task_id", "continuation_task_id",
                        "continues_task_id"} & set(before)
                    or replacement["status"] != "reserved"
                    or replacement["work_binding"] != before["work_binding"]
                    or replacement.get("replaces_task_id") != old
                    or replacement.get("replacement_authority_sha256") != authority):
                raise ValueError("Only explicitly approved original zero-update representation work can be replaced")
            expected = dict(before, replacement_task_id=key, replacement_authority_sha256=authority,
                            zero_update_evidence_sha256=evidence)
            if before["status"] == "reserved":
                expected.update(status="cancelled", evidence_sha256=evidence)
            if after != expected or any(current[k] != v for k, v in previous.items() if k != old):
                raise ValueError("Replacement altered unrelated or retained failure evidence")
            return
        expected_fields = {"type", "task_id"}
        if kind == "verified_storage_inventory":
            expected_fields.add("deletion_performed_by_journal")
            if action.get("deletion_performed_by_journal") is not False:
                raise ValueError("Journal cannot claim file deletion")
        if set(action) != expected_fields:
            raise ValueError("Unknown journal action fields")
        if kind == "reserve_before_submission":
            if key in previous or set(current) != set(previous) | {key} or current[key]["status"] != "reserved":
                raise ValueError("Invalid reserve transition")
        elif kind in ("exact_scheduler_reconciliation", "verified_storage_inventory"):
            if key not in previous or set(current) != set(previous):
                raise ValueError("Existing task inventory changed during reconciliation")
            before, after = previous[key], current[key]
            fields = ({"status", "scheduler_id", "evidence_sha256"} if kind == "exact_scheduler_reconciliation"
                      else {"storage_reserved_bytes", "retained_inventory_sha256"})
            if {k: v for k, v in before.items() if k not in fields} != {
                    k: v for k, v in after.items() if k not in fields}:
                raise ValueError("Immutable reservation identity/workload changed")
            if kind == "exact_scheduler_reconciliation":
                if before["status"] in TERMINAL or after["status"] == "reserved":
                    raise ValueError("Terminal work cannot be retried")
                if before["scheduler_id"] is not None and after["scheduler_id"] != before["scheduler_id"]:
                    raise ValueError("Scheduler object identity changed")
            elif (before["status"] not in TERMINAL
                  or after["storage_reserved_bytes"] > before["storage_reserved_bytes"]):
                raise ValueError("Only terminal inventory can reduce storage reservations")
        else:
            raise ValueError("Unknown journal action")
        if any(current[old] != previous[old] for old in previous if old != key):
            raise ValueError("An action changed unrelated prior reservations")

    def _append(self, state, action):
        self._validate(state["tasks"])
        previous = self.snapshot()
        if previous["sequence"] != state["sequence"] or previous["last_sha256"] != state["last_sha256"]:
            raise ValueError("Admission state changed before append")
        self._validate_transition(previous["tasks"], state["tasks"], action)
        sequence = state["sequence"] + 1
        event = {"sequence": sequence, "previous_sha256": state["last_sha256"],
                 "study_identity": self.identity, "action": action, "tasks": state["tasks"]}
        _write_once(self.root / "intents" / f"{sequence:08d}.json",
                    {"sequence": sequence, "event_sha256": _digest(event)})
        _write_once(self.root / "events" / f"{sequence:08d}.json", event)
        return self.snapshot()

    def reserve(self, task_id, *, worker_id, kind, scientific_updates, output_bytes,
                admission_sha256, capacity_receipt_sha256, fileset_headroom_bytes, work_binding):
        state = self.snapshot()
        if not isinstance(work_binding, WorkBinding):
            raise ValueError("Typed canonical scientific work binding required")
        _label(task_id)
        if task_id in state["tasks"]:
            raise ValueError("Task already reserved; no duplicate or implicit retry")
        if (type(output_bytes) is not int or output_bytes < 0
                or type(fileset_headroom_bytes) is not int or fileset_headroom_bytes < 0):
            raise ValueError("Exact output and fresh fileset-headroom bytes required")
        # Live/unreconciled reservations remain conservative: some bytes may
        # already be on disk. A terminal, verified retained inventory cannot
        # grow and is already included in fresh fileset usage, so do not charge
        # it again against free bytes. It STILL counts toward the study's total
        # live-output cap in _validate. Actual inventory/terminal proof belongs
        # to the worker owner, not the hash label alone.
        reserved = sum(item["storage_reserved_bytes"] for item in state["tasks"].values()
                       if item["status"] not in TERMINAL or "retained_inventory_sha256" not in item)
        if fileset_headroom_bytes < reserved + output_bytes + self.identity["unallocated_reserve_bytes"]:
            raise ValueError("Insufficient freshly measured quota headroom including outstanding reservations")
        state["tasks"][task_id] = {"worker_id": _label(worker_id), "kind": kind,
            "scientific_update_reservation": scientific_updates, "storage_reserved_bytes": output_bytes,
            "admission_sha256": _sha(admission_sha256), "capacity_receipt_sha256": _sha(capacity_receipt_sha256),
            "capacity_headroom_bytes": fileset_headroom_bytes, "status": "reserved", "scheduler_id": None,
            "work_binding": work_binding.to_dict()}
        return self._append(state, {"type": "reserve_before_submission", "task_id": task_id})

    def reconcile(self, task_id, *, status, exact_scheduler_id, evidence_sha256):
        state = self.snapshot()
        item = state["tasks"][task_id]
        if item["status"] in TERMINAL or status not in (LIVE - {"reserved"}) | TERMINAL:
            raise ValueError("Terminal task cannot be retried; unknown transition")
        if exact_scheduler_id is not None:
            _label(exact_scheduler_id)
        if item["scheduler_id"] is not None and exact_scheduler_id != item["scheduler_id"]:
            raise ValueError("Scheduler object identity changed")
        if exact_scheduler_id is None and status not in ("uncertain", "failed", "cancelled"):
            raise ValueError("Known exact scheduler identity required for this status")
        item.update(status=status, scheduler_id=exact_scheduler_id, evidence_sha256=_sha(evidence_sha256))
        return self._append(state, {"type": "exact_scheduler_reconciliation", "task_id": task_id})

    def replace_zero_update_representation(self, predecessor_id, task_id, *, authority_sha256,
            zero_update_evidence_sha256, admission_sha256, capacity_receipt_sha256,
            fileset_headroom_bytes, output_bytes=4*GIB):
        """Explicit one-generation recovery, never automatic retry.

        The caller must validate actual zero-work/absence and user approval.
        This ledger retains the prior immutable events and task/storage record,
        transfers its unused scientific reservation and appends a new attempt.
        A completed fit, live submitted worker or replacement cannot use this.
        """
        state = self.snapshot()
        _label(predecessor_id); _label(task_id)
        _sha(authority_sha256); _sha(zero_update_evidence_sha256)
        if predecessor_id not in state["tasks"] or task_id in state["tasks"]:
            raise ValueError("Known predecessor and absent replacement required")
        old = state["tasks"][predecessor_id]
        if (old["kind"] != "representation" or old["status"] not in ("reserved", "failed", "cancelled")
                or {"replacement_task_id", "replaces_task_id", "continuation_task_id",
                    "continues_task_id"} & set(old)
                or type(output_bytes) is not int or output_bytes != 4*GIB
                or type(fileset_headroom_bytes) is not int or fileset_headroom_bytes < 0):
            raise ValueError("Exact zero-update original observation replacement scope required")
        reserved = sum(t["storage_reserved_bytes"] for t in state["tasks"].values()
                       if t["status"] not in TERMINAL or "retained_inventory_sha256" not in t)
        if fileset_headroom_bytes < reserved + output_bytes + self.identity["unallocated_reserve_bytes"]:
            raise ValueError("Replacement must also fund retained prior output and reserve")
        old.update(replacement_task_id=task_id, replacement_authority_sha256=authority_sha256,
                   zero_update_evidence_sha256=zero_update_evidence_sha256)
        if old["status"] == "reserved":
            old.update(status="cancelled", evidence_sha256=zero_update_evidence_sha256)
        state["tasks"][task_id] = dict(worker_id=task_id, kind="representation",
            scientific_update_reservation=old["scientific_update_reservation"], storage_reserved_bytes=output_bytes,
            admission_sha256=_sha(admission_sha256), capacity_receipt_sha256=_sha(capacity_receipt_sha256),
            capacity_headroom_bytes=fileset_headroom_bytes, status="reserved", scheduler_id=None,
            work_binding=old["work_binding"], replaces_task_id=predecessor_id,
            replacement_authority_sha256=authority_sha256)
        return self._append(state, dict(type="authorized_zero_update_representation_replacement",
            task_id=task_id, predecessor_id=predecessor_id, authority_sha256=authority_sha256,
            evidence_sha256=zero_update_evidence_sha256))

    def continue_failed_representation(self, predecessor_id, task_id, *, accepted_updates,
            authority_sha256, failure_evidence_sha256, checkpoint_sha256, recovery_completion_sha256,
            retained_parent_bytes, retained_inventory_sha256, admission_sha256,
            capacity_receipt_sha256, fileset_headroom_bytes, output_bytes):
        """Admit one explicitly approved source-changing 1,280-update recovery.

        This does not declare a failed fit complete or reset its scientific
        budget. A new event accounts 1,280 consumed updates on the failed parent
        and 18,720 remaining on its fresh child; original events, source binding,
        scheduler identity and failure evidence remain intact. The owner must
        verify the checkpoint, zero-update recovery completion and immutable
        parent inventory before supplying their hashes. Parent plus child stay
        within the original four-GiB allowance; no files are deleted here.
        """
        state = self.snapshot()
        _label(predecessor_id); _label(task_id)
        if predecessor_id not in state["tasks"] or task_id in state["tasks"]:
            raise ValueError("Known failed predecessor and absent continuation required")
        old = state["tasks"][predecessor_id]
        evidence = _continuation_binding(dict(authority_sha256=authority_sha256,
            failure_evidence_sha256=failure_evidence_sha256, checkpoint_sha256=checkpoint_sha256,
            recovery_completion_sha256=recovery_completion_sha256,
            retained_inventory_sha256=retained_inventory_sha256, accepted_updates=accepted_updates,
            target_updates=20000, retained_parent_bytes=retained_parent_bytes,
            child_output_cap_bytes=output_bytes, combined_output_cap_bytes=4 * GIB))
        original = WorkBinding.from_dict(old["work_binding"])
        if (old["kind"] != "representation" or old["status"] != "failed"
                or old["scheduler_id"] is None or old["evidence_sha256"] != failure_evidence_sha256
                or {"replacement_task_id", "continuation_task_id", "continues_task_id",
                    "original_work_binding", "continuation_binding"} & set(old)
                or original.start_updates != 0 or original.target_updates != 20000
                or not retained_parent_bytes <= old["storage_reserved_bytes"] <= 4 * GIB
                or type(fileset_headroom_bytes) is not int or fileset_headroom_bytes < 0):
            raise ValueError("Exact failed observation, retained inventory and remaining-work authority required")
        # The inventory is already on disk and included in freshly observed
        # usage. Charge its retained bytes to the study total, not free capacity.
        old.update(scientific_update_reservation=1280,
            work_binding=WorkBinding.representation(original.identity[0], 1280).to_dict(),
            original_work_binding=original.to_dict(), continuation_task_id=task_id,
            continuation_binding=evidence, storage_reserved_bytes=retained_parent_bytes,
            retained_inventory_sha256=retained_inventory_sha256)
        reserved = sum(t["storage_reserved_bytes"] for t in state["tasks"].values()
                       if t["status"] not in TERMINAL or "retained_inventory_sha256" not in t)
        if fileset_headroom_bytes < reserved + output_bytes + self.identity["unallocated_reserve_bytes"]:
            raise ValueError("Continuation requires fresh headroom for all outstanding outputs and reserve")
        state["tasks"][task_id] = dict(worker_id=task_id, kind="representation",
            scientific_update_reservation=18720, storage_reserved_bytes=output_bytes,
            admission_sha256=_sha(admission_sha256), capacity_receipt_sha256=_sha(capacity_receipt_sha256),
            capacity_headroom_bytes=fileset_headroom_bytes, status="reserved", scheduler_id=None,
            work_binding=WorkBinding.representation(original.identity[0], 20000, 1280).to_dict(),
            continues_task_id=predecessor_id, continuation_binding=evidence)
        return self._append(state, dict(type="authorized_remaining_observation_continuation",
            task_id=task_id, predecessor_id=predecessor_id, continuation_binding=evidence))

    def replace_failed_continuation(self, predecessor_id, task_id, *, authority_sha256,
            zero_update_evidence_sha256, admission_sha256, capacity_receipt_sha256,
            fileset_headroom_bytes):
        """One explicit replacement of a failed zero-new-update continuation.

        The caller must verify that the exact failed attempt applied no update
        and created no scientific output. Its current failure receipt must bind
        that proof and exact scheduler absence. No historical event, original
        1,280-update parent, source lineage or remaining-work interval changes.
        The failed attempt's unused output reservation is transferred rather
        than doubled; scheduler/controller failure receipts remain retained.
        This does not authorize another replacement if the fresh attempt fails.
        """
        state = self.snapshot()
        _label(predecessor_id); _label(task_id)
        _sha(authority_sha256); _sha(zero_update_evidence_sha256)
        _sha(admission_sha256); _sha(capacity_receipt_sha256)
        if predecessor_id not in state["tasks"] or task_id in state["tasks"]:
            raise ValueError("Known failed continuation and absent replacement required")
        old = state["tasks"][predecessor_id]
        if (old["kind"] != "representation" or old["status"] != "failed"
                or old["scheduler_id"] is None or old["evidence_sha256"] != zero_update_evidence_sha256
                or "continues_task_id" not in old
                or {"continuation_replacement_task_id", "replaces_continuation_task_id",
                    "replacement_task_id", "replaces_task_id"} & set(old)
                or old["admission_sha256"] != admission_sha256
                or type(fileset_headroom_bytes) is not int or fileset_headroom_bytes < 0):
            raise ValueError("Exact terminal zero-work continuation and new explicit authority required")
        evidence = _continuation_binding(old["continuation_binding"])
        work = WorkBinding.from_dict(old["work_binding"])
        if (work.start_updates != 1280 or work.target_updates != 20000
                or old["scientific_update_reservation"] != 18720
                or old["storage_reserved_bytes"] != evidence["child_output_cap_bytes"]
                or authority_sha256 == evidence["authority_sha256"]):
            raise ValueError("Same unused 18,720-update and residual-storage reservation required")
        output_bytes = old["storage_reserved_bytes"]
        old.update(continuation_replacement_task_id=task_id, replacement_authority_sha256=authority_sha256,
            zero_update_evidence_sha256=zero_update_evidence_sha256, storage_reserved_bytes=0,
            retained_inventory_sha256=zero_update_evidence_sha256)
        reserved = sum(t["storage_reserved_bytes"] for t in state["tasks"].values()
                       if t["status"] not in TERMINAL or "retained_inventory_sha256" not in t)
        if fileset_headroom_bytes < reserved + output_bytes + self.identity["unallocated_reserve_bytes"]:
            raise ValueError("Replacement requires fresh headroom for unchanged outstanding outputs and reserve")
        state["tasks"][task_id] = dict(worker_id=task_id, kind="representation",
            scientific_update_reservation=18720, storage_reserved_bytes=output_bytes,
            admission_sha256=admission_sha256, capacity_receipt_sha256=capacity_receipt_sha256,
            capacity_headroom_bytes=fileset_headroom_bytes, status="reserved", scheduler_id=None,
            work_binding=work.to_dict(), continues_task_id=old["continues_task_id"],
            continuation_binding=evidence, replaces_continuation_task_id=predecessor_id,
            replacement_authority_sha256=authority_sha256, zero_update_evidence_sha256=zero_update_evidence_sha256)
        return self._append(state, dict(type="authorized_zero_new_update_continuation_replacement",
            task_id=task_id, predecessor_id=predecessor_id, authority_sha256=authority_sha256,
            evidence_sha256=zero_update_evidence_sha256))

    def release_storage(self, task_id, *, retained_bytes, inventory_receipt_sha256):
        state = self.snapshot()
        item = state["tasks"][task_id]
        if (item["status"] not in TERMINAL or type(retained_bytes) is not int
                or not 0 <= retained_bytes <= item["storage_reserved_bytes"]):
            raise ValueError("Only reconciled terminal storage may reduce its reservation")
        item.update(storage_reserved_bytes=retained_bytes,
                    retained_inventory_sha256=_sha(inventory_receipt_sha256))
        return self._append(state, {"type": "verified_storage_inventory", "task_id": task_id,
                                    "deletion_performed_by_journal": False})
