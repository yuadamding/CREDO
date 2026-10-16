"""Metadata-only nested-study graph and single-GPU admission accounting.

No model, outcome, scheduler, filesystem, or network is accessed here. A score
edge names the *only* validation donor whose result an inner selector may read.
Sharing a continuation never expands another selector's candidate eligibility.

``AdmissionLedger`` is a thread-safe in-memory contract model, not a durable
multi-process scheduler lock. A production owner must persist/reconcile exact
worker and scheduler identities before it can use this contract operationally.
"""

from dataclasses import dataclass, replace
from itertools import combinations
from threading import RLock
from types import MappingProxyType

from .configuration import DONORS, FAMILIES, SEEDS, STAGE_UPDATES, get_configuration, registry


MAIN_UPDATE_CAP = 847_872
H100_CAP = 4
INNER_PAIRS = tuple(combinations(DONORS, 2))
OUTER_TRIPLES = tuple(tuple(donor for donor in DONORS if donor != held) for held in DONORS)
ACTIVE_STATUSES = frozenset(("pending", "running", "uncertain"))
TERMINAL_STATUSES = frozenset(("complete", "failed", "cancelled"))


def _selector(value):
    if value not in DONORS:
        raise ValueError(f"Unknown outer selector donor: {value}")
    return value


def _canonical_donors(values):
    values = tuple(values)
    if len(set(values)) != len(values) or any(value not in DONORS for value in values):
        raise ValueError("Training donor identities must be unique registered donors")
    return tuple(donor for donor in DONORS if donor in values)


def training_pairs(selector):
    """The three training pairs available inside one outer selector."""
    _selector(selector)
    return tuple(pair for pair in INNER_PAIRS if selector not in pair)


def inner_edges():
    """Twelve (outer selector, training pair, inner validation donor) edges."""
    return tuple(
        (selector, pair, next(d for d in DONORS if d not in pair and d != selector))
        for selector in DONORS
        for pair in training_pairs(selector)
    )


@dataclass(frozen=True, order=True)
class TrajectoryKey:
    training_donors: tuple
    configuration_id: str
    seed: int

    def __post_init__(self):
        canonical = _canonical_donors(self.training_donors)
        if len(canonical) not in (2, 3):
            raise ValueError("A trajectory requires a two-donor pair or three-donor triple")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise ValueError("Trajectory seed must be one of 0, 1, 2")
        if len(canonical) == 2 and self.seed != 0:
            raise ValueError("Inner search uses seed 0 only")
        get_configuration(self.configuration_id)
        object.__setattr__(self, "training_donors", canonical)

    @property
    def identifier(self):
        return f"{'-'.join(self.training_donors)}/{self.configuration_id}/seed{self.seed}"

    @property
    def storage_identifier(self):
        """Safe lease/path label; keep ``identifier`` as logical metadata too.

        Registered donors/configurations contain no double-hyphen separators,
        so this encoding is one-to-one over the frozen trajectory inventory.
        It is not a pathname and does not identify a checkpoint's tensor bytes.
        """
        return f"{'-'.join(self.training_donors)}--{self.configuration_id}--seed{self.seed}"


@dataclass(frozen=True)
class TrainingTask:
    key: TrajectoryKey
    start_updates: int
    target_updates: int
    requesting_selectors: tuple

    def __post_init__(self):
        if type(self.start_updates) is not int or type(self.target_updates) is not int:
            raise ValueError("Update prefixes must be exact integer counts")
        permitted = {(0, STAGE_UPDATES[0]),
                     (STAGE_UPDATES[0], STAGE_UPDATES[1]),
                     (STAGE_UPDATES[1], STAGE_UPDATES[2])}
        if len(self.key.training_donors) == 3:
            permitted = {(0, STAGE_UPDATES[2])}
        if (self.start_updates, self.target_updates) not in permitted:
            raise ValueError("Training task does not follow a registered update-prefix transition")
        selectors = tuple(self.requesting_selectors)
        if not selectors or len(set(selectors)) != len(selectors):
            raise ValueError("A training task requires unique requesting selectors")
        for selector in selectors:
            _selector(selector)
            if selector in self.key.training_donors:
                raise ValueError("An outer selector cannot be a fitting donor for its task")
        if len(self.key.training_donors) == 3 and len(selectors) != 1:
            raise ValueError("An outer trajectory has exactly one assessment donor")
        object.__setattr__(self, "requesting_selectors", tuple(d for d in DONORS if d in selectors))

    @property
    def additional_updates(self):
        return self.target_updates - self.start_updates

    @property
    def identifier(self):
        return f"train/{self.key.identifier}/updates{self.start_updates}-{self.target_updates}"


@dataclass(frozen=True)
class ScoreTask:
    key: TrajectoryKey
    selector: str
    validation_donor: str
    updates: int
    phase: str = "inner"

    def __post_init__(self):
        if type(self.updates) is not int:
            raise ValueError("Scoring prefix must be an exact integer count")
        _selector(self.selector)
        _selector(self.validation_donor)
        if self.validation_donor in self.key.training_donors:
            raise ValueError("Validation donor appears in fitting inputs")
        if self.selector in self.key.training_donors:
            raise ValueError("Outer selector appears in fitting inputs")
        if self.phase == "inner":
            if len(self.key.training_donors) != 2 or self.selector == self.validation_donor:
                raise ValueError("Inner selector may not read its own outer assessment donor")
            if self.updates not in STAGE_UPDATES:
                raise ValueError("Unknown inner scoring prefix")
        elif self.phase == "outer":
            if (len(self.key.training_donors) != 3 or self.selector != self.validation_donor
                    or self.updates != STAGE_UPDATES[2]):
                raise ValueError("Outer score must use the held-out donor after the final fit")
        else:
            raise ValueError("Unknown score phase")

    @property
    def identifier(self):
        return (f"score/{self.phase}/{self.key.identifier}/updates{self.updates}"
                f"/selector{self.selector}/validation{self.validation_donor}")


@dataclass(frozen=True)
class StagePlan:
    name: str
    training: tuple
    scores: tuple
    eligibility: tuple

    def __post_init__(self):
        prefixes = {"screen": (0, 1024), "family": (1024, 3072),
                    "finalist": (3072, 9216), "outer": (0, 9216)}
        if self.name not in prefixes:
            raise ValueError("Unknown registered stage")
        object.__setattr__(self, "training", tuple(self.training))
        object.__setattr__(self, "scores", tuple(self.scores))
        object.__setattr__(self, "eligibility", tuple((d, tuple(c)) for d, c in self.eligibility))
        if len({task.identifier for task in self.training}) != len(self.training):
            raise ValueError("Duplicate training task")
        if len({task.identifier for task in self.scores}) != len(self.scores):
            raise ValueError("Duplicate score edge")
        if tuple(selector for selector, _ in self.eligibility) != DONORS:
            raise ValueError("Eligibility must cover each registered outer selector exactly once")
        for selector, candidates in self.eligibility:
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"Duplicate eligible configuration for {selector}")
            for candidate in candidates:
                get_configuration(candidate)
            families = [get_configuration(candidate).family for candidate in candidates]
            if self.name == "screen" and set(candidates) != set(registry()):
                raise ValueError("Screen eligibility must contain every registered configuration")
            if self.name == "family" and (len(candidates) != 6 or set(families) != set(FAMILIES)):
                raise ValueError("Family stage requires exactly one winner from each family")
            if self.name in ("finalist", "outer") and (
                    len(candidates) != 3 or families.count("F0") != 1
                    or sum(f in ("F1", "F2", "F3") for f in families) != 1
                    or sum(f in ("F4", "F5") for f in families) != 1):
                raise ValueError("Finalist roles must contain incumbent, non-incumbent direct and DDR")
        tasks = {(task.key, task.target_updates): task for task in self.training}
        for score in self.scores:
            task = tasks.get((score.key, score.updates))
            if task is None or score.selector not in task.requesting_selectors:
                raise ValueError("Score edge lacks an authorized training request")
            if score.key.configuration_id not in self.eligible_for(score.selector):
                raise ValueError("Union continuation must not expand selector eligibility")
        # Completeness is an intrinsic stage invariant, not merely a check in
        # the optional whole-study budget summarizer. Otherwise a truncated
        # stage could make an empty SelectorBarrier report ready.
        start, target = prefixes[self.name]
        expected_scores = []
        expected_requests = {}
        for selector, candidates in self.eligibility:
            pairs = (tuple(d for d in DONORS if d != selector),) if self.name == "outer" else training_pairs(selector)
            seeds = SEEDS if self.name == "outer" else (0,)
            for fitting_donors in pairs:
                validation = selector if self.name == "outer" else next(
                    d for d in DONORS if d not in fitting_donors and d != selector)
                for candidate in candidates:
                    for seed in seeds:
                        key = TrajectoryKey(fitting_donors, candidate, seed)
                        expected_requests.setdefault(key, set()).add(selector)
                        expected_scores.append(ScoreTask(key, selector, validation, target,
                                                         "outer" if self.name == "outer" else "inner"))
        if set(self.scores) != set(expected_scores):
            raise ValueError("Stage requires every exact selector-specific score edge, with no extras")
        expected_tasks = {TrainingTask(key, start, target, tuple(sorted(selectors)))
                          for key, selectors in expected_requests.items()}
        if set(self.training) != expected_tasks:
            raise ValueError("Stage training tasks must equal its deduplicated eligible requests")

    def eligible_for(self, selector):
        _selector(selector)
        return dict(self.eligibility)[selector]

    def scores_for(self, selector):
        _selector(selector)
        return tuple(score for score in self.scores if score.selector == selector)

    @property
    def additional_updates(self):
        return sum(task.additional_updates for task in self.training)


def _choices(selection, expected_count):
    if set(selection) != set(DONORS):
        raise ValueError("Selections must contain all four outer selectors and no others")
    result = {}
    for selector in DONORS:
        candidates = tuple(selection[selector])
        if len(candidates) != expected_count or len(set(candidates)) != expected_count:
            raise ValueError(f"{selector} requires {expected_count} distinct configurations")
        for candidate in candidates:
            get_configuration(candidate)
        result[selector] = tuple(sorted(candidates))
    return result


def _inner_plan(name, selected, start, target):
    requests = {}
    scores = []
    for selector, pair, validation in inner_edges():
        for candidate in selected[selector]:
            key = TrajectoryKey(pair, candidate, 0)
            requests.setdefault(key, set()).add(selector)
            scores.append(ScoreTask(key, selector, validation, target))
    training = tuple(TrainingTask(key, start, target, tuple(sorted(selectors)))
                     for key, selectors in sorted(requests.items()))
    return StagePlan(name, training, tuple(scores),
                     tuple((selector, tuple(selected[selector])) for selector in DONORS))


def screen_plan():
    """144 initial trajectories and 288 selectively readable inner scores."""
    candidates = tuple(registry())
    return _inner_plan("screen", {d: candidates for d in DONORS}, 0, STAGE_UPDATES[0])


def family_plan(selection):
    """Continue each selector's one winner per family; deduplicate weights only.

    ``selection`` is a metadata decision, not scores or an implementation of Q.
    Production admission must also require that selector's previous score barrier.
    """
    selected = _choices(selection, len(FAMILIES))
    for selector, candidates in selected.items():
        families = [get_configuration(candidate).family for candidate in candidates]
        if set(families) != set(FAMILIES):
            raise ValueError(f"{selector} must retain exactly one winner in every family")
    return _inner_plan("family", selected, STAGE_UPDATES[0], STAGE_UPDATES[1])


def finalist_plan(families, selection):
    if families.name != "family":
        raise ValueError("Finalists must follow the family-comparison plan")
    selected = _choices(selection, 3)
    for selector, candidates in selected.items():
        if not set(candidates).issubset(families.eligible_for(selector)):
            raise ValueError("A candidate promoted for another selector is not eligible here")
        chosen_families = [get_configuration(candidate).family for candidate in candidates]
        if (chosen_families.count("F0") != 1
                or sum(f in ("F1", "F2", "F3") for f in chosen_families) != 1
                or sum(f in ("F4", "F5") for f in chosen_families) != 1):
            raise ValueError("Finalists require one F0, one non-incumbent direct, and one DDR")
    return _inner_plan("finalist", selected, STAGE_UPDATES[1], STAGE_UPDATES[2])


def outer_plan(finalists):
    if finalists.name != "finalist":
        raise ValueError("Outer fitting requires the final inner comparison plan")
    training = []
    scores = []
    for selector in DONORS:
        triple = tuple(d for d in DONORS if d != selector)
        for candidate in finalists.eligible_for(selector):
            for seed in SEEDS:
                key = TrajectoryKey(triple, candidate, seed)
                training.append(TrainingTask(key, 0, STAGE_UPDATES[2], (selector,)))
                scores.append(ScoreTask(key, selector, selector, STAGE_UPDATES[2], "outer"))
    return StagePlan("outer", tuple(training), tuple(scores), finalists.eligibility)


def main_budget(plans):
    """Validate the complete four-stage metadata graph and its update ceiling."""
    plans = tuple(plans)
    if tuple(plan.name for plan in plans) != ("screen", "family", "finalist", "outer"):
        raise ValueError("Expected the screen, family, finalist, outer plans once in order")
    if plans[0] != screen_plan():
        raise ValueError("Initial screen differs from the 24-configuration design")
    expected_family = family_plan(dict(plans[1].eligibility))
    expected_finalist = finalist_plan(expected_family, dict(plans[2].eligibility))
    expected_outer = outer_plan(expected_finalist)
    if plans[1:] != (expected_family, expected_finalist, expected_outer):
        raise ValueError("Stage tasks or selector edges differ from registered eligibility")
    total = sum(plan.additional_updates for plan in plans)
    if total > MAIN_UPDATE_CAP:
        raise ValueError("Main scientific update allowance exceeded")
    return {
        "main_update_cap": MAIN_UPDATE_CAP,
        "planned_main_updates": total,
        "unused_updates_not_reallocatable": MAIN_UPDATE_CAP - total,
        "stages": {
            plan.name: {"training_tasks": len(plan.training), "score_tasks": len(plan.scores),
                        "additional_updates": plan.additional_updates}
            for plan in plans
        },
    }


class SelectorBarrier:
    """Metadata receipt barrier; it contains no scores or outcome-reading API."""

    def __init__(self, plan):
        if plan.name == "outer":
            raise ValueError("Outer outcomes cannot be fed back into a selection barrier")
        self._allowed = MappingProxyType({d: frozenset(s.identifier for s in plan.scores_for(d))
                                          for d in DONORS})
        self._complete = {d: set() for d in DONORS}
        self._lock = RLock()

    def record(self, selector, score_identifier):
        _selector(selector)
        with self._lock:
            if score_identifier not in self._allowed[selector]:
                raise ValueError("Score receipt is outside this selector's permitted view")
            if score_identifier in self._complete[selector]:
                raise ValueError("Duplicate completion receipt")
            self._complete[selector].add(score_identifier)

    def ready(self, selector):
        _selector(selector)
        with self._lock:
            return self._complete[selector] == self._allowed[selector]

    def require_ready(self, selector):
        if not self.ready(selector):
            raise ValueError("Required selector-specific inner score receipts are incomplete")

    def missing(self, selector):
        _selector(selector)
        with self._lock:
            return tuple(sorted(self._allowed[selector] - self._complete[selector]))


@dataclass(frozen=True)
class Ownership:
    owner_id: str
    task: TrainingTask
    status: str


class AdmissionLedger:
    """Bound independent single-GPU training owners, including uncertain jobs.

    Reservation consumes a slot *before* any hypothetical submission. Only an
    exact terminal completion releases its trajectory for the next prefix.
    Failed/cancelled trajectories remain blocked; this class has no retry API.
    Completing a future production task requires independent durable receipts;
    this in-memory exercise itself proves neither work nor scheduler completion.
    """

    def __init__(self, maximum_active=H100_CAP, maximum_updates=MAIN_UPDATE_CAP):
        if type(maximum_active) is not int or not 1 <= maximum_active <= H100_CAP:
            raise ValueError("Independent single-H100 concurrency must be between one and four")
        if (type(maximum_updates) is not int or maximum_updates < 1
                or maximum_updates > MAIN_UPDATE_CAP):
            raise ValueError("Scientific update cap must not exceed the authorized ceiling")
        self.maximum_active = maximum_active
        self.maximum_updates = maximum_updates
        self._owners = {}
        self._prefix = {}
        self._blocked = set()
        self._committed = 0
        self._lock = RLock()

    def _active(self):
        return tuple(item for item in self._owners.values() if item.status in ACTIVE_STATUSES)

    @property
    def active_count(self):
        with self._lock:
            return len(self._active())

    @property
    def committed_updates(self):
        with self._lock:
            return self._committed

    def reserve(self, owner_id, task):
        with self._lock:
            if not isinstance(owner_id, str) or not owner_id.strip():
                raise ValueError("A nonempty exact owner identity is required")
            if owner_id in self._owners:
                raise ValueError("An owner identity cannot be reused")
            if not isinstance(task, TrainingTask):
                raise ValueError("Admission requires a validated training task")
            if task.key in self._blocked:
                raise ValueError("Failed/cancelled trajectory requires a separately reviewed transition")
            active = self._active()
            if any(item.task.key == task.key for item in active):
                raise ValueError("Trajectory already has a pending/running/uncertain owner")
            if len(active) >= self.maximum_active:
                raise ValueError("No independent H100 slot remains")
            if self._prefix.get(task.key, 0) != task.start_updates:
                raise ValueError("Continuation prefix does not match exact completed trajectory state")
            # Charge the entire admitted chunk conservatively. Failed work does
            # not become free budget for another recipe or an automatic retry.
            if self._committed + task.additional_updates > self.maximum_updates:
                raise ValueError("Admitting this task exceeds the scientific update allowance")
            self._committed += task.additional_updates
            item = Ownership(owner_id, task, "pending")
            self._owners[owner_id] = item
            return item

    def transition(self, owner_id, status):
        with self._lock:
            if owner_id not in self._owners:
                raise ValueError("Unknown owner identity")
            previous = self._owners[owner_id]
            if previous.status in TERMINAL_STATUSES:
                raise ValueError("Terminal ownership cannot be restarted or rewritten")
            if status not in ("pending", "running", "uncertain", "failed", "cancelled"):
                raise ValueError("Use complete() for verified terminal prefix completion")
            allowed = {
                "pending": {"running", "uncertain", "failed", "cancelled"},
                "running": {"uncertain", "failed", "cancelled"},
                "uncertain": {"pending", "running", "failed", "cancelled"},
            }
            if status not in allowed[previous.status]:
                raise ValueError("Unsupported ownership transition")
            updated = replace(previous, status=status)
            self._owners[owner_id] = updated
            if status in ("failed", "cancelled"):
                self._blocked.add(previous.task.key)
            return updated

    def complete(self, owner_id, cumulative_updates):
        with self._lock:
            if owner_id not in self._owners:
                raise ValueError("Unknown owner identity")
            previous = self._owners[owner_id]
            if previous.status not in ("running", "uncertain"):
                raise ValueError("Completion requires a running/reconciled uncertain owner")
            if type(cumulative_updates) is not int or cumulative_updates != previous.task.target_updates:
                raise ValueError("Completion must match the exact registered update prefix")
            updated = replace(previous, status="complete")
            self._owners[owner_id] = updated
            self._prefix[previous.task.key] = cumulative_updates
            return updated

    def snapshot(self):
        with self._lock:
            return {
                "scope": "in_memory_metadata_only_not_scheduler_or_durable_lock",
                "maximum_active_h100": self.maximum_active,
                "gpu_per_worker": 1,
                "distributed_training": False,
                "active": len(self._active()),
                "committed_update_allowance": self._committed,
                "maximum_updates": self.maximum_updates,
                "owners": tuple({"owner_id": item.owner_id, "trajectory": item.task.key.identifier,
                                  "start_updates": item.task.start_updates,
                                  "target_updates": item.task.target_updates, "status": item.status}
                                 for item in self._owners.values()),
            }
