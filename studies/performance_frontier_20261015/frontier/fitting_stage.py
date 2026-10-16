"""One registered fitting task under a live lease, with completion after cache exit.

This is a worker lifecycle component, not a scheduler, launch authorization or
resource admission. The enclosing worker must admit its source/runtime/storage
and provide actual VerifiedContinuationCustody before scientific continuation.
Bare edge IDs remain accepted only by explicitly synthetic fixtures.
The component never reads query outcomes or selects/promotes a configuration.

Enter this context on an OUTER ExitStack while the prepared mmap cache is open.
Run inside the cache context; exit the cache first, then this context, then the
trajectory lease. Exiting in the wrong order fails rather than publishing a
successful stage from an unverified input. A post-training input/publication
failure marks the owner failed, preventing automatic checkpoint continuation.
"""

from dataclasses import asdict
import hashlib
import json
import os

from .cache_storage import MappedFoldCache, _json_bytes, _open_verified, _parse, _path, _write_once
from .checkpoints import TrajectoryLease, file_sha256
from .configuration import STAGE_UPDATES
from .fitting_factory import PreparedFitting
from .planner import StagePlan, TrainingTask
from .prediction_plan import require_continuation_forecasts
from .retention import RetentionPolicy, identity_sha256


SCHEMA = "credo.frontier.fitting-stage-completion.v1"


def task_boundaries(task, schedule):
    """Scientific task prefixes stay exact; tiny fixtures are explicitly labelled."""
    if type(task) is not TrainingTask:
        raise TypeError("Exact registered training task required")
    task.__post_init__(); schedule.__post_init__()
    if len(schedule.stages) != len(STAGE_UPDATES):
        raise ValueError("One owner boundary per registered stage required")
    prefixes = dict(zip(STAGE_UPDATES, schedule.stages))
    start = 0 if task.start_updates == 0 else prefixes[task.start_updates]
    target = prefixes[task.target_updates]
    return start, tuple(n for n in schedule.stages if start < n <= target)


class FittingStage:
    """Single-use task owner; retains all stage checkpoints and latest two payloads.

    A supplied StagePlan proves the registered metadata workload, not the prior
    scientific selection. Caller admission must bind the corresponding saved
    selector decisions and resource reservation. Synthetic receipts cannot be
    used to claim that the task's scientific update allowance was executed.
    """

    def __init__(self, *, prepared, lease, plan, task, study_root, resume=None,
                 previous_stage=None, verified_complete_edge_ids=(), resource_guard=None,
                 continuation_custody=None):
        if (not isinstance(prepared, PreparedFitting) or not isinstance(lease, TrajectoryLease)
                or type(plan) is not StagePlan or type(task) is not TrainingTask
                or not isinstance(prepared.cache, MappedFoldCache)):
            raise TypeError("Actual factory, live mapped cache, lease and registered task required")
        plan.__post_init__(); lease.assert_active(); prepared.cache.assert_unchanged()
        if task not in plan.training:
            raise ValueError("Task is not in this complete registered stage plan")
        args = prepared.arguments
        if (prepared.identity != lease.identity
                or args["model"].config.configuration_id != task.key.configuration_id
                or prepared.identity["seed"] != task.key.seed
                or prepared.cache.manifest.fold.training_donors != task.key.training_donors
                or prepared.identity["trajectory_id"] != task.key.storage_identifier):
            raise ValueError("Task differs from actual prepared trajectory/fold/seed")
        self.start, self.targets = task_boundaries(task, args["schedule"])
        if (not self.targets or (self.start == 0 and resume is not None)
                or (self.start > 0 and (not isinstance(resume, dict)
                                     or resume.get("accepted_updates") != self.start))):
            raise ValueError("Exact fresh start or checkpointed continuation prefix required")
        if self.start:
            if continuation_custody is None:
                if args["schedule"].execution_scope=="scientific":
                    raise ValueError("Scientific continuation requires actual verified forecast-file custody")
            else:
                from .continuation_custody import continuation_edge_ids
                verified_complete_edge_ids=continuation_edge_ids(continuation_custody,
                    previous_stage=previous_stage,next_task=task,checkpoint_identity=prepared.identity,
                    checkpoint_sha256=resume["checkpoint_sha256"],execution_scope=args["schedule"].execution_scope,
                    supplied_edge_ids=verified_complete_edge_ids)
            required = require_continuation_forecasts(previous_stage, task, verified_complete_edge_ids)
        else:
            if previous_stage is not None or verified_complete_edge_ids or continuation_custody is not None:
                raise ValueError("Fresh fitting must not receive prior outcome/forecast edges")
            required = ()
        root = _path(study_root)
        policy = RetentionPolicy(str(root), str(lease.root), identity_sha256(lease.identity),
                                 args["schedule"].stages)
        self.prepared, self.lease, self.plan, self.task = prepared, lease, plan, task
        self.policy, self.resume, self.required_edges = policy, resume, required
        self.resource_guard = resource_guard
        self.continuation_custody = continuation_custody
        self.owner = None
        self._entered = self._ran = self._finished = False
        self.receipt = None
        self.path = lease.root / ("FITTING_STAGE_"+plan.name+".json")
        self.failure_path = lease.root / ("FITTING_STAGE_"+plan.name+"_FAILED.json")
        if os.path.lexists(self.path) or os.path.lexists(self.failure_path):
            raise FileExistsError("Stage already attempted; no overwrite or automatic retry")
        if self.start:
            prior_path = lease.root / ("FITTING_STAGE_"+previous_stage.name+".json")
            if (not prior_path.is_file() or prior_path.is_symlink()
                    or prior_path.stat().st_size > 1024**2):
                raise ValueError("Accepted prior worker-stage completion required")
            previous_sha = file_sha256(prior_path)
            descriptor = os.open(_path(lease.root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                verified = _open_verified(descriptor, prior_path.name, previous_sha, maximum=1024**2)
                try: prior = _parse(verified.stream.read())
                finally: verified.stream.close()
            finally:
                os.close(descriptor)
            if (prior.get("schema") != SCHEMA or prior.get("identity") != lease.identity
                    or prior.get("accepted_updates") != self.start
                    or prior.get("checkpoint") != resume or prior.get("status") != "complete"
                    or prior.get("cache_verification", {}).get("exit_full_hash_verified") is not True):
                raise ValueError("Prior worker-stage completion does not bind this continuation")
            self.previous_completion_sha256 = previous_sha
        else:
            self.previous_completion_sha256 = None

    def __enter__(self):
        if self._entered:
            raise ValueError("Fitting stage is single-use")
        self._entered = True
        self.owner = self.prepared.open_owner(self.lease, resume=self.resume, retention_policy=self.policy,
                                              resource_guard=self.resource_guard)
        if self.owner.accepted_updates != self.start:
            raise ValueError("Loaded owner prefix differs from task")
        return self

    def run(self):
        if not self._entered or self._ran or self._finished:
            raise ValueError("Exactly one stage invocation under its live context required")
        self._ran = True
        for target in self.targets:
            self.owner.run_stage(target)
        return self.owner._state()

    def _publish(self, name, value):
        raw = _json_bytes(value)
        if len(raw) > 1024**2:
            raise ValueError("Stage metadata exceeds one MiB; no history embedding")
        self.lease.assert_active()
        fd = os.open(_path(self.lease.root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            digest = hashlib.sha256(raw).hexdigest()
            _write_once(fd, name, (raw,), expected_sha256=digest, expected_bytes=len(raw))
            return dict(path=str(self.lease.root/name), sha256=digest, size_bytes=len(raw))
        finally:
            os.close(fd)

    def _fail(self, error):
        if self.owner is not None:
            numerical_failure = self.owner.failure
            self.owner.status = "failed"
            self.owner.failure = dict(error_type=type(error).__name__,
                message=str(error)[:2000], stage_lifecycle_failure=True,
                preceding_numerical_failure=numerical_failure,
                optimizer_invoked_beyond_accepted=self.owner.optimizer_calls > self.owner.accepted_updates)
            self.owner._write_status()
        return self._publish(self.failure_path.name, dict(schema="credo.frontier.fitting-stage-failure.v1",
            status="failed", identity=self.lease.identity, task=asdict(self.task),
            exception_type=type(error).__name__, automatic_retry=False))

    def __exit__(self, exc_type, error, traceback):
        self._finished = True
        if error is not None:
            try: self._fail(error)
            except BaseException as failure_error:
                error.add_note("Stage failure recording also failed: "+type(failure_error).__name__)
            return False
        try:
            cache = self.prepared.cache
            # The cache cannot be consumed after close. Inspect its retained
            # verified exit record rather than reopening inputs or redoing math.
            if (not self._ran or self.owner.status != "stage_complete"
                    or self.owner.accepted_updates != self.targets[-1]
                    or cache._closed is not True
                    or cache.verification.get("exit_full_hash_verified") is not True):
                raise ValueError("Complete numerical stage and full cache exit verification required")
            checkpoint = self.owner.last_checkpoint
            if (checkpoint is None or checkpoint["accepted_updates"] != self.targets[-1]
                    or file_sha256(self.lease.root/checkpoint["directory"]/"checkpoint.pt") != checkpoint["checkpoint_sha256"]):
                raise ValueError("Final complete checkpoint must survive rolling retention")
            record = dict(schema=SCHEMA, status="complete", identity=self.lease.identity,
                task=asdict(self.task), stage=self.plan.name,
                plan_sha256=identity_sha256(asdict(self.plan)), retention_policy_sha256=self.policy.identity,
                previous_completion_sha256=self.previous_completion_sha256,
                prior_forecast_edge_ids=list(self.required_edges),
                prior_forecast_files_verified_by_this_component=False,
                continuation_custody_sha256=None if self.continuation_custody is None else self.continuation_custody.identity,
                continuation_custody=None if self.continuation_custody is None else self.continuation_custody.record(),
                accepted_updates=self.owner.accepted_updates, attempted_updates=self.owner.attempted_updates,
                optimizer_calls_this_task=self.owner.optimizer_calls-self.start,
                execution_scope=self.owner.schedule.execution_scope,
                scientific_updates_this_task=(self.owner.optimizer_calls-self.start
                    if self.owner.schedule.execution_scope == "scientific" else 0),
                checkpoint=checkpoint, cache_verification=dict(cache.verification),
                source_sha256=file_sha256(__file__), scientific_launch_authority=False,
                query_outcomes_opened=False, automatic_successor=False)
            self.receipt = self._publish(self.path.name, record)
        except BaseException as publication_error:
            try: self._fail(publication_error)
            except BaseException as failure_error:
                publication_error.add_note("Stage failure recording also failed: "+type(failure_error).__name__)
            raise
        return False
