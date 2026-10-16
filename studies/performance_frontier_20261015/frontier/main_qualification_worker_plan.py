"""Compact full-entry cases, distinct from full-shape owner qualification.

These three cases exercise the actual file-backed worker, including one
checkpoint continuation. They require separate CUDA authority; their seven
disposable updates are not main-study reservations or scientific fits.
"""

from dataclasses import asdict, dataclass

from .retention import identity_sha256
from .training_owner import OwnerSchedule


CONTRACT_SCHEMA = "credo.frontier.main-qualification-worker-contract.v1"
SCOPE = "synthetic_cuda_qualification"
PLANNED_OPTIMIZER_CALLS = 7


def worker_schedule():
    # Four laws ensure a targeting law is present in the fixture: its two/three
    # donors each contribute one targeting and one NTC law. Keep the production
    # no-redraw rule; do not rescue an all-control molecular minibatch later.
    return OwnerSchedule(SCOPE, (2, 3, 4), 1, 2, 32, 1, 4, 4, 4)


@dataclass(frozen=True)
class WorkerCase:
    stage: str
    configuration_id: str
    training_donors: tuple
    accepted_start: int
    accepted_target: int

    @property
    def optimizer_calls(self):
        return self.accepted_target - self.accepted_start


def cases():
    return (
        WorkerCase("screen", "F0-V2", ("D3", "D4"), 0, 2),
        WorkerCase("family", "F0-V2", ("D3", "D4"), 2, 3),
        WorkerCase("outer", "F5-V2", ("D2", "D3", "D4"), 0, 4),
    )


def case_for(plan, task):
    matches = [c for c in cases() if c.stage == plan.name
        and c.configuration_id == task.key.configuration_id
        and c.training_donors == task.key.training_donors and task.key.seed == 0]
    if len(matches) != 1 or task not in plan.training:
        raise ValueError("Only the three fixed disposable worker-entry cases are admitted")
    return matches[0]


def plan():
    return dict(schema="credo.frontier.main-qualification-worker-plan.v1",
        status="proposal_not_launch_authority", execution_scope=SCOPE,
        schedule=asdict(worker_schedule()), cases=[asdict(c) for c in cases()],
        planned_optimizer_calls=PLANNED_OPTIMIZER_CALLS, scientific_updates=0,
        real_biological_payload_access=False, full_axis_memory_qualification=False,
        forecast_custody="Explicitly fabricated bytes, verified by the production custody reader",
        observation_artifact="Explicitly fabricated, not a fitted scientific observation model",
        automatic_retry=False, automatic_successor=False)


def identity():
    return identity_sha256(plan())
