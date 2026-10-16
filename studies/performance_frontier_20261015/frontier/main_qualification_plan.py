"""Finite proposal for main CUDA numerical cases, not execution authority.

All 24 recipes are exercised on two- and three-donor fabricated inputs, with
state-to-joint transition. Two exact restart comparisons add no new scientific
recipes. This does not prove precision for all observed biological laws.
"""

from dataclasses import asdict, dataclass

from .configuration import get_configuration, configurations
from .retention import identity_sha256
from .training_owner import OwnerSchedule


SCHEMA = "credo.frontier.main-cuda-qualification-plan.v1"
SCOPE = "synthetic_cuda_qualification"
DONOR_SUBSETS = (("D3", "D4"), ("D2", "D3", "D4"))
RESTART_CASES = (("F0-V2", DONOR_SUBSETS[0]), ("F5-V2", DONOR_SUBSETS[1]))
PLANNED_OPTIMIZER_CALLS = 104
MAXIMUM_OPTIMIZER_CALLS = 112


def qualification_schedule():
    return OwnerSchedule(SCOPE, (2, 3, 4), 1, 2, 32, 1, 1024, 128, 256)


@dataclass(frozen=True)
class QualificationCase:
    configuration_id: str
    training_donors: tuple
    restart_comparison: bool

    def __post_init__(self):
        get_configuration(self.configuration_id)
        if (type(self.training_donors) is not tuple or self.training_donors not in DONOR_SUBSETS
                or type(self.restart_comparison) is not bool
                or self.restart_comparison is not ((self.configuration_id, self.training_donors) in RESTART_CASES)):
            raise ValueError("Exact finite qualification case required")

    @property
    def identifier(self):
        return "qual-" + self.configuration_id + "-" + "-".join(self.training_donors)

    @property
    def optimizer_calls(self):
        # Base 2; restart: +1 resumed and +3 fresh continuous reference.
        return 6 if self.restart_comparison else 2


def cases():
    return tuple(QualificationCase(config.configuration_id, donors,
        (config.configuration_id, donors) in RESTART_CASES)
        for config in configurations() for donors in DONOR_SUBSETS)


def plan():
    # Deferred import avoids a module-initialization cycle: precision fixtures
    # consume the same exact QualificationCase inventory. This adds no launch
    # authority or optimizer calls and does not change scientific schedules.
    from .main_qualification_precision import precision_plan
    from .main_qualification_worker_plan import plan as worker_plan
    inventory = cases()
    if sum(c.optimizer_calls for c in inventory) != PLANNED_OPTIMIZER_CALLS:
        raise ValueError("Disposable CUDA workload arithmetic changed")
    return dict(schema=SCHEMA, status="proposal_not_launch_authority", execution_scope=SCOPE,
        cases=[asdict(c) for c in inventory], scientific_updates=0,
        planned_optimizer_calls=PLANNED_OPTIMIZER_CALLS + worker_plan()["planned_optimizer_calls"],
        owner_optimizer_calls=PLANNED_OPTIMIZER_CALLS, maximum_optimizer_calls=MAXIMUM_OPTIMIZER_CALLS,
        schedule=asdict(qualification_schedule()), real_biological_payload_access=False,
        resource_proposal=dict(h100s=1, cpus=24, ram_gib=200, threads=8, output_gib=16,
                               minimum_gpu_headroom_gib=8),
        input_dimensions=dict(latent=48, genes=18129, catalog_per_donor=26504,
            modeled_laws_per_donor=768, logical_laws=1024, particles=128, cache_particles=32,
            target_embedding_inventory=26504, observed_support_cycle=[1, 3, 16, 64, 256]),
        seed=419, automatic_retry=False, automatic_successor=False, elapsed_time_deadline=None,
        paired_numerical_precision=precision_plan(),
        worker_entry=worker_plan(),
        exclusions=["No scientific fitting or real observation access",
            "No blanket qualification of unseen ragged support or microbatch shapes",
            "No precision or predictive acceptance from finite losses or successful gradients",
            "No full worker-entry acceptance from a standalone owner-case probe"])


def plan_identity():
    return identity_sha256(plan())
