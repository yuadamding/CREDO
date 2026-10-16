"""Fabricated inputs for the real main fitting owner; never biological data.

Full dimensions are CUDA-only. A separately labelled tiny CPU mode permits
input/objective checks with zero optimizer calls. Neither mode grants execution
authority or produces a scientific cache/observation artifact.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from .count_objective import CatalogBlock, FitnessCache
from .geometry import ObservedLaw
from .main_qualification_plan import QualificationCase, SCOPE
from .retention import identity_sha256
from .training_objective import LogicalTrainingInputs, MolecularTrainingInputs, PooledControlInputs
from .training_owner import FittingInventory, PreparedUpdate, CacheRefreshBatch


@dataclass(frozen=True)
class FixtureDimensions:
    scope: str = SCOPE
    genes: int = 18129
    catalog: int = 26504
    laws_per_donor: int = 768
    logical_laws: int = 1024
    particles: int = 128
    target_inventory: int = 26504
    observed_support_cycle: tuple = (1, 3, 16, 64, 256)

    def __post_init__(self):
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in
               ("genes", "catalog", "laws_per_donor", "logical_laws", "particles", "target_inventory")):
            raise ValueError("Exact positive fixture dimensions required")
        if (self.laws_per_donor < 2 or self.laws_per_donor > self.catalog
                or self.target_inventory < self.laws_per_donor - 1
                or type(self.observed_support_cycle) is not tuple or not self.observed_support_cycle
                or any(type(n) is not int or n < 1 for n in self.observed_support_cycle)):
            raise ValueError("Complete fixture support/catalog dimensions required")
        if self.scope == SCOPE:
            if tuple(getattr(self, k) for k in ("genes", "catalog", "laws_per_donor", "logical_laws",
                    "particles", "target_inventory", "observed_support_cycle")) != (
                    18129, 26504, 768, 1024, 128, 26504, (1, 3, 16, 64, 256)):
                raise ValueError("Full CUDA fixture dimensions are fixed")
        elif self.scope == "synthetic_cpu_only":
            if (self.genes > 64 or self.catalog > 128 or self.laws_per_donor > 16
                    or self.logical_laws > 32 or self.particles > 128 or max(self.observed_support_cycle) > 64):
                raise ValueError("CPU metadata/objective fixture exceeds small local scope")
        else:
            raise ValueError("Explicit fabricated input scope required")

    @classmethod
    def tiny(cls):
        return cls("synthetic_cpu_only", 3, 17, 4, 4, 4, 8, (1, 3, 5))


class QualificationProvider:
    def __init__(self, case, dimensions, *, device):
        if type(case) is not QualificationCase or type(dimensions) is not FixtureDimensions:
            raise TypeError("Bound finite case and fabricated dimensions required")
        case.__post_init__(); dimensions.__post_init__()
        dev = torch.device(device)
        if dimensions.scope == SCOPE and dev != torch.device("cuda:0"):
            raise ValueError("Full fixture is CUDA-only; do not allocate it on local CPU")
        if dimensions.scope == "synthetic_cpu_only" and (dev.type != "cpu" or torch.get_num_threads() > 2):
            raise ValueError("Tiny fixture is CPU-only with at most two threads")
        self.case, self.dimensions, self.device = case, dimensions, dev
        self.position = 0
        donors = case.training_donors
        n = dimensions.laws_per_donor
        if dimensions.logical_laws > len(donors)*n or dimensions.logical_laws < len(donors):
            raise ValueError("Logical fixture must fit the modeled donor populations")
        self.target_ids = tuple("fabricated-target-%05d" % i for i in range(dimensions.target_inventory))
        laws = tuple(d+"-fabricated-guide-%05d" % i for d in donors for i in range(n))
        donor_ids = tuple(d for d in donors for _ in range(n))
        targets = tuple("NTC" if i == 0 else self.target_ids[i-1] for _d in donors for i in range(n))
        # The model recipe is deliberately not part of input/noise generation:
        # family/width comparisons receive identical fabricated observations.
        self.input_identity = identity_sha256(dict(schema="credo.main-qualification-fabricated-inputs.v1",
            donors=donors, dimensions=asdict(dimensions), seed=419))
        self.inventory = FittingInventory(donors, laws, donor_ids, targets,
            self.input_identity, identity_sha256(["catalog", self.input_identity]),
            identity_sha256(["sampler", self.input_identity]), identity_sha256(["random-decoder", donors, dimensions.genes]))
        self.refresh_events = []

    def _generator(self, purpose, update):
        key = json.dumps([self.input_identity, purpose, update], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63-1)
        return torch.Generator(device=self.device).manual_seed(seed)

    def _particles(self, rows, particles, generator):
        z = .15*torch.randn(len(rows), particles, 48, generator=generator, device=self.device)
        noise = torch.randn(8, len(rows), particles, 48, generator=generator, device=self.device)
        return z, noise

    def _composition(self, rows, offset):
        # Nonidentical target/control compositions with no decoder-derived
        # pseudo-truth. Values and signs have no biological interpretation.
        genes = torch.arange(self.dimensions.genes, device=self.device, dtype=torch.float64)
        r = torch.as_tensor(rows, device=self.device, dtype=torch.float64)
        values = 1.1 + .1*torch.sin(r[:, None]*.17 + genes[None]*.03 + offset)
        return values / values.sum(-1, keepdim=True)

    def sample(self, *, accepted_updates, phase):
        if type(accepted_updates) is not int or accepted_updates != self.position or phase not in ("state", "joint"):
            raise ValueError("Exact fabricated provider position/phase required")
        d, donors = self.dimensions, self.case.training_donors
        generator = self._generator("logical", accepted_updates)
        rows = torch.randperm(len(self.inventory.law_ids), generator=generator,
            device=self.device)[:d.logical_laws].tolist()
        indices = torch.tensor(rows, device=self.device, dtype=torch.int64)
        source, noise = self._particles(rows, d.particles, generator)
        observations = []
        for i in rows:
            count = d.observed_support_cycle[i % len(d.observed_support_cycle)]
            law = .15*torch.randn(count, 48, generator=generator, device=self.device) + .03
            weights = torch.ones(count, device=self.device)
            if count > 1: weights[0] = 0
            observations.append(ObservedLaw(self.inventory.law_ids[i], law, weights))
        value = LogicalTrainingInputs(tuple(self.inventory.law_ids[i] for i in rows),
            tuple(self.inventory.donor_ids[i] for i in rows), tuple(self.inventory.target_ids[i] for i in rows),
            source, noise, indices, tuple(observations),
            torch.full((len(rows),), 1/len(rows), device=self.device),
            torch.full((len(rows),), 1/len(rows), device=self.device))
        if phase != "joint" or self.case.configuration_id[:2] not in ("F3", "F4", "F5"):
            return PreparedUpdate(value, None, len(self.inventory.law_ids)/d.logical_laws)
        positions = [j for j, i in enumerate(rows) if self.inventory.target_ids[i] != "NTC"]
        selected = [rows[j] for j in positions]
        controls_source, controls_noise = self._particles(list(range(len(donors))), d.particles,
            self._generator("controls", accepted_updates))
        controls = PooledControlInputs(donors, tuple(d+"-fabricated-pooled-NTC" for d in donors),
            torch.arange(len(donors), device=self.device), tuple("NTC" for _ in donors),
            controls_source, controls_noise, torch.ones(len(donors), device=self.device, dtype=torch.float64),
            self._composition(list(range(len(donors))), .4),
            torch.full((len(donors),), 1/len(donors), device=self.device, dtype=torch.float64))
        multiplier = len(self.inventory.law_ids)/d.logical_laws * len(selected)/(len(donors)*(d.laws_per_donor-1))
        molecular = MolecularTrainingInputs(torch.tensor(positions, device=self.device, dtype=torch.int64),
            self._composition(selected, .9), torch.full((len(selected),), 1/len(selected), device=self.device, dtype=torch.float64),
            controls, multiplier)
        return PreparedUpdate(value, molecular, len(self.inventory.law_ids)/d.logical_laws)

    def cache_batches(self, *, accepted_updates, particles, maximum_laws):
        if accepted_updates != self.position or particles != 32 or maximum_laws < 1:
            raise ValueError("Exact fabricated refresh position/resolution required")
        self.refresh_events.append(accepted_updates)
        generator = self._generator("cache", accepted_updates)
        for start in range(0, len(self.inventory.law_ids), maximum_laws):
            rows = list(range(start, min(start+maximum_laws, len(self.inventory.law_ids))))
            source, noise = self._particles(rows, particles, generator)
            yield CacheRefreshBatch(torch.tensor(rows, device=self.device),
                tuple(self.inventory.law_ids[i] for i in rows), tuple(self.inventory.donor_ids[i] for i in rows),
                tuple(self.inventory.target_ids[i] for i in rows), source, noise)

    def count_inputs(self):
        d = self.dimensions
        count_cache = FitnessCache(torch.zeros(len(self.inventory.law_ids), device=self.device),
            seen=torch.zeros(len(self.inventory.law_ids), device=self.device, dtype=torch.bool),
            age=torch.zeros(len(self.inventory.law_ids), device=self.device, dtype=torch.int64),
            binding=dict(model_recipe=self.case.configuration_id, training_subset="-".join(self.case.training_donors),
                catalog_layout=self.inventory.manifest_sha256, source_sampler=self.inventory.sampler_sha256))
        blocks = []
        categories = torch.arange(d.catalog, device=self.device)
        source_counts = 1 + categories % 31
        source = (source_counts.double()+.5)/(source_counts.sum()+.5*d.catalog)
        for i, donor in enumerate(self.case.training_donors):
            endpoint = (categories*3+i) % 37
            blocks.append(CatalogBlock(donor, "fabricated-complete-catalog", source.log().float(), endpoint,
                torch.arange(d.laws_per_donor, device=self.device),
                torch.arange(i*d.laws_per_donor, (i+1)*d.laws_per_donor, device=self.device)))
        return tuple(blocks), count_cache

    def accepted(self, *, accepted_updates):
        if accepted_updates != self.position+1:
            raise ValueError("Exactly one fabricated update commit required")
        self.position = accepted_updates

    def state_dict(self):
        return dict(position=self.position, input_identity=self.input_identity, refresh_events=list(self.refresh_events))

    def load_state_dict(self, value):
        if (set(value) != {"position", "input_identity", "refresh_events"}
                or value["input_identity"] != self.input_identity or type(value["position"]) is not int
                or not 0 <= value["position"] <= 4 or type(value["refresh_events"]) is not list):
            raise ValueError("Exact fabricated sampler checkpoint required")
        self.position = value["position"]
        self.refresh_events = list(value["refresh_events"])
