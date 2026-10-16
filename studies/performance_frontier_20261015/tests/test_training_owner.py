"""Twelve fabricated CPU Adam calls; zero scientific updates or dataset access.

The phase boundary and stages here are ONLY numerical fixtures. Scientific
schedule remains 256 state updates, 64-update refresh, 1024/3072/9216 stages.
"""

from dataclasses import replace
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from frontier.checkpoints import TrajectoryLease
from frontier.count_objective import CatalogBlock, FitnessCache
from frontier.geometry import GeometryOptions, ObservedLaw, RaggedSinkhornGeometry
from frontier.models import build_model
from frontier.observation import ObservationFold, SparseR48
from frontier.training_objective import LogicalTrainingInputs, ObjectiveExecution, ObjectiveProtocol
from frontier.training_owner import (CacheRefreshBatch, FittingInventory, ObservationBinding, OwnerSchedule,
    PreparedUpdate, TrainingOwner, observation_state_digest, owner_binding)


OPTIMIZER_CALLS = 0
ORIGINAL_STEP = torch.optim.Adam.step


def counted_step(optimizer, *args, **kwargs):
    global OPTIMIZER_CALLS
    OPTIMIZER_CALLS += 1
    if OPTIMIZER_CALLS > 24:
        raise RuntimeError("Local fabricated optimizer-call allowance exceeded")
    return ORIGINAL_STEP(optimizer, *args, **kwargs)


def seed():
    random.seed(51)
    np.random.seed(51)
    torch.manual_seed(51)


def assert_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class FabricatedProvider:
    def __init__(self):
        self.inventory = FittingInventory(("D3", "D4"), ("a", "b", "c", "d"),
            ("D3", "D3", "D4", "D4"), ("T1", "NTC", "T2", "NTC"),
            "1" * 64, "2" * 64, "3" * 64, "4" * 64)
        self.position = 0
        self.refresh_events = []
        self.fail_at = None

    def sample(self, *, accepted_updates, phase):
        if self.fail_at == accepted_updates:
            raise RuntimeError("Deliberate provider failure before optimizer")
        generator = torch.Generator().manual_seed(100 + accepted_updates)
        z = .1 * torch.randn(2, 4, 48, generator=generator)
        noise = torch.randn(8, 2, 4, 48, generator=generator)
        obs = tuple(ObservedLaw(name, .1 * torch.randn(3, 48, generator=generator) + .1,
                               torch.ones(3)) for name in ("a", "c"))
        value = LogicalTrainingInputs(("a", "c"), ("D3", "D4"), ("T1", "T2"), z, noise,
            torch.tensor([0, 2]), obs, torch.tensor([.5, .5]), torch.tensor([.5, .5]))
        return PreparedUpdate(value, None, 2.)

    def cache_batches(self, *, accepted_updates, particles, maximum_laws):
        self.refresh_events.append(accepted_updates)
        generator = torch.Generator().manual_seed(900 + accepted_updates)
        source = .1 * torch.randn(4, particles, 48, generator=generator)
        noise = torch.randn(8, 4, particles, 48, generator=generator)
        for start in range(0, 4, maximum_laws):
            stop = min(start + maximum_laws, 4)
            yield CacheRefreshBatch(torch.arange(start, stop), self.inventory.law_ids[start:stop],
                self.inventory.donor_ids[start:stop], self.inventory.target_ids[start:stop],
                source[start:stop], noise[:, start:stop])

    def accepted(self, *, accepted_updates):
        if accepted_updates != self.position + 1:
            raise ValueError("Not exactly one provider commit")
        self.position = accepted_updates

    def state_dict(self):
        return dict(position=self.position, refresh_events=list(self.refresh_events), inventory="synthetic-v1")

    def load_state_dict(self, state):
        if state.get("inventory") != "synthetic-v1":
            raise ValueError("Provider binding changed")
        self.position = state["position"]
        self.refresh_events = list(state["refresh_events"])


def fixture(schedule=None):
    model = build_model("F0-V0", seed=17)
    provider = FabricatedProvider()
    schedule = schedule or OwnerSchedule("synthetic_cpu_only", (2, 4, 6, 8), 2, 2, 32, 2, 2, 4, 2)
    protocol = ObjectiveProtocol("fabricated-base-F0", "F0", ("D3", "D4"),
        1., 1., 0., 0., 0., 1., 1., 1., 1., 1.,
        .001, .001, .001, .001, 0., 0., (.1, 1.))
    execution = ObjectiveExecution(1, 1, 4, 4, 4, 1000, 3e-6, 3e-6)
    geometry = RaggedSinkhornGeometry(options=GeometryOptions(1024, 2, "0.2.6"))
    cache = FitnessCache(torch.zeros(4), seen=torch.zeros(4, dtype=torch.bool), age=torch.zeros(4, dtype=torch.int64),
        binding=dict(model_recipe="F0-V0", training_subset="D3-D4", catalog_layout="2" * 64, source_sampler="3" * 64))
    blocks = tuple(CatalogBlock(donor, "catalog-three", torch.tensor([.5, .3, .2]).log(),
        torch.tensor([3, 9, 2]) if donor == "D3" else torch.tensor([8, 2, 4]),
        torch.tensor([0, 1]), torch.tensor([offset, offset + 1])) for donor, offset in (("D3", 0), ("D4", 2)))
    binding = owner_binding(model=model, protocol=protocol, execution=execution, geometry=geometry,
        inventory=provider.inventory, schedule=schedule)
    identity = dict(trajectory_id="fabricated-F0", fold_id="D3D4", seed=17, input_sha256="1" * 64,
        configuration_sha256=binding["configuration_sha256"], source_sha256="5" * 64, training_owner=binding)
    kwargs = dict(model=model, provider=provider, protocol=protocol, execution=execution,
        geometry=geometry, blocks=blocks, cache=cache, schedule=schedule)
    return identity, kwargs


class TrainingOwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        print(f"TRAINING_OWNER_DISPOSABLE_CPU_OPTIMIZER_CALLS={OPTIMIZER_CALLS}; SCIENTIFIC_UPDATES=0")

    def test_actual_F0_phase_promotion_resume_cache_and_terminal_failure(self):
        with tempfile.TemporaryDirectory(prefix="credo-owner-synthetic-") as directory, patch.object(torch.optim.Adam, "step", counted_step):
            root = Path(directory)
            seed()
            identity, args = fixture()
            with TrajectoryLease(root / "continuous", identity, "a") as lease:
                full = TrainingOwner(lease=lease, **args)
                state_optimizer = full.optimizer
                full.run_stage(2)
                self.assertEqual(full.model.phase, "state")
                self.assertEqual(full.refresh_updates, [])
                self.assertEqual(full.joint_entries, 0)
                self.assertTrue(all(int(state["step"]) == 2 for state in state_optimizer.state.values()))
                full.run_stage(4)
                self.assertIsNot(full.optimizer, state_optimizer)
                joint_optimizer = full.optimizer
                self.assertEqual(full.refresh_updates, [2])
                self.assertTrue(all(int(state["step"]) == 2 for state in joint_optimizer.state.values()))
                full.run_stage(6)
                self.assertIs(full.optimizer, joint_optimizer)
                self.assertEqual(full.joint_entries, 1)
                self.assertEqual(full.refresh_updates, [2, 4])
                self.assertTrue(all(int(state["step"]) == 4 for state in joint_optimizer.state.values()))
                self.assertEqual(full.cache.last_full_refresh_update, 4)
                self.assertEqual(full.cache.refresh_particles, 32)
                self.assertEqual(full.cache.age.tolist(), [0, 2, 0, 2])
                expected = copy.deepcopy((full.model.state_dict(), full.optimizer.state_dict(), full.cache.state_dict(), full.provider.state_dict()))
                expected_metrics = full.last_metrics
                histories = [json.loads(row) for row in full.history.read_text().splitlines()]
                self.assertEqual([row["update"] for row in histories], list(range(1, 7)))
                self.assertTrue(all(row["count_ce"] is None for row in histories[:2]))
                self.assertTrue(all(row["count_ce"] is not None for row in histories[2:]))
                expected_draws = (random.random(), np.random.random(), torch.randn(3))
            seed()
            identity, args = fixture()
            with TrajectoryLease(root / "resumed", identity, "b") as lease:
                partial = TrainingOwner(lease=lease, **args)
                partial.run_stage(2)
                record = partial.last_checkpoint
            for stop in (4, 6):
                random.random()
                np.random.normal(size=3)
                torch.randn(3)
                identity, args = fixture()
                with TrajectoryLease(root / "resumed", identity, f"worker-{stop}") as lease:
                    resumed = TrainingOwner(lease=lease, resume=record, **args)
                    before = resumed.optimizer
                    resumed.run_stage(stop)
                    if stop == 4:
                        self.assertIsNot(resumed.optimizer, before)
                    else:
                        self.assertIs(resumed.optimizer, before)
                    record = resumed.last_checkpoint
                    self.assertEqual(resumed.accepted_updates, resumed.attempted_updates)
                    self.assertEqual(resumed.optimizer_calls, stop)
                    self.assertEqual(resumed.joint_entries, 1)
                    if stop == 6:
                        actual = (resumed.model.state_dict(), resumed.optimizer.state_dict(), resumed.cache.state_dict(), resumed.provider.state_dict())
                        assert_tree(self, expected, actual)
                        self.assertEqual(resumed.last_metrics, expected_metrics)
                        assert_tree(self, expected_draws, (random.random(), np.random.random(), torch.randn(3)))
                        # Failure AFTER a completed checkpoint but BEFORE another
                        # optimizer call remains terminal, including on reopening.
                        resumed.provider.fail_at = 6
                        with self.assertRaisesRegex(RuntimeError, "Deliberate provider failure"):
                            resumed.run_stage(8)
                        self.assertEqual(resumed.status, "failed")
                        self.assertEqual((resumed.accepted_updates, resumed.attempted_updates, resumed.optimizer_calls), (6, 7, 6))
                        with self.assertRaisesRegex(RuntimeError, "terminal"):
                            resumed.run_stage(8)
            identity, args = fixture()
            with TrajectoryLease(root / "resumed", identity, "no-retry") as lease:
                with self.assertRaisesRegex(ValueError, "cannot automatically resume"):
                    TrainingOwner(lease=lease, resume=record, **args)
        self.assertEqual(OPTIMIZER_CALLS, 12)

    def test_scientific_schedule_cadence_and_cpu_rejection_without_steps(self):
        schedule = OwnerSchedule()
        self.assertEqual([u for u in range(1024) if schedule.refresh_before(u)], list(range(256, 1024, 64)))
        self.assertEqual(sum(schedule.refresh_before(u) for u in range(9216)), 140)
        self.assertEqual(schedule.phase_after(256), "state")
        self.assertEqual(schedule.phase_after(257), "joint")
        with self.assertRaisesRegex(ValueError, "fixed"):
            replace(schedule, state_updates=2)
        with self.assertRaisesRegex(ValueError, "scope"):
            OwnerSchedule("synthetic_cpu_only", (2, 9), 2, 2, 32, 2, 2, 4, 2)
        with tempfile.TemporaryDirectory(prefix="credo-owner-reject-") as directory:
            identity, args = fixture(schedule)
            with TrajectoryLease(Path(directory) / "trajectory", identity, "cpu-reject") as lease:
                with self.assertRaisesRegex(ValueError, "CUDA-only"):
                    TrainingOwner(lease=lease, **args)

    def test_binding_and_no_skipped_stage_without_steps(self):
        with tempfile.TemporaryDirectory(prefix="credo-owner-binding-") as directory:
            identity, args = fixture()
            with TrajectoryLease(Path(directory) / "trajectory", identity, "binding") as lease:
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    TrainingOwner(lease=lease, **{**args, "protocol": replace(args["protocol"], count_scale=2.)})
                owner = TrainingOwner(lease=lease, **args)
                with self.assertRaisesRegex(ValueError, "next registered stage"):
                    owner.run_stage(4)
                self.assertEqual(owner.optimizer_calls, 0)
                self.assertEqual(owner.status, "ready")

    def test_actual_decoder_digest_and_mutation_guard_without_steps(self):
        identity, args = fixture()
        decoder = SparseR48(ObservationFold("synthetic-D3D4", ("D3", "D4"), "7" * 64),
            features=9, hidden_dims=(8, 4), maximum_rows=4, seed=41)
        with torch.no_grad():
            decoder.center.zero_()
            decoder.scale.fill_(1.)
            decoder.scaling_ready.fill_(True)
        decoder.freeze()
        args["protocol"] = replace(args["protocol"], endpoint_coefficient=.1)
        args["decoder"] = decoder
        args["observation_binding"] = ObservationBinding("4" * 64, observation_state_digest(decoder))
        binding_arguments = {name: args[name] for name in ("model", "protocol", "execution", "geometry", "schedule", "decoder", "observation_binding")}
        binding_arguments["inventory"] = args["provider"].inventory
        with self.assertRaisesRegex(ValueError, "differs from the observation"):
            owner_binding(**{**binding_arguments, "observation_binding": ObservationBinding("4" * 64, "f" * 64)})
        identity["training_owner"] = owner_binding(**binding_arguments)
        with tempfile.TemporaryDirectory(prefix="credo-owner-decoder-") as directory:
            with TrajectoryLease(Path(directory) / "trajectory", identity, "decoder-binding") as lease:
                owner = TrainingOwner(lease=lease, **args)
                with torch.no_grad():
                    decoder.center.add_(.1)
                self.assertNotEqual(observation_state_digest(decoder), args["observation_binding"].state_sha256)
                with self.assertRaisesRegex(ValueError, "observation weights/scaler"):
                    owner.run_stage(2)
                self.assertEqual(owner.optimizer_calls, 0)
                self.assertEqual(owner.attempted_updates, 0)


if __name__ == "__main__":
    unittest.main()
