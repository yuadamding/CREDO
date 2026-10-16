"""Fabricated CPU arrays/metadata only. No encoder, model or optimizer calls."""

from dataclasses import asdict, replace
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from frontier.fitting_provider import (CatalogGuide, FoldCacheManifest, FoldFittingCache,
    FoldFittingProvider, PopulationArrays, PopulationReference, ProviderLimits, _hash)
from frontier.native_inputs import FoldInputs
from frontier.objective_registry import ObjectiveRules, RULES_SHA256
from frontier.sampling import PooledSourceIdentity
from frontier.training_owner import ObservationBinding, OwnerSchedule


def readonly(value, dtype=None):
    value = np.asarray(value, dtype=dtype).copy(order="C")
    value.setflags(write=False)
    return value


def fixture(*, source_access="a" * 64, limits=None):
    catalog = tuple(CatalogGuide(g, t, c) for g, t, c in (
        ("a0", "A", False), ("a1", "A", False), ("b", "B", False),
        ("ntc0", "NTC", True), ("noSource", "A", False),
        ("countOnly", "C", False), ("ntc1", "NTC", True)))
    catalog_sha = _hash([asdict(g) for g in catalog])
    observation = ObservationBinding("1" * 64, "2" * 64)
    feature_sha = "3" * 64
    populations = []
    for donor_index, donor in enumerate(("D3", "D4")):
        for time_index, condition in enumerate(("Stim8hr", "Stim48hr")):
            counts = [3, 2, 4, 3, 0, 2, 2] if time_index == 0 else [3, 4, 2, 3, 5, 0, 6]
            positive = readonly(counts, np.int64)
            captured = readonly(positive + np.array([1, 2, 0, 1, 2, 3, 0]), np.int64)
            offsets = readonly(np.r_[0, np.cumsum(positive)], np.int64)
            states = readonly(np.arange(int(positive.sum()) * 48).reshape(-1, 48) * .001
                              + donor_index + time_index * .1, np.float32)
            means = np.zeros((len(catalog), 5), np.float64)
            for guide in range(len(catalog)):
                if positive[guide]:
                    row = np.arange(1, 6, dtype=np.float64) + guide + donor_index + time_index
                    means[guide] = row / row.sum()
            means = readonly(means)
            access = source_access if time_index == 0 else "b" * 64
            reference = PopulationReference(donor, condition, donor + "_" + condition,
                "captured-donor-aggregate-culture-unresolved", access, "c" * 64,
                _hash([donor, condition, "row-order"]), "0" * 64, "0" * 64)
            p = PopulationArrays(reference, captured, positive, offsets, states, means)
            reference = replace(reference,
                raw_summary_sha256=p.raw_identity(catalog_sha256=catalog_sha, feature_axis_sha256=feature_sha),
                latent_sha256=p.latent_identity(catalog_sha256=catalog_sha, observation_binding=observation))
            populations.append(replace(p, reference=reference))
    manifest = FoldCacheManifest(FoldInputs("fixture-D3-D4", ("D3", "D4")), catalog, feature_sha, 5,
        observation, source_access, "b" * 64, tuple(p.reference for p in populations), synthetic_fixture=True)
    limits = limits or ProviderLimits(32 * 1024**2, 32 * 1024**2, 100)
    cache = FoldFittingCache(manifest, populations, expected_manifest_sha256=manifest.identity, limits=limits)
    return cache, tuple(populations)


class FittingProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (Path(__file__).resolve().parents[1] / "MAIN_OBJECTIVE_RULES.json").read_bytes()
        cls.rules = ObjectiveRules(raw, RULES_SHA256)

    def provider(self, *, family="F3", batch=4, cache=None, seed=0, **kwargs):
        cache = fixture()[0] if cache is None else cache
        schedule = OwnerSchedule(execution_scope="synthetic_cpu_only", stages=(2, 4), state_updates=1,
            cache_period=2, checkpoint_period=2, logical_batch_laws=batch,
            training_particles=8, maximum_refresh_laws=4)
        return FoldFittingProvider(cache, rules=self.rules, configuration_id=family + "-V0", model_seed=seed,
            study_id="fabricated-provider", study_seed=731, schedule=schedule, **kwargs)

    def joint(self, provider):
        provider.sample(accepted_updates=0, phase="state")
        # Simulate owner's commit protocol only; no optimizer is invoked.
        provider.accepted(accepted_updates=1)
        return provider.sample(accepted_updates=1, phase="joint")

    def test_complete_catalog_retains_zero_source_state_complement_and_count_only_law(self):
        provider = self.provider()
        blocks, cache = provider.count_inputs()
        self.assertEqual(len(provider.inventory.law_ids), 12)
        self.assertEqual(len(provider.cache.paired), 10)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(cache.values), 12)
        self.assertFalse(bool(cache.seen.any()))
        routed = []
        for block in blocks:
            block.validate(cache)
            self.assertEqual(len(block.log_source), 7)
            self.assertEqual(block.modeled_indices.tolist(), [0, 1, 2, 3, 5, 6])
            self.assertEqual(block.endpoint_counts[5].item(), 3)  # zero RNA, not zero capture
            expected = (provider.cache.populations[(block.donor_id, "Stim8hr")].captured_counts.astype(float) + .5)
            expected /= expected.sum()
            np.testing.assert_allclose(block.log_source.exp().numpy(), expected, atol=1e-7)
            routed.extend(block.cache_indices.tolist())
        self.assertEqual(sorted(routed), list(range(12)))

    def test_state_sampling_paired_uniform_distinct_and_full_endpoint_support(self):
        provider = self.provider()
        result = provider.sample(accepted_updates=0, phase="state")
        indices = result.inputs.cache_indices.tolist()
        self.assertEqual(len(set(indices)), 4)
        self.assertTrue(set(indices) <= provider.cache.paired_set)
        self.assertIsNone(result.molecular)
        self.assertTrue(torch.equal(result.inputs.geometry_weights, torch.ones(4)))
        self.assertTrue(torch.equal(result.inputs.action_weights, torch.full((4,), .25)))
        for index, law in zip(indices, result.inputs.observations):
            donor, guide = provider.cache.laws[index]
            self.assertEqual(len(law.support), provider.cache.populations[(donor, "Stim48hr")].positive_counts[guide])
            self.assertAlmostEqual(law.weights.sum().item(), 1., places=6)

    def test_pooled_ntc_is_typed_all_cells_and_not_a_catalog_or_cache_law(self):
        provider = self.provider()
        for donor, law, truth in zip(provider.inventory.training_donors, provider.cache.pool_laws, provider.cache.pool_truth):
            self.assertIsInstance(law.identity, PooledSourceIdentity)
            self.assertEqual(len(law.states), 5)
            self.assertNotIn(law.identity.pool_id, provider.inventory.law_ids)
            endpoint = provider.cache.populations[(donor, "Stim48hr")]
            np.testing.assert_allclose(truth, (3 * endpoint.mean_compositions[3] + 6 * endpoint.mean_compositions[6]) / 9)
        self.assertEqual(len(provider.cache.manifest.catalog), 7)
        result = self.joint(provider)
        controls = result.molecular.controls
        self.assertEqual(controls.component_donor_indices.tolist(), [0, 1])
        self.assertEqual(controls.reduction_weights.tolist(), [.5, .5])
        self.assertEqual(controls.mixture_weights.tolist(), [1., 1.])

    def test_global_macro_weights_and_horvitz_multiplier_not_batch_macro(self):
        provider = self.provider()
        global_weights = provider.cache.molecular_weights
        self.assertAlmostEqual(sum(global_weights.values()), 1.)
        for index, weight in global_weights.items():
            target = provider.inventory.target_ids[index]
            self.assertEqual(weight, .125 if target == "A" else .25)
        result = self.joint(provider)
        selected = result.inputs.cache_indices.tolist()
        rows = result.molecular.target_law_indices.tolist()
        total = sum(global_weights[selected[row]] for row in rows)
        self.assertAlmostEqual(result.molecular.target_estimator_multiplier, 3. * total)
        expected = [global_weights[selected[row]] / total for row in rows]
        np.testing.assert_allclose(result.molecular.target_weights.numpy(), expected, atol=0)
        # Per-row estimator coefficient equals N/B times fixed global weight.
        np.testing.assert_allclose(result.molecular.target_weights.numpy() * result.molecular.target_estimator_multiplier,
                                   [3 * global_weights[selected[row]] for row in rows])

    def test_common_draws_across_families_and_repeat_calls_do_not_advance(self):
        original = self.provider(family="F0")
        expected = original.sample(accepted_updates=0, phase="state")
        for family in ("F1", "F2", "F3", "F4", "F5"):
            candidate = self.provider(family=family)
            value = candidate.sample(accepted_updates=0, phase="state")
            self.assertTrue(torch.equal(value.inputs.cache_indices, expected.inputs.cache_indices))
            self.assertTrue(torch.equal(value.inputs.source, expected.inputs.source))
            self.assertTrue(torch.equal(value.inputs.noise, expected.inputs.noise))
        repeated = original.sample(accepted_updates=0, phase="state")
        self.assertTrue(torch.equal(repeated.inputs.noise, expected.inputs.noise))
        self.assertEqual(original.state_dict()["position"], 0)

    def test_cache_refresh_complete_and_rng_independent_of_refresh_partition(self):
        provider = self.provider(family="F0")
        self.joint(provider)
        batches = tuple(provider.cache_batches(accepted_updates=1, particles=32, maximum_laws=4))
        self.assertEqual(torch.cat([b.cache_indices for b in batches]).tolist(), list(range(12)))
        self.assertTrue(all(b.source.shape == (4, 32, 48) for b in batches))
        again = tuple(provider.cache_batches(accepted_updates=1, particles=32, maximum_laws=4))
        self.assertTrue(torch.equal(again[0].noise, batches[0].noise))
        self.assertEqual(provider.state_dict()["position"], 1)
        with self.assertRaises(ValueError):
            tuple(provider.cache_batches(accepted_updates=1, particles=16, maximum_laws=4))

    def test_committed_resume_is_exact_and_cross_seed_binding_rejected(self):
        provider = self.provider()
        expected = self.joint(provider)
        state = provider.state_dict()
        resumed = self.provider()
        resumed.load_state_dict(state)
        actual = resumed.sample(accepted_updates=1, phase="joint")
        self.assertTrue(torch.equal(actual.inputs.source, expected.inputs.source))
        self.assertTrue(torch.equal(actual.inputs.noise, expected.inputs.noise))
        self.assertTrue(torch.equal(actual.molecular.controls.noise, expected.molecular.controls.noise))
        with self.assertRaises(ValueError):
            self.provider(seed=1).load_state_dict(state)
        with self.assertRaises(ValueError):
            resumed.accepted(accepted_updates=3)
        resumed.accepted(accepted_updates=2)
        with self.assertRaises(ValueError):
            resumed.accepted(accepted_updates=2)

    def test_bound_context_changes_rejected_before_use_or_restoration(self):
        mutations = {
            "model_seed": lambda p: setattr(p, "model_seed", 1),
            "study_seed": lambda p: setattr(p, "study_seed", 732),
            "study_id": lambda p: setattr(p, "study_id", "different-study"),
            "configuration": lambda p: setattr(p, "configuration_id", "F0-V1"),
            "schedule": lambda p: setattr(p, "schedule", replace(p.schedule, logical_batch_laws=3)),
            "specification": lambda p: setattr(p, "specification", replace(p.specification, maximum_particles=4096)),
            "device": lambda p: setattr(p, "device", torch.device("cpu:0")),
            "pin_memory": lambda p: setattr(p, "pin_memory", True),
            "molecular": lambda p: setattr(p, "molecular_enabled", True),
            "inventory": lambda p: setattr(p, "inventory", replace(p.inventory, sampler_sha256="f" * 64)),
            "binding": lambda p: setattr(p, "binding", "f" * 64),
            "routing_hash": lambda p: setattr(p, "_ordered_laws_sha256", "f" * 64),
            "routing_array": lambda p: setattr(p, "_state_indices", p._state_indices[::-1].copy()),
            "cache": lambda p: setattr(p, "cache", fixture()[0]),
            "limits": lambda p: setattr(p.cache, "limits", replace(p.cache.limits, maximum_prepared_bytes=1000)),
        }
        for name, change in mutations.items():
            with self.subTest(mutation=name):
                provider = self.provider(family="F0")
                state = provider.state_dict()
                change(provider)
                with patch("frontier.fitting_provider.sample_batch", side_effect=AssertionError("Changed context sampled")):
                    for operation in (
                        provider.state_dict,
                        lambda: provider.sample(accepted_updates=0, phase="state"),
                        lambda: provider.load_state_dict(state),
                        provider.count_inputs,
                    ):
                        with self.assertRaisesRegex(ValueError, "context/routing/transfer"):
                            operation()
                self.assertEqual(provider.position, 0)

    def test_context_change_after_preparation_prevents_commit(self):
        provider = self.provider(family="F0")
        provider.sample(accepted_updates=0, phase="state")
        provider.study_seed += 1
        with self.assertRaisesRegex(ValueError, "context/routing/transfer"):
            provider.accepted(accepted_updates=1)
        self.assertEqual(provider.position, 0)

    def test_suspended_refresh_rechecks_context_before_each_yield(self):
        provider = self.provider(family="F0")
        self.joint(provider)
        batches = provider.cache_batches(accepted_updates=1, particles=32, maximum_laws=4)
        first = next(batches)
        self.assertEqual(first.cache_indices.tolist(), [0, 1, 2, 3])
        provider.model_seed = 1
        with patch("frontier.fitting_provider.sample_batch", side_effect=AssertionError("Changed context sampled")):
            with self.assertRaisesRegex(ValueError, "context/routing/transfer"):
                next(batches)
        self.assertEqual(provider.position, 1)

    def test_routing_arrays_are_immutable_not_merely_flagged_readonly(self):
        provider = self.provider(family="F0")
        for array in (provider._state_indices, provider._all_indices):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array[0] = array[-1]
        # Read-only routes still permit all legitimate position/resume paths.
        expected = self.joint(provider)
        restored = self.provider(family="F0")
        restored.load_state_dict(provider.state_dict())
        actual = restored.sample(accepted_updates=1, phase="joint")
        self.assertTrue(torch.equal(actual.inputs.cache_indices, expected.inputs.cache_indices))
        self.assertTrue(torch.equal(actual.inputs.noise, expected.inputs.noise))

    def test_empty_required_objectives_fail_without_redraw_or_commit(self):
        provider = self.provider()
        self.joint(provider)
        count_only = tuple(i for i, (_, g) in enumerate(provider.cache.laws) if g == 5)
        with patch.object(provider, "selected_indices", return_value=count_only), self.assertRaisesRegex(ValueError, "no observed state"):
            provider.sample(accepted_updates=1, phase="joint")
        no_targets = tuple(i for i in range(len(provider.cache.laws)) if i not in provider.cache.molecular_weights)[:4]
        with patch.object(provider, "selected_indices", return_value=no_targets), self.assertRaisesRegex(ValueError, "no supported targeting"):
            provider.sample(accepted_updates=1, phase="joint")
        self.assertEqual(provider.position, 1)
        with self.assertRaises(ValueError):
            provider.accepted(accepted_updates=2)

    def test_raw_summary_reused_across_fold_access_but_latent_identity_bound(self):
        first, _ = fixture()
        second, _ = fixture(source_access="d" * 64)
        old = first.populations[("D3", "Stim8hr")].reference
        new = second.populations[("D3", "Stim8hr")].reference
        self.assertEqual(old.raw_summary_sha256, new.raw_summary_sha256)
        self.assertNotEqual(old.latent_sha256, new.latent_sha256)
        self.assertNotEqual(first.manifest.identity, second.manifest.identity)

    def test_array_hash_missing_zero_and_readonly_guards(self):
        cache, populations = fixture()
        p = populations[0]
        changed = p.states.copy(); changed[0, 0] += 1; changed.setflags(write=False)
        replacement = (replace(p, states=changed), *populations[1:])
        with self.assertRaisesRegex(ValueError, "content differs"):
            FoldFittingCache(cache.manifest, replacement, expected_manifest_sha256=cache.manifest.identity, limits=cache.limits)
        means = p.mean_compositions.copy(); means[4] = .2; means.setflags(write=False)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            FoldFittingCache(cache.manifest, (replace(p, mean_compositions=means), *populations[1:]),
                expected_manifest_sha256=cache.manifest.identity, limits=cache.limits)
        with self.assertRaisesRegex(ValueError, "Read-only"):
            FoldFittingCache(cache.manifest, (replace(p, states=p.states.copy()), *populations[1:]),
                expected_manifest_sha256=cache.manifest.identity, limits=cache.limits)
        p.states.setflags(write=True)
        with self.assertRaisesRegex(ValueError, "read-only"):
            cache.assert_unchanged()
        p.states.setflags(write=False)

    def test_role_manifest_and_resource_caps_fail_without_truncation(self):
        cache, populations = fixture()
        with self.assertRaises(ValueError):
            replace(cache.manifest, fold=FoldInputs("wrong", ("D2", "D3")))
        with self.assertRaises(ValueError):
            replace(cache.manifest, populations=tuple(reversed(cache.manifest.populations)))
        with self.assertRaisesRegex(ValueError, "host-byte cap"):
            fixture(limits=ProviderLimits(10, 10**6, 100))
        with self.assertRaisesRegex(ValueError, "NTC pool"):
            fixture(limits=ProviderLimits(10**6, 10**6, 4))
        tiny = fixture(limits=ProviderLimits(10**6, 100, 100))[0]
        provider = self.provider(cache=tiny)
        with self.assertRaisesRegex(ValueError, "prepared-input envelope"):
            provider.sample(accepted_updates=0, phase="state")
        with self.assertRaises(ValueError):
            provider.count_inputs()

    def test_no_global_rng_changes_or_numpy_write_permission_changes(self):
        provider = self.provider()
        python, numpy, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        provider.sample(accepted_updates=0, phase="state")
        self.assertEqual(random.getstate(), python)
        now = np.random.get_state()
        self.assertEqual(now[0], numpy[0]); self.assertTrue(np.array_equal(now[1], numpy[1])); self.assertEqual(now[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        provider.cache.assert_unchanged()


if __name__ == "__main__":
    unittest.main()
