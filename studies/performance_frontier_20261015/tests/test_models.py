"""CPU fabricated-tensor tests; zero optimizer calls and zero scientific inputs."""

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import torch

from frontier.configuration import configurations, get_configuration
from frontier.models import FrontierModel, build_model, time_features, mixture_component_counts


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self, laws=3, particles=8):
        generator = torch.Generator().manual_seed(421)
        source = torch.randn((laws, particles, 48), generator=generator)
        noise = torch.randn((8, laws, particles, 48), generator=generator)
        return source, noise

    def make(self, identifier, seed=3):
        kwargs = dict(target_ids=("T1", "T2"), support_factors=[1., .25]) if identifier.startswith(("F3", "F5")) else {}
        return build_model(identifier, seed=seed, **kwargs)

    def test_construction_is_cpu_fp32_under_float64_and_meta_defaults(self):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device("meta"):
                for family in ("F0", "F1", "F2", "F3", "F4", "F5"):
                    model = self.make(family + "-V0")
                    for value in (*model.parameters(), *model.buffers()):
                        self.assertEqual(value.device.type, "cpu")
                        self.assertEqual(value.dtype, torch.float32)
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_all_registered_models_forward_finite_bound_and_self_describe(self):
        source, noise = self.fixture()
        for config in configurations():
            with self.subTest(configuration=config.configuration_id):
                model = self.make(config.configuration_id).set_phase("joint")
                result = model(source, noise, target_ids=("T1", "NTC", "UNSEEN"))
                self.assertEqual(result.states.shape, source.shape)
                self.assertTrue(bool(torch.isfinite(result.log_weights).all()))
                self.assertTrue(bool(torch.isfinite(result.states).all()))
                self.assertTrue(bool((result.reaction_log_increment.abs() <= 3.000001).all()))
                self.assertTrue(bool((result.log_mass_increment.abs() <= 3.000001).all()))
                torch.testing.assert_close(result.weights.sum(-1), torch.ones(3))
                info = model.architecture()
                self.assertEqual(info["configuration"]["configuration_id"], config.configuration_id)
                self.assertFalse(info["ecological_context"])
                self.assertEqual(info["parameter_count"], sum(p.numel() for p in model.parameters()))

    def test_shared_identity_invariance_and_no_private_parameters(self):
        source, noise = self.fixture()
        for family in ("F0", "F1", "F2", "F4"):
            model = self.make(family + "-V0").set_phase("joint")
            first = model(source, noise, target_ids=("A", "B", "NTC"))
            second = model(source, noise, target_ids=("OTHER", "NTC", "UNSEEN"))
            self.assertTrue(torch.equal(first.states, second.states))
            self.assertTrue(torch.equal(first.log_weights, second.log_weights))
            self.assertFalse(any("target_" in key or "support_factors" in key for key in model.state_dict()))

    def test_target_controls_unseen_and_zero_support_are_exact_zero(self):
        for family in ("F3", "F5"):
            model = build_model(family + "-V0", seed=3, target_ids=("T1", "T2", "ZERO"), support_factors=[1., .25, 0.])
            with torch.no_grad():
                model.target_codes.fill_(1.)
                model.target_projection.fill_(.5)
                model.target_growth_intercepts.fill_(2.)
                model.reference_embedding.fill_(.125)
            indices = model.target_indices(("T1", "T2", "ZERO", "NTC", "UNSEEN"))
            effective, intercepts = model.conditioning(indices)
            torch.testing.assert_close(effective[0], torch.full((8,), 2.125))
            torch.testing.assert_close(effective[1], torch.full((8,), .625))
            self.assertTrue(torch.equal(effective[2:], model.reference_embedding[None].expand(3, -1)))
            self.assertTrue(torch.equal(intercepts, torch.tensor([2., .5, 0., 0., 0.])))
            self.assertFalse(model.support_factors.requires_grad)
            self.assertEqual(model.target_codes.shape[1], 4)

    def test_target_fresh_initialization_and_shared_common_tensors(self):
        for shared_family, target_family in (("F0", "F3"), ("F4", "F5")):
            shared = self.make(shared_family + "-V0", seed=7)
            target = self.make(target_family + "-V0", seed=7)
            for key, value in shared.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
            self.assertEqual(torch.count_nonzero(target.target_codes).item(), 0)
            self.assertEqual(torch.count_nonzero(target.target_growth_intercepts).item(), 0)
            self.assertGreater(target.target_projection.abs().sum().item(), 0)

    def test_state_phase_exact_unit_mass_and_reaction_frozen(self):
        source, noise = self.fixture()
        for family in ("F0", "F1", "F2", "F3", "F4", "F5"):
            model = self.make(family + "-V0")
            if family == "F2":
                with torch.no_grad():
                    model.mixture_logits.copy_(torch.tensor([-4., -1., 2., 3.]))
            result = model(source, noise, target_ids=("T1", "NTC", "UNSEEN"))
            self.assertTrue(torch.equal(result.mass_multiplier, torch.ones(3)))
            self.assertEqual(torch.count_nonzero(result.reaction_log_increment).item(), 0)
            torch.testing.assert_close(torch.exp(result.log_weights).sum(-1), torch.ones(3))
            for name, parameter in model.named_parameters():
                if any(part in name for part in ("weight_reference", "weight_modulation", "growth_reference", "growth_residual", "target_growth_intercepts")):
                    self.assertFalse(parameter.requires_grad)
            model.set_phase("joint")
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_time_features_exact_retained_order(self):
        tau = torch.tensor(.375)
        expected = torch.stack([tau, *[value for k in range(1, 5) for value in
                                     (torch.sin(k * torch.pi * tau), torch.cos(k * torch.pi * tau))]])
        self.assertTrue(torch.equal(time_features(tau), expected))

    def test_ddr_rollout_matches_explicit_eight_step_equations(self):
        source, noise = self.fixture()
        model = self.make("F4-V0").set_phase("joint")
        embedding, intercept = model.conditioning(torch.full((3,), -1))
        state, rho = source, torch.zeros(source.shape[:2])
        for step in range(8):
            drift, sigma, reaction = model.ddr_primitive(state, source.new_tensor(step / 8), embedding, intercept)
            self.assertTrue(bool((sigma > .001).all()))
            state = state + drift / 8 + sigma * source.new_tensor(1 / 8).sqrt() * noise[step]
            rho = rho + reaction / 8
        result = model(source, noise)
        self.assertTrue(torch.equal(result.states, state))
        self.assertTrue(torch.equal(result.reaction_log_increment, rho))
        self.assertEqual(result.drift_steps.shape, noise.shape)
        self.assertEqual(result.sigma_steps.shape, noise.shape)
        self.assertEqual(result.reaction_steps.shape, (8, 3, 8))

    def test_f0_matches_copied_retained_direct_tensors(self):
        # Source-only import of the historical constructor. No checkpoint/data
        # load, optimization, or import of an execution owner is performed.
        study = Path(__file__).resolve().parents[2] / "credo_full_ddr_gse314342_20261006_v1"
        implementation = study / "comparator_implementation_v1"
        sys.path.insert(0, str(implementation))
        try:
            from comparator_models import DirectComparatorModel
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9)
                retained = DirectComparatorModel(("NTC", "T1"), ("NTC",), shared=True)
            retained.set_phase("mass")
            model = self.make("F0-V0").set_phase("joint")
            with torch.no_grad():
                retained.reference_embedding.copy_(torch.linspace(-.4, .5, 8))
                model.reference_embedding.copy_(retained.reference_embedding)
                for key, value in model.components[0].state_dict().items():
                    value.copy_(retained.state_dict()[key])
            source, noise = self.fixture()
            epsilon = noise.sum(0) / math.sqrt(8)
            effective = retained.reference_embedding[None].expand(3, -1)
            shift, rho = retained(source, epsilon, effective, torch.zeros(3))
            result = model(source, noise)
            self.assertTrue(torch.equal(result.states, source + shift))
            self.assertTrue(torch.equal(result.reaction_log_increment, rho))
            torch.testing.assert_close(result.log_mass_increment, torch.logsumexp(rho - math.log(8), -1))
        finally:
            sys.path.remove(str(implementation))

    def test_ddr_fields_match_copied_retained_tensors(self):
        study = Path(__file__).resolve().parents[2] / "credo_full_ddr_gse314342_20261006_v1"
        implementation = study / "comparator_implementation_v1"
        sys.path.insert(0, str(implementation))
        try:
            from comparator_models import DDRComparatorModel
            from gpu_native import TensorDynamics
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(9)
                retained = DDRComparatorModel(("NTC", "T1"), ("NTC",), shared=True)
            retained.set_phase("mass")
            model = self.make("F4-V0").set_phase("joint")
            with torch.no_grad():
                retained.reference_embedding.copy_(torch.linspace(-.4, .5, 8))
                model.reference_embedding.copy_(retained.reference_embedding)
                for key, value in model.fields.state_dict().items():
                    value.copy_(retained.state_dict()[key])
            source, _ = self.fixture()
            effective = retained.reference_embedding[None].expand(3, -1)
            intercept = torch.zeros(3)
            for value in (0., .125, .5, .875):
                tau = source.new_tensor(value)
                expected = TensorDynamics(retained)(source, tau, effective, intercept)
                actual = model.ddr_primitive(source, tau, effective, intercept)
                for left, right in zip(expected, actual):
                    self.assertTrue(torch.equal(left, right))
        finally:
            sys.path.remove(str(implementation))

    def test_mixture_keeps_components_not_coordinate_averages_and_gate_gradients(self):
        model = self.make("F2-V0")
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([-1., 0., 1., 2.]))
            for component, mapping in enumerate(model.components):
                for parameter in mapping.parameters():
                    parameter.zero_()
                mapping.displacement_reference.bias.fill_(component + 1.)
        source, noise = self.fixture(laws=1, particles=8)
        result = model(torch.zeros_like(source), noise)
        self.assertTrue(torch.equal(result.states[0, :, 0], torch.tensor([1., 2., 3., 4., 1., 2., 3., 4.])))
        pi = torch.softmax(model.mixture_logits, 0)
        for component in range(4):
            torch.testing.assert_close(result.weights[:, result.component_ids == component].sum(-1), pi[component:component + 1])
        mean = (result.weights * result.states[:, :, 0]).sum()
        mean.backward()
        self.assertGreater(model.mixture_logits.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.mixture_logits.grad).all())

    def test_mixture_constant_reaction_and_bounded_mass(self):
        model = self.make("F2-V0").set_phase("joint")
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([-6., -1., 2., 4.]))
            for mapping in model.components:
                mapping.weight_reference.weight.zero_()
                mapping.weight_reference.bias.fill_(math.atanh(.7 / 3))
                mapping.weight_modulation.weight.zero_()
                mapping.weight_modulation.bias.zero_()
        source, noise = self.fixture()
        result = model(source, noise)
        torch.testing.assert_close(result.log_mass_increment, torch.full((3,), .7))

    def test_mixture_nondivisible_particles_preserve_component_mass(self):
        model = self.make("F2-V0")
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([-3., -.5, 1., 2.]))
        for particles in (4, 5, 7):
            source, noise = self.fixture(laws=1, particles=particles)
            result = model(source, noise)
            pi = torch.softmax(model.mixture_logits, 0)
            counts = mixture_component_counts(particles)
            self.assertEqual(tuple(torch.bincount(result.component_ids, minlength=4).tolist()), counts)
            self.assertTrue(torch.equal(result.mass_multiplier, torch.ones(1)))
            torch.testing.assert_close(result.log_weights.exp().sum(-1), torch.ones(1))
            for component, count in enumerate(counts):
                selected = result.component_ids == component
                torch.testing.assert_close(result.weights[:, selected].sum(-1), pi[component:component + 1])
                expected = math.log(particles) + torch.log(pi[component]) - math.log(count)
                torch.testing.assert_close(result.mixture_log_correction[:, selected], expected.expand(1, count))

    def test_prefix_reuses_states_recomputes_mixture_mass_and_original_gate(self):
        source, noise = self.fixture(laws=2, particles=12)
        model = self.make("F2-V0").set_phase("joint")
        with torch.no_grad():
            model.mixture_logits.copy_(torch.tensor([-2., -.5, 1., 2.]))
        result = model(source, noise)
        old_log_pi = result.mixture_log_probabilities.clone()
        # Prefix publication cannot accidentally read a changed live gate/phase.
        with torch.no_grad():
            model.mixture_logits.add_(torch.tensor([3., 2., 1., -2.]))
        model.set_phase("state")
        before = torch.get_rng_state().clone()
        for particles in (4, 5, 7, 12):
            prefix = model.prefix_rollout(result, particles)
            self.assertTrue(torch.equal(prefix.states, result.states[:, :particles]))
            self.assertTrue(torch.equal(prefix.anchor_displacement, result.anchor_displacement[:, :particles]))
            self.assertTrue(prefix.reaction_enabled)
            counts = mixture_component_counts(particles)
            raw_weight = torch.empty_like(prefix.log_weights)
            for component, count in enumerate(counts):
                selected = prefix.component_ids == component
                raw_weight[:, selected] = torch.exp(old_log_pi[component] + result.reaction_log_increment[:, :particles][:, selected]) / count
            torch.testing.assert_close(prefix.log_mass_increment, raw_weight.sum(-1).log())
            torch.testing.assert_close(prefix.weights, raw_weight / raw_weight.sum(-1, keepdim=True))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        prefix = model.prefix_rollout(result, 7)
        truncated_and_renormalized = result.weights[:, :7] / result.weights[:, :7].sum(-1, keepdim=True)
        self.assertFalse(torch.allclose(prefix.weights, truncated_and_renormalized))

    def test_ordinary_prefix_recomputes_uniform_base_and_slices_ddr_actions(self):
        source, noise = self.fixture(laws=2, particles=8)
        for family in ("F0", "F4"):
            model = self.make(family + "-V0").set_phase("joint")
            result = model(source, noise)
            for particles in (1, 4, 5, 7):
                prefix = model.prefix_rollout(result, particles)
                self.assertTrue(torch.equal(prefix.states, result.states[:, :particles]))
                expected_log_weights = result.reaction_log_increment[:, :particles] - math.log(particles)
                self.assertTrue(torch.equal(prefix.log_weights, expected_log_weights))
                torch.testing.assert_close(prefix.log_mass_increment, torch.logsumexp(expected_log_weights, -1))
                if family == "F4":
                    self.assertTrue(torch.equal(prefix.drift_steps, result.drift_steps[:, :, :particles]))
                    self.assertTrue(torch.equal(prefix.sigma_steps, result.sigma_steps[:, :, :particles]))
                    self.assertTrue(torch.equal(prefix.reaction_steps, result.reaction_steps[:, :, :particles]))

    def test_ensemble_primary_refinement_component_allocation_metadata(self):
        # Metadata arithmetic only: no fixture exceeds the 128-particle scope.
        allocations = {
            4096: (1366, 1365, 1365),
            8192: (2731, 2731, 2730),
        }
        expected = {
            1366: (342, 342, 341, 341),
            1365: (342, 341, 341, 341),
            2731: (683, 683, 683, 682),
            2730: (683, 683, 682, 682),
        }
        for total, seed_particles in allocations.items():
            self.assertEqual(sum(seed_particles), total)
            for count in seed_particles:
                self.assertEqual(mixture_component_counts(count), expected[count])
                self.assertEqual(sum(mixture_component_counts(count)), count)
        self.assertEqual(mixture_component_counts(4096), (1024, 1024, 1024, 1024))
        self.assertEqual(mixture_component_counts(8192), (2048, 2048, 2048, 2048))

    def test_prefix_validation_and_state_phase_unit_mass(self):
        source, noise = self.fixture(laws=1, particles=8)
        model = self.make("F2-V0")
        result = model(source, noise)
        for particles in (4, 5, 7):
            prefix = model.prefix_rollout(result, particles)
            self.assertTrue(torch.equal(prefix.mass_multiplier, torch.ones(1)))
            torch.testing.assert_close(prefix.log_weights.exp().sum(-1), torch.ones(1))
        for particles in (0, 3, 9, True, 5.0):
            with self.assertRaises(ValueError):
                model.prefix_rollout(result, particles)
        with self.assertRaises(ValueError):
            self.make("F1-V0").prefix_rollout(result, 4)
        with self.assertRaises(ValueError):
            model.prefix_rollout(replace(result, component_ids=torch.ones(8, dtype=torch.long)), 4)
        with self.assertRaises(ValueError):
            model.prefix_rollout(replace(result, mixture_log_probabilities=None), 4)

    def test_direct_anchor_no_new_rng_and_no_anchor_forecast_equivalence(self):
        model = self.make("F2-V0").set_phase("joint")
        source, noise = self.fixture()
        rng = torch.get_rng_state().clone()
        with_anchor = model(source, noise, compute_anchor=True)
        without_anchor = model(source, noise, compute_anchor=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(with_anchor.states, without_anchor.states))
        self.assertTrue(torch.equal(with_anchor.log_weights, without_anchor.log_weights))
        self.assertIsNotNone(with_anchor.anchor_displacement)
        self.assertIsNone(without_anchor.anchor_displacement)

    def test_joint_gradients_reach_state_reaction_and_target_blocks(self):
        source, noise = self.fixture()
        for family in ("F0", "F2", "F3", "F4", "F5"):
            model = self.make(family + "-V0").set_phase("joint")
            result = model(source, noise, target_ids=("T1", "T2", "NTC"))
            objective = (result.states.square().mean() + result.log_mass_increment.square().mean()
                         + (result.weights * result.states[:, :, 0]).sum())
            objective.backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None), 0)
            self.assertTrue(all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None))
            if family in ("F3", "F5"):
                self.assertGreater(model.target_codes.grad.abs().sum().item(), 0)
                self.assertGreater(model.target_growth_intercepts.grad.abs().sum().item(), 0)

    def test_invalid_configuration_support_metadata_and_noise_rejected(self):
        with self.assertRaises(ValueError):
            FrontierModel(replace(get_configuration("F0-V0"), width=7))
        for factors in ([1.], [1., float("nan")], [1., -1.], [1., 2.]):
            with self.assertRaises(ValueError):
                build_model("F3-V0", seed=1, target_ids=("T1", "T2"), support_factors=factors)
        with self.assertRaises(ValueError):
            build_model("F0-V0", seed=1, support_factors=[1.])
        with self.assertRaises(ValueError):
            build_model("F3-V0", seed=1, target_ids=("NTC",), support_factors=[1.])
        model = self.make("F2-V0")
        source, noise = self.fixture(particles=3)
        with self.assertRaises(ValueError):
            model(source, noise)
        source, noise = self.fixture()
        with self.assertRaises(ValueError):
            model(source, noise[:7])
        with self.assertRaises(ValueError):
            model(source.double(), noise.double())
        with self.assertRaises(ValueError):
            model(source, noise, target_ids=("A",))
        with self.assertRaises(ValueError):
            model.set_phase("mass")

    def test_fresh_build_is_reproducible_and_preserves_global_rng(self):
        before = torch.get_rng_state().clone()
        one, two = self.make("F3-V0", seed=11), self.make("F3-V0", seed=11)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for key, value in one.state_dict().items():
            self.assertTrue(torch.equal(value, two.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
