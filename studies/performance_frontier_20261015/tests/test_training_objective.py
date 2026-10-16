"""Integrated fabricated-tensor objective proofs; zero optimizer steps."""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from frontier.count_objective import ActiveFitnessChunk, CatalogBlock, FitnessCache, cached_count_objective
from frontier.geometry import GeometryOptions, ObservedLaw, RaggedSinkhornGeometry
from frontier.measure_ops import multiscale_mmd2, source_log_frequencies
from frontier.models import build_model
from frontier.molecular import transform_composition
from frontier.molecular_adjoint import molecular_adjoint
from frontier.observation import ObservationFold, SparseR48
from frontier.training_objective import (LogicalTrainingInputs, MolecularTrainingInputs,
    ObjectiveExecution, ObjectiveProtocol, PooledControlInputs, accumulate_training_objective,
    molecular_working_element_bound)


class TrainingObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, family="F0", phase="joint"):
        generator = torch.Generator().manual_seed(913)
        kwargs = dict(target_ids=("T1", "T2", "T3"), support_factors=[1., .4, .2]) if family in ("F3", "F5") else {}
        model = build_model(family + "-V0", seed=13, **kwargs).set_phase(phase)
        with torch.no_grad():
            if family == "F2":
                model.mixture_logits.copy_(torch.tensor([-1., .5, 1., -.2]))
            if model.config.target_residual:
                model.target_codes.fill_(.02)
                model.target_growth_intercepts.copy_(torch.tensor([.02, -.01, .03]))
        source = .2 * torch.randn(4, 8, 48, generator=generator)
        noise = torch.randn(8, 4, 8, 48, generator=generator)
        observations = tuple(ObservedLaw(f"law-{i}", .25 * torch.randn(5 + i, 48, generator=generator) + .1,
                            torch.ones(5 + i)) if i != 3 else None for i in range(4))
        inputs = LogicalTrainingInputs(tuple(f"law-{i}" for i in range(4)), ("D3", "D3", "D4", "D4"),
            ("T1", "NTC", "T2", "T3"), source, noise, torch.tensor([0, 1, 3, 5]), observations,
            torch.tensor([1., 2., 3., 4.]), torch.tensor([.1, .2, .3, .4]))
        blocks = (
            CatalogBlock("D3", "six", source_log_frequencies(torch.tensor([2, 0, 7, 1, 3, 0])).float(),
                         torch.tensor([1, 10, 20, 0, 4, 2]), torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2])),
            CatalogBlock("D4", "six", source_log_frequencies(torch.tensor([1, 4, 0, 2, 0, 8])).float(),
                         torch.tensor([12, 1, 2, 30, 0, 7]), torch.tensor([1, 3, 5]), torch.tensor([3, 4, 5])))
        cache = FitnessCache(torch.tensor([.3, -.4, .1, .2, -.1, .5]),
            binding=dict(model_recipe="synthetic", training_subset="D3-D4", catalog_layout="six", source_sampler="explicit"),
            seen=torch.ones(6, dtype=torch.bool), age=torch.arange(6))
        fold = ObservationFold("synthetic-D3-D4", ("D3", "D4"), "1" * 64)
        decoder = SparseR48(fold, features=11, hidden_dims=(16, 8), maximum_rows=8, seed=14)
        with torch.no_grad():
            decoder.center.fill_(.1)
            decoder.scale.fill_(.8)
            decoder.scaling_ready.fill_(True)
        decoder.freeze()
        controls = PooledControlInputs(("D3", "D4"), ("c0", "c1", "c2", "c3"), torch.tensor([0, 0, 1, 1]),
            ("NTC",) * 4, .2 * torch.randn(4, 8, 48, generator=generator), torch.randn(8, 4, 8, 48, generator=generator),
            torch.tensor([.25, .75, .4, .6], dtype=torch.float64),
            torch.randn(2, 11, generator=generator, dtype=torch.float64).softmax(-1),
            torch.tensor([.5, .5], dtype=torch.float64))
        molecular = MolecularTrainingInputs(torch.tensor([0, 2]), torch.randn(2, 11, generator=generator, dtype=torch.float64).softmax(-1),
            torch.tensor([.3, .7], dtype=torch.float64), controls)
        # Synthetic coefficients deliberately exercise every primitive, not a
        # claim that this matrix has been chosen for the scientific F0 recipe.
        protocol = ObjectiveProtocol("synthetic-all-components", family, ("D3", "D4"),
            .5, .5, .1, .1, .1, 2., .4, .3, .7, .8,
            1e-4, 1e-5, 1e-4, 1e-4, 1e-3, 1e-3, (.1, .3, 1.))
        geometry = RaggedSinkhornGeometry(options=GeometryOptions(8192, 4, "0.2.6"))
        return model, inputs, protocol, geometry, blocks, cache, decoder, molecular

    def execution(self, chunk):
        return ObjectiveExecution(chunk, chunk, 4, 4, 4, 10000, 3e-6, 3e-6)

    def decode(self, decoder, rollout):
        return torch.stack([decoder.weighted_composition(rollout.states[i], rollout.weights[i],
            gene_block=4, row_block=4, recompute_backward=True) for i in range(len(rollout.states))])

    def reference(self, model, inputs, protocol, geometry, blocks, cache, decoder, molecular):
        result = model(inputs.source, inputs.noise, target_ids=inputs.target_ids)
        geo = geometry(result.states, torch.log_softmax(result.log_weights, -1), law_ids=inputs.law_ids, observations=inputs.observations)
        denominator = inputs.geometry_weights[geo.available_indices].sum()
        value = protocol.state_coefficient / protocol.state_scale * geo.weighted_sum(inputs.geometry_weights) / denominator
        if model.config.dynamics == "direct":
            base = result.mixture_log_correction.exp() / inputs.source.shape[1]
            det = (base * result.anchor_displacement.square().mean(-1)).sum(-1)
            sto = (base * (result.displacement - result.anchor_displacement).square().mean(-1)).sum(-1)
            reaction = (base * result.reaction_log_increment.square()).sum(-1)
        else:
            det = result.drift_steps.square().mean((0, 2, 3))
            sto = result.sigma_steps.square().mean((0, 2, 3))
            reaction = result.reaction_steps.square().mean((0, 2))
        value = value + (inputs.action_weights * (protocol.deterministic_action * det + protocol.stochastic_action * sto)).sum()
        value = value + protocol.reference_embedding_penalty * model.reference_embedding.square().mean()
        if model.config.target_residual:
            value = value + protocol.target_embedding_penalty * (model.target_codes @ model.target_projection).square().mean()
        if model.phase == "state":
            value.backward()
            return value.detach()
        value = value + protocol.reaction_action * (inputs.action_weights * reaction).sum()
        if model.config.target_residual:
            value = value + protocol.target_growth_penalty * model.target_growth_intercepts.square().mean()
        if protocol.count_coefficient:
            count = cached_count_objective(blocks, cache, [ActiveFitnessChunk(inputs.cache_indices, result.log_mass_increment)], sampling_scale=1.5)
            value = value + protocol.count_coefficient / protocol.count_scale * count.surrogate
        if protocol.shape_coefficient:
            shape = sum(inputs.geometry_weights[i] * multiscale_mmd2(result.states[i], result.weights[i], law.support,
                        law.weights / law.weights.sum(), protocol.shape_bandwidths, block=4)
                        for i, law in enumerate(inputs.observations) if law is not None)
            value = value + protocol.shape_coefficient / protocol.shape_scale * shape / denominator
        if protocol.endpoint_coefficient or protocol.effect_coefficient:
            target = self.decode(decoder, result)[molecular.target_law_indices]
            control = molecular.controls
            cresult = model(control.source, control.noise, target_ids=control.component_target_ids, compute_anchor=False)
            components = self.decode(decoder, cresult)
            pool = components.new_zeros(2, decoder.features).index_add(0, control.component_donor_indices,
                components * control.mixture_weights[:, None])
            et = transform_composition(target) - transform_composition(molecular.observed_targets)
            ec = transform_composition(pool) - transform_composition(control.observed_compositions)
            target_mse = (molecular.target_weights * et.square().mean(-1)).sum()
            control_mse = (control.reduction_weights * ec.square().mean(-1)).sum()
            effect = (molecular.target_weights * (et - ec).square().mean(-1)).sum()
            multiplier = molecular.target_estimator_multiplier
            value = value + .5 * protocol.endpoint_coefficient / protocol.endpoint_scale * (multiplier * target_mse + control_mse)
            value = value + multiplier * protocol.effect_coefficient / protocol.effect_scale * effect
        value.backward()
        return value.detach()

    def run_integrated(self, fixture, chunk, **overrides):
        model, inputs, protocol, geometry, blocks, cache, decoder, molecular = fixture
        arguments = dict(protocol=protocol, execution=self.execution(chunk), geometry=geometry,
                         blocks=blocks, cache=cache, sampling_scale=1.5, decoder=decoder, molecular=molecular)
        arguments.update(overrides)
        return accumulate_training_objective(model, inputs, **arguments)

    def test_complete_objective_and_gradients_match_monolithic_across_microbatches(self):
        for family in ("F0", "F2", "F4", "F5"):
            maximum = 0.
            for chunk in (1, 2, 4):
                with self.subTest(family=family, chunk=chunk):
                    fixture = self.fixture(family)
                    model, inputs, protocol, geometry, blocks, cache, decoder, molecular = fixture
                    reference = copy.deepcopy(model)
                    expected = self.reference(reference, inputs, protocol, geometry, blocks, cache, decoder, molecular)
                    actual = self.run_integrated(fixture, chunk)
                    torch.testing.assert_close(actual.displayed_objective, expected.double(), atol=3e-5, rtol=3e-5)
                    for parameter, exact in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(parameter.grad, exact.grad, atol=3e-5, rtol=3e-4)
                        if parameter.grad is not None:
                            maximum = max(maximum, (parameter.grad - exact.grad).abs().max().item())
                    self.assertEqual(actual.state_observations, 3)
                    self.assertEqual(actual.target_molecular_observations, 2)
                    self.assertEqual(actual.control_molecular_observations, 2)
                    self.assertEqual(actual.control_component_first_passes, 4)
                    self.assertEqual(actual.control_component_replays, 4)
                    self.assertEqual(actual.optimizer_calls, 0)
                    self.assertEqual(actual.cache_writes, 0)
                    self.assertEqual(actual.logical_result.local_loss_sums["molecular_proxy"].item(), 0)
                    self.assertTrue(all(p.grad is None for p in decoder.parameters()))
            print(f"Synthetic integrated {family}: max absolute gradient difference={maximum:.9g}; atol3e-5 rtol3e-4")

    def test_target_sampling_multiplier_matches_monolithic_objective_and_gradients(self):
        for family in ("F0", "F4"):
            for chunk in (1, 3):
                with self.subTest(family=family, chunk=chunk):
                    fixture = list(self.fixture(family))
                    fixture[7] = replace(fixture[7], target_estimator_multiplier=1.75)
                    model, inputs, protocol, geometry, blocks, cache, decoder, molecular = fixture
                    reference = copy.deepcopy(model)
                    expected = self.reference(reference, inputs, protocol, geometry, blocks, cache, decoder, molecular)
                    actual = self.run_integrated(fixture, chunk)
                    torch.testing.assert_close(actual.displayed_objective, expected.double(), atol=3e-5, rtol=3e-5)
                    for parameter, exact in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(parameter.grad, exact.grad, atol=3e-5, rtol=3e-4)
                    terms = actual.molecular_terms
                    self.assertEqual(terms["target_estimator_multiplier"], 1.75)
                    torch.testing.assert_close(terms["target_mse_estimator"], 1.75 * terms["target_mse"])
                    torch.testing.assert_close(terms["effect_mse_estimator"], 1.75 * terms["effect_mse"])
                    reconstructed = (.5 * protocol.endpoint_coefficient / protocol.endpoint_scale
                                     * (terms["target_mse_estimator"] + terms["control_mse"])
                                     + protocol.effect_coefficient / protocol.effect_scale * terms["effect_mse_estimator"])
                    torch.testing.assert_close(actual.molecular_objective, reconstructed, atol=1e-12, rtol=1e-12)
                    self.assertEqual(actual.optimizer_calls, 0)

    def test_target_multiplier_scales_shared_control_contrast_gradient_not_control_endpoint(self):
        fixture = list(self.fixture("F0"))
        fixture[7] = replace(fixture[7], target_estimator_multiplier=2.5)
        protocol, molecular = fixture[2], fixture[7]
        captured = []

        def capture(*args, **kwargs):
            plan = molecular_adjoint(*args, **kwargs)
            captured.append((args, kwargs, plan))
            return plan

        with patch("frontier.training_objective.molecular_adjoint", side_effect=capture):
            self.run_integrated(fixture, 2)
        self.assertEqual(len(captured), 1)
        args, coefficients, plan = captured[0]
        ct = .5 * protocol.endpoint_coefficient / protocol.endpoint_scale
        ce = protocol.effect_coefficient / protocol.effect_scale
        self.assertEqual(coefficients["target_coefficient"], 2.5 * ct)
        self.assertEqual(coefficients["control_coefficient"], ct)
        self.assertEqual(coefficients["effect_coefficient"], 2.5 * ce)
        qt = plan.target_compositions.clone().requires_grad_(True)
        qc = plan.control_compositions.clone().requires_grad_(True)
        et = transform_composition(qt) - transform_composition(args[1])
        ec = transform_composition(qc) - transform_composition(args[3])
        contrast = et - ec[coefficients["target_donor_indices"]]
        targeting = (molecular.target_weights * et.square().mean(-1)).sum()
        effect = (molecular.target_weights * contrast.square().mean(-1)).sum()
        control = (molecular.controls.reduction_weights * ec.square().mean(-1)).sum()
        control_only = torch.autograd.grad(ct * control, qc, retain_graph=True)[0]
        exact = torch.autograd.grad(2.5 * ct * targeting + ct * control + 2.5 * ce * effect, (qt, qc))
        torch.testing.assert_close(plan.target_derivatives, exact[0], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(plan.control_derivatives, exact[1], atol=1e-10, rtol=1e-10)
        self.assertGreater(float((plan.control_derivatives - control_only).abs().max()), 1e-8)
        # Multiplying the whole molecular objective would wrongly scale the
        # once-per-donor control endpoint by the target sampling multiplier.
        self.assertGreater(float(((2.5 - 1) * control_only).abs().max()), 1e-8)

    def test_target_sampling_multiplier_requires_positive_finite_fixed_scalar(self):
        fixture = self.fixture("F0")
        for invalid in (0., -1., float("nan"), float("inf"), float("-inf"), True, "1"):
            with self.subTest(multiplier=invalid), self.assertRaises(ValueError):
                replace(fixture[7], target_estimator_multiplier=invalid)
        self.assertEqual(fixture[7].target_estimator_multiplier, 1.0)
        self.assertTrue(all(parameter.grad is None for parameter in fixture[0].parameters()))

    def test_state_disables_count_and_aux_without_reading_molecular_inputs(self):
        fixture = self.fixture("F3", "state")
        model, inputs, protocol, geometry, _, _, decoder, molecular = fixture
        reference = copy.deepcopy(model)
        expected = self.reference(reference, inputs, protocol, geometry, None, None, None, None)
        actual = self.run_integrated(fixture, 2, blocks=None, cache=None, decoder=object(), molecular=object())
        torch.testing.assert_close(actual.displayed_objective, expected, atol=3e-5, rtol=3e-5)
        self.assertIsNone(actual.molecular_objective)
        self.assertIsNone(actual.logical_result.count_ce)
        self.assertNotIn("shape", actual.logical_result.local_loss_sums)
        self.assertEqual(actual.decoder_law_calls, 0)
        self.assertEqual(actual.model_forward_calls, 2)
        for parameter, exact in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, exact.grad, atol=3e-5, rtol=3e-4)

    def test_no_rng_cache_parameter_or_decoder_mutation(self):
        fixture = self.fixture("F2")
        model, inputs, _, _, _, cache, decoder, _ = fixture
        before_model = copy.deepcopy(model.state_dict())
        before_decoder = copy.deepcopy(decoder.state_dict())
        before_cache = cache.state_dict()
        rng = torch.get_rng_state().clone()
        self.run_integrated(fixture, 2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before_model[key]))
        for key, value in decoder.state_dict().items():
            self.assertTrue(torch.equal(value, before_decoder[key]))
        for key in ("values", "age", "seen"):
            self.assertTrue(torch.equal(cache.state_dict()[key], before_cache[key]))

    def test_molecular_without_count_gets_an_explicit_complete_prepass(self):
        fixture = list(self.fixture("F0"))
        fixture[2] = replace(fixture[2], count_coefficient=0.)
        reference = copy.deepcopy(fixture[0])
        expected = self.reference(reference, *fixture[1:])
        actual = self.run_integrated(fixture, 2)
        torch.testing.assert_close(actual.displayed_objective, expected, atol=3e-5, rtol=3e-5)
        self.assertEqual(actual.logical_result.first_pass_calls, 0)
        self.assertEqual(actual.model_forward_calls, 8)  # Two target passes and two control passes.
        for parameter, exact in zip(fixture[0].parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, exact.grad, atol=3e-5, rtol=3e-4)

    def test_retained_joint_base_needs_no_molecular_or_shape_payload(self):
        fixture = list(self.fixture("F0"))
        fixture[2] = replace(fixture[2], shape_coefficient=0., endpoint_coefficient=0., effect_coefficient=0.)
        reference = copy.deepcopy(fixture[0])
        expected = self.reference(reference, *fixture[1:])
        actual = self.run_integrated(fixture, 2, decoder=None, molecular=None)
        torch.testing.assert_close(actual.displayed_objective, expected, atol=3e-5, rtol=3e-5)
        self.assertIsNone(actual.molecular_objective)
        self.assertEqual(actual.decoder_law_calls, 0)
        self.assertEqual(actual.model_forward_calls, 4)
        self.assertNotIn("shape", actual.logical_result.local_loss_sums)
        for parameter, exact in zip(fixture[0].parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, exact.grad, atol=3e-5, rtol=3e-4)

    def test_frozen_decoder_pool_and_fitting_routing_admission(self):
        for problem in ("decoder", "pool", "donor", "catalog_donor", "budget", "target_is_control"):
            with self.subTest(problem=problem):
                fixture = list(self.fixture("F0"))
                if problem == "decoder":
                    fixture[6].requires_grad_(True)
                elif problem == "pool":
                    fixture[7] = replace(fixture[7], controls=replace(fixture[7].controls,
                        mixture_weights=torch.tensor([.5, .75, .4, .6], dtype=torch.float64)))
                elif problem == "donor":
                    fixture[1] = replace(fixture[1], donor_ids=("D1", "D3", "D4", "D4"))
                elif problem == "catalog_donor":
                    fixture[1] = replace(fixture[1], donor_ids=("D4", "D3", "D4", "D4"))
                elif problem == "target_is_control":
                    fixture[7] = replace(fixture[7], target_law_indices=torch.tensor([1, 2]))
                kwargs = {"execution": replace(self.execution(2), maximum_molecular_elements=1)} if problem == "budget" else {}
                with self.assertRaises(ValueError):
                    self.run_integrated(fixture, 2, **kwargs)
                self.assertTrue(all(p.grad is None for p in fixture[0].parameters()))

    def test_molecular_working_bound_counts_peak_buffers_and_rejects_old_cap_before_forward(self):
        fixture = self.fixture("F0")
        model, _, _, _, _, _, decoder, molecular = fixture
        targets, donors, components = (len(molecular.target_law_indices),
                                      len(molecular.controls.donor_ids), len(molecular.controls.source))
        required = molecular_working_element_bound(targets=targets, donors=donors,
            components=components, genes=decoder.features, target_microbatch=2, control_microbatch=2)
        self.assertEqual(required, (12 * 2 + 10 * 2 + 4 * 4 + 12 * 2 + 12 * 2 + 8) * 11)
        old_cap = (2 * targets + 3 * donors + 2 * components) * decoder.features
        self.assertGreater(required, old_cap)
        # The old boundary used to admit this fixture. Both that old cap and
        # one below the corrected boundary must fail without any forward or
        # backward call. No existing execution cap is enlarged in this test.
        for cap in (old_cap, required - 1):
            with self.subTest(cap=cap), patch.object(model, "forward", side_effect=AssertionError("must reject before forward")):
                with self.assertRaisesRegex(ValueError, "working-buffer envelope.*unchanged"):
                    self.run_integrated(fixture, 2,
                        execution=replace(self.execution(2), maximum_molecular_elements=cap))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(self.execution(2).maximum_molecular_elements, 10000)

    def test_molecular_working_bound_for_full_logical_batch_and_invalid_metadata(self):
        arguments = dict(targets=1024, donors=3, components=3, genes=18129,
                         target_microbatch=32, control_microbatch=32)
        actual = molecular_working_element_bound(**arguments)
        self.assertEqual(actual, 231289782)
        self.assertEqual(actual * 8, 1850318256)
        self.assertGreater(actual, (2 * 1024 + 3 * 3 + 2 * 3) * 18129)
        smaller = molecular_working_element_bound(**dict(arguments, target_microbatch=1, control_microbatch=1))
        self.assertLess(smaller, actual)
        # A component pool smaller than the microbatch does not fabricate
        # extra component rows; a larger real pool must increase admission.
        self.assertEqual(molecular_working_element_bound(**dict(arguments, control_microbatch=128)), actual)
        self.assertGreater(molecular_working_element_bound(**dict(arguments, components=300)), actual)
        for field, value in (("targets", 0), ("donors", True), ("genes", 18129.), ("target_microbatch", -1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                molecular_working_element_bound(**dict(arguments, **{field: value}))


if __name__ == "__main__":
    unittest.main()
