"""Tiny actual-model first-order proofs; no optimizer step or real input."""

import copy
from dataclasses import replace
import unittest

import torch

from frontier.count_objective import (ActiveFitnessChunk, CatalogBlock, FitnessCache,
                                      cached_count_objective)
from frontier.logical_batch import (LocalForward, LogicalMicrobatch, LossReduction,
                                    accumulate_logical_batch)
from frontier.measure_ops import source_log_frequencies
from frontier.models import build_model


BINDING = dict(model_recipe="synthetic", training_subset="two-synthetic-donors",
               catalog_layout="six-per-donor", source_sampler="prebound-source-noise")


class LogicalBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self, family="F0", microbatch=2, phase="joint"):
        model = build_model(family + "-V0", seed=174).set_phase(phase)
        if family == "F2":
            with torch.no_grad():
                model.mixture_logits.copy_(torch.tensor([-1., .2, .8, -.5]))
        generator = torch.Generator().manual_seed(211)
        source = torch.randn((4, 8, 48), generator=generator)
        noise = torch.randn((8, 4, 8, 48), generator=generator)
        endpoint = .3 * source + .7
        weights = torch.tensor([1., .5, 2., 1.5])
        active = torch.tensor([0, 1, 3, 5])
        batches = []
        for start in range(0, 4, microbatch):
            stop = min(4, start + microbatch)
            batches.append(LogicalMicrobatch(f"slice-{start}-{stop}", active[start:stop],
                dict(source=source[start:stop], noise=noise[:, start:stop], endpoint=endpoint[start:stop],
                     guide_weight=weights[start:stop])))
        blocks = (
            CatalogBlock("A", "six", source_log_frequencies(torch.tensor([2, 0, 7, 1, 3, 0])).float(),
                         torch.tensor([1, 10, 20, 0, 4, 2]), torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2])),
            CatalogBlock("B", "six", source_log_frequencies(torch.tensor([1, 4, 0, 2, 0, 8])).float(),
                         torch.tensor([12, 1, 2, 30, 0, 7]), torch.tensor([1, 3, 5]), torch.tensor([3, 4, 5])),
        )
        cache = FitnessCache(torch.tensor([.3, -.4, .1, .2, -.1, .5]), binding=BINDING,
                             seen=torch.ones(6, dtype=torch.bool), age=torch.arange(6),
                             last_full_refresh_update=11, refresh_particles=32)
        whole = LogicalMicrobatch("whole", active, dict(source=source, noise=noise, endpoint=endpoint, guide_weight=weights))
        reductions = {"state": LossReduction(.7, weights.sum().item()), "aux": LossReduction(.2, 4.)}
        return model, batches, blocks, cache, whole, reductions

    def callback(self, model, reductions, calls=None):
        def forward(batch, *, phase, include_local_losses):
            if calls is not None:
                calls.append((batch.identity, include_local_losses, torch.is_grad_enabled()))
            self.assertEqual(model.phase, phase)
            payload = batch.payload
            rollout = model(payload["source"], payload["noise"], compute_anchor=False)
            terms = {}
            if include_local_losses:
                state = ((rollout.states - payload["endpoint"]).square().mean((1, 2)) * payload["guide_weight"]).sum()
                aux = rollout.reaction_log_increment.square().mean(1).sum()
                terms = {key: {"state": state, "aux": aux}[key] for key in reductions}
            return LocalForward(rollout.log_mass_increment, terms,
                                dict(identity=batch.identity, laws=len(batch.cache_indices), phase=phase))
        return forward

    def regularizer(self, model):
        return sum(parameter.square().sum() for parameter in model.parameters()) / sum(parameter.numel() for parameter in model.parameters())

    def reference(self, model, whole, blocks, cache, reductions, count_coefficient=.4, global_coefficient=.03):
        result = self.callback(model, reductions)(whole, phase=model.phase, include_local_losses=True)
        value = sum(reductions[name].coefficient * numerator / reductions[name].logical_denominator
                    for name, numerator in result.loss_sums.items())
        if count_coefficient:
            count = cached_count_objective(blocks, cache,
                [ActiveFitnessChunk(whole.cache_indices, result.log_mass)], sampling_scale=6 / 4)
            value = value + count_coefficient * count.surrogate
        if global_coefficient:
            value = value + global_coefficient * self.regularizer(model)
        value.backward()
        return value.detach()

    def run_adapter(self, model, batches, blocks, cache, reductions, *, count_coefficient=.4,
                    global_coefficient=.03, forward=None, global_callback=None):
        return accumulate_logical_batch(model, batches, forward or self.callback(model, reductions), phase=model.phase,
            local_reductions=reductions, count_coefficient=count_coefficient, sampling_scale=6 / 4,
            blocks=blocks, cache=cache, global_regularizer=global_callback or (lambda: self.regularizer(model)),
            global_regularizer_coefficient=global_coefficient, replay_atol=2e-6, replay_rtol=2e-6)

    def test_actual_f0_f2_f4_all_microbatch_sizes_match_monolithic_gradients(self):
        for family in ("F0", "F2", "F4"):
            maximum = 0.
            maximum_value = 0.
            for chunk in (1, 2, 3, 4):
                with self.subTest(family=family, chunk=chunk):
                    model, batches, blocks, cache, whole, reductions = self.fixture(family, chunk)
                    reference = copy.deepcopy(model)
                    expected_value = self.reference(reference, whole, blocks, cache, reductions)
                    actual = self.run_adapter(model, batches, blocks, cache, reductions)
                    torch.testing.assert_close(actual.displayed_objective, expected_value, atol=2e-5, rtol=2e-5)
                    maximum_value = max(maximum_value, abs(actual.displayed_objective.item() - expected_value.item()))
                    for (name, parameter), (ref_name, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
                        self.assertEqual(name, ref_name)
                        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-5, rtol=2e-5)
                        maximum = max(maximum, (parameter.grad - ref_parameter.grad).abs().max().item())
                    self.assertEqual(actual.first_pass_calls, len(batches))
                    self.assertEqual(actual.replay_calls, len(batches))
                    self.assertEqual(actual.optimizer_calls, 0)
                    self.assertEqual(actual.cache_writes, 0)
            print(f"Synthetic {family} logical-batch agreement: max absolute value difference={maximum_value:.9g}, gradient difference={maximum:.9g}; bound atol=rtol=2e-5")

    def test_count_only_gradient_accumulates_across_all_chunks(self):
        model, batches, blocks, cache, whole, _ = self.fixture("F0", 1)
        reference = copy.deepcopy(model)
        expected_value = self.reference(reference, whole, blocks, cache, {}, global_coefficient=0.)
        actual = self.run_adapter(model, batches, blocks, cache, {}, global_coefficient=0.)
        torch.testing.assert_close(actual.displayed_objective, expected_value)
        matched_nonzero = 0
        for actual_parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            if ref_parameter.grad is not None:
                torch.testing.assert_close(actual_parameter.grad, ref_parameter.grad, atol=2e-5, rtol=2e-5)
                matched_nonzero += int(ref_parameter.grad.abs().sum().item() > 0)
        self.assertGreater(matched_nonzero, 0)
        self.assertEqual(actual.backward_calls, len(batches))

    def test_preserves_existing_gradients_and_global_regularizer_called_once(self):
        model, batches, blocks, cache, whole, reductions = self.fixture("F4", 2)
        reference = copy.deepcopy(model)
        self.reference(reference, whole, blocks, cache, reductions)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .125)
        calls = []
        result = self.run_adapter(model, batches, blocks, cache, reductions,
            global_callback=lambda: calls.append("global") or self.regularizer(model))
        self.assertEqual(calls, ["global"])
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad + .125, atol=2e-5, rtol=2e-5)
        self.assertEqual(result.backward_calls, len(batches) + 1)

    def test_source_noise_rng_parameters_and_cache_unchanged(self):
        model, batches, blocks, cache, _, reductions = self.fixture("F2", 2)
        initial_model = {key: value.clone() for key, value in model.state_dict().items()}
        initial_payload = [{key: value.clone() for key, value in batch.payload.items()} for batch in batches]
        initial_cache = cache.state_dict()
        rng = torch.get_rng_state().clone()
        calls = []
        result = self.run_adapter(model, batches, blocks, cache, reductions,
                                  forward=self.callback(model, reductions, calls))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, initial_model[key]))
        for batch, initial in zip(batches, initial_payload):
            for key, value in batch.payload.items():
                self.assertTrue(torch.equal(value, initial[key]))
        for key in ("values", "seen", "age"):
            self.assertTrue(torch.equal(cache.state_dict()[key], initial_cache[key]))
        self.assertEqual(calls, [(batch.identity, False, False) for batch in batches]
                         + [(batch.identity, True, True) for batch in batches])
        self.assertFalse(result.pre_update_log_mass.requires_grad)
        self.assertTrue(all(not value.requires_grad for value in result.local_loss_sums.values()))

    def test_state_stage_skips_count_first_pass_and_uses_full_denominator(self):
        model, batches, _, _, whole, reductions = self.fixture("F0", 3, "state")
        reference = copy.deepcopy(model)
        expected_value = self.reference(reference, whole, None, None, reductions, count_coefficient=0.)
        result = self.run_adapter(model, batches, None, None, reductions, count_coefficient=0.)
        torch.testing.assert_close(result.displayed_objective, expected_value, atol=2e-5, rtol=2e-5)
        self.assertIsNone(result.count_ce)
        self.assertIsNone(result.count_gradient_coefficients)
        self.assertEqual(result.first_pass_calls, 0)
        self.assertEqual(result.replay_calls, 2)
        for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-5, rtol=2e-5)

    def test_hidden_rng_and_parameter_input_or_cache_mutations_rejected(self):
        for mutation in ("rng", "parameter", "input", "cache"):
            with self.subTest(mutation=mutation), torch.random.fork_rng(devices=[]):
                model, batches, blocks, cache, _, reductions = self.fixture()
                original = self.callback(model, reductions)
                def bad(batch, **kwargs):
                    output = original(batch, **kwargs)
                    if mutation == "rng":
                        torch.rand(1)
                    elif mutation == "parameter":
                        with torch.no_grad():
                            next(model.parameters()).add_(1)
                    elif mutation == "input":
                        batch.payload["source"].add_(1)
                    else:
                        cache.tick()
                    return output
                with self.assertRaises(RuntimeError):
                    self.run_adapter(model, batches, blocks, cache, reductions, forward=bad)

    def test_replay_mass_metadata_or_detached_mass_failure(self):
        for mismatch in ("mass", "metadata", "detached"):
            model, batches, blocks, cache, _, reductions = self.fixture()
            original = self.callback(model, reductions)
            def bad(batch, *, phase, include_local_losses):
                output = original(batch, phase=phase, include_local_losses=include_local_losses)
                if include_local_losses:
                    if mismatch == "mass":
                        output.log_mass = output.log_mass + .1
                    elif mismatch == "metadata":
                        output.metadata = {**output.metadata, "phase": "changed"}
                    else:
                        output.log_mass = output.log_mass.detach()
                return output
            with self.assertRaises((ValueError, RuntimeError)):
                self.run_adapter(model, batches, blocks, cache, reductions, forward=bad)

    def test_duplicate_laws_and_invalid_phase_denominator_fail_before_forward(self):
        model, batches, blocks, cache, _, reductions = self.fixture()
        called = []
        def no_call(*args, **kwargs):
            called.append(True)
            raise AssertionError("Admission should fail before callback")
        duplicated = [batches[0], replace(batches[1], cache_indices=batches[0].cache_indices)]
        with self.assertRaises(ValueError):
            self.run_adapter(model, duplicated, blocks, cache, reductions, forward=no_call)
        with self.assertRaises(ValueError):
            self.run_adapter(model, batches, blocks, cache, {"state": LossReduction(1., 0.)}, forward=no_call)
        model.set_phase("state")
        with self.assertRaises(ValueError):
            self.run_adapter(model, batches, blocks, cache, reductions, forward=no_call)
        self.assertEqual(called, [])

    def test_disabled_autograd_rejected(self):
        model, batches, blocks, cache, _, reductions = self.fixture()
        with torch.no_grad(), self.assertRaises(ValueError):
            self.run_adapter(model, batches, blocks, cache, reductions)


if __name__ == "__main__":
    unittest.main()
