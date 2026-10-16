"""Fabricated catalog/gradient checks, without optimizers or scientific inputs."""

from dataclasses import replace
import unittest

import torch

from frontier.count_objective import (ActiveFitnessChunk, CatalogBlock, FitnessCache,
    cached_count_objective, sampling_scale_from_sizes, value_preserving_surrogate)
from frontier.measure_ops import catalog_log_frequencies, source_log_frequencies


BINDING = dict(model_recipe="synthetic-F0", training_subset="synthetic-pair",
               catalog_layout="synthetic-six-category-layout", source_sampler="explicit-fixture")


class CountObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self, dtype=torch.float64):
        values = torch.tensor([.1, -.2, .3, -.4], dtype=dtype, requires_grad=True)
        cache = FitnessCache(values, binding=BINDING, seen=torch.ones(4, dtype=torch.bool),
                             age=torch.tensor([2, 1, 0, 3]),
                             last_full_refresh_update=20, refresh_particles=32)
        blocks = (
            CatalogBlock("A", "catalog-six", source_log_frequencies(torch.tensor([2, 0, 3, 1, 0, 4])).to(dtype),
                         torch.tensor([1, 0, 7, 4, 0, 3]), torch.tensor([0, 2]), torch.tensor([0, 1])),
            CatalogBlock("B", "catalog-six", source_log_frequencies(torch.tensor([4, 2, 0, 0, 1, 3])).to(dtype),
                         torch.tensor([100, 50, 0, 300, 0, 20]), torch.tensor([1, 3]), torch.tensor([2, 3])),
        )
        active = torch.tensor([.7, -.8, .5], dtype=dtype, requires_grad=True)
        indices = torch.tensor([0, 3, 1])
        return values, cache, blocks, active, indices

    def reference(self, cache, blocks, active, indices, scale):
        full = cache.values.detach().clone().index_copy(0, indices, active)
        losses = []
        for block in blocks:
            delta = torch.zeros_like(block.log_source).index_copy(0, block.modeled_indices, full[block.cache_indices])
            p = block.endpoint_counts.to(block.log_source.dtype)
            p = p / p.sum()
            losses.append(-(p * torch.log_softmax(block.log_source + delta, 0)).sum())
        ce = torch.stack(losses).mean()
        return ce, ce + (scale - 1.) * (ce - ce.detach())

    def test_chunked_logical_batch_matches_scalar_value_and_gradient(self):
        for dtype in (torch.float32, torch.float64):
            _, cache, blocks, active, indices = self.fixture(dtype)
            scale = sampling_scale_from_sizes(population=4, logical_batch=3)
            chunks = (ActiveFitnessChunk(indices[:1], active[:1]), ActiveFitnessChunk(indices[1:], active[1:]))
            result = cached_count_objective(blocks, cache, chunks, sampling_scale=scale)
            ref_active = active.detach().clone().requires_grad_()
            expected_ce, expected_surrogate = self.reference(cache, blocks, ref_active, indices, scale)
            self.assertTrue(torch.equal(result.ce, expected_ce))
            self.assertTrue(torch.equal(result.surrogate, result.ce))
            actual_grad, = torch.autograd.grad(result.surrogate, active)
            expected_grad, = torch.autograd.grad(expected_surrogate, ref_active)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
            torch.testing.assert_close(actual_grad, result.active_derivative)
            self.assertFalse(result.active_derivative.requires_grad)
            self.assertEqual(result.modeled_population, 4)

    def test_unchunked_and_three_chunks_are_identical(self):
        _, cache, blocks, active, indices = self.fixture()
        one = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=2.)
        ref = active.detach().clone().requires_grad_()
        many = cached_count_objective(blocks, cache,
            [ActiveFitnessChunk(indices[i:i + 1], ref[i:i + 1]) for i in range(3)], sampling_scale=2.)
        self.assertTrue(torch.equal(one.ce, many.ce))
        left, = torch.autograd.grad(one.surrogate, active)
        right, = torch.autograd.grad(many.surrogate, ref)
        self.assertTrue(torch.equal(left, right))

    def test_equal_donor_reduction_includes_donor_without_active_laws(self):
        _, cache, blocks, active, indices = self.fixture()
        result = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices[:1], active[:1])], sampling_scale=4.)
        self.assertEqual(result.donor_ids, ("A", "B"))
        self.assertTrue(torch.equal(result.ce, result.donor_ce.mean()))
        totals = torch.tensor([b.endpoint_counts.sum().item() for b in blocks], dtype=result.ce.dtype)
        total_count_weighted = (result.donor_ce * totals / totals.sum()).sum()
        self.assertFalse(torch.allclose(result.ce, total_count_weighted))
        expected_b = catalog_log_frequencies(blocks[1].log_source, blocks[1].modeled_indices,
                                              cache.values[blocks[1].cache_indices])
        torch.testing.assert_close(result.catalog_log_probabilities[1], expected_b)

    def test_inactive_cache_has_no_gradient_and_cache_not_mutated(self):
        original, cache, blocks, active, indices = self.fixture()
        before = cache.state_dict()
        # Even an accidentally grad-enabled cache buffer must remain detached.
        cache.values.requires_grad_(True)
        result = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=3.)
        result.surrogate.backward()
        self.assertIsNone(original.grad)
        self.assertIsNone(cache.values.grad)
        self.assertIsNotNone(active.grad)
        self.assertTrue(torch.equal(cache.values, before["values"]))
        self.assertTrue(torch.equal(cache.age, before["age"]))
        self.assertTrue(torch.equal(cache.seen, before["seen"]))

    def test_active_mass_changes_entire_donor_catalog_not_complement_ratios(self):
        _, cache, blocks, active, indices = self.fixture()
        first = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=1.)
        second_active = active.detach().clone()
        second_active[0] += .75
        second = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, second_active)], sampling_scale=1.)
        qa, qb = first.catalog_log_probabilities[0], second.catalog_log_probabilities[0]
        self.assertGreater(qb[0].item(), qa[0].item())
        self.assertTrue(bool((qb[1:] < qa[1:]).all()))
        for left, right in ((1, 3), (4, 5)):
            torch.testing.assert_close(qb[left] - qb[right], blocks[0].log_source[left] - blocks[0].log_source[right])
        self.assertTrue(torch.equal(first.catalog_log_probabilities[1], second.catalog_log_probabilities[1]))

    def test_full_catalog_26504_zeros_and_measure_ops_agreement(self):
        # Only two fabricated modeled laws; count catalog dimension is metadata.
        total = 26504
        source_counts = torch.zeros(total, dtype=torch.int64)
        source_counts[3], source_counts[200] = 50, 100
        counts = torch.zeros(total, dtype=torch.int64)
        counts[3], counts[1000] = 20, 80
        log_source = source_log_frequencies(source_counts)
        cache = FitnessCache(torch.zeros(2, dtype=torch.float64), binding=BINDING,
                             seen=torch.ones(2, dtype=torch.bool), age=torch.zeros(2, dtype=torch.int64))
        block = CatalogBlock("A", "all-26504", log_source, counts, torch.tensor([3, 200]), torch.tensor([0, 1]))
        fitness = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
        result = cached_count_objective([block], cache, [ActiveFitnessChunk(torch.tensor([0]), fitness)], sampling_scale=2.)
        expected = catalog_log_frequencies(log_source, block.modeled_indices, torch.cat((fitness, fitness.new_zeros(1))))
        torch.testing.assert_close(result.catalog_log_probabilities[0], expected)
        self.assertEqual(result.catalog_log_probabilities[0].numel(), total)
        self.assertTrue(bool(torch.isfinite(result.catalog_log_probabilities[0]).all()))
        self.assertEqual(torch.count_nonzero(counts).item(), 2)
        torch.testing.assert_close(result.ce, -.2 * expected[3] - .8 * expected[1000])

    def test_independent_microbatch_normalization_is_not_logical_objective(self):
        _, cache, blocks, active, indices = self.fixture()
        logical = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=4 / 3)
        separate_a = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices[:1], active[:1])], sampling_scale=4.)
        separate_b = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices[1:], active[1:])], sampling_scale=2.)
        wrong = (separate_a.surrogate + separate_b.surrogate) / 2
        self.assertFalse(torch.allclose(logical.ce, wrong))
        intended, = torch.autograd.grad(logical.surrogate, active, retain_graph=True)
        wrong_gradient, = torch.autograd.grad(wrong, active)
        self.assertFalse(torch.allclose(intended, wrong_gradient))

    def test_duplicates_in_or_across_chunks_rejected(self):
        _, cache, blocks, active, indices = self.fixture()
        for chunks in (
            [ActiveFitnessChunk(torch.tensor([0, 0]), active[:2])],
            [ActiveFitnessChunk(indices[:1], active[:1]), ActiveFitnessChunk(indices[:1], active[:1])],
        ):
            with self.assertRaises(ValueError):
                cached_count_objective(blocks, cache, chunks, sampling_scale=1.)

    def test_cache_roundtrip_preserves_objective_gradient_and_metadata(self):
        _, cache, blocks, active, indices = self.fixture()
        cache.tick()
        pre_update = torch.tensor([.8], dtype=torch.float64, requires_grad=True)
        cache.update_active(torch.tensor([2]), pre_update)
        self.assertFalse(cache.values.requires_grad)
        self.assertEqual(cache.age.tolist(), [3, 2, 0, 4])
        payload = cache.state_dict()
        recovered = FitnessCache.from_state_dict(payload, expected_binding=BINDING, expected_size=4, expected_dtype=torch.float64)
        first = cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=2.)
        second_active = active.detach().clone().requires_grad_()
        second = cached_count_objective(blocks, recovered, [ActiveFitnessChunk(indices, second_active)], sampling_scale=2.)
        self.assertTrue(torch.equal(first.ce, second.ce))
        left, = torch.autograd.grad(first.surrogate, active)
        right, = torch.autograd.grad(second.surrogate, second_active)
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(recovered.last_full_refresh_update, 20)
        self.assertEqual(recovered.refresh_particles, 32)
        payload["values"].zero_()
        self.assertFalse(torch.equal(payload["values"], recovered.values))

    def test_cache_restore_rejects_binding_dtype_size_and_unknown_fields(self):
        _, cache, _, _, _ = self.fixture()
        payload = cache.state_dict()
        for changed in (
            {**payload, "binding": {**BINDING, "training_subset": "different"}},
            {**payload, "schema": "unrecognized"},
            {**payload, "optimizer_state": {}},
            {**payload, "values": payload["values"].float()},
            {**payload, "last_full_refresh_update": -1},
        ):
            with self.assertRaises(ValueError):
                FitnessCache.from_state_dict(changed, expected_binding=BINDING, expected_size=4, expected_dtype=torch.float64)
        with self.assertRaises(ValueError):
            FitnessCache.from_state_dict(payload, expected_binding=BINDING, expected_size=3, expected_dtype=torch.float64)

    def test_explicit_full_refresh_and_incomplete_cache_guard(self):
        _, cache, blocks, active, indices = self.fixture()
        cache.seen[1] = False
        with self.assertRaises(ValueError):
            cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=1.)
        values = torch.tensor([1., 2., 3., 4.], dtype=torch.float64, requires_grad=True)
        cache.apply_full_refresh(values, update_index=21, particles=17)
        self.assertTrue(bool(cache.seen.all()))
        self.assertEqual(cache.age.sum().item(), 0)
        self.assertEqual(cache.refresh_particles, 17)
        self.assertEqual(cache.last_full_refresh_update, 21)
        self.assertFalse(cache.values.requires_grad)

    def test_layout_and_data_validation(self):
        _, cache, blocks, active, indices = self.fixture()
        invalid_blocks = (
            [blocks[0], replace(blocks[1], donor_id="A")],
            [blocks[0], replace(blocks[1], catalog_id="another")],
            [blocks[0], replace(blocks[1], cache_indices=torch.tensor([1, 3]))],
            [blocks[0], replace(blocks[1], endpoint_counts=torch.zeros(6, dtype=torch.long))],
            [blocks[0], replace(blocks[1], endpoint_counts=blocks[1].endpoint_counts.double())],
            [blocks[0], replace(blocks[1], log_source=blocks[1].log_source.float())],
            [blocks[0], replace(blocks[1], log_source=blocks[1].log_source + 1)],
        )
        for bad in invalid_blocks:
            with self.assertRaises(ValueError):
                cached_count_objective(bad, cache, [ActiveFitnessChunk(indices, active)], sampling_scale=1.)
        with self.assertRaises(ValueError):
            cached_count_objective(blocks, cache, [ActiveFitnessChunk(indices, active.float())], sampling_scale=1.)

    def test_sampling_scale_value_and_gradient_contract(self):
        x = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        y = x.square()
        surrogate = value_preserving_surrogate(y, sampling_scale=3.)
        self.assertTrue(torch.equal(surrogate, y))
        self.assertEqual(torch.autograd.grad(surrogate, x)[0].item(), 12.)
        self.assertEqual(sampling_scale_from_sizes(population=17, logical_batch=3), 17 / 3)
        for scale in (0., -1., float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                value_preserving_surrogate(y, sampling_scale=scale)
        for population, batch in ((0, 1), (1, 0), (2, 3), (True, 1)):
            with self.assertRaises(ValueError):
                sampling_scale_from_sizes(population=population, logical_batch=batch)


if __name__ == "__main__":
    unittest.main()
