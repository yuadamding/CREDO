import unittest

import torch

from frontier.measure_ops import (particle_integrals, source_log_frequencies,
    catalog_log_frequencies, calibrate_abundance, calibrate_dispersion,
    round_robin_component_counts, ensemble_catalog, ensemble_guide,
    weighted_multiscale_kernel, multiscale_mmd2, nonnegative_score)


class MeasureTests(unittest.TestCase):
    def test_nonnegative_scoring_preserves_raw_roundoff(self):
        record = nonnegative_score(torch.tensor(-1e-15, dtype=torch.float64), component_absolute_sum=2.)
        self.assertEqual(record["risk"], 0.)
        self.assertEqual(record["raw_risk"], -1e-15)
        self.assertTrue(record["negative_roundoff_corrected"])
        with self.assertRaises(ValueError):
            nonnegative_score(torch.tensor(-1e-7, dtype=torch.float64), component_absolute_sum=2.)
        with self.assertRaises(ValueError):
            nonnegative_score(torch.tensor(0., dtype=torch.float32), component_absolute_sum=1.)

    def test_constant_reaction_preserves_state_and_changes_mass(self):
        logw, logmass = particle_integrals(torch.full((5,), 0.8, dtype=torch.float64))
        torch.testing.assert_close(logw.exp(), torch.full((5,), .2, dtype=torch.float64))
        self.assertAlmostEqual(logmass.item(), .8)

    def test_weighted_base_and_gradient(self):
        multipliers = torch.tensor([-.2, .7], dtype=torch.float64, requires_grad=True)
        base = torch.tensor([.8, .2], dtype=torch.float64)
        logw, logmass = particle_integrals(multipliers, base.log())
        logmass.backward()
        torch.testing.assert_close(multipliers.grad, logw.exp())

    def test_prefix_mass_recomputed(self):
        values = torch.tensor([0., 1., 3.], dtype=torch.float64)
        small, mass = particle_integrals(values[:2])
        large, _ = particle_integrals(values)
        self.assertAlmostEqual(small.exp().sum().item(), 1.)
        self.assertLess(large[:2].exp().sum().item(), 1.)
        self.assertAlmostEqual(mass.item(), torch.log(torch.exp(values[:2]).mean()).item())

    def test_zero_counts_and_complement_ratios(self):
        source = source_log_frequencies(torch.tensor([0, 4, 9, 2]))
        prediction = catalog_log_frequencies(source, torch.tensor([1]), torch.tensor([.7]))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        torch.testing.assert_close(prediction[0] - prediction[2], source[0] - source[2])
        self.assertAlmostEqual(prediction.exp().sum().item(), 1.)

    def test_duplicate_or_outside_catalog_rejected(self):
        source = source_log_frequencies(torch.tensor([1, 2, 3]))
        for indices in ([1, 1], [0, 3]):
            with self.assertRaises(ValueError):
                catalog_log_frequencies(source, torch.tensor(indices), torch.zeros(2))

    def test_calibration_endpoints_and_finite_logs(self):
        source = torch.log_softmax(torch.tensor([-2., -3., -4.], dtype=torch.float64), 0)
        prediction = torch.log_softmax(torch.tensor([-2000., -1., -3.], dtype=torch.float64), 0)
        torch.testing.assert_close(calibrate_abundance(source, prediction, 0), source)
        torch.testing.assert_close(calibrate_abundance(source, prediction, 1), prediction)
        self.assertTrue(bool(torch.isfinite(calibrate_abundance(source, prediction, .25)).all()))

    def test_spread_calibration_preserves_weighted_mean(self):
        x = torch.tensor([[0., 1.], [1., 3.], [4., 2.]], dtype=torch.float64)
        w = torch.tensor([.2, .3, .5], dtype=torch.float64)
        y = calibrate_dispersion(x, w.log(), 1.3)
        torch.testing.assert_close((x * w[:, None]).sum(0), (y * w[:, None]).sum(0))
        torch.testing.assert_close(torch.cdist(y, y), 1.3 * torch.cdist(x, x))

    def test_ensemble_component_counts(self):
        self.assertEqual(round_robin_component_counts(4096), (1366, 1365, 1365))
        self.assertEqual(round_robin_component_counts(8192), (2731, 2731, 2730))

    def test_finite_measure_mixture_not_equal_cells(self):
        states = [torch.zeros((2, 1), dtype=torch.float64), torch.full((3, 1), 10., dtype=torch.float64)]
        logw = [torch.full((2,), .5, dtype=torch.float64).log(), torch.full((3,), 1 / 3, dtype=torch.float64).log()]
        q = torch.tensor([.1, .4], dtype=torch.float64)
        combined, lw, lq = ensemble_guide(states, logw, q.log())
        self.assertEqual(len(combined), 5)
        self.assertAlmostEqual(lw[:2].exp().sum().item(), .2)
        self.assertAlmostEqual(lw[2:].exp().sum().item(), .8)
        self.assertAlmostEqual(lq.exp().item(), .25)
        self.assertAlmostEqual((combined[:, 0] * lw.exp()).sum().item(), 8.)

    def test_ensemble_catalog_is_average_frequencies(self):
        a = torch.tensor([.1, .9], dtype=torch.float64)
        b = torch.tensor([.4, .6], dtype=torch.float64)
        torch.testing.assert_close(ensemble_catalog([a.log(), b.log()]).exp(), (a + b) / 2)

    def test_multiscale_blocks_and_identical_laws(self):
        generator = torch.Generator().manual_seed(2)
        x = torch.randn((7, 3), generator=generator, dtype=torch.float64)
        y = torch.randn((5, 3), generator=generator, dtype=torch.float64)
        w = torch.ones(7, dtype=torch.float64) / 7
        v = torch.ones(5, dtype=torch.float64) / 5
        a = weighted_multiscale_kernel(x, w, y, v, [.5, 1., 2.], block=2)
        b = weighted_multiscale_kernel(x, w, y, v, [.5, 1., 2.], block=128)
        torch.testing.assert_close(a, b)
        self.assertAlmostEqual(multiscale_mmd2(x, w, x, w, [.5, 1., 2.], block=3).item(), 0.)

    def test_mmd_gradients_and_invalid_bandwidth(self):
        x = torch.tensor([[0.], [1.]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[3.], [5.]], dtype=torch.float64)
        w = torch.tensor([.5, .5], dtype=torch.float64)
        value = multiscale_mmd2(x, w, y, w, [.5, 1.], block=1)
        self.assertGreater(value.item(), 0)
        value.backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            multiscale_mmd2(x, w, y, w, [0.])

    def test_recomputed_block_backward_matches_full_graph(self):
        x = torch.tensor([[0., 1.], [1., 2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
        logits = torch.tensor([-.3, .2, .1], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[2., 3.], [4., 1.]], dtype=torch.float64)
        wy = torch.tensor([.4, .6], dtype=torch.float64)
        values, grads = [], []
        for recompute in (False, True):
            value = multiscale_mmd2(x, logits.softmax(0), y, wy, [.5, 1., 2.], block=2,
                                   recompute_backward=recompute)
            values.append(value.detach())
            grads.append(torch.autograd.grad(value, (x, logits)))
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        for first, second in zip(grads[0], grads[1]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
