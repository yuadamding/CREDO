"""Fabricated full-support Sinkhorn checks; no optimizer or scientific input."""

import ast
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from geomloss import SamplesLoss

from frontier import geometry
from frontier.geometry import GeometryOptions, ObservedLaw, RaggedSinkhornGeometry, geometry_width


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def options(self, elements=4096, guides=2):
        return GeometryOptions(elements, guides, "0.2.6")

    def fixture(self, dtype=torch.float64):
        generator = torch.Generator().manual_seed(331)
        x = torch.randn((4, 8, 48), generator=generator, dtype=dtype).requires_grad_()
        logits = torch.randn((4, 8), generator=generator, dtype=dtype).requires_grad_()
        observed = []
        for row, length in enumerate((5, 9, 0, 15)):
            if length == 0:
                observed.append(None)
                continue
            scale = (1., .05, 1., 3.)[row]
            support = torch.randn((length, 48), generator=generator, dtype=dtype) * scale + row
            weight = torch.rand((length,), generator=generator, dtype=dtype) + .01
            observed.append(ObservedLaw(f"law-{row}", support, weight))
        return x, logits, tuple(f"law-{row}" for row in range(4)), observed

    def test_retained_kernel_ast_and_source_identity(self):
        root = Path(__file__).resolve().parents[2]
        source = root / "credo_full_ddr_gse314342_20261006_v1/fitting_gpu_native_v6/bundle/gpu_native.py"
        payload = source.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), geometry.RETAINED_SOURCE_SHA256)
        original = {node.name: node for node in ast.parse(payload).body if isinstance(node, ast.FunctionDef)}
        copied = {node.name: node for node in ast.parse(Path(geometry.__file__).read_text()).body if isinstance(node, ast.FunctionDef)}
        for name in ("geomloss_backend", "softmin_batched", "sinkhorn_batched", "geometry_width"):
            self.assertEqual(ast.dump(original[name]), ast.dump(copied[name]), name)

    def test_scalar_value_and_state_weight_gradients_full_ragged_support(self):
        for dtype, tolerance in ((torch.float64, 2e-9), (torch.float32, 2e-4)):
            with self.subTest(dtype=dtype):
                x, logits, ids, observed = self.fixture(dtype)
                actual = RaggedSinkhornGeometry(options=self.options())(x, logits.log_softmax(1), law_ids=ids, observations=observed)
                actual_grad = torch.autograd.grad(actual.loss_sum, (x, logits))
                ref_x, ref_logits = x.detach().clone().requires_grad_(), logits.detach().clone().requires_grad_()
                scalar = SamplesLoss("sinkhorn", p=2, blur=math.sqrt(.1), scaling=.5, debias=True, backend="tensorized")
                expected = torch.stack([scalar(ref_logits[row].softmax(0), ref_x[row],
                    law.weights.clamp_min(1e-30) / law.weights.clamp_min(1e-30).sum(), law.support)
                    for row, law in enumerate(observed) if law is not None])
                expected_grad = torch.autograd.grad(expected.sum(), (ref_x, ref_logits))
                torch.testing.assert_close(actual.per_law_losses, expected, atol=tolerance, rtol=tolerance)
                for actual_value, expected_value in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual_value, expected_value, atol=tolerance, rtol=tolerance)
                value_error = (actual.per_law_losses - expected).abs().max().item()
                gradient_error = max((actual_value - expected_value).abs().max().item()
                                     for actual_value, expected_value in zip(actual_grad, expected_grad))
                print(f"Synthetic {dtype} retained Sinkhorn/scalar agreement: max absolute value difference={value_error:.9g}, gradient difference={gradient_error:.9g}; atol=rtol={tolerance:g}")
                self.assertEqual(actual.available_indices.tolist(), [0, 1, 3])
                self.assertEqual(actual.unavailable_indices.tolist(), [2])
                self.assertEqual(actual.observation_support_counts, (5, 9, 0, 15))
                self.assertEqual(actual.dropped_observed_rows, 0)
                self.assertEqual(actual.observed_weight_clamp_counts, (0, 0, 0, 0))
                self.assertEqual(actual.backend_identity["version"], "0.2.6")
                self.assertTrue(torch.equal(actual_grad[0][2], torch.zeros_like(actual_grad[0][2])))

    def test_independent_diameters_schedules_and_chunk_invariance(self):
        x, logits, ids, observed = self.fixture()
        one = RaggedSinkhornGeometry(options=self.options(guides=1))(x, logits.log_softmax(1), law_ids=ids, observations=observed)
        many = RaggedSinkhornGeometry(options=self.options(guides=4))(x, logits.log_softmax(1), law_ids=ids, observations=observed)
        torch.testing.assert_close(one.per_law_losses, many.per_law_losses, atol=2e-9, rtol=2e-9)
        self.assertGreater(len(set(one.diameters)), 1)
        self.assertGreater(len(set(one.annealing_schedules)), 1)
        for index, row in enumerate((0, 1, 3)):
            support = torch.cat((x[row].detach(), observed[row].support))
            diameter = (support.amax(0) - support.amin(0)).norm().item()
            self.assertAlmostEqual(one.diameters[index], diameter)
        for info in many.kernel_batches:
            self.assertLessEqual(info["nominal_cost_elements"], self.options().maximum_geometry_elements)

    def test_padding_exact_width_fallback_and_true_oversize_fail_closed(self):
        options = self.options(elements=400)
        self.assertEqual(geometry_width(9, 8, options), 9)
        self.assertEqual(geometry_width(8, 8, options), 8)
        with self.assertRaises(RuntimeError):
            geometry_width(15, 8, options)
        x, logits, ids, observed = self.fixture()
        actual = RaggedSinkhornGeometry(options=options)(x[:2], logits[:2].log_softmax(1), law_ids=ids[:2], observations=observed[:2])
        self.assertEqual(tuple(item["padded_width"] for item in actual.kernel_batches), (8, 9))
        self.assertEqual(actual.observation_support_counts, (5, 9))
        with patch.object(geometry, "sinkhorn_batched", side_effect=AssertionError("No kernel before complete admission")):
            with self.assertRaises(RuntimeError):
                RaggedSinkhornGeometry(options=options)(x, logits.log_softmax(1), law_ids=ids, observations=observed)

    def test_missing_denominator_and_logical_weighted_numerator(self):
        x, logits, ids, observed = self.fixture()
        result = RaggedSinkhornGeometry(options=self.options())(x, logits.log_softmax(1), law_ids=ids, observations=observed)
        self.assertEqual(result.eligible_count, 3)
        torch.testing.assert_close(result.loss_mean, result.loss_sum / 3)
        weight = x.new_tensor([1., 2., 900., 3.])
        torch.testing.assert_close(result.weighted_sum(weight), (result.per_law_losses * weight[[0, 1, 3]]).sum())
        absent = RaggedSinkhornGeometry(options=self.options())(x, logits.log_softmax(1), law_ids=ids, observations=[None] * 4)
        self.assertEqual(absent.eligible_count, 0)
        self.assertIsNone(absent.loss_mean)
        self.assertEqual(absent.per_law_losses.numel(), 0)
        self.assertEqual(absent.loss_sum.item(), 0)
        self.assertTrue(absent.loss_sum.requires_grad)

    def test_training_weight_clamp_reported_without_discarding_rows(self):
        x, logits, ids, observed = self.fixture()
        w = observed[0].weights.clone()
        w[:2] = torch.tensor([0., 1e-40], dtype=w.dtype)
        observed[0] = replace(observed[0], weights=w)
        result = RaggedSinkhornGeometry(options=self.options())(x, logits.log_softmax(1), law_ids=ids, observations=observed)
        self.assertEqual(result.observed_weight_clamp_counts, (2, 0, 0, 0))
        self.assertEqual(result.observation_support_counts[0], 5)
        self.assertEqual(result.dropped_observed_rows, 0)
        self.assertTrue(bool(torch.isfinite(result.per_law_losses).all()))
        observed[0] = replace(observed[0], weights=torch.zeros_like(w))
        with self.assertRaises(ValueError):
            RaggedSinkhornGeometry(options=self.options())(x, logits.log_softmax(1), law_ids=ids, observations=observed)

    def test_backend_pin_metadata_and_no_automatic_version_fallback(self):
        with self.assertRaises(ValueError):
            GeometryOptions(4096, 2, "latest")
        with self.assertRaises(RuntimeError):
            RaggedSinkhornGeometry(options=GeometryOptions(4096, 2, "0.3.1"))
        result = RaggedSinkhornGeometry(options=self.options())
        self.assertEqual(result.backend_identity["blur"], math.sqrt(.1))
        self.assertEqual(result.backend_identity["p"], 2)
        self.assertTrue(result.backend_identity["debias"])

    def test_grad_mode_preserved_on_success_and_backend_failure(self):
        x, logits, ids, observed = self.fixture()
        obj = RaggedSinkhornGeometry(options=self.options())
        with torch.no_grad():
            result = obj(x, logits.log_softmax(1), law_ids=ids, observations=observed)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(result.loss_sum.requires_grad)
        backend = geometry.geomloss_backend()
        def failed(*args, **kwargs):
            torch.set_grad_enabled(False)
            raise RuntimeError("fabricated backend error")
        with patch.object(backend, "sinkhorn_loop", failed):
            with self.assertRaisesRegex(RuntimeError, "fabricated"):
                obj(x, logits.log_softmax(1), law_ids=ids, observations=observed)
        self.assertTrue(torch.is_grad_enabled())

    def test_invalid_identity_non_normalized_and_degenerate_inputs(self):
        x, logits, ids, observed = self.fixture()
        obj = RaggedSinkhornGeometry(options=self.options())
        with self.assertRaises(ValueError):
            obj(x, logits, law_ids=ids, observations=observed)
        with self.assertRaises(ValueError):
            obj(x, logits.log_softmax(1), law_ids=("wrong", *ids[1:]), observations=observed)
        with self.assertRaises(ValueError):
            obj(x, logits.log_softmax(1), law_ids=(ids[0],) * 4, observations=observed)
        zero = torch.zeros_like(x[:1])
        law = ObservedLaw("zero", torch.zeros_like(x[0, :1]), x.new_ones(1))
        with self.assertRaisesRegex(RuntimeError, "Degenerate"):
            obj(zero, logits[:1].log_softmax(1), law_ids=("zero",), observations=(law,))


if __name__ == "__main__":
    unittest.main()
