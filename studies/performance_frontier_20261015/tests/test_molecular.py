import unittest
import numpy as np
from scipy import sparse
import torch

from frontier.molecular import (CellCompositionAccumulator, endpoint_effect_terms,
                                pool_compositions, transform_composition)


class MolecularTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cell_means_are_not_depth_weighted_pseudobulk(self):
        counts = sparse.csr_matrix(np.array([[1, 0], [0, 9], [0, 0]], dtype=np.int32))
        acc = CellCompositionAccumulator(2)
        acc.update(counts)
        result = acc.result()
        np.testing.assert_array_equal(result["composition"], [.5, .5])
        self.assertEqual(result["captured_rows"], 3)
        self.assertEqual(result["positive_RNA_rows"], 2)
        self.assertEqual(result["zero_RNA_rows"], 1)

    def test_streaming_weights_and_missing_support(self):
        counts = sparse.csr_matrix(np.array([[1, 1], [0, 2], [5, 0]], dtype=np.int32))
        full, streamed = CellCompositionAccumulator(2), CellCompositionAccumulator(2)
        full.update(counts, [1, 2, 3])
        streamed.update(counts[:1], [1])
        streamed.update(counts[1:], [2, 3])
        np.testing.assert_allclose(full.result()["composition"], streamed.result()["composition"], atol=1e-15)
        empty = CellCompositionAccumulator(2)
        empty.update(sparse.csr_matrix((2, 2), dtype=np.int32))
        self.assertFalse(empty.result()["available"])
        self.assertIsNone(empty.result()["composition"])

    def test_pool_before_transform(self):
        q = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        w = torch.tensor([.25, .75], dtype=torch.float64)
        pooled = pool_compositions(q, w)
        torch.testing.assert_close(pooled, w)
        self.assertFalse(torch.allclose(transform_composition(pooled), (transform_composition(q) * w[:, None]).sum(0)))
        with self.assertRaises(ValueError):
            pool_compositions(q, w / 2)

    def test_endpoint_interaction_identity_and_one_control(self):
        pt = torch.tensor([[.8, .2], [.3, .7]], dtype=torch.float64, requires_grad=True)
        ot = torch.tensor([[.6, .4], [.1, .9]], dtype=torch.float64)
        pc = torch.tensor([.4, .6], dtype=torch.float64, requires_grad=True)
        oc = torch.tensor([.5, .5], dtype=torch.float64)
        terms = endpoint_effect_terms(pt, ot, pc, oc)
        torch.testing.assert_close(terms["effect_mse_per_law"], terms["target_mse_per_law"] + terms["control_mse"] + terms["interaction_per_law"])
        self.assertEqual(terms["control_endpoint_observations"], 1)
        terms["endpoint_objective"].backward()
        self.assertTrue(bool(torch.isfinite(pt.grad).all()))
        self.assertTrue(bool(torch.isfinite(pc.grad).all()))

    def test_duplicate_target_does_not_duplicate_control_loss(self):
        p, o = torch.tensor([[.8, .2]]), torch.tensor([[.6, .4]])
        pc, oc = torch.tensor([.4, .6]), torch.tensor([.5, .5])
        one, many = endpoint_effect_terms(p, o, pc, oc), endpoint_effect_terms(p.repeat(3, 1), o.repeat(3, 1), pc, oc)
        torch.testing.assert_close(one["endpoint_objective"], many["endpoint_objective"])
        torch.testing.assert_close(one["effect_mse"], many["effect_mse"])

    def test_mismatched_or_unnormalized_axes_rejected(self):
        with self.assertRaises(ValueError):
            transform_composition(torch.tensor([.2, .2]))
        with self.assertRaises(ValueError):
            endpoint_effect_terms(torch.tensor([[1., 0.]]), torch.tensor([[1., 0.]]),
                                  torch.tensor([1.]), torch.tensor([1.]))


if __name__ == "__main__":
    unittest.main()
