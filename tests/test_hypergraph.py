import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.Hypergraph import Hypergraph, HypergraphCache, temporal_aggregate
from model.module import Module


class HypergraphTest(unittest.TestCase):
    def test_cache_shuffling_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        x = torch.randn(4, 8, 18, 3)
        model = Module('MIT_Skate', device='cpu', num_frame=8, num_center=3).eval()
        cache = HypergraphCache(model.structure, x, batch_size=1)
        order = torch.tensor([3, 0, 2])
        graphs = cache.batch(order, 'cpu')
        torch.testing.assert_close(graphs[0], model.structure.spatial(x[order]))
        for cached, direct in zip(graphs[1], model.structure.temporal(x[order])):
            torch.testing.assert_close(cached.to_dense(), direct.to_dense())
        direct = model(x[order])
        direct.sum().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        model.zero_grad()
        with patch.object(model.structure, 'spatial', side_effect=AssertionError('Rebuilt spatial graph')), \
                patch.object(model.structure, 'temporal', side_effect=AssertionError('Rebuilt temporal graph')):
            cached = model(x[order], hypergraphs=graphs)
            cached.sum().backward()
        torch.testing.assert_close(cached, direct)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name])

    def test_temporal_neighbors(self):
        x = torch.tensor([[[[0.0], [100.0]], [[30.0], [60.0]], [[0.1], [99.9]]]])
        H = Hypergraph(temporal_k=1).temporal(x)[0].to_dense()
        self.assertEqual(tuple(H.shape), (6, 6))
        self.assertEqual(H[4, 0].item(), 1)
        self.assertEqual(H[1, 0].item(), 0)
        self.assertTrue(torch.equal(H.sum(dim=0), torch.full((6,), 2.0)))
        self.assertTrue(torch.equal(H.diag(), torch.ones(6)))
        for column in range(6):
            members = H[:, column].nonzero().flatten()
            others = members[members != column]
            self.assertTrue(torch.all(others // 2 != column // 2))

    def test_sparse_aggregation_and_restriction(self):
        torch.manual_seed(7)
        coordinates = torch.randn(2, 5, 3, 3)
        graphs = Hypergraph(temporal_k=2).temporal(coordinates)
        feature = torch.randn(2, 4, 5, 3, requires_grad=True)
        actual = temporal_aggregate(feature, graphs)
        for sample, H in enumerate(graphs):
            H = H.to_dense()
            scale = H.sum(dim=1).clamp_min(1).rsqrt()
            kernel = scale[:, None] * ((H / H.sum(dim=0)) @ H.T) * scale[None, :]
            expected = kernel @ feature[sample].permute(1, 2, 0).reshape(15, 4)
            torch.testing.assert_close(actual[sample].permute(1, 2, 0).reshape(15, 4), expected)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        frames = torch.tensor([[0, 3], [1, 4]])
        reduced = Hypergraph.restrict(graphs, frames, 3)
        for sample in range(2):
            nodes = (frames[sample, :, None] * 3 + torch.arange(3)).reshape(-1)
            expected = graphs[sample].to_dense()[nodes][:, nodes]
            torch.testing.assert_close(reduced[sample].to_dense(), expected)
        selected = torch.stack([feature[i, :, frames[i], :] for i in range(2)])
        output = temporal_aggregate(selected, reduced)
        self.assertEqual(tuple(output.shape), (2, 4, 2, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_spatial_motion_and_sparsity(self):
        x = torch.zeros(1, 4, 3, 1)
        x[0, :, 0, 0] = torch.arange(4.0)
        x[0, :, 1, 0] = 0.5 * torch.arange(4.0)
        kernel = Hypergraph(num_center=1, sparse_lambda=0.1).spatial(x)
        expected = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]])
        torch.testing.assert_close(kernel, expected)
        isolated = Hypergraph(num_center=1, sparse_lambda=2).spatial(x)
        torch.testing.assert_close(isolated, torch.diag_embed(torch.tensor([[1.0, 0.0, 0.0]])))
        self.assertTrue(torch.isfinite(Hypergraph(num_center=2).spatial(torch.zeros_like(x))).all())

    def test_model_pooling_and_backward(self):
        torch.set_num_threads(2)
        for dataset, joints in [('Kimore_ex2', 25), ('UI_PRMD_ex1', 39), ('MIT_Skate', 18)]:
            with self.subTest(dataset=dataset):
                model = Module(dataset, device='cpu', num_frame=8, num_center=3, use_topk=True)
                seen = []
                handles = [getattr(model, f'l{i}').register_forward_hook(
                    lambda layer, inputs, output: seen.append(output[0].shape[2])) for i in (5, 8)]
                output = model(torch.randn(2, 8, joints, 3))
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertEqual(seen, [4, 2])
                output.square().mean().backward()
                for layer in (model.l5.pool, model.l8.pool):
                    self.assertIsNotNone(layer.fc_tem[0].weight.grad)
                    self.assertTrue(torch.isfinite(layer.fc_tem[0].weight.grad).all())
                self.assertIsNotNone(model.l2.tcn1.conv.weight.grad)
                for handle in handles:
                    handle.remove()


if __name__ == '__main__':
    unittest.main()
