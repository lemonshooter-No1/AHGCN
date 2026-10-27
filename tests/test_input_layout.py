import unittest

import torch

from graph.Hypergraph import HypergraphCache
from model.module import Module


class InputLayoutTest(unittest.TestCase):
    def test_joint_coordinate_channels_preserve_frame_order(self):
        torch.set_num_threads(2)
        for dataset, joints in [('Kimore_ex1', 25), ('UI_PRMD_ex1', 39), ('MIT_Skate', 18)]:
            with self.subTest(dataset=dataset):
                model = Module(dataset, device='cpu', num_frame=8).eval()
                x = torch.arange(2 * 8 * joints * 3, dtype=torch.float32).reshape(2, 8, joints, 3) / 100
                with torch.no_grad():
                    model.data_bn.running_mean.copy_(torch.arange(joints * 3) / 10)
                    model.data_bn.running_var.copy_(torch.arange(joints * 3) / 100 + 1)
                captured = []
                handle = model.l1.register_forward_pre_hook(lambda layer, inputs: captured.append(inputs[0].detach().clone()))
                cache = HypergraphCache(model.structure, x)
                output = model(x, hypergraphs=cache.batch(range(2), 'cpu'))
                handle.remove()
                mean = model.data_bn.running_mean.view(1, 1, joints, 3)
                variance = model.data_bn.running_var.view(1, 1, joints, 3)
                expected = ((x - mean) / (variance + model.data_bn.eps).sqrt()).permute(0, 3, 1, 2)
                torch.testing.assert_close(captured[0], expected)
                self.assertEqual(tuple(output.shape), (2, 1))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(model.gru.weight_ih_l0.grad).all())


if __name__ == '__main__':
    unittest.main()
