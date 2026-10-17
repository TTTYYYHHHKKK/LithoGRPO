import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from lithogrpo.flow import advantages, grpo_loss, sde_sample, trajectory_log_ratio
from lithogrpo.model import FlowUNet
from lithogrpo.physics import Lithography
from lithogrpo.metrics import epe_normal_checks, normalized_reward
from shotcount import FastShotCounter


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rectangles_contain_only_foreground_and_cover_all_3x3_masks(self):
        counter = FastShotCounter(device='cpu')
        for values in itertools.product((0, 1), repeat=9):
            mask = np.array(values).reshape(3, 3)
            cover = np.zeros_like(mask)
            for y, x, h, w in counter.prune_rectangles_numba(counter.maximal_rectangles(mask)):
                self.assertTrue(mask[y:y+h, x:x+w].all())
                cover[y:y+h, x:x+w] = 1
            np.testing.assert_array_equal(cover, mask)

    def test_ilp_known_optima_scanline_and_per_pixel(self):
        masks = [(np.zeros((3, 3)), 0), (np.ones((3, 3)), 1),
                 (np.eye(3), 3), (np.array([[0,1,0],[1,1,1],[0,1,0]]), 2),
                 (np.array([[1,0,1],[1,1,1]]), 3)]
        for mask, expected in masks:
            for scan in (False, True):
                self.assertEqual(FastShotCounter(scan_line=scan, device='cpu').run(mask, mask.shape), expected)

    def test_socs_matches_direct_frequency_multiplication_and_has_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'kernels').mkdir()
            (root/'scales').mkdir()
            k = torch.randn(3, 3, 2, dtype=torch.complex64)
            for name in ('focus', 'defocus'):
                torch.save(k, root/'kernels'/f'{name}.pt')
                torch.save(torch.tensor([.4, .6]), root/'scales'/f'{name}.pt')
            sim = Lithography(root, kernel_num=2)
            mask = torch.rand(1, 1, 16, 16, requires_grad=True)
            transfer = torch.zeros(2, 16, 16, dtype=torch.complex64)
            transfer[:, 7:10, 7:10] = k.permute(2, 0, 1)
            fields = torch.fft.ifft2(torch.fft.fft2(mask)*torch.fft.ifftshift(transfer, dim=(-2,-1)))
            expected = (fields.abs().square()*torch.tensor([.4,.6])[None,:,None,None]).sum(1, keepdim=True)
            torch.testing.assert_close(sim.aerial(mask), expected)
            sim(mask)[0].sum().backward()
            self.assertTrue(torch.isfinite(mask.grad).all())
            self.assertGreater(mask.grad.abs().sum(), 0)

    def test_policy_ratio_is_one_then_has_nonzero_gradients(self):
        torch.manual_seed(1)
        model = FlowUNet(8, 32)
        target = torch.zeros(2, 1, 32, 32)
        _, trajectory = sde_sample(model, target, steps=2)
        ratio = trajectory_log_ratio(model, target, trajectory, .1)
        torch.testing.assert_close(ratio, torch.zeros_like(ratio))
        loss = grpo_loss(ratio, torch.tensor([-1., 1.]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
        with torch.no_grad():
            model.output.bias.add_(.00001)
        self.assertTrue((trajectory_log_ratio(model, target, trajectory, .1) != 0).any())

    def test_zero_variance_group_and_zero_baseline(self):
        torch.testing.assert_close(advantages(torch.ones(6), 3), torch.zeros(6))
        reward = normalized_reward({k: torch.ones(2) for k in ('L2','PVB','EPE','Shot')},
                                   {k: 0 for k in ('L2','PVB','EPE','Shot')})
        torch.testing.assert_close(reward, torch.full((2,), -4.))

    def test_reference_counter_simple_component(self):
        from shotcount import ShotCounter
        mask = torch.zeros(16, 16)
        mask[3:10, 4:11] = 1
        self.assertEqual(ShotCounter(device='cpu').run(mask, shape=(16,16)), 1)

    def test_epe_detects_disappearing_and_expanding_shapes(self):
        target = np.zeros((64,64), dtype=bool)
        target[16:48, 16:48] = True
        self.assertEqual(epe_normal_checks(target, target, field_nm=64), 0)
        self.assertGreater(epe_normal_checks(np.zeros_like(target), target, field_nm=64), 0)
        self.assertGreater(epe_normal_checks(np.ones_like(target), target, field_nm=64), 0)


if __name__ == '__main__':
    unittest.main()
