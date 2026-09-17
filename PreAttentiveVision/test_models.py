"""Focused encoder integration checks; CPU only, no model training."""
import unittest
import torch
from PreAttentiveVision.models import MODEL_NAMES, build_encoder


class EncoderContract(unittest.TestCase):
    def test_shared_frame_shapes_gradients_and_finiteness(self):
        torch.set_num_threads(2)
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                torch.manual_seed(5701)
                encoder = build_encoder(name)
                first = torch.randn(2, 3, 100, 100, requires_grad=True)
                second = torch.randn(2, 3, 100, 100, requires_grad=True)
                left, right = encoder(first), encoder(second)
                self.assertEqual(encoder.out_channels, (24, 48, 96))
                self.assertLess(encoder.parameter_count, 1_000_000)
                loss = 0
                for a, b, c, size in zip(left, right, (24, 48, 96), (50, 25, 13)):
                    self.assertEqual(tuple(a.shape), (2, c, size, size))
                    self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
                    loss = loss + (a - b).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0)
                self.assertTrue(torch.isfinite(second.grad).all() and second.grad.abs().sum() > 0)
                gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
                self.assertTrue(gradients and all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                with torch.no_grad():
                    repeated = encoder(first.detach())
                    self.assertTrue(all(torch.equal(a.detach(), b) for a, b in zip(left, repeated)))
                    blank = encoder(torch.zeros(1, 3, 100, 100))
                    self.assertTrue(all(torch.isfinite(t).all() for t in blank))
                if name == "vone_resnet":
                    self.assertEqual(len(list(encoder.frontend.parameters())), 0)
                print(name, encoder.parameter_count, "parameters; shape/gradient/determinism OK")


if __name__ == "__main__":
    unittest.main()
