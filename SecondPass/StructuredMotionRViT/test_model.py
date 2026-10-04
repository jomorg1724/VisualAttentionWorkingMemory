"""Focused authors-frontend integration checks; no accelerator or training run."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch
from torch.nn import functional as F
from SecondPass.StructuredMotionRViT.model import SimoncelliHeegerFrontend, StructuredMotionRViT
from SecondPass.StructuredMotionRViT.upstream.simoncelli_heeger import SimoncelliHeegerModel

EVIDENCE = {}


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(102081)

    def test_author_coefficients_layout_and_causal_prefix(self):
        frontend = SimoncelliHeegerFrontend()
        original = SimoncelliHeegerModel()
        for name in ("conv_t", "conv_x", "conv_y"):
            expected = getattr(original.v1_linear, name).weight.permute(0, 1, 4, 2, 3)
            torch.testing.assert_close(getattr(frontend.motion_cnn.v1_linear, name).weight, expected, atol=0, rtol=0)
        self.assertEqual(len(list(frontend.parameters())), 0)
        images = torch.rand(1, 3, 3, 100, 100)
        with torch.no_grad():
            complete = frontend(images)
            self.assertEqual(tuple(complete.shape), (1, 3, 95, 100, 100))
            history = None
            streaming = []
            for index in range(3):
                feature, history = frontend.stream_step(images[:, index], history)
                streaming.append(feature)
            torch.testing.assert_close(complete, torch.stack(streaming, dim=1), atol=2e-5, rtol=2e-4)
            changed = images.clone()
            changed[:, 2] = 1 - changed[:, 2]
            torch.testing.assert_close(complete[:, :2], frontend(changed)[:, :2], atol=2e-5, rtol=2e-4)
        EVIDENCE.update(motion_channels=95, published_scales=5,
            analytic_coefficients_unchanged=True, temporal_layout_corrected=True,
            motion_frontend_parameter_tensors=0, causal_streaming_verified=True)

    def test_directional_discrimination_for_reversed_grating(self):
        frontend = SimoncelliHeegerFrontend()
        columns = torch.arange(100).float()[None, None, None, None, :]
        times = torch.arange(9).float()[None, :, None, None, None]
        plus = (0.5 + 0.4 * torch.cos(2 * torch.pi * (columns - 0.375 * times) / 8)).expand(1, 9, 3, 100, 100)
        minus = (0.5 + 0.4 * torch.cos(2 * torch.pi * (columns + 0.375 * times) / 8)).expand_as(plus)
        with torch.no_grad():
            positive = frontend(plus)[:, -1].mean((-1, -2))[0]
            negative = frontend(minus)[:, -1].mean((-1, -2))[0]
        self.assertTrue(torch.isfinite(positive).all() and torch.isfinite(negative).all())
        difference = (positive - negative).abs().max().item()
        self.assertGreater(difference, 1e-4)
        EVIDENCE.update(native_speed_grating_pixels_per_frame=0.375,
            reversed_direction_feature_max_abs=difference,
            positive_direction_population=positive.tolist(), negative_direction_population=negative.tolist())

    def test_native29_forward_and_short_full_gradients(self):
        model = StructuredMotionRViT(checkpoint_encoder=False)
        from SecondPass.SequenceKDA16.worker import FreshStream, TASK
        images, _, _ = FreshStream("val").batch(1, TASK, "B12")
        self.assertEqual(images.shape[1], 29)
        with torch.no_grad():
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertTrue(torch.isfinite(logits).all())
        short = images[:, :2].clone().requires_grad_()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(short), torch.tensor([1])).backward()
        for name, parameter in model.named_parameters():
            self.assertTrue(parameter.requires_grad, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(short.grad[:, 0].abs().sum().item(), 0)
        EVIDENCE.update(learned_parameter_count=sum(p.numel() for p in model.parameters()),
            learned_parameter_tensors=len(list(model.parameters())),
            analytic_buffer_values=sum(b.numel() for b in model.motion_frontend.buffers()),
            native29_forward_finite=True, all_learned_gradients_finite=True,
            earliest_frame_abs_gradient=short.grad[:, 0].abs().sum().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
            device="cpu", torch_version=torch.__version__, threads=torch.get_num_threads(),
            seconds=time.monotonic() - started, evidence=EVIDENCE), indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
