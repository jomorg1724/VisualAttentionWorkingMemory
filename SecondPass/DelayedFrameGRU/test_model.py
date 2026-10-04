"""Focused CPU checks for architecture, causal delay and full gradients."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from SecondPass.DelayedFrameGRU.model import DelayedFrameGRU

EVIDENCE = {}


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(102041)

    def test_stride_one_independent_weights_and_causal_pair(self):
        model = DelayedFrameGRU()
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertEqual(len(convolutions), 24)
        self.assertTrue(all(module.stride == (1, 1) for module in convolutions))
        self.assertFalse(any(isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d))
                             for module in model.modules()))
        self.assertTrue(all(current.data_ptr() != previous.data_ptr() for current, previous in
                            zip(model.current_encoder.parameters(), model.previous_encoder.parameters())))
        self.assertIsInstance(model.gru, nn.GRU)
        frames = torch.rand(1, 3, 3, 100, 100, requires_grad=True)
        features = model.pair_features(frames)
        self.assertEqual(tuple(features.shape), (1, 3, 512))
        self.assertEqual(features[:, 0, 256:].count_nonzero().item(), 0)
        expected = model.previous_encoder(frames[:, :-1].reshape(2, 3, 100, 100)).reshape(1, 2, 256)
        torch.testing.assert_close(features[:, 1:, 256:], expected, atol=0, rtol=0)
        delayed_grad = torch.autograd.grad(features[:, 1, 256:].sum(), frames)[0]
        self.assertGreater(delayed_grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(delayed_grad[:, 1:].count_nonzero().item(), 0)
        with torch.no_grad():
            changed = frames.detach().clone()
            changed[:, 2] = 1 - changed[:, 2]
            torch.testing.assert_close(features[:, :2], model.pair_features(changed)[:, :2], atol=0, rtol=0)
        EVIDENCE.update(convolution_count=len(convolutions), all_conv_stride_one=True,
                        delayed_previous_frame_abs_gradient=delayed_grad[:, 0].abs().sum().item())

    def test_movie_stream_parity_and_state_reset(self):
        model = DelayedFrameGRU().eval()
        frames = torch.rand(1, 3, 3, 100, 100)
        with torch.no_grad():
            movie = model(frames)
            previous = hidden = None
            for step in range(frames.shape[1]):
                logits, previous, hidden = model.stream_step(frames[:, step], previous, hidden)
            torch.testing.assert_close(logits, movie, atol=3e-6, rtol=2e-5)
            self.assertGreater((movie - model(frames.flip(1))).abs().max().item(), 1e-7)
            model(torch.rand_like(frames))
            torch.testing.assert_close(movie, model(frames), atol=0, rtol=0)
        EVIDENCE["movie_stream_max_abs"] = (movie - logits).abs().max().item()

    def test_fresh_all_parameter_gradient_and_update(self):
        model = DelayedFrameGRU()
        frames = torch.rand(1, 2, 3, 100, 100, requires_grad=True)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        logits = model(frames)
        F.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertGreater(frames.grad[:, 0].abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertTrue(parameter.requires_grad and parameter.dtype == torch.float32, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        changed = {name: (parameter - before[name]).abs().max().item() for name, parameter in model.named_parameters()}
        self.assertTrue(all(change > 0 for change in changed.values()))
        EVIDENCE.update(parameter_count=sum(p.numel() for p in model.parameters()),
            parameter_tensors=len(list(model.parameters())), optimizer_state_count=len(optimizer.state),
            earliest_frame_abs_gradient=frames.grad[:, 0].abs().sum().item(), all_parameters_advanced=True)


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
