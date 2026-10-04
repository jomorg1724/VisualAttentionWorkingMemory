"""CPU-only scientific operator/architecture checks (at most two threads).

Run: python -m SecondPass.SequenceKDA.test_model --receipt /path/receipt.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from SecondPass.SequenceKDA.kda_backend import chunk_kda, recurrent_kda
from SecondPass.SequenceKDA.model import KDALayer, SequenceKDA, patchify, temporal_features

EVIDENCE = {}


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(102021)

    def parity(self, length, keys, values, chunk_size, final_only=False, strong=False):
        shape = (1, length, 2, keys)
        q = F.normalize(torch.randn(shape), dim=-1)
        k = F.normalize(torch.randn(shape), dim=-1)
        v = torch.randn(1, length, 2, values)
        g = -torch.rand(shape) * (1000 if strong else (0.15 if length < 100 else 0.0004))
        beta = torch.rand(1, length, 2) * (0.5 if length < 100 else 0.02)
        state = torch.randn(1, 2, keys, values) * 0.1
        inputs = (q, k, v, g, beta, state)
        reference_inputs = tuple(x.clone().requires_grad_() for x in inputs)
        actual_inputs = tuple(x.clone().requires_grad_() for x in inputs)
        expected, es = recurrent_kda(*reference_inputs[:5], initial_state=reference_inputs[5], final_only=final_only)
        actual, ac = chunk_kda(*actual_inputs[:5], initial_state=actual_inputs[5], chunk_size=chunk_size, final_only=final_only)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(ac.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-4)
        torch.testing.assert_close(ac, es, atol=3e-5, rtol=2e-4)
        output_weight, state_weight = torch.randn_like(actual), torch.randn_like(ac)
        expected_grad = torch.autograd.grad((expected * output_weight).sum() + (es * state_weight).sum(), reference_inputs)
        actual_grad = torch.autograd.grad((actual * output_weight).sum() + (ac * state_weight).sum(), actual_inputs)
        if final_only:
            self.assertEqual(actual_grad[0][:, :-1].count_nonzero().item(), 0)
            self.assertGreater(actual_grad[0][:, -1].abs().sum().item(), 0)
        errors = {}
        for name, got, want in zip(("q", "k", "v", "g", "beta", "initial_state"), actual_grad, expected_grad):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, want, atol=5e-5, rtol=4e-4)
            errors[name] = (got - want).abs().max().item()
        EVIDENCE[f"parity_L{length}_K{keys}_C{chunk_size}_final{final_only}_strong{strong}"] = {
            "output_max_abs": (actual - expected).abs().max().item(),
            "state_max_abs": (ac - es).abs().max().item(), "gradient_max_abs": errors}

    def test_short_all_read_and_boundary_parity(self):
        for chunk_size in (1, 5, 8, 16):
            self.parity(23, 7, 11, chunk_size)
        self.parity(23, 7, 11, 8, final_only=True)

    def test_native_4501_token_full_width_parity(self):
        self.parity(4501, 64, 64, 32, final_only=True)

    def test_strong_decays_do_not_overflow(self):
        self.parity(23, 7, 11, 8, strong=True)

    def test_patch_coverage_order_and_reconstruction(self):
        movie = torch.arange(2 * 3 * 100 * 100, dtype=torch.float32).reshape(1, 2, 3, 100, 100)
        patches = patchify(movie)
        recovered = patches.reshape(1, 2, 10, 10, 3, 10, 10).permute(0, 1, 4, 2, 5, 3, 6).reshape_as(movie)
        self.assertTrue(torch.equal(movie, recovered))
        for t, row, column in ((0, 0, 0), (0, 3, 7), (1, 9, 9)):
            self.assertTrue(torch.equal(patches[0, t, row, column], movie[0, t, :, row * 10:(row + 1) * 10, column * 10:(column + 1) * 10].flatten()))

    def test_architecture_initialization_and_time(self):
        model = SequenceKDA()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 158340)
        self.assertEqual(sum(isinstance(m, KDALayer) for m in model.modules()), 1)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()))
        for head, half_life in enumerate((32, 128)):
            got = -F.softplus(model.kda.decay_proj.bias[head * 64:(head + 1) * 64])
            torch.testing.assert_close(got, torch.full((64,), -math.log(2) / (100 * half_life)))
        torch.testing.assert_close(model.kda.write_proj.bias.sigmoid(), torch.full((2,), 0.01))
        self.assertEqual(model.kda.decay_proj.weight.count_nonzero().item(), 0)
        self.assertEqual(model.kda.write_proj.weight.count_nonzero().item(), 0)
        torch.testing.assert_close(temporal_features(torch.arange(29)), temporal_features(torch.arange(45))[:29], rtol=0, atol=0)
        self.assertEqual(model.tokens(torch.zeros(1, 2, 3, 100, 100)).shape, (1, 201, 128))

    def test_native_forward_optimizer_early_gradient_order_and_reset(self):
        model = SequenceKDA()
        movie = torch.rand(1, 45, 3, 100, 100, requires_grad=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        started = time.monotonic()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        logits = model(movie)
        F.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertGreater(movie.grad[:, 0].abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        changed = {name: (parameter - before[name]).abs().max().item() for name, parameter in model.named_parameters()}
        self.assertTrue(all(value > 0 for value in changed.values()))
        EVIDENCE["native45_optimizer"] = {"seconds_two_cpu_threads": time.monotonic() - started,
            "earliest_frame_abs_gradient": movie.grad[:, 0].abs().sum().item(),
            "parameter_updates": changed, "optimizer_state_count": len(optimizer.state)}
        with torch.no_grad():
            first = model(movie.detach())
            reversed_output = model(movie.detach().flip(1))
            self.assertGreater((first - reversed_output).abs().max().item(), 1e-7)
            model(torch.rand_like(movie.detach()))
            torch.testing.assert_close(first, model(movie.detach()), rtol=0, atol=0)

    def test_fp32_precision_contract(self):
        q = torch.randn(1, 3, 2, 4)
        beta = torch.ones(1, 3, 2) * 0.01
        with self.assertRaises(ValueError):
            chunk_kda(q.double(), q.double(), q.double(), q.double(), beta.double())
        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaises(ValueError):
            chunk_kda(q, q, q, q, beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps({"passed": result.wasSuccessful(), "tests": result.testsRun,
            "torch_version": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
            "seconds": time.monotonic() - started, "evidence": EVIDENCE}, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
