"""Focused CPU correctness checks for the single-layer 16-head candidate."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch
from torch.nn import functional as F
from SecondPass.SequenceKDA.kda_backend import chunk_kda, recurrent_kda
from SecondPass.SequenceKDA16.model import KDALayer, SequenceKDA, patchify, temporal_features

EVIDENCE = {}


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(102061)

    def test_short_16_head_operator_output_state_and_gradient_parity(self):
        shape = (1, 17, 16, 64)
        inputs = (F.normalize(torch.randn(shape), dim=-1), F.normalize(torch.randn(shape), dim=-1),
                  torch.randn(shape), -torch.rand(shape) * 0.02, torch.rand(1, 17, 16) * 0.1,
                  torch.randn(1, 16, 64, 64) * 0.1)
        reference = tuple(x.clone().requires_grad_() for x in inputs)
        actual = tuple(x.clone().requires_grad_() for x in inputs)
        expected_output, expected_state = recurrent_kda(*reference[:5], initial_state=reference[5])
        output, state = chunk_kda(*actual[:5], initial_state=actual[5], chunk_size=8)
        torch.testing.assert_close(output, expected_output, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(state, expected_state, atol=3e-6, rtol=3e-5)
        output_weights, state_weights = torch.randn_like(output), torch.randn_like(state)
        expected_gradients = torch.autograd.grad((expected_output * output_weights).sum() + (expected_state * state_weights).sum(), reference)
        gradients = torch.autograd.grad((output * output_weights).sum() + (state * state_weights).sum(), actual)
        errors = {}
        for name, got, expected in zip(("q", "k", "v", "g", "beta", "initial_state"), gradients, expected_gradients):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=5e-4)
            errors[name] = (got - expected).abs().max().item()
        EVIDENCE.update(output_max_abs=(output - expected_output).abs().max().item(),
            state_max_abs=(state - expected_state).abs().max().item(), input_gradient_max_abs=errors)

    def test_single_layer_architecture_order_and_all_head_updates(self):
        model = SequenceKDA()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 737170)
        self.assertEqual(len(list(model.parameters())), 27)
        self.assertEqual(sum(isinstance(module, KDALayer) for module in model.modules()), 1)
        self.assertEqual(model.kda.heads, 16)
        self.assertEqual(model.kda.head_dim, 64)
        self.assertEqual(model.patch_projection.out_features, 128)
        self.assertEqual(model.kda.q_proj.out_features, 1024)
        self.assertEqual(model.kda.out_proj.out_features, 128)
        self.assertEqual(model.kda.decay_proj.weight.count_nonzero().item(), 0)
        self.assertEqual(model.kda.write_proj.weight.count_nonzero().item(), 0)
        decay = -F.softplus(model.kda.decay_proj.bias.reshape(16, 64))
        for head in range(16):
            half_life = 32 if head < 8 else 128
            torch.testing.assert_close(decay[head], torch.full((64,), -torch.log(torch.tensor(2.)).item() / (100 * half_life)))
        torch.testing.assert_close(model.kda.write_proj.bias.sigmoid(), torch.full((16,), 0.01))
        frames = torch.rand(1, 2, 3, 100, 100, requires_grad=True)
        patches = patchify(frames - 0.5)
        tokens = model.tokens(frames)
        for frame, row, column in ((0, 0, 0), (0, 3, 7), (1, 9, 9)):
            embedded = model.patch_norm(F.gelu(model.patch_projection(patches[:, frame, row, column])))
            expected = embedded + model.row_embedding[row] + model.column_embedding[column] + temporal_features(torch.tensor(frame))
            torch.testing.assert_close(tokens[:, frame * 100 + row * 10 + column], expected, atol=3e-6, rtol=2e-5)
        self.assertEqual(tuple(tokens.shape), (1, 201, 128))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        logits = model(frames)
        F.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertGreater(frames.grad[:, 0].abs().sum().item(), 0)
        head_gradients = {}
        for name, p in model.named_parameters():
            self.assertTrue(p.requires_grad and p.dtype == torch.float32, name)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        for projection in ("q_proj", "k_proj", "v_proj", "decay_proj"):
            per_head = getattr(model.kda, projection).weight.grad.reshape(16, 64, 128).abs().sum((1, 2))
            self.assertTrue((per_head > 0).all(), projection)
            head_gradients[projection] = per_head.tolist()
        self.assertTrue((model.kda.write_proj.weight.grad.abs().sum(1) > 0).all())
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, before[name]) for name, p in model.named_parameters()))
        with torch.no_grad():
            first = model(frames.detach())
            model(torch.rand_like(frames))
            torch.testing.assert_close(first, model(frames.detach()), atol=0, rtol=0)
        EVIDENCE.update(parameter_count=737170, parameter_tensors=27, heads=16, head_key_value_width=64,
            state_values_per_trial=65536, earliest_frame_abs_gradient=frames.grad[:, 0].abs().sum().item(),
            all_parameters_advanced=True, optimizer_state_count=len(optimizer.state), per_head_gradient_sums=head_gradients)


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
