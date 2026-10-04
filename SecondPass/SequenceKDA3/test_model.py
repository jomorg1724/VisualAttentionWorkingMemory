"""Focused CPU engineering checks for the fresh three-layer stack."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch
from torch.nn import functional as F
from SecondPass.SequenceKDA3.model import KDALayer, SequenceKDA

EVIDENCE = {}


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(102031)

    def test_three_layers_full_gradient_and_optimizer(self):
        model = SequenceKDA()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 324488)
        self.assertEqual(len(list(model.parameters())), 55)
        self.assertEqual(sum(isinstance(module, KDALayer) for module in model.modules()), 3)
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()))
        frames = torch.rand(1, 3, 3, 100, 100, requires_grad=True)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        outputs = []
        handles = [layer.register_forward_hook(lambda module, args, result: outputs.append(list(result.shape))) for layer in model.kda_layers]
        logits = model(frames, "krauzlis_cued_motion")
        for handle in handles:
            handle.remove()
        self.assertEqual(outputs, [[1, 301, 128], [1, 301, 128], [1, 128]])
        self.assertEqual(tuple(logits.shape), (1, 2))
        F.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertGreater(frames.grad[:, 0].abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        changes = {name: (p - before[name]).abs().max().item() for name, p in model.named_parameters()}
        self.assertTrue(all(change > 0 for change in changes.values()))
        with torch.no_grad():
            first = model(frames.detach())
            model(torch.rand_like(frames))
            torch.testing.assert_close(first, model(frames.detach()), rtol=0, atol=0)
        EVIDENCE.update(parameter_count=324488, parameter_tensors=55, layer_output_shapes=outputs,
            earliest_frame_abs_gradient=frames.grad[:, 0].abs().sum().item(),
            parameter_updates=changes, optimizer_state_count=len(optimizer.state))

    def test_stack_chunk_reference_parity(self):
        chunk = SequenceKDA()
        reference = SequenceKDA(backend="reference")
        reference.load_state_dict(chunk.state_dict())
        frames = torch.rand(1, 2, 3, 100, 100)
        expected, actual = reference(frames), chunk(frames)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
        expected_grads = torch.autograd.grad(expected.sum(), tuple(reference.parameters()))
        actual_grads = torch.autograd.grad(actual.sum(), tuple(chunk.parameters()))
        maximum = 0.
        for actual_gradient, expected_gradient in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=5e-6, rtol=5e-4)
            maximum = max(maximum, (actual_gradient - expected_gradient).abs().max().item())
        EVIDENCE.update(stack_output_max_abs=(actual - expected).abs().max().item(),
            stack_parameter_gradient_max_abs=maximum)


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
