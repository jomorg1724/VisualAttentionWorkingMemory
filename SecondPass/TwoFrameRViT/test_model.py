"""Focused CPU engineering checks for the ordered-pair recurrent transformer."""
import argparse
import json
from pathlib import Path
import time
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from SecondPass.TwoFrameRViT.model import TwoFrameRViT

EVIDENCE = {}


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(102071)

    def test_architecture_pair_order_shapes_and_memory_influence(self):
        model = TwoFrameRViT(checkpoint_encoder=False).eval()
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertTrue(all(module.stride == (1, 1) for module in convolutions))
        self.assertFalse(any(isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)) for module in model.modules()))
        self.assertEqual(sum(isinstance(module, nn.PixelUnshuffle) for module in model.modules()), 3)
        attentions = [module for module in model.modules() if isinstance(module, nn.MultiheadAttention)]
        self.assertEqual(len(attentions), 2)
        self.assertTrue(all(module.num_heads == 8 and module.dropout == 0 for module in attentions))
        self.assertNotEqual(attentions[0].in_proj_weight.data_ptr(), attentions[1].in_proj_weight.data_ptr())
        frames = torch.rand(1, 2, 3, 100, 100)
        seen = []
        handle = model.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        with torch.no_grad():
            model(frames)
        handle.remove()
        torch.testing.assert_close(seen[0], torch.cat((frames[:, 0], frames[:, 0]), dim=1), atol=0, rtol=0)
        torch.testing.assert_close(seen[1], torch.cat((frames[:, 0], frames[:, 1]), dim=1), atol=0, rtol=0)
        with torch.no_grad():
            tokens = model.encode_pair(frames[:, 0], frames[:, 1])
            self.assertEqual(tuple(tokens.shape), (1, 169, 256))
            memory = model.update_memory(tokens)
            features = model.token_readout(model.readout_norm(memory))
            self.assertEqual(tuple(features.shape), (1, 169, 16))
            self.assertEqual(tuple(features.flatten(1).shape), (1, 2704))
            self.assertEqual(tuple(model.decode(memory).shape), (1, 2))
            changed_memory = model.update_memory(tokens, torch.randn_like(tokens))
            self.assertGreater((memory - changed_memory).abs().max().item(), 1e-5)
        EVIDENCE.update(convolution_count=len(convolutions), all_convolution_stride_one=True,
            token_shape=[169, 256], token_readout_shape=[169, 16], flattened_readout=2704,
            independent_attention_modules=2, old_memory_changes_output=True)

    def test_movie_stream_equivalence_order_and_reset(self):
        model = TwoFrameRViT(checkpoint_encoder=False).eval()
        frames = torch.rand(1, 3, 3, 100, 100)
        with torch.no_grad():
            sequence = model(frames)
            previous = memory = None
            for index in range(3):
                output, previous, memory = model.stream_step(frames[:, index], previous, memory)
            torch.testing.assert_close(sequence, output, atol=3e-6, rtol=2e-5)
            self.assertGreater((sequence - model(frames.flip(1))).abs().max().item(), 1e-7)
            model(torch.rand_like(frames))
            torch.testing.assert_close(sequence, model(frames), atol=0, rtol=0)
        EVIDENCE["movie_stream_max_abs"] = (sequence - output).abs().max().item()

    def test_checkpointed_full_sequence_gradients_and_parameter_updates(self):
        model = TwoFrameRViT()
        self.assertTrue(model.checkpoint_encoder)
        frames = torch.rand(1, 2, 3, 100, 100, requires_grad=True)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        logits = model(frames)
        F.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertGreater(frames.grad[:, 0].abs().sum().item(), 0)
        for name, p in model.named_parameters():
            self.assertTrue(p.requires_grad and p.dtype == torch.float32, name)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, before[name]) for name, p in model.named_parameters()))
        EVIDENCE.update(parameter_count=sum(p.numel() for p in model.parameters()),
            parameter_tensors=len(list(model.parameters())), optimizer_state_count=len(optimizer.state),
            earliest_frame_abs_gradient=frames.grad[:, 0].abs().sum().item(), all_parameters_advanced=True,
            checkpointed_full_bptt=True)


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
