"""One focused CPU check of trained-parent identity and branch-opening gradients."""
from pathlib import Path
import unittest

import torch

from PreAttentiveVision.decoder_multitask import MultitaskPairClassifier
from PreAttentiveVision.hybrid_models import (
    HYBRID_NAMES, NEW_PARAMETER_PREFIXES, NEW_STATE_PREFIXES, build_hybrid,
)
from PreAttentiveVision.models import build_encoder

PARENT = Path(__file__).parent / (
    "runs/multitask_20260912_141316/convnext_grn_seed20271/checkpoint_000756.pt"
)
TASK_CLASSES = dict(motion_direction=4, orientation=2, contrast=2,
                   spatial_frequency=2, chromatic_increment=2, contour=2,
                   natural_spectrum=2)


class HybridContract(unittest.TestCase):
    def test_trained_parent_identity_and_branch_opening(self):
        torch.set_num_threads(2)
        torch.manual_seed(9150912)
        saved = torch.load(PARENT, map_location="cpu")
        self.assertEqual(saved["step"], 756)
        base = MultitaskPairClassifier(build_encoder("convnext_grn"), TASK_CLASSES).eval()
        base.load_state_dict(saved["model"], strict=True)
        images = torch.rand(1, 2, 3, 100, 100)
        frames = (torch.cat((images[:, 0], images[:, 1]), 0) - .5) / .5
        with torch.no_grad():
            original = base.encoder(frames)
            logits = {task: base(images, task) for task in TASK_CLASSES}
        base_parameters = dict(base.encoder.named_parameters())
        for name, added_count in zip(HYBRID_NAMES, (1296, 4728)):
            with self.subTest(name=name):
                model = MultitaskPairClassifier(build_hybrid(name), TASK_CLASSES).eval()
                loaded = model.load_state_dict(saved["model"], strict=False)
                self.assertEqual(loaded.unexpected_keys, [])
                self.assertTrue(loaded.missing_keys)
                self.assertTrue(all(k.startswith(tuple("encoder." + p for p in NEW_STATE_PREFIXES[name]))
                                    for k in loaded.missing_keys))
                encoder = model.encoder
                new = {k: p for k, p in encoder.named_parameters() if k not in base_parameters}
                self.assertEqual(sum(p.numel() for p in new.values()), added_count)
                self.assertTrue(all(k.startswith(NEW_PARAMETER_PREFIXES[name]) for k in new))
                with torch.no_grad():
                    output = encoder(frames)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(output, original)))
                    self.assertTrue(all(torch.equal(model(images, task), logits[task]) for task in TASK_CLASSES))

                altered = encoder.config["altered_output"]
                encoder(frames)[altered].square().mean().backward()
                first = encoder.gabor_alpha if altered == 0 else encoder.se_expand.weight
                self.assertIsNotNone(first.grad)
                self.assertTrue(torch.isfinite(first.grad).all())
                self.assertGreater(first.grad.abs().sum().item(), 0)
                # Projection/reduction can legitimately receive zero gradient at
                # identity. Open the gate, then check these weights can learn.
                encoder.zero_grad(set_to_none=True)
                with torch.no_grad():
                    if altered == 0:
                        encoder.gabor_alpha.fill_(.1)
                    else:
                        encoder.se_expand.weight.normal_(0, .01)
                        encoder.se_expand.bias.fill_(.1)
                output = encoder(frames)
                self.assertFalse(torch.equal(output[altered], original[altered]))
                for i in set(range(3)) - {altered}:
                    self.assertTrue(torch.equal(output[i], original[i]))
                output[altered].square().mean().backward()
                later = encoder.gabor_projection.weight if altered == 0 else encoder.se_reduce.weight
                self.assertTrue(torch.isfinite(later.grad).all())
                self.assertGreater(later.grad.abs().sum().item(), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in encoder.parameters()
                                    if p.grad is not None))
                print(name, encoder.parameter_count, "parameters; trained-parent identity, all seven logits,"
                      " isolated output and branch gradients OK")


if __name__ == "__main__":
    unittest.main()
