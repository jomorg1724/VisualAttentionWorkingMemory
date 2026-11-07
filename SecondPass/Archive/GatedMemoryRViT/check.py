"""Bounded CPU gradient check on fresh paired models and native full movies.

No optimizer steps, trained-parent weights, accelerator use or live-run changes.
"""
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from SecondPass.GatedMemoryRViT.model import GatedStructuredMotionRViT, GatedTwoFrameRViT
from SecondPass.StructuredMotionRViT.model import StructuredMotionRViT
from SecondPass.TwoFrameRViT.model import TwoFrameRViT
from SecondPass.TwoFrameRViTReplay import worker as replay


def measure(model, images, labels):
    images = images.clone().requires_grad_()
    states = []
    def retain(module, args, state):
        state.retain_grad()
        states.append(state)
    handle = model.recurrent_block.register_forward_hook(retain)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    handle.remove()
    # FP64 measurement avoids squaring small FP32 values into underflow.
    raw = images.grad.double().flatten(2).norm(dim=2)[0].tolist()
    memory = [float(state.grad.double().norm()) for state in states]
    grads = {name: float(p.grad.double().norm()) if p.grad is not None else None
             for name, p in model.named_parameters()}
    assert all(g is not None and math.isfinite(g) and g > 0 for g in grads.values())
    result = dict(loss=float(loss.detach()), input_frame_gradient_l2=raw,
                  memory_state_gradient_l2=memory,
                  cue_to_final_ratio=max(raw[:2]) / raw[-1],
                  first_memory_to_final_ratio=memory[0] / memory[-1],
                  learned_parameters=sum(p.numel() for p in model.parameters()),
                  parameter_tensors=len(grads), all_learned_gradients_finite_nonzero=True)
    if hasattr(model.recurrent_block, 'gate'):
        result['gate_gradient_norms'] = {n: g for n, g in grads.items() if '.gate.' in n}
    return result


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    result = dict(device='cpu', trained_weights_loaded=False, optimizer_steps=0,
                  initial_carry=0.98, seed=29, pairs=[])
    for name, original, gated in (
        ('two_frame', TwoFrameRViT, GatedTwoFrameRViT),
        ('structured_motion', StructuredMotionRViT, GatedStructuredMotionRViT),
    ):
        torch.manual_seed(29)
        plain = original(checkpoint_encoder=False)
        torch.manual_seed(29)
        carry = gated(checkpoint_encoder=False)
        shared = carry.state_dict()
        for key, value in plain.state_dict().items():
            new_key = key.replace('recurrent_block.', 'recurrent_block.proposal.', 1)
            assert torch.equal(value, shared[new_key]), key
        for cell in ('B12', 'B28'):
            images, labels, metadata = replay.base.FreshStream('val').batch(1, replay.TASK, cell)
            plain.zero_grad(set_to_none=True)
            carry.zero_grad(set_to_none=True)
            item = dict(architecture=name, condition=cell, frames=images.shape[1],
                        event=metadata[0]['event_type'], shared_fresh_weights_identical=True,
                        original=measure(plain, images, labels),
                        gated=measure(carry, images, labels))
            result['pairs'].append(item)
            print(json.dumps({key:item[key] for key in ('architecture', 'condition', 'frames')} |
                             {'original_memory_ratio':item['original']['first_memory_to_final_ratio'],
                              'gated_memory_ratio':item['gated']['first_memory_to_final_ratio'],
                              'original_cue_ratio':item['original']['cue_to_final_ratio'],
                              'gated_cue_ratio':item['gated']['cue_to_final_ratio']}), flush=True)
    result['seconds'] = time.monotonic() - started
    result['complete'] = True
    Path(__file__).with_name('gradient_check.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
