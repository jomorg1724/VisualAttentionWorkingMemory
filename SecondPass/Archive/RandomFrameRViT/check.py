"""Focused CPU proof of the sampled estimator and suffix gradient bridge."""
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from SecondPass.GatedMemoryRViT.model import GatedStructuredMotionRViT
from SecondPass.RandomFrameRViT.model import RandomFrameRViT
from SecondPass.TwoFrameRViTReplay import worker as replay


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.monotonic()
    torch.manual_seed(29)
    full = GatedStructuredMotionRViT(checkpoint_encoder=False)
    torch.manual_seed(29)
    sampled = RandomFrameRViT(checkpoint_encoder=False)
    for key, tensor in full.state_dict().items():
        assert torch.equal(tensor, sampled.state_dict()[key])
    # Three steps allow exact enumeration of every selected frame cheaply.
    torch.manual_seed(311)
    images, labels = torch.rand(1, 3, 3, 100, 100), torch.tensor([1])
    expected = full(images)
    F.cross_entropy(expected, labels).backward()
    target = {name: p.grad.clone() for name, p in full.named_parameters()}
    average = {name: torch.zeros_like(value) for name, value in target.items()}
    for chosen in range(3):
        sampled.zero_grad(set_to_none=True)
        actual = sampled(images, selected_frames=[chosen])
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        F.cross_entropy(actual, labels).backward()
        for name, p in sampled.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            average[name] += p.grad / 3
    for name in target:
        torch.testing.assert_close(average[name], target[name], atol=2e-6, rtol=2e-4, msg=name)

    # Long native suffix: only frame zero's learned interpretation receives an
    # update, and gradients must traverse all 44 subsequent memory operations.
    native, labels, metadata = replay.base.FreshStream('val').batch(1, replay.TASK, 'B28')
    native.requires_grad_()
    sampled.zero_grad(set_to_none=True)
    states = []
    def retain(module, args, state):
        if state.requires_grad:
            state.retain_grad()
            states.append(state)
    handle = sampled.recurrent_block.register_forward_hook(retain)
    logits = sampled(native, selected_frames=[0])
    F.cross_entropy(logits, labels).backward()
    handle.remove()
    gradients = native.grad.double().flatten(2).norm(dim=2)[0]
    assert gradients[0] > 0 and torch.count_nonzero(gradients[1:]) == 0
    assert len(states) == 45 and all(s.grad is not None and s.grad.count_nonzero() > 0 for s in states)
    memory = [float(s.grad.double().norm()) for s in states]
    all_grads = {n: float(p.grad.double().norm()) for n,p in sampled.named_parameters()}
    assert all(torch.isfinite(torch.tensor(g)) for g in all_grads.values())
    assert all(g > 0 for n,g in all_grads.items() if n.startswith('encoder.'))
    result = dict(complete=True, device='cpu', optimizer_steps=0, trained_weights_loaded=False,
                  exact_three_frame_estimator_matches_full_bptt=True,
                  native_frames=45, selected_frame=0,
                  input_frames_with_nonzero_grad=gradients.nonzero().flatten().tolist(),
                  all_44_suffix_steps_preserve_memory_gradient=True,
                  first_memory_to_final_gradient_ratio=memory[0]/memory[-1],
                  all_parameter_tensors_finite=len(all_grads),
                  zero_gradient_tensors_for_frame_zero=[n for n,g in all_grads.items() if g == 0],
                  seconds=time.monotonic()-started)
    Path(__file__).with_name('check_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
