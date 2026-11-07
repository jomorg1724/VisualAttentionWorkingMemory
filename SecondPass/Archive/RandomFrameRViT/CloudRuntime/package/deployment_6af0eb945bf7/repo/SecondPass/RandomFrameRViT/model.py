"""Unbiased sampled-timestep parameter gradients with a differentiable suffix.

Prefix: no gradient graph. Selected step: learned weights receive T times their
local derivative. Suffix: detached weights, differentiable memory, constant visual
features. Decoder: ordinary full-loss gradients. All forward values are unchanged.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.func import functional_call

from SecondPass.GatedMemoryRViT.model import GatedStructuredMotionRViT


class _EncodeFrame(nn.Module):
    """Temporary functional wrapper; never registered inside its own model."""
    def __init__(self, model: GatedStructuredMotionRViT):
        super().__init__()
        self.model = model

    def forward(self, previous: Tensor, current: Tensor, motion: Tensor) -> Tensor:
        return self.model.encode_fused(previous, current, motion)


class RandomFrameRViT(GatedStructuredMotionRViT):
    """Fresh whole motion-energy RViT, with sampled temporal gradient policy."""
    def forward(self, images: Tensor, task: str = 'krauzlis_cued_motion',
                selected_frames: Tensor | list[int] | None = None) -> Tensor:
        if task not in ('krauzlis_cued_motion', 'krauzlis'):
            raise ValueError('native Krauzlis-only candidate')
        self._validate(images, movie=True)
        if not self.training or not torch.is_grad_enabled():
            return super().forward(images, task)

        batch, times = images.shape[:2]
        if selected_frames is None:
            selected_frames = torch.randint(times, (batch,), device='cpu')
        selected = torch.as_tensor(selected_frames, dtype=torch.long, device='cpu')
        if selected.shape != (batch,) or bool(((selected < 0) | (selected >= times)).any()):
            raise ValueError('one selected frame index in [0,time) per movie is required')
        self.last_selected_frames = selected.tolist()

        # These analytic filters have no learned parameters. Their outputs are
        # observations; only the selected frame's learned encoder is differentiated.
        with torch.no_grad():
            motion = self.motion_features(images)

        wrapper = _EncodeFrame(self)
        frozen = {name: p.detach() for name, p in wrapper.named_parameters()}
        recurrence_weights = {name.removeprefix('model.recurrent_block.'): value
                              for name, value in frozen.items()
                              if name.startswith('model.recurrent_block.')}
        # Same FP32 values, but derivative w.r.t. a shared parameter is times.
        # Apply only to the selected frame update, not to the final decoder.
        selected_weights = {name: p.detach() + times * (p - p.detach())
                            for name, p in wrapper.named_parameters()}
        selected_recurrence = {name.removeprefix('model.recurrent_block.'): value
                               for name, value in selected_weights.items()
                               if name.startswith('model.recurrent_block.')}
        rows = torch.arange(batch, device=images.device)
        chosen_on_device = selected.to(images.device)
        # All selected CNN steps share one batched forward/backward, even when
        # their selected indices differ. GroupNorm has no batch-running state.
        selected_tokens = functional_call(wrapper, selected_weights,
            (images[rows, (chosen_on_device-1).clamp_min(0)].detach(),
             images[rows, chosen_on_device], motion[rows, chosen_on_device]))
        memory = selected_tokens.new_zeros((batch, 169, 256))
        for step in range(times):
            active = (selected == step).nonzero().flatten().to(images.device)
            prefix = (selected > step).nonzero().flatten().to(images.device)
            suffix = (selected < step).nonzero().flatten().to(images.device)
            with torch.no_grad():
                tokens = self.encode_fused(images[:, max(step-1, 0)], images[:, step], motion[:, step])
                next_memory = torch.zeros_like(memory)
                if prefix.numel():
                    prior = self.recurrent_block(tokens.index_select(0, prefix), memory.index_select(0, prefix))
                    next_memory.index_copy_(0, prefix, prior)
            if active.numel():
                updated = functional_call(self.recurrent_block, selected_recurrence,
                    (selected_tokens.index_select(0, active), memory.index_select(0, active).detach()))
                next_memory = next_memory.index_copy(0, active, updated)
            if suffix.numel():
                # Detached weights, live memory Jacobian. No suffix detach.
                updated = functional_call(self.recurrent_block, recurrence_weights,
                    (tokens.index_select(0, suffix), memory.index_select(0, suffix)))
                next_memory = next_memory.index_copy(0, suffix, updated)
            memory = next_memory
        return self.decode(memory)
