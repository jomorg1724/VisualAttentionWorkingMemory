"""Framewise ViT feedback -> spatial ConvGRU and persistent CLS GRU.

Spatial recurrence follows FinalConvGRU in SecondPass/SpatialReadout/model.py.
The CLS uses the same gate equations with linear maps, not a fictitious grid cell.
"""
import math
import torch
from torch import nn


class SpatialGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gates = nn.Conv2d(2 * width, 2 * width, 3, padding=1)
        self.candidate = nn.Conv2d(2 * width, width, 3, padding=1)
        nn.init.zeros_(self.gates.bias)
        nn.init.zeros_(self.candidate.bias)

    def forward(self, visual, previous):
        write, reset = self.gates(torch.cat((visual, previous), 1)).sigmoid().chunk(2, 1)
        candidate = self.candidate(torch.cat((visual, reset * previous), 1)).tanh()
        return (1 - write) * previous + write * candidate


class CLSGRU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gates = nn.Linear(2 * width, 2 * width)
        self.candidate = nn.Linear(2 * width, width)
        nn.init.zeros_(self.gates.bias)
        nn.init.zeros_(self.candidate.bias)

    def forward(self, visual, previous):
        write, reset = self.gates(torch.cat((visual, previous), -1)).sigmoid().chunk(2, -1)
        candidate = self.candidate(torch.cat((visual, reset * previous), -1)).tanh()
        return (1 - write) * previous + write * candidate


class OutcomeContrastiveRViT(nn.Module):
    def __init__(self, width=128, heads=4, patch_size=10):
        super().__init__()
        self.width, self.patch_size = width, patch_size
        if 100 % patch_size or width % heads:
            raise ValueError('Patches must tile100x100 and width must divide into heads')
        self.grid = 100 // patch_size
        self.patch_embedding = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.visual_cls = nn.Parameter(torch.empty(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, self.grid ** 2 + 1, width))
        nn.init.normal_(self.visual_cls, std=.02)
        nn.init.normal_(self.position, std=.02)
        self.visual_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.visual_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.feedback_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.spatial_gru = SpatialGRU(width)
        self.cls_gru = CLSGRU(width)
        self.readout_norm = nn.LayerNorm(width)
        self.classifier = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def initial_state(self, frame):
        return (frame.new_zeros(len(frame), self.width, self.grid, self.grid),
                frame.new_zeros(len(frame), self.width))

    def step(self, frame, state=None):
        if frame.ndim != 4 or tuple(frame.shape[1:]) != (3, 100, 100):
            raise ValueError('Need raw RGB100x100 frames')
        field, cls = self.initial_state(frame) if state is None else state
        patches = self.patch_embedding(frame - .5).flatten(2).transpose(1, 2)
        current = torch.cat((self.visual_cls.expand(len(frame), -1, -1), patches), 1) + self.position
        previous = torch.cat((cls[:, None], field.flatten(2).transpose(1, 2)), 1)
        q = self.visual_norm(current)
        memory = self.memory_norm(previous)
        visual, _ = self.visual_attention(q, q, q, need_weights=False)
        feedback, _ = self.feedback_attention(q, memory, memory, need_weights=False)
        representation = current + (visual + feedback) / math.sqrt(2.)
        representation = representation + self.ffn(self.ffn_norm(representation))
        next_field = self.spatial_gru(representation[:, 1:].transpose(1, 2).reshape(
            len(frame), self.width, self.grid, self.grid), field)
        next_cls = self.cls_gru(representation[:, 0], cls)
        return next_field, next_cls

    def forward(self, frames, return_history=False):
        if frames.ndim != 5 or frames.shape[1] != 6 or tuple(frames.shape[2:]) != (3, 100, 100):
            raise ValueError('Each trial must contain exactly six raw RGB100x100 frames')
        state = None
        history = []
        for frame in frames.unbind(1):
            state = self.step(frame, state)
            if return_history:
                history.append(state)
        logits = self.classifier(self.readout_norm(state[1]))
        if return_history:
            return logits, state[1], history
        return logits, state[1]
