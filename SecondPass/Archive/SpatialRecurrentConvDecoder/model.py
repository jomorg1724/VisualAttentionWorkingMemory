"""Spatial-only recurrent H queries joint {Z,H}; final H feeds a conv readout.

The KDA encoder and transformer spatial paths retain the CLS predecessor's
computations. No CLS, sensory/readout bypass, phase inputs or temporal detach.
"""
import math

import torch
from torch import nn

from PreAttentiveVision.decoder import ConvNormAct
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline

VERSION = 'spatial_recurrent_conv_decoder_no_cls_v1'


def tokens(field):
    return field.flatten(2).transpose(1, 2)


def grid(sequence):
    return sequence.transpose(1, 2).reshape(sequence.shape[0], 64, 7, 7)


class ConvTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_norm = nn.LayerNorm(64)
        self.visual_norm = nn.LayerNorm(64)
        self.ffn_norm = nn.LayerNorm(64)
        self.position = nn.Parameter(torch.empty(1, 49, 64))
        self.source = nn.Parameter(torch.empty(2, 1, 64))
        self.spatial_q = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.spatial_k = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.spatial_v = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.spatial_out = nn.Conv2d(64, 64, 3, padding=1)
        self.spatial_ffn = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1))
        nn.init.normal_(self.position, std=.02)
        nn.init.normal_(self.source, std=.02)

    def forward(self, hidden, visual):
        h = self.memory_norm(hidden) + self.position + self.source[1]
        z = self.visual_norm(tokens(visual)) + self.position + self.source[0]
        q = tokens(self.spatial_q(grid(h)))
        k = torch.cat((tokens(self.spatial_k(grid(z))),
                       tokens(self.spatial_k(grid(h)))), 1)
        v = torch.cat((tokens(self.spatial_v(grid(z))),
                       tokens(self.spatial_v(grid(h)))), 1)
        b = hidden.shape[0]
        q = q.reshape(b, 49, 2, 32).transpose(1, 2)
        k = k.reshape(b, 98, 2, 32).transpose(1, 2)
        v = v.reshape(b, 98, 2, 32).transpose(1, 2)
        attention = (q @ k.transpose(-1, -2) / math.sqrt(32)).softmax(-1)
        attended = (attention @ v).transpose(1, 2).reshape(b, 49, 64)
        hidden = hidden + tokens(self.spatial_out(grid(attended)))
        return hidden + tokens(self.spatial_ffn(grid(self.ffn_norm(hidden))))


class RecurrentMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_spatial = nn.Parameter(torch.empty(1, 49, 64))
        nn.init.normal_(self.initial_spatial, std=.02)
        self.layers = nn.ModuleList([ConvTransformerBlock(), ConvTransformerBlock()])

    def forward(self, visual, hidden=None):
        if hidden is None:
            hidden = self.initial_spatial.expand(visual.shape[0], -1, -1)
        for layer in self.layers:
            hidden = layer(hidden, visual)
        return hidden


class SpatialReadout(nn.Module):
    """Single-field adaptation of StreamingReadout, not a pair decoder."""
    def __init__(self):
        super().__init__()
        self.local = ConvNormAct(64, 32)
        self.fusion = ConvNormAct(32, 64)
        self.trunk = nn.Sequential(nn.Linear(128, 256), nn.SiLU())

    def forward(self, field):
        field = self.fusion(self.local(field))
        pooled = torch.cat((field.mean((2, 3)), field.amax((2, 3))), 1)
        return self.trunk(pooled)


class SpatialRecurrentConvDecoder(AccumulatorBaseline):
    def __init__(self, task_classes):
        super().__init__(task_classes, stack=3, center=True, accumulator='kda')
        del self.feat, self.gru, self.norm
        self.spatial_input = nn.Conv2d(160, 64, 1)
        self.memory = RecurrentMemory()
        self.readout = SpatialReadout()

    def recurrent_states(self, images):
        images = self.frames(images)
        encoder_states = hidden = None
        history = []
        for t in range(images.shape[1]):
            field, encoder_states = self.encode_frame(images[:, t], encoder_states)
            hidden = self.memory(self.spatial_input(field), hidden)
            history.append(hidden)
        return history

    def classify(self, hidden, task):
        return self.heads[task](self.readout(grid(hidden)))

    def forward(self, images, task):
        return self.classify(self.recurrent_states(images)[-1], task)
