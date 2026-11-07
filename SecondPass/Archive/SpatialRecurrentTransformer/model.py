"""Recurrent H queries joint {Z,H}; only persistent CLS reaches the heads.

Two pre-norm convolutional transformer blocks. Spatial Q/K/V and FFNs are
3x3 convolutions; CLS has independent linear projections and FFN, never a
fictitious grid cell. No attention-logit bias, phase inputs, GRU, comparator,
field flattening readout, sensory/head bypass, dropout or temporal detach.
"""
import math
import torch
from torch import nn
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline

VERSION = 'spatial_recurrent_conv_transformer_cls_v1'


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
        self.position = nn.Parameter(torch.empty(1, 50, 64))
        self.source = nn.Parameter(torch.empty(2, 1, 64))
        self.spatial_q = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.spatial_k = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.spatial_v = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.cls_q = nn.Linear(64, 64, bias=False)
        self.cls_k = nn.Linear(64, 64, bias=False)
        self.cls_v = nn.Linear(64, 64, bias=False)
        self.spatial_out = nn.Conv2d(64, 64, 3, padding=1)
        self.cls_out = nn.Linear(64, 64)
        self.spatial_ffn = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 64, 3, padding=1))
        self.cls_ffn = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
        nn.init.normal_(self.position, std=.02)
        nn.init.normal_(self.source, std=.02)

    def forward(self, hidden, visual):
        h = self.memory_norm(hidden) + self.position + self.source[1]
        z = self.visual_norm(tokens(visual)) + self.position[:, 1:] + self.source[0]
        q = torch.cat((self.cls_q(h[:, :1]), tokens(self.spatial_q(grid(h[:, 1:])))), 1)
        k = torch.cat((tokens(self.spatial_k(grid(z))), self.cls_k(h[:, :1]), tokens(self.spatial_k(grid(h[:, 1:])))), 1)
        v = torch.cat((tokens(self.spatial_v(grid(z))), self.cls_v(h[:, :1]), tokens(self.spatial_v(grid(h[:, 1:])))), 1)
        b = hidden.shape[0]
        q = q.reshape(b, 50, 2, 32).transpose(1, 2)
        k = k.reshape(b, 99, 2, 32).transpose(1, 2)
        v = v.reshape(b, 99, 2, 32).transpose(1, 2)
        attention = (q @ k.transpose(-1, -2) / math.sqrt(32)).softmax(-1)
        attended = (attention @ v).transpose(1, 2).reshape(b, 50, 64)
        hidden = hidden + torch.cat((self.cls_out(attended[:, :1]), tokens(self.spatial_out(grid(attended[:, 1:])))), 1)
        normalized = self.ffn_norm(hidden)
        return hidden + torch.cat((self.cls_ffn(normalized[:, :1]), tokens(self.spatial_ffn(grid(normalized[:, 1:])))), 1)


class RecurrentMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_spatial = nn.Parameter(torch.empty(1, 49, 64))
        self.initial_cls = nn.Parameter(torch.empty(1, 1, 64))
        nn.init.normal_(self.initial_spatial, std=.02)
        nn.init.normal_(self.initial_cls, std=.02)
        self.layers = nn.ModuleList([ConvTransformerBlock(), ConvTransformerBlock()])

    def forward(self, visual, hidden=None):
        if hidden is None:
            hidden = torch.cat((self.initial_cls, self.initial_spatial), 1).expand(visual.shape[0], -1, -1)
        for layer in self.layers:
            hidden = layer(hidden, visual)
        return hidden


class SpatialRecurrentTransformer(AccumulatorBaseline):
    def __init__(self, task_classes):
        super().__init__(task_classes, stack=3, center=True, accumulator='kda')
        del self.feat, self.gru, self.norm
        self.spatial_input = nn.Conv2d(160, 64, 1)
        self.memory = RecurrentMemory()
        self.cls_readout = nn.Linear(64, 256)

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
        return self.heads[task](self.cls_readout(hidden[:, 0]).relu())

    def forward(self, images, task):
        return self.classify(self.recurrent_states(images)[-1], task)
