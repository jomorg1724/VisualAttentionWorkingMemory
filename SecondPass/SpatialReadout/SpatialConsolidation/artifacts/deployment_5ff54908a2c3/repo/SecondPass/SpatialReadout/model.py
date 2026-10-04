"""Preserve the three-scale KDA encoder; compress only AFTER final recurrence.

Conv/linear weights use ordinary PyTorch reset_parameters (Kaiming-uniform
with a=sqrt(5)); new projection/linear biases use its default fan-in uniform.
Both recurrent convolution biases are explicitly zero. There is no pooling,
extra normalization, feedback, phase gating or temporal detach.
"""
import torch
from torch import nn
from WorkingMemory.PlainBaseline.accum import AccumulatorBaseline


class FinalConvGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = nn.Conv2d(128,128,3,padding=1)
        self.candidate = nn.Conv2d(128,64,3,padding=1)
        nn.init.zeros_(self.gates.bias)
        nn.init.zeros_(self.candidate.bias)

    def forward(self, x, previous=None):
        if previous is None:
            previous = torch.zeros_like(x)
        write, reset = self.gates(torch.cat((x,previous),1)).sigmoid().chunk(2,1)
        candidate = self.candidate(torch.cat((x,reset*previous),1)).tanh()
        return (1-write)*previous + write*candidate


class SpatialReadout(AccumulatorBaseline):
    def __init__(self, task_classes):
        super().__init__(task_classes, stack=3, center=True, accumulator='kda')
        # Keep inherited names for exact name+shape migration of CNN/KDA/heads.
        del self.feat, self.gru, self.norm
        self.spatial_input = nn.Conv2d(160,64,1)
        self.spatial_gru = FinalConvGRU()
        self.readout = nn.Linear(64*7*7,256)

    def recurrent_states(self, images):
        images = self.frames(images)
        encoder_states = None
        hidden = None
        history = []
        for t in range(images.shape[1]):
            field, encoder_states = self.encode_frame(images[:,t],encoder_states)
            hidden = self.spatial_gru(self.spatial_input(field), hidden)
            history.append(hidden)
        return history

    def forward(self, images, task):
        hidden = self.recurrent_states(images)[-1]
        return self.heads[task](self.readout(hidden.flatten(1)).relu())
