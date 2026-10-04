"""Three causal spatial accumulators and a frozen-PAV streaming classifier.

All temporal state is explicit. No module caches a previous frame or feature map.
The decoder receives current projected features and emitted accumulator fields.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from PreAttentiveVision.decoder import ConvNormAct

MODEL_NAMES = ("spatial_kda", "convgru", "opponent")
IMPLEMENTATION_VERSION = "pav_causal_accumulators_v1"


def kda_update(state, q, k, v, alpha, beta):
    """Decay rows, correct current-key prediction, then read updated state.

    Leading dimensions are arbitrary; state ends in [key,value].
    beta ends in [1], alpha/q/k in [key], and v in [value].
    """
    decayed = alpha.unsqueeze(-1) * state
    error = v - (decayed * k.unsqueeze(-1)).sum(-2)
    updated = decayed + beta.unsqueeze(-1) * k.unsqueeze(-1) * error.unsqueeze(-2)
    output = (updated * q.unsqueeze(-1)).sum(-2)
    return output, updated


class SpatialKDA(nn.Module):
    channels = 32
    heads = 2
    key_dim = 8
    value_dim = 16

    def __init__(self):
        super().__init__()
        self.inputs = nn.Conv2d(32, 82, 3, padding=1)
        self.output = nn.Conv2d(32, 32, 1)
        # Each head contains q8,k8,v16,alpha8,beta1. Initial gates are
        # input-independent .9/.5, but gate weights remain trainable.
        with torch.no_grad():
            for head in range(2):
                start = head * 41
                self.inputs.weight[start+32:start+41].zero_()
                self.inputs.bias[start+32:start+40].fill_(math.log(9.))
                self.inputs.bias[start+40].zero_()

    def forward(self, current, state=None):
        batch, _, height, width = current.shape
        packed = self.inputs(current).view(batch, 2, 41, height, width)
        packed = packed.permute(0, 3, 4, 1, 2)
        q = F.normalize(packed[..., :8], dim=-1, eps=1e-6)
        k = F.normalize(packed[..., 8:16], dim=-1, eps=1e-6)
        v = packed[..., 16:32]
        alpha = packed[..., 32:40].sigmoid()
        beta = packed[..., 40:41].sigmoid()
        if state is None:
            state = current.new_zeros(batch, height, width, 2, 8, 16)
        elif tuple(state.shape) != (batch, height, width, 2, 8, 16):
            raise ValueError("KDA state shape does not match current batch/map")
        result, state = kda_update(state, q, k, v, alpha, beta)
        result = result.permute(0, 3, 4, 1, 2).reshape(batch, 32, height, width)
        return self.output(result), state


class SpatialConvGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = nn.Conv2d(64, 64, 3, padding=1)
        self.candidate = nn.Conv2d(64, 32, 3, padding=1)
        nn.init.zeros_(self.gates.bias)
        nn.init.zeros_(self.candidate.bias)

    def forward(self, current, state=None):
        if state is None:
            state = torch.zeros_like(current)
        elif state.shape != current.shape:
            raise ValueError("GRU state shape does not match current batch/map")
        write, reset = self.gates(torch.cat((current, state), 1)).sigmoid().chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat((current, reset * state), 1)))
        state = (1. - write) * state + write * candidate
        return state, state


def quadrature_bank():
    """Four axis/frequency pairs, even then odd; fixed feature-site units."""
    axis = torch.arange(-3., 4.)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    envelope = torch.exp(-(xx.square() + yy.square()) / (2 * 1.5 ** 2))
    kernels = []
    for coordinate in (xx, yy):
        for frequency in (.125, .25):
            for phase in (0., math.pi/2):
                kernel = envelope * torch.cos(2 * math.pi * frequency * coordinate + phase)
                kernel = kernel - kernel.mean()
                kernels.append(kernel / kernel.square().sum().sqrt())
    return torch.stack(kernels)[:, None]


class OpponentAccumulator(nn.Module):
    fast_retention = .25
    slow_retention = .75

    def __init__(self):
        super().__init__()
        self.register_buffer("kernels", quadrature_bank().repeat(32, 1, 1, 1))
        self.output = nn.Conv2d(72, 32, 1)

    def energy_channels(self, fast, slow):
        batch, channels, height, width = fast.shape
        # Matched linear kernels for both traces. The only retained state is
        # fast/slow, not these temporary quadrature responses.
        response = F.conv2d(F.pad(torch.cat((fast, slow), 0), (3,)*4, mode="reflect"),
                            self.kernels, groups=32)
        response = response.view(2*batch, channels, 4, 2, height, width)
        fast_response, slow_response = response.split(batch, 0)
        fe, fo = fast_response.unbind(3)
        se, so = slow_response.unbind(3)
        positive = (fe+so).square() + (fo-se).square()
        negative = (fe-so).square() + (fo+se).square()
        # Channels are axis/frequency-major, with + then - for each pair.
        energies = torch.stack((positive.mean(1), negative.mean(1)), 2)
        energies = energies.flatten(1, 2)
        pooled = F.avg_pool2d(F.pad(energies, (1,)*4, mode="reflect"), 3, stride=1)
        normalized = pooled / (1e-4 + pooled.mean(1, keepdim=True))
        raw_opponent = positive.mean(1) - negative.mean(1)
        return normalized, raw_opponent

    def forward(self, current, state=None):
        if state is None:
            fast = slow = current
        else:
            if len(state) != 2 or any(x.shape != current.shape for x in state):
                raise ValueError("Opponent trace shapes do not match current batch/map")
            fast = .25 * state[0] + .75 * current
            slow = .75 * state[1] + .25 * current
        energies, _ = self.energy_channels(fast, slow)
        features = torch.cat((.5*(fast+slow), fast-slow, energies), 1)
        return self.output(features), (fast, slow)


class StreamingReadout(nn.Module):
    def __init__(self, task_classes, dropout=.1):
        super().__init__()
        self.local = nn.ModuleList([ConvNormAct(64, 32) for _ in range(3)])
        self.fusion = ConvNormAct(96, 64)
        self.trunk = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({name: nn.Linear(128, count)
                                    for name, count in task_classes.items()})

    def encode(self, current, emitted):
        scales = [F.adaptive_avg_pool2d(layer(torch.cat((u, o), 1)), (13, 13))
                  for u, o, layer in zip(current, emitted, self.local)]
        field = self.fusion(torch.cat(scales, 1))
        pooled = torch.cat((field.mean((2, 3)), field.amax((2, 3))), 1)
        return self.trunk(pooled)

    def classify(self, features, task):
        return self.heads[task](features)


class StreamingPAVClassifier(nn.Module):
    """step(frame,state)->(readout_features,new_state); classify separately.

    forward(pair,task) is exactly two causal encoder/state calls followed by
    the same final readout. All state is returned to the caller, never cached.
    """
    def __init__(self, encoder, task_classes, accumulator,
                 common_seed=20271, core_seed=20272):
        super().__init__()
        if accumulator not in MODEL_NAMES:
            raise ValueError(f"Unknown accumulator {accumulator!r}")
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.accumulator_name = accumulator
        self.task_classes = dict(task_classes)
        # fork_rng preserves the caller's RNG; common tensors do not depend
        # on the number/type of random core parameters initialized afterward.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(common_seed)
            self.projections = nn.ModuleList([nn.Conv2d(c, 32, 1)
                                              for c in encoder.out_channels])
            self.readout = StreamingReadout(task_classes)
        constructor = dict(zip(MODEL_NAMES, (SpatialKDA, SpatialConvGRU, OpponentAccumulator)))[accumulator]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(core_seed)
            self.accumulators = nn.ModuleList([constructor() for _ in range(3)])
        self.config = dict(implementation=IMPLEMENTATION_VERSION, accumulator=accumulator,
                           channels=32, encoder_frozen=True, encoder_eval=True,
                           common_seed=common_seed, core_seed=core_seed,
                           input_shape=[3, 100, 100], steps_per_pair=2,
                           state_scalars_per_site={"spatial_kda":256,"convgru":32,"opponent":64}[accumulator])

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @property
    def state_elements_per_example(self):
        return self.config["state_scalars_per_site"] * (50*50 + 25*25 + 13*13)

    def _advance(self, frame, state=None):
        if frame.ndim != 4 or tuple(frame.shape[1:]) != (3, 100, 100):
            raise ValueError("Expected one current RGB frame [B,3,100,100]")
        if state is None:
            state = (None,) * 3
        elif len(state) != 3:
            raise ValueError("Expected one explicit state per encoder scale")
        with torch.no_grad():
            fields = self.encoder((frame-.5)/.5)
        current = [projection(field) for projection, field in zip(self.projections, fields)]
        emitted, updated = [], []
        for core, u, old_state in zip(self.accumulators, current, state):
            output, new_state = core(u, old_state)
            emitted.append(output)
            updated.append(new_state)
        return current, emitted, tuple(updated)

    def step(self, frame, state=None):
        current, emitted, state = self._advance(frame, state)
        return self.readout.encode(current, emitted), state

    def classify(self, features, task):
        return self.readout.classify(features, task)

    def forward(self, images, task, reset_before_second=False):
        if images.ndim != 5 or tuple(images.shape[1:]) != (2, 3, 100, 100):
            raise ValueError("Expected ordered RGB pair [B,2,3,100,100]")
        # First-frame readout is not needed; omitting it saves work and avoids
        # consuming a dropout draw for a prediction outside the task objective.
        _, _, state = self._advance(images[:, 0], None)
        features, _ = self.step(images[:, 1], None if reset_before_second else state)
        return self.classify(features, task)
