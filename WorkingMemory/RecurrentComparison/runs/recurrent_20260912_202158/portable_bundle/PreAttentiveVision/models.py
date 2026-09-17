"""Five compact, spatial PAV encoders; paper-inspired adaptations, not replicas.

Each encoder processes one frame and returns a three-resolution feature pyramid.
There is no temporal state, pretrained download, classifier or pooled output.
SE/GRN use spatial summary statistics only to modulate retained spatial features.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ["aa_resnet", "convnext_grn", "inceptionnext", "mobilenet_se", "vone_resnet"]
OUT_CHANNELS = (24, 48, 96)
MODEL_DESCRIPTIONS = {
    "aa_resnet": "Binomial low-pass before decimation; learned residual 3x3 hierarchy.",
    "convnext_grn": "Depthwise 7x7 mixing, channel LayerNorm, expanded GELU MLP and GRN.",
    "inceptionnext": "Parallel identity/square/horizontal/vertical depthwise mixing and residual MLP.",
    "mobilenet_se": "Expanded depthwise inverted residuals, h-swish and squeeze-excitation.",
    "vone_resnet": "Fixed chromatic Gabor/simple-complex features plus learned anti-aliased residual hierarchy.",
}


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels independently at each spatial position."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        return self.norm(x.movedim(1, -1)).movedim(-1, 1)


def group_norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class BlurDown(nn.Module):
    """Fixed separable [1,2,1]/4 filter, then factor-two decimation (ceil size)."""
    def __init__(self, channels):
        super().__init__()
        v = torch.tensor([1., 2., 1.])
        kernel = (v[:, None] * v[None, :]) / 16
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"),
                        self.kernel, stride=2, groups=self.channels)


class Reduction(nn.Sequential):
    def __init__(self, in_channels, out_channels, *, antialias=False, layer_norm=False):
        norm = ChannelLayerNorm if layer_norm else group_norm
        layers = [nn.Conv2d(in_channels, out_channels, 3,
                            stride=1 if antialias else 2, padding=1, bias=False),
                  norm(out_channels), nn.GELU()]
        if antialias:
            layers.append(BlurDown(out_channels))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), group_norm(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), group_norm(channels))

    def forward(self, x):
        return F.silu(x + self.body(x))


class GlobalResponseNorm(nn.Module):
    """GRN: spatial L2 response normalized by mean channel response, with residual."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        response = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        relative = response / (response.mean(dim=1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * relative) + self.beta


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            ChannelLayerNorm(channels), nn.Conv2d(channels, 4 * channels, 1),
            nn.GELU(), GlobalResponseNorm(4 * channels), nn.Conv2d(4 * channels, channels, 1))

    def forward(self, x):
        return x + self.body(x)


class InceptionMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch = channels // 8
        self.splits = (channels - 3 * branch, branch, branch, branch)
        self.square = nn.Conv2d(branch, branch, 3, padding=1, groups=branch)
        self.horizontal = nn.Conv2d(branch, branch, (1, 11), padding=(0, 5), groups=branch)
        self.vertical = nn.Conv2d(branch, branch, (11, 1), padding=(5, 0), groups=branch)

    def forward(self, x):
        identity, square, horizontal, vertical = x.split(self.splits, dim=1)
        return torch.cat((identity, self.square(square), self.horizontal(horizontal), self.vertical(vertical)), dim=1)


class InceptionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(InceptionMix(channels), group_norm(channels),
                                  nn.Conv2d(channels, 4 * channels, 1), nn.GELU(),
                                  nn.Conv2d(4 * channels, channels, 1))
        # Larger than published deep-network 1e-6: short scratch-training adaptation.
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        return x + self.scale * self.body(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels, max(8, channels // 4), 1), nn.ReLU(),
                                  nn.Conv2d(max(8, channels // 4), channels, 1), nn.Hardsigmoid())

    def forward(self, x):
        # Summary controls channel gain; the returned tensor preserves every site.
        return x * self.gate(x.mean((2, 3), keepdim=True))


class MobileBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded = 3 * channels
        self.body = nn.Sequential(
            nn.Conv2d(channels, expanded, 1, bias=False), group_norm(expanded), nn.Hardswish(),
            nn.Conv2d(expanded, expanded, 5, padding=2, groups=expanded, bias=False),
            group_norm(expanded), nn.Hardswish(), SqueezeExcitation(expanded),
            nn.Conv2d(expanded, channels, 1, bias=False), group_norm(channels))

    def forward(self, x):
        return x + self.body(x)


class FixedGaborFrontEnd(nn.Module):
    """Deterministic 9x9 Gabor bank: 3 color axes x 4 orientations x 2 frequencies.

    Fixed kernels, 24 rectified simple responses and 24 quadrature energies.
    A raw RGB route retains DC/color information; all subsequent mixing is learned.
    No noise is added. Uniform parameter grid is not a fit to V1 distributions.
    """
    out_channels = 51

    def __init__(self):
        super().__init__()
        axis = torch.arange(-4, 5, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        color_axes = torch.tensor([[1., 1., 1.], [1., -1., 0.], [-.5, -.5, 1.]])
        color_axes = F.normalize(color_axes, dim=1)
        real, imag = [], []
        for color in color_axes:
            for orientation in range(4):
                theta = orientation * math.pi / 4
                u = xx * math.cos(theta) + yy * math.sin(theta)
                v = -xx * math.sin(theta) + yy * math.cos(theta)
                for frequency, sigma in ((.12, 2.), (.25, 1.5)):
                    envelope = torch.exp(-(u.square() + .5 * v.square()) / (2 * sigma**2))
                    for phase, target in ((0., real), (math.pi / 2, imag)):
                        kernel = envelope * torch.cos(2 * math.pi * frequency * u + phase)
                        kernel = kernel - kernel.mean()
                        kernel = kernel / kernel.square().sum().sqrt()
                        target.append(color[:, None, None] * kernel)
        self.register_buffer("real", torch.stack(real))
        self.register_buffer("imag", torch.stack(imag))

    def forward(self, x):
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
        a = F.conv2d(padded, self.real)
        b = F.conv2d(padded, self.imag)
        energy = torch.sqrt(a.square() + b.square() + 1e-6) - .001
        return torch.cat((F.relu(a), energy, x), dim=1)


class SpatialEncoder(nn.Module):
    out_channels = OUT_CHANNELS

    def __init__(self, name):
        super().__init__()
        block = {"aa_resnet": ResidualBlock, "convnext_grn": ConvNeXtBlock,
                 "inceptionnext": InceptionBlock, "mobilenet_se": MobileBlock,
                 "vone_resnet": ResidualBlock}[name]
        self.name = name
        self.description = MODEL_DESCRIPTIONS[name]
        self.config = dict(name=name, widths=list(OUT_CHANNELS), depths=[1, 2, 2],
                           output_sizes=[50, 25, 13], temporal_state=False,
                           pretrained=False, added_noise=False,
                           implementation="PAV compact adaptation v1")
        self.frontend = FixedGaborFrontEnd() if name == "vone_resnet" else nn.Identity()
        previous = 51 if name == "vone_resnet" else 3
        self.stages = nn.ModuleList()
        for channels, depth in zip(OUT_CHANNELS, (1, 2, 2)):
            self.stages.append(nn.Sequential(
                Reduction(previous, channels, antialias=name in ("aa_resnet", "vone_resnet"),
                          layer_norm=name == "convnext_grn"),
                *(block(channels) for _ in range(depth))))
            previous = channels

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 100, 100):
            raise ValueError("Expected [B,3,100,100] frame tensor")
        x = self.frontend(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def build_encoder(name: str) -> SpatialEncoder:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown encoder {name!r}; choose one of {MODEL_NAMES}")
    return SpatialEncoder(name)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_configs():
    return {name: dict(description=MODEL_DESCRIPTIONS[name], widths=list(OUT_CHANNELS),
                       depths=[1, 2, 2]) for name in MODEL_NAMES}
