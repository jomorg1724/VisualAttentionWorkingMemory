"""Identity-initialized output additions to the trained compact ConvNeXt encoder.

The inherited frontend/stages retain their original state-dict names. Neither
addition changes the computation feeding the next base stage. See
component_combination_research.md for hypotheses and primary-source rationale.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from PreAttentiveVision.models import (
    BlurDown, ChannelLayerNorm, FixedGaborFrontEnd, SpatialEncoder,
)

HYBRID_NAMES = ["convnext_gabor_residual", "convnext_se_residual"]
NEW_PARAMETER_PREFIXES = {
    "convnext_gabor_residual": (
        "gabor_alpha", "gabor_norm.", "gabor_projection.",
    ),
    "convnext_se_residual": ("se_reduce.", "se_expand."),
}
NEW_STATE_PREFIXES = {
    "convnext_gabor_residual": NEW_PARAMETER_PREFIXES["convnext_gabor_residual"]
    + ("gabor_frontend.", "gabor_blur."),
    "convnext_se_residual": NEW_PARAMETER_PREFIXES["convnext_se_residual"],
}


class GaborResidualEncoder(SpatialEncoder):
    """Add fixed oriented measurements only to the returned 50x50 field."""

    def __init__(self):
        super().__init__("convnext_grn")
        self.name = HYBRID_NAMES[0]
        self.description = "ConvNeXt plus an identity-initialized 50px Gabor output side branch."
        self.gabor_frontend = FixedGaborFrontEnd()
        self.gabor_blur = BlurDown(48)
        self.gabor_norm = ChannelLayerNorm(48)
        # Ordinary nonzero projection weights allow alpha to learn immediately.
        self.gabor_projection = nn.Conv2d(48, 24, 1)
        self.gabor_alpha = nn.Parameter(torch.zeros(1, 24, 1, 1))
        self.config.update(
            name=self.name, parent_architecture="convnext_grn",
            implementation="PAV output hybrid v1", identity_initialization=True,
            altered_output=0, new_trainable_parameters=1296,
            gabor_frequencies_cycles_per_pixel=[0.12, 0.25],
            gabor_channels=48, gabor_kernels_fixed=True,
        )

    def forward(self, x):
        outputs = super().forward(x)
        # The raw-RGB channels returned by the bank are already present in the
        # base path. Retain only its 24 simple and 24 complex features here.
        branch = self.gabor_frontend(x)[:, :48]
        branch = self.gabor_projection(self.gabor_norm(self.gabor_blur(branch)))
        return [outputs[0] + self.gabor_alpha * branch, outputs[1], outputs[2]]


class SEResidualEncoder(SpatialEncoder):
    """Center an SE-inspired channel gate at identity on the 13x13 output."""

    def __init__(self):
        super().__init__("convnext_grn")
        self.name = HYBRID_NAMES[1]
        self.description = "ConvNeXt plus an identity-initialized final-field SE-style gate."
        self.se_reduce = nn.Conv2d(96, 24, 1)
        self.se_expand = nn.Conv2d(24, 96, 1)
        nn.init.zeros_(self.se_expand.weight)
        nn.init.zeros_(self.se_expand.bias)
        self.config.update(
            name=self.name, parent_architecture="convnext_grn",
            implementation="PAV output hybrid v1", identity_initialization=True,
            altered_output=2, new_trainable_parameters=4728,
            se_hidden_channels=24, se_gain_limits=[0.5, 1.5],
        )

    def forward(self, x):
        outputs = super().forward(x)
        summary = outputs[2].mean((2, 3), keepdim=True)
        gate = 1.0 + 0.5 * torch.tanh(self.se_expand(F.relu(self.se_reduce(summary))))
        return [outputs[0], outputs[1], outputs[2] * gate]


def build_hybrid(name: str) -> SpatialEncoder:
    constructors = dict(zip(HYBRID_NAMES, (GaborResidualEncoder, SEResidualEncoder)))
    if name not in constructors:
        raise ValueError(f"Unknown hybrid {name!r}; choose one of {HYBRID_NAMES}")
    return constructors[name]()
