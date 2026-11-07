"""Causal decayed RGB mean, shared residual CNN, and one standard GRU."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from SecondPass.TwoFrameRViT.model import CompressionStage, ResidualBlock


class WeightedFrameEncoder(nn.Module):
    """Stride-one convolutions; space-to-depth keeps local pixel information."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            ResidualBlock(32), ResidualBlock(32),
        )
        self.stage50 = CompressionStage(32, 64)
        self.stage25 = CompressionStage(64, 128)
        self.stage13 = CompressionStage(128, 256)
        # Keep all 169 spatial positions, with 16 learned features at each one.
        self.channel_readout = nn.Sequential(nn.Conv2d(256, 16, 1), nn.GELU())

    def forward(self, image: Tensor) -> Tensor:
        field = self.stage25(self.stage50(self.stem(image - 0.5)))
        field = self.stage13(F.pad(field, (0, 1, 0, 1), mode="replicate"))
        return self.channel_readout(field).flatten(1)


class WeightedMeanConvGRU(nn.Module):
    """Every frame contributes to the final sequence decision through full BPTT."""

    def __init__(self, checkpoint_encoder: bool = True):
        super().__init__()
        self.checkpoint_encoder = checkpoint_encoder
        self.encoder = WeightedFrameEncoder()
        self.gru = nn.GRU(input_size=16 * 13 * 13, hidden_size=256,
                          num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(),
                                        nn.Linear(128, 2))

    @staticmethod
    def _validate(images: Tensor, movie: bool = True) -> None:
        if images.ndim != (5 if movie else 4) or tuple(images.shape[-3:]) != (3, 100, 100):
            raise ValueError("expected native [B,T,3,100,100] movies or [B,3,100,100] frames")
        if images.dtype != torch.float32 or torch.is_autocast_enabled(images.device.type):
            raise ValueError("WeightedMeanConvGRU requires FP32 and disabled autocast")
        if movie and images.shape[1] < 1:
            raise ValueError("empty movie")

    @staticmethod
    def weighted_frames(images: Tensor) -> Tensor:
        """0.5 X[t] + 0.4 X[t-1] + 0.1 X[t-2], repeating X[0] at startup."""
        WeightedMeanConvGRU._validate(images)
        if images.shape[1] == 1:
            return images
        previous = torch.cat((images[:, :1], images[:, :-1]), dim=1)
        older = torch.cat((images[:, :1].expand(-1, 2, -1, -1, -1),
                           images[:, :-2]), dim=1)
        blended = 0.5 * images + 0.4 * previous + 0.1 * older
        # Exact identity for the first frame; algebraically all three terms are X[0].
        return torch.cat((images[:, :1], blended[:, 1:]), dim=1)

    def encode(self, image: Tensor) -> Tensor:
        self._validate(image, movie=False)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, image, use_reentrant=False)
        return self.encoder(image)

    def forward(self, images: Tensor, task: str = "krauzlis_cued_motion") -> Tensor:
        if task not in ("krauzlis_cued_motion", "krauzlis"):
            raise ValueError("this experiment is single-stimulus Krauzlis only")
        blended = self.weighted_frames(images)
        # Encode one timestep at a time to bound CNN activation memory. No detaches.
        features = torch.stack([self.encode(blended[:, t])
                                for t in range(blended.shape[1])], dim=1)
        _, final_state = self.gru(features)
        return self.classifier(final_state[-1])
