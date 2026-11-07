"""Ordered two-frame CNN driving one shared recurrent visual transformer."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, stride=1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, stride=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, field: Tensor) -> Tensor:
        return self.activation(field + self.layers(field))


class CompressionStage(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(nn.PixelUnshuffle(2),
            nn.Conv2d(4 * incoming, outgoing, 3, padding=1, stride=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.GELU(),
            ResidualBlock(outgoing), ResidualBlock(outgoing))

    def forward(self, field: Tensor) -> Tensor:
        return self.layers(field)


class TwoFrameEncoder(nn.Module):
    """Stride-one convolutions and learned space-to-depth compression."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1, stride=1, bias=False),
                                  nn.GroupNorm(8, 32), nn.GELU(),
                                  ResidualBlock(32), ResidualBlock(32))
        self.stage50 = CompressionStage(32, 64)
        self.stage25 = CompressionStage(64, 128)
        self.stage13 = CompressionStage(128, 256)

    def forward(self, pair: Tensor) -> Tensor:
        field = self.stage25(self.stage50(self.stem(pair - 0.5)))
        # The native 25x25 field becomes26x26 by replicating its edge; no crop.
        return self.stage13(F.pad(field, (0, 1, 0, 1), mode="replicate"))


class RecurrentVisualBlock(nn.Module):
    """Current-frame queries use separate visual and prior-memory softmaxes."""
    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(256)
        self.memory_norm = nn.LayerNorm(256)
        self.visual_attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
        self.memory_attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, current: Tensor, memory: Tensor) -> Tensor:
        queries = self.query_norm(current)
        previous = self.memory_norm(memory)
        visual, _ = self.visual_attention(queries, queries, queries, need_weights=False)
        recalled, _ = self.memory_attention(queries, previous, previous, need_weights=False)
        updated = current + (visual + recalled) / math.sqrt(2.)
        return updated + self.ffn(self.ffn_norm(updated))


class TwoFrameRViT(nn.Module):
    def __init__(self, checkpoint_encoder: bool = True):
        super().__init__()
        self.checkpoint_encoder = checkpoint_encoder
        self.encoder = TwoFrameEncoder()
        self.row_embedding = nn.Parameter(torch.empty(13, 256))
        self.column_embedding = nn.Parameter(torch.empty(13, 256))
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.column_embedding, std=0.02)
        self.token_norm = nn.LayerNorm(256)
        self.recurrent_block = RecurrentVisualBlock()
        self.readout_norm = nn.LayerNorm(256)
        self.token_readout = nn.Sequential(nn.Linear(256, 16), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(2704, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))

    @staticmethod
    def _validate(images: Tensor, movie: bool) -> None:
        if images.ndim != (5 if movie else 4) or tuple(images.shape[-3:]) != (3, 100, 100):
            raise ValueError("expected native RGB100x100 frames with batch and optional time")
        if images.dtype != torch.float32 or torch.is_autocast_enabled(images.device.type):
            raise ValueError("TwoFrameRViT requires FP32 and disabled autocast")
        if movie and images.shape[1] < 1:
            raise ValueError("empty movie")

    def encode_pair(self, previous: Tensor, current: Tensor) -> Tensor:
        self._validate(current, movie=False)
        if previous.shape != current.shape or previous.dtype != current.dtype or previous.device != current.device:
            raise ValueError("previous raw frame must match current frame")
        pair = torch.cat((previous, current), dim=1)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            field = checkpoint(self.encoder, pair, use_reentrant=False)
        else:
            field = self.encoder(pair)
        positions = self.row_embedding[:, None] + self.column_embedding[None, :]
        tokens = field.permute(0, 2, 3, 1) + positions
        return self.token_norm(tokens.reshape(current.shape[0], 169, 256))

    def update_memory(self, current: Tensor, memory: Tensor | None = None) -> Tensor:
        if memory is None:
            memory = current.new_zeros(current.shape)
        if memory.shape != current.shape or memory.dtype != current.dtype or memory.device != current.device:
            raise ValueError("memory must match current [batch,169,256] tokens")
        return self.recurrent_block(current, memory)

    def decode(self, memory: Tensor) -> Tensor:
        features = self.token_readout(self.readout_norm(memory))
        return self.classifier(features.flatten(1))

    def forward(self, images: Tensor, task: str = "krauzlis_cued_motion") -> Tensor:
        if task not in ("krauzlis_cued_motion", "krauzlis"):
            raise ValueError("TwoFrameRViT is the native Krauzlis-only experiment")
        self._validate(images, movie=True)
        previous = images[:, 0]
        memory = None
        for step in range(images.shape[1]):
            current = images[:, step]
            memory = self.update_memory(self.encode_pair(previous, current), memory)
            previous = current
        return self.decode(memory)

    def stream_step(self, current: Tensor, previous_frame: Tensor | None = None,
                    memory: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Stateless streaming API; first pair duplicates current, no detaches."""
        previous = current if previous_frame is None else previous_frame
        updated = self.update_memory(self.encode_pair(previous, current), memory)
        return self.decode(updated), current, updated
