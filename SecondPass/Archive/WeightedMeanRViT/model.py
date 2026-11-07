"""Fresh causal RGB weighted mean, spatial CNN tokens, and one shared RViT."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from SecondPass.TwoFrameRViT.model import CompressionStage, ResidualBlock, RecurrentVisualBlock
from SecondPass.WeightedMeanConvGRU.model import WeightedMeanConvGRU


class WeightedFieldEncoder(nn.Module):
    """Keep256 channels at all169 positions before recurrent attention."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=1, bias=False),
                                  nn.GroupNorm(8, 32), nn.GELU(),
                                  ResidualBlock(32), ResidualBlock(32))
        self.stage50 = CompressionStage(32, 64)
        self.stage25 = CompressionStage(64, 128)
        self.stage13 = CompressionStage(128, 256)

    def forward(self, image: Tensor) -> Tensor:
        field = self.stage25(self.stage50(self.stem(image - .5)))
        return self.stage13(F.pad(field, (0, 1, 0, 1), mode='replicate'))


class WeightedMeanRViT(nn.Module):
    def __init__(self, checkpoint_encoder: bool = True, checkpoint_recurrent: bool = True):
        super().__init__()
        self.checkpoint_encoder = checkpoint_encoder
        self.checkpoint_recurrent = checkpoint_recurrent
        self.encoder = WeightedFieldEncoder()
        self.row_embedding = nn.Parameter(torch.empty(13, 256))
        self.column_embedding = nn.Parameter(torch.empty(13, 256))
        nn.init.normal_(self.row_embedding, std=.02)
        nn.init.normal_(self.column_embedding, std=.02)
        self.token_norm = nn.LayerNorm(256)
        self.recurrent_block = RecurrentVisualBlock()
        self.readout_norm = nn.LayerNorm(256)
        self.token_readout = nn.Sequential(nn.Linear(256, 16), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(2704, 512), nn.GELU(),
                                        nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))

    @staticmethod
    def _validate(images: Tensor, movie: bool = True) -> None:
        if images.ndim != (5 if movie else 4) or tuple(images.shape[-3:]) != (3, 100, 100):
            raise ValueError('expected native [B,T,3,100,100] movies or [B,3,100,100] frames')
        if images.dtype != torch.float32 or torch.is_autocast_enabled(images.device.type):
            raise ValueError('WeightedMeanRViT requires FP32 and disabled autocast')
        if movie and images.shape[1] < 1:
            raise ValueError('empty movie')

    @staticmethod
    def weighted_frames(images: Tensor) -> Tensor:
        # Reuse the exact fixed causal arithmetic and first-frame identity.
        return WeightedMeanConvGRU.weighted_frames(images)

    def encode(self, image: Tensor) -> Tensor:
        self._validate(image, movie=False)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            return checkpoint(self.encoder, image, use_reentrant=False)
        return self.encoder(image)

    def encode_tokens(self, image: Tensor) -> Tensor:
        tokens = self.encode(image).flatten(2).transpose(1, 2)
        positions = (self.row_embedding[:, None] + self.column_embedding[None, :]).reshape(169, 256)
        return self.token_norm(tokens + positions)

    def update_memory(self, current: Tensor, memory: Tensor | None = None) -> Tensor:
        if memory is None:
            memory = current.new_zeros(current.shape)
        if memory.shape != current.shape or memory.dtype != current.dtype or memory.device != current.device:
            raise ValueError('memory must match current [B,169,256] tokens')
        if self.checkpoint_recurrent and self.training and torch.is_grad_enabled():
            return checkpoint(self.recurrent_block, current, memory, use_reentrant=False)
        return self.recurrent_block(current, memory)

    def decode(self, memory: Tensor) -> Tensor:
        return self.classifier(self.token_readout(self.readout_norm(memory)).flatten(1))

    def forward(self, images: Tensor, task: str = 'krauzlis_cued_motion') -> Tensor:
        if task not in ('krauzlis_cued_motion', 'krauzlis'):
            raise ValueError('WeightedMeanRViT is the single-stimulus Krauzlis candidate')
        self._validate(images)
        weighted = self.weighted_frames(images)
        memory = None
        for image in weighted.unbind(1):
            memory = self.update_memory(self.encode_tokens(image), memory)
        return self.decode(memory)

    def stream_step(self, current: Tensor, previous_frames: Tensor | None = None,
                    memory: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, next two raw frames, next memory without internal state."""
        self._validate(current, movie=False)
        if previous_frames is None:
            previous_frames = current.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        self._validate(previous_frames)
        if previous_frames.shape[:2] != (current.shape[0], 2) or previous_frames.device != current.device:
            raise ValueError('history must contain two prior raw frames [B,2,3,100,100]')
        triplet = torch.cat((previous_frames, current.unsqueeze(1)), dim=1)
        # The final filtered frame uses the actual two previous observations.
        weighted = self.weighted_frames(triplet)[:, -1]
        updated = self.update_memory(self.encode_tokens(weighted), memory)
        return self.decode(updated), triplet[:, 1:], updated
