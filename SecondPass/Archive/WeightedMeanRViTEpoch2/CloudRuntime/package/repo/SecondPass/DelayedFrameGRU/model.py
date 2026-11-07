"""Independent stride-one CNN views of current/previous frames into a GRU."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=dilation, dilation=dilation, stride=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=dilation, dilation=dilation, stride=1, bias=False),
            nn.GroupNorm(8, 32),
        )
        self.activation = nn.GELU()

    def forward(self, images: Tensor) -> Tensor:
        return self.activation(images + self.layers(images))


class FrameEncoder(nn.Module):
    """Full 100x100 field with a learned readout retaining every pixel address."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=1, bias=False),
                                  nn.GroupNorm(8, 32), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock(dilation) for dilation in (1, 1, 2, 4, 1)))
        self.readout = nn.Sequential(nn.Conv2d(32, 4, 1, stride=1), nn.Flatten(),
                                     nn.Linear(40000, 256), nn.LayerNorm(256), nn.GELU())

    def forward(self, images: Tensor) -> Tensor:
        return self.readout(self.blocks(self.stem(images - 0.5)))


class DelayedFrameGRU(nn.Module):
    """Each step consumes [current CNN(frame_t), delayed CNN(frame_(t-1))].

    The two CNNs have independent trainable weights. The initial delayed input
    and GRU state are zero. Full-movie forwards reset state automatically.
    Delayed representations retain their gradient to the preceding raw frame.
    """
    def __init__(self):
        super().__init__()
        self.current_encoder = FrameEncoder()
        self.previous_encoder = FrameEncoder()
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2))

    @staticmethod
    def _validate(images: Tensor, movie: bool) -> None:
        expected_dimensions = 5 if movie else 4
        if images.ndim != expected_dimensions or tuple(images.shape[-3:]) != (3, 100, 100):
            raise ValueError("expected native RGB 100x100 frames, with batch and optional time")
        if images.dtype != torch.float32 or torch.is_autocast_enabled(images.device.type):
            raise ValueError("DelayedFrameGRU requires FP32 input and disabled autocast")
        if movie and images.shape[1] < 1:
            raise ValueError("empty movie")

    def pair_features(self, images: Tensor) -> Tensor:
        """Causal [B,T,512] features, with explicit zero delayed half at t=0."""
        self._validate(images, movie=True)
        batch, times = images.shape[:2]
        current = self.current_encoder(images.reshape(batch * times, 3, 100, 100)).reshape(batch, times, 256)
        zero = current.new_zeros(batch, 1, 256)
        if times == 1:
            previous = zero
        else:
            delayed = self.previous_encoder(images[:, :-1].reshape(batch * (times - 1), 3, 100, 100)).reshape(batch, times - 1, 256)
            previous = torch.cat((zero, delayed), dim=1)
        return torch.cat((current, previous), dim=-1)

    def forward(self, images: Tensor, task: str = "krauzlis_cued_motion") -> Tensor:
        if task not in ("krauzlis_cued_motion", "krauzlis"):
            raise ValueError("DelayedFrameGRU is the native Krauzlis-only experiment")
        output, _ = self.gru(self.pair_features(images))
        return self.classifier(output[:, -1])

    def stream_step(self, frame: Tensor, previous_features: Tensor | None = None,
                    hidden: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Explicit streaming equivalent; caller carries only these two tensors.

        Returns logits for this step, delayed-branch encoding for the next step,
        and standard GRU hidden state. No cache is stored on the model. Nothing
        is detached: streaming training also retains gradients across steps.
        """
        self._validate(frame, movie=False)
        current = self.current_encoder(frame)
        previous = current.new_zeros(current.shape) if previous_features is None else previous_features
        if previous.shape != current.shape or previous.dtype != current.dtype or previous.device != current.device:
            raise ValueError("previous_features must be a compatible [batch,256] encoding")
        pair = torch.cat((current, previous), dim=-1).unsqueeze(1)
        output, next_hidden = self.gru(pair, hidden)
        return self.classifier(output[:, 0]), self.previous_encoder(frame), next_hidden
