"""Authors' analytic Simoncelli–Heeger CNN with a fresh task-specific RViT.

Temporal windows end at the current frame. Appearance/cue pixels accompany the
published five-scale MT population; no pretrained segmentation weights are used.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from SecondPass.TwoFrameRViT.model import TwoFrameEncoder, TwoFrameRViT
from .author_cnn import ImagePyramid, SimoncelliHeegerCNN


def analytic_parameters_to_buffers(module: nn.Module) -> None:
    """Fixed published filter coefficients are math buffers, not frozen learners."""
    for child in module.modules():
        for name, parameter in list(child.named_parameters(recurse=False)):
            del child._parameters[name]
            child.register_buffer(name, parameter.detach().clone())


class SimoncelliHeegerFrontend(nn.Module):
    channels = 95
    history_frames = 8

    def __init__(self):
        super().__init__()
        self.pyramid = ImagePyramid([0, 1, 2, 3, 4])
        self.motion_cnn = SimoncelliHeegerCNN.from_original_model(padding="same", padding_time="valid")
        # Upstream original-model weights use (H,W,T), while its CNN consumes
        # (T,H,W). Correct only axis layout, preserving every coefficient/sign.
        for name in ("conv_t", "conv_y", "conv_x"):
            convolution = getattr(self.motion_cnn.v1_linear, name)
            convolution.weight = nn.Parameter(convolution.weight.detach().permute(0, 1, 4, 2, 3).contiguous())
        assert tuple(self.motion_cnn.v1_linear.conv_t.weight.shape) == (10, 1, 9, 1, 1)
        assert tuple(self.motion_cnn.v1_linear.conv_y.weight.shape) == (10, 1, 1, 9, 1)
        assert tuple(self.motion_cnn.v1_linear.conv_x.weight.shape) == (10, 1, 1, 1, 9)
        analytic_parameters_to_buffers(self.pyramid)
        analytic_parameters_to_buffers(self.motion_cnn)

    def _window_features(self, grayscale: Tensor) -> Tensor:
        """[B,1,T>=9,H,W] -> [B,T-8,95,H,W], author operation sequence."""
        batch, _, times, height, width = grayscale.shape
        scales = self.pyramid(grayscale)
        features = []
        for scale in scales:
            energy = self.motion_cnn(scale)
            length = energy.shape[2]
            spatial = energy.permute(0, 2, 1, 3, 4).reshape(batch * length, 19, *energy.shape[-2:])
            spatial = F.interpolate(spatial, size=(height, width), mode="bilinear", align_corners=False)
            features.append(spatial.reshape(batch, length, 19, height, width))
        result = torch.cat(features, dim=2)
        if result.shape != (batch, times - 8, 95, height, width):
            raise RuntimeError("unexpected Simoncelli–Heeger temporal/spatial shape")
        return result

    def forward(self, images: Tensor) -> Tensor:
        """Complete movie, causal outputs; replicate the first frame at startup."""
        if images.ndim != 5 or images.shape[2] != 3 or images.dtype != torch.float32:
            raise ValueError("expected FP32 [batch,time,3,height,width] movie")
        gray = images.mean(dim=2).unsqueeze(1)
        history = gray[:, :, :1].expand(-1, -1, 8, -1, -1)
        return self._window_features(torch.cat((history, gray), dim=2))

    def stream_step(self, current: Tensor, history: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return current95-channel field and eight past gray frames, no detach."""
        gray = current.mean(dim=1, keepdim=True).unsqueeze(2)
        if history is None:
            history = gray.expand(-1, -1, 8, -1, -1)
        expected = (current.shape[0], 1, 8, *current.shape[-2:])
        if history.shape != expected or history.device != current.device or history.dtype != current.dtype:
            raise ValueError("history must contain eight compatible grayscale frames")
        window = torch.cat((history, gray), dim=2)
        return self._window_features(window)[:, 0], window[:, :, 1:]


class MotionFusionEncoder(TwoFrameEncoder):
    def __init__(self):
        super().__init__()
        self.stem[0] = nn.Conv2d(101, 32, 3, padding=1, stride=1, bias=False)

    def forward(self, fused: Tensor) -> Tensor:
        # Appearance is already centered; normalized motion must not be shifted.
        field = self.stage25(self.stage50(self.stem(fused)))
        return self.stage13(F.pad(field, (0, 1, 0, 1), mode="replicate"))


class StructuredMotionRViT(TwoFrameRViT):
    def __init__(self, checkpoint_encoder: bool = True):
        super().__init__(checkpoint_encoder=checkpoint_encoder)
        self.motion_frontend = SimoncelliHeegerFrontend()
        self.encoder = MotionFusionEncoder()

    def motion_features(self, images: Tensor) -> Tensor:
        # Keep the fixed analytic computation on CPU for Apple execution:
        # MPS lacks AvgPool3d and gave material differences after channel norm.
        # Copies remain differentiable; every learned computation stays on MPS.
        if images.device.type == "mps":
            self.motion_frontend.cpu()
            return self.motion_frontend(images.cpu()).to(images.device)
        return self.motion_frontend(images)

    def encode_fused(self, previous: Tensor, current: Tensor, motion: Tensor) -> Tensor:
        appearance = torch.cat((previous, current), dim=1) - 0.5
        fused = torch.cat((appearance, motion), dim=1)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            field = checkpoint(self.encoder, fused, use_reentrant=False)
        else:
            field = self.encoder(fused)
        positions = self.row_embedding[:, None] + self.column_embedding[None, :]
        return self.token_norm((field.permute(0, 2, 3, 1) + positions).reshape(current.shape[0], 169, 256))

    def forward(self, images: Tensor, task: str = "krauzlis_cued_motion") -> Tensor:
        if task not in ("krauzlis_cued_motion", "krauzlis"):
            raise ValueError("StructuredMotionRViT is the native Krauzlis-only candidate")
        self._validate(images, movie=True)
        motion = self.motion_features(images)
        previous = images[:, 0]
        memory = None
        for step in range(images.shape[1]):
            current = images[:, step]
            memory = self.update_memory(self.encode_fused(previous, current, motion[:, step]), memory)
            previous = current
        return self.decode(memory)

    def stream_step(self, current: Tensor, previous_frame: Tensor | None = None,
                    memory: Tensor | None = None, motion_history: Tensor | None = None
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self._validate(current, movie=False)
        previous = current if previous_frame is None else previous_frame
        if current.device.type == "mps":
            self.motion_frontend.cpu()
            history = None if motion_history is None else motion_history.cpu()
            motion, next_history = self.motion_frontend.stream_step(current.cpu(), history)
            motion, next_history = motion.to(current.device), next_history.to(current.device)
        else:
            motion, next_history = self.motion_frontend.stream_step(current, motion_history)
        updated = self.update_memory(self.encode_fused(previous, current, motion), memory)
        return self.decode(updated), current, updated, next_history
