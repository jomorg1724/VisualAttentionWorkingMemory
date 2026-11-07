"""Fresh three-frame CNN and an angle-graded contrastive spring objective."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from SecondPass.ThreeFrameConvVAE.model import TripletEncoder


class MotionEncoder(nn.Module):
    def __init__(self, checkpoint_encoder=True):
        super().__init__()
        self.encoder = TripletEncoder()
        self.projection = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128))
        self.checkpoint_encoder = checkpoint_encoder

    def forward(self, frames):
        if frames.dtype != torch.float32 or tuple(frames.shape[1:]) != (3, 3, 100, 100):
            raise ValueError('Expected FP32 [batch,3 ordered frames,3 RGB,100,100]')
        ordered = frames.reshape(-1, 9, 100, 100)
        field = (checkpoint(self.encoder, ordered, use_reentrant=False)
                 if self.training and torch.is_grad_enabled() and self.checkpoint_encoder
                 else self.encoder(ordered))
        return F.normalize(self.projection(field.mean(dim=(-2, -1))), dim=-1, eps=1e-6)


def angular_difference(first, second):
    """Unsigned shortest difference in radians, including the 0/2pi wrap."""
    return torch.remainder(second - first + math.pi, 2 * math.pi).sub(math.pi).abs()


def distance(first, second):
    # Smooth norm; subtracting sqrt(epsilon) keeps identical vectors at zero.
    return ((first - second).square().sum(-1) + 1e-8).sqrt() - 1e-4


def contrastive_loss(first, second, delta_radians):
    target = delta_radians / math.pi
    observed = distance(first, second)
    return .5 * (observed - target).square().mean()
