"""Frozen predictive representation and a plain concatenation classifier."""
import torch
from torch import nn
from SecondPass.VariationalMotionPredictor.model import VariationalMotionPredictor


class FrozenMotionEncoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        predictor = VariationalMotionPredictor(checkpoint_encoder=False)
        predictor.load_state_dict(saved['model'])
        self.source_step = saved['state']['step']
        self.encoder = predictor.encoder
        self.vector_features = predictor.vector_features
        self.mu_head = predictor.mu_head
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, frames):
        field = self.encoder(frames.reshape(-1, 9, 100, 100))
        return self.mu_head(self.vector_features(field))


class ChangeFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256),
                                    nn.GELU(), nn.Linear(256, 64), nn.GELU(),
                                    nn.Linear(64, 2))

    def forward(self, before, after):
        return self.layers(torch.cat((before, after), dim=-1))
