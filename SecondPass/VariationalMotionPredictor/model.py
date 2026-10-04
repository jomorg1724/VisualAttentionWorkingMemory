"""Past-three-frame variational CNN with one vector latent predicting frame four."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from SecondPass.ThreeFrameConvVAE.model import TripletEncoder
from SecondPass.TwoFrameRViT.model import ResidualBlock


class FutureFrameDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.expand = nn.Sequential(nn.Linear(latent_dim, 32 * 13 * 13), nn.GELU(),
                                     nn.Unflatten(1, (32, 13, 13)),
                                     nn.Conv2d(32, 256, 1), nn.GroupNorm(8, 256), nn.GELU())
        self.latent_layers = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.up25 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.PixelShuffle(2),
                                  nn.GroupNorm(8, 128), nn.GELU())
        self.layers25 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.up50 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.PixelShuffle(2),
                                  nn.GroupNorm(8, 64), nn.GELU(),
                                  ResidualBlock(64), ResidualBlock(64))
        self.up100 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.PixelShuffle(2),
                                   nn.GroupNorm(8, 32), nn.GELU(),
                                   ResidualBlock(32), ResidualBlock(32))
        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, latent: Tensor) -> Tensor:
        field = self.up25(self.latent_layers(self.expand(latent)))[:, :, :25, :25]
        return self.output(self.up100(self.up50(self.layers25(field)))).sigmoid()


class VariationalMotionPredictor(nn.Module):
    def __init__(self, latent_dim: int = 512, checkpoint_encoder: bool = True,
                 checkpoint_decoder: bool = True):
        super().__init__()
        if latent_dim < 1:
            raise ValueError('latent_dim must be positive')
        self.latent_dim = latent_dim
        self.checkpoint_encoder = checkpoint_encoder
        self.checkpoint_decoder = checkpoint_decoder
        self.encoder = TripletEncoder()
        self.vector_features = nn.Sequential(nn.Conv2d(256, 32, 1), nn.GELU(), nn.Flatten(),
                                            nn.Linear(32 * 13 * 13, 512), nn.LayerNorm(512), nn.GELU())
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -4.)
        self.decoder = FutureFrameDecoder(latent_dim)

    @staticmethod
    def _validate(past: Tensor) -> None:
        if past.ndim != 5 or tuple(past.shape[1:]) != (3, 3, 100, 100):
            raise ValueError('expected ordered [B,3 frames,3 RGB,100,100] observations')
        if past.dtype != torch.float32 or torch.is_autocast_enabled(past.device.type):
            raise ValueError('VariationalMotionPredictor requires FP32 and disabled autocast')

    def posterior(self, past: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(past)
        ordered = past.reshape(past.shape[0], 9, 100, 100)
        if self.checkpoint_encoder and self.training and torch.is_grad_enabled():
            field = checkpoint(self.encoder, ordered, use_reentrant=False)
        else:
            field = self.encoder(ordered)
        vector = self.vector_features(field)
        return self.mu_head(vector), self.logvar_head(vector).clamp(-8., 4.)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + (.5 * logvar).exp() * torch.randn_like(mu)

    def encode(self, past: Tensor, sample: bool = False) -> Tensor:
        mu, logvar = self.posterior(past)
        return self.reparameterize(mu, logvar) if sample else mu

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError('decoder requires a single [B,latent_dim] vector')
        if latent.dtype != torch.float32 or torch.is_autocast_enabled(latent.device.type):
            raise ValueError('decoder requires FP32 and disabled autocast')
        if self.checkpoint_decoder and self.training and torch.is_grad_enabled():
            return checkpoint(self.decoder, latent, use_reentrant=False)
        return self.decoder(latent)

    def forward(self, past: Tensor, sample: bool | None = None) -> dict[str, Tensor]:
        mu, logvar = self.posterior(past)
        z = self.reparameterize(mu, logvar) if (self.training if sample is None else sample) else mu
        return dict(prediction=self.decode(z), mu=mu, logvar=logvar, z=z)


def losses(prediction: Tensor, target: Tensor, past: Tensor, mu: Tensor, logvar: Tensor,
           beta: float = 1e-4) -> dict[str, Tensor]:
    """Union-support balanced future prediction and mean vector-Gaussian KL.

    Target pixels determine loss weights only. No target enters the encoder or
    decoder. A single available region receives full weight if the other is empty.
    """
    VariationalMotionPredictor._validate(past)
    if target.shape != (past.shape[0], 3, 100, 100) or prediction.shape != target.shape:
        raise ValueError('target/prediction must be native [B,3,100,100] fourth frames')
    if target.dtype != torch.float32 or prediction.dtype != torch.float32 or mu.shape != logvar.shape:
        raise ValueError('prediction/target require FP32 and posterior shapes must match')
    if not beta >= 0:
        raise ValueError('beta must be nonnegative')
    past_support = (past - .5).abs().amax(dim=2).amax(dim=1) > (1. / 255.)
    support = past_support | ((target - .5).abs().amax(dim=1) > (1. / 255.))
    error = (prediction - target).square().mean(dim=1)
    n_support, n_background = support.sum(dim=(-2, -1)), (~support).sum(dim=(-2, -1))
    support_error = (error * support).sum(dim=(-2, -1)) / n_support.clamp_min(1)
    background_error = (error * ~support).sum(dim=(-2, -1)) / n_background.clamp_min(1)
    available = (n_support > 0).to(error.dtype) + (n_background > 0).to(error.dtype)
    recon = ((support_error + background_error) / available).mean()
    kl = .5 * (mu.square() + logvar.exp() - 1. - logvar).mean()
    changed = (target - past[:, -1]).abs().amax(dim=1) > (1. / 255.)
    n_changed = changed.sum(dim=(-2, -1))
    change_error = (error * changed).sum(dim=(-2, -1)) / n_changed.clamp_min(1)
    return dict(loss=recon + beta * kl, recon=recon,
                support_mse=support_error.sum() / (n_support > 0).sum().clamp_min(1),
                background_mse=background_error.sum() / (n_background > 0).sum().clamp_min(1),
                full_mse=error.mean(), change_mse=change_error.sum() / (n_changed > 0).sum().clamp_min(1),
                kl=kl)
